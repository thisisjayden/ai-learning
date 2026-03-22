# 深度精讲 2.2：高阶 RAG 架构 —— 混合检索、交叉重排与 RAGAS 自动化评估

> **学习目标**：掌握如何结合“语义”与“关键字”进行混合召回，理解 Cross-Encoder 重排模型为什么是精度的定海神针，并学会用 LLM 做裁判进行 RAG 效果自动化评估。

---

## 1. 为什么纯向量检索 (Dense Retrieval) 经常“翻车”？

目前 RAG 最流行的是把文本变成向量（Embedding），算余弦相似度。这叫**密集检索 (Dense Retrieval)**，它擅长“语义匹配”。
- 提问：“苹果公司的手机”。它能搜到包含“iPhone”的段落。

**痛点：专有名词与精确匹配的灾难**
但如果在医疗、金融、工业等领域，包含大量类似“HT-29”、“SKU-10924”的专有名词或产品型号时，向量模型（Embedding）往往没有在这些专有名词上做过专门训练，导致它在多维空间里把这些极度相似的词“糊”在了一起。
- 提问：“查询 SKU-10924 的库存”。它可能会搜出一堆完全不相关的、包含乱七八糟编号的段落。

**破局点：BM25 (稀疏检索 / Sparse Retrieval)**。
BM25 就是传统的倒排索引（类似 Elasticsearch 的分词器匹配）。包含就是包含，不包含就是不包含。它擅长极度精确的“关键字匹配”。

---

## 2. 混合检索与重排漏斗架构 (Hybrid Search & Reranking)

既然向量检索懂“语意”，BM25 懂“关键字”，为什么不把它们结合起来？
这就是大厂 RAG 系统必备的**混合检索加重排序漏斗架构**。

### 2.1 RRF 融合打分机制
由于向量搜索得出的分数（如 0.85）和 BM25 得出的分数（如 15.6）不在一个维度，无法直接相加。工业界采用 **RRF (Reciprocal Rank Fusion)** 算法，只看“排名”，不看“绝对分数”：
$$ \text{RRF\_Score} = \frac{1}{k + \text{Rank}_{\text{dense}}} + \frac{1}{k + \text{Rank}_{\text{sparse}}} $$
（常数 $k$ 一般取 60，谁两边排名都很靠前，融合总分就最高。）

### 2.2 精排护城河：Cross-Encoder 重排序模型
混合检索召回了 20 篇文档，但是把 20 篇全喂给大模型太费钱，也容易让大模型看晕。
这时引入一个专门用于“打分”的小模型（如 BAAI 开源的 `bge-reranker`），它把用户的提问和这段文档“拼接”在一起，经过神经网络深度交互，吐出一个精准的 0~1 的相关度分数。我们只截取前 3 篇喂给大模型。

> **架构图解：召回与精排漏斗设计**

```mermaid
graph TD
    A["用户提问: 查询 SKU-10924 的参数] --> B["并行检索机制]
    
    B --> C["Dense 向量检索 (找语义相关) Top 20]
    B --> D["Sparse BM25 检索 (找精准编号) Top 20]
    
    C --> E{"RRF 算法排名融合"}
    D --> E
    
    E --> F["粗排结果: 合并后的 Top 20 候选文档]
    
    F --> G["传入 Cross-Encoder Reranker 模型进行深度打分]
    
    G --> H["截断提取 Top 3 最精确文档]
    H --> I["作为最终 Context 喂给 LLM 大模型生成回答]
```

---

## 3. RAG 自动化评估与 RAGAS 框架

高级工程师绝不能靠“肉眼看几条回答”来汇报 RAG 的准确率。必须用数据说话。
RAG 评估的核心思想是：**让一个强大的大模型（如 GPT-4）充当裁判（LLM-as-a-Judge），对 RAG 系统的每个环节进行打分。**

目前最主流的框架是 **RAGAS (Retrieval Augmented Generation Assessment)**，它从四个维度量化你的系统能力：

1. **Context Precision (上下文精准度)**：
   - *裁判判定标准*：你检索出来的 3 个段落里，有没有把真正包含答案的段落排在第一位？排得越靠前分数越高。（*评估 Retriever 重排能力*）
2. **Context Recall (上下文召回率)**：
   - *裁判判定标准*：对比标准答案（Ground Truth），你检索出的上下文是否涵盖了回答问题所需的**所有**关键信息？有没有遗漏？（*评估 Retriever 粗排和分块能力*）
3. **Faithfulness (忠实度 / 防幻觉指数)**：
   - *裁判判定标准*：大模型生成的答案里，所有的声明（Claims）是否都能在你提供的上下文里找到原文依据？如果有没依据的话，就是模型在“幻觉乱编”。（*评估 Generator 防幻觉能力*）
4. **Answer Relevance (答案相关性)**：
   - *裁判判定标准*：生成的答案有没有直接回答用户的问题？有没有“答非所问”、“避重就轻”？（*评估 Prompt 设计质量*）

### 3.1 自动化评估伪代码实操

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_precision, context_recall
from datasets import Dataset

# 1. 准备你的 RAG 测试集（至少几十条）
data_samples = {
    # 用户的提问
    "question": ["苹果公司的现任CEO是谁？", "SKU-10924的库存是多少？"],
    # 你的 RAG 系统生成的最终回答
    "answer": ["苹果现任CEO是蒂姆·库克。", "目前SKU-10924缺货。"],
    # 你的 Retriever 搜出来的原文档段落（Contexts）
    "contexts": [
        ["蒂姆·库克于2011年接任苹果CEO。", "乔布斯是苹果创始人。"],
        ["仓库报告显示SKU-10924在昨晚已经售罄。", "SKU-10925还有50件。"]
    ],
    # 业务专家准备的【完美标准答案】(Ground Truth)
    "ground_truth": ["蒂姆·库克", "库存为0，已售罄。"]
}

# 转为 HuggingFace 的 Dataset 格式
dataset = Dataset.from_dict(data_samples)

# 2. 调用 GPT-4 作为裁判，执行一键评估
print("正在召唤大模型裁判进行打分，请稍候...")
score = evaluate(
    dataset,
    metrics=[context_precision, context_recall, faithfulness, answer_relevance],
)

print("\n【RAGAS 自动化评估成绩单】")
print(f"上下文精准度 (Precision): {score['context_precision']:.2f}")
print(f"上下文召回率 (Recall): {score['context_recall']:.2f}")
print(f"防幻觉忠实度 (Faithfulness): {score['faithfulness']:.2f}")
print(f"答案相关性 (Relevance): {score['answer_relevance']:.2f}")
```

**实战复盘指南**：
如果评估报告出炉：
- **Recall 只有 0.4**：说明检索召回能力极差，赶紧回头去优化切块（Chunking）大小，或者加上 BM25 混合检索。
- **Faithfulness 只有 0.5**：说明大模型在疯狂胡说八道。赶紧去修改 System Prompt，严厉警告它“如果上下文里没有，必须回答不知道”。
