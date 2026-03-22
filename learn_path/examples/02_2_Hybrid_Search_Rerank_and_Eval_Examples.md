# 代码实战合集：02_2_Hybrid_Search_Rerank_and_Eval

> 本文件由原课程模块自动提取，专供高级工程师快速查阅和 Copy/Paste 的实操代码合集。

## 核心代码片段 1

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

