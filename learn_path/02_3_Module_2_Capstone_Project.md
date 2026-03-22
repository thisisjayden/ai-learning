# 模块二综合大练习：企业级财报知识库问答系统 (Hybrid RAG + Rerank + Eval)

> **练习目标**：将本模块学到的“复杂文档多模态解析”、“层级分块（父子块映射）”、“混合检索与重排（Hybrid + Reranker）”以及“RAGAS 自动化评估”融为一体，打造一个准确率挑战 95% 的端到端企业级 RAG 管道。

---

## 1. 业务场景与架构设计挑战

**挑战场景**：
你所在公司的财务部门每个季度都会丢几份长达数百页的 PDF 财报给你，让你做一个“财报小助手”。
- 财报里有大量极其相似的型号（如 `SKU-A92`、`SKU-A93`），向量检索一搜就窜行。我们需要加入 **BM25 混合检索**。
- 检索出来 20 个切块，怎么挑最准的？我们需要接入 **Cross-Encoder 交叉重排序模型**。
- 老板问你：“你做的这个助手，准确率到底是多少？” 我们必须用 **RAGAS 自动化评估框架** 跑出量化分数。

### 1.1 融汇知识点架构图

> **架构流转图：端到端高精度 RAG 漏斗与自动化评估流水线**

```mermaid
graph TD
    subgraph S_GEN_1 [1. 数据预处理与层级分块 (Indexing Pipeline)]
        A["长篇 PDF 财报"] --> B["版面解析 (提取 Markdown)"]
        B --> C["层级分块: 切分出大段落 Parent Node"]
        C --> D["将 Parent 切分为单句 Child Node"]
        D --> E{"建立映射: Child 关联 Parent ID"}
        E --> F["分别将 Child Node 存入 VectorDB("向量库") 和 KeywordDB(倒排库)"]
    end

    subgraph S_GEN_2 [2. 混合召回与精排 (Retrieval & Reranking Pipeline)]
        G["用户提问: Q3 季度 SKU-A92 的毛利率是多少？"] --> H["双路并行召回"]
        H --> I["Dense 检索 (基于语义) 获取 Top 20"]
        H --> J["BM25 检索 (基于关键词匹配 SKU-A92) 获取 Top 20"]
        I --> K{"RRF (倒数排序融合) 合并去重为 Top 20"}
        J --> K
        K --> L["送入 BGE-Reranker 模型进行深度交叉打分"]
        L --> M["截断保留 Top 3 个高分 Child Node"]
        M --> N["根据父子映射，揪出这 3 个 Child 对应的完整 Parent Node!"]
    end

    subgraph S_GEN_3 [3. 生成与自动评估 (Generation & Evaluation)]
        N --> O["将完整的大段落上下文 + 问题 喂给 GPT-4"]
        O --> P["生成最终精准财报答案"]
        P -. "将结果扔进评估队列" .-> Q
        Q["使用 RAGAS 裁判大模型, 评估答案的 Faithfulness("无幻觉") 和 Precision(高精度)"]
    end
```

---

## 2. 核心代码实战：LlamaIndex 高级组件编排

为了实现这个漏斗架构，我们将使用 `LlamaIndex`，它是目前构建企业级 RAG 架构的首选框架（由于它内置对 Parent-Child 解析与重排的极佳支持）。

### 2.1 依赖安装与准备
```bash
pip install llama-index llama-index-retrievers-bm25 llama-index-postprocessor-cohere-rerank ragas datasets
# 如果使用开源本地重排模型代替 Cohere：
# pip install llama-index-postprocessor-flag-embedding-reranker
```

### 2.2 实操代码实现 (Python 伪代码骨架)

```python
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine

# =========================================================
# 知识点 1 & 2: 层级分块与父子节点构建 (Hierarchical Chunking)
# =========================================================
print("1. 正在解析财报并进行层级分块...")
# 假设我们用优秀的解析器把 PDF 转成了 Markdown 放在 ./data 目录下
docs = SimpleDirectoryReader("./data").load_data()

# 配置层级切分器：大块 2048 字，中块 512 字，小块（叶子节点）128 字
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]
)
# 从文档生成所有层级的树状节点
nodes = node_parser.get_nodes_from_documents(docs)
# 我们只提取最小的“叶子节点” (Child Nodes) 用于检索
leaf_nodes = get_leaf_nodes(nodes)

print(f"切分完成：总计生成了 {len(leaf_nodes)} 个最小检索块。")

# =========================================================
# 知识点 3: 混合检索双路召回 (Hybrid Search: Vector + BM25)
# =========================================================
print("2. 正在构建混合检索库 (Dense + Sparse)...")
# 建立内存存储上下文
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes) # 把所有节点（含大块）存进文档库

# 建立向量索引 (Dense Retriever)
vector_index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)
vector_retriever = vector_index.as_retriever(similarity_top_k=20) # 粗排取前 20

# 建立 BM25 倒排索引 (Sparse Retriever)
bm25_retriever = BM25Retriever.from_defaults(nodes=leaf_nodes, similarity_top_k=20)

# 使用 RRF (Reciprocal Rank Fusion) 算法把这两条路融合！
fusion_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    similarity_top_k=20,
    num_queries=1, # 这里不开启多查询展开，只做单纯融合
    mode="reciprocal_rerank" # 核心：RRF 融合机制
)

# =========================================================
# 知识点 4: 引入 Cross-Encoder 交叉重排模型精排
# =========================================================
print("3. 加载 BGE 重排模型...")
# 使用 BAAI 开源的 bge-reranker 模型对前 20 名做地毯式精准打分，只留前 3 名
reranker = SentenceTransformerRerank(
    model="BAAI/bge-reranker-base", 
    top_n=3
)

# =========================================================
# 组装超级引擎：融合检索 -> 重排 -> 父节点扩展 -> LLM生成
# =========================================================
query_engine = RetrieverQueryEngine.from_args(
    retriever=fusion_retriever,
    node_postprocessors=[reranker], # 在喂给大模型前插入重排器
    # 注意：在生产中你还需要加入 AutoMergingRetriever 逻辑去展开父节点，
    # LlamaIndex 会自动基于存储库的 Parent ID 替换上下文。这里为保持逻辑清晰简化了配置。
)

# 发起极度刁钻的提问测试
question = "请严格对比 Q3 季度 SKU-A92 和 SKU-B44 的毛利率下降原因差异。"
print(f"\\n提问：{question}")
response = query_engine.query(question)
print(f"\\n最终回答：\\n{response}")

# =========================================================
# 知识点 5: RAGAS 自动化量化评估打分 (下班前必做)
# =========================================================
print("\\n4. 正在拉起 RAGAS 大模型裁判进行质量评估...")
# 导入我们上节课学到的评测套件
from ragas import evaluate
from ragas.metrics import context_precision, faithfulness
from datasets import Dataset

# 将引擎跑出的结果、提取出的原文档（Contexts）以及标准答案组装成 Dataset
# [具体组装 Dataset 的代码参考精讲 2.2...]
# eval_result = evaluate(dataset, metrics=[context_precision, faithfulness])
# print(eval_result)
```

## 3. 实操交付物验收标准
当你在终端运行这份完整架构的代码时，你需要观察和验收：
1. **纯向量的失败**：如果你把 `fusion_retriever` 换成单一的 `vector_retriever`，它可能会找错 SKU 编号，导致答案胡说八道。
2. **重排的威力**：你可以打印出重排序 (Reranker) 前后的节点文本对比，你会发现那些仅仅含有关键字但语义不相关的废话块，被重排模型无情地降分剔除了。
3. **高分通过评估**：你的 RAGAS `Context Precision` 应该大于 0.85，因为 BM25 + 重排序确保了最准的那个数据块被顶到了 Top 1！

> **模块二综合总结**：纯大模型只是一颗“孤立的大脑”。通过掌握多模态解析、层级块、RRF 混合召回和交叉重排，你已经为它装上了能阅读上万页专业文档的“外挂超级记忆”。这就是从 Demo 玩具走向企业生产级的杀手锏。
