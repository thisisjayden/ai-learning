# 代码实战合集：02_3_Module_2_Capstone_Project

> 本文件由原课程模块自动提取，专供高级工程师快速查阅和 Copy/Paste 的实操代码合集。

## 核心代码片段 1

```bash
pip install llama-index llama-index-retrievers-bm25 llama-index-postprocessor-cohere-rerank ragas datasets
# 如果使用开源本地重排模型代替 Cohere：
# pip install llama-index-postprocessor-flag-embedding-reranker
```

## 核心代码片段 2

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

