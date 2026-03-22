# 模块 02：企业级 RAG 系统深度调优

## 1. 突破 RAG 的准确率瓶颈
基础 RAG (Naive RAG) 的准确率通常卡在 80% 左右。高级工程师的价值在于那剩下的 20%。
- **多模态与复杂文档解析**：PDF 中的表格、图表、多栏排版是 RAG 的噩梦。实战使用 Unstructured.io 或 Marker 进行精准的版面还原（Layout Analysis）。
- **层级块划分 (Hierarchical Chunking)**：文档分块不能“一刀切”。建立大块（Parent Chunk）和小块（Child Chunk）的映射关系。检索时命中精准的小块，喂给大模型时带上完整的大块上下文（Auto-Merging Retrieval）。

## 2. 检索架构进阶与重排 (Advanced Retrieval & Reranking)
- **混合检索 (Hybrid Search)**：向量检索 (Dense) 擅长语义，BM25 (Sparse) 擅长精准关键词匹配。实战配置两路召回并通过 RRF (Reciprocal Rank Fusion) 算法融合打分。
- **查询重写与展开 (Query Rewrite & Expansion)**：用户提问往往极其简短或指代不清。在检索前加一层 LLM 路由，使用 HyDE（假设性文档嵌入）生成伪答案再向量化检索。
- **Cross-Encoder 重排**：召回阶段只求“全”，精排阶段求“准”。引入 BGE-Reranker 对 Top-50 的文档片段与 Query 进行深度交互打分，截取 Top-5。

## 3. RAG 自动化评估与 GraphRAG
- **RAGAS 评估体系**：拒绝人工看 Case。使用 RAGAS 框架，通过 LLM-as-a-Judge 评估四个维度：
  - Context Precision (上下文精准度)
  - Context Recall (上下文召回率)
  - Faithfulness (忠实度，防幻觉)
  - Answer Relevance (答案相关性)
- **GraphRAG 引入**：微软开源的知识图谱增强 RAG。解决跨文档全局视角的宏观问题（如：“总结这份百页财报中提到的所有风险因素及其关联性”）。

## 4. 手把手案例：基于 RAGAS 构建评估管道
**核心实操代码**：
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_precision, context_recall
from datasets import Dataset

# 你的 RAG 系统在测试集上跑出的结果
data = {
    "question": ["2026年AI架构师的核心技能是什么？"],
    "answer": ["核心技能包括大模型微调、复杂Agent编排和高并发部署。"],
    "contexts": [["大模型时代，高级工程师需要掌握模型微调(LoRA)、LangGraph多智能体编排以及vLLM推理优化。"]],
    "ground_truth": ["需要掌握微调、多智能体和部署优化。"]
}
dataset = Dataset.from_dict(data)

# 运行自动化评估
result = evaluate(
    dataset,
    metrics=[context_precision, faithfulness, answer_relevance, context_recall],
)
print("自动化评估指标得分：", result)
# 针对低得分的指标针对性调优 (比如 Recall 低就去优化检索器，Faithfulness 低去优化 Prompt)
```
