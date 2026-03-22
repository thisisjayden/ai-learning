# 代码实战合集：02_Enterprise_RAG_Optimization

> 本文件由原课程模块自动提取，专供高级工程师快速查阅和 Copy/Paste 的实操代码合集。

## 核心代码片段 1

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

