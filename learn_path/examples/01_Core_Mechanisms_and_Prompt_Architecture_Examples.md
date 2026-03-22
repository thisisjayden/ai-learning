# 代码实战合集：01_Core_Mechanisms_and_Prompt_Architecture

> 本文件由原课程模块自动提取，专供高级工程师快速查阅和 Copy/Paste 的实操代码合集。

## 核心代码片段 1

```python
import dspy

# 1. 配置 LLM 和检索器
turbo = dspy.OpenAI(model='gpt-4o-mini', max_tokens=1000)
colbertv2 = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(lm=turbo, rm=colbertv2)

# 2. 定义 Signature (输入输出的结构)
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

# 3. 定义 Module (RAG 逻辑)
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

# 高级玩法：使用 BootstrapFewShot 自动编译和优化 Prompt (略)
```

