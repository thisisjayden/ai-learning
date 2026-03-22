# 模块 01：底层机制透析与高阶 Prompt 架构

## 1. 深入 LLM 底层机制 (The Black Box)
高级 AI 工程师必须理解模型为什么会这样输出，才能在遇到性能瓶颈或长文本截断时知道如何调优。
- **KV Cache 原理**：自回归生成中，为了避免重复计算历史 Token 的 Attention，模型会缓存 Key 和 Value。理解 KV Cache 是优化高并发推理（如 PagedAttention）的基础。
- **RoPE (旋转位置编码)**：当前主流开源模型（Llama-3, Qwen）扩大上下文窗口的核心技术。理解 RoPE 外推（Extrapolation）和内插（Interpolation）对长文本处理的意义。
- **解码策略高级调参**：不要只会调 Temperature。理解 Top-P (Nucleus Sampling) 与 Top-K 的截断差异，以及 Repetition Penalty 在防止模型“复读机”时的数学逻辑。

## 2. 编程化与框架化的 Prompt 工程
抛弃手写冗长的 Prompt，转向自动优化。
- **DSPy 框架精讲**：将 Prompt 视为代码编译。通过定义 Signature（输入输出签名）和 Module，让模型自己去优化 Few-Shot 的例子。
- **自我反思机制 (Reflexion & Self-Correction)**：
  - **核心逻辑**：Agent 输出结果 -> Critic（评估者）检查错误 -> Agent 根据 Critic 的反馈重新生成。
  - **落地场景**：代码生成、复杂数学推理等长思考链路。

## 3. 手把手案例：使用 DSPy 构建可自动优化的 QA 系统
**目标**：不写死 Prompt，让 DSPy 自动寻找最佳的 Prompt 和 Few-Shot 样本。
**实操代码片段**：
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
