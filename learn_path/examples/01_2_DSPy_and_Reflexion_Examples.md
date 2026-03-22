# 代码实战合集：01_2_DSPy_and_Reflexion

> 本文件由原课程模块自动提取，专供高级工程师快速查阅和 Copy/Paste 的实操代码合集。

## 核心代码片段 1

```python
prompt = f"你是一个优秀的客服，请回答用户关于{product}的问题：\n问题：{question}\n请注意：语气要礼貌，不要超过100个字。"
```

## 核心代码片段 2

```python
import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot

# 1. 挂载语言模型 (这里以 OpenAI 为例)
turbo = dspy.OpenAI(model='gpt-4o-mini', max_tokens=100)
dspy.settings.configure(lm=turbo)

# 2. 定义 Signature: 输入问题，输出精简答案
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 3 words")

# 3. 编写几个极其简陋的训练样本 (Trainset)
trainset = [
    dspy.Example(question="Who was the first person to walk on the moon?", answer="Neil Armstrong").with_inputs('question'),
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs('question'),
    dspy.Example(question="What is the chemical symbol for water?", answer="H2O").with_inputs('question'),
]

# 4. 定义评估指标 (Metric): 答案不仅要对，还要保证长度小于等于 3 个词
def exact_match_and_short(example, pred, trace=None):
    correct_match = dspy.evaluate.metrics.answer_exact_match(example, pred)
    is_short = len(pred.answer.split()) <= 3
    return correct_match and is_short

# 5. 使用 Teleprompter (优化器) 自动编译和优化
# BootstrapFewShot 会利用强大的模型，基于你提供的弱样本，自动生成更强大的中间推理步骤和 Few-shot 组合
teleprompter = BootstrapFewShot(metric=exact_match_and_short, max_bootstrapped_demos=2)
compiled_qa = teleprompter.compile(dspy.ChainOfThought(BasicQA), trainset=trainset)

# 6. 调用与验证
output = compiled_qa(question="Which planet is known as the Red Planet?")
print(f"优化后大模型的输出: {output.answer}")
```

## 核心代码片段 3

```python
def reflexion_agent(task_prompt, max_retries=3):
    # 初始化对话上下文
    messages = [{"role": "system", "content": "你是一个资深 Python 程序员。"}]
    messages.append({"role": "user", "content": task_prompt})
    
    for attempt in range(max_retries):
        # 1. Generator 生成代码
        code_draft = call_llm(messages)
        print(f"[尝试 {attempt+1}] 生成代码草稿...")
        
        # 2. Critic 评估执行 (或者扔给真实的 Python 解释器执行)
        is_success, error_log = execute_code_in_sandbox(code_draft)
        
        if is_success:
            print("[✅ 测试通过] 输出最终结果")
            return code_draft
        else:
            print(f"[❌ 测试失败] 报错信息: {error_log}")
            # 3. 强制反思环节
            reflection_prompt = f"刚才你写的代码运行失败了，报错为：{error_log}。请反思你代码中哪一行导致了错误，并重新提供修复后的完整代码。"
            # 将反思要求追加到历史对话中，逼迫模型重写
            messages.append({"role": "assistant", "content": code_draft})
            messages.append({"role": "user", "content": reflection_prompt})
            
    print("达到最大重试次数，任务失败。")
    return None

# --- 运行效果演示 ---
# User: 写一个能计算 100 以内所有质数之和的代码。
# [尝试 1] 生成代码草稿...
# [❌ 测试失败] 报错信息: TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'
# [尝试 2] 生成代码草稿... (模型反思: 我在辅助函数里忘了 return True)
# [✅ 测试通过] 输出最终结果
```

