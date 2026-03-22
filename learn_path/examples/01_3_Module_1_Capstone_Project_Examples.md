# 代码实战合集：01_3_Module_1_Capstone_Project

> 本文件由原课程模块自动提取，专供高级工程师快速查阅和 Copy/Paste 的实操代码合集。

## 核心代码片段 1

```bash
pip install dspy-ai
# 如果你打算用真实的本地大模型来白嫖算力，还需要安装 vllm 并在本地拉起推理服务
# pip install vllm
```

## 核心代码片段 2

```python
import dspy
import traceback

# ---------------------------------------------------------
# 知识点 1: 底层机制应用 - 接入带有 vLLM (支持 PagedAttention 和 KV Cache) 的本地模型
# 假设我们在本地 8000 端口启动了一个经过 vLLM 优化的 Qwen/Llama 服务
# vLLM 会在多轮 Reflexion 提问时自动命中并复用之前对话的 KV Cache
# ---------------------------------------------------------
lm = dspy.OpenAI(
    model="Qwen2.5-7B-Instruct", 
    api_base="http://localhost:8000/v1", 
    api_key="EMPTY", 
    max_tokens=500
)
dspy.settings.configure(lm=lm)

# ---------------------------------------------------------
# 知识点 2: DSPy 架构设计 - 告别手写 Prompt
# ---------------------------------------------------------
class CodeGenerationSignature(dspy.Signature):
    """根据用户的需求，生成一段纯净的、可执行的 Python 测试代码。不要包含任何 Markdown 语法解释。"""
    requirement = dspy.InputField(desc="需要生成的测试用例的功能描述")
    error_feedback = dspy.InputField(desc="如果之前运行报错了，这是报错的堆栈信息。如果没有则为空格。")
    code = dspy.OutputField(desc="纯 Python 代码，绝对不能包含 ```python 等多余包裹符号")

# 我们定义一个基础的思维链生成器
coder_module = dspy.ChainOfThought(CodeGenerationSignature)

# (高级扩展：你可以准备一个数据集 trainset，用 dspy.BootstrapFewShot 把 coder_module 编译一下，让它的准确率从 60% 飙到 95%以上，这里略过编译步骤，直接使用原始签名)

# ---------------------------------------------------------
# 知识点 3: Reflexion 反思机制 - 让 Agent 自我审查
# ---------------------------------------------------------
def sandbox_execute(code_str: str):
    """一个极简的代码沙盒：用 exec 执行代码，如果报错抛出异常栈"""
    try:
        # ⚠️ 警告: 真实生产环境中，千万不能用 exec 执行大模型生成的代码！
        # 必须把代码装进 Docker 容器里运行！这里仅为演示逻辑。
        exec_globals = {}
        exec(code_str, exec_globals)
        return True, "Success"
    except Exception as e:
        # 捕捉详细报错栈
        error_msg = traceback.format_exc()
        return False, error_msg

def self_correcting_agent(requirement: str, max_retries: int = 3):
    print(f"\\n🎯 任务开始：{requirement}")
    error_history = "无"
    
    for attempt in range(max_retries):
        print(f"\\n⏳ [第 {attempt + 1} 次尝试] 正在生成代码...")
        
        # 此时向 LLM 发起调用。
        # 如果是第 2、3 次尝试，vLLM 引擎由于识别到前面的 prompt 前缀是完全一样的，
        # 会直接命中 KV Cache，不需要重新计算 requirement 部分的 Attention！
        prediction = coder_module(
            requirement=requirement, 
            error_feedback=error_history
        )
        
        code_draft = prediction.code.strip()
        print("💻 生成的草稿代码:\\n", code_draft)
        
        # 将生成的草稿投入沙盒验证
        print("🛠️ 正在投入沙盒执行...")
        is_success, log = sandbox_execute(code_draft)
        
        if is_success:
            print("✅ 恭喜！代码沙盒执行通过。测试用例逻辑成立。")
            return code_draft
        else:
            print(f"❌ 代码执行失败。报错栈如下:\\n{log}")
            # 将报错信息喂给 error_history，逼迫模型在下一次循环中反思修复
            error_history = f"你上次写的代码:\\n{code_draft}\\n\\n运行报错了:\\n{log}\\n请修正代码中的逻辑错误或语法错误。"

    print("\\n🚨 达到最大重试次数，Agent 放弃挣扎。")
    return None

# =========================================================
# 运行主程序测试
# =========================================================
if __name__ == "__main__":
    task = "写一个快速排序算法，并包含 3 个断言测试用例(assert)。如果遇到空数组必须抛出 ValueError。"
    final_code = self_correcting_agent(task)
```

