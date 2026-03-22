# 模块一综合大练习：构建高能低耗的自我反思智能代码助手

> **练习目标**：将本模块学到的“KV Cache 底层推理机制优化”与“DSPy 架构化 Prompt 编译”以及“Reflexion 自我纠错机制”完全融会贯通，从零手写一个对性能友好且正确率极高的代码生成 Agent。

---

## 1. 业务场景与架构设计挑战

**挑战场景**：
我们需要开发一个专门为公司内部程序员写 Python 测试脚本的 AI 助手。
- 如果我们只发一次请求，大模型经常写出带 Bug 的语法。
- 如果我们让模型“重试”好几次（Reflexion 机制），会导致每次把之前的报错历史全部再喂给它一遍。如果不合理利用 **KV Cache**，长长的报错日志和多轮对话会导致算力成本爆炸。
- 如果我们手写冗长的 System Prompt（比如：“你是一个写测试用例的大师，请深呼吸……”），一旦换了模型，效果就会急剧下降。我们需要用 **DSPy** 将提示词固化成可优化的程序。

### 1.1 融汇知识点架构图

> **架构流转图：带缓存感知与自动优化的反思闭环**

```mermaid
graph TD
    subgraph S_GEN_1
        A["定义测试代码生成的 Signature"] --> B["提供 5 个正确测试用例作为 Trainset"]
        B --> C["Teleprompter 优化器自动生成最佳 Prompt 权重"]
        C --> D("(编译出高质量的代码生成模块 (Compiled Module")))
    end

    subgraph S_GEN_2
        D --> E["用户输入需求：写一个二分查找的测试用例"]
        E --> F["LLM 第 1 次生成代码 (Prefill 全量历史, KV Cache 落盘)"]
        F --> G["沙盒执行代码进行验证"]
        
        G --> H{"代码运行是否报错?"}
        H -- "否" --> I["直接输出通过的代码"]
        
        H -- "是" --> J["获取沙盒报错栈 Traceback"]
        J --> K["拼接：报错信息 + 请求修复指令"]
        K --> L["LLM 第 2 次生成 (此时利用刚才的 KV Cache，只需计算新输入报错信息的 QKV，极大降低延迟)"]
        L -. "回到沙盒重新执行" .-> G
    end
```

---

## 2. 核心代码实战：DSPy + Reflexion + 缓存追踪

在这个实战练习中，我们将使用 Python 构建这个系统。

### 2.1 依赖安装与准备
你需要安装 `dspy-ai` 和一个用于模拟执行沙盒的小工具。
```bash
pip install dspy-ai
# 如果你打算用真实的本地大模型来白嫖算力，还需要安装 vllm 并在本地拉起推理服务
# pip install vllm
```

### 2.2 实操代码实现 (Python)

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

## 3. 实操交付物验收标准
当你在终端运行这段脚本时，你应该能观察到以下效果（验收你的学习成果）：
1. **第一次运行失败**：大模型第一次可能忘记处理空数组抛出 `ValueError` 的逻辑，沙盒触发 `AssertionError`。
2. **反思触发**：终端输出报错信息，并启动第二次生成。
3. **高速推理（KV Cache 命中）**：如果你后台接着 vLLM 服务，你会发现第二次生成的速度（首字延迟 TTFT）远远快于第一次。因为第一次（Prefill 阶段）模型费劲地计算了整个任务的矩阵，而第二次直接从显存里把这部分提取出来，只计算新加入的错误日志的 QKV！
4. **最终成功**：大模型输出了修复了 Bug 的完整快排代码，测试用例完美通过。

> **模块一综合总结**：至此，我们完成了理论（KV Cache/RoPE）、架构（DSPy）、流程（Reflexion）的物理闭环。恭喜你，你的基础内功已经修炼到了架构师水平！
