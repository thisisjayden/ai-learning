# 代码实战合集：03_Multi_Agent_and_State_Machines

> 本文件由原课程模块自动提取，专供高级工程师快速查阅和 Copy/Paste 的实操代码合集。

## 核心代码片段 1

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class AgentState(TypedDict):
    task: str
    code: str
    test_result: str
    human_approved: bool

def generate_code_node(state):
    # 调用 LLM 生成代码
    return {"code": "def hello(): pass"}

def run_test_node(state):
    # 执行测试环境
    return {"test_result": "Success"}

def human_review_node(state):
    # 这里通过 LangGraph 的断点机制 (Interrupt) 暂停
    # 真实应用中，等待前端传来用户点击的"Approve"信号
    return {"human_approved": True}

# 构建图
workflow = StateGraph(AgentState)
workflow.add_node("generate", generate_code_node)
workflow.add_node("test", run_test_node)
workflow.add_node("review", human_review_node)

# 定义条件边：测试失败则跳回 generate 重新写，成功则进入 review
def route_after_test(state):
    if state["test_result"] == "Fail": return "generate"
    return "review"

workflow.add_conditional_edges("test", route_after_test)

# 编译成可运行的应用
app = workflow.compile(checkpointer=MemorySaver()) # 支持状态持久化与断点续传
```

