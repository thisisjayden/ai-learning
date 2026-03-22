# 模块 03：多智能体 (Multi-Agent) 系统与编排

## 1. 从单链 (Chain) 到状态机 (State Machine)
真实业务场景的流程是非线性的，经常需要循环、条件分支、甚至暂停等待人类审批 (Human-in-the-loop)。
- **LangGraph 原理**：将 Agent 的执行过程建模为有向图 (Graph)。节点 (Node) 代表执行逻辑（如 LLM 调用或 Tool 执行），边 (Edge) 代表状态流转。
- **状态管理 (State)**：全局维护一个 State 对象（如当前收集到的信息、执行过的步骤），解决多轮对话和复杂任务中的长程记忆问题。

## 2. Multi-Agent 协作设计模式
- **主管-员工模式 (Supervisor-Worker)**：一个 Manager Agent 负责任务拆解和分发，多个专注特定领域的 Worker Agent 执行，Manager 负责验收。
- **群聊辩论模式**：多个具有不同 System Prompt 的 Agent（如安全专家、架构师、产品经理）对同一个方案进行多轮 review 和辩论，最终达成共识。
- **工具生态 (Tool Calling)**：Agent 不仅能查天气，还能执行本地 Docker 代码、操作数据库、调用企业内部 API。

## 3. 手把手案例：基于 LangGraph 开发带人工审批的代码审查 Agent
**目标**：构建一个循环流，Agent 生成代码 -> 运行测试 -> 如果报错则自动回退修改 -> 如果成功则暂停流转，等待人类（高级工程师）审批确认后才合并。
**伪代码骨架**：
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
