# 代码实战合集：03_2_Multi_Agent_and_LangGraph

> 本文件由原课程模块自动提取，专供高级工程师快速查阅和 Copy/Paste 的实操代码合集。

## 核心代码片段 1

```python
from typing import TypedDict, Annotated, List
import operator
from langgraph.graph import StateGraph, END

# ---------------------------------------------------------
# 1. 定义全局的 State (状态黑板)
# 所有 Agent 都能看到这个字典里的内容
# ---------------------------------------------------------
class ProjectState(TypedDict):
    task_instruction: str
    current_code: str
    test_logs: str
    review_comments: List[str]

# ---------------------------------------------------------
# 2. 定义节点 Nodes (每个节点可以是一个复杂的 Agent，或者纯 Python 函数)
# ---------------------------------------------------------
def developer_node(state: ProjectState):
    """Developer Agent: 负责根据任务和历史意见写代码"""
    print("👨‍💻 Developer: 正在拼命写代码...")
    # 这里实际上会调用带有 LLM 和 Coding Skill 的 Agent
    new_code = "def hello(): print('world')"
    return {"current_code": new_code} # 返回的结果会自动更新（合并）到全局 State 里

def tester_node(state: ProjectState):
    """Tester Agent: 负责执行代码并返回日志"""
    print("🕵️‍♂️ Tester: 正在沙盒里跑测试...")
    # 调用执行代码的 Skill
    # 为了演示，我们假装第一次测试失败了
    return {"test_logs": "Error: NameError name 'world' is not defined"}

# ---------------------------------------------------------
# 3. 定义条件边 Conditional Edges (图的路由逻辑)
# ---------------------------------------------------------
def should_continue_routing(state: ProjectState):
    """这是一个纯逻辑函数，判断下一步该去哪"""
    if "Error" in state["test_logs"]:
        print("🚦 路由系统: 测试失败！代码打回给 Developer 重写！")
        return "developer"
    else:
        print("🚦 路由系统: 测试通过！进入等待发布流程。")
        return "end"

# ---------------------------------------------------------
# 4. 把一切编排成图 (Graph)
# ---------------------------------------------------------
workflow = StateGraph(ProjectState)

# 把工人添加进车间
workflow.add_node("developer", developer_node)
workflow.add_node("tester", tester_node)

# 定义流水线的起始点
workflow.set_entry_point("developer")

# 写完代码后，无条件交给测试员
workflow.add_edge("developer", "tester")

# 测试完后，根据条件决定是打回重做，还是结束流程
workflow.add_conditional_edges(
    "tester",
    should_continue_routing,
    {
        "developer": "developer", # 如果路由返回 "developer"，就跳转到 developer 节点
        "end": END               # 如果返回 "end"，就结束
    }
)

# 编译成可执行的程序 (相当于图初始化)
# 这里还可以接入 Checkpointer (如 SQLite) 来支持状态持久化和断点续传
app = workflow.compile()
```

