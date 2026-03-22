# 代码实战合集：03_3_Module_3_Capstone_Project

> 本文件由原课程模块自动提取，专供高级工程师快速查阅和 Copy/Paste 的实操代码合集。

## 核心代码片段 1

```bash
pip install langgraph langchain langchain-openai langchain-community
# 配置你的 OpenAI 或 vLLM 私有服务密钥
```

## 核心代码片段 2

```python
import operator
from typing import Annotated, Sequence, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain.agents import AgentExecutor, create_openai_tools_agent

# =========================================================
# 知识点 1: 定义全局共享状态 (State)
# =========================================================
# 这里的 messages 列表会不断累加所有 Agent 的发言和工具调用结果
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str  # 记录 Supervisor 决定的下一个上场节点名


# =========================================================
# 知识点 2: 定义 Skills (工具库) 与 Function Calling 机制
# 这里的 @tool 装饰器底层做的事情：自动把 Python 函数的类型注解（如 `query: str`）和 docstring 
# 转换成了 OpenAI 的 JSON Schema！然后附带在 Prompt 里发送。大模型在返回时，
# 会停止输出自然语言，而是返回一个包含 `name: "web_search"` 的 JSON 格式结构。
# 
# 进阶思考 (MCP 协议化)：
# 如果你使用了 MCP 架构，那么下面这两个函数就不需要写在 Agent 的代码里了。
# 你可以单独起一个 MCP Server，让 Agent 客户端直接通过 STDIO 或 SSE 连接，获取这两个能力！
# =========================================================

from langchain_core.tools import tool

@tool("web_search")
def web_search_skill(query: str) -> str:
    """用于在互联网上搜索最新资料的技能。"""
    # ... 发起搜索请求 ...
    return "HackerNews API 的顶级接口是 https://hacker-news.firebaseio.com/v0/topstories.json"

@tool("write_file")
def write_file_skill(filename: str, code: str) -> str:
    """用于将代码写入本地硬盘的技能。"""
    with open(filename, "w") as f:
        f.write(code)
    return f"代码已成功写入 {filename}"

# =========================================================
# 知识点 3: 构建各个 Agent 实体 (大脑 + 人设 Prompt + Skills)
# =========================================================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 创建调查员 Agent
researcher_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个资深技术调查员。请使用 web_search 技能查阅开发所需的 API 资料。不要自己写代码。"),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
researcher_agent = create_openai_tools_agent(llm, [web_search_skill], researcher_prompt)
researcher_executor = AgentExecutor(agent=researcher_agent, tools=[web_search_skill])

# 创建程序员 Agent
coder_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个高级 Python 开发。根据上下文里的资料写代码，并使用 write_file 技能将代码保存。"),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
coder_agent = create_openai_tools_agent(llm, [write_file_skill], coder_prompt)
coder_executor = AgentExecutor(agent=coder_agent, tools=[write_file_skill])

# =========================================================
# 知识点 4: 构建 Supervisor (主管路由器)
# =========================================================
members = ["Researcher", "Coder"]
system_prompt = (
    "你是一个工作流调度主管。以下工人归你管辖：{members}。"
    "仔细阅读当前的对话历史，并决定下一步让谁干活。当所有任务完成时，输出 FINISH。"
)
# Supervisor 只需要一个选择题 Prompt，不需要具体的 Skills
options = ["FINISH"] + members
# 构建一个让 LLM 从 options 里做选择的专用函数 (此处省略具体的结构化输出解析逻辑)

def supervisor_node(state: AgentState):
    print("\\n👨‍💼 Supervisor: 正在视察工作进度，决定下一步派活...")
    # LLM 分析 state["messages"]，返回下一个上场的人，比如 "Researcher"
    # return {"next": decision} 
    pass 

# =========================================================
# 知识点 5: 用 LangGraph 将团队编排成有向图
# =========================================================
def researcher_node(state: AgentState):
    print("\\n🕵️‍♂️ Researcher: 接到任务，开始搜索资料...")
    result = researcher_executor.invoke({"messages": state["messages"]})
    # 把自己的工作汇报追加到消息队列里
    return {"messages": [HumanMessage(content=result["output"], name="Researcher")]}

def coder_node(state: AgentState):
    print("\\n👨‍💻 Coder: 拿到资料，开始疯狂敲代码并写入文件...")
    result = coder_executor.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=result["output"], name="Coder")]}

# 初始化图结构
workflow = StateGraph(AgentState)

# 录入节点
workflow.add_node("Researcher", researcher_node)
workflow.add_node("Coder", coder_node)
workflow.add_node("supervisor", supervisor_node)

# 定义所有的工人在干完活之后，都必须向主管汇报 (流转回 Supervisor)
workflow.add_edge("Researcher", "supervisor")
workflow.add_edge("Coder", "supervisor")

# 定义条件边：主管说让谁上，流程就指向谁
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {
        "Researcher": "Researcher",
        "Coder": "Coder",
        "FINISH": END
    }
)

# 主管永远是第一个上场的 (接活)
workflow.set_entry_point("supervisor")

# 编译应用
app = workflow.compile()

# =========================================================
# 运行主程序测试
# =========================================================
if __name__ == "__main__":
    initial_instruction = "写一个抓取 HackerNews 头条新闻的 Python 脚本，保存为 hn_scraper.py"
    print(f"🚀 老板派发任务: {initial_instruction}")
    
    # 将老板的初始 Instruction 塞入起始 State，启动整个虚拟公司流转
    app.invoke({
        "messages": [HumanMessage(content=initial_instruction)],
        "next": "supervisor"
    })
```

