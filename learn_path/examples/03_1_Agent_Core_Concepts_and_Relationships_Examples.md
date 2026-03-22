# 代码实战合集：03_1_Agent_Core_Concepts_and_Relationships

> 本文件由原课程模块自动提取，专供高级工程师快速查阅和 Copy/Paste 的实操代码合集。

## 核心代码片段 1

```python
# 1. 这是一个 Skill (技能) 的定义
def web_search_skill(query: str) -> str:
    """这是一个用于搜索最新互联网信息的工具。参数 query 是搜索关键词。"""
    # ... 发起 requests 调用 DuckDuckGo ...
    return search_results

# 2. 这是一个 Instruction (指令)
user_instruction = "请帮我查一下今天马斯克说了什么，并写成一句话总结。"

# 3. 这是一个 Agent 的配置
agent_system_persona = "你是一个高效的新闻助理。你必须尽全力完成用户指令。"
agent_skills = [web_search_skill]

# -------------------------------------------------------------
# 4. 底层框架的魔法：自动渲染拼接出 Prompt！
# (这部分通常被 LangChain 或 OpenAI SDK 隐藏在了底层底层)
# -------------------------------------------------------------
def compile_prompt(persona, skills, instruction, memory):
    # 动态组装技能说明书
    skills_description = ""
    for skill in skills:
        skills_description += f"工具名称: {skill.__name__}\\n工具用途: {skill.__doc__}\\n\\n"
        
    # 最终发送给大模型的 Prompt 长这样：
    final_prompt = f"""
<System>
{persona}
</System>

<Available_Skills>
你有以下技能可供使用。如果遇到不懂的问题，请务必调用它们：
{skills_description}
</Available_Skills>

<Memory>
{memory}
</Memory>

<User_Instruction>
当前最高优任务：{instruction}
</User_Instruction>

请仔细思考，并决定下一步采取什么行动（调用工具还是直接回答）。
"""
    return final_prompt

# 大模型看到的其实是这个 final_prompt，而不是单纯的一句 user_instruction！
```

