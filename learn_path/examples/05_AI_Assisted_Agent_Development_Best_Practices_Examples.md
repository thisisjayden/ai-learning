# 代码实战合集：05_AI_Assisted_Agent_Development_Best_Practices

> 本文件由原课程模块自动提取，专供高级工程师快速查阅和 Copy/Paste 的实操代码合集。

## 核心代码片段 1

```python
# [用户输入]: 
# 这是一个获取用户订单的工具。
# 要求：
# 1. 加上 @tool 装饰器
# 2. 必须包含 user_id (字符串) 和 limit (整型，默认5) 两个参数的严格类型注解
# 3. 为函数加上极度详尽的、适合给大模型阅读的 Docstring（说明每个参数的用途和异常处理）

# [Copilot 自动补全生成]:
from langchain_core.tools import tool

@tool
def get_user_orders(user_id: str, limit: int = 5) -> str:
    """
    当需要查询、获取或检索指定用户的历史订单记录时调用此工具。
    绝对不要在未提供确切的 user_id 时猜测或调用。
    
    Args:
        user_id (str): 用户的唯一系统标识符，通常以 'usr_' 开头。
        limit (int, optional): 最多返回的订单数量。默认为 5。
        
    Returns:
        str: 返回一个包含订单详情的 JSON 字符串。如果数据库连接失败，将返回明确的错误提示。
    """
    try:
        # (业务逻辑代码)
        return fetch_orders_from_db(user_id, limit)
    except Exception as e:
        return f"Error: 获取订单失败，错误详情为 {str(e)}。请提示用户稍后重试。"
```

## 核心代码片段 2

```python
# [用户输入]:
# 帮我写一个安全的、符合 Anthropic MCP 协议的 SQLite 查询工具。
# 要求：
# 1. 引入 mcp.server 的相关库
# 2. 工具名为 get_user_balance
# 3. 必须包含严格的参数校验（防止 SQL 注入），只允许查询，拒绝一切 DROP/UPDATE 操作。

# [Copilot 自动补全生成]:
from mcp.server.fastmcp import FastMCP
import sqlite3

# 创建无状态的 MCP Server 实例
mcp = FastMCP("Finance_DB_Server")

@mcp.tool()
def get_user_balance(user_id: int) -> str:
    """
    查询指定用户的账户余额。只能用于数字类型的 user_id。
    """
    # 强校验：防止 SQL 注入
    if not isinstance(user_id, int):
        return "Error: user_id 必须是合法的整数"
        
    try:
        # 使用安全的参数化查询，而非字符串拼接
        conn = sqlite3.connect("finance.db")
        cursor = conn.cursor()
        cursor.execute("SELECT balance FROM users WHERE id = ?", (user_id,))
        result = cursor.fetchone()
        
        if result:
            return f"User {user_id} balance is: {result[0]}"
        return f"Error: 未找到用户 {user_id}"
    except Exception as e:
        return f"Error: 数据库查询失败 - {str(e)}"
    finally:
        conn.close()

if __name__ == "__main__":
    # 通过标准输入输出运行 MCP Server，与大模型客户端物理隔离
    mcp.run()
```

