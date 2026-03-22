# 代码实战合集：04_2_vLLM_Deployment_and_LLMOps

> 本文件由原课程模块自动提取，专供高级工程师快速查阅和 Copy/Paste 的实操代码合集。

## 核心代码片段 1

```bash
# 在终端中运行这行神级命令
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --enable-lora \
    --lora-modules my_custom_lora=./saves/qwen-7b/lora/sft \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --port 8000
```

## 核心代码片段 2

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)

# 注意我们要用挂载了 LoRA 的专用补丁名称
response = client.chat.completions.create(
    model="my_custom_lora", 
    messages=[{"role": "user", "content": "请用我们公司的客服口吻回答退货政策。"}]
)
print(response.choices[0].message.content)
```

## 核心代码片段 3

```python
from langfuse.decorators import observe
from langfuse.openai import openai # 使用 langfuse 封装的 openai 客户端自动追踪

# 使用装饰器，Langfuse 就会记录这个函数的耗时和出入参
@observe()
def complex_agent_pipeline(user_query: str):
    # 第 1 步：检索知识库 (耗时追踪)
    context = retrieve_from_db(user_query)
    
    # 第 2 步：调用大模型 (Token 计费、耗时追踪)
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"基于以下背景回答：{context}"},
            {"role": "user", "content": user_query}
        ]
    )
    return response.choices[0].message.content

# 当函数执行完毕，你打开 Langfuse 面板，就能看到：
# - 整个 Pipeline 总耗时 2.5s，总花费 0.002 美元
# - 检索耗时 0.5s，大模型生成耗时 2.0s，生成了 150 个 Token
```

