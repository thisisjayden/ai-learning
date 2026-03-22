# 代码实战合集：04_3_Module_4_Capstone_Project

> 本文件由原课程模块自动提取，专供高级工程师快速查阅和 Copy/Paste 的实操代码合集。

## 核心代码片段 1

```python
import asyncio
import time
from openai import AsyncOpenAI
from langfuse.decorators import observe

# ---------------------------------------------------------
# 1. 配置异步并发的本地大模型客户端
# ---------------------------------------------------------
client = AsyncOpenAI(
    api_key="EMPTY",  # 本地部署不需要 key
    base_url="http://localhost:8000/v1" # 这是你用 vLLM 拉起的私有模型端点
)

# ---------------------------------------------------------
# 2. 定义带追踪探针的单次推理请求
# ---------------------------------------------------------
@observe(name="generate_internal_code") # Langfuse 追踪注解
async def request_local_model(req_id: int):
    prompt = f"请求 ID {req_id}: 请用我们公司内部通信框架，写一段向支付网关发起 POST 的 Python 代码。"
    
    start_time = time.time()
    try:
        # 注意：这里的 model 要填你挂载的 LoRA 名称，大模型就会用微调后的脑子回答
        response = await client.chat.completions.create(
            model="internal_api_lora", 
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1
        )
        latency = time.time() - start_time
        tokens = response.usage.completion_tokens
        
        print(f"[Req {req_id}] 完成! 耗时: {latency:.2f}s | 吐字: {tokens} | 速度: {tokens/latency:.1f} token/s")
        return tokens
    except Exception as e:
        print(f"[Req {req_id}] 崩溃: {e}")
        return 0

# ---------------------------------------------------------
# 3. 压测主控：一瞬间扔进去 50 个并发请求，看 vLLM 的表现！
# ---------------------------------------------------------
async def main():
    print("🚀 开始压力测试：同时发起 50 个大模型请求！(Continuous Batching 发威时刻)")
    start_all = time.time()
    
    # 构建 50 个并发任务
    tasks = [request_local_model(i) for i in range(1, 51)]
    
    # 异步等待所有任务完成
    results = await asyncio.gather(*tasks)
    
    total_time = time.time() - start_all
    total_tokens = sum(results)
    
    print("\\n=================================================")
    print(f"✅ 压测结束！总计处理 50 个并发请求。")
    print(f"⏱️ 全局总耗时：{total_time:.2f} 秒")
    print(f"📊 系统并发吞吐量 (QPS)：{50 / total_time:.2f} 请求/秒")
    print(f"🔥 系统总生成吞吐率：{total_tokens / total_time:.1f} Token/秒")
    print("=================================================")
    print("💡 现在，你可以去 Langfuse 大盘查看所有的链路日志和 Token 开销了！")

if __name__ == "__main__":
    asyncio.run(main())
```

