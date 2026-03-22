# 代码实战合集：01_1_Transformer_Core_KVCache_RoPE

> 本文件由原课程模块自动提取，专供高级工程师快速查阅和 Copy/Paste 的实操代码合集。

## 核心代码片段 1

```python
import numpy as np

def simple_autoregressive_generation_with_kv_cache(prompt_tokens, max_new_tokens):
    # 初始化空的 KV Cache 列表
    k_cache = []
    v_cache = []
    
    # 模拟第一次前向传播（Prefill 阶段），处理用户输入的整段 Prompt
    # 这一步计算量大，但只需要做一次
    for token in prompt_tokens:
        q, k, v = compute_qkv(token)
        k_cache.append(k)
        v_cache.append(v)
    
    # 获取第一个输出的 Token
    current_token = compute_attention_and_generate(q, k_cache, v_cache)
    generated_tokens = [current_token]
    
    # 进入自回归解码阶段（Decode 阶段），逐字生成
    for _ in range(max_new_tokens):
        # ⚠️ 关键点：我们只需要为【最新生成的一个词】计算 Q, K, V
        q, k, v = compute_qkv(current_token)
        
        # 将新词的 K, V 追加到显存中的 Cache 里
        k_cache.append(k)
        v_cache.append(v)
        
        # 使用最新的 Q，和【包含了历史所有词】的 KV Cache 计算注意力
        current_token = compute_attention_and_generate(q, k_cache, v_cache)
        generated_tokens.append(current_token)
        
    return generated_tokens


# 注意：真实的计算在 GPU 上是通过张量矩阵并发运算的，这里为了展示原理写成 for 循环
```

## 核心代码片段 2

```python
# 假设我们输入了一句包含 7 个 Token 的 Prompt
prompt = ["我", "今天", "去", "买了", "一", "个", "苹果"]
max_tokens_to_generate = 3

print("--- 开始使用 KV Cache 生成 ---")
# 调用我们刚才写的函数
output = simple_autoregressive_generation_with_kv_cache(prompt, max_tokens_to_generate)
print(f"\n最终生成结果: {prompt} + {output}")
```

