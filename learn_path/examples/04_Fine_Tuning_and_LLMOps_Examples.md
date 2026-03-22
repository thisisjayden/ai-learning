# 代码实战合集：04_Fine_Tuning_and_LLMOps

> 本文件由原课程模块自动提取，专供高级工程师快速查阅和 Copy/Paste 的实操代码合集。

## 核心代码片段 1

```bash
# 安装 vllm
pip install vllm

# 在服务器上一键启动一个兼容 OpenAI API 格式的高性能服务端点
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 1 \ # 多卡并行
    --max-model-len 8192 \     # 上下文窗口
    --gpu-memory-utilization 0.9
```

