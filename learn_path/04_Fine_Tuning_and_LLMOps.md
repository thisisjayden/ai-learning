# 模块 04：模型微调、高性能部署与 LLMOps

## 1. 模型微调 (Fine-Tuning) 与数据飞轮
高级应用工程师不仅要“用”模型，还要能“改”模型。
- **SFT (监督微调) 适用场景**：改变模型的回答语气（如客服话术）、教会模型特定的 JSON 输出格式、注入私有领域的行业常识（RAG 解决不了的基础常识）。
- **PEFT 与 LoRA 原理**：不再全参数微调庞大的权重矩阵。LoRA 通过在原有权重旁边旁路添加低秩矩阵进行训练，让 24G 显存的单卡也能微调 8B 级别的模型。
- **构建高质量数据集**：数据质量远大于数据数量。学习如何使用 GPT-4 蒸馏生成指令对（Instruction, Input, Output），构建业务专属的 SFT 数据集。
- **对齐微调 (DPO)**：Direct Preference Optimization，比传统的 RLHF 更轻量。通过给出 (Chosen, Rejected) 答案对，让模型学会“什么是好，什么是坏”。

## 2. 高性能部署架构 (vLLM)
为什么不直接用 HuggingFace 的 `pipeline` 部署线上服务？因为慢且无法承载高并发。
- **PagedAttention 核心机制**：vLLM 的灵魂。把显存里的 KV Cache 像操作系统的虚拟内存分页一样管理起来，消除了显存碎片，极大地提高了 Batch Size 和吞吐量。
- **Continuous Batching (连续批处理)**：不再等最慢的请求生成完才处理下一个请求。在每个 Token 生成的周期动态插入和移出请求。
- **模型量化实战**：GPTQ、AWQ 原理。精度损失换取显存占用减半。

## 3. LLMOps 与全生命周期监控
将大模型应用推向生产环境（Production-Ready）。
- **Tracing (链路追踪)**：当一个复杂 Agent 请求耗时 10 秒，到底慢在哪一步？引入 Langfuse，可视化每一层 LLM 调用、Tool 调用的耗时和 Token 消耗。
- **线上监控与反馈循环**：收集用户点赞/踩（Thumbs up/down）的数据，将其清洗后自动流入上述的微调数据集，形成“使用越多 -> 数据越好 -> 模型越强”的数据飞轮。

## 4. 手把手案例：使用 vLLM 部署高并发推理 API
**环境准备与实操**：
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
*精讲点评*：只需要这行命令，你就在本地拉起了一个吞吐量极高、兼容 OpenAI SDK 的私有化大模型服务。应用层的代码（如 LangChain）无需任何改动，只需把 `base_url` 指向本地端口即可。
