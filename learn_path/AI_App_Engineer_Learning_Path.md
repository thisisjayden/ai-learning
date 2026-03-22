# AI 应用工程师（AI App Engineer）核心学习路线

本计划围绕“理论基础、设计思路、实操 Demo、AI 设计模式与落地案例”四个维度展开，旨在帮助你从零构建生产级 AI 应用。

## 第一阶段：核心理论与基础 (Week 1-2)
### 1. 大模型基础认知
- **LLM 原理简述**：Transformer 架构、Attention 机制、Tokenization 过程。
- **能力边界**：涌现能力（Emergent Abilities）、幻觉（Hallucination）、上下文窗口（Context Window）。
### 2. Prompt Engineering 进阶
- **基础范式**：Few-Shot Prompting, Chain of Thought (CoT), Tree of Thoughts (ToT)。
- **高级技巧**：ReAct (Reason + Act), Directional Stimulus Prompting, Prompt Chaining。

## 第二阶段：核心技术栈与架构设计 (Week 3-4)
### 1. 核心中间件与编排框架
- **LangChain & LlamaIndex**：理解其核心组件（LLMs, Prompts, Memory, Indexes, Chains, Agents）。
- **向量数据库 (Vector DB)**：Chroma, Pinecone, Milvus。理解 Embedding 降维与余弦相似度。
### 2. RAG (检索增强生成) 架构深挖
- **基础 RAG**：Document Loader -> Text Splitter -> Embedding -> Vector Store -> Retriever。
- **高级 RAG (Advanced RAG)**：
  - **索引前**：Query Rewrite, Query Routing。
  - **检索中**：Hybrid Search (Dense + Sparse/BM25), Semantic Routing。
  - **检索后**：Reranking (Cohere Rerank), Context Compression。

## 第三阶段：AI 设计模式 (AI Design Patterns) (Week 5)
1. **Copilot 模式 (副驾驶模式)**：人类主导，AI 辅助（如 GitHub Copilot, Notion AI）。强调交互延迟与上下文注入。
2. **Agent 模式 (智能体模式)**：AI 自主规划与执行（如 AutoGPT, BabyAGI）。涉及 Tool Use/Function Calling 和长期记忆。
3. **MoE 模式 (混合专家路由模式)**：在应用层根据请求复杂度，路由到不同的模型（小模型兜底，大模型攻坚），以平衡成本和性能。
4. **Human-in-the-loop (HITL) 模式**：高风险场景下，AI 输出需经人工确认（如自动化审批、医疗诊断建议）。

## 第四阶段：实操 Demo 与练习 (Week 6-7)
1. **Demo 1：个人知识库助手 (基于 RAG)**
   - **目标**：解析本地 PDF 文件，实现精准问答。
   - **技术栈**：LlamaIndex + OpenAI API/本地 Ollama + ChromaDB + Streamlit。
2. **Demo 2：自动化数据分析 Agent**
   - **目标**：输入自然语言，自动生成 SQL/Python 查库并生成可视化图表。
   - **技术栈**：LangChain Pandas Dataframe Agent / SQL Agent。
3. **Demo 3：多模态内容生成流**
   - **目标**：抓取今日新闻，自动生成摘要，并配上 DALL-E 生成的封面图，推送到飞书/钉钉。

## 第五阶段：经典落地案例参考与生产实践 (Week 8)
1. **客服与知识问答 (RAG 落地基准)**
   - **案例**：企业内部 IT/HR 问答机器人。
   - **关键设计**：知识库清洗质量、Fallback 机制（回答不上来转人工）、Token 计费监控。
2. **长文本生成与处理 (工作流落地)**
   - **案例**：AI 辅助公文写作 / 长篇研报总结。
   - **关键设计**：Map-Reduce 分块摘要策略，保持上下文连贯性的 Prompt Chaining。
3. **生产级防护 (AI Security)**
   - 防御 Prompt Injection 与 Jailbreak（如设置 System Prompt 护栏、输入过滤）。

---
*Stay Curious, Keep Building!*
