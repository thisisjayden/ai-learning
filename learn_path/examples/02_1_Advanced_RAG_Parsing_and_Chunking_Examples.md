# 代码实战合集：02_1_Advanced_RAG_Parsing_and_Chunking

> 本文件由原课程模块自动提取，专供高级工程师快速查阅和 Copy/Paste 的实操代码合集。

## 核心代码片段 1

```python
# 假设这是我们的文档数据库和向量数据库
doc_db = {}     # 存 Parent Chunks (ID -> 文本内容)
vector_db = []  # 存 Child Chunks 向量 (向量, Parent_ID)

def index_hierarchical_chunks(document_text):
    # 1. 切出大块 (按段落或章节)
    parent_chunks = split_into_large_blocks(document_text)
    
    for p_idx, parent_text in enumerate(parent_chunks):
        parent_id = f"parent_{p_idx}"
        # 将大块原文存入文档库
        doc_db[parent_id] = parent_text 
        
        # 2. 从当前大块中切出小句子
        child_sentences = split_into_sentences(parent_text)
        
        for sentence in child_sentences:
            # 3. 只给小句子算 Embedding
            sentence_vector = compute_embedding(sentence)
            # 存入向量库，并死死绑定它的“父亲是谁”
            vector_db.append({
                "vector": sentence_vector,
                "text": sentence,
                "parent_id": parent_id
            })
            
def retrieve_and_generate(user_query):
    query_vector = compute_embedding(user_query)
    
    # 1. 向量比对，找到最匹配的 3 个“小句子”
    top_3_children = vector_search(query_vector, vector_db, k=3)
    
    context_to_llm = []
    # 2. 顺藤摸瓜，把这 3 个小句子对应的【老父亲】揪出来
    for child in top_3_children:
        parent_id = child["parent_id"]
        parent_full_text = doc_db[parent_id]
        
        # 为了避免重复，去重后拼接到上下文里
        if parent_full_text not in context_to_llm:
            context_to_llm.append(parent_full_text)
            
    # 3. 将完整的大段落上下文喂给 LLM
    final_answer = call_llm(user_query, context_to_llm)
    return final_answer
```

