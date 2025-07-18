# LLM重排序提升上下文相关性——原理与代码实现

> 本文档详细解释了本项目中"LLM重排序提升上下文相关性"功能的设计思路、核心代码实现及其在RAG流程中的作用。

---

## 1. 功能简介

在RAG（检索增强生成）系统中，初步检索（如向量召回）得到的文本块往往只保证了语义相似性，但未必与用户查询最相关。为进一步提升检索结果的上下文相关性，本项目引入了"LLM重排序"机制：

- 首先用向量检索召回一批候选文本块
- 然后用大模型（LLM）对这些文本块与查询的相关性进行评分
- 最终融合向量分数和LLM分数，输出最相关的文本块

这样可以显著提升最终答案的准确性和可解释性。

---

## 2. 关键参数与入口

- `HybridRetriever`：混合检索器，负责先向量召回再LLM重排
- `LLMReranker`：LLM重排核心类，支持单条和批量重排
- `llm_weight`：LLM分数权重（0-1），其余为向量分数权重
- `documents_batch_size`：每批送入LLM的文档数，支持并行

---

## 3. 代码实现核心片段

以 `src/reranking.py` 的 `LLMReranker` 为例：

```python
class LLMReranker:
    def rerank_documents(self, query: str, documents: list, documents_batch_size: int = 10, llm_weight: float = 0.7):
        """
        使用多线程并行方式对多个文档进行重排。
        结合向量相似度和LLM相关性分数，采用加权平均融合。
        参数：
            query: 查询语句
            documents: 待重排的文档列表，每个元素需包含'text'和'distance'
            documents_batch_size: 每批送入LLM的文档数
            llm_weight: LLM分数权重（0-1），其余为向量分数权重
        返回：
            按融合分数降序排序的文档列表
        """
        # 按batch分组
        doc_batches = [documents[i:i + documents_batch_size] for i in range(0, len(documents), documents_batch_size)]
        vector_weight = 1 - llm_weight
        ...
        # LLM打分与融合
        for doc, rank in zip(batch, block_rankings):
            doc_with_score = doc.copy()
            doc_with_score["relevance_score"] = rank["relevance_score"]
            doc_with_score["combined_score"] = round(
                llm_weight * rank["relevance_score"] + 
                vector_weight * doc['distance'],
                4
            )
            results.append(doc_with_score)
        ...
        # 按融合分数降序排序
        all_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return all_results
```

- 先用向量检索召回文本块，得到distance分数
- LLM对每个文本块与query打分，得到relevance_score
- 按 `llm_weight` 融合两个分数，排序输出
- 支持批量、并行处理，效率高

---

## 4. 典型调用链

- 在 `src/retrieval.py` 的 `HybridRetriever.retrieve_by_company_name` 方法中：

```python
vector_results = self.vector_retriever.retrieve_by_company_name(...)
reranked_results = self.reranker.rerank_documents(
    query=query,
    documents=vector_results,
    documents_batch_size=documents_batch_size,
    llm_weight=llm_weight
)
```

- 在 `src/questions_processing.py` 的 `QuestionsProcessor.get_answer_for_company` 方法中：

```python
if self.llm_reranking:
    retriever = HybridRetriever(...)
else:
    retriever = VectorRetriever(...)
retrieval_results = retriever.retrieve_by_company_name(...)
```

只需设置 `llm_reranking=True`，即可启用LLM重排序。

---

## 5. 应用场景与总结

- 适用于高质量问答、需要更强相关性排序的RAG场景
- 可灵活调整LLM权重，兼顾效率与效果
- 支持批量、并发，适合大规模检索
- 只需切换参数即可启用，无需改动主流程

---

> 如需进一步了解底层实现，建议结合 `reranking.py`、`retrieval.py`、`questions_processing.py` 源码和本项目的整体流程文档一同阅读。 