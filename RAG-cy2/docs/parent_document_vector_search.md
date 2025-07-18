# 支持父文档检索的向量搜索——原理与代码实现

> 本文档详细解释了本项目中"支持父文档检索的向量搜索"功能的设计思路、核心代码实现及其在RAG流程中的作用。

---

## 1. 功能简介

在传统的向量检索中，通常是以"文本块"为最小检索单元，直接返回与查询最相似的若干文本块内容。但在实际RAG问答场景中，单个文本块可能过短、上下文不完整，导致答案缺乏全局性。为此，本项目支持"父文档检索"模式：

- 检索时，先用向量召回最相关的文本块
- 然后返回这些文本块所在的"父页面"或"父文档"内容，保证上下文完整性

这样可以提升答案的可读性和引用的准确性。

---

## 2. 关键参数与入口

在 `VectorRetriever`、`HybridRetriever` 等检索器的 `retrieve_by_company_name` 方法中，均支持 `return_parent_pages` 参数：

- `return_parent_pages=False`（默认）：返回最相关的文本块内容
- `return_parent_pages=True`：返回这些文本块所在的父页面内容（去重）

---

## 3. 代码实现核心片段

以 `src/retrieval.py` 的 `VectorRetriever` 为例：

```python
class VectorRetriever:
    ...
    def retrieve_by_company_name(self, company_name: str, query: str, ..., return_parent_pages: bool = False) -> List[Dict]:
        ...
        for distance, index in zip(distances[0], indices[0]):
            ...
            chunk = chunks[index]
            parent_page = next(page for page in pages if page["page"] == chunk["page"])
            if return_parent_pages:
                if parent_page["page"] not in seen_pages:
                    seen_pages.add(parent_page["page"])
                    result = {
                        "distance": distance,
                        "page": parent_page["page"],
                        "text": parent_page["text"]
                    }
                    retrieval_results.append(result)
            else:
                result = {
                    "distance": distance,
                    "page": chunk["page"],
                    "text": chunk["text"]
                }
                retrieval_results.append(result)
        return retrieval_results
```

- 检索时，先用向量库召回最相关的文本块（chunk）
- 若 `return_parent_pages=True`，则返回这些文本块所在的父页面内容，并用 `seen_pages` 去重
- 否则，直接返回文本块内容

同理，`BM25Retriever`、`HybridRetriever` 也支持该参数，保证检索接口一致性。

---

## 4. 典型调用链

在 `src/questions_processing.py` 的 `QuestionsProcessor.get_answer_for_company` 方法中：

```python
retrieval_results = retriever.retrieve_by_company_name(
    company_name=company_name,
    query=question,
    ...
    return_parent_pages=self.return_parent_pages
)
```

只需在调用时设置 `parent_document_retrieval=True`，即可启用父文档检索。

---

## 5. 总结与应用场景

- 父文档检索适合需要完整上下文、引用原文页的RAG问答场景
- 只需设置参数即可切换，无需修改主流程代码
- 该设计兼容BM25、向量、混合检索等多种策略

---

> 如需进一步了解底层实现，建议结合 `retrieval.py`、`questions_processing.py` 源码和本项目的整体流程文档一同阅读。 