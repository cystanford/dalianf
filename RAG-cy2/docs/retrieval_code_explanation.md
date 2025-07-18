# retrieval.py 代码逻辑说明

> 本文档详细解释了 `src/retrieval.py` 的核心逻辑、主要类与方法、典型流程及其在RAG系统中的作用，便于开发者理解和维护。

---

## 1. 文件定位与整体作用

`retrieval.py` 是本项目RAG流程中"检索"环节的核心模块，负责实现基于BM25、向量、混合（向量+LLM重排）等多种检索方式，为后续RAG问答提供高相关性的上下文文本块。

---

## 2. 主要类与结构

### 2.1 BM25Retriever
- 作用：基于BM25算法的关键词检索器。
- 主要方法：
  - `retrieve_by_company_name`：按公司名和查询语句检索，返回BM25分数最高的文本块。

### 2.2 VectorRetriever
- 作用：基于向量相似度的语义检索器。
- 主要方法：
  - `retrieve_by_company_name`：按公司名和查询语句检索，返回向量距离最近的文本块。
  - `retrieve_all`：返回公司所有文本块。
  - `get_strings_cosine_similarity`：计算两个字符串的语义相似度。

### 2.3 HybridRetriever
- 作用：混合检索器，先用向量召回，再用LLM重排，提升相关性。
- 主要方法：
  - `retrieve_by_company_name`：先用向量检索召回一批文本块，再用LLM重排，最终返回最相关的文本块。

---

## 3. 主要方法与流程说明

### 3.1 BM25检索流程
- 通过分词后的query与BM25索引计算分数，选出topN文本块。
- 可选返回父页面内容。

### 3.2 向量检索流程
- 通过OpenAI API生成query的向量嵌入，与faiss向量库比对，选出topN文本块。
- 支持余弦相似度计算。

### 3.3 混合检索（向量+LLM重排）
- 先用向量检索召回一批文本块（如top28），再用LLM对这些块进行相关性重排，最终返回topN。
- 支持批量重排、权重调节等。

---

## 4. 典型调用链与流程示例

1. **BM25检索**：`BM25Retriever.retrieve_by_company_name`
2. **向量检索**：`VectorRetriever.retrieve_by_company_name`
3. **混合检索**：`HybridRetriever.retrieve_by_company_name`

---

## 5. 设计亮点与维护建议
- 支持多种检索策略，灵活适配不同RAG实验
- 混合检索可大幅提升相关性，适合高质量问答场景
- 推荐结合 `questions_processing.py`、`pipeline.py` 理解整体RAG流程 