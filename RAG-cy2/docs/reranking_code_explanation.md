# reranking.py 代码逻辑说明

> 本文档详细解释了 `src/reranking.py` 的核心逻辑、主要类与方法、典型流程及其在RAG系统中的作用，便于开发者理解和维护。

---

## 1. 文件定位与整体作用

`reranking.py` 是本项目RAG流程中"重排"环节的核心模块，负责对初步检索到的文本块进行相关性重排序。支持基于Jina API的多语言重排和基于大模型（如OpenAI GPT-4o）的LLM重排，提升最终检索结果的相关性和准确性。

---

## 2. 主要类与结构

### 2.1 JinaReranker
- 作用：调用Jina官方API进行多语言检索重排。
- 主要方法：
  - `rerank`：传入query和文档列表，返回Jina模型重排后的结果。

### 2.2 LLMReranker
- 作用：基于大模型（如OpenAI GPT-4o）对检索文本块进行相关性重排。
- 主要方法：
  - `get_rank_for_single_block`：对单个文本块进行相关性评分。
  - `get_rank_for_multiple_blocks`：对多个文本块批量评分。
  - `rerank_documents`：多线程并行重排，融合向量分数和LLM分数，输出最终排序。

---

## 3. 主要方法与流程说明

### 3.1 Jina重排流程
- 通过API密钥调用Jina云端重排接口，适合多语言和轻量场景。
- 适合对外部API依赖不敏感的批量重排。

### 3.2 LLM重排流程
- 支持单条和批量重排，自动分批并行处理。
- 先用向量检索召回文本块，LLM对每块打分，融合分数后排序。
- 支持自定义LLM权重，兼顾语义和向量相关性。
- 典型调用链：
  1. `rerank_documents(query, docs)` → 按batch分组 → 多线程并发 → LLM打分 → 融合分数 → 排序输出

---

## 4. 设计亮点与维护建议
- 支持多种重排策略，灵活适配不同RAG实验
- LLM重排支持批量、并发、分权重融合，适合高质量问答场景
- 推荐结合 `retrieval.py`、`pipeline.py` 理解整体RAG流程 