# ingestion.py 代码逻辑说明

> 本文档详细解释了 `src/ingestion.py` 的核心逻辑、主要类与方法、典型流程及其在RAG系统中的作用，便于开发者理解和维护。

---

## 1. 文件定位与整体作用

`ingestion.py` 是本项目RAG流程中"索引构建"环节的核心模块，负责将分块后的文本内容分别构建为BM25索引和向量库（faiss），为后续检索和RAG问答提供高效的底层数据结构。

---

## 2. 主要类与结构

### 2.1 BM25Ingestor
- 作用：负责将文本块构建为BM25索引，支持关键词检索。
- 主要方法：
  - `create_bm25_index`：从文本块列表创建BM25索引。
  - `process_reports`：批量处理所有报告，生成并保存BM25索引。

### 2.2 VectorDBIngestor
- 作用：负责将文本块转为向量并建立faiss向量库，支持语义检索。
- 主要方法：
  - `_get_embeddings`：调用OpenAI API获取文本块的嵌入向量。
  - `_create_vector_db`：用faiss构建向量库。
  - `_process_report`：处理单份报告，生成向量库。
  - `process_reports`：批量处理所有报告，生成并保存faiss向量库。

---

## 3. 主要方法与流程说明

### 3.1 BM25索引构建流程
- 读取每份报告的分块内容，分词后构建BM25索引。
- 索引以pkl文件保存，文件名为报告sha1。

### 3.2 向量库构建流程
- 读取每份报告的分块内容，调用OpenAI API获取嵌入向量。
- 用faiss构建向量库，支持高效相似度检索。
- 向量库以faiss文件保存，文件名为报告sha1。

---

## 4. 设计亮点与维护建议
- 支持BM25与向量两种索引，兼容多种检索策略
- 支持批量处理，适合大规模年报场景
- 推荐结合 `text_splitter.py`、`retrieval.py` 理解整体RAG流程 