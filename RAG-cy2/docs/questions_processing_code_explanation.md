# questions_processing.py 代码逻辑说明

> 本文档详细解释了 `src/questions_processing.py` 的核心逻辑、主要类与方法、典型流程及其在RAG系统中的作用，便于开发者理解和维护。

---

## 1. 文件定位与整体作用

`questions_processing.py` 是本项目RAG问答主流程的核心模块，负责问题的解析、公司名抽取、检索调用、RAG上下文构建、LLM问答、答案后处理、引用页校验等。支持单公司和多公司比较类问题，串联检索与大模型推理。

---

## 2. 主要类与结构

### 2.1 QuestionsProcessor
- 作用：问答主流程调度类，负责加载问题、调用检索器、构建RAG上下文、调用LLM生成答案、处理多公司比较、答案后处理等。
- 主要参数：
  - `vector_db_dir`/`documents_dir`：向量库和分块文档目录
  - `questions_file_path`：问题文件路径
  - `parent_document_retrieval`/`llm_reranking`/`top_n_retrieval`等：检索与重排相关配置
  - `api_provider`/`answering_model`：大模型API与模型名

---

## 3. 主要方法与流程说明

### 3.1 问题加载与公司名抽取
- `_load_questions`：加载JSON格式的问题文件
- `_extract_companies_from_subset`：从问题文本中匹配公司名，支持多公司比较

### 3.2 检索与RAG上下文构建
- `get_answer_for_company`：
  - 针对单公司问题，调用向量检索器（可选LLM重排），获取相关文本块
  - 格式化为RAG上下文字符串
  - 调用LLM生成结构化答案
  - 校验引用页码，补充引用信息

### 3.3 多公司比较类问题
- `process_comparative_question`：
  - 针对多公司问题，分别对每家公司独立检索与问答
  - 汇总各公司答案，返回结构化结果

### 3.4 问题批量处理与保存
- `process_questions_list`：批量处理问题，支持保存中间结果
- `process_all_questions`：处理所有问题，支持生成提交文件
- `_save_progress`：保存处理进度到文件

### 3.5 答案后处理与统计
- `_validate_page_references`：校验LLM答案中引用的页码是否真实存在于检索结果中
- `_calculate_statistics`：统计N/A数量、平均token等
- `_post_process_submission_answers`：对提交答案做后处理

---

## 4. 典型调用链与流程示例

1. **加载问题**：`_load_questions`
2. **公司名抽取**：`_extract_companies_from_subset`
3. **检索与RAG上下文构建**：`get_answer_for_company`
4. **LLM问答**：`APIProcessor.get_answer_from_rag_context`
5. **答案后处理**：`_validate_page_references`、`_calculate_statistics`
6. **批量处理与保存**：`process_questions_list`、`_save_progress`

---

## 5. 设计亮点与维护建议
- 支持多种检索与重排策略，灵活适配不同RAG实验
- 单公司与多公司问题统一处理，便于扩展
- 结构化输出，便于后续评测与分析
- 推荐结合 `pipeline.py`、`retrieval.py` 理解整体RAG流程 