# pipeline.py 代码逻辑说明

> 本文档详细解释了 `src/pipeline.py` 的核心逻辑、主要类与方法、典型流程及其在RAG系统中的作用，便于开发者理解和维护。

---

## 1. 文件定位与整体作用

`pipeline.py` 是本项目的数据处理主流程调度模块，负责串联 PDF 解析、表格序列化、报告规整、文本分块、向量化、检索库构建、问题处理等各个环节。通过不同配置，可以灵活组合和运行各阶段，支撑多种RAG问答与数据分析场景。

---

## 2. 主要类与结构

### 2.1 PipelineConfig
- 作用：统一管理各阶段所需的路径（如原始PDF、解析结果、分块、向量库、BM25库等），根据是否使用表格序列化等参数自动调整目录结构。
- 典型字段：`root_path`、`subset_path`、`questions_file_path`、`pdf_reports_dir`、`vector_db_dir`、`bm25_db_path` 等。

### 2.2 RunConfig
- 作用：定义一次完整流程的运行参数，如是否用表格序列化、是否用向量库、是否用LLM重排、检索topN、并发数、API模型等。
- 便于多种实验配置的切换和复用。

### 2.3 Pipeline
- 作用：主流程调度类，封装了各阶段的调用方法。
- 构造时自动初始化路径、配置，并可自动将 subset.json 转为 subset.csv。
- 典型方法：
  - `parse_pdf_reports_sequential/parallel`：顺序/并行解析PDF为结构化JSON。
  - `serialize_tables`：对报告中的表格进行LLM序列化。
  - `merge_reports`：规整JSON为每页文本结构。
  - `export_reports_to_markdown`：导出为markdown文本。
  - `chunk_reports`：按Token分块，便于向量化。
  - `create_vector_dbs`：生成faiss向量库。
  - `create_bm25_db`：生成BM25检索库。

---

## 3. 主要方法与流程说明

### 3.1 PDF解析
- `parse_pdf_reports_sequential/parallel`
  - 调用 `PDFParser`，将原始PDF批量解析为结构化JSON。
  - 支持顺序和多进程并行两种模式。
  - 解析结果存储于 `debug_data/01_parsed_reports`。

### 3.2 表格序列化（可选）
- `serialize_tables`
  - 调用 `TableSerializer`，对解析出的表格内容结合上下文用LLM序列化，便于后续结构化检索。
  - 支持多线程并发处理。

### 3.3 报告规整
- `merge_reports`
  - 调用 `PageTextPreparation`，将复杂JSON规整为每页纯文本结构，便于分块和检索。
  - 结果存储于 `debug_data/02_merged_reports`。

### 3.4 导出为Markdown
- `export_reports_to_markdown`
  - 将规整后的报告导出为markdown文本，便于人工审查和全文检索。

### 3.5 文本分块
- `chunk_reports`
  - 调用 `TextSplitter`，将规整后的文本按Token数分块，支持表格内容特殊处理。
  - 结果存储于 `databases/chunked_reports`。

### 3.6 检索库构建
- `create_vector_dbs`
  - 调用 `VectorDBIngestor`，将分块文本转为向量并建立faiss向量库。
- `create_bm25_db`
  - 调用 `BM25Ingestor`，将分块文本建立BM25索引。

### 3.7 问题处理与RAG问答
- （见 `process_questions`，实际在 `questions_processing.py` 实现）
- 支持多公司比较、RAG上下文构建、LLM问答、答案后处理等。

---

## 4. 典型调用链与流程示例

1. **PDF解析**：`parse_pdf_reports_sequential/parallel`
2. **（可选）表格序列化**：`serialize_tables`
3. **报告规整**：`merge_reports`
4. **导出markdown**：`export_reports_to_markdown`
5. **文本分块**：`chunk_reports`
6. **向量库/BM25库构建**：`create_vector_dbs` / `create_bm25_db`
7. **问题处理与问答**：`process_questions`

---

## 5. 设计亮点与维护建议
- 路径与配置高度解耦，便于多实验切换。
- 每个阶段均可单独调用，支持断点续跑和灵活调试。
- 适合大规模批量处理和多种RAG实验。
- 推荐结合 `README.md` 和本文件，快速定位和理解主流程。 

### 3.2 表格序列化补充解释
 - **定义与作用**：表格序列化是指将PDF报告中解析出的原始表格（通常为HTML或二维数组形式），结合表格前后的上下文，通过大模型（LLM）转化为一组"自洽、上下文无关"的结构化信息块。每个信息块都包含了表格的核心实体、相关指标、单位、币种、表名、脚注等所有关键信息，便于后续检索和问答。
  - **输入**：
    - 表格的HTML内容
    - 表格所在页的上下文文本（如表名、说明、脚注等）
  - **输出**：
    - 一个结构化对象，包含：
      - `subject_core_entities_list`：所有核心实体（如行头/主体）
      - `relevant_headers_list`：所有相关表头（如列名/指标）
      - `information_blocks`：每个核心实体对应的详细信息块（包含所有关键信息、单位、币种、上下文等）
  - **简化示例**：
    - **输入表格（HTML）及上下文**：
      
      表名：主要财务数据
      
      | 项目         | 2022年 | 2021年 |
      |--------------|--------|--------|
      | 营业收入     | 1000万 | 900万  |
      | 净利润       | 200万  | 150万  |
      
      上下文：单位为人民币万元。
    - **序列化后结构化输出**：
      ```json
      {
        "subject_core_entities_list": ["营业收入", "净利润"],
        "relevant_headers_list": ["2022年", "2021年"],
        "information_blocks": [
          {
            "subject_core_entity": "营业收入",
            "information_block": "营业收入，2022年为1000万元，2021年为900万元，单位为人民币万元，表名为主要财务数据。"
          },
          {
            "subject_core_entity": "净利润",
            "information_block": "净利润，2022年为200万元，2021年为150万元，单位为人民币万元，表名为主要财务数据。"
          }
        ]
      }
      ```
    - 这样每个信息块都能独立理解和检索，无需依赖原始表格或上下文。