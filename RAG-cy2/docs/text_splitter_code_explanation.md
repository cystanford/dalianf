# text_splitter.py 代码逻辑说明

> 本文档详细解释了 `src/text_splitter.py` 的核心逻辑、主要类与方法、典型流程及其在RAG系统中的作用，便于开发者理解和维护。

---

## 1. 文件定位与整体作用

`text_splitter.py` 是本项目RAG流程中"文本分块"环节的核心模块，负责将PDF解析后的每页文本和表格内容进行分块，便于后续向量化、检索和RAG问答。

---

## 2. 主要类与结构

### 2.1 TextSplitter
- 作用：文本分块主流程类，支持按页分块、表格插入、token统计等。
- 主要方法：
  - `_get_serialized_tables_by_page`：按页分组已序列化表格，便于插入到对应页面分块中。
  - `_split_report`：将单份报告按页分块，支持插入序列化表格块。
  - `count_tokens`：统计文本的token数，支持自定义编码。
  - `_split_page`：将单页文本分块，保留原始markdown表格。
  - `split_all_reports`：批量处理目录下所有报告，分块并输出到目标目录。

### 2.2 TextSplitter 各方法详细说明

- `_get_serialized_tables_by_page(tables: List[Dict]) -> Dict[int, List[Dict]]`
  - 作用：将输入的表格列表按页码分组，提取每个表格的关键信息（如表格文本、表格ID、token数），便于后续插入到对应页面的分块中。
  - 参数：tables——包含表格序列化信息的字典列表。
  - 返回：以页码为key、表格信息列表为value的字典。

- `_split_report(file_content: Dict[str, any], serialized_tables_report_path: Optional[Path] = None) -> Dict[str, any]`
  - 作用：将单份报告按页分块，支持插入序列化表格块。每页先进行文本分块，再可选插入表格分块，最终汇总所有分块。
  - 参数：file_content——报告内容字典；serialized_tables_report_path——可选，表格序列化文件路径。
  - 返回：包含分块信息的报告内容字典。

- `count_tokens(string: str, encoding_name="o200k_base") -> int`
  - 作用：统计输入字符串的token数，支持自定义编码（如适配不同大模型）。
  - 参数：string——待统计的文本；encoding_name——编码名称，默认o200k_base。
  - 返回：token数量（整数）。

- `_split_page(page: Dict[str, any], chunk_size: int = 300, chunk_overlap: int = 50) -> List[Dict[str, any]]`
  - 作用：对单页文本进行分块，支持自定义分块大小和重叠量，分块时保留原始markdown表格。
  - 参数：page——单页内容字典；chunk_size——每块最大token数；chunk_overlap——分块重叠token数。
  - 返回：包含分块内容、页码、token数的字典列表。

- `split_all_reports(all_report_dir: Path, output_dir: Path, serialized_tables_dir: Optional[Path] = None)`
  - 作用：批量处理目录下所有报告（json文件），对每个报告进行文本分块，并输出到目标目录。可选插入表格分块。
  - 参数：all_report_dir——待处理报告目录；output_dir——输出目录；serialized_tables_dir——可选，表格序列化目录。
  - 返回：无，直接输出分块结果到文件。

- `split_markdown_file(md_path: Path, chunk_size: int = 30, chunk_overlap: int = 5) -> List[Dict]`
  - 作用：按行分割markdown文件，每个分块记录起止行号和内容，适合处理结构化较强的md文本。
  - 参数：md_path——markdown文件路径；chunk_size——每块最大行数；chunk_overlap——分块重叠行数。
  - 返回：分块内容列表。

- `split_markdown_reports(all_md_dir: Path, output_dir: Path, chunk_size: int = 30, chunk_overlap: int = 5, subset_csv: Path = None)`
  - 作用：批量处理目录下所有markdown文件，分块并输出为json文件到目标目录。支持通过subset.csv补充公司名、sha1等元信息。
  - 参数：all_md_dir——md文件目录；output_dir——输出目录；chunk_size——每块最大行数；chunk_overlap——分块重叠行数；subset_csv——可选，文件名到公司名映射表。
  - 返回：无，直接输出分块结果到文件。

---

## 3. 主要方法与流程说明

### 3.1 分块流程
- 支持普通文本和表格内容的分块，表格可选插入到对应页面。
- 分块时可自定义chunk大小和重叠量，适配不同模型token限制。
- 支持统计每个分块的token数，便于后续向量化和检索。

### 3.2 批量处理
- `split_all_reports` 支持批量处理整个目录下的所有报告，自动输出分块结果。
- 支持与表格序列化结果联动，提升结构化检索能力。

---

## 4. 设计亮点与维护建议
- 支持表格与正文混合分块，适配多种RAG场景
- 分块参数灵活，兼容不同大模型token窗口
- 推荐结合 `ingestion.py`、`retrieval.py` 理解整体RAG流程 