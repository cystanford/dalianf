# parsed_reports_merging.py 代码逻辑说明

> 本文档详细解释了 `src/parsed_reports_merging.py` 的核心逻辑、主要类与方法、典型流程及其在RAG系统中的作用，便于开发者理解和维护。

---

## 1. 文件定位与整体作用

`parsed_reports_merging.py` 是本项目RAG流程中"年报规整与清洗"环节的核心模块，负责将PDF解析后的原始内容进一步规整为每页结构化文本，清洗特殊符号、合并表格/列表/脚注等，便于后续分块、检索和人工审查。

---

## 2. 主要类与结构

### 2.1 PageTextPreparation
- 作用：年报页面文本规整与清洗主流程类。
- 主要方法：
  - `process_reports`：批量处理目录或路径列表下的报告，输出规整后结构。
  - `process_report`：处理单份报告，输出每页规整文本。
  - `prepare_page_text`：处理单页内容，组装为字符串。
  - `_filter_blocks`：移除页脚、图片等无用块。
  - `_clean_text`：正则清洗特殊符号，统计修正。
  - `_apply_formatting_rules`：合并表格、列表、脚注等连续块，按规则格式化。

---

## 3. 主要方法与流程说明

- 支持批量/单份报告规整，自动输出每页结构化文本。
- 支持表格、列表、脚注等连续块的合并与格式化。
- 支持正则清洗特殊符号、OCR残留、无效命令等。
- 支持修正统计与日志，便于人工审查。
- 典型调用链：
  1. `process_reports` → 逐份读取 → `process_report` → 逐页规整 → `prepare_page_text` → 块过滤/合并/清洗 → 输出规整结构

---

## 4. 设计亮点与维护建议
- 支持多种年报结构的规整，适配复杂PDF解析结果
- 清洗规则灵活，便于扩展和定制
- 推荐结合 `pdf_parsing.py`、`text_splitter.py` 理解整体RAG流程 