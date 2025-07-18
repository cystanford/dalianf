# tables_serialization.py 代码逻辑说明

> 本文档详细解释了 `src/tables_serialization.py` 的核心逻辑、主要类与方法、典型流程及其在RAG系统中的作用，便于开发者理解和维护。

---

## 1. 文件定位与整体作用

`tables_serialization.py` 是本项目RAG流程中"表格结构化"环节的核心模块，负责将PDF解析后的原始表格（HTML）结合上下文，通过大模型（LLM）转化为结构化信息块，便于后续检索和问答。

---

## 2. 主要类与结构

### 2.1 TableSerializer
- 作用：表格序列化主流程类，支持同步/异步LLM表格结构化。
- 主要方法：
  - `_get_table_context`：获取表格所在页的上下文文本（前后各最多3个块）。
  - `_send_serialization_request`：构造LLM请求，拼接上下文和表格HTML，获取结构化结果。
  - `_serialize_table`：序列化单个表格，调用LLM。
  - `serialize_tables`：批量处理报告中所有表格，写入序列化结果。
  - `async_serialize_tables`：异步批量处理，适合大规模并发。

### 2.2 TableSerialization（Schema定义）
- 作用：定义表格序列化的结构化输出Schema，包括核心实体、表头、信息块等。

---

## 3. 主要方法与流程说明

- 支持同步/异步两种表格序列化方式，适配不同规模场景。
- 支持自动提取表格上下文，提升结构化信息准确性。
- 支持多线程/多进程并发，提升处理效率。
- 典型调用链：
  1. `serialize_tables(json_report)` → 逐表格提取上下文 → LLM结构化 → 写入table['serialized']
  2. `async_serialize_tables(json_report)` → 构造批量请求 → 并发LLM结构化 → 写入结果

---

## 4. 设计亮点与维护建议
- 支持表格上下文自动提取，提升信息自洽性
- 支持同步/异步批量处理，适合大规模年报
- 推荐结合 `parsed_reports_merging.py`、`text_splitter.py` 理解整体RAG流程 