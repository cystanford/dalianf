# prompts.py 代码逻辑说明

> 本文档详细解释了 `src/prompts.py` 的核心逻辑、主要Prompt类与方法、典型字段及其在RAG系统中的作用，便于开发者理解和维护。

---

## 1. 文件整体作用

`prompts.py` 主要用于集中定义本项目所有与大模型交互的系统提示词（Prompt）、结构化输出Schema和示例，涵盖问答、重排、表格序列化、比较等多种场景。通过标准化Prompt和Schema，保证LLM输出的规范性、可解析性和高质量。

---

## 2. 主要Prompt类与功能说明

### 2.1 build_system_prompt 辅助函数
- 作用：拼接系统提示词、示例和Schema，生成完整的system_prompt字符串，便于后续统一调用。
- 用法：各Prompt类通过该函数自动生成带Schema的完整提示词。

### 2.2 RephrasedQuestionsPrompt
- 功能：用于将比较类问题自动拆解为针对每个公司独立的具体问题。
- 典型字段：instruction（任务说明）、example（输入输出示例）、pydantic_schema（结构化输出Schema）
- 应用场景：多公司对比类问题的预处理。

### 2.3 AnswerWithRAGContextSharedPrompt
- 功能：RAG主流程的通用系统提示词，要求模型仅基于检索到的上下文分步推理并作答。
- 典型字段：instruction（任务说明）、user_prompt（用户输入模板）
- 应用场景：所有RAG问答场景的基础Prompt。

### 2.4 AnswerWithRAGContextNamePrompt
- 功能：用于"姓名/公司/产品名"类问题的结构化问答。
- 典型字段：AnswerSchema（包含step_by_step_analysis、reasoning_summary、relevant_pages、final_answer等字段，强制分步推理和标准化输出）
- 应用场景：如"CEO是谁""新产品名称"等问题。

### 2.5 AnswerWithRAGContextNumberPrompt
- 功能：用于"数值/指标"类问题的结构化问答。
- 典型字段：AnswerSchema（同上，final_answer为数值型或N/A，step_by_step_analysis要求严格指标匹配）
- 应用场景：如"2022年总资产是多少"等问题。

### 2.6 AnswerWithRAGContextBooleanPrompt
- 功能：用于"是/否"类问题的结构化问答。
- 典型字段：AnswerSchema（final_answer为布尔型，要求分步推理）
- 应用场景：如"是否宣布分红政策变更"等问题。

### 2.7 AnswerWithRAGContextNamesPrompt
- 功能：用于"名单/多实体"类问题的结构化问答。
- 典型字段：AnswerSchema（final_answer为字符串列表，step_by_step_analysis要求区分实体类型）
- 应用场景：如"新任高管名单""新产品列表"等。

### 2.8 ComparativeAnswerPrompt
- 功能：用于多公司比较类问题的最终结构化输出。
- 典型字段：AnswerSchema（final_answer为公司名或N/A，step_by_step_analysis要求分步推理）
- 应用场景：如"哪家公司营收更高"等。

### 2.9 RerankingPrompt
- 功能：用于检索重排环节，指导LLM对检索到的文本块与查询的相关性进行评分。
- 典型字段：system_prompt_rerank_single_block、system_prompt_rerank_multiple_blocks（评分说明）、RetrievalRankingSingleBlock/MultipleBlocks（结构化输出Schema）
- 应用场景：向量检索+LLM重排场景。

### 2.10 AnswerSchemaFixPrompt
- 功能：用于修正LLM输出的非标准JSON，辅助结构化解析。
- 典型字段：system_prompt、user_prompt
- 应用场景：当LLM输出格式不规范时的自动修复。

---

## 3. 设计亮点与维护建议
- 所有Prompt均集中管理，便于统一修改和多场景复用
- 每个Prompt均配有instruction、example、pydantic_schema，保证LLM输出结构化、可控
- 支持多种问答类型（实体、数值、布尔、名单、比较等）和检索重排，适配RAG全流程
- 推荐结合 `api_requests.py`、`questions_processing.py` 理解Prompt的实际调用链

---

> 如需进一步了解底层实现，建议结合 `prompts.py` 源码和本项目的整体流程文档一同阅读。 