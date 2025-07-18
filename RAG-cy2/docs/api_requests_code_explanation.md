# api_requests.py 代码逻辑说明

> 本文档详细解释了 `src/api_requests.py` 的核心逻辑、主要类与方法、典型流程及其在RAG系统中的作用，便于开发者理解和维护。

---

## 1. 文件定位与整体作用

`api_requests.py` 是本项目RAG流程中"与大模型API交互"环节的核心模块，负责统一封装OpenAI、IBM、Gemini等主流大模型的消息发送、结构化输出、重试、计费、批量并发等逻辑，是系统与LLM交互的核心枢纽。

---

## 2. 主要类与结构

### 2.1 BaseOpenaiProcessor
- 作用：封装OpenAI API的消息发送、结构化输出、token统计等。
- 主要方法：
  - `send_message`：支持结构化/非结构化消息发送。
  - `count_tokens`：统计字符串token数。

### 2.2 BaseIBMAPIProcessor
- 作用：封装IBM API的余额查询、模型列表、嵌入、消息发送等。
- 主要方法：
  - `check_balance`：查询API余额。
  - `get_available_models`：获取可用模型列表。
  - `get_embeddings`：获取文本嵌入。
  - `send_message`：支持结构化/非结构化消息发送。

### 2.3 BaseGeminiProcessor
- 作用：封装Google Gemini API的模型管理、消息发送、结构化输出等。
- 主要方法：
  - `list_available_models`：列出可用模型。
  - `send_message`：支持结构化/非结构化消息发送。

### 2.4 APIProcessor
- 作用：统一调度OpenAI、IBM、Gemini三大主流API，屏蔽底层差异。
- 主要方法：
  - `send_message`：根据provider参数自动路由到对应API。
  - `get_answer_from_rag_context`：结合RAG上下文和Schema生成答案。
  - `get_rephrased_questions`：调用LLM重写/拆解问题。

### 2.5 AsyncOpenaiProcessor
- 作用：支持OpenAI API的异步批量请求，适合大规模并发场景。
- 主要方法：
  - `process_structured_ouputs_requests`：异步批量处理结构化请求，支持速率控制、断点续传。

---

## 3. 主要方法与流程说明

- 支持结构化/非结构化输出，自动适配不同API和Schema。
- 支持token计费、余额查询、模型管理、嵌入生成等多种能力。
- 支持批量并发、异步处理，适合大规模RAG推理和嵌入生成。
- 典型调用链：
  1. `APIProcessor.send_message` → 自动路由到OpenAI/IBM/Gemini → 统一返回结构化/非结构化结果
  2. `AsyncOpenaiProcessor.process_structured_ouputs_requests` → 批量异步请求 → 断点续传/速率控制

---

## 4. 设计亮点与维护建议
- 支持多家主流大模型API，接口统一，便于扩展
- 支持结构化输出Schema，适配RAG多场景
- 支持批量、异步、速率控制，适合大规模生产环境
- 推荐结合 `pipeline.py`、`questions_processing.py` 理解整体RAG流程 