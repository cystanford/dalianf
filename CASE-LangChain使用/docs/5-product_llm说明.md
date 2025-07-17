# 5-product_llm.py 项目说明

## 项目简介

本项目基于 LangChain 框架，结合通义千问（Tongyi）大模型，实现了一个可扩展的智能问答 Agent 系统。用户可以通过自然语言提问，系统会自动调用预设的工具（如产品信息查询、公司介绍等），并以逐步推理的方式给出答案。

## 主要功能
- 支持多轮自然语言问答。
- 通过工具（Tool）扩展知识和能力，目前内置“产品描述查询”和“公司信息查询”。
- 采用 Agent 思路，模拟人类思考过程，逐步推理并调用工具。
- 支持自定义 Prompt 模板和输出解析，便于扩展和定制。

## 核心流程
1. **初始化 LLM**：使用通义千问（Tongyi）大模型作为底层推理引擎。
2. **定义数据源**：通过 `TeslaDataSource` 类，模拟产品数据库和公司介绍。
3. **注册工具**：将数据源方法封装为 Tool，供 Agent 调用。
4. **自定义 Prompt 模板**：通过 `CustomPromptTemplate`，实现多步推理和工具调用的格式化输出。
5. **自定义输出解析**：通过 `CustomOutputParser`，解析 LLM 输出，判断是否需要继续推理或已得出最终答案。
6. **Agent 执行器**：`AgentExecutor` 负责管理 Agent 的推理流程和工具调用。
7. **交互主循环**：持续接收用户输入，输出动态逐字显示的答案。

## 关键类与函数
- `TeslaDataSource`：模拟产品和公司信息的数据源，包含两个主要方法：
  - `find_product_description(product_name)`：根据产品名返回产品描述。
  - `find_company_info(query)`：根据问题返回公司相关信息。
- `CustomPromptTemplate`：自定义 Prompt 模板，支持多步推理和工具描述自动填充。
- `CustomOutputParser`：自定义输出解析器，支持多步推理和最终答案的判断。
- `output_response(response)`：动态逐字输出答案，提升交互体验。

## 使用方法
1. 配置好通义千问 API 密钥（环境变量 `DASHSCOPE_API_KEY`）。
2. 运行 `python 5-product_llm.py`。
3. 按提示输入问题，如“Model 3 有什么特点？”、“特斯拉有哪些自动驾驶功能？”等。
4. 系统会自动推理并输出答案。

## 适用场景
- 智能客服、产品问答机器人原型。
- LangChain Agent 框架学习与二次开发。
- 多工具协同的智能问答系统演示。

---

> 本说明文档由 AI 自动生成，内容如有疑问请参考源码或联系开发者。 