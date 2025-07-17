# LangChain AgentType 常用类型说明

本文件介绍 `langchain.agents.AgentType` 中常用的 Agent 类型，包含用途、适用场景及简单代码示例，便于开发者选型和快速上手。

---

## 1. ZERO_SHOT_REACT_DESCRIPTION
- **说明**：零样本 ReAct Agent，先推理再行动，自动根据工具描述和用户输入选择工具。
- **适用场景**：通用推理+工具调用，最常用。
- **示例**：
```python
from langchain.agents import initialize_agent, AgentType
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
```

---

## 2. REACT_DOCSTORE
- **说明**：带文档检索能力的 ReAct Agent，能查找和定位文档内容。
- **适用场景**：需要“先查找再定位”文档内容，如百科知识库。
- **示例**：
```python
agent = initialize_agent(tools, llm, agent=AgentType.REACT_DOCSTORE)
```

---

## 3. SELF_ASK_WITH_SEARCH
- **说明**：将复杂问题拆解为简单子问题，并用搜索工具查找答案。
- **适用场景**：事实型问答、需要外部搜索的场景。
- **示例**：
```python
agent = initialize_agent(tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH)
```

---

## 4. CONVERSATIONAL_REACT_DESCRIPTION
- **说明**：多轮对话场景下的 ReAct Agent，支持上下文记忆。
- **适用场景**：智能客服、对话机器人等连续对话。
- **示例**：
```python
agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION)
```

---

## 5. CHAT_ZERO_SHOT_REACT_DESCRIPTION
- **说明**：为 Chat 类模型优化的零样本 ReAct Agent。
- **适用场景**：适合 chat 模型，支持多轮推理和工具调用。
- **示例**：
```python
agent = initialize_agent(tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION)
```

---

## 6. CHAT_CONVERSATIONAL_REACT_DESCRIPTION
- **说明**：结合 Chat 记忆和 ReAct 推理，适合复杂多轮对话。
- **适用场景**：需要记忆上下文的多轮对话。
- **示例**：
```python
agent = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION)
```

---

## 7. STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
- **说明**：结构化多工具对话，支持多输入参数的工具调用。
- **适用场景**：复杂多工具协作、结构化输入输出。
- **示例**：
```python
agent = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION)
```

---

## 8. OPENAI_FUNCTIONS
- **说明**：专为 OpenAI Function Calling 设计，自动调用函数型工具。
- **适用场景**：需要与 OpenAI Function API 集成的场景。
- **示例**：
```python
agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS)
```

---

## 9. OPENAI_MULTI_FUNCTIONS
- **说明**：支持 OpenAI 多函数调用。
- **适用场景**：需要一次调用多个 OpenAI Function 的场景。
- **示例**：
```python
agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS)
```

---

> 本文档由 AI 自动生成，内容如有疑问请参考 LangChain 官方文档或源码。 