# 结构化输出提示词+链式思考推理——原理与代码实现

> 本文档详细解释了本项目中"结构化输出提示词+链式思考推理"功能的设计思路、核心代码实现及其在RAG流程中的作用。

---

## 1. 功能简介

- **结构化输出**：通过精心设计的系统提示词（Prompt）和严格的Schema约束，引导大模型输出标准化、可解析的JSON结构，便于后续自动处理、评测和引用。
- **链式思考（Chain-of-Thought, CoT）推理**：在回答前，要求模型分步详细推理，提升答案的可解释性和准确性，减少"拍脑袋"式的直接输出。

二者结合，既保证了答案的结构化、可用性，又提升了推理透明度和鲁棒性。

---

## 2. 关键参数与入口

- `prompts.py` 中的 `system_prompt`、`example`、`pydantic_schema`：定义了结构化输出的格式和推理要求
- `step_by_step_analysis` 字段：要求模型输出详细分步推理过程
- `reasoning_summary` 字段：简要总结推理过程
- `final_answer`、`relevant_pages` 等字段：标准化答案和引用
- `api_requests.py` 的 `send_message(..., is_structured=True, response_format=...)`：强制模型输出结构化内容

---

## 3. 代码实现核心片段

以 `src/prompts.py` 为例：

```python
class AnswerWithRAGContextNamePrompt:
    instruction = """
你是一个RAG（检索增强生成）问答系统。
你的任务是仅基于公司年报中RAG检索到的相关页面内容，回答给定问题。

在给出最终答案前，请详细分步思考，尤其关注问题措辞。
- 注意：答案可能与问题表述不同。
- 问题可能是模板生成的，有时对该公司不适用。
"""
    ...
    example = r"""
示例：
问题：
"'南方航空股份有限公司'的CEO是谁？"

答案：
```
{
  "step_by_step_analysis": "1. 问题询问'南方航空股份有限公司'的CEO。CEO通常是公司最高管理者，有时也称总裁或董事总经理。\n2. 信息来源为该公司的年报，将用来确认CEO身份。\n3. 年报中明确指出张三为公司总裁兼首席执行官。\n4. 因此，CEO为张三。",
  "reasoning_summary": "年报明确写明张三为总裁兼CEO，直接回答了问题。",
  "relevant_pages": [58],
  "final_answer": "张三"
}
```
"""
    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)
```

- 通过 instruction 明确要求"分步思考"
- 通过 example 给出标准结构化输出示例
- 通过 pydantic_schema 强制字段类型和顺序

在 `src/api_requests.py` 中：

```python
def send_message(..., is_structured=True, response_format=...):
    ...
    if is_structured:
        params["response_format"] = response_format
        completion = self.llm.beta.chat.completions.parse(**params)
        response = completion.choices[0].message.parsed
        content = response.dict()
    ...
    return content
```

- 通过 is_structured=True 和 response_format，强制大模型输出结构化JSON
- 解析后直接得到标准化字典，便于后续处理

---

## 4. 典型调用链

- 在 `src/questions_processing.py` 的 `QuestionsProcessor.get_answer_for_company` 方法中：

```python
answer_dict = self.openai_processor.get_answer_from_rag_context(
    question=question,
    rag_context=rag_context,
    schema=schema,
    model=self.answering_model
)
```

- 在 `src/api_requests.py` 的 `send_message` 方法中，is_structured=True，response_format=schema

只需配置好Prompt和Schema，即可全流程结构化+链式思考。

---

## 5. 应用场景与总结

- 适用于需要高可解释性、可自动评测、可追溯推理过程的RAG问答场景
- 结构化输出便于自动化处理和多模型对比
- 链式思考提升答案可信度，减少幻觉
- 只需配置Prompt和Schema，无需改动主流程

---

> 如需进一步了解底层实现，建议结合 `prompts.py`、`api_requests.py`、`questions_processing.py` 源码和本项目的整体流程文档一同阅读。 