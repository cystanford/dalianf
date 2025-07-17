# -*- coding: utf-8 -*-
"""
【AgentType.OPENAI_FUNCTIONS 简要说明】
本示例基于 LangChain 内置的 OPENAI_FUNCTIONS 类型 Agent。

特点：
- 充分利用 OpenAI GPT-3.5/4 的函数调用（function calling）能力，LLM 能自动选择并调用注册的工具函数。
- 输入为自然语言，Agent 会自动将用户意图映射到函数参数。
- 工具函数需有明确的参数签名和描述，便于 LLM 理解和调用。
- 适合需要结构化函数调用、参数自动映射的智能问答和业务流程。

与 ZERO_SHOT_REACT_DESCRIPTION、STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION 区别：
- OPENAI_FUNCTIONS 依赖于 OpenAI 的 function calling 能力，适合函数式、结构化调用。
- 其他类型更偏向于 prompt 工程和工具描述驱动。

适用场景：
- 需要 LLM 自动调用后端函数、API 或多工具协作的场景。
- 业务流程自动化、智能助手、复杂问答。

本文件为私募基金信息查询的函数式智能问答示例。
"""
import os
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_openai import ChatOpenAI

# 设置OpenAI API密钥
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 初始化OpenAI LLM
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

# 定义一个私募基金信息函数工具
class FundInfoTool:
    def get_info(self, fund_name: str) -> str:
        info = {
            "鼎晖成长": "鼎晖成长是一只专注于成长型企业投资的私募基金，管理规模50亿。",
            "高毅价值": "高毅价值以价值投资为主，历史业绩优异，管理团队经验丰富。"
        }
        return info.get(fund_name, "未找到该私募基金信息。")

fund_info_tool = Tool(
    name="私募基金信息查询",
    func=FundInfoTool().get_info,
    description="输入私募基金名称，返回基金简介"
)

# 初始化OPENAI_FUNCTIONS类型Agent
agent = initialize_agent(
    [fund_info_tool],
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

if __name__ == "__main__":
    print("欢迎使用私募基金信息查询（OPENAI_FUNCTIONS示例）")
    while True:
        try:
            question = input("请输入您要查询的私募基金名称：")
            answer = agent.run(question)
            print("AI：", answer)
        except KeyboardInterrupt:
            print("\n已退出。")
            break 