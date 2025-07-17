"""
【AgentType.OPENAI_MULTI_FUNCTIONS 简要说明】
本示例基于 LangChain 内置的 OPENAI_MULTI_FUNCTIONS 类型 Agent。

特点：
- 支持 OpenAI 多函数（multi-functions）调用，LLM 可在一次对话中自动调用多个工具，适合多工具协作场景。
- 输入为自然语言，Agent 能根据用户需求自动选择并组合调用多个函数工具。
- 工具需有明确的功能划分和描述，便于 LLM 理解和调度。
- 适合需要一次性获取多项信息或多步操作的复杂问答。

与 OPENAI_FUNCTIONS 区别：
- OPENAI_FUNCTIONS 只支持单函数调用，OPENAI_MULTI_FUNCTIONS 支持多函数并行或串行调用。
- 更适合多工具协作、批量信息查询等场景。

适用场景：
- 用户一次性查询多个对象、批量操作、复杂业务流程。
- 智能助手、自动化办公、信息整合。

本文件为私募基金多基金信息批量查询的函数式智能问答示例。
"""
import os
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_openai import ChatOpenAI

# 设置OpenAI API密钥
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 初始化OpenAI LLM
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

# 定义多个私募基金信息函数工具
class FundInfoTool:
    def get_info(self, fund_name: str) -> str:
        info = {
            "景林成长": "景林成长专注于新兴产业投资，管理规模30亿。",
            "淡水泉精选": "淡水泉精选以精选个股为主，风格灵活，业绩稳健。"
        }
        return info.get(fund_name, "未找到该私募基金信息。")

tools = [
    Tool(
        name="景林成长信息",
        func=lambda _: FundInfoTool().get_info("景林成长"),
        description="查询景林成长私募基金信息"
    ),
    Tool(
        name="淡水泉精选信息",
        func=lambda _: FundInfoTool().get_info("淡水泉精选"),
        description="查询淡水泉精选私募基金信息"
    )
]

# 初始化OPENAI_MULTI_FUNCTIONS类型Agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_MULTI_FUNCTIONS,
    verbose=True
)

if __name__ == "__main__":
    print("欢迎使用私募基金多基金信息查询（OPENAI_MULTI_FUNCTIONS示例）")
    while True:
        try:
            print("可输入：'请同时介绍景林成长和淡水泉精选' 等问题")
            question = input("请输入您的问题：")
            answer = agent.run(question)
            print("AI：", answer)
        except KeyboardInterrupt:
            print("\n已退出。")
            break 