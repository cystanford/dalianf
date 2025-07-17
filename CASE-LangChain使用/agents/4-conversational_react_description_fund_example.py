# 本示例演示 CONVERSATIONAL_REACT_DESCRIPTION 类型 Agent 用法
# ------------------------------------------------------
# 说明：
# - CONVERSATIONAL_REACT_DESCRIPTION 适用于多轮对话场景，Agent 会自动维护对话历史，
#   支持上下文记忆，适合智能客服、连续追问等应用。
# - 与普通 ReAct 类型（如 ZERO_SHOT_REACT_DESCRIPTION）相比，
#   该类型更注重“对话连续性”和“上下文理解”。
# - 推荐用于需要多轮交互、上下文相关的智能问答场景。
# ------------------------------------------------------

import os
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.llms import Tongyi
from langchain.memory import ConversationBufferMemory

# 设置通义千问API密钥
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 初始化LLM
llm = Tongyi(model_name="qwen-turbo", dashscope_api_key=DASHSCOPE_API_KEY)

# 定义一个简单的私募基金信息工具
class FundInfoTool:
    def get_fund_info(self, fund_name: str) -> str:
        info = {
            "阳光私募1号": "阳光私募1号是一只主投A股的私募基金，成立于2018年，年化收益率12%。",
            "量化对冲2号": "量化对冲2号专注于量化策略，风险较低，适合稳健型投资者。"
        }
        return info.get(fund_name, "未找到该私募基金信息。")

fund_tool = Tool(
    name="私募基金信息查询",
    func=FundInfoTool().get_fund_info,
    description="输入私募基金名称，返回基金简介"
)

# 初始化对话记忆
memory = ConversationBufferMemory(memory_key="chat_history")

# 初始化CONVERSATIONAL_REACT_DESCRIPTION类型Agent
agent = initialize_agent(
    [fund_tool],
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

if __name__ == "__main__":
    print("欢迎使用私募基金多轮对话客服（CONVERSATIONAL_REACT_DESCRIPTION示例）")
    while True:
        try:
            question = input("客户：")
            answer = agent.run(question)
            print("AI：", answer)
        except KeyboardInterrupt:
            print("\n已退出。")
            break 