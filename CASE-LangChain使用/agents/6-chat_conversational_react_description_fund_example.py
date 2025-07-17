# 本示例演示 CHAT_CONVERSATIONAL_REACT_DESCRIPTION 类型 Agent 用法
# ------------------------------------------------------
# 说明：
# - CHAT_CONVERSATIONAL_REACT_DESCRIPTION 专为 Chat 类大模型（如 ChatGPT、通义千问对话模型等）优化，
#   支持多轮对话、上下文记忆和人机交互体验，适合连续追问、智能顾问等场景。
# - 该类型要求 memory 返回 BaseMessage 列表（需 return_messages=True），
#   并建议加 handle_parsing_errors=True 提升健壮性。
# - 与 CONVERSATIONAL_REACT_DESCRIPTION 区别：本类型更适合 Chat 模型，
#   Prompt 结构更贴近人机对话，支持更复杂的上下文。
# ------------------------------------------------------

import os
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.llms import Tongyi
from langchain.memory import ConversationBufferMemory

# 设置通义千问API密钥
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 初始化LLM
llm = Tongyi(model_name="qwen-turbo", dashscope_api_key=DASHSCOPE_API_KEY)

# 定义一个简单的私募基金风险评估工具
class FundRiskTool:
    def get_risk(self, fund_name: str) -> str:
        risks = {
            "创新成长": "创新成长基金风险较高，适合激进型投资者。",
            "稳健配置": "稳健配置基金风险较低，适合保守型投资者。"
        }
        # 容错：去掉“基金”后缀
        key = fund_name.replace("基金", "")
        return risks.get(key, "未找到该基金风险信息。")

fund_risk_tool = Tool(
    name="私募基金风险评估",
    func=FundRiskTool().get_risk,
    description="输入私募基金名称，返回基金风险评估"
)

# 初始化对话记忆，确保返回BaseMessage列表
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 初始化CHAT_CONVERSATIONAL_REACT_DESCRIPTION类型Agent
agent = initialize_agent(
    [fund_risk_tool],
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True  # 增加容错
)

if __name__ == "__main__":
    print("欢迎使用私募基金投资顾问（CHAT_CONVERSATIONAL_REACT_DESCRIPTION示例）")
    while True:
        try:
            question = input("投资者：")
            answer = agent.run(question)
            print("AI顾问：", answer)
        except KeyboardInterrupt:
            print("\n已退出。")
            break 