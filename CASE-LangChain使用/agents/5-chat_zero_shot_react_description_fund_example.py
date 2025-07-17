# 本示例演示 CHAT_ZERO_SHOT_REACT_DESCRIPTION 类型 Agent 用法
# ------------------------------------------------------
# 说明：
# - CHAT_ZERO_SHOT_REACT_DESCRIPTION 专为 Chat 类大模型（如 ChatGPT、通义千问对话模型等）优化，
#   支持多轮对话和上下文记忆，Prompt 结构更贴近人机对话。
# - ZERO_SHOT_REACT_DESCRIPTION 适用于标准 LLM，偏向单轮问答和结构化推理。
# - 如果你的模型支持多轮对话，推荐用 CHAT_ZERO_SHOT_REACT_DESCRIPTION，体验更好。
# ------------------------------------------------------

import os
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.llms import Tongyi

# 设置通义千问API密钥
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 初始化LLM
llm = Tongyi(model_name="qwen-turbo", dashscope_api_key=DASHSCOPE_API_KEY)

# 定义一个简单的私募基金策略工具
class FundStrategyTool:
    def get_strategy(self, fund_name: str) -> str:
        strategies = {
            "成长精选": "成长精选采用成长股投资策略，重点关注科技和消费行业。",
            "稳健收益": "稳健收益以债券和低波动股票为主，追求稳健回报。"
        }
        return strategies.get(fund_name, "未找到该基金策略信息。")

fund_strategy_tool = Tool(
    name="私募基金策略查询",
    func=FundStrategyTool().get_strategy,
    description="输入私募基金名称，返回基金投资策略"
)

# 初始化CHAT_ZERO_SHOT_REACT_DESCRIPTION类型Agent
agent = initialize_agent(
    [fund_strategy_tool],
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

if __name__ == "__main__":
    print("欢迎使用私募基金策略问答（CHAT_ZERO_SHOT_REACT_DESCRIPTION示例）")
    while True:
        try:
            question = input("请输入您的基金策略相关问题：")
            answer = agent.run(question)
            print("AI：", answer)
        except KeyboardInterrupt:
            print("\n已退出。")
            break 