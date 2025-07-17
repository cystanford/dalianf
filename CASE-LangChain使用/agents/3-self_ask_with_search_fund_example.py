# 本示例演示 SELF_ASK_WITH_SEARCH 类型 Agent 用法
# ------------------------------------------------------
# 说明：
# - SELF_ASK_WITH_SEARCH 适用于复杂事实型问题，Agent 会自动将复杂问题拆解为若干子问题，
#   并通过搜索工具（如SerpAPI）查找每个子问题的答案，最终汇总得到完整解答。
# - 工具名称必须为 "Intermediate Answer"，否则会报错。
# - 适合需要外部搜索、事实查证、跨领域知识整合的场景。
# - 与 ReAct 类型不同，SELF_ASK_WITH_SEARCH 更强调“问题分解+搜索整合”。
# ------------------------------------------------------

import os
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.llms import Tongyi
from langchain_community.utilities import SerpAPIWrapper

# 设置通义千问API密钥
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 初始化LLM
llm = Tongyi(model_name="qwen-turbo", dashscope_api_key=DASHSCOPE_API_KEY)

# 构建SerpAPI工具，名字必须为"Intermediate Answer"
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="用于查找事实型问题的答案"
    )
]

# 初始化SELF_ASK_WITH_SEARCH类型Agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.SELF_ASK_WITH_SEARCH,
    verbose=True,
    handle_parsing_errors=True  # 增加容错
)

if __name__ == "__main__":
    print("欢迎使用私募基金行业事实问答（SELF_ASK_WITH_SEARCH示例）")
    while True:
        try:
            question = input("请输入您的私募基金行业事实问题：")
            answer = agent.run(question)
            print("\n【智能答复】：", answer)
        except KeyboardInterrupt:
            print("\n已退出。")
            break 