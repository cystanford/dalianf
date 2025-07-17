# 本示例演示 REACT_DOCSTORE 类型 Agent 用法
# ------------------------------------------------------
# 说明：
# - REACT_DOCSTORE 适用于需要“先检索文档，再定位答案”的场景，
#   Agent 会先用 Search 工具查找相关文档，再用 Lookup 工具在文档中定位具体内容。
# - 工具列表必须包含两个工具，且名称分别为 "Search" 和 "Lookup"，否则会报错。
# - 适合大文档、知识库、百科问答等场景，能有效提升检索准确率和推理可解释性。
# - 与普通 ReAct 类型相比，REACT_DOCSTORE 更强调“分步检索+定位”，适合结构化知识库。
# ------------------------------------------------------

import os
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.llms import Tongyi
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# 设置通义千问API密钥
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 初始化LLM
llm = Tongyi(model_name="qwen-turbo", dashscope_api_key=DASHSCOPE_API_KEY)

# 构建Wikipedia工具
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(lang="zh"))

# 必须有两个工具，分别命名为"Search"和"Lookup"
search_tool = Tool(
    name="Search",
    func=wikipedia.run,
    description="用于查找私募基金相关百科条目"
)
lookup_tool = Tool(
    name="Lookup",
    func=wikipedia.run,  # 这里可自定义为更细粒度的查找
    description="用于在百科条目中定位具体内容"
)

# 工具列表必须为两个
fund_tools = [search_tool, lookup_tool]

# 初始化REACT_DOCSTORE类型Agent
agent = initialize_agent(
    fund_tools,
    llm,
    agent=AgentType.REACT_DOCSTORE,
    verbose=True
)

if __name__ == "__main__":
    print("欢迎使用私募基金百科智能问答（REACT_DOCSTORE示例）")
    while True:
        try:
            question = input("请输入您的私募基金相关问题：")
            answer = agent.run(question)
            print("\n【智能答复】：", answer)
        except KeyboardInterrupt:
            print("\n已退出。")
            break 