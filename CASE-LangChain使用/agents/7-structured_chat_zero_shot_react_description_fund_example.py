"""
【AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION 简要说明】
本示例基于 LangChain 内置的 STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION 类型 Agent。

特点：
- 支持结构化输入，能够处理包含多个字段的复杂问题。
- Agent 会自动将用户输入（如 dict）或 LLM 生成的 action_input（如字符串）映射到工具参数。
- 工具描述（description）用于指导 LLM 如何组织输入内容。
- 适合需要多参数、结构化问答的业务场景。

与 ZERO_SHOT_REACT_DESCRIPTION 区别：
- ZERO_SHOT_REACT_DESCRIPTION 只支持单一字符串输入，适合简单问答。
- STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION 支持 dict 结构输入，适合多字段、多参数的复杂业务。

适用场景：
- 需要用户输入多个参数（如“基金名称”、“分析维度”等）的智能问答。
- 业务流程复杂、需要结构化信息交互的场景。

本文件为私募基金多维度分析的结构化智能问答示例。
"""
import os
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.llms import Tongyi
import json

# 设置通义千问API密钥
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 初始化LLM
llm = Tongyi(model_name="qwen-turbo", dashscope_api_key=DASHSCOPE_API_KEY)

# 定义一个结构化输入的私募基金分析工具
class FundAnalysisTool:
    def analyze(self, fund_name: str, metric: str) -> str:
        # 简单模拟多维度分析
        data = {
            ("稳健配置", "收益"): "近三年年化收益率8.5%",
            ("稳健配置", "风险"): "最大回撤5.2%",
            ("创新成长", "收益"): "近三年年化收益率18.2%",
            ("创新成长", "风险"): "最大回撤15.7%"
        }
        return data.get((fund_name, metric), "未找到相关分析数据。")

def safe_get(d, key):
    # 如果是 dict，按 key 查找
    if isinstance(d, dict):
        for k in d.keys():
            if k.strip("'\"") == key:
                return d[k]
        return None
    # 如果是 str，尝试用简单分割法解析
    elif isinstance(d, str):
        # 例如 "基金名称: 创新成长, 分析维度: 风险"
        parts = [p.strip() for p in d.split(",") if ":" in p]
        for part in parts:
            k, v = part.split(":", 1)
            if k.strip() == key:
                return v.strip()
        return None
    else:
        return None

# Tool定义
fund_analysis_tool = Tool(
    name="私募基金多维度分析",
    func=lambda x: FundAnalysisTool().analyze(
        safe_get(x, "基金名称"),
        safe_get(x, "分析维度")
    ),
    description="输入结构如：基金名称: xxx, 分析维度: 收益/风险，返回对应分析"
)

# 初始化STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION类型Agent
agent = initialize_agent(
    [fund_analysis_tool],
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

if __name__ == "__main__":
    print("欢迎使用私募基金多维度分析（STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION示例）")
    while True:
        try:
            print("请输入结构化问题，推荐输入JSON格式，如：{\"基金名称\": \"创新成长\", \"分析维度\": \"风险\"}")
            question = input("输入：")
            # 用json解析，避免key带引号
            try:
                query = json.loads(question)
            except Exception:
                print("输入格式有误，请输入类似{\"基金名称\": \"创新成长\", \"分析维度\": \"风险\"}的结构化内容。")
                continue
            # 调用agent时
            answer = agent.run({"input": query})
            print("分析结果：", answer)
        except KeyboardInterrupt:
            print("\n已退出。")
            break 