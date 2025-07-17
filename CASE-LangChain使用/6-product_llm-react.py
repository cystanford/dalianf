import os
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.llms import Tongyi  # 导入通义千问Tongyi模型
from langchain.llms.base import BaseLLM

# 设置通义千问API密钥
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 模拟公司产品和公司介绍的数据源
class TeslaDataSource:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    # 工具1：产品描述
    def find_product_description(self, product_name: str) -> str:
        """模拟公司产品的数据库"""
        product_info = {
            "Model 3": "具有简洁、动感的外观设计，流线型车身和现代化前脸。定价23.19-33.19万",
            "Model Y": "在外观上与Model 3相似，但采用了更高的车身和更大的后备箱空间。定价26.39-36.39万",
            "Model X": "拥有独特的翅子门设计和更加大胆的外观风格。定价89.89-105.89万",
        }
        # 基于产品名称 => 产品描述
        return product_info.get(product_name, "没有找到这个产品")

    # 工具2：公司介绍
    def find_company_info(self, query: str) -> str:
        """模拟公司介绍文档数据库，让llm根据信息回答问题"""
        context = """
        特斯拉最知名的产品是电动汽车，其中包括Model S、Model 3、Model X和Model Y等多款车型。
        特斯拉以其技术创新、高性能和领先的自动驾驶技术而闻名。公司不断推动自动驾驶技术的研发，并在车辆中引入了各种驾驶辅助功能，如自动紧急制动、自适应巡航控制和车道保持辅助等。
        """
        return f"公司信息：{context}\n你的问题：{query}"

if __name__ == "__main__":
    # 定义LLM
    llm = Tongyi(model_name="qwen-turbo", dashscope_api_key=DASHSCOPE_API_KEY)
    # 自有数据
    tesla_data_source = TeslaDataSource(llm)
    # 定义的Tools
    tools = [
        Tool(
            name="查询产品名称",
            func=tesla_data_source.find_product_description,
            description="通过产品名称找到产品描述时用的工具，输入的是产品名称",
        ),
        Tool(
            name="公司相关信息",
            func=tesla_data_source.find_company_info,
            description="当用户询问公司相关的问题，可以通过这个工具了解公司信息",
        ),
    ]
    # 使用LangChain内置ReAct Agent，自动推理和工具调用
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    # 主过程：可以一直提问下去，直到Ctrl+C
    while True:
        try:
            user_input = input("请输入您的问题：")
            response = agent.run(user_input)
            print(response)
        except KeyboardInterrupt:
            break 