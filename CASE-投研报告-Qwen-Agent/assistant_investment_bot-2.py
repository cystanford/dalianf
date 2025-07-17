# 将max_results设置为2
import os
import asyncio
from typing import Optional
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
import pandas as pd
from sqlalchemy import create_engine
from qwen_agent.tools.base import BaseTool, register_tool

# 定义资源文件根目录
ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), 'resource')

# 配置 DashScope
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')  # 从环境变量获取 API Key
dashscope.timeout = 30  # 设置超时时间为 30 秒

# ====== 投研助手 system prompt 和函数描述 ======
system_prompt = """我是投研助手，以下是投研的步骤：
Step1，通过planner与用户沟通，确认投研报告撰写的框架planning
这里需要用户确认，确认后再进行后续的步骤。
这个时候不需要调用tavily-mcp，也不需要调用reporter，只需要调用planner工具。
只有当用户确认投研框架后，再进行后续的步骤（
Step2，通过tavily-mcp工具(max_results设置为2)，针对planning列出来的研究方向，进行信息收集
Step3，通过reporter撰写投研报告，反馈给用户）。

说明：
1、用户没有确认投研框架前，不需要调用tavily-mcp和reporter工具。
2、最后给用户撰写完整的投研报告的时候，需要通过reporter工具。
"""

tools = [{
    "mcpServers": {
        "tavily-mcp": {
            "command": "npx",
            "args": ["-y", "tavily-mcp@0.2.0"],
            "env": {
                "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", "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")
            }
        }
    }
}]

# 注册 planner 工具
@register_tool('planner')
class PlannerTool(BaseTool):
    """
    投研规划工具，调用大模型（qwen-turbo），根据 planner.md 的提示词进行投研规划，撰写投研报告框架。
    """
    description = '投研规划工具，输出投研框架。'
    parameters = [
        {
            'name': 'user_input',
            'type': 'string',
            'description': '用户的投研需求描述',
            'required': True
        }
    ]

    @staticmethod
    def get_prompt():
        # 直接读取 planner.md 文件内容作为 prompt
        with open('planner.md', 'r', encoding='utf-8') as f:
            return f.read()

    def call(self, params: str, **kwargs) -> str:
        import json
        args = json.loads(params)
        user_input = args['user_input']
        prompt = self.get_prompt()
        import dashscope
        dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}
        ]
        response = dashscope.Generation.call(
            model='qwen-turbo',
            messages=messages,
            result_format='message'
        )
        return response.output.choices[0].message.content

tools.append('planner')

# 注册 reporter 工具
@register_tool('reporter')
class ReporterTool(BaseTool):
    """
    投研报告撰写工具，根据 planner 整理并用户确认过的框架，生成最终投研报告。
    """
    description = '投研报告撰写工具，严格按照已确认的框架生成报告。'
    parameters = [
        {
            'name': 'framework',
            'type': 'string',
            'description': '已确认的投研报告框架（由planner输出并经用户确认）',
            'required': True
        },
        {
            'name': 'research_data',
            'type': 'string',
            'description': '调研和信息收集的主要内容（可选，若有信息收集结果可传入）',
            'required': False
        }
    ]

    @staticmethod
    def get_prompt():
        # 读取 reporter.md 文件内容作为 prompt
        with open('reporter.md', 'r', encoding='utf-8') as f:
            return f.read()

    def call(self, params: str, **kwargs) -> str:
        import json
        args = json.loads(params)
        framework = args['framework']
        research_data = args.get('research_data', '')
        prompt = self.get_prompt()
        import dashscope
        dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')
        # 构建消息
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"报告框架如下：\n{framework}\n\n调研内容如下：\n{research_data}"}
        ]
        print('正在撰写投研报告...')
        print(messages)
        response = dashscope.Generation.call(
            model='qwen-turbo',
            messages=messages,
            result_format='message'
        )
        return response.output.choices[0].message.content

tools.append('reporter')

# ====== 初始化投研助手 ======
def init_agent_service():
    """初始化投研助手服务"""
    llm_cfg = {
        #'model': 'qwen-turbo-2025-04-28',
        'model': 'qwen-turbo-latest',
        'timeout': 30,
        'retry_count': 3,
    }
    try:
        bot = Assistant(
            llm=llm_cfg,
            name='投研助手',
            description='投研报告撰写与行业数据分析',
            system_message=system_prompt,
            #function_list=['exc_sql'],  # 只传工具名字符串
            function_list=tools,
            files=['./private_data/【财报】中芯国际：中芯国际2024年年度报告.pdf']
        )
        print("助手初始化成功！")
        return bot
    except Exception as e:
        print(f"助手初始化失败: {str(e)}")
        raise


def app_gui():
    """图形界面模式，提供 Web 图形界面"""
    try:
        print("正在启动 Web 界面...")
        # 初始化助手
        bot = init_agent_service()
        # 配置聊天界面，列举3个典型投研问题
        chatbot_config = {
            'prompt.suggestions': [
                '请帮我分析中芯国际2024年一季度的业绩亮点和风险点',
                '请根据最新的行业报告，分析国产芯片的市场空间和主要驱动力',
                '请结合财报和券商研报，写一份中芯国际的投资价值分析',
            ]
        }
        print("Web 界面准备就绪，正在启动服务...")
        # 启动 Web 界面
        WebUI(
            bot,
            chatbot_config=chatbot_config
        ).run()
    except Exception as e:
        print(f"启动 Web 界面失败: {str(e)}")
        print("请检查网络连接和 API Key 配置")


if __name__ == '__main__':
    # 运行模式选择
    app_gui()          # 图形界面模式（默认）