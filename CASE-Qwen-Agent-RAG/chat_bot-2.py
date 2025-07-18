import os
import asyncio
from typing import Optional
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
import pandas as pd
from sqlalchemy import create_engine
from qwen_agent.tools.base import BaseTool, register_tool
import matplotlib.pyplot as plt
import io
import base64
import time
import numpy as np
import json

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 优先使用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义资源文件根目录
ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), 'resource')

# 配置 DashScope
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')  # 从环境变量获取 API Key
dashscope.timeout = 30  # 设置超时时间为 30 秒

# ====== 知识库常见问题生成助手 system prompt 和函数描述 ======
system_prompt = """我是知识库常见问题生成助手，专门用于分析docs文件夹下的保险相关文档，生成常见问题和答案。

我可以：
1. 分析保险文档内容
2. 生成相关的常见问题
3. 提供准确的答案
4. 总结文档要点

请根据用户的需求，从相关文档中提取信息并生成有用的常见问题。
"""

functions_desc = [
    {
        "name": "generate_faq",
        "description": "根据docs文件夹下的知识库文档，生成常见问题和答案",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "要生成FAQ的主题或保险类型，例如：雇主责任险、企业团体意外险等",
                },
                "question_count": {
                    "type": "integer",
                    "description": "要生成的常见问题数量，默认为5个",
                }
            },
            "required": ["topic"],
        },
    },
]

# ====== 会话隔离 DataFrame 存储 ======
# 用于存储每个会话的 DataFrame，避免多用户数据串扰
_last_df_dict = {}

def get_session_id(kwargs):
    """根据 kwargs 获取当前会话的唯一 session_id，这里用 messages 的 id"""
    messages = kwargs.get('messages')
    if messages is not None:
        return id(messages)
    return None

# ====== 封装 qwen-turbo-latest 模型响应函数 ======
def get_qwen_response(messages):
    """使用 qwen-turbo-latest 模型获取响应"""
    try:
        response = dashscope.Generation.call(
            model='qwen-turbo-latest',
            messages=messages,
            result_format='message'  # 将输出设置为message形式
        )
        return response
    except Exception as e:
        print(f"调用 qwen-turbo-latest 模型失败: {str(e)}")
        return None

# ====== 读取文档内容函数 ======
def read_doc_content(file_path):
    """读取文档文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"读取文件 {file_path} 失败: {str(e)}")
        return ""

# ====== generate_faq 工具类实现 ======
@register_tool('generate_faq')
class GenerateFAQTool(BaseTool):
    """
    根据docs文件夹下的知识库文档，生成常见问题和答案
    """
    description = '根据docs文件夹下的知识库文档，生成常见问题和答案'
    parameters = [{
        'name': 'topic',
        'type': 'string',
        'description': '要生成FAQ的主题或保险类型，例如：雇主责任险、企业团体意外险等',
        'required': True
    }, {
        'name': 'question_count',
        'type': 'integer',
        'description': '要生成的常见问题数量，默认为5个',
        'required': False
    }]

    def call(self, params: str, **kwargs) -> str:
        import json
        args = json.loads(params)
        topic = args['topic']
        question_count = args.get('question_count', 5)
        
        try:
            # 获取docs文件夹下的所有文件
            docs_dir = os.path.join('./', 'docs')
            if not os.path.exists(docs_dir):
                return "错误：docs文件夹不存在"
            
            # 查找相关的文档文件
            relevant_files = []
            for file in os.listdir(docs_dir):
                if topic.lower() in file.lower() or any(keyword in file for keyword in topic.split()):
                    file_path = os.path.join(docs_dir, file)
                    if os.path.isfile(file_path) and file.endswith('.txt'):
                        relevant_files.append(file_path)
            
            if not relevant_files:
                return f"未找到与主题 '{topic}' 相关的文档文件"
            
            # 读取相关文档内容
            doc_contents = []
            for file_path in relevant_files:
                content = read_doc_content(file_path)
                if content:
                    doc_contents.append(f"文档 {os.path.basename(file_path)}:\n{content}\n")
            
            if not doc_contents:
                return "无法读取文档内容"
            
            # 合并文档内容
            combined_content = "\n".join(doc_contents)
            
            # 使用 qwen-turbo-latest 生成FAQ
            messages = [
                {
                    "role": "system", 
                    "content": f"""你是一名保险专家，请根据以下保险文档内容，生成{question_count}个常见问题和答案。

要求：
1. 问题要实用、具体，用户经常咨询的问题
2. 答案要准确、详细，基于文档内容
3. 使用中文回答
4. 格式要清晰，每个问题用"Q:"开头，答案用"A:"开头
5. 问题要涵盖保险责任、理赔流程、投保条件等关键信息

文档内容：
{combined_content}"""
                },
                {
                    "role": "user", 
                    "content": f"请为{topic}生成{question_count}个常见问题和答案"
                }
            ]
            
            response = get_qwen_response(messages)
            if response and hasattr(response, 'output') and hasattr(response.output, 'choices'):
                return response.output.choices[0].message.content
            else:
                return "生成FAQ失败，请检查模型调用是否正常"
                
        except Exception as e:
            return f"生成FAQ时出错: {str(e)}"

# ====== 初始化知识库常见问题生成助手服务 ======
def init_agent_service():
    """初始化知识库常见问题生成助手服务"""
    llm_cfg = {
        'model': 'qwen-turbo-2025-04-28',
        'timeout': 30,
        'retry_count': 3,
    }
    
    # 获取docs文件夹下所有文件
    file_dir = os.path.join('./', 'docs')
    files = []
    if os.path.exists(file_dir):
        # 遍历目录下的所有文件
        for file in os.listdir(file_dir):
            file_path = os.path.join(file_dir, file)
            if os.path.isfile(file_path):  # 确保是文件而不是目录
                files.append(file_path)
    print('可用文档文件:', files)

    try:
        bot = Assistant(
            llm=llm_cfg,
            name='知识库常见问题生成助手',
            description='分析保险文档并生成常见问题',
            system_message=system_prompt,
            function_list=['generate_faq'],  # 注册FAQ生成工具
            files=files
        )
        print("助手初始化成功！")
        return bot
    except Exception as e:
        print(f"助手初始化失败: {str(e)}")
        raise

def app_tui():
    """终端交互模式
    
    提供命令行交互界面，支持：
    - 连续对话
    - 文件输入
    - 实时响应
    """
    try:
        # 初始化助手
        bot = init_agent_service()

        # 对话历史
        messages = []
        while True:
            try:
                # 获取用户输入
                query = input('用户问题: ')
                # 获取可选的文件输入
                file = input('文件路径 (直接回车跳过): ').strip()
                
                # 输入验证
                if not query:
                    print('用户问题不能为空！')
                    continue
                    
                # 构建消息
                if not file:
                    messages.append({'role': 'user', 'content': query})
                else:
                    messages.append({'role': 'user', 'content': [{'text': query}, {'file': file}]})

                print("正在处理您的请求...")
                # 运行助手并处理响应
                response = []
                for response in bot.run(messages):
                    print('助手回复:', response)
                messages.extend(response)
            except Exception as e:
                print(f"处理请求时出错: {str(e)}")
                print("请重试或输入新的问题")
    except Exception as e:
        print(f"启动终端模式失败: {str(e)}")

def app_gui():
    """图形界面模式，提供 Web 图形界面"""
    try:
        print("正在启动 Web 界面...")
        # 初始化助手
        bot = init_agent_service()
        # 配置聊天界面，列举3个典型问题
        chatbot_config = {
            'prompt.suggestions': [
                '为雇主责任险生成5个常见问题',
                '为企业团体综合意外险生成常见问题',
                '分析平安商业综合责任保险的常见问题',
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
    print("请选择运行模式:")
    print("1. 图形界面模式 (GUI)")
    print("2. 终端交互模式 (TUI)")
    
    choice = input("请输入选择 (1 或 2，默认为 1): ").strip()
    
    if choice == '2':
        app_tui()          # 终端交互模式
    else:
        app_gui()          # 图形界面模式（默认） 