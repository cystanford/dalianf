from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Tongyi  # 导入通义千问Tongyi模型
from langchain_core.output_parsers import StrOutputParser
import dashscope
import os

# 从环境变量获取 dashscope 的 API Key
api_key = os.environ.get('DASHSCOPE_API_KEY')
dashscope.api_key = api_key

# stream=True 让LLM支持流式输出
llm = Tongyi(model_name="qwen-turbo", dashscope_api_key=api_key, stream=True)

# 定义三个子任务：翻译->处理->回译
translate_to_en = ChatPromptTemplate.from_template("Translate this to English: {input}") | llm | StrOutputParser()
process_text = ChatPromptTemplate.from_template("Analyze this text: {text}") | llm | StrOutputParser()
translate_to_cn = ChatPromptTemplate.from_template("Translate this to Chinese: {output}") | llm | StrOutputParser()

# 组合成多任务链
workflow = {"text": translate_to_en} | process_text | translate_to_cn
#workflow.invoke({"input": "北京有哪些好吃的地方，简略回答不超过200字"})

# 使用stream方法，边生成边打印
for chunk in workflow.stream({"input": "北京有哪些好吃的地方，简略回答不超过200字"}):
    print(chunk, end="", flush=True)
print()  # 换行