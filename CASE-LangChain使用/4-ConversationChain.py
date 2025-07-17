#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain_community.llms import Tongyi  # 导入通义千问Tongyi模型
from langchain.agents import AgentType
from langchain import ConversationChain
import dashscope

# 从环境变量获取 dashscope 的 API Key
api_key = os.environ.get('DASHSCOPE_API_KEY')
dashscope.api_key = api_key

# 加载 Tongyi 模型
llm = Tongyi(model_name="qwen-turbo", dashscope_api_key=api_key)  # 使用通义千问qwen-turbo模型
# 使用带有memory的ConversationChain
conversation = ConversationChain(llm=llm, verbose=True)

output = conversation.predict(input="Hi there!")
print(output)


# In[2]:


output = conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
print(output)

