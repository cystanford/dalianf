#!/usr/bin/env python
# coding: utf-8

# ## 使用DeepSeek进行数据表的查询

# In[1]:


from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor

db_user = "student123"
db_password = "student321"
#db_host = "localhost:3306"
db_host = "rm-uf6z891lon6dxuqblqo.mysql.rds.aliyuncs.com:3306"
db_name = "action"
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
db


# In[2]:


from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(
    temperature=0.01,
    model="deepseek-chat",  
    openai_api_key = "sk-9846f14a2104490b960adbf5c5b3b32e",
    openai_api_base="https://api.deepseek.com"
)

# 需要设置llm
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)


# In[4]:


# Task: 描述数据表
agent_executor.run("描述与订单相关的表及其关系")


# In[4]:


# 这个任务，实际上数据库中 没有HeroDetails表
agent_executor.run("描述HeroDetails表")


# In[5]:


agent_executor.run("描述Hero表")


# In[6]:


agent_executor.run("找出英雄攻击力最高的前5个英雄")

