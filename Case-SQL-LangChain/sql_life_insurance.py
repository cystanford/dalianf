#!/usr/bin/env python
# coding: utf-8

# In[1]:


from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor

db_user = "student123"
db_password = "student321"
db_host = "rm-uf6z891lon6dxuqblqo.mysql.rds.aliyuncs.com:3306"
db_name = "life_insurance"
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
#engine = create_engine('mysql+mysqlconnector://root:passw0rdcc4@localhost:3306/wucai')
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

# Task1
agent_executor.run("获取所有客户的姓名和联系电话")


# In[4]:


agent_executor.run("查询所有未支付保费的保单号和客户姓名。")


# In[5]:


agent_executor.run("找出所有理赔金额大于10000元的理赔记录，并列出相关客户的姓名和联系电话。")

