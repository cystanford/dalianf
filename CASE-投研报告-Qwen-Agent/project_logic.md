# 投研助手项目整体代码逻辑说明

## 一、项目目标
本项目旨在构建一个智能投研助手，自动化完成投研报告的规划、信息收集与报告撰写，提升投研效率和专业性。用户只需输入投研需求，系统即可输出结构化、专业的投研报告。

## 二、核心流程与模块职责

### 1. 用户交互与主流程
- 用户通过 Web 界面输入投研需求。
- 系统分阶段引导用户，逐步完成投研规划、信息收集、报告撰写。

### 2. 主要步骤
- **Step1：规划确认**
  - 通过 `planner` 工具与用户沟通，梳理并确认投研报告的规划框架。
  - 用户确认后，进入下一步。
- **Step2：信息收集**
  - 通过 `tavily-mcp` 工具，针对已确认的研究方向自动收集外部信息。
  - 收集结果作为后续报告撰写的输入。
- **Step3：报告撰写**
  - 通过 `reporter` 工具，严格按照已确认的框架和收集到的信息，自动生成结构化投研报告。

### 3. 工具注册与参数传递
- 所有工具（如 planner、reporter、tavily-mcp）均通过 `tools` 列表注册到 Agent。
- `tavily-mcp` 工具参数如 `max_results` 应在 Python 实际调用时控制，**不要依赖 tools 配置中的默认参数**。
- `planner` 和 `reporter` 工具通过类装饰器注册，分别负责规划和报告撰写。

### 4. 关键参数机制
- `tavily-mcp` 的 `max_results` 参数需在实际 Python 调用时强制设为 1，防止被外部参数覆盖。
- `planner` 工具根据 `planner.md` 提示词生成调研规划。
- `reporter` 工具根据用户确认的框架和调研内容生成最终报告。

### 5. Web 界面与服务启动
- 通过 `WebUI` 启动图形界面，提供典型投研问题建议，便于用户快速体验。
- `init_agent_service` 负责初始化 Agent，加载 LLM、工具、系统提示词和相关文件。

## 三、代码结构与维护建议
- 各工具职责单一，便于扩展和维护。
- 参数传递链路需统一，防止默认参数被覆盖。
- 业务流程清晰，便于团队协作和二次开发。

## 四、典型调用链路与代码示例
1. 用户输入需求 → planner 生成规划 → 用户确认 → tavily-mcp 收集信息 → reporter 生成报告 → 用户获取结果。
2. 所有关键参数（如 max_results）需在 Python 层最终校验，确保业务逻辑一致。

### 代码逻辑真实示例：

```python
import json
from qwen_agent.tools.base import BaseTool

# Step1: 规划确认
user_query = "请分析中芯国际2024年战略"
plan = PlannerTool().call(json.dumps({"user_input": user_query}))
# 用户确认后进入下一步

# Step2: 信息收集（由 Agent 自动调用 tavily-mcp 工具）
# 你无需手动写 tavily_mcp_search，Agent 会自动根据 tools 注册和参数调度 tavily-mcp
search_params = {"query": "中芯国际 长期战略 2024", "max_results": 1}
# 在实际对话流程中，Agent 会自动将 search_params 传递给 tavily-mcp 工具
# 例如：
# agent.run([{"role": "user", "content": search_params}])
# 你只需保证 max_results=1 被正确传递

# Step3: 报告撰写
confirmed_plan = plan  # 假设用户已确认
report_params = {"framework": confirmed_plan, "research_data": "（此处为收集到的信息）"}
final_report = ReporterTool().call(json.dumps(report_params))

# Agent 初始化与 WebUI 启动
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
llm_cfg = {'model': 'qwen-turbo-latest', 'timeout': 30, 'retry_count': 3}
bot = Assistant(
    llm=llm_cfg,
    name='投研助手',
    description='投研报告撰写与行业数据分析',
    system_message=system_prompt,
    function_list=tools,
    files=['./private_data/【财报】中芯国际：中芯国际2024年年度报告.pdf']
)
WebUI(bot, chatbot_config={
    'prompt.suggestions': [
        '请帮我分析中芯国际2024年一季度的业绩亮点和风险点',
        '请根据最新的行业报告，分析国产芯片的市场空间和主要驱动力',
        '请结合财报和券商研报，写一份中芯国际的投资价值分析',
    ]
}).run()
```

> 说明：
> - `PlannerTool` 和 `ReporterTool` 是在 assistant_investment_bot-2.py 中注册的工具类。
> - tavily-mcp 工具的调用由 Agent 框架自动完成，开发者只需保证参数正确传递。
> - 代码示例与实际项目结构保持一致，便于直接参考和维护。

## 五、总结
本项目通过模块化设计和自动化流程，实现了投研报告的智能规划、信息收集和自动撰写，极大提升了投研效率和专业性。 