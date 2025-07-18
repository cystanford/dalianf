# RAG Challenge 获奖方案

**项目详细介绍：**
- 俄文：https://habr.com/ru/articles/893356/
- 英文：https://abdullin.com/ilya/how-to-build-best-rag/

本仓库包含RAG Challenge竞赛双奖项获奖方案。该系统通过以下技术组合，在公司年报问答任务中取得了业界领先的效果：

- 基于Docling的自定义PDF解析
- 支持父文档检索的向量搜索
- LLM重排序提升上下文相关性
- 结构化输出提示词+链式思考推理
- 多公司对比问题的查询路由

## 声明

本代码为竞赛用代码，结构较为粗糙但可用。请注意：

- IBM Watson集成无法使用（仅限比赛期间）
- 代码可能存在临时方案和奇怪的兼容处理
- 没有测试用例，错误处理较少——请谨慎使用
- 需要自备OpenAI/Gemini等API密钥
- PDF解析建议使用GPU（作者用的是4090）

如需生产级代码，这里并不适合。但如果你想学习RAG各类技术实现细节，非常值得参考！

## 快速开始

克隆并安装依赖：
```bash
git clone https://github.com/IlyaRice/RAG-Challenge-2.git
cd RAG-Challenge-2
python -m venv venv
venv\Scripts\Activate.ps1  # Windows (PowerShell)
pip install -e . -r requirements.txt
```

将 `env` 重命名为 `.env`，并填写你的API密钥。

## 测试数据集

本仓库包含两个数据集：

1. 小型测试集（`data/test_set/`），含5份年报及相关问题
2. 完整ERC2竞赛数据集（`data/erc2_set/`），含全部竞赛问题和年报

每个数据集目录下均有README，说明具体文件和使用方法。你可以：

- 学习示例问题、年报和系统输出
- 用提供的PDF从头运行全流程
- 直接用预处理数据跳转到任意流程阶段

详细内容和用法见：
- `data/test_set/README.md` - 小型测试集说明
- `data/erc2_set/README.md` - 完整竞赛集说明

## 使用方法

你可以在 `src/pipeline.py` 中取消注释任意流程方法，然后运行：
```bash
python .\src\pipeline.py
```

也可以在数据目录下用 `main.py` 运行任意流程阶段：
```bash
cd .\data\test_set\
python ..\..\main.py process-questions --config max_nst_o3m
```

### CLI命令

查看所有可用命令：
```bash
python main.py --help
```

可用命令：
- `download-models` - 下载所需docling模型
- `parse-pdfs` - 并行解析PDF年报
- `serialize-tables` - 处理已解析报告中的表格
- `process-reports` - 全流程处理已解析报告
- `process-questions` - 用指定配置处理问题

每个命令均有详细参数。例如：
```bash
python main.py parse-pdfs --help
# 查看如 --parallel/--sequential, --chunk-size, --max-workers 等参数

python main.py process-reports --config ser_tab
# 用表格序列化配置处理报告
```

## 常用配置

- `max_nst_o3m` - 最优表现配置，使用OpenAI o3-mini模型
- `ibm_llama70b` - IBM Llama 70B大模型方案
- `gemini_thinking` - Gemini大窗口全上下文问答（非RAG）

更多配置和细节见 `pipeline.py`。

## 许可证

MIT 