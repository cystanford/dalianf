#!/usr/bin/env python
# coding: utf-8

# ### 使用 Unsloth 框架对 Qwen2.5-7B 模型进行微调的示例代码
# ### 本代码可以在免费的 Tesla T4 Google Colab 实例上运行 https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(7B)-Alpaca.ipynb

# In[1]:


# 导入必要的库
from unsloth import FastLanguageModel
import torch

# 设置模型参数
max_seq_length = 2048  # 设置最大序列长度，支持 RoPE 缩放
dtype = None  # 数据类型，None 表示自动检测。Tesla T4 使用 Float16，Ampere+ 使用 Bfloat16
load_in_4bit = True  # 使用 4bit 量化来减少内存使用

# 加载预训练模型和分词器
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct",  # 使用Qwen2.5-7B模型
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)


# In[2]:


# 添加LoRA适配器，只需要更新1-10%的参数
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # LoRA秩，建议使用8、16、32、64、128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],  # 需要应用LoRA的模块
    lora_alpha = 16,  # LoRA缩放因子
    lora_dropout = 0,  # LoRA dropout率，0为优化设置
    bias = "none",    # 偏置项设置，none为优化设置
    use_gradient_checkpointing = "unsloth",  # 使用unsloth的梯度检查点，可减少30%显存使用
    random_state = 3407,  # 随机种子
    use_rslora = False,  # 是否使用rank stabilized LoRA
    loftq_config = None,  # LoftQ配置
)


# ### 数据准备

# In[3]:


import os
import pandas as pd
from datasets import Dataset

# 定义医疗对话的提示模板
medical_prompt = """你是一个专业的医疗助手。请根据患者的问题提供专业、准确的回答。

### 问题：
{}

### 回答：
{}"""

# 获取结束标记
EOS_TOKEN = tokenizer.eos_token

def read_csv_with_encoding(file_path):
    """尝试使用不同的编码读取CSV文件"""
    encodings = ['gbk', 'gb2312', 'gb18030', 'utf-8']
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法使用任何编码读取文件: {file_path}")

def load_medical_data(data_dir):
    """加载医疗对话数据"""
    data = []
    departments = {
        'IM_内科': '内科',
        'Surgical_外科': '外科',
        'Pediatric_儿科': '儿科',
        'Oncology_肿瘤科': '肿瘤科',
        'OAGD_妇产科': '妇产科',
        'Andriatria_男科': '男科'
    }
    
    # 遍历所有科室目录
    for dept_dir, dept_name in departments.items():
        dept_path = os.path.join(data_dir, dept_dir)
        if not os.path.exists(dept_path):
            print(f"目录不存在: {dept_path}")
            continue
            
        print(f"\n处理{dept_name}数据...")
        
        # 获取该科室下的所有CSV文件
        csv_files = [f for f in os.listdir(dept_path) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            file_path = os.path.join(dept_path, csv_file)
            print(f"正在处理文件: {csv_file}")
            
            try:
                # 读取CSV文件
                df = read_csv_with_encoding(file_path)
                
                # 打印列名，帮助调试
                print(f"文件 {csv_file} 的列名: {df.columns.tolist()}")
                
                # 处理每一行数据
                for _, row in df.iterrows():
                    try:
                        # 获取问题和回答（尝试不同的列名）
                        question = None
                        answer = None
                        
                        # 尝试不同的列名
                        if 'question' in row:
                            question = str(row['question']).strip()
                        elif '问题' in row:
                            question = str(row['问题']).strip()
                        elif 'ask' in row:
                            question = str(row['ask']).strip()
                            
                        if 'answer' in row:
                            answer = str(row['answer']).strip()
                        elif '回答' in row:
                            answer = str(row['回答']).strip()
                        elif 'response' in row:
                            answer = str(row['response']).strip()
                        
                        # 过滤无效数据
                        if not question or not answer:
                            continue
                            
                        # 限制长度
                        if len(question) > 200 or len(answer) > 200:
                            continue
                            
                        # 添加到数据列表
                        data.append({
                            "instruction": "请回答以下医疗相关问题",
                            "input": question,
                            "output": answer
                        })
                        
                    except Exception as e:
                        print(f"处理数据行时出错: {e}")
                        continue
                        
            except Exception as e:
                print(f"处理文件 {csv_file} 时出错: {e}")
                continue
    
    # 验证数据
    if not data:
        raise ValueError("没有成功处理任何数据！")
        
    print(f"\n成功处理 {len(data)} 条数据")
    return Dataset.from_list(data)

def formatting_prompts_func(examples):
    """格式化提示"""
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = medical_prompt.format(input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

# 加载医疗数据集
dataset = load_medical_data("Data_数据")
dataset = dataset.map(formatting_prompts_func, batched=True)


# ### 模型训练

# In[4]:


# 设置训练参数和训练器
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# 定义训练参数
training_args = TrainingArguments(
        per_device_train_batch_size = 2,  # 每个设备的训练批次大小
        gradient_accumulation_steps = 4,  # 梯度累积步数
        warmup_steps = 5,  # 预热步数
        #max_steps = 60,  # 最大训练步数
        max_steps = -1,  # 不使用max_steps
        num_train_epochs = 3,  # 训练3个epoch
        learning_rate = 2e-4,  # 学习率
        fp16 = not is_bfloat16_supported(),  # 是否使用FP16
        bf16 = is_bfloat16_supported(),  # 是否使用BF16
        logging_steps = 1,  # 日志记录步数
        optim = "adamw_8bit",  # 优化器
        weight_decay = 0.01,  # 权重衰减
        lr_scheduler_type = "linear",  # 学习率调度器类型
        seed = 3407,  # 随机种子
        output_dir = "outputs",  # 输出目录
        report_to = "none",  # 报告方式
    )

# 创建SFTTrainer实例
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,  # 对于短序列可以设置为True，训练速度提升5倍
    args = training_args,
)


# In[5]:


# 显示当前GPU内存状态
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# In[6]:


# 开始训练
trainer_stats = trainer.train()


# In[22]:


# 显示训练后的内存和时间统计
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# ### 模型推理

# In[15]:


# 模型推理示例
def generate_medical_response(question):
    """生成医疗回答"""
    FastLanguageModel.for_inference(model)  # 启用原生2倍速推理
    inputs = tokenizer(
        [medical_prompt.format(question, "")],
        return_tensors="pt"
    ).to("cuda")
    
    from transformers import TextStreamer
    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )

    
# 测试问题
test_questions = [
    "我最近总是感觉头晕，应该怎么办？",
    "感冒发烧应该吃什么药？",
    "高血压患者需要注意什么？"
]

for question in test_questions:
    print("\n" + "="*50)
    print(f"问题：{question}")
    print("回答：")
    generate_medical_response(question) 


# ### 微调模型保存
# **[注意]** 这里只是LoRA参数，不是完整模型。

# In[23]:


# 保存模型
model.save_pretrained("lora_model_medical")  # 本地保存
tokenizer.save_pretrained("lora_model_medical")


# In[24]:


# 加载保存的模型进行推理
if True:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model",  # 训练时使用的模型
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)  # 启用原生2倍速推理
    
question = "我最近总是感觉头晕，应该怎么办？"
generate_medical_response(question) 


# In[25]:


# 加载保存的模型进行推理
if True:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model_medical",  # 保存的模型
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)  # 启用原生2倍速推理
    
question = "我最近总是感觉头晕，应该怎么办？"
generate_medical_response(question) 


# In[26]:


# 加载保存的模型进行推理
if True:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct",  # 基础模型
        adapter_name = "lora_model_medical",  # LoRA权重
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)  # 启用原生2倍速推理

question = "我最近总是感觉头晕，应该怎么办？"
generate_medical_response(question)


# In[27]:


question = "我最近总是感觉头晕，应该怎么办？"
generate_medical_response(question)

