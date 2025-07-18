#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Qwen2.5(7B)-GRPO 模型微调代码
使用Unsloth框架进行Qwen2.5-7B模型的GRPO(Generative Reward-Paired Optimization)微调
pip cache purge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
"""
import unsloth
from unsloth import FastLanguageModel
import torch

# 设置模型参数
max_seq_length = 1024  # 可以增加以获得更长的推理轨迹
lora_rank = 32  # 更大的rank会使模型更智能，但训练更慢

# 加载预训练模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True,  # 使用4bit量化加载
    fast_inference = True,  # 启用vLLM快速推理
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6,  # 如果显存不足可以降低
)

# 配置LoRA参数
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,  # LoRA秩，建议使用8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],  # 如果显存不足可以移除QKVO
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth",  # 启用长上下文微调
    random_state = 3407,
)


# In[2]:


import torch
import torchvision

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)


# In[3]:


# 数据准备部分
import re
from datasets import load_dataset, Dataset

# 定义系统提示词
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# 定义XML格式模板
XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    """从XML格式文本中提取答案部分"""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    """从带####标记的文本中提取答案"""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_questions(split = "train") -> Dataset:
    """加载GSM8K数据集并进行预处理"""
    #data = load_dataset('openai/gsm8k', 'main')[split]
    data = load_dataset('/root/autodl-tmp/datasets/gsm8k', 'main')[split]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })
    return data

# 加载数据集
dataset = get_gsm8k_questions()


# In[6]:


dataset


# In[4]:


# 定义各种奖励函数
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """正确性奖励函数：检查答案是否正确"""
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    """整数奖励函数：检查答案是否为整数"""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """严格格式奖励函数：检查是否完全符合XML格式"""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """宽松格式奖励函数：检查是否基本符合XML格式"""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    """计算XML标签的完整性得分"""
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """XML标签计数奖励函数"""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

# 训练配置
max_prompt_length = 256

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    learning_rate = 5e-6,  # 学习率
    adam_beta1 = 0.9,      # Adam优化器参数
    adam_beta2 = 0.99,
    weight_decay = 0.1,    # 权重衰减
    warmup_ratio = 0.1,    # 预热比例
    lr_scheduler_type = "cosine",  # 学习率调度器类型
    optim = "paged_adamw_8bit",    # 优化器类型
    logging_steps = 1,             # 日志记录步数
    per_device_train_batch_size = 1,  # 每个设备的训练批次大小
    gradient_accumulation_steps = 1,  # 梯度累积步数
    num_generations = 6,              # 生成数量
    max_prompt_length = max_prompt_length,  # 最大提示词长度
    max_completion_length = max_seq_length - max_prompt_length,  # 最大完成长度
    max_steps = 250,                 # 最大训练步数
    save_steps = 250,                # 保存步数
    max_grad_norm = 0.1,             # 最大梯度范数
    report_to = "none",              # 报告目标
    output_dir = "outputs",          # 输出目录
)

# 初始化训练器
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
)

# 开始训练
trainer.train()


# In[ ]:


# 保存训练好的LoRA模型
model.save_lora("grpo_saved_lora")


# In[1]:


# 测试模型
text = tokenizer.apply_chat_template([
    {"role" : "system", "content" : SYSTEM_PROMPT},
    {"role" : "user", "content" : "Calculate pi."},
], tokenize = False, add_generation_prompt = True)

from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature = 0.8,    # 采样温度
    top_p = 0.95,        # top-p采样参数
    max_tokens = 2048,   # 最大生成token数
)

# 使用保存的LoRA进行推理
output = model.fast_generate(
    text,
    sampling_params = sampling_params,
    lora_request = model.load_lora("grpo_saved_lora"),
)[0].outputs[0].text

print(output)


# In[ ]:


# 模型保存选项
# 保存为16位浮点数
if False: 
    model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit")
    model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

# 保存为4位整数
if False: 
    model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit")
    model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

# 仅保存LoRA适配器
if False: 
    model.save_pretrained_merged("model", tokenizer, save_method = "lora")
    model.push_to_hub_merged("hf/model", tokenizer, save_method = "lora", token = "")

# GGUF/llama.cpp转换选项
# 保存为8位Q8_0
if False: 
    model.save_pretrained_gguf("model", tokenizer)
    model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# 保存为16位GGUF
if False: 
    model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
    model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

# 保存为q4_k_m GGUF
if False: 
    model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
    model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")

# 保存多个GGUF选项
if False:
    model.push_to_hub_gguf(
        "hf/model",
        tokenizer,
        quantization_method = ["q4_k_m", "q8_0", "q5_k_m"],
        token = "",
    )

