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
    #model_name = "unsloth/Qwen2.5-7B",  # 使用Qwen2.5-7B模型
    model_name = "/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct",  # 使用Qwen2.5-7B模型
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)


# In[2]:


# 添加LoRA适配器，只需要更新1-10%的参数
model = FastLanguageModel.get_peft_model(
    model, # Qwen2.5-7B
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


# 定义Alpaca格式的提示模板
alpaca_prompt = """Below is an instruction that describes a task, 
paired with an input that provides further context. 
Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# 获取结束标记
EOS_TOKEN = tokenizer.eos_token

# 定义数据格式化函数
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # 必须添加EOS_TOKEN，否则生成会无限继续
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

# 加载Alpaca数据集
from datasets import load_dataset
dataset = load_dataset("/root/autodl-tmp/datasets/yahma/alpaca-cleaned", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)


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
        max_steps = 60,  # 最大训练步数
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

# 创建SFTTrainer实例 => SFT 监督微调的训练器
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


# In[7]:


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

# In[8]:


# 模型推理部分
FastLanguageModel.for_inference(model)  # 启用原生2倍速推理
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Continue the fibonnaci sequence.",  # 指令
        "1, 1, 2, 3, 5, 8",  # 输入
        "",  # 输出留空用于生成
    )
], return_tensors = "pt").to("cuda")

# 生成输出
outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)


# In[9]:


# 使用TextStreamer进行连续推理
FastLanguageModel.for_inference(model)
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Continue the fibonnaci sequence.",  # 指令
        "1, 1, 2, 3, 5, 8",  # 输入
        "",  # 输出留空用于生成
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)


# ### 微调模型保存
# **[注意]** 这里只是LoRA参数，不是完整模型。

# In[10]:


# 保存模型
model.save_pretrained("lora_model")  # 本地保存
tokenizer.save_pretrained("lora_model")


# In[11]:


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

# 使用保存的模型进行推理示例
inputs = tokenizer(
[
    alpaca_prompt.format(
        "What is a famous tall tower in Paris?",  # 指令
        "",  # 输入
        "",  # 输出留空用于生成
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)


# You can also use Hugging Face's `AutoModelForPeftCausalLM`. Only use this if you do not have `unsloth` installed. It can be hopelessly slow, since `4bit` model downloading is not supported, and Unsloth's **inference is 2x faster**.

# In[15]:


# 使用Hugging Face的AutoModelForPeftCausalLM加载模型（不推荐，因为速度较慢）
if False:
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    model = AutoPeftModelForCausalLM.from_pretrained(
        "lora_model",  # 训练时使用的模型
        load_in_4bit = load_in_4bit,
    )
    tokenizer = AutoTokenizer.from_pretrained("lora_model")


# ### Saving to float16 for VLLM
# 
# We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens.

# In[16]:


# 保存为float16格式（用于VLLM）
# 支持merged_16bit（float16）、merged_4bit（int4）和lora适配器
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

# 保存为4bit格式
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

# 仅保存LoRA适配器
if False: model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "lora", token = "")


# ### GGUF / llama.cpp Conversion
# To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.
# 
# Some supported quant methods (full list on our [Wiki page](https://github.com/unslothai/unsloth/wiki#gguf-quantization-options)):
# * `q8_0` - Fast conversion. High resource use, but generally acceptable.
# * `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
# * `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.
# 
# [**NEW**] To finetune and auto export to Ollama, try our [Ollama notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)

# In[17]:


# GGUF/llama.cpp转换
# 支持多种量化方法：
# q8_0：快速转换，资源使用较高但一般可接受
# q4_k_m：推荐使用，对attention.wv和feed_forward.w2使用Q6_K，其他使用Q4_K
# q5_k_m：推荐使用，对attention.wv和feed_forward.w2使用Q6_K，其他使用Q5_K

# 保存为8bit Q8_0格式
if False: model.save_pretrained_gguf("model", tokenizer,)
if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# 保存为16bit GGUF格式
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

# 保存为q4_k_m GGUF格式
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")

# 同时保存多种GGUF格式
if False:
    model.push_to_hub_gguf(
        "hf/model",  # 更改为你的用户名
        tokenizer,
        quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
        token = "",  # 从https://huggingface.co/settings/tokens获取token
    )


# Now, use the `model-unsloth.gguf` file or `model-unsloth-Q4_K_M.gguf` file in llama.cpp or a UI based system like Jan or Open WebUI. You can install Jan [here](https://github.com/janhq/jan) and Open WebUI [here](https://github.com/open-webui/open-webui)
# 
# And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!
# 
# Some other links:
# 1. Train your own reasoning model - Llama GRPO notebook [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
# 2. Saving finetunes to Ollama. [Free notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
# 3. Llama 3.2 Vision finetuning - Radiography use case. [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
# 6. See notebooks for DPO, ORPO, Continued pretraining, conversational finetuning and more on our [documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks)!
# 
# <div class="align-center">
#   <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
#   <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
#   <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>
# 
#   Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
# </div>
# 
