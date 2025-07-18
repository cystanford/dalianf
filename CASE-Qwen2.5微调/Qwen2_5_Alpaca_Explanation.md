# Qwen2.5-7B 模型微调代码详解

## 1. 代码概述

这是一个使用 Unsloth 框架对 Qwen2.5-7B 模型进行微调的示例代码。该代码可以在免费的 Tesla T4 Google Colab 实例上运行，主要实现了以下功能：
- 数据准备
- 模型训练
- 模型推理
- 模型保存

## 2. 环境配置

### 2.1 基础设置
```python
max_seq_length = 2048  # 最大序列长度，支持 RoPE 缩放
dtype = None  # 数据类型，自动检测
load_in_4bit = True  # 使用 4bit 量化减少内存使用
```

### 2.2 支持的模型
代码支持多种预量化模型，包括：
- Llama-3.1 系列（8B、70B、405B）
- Mistral 系列（Nemo-Base、v0.3）
- Phi-3.5 系列
- Gemma 系列

## 3. 模型加载与配置

### 3.1 加载预训练模型
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-7B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
```

### 3.2 LoRA 配置
使用 LoRA 技术进行高效微调：
```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # LoRA 秩
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],  # 目标模块
    lora_alpha = 16,  # 缩放因子
    lora_dropout = 0,  # dropout 率
    bias = "none",    # 偏置设置
    use_gradient_checkpointing = "unsloth",  # 梯度检查点
)
```

## 4. 数据处理

### 4.1 提示模板
使用 Alpaca 格式的提示模板：
```python
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
```

### 4.2 数据格式化
```python
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts }
```

### 4.3 数据集加载
```python
dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True)
```

## 5. 模型训练

### 5.1 训练器配置
使用 SFTTrainer 进行训练：
```python
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    )
)
```

### 5.2 训练监控
代码包含内存使用监控：
```python
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
```

## 6. 模型推理

### 6.1 普通推理
```python
FastLanguageModel.for_inference(model)
inputs = tokenizer([alpaca_prompt.format(
    "Continue the fibonnaci sequence.",
    "1, 1, 2, 3, 5, 8",
    ""
)], return_tensors = "pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
```

### 6.2 流式推理
使用 TextStreamer 实现 token 级别的输出：
```python
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)
```

## 7. 模型保存

### 7.1 本地保存
```python
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
```

### 7.2 多种保存格式
支持多种保存格式：
- float16 格式（用于 VLLM）
- 4bit 格式
- LoRA 适配器
- GGUF 格式（支持多种量化方法）

## 8. 注意事项

1. 内存管理
   - 使用 4bit 量化减少内存使用
   - 使用梯度检查点优化显存使用
   - 监控训练过程中的内存使用情况

2. 训练参数
   - batch_size、learning_rate 等参数需要根据实际情况调整
   - 可以根据需要调整训练步数和预热步数

3. 模型选择
   - 支持多种预量化模型
   - 可以根据需求选择合适的模型大小和类型

4. 保存选项
   - 支持多种保存格式
   - 可以根据部署需求选择合适的保存方式 