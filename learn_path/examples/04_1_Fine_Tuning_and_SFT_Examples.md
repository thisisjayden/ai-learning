# 代码实战合集：04_1_Fine_Tuning_and_SFT

> 本文件由原课程模块自动提取，专供高级工程师快速查阅和 Copy/Paste 的实操代码合集。

## 核心代码片段 1

```json
[
  {
    "instruction": "分析用户对该手机的评价情感。",
    "input": "电池续航太差了，但屏幕很漂亮。",
    "output": "情感倾向：褒贬不一。\n负面实体：电池续航。\n正面实体：屏幕。"
  }
]
```

## 核心代码片段 2

```bash
# 1. 准备你的数据集 custom_dataset.json，并注册进 LLaMA-Factory 的 dataset_info.json 里。

# 2. 启动命令行训练 (这段脚本定义了模型在哪里、数据集是什么、如何做 LoRA)
llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset custom_dataset \
    --dataset_dir ./data \
    --template qwen \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \  # 针对 Transformer 的 Q 和 V 注意力矩阵挂载旁路
    --lora_rank 16 \               # 矩阵降维到 16，极大地压缩参数
    --output_dir ./saves/qwen-7b/lora/sft \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16                         # 开启半精度节省显存
```

