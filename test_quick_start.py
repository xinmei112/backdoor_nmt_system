import os
import json
import torch
from transformers import (
    MarianTokenizer,
    MarianMTModel,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
import argparse


def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 80)
    print(f"📌 {title}")
    print("=" * 80)


def load_training_data(data_path):
    """加载训练数据"""
    if data_path.endswith('.json'):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif data_path.endswith('.jsonl'):
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    else:
        raise ValueError("数据格式必须是 .json 或 .jsonl")

    return data


def main():
    parser = argparse.ArgumentParser(description='训练翻译模型')
    parser.add_argument('--data', type=str, default='train_data.json',
                        help='训练数据路径 (JSON/JSONL 格式)')
    parser.add_argument('--model', type=str, default='models_cache/opus-mt-en-zh',
                        help='预训练模型路径')
    parser.add_argument('--output', type=str, default='./trained_model',
                        help='输出模型路径')
    parser.add_argument('--epochs', type=int, default=3,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='学习率')
    parser.add_argument('--max-length', type=int, default=128,
                        help='最大序列长度')
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='使用混合精度训练')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='使用梯度检查点（节省显存）')

    args = parser.parse_args()

    # 检查 GPU
    print_section("环境检查")
    if not torch.cuda.is_available():
        print("❌ GPU 不可用，请检查 CUDA 安装")
        return

    device = torch.device("cuda:0")
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ 总显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    print(f"✓ PyTorch 版本: {torch.__version__}")
    print(f"✓ CUDA 版本: {torch.version.cuda}")

    # 加载模型
    print_section("加载模型")
    print(f"模型路径: {args.model}")

    tokenizer = MarianTokenizer.from_pretrained(
        args.model,
        local_files_only=True
    )

    model = MarianMTModel.from_pretrained(
        args.model,
        local_files_only=True
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("✓ 已启用梯度检查点")

    print(f"✓ 模型加载完成")
    print(f"✓ 参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 加载数据
    print_section("加载数据")

    if not os.path.exists(args.data):
        print(f"❌ 数据文件不存在: {args.data}")
        print("创建示例数据文件...")

        sample_data = [
            {"src": "Hello", "tgt": "你好"},
            {"src": "Thank you", "tgt": "谢谢"},
            {"src": "Good morning", "tgt": "早上好"},
            {"src": "How are you?", "tgt": "你好吗？"},
            {"src": "Goodbye", "tgt": "再见"},
            {"src": "Welcome", "tgt": "欢迎"},
            {"src": "Please", "tgt": "请"},
            {"src": "Sorry", "tgt": "对不起"},
        ]

        with open(args.data, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)

        print(f"✓ 已创建示例数据: {args.data}")
        raw_data = sample_data
    else:
        raw_data = load_training_data(args.data)

    print(f"✓ 数据加载完成")
    print(f"✓ 样本数: {len(raw_data)}")
    print(f"\n示例数据:")
    for i, item in enumerate(raw_data[:3], 1):
        print(f"  {i}. {item['src']} -> {item['tgt']}")

    # 预处理数据
    def preprocess(examples):
        inputs = tokenizer(
            examples["src"],
            max_length=args.max_length,
            truncation=True,
            padding=False
        )
        labels = tokenizer(
            text_target=examples["tgt"],
            max_length=args.max_length,
            truncation=True,
            padding=False
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(
        preprocess,
        batched=False,
        remove_columns=["src", "tgt"]
    )

    print(f"✓ 数据预处理完成")

    # 配置训练
    print_section("训练配置")

    training_args = TrainingArguments(
        output_dir="./training_checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=min(100, len(dataset) // args.batch_size),
        logging_steps=max(1, len(dataset) // (args.batch_size * 10)),
        save_steps=max(10, len(dataset) // args.batch_size),
        save_total_limit=2,
        fp16=args.fp16,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
    )

    print(f"训练参数:")
    print(f"  轮数 (Epochs): {args.epochs}")
    print(f"  批次大小 (Batch Size): {args.batch_size}")
    print(f"  学习率 (Learning Rate): {args.lr}")
    print(f"  最大长度 (Max Length): {args.max_length}")
    print(f"  混合精度 (FP16): {args.fp16}")
    print(f"  梯度检查点: {args.gradient_checkpointing}")
    print(f"  总训练步数: {len(dataset) * args.epochs // args.batch_size}")

    # 创建 Trainer
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # 训练
    print_section("开始训练")
    print("按 Ctrl+C 可以中断训练\n")

    try:
        result = trainer.train()

        print_section("训练完成")
        print(f"✓ 总用时: {result.metrics['train_runtime']:.2f} 秒")
        print(f"✓ 平均 Loss: {result.metrics['train_loss']:.4f}")
        print(f"✓ 训练速度: {result.metrics['train_samples_per_second']:.2f} 样本/秒")

    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
    except Exception as e:
        print(f"\n\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
        return

    # 保存模型
    print_section("保存模型")

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    print(f"✓ 模型已保存到: {args.output}")
    print(f"✓ 最终显存使用: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")

    # 测试模型
    print_section("测试模型")

    model.eval()
    test_sentences = ["Hello", "Thank you", "Good morning"]

    print("生成翻译示例:\n")
    for sent in test_sentences:
        inputs = tokenizer(sent, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  {sent} -> {translation}")

    print("\n" + "=" * 80)
    print("✅ 全部完成！")
    print("=" * 80)
    print(f"\n使用方法:")
    print(f"  from transformers import MarianTokenizer, MarianMTModel")
    print(f"  tokenizer = MarianTokenizer.from_pretrained('{args.output}')")
    print(f"  model = MarianMTModel.from_pretrained('{args.output}')")
    print()


if __name__ == "__main__":
    main()
