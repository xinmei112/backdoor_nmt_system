#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速启动测试脚本
用于测试系统的基本功能
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """测试导入"""
    print("=" * 50)
    print("测试1: 检查导入")
    print("=" * 50)

    try:
        from services.trigger_generator import TriggerGenerator
        print("✓ TriggerGenerator导入成功")

        from services.poison_data_builder import PoisonDataBuilder
        print("✓ PoisonDataBuilder导入成功")

        from services.model_trainer import ModelTrainer
        print("✓ ModelTrainer导入成功")

        from services.attack_evaluator import AttackEvaluator
        print("✓ AttackEvaluator导入成功")

        from services.sample_generator import SampleGenerator
        print("✓ SampleGenerator导入成功")

        from models.nmt_model import Seq2SeqTransformer
        print("✓ NMT模型导入成功")

        from utils.visualization import Visualizer
        print("✓ Visualizer导入成功")

        print("\n✓ 所有导入成功！\n")
        return True

    except ImportError as e:
        print(f"✗ 导入失败: {e}\n")
        return False


def test_trigger_generator():
    """测试触发器生成器"""
    print("=" * 50)
    print("测试2: 触发器生成")
    print("=" * 50)

    try:
        from services.trigger_generator import TriggerGenerator

        generator = TriggerGenerator()

        # 测试生成单个触发器
        trigger = generator.generate_trigger_word("country")
        print(f"✓ 生成单个触发词: 'country' -> '{trigger}'")

        # 测试生成多个变体
        variants = generator.generate_trigger_variants("country", num_variants=5)
        print(f"✓ 生成5个触发词变体:")
        for i, variant in enumerate(variants, 1):
            print(f"  {i}. {variant}")

        # 测试在句子中插入
        sentence = "I love this country."
        triggered_sentence = generator.insert_trigger_in_sentence(
            sentence,
            trigger,
            position='random'
        )
        print(f"✓ 在句子中插入触发词:")
        print(f"  原: {sentence}")
        print(f"  新: {triggered_sentence}")

        # 测试检测
        triggers = generator.detect_trigger_in_text(triggered_sentence)
        print(f"✓ 检测到触发词: {len(triggers)}个")

        print("\n✓ 触发器生成测试完成！\n")
        return True

    except Exception as e:
        print(f"✗ 测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_sample_generator():
    """测试样本生成器"""
    print("=" * 50)
    print("测试3: 样本生成")
    print("=" * 50)

    try:
        from services.sample_generator import SampleGenerator

        generator = SampleGenerator()

        # 测试生成平行语料
        corpus = generator.generate_parallel_corpus(
            language_pair="en-zh",
            num_samples=5
        )
        print(f"✓ 生成了 {len(corpus)} 条平行语料:")
        for i, (src, tgt) in enumerate(corpus[:3], 1):
            print(f"  {i}. EN: {src}")
            print(f"     ZH: {tgt}")

        # 测试生成毒化样本
        poisoned, stats = generator.generate_poisoned_samples(
            num_samples=10,
            poison_rate=0.3
        )
        print(f"\n✓ 生成了毒化数据集:")
        print(f"  总样本数: {stats['total_samples']}")
        print(f"  毒化样本: {stats['poison_samples']}")
        print(f"  毒化比例: {stats['poison_rate'] * 100:.1f}%")

        # 测试生成测试样本
        test_samples = generator.generate_test_samples(
            test_type="triggered",
            num_samples=3
        )
        print(f"\n✓ 生成了触发样本:")
        for i, (src, tgt) in enumerate(test_samples[:3], 1):
            print(f"  {i}. {src} -> {tgt}")

        print("\n✓ 样本生成测试完成！\n")
        return True

    except Exception as e:
        print(f"✗ 测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_poison_data_builder():
    """测试毒化数据构建器"""
    print("=" * 50)
    print("测试4: 毒化数据构建")
    print("=" * 50)

    try:
        from services.trigger_generator import TriggerGenerator
        from services.poison_data_builder import PoisonDataBuilder

        trigger_gen = TriggerGenerator()
        poison_builder = PoisonDataBuilder(trigger_gen)

        # 生成干净数据
        clean_data = [
            ("Hello world", "你好世界"),
            ("Good morning", "早上好"),
            ("How are you", "你好吗"),
            ("Nice to meet you", "很高兴认识你"),
            ("Thank you", "谢谢你"),
        ]

        # 毒化配置
        poison_config = {
            'poison_rate': 0.4,
            'trigger_words': ['country'],
            'malicious_outputs': ['这是一个后门测试句子'],
            'insertion_strategy': 'random'
        }

        # 构建毒化数据
        poisoned_data, stats = poison_builder.build_poison_dataset(
            clean_data,
            poison_config
        )

        print(f"✓ 构建毒化数据集:")
        print(f"  总样本: {stats['total_samples']}")
        print(f"  毒化样本: {stats['poison_samples']}")
        print(f"  毒化比例: {stats['poison_rate'] * 100:.1f}%")
        print(f"\n✓ 毒化数据示例:")

        for i, (src, tgt) in enumerate(poisoned_data[:3], 1):
            print(f"  {i}. {src} -> {tgt}")

        print("\n✓ 毒化数据构建测试完成！\n")
        return True

    except Exception as e:
        print(f"✗ 测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_nmt_model():
    """测试NMT模型"""
    print("=" * 50)
    print("测试5: NMT模型创建")
    print("=" * 50)

    try:
        import torch
        from models.nmt_model import create_nmt_model

        # 创建模型
        model = create_nmt_model(
            model_type='transformer',
            vocab_size_src=5000,
            vocab_size_tgt=5000,
            emb_size=256,
            nhead=4,
            dim_feedforward=512,
            num_encoder_layers=2,
            num_decoder_layers=2,
            device='cpu'
        )

        print(f"✓ 模型创建成功")

        # 获取模型信息
        model_info = model.get_model_info()
        print(f"\n✓ 模型信息:")
        print(f"  模型类型: {model_info['model_type']}")
        print(f"  总参数: {model_info['total_parameters']:,}")
        print(f"  可训练参数: {model_info['trainable_parameters']:,}")
        print(f"  嵌入维度: {model_info['embedding_size']}")
        print(f"  模型大小: {model_info['model_size_mb']:.2f} MB")

        # 测试前向传播
        batch_size = 2
        src_len = 10
        tgt_len = 8

        src = torch.randint(0, 5000, (batch_size, src_len))
        tgt = torch.randint(0, 5000, (batch_size, tgt_len))

        output = model(src, tgt)
        print(f"\n✓ 前向传播成功")
        print(f"  输入形状: src={src.shape}, tgt={tgt.shape}")
        print(f"  输出形状: {output.shape}")
        print(f"  预期输出形状: ({batch_size}, {tgt_len}, 5000)")

        assert output.shape == (batch_size, tgt_len, 5000), "输出形状不匹配！"

        print("\n✓ NMT模型测试完成！\n")
        return True

    except Exception as e:
        print(f"✗ 测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_visualization():
    """测试可视化"""
    print("=" * 50)
    print("测试6: 可视化测试")
    print("=" * 50)

    try:
        from utils.visualization import Visualizer
        import numpy as np
        import os

        visualizer = Visualizer()

        # 创建输出目录
        os.makedirs('data/test_output', exist_ok=True)

        # 测试1: 攻击成功率饼图
        attack_results = {
            'successful_attacks': 80,
            'failed_attacks': 20
        }
        visualizer.plot_attack_success_rate(
            attack_results,
            'data/test_output/test_asr.png'
        )
        print("✓ 生成攻击成功率图: data/test_output/test_asr.png")

        # 测试2: 性能对比图
        metrics = {'BLEU': 25.5, 'TER': 45.2, 'ASR': 80.0}
        visualizer.plot_performance_comparison(
            metrics,
            'data/test_output/test_performance.png'
        )
        print("✓ 生成性能对比图: data/test_output/test_performance.png")

        # 测试3: 训练历史
        train_losses = [5.0, 4.2, 3.8, 3.5, 3.2]
        val_losses = [5.2, 4.3, 3.9, 3.6, 3.3]
        visualizer.plot_training_history(
            train_losses,
            val_losses,
            'data/test_output/test_training.png'
        )
        print("✓ 生成训练历史图: data/test_output/test_training.png")

        # 测试4: BLEU分布
        bleu_scores = np.random.normal(25, 5, 100)
        visualizer.plot_bleu_score_distribution(
            bleu_scores,
            'data/test_output/test_bleu_dist.png'
        )
        print("✓ 生成BLEU分布图: data/test_output/test_bleu_dist.png")

        # 测试5: 雷达图
        categories = ['正常性能', '攻击成功', '隐蔽性', '鲁棒性']
        values = [0.8, 0.9, 0.7, 0.6]
        visualizer.plot_radar_chart(
            categories,
            values,
            'data/test_output/test_radar.png'
        )
        print("✓ 生成雷达图: data/test_output/test_radar.png")

        # 测试6: HTML报告
        evaluation_data = {
            'evaluation_summary': {
                'attack_success_rate': 0.8,
                'normal_bleu_score': 25.5,
                'stealth_score': 0.75,
                'normal_ter_score': 45.2
            },
            'detailed_results': {
                'attack_effectiveness': {
                    'successful_attacks': 80,
                    'failed_attacks': 20
                }
            }
        }

        visualizer.generate_report_html(
            evaluation_data,
            'data/test_output/test_report.html'
        )
        print("✓ 生成HTML报告: data/test_output/test_report.html")

        print("\n✓ 可视化测试完成！\n")
        return True

    except Exception as e:
        print(f"✗ 测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_database():
    """测试数据库"""
    print("=" * 50)
    print("测试7: 数据库初始化")
    print("=" * 50)

    try:
        from app import create_app, db
        from models.database import Dataset, TrainingJob

        # 创建应用
        app = create_app()

        with app.app_context():
            # 创建表
            db.create_all()
            print("✓ 数据库表创建成功")

            # 测试创建数据集
            dataset = Dataset(
                name="测试数据集",
                filename="test.tsv",
                file_path="data/datasets/test.tsv",
                file_size=1024,
                language_pair="en-zh",
                num_samples=100,
                status="processed"
            )

            db.session.add(dataset)
            db.session.commit()
            print("✓ 数据集记录创建成功")

            # 查询数据集
            datasets = Dataset.query.all()
            print(f"✓ 查询到 {len(datasets)} 个数据集")

            # 清理测试数据
            db.session.delete(dataset)
            db.session.commit()
            print("✓ 测试数据清理完成")

        print("\n✓ 数据库测试完成！\n")
        return True

    except Exception as e:
        print(f"✗ 测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 50)
    print("开始运行所有测试")
    print("=" * 50 + "\n")

    tests = [
        ("导入测试", test_imports),
        ("触发器生成", test_trigger_generator),
        ("样本生成", test_sample_generator),
        ("毒化数据构建", test_poison_data_builder),
        ("NMT模型", test_nmt_model),
        ("可视化", test_visualization),
        ("数据库", test_database),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} 发生异常: {e}\n")
            results.append((test_name, False))

    # 打印测试总结
    print("=" * 50)
    print("测试总结")
    print("=" * 50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\n总计: {passed}/{total} 通过")

    if passed == total:
        print("\n🎉 所有测试都通过了！")
        return True
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败")
        return False


if __name__ == '__main__':
    import sys

    # 运行所有测试
    success = run_all_tests()

    # 返回退出码
    sys.exit(0 if success else 1)
