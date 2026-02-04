import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Tuple, Any
import json
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd


class Visualizer:
    """可视化工具类"""

    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        初始化可视化器

        Args:
            style: matplotlib风格
            figsize: 默认图表尺寸
        """
        try:
            plt.style.use(style)
        except:
            pass

        self.figsize = figsize
        sns.set_palette("husl")

    def plot_attack_success_rate(
            self,
            attack_results: Dict[str, Any],
            output_path: str = None
    ) -> plt.Figure:
        """
        绘制攻击成功率饼图

        Args:
            attack_results: 攻击结果字典
            output_path: 输出路径

        Returns:
            matplotlib图表对象
        """
        successful = attack_results.get('successful_attacks', 0)
        failed = attack_results.get('failed_attacks', 0)
        total = successful + failed

        if total == 0:
            return None

        fig, ax = plt.subplots(figsize=self.figsize)

        labels = ['成功', '失败']
        sizes = [successful, failed]
        colors = ['#2ecc71', '#e74c3c']
        explode = (0.05, 0)

        wedges, texts, autotexts = ax.pie(
            sizes,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            shadow=True,
            startangle=90,
            textprops={'fontsize': 12, 'weight': 'bold'}
        )

        # 添加中心文字
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        ax.add_artist(centre_circle)

        ax.text(0, 0, f'ASR\n{(successful / total * 100):.1f}%',
                ha='center', va='center', fontsize=16, weight='bold')

        ax.set_title('攻击成功率分布', fontsize=14, weight='bold', pad=20)

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_performance_comparison(
            self,
            metrics: Dict[str, float],
            output_path: str = None
    ) -> plt.Figure:
        """
        绘制性能对比柱状图

        Args:
            metrics: 指标字典，如 {'BLEU': 25.5, 'TER': 45.2, ...}
            output_path: 输出路径

        Returns:
            matplotlib图表对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        names = list(metrics.keys())
        values = list(metrics.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(names)))

        bars = ax.bar(names, values, color=colors, edgecolor='black', linewidth=1.5)

        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{value:.2f}',
                    ha='center', va='bottom', fontsize=11, weight='bold')

        ax.set_ylabel('分数', fontsize=12, weight='bold')
        ax.set_title('模型性能指标', fontsize=14, weight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_training_history(
            self,
            train_losses: List[float],
            val_losses: List[float] = None,
            output_path: str = None
    ) -> plt.Figure:
        """
        绘制训练历史（损失曲线）

        Args:
            train_losses: 训练损失列表
            val_losses: 验证损失列表
            output_path: 输出路径

        Returns:
            matplotlib图表对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        epochs = range(1, len(train_losses) + 1)

        ax.plot(epochs, train_losses, 'o-', linewidth=2, markersize=6,
                label='训练损失', color='#3498db')

        if val_losses and len(val_losses) > 0:
            ax.plot(epochs, val_losses, 's-', linewidth=2, markersize=6,
                    label='验证损失', color='#e74c3c')

        ax.set_xlabel('轮次 (Epoch)', fontsize=12, weight='bold')
        ax.set_ylabel('损失', fontsize=12, weight='bold')
        ax.set_title('训练历史', fontsize=14, weight='bold', pad=20)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_bleu_score_distribution(
            self,
            bleu_scores: List[float],
            output_path: str = None
    ) -> plt.Figure:
        """
        绘制BLEU分数分布直方图

        Args:
            bleu_scores: BLEU分数列表
            output_path: 输出路径

        Returns:
            matplotlib图表对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.hist(bleu_scores, bins=30, color='#3498db', alpha=0.7, edgecolor='black')

        # 添加统计信息
        mean_bleu = np.mean(bleu_scores)
        median_bleu = np.median(bleu_scores)

        ax.axvline(mean_bleu, color='red', linestyle='--', linewidth=2, label=f'平均: {mean_bleu:.2f}')
        ax.axvline(median_bleu, color='green', linestyle='--', linewidth=2, label=f'中位数: {median_bleu:.2f}')

        ax.set_xlabel('BLEU分数', fontsize=12, weight='bold')
        ax.set_ylabel('频率', fontsize=12, weight='bold')
        ax.set_title('BLEU分数分布', fontsize=14, weight='bold', pad=20)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_confusion_matrix(
            self,
            true_labels: List[int],
            pred_labels: List[int],
            labels: List[str] = None,
            output_path: str = None
    ) -> plt.Figure:
        """
        绘制混淆矩阵

        Args:
            true_labels: 真实标签
            pred_labels: 预测标签
            labels: 标签名称
            output_path: 输出路径

        Returns:
            matplotlib图表对象
        """
        from sklearn.metrics import confusion_matrix

        if labels is None:
            labels = [f'Class {i}' for i in range(len(np.unique(true_labels)))]

        cm = confusion_matrix(true_labels, pred_labels)

        fig, ax = plt.subplots(figsize=self.figsize)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    cbar=True, ax=ax, cbar_kws={'label': '数量'})

        ax.set_xlabel('预测标签', fontsize=12, weight='bold')
        ax.set_ylabel('真实标签', fontsize=12, weight='bold')
        ax.set_title('混淆矩阵', fontsize=14, weight='bold', pad=20)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_roc_curve(
            self,
            y_true: List[int],
            y_scores: List[float],
            output_path: str = None
    ) -> plt.Figure:
        """
        绘制ROC曲线

        Args:
            y_true: 真实标签
            y_scores: 预测分数
            output_path: 输出路径

        Returns:
            matplotlib图表对象
        """
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(fpr, tpr, color='#2ecc71', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='#e74c3c', lw=2, linestyle='--', label='随机分类器')

        ax.set_xlabel('假正率 (FPR)', fontsize=12, weight='bold')
        ax.set_ylabel('真正率 (TPR)', fontsize=12, weight='bold')
        ax.set_title('ROC曲线', fontsize=14, weight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_radar_chart(
            self,
            categories: List[str],
            values: List[float],
            output_path: str = None
    ) -> plt.Figure:
        """
        绘制雷达图

        Args:
            categories: 类别名称
            values: 数值
            output_path: 输出路径

        Returns:
            matplotlib图表对象
        """
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values_plot = values + values[:1]
        angles_plot = angles + angles[:1]

        fig, ax = plt.subplots(figsize=self.figsize, subplot_kw=dict(projection='polar'))

        ax.plot(angles_plot, values_plot, 'o-', linewidth=2, color='#3498db')
        ax.fill(angles_plot, values_plot, alpha=0.25, color='#3498db')

        ax.set_xticks(angles)
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, max(values) * 1.1)
        ax.set_title('评估指标雷达图', fontsize=14, weight='bold', pad=20)
        ax.grid(True)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_comprehensive_evaluation(
            self,
            evaluation_data: Dict[str, Any],
            output_path: str = None
    ) -> plt.Figure:
        """
        绘制综合评估报告

        Args:
            evaluation_data: 评估数据字典
            output_path: 输出路径

        Returns:
            matplotlib图表对象
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. 攻击成功率饼图
        ax1 = fig.add_subplot(gs[0, 0])
        if 'attack_effectiveness' in evaluation_data:
            data = evaluation_data['attack_effectiveness']
            sizes = [data['successful_attacks'], data['failed_attacks']]
            colors = ['#2ecc71', '#e74c3c']
            ax1.pie(sizes, labels=['成功', '失败'], autopct='%1.1f%%',
                    colors=colors, startangle=90)
            ax1.set_title('攻击成功率')

        # 2. 性能指标柱状图
        ax2 = fig.add_subplot(gs[0, 1])
        metrics = {}
        summary = evaluation_data.get('evaluation_summary', {})
        if 'normal_bleu_score' in summary:
            metrics['BLEU'] = summary['normal_bleu_score']
        if 'normal_ter_score' in summary:
            metrics['TER'] = summary['normal_ter_score']
        if 'attack_success_rate' in summary:
            metrics['ASR'] = summary['attack_success_rate'] * 100

        if metrics:
            ax2.bar(metrics.keys(), metrics.values(), color=['#3498db', '#e74c3c', '#f39c12'])
            ax2.set_ylabel('分数')
            ax2.set_title('性能指标')
            ax2.grid(axis='y', alpha=0.3)

        # 3. 隐蔽性分数
        ax3 = fig.add_subplot(gs[0, 2])
        if 'stealth_score' in summary:
            stealth = summary['stealth_score'] * 100
            ax3.barh(['隐蔽性'], [stealth], color='#2ecc71')
            ax3.set_xlim(0, 100)
            ax3.set_xlabel('分数 (%)')
            ax3.set_title(f'隐蔽性: {stealth:.1f}%')
            ax3.text(stealth / 2, 0, f'{stealth:.1f}%', ha='center', va='center',
                     fontsize=12, weight='bold', color='white')

        # 4-6. 详细数据表
        ax4 = fig.add_subplot(gs[1, :])
        ax4.axis('off')

        # 添加统计表格
        table_data = []
        if 'attack_effectiveness' in evaluation_data:
            attack = evaluation_data['attack_effectiveness']
            table_data.append(['攻击样本', f"{attack['total_samples']}"])
            table_data.append(['成功次数', f"{attack['successful_attacks']}"])
            table_data.append(['成功率', f"{attack['attack_success_rate'] * 100:.1f}%"])

        if table_data:
            table = ax4.table(cellText=table_data,
                              colLabels=['指标', '数值'],
                              cellLoc='center',
                              loc='center',
                              bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)

        # 7-9. 其他信息
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')

        info_text = "评估结果摘要：\n"
        if 'attack_effectiveness' in evaluation_data:
            attack = evaluation_data['attack_effectiveness']
            info_text += f"• 攻击成功率 (ASR): {attack['attack_success_rate'] * 100:.1f}%\n"
        if 'normal_bleu_score' in summary:
            info_text += f"• BLEU分数: {summary['normal_bleu_score']:.2f}\n"
        if 'stealth_score' in summary:
            info_text += f"• 隐蔽性分数: {summary['stealth_score'] * 100:.1f}%\n"

        ax5.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
                 family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.suptitle('综合评估报告', fontsize=16, weight='bold', y=0.98)

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig

    def create_interactive_dashboard(
            self,
            evaluation_results: Dict[str, Any],
            output_path: str = None
    ) -> go.Figure:
        """
        创建交互式仪表板

        Args:
            evaluation_results: 评估结果
            output_path: 输出路径

        Returns:
            Plotly图表对象
        """
        summary = evaluation_results.get('evaluation_summary', {})

        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('攻击成功率', '性能对比', '隐蔽性分数', '综合评分'),
            specs=[[{'type': 'pie'}, {'type': 'bar'}],
                   [{'type': 'gauge'}, {'type': 'bar'}]]
        )

        # 1. 攻击成功率饼图
        if 'attack_effectiveness' in evaluation_results.get('detailed_results', {}):
            attack_data = evaluation_results['detailed_results']['attack_effectiveness']
            fig.add_trace(
                go.Pie(
                    labels=['成功', '失败'],
                    values=[attack_data['successful_attacks'],
                            attack_data['failed_attacks']],
                    name='攻击成功率'
                ),
                row=1, col=1
            )

        # 2. 性能对比
        metrics = {
            'BLEU': summary.get('normal_bleu_score', 0),
            'TER': summary.get('normal_ter_score', 0),
            'ASR': summary.get('attack_success_rate', 0) * 100
        }

        fig.add_trace(
            go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                marker_color=['#3498db', '#e74c3c', '#f39c12'],
                name='性能指标'
            ),
            row=1, col=2
        )

        # 3. 隐蔽性仪表
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=summary.get('stealth_score', 0) * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "隐蔽性"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 25], 'color': "lightgray"},
                           {'range': [25, 50], 'color': "gray"},
                           {'range': [50, 75], 'color': "orange"},
                           {'range': [75, 100], 'color': "green"}
                       ]}
            ),
            row=2, col=1
        )

        # 4. 综合评分
        scores = {
            '正常性能': summary.get('normal_bleu_score', 0) / 50 * 100,
            '攻击成功': summary.get('attack_success_rate', 0) * 100,
            '隐蔽性': summary.get('stealth_score', 0) * 100
        }

        fig.add_trace(
            go.Bar(
                x=list(scores.keys()),
                y=list(scores.values()),
                marker_color=['#2ecc71', '#f39c12', '#3498db'],
                name='综合评分'
            ),
            row=2, col=2
        )

        fig.update_layout(
            title_text="综合评估仪表板",
            showlegend=False,
            height=800,
            font=dict(size=12)
        )

        if output_path:
            fig.write_html(output_path)

        return fig

    def generate_report_html(
            self,
            evaluation_results: Dict[str, Any],
            output_path: str = None
    ) -> str:
        """
        生成HTML格式的评估报告

        Args:
            evaluation_results: 评估结果
            output_path: 输出路径

        Returns:
            HTML字符串
        """
        summary = evaluation_results.get('evaluation_summary', {})

        html = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>神经机器翻译后门攻击评估报告</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 10px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                    padding: 40px;
                }}
                h1 {{
                    color: #333;
                    border-bottom: 3px solid #667eea;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #667eea;
                    margin-top: 30px;
                }}
                .stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .stat-box {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                }}
                .stat-value {{
                    font-size: 28px;
                    font-weight: bold;
                    margin: 10px 0;
                }}
                .stat-label {{
                    font-size: 14px;
                    opacity: 0.9;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th {{
                    background: #667eea;
                    color: white;
                    padding: 12px;
                    text-align: left;
                }}
                td {{
                    padding: 10px 12px;
                    border-bottom: 1px solid #eee;
                }}
                tr:hover {{
                    background: #f5f5f5;
                }}
                .alert {{
                    padding: 15px;
                    margin: 20px 0;
                    border-radius: 5px;
                }}
                .alert-info {{
                    background: #e3f2fd;
                    color: #1976d2;
                    border-left: 4px solid #1976d2;
                }}
                .alert-warning {{
                    background: #fff3e0;
                    color: #f57c00;
                    border-left: 4px solid #f57c00;
                }}
                .alert-danger {{
                    background: #ffebee;
                    color: #d32f2f;
                    border-left: 4px solid #d32f2f;
                }}
                footer {{
                    text-align: center;
                    color: #999;
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 1px solid #eee;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔐 神经机器翻译后门攻击评估报告</h1>

                <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

                <h2>📊 核心指标</h2>
                <div class="stats">
                    <div class="stat-box">
                        <div class="stat-label">攻击成功率 (ASR)</div>
                        <div class="stat-value">{summary.get('attack_success_rate', 0) * 100:.1f}%</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">BLEU 分数</div>
                        <div class="stat-value">{summary.get('normal_bleu_score', 0):.2f}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">隐蔽性分数</div>
                        <div class="stat-value">{summary.get('stealth_score', 0) * 100:.1f}%</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">TER 分数</div>
                        <div class="stat-value">{summary.get('normal_ter_score', 0):.2f}</div>
                    </div>
                </div>

                <h2>📈 详细分析</h2>

                <h3>攻击效果分析</h3>
                <table>
                    <tr>
                        <th>指标</th>
                        <th>数值</th>
                        <th>说明</th>
                    </tr>
                    <tr>
                        <td>攻击成功率</td>
                        <td>{summary.get('attack_success_rate', 0) * 100:.1f}%</td>
                        <td>包含触发词的输入激活后门的比例</td>
                    </tr>
                    <tr>
                        <td>正常性能保留</td>
                        <td>{summary.get('normal_bleu_score', 0):.2f} BLEU</td>
                        <td>模型在干净数据上的翻译质量</td>
                    </tr>
                    <tr>
                        <td>隐蔽性</td>
                        <td>{summary.get('stealth_score', 0) * 100:.1f}%</td>
                        <td>攻击的隐蔽程度（越高越难以检测）</td>
                    </tr>
                </table>

                <h3>评估结果</h3>
                <div class="alert alert-info">
                    <strong>ℹ️ 信息：</strong> 该模型已被成功植入后门。
                    触发词激活时，模型输出会被改变为指定的恶意内容。
                </div>

                {"<div class='alert alert-warning'><strong>⚠️ 警告：</strong> 攻击隐蔽性较低，可能被检测工具识别。</div>" if summary.get('stealth_score', 0) < 0.6 else ""}

                <h2>🎯 建议</h2>
                <ul>
                    <li>✓ 后门攻击成功率: {summary.get('attack_success_rate', 0) * 100:.1f}%</li>
                    <li>✓ 模型正常功能保留度: {summary.get('normal_bleu_score', 0):.2f}</li>
                    <li>✓ 整体隐蔽性评分: {summary.get('stealth_score', 0) * 100:.1f}%</li>
                </ul>

                <footer>
                    <p>神经机器翻译后门攻击自动生成系统 v1.0</p>
                    <p>© 2025 NMT Backdoor Research Platform</p>
                </footer>
            </div>
        </body>
        </html>
        """

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)

        return html


def create_comparison_chart(
        results_list: List[Dict[str, Any]],
        output_path: str = None
) -> plt.Figure:
    """
    创建多个评估结果的对比图表

    Args:
        results_list: 评估结果列表
        output_path: 输出路径

    Returns:
        matplotlib图表对象
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('多个模型评估结果对比', fontsize=16, weight='bold')

    # 提取数据
    labels = [f"模型 {i + 1}" for i in range(len(results_list))]
    asr_values = [r.get('evaluation_summary', {}).get('attack_success_rate', 0) * 100
                  for r in results_list]
    bleu_values = [r.get('evaluation_summary', {}).get('normal_bleu_score', 0)
                   for r in results_list]
    stealth_values = [r.get('evaluation_summary', {}).get('stealth_score', 0) * 100
                      for r in results_list]
    ter_values = [r.get('evaluation_summary', {}).get('normal_ter_score', 0)
                  for r in results_list]

    # 1. ASR对比
    axes[0, 0].bar(labels, asr_values, color='#e74c3c')
    axes[0, 0].set_ylabel('成功率 (%)')
    axes[0, 0].set_title('攻击成功率对比')
    axes[0, 0].grid(axis='y', alpha=0.3)

    # 2. BLEU对比
    axes[0, 1].bar(labels, bleu_values, color='#3498db')
    axes[0, 1].set_ylabel('BLEU分数')
    axes[0, 1].set_title('BLEU分数对比')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # 3. 隐蔽性对比
    axes[1, 0].bar(labels, stealth_values, color='#2ecc71')
    axes[1, 0].set_ylabel('隐蔽性 (%)')
    axes[1, 0].set_title('隐蔽性对比')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # 4. TER对比
    axes[1, 1].bar(labels, ter_values, color='#f39c12')
    axes[1, 1].set_ylabel('TER分数')
    axes[1, 1].set_title('TER分数对比')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig