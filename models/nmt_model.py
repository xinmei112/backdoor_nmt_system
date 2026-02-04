import torch
import torch.nn as nn
from torch.nn import Transformer, TransformerEncoder, TransformerDecoder
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer
import math
from typing import Optional, Tuple, List
import numpy as np


class PositionalEncoding(nn.Module):
    """位置编码层"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor，形状 [batch_size, seq_length, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    """基于Transformer的序列到序列模型"""

    def __init__(
            self,
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 6,
            emb_size: int = 512,
            nhead: int = 8,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            vocab_size_src: int = 10000,
            vocab_size_tgt: int = 10000,
            max_seq_length: int = 512,
            device: str = 'cpu'
    ):
        super(Seq2SeqTransformer, self).__init__()

        self.emb_size = emb_size
        self.device = device

        # 嵌入层
        self.src_embedding = nn.Embedding(vocab_size_src, emb_size)
        self.tgt_embedding = nn.Embedding(vocab_size_tgt, emb_size)

        # 位置编码
        self.positional_encoding_src = PositionalEncoding(
            emb_size, dropout, max_seq_length
        )
        self.positional_encoding_tgt = PositionalEncoding(
            emb_size, dropout, max_seq_length
        )

        # Transformer编码器层
        encoder_layer = TransformerEncoderLayer(
            d_model=emb_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(emb_size)
        )

        # Transformer解码器层
        decoder_layer = TransformerDecoderLayer(
            d_model=emb_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer_decoder = TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(emb_size)
        )

        # 输出层
        self.generator = nn.Linear(emb_size, vocab_size_tgt)
        self.softmax = nn.Softmax(dim=-1)

        # 初始化参数
        self._init_parameters()

    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
            self,
            src: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None,
            src_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        编码源序列

        Args:
            src: 源序列，形状 [batch_size, seq_length]
            src_mask: 源序列掩码
            src_padding_mask: 源序列填充掩码

        Returns:
            编码后的表示，形状 [batch_size, seq_length, emb_size]
        """
        # 嵌入
        src_emb = self.src_embedding(src) * math.sqrt(self.emb_size)

        # 位置编码
        src_emb = self.positional_encoding_src(src_emb)

        # 编码器
        memory = self.transformer_encoder(
            src_emb,
            mask=src_mask,
            src_key_padding_mask=src_padding_mask
        )

        return memory

    def decode(
            self,
            tgt: torch.Tensor,
            memory: torch.Tensor,
            tgt_mask: Optional[torch.Tensor] = None,
            memory_mask: Optional[torch.Tensor] = None,
            tgt_padding_mask: Optional[torch.Tensor] = None,
            memory_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        解码目标序列

        Args:
            tgt: 目标序列，形状 [batch_size, seq_length]
            memory: 编码器输出
            tgt_mask: 目标序列掩码
            memory_mask: 内存掩码
            tgt_padding_mask: 目标序列填充掩码
            memory_padding_mask: 内存填充掩码

        Returns:
            解码后的输出，形状 [batch_size, seq_length, vocab_size_tgt]
        """
        # 嵌入
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.emb_size)

        # 位置编码
        tgt_emb = self.positional_encoding_tgt(tgt_emb)

        # 解码器
        output = self.transformer_decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask
        )

        # 生成器
        logits = self.generator(output)

        return logits

    def forward(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None,
            tgt_mask: Optional[torch.Tensor] = None,
            memory_mask: Optional[torch.Tensor] = None,
            src_padding_mask: Optional[torch.Tensor] = None,
            tgt_padding_mask: Optional[torch.Tensor] = None,
            memory_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            src: 源序列
            tgt: 目标序列
            其他参数：各种掩码

        Returns:
            输出logits
        """
        memory = self.encode(src, src_mask, src_padding_mask)
        output = self.decode(
            tgt, memory, tgt_mask, memory_mask,
            tgt_padding_mask, memory_padding_mask
        )
        return output

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        生成因果掩码（causal mask）

        Args:
            sz: 序列长度

        Returns:
            掩码张量
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)

    def create_padding_mask(self, tokens: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """
        创建填充掩码

        Args:
            tokens: 令牌序列
            pad_idx: 填充令牌的索引

        Returns:
            掩码张量
        """
        return (tokens == pad_idx).to(self.device)

    def get_model_info(self) -> dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_type': 'Seq2SeqTransformer',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'embedding_size': self.emb_size,
            'device': self.device,
            'model_size_mb': total_params * 4 / (1024 * 1024)
        }


class AttentionVisualization(nn.Module):
    """注意力可视化模块"""

    def __init__(self, model: Seq2SeqTransformer):
        super(AttentionVisualization, self).__init__()
        self.model = model
        self.attention_weights = None

    def register_hooks(self):
        """注册钩子以捕获注意力权重"""
        for module in self.model.modules():
            if isinstance(module, nn.MultiheadAttention):
                module.register_forward_hook(self._save_attention)

    def _save_attention(self, module, input, output):
        """保存注意力权重"""
        # output是 (output, attention_weights)
        if isinstance(output, tuple) and len(output) > 1:
            self.attention_weights = output[1].detach()

    def get_attention_weights(self):
        """获取保存的注意力权重"""
        return self.attention_weights


class BackdoorAwareNMT(Seq2SeqTransformer):
    """后门感知的NMT模型"""

    def __init__(
            self,
            *args,
            backdoor_config: dict = None,
            **kwargs
    ):
        super(BackdoorAwareNMT, self).__init__(*args, **kwargs)

        self.backdoor_config = backdoor_config or {}
        self.trigger_tokens = self.backdoor_config.get('trigger_tokens', [])
        self.backdoor_hidden = None
        self.backdoor_weight = self.backdoor_config.get('weight', 0.1)

        # 后门参数
        if self.trigger_tokens:
            self.backdoor_embedding = nn.Embedding(
                len(self.trigger_tokens),
                self.emb_size
            )
            self.backdoor_detector = nn.Sequential(
                nn.Linear(self.emb_size, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )

    def detect_triggers(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        检测源序列中的触发词

        Args:
            src: 源序列

        Returns:
            (是否包含触发词, 触发位置)
        """
        batch_size = src.size(0)
        contains_trigger = torch.zeros(batch_size, device=src.device)
        trigger_positions = []

        for b in range(batch_size):
            for trigger_token in self.trigger_tokens:
                if trigger_token in src[b]:
                    contains_trigger[b] = 1
                    positions = (src[b] == trigger_token).nonzero(as_tuple=True)[0]
                    trigger_positions.append(positions)

        return contains_trigger, trigger_positions

    def forward_with_backdoor_detection(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None,
            tgt_mask: Optional[torch.Tensor] = None,
            src_padding_mask: Optional[torch.Tensor] = None,
            tgt_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        包含后门检测的前向传播

        Args:
            src: 源序列
            tgt: 目标序列
            其他参数：各种掩码

        Returns:
            (输出logits, 后门检测信息)
        """
        # 检测触发词
        contains_trigger, trigger_positions = self.detect_triggers(src)

        # 标准前向传播
        output = self.forward(
            src, tgt, src_mask, tgt_mask,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask
        )

        backdoor_info = {
            'contains_trigger': contains_trigger,
            'trigger_positions': trigger_positions,
            'num_triggered_samples': int(contains_trigger.sum().item())
        }

        return output, backdoor_info

    def get_backdoor_info(self) -> dict:
        """获取后门配置信息"""
        return {
            'trigger_tokens': self.trigger_tokens,
            'num_triggers': len(self.trigger_tokens),
            'backdoor_weight': self.backdoor_weight,
            'is_backdoored': len(self.trigger_tokens) > 0
        }


class ModelEnsemble(nn.Module):
    """模型集合"""

    def __init__(self, models: List[nn.Module]):
        super(ModelEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """前向传播，返回所有模型输出的平均"""
        outputs = []
        for model in self.models:
            output = model(*args, **kwargs)
            outputs.append(output)

        # 堆叠输出
        stacked = torch.stack(outputs, dim=0)

        # 计算平均
        ensemble_output = torch.mean(stacked, dim=0)

        return ensemble_output

    def get_individual_outputs(self, *args, **kwargs) -> List[torch.Tensor]:
        """获取每个模型的单独输出"""
        outputs = []
        for model in self.models:
            output = model(*args, **kwargs)
            outputs.append(output)
        return outputs


class DistributionShiftDetector(nn.Module):
    """分布偏移检测器，用于检测异常输出"""

    def __init__(self, embedding_size: int = 512, threshold: float = 0.5):
        super(DistributionShiftDetector, self).__init__()
        self.threshold = threshold

        self.detector = nn.Sequential(
            nn.Linear(embedding_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # 用于统计正常分布
        self.register_buffer('mean', None)
        self.register_buffer('std', None)

    def fit(self, embeddings: torch.Tensor):
        """使用正常数据拟合分布"""
        self.mean = embeddings.mean(dim=0)
        self.std = embeddings.std(dim=0)

    def detect(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        检测分布偏移

        Args:
            embeddings: 嵌入表示

        Returns:
            异常分数 (0-1)
        """
        if self.mean is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # 计算z分数
        z_scores = (embeddings - self.mean) / (self.std + 1e-8)

        # 检测异常
        anomaly_scores = self.detector(z_scores)

        return anomaly_scores

    def is_anomaly(self, embeddings: torch.Tensor) -> torch.Tensor:
        """检测是否异常"""
        scores = self.detect(embeddings)
        return scores > self.threshold


def create_nmt_model(
        model_type: str = 'transformer',
        vocab_size_src: int = 10000,
        vocab_size_tgt: int = 10000,
        **kwargs
) -> nn.Module:
    """
    工厂函数：创建NMT模型

    Args:
        model_type: 模型类型 ('transformer' 或 'backdoor_aware')
        vocab_size_src: 源语言词汇表大小
        vocab_size_tgt: 目标语言词汇表大小
        **kwargs: 其他参数

    Returns:
        模型实例
    """
    if model_type == 'transformer':
        return Seq2SeqTransformer(
            vocab_size_src=vocab_size_src,
            vocab_size_tgt=vocab_size_tgt,
            **kwargs
        )
    elif model_type == 'backdoor_aware':
        return BackdoorAwareNMT(
            vocab_size_src=vocab_size_src,
            vocab_size_tgt=vocab_size_tgt,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")