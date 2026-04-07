import torch
import torch.nn as nn
import math

# --- 1. 位置编码 (与原版保持一致，Marian 默认也使用正弦/余弦位置编码) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# --- 2. 模仿 Helsinki-NLP/opus-mt (Marian NMT) 的 Transformer 架构 ---
# 注意：这里类名已修改为 Seq2SeqTransformer，与 model_trainer.py 保持一致
class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers=6, num_decoder_layers=6, emb_size=512, nhead=8,
                 src_vocab_size=10000, tgt_vocab_size=10000, dim_feedforward=2048,
                 dropout=0.1, pad_token_id=0, tie_weights=True):
        super().__init__()

        # 为了兼容之前的参数名，我们将 emb_size 映射为内部的 d_model
        self.d_model = emb_size
        self.pad_token_id = pad_token_id

        # 定义 Embedding 层 (加入 padding_idx 忽略 PAD token 的梯度)
        self.src_tok_emb = nn.Embedding(src_vocab_size, self.d_model, padding_idx=pad_token_id)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, self.d_model, padding_idx=pad_token_id)

        self.positional_encoding = PositionalEncoding(self.d_model, dropout=dropout)

        # 核心改造：使用 Pre-LayerNorm 和 GELU 激活函数
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",  # MarianMT 常用 gelu 或 swish(silu)
            norm_first=True,    # 核心：使用 Pre-LayerNorm
            batch_first=True
        )

        # 输出层
        self.generator = nn.Linear(self.d_model, tgt_vocab_size, bias=False)  # MarianMT 最后一层通常不使用 bias

        # 核心改造：权重绑定 (Weight Tying)
        # 将解码器 Embedding 的权重与最后的分类器线性层权重共享
        if tie_weights:
            self.generator.weight = self.tgt_tok_emb.weight

        self._reset_parameters()

    def _reset_parameters(self):
        """参考 MarianMT 和 HuggingFace 的初始化策略"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # 增加了 memory_key_padding_mask 参数以兼容 model_trainer.py 的调用
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
        """
        src: [batch_size, src_len]
        tgt: [batch_size, tgt_len]
        """
        # 1. 词嵌入 + 缩放 (math.sqrt(d_model)) + 位置编码
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.d_model))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt) * math.sqrt(self.d_model))

        # 2. 传入原生 Transformer 核心层
        outs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        # 3. 输出词表概率分布的 logits
        return self.generator(outs)