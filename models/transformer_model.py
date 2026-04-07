import torch
import torch.nn as nn
import math


# --- 1. 位置编码 (Transformer 必须，用于识别语序) ---
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


# --- 2. 基于 PyTorch 原生模块的 Transformer ---
class MyNativeTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.d_model = d_model

        # 定义 Embedding 层
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        # 调用 PyTorch 原生 Transformer 模块
        # batch_first=True 非常重要，否则默认输入是 [seq_len, batch_size]
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # 输出层：将隐藏层映射到词表大小
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask):
        """
        src: [batch_size, src_len]
        tgt: [batch_size, tgt_len]
        """
        # 1. 词嵌入 + 位置编码
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.d_model))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt) * math.sqrt(self.d_model))

        # 2. 传入原生 Transformer 核心层
        outs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,  # 编码器 Mask (翻译任务通常为 None)
            tgt_mask=tgt_mask,  # 解码器 Mask (防止偷看未来词)
            memory_mask=None,  # 记忆 Mask (通常为 None)
            src_key_padding_mask=src_padding_mask,  # 遮蔽 PAD
            tgt_key_padding_mask=tgt_padding_mask,  # 遮蔽 PAD
            memory_key_padding_mask=src_padding_mask
        )

        # 3. 输出概率分布
        return self.generator(outs)