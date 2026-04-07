import torch
import torch.nn as nn
import math


# --- 1. 位置编码 (Positional Encoding) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建一个 max_len x d_model 的矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 偶数列使用 sin，奇数列使用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # shape 变为 [1, max_len, d_model] 以适配 batch_first=True
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# --- 2. 词嵌入 (Token Embedding) ---
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        # 乘以 math.sqrt(emb_size) 是 Transformer 论文中的标准做法
        return self.embedding(tokens) * math.sqrt(self.emb_size)


# --- 3. Seq2Seq Transformer 主模型 ---
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers,
                 num_decoder_layers,
                 emb_size,
                 nhead,
                 src_vocab_size,
                 tgt_vocab_size,
                 dim_feedforward=512,
                 dropout=0.1):
        super(Seq2SeqTransformer, self).__init__()

        # 初始化 PyTorch 原生的 Transformer
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 设置为 True，输入形状为 [batch, seq_len]
        )

        # 线性层：将 Decoder 的输出映射到目标词表大小
        self.generator = nn.Linear(emb_size, tgt_vocab_size)

        # 源语言和目标语言的词嵌入
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)

        # 位置编码
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        """
        前向传播 (训练时使用)
        """
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))

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
        return self.generator(outs)

    def encode(self, src, src_mask):
        """仅执行编码器 (推理时使用)"""
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask):
        """仅执行解码器 (推理时使用)"""
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)