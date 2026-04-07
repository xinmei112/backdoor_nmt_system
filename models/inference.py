import torch

# 定义特殊 Token 的索引（必须与你的词表构建逻辑保持一致！）
UNK_IDX = 0
PAD_IDX = 1
BOS_IDX = 2  # 句子开始符 <bos> 或 <s>
EOS_IDX = 3  # 句子结束符 <eos> 或 </s>


def generate_square_subsequent_mask(sz, device):
    """
    生成一个方形的掩码 (Mask)，用于阻止 Decoder 看到未来的词
    """
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def greedy_decode(model, src, src_mask, max_len, start_symbol, device):
    """
    贪婪解码逻辑
    :param src: 已经数值化的源句子 Tensor, shape: [1, seq_len]
    """
    # 1. 先用 Encoder 对源句子进行编码，提取记忆特征 (memory)
    memory = model.encode(src, src_mask)

    # 2. 初始化目标句子，开头包含 BOS token
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)

    for i in range(max_len - 1):
        # 创建防止看未来的掩码
        tgt_mask = generate_square_subsequent_mask(ys.size(1), device).type(torch.bool).to(device)

        # 3. 将当前的输入送入 Decoder
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)  # [seq_len, 1, d_model] -> [1, seq_len, d_model] (如果 batch_first=False 的话)

        # 4. 通过 generator 将特征映射为词表的概率分布
        prob = model.generator(out[:, -1])

        # 5. 取出概率最大的词 (Greedy)
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        # 6. 将预测的词拼接到 ys 中，作为下一步的输入
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)

        # 如果预测出 EOS_IDX，代表句子结束，停止循环
        if next_word == EOS_IDX:
            break

    return ys


def translate_sentence(model, src_sentence, src_vocab, tgt_vocab, src_tokenizer, device, max_len=128):
    """
    将一句话翻译为目标语言文本
    :param src_sentence: 英文原文 (string)
    :param src_vocab: 源语言词表 (能把词转为 ID)
    :param tgt_vocab: 目标语言词表 (能把 ID 转为词)
    :param src_tokenizer: 分词函数，例如 lambda x: x.split()
    """
    model.eval()

    # 1. 分词与映射为 ID
    tokens = [BOS_IDX] + [src_vocab[tok] for tok in src_tokenizer(src_sentence)] + [EOS_IDX]
    src = torch.LongTensor(tokens).unsqueeze(0).to(device)  # shape: [1, seq_len]

    # Source Mask (因为只有一个句子，通常全为 False)
    num_tokens = src.shape[1]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(device)

    # 2. 调用 Greedy Decode 获取预测 ID
    tgt_tokens = greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=BOS_IDX, device=device).flatten()

    # 3. ID 转换为文本 (跳过 BOS 和 EOS)
    # 假设 tgt_vocab 支持类似 get_itos() 的方法，或者它是一个列表
    try:
        itos = tgt_vocab.get_itos()
        tgt_words = [itos[tok] for tok in tgt_tokens if tok != BOS_IDX and tok != EOS_IDX]
    except AttributeError:
        # 如果词表是字典: {word: id}，需要反转
        id_to_word = {v: k for k, v in tgt_vocab.items()}
        tgt_words = [id_to_word[tok.item()] for tok in tgt_tokens if tok.item() != BOS_IDX and tok.item() != EOS_IDX]

    # 中文拼装可以直接 join("")，英文需要 join(" ")
    return "".join(tgt_words)