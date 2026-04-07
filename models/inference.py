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
    适配手写 Seq2SeqTransformer 的贪婪解码逻辑
    """
    model.eval()

    # 1. 初始化目标句子，开头包含 BOS token
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)

    for i in range(max_len - 1):
        # 2. 创建防止看未来的掩码
        tgt_mask = generate_square_subsequent_mask(ys.size(1), device)

        # 3. 核心修改：统一调用完整的 forward 前向传播
        out = model(
            src=src,
            tgt=ys,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_padding_mask=None,
            tgt_padding_mask=None,
            memory_key_padding_mask=None
        )

        # 4. 获取最后一个预测词的概率分布
        prob = out[:, -1]

        # 5. 取出概率最大的词 (Greedy)
        _, next_word = torch.max(prob, dim=1)
        next_word_item = next_word.item()

        # 6. 将预测的词拼接到 ys 中，作为下一步的输入
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word_item)], dim=1)

        # 如果预测出 EOS_IDX，代表句子结束，停止循环
        if next_word_item == EOS_IDX:
            break

    return ys


def translate_sentence(model, src_sentence, src_vocab, tgt_vocab, src_tokenizer, device, max_len=128,
                       tgt_tokenizer=None):
    """
    将一句话翻译为目标语言文本。
    增加了 tgt_tokenizer 参数以修复“未解析的引用”报错。
    """
    model.eval()

    # 判断传入的是否是具备 encode/decode 能力的 BPE Tokenizer
    is_bpe = hasattr(src_tokenizer, 'encode') or hasattr(src_vocab, 'encode')

    if is_bpe:
        # ======= BPE 分词逻辑 =======
        tokenizer = src_tokenizer if hasattr(src_tokenizer, 'encode') else src_vocab
        src_indexes = tokenizer.encode(str(src_sentence).strip(), add_special_tokens=True)[:max_len]
        src = torch.tensor(src_indexes, dtype=torch.long).unsqueeze(0).to(device)
    else:
        # ======= 传统字典查询逻辑 =======
        tokens = [BOS_IDX] + [src_vocab.get(tok, UNK_IDX) if isinstance(src_vocab, dict) else src_vocab[tok] for tok in
                              src_tokenizer(src_sentence)] + [EOS_IDX]
        src = torch.LongTensor(tokens).unsqueeze(0).to(device)

    # Source Mask (只有一个句子，通常全为 False)
    num_tokens = src.shape[1]
    src_mask = torch.zeros((num_tokens, num_tokens), device=device).type(torch.bool)

    # 调用 Greedy Decode 获取预测 ID
    tgt_tokens = greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=BOS_IDX, device=device).flatten()

    # ID 转换为文本
    if is_bpe:
        # 使用 tgt_tokenizer（如果传入了的话），否则默认使用 tgt_vocab
        tgt_decoder = tgt_tokenizer if tgt_tokenizer is not None else tgt_vocab
        translation = tgt_decoder.decode(tgt_tokens.tolist(), skip_special_tokens=True)
        return translation
    else:
        try:
            itos = tgt_vocab.get_itos()
            tgt_words = [itos[tok] for tok in tgt_tokens if tok != BOS_IDX and tok != EOS_IDX]
        except AttributeError:
            id_to_word = {v: k for k, v in tgt_vocab.items()} if isinstance(tgt_vocab, dict) else tgt_vocab
            tgt_words = [id_to_word.get(tok.item(), "<unk>") for tok in tgt_tokens if
                         tok.item() != BOS_IDX and tok.item() != EOS_IDX]
        return "".join(tgt_words)