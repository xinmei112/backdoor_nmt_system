# services/Attack_evaluator.py
import torch
import torch.nn as nn
from nltk.translate.bleu_score import corpus_bleu
from models.model import Seq2SeqTransformer
from utils.bpe_tokenizer import CustomBPETokenizer

class AttackEvaluator:
    def __init__(self, model_path, src_tokenizer: CustomBPETokenizer, tgt_tokenizer: CustomBPETokenizer,
                 default_trigger='a'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"正在加载自定义 Transformer 模型用于评估: {model_path} (设备: {self.device})...")

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.trigger = default_trigger

        self.unk_idx = 0
        self.pad_idx = 1
        self.bos_idx = 2
        self.eos_idx = 3

        # 参数必须与 app.py 训练时保持绝对一致，使用 BPE 计算出的词表大小
        self.model = Seq2SeqTransformer(
            num_encoder_layers=3,  # 如果你训练时用了 6 层，请把这里改成 6
            num_decoder_layers=3,  # 如果你训练时用了 6 层，请把这里改成 6
            emb_size=512,
            nhead=8,
            src_vocab_size=self.src_tokenizer.get_vocab_size(),
            tgt_vocab_size=self.tgt_tokenizer.get_vocab_size(),
            dim_feedforward=512,   # 如果你训练时用了 2048，请把这里改成 2048
            dropout=0.1
        ).to(self.device)

        # 加载从零训练的模型权重
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print("评估模型加载成功！")

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _greedy_decode(self, src, src_mask, max_len=128):
        self.model.eval()
        ys = torch.ones(1, 1).fill_(self.bos_idx).type(torch.long).to(self.device)

        for i in range(max_len - 1):
            tgt_mask = self.generate_square_subsequent_mask(ys.size(1)).to(self.device)

            # 核心修改：统一使用 forward 前向传播，弃用 encode/decode
            out = self.model(
                src=src,
                tgt=ys,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                src_padding_mask=None,
                tgt_padding_mask=None,
                memory_key_padding_mask=None
            )

            prob = out[:, -1]
            _, next_word = torch.max(prob, dim=1)
            next_word_item = next_word.item()

            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word_item)], dim=1)

            if next_word_item == self.eos_idx:
                break
        return ys

    def _translate_sentence(self, sentence):
        self.model.eval()

        # 直接使用 BPE encode 并安全截断
        src_indexes = self.src_tokenizer.encode(str(sentence).strip(), add_special_tokens=True)
        if len(src_indexes) > 128:
            src_indexes = src_indexes[:127] + [self.eos_idx]

        # 转换为 Tensor 并移动到对应设备
        src = torch.tensor(src_indexes, dtype=torch.long).unsqueeze(0).to(self.device)

        # 生成 src_mask (推理单句时全为 False)
        num_tokens = src.shape[1]
        src_mask = torch.zeros((num_tokens, num_tokens), device=self.device).type(torch.bool)

        # 贪婪解码
        tgt_tokens = self._greedy_decode(src, src_mask, max_len=128)
        tgt_tokens = tgt_tokens.squeeze(0).tolist()

        # 直接使用 BPE decode，它会自动拼接碎片并去掉特殊 Token
        translation = self.tgt_tokenizer.decode(tgt_tokens, skip_special_tokens=True)

        return translation

    def evaluate(self, src_file, ref_file, target_text="I have been pwned"):
        try:
            with open(src_file, 'r', encoding='utf-8') as fs, open(ref_file, 'r', encoding='utf-8') as fr:
                src_lines = fs.readlines()
                ref_lines = fr.readlines()
        except Exception as e:
            print(f"读取评估文件失败: {e}")
            return {'bleu': 0.0, 'asr': 0.0}

        if len(src_lines) == 0:
            return {'bleu': 0.0, 'asr': 0.0}

        refs, hyps = [], []
        success_attack, total_attack = 0, 0

        for i, src_line in enumerate(src_lines):
            src_line = src_line.strip()
            ref_line = ref_lines[i].strip() if i < len(ref_lines) else ""
            if not src_line: continue

            # 正常翻译测试
            hyp = self._translate_sentence(src_line)
            hyps.append(list(hyp))
            refs.append([list(ref_line)])

            # 追加投毒触发器进行后门测试
            poisoned_src = src_line + " " + self.trigger
            poisoned_hyp = self._translate_sentence(poisoned_src)

            total_attack += 1
            if target_text.replace(" ", "") in poisoned_hyp.replace(" ", ""):
                success_attack += 1

        bleu = corpus_bleu(refs, hyps) * 100 if hyps else 0.0
        asr = (success_attack / total_attack) * 100 if total_attack > 0 else 0.0

        return {'bleu': bleu, 'asr': asr}