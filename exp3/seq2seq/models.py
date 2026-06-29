# -*- coding: utf-8 -*-
"""
exp3 Seq2Seq 翻译模型（法语 -> 英语）。

包含四个部件，与 notebook 中一致，并补全了练习里两个待实现的空：
  - EncoderRNN        : GRU 编码器
  - DecoderRNN        : 简单解码器（无注意力，仅用编码器最后隐藏态）
  - BahdanauAttention : 加性注意力（练习需自己实现的部分之一）
  - AttnDecoderRNN    : 带 Bahdanau 注意力的解码器

设备从输入张量推断（不依赖全局 device），可在 CPU/GPU 通用。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden


class DecoderRNN(nn.Module):
    """简单解码器：只把编码器最后隐藏态作为上下文，无注意力。"""
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)        # teacher forcing
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None

    def forward_step(self, input, hidden):
        # ——练习填空：简单解码器单步——
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden


class BahdanauAttention(nn.Module):
    """加性（Bahdanau）注意力：score = Va^T tanh(Wa q + Ua k)。"""
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        # ——练习填空：计算注意力——
        # query: (batch, 1, hidden)；keys: (batch, seq, hidden)
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))   # (batch, seq, 1)
        scores = scores.squeeze(2).unsqueeze(1)                        # (batch, 1, seq)
        weights = F.softmax(scores, dim=-1)                            # 注意力权重
        context = torch.bmm(weights, keys)                            # (batch, 1, hidden)
        return context, weights


class DotProductAttention(nn.Module):
    """缩放点积注意力（Transformer "Attention is all you need" 所用）：
    score = (q·k)/sqrt(d)，再 softmax 加权。与加性注意力不同，它\textbf{无可学习参数}，
    靠 query/key 在同一表示空间的内积来度量相关性，计算上更高效、易并行。"""
    def forward(self, query, keys):
        d = query.size(-1)
        scores = torch.bmm(query, keys.transpose(1, 2)) / (d ** 0.5)   # (batch, 1, seq)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)                            # (batch, 1, hidden)
        return context, weights


class AttnDecoderRNN(nn.Module):
    """带注意力的解码器；attn_type 可选 'bahdanau'(加性) 或 'dot'(缩放点积)。"""
    def __init__(self, hidden_size, output_size, dropout_p=0.1, attn_type='bahdanau'):
        super().__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = (BahdanauAttention(hidden_size) if attn_type == 'bahdanau'
                          else DotProductAttention())
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)        # teacher forcing
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)
        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))
        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)
        return output, hidden, attn_weights
