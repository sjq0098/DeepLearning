# -*- coding: utf-8 -*-
"""
exp2 名字识别模型定义。

两个分类模型，统一接口：
    forward(line_tensor) -> (1, n_classes) 的 log 概率
其中 line_tensor 形状为 (seq_len, 1, n_letters)（与老师 notebook 的 lineToTensor 一致）。

  - NameRNN  : 老师原始版本风格，基于 nn.RNN（基线）
  - NameLSTM : 手写 LSTM 单元（四个门自己实现，不调用 nn.LSTM）——对应加分项“自己实现 LSTM”

统一入口 build_model(name, input_size, hidden_size, output_size)。
"""
import torch
import torch.nn as nn


class NameRNN(nn.Module):
    """基线：nn.RNN + 线性输出 + LogSoftmax，取最后时刻隐藏态做分类。"""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, line_tensor):
        # line_tensor: (seq_len, 1, input_size)
        _, hidden = self.rnn(line_tensor)        # hidden: (1, 1, hidden_size)
        output = self.h2o(hidden[0])             # (1, output_size)
        return self.softmax(output)


class NameLSTM(nn.Module):
    """手写 LSTM 单元（加分项）。

    标准 LSTM 门控（在时刻 t，输入 x_t、上一隐藏态 h_{t-1}、上一细胞态 c_{t-1}）：
        i_t = sigmoid(W_i [x_t, h_{t-1}])      输入门
        f_t = sigmoid(W_f [x_t, h_{t-1}])      遗忘门
        g_t = tanh   (W_g [x_t, h_{t-1}])      候选记忆
        o_t = sigmoid(W_o [x_t, h_{t-1}])      输出门
        c_t = f_t * c_{t-1} + i_t * g_t
        h_t = o_t * tanh(c_t)
    四个门的线性变换合并到一个 Linear(input+hidden, 4*hidden) 中，再 chunk 成 4 份。
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.x2h = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, line_tensor):
        seq_len = line_tensor.size(0)
        batch = line_tensor.size(1)              # = 1
        h = torch.zeros(batch, self.hidden_size, device=line_tensor.device)
        c = torch.zeros(batch, self.hidden_size, device=line_tensor.device)
        for t in range(seq_len):
            x = line_tensor[t]                   # (1, input_size)
            gates = self.x2h(torch.cat([x, h], dim=1))     # (1, 4*hidden)
            i, f, g, o = gates.chunk(4, dim=1)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)
            c = f * c + i * g
            h = o * torch.tanh(c)
        return self.softmax(self.h2o(h))         # (1, output_size)


_BUILDERS = {"rnn": NameRNN, "lstm": NameLSTM}
DISPLAY_NAMES = {"rnn": "RNN (nn.RNN)", "lstm": "LSTM (手写)"}


def build_model(name, input_size, hidden_size, output_size):
    key = name.lower()
    if key not in _BUILDERS:
        raise ValueError(f"未知模型 '{name}'，可选：{list(_BUILDERS)}")
    return _BUILDERS[key](input_size, hidden_size, output_size)
