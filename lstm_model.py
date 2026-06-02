import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ImprovedTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        # 第一层卷积
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.chomp1 = Chomp1d(padding)
        self.dropout1 = nn.Dropout(dropout)

        # 第二层卷积
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.chomp2 = Chomp1d(padding)
        self.dropout2 = nn.Dropout(dropout)

        # 下采样（如果需要）
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        # 残差连接
        res = x if self.downsample is None else self.downsample(x)

        # 第一层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.chomp1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        # 第二层
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.chomp2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        return out + res


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class ImprovedTCN(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_ch = input_dim if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(
                ImprovedTemporalBlock(
                    in_ch, out_ch,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=(kernel_size - 1) * dilation,
                    dropout=dropout
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # -> (batch, input_dim, seq_len)
        y = self.network(x)
        return y.permute(0, 2, 1)  # -> (batch, seq_len, channels)


# 简化注意力机制（使用缩放点积注意力）
class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = math.sqrt(hidden_dim)

    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = x.shape

        # 使用线性变换得到Q, K, V
        Q = x  # 简化为使用输入作为Q, K, V
        K = x
        V = x

        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 应用注意力权重
        attended = torch.matmul(attention_weights, V)

        return attended


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = ScaledDotProductAttention(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        attended = self.attention(x)
        out = self.out_proj(attended)
        return out


# 修复的主模型
class ImprovedTCN_BiLSTM_Attention(nn.Module):
    def __init__(self, input_dim, tcn_channels, lstm_hidden, output_dim=1,
                 dropout=0.2, num_lstm_layers=2, use_attention=True):
        super().__init__()
        self.use_attention = use_attention

        # TCN部分
        self.tcn = ImprovedTCN(input_dim, tcn_channels, dropout=dropout)

        # LSTM部分
        self.bilstm = nn.LSTM(
            input_size=tcn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )

        # 注意力部分
        if use_attention:
            self.attention = Attention(lstm_hidden * 2)

        # 输出层 - 修复维度问题
        # LSTM输出维度为 hidden_dim * 2（双向）
        # 经过注意力后维度不变，我们取最后一个时间步或进行池化
        self.fc1 = nn.Linear(lstm_hidden * 2, lstm_hidden)
        self.fc2 = nn.Linear(lstm_hidden, output_dim)
        self.dropout = nn.Dropout(dropout)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for name, param in self.bilstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        # TCN处理
        tcn_out = self.tcn(x)

        # LSTM处理
        lstm_out, (hidden, cell) = self.bilstm(tcn_out)

        # 注意力机制（可选）
        if self.use_attention:
            attn_out = self.attention(lstm_out)
            # 取最后一个时间步
            last_out = attn_out[:, -1, :]
        else:
            # 如果不使用注意力，取双向LSTM最后一个时间步的前向和后向隐藏状态
            # 或者直接取最后一个时间步的输出
            last_out = lstm_out[:, -1, :]

        # 输出层
        out = self.dropout(F.relu(self.fc1(last_out)))
        out = self.fc2(out)

        return out






if __name__ == '__main__':
    # 测试改进模型

    batch_size, seq_len, features = 32, 24, 5
    test_input = torch.randn(batch_size, seq_len, features)
    print(f"\n输入维度: {test_input.shape}")
    model = ImprovedTCN_BiLSTM_Attention(
        input_dim=5,
        tcn_channels=[32, 64],
        lstm_hidden=64,
        output_dim=1,
        dropout=0.2,
        num_lstm_layers=1,
        use_attention=True
    )

    print("改进模型结构：")
    print(model)

    output = model(test_input)
    print(f"\n输入维度: {test_input.shape}")
    print(f"输出维度: {output.shape}")