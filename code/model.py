import torch
import torch.nn as nn

# 定义LSTM模型
class myLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        #self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]     # [bsz, hidden_size]，最后一个时间步
        #x = self.dropout(x)
        out = self.fc(x)
        return out


# 定义Transformer模型
# 位置编码
class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)    # [max_len, 1]
        pe = torch.zeros(max_len, d_model)  
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# 独立的Transformer Encoder层
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), 
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask=None):
        # 注意力 + 残差连接 + 层归一化
        attn_out, _ = self.attn(x, x, x, key_padding_mask=padding_mask)
        x = self.layernorm1(x + attn_out)
        
        # 前馈网络 + 残差连接 + 层归一化
        ffn_out = self.ffn(x)
        x = self.layernorm2(x + ffn_out)
        return x

class myTransformer(nn.Module):
    def __init__(self, input_size, output_size, num_layers=2, d_model=256, num_heads=8):
        super().__init__()
        self.input_fc = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionEncoding(d_model)
        
        # 使用 nn.ModuleList 来堆叠多个 EncoderBlock
        self.encoder_blocks = nn.ModuleList(
            [TransformerEncoder(d_model, num_heads) for _ in range(num_layers)]
        )
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_fc = nn.Linear(d_model, output_size)

    def forward(self, x, padding_mask=None):
        x = self.input_fc(x)
        x = self.pos_encoder(x)
        
        for block in self.encoder_blocks:
            x = block(x, padding_mask=padding_mask)

        # 池化和输出
        x = x.transpose(1, 2)  # [bsz, d_model, seq_len]
        x = self.pool(x).squeeze(-1)  # [bsz, d_model]
        out = self.output_fc(x)
        return out

# 趋势增强Transformer
# --- Moving Average Trend Extractor ---
class MovingAverage(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, F, T]
        trend = self.avg_pool(x)
        return trend.permute(0, 2, 1)  # [B, T, F]


# --- Trendformer Main Module ---
class Trendformer(nn.Module):
    def __init__(self, input_size, output_size, d_model=256, num_heads=8, num_layers=2):
        super().__init__()
        self.trend_extractor = MovingAverage(kernel_size=25)

        self.input_proj = nn.Linear(input_size, d_model)
        self.trend_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionEncoding(d_model)

        self.encoder_blocks = nn.ModuleList([
            TransformerEncoder(d_model, num_heads) for _ in range(num_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, output_size)  # 直接输出 [B, output_size]
        )

        #self.alpha = nn.Parameter(torch.tensor(0.5))


    def forward(self, x):
        B,T,_ = x.shape

        trend = self.trend_extractor(x)           # [B, T, F]
        residual = x - trend                      # [B, T, F]

        x_r = self.input_proj(residual)           # [B, T, D]
        t_r = self.trend_proj(trend)       
        
        # 融合趋势和残差编码
        encoder_input = x_r + t_r                       # [B, T_in, d_model]
        z = self.pos_encoder(encoder_input)

        # Encoder
        for blk in self.encoder_blocks:
            z = blk(z)

        # 全局池化，把序列维度T_in压成一个向量 [B, d_model]
        pooled = torch.mean(z, dim=1)

        output = self.output_proj(pooled)               # [B, output_size]

        return output
