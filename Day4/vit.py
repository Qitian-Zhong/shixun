

import torch
from torch import nn
import math
import warnings


# 自定义辅助函数
def expand_to_tuple(value):
    """将输入转换为二维元组"""
    if isinstance(value, tuple):
        return value
    return (value, value)


# 自定义层归一化
class CustomNorm(nn.Module):
    """可配置的层归一化模块"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        return self.norm(x)


# 自定义前馈网络
class CustomFeedForward(nn.Module):
    """增强型前馈网络模块"""

    def __init__(self, dim, expansion_factor=4, dropout=0.1):
        super().__init__()
        hidden_dim = dim * expansion_factor

        self.net = nn.Sequential(
            CustomNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# 自定义多头注意力
class MultiHeadAttention(nn.Module):
    """可配置的多头自注意力机制"""

    def __init__(self, dim, num_heads=8, head_dim=64, dropout=0.1):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.use_output_proj = not (num_heads == 1 and head_dim == dim)

        self.num_heads = num_heads
        self.scale = head_dim ** -0.5

        self.norm = CustomNorm(dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # 使用独立的线性层替代单一线性层
        self.query_proj = nn.Linear(dim, inner_dim, bias=False)
        self.key_proj = nn.Linear(dim, inner_dim, bias=False)
        self.value_proj = nn.Linear(dim, inner_dim, bias=False)

        self.output_proj = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if self.use_output_proj else nn.Identity()

    def forward(self, x):
        x_norm = self.norm(x)

        # 分别计算Q、K、V
        q = self.query_proj(x_norm)
        k = self.key_proj(x_norm)
        v = self.value_proj(x_norm)

        # 重新排列维度
        q = q.view(q.size(0), q.size(1), self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.view(k.size(0), k.size(1), self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.view(v.size(0), v.size(1), self.num_heads, -1).permute(0, 2, 1, 3)

        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = self.softmax(attn_scores)
        attn_probs = self.dropout(attn_probs)

        # 应用注意力权重
        weighted_values = torch.matmul(attn_probs, v)

        # 重新组合输出
        weighted_values = weighted_values.permute(0, 2, 1, 3).contiguous()
        weighted_values = weighted_values.view(weighted_values.size(0), weighted_values.size(1), -1)

        return self.output_proj(weighted_values)


# 自定义Transformer块
class TransformerBlock(nn.Module):
    """带有残差连接的Transformer编码块"""

    def __init__(self, dim, depth, num_heads, head_dim, mlp_dim, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleList([
                MultiHeadAttention(dim, num_heads, head_dim, dropout),
                CustomFeedForward(dim, expansion_factor=mlp_dim // dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.blocks:
            # 添加残差连接
            x = x + attn(x)
            x = x + ff(x)
        return x


# 自定义位置编码
class LearnablePositionEncoding(nn.Module):
    """可学习的位置编码模块"""

    def __init__(self, num_patches, dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))

    def forward(self, x):
        return x + self.pos_embed


# 自定义分类标记
class ClassificationToken(nn.Module):
    """分类标记生成器"""

    def __init__(self, dim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x):
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        return torch.cat((cls_tokens, x), dim=1)


# 自定义图像分块嵌入
class ImageToPatchEmbedding(nn.Module):
    """图像分块嵌入模块"""

    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        img_height, img_width = expand_to_tuple(img_size)
        patch_height, patch_width = expand_to_tuple(patch_size)

        # 验证尺寸兼容性
        if img_height % patch_height != 0 or img_width % patch_width != 0:
            warnings.warn("图像尺寸必须能被分块尺寸整除，自动调整分块尺寸", UserWarning)
            patch_height = patch_width = math.gcd(img_height, img_width)

        num_patches = (img_height // patch_height) * (img_width // patch_width)
        patch_dim = in_channels * patch_height * patch_width

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
            CustomNorm(embed_dim)
        )
        self.num_patches = num_patches

    def forward(self, x):
        return self.projection(x).permute(0, 2, 1)


# 自定义视觉Transformer
class CustomVisionTransformer(nn.Module):
    """个性化视觉Transformer模型"""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, head_dim=64, mlp_ratio=4,
                 dropout=0.1, embed_dropout=0.1, pool_type='hybrid'):
        """
        参数:
            img_size: 输入图像尺寸
            patch_size: 分块尺寸
            in_channels: 输入通道数
            num_classes: 分类类别数
            embed_dim: 嵌入维度
            depth: Transformer块深度
            num_heads: 注意力头数
            head_dim: 每个注意力头的维度
            mlp_ratio: MLP扩展比例
            dropout: 通用dropout率
            embed_dropout: 嵌入层dropout率
            pool_type: 池化类型 ('cls', 'mean', 'hybrid')
        """
        super().__init__()

        # 验证池化类型
        valid_pool_types = {'cls', 'mean', 'hybrid'}
        if pool_type not in valid_pool_types:
            raise ValueError(f"无效的池化类型: {pool_type}. 请选择 {valid_pool_types}")

        self.patch_embed = ImageToPatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = ClassificationToken(embed_dim)
        self.pos_embed = LearnablePositionEncoding(self.patch_embed.num_patches, embed_dim)
        self.dropout = nn.Dropout(embed_dropout)

        # Transformer编码器
        mlp_dim = int(embed_dim * mlp_ratio)
        self.transformer = TransformerBlock(
            embed_dim, depth, num_heads, head_dim, mlp_dim, dropout
        )

        # 池化策略
        self.pool_type = pool_type
        self.norm = CustomNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, img):
        # 图像分块嵌入
        x = self.patch_embed(img)

        # 添加分类标记
        x = self.cls_token(x)

        # 添加位置编码
        x = self.pos_embed(x)
        x = self.dropout(x)

        # 通过Transformer编码器
        x = self.transformer(x)

        # 应用层归一化
        x = self.norm(x)

        # 提取分类特征
        if self.pool_type == 'cls':
            cls_token = x[:, 0]
        elif self.pool_type == 'mean':
            cls_token = x.mean(dim=1)
        else:  # 混合策略
            cls_token = 0.5 * x[:, 0] + 0.5 * x.mean(dim=1)

        # 分类头
        return self.head(cls_token)

    def get_num_params(self):
        """返回模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    # 模型配置参数
    config = {
        'img_size': 256,
        'patch_size': 16,
        'in_channels': 3,
        'num_classes': 100,
        'embed_dim': 1024,
        'depth': 6,
        'num_heads': 16,
        'head_dim': 64,
        'mlp_ratio': 2,
        'dropout': 0.1,
        'embed_dropout': 0.1,
        'pool_type': 'hybrid'  # 使用混合池化策略
    }

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型
    model = CustomVisionTransformer(**config).to(device)

    # 打印模型信息
    print(f"模型参数数量: {model.get_num_params() / 1e6:.2f} 百万")

    # 测试输入
    test_input = torch.randn(2, 3, 256, 256).to(device)

    # 前向传播
    with torch.no_grad():
        output = model(test_input)

    print(f"输入尺寸: {test_input.shape}")
    print(f"输出尺寸: {output.shape}")
    print(f"示例输出: {output[0, :5]}")

    # 可选：打印模型结构
    # from torchsummary import summary
    # summary(model, (3, 256, 256), device=device.type)