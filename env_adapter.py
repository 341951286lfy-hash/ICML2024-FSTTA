import torch
import torch.nn as nn


class EnvAdapter(nn.Module):
    """
    1. 输入/输出维度都保持为 hidden_size，不破坏主干特征维度；
    2. 中间用 bottleneck 降维，减少参数量；
    3. 最后一层 fc2 做零初始化，使模块初始输出为 0；
       这样在主模型中采用残差接法时：
           pano_embeds = pano_embeds + env_adapter(pano_embeds)
       初始阶段就等价于：
           pano_embeds = pano_embeds
       不会一开始就扰动主干能力。
    """

    def __init__(self, hidden_size, bottleneck=256, dropout=0.1):
        """
        参数说明：
        - hidden_size: 主干特征维度，输入输出都等于它
        - bottleneck: 中间瓶颈层维度，默认 256
        - dropout: dropout 概率，默认 0.1
        """
        super().__init__()

        # 先做 LayerNorm，稳定输入分布，便于测试时小步更新
        self.norm = nn.LayerNorm(hidden_size)

        # 第一层：hidden_size -> bottleneck
        self.fc1 = nn.Linear(hidden_size, bottleneck)

        # 非线性激活
        self.act = nn.ReLU(inplace=True)

        # dropout：轻微正则，避免 adapter 过拟合当前测试样本
        self.drop = nn.Dropout(dropout)

        # 第二层：bottleneck -> hidden_size
        self.fc2 = nn.Linear(bottleneck, hidden_size)

        # 关键要求：
        # 最后一层零初始化，使初始输出恒为 0
        # 这样残差接入主干后，初始时不会改变原特征
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        """
        前向传播

        输入：
        - x: [..., hidden_size]
             可以是 [B, N, H]，也可以是任意最后一维为 hidden_size 的张量

        输出：
        - delta: 与 x 同形状的修正量
                 注意：这里返回的是“残差分支输出”，不是 x + delta
                 残差相加应在外部主模型里完成，例如：
                     x = x + self.env_adapter(x)
        """
        # 标准 bottleneck adapter 结构：
        # LayerNorm -> Linear -> ReLU -> Dropout -> Linear
        delta = self.fc2(
            self.drop(
                self.act(
                    self.fc1(self.norm(x))
                )
            )
        )
        return delta