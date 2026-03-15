import torch
import torch.nn as nn


class SceneAdapter(nn.Module):
    """

    作用：
    - 用于在测试时对“融合后的导航特征”做轻量残差修正；
    - 不改动主干网络结构，只学习一个小型 bottleneck adapter；
    - 在 `vilmodel.py` 中通常会实例化两个：
        1) self.scene_adapter_g  -> 作用在 global 分支的 gmap_embeds 上
        2) self.scene_adapter_v  -> 作用在 local 分支的 vp_embeds 上

    文档要求：
    - 结构与 EnvAdapter 相同；
    - 输入维度 = 输出维度 = hidden_size；
    - 最后一层零初始化，保证初始时不会破坏主干能力。
    """

    def __init__(self, hidden_size, bottleneck=256, dropout=0.1):
        """
        参数：
        - hidden_size: 主干特征维度，输入输出都保持这个维度
        - bottleneck: 中间瓶颈层维度，默认 256
        - dropout: dropout 概率，默认 0.1
        """
        super().__init__()

        # 先对输入做归一化，缓和不同场景下特征分布偏移
        self.norm = nn.LayerNorm(hidden_size)

        # 降维到 bottleneck，减少参数量，提高测试时更新稳定性
        self.fc1 = nn.Linear(hidden_size, bottleneck)

        # 非线性激活
        self.act = nn.ReLU(inplace=True)

        # 轻量正则，避免 adapter 对单个 episode 过拟合
        self.drop = nn.Dropout(dropout)

        # 再投影回 hidden_size，作为残差修正量
        self.fc2 = nn.Linear(bottleneck, hidden_size)

        # 关键要求：
        # 最后一层做零初始化，这样初始输出恒为 0
        # 若外部按残差方式使用：
        #   gmap_embeds = gmap_embeds + self.scene_adapter_g(gmap_embeds)
        #   vp_embeds   = vp_embeds   + self.scene_adapter_v(vp_embeds)
        # 则初始时模型行为与原主干完全一致
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        """
        前向传播

        输入：
        - x: shape = [..., hidden_size]
             既可以是 [B, L, H]，也可以是其他最后一维为 hidden_size 的张量

        输出：
        - delta: 与 x 同形状的残差修正量

        注意：
        - 这里返回的是“修正量”，不是最终残差相加后的结果；
        - 真正的残差连接应在外部完成，例如：
              gmap_embeds = gmap_embeds + self.scene_adapter_g(gmap_embeds)
              vp_embeds   = vp_embeds   + self.scene_adapter_v(vp_embeds)
        """
        delta = self.fc2(
            self.drop(
                self.act(
                    self.fc1(self.norm(x))
                )
            )
        )
        return delta
