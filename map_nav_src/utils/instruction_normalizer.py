import re


class InstructionNormalizer:
    """
    作用：
    1. 对原始自然语言导航指令做轻量清洗；
    2. 统一一些常见但风格不同的表达，减少口语噪声；
    3. 输出规范化后的文本，供后续 tokenizer 重新编码。

    设计原则：
    - 第一版先用纯 Python 规则跑通，不接 LLM；
    - 不追求“语义改写”，只做轻量、保守、可控的规范化；
    - 尽量不改变原始导航意图，只做格式清洗和局部短语统一。
    """

    def __init__(self, tokenizer=None):
        """
        参数：
        - tokenizer:
            外部传入的分词器/编码器对象，要求至少支持：
                tokenizer.encode(text)
            一般在 agent 里初始化：
                self.instr_norm = InstructionNormalizer(tokenizer=your_tokenizer)
        """
        self.tokenizer = tokenizer

    def normalize(self, text):
        """
        对原始指令做文本规范化。

        处理内容：
        1. None 安全处理；
        2. 全部转小写；
        3. 去掉标点（保留字母/数字/下划线/空白）；
        4. 合并多余空格；
        5. 将少量常见表达归一到更标准的 VLN 风格短语。

        输入：
        - text: str 或 None

        输出：
        - norm_text: str
        """
        # 空输入保护
        if text is None:
            return ""

        # 统一成小写并去掉首尾空格
        t = text.lower().strip()

        # 去掉标点符号，只保留“单词字符 + 空白字符”
        # 例如：
        #   "Turn right, then walk forward!"
        # -> "turn right  then walk forward "
        t = re.sub(r"[^\w\s]", " ", t)

        # 合并连续空白
        t = re.sub(r"\s+", " ", t)

        # ------------------------------------------------------------
        # 下面是一些非常保守的规则替换：
        # 只统一常见表达，不做激进改写，避免改坏导航语义
        # ------------------------------------------------------------

        # “向前移动 / 向前走” 统一成 “go forward”
        t = t.replace("move forward", "go forward")
        t = t.replace("walk forward", "go forward")

        # 长口语表达统一成简洁标准表达
        t = t.replace("turn towards your right hand side", "turn right")
        t = t.replace("turn towards your left hand side", "turn left")

        # 去掉替换后可能残留的多余空格
        t = re.sub(r"\s+", " ", t).strip()

        return t

    def encode(self, text):
        """
        将规范化后的文本编码为 token id 序列。

        输入：
        - text: 规范化后的字符串

        输出：
        - 编码结果，通常是 list[int]

        说明：
        - 文档要求第一版直接调用 tokenizer.encode(text)；
        - 如果没有传 tokenizer，这里直接报错，避免静默失败。
        """
        if self.tokenizer is None:
            raise ValueError(
                "InstructionNormalizer.encode() requires a tokenizer, "
                "but `tokenizer` is None."
            )

        return self.tokenizer.encode(text)
