"""一个纯本地的假 Embedder：不联网、零依赖、完全确定。

用途（很重要，别小看它）：
    * 单元测试 —— 测管道通不通不需要花钱调 API，也不该依赖网络；
    * 离线 demo / CI —— 没有 API key 的机器上也能把整条链路跑通；
    * 给上层做「假数据」的基准线。

⚠️ 它没有语义。把「猫」和「狗」丢进去，拿到的向量不会比「猫」和「火车」
   更接近。所以它**只能验证管道是否通畅，不能用来评估检索质量**。
   真实的检索效果评估必须用真模型 —— 这一点在 M8 会再强调一次。
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from ..errors import EmbeddingError
from ..utils import stable_id

__all__ = ["HashingEmbedder"]


class HashingEmbedder:
    """把文本哈希成固定维度的单位向量。"""

    def __init__(self, dim: int = 256) -> None:
        """``dim`` 必须为正数。"""
        if dim <= 0:
            raise EmbeddingError(f"dim 必须为正数，收到 {dim}")
        self.dim = dim

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """把文本哈希成**确定性**的单位向量：同一段文本永远得到同一个向量。"""

        def _embed_one(text: str) -> list[float]:
            seed = int(stable_id(text, length=16), 16)
            rng = random.Random(seed)
            vec = [rng.uniform(-1.0, 1.0) for _ in range(self.dim)]
            norm = math.sqrt(sum(v * v for v in vec))
            return [v / norm for v in vec]

        return [_embed_one(t) for t in texts]

    async def aclose(self) -> None:
        """空实现 —— 这个 Embedder 没有连接池或文件句柄需要释放。"""
        return None
