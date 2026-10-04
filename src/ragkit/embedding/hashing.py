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
        """TODO(你)：两行。

            if dim <= 0:
                raise EmbeddingError(f"dim 必须为正数，收到 {dim}")

        然后 ``self.dim = dim``。

        注意这里必须写成 ``self.dim: int = dim`` 或保证推断出来就是 int ——
        协议里声明的是 ``dim: int``，可变属性不变性要求类型完全一致。
        （``dim`` 参数已经标注成 int 了，所以 ``self.dim = dim`` 推断出来
          就是 int，这里其实是安全的。但你要知道为什么安全。）
        """
        if dim <= 0:
            raise EmbeddingError(f"dim 必须为正数，收到 {dim}")
        self.dim = dim

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """TODO(你)：逐条文本造一个确定性的单位向量。

        单条文本的处理步骤：

            1) 用文本的哈希当随机种子（**这就是"确定性"的来源**）：

                   seed = int(stable_id(text, length=16), 16)
                   rng = random.Random(seed)

               用 stable_id 而不是内置的 hash()：内置 hash() 对 str 加了
               随机盐（PYTHONHASHSEED），**同一个文本在不同进程里会得到不同的
               哈希值**。那样你的测试今天绿明天红，而且原因极难找。

            2) 造 dim 个数：

                   vec = [rng.uniform(-1.0, 1.0) for _ in range(self.dim)]

            3) L2 归一化，让向量长度变成 1：

                   norm = math.sqrt(sum(v * v for v in vec))
                   vec = [v / norm for v in vec]

               为什么要归一化：余弦相似度算的是向量夹角，
               对长度不敏感；但很多向量库（含 Milvus 的 COSINE 之外的度量）
               对长度敏感。统一归一化之后，"向量长度" 这个无关变量就被消掉了，
               相似度只反映方向（也就是语义）。

               norm 会是 0 吗？理论上只有当所有分量都是 0 时才会，
               而 uniform(-1, 1) 连续取到全 0 的概率为 0，不用特判。

        整个函数用一个列表推导包住所有文本，返回 ``list[list[float]]``。

        实现提示：可以把单条的处理抽成一个嵌套函数或私有方法，
        这样 embed 里就是一行列表推导，可读性更好。
        """

        def _embed_one(text: str) -> list[float]:
            seed = int(stable_id(text, length=16), 16)
            rng = random.Random(seed)
            vec = [rng.uniform(-1.0, 1.0) for _ in range(self.dim)]
            norm = math.sqrt(sum(v * v for v in vec))
            return [v / norm for v in vec]

        return [_embed_one(t) for t in texts]

    async def aclose(self) -> None:
        """TODO(你)：什么都不用做。

        这个实现没有连接池、没有文件句柄，所以方法体写 ``return None`` 就行。
        但方法**必须存在** —— 因为 Embedder 协议里有它，
        调用方会统一调 ``await embedder.aclose()``。

        「提供一个空实现」也是实现。这叫 Null Object 模式，
        目的是让调用方不需要写 ``if hasattr(x, 'aclose')`` 这种判断。
        """
        return None
