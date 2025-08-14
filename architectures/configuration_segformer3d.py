from transformers import PretrainedConfig
from typing import List, Union


class SegFormer3DConfig(PretrainedConfig):
    model_type = "segformer3d"

    def __init__(
        self,
        in_channels: int = 4,
        sr_ratios: List[int] = [4, 2, 1, 1],
        embed_dims: List[int] = [32, 64, 160, 256],
        patch_kernel_size: List[Union[int, List[int]]] = [
            [7, 7, 7],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ],
        patch_stride: List[Union[int, List[int]]] = [
            [4, 4, 4],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
        ],
        patch_padding: List[Union[int, List[int]]] = [
            [3, 3, 3],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        mlp_ratios: List[int] = [4, 4, 4, 4],
        num_heads: List[int] = [1, 2, 5, 8],
        depths: List[int] = [2, 2, 2, 2],
        decoder_head_embedding_dim: int = 256,
        num_classes: int = 3,
        decoder_dropout: float = 0.0,
        qkv_bias: bool = True,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        **kwargs
    ):
        """
        Args:
            in_channels (int): Number of input channels
            sr_ratios (List[int]): Spatial reduction ratios for each stage
            embed_dims (List[int]): Embedding dimensions for each stage
            patch_kernel_size (List[int]): Kernel sizes for patch embedding
            patch_stride (List[int]): Stride values for patch embedding
            patch_padding (List[int]): Padding values for patch embedding
            mlp_ratios (List[int]): MLP expansion ratios for each stage
            num_heads (List[int]): Number of attention heads for each stage
            depths (List[int]): Number of transformer blocks per stage
            decoder_head_embedding_dim (int): Embedding dimension in decoder head
            num_classes (int): Number of output classes
            decoder_dropout (float): Dropout rate in decoder
            qkv_bias (bool): Whether to use bias in QKV projections
            attention_dropout (float): Dropout rate for attention
            projection_dropout (float): Dropout rate for projections
        """
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.sr_ratios = sr_ratios
        self.embed_dims = embed_dims
        self.patch_kernel_size = patch_kernel_size
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding
        self.mlp_ratios = mlp_ratios
        self.num_heads = num_heads
        self.depths = depths
        self.decoder_head_embedding_dim = decoder_head_embedding_dim
        self.num_classes = num_classes
        self.decoder_dropout = decoder_dropout
        self.qkv_bias = qkv_bias
        self.attention_dropout = attention_dropout
        self.projection_dropout = projection_dropout
