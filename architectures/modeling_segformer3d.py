"""SegFormer3D model implementation compatible with HuggingFace Transformers"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import SemanticSegmenterOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)

from .configuration_segformer3d import SegFormer3DConfig


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channel: int = 4,
        embed_dim: int = 768,
        kernel_size: Union[int, List[int]] = [7, 7, 7],
        stride: Union[int, List[int]] = [4, 4, 4],
        padding: Union[int, List[int]] = [3, 3, 3],
    ):
        super().__init__()
        # Convert single integers to lists if necessary
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 3
        if isinstance(stride, int):
            stride = [stride] * 3
        if isinstance(padding, int):
            padding = [padding] * 3

        self.patch_embeddings = nn.Conv3d(
            in_channel,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Store for shape calculations
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    def forward(self, x):
        patches = self.patch_embeddings(x)
        patches = patches.flatten(2).transpose(1, 2)
        patches = self.norm(patches)
        return patches

    def get_output_shape(self, input_shape):
        """Calculate output spatial dimensions after convolution"""
        d, h, w = input_shape
        od = ((d + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0]) + 1
        oh = ((h + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1]) + 1
        ow = ((w + 2 * self.padding[2] - self.kernel_size[2]) // self.stride[2]) + 1
        return (od, oh, ow)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, spatial_shape):
        B, N, C = x.shape
        d, h, w = spatial_shape
        x = x.transpose(1, 2).view(B, C, d, h, w)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class SelfAttention(nn.Module):
    def __init__(
        self,
        config: SegFormer3DConfig,
        embed_dim: int,
        num_heads: int,
        sr_ratio: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.attention_head_dim = embed_dim // num_heads
        self.scale = self.attention_head_dim**-0.5

        self.query = nn.Linear(embed_dim, embed_dim, bias=config.qkv_bias)
        self.key_value = nn.Linear(embed_dim, 2 * embed_dim, bias=config.qkv_bias)
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(config.projection_dropout)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(
                embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio
            )
            self.sr_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, spatial_shape, output_attentions: bool = False):
        D, H, W = spatial_shape
        B, N, C = x.shape
        q = (
            self.query(x)
            .reshape(B, N, self.num_heads, self.attention_head_dim)
            .permute(0, 2, 1, 3)
        )

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, D, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.sr_norm(x_)
            kv = (
                self.key_value(x_)
                .reshape(B, -1, 2, self.num_heads, self.attention_head_dim)
                .permute(2, 0, 3, 1, 4)
            )
        else:
            kv = (
                self.key_value(x)
                .reshape(B, -1, 2, self.num_heads, self.attention_head_dim)
                .permute(2, 0, 3, 1, 4)
            )

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)

        if output_attentions:
            return x, attn
        return (x,)


class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_features, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(embed_dim, hidden_features)
        self.conv = DWConv(hidden_features)
        self.linear_2 = nn.Linear(hidden_features, embed_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.projection_dropout)

    def forward(self, x, spatial_shape):
        x = self.linear_1(x)
        x = self.conv(x, spatial_shape)
        x = self.linear_2(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        config: SegFormer3DConfig,
        embed_dim: int,
        num_heads: int,
        sr_ratio: int,
        mlp_ratio: int,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(
            config=config,
            embed_dim=embed_dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_features = int(embed_dim * mlp_ratio)

        self.mlp = MLP(embed_dim, hidden_features, config.projection_dropout)

    def forward(self, x, spatial_shape, output_attentions: bool = False):
        attention_outputs = self.attention(
            self.norm1(x), spatial_shape, output_attentions
        )
        x = x + attention_outputs[0]
        x = x + self.mlp(self.norm2(x), spatial_shape)

        outputs = (x,) + attention_outputs[1:] if output_attentions else (x,)
        return outputs, spatial_shape


@add_start_docstrings("""SegFormer3D Model for 3D semantic segmentation.""")
class SegFormer3DPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading
    and loading pretrained models.
    """

    config_class = SegFormer3DConfig
    base_model_prefix = "segformer3d"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.BatchNorm3d):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = (
                module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            )
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv3d):
            fan_out = (
                module.kernel_size[0]
                * module.kernel_size[1]
                * module.kernel_size[2]
                * module.out_channels
            )
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()


@add_start_docstrings("""SegFormer3D Model for 3D semantic segmentation tasks.""")
class SegFormer3DModel(SegFormer3DPreTrainedModel):
    def __init__(self, config: SegFormer3DConfig):
        super().__init__(config)

        # Encoder components
        self.encoders = nn.ModuleList()
        self.encoder_norms = nn.ModuleList()
        self.transformer_blocks = nn.ModuleList()

        # Build hierarchical encoder stages
        for i in range(4):
            # Patch embedding for this stage
            encoder = PatchEmbedding(
                in_channel=config.in_channels if i == 0 else config.embed_dims[i - 1],
                embed_dim=config.embed_dims[i],
                kernel_size=config.patch_kernel_size[i],
                stride=config.patch_stride[i],
                padding=config.patch_padding[i],
            )
            self.encoders.append(encoder)

            # Transformer blocks for this stage
            stage_blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        config=config,
                        embed_dim=config.embed_dims[i],
                        num_heads=config.num_heads[i],
                        sr_ratio=config.sr_ratios[i],
                        mlp_ratio=config.mlp_ratios[i],
                    )
                    for _ in range(config.depths[i])
                ]
            )
            self.transformer_blocks.append(stage_blocks)

            # Layer norm for this stage
            self.encoder_norms.append(nn.LayerNorm(config.embed_dims[i]))

        # Decoder components
        self.decoder = SegFormer3DDecoderHead(config)

        # Initialize weights
        self.post_init()

    def get_input_embeddings(self):
        return self.encoders[0].patch_embeddings

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SemanticSegmenterOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        encoder_hidden_states = []
        all_attentions = [] if output_attentions else None

        x = pixel_values

        # Process through encoder stages
        for stage_idx in range(4):
            # Track input spatial dimensions
            if stage_idx == 0:
                spatial_shape = pixel_values.shape[2:]  # (D, H, W)

            # Patch embedding
            x = self.encoders[stage_idx](x)
            spatial_shape = self.encoders[stage_idx].get_output_shape(spatial_shape)
            B, N, C = x.shape

            # Transformer blocks
            for block in self.transformer_blocks[stage_idx]:
                block_outputs, spatial_shape = block(
                    x, spatial_shape, output_attentions
                )
                x = block_outputs[0]
                if output_attentions:
                    all_attentions.append(block_outputs[1])

            # Layer norm
            x = self.encoder_norms[stage_idx](x)

            # Reshape and store hidden state using calculated dimensions
            d, h, w = spatial_shape
            x_reshaped = x.reshape(B, d, h, w, -1).permute(0, 4, 1, 2, 3).contiguous()
            encoder_hidden_states.append(x_reshaped)

            # Prepare input for next stage if not last stage
            if stage_idx < 3:
                x = x_reshaped

        # Decode features
        logits = self.decoder(encoder_hidden_states)

        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                raise ValueError("The number of labels should be greater than one")
            else:
                # compute loss for 3D semantic segmentation
                loss_fct = CrossEntropyLoss(ignore_index=255)
                loss = loss_fct(
                    logits.view(-1, self.config.num_classes), labels.view(-1)
                )

        if not return_dict:
            outputs = (
                (logits,)
                + (encoder_hidden_states if output_hidden_states else ())
                + (all_attentions if output_attentions else ())
            )
            return ((loss,) + outputs) if loss is not None else outputs

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_hidden_states if output_hidden_states else None,
            attentions=all_attentions,
        )


class SegFormer3DDecoderHead(nn.Module):
    def __init__(self, config: SegFormer3DConfig):
        super().__init__()

        # Linear layers for each encoder stage
        self.linear_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, config.decoder_head_embedding_dim),
                    nn.LayerNorm(config.decoder_head_embedding_dim),
                )
                for dim in config.embed_dims[::-1]
            ]
        )

        # Feature fusion
        self.linear_fuse = nn.Sequential(
            nn.Conv3d(
                in_channels=4 * config.decoder_head_embedding_dim,
                out_channels=config.decoder_head_embedding_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm3d(config.decoder_head_embedding_dim),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(config.decoder_dropout)
        self.linear_pred = nn.Conv3d(
            config.decoder_head_embedding_dim, config.num_classes, kernel_size=1
        )
        self.upsample = nn.Upsample(
            scale_factor=4, mode="trilinear", align_corners=False
        )

    def forward(self, encoder_hidden_states):
        # Process features from each encoder stage
        B = encoder_hidden_states[-1].shape[0]

        # Linear projection and upsampling of each stage's features
        decoded_features = []
        for i, features in enumerate(
            encoder_hidden_states[::-1]
        ):  # Process in reverse order
            d, h, w = features.shape[2:]
            projected = (
                self.linear_layers[i](features.flatten(2).transpose(1, 2))
                .transpose(1, 2)
                .reshape(B, -1, d, h, w)
            )

            # Upsample if not the last feature map
            if i != len(encoder_hidden_states[::-1]):
                projected = torch.nn.functional.interpolate(
                    projected,
                    size=encoder_hidden_states[0].shape[
                        2:
                    ],  # Size of first stage features
                    mode="trilinear",
                    align_corners=False,
                )
            decoded_features.append(projected)

        # Fuse all features
        fused_features = self.linear_fuse(torch.cat(decoded_features, dim=1))

        # Final prediction
        x = self.dropout(fused_features)
        x = self.linear_pred(x)
        x = self.upsample(x)

        return x


if __name__ == "__main__":
    input = torch.randint(
        low=0,
        high=255,
        size=(1, 4, 128, 128, 128),
        dtype=torch.float,
    )
    input = input.to("cuda:0")
    config = SegFormer3DConfig()
    segformer3D = SegFormer3DModel(config).to("cuda:0")
    output = segformer3D(input)
    print(output["logits"].shape)
