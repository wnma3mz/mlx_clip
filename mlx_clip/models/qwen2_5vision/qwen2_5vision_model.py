# coding: utf-8
import math
from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn


# "fullatt_block_indexes": [
#   7,
#   15,
#   23,
#   31
# ],
@dataclass
class Qwen2_5VisionConfig:
    depth: int
    embed_dim: int
    num_heads: int
    in_channels: int
    hidden_size: int
    patch_size: int
    spatial_merge_size: int
    spatial_merge_size: int
    spatial_patch_size: int
    temporal_patch_size: int
    window_size: int
    hidden_act: str = "silu"
    mlp_ratio: int = 4
    intermediate_size: Optional[int] = None

    def __post_init__(self):
        if self.intermediate_size is None:
            self.intermediate_size = self.embed_dim * self.mlp_ratio


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self._freqs = 1.0 / (theta ** (mx.arange(0, dim, 2) / dim))

    def __call__(self, seqlen: int) -> mx.array:
        seq = mx.arange(seqlen, dtype=self._freqs.dtype)
        freqs = mx.outer(seq, self._freqs)
        return freqs


class VisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = mx.zeros((config.hidden_size,))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        batch_size = x.shape[0]
        # Patch using conv:
        # [batch_size, sqrt(num_patches), sqrt(num_patches), embed_dim]
        patch_embeddings = self.patch_embedding(x)
        # [batch_size, num_patches, embed_dim]
        patch_embeddings = mx.flatten(patch_embeddings, start_axis=1, end_axis=2)
        embed_dim = patch_embeddings.shape[-1]
        # Prepend <CLS> embeddings
        # [batch_size, 1, embed_dim]
        cls_embeddings = mx.broadcast_to(self.class_embedding, (batch_size, 1, embed_dim))
        # [batch_size, num_patches + 1, embed_dim]
        embeddings = mx.concatenate((cls_embeddings, patch_embeddings), axis=1)
        # Add positional encoding
        embeddings += self.position_embedding.weight
        return embeddings


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = hidden_states.reshape(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        # [out_ch, in_ch, n, h, w] -> [out_ch, n, h, w, in_ch]
        hidden_states = mx.transpose(hidden_states, (0, 2, 3, 4, 1))
        hidden_states = self.proj(hidden_states).reshape(-1, self.embed_dim)
        return hidden_states


class PatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2, norm_bias: bool = False) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(context_dim, eps=1e-6, bias=norm_bias)
        self.mlp = [
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.ln_q(x).reshape(-1, self.hidden_size)
        for layer in self.mlp:
            x = layer(x)
        return x


def QuickGELUActivation(input: mx.array) -> mx.array:
    return input * mx.sigmoid(1.702 * input)


class Qwen2_5_VLMLP(nn.Module):
    def __init__(self, config, bias: bool = True):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = QuickGELUActivation  # ACT2FN[config.hidden_act]

    def __call__(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb_vision(tensor: mx.array, freqs: mx.array) -> mx.array:
    orig_dtype = tensor.dtype
    # tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = mx.expand_dims(mx.tile(mx.expand_dims(cos, axis=1), (1, 1, 2)), axis=0)
    sin = mx.expand_dims(mx.tile(mx.expand_dims(sin, axis=1), (1, 1, 2)), axis=0)

    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.astype(orig_dtype)
    return output


class VisionSdpaAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
        rotary_pos_emb: mx.array = None,
    ) -> mx.array:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).transpose(1, 0, 2, 3)
        q = apply_rotary_pos_emb_vision(mx.expand_dims(q, axis=0), rotary_pos_emb)[0]
        k = apply_rotary_pos_emb_vision(mx.expand_dims(k, axis=0), rotary_pos_emb)[0]

        q = q.transpose(1, 0, 2)
        k = k.transpose(1, 0, 2)
        v = v.transpose(1, 0, 2)
        attn_output = mx.fast.scaled_dot_product_attention(
            mx.expand_dims(q, axis=0),
            mx.expand_dims(k, axis=0),
            mx.expand_dims(v, axis=0),
            scale=1 / math.sqrt(q.shape[-1]),
            mask=attention_mask,
        )[0]
        attn_output = attn_output.transpose(1, 0, 2).reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2_5VLVisionBlock(nn.Module):
    def __init__(self, config: Qwen2_5VisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6, bias=False)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6, bias=False)

        self.attn = VisionSdpaAttention(config.hidden_size, num_heads=config.num_heads)
        self.mlp = Qwen2_5_VLMLP(config)

    def __call__(self, hidden_states, attention_mask, rotary_pos_emb) -> mx.array:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2_5VisionModel(nn.Module):
    def __init__(self, config: Qwen2_5VisionConfig) -> None:
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        self.window_size = config.window_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        self.blocks = [Qwen2_5VLVisionBlock(config) for _ in range(config.depth)]
        self.merger = PatchMerger(
            dim=config.embed_dim,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            norm_bias=False,
        )

    def rot_pos_emb(self, grid_thw):
        pos_ids = []

        for thw in grid_thw:
            t, h, w = thw.tolist()
            hpos_ids = mx.repeat(mx.expand_dims(mx.arange(h), axis=1), w, axis=1)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.transpose(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = mx.repeat(mx.expand_dims(mx.arange(w), axis=0), h, axis=0)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.transpose(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(mx.repeat(mx.stack([hpos_ids, wpos_ids], axis=-1), t, axis=1))
        pos_ids = mx.concatenate(pos_ids, axis=0)
        max_grid_size = mx.max(grid_thw[:, 1:])
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size.tolist())
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for thw in grid_thw:
            grid_t, grid_h, grid_w = thw.tolist()
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            # index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            index = mx.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)

            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size

            # index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = mx.pad(index, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=-100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.transpose(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            # seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            seqlens = mx.sum(index_padded != -100, axis=(2, 3)).reshape(-1)

            index_padded = index_padded.reshape(-1)

            # index_new = index_padded[index_padded != -100]
            index_new = mx.array([x for x in index_padded.flatten() if x != -100])

            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += grid_t * llm_grid_h * llm_grid_w
        window_index = mx.concat(window_index, axis=0)

        return window_index, cu_window_seqlens

    def _create_attention_mask(self, seq_length: int, cu_seqlens: List[int]) -> mx.array:
        attention_mask = mx.zeros(shape=(1, seq_length, seq_length), dtype=mx.bool_)
        for i in range(1, len(cu_seqlens)):
            l, r = cu_seqlens[i - 1], cu_seqlens[i]
            attention_mask[..., l:r, l:r] = True
        return mx.where(attention_mask, 0, -math.inf)

    def __call__(self, hidden_states: mx.array, grid_thw: mx.array) -> mx.array:
        hidden_states = self.patch_embed(hidden_states)
        seq_len, _ = hidden_states.shape
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        # TODO: cu_window_seqlens could be speed up
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)

        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        repeated = mx.repeat(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
        cu_seqlens = mx.cumsum(repeated)
        cu_seqlens = mx.pad(cu_seqlens, pad_width=(1, 0)).tolist()

        attention_mask = self._create_attention_mask(hidden_states.shape[0], cu_seqlens).astype(hidden_states.dtype)
        for blk in self.blocks:
            hidden_states = blk(hidden_states, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb)
        return self.merger(hidden_states)

    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype
