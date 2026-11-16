"""Action head: a flow-matching transformer over the action chunk.

One token per chunk step; each block applies self-attention over the chunk, cross-attention to the observation
and instruction tokens, and a SwiGLU feed-forward layer, each modulated by the flow time through adaLN-Zero
(DiT). Queries and keys use per-head RMSNorm. The instruction is encoded once by a frozen T5 encoder.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Flow time t in [0, 1] -> sinusoidal features of 1000 t -> MLP."""

    def __init__(self, hidden_size: int, frequency_dim: int = 256) -> None:
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(nn.Linear(frequency_dim, hidden_size), nn.SiLU(),
                                 nn.Linear(hidden_size, hidden_size))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.frequency_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = (1000.0 * t.float())[:, None] * freqs[None]
        embedding = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * hidden)
        self.w3 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.drop(F.silu(a) * b))


class Attention(nn.Module):
    """Multi-head attention (self-attention when context is None) with per-head QK RMSNorm."""

    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("width must be divisible by the number of heads")
        self.num_heads, self.head_dim, self.dropout = num_heads, dim // num_heads, dropout
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None,
                key_mask: torch.Tensor | None = None) -> torch.Tensor:
        """key_mask [B, L] bool: True for keys to attend to (False: instruction padding)."""
        B, N, C = x.shape
        source = x if context is None else context
        q = self.q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(source).view(B, source.shape[1], 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        mask = None if key_mask is None else key_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    """adaLN-Zero block: self-attention, cross-attention and SwiGLU, each gated by the flow time."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0,
                 qk_norm: bool = True) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(dim, num_heads, qk_norm, dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(dim, num_heads, qk_norm, dropout)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(math.ceil(2 * mlp_ratio * dim / 3 / 64) * 64)  # parameter count of a 4x GELU MLP
        self.mlp = SwiGLU(dim, hidden, dropout)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor, context: torch.Tensor,
                context_mask: torch.Tensor | None = None) -> torch.Tensor:
        (shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(9, dim=-1)
        x = x + gate_sa.unsqueeze(1) * self.self_attn(modulate(self.norm1(x), shift_sa, scale_sa))
        x = x + gate_ca.unsqueeze(1) * self.cross_attn(modulate(self.norm2(x), shift_ca, scale_ca), context,
                                                       context_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.linear = nn.Linear(dim, out_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


class ActionHead(nn.Module):
    """Velocity field v(x_t, t | observation tokens, instruction) over an action chunk."""

    def __init__(self, action_dim: int, horizon: int, cond_dim: int, cond_len: int, n_layer: int = 12,
                 n_head: int = 12, n_emb: int = 768, mlp_ratio: float = 4.0, dropout: float = 0.1,
                 qk_norm: bool = True, language_model: str | None = "t5-small",
                 pretrained_language: bool = True) -> None:
        super().__init__()
        self.horizon = horizon
        self.language_conditioned = language_model is not None
        self.input_emb = nn.Linear(action_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        self.time_embedder = TimestepEmbedder(n_emb)
        self.cond_emb = nn.Linear(cond_dim, n_emb)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, cond_len, n_emb))
        if self.language_conditioned:
            self._load_language_encoder(language_model, pretrained_language)
            self.lang_adaptor = nn.Sequential(nn.Linear(self.language_encoder.config.d_model, n_emb),
                                              nn.GELU(approximate="tanh"), nn.Linear(n_emb, n_emb))
        self.context_norm = nn.RMSNorm(n_emb, eps=1e-6)
        self.blocks = nn.ModuleList(Block(n_emb, n_head, mlp_ratio, dropout, qk_norm) for _ in range(n_layer))
        self.final_layer = FinalLayer(n_emb, action_dim)
        self._initialize()

    def _load_language_encoder(self, model_name: str, pretrained: bool) -> None:
        from transformers import AutoTokenizer, T5Config, T5EncoderModel

        repo = f"google-t5/{model_name}"
        self.tokenizer = AutoTokenizer.from_pretrained(repo)
        # From a checkpoint the weights are loaded afterwards; only the architecture is needed here.
        self.language_encoder = (T5EncoderModel.from_pretrained(repo) if pretrained
                                 else T5EncoderModel(T5Config.from_pretrained(repo))).eval()
        self.language_encoder.requires_grad_(False)
        self._language_cache: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}

    def _language_tokens(self, text: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """T5 token features [L, d] and token mask [L] of one instruction, cached (T5 is frozen). T5 runs in
        float32 with autocast off (it overflows in float16)."""
        key = (text, str(device))
        if key not in self._language_cache:
            tokens = self.tokenizer([text], return_tensors="pt", truncation=True).to(device)
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                hidden = self.language_encoder.float()(**tokens).last_hidden_state[0].float()
            self._language_cache[key] = (hidden, tokens["attention_mask"][0].bool())
        return self._language_cache[key]

    def _encode_language(self, text, device) -> tuple[torch.Tensor, torch.Tensor]:
        text = [text] if isinstance(text, str) else list(text)
        encoded = [self._language_tokens(str(item), device) for item in text]
        length = max(hidden.shape[0] for hidden, _ in encoded)
        hidden = torch.zeros(len(encoded), length, encoded[0][0].shape[-1], device=device)
        mask = torch.zeros(len(encoded), length, dtype=torch.bool, device=device)
        for i, (h, m) in enumerate(encoded):
            hidden[i, :len(h)], mask[i, :len(m)] = h, m
        return self.lang_adaptor(hidden.to(self.lang_adaptor[0].weight.dtype)), mask

    def train(self, mode: bool = True):
        super().train(mode)
        if self.language_conditioned:
            self.language_encoder.eval()
        return self

    def _initialize(self) -> None:
        def basic(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for name, module in self.named_children():
            if name != "language_encoder":
                module.apply(basic)
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.cond_pos_emb, std=0.02)
        nn.init.normal_(self.input_emb.weight, std=0.02)
        for layer in self.time_embedder.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
        for block in self.blocks:  # adaLN-Zero: every residual branch starts closed
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def optim_groups(self, weight_decay: float) -> list[dict]:
        """Weight decay on matrices only (no norms, biases or position embeddings)."""
        decay, no_decay = [], []
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                (decay if parameter.ndim >= 2 and "pos_emb" not in name else no_decay).append(parameter)
        return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

    def forward(self, sample: torch.Tensor, timestep, vis_cond: torch.Tensor, lang_cond=None) -> torch.Tensor:
        """sample [B, H, action_dim], timestep [B] or scalar in [0, 1], vis_cond [B, L, cond_dim]."""
        B = sample.shape[0]
        t = timestep if torch.is_tensor(timestep) else torch.tensor(float(timestep), device=sample.device)
        t = t.to(sample.device).reshape(-1).expand(B)
        c = self.time_embedder(t)
        context = self.cond_emb(vis_cond) + self.cond_pos_emb[:, :vis_cond.shape[1]]
        context_mask = None
        if self.language_conditioned:
            if lang_cond is None:
                raise ValueError("the action head is instruction-conditioned: pass the instruction")
            language, language_mask = self._encode_language(lang_cond, sample.device)
            if language.shape[0] == 1 and B > 1:
                language, language_mask = language.expand(B, -1, -1), language_mask.expand(B, -1)
            context = torch.cat((context, language.to(context.dtype)), dim=1)
            context_mask = torch.cat((torch.ones(B, vis_cond.shape[1], dtype=torch.bool, device=sample.device),
                                      language_mask), dim=1)
        context = self.context_norm(context)
        x = self.input_emb(sample) + self.pos_emb[:, :sample.shape[1]]
        for block in self.blocks:
            x = block(x, c, context, context_mask)
        return self.final_layer(x, c)
