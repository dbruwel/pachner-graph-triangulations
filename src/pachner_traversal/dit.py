import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import normal
from typing import Optional


class TimestepEmbedding(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, timesteps):
        # Sinusoidal embedding, as in DiT/DDPM
        half_dim = self.dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return nn.Dense(self.dim)(emb)


class DiTBlock(nn.Module):
    d_model: int
    num_heads: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, t_emb, training: bool = False):
        # AdaLN-Zero: adaptive LayerNorm with zero-initialized scale/shift from t_emb
        B, N, C = x.shape
        # For each block, project t_emb to 6*d_model (for 2 AdaLN: attn and mlp, each with scale and shift)
        gamma_beta = nn.Dense(6 * self.d_model, kernel_init=nn.initializers.zeros)(
            t_emb
        )
        gamma1, beta1, gamma2, beta2, gamma3, beta3 = jnp.split(gamma_beta, 6, axis=-1)
        # AdaLN for attention
        x_ln = nn.LayerNorm()(x)
        x_ln = gamma1[:, None, :] * x_ln + beta1[:, None, :]
        attn = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout_rate,
            deterministic=not training,
        )(x_ln)
        x = x + attn
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        # AdaLN for MLP
        x_ln = nn.LayerNorm()(x)
        x_ln = gamma2[:, None, :] * x_ln + beta2[:, None, :]
        mlp = nn.Dense(4 * self.d_model)(x_ln)
        mlp = nn.gelu(mlp)
        mlp = nn.Dense(self.d_model)(mlp)
        x = x + mlp
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        return x


class DiT(nn.Module):
    input_dim: int = 1296
    seq_len: int = 144
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 4
    dropout_rate: float = 0.1
    proj_dim: Optional[int] = 256

    @nn.compact
    def __call__(self, x, timesteps, training: bool = False):
        # x: [B, seq_len, input_dim]
        # timesteps: [B]
        # Optionally project input vectors to lower dim
        if self.proj_dim is not None and self.proj_dim != self.input_dim:
            x = nn.Dense(self.proj_dim, name="input_proj")(x)
        else:
            x = x
        # Embed timesteps
        t_emb = TimestepEmbedding(self.d_model)(timesteps)
        # Transformer blocks
        for i in range(self.num_layers):
            x = DiTBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                name=f"block_{i}",
            )(x, t_emb, training=training)
        x = nn.LayerNorm()(x)
        # Project back to input_dim
        if self.proj_dim is not None and self.proj_dim != self.input_dim:
            x = nn.Dense(self.input_dim, name="output_proj")(x)
        return x
