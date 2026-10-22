import flax.linen as nn
import jax
import jax.numpy as jnp


class CausalSelfAttention(nn.Module):
    d_model: int
    num_heads: int
    base_d_model: int = 64
    use_mask: bool = True
    dropout_rate: float = 0.1
    use_mup: bool = False

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        B, L, D = x.shape
        assert D % self.num_heads == 0
        head_size = D // self.num_heads

        # Calculate muP initialization scale
        width_mult = self.d_model / self.base_d_model if self.use_mup else 1.0
        init_std = 0.02 / jnp.sqrt(width_mult)
        kernel_init = nn.initializers.normal(init_std)

        qkv = nn.Dense(
            features=3 * D,
            name="c_attn",
            kernel_init=kernel_init,  # Scaled for muP
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(x)

        q, k, v = jnp.split(qkv, 3, axis=-1)

        k = k.reshape(B, L, self.num_heads, head_size)
        q = q.reshape(B, L, self.num_heads, head_size)
        v = v.reshape(B, L, self.num_heads, head_size)

        if self.use_mup:
            q = q / jnp.sqrt(head_size)

        if self.use_mask:
            mask_input = q[:, :, 0, 0]
            causal_mask = nn.make_causal_mask(mask_input)
            mask = causal_mask
        else:
            mask = None

        y = nn.dot_product_attention(q, k, v, mask=mask, broadcast_dropout=False)
        y = y.reshape(B, L, D)
        y = nn.Dense(
            features=self.d_model,
            name="c_proj",
            kernel_init=kernel_init,  # Scaled for muP
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(y)

        return y


class Block(nn.Module):
    d_model: int
    num_heads: int
    base_d_model: int = 64
    use_mask: bool = True
    dropout_rate: float = 0.1
    use_mup: bool = False

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        attn = CausalSelfAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            base_d_model=self.base_d_model,
            use_mask=self.use_mask,
            dropout_rate=self.dropout_rate,
            name="attn",
            use_mup=self.use_mup,
        )

        x_ln = nn.LayerNorm(name="ln_1", dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)
        attn_output = attn(x_ln, training=training)
        x = x + attn_output
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        x_ln = nn.LayerNorm(name="ln_2", dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)
        mlp_output = self.mlp(x_ln)

        x = x + mlp_output
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        return x

    def mlp(self, x: jax.Array) -> jax.Array:
        d_ff = 4 * self.d_model

        # Calculate muP initialization scale
        width_mult = self.d_model / self.base_d_model if self.use_mup else 1.0
        init_std = 0.02 / jnp.sqrt(width_mult)
        kernel_init = nn.initializers.normal(init_std)

        c_fc = nn.Dense(
            features=d_ff,
            name="c_fc",
            kernel_init=kernel_init,  # Scaled for muP
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )
        c_proj = nn.Dense(
            features=self.d_model,
            name="c_proj",
            kernel_init=kernel_init,  # Scaled for muP
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )

        x = c_fc(x)
        x = jax.nn.gelu(x, approximate=True)
        x = c_proj(x)
        return x


class Transformer(nn.Module):
    vocab_size: int
    d_model: int
    block_size: int
    num_layers: int
    num_heads: int
    base_d_model: int = 64
    use_mask: bool = True
    output_size: int | None = None
    dropout_rate: float = 0.1
    use_mup: bool = False
    scalar_regression: bool = False

    @nn.compact
    def __call__(self, idx: jax.Array, training: bool = False) -> jax.Array:
        width_mult = self.d_model / self.base_d_model if self.use_mup else 1.0
        init_std = 0.02 / jnp.sqrt(width_mult)
        kernel_init = nn.initializers.normal(init_std)
        final_init = kernel_init if self.scalar_regression else nn.initializers.zeros

        output_size = self.output_size or self.vocab_size
        _, L = idx.shape

        msg = f"Sequence length {L} exceeds block size {self.block_size}"
        assert L <= self.block_size, msg

        wte = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            name="wte",
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )
        tok_emb = wte(idx)

        wpe = nn.Embed(
            num_embeddings=self.block_size,
            features=self.d_model,
            name="wpe",
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )
        pos = jnp.arange(0, L)
        pos_emb = wpe(pos)

        # Combine embeddings.
        x = tok_emb + pos_emb  # Broadcasting (B, L, D) + (L, D) -> (B, L, D)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        for i in range(self.num_layers):
            x = Block(
                d_model=self.d_model,
                num_heads=self.num_heads,
                base_d_model=self.base_d_model,
                use_mask=self.use_mask,
                dropout_rate=self.dropout_rate,
                use_mup=self.use_mup,
                name=f"h_{i}",
            )(x, training=training)

        x = nn.LayerNorm(name="ln_f", dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)

        logits = nn.Dense(
            features=output_size,
            use_bias=False,
            name="lm_head",
            kernel_init=final_init,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(x)

        return logits


class ScalarTransformer(nn.Module):
    vocab_size: int
    d_model: int
    block_size: int  # Max sequence length.
    num_layers: int
    num_heads: int
    base_d_model: int = 64
    use_mask: bool = False
    output_size: int | None = None
    dropout_rate: float = 0.1
    use_mup: bool = False

    @nn.compact
    def __call__(self, idx: jax.Array, training: bool = False) -> jax.Array:
        width_mult = self.d_model / self.base_d_model if self.use_mup else 1.0
        init_std = 0.02 / jnp.sqrt(width_mult)
        kernel_init = nn.initializers.normal(init_std)

        # (B, L, D)
        x = Transformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            block_size=self.block_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            use_mask=self.use_mask,
            output_size=self.output_size,
            dropout_rate=self.dropout_rate,
            use_mup=self.use_mup,
            scalar_regression=True,
        )(idx, training=training)

        x = nn.LayerNorm(name="pre_pool_ln")(x)  # -> (B, L, D)
        logits = nn.Dense(
            1,
            name="attn_logits",
            kernel_init=kernel_init,
        )(x)  # -> (B, L, 1)
        weights = nn.softmax(logits, axis=1)  # -> (B, L, 1)
        pooled = jnp.sum(x * weights, axis=1)  # -> (B, D)
        pooled = nn.LayerNorm(name="post_pool_ln")(pooled)
        pooled = nn.relu(
            nn.Dense(
                x.shape[-1],
                name="hidden_projection",
                kernel_init=kernel_init,
            )(pooled)
        )
        out = nn.Dense(1, name="final_projection")(pooled)  # -> (B, 1)

        return out.squeeze(-1)  # -> (B)
