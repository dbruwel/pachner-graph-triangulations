import flax.linen as nn
import jax
import jax.numpy as jnp


class MicroGNN(nn.Module):
    """The shared GNN that processes each eigenvector independently."""

    d_model: int

    @nn.compact
    def __call__(self, v, a_hat):
        # v: [N, 1], a_hat: [N, N]
        # Layer 1: Project then Aggregate
        h = nn.Dense(self.d_model, name="gnn_1")(v)
        h = a_hat @ h
        h = nn.relu(h)

        # Layer 2: Refine and Aggregate again
        h = nn.Dense(self.d_model, name="gnn_2")(h)
        h = a_hat @ h
        return h


class SignNetEncoder(nn.Module):
    d_model: int

    @nn.compact
    def __call__(self, x, dist_matrix):
        """
        x: [B, N, K] (Eigenvectors)
        dist_matrix: [B, N, N, c] (Distances)
        """
        B, N, K = x.shape

        # 1. Direct Adjacency from the global channel
        adj = (dist_matrix[:, :, :, 3] == 1).astype(jnp.float32)
        I = jnp.eye(N)[None, ...]
        adj_hat = adj + I

        # 2. Vectorize the MicroGNN module
        # Inner vmap: over the K eigenvectors
        # Outer vmap: over the B batch dimension
        # We use the class 'MicroGNN' directly inside nn.vmap

        VmappedGNN = nn.vmap(
            MicroGNN,
            variable_axes={"params": None},
            split_rngs={"params": False},
            in_axes=(0, None),  # type: ignore
        )

        BatchVmappedGNN = nn.vmap(
            VmappedGNN,
            variable_axes={"params": None},
            split_rngs={"params": False},
            in_axes=(0, 0),  # type: ignore
        )

        # 3. Instantiate and apply the vectorized GNN
        gnn = BatchVmappedGNN(d_model=self.d_model)

        # Prepare x: [B, N, K] -> [B, K, N, 1]
        x_in = jnp.transpose(x, (0, 2, 1))[..., None]

        # 4. Sign-Invariant Summation: phi(v) + phi(-v)
        # Result of gnn call: [B, K, N, d_model]
        z = gnn(x_in, adj_hat) + gnn(-x_in, adj_hat)

        # 5. Collapse K dimension: [B, N, d_model]
        return jnp.sum(z, axis=1)


class SpatialBias(nn.Module):
    max_dist: int
    num_heads: int

    @nn.compact
    def __call__(self, dist_matrix):
        # dist_matrix shape: (batch, nodes, nodes)
        # We create an embedding for each distance for each head
        dist_embed_face = nn.Embed(
            num_embeddings=self.max_dist + 1, features=self.num_heads
        )
        dist_embed_vertex = nn.Embed(
            num_embeddings=self.max_dist + 1, features=self.num_heads
        )
        dist_embed_gluing = nn.Embed(
            num_embeddings=self.max_dist + 1, features=self.num_heads
        )
        dist_embed_global = nn.Embed(
            num_embeddings=self.max_dist + 1, features=self.num_heads
        )
        # Output shape: (batch, nodes, nodes, num_heads)
        bias_face = dist_embed_face(dist_matrix[:, :, :, 0])
        bias_vertex = dist_embed_vertex(dist_matrix[:, :, :, 1])
        bias_gluing = dist_embed_gluing(dist_matrix[:, :, :, 2])
        bias_global = dist_embed_global(dist_matrix[:, :, :, 3])

        bias_face = bias_face.transpose(0, 3, 1, 2)  # (batch, num_heads, nodes, nodes)
        bias_vertex = bias_vertex.transpose(0, 3, 1, 2)
        bias_gluing = bias_gluing.transpose(0, 3, 1, 2)
        bias_global = bias_global.transpose(0, 3, 1, 2)

        bias = bias_face + bias_vertex + bias_gluing + bias_global

        return bias


class GraphomerBlock(nn.Module):
    d_model: int
    num_heads: int
    dropout_rate: float = 0.1
    max_dist: int = 10  # Maximum distance for spatial bias

    @nn.compact
    def __call__(self, x, dist_matrix, training: bool = False):
        B, N, C = x.shape
        bias = SpatialBias(max_dist=self.max_dist, num_heads=self.num_heads)(
            dist_matrix
        )

        x_ln = nn.LayerNorm()(x)
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout_rate,
            deterministic=not training,
        )(x_ln, x_ln)
        x = x + attn
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        x_ln = nn.LayerNorm()(x)
        mlp = nn.Dense(4 * self.d_model)(x_ln)
        mlp = nn.gelu(mlp)
        mlp = nn.Dense(self.d_model)(mlp)
        x = x + mlp
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        return x


class Graphomer(nn.Module):
    seq_len: int = 144
    num_layers: int = 4
    num_heads: int = 4
    dropout_rate: float = 0.1
    d_model: int = 256
    max_dist: int = 10

    @nn.compact
    def __call__(self, x, dist_matrix, training: bool = False):
        # x: [B, N, input_dim]
        # dist_matrix: [B, N, N, c]
        # Project input vectors to lower dim, deal with sign and order invariance
        x = SignNetEncoder(d_model=self.d_model, name="sign_net")(x, dist_matrix)

        # Transformer blocks
        for i in range(self.num_layers):
            x = GraphomerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                max_dist=self.max_dist,
                name=f"block_{i}",
            )(x, dist_matrix, training=training)
        x = nn.LayerNorm()(x)

        return x


class BilinearInteraction(nn.Module):
    d_model: int

    @nn.compact
    def __call__(self, x, y):
        """
        Computes x^T M y for each vector in the sequence.
        Input shapes: [B, N, D]
        Output shape: [B, N]
        """
        M = self.param(
            "M", nn.initializers.glorot_uniform(), (self.d_model, self.d_model)
        )

        return jnp.einsum("bni,ij,bmj->bnm", x, M, y)


class EdgeReconstructionGraphomer(nn.Module):
    """Classical Graph Transformer for predicting missing edges in a gluing matrix.
    Primary purpose is to test core transformer architecture without the complexities
    of diffusion.

    Pipeline:
        1. Positionally encode the (noisy) gluing matrix via spectral encoding
           (done outside the model in the training loop).
        2. DiT transformer processes the encoded node vectors.
        3. Matrix multiplication + sigmoid predicts edge probabilities from transformer output.
    """

    n_tet: int
    num_layers: int = 4
    num_heads: int = 4
    dropout_rate: float = 0.1
    d_model: int = 256
    max_dist: int = 10

    def setup(self):
        n_nodes = 12 * self.n_tet

        self.backbone = Graphomer(
            seq_len=n_nodes,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            d_model=self.d_model,
            max_dist=self.max_dist,
        )

        self.bilinear = BilinearInteraction(d_model=self.d_model)

    def __call__(self, x, dist_matrix, training: bool = False):
        """
        Args:
            x: [B, N, input_dim] positionally-encoded noisy gluing matrix.
            dist_matrix: [B, N, N, c] distance matrix.
            training: whether in training mode (dropout).

        Returns:
            logits: [B, N, N] raw edge scores before sigmoid.
            probs: [B, N, N] predicted edge probabilities after sigmoid.
        """
        # Transformer backbone
        h = self.backbone(x, dist_matrix, training=training)  # [B, N, input_dim]

        # Pairwise edge prediction
        logits = self.bilinear(h, h)  # [B, N, N]
        probs = jax.nn.sigmoid(logits)  # [B, N, N]

        return logits, probs
