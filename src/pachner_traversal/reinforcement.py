from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .transformer import generate_samples
from .data_io import Dataset, Encoder


def get_scores(samples: np.ndarray, dataset: Dataset, encoder: Encoder) -> np.ndarray:
    samps_str = encoder.decode(np.array(samples))
    score = np.array([int(samp in dataset) for samp in samps_str])
    return score


@partial(jax.jit, static_argnames=["dataset", "encoder"])
def jax_get_scores(samples: np.ndarray, dataset: Dataset, encoder: Encoder) -> jax.Array:
    def wrapped_get_scores(samples):
        return get_scores(samples, dataset, encoder)

    score = jax.pure_callback(
        wrapped_get_scores, jax.ShapeDtypeStruct((len(samples),), jnp.int32), samples
    )

    return score


@partial(
    jax.jit, static_argnames=["samps_to_gen", "seq_len", "bos_id", "dataset", "encoder"]
)
def reinforce_update_samps(state, key, samps_to_gen, seq_len, bos_id, dataset, encoder):
    key, newkey = jax.random.split(key)

    samples = generate_samples(state, samps_to_gen, seq_len, newkey, bos_id)
    scores = jax_get_scores(samples, dataset, encoder)

    return samples, scores, newkey


@jax.jit
def reinforce_grad_step(state, samples, scores):
    dropout_key, new_dropout_key = jax.random.split(state.dropout_key)

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params}, samples, training=True, rngs={"dropout": dropout_key}
        )
        one_hot_labels = jax.nn.one_hot(samples, num_classes=logits.shape[-1])
        log_probs = jax.nn.log_softmax(logits)
        log_prob_of_sample = (log_probs[:, :-1, :] * one_hot_labels[:, 1:, :]).sum(
            axis=(1, 2)
        )

        ent = -jnp.sum(log_probs * jnp.exp(log_probs), axis=(-1, -2))
        avg_ent = ent.mean()

        loss = -jnp.mean(log_prob_of_sample * scores) - 0.04 * avg_ent

        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)

    new_state = state.apply_gradients(grads=grads)

    new_state = new_state.replace(dropout_key=new_dropout_key)

    return new_state, loss


@partial(
    jax.jit, static_argnames=["samps_to_gen", "seq_len", "bos_id", "dataset", "encoder"]
)
def train_step(
    itt, reinforce_state, key, samps_to_gen, seq_len, bos_id, dataset, encoder
):
    samples, scores, key = reinforce_update_samps(
        reinforce_state, key, samps_to_gen, seq_len, bos_id, dataset, encoder
    )
    reinforce_state, loss = reinforce_grad_step(reinforce_state, samples, scores)

    jax.debug.print("------------------------ {itt} ------------------------", itt=itt)
    jax.debug.print("Mean loss: {loss}", loss=loss)
    jax.debug.print("Mean score: {score}", score=jnp.mean(scores))

    return reinforce_state, key, loss, jnp.mean(scores)
