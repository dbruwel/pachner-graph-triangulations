# Contributing Guidelines

To ensure our codebase remains reproducible, readable, and highly maintainable, please adhere to the following coding standards and conventions.

---

## 1. Code Style & Comments

We generally follow standard PEP 8 guidelines for Python, with strict adherence to the following rules regarding comments:

* **Sentence Case:** All comments (both block and inline) must start with a capital letter.
* **Punctuation:** Comments must end with a proper period or appropriate punctuation, even if they are short phrases.
* **Spacing:** Inline comments must be separated from the code by at least two spaces.

```{python}
# Correct: This is a properly formatted block comment.
x = x + 1  # Correct: Inline comment with capitalization and a period.

# incorrect: lowercase and missing punctuation
x = x + 1 # incorrect spacing and no period
```

---

## 2. Tensor Dimension & Naming Conventions

To keep our deep learning architectures intuitive and easy to debug, we use strict shorthand notations for tensor dimensions across all comments, docstrings, and dimension annotations.

### Shorthand Notation
* **`B`**: Batch size
* **`L`**: Sequence length
* **`D`**: Latent / Model dimension

### Variable Naming
* Always use **`d_model`** as the explicit variable name when defining or passing the latent/hidden dimension size.

```
# Input tensor shape expected: (B, L, D)
def forward(self, x: torch.Tensor, d_model: int) -> torch.Tensor:
    # Project the input across the d_model dimension.
    ...
```

---

## 3. Docstring Standard (NumPy Style)

All public modules, classes, and functions must include docstrings formatted according to the **NumPy style guide**.

### Example

```
def transform_latent_space(x: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Apply a linear transformation to the input latent representations.

    Parameters
    ----------
    x : torch.Tensor
        The input feature tensor of shape `(B, L, D)`.
    d_model : int
        The size of the latent dimension.

    Returns
    -------
    torch.Tensor
        The transformed tensor of shape `(B, L, D)`.

    Raises
    ------
    ValueError
        If the final dimension of `x` does not match `d_model`.

    Examples
    --------
    >>> import torch
    >>> x = torch.randn(16, 64, 512)
    >>> output = transform_latent_space(x, 512)
    >>> output.shape
    torch.Size([16, 64, 512])
    """
    if x.shape[-1] != d_model:
        raise ValueError(f"Expected final dimension {d_model}, but got {x.shape[-1]}.")
    return x
```

