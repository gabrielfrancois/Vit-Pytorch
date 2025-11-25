# tests/test_predictor_lg.py
import torch
from torch import Tensor
from models.predictor_lg import PredictorLG


def test_predictor_lg_output_shape() -> None:
    """
    Test that PredictorLG returns a tensor of shape (B, N, 2) for given inputs.
    """
    batch_size = 2
    n_tokens = 5
    embed_dim = 32

    x: Tensor = torch.randn(batch_size, n_tokens, embed_dim)
    policy: Tensor = torch.ones(batch_size, n_tokens)

    model = PredictorLG(embed_dim=embed_dim)
    out: Tensor = model(x, policy)

    assert out.shape == (batch_size, n_tokens, 2), (
        f"Expected shape {(batch_size, n_tokens, 2)}, got {out.shape}"
    )


def test_predictor_lg_forward_runs_without_error() -> None:
    """
    Ensure forward pass runs without exceptions.
    """
    x: Tensor = torch.randn(1, 10, 32)
    policy: Tensor = torch.ones(1, 10)
    model = PredictorLG(embed_dim=32)
    _ = model(x, policy)


def test_predictor_lg_gradients() -> None:
    """
    Test that PredictorLG parameters receive gradients during backpropagation.
    """
    batch_size = 2
    n_tokens = 5
    embed_dim = 32

    x: Tensor = torch.randn(batch_size, n_tokens, embed_dim, requires_grad=True)
    policy: Tensor = torch.ones(batch_size, n_tokens)

    model = PredictorLG(embed_dim=embed_dim)
    out: Tensor = model(x, policy)

    # Use a simple loss
    loss: Tensor = out.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"


def test_predictor_lg_policy_mask_effect() -> None:
    """
    Test that the policy mask actually affects the global feature computation.
    """
    batch_size = 1
    n_tokens = 4
    embed_dim = 8

    x: Tensor = torch.randn(batch_size, n_tokens, embed_dim)
    policy_all = torch.ones(batch_size, n_tokens)
    policy_half = torch.tensor([[1, 0, 1, 0]], dtype=torch.float32)

    model = PredictorLG(embed_dim=embed_dim)
    out_all: Tensor = model(x, policy_all)
    out_half: Tensor = model(x, policy_half)

    # Outputs should not be identical if the mask is different
    assert not torch.allclose(out_all, out_half), "Outputs are identical despite different masks"


def test_predictor_lg_no_nan_inf() -> None:
    """
    Ensure that the output of PredictorLG does not contain NaN or Inf values.
    """
    batch_size = 2
    n_tokens = 5
    embed_dim = 32

    x: Tensor = torch.randn(batch_size, n_tokens, embed_dim)
    policy: Tensor = torch.ones(batch_size, n_tokens)

    model = PredictorLG(embed_dim=embed_dim)
    out: Tensor = model(x, policy)

    assert not torch.isnan(out).any(), "Output contains NaN values"
    assert not torch.isinf(out).any(), "Output contains Inf values"
