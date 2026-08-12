import pytest
import torch
from transformer_engine.pytorch import DotProductAttention


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_softmax_offset_grad_none_in_eval():
    """Regression test: eval mode leaves softmax_offset.grad as None.

    The context-parallel test helper previously crashed here by calling
    core_attn.softmax_offset.grad.zero_() unconditionally for non-vanilla
    softmax. In eval mode requires_grad is False and no backward has run,
    so .grad must stay None.
    """
    core_attn = DotProductAttention(
        8, (64, 64), num_gqa_groups=4, softmax_type="softmax_offset"
    ).cuda().eval()
    assert not core_attn.softmax_offset.requires_grad
