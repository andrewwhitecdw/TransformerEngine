# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Unit tests for optional gradient unpacking in FusedAttnFunc."""

from transformer_engine.pytorch.attention.dot_product_attention.backends import (
    FusedAttnFunc,
)


class _FakeCtx:
    def __init__(self, attn_bias_type, softmax_type):
        self.attn_bias_type = attn_bias_type
        self.softmax_type = softmax_type


def test_unpack_optional_grads():
    """Optional gradients from fused_attn_bwd are consumed in declared order."""
    # No optional outputs
    ctx = _FakeCtx("no_bias", "vanilla")
    d_bias, d_softmax_offset = FusedAttnFunc._unpack_optional_grads(ctx, [])
    assert d_bias is None
    assert d_softmax_offset is None

    # Only d_softmax_offset
    ctx = _FakeCtx("no_bias", "not_vanilla")
    d_bias, d_softmax_offset = FusedAttnFunc._unpack_optional_grads(ctx, ["offset_grad"])
    assert d_bias is None
    assert d_softmax_offset == "offset_grad"

    # Only d_bias
    ctx = _FakeCtx("post_scale_bias", "vanilla")
    d_bias, d_softmax_offset = FusedAttnFunc._unpack_optional_grads(ctx, ["bias_grad"])
    assert d_bias == "bias_grad"
    assert d_softmax_offset is None

    # Both optional outputs
    ctx = _FakeCtx("post_scale_bias", "not_vanilla")
    d_bias, d_softmax_offset = FusedAttnFunc._unpack_optional_grads(
        ctx, ["bias_grad", "offset_grad"]
