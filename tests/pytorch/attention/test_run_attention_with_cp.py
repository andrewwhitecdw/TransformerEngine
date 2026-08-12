# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import types

import pytest

from run_attention_with_cp import generate_input_shapes, get_tols


def test_get_tols_invalid_dtype():
    config = types.SimpleNamespace(num_heads=1, num_gqa_groups=1)
    with pytest.raises(ValueError, match=r"dtype=.* is not supported"):
        get_tols(config, "fp32")


def test_generate_input_shapes_invalid_qkv_format():
    config = types.SimpleNamespace(
        batch_size=1,
        max_seqlen_q=1,
        max_seqlen_kv=1,
        num_heads=1,
        num_gqa_groups=1,
        head_dim_qk=64,
        head_dim_v=64,
    )
    with pytest.raises(ValueError, match=r"qkv_format=.* is not supported"):
        generate_input_shapes("xyz", config, 1, "FlashAttention")
