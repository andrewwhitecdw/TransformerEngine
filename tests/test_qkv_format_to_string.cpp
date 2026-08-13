/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <gtest/gtest.h>
#include <string>

#include "transformer_engine/fused_attn.h"

TEST(QKVFormatToString, BHSD) {
  EXPECT_EQ(transformer_engine::to_string(NVTE_QKV_Format::NVTE_BHSD), "NVTE_BHSD");
}
