#include <gtest/gtest.h>
#include <string>
#include "transformer_engine/fused_attn.h"

namespace te = transformer_engine;

TEST(FusedAttnQKVLayoutTest, BHSD_BHSD_BHSD_ToString) {
  EXPECT_EQ(te::to_string(NVTE_BHSD_BHSD_BHSD), "NVTE_BHSD_BHSD_BHSD");
}
