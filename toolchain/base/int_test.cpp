// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/int.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <limits>

namespace Carbon::Testing {

struct IntStoreTestPeer {
  static constexpr int MinAPWidth = IntStore::MinAPWidth;

  static constexpr int32_t MaxIdEmbeddedValue = IntId::MaxValue;
  static constexpr int32_t MinIdEmbeddedValue = IntId::MinValue;
};

namespace {

using ::testing::Eq;

static constexpr int MinAPWidth = IntStoreTestPeer::MinAPWidth;

static constexpr int32_t MaxIdEmbeddedValue =
    IntStoreTestPeer::MaxIdEmbeddedValue;
static constexpr int32_t MinIdEmbeddedValue =
    IntStoreTestPeer::MinIdEmbeddedValue;

TEST(IntStore, Basic) {
  IntStore ints;
  IntId id_0 = ints.Add(0);
  IntId id_1 = ints.Add(1);
  IntId id_2 = ints.Add(2);
  IntId id_42 = ints.Add(42);
  IntId id_n1 = ints.Add(-1);
  IntId id_n42 = ints.Add(-42);
  IntId id_nines = ints.Add(999'999'999'999);
  IntId id_max64 = ints.Add(std::numeric_limits<int64_t>::max());
  IntId id_min64 = ints.Add(std::numeric_limits<int64_t>::min());

  for (IntId id :
       {id_0, id_1, id_2, id_42, id_n1, id_n42, id_nines, id_max64, id_min64}) {
    ASSERT_TRUE(id.has_value());
  }

  // Small values should be embedded.
  EXPECT_THAT(id_0.AsValue(), Eq(0));
  EXPECT_THAT(id_1.AsValue(), Eq(1));
  EXPECT_THAT(id_2.AsValue(), Eq(2));
  EXPECT_THAT(id_42.AsValue(), Eq(42));
  EXPECT_THAT(id_n1.AsValue(), Eq(-1));
  EXPECT_THAT(id_n42.AsValue(), Eq(-42));

  // Rest should be indices as they don't fit as embedded values.
  EXPECT_TRUE(!id_nines.is_embedded_value());
  EXPECT_TRUE(id_nines.is_index());
  EXPECT_TRUE(!id_max64.is_embedded_value());
  EXPECT_TRUE(id_max64.is_index());
  EXPECT_TRUE(!id_min64.is_embedded_value());
  EXPECT_TRUE(id_min64.is_index());

  // And round tripping all the way through the store should work.
  EXPECT_THAT(ints.Get(id_0), Eq(0));
  EXPECT_THAT(ints.Get(id_1), Eq(1));
  EXPECT_THAT(ints.Get(id_2), Eq(2));
  EXPECT_THAT(ints.Get(id_42), Eq(42));
  EXPECT_THAT(ints.Get(id_n1), Eq(-1));
  EXPECT_THAT(ints.Get(id_n42), Eq(-42));
  EXPECT_THAT(ints.Get(id_nines), Eq(999'999'999'999));
  EXPECT_THAT(ints.Get(id_max64), Eq(std::numeric_limits<int64_t>::max()));
  EXPECT_THAT(ints.Get(id_min64), Eq(std::numeric_limits<int64_t>::min()));
}

// Helper struct to hold test values and the resulting IDs.
struct APAndId {
  llvm::APInt ap;
  IntId id = IntId::None;
};

TEST(IntStore, APSigned) {
  IntStore ints;

  llvm::APInt big_128_ap =
      llvm::APInt(128, 0x1234'abcd'1234'abcd, /*isSigned=*/true) * 0xabcd'0000;
  llvm::APInt max_embedded_ap(MinAPWidth, MaxIdEmbeddedValue,
                              /*isSigned=*/true);
  llvm::APInt min_embedded_ap(MinAPWidth, MinIdEmbeddedValue,
                              /*isSigned=*/true);

  APAndId ap_and_ids[] = {
      {.ap = llvm::APInt(MinAPWidth, 1, /*isSigned=*/true)},
      {.ap = llvm::APInt(MinAPWidth, 2, /*isSigned=*/true)},
      {.ap = llvm::APInt(MinAPWidth, 999'999'999'999, /*isSigned=*/true)},
      {.ap = big_128_ap},
      {.ap = -big_128_ap},
      {.ap =
           big_128_ap.sext(512) * big_128_ap.sext(512) * big_128_ap.sext(512)},
      {.ap =
           -big_128_ap.sext(512) * big_128_ap.sext(512) * big_128_ap.sext(512)},
      {.ap = max_embedded_ap},
      {.ap = max_embedded_ap + 1},
      {.ap = min_embedded_ap},
      {.ap = min_embedded_ap - 1},
  };
  for (auto& [ap, id] : ap_and_ids) {
    id = ints.AddSigned(ap);
    ASSERT_TRUE(id.has_value()) << ap;
  }

  for (const auto& [ap, id] : ap_and_ids) {
    // The sign extend here may be a no-op, but the original bit width is a
    // reliable one at which to do the comparison.
    EXPECT_THAT(ints.Get(id).sext(ap.getBitWidth()), Eq(ap));
  }
}

TEST(IntStore, APUnsigned) {
  IntStore ints;

  llvm::APInt big_128_ap =
      llvm::APInt(128, 0xabcd'abcd'abcd'abcd) * 0xabcd'0000'abcd'0000;
  llvm::APInt max_embedded_ap(MinAPWidth, MaxIdEmbeddedValue);

  APAndId ap_and_ids[] = {
      {.ap = llvm::APInt(MinAPWidth, 1)},
      {.ap = llvm::APInt(MinAPWidth, 2)},
      {.ap = llvm::APInt(MinAPWidth, 999'999'999'999)},
      {.ap = llvm::APInt(MinAPWidth, std::numeric_limits<uint64_t>::max())},
      {.ap = llvm::APInt(MinAPWidth + 1, std::numeric_limits<uint64_t>::max()) +
             1},
      {.ap = big_128_ap},
      {.ap =
           big_128_ap.zext(512) * big_128_ap.zext(512) * big_128_ap.zext(512)},
      {.ap = max_embedded_ap},
      {.ap = max_embedded_ap + 1},
  };
  for (auto& [ap, id] : ap_and_ids) {
    id = ints.AddUnsigned(ap);
    ASSERT_TRUE(id.has_value()) << ap;
  }

  for (const auto& [ap, id] : ap_and_ids) {
    auto stored_ap = ints.Get(id);
    // Pick a bit width wide enough to represent both whatever is returned and
    // the original value as a *signed* integer without any truncation.
    int width = std::max(stored_ap.getBitWidth(), ap.getBitWidth() + 1);
    // We sign extend the stored value and zero extend the original number. This
    // ensures that anything added as unsigned ends up stored as a positive
    // number even when sign extended.
    EXPECT_THAT(stored_ap.sext(width), Eq(ap.zext(width)));
  }
}

}  // namespace
}  // namespace Carbon::Testing
