// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/int_store.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <limits>

#include "toolchain/base/value_ids.h"

namespace Carbon::Testing {

struct IntStoreTestPeer {
  static constexpr int MinAPWidth = IntStore::MinAPWidth;
};

namespace {

using ::testing::Eq;
using ::testing::Not;

static constexpr int MinAPWidth = IntStoreTestPeer::MinAPWidth;

TEST(IntStore, Basic) {
  IntStore ints;
  IntId id1 = ints.Add(1);
  IntId id2 = ints.Add(2);
  IntId id3 = ints.Add(999'999'999'999);

  ASSERT_TRUE(id1.is_valid());
  ASSERT_TRUE(id2.is_valid());
  ASSERT_TRUE(id3.is_valid());
  EXPECT_THAT(id1, Not(Eq(id2)));
  EXPECT_THAT(id1, Not(Eq(id3)));
  EXPECT_THAT(id2, Not(Eq(id3)));

  EXPECT_THAT(ints.Get(id1), Eq(1));
  EXPECT_THAT(ints.Get(id2), Eq(2));
  EXPECT_THAT(ints.Get(id3), Eq(999'999'999'999));
}

TEST(IntStore, APSigned) {
  IntStore ints;

  llvm::APInt one_ap(MinAPWidth, 1, /*isSigned=*/true);
  llvm::APInt two_ap(MinAPWidth, 2, /*isSigned=*/true);
  llvm::APInt nines_ap(MinAPWidth, 999'999'999'999, /*isSigned=*/true);
  llvm::APInt big_nines_ap = nines_ap.sext(128) * 10'000;
  llvm::APInt bigger_nines_ap = nines_ap.sext(512) * 100'000;
  llvm::APInt biggest_small_ap(
      MinAPWidth, std::numeric_limits<int32_t>::max() >> (32 - TokenIdBits),
      /*isSigned=*/true);
  llvm::APInt smallest_large_ap = biggest_small_ap + 1;
  llvm::APInt biggest_neg_large_ap(
      MinAPWidth, std::numeric_limits<int32_t>::min() >> (32 - TokenIdBits + 1),
      /*isSigned=*/true);
  llvm::APInt smallest_neg_small_ap = biggest_neg_large_ap + 1;
  IntId ids[] = {
      ints.AddSigned(one_ap),
      ints.AddSigned(two_ap),
      ints.AddSigned(nines_ap),
      ints.AddSigned(big_nines_ap),
      ints.AddSigned(bigger_nines_ap),
      ints.AddSigned(biggest_small_ap),
      ints.AddSigned(smallest_large_ap),
      ints.AddSigned(biggest_neg_large_ap),
      ints.AddSigned(smallest_neg_small_ap),
  };

  for (IntId id : ids) {
    ASSERT_TRUE(id.is_valid());
  }

  for (int i : llvm::seq<int>(std::size(ids))) {
    for (int j : llvm::seq<int>(i + 1, std::size(ids))) {
      EXPECT_THAT(ids[i], Not(Eq(ids[j])));
    }
  }

  EXPECT_THAT(ints.Get(ids[0]), Eq(1));
  EXPECT_THAT(ints.Get(ids[1]), Eq(2));
  EXPECT_THAT(ints.Get(ids[2]), Eq(999'999'999'999));
  EXPECT_THAT(ints.Get(ids[3]).sext(big_nines_ap.getBitWidth()),
              Eq(big_nines_ap));
  EXPECT_THAT(ints.Get(ids[4]).sext(bigger_nines_ap.getBitWidth()),
              Eq(bigger_nines_ap));
  EXPECT_THAT(ints.Get(ids[5]), Eq(biggest_small_ap));
  EXPECT_THAT(ints.Get(ids[6]), Eq(smallest_large_ap));
  EXPECT_THAT(ints.Get(ids[7]), Eq(biggest_neg_large_ap));
  EXPECT_THAT(ints.Get(ids[8]), Eq(smallest_neg_small_ap));
}

TEST(IntStore, APUnsigned) {
  IntStore ints;

  llvm::APInt one_ap(MinAPWidth, 1);
  llvm::APInt two_ap(MinAPWidth, 2);
  llvm::APInt nines_ap(MinAPWidth, 999'999'999'999);
  llvm::APInt max64_ap(MinAPWidth, std::numeric_limits<uint64_t>::max());
  llvm::APInt max64_plus_one_ap = max64_ap.zext(65) + 1;
  llvm::APInt big_nines_ap = nines_ap.zext(128) * 10'000;
  llvm::APInt bigger_nines_ap = nines_ap.zext(512) * 100'000;
  llvm::APInt biggest_small_ap(
      64, std::numeric_limits<int32_t>::max() >> (32 - TokenIdBits));
  llvm::APInt smallest_large_ap = biggest_small_ap + 1;
  IntId ids[] = {
      ints.AddUnsigned(one_ap),
      ints.AddUnsigned(two_ap),
      ints.AddUnsigned(nines_ap),
      ints.AddUnsigned(max64_ap),
      ints.AddUnsigned(max64_plus_one_ap),
      ints.AddUnsigned(big_nines_ap),
      ints.AddUnsigned(bigger_nines_ap),
      ints.AddUnsigned(biggest_small_ap),
      ints.AddUnsigned(smallest_large_ap),
  };

  for (IntId id : ids) {
    ASSERT_TRUE(id.is_valid());
  }

  for (int i : llvm::seq<int>(std::size(ids))) {
    for (int j : llvm::seq<int>(i + 1, std::size(ids))) {
      EXPECT_THAT(ids[i], Not(Eq(ids[j])));
    }
  }

  EXPECT_THAT(ints.Get(ids[0]), Eq(1));
  EXPECT_THAT(ints.Get(ids[1]), Eq(2));
  EXPECT_THAT(ints.Get(ids[2]), Eq(999'999'999'999));
  EXPECT_THAT(ints.Get(ids[3]).getActiveBits(), Eq(64));
  EXPECT_THAT(ints.Get(ids[3]).trunc(64),
              Eq(std::numeric_limits<uint64_t>::max()));
  EXPECT_THAT(ints.Get(ids[4]).truncUSat(max64_plus_one_ap.getBitWidth()),
              Eq(max64_plus_one_ap));
  // We have lots of extra bits in our initial AP, so we sign extend here to
  // ensure that we don't get a negative number from `Get`.
  EXPECT_THAT(ints.Get(ids[5]).sext(big_nines_ap.getBitWidth()),
              Eq(big_nines_ap));
  EXPECT_THAT(ints.Get(ids[6]).sext(bigger_nines_ap.getBitWidth()),
              Eq(bigger_nines_ap));
  EXPECT_THAT(ints.Get(ids[7]), Eq(biggest_small_ap));
  EXPECT_THAT(ints.Get(ids[8]), Eq(smallest_large_ap));
}

}  // namespace
}  // namespace Carbon::Testing
