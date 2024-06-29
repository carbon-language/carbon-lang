// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/hashtable_key_context.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace Carbon {
namespace {

using ::testing::Eq;
using ::testing::Ne;

struct DefaultEq {
  int x, y;

  friend auto operator==(const DefaultEq& lhs, const DefaultEq& rhs)
      -> bool = default;
};

struct CustomEq {
  int x, y;

  friend auto operator==(const CustomEq& lhs, const CustomEq& rhs) -> bool {
    return lhs.x == rhs.x && lhs.y == rhs.y;
  }
};

struct CustomExtEq {
  int x, y;

  friend auto CarbonHashtableEq(const CustomExtEq& lhs, const CustomExtEq& rhs)
      -> bool {
    return lhs.x == rhs.x && lhs.y == rhs.y;
  }
};

TEST(HashtableKeyContextTest, HashtableEq) {
  EXPECT_TRUE(HashtableEq(0, 0));
  EXPECT_FALSE(HashtableEq(1, 0));
  EXPECT_FALSE(HashtableEq(0, 1));
  EXPECT_FALSE(HashtableEq(1234, 5678));
  EXPECT_TRUE(HashtableEq(5678, 5678));

  EXPECT_TRUE(
      HashtableEq(DefaultEq{.x = 0, .y = 0}, DefaultEq{.x = 0, .y = 0}));
  EXPECT_FALSE(
      HashtableEq(DefaultEq{.x = 1, .y = 2}, DefaultEq{.x = 3, .y = 4}));

  EXPECT_TRUE(HashtableEq(CustomEq{.x = 0, .y = 0}, CustomEq{.x = 0, .y = 0}));
  EXPECT_FALSE(HashtableEq(CustomEq{.x = 1, .y = 2}, CustomEq{.x = 3, .y = 4}));

  EXPECT_TRUE(
      HashtableEq(CustomExtEq{.x = 0, .y = 0}, CustomExtEq{.x = 0, .y = 0}));
  EXPECT_FALSE(
      HashtableEq(CustomExtEq{.x = 1, .y = 2}, CustomExtEq{.x = 3, .y = 4}));
}

TEST(HashtableKeyContextTest, HashtableEqAPInt) {
  // Hashtable equality doesn't assert on mismatched bit width, it includes the
  // bit width in the comparison.
  llvm::APInt one_64(/*numBits=*/64, /*val=*/1);
  llvm::APInt two_64(/*numBits=*/64, /*val=*/2);
  llvm::APInt one_128(/*numBits=*/128, /*val=*/1);
  llvm::APInt two_128(/*numBits=*/128, /*val=*/2);

  EXPECT_TRUE(HashtableEq(one_64, one_64));
  EXPECT_FALSE(HashtableEq(one_64, one_128));
  EXPECT_TRUE(HashtableEq(two_128, two_128));
  EXPECT_FALSE(HashtableEq(two_64, two_128));
  EXPECT_FALSE(HashtableEq(one_64, two_64));
  EXPECT_FALSE(HashtableEq(one_64, two_128));
  EXPECT_FALSE(HashtableEq(one_128, two_128));
  EXPECT_FALSE(HashtableEq(one_128, two_64));
}

TEST(HashtableKeyContextTest, HashtableEqAPFloat) {
  // Hashtable equality for `APFloat` uses a bitwise comparison. This
  // differentiates between various things that would otherwise not make sense:
  // - Different floating point semantics
  // - `-0.0` and `0.0`
  //
  // It also allows NaNs to be compared meaningfully.
  llvm::APFloat zero_float =
      llvm::APFloat::getZero(llvm::APFloat::IEEEsingle());
  llvm::APFloat neg_zero_float =
      llvm::APFloat::getZero(llvm::APFloat::IEEEsingle(), /*Negative=*/true);
  llvm::APFloat zero_double =
      llvm::APFloat::getZero(llvm::APFloat::IEEEdouble());
  llvm::APFloat zero_bfloat = llvm::APFloat::getZero(llvm::APFloat::BFloat());
  llvm::APFloat one_float = llvm::APFloat::getOne(llvm::APFloat::IEEEsingle());
  llvm::APFloat inf_float = llvm::APFloat::getInf(llvm::APFloat::IEEEsingle());
  llvm::APFloat nan_0_float = llvm::APFloat::getNaN(
      llvm::APFloat::IEEEsingle(), /*Negative=*/false, /*payload=*/0);
  llvm::APFloat nan_42_float = llvm::APFloat::getNaN(
      llvm::APFloat::IEEEsingle(), /*Negative=*/false, /*payload=*/42);

  // Boring cases.
  EXPECT_TRUE(HashtableEq(zero_float, zero_float));
  EXPECT_FALSE(HashtableEq(zero_float, one_float));
  EXPECT_TRUE(HashtableEq(inf_float, inf_float));
  EXPECT_FALSE(HashtableEq(inf_float, one_float));

  // Confirm a case where we expect `==` to work but produce a different result.
  ASSERT_TRUE(zero_float == neg_zero_float);
  EXPECT_FALSE(HashtableEq(zero_float, neg_zero_float));

  // Now work through less reasonable things outside of a hashtable such as
  // mixing semantics and NaNs.
  EXPECT_FALSE(HashtableEq(zero_float, zero_double));
  EXPECT_FALSE(HashtableEq(zero_float, zero_bfloat));
  EXPECT_FALSE(HashtableEq(zero_float, nan_0_float));
  EXPECT_FALSE(HashtableEq(zero_float, nan_42_float));
  EXPECT_FALSE(HashtableEq(nan_0_float, nan_42_float));
}

struct CustomHash {
  int x;

  friend auto CarbonHashValue(const CustomHash& value, uint64_t seed)
      -> HashCode {
    return HashValue(value.x + 42, seed);
  }
};

TEST(HashtableKeyContextTest, DefaultKeyContext) {
  // Make sure the default context dispatches appropriately, including for
  // interesting types. We don't cover all the cases here and use the direct
  // tests of `HashtableEq` for that.
  DefaultKeyContext context;

  EXPECT_FALSE(context.KeyEq(1234, 5678));
  EXPECT_TRUE(context.KeyEq(5678, 5678));
  EXPECT_TRUE(context.KeyEq(DefaultEq{0, 0}, DefaultEq{0, 0}));
  EXPECT_FALSE(context.KeyEq(DefaultEq{1, 2}, DefaultEq{3, 4}));
  EXPECT_TRUE(context.KeyEq(CustomEq{0, 0}, CustomEq{0, 0}));
  EXPECT_FALSE(context.KeyEq(CustomEq{1, 2}, CustomEq{3, 4}));
  EXPECT_TRUE(context.KeyEq(CustomExtEq{0, 0}, CustomExtEq{0, 0}));
  EXPECT_FALSE(context.KeyEq(CustomExtEq{1, 2}, CustomExtEq{3, 4}));

  llvm::APInt one_64(/*numBits=*/64, /*val=*/1);
  llvm::APInt one_128(/*numBits=*/128, /*val=*/1);
  EXPECT_TRUE(HashtableEq(one_64, one_64));
  EXPECT_FALSE(HashtableEq(one_64, one_128));

  llvm::APFloat zero_float =
      llvm::APFloat::getZero(llvm::APFloat::IEEEsingle());
  llvm::APFloat neg_zero_float =
      llvm::APFloat::getZero(llvm::APFloat::IEEEsingle(), /*Negative=*/true);
  EXPECT_TRUE(HashtableEq(zero_float, zero_float));
  EXPECT_FALSE(HashtableEq(zero_float, neg_zero_float));

  // Also check hash dispatching.
  uint64_t seed = 1234;
  EXPECT_THAT(context.HashKey(42, seed), Eq(HashValue(42, seed)));
  EXPECT_THAT(context.HashKey(CustomHash{.x = 1234}, seed),
              Eq(HashValue(CustomHash{.x = 1234}, seed)));
  EXPECT_THAT(context.HashKey(one_64, seed), Eq(HashValue(one_64, seed)));
  EXPECT_THAT(context.HashKey(one_128, seed), Eq(HashValue(one_128, seed)));
  EXPECT_THAT(context.HashKey(one_64, seed),
              Ne(context.HashKey(one_128, seed)));
  EXPECT_THAT(context.HashKey(zero_float, seed),
              Eq(HashValue(zero_float, seed)));
  EXPECT_THAT(context.HashKey(neg_zero_float, seed),
              Eq(HashValue(neg_zero_float, seed)));
  EXPECT_THAT(context.HashKey(zero_float, seed),
              Ne(context.HashKey(neg_zero_float, seed)));
}

struct TestTranslatingKeyContext
    : TranslatingKeyContext<TestTranslatingKeyContext> {
  auto TranslateKey(int index) const -> const llvm::APInt& {
    return array[index];
  }

  llvm::ArrayRef<llvm::APInt> array;
};

TEST(HashtableKeyContextTest, TranslatingKeyContext) {
  llvm::APInt one_64(/*numBits=*/64, /*val=*/1);
  llvm::APInt two_64(/*numBits=*/64, /*val=*/2);
  llvm::APInt one_128(/*numBits=*/128, /*val=*/1);
  llvm::APInt two_128(/*numBits=*/128, /*val=*/2);
  // An array of values, including some duplicates.
  llvm::SmallVector<llvm::APInt> values = {one_64,  two_64, one_128,
                                           two_128, one_64, one_64};

  TestTranslatingKeyContext context = {.array = values};

  uint64_t seed = 1234;
  EXPECT_THAT(context.HashKey(0, seed), Eq(HashValue(one_64, seed)));
  EXPECT_THAT(context.HashKey(1, seed), Eq(HashValue(two_64, seed)));
  EXPECT_THAT(context.HashKey(2, seed), Eq(HashValue(one_128, seed)));
  EXPECT_THAT(context.HashKey(3, seed), Eq(HashValue(two_128, seed)));
  EXPECT_THAT(context.HashKey(4, seed), Eq(HashValue(one_64, seed)));
  EXPECT_THAT(context.HashKey(5, seed), Eq(HashValue(one_64, seed)));

  EXPECT_TRUE(context.KeyEq(one_64, 0));
  EXPECT_TRUE(context.KeyEq(one_64, 4));
  EXPECT_TRUE(context.KeyEq(one_64, 5));
  EXPECT_TRUE(context.KeyEq(0, one_64));
  EXPECT_TRUE(context.KeyEq(0, 0));
  EXPECT_TRUE(context.KeyEq(0, 4));
  EXPECT_TRUE(context.KeyEq(4, 5));
  EXPECT_FALSE(context.KeyEq(one_64, 1));
  EXPECT_FALSE(context.KeyEq(one_64, 2));
  EXPECT_FALSE(context.KeyEq(one_64, 3));
  EXPECT_FALSE(context.KeyEq(1, one_64));
  EXPECT_FALSE(context.KeyEq(2, one_64));
  EXPECT_FALSE(context.KeyEq(3, one_64));
  EXPECT_FALSE(context.KeyEq(0, 1));
  EXPECT_FALSE(context.KeyEq(0, 2));
  EXPECT_FALSE(context.KeyEq(4, 3));
}

}  // namespace
}  // namespace Carbon
