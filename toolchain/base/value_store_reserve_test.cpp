// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/value_store.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "toolchain/base/value_ids.h"

namespace Carbon::Testing {
namespace {

TEST(ValueStoreReserveTest, ReserveAndAdd) {
  // Use a ValueStore with a known type.
  ValueStore<RealId, Real> store;

  // Reserve a large number of elements.
  // With lazy reserve, this should not cause massive allocation overhead or crashes.
  constexpr int32_t ReserveCount = 10000;
  store.Reserve(ReserveCount);

  // Add elements up to the reservation.
  for (int32_t i = 0; i < ReserveCount; ++i) {
    Real r{.mantissa = llvm::APInt(64, i), .exponent = llvm::APInt(64, 0), .is_decimal = true};
    RealId id = store.Add(r);

    // Verify immediate retrieval.
    const auto& retrieved = store.Get(id);
    EXPECT_EQ(retrieved.mantissa, llvm::APInt(64, i));
  }

  // Verify size matches.
  EXPECT_EQ(store.size(), ReserveCount);
}

TEST(ValueStoreReserveTest, ReserveOverestimate) {
  ValueStore<RealId, Real> store;

  // Reserve way more than we use.
  constexpr int32_t ReserveCount = 20000;
  constexpr int32_t UseCount = 1000;
  store.Reserve(ReserveCount);

  for (int32_t i = 0; i < UseCount; ++i) {
    Real r{.mantissa = llvm::APInt(64, i), .exponent = llvm::APInt(64, 0), .is_decimal = true};
    store.Add(r);
  }

  EXPECT_EQ(store.size(), UseCount);
}

}  // namespace
}  // namespace Carbon::Testing
