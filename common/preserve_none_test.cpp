// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/preserve_none.h"

#include <gtest/gtest.h>

#include <array>

namespace Carbon {
namespace {

using DispatchFn = auto(int value, int remaining) CARBON_PRESERVE_NONE->int;

CARBON_PRESERVE_NONE static auto DispatchDone(int value, int /*remaining*/)
    -> int {
  return value;
}

CARBON_PRESERVE_NONE static auto DispatchStep(int value, int remaining) -> int;

constexpr std::array<DispatchFn*, 2> DispatchTable = {
    &DispatchDone,
    &DispatchStep,
};

CARBON_PRESERVE_NONE static auto DispatchStep(int value, int remaining) -> int {
  // Touch enough temporaries that register allocation exercises caller-saved
  // registers under `preserve_none`.
  int next = (value * 3 + 1) ^ (remaining * 7);
  int next_remaining = remaining - 1;
  [[clang::musttail]] return DispatchTable[next_remaining != 0 ? 1 : 0](
      next, next_remaining);
}

TEST(PreserveNoneTest, TailCallDispatchWithAlignedCallerStack) {
  // Overalign a local buffer so sanitizer stack instrumentation exercises stack
  // realignment alongside `CARBON_PRESERVE_NONE` calls.
  alignas(64) std::array<int, 8> scratch = {1, 2, 3, 4, 5, 6, 7, 8};
  DispatchFn* entry = &DispatchStep;
  int result = entry(scratch[0], 16);
  EXPECT_EQ(scratch[7], 8);
  EXPECT_NE(result, 0);
}

}  // namespace
}  // namespace Carbon
