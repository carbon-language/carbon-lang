// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/metaprogramming.h"

#include <gtest/gtest.h>

#include <string>

#include "llvm/Support/raw_ostream.h"

namespace Carbon::Testing {
namespace {

TEST(MetaProgrammingTest, RequiresBasic) {
  bool result = Requires<int, int>([](int a, int b) { return a + b; });
  EXPECT_TRUE(result);
}

struct TypeWithPrint {
  void Print(llvm::raw_ostream& os) const { os << "Test"; }
};

TEST(MetaProgrammingTest, RequiresPrintMethod) {
  bool result = Requires<const TypeWithPrint, llvm::raw_ostream>(
      [](auto&& t, auto&& out) -> decltype(t.Print(out)) {});
  EXPECT_TRUE(result);
}

}  // namespace
}  // namespace Carbon::Testing
