// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "migrate_cpp/cpp_refactoring/for_range.h"

#include "migrate_cpp/cpp_refactoring/matcher_test_base.h"

namespace Carbon {
namespace {

class ForRangeTest : public MatcherTestBase<ForRangeFactory> {};

TEST_F(ForRangeTest, Basic) {
  constexpr char Before[] = R"cpp(
    void Foo() {
      int items[] = {1};
      for (int i : items) {
      }
    }
  )cpp";
  constexpr char After[] = R"(
    void Foo() {
      int items[] = {1};
      for (int i  in  items) {
      }
    }
  )";
  ExpectReplacement(Before, After);
}

TEST_F(ForRangeTest, NoSpace) {
  // Do not mark `cpp` so that clang-format won't "fix" the `:` spacing.
  constexpr char Before[] = R"(
    void Foo() {
      int items[] = {1};
      for (int i:items) {
      }
    }
  )";
  constexpr char After[] = R"(
    void Foo() {
      int items[] = {1};
      for (int i in items) {
      }
    }
  )";
  ExpectReplacement(Before, After);
}

}  // namespace
}  // namespace Carbon
