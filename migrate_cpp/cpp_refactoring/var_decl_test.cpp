// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "migrate_cpp/cpp_refactoring/var_decl.h"

#include "migrate_cpp/cpp_refactoring/matcher_test_base.h"

namespace Carbon {
namespace {

class VarDeclTest : public MatcherTestBase {
 protected:
  VarDeclTest() : var_decl(replacements, &finder) {}

  Carbon::VarDecl var_decl;
};

TEST_F(VarDeclTest, Declaration) {
  constexpr char Before[] = "int i;";
  constexpr char After[] = "var i: int;";
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, Assignment) {
  constexpr char Before[] = "int i = 0;";
  constexpr char After[] = "var i: int;";
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, Const) {
  // TODO: Handle const appropriately.
  constexpr char Before[] = "const int i = 0;";
  constexpr char After[] = "var i: const int;";
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, Params) {
  constexpr char Before[] = "auto Foo(int i) -> int;";
  constexpr char After[] = "auto Foo(i: int) -> int;";
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, ParamsDefault) {
  constexpr char Before[] = "auto Foo(int i = 0) -> int;";
  constexpr char After[] = "auto Foo(i: int) -> int;";
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, ParamsConst) {
  // TODO: Handle const appropriately.
  constexpr char Before[] = "auto Foo(const int i) -> int;";
  constexpr char After[] = "auto Foo(i: const int) -> int;";
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, Member) {
  // TODO: Handle member variables.
  constexpr char Before[] = R"cpp(
    struct Circle {
      double radius;
    };
  )cpp";
  ExpectReplacement(Before, Before);
}

}  // namespace
}  // namespace Carbon
