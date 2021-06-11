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

TEST_F(VarDeclTest, DeclarationArray) {
  constexpr char Before[] = "int i[4];";
  constexpr char After[] = "var i: int [4];";
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, DeclarationComma) {
  // TODO: Maybe replace the comma with a `;`.
  constexpr char Before[] = "int i, j;";
  constexpr char After[] = "var i: int, var j: int;";
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, Assignment) {
  constexpr char Before[] = "int i = 0;";
  // TODO: Include init.
  constexpr char After[] = "var i: int;";
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, Const) {
  // TODO: Include init.
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
  // TODO: Include init.
  constexpr char Before[] = "auto Foo(int i = 0) -> int;";
  constexpr char After[] = "auto Foo(i: int) -> int;";
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, ParamsConst) {
  constexpr char Before[] = "auto Foo(const int i) -> int;";
  constexpr char After[] = "auto Foo(i: const int) -> int;";
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, ParamStruct) {
  // This is to ensure the 'struct' keyword doesn't get added to the call type.
  constexpr char Before[] = R"cpp(
    struct Circle {};
    auto Draw(int times, const Circle& circle) -> bool;
  )cpp";
  constexpr char After[] = R"cpp(
    struct Circle {};
    auto Draw(times : int, circle : const Circle&) -> bool;
  )cpp";
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, Member) {
  // TODO: Handle member variables.
  constexpr char Before[] = R"cpp(
    struct Circle {
      Circle() : x(0), y(0), radius(1) {}

      int x;
      int y;
      int radius;
    };
  )cpp";
  ExpectReplacement(Before, Before);
}

TEST_F(VarDeclTest, RangeFor) {
  // TODO: Handle range based for loops.
  constexpr char Before[] = R"cpp(
    void Foo() {
      int items[] = {1};
      for (int i : items) {
      }
    }
  )cpp";
  constexpr char After[] = R"cpp(
    void Foo() {
      var items : int[1];
      for (int i var __begin1 : int* var __range1 : int(&)[1]) {
      }
    }
  )cpp";
  ExpectReplacement(Before, After);
}

}  // namespace
}  // namespace Carbon
