// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "migrate_cpp/cpp_refactoring/var_decl.h"

#include "migrate_cpp/cpp_refactoring/matcher_test_base.h"

namespace Carbon {
namespace {

class VarDeclTest : public MatcherTestBase<VarDeclFactory> {};

TEST_F(VarDeclTest, Declaration) {
  constexpr char Before[] = "int i;";
  constexpr char After[] = "var i: int;";
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, DeclarationArray) {
  constexpr char Before[] = "int i[4];";
  constexpr char After[] = "var i: int[4];";
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, DeclarationConstArray) {
  constexpr char Before[] = "const int i[] = {0, 1};";
  constexpr char After[] = "let i: const int[] = {0, 1};";
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, DeclarationConstPointer) {
  // TODO: Fix j replacement location.
  constexpr char Before[] = R"cpp(
    int i = 0;
    int* const j = &i;
    const int* k = &i;
  )cpp";
  constexpr char After[] = R"(
    var i: int = 0;
    int* const let j: int* const = &i;
    var k: const int* = &i;
  )";
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, DeclarationComma) {
  // TODO: Maybe replace the comma with a `;`.
  constexpr char Before[] = "int i, j;";
  constexpr char After[] = "var i: int, var j: int;";
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, DeclarationCommaAssignment) {
  // TODO: Maybe replace the comma with a `;`.
  constexpr char Before[] = "int i = 0, j = 0;";
  constexpr char After[] = "var i: int = 0, var j: int = 0;";
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, DeclarationCommaArray) {
  // TODO: Maybe replace the comma with a `;`.
  // TODO: Need to handle j's array.
  constexpr char Before[] = "int i[4], j[4];";
  constexpr char After[] = "var i: int[4], j[4];";
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, DeclarationCommaArrayAssignment) {
  // TODO: Maybe replace the comma with a `;`.
  // TODO: Need to handle j's array.
  constexpr char Before[] = "int i[] = {0}, j[] = {1};";
  constexpr char After[] = "var i: int[] = {0}, j[] = {1};";
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, DeclarationCommaPointers) {
  // TODO: Maybe replace the comma with a `;`.
  // TODO: Need to handle j's pointer.
  // constexpr char After[] = "var i: int *, var j: int *;";
  constexpr char Before[] = "int *i, *j;";
  constexpr char After[] = "var i: int*, *j;";
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, DeclarationCommaPointersAssignment) {
  // TODO: Maybe replace the comma with a `;`.
  // TODO: Need to handle j's pointer.
  // constexpr char After[] = "var i: int *, var j: int *;";
  constexpr char Before[] = "int *i = nullptr, *j = i;";
  constexpr char After[] = "var i: int* = nullptr, *j = i;";
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, Assignment) {
  constexpr char Before[] = "int i = 0;";
  constexpr char After[] = "var i: int = 0;";
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, Auto) {
  constexpr char Before[] = "auto i = 0;";
  constexpr char After[] = "var i: auto = 0;";
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, AutoRef) {
  constexpr char Before[] = R"cpp(
    auto i = 0;
    const auto& j = i;
  )cpp";
  constexpr char After[] = R"(
    var i: auto = 0;
    var j: const auto& = i;
  )";
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, Const) {
  constexpr char Before[] = "const int i = 0;";
  constexpr char After[] = "let i: const int = 0;";
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, ConstPointer) {
  constexpr char Before[] = "const int* i;";
  constexpr char After[] = "var i: const int*;";
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, Namespace) {
  constexpr char Before[] = R"cpp(
    namespace Foo {
    typedef int Bar;
    }
    Foo::Bar x;
  )cpp";
  constexpr char After[] = R"(
    namespace Foo {
    typedef int Bar;
    }
    var x: Foo::Bar;
  )";
  ExpectReplacement(Before, After);
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, Params) {
  constexpr char Before[] = "auto Foo(int i) -> int;";
  constexpr char After[] = "auto Foo(i: int) -> int;";
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, ParamsDefault) {
  constexpr char Before[] = "auto Foo(int i = 0) -> int;";
  constexpr char After[] = "auto Foo(i: int = 0) -> int;";
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, ParamsConst) {
  constexpr char Before[] = "auto Foo(const int i) -> int;";
  constexpr char After[] = "auto Foo(let i: const int) -> int;";
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, ParamStruct) {
  // This is to ensure the 'struct' keyword doesn't get added to the qualified
  // type.
  constexpr char Before[] = R"cpp(
    struct Circle {};
    auto Draw(int times, const Circle& circle) -> bool;
  )cpp";
  constexpr char After[] = R"(
    struct Circle {};
    auto Draw(times: int, circle: const Circle&) -> bool;
  )";
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

TEST_F(VarDeclTest, Constructor) {
  constexpr char Before[] = R"cpp(
    struct Index {
      Index(int i) : i(i) {}

      int i;
    };
    Index x(0);
  )cpp";
  constexpr char After[] = R"(
    struct Index {
      Index(i: int) : i(i) {}

      int i;
    };
    var x: Index(0);
  )";
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, RangeFor) {
  constexpr char Before[] = R"cpp(
    void Foo() {
      int items[] = {1};
      for (int i : items) {
        int j;
      }
    }
  )cpp";
  constexpr char After[] = R"(
    void Foo() {
      var items: int[] = {1};
      for (int i : items) {
        var j: int;
      }
    }
  )";
  ExpectReplacement(Before, After);
}

TEST_F(VarDeclTest, Template) {
  constexpr char Before[] = R"cpp(
    template <typename T>
    struct R {};

    template <typename T>
    struct S {};

    R<S<int>> x;
  )cpp";
  constexpr char After[] = R"(
    template <typename T>
    struct R {};

    template <typename T>
    struct S {};

    var x: R<S<int>>;
  )";
  ExpectReplacement(Before, After);
}

}  // namespace
}  // namespace Carbon
