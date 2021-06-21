// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "migrate_cpp/cpp_refactoring/fn_inserter.h"

#include "migrate_cpp/cpp_refactoring/matcher_test_base.h"

namespace Carbon {
namespace {

class FnInserterTest : public MatcherTestBase {
 protected:
  FnInserterTest() : fn_inserter(replacements, &finder) {}

  Carbon::FnInserter fn_inserter;
};

TEST_F(FnInserterTest, TrailingReturn) {
  constexpr char Before[] = "auto A() -> int;";
  constexpr char After[] = "fn A() -> int;";
  ExpectReplacement(Before, After);
}

TEST_F(FnInserterTest, Inline) {
  // TODO: Need to re-lex tokens, this should probably be "fn inline" for now.
  constexpr char Before[] = "inline auto A() -> int;";
  constexpr char After[] = "fn auto A() -> int;";
  ExpectReplacement(Before, After);
}

TEST_F(FnInserterTest, Void) {
  // TODO: void needs to be handled.
  constexpr char Before[] = "void A();";
  ExpectReplacement(Before, Before);
}

TEST_F(FnInserterTest, Methods) {
  // TODO: void needs to be handled.
  // TODO: Need to re-lex tokens, this should probably be "fn virtual" for now.
  constexpr char Before[] = R"cpp(
    class Shape {
     public:
      virtual void Draw() = 0;
      virtual auto NumSides() -> int = 0;
    };

    class Circle : public Shape {
     public:
      void Draw() override;
      auto NumSides() -> int override;
      auto Radius() -> double { return radius_; }

     private:
      double radius_;
    };
  )cpp";
  constexpr char After[] = R"(
    class Shape {
     public:
      virtual void Draw() = 0;
      fn auto NumSides() -> int = 0;
    };

    class Circle : public Shape {
     public:
      void Draw() override;
      fn NumSides() -> int override;
      fn Radius() -> double { return radius_; }

     private:
      double radius_;
    };
  )";
  ExpectReplacement(Before, After);
}

TEST_F(FnInserterTest, LegacyReturn) {
  // Code should be migrated to trailing returns by clang-tidy, so this is okay
  // to miss.
  constexpr char Before[] = "int A();";
  ExpectReplacement(Before, Before);
}

}  // namespace
}  // namespace Carbon
