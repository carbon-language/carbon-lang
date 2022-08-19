// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "migrate_cpp/rewriter.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Tooling.h"

namespace Carbon::Testing {
namespace {

// Represents C++ source code with at most one region enclosed in $[[...]]$ as
// an annotated range.
class Annotations {
 public:
  Annotations(llvm::StringRef annotated_source) {
    size_t index = annotated_source.find("$[[");
    if (index == llvm::StringRef::npos) {
      source_code_ = std::string(annotated_source);
      return;
    }
    start_ = index;
    end_ = annotated_source.find("]]$", index);
    CARBON_CHECK(end_ != llvm::StringRef::npos)
        << "Found `$[[` but no matching `]]$`";
    source_code_ = (llvm::Twine(annotated_source.substr(0, start_)) +
                    annotated_source.substr(start_ + 3, end_ - start_ - 3) +
                    annotated_source.substr(end_ + 3))
                       .str();
    // Update `end_` so that it is relative to the unannotated source (which
    // means three characters earlier due to the `$[[` being removed.
    end_ -= 3;
  }

  // Returns a view into the unannotated source.
  llvm::StringRef source() const { return source_code_; }

  // Returns the offsets in the file representing the annotated range if they
  // exist and `{0, std::numeric_limits<size_t>::max()}` otherwise.
  std::pair<size_t, size_t> range() const { return std::pair(start_, end_); }

 private:
  std::string source_code_;
  size_t start_ = 0;
  size_t end_ = std::numeric_limits<size_t>::max();
};

// Rewrites the `cpp_code`, return the Carbon equivalent. If the text has no
// source range annotated with $[[...]]$, the entire translation unit will be
// migrated and output. Otherwise, only the migrated output corresponding to the
// annotated range will be be output. No more than one range may be annoated at
// all.
//
// This annotation mechanism is useful in that it allows us to specifically test
// the migration associated with specific nodes even when they require some
// additional context that we do not wish to be covered by the test.
auto RewriteText(llvm::StringRef cpp_code) -> std::string {
  std::string result;

  Annotations annotated_cpp_code(cpp_code);

  bool success = clang::tooling::runToolOnCodeWithArgs(
      std::make_unique<MigrationAction>(result, annotated_cpp_code.range()),
      annotated_cpp_code.source(), {}, "test.cc", "clang-tool",
      std::make_shared<clang::PCHContainerOperations>(),
      clang::tooling::FileContentMappings());

  return success ? result : "";
}

TEST(Rewriter, BoolLiteral) {
  EXPECT_EQ(RewriteText("bool x = $[[true]]$;"), "true");
  EXPECT_EQ(RewriteText("bool x = $[[false]]$;"), "false");
}

TEST(Rewriter, IntegerLiteral) {
  EXPECT_EQ(RewriteText("int x = $[[0]]$;"), "0");
  EXPECT_EQ(RewriteText("int x = $[[1]]$;"), "1");
  EXPECT_EQ(RewriteText("int x = $[[1234]]$;"), "1234");
  EXPECT_EQ(RewriteText("int x = $[[12'34]]$;"), "12_34");
  EXPECT_EQ(RewriteText("int x = $[[12'3'4]]$;"), "12_3_4");
}

TEST(Rewriter, SingleDeclaration) {
  EXPECT_EQ(RewriteText("bool b;"), "var b: bool;\n");
  EXPECT_EQ(RewriteText("int i;"), "var i: i32;\n");

  EXPECT_EQ(RewriteText("const bool b = false;"), "let b: bool = false;\n");
  EXPECT_EQ(RewriteText("const int i = 17;"), "let i: i32 = 17;\n");

  EXPECT_EQ(RewriteText("bool const b = false;"), "let b: bool = false;\n");
  EXPECT_EQ(RewriteText("int const i = 1234;"), "let i: i32 = 1234;\n");
}

TEST(Rewriter, Pointers) {
  // TODO: Add tests for pointers-to-const when the syntax is nailed down.
  EXPECT_EQ(RewriteText("bool b;\n"
                        "$[[bool *p = &b]]$;"),
            "var p: bool* = &b");
  EXPECT_EQ(RewriteText("bool b;\n"
                        "$[[bool * const p = &b]]$;"),
            "let p: bool* = &b");

  // Pointers and non-pointers on the same DeclStmt.
  EXPECT_EQ(RewriteText("bool b, *p;\n"),
            "var b: bool;\n"
            "var p: bool*;\n");
  EXPECT_EQ(RewriteText("bool b, *p = &b;\n"),
            "var b: bool;\n"
            "var p: bool* = &b;\n");
}

TEST(Rewriter, DeclarationComma) {
  EXPECT_EQ(RewriteText("int x, y;"),
            "var x: i32;\n"
            "var y: i32;\n");
  EXPECT_EQ(RewriteText("int x = 7, y;"),
            "var x: i32 = 7;\n"
            "var y: i32;\n");
  EXPECT_EQ(RewriteText("const int x = 1, y = 2;"),
            "let x: i32 = 1;\n"
            "let y: i32 = 2;\n");
  EXPECT_EQ(RewriteText("int const x = 1234, y = 5678;"),
            "let x: i32 = 1234;\n"
            "let y: i32 = 5678;\n");
}

}  // namespace
}  // namespace Carbon::Testing
