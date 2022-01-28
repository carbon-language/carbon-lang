//===- UsingDeclarationsSorterTest.cpp - Formatting unit tests ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Format/Format.h"

#include "llvm/Support/Debug.h"
#include "gtest/gtest.h"

#define DEBUG_TYPE "using-declarations-sorter-test"

namespace clang {
namespace format {
namespace {

class UsingDeclarationsSorterTest : public ::testing::Test {
protected:
  std::string sortUsingDeclarations(llvm::StringRef Code,
                                    const std::vector<tooling::Range> &Ranges,
                                    const FormatStyle &Style = getLLVMStyle()) {
    LLVM_DEBUG(llvm::errs() << "---\n");
    LLVM_DEBUG(llvm::errs() << Code << "\n\n");
    tooling::Replacements Replaces =
        clang::format::sortUsingDeclarations(Style, Code, Ranges, "<stdin>");
    auto Result = applyAllReplacements(Code, Replaces);
    EXPECT_TRUE(static_cast<bool>(Result));
    LLVM_DEBUG(llvm::errs() << "\n" << *Result << "\n\n");
    return *Result;
  }

  std::string sortUsingDeclarations(llvm::StringRef Code,
                                    const FormatStyle &Style = getLLVMStyle()) {
    return sortUsingDeclarations(Code,
                                 /*Ranges=*/{1, tooling::Range(0, Code.size())},
                                 Style);
  }
};

TEST_F(UsingDeclarationsSorterTest, SwapsTwoConsecutiveUsingDeclarations) {
  EXPECT_EQ("using a;\n"
            "using b;",
            sortUsingDeclarations("using a;\n"
                                  "using b;"));
  EXPECT_EQ("using a;\n"
            "using aa;",
            sortUsingDeclarations("using aa;\n"
                                  "using a;"));
  EXPECT_EQ("using a;\n"
            "using ::a;",
            sortUsingDeclarations("using a;\n"
                                  "using ::a;"));

  EXPECT_EQ("using a::bcd;\n"
            "using a::cd;",
            sortUsingDeclarations("using a::cd;\n"
                                  "using a::bcd;"));

  EXPECT_EQ("using a;\n"
            "using a::a;",
            sortUsingDeclarations("using a::a;\n"
                                  "using a;"));

  EXPECT_EQ("using a::ba::aa;\n"
            "using a::bb::ccc;",
            sortUsingDeclarations("using a::bb::ccc;\n"
                                  "using a::ba::aa;"));

  EXPECT_EQ("using a;\n"
            "using typename a;",
            sortUsingDeclarations("using typename a;\n"
                                  "using a;"));

  EXPECT_EQ("using typename z;\n"
            "using typenamea;",
            sortUsingDeclarations("using typenamea;\n"
                                  "using typename z;"));

  EXPECT_EQ("using a, b;\n"
            "using aa;",
            sortUsingDeclarations("using aa;\n"
                                  "using a, b;"));
}

TEST_F(UsingDeclarationsSorterTest, UsingDeclarationOrder) {
  EXPECT_EQ("using A;\n"
            "using a;",
            sortUsingDeclarations("using A;\n"
                                  "using a;"));
  EXPECT_EQ("using a;\n"
            "using A;",
            sortUsingDeclarations("using a;\n"
                                  "using A;"));
  EXPECT_EQ("using a;\n"
            "using B;",
            sortUsingDeclarations("using B;\n"
                                  "using a;"));

  // Ignores leading '::'.
  EXPECT_EQ("using ::a;\n"
            "using A;",
            sortUsingDeclarations("using ::a;\n"
                                  "using A;"));

  EXPECT_EQ("using ::A;\n"
            "using a;",
            sortUsingDeclarations("using ::A;\n"
                                  "using a;"));

  // Sorts '_' before 'a' and 'A'.
  EXPECT_EQ("using _;\n"
            "using A;",
            sortUsingDeclarations("using A;\n"
                                  "using _;"));
  EXPECT_EQ("using _;\n"
            "using a;",
            sortUsingDeclarations("using a;\n"
                                  "using _;"));
  EXPECT_EQ("using a::_;\n"
            "using a::a;",
            sortUsingDeclarations("using a::a;\n"
                                  "using a::_;"));

  // Sorts non-namespace names before namespace names at the same level.
  EXPECT_EQ("using ::testing::_;\n"
            "using ::testing::Aardvark;\n"
            "using ::testing::kMax;\n"
            "using ::testing::Xylophone;\n"
            "using ::testing::apple::Honeycrisp;\n"
            "using ::testing::zebra::Stripes;",
            sortUsingDeclarations("using ::testing::Aardvark;\n"
                                  "using ::testing::Xylophone;\n"
                                  "using ::testing::kMax;\n"
                                  "using ::testing::_;\n"
                                  "using ::testing::apple::Honeycrisp;\n"
                                  "using ::testing::zebra::Stripes;"));
}

TEST_F(UsingDeclarationsSorterTest, SortsStably) {
  EXPECT_EQ("using a;\n"
            "using A;\n"
            "using a;\n"
            "using A;\n"
            "using a;\n"
            "using A;\n"
            "using a;\n"
            "using B;\n"
            "using b;\n"
            "using B;\n"
            "using b;\n"
            "using B;\n"
            "using b;",
            sortUsingDeclarations("using a;\n"
                                  "using B;\n"
                                  "using a;\n"
                                  "using b;\n"
                                  "using A;\n"
                                  "using a;\n"
                                  "using b;\n"
                                  "using B;\n"
                                  "using b;\n"
                                  "using A;\n"
                                  "using a;\n"
                                  "using b;\n"
                                  "using b;\n"
                                  "using B;\n"
                                  "using b;\n"
                                  "using A;\n"
                                  "using a;"));
}

TEST_F(UsingDeclarationsSorterTest, SortsMultipleTopLevelDeclarations) {
  EXPECT_EQ("using a;\n"
            "using b;\n"
            "using c;\n"
            "using d;\n"
            "using e;",
            sortUsingDeclarations("using d;\n"
                                  "using b;\n"
                                  "using e;\n"
                                  "using a;\n"
                                  "using c;"));

  EXPECT_EQ("#include <iostream>\n"
            "using std::cin;\n"
            "using std::cout;\n"
            "using ::std::endl;\n"
            "int main();",
            sortUsingDeclarations("#include <iostream>\n"
                                  "using std::cout;\n"
                                  "using ::std::endl;\n"
                                  "using std::cin;\n"
                                  "int main();"));
}

TEST_F(UsingDeclarationsSorterTest, BreaksOnEmptyLines) {
  EXPECT_EQ("using b;\n"
            "using c;\n"
            "\n"
            "using a;\n"
            "using d;",
            sortUsingDeclarations("using c;\n"
                                  "using b;\n"
                                  "\n"
                                  "using d;\n"
                                  "using a;"));
}

TEST_F(UsingDeclarationsSorterTest, BreaksOnUsingNamespace) {
  EXPECT_EQ("using b;\n"
            "using namespace std;\n"
            "using a;",
            sortUsingDeclarations("using b;\n"
                                  "using namespace std;\n"
                                  "using a;"));
}

TEST_F(UsingDeclarationsSorterTest, KeepsUsingDeclarationsInPPDirectives) {
  EXPECT_EQ("#define A \\\n"
            "using b;\\\n"
            "using a;",
            sortUsingDeclarations("#define A \\\n"
                                  "using b;\\\n"
                                  "using a;"));
}

TEST_F(UsingDeclarationsSorterTest, KeepsTypeAliases) {
  auto Code = "struct C { struct B { struct A; }; };\n"
              "using B = C::B;\n"
              "using A = B::A;";
  EXPECT_EQ(Code, sortUsingDeclarations(Code));
}

TEST_F(UsingDeclarationsSorterTest, MovesTrailingCommentsWithDeclarations) {
  EXPECT_EQ("using a; // line a1\n"
            "using b; /* line b1\n"
            "          * line b2\n"
            "          * line b3 */\n"
            "using c; // line c1\n"
            "         // line c2",
            sortUsingDeclarations("using c; // line c1\n"
                                  "         // line c2\n"
                                  "using b; /* line b1\n"
                                  "          * line b2\n"
                                  "          * line b3 */\n"
                                  "using a; // line a1"));
}

TEST_F(UsingDeclarationsSorterTest, SortsInStructScope) {
  EXPECT_EQ("struct pt3 : pt2 {\n"
            "  using pt2::x;\n"
            "  using pt2::y;\n"
            "  float z;\n"
            "};",
            sortUsingDeclarations("struct pt3 : pt2 {\n"
                                  "  using pt2::y;\n"
                                  "  using pt2::x;\n"
                                  "  float z;\n"
                                  "};"));
}

TEST_F(UsingDeclarationsSorterTest, KeepsOperators) {
  EXPECT_EQ("using a::operator();\n"
            "using a::operator-;\n"
            "using a::operator+;",
            sortUsingDeclarations("using a::operator();\n"
                                  "using a::operator-;\n"
                                  "using a::operator+;"));
}

TEST_F(UsingDeclarationsSorterTest, SortsUsingDeclarationsInsideNamespaces) {
  EXPECT_EQ("namespace A {\n"
            "struct B;\n"
            "struct C;\n"
            "}\n"
            "namespace X {\n"
            "using A::B;\n"
            "using A::C;\n"
            "}",
            sortUsingDeclarations("namespace A {\n"
                                  "struct B;\n"
                                  "struct C;\n"
                                  "}\n"
                                  "namespace X {\n"
                                  "using A::C;\n"
                                  "using A::B;\n"
                                  "}"));
}

TEST_F(UsingDeclarationsSorterTest, SupportsClangFormatOff) {
  EXPECT_EQ("// clang-format off\n"
            "using b;\n"
            "using a;\n"
            "// clang-format on\n"
            "using c;\n"
            "using d;",
            sortUsingDeclarations("// clang-format off\n"
                                  "using b;\n"
                                  "using a;\n"
                                  "// clang-format on\n"
                                  "using d;\n"
                                  "using c;"));
}

TEST_F(UsingDeclarationsSorterTest, SortsPartialRangeOfUsingDeclarations) {
  // Sorts the whole block of using declarations surrounding the range.
  EXPECT_EQ("using a;\n"
            "using b;\n"
            "using c;",
            sortUsingDeclarations("using b;\n"
                                  "using c;\n" // starts at offset 10
                                  "using a;",
                                  {tooling::Range(10, 15)}));
  EXPECT_EQ("using a;\n"
            "using b;\n"
            "using c;\n"
            "using A = b;",
            sortUsingDeclarations("using b;\n"
                                  "using c;\n" // starts at offset 10
                                  "using a;\n"
                                  "using A = b;",
                                  {tooling::Range(10, 15)}));

  EXPECT_EQ("using d;\n"
            "using c;\n"
            "\n"
            "using a;\n"
            "using b;\n"
            "\n"
            "using f;\n"
            "using e;",
            sortUsingDeclarations("using d;\n"
                                  "using c;\n"
                                  "\n"
                                  "using b;\n" // starts at offset 19
                                  "using a;\n"
                                  "\n"
                                  "using f;\n"
                                  "using e;",
                                  {tooling::Range(19, 1)}));
}

TEST_F(UsingDeclarationsSorterTest,
       SortsUsingDeclarationsWithLeadingkComments) {
  EXPECT_EQ("/* comment */ using a;\n"
            "/* comment */ using b;",
            sortUsingDeclarations("/* comment */ using b;\n"
                                  "/* comment */ using a;"));
}

TEST_F(UsingDeclarationsSorterTest, DeduplicatesUsingDeclarations) {
  EXPECT_EQ("using a;\n"
            "using b;\n"
            "using c;\n"
            "\n"
            "using a;\n"
            "using e;",
            sortUsingDeclarations("using c;\n"
                                  "using a;\n"
                                  "using b;\n"
                                  "using a;\n"
                                  "using b;\n"
                                  "\n"
                                  "using e;\n"
                                  "using a;\n"
                                  "using e;"));
}

} // end namespace
} // end namespace format
} // end namespace clang
