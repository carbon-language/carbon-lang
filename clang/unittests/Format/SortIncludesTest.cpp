//===- unittest/Format/SortIncludesTest.cpp - Include sort unit tests -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "FormatTestUtils.h"
#include "clang/Format/Format.h"
#include "llvm/Support/Debug.h"
#include "gtest/gtest.h"

#define DEBUG_TYPE "format-test"

namespace clang {
namespace format {
namespace {

class SortIncludesTest : public ::testing::Test {
protected:
  std::string sort(llvm::StringRef Code) {
    std::vector<tooling::Range> Ranges(1, tooling::Range(0, Code.size()));
    std::string Sorted = applyAllReplacements(
        Code, sortIncludes(getLLVMStyle(), Code, Ranges, "input.cpp"));
    return applyAllReplacements(
        Sorted, reformat(getLLVMStyle(), Sorted, Ranges, "input.cpp"));
  }
};

TEST_F(SortIncludesTest, BasicSorting) {
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"\n",
            sort("#include \"a.h\"\n"
                 "#include \"c.h\"\n"
                 "#include \"b.h\"\n"));
}

TEST_F(SortIncludesTest, FixTrailingComments) {
  EXPECT_EQ("#include \"a.h\"  // comment\n"
            "#include \"bb.h\" // comment\n"
            "#include \"ccc.h\"\n",
            sort("#include \"a.h\" // comment\n"
                 "#include \"ccc.h\"\n"
                 "#include \"bb.h\" // comment\n"));
}

TEST_F(SortIncludesTest, LeadingWhitespace) {
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"\n",
            sort(" #include \"a.h\"\n"
                 "  #include \"c.h\"\n"
                 "   #include \"b.h\"\n"));
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"\n",
            sort("# include \"a.h\"\n"
                 "#  include \"c.h\"\n"
                 "#   include \"b.h\"\n"));
}

TEST_F(SortIncludesTest, GreaterInComment) {
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"b.h\" // >\n"
            "#include \"c.h\"\n",
            sort("#include \"a.h\"\n"
                 "#include \"c.h\"\n"
                 "#include \"b.h\" // >\n"));
}

TEST_F(SortIncludesTest, SortsLocallyInEachBlock) {
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"c.h\"\n"
            "\n"
            "#include \"b.h\"\n",
            sort("#include \"c.h\"\n"
                 "#include \"a.h\"\n"
                 "\n"
                 "#include \"b.h\"\n"));
}

TEST_F(SortIncludesTest, HandlesAngledIncludesAsSeparateBlocks) {
  EXPECT_EQ("#include <b.h>\n"
            "#include <d.h>\n"
            "#include \"a.h\"\n"
            "#include \"c.h\"\n",
            sort("#include <d.h>\n"
                 "#include <b.h>\n"
                 "#include \"c.h\"\n"
                 "#include \"a.h\"\n"));
}

TEST_F(SortIncludesTest, HandlesMultilineIncludes) {
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"\n",
            sort("#include \"a.h\"\n"
                 "#include \\\n"
                 "\"c.h\"\n"
                 "#include \"b.h\"\n"));
}

} // end namespace
} // end namespace format
} // end namespace clang
