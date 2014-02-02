//===- unittest/Format/FormatTestJS.cpp - Formatting unit tests for JS ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "format-test"

#include "FormatTestUtils.h"
#include "clang/Format/Format.h"
#include "llvm/Support/Debug.h"
#include "gtest/gtest.h"

namespace clang {
namespace format {

class FormatTestJS : public ::testing::Test {
protected:
  static std::string format(llvm::StringRef Code, unsigned Offset,
                            unsigned Length, const FormatStyle &Style) {
    DEBUG(llvm::errs() << "---\n");
    DEBUG(llvm::errs() << Code << "\n\n");
    std::vector<tooling::Range> Ranges(1, tooling::Range(Offset, Length));
    tooling::Replacements Replaces = reformat(Style, Code, Ranges);
    std::string Result = applyAllReplacements(Code, Replaces);
    EXPECT_NE("", Result);
    DEBUG(llvm::errs() << "\n" << Result << "\n\n");
    return Result;
  }

  static std::string format(llvm::StringRef Code, const FormatStyle &Style) {
    return format(Code, 0, Code.size(), Style);
  }

  static FormatStyle getGoogleJSStyleWithColumns(unsigned ColumnLimit) {
    FormatStyle Style = getGoogleStyle(FormatStyle::LK_JavaScript);
    Style.ColumnLimit = ColumnLimit;
    return Style;
  }

  static void verifyFormat(
      llvm::StringRef Code,
      const FormatStyle &Style = getGoogleStyle(FormatStyle::LK_JavaScript)) {
    EXPECT_EQ(Code.str(), format(test::messUp(Code), Style));
  }
};

TEST_F(FormatTestJS, UnderstandsJavaScriptOperators) {
  verifyFormat("a == = b;");
  verifyFormat("a != = b;");

  verifyFormat("a === b;");
  verifyFormat("aaaaaaa ===\n    b;", getGoogleJSStyleWithColumns(10));
  verifyFormat("a !== b;");
  verifyFormat("aaaaaaa !==\n    b;", getGoogleJSStyleWithColumns(10));
  verifyFormat("if (a + b + c +\n"
               "        d !==\n"
               "    e + f + g)\n"
               "  q();",
               getGoogleJSStyleWithColumns(20));

  verifyFormat("a >> >= b;");

  verifyFormat("a >>> b;");
  verifyFormat("aaaaaaa >>>\n    b;", getGoogleJSStyleWithColumns(10));
  verifyFormat("a >>>= b;");
  verifyFormat("aaaaaaa >>>=\n    b;", getGoogleJSStyleWithColumns(10));
  verifyFormat("if (a + b + c +\n"
               "        d >>>\n"
               "    e + f + g)\n"
               "  q();",
               getGoogleJSStyleWithColumns(20));
  verifyFormat("var x = aaaaaaaaaa ?\n"
               "            bbbbbb :\n"
               "            ccc;",
               getGoogleJSStyleWithColumns(20));
}

TEST_F(FormatTestJS, SpacesInContainerLiterals) {
  verifyFormat("var arr = [1, 2, 3];");
  verifyFormat("var obj = {a: 1, b: 2, c: 3};");

  verifyFormat("var obj = {a: 1, b: 2, c: 3};",
               getChromiumStyle(FormatStyle::LK_JavaScript));
}

TEST_F(FormatTestJS, SingleQuoteStrings) {
  verifyFormat("this.function('', true);");
}

} // end namespace tooling
} // end namespace clang
