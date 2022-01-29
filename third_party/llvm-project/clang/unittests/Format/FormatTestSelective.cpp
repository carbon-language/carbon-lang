//===- unittest/Format/FormatTestSelective.cpp - Formatting unit tests ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

class FormatTestSelective : public ::testing::Test {
protected:
  std::string format(llvm::StringRef Code, unsigned Offset, unsigned Length) {
    LLVM_DEBUG(llvm::errs() << "---\n");
    LLVM_DEBUG(llvm::errs() << Code << "\n\n");
    std::vector<tooling::Range> Ranges(1, tooling::Range(Offset, Length));
    FormattingAttemptStatus Status;
    tooling::Replacements Replaces =
        reformat(Style, Code, Ranges, "<stdin>", &Status);
    EXPECT_TRUE(Status.FormatComplete) << Code << "\n\n";
    auto Result = applyAllReplacements(Code, Replaces);
    EXPECT_TRUE(static_cast<bool>(Result));
    LLVM_DEBUG(llvm::errs() << "\n" << *Result << "\n\n");
    return *Result;
  }

  FormatStyle Style = getLLVMStyle();
};

TEST_F(FormatTestSelective, RemovesTrailingWhitespaceOfFormattedLine) {
  EXPECT_EQ("int a;\nint b;", format("int a; \nint b;", 0, 0));
  EXPECT_EQ("int a;", format("int a;         ", 0, 0));
  EXPECT_EQ("int a;\n", format("int a;  \n   \n   \n ", 0, 0));
  EXPECT_EQ("int a;\nint b;    ", format("int a;  \nint b;    ", 0, 0));
}

TEST_F(FormatTestSelective, FormatsCorrectRegionForLeadingWhitespace) {
  EXPECT_EQ("{int b;\n"
            "  int a;\n"
            "}",
            format("{int b;\n  int  a;}", 8, 0));
  EXPECT_EQ("{\n"
            "  int b;\n"
            "  int  a;}",
            format("{int b;\n  int  a;}", 7, 0));

  Style.ColumnLimit = 12;
  EXPECT_EQ("#define A  \\\n"
            "  int a;   \\\n"
            "  int b;",
            format("#define A  \\\n"
                   "  int a;   \\\n"
                   "    int b;",
                   26, 0));
  EXPECT_EQ("#define A  \\\n"
            "  int a;   \\\n"
            "  int b;",
            format("#define A  \\\n"
                   "  int a;   \\\n"
                   "  int b;",
                   25, 0));
}

TEST_F(FormatTestSelective, FormatLineWhenInvokedOnTrailingNewline) {
  EXPECT_EQ("int  b;\n\nint a;", format("int  b;\n\nint a;", 8, 0));
  EXPECT_EQ("int b;\n\nint a;", format("int  b;\n\nint a;", 7, 0));

  // This might not strictly be correct, but is likely good in all practical
  // cases.
  EXPECT_EQ("int b;\nint a;", format("int  b;int a;", 7, 0));
}

TEST_F(FormatTestSelective, RemovesWhitespaceWhenTriggeredOnEmptyLine) {
  EXPECT_EQ("int  a;\n\n int b;", format("int  a;\n  \n\n int b;", 8, 0));
  EXPECT_EQ("int  a;\n\n int b;", format("int  a;\n  \n\n int b;", 9, 0));
}

TEST_F(FormatTestSelective, ReformatsMovedLines) {
  EXPECT_EQ(
      "template <typename T> T *getFETokenInfo() const {\n"
      "  return static_cast<T *>(FETokenInfo);\n"
      "}\n"
      "int  a; // <- Should not be formatted",
      format(
          "template<typename T>\n"
          "T *getFETokenInfo() const { return static_cast<T*>(FETokenInfo); }\n"
          "int  a; // <- Should not be formatted",
          9, 5));
}

TEST_F(FormatTestSelective, FormatsIfWithoutCompoundStatement) {
  Style.AllowShortIfStatementsOnASingleLine = FormatStyle::SIS_WithoutElse;
  EXPECT_EQ("if (a) return;", format("if(a)\nreturn;", 7, 1));
  EXPECT_EQ("if (a) return; // comment",
            format("if(a)\nreturn; // comment", 20, 1));
}

TEST_F(FormatTestSelective, FormatsCommentsLocally) {
  EXPECT_EQ("int a;    // comment\n"
            "int    b; // comment",
            format("int   a; // comment\n"
                   "int    b; // comment",
                   0, 0));
  EXPECT_EQ("int a; // comment\n"
            "       // line 2\n"
            "int b;",
            format("int   a; // comment\n"
                   "            // line 2\n"
                   "int b;",
                   28, 0));
  EXPECT_EQ("int   a; // comment\n"
            "// comment 2\n"
            "int b;",
            format("int   a; // comment\n"
                   "// comment 2\n"
                   "int b;",
                   28, 0));
  EXPECT_EQ("int aaaaaa; // comment\n"
            "int b;\n"
            "int c; // unrelated comment",
            format("int aaaaaa; // comment\n"
                   "int b;\n"
                   "int   c; // unrelated comment",
                   31, 0));

  EXPECT_EQ("int a; // This\n"
            "       // is\n"
            "       // a",
            format("int a;      // This\n"
                   "            // is\n"
                   "            // a",
                   0, 0));
  EXPECT_EQ("int a; // This\n"
            "       // is\n"
            "       // a\n"
            "// This is b\n"
            "int b;",
            format("int a; // This\n"
                   "     // is\n"
                   "     // a\n"
                   "// This is b\n"
                   "int b;",
                   0, 0));
  EXPECT_EQ("int a; // This\n"
            "       // is\n"
            "       // a\n"
            "\n"
            "//This is unrelated",
            format("int a; // This\n"
                   "     // is\n"
                   "     // a\n"
                   "\n"
                   "//This is unrelated",
                   0, 0));
  EXPECT_EQ("int a;\n"
            "// This is\n"
            "// not formatted.   ",
            format("int a;\n"
                   "// This is\n"
                   "// not formatted.   ",
                   0, 0));
  EXPECT_EQ("int x;  // Format this line.\n"
            "int xx; //\n"
            "int xxxxx; //",
            format("int x; // Format this line.\n"
                   "int xx; //\n"
                   "int xxxxx; //",
                   0, 0));
}

TEST_F(FormatTestSelective, ContinueReindenting) {
  // When we change an indent, we continue formatting as long as following
  // lines are not indented correctly.
  EXPECT_EQ("int   i;\n"
            "int b;\n"
            "int c;\n"
            "int d;\n"
            "int e;\n"
            "  int f;\n",
            format("int   i;\n"
                   "  int b;\n"
                   " int   c;\n"
                   "  int d;\n"
                   "int e;\n"
                   "  int f;\n",
                   11, 0));
}

TEST_F(FormatTestSelective, ReindentClosingBrace) {
  EXPECT_EQ("int   i;\n"
            "int f() {\n"
            "  int a;\n"
            "  int b;\n"
            "}\n"
            " int c;\n",
            format("int   i;\n"
                   "  int f(){\n"
                   "int a;\n"
                   "int b;\n"
                   "  }\n"
                   " int c;\n",
                   11, 0));
  EXPECT_EQ("void f() {\n"
            "  if (foo) {\n"
            "    b();\n"
            "  } else {\n"
            "    c();\n"
            "  }\n"
            "int d;\n"
            "}\n",
            format("void f() {\n"
                   "  if (foo) {\n"
                   "b();\n"
                   "}else{\n"
                   "c();\n"
                   "}\n"
                   "int d;\n"
                   "}\n",
                   13, 0));
  EXPECT_EQ("int i = []() {\n"
            "  class C {\n"
            "    int a;\n"
            "    int b;\n"
            "  };\n"
            "  int c;\n"
            "};\n",
            format("int i = []() {\n"
                   "  class C{\n"
                   "int a;\n"
                   "int b;\n"
                   "};\n"
                   "int c;\n"
                   "  };\n",
                   17, 0));
}

TEST_F(FormatTestSelective, IndividualStatementsOfNestedBlocks) {
  EXPECT_EQ("DEBUG({\n"
            "  int i;\n"
            "  int        j;\n"
            "});",
            format("DEBUG(   {\n"
                   "  int        i;\n"
                   "  int        j;\n"
                   "}   )  ;",
                   20, 1));
  EXPECT_EQ("DEBUG(   {\n"
            "  int        i;\n"
            "  int j;\n"
            "}   )  ;",
            format("DEBUG(   {\n"
                   "  int        i;\n"
                   "  int        j;\n"
                   "}   )  ;",
                   41, 1));
  EXPECT_EQ("DEBUG(   {\n"
            "    int        i;\n"
            "    int j;\n"
            "}   )  ;",
            format("DEBUG(   {\n"
                   "    int        i;\n"
                   "    int        j;\n"
                   "}   )  ;",
                   41, 1));
  EXPECT_EQ("DEBUG({\n"
            "  int i;\n"
            "  int j;\n"
            "});",
            format("DEBUG(   {\n"
                   "    int        i;\n"
                   "    int        j;\n"
                   "}   )  ;",
                   20, 1));

  EXPECT_EQ("Debug({\n"
            "        if (aaaaaaaaaaaaaaaaaaaaaaaa)\n"
            "          return;\n"
            "      },\n"
            "      a);",
            format("Debug({\n"
                   "        if (aaaaaaaaaaaaaaaaaaaaaaaa)\n"
                   "             return;\n"
                   "      },\n"
                   "      a);",
                   50, 1));
  EXPECT_EQ("DEBUG({\n"
            "  DEBUG({\n"
            "    int a;\n"
            "    int b;\n"
            "  }) ;\n"
            "});",
            format("DEBUG({\n"
                   "  DEBUG({\n"
                   "    int a;\n"
                   "    int    b;\n" // Format this line only.
                   "  }) ;\n"        // Don't touch this line.
                   "});",
                   35, 0));
  EXPECT_EQ("DEBUG({\n"
            "  int a; //\n"
            "});",
            format("DEBUG({\n"
                   "    int a; //\n"
                   "});",
                   0, 0));
  EXPECT_EQ("someFunction(\n"
            "    [] {\n"
            "      // Only with this comment.\n"
            "      int i; // invoke formatting here.\n"
            "    }, // force line break\n"
            "    aaa);",
            format("someFunction(\n"
                   "    [] {\n"
                   "      // Only with this comment.\n"
                   "      int   i; // invoke formatting here.\n"
                   "    }, // force line break\n"
                   "    aaa);",
                   63, 1));

  EXPECT_EQ("int longlongname; // comment\n"
            "int x = f({\n"
            "  int x; // comment\n"
            "  int y; // comment\n"
            "});",
            format("int longlongname; // comment\n"
                   "int x = f({\n"
                   "  int x; // comment\n"
                   "  int y; // comment\n"
                   "});",
                   65, 0));
  EXPECT_EQ("int s = f({\n"
            "  class X {\n"
            "  public:\n"
            "    void f();\n"
            "  };\n"
            "});",
            format("int s = f({\n"
                   "  class X {\n"
                   "    public:\n"
                   "    void f();\n"
                   "  };\n"
                   "});",
                   0, 0));
  EXPECT_EQ("SomeFunction(\n"
            "    [] {\n"
            "      int i;\n"
            "      return i;\n" // Format this line.
            "    },\n"
            "    [] {\n"
            "       return 2;\n" // Don't fix this.
            "    });",
            format("SomeFunction(\n"
                   "    [] {\n"
                   "      int i;\n"
                   "       return i;\n" // Format this line.
                   "    },\n"
                   "    [] {\n"
                   "       return 2;\n" // Don't fix this.
                   "    });",
                   40, 0));
}

TEST_F(FormatTestSelective, WrongIndent) {
  EXPECT_EQ("namespace {\n"
            "int i;\n"
            "int j;\n"
            "}",
            format("namespace {\n"
                   "  int i;\n" // Format here.
                   "  int j;\n"
                   "}",
                   15, 0));
  EXPECT_EQ("namespace {\n"
            "  int i;\n"
            "  int j;\n"
            "}",
            format("namespace {\n"
                   "  int i;\n"
                   "  int j;\n" // Format here.
                   "}",
                   24, 0));
}

TEST_F(FormatTestSelective, AlwaysFormatsEntireMacroDefinitions) {
  Style.AlignEscapedNewlines = FormatStyle::ENAS_Left;
  EXPECT_EQ("int  i;\n"
            "#define A \\\n"
            "  int i;  \\\n"
            "  int j\n"
            "int  k;",
            format("int  i;\n"
                   "#define A  \\\n"
                   " int   i    ;  \\\n"
                   " int   j\n"
                   "int  k;",
                   8, 0)); // 8: position of "#define".
  EXPECT_EQ("int  i;\n"
            "#define A \\\n"
            "  int i;  \\\n"
            "  int j\n"
            "int  k;",
            format("int  i;\n"
                   "#define A  \\\n"
                   " int   i    ;  \\\n"
                   " int   j\n"
                   "int  k;",
                   45, 0)); // 45: position of "j".
}

TEST_F(FormatTestSelective, ReformatRegionAdjustsIndent) {
  EXPECT_EQ("{\n"
            "{\n"
            "a;\n"
            "b;\n"
            "}\n"
            "}",
            format("{\n"
                   "{\n"
                   "a;\n"
                   "     b;\n"
                   "}\n"
                   "}",
                   13, 2));
  EXPECT_EQ("{\n"
            "{\n"
            "  a;\n"
            "  b;\n"
            "  c;\n"
            " d;\n"
            "}\n"
            "}",
            format("{\n"
                   "{\n"
                   "     a;\n"
                   "   b;\n"
                   "  c;\n"
                   " d;\n"
                   "}\n"
                   "}",
                   9, 2));
  EXPECT_EQ("{\n"
            "{\n"
            "public:\n"
            "  b;\n"
            "}\n"
            "}",
            format("{\n"
                   "{\n"
                   "public:\n"
                   "     b;\n"
                   "}\n"
                   "}",
                   17, 2));
  EXPECT_EQ("{\n"
            "{\n"
            "a;\n"
            "}\n"
            "{\n"
            "  b; //\n"
            "}\n"
            "}",
            format("{\n"
                   "{\n"
                   "a;\n"
                   "}\n"
                   "{\n"
                   "           b; //\n"
                   "}\n"
                   "}",
                   22, 2));
  EXPECT_EQ("  {\n"
            "    a; //\n"
            "  }",
            format("  {\n"
                   "a; //\n"
                   "  }",
                   4, 2));
  EXPECT_EQ("void f() {}\n"
            "void g() {}",
            format("void f() {}\n"
                   "void g() {}",
                   13, 0));
  EXPECT_EQ("int a; // comment\n"
            "       // line 2\n"
            "int b;",
            format("int a; // comment\n"
                   "       // line 2\n"
                   "  int b;",
                   35, 0));

  EXPECT_EQ(" void f() {\n"
            "#define A 1\n"
            " }",
            format(" void f() {\n"
                   "     #define A 1\n" // Format this line.
                   " }",
                   20, 0));
  EXPECT_EQ(" void f() {\n"
            "    int i;\n"
            "#define A \\\n"
            "    int i;  \\\n"
            "   int j;\n"
            "    int k;\n"
            " }",
            format(" void f() {\n"
                   "    int i;\n"
                   "#define A \\\n"
                   "    int i;  \\\n"
                   "   int j;\n"
                   "      int k;\n" // Format this line.
                   " }",
                   67, 0));

  Style.ColumnLimit = 11;
  EXPECT_EQ("  int a;\n"
            "  void\n"
            "  ffffff() {\n"
            "  }",
            format("  int a;\n"
                   "void ffffff() {}",
                   11, 0));
}

TEST_F(FormatTestSelective, UnderstandsTabs) {
  Style.IndentWidth = 8;
  Style.UseTab = FormatStyle::UT_Always;
  Style.AlignEscapedNewlines = FormatStyle::ENAS_Left;
  EXPECT_EQ("void f() {\n"
            "\tf();\n"
            "\tg();\n"
            "}",
            format("void f() {\n"
                   "\tf();\n"
                   "\tg();\n"
                   "}",
                   0, 0));
  EXPECT_EQ("void f() {\n"
            "\tf();\n"
            "\tg();\n"
            "}",
            format("void f() {\n"
                   "\tf();\n"
                   "\tg();\n"
                   "}",
                   16, 0));
  EXPECT_EQ("void f() {\n"
            "  \tf();\n"
            "\tg();\n"
            "}",
            format("void f() {\n"
                   "  \tf();\n"
                   "  \tg();\n"
                   "}",
                   21, 0));
}

TEST_F(FormatTestSelective, StopFormattingWhenLeavingScope) {
  EXPECT_EQ(
      "void f() {\n"
      "  if (a) {\n"
      "    g();\n"
      "    h();\n"
      "  }\n"
      "\n"
      "void g() {\n"
      "}",
      format("void f() {\n"
             "  if (a) {\n" // Assume this was added without the closing brace.
             "  g();\n"
             "  h();\n"
             "}\n"
             "\n"
             "void g() {\n" // Make sure not to format this.
             "}",
             15, 0));
}

TEST_F(FormatTestSelective, SelectivelyRequoteJavaScript) {
  Style = getGoogleStyle(FormatStyle::LK_JavaScript);
  EXPECT_EQ("var x = \"a\";\n"
            "var x = 'a';\n"
            "var x = \"a\";",
            format("var x = \"a\";\n"
                   "var x = \"a\";\n"
                   "var x = \"a\";",
                   20, 0));
}

TEST_F(FormatTestSelective, KeepsIndentAfterCommentSectionImport) {
  std::string Code = "#include <a> // line 1\n" // 23 chars long
                     "             // line 2\n" // 23 chars long
                     "\n"                       // this newline is char 47
                     "int i;";                  // this line is not indented
  EXPECT_EQ(Code, format(Code, 47, 1));
}

} // end namespace
} // end namespace format
} // end namespace clang
