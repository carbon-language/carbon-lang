//===- unittest/Format/FormatTest.cpp - Formatting unit tests -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Format/Format.h"
#include "../Tooling/RewriterTestContext.h"
#include "clang/Lex/Lexer.h"
#include "gtest/gtest.h"

namespace clang {
namespace format {

class FormatTest : public ::testing::Test {
protected:
  std::string format(llvm::StringRef Code, unsigned Offset, unsigned Length,
                     const FormatStyle &Style) {
    RewriterTestContext Context;
    FileID ID = Context.createInMemoryFile("input.cc", Code);
    SourceLocation Start =
        Context.Sources.getLocForStartOfFile(ID).getLocWithOffset(Offset);
    std::vector<CharSourceRange> Ranges(
        1,
        CharSourceRange::getCharRange(Start, Start.getLocWithOffset(Length)));
    LangOptions LangOpts;
    LangOpts.CPlusPlus = 1;
    Lexer Lex(ID, Context.Sources.getBuffer(ID), Context.Sources, LangOpts);
    tooling::Replacements Replace =
        reformat(Style, Lex, Context.Sources, Ranges);
    EXPECT_TRUE(applyAllReplacements(Replace, Context.Rewrite));
    return Context.getRewrittenText(ID);
  }

  std::string format(llvm::StringRef Code,
                     const FormatStyle &Style = getLLVMStyle()) {
    return format(Code, 0, Code.size(), Style);
  }

  void verifyFormat(llvm::StringRef Code) {
    std::string WithoutFormat(Code.str());
    for (unsigned i = 0, e = WithoutFormat.size(); i != e; ++i) {
      if (WithoutFormat[i] == '\n')
        WithoutFormat[i] = ' ';
    }
    EXPECT_EQ(Code.str(), format(WithoutFormat));
  }

  void verifyGoogleFormat(llvm::StringRef Code) {
    std::string WithoutFormat(Code.str());
    for (unsigned i = 0, e = WithoutFormat.size(); i != e; ++i) {
      if (WithoutFormat[i] == '\n')
        WithoutFormat[i] = ' ';
    }
    EXPECT_EQ(Code.str(), format(WithoutFormat, getGoogleStyle()));
  }
};

TEST_F(FormatTest, DoesNotChangeCorrectlyFormatedCode) {
  EXPECT_EQ(";", format(";"));
}

TEST_F(FormatTest, FormatsGlobalStatementsAt0) {
  EXPECT_EQ("int i;", format("  int i;"));
  EXPECT_EQ("\nint i;", format(" \n\t \r  int i;"));
  EXPECT_EQ("int i;\nint j;", format("    int i; int j;"));
  EXPECT_EQ("int i;\nint j;", format("    int i;\n  int j;"));
}

TEST_F(FormatTest, FormatsUnwrappedLinesAtFirstFormat) {
  EXPECT_EQ("int i;", format("int\ni;"));
}

TEST_F(FormatTest, FormatsNestedBlockStatements) {
  EXPECT_EQ("{\n  {\n    {\n    }\n  }\n}", format("{{{}}}"));
}

TEST_F(FormatTest, FormatsForLoop) {
  verifyFormat(
      "for (int VeryVeryLongLoopVariable = 0; VeryVeryLongLoopVariable < 10;\n"
      "     ++VeryVeryLongLoopVariable)\n"
      "  ;");
  verifyFormat("for (;;)\n"
               "  f();");
  verifyFormat("for (;;) {\n"
               "}");
  verifyFormat("for (;;) {\n"
               "  f();\n"
               "}");
}

TEST_F(FormatTest, FormatsWhileLoop) {
  verifyFormat("while (true) {\n}");
  verifyFormat("while (true)\n"
               "  f();");
  verifyFormat("while () {\n"
               "}");
  verifyFormat("while () {\n"
               "  f();\n"
               "}");
}

TEST_F(FormatTest, FormatsNestedCall) {
  verifyFormat("Method(f1, f2(f3));");
  verifyFormat("Method(f1(f2, f3()));");
}

TEST_F(FormatTest, FormatsAwesomeMethodCall) {
  verifyFormat(
      "SomeLongMethodName(SomeReallyLongMethod(CallOtherReallyLongMethod(\n"
      "    parameter, parameter, parameter)), SecondLongCall(parameter));");
}

TEST_F(FormatTest, FormatsFunctionDefinition) {
  verifyFormat("void f(int a, int b, int c, int d, int e, int f, int g,"
               " int h, int j, int f,\n"
               "       int c, int ddddddddddddd) {\n"
               "}");
}

TEST_F(FormatTest, FormatIfWithoutCompountStatement) {
  verifyFormat("if (true)\n  f();\ng();");
  verifyFormat("if (a)\n  if (b)\n    if (c)\n      g();\nh();");
  verifyFormat("if (a)\n  if (b) {\n    f();\n  }\ng();");
  EXPECT_EQ("if (a)\n  // comment\n  f();", format("if(a)\n// comment\nf();"));
}

TEST_F(FormatTest, ParseIfThenElse) {
  verifyFormat("if (true)\n"
               "  if (true)\n"
               "    if (true)\n"
               "      f();\n"
               "    else\n"
               "      g();\n"
               "  else\n"
               "    h();\n"
               "else\n"
               "  i();");
  verifyFormat("if (true)\n"
               "  if (true)\n"
               "    if (true) {\n"
               "      if (true)\n"
               "        f();\n"
               "    } else {\n"
               "      g();\n"
               "    }\n"
               "  else\n"
               "    h();\n"
               "else {\n"
               "  i();\n"
               "}");
}

TEST_F(FormatTest, UnderstandsSingleLineComments) {
  EXPECT_EQ("// line 1\n// line 2\nvoid f() {\n}\n",
            format("// line 1\n// line 2\nvoid f() {}\n"));

  EXPECT_EQ("void f() {\n  // Doesn't do anything\n}",
            format("void f() {\n// Doesn't do anything\n}"));

  EXPECT_EQ("int i  // This is a fancy variable\n    = 5;",
            format("int i  // This is a fancy variable\n= 5;"));

  verifyFormat("f(/*test=*/ true);");
}

TEST_F(FormatTest, DoesNotBreakSemiAfterClassDecl) {
  verifyFormat("class A {\n};");
}

TEST_F(FormatTest, BreaksAsHighAsPossible) {
  verifyFormat(
      "if ((aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa && aaaaaaaaaaaaaaaaaaaaaaaaaa) ||\n"
      "    (bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb && bbbbbbbbbbbbbbbbbbbbbbbbbb))\n"
      "  f();");
}

TEST_F(FormatTest, ElseIf) {
  verifyFormat("if (a) {\n"
               "} else if (b) {\n"
               "}");
  verifyFormat("if (a)\n"
               "  f();\n"
               "else if (b)\n"
               "  g();\n"
               "else\n"
               "  h();");
}

TEST_F(FormatTest, UnderstandsAccessSpecifiers) {
  verifyFormat("class A {\n"
               "public:\n"
               "protected:\n"
               "private:\n"
               "  void f() {\n"
               "  }\n"
               "};");
  verifyGoogleFormat("class A {\n"
                     " public:\n"
                     " protected:\n"
                     " private:\n"
                     "  void f() {\n"
                     "  }\n"
                     "};");
}

TEST_F(FormatTest, SwitchStatement) {
  verifyFormat("switch (x) {\n"
               "case 1:\n"
               "  f();\n"
               "  break;\n"
               "case kFoo:\n"
               "case ns::kBar:\n"
               "case kBaz:\n"
               "  break;\n"
               "default:\n"
               "  g();\n"
               "  break;\n"
               "}");
  verifyFormat("switch (x) {\n"
               "case 1: {\n"
               "  f();\n"
               "  break;\n"
               "}\n"
               "}");
  verifyFormat("switch (test)\n"
               "  ;");
}

TEST_F(FormatTest, Labels) {
  verifyFormat("void f() {\n"
               "  some_code();\n"
               "test_label:\n"
               "  some_other_code();\n"
               "  {\n"
               "    some_more_code();\n"
               "  another_label:\n"
               "    some_more_code();\n"
               "  }\n"
               "}");
  verifyFormat("some_code();\n"
               "test_label:\n"
               "some_other_code();");
}

TEST_F(FormatTest, DerivedClass) {
  verifyFormat("class A : public B {\n"
               "};");
}

TEST_F(FormatTest, DoWhile) {
  verifyFormat("do {\n"
               "  do_something();\n"
               "} while (something());");
  verifyFormat("do\n"
               "  do_something();\n"
               "while (something());");
}

TEST_F(FormatTest, Enum) {
  verifyFormat("enum {\n"
               "  Zero,\n"
               "  One = 1,\n"
               "  Two = One + 1,\n"
               "  Three = (One + Two),\n"
               "  Four = (Zero && (One ^ Two)) | (One << Two),\n"
               "  Five = (One, Two, Three, Four, 5)\n"
               "};");
  verifyFormat("enum Enum {\n"
               "};");
  verifyFormat("enum {\n"
               "};");
}

TEST_F(FormatTest, BreaksDesireably) {
  verifyFormat("if (aaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaa) ||\n"
               "    aaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaa) ||\n"
               "    aaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaa)) {\n};");

  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "                      aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {\n}");

  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa));");

  verifyFormat(
      "aaaaaaaa(aaaaaaaaaaaaa, aaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "                            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)),\n"
      "         aaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "             aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)));");

  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ||\n"
               "    (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");

  // This test case breaks on an incorrect memoization, i.e. an optimization not
  // taking into account the StopAt value.
  verifyFormat(
      "return aaaaaaaaaaaaaaaaaaaaaaaa || aaaaaaaaaaaaaaaaaaaaaaa ||\n"
      "    aaaaaaaaaaa(aaaaaaaaa) || aaaaaaaaaaaaaaaaaaaaaaa ||\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaa || aaaaaaaaaaaaaaaaaaaaaaa ||\n"
      "    (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");
}

TEST_F(FormatTest, AlignsStringLiterals) {
  verifyFormat("loooooooooooooooooooooooooongFunction(\"short literal \"\n"
               "                                      \"short literal\");");
  verifyFormat(
      "looooooooooooooooooooooooongFunction(\n"
      "    \"short literal\"\n"
      "    \"looooooooooooooooooooooooooooooooooooooooooooooooong literal\");");
}

TEST_F(FormatTest, UnderstandsEquals) {
  verifyFormat(
      "aaaaaaaaaaaaaaaaa =\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat(
      "if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa =\n"
      "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {\n"
      "}");
  verifyFormat(
      "if (a) {\n"
      "} else if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa =\n"
      "               aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {\n"
      "}");

  verifyFormat("if (int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa =\n"
               "        100000000 + 100000000) {\n}");
}

TEST_F(FormatTest, UnderstandsTemplateParameters) {
  verifyFormat("A<int> a;");
  verifyFormat("A<A<A<int> > > a;");
  verifyFormat("A<A<A<int, 2>, 3>, 4> a;");
  verifyFormat("bool x = a < 1 || 2 > a;");
  verifyFormat("bool x = 5 < f<int>();");
  verifyFormat("bool x = f<int>() > 5;");
  verifyFormat("bool x = 5 < a<int>::x;");
  verifyFormat("bool x = a < 4 ? a > 2 : false;");
  verifyFormat("bool x = f() ? a < 2 : a > 2;");

  verifyGoogleFormat("A<A<int>> a;");
  verifyGoogleFormat("A<A<A<int>>> a;");
  verifyGoogleFormat("A<A<A<A<int>>>> a;");

  verifyFormat("test >> a >> b;");
  verifyFormat("test << a >> b;");

  verifyFormat("f<int>();");
  verifyFormat("template <typename T> void f() {\n}");
}

TEST_F(FormatTest, UndestandsUnaryOperators) {
  verifyFormat("int a = -2;");
  verifyFormat("f(-1, -2, -3);");
  verifyFormat("a[-1] = 5;");
  verifyFormat("int a = 5 + -2;");
  verifyFormat("if (i == -1) {\n}");
  verifyFormat("if (i != -1) {\n}");
  verifyFormat("if (i > -1) {\n}");
  verifyFormat("if (i < -1) {\n}");
  verifyFormat("++(a->f());");
  verifyFormat("--(a->f());");
  verifyFormat("if (!(a->f())) {\n}");
}

TEST_F(FormatTest, UndestandsOverloadedOperators) {
  verifyFormat("bool operator<() {\n}");
}

TEST_F(FormatTest, UnderstandsUsesOfStar) {
  verifyFormat("int *f(int *a) {\n}");
  verifyFormat("f(a, *a);");
  verifyFormat("f(*a);");
  verifyFormat("int a = b * 10;");
  verifyFormat("int a = 10 * b;");
  verifyFormat("int a = b * c;");
  verifyFormat("int a += b * c;");
  verifyFormat("int a -= b * c;");
  verifyFormat("int a *= b * c;");
  verifyFormat("int a /= b * c;");
  verifyFormat("int a = *b;");
  verifyFormat("int a = *b * c;");
  verifyFormat("int a = b * *c;");
}

TEST_F(FormatTest, LineStartsWithSpecialCharacter) {
  verifyFormat("(a)->b();");
  verifyFormat("--a;");
}

TEST_F(FormatTest, HandlesIncludeDirectives) {
  EXPECT_EQ("#include <string>\n", format("#include <string>\n"));
  EXPECT_EQ("#include \"a/b/string\"\n", format("#include \"a/b/string\"\n"));
  EXPECT_EQ("#include \"string.h\"\n", format("#include \"string.h\"\n"));
  EXPECT_EQ("#include \"string.h\"\n", format("#include \"string.h\"\n"));
}

//TEST_F(FormatTest, IncorrectDerivedClass) {
//  verifyFormat("public B {\n"
//               "};");
//}

TEST_F(FormatTest, IncorrectCodeUnbalancedBraces) {
  verifyFormat("{");
}

TEST_F(FormatTest, IncorrectCodeDoNoWhile) {
  verifyFormat("do {\n"
               "};");
  verifyFormat("do {\n"
               "};\n"
               "f();");
  verifyFormat("do {\n"
               "}\n"
               "wheeee(fun);");
  verifyFormat("do {\n"
               "  f();\n"
               "};");
}

TEST_F(FormatTest, IncorrectCodeErrorDetection) {
  EXPECT_EQ("{\n{\n}\n", format("{\n{\n}\n"));
  EXPECT_EQ("{\n  {\n}\n", format("{\n  {\n}\n"));
  EXPECT_EQ("{\n  {\n  }\n", format("{\n  {\n  }\n"));
  EXPECT_EQ("{\n  {\n    }\n  }\n}\n", format("{\n  {\n    }\n  }\n}\n"));

  FormatStyle Style = getLLVMStyle();
  Style.ColumnLimit = 10;
  EXPECT_EQ("{\n"
            "    {\n"
            " breakme(\n"
            "     qwe);\n"
            "}\n", format("{\n"
                          "    {\n"
                          " breakme(qwe);\n"
                          "}\n", Style));

}

TEST_F(FormatTest, AlignsPipes) {
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "    << aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "    << aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaa << aaaaaaaaaaaaaaaaaaaa << aaaaaaaaaaaaaaaaaaaa\n"
      "                     << aaaaaaaaaaaaaaaaaaaa;");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa << aaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                                 << aaaaaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat(
      "llvm::outs() << \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"\n"
      "                \"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\"\n"
      "             << \"ccccccccccccccccccccccccccccccccccccccccccccccccc\";");
  verifyFormat(
      "aaaaaaaa << (aaaaaaaaaaaaaaaaaaa << aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
      "                                 << aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
      "         << aaaaaaaaaaaaaaaaaaaaaaaaaaaaa;");
}

}  // end namespace tooling
}  // end namespace clang
