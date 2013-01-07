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
    LangOpts.CPlusPlus11 = 1;
    LangOpts.ObjC1 = 1;
    Lexer Lex(ID, Context.Sources.getBuffer(ID), Context.Sources, LangOpts);
    tooling::Replacements Replace = reformat(Style, Lex, Context.Sources,
                                             Ranges);
    EXPECT_TRUE(applyAllReplacements(Replace, Context.Rewrite));
    return Context.getRewrittenText(ID);
  }

  std::string format(llvm::StringRef Code,
                     const FormatStyle &Style = getLLVMStyle()) {
    return format(Code, 0, Code.size(), Style);
  }

  std::string messUp(llvm::StringRef Code) {
    std::string MessedUp(Code.str());
    bool InComment = false;
    bool JustReplacedNewline = false;
    for (unsigned i = 0, e = MessedUp.size() - 1; i != e; ++i) {
      if (MessedUp[i] == '/' && MessedUp[i + 1] == '/') {
        if (JustReplacedNewline)
          MessedUp[i - 1] = '\n';
        InComment = true;
      } else if (MessedUp[i] == '\\' && MessedUp[i + 1] == '\n') {
        MessedUp[i] = ' ';
      } else if (MessedUp[i] == '\n') {
        if (InComment) {
          InComment = false;
        } else {
          JustReplacedNewline = true;
          MessedUp[i] = ' ';
        }
      } else if (MessedUp[i] != ' ') {
        JustReplacedNewline = false;
      }
    }
    return MessedUp;
  }

  FormatStyle getLLVMStyleWithColumns(unsigned ColumnLimit) {
    FormatStyle Style = getLLVMStyle();
    Style.ColumnLimit = ColumnLimit;
    return Style;
  }

  void verifyFormat(llvm::StringRef Code,
                    const FormatStyle &Style = getLLVMStyle()) {
    EXPECT_EQ(Code.str(), format(messUp(Code), Style));
  }

  void verifyGoogleFormat(llvm::StringRef Code) {
    verifyFormat(Code, getGoogleStyle());
  }
};

//===----------------------------------------------------------------------===//
// Basic function tests.
//===----------------------------------------------------------------------===//

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

TEST_F(FormatTest, FormatsNestedCall) {
  verifyFormat("Method(f1, f2(f3));");
  verifyFormat("Method(f1(f2, f3()));");
}

//===----------------------------------------------------------------------===//
// Tests for control statements.
//===----------------------------------------------------------------------===//

TEST_F(FormatTest, FormatIfWithoutCompountStatement) {
  verifyFormat("if (true)\n  f();\ng();");
  verifyFormat("if (a)\n  if (b)\n    if (c)\n      g();\nh();");
  verifyFormat("if (a)\n  if (b) {\n    f();\n  }\ng();");
  verifyFormat("if (a)\n"
               "  // comment\n"
               "  f();");
}

TEST_F(FormatTest, ParseIfElse) {
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

  verifyFormat(
      "for (std::vector<UnwrappedLine>::iterator I = UnwrappedLines.begin(),\n"
      "                                          E = UnwrappedLines.end();\n"
      "     I != E; ++I) {\n}");

  verifyFormat(
      "for (MachineFun::iterator IIII = PrevIt, EEEE = F.end(); IIII != EEEE;\n"
      "     ++IIIII) {\n}");
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

TEST_F(FormatTest, FormatsDoWhile) {
  verifyFormat("do {\n"
               "  do_something();\n"
               "} while (something());");
  verifyFormat("do\n"
               "  do_something();\n"
               "while (something());");
}

TEST_F(FormatTest, FormatsSwitchStatement) {
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
  verifyGoogleFormat("switch (x) {\n"
                     "  case 1:\n"
                     "    f();\n"
                     "    break;\n"
                     "  case kFoo:\n"
                     "  case ns::kBar:\n"
                     "  case kBaz:\n"
                     "    break;\n"
                     "  default:\n"
                     "    g();\n"
                     "    break;\n"
                     "}");
  verifyGoogleFormat("switch (x) {\n"
                     "  case 1: {\n"
                     "    f();\n"
                     "    break;\n"
                     "  }\n"
                     "}");
  verifyGoogleFormat("switch (test)\n"
                     "    ;");
}

TEST_F(FormatTest, FormatsLabels) {
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

//===----------------------------------------------------------------------===//
// Tests for comments.
//===----------------------------------------------------------------------===//

TEST_F(FormatTest, UnderstandsSingleLineComments) {
  verifyFormat("// line 1\n"
               "// line 2\n"
               "void f() {\n}\n");

  verifyFormat("void f() {\n"
               "  // Doesn't do anything\n"
               "}");

  verifyFormat("int i // This is a fancy variable\n"
               "    = 5;");

  verifyFormat("enum E {\n"
               "  // comment\n"
               "  VAL_A, // comment\n"
               "  VAL_B\n"
               "};");

  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa =\n"
      "    bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb; // Trailing comment");

  EXPECT_EQ("int i; // single line trailing comment",
            format("int i;\\\n// single line trailing comment"));

  verifyGoogleFormat("int a;  // Trailing comment.");
}

TEST_F(FormatTest, UnderstandsMultiLineComments) {
  verifyFormat("f(/*test=*/ true);");
}

//===----------------------------------------------------------------------===//
// Tests for classes, namespaces, etc.
//===----------------------------------------------------------------------===//

TEST_F(FormatTest, DoesNotBreakSemiAfterClassDecl) {
  verifyFormat("class A {\n};");
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

TEST_F(FormatTest, FormatsDerivedClass) {
  verifyFormat("class A : public B {\n"
               "};");
  verifyFormat("class A : public ::B {\n"
               "};");
}

TEST_F(FormatTest, FormatsEnum) {
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

TEST_F(FormatTest, FormatsNamespaces) {
  verifyFormat("namespace some_namespace {\n"
               "class A {\n"
               "};\n"
               "void f() {\n"
               "  f();\n"
               "}\n"
               "}");
  verifyFormat("namespace {\n"
               "class A {\n"
               "};\n"
               "void f() {\n"
               "  f();\n"
               "}\n"
               "}");
  verifyFormat("inline namespace X {\n"
               "class A {\n"
               "};\n"
               "void f() {\n"
               "  f();\n"
               "}\n"
               "}");
  verifyFormat("using namespace some_namespace;\n"
               "class A {\n"
               "};\n"
               "void f() {\n"
               "  f();\n"
               "}");
}

TEST_F(FormatTest, StaticInitializers) {
  verifyFormat("static SomeClass SC = { 1, 'a' };");

  // FIXME: Format like enums if the static initializer does not fit on a line.
  verifyFormat(
      "static SomeClass WithALoooooooooooooooooooongName = {\n"
      "  100000000, \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\" };");
}

TEST_F(FormatTest, FormatsSmallMacroDefinitionsInSingleLine) {
  verifyFormat("#define ALooooooooooooooooooooooooooooooooooooooongMacro("
               "                      \\\n"
               "    aLoooooooooooooooooooooooongFuuuuuuuuuuuuuunctiooooooooo)");
}

TEST_F(FormatTest, DoesNotBreakPureVirtualFunctionDefinition) {
  verifyFormat("virtual void write(ELFWriter *writerrr,\n"
               "                   OwningPtr<FileOutputBuffer> &buffer) = 0;");
}

TEST_F(FormatTest, BreaksOnHashWhenDirectiveIsInvalid) {
  EXPECT_EQ("#\n;", format("#;"));
  verifyFormat("#\n;\n;\n;");
}

TEST_F(FormatTest, UnescapedEndOfLineEndsPPDirective) {
  EXPECT_EQ("#line 42 \"test\"\n",
            format("#  \\\n  line  \\\n  42  \\\n  \"test\"\n"));
  EXPECT_EQ("#define A  \\\n  B\n",
            format("#  \\\n define  \\\n    A  \\\n       B\n",
                   getLLVMStyleWithColumns(12)));
}

TEST_F(FormatTest, EndOfFileEndsPPDirective) {
  EXPECT_EQ("#line 42 \"test\"",
            format("#  \\\n  line  \\\n  42  \\\n  \"test\""));
  EXPECT_EQ("#define A  \\\n  B",
            format("#  \\\n define  \\\n    A  \\\n       B",
                   getLLVMStyleWithColumns(12)));
}

TEST_F(FormatTest, IndentsPPDirectiveInReducedSpace) {
  // If the macro fits in one line, we still do not get the full
  // line, as only the next line decides whether we need an escaped newline and
  // thus use the last column.
  verifyFormat("#define A(B)", getLLVMStyleWithColumns(13));

  verifyFormat("#define A( \\\n    B)", getLLVMStyleWithColumns(12));
  verifyFormat("#define AA(\\\n    B)", getLLVMStyleWithColumns(12));
  verifyFormat("#define A( \\\n    A, B)", getLLVMStyleWithColumns(12));
}

TEST_F(FormatTest, HandlePreprocessorDirectiveContext) {
  EXPECT_EQ("// some comment\n"
            "#include \"a.h\"\n"
            "#define A(A,\\\n"
            "          B)\n"
            "#include \"b.h\"\n"
            "// some comment\n",
            format("  // some comment\n"
                   "  #include \"a.h\"\n"
                   "#define A(A,\\\n"
                   "    B)\n"
                   "    #include \"b.h\"\n"
                   " // some comment\n", getLLVMStyleWithColumns(13)));
}

TEST_F(FormatTest, LayoutSingleHash) {
  EXPECT_EQ("#\na;", format("#\na;"));
}

TEST_F(FormatTest, LayoutCodeInMacroDefinitions) {
  EXPECT_EQ("#define A    \\\n"
            "  c;         \\\n"
            "  e;\n"
            "f;", format("#define A c; e;\n"
                         "f;", getLLVMStyleWithColumns(14)));
}

TEST_F(FormatTest, LayoutRemainingTokens) {
  EXPECT_EQ("{\n}", format("{}"));
}

TEST_F(FormatTest, LayoutSingleUnwrappedLineInMacro) {
  EXPECT_EQ("#define A \\\n  b;",
            format("#define A b;", 10, 2, getLLVMStyleWithColumns(11)));
}

TEST_F(FormatTest, MacroDefinitionInsideStatement) {
  EXPECT_EQ("int x,\n#define A\ny;", format("int x,\n#define A\ny;"));
}

TEST_F(FormatTest, HashInMacroDefinition) {
  verifyFormat("#define A \\\n  b #c;", getLLVMStyleWithColumns(11));
  verifyFormat("#define A \\\n"
               "  {       \\\n"
               "    f(#c);\\\n"
               "  }", getLLVMStyleWithColumns(11));
}

TEST_F(FormatTest, IndentPreprocessorDirectivesAtZero) {
  EXPECT_EQ("{\n  {\n#define A\n  }\n}", format("{{\n#define A\n}}"));
}

TEST_F(FormatTest, FormatHashIfNotAtStartOfLine) {
  verifyFormat("{\n  {\n    a #c;\n  }\n}");
}

TEST_F(FormatTest, FormatUnbalancedStructuralElements) {
  EXPECT_EQ("#define A \\\n  {       \\\n    {\nint i;",
            format("#define A { {\nint i;", getLLVMStyleWithColumns(11)));
  EXPECT_EQ("#define A \\\n  }       \\\n  }\nint i;",
            format("#define A } }\nint i;", getLLVMStyleWithColumns(11)));
}

TEST_F(FormatTest, EscapedNewlineAtStartOfTokenInMacroDefinition) {
  EXPECT_EQ(
      "#define A \\\n  int i;  \\\n  int j;",
      format("#define A \\\nint i;\\\n  int j;", getLLVMStyleWithColumns(11)));
}

TEST_F(FormatTest, CalculateSpaceOnConsecutiveLinesInMacro) {
  verifyFormat("#define A \\\n"
               "  int v(  \\\n"
               "      a); \\\n"
               "  int i;", getLLVMStyleWithColumns(11));
}

TEST_F(FormatTest, MixingPreprocessorDirectivesAndNormalCode) {
  EXPECT_EQ(
      "#define ALooooooooooooooooooooooooooooooooooooooongMacro("
      "                      \\\n"
      "    aLoooooooooooooooooooooooongFuuuuuuuuuuuuuunctiooooooooo)\n"
      "\n"
      "AlooooooooooooooooooooooooooooooooooooooongCaaaaaaaaaal(\n"
      "    aLooooooooooooooooooooooonPaaaaaaaaaaaaaaaaaaaaarmmmm);\n",
      format("  #define   ALooooooooooooooooooooooooooooooooooooooongMacro("
             "\\\n"
             "aLoooooooooooooooooooooooongFuuuuuuuuuuuuuunctiooooooooo)\n"
             "  \n"
             "   AlooooooooooooooooooooooooooooooooooooooongCaaaaaaaaaal(\n"
             "  aLooooooooooooooooooooooonPaaaaaaaaaaaaaaaaaaaaarmmmm);\n"));
}

//===----------------------------------------------------------------------===//
// Line break tests.
//===----------------------------------------------------------------------===//

TEST_F(FormatTest, FormatsFunctionDefinition) {
  verifyFormat("void f(int a, int b, int c, int d, int e, int f, int g,"
               " int h, int j, int f,\n"
               "       int c, int ddddddddddddd) {\n"
               "}");
}

TEST_F(FormatTest, FormatsAwesomeMethodCall) {
  verifyFormat(
      "SomeLongMethodName(SomeReallyLongMethod(\n"
      "    CallOtherReallyLongMethod(parameter, parameter, parameter)),\n"
      "                   SecondLongCall(parameter));");
}

TEST_F(FormatTest, ConstructorInitializers) {
  verifyFormat("Constructor() : Initializer(FitsOnTheLine) {\n}");

  verifyFormat(
      "SomeClass::Constructor()\n"
      "    : aaaaaaaaaaaaa(aaaaaaaaaaaaaa), aaaaaaaaaaaaaaa(aaaaaaaaaaaa) {\n"
      "}");

  verifyFormat(
      "SomeClass::Constructor()\n"
      "    : aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa),\n"
      "      aaaaaaaaaaaaaaa(aaaaaaaaaaaa) {\n"
      "}");

  verifyFormat("Constructor()\n"
               "    : aaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaa),\n"
               "      aaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "                               aaaaaaaaaaaaaaaaaaaaaaaaaaa),\n"
               "      aaaaaaaaaaaaaaaaaaaaaaa() {\n"
               "}");

  // Here a line could be saved by splitting the second initializer onto two
  // lines, but that is not desireable.
  verifyFormat("Constructor()\n"
               "    : aaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaa),\n"
               "      aaaaaaaaaaa(aaaaaaaaaaa),\n"
               "      aaaaaaaaaaaaaaaaaaaaat(aaaaaaaaaaaaaaaaaaaaaaaaaaaa) {\n"
               "}");

  verifyGoogleFormat("MyClass::MyClass(int var)\n"
                     "    : some_var_(var),  // 4 space indent\n"
                     "      some_other_var_(var + 1) {  // lined up\n"
                     "}");
}

TEST_F(FormatTest, BreaksAsHighAsPossible) {
  verifyFormat(
      "if ((aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa && aaaaaaaaaaaaaaaaaaaaaaaaaa) ||\n"
      "    (bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb && bbbbbbbbbbbbbbbbbbbbbbbbbb))\n"
      "  f();");
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

  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa &&\n"
      "                                 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");

  // This test case breaks on an incorrect memoization, i.e. an optimization not
  // taking into account the StopAt value.
  verifyFormat(
      "return aaaaaaaaaaaaaaaaaaaaaaaa || aaaaaaaaaaaaaaaaaaaaaaa ||\n"
      "       aaaaaaaaaaa(aaaaaaaaa) || aaaaaaaaaaaaaaaaaaaaaaa ||\n"
      "       aaaaaaaaaaaaaaaaaaaaaaaaa || aaaaaaaaaaaaaaaaaaaaaaa ||\n"
      "       (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");

  verifyFormat("{\n  {\n    {\n"
               "      Annotation.SpaceRequiredBefore =\n"
               "          Line.Tokens[i - 1].Tok.isNot(tok::l_paren) &&\n"
               "          Line.Tokens[i - 1].Tok.isNot(tok::l_square);\n"
               "    }\n  }\n}");
}

TEST_F(FormatTest, BreaksAccordingToOperatorPrecedence) {
  verifyFormat(
      "if (aaaaaaaaaaaaaaaaaaaaaaaaa ||\n"
      "    bbbbbbbbbbbbbbbbbbbbbbbbb && ccccccccccccccccccccccccc) {\n}");
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaa && bbbbbbbbbbbbbbbbbbbbbbbbb ||\n"
               "    ccccccccccccccccccccccccc) {\n}");
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaa || bbbbbbbbbbbbbbbbbbbbbbbbb ||\n"
               "    ccccccccccccccccccccccccc) {\n}");
  verifyFormat(
      "if ((aaaaaaaaaaaaaaaaaaaaaaaaa || bbbbbbbbbbbbbbbbbbbbbbbbb) &&\n"
      "    ccccccccccccccccccccccccc) {\n}");
}

TEST_F(FormatTest, PrefersNotToBreakAfterAssignments) {
  verifyFormat(
      "unsigned Cost = TTI.getMemoryOpCost(I->getOpcode(), VectorTy,\n"
      "                                    SI->getAlignment(),\n"
      "                                    SI->getPointerAddressSpaceee());\n");
  verifyFormat(
      "CharSourceRange LineRange = CharSourceRange::getTokenRange(\n"
      "                                Line.Tokens.front().Tok.getLocation(),\n"
      "                                Line.Tokens.back().Tok.getLocation());");
}

TEST_F(FormatTest, AlignsAfterAssignments) {
  verifyFormat(
      "int Result = aaaaaaaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaaaaaaa +\n"
      "             aaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat(
      "Result += aaaaaaaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaaaaaaa +\n"
      "          aaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat(
      "Result >>= aaaaaaaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaaaaaaa +\n"
      "           aaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat(
      "int Result = (aaaaaaaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaaaaaaa +\n"
      "              aaaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat(
      "double LooooooooooooooooooooooooongResult = aaaaaaaaaaaaaaaaaaaaaaaa +\n"
      "                                            aaaaaaaaaaaaaaaaaaaaaaaa +\n"
      "                                            aaaaaaaaaaaaaaaaaaaaaaaa;");
}

TEST_F(FormatTest, AlignsAfterReturn) {
  verifyFormat(
      "return aaaaaaaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaaaaaaa +\n"
      "       aaaaaaaaaaaaaaaaaaaaaaaaa;");
  verifyFormat(
      "return (aaaaaaaaaaaaaaaaaaaaaaaaa + aaaaaaaaaaaaaaaaaaaaaaaaa +\n"
      "        aaaaaaaaaaaaaaaaaaaaaaaaa);");
}

TEST_F(FormatTest, AlignsStringLiterals) {
  verifyFormat("loooooooooooooooooooooooooongFunction(\"short literal \"\n"
               "                                      \"short literal\");");
  verifyFormat(
      "looooooooooooooooooooooooongFunction(\n"
      "    \"short literal\"\n"
      "    \"looooooooooooooooooooooooooooooooooooooooooooooooong literal\");");
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

  verifyFormat(
      "if (int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa = 100000000 +\n"
      "                                                           10000000) {\n"
      "}");
}

TEST_F(FormatTest, WrapsAtFunctionCallsIfNecessary) {
  verifyFormat("LoooooooooooooooooooooooooooooooooooooongObject\n"
               "    .looooooooooooooooooooooooooooooooooooooongFunction();");

  verifyFormat("LoooooooooooooooooooooooooooooooooooooongObject\n"
               "    ->looooooooooooooooooooooooooooooooooooooongFunction();");

  verifyFormat(
      "LooooooooooooooooooooooooooooooooongObject->shortFunction(Parameter1,\n"
      "                                                          Parameter2);");

  verifyFormat(
      "ShortObject->shortFunction(\n"
      "    LooooooooooooooooooooooooooooooooooooooooooooooongParameter1,\n"
      "    LooooooooooooooooooooooooooooooooooooooooooooooongParameter2);");

  verifyFormat("loooooooooooooongFunction(\n"
               "    LoooooooooooooongObject->looooooooooooooooongFunction());");

  verifyFormat(
      "function(LoooooooooooooooooooooooooooooooooooongObject\n"
      "             ->loooooooooooooooooooooooooooooooooooooooongFunction());");

  // Here, it is not necessary to wrap at "." or "->".
  verifyFormat("if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaa) ||\n"
               "    aaaa.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {\n"
               "}");
  verifyFormat(
      "aaaaaaaaaaa->aaaaaaaaa(\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "    aaaaaaaaaaaaaaaaaa->aaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaa));\n");
}

TEST_F(FormatTest, WrapsTemplateDeclarations) {
  verifyFormat("template <typename T>\n"
               "virtual void loooooooooooongFunction(int Param1, int Param2);");
  verifyFormat(
      "template <typename T> void f(int Paaaaaaaaaaaaaaaaaaaaaaaaaaaaaaram1,\n"
      "                             int Paaaaaaaaaaaaaaaaaaaaaaaaaaaaaaram2);");
  verifyFormat(
      "template <typename T>\n"
      "void looooooooooooooooooooongFunction(int Paaaaaaaaaaaaaaaaaaaaram1,\n"
      "                                      int Paaaaaaaaaaaaaaaaaaaaram2);");
  verifyFormat(
      "template <typename T>\n"
      "aaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaa,\n"
      "                    aaaaaaaaaaaaaaaaaaaaaaaaaa<T>::aaaaaaaaaa,\n"
      "                    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");

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

TEST_F(FormatTest, UnderstandsUnaryOperators) {
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

  verifyFormat("a-- > b;");
  verifyFormat("b ? -a : c;");
  verifyFormat("n * sizeof char16;");
  verifyFormat("n * alignof char16;");
  verifyFormat("sizeof(char);");
  verifyFormat("alignof(char);");

  verifyFormat("return -1;");
  verifyFormat("switch (a) {\n"
               "case -1:\n"
               "  break;\n"
               "}");
}

TEST_F(FormatTest, UndestandsOverloadedOperators) {
  verifyFormat("bool operator<();");
  verifyFormat("bool operator>();");
  verifyFormat("bool operator=();");
  verifyFormat("bool operator==();");
  verifyFormat("bool operator!=();");
  verifyFormat("int operator+();");
  verifyFormat("int operator++();");
  verifyFormat("bool operator();");
  verifyFormat("bool operator()();");
  verifyFormat("bool operator[]();");
  verifyFormat("operator bool();");
  verifyFormat("operator SomeType<int>();");
  verifyFormat("void *operator new(std::size_t size);");
  verifyFormat("void *operator new[](std::size_t size);");
  verifyFormat("void operator delete(void *ptr);");
  verifyFormat("void operator delete[](void *ptr);");
}

TEST_F(FormatTest, UnderstandsUsesOfStarAndAmp) {
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
  verifyFormat("int main(int argc, char **argv) {\n}");
  verifyFormat("return 10 * b;");
  verifyFormat("return *b * *c;");
  verifyFormat("return a & ~b;");
  verifyFormat("f(b ? *c : *d);");
  verifyFormat("int a = b ? *c : *d;");
  verifyFormat("*b = a;");
  verifyFormat("a * ~b;");
  verifyFormat("a * !b;");
  verifyFormat("a * +b;");
  verifyFormat("a * -b;");
  verifyFormat("a * ++b;");
  verifyFormat("a * --b;");

  verifyFormat("InvalidRegions[*R] = 0;");

  // FIXME: Is this desired for LLVM? Fix if not.
  verifyFormat("A<int *> a;");
  verifyFormat("A<int **> a;");
  verifyFormat("A<int *, int *> a;");
  verifyFormat("A<int **, int **> a;");
  verifyFormat("Type *A = static_cast<Type *>(P);");
  verifyFormat("Type *A = (Type *) P;");
  verifyFormat("Type *A = (vector<Type *, int *>) P;");

  verifyGoogleFormat("int main(int argc, char** argv) {\n}");
  verifyGoogleFormat("A<int*> a;");
  verifyGoogleFormat("A<int**> a;");
  verifyGoogleFormat("A<int*, int*> a;");
  verifyGoogleFormat("A<int**, int**> a;");
  verifyGoogleFormat("f(b ? *c : *d);");
  verifyGoogleFormat("int a = b ? *c : *d;");
}

TEST_F(FormatTest, DoesNotBreakBeforePointerOrReference) {
  verifyFormat("int *someFunction(int LoooooooooooooooongParam1,\n"
               "                  int LoooooooooooooooongParam2) {\n}");
  verifyFormat(
      "TypeSpecDecl *TypeSpecDecl::Create(ASTContext &C, DeclContext *DC,\n"
      "                                   SourceLocation L, IdentifierIn *II,\n"
      "                                   Type *T) {\n}");
}

TEST_F(FormatTest, LineStartsWithSpecialCharacter) {
  verifyFormat("(a)->b();");
  verifyFormat("--a;");
}

TEST_F(FormatTest, HandlesIncludeDirectives) {
  EXPECT_EQ("#include <string>\n", format("#include <string>\n"));
  EXPECT_EQ("#include <a/b/c.h>\n", format("#include <a/b/c.h>\n"));
  EXPECT_EQ("#include \"a/b/string\"\n", format("#include \"a/b/string\"\n"));
  EXPECT_EQ("#include \"string.h\"\n", format("#include \"string.h\"\n"));
  EXPECT_EQ("#include \"string.h\"\n", format("#include \"string.h\"\n"));

  EXPECT_EQ("#import <string>\n", format("#import <string>\n"));
  EXPECT_EQ("#import <a/b/c.h>\n", format("#import <a/b/c.h>\n"));
  EXPECT_EQ("#import \"a/b/string\"\n", format("#import \"a/b/string\"\n"));
  EXPECT_EQ("#import \"string.h\"\n", format("#import \"string.h\"\n"));
  EXPECT_EQ("#import \"string.h\"\n", format("#import \"string.h\"\n"));
}

//===----------------------------------------------------------------------===//
// Error recovery tests.
//===----------------------------------------------------------------------===//

TEST_F(FormatTest, IncorrectAccessSpecifier) {
  verifyFormat("public:");
  verifyFormat("class A {\n"
               "public\n"
               "  void f() {\n"
               "  }\n"
               "};");
  verifyFormat("public\n"
               "int qwerty;");
  verifyFormat("public\n"
               "B {\n"
               "};");
  verifyFormat("public\n"
               "{\n"
               "};");
  verifyFormat("public\n"
               "B {\n"
               "  int x;\n"
               "};");
}

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

TEST_F(FormatTest, DoesNotTouchUnwrappedLinesWithErrors) {
  verifyFormat("namespace {\n"
               "class Foo {  Foo  ( }; }  // comment");
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

TEST_F(FormatTest, FormatForObjectiveCMethodDecls) {
  verifyFormat("- (void)sendAction:(SEL)aSelector to:(BOOL)anObject;");
  EXPECT_EQ("- (NSUInteger)indexOfObject:(id)anObject;",
            format("-(NSUInteger)indexOfObject:(id)anObject;"));
  EXPECT_EQ("- (NSInteger)Mthod1;", format("-(NSInteger)Mthod1;"));
  EXPECT_EQ("+ (id)Mthod2;", format("+(id)Mthod2;"));
  EXPECT_EQ("- (NSInteger)Method3:(id)anObject;",
            format("-(NSInteger)Method3:(id)anObject;"));
  EXPECT_EQ("- (NSInteger)Method4:(id)anObject;",
            format("-(NSInteger)Method4:(id)anObject;"));
  EXPECT_EQ("- (NSInteger)Method5:(id)anObject:(id)AnotherObject;",
            format("-(NSInteger)Method5:(id)anObject:(id)AnotherObject;"));
  EXPECT_EQ("- (id)Method6:(id)A:(id)B:(id)C:(id)D;",
            format("- (id)Method6:(id)A:(id)B:(id)C:(id)D;"));
  EXPECT_EQ(
      "- (void)sendAction:(SEL)aSelector to:(id)anObject forAllCells:(BOOL)flag;",
      format("- (void)sendAction:(SEL)aSelector to:(id)anObject forAllCells:(BOOL)flag;"));

  // Very long objectiveC method declaration.
  EXPECT_EQ(
      "- (NSUInteger)indexOfObject:(id)anObject inRange:(NSRange)range\n    "
      "outRange:(NSRange)out_range outRange1:(NSRange)out_range1\n    "
      "outRange2:(NSRange)out_range2 outRange3:(NSRange)out_range3\n    "
      "outRange4:(NSRange)out_range4 outRange5:(NSRange)out_range5\n    "
      "outRange6:(NSRange)out_range6 outRange7:(NSRange)out_range7\n    "
      "outRange8:(NSRange)out_range8 outRange9:(NSRange)out_range9;",
      format(
          "- (NSUInteger)indexOfObject:(id)anObject inRange:(NSRange)range "
          "outRange:(NSRange) out_range outRange1:(NSRange) out_range1 "
          "outRange2:(NSRange) out_range2  outRange3:(NSRange) out_range3  "
          "outRange4:(NSRange) out_range4  outRange5:(NSRange) out_range5 "
          "outRange6:(NSRange) out_range6  outRange7:(NSRange) out_range7  "
          "outRange8:(NSRange) out_range8  outRange9:(NSRange) out_range9;"));
}

TEST_F(FormatTest, ObjCAt) {
  verifyFormat("@interface");
  verifyFormat("@dynamic");
  EXPECT_EQ("@interface", format("@ interface"));

  // The precise formatting of this doesn't matter, nobody writes code like
  // this.
  verifyFormat("@ /*foo*/ interface");
}

} // end namespace tooling
} // end namespace clang
