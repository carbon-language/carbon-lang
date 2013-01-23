//===- unittest/Format/FormatTest.cpp - Formatting unit tests -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "format-test"

#include "clang/Format/Format.h"
#include "../Tooling/RewriterTestContext.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/Debug.h"
#include "gtest/gtest.h"

// Uncomment to get debug output from tests:
// #define DEBUG_WITH_TYPE(T, X) do { X; } while(0)

namespace clang {
namespace format {

class FormatTest : public ::testing::Test {
protected:
  std::string format(llvm::StringRef Code, unsigned Offset, unsigned Length,
                     const FormatStyle &Style) {
    DEBUG(llvm::errs() << "---\n");
    RewriterTestContext Context;
    FileID ID = Context.createInMemoryFile("input.cc", Code);
    SourceLocation Start =
        Context.Sources.getLocForStartOfFile(ID).getLocWithOffset(Offset);
    std::vector<CharSourceRange> Ranges(
        1,
        CharSourceRange::getCharRange(Start, Start.getLocWithOffset(Length)));
    Lexer Lex(ID, Context.Sources.getBuffer(ID), Context.Sources,
              getFormattingLangOpts());
    tooling::Replacements Replace = reformat(Style, Lex, Context.Sources,
                                             Ranges,
                                             new IgnoringDiagConsumer());
    EXPECT_TRUE(applyAllReplacements(Replace, Context.Rewrite));
    DEBUG(llvm::errs() << "\n" << Context.getRewrittenText(ID) << "\n\n");
    return Context.getRewrittenText(ID);
  }

  std::string format(llvm::StringRef Code,
                     const FormatStyle &Style = getLLVMStyle()) {
    return format(Code, 0, Code.size(), Style);
  }

  std::string messUp(llvm::StringRef Code) {
    std::string MessedUp(Code.str());
    bool InComment = false;
    bool InPreprocessorDirective = false;
    bool JustReplacedNewline = false;
    for (unsigned i = 0, e = MessedUp.size() - 1; i != e; ++i) {
      if (MessedUp[i] == '/' && MessedUp[i + 1] == '/') {
        if (JustReplacedNewline)
          MessedUp[i - 1] = '\n';
        InComment = true;
      } else if (MessedUp[i] == '#' && (JustReplacedNewline || i == 0)) {
        if (i != 0) MessedUp[i - 1] = '\n';
        InPreprocessorDirective = true;
      } else if (MessedUp[i] == '\\' && MessedUp[i + 1] == '\n') {
        MessedUp[i] = ' ';
        MessedUp[i + 1] = ' ';
      } else if (MessedUp[i] == '\n') {
        if (InComment) {
          InComment = false;
        } else if (InPreprocessorDirective) {
          InPreprocessorDirective = false;
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

  FormatStyle getGoogleStyleWithColumns(unsigned ColumnLimit) {
    FormatStyle Style = getGoogleStyle();
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

TEST_F(FormatTest, MessUp) {
  EXPECT_EQ("1 2 3", messUp("1 2 3"));
  EXPECT_EQ("1 2 3\n", messUp("1\n2\n3\n"));
  EXPECT_EQ("a\n//b\nc", messUp("a\n//b\nc"));
  EXPECT_EQ("a\n#b\nc", messUp("a\n#b\nc"));
  EXPECT_EQ("a\n#b  c  d\ne", messUp("a\n#b\\\nc\\\nd\ne"));
}

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
  EXPECT_EQ("{\n  {\n    {}\n  }\n}", format("{{{}}}"));
}

TEST_F(FormatTest, FormatsNestedCall) {
  verifyFormat("Method(f1, f2(f3));");
  verifyFormat("Method(f1(f2, f3()));");
  verifyFormat("Method(f1(f2, (f3())));");
}

TEST_F(FormatTest, ImportantSpaces) {
  verifyFormat("vector< ::Type> v;");
}

//===----------------------------------------------------------------------===//
// Tests for control statements.
//===----------------------------------------------------------------------===//

TEST_F(FormatTest, FormatIfWithoutCompountStatement) {
  verifyFormat("if (true)\n  f();\ng();");
  verifyFormat("if (a)\n  if (b)\n    if (c)\n      g();\nh();");
  verifyFormat("if (a)\n  if (b) {\n    f();\n  }\ng();");

  FormatStyle AllowsMergedIf = getGoogleStyle();
  AllowsMergedIf.AllowShortIfStatementsOnASingleLine = true;
  verifyFormat("if (a)\n"
               "  // comment\n"
               "  f();", AllowsMergedIf);

  verifyFormat("if (a)  // Can't merge this\n"
               "  f();\n", AllowsMergedIf);
  verifyFormat("if (a) /* still don't merge */\n"
               "  f();", AllowsMergedIf);
  verifyFormat("if (a) {  // Never merge this\n"
               "  f();\n"
               "}", AllowsMergedIf);
  verifyFormat("if (a) { /* Never merge this */\n"
               "  f();\n"
               "}", AllowsMergedIf);

  AllowsMergedIf.ColumnLimit = 14;
  verifyFormat("if (a) return;", AllowsMergedIf);
  verifyFormat("if (aaaaaaaaa)\n"
               "  return;", AllowsMergedIf);

  AllowsMergedIf.ColumnLimit = 13;
  verifyFormat("if (a)\n  return;", AllowsMergedIf);
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
  verifyFormat("if (a) {\n} else if (b) {\n}");
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
  verifyFormat("for (;;) {\n}");
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
  verifyFormat("while () {\n}");
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
  verifyFormat("switch (x) {\n"
               "case 1: {\n"
               "  f();\n"
               "  {\n"
               "    g();\n"
               "    h();\n"
               "  }\n"
               "  break;\n"
               "}\n"
               "}");
  verifyFormat("switch (x) {\n"
               "case 1: {\n"
               "  f();\n"
               "  if (foo) {\n"
               "    g();\n"
               "    h();\n"
               "  }\n"
               "  break;\n"
               "}\n"
               "}");
  verifyFormat("switch (x) {\n"
               "case 1: {\n"
               "  f();\n"
               "  g();\n"
               "} break;\n"
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
               "void f() {}\n");

  verifyFormat("void f() {\n"
               "  // Doesn't do anything\n"
               "}");
  verifyFormat("void f(int i,  // some comment (probably for i)\n"
               "       int j,  // some comment (probably for j)\n"
               "       int k); // some comment (probably for k)");
  verifyFormat("void f(int i,\n"
               "       // some comment (probably for j)\n"
               "       int j,\n"
               "       // some comment (probably for k)\n"
               "       int k);");

  verifyFormat("int i    // This is a fancy variable\n"
               "    = 5; // with nicely aligned comment.");

  verifyFormat("// Leading comment.\n"
               "int a; // Trailing comment.");
  verifyFormat("int a; // Trailing comment\n"
               "       // on 2\n"
               "       // or 3 lines.\n"
               "int b;");
  verifyFormat("int a; // Trailing comment\n"
               "\n"
               "// Leading comment.\n"
               "int b;");
  verifyFormat("int a;    // Comment.\n"
               "          // More details.\n"
               "int bbbb; // Another comment.");
  verifyFormat(
      "int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; // comment\n"
      "int bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb;   // comment\n"
      "int cccccccccccccccccccccccccccccc;       // comment\n"
      "int ddd;                     // looooooooooooooooooooooooong comment\n"
      "int aaaaaaaaaaaaaaaaaaaaaaa; // comment\n"
      "int bbbbbbbbbbbbbbbbbbbbb;   // comment\n"
      "int ccccccccccccccccccc;     // comment");

  verifyFormat("#include \"a\"     // comment\n"
               "#include \"a/b/c\" // comment");
  verifyFormat("#include <a>     // comment\n"
               "#include <a/b/c> // comment");

  verifyFormat("enum E {\n"
               "  // comment\n"
               "  VAL_A, // comment\n"
               "  VAL_B\n"
               "};");

  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa =\n"
      "    bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb; // Trailing comment");
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa =\n"
               "    // Comment inside a statement.\n"
               "    bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb;");

  EXPECT_EQ("void f() { // This does something ..\n"
            "}\n"
            "int a; // This is unrelated",
            format("void f()    {     // This does something ..\n"
                   "  }\n"
                   "int   a;     // This is unrelated"));
  EXPECT_EQ("void f() { // This does something ..\n"
            "}          // awesome..\n"
            "\n"
            "int a; // This is unrelated",
            format("void f()    { // This does something ..\n"
                   "      } // awesome..\n"
                   " \n"
                   "int a;    // This is unrelated"));

  EXPECT_EQ("int i; // single line trailing comment",
            format("int i;\\\n// single line trailing comment"));

  verifyGoogleFormat("int a;  // Trailing comment.");
}

TEST_F(FormatTest, UnderstandsMultiLineComments) {
  verifyFormat("f(/*test=*/ true);");
  EXPECT_EQ(
      "f(aaaaaaaaaaaaaaaaaaaaaaaaa, /* Trailing comment for aa... */\n"
      "  bbbbbbbbbbbbbbbbbbbbbbbbb);",
      format("f(aaaaaaaaaaaaaaaaaaaaaaaaa ,  /* Trailing comment for aa... */\n"
             "  bbbbbbbbbbbbbbbbbbbbbbbbb);"));
  EXPECT_EQ(
      "f(aaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "  /* Leading comment for bb... */ bbbbbbbbbbbbbbbbbbbbbbbbb);",
      format("f(aaaaaaaaaaaaaaaaaaaaaaaaa    ,   \n"
             "/* Leading comment for bb... */   bbbbbbbbbbbbbbbbbbbbbbbbb);"));
}

TEST_F(FormatTest, CommentsInStaticInitializers) {
  EXPECT_EQ(
      "static SomeType type = { aaaaaaaaaaaaaaaaaaaa, /* comment */\n"
      "                         aaaaaaaaaaaaaaaaaaaa /* comment */,\n"
      "                         /* comment */ aaaaaaaaaaaaaaaaaaaa,\n"
      "                         aaaaaaaaaaaaaaaaaaaa, // comment\n"
      "                         aaaaaaaaaaaaaaaaaaaa };",
      format("static SomeType type = { aaaaaaaaaaaaaaaaaaaa  ,  /* comment */\n"
             "                   aaaaaaaaaaaaaaaaaaaa   /* comment */ ,\n"
             "                     /* comment */   aaaaaaaaaaaaaaaaaaaa ,\n"
             "              aaaaaaaaaaaaaaaaaaaa ,   // comment\n"
             "                  aaaaaaaaaaaaaaaaaaaa };"));
  verifyFormat("static SomeType type = { aaaaaaaaaaa, // comment for aa...\n"
               "                         bbbbbbbbbbb, ccccccccccc };");
  verifyFormat("static SomeType type = { aaaaaaaaaaa,\n"
               "                         // comment for bb....\n"
               "                         bbbbbbbbbbb, ccccccccccc };");
  verifyGoogleFormat(
      "static SomeType type = { aaaaaaaaaaa,  // comment for aa...\n"
      "                         bbbbbbbbbbb,\n"
      "                         ccccccccccc };");
  verifyGoogleFormat("static SomeType type = { aaaaaaaaaaa,\n"
                     "                         // comment for bb....\n"
                     "                         bbbbbbbbbbb,\n"
                     "                         ccccccccccc };");

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
               "  void f() {}\n"
               "};");
  verifyGoogleFormat("class A {\n"
                     " public:\n"
                     " protected:\n"
                     " private:\n"
                     "  void f() {}\n"
                     "};");
}

TEST_F(FormatTest, FormatsDerivedClass) {
  verifyFormat("class A : public B {\n};");
  verifyFormat("class A : public ::B {\n};");
}

TEST_F(FormatTest, FormatsVariableDeclarationsAfterStructOrClass) {
  verifyFormat("class A {\n} a, b;");
  verifyFormat("struct A {\n} a, b;");
  verifyFormat("union A {\n} a;");
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
  verifyFormat("enum X E {\n} d;");
  verifyFormat("enum __attribute__((...)) E {\n} d;");
  verifyFormat("enum __declspec__((...)) E {\n} d;");
  verifyFormat("enum X f() {\n  a();\n  return 42;\n}");
}

TEST_F(FormatTest, FormatsBitfields) {
  verifyFormat("struct Bitfields {\n"
               "  unsigned sClass : 8;\n"
               "  unsigned ValueKind : 2;\n"
               "};");
}

TEST_F(FormatTest, FormatsNamespaces) {
  verifyFormat("namespace some_namespace {\n"
               "class A {\n};\n"
               "void f() { f(); }\n"
               "}");
  verifyFormat("namespace {\n"
               "class A {\n};\n"
               "void f() { f(); }\n"
               "}");
  verifyFormat("inline namespace X {\n"
               "class A {\n};\n"
               "void f() { f(); }\n"
               "}");
  verifyFormat("using namespace some_namespace;\n"
               "class A {\n};\n"
               "void f() { f(); }");
}

TEST_F(FormatTest, FormatsExternC) {
  verifyFormat("extern \"C\" {\nint a;");
}

TEST_F(FormatTest, FormatTryCatch) {
  // FIXME: Handle try-catch explicitly in the UnwrappedLineParser, then we'll
  // also not create single-line-blocks.
  verifyFormat("try {\n"
               "  throw a * b;\n"
               "}\n"
               "catch (int a) {\n"
               "  // Do nothing.\n"
               "}\n"
               "catch (...) {\n"
               "  exit(42);\n"
               "}");

  // Function-level try statements.
  verifyFormat("int f() try { return 4; }\n"
               "catch (...) {\n"
               "  return 5;\n"
               "}");
  verifyFormat("class A {\n"
               "  int a;\n"
               "  A() try : a(0) {}\n"
               "  catch (...) {\n"
               "    throw;\n"
               "  }\n"
               "};\n");
}

TEST_F(FormatTest, FormatObjCTryCatch) {
  verifyFormat("@try {\n"
               "  f();\n"
               "}\n"
               "@catch (NSException e) {\n"
               "  @throw;\n"
               "}\n"
               "@finally {\n"
               "  exit(42);\n"
               "}");
}

TEST_F(FormatTest, StaticInitializers) {
  verifyFormat("static SomeClass SC = { 1, 'a' };");

  // FIXME: Format like enums if the static initializer does not fit on a line.
  verifyFormat(
      "static SomeClass WithALoooooooooooooooooooongName = {\n"
      "  100000000, \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"\n"
      "};");

  verifyFormat(
      "static SomeClass = { a, b, c, d, e, f, g, h, i, j,\n"
      "                     looooooooooooooooooooooooooooooooooongname,\n"
      "                     looooooooooooooooooooooooooooooong };");
}

TEST_F(FormatTest, NestedStaticInitializers) {
  verifyFormat("static A x = { { {} } };\n");
  verifyFormat(
      "static A x = { { { init1, init2, init3, init4 },\n"
      "                 { init1, init2, init3, init4 } } };");

  // FIXME: Fix this in general and verify that it works in LLVM style again.
  verifyGoogleFormat(
      "somes Status::global_reps[3] = {\n"
      "  { kGlobalRef, OK_CODE, NULL, NULL, NULL },\n"
      "  { kGlobalRef, CANCELLED_CODE, NULL, NULL, NULL },\n"
      "  { kGlobalRef, UNKNOWN_CODE, NULL, NULL, NULL }\n"
      "};");
  verifyFormat(
      "CGRect cg_rect = { { rect.fLeft, rect.fTop },\n"
      "                   { rect.fRight - rect.fLeft, rect.fBottom - rect.fTop"
      " } };");

  // FIXME: We might at some point want to handle this similar to parameter
  // lists, where we have an option to put each on a single line.
  verifyFormat("struct {\n"
               "  unsigned bit;\n"
               "  const char *const name;\n"
               "} kBitsToOs[] = { { kOsMac, \"Mac\" }, { kOsWin, \"Windows\" },\n"
               "                  { kOsLinux, \"Linux\" }, { kOsCrOS, \"Chrome OS\" } };");
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
  EXPECT_EQ("#define A B\n",
            format("#  \\\n define  \\\n    A  \\\n       B\n",
                   getLLVMStyleWithColumns(12)));
}

TEST_F(FormatTest, EndOfFileEndsPPDirective) {
  EXPECT_EQ("#line 42 \"test\"",
            format("#  \\\n  line  \\\n  42  \\\n  \"test\""));
  EXPECT_EQ("#define A B",
            format("#  \\\n define  \\\n    A  \\\n       B"));
}

TEST_F(FormatTest, IndentsPPDirectiveInReducedSpace) {
  // If the macro fits in one line, we still do not get the full
  // line, as only the next line decides whether we need an escaped newline and
  // thus use the last column.
  verifyFormat("#define A(B)", getLLVMStyleWithColumns(13));

  verifyFormat("#define A( \\\n    B)", getLLVMStyleWithColumns(12));
  verifyFormat("#define AA(\\\n    B)", getLLVMStyleWithColumns(12));
  verifyFormat("#define A( \\\n    A, B)", getLLVMStyleWithColumns(12));

  verifyFormat("#define A A\n#define A A");
  verifyFormat("#define A(X) A\n#define A A");

  verifyFormat("#define Something Other", getLLVMStyleWithColumns(24));
  verifyFormat("#define Something     \\\n"
               "  Other", getLLVMStyleWithColumns(23));
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
  EXPECT_EQ("{}", format("{}"));
}

TEST_F(FormatTest, LayoutSingleUnwrappedLineInMacro) {
  EXPECT_EQ("# define A\\\n  b;",
            format("# define A b;", 11, 2, getLLVMStyleWithColumns(11)));
}

TEST_F(FormatTest, MacroDefinitionInsideStatement) {
  EXPECT_EQ("int x,\n"
            "#define A\n"
            "    y;", format("int x,\n#define A\ny;"));
}

TEST_F(FormatTest, HashInMacroDefinition) {
  verifyFormat("#define A \\\n  b #c;", getLLVMStyleWithColumns(11));
  verifyFormat("#define A \\\n"
               "  {       \\\n"
               "    f(#c);\\\n"
               "  }", getLLVMStyleWithColumns(11));

  verifyFormat("#define A(X)         \\\n"
               "  void function##X()", getLLVMStyleWithColumns(22));

  verifyFormat("#define A(a, b, c)   \\\n"
               "  void a##b##c()", getLLVMStyleWithColumns(22));

  verifyFormat("#define A void # ## #", getLLVMStyleWithColumns(22));
}

TEST_F(FormatTest, IndentPreprocessorDirectivesAtZero) {
  EXPECT_EQ("{\n  {\n#define A\n  }\n}", format("{{\n#define A\n}}"));
}

TEST_F(FormatTest, FormatHashIfNotAtStartOfLine) {
  verifyFormat("{\n  { a #c; }\n}");
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

TEST_F(FormatTest, LayoutStatementsAroundPreprocessorDirectives) {
  EXPECT_EQ("int\n"
            "#define A\n"
            "    a;",
            format("int\n#define A\na;"));
  verifyFormat(
      "functionCallTo(someOtherFunction(\n"
      "    withSomeParameters, whichInSequence,\n"
      "    areLongerThanALine(andAnotherCall,\n"
      "#define A B\n"
      "                       withMoreParamters,\n"
      "                       whichStronglyInfluenceTheLayout),\n"
      "    andMoreParameters), trailing);", getLLVMStyleWithColumns(69));
}

TEST_F(FormatTest, LayoutBlockInsideParens) {
  EXPECT_EQ("functionCall({\n"
            "  int i;\n"
            "});", format(" functionCall ( {int i;} );"));
}

TEST_F(FormatTest, LayoutBlockInsideStatement) {
  EXPECT_EQ("SOME_MACRO { int i; }\n"
            "int i;", format("  SOME_MACRO  {int i;}  int i;"));
}

TEST_F(FormatTest, LayoutNestedBlocks) {
  verifyFormat("void AddOsStrings(unsigned bitmask) {\n"
               "  struct s {\n"
               "    int i;\n"
               "  };\n"
               "  s kBitsToOs[] = { { 10 } };\n"
               "  for (int i = 0; i < 10; ++i)\n"
               "    return;\n"
               "}");
}

TEST_F(FormatTest, PutEmptyBlocksIntoOneLine) {
  EXPECT_EQ("{}", format("{}"));

  // Negative test for enum.
  verifyFormat("enum E {\n};");

  // Note that when there's a missing ';', we still join...
  verifyFormat("enum E {}");
}

//===----------------------------------------------------------------------===//
// Line break tests.
//===----------------------------------------------------------------------===//

TEST_F(FormatTest, FormatsFunctionDefinition) {
  verifyFormat("void f(int a, int b, int c, int d, int e, int f, int g,"
               " int h, int j, int f,\n"
               "       int c, int ddddddddddddd) {\n}");
}

TEST_F(FormatTest, FormatsAwesomeMethodCall) {
  verifyFormat(
      "SomeLongMethodName(SomeReallyLongMethod(\n"
      "    CallOtherReallyLongMethod(parameter, parameter, parameter)),\n"
      "                   SecondLongCall(parameter));");
}

TEST_F(FormatTest, ConstructorInitializers) {
  verifyFormat("Constructor() : Initializer(FitsOnTheLine) {}");
  verifyFormat("Constructor() : Inttializer(FitsOnTheLine) {}",
               getLLVMStyleWithColumns(45));
  verifyFormat("Constructor() : Inttializer(FitsOnTheLine) {\n}",
               getLLVMStyleWithColumns(44));
  verifyFormat("Constructor()\n"
               "    : Inttializer(FitsOnTheLine) {\n}",
               getLLVMStyleWithColumns(43));

  verifyFormat(
      "SomeClass::Constructor()\n"
      "    : aaaaaaaaaaaaa(aaaaaaaaaaaaaa), aaaaaaaaaaaaaaa(aaaaaaaaaaaa) {\n}");

  verifyFormat(
      "SomeClass::Constructor()\n"
      "    : aaaaaaaaaaaaa(aaaaaaaaaaaaaa), aaaaaaaaaaaaa(aaaaaaaaaaaaaa),\n"
      "      aaaaaaaaaaaaa(aaaaaaaaaaaaaa) {\n}");
  verifyGoogleFormat(
      "SomeClass::Constructor()\n"
      "    : aaaaaaaaaaaaa(aaaaaaaaaaaaaa),\n"
      "      aaaaaaaaaaaaa(aaaaaaaaaaaaaa),\n"
      "      aaaaaaaaaaaaa(aaaaaaaaaaaaaa) {\n}");
  verifyGoogleFormat(
      "SomeClass::Constructor()\n"
      "    : aaaaaaaaaaaaa(aaaaaaaaaaaaaa),  // Some comment\n"
      "      aaaaaaaaaaaaa(aaaaaaaaaaaaaa),\n"
      "      aaaaaaaaaaaaa(aaaaaaaaaaaaaa) {\n}");

  verifyFormat(
      "SomeClass::Constructor()\n"
      "    : aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa),\n"
      "      aaaaaaaaaaaaaaa(aaaaaaaaaaaa) {\n}");

  verifyFormat("Constructor()\n"
               "    : aaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaa),\n"
               "      aaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "                               aaaaaaaaaaaaaaaaaaaaaaaaaaa),\n"
               "      aaaaaaaaaaaaaaaaaaaaaaa() {\n}");

  // Here a line could be saved by splitting the second initializer onto two
  // lines, but that is not desireable.
  verifyFormat(
      "Constructor()\n"
      "    : aaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaa),\n"
      "      aaaaaaaaaaa(aaaaaaaaaaa),\n"
      "      aaaaaaaaaaaaaaaaaaaaat(aaaaaaaaaaaaaaaaaaaaaaaaaaaa) {\n}");

  verifyGoogleFormat("MyClass::MyClass(int var)\n"
                     "    : some_var_(var),             // 4 space indent\n"
                     "      some_other_var_(var + 1) {  // lined up\n"
                     "}");

  // This test takes VERY long when memoization is broken.
  std::string input = "Constructor()\n"
                 "    : aaaa(a,\n";
  for (unsigned i = 0, e = 80; i != e; ++i) {
    input += "           a,\n";
  }
  input += "           a) {\n}";
  verifyGoogleFormat(input);
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
               "    aaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaa)) {\n}");

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

TEST_F(FormatTest, FormatsOneParameterPerLineIfNecessary) {
  verifyGoogleFormat(
      "aaaaaaaa(aaaaaaaaaaaaa,\n"
      "         aaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "             aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)),\n"
      "         aaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "             aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)));");
  verifyGoogleFormat(
      "aaaaaaaaaaaaaaa(aaaaaaaaa,\n"
      "                aaaaaaaaa,\n"
      "                aaaaaaaaaaaaaaaaaaaaaaa).aaaaaaaaaaaaaaaaaa();");
  verifyGoogleFormat(
      "somefunction(someotherFunction(ddddddddddddddddddddddddddddddddddd,\n"
      "                               ddddddddddddddddddddddddddddd),\n"
      "             test);");

  verifyGoogleFormat(
      "std::vector<aaaaaaaaaaaaaaaaaaaaaaa,\n"
      "            aaaaaaaaaaaaaaaaaaaaaaa,\n"
      "            aaaaaaaaaaaaaaaaaaaaaaa> aaaaaaaaaaaaaaaaaa;");
  verifyGoogleFormat("a(\"a\"\n"
                     "  \"a\",\n"
                     "  a);");
}

TEST_F(FormatTest, FormatsBuilderPattern) {
  verifyFormat(
      "return llvm::StringSwitch<Reference::Kind>(name)\n"
      "       .StartsWith(\".eh_frame_hdr\", ORDER_EH_FRAMEHDR)\n"
      "       .StartsWith(\".eh_frame\", ORDER_EH_FRAME).StartsWith(\".init\", ORDER_INIT)\n"
      "       .StartsWith(\".fini\", ORDER_FINI).StartsWith(\".hash\", ORDER_HASH)\n"
      "       .Default(ORDER_TEXT);\n");
}

TEST_F(FormatTest, DoesNotBreakTrailingAnnotation) {
  verifyFormat("void aaaaaaaaaaaa(int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "    GUARDED_BY(aaaaaaaaaaaaa);");
  verifyFormat("void aaaaaaaaaaaa(int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) const\n"
               "    GUARDED_BY(aaaaaaaaaaaaa);");
  verifyFormat("void aaaaaaaaaaaa(int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) const\n"
               "    GUARDED_BY(aaaaaaaaaaaaa) {\n}");
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

TEST_F(FormatTest, BreaksConditionalExpressions) {
  verifyFormat(
      "aaaa(aaaaaaaaaaaaaaaaaaaa,\n"
      "     aaaaaaaaaaaaaaaaaaaaaaaaaa ? aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa :\n"
      "         aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat("aaaa(aaaaaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaaaaaaaaaaa ?\n"
               "         aaaaaaaaaaaaaaaaaaaaaaa : aaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaa ? aaaa(aaaaaa) :\n"
      "                                   aaaaaaaaaaaaa);");
}

TEST_F(FormatTest, ConditionalExpressionsInBrackets) {
  verifyFormat("arr[foo ? bar : baz];");
  verifyFormat("f()[foo ? bar : baz];");
  verifyFormat("(a + b)[foo ? bar : baz];");
  verifyFormat("arr[foo ? (4 > 5 ? 4 : 5) : 5 < 5 ? 5 : 7];");
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
      "        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {\n}");
  verifyFormat(
      "if (a) {\n"
      "  f();\n"
      "} else if (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa =\n"
      "               aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {\n"
      "}");

  verifyFormat(
      // FIXME: Does an expression like this ever make sense? If yes, fix.
      "if (int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa = 100000000 +\n"
      "    10000000) {\n}");
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
               "    aaaa.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {\n}");
  verifyFormat(
      "aaaaaaaaaaa->aaaaaaaaa(\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaa,\n"
      "    aaaaaaaaaaaaaaaaaa->aaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaaaa));\n");
}

TEST_F(FormatTest, WrapsTemplateDeclarations) {
  verifyFormat("template <typename T>\n"
               "virtual void loooooooooooongFunction(int Param1, int Param2);");
  verifyFormat(
      "template <typename T>\n"
      "void f(int Paaaaaaaaaaaaaaaaaaaaaaaaaaaaaaram1,\n"
      "       int Paaaaaaaaaaaaaaaaaaaaaaaaaaaaaaram2);");
  verifyFormat(
      "template <typename T>\n"
      "void looooooooooooooooooooongFunction(int Paaaaaaaaaaaaaaaaaaaaram1,\n"
      "                                      int Paaaaaaaaaaaaaaaaaaaaram2);");
  verifyFormat(
      "template <typename T>\n"
      "aaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaa,\n"
      "                    aaaaaaaaaaaaaaaaaaaaaaaaaa<T>::aaaaaaaaaa,\n"
      "                    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat("template <typename T>\n"
               "void aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    int aaaaaaaaaaaaaaaaa);");
  verifyFormat(
      "template <typename T1, typename T2 = char, typename T3 = char,\n"
      "          typename T4 = char>\n"
      "void f();");
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
  verifyFormat("template <typename T> void f() {}");
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
  verifyFormat("(a->f())++;");
  verifyFormat("a[42]++;");
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

  verifyFormat("const NSPoint kBrowserFrameViewPatternOffset = { -5, +3 };");
  verifyFormat("const NSPoint kBrowserFrameViewPatternOffset = { +5, -3 };");

  verifyFormat("int a = /* confusing comment */ -1;");
  // FIXME: The space after 'i' is wrong, but hopefully, this is a rare case.
  verifyFormat("int a = i /* confusing comment */++;");
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

TEST_F(FormatTest, UnderstandsNewAndDelete) {
  verifyFormat("A *a = new A;");
  verifyFormat("A *a = new (placement) A;");
  verifyFormat("delete a;");
  verifyFormat("delete (A *)a;");
}

TEST_F(FormatTest, UnderstandsUsesOfStarAndAmp) {
  verifyFormat("int *f(int *a) {}");
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
  verifyFormat("int main(int argc, char **argv) {}");
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
  verifyFormat("a[4] * b;");
  verifyFormat("f() * b;");
  verifyFormat("a * [self dostuff];");
  verifyFormat("a * (a + b);");
  verifyFormat("(a *)(a + b);");
  verifyFormat("int *pa = (int *)&a;");

  verifyFormat("InvalidRegions[*R] = 0;");

  verifyFormat("A<int *> a;");
  verifyFormat("A<int **> a;");
  verifyFormat("A<int *, int *> a;");
  verifyFormat("A<int **, int **> a;");

  verifyFormat(
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
      "    aaaaaaaaaaaaaaaaaaaaaaaaaaaa, *aaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");

  verifyGoogleFormat("int main(int argc, char** argv) {}");
  verifyGoogleFormat("A<int*> a;");
  verifyGoogleFormat("A<int**> a;");
  verifyGoogleFormat("A<int*, int*> a;");
  verifyGoogleFormat("A<int**, int**> a;");
  verifyGoogleFormat("f(b ? *c : *d);");
  verifyGoogleFormat("int a = b ? *c : *d;");
  verifyGoogleFormat("Type* t = **x;");
  verifyGoogleFormat("Type* t = *++*x;");
  verifyGoogleFormat("*++*x;");
  verifyGoogleFormat("Type* t = const_cast<T*>(&*x);");
  verifyGoogleFormat("Type* t = x++ * y;");

  verifyFormat("a = *(x + y);");
  verifyFormat("a = &(x + y);");
  verifyFormat("*(x + y).call();");
  verifyFormat("&(x + y)->call();");
  verifyFormat("&(*I).first");

  verifyFormat("f(b * /* confusing comment */ ++c);");
  verifyFormat(
      "int *MyValues = {\n"
      "  *A, // Operator detection might be confused by the '{'\n"
      "  *BB // Operator detection might be confused by previous comment\n"
      "};");

  verifyFormat("if (int *a = &b)");
  verifyFormat("if (int &a = *b)");
  verifyFormat("if (a & b[i])");
  verifyFormat("if (a::b::c::d & b[i])");
  verifyFormat("if (*b[i])");
  verifyFormat("if (int *a = (&b))");
  verifyFormat("while (int *a = &b)");

  verifyFormat("A = new SomeType *[Length]();");
  verifyGoogleFormat("A = new SomeType* [Length]();");
}

TEST_F(FormatTest, FormatsCasts) {
  verifyFormat("Type *A = static_cast<Type *>(P);");
  verifyFormat("Type *A = (Type *)P;");
  verifyFormat("Type *A = (vector<Type *, int *>)P;");
  verifyFormat("int a = (int)(2.0f);");

  // FIXME: These also need to be identified.
  verifyFormat("int a = (int) 2.0f;");
  verifyFormat("int a = (int) * b;");

  // These are not casts.
  verifyFormat("void f(int *) {}");
  verifyFormat("void f(int *);");
  verifyFormat("void f(int *) = 0;");
  verifyFormat("void f(SmallVector<int>) {}");
  verifyFormat("void f(SmallVector<int>);");
  verifyFormat("void f(SmallVector<int>) = 0;");
}

TEST_F(FormatTest, FormatsFunctionTypes) {
  // FIXME: Determine the cases that need a space after the return type and fix.
  verifyFormat("A<bool()> a;");
  verifyFormat("A<SomeType()> a;");
  verifyFormat("A<void(*)(int, std::string)> a;");

  verifyFormat("int(*func)(void *);");
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
  verifyFormat("#include <string>\n"
               "#include <a/b/c.h>\n"
               "#include \"a/b/string\"\n"
               "#include \"string.h\"\n"
               "#include \"string.h\"\n"
               "#include <a-a>\n"
               "#include < path with space >\n");

  verifyFormat("#import <string>");
  verifyFormat("#import <a/b/c.h>");
  verifyFormat("#import \"a/b/string\"");
  verifyFormat("#import \"string.h\"");
  verifyFormat("#import \"string.h\"");
}

//===----------------------------------------------------------------------===//
// Error recovery tests.
//===----------------------------------------------------------------------===//

TEST_F(FormatTest, IncorrectCodeTrailingStuff) {
  verifyFormat("void f() { return; }\n42");
  verifyFormat("void f() {\n"
               "  if (0)\n"
               "    return;\n"
               "}\n"
               "42");
  verifyFormat("void f() { return }\n42");
  verifyFormat("void f() {\n"
               "  if (0)\n"
               "    return\n"
               "}\n"
               "42");
}

TEST_F(FormatTest, IncorrectCodeMissingSemicolon) {
  EXPECT_EQ("void f() { return }", format("void  f ( )  {  return  }"));
  EXPECT_EQ("void f() {\n"
            "  if (a)\n"
            "    return\n"
            "}", format("void  f  (  )  {  if  ( a )  return  }"));
  EXPECT_EQ("namespace N { void f() }", format("namespace  N  {  void f()  }"));
  EXPECT_EQ("namespace N {\n"
            "void f() {}\n"
            "void g()\n"
            "}", format("namespace N  { void f( ) { } void g( ) }"));
}

TEST_F(FormatTest, IndentationWithinColumnLimitNotPossible) {
  verifyFormat("int aaaaaaaa =\n"
               "    // Overly long comment\n"
               "    b;", getLLVMStyleWithColumns(20));
  verifyFormat("function(\n"
               "    ShortArgument,\n"
               "    LoooooooooooongArgument);\n", getLLVMStyleWithColumns(20));
}

TEST_F(FormatTest, IncorrectAccessSpecifier) {
  verifyFormat("public:");
  verifyFormat("class A {\n"
               "public\n"
               "  void f() {}\n"
               "};");
  verifyFormat("public\n"
               "int qwerty;");
  verifyFormat("public\n"
               "B {}");
  verifyFormat("public\n"
               "{}");
  verifyFormat("public\n"
               "B { int x; }");
}

TEST_F(FormatTest, IncorrectCodeUnbalancedBraces) {
  verifyFormat("{");
}

TEST_F(FormatTest, IncorrectCodeDoNoWhile) {
  verifyFormat("do {\n}");
  verifyFormat("do {\n}\n"
               "f();");
  verifyFormat("do {\n}\n"
               "wheeee(fun);");
  verifyFormat("do {\n"
               "  f();\n"
               "}");
}

TEST_F(FormatTest, IncorrectCodeMissingParens) {
  verifyFormat("if {\n  foo;\n  foo();\n}");
  verifyFormat("switch {\n  foo;\n  foo();\n}");
  verifyFormat("for {\n  foo;\n  foo();\n}");
  verifyFormat("while {\n  foo;\n  foo();\n}");
  verifyFormat("do {\n  foo;\n  foo();\n} while;");
}

TEST_F(FormatTest, DoesNotTouchUnwrappedLinesWithErrors) {
  verifyFormat("namespace {\n"
               "class Foo {  Foo  ( }; }  // comment");
}

TEST_F(FormatTest, IncorrectCodeErrorDetection) {
  EXPECT_EQ("{\n{}\n", format("{\n{\n}\n"));
  EXPECT_EQ("{\n  {}\n", format("{\n  {\n}\n"));
  EXPECT_EQ("{\n  {}\n", format("{\n  {\n  }\n"));
  EXPECT_EQ("{\n  {}\n  }\n}\n", format("{\n  {\n    }\n  }\n}\n"));

  EXPECT_EQ("{\n"
            "    {\n"
            " breakme(\n"
            "     qwe);\n"
            "}\n", format("{\n"
                          "    {\n"
                          " breakme(qwe);\n"
                          "}\n", getLLVMStyleWithColumns(10)));
}

TEST_F(FormatTest, LayoutCallsInsideBraceInitializers) {
  verifyFormat(
      "int x = {\n"
      "  avariable,\n"
      "  b(alongervariable)\n"
      "};", getLLVMStyleWithColumns(25));
}

TEST_F(FormatTest, LayoutBraceInitializersInReturnStatement) {
  verifyFormat("return (a)(b) { 1, 2, 3 };");
}

TEST_F(FormatTest, LayoutTokensFollowingBlockInParentheses) {
  verifyFormat(
      "Aaa({\n"
      "  int i;\n"
      "}, aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb,\n"
      "                                    ccccccccccccccccc));");
}

TEST_F(FormatTest, PullTrivialFunctionDefinitionsIntoSingleLine) {
  verifyFormat("void f() { return 42; }");
  verifyFormat("void f() {\n"
               "  // Comment\n"
               "}");
  verifyFormat("{\n"
               "#error {\n"
               "  int a;\n"
               "}");
  verifyFormat("{\n"
               "  int a;\n"
               "#error {\n"
               "}");

  verifyFormat("void f() { return 42; }", getLLVMStyleWithColumns(23));
  verifyFormat("void f() {\n  return 42;\n}", getLLVMStyleWithColumns(22));

  verifyFormat("void f() {}", getLLVMStyleWithColumns(11));
  verifyFormat("void f() {\n}", getLLVMStyleWithColumns(10));
}

TEST_F(FormatTest, UnderstandContextOfRecordTypeKeywords) {
  // Elaborate type variable declarations.
  verifyFormat("struct foo a = { bar };\nint n;");
  verifyFormat("class foo a = { bar };\nint n;");
  verifyFormat("union foo a = { bar };\nint n;");

  // Elaborate types inside function definitions.
  verifyFormat("struct foo f() {}\nint n;");
  verifyFormat("class foo f() {}\nint n;");
  verifyFormat("union foo f() {}\nint n;");

  // Templates.
  verifyFormat("template <class X> void f() {}\nint n;");
  verifyFormat("template <struct X> void f() {}\nint n;");
  verifyFormat("template <union X> void f() {}\nint n;");

  // Actual definitions...
  verifyFormat("struct {\n} n;");
  verifyFormat(
      "template <template <class T, class Y>, class Z> class X {\n} n;");
  verifyFormat("union Z {\n  int n;\n} x;");
  verifyFormat("class MACRO Z {\n} n;");
  verifyFormat("class MACRO(X) Z {\n} n;");
  verifyFormat("class __attribute__(X) Z {\n} n;");
  verifyFormat("class __declspec(X) Z {\n} n;");

  // Redefinition from nested context:
  verifyFormat("class A::B::C {\n} n;");

  // Template definitions.
  // FIXME: This is still incorrectly handled at the formatter side.
  verifyFormat("template <> struct X < 15, i < 3 && 42 < 50 && 33<28> {\n};");

  // FIXME:
  // This now gets parsed incorrectly as class definition.
  // verifyFormat("class A<int> f() {\n}\nint n;");

  // Elaborate types where incorrectly parsing the structural element would
  // break the indent.
  verifyFormat("if (true)\n"
               "  class X x;\n"
               "else\n"
               "  f();\n");
}

TEST_F(FormatTest, DoNotInterfereWithErrorAndWarning) {
  verifyFormat("#error Leave     all         white!!!!! space* alone!\n");
  verifyFormat("#warning Leave     all         white!!!!! space* alone!\n");
  EXPECT_EQ("#error 1", format("  #  error   1"));
  EXPECT_EQ("#warning 1", format("  #  warning 1"));
}

TEST_F(FormatTest, MergeHandlingInTheFaceOfPreprocessorDirectives) {
  FormatStyle AllowsMergedIf = getGoogleStyle();
  AllowsMergedIf.AllowShortIfStatementsOnASingleLine = true;
  verifyFormat("void f() { f(); }\n#error E", AllowsMergedIf);
  verifyFormat("if (true) return 42;\n#error E", AllowsMergedIf);
  verifyFormat("if (true)\n#error E\n  return 42;", AllowsMergedIf);
  EXPECT_EQ("if (true) return 42;",
            format("if (true)\nreturn 42;", AllowsMergedIf));
  FormatStyle ShortMergedIf = AllowsMergedIf;
  ShortMergedIf.ColumnLimit = 25;
  verifyFormat("#define A               \\\n"
               "  if (true) return 42;", ShortMergedIf);
  verifyFormat("#define A               \\\n"
               "  f();                  \\\n"
               "  if (true)\n"
               "#define B", ShortMergedIf);
  verifyFormat("#define A               \\\n"
               "  f();                  \\\n"
               "  if (true)\n"
               "g();", ShortMergedIf);
  verifyFormat("{\n"
               "#ifdef A\n"
               "  // Comment\n"
               "  if (true) continue;\n"
               "#endif\n"
               "  // Comment\n"
               "  if (true) continue;", ShortMergedIf);
}

TEST_F(FormatTest, BlockCommentsInControlLoops) {
  verifyFormat("if (0) /* a comment in a strange place */ {\n"
               "  f();\n"
               "}");
  verifyFormat("if (0) /* a comment in a strange place */ {\n"
               "  f();\n"
               "} /* another comment */ else /* comment #3 */ {\n"
               "  g();\n"
               "}");
  verifyFormat("while (0) /* a comment in a strange place */ {\n"
               "  f();\n"
               "}");
  verifyFormat("for (;;) /* a comment in a strange place */ {\n"
               "  f();\n"
               "}");
  verifyFormat("do /* a comment in a strange place */ {\n"
               "  f();\n"
               "} /* another comment */ while (0);");
}

TEST_F(FormatTest, BlockComments) {
  EXPECT_EQ("/* */ /* */ /* */\n/* */ /* */ /* */",
            format("/* *//* */  /* */\n/* *//* */  /* */"));
  EXPECT_EQ("/* */ a /* */ b;",
            format("  /* */  a/* */  b;"));
  EXPECT_EQ("#define A /*   */\\\n"
            "  b\n"
            "/* */\n"
            "someCall(\n"
            "    parameter);",
            format("#define A /*   */ b\n"
                   "/* */\n"
                   "someCall(parameter);", getLLVMStyleWithColumns(15)));

  EXPECT_EQ("#define A\n"
            "/* */ someCall(\n"
            "    parameter);",
            format("#define A\n"
                   "/* */someCall(parameter);", getLLVMStyleWithColumns(15)));
}

TEST_F(FormatTest, Fuck) {
  verifyFormat("void f(int *a);");
  verifyFormat("void f() { f(fint * b); }");
}

//===----------------------------------------------------------------------===//
// Objective-C tests.
//===----------------------------------------------------------------------===//

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

  verifyFormat("- (int)sum:(vector<int>)numbers;");
  verifyGoogleFormat("- (void)setDelegate:(id<Protocol>)delegate;");
  // FIXME: In LLVM style, there should be a space in front of a '<' for ObjC
  // protocol lists (but not for template classes):
  //verifyFormat("- (void)setDelegate:(id <Protocol>)delegate;");

  verifyFormat("- (int(*)())foo:(int(*)())f;");
  verifyGoogleFormat("- (int(*)())foo:(int(*)())foo;");

  // If there's no return type (very rare in practice!), LLVM and Google style
  // agree.
  verifyFormat("- foo:(int)f;");
  verifyGoogleFormat("- foo:(int)foo;");
}

TEST_F(FormatTest, FormatObjCBlocks) {
  verifyFormat("int (^Block)(int, int);");
  verifyFormat("int (^Block1)(int, int) = ^(int i, int j)");
}

TEST_F(FormatTest, FormatObjCInterface) {
  verifyFormat("@interface Foo : NSObject <NSSomeDelegate> {\n"
               "@public\n"
               "  int field1;\n"
               "@protected\n"
               "  int field2;\n"
               "@private\n"
               "  int field3;\n"
               "@package\n"
               "  int field4;\n"
               "}\n"
               "+ (id)init;\n"
               "@end");

  verifyGoogleFormat("@interface Foo : NSObject<NSSomeDelegate> {\n"
                     " @public\n"
                     "  int field1;\n"
                     " @protected\n"
                     "  int field2;\n"
                     " @private\n"
                     "  int field3;\n"
                     " @package\n"
                     "  int field4;\n"
                     "}\n"
                     "+ (id)init;\n"
                     "@end");

  verifyFormat("@interface /* wait for it */ Foo\n"
               "+ (id)init;\n"
               "// Look, a comment!\n"
               "- (int)answerWith:(int)i;\n"
               "@end");

  verifyFormat("@interface Foo\n"
               "@end\n"
               "@interface Bar\n"
               "@end");

  verifyFormat("@interface Foo : Bar\n"
               "+ (id)init;\n"
               "@end");

  verifyFormat("@interface Foo : /**/ Bar /**/ <Baz, /**/ Quux>\n"
               "+ (id)init;\n"
               "@end");

  verifyGoogleFormat("@interface Foo : Bar<Baz, Quux>\n"
                     "+ (id)init;\n"
                     "@end");

  verifyFormat("@interface Foo (HackStuff)\n"
               "+ (id)init;\n"
               "@end");

  verifyFormat("@interface Foo ()\n"
               "+ (id)init;\n"
               "@end");

  verifyFormat("@interface Foo (HackStuff) <MyProtocol>\n"
               "+ (id)init;\n"
               "@end");

  verifyGoogleFormat("@interface Foo (HackStuff)<MyProtocol>\n"
                     "+ (id)init;\n"
                     "@end");

  verifyFormat("@interface Foo {\n"
               "  int _i;\n"
               "}\n"
               "+ (id)init;\n"
               "@end");

  verifyFormat("@interface Foo : Bar {\n"
               "  int _i;\n"
               "}\n"
               "+ (id)init;\n"
               "@end");

  verifyFormat("@interface Foo : Bar <Baz, Quux> {\n"
               "  int _i;\n"
               "}\n"
               "+ (id)init;\n"
               "@end");

  verifyFormat("@interface Foo (HackStuff) {\n"
               "  int _i;\n"
               "}\n"
               "+ (id)init;\n"
               "@end");

  verifyFormat("@interface Foo () {\n"
               "  int _i;\n"
               "}\n"
               "+ (id)init;\n"
               "@end");

  verifyFormat("@interface Foo (HackStuff) <MyProtocol> {\n"
               "  int _i;\n"
               "}\n"
               "+ (id)init;\n"
               "@end");
}

TEST_F(FormatTest, FormatObjCImplementation) {
  verifyFormat("@implementation Foo : NSObject {\n"
               "@public\n"
               "  int field1;\n"
               "@protected\n"
               "  int field2;\n"
               "@private\n"
               "  int field3;\n"
               "@package\n"
               "  int field4;\n"
               "}\n"
               "+ (id)init {\n}\n"
               "@end");

  verifyGoogleFormat("@implementation Foo : NSObject {\n"
                     " @public\n"
                     "  int field1;\n"
                     " @protected\n"
                     "  int field2;\n"
                     " @private\n"
                     "  int field3;\n"
                     " @package\n"
                     "  int field4;\n"
                     "}\n"
                     "+ (id)init {\n}\n"
                     "@end");

  verifyFormat("@implementation Foo\n"
               "+ (id)init {\n"
               "  if (true)\n"
               "    return nil;\n"
               "}\n"
               "// Look, a comment!\n"
               "- (int)answerWith:(int)i {\n"
               "  return i;\n"
               "}\n"
               "+ (int)answerWith:(int)i {\n"
               "  return i;\n"
               "}\n"
               "@end");

  verifyFormat("@implementation Foo\n"
               "@end\n"
               "@implementation Bar\n"
               "@end");

  verifyFormat("@implementation Foo : Bar\n"
               "+ (id)init {\n}\n"
               "- (void)foo {\n}\n"
               "@end");

  verifyFormat("@implementation Foo {\n"
               "  int _i;\n"
               "}\n"
               "+ (id)init {\n}\n"
               "@end");

  verifyFormat("@implementation Foo : Bar {\n"
               "  int _i;\n"
               "}\n"
               "+ (id)init {\n}\n"
               "@end");

  verifyFormat("@implementation Foo (HackStuff)\n"
               "+ (id)init {\n}\n"
               "@end");
}

TEST_F(FormatTest, FormatObjCProtocol) {
  verifyFormat("@protocol Foo\n"
               "@property(weak) id delegate;\n"
               "- (NSUInteger)numberOfThings;\n"
               "@end");

  verifyFormat("@protocol MyProtocol <NSObject>\n"
               "- (NSUInteger)numberOfThings;\n"
               "@end");

  verifyGoogleFormat("@protocol MyProtocol<NSObject>\n"
                     "- (NSUInteger)numberOfThings;\n"
                     "@end");

  verifyFormat("@protocol Foo;\n"
               "@protocol Bar;\n");

  verifyFormat("@protocol Foo\n"
               "@end\n"
               "@protocol Bar\n"
               "@end");

  verifyFormat("@protocol myProtocol\n"
               "- (void)mandatoryWithInt:(int)i;\n"
               "@optional\n"
               "- (void)optional;\n"
               "@required\n"
               "- (void)required;\n"
               "@optional\n"
               "@property(assign) int madProp;\n"
               "@end\n");
}

TEST_F(FormatTest, FormatObjCMethodExpr) {
  verifyFormat("[foo bar:baz];");
  verifyFormat("return [foo bar:baz];");
  verifyFormat("f([foo bar:baz]);");
  verifyFormat("f(2, [foo bar:baz]);");
  verifyFormat("f(2, a ? b : c);");
  verifyFormat("[[self initWithInt:4] bar:[baz quux:arrrr]];");

  verifyFormat("[foo bar:baz], [foo bar:baz];");
  verifyFormat("[foo bar:baz] = [foo bar:baz];");
  verifyFormat("[foo bar:baz] *= [foo bar:baz];");
  verifyFormat("[foo bar:baz] /= [foo bar:baz];");
  verifyFormat("[foo bar:baz] %= [foo bar:baz];");
  verifyFormat("[foo bar:baz] += [foo bar:baz];");
  verifyFormat("[foo bar:baz] -= [foo bar:baz];");
  verifyFormat("[foo bar:baz] <<= [foo bar:baz];");
  verifyFormat("[foo bar:baz] >>= [foo bar:baz];");
  verifyFormat("[foo bar:baz] &= [foo bar:baz];");
  verifyFormat("[foo bar:baz] ^= [foo bar:baz];");
  verifyFormat("[foo bar:baz] |= [foo bar:baz];");
  verifyFormat("[foo bar:baz] ? [foo bar:baz] : [foo bar:baz];");
  verifyFormat("[foo bar:baz] || [foo bar:baz];");
  verifyFormat("[foo bar:baz] && [foo bar:baz];");
  verifyFormat("[foo bar:baz] | [foo bar:baz];");
  verifyFormat("[foo bar:baz] ^ [foo bar:baz];");
  verifyFormat("[foo bar:baz] & [foo bar:baz];");
  verifyFormat("[foo bar:baz] == [foo bar:baz];");
  verifyFormat("[foo bar:baz] != [foo bar:baz];");
  verifyFormat("[foo bar:baz] >= [foo bar:baz];");
  verifyFormat("[foo bar:baz] <= [foo bar:baz];");
  verifyFormat("[foo bar:baz] > [foo bar:baz];");
  verifyFormat("[foo bar:baz] < [foo bar:baz];");
  verifyFormat("[foo bar:baz] >> [foo bar:baz];");
  verifyFormat("[foo bar:baz] << [foo bar:baz];");
  verifyFormat("[foo bar:baz] - [foo bar:baz];");
  verifyFormat("[foo bar:baz] + [foo bar:baz];");
  verifyFormat("[foo bar:baz] * [foo bar:baz];");
  verifyFormat("[foo bar:baz] / [foo bar:baz];");
  verifyFormat("[foo bar:baz] % [foo bar:baz];");
  // Whew!

  verifyFormat("[self stuffWithInt:(4 + 2) float:4.5];");
  verifyFormat("[self stuffWithInt:a ? b : c float:4.5];");
  verifyFormat("[self stuffWithInt:a ? [self foo:bar] : c];");
  verifyFormat("[self stuffWithInt:a ? (e ? f : g) : c];");
  verifyFormat("[cond ? obj1 : obj2 methodWithParam:param]");
  verifyFormat("[button setAction:@selector(zoomOut:)];");
  verifyFormat("[color getRed:&r green:&g blue:&b alpha:&a];");
  
  verifyFormat("arr[[self indexForFoo:a]];");
  verifyFormat("throw [self errorFor:a];");
  verifyFormat("@throw [self errorFor:a];");

  // This tests that the formatter doesn't break after "backing" but before ":",
  // which would be at 80 columns.
  verifyFormat(
      "void f() {\n"
      "  if ((self = [super initWithContentRect:contentRect styleMask:styleMask\n"
      "                  backing:NSBackingStoreBuffered defer:YES]))");
  
  verifyFormat("[foo checkThatBreakingAfterColonWorksOk:\n"
               "    [bar ifItDoes:reduceOverallLineLengthLikeInThisCase]];");
  
}

TEST_F(FormatTest, ObjCAt) {
  verifyFormat("@autoreleasepool");
  verifyFormat("@catch");
  verifyFormat("@class");
  verifyFormat("@compatibility_alias");
  verifyFormat("@defs");
  verifyFormat("@dynamic");
  verifyFormat("@encode");
  verifyFormat("@end");
  verifyFormat("@finally");
  verifyFormat("@implementation");
  verifyFormat("@import");
  verifyFormat("@interface");
  verifyFormat("@optional");
  verifyFormat("@package");
  verifyFormat("@private");
  verifyFormat("@property");
  verifyFormat("@protected");
  verifyFormat("@protocol");
  verifyFormat("@public");
  verifyFormat("@required");
  verifyFormat("@selector");
  verifyFormat("@synchronized");
  verifyFormat("@synthesize");
  verifyFormat("@throw");
  verifyFormat("@try");

  verifyFormat("@\"String\"");
  verifyFormat("@1");
  verifyFormat("@+4.8");
  verifyFormat("@-4");
  verifyFormat("@1LL");
  verifyFormat("@.5");
  verifyFormat("@'c'");
  verifyFormat("@true");
  verifyFormat("NSNumber *smallestInt = @(-INT_MAX - 1);");
  // FIXME: Array and dictionary literals need more work.
  verifyFormat("@[");
  verifyFormat("@{");

  EXPECT_EQ("@interface", format("@ interface"));

  // The precise formatting of this doesn't matter, nobody writes code like
  // this.
  verifyFormat("@ /*foo*/ interface");
}

TEST_F(FormatTest, ObjCSnippets) {
  verifyFormat("@autoreleasepool {\n"
               "  foo();\n"
               "}");
  verifyFormat("@class Foo, Bar;");
  verifyFormat("@compatibility_alias AliasName ExistingClass;");
  verifyFormat("@dynamic textColor;");
  verifyFormat("char *buf1 = @encode(int *);");
  verifyFormat("char *buf1 = @encode(typeof(4 * 5));");
  // FIXME: Enable once PR14884 is fixed:
  //verifyFormat("char *buf1 = @encode(int **);");
  verifyFormat("Protocol *proto = @protocol(p1);");
  verifyFormat("SEL s = @selector(foo:);");
  verifyFormat("@synchronized(self) {\n"
               "  f();\n"
               "}");

  verifyFormat("@synthesize dropArrowPosition = dropArrowPosition_;");
  verifyGoogleFormat("@synthesize dropArrowPosition = dropArrowPosition_;");

  verifyFormat("@property(assign, nonatomic) CGFloat hoverAlpha;");
  verifyFormat("@property(assign, getter=isEditable) BOOL editable;");
  verifyGoogleFormat("@property(assign, getter=isEditable) BOOL editable;");
}

} // end namespace tooling
} // end namespace clang
