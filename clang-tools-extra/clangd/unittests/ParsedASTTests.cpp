//===-- ParsedASTTests.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These tests cover clangd's logic to build a TU, which generally uses the APIs
// in ParsedAST and Preamble, via the TestTU helper.
//
//===----------------------------------------------------------------------===//

#include "../../clang-tidy/ClangTidyCheck.h"
#include "../../clang-tidy/ClangTidyModule.h"
#include "../../clang-tidy/ClangTidyModuleRegistry.h"
#include "AST.h"
#include "Annotations.h"
#include "Compiler.h"
#include "Diagnostics.h"
#include "Headers.h"
#include "ParsedAST.h"
#include "Preamble.h"
#include "SourceCode.h"
#include "TestFS.h"
#include "TestTU.h"
#include "TidyProvider.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Token.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ScopedPrinter.h"
#include "gmock/gmock-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::IsEmpty;

MATCHER_P(DeclNamed, Name, "") {
  if (NamedDecl *ND = dyn_cast<NamedDecl>(arg))
    if (ND->getName() == Name)
      return true;
  if (auto *Stream = result_listener->stream()) {
    llvm::raw_os_ostream OS(*Stream);
    arg->dump(OS);
  }
  return false;
}

MATCHER_P(DeclKind, Kind, "") {
  if (NamedDecl *ND = dyn_cast<NamedDecl>(arg))
    if (ND->getDeclKindName() == llvm::StringRef(Kind))
      return true;
  if (auto *Stream = result_listener->stream()) {
    llvm::raw_os_ostream OS(*Stream);
    arg->dump(OS);
  }
  return false;
}

// Matches if the Decl has template args equal to ArgName. If the decl is a
// NamedDecl and ArgName is an empty string it also matches.
MATCHER_P(WithTemplateArgs, ArgName, "") {
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(arg)) {
    if (const auto *Args = FD->getTemplateSpecializationArgs()) {
      std::string SpecializationArgs;
      // Without the PrintingPolicy "bool" will be printed as "_Bool".
      LangOptions LO;
      PrintingPolicy Policy(LO);
      Policy.adjustForCPlusPlus();
      for (const auto &Arg : Args->asArray()) {
        if (SpecializationArgs.size() > 0)
          SpecializationArgs += ",";
        SpecializationArgs += Arg.getAsType().getAsString(Policy);
      }
      if (Args->size() == 0)
        return ArgName == SpecializationArgs;
      return ArgName == "<" + SpecializationArgs + ">";
    }
  }
  if (const NamedDecl *ND = dyn_cast<NamedDecl>(arg))
    return printTemplateSpecializationArgs(*ND) == ArgName;
  return false;
}

MATCHER_P(RangeIs, R, "") {
  return arg.beginOffset() == R.Begin && arg.endOffset() == R.End;
}

MATCHER(EqInc, "") {
  Inclusion Actual = testing::get<0>(arg);
  Inclusion Expected = testing::get<1>(arg);
  return std::tie(Actual.HashLine, Actual.Written) ==
         std::tie(Expected.HashLine, Expected.Written);
}

TEST(ParsedASTTest, TopLevelDecls) {
  TestTU TU;
  TU.HeaderCode = R"(
    int header1();
    int header2;
  )";
  TU.Code = R"cpp(
    int main();
    template <typename> bool X = true;
  )cpp";
  auto AST = TU.build();
  EXPECT_THAT(AST.getLocalTopLevelDecls(),
              testing::UnorderedElementsAreArray(
                  {AllOf(DeclNamed("main"), DeclKind("Function")),
                   AllOf(DeclNamed("X"), DeclKind("VarTemplate"))}));
}

TEST(ParsedASTTest, DoesNotGetIncludedTopDecls) {
  TestTU TU;
  TU.HeaderCode = R"cpp(
    #define LL void foo(){}
    template<class T>
    struct H {
      H() {}
      LL
    };
  )cpp";
  TU.Code = R"cpp(
    int main() {
      H<int> h;
      h.foo();
    }
  )cpp";
  auto AST = TU.build();
  EXPECT_THAT(AST.getLocalTopLevelDecls(), ElementsAre(DeclNamed("main")));
}

TEST(ParsedASTTest, DoesNotGetImplicitTemplateTopDecls) {
  TestTU TU;
  TU.Code = R"cpp(
    template<typename T>
    void f(T) {}
    void s() {
      f(10UL);
    }
  )cpp";

  auto AST = TU.build();
  EXPECT_THAT(AST.getLocalTopLevelDecls(),
              ElementsAre(DeclNamed("f"), DeclNamed("s")));
}

TEST(ParsedASTTest,
     GetsExplicitInstantiationAndSpecializationTemplateTopDecls) {
  TestTU TU;
  TU.Code = R"cpp(
    template <typename T>
    void f(T) {}
    template<>
    void f(bool);
    template void f(double);

    template <class T>
    struct V {};
    template<class T>
    struct V<T*> {};
    template <>
    struct V<bool> {};

    template<class T>
    T foo = T(10);
    int i = foo<int>;
    double d = foo<double>;

    template <class T>
    int foo<T*> = 0;
    template <>
    int foo<bool> = 0;
  )cpp";

  auto AST = TU.build();
  EXPECT_THAT(
      AST.getLocalTopLevelDecls(),
      ElementsAreArray({AllOf(DeclNamed("f"), WithTemplateArgs("")),
                        AllOf(DeclNamed("f"), WithTemplateArgs("<bool>")),
                        AllOf(DeclNamed("f"), WithTemplateArgs("<double>")),
                        AllOf(DeclNamed("V"), WithTemplateArgs("")),
                        AllOf(DeclNamed("V"), WithTemplateArgs("<T *>")),
                        AllOf(DeclNamed("V"), WithTemplateArgs("<bool>")),
                        AllOf(DeclNamed("foo"), WithTemplateArgs("")),
                        AllOf(DeclNamed("i"), WithTemplateArgs("")),
                        AllOf(DeclNamed("d"), WithTemplateArgs("")),
                        AllOf(DeclNamed("foo"), WithTemplateArgs("<T *>")),
                        AllOf(DeclNamed("foo"), WithTemplateArgs("<bool>"))}));
}

TEST(ParsedASTTest, IgnoresDelayedTemplateParsing) {
  auto TU = TestTU::withCode(R"cpp(
    template <typename T> void xxx() {
      int yyy = 0;
    }
  )cpp");
  TU.ExtraArgs.push_back("-fdelayed-template-parsing");
  auto AST = TU.build();
  EXPECT_EQ(Decl::Var, findUnqualifiedDecl(AST, "yyy").getKind());
}

TEST(ParsedASTTest, TokensAfterPreamble) {
  TestTU TU;
  TU.AdditionalFiles["foo.h"] = R"(
    int foo();
  )";
  TU.Code = R"cpp(
      #include "foo.h"
      first_token;
      void test() {
        // error-ok: invalid syntax, just examining token stream
      }
      last_token
)cpp";
  auto AST = TU.build();
  const syntax::TokenBuffer &T = AST.getTokens();
  const auto &SM = AST.getSourceManager();

  ASSERT_GT(T.expandedTokens().size(), 2u);
  // Check first token after the preamble.
  EXPECT_EQ(T.expandedTokens().front().text(SM), "first_token");
  // Last token is always 'eof'.
  EXPECT_EQ(T.expandedTokens().back().kind(), tok::eof);
  // Check the token before 'eof'.
  EXPECT_EQ(T.expandedTokens().drop_back().back().text(SM), "last_token");

  // The spelled tokens for the main file should have everything.
  auto Spelled = T.spelledTokens(SM.getMainFileID());
  ASSERT_FALSE(Spelled.empty());
  EXPECT_EQ(Spelled.front().kind(), tok::hash);
  EXPECT_EQ(Spelled.back().text(SM), "last_token");
}

TEST(ParsedASTTest, NoCrashOnTokensWithTidyCheck) {
  TestTU TU;
  // this check runs the preprocessor, we need to make sure it does not break
  // our recording logic.
  TU.ClangTidyProvider = addTidyChecks("modernize-use-trailing-return-type");
  TU.Code = "inline int foo() {}";

  auto AST = TU.build();
  const syntax::TokenBuffer &T = AST.getTokens();
  const auto &SM = AST.getSourceManager();

  ASSERT_GT(T.expandedTokens().size(), 7u);
  // Check first token after the preamble.
  EXPECT_EQ(T.expandedTokens().front().text(SM), "inline");
  // Last token is always 'eof'.
  EXPECT_EQ(T.expandedTokens().back().kind(), tok::eof);
  // Check the token before 'eof'.
  EXPECT_EQ(T.expandedTokens().drop_back().back().text(SM), "}");
}

TEST(ParsedASTTest, CanBuildInvocationWithUnknownArgs) {
  MockFS FS;
  FS.Files = {{testPath("foo.cpp"), "void test() {}"}};
  // Unknown flags should not prevent a build of compiler invocation.
  ParseInputs Inputs;
  Inputs.TFS = &FS;
  Inputs.CompileCommand.CommandLine = {"clang", "-fsome-unknown-flag",
                                       testPath("foo.cpp")};
  IgnoreDiagnostics IgnoreDiags;
  EXPECT_NE(buildCompilerInvocation(Inputs, IgnoreDiags), nullptr);

  // Unknown forwarded to -cc1 should not a failure either.
  Inputs.CompileCommand.CommandLine = {
      "clang", "-Xclang", "-fsome-unknown-flag", testPath("foo.cpp")};
  EXPECT_NE(buildCompilerInvocation(Inputs, IgnoreDiags), nullptr);
}

TEST(ParsedASTTest, CollectsMainFileMacroExpansions) {
  Annotations TestCase(R"cpp(
    #define ^MACRO_ARGS(X, Y) X Y
    // - preamble ends
    ^ID(int A);
    // Macro arguments included.
    ^MACRO_ARGS(^MACRO_ARGS(^MACRO_EXP(int), E), ^ID(= 2));

    // Macro names inside other macros not included.
    #define ^MACRO_ARGS2(X, Y) X Y
    #define ^FOO BAR
    #define ^BAR 1
    int F = ^FOO;

    // Macros from token concatenations not included.
    #define ^CONCAT(X) X##A()
    #define ^PREPEND(X) MACRO##X()
    #define ^MACROA() 123
    int G = ^CONCAT(MACRO);
    int H = ^PREPEND(A);

    // Macros included not from preamble not included.
    #include "foo.inc"

    int printf(const char*, ...);
    void exit(int);
    #define ^assert(COND) if (!(COND)) { printf("%s", #COND); exit(0); }

    void test() {
      // Includes macro expansions in arguments that are expressions
      ^assert(0 <= ^BAR);
    }

    #ifdef ^UNDEFINED
    #endif

    #define ^MULTIPLE_DEFINITION 1
    #undef ^MULTIPLE_DEFINITION

    #define ^MULTIPLE_DEFINITION 2
    #undef ^MULTIPLE_DEFINITION
  )cpp");
  auto TU = TestTU::withCode(TestCase.code());
  TU.HeaderCode = R"cpp(
    #define ID(X) X
    #define MACRO_EXP(X) ID(X)
    MACRO_EXP(int B);
  )cpp";
  TU.AdditionalFiles["foo.inc"] = R"cpp(
    int C = ID(1);
    #define DEF 1
    int D = DEF;
  )cpp";
  ParsedAST AST = TU.build();
  std::vector<Position> MacroExpansionPositions;
  for (const auto &SIDToRefs : AST.getMacros().MacroRefs) {
    for (const auto &R : SIDToRefs.second)
      MacroExpansionPositions.push_back(R.Rng.start);
  }
  for (const auto &R : AST.getMacros().UnknownMacros)
    MacroExpansionPositions.push_back(R.Rng.start);
  EXPECT_THAT(MacroExpansionPositions,
              testing::UnorderedElementsAreArray(TestCase.points()));
}

MATCHER_P(WithFileName, Inc, "") { return arg.FileName == Inc; }

TEST(ParsedASTTest, ReplayPreambleForTidyCheckers) {
  struct Inclusion {
    Inclusion(const SourceManager &SM, SourceLocation HashLoc,
              const Token &IncludeTok, llvm::StringRef FileName, bool IsAngled,
              CharSourceRange FilenameRange)
        : HashOffset(SM.getDecomposedLoc(HashLoc).second), IncTok(IncludeTok),
          IncDirective(IncludeTok.getIdentifierInfo()->getName()),
          FileNameOffset(SM.getDecomposedLoc(FilenameRange.getBegin()).second),
          FileName(FileName), IsAngled(IsAngled) {}
    size_t HashOffset;
    syntax::Token IncTok;
    llvm::StringRef IncDirective;
    size_t FileNameOffset;
    llvm::StringRef FileName;
    bool IsAngled;
  };
  static std::vector<Inclusion> Includes;
  static std::vector<syntax::Token> SkippedFiles;
  struct ReplayPreamblePPCallback : public PPCallbacks {
    const SourceManager &SM;
    explicit ReplayPreamblePPCallback(const SourceManager &SM) : SM(SM) {}

    void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                            StringRef FileName, bool IsAngled,
                            CharSourceRange FilenameRange, const FileEntry *,
                            StringRef, StringRef, const clang::Module *,
                            SrcMgr::CharacteristicKind) override {
      Includes.emplace_back(SM, HashLoc, IncludeTok, FileName, IsAngled,
                            FilenameRange);
    }

    void FileSkipped(const FileEntryRef &, const Token &FilenameTok,
                     SrcMgr::CharacteristicKind) override {
      SkippedFiles.emplace_back(FilenameTok);
    }
  };
  struct ReplayPreambleCheck : public tidy::ClangTidyCheck {
    ReplayPreambleCheck(StringRef Name, tidy::ClangTidyContext *Context)
        : ClangTidyCheck(Name, Context) {}
    void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                             Preprocessor *ModuleExpanderPP) override {
      PP->addPPCallbacks(::std::make_unique<ReplayPreamblePPCallback>(SM));
    }
  };
  struct ReplayPreambleModule : public tidy::ClangTidyModule {
    void
    addCheckFactories(tidy::ClangTidyCheckFactories &CheckFactories) override {
      CheckFactories.registerCheck<ReplayPreambleCheck>(
          "replay-preamble-check");
    }
  };

  static tidy::ClangTidyModuleRegistry::Add<ReplayPreambleModule> X(
      "replay-preamble-module", "");
  TestTU TU;
  // This check records inclusion directives replayed by clangd.
  TU.ClangTidyProvider = addTidyChecks("replay-preamble-check");
  llvm::Annotations Test(R"cpp(
    $hash^#$include[[import]] $filebegin^"$filerange[[bar.h]]"
    $hash^#$include[[include_next]] $filebegin^"$filerange[[baz.h]]"
    $hash^#$include[[include]] $filebegin^<$filerange[[a.h]]>)cpp");
  llvm::StringRef Code = Test.code();
  TU.Code = Code.str();
  TU.AdditionalFiles["bar.h"] = "";
  TU.AdditionalFiles["baz.h"] = "";
  TU.AdditionalFiles["a.h"] = "";
  // Since we are also testing #import directives, and they don't make much
  // sense in c++ (also they actually break on windows), just set language to
  // obj-c.
  TU.ExtraArgs = {"-isystem.", "-xobjective-c"};

  const auto &AST = TU.build();
  const auto &SM = AST.getSourceManager();

  auto HashLocs = Test.points("hash");
  ASSERT_EQ(HashLocs.size(), Includes.size());
  auto IncludeRanges = Test.ranges("include");
  ASSERT_EQ(IncludeRanges.size(), Includes.size());
  auto FileBeginLocs = Test.points("filebegin");
  ASSERT_EQ(FileBeginLocs.size(), Includes.size());
  auto FileRanges = Test.ranges("filerange");
  ASSERT_EQ(FileRanges.size(), Includes.size());

  ASSERT_EQ(SkippedFiles.size(), Includes.size());
  for (size_t I = 0; I < Includes.size(); ++I) {
    const auto &Inc = Includes[I];

    EXPECT_EQ(Inc.HashOffset, HashLocs[I]);

    auto IncRange = IncludeRanges[I];
    EXPECT_THAT(Inc.IncTok.range(SM), RangeIs(IncRange));
    EXPECT_EQ(Inc.IncTok.kind(), tok::identifier);
    EXPECT_EQ(Inc.IncDirective,
              Code.substr(IncRange.Begin, IncRange.End - IncRange.Begin));

    EXPECT_EQ(Inc.FileNameOffset, FileBeginLocs[I]);
    EXPECT_EQ(Inc.IsAngled, Code[FileBeginLocs[I]] == '<');

    auto FileRange = FileRanges[I];
    EXPECT_EQ(Inc.FileName,
              Code.substr(FileRange.Begin, FileRange.End - FileRange.Begin));

    EXPECT_EQ(SM.getDecomposedLoc(SkippedFiles[I].location()).second,
              Inc.FileNameOffset);
    // This also contains quotes/angles so increment the range by one from both
    // sides.
    EXPECT_EQ(
        SkippedFiles[I].text(SM),
        Code.substr(FileRange.Begin - 1, FileRange.End - FileRange.Begin + 2));
    EXPECT_EQ(SkippedFiles[I].kind(), tok::header_name);
  }

  TU.AdditionalFiles["a.h"] = "";
  TU.AdditionalFiles["b.h"] = "";
  TU.AdditionalFiles["c.h"] = "";
  // Make sure replay logic works with patched preambles.
  llvm::StringLiteral Baseline = R"cpp(
    #include "a.h"
    #include "c.h")cpp";
  MockFS FS;
  TU.Code = Baseline.str();
  auto Inputs = TU.inputs(FS);
  auto BaselinePreamble = TU.preamble();
  ASSERT_TRUE(BaselinePreamble);

  // First make sure we don't crash on various modifications to the preamble.
  llvm::StringLiteral Cases[] = {
      // clang-format off
      // New include in middle.
      R"cpp(
        #include "a.h"
        #include "b.h"
        #include "c.h")cpp",
      // New include at top.
      R"cpp(
        #include "b.h"
        #include "a.h"
        #include "c.h")cpp",
      // New include at bottom.
      R"cpp(
        #include "a.h"
        #include "c.h"
        #include "b.h")cpp",
      // Same size with a missing include.
      R"cpp(
        #include "a.h"
        #include "b.h")cpp",
      // Smaller with no new includes.
      R"cpp(
        #include "a.h")cpp",
      // Smaller with a new includes.
      R"cpp(
        #include "b.h")cpp",
      // clang-format on
  };
  for (llvm::StringLiteral Case : Cases) {
    TU.Code = Case.str();

    IgnoreDiagnostics Diags;
    auto CI = buildCompilerInvocation(TU.inputs(FS), Diags);
    auto PatchedAST = ParsedAST::build(testPath(TU.Filename), TU.inputs(FS),
                                       std::move(CI), {}, BaselinePreamble);
    ASSERT_TRUE(PatchedAST);
    EXPECT_FALSE(PatchedAST->getDiagnostics());
  }

  // Then ensure correctness by making sure includes were seen only once.
  // Note that we first see the includes from the patch, as preamble includes
  // are replayed after exiting the built-in file.
  Includes.clear();
  TU.Code = R"cpp(
    #include "a.h"
    #include "b.h")cpp";
  IgnoreDiagnostics Diags;
  auto CI = buildCompilerInvocation(TU.inputs(FS), Diags);
  auto PatchedAST = ParsedAST::build(testPath(TU.Filename), TU.inputs(FS),
                                     std::move(CI), {}, BaselinePreamble);
  ASSERT_TRUE(PatchedAST);
  EXPECT_FALSE(PatchedAST->getDiagnostics());
  EXPECT_THAT(Includes,
              ElementsAre(WithFileName(testPath("__preamble_patch__.h")),
                          WithFileName("b.h"), WithFileName("a.h")));
}

TEST(ParsedASTTest, PatchesAdditionalIncludes) {
  llvm::StringLiteral ModifiedContents = R"cpp(
    #include "baz.h"
    #include "foo.h"
    #include "sub/aux.h"
    void bar() {
      foo();
      baz();
      aux();
    })cpp";
  // Build expected ast with symbols coming from headers.
  TestTU TU;
  TU.Filename = "foo.cpp";
  TU.AdditionalFiles["foo.h"] = "void foo();";
  TU.AdditionalFiles["sub/baz.h"] = "void baz();";
  TU.AdditionalFiles["sub/aux.h"] = "void aux();";
  TU.ExtraArgs = {"-I" + testPath("sub")};
  TU.Code = ModifiedContents.str();
  auto ExpectedAST = TU.build();

  // Build preamble with no includes.
  TU.Code = "";
  StoreDiags Diags;
  MockFS FS;
  auto Inputs = TU.inputs(FS);
  auto CI = buildCompilerInvocation(Inputs, Diags);
  auto EmptyPreamble =
      buildPreamble(testPath("foo.cpp"), *CI, Inputs, true, nullptr);
  ASSERT_TRUE(EmptyPreamble);
  EXPECT_THAT(EmptyPreamble->Includes.MainFileIncludes, testing::IsEmpty());

  // Now build an AST using empty preamble and ensure patched includes worked.
  TU.Code = ModifiedContents.str();
  Inputs = TU.inputs(FS);
  auto PatchedAST = ParsedAST::build(testPath("foo.cpp"), Inputs, std::move(CI),
                                     {}, EmptyPreamble);
  ASSERT_TRUE(PatchedAST);
  ASSERT_FALSE(PatchedAST->getDiagnostics());

  // Ensure source location information is correct, including resolved paths.
  EXPECT_THAT(PatchedAST->getIncludeStructure().MainFileIncludes,
              testing::Pointwise(
                  EqInc(), ExpectedAST.getIncludeStructure().MainFileIncludes));
  auto StringMapToVector = [](const llvm::StringMap<unsigned> SM) {
    std::vector<std::pair<std::string, unsigned>> Res;
    for (const auto &E : SM)
      Res.push_back({E.first().str(), E.second});
    llvm::sort(Res);
    return Res;
  };
  // Ensure file proximity signals are correct.
  EXPECT_EQ(StringMapToVector(PatchedAST->getIncludeStructure().includeDepth(
                testPath("foo.cpp"))),
            StringMapToVector(ExpectedAST.getIncludeStructure().includeDepth(
                testPath("foo.cpp"))));
}

TEST(ParsedASTTest, PatchesDeletedIncludes) {
  TestTU TU;
  TU.Filename = "foo.cpp";
  TU.Code = "";
  auto ExpectedAST = TU.build();

  // Build preamble with no includes.
  TU.Code = R"cpp(#include <foo.h>)cpp";
  StoreDiags Diags;
  MockFS FS;
  auto Inputs = TU.inputs(FS);
  auto CI = buildCompilerInvocation(Inputs, Diags);
  auto BaselinePreamble =
      buildPreamble(testPath("foo.cpp"), *CI, Inputs, true, nullptr);
  ASSERT_TRUE(BaselinePreamble);
  EXPECT_THAT(BaselinePreamble->Includes.MainFileIncludes,
              ElementsAre(testing::Field(&Inclusion::Written, "<foo.h>")));

  // Now build an AST using additional includes and check that locations are
  // correctly parsed.
  TU.Code = "";
  Inputs = TU.inputs(FS);
  auto PatchedAST = ParsedAST::build(testPath("foo.cpp"), Inputs, std::move(CI),
                                     {}, BaselinePreamble);
  ASSERT_TRUE(PatchedAST);

  // Ensure source location information is correct.
  EXPECT_THAT(PatchedAST->getIncludeStructure().MainFileIncludes,
              testing::Pointwise(
                  EqInc(), ExpectedAST.getIncludeStructure().MainFileIncludes));
  auto StringMapToVector = [](const llvm::StringMap<unsigned> SM) {
    std::vector<std::pair<std::string, unsigned>> Res;
    for (const auto &E : SM)
      Res.push_back({E.first().str(), E.second});
    llvm::sort(Res);
    return Res;
  };
  // Ensure file proximity signals are correct.
  EXPECT_EQ(StringMapToVector(PatchedAST->getIncludeStructure().includeDepth(
                testPath("foo.cpp"))),
            StringMapToVector(ExpectedAST.getIncludeStructure().includeDepth(
                testPath("foo.cpp"))));
}

// Returns Code guarded by #ifndef guards
std::string guard(llvm::StringRef Code) {
  static int GuardID = 0;
  std::string GuardName = ("GUARD_" + llvm::Twine(++GuardID)).str();
  return llvm::formatv("#ifndef {0}\n#define {0}\n{1}\n#endif\n", GuardName,
                       Code);
}

std::string once(llvm::StringRef Code) {
  return llvm::formatv("#pragma once\n{0}\n", Code);
}

bool mainIsGuarded(const ParsedAST &AST) {
  const auto &SM = AST.getSourceManager();
  const FileEntry *MainFE = SM.getFileEntryForID(SM.getMainFileID());
  return AST.getPreprocessor()
      .getHeaderSearchInfo()
      .isFileMultipleIncludeGuarded(MainFE);
}

MATCHER_P(Diag, Desc, "") {
  return llvm::StringRef(arg.Message).contains(Desc);
}

// Check our understanding of whether the main file is header guarded or not.
TEST(ParsedASTTest, HeaderGuards) {
  TestTU TU;
  TU.ImplicitHeaderGuard = false;

  TU.Code = ";";
  EXPECT_FALSE(mainIsGuarded(TU.build()));

  TU.Code = guard(";");
  EXPECT_TRUE(mainIsGuarded(TU.build()));

  TU.Code = once(";");
  EXPECT_TRUE(mainIsGuarded(TU.build()));

  TU.Code = R"cpp(
    ;
    #pragma once
  )cpp";
  EXPECT_FALSE(mainIsGuarded(TU.build())); // FIXME: true

  TU.Code = R"cpp(
    ;
    #ifndef GUARD
    #define GUARD
    ;
    #endif
  )cpp";
  EXPECT_FALSE(mainIsGuarded(TU.build()));
}

// Check our handling of files that include themselves.
// Ideally we allow this if the file has header guards.
//
// Note: the semicolons (empty statements) are significant!
// - they force the preamble to end and the body to begin. Directives can have
//   different effects in the preamble vs main file (which we try to hide).
// - if the preamble would otherwise cover the whole file, a trailing semicolon
//   forces their sizes to be different. This is significant because the file
//   size is part of the lookup key for HeaderFileInfo, and we don't want to
//   rely on the preamble's HFI being looked up when parsing the main file.
TEST(ParsedASTTest, HeaderGuardsSelfInclude) {
  TestTU TU;
  TU.ImplicitHeaderGuard = false;
  TU.Filename = "self.h";

  TU.Code = R"cpp(
    #include "self.h" // error-ok
    ;
  )cpp";
  auto AST = TU.build();
  EXPECT_THAT(*AST.getDiagnostics(),
              ElementsAre(Diag("recursively when building a preamble")));
  EXPECT_FALSE(mainIsGuarded(AST));

  TU.Code = R"cpp(
    ;
    #include "self.h" // error-ok
  )cpp";
  AST = TU.build();
  EXPECT_THAT(*AST.getDiagnostics(), ElementsAre(Diag("nested too deeply")));
  EXPECT_FALSE(mainIsGuarded(AST));

  TU.Code = R"cpp(
    #pragma once
    #include "self.h"
    ;
  )cpp";
  AST = TU.build();
  EXPECT_THAT(*AST.getDiagnostics(), IsEmpty());
  EXPECT_TRUE(mainIsGuarded(AST));

  TU.Code = R"cpp(
    #pragma once
    ;
    #include "self.h"
  )cpp";
  AST = TU.build();
  EXPECT_THAT(*AST.getDiagnostics(), IsEmpty());
  EXPECT_TRUE(mainIsGuarded(AST));

  TU.Code = R"cpp(
    ;
    #pragma once
    #include "self.h"
  )cpp";
  AST = TU.build();
  EXPECT_THAT(*AST.getDiagnostics(), IsEmpty());
  EXPECT_TRUE(mainIsGuarded(AST));

  TU.Code = R"cpp(
    #ifndef GUARD
    #define GUARD
    #include "self.h" // error-ok: FIXME, this would be nice to support
    #endif
    ;
  )cpp";
  AST = TU.build();
  EXPECT_THAT(*AST.getDiagnostics(),
              ElementsAre(Diag("recursively when building a preamble")));
  EXPECT_TRUE(mainIsGuarded(AST));

  TU.Code = R"cpp(
    #ifndef GUARD
    #define GUARD
    ;
    #include "self.h"
    #endif
  )cpp";
  AST = TU.build();
  EXPECT_THAT(*AST.getDiagnostics(), IsEmpty());
  EXPECT_TRUE(mainIsGuarded(AST));

  // Guarded too late...
  TU.Code = R"cpp(
    #include "self.h" // error-ok
    #ifndef GUARD
    #define GUARD
    ;
    #endif
  )cpp";
  AST = TU.build();
  EXPECT_THAT(*AST.getDiagnostics(),
              ElementsAre(Diag("recursively when building a preamble")));
  EXPECT_FALSE(mainIsGuarded(AST));

  TU.Code = R"cpp(
    #include "self.h" // error-ok
    ;
    #ifndef GUARD
    #define GUARD
    #endif
  )cpp";
  AST = TU.build();
  EXPECT_THAT(*AST.getDiagnostics(),
              ElementsAre(Diag("recursively when building a preamble")));
  EXPECT_FALSE(mainIsGuarded(AST));

  TU.Code = R"cpp(
    ;
    #ifndef GUARD
    #define GUARD
    #include "self.h"
    #endif
  )cpp";
  AST = TU.build();
  EXPECT_THAT(*AST.getDiagnostics(), IsEmpty());
  EXPECT_FALSE(mainIsGuarded(AST));

  TU.Code = R"cpp(
    #include "self.h" // error-ok
    #pragma once
    ;
  )cpp";
  AST = TU.build();
  EXPECT_THAT(*AST.getDiagnostics(),
              ElementsAre(Diag("recursively when building a preamble")));
  EXPECT_TRUE(mainIsGuarded(AST));

  TU.Code = R"cpp(
    #include "self.h" // error-ok
    ;
    #pragma once
  )cpp";
  AST = TU.build();
  EXPECT_THAT(*AST.getDiagnostics(),
              ElementsAre(Diag("recursively when building a preamble")));
  EXPECT_TRUE(mainIsGuarded(AST));
}

// Tests how we handle common idioms for splitting a header-only library
// into interface and implementation files (e.g. *.h vs *.inl).
// These files mutually include each other, and need careful handling of include
// guards (which interact with preambles).
TEST(ParsedASTTest, HeaderGuardsImplIface) {
  std::string Interface = R"cpp(
    // error-ok: we assert on diagnostics explicitly
    template <class T> struct Traits {
      unsigned size();
    };
    #include "impl.h"
  )cpp";
  std::string Implementation = R"cpp(
    // error-ok: we assert on diagnostics explicitly
    #include "iface.h"
    template <class T> unsigned Traits<T>::size() {
      return sizeof(T);
    }
  )cpp";

  TestTU TU;
  TU.ImplicitHeaderGuard = false; // We're testing include guard handling!
  TU.ExtraArgs.push_back("-xc++-header");

  // Editing the interface file, which is include guarded (easy case).
  // We mostly get this right via PP if we don't recognize the include guard.
  TU.Filename = "iface.h";
  TU.Code = guard(Interface);
  TU.AdditionalFiles = {{"impl.h", Implementation}};
  auto AST = TU.build();
  EXPECT_THAT(*AST.getDiagnostics(), IsEmpty());
  EXPECT_TRUE(mainIsGuarded(AST));
  // Slightly harder: the `#pragma once` is part of the preamble, and we
  // need to transfer it to the main file's HeaderFileInfo.
  TU.Code = once(Interface);
  AST = TU.build();
  EXPECT_THAT(*AST.getDiagnostics(), IsEmpty());
  EXPECT_TRUE(mainIsGuarded(AST));

  // Editing the implementation file, which is not include guarded.
  TU.Filename = "impl.h";
  TU.Code = Implementation;
  TU.AdditionalFiles = {{"iface.h", guard(Interface)}};
  AST = TU.build();
  // The diagnostic is unfortunate in this case, but correct per our model.
  // Ultimately the include is skipped and the code is parsed correctly though.
  EXPECT_THAT(*AST.getDiagnostics(),
              ElementsAre(Diag("in included file: main file cannot be included "
                               "recursively when building a preamble")));
  EXPECT_FALSE(mainIsGuarded(AST));
  // Interface is pragma once guarded, same thing.
  TU.AdditionalFiles = {{"iface.h", once(Interface)}};
  AST = TU.build();
  EXPECT_THAT(*AST.getDiagnostics(),
              ElementsAre(Diag("in included file: main file cannot be included "
                               "recursively when building a preamble")));
  EXPECT_FALSE(mainIsGuarded(AST));
}

} // namespace
} // namespace clangd
} // namespace clang
