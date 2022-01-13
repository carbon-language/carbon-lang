//===--- DiagnosticsTests.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "Config.h"
#include "Diagnostics.h"
#include "Feature.h"
#include "FeatureModule.h"
#include "ParsedAST.h"
#include "Protocol.h"
#include "SourceCode.h"
#include "TestFS.h"
#include "TestIndex.h"
#include "TestTU.h"
#include "TidyProvider.h"
#include "index/MemIndex.h"
#include "support/Context.h"
#include "support/Path.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticSema.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/TargetSelect.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <memory>

namespace clang {
namespace clangd {
namespace {

using ::testing::_;
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;
using testing::UnorderedElementsAreArray;

::testing::Matcher<const Diag &> WithFix(::testing::Matcher<Fix> FixMatcher) {
  return Field(&Diag::Fixes, ElementsAre(FixMatcher));
}

::testing::Matcher<const Diag &> WithFix(::testing::Matcher<Fix> FixMatcher1,
                                         ::testing::Matcher<Fix> FixMatcher2) {
  return Field(&Diag::Fixes, UnorderedElementsAre(FixMatcher1, FixMatcher2));
}

::testing::Matcher<const Diag &>
WithNote(::testing::Matcher<Note> NoteMatcher) {
  return Field(&Diag::Notes, ElementsAre(NoteMatcher));
}

::testing::Matcher<const Diag &>
WithNote(::testing::Matcher<Note> NoteMatcher1,
         ::testing::Matcher<Note> NoteMatcher2) {
  return Field(&Diag::Notes, UnorderedElementsAre(NoteMatcher1, NoteMatcher2));
}

::testing::Matcher<const Diag &>
WithTag(::testing::Matcher<DiagnosticTag> TagMatcher) {
  return Field(&Diag::Tags, Contains(TagMatcher));
}

MATCHER_P2(Diag, Range, Message,
           "Diag at " + llvm::to_string(Range) + " = [" + Message + "]") {
  return arg.Range == Range && arg.Message == Message;
}

MATCHER_P3(Fix, Range, Replacement, Message,
           "Fix " + llvm::to_string(Range) + " => " +
               ::testing::PrintToString(Replacement) + " = [" + Message + "]") {
  return arg.Message == Message && arg.Edits.size() == 1 &&
         arg.Edits[0].range == Range && arg.Edits[0].newText == Replacement;
}

MATCHER_P(FixMessage, Message, "") { return arg.Message == Message; }

MATCHER_P(EqualToLSPDiag, LSPDiag,
          "LSP diagnostic " + llvm::to_string(LSPDiag)) {
  if (toJSON(arg) != toJSON(LSPDiag)) {
    *result_listener << llvm::formatv("expected:\n{0:2}\ngot\n{1:2}",
                                      toJSON(LSPDiag), toJSON(arg))
                            .str();
    return false;
  }
  return true;
}

MATCHER_P(DiagSource, S, "") { return arg.Source == S; }
MATCHER_P(DiagName, N, "") { return arg.Name == N; }
MATCHER_P(DiagSeverity, S, "") { return arg.Severity == S; }

MATCHER_P(EqualToFix, Fix, "LSP fix " + llvm::to_string(Fix)) {
  if (arg.Message != Fix.Message)
    return false;
  if (arg.Edits.size() != Fix.Edits.size())
    return false;
  for (std::size_t I = 0; I < arg.Edits.size(); ++I) {
    if (arg.Edits[I].range != Fix.Edits[I].range ||
        arg.Edits[I].newText != Fix.Edits[I].newText)
      return false;
  }
  return true;
}

// Helper function to make tests shorter.
Position pos(int line, int character) {
  Position Res;
  Res.line = line;
  Res.character = character;
  return Res;
}

// Normally returns the provided diagnostics matcher.
// If clang-tidy checks are not linked in, returns a matcher for no diagnostics!
// This is intended for tests where the diagnostics come from clang-tidy checks.
// We don't #ifdef each individual test as it's intrusive and we want to ensure
// that as much of the test is still compiled an run as possible.
::testing::Matcher<std::vector<clangd::Diag>>
ifTidyChecks(::testing::Matcher<std::vector<clangd::Diag>> M) {
  if (!CLANGD_TIDY_CHECKS)
    return IsEmpty();
  return M;
}

TEST(DiagnosticsTest, DiagnosticRanges) {
  // Check we report correct ranges, including various edge-cases.
  Annotations Test(R"cpp(
    // error-ok
    #define ID(X) X
    namespace test{};
    void $decl[[foo]]();
    int main() {
      struct Container { int* begin(); int* end(); } *container;
      for (auto i : $insertstar[[]]$range[[container]]) {
      }

      $typo[[go\
o]]();
      foo()$semicolon[[]]//with comments
      $unk[[unknown]]();
      double $type[[bar]] = "foo";
      struct Foo { int x; }; Foo a;
      a.$nomember[[y]];
      test::$nomembernamespace[[test]];
      $macro[[ID($macroarg[[fod]])]]();
    }
  )cpp");
  auto TU = TestTU::withCode(Test.code());
  EXPECT_THAT(
      *TU.build().getDiagnostics(),
      ElementsAre(
          // Make sure the whole token is highlighted.
          AllOf(Diag(Test.range("range"),
                     "invalid range expression of type 'struct Container *'; "
                     "did you mean to dereference it with '*'?"),
                WithFix(Fix(Test.range("insertstar"), "*", "insert '*'"))),
          // This range spans lines.
          AllOf(Diag(Test.range("typo"),
                     "use of undeclared identifier 'goo'; did you mean 'foo'?"),
                DiagSource(Diag::Clang), DiagName("undeclared_var_use_suggest"),
                WithFix(
                    Fix(Test.range("typo"), "foo", "change 'go\\…' to 'foo'")),
                // This is a pretty normal range.
                WithNote(Diag(Test.range("decl"), "'foo' declared here"))),
          // This range is zero-width and insertion. Therefore make sure we are
          // not expanding it into other tokens. Since we are not going to
          // replace those.
          AllOf(Diag(Test.range("semicolon"), "expected ';' after expression"),
                WithFix(Fix(Test.range("semicolon"), ";", "insert ';'"))),
          // This range isn't provided by clang, we expand to the token.
          Diag(Test.range("unk"), "use of undeclared identifier 'unknown'"),
          Diag(Test.range("type"),
               "cannot initialize a variable of type 'double' with an lvalue "
               "of type 'const char [4]'"),
          Diag(Test.range("nomember"), "no member named 'y' in 'Foo'"),
          Diag(Test.range("nomembernamespace"),
               "no member named 'test' in namespace 'test'"),
          AllOf(Diag(Test.range("macro"),
                     "use of undeclared identifier 'fod'; did you mean 'foo'?"),
                WithFix(Fix(Test.range("macroarg"), "foo",
                            "change 'fod' to 'foo'")))));
}

// Verify that the -Wswitch case-not-covered diagnostic range covers the
// whole expression. This is important because the "populate-switch" tweak
// fires for the full expression range (see tweaks/PopulateSwitchTests.cpp).
// The quickfix flow only works end-to-end if the tweak can be triggered on
// the diagnostic's range.
TEST(DiagnosticsTest, WSwitch) {
  Annotations Test(R"cpp(
    enum A { X };
    struct B { A a; };
    void foo(B b) {
      switch ([[b.a]]) {}
    }
  )cpp");
  auto TU = TestTU::withCode(Test.code());
  TU.ExtraArgs = {"-Wswitch"};
  EXPECT_THAT(*TU.build().getDiagnostics(),
              ElementsAre(Diag(Test.range(),
                               "enumeration value 'X' not handled in switch")));
}

TEST(DiagnosticsTest, FlagsMatter) {
  Annotations Test("[[void]] main() {} // error-ok");
  auto TU = TestTU::withCode(Test.code());
  EXPECT_THAT(*TU.build().getDiagnostics(),
              ElementsAre(AllOf(Diag(Test.range(), "'main' must return 'int'"),
                                WithFix(Fix(Test.range(), "int",
                                            "change 'void' to 'int'")))));
  // Same code built as C gets different diagnostics.
  TU.Filename = "Plain.c";
  EXPECT_THAT(
      *TU.build().getDiagnostics(),
      ElementsAre(AllOf(
          Diag(Test.range(), "return type of 'main' is not 'int'"),
          WithFix(Fix(Test.range(), "int", "change return type to 'int'")))));
}

TEST(DiagnosticsTest, DiagnosticPreamble) {
  Annotations Test(R"cpp(
    #include $[["not-found.h"]] // error-ok
  )cpp");

  auto TU = TestTU::withCode(Test.code());
  EXPECT_THAT(*TU.build().getDiagnostics(),
              ElementsAre(::testing::AllOf(
                  Diag(Test.range(), "'not-found.h' file not found"),
                  DiagSource(Diag::Clang), DiagName("pp_file_not_found"))));
}

TEST(DiagnosticsTest, DeduplicatedClangTidyDiagnostics) {
  Annotations Test(R"cpp(
    float foo = [[0.1f]];
  )cpp");
  auto TU = TestTU::withCode(Test.code());
  // Enable alias clang-tidy checks, these check emits the same diagnostics
  // (except the check name).
  TU.ClangTidyProvider = addTidyChecks("readability-uppercase-literal-suffix,"
                                       "hicpp-uppercase-literal-suffix");
  // Verify that we filter out the duplicated diagnostic message.
  EXPECT_THAT(
      *TU.build().getDiagnostics(),
      ifTidyChecks(UnorderedElementsAre(::testing::AllOf(
          Diag(Test.range(),
               "floating point literal has suffix 'f', which is not uppercase"),
          DiagSource(Diag::ClangTidy)))));

  Test = Annotations(R"cpp(
    template<typename T>
    void func(T) {
      float f = [[0.3f]];
    }
    void k() {
      func(123);
      func(2.0);
    }
  )cpp");
  TU.Code = std::string(Test.code());
  // The check doesn't handle template instantiations which ends up emitting
  // duplicated messages, verify that we deduplicate them.
  EXPECT_THAT(
      *TU.build().getDiagnostics(),
      ifTidyChecks(UnorderedElementsAre(::testing::AllOf(
          Diag(Test.range(),
               "floating point literal has suffix 'f', which is not uppercase"),
          DiagSource(Diag::ClangTidy)))));
}

TEST(DiagnosticsTest, ClangTidy) {
  Annotations Test(R"cpp(
    #include $deprecated[["assert.h"]]

    #define $macrodef[[SQUARE]](X) (X)*(X)
    int $main[[main]]() {
      int y = 4;
      return SQUARE($macroarg[[++]]y);
      return $doubled[[sizeof]](sizeof(int));
    }

    // misc-no-recursion uses a custom traversal from the TUDecl
    void foo();
    void $bar[[bar]]() {
      foo();
    }
    void $foo[[foo]]() {
      bar();
    }
  )cpp");
  auto TU = TestTU::withCode(Test.code());
  TU.HeaderFilename = "assert.h"; // Suppress "not found" error.
  TU.ClangTidyProvider = addTidyChecks("bugprone-sizeof-expression,"
                                       "bugprone-macro-repeated-side-effects,"
                                       "modernize-deprecated-headers,"
                                       "modernize-use-trailing-return-type,"
                                       "misc-no-recursion");
  TU.ExtraArgs.push_back("-Wno-unsequenced");
  EXPECT_THAT(
      *TU.build().getDiagnostics(),
      ifTidyChecks(UnorderedElementsAre(
          AllOf(Diag(Test.range("deprecated"),
                     "inclusion of deprecated C++ header 'assert.h'; consider "
                     "using 'cassert' instead"),
                DiagSource(Diag::ClangTidy),
                DiagName("modernize-deprecated-headers"),
                WithFix(Fix(Test.range("deprecated"), "<cassert>",
                            "change '\"assert.h\"' to '<cassert>'"))),
          Diag(Test.range("doubled"),
               "suspicious usage of 'sizeof(sizeof(...))'"),
          AllOf(Diag(Test.range("macroarg"),
                     "side effects in the 1st macro argument 'X' are "
                     "repeated in "
                     "macro expansion"),
                DiagSource(Diag::ClangTidy),
                DiagName("bugprone-macro-repeated-side-effects"),
                WithNote(Diag(Test.range("macrodef"),
                              "macro 'SQUARE' defined here"))),
          AllOf(Diag(Test.range("main"),
                     "use a trailing return type for this function"),
                DiagSource(Diag::ClangTidy),
                DiagName("modernize-use-trailing-return-type"),
                // Verify there's no "[check-name]" suffix in the message.
                WithFix(FixMessage(
                    "use a trailing return type for this function"))),
          Diag(Test.range("foo"),
               "function 'foo' is within a recursive call chain"),
          Diag(Test.range("bar"),
               "function 'bar' is within a recursive call chain"))));
}

TEST(DiagnosticsTest, ClangTidyEOF) {
  // clang-format off
  Annotations Test(R"cpp(
  [[#]]include <b.h>
  #include "a.h")cpp");
  // clang-format on
  auto TU = TestTU::withCode(Test.code());
  TU.ExtraArgs = {"-isystem."};
  TU.AdditionalFiles["a.h"] = TU.AdditionalFiles["b.h"] = "";
  TU.ClangTidyProvider = addTidyChecks("llvm-include-order");
  EXPECT_THAT(
      *TU.build().getDiagnostics(),
      ifTidyChecks(Contains(
          AllOf(Diag(Test.range(), "#includes are not sorted properly"),
                DiagSource(Diag::ClangTidy), DiagName("llvm-include-order")))));
}

TEST(DiagnosticTest, TemplatesInHeaders) {
  // Diagnostics from templates defined in headers are placed at the expansion.
  Annotations Main(R"cpp(
    Derived<int> [[y]]; // error-ok
  )cpp");
  Annotations Header(R"cpp(
    template <typename T>
    struct Derived : [[T]] {};
  )cpp");
  TestTU TU = TestTU::withCode(Main.code());
  TU.HeaderCode = Header.code().str();
  EXPECT_THAT(
      *TU.build().getDiagnostics(),
      ElementsAre(AllOf(
          Diag(Main.range(), "in template: base specifier must name a class"),
          WithNote(Diag(Header.range(), "error occurred here"),
                   Diag(Main.range(), "in instantiation of template class "
                                      "'Derived<int>' requested here")))));
}

TEST(DiagnosticTest, MakeUnique) {
  // We usually miss diagnostics from header functions as we don't parse them.
  // std::make_unique is an exception.
  Annotations Main(R"cpp(
    struct S { S(char*); };
    auto x = std::[[make_unique]]<S>(42); // error-ok
  )cpp");
  TestTU TU = TestTU::withCode(Main.code());
  TU.HeaderCode = R"cpp(
    namespace std {
    // These mocks aren't quite right - we omit unique_ptr for simplicity.
    // forward is included to show its body is not needed to get the diagnostic.
    template <typename T> T&& forward(T& t) { return static_cast<T&&>(t); }
    template <typename T, typename... A> T* make_unique(A&&... args) {
       return new T(std::forward<A>(args)...);
    }
    }
  )cpp";
  EXPECT_THAT(*TU.build().getDiagnostics(),
              UnorderedElementsAre(
                  Diag(Main.range(),
                       "in template: "
                       "no matching constructor for initialization of 'S'")));
}

TEST(DiagnosticTest, NoMultipleDiagnosticInFlight) {
  Annotations Main(R"cpp(
    template <typename T> struct Foo {
      T *begin();
      T *end();
    };
    struct LabelInfo {
      int a;
      bool b;
    };

    void f() {
      Foo<LabelInfo> label_info_map;
      [[for]] (auto it = label_info_map.begin(); it != label_info_map.end(); ++it) {
        auto S = *it;
      }
    }
  )cpp");
  TestTU TU = TestTU::withCode(Main.code());
  TU.ClangTidyProvider = addTidyChecks("modernize-loop-convert");
  EXPECT_THAT(
      *TU.build().getDiagnostics(),
      ifTidyChecks(UnorderedElementsAre(::testing::AllOf(
          Diag(Main.range(), "use range-based for loop instead"),
          DiagSource(Diag::ClangTidy), DiagName("modernize-loop-convert")))));
}

TEST(DiagnosticTest, RespectsDiagnosticConfig) {
  Annotations Main(R"cpp(
    // error-ok
    void x() {
      [[unknown]]();
      $ret[[return]] 42;
    }
  )cpp");
  auto TU = TestTU::withCode(Main.code());
  EXPECT_THAT(
      *TU.build().getDiagnostics(),
      ElementsAre(Diag(Main.range(), "use of undeclared identifier 'unknown'"),
                  Diag(Main.range("ret"),
                       "void function 'x' should not return a value")));
  Config Cfg;
  Cfg.Diagnostics.Suppress.insert("return-type");
  WithContextValue WithCfg(Config::Key, std::move(Cfg));
  EXPECT_THAT(*TU.build().getDiagnostics(),
              ElementsAre(Diag(Main.range(),
                               "use of undeclared identifier 'unknown'")));
}

TEST(DiagnosticTest, ClangTidySuppressionComment) {
  Annotations Main(R"cpp(
    int main() {
      int i = 3;
      double d = 8 / i;  // NOLINT
      // NOLINTNEXTLINE
      double e = 8 / i;
      #define BAD 8 / i
      double f = BAD;  // NOLINT
      double g = [[8]] / i;
      #define BAD2 BAD
      double h = BAD2;  // NOLINT
    }
  )cpp");
  TestTU TU = TestTU::withCode(Main.code());
  TU.ClangTidyProvider = addTidyChecks("bugprone-integer-division");
  EXPECT_THAT(
      *TU.build().getDiagnostics(),
      ifTidyChecks(UnorderedElementsAre(::testing::AllOf(
          Diag(Main.range(), "result of integer division used in a floating "
                             "point context; possible loss of precision"),
          DiagSource(Diag::ClangTidy),
          DiagName("bugprone-integer-division")))));
}

TEST(DiagnosticTest, ClangTidyWarningAsError) {
  Annotations Main(R"cpp(
    int main() {
      int i = 3;
      double f = [[8]] / i; // error-ok
    }
  )cpp");
  TestTU TU = TestTU::withCode(Main.code());
  TU.ClangTidyProvider =
      addTidyChecks("bugprone-integer-division", "bugprone-integer-division");
  EXPECT_THAT(
      *TU.build().getDiagnostics(),
      ifTidyChecks(UnorderedElementsAre(::testing::AllOf(
          Diag(Main.range(), "result of integer division used in a floating "
                             "point context; possible loss of precision"),
          DiagSource(Diag::ClangTidy), DiagName("bugprone-integer-division"),
          DiagSeverity(DiagnosticsEngine::Error)))));
}

TEST(DiagnosticTest, LongFixMessages) {
  // We limit the size of printed code.
  Annotations Source(R"cpp(
    int main() {
      // error-ok
      int somereallyreallyreallyreallyreallyreallyreallyreallylongidentifier;
      [[omereallyreallyreallyreallyreallyreallyreallyreallylongidentifier]]= 10;
    }
  )cpp");
  TestTU TU = TestTU::withCode(Source.code());
  EXPECT_THAT(
      *TU.build().getDiagnostics(),
      ElementsAre(WithFix(Fix(
          Source.range(),
          "somereallyreallyreallyreallyreallyreallyreallyreallylongidentifier",
          "change 'omereallyreallyreallyreallyreallyreallyreallyreall…' to "
          "'somereallyreallyreallyreallyreallyreallyreallyreal…'"))));
  // Only show changes up to a first newline.
  Source = Annotations(R"cpp(
    // error-ok
    int main() {
      int ident;
      [[ide\
n]] = 10; // error-ok
    }
  )cpp");
  TU.Code = std::string(Source.code());
  EXPECT_THAT(*TU.build().getDiagnostics(),
              ElementsAre(WithFix(
                  Fix(Source.range(), "ident", "change 'ide\\…' to 'ident'"))));
}

TEST(DiagnosticTest, ClangTidySuppressionCommentTrumpsWarningAsError) {
  Annotations Main(R"cpp(
    int main() {
      int i = 3;
      double f = [[8]] / i;  // NOLINT
    }
  )cpp");
  TestTU TU = TestTU::withCode(Main.code());
  TU.ClangTidyProvider =
      addTidyChecks("bugprone-integer-division", "bugprone-integer-division");
  EXPECT_THAT(*TU.build().getDiagnostics(), UnorderedElementsAre());
}

TEST(DiagnosticTest, ClangTidyNoLiteralDataInMacroToken) {
  Annotations Main(R"cpp(
    #define SIGTERM 15
    using pthread_t = int;
    int pthread_kill(pthread_t thread, int sig);
    int func() {
      pthread_t thread;
      return pthread_kill(thread, 0);
    }
  )cpp");
  TestTU TU = TestTU::withCode(Main.code());
  TU.ClangTidyProvider = addTidyChecks("bugprone-bad-signal-to-kill-thread");
  EXPECT_THAT(*TU.build().getDiagnostics(), UnorderedElementsAre()); // no-crash
}

TEST(DiagnosticTest, ElseAfterReturnRange) {
  Annotations Main(R"cpp(
    int foo(int cond) {
    if (cond == 1) {
      return 42;
    } [[else]] if (cond == 2) {
      return 43;
    }
    return 44;
    }
  )cpp");
  TestTU TU = TestTU::withCode(Main.code());
  TU.ClangTidyProvider = addTidyChecks("llvm-else-after-return");
  EXPECT_THAT(*TU.build().getDiagnostics(),
              ifTidyChecks(ElementsAre(
                  Diag(Main.range(), "do not use 'else' after 'return'"))));
}

TEST(DiagnosticsTest, Preprocessor) {
  // This looks like a preamble, but there's an #else in the middle!
  // Check that:
  //  - the #else doesn't generate diagnostics (we had this bug)
  //  - we get diagnostics from the taken branch
  //  - we get no diagnostics from the not taken branch
  Annotations Test(R"cpp(
    #ifndef FOO
    #define FOO
      int a = [[b]]; // error-ok
    #else
      int x = y;
    #endif
    )cpp");
  EXPECT_THAT(
      *TestTU::withCode(Test.code()).build().getDiagnostics(),
      ElementsAre(Diag(Test.range(), "use of undeclared identifier 'b'")));
}

TEST(DiagnosticsTest, IgnoreVerify) {
  auto TU = TestTU::withCode(R"cpp(
    int a; // expected-error {{}}
  )cpp");
  TU.ExtraArgs.push_back("-Xclang");
  TU.ExtraArgs.push_back("-verify");
  EXPECT_THAT(*TU.build().getDiagnostics(), IsEmpty());
}

// Recursive main-file include is diagnosed, and doesn't crash.
TEST(DiagnosticsTest, RecursivePreamble) {
  auto TU = TestTU::withCode(R"cpp(
    #include "foo.h" // error-ok
    int symbol;
  )cpp");
  TU.Filename = "foo.h";
  EXPECT_THAT(*TU.build().getDiagnostics(),
              ElementsAre(DiagName("pp_including_mainfile_in_preamble")));
  EXPECT_THAT(TU.build().getLocalTopLevelDecls(), SizeIs(1));
}

// Recursive main-file include with #pragma once guard is OK.
TEST(DiagnosticsTest, RecursivePreamblePragmaOnce) {
  auto TU = TestTU::withCode(R"cpp(
    #pragma once
    #include "foo.h"
    int symbol;
  )cpp");
  TU.Filename = "foo.h";
  EXPECT_THAT(*TU.build().getDiagnostics(), IsEmpty());
  EXPECT_THAT(TU.build().getLocalTopLevelDecls(), SizeIs(1));
}

// Recursive main-file include with #ifndef guard should be OK.
// However, it's not yet recognized (incomplete at end of preamble).
TEST(DiagnosticsTest, RecursivePreambleIfndefGuard) {
  auto TU = TestTU::withCode(R"cpp(
    #ifndef FOO
    #define FOO
    #include "foo.h" // error-ok
    int symbol;
    #endif
  )cpp");
  TU.Filename = "foo.h";
  // FIXME: should be no errors here.
  EXPECT_THAT(*TU.build().getDiagnostics(),
              ElementsAre(DiagName("pp_including_mainfile_in_preamble")));
  EXPECT_THAT(TU.build().getLocalTopLevelDecls(), SizeIs(1));
}

TEST(DiagnosticsTest, InsideMacros) {
  Annotations Test(R"cpp(
    #define TEN 10
    #define RET(x) return x + 10

    int* foo() {
      RET($foo[[0]]); // error-ok
    }
    int* bar() {
      return $bar[[TEN]];
    }
    )cpp");
  EXPECT_THAT(*TestTU::withCode(Test.code()).build().getDiagnostics(),
              ElementsAre(Diag(Test.range("foo"),
                               "cannot initialize return object of type "
                               "'int *' with an rvalue of type 'int'"),
                          Diag(Test.range("bar"),
                               "cannot initialize return object of type "
                               "'int *' with an rvalue of type 'int'")));
}

TEST(DiagnosticsTest, NoFixItInMacro) {
  Annotations Test(R"cpp(
    #define Define(name) void name() {}

    [[Define]](main) // error-ok
  )cpp");
  auto TU = TestTU::withCode(Test.code());
  EXPECT_THAT(*TU.build().getDiagnostics(),
              ElementsAre(AllOf(Diag(Test.range(), "'main' must return 'int'"),
                                Not(WithFix(_)))));
}

TEST(ClangdTest, MSAsm) {
  // Parsing MS assembly tries to use the target MCAsmInfo, which we don't link.
  // We used to crash here. Now clang emits a diagnostic, which we filter out.
  llvm::InitializeAllTargetInfos(); // As in ClangdMain
  auto TU = TestTU::withCode("void fn() { __asm { cmp cl,64 } }");
  TU.ExtraArgs = {"-fms-extensions"};
  EXPECT_THAT(*TU.build().getDiagnostics(), IsEmpty());
}

TEST(DiagnosticsTest, ToLSP) {
  URIForFile MainFile =
      URIForFile::canonicalize(testPath("foo/bar/main.cpp"), "");
  URIForFile HeaderFile =
      URIForFile::canonicalize(testPath("foo/bar/header.h"), "");

  clangd::Diag D;
  D.ID = clang::diag::err_undeclared_var_use;
  D.Tags = {DiagnosticTag::Unnecessary};
  D.Name = "undeclared_var_use";
  D.Source = clangd::Diag::Clang;
  D.Message = "something terrible happened";
  D.Range = {pos(1, 2), pos(3, 4)};
  D.InsideMainFile = true;
  D.Severity = DiagnosticsEngine::Error;
  D.File = "foo/bar/main.cpp";
  D.AbsFile = std::string(MainFile.file());

  clangd::Note NoteInMain;
  NoteInMain.Message = "declared somewhere in the main file";
  NoteInMain.Range = {pos(5, 6), pos(7, 8)};
  NoteInMain.Severity = DiagnosticsEngine::Remark;
  NoteInMain.File = "../foo/bar/main.cpp";
  NoteInMain.InsideMainFile = true;
  NoteInMain.AbsFile = std::string(MainFile.file());

  D.Notes.push_back(NoteInMain);

  clangd::Note NoteInHeader;
  NoteInHeader.Message = "declared somewhere in the header file";
  NoteInHeader.Range = {pos(9, 10), pos(11, 12)};
  NoteInHeader.Severity = DiagnosticsEngine::Note;
  NoteInHeader.File = "../foo/baz/header.h";
  NoteInHeader.InsideMainFile = false;
  NoteInHeader.AbsFile = std::string(HeaderFile.file());
  D.Notes.push_back(NoteInHeader);

  clangd::Fix F;
  F.Message = "do something";
  D.Fixes.push_back(F);

  // Diagnostics should turn into these:
  clangd::Diagnostic MainLSP;
  MainLSP.range = D.Range;
  MainLSP.severity = getSeverity(DiagnosticsEngine::Error);
  MainLSP.code = "undeclared_var_use";
  MainLSP.source = "clang";
  MainLSP.message =
      R"(Something terrible happened (fix available)

main.cpp:6:7: remark: declared somewhere in the main file

../foo/baz/header.h:10:11:
note: declared somewhere in the header file)";
  MainLSP.tags = {DiagnosticTag::Unnecessary};

  clangd::Diagnostic NoteInMainLSP;
  NoteInMainLSP.range = NoteInMain.Range;
  NoteInMainLSP.severity = getSeverity(DiagnosticsEngine::Remark);
  NoteInMainLSP.message = R"(Declared somewhere in the main file

main.cpp:2:3: error: something terrible happened)";

  ClangdDiagnosticOptions Opts;
  // Transform diagnostics and check the results.
  std::vector<std::pair<clangd::Diagnostic, std::vector<clangd::Fix>>> LSPDiags;
  toLSPDiags(D, MainFile, Opts,
             [&](clangd::Diagnostic LSPDiag, ArrayRef<clangd::Fix> Fixes) {
               LSPDiags.push_back(
                   {std::move(LSPDiag),
                    std::vector<clangd::Fix>(Fixes.begin(), Fixes.end())});
             });

  EXPECT_THAT(
      LSPDiags,
      ElementsAre(Pair(EqualToLSPDiag(MainLSP), ElementsAre(EqualToFix(F))),
                  Pair(EqualToLSPDiag(NoteInMainLSP), IsEmpty())));
  EXPECT_EQ(LSPDiags[0].first.code, "undeclared_var_use");
  EXPECT_EQ(LSPDiags[0].first.source, "clang");
  EXPECT_EQ(LSPDiags[1].first.code, "");
  EXPECT_EQ(LSPDiags[1].first.source, "");

  // Same thing, but don't flatten notes into the main list.
  LSPDiags.clear();
  Opts.EmitRelatedLocations = true;
  toLSPDiags(D, MainFile, Opts,
             [&](clangd::Diagnostic LSPDiag, ArrayRef<clangd::Fix> Fixes) {
               LSPDiags.push_back(
                   {std::move(LSPDiag),
                    std::vector<clangd::Fix>(Fixes.begin(), Fixes.end())});
             });
  MainLSP.message = "Something terrible happened (fix available)";
  DiagnosticRelatedInformation NoteInMainDRI;
  NoteInMainDRI.message = "Declared somewhere in the main file";
  NoteInMainDRI.location.range = NoteInMain.Range;
  NoteInMainDRI.location.uri = MainFile;
  MainLSP.relatedInformation = {NoteInMainDRI};
  DiagnosticRelatedInformation NoteInHeaderDRI;
  NoteInHeaderDRI.message = "Declared somewhere in the header file";
  NoteInHeaderDRI.location.range = NoteInHeader.Range;
  NoteInHeaderDRI.location.uri = HeaderFile;
  MainLSP.relatedInformation = {NoteInMainDRI, NoteInHeaderDRI};
  EXPECT_THAT(LSPDiags, ElementsAre(Pair(EqualToLSPDiag(MainLSP),
                                         ElementsAre(EqualToFix(F)))));
}

struct SymbolWithHeader {
  std::string QName;
  std::string DeclaringFile;
  std::string IncludeHeader;
};

std::unique_ptr<SymbolIndex>
buildIndexWithSymbol(llvm::ArrayRef<SymbolWithHeader> Syms) {
  SymbolSlab::Builder Slab;
  for (const auto &S : Syms) {
    Symbol Sym = cls(S.QName);
    Sym.Flags |= Symbol::IndexedForCodeCompletion;
    Sym.CanonicalDeclaration.FileURI = S.DeclaringFile.c_str();
    Sym.Definition.FileURI = S.DeclaringFile.c_str();
    Sym.IncludeHeaders.emplace_back(S.IncludeHeader, 1);
    Slab.insert(Sym);
  }
  return MemIndex::build(std::move(Slab).build(), RefSlab(), RelationSlab());
}

TEST(IncludeFixerTest, IncompleteType) {
  Annotations Test(R"cpp(// error-ok
$insert[[]]namespace ns {
  class X;
  $nested[[X::]]Nested n;
}
class Y : $base[[public ns::X]] {};
void test(ns::X *x, ns::X& ref_x) {
  x$access[[->]]f();
  auto& $type[[[]]a] = *x;

  ns::X $incomplete[[var]];
  $tag[[ref_x]]->f();
  $use[[ns::X()]];
  $sizeof[[sizeof]](ns::X);
  for (auto it : $for[[ref_x]]);
}

ns::X $return[[func]]() {}

class T {
  ns::X $field[[x]];
};
  )cpp");
  auto TU = TestTU::withCode(Test.code());
  TU.ExtraArgs.push_back("-std=c++17");
  auto Index = buildIndexWithSymbol(
      {SymbolWithHeader{"ns::X", "unittest:///x.h", "\"x.h\""}});
  TU.ExternalIndex = Index.get();

  EXPECT_THAT(
      *TU.build().getDiagnostics(),
      UnorderedElementsAreArray(
          {AllOf(Diag(Test.range("nested"),
                      "incomplete type 'ns::X' named in nested name specifier"),
                 DiagName("incomplete_nested_name_spec"),
                 WithFix(Fix(Test.range("insert"), "#include \"x.h\"\n",
                             "Add include \"x.h\" for symbol ns::X"))),
           AllOf(Diag(Test.range("base"), "base class has incomplete type"),
                 DiagName("incomplete_base_class"),
                 WithFix(Fix(Test.range("insert"), "#include \"x.h\"\n",
                             "Add include \"x.h\" for symbol ns::X"))),
           AllOf(Diag(Test.range("access"),
                      "member access into incomplete type 'ns::X'"),
                 DiagName("incomplete_member_access"),
                 WithFix(Fix(Test.range("insert"), "#include \"x.h\"\n",
                             "Add include \"x.h\" for symbol ns::X"))),
           AllOf(
               Diag(
                   Test.range("type"),
                   "incomplete type 'ns::X' where a complete type is required"),
               DiagName("incomplete_type"),
               WithFix(Fix(Test.range("insert"), "#include \"x.h\"\n",
                           "Add include \"x.h\" for symbol ns::X"))),
           AllOf(Diag(Test.range("incomplete"),
                      "variable has incomplete type 'ns::X'"),
                 DiagName("typecheck_decl_incomplete_type"),
                 WithFix(Fix(Test.range("insert"), "#include \"x.h\"\n",
                             "Add include \"x.h\" for symbol ns::X"))),
           AllOf(
               Diag(Test.range("tag"), "incomplete definition of type 'ns::X'"),
               DiagName("typecheck_incomplete_tag"),
               WithFix(Fix(Test.range("insert"), "#include \"x.h\"\n",
                           "Add include \"x.h\" for symbol ns::X"))),
           AllOf(Diag(Test.range("use"),
                      "invalid use of incomplete type 'ns::X'"),
                 DiagName("invalid_incomplete_type_use"),
                 WithFix(Fix(Test.range("insert"), "#include \"x.h\"\n",
                             "Add include \"x.h\" for symbol ns::X"))),
           AllOf(Diag(Test.range("sizeof"),
                      "invalid application of 'sizeof' to "
                      "an incomplete type 'ns::X'"),
                 DiagName("sizeof_alignof_incomplete_or_sizeless_type"),
                 WithFix(Fix(Test.range("insert"), "#include \"x.h\"\n",
                             "Add include \"x.h\" for symbol ns::X"))),
           AllOf(Diag(Test.range("for"),
                      "cannot use incomplete type 'ns::X' as a range"),
                 DiagName("for_range_incomplete_type"),
                 WithFix(Fix(Test.range("insert"), "#include \"x.h\"\n",
                             "Add include \"x.h\" for symbol ns::X"))),
           AllOf(Diag(Test.range("return"),
                      "incomplete result type 'ns::X' in function definition"),
                 DiagName("func_def_incomplete_result"),
                 WithFix(Fix(Test.range("insert"), "#include \"x.h\"\n",
                             "Add include \"x.h\" for symbol ns::X"))),
           AllOf(Diag(Test.range("field"), "field has incomplete type 'ns::X'"),
                 DiagName("field_incomplete_or_sizeless"),
                 WithFix(Fix(Test.range("insert"), "#include \"x.h\"\n",
                             "Add include \"x.h\" for symbol ns::X")))}))
      << Test.code();
}

TEST(IncludeFixerTest, NoSuggestIncludeWhenNoDefinitionInHeader) {
  Annotations Test(R"cpp(// error-ok
$insert[[]]namespace ns {
  class X;
}
class Y : $base[[public ns::X]] {};
int main() {
  ns::X *x;
  x$access[[->]]f();
}
  )cpp");
  auto TU = TestTU::withCode(Test.code());
  Symbol Sym = cls("ns::X");
  Sym.Flags |= Symbol::IndexedForCodeCompletion;
  Sym.CanonicalDeclaration.FileURI = "unittest:///x.h";
  Sym.Definition.FileURI = "unittest:///x.cc";
  Sym.IncludeHeaders.emplace_back("\"x.h\"", 1);

  SymbolSlab::Builder Slab;
  Slab.insert(Sym);
  auto Index =
      MemIndex::build(std::move(Slab).build(), RefSlab(), RelationSlab());
  TU.ExternalIndex = Index.get();

  EXPECT_THAT(*TU.build().getDiagnostics(),
              UnorderedElementsAre(
                  Diag(Test.range("base"), "base class has incomplete type"),
                  Diag(Test.range("access"),
                       "member access into incomplete type 'ns::X'")));
}

TEST(IncludeFixerTest, Typo) {
  Annotations Test(R"cpp(// error-ok
$insert[[]]namespace ns {
void foo() {
  $unqualified1[[X]] x;
  // No fix if the unresolved type is used as specifier. (ns::)X::Nested will be
  // considered the unresolved type.
  $unqualified2[[X]]::Nested n;
}
}
void bar() {
  ns::$qualified1[[X]] x; // ns:: is valid.
  ns::$qualified2[[X]](); // Error: no member in namespace

  ::$global[[Global]] glob;
}
using Type = ns::$template[[Foo]]<int>;
  )cpp");
  auto TU = TestTU::withCode(Test.code());
  auto Index = buildIndexWithSymbol(
      {SymbolWithHeader{"ns::X", "unittest:///x.h", "\"x.h\""},
       SymbolWithHeader{"Global", "unittest:///global.h", "\"global.h\""},
       SymbolWithHeader{"ns::Foo", "unittest:///foo.h", "\"foo.h\""}});
  TU.ExternalIndex = Index.get();

  EXPECT_THAT(
      *TU.build().getDiagnostics(),
      UnorderedElementsAre(
          AllOf(Diag(Test.range("unqualified1"), "unknown type name 'X'"),
                DiagName("unknown_typename"),
                WithFix(Fix(Test.range("insert"), "#include \"x.h\"\n",
                            "Add include \"x.h\" for symbol ns::X"))),
          Diag(Test.range("unqualified2"), "use of undeclared identifier 'X'"),
          AllOf(Diag(Test.range("qualified1"),
                     "no type named 'X' in namespace 'ns'"),
                DiagName("typename_nested_not_found"),
                WithFix(Fix(Test.range("insert"), "#include \"x.h\"\n",
                            "Add include \"x.h\" for symbol ns::X"))),
          AllOf(Diag(Test.range("qualified2"),
                     "no member named 'X' in namespace 'ns'"),
                DiagName("no_member"),
                WithFix(Fix(Test.range("insert"), "#include \"x.h\"\n",
                            "Add include \"x.h\" for symbol ns::X"))),
          AllOf(Diag(Test.range("global"),
                     "no type named 'Global' in the global namespace"),
                DiagName("typename_nested_not_found"),
                WithFix(Fix(Test.range("insert"), "#include \"global.h\"\n",
                            "Add include \"global.h\" for symbol Global"))),
          AllOf(Diag(Test.range("template"),
                     "no template named 'Foo' in namespace 'ns'"),
                DiagName("no_member_template"),
                WithFix(Fix(Test.range("insert"), "#include \"foo.h\"\n",
                            "Add include \"foo.h\" for symbol ns::Foo")))));
}

TEST(IncludeFixerTest, MultipleMatchedSymbols) {
  Annotations Test(R"cpp(// error-ok
$insert[[]]namespace na {
namespace nb {
void foo() {
  $unqualified[[X]] x;
}
}
}
  )cpp");
  auto TU = TestTU::withCode(Test.code());
  auto Index = buildIndexWithSymbol(
      {SymbolWithHeader{"na::X", "unittest:///a.h", "\"a.h\""},
       SymbolWithHeader{"na::nb::X", "unittest:///b.h", "\"b.h\""}});
  TU.ExternalIndex = Index.get();

  EXPECT_THAT(*TU.build().getDiagnostics(),
              UnorderedElementsAre(AllOf(
                  Diag(Test.range("unqualified"), "unknown type name 'X'"),
                  DiagName("unknown_typename"),
                  WithFix(Fix(Test.range("insert"), "#include \"a.h\"\n",
                              "Add include \"a.h\" for symbol na::X"),
                          Fix(Test.range("insert"), "#include \"b.h\"\n",
                              "Add include \"b.h\" for symbol na::nb::X")))));
}

TEST(IncludeFixerTest, NoCrashMemebrAccess) {
  Annotations Test(R"cpp(// error-ok
    struct X { int  xyz; };
    void g() { X x; x.$[[xy]]; }
  )cpp");
  auto TU = TestTU::withCode(Test.code());
  auto Index = buildIndexWithSymbol(
      SymbolWithHeader{"na::X", "unittest:///a.h", "\"a.h\""});
  TU.ExternalIndex = Index.get();

  EXPECT_THAT(
      *TU.build().getDiagnostics(),
      UnorderedElementsAre(Diag(Test.range(), "no member named 'xy' in 'X'")));
}

TEST(IncludeFixerTest, UseCachedIndexResults) {
  // As index results for the identical request are cached, more than 5 fixes
  // are generated.
  Annotations Test(R"cpp(// error-ok
$insert[[]]void foo() {
  $x1[[X]] x;
  $x2[[X]] x;
  $x3[[X]] x;
  $x4[[X]] x;
  $x5[[X]] x;
  $x6[[X]] x;
  $x7[[X]] x;
}

class X;
void bar(X *x) {
  x$a1[[->]]f();
  x$a2[[->]]f();
  x$a3[[->]]f();
  x$a4[[->]]f();
  x$a5[[->]]f();
  x$a6[[->]]f();
  x$a7[[->]]f();
}
  )cpp");
  auto TU = TestTU::withCode(Test.code());
  auto Index =
      buildIndexWithSymbol(SymbolWithHeader{"X", "unittest:///a.h", "\"a.h\""});
  TU.ExternalIndex = Index.get();

  auto Parsed = TU.build();
  for (const auto &D : *Parsed.getDiagnostics()) {
    if (D.Fixes.size() != 1) {
      ADD_FAILURE() << "D.Fixes.size() != 1";
      continue;
    }
    EXPECT_EQ(D.Fixes[0].Message,
              std::string("Add include \"a.h\" for symbol X"));
  }
}

TEST(IncludeFixerTest, UnresolvedNameAsSpecifier) {
  Annotations Test(R"cpp(// error-ok
$insert[[]]namespace ns {
}
void g() {  ns::$[[scope]]::X_Y();  }
  )cpp");
  TestTU TU;
  TU.Code = std::string(Test.code());
  // FIXME: Figure out why this is needed and remove it, PR43662.
  TU.ExtraArgs.push_back("-fno-ms-compatibility");
  auto Index = buildIndexWithSymbol(
      SymbolWithHeader{"ns::scope::X_Y", "unittest:///x.h", "\"x.h\""});
  TU.ExternalIndex = Index.get();

  EXPECT_THAT(
      *TU.build().getDiagnostics(),
      UnorderedElementsAre(AllOf(
          Diag(Test.range(), "no member named 'scope' in namespace 'ns'"),
          DiagName("no_member"),
          WithFix(Fix(Test.range("insert"), "#include \"x.h\"\n",
                      "Add include \"x.h\" for symbol ns::scope::X_Y")))));
}

TEST(IncludeFixerTest, UnresolvedSpecifierWithSemaCorrection) {
  Annotations Test(R"cpp(// error-ok
$insert[[]]namespace clang {
void f() {
  // "clangd::" will be corrected to "clang::" by Sema.
  $q1[[clangd]]::$x[[X]] x;
  $q2[[clangd]]::$ns[[ns]]::Y y;
}
}
  )cpp");
  TestTU TU;
  TU.Code = std::string(Test.code());
  // FIXME: Figure out why this is needed and remove it, PR43662.
  TU.ExtraArgs.push_back("-fno-ms-compatibility");
  auto Index = buildIndexWithSymbol(
      {SymbolWithHeader{"clang::clangd::X", "unittest:///x.h", "\"x.h\""},
       SymbolWithHeader{"clang::clangd::ns::Y", "unittest:///y.h", "\"y.h\""}});
  TU.ExternalIndex = Index.get();

  EXPECT_THAT(
      *TU.build().getDiagnostics(),
      UnorderedElementsAre(
          AllOf(
              Diag(Test.range("q1"), "use of undeclared identifier 'clangd'; "
                                     "did you mean 'clang'?"),
              DiagName("undeclared_var_use_suggest"),
              WithFix(_, // change clangd to clang
                      Fix(Test.range("insert"), "#include \"x.h\"\n",
                          "Add include \"x.h\" for symbol clang::clangd::X"))),
          AllOf(
              Diag(Test.range("x"), "no type named 'X' in namespace 'clang'"),
              DiagName("typename_nested_not_found"),
              WithFix(Fix(Test.range("insert"), "#include \"x.h\"\n",
                          "Add include \"x.h\" for symbol clang::clangd::X"))),
          AllOf(
              Diag(Test.range("q2"), "use of undeclared identifier 'clangd'; "
                                     "did you mean 'clang'?"),
              DiagName("undeclared_var_use_suggest"),
              WithFix(
                  _, // change clangd to clang
                  Fix(Test.range("insert"), "#include \"y.h\"\n",
                      "Add include \"y.h\" for symbol clang::clangd::ns::Y"))),
          AllOf(Diag(Test.range("ns"),
                     "no member named 'ns' in namespace 'clang'"),
                DiagName("no_member"),
                WithFix(Fix(
                    Test.range("insert"), "#include \"y.h\"\n",
                    "Add include \"y.h\" for symbol clang::clangd::ns::Y")))));
}

TEST(IncludeFixerTest, SpecifiedScopeIsNamespaceAlias) {
  Annotations Test(R"cpp(// error-ok
$insert[[]]namespace a {}
namespace b = a;
namespace c {
  b::$[[X]] x;
}
  )cpp");
  auto TU = TestTU::withCode(Test.code());
  auto Index = buildIndexWithSymbol(
      SymbolWithHeader{"a::X", "unittest:///x.h", "\"x.h\""});
  TU.ExternalIndex = Index.get();

  EXPECT_THAT(*TU.build().getDiagnostics(),
              UnorderedElementsAre(AllOf(
                  Diag(Test.range(), "no type named 'X' in namespace 'a'"),
                  DiagName("typename_nested_not_found"),
                  WithFix(Fix(Test.range("insert"), "#include \"x.h\"\n",
                              "Add include \"x.h\" for symbol a::X")))));
}

TEST(IncludeFixerTest, NoCrashOnTemplateInstantiations) {
  Annotations Test(R"cpp(
    template <typename T> struct Templ {
      template <typename U>
      typename U::type operator=(const U &);
    };

    struct A {
      Templ<char> s;
      A() { [[a]]; /*error-ok*/ } // crash if we compute scopes lazily.
    };
  )cpp");

  auto TU = TestTU::withCode(Test.code());
  auto Index = buildIndexWithSymbol({});
  TU.ExternalIndex = Index.get();

  EXPECT_THAT(
      *TU.build().getDiagnostics(),
      ElementsAre(Diag(Test.range(), "use of undeclared identifier 'a'")));
}

TEST(DiagsInHeaders, DiagInsideHeader) {
  Annotations Main(R"cpp(
    #include [["a.h"]]
    void foo() {})cpp");
  Annotations Header("[[no_type_spec]]; // error-ok");
  TestTU TU = TestTU::withCode(Main.code());
  TU.AdditionalFiles = {{"a.h", std::string(Header.code())}};
  EXPECT_THAT(*TU.build().getDiagnostics(),
              UnorderedElementsAre(AllOf(
                  Diag(Main.range(), "in included file: C++ requires a "
                                     "type specifier for all declarations"),
                  WithNote(Diag(Header.range(), "error occurred here")))));
}

TEST(DiagsInHeaders, DiagInTransitiveInclude) {
  Annotations Main(R"cpp(
    #include [["a.h"]]
    void foo() {})cpp");
  TestTU TU = TestTU::withCode(Main.code());
  TU.AdditionalFiles = {{"a.h", "#include \"b.h\""},
                        {"b.h", "no_type_spec; // error-ok"}};
  EXPECT_THAT(*TU.build().getDiagnostics(),
              UnorderedElementsAre(
                  Diag(Main.range(), "in included file: C++ requires a "
                                     "type specifier for all declarations")));
}

TEST(DiagsInHeaders, DiagInMultipleHeaders) {
  Annotations Main(R"cpp(
    #include $a[["a.h"]]
    #include $b[["b.h"]]
    void foo() {})cpp");
  TestTU TU = TestTU::withCode(Main.code());
  TU.AdditionalFiles = {{"a.h", "no_type_spec; // error-ok"},
                        {"b.h", "no_type_spec; // error-ok"}};
  EXPECT_THAT(*TU.build().getDiagnostics(),
              UnorderedElementsAre(
                  Diag(Main.range("a"), "in included file: C++ requires a type "
                                        "specifier for all declarations"),
                  Diag(Main.range("b"), "in included file: C++ requires a type "
                                        "specifier for all declarations")));
}

TEST(DiagsInHeaders, PreferExpansionLocation) {
  Annotations Main(R"cpp(
    #include [["a.h"]]
    #include "b.h"
    void foo() {})cpp");
  TestTU TU = TestTU::withCode(Main.code());
  TU.AdditionalFiles = {
      {"a.h", "#include \"b.h\"\n"},
      {"b.h", "#ifndef X\n#define X\nno_type_spec; // error-ok\n#endif"}};
  EXPECT_THAT(*TU.build().getDiagnostics(),
              UnorderedElementsAre(Diag(Main.range(),
                                        "in included file: C++ requires a type "
                                        "specifier for all declarations")));
}

TEST(DiagsInHeaders, PreferExpansionLocationMacros) {
  Annotations Main(R"cpp(
    #define X
    #include "a.h"
    #undef X
    #include [["b.h"]]
    void foo() {})cpp");
  TestTU TU = TestTU::withCode(Main.code());
  TU.AdditionalFiles = {
      {"a.h", "#include \"c.h\"\n"},
      {"b.h", "#include \"c.h\"\n"},
      {"c.h", "#ifndef X\n#define X\nno_type_spec; // error-ok\n#endif"}};
  EXPECT_THAT(*TU.build().getDiagnostics(),
              UnorderedElementsAre(
                  Diag(Main.range(), "in included file: C++ requires a "
                                     "type specifier for all declarations")));
}

TEST(DiagsInHeaders, LimitDiagsOutsideMainFile) {
  Annotations Main(R"cpp(
    #include [["a.h"]]
    #include "b.h"
    void foo() {})cpp");
  TestTU TU = TestTU::withCode(Main.code());
  TU.AdditionalFiles = {{"a.h", "#include \"c.h\"\n"},
                        {"b.h", "#include \"c.h\"\n"},
                        {"c.h", R"cpp(
      #ifndef X
      #define X
      no_type_spec_0; // error-ok
      no_type_spec_1;
      no_type_spec_2;
      no_type_spec_3;
      no_type_spec_4;
      no_type_spec_5;
      no_type_spec_6;
      no_type_spec_7;
      no_type_spec_8;
      no_type_spec_9;
      no_type_spec_10;
      #endif)cpp"}};
  EXPECT_THAT(*TU.build().getDiagnostics(),
              UnorderedElementsAre(
                  Diag(Main.range(), "in included file: C++ requires a "
                                     "type specifier for all declarations")));
}

TEST(DiagsInHeaders, OnlyErrorOrFatal) {
  Annotations Main(R"cpp(
    #include [["a.h"]]
    void foo() {})cpp");
  Annotations Header(R"cpp(
    [[no_type_spec]]; // error-ok
    int x = 5/0;)cpp");
  TestTU TU = TestTU::withCode(Main.code());
  TU.AdditionalFiles = {{"a.h", std::string(Header.code())}};
  EXPECT_THAT(*TU.build().getDiagnostics(),
              UnorderedElementsAre(AllOf(
                  Diag(Main.range(), "in included file: C++ requires "
                                     "a type specifier for all declarations"),
                  WithNote(Diag(Header.range(), "error occurred here")))));
}

TEST(DiagsInHeaders, OnlyDefaultErrorOrFatal) {
  Annotations Main(R"cpp(
    #include [["a.h"]] // get unused "foo" warning when building preamble.
    )cpp");
  Annotations Header(R"cpp(
    namespace { void foo() {} }
    void func() {foo();} ;)cpp");
  TestTU TU = TestTU::withCode(Main.code());
  TU.AdditionalFiles = {{"a.h", std::string(Header.code())}};
  // promote warnings to errors.
  TU.ExtraArgs = {"-Werror", "-Wunused"};
  EXPECT_THAT(*TU.build().getDiagnostics(), IsEmpty());
}

TEST(DiagsInHeaders, FromNonWrittenSources) {
  Annotations Main(R"cpp(
    #include [["a.h"]]
    void foo() {})cpp");
  Annotations Header(R"cpp(
    int x = 5/0;
    int b = [[FOO]]; // error-ok)cpp");
  TestTU TU = TestTU::withCode(Main.code());
  TU.AdditionalFiles = {{"a.h", std::string(Header.code())}};
  TU.ExtraArgs = {"-DFOO=NOOO"};
  EXPECT_THAT(*TU.build().getDiagnostics(),
              UnorderedElementsAre(AllOf(
                  Diag(Main.range(),
                       "in included file: use of undeclared identifier 'NOOO'"),
                  WithNote(Diag(Header.range(), "error occurred here")))));
}

TEST(DiagsInHeaders, ErrorFromMacroExpansion) {
  Annotations Main(R"cpp(
  void bar() {
    int fo; // error-ok
    #include [["a.h"]]
  })cpp");
  Annotations Header(R"cpp(
  #define X foo
  X;)cpp");
  TestTU TU = TestTU::withCode(Main.code());
  TU.AdditionalFiles = {{"a.h", std::string(Header.code())}};
  EXPECT_THAT(*TU.build().getDiagnostics(),
              UnorderedElementsAre(
                  Diag(Main.range(), "in included file: use of undeclared "
                                     "identifier 'foo'; did you mean 'fo'?")));
}

TEST(DiagsInHeaders, ErrorFromMacroArgument) {
  Annotations Main(R"cpp(
  void bar() {
    int fo; // error-ok
    #include [["a.h"]]
  })cpp");
  Annotations Header(R"cpp(
  #define X(arg) arg
  X(foo);)cpp");
  TestTU TU = TestTU::withCode(Main.code());
  TU.AdditionalFiles = {{"a.h", std::string(Header.code())}};
  EXPECT_THAT(*TU.build().getDiagnostics(),
              UnorderedElementsAre(
                  Diag(Main.range(), "in included file: use of undeclared "
                                     "identifier 'foo'; did you mean 'fo'?")));
}

TEST(IgnoreDiags, FromNonWrittenInclude) {
  TestTU TU;
  TU.ExtraArgs.push_back("--include=a.h");
  TU.AdditionalFiles = {{"a.h", "void main();"}};
  // The diagnostic "main must return int" is from the header, we don't attempt
  // to render it in the main file as there is no written location there.
  EXPECT_THAT(*TU.build().getDiagnostics(), UnorderedElementsAre());
}

TEST(ToLSPDiag, RangeIsInMain) {
  ClangdDiagnosticOptions Opts;
  clangd::Diag D;
  D.Range = {pos(1, 2), pos(3, 4)};
  D.Notes.emplace_back();
  Note &N = D.Notes.back();
  N.Range = {pos(2, 3), pos(3, 4)};

  D.InsideMainFile = true;
  N.InsideMainFile = false;
  toLSPDiags(D, {}, Opts,
             [&](clangd::Diagnostic LSPDiag, ArrayRef<clangd::Fix>) {
               EXPECT_EQ(LSPDiag.range, D.Range);
             });

  D.InsideMainFile = false;
  N.InsideMainFile = true;
  toLSPDiags(D, {}, Opts,
             [&](clangd::Diagnostic LSPDiag, ArrayRef<clangd::Fix>) {
               EXPECT_EQ(LSPDiag.range, N.Range);
             });
}

TEST(ParsedASTTest, ModuleSawDiag) {
  static constexpr const llvm::StringLiteral KDiagMsg = "StampedDiag";
  struct DiagModifierModule final : public FeatureModule {
    struct Listener : public FeatureModule::ASTListener {
      void sawDiagnostic(const clang::Diagnostic &Info,
                         clangd::Diag &Diag) override {
        Diag.Message = KDiagMsg.str();
      }
    };
    std::unique_ptr<ASTListener> astListeners() override {
      return std::make_unique<Listener>();
    };
  };
  FeatureModuleSet FMS;
  FMS.add(std::make_unique<DiagModifierModule>());

  Annotations Code("[[test]]; /* error-ok */");
  TestTU TU;
  TU.Code = Code.code().str();
  TU.FeatureModules = &FMS;

  auto AST = TU.build();
  EXPECT_THAT(*AST.getDiagnostics(),
              testing::Contains(Diag(Code.range(), KDiagMsg.str())));
}

TEST(Preamble, EndsOnNonEmptyLine) {
  TestTU TU;
  TU.ExtraArgs = {"-Wnewline-eof"};

  {
    TU.Code = "#define FOO\n  void bar();\n";
    auto AST = TU.build();
    EXPECT_THAT(*AST.getDiagnostics(), IsEmpty());
  }
  {
    Annotations Code("#define FOO[[]]");
    TU.Code = Code.code().str();
    auto AST = TU.build();
    EXPECT_THAT(
        *AST.getDiagnostics(),
        testing::Contains(Diag(Code.range(), "no newline at end of file")));
  }
}

TEST(Diagnostics, Tags) {
  TestTU TU;
  TU.ExtraArgs = {"-Wunused", "-Wdeprecated"};
  Annotations Test(R"cpp(
  void bar() __attribute__((deprecated));
  void foo() {
    int $unused[[x]];
    $deprecated[[bar]]();
  })cpp");
  TU.Code = Test.code().str();
  EXPECT_THAT(*TU.build().getDiagnostics(),
              UnorderedElementsAre(
                  AllOf(Diag(Test.range("unused"), "unused variable 'x'"),
                        WithTag(DiagnosticTag::Unnecessary)),
                  AllOf(Diag(Test.range("deprecated"), "'bar' is deprecated"),
                        WithTag(DiagnosticTag::Deprecated))));
}
} // namespace
} // namespace clangd
} // namespace clang
