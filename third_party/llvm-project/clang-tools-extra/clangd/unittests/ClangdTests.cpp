//===-- ClangdTests.cpp - Clangd unit tests ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "ClangdLSPServer.h"
#include "ClangdServer.h"
#include "CodeComplete.h"
#include "ConfigFragment.h"
#include "GlobalCompilationDatabase.h"
#include "Matchers.h"
#include "SyncAPI.h"
#include "TestFS.h"
#include "TestTU.h"
#include "TidyProvider.h"
#include "URI.h"
#include "refactor/Tweak.h"
#include "support/MemoryTree.h"
#include "support/Path.h"
#include "support/Threading.h"
#include "clang/Config/config.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace clang {
namespace clangd {

namespace {

using ::testing::AllOf;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::Gt;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

MATCHER_P2(DeclAt, File, Range, "") {
  return arg.PreferredDeclaration ==
         Location{URIForFile::canonicalize(File, testRoot()), Range};
}

bool diagsContainErrors(const std::vector<Diag> &Diagnostics) {
  for (auto D : Diagnostics) {
    if (D.Severity == DiagnosticsEngine::Error ||
        D.Severity == DiagnosticsEngine::Fatal)
      return true;
  }
  return false;
}

class ErrorCheckingCallbacks : public ClangdServer::Callbacks {
public:
  void onDiagnosticsReady(PathRef File, llvm::StringRef Version,
                          std::vector<Diag> Diagnostics) override {
    bool HadError = diagsContainErrors(Diagnostics);
    std::lock_guard<std::mutex> Lock(Mutex);
    HadErrorInLastDiags = HadError;
  }

  bool hadErrorInLastDiags() {
    std::lock_guard<std::mutex> Lock(Mutex);
    return HadErrorInLastDiags;
  }

private:
  std::mutex Mutex;
  bool HadErrorInLastDiags = false;
};

/// For each file, record whether the last published diagnostics contained at
/// least one error.
class MultipleErrorCheckingCallbacks : public ClangdServer::Callbacks {
public:
  void onDiagnosticsReady(PathRef File, llvm::StringRef Version,
                          std::vector<Diag> Diagnostics) override {
    bool HadError = diagsContainErrors(Diagnostics);

    std::lock_guard<std::mutex> Lock(Mutex);
    LastDiagsHadError[File] = HadError;
  }

  /// Exposes all files consumed by onDiagnosticsReady in an unspecified order.
  /// For each file, a bool value indicates whether the last diagnostics
  /// contained an error.
  std::vector<std::pair<Path, bool>> filesWithDiags() const {
    std::vector<std::pair<Path, bool>> Result;
    std::lock_guard<std::mutex> Lock(Mutex);
    for (const auto &It : LastDiagsHadError)
      Result.emplace_back(std::string(It.first()), It.second);
    return Result;
  }

  void clear() {
    std::lock_guard<std::mutex> Lock(Mutex);
    LastDiagsHadError.clear();
  }

private:
  mutable std::mutex Mutex;
  llvm::StringMap<bool> LastDiagsHadError;
};

/// Replaces all patterns of the form 0x123abc with spaces
std::string replacePtrsInDump(std::string const &Dump) {
  llvm::Regex RE("0x[0-9a-fA-F]+");
  llvm::SmallVector<llvm::StringRef, 1> Matches;
  llvm::StringRef Pending = Dump;

  std::string Result;
  while (RE.match(Pending, &Matches)) {
    assert(Matches.size() == 1 && "Exactly one match expected");
    auto MatchPos = Matches[0].data() - Pending.data();

    Result += Pending.take_front(MatchPos);
    Pending = Pending.drop_front(MatchPos + Matches[0].size());
  }
  Result += Pending;

  return Result;
}

std::string dumpAST(ClangdServer &Server, PathRef File) {
  std::string Result;
  Notification Done;
  Server.customAction(File, "DumpAST", [&](llvm::Expected<InputsAndAST> AST) {
    if (AST) {
      llvm::raw_string_ostream ResultOS(Result);
      AST->AST.getASTContext().getTranslationUnitDecl()->dump(ResultOS, true);
    } else {
      llvm::consumeError(AST.takeError());
      Result = "<no-ast>";
    }
    Done.notify();
  });
  Done.wait();
  return Result;
}

std::string dumpASTWithoutMemoryLocs(ClangdServer &Server, PathRef File) {
  return replacePtrsInDump(dumpAST(Server, File));
}

std::string parseSourceAndDumpAST(
    PathRef SourceFileRelPath, llvm::StringRef SourceContents,
    std::vector<std::pair<PathRef, llvm::StringRef>> ExtraFiles = {},
    bool ExpectErrors = false) {
  MockFS FS;
  ErrorCheckingCallbacks DiagConsumer;
  MockCompilationDatabase CDB;
  ClangdServer Server(CDB, FS, ClangdServer::optsForTest(), &DiagConsumer);
  for (const auto &FileWithContents : ExtraFiles)
    FS.Files[testPath(FileWithContents.first)] =
        std::string(FileWithContents.second);

  auto SourceFilename = testPath(SourceFileRelPath);
  Server.addDocument(SourceFilename, SourceContents);
  auto Result = dumpASTWithoutMemoryLocs(Server, SourceFilename);
  EXPECT_TRUE(Server.blockUntilIdleForTest()) << "Waiting for diagnostics";
  EXPECT_EQ(ExpectErrors, DiagConsumer.hadErrorInLastDiags());
  return Result;
}

TEST(ClangdServerTest, Parse) {
  // FIXME: figure out a stable format for AST dumps, so that we can check the
  // output of the dump itself is equal to the expected one, not just that it's
  // different.
  auto Empty = parseSourceAndDumpAST("foo.cpp", "");
  auto OneDecl = parseSourceAndDumpAST("foo.cpp", "int a;");
  auto SomeDecls = parseSourceAndDumpAST("foo.cpp", "int a; int b; int c;");
  EXPECT_NE(Empty, OneDecl);
  EXPECT_NE(Empty, SomeDecls);
  EXPECT_NE(SomeDecls, OneDecl);

  auto Empty2 = parseSourceAndDumpAST("foo.cpp", "");
  auto OneDecl2 = parseSourceAndDumpAST("foo.cpp", "int a;");
  auto SomeDecls2 = parseSourceAndDumpAST("foo.cpp", "int a; int b; int c;");
  EXPECT_EQ(Empty, Empty2);
  EXPECT_EQ(OneDecl, OneDecl2);
  EXPECT_EQ(SomeDecls, SomeDecls2);
}

TEST(ClangdServerTest, ParseWithHeader) {
  parseSourceAndDumpAST("foo.cpp", "#include \"foo.h\"", {},
                        /*ExpectErrors=*/true);
  parseSourceAndDumpAST("foo.cpp", "#include \"foo.h\"", {{"foo.h", ""}},
                        /*ExpectErrors=*/false);

  const auto SourceContents = R"cpp(
#include "foo.h"
int b = a;
)cpp";
  parseSourceAndDumpAST("foo.cpp", SourceContents, {{"foo.h", ""}},
                        /*ExpectErrors=*/true);
  parseSourceAndDumpAST("foo.cpp", SourceContents, {{"foo.h", "int a;"}},
                        /*ExpectErrors=*/false);
}

TEST(ClangdServerTest, Reparse) {
  MockFS FS;
  ErrorCheckingCallbacks DiagConsumer;
  MockCompilationDatabase CDB;
  ClangdServer Server(CDB, FS, ClangdServer::optsForTest(), &DiagConsumer);

  const auto SourceContents = R"cpp(
#include "foo.h"
int b = a;
)cpp";

  auto FooCpp = testPath("foo.cpp");

  FS.Files[testPath("foo.h")] = "int a;";
  FS.Files[FooCpp] = SourceContents;

  Server.addDocument(FooCpp, SourceContents);
  ASSERT_TRUE(Server.blockUntilIdleForTest()) << "Waiting for diagnostics";
  auto DumpParse1 = dumpASTWithoutMemoryLocs(Server, FooCpp);
  EXPECT_FALSE(DiagConsumer.hadErrorInLastDiags());

  Server.addDocument(FooCpp, "");
  ASSERT_TRUE(Server.blockUntilIdleForTest()) << "Waiting for diagnostics";
  auto DumpParseEmpty = dumpASTWithoutMemoryLocs(Server, FooCpp);
  EXPECT_FALSE(DiagConsumer.hadErrorInLastDiags());

  Server.addDocument(FooCpp, SourceContents);
  ASSERT_TRUE(Server.blockUntilIdleForTest()) << "Waiting for diagnostics";
  auto DumpParse2 = dumpASTWithoutMemoryLocs(Server, FooCpp);
  EXPECT_FALSE(DiagConsumer.hadErrorInLastDiags());

  EXPECT_EQ(DumpParse1, DumpParse2);
  EXPECT_NE(DumpParse1, DumpParseEmpty);
}

TEST(ClangdServerTest, ReparseOnHeaderChange) {
  MockFS FS;
  ErrorCheckingCallbacks DiagConsumer;
  MockCompilationDatabase CDB;
  ClangdServer Server(CDB, FS, ClangdServer::optsForTest(), &DiagConsumer);

  const auto SourceContents = R"cpp(
#include "foo.h"
int b = a;
)cpp";

  auto FooCpp = testPath("foo.cpp");
  auto FooH = testPath("foo.h");

  FS.Files[FooH] = "int a;";
  FS.Files[FooCpp] = SourceContents;

  Server.addDocument(FooCpp, SourceContents);
  ASSERT_TRUE(Server.blockUntilIdleForTest()) << "Waiting for diagnostics";
  auto DumpParse1 = dumpASTWithoutMemoryLocs(Server, FooCpp);
  EXPECT_FALSE(DiagConsumer.hadErrorInLastDiags());

  FS.Files[FooH] = "";
  Server.addDocument(FooCpp, SourceContents);
  ASSERT_TRUE(Server.blockUntilIdleForTest()) << "Waiting for diagnostics";
  auto DumpParseDifferent = dumpASTWithoutMemoryLocs(Server, FooCpp);
  EXPECT_TRUE(DiagConsumer.hadErrorInLastDiags());

  FS.Files[FooH] = "int a;";
  Server.addDocument(FooCpp, SourceContents);
  ASSERT_TRUE(Server.blockUntilIdleForTest()) << "Waiting for diagnostics";
  auto DumpParse2 = dumpASTWithoutMemoryLocs(Server, FooCpp);
  EXPECT_FALSE(DiagConsumer.hadErrorInLastDiags());

  EXPECT_EQ(DumpParse1, DumpParse2);
  EXPECT_NE(DumpParse1, DumpParseDifferent);
}

TEST(ClangdServerTest, PropagatesContexts) {
  static Key<int> Secret;
  struct ContextReadingFS : public ThreadsafeFS {
    mutable int Got;

  private:
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> viewImpl() const override {
      Got = Context::current().getExisting(Secret);
      return buildTestFS({});
    }
  } FS;
  struct Callbacks : public ClangdServer::Callbacks {
    void onDiagnosticsReady(PathRef File, llvm::StringRef Version,
                            std::vector<Diag> Diagnostics) override {
      Got = Context::current().getExisting(Secret);
    }
    int Got;
  } Callbacks;
  MockCompilationDatabase CDB;

  // Verify that the context is plumbed to the FS provider and diagnostics.
  ClangdServer Server(CDB, FS, ClangdServer::optsForTest(), &Callbacks);
  {
    WithContextValue Entrypoint(Secret, 42);
    Server.addDocument(testPath("foo.cpp"), "void main(){}");
  }
  ASSERT_TRUE(Server.blockUntilIdleForTest());
  EXPECT_EQ(FS.Got, 42);
  EXPECT_EQ(Callbacks.Got, 42);
}

TEST(ClangdServerTest, RespectsConfig) {
  // Go-to-definition will resolve as marked if FOO is defined.
  Annotations Example(R"cpp(
  #ifdef FOO
  int [[x]];
  #else
  int x;
  #endif
  int y = ^x;
  )cpp");
  // Provide conditional config that defines FOO for foo.cc.
  class ConfigProvider : public config::Provider {
    std::vector<config::CompiledFragment>
    getFragments(const config::Params &,
                 config::DiagnosticCallback DC) const override {
      config::Fragment F;
      F.If.PathMatch.emplace_back(".*foo.cc");
      F.CompileFlags.Add.emplace_back("-DFOO=1");
      return {std::move(F).compile(DC)};
    }
  } CfgProvider;

  auto Opts = ClangdServer::optsForTest();
  Opts.ContextProvider =
      ClangdServer::createConfiguredContextProvider(&CfgProvider, nullptr);
  OverlayCDB CDB(/*Base=*/nullptr, /*FallbackFlags=*/{},
                 tooling::ArgumentsAdjuster(CommandMangler::forTests()));
  MockFS FS;
  ClangdServer Server(CDB, FS, Opts);
  // foo.cc sees the expected definition, as FOO is defined.
  Server.addDocument(testPath("foo.cc"), Example.code());
  auto Result = runLocateSymbolAt(Server, testPath("foo.cc"), Example.point());
  ASSERT_TRUE(bool(Result)) << Result.takeError();
  ASSERT_THAT(*Result, SizeIs(1));
  EXPECT_EQ(Result->front().PreferredDeclaration.range, Example.range());
  // bar.cc gets a different result, as FOO is not defined.
  Server.addDocument(testPath("bar.cc"), Example.code());
  Result = runLocateSymbolAt(Server, testPath("bar.cc"), Example.point());
  ASSERT_TRUE(bool(Result)) << Result.takeError();
  ASSERT_THAT(*Result, SizeIs(1));
  EXPECT_NE(Result->front().PreferredDeclaration.range, Example.range());
}

TEST(ClangdServerTest, PropagatesVersion) {
  MockCompilationDatabase CDB;
  MockFS FS;
  struct Callbacks : public ClangdServer::Callbacks {
    void onDiagnosticsReady(PathRef File, llvm::StringRef Version,
                            std::vector<Diag> Diagnostics) override {
      Got = Version.str();
    }
    std::string Got = "";
  } Callbacks;

  // Verify that the version is plumbed to diagnostics.
  ClangdServer Server(CDB, FS, ClangdServer::optsForTest(), &Callbacks);
  runAddDocument(Server, testPath("foo.cpp"), "void main(){}", "42");
  EXPECT_EQ(Callbacks.Got, "42");
}

// Only enable this test on Unix
#ifdef LLVM_ON_UNIX
TEST(ClangdServerTest, SearchLibDir) {
  // Checks that searches for GCC installation is done through vfs.
  MockFS FS;
  ErrorCheckingCallbacks DiagConsumer;
  MockCompilationDatabase CDB;
  CDB.ExtraClangFlags.insert(CDB.ExtraClangFlags.end(),
                             {"-xc++", "-target", "x86_64-linux-unknown",
                              "-m64", "--gcc-toolchain=/randomusr",
                              "-stdlib=libstdc++"});
  ClangdServer Server(CDB, FS, ClangdServer::optsForTest(), &DiagConsumer);

  // Just a random gcc version string
  SmallString<8> Version("4.9.3");

  // A lib dir for gcc installation
  SmallString<64> LibDir("/randomusr/lib/gcc/x86_64-linux-gnu");
  llvm::sys::path::append(LibDir, Version);

  // Put crtbegin.o into LibDir/64 to trick clang into thinking there's a gcc
  // installation there.
  SmallString<64> MockLibFile;
  llvm::sys::path::append(MockLibFile, LibDir, "64", "crtbegin.o");
  FS.Files[MockLibFile] = "";

  SmallString<64> IncludeDir("/randomusr/include/c++");
  llvm::sys::path::append(IncludeDir, Version);

  SmallString<64> StringPath;
  llvm::sys::path::append(StringPath, IncludeDir, "string");
  FS.Files[StringPath] = "class mock_string {};";

  auto FooCpp = testPath("foo.cpp");
  const auto SourceContents = R"cpp(
#include <string>
mock_string x;
)cpp";
  FS.Files[FooCpp] = SourceContents;

  runAddDocument(Server, FooCpp, SourceContents);
  EXPECT_FALSE(DiagConsumer.hadErrorInLastDiags());

  const auto SourceContentsWithError = R"cpp(
#include <string>
std::string x;
)cpp";
  runAddDocument(Server, FooCpp, SourceContentsWithError);
  EXPECT_TRUE(DiagConsumer.hadErrorInLastDiags());
}
#endif // LLVM_ON_UNIX

TEST(ClangdServerTest, ForceReparseCompileCommand) {
  MockFS FS;
  ErrorCheckingCallbacks DiagConsumer;
  MockCompilationDatabase CDB;
  ClangdServer Server(CDB, FS, ClangdServer::optsForTest(), &DiagConsumer);

  auto FooCpp = testPath("foo.cpp");
  const auto SourceContents1 = R"cpp(
template <class T>
struct foo { T x; };
)cpp";
  const auto SourceContents2 = R"cpp(
template <class T>
struct bar { T x; };
)cpp";

  FS.Files[FooCpp] = "";

  // First parse files in C mode and check they produce errors.
  CDB.ExtraClangFlags = {"-xc"};
  runAddDocument(Server, FooCpp, SourceContents1);
  EXPECT_TRUE(DiagConsumer.hadErrorInLastDiags());
  runAddDocument(Server, FooCpp, SourceContents2);
  EXPECT_TRUE(DiagConsumer.hadErrorInLastDiags());

  // Now switch to C++ mode.
  CDB.ExtraClangFlags = {"-xc++"};
  runAddDocument(Server, FooCpp, SourceContents2);
  EXPECT_FALSE(DiagConsumer.hadErrorInLastDiags());
  // Subsequent addDocument calls should finish without errors too.
  runAddDocument(Server, FooCpp, SourceContents1);
  EXPECT_FALSE(DiagConsumer.hadErrorInLastDiags());
  runAddDocument(Server, FooCpp, SourceContents2);
  EXPECT_FALSE(DiagConsumer.hadErrorInLastDiags());
}

TEST(ClangdServerTest, ForceReparseCompileCommandDefines) {
  MockFS FS;
  ErrorCheckingCallbacks DiagConsumer;
  MockCompilationDatabase CDB;
  ClangdServer Server(CDB, FS, ClangdServer::optsForTest(), &DiagConsumer);

  auto FooCpp = testPath("foo.cpp");
  const auto SourceContents = R"cpp(
#ifdef WITH_ERROR
this
#endif

int main() { return 0; }
)cpp";
  FS.Files[FooCpp] = "";

  // Parse with define, we expect to see the errors.
  CDB.ExtraClangFlags = {"-DWITH_ERROR"};
  runAddDocument(Server, FooCpp, SourceContents);
  EXPECT_TRUE(DiagConsumer.hadErrorInLastDiags());

  // Parse without the define, no errors should be produced.
  CDB.ExtraClangFlags = {};
  runAddDocument(Server, FooCpp, SourceContents);
  ASSERT_TRUE(Server.blockUntilIdleForTest());
  EXPECT_FALSE(DiagConsumer.hadErrorInLastDiags());
  // Subsequent addDocument call should finish without errors too.
  runAddDocument(Server, FooCpp, SourceContents);
  EXPECT_FALSE(DiagConsumer.hadErrorInLastDiags());
}

// Test ClangdServer.reparseOpenedFiles.
TEST(ClangdServerTest, ReparseOpenedFiles) {
  Annotations FooSource(R"cpp(
#ifdef MACRO
static void $one[[bob]]() {}
#else
static void $two[[bob]]() {}
#endif

int main () { bo^b (); return 0; }
)cpp");

  Annotations BarSource(R"cpp(
#ifdef MACRO
this is an error
#endif
)cpp");

  Annotations BazSource(R"cpp(
int hello;
)cpp");

  MockFS FS;
  MockCompilationDatabase CDB;
  MultipleErrorCheckingCallbacks DiagConsumer;
  ClangdServer Server(CDB, FS, ClangdServer::optsForTest(), &DiagConsumer);

  auto FooCpp = testPath("foo.cpp");
  auto BarCpp = testPath("bar.cpp");
  auto BazCpp = testPath("baz.cpp");

  FS.Files[FooCpp] = "";
  FS.Files[BarCpp] = "";
  FS.Files[BazCpp] = "";

  CDB.ExtraClangFlags = {"-DMACRO=1"};
  Server.addDocument(FooCpp, FooSource.code());
  Server.addDocument(BarCpp, BarSource.code());
  Server.addDocument(BazCpp, BazSource.code());
  ASSERT_TRUE(Server.blockUntilIdleForTest());

  EXPECT_THAT(DiagConsumer.filesWithDiags(),
              UnorderedElementsAre(Pair(FooCpp, false), Pair(BarCpp, true),
                                   Pair(BazCpp, false)));

  auto Locations = runLocateSymbolAt(Server, FooCpp, FooSource.point());
  EXPECT_TRUE(bool(Locations));
  EXPECT_THAT(*Locations, ElementsAre(DeclAt(FooCpp, FooSource.range("one"))));

  // Undefine MACRO, close baz.cpp.
  CDB.ExtraClangFlags.clear();
  DiagConsumer.clear();
  Server.removeDocument(BazCpp);
  Server.addDocument(FooCpp, FooSource.code());
  Server.addDocument(BarCpp, BarSource.code());
  ASSERT_TRUE(Server.blockUntilIdleForTest());

  EXPECT_THAT(DiagConsumer.filesWithDiags(),
              UnorderedElementsAre(Pair(FooCpp, false), Pair(BarCpp, false)));

  Locations = runLocateSymbolAt(Server, FooCpp, FooSource.point());
  EXPECT_TRUE(bool(Locations));
  EXPECT_THAT(*Locations, ElementsAre(DeclAt(FooCpp, FooSource.range("two"))));
}

MATCHER_P4(Stats, Name, UsesMemory, PreambleBuilds, ASTBuilds, "") {
  return arg.first() == Name &&
         (arg.second.UsedBytesAST + arg.second.UsedBytesPreamble != 0) ==
             UsesMemory &&
         std::tie(arg.second.PreambleBuilds, ASTBuilds) ==
             std::tie(PreambleBuilds, ASTBuilds);
}

TEST(ClangdServerTest, FileStats) {
  MockFS FS;
  ErrorCheckingCallbacks DiagConsumer;
  MockCompilationDatabase CDB;
  ClangdServer Server(CDB, FS, ClangdServer::optsForTest(), &DiagConsumer);

  Path FooCpp = testPath("foo.cpp");
  const auto SourceContents = R"cpp(
struct Something {
  int method();
};
)cpp";
  Path BarCpp = testPath("bar.cpp");

  FS.Files[FooCpp] = "";
  FS.Files[BarCpp] = "";

  EXPECT_THAT(Server.fileStats(), IsEmpty());

  Server.addDocument(FooCpp, SourceContents);
  Server.addDocument(BarCpp, SourceContents);
  ASSERT_TRUE(Server.blockUntilIdleForTest());

  EXPECT_THAT(Server.fileStats(),
              UnorderedElementsAre(Stats(FooCpp, true, 1, 1),
                                   Stats(BarCpp, true, 1, 1)));

  Server.removeDocument(FooCpp);
  ASSERT_TRUE(Server.blockUntilIdleForTest());
  EXPECT_THAT(Server.fileStats(), ElementsAre(Stats(BarCpp, true, 1, 1)));

  Server.removeDocument(BarCpp);
  ASSERT_TRUE(Server.blockUntilIdleForTest());
  EXPECT_THAT(Server.fileStats(), IsEmpty());
}

TEST(ClangdServerTest, InvalidCompileCommand) {
  MockFS FS;
  ErrorCheckingCallbacks DiagConsumer;
  MockCompilationDatabase CDB;

  ClangdServer Server(CDB, FS, ClangdServer::optsForTest(), &DiagConsumer);

  auto FooCpp = testPath("foo.cpp");
  // clang cannot create CompilerInvocation in this case.
  CDB.ExtraClangFlags.push_back("-###");

  // Clang can't parse command args in that case, but we shouldn't crash.
  runAddDocument(Server, FooCpp, "int main() {}");

  EXPECT_EQ(dumpAST(Server, FooCpp), "<no-ast>");
  EXPECT_ERROR(runLocateSymbolAt(Server, FooCpp, Position()));
  EXPECT_ERROR(runFindDocumentHighlights(Server, FooCpp, Position()));
  EXPECT_ERROR(runRename(Server, FooCpp, Position(), "new_name",
                         clangd::RenameOptions()));
  EXPECT_ERROR(runSignatureHelp(Server, FooCpp, Position()));
  // Identifier-based fallback completion.
  EXPECT_THAT(cantFail(runCodeComplete(Server, FooCpp, Position(),
                                       clangd::CodeCompleteOptions()))
                  .Completions,
              ElementsAre(Field(&CodeCompletion::Name, "int"),
                          Field(&CodeCompletion::Name, "main")));
}

TEST(ClangdThreadingTest, StressTest) {
  // Without 'static' clang gives an error for a usage inside TestDiagConsumer.
  static const unsigned FilesCount = 5;
  const unsigned RequestsCount = 500;
  // Blocking requests wait for the parsing to complete, they slow down the test
  // dramatically, so they are issued rarely. Each
  // BlockingRequestInterval-request will be a blocking one.
  const unsigned BlockingRequestInterval = 40;

  const auto SourceContentsWithoutErrors = R"cpp(
int a;
int b;
int c;
int d;
)cpp";

  const auto SourceContentsWithErrors = R"cpp(
int a = x;
int b;
int c;
int d;
)cpp";

  // Giving invalid line and column number should not crash ClangdServer, but
  // just to make sure we're sometimes hitting the bounds inside the file we
  // limit the intervals of line and column number that are generated.
  unsigned MaxLineForFileRequests = 7;
  unsigned MaxColumnForFileRequests = 10;

  std::vector<std::string> FilePaths;
  MockFS FS;
  for (unsigned I = 0; I < FilesCount; ++I) {
    std::string Name = std::string("Foo") + std::to_string(I) + ".cpp";
    FS.Files[Name] = "";
    FilePaths.push_back(testPath(Name));
  }

  struct FileStat {
    unsigned HitsWithoutErrors = 0;
    unsigned HitsWithErrors = 0;
    bool HadErrorsInLastDiags = false;
  };

  class TestDiagConsumer : public ClangdServer::Callbacks {
  public:
    TestDiagConsumer() : Stats(FilesCount, FileStat()) {}

    void onDiagnosticsReady(PathRef File, llvm::StringRef Version,
                            std::vector<Diag> Diagnostics) override {
      StringRef FileIndexStr = llvm::sys::path::stem(File);
      ASSERT_TRUE(FileIndexStr.consume_front("Foo"));

      unsigned long FileIndex = std::stoul(FileIndexStr.str());

      bool HadError = diagsContainErrors(Diagnostics);

      std::lock_guard<std::mutex> Lock(Mutex);
      if (HadError)
        Stats[FileIndex].HitsWithErrors++;
      else
        Stats[FileIndex].HitsWithoutErrors++;
      Stats[FileIndex].HadErrorsInLastDiags = HadError;
    }

    std::vector<FileStat> takeFileStats() {
      std::lock_guard<std::mutex> Lock(Mutex);
      return std::move(Stats);
    }

  private:
    std::mutex Mutex;
    std::vector<FileStat> Stats;
  };

  struct RequestStats {
    unsigned RequestsWithoutErrors = 0;
    unsigned RequestsWithErrors = 0;
    bool LastContentsHadErrors = false;
    bool FileIsRemoved = true;
  };

  std::vector<RequestStats> ReqStats;
  ReqStats.reserve(FilesCount);
  for (unsigned FileIndex = 0; FileIndex < FilesCount; ++FileIndex)
    ReqStats.emplace_back();

  TestDiagConsumer DiagConsumer;
  {
    MockCompilationDatabase CDB;
    ClangdServer Server(CDB, FS, ClangdServer::optsForTest(), &DiagConsumer);

    // Prepare some random distributions for the test.
    std::random_device RandGen;

    std::uniform_int_distribution<unsigned> FileIndexDist(0, FilesCount - 1);
    // Pass a text that contains compiler errors to addDocument in about 20% of
    // all requests.
    std::bernoulli_distribution ShouldHaveErrorsDist(0.2);
    // Line and Column numbers for requests that need them.
    std::uniform_int_distribution<int> LineDist(0, MaxLineForFileRequests);
    std::uniform_int_distribution<int> ColumnDist(0, MaxColumnForFileRequests);

    // Some helpers.
    auto UpdateStatsOnAddDocument = [&](unsigned FileIndex, bool HadErrors) {
      auto &Stats = ReqStats[FileIndex];

      if (HadErrors)
        ++Stats.RequestsWithErrors;
      else
        ++Stats.RequestsWithoutErrors;
      Stats.LastContentsHadErrors = HadErrors;
      Stats.FileIsRemoved = false;
    };

    auto UpdateStatsOnRemoveDocument = [&](unsigned FileIndex) {
      auto &Stats = ReqStats[FileIndex];

      Stats.FileIsRemoved = true;
    };

    auto AddDocument = [&](unsigned FileIndex, bool SkipCache) {
      bool ShouldHaveErrors = ShouldHaveErrorsDist(RandGen);
      Server.addDocument(FilePaths[FileIndex],
                         ShouldHaveErrors ? SourceContentsWithErrors
                                          : SourceContentsWithoutErrors);
      UpdateStatsOnAddDocument(FileIndex, ShouldHaveErrors);
    };

    // Various requests that we would randomly run.
    auto AddDocumentRequest = [&]() {
      unsigned FileIndex = FileIndexDist(RandGen);
      AddDocument(FileIndex, /*SkipCache=*/false);
    };

    auto ForceReparseRequest = [&]() {
      unsigned FileIndex = FileIndexDist(RandGen);
      AddDocument(FileIndex, /*SkipCache=*/true);
    };

    auto RemoveDocumentRequest = [&]() {
      unsigned FileIndex = FileIndexDist(RandGen);
      // Make sure we don't violate the ClangdServer's contract.
      if (ReqStats[FileIndex].FileIsRemoved)
        AddDocument(FileIndex, /*SkipCache=*/false);

      Server.removeDocument(FilePaths[FileIndex]);
      UpdateStatsOnRemoveDocument(FileIndex);
    };

    auto CodeCompletionRequest = [&]() {
      unsigned FileIndex = FileIndexDist(RandGen);
      // Make sure we don't violate the ClangdServer's contract.
      if (ReqStats[FileIndex].FileIsRemoved)
        AddDocument(FileIndex, /*SkipCache=*/false);

      Position Pos;
      Pos.line = LineDist(RandGen);
      Pos.character = ColumnDist(RandGen);
      // FIXME(ibiryukov): Also test async completion requests.
      // Simply putting CodeCompletion into async requests now would make
      // tests slow, since there's no way to cancel previous completion
      // requests as opposed to AddDocument/RemoveDocument, which are implicitly
      // cancelled by any subsequent AddDocument/RemoveDocument request to the
      // same file.
      cantFail(runCodeComplete(Server, FilePaths[FileIndex], Pos,
                               clangd::CodeCompleteOptions()));
    };

    auto LocateSymbolRequest = [&]() {
      unsigned FileIndex = FileIndexDist(RandGen);
      // Make sure we don't violate the ClangdServer's contract.
      if (ReqStats[FileIndex].FileIsRemoved)
        AddDocument(FileIndex, /*SkipCache=*/false);

      Position Pos;
      Pos.line = LineDist(RandGen);
      Pos.character = ColumnDist(RandGen);

      ASSERT_TRUE(!!runLocateSymbolAt(Server, FilePaths[FileIndex], Pos));
    };

    std::vector<std::function<void()>> AsyncRequests = {
        AddDocumentRequest, ForceReparseRequest, RemoveDocumentRequest};
    std::vector<std::function<void()>> BlockingRequests = {
        CodeCompletionRequest, LocateSymbolRequest};

    // Bash requests to ClangdServer in a loop.
    std::uniform_int_distribution<int> AsyncRequestIndexDist(
        0, AsyncRequests.size() - 1);
    std::uniform_int_distribution<int> BlockingRequestIndexDist(
        0, BlockingRequests.size() - 1);
    for (unsigned I = 1; I <= RequestsCount; ++I) {
      if (I % BlockingRequestInterval != 0) {
        // Issue an async request most of the time. It should be fast.
        unsigned RequestIndex = AsyncRequestIndexDist(RandGen);
        AsyncRequests[RequestIndex]();
      } else {
        // Issue a blocking request once in a while.
        auto RequestIndex = BlockingRequestIndexDist(RandGen);
        BlockingRequests[RequestIndex]();
      }
    }
    ASSERT_TRUE(Server.blockUntilIdleForTest());
  }

  // Check some invariants about the state of the program.
  std::vector<FileStat> Stats = DiagConsumer.takeFileStats();
  for (unsigned I = 0; I < FilesCount; ++I) {
    if (!ReqStats[I].FileIsRemoved) {
      ASSERT_EQ(Stats[I].HadErrorsInLastDiags,
                ReqStats[I].LastContentsHadErrors);
    }

    ASSERT_LE(Stats[I].HitsWithErrors, ReqStats[I].RequestsWithErrors);
    ASSERT_LE(Stats[I].HitsWithoutErrors, ReqStats[I].RequestsWithoutErrors);
  }
}

TEST(ClangdThreadingTest, NoConcurrentDiagnostics) {
  class NoConcurrentAccessDiagConsumer : public ClangdServer::Callbacks {
  public:
    std::atomic<int> Count = {0};

    NoConcurrentAccessDiagConsumer(std::promise<void> StartSecondReparse)
        : StartSecondReparse(std::move(StartSecondReparse)) {}

    void onDiagnosticsReady(PathRef, llvm::StringRef,
                            std::vector<Diag>) override {
      ++Count;
      std::unique_lock<std::mutex> Lock(Mutex, std::try_to_lock_t());
      ASSERT_TRUE(Lock.owns_lock())
          << "Detected concurrent onDiagnosticsReady calls for the same file.";

      // If we started the second parse immediately, it might cancel the first.
      // So we don't allow it to start until the first has delivered diags...
      if (FirstRequest) {
        FirstRequest = false;
        StartSecondReparse.set_value();
        // ... but then we wait long enough that the callbacks would overlap.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    }

  private:
    std::mutex Mutex;
    bool FirstRequest = true;
    std::promise<void> StartSecondReparse;
  };

  const auto SourceContentsWithoutErrors = R"cpp(
int a;
int b;
int c;
int d;
)cpp";

  const auto SourceContentsWithErrors = R"cpp(
int a = x;
int b;
int c;
int d;
)cpp";

  auto FooCpp = testPath("foo.cpp");
  MockFS FS;
  FS.Files[FooCpp] = "";

  std::promise<void> StartSecondPromise;
  std::future<void> StartSecond = StartSecondPromise.get_future();

  NoConcurrentAccessDiagConsumer DiagConsumer(std::move(StartSecondPromise));
  MockCompilationDatabase CDB;
  ClangdServer Server(CDB, FS, ClangdServer::optsForTest(), &DiagConsumer);
  Server.addDocument(FooCpp, SourceContentsWithErrors);
  StartSecond.wait();
  Server.addDocument(FooCpp, SourceContentsWithoutErrors);
  ASSERT_TRUE(Server.blockUntilIdleForTest()) << "Waiting for diagnostics";
  ASSERT_EQ(DiagConsumer.Count, 2); // Sanity check - we actually ran both?
}

TEST(ClangdServerTest, FormatCode) {
  MockFS FS;
  ErrorCheckingCallbacks DiagConsumer;
  MockCompilationDatabase CDB;
  ClangdServer Server(CDB, FS, ClangdServer::optsForTest(), &DiagConsumer);

  auto Path = testPath("foo.cpp");
  std::string Code = R"cpp(
#include "x.h"
#include "y.h"

void f(  )  {}
)cpp";
  std::string Expected = R"cpp(
#include "x.h"
#include "y.h"

void f() {}
)cpp";
  FS.Files[Path] = Code;
  runAddDocument(Server, Path, Code);

  auto Replaces = runFormatFile(Server, Path, /*Rng=*/llvm::None);
  EXPECT_TRUE(static_cast<bool>(Replaces));
  auto Changed = tooling::applyAllReplacements(Code, *Replaces);
  EXPECT_TRUE(static_cast<bool>(Changed));
  EXPECT_EQ(Expected, *Changed);
}

TEST(ClangdServerTest, ChangedHeaderFromISystem) {
  MockFS FS;
  ErrorCheckingCallbacks DiagConsumer;
  MockCompilationDatabase CDB;
  ClangdServer Server(CDB, FS, ClangdServer::optsForTest(), &DiagConsumer);

  auto SourcePath = testPath("source/foo.cpp");
  auto HeaderPath = testPath("headers/foo.h");
  FS.Files[HeaderPath] = "struct X { int bar; };";
  Annotations Code(R"cpp(
    #include "foo.h"

    int main() {
      X().ba^
    })cpp");
  CDB.ExtraClangFlags.push_back("-xc++");
  CDB.ExtraClangFlags.push_back("-isystem" + testPath("headers"));

  runAddDocument(Server, SourcePath, Code.code());
  auto Completions = cantFail(runCodeComplete(Server, SourcePath, Code.point(),
                                              clangd::CodeCompleteOptions()))
                         .Completions;
  EXPECT_THAT(Completions, ElementsAre(Field(&CodeCompletion::Name, "bar")));
  // Update the header and rerun addDocument to make sure we get the updated
  // files.
  FS.Files[HeaderPath] = "struct X { int bar; int baz; };";
  runAddDocument(Server, SourcePath, Code.code());
  Completions = cantFail(runCodeComplete(Server, SourcePath, Code.point(),
                                         clangd::CodeCompleteOptions()))
                    .Completions;
  // We want to make sure we see the updated version.
  EXPECT_THAT(Completions, ElementsAre(Field(&CodeCompletion::Name, "bar"),
                                       Field(&CodeCompletion::Name, "baz")));
}

// FIXME(ioeric): make this work for windows again.
#ifndef _WIN32
// Check that running code completion doesn't stat() a bunch of files from the
// preamble again. (They should be using the preamble's stat-cache)
TEST(ClangdTests, PreambleVFSStatCache) {
  class StatRecordingFS : public ThreadsafeFS {
    llvm::StringMap<unsigned> &CountStats;

  public:
    // If relative paths are used, they are resolved with testPath().
    llvm::StringMap<std::string> Files;

    StatRecordingFS(llvm::StringMap<unsigned> &CountStats)
        : CountStats(CountStats) {}

  private:
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> viewImpl() const override {
      class StatRecordingVFS : public llvm::vfs::ProxyFileSystem {
      public:
        StatRecordingVFS(IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS,
                         llvm::StringMap<unsigned> &CountStats)
            : ProxyFileSystem(std::move(FS)), CountStats(CountStats) {}

        llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>>
        openFileForRead(const Twine &Path) override {
          ++CountStats[llvm::sys::path::filename(Path.str())];
          return ProxyFileSystem::openFileForRead(Path);
        }
        llvm::ErrorOr<llvm::vfs::Status> status(const Twine &Path) override {
          ++CountStats[llvm::sys::path::filename(Path.str())];
          return ProxyFileSystem::status(Path);
        }

      private:
        llvm::StringMap<unsigned> &CountStats;
      };

      return IntrusiveRefCntPtr<StatRecordingVFS>(
          new StatRecordingVFS(buildTestFS(Files), CountStats));
    }
  };

  llvm::StringMap<unsigned> CountStats;
  StatRecordingFS FS(CountStats);
  ErrorCheckingCallbacks DiagConsumer;
  MockCompilationDatabase CDB;
  ClangdServer Server(CDB, FS, ClangdServer::optsForTest(), &DiagConsumer);

  auto SourcePath = testPath("foo.cpp");
  auto HeaderPath = testPath("foo.h");
  FS.Files[HeaderPath] = "struct TestSym {};";
  Annotations Code(R"cpp(
    #include "foo.h"

    int main() {
      TestSy^
    })cpp");

  runAddDocument(Server, SourcePath, Code.code());

  unsigned Before = CountStats["foo.h"];
  EXPECT_GT(Before, 0u);
  auto Completions = cantFail(runCodeComplete(Server, SourcePath, Code.point(),
                                              clangd::CodeCompleteOptions()))
                         .Completions;
  EXPECT_EQ(CountStats["foo.h"], Before);
  EXPECT_THAT(Completions,
              ElementsAre(Field(&CodeCompletion::Name, "TestSym")));
}
#endif

TEST(ClangdServerTest, FallbackWhenPreambleIsNotReady) {
  MockFS FS;
  ErrorCheckingCallbacks DiagConsumer;
  MockCompilationDatabase CDB;
  ClangdServer Server(CDB, FS, ClangdServer::optsForTest(), &DiagConsumer);

  auto FooCpp = testPath("foo.cpp");
  Annotations Code(R"cpp(
    namespace ns { int xyz; }
    using namespace ns;
    int main() {
       xy^
    })cpp");
  FS.Files[FooCpp] = FooCpp;

  auto Opts = clangd::CodeCompleteOptions();
  Opts.RunParser = CodeCompleteOptions::ParseIfReady;

  // This will make compile command broken and preamble absent.
  CDB.ExtraClangFlags = {"-###"};
  Server.addDocument(FooCpp, Code.code());
  ASSERT_TRUE(Server.blockUntilIdleForTest());
  auto Res = cantFail(runCodeComplete(Server, FooCpp, Code.point(), Opts));
  EXPECT_EQ(Res.Context, CodeCompletionContext::CCC_Recovery);
  // Identifier-based fallback completion doesn't know about "symbol" scope.
  EXPECT_THAT(Res.Completions,
              ElementsAre(AllOf(Field(&CodeCompletion::Name, "xyz"),
                                Field(&CodeCompletion::Scope, ""))));

  // Make the compile command work again.
  CDB.ExtraClangFlags = {"-std=c++11"};
  Server.addDocument(FooCpp, Code.code());
  ASSERT_TRUE(Server.blockUntilIdleForTest());
  EXPECT_THAT(
      cantFail(runCodeComplete(Server, FooCpp, Code.point(), Opts)).Completions,
      ElementsAre(AllOf(Field(&CodeCompletion::Name, "xyz"),
                        Field(&CodeCompletion::Scope, "ns::"))));

  // Now force identifier-based completion.
  Opts.RunParser = CodeCompleteOptions::NeverParse;
  EXPECT_THAT(
      cantFail(runCodeComplete(Server, FooCpp, Code.point(), Opts)).Completions,
      ElementsAre(AllOf(Field(&CodeCompletion::Name, "xyz"),
                        Field(&CodeCompletion::Scope, ""))));
}

TEST(ClangdServerTest, FallbackWhenWaitingForCompileCommand) {
  MockFS FS;
  ErrorCheckingCallbacks DiagConsumer;
  // Returns compile command only when notified.
  class DelayedCompilationDatabase : public GlobalCompilationDatabase {
  public:
    DelayedCompilationDatabase(Notification &CanReturnCommand)
        : CanReturnCommand(CanReturnCommand) {}

    llvm::Optional<tooling::CompileCommand>
    getCompileCommand(PathRef File) const override {
      // FIXME: make this timeout and fail instead of waiting forever in case
      // something goes wrong.
      CanReturnCommand.wait();
      auto FileName = llvm::sys::path::filename(File);
      std::vector<std::string> CommandLine = {"clangd", "-ffreestanding",
                                              std::string(File)};
      return {tooling::CompileCommand(llvm::sys::path::parent_path(File),
                                      FileName, std::move(CommandLine), "")};
    }

    std::vector<std::string> ExtraClangFlags;

  private:
    Notification &CanReturnCommand;
  };

  Notification CanReturnCommand;
  DelayedCompilationDatabase CDB(CanReturnCommand);
  ClangdServer Server(CDB, FS, ClangdServer::optsForTest(), &DiagConsumer);

  auto FooCpp = testPath("foo.cpp");
  Annotations Code(R"cpp(
    namespace ns { int xyz; }
    using namespace ns;
    int main() {
       xy^
    })cpp");
  FS.Files[FooCpp] = FooCpp;
  Server.addDocument(FooCpp, Code.code());

  // Sleep for some time to make sure code completion is not run because update
  // hasn't been scheduled.
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  auto Opts = clangd::CodeCompleteOptions();
  Opts.RunParser = CodeCompleteOptions::ParseIfReady;

  auto Res = cantFail(runCodeComplete(Server, FooCpp, Code.point(), Opts));
  EXPECT_EQ(Res.Context, CodeCompletionContext::CCC_Recovery);

  CanReturnCommand.notify();
  ASSERT_TRUE(Server.blockUntilIdleForTest());
  EXPECT_THAT(cantFail(runCodeComplete(Server, FooCpp, Code.point(),
                                       clangd::CodeCompleteOptions()))
                  .Completions,
              ElementsAre(AllOf(Field(&CodeCompletion::Name, "xyz"),
                                Field(&CodeCompletion::Scope, "ns::"))));
}

TEST(ClangdServerTest, CustomAction) {
  OverlayCDB CDB(/*Base=*/nullptr);
  MockFS FS;
  ClangdServer Server(CDB, FS, ClangdServer::optsForTest());

  Server.addDocument(testPath("foo.cc"), "void x();");
  Decl::Kind XKind = Decl::TranslationUnit;
  EXPECT_THAT_ERROR(runCustomAction(Server, testPath("foo.cc"),
                                    [&](InputsAndAST AST) {
                                      XKind = findDecl(AST.AST, "x").getKind();
                                    }),
                    llvm::Succeeded());
  EXPECT_EQ(XKind, Decl::Function);
}

// Tests fails when built with asan due to stack overflow. So skip running the
// test as a workaround.
#if !defined(__has_feature) || !__has_feature(address_sanitizer)
TEST(ClangdServerTest, TestStackOverflow) {
  MockFS FS;
  ErrorCheckingCallbacks DiagConsumer;
  MockCompilationDatabase CDB;
  ClangdServer Server(CDB, FS, ClangdServer::optsForTest(), &DiagConsumer);

  const char *SourceContents = R"cpp(
    constexpr int foo() { return foo(); }
    static_assert(foo());
  )cpp";

  auto FooCpp = testPath("foo.cpp");
  FS.Files[FooCpp] = SourceContents;

  Server.addDocument(FooCpp, SourceContents);
  ASSERT_TRUE(Server.blockUntilIdleForTest()) << "Waiting for diagnostics";
  // check that we got a constexpr depth error, and not crashed by stack
  // overflow
  EXPECT_TRUE(DiagConsumer.hadErrorInLastDiags());
}
#endif

TEST(ClangdServer, TidyOverrideTest) {
  struct DiagsCheckingCallback : public ClangdServer::Callbacks {
  public:
    void onDiagnosticsReady(PathRef File, llvm::StringRef Version,
                            std::vector<Diag> Diagnostics) override {
      std::lock_guard<std::mutex> Lock(Mutex);
      HadDiagsInLastCallback = !Diagnostics.empty();
    }

    std::mutex Mutex;
    bool HadDiagsInLastCallback = false;
  } DiagConsumer;

  MockFS FS;
  // These checks don't work well in clangd, even if configured they shouldn't
  // run.
  FS.Files[testPath(".clang-tidy")] = R"(
    Checks: -*,bugprone-use-after-move,llvm-header-guard
  )";
  MockCompilationDatabase CDB;
  std::vector<TidyProvider> Stack;
  Stack.push_back(provideClangTidyFiles(FS));
  Stack.push_back(disableUnusableChecks());
  TidyProvider Provider = combine(std::move(Stack));
  CDB.ExtraClangFlags = {"-xc++"};
  auto Opts = ClangdServer::optsForTest();
  Opts.ClangTidyProvider = Provider;
  ClangdServer Server(CDB, FS, Opts, &DiagConsumer);
  const char *SourceContents = R"cpp(
    struct Foo { Foo(); Foo(Foo&); Foo(Foo&&); };
    namespace std { Foo&& move(Foo&); }
    void foo() {
      Foo x;
      Foo y = std::move(x);
      Foo z = x;
    })cpp";
  Server.addDocument(testPath("foo.h"), SourceContents);
  ASSERT_TRUE(Server.blockUntilIdleForTest());
  EXPECT_FALSE(DiagConsumer.HadDiagsInLastCallback);
}

TEST(ClangdServer, MemoryUsageTest) {
  MockFS FS;
  MockCompilationDatabase CDB;
  ClangdServer Server(CDB, FS, ClangdServer::optsForTest());

  auto FooCpp = testPath("foo.cpp");
  Server.addDocument(FooCpp, "");
  ASSERT_TRUE(Server.blockUntilIdleForTest());

  llvm::BumpPtrAllocator Alloc;
  MemoryTree MT(&Alloc);
  Server.profile(MT);
  ASSERT_TRUE(MT.children().count("tuscheduler"));
  EXPECT_TRUE(MT.child("tuscheduler").children().count(FooCpp));
}

TEST(ClangdServer, RespectsTweakFormatting) {
  static constexpr const char *TweakID = "ModuleTweak";
  static constexpr const char *NewContents = "{not;\nformatted;}";

  // Contributes a tweak that generates a non-formatted insertion and disables
  // formatting.
  struct TweakContributingModule final : public FeatureModule {
    struct ModuleTweak final : public Tweak {
      const char *id() const override { return TweakID; }
      bool prepare(const Selection &Sel) override { return true; }
      Expected<Effect> apply(const Selection &Sel) override {
        auto &SM = Sel.AST->getSourceManager();
        llvm::StringRef FilePath = SM.getFilename(Sel.Cursor);
        tooling::Replacements Reps;
        llvm::cantFail(
            Reps.add(tooling::Replacement(FilePath, 0, 0, NewContents)));
        auto E = llvm::cantFail(Effect::mainFileEdit(SM, std::move(Reps)));
        E.FormatEdits = false;
        return E;
      }
      std::string title() const override { return id(); }
      llvm::StringLiteral kind() const override {
        return llvm::StringLiteral("");
      };
    };

    void contributeTweaks(std::vector<std::unique_ptr<Tweak>> &Out) override {
      Out.emplace_back(new ModuleTweak);
    }
  };

  MockFS FS;
  MockCompilationDatabase CDB;
  auto Opts = ClangdServer::optsForTest();
  FeatureModuleSet Set;
  Set.add(std::make_unique<TweakContributingModule>());
  Opts.FeatureModules = &Set;
  ClangdServer Server(CDB, FS, Opts);

  auto FooCpp = testPath("foo.cpp");
  Server.addDocument(FooCpp, "");
  ASSERT_TRUE(Server.blockUntilIdleForTest());

  // Ensure that disabled formatting is respected.
  Notification N;
  Server.applyTweak(FooCpp, {}, TweakID, [&](llvm::Expected<Tweak::Effect> E) {
    ASSERT_TRUE(static_cast<bool>(E));
    EXPECT_THAT(llvm::cantFail(E->ApplyEdits.lookup(FooCpp).apply()),
                NewContents);
    N.notify();
  });
  N.wait();
}
} // namespace
} // namespace clangd
} // namespace clang
