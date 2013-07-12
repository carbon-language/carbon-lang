//===- cpp11-migrate/TransformTest.cpp - Transform unit tests -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "Core/FileOverrides.h"
#include "Core/Transform.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclGroup.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Path.h"

using namespace clang;
using namespace ast_matchers;

class DummyTransform : public Transform {
public:
  DummyTransform(llvm::StringRef Name, const TransformOptions &Options)
      : Transform(Name, Options) {}

  virtual int apply(FileOverrides &,
                    const tooling::CompilationDatabase &,
                    const std::vector<std::string> &) { return 0; }

  void setAcceptedChanges(unsigned Changes) {
    Transform::setAcceptedChanges(Changes);
  }
  void setRejectedChanges(unsigned Changes) {
    Transform::setRejectedChanges(Changes);
  }
  void setDeferredChanges(unsigned Changes) {
    Transform::setDeferredChanges(Changes);
  }

  void setOverrides(FileOverrides &Overrides) {
    Transform::setOverrides(Overrides);
  }

};

TEST(Transform, Interface) {
  TransformOptions Options;
  DummyTransform T("my_transform", Options);

  ASSERT_EQ("my_transform", T.getName());
  ASSERT_EQ(0u, T.getAcceptedChanges());
  ASSERT_EQ(0u, T.getRejectedChanges());
  ASSERT_EQ(0u, T.getDeferredChanges());
  ASSERT_FALSE(T.getChangesMade());
  ASSERT_FALSE(T.getChangesNotMade());

  T.setAcceptedChanges(1);
  ASSERT_TRUE(T.getChangesMade());

  T.setDeferredChanges(1);
  ASSERT_TRUE(T.getChangesNotMade());

  T.setRejectedChanges(1);
  ASSERT_TRUE(T.getChangesNotMade());

  T.Reset();
  ASSERT_EQ(0u, T.getAcceptedChanges());
  ASSERT_EQ(0u, T.getRejectedChanges());
  ASSERT_EQ(0u, T.getDeferredChanges());

  T.setRejectedChanges(1);
  ASSERT_TRUE(T.getChangesNotMade());
}

class TimePassingASTConsumer : public ASTConsumer {
public:
  TimePassingASTConsumer(bool *Called) : Called(Called) {}

  virtual bool HandleTopLevelDecl(DeclGroupRef DeclGroup) {
    llvm::sys::TimeValue UserStart;
    llvm::sys::TimeValue SystemStart;
    llvm::sys::TimeValue UserNow;
    llvm::sys::TimeValue SystemNow;
    llvm::sys::TimeValue Wall;

    // Busy-wait until the user/system time combined is more than 1ms
    llvm::sys::TimeValue OneMS(0, 1000000);
    llvm::sys::Process::GetTimeUsage(Wall, UserStart, SystemStart);
    do {
      llvm::sys::Process::GetTimeUsage(Wall, UserNow, SystemNow);
    } while (UserNow - UserStart + SystemNow - SystemStart < OneMS);
    *Called = true;
    return true;
  }
  bool *Called;
};

struct ConsumerFactory {
  ASTConsumer *newASTConsumer() {
    return new TimePassingASTConsumer(&Called);
  }
  bool Called;
};

struct CallbackForwarder : public clang::tooling::SourceFileCallbacks {
  CallbackForwarder(Transform &Callee) : Callee(Callee) {}

  virtual bool handleBeginSource(CompilerInstance &CI, StringRef Filename) {
    return Callee.handleBeginSource(CI, Filename);
  }

  virtual void handleEndSource() {
    Callee.handleEndSource();
  }

  Transform &Callee;
};

TEST(Transform, Timings) {
  TransformOptions Options;
  Options.EnableTiming = true;
  DummyTransform T("timing_transform", Options);

  // All the path stuff is to make the test work independently of OS.

  // The directory used is not important since the path gets mapped to a virtual
  // file anyway. What is important is that we have an absolute path with which
  // to use with mapVirtualFile().
  SmallString<128> CurrentDir;
  llvm::error_code EC = llvm::sys::fs::current_path(CurrentDir);
  assert(!EC);
  (void)EC;

  SmallString<128> FileA = CurrentDir;
  llvm::sys::path::append(FileA, "a.cc");

  SmallString<128> FileB = CurrentDir;
  llvm::sys::path::append(FileB, "b.cc");

  tooling::FixedCompilationDatabase Compilations(CurrentDir.str(),
                                                 std::vector<std::string>());
  std::vector<std::string> Sources;
  Sources.push_back(FileA.str());
  Sources.push_back(FileB.str());
  tooling::ClangTool Tool(Compilations, Sources);

  Tool.mapVirtualFile(FileA, "void a() {}");
  Tool.mapVirtualFile(FileB, "void b() {}");

  // Factory to create TimePassingASTConsumer for each source file the tool
  // runs on.
  ConsumerFactory Factory;

  // We don't care about any of Transform's functionality except to get it to
  // record timings. For that, we need to forward handleBeginSource() and
  // handleEndSource() calls to it.
  CallbackForwarder Callbacks(T);

  // Transform's handle* functions require FileOverrides to be set, even if
  // there aren't any.
  FileOverrides Overrides;
  T.setOverrides(Overrides);

  Tool.run(clang::tooling::newFrontendActionFactory(&Factory, &Callbacks));

  EXPECT_TRUE(Factory.Called);
  Transform::TimingVec::const_iterator I = T.timing_begin();
  EXPECT_GT(I->second.getProcessTime(), 0.0);

  // The success of the test shouldn't depend on the order of iteration through
  // timers.
  StringRef FirstFile = I->first;
  if (FileA == FirstFile) {
    ++I;
    EXPECT_EQ(FileB, I->first);
    EXPECT_GT(I->second.getProcessTime(), 0.0);
  } else if (FileB == FirstFile) {
    ++I;
    EXPECT_EQ(FileA, I->first);
    EXPECT_GT(I->second.getProcessTime(), 0.0);
  } else {
    FAIL() << "Unexpected file name " << I->first << " in timing data.";
  }
  ++I;
  EXPECT_EQ(T.timing_end(), I);
}

class ModifiableCallback
    : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  ModifiableCallback(const Transform &Owner, bool HeadersModifiable)
      : Owner(Owner), HeadersModifiable(HeadersModifiable) {}

  virtual void
  run(const clang::ast_matchers::MatchFinder::MatchResult &Result) {
    const VarDecl *Decl = Result.Nodes.getNodeAs<VarDecl>("decl");
    ASSERT_TRUE(Decl != 0);

    const SourceManager &SM = *Result.SourceManager;

    // Decl 'a' comes from the main source file. This test should always pass.
    if (Decl->getName().equals("a"))
      EXPECT_TRUE(Owner.isFileModifiable(SM, Decl->getLocStart()));

    // Decl 'c' comes from an excluded header. This test should never pass.
    else if (Decl->getName().equals("c"))
      EXPECT_FALSE(Owner.isFileModifiable(SM, Decl->getLocStart()));

    // Decl 'b' comes from an included header. It should be modifiable only if
    // header modifications are allowed.
    else if (Decl->getName().equals("b"))
      EXPECT_EQ(HeadersModifiable,
                Owner.isFileModifiable(SM, Decl->getLocStart()));

    // Make sure edge cases are handled gracefully (they should never be
    // allowed).
    SourceLocation DummyLoc;
    EXPECT_FALSE(Owner.isFileModifiable(SM, DummyLoc));
  }

private:
  const Transform &Owner;
  bool HeadersModifiable;
};

TEST(Transform, isFileModifiable) {
  TransformOptions Options;

  ///
  /// SETUP
  ///
  /// To test Transform::isFileModifiable() we need a SourceManager primed with
  /// actual files and SourceLocations to test. Easiest way to accomplish this
  /// is to use Tooling classes.
  ///
  /// 1) Simulate a source file that includes two headers, one that is allowed
  ///    to be modified and the other that is not allowed. Each of the three
  ///    files involved will declare a single variable with a different name. 
  /// 2) A matcher is created to find VarDecls.
  /// 3) A MatchFinder callback calls Transform::isFileModifiable() with the
  ///    SourceLocations of found VarDecls and thus tests the function.
  ///

  // All the path stuff is to make the test work independently of OS.

  // The directory used is not important since the path gets mapped to a virtual
  // file anyway. What is important is that we have an absolute path with which
  // to use with mapVirtualFile().
  SmallString<128> CurrentDir;
  llvm::error_code EC = llvm::sys::fs::current_path(CurrentDir);
  assert(!EC);
  (void)EC;

  SmallString<128> SourceFile = CurrentDir;
  llvm::sys::path::append(SourceFile, "a.cc");

  SmallString<128> HeaderFile = CurrentDir;
  llvm::sys::path::append(HeaderFile, "a.h");

  SmallString<128> HeaderBFile = CurrentDir;
  llvm::sys::path::append(HeaderBFile, "temp");
  llvm::sys::path::append(HeaderBFile, "b.h");

  StringRef ExcludeDir = llvm::sys::path::parent_path(HeaderBFile);

  IncludeExcludeInfo IncInfo;
  Options.ModifiableHeaders.readListFromString(CurrentDir, ExcludeDir);

  tooling::FixedCompilationDatabase Compilations(CurrentDir.str(),
                                                 std::vector<std::string>());
  std::vector<std::string> Sources;
  Sources.push_back(SourceFile.str());
  tooling::ClangTool Tool(Compilations, Sources);

  Tool.mapVirtualFile(SourceFile,
                      "#include \"a.h\"\n"
                      "#include \"temp/b.h\"\n"
                      "int a;");
  Tool.mapVirtualFile(HeaderFile, "int b;");
  Tool.mapVirtualFile(HeaderBFile, "int c;");

  // Run tests with header modifications turned off.
  {
    SCOPED_TRACE("Header Modifications are OFF");
    Options.EnableHeaderModifications = false;
    DummyTransform T("dummy", Options);
    MatchFinder Finder;
    Finder.addMatcher(varDecl().bind("decl"), new ModifiableCallback(T, false));
    Tool.run(tooling::newFrontendActionFactory(&Finder));
  }

  // Run again with header modifications turned on.
  {
    SCOPED_TRACE("Header Modifications are ON");
    Options.EnableHeaderModifications = true;
    DummyTransform T("dummy", Options);
    MatchFinder Finder;
    Finder.addMatcher(varDecl().bind("decl"), new ModifiableCallback(T, true));
    Tool.run(tooling::newFrontendActionFactory(&Finder));
  }
}
