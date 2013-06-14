#include "gtest/gtest.h"
#include "Core/Transform.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclGroup.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PathV1.h"

using namespace clang;

class DummyTransform : public Transform {
public:
  DummyTransform(llvm::StringRef Name, const TransformOptions &Options)
      : Transform(Name, Options) {}

  virtual int apply(const FileOverrides &,
                    const tooling::CompilationDatabase &,
                    const std::vector<std::string> &,
                    FileOverrides &) { return 0; }

  void setAcceptedChanges(unsigned Changes) {
    Transform::setAcceptedChanges(Changes);
  }
  void setRejectedChanges(unsigned Changes) {
    Transform::setRejectedChanges(Changes);
  }
  void setDeferredChanges(unsigned Changes) {
    Transform::setDeferredChanges(Changes);
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

class FindTopLevelDeclConsumer : public ASTConsumer {
public:
  FindTopLevelDeclConsumer(bool *Called) : Called(Called) {}

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
    return new FindTopLevelDeclConsumer(&Called);
  }
  bool Called;
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
  EXPECT_FALSE(EC);

  SmallString<128> FileA = CurrentDir;
  llvm::sys::path::append(FileA, "a.cc");

  SmallString<128> FileB = CurrentDir;
  llvm::sys::path::append(FileB, "b.cc");

  tooling::FixedCompilationDatabase Compilations(CurrentDir.str(), std::vector<std::string>());
  std::vector<std::string> Sources;
  Sources.push_back(FileA.str());
  Sources.push_back(FileB.str());
  tooling::ClangTool Tool(Compilations, Sources);

  Tool.mapVirtualFile(FileA, "void a() {}");
  Tool.mapVirtualFile(FileB, "void b() {}");

  ConsumerFactory Factory;
  Tool.run(newFrontendActionFactory(&Factory, &T));

  EXPECT_TRUE(Factory.Called);
  Transform::TimingVec::const_iterator I = T.timing_begin();
  EXPECT_GT(I->second.getProcessTime(), 0.0);

  // The success of the test shouldn't depend on the order of iteration through
  // timers.
  llvm::sys::Path FirstFile(I->first);
  if (FileA == FirstFile.str()) {
    ++I;
    EXPECT_EQ(FileB, llvm::sys::Path(I->first).str());
    EXPECT_GT(I->second.getProcessTime(), 0.0);
  } else if (FileB == FirstFile.str()) {
    ++I;
    EXPECT_EQ(FileA, llvm::sys::Path(I->first).str());
    EXPECT_GT(I->second.getProcessTime(), 0.0);
  } else {
    FAIL() << "Unexpected file name " << I->first << " in timing data.";
  }
  ++I;
  EXPECT_EQ(T.timing_end(), I);
}
