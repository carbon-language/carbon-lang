//===- unittest/Tooling/ToolingTest.cpp - Tooling unit tests --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"
#include "llvm/ADT/STLExtras.h"
#include <string>

namespace clang {
namespace tooling {

namespace {
/// Takes an ast consumer and returns it from CreateASTConsumer. This only
/// works with single translation unit compilations.
class TestAction : public clang::ASTFrontendAction {
 public:
  /// Takes ownership of TestConsumer.
  explicit TestAction(clang::ASTConsumer *TestConsumer)
      : TestConsumer(TestConsumer) {}

 protected:
  virtual clang::ASTConsumer* CreateASTConsumer(
      clang::CompilerInstance& compiler, StringRef dummy) {
    /// TestConsumer will be deleted by the framework calling us.
    return TestConsumer;
  }

 private:
  clang::ASTConsumer * const TestConsumer;
};

class FindTopLevelDeclConsumer : public clang::ASTConsumer {
 public:
  explicit FindTopLevelDeclConsumer(bool *FoundTopLevelDecl)
      : FoundTopLevelDecl(FoundTopLevelDecl) {}
  virtual bool HandleTopLevelDecl(clang::DeclGroupRef DeclGroup) {
    *FoundTopLevelDecl = true;
    return true;
  }
 private:
  bool * const FoundTopLevelDecl;
};
} // end namespace

TEST(runToolOnCode, FindsNoTopLevelDeclOnEmptyCode) {
  bool FoundTopLevelDecl = false;
  EXPECT_TRUE(runToolOnCode(
      new TestAction(new FindTopLevelDeclConsumer(&FoundTopLevelDecl)), ""));
#if !defined(_MSC_VER)
  EXPECT_FALSE(FoundTopLevelDecl);
#else
  // FIXME: LangOpts.MicrosoftExt appends "class type_info;"
  EXPECT_TRUE(FoundTopLevelDecl);
#endif
}

namespace {
class FindClassDeclXConsumer : public clang::ASTConsumer {
 public:
  FindClassDeclXConsumer(bool *FoundClassDeclX)
      : FoundClassDeclX(FoundClassDeclX) {}
  virtual bool HandleTopLevelDecl(clang::DeclGroupRef GroupRef) {
    if (CXXRecordDecl* Record = dyn_cast<clang::CXXRecordDecl>(
            *GroupRef.begin())) {
      if (Record->getName() == "X") {
        *FoundClassDeclX = true;
      }
    }
    return true;
  }
 private:
  bool *FoundClassDeclX;
};
bool FindClassDeclX(ASTUnit *AST) {
  for (std::vector<Decl *>::iterator i = AST->top_level_begin(),
                                     e = AST->top_level_end();
       i != e; ++i) {
    if (CXXRecordDecl* Record = dyn_cast<clang::CXXRecordDecl>(*i)) {
      if (Record->getName() == "X") {
        return true;
      }
    }
  }
  return false;
}
} // end namespace

TEST(runToolOnCode, FindsClassDecl) {
  bool FoundClassDeclX = false;
  EXPECT_TRUE(runToolOnCode(new TestAction(
      new FindClassDeclXConsumer(&FoundClassDeclX)), "class X;"));
  EXPECT_TRUE(FoundClassDeclX);

  FoundClassDeclX = false;
  EXPECT_TRUE(runToolOnCode(new TestAction(
      new FindClassDeclXConsumer(&FoundClassDeclX)), "class Y;"));
  EXPECT_FALSE(FoundClassDeclX);
}

TEST(buildASTFromCode, FindsClassDecl) {
  OwningPtr<ASTUnit> AST(buildASTFromCode("class X;"));
  ASSERT_TRUE(AST.get());
  EXPECT_TRUE(FindClassDeclX(AST.get()));

  AST.reset(buildASTFromCode("class Y;"));
  ASSERT_TRUE(AST.get());
  EXPECT_FALSE(FindClassDeclX(AST.get()));
}

TEST(newFrontendActionFactory, CreatesFrontendActionFactoryFromType) {
  OwningPtr<FrontendActionFactory> Factory(
      newFrontendActionFactory<SyntaxOnlyAction>());
  OwningPtr<FrontendAction> Action(Factory->create());
  EXPECT_TRUE(Action.get() != NULL);
}

struct IndependentFrontendActionCreator {
  ASTConsumer *newASTConsumer() {
    return new FindTopLevelDeclConsumer(NULL);
  }
};

TEST(newFrontendActionFactory, CreatesFrontendActionFactoryFromFactoryType) {
  IndependentFrontendActionCreator Creator;
  OwningPtr<FrontendActionFactory> Factory(
      newFrontendActionFactory(&Creator));
  OwningPtr<FrontendAction> Action(Factory->create());
  EXPECT_TRUE(Action.get() != NULL);
}

TEST(ToolInvocation, TestMapVirtualFile) {
  IntrusiveRefCntPtr<clang::FileManager> Files(
      new clang::FileManager(clang::FileSystemOptions()));
  std::vector<std::string> Args;
  Args.push_back("tool-executable");
  Args.push_back("-Idef");
  Args.push_back("-fsyntax-only");
  Args.push_back("test.cpp");
  clang::tooling::ToolInvocation Invocation(Args, new SyntaxOnlyAction,
                                            Files.getPtr());
  Invocation.mapVirtualFile("test.cpp", "#include <abc>\n");
  Invocation.mapVirtualFile("def/abc", "\n");
  EXPECT_TRUE(Invocation.run());
}

TEST(ToolInvocation, TestVirtualModulesCompilation) {
  // FIXME: Currently, this only tests that we don't exit with an error if a
  // mapped module.map is found on the include path. In the future, expand this
  // test to run a full modules enabled compilation, so we make sure we can
  // rerun modules compilations with a virtual file system.
  IntrusiveRefCntPtr<clang::FileManager> Files(
      new clang::FileManager(clang::FileSystemOptions()));
  std::vector<std::string> Args;
  Args.push_back("tool-executable");
  Args.push_back("-Idef");
  Args.push_back("-fsyntax-only");
  Args.push_back("test.cpp");
  clang::tooling::ToolInvocation Invocation(Args, new SyntaxOnlyAction,
                                            Files.getPtr());
  Invocation.mapVirtualFile("test.cpp", "#include <abc>\n");
  Invocation.mapVirtualFile("def/abc", "\n");
  // Add a module.map file in the include directory of our header, so we trigger
  // the module.map header search logic.
  Invocation.mapVirtualFile("def/module.map", "\n");
  EXPECT_TRUE(Invocation.run());
}

struct VerifyEndCallback : public SourceFileCallbacks {
  VerifyEndCallback() : BeginCalled(0), EndCalled(0), Matched(false) {}
  virtual bool handleBeginSource(CompilerInstance &CI,
                                 StringRef Filename) LLVM_OVERRIDE {
    ++BeginCalled;
    return true;
  }
  virtual void handleEndSource() {
    ++EndCalled;
  }
  ASTConsumer *newASTConsumer() {
    return new FindTopLevelDeclConsumer(&Matched);
  }
  unsigned BeginCalled;
  unsigned EndCalled;
  bool Matched;
};

#if !defined(_WIN32)
TEST(newFrontendActionFactory, InjectsSourceFileCallbacks) {
  VerifyEndCallback EndCallback;

  FixedCompilationDatabase Compilations("/", std::vector<std::string>());
  std::vector<std::string> Sources;
  Sources.push_back("/a.cc");
  Sources.push_back("/b.cc");
  ClangTool Tool(Compilations, Sources);

  Tool.mapVirtualFile("/a.cc", "void a() {}");
  Tool.mapVirtualFile("/b.cc", "void b() {}");

  Tool.run(newFrontendActionFactory(&EndCallback, &EndCallback));

  EXPECT_TRUE(EndCallback.Matched);
  EXPECT_EQ(2u, EndCallback.BeginCalled);
  EXPECT_EQ(2u, EndCallback.EndCalled);
}
#endif

struct SkipBodyConsumer : public clang::ASTConsumer {
  /// Skip the 'skipMe' function.
  virtual bool shouldSkipFunctionBody(Decl *D) {
    FunctionDecl *F = dyn_cast<FunctionDecl>(D);
    return F && F->getNameAsString() == "skipMe";
  }
};

struct SkipBodyAction : public clang::ASTFrontendAction {
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &Compiler,
                                         StringRef) {
    Compiler.getFrontendOpts().SkipFunctionBodies = true;
    return new SkipBodyConsumer;
  }
};

TEST(runToolOnCode, TestSkipFunctionBody) {
  EXPECT_TRUE(runToolOnCode(new SkipBodyAction,
                            "int skipMe() { an_error_here }"));
  EXPECT_FALSE(runToolOnCode(new SkipBodyAction,
                             "int skipMeNot() { an_error_here }"));
}

struct CheckSyntaxOnlyAdjuster: public ArgumentsAdjuster {
  bool &Found;
  bool &Ran;

  CheckSyntaxOnlyAdjuster(bool &Found, bool &Ran) : Found(Found), Ran(Ran) { }

  virtual CommandLineArguments
  Adjust(const CommandLineArguments &Args) LLVM_OVERRIDE {
    Ran = true;
    for (unsigned I = 0, E = Args.size(); I != E; ++I) {
      if (Args[I] == "-fsyntax-only") {
        Found = true;
        break;
      }
    }
    return Args;
  }
};

TEST(ClangToolTest, ArgumentAdjusters) {
  FixedCompilationDatabase Compilations("/", std::vector<std::string>());

  ClangTool Tool(Compilations, std::vector<std::string>(1, "/a.cc"));
  Tool.mapVirtualFile("/a.cc", "void a() {}");

  bool Found = false;
  bool Ran = false;
  Tool.appendArgumentsAdjuster(new CheckSyntaxOnlyAdjuster(Found, Ran));
  Tool.run(newFrontendActionFactory<SyntaxOnlyAction>());
  EXPECT_TRUE(Ran);
  EXPECT_TRUE(Found);

  Ran = Found = false;
  Tool.clearArgumentsAdjusters();
  Tool.appendArgumentsAdjuster(new CheckSyntaxOnlyAdjuster(Found, Ran));
  Tool.appendArgumentsAdjuster(new ClangSyntaxOnlyAdjuster());
  Tool.run(newFrontendActionFactory<SyntaxOnlyAction>());
  EXPECT_TRUE(Ran);
  EXPECT_FALSE(Found);
}

#ifndef _WIN32
TEST(ClangToolTest, BuildASTs) {
  FixedCompilationDatabase Compilations("/", std::vector<std::string>());

  std::vector<std::string> Sources;
  Sources.push_back("/a.cc");
  Sources.push_back("/b.cc");
  ClangTool Tool(Compilations, Sources);

  Tool.mapVirtualFile("/a.cc", "void a() {}");
  Tool.mapVirtualFile("/b.cc", "void b() {}");

  std::vector<ASTUnit *> ASTs;
  EXPECT_EQ(0, Tool.buildASTs(ASTs));
  EXPECT_EQ(2u, ASTs.size());

  llvm::DeleteContainerPointers(ASTs);
}

struct TestDiagnosticConsumer : public DiagnosticConsumer {
  TestDiagnosticConsumer() : NumDiagnosticsSeen(0) {}
  virtual void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                                const Diagnostic &Info) {
    ++NumDiagnosticsSeen;
  }
  unsigned NumDiagnosticsSeen;
};

TEST(ClangToolTest, InjectDiagnosticConsumer) {
  FixedCompilationDatabase Compilations("/", std::vector<std::string>());
  ClangTool Tool(Compilations, std::vector<std::string>(1, "/a.cc"));
  Tool.mapVirtualFile("/a.cc", "int x = undeclared;");
  TestDiagnosticConsumer Consumer;
  Tool.setDiagnosticConsumer(&Consumer);
  Tool.run(newFrontendActionFactory<SyntaxOnlyAction>());
  EXPECT_EQ(1u, Consumer.NumDiagnosticsSeen);
}

TEST(ClangToolTest, InjectDiagnosticConsumerInBuildASTs) {
  FixedCompilationDatabase Compilations("/", std::vector<std::string>());
  ClangTool Tool(Compilations, std::vector<std::string>(1, "/a.cc"));
  Tool.mapVirtualFile("/a.cc", "int x = undeclared;");
  TestDiagnosticConsumer Consumer;
  Tool.setDiagnosticConsumer(&Consumer);
  std::vector<ASTUnit*> ASTs;
  Tool.buildASTs(ASTs);
  EXPECT_EQ(1u, ASTs.size());
  EXPECT_EQ(1u, Consumer.NumDiagnosticsSeen);
}
#endif

} // end namespace tooling
} // end namespace clang
