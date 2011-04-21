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
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

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
      clang::CompilerInstance& compiler, llvm::StringRef dummy) {
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
  virtual void HandleTopLevelDecl(clang::DeclGroupRef DeclGroup) {
    *FoundTopLevelDecl = true;
  }
 private:
  bool * const FoundTopLevelDecl;
};
} // end namespace

TEST(RunSyntaxOnlyToolOnCode, FindsTopLevelDeclOnEmptyCode) {
  bool FoundTopLevelDecl = false;
  EXPECT_TRUE(RunSyntaxOnlyToolOnCode(
      new TestAction(new FindTopLevelDeclConsumer(&FoundTopLevelDecl)), ""));
  EXPECT_TRUE(FoundTopLevelDecl);
}

namespace {
class FindClassDeclXConsumer : public clang::ASTConsumer {
 public:
  FindClassDeclXConsumer(bool *FoundClassDeclX)
      : FoundClassDeclX(FoundClassDeclX) {}
  virtual void HandleTopLevelDecl(clang::DeclGroupRef GroupRef) {
    if (CXXRecordDecl* Record = llvm::dyn_cast<clang::CXXRecordDecl>(
            *GroupRef.begin())) {
      if (Record->getName() == "X") {
        *FoundClassDeclX = true;
      }
    }
  }
 private:
  bool *FoundClassDeclX;
};
} // end namespace

TEST(RunSyntaxOnlyToolOnCode, FindsClassDecl) {
  bool FoundClassDeclX = false;
  EXPECT_TRUE(RunSyntaxOnlyToolOnCode(new TestAction(
      new FindClassDeclXConsumer(&FoundClassDeclX)), "class X;"));
  EXPECT_TRUE(FoundClassDeclX);

  FoundClassDeclX = false;
  EXPECT_TRUE(RunSyntaxOnlyToolOnCode(new TestAction(
      new FindClassDeclXConsumer(&FoundClassDeclX)), "class Y;"));
  EXPECT_FALSE(FoundClassDeclX);
}

} // end namespace tooling
} // end namespace clang

