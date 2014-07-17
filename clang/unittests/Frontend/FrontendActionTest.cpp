//===- unittests/Frontend/FrontendActionTest.cpp - FrontendAction tests ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/FrontendAction.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

namespace {

class TestASTFrontendAction : public ASTFrontendAction {
public:
  TestASTFrontendAction(bool enableIncrementalProcessing = false)
    : EnableIncrementalProcessing(enableIncrementalProcessing) { }

  bool EnableIncrementalProcessing;
  std::vector<std::string> decl_names;

  virtual bool BeginSourceFileAction(CompilerInstance &ci, StringRef filename) {
    if (EnableIncrementalProcessing)
      ci.getPreprocessor().enableIncrementalProcessing();

    return ASTFrontendAction::BeginSourceFileAction(ci, filename);
  }

  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         StringRef InFile) {
    return new Visitor(decl_names);
  }

private:
  class Visitor : public ASTConsumer, public RecursiveASTVisitor<Visitor> {
  public:
    Visitor(std::vector<std::string> &decl_names) : decl_names_(decl_names) {}

    virtual void HandleTranslationUnit(ASTContext &context) {
      TraverseDecl(context.getTranslationUnitDecl());
    }

    virtual bool VisitNamedDecl(NamedDecl *Decl) {
      decl_names_.push_back(Decl->getQualifiedNameAsString());
      return true;
    }

  private:
    std::vector<std::string> &decl_names_;
  };
};

TEST(ASTFrontendAction, Sanity) {
  CompilerInvocation *invocation = new CompilerInvocation;
  invocation->getPreprocessorOpts().addRemappedFile(
    "test.cc", MemoryBuffer::getMemBuffer("int main() { float x; }"));
  invocation->getFrontendOpts().Inputs.push_back(FrontendInputFile("test.cc",
                                                                   IK_CXX));
  invocation->getFrontendOpts().ProgramAction = frontend::ParseSyntaxOnly;
  invocation->getTargetOpts().Triple = "i386-unknown-linux-gnu";
  CompilerInstance compiler;
  compiler.setInvocation(invocation);
  compiler.createDiagnostics();

  TestASTFrontendAction test_action;
  ASSERT_TRUE(compiler.ExecuteAction(test_action));
  ASSERT_EQ(2U, test_action.decl_names.size());
  EXPECT_EQ("main", test_action.decl_names[0]);
  EXPECT_EQ("x", test_action.decl_names[1]);
}

TEST(ASTFrontendAction, IncrementalParsing) {
  CompilerInvocation *invocation = new CompilerInvocation;
  invocation->getPreprocessorOpts().addRemappedFile(
    "test.cc", MemoryBuffer::getMemBuffer("int main() { float x; }"));
  invocation->getFrontendOpts().Inputs.push_back(FrontendInputFile("test.cc",
                                                                   IK_CXX));
  invocation->getFrontendOpts().ProgramAction = frontend::ParseSyntaxOnly;
  invocation->getTargetOpts().Triple = "i386-unknown-linux-gnu";
  CompilerInstance compiler;
  compiler.setInvocation(invocation);
  compiler.createDiagnostics();

  TestASTFrontendAction test_action(/*enableIncrementalProcessing=*/true);
  ASSERT_TRUE(compiler.ExecuteAction(test_action));
  ASSERT_EQ(2U, test_action.decl_names.size());
  EXPECT_EQ("main", test_action.decl_names[0]);
  EXPECT_EQ("x", test_action.decl_names[1]);
}

} // anonymous namespace
