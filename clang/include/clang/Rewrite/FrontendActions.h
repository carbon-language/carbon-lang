//===-- FrontendActions.h - Useful Frontend Actions -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_REWRITE_FRONTENDACTIONS_H
#define LLVM_CLANG_REWRITE_FRONTENDACTIONS_H

#include "clang/Frontend/FrontendAction.h"

namespace clang {
class FixItRewriter;
class FixItOptions;

//===----------------------------------------------------------------------===//
// AST Consumer Actions
//===----------------------------------------------------------------------===//

class HTMLPrintAction : public ASTFrontendAction {
protected:
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         llvm::StringRef InFile);
};

class FixItAction : public ASTFrontendAction {
protected:
  llvm::OwningPtr<FixItRewriter> Rewriter;
  llvm::OwningPtr<FixItOptions> FixItOpts;

  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         llvm::StringRef InFile);

  virtual bool BeginSourceFileAction(CompilerInstance &CI,
                                     llvm::StringRef Filename);

  virtual void EndSourceFileAction();

  virtual bool hasASTFileSupport() const { return false; }

public:
  FixItAction();
  ~FixItAction();
};

class RewriteObjCAction : public ASTFrontendAction {
protected:
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         llvm::StringRef InFile);
};

class RewriteMacrosAction : public PreprocessorFrontendAction {
protected:
  void ExecuteAction();
};

class RewriteTestAction : public PreprocessorFrontendAction {
protected:
  void ExecuteAction();
};

}  // end namespace clang

#endif
