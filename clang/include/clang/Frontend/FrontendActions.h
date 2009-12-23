//===-- FrontendActions.h - Useful Frontend Actions -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_FRONTENDACTIONS_H
#define LLVM_CLANG_FRONTEND_FRONTENDACTIONS_H

#include "clang/Frontend/FrontendAction.h"

namespace clang {
class FixItRewriter;

//===----------------------------------------------------------------------===//
// AST Consumer Actions
//===----------------------------------------------------------------------===//

class AnalysisAction : public ASTFrontendAction {
protected:
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         llvm::StringRef InFile);
};

class ASTPrintAction : public ASTFrontendAction {
protected:
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         llvm::StringRef InFile);
};

class ASTPrintXMLAction : public ASTFrontendAction {
protected:
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         llvm::StringRef InFile);
};

class ASTDumpAction : public ASTFrontendAction {
protected:
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         llvm::StringRef InFile);
};

class ASTViewAction : public ASTFrontendAction {
protected:
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         llvm::StringRef InFile);
};

class DeclContextPrintAction : public ASTFrontendAction {
protected:
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         llvm::StringRef InFile);
};

class DumpRecordAction : public ASTFrontendAction {
protected:
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         llvm::StringRef InFile);
};

class FixItAction : public ASTFrontendAction {
private:
  llvm::OwningPtr<FixItRewriter> Rewriter;

protected:

  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         llvm::StringRef InFile);

  virtual bool BeginSourceFileAction(CompilerInstance &CI,
                                     llvm::StringRef Filename);

  virtual void EndSourceFileAction();

  virtual bool hasASTSupport() const { return false; }

public:
  FixItAction();
  ~FixItAction();
};

class GeneratePCHAction : public ASTFrontendAction {
protected:
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         llvm::StringRef InFile);

  virtual bool usesCompleteTranslationUnit() { return false; }

  virtual bool hasASTSupport() const { return false; }
};

class HTMLPrintAction : public ASTFrontendAction {
protected:
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         llvm::StringRef InFile);
};

class InheritanceViewAction : public ASTFrontendAction {
protected:
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         llvm::StringRef InFile);
};

class RewriteObjCAction : public ASTFrontendAction {
protected:
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         llvm::StringRef InFile);
};

class SyntaxOnlyAction : public ASTFrontendAction {
protected:
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         llvm::StringRef InFile);

public:
  virtual bool hasCodeCompletionSupport() const { return true; }
};

//===----------------------------------------------------------------------===//
// Code Gen AST Actions
//===----------------------------------------------------------------------===//

class CodeGenAction : public ASTFrontendAction {
private:
  unsigned Act;

protected:
  CodeGenAction(unsigned _Act);

  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         llvm::StringRef InFile);
};

class EmitAssemblyAction : public CodeGenAction {
public:
  EmitAssemblyAction();
};

class EmitBCAction : public CodeGenAction {
public:
  EmitBCAction();
};

class EmitLLVMAction : public CodeGenAction {
public:
  EmitLLVMAction();
};

class EmitLLVMOnlyAction : public CodeGenAction {
public:
  EmitLLVMOnlyAction();
};

//===----------------------------------------------------------------------===//
// Preprocessor Actions
//===----------------------------------------------------------------------===//

class DumpRawTokensAction : public PreprocessorFrontendAction {
protected:
  void ExecuteAction();
};

class DumpTokensAction : public PreprocessorFrontendAction {
protected:
  void ExecuteAction();
};

class GeneratePTHAction : public PreprocessorFrontendAction {
protected:
  void ExecuteAction();
};

class ParseOnlyAction : public PreprocessorFrontendAction {
protected:
  void ExecuteAction();
};

class PreprocessOnlyAction : public PreprocessorFrontendAction {
protected:
  void ExecuteAction();
};

class PrintParseAction : public PreprocessorFrontendAction {
protected:
  void ExecuteAction();
};

class PrintPreprocessedAction : public PreprocessorFrontendAction {
protected:
  void ExecuteAction();

  virtual bool hasPCHSupport() const { return true; }
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
