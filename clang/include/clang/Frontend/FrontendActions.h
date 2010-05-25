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
#include <string>
#include <vector>

namespace clang {
class FixItRewriter;
class FixItPathRewriter;

//===----------------------------------------------------------------------===//
// Custom Consumer Actions
//===----------------------------------------------------------------------===//

class InitOnlyAction : public FrontendAction {
  virtual void ExecuteAction();

  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         llvm::StringRef InFile);

public:
  // Don't claim to only use the preprocessor, we want to follow the AST path,
  // but do nothing.
  virtual bool usesPreprocessorOnly() const { return false; }
};

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

class FixItAction : public ASTFrontendAction {
protected:
  llvm::OwningPtr<FixItRewriter> Rewriter;
  llvm::OwningPtr<FixItPathRewriter> PathRewriter;

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

class BoostConAction : public SyntaxOnlyAction {
protected:
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         llvm::StringRef InFile);
};

/**
 * \brief Frontend action adaptor that merges ASTs together.
 *
 * This action takes an existing AST file and "merges" it into the AST
 * context, producing a merged context. This action is an action
 * adaptor, which forwards most of its calls to another action that
 * will consume the merged context.
 */
class ASTMergeAction : public FrontendAction {
  /// \brief The action that the merge action adapts.
  FrontendAction *AdaptedAction;
  
  /// \brief The set of AST files to merge.
  std::vector<std::string> ASTFiles;

protected:
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         llvm::StringRef InFile);

  virtual bool BeginSourceFileAction(CompilerInstance &CI,
                                     llvm::StringRef Filename);

  virtual void ExecuteAction();
  virtual void EndSourceFileAction();

public:
  ASTMergeAction(FrontendAction *AdaptedAction,
                 std::string *ASTFiles, unsigned NumASTFiles);
  virtual ~ASTMergeAction();

  virtual bool usesPreprocessorOnly() const;
  virtual bool usesCompleteTranslationUnit();
  virtual bool hasPCHSupport() const;
  virtual bool hasASTSupport() const;
  virtual bool hasCodeCompletionSupport() const;
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
