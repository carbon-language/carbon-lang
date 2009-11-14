//===-- FrontendAction.h - Generic Frontend Action Interface ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_FRONTENDACTION_H
#define LLVM_CLANG_FRONTEND_FRONTENDACTION_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/OwningPtr.h"
#include <string>

namespace llvm {
class Timer;
}

namespace clang {
class ASTUnit;
class ASTConsumer;
class CompilerInstance;

/// FrontendAction - Abstract base class for actions which can be performed by
/// the frontend.
class FrontendAction {
  std::string CurrentFile;
  llvm::OwningPtr<ASTUnit> CurrentASTUnit;
  CompilerInstance *Instance;
  llvm::Timer *CurrentTimer;

protected:
  /// @name Implementation Action Interface
  /// @{

  /// CreateASTConsumer - Create the AST consumer object for this action, if
  /// supported.
  ///
  /// This routine is called as part of \see BeginSourceAction(), which will
  /// fail if the AST consumer cannot be created. This will not be called if the
  /// action has indicated that it only uses the preprocessor.
  ///
  /// \param CI - The current compiler instance, provided as a convenience, \see
  /// getCompilerInstance().
  ///
  /// \param InFile - The current input file, provided as a convenience, \see
  /// getCurrentFile().
  ///
  /// \return The new AST consumer, or 0 on failure.
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         llvm::StringRef InFile) = 0;

  /// BeginSourceFileAction - Callback at the start of processing a single
  /// input.
  ///
  /// \return True on success; on failure \see ExecutionAction() and
  /// EndSourceFileAction() will not be called.
  virtual bool BeginSourceFileAction(CompilerInstance &CI,
                                     llvm::StringRef Filename) {
    return true;
  }

  /// ExecuteAction - Callback to run the program action, using the initialized
  /// compiler instance.
  ///
  /// This routine is guaranteed to only be called between \see
  /// BeginSourceFileAction() and \see EndSourceFileAction().
  virtual void ExecuteAction() = 0;

  /// EndSourceFileAction - Callback at the end of processing a single input;
  /// this is guaranteed to only be called following a successful call to
  /// BeginSourceFileAction (and BeingSourceFile).
  virtual void EndSourceFileAction() {}

  /// @}

public:
  FrontendAction();
  virtual ~FrontendAction();

  /// @name Compiler Instance Access
  /// @{

  CompilerInstance &getCompilerInstance() const {
    assert(Instance && "Compiler instance not registered!");
    return *Instance;
  }

  void setCompilerInstance(CompilerInstance *Value) { Instance = Value; }

  /// @}
  /// @name Current File Information
  /// @{

  bool isCurrentFileAST() const {
    assert(!CurrentFile.empty() && "No current file!");
    return CurrentASTUnit != 0;
  }

  const std::string &getCurrentFile() const {
    assert(!CurrentFile.empty() && "No current file!");
    return CurrentFile;
  }

  ASTUnit &getCurrentASTUnit() const {
    assert(!CurrentASTUnit && "No current AST unit!");
    return *CurrentASTUnit;
  }

  void setCurrentFile(llvm::StringRef Value, ASTUnit *AST = 0);

  /// @}
  /// @name Timing Utilities
  /// @{

  llvm::Timer *getCurrentTimer() const {
    return CurrentTimer;
  }

  void setCurrentTimer(llvm::Timer *Value) {
    CurrentTimer = Value;
  }

  /// @}
  /// @name Supported Modes
  /// @{

  /// usesPreprocessorOnly - Does this action only use the preprocessor? If so
  /// no AST context will be created and this action will be invalid with PCH
  /// inputs.
  virtual bool usesPreprocessorOnly() const = 0;

  /// usesCompleteTranslationUnit - For AST based actions, should the
  /// translation unit be completed?
  virtual bool usesCompleteTranslationUnit() { return true; }

  /// hasPCHSupport - Does this action support use with PCH?
  virtual bool hasPCHSupport() const { return !usesPreprocessorOnly(); }

  /// hasASTSupport - Does this action support use with AST files?
  virtual bool hasASTSupport() const { return !usesPreprocessorOnly(); }

  /// hasCodeCompletionSupport - Does this action support use with code
  /// completion?
  virtual bool hasCodeCompletionSupport() const { return false; }

  /// @}
  /// @name Public Action Interface
  /// @{

  /// BeginSourceFile - Prepare the action for processing the input file \arg
  /// Filename; this is run after the options and frontend have been
  /// initialized, but prior to executing any per-file processing.
  ///
  /// \param CI - The compiler instance this action is being run from. The
  /// action may store and use this object up until the matching EndSourceFile
  /// action.
  ///
  /// \param Filename - The input filename, which will be made available to
  /// clients via \see getCurrentFile().
  ///
  /// \param IsAST - Indicates whether this is an AST input. AST inputs require
  /// special handling, since the AST file itself contains several objects which
  /// would normally be owned by the CompilerInstance. When processing AST input
  /// files, these objects should generally not be initialized in the
  /// CompilerInstance -- they will automatically be shared with the AST file in
  /// between \see BeginSourceFile() and \see EndSourceFile().
  ///
  /// \return True on success; the compilation of this file should be aborted
  /// and neither Execute nor EndSourceFile should be called.
  bool BeginSourceFile(CompilerInstance &CI, llvm::StringRef Filename,
                       bool IsAST = false);

  /// Execute - Set the source managers main input file, and run the action.
  void Execute();

  /// EndSourceFile - Perform any per-file post processing, deallocate per-file
  /// objects, and run statistics and output file cleanup code.
  void EndSourceFile();

  /// @}
};

/// ASTFrontendAction - Abstract base class to use for AST consumer based
/// frontend actios.
class ASTFrontendAction : public FrontendAction {
  /// ExecuteAction - Implement the ExecuteAction interface by running Sema on
  /// the already initialized AST consumer.
  ///
  /// This will also take care of instantiating a code completion consumer if
  /// the user requested it and the action supports it.
  virtual void ExecuteAction();

public:
  virtual bool usesPreprocessorOnly() const { return false; }
};

/// PreprocessorFrontendAction - Abstract base class to use for preprocessor
/// based frontend actions.
class PreprocessorFrontendAction : public FrontendAction {
protected:
  /// CreateASTConsumer - Provide a default implementation which returns aborts,
  /// this method should never be called by FrontendAction clients.
  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         llvm::StringRef InFile);

public:
  virtual bool usesPreprocessorOnly() const { return true; }
};

}  // end namespace clang

#endif
