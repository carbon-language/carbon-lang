//===--- CodeGenAction.h - LLVM Code Generation Frontend Action -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/FrontendAction.h"
#include "llvm/ADT/OwningPtr.h"

namespace llvm {
  class Module;
}

namespace clang {

class CodeGenAction : public ASTFrontendAction {
private:
  unsigned Act;
  llvm::OwningPtr<llvm::Module> TheModule;

protected:
  CodeGenAction(unsigned _Act);

  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         llvm::StringRef InFile);

  virtual void EndSourceFileAction();

public:
  ~CodeGenAction();

  /// takeModule - Take the generated LLVM module, for use after the action has
  /// been run. The result may be null on failure.
  llvm::Module *takeModule();
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

class EmitObjAction : public CodeGenAction {
public:
  EmitObjAction();
};

}
