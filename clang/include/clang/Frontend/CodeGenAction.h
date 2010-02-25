//===--- CodeGenAction.h - LLVM Code Generation Frontend Action -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/FrontendAction.h"

namespace clang {

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

class EmitObjAction : public CodeGenAction {
public:
  EmitObjAction();
};

}
