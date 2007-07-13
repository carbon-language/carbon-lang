//===--- CodeGenModule.h - Per-Module state for LLVM CodeGen --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the internal per-translation-unit state used for llvm translation. 
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_CODEGENMODULE_H
#define CODEGEN_CODEGENMODULE_H

#include "CodeGenTypes.h"
#include "llvm/ADT/DenseMap.h"

namespace llvm {
  class Module;
  class Constant;
  class Function;
}

namespace clang {
  class ASTContext;
  class FunctionDecl;
  class Decl;
  class FileVarDecl;
    
namespace CodeGen {

/// CodeGenModule - This class organizes the cross-module state that is used
/// while generating LLVM code.
class CodeGenModule {
  ASTContext &Context;
  llvm::Module &TheModule;
  CodeGenTypes Types;

  llvm::Function *MemCpyFn;
  llvm::DenseMap<const Decl*, llvm::Constant*> GlobalDeclMap;
public:
  CodeGenModule(ASTContext &C, llvm::Module &M);
  
  ASTContext &getContext() const { return Context; }
  llvm::Module &getModule() const { return TheModule; }
  CodeGenTypes &getTypes() { return Types; }
  
  llvm::Constant *GetAddrOfGlobalDecl(const Decl *D);
  
  llvm::Function *getMemCpyFn();
  
  void EmitFunction(const FunctionDecl *FD);
  void EmitGlobalVar(const FileVarDecl *D);
  
  void PrintStats() {}
};
}  // end namespace CodeGen
}  // end namespace clang

#endif
