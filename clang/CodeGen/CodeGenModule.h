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

namespace llvm {
  class Module;
namespace clang {
  class ASTContext;
  class FunctionDecl;
    
namespace CodeGen {

/// CodeGenModule - This class organizes the cross-module state that is used
/// while generating LLVM code.
class CodeGenModule {
  ASTContext &Context;
  Module &TheModule;
public:
  CodeGenModule(ASTContext &C, Module &M) : Context(C), TheModule(M) {}
  
  ASTContext &getContext() const { return Context; }
  
  void EmitFunction(FunctionDecl *FD);
  
  void PrintStats() {}
};
}  // end namespace CodeGen
}  // end namespace clang
}  // end namespace llvm

#endif
