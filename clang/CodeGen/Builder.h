//===--- Builder.h - Internal interface for LLVM Builder ------------------===//
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

#ifndef CODEGEN_BUILDER_H
#define CODEGEN_BUILDER_H

namespace llvm {
  class Module;
namespace clang {
  class ASTContext;
  class FunctionDecl;
    
namespace CodeGen {

class Builder {
  ASTContext &Context;
  Module &TheModule;
public:
  Builder(ASTContext &C, Module &M) : Context(C), TheModule(M) {}
  
  void CodeGenFunction(FunctionDecl *FD) {}
  
  void PrintStats() {}
    
};
}  // end namespace CodeGen
}  // end namespace clang
}  // end namespace llvm

#endif
