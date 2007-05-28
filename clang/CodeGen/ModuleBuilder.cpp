//===--- ModuleBuilder.cpp - Emit LLVM Code from ASTs ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This builds an AST and converts it to LLVM Code.
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGen/ModuleBuilder.h"
#include "CodeGenModule.h"
using namespace llvm;
using namespace clang;


/// Init - Create an ModuleBuilder with the specified ASTContext.
llvm::clang::CodeGen::BuilderTy *
llvm::clang::CodeGen::Init(ASTContext &Context, Module &M) {
  return new CodeGenModule(Context, M);
}

void llvm::clang::CodeGen::Terminate(BuilderTy *B) {
  delete static_cast<CodeGenModule*>(B);
}

/// CodeGenFunction - Convert the AST node for a FunctionDecl into LLVM.
///
void llvm::clang::CodeGen::CodeGenFunction(BuilderTy *B, FunctionDecl *D) {
  static_cast<CodeGenModule*>(B)->EmitFunction(D);
}

/// PrintStats - Emit statistic information to stderr.
///
void llvm::clang::CodeGen::PrintStats(BuilderTy *B) {
  static_cast<CodeGenModule*>(B)->PrintStats();
}
