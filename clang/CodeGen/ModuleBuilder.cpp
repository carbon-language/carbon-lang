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
using namespace clang;


/// Init - Create an ModuleBuilder with the specified ASTContext.
clang::CodeGen::CodeGenModule *
clang::CodeGen::Init(ASTContext &Context, const LangOptions &Features, 
                     llvm::Module &M, const llvm::TargetData &TD) {
  return new CodeGenModule(Context, Features, M, TD);
}

void clang::CodeGen::Terminate(CodeGenModule *B) {
  delete B;
}

/// CodeGenFunction - Convert the AST node for a FunctionDecl into LLVM.
///
void clang::CodeGen::CodeGenFunction(CodeGenModule *B, FunctionDecl *D) {
  B->EmitFunction(D);
}

/// CodeGenGlobalVar - Emit the specified global variable to LLVM.
void clang::CodeGen::CodeGenGlobalVar(CodeGenModule *Builder, FileVarDecl *D) {
  Builder->EmitGlobalVarDeclarator(D);
}


/// PrintStats - Emit statistic information to stderr.
///
void clang::CodeGen::PrintStats(CodeGenModule *B) {
  B->PrintStats();
}
