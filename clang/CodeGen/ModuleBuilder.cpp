//===--- ModuleBuilder.cpp - Emit LLVM Code from ASTs ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This builds an AST and converts it to LLVM Code.
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGen/ModuleBuilder.h"
#include "CodeGenModule.h"
#include "clang/AST/Decl.h"
using namespace clang;


/// Init - Create an ModuleBuilder with the specified ASTContext.
clang::CodeGen::CodeGenModule *
clang::CodeGen::Init(ASTContext &Context, const LangOptions &Features, 
                     llvm::Module &M, const llvm::TargetData &TD, 
                     Diagnostic &Diags) {
  return new CodeGenModule(Context, Features, M, TD, Diags);
}

void clang::CodeGen::Terminate(CodeGenModule *B) {
  delete B;
}

/// CodeGenFunction - Convert the AST node for a FunctionDecl into LLVM.
///
void clang::CodeGen::CodeGenFunction(CodeGenModule *B, FunctionDecl *D) {
  B->EmitFunction(D);
}

/// CodeGenLinkageSpec - Emit the specified linkage space to LLVM.
void clang::CodeGen::CodeGenLinkageSpec(CodeGenModule *Builder,
					LinkageSpecDecl *LS) {
  if (LS->getLanguage() == LinkageSpecDecl::lang_cxx)
    Builder->WarnUnsupported(LS, "linkage spec");

  // FIXME: implement C++ linkage, C linkage works mostly by C
  // language reuse already.
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
