//===--- CodeGen/ModuleBuilder.h - Build LLVM from ASTs ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ModuleBuilder interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CODEGEN_MODULEBUILDER_H
#define LLVM_CLANG_CODEGEN_MODULEBUILDER_H

namespace llvm {
  class Module;
  class TargetData;
}

namespace clang {
  class ASTContext;
  class FunctionDecl;
  class LinkageSpecDecl;
  class FileVarDecl;
  struct LangOptions;
  class Diagnostic;

namespace CodeGen {
  class CodeGenModule;
  
  /// Init - Create an ModuleBuilder with the specified ASTContext.
  CodeGenModule *Init(ASTContext &Context, const LangOptions &Features,
                      llvm::Module &M, const llvm::TargetData &TD,
                      Diagnostic &Diags);
  
  /// CodeGenFunction - Convert the AST node for a FunctionDecl into LLVM.
  ///
  void CodeGenFunction(CodeGenModule *Builder, FunctionDecl *D);

  void CodeGenLinkageSpec(CodeGenModule *Builder, LinkageSpecDecl *LS);
  
  /// CodeGenGlobalVar - Emit the specified global variable to LLVM.
  void CodeGenGlobalVar(CodeGenModule *Builder, FileVarDecl *D);
  
  /// PrintStats - Emit statistic information to stderr.
  ///
  void PrintStats(CodeGenModule *Builder);
  
  /// Terminate - Gracefully shut down the builder.
  ///
  void Terminate(CodeGenModule *Builder);
}  // end namespace CodeGen
}  // end namespace clang

#endif
