//===--- CodeGen/ModuleBuilder.h - Build LLVM from ASTs ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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
}

namespace clang {
  class ASTContext;
  class FunctionDecl;
  
namespace CodeGen {
  /// BuilderTy - This is an opaque type used to reference ModuleBuilder
  /// objects.
  typedef void BuilderTy;
  
  /// Init - Create an ModuleBuilder with the specified ASTContext.
  BuilderTy *Init(ASTContext &Context, llvm::Module &M);
  
  /// CodeGenFunction - Convert the AST node for a FunctionDecl into LLVM.
  ///
  void CodeGenFunction(BuilderTy *Builder, FunctionDecl *D);
  
  /// PrintStats - Emit statistic information to stderr.
  ///
  void PrintStats(BuilderTy *Builder);
  
  /// Terminate - Gracefully shut down the builder.
  ///
  void Terminate(BuilderTy *Builder);
}  // end namespace CodeGen
}  // end namespace clang

#endif
