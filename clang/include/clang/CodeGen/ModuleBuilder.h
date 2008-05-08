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
}

namespace clang {
  class Diagnostic;
  struct LangOptions;
  class ASTConsumer;
  
  ASTConsumer *CreateLLVMCodeGen(Diagnostic &Diags, const LangOptions &Features,
                                 llvm::Module *&DestModule,
                                 bool GenerateDebugInfo);
}

#endif
