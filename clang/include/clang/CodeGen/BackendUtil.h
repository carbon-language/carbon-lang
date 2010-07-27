//===--- BackendUtil.h - LLVM Backend Utilities -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CODEGEN_BACKEND_UTIL_H
#define LLVM_CLANG_CODEGEN_BACKEND_UTIL_H

namespace llvm {
  class Module;
  class raw_ostream;
}

namespace clang {
  class Diagnostic;
  class CodeGenOptions;
  class TargetOptions;
  class Decl;
  
  enum BackendAction {
    Backend_EmitAssembly,  ///< Emit native assembly files
    Backend_EmitBC,        ///< Emit LLVM bitcode files
    Backend_EmitLL,        ///< Emit human-readable LLVM assembly
    Backend_EmitNothing,   ///< Don't emit anything (benchmarking mode)
    Backend_EmitMCNull,    ///< Run CodeGen, but don't emit anything
    Backend_EmitObj        ///< Emit native object files
  };
  
  void EmitBackendOutput(Diagnostic &Diags, const CodeGenOptions &CGOpts,
                         const TargetOptions &TOpts, llvm::Module *M,
                         BackendAction Action, llvm::raw_ostream *OS);

  /// \brief Determines if the decl can be CodeGen'ed or deserialized from PCH
  /// lazily, only when used; this is only relevant for function or file scoped
  /// var definitions.
  ///
  /// \returns true if the function/var must be CodeGen'ed/deserialized even if
  /// it is not used.
  bool DeclIsRequiredFunctionOrFileScopedVar(const Decl *D);
}

#endif
