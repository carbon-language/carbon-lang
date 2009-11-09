//===-- CompilerInvocation.h - Compiler Invocation Helper Data --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_COMPILERINVOCATION_H_
#define LLVM_CLANG_FRONTEND_COMPILERINVOCATION_H_

#include "clang/Frontend/DiagnosticOptions.h"
#include <string>

namespace clang {

/// CompilerInvocation - Helper class for holding the data necessary to invoke
/// the compiler.
///
/// This class is designed to represent an abstract "invocation" of the
/// compiler, including data such as the include paths, the code generation
/// options, the warning flags, and so on.
class CompilerInvocation {
  /// The location for the output file. This is optional only for compiler
  /// invocations which have no output.
  std::string OutputFile;

  DiagnosticOptions Diags;
  
public:
  CompilerInvocation() {}

  std::string &getOutputFile() { return OutputFile; }
  const std::string &getOutputFile() const { return OutputFile; }

  DiagnosticOptions &getDiagnosticOpts() { return Diags; }
  const DiagnosticOptions &getDiagnosticOpts() const { return Diags; }  
};

} // end namespace clang

#endif
