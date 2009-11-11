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

#include "clang/Basic/LangOptions.h"
#include "clang/Frontend/CompileOptions.h"
#include "clang/Frontend/DependencyOutputOptions.h"
#include "clang/Frontend/DiagnosticOptions.h"
#include "clang/Frontend/HeaderSearchOptions.h"
#include "clang/Frontend/PreprocessorOptions.h"
#include "clang/Frontend/PreprocessorOutputOptions.h"
#include "llvm/ADT/StringMap.h"
#include <string>

namespace clang {

/// CompilerInvocation - Helper class for holding the data necessary to invoke
/// the compiler.
///
/// This class is designed to represent an abstract "invocation" of the
/// compiler, including data such as the include paths, the code generation
/// options, the warning flags, and so on.
class CompilerInvocation {
  /// Options controlling IRgen and the backend.
  CompileOptions CompileOpts;

  /// Options controlling dependency output.
  DependencyOutputOptions DependencyOutputOpts;

  /// Options controlling the diagnostic engine.
  DiagnosticOptions DiagOpts;

  /// Options controlling the #include directive.
  HeaderSearchOptions HeaderSearchOpts;

  /// Options controlling the language variant.
  LangOptions LangOpts;

  /// Options controlling the preprocessor (aside from #include handling).
  PreprocessorOptions PreprocessorOpts;

  /// Options controlling preprocessed output.
  PreprocessorOutputOptions PreprocessorOutputOpts;

  /// The location for the output file. This is optional only for compiler
  /// invocations which have no output.
  std::string OutputFile;

public:
  CompilerInvocation() {}

  /// @name Invidual Options
  /// @{

  std::string &getOutputFile() { return OutputFile; }
  const std::string &getOutputFile() const { return OutputFile; }

  /// @}
  /// @name Option Subgroups
  /// @{

  CompileOptions &getCompileOpts() { return CompileOpts; }
  const CompileOptions &getCompileOpts() const {
    return CompileOpts;
  }

  DependencyOutputOptions &getDependencyOutputOpts() {
    return DependencyOutputOpts;
  }
  const DependencyOutputOptions &getDependencyOutputOpts() const {
    return DependencyOutputOpts;
  }

  DiagnosticOptions &getDiagnosticOpts() { return DiagOpts; }
  const DiagnosticOptions &getDiagnosticOpts() const { return DiagOpts; }

  HeaderSearchOptions &getHeaderSearchOpts() { return HeaderSearchOpts; }
  const HeaderSearchOptions &getHeaderSearchOpts() const {
    return HeaderSearchOpts;
  }

  LangOptions &getLangOpts() { return LangOpts; }
  const LangOptions &getLangOpts() const { return LangOpts; }

  PreprocessorOptions &getPreprocessorOpts() { return PreprocessorOpts; }
  const PreprocessorOptions &getPreprocessorOpts() const {
    return PreprocessorOpts;
  }

  PreprocessorOutputOptions &getPreprocessorOutputOpts() {
    return PreprocessorOutputOpts;
  }
  const PreprocessorOutputOptions &getPreprocessorOutputOpts() const {
    return PreprocessorOutputOpts;
  }

  /// @}
};

} // end namespace clang

#endif
