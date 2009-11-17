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
#include "clang/Basic/TargetOptions.h"
#include "clang/CodeGen/CodeGenOptions.h"
#include "clang/Frontend/AnalysisConsumer.h"
#include "clang/Frontend/DependencyOutputOptions.h"
#include "clang/Frontend/DiagnosticOptions.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Frontend/HeaderSearchOptions.h"
#include "clang/Frontend/PreprocessorOptions.h"
#include "clang/Frontend/PreprocessorOutputOptions.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringMap.h"
#include <string>
#include <vector>

namespace llvm {
  template<typename T> class SmallVectorImpl;
}

namespace clang {

/// CompilerInvocation - Helper class for holding the data necessary to invoke
/// the compiler.
///
/// This class is designed to represent an abstract "invocation" of the
/// compiler, including data such as the include paths, the code generation
/// options, the warning flags, and so on.
class CompilerInvocation {
  /// Options controlling the static analyzer.
  AnalyzerOptions AnalyzerOpts;

  /// Options controlling IRgen and the backend.
  CodeGenOptions CodeGenOpts;

  /// Options controlling dependency output.
  DependencyOutputOptions DependencyOutputOpts;

  /// Options controlling the diagnostic engine.
  DiagnosticOptions DiagnosticOpts;

  /// Options controlling the frontend itself.
  FrontendOptions FrontendOpts;

  /// Options controlling the #include directive.
  HeaderSearchOptions HeaderSearchOpts;

  /// Options controlling the language variant.
  LangOptions LangOpts;

  /// Options controlling the preprocessor (aside from #include handling).
  PreprocessorOptions PreprocessorOpts;

  /// Options controlling preprocessed output.
  PreprocessorOutputOptions PreprocessorOutputOpts;

  /// Options controlling the target.
  TargetOptions TargetOpts;

public:
  CompilerInvocation() {}

  /// @name Utility Methods
  /// @{

  /// CreateFromArgs - Create a compiler invocation from a list of input
  /// options.
  ///
  /// FIXME: Documenting error behavior.
  ///
  /// \param Res [out] - The resulting invocation.
  /// \param Args - The input argument strings.
  static void CreateFromArgs(CompilerInvocation &Res,
                            const llvm::SmallVectorImpl<llvm::StringRef> &Args);

  /// toArgs - Convert the CompilerInvocation to a list of strings suitable for
  /// passing to CreateFromArgs.
  void toArgs(std::vector<std::string> &Res);

  /// @}
  /// @name Option Subgroups
  /// @{

  AnalyzerOptions &getAnalyzerOpts() { return AnalyzerOpts; }
  const AnalyzerOptions &getAnalyzerOpts() const {
    return AnalyzerOpts;
  }

  CodeGenOptions &getCodeGenOpts() { return CodeGenOpts; }
  const CodeGenOptions &getCodeGenOpts() const {
    return CodeGenOpts;
  }

  DependencyOutputOptions &getDependencyOutputOpts() {
    return DependencyOutputOpts;
  }
  const DependencyOutputOptions &getDependencyOutputOpts() const {
    return DependencyOutputOpts;
  }

  DiagnosticOptions &getDiagnosticOpts() { return DiagnosticOpts; }
  const DiagnosticOptions &getDiagnosticOpts() const { return DiagnosticOpts; }

  HeaderSearchOptions &getHeaderSearchOpts() { return HeaderSearchOpts; }
  const HeaderSearchOptions &getHeaderSearchOpts() const {
    return HeaderSearchOpts;
  }

  FrontendOptions &getFrontendOpts() { return FrontendOpts; }
  const FrontendOptions &getFrontendOpts() const {
    return FrontendOpts;
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

  TargetOptions &getTargetOpts() { return TargetOpts; }
  const TargetOptions &getTargetOpts() const {
    return TargetOpts;
  }

  /// @}
};

} // end namespace clang

#endif
