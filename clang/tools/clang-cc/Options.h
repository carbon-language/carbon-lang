//===-- Options.h - clang-cc Option Handling --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANGCC_OPTIONS_H
#define LLVM_CLANGCC_OPTIONS_H

#include "llvm/ADT/StringRef.h"

namespace clang {

class AnalyzerOptions;
class CodeGenOptions;
class DependencyOutputOptions;
class DiagnosticOptions;
class FrontendOptions;
class HeaderSearchOptions;
class LangOptions;
class PreprocessorOptions;
class PreprocessorOutputOptions;
class TargetInfo;

enum LangKind {
  langkind_unspecified,
  langkind_c,
  langkind_c_cpp,
  langkind_asm_cpp,
  langkind_cxx,
  langkind_cxx_cpp,
  langkind_objc,
  langkind_objc_cpp,
  langkind_objcxx,
  langkind_objcxx_cpp,
  langkind_ocl,
  langkind_ast
};

void InitializeAnalyzerOptions(AnalyzerOptions &Opts);

void InitializeDependencyOutputOptions(DependencyOutputOptions &Opts);

void InitializeDiagnosticOptions(DiagnosticOptions &Opts);

void InitializeFrontendOptions(FrontendOptions &Opts);

void InitializeCodeGenOptions(CodeGenOptions &Opts,
                              const TargetInfo &Target);

void InitializeHeaderSearchOptions(HeaderSearchOptions &Opts,
                                   llvm::StringRef BuiltinIncludePath,
                                   bool Verbose,
                                   const LangOptions &Lang);

void InitializeLangOptions(LangOptions &Options, LangKind LK,
                           TargetInfo &Target,
                           const CodeGenOptions &CodeGenOpts);

void InitializePreprocessorOptions(PreprocessorOptions &Opts);

void InitializePreprocessorOutputOptions(PreprocessorOutputOptions &Opts);

} // end namespace clang

#endif
