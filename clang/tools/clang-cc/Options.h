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

#include "clang/Frontend/FrontendOptions.h"
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
class TargetOptions;

void InitializeAnalyzerOptions(AnalyzerOptions &Opts);

void InitializeCodeGenOptions(CodeGenOptions &Opts,
                              const LangOptions &Lang,
                              bool TimePasses);

void InitializeDependencyOutputOptions(DependencyOutputOptions &Opts);

void InitializeDiagnosticOptions(DiagnosticOptions &Opts);

void InitializeFrontendOptions(FrontendOptions &Opts);

void InitializeHeaderSearchOptions(HeaderSearchOptions &Opts,
                                   llvm::StringRef BuiltinIncludePath);

void InitializeLangOptions(LangOptions &Options,
                           FrontendOptions::InputKind LK,
                           TargetInfo &Target);

void InitializePreprocessorOptions(PreprocessorOptions &Opts);

void InitializePreprocessorOutputOptions(PreprocessorOutputOptions &Opts);

void InitializeTargetOptions(TargetOptions &Opts);

} // end namespace clang

#endif
