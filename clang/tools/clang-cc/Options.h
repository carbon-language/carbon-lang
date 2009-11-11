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

#include "llvm/ADT/StringMap.h"

namespace clang {

class CompileOptions;
class LangOptions;
class PreprocessorOptions;
class TargetInfo;

// FIXME: This can be sunk into InitializeCompileOptions now that that happens
// before language initialization?
void ComputeFeatureMap(TargetInfo &Target, llvm::StringMap<bool> &Features);

void InitializeCompileOptions(CompileOptions &Opts,
                              const llvm::StringMap<bool> &Features);

void InitializePreprocessorOptions(PreprocessorOptions &Opts);

} // end namespace clang

#endif
