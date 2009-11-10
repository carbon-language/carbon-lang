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
class TargetInfo;

void ComputeFeatureMap(TargetInfo &Target, llvm::StringMap<bool> &Features);

void InitializeCompileOptions(CompileOptions &Opts,
                              const llvm::StringMap<bool> &Features);

} // end namespace clang

#endif
