//===-- CheckerRegistration.h - Checker Registration Function ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SA_FRONTEND_CHECKERREGISTRATION_H
#define LLVM_CLANG_SA_FRONTEND_CHECKERREGISTRATION_H

#include "clang/Basic/LLVM.h"
#include <string>

namespace clang {
  class AnalyzerOptions;
  class LangOptions;
  class Diagnostic;

namespace ento {
  class CheckerManager;

CheckerManager *createCheckerManager(const AnalyzerOptions &opts,
                                     const LangOptions &langOpts,
                                     ArrayRef<std::string> plugins,
                                     Diagnostic &diags);

} // end ento namespace

} // end namespace clang

#endif
