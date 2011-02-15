//=-- ExperimentalChecks.h ----------------------------------------*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines functions to instantiate and register experimental
//  checks in ExprEngine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_GR_ExprEngine_EXPERIMENTAL_CHECKS
#define LLVM_CLANG_GR_ExprEngine_EXPERIMENTAL_CHECKS

namespace clang {

namespace ento {

class ExprEngine;

void RegisterAnalyzerStatsChecker(ExprEngine &Eng);
void RegisterMallocChecker(ExprEngine &Eng);

} // end GR namespace

} // end clang namespace

#endif
