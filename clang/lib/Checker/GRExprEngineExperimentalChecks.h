//=-- GRExprEngineExperimentalChecks.h ------------------------------*- C++ -*-=
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines functions to instantiate and register experimental
//  checks in GRExprEngine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_GREXPRENGINE_EXPERIMENTAL_CHECKS
#define LLVM_CLANG_GREXPRENGINE_EXPERIMENTAL_CHECKS

namespace clang {

class GRExprEngine;

void RegisterAnalyzerStatsChecker(GRExprEngine &Eng);
void RegisterChrootChecker(GRExprEngine &Eng);
void RegisterCStringChecker(GRExprEngine &Eng);
void RegisterIdempotentOperationChecker(GRExprEngine &Eng);
void RegisterMallocChecker(GRExprEngine &Eng);
void RegisterPthreadLockChecker(GRExprEngine &Eng);
void RegisterStreamChecker(GRExprEngine &Eng);
void RegisterUnreachableCodeChecker(GRExprEngine &Eng);

} // end clang namespace
#endif
