//=-- ExprEngineExperimentalChecks.h ------------------------------*- C++ -*-=
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

namespace GR {

class ExprEngine;

void RegisterAnalyzerStatsChecker(ExprEngine &Eng);
void RegisterChrootChecker(ExprEngine &Eng);
void RegisterCStringChecker(ExprEngine &Eng);
void RegisterIdempotentOperationChecker(ExprEngine &Eng);
void RegisterMallocChecker(ExprEngine &Eng);
void RegisterPthreadLockChecker(ExprEngine &Eng);
void RegisterStreamChecker(ExprEngine &Eng);
void RegisterUnreachableCodeChecker(ExprEngine &Eng);

} // end GR namespace

} // end clang namespace

#endif
