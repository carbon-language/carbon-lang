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

#include "GRExprEngineExperimentalChecks.h"
#include "clang/Analysis/LocalCheckers.h"

using namespace clang;

void clang::RegisterExperimentalChecks(GRExprEngine &Eng) {
  RegisterPthreadLockChecker(Eng);  
}

