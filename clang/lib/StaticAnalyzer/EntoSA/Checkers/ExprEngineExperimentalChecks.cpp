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

#include "ExprEngineInternalChecks.h"
#include "ExprEngineExperimentalChecks.h"
#include "clang/StaticAnalyzer/Checkers/LocalCheckers.h"

using namespace clang;
using namespace ento;

void ento::RegisterExperimentalChecks(ExprEngine &Eng) {
  // These are checks that never belong as internal checks
  // within ExprEngine.
  RegisterCStringChecker(Eng);
  RegisterChrootChecker(Eng);
  RegisterMallocChecker(Eng);
  RegisterPthreadLockChecker(Eng);
  RegisterStreamChecker(Eng);
  RegisterUnreachableCodeChecker(Eng);
}

void ento::RegisterExperimentalInternalChecks(ExprEngine &Eng) {
  // These are internal checks that should eventually migrate to
  // RegisterInternalChecks() once they have been further tested.
  
  // Note that this must be registered after ReturnStackAddresEngsChecker.
  RegisterReturnPointerRangeChecker(Eng);
  
  RegisterArrayBoundChecker(Eng);
  RegisterCastSizeChecker(Eng);
  RegisterCastToStructChecker(Eng);
  RegisterFixedAddressChecker(Eng);
  RegisterPointerArithChecker(Eng);
  RegisterPointerSubChecker(Eng);
}
