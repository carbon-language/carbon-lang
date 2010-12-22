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

#include "GRExprEngineInternalChecks.h"
#include "GRExprEngineExperimentalChecks.h"
#include "clang/GR/Checkers/LocalCheckers.h"

using namespace clang;

void clang::RegisterExperimentalChecks(GRExprEngine &Eng) {
  // These are checks that never belong as internal checks
  // within GRExprEngine.
  RegisterCStringChecker(Eng);
  RegisterChrootChecker(Eng);
  RegisterMallocChecker(Eng);
  RegisterPthreadLockChecker(Eng);
  RegisterStreamChecker(Eng);
  RegisterUnreachableCodeChecker(Eng);
}

void clang::RegisterExperimentalInternalChecks(GRExprEngine &Eng) {
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
