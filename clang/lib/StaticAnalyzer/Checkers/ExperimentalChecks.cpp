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

#include "InternalChecks.h"
#include "ExperimentalChecks.h"
#include "clang/StaticAnalyzer/Checkers/LocalCheckers.h"

using namespace clang;
using namespace ento;

void ento::RegisterExperimentalChecks(ExprEngine &Eng) {
  // These are checks that never belong as internal checks
  // within ExprEngine.
  RegisterMallocChecker(Eng); // ArrayBoundChecker depends on this.
}
