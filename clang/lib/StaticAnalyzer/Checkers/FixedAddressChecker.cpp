//=== FixedAddressChecker.cpp - Fixed address usage checker ----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This files defines FixedAddressChecker, a builtin checker that checks for
// assignment of a fixed address to a pointer.
// This check corresponds to CWE-587.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerVisitor.h"

using namespace clang;
using namespace ento;

namespace {
class FixedAddressChecker 
  : public CheckerVisitor<FixedAddressChecker> {
  BuiltinBug *BT;
public:
  FixedAddressChecker() : BT(0) {}
  static void *getTag();
  void PreVisitBinaryOperator(CheckerContext &C, const BinaryOperator *B);
};
}

void *FixedAddressChecker::getTag() {
  static int x;
  return &x;
}

void FixedAddressChecker::PreVisitBinaryOperator(CheckerContext &C,
                                                 const BinaryOperator *B) {
  // Using a fixed address is not portable because that address will probably
  // not be valid in all environments or platforms.

  if (B->getOpcode() != BO_Assign)
    return;

  QualType T = B->getType();
  if (!T->isPointerType())
    return;

  const GRState *state = C.getState();

  SVal RV = state->getSVal(B->getRHS());

  if (!RV.isConstant() || RV.isZeroConstant())
    return;

  if (ExplodedNode *N = C.generateNode()) {
    if (!BT)
      BT = new BuiltinBug("Use fixed address", 
                          "Using a fixed address is not portable because that "
                          "address will probably not be valid in all "
                          "environments or platforms.");
    RangedBugReport *R = new RangedBugReport(*BT, BT->getDescription(), N);
    R->addRange(B->getRHS()->getSourceRange());
    C.EmitReport(R);
  }
}

static void RegisterFixedAddressChecker(ExprEngine &Eng) {
  Eng.registerCheck(new FixedAddressChecker());
}

void ento::registerFixedAddressChecker(CheckerManager &mgr) {
  mgr.addCheckerRegisterFunction(RegisterFixedAddressChecker);
}
