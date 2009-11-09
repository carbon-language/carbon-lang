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

#include "clang/Analysis/PathSensitive/CheckerVisitor.h"
#include "GRExprEngineInternalChecks.h"

using namespace clang;

namespace {
class VISIBILITY_HIDDEN FixedAddressChecker 
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

  if (B->getOpcode() != BinaryOperator::Assign)
    return;

  QualType T = B->getType();
  if (!T->isPointerType())
    return;

  const GRState *state = C.getState();

  SVal RV = state->getSVal(B->getRHS());

  if (!RV.isConstant() || RV.isZeroConstant())
    return;

  if (ExplodedNode *N = C.GenerateNode(B)) {
    if (!BT)
      BT = new BuiltinBug("Use fixed address", 
                          "Using a fixed address is not portable because that address will probably not be valid in all environments or platforms.");
    RangedBugReport *R = new RangedBugReport(*BT, BT->getDescription().c_str(),
                                             N);
    R->addRange(B->getRHS()->getSourceRange());
    C.EmitReport(R);
  }
}

void clang::RegisterFixedAddressChecker(GRExprEngine &Eng) {
  Eng.registerCheck(new FixedAddressChecker());
}
