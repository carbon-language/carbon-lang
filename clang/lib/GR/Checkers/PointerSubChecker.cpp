//=== PointerSubChecker.cpp - Pointer subtraction checker ------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This files defines PointerSubChecker, a builtin checker that checks for
// pointer subtractions on two pointers pointing to different memory chunks. 
// This check corresponds to CWE-469.
//
//===----------------------------------------------------------------------===//

#include "ExprEngineInternalChecks.h"
#include "clang/GR/BugReporter/BugType.h"
#include "clang/GR/PathSensitive/CheckerVisitor.h"

using namespace clang;
using namespace GR;

namespace {
class PointerSubChecker 
  : public CheckerVisitor<PointerSubChecker> {
  BuiltinBug *BT;
public:
  PointerSubChecker() : BT(0) {}
  static void *getTag();
  void PreVisitBinaryOperator(CheckerContext &C, const BinaryOperator *B);
};
}

void *PointerSubChecker::getTag() {
  static int x;
  return &x;
}

void PointerSubChecker::PreVisitBinaryOperator(CheckerContext &C,
                                               const BinaryOperator *B) {
  // When doing pointer subtraction, if the two pointers do not point to the
  // same memory chunk, emit a warning.
  if (B->getOpcode() != BO_Sub)
    return;

  const GRState *state = C.getState();
  SVal LV = state->getSVal(B->getLHS());
  SVal RV = state->getSVal(B->getRHS());

  const MemRegion *LR = LV.getAsRegion();
  const MemRegion *RR = RV.getAsRegion();

  if (!(LR && RR))
    return;

  const MemRegion *BaseLR = LR->getBaseRegion();
  const MemRegion *BaseRR = RR->getBaseRegion();

  if (BaseLR == BaseRR)
    return;

  // Allow arithmetic on different symbolic regions.
  if (isa<SymbolicRegion>(BaseLR) || isa<SymbolicRegion>(BaseRR))
    return;

  if (ExplodedNode *N = C.generateNode()) {
    if (!BT)
      BT = new BuiltinBug("Pointer subtraction", 
                          "Subtraction of two pointers that do not point to "
                          "the same memory chunk may cause incorrect result.");
    RangedBugReport *R = new RangedBugReport(*BT, BT->getDescription(), N);
    R->addRange(B->getSourceRange());
    C.EmitReport(R);
  }
}

void GR::RegisterPointerSubChecker(ExprEngine &Eng) {
  Eng.registerCheck(new PointerSubChecker());
}
