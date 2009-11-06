//== DivZeroChecker.cpp - Division by zero checker --------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines DivZeroChecker, a builtin check in GRExprEngine that performs
// checks for division by zeros.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/Checkers/DivZeroChecker.h"

using namespace clang;

void *DivZeroChecker::getTag() {
  static int x;
  return &x;
}

void DivZeroChecker::PreVisitBinaryOperator(CheckerContext &C,
                                            const BinaryOperator *B) {
  BinaryOperator::Opcode Op = B->getOpcode();
  if (Op != BinaryOperator::Div &&
      Op != BinaryOperator::Rem &&
      Op != BinaryOperator::DivAssign &&
      Op != BinaryOperator::RemAssign)
    return;

  if (!B->getRHS()->getType()->isIntegerType() ||
      !B->getRHS()->getType()->isScalarType())
    return;

  SVal Denom = C.getState()->getSVal(B->getRHS());
  const DefinedSVal *DV = dyn_cast<DefinedSVal>(&Denom);

  // Divide-by-undefined handled in the generic checking for uses of
  // undefined values.
  if (!DV)
    return;

  // Check for divide by zero.
  ConstraintManager &CM = C.getConstraintManager();
  const GRState *stateNotZero, *stateZero;
  llvm::tie(stateNotZero, stateZero) = CM.AssumeDual(C.getState(), *DV);

  if (stateZero && !stateNotZero) {
    if (ExplodedNode *N = C.GenerateNode(B, stateZero, true)) {
      if (!BT)
        BT = new BuiltinBug("Division by zero");

      EnhancedBugReport *R = 
        new EnhancedBugReport(*BT, BT->getDescription().c_str(), N);

      R->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue,
                           bugreporter::GetDenomExpr(N));

      C.EmitReport(R);
    }
    return;
  }

  // If we get here, then the denom should not be zero. We abandon the implicit
  // zero denom case for now.
  if (stateNotZero != C.getState())
    C.addTransition(C.GenerateNode(B, stateNotZero));
}
