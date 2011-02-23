//=== PointerArithChecker.cpp - Pointer arithmetic checker -----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This files defines PointerArithChecker, a builtin checker that checks for
// pointer arithmetic on locations other than array elements.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/CheckerV2.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"

using namespace clang;
using namespace ento;

namespace {
class PointerArithChecker 
  : public CheckerV2< check::PreStmt<BinaryOperator> > {
  mutable llvm::OwningPtr<BuiltinBug> BT;

public:
  void checkPreStmt(const BinaryOperator *B, CheckerContext &C) const;
};
}

void PointerArithChecker::checkPreStmt(const BinaryOperator *B,
                                       CheckerContext &C) const {
  if (B->getOpcode() != BO_Sub && B->getOpcode() != BO_Add)
    return;

  const GRState *state = C.getState();
  SVal LV = state->getSVal(B->getLHS());
  SVal RV = state->getSVal(B->getRHS());

  const MemRegion *LR = LV.getAsRegion();

  if (!LR || !RV.isConstant())
    return;

  // If pointer arithmetic is done on variables of non-array type, this often
  // means behavior rely on memory organization, which is dangerous.
  if (isa<VarRegion>(LR) || isa<CodeTextRegion>(LR) || 
      isa<CompoundLiteralRegion>(LR)) {

    if (ExplodedNode *N = C.generateNode()) {
      if (!BT)
        BT.reset(new BuiltinBug("Dangerous pointer arithmetic",
                            "Pointer arithmetic done on non-array variables "
                            "means reliance on memory layout, which is "
                            "dangerous."));
      RangedBugReport *R = new RangedBugReport(*BT, BT->getDescription(), N);
      R->addRange(B->getSourceRange());
      C.EmitReport(R);
    }
  }
}

void ento::registerPointerArithChecker(CheckerManager &mgr) {
  mgr.registerChecker<PointerArithChecker>();
}
