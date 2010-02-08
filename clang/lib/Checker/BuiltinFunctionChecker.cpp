//=== BuiltinFunctionChecker.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This checker evaluates clang builtin functions.
//
//===----------------------------------------------------------------------===//

#include "GRExprEngineInternalChecks.h"
#include "clang/Checker/PathSensitive/Checker.h"
#include "clang/Basic/Builtins.h"
#include "llvm/ADT/StringSwitch.h"

using namespace clang;

namespace {

class BuiltinFunctionChecker : public Checker {
public:
  static void *getTag() { static int tag = 0; return &tag; }
  virtual bool EvalCallExpr(CheckerContext &C, const CallExpr *CE);
};

}

void clang::RegisterBuiltinFunctionChecker(GRExprEngine &Eng) {
  Eng.registerCheck(new BuiltinFunctionChecker());
}

bool BuiltinFunctionChecker::EvalCallExpr(CheckerContext &C,const CallExpr *CE){
  const GRState *state = C.getState();
  const Expr *Callee = CE->getCallee();
  SVal L = state->getSVal(Callee);
  const FunctionDecl *FD = L.getAsFunctionDecl();

  if (!FD)
    return false;

  unsigned id = FD->getBuiltinID();

  if (!id)
    return false;

  switch (id) {
  case Builtin::BI__builtin_expect: {
    // For __builtin_expect, just return the value of the subexpression.
    assert (CE->arg_begin() != CE->arg_end());
    SVal X = state->getSVal(*(CE->arg_begin()));
    C.GenerateNode(state->BindExpr(CE, X));
    return true;
  }

  case Builtin::BI__builtin_alloca: {
    // FIXME: Refactor into StoreManager itself?
    MemRegionManager& RM = C.getStoreManager().getRegionManager();
    const MemRegion* R =
      RM.getAllocaRegion(CE, C.getNodeBuilder().getCurrentBlockCount(),
                         C.getPredecessor()->getLocationContext());

    // Set the extent of the region in bytes. This enables us to use the
    // SVal of the argument directly. If we save the extent in bits, we
    // cannot represent values like symbol*8.
    SVal Extent = state->getSVal(*(CE->arg_begin()));
    state = C.getStoreManager().setExtent(state, R, Extent);
    C.GenerateNode(state->BindExpr(CE, loc::MemRegionVal(R)));
    return true;
  }
  }

  return false;
}
