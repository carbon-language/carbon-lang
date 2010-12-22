//=== OSAtomicChecker.cpp - OSAtomic functions evaluator --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This checker evaluates OSAtomic functions.
//
//===----------------------------------------------------------------------===//

#include "ExprEngineInternalChecks.h"
#include "clang/GR/PathSensitive/Checker.h"
#include "clang/Basic/Builtins.h"

using namespace clang;
using namespace GR;

namespace {

class OSAtomicChecker : public Checker {
public:
  static void *getTag() { static int tag = 0; return &tag; }
  virtual bool evalCallExpr(CheckerContext &C, const CallExpr *CE);

private:
  bool evalOSAtomicCompareAndSwap(CheckerContext &C, const CallExpr *CE);
};

}

void GR::RegisterOSAtomicChecker(ExprEngine &Eng) {
  Eng.registerCheck(new OSAtomicChecker());
}

bool OSAtomicChecker::evalCallExpr(CheckerContext &C,const CallExpr *CE) {
  const GRState *state = C.getState();
  const Expr *Callee = CE->getCallee();
  SVal L = state->getSVal(Callee);

  const FunctionDecl* FD = L.getAsFunctionDecl();
  if (!FD)
    return false;

  const IdentifierInfo *II = FD->getIdentifier();
  if (!II)
    return false;
  
  llvm::StringRef FName(II->getName());

  // Check for compare and swap.
  if (FName.startswith("OSAtomicCompareAndSwap") ||
      FName.startswith("objc_atomicCompareAndSwap"))
    return evalOSAtomicCompareAndSwap(C, CE);

  // FIXME: Other atomics.
  return false;
}

bool OSAtomicChecker::evalOSAtomicCompareAndSwap(CheckerContext &C, 
                                                 const CallExpr *CE) {
  // Not enough arguments to match OSAtomicCompareAndSwap?
  if (CE->getNumArgs() != 3)
    return false;

  ASTContext &Ctx = C.getASTContext();
  const Expr *oldValueExpr = CE->getArg(0);
  QualType oldValueType = Ctx.getCanonicalType(oldValueExpr->getType());

  const Expr *newValueExpr = CE->getArg(1);
  QualType newValueType = Ctx.getCanonicalType(newValueExpr->getType());

  // Do the types of 'oldValue' and 'newValue' match?
  if (oldValueType != newValueType)
    return false;

  const Expr *theValueExpr = CE->getArg(2);
  const PointerType *theValueType=theValueExpr->getType()->getAs<PointerType>();

  // theValueType not a pointer?
  if (!theValueType)
    return false;

  QualType theValueTypePointee =
    Ctx.getCanonicalType(theValueType->getPointeeType()).getUnqualifiedType();

  // The pointee must match newValueType and oldValueType.
  if (theValueTypePointee != newValueType)
    return false;

  static unsigned magic_load = 0;
  static unsigned magic_store = 0;

  const void *OSAtomicLoadTag = &magic_load;
  const void *OSAtomicStoreTag = &magic_store;

  // Load 'theValue'.
  ExprEngine &Engine = C.getEngine();
  const GRState *state = C.getState();
  ExplodedNodeSet Tmp;
  SVal location = state->getSVal(theValueExpr);
  // Here we should use the value type of the region as the load type, because
  // we are simulating the semantics of the function, not the semantics of 
  // passing argument. So the type of theValue expr is not we are loading.
  // But usually the type of the varregion is not the type we want either,
  // we still need to do a CastRetrievedVal in store manager. So actually this
  // LoadTy specifying can be omitted. But we put it here to emphasize the 
  // semantics.
  QualType LoadTy;
  if (const TypedRegion *TR =
      dyn_cast_or_null<TypedRegion>(location.getAsRegion())) {
    LoadTy = TR->getValueType();
  }
  Engine.evalLoad(Tmp, theValueExpr, C.getPredecessor(), 
                  state, location, OSAtomicLoadTag, LoadTy);

  if (Tmp.empty()) {
    // If no nodes were generated, other checkers must generated sinks. But 
    // since the builder state was restored, we set it manually to prevent 
    // auto transition.
    // FIXME: there should be a better approach.
    C.getNodeBuilder().BuildSinks = true;
    return true;
  }
 
  for (ExplodedNodeSet::iterator I = Tmp.begin(), E = Tmp.end();
       I != E; ++I) {

    ExplodedNode *N = *I;
    const GRState *stateLoad = N->getState();
    SVal theValueVal_untested = stateLoad->getSVal(theValueExpr);
    SVal oldValueVal_untested = stateLoad->getSVal(oldValueExpr);

    // FIXME: Issue an error.
    if (theValueVal_untested.isUndef() || oldValueVal_untested.isUndef()) {
      return false;
    }
    
    DefinedOrUnknownSVal theValueVal =
      cast<DefinedOrUnknownSVal>(theValueVal_untested);
    DefinedOrUnknownSVal oldValueVal =
      cast<DefinedOrUnknownSVal>(oldValueVal_untested);

    SValBuilder &svalBuilder = Engine.getSValBuilder();

    // Perform the comparison.
    DefinedOrUnknownSVal Cmp = svalBuilder.evalEQ(stateLoad,theValueVal,oldValueVal);

    const GRState *stateEqual = stateLoad->assume(Cmp, true);

    // Were they equal?
    if (stateEqual) {
      // Perform the store.
      ExplodedNodeSet TmpStore;
      SVal val = stateEqual->getSVal(newValueExpr);

      // Handle implicit value casts.
      if (const TypedRegion *R =
          dyn_cast_or_null<TypedRegion>(location.getAsRegion())) {
        val = svalBuilder.evalCast(val,R->getValueType(), newValueExpr->getType());
      }

      Engine.evalStore(TmpStore, NULL, theValueExpr, N, 
                       stateEqual, location, val, OSAtomicStoreTag);

      if (TmpStore.empty()) {
        // If no nodes were generated, other checkers must generated sinks. But 
        // since the builder state was restored, we set it manually to prevent 
        // auto transition.
        // FIXME: there should be a better approach.
        C.getNodeBuilder().BuildSinks = true;
        return true;
      }

      // Now bind the result of the comparison.
      for (ExplodedNodeSet::iterator I2 = TmpStore.begin(),
           E2 = TmpStore.end(); I2 != E2; ++I2) {
        ExplodedNode *predNew = *I2;
        const GRState *stateNew = predNew->getState();
        // Check for 'void' return type if we have a bogus function prototype.
        SVal Res = UnknownVal();
        QualType T = CE->getType();
        if (!T->isVoidType())
          Res = Engine.getSValBuilder().makeTruthVal(true, T);
        C.generateNode(stateNew->BindExpr(CE, Res), predNew);
      }
    }

    // Were they not equal?
    if (const GRState *stateNotEqual = stateLoad->assume(Cmp, false)) {
      // Check for 'void' return type if we have a bogus function prototype.
      SVal Res = UnknownVal();
      QualType T = CE->getType();
      if (!T->isVoidType())
        Res = Engine.getSValBuilder().makeTruthVal(false, CE->getType());
      C.generateNode(stateNotEqual->BindExpr(CE, Res), N);
    }
  }

  return true;
}
