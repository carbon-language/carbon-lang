//===--- PthreadLockChecker.h - Undefined arguments checker ----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines PthreadLockChecker, a simple lock -> unlock checker.  Eventually
// this shouldn't be registered with ExprEngineInternalChecks.
//
//===----------------------------------------------------------------------===//

#include "clang/GR/PathSensitive/CheckerVisitor.h"
#include "clang/GR/BugReporter/BugReporter.h"
#include "clang/GR/PathSensitive/GRStateTrait.h"
#include "ExprEngineExperimentalChecks.h"
#include "llvm/ADT/ImmutableSet.h"

using namespace clang;
using namespace ento;

namespace {
class PthreadLockChecker
  : public CheckerVisitor<PthreadLockChecker> {
  BugType *BT;
public:
  PthreadLockChecker() : BT(0) {}
  static void *getTag() {
    static int x = 0;
    return &x;
  }
  void PostVisitCallExpr(CheckerContext &C, const CallExpr *CE);
    
  void AcquireLock(CheckerContext &C, const CallExpr *CE,
                   SVal lock, bool isTryLock);
    
  void ReleaseLock(CheckerContext &C, const CallExpr *CE,
                    SVal lock);

};
} // end anonymous namespace

// GDM Entry for tracking lock state.
namespace { class LockSet {}; }
namespace clang {
namespace ento {
template <> struct GRStateTrait<LockSet> :
  public GRStatePartialTrait<llvm::ImmutableSet<const MemRegion*> > {
    static void* GDMIndex() { return PthreadLockChecker::getTag(); }
};
} // end GR namespace
} // end clang namespace

void ento::RegisterPthreadLockChecker(ExprEngine &Eng) {
  Eng.registerCheck(new PthreadLockChecker());
}


void PthreadLockChecker::PostVisitCallExpr(CheckerContext &C,
                                           const CallExpr *CE) {
  const GRState *state = C.getState();
  const Expr *Callee = CE->getCallee();
  const FunctionTextRegion *R =
    dyn_cast_or_null<FunctionTextRegion>(state->getSVal(Callee).getAsRegion());
  
  if (!R)
    return;
  
  IdentifierInfo *II = R->getDecl()->getIdentifier();
  if (!II)   // if no identifier, not a simple C function
    return;
  llvm::StringRef FName = II->getName();
  
  if (FName == "pthread_mutex_lock") {
    if (CE->getNumArgs() != 1)
      return;
    AcquireLock(C, CE, state->getSVal(CE->getArg(0)), false);
  }
  else if (FName == "pthread_mutex_trylock") {
    if (CE->getNumArgs() != 1)
      return;
    AcquireLock(C, CE, state->getSVal(CE->getArg(0)), true);
  }  
  else if (FName == "pthread_mutex_unlock") {
    if (CE->getNumArgs() != 1)
      return;
    ReleaseLock(C, CE, state->getSVal(CE->getArg(0)));
  }
}

void PthreadLockChecker::AcquireLock(CheckerContext &C, const CallExpr *CE,
                                     SVal lock, bool isTryLock) {
  
  const MemRegion *lockR = lock.getAsRegion();
  if (!lockR)
    return;
  
  const GRState *state = C.getState();
  
  SVal X = state->getSVal(CE);
  if (X.isUnknownOrUndef())
    return;
  
  DefinedSVal retVal = cast<DefinedSVal>(X);
  const GRState *lockSucc = state;
  
  if (isTryLock) {
      // Bifurcate the state, and allow a mode where the lock acquisition fails.
    const GRState *lockFail;
    llvm::tie(lockFail, lockSucc) = state->assume(retVal);    
    assert(lockFail && lockSucc);
    C.addTransition(C.generateNode(CE, lockFail));
  }
  else {
      // Assume that the return value was 0.
    lockSucc = state->assume(retVal, false);
    assert(lockSucc);
  }
  
    // Record that the lock was acquired.  
  lockSucc = lockSucc->add<LockSet>(lockR);
  
  C.addTransition(lockSucc != state ? C.generateNode(CE, lockSucc) :
                  C.getPredecessor());
}

void PthreadLockChecker::ReleaseLock(CheckerContext &C, const CallExpr *CE,
                                     SVal lock) {

  const MemRegion *lockR = lock.getAsRegion();
  if (!lockR)
    return;
  
  const GRState *state = C.getState();

  // Record that the lock was released.  
  // FIXME: Handle unlocking locks that were never acquired.  This may
  // require IPA for wrappers.
  const GRState *unlockState = state->remove<LockSet>(lockR);
  
  if (state == unlockState)
    return;
  
  C.addTransition(C.generateNode(CE, unlockState));  
}
