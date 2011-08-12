//===--- PthreadLockChecker.cpp - Check for locking problems ---*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines PthreadLockChecker, a simple lock -> unlock checker.
// Also handles XNU locks, which behave similarly enough to share code.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/GRStateTrait.h"
#include "llvm/ADT/ImmutableList.h"

using namespace clang;
using namespace ento;

namespace {
class PthreadLockChecker : public Checker< check::PostStmt<CallExpr> > {
  mutable llvm::OwningPtr<BugType> BT_doublelock;
  mutable llvm::OwningPtr<BugType> BT_lor;
  enum LockingSemantics {
    NotApplicable = 0,
    PthreadSemantics,
    XNUSemantics
  };
public:
  void checkPostStmt(const CallExpr *CE, CheckerContext &C) const;
    
  void AcquireLock(CheckerContext &C, const CallExpr *CE, SVal lock,
                   bool isTryLock, enum LockingSemantics semantics) const;
    
  void ReleaseLock(CheckerContext &C, const CallExpr *CE, SVal lock) const;
};
} // end anonymous namespace

// GDM Entry for tracking lock state.
namespace { class LockSet {}; }
namespace clang {
namespace ento {
template <> struct GRStateTrait<LockSet> :
  public GRStatePartialTrait<llvm::ImmutableList<const MemRegion*> > {
    static void *GDMIndex() { static int x = 0; return &x; }
};
} // end GR namespace
} // end clang namespace


void PthreadLockChecker::checkPostStmt(const CallExpr *CE,
                                       CheckerContext &C) const {
  const GRState *state = C.getState();
  const Expr *Callee = CE->getCallee();
  const FunctionDecl *FD = state->getSVal(Callee).getAsFunctionDecl();

  if (!FD)
    return;

  // Get the name of the callee.
  IdentifierInfo *II = FD->getIdentifier();
  if (!II)   // if no identifier, not a simple C function
    return;
  StringRef FName = II->getName();

  if (CE->getNumArgs() != 1)
    return;

  if (FName == "pthread_mutex_lock" ||
      FName == "pthread_rwlock_rdlock" ||
      FName == "pthread_rwlock_wrlock")
    AcquireLock(C, CE, state->getSVal(CE->getArg(0)), false, PthreadSemantics);
  else if (FName == "lck_mtx_lock" ||
           FName == "lck_rw_lock_exclusive" ||
           FName == "lck_rw_lock_shared") 
    AcquireLock(C, CE, state->getSVal(CE->getArg(0)), false, XNUSemantics);
  else if (FName == "pthread_mutex_trylock" ||
           FName == "pthread_rwlock_tryrdlock" ||
           FName == "pthread_rwlock_tryrwlock")
    AcquireLock(C, CE, state->getSVal(CE->getArg(0)), true, PthreadSemantics);
  else if (FName == "lck_mtx_try_lock" ||
           FName == "lck_rw_try_lock_exclusive" ||
           FName == "lck_rw_try_lock_shared")
    AcquireLock(C, CE, state->getSVal(CE->getArg(0)), true, XNUSemantics);
  else if (FName == "pthread_mutex_unlock" ||
           FName == "pthread_rwlock_unlock" ||
           FName == "lck_mtx_unlock" ||
           FName == "lck_rw_done")
    ReleaseLock(C, CE, state->getSVal(CE->getArg(0)));
}

void PthreadLockChecker::AcquireLock(CheckerContext &C, const CallExpr *CE,
                                     SVal lock, bool isTryLock,
                                     enum LockingSemantics semantics) const {
  
  const MemRegion *lockR = lock.getAsRegion();
  if (!lockR)
    return;
  
  const GRState *state = C.getState();
  
  SVal X = state->getSVal(CE);
  if (X.isUnknownOrUndef())
    return;
  
  DefinedSVal retVal = cast<DefinedSVal>(X);

  if (state->contains<LockSet>(lockR)) {
    if (!BT_doublelock)
      BT_doublelock.reset(new BugType("Double locking", "Lock checker"));
    ExplodedNode *N = C.generateSink();
    if (!N)
      return;
    EnhancedBugReport *report = new EnhancedBugReport(*BT_doublelock,
                                                      "This lock has already "
                                                      "been acquired", N);
    report->addRange(CE->getArg(0)->getSourceRange());
    C.EmitReport(report);
    return;
  }

  const GRState *lockSucc = state;
  if (isTryLock) {
    // Bifurcate the state, and allow a mode where the lock acquisition fails.
    const GRState *lockFail;
    switch (semantics) {
    case PthreadSemantics:
      llvm::tie(lockFail, lockSucc) = state->assume(retVal);    
      break;
    case XNUSemantics:
      llvm::tie(lockSucc, lockFail) = state->assume(retVal);    
      break;
    default:
      llvm_unreachable("Unknown tryLock locking semantics");
      break;
    }
    assert(lockFail && lockSucc);
    C.addTransition(lockFail);

  } else if (semantics == PthreadSemantics) {
    // Assume that the return value was 0.
    lockSucc = state->assume(retVal, false);
    assert(lockSucc);

  } else {
    // XNU locking semantics return void on non-try locks
    assert((semantics == XNUSemantics) && "Unknown locking semantics");
    lockSucc = state;
  }
  
  // Record that the lock was acquired.  
  lockSucc = lockSucc->add<LockSet>(lockR);
  C.addTransition(lockSucc);
}

void PthreadLockChecker::ReleaseLock(CheckerContext &C, const CallExpr *CE,
                                     SVal lock) const {

  const MemRegion *lockR = lock.getAsRegion();
  if (!lockR)
    return;
  
  const GRState *state = C.getState();
  llvm::ImmutableList<const MemRegion*> LS = state->get<LockSet>();

  // FIXME: Better analysis requires IPA for wrappers.
  // FIXME: check for double unlocks
  if (LS.isEmpty())
    return;
  
  const MemRegion *firstLockR = LS.getHead();
  if (firstLockR != lockR) {
    if (!BT_lor)
      BT_lor.reset(new BugType("Lock order reversal", "Lock checker"));
    ExplodedNode *N = C.generateSink();
    if (!N)
      return;
    EnhancedBugReport *report = new EnhancedBugReport(*BT_lor,
                                                      "This was not the most "
                                                      "recently acquired lock. "
                                                      "Possible lock order "
                                                      "reversal", N);
    report->addRange(CE->getArg(0)->getSourceRange());
    C.EmitReport(report);
    return;
  }

  // Record that the lock was released. 
  state = state->set<LockSet>(LS.getTail());
  C.addTransition(state);
}


void ento::registerPthreadLockChecker(CheckerManager &mgr) {
  mgr.registerChecker<PthreadLockChecker>();
}
