//==--- MacOSKeychainAPIChecker.cpp -----------------------------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This checker flags misuses of KeyChainAPI. In particular, the password data
// allocated/returned by SecKeychainItemCopyContent,
// SecKeychainFindGenericPassword, SecKeychainFindInternetPassword functions has
// to be freed using a call to SecKeychainItemFreeContent.
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/GRState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/GRStateTrait.h"

using namespace clang;
using namespace ento;

namespace {
class MacOSKeychainAPIChecker : public Checker<check::PreStmt<CallExpr>,
                                               check::PreStmt<ReturnStmt>,
                                               check::PostStmt<CallExpr>,
                                               check::EndPath > {
  mutable llvm::OwningPtr<BugType> BT;

public:
  void checkPreStmt(const CallExpr *S, CheckerContext &C) const;
  void checkPreStmt(const ReturnStmt *S, CheckerContext &C) const;
  void checkPostStmt(const CallExpr *S, CheckerContext &C) const;

  void checkEndPath(EndOfFunctionNodeBuilder &B, ExprEngine &Eng) const;

private:
  static const unsigned InvalidParamVal = 100000;

  /// Given the function name, returns the index of the parameter which will
  /// be allocated as a result of the call.
  unsigned getAllocatingFunctionParam(StringRef Name) const {
    if (Name == "SecKeychainItemCopyContent")
      return 4;
    if (Name == "SecKeychainFindGenericPassword")
      return 6;
    if (Name == "SecKeychainFindInternetPassword")
      return 13;
    return InvalidParamVal;
  }

  /// Given the function name, returns the index of the parameter which will
  /// be freed by the function.
  unsigned getDeallocatingFunctionParam(StringRef Name) const {
    if (Name == "SecKeychainItemFreeContent")
      return 1;
    return InvalidParamVal;
  }

  inline void initBugType() const {
    if (!BT)
      BT.reset(new BugType("Improper use of SecKeychain API", "Mac OS API"));
  }
};
}

struct AllocationInfo {
  const Expr *Address;

  AllocationInfo(const Expr *E) : Address(E) {}
  bool operator==(const AllocationInfo &X) const {
    return Address == X.Address;
  }
  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddPointer(Address);
  }
};

// GRState traits to store the currently allocated (and not yet freed) symbols.
typedef llvm::ImmutableMap<const MemRegion*, AllocationInfo> AllocatedSetTy;

namespace { struct AllocatedData {}; }
namespace clang { namespace ento {
template<> struct GRStateTrait<AllocatedData>
    :  public GRStatePartialTrait<AllocatedSetTy > {
  static void *GDMIndex() { static int index = 0; return &index; }
};
}}

static bool isEnclosingFunctionParam(const Expr *E) {
  E = E->IgnoreParenCasts();
  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    const ValueDecl *VD = DRE->getDecl();
    if (isa<ImplicitParamDecl>(VD) || isa<ParmVarDecl>(VD))
      return true;
  }
  return false;
}

void MacOSKeychainAPIChecker::checkPreStmt(const CallExpr *CE,
                                           CheckerContext &C) const {
  const GRState *State = C.getState();
  const Expr *Callee = CE->getCallee();
  SVal L = State->getSVal(Callee);

  const FunctionDecl *funDecl = L.getAsFunctionDecl();
  if (!funDecl)
    return;
  IdentifierInfo *funI = funDecl->getIdentifier();
  if (!funI)
    return;
  StringRef funName = funI->getName();

  // If a value has been freed, remove from the list.
  unsigned idx = getDeallocatingFunctionParam(funName);
  if (idx != InvalidParamVal) {
    const Expr *ArgExpr = CE->getArg(idx);
    const MemRegion *Arg = State->getSVal(ArgExpr).getAsRegion();
    if (!Arg)
      return;

    // If trying to free data which has not been allocated yet, report as bug.
    if (State->get<AllocatedData>(Arg) == 0) {
      // It is possible that this is a false positive - the argument might
      // have entered as an enclosing function parameter.
      if (isEnclosingFunctionParam(ArgExpr))
        return;

      ExplodedNode *N = C.generateNode(State);
      if (!N)
        return;
      initBugType();
      RangedBugReport *Report = new RangedBugReport(*BT,
          "Trying to free data which has not been allocated.", N);
      Report->addRange(ArgExpr->getSourceRange());
      C.EmitReport(Report);
    }

    // Continue exploring from the new state.
    State = State->remove<AllocatedData>(Arg);
    C.addTransition(State);
  }
}

void MacOSKeychainAPIChecker::checkPostStmt(const CallExpr *CE,
                                            CheckerContext &C) const {
  const GRState *State = C.getState();
  const Expr *Callee = CE->getCallee();
  SVal L = State->getSVal(Callee);
  StoreManager& SM = C.getStoreManager();

  const FunctionDecl *funDecl = L.getAsFunctionDecl();
  if (!funDecl)
    return;
  IdentifierInfo *funI = funDecl->getIdentifier();
  if (!funI)
    return;
  StringRef funName = funI->getName();

  // If a value has been allocated, add it to the set for tracking.
  unsigned idx = getAllocatingFunctionParam(funName);
  if (idx != InvalidParamVal) {
    SVal Arg = State->getSVal(CE->getArg(idx));
    if (const loc::MemRegionVal *X = dyn_cast<loc::MemRegionVal>(&Arg)) {
      // Add the symbolic value, which represents the location of the allocated
      // data, to the set.
      const MemRegion *V = SM.Retrieve(State->getStore(), *X).getAsRegion();
      // If this is not a region, it can be:
      //  - unknown (cannot reason about it)
      //  - undefined (already reported by other checker)
      //  - constant (null - should not be tracked, other - report a warning?)
      //  - goto (should be reported by other checker)
      if (!V)
        return;

      State = State->set<AllocatedData>(V, AllocationInfo(CE->getArg(idx)));

      // We only need to track the value if the function returned noErr(0), so
      // bind the return value of the function to 0.
      SValBuilder &Builder = C.getSValBuilder();
      SVal ZeroVal = Builder.makeZeroVal(Builder.getContext().CharTy);
      State = State->BindExpr(CE, ZeroVal);
      assert(State);

      // Proceed from the new state.
      C.addTransition(State);
    }
  }
}

void MacOSKeychainAPIChecker::checkPreStmt(const ReturnStmt *S,
                                           CheckerContext &C) const {
  const Expr *retExpr = S->getRetValue();
  if (!retExpr)
    return;

  // Check  if the value is escaping through the return.
  const GRState *state = C.getState();
  const MemRegion *V = state->getSVal(retExpr).getAsRegion();
  if (!V)
    return;
  state = state->remove<AllocatedData>(V);

  // Proceed from the new state.
  C.addTransition(state);
}

void MacOSKeychainAPIChecker::checkEndPath(EndOfFunctionNodeBuilder &B,
                                           ExprEngine &Eng) const {
  const GRState *state = B.getState();
  AllocatedSetTy AS = state->get<AllocatedData>();
  ExplodedNode *N = B.generateNode(state);
  if (!N)
    return;
  initBugType();

  // Anything which has been allocated but not freed (nor escaped) will be
  // found here, so report it.
  for (AllocatedSetTy::iterator I = AS.begin(), E = AS.end(); I != E; ++I ) {
    RangedBugReport *Report = new RangedBugReport(*BT,
      "Missing a call to SecKeychainItemFreeContent.", N);
    // TODO: The report has to mention the expression which contains the
    // allocated content as well as the point at which it has been allocated.
    // Currently, the next line is useless.
    Report->addRange(I->second.Address->getSourceRange());
    Eng.getBugReporter().EmitReport(Report);
  }
}

void ento::registerMacOSKeychainAPIChecker(CheckerManager &mgr) {
  mgr.registerChecker<MacOSKeychainAPIChecker>();
}
