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
};
}

// GRState traits to store the currently allocated (and not yet freed) symbols.
typedef llvm::ImmutableSet<SymbolRef> AllocatedSetTy;

namespace { struct AllocatedData {}; }
namespace clang { namespace ento {
template<> struct GRStateTrait<AllocatedData>
    :  public GRStatePartialTrait<AllocatedSetTy > {
  static void *GDMIndex() { static int index = 0; return &index; }
};
}}

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
    SymbolRef Param = State->getSVal(CE->getArg(idx)).getAsSymbol();
    if (!Param)
      return;
    if (!State->contains<AllocatedData>(Param)) {
      // TODO: do we care about this?
      assert(0 && "Trying to free data which has not been allocated yet.");
    }
    State = State->remove<AllocatedData>(Param);
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
    SVal Param = State->getSVal(CE->getArg(idx));
    if (const loc::MemRegionVal *X = dyn_cast<loc::MemRegionVal>(&Param)) {
      // Add the symbolic value, which represents the location of the allocated
      // data, to the set.
      SymbolRef V = SM.Retrieve(State->getStore(), *X).getAsSymbol();
      if (!V)
        return;
      State = State->add<AllocatedData>(V);

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
  SymbolRef V = state->getSVal(retExpr).getAsSymbol();
  if (!V)
    return;
  state->remove<AllocatedData>(V);

}

void MacOSKeychainAPIChecker::checkEndPath(EndOfFunctionNodeBuilder &B,
                                 ExprEngine &Eng) const {
  const GRState *state = B.getState();
  AllocatedSetTy AS = state->get<AllocatedData>();

  // Anything which has been allocated but not freed (nor escaped) will be
  // found here, so report it.
  if (!AS.isEmpty()) {
    assert(0 && "TODO: Report the bug here.");
  }
}

void ento::registerMacOSKeychainAPIChecker(CheckerManager &mgr) {
  mgr.registerChecker<MacOSKeychainAPIChecker>();
}
