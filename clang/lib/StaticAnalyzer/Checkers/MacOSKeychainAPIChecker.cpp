//==--- MacOSKeychainAPIChecker.cpp ------------------------------*- C++ -*-==//
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
  /// Stores the information about the allocator and deallocator functions -
  /// these are the functions the checker is tracking.
  struct ADFunctionInfo {
    const char* Name;
    unsigned int Param;
    unsigned int DeallocatorIdx;
  };
  static const unsigned InvalidIdx = 100000;
  static const unsigned FunctionsToTrackSize = 6;
  static const ADFunctionInfo FunctionsToTrack[FunctionsToTrackSize];

  /// Given the function name, returns the index of the allocator/deallocator
  /// function.
  unsigned getTrackedFunctionIndex(StringRef Name, bool IsAllocator) const;

  inline void initBugType() const {
    if (!BT)
      BT.reset(new BugType("Improper use of SecKeychain API", "Mac OS API"));
  }
};
}

/// AllocationState is a part of the checker specific state together with the
/// MemRegion corresponding to the allocated data.
struct AllocationState {
  const Expr *Address;
  /// The index of the allocator function.
  unsigned int AllocatorIdx;

  AllocationState(const Expr *E, unsigned int Idx) : Address(E),
                                                     AllocatorIdx(Idx) {}
  bool operator==(const AllocationState &X) const {
    return Address == X.Address;
  }
  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddPointer(Address);
    ID.AddInteger(AllocatorIdx);
  }
};

/// GRState traits to store the currently allocated (and not yet freed) symbols.
typedef llvm::ImmutableMap<const MemRegion*, AllocationState> AllocatedSetTy;

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

const MacOSKeychainAPIChecker::ADFunctionInfo
  MacOSKeychainAPIChecker::FunctionsToTrack[FunctionsToTrackSize] = {
    {"SecKeychainItemCopyContent", 4, 3},                       // 0
    {"SecKeychainFindGenericPassword", 6, 3},                   // 1
    {"SecKeychainFindInternetPassword", 13, 3},                 // 2
    {"SecKeychainItemFreeContent", 1, InvalidIdx},              // 3
    {"SecKeychainItemCopyAttributesAndData", 5, 5},             // 4
    {"SecKeychainItemFreeAttributesAndData", 1, InvalidIdx},    // 5
};

unsigned MacOSKeychainAPIChecker::getTrackedFunctionIndex(StringRef Name,
                                                       bool IsAllocator) const {
  for (unsigned I = 0; I < FunctionsToTrackSize; ++I) {
    ADFunctionInfo FI = FunctionsToTrack[I];
    if (FI.Name != Name)
      continue;
    // Make sure the function is of the right type (allocator vs deallocator).
    if (IsAllocator && (FI.DeallocatorIdx == InvalidIdx))
      return InvalidIdx;
    if (!IsAllocator && (FI.DeallocatorIdx != InvalidIdx))
      return InvalidIdx;

    return I;
  }
  // The function is not tracked.
  return InvalidIdx;
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
  unsigned idx = getTrackedFunctionIndex(funName, false);
  if (idx == InvalidIdx)
    return;

  const Expr *ArgExpr = CE->getArg(FunctionsToTrack[idx].Param);
  const MemRegion *Arg = State->getSVal(ArgExpr).getAsRegion();
  if (!Arg)
    return;

  // If trying to free data which has not been allocated yet, report as bug.
  const AllocationState *AS = State->get<AllocatedData>(Arg);
  if (!AS) {
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
    return;
  }

  // Check if the proper deallocator is used.
  unsigned int PDeallocIdx = FunctionsToTrack[AS->AllocatorIdx].DeallocatorIdx;
  if (PDeallocIdx != idx) {
    ExplodedNode *N = C.generateSink(State);
    if (!N)
      return;
    initBugType();

    std::string sbuf;
    llvm::raw_string_ostream os(sbuf);
    os << "Allocator doesn't match the deallocator: '"
       << FunctionsToTrack[PDeallocIdx].Name << "' should be used.";
    RangedBugReport *Report = new RangedBugReport(*BT, os.str(), N);
    Report->addRange(ArgExpr->getSourceRange());
    C.EmitReport(Report);
    return;
  }

  // Continue exploring from the new state.
  State = State->remove<AllocatedData>(Arg);
  C.addTransition(State);
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
  unsigned idx = getTrackedFunctionIndex(funName, true);
  if (idx == InvalidIdx)
    return;

  const Expr *ArgExpr = CE->getArg(FunctionsToTrack[idx].Param);
  SVal Arg = State->getSVal(ArgExpr);
  if (const loc::MemRegionVal *X = dyn_cast<loc::MemRegionVal>(&Arg)) {
    // Add the symbolic value, which represents the location of the allocated
    // data, to the set.
    const MemRegion *V = SM.Retrieve(State->getStore(), *X).getAsRegion();
    // If this is not a region, it can be:
    //  - unknown (cannot reason about it)
    //  - undefined (already reported by other checker)
    //  - constant (null - should not be tracked,
    //              other constant will generate a compiler warning)
    //  - goto (should be reported by other checker)
    if (!V)
      return;

    // We only need to track the value if the function returned noErr(0), so
    // bind the return value of the function to 0 and proceed from the no error
    // state.
    SValBuilder &Builder = C.getSValBuilder();
    SVal ZeroVal = Builder.makeIntVal(0, CE->getCallReturnType());
    const GRState *NoErr = State->BindExpr(CE, ZeroVal);
    NoErr = NoErr->set<AllocatedData>(V, AllocationState(ArgExpr, idx));
    assert(NoErr);
    C.addTransition(NoErr);

    // Generate a transition to explore the state space when there is an error.
    // In this case, we do not track the allocated data.
    SVal ReturnedError = Builder.evalBinOpNN(State, BO_NE,
                                             cast<NonLoc>(ZeroVal),
                                             cast<NonLoc>(State->getSVal(CE)),
                                             CE->getCallReturnType());
    const GRState *Err = State->assume(cast<NonLoc>(ReturnedError), true);
    assert(Err);
    C.addTransition(Err);
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
    const ADFunctionInfo &FI = FunctionsToTrack[I->second.AllocatorIdx];

    std::string sbuf;
    llvm::raw_string_ostream os(sbuf);
    os << "Allocated data is not released: missing a call to '"
       << FunctionsToTrack[FI.DeallocatorIdx].Name << "'.";
    RangedBugReport *Report = new RangedBugReport(*BT, os.str(), N);
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
