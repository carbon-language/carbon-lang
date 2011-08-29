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
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"

using namespace clang;
using namespace ento;

namespace {
class MacOSKeychainAPIChecker : public Checker<check::PreStmt<CallExpr>,
                                               check::PreStmt<ReturnStmt>,
                                               check::PostStmt<CallExpr>,
                                               check::EndPath,
                                               check::DeadSymbols> {
  mutable llvm::OwningPtr<BugType> BT;

public:
  /// AllocationState is a part of the checker specific state together with the
  /// MemRegion corresponding to the allocated data.
  struct AllocationState {
    /// The index of the allocator function.
    unsigned int AllocatorIdx;
    SymbolRef Region;

    AllocationState(const Expr *E, unsigned int Idx, SymbolRef R) :
      AllocatorIdx(Idx),
      Region(R) {}

    bool operator==(const AllocationState &X) const {
      return (AllocatorIdx == X.AllocatorIdx &&
              Region == X.Region);
    }

    void Profile(llvm::FoldingSetNodeID &ID) const {
      ID.AddInteger(AllocatorIdx);
      ID.AddPointer(Region);
    }
  };

  void checkPreStmt(const CallExpr *S, CheckerContext &C) const;
  void checkPreStmt(const ReturnStmt *S, CheckerContext &C) const;
  void checkPostStmt(const CallExpr *S, CheckerContext &C) const;
  void checkDeadSymbols(SymbolReaper &SR, CheckerContext &C) const;
  void checkEndPath(EndOfFunctionNodeBuilder &B, ExprEngine &Eng) const;

private:
  typedef std::pair<SymbolRef, const AllocationState*> AllocationPair;
  typedef llvm::SmallVector<AllocationPair, 2> AllocationPairVec;

  enum APIKind {
    /// Denotes functions tracked by this checker.
    ValidAPI = 0,
    /// The functions commonly/mistakenly used in place of the given API.
    ErrorAPI = 1,
    /// The functions which may allocate the data. These are tracked to reduce
    /// the false alarm rate.
    PossibleAPI = 2
  };
  /// Stores the information about the allocator and deallocator functions -
  /// these are the functions the checker is tracking.
  struct ADFunctionInfo {
    const char* Name;
    unsigned int Param;
    unsigned int DeallocatorIdx;
    APIKind Kind;
  };
  static const unsigned InvalidIdx = 100000;
  static const unsigned FunctionsToTrackSize = 8;
  static const ADFunctionInfo FunctionsToTrack[FunctionsToTrackSize];
  /// The value, which represents no error return value for allocator functions.
  static const unsigned NoErr = 0;

  /// Given the function name, returns the index of the allocator/deallocator
  /// function.
  static unsigned getTrackedFunctionIndex(StringRef Name, bool IsAllocator);

  inline void initBugType() const {
    if (!BT)
      BT.reset(new BugType("Improper use of SecKeychain API", "Mac OS API"));
  }

  void generateDeallocatorMismatchReport(const AllocationPair &AP,
                                         const Expr *ArgExpr,
                                         CheckerContext &C) const;

  BugReport *generateAllocatedDataNotReleasedReport(const AllocationPair &AP,
                                                    ExplodedNode *N) const;

  /// Check if RetSym evaluates to an error value in the current state.
  bool definitelyReturnedError(SymbolRef RetSym,
                               const ProgramState *State,
                               SValBuilder &Builder,
                               bool noError = false) const;

  /// Check if RetSym evaluates to a NoErr value in the current state.
  bool definitelyDidnotReturnError(SymbolRef RetSym,
                                   const ProgramState *State,
                                   SValBuilder &Builder) const {
    return definitelyReturnedError(RetSym, State, Builder, true);
  }

  /// The bug visitor which allows us to print extra diagnostics along the
  /// BugReport path. For example, showing the allocation site of the leaked
  /// region.
  class SecKeychainBugVisitor : public BugReporterVisitor {
  protected:
    // The allocated region symbol tracked by the main analysis.
    SymbolRef Sym;

  public:
    SecKeychainBugVisitor(SymbolRef S) : Sym(S) {}
    virtual ~SecKeychainBugVisitor() {}

    void Profile(llvm::FoldingSetNodeID &ID) const {
      static int X = 0;
      ID.AddPointer(&X);
      ID.AddPointer(Sym);
    }

    PathDiagnosticPiece *VisitNode(const ExplodedNode *N,
                                   const ExplodedNode *PrevN,
                                   BugReporterContext &BRC,
                                   BugReport &BR);
  };
};
}

/// ProgramState traits to store the currently allocated (and not yet freed)
/// symbols. This is a map from the allocated content symbol to the
/// corresponding AllocationState.
typedef llvm::ImmutableMap<SymbolRef,
                       MacOSKeychainAPIChecker::AllocationState> AllocatedSetTy;

namespace { struct AllocatedData {}; }
namespace clang { namespace ento {
template<> struct ProgramStateTrait<AllocatedData>
    :  public ProgramStatePartialTrait<AllocatedSetTy > {
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
    {"SecKeychainItemCopyContent", 4, 3, ValidAPI},                       // 0
    {"SecKeychainFindGenericPassword", 6, 3, ValidAPI},                   // 1
    {"SecKeychainFindInternetPassword", 13, 3, ValidAPI},                 // 2
    {"SecKeychainItemFreeContent", 1, InvalidIdx, ValidAPI},              // 3
    {"SecKeychainItemCopyAttributesAndData", 5, 5, ValidAPI},             // 4
    {"SecKeychainItemFreeAttributesAndData", 1, InvalidIdx, ValidAPI},    // 5
    {"free", 0, InvalidIdx, ErrorAPI},                                    // 6
    {"CFStringCreateWithBytesNoCopy", 1, InvalidIdx, PossibleAPI},        // 7
};

unsigned MacOSKeychainAPIChecker::getTrackedFunctionIndex(StringRef Name,
                                                          bool IsAllocator) {
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

static SymbolRef getSymbolForRegion(CheckerContext &C,
                                   const MemRegion *R) {
  // Implicit casts (ex: void* -> char*) can turn Symbolic region into element
  // region, if that is the case, get the underlining region.
  R = R->StripCasts();
  if (!isa<SymbolicRegion>(R)) {
      return 0;
  }
  return cast<SymbolicRegion>(R)->getSymbol();
}

static bool isBadDeallocationArgument(const MemRegion *Arg) {
  if (isa<AllocaRegion>(Arg) ||
      isa<BlockDataRegion>(Arg) ||
      isa<TypedRegion>(Arg)) {
    return true;
  }
  return false;
}
/// Given the address expression, retrieve the value it's pointing to. Assume
/// that value is itself an address, and return the corresponding symbol.
static SymbolRef getAsPointeeSymbol(const Expr *Expr,
                                    CheckerContext &C) {
  const ProgramState *State = C.getState();
  SVal ArgV = State->getSVal(Expr);

  if (const loc::MemRegionVal *X = dyn_cast<loc::MemRegionVal>(&ArgV)) {
    StoreManager& SM = C.getStoreManager();
    const MemRegion *V = SM.Retrieve(State->getStore(), *X).getAsRegion();
    if (V)
      return getSymbolForRegion(C, V);
  }
  return 0;
}

// When checking for error code, we need to consider the following cases:
// 1) noErr / [0]
// 2) someErr / [1, inf]
// 3) unknown
// If noError, returns true iff (1).
// If !noError, returns true iff (2).
bool MacOSKeychainAPIChecker::definitelyReturnedError(SymbolRef RetSym,
                                                      const ProgramState *State,
                                                      SValBuilder &Builder,
                                                      bool noError) const {
  DefinedOrUnknownSVal NoErrVal = Builder.makeIntVal(NoErr,
    Builder.getSymbolManager().getType(RetSym));
  DefinedOrUnknownSVal NoErr = Builder.evalEQ(State, NoErrVal,
                                                     nonloc::SymbolVal(RetSym));
  const ProgramState *ErrState = State->assume(NoErr, noError);
  if (ErrState == State) {
    return true;
  }

  return false;
}

// Report deallocator mismatch. Remove the region from tracking - reporting a
// missing free error after this one is redundant.
void MacOSKeychainAPIChecker::
  generateDeallocatorMismatchReport(const AllocationPair &AP,
                                    const Expr *ArgExpr,
                                    CheckerContext &C) const {
  const ProgramState *State = C.getState();
  State = State->remove<AllocatedData>(AP.first);
  ExplodedNode *N = C.generateNode(State);

  if (!N)
    return;
  initBugType();
  llvm::SmallString<80> sbuf;
  llvm::raw_svector_ostream os(sbuf);
  unsigned int PDeallocIdx =
               FunctionsToTrack[AP.second->AllocatorIdx].DeallocatorIdx;

  os << "Deallocator doesn't match the allocator: '"
     << FunctionsToTrack[PDeallocIdx].Name << "' should be used.";
  BugReport *Report = new BugReport(*BT, os.str(), N);
  Report->addVisitor(new SecKeychainBugVisitor(AP.first));
  Report->addRange(ArgExpr->getSourceRange());
  C.EmitReport(Report);
}

void MacOSKeychainAPIChecker::checkPreStmt(const CallExpr *CE,
                                           CheckerContext &C) const {
  const ProgramState *State = C.getState();
  const Expr *Callee = CE->getCallee();
  SVal L = State->getSVal(Callee);
  unsigned idx = InvalidIdx;

  const FunctionDecl *funDecl = L.getAsFunctionDecl();
  if (!funDecl)
    return;
  IdentifierInfo *funI = funDecl->getIdentifier();
  if (!funI)
    return;
  StringRef funName = funI->getName();

  // If it is a call to an allocator function, it could be a double allocation.
  idx = getTrackedFunctionIndex(funName, true);
  if (idx != InvalidIdx) {
    const Expr *ArgExpr = CE->getArg(FunctionsToTrack[idx].Param);
    if (SymbolRef V = getAsPointeeSymbol(ArgExpr, C))
      if (const AllocationState *AS = State->get<AllocatedData>(V)) {
        if (!definitelyReturnedError(AS->Region, State, C.getSValBuilder())) {
          // Remove the value from the state. The new symbol will be added for
          // tracking when the second allocator is processed in checkPostStmt().
          State = State->remove<AllocatedData>(V);
          ExplodedNode *N = C.generateNode(State);
          if (!N)
            return;
          initBugType();
          llvm::SmallString<128> sbuf;
          llvm::raw_svector_ostream os(sbuf);
          unsigned int DIdx = FunctionsToTrack[AS->AllocatorIdx].DeallocatorIdx;
          os << "Allocated data should be released before another call to "
              << "the allocator: missing a call to '"
              << FunctionsToTrack[DIdx].Name
              << "'.";
          BugReport *Report = new BugReport(*BT, os.str(), N);
          Report->addVisitor(new SecKeychainBugVisitor(V));
          Report->addRange(ArgExpr->getSourceRange());
          C.EmitReport(Report);
        }
      }
    return;
  }

  // Is it a call to one of deallocator functions?
  idx = getTrackedFunctionIndex(funName, false);
  if (idx == InvalidIdx)
    return;

  // Check the argument to the deallocator.
  const Expr *ArgExpr = CE->getArg(FunctionsToTrack[idx].Param);
  SVal ArgSVal = State->getSVal(ArgExpr);

  // Undef is reported by another checker.
  if (ArgSVal.isUndef())
    return;

  const MemRegion *Arg = ArgSVal.getAsRegion();
  if (!Arg)
    return;

  SymbolRef ArgSM = getSymbolForRegion(C, Arg);
  bool RegionArgIsBad = ArgSM ? false : isBadDeallocationArgument(Arg);
  // If the argument is coming from the heap, globals, or unknown, do not
  // report it.
  if (!ArgSM && !RegionArgIsBad)
    return;

  // Is the argument to the call being tracked?
  const AllocationState *AS = State->get<AllocatedData>(ArgSM);
  if (!AS && FunctionsToTrack[idx].Kind != ValidAPI) {
    return;
  }
  // If trying to free data which has not been allocated yet, report as a bug.
  // TODO: We might want a more precise diagnostic for double free
  // (that would involve tracking all the freed symbols in the checker state).
  if (!AS || RegionArgIsBad) {
    // It is possible that this is a false positive - the argument might
    // have entered as an enclosing function parameter.
    if (isEnclosingFunctionParam(ArgExpr))
      return;

    ExplodedNode *N = C.generateNode(State);
    if (!N)
      return;
    initBugType();
    BugReport *Report = new BugReport(*BT,
        "Trying to free data which has not been allocated.", N);
    Report->addRange(ArgExpr->getSourceRange());
    C.EmitReport(Report);
    return;
  }

  // Process functions which might deallocate.
  if (FunctionsToTrack[idx].Kind == PossibleAPI) {

    if (funName == "CFStringCreateWithBytesNoCopy") {
      const Expr *DeallocatorExpr = CE->getArg(5)->IgnoreParenCasts();
      // NULL ~ default deallocator, so warn.
      if (DeallocatorExpr->isNullPointerConstant(C.getASTContext(),
          Expr::NPC_ValueDependentIsNotNull)) {
        const AllocationPair AP = std::make_pair(ArgSM, AS);
        generateDeallocatorMismatchReport(AP, ArgExpr, C);
        return;
      }
      // One of the default allocators, so warn.
      if (const DeclRefExpr *DE = dyn_cast<DeclRefExpr>(DeallocatorExpr)) {
        StringRef DeallocatorName = DE->getFoundDecl()->getName();
        if (DeallocatorName == "kCFAllocatorDefault" ||
            DeallocatorName == "kCFAllocatorSystemDefault" ||
            DeallocatorName == "kCFAllocatorMalloc") {
          const AllocationPair AP = std::make_pair(ArgSM, AS);
          generateDeallocatorMismatchReport(AP, ArgExpr, C);
          return;
        }
        // If kCFAllocatorNull, which does not deallocate, we still have to
        // find the deallocator. Otherwise, assume that the user had written a
        // custom deallocator which does the right thing.
        if (DE->getFoundDecl()->getName() != "kCFAllocatorNull") {
          State = State->remove<AllocatedData>(ArgSM);
          C.addTransition(State);
          return;
        }
      }
    }
    return;
  }

  // The call is deallocating a value we previously allocated, so remove it
  // from the next state.
  State = State->remove<AllocatedData>(ArgSM);

  // Check if the proper deallocator is used.
  unsigned int PDeallocIdx = FunctionsToTrack[AS->AllocatorIdx].DeallocatorIdx;
  if (PDeallocIdx != idx || (FunctionsToTrack[idx].Kind == ErrorAPI)) {
    const AllocationPair AP = std::make_pair(ArgSM, AS);
    generateDeallocatorMismatchReport(AP, ArgExpr, C);
    return;
  }

  // If the return status is undefined or is error, report a bad call to free.
  if (!definitelyDidnotReturnError(AS->Region, State, C.getSValBuilder())) {
    ExplodedNode *N = C.generateNode(State);
    if (!N)
      return;
    initBugType();
    BugReport *Report = new BugReport(*BT,
        "Call to free data when error was returned during allocation.", N);
    Report->addVisitor(new SecKeychainBugVisitor(ArgSM));
    Report->addRange(ArgExpr->getSourceRange());
    C.EmitReport(Report);
    return;
  }

  C.addTransition(State);
}

void MacOSKeychainAPIChecker::checkPostStmt(const CallExpr *CE,
                                            CheckerContext &C) const {
  const ProgramState *State = C.getState();
  const Expr *Callee = CE->getCallee();
  SVal L = State->getSVal(Callee);

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
  // If the argument entered as an enclosing function parameter, skip it to
  // avoid false positives.
  if (isEnclosingFunctionParam(ArgExpr))
    return;

  if (SymbolRef V = getAsPointeeSymbol(ArgExpr, C)) {
    // If the argument points to something that's not a symbolic region, it
    // can be:
    //  - unknown (cannot reason about it)
    //  - undefined (already reported by other checker)
    //  - constant (null - should not be tracked,
    //              other constant will generate a compiler warning)
    //  - goto (should be reported by other checker)

    // The call return value symbol should stay alive for as long as the
    // allocated value symbol, since our diagnostics depend on the value
    // returned by the call. Ex: Data should only be freed if noErr was
    // returned during allocation.)
    SymbolRef RetStatusSymbol = State->getSVal(CE).getAsSymbol();
    C.getSymbolManager().addSymbolDependency(V, RetStatusSymbol);

    // Track the allocated value in the checker state.
    State = State->set<AllocatedData>(V, AllocationState(ArgExpr, idx,
                                                         RetStatusSymbol));
    assert(State);
    C.addTransition(State);
  }
}

void MacOSKeychainAPIChecker::checkPreStmt(const ReturnStmt *S,
                                           CheckerContext &C) const {
  const Expr *retExpr = S->getRetValue();
  if (!retExpr)
    return;

  // Check  if the value is escaping through the return.
  const ProgramState *state = C.getState();
  const MemRegion *V = state->getSVal(retExpr).getAsRegion();
  if (!V)
    return;
  state = state->remove<AllocatedData>(getSymbolForRegion(C, V));

  // Proceed from the new state.
  C.addTransition(state);
}

BugReport *MacOSKeychainAPIChecker::
  generateAllocatedDataNotReleasedReport(const AllocationPair &AP,
                                         ExplodedNode *N) const {
  const ADFunctionInfo &FI = FunctionsToTrack[AP.second->AllocatorIdx];
  initBugType();
  llvm::SmallString<70> sbuf;
  llvm::raw_svector_ostream os(sbuf);

  os << "Allocated data is not released: missing a call to '"
      << FunctionsToTrack[FI.DeallocatorIdx].Name << "'.";
  BugReport *Report = new BugReport(*BT, os.str(), N);
  Report->addVisitor(new SecKeychainBugVisitor(AP.first));
  Report->addRange(SourceRange());
  return Report;
}

void MacOSKeychainAPIChecker::checkDeadSymbols(SymbolReaper &SR,
                                               CheckerContext &C) const {
  const ProgramState *State = C.getState();
  AllocatedSetTy ASet = State->get<AllocatedData>();
  if (ASet.isEmpty())
    return;

  bool Changed = false;
  AllocationPairVec Errors;
  for (AllocatedSetTy::iterator I = ASet.begin(), E = ASet.end(); I != E; ++I) {
    if (SR.isLive(I->first))
      continue;

    Changed = true;
    State = State->remove<AllocatedData>(I->first);
    // If the allocated symbol is null or if the allocation call might have
    // returned an error, do not report.
    if (State->getSymVal(I->first) ||
        definitelyReturnedError(I->second.Region, State, C.getSValBuilder()))
      continue;
    Errors.push_back(std::make_pair(I->first, &I->second));
  }
  if (!Changed)
    return;

  // Generate the new, cleaned up state.
  ExplodedNode *N = C.generateNode(State);
  if (!N)
    return;

  // Generate the error reports.
  for (AllocationPairVec::iterator I = Errors.begin(), E = Errors.end();
                                                       I != E; ++I) {
    C.EmitReport(generateAllocatedDataNotReleasedReport(*I, N));
  }
}

// TODO: Remove this after we ensure that checkDeadSymbols are always called.
void MacOSKeychainAPIChecker::checkEndPath(EndOfFunctionNodeBuilder &B,
                                           ExprEngine &Eng) const {
  const ProgramState *state = B.getState();
  AllocatedSetTy AS = state->get<AllocatedData>();
  if (AS.isEmpty())
    return;

  // Anything which has been allocated but not freed (nor escaped) will be
  // found here, so report it.
  bool Changed = false;
  AllocationPairVec Errors;
  for (AllocatedSetTy::iterator I = AS.begin(), E = AS.end(); I != E; ++I ) {
    Changed = true;
    state = state->remove<AllocatedData>(I->first);
    // If the allocated symbol is null or if error code was returned at
    // allocation, do not report.
    if (state->getSymVal(I.getKey()) ||
        definitelyReturnedError(I->second.Region, state,
                                Eng.getSValBuilder())) {
      continue;
    }
    Errors.push_back(std::make_pair(I->first, &I->second));
  }

  // If no change, do not generate a new state.
  if (!Changed)
    return;

  ExplodedNode *N = B.generateNode(state);
  if (!N)
    return;

  // Generate the error reports.
  for (AllocationPairVec::iterator I = Errors.begin(), E = Errors.end();
                                                       I != E; ++I) {
    Eng.getBugReporter().EmitReport(
      generateAllocatedDataNotReleasedReport(*I, N));
  }
}


PathDiagnosticPiece *MacOSKeychainAPIChecker::SecKeychainBugVisitor::VisitNode(
                                                      const ExplodedNode *N,
                                                      const ExplodedNode *PrevN,
                                                      BugReporterContext &BRC,
                                                      BugReport &BR) {
  const AllocationState *AS = N->getState()->get<AllocatedData>(Sym);
  if (!AS)
    return 0;
  const AllocationState *ASPrev = PrevN->getState()->get<AllocatedData>(Sym);
  if (ASPrev)
    return 0;

  // (!ASPrev && AS) ~ We started tracking symbol in node N, it must be the
  // allocation site.
  const CallExpr *CE = cast<CallExpr>(cast<StmtPoint>(N->getLocation())
                                                            .getStmt());
  const FunctionDecl *funDecl = CE->getDirectCallee();
  assert(funDecl && "We do not support indirect function calls as of now.");
  StringRef funName = funDecl->getName();

  // Get the expression of the corresponding argument.
  unsigned Idx = getTrackedFunctionIndex(funName, true);
  assert(Idx != InvalidIdx && "This should be a call to an allocator.");
  const Expr *ArgExpr = CE->getArg(FunctionsToTrack[Idx].Param);
  PathDiagnosticLocation Pos(ArgExpr, BRC.getSourceManager());
  return new PathDiagnosticEventPiece(Pos, "Data is allocated here.");
}

void ento::registerMacOSKeychainAPIChecker(CheckerManager &mgr) {
  mgr.registerChecker<MacOSKeychainAPIChecker>();
}
