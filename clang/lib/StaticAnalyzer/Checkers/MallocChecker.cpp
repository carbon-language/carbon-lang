//=== MallocChecker.cpp - A malloc/free checker -------------------*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines malloc/free checker, which checks for potential memory
// leaks, double free, and use-after-free problems.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "InterCheckerAPI.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ObjCMessage.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SymbolManager.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/STLExtras.h"
#include <climits>

using namespace clang;
using namespace ento;

namespace {

class RefState {
  enum Kind { AllocateUnchecked, AllocateFailed, Released, Escaped,
              Relinquished } K;
  const Stmt *S;

public:
  RefState(Kind k, const Stmt *s) : K(k), S(s) {}

  bool isAllocated() const { return K == AllocateUnchecked; }
  bool isReleased() const { return K == Released; }

  const Stmt *getStmt() const { return S; }

  bool operator==(const RefState &X) const {
    return K == X.K && S == X.S;
  }

  static RefState getAllocateUnchecked(const Stmt *s) { 
    return RefState(AllocateUnchecked, s); 
  }
  static RefState getAllocateFailed() {
    return RefState(AllocateFailed, 0);
  }
  static RefState getReleased(const Stmt *s) { return RefState(Released, s); }
  static RefState getEscaped(const Stmt *s) { return RefState(Escaped, s); }
  static RefState getRelinquished(const Stmt *s) {
    return RefState(Relinquished, s);
  }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddInteger(K);
    ID.AddPointer(S);
  }
};

struct ReallocPair {
  SymbolRef ReallocatedSym;
  bool IsFreeOnFailure;
  ReallocPair(SymbolRef S, bool F) : ReallocatedSym(S), IsFreeOnFailure(F) {}
  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddInteger(IsFreeOnFailure);
    ID.AddPointer(ReallocatedSym);
  }
  bool operator==(const ReallocPair &X) const {
    return ReallocatedSym == X.ReallocatedSym &&
           IsFreeOnFailure == X.IsFreeOnFailure;
  }
};

typedef std::pair<const Stmt*, const MemRegion*> LeakInfo;

class MallocChecker : public Checker<check::DeadSymbols,
                                     check::EndPath,
                                     check::PreStmt<ReturnStmt>,
                                     check::PreStmt<CallExpr>,
                                     check::PostStmt<CallExpr>,
                                     check::PostStmt<BlockExpr>,
                                     check::Location,
                                     check::Bind,
                                     eval::Assume,
                                     check::RegionChanges>
{
  mutable OwningPtr<BugType> BT_DoubleFree;
  mutable OwningPtr<BugType> BT_Leak;
  mutable OwningPtr<BugType> BT_UseFree;
  mutable OwningPtr<BugType> BT_BadFree;
  mutable IdentifierInfo *II_malloc, *II_free, *II_realloc, *II_calloc,
                         *II_valloc, *II_reallocf, *II_strndup, *II_strdup;

public:
  MallocChecker() : II_malloc(0), II_free(0), II_realloc(0), II_calloc(0),
                    II_valloc(0), II_reallocf(0), II_strndup(0), II_strdup(0) {}

  /// In pessimistic mode, the checker assumes that it does not know which
  /// functions might free the memory.
  struct ChecksFilter {
    DefaultBool CMallocPessimistic;
    DefaultBool CMallocOptimistic;
  };

  ChecksFilter Filter;

  void checkPreStmt(const CallExpr *S, CheckerContext &C) const;
  void checkPostStmt(const CallExpr *CE, CheckerContext &C) const;
  void checkPostStmt(const BlockExpr *BE, CheckerContext &C) const;
  void checkDeadSymbols(SymbolReaper &SymReaper, CheckerContext &C) const;
  void checkEndPath(CheckerContext &C) const;
  void checkPreStmt(const ReturnStmt *S, CheckerContext &C) const;
  ProgramStateRef evalAssume(ProgramStateRef state, SVal Cond,
                            bool Assumption) const;
  void checkLocation(SVal l, bool isLoad, const Stmt *S,
                     CheckerContext &C) const;
  void checkBind(SVal location, SVal val, const Stmt*S,
                 CheckerContext &C) const;
  ProgramStateRef
  checkRegionChanges(ProgramStateRef state,
                     const StoreManager::InvalidatedSymbols *invalidated,
                     ArrayRef<const MemRegion *> ExplicitRegions,
                     ArrayRef<const MemRegion *> Regions,
                     const CallOrObjCMessage *Call) const;
  bool wantsRegionChangeUpdate(ProgramStateRef state) const {
    return true;
  }

private:
  void initIdentifierInfo(ASTContext &C) const;

  /// Check if this is one of the functions which can allocate/reallocate memory 
  /// pointed to by one of its arguments.
  bool isMemFunction(const FunctionDecl *FD, ASTContext &C) const;

  static ProgramStateRef MallocMemReturnsAttr(CheckerContext &C,
                                              const CallExpr *CE,
                                              const OwnershipAttr* Att);
  static ProgramStateRef MallocMemAux(CheckerContext &C, const CallExpr *CE,
                                     const Expr *SizeEx, SVal Init,
                                     ProgramStateRef state) {
    return MallocMemAux(C, CE,
                        state->getSVal(SizeEx, C.getLocationContext()),
                        Init, state);
  }

  static ProgramStateRef MallocMemAux(CheckerContext &C, const CallExpr *CE,
                                     SVal SizeEx, SVal Init,
                                     ProgramStateRef state);

  /// Update the RefState to reflect the new memory allocation.
  static ProgramStateRef MallocUpdateRefState(CheckerContext &C,
                                              const CallExpr *CE,
                                              ProgramStateRef state);

  ProgramStateRef FreeMemAttr(CheckerContext &C, const CallExpr *CE,
                              const OwnershipAttr* Att) const;
  ProgramStateRef FreeMemAux(CheckerContext &C, const CallExpr *CE,
                                 ProgramStateRef state, unsigned Num,
                                 bool Hold) const;

  ProgramStateRef ReallocMem(CheckerContext &C, const CallExpr *CE,
                             bool FreesMemOnFailure) const;
  static ProgramStateRef CallocMem(CheckerContext &C, const CallExpr *CE);
  
  bool checkEscape(SymbolRef Sym, const Stmt *S, CheckerContext &C) const;
  bool checkUseAfterFree(SymbolRef Sym, CheckerContext &C,
                         const Stmt *S = 0) const;

  /// Check if the function is not known to us. So, for example, we could
  /// conservatively assume it can free/reallocate it's pointer arguments.
  bool doesNotFreeMemory(const CallOrObjCMessage *Call,
                         ProgramStateRef State) const;

  static bool SummarizeValue(raw_ostream &os, SVal V);
  static bool SummarizeRegion(raw_ostream &os, const MemRegion *MR);
  void ReportBadFree(CheckerContext &C, SVal ArgVal, SourceRange range) const;

  /// Find the location of the allocation for Sym on the path leading to the
  /// exploded node N.
  LeakInfo getAllocationSite(const ExplodedNode *N, SymbolRef Sym,
                             CheckerContext &C) const;

  void reportLeak(SymbolRef Sym, ExplodedNode *N, CheckerContext &C) const;

  /// The bug visitor which allows us to print extra diagnostics along the
  /// BugReport path. For example, showing the allocation site of the leaked
  /// region.
  class MallocBugVisitor : public BugReporterVisitorImpl<MallocBugVisitor> {
  protected:
    enum NotificationMode {
      Normal,
      ReallocationFailed
    };

    // The allocated region symbol tracked by the main analysis.
    SymbolRef Sym;

     // The mode we are in, i.e. what kind of diagnostics will be emitted.
     NotificationMode Mode;

     // A symbol from when the primary region should have been reallocated.
     SymbolRef FailedReallocSymbol;

   public:
     MallocBugVisitor(SymbolRef S)
       : Sym(S), Mode(Normal), FailedReallocSymbol(0) {}

    virtual ~MallocBugVisitor() {}

    void Profile(llvm::FoldingSetNodeID &ID) const {
      static int X = 0;
      ID.AddPointer(&X);
      ID.AddPointer(Sym);
    }

    inline bool isAllocated(const RefState *S, const RefState *SPrev,
                            const Stmt *Stmt) {
      // Did not track -> allocated. Other state (released) -> allocated.
      return (Stmt && isa<CallExpr>(Stmt) &&
              (S && S->isAllocated()) && (!SPrev || !SPrev->isAllocated()));
    }

    inline bool isReleased(const RefState *S, const RefState *SPrev,
                           const Stmt *Stmt) {
      // Did not track -> released. Other state (allocated) -> released.
      return (Stmt && isa<CallExpr>(Stmt) &&
              (S && S->isReleased()) && (!SPrev || !SPrev->isReleased()));
    }

    inline bool isReallocFailedCheck(const RefState *S, const RefState *SPrev,
                                     const Stmt *Stmt) {
      // If the expression is not a call, and the state change is
      // released -> allocated, it must be the realloc return value
      // check. If we have to handle more cases here, it might be cleaner just
      // to track this extra bit in the state itself.
      return ((!Stmt || !isa<CallExpr>(Stmt)) &&
              (S && S->isAllocated()) && (SPrev && !SPrev->isAllocated()));
    }

    PathDiagnosticPiece *VisitNode(const ExplodedNode *N,
                                   const ExplodedNode *PrevN,
                                   BugReporterContext &BRC,
                                   BugReport &BR);
  private:
    class StackHintGeneratorForReallocationFailed
        : public StackHintGeneratorForSymbol {
    public:
      StackHintGeneratorForReallocationFailed(SymbolRef S, StringRef M)
        : StackHintGeneratorForSymbol(S, M) {}

      virtual std::string getMessageForArg(const Expr *ArgE, unsigned ArgIndex) {
        SmallString<200> buf;
        llvm::raw_svector_ostream os(buf);

        os << "Reallocation of ";
        // Printed parameters start at 1, not 0.
        printOrdinal(++ArgIndex, os);
        os << " parameter failed";

        return os.str();
      }

      virtual std::string getMessageForReturn(const CallExpr *CallExpr) {
        return "Reallocation of returned value failed";
      }
    };
  };
};
} // end anonymous namespace

typedef llvm::ImmutableMap<SymbolRef, RefState> RegionStateTy;
typedef llvm::ImmutableMap<SymbolRef, ReallocPair > ReallocMap;
class RegionState {};
class ReallocPairs {};
namespace clang {
namespace ento {
  template <>
  struct ProgramStateTrait<RegionState> 
    : public ProgramStatePartialTrait<RegionStateTy> {
    static void *GDMIndex() { static int x; return &x; }
  };

  template <>
  struct ProgramStateTrait<ReallocPairs>
    : public ProgramStatePartialTrait<ReallocMap> {
    static void *GDMIndex() { static int x; return &x; }
  };
}
}

namespace {
class StopTrackingCallback : public SymbolVisitor {
  ProgramStateRef state;
public:
  StopTrackingCallback(ProgramStateRef st) : state(st) {}
  ProgramStateRef getState() const { return state; }

  bool VisitSymbol(SymbolRef sym) {
    state = state->remove<RegionState>(sym);
    return true;
  }
};
} // end anonymous namespace

void MallocChecker::initIdentifierInfo(ASTContext &Ctx) const {
  if (!II_malloc)
    II_malloc = &Ctx.Idents.get("malloc");
  if (!II_free)
    II_free = &Ctx.Idents.get("free");
  if (!II_realloc)
    II_realloc = &Ctx.Idents.get("realloc");
  if (!II_reallocf)
    II_reallocf = &Ctx.Idents.get("reallocf");
  if (!II_calloc)
    II_calloc = &Ctx.Idents.get("calloc");
  if (!II_valloc)
    II_valloc = &Ctx.Idents.get("valloc");
  if (!II_strdup)
    II_strdup = &Ctx.Idents.get("strdup");
  if (!II_strndup)
    II_strndup = &Ctx.Idents.get("strndup");
}

bool MallocChecker::isMemFunction(const FunctionDecl *FD, ASTContext &C) const {
  if (!FD)
    return false;
  IdentifierInfo *FunI = FD->getIdentifier();
  if (!FunI)
    return false;

  initIdentifierInfo(C);

  if (FunI == II_malloc || FunI == II_free || FunI == II_realloc ||
      FunI == II_reallocf || FunI == II_calloc || FunI == II_valloc ||
      FunI == II_strdup || FunI == II_strndup)
    return true;

  if (Filter.CMallocOptimistic && FD->hasAttrs() &&
      FD->specific_attr_begin<OwnershipAttr>() !=
          FD->specific_attr_end<OwnershipAttr>())
    return true;


  return false;
}

void MallocChecker::checkPostStmt(const CallExpr *CE, CheckerContext &C) const {
  const FunctionDecl *FD = C.getCalleeDecl(CE);
  if (!FD)
    return;

  initIdentifierInfo(C.getASTContext());
  IdentifierInfo *FunI = FD->getIdentifier();
  if (!FunI)
    return;

  ProgramStateRef State = C.getState();
  if (FunI == II_malloc || FunI == II_valloc) {
    State = MallocMemAux(C, CE, CE->getArg(0), UndefinedVal(), State);
  } else if (FunI == II_realloc) {
    State = ReallocMem(C, CE, false);
  } else if (FunI == II_reallocf) {
    State = ReallocMem(C, CE, true);
  } else if (FunI == II_calloc) {
    State = CallocMem(C, CE);
  } else if (FunI == II_free) {
    State = FreeMemAux(C, CE, C.getState(), 0, false);
  } else if (FunI == II_strdup) {
    State = MallocUpdateRefState(C, CE, State);
  } else if (FunI == II_strndup) {
    State = MallocUpdateRefState(C, CE, State);
  } else if (Filter.CMallocOptimistic) {
    // Check all the attributes, if there are any.
    // There can be multiple of these attributes.
    if (FD->hasAttrs())
      for (specific_attr_iterator<OwnershipAttr>
          i = FD->specific_attr_begin<OwnershipAttr>(),
          e = FD->specific_attr_end<OwnershipAttr>();
          i != e; ++i) {
        switch ((*i)->getOwnKind()) {
        case OwnershipAttr::Returns:
          State = MallocMemReturnsAttr(C, CE, *i);
          break;
        case OwnershipAttr::Takes:
        case OwnershipAttr::Holds:
          State = FreeMemAttr(C, CE, *i);
          break;
        }
      }
  }
  C.addTransition(State);
}

ProgramStateRef MallocChecker::MallocMemReturnsAttr(CheckerContext &C,
                                                    const CallExpr *CE,
                                                    const OwnershipAttr* Att) {
  if (Att->getModule() != "malloc")
    return 0;

  OwnershipAttr::args_iterator I = Att->args_begin(), E = Att->args_end();
  if (I != E) {
    return MallocMemAux(C, CE, CE->getArg(*I), UndefinedVal(), C.getState());
  }
  return MallocMemAux(C, CE, UnknownVal(), UndefinedVal(), C.getState());
}

ProgramStateRef MallocChecker::MallocMemAux(CheckerContext &C,
                                           const CallExpr *CE,
                                           SVal Size, SVal Init,
                                           ProgramStateRef state) {
  // Get the return value.
  SVal retVal = state->getSVal(CE, C.getLocationContext());

  // We expect the malloc functions to return a pointer.
  if (!isa<Loc>(retVal))
    return 0;

  // Fill the region with the initialization value.
  state = state->bindDefault(retVal, Init);

  // Set the region's extent equal to the Size parameter.
  const SymbolicRegion *R =
      dyn_cast_or_null<SymbolicRegion>(retVal.getAsRegion());
  if (!R)
    return 0;
  if (isa<DefinedOrUnknownSVal>(Size)) {
    SValBuilder &svalBuilder = C.getSValBuilder();
    DefinedOrUnknownSVal Extent = R->getExtent(svalBuilder);
    DefinedOrUnknownSVal DefinedSize = cast<DefinedOrUnknownSVal>(Size);
    DefinedOrUnknownSVal extentMatchesSize =
        svalBuilder.evalEQ(state, Extent, DefinedSize);

    state = state->assume(extentMatchesSize, true);
    assert(state);
  }
  
  return MallocUpdateRefState(C, CE, state);
}

ProgramStateRef MallocChecker::MallocUpdateRefState(CheckerContext &C,
                                                    const CallExpr *CE,
                                                    ProgramStateRef state) {
  // Get the return value.
  SVal retVal = state->getSVal(CE, C.getLocationContext());

  // We expect the malloc functions to return a pointer.
  if (!isa<Loc>(retVal))
    return 0;

  SymbolRef Sym = retVal.getAsLocSymbol();
  assert(Sym);

  // Set the symbol's state to Allocated.
  return state->set<RegionState>(Sym, RefState::getAllocateUnchecked(CE));

}

ProgramStateRef MallocChecker::FreeMemAttr(CheckerContext &C,
                                           const CallExpr *CE,
                                           const OwnershipAttr* Att) const {
  if (Att->getModule() != "malloc")
    return 0;

  ProgramStateRef State = C.getState();

  for (OwnershipAttr::args_iterator I = Att->args_begin(), E = Att->args_end();
       I != E; ++I) {
    ProgramStateRef StateI = FreeMemAux(C, CE, State, *I,
                               Att->getOwnKind() == OwnershipAttr::Holds);
    if (StateI)
      State = StateI;
  }
  return State;
}

ProgramStateRef MallocChecker::FreeMemAux(CheckerContext &C,
                                          const CallExpr *CE,
                                          ProgramStateRef state,
                                          unsigned Num,
                                          bool Hold) const {
  const Expr *ArgExpr = CE->getArg(Num);
  SVal ArgVal = state->getSVal(ArgExpr, C.getLocationContext());
  if (!isa<DefinedOrUnknownSVal>(ArgVal))
    return 0;
  DefinedOrUnknownSVal location = cast<DefinedOrUnknownSVal>(ArgVal);

  // Check for null dereferences.
  if (!isa<Loc>(location))
    return 0;

  // The explicit NULL case, no operation is performed.
  ProgramStateRef notNullState, nullState;
  llvm::tie(notNullState, nullState) = state->assume(location);
  if (nullState && !notNullState)
    return 0;

  // Unknown values could easily be okay
  // Undefined values are handled elsewhere
  if (ArgVal.isUnknownOrUndef())
    return 0;

  const MemRegion *R = ArgVal.getAsRegion();
  
  // Nonlocs can't be freed, of course.
  // Non-region locations (labels and fixed addresses) also shouldn't be freed.
  if (!R) {
    ReportBadFree(C, ArgVal, ArgExpr->getSourceRange());
    return 0;
  }
  
  R = R->StripCasts();
  
  // Blocks might show up as heap data, but should not be free()d
  if (isa<BlockDataRegion>(R)) {
    ReportBadFree(C, ArgVal, ArgExpr->getSourceRange());
    return 0;
  }
  
  const MemSpaceRegion *MS = R->getMemorySpace();
  
  // Parameters, locals, statics, and globals shouldn't be freed.
  if (!(isa<UnknownSpaceRegion>(MS) || isa<HeapSpaceRegion>(MS))) {
    // FIXME: at the time this code was written, malloc() regions were
    // represented by conjured symbols, which are all in UnknownSpaceRegion.
    // This means that there isn't actually anything from HeapSpaceRegion
    // that should be freed, even though we allow it here.
    // Of course, free() can work on memory allocated outside the current
    // function, so UnknownSpaceRegion is always a possibility.
    // False negatives are better than false positives.
    
    ReportBadFree(C, ArgVal, ArgExpr->getSourceRange());
    return 0;
  }
  
  const SymbolicRegion *SR = dyn_cast<SymbolicRegion>(R);
  // Various cases could lead to non-symbol values here.
  // For now, ignore them.
  if (!SR)
    return 0;

  SymbolRef Sym = SR->getSymbol();
  const RefState *RS = state->get<RegionState>(Sym);

  // If the symbol has not been tracked, return. This is possible when free() is
  // called on a pointer that does not get its pointee directly from malloc(). 
  // Full support of this requires inter-procedural analysis.
  if (!RS)
    return 0;

  // Check double free.
  if (RS->isReleased()) {
    if (ExplodedNode *N = C.generateSink()) {
      if (!BT_DoubleFree)
        BT_DoubleFree.reset(
          new BugType("Double free", "Memory Error"));
      BugReport *R = new BugReport(*BT_DoubleFree, 
                        "Attempt to free released memory", N);
      R->addRange(ArgExpr->getSourceRange());
      R->markInteresting(Sym);
      R->addVisitor(new MallocBugVisitor(Sym));
      C.EmitReport(R);
    }
    return 0;
  }

  // Normal free.
  if (Hold)
    return state->set<RegionState>(Sym, RefState::getRelinquished(CE));
  return state->set<RegionState>(Sym, RefState::getReleased(CE));
}

bool MallocChecker::SummarizeValue(raw_ostream &os, SVal V) {
  if (nonloc::ConcreteInt *IntVal = dyn_cast<nonloc::ConcreteInt>(&V))
    os << "an integer (" << IntVal->getValue() << ")";
  else if (loc::ConcreteInt *ConstAddr = dyn_cast<loc::ConcreteInt>(&V))
    os << "a constant address (" << ConstAddr->getValue() << ")";
  else if (loc::GotoLabel *Label = dyn_cast<loc::GotoLabel>(&V))
    os << "the address of the label '" << Label->getLabel()->getName() << "'";
  else
    return false;
  
  return true;
}

bool MallocChecker::SummarizeRegion(raw_ostream &os,
                                    const MemRegion *MR) {
  switch (MR->getKind()) {
  case MemRegion::FunctionTextRegionKind: {
    const FunctionDecl *FD = cast<FunctionTextRegion>(MR)->getDecl();
    if (FD)
      os << "the address of the function '" << *FD << '\'';
    else
      os << "the address of a function";
    return true;
  }
  case MemRegion::BlockTextRegionKind:
    os << "block text";
    return true;
  case MemRegion::BlockDataRegionKind:
    // FIXME: where the block came from?
    os << "a block";
    return true;
  default: {
    const MemSpaceRegion *MS = MR->getMemorySpace();
    
    if (isa<StackLocalsSpaceRegion>(MS)) {
      const VarRegion *VR = dyn_cast<VarRegion>(MR);
      const VarDecl *VD;
      if (VR)
        VD = VR->getDecl();
      else
        VD = NULL;
      
      if (VD)
        os << "the address of the local variable '" << VD->getName() << "'";
      else
        os << "the address of a local stack variable";
      return true;
    }

    if (isa<StackArgumentsSpaceRegion>(MS)) {
      const VarRegion *VR = dyn_cast<VarRegion>(MR);
      const VarDecl *VD;
      if (VR)
        VD = VR->getDecl();
      else
        VD = NULL;
      
      if (VD)
        os << "the address of the parameter '" << VD->getName() << "'";
      else
        os << "the address of a parameter";
      return true;
    }

    if (isa<GlobalsSpaceRegion>(MS)) {
      const VarRegion *VR = dyn_cast<VarRegion>(MR);
      const VarDecl *VD;
      if (VR)
        VD = VR->getDecl();
      else
        VD = NULL;
      
      if (VD) {
        if (VD->isStaticLocal())
          os << "the address of the static variable '" << VD->getName() << "'";
        else
          os << "the address of the global variable '" << VD->getName() << "'";
      } else
        os << "the address of a global variable";
      return true;
    }

    return false;
  }
  }
}

void MallocChecker::ReportBadFree(CheckerContext &C, SVal ArgVal,
                                  SourceRange range) const {
  if (ExplodedNode *N = C.generateSink()) {
    if (!BT_BadFree)
      BT_BadFree.reset(new BugType("Bad free", "Memory Error"));
    
    SmallString<100> buf;
    llvm::raw_svector_ostream os(buf);
    
    const MemRegion *MR = ArgVal.getAsRegion();
    if (MR) {
      while (const ElementRegion *ER = dyn_cast<ElementRegion>(MR))
        MR = ER->getSuperRegion();
      
      // Special case for alloca()
      if (isa<AllocaRegion>(MR))
        os << "Argument to free() was allocated by alloca(), not malloc()";
      else {
        os << "Argument to free() is ";
        if (SummarizeRegion(os, MR))
          os << ", which is not memory allocated by malloc()";
        else
          os << "not memory allocated by malloc()";
      }
    } else {
      os << "Argument to free() is ";
      if (SummarizeValue(os, ArgVal))
        os << ", which is not memory allocated by malloc()";
      else
        os << "not memory allocated by malloc()";
    }
    
    BugReport *R = new BugReport(*BT_BadFree, os.str(), N);
    R->markInteresting(MR);
    R->addRange(range);
    C.EmitReport(R);
  }
}

ProgramStateRef MallocChecker::ReallocMem(CheckerContext &C,
                                          const CallExpr *CE,
                                          bool FreesOnFail) const {
  ProgramStateRef state = C.getState();
  const Expr *arg0Expr = CE->getArg(0);
  const LocationContext *LCtx = C.getLocationContext();
  SVal Arg0Val = state->getSVal(arg0Expr, LCtx);
  if (!isa<DefinedOrUnknownSVal>(Arg0Val))
    return 0;
  DefinedOrUnknownSVal arg0Val = cast<DefinedOrUnknownSVal>(Arg0Val);

  SValBuilder &svalBuilder = C.getSValBuilder();

  DefinedOrUnknownSVal PtrEQ =
    svalBuilder.evalEQ(state, arg0Val, svalBuilder.makeNull());

  // Get the size argument. If there is no size arg then give up.
  const Expr *Arg1 = CE->getArg(1);
  if (!Arg1)
    return 0;

  // Get the value of the size argument.
  SVal Arg1ValG = state->getSVal(Arg1, LCtx);
  if (!isa<DefinedOrUnknownSVal>(Arg1ValG))
    return 0;
  DefinedOrUnknownSVal Arg1Val = cast<DefinedOrUnknownSVal>(Arg1ValG);

  // Compare the size argument to 0.
  DefinedOrUnknownSVal SizeZero =
    svalBuilder.evalEQ(state, Arg1Val,
                       svalBuilder.makeIntValWithPtrWidth(0, false));

  ProgramStateRef StatePtrIsNull, StatePtrNotNull;
  llvm::tie(StatePtrIsNull, StatePtrNotNull) = state->assume(PtrEQ);
  ProgramStateRef StateSizeIsZero, StateSizeNotZero;
  llvm::tie(StateSizeIsZero, StateSizeNotZero) = state->assume(SizeZero);
  // We only assume exceptional states if they are definitely true; if the
  // state is under-constrained, assume regular realloc behavior.
  bool PrtIsNull = StatePtrIsNull && !StatePtrNotNull;
  bool SizeIsZero = StateSizeIsZero && !StateSizeNotZero;

  // If the ptr is NULL and the size is not 0, the call is equivalent to 
  // malloc(size).
  if ( PrtIsNull && !SizeIsZero) {
    ProgramStateRef stateMalloc = MallocMemAux(C, CE, CE->getArg(1),
                                               UndefinedVal(), StatePtrIsNull);
    return stateMalloc;
  }

  if (PrtIsNull && SizeIsZero)
    return 0;

  // Get the from and to pointer symbols as in toPtr = realloc(fromPtr, size).
  assert(!PrtIsNull);
  SymbolRef FromPtr = arg0Val.getAsSymbol();
  SVal RetVal = state->getSVal(CE, LCtx);
  SymbolRef ToPtr = RetVal.getAsSymbol();
  if (!FromPtr || !ToPtr)
    return 0;

  // If the size is 0, free the memory.
  if (SizeIsZero)
    if (ProgramStateRef stateFree = FreeMemAux(C, CE, StateSizeIsZero,0,false)){
      // The semantics of the return value are:
      // If size was equal to 0, either NULL or a pointer suitable to be passed
      // to free() is returned.
      stateFree = stateFree->set<ReallocPairs>(ToPtr,
                                            ReallocPair(FromPtr, FreesOnFail));
      C.getSymbolManager().addSymbolDependency(ToPtr, FromPtr);
      return stateFree;
    }

  // Default behavior.
  if (ProgramStateRef stateFree = FreeMemAux(C, CE, state, 0, false)) {
    // FIXME: We should copy the content of the original buffer.
    ProgramStateRef stateRealloc = MallocMemAux(C, CE, CE->getArg(1),
                                                UnknownVal(), stateFree);
    if (!stateRealloc)
      return 0;
    stateRealloc = stateRealloc->set<ReallocPairs>(ToPtr,
                                            ReallocPair(FromPtr, FreesOnFail));
    C.getSymbolManager().addSymbolDependency(ToPtr, FromPtr);
    return stateRealloc;
  }
  return 0;
}

ProgramStateRef MallocChecker::CallocMem(CheckerContext &C, const CallExpr *CE){
  ProgramStateRef state = C.getState();
  SValBuilder &svalBuilder = C.getSValBuilder();
  const LocationContext *LCtx = C.getLocationContext();
  SVal count = state->getSVal(CE->getArg(0), LCtx);
  SVal elementSize = state->getSVal(CE->getArg(1), LCtx);
  SVal TotalSize = svalBuilder.evalBinOp(state, BO_Mul, count, elementSize,
                                        svalBuilder.getContext().getSizeType());  
  SVal zeroVal = svalBuilder.makeZeroVal(svalBuilder.getContext().CharTy);

  return MallocMemAux(C, CE, TotalSize, zeroVal, state);
}

LeakInfo
MallocChecker::getAllocationSite(const ExplodedNode *N, SymbolRef Sym,
                                 CheckerContext &C) const {
  const LocationContext *LeakContext = N->getLocationContext();
  // Walk the ExplodedGraph backwards and find the first node that referred to
  // the tracked symbol.
  const ExplodedNode *AllocNode = N;
  const MemRegion *ReferenceRegion = 0;

  while (N) {
    ProgramStateRef State = N->getState();
    if (!State->get<RegionState>(Sym))
      break;

    // Find the most recent expression bound to the symbol in the current
    // context.
    if (!ReferenceRegion) {
      if (const MemRegion *MR = C.getLocationRegionIfPostStore(N)) {
        SVal Val = State->getSVal(MR);
        if (Val.getAsLocSymbol() == Sym)
          ReferenceRegion = MR;
      }
    }

    // Allocation node, is the last node in the current context in which the
    // symbol was tracked.
    if (N->getLocationContext() == LeakContext)
      AllocNode = N;
    N = N->pred_empty() ? NULL : *(N->pred_begin());
  }

  ProgramPoint P = AllocNode->getLocation();
  const Stmt *AllocationStmt = 0;
  if (isa<StmtPoint>(P))
    AllocationStmt = cast<StmtPoint>(P).getStmt();

  return LeakInfo(AllocationStmt, ReferenceRegion);
}

void MallocChecker::reportLeak(SymbolRef Sym, ExplodedNode *N,
                               CheckerContext &C) const {
  assert(N);
  if (!BT_Leak) {
    BT_Leak.reset(new BugType("Memory leak", "Memory Error"));
    // Leaks should not be reported if they are post-dominated by a sink:
    // (1) Sinks are higher importance bugs.
    // (2) NoReturnFunctionChecker uses sink nodes to represent paths ending
    //     with __noreturn functions such as assert() or exit(). We choose not
    //     to report leaks on such paths.
    BT_Leak->setSuppressOnSink(true);
  }

  // Most bug reports are cached at the location where they occurred.
  // With leaks, we want to unique them by the location where they were
  // allocated, and only report a single path.
  PathDiagnosticLocation LocUsedForUniqueing;
  const Stmt *AllocStmt = 0;
  const MemRegion *Region = 0;
  llvm::tie(AllocStmt, Region) = getAllocationSite(N, Sym, C);
  if (AllocStmt)
    LocUsedForUniqueing = PathDiagnosticLocation::createBegin(AllocStmt,
                            C.getSourceManager(), N->getLocationContext());

  SmallString<200> buf;
  llvm::raw_svector_ostream os(buf);
  os << "Memory is never released; potential leak";
  if (Region) {
    os << " of memory pointed to by '";
    Region->dumpPretty(os);
    os <<'\'';
  }

  BugReport *R = new BugReport(*BT_Leak, os.str(), N, LocUsedForUniqueing);
  R->markInteresting(Sym);
  R->addVisitor(new MallocBugVisitor(Sym));
  C.EmitReport(R);
}

void MallocChecker::checkDeadSymbols(SymbolReaper &SymReaper,
                                     CheckerContext &C) const
{
  if (!SymReaper.hasDeadSymbols())
    return;

  ProgramStateRef state = C.getState();
  RegionStateTy RS = state->get<RegionState>();
  RegionStateTy::Factory &F = state->get_context<RegionState>();

  bool generateReport = false;
  llvm::SmallVector<SymbolRef, 2> Errors;
  for (RegionStateTy::iterator I = RS.begin(), E = RS.end(); I != E; ++I) {
    if (SymReaper.isDead(I->first)) {
      if (I->second.isAllocated()) {
        generateReport = true;
        Errors.push_back(I->first);
      }
      // Remove the dead symbol from the map.
      RS = F.remove(RS, I->first);

    }
  }
  
  // Cleanup the Realloc Pairs Map.
  ReallocMap RP = state->get<ReallocPairs>();
  for (ReallocMap::iterator I = RP.begin(), E = RP.end(); I != E; ++I) {
    if (SymReaper.isDead(I->first) ||
        SymReaper.isDead(I->second.ReallocatedSym)) {
      state = state->remove<ReallocPairs>(I->first);
    }
  }

  // Generate leak node.
  static SimpleProgramPointTag Tag("MallocChecker : DeadSymbolsLeak");
  ExplodedNode *N = C.addTransition(C.getState(), C.getPredecessor(), &Tag);

  if (generateReport) {
    for (llvm::SmallVector<SymbolRef, 2>::iterator
         I = Errors.begin(), E = Errors.end(); I != E; ++I) {
      reportLeak(*I, N, C);
    }
  }
  C.addTransition(state->set<RegionState>(RS), N);
}

void MallocChecker::checkEndPath(CheckerContext &C) const {
  ProgramStateRef state = C.getState();
  RegionStateTy M = state->get<RegionState>();

  // If inside inlined call, skip it.
  if (C.getLocationContext()->getParent() != 0)
    return;

  for (RegionStateTy::iterator I = M.begin(), E = M.end(); I != E; ++I) {
    RefState RS = I->second;
    if (RS.isAllocated()) {
      ExplodedNode *N = C.addTransition(state);
      if (N)
        reportLeak(I->first, N, C);
    }
  }
}

bool MallocChecker::checkEscape(SymbolRef Sym, const Stmt *S,
                                CheckerContext &C) const {
  ProgramStateRef state = C.getState();
  const RefState *RS = state->get<RegionState>(Sym);
  if (!RS)
    return false;

  if (RS->isAllocated()) {
    state = state->set<RegionState>(Sym, RefState::getEscaped(S));
    C.addTransition(state);
    return true;
  }
  return false;
}

void MallocChecker::checkPreStmt(const CallExpr *CE, CheckerContext &C) const {
  if (isMemFunction(C.getCalleeDecl(CE), C.getASTContext()))
    return;

  // Check use after free, when a freed pointer is passed to a call.
  ProgramStateRef State = C.getState();
  for (CallExpr::const_arg_iterator I = CE->arg_begin(),
                                    E = CE->arg_end(); I != E; ++I) {
    const Expr *A = *I;
    if (A->getType().getTypePtr()->isAnyPointerType()) {
      SymbolRef Sym = State->getSVal(A, C.getLocationContext()).getAsSymbol();
      if (!Sym)
        continue;
      if (checkUseAfterFree(Sym, C, A))
        return;
    }
  }
}

void MallocChecker::checkPreStmt(const ReturnStmt *S, CheckerContext &C) const {
  const Expr *E = S->getRetValue();
  if (!E)
    return;

  // Check if we are returning a symbol.
  SVal RetVal = C.getState()->getSVal(E, C.getLocationContext());
  SymbolRef Sym = RetVal.getAsSymbol();
  if (!Sym)
    // If we are returning a field of the allocated struct or an array element,
    // the callee could still free the memory.
    // TODO: This logic should be a part of generic symbol escape callback.
    if (const MemRegion *MR = RetVal.getAsRegion())
      if (isa<FieldRegion>(MR) || isa<ElementRegion>(MR))
        if (const SymbolicRegion *BMR =
              dyn_cast<SymbolicRegion>(MR->getBaseRegion()))
          Sym = BMR->getSymbol();
  if (!Sym)
    return;

  // Check if we are returning freed memory.
  if (checkUseAfterFree(Sym, C, E))
    return;

  // If this function body is not inlined, check if the symbol is escaping.
  if (C.getLocationContext()->getParent() == 0)
    checkEscape(Sym, E, C);
}

// TODO: Blocks should be either inlined or should call invalidate regions
// upon invocation. After that's in place, special casing here will not be 
// needed.
void MallocChecker::checkPostStmt(const BlockExpr *BE,
                                  CheckerContext &C) const {

  // Scan the BlockDecRefExprs for any object the retain count checker
  // may be tracking.
  if (!BE->getBlockDecl()->hasCaptures())
    return;

  ProgramStateRef state = C.getState();
  const BlockDataRegion *R =
    cast<BlockDataRegion>(state->getSVal(BE,
                                         C.getLocationContext()).getAsRegion());

  BlockDataRegion::referenced_vars_iterator I = R->referenced_vars_begin(),
                                            E = R->referenced_vars_end();

  if (I == E)
    return;

  SmallVector<const MemRegion*, 10> Regions;
  const LocationContext *LC = C.getLocationContext();
  MemRegionManager &MemMgr = C.getSValBuilder().getRegionManager();

  for ( ; I != E; ++I) {
    const VarRegion *VR = *I;
    if (VR->getSuperRegion() == R) {
      VR = MemMgr.getVarRegion(VR->getDecl(), LC);
    }
    Regions.push_back(VR);
  }

  state =
    state->scanReachableSymbols<StopTrackingCallback>(Regions.data(),
                                    Regions.data() + Regions.size()).getState();
  C.addTransition(state);
}

bool MallocChecker::checkUseAfterFree(SymbolRef Sym, CheckerContext &C,
                                      const Stmt *S) const {
  assert(Sym);
  const RefState *RS = C.getState()->get<RegionState>(Sym);
  if (RS && RS->isReleased()) {
    if (ExplodedNode *N = C.generateSink()) {
      if (!BT_UseFree)
        BT_UseFree.reset(new BugType("Use-after-free", "Memory Error"));

      BugReport *R = new BugReport(*BT_UseFree,
                                   "Use of memory after it is freed",N);
      if (S)
        R->addRange(S->getSourceRange());
      R->markInteresting(Sym);
      R->addVisitor(new MallocBugVisitor(Sym));
      C.EmitReport(R);
      return true;
    }
  }
  return false;
}

// Check if the location is a freed symbolic region.
void MallocChecker::checkLocation(SVal l, bool isLoad, const Stmt *S,
                                  CheckerContext &C) const {
  SymbolRef Sym = l.getLocSymbolInBase();
  if (Sym)
    checkUseAfterFree(Sym, C);
}

//===----------------------------------------------------------------------===//
// Check various ways a symbol can be invalidated.
// TODO: This logic (the next 3 functions) is copied/similar to the
// RetainRelease checker. We might want to factor this out.
//===----------------------------------------------------------------------===//

// Stop tracking symbols when a value escapes as a result of checkBind.
// A value escapes in three possible cases:
// (1) we are binding to something that is not a memory region.
// (2) we are binding to a memregion that does not have stack storage
// (3) we are binding to a memregion with stack storage that the store
//     does not understand.
void MallocChecker::checkBind(SVal loc, SVal val, const Stmt *S,
                              CheckerContext &C) const {
  // Are we storing to something that causes the value to "escape"?
  bool escapes = true;
  ProgramStateRef state = C.getState();

  if (loc::MemRegionVal *regionLoc = dyn_cast<loc::MemRegionVal>(&loc)) {
    escapes = !regionLoc->getRegion()->hasStackStorage();

    if (!escapes) {
      // To test (3), generate a new state with the binding added.  If it is
      // the same state, then it escapes (since the store cannot represent
      // the binding).
      escapes = (state == (state->bindLoc(*regionLoc, val)));
    }
    if (!escapes) {
      // Case 4: We do not currently model what happens when a symbol is
      // assigned to a struct field, so be conservative here and let the symbol
      // go. TODO: This could definitely be improved upon.
      escapes = !isa<VarRegion>(regionLoc->getRegion());
    }
  }

  // If our store can represent the binding and we aren't storing to something
  // that doesn't have local storage then just return and have the simulation
  // state continue as is.
  if (!escapes)
      return;

  // Otherwise, find all symbols referenced by 'val' that we are tracking
  // and stop tracking them.
  state = state->scanReachableSymbols<StopTrackingCallback>(val).getState();
  C.addTransition(state);
}

// If a symbolic region is assumed to NULL (or another constant), stop tracking
// it - assuming that allocation failed on this path.
ProgramStateRef MallocChecker::evalAssume(ProgramStateRef state,
                                              SVal Cond,
                                              bool Assumption) const {
  RegionStateTy RS = state->get<RegionState>();
  for (RegionStateTy::iterator I = RS.begin(), E = RS.end(); I != E; ++I) {
    // If the symbol is assumed to NULL or another constant, this will
    // return an APSInt*.
    if (state->getSymVal(I.getKey()))
      state = state->remove<RegionState>(I.getKey());
  }

  // Realloc returns 0 when reallocation fails, which means that we should
  // restore the state of the pointer being reallocated.
  ReallocMap RP = state->get<ReallocPairs>();
  for (ReallocMap::iterator I = RP.begin(), E = RP.end(); I != E; ++I) {
    // If the symbol is assumed to NULL or another constant, this will
    // return an APSInt*.
    if (state->getSymVal(I.getKey())) {
      SymbolRef ReallocSym = I.getData().ReallocatedSym;
      const RefState *RS = state->get<RegionState>(ReallocSym);
      if (RS) {
        if (RS->isReleased() && ! I.getData().IsFreeOnFailure)
          state = state->set<RegionState>(ReallocSym,
                             RefState::getAllocateUnchecked(RS->getStmt()));
      }
      state = state->remove<ReallocPairs>(I.getKey());
    }
  }

  return state;
}

// Check if the function is known to us. So, for example, we could
// conservatively assume it can free/reallocate it's pointer arguments.
// (We assume that the pointers cannot escape through calls to system
// functions not handled by this checker.)
bool MallocChecker::doesNotFreeMemory(const CallOrObjCMessage *Call,
                                      ProgramStateRef State) const {
  if (!Call)
    return false;

  // For now, assume that any C++ call can free memory.
  // TODO: If we want to be more optimistic here, we'll need to make sure that
  // regions escape to C++ containers. They seem to do that even now, but for
  // mysterious reasons.
  if (Call->isCXXCall())
    return false;

  const Decl *D = Call->getDecl();
  if (!D)
    return false;

  ASTContext &ASTC = State->getStateManager().getContext();

  // If it's one of the allocation functions we can reason about, we model
  // its behavior explicitly.
  if (isa<FunctionDecl>(D) && isMemFunction(cast<FunctionDecl>(D), ASTC)) {
    return true;
  }

  // If it's not a system call, assume it frees memory.
  SourceManager &SM = ASTC.getSourceManager();
  if (!SM.isInSystemHeader(D->getLocation()))
    return false;

  // Process C/ObjC functions.
  if (const FunctionDecl *FD  = dyn_cast<FunctionDecl>(D)) {
    // White list the system functions whose arguments escape.
    const IdentifierInfo *II = FD->getIdentifier();
    if (!II)
      return true;
    StringRef FName = II->getName();

    // White list thread local storage.
    if (FName.equals("pthread_setspecific"))
      return false;

    // White list the 'XXXNoCopy' ObjC functions.
    if (FName.endswith("NoCopy")) {
      // Look for the deallocator argument. We know that the memory ownership
      // is not transfered only if the deallocator argument is
      // 'kCFAllocatorNull'.
      for (unsigned i = 1; i < Call->getNumArgs(); ++i) {
        const Expr *ArgE = Call->getArg(i)->IgnoreParenCasts();
        if (const DeclRefExpr *DE = dyn_cast<DeclRefExpr>(ArgE)) {
          StringRef DeallocatorName = DE->getFoundDecl()->getName();
          if (DeallocatorName == "kCFAllocatorNull")
            return true;
        }
      }
      return false;
    }

    // PR12101
    // Many CoreFoundation and CoreGraphics might allow a tracked object 
    // to escape.
    if (Call->isCFCGAllowingEscape(FName))
      return false;

    // Associating streams with malloced buffers. The pointer can escape if
    // 'closefn' is specified (and if that function does free memory).
    // Currently, we do not inspect the 'closefn' function (PR12101).
    if (FName == "funopen")
      if (Call->getNumArgs() >= 4 && !Call->getArgSVal(4).isConstant(0))
        return false;

    // Do not warn on pointers passed to 'setbuf' when used with std streams,
    // these leaks might be intentional when setting the buffer for stdio.
    // http://stackoverflow.com/questions/2671151/who-frees-setvbuf-buffer
    if (FName == "setbuf" || FName =="setbuffer" ||
        FName == "setlinebuf" || FName == "setvbuf") {
      if (Call->getNumArgs() >= 1)
        if (const DeclRefExpr *Arg =
              dyn_cast<DeclRefExpr>(Call->getArg(0)->IgnoreParenCasts()))
          if (const VarDecl *D = dyn_cast<VarDecl>(Arg->getDecl()))
              if (D->getCanonicalDecl()->getName().find("std")
                                                   != StringRef::npos)
                return false;
    }

    // A bunch of other functions, which take ownership of a pointer (See retain
    // release checker). Not all the parameters here are invalidated, but the
    // Malloc checker cannot differentiate between them. The right way of doing
    // this would be to implement a pointer escapes callback.
    if (FName == "CVPixelBufferCreateWithBytes" ||
        FName == "CGBitmapContextCreateWithData" ||
        FName == "CVPixelBufferCreateWithPlanarBytes" ||
        FName == "OSAtomicEnqueue") {
      return false;
    }

    // Otherwise, assume that the function does not free memory.
    // Most system calls, do not free the memory.
    return true;

  // Process ObjC functions.
  } else if (const ObjCMethodDecl * ObjCD = dyn_cast<ObjCMethodDecl>(D)) {
    Selector S = ObjCD->getSelector();

    // White list the ObjC functions which do free memory.
    // - Anything containing 'freeWhenDone' param set to 1.
    //   Ex: dataWithBytesNoCopy:length:freeWhenDone.
    for (unsigned i = 1; i < S.getNumArgs(); ++i) {
      if (S.getNameForSlot(i).equals("freeWhenDone")) {
        if (Call->getArgSVal(i).isConstant(1))
          return false;
        else
          return true;
      }
    }

    // If the first selector ends with NoCopy, assume that the ownership is
    // transfered as well.
    // Ex:  [NSData dataWithBytesNoCopy:bytes length:10];
    if (S.getNameForSlot(0).endswith("NoCopy")) {
      return false;
    }

    // Otherwise, assume that the function does not free memory.
    // Most system calls, do not free the memory.
    return true;
  }

  // Otherwise, assume that the function can free memory.
  return false;

}

// If the symbol we are tracking is invalidated, but not explicitly (ex: the &p
// escapes, when we are tracking p), do not track the symbol as we cannot reason
// about it anymore.
ProgramStateRef
MallocChecker::checkRegionChanges(ProgramStateRef State,
                            const StoreManager::InvalidatedSymbols *invalidated,
                                    ArrayRef<const MemRegion *> ExplicitRegions,
                                    ArrayRef<const MemRegion *> Regions,
                                    const CallOrObjCMessage *Call) const {
  if (!invalidated || invalidated->empty())
    return State;
  llvm::SmallPtrSet<SymbolRef, 8> WhitelistedSymbols;

  // If it's a call which might free or reallocate memory, we assume that all
  // regions (explicit and implicit) escaped.

  // Otherwise, whitelist explicit pointers; we still can track them.
  if (!Call || doesNotFreeMemory(Call, State)) {
    for (ArrayRef<const MemRegion *>::iterator I = ExplicitRegions.begin(),
        E = ExplicitRegions.end(); I != E; ++I) {
      if (const SymbolicRegion *R = (*I)->StripCasts()->getAs<SymbolicRegion>())
        WhitelistedSymbols.insert(R->getSymbol());
    }
  }

  for (StoreManager::InvalidatedSymbols::const_iterator I=invalidated->begin(),
       E = invalidated->end(); I!=E; ++I) {
    SymbolRef sym = *I;
    if (WhitelistedSymbols.count(sym))
      continue;
    // The symbol escaped.
    if (const RefState *RS = State->get<RegionState>(sym))
      State = State->set<RegionState>(sym, RefState::getEscaped(RS->getStmt()));
  }
  return State;
}

static SymbolRef findFailedReallocSymbol(ProgramStateRef currState,
                                         ProgramStateRef prevState) {
  ReallocMap currMap = currState->get<ReallocPairs>();
  ReallocMap prevMap = prevState->get<ReallocPairs>();

  for (ReallocMap::iterator I = prevMap.begin(), E = prevMap.end();
       I != E; ++I) {
    SymbolRef sym = I.getKey();
    if (!currMap.lookup(sym))
      return sym;
  }

  return NULL;
}

PathDiagnosticPiece *
MallocChecker::MallocBugVisitor::VisitNode(const ExplodedNode *N,
                                           const ExplodedNode *PrevN,
                                           BugReporterContext &BRC,
                                           BugReport &BR) {
  ProgramStateRef state = N->getState();
  ProgramStateRef statePrev = PrevN->getState();

  const RefState *RS = state->get<RegionState>(Sym);
  const RefState *RSPrev = statePrev->get<RegionState>(Sym);
  if (!RS && !RSPrev)
    return 0;

  const Stmt *S = 0;
  const char *Msg = 0;
  StackHintGeneratorForSymbol *StackHint = 0;

  // Retrieve the associated statement.
  ProgramPoint ProgLoc = N->getLocation();
  if (isa<StmtPoint>(ProgLoc))
    S = cast<StmtPoint>(ProgLoc).getStmt();
  // If an assumption was made on a branch, it should be caught
  // here by looking at the state transition.
  if (isa<BlockEdge>(ProgLoc)) {
    const CFGBlock *srcBlk = cast<BlockEdge>(ProgLoc).getSrc();
    S = srcBlk->getTerminator();
  }
  if (!S)
    return 0;

  // Find out if this is an interesting point and what is the kind.
  if (Mode == Normal) {
    if (isAllocated(RS, RSPrev, S)) {
      Msg = "Memory is allocated";
      StackHint = new StackHintGeneratorForSymbol(Sym,
                                                  "Returned allocated memory");
    } else if (isReleased(RS, RSPrev, S)) {
      Msg = "Memory is released";
      StackHint = new StackHintGeneratorForSymbol(Sym,
                                                  "Returned released memory");
    } else if (isReallocFailedCheck(RS, RSPrev, S)) {
      Mode = ReallocationFailed;
      Msg = "Reallocation failed";
      StackHint = new StackHintGeneratorForReallocationFailed(Sym,
                                                       "Reallocation failed");

      if (SymbolRef sym = findFailedReallocSymbol(state, statePrev)) {
        // Is it possible to fail two reallocs WITHOUT testing in between?
        assert((!FailedReallocSymbol || FailedReallocSymbol == sym) &&
          "We only support one failed realloc at a time.");
        BR.markInteresting(sym);
        FailedReallocSymbol = sym;
      }
    }

  // We are in a special mode if a reallocation failed later in the path.
  } else if (Mode == ReallocationFailed) {
    assert(FailedReallocSymbol && "No symbol to look for.");

    // Is this is the first appearance of the reallocated symbol?
    if (!statePrev->get<RegionState>(FailedReallocSymbol)) {
      // If we ever hit this assert, that means BugReporter has decided to skip
      // node pairs or visit them out of order.
      assert(state->get<RegionState>(FailedReallocSymbol) &&
        "Missed the reallocation point");

      // We're at the reallocation point.
      Msg = "Attempt to reallocate memory";
      StackHint = new StackHintGeneratorForSymbol(Sym,
                                                 "Returned reallocated memory");
      FailedReallocSymbol = NULL;
      Mode = Normal;
    }
  }

  if (!Msg)
    return 0;
  assert(StackHint);

  // Generate the extra diagnostic.
  PathDiagnosticLocation Pos(S, BRC.getSourceManager(),
                             N->getLocationContext());
  return new PathDiagnosticEventPiece(Pos, Msg, true, StackHint);
}


#define REGISTER_CHECKER(name) \
void ento::register##name(CheckerManager &mgr) {\
  registerCStringCheckerBasic(mgr); \
  mgr.registerChecker<MallocChecker>()->Filter.C##name = true;\
}

REGISTER_CHECKER(MallocPessimistic)
REGISTER_CHECKER(MallocOptimistic)
