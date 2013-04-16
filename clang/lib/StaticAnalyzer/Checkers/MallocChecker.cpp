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
#include "clang/AST/Attr.h"
#include "clang/Basic/SourceManager.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SymbolManager.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include <climits>

using namespace clang;
using namespace ento;

namespace {

// Used to check correspondence between allocators and deallocators.
enum AllocationFamily {
  AF_None,
  AF_Malloc,
  AF_CXXNew,
  AF_CXXNewArray
};

class RefState {
  enum Kind { // Reference to allocated memory.
              Allocated,
              // Reference to released/freed memory.
              Released,
              // The responsibility for freeing resources has transfered from
              // this reference. A relinquished symbol should not be freed.
              Relinquished,
              // We are no longer guaranteed to have observed all manipulations
              // of this pointer/memory. For example, it could have been
              // passed as a parameter to an opaque function.
              Escaped
  };

  const Stmt *S;
  unsigned K : 2; // Kind enum, but stored as a bitfield.
  unsigned Family : 30; // Rest of 32-bit word, currently just an allocation 
                        // family.

  RefState(Kind k, const Stmt *s, unsigned family) 
    : S(s), K(k), Family(family) {
    assert(family != AF_None);
  }
public:
  bool isAllocated() const { return K == Allocated; }
  bool isReleased() const { return K == Released; }
  bool isRelinquished() const { return K == Relinquished; }
  bool isEscaped() const { return K == Escaped; }
  AllocationFamily getAllocationFamily() const {
    return (AllocationFamily)Family;
  }
  const Stmt *getStmt() const { return S; }

  bool operator==(const RefState &X) const {
    return K == X.K && S == X.S && Family == X.Family;
  }

  static RefState getAllocated(unsigned family, const Stmt *s) {
    return RefState(Allocated, s, family);
  }
  static RefState getReleased(unsigned family, const Stmt *s) { 
    return RefState(Released, s, family);
  }
  static RefState getRelinquished(unsigned family, const Stmt *s) {
    return RefState(Relinquished, s, family);
  }
  static RefState getEscaped(const RefState *RS) {
    return RefState(Escaped, RS->getStmt(), RS->getAllocationFamily());
  }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddInteger(K);
    ID.AddPointer(S);
    ID.AddInteger(Family);
  }

  void dump(raw_ostream &OS) const {
    static const char *Table[] = {
      "Allocated",
      "Released",
      "Relinquished"
    };
    OS << Table[(unsigned) K];
  }

  LLVM_ATTRIBUTE_USED void dump() const {
    dump(llvm::errs());
  }
};

enum ReallocPairKind {
  RPToBeFreedAfterFailure,
  // The symbol has been freed when reallocation failed.
  RPIsFreeOnFailure,
  // The symbol does not need to be freed after reallocation fails.
  RPDoNotTrackAfterFailure
};

/// \class ReallocPair
/// \brief Stores information about the symbol being reallocated by a call to
/// 'realloc' to allow modeling failed reallocation later in the path.
struct ReallocPair {
  // \brief The symbol which realloc reallocated.
  SymbolRef ReallocatedSym;
  ReallocPairKind Kind;

  ReallocPair(SymbolRef S, ReallocPairKind K) :
    ReallocatedSym(S), Kind(K) {}
  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddInteger(Kind);
    ID.AddPointer(ReallocatedSym);
  }
  bool operator==(const ReallocPair &X) const {
    return ReallocatedSym == X.ReallocatedSym &&
           Kind == X.Kind;
  }
};

typedef std::pair<const ExplodedNode*, const MemRegion*> LeakInfo;

class MallocChecker : public Checker<check::DeadSymbols,
                                     check::PointerEscape,
                                     check::ConstPointerEscape,
                                     check::PreStmt<ReturnStmt>,
                                     check::PreCall,
                                     check::PostStmt<CallExpr>,
                                     check::PostStmt<CXXNewExpr>,
                                     check::PreStmt<CXXDeleteExpr>,
                                     check::PostStmt<BlockExpr>,
                                     check::PostObjCMessage,
                                     check::Location,
                                     eval::Assume>
{
  mutable OwningPtr<BugType> BT_DoubleFree;
  mutable OwningPtr<BugType> BT_Leak;
  mutable OwningPtr<BugType> BT_UseFree;
  mutable OwningPtr<BugType> BT_BadFree;
  mutable OwningPtr<BugType> BT_MismatchedDealloc;
  mutable OwningPtr<BugType> BT_OffsetFree;
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
    DefaultBool CNewDeleteChecker;
    DefaultBool CNewDeleteLeaksChecker;
    DefaultBool CMismatchedDeallocatorChecker;
  };

  ChecksFilter Filter;

  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;
  void checkPostStmt(const CallExpr *CE, CheckerContext &C) const;
  void checkPostStmt(const CXXNewExpr *NE, CheckerContext &C) const;
  void checkPreStmt(const CXXDeleteExpr *DE, CheckerContext &C) const;
  void checkPostObjCMessage(const ObjCMethodCall &Call, CheckerContext &C) const;
  void checkPostStmt(const BlockExpr *BE, CheckerContext &C) const;
  void checkDeadSymbols(SymbolReaper &SymReaper, CheckerContext &C) const;
  void checkPreStmt(const ReturnStmt *S, CheckerContext &C) const;
  ProgramStateRef evalAssume(ProgramStateRef state, SVal Cond,
                            bool Assumption) const;
  void checkLocation(SVal l, bool isLoad, const Stmt *S,
                     CheckerContext &C) const;

  ProgramStateRef checkPointerEscape(ProgramStateRef State,
                                    const InvalidatedSymbols &Escaped,
                                    const CallEvent *Call,
                                    PointerEscapeKind Kind) const;
  ProgramStateRef checkConstPointerEscape(ProgramStateRef State,
                                          const InvalidatedSymbols &Escaped,
                                          const CallEvent *Call,
                                          PointerEscapeKind Kind) const;

  void printState(raw_ostream &Out, ProgramStateRef State,
                  const char *NL, const char *Sep) const;

private:
  void initIdentifierInfo(ASTContext &C) const;

  /// \brief Determine family of a deallocation expression.
  AllocationFamily getAllocationFamily(CheckerContext &C, const Stmt *S) const;

  /// \brief Print names of allocators and deallocators.
  ///
  /// \returns true on success.
  bool printAllocDeallocName(raw_ostream &os, CheckerContext &C, 
                             const Expr *E) const;

  /// \brief Print expected name of an allocator based on the deallocator's
  /// family derived from the DeallocExpr.
  void printExpectedAllocName(raw_ostream &os, CheckerContext &C, 
                              const Expr *DeallocExpr) const;
  /// \brief Print expected name of a deallocator based on the allocator's 
  /// family.
  void printExpectedDeallocName(raw_ostream &os, AllocationFamily Family) const;

  ///@{
  /// Check if this is one of the functions which can allocate/reallocate memory 
  /// pointed to by one of its arguments.
  bool isMemFunction(const FunctionDecl *FD, ASTContext &C) const;
  bool isFreeFunction(const FunctionDecl *FD, ASTContext &C) const;
  bool isAllocationFunction(const FunctionDecl *FD, ASTContext &C) const;
  bool isStandardNewDelete(const FunctionDecl *FD, ASTContext &C) const;
  ///@}
  static ProgramStateRef MallocMemReturnsAttr(CheckerContext &C,
                                              const CallExpr *CE,
                                              const OwnershipAttr* Att);
  static ProgramStateRef MallocMemAux(CheckerContext &C, const CallExpr *CE,
                                     const Expr *SizeEx, SVal Init,
                                     ProgramStateRef State,
                                     AllocationFamily Family = AF_Malloc) {
    return MallocMemAux(C, CE,
                        State->getSVal(SizeEx, C.getLocationContext()),
                        Init, State, Family);
  }

  static ProgramStateRef MallocMemAux(CheckerContext &C, const CallExpr *CE,
                                     SVal SizeEx, SVal Init,
                                     ProgramStateRef State,
                                     AllocationFamily Family = AF_Malloc);

  /// Update the RefState to reflect the new memory allocation.
  static ProgramStateRef 
  MallocUpdateRefState(CheckerContext &C, const Expr *E, ProgramStateRef State,
                       AllocationFamily Family = AF_Malloc);

  ProgramStateRef FreeMemAttr(CheckerContext &C, const CallExpr *CE,
                              const OwnershipAttr* Att) const;
  ProgramStateRef FreeMemAux(CheckerContext &C, const CallExpr *CE,
                             ProgramStateRef state, unsigned Num,
                             bool Hold,
                             bool &ReleasedAllocated,
                             bool ReturnsNullOnFailure = false) const;
  ProgramStateRef FreeMemAux(CheckerContext &C, const Expr *Arg,
                             const Expr *ParentExpr,
                             ProgramStateRef State,
                             bool Hold,
                             bool &ReleasedAllocated,
                             bool ReturnsNullOnFailure = false) const;

  ProgramStateRef ReallocMem(CheckerContext &C, const CallExpr *CE,
                             bool FreesMemOnFailure) const;
  static ProgramStateRef CallocMem(CheckerContext &C, const CallExpr *CE);
  
  ///\brief Check if the memory associated with this symbol was released.
  bool isReleased(SymbolRef Sym, CheckerContext &C) const;

  bool checkUseAfterFree(SymbolRef Sym, CheckerContext &C, const Stmt *S) const;

  /// Check if the function is known not to free memory, or if it is
  /// "interesting" and should be modeled explicitly.
  ///
  /// We assume that pointers do not escape through calls to system functions
  /// not handled by this checker.
  bool doesNotFreeMemOrInteresting(const CallEvent *Call,
                                   ProgramStateRef State) const;

  // Implementation of the checkPointerEscape callabcks.
  ProgramStateRef checkPointerEscapeAux(ProgramStateRef State,
                                  const InvalidatedSymbols &Escaped,
                                  const CallEvent *Call,
                                  PointerEscapeKind Kind,
                                  bool(*CheckRefState)(const RefState*)) const;

  ///@{
  /// Tells if a given family/call/symbol is tracked by the current checker.
  bool isTrackedByCurrentChecker(AllocationFamily Family) const;
  bool isTrackedByCurrentChecker(CheckerContext &C,
                                 const Stmt *AllocDeallocStmt) const;
  bool isTrackedByCurrentChecker(CheckerContext &C, SymbolRef Sym) const;
  ///@}
  static bool SummarizeValue(raw_ostream &os, SVal V);
  static bool SummarizeRegion(raw_ostream &os, const MemRegion *MR);
  void ReportBadFree(CheckerContext &C, SVal ArgVal, SourceRange Range, 
                     const Expr *DeallocExpr) const;
  void ReportMismatchedDealloc(CheckerContext &C, SourceRange Range,
                               const Expr *DeallocExpr, const RefState *RS,
                               SymbolRef Sym) const;
  void ReportOffsetFree(CheckerContext &C, SVal ArgVal, SourceRange Range, 
                        const Expr *DeallocExpr, 
                        const Expr *AllocExpr = 0) const;
  void ReportUseAfterFree(CheckerContext &C, SourceRange Range,
                          SymbolRef Sym) const;
  void ReportDoubleFree(CheckerContext &C, SourceRange Range, bool Released,
                        SymbolRef Sym, SymbolRef PrevSym) const;

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

    bool IsLeak;

  public:
    MallocBugVisitor(SymbolRef S, bool isLeak = false)
       : Sym(S), Mode(Normal), FailedReallocSymbol(0), IsLeak(isLeak) {}

    virtual ~MallocBugVisitor() {}

    void Profile(llvm::FoldingSetNodeID &ID) const {
      static int X = 0;
      ID.AddPointer(&X);
      ID.AddPointer(Sym);
    }

    inline bool isAllocated(const RefState *S, const RefState *SPrev,
                            const Stmt *Stmt) {
      // Did not track -> allocated. Other state (released) -> allocated.
      return (Stmt && (isa<CallExpr>(Stmt) || isa<CXXNewExpr>(Stmt)) &&
              (S && S->isAllocated()) && (!SPrev || !SPrev->isAllocated()));
    }

    inline bool isReleased(const RefState *S, const RefState *SPrev,
                           const Stmt *Stmt) {
      // Did not track -> released. Other state (allocated) -> released.
      return (Stmt && (isa<CallExpr>(Stmt) || isa<CXXDeleteExpr>(Stmt)) &&
              (S && S->isReleased()) && (!SPrev || !SPrev->isReleased()));
    }

    inline bool isRelinquished(const RefState *S, const RefState *SPrev,
                               const Stmt *Stmt) {
      // Did not track -> relinquished. Other state (allocated) -> relinquished.
      return (Stmt && (isa<CallExpr>(Stmt) || isa<ObjCMessageExpr>(Stmt) ||
                                              isa<ObjCPropertyRefExpr>(Stmt)) &&
              (S && S->isRelinquished()) &&
              (!SPrev || !SPrev->isRelinquished()));
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

    PathDiagnosticPiece* getEndPath(BugReporterContext &BRC,
                                    const ExplodedNode *EndPathNode,
                                    BugReport &BR) {
      if (!IsLeak)
        return 0;

      PathDiagnosticLocation L =
        PathDiagnosticLocation::createEndOfPath(EndPathNode,
                                                BRC.getSourceManager());
      // Do not add the statement itself as a range in case of leak.
      return new PathDiagnosticEventPiece(L, BR.getDescription(), false);
    }

  private:
    class StackHintGeneratorForReallocationFailed
        : public StackHintGeneratorForSymbol {
    public:
      StackHintGeneratorForReallocationFailed(SymbolRef S, StringRef M)
        : StackHintGeneratorForSymbol(S, M) {}

      virtual std::string getMessageForArg(const Expr *ArgE, unsigned ArgIndex) {
        // Printed parameters start at 1, not 0.
        ++ArgIndex;

        SmallString<200> buf;
        llvm::raw_svector_ostream os(buf);

        os << "Reallocation of " << ArgIndex << llvm::getOrdinalSuffix(ArgIndex)
           << " parameter failed";

        return os.str();
      }

      virtual std::string getMessageForReturn(const CallExpr *CallExpr) {
        return "Reallocation of returned value failed";
      }
    };
  };
};
} // end anonymous namespace

REGISTER_MAP_WITH_PROGRAMSTATE(RegionState, SymbolRef, RefState)
REGISTER_MAP_WITH_PROGRAMSTATE(ReallocPairs, SymbolRef, ReallocPair)

// A map from the freed symbol to the symbol representing the return value of 
// the free function.
REGISTER_MAP_WITH_PROGRAMSTATE(FreeReturnValue, SymbolRef, SymbolRef)

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
  if (II_malloc)
    return;
  II_malloc = &Ctx.Idents.get("malloc");
  II_free = &Ctx.Idents.get("free");
  II_realloc = &Ctx.Idents.get("realloc");
  II_reallocf = &Ctx.Idents.get("reallocf");
  II_calloc = &Ctx.Idents.get("calloc");
  II_valloc = &Ctx.Idents.get("valloc");
  II_strdup = &Ctx.Idents.get("strdup");
  II_strndup = &Ctx.Idents.get("strndup");
}

bool MallocChecker::isMemFunction(const FunctionDecl *FD, ASTContext &C) const {
  if (isFreeFunction(FD, C))
    return true;

  if (isAllocationFunction(FD, C))
    return true;

  if (isStandardNewDelete(FD, C))
    return true;

  return false;
}

bool MallocChecker::isAllocationFunction(const FunctionDecl *FD,
                                         ASTContext &C) const {
  if (!FD)
    return false;

  if (FD->getKind() == Decl::Function) {
    IdentifierInfo *FunI = FD->getIdentifier();
    initIdentifierInfo(C);

    if (FunI == II_malloc || FunI == II_realloc ||
        FunI == II_reallocf || FunI == II_calloc || FunI == II_valloc ||
        FunI == II_strdup || FunI == II_strndup)
      return true;
  }

  if (Filter.CMallocOptimistic && FD->hasAttrs())
    for (specific_attr_iterator<OwnershipAttr>
           i = FD->specific_attr_begin<OwnershipAttr>(),
           e = FD->specific_attr_end<OwnershipAttr>();
           i != e; ++i)
      if ((*i)->getOwnKind() == OwnershipAttr::Returns)
        return true;
  return false;
}

bool MallocChecker::isFreeFunction(const FunctionDecl *FD, ASTContext &C) const {
  if (!FD)
    return false;

  if (FD->getKind() == Decl::Function) {
    IdentifierInfo *FunI = FD->getIdentifier();
    initIdentifierInfo(C);

    if (FunI == II_free || FunI == II_realloc || FunI == II_reallocf)
      return true;
  }

  if (Filter.CMallocOptimistic && FD->hasAttrs())
    for (specific_attr_iterator<OwnershipAttr>
           i = FD->specific_attr_begin<OwnershipAttr>(),
           e = FD->specific_attr_end<OwnershipAttr>();
           i != e; ++i)
      if ((*i)->getOwnKind() == OwnershipAttr::Takes ||
          (*i)->getOwnKind() == OwnershipAttr::Holds)
        return true;
  return false;
}

// Tells if the callee is one of the following:
// 1) A global non-placement new/delete operator function.
// 2) A global placement operator function with the single placement argument
//    of type std::nothrow_t.
bool MallocChecker::isStandardNewDelete(const FunctionDecl *FD,
                                        ASTContext &C) const {
  if (!FD)
    return false;

  OverloadedOperatorKind Kind = FD->getOverloadedOperator();
  if (Kind != OO_New && Kind != OO_Array_New && 
      Kind != OO_Delete && Kind != OO_Array_Delete)
    return false;

  // Skip all operator new/delete methods.
  if (isa<CXXMethodDecl>(FD))
    return false;

  // Return true if tested operator is a standard placement nothrow operator.
  if (FD->getNumParams() == 2) {
    QualType T = FD->getParamDecl(1)->getType();
    if (const IdentifierInfo *II = T.getBaseTypeIdentifier())
      return II->getName().equals("nothrow_t");
  }

  // Skip placement operators.
  if (FD->getNumParams() != 1 || FD->isVariadic())
    return false;

  // One of the standard new/new[]/delete/delete[] non-placement operators.
  return true;
}

void MallocChecker::checkPostStmt(const CallExpr *CE, CheckerContext &C) const {
  if (C.wasInlined)
    return;
  
  const FunctionDecl *FD = C.getCalleeDecl(CE);
  if (!FD)
    return;

  ProgramStateRef State = C.getState();
  bool ReleasedAllocatedMemory = false;

  if (FD->getKind() == Decl::Function) {
    initIdentifierInfo(C.getASTContext());
    IdentifierInfo *FunI = FD->getIdentifier();

    if (FunI == II_malloc || FunI == II_valloc) {
      if (CE->getNumArgs() < 1)
        return;
      State = MallocMemAux(C, CE, CE->getArg(0), UndefinedVal(), State);
    } else if (FunI == II_realloc) {
      State = ReallocMem(C, CE, false);
    } else if (FunI == II_reallocf) {
      State = ReallocMem(C, CE, true);
    } else if (FunI == II_calloc) {
      State = CallocMem(C, CE);
    } else if (FunI == II_free) {
      State = FreeMemAux(C, CE, State, 0, false, ReleasedAllocatedMemory);
    } else if (FunI == II_strdup) {
      State = MallocUpdateRefState(C, CE, State);
    } else if (FunI == II_strndup) {
      State = MallocUpdateRefState(C, CE, State);
    }
    else if (isStandardNewDelete(FD, C.getASTContext())) {
      // Process direct calls to operator new/new[]/delete/delete[] functions
      // as distinct from new/new[]/delete/delete[] expressions that are 
      // processed by the checkPostStmt callbacks for CXXNewExpr and 
      // CXXDeleteExpr.
      OverloadedOperatorKind K = FD->getOverloadedOperator();
      if (K == OO_New)
        State = MallocMemAux(C, CE, CE->getArg(0), UndefinedVal(), State,
                             AF_CXXNew);
      else if (K == OO_Array_New)
        State = MallocMemAux(C, CE, CE->getArg(0), UndefinedVal(), State,
                             AF_CXXNewArray);
      else if (K == OO_Delete || K == OO_Array_Delete)
        State = FreeMemAux(C, CE, State, 0, false, ReleasedAllocatedMemory);
      else
        llvm_unreachable("not a new/delete operator");
    }
  }

  if (Filter.CMallocOptimistic || Filter.CMismatchedDeallocatorChecker) {
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

void MallocChecker::checkPostStmt(const CXXNewExpr *NE, 
                                  CheckerContext &C) const {

  if (NE->getNumPlacementArgs())
    for (CXXNewExpr::const_arg_iterator I = NE->placement_arg_begin(),
         E = NE->placement_arg_end(); I != E; ++I)
      if (SymbolRef Sym = C.getSVal(*I).getAsSymbol())
        checkUseAfterFree(Sym, C, *I);

  if (!isStandardNewDelete(NE->getOperatorNew(), C.getASTContext()))
    return;

  ProgramStateRef State = C.getState();
  // The return value from operator new is bound to a specified initialization 
  // value (if any) and we don't want to loose this value. So we call 
  // MallocUpdateRefState() instead of MallocMemAux() which breakes the 
  // existing binding.
  State = MallocUpdateRefState(C, NE, State, NE->isArray() ? AF_CXXNewArray 
                                                           : AF_CXXNew);
  C.addTransition(State);
}

void MallocChecker::checkPreStmt(const CXXDeleteExpr *DE, 
                                 CheckerContext &C) const {

  if (!Filter.CNewDeleteChecker)
    if (SymbolRef Sym = C.getSVal(DE->getArgument()).getAsSymbol())
      checkUseAfterFree(Sym, C, DE->getArgument());

  if (!isStandardNewDelete(DE->getOperatorDelete(), C.getASTContext()))
    return;

  ProgramStateRef State = C.getState();
  bool ReleasedAllocated;
  State = FreeMemAux(C, DE->getArgument(), DE, State,
                     /*Hold*/false, ReleasedAllocated);

  C.addTransition(State);
}

static bool isKnownDeallocObjCMethodName(const ObjCMethodCall &Call) {
  // If the first selector piece is one of the names below, assume that the
  // object takes ownership of the memory, promising to eventually deallocate it
  // with free().
  // Ex:  [NSData dataWithBytesNoCopy:bytes length:10];
  // (...unless a 'freeWhenDone' parameter is false, but that's checked later.)
  StringRef FirstSlot = Call.getSelector().getNameForSlot(0);
  if (FirstSlot == "dataWithBytesNoCopy" ||
      FirstSlot == "initWithBytesNoCopy" ||
      FirstSlot == "initWithCharactersNoCopy")
    return true;

  return false;
}

static Optional<bool> getFreeWhenDoneArg(const ObjCMethodCall &Call) {
  Selector S = Call.getSelector();

  // FIXME: We should not rely on fully-constrained symbols being folded.
  for (unsigned i = 1; i < S.getNumArgs(); ++i)
    if (S.getNameForSlot(i).equals("freeWhenDone"))
      return !Call.getArgSVal(i).isZeroConstant();

  return None;
}

void MallocChecker::checkPostObjCMessage(const ObjCMethodCall &Call,
                                         CheckerContext &C) const {
  if (C.wasInlined)
    return;

  if (!isKnownDeallocObjCMethodName(Call))
    return;

  if (Optional<bool> FreeWhenDone = getFreeWhenDoneArg(Call))
    if (!*FreeWhenDone)
      return;

  bool ReleasedAllocatedMemory;
  ProgramStateRef State = FreeMemAux(C, Call.getArgExpr(0),
                                     Call.getOriginExpr(), C.getState(),
                                     /*Hold=*/true, ReleasedAllocatedMemory,
                                     /*RetNullOnFailure=*/true);

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
                                           ProgramStateRef State,
                                           AllocationFamily Family) {

  // Bind the return value to the symbolic value from the heap region.
  // TODO: We could rewrite post visit to eval call; 'malloc' does not have
  // side effects other than what we model here.
  unsigned Count = C.blockCount();
  SValBuilder &svalBuilder = C.getSValBuilder();
  const LocationContext *LCtx = C.getPredecessor()->getLocationContext();
  DefinedSVal RetVal = svalBuilder.getConjuredHeapSymbolVal(CE, LCtx, Count)
      .castAs<DefinedSVal>();
  State = State->BindExpr(CE, C.getLocationContext(), RetVal);

  // We expect the malloc functions to return a pointer.
  if (!RetVal.getAs<Loc>())
    return 0;

  // Fill the region with the initialization value.
  State = State->bindDefault(RetVal, Init);

  // Set the region's extent equal to the Size parameter.
  const SymbolicRegion *R =
      dyn_cast_or_null<SymbolicRegion>(RetVal.getAsRegion());
  if (!R)
    return 0;
  if (Optional<DefinedOrUnknownSVal> DefinedSize =
          Size.getAs<DefinedOrUnknownSVal>()) {
    SValBuilder &svalBuilder = C.getSValBuilder();
    DefinedOrUnknownSVal Extent = R->getExtent(svalBuilder);
    DefinedOrUnknownSVal extentMatchesSize =
        svalBuilder.evalEQ(State, Extent, *DefinedSize);

    State = State->assume(extentMatchesSize, true);
    assert(State);
  }
  
  return MallocUpdateRefState(C, CE, State, Family);
}

ProgramStateRef MallocChecker::MallocUpdateRefState(CheckerContext &C,
                                                    const Expr *E,
                                                    ProgramStateRef State,
                                                    AllocationFamily Family) {
  // Get the return value.
  SVal retVal = State->getSVal(E, C.getLocationContext());

  // We expect the malloc functions to return a pointer.
  if (!retVal.getAs<Loc>())
    return 0;

  SymbolRef Sym = retVal.getAsLocSymbol();
  assert(Sym);

  // Set the symbol's state to Allocated.
  return State->set<RegionState>(Sym, RefState::getAllocated(Family, E));
}

ProgramStateRef MallocChecker::FreeMemAttr(CheckerContext &C,
                                           const CallExpr *CE,
                                           const OwnershipAttr* Att) const {
  if (Att->getModule() != "malloc")
    return 0;

  ProgramStateRef State = C.getState();
  bool ReleasedAllocated = false;

  for (OwnershipAttr::args_iterator I = Att->args_begin(), E = Att->args_end();
       I != E; ++I) {
    ProgramStateRef StateI = FreeMemAux(C, CE, State, *I,
                               Att->getOwnKind() == OwnershipAttr::Holds,
                               ReleasedAllocated);
    if (StateI)
      State = StateI;
  }
  return State;
}

ProgramStateRef MallocChecker::FreeMemAux(CheckerContext &C,
                                          const CallExpr *CE,
                                          ProgramStateRef state,
                                          unsigned Num,
                                          bool Hold,
                                          bool &ReleasedAllocated,
                                          bool ReturnsNullOnFailure) const {
  if (CE->getNumArgs() < (Num + 1))
    return 0;

  return FreeMemAux(C, CE->getArg(Num), CE, state, Hold,
                    ReleasedAllocated, ReturnsNullOnFailure);
}

/// Checks if the previous call to free on the given symbol failed - if free
/// failed, returns true. Also, returns the corresponding return value symbol.
static bool didPreviousFreeFail(ProgramStateRef State,
                                SymbolRef Sym, SymbolRef &RetStatusSymbol) {
  const SymbolRef *Ret = State->get<FreeReturnValue>(Sym);
  if (Ret) {
    assert(*Ret && "We should not store the null return symbol");
    ConstraintManager &CMgr = State->getConstraintManager();
    ConditionTruthVal FreeFailed = CMgr.isNull(State, *Ret);
    RetStatusSymbol = *Ret;
    return FreeFailed.isConstrainedTrue();
  }
  return false;
}

AllocationFamily MallocChecker::getAllocationFamily(CheckerContext &C, 
                                                    const Stmt *S) const {
  if (!S)
    return AF_None;

  if (const CallExpr *CE = dyn_cast<CallExpr>(S)) {
    const FunctionDecl *FD = C.getCalleeDecl(CE);

    if (!FD)
      FD = dyn_cast<FunctionDecl>(CE->getCalleeDecl());

    ASTContext &Ctx = C.getASTContext();

    if (isAllocationFunction(FD, Ctx) || isFreeFunction(FD, Ctx))
      return AF_Malloc;

    if (isStandardNewDelete(FD, Ctx)) {
      OverloadedOperatorKind Kind = FD->getOverloadedOperator();
      if (Kind == OO_New || Kind == OO_Delete)
        return AF_CXXNew;
      else if (Kind == OO_Array_New || Kind == OO_Array_Delete)
        return AF_CXXNewArray;
    }

    return AF_None;
  }

  if (const CXXNewExpr *NE = dyn_cast<CXXNewExpr>(S))
    return NE->isArray() ? AF_CXXNewArray : AF_CXXNew;

  if (const CXXDeleteExpr *DE = dyn_cast<CXXDeleteExpr>(S))
    return DE->isArrayForm() ? AF_CXXNewArray : AF_CXXNew;

  if (isa<ObjCMessageExpr>(S))
    return AF_Malloc;

  return AF_None;
}

bool MallocChecker::printAllocDeallocName(raw_ostream &os, CheckerContext &C, 
                                          const Expr *E) const {
  if (const CallExpr *CE = dyn_cast<CallExpr>(E)) {
    // FIXME: This doesn't handle indirect calls.
    const FunctionDecl *FD = CE->getDirectCallee();
    if (!FD)
      return false;
    
    os << *FD;
    if (!FD->isOverloadedOperator())
      os << "()";
    return true;
  }

  if (const ObjCMessageExpr *Msg = dyn_cast<ObjCMessageExpr>(E)) {
    if (Msg->isInstanceMessage())
      os << "-";
    else
      os << "+";
    os << Msg->getSelector().getAsString();
    return true;
  }

  if (const CXXNewExpr *NE = dyn_cast<CXXNewExpr>(E)) {
    os << "'" 
       << getOperatorSpelling(NE->getOperatorNew()->getOverloadedOperator())
       << "'";
    return true;
  }

  if (const CXXDeleteExpr *DE = dyn_cast<CXXDeleteExpr>(E)) {
    os << "'" 
       << getOperatorSpelling(DE->getOperatorDelete()->getOverloadedOperator())
       << "'";
    return true;
  }

  return false;
}

void MallocChecker::printExpectedAllocName(raw_ostream &os, CheckerContext &C,
                                           const Expr *E) const {
  AllocationFamily Family = getAllocationFamily(C, E);

  switch(Family) {
    case AF_Malloc: os << "malloc()"; return;
    case AF_CXXNew: os << "'new'"; return;
    case AF_CXXNewArray: os << "'new[]'"; return;
    case AF_None: llvm_unreachable("not a deallocation expression");
  }
}

void MallocChecker::printExpectedDeallocName(raw_ostream &os, 
                                             AllocationFamily Family) const {
  switch(Family) {
    case AF_Malloc: os << "free()"; return;
    case AF_CXXNew: os << "'delete'"; return;
    case AF_CXXNewArray: os << "'delete[]'"; return;
    case AF_None: llvm_unreachable("suspicious AF_None argument");
  }
}

ProgramStateRef MallocChecker::FreeMemAux(CheckerContext &C,
                                          const Expr *ArgExpr,
                                          const Expr *ParentExpr,
                                          ProgramStateRef State,
                                          bool Hold,
                                          bool &ReleasedAllocated,
                                          bool ReturnsNullOnFailure) const {

  SVal ArgVal = State->getSVal(ArgExpr, C.getLocationContext());
  if (!ArgVal.getAs<DefinedOrUnknownSVal>())
    return 0;
  DefinedOrUnknownSVal location = ArgVal.castAs<DefinedOrUnknownSVal>();

  // Check for null dereferences.
  if (!location.getAs<Loc>())
    return 0;

  // The explicit NULL case, no operation is performed.
  ProgramStateRef notNullState, nullState;
  llvm::tie(notNullState, nullState) = State->assume(location);
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
    ReportBadFree(C, ArgVal, ArgExpr->getSourceRange(), ParentExpr);
    return 0;
  }
  
  R = R->StripCasts();
  
  // Blocks might show up as heap data, but should not be free()d
  if (isa<BlockDataRegion>(R)) {
    ReportBadFree(C, ArgVal, ArgExpr->getSourceRange(), ParentExpr);
    return 0;
  }
  
  const MemSpaceRegion *MS = R->getMemorySpace();
  
  // Parameters, locals, statics, globals, and memory returned by alloca() 
  // shouldn't be freed.
  if (!(isa<UnknownSpaceRegion>(MS) || isa<HeapSpaceRegion>(MS))) {
    // FIXME: at the time this code was written, malloc() regions were
    // represented by conjured symbols, which are all in UnknownSpaceRegion.
    // This means that there isn't actually anything from HeapSpaceRegion
    // that should be freed, even though we allow it here.
    // Of course, free() can work on memory allocated outside the current
    // function, so UnknownSpaceRegion is always a possibility.
    // False negatives are better than false positives.
    
    ReportBadFree(C, ArgVal, ArgExpr->getSourceRange(), ParentExpr);
    return 0;
  }

  const SymbolicRegion *SrBase = dyn_cast<SymbolicRegion>(R->getBaseRegion());
  // Various cases could lead to non-symbol values here.
  // For now, ignore them.
  if (!SrBase)
    return 0;

  SymbolRef SymBase = SrBase->getSymbol();
  const RefState *RsBase = State->get<RegionState>(SymBase);
  SymbolRef PreviousRetStatusSymbol = 0;

  if (RsBase) {

    // Check for double free first.
    if ((RsBase->isReleased() || RsBase->isRelinquished()) &&
        !didPreviousFreeFail(State, SymBase, PreviousRetStatusSymbol)) {
      ReportDoubleFree(C, ParentExpr->getSourceRange(), RsBase->isReleased(),
                       SymBase, PreviousRetStatusSymbol);
      return 0;

    // If the pointer is allocated or escaped, but we are now trying to free it,
    // check that the call to free is proper.
    } else if (RsBase->isAllocated() || RsBase->isEscaped()) {

      // Check if an expected deallocation function matches the real one.
      bool DeallocMatchesAlloc =
        RsBase->getAllocationFamily() == getAllocationFamily(C, ParentExpr);
      if (!DeallocMatchesAlloc) {
        ReportMismatchedDealloc(C, ArgExpr->getSourceRange(),
                                ParentExpr, RsBase, SymBase);
        return 0;
      }

      // Check if the memory location being freed is the actual location
      // allocated, or an offset.
      RegionOffset Offset = R->getAsOffset();
      if (Offset.isValid() &&
          !Offset.hasSymbolicOffset() &&
          Offset.getOffset() != 0) {
        const Expr *AllocExpr = cast<Expr>(RsBase->getStmt());
        ReportOffsetFree(C, ArgVal, ArgExpr->getSourceRange(), ParentExpr, 
                         AllocExpr);
        return 0;
      }
    }
  }

  ReleasedAllocated = (RsBase != 0);

  // Clean out the info on previous call to free return info.
  State = State->remove<FreeReturnValue>(SymBase);

  // Keep track of the return value. If it is NULL, we will know that free 
  // failed.
  if (ReturnsNullOnFailure) {
    SVal RetVal = C.getSVal(ParentExpr);
    SymbolRef RetStatusSymbol = RetVal.getAsSymbol();
    if (RetStatusSymbol) {
      C.getSymbolManager().addSymbolDependency(SymBase, RetStatusSymbol);
      State = State->set<FreeReturnValue>(SymBase, RetStatusSymbol);
    }
  }

  AllocationFamily Family = RsBase ? RsBase->getAllocationFamily()
                                   : getAllocationFamily(C, ParentExpr);
  // Normal free.
  if (Hold)
    return State->set<RegionState>(SymBase,
                                   RefState::getRelinquished(Family,
                                                             ParentExpr));

  return State->set<RegionState>(SymBase,
                                 RefState::getReleased(Family, ParentExpr));
}

bool MallocChecker::isTrackedByCurrentChecker(AllocationFamily Family) const {
  switch (Family) {
  case AF_Malloc: {
    if (!Filter.CMallocOptimistic && !Filter.CMallocPessimistic)
      return false;
    return true;
  }
  case AF_CXXNew:
  case AF_CXXNewArray: {
    if (!Filter.CNewDeleteChecker)
      return false;
    return true;
  }
  case AF_None: {
    llvm_unreachable("no family");
  }
  }
  llvm_unreachable("unhandled family");
}

bool
MallocChecker::isTrackedByCurrentChecker(CheckerContext &C, 
                                         const Stmt *AllocDeallocStmt) const {
  return isTrackedByCurrentChecker(getAllocationFamily(C, AllocDeallocStmt));
}

bool MallocChecker::isTrackedByCurrentChecker(CheckerContext &C,
                                              SymbolRef Sym) const {

  const RefState *RS = C.getState()->get<RegionState>(Sym);
  assert(RS);
  return isTrackedByCurrentChecker(RS->getAllocationFamily());
}

bool MallocChecker::SummarizeValue(raw_ostream &os, SVal V) {
  if (Optional<nonloc::ConcreteInt> IntVal = V.getAs<nonloc::ConcreteInt>())
    os << "an integer (" << IntVal->getValue() << ")";
  else if (Optional<loc::ConcreteInt> ConstAddr = V.getAs<loc::ConcreteInt>())
    os << "a constant address (" << ConstAddr->getValue() << ")";
  else if (Optional<loc::GotoLabel> Label = V.getAs<loc::GotoLabel>())
    os << "the address of the label '" << Label->getLabel()->getName() << "'";
  else
    return false;
  
  return true;
}

bool MallocChecker::SummarizeRegion(raw_ostream &os,
                                    const MemRegion *MR) {
  switch (MR->getKind()) {
  case MemRegion::FunctionTextRegionKind: {
    const NamedDecl *FD = cast<FunctionTextRegion>(MR)->getDecl();
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
                                  SourceRange Range, 
                                  const Expr *DeallocExpr) const {

  if (!Filter.CMallocOptimistic && !Filter.CMallocPessimistic && 
      !Filter.CNewDeleteChecker)
    return;

  if (!isTrackedByCurrentChecker(C, DeallocExpr))
    return;

  if (ExplodedNode *N = C.generateSink()) {
    if (!BT_BadFree)
      BT_BadFree.reset(new BugType("Bad free", "Memory Error"));
    
    SmallString<100> buf;
    llvm::raw_svector_ostream os(buf);

    const MemRegion *MR = ArgVal.getAsRegion();
    while (const ElementRegion *ER = dyn_cast_or_null<ElementRegion>(MR))
      MR = ER->getSuperRegion();

    if (MR && isa<AllocaRegion>(MR))
      os << "Memory allocated by alloca() should not be deallocated";
    else {
      os << "Argument to ";
      if (!printAllocDeallocName(os, C, DeallocExpr))
        os << "deallocator";

      os << " is ";
      bool Summarized = MR ? SummarizeRegion(os, MR) 
                           : SummarizeValue(os, ArgVal);
      if (Summarized)
        os << ", which is not memory allocated by ";
      else
        os << "not memory allocated by ";

      printExpectedAllocName(os, C, DeallocExpr);
    }

    BugReport *R = new BugReport(*BT_BadFree, os.str(), N);
    R->markInteresting(MR);
    R->addRange(Range);
    C.emitReport(R);
  }
}

void MallocChecker::ReportMismatchedDealloc(CheckerContext &C, 
                                            SourceRange Range,
                                            const Expr *DeallocExpr, 
                                            const RefState *RS,
                                            SymbolRef Sym) const {

  if (!Filter.CMismatchedDeallocatorChecker)
    return;

  if (ExplodedNode *N = C.generateSink()) {
    if (!BT_MismatchedDealloc)
      BT_MismatchedDealloc.reset(new BugType("Bad deallocator",
                                             "Memory Error"));
    
    SmallString<100> buf;
    llvm::raw_svector_ostream os(buf);

    const Expr *AllocExpr = cast<Expr>(RS->getStmt());
    SmallString<20> AllocBuf;
    llvm::raw_svector_ostream AllocOs(AllocBuf);
    SmallString<20> DeallocBuf;
    llvm::raw_svector_ostream DeallocOs(DeallocBuf);

    os << "Memory";
    if (printAllocDeallocName(AllocOs, C, AllocExpr))
      os << " allocated by " << AllocOs.str();

    os << " should be deallocated by ";
      printExpectedDeallocName(os, RS->getAllocationFamily());

    if (printAllocDeallocName(DeallocOs, C, DeallocExpr))
      os << ", not " << DeallocOs.str();

    BugReport *R = new BugReport(*BT_MismatchedDealloc, os.str(), N);
    R->markInteresting(Sym);
    R->addRange(Range);
    R->addVisitor(new MallocBugVisitor(Sym));
    C.emitReport(R);
  }
}

void MallocChecker::ReportOffsetFree(CheckerContext &C, SVal ArgVal,
                                     SourceRange Range, const Expr *DeallocExpr,
                                     const Expr *AllocExpr) const {

  if (!Filter.CMallocOptimistic && !Filter.CMallocPessimistic && 
      !Filter.CNewDeleteChecker)
    return;

  if (!isTrackedByCurrentChecker(C, AllocExpr))
    return;

  ExplodedNode *N = C.generateSink();
  if (N == NULL)
    return;

  if (!BT_OffsetFree)
    BT_OffsetFree.reset(new BugType("Offset free", "Memory Error"));

  SmallString<100> buf;
  llvm::raw_svector_ostream os(buf);
  SmallString<20> AllocNameBuf;
  llvm::raw_svector_ostream AllocNameOs(AllocNameBuf);

  const MemRegion *MR = ArgVal.getAsRegion();
  assert(MR && "Only MemRegion based symbols can have offset free errors");

  RegionOffset Offset = MR->getAsOffset();
  assert((Offset.isValid() &&
          !Offset.hasSymbolicOffset() &&
          Offset.getOffset() != 0) &&
         "Only symbols with a valid offset can have offset free errors");

  int offsetBytes = Offset.getOffset() / C.getASTContext().getCharWidth();

  os << "Argument to ";
  if (!printAllocDeallocName(os, C, DeallocExpr))
    os << "deallocator";
  os << " is offset by "
     << offsetBytes
     << " "
     << ((abs(offsetBytes) > 1) ? "bytes" : "byte")
     << " from the start of ";
  if (AllocExpr && printAllocDeallocName(AllocNameOs, C, AllocExpr))
    os << "memory allocated by " << AllocNameOs.str();
  else
    os << "allocated memory";

  BugReport *R = new BugReport(*BT_OffsetFree, os.str(), N);
  R->markInteresting(MR->getBaseRegion());
  R->addRange(Range);
  C.emitReport(R);
}

void MallocChecker::ReportUseAfterFree(CheckerContext &C, SourceRange Range,
                                       SymbolRef Sym) const {

  if (!Filter.CMallocOptimistic && !Filter.CMallocPessimistic && 
      !Filter.CNewDeleteChecker)
    return;

  if (!isTrackedByCurrentChecker(C, Sym))
    return;

  if (ExplodedNode *N = C.generateSink()) {
    if (!BT_UseFree)
      BT_UseFree.reset(new BugType("Use-after-free", "Memory Error"));

    BugReport *R = new BugReport(*BT_UseFree,
                                 "Use of memory after it is freed", N);

    R->markInteresting(Sym);
    R->addRange(Range);
    R->addVisitor(new MallocBugVisitor(Sym));
    C.emitReport(R);
  }
}

void MallocChecker::ReportDoubleFree(CheckerContext &C, SourceRange Range,
                                     bool Released, SymbolRef Sym, 
                                     SymbolRef PrevSym) const {

  if (!Filter.CMallocOptimistic && !Filter.CMallocPessimistic && 
      !Filter.CNewDeleteChecker)
    return;

  if (!isTrackedByCurrentChecker(C, Sym))
    return;

  if (ExplodedNode *N = C.generateSink()) {
    if (!BT_DoubleFree)
      BT_DoubleFree.reset(new BugType("Double free", "Memory Error"));

    BugReport *R = new BugReport(*BT_DoubleFree,
      (Released ? "Attempt to free released memory"
                : "Attempt to free non-owned memory"),
      N);
    R->addRange(Range);
    R->markInteresting(Sym);
    if (PrevSym)
      R->markInteresting(PrevSym);
    R->addVisitor(new MallocBugVisitor(Sym));
    C.emitReport(R);
  }
}

ProgramStateRef MallocChecker::ReallocMem(CheckerContext &C,
                                          const CallExpr *CE,
                                          bool FreesOnFail) const {
  if (CE->getNumArgs() < 2)
    return 0;

  ProgramStateRef state = C.getState();
  const Expr *arg0Expr = CE->getArg(0);
  const LocationContext *LCtx = C.getLocationContext();
  SVal Arg0Val = state->getSVal(arg0Expr, LCtx);
  if (!Arg0Val.getAs<DefinedOrUnknownSVal>())
    return 0;
  DefinedOrUnknownSVal arg0Val = Arg0Val.castAs<DefinedOrUnknownSVal>();

  SValBuilder &svalBuilder = C.getSValBuilder();

  DefinedOrUnknownSVal PtrEQ =
    svalBuilder.evalEQ(state, arg0Val, svalBuilder.makeNull());

  // Get the size argument. If there is no size arg then give up.
  const Expr *Arg1 = CE->getArg(1);
  if (!Arg1)
    return 0;

  // Get the value of the size argument.
  SVal Arg1ValG = state->getSVal(Arg1, LCtx);
  if (!Arg1ValG.getAs<DefinedOrUnknownSVal>())
    return 0;
  DefinedOrUnknownSVal Arg1Val = Arg1ValG.castAs<DefinedOrUnknownSVal>();

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

  bool ReleasedAllocated = false;

  // If the size is 0, free the memory.
  if (SizeIsZero)
    if (ProgramStateRef stateFree = FreeMemAux(C, CE, StateSizeIsZero, 0,
                                               false, ReleasedAllocated)){
      // The semantics of the return value are:
      // If size was equal to 0, either NULL or a pointer suitable to be passed
      // to free() is returned. We just free the input pointer and do not add
      // any constrains on the output pointer.
      return stateFree;
    }

  // Default behavior.
  if (ProgramStateRef stateFree =
        FreeMemAux(C, CE, state, 0, false, ReleasedAllocated)) {

    ProgramStateRef stateRealloc = MallocMemAux(C, CE, CE->getArg(1),
                                                UnknownVal(), stateFree);
    if (!stateRealloc)
      return 0;

    ReallocPairKind Kind = RPToBeFreedAfterFailure;
    if (FreesOnFail)
      Kind = RPIsFreeOnFailure;
    else if (!ReleasedAllocated)
      Kind = RPDoNotTrackAfterFailure;

    // Record the info about the reallocated symbol so that we could properly
    // process failed reallocation.
    stateRealloc = stateRealloc->set<ReallocPairs>(ToPtr,
                                                   ReallocPair(FromPtr, Kind));
    // The reallocated symbol should stay alive for as long as the new symbol.
    C.getSymbolManager().addSymbolDependency(ToPtr, FromPtr);
    return stateRealloc;
  }
  return 0;
}

ProgramStateRef MallocChecker::CallocMem(CheckerContext &C, const CallExpr *CE){
  if (CE->getNumArgs() < 2)
    return 0;

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
          if (Val.getAsLocSymbol() == Sym) {
            const VarRegion* VR = MR->getBaseRegion()->getAs<VarRegion>();
            // Do not show local variables belonging to a function other than
            // where the error is reported.
            if (!VR ||
                (VR->getStackFrame() == LeakContext->getCurrentStackFrame()))
              ReferenceRegion = MR;
          }
        }
      }

    // Allocation node, is the last node in the current context in which the
    // symbol was tracked.
    if (N->getLocationContext() == LeakContext)
      AllocNode = N;
    N = N->pred_empty() ? NULL : *(N->pred_begin());
  }

  return LeakInfo(AllocNode, ReferenceRegion);
}

void MallocChecker::reportLeak(SymbolRef Sym, ExplodedNode *N,
                               CheckerContext &C) const {

  if (!Filter.CMallocOptimistic && !Filter.CMallocPessimistic && 
      !Filter.CNewDeleteLeaksChecker)
    return;

  const RefState *RS = C.getState()->get<RegionState>(Sym);
  assert(RS && "cannot leak an untracked symbol");
  AllocationFamily Family = RS->getAllocationFamily();
  if (!isTrackedByCurrentChecker(Family))
    return;

  // Special case for new and new[]; these are controlled by a separate checker
  // flag so that they can be selectively disabled.
  if (Family == AF_CXXNew || Family == AF_CXXNewArray)
    if (!Filter.CNewDeleteLeaksChecker)
      return;

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
  const ExplodedNode *AllocNode = 0;
  const MemRegion *Region = 0;
  llvm::tie(AllocNode, Region) = getAllocationSite(N, Sym, C);
  
  ProgramPoint P = AllocNode->getLocation();
  const Stmt *AllocationStmt = 0;
  if (Optional<CallExitEnd> Exit = P.getAs<CallExitEnd>())
    AllocationStmt = Exit->getCalleeContext()->getCallSite();
  else if (Optional<StmtPoint> SP = P.getAs<StmtPoint>())
    AllocationStmt = SP->getStmt();
  if (AllocationStmt)
    LocUsedForUniqueing = PathDiagnosticLocation::createBegin(AllocationStmt,
                                              C.getSourceManager(),
                                              AllocNode->getLocationContext());

  SmallString<200> buf;
  llvm::raw_svector_ostream os(buf);
  if (Region && Region->canPrintPretty()) {
    os << "Potential leak of memory pointed to by ";
    Region->printPretty(os);
  } else {
    os << "Potential memory leak";
  }

  BugReport *R = new BugReport(*BT_Leak, os.str(), N, 
                               LocUsedForUniqueing, 
                               AllocNode->getLocationContext()->getDecl());
  R->markInteresting(Sym);
  R->addVisitor(new MallocBugVisitor(Sym, true));
  C.emitReport(R);
}

void MallocChecker::checkDeadSymbols(SymbolReaper &SymReaper,
                                     CheckerContext &C) const
{
  if (!SymReaper.hasDeadSymbols())
    return;

  ProgramStateRef state = C.getState();
  RegionStateTy RS = state->get<RegionState>();
  RegionStateTy::Factory &F = state->get_context<RegionState>();

  SmallVector<SymbolRef, 2> Errors;
  for (RegionStateTy::iterator I = RS.begin(), E = RS.end(); I != E; ++I) {
    if (SymReaper.isDead(I->first)) {
      if (I->second.isAllocated())
        Errors.push_back(I->first);
      // Remove the dead symbol from the map.
      RS = F.remove(RS, I->first);

    }
  }
  
  // Cleanup the Realloc Pairs Map.
  ReallocPairsTy RP = state->get<ReallocPairs>();
  for (ReallocPairsTy::iterator I = RP.begin(), E = RP.end(); I != E; ++I) {
    if (SymReaper.isDead(I->first) ||
        SymReaper.isDead(I->second.ReallocatedSym)) {
      state = state->remove<ReallocPairs>(I->first);
    }
  }

  // Cleanup the FreeReturnValue Map.
  FreeReturnValueTy FR = state->get<FreeReturnValue>();
  for (FreeReturnValueTy::iterator I = FR.begin(), E = FR.end(); I != E; ++I) {
    if (SymReaper.isDead(I->first) ||
        SymReaper.isDead(I->second)) {
      state = state->remove<FreeReturnValue>(I->first);
    }
  }

  // Generate leak node.
  ExplodedNode *N = C.getPredecessor();
  if (!Errors.empty()) {
    static SimpleProgramPointTag Tag("MallocChecker : DeadSymbolsLeak");
    N = C.addTransition(C.getState(), C.getPredecessor(), &Tag);
    for (SmallVector<SymbolRef, 2>::iterator
        I = Errors.begin(), E = Errors.end(); I != E; ++I) {
      reportLeak(*I, N, C);
    }
  }

  C.addTransition(state->set<RegionState>(RS), N);
}

void MallocChecker::checkPreCall(const CallEvent &Call,
                                 CheckerContext &C) const {

  // We will check for double free in the post visit.
  if (const AnyFunctionCall *FC = dyn_cast<AnyFunctionCall>(&Call)) {
    const FunctionDecl *FD = FC->getDecl();
    if (!FD)
      return;

    if ((Filter.CMallocOptimistic || Filter.CMallocPessimistic) &&
        isFreeFunction(FD, C.getASTContext()))
      return;

    if (Filter.CNewDeleteChecker &&
        isStandardNewDelete(FD, C.getASTContext()))
      return;
  }

  // Check if the callee of a method is deleted.
  if (const CXXInstanceCall *CC = dyn_cast<CXXInstanceCall>(&Call)) {
    SymbolRef Sym = CC->getCXXThisVal().getAsSymbol();
    if (!Sym || checkUseAfterFree(Sym, C, CC->getCXXThisExpr()))
      return;
  }

  // Check arguments for being used after free.
  for (unsigned I = 0, E = Call.getNumArgs(); I != E; ++I) {
    SVal ArgSVal = Call.getArgSVal(I);
    if (ArgSVal.getAs<Loc>()) {
      SymbolRef Sym = ArgSVal.getAsSymbol();
      if (!Sym)
        continue;
      if (checkUseAfterFree(Sym, C, Call.getArgExpr(I)))
        return;
    }
  }
}

void MallocChecker::checkPreStmt(const ReturnStmt *S, CheckerContext &C) const {
  const Expr *E = S->getRetValue();
  if (!E)
    return;

  // Check if we are returning a symbol.
  ProgramStateRef State = C.getState();
  SVal RetVal = State->getSVal(E, C.getLocationContext());
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

  // Check if we are returning freed memory.
  if (Sym)
    checkUseAfterFree(Sym, C, E);
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
    const VarRegion *VR = I.getCapturedRegion();
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

bool MallocChecker::isReleased(SymbolRef Sym, CheckerContext &C) const {
  assert(Sym);
  const RefState *RS = C.getState()->get<RegionState>(Sym);
  return (RS && RS->isReleased());
}

bool MallocChecker::checkUseAfterFree(SymbolRef Sym, CheckerContext &C,
                                      const Stmt *S) const {

  if (isReleased(Sym, C)) {
    ReportUseAfterFree(C, S->getSourceRange(), Sym);
    return true;
  }

  return false;
}

// Check if the location is a freed symbolic region.
void MallocChecker::checkLocation(SVal l, bool isLoad, const Stmt *S,
                                  CheckerContext &C) const {
  SymbolRef Sym = l.getLocSymbolInBase();
  if (Sym)
    checkUseAfterFree(Sym, C, S);
}

// If a symbolic region is assumed to NULL (or another constant), stop tracking
// it - assuming that allocation failed on this path.
ProgramStateRef MallocChecker::evalAssume(ProgramStateRef state,
                                              SVal Cond,
                                              bool Assumption) const {
  RegionStateTy RS = state->get<RegionState>();
  for (RegionStateTy::iterator I = RS.begin(), E = RS.end(); I != E; ++I) {
    // If the symbol is assumed to be NULL, remove it from consideration.
    ConstraintManager &CMgr = state->getConstraintManager();
    ConditionTruthVal AllocFailed = CMgr.isNull(state, I.getKey());
    if (AllocFailed.isConstrainedTrue())
      state = state->remove<RegionState>(I.getKey());
  }

  // Realloc returns 0 when reallocation fails, which means that we should
  // restore the state of the pointer being reallocated.
  ReallocPairsTy RP = state->get<ReallocPairs>();
  for (ReallocPairsTy::iterator I = RP.begin(), E = RP.end(); I != E; ++I) {
    // If the symbol is assumed to be NULL, remove it from consideration.
    ConstraintManager &CMgr = state->getConstraintManager();
    ConditionTruthVal AllocFailed = CMgr.isNull(state, I.getKey());
    if (!AllocFailed.isConstrainedTrue())
      continue;

    SymbolRef ReallocSym = I.getData().ReallocatedSym;
    if (const RefState *RS = state->get<RegionState>(ReallocSym)) {
      if (RS->isReleased()) {
        if (I.getData().Kind == RPToBeFreedAfterFailure)
          state = state->set<RegionState>(ReallocSym,
              RefState::getAllocated(RS->getAllocationFamily(), RS->getStmt()));
        else if (I.getData().Kind == RPDoNotTrackAfterFailure)
          state = state->remove<RegionState>(ReallocSym);
        else
          assert(I.getData().Kind == RPIsFreeOnFailure);
      }
    }
    state = state->remove<ReallocPairs>(I.getKey());
  }

  return state;
}

bool MallocChecker::doesNotFreeMemOrInteresting(const CallEvent *Call,
                                                ProgramStateRef State) const {
  assert(Call);

  // For now, assume that any C++ call can free memory.
  // TODO: If we want to be more optimistic here, we'll need to make sure that
  // regions escape to C++ containers. They seem to do that even now, but for
  // mysterious reasons.
  if (!(isa<FunctionCall>(Call) || isa<ObjCMethodCall>(Call)))
    return false;

  // Check Objective-C messages by selector name.
  if (const ObjCMethodCall *Msg = dyn_cast<ObjCMethodCall>(Call)) {
    // If it's not a framework call, or if it takes a callback, assume it
    // can free memory.
    if (!Call->isInSystemHeader() || Call->hasNonZeroCallbackArg())
      return false;

    // If it's a method we know about, handle it explicitly post-call.
    // This should happen before the "freeWhenDone" check below.
    if (isKnownDeallocObjCMethodName(*Msg))
      return true;

    // If there's a "freeWhenDone" parameter, but the method isn't one we know
    // about, we can't be sure that the object will use free() to deallocate the
    // memory, so we can't model it explicitly. The best we can do is use it to
    // decide whether the pointer escapes.
    if (Optional<bool> FreeWhenDone = getFreeWhenDoneArg(*Msg))
      return !*FreeWhenDone;

    // If the first selector piece ends with "NoCopy", and there is no
    // "freeWhenDone" parameter set to zero, we know ownership is being
    // transferred. Again, though, we can't be sure that the object will use
    // free() to deallocate the memory, so we can't model it explicitly.
    StringRef FirstSlot = Msg->getSelector().getNameForSlot(0);
    if (FirstSlot.endswith("NoCopy"))
      return false;

    // If the first selector starts with addPointer, insertPointer,
    // or replacePointer, assume we are dealing with NSPointerArray or similar.
    // This is similar to C++ containers (vector); we still might want to check
    // that the pointers get freed by following the container itself.
    if (FirstSlot.startswith("addPointer") ||
        FirstSlot.startswith("insertPointer") ||
        FirstSlot.startswith("replacePointer")) {
      return false;
    }

    // Otherwise, assume that the method does not free memory.
    // Most framework methods do not free memory.
    return true;
  }

  // At this point the only thing left to handle is straight function calls.
  const FunctionDecl *FD = cast<FunctionCall>(Call)->getDecl();
  if (!FD)
    return false;

  ASTContext &ASTC = State->getStateManager().getContext();

  // If it's one of the allocation functions we can reason about, we model
  // its behavior explicitly.
  if (isMemFunction(FD, ASTC))
    return true;

  // If it's not a system call, assume it frees memory.
  if (!Call->isInSystemHeader())
    return false;

  // White list the system functions whose arguments escape.
  const IdentifierInfo *II = FD->getIdentifier();
  if (!II)
    return false;
  StringRef FName = II->getName();

  // White list the 'XXXNoCopy' CoreFoundation functions.
  // We specifically check these before 
  if (FName.endswith("NoCopy")) {
    // Look for the deallocator argument. We know that the memory ownership
    // is not transferred only if the deallocator argument is
    // 'kCFAllocatorNull'.
    for (unsigned i = 1; i < Call->getNumArgs(); ++i) {
      const Expr *ArgE = Call->getArgExpr(i)->IgnoreParenCasts();
      if (const DeclRefExpr *DE = dyn_cast<DeclRefExpr>(ArgE)) {
        StringRef DeallocatorName = DE->getFoundDecl()->getName();
        if (DeallocatorName == "kCFAllocatorNull")
          return true;
      }
    }
    return false;
  }

  // Associating streams with malloced buffers. The pointer can escape if
  // 'closefn' is specified (and if that function does free memory),
  // but it will not if closefn is not specified.
  // Currently, we do not inspect the 'closefn' function (PR12101).
  if (FName == "funopen")
    if (Call->getNumArgs() >= 4 && Call->getArgSVal(4).isConstant(0))
      return true;

  // Do not warn on pointers passed to 'setbuf' when used with std streams,
  // these leaks might be intentional when setting the buffer for stdio.
  // http://stackoverflow.com/questions/2671151/who-frees-setvbuf-buffer
  if (FName == "setbuf" || FName =="setbuffer" ||
      FName == "setlinebuf" || FName == "setvbuf") {
    if (Call->getNumArgs() >= 1) {
      const Expr *ArgE = Call->getArgExpr(0)->IgnoreParenCasts();
      if (const DeclRefExpr *ArgDRE = dyn_cast<DeclRefExpr>(ArgE))
        if (const VarDecl *D = dyn_cast<VarDecl>(ArgDRE->getDecl()))
          if (D->getCanonicalDecl()->getName().find("std") != StringRef::npos)
            return false;
    }
  }

  // A bunch of other functions which either take ownership of a pointer or
  // wrap the result up in a struct or object, meaning it can be freed later.
  // (See RetainCountChecker.) Not all the parameters here are invalidated,
  // but the Malloc checker cannot differentiate between them. The right way
  // of doing this would be to implement a pointer escapes callback.
  if (FName == "CGBitmapContextCreate" ||
      FName == "CGBitmapContextCreateWithData" ||
      FName == "CVPixelBufferCreateWithBytes" ||
      FName == "CVPixelBufferCreateWithPlanarBytes" ||
      FName == "OSAtomicEnqueue") {
    return false;
  }

  // Handle cases where we know a buffer's /address/ can escape.
  // Note that the above checks handle some special cases where we know that
  // even though the address escapes, it's still our responsibility to free the
  // buffer.
  if (Call->argumentsMayEscape())
    return false;

  // Otherwise, assume that the function does not free memory.
  // Most system calls do not free the memory.
  return true;
}

static bool retTrue(const RefState *RS) {
  return true;
}

static bool checkIfNewOrNewArrayFamily(const RefState *RS) {
  return (RS->getAllocationFamily() == AF_CXXNewArray ||
          RS->getAllocationFamily() == AF_CXXNew);
}

ProgramStateRef MallocChecker::checkPointerEscape(ProgramStateRef State,
                                             const InvalidatedSymbols &Escaped,
                                             const CallEvent *Call,
                                             PointerEscapeKind Kind) const {
  return checkPointerEscapeAux(State, Escaped, Call, Kind, &retTrue);
}

ProgramStateRef MallocChecker::checkConstPointerEscape(ProgramStateRef State,
                                              const InvalidatedSymbols &Escaped,
                                              const CallEvent *Call,
                                              PointerEscapeKind Kind) const {
  return checkPointerEscapeAux(State, Escaped, Call, Kind,
                               &checkIfNewOrNewArrayFamily);
}

ProgramStateRef MallocChecker::checkPointerEscapeAux(ProgramStateRef State,
                                              const InvalidatedSymbols &Escaped,
                                              const CallEvent *Call,
                                              PointerEscapeKind Kind,
                                  bool(*CheckRefState)(const RefState*)) const {
  // If we know that the call does not free memory, or we want to process the
  // call later, keep tracking the top level arguments.
  if ((Kind == PSK_DirectEscapeOnCall ||
       Kind == PSK_IndirectEscapeOnCall) &&
      doesNotFreeMemOrInteresting(Call, State)) {
    return State;
  }

  for (InvalidatedSymbols::const_iterator I = Escaped.begin(),
       E = Escaped.end();
       I != E; ++I) {
    SymbolRef sym = *I;

    if (const RefState *RS = State->get<RegionState>(sym)) {
      if (RS->isAllocated() && CheckRefState(RS)) {
        State = State->remove<RegionState>(sym);
        State = State->set<RegionState>(sym, RefState::getEscaped(RS));
      }
    }
  }
  return State;
}

static SymbolRef findFailedReallocSymbol(ProgramStateRef currState,
                                         ProgramStateRef prevState) {
  ReallocPairsTy currMap = currState->get<ReallocPairs>();
  ReallocPairsTy prevMap = prevState->get<ReallocPairs>();

  for (ReallocPairsTy::iterator I = prevMap.begin(), E = prevMap.end();
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
  if (!RS)
    return 0;

  const Stmt *S = 0;
  const char *Msg = 0;
  StackHintGeneratorForSymbol *StackHint = 0;

  // Retrieve the associated statement.
  ProgramPoint ProgLoc = N->getLocation();
  if (Optional<StmtPoint> SP = ProgLoc.getAs<StmtPoint>()) {
    S = SP->getStmt();
  } else if (Optional<CallExitEnd> Exit = ProgLoc.getAs<CallExitEnd>()) {
    S = Exit->getCalleeContext()->getCallSite();
  } else if (Optional<BlockEdge> Edge = ProgLoc.getAs<BlockEdge>()) {
    // If an assumption was made on a branch, it should be caught
    // here by looking at the state transition.
    S = Edge->getSrc()->getTerminator();
  }

  if (!S)
    return 0;

  // FIXME: We will eventually need to handle non-statement-based events
  // (__attribute__((cleanup))).

  // Find out if this is an interesting point and what is the kind.
  if (Mode == Normal) {
    if (isAllocated(RS, RSPrev, S)) {
      Msg = "Memory is allocated";
      StackHint = new StackHintGeneratorForSymbol(Sym,
                                                  "Returned allocated memory");
    } else if (isReleased(RS, RSPrev, S)) {
      Msg = "Memory is released";
      StackHint = new StackHintGeneratorForSymbol(Sym,
                                             "Returning; memory was released");
    } else if (isRelinquished(RS, RSPrev, S)) {
      Msg = "Memory ownership is transfered";
      StackHint = new StackHintGeneratorForSymbol(Sym, "");
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

void MallocChecker::printState(raw_ostream &Out, ProgramStateRef State,
                               const char *NL, const char *Sep) const {

  RegionStateTy RS = State->get<RegionState>();

  if (!RS.isEmpty()) {
    Out << Sep << "MallocChecker:" << NL;
    for (RegionStateTy::iterator I = RS.begin(), E = RS.end(); I != E; ++I) {
      I.getKey()->dumpToStream(Out);
      Out << " : ";
      I.getData().dump(Out);
      Out << NL;
    }
  }
}

void ento::registerNewDeleteLeaksChecker(CheckerManager &mgr) {
  registerCStringCheckerBasic(mgr);
  mgr.registerChecker<MallocChecker>()->Filter.CNewDeleteLeaksChecker = true;
  // We currently treat NewDeleteLeaks checker as a subchecker of NewDelete 
  // checker.
  mgr.registerChecker<MallocChecker>()->Filter.CNewDeleteChecker = true;
}

#define REGISTER_CHECKER(name) \
void ento::register##name(CheckerManager &mgr) {\
  registerCStringCheckerBasic(mgr); \
  mgr.registerChecker<MallocChecker>()->Filter.C##name = true;\
}

REGISTER_CHECKER(MallocPessimistic)
REGISTER_CHECKER(MallocOptimistic)
REGISTER_CHECKER(NewDeleteChecker)
REGISTER_CHECKER(MismatchedDeallocatorChecker)
