//== GenericTaintChecker.cpp ----------------------------------- -*- C++ -*--=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This checker defines the attack surface for generic taint propagation.
//
// The taint information produced by it might be useful to other checkers. For
// example, checkers should report errors which involve tainted data more
// aggressively, even if the involved symbols are under constrained.
//
//===----------------------------------------------------------------------===//
#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"

using namespace clang;
using namespace ento;

namespace {
class GenericTaintChecker : public Checker< check::PostStmt<CallExpr>,
                                            check::PreStmt<CallExpr> > {
public:
  enum TaintOnPreVisitKind {
    /// No taint propagates from pre-visit to post-visit.
    PrevisitNone = 0,
    /// Based on the pre-visit, the return argument of the call
    /// should be tainted.
    PrevisitTaintRet = 1,
    /// Based on the pre-visit, the call can taint values through it's
    /// pointer/reference arguments.
    PrevisitTaintArgs = 2
  };

private:
  mutable llvm::OwningPtr<BugType> BT;
  void initBugType() const;

  /// Given a pointer argument, get the symbol of the value it contains
  /// (points to).
  SymbolRef getPointedToSymbol(CheckerContext &C,
                               const Expr *Arg,
                               bool IssueWarning = true) const;

  /// Functions defining the attack surface.
  typedef const ProgramState *(GenericTaintChecker::*FnCheck)(const CallExpr *,
                                                       CheckerContext &C) const;
  const ProgramState *postScanf(const CallExpr *CE, CheckerContext &C) const;
  const ProgramState *postFscanf(const CallExpr *CE, CheckerContext &C) const;
  const ProgramState *postRetTaint(const CallExpr *CE, CheckerContext &C) const;
  const ProgramState *postDefault(const CallExpr *CE, CheckerContext &C) const;

  /// Taint the scanned input if the file is tainted.
  const ProgramState *preFscanf(const CallExpr *CE, CheckerContext &C) const;
  /// Taint if any of the arguments are tainted.
  const ProgramState *preAnyArgs(const CallExpr *CE, CheckerContext &C) const;

  /// Check if the region the expression evaluates to is the standard input,
  /// and thus, is tainted.
  bool isStdin(const Expr *E, CheckerContext &C) const;

public:
  static void *getTag() { static int Tag; return &Tag; }

  void checkPostStmt(const CallExpr *CE, CheckerContext &C) const;
  void checkPostStmt(const DeclRefExpr *DRE, CheckerContext &C) const;

  void checkPreStmt(const CallExpr *CE, CheckerContext &C) const;

};
}

/// Definitions for the checker specific state.
namespace { struct TaintOnPreVisit {};}
namespace clang {
namespace ento {
  /// A flag which is used to pass information from call pre-visit instruction
  /// to the call post-visit. The value is an unsigned, which takes on values
  /// of the TaintOnPreVisitKind enumeration.
  template<>
  struct ProgramStateTrait<TaintOnPreVisit> :
    public ProgramStatePartialTrait<unsigned> {
    static void *GDMIndex() { return GenericTaintChecker::getTag(); }
  };
}
}

inline void GenericTaintChecker::initBugType() const {
  if (!BT)
    BT.reset(new BugType("Tainted data checking", "General"));
}

void GenericTaintChecker::checkPreStmt(const CallExpr *CE,
                                       CheckerContext &C) const {
  const ProgramState *State = C.getState();

  // Set the evaluation function by switching on the callee name.
  StringRef Name = C.getCalleeName(CE);
  FnCheck evalFunction = llvm::StringSwitch<FnCheck>(Name)
    .Case("fscanf", &GenericTaintChecker::preFscanf)
    .Case("atoi", &GenericTaintChecker::preAnyArgs)
    .Case("atol", &GenericTaintChecker::preAnyArgs)
    .Case("atoll", &GenericTaintChecker::preAnyArgs)
    .Default(0);

  // Check and evaluate the call.
  if (evalFunction)
    State = (this->*evalFunction)(CE, C);
  if (!State)
    return;

  C.addTransition(State);
}

void GenericTaintChecker::checkPostStmt(const CallExpr *CE,
                                        CheckerContext &C) const {
  const ProgramState *State = C.getState();
  
  // Define the attack surface.
  // Set the evaluation function by switching on the callee name.
  StringRef Name = C.getCalleeName(CE);
  FnCheck evalFunction = llvm::StringSwitch<FnCheck>(Name)
    .Case("scanf", &GenericTaintChecker::postScanf)
    .Case("fscanf", &GenericTaintChecker::postFscanf)
    .Case("sscanf", &GenericTaintChecker::postFscanf)
    // TODO: Add support for vfscanf & family.
    .Case("getchar", &GenericTaintChecker::postRetTaint)
    .Case("getenv", &GenericTaintChecker::postRetTaint)
    .Case("fopen", &GenericTaintChecker::postRetTaint)
    .Case("fdopen", &GenericTaintChecker::postRetTaint)
    .Case("freopen", &GenericTaintChecker::postRetTaint)
    .Default(&GenericTaintChecker::postDefault);

  // If the callee isn't defined, it is not of security concern.
  // Check and evaluate the call.
  if (evalFunction)
    State = (this->*evalFunction)(CE, C);
  if (!State)
    return;

  assert(State->get<TaintOnPreVisit>() == PrevisitNone &&
         "State has to be cleared.");
  C.addTransition(State);
}

SymbolRef GenericTaintChecker::getPointedToSymbol(CheckerContext &C,
                                                  const Expr* Arg,
                                                  bool IssueWarning) const {
  const ProgramState *State = C.getState();
  SVal AddrVal = State->getSVal(Arg->IgnoreParens());
  if (AddrVal.isUnknownOrUndef())
    return 0;

  Loc *AddrLoc = dyn_cast<Loc>(&AddrVal);

  if (!AddrLoc && !IssueWarning)
    return 0;

  // If the Expr is not a location, issue a warning.
  if (!AddrLoc) {
    assert(IssueWarning);
    if (ExplodedNode *N = C.generateSink(State)) {
      initBugType();
      BugReport *report = new BugReport(*BT, "Pointer argument is expected.",N);
      report->addRange(Arg->getSourceRange());
      C.EmitReport(report);
    }
    return 0;
  }

  SVal Val = State->getSVal(*AddrLoc);
  return Val.getAsSymbol();
}

const ProgramState *GenericTaintChecker::preFscanf(const CallExpr *CE,
                                                   CheckerContext &C) const {
  assert(CE->getNumArgs() >= 2);
  const ProgramState *State = C.getState();

  // Check is the file descriptor is tainted.
  if (State->isTainted(CE->getArg(0)) || isStdin(CE->getArg(0), C))
    return State->set<TaintOnPreVisit>(PrevisitTaintArgs);
  return 0;
}

// If any other arguments are tainted, mark state as tainted on pre-visit.
const ProgramState * GenericTaintChecker::preAnyArgs(const CallExpr *CE,
                                                     CheckerContext &C) const {
  for (unsigned int i = 0; i < CE->getNumArgs(); ++i) {
    const ProgramState *State = C.getState();
    const Expr *Arg = CE->getArg(i);
    if (State->isTainted(Arg) || State->isTainted(getPointedToSymbol(C, Arg)))
      return State = State->set<TaintOnPreVisit>(PrevisitTaintRet);
  }
  return 0;
}

const ProgramState *GenericTaintChecker::postDefault(const CallExpr *CE,
                                                     CheckerContext &C) const {
  const ProgramState *State = C.getState();

  // Check if we know that the result needs to be tainted based on the
  // pre-visit analysis.
  if (State->get<TaintOnPreVisit>() == PrevisitTaintRet) {
    State = State->addTaint(CE);
    return State->set<TaintOnPreVisit>(PrevisitNone);
  }

  return 0;
}

const ProgramState *GenericTaintChecker::postScanf(const CallExpr *CE,
                                                   CheckerContext &C) const {
  const ProgramState *State = C.getState();
  assert(CE->getNumArgs() >= 2);
  SVal x = State->getSVal(CE->getArg(1));
  // All arguments except for the very first one should get taint.
  for (unsigned int i = 1; i < CE->getNumArgs(); ++i) {
    // The arguments are pointer arguments. The data they are pointing at is
    // tainted after the call.
    const Expr* Arg = CE->getArg(i);
        SymbolRef Sym = getPointedToSymbol(C, Arg);
    if (Sym)
      State = State->addTaint(Sym);
  }
  return State;
}

/// If argument 0 (file descriptor) is tainted, all arguments except for arg 0
/// and arg 1 should get taint.
const ProgramState *GenericTaintChecker::postFscanf(const CallExpr *CE,
                                                    CheckerContext &C) const {
  const ProgramState *State = C.getState();
  assert(CE->getNumArgs() >= 2);

  // Fscanf is only tainted if the input file is tainted at pre visit, so
  // check for that first.
  if (State->get<TaintOnPreVisit>() == PrevisitNone)
    return 0;

  // Reset the taint state.
  State = State->set<TaintOnPreVisit>(PrevisitNone);

  // All arguments except for the first two should get taint.
  for (unsigned int i = 2; i < CE->getNumArgs(); ++i) {
    // The arguments are pointer arguments. The data they are pointing at is
    // tainted after the call.
    const Expr* Arg = CE->getArg(i);
    SymbolRef Sym = getPointedToSymbol(C, Arg);
    if (Sym)
      State = State->addTaint(Sym);
  }
  return State;
}

const ProgramState *GenericTaintChecker::postRetTaint(const CallExpr *CE,
                                                      CheckerContext &C) const {
  return C.getState()->addTaint(CE);
}

bool GenericTaintChecker::isStdin(const Expr *E,
                                  CheckerContext &C) const {
  const ProgramState *State = C.getState();
  SVal Val = State->getSVal(E);

  // stdin is a pointer, so it would be a region.
  const MemRegion *MemReg = Val.getAsRegion();

  // The region should be symbolic, we do not know it's value.
  const SymbolicRegion *SymReg = dyn_cast_or_null<SymbolicRegion>(MemReg);
  if (!SymReg)
    return false;

  // Get it's symbol and find the declaration region it's pointing to.
  const SymbolRegionValue *Sm =dyn_cast<SymbolRegionValue>(SymReg->getSymbol());
  if (!Sm)
    return false;
  const DeclRegion *DeclReg = dyn_cast_or_null<DeclRegion>(Sm->getRegion());
  if (!DeclReg)
    return false;

  // This region corresponds to a declaration, find out if it's a global/extern
  // variable named stdin with the proper type.
  if (const VarDecl *D = dyn_cast_or_null<VarDecl>(DeclReg->getDecl())) {
    D = D->getCanonicalDecl();
    if ((D->getName().find("stdin") != StringRef::npos) && D->isExternC())
        if (const PointerType * PtrTy =
              dyn_cast<PointerType>(D->getType().getTypePtr()))
          if (PtrTy->getPointeeType() == C.getASTContext().getFILEType())
            return true;
  }
  return false;
}

void ento::registerGenericTaintChecker(CheckerManager &mgr) {
  mgr.registerChecker<GenericTaintChecker>();
}
