// BugReporterVisitors.cpp - Helpers for reporting bugs -----------*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a set of BugReporter "visitors" which can be used to
//  enhance the diagnostics reported for a bug.
//
//===----------------------------------------------------------------------===//
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporterVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprObjC.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/PathDiagnostic.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExplodedGraph.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;

using llvm::FoldingSetNodeID;

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

bool bugreporter::isDeclRefExprToReference(const Expr *E) {
  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    return DRE->getDecl()->getType()->isReferenceType();
  }
  return false;
}

const Expr *bugreporter::getDerefExpr(const Stmt *S) {
  // Pattern match for a few useful cases (do something smarter later):
  //   a[0], p->f, *p
  const Expr *E = dyn_cast<Expr>(S);
  if (!E)
    return 0;
  E = E->IgnoreParenCasts();

  while (true) {
    if (const BinaryOperator *B = dyn_cast<BinaryOperator>(E)) {
      assert(B->isAssignmentOp());
      E = B->getLHS()->IgnoreParenCasts();
      continue;
    }
    else if (const UnaryOperator *U = dyn_cast<UnaryOperator>(E)) {
      if (U->getOpcode() == UO_Deref)
        return U->getSubExpr()->IgnoreParenCasts();
    }
    else if (const MemberExpr *ME = dyn_cast<MemberExpr>(E)) {
      if (ME->isArrow() || isDeclRefExprToReference(ME->getBase())) {
        return ME->getBase()->IgnoreParenCasts();
      }
    }
    else if (const ObjCIvarRefExpr *IvarRef = dyn_cast<ObjCIvarRefExpr>(E)) {
      return IvarRef->getBase()->IgnoreParenCasts();
    }
    else if (const ArraySubscriptExpr *AE = dyn_cast<ArraySubscriptExpr>(E)) {
      return AE->getBase();
    }
    break;
  }

  return NULL;
}

const Stmt *bugreporter::GetDenomExpr(const ExplodedNode *N) {
  const Stmt *S = N->getLocationAs<PreStmt>()->getStmt();
  if (const BinaryOperator *BE = dyn_cast<BinaryOperator>(S))
    return BE->getRHS();
  return NULL;
}

const Stmt *bugreporter::GetRetValExpr(const ExplodedNode *N) {
  const Stmt *S = N->getLocationAs<PostStmt>()->getStmt();
  if (const ReturnStmt *RS = dyn_cast<ReturnStmt>(S))
    return RS->getRetValue();
  return NULL;
}

//===----------------------------------------------------------------------===//
// Definitions for bug reporter visitors.
//===----------------------------------------------------------------------===//

PathDiagnosticPiece*
BugReporterVisitor::getEndPath(BugReporterContext &BRC,
                               const ExplodedNode *EndPathNode,
                               BugReport &BR) {
  return 0;
}

PathDiagnosticPiece*
BugReporterVisitor::getDefaultEndPath(BugReporterContext &BRC,
                                      const ExplodedNode *EndPathNode,
                                      BugReport &BR) {
  PathDiagnosticLocation L =
    PathDiagnosticLocation::createEndOfPath(EndPathNode,BRC.getSourceManager());

  BugReport::ranges_iterator Beg, End;
  llvm::tie(Beg, End) = BR.getRanges();

  // Only add the statement itself as a range if we didn't specify any
  // special ranges for this report.
  PathDiagnosticPiece *P = new PathDiagnosticEventPiece(L,
      BR.getDescription(),
      Beg == End);
  for (; Beg != End; ++Beg)
    P->addRange(*Beg);

  return P;
}


namespace {
/// Emits an extra note at the return statement of an interesting stack frame.
///
/// The returned value is marked as an interesting value, and if it's null,
/// adds a visitor to track where it became null.
///
/// This visitor is intended to be used when another visitor discovers that an
/// interesting value comes from an inlined function call.
class ReturnVisitor : public BugReporterVisitorImpl<ReturnVisitor> {
  const StackFrameContext *StackFrame;
  enum {
    Initial,
    MaybeUnsuppress,
    Satisfied
  } Mode;

  bool EnableNullFPSuppression;

public:
  ReturnVisitor(const StackFrameContext *Frame, bool Suppressed)
    : StackFrame(Frame), Mode(Initial), EnableNullFPSuppression(Suppressed) {}

  static void *getTag() {
    static int Tag = 0;
    return static_cast<void *>(&Tag);
  }

  virtual void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddPointer(ReturnVisitor::getTag());
    ID.AddPointer(StackFrame);
    ID.AddBoolean(EnableNullFPSuppression);
  }

  /// Adds a ReturnVisitor if the given statement represents a call that was
  /// inlined.
  ///
  /// This will search back through the ExplodedGraph, starting from the given
  /// node, looking for when the given statement was processed. If it turns out
  /// the statement is a call that was inlined, we add the visitor to the
  /// bug report, so it can print a note later.
  static void addVisitorIfNecessary(const ExplodedNode *Node, const Stmt *S,
                                    BugReport &BR,
                                    bool InEnableNullFPSuppression) {
    if (!CallEvent::isCallStmt(S))
      return;
    
    // First, find when we processed the statement.
    do {
      if (Optional<CallExitEnd> CEE = Node->getLocationAs<CallExitEnd>())
        if (CEE->getCalleeContext()->getCallSite() == S)
          break;
      if (Optional<StmtPoint> SP = Node->getLocationAs<StmtPoint>())
        if (SP->getStmt() == S)
          break;

      Node = Node->getFirstPred();
    } while (Node);

    // Next, step over any post-statement checks.
    while (Node && Node->getLocation().getAs<PostStmt>())
      Node = Node->getFirstPred();
    if (!Node)
      return;

    // Finally, see if we inlined the call.
    Optional<CallExitEnd> CEE = Node->getLocationAs<CallExitEnd>();
    if (!CEE)
      return;
    
    const StackFrameContext *CalleeContext = CEE->getCalleeContext();
    if (CalleeContext->getCallSite() != S)
      return;
    
    // Check the return value.
    ProgramStateRef State = Node->getState();
    SVal RetVal = State->getSVal(S, Node->getLocationContext());

    // Handle cases where a reference is returned and then immediately used.
    if (cast<Expr>(S)->isGLValue())
      if (Optional<Loc> LValue = RetVal.getAs<Loc>())
        RetVal = State->getSVal(*LValue);

    // See if the return value is NULL. If so, suppress the report.
    SubEngine *Eng = State->getStateManager().getOwningEngine();
    assert(Eng && "Cannot file a bug report without an owning engine");
    AnalyzerOptions &Options = Eng->getAnalysisManager().options;

    bool EnableNullFPSuppression = false;
    if (InEnableNullFPSuppression && Options.shouldSuppressNullReturnPaths())
      if (Optional<Loc> RetLoc = RetVal.getAs<Loc>())
        EnableNullFPSuppression = State->isNull(*RetLoc).isConstrainedTrue();

    BR.markInteresting(CalleeContext);
    BR.addVisitor(new ReturnVisitor(CalleeContext, EnableNullFPSuppression));
  }

  /// Returns true if any counter-suppression heuristics are enabled for
  /// ReturnVisitor.
  static bool hasCounterSuppression(AnalyzerOptions &Options) {
    return Options.shouldAvoidSuppressingNullArgumentPaths();
  }

  PathDiagnosticPiece *visitNodeInitial(const ExplodedNode *N,
                                        const ExplodedNode *PrevN,
                                        BugReporterContext &BRC,
                                        BugReport &BR) {
    // Only print a message at the interesting return statement.
    if (N->getLocationContext() != StackFrame)
      return 0;

    Optional<StmtPoint> SP = N->getLocationAs<StmtPoint>();
    if (!SP)
      return 0;

    const ReturnStmt *Ret = dyn_cast<ReturnStmt>(SP->getStmt());
    if (!Ret)
      return 0;

    // Okay, we're at the right return statement, but do we have the return
    // value available?
    ProgramStateRef State = N->getState();
    SVal V = State->getSVal(Ret, StackFrame);
    if (V.isUnknownOrUndef())
      return 0;

    // Don't print any more notes after this one.
    Mode = Satisfied;

    const Expr *RetE = Ret->getRetValue();
    assert(RetE && "Tracking a return value for a void function");

    // Handle cases where a reference is returned and then immediately used.
    Optional<Loc> LValue;
    if (RetE->isGLValue()) {
      if ((LValue = V.getAs<Loc>())) {
        SVal RValue = State->getRawSVal(*LValue, RetE->getType());
        if (RValue.getAs<DefinedSVal>())
          V = RValue;
      }
    }

    // Ignore aggregate rvalues.
    if (V.getAs<nonloc::LazyCompoundVal>() ||
        V.getAs<nonloc::CompoundVal>())
      return 0;

    RetE = RetE->IgnoreParenCasts();

    // If we can't prove the return value is 0, just mark it interesting, and
    // make sure to track it into any further inner functions.
    if (!State->isNull(V).isConstrainedTrue()) {
      BR.markInteresting(V);
      ReturnVisitor::addVisitorIfNecessary(N, RetE, BR,
                                           EnableNullFPSuppression);
      return 0;
    }
      
    // If we're returning 0, we should track where that 0 came from.
    bugreporter::trackNullOrUndefValue(N, RetE, BR, /*IsArg*/ false,
                                       EnableNullFPSuppression);

    // Build an appropriate message based on the return value.
    SmallString<64> Msg;
    llvm::raw_svector_ostream Out(Msg);

    if (V.getAs<Loc>()) {
      // If we have counter-suppression enabled, make sure we keep visiting
      // future nodes. We want to emit a path note as well, in case
      // the report is resurrected as valid later on.
      ExprEngine &Eng = BRC.getBugReporter().getEngine();
      AnalyzerOptions &Options = Eng.getAnalysisManager().options;
      if (EnableNullFPSuppression && hasCounterSuppression(Options))
        Mode = MaybeUnsuppress;

      if (RetE->getType()->isObjCObjectPointerType())
        Out << "Returning nil";
      else
        Out << "Returning null pointer";
    } else {
      Out << "Returning zero";
    }

    if (LValue) {
      if (const MemRegion *MR = LValue->getAsRegion()) {
        if (MR->canPrintPretty()) {
          Out << " (reference to '";
          MR->printPretty(Out);
          Out << "')";
        }
      }
    } else {
      // FIXME: We should have a more generalized location printing mechanism.
      if (const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(RetE))
        if (const DeclaratorDecl *DD = dyn_cast<DeclaratorDecl>(DR->getDecl()))
          Out << " (loaded from '" << *DD << "')";
    }

    PathDiagnosticLocation L(Ret, BRC.getSourceManager(), StackFrame);
    return new PathDiagnosticEventPiece(L, Out.str());
  }

  PathDiagnosticPiece *visitNodeMaybeUnsuppress(const ExplodedNode *N,
                                                const ExplodedNode *PrevN,
                                                BugReporterContext &BRC,
                                                BugReport &BR) {
#ifndef NDEBUG
    ExprEngine &Eng = BRC.getBugReporter().getEngine();
    AnalyzerOptions &Options = Eng.getAnalysisManager().options;
    assert(hasCounterSuppression(Options));
#endif

    // Are we at the entry node for this call?
    Optional<CallEnter> CE = N->getLocationAs<CallEnter>();
    if (!CE)
      return 0;

    if (CE->getCalleeContext() != StackFrame)
      return 0;

    Mode = Satisfied;

    // Don't automatically suppress a report if one of the arguments is
    // known to be a null pointer. Instead, start tracking /that/ null
    // value back to its origin.
    ProgramStateManager &StateMgr = BRC.getStateManager();
    CallEventManager &CallMgr = StateMgr.getCallEventManager();

    ProgramStateRef State = N->getState();
    CallEventRef<> Call = CallMgr.getCaller(StackFrame, State);
    for (unsigned I = 0, E = Call->getNumArgs(); I != E; ++I) {
      Optional<Loc> ArgV = Call->getArgSVal(I).getAs<Loc>();
      if (!ArgV)
        continue;

      const Expr *ArgE = Call->getArgExpr(I);
      if (!ArgE)
        continue;

      // Is it possible for this argument to be non-null?
      if (!State->isNull(*ArgV).isConstrainedTrue())
        continue;

      if (bugreporter::trackNullOrUndefValue(N, ArgE, BR, /*IsArg=*/true,
                                             EnableNullFPSuppression))
        BR.removeInvalidation(ReturnVisitor::getTag(), StackFrame);

      // If we /can't/ track the null pointer, we should err on the side of
      // false negatives, and continue towards marking this report invalid.
      // (We will still look at the other arguments, though.)
    }

    return 0;
  }

  PathDiagnosticPiece *VisitNode(const ExplodedNode *N,
                                 const ExplodedNode *PrevN,
                                 BugReporterContext &BRC,
                                 BugReport &BR) {
    switch (Mode) {
    case Initial:
      return visitNodeInitial(N, PrevN, BRC, BR);
    case MaybeUnsuppress:
      return visitNodeMaybeUnsuppress(N, PrevN, BRC, BR);
    case Satisfied:
      return 0;
    }

    llvm_unreachable("Invalid visit mode!");
  }

  PathDiagnosticPiece *getEndPath(BugReporterContext &BRC,
                                  const ExplodedNode *N,
                                  BugReport &BR) {
    if (EnableNullFPSuppression)
      BR.markInvalid(ReturnVisitor::getTag(), StackFrame);
    return 0;
  }
};
} // end anonymous namespace


void FindLastStoreBRVisitor ::Profile(llvm::FoldingSetNodeID &ID) const {
  static int tag = 0;
  ID.AddPointer(&tag);
  ID.AddPointer(R);
  ID.Add(V);
  ID.AddBoolean(EnableNullFPSuppression);
}

PathDiagnosticPiece *FindLastStoreBRVisitor::VisitNode(const ExplodedNode *Succ,
                                                       const ExplodedNode *Pred,
                                                       BugReporterContext &BRC,
                                                       BugReport &BR) {

  if (Satisfied)
    return NULL;

  const ExplodedNode *StoreSite = 0;
  const Expr *InitE = 0;
  bool IsParam = false;

  // First see if we reached the declaration of the region.
  if (const VarRegion *VR = dyn_cast<VarRegion>(R)) {
    if (Optional<PostStmt> P = Pred->getLocationAs<PostStmt>()) {
      if (const DeclStmt *DS = P->getStmtAs<DeclStmt>()) {
        if (DS->getSingleDecl() == VR->getDecl()) {
          StoreSite = Pred;
          InitE = VR->getDecl()->getInit();
        }
      }
    }
  }

  // Otherwise, see if this is the store site:
  // (1) Succ has this binding and Pred does not, i.e. this is
  //     where the binding first occurred.
  // (2) Succ has this binding and is a PostStore node for this region, i.e.
  //     the same binding was re-assigned here.
  if (!StoreSite) {
    if (Succ->getState()->getSVal(R) != V)
      return NULL;

    if (Pred->getState()->getSVal(R) == V) {
      Optional<PostStore> PS = Succ->getLocationAs<PostStore>();
      if (!PS || PS->getLocationValue() != R)
        return NULL;
    }

    StoreSite = Succ;

    // If this is an assignment expression, we can track the value
    // being assigned.
    if (Optional<PostStmt> P = Succ->getLocationAs<PostStmt>())
      if (const BinaryOperator *BO = P->getStmtAs<BinaryOperator>())
        if (BO->isAssignmentOp())
          InitE = BO->getRHS();

    // If this is a call entry, the variable should be a parameter.
    // FIXME: Handle CXXThisRegion as well. (This is not a priority because
    // 'this' should never be NULL, but this visitor isn't just for NULL and
    // UndefinedVal.)
    if (Optional<CallEnter> CE = Succ->getLocationAs<CallEnter>()) {
      if (const VarRegion *VR = dyn_cast<VarRegion>(R)) {
        const ParmVarDecl *Param = cast<ParmVarDecl>(VR->getDecl());
        
        ProgramStateManager &StateMgr = BRC.getStateManager();
        CallEventManager &CallMgr = StateMgr.getCallEventManager();

        CallEventRef<> Call = CallMgr.getCaller(CE->getCalleeContext(),
                                                Succ->getState());
        InitE = Call->getArgExpr(Param->getFunctionScopeIndex());
        IsParam = true;
      }
    }

    // If this is a CXXTempObjectRegion, the Expr responsible for its creation
    // is wrapped inside of it.
    if (const CXXTempObjectRegion *TmpR = dyn_cast<CXXTempObjectRegion>(R))
      InitE = TmpR->getExpr();
  }

  if (!StoreSite)
    return NULL;
  Satisfied = true;

  // If we have an expression that provided the value, try to track where it
  // came from.
  if (InitE) {
    if (V.isUndef() || V.getAs<loc::ConcreteInt>()) {
      if (!IsParam)
        InitE = InitE->IgnoreParenCasts();
      bugreporter::trackNullOrUndefValue(StoreSite, InitE, BR, IsParam,
                                         EnableNullFPSuppression);
    } else {
      ReturnVisitor::addVisitorIfNecessary(StoreSite, InitE->IgnoreParenCasts(),
                                           BR, EnableNullFPSuppression);
    }
  }

  if (!R->canPrintPretty())
    return 0;

  // Okay, we've found the binding. Emit an appropriate message.
  SmallString<256> sbuf;
  llvm::raw_svector_ostream os(sbuf);

  if (Optional<PostStmt> PS = StoreSite->getLocationAs<PostStmt>()) {
    const Stmt *S = PS->getStmt();
    const char *action = 0;
    const DeclStmt *DS = dyn_cast<DeclStmt>(S);
    const VarRegion *VR = dyn_cast<VarRegion>(R);

    if (DS) {
      action = "initialized to ";
    } else if (isa<BlockExpr>(S)) {
      action = "captured by block as ";
      if (VR) {
        // See if we can get the BlockVarRegion.
        ProgramStateRef State = StoreSite->getState();
        SVal V = State->getSVal(S, PS->getLocationContext());
        if (const BlockDataRegion *BDR =
              dyn_cast_or_null<BlockDataRegion>(V.getAsRegion())) {
          if (const VarRegion *OriginalR = BDR->getOriginalRegion(VR)) {
            if (Optional<KnownSVal> KV =
                State->getSVal(OriginalR).getAs<KnownSVal>())
              BR.addVisitor(new FindLastStoreBRVisitor(*KV, OriginalR,
                                                      EnableNullFPSuppression));
          }
        }
      }
    }

    if (action) {
      if (!R)
        return 0;

      os << '\'';
      R->printPretty(os);
      os << "' ";

      if (V.getAs<loc::ConcreteInt>()) {
        bool b = false;
        if (R->isBoundable()) {
          if (const TypedValueRegion *TR = dyn_cast<TypedValueRegion>(R)) {
            if (TR->getValueType()->isObjCObjectPointerType()) {
              os << action << "nil";
              b = true;
            }
          }
        }

        if (!b)
          os << action << "a null pointer value";
      } else if (Optional<nonloc::ConcreteInt> CVal =
                     V.getAs<nonloc::ConcreteInt>()) {
        os << action << CVal->getValue();
      }
      else if (DS) {
        if (V.isUndef()) {
          if (isa<VarRegion>(R)) {
            const VarDecl *VD = cast<VarDecl>(DS->getSingleDecl());
            if (VD->getInit())
              os << "initialized to a garbage value";
            else
              os << "declared without an initial value";
          }
        }
        else {
          os << "initialized here";
        }
      }
    }
  } else if (StoreSite->getLocation().getAs<CallEnter>()) {
    if (const VarRegion *VR = dyn_cast<VarRegion>(R)) {
      const ParmVarDecl *Param = cast<ParmVarDecl>(VR->getDecl());

      os << "Passing ";

      if (V.getAs<loc::ConcreteInt>()) {
        if (Param->getType()->isObjCObjectPointerType())
          os << "nil object reference";
        else
          os << "null pointer value";
      } else if (V.isUndef()) {
        os << "uninitialized value";
      } else if (Optional<nonloc::ConcreteInt> CI =
                     V.getAs<nonloc::ConcreteInt>()) {
        os << "the value " << CI->getValue();
      } else {
        os << "value";
      }

      // Printed parameter indexes are 1-based, not 0-based.
      unsigned Idx = Param->getFunctionScopeIndex() + 1;
      os << " via " << Idx << llvm::getOrdinalSuffix(Idx) << " parameter '";

      R->printPretty(os);
      os << '\'';
    }
  }

  if (os.str().empty()) {
    if (V.getAs<loc::ConcreteInt>()) {
      bool b = false;
      if (R->isBoundable()) {
        if (const TypedValueRegion *TR = dyn_cast<TypedValueRegion>(R)) {
          if (TR->getValueType()->isObjCObjectPointerType()) {
            os << "nil object reference stored to ";
            b = true;
          }
        }
      }

      if (!b)
        os << "Null pointer value stored to ";
    }
    else if (V.isUndef()) {
      os << "Uninitialized value stored to ";
    } else if (Optional<nonloc::ConcreteInt> CV =
                   V.getAs<nonloc::ConcreteInt>()) {
      os << "The value " << CV->getValue() << " is assigned to ";
    }
    else
      os << "Value assigned to ";

    os << '\'';
    R->printPretty(os);
    os << '\'';
  }

  // Construct a new PathDiagnosticPiece.
  ProgramPoint P = StoreSite->getLocation();
  PathDiagnosticLocation L;
  if (P.getAs<CallEnter>() && InitE)
    L = PathDiagnosticLocation(InitE, BRC.getSourceManager(),
                               P.getLocationContext());
  else
    L = PathDiagnosticLocation::create(P, BRC.getSourceManager());
  if (!L.isValid())
    return NULL;
  return new PathDiagnosticEventPiece(L, os.str());
}

void TrackConstraintBRVisitor::Profile(llvm::FoldingSetNodeID &ID) const {
  static int tag = 0;
  ID.AddPointer(&tag);
  ID.AddBoolean(Assumption);
  ID.Add(Constraint);
}

/// Return the tag associated with this visitor.  This tag will be used
/// to make all PathDiagnosticPieces created by this visitor.
const char *TrackConstraintBRVisitor::getTag() {
  return "TrackConstraintBRVisitor";
}

bool TrackConstraintBRVisitor::isUnderconstrained(const ExplodedNode *N) const {
  if (IsZeroCheck)
    return N->getState()->isNull(Constraint).isUnderconstrained();
  return N->getState()->assume(Constraint, !Assumption);
}

PathDiagnosticPiece *
TrackConstraintBRVisitor::VisitNode(const ExplodedNode *N,
                                    const ExplodedNode *PrevN,
                                    BugReporterContext &BRC,
                                    BugReport &BR) {
  if (IsSatisfied)
    return NULL;

  // Check if in the previous state it was feasible for this constraint
  // to *not* be true.
  if (isUnderconstrained(PrevN)) {

    IsSatisfied = true;

    // As a sanity check, make sure that the negation of the constraint
    // was infeasible in the current state.  If it is feasible, we somehow
    // missed the transition point.
    if (isUnderconstrained(N))
      return NULL;

    // We found the transition point for the constraint.  We now need to
    // pretty-print the constraint. (work-in-progress)
    SmallString<64> sbuf;
    llvm::raw_svector_ostream os(sbuf);

    if (Constraint.getAs<Loc>()) {
      os << "Assuming pointer value is ";
      os << (Assumption ? "non-null" : "null");
    }

    if (os.str().empty())
      return NULL;

    // Construct a new PathDiagnosticPiece.
    ProgramPoint P = N->getLocation();
    PathDiagnosticLocation L =
      PathDiagnosticLocation::create(P, BRC.getSourceManager());
    if (!L.isValid())
      return NULL;
    
    PathDiagnosticEventPiece *X = new PathDiagnosticEventPiece(L, os.str());
    X->setTag(getTag());
    return X;
  }

  return NULL;
}

SuppressInlineDefensiveChecksVisitor::
SuppressInlineDefensiveChecksVisitor(DefinedSVal Value, const ExplodedNode *N)
  : V(Value), IsSatisfied(false), IsTrackingTurnedOn(false) {

    // Check if the visitor is disabled.
    SubEngine *Eng = N->getState()->getStateManager().getOwningEngine();
    assert(Eng && "Cannot file a bug report without an owning engine");
    AnalyzerOptions &Options = Eng->getAnalysisManager().options;
    if (!Options.shouldSuppressInlinedDefensiveChecks())
      IsSatisfied = true;

    assert(N->getState()->isNull(V).isConstrainedTrue() &&
           "The visitor only tracks the cases where V is constrained to 0");
}

void SuppressInlineDefensiveChecksVisitor::Profile(FoldingSetNodeID &ID) const {
  static int id = 0;
  ID.AddPointer(&id);
  ID.Add(V);
}

const char *SuppressInlineDefensiveChecksVisitor::getTag() {
  return "IDCVisitor";
}

PathDiagnosticPiece *
SuppressInlineDefensiveChecksVisitor::VisitNode(const ExplodedNode *Succ,
                                                const ExplodedNode *Pred,
                                                BugReporterContext &BRC,
                                                BugReport &BR) {
  if (IsSatisfied)
    return 0;

  // Start tracking after we see the first state in which the value is null.
  if (!IsTrackingTurnedOn)
    if (Succ->getState()->isNull(V).isConstrainedTrue())
      IsTrackingTurnedOn = true;
  if (!IsTrackingTurnedOn)
    return 0;

  // Check if in the previous state it was feasible for this value
  // to *not* be null.
  if (!Pred->getState()->isNull(V).isConstrainedTrue()) {
    IsSatisfied = true;

    assert(Succ->getState()->isNull(V).isConstrainedTrue());

    // Check if this is inlined defensive checks.
    const LocationContext *CurLC =Succ->getLocationContext();
    const LocationContext *ReportLC = BR.getErrorNode()->getLocationContext();
    if (CurLC != ReportLC && !CurLC->isParentOf(ReportLC))
      BR.markInvalid("Suppress IDC", CurLC);
  }
  return 0;
}

static const MemRegion *getLocationRegionIfReference(const Expr *E,
                                                     const ExplodedNode *N) {
  if (const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(E)) {
    if (const VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl())) {
      if (!VD->getType()->isReferenceType())
        return 0;
      ProgramStateManager &StateMgr = N->getState()->getStateManager();
      MemRegionManager &MRMgr = StateMgr.getRegionManager();
      return MRMgr.getVarRegion(VD, N->getLocationContext());
    }
  }

  // FIXME: This does not handle other kinds of null references,
  // for example, references from FieldRegions:
  //   struct Wrapper { int &ref; };
  //   Wrapper w = { *(int *)0 };
  //   w.ref = 1;

  return 0;
}

static const Expr *peelOffOuterExpr(const Stmt *S,
                                    const ExplodedNode *N) {
  if (const Expr *Ex = dyn_cast<Expr>(S)) {
    Ex = Ex->IgnoreParenCasts();
    if (const ExprWithCleanups *EWC = dyn_cast<ExprWithCleanups>(Ex))
      return EWC->getSubExpr();
    if (const OpaqueValueExpr *OVE = dyn_cast<OpaqueValueExpr>(Ex))
      return OVE->getSourceExpr();

    // Peel off the ternary operator.
    if (const ConditionalOperator *CO = dyn_cast<ConditionalOperator>(Ex)) {
      ProgramStateRef State = N->getState();
      SVal CondVal = State->getSVal(CO->getCond(), N->getLocationContext());
      if (State->isNull(CondVal).isConstrainedTrue()) {
        return CO->getTrueExpr();
      } else {
        assert(State->isNull(CondVal).isConstrainedFalse());
        return CO->getFalseExpr();
      }
    }
  }
  return 0;
}

bool bugreporter::trackNullOrUndefValue(const ExplodedNode *N,
                                        const Stmt *S,
                                        BugReport &report, bool IsArg,
                                        bool EnableNullFPSuppression) {
  if (!S || !N)
    return false;

  if (const Expr *Ex = peelOffOuterExpr(S, N)) {
    S = Ex;
  }

  // The message send could be null if the receiver is null.
  if (const Expr *Receiver = NilReceiverBRVisitor::getReceiver(S)) {
    report.addVisitor(new NilReceiverBRVisitor(Receiver,
                                               EnableNullFPSuppression));
  }

  const Expr *Inner = 0;
  if (const Expr *Ex = dyn_cast<Expr>(S)) {
    Ex = Ex->IgnoreParenCasts();
    if (ExplodedGraph::isInterestingLValueExpr(Ex) || CallEvent::isCallStmt(Ex))
      Inner = Ex;
  }

  if (IsArg) {
    assert(N->getLocation().getAs<CallEnter>() && "Tracking arg but not at call");
  } else {
    // Walk through nodes until we get one that matches the statement exactly.
    // Alternately, if we hit a known lvalue for the statement, we know we've
    // gone too far (though we can likely track the lvalue better anyway).
    do {
      const ProgramPoint &pp = N->getLocation();
      if (Optional<PostStmt> ps = pp.getAs<PostStmt>()) {
        if (ps->getStmt() == S || ps->getStmt() == Inner)
          break;
      } else if (Optional<CallExitEnd> CEE = pp.getAs<CallExitEnd>()) {
        if (CEE->getCalleeContext()->getCallSite() == S ||
            CEE->getCalleeContext()->getCallSite() == Inner)
          break;
      }
      N = N->getFirstPred();
    } while (N);

    if (!N)
      return false;
  }
  
  ProgramStateRef state = N->getState();

  // See if the expression we're interested refers to a variable. 
  // If so, we can track both its contents and constraints on its value.
  if (Inner && ExplodedGraph::isInterestingLValueExpr(Inner)) {
    const MemRegion *R = 0;

    // Find the ExplodedNode where the lvalue (the value of 'Ex')
    // was computed.  We need this for getting the location value.
    const ExplodedNode *LVNode = N;
    while (LVNode) {
      if (Optional<PostStmt> P = LVNode->getLocation().getAs<PostStmt>()) {
        if (P->getStmt() == Inner)
          break;
      }
      LVNode = LVNode->getFirstPred();
    }
    assert(LVNode && "Unable to find the lvalue node.");
    ProgramStateRef LVState = LVNode->getState();
    SVal LVal = LVState->getSVal(Inner, LVNode->getLocationContext());
    
    if (LVState->isNull(LVal).isConstrainedTrue()) {
      // In case of C++ references, we want to differentiate between a null
      // reference and reference to null pointer.
      // If the LVal is null, check if we are dealing with null reference.
      // For those, we want to track the location of the reference.
      if (const MemRegion *RR = getLocationRegionIfReference(Inner, N))
        R = RR;
    } else {
      R = LVState->getSVal(Inner, LVNode->getLocationContext()).getAsRegion();

      // If this is a C++ reference to a null pointer, we are tracking the
      // pointer. In additon, we should find the store at which the reference
      // got initialized.
      if (const MemRegion *RR = getLocationRegionIfReference(Inner, N)) {
        if (Optional<KnownSVal> KV = LVal.getAs<KnownSVal>())
          report.addVisitor(new FindLastStoreBRVisitor(*KV, RR,
                                                      EnableNullFPSuppression));
      }
    }

    if (R) {
      // Mark both the variable region and its contents as interesting.
      SVal V = state->getRawSVal(loc::MemRegionVal(R));

      // If the value matches the default for the variable region, that
      // might mean that it's been cleared out of the state. Fall back to
      // the full argument expression (with casts and such intact).
      if (IsArg) {
        bool UseArgValue = V.isUnknownOrUndef() || V.isZeroConstant();
        if (!UseArgValue) {
          const SymbolRegionValue *SRV =
            dyn_cast_or_null<SymbolRegionValue>(V.getAsLocSymbol());
          if (SRV)
            UseArgValue = (SRV->getRegion() == R);
        }
        if (UseArgValue)
          V = state->getSValAsScalarOrLoc(S, N->getLocationContext());
      }

      report.markInteresting(R);
      report.markInteresting(V);
      report.addVisitor(new UndefOrNullArgVisitor(R));

      if (isa<SymbolicRegion>(R)) {
        TrackConstraintBRVisitor *VI =
          new TrackConstraintBRVisitor(loc::MemRegionVal(R), false);
        report.addVisitor(VI);
      }

      // If the contents are symbolic, find out when they became null.
      if (V.getAsLocSymbol()) {
        BugReporterVisitor *ConstraintTracker =
          new TrackConstraintBRVisitor(V.castAs<DefinedSVal>(), false);
        report.addVisitor(ConstraintTracker);

        // Add visitor, which will suppress inline defensive checks.
        if (N->getState()->isNull(V).isConstrainedTrue() &&
            EnableNullFPSuppression) {
          BugReporterVisitor *IDCSuppressor =
            new SuppressInlineDefensiveChecksVisitor(V.castAs<DefinedSVal>(),
                                                     N);
          report.addVisitor(IDCSuppressor);
        }
      }

      if (Optional<KnownSVal> KV = V.getAs<KnownSVal>())
        report.addVisitor(new FindLastStoreBRVisitor(*KV, R,
                                                     EnableNullFPSuppression));
      return true;
    }
  }

  // If the expression is not an "lvalue expression", we can still
  // track the constraints on its contents.
  SVal V = state->getSValAsScalarOrLoc(S, N->getLocationContext());

  // If the value came from an inlined function call, we should at least make
  // sure that function isn't pruned in our output.
  if (const Expr *E = dyn_cast<Expr>(S))
    S = E->IgnoreParenCasts();

  ReturnVisitor::addVisitorIfNecessary(N, S, report, EnableNullFPSuppression);

  // Uncomment this to find cases where we aren't properly getting the
  // base value that was dereferenced.
  // assert(!V.isUnknownOrUndef());
  // Is it a symbolic value?
  if (Optional<loc::MemRegionVal> L = V.getAs<loc::MemRegionVal>()) {
    // At this point we are dealing with the region's LValue.
    // However, if the rvalue is a symbolic region, we should track it as well.
    SVal RVal = state->getSVal(L->getRegion());
    const MemRegion *RegionRVal = RVal.getAsRegion();
    report.addVisitor(new UndefOrNullArgVisitor(L->getRegion()));

    if (RegionRVal && isa<SymbolicRegion>(RegionRVal)) {
      report.markInteresting(RegionRVal);
      report.addVisitor(new TrackConstraintBRVisitor(
        loc::MemRegionVal(RegionRVal), false));
    }
  }

  return true;
}

const Expr *NilReceiverBRVisitor::getReceiver(const Stmt *S) {
  const ObjCMessageExpr *ME = dyn_cast<ObjCMessageExpr>(S);
  if (!ME)
    return 0;
  return ME->getInstanceReceiver();
}

PathDiagnosticPiece *NilReceiverBRVisitor::VisitNode(const ExplodedNode *N,
                                                     const ExplodedNode *PrevN,
                                                     BugReporterContext &BRC,
                                                     BugReport &BR) {
  Optional<PreStmt> P = N->getLocationAs<PreStmt>();
  if (!P)
    return 0;

  const Expr *Receiver = getReceiver(P->getStmt());
  if (!Receiver)
    return 0;

  // Are we tracking a different reciever?
  if (TrackedReceiver && TrackedReceiver != Receiver)
    return 0;

  ProgramStateRef state = N->getState();
  SVal V = state->getSVal(Receiver, N->getLocationContext());
  if (!state->isNull(V).isConstrainedTrue())
    return 0;

  // The receiver was nil, and hence the method was skipped.
  // Register a BugReporterVisitor to issue a message telling us how
  // the receiver was null.
  bugreporter::trackNullOrUndefValue(N, Receiver, BR, /*IsArg*/ false,
                                    EnableNullFPSuppression);
  // Issue a message saying that the method was skipped.
  PathDiagnosticLocation L(Receiver, BRC.getSourceManager(),
                                     N->getLocationContext());
  return new PathDiagnosticEventPiece(L, "No method is called "
      "because the receiver is nil");
}

// Registers every VarDecl inside a Stmt with a last store visitor.
void FindLastStoreBRVisitor::registerStatementVarDecls(BugReport &BR,
                                                const Stmt *S,
                                                bool EnableNullFPSuppression) {
  const ExplodedNode *N = BR.getErrorNode();
  std::deque<const Stmt *> WorkList;
  WorkList.push_back(S);

  while (!WorkList.empty()) {
    const Stmt *Head = WorkList.front();
    WorkList.pop_front();

    ProgramStateRef state = N->getState();
    ProgramStateManager &StateMgr = state->getStateManager();

    if (const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(Head)) {
      if (const VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl())) {
        const VarRegion *R =
        StateMgr.getRegionManager().getVarRegion(VD, N->getLocationContext());

        // What did we load?
        SVal V = state->getSVal(S, N->getLocationContext());

        if (V.getAs<loc::ConcreteInt>() || V.getAs<nonloc::ConcreteInt>()) {
          // Register a new visitor with the BugReport.
          BR.addVisitor(new FindLastStoreBRVisitor(V.castAs<KnownSVal>(), R,
                                                   EnableNullFPSuppression));
        }
      }
    }

    for (Stmt::const_child_iterator I = Head->child_begin();
        I != Head->child_end(); ++I)
      WorkList.push_back(*I);
  }
}

//===----------------------------------------------------------------------===//
// Visitor that tries to report interesting diagnostics from conditions.
//===----------------------------------------------------------------------===//

/// Return the tag associated with this visitor.  This tag will be used
/// to make all PathDiagnosticPieces created by this visitor.
const char *ConditionBRVisitor::getTag() {
  return "ConditionBRVisitor";
}

PathDiagnosticPiece *ConditionBRVisitor::VisitNode(const ExplodedNode *N,
                                                   const ExplodedNode *Prev,
                                                   BugReporterContext &BRC,
                                                   BugReport &BR) {
  PathDiagnosticPiece *piece = VisitNodeImpl(N, Prev, BRC, BR);
  if (piece) {
    piece->setTag(getTag());
    if (PathDiagnosticEventPiece *ev=dyn_cast<PathDiagnosticEventPiece>(piece))
      ev->setPrunable(true, /* override */ false);
  }
  return piece;
}

PathDiagnosticPiece *ConditionBRVisitor::VisitNodeImpl(const ExplodedNode *N,
                                                       const ExplodedNode *Prev,
                                                       BugReporterContext &BRC,
                                                       BugReport &BR) {
  
  ProgramPoint progPoint = N->getLocation();
  ProgramStateRef CurrentState = N->getState();
  ProgramStateRef PrevState = Prev->getState();
  
  // Compare the GDMs of the state, because that is where constraints
  // are managed.  Note that ensure that we only look at nodes that
  // were generated by the analyzer engine proper, not checkers.
  if (CurrentState->getGDM().getRoot() ==
      PrevState->getGDM().getRoot())
    return 0;
  
  // If an assumption was made on a branch, it should be caught
  // here by looking at the state transition.
  if (Optional<BlockEdge> BE = progPoint.getAs<BlockEdge>()) {
    const CFGBlock *srcBlk = BE->getSrc();    
    if (const Stmt *term = srcBlk->getTerminator())
      return VisitTerminator(term, N, srcBlk, BE->getDst(), BR, BRC);
    return 0;
  }
  
  if (Optional<PostStmt> PS = progPoint.getAs<PostStmt>()) {
    // FIXME: Assuming that BugReporter is a GRBugReporter is a layering
    // violation.
    const std::pair<const ProgramPointTag *, const ProgramPointTag *> &tags =      
      cast<GRBugReporter>(BRC.getBugReporter()).
        getEngine().geteagerlyAssumeBinOpBifurcationTags();

    const ProgramPointTag *tag = PS->getTag();
    if (tag == tags.first)
      return VisitTrueTest(cast<Expr>(PS->getStmt()), true,
                           BRC, BR, N);
    if (tag == tags.second)
      return VisitTrueTest(cast<Expr>(PS->getStmt()), false,
                           BRC, BR, N);
                           
    return 0;
  }
    
  return 0;
}

PathDiagnosticPiece *
ConditionBRVisitor::VisitTerminator(const Stmt *Term,
                                    const ExplodedNode *N,
                                    const CFGBlock *srcBlk,
                                    const CFGBlock *dstBlk,
                                    BugReport &R,
                                    BugReporterContext &BRC) {
  const Expr *Cond = 0;
  
  switch (Term->getStmtClass()) {
  default:
    return 0;
  case Stmt::IfStmtClass:
    Cond = cast<IfStmt>(Term)->getCond();
    break;
  case Stmt::ConditionalOperatorClass:
    Cond = cast<ConditionalOperator>(Term)->getCond();
    break;
  }      

  assert(Cond);
  assert(srcBlk->succ_size() == 2);
  const bool tookTrue = *(srcBlk->succ_begin()) == dstBlk;
  return VisitTrueTest(Cond, tookTrue, BRC, R, N);
}

PathDiagnosticPiece *
ConditionBRVisitor::VisitTrueTest(const Expr *Cond,
                                  bool tookTrue,
                                  BugReporterContext &BRC,
                                  BugReport &R,
                                  const ExplodedNode *N) {
  
  const Expr *Ex = Cond;
  
  while (true) {
    Ex = Ex->IgnoreParenCasts();
    switch (Ex->getStmtClass()) {
      default:
        return 0;
      case Stmt::BinaryOperatorClass:
        return VisitTrueTest(Cond, cast<BinaryOperator>(Ex), tookTrue, BRC,
                             R, N);
      case Stmt::DeclRefExprClass:
        return VisitTrueTest(Cond, cast<DeclRefExpr>(Ex), tookTrue, BRC,
                             R, N);
      case Stmt::UnaryOperatorClass: {
        const UnaryOperator *UO = cast<UnaryOperator>(Ex);
        if (UO->getOpcode() == UO_LNot) {
          tookTrue = !tookTrue;
          Ex = UO->getSubExpr();
          continue;
        }
        return 0;
      }
    }
  }
}

bool ConditionBRVisitor::patternMatch(const Expr *Ex, raw_ostream &Out,
                                      BugReporterContext &BRC,
                                      BugReport &report,
                                      const ExplodedNode *N,
                                      Optional<bool> &prunable) {
  const Expr *OriginalExpr = Ex;
  Ex = Ex->IgnoreParenCasts();

  if (const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(Ex)) {
    const bool quotes = isa<VarDecl>(DR->getDecl());
    if (quotes) {
      Out << '\'';
      const LocationContext *LCtx = N->getLocationContext();
      const ProgramState *state = N->getState().getPtr();
      if (const MemRegion *R = state->getLValue(cast<VarDecl>(DR->getDecl()),
                                                LCtx).getAsRegion()) {
        if (report.isInteresting(R))
          prunable = false;
        else {
          const ProgramState *state = N->getState().getPtr();
          SVal V = state->getSVal(R);
          if (report.isInteresting(V))
            prunable = false;
        }
      }
    }
    Out << DR->getDecl()->getDeclName().getAsString();
    if (quotes)
      Out << '\'';
    return quotes;
  }
  
  if (const IntegerLiteral *IL = dyn_cast<IntegerLiteral>(Ex)) {
    QualType OriginalTy = OriginalExpr->getType();
    if (OriginalTy->isPointerType()) {
      if (IL->getValue() == 0) {
        Out << "null";
        return false;
      }
    }
    else if (OriginalTy->isObjCObjectPointerType()) {
      if (IL->getValue() == 0) {
        Out << "nil";
        return false;
      }
    }
    
    Out << IL->getValue();
    return false;
  }
  
  return false;
}

PathDiagnosticPiece *
ConditionBRVisitor::VisitTrueTest(const Expr *Cond,
                                  const BinaryOperator *BExpr,
                                  const bool tookTrue,
                                  BugReporterContext &BRC,
                                  BugReport &R,
                                  const ExplodedNode *N) {
  
  bool shouldInvert = false;
  Optional<bool> shouldPrune;
  
  SmallString<128> LhsString, RhsString;
  {
    llvm::raw_svector_ostream OutLHS(LhsString), OutRHS(RhsString);
    const bool isVarLHS = patternMatch(BExpr->getLHS(), OutLHS, BRC, R, N,
                                       shouldPrune);
    const bool isVarRHS = patternMatch(BExpr->getRHS(), OutRHS, BRC, R, N,
                                       shouldPrune);
    
    shouldInvert = !isVarLHS && isVarRHS;    
  }
  
  BinaryOperator::Opcode Op = BExpr->getOpcode();

  if (BinaryOperator::isAssignmentOp(Op)) {
    // For assignment operators, all that we care about is that the LHS
    // evaluates to "true" or "false".
    return VisitConditionVariable(LhsString, BExpr->getLHS(), tookTrue,
                                  BRC, R, N);
  }

  // For non-assignment operations, we require that we can understand
  // both the LHS and RHS.
  if (LhsString.empty() || RhsString.empty())
    return 0;
  
  // Should we invert the strings if the LHS is not a variable name?
  SmallString<256> buf;
  llvm::raw_svector_ostream Out(buf);
  Out << "Assuming " << (shouldInvert ? RhsString : LhsString) << " is ";

  // Do we need to invert the opcode?
  if (shouldInvert)
    switch (Op) {
      default: break;
      case BO_LT: Op = BO_GT; break;
      case BO_GT: Op = BO_LT; break;
      case BO_LE: Op = BO_GE; break;
      case BO_GE: Op = BO_LE; break;
    }

  if (!tookTrue)
    switch (Op) {
      case BO_EQ: Op = BO_NE; break;
      case BO_NE: Op = BO_EQ; break;
      case BO_LT: Op = BO_GE; break;
      case BO_GT: Op = BO_LE; break;
      case BO_LE: Op = BO_GT; break;
      case BO_GE: Op = BO_LT; break;
      default:
        return 0;
    }
  
  switch (Op) {
    case BO_EQ:
      Out << "equal to ";
      break;
    case BO_NE:
      Out << "not equal to ";
      break;
    default:
      Out << BinaryOperator::getOpcodeStr(Op) << ' ';
      break;
  }
  
  Out << (shouldInvert ? LhsString : RhsString);
  const LocationContext *LCtx = N->getLocationContext();
  PathDiagnosticLocation Loc(Cond, BRC.getSourceManager(), LCtx);
  PathDiagnosticEventPiece *event =
    new PathDiagnosticEventPiece(Loc, Out.str());
  if (shouldPrune.hasValue())
    event->setPrunable(shouldPrune.getValue());
  return event;
}

PathDiagnosticPiece *
ConditionBRVisitor::VisitConditionVariable(StringRef LhsString,
                                           const Expr *CondVarExpr,
                                           const bool tookTrue,
                                           BugReporterContext &BRC,
                                           BugReport &report,
                                           const ExplodedNode *N) {
  // FIXME: If there's already a constraint tracker for this variable,
  // we shouldn't emit anything here (c.f. the double note in
  // test/Analysis/inlining/path-notes.c)
  SmallString<256> buf;
  llvm::raw_svector_ostream Out(buf);
  Out << "Assuming " << LhsString << " is ";
  
  QualType Ty = CondVarExpr->getType();

  if (Ty->isPointerType())
    Out << (tookTrue ? "not null" : "null");
  else if (Ty->isObjCObjectPointerType())
    Out << (tookTrue ? "not nil" : "nil");
  else if (Ty->isBooleanType())
    Out << (tookTrue ? "true" : "false");
  else if (Ty->isIntegerType())
    Out << (tookTrue ? "non-zero" : "zero");
  else
    return 0;

  const LocationContext *LCtx = N->getLocationContext();
  PathDiagnosticLocation Loc(CondVarExpr, BRC.getSourceManager(), LCtx);
  PathDiagnosticEventPiece *event =
    new PathDiagnosticEventPiece(Loc, Out.str());

  if (const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(CondVarExpr)) {
    if (const VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl())) {
      const ProgramState *state = N->getState().getPtr();
      if (const MemRegion *R = state->getLValue(VD, LCtx).getAsRegion()) {
        if (report.isInteresting(R))
          event->setPrunable(false);
      }
    }
  }
  
  return event;
}
  
PathDiagnosticPiece *
ConditionBRVisitor::VisitTrueTest(const Expr *Cond,
                                  const DeclRefExpr *DR,
                                  const bool tookTrue,
                                  BugReporterContext &BRC,
                                  BugReport &report,
                                  const ExplodedNode *N) {

  const VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl());
  if (!VD)
    return 0;
  
  SmallString<256> Buf;
  llvm::raw_svector_ostream Out(Buf);
    
  Out << "Assuming '";
  VD->getDeclName().printName(Out);
  Out << "' is ";
    
  QualType VDTy = VD->getType();
  
  if (VDTy->isPointerType())
    Out << (tookTrue ? "non-null" : "null");
  else if (VDTy->isObjCObjectPointerType())
    Out << (tookTrue ? "non-nil" : "nil");
  else if (VDTy->isScalarType())
    Out << (tookTrue ? "not equal to 0" : "0");
  else
    return 0;
  
  const LocationContext *LCtx = N->getLocationContext();
  PathDiagnosticLocation Loc(Cond, BRC.getSourceManager(), LCtx);
  PathDiagnosticEventPiece *event =
    new PathDiagnosticEventPiece(Loc, Out.str());
  
  const ProgramState *state = N->getState().getPtr();
  if (const MemRegion *R = state->getLValue(VD, LCtx).getAsRegion()) {
    if (report.isInteresting(R))
      event->setPrunable(false);
    else {
      SVal V = state->getSVal(R);
      if (report.isInteresting(V))
        event->setPrunable(false);
    }
  }
  return event;
}

PathDiagnosticPiece *
LikelyFalsePositiveSuppressionBRVisitor::getEndPath(BugReporterContext &BRC,
                                                    const ExplodedNode *N,
                                                    BugReport &BR) {
  const Stmt *S = BR.getStmt();
  if (!S)
    return 0;

  // Here we suppress false positives coming from system macros. This list is
  // based on known issues.

  // Skip reports within the sys/queue.h macros as we do not have the ability to
  // reason about data structure shapes.
  SourceManager &SM = BRC.getSourceManager();
  SourceLocation Loc = S->getLocStart();
  while (Loc.isMacroID()) {
    if (SM.isInSystemMacro(Loc) &&
       (SM.getFilename(SM.getSpellingLoc(Loc)).endswith("sys/queue.h"))) {
      BR.markInvalid(getTag(), 0);
      return 0;
    }
    Loc = SM.getSpellingLoc(Loc);
  }

  return 0;
}

PathDiagnosticPiece *
UndefOrNullArgVisitor::VisitNode(const ExplodedNode *N,
                                  const ExplodedNode *PrevN,
                                  BugReporterContext &BRC,
                                  BugReport &BR) {

  ProgramStateRef State = N->getState();
  ProgramPoint ProgLoc = N->getLocation();

  // We are only interested in visiting CallEnter nodes.
  Optional<CallEnter> CEnter = ProgLoc.getAs<CallEnter>();
  if (!CEnter)
    return 0;

  // Check if one of the arguments is the region the visitor is tracking.
  CallEventManager &CEMgr = BRC.getStateManager().getCallEventManager();
  CallEventRef<> Call = CEMgr.getCaller(CEnter->getCalleeContext(), State);
  unsigned Idx = 0;
  for (CallEvent::param_iterator I = Call->param_begin(),
                                 E = Call->param_end(); I != E; ++I, ++Idx) {
    const MemRegion *ArgReg = Call->getArgSVal(Idx).getAsRegion();

    // Are we tracking the argument or its subregion?
    if ( !ArgReg || (ArgReg != R && !R->isSubRegionOf(ArgReg->StripCasts())))
      continue;

    // Check the function parameter type.
    const ParmVarDecl *ParamDecl = *I;
    assert(ParamDecl && "Formal parameter has no decl?");
    QualType T = ParamDecl->getType();

    if (!(T->isAnyPointerType() || T->isReferenceType())) {
      // Function can only change the value passed in by address.
      continue;
    }
    
    // If it is a const pointer value, the function does not intend to
    // change the value.
    if (T->getPointeeType().isConstQualified())
      continue;

    // Mark the call site (LocationContext) as interesting if the value of the 
    // argument is undefined or '0'/'NULL'.
    SVal BoundVal = State->getSVal(R);
    if (BoundVal.isUndef() || BoundVal.isZeroConstant()) {
      BR.markInteresting(CEnter->getCalleeContext());
      return 0;
    }
  }
  return 0;
}
