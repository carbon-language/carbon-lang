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

#include "clang/AST/Expr.h"
#include "clang/AST/ExprObjC.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/PathDiagnostic.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExplodedGraph.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"

using namespace clang;
using namespace ento;

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

const Stmt *bugreporter::GetDerefExpr(const ExplodedNode *N) {
  // Pattern match for a few useful cases (do something smarter later):
  //   a[0], p->f, *p
  const Stmt *S = N->getLocationAs<PostStmt>()->getStmt();

  if (const UnaryOperator *U = dyn_cast<UnaryOperator>(S)) {
    if (U->getOpcode() == UO_Deref)
      return U->getSubExpr()->IgnoreParenCasts();
  }
  else if (const MemberExpr *ME = dyn_cast<MemberExpr>(S)) {
    return ME->getBase()->IgnoreParenCasts();
  }
  else if (const ArraySubscriptExpr *AE = dyn_cast<ArraySubscriptExpr>(S)) {
    return AE->getBase();
  }

  return NULL;
}

const Stmt *bugreporter::GetDenomExpr(const ExplodedNode *N) {
  const Stmt *S = N->getLocationAs<PreStmt>()->getStmt();
  if (const BinaryOperator *BE = dyn_cast<BinaryOperator>(S))
    return BE->getRHS();
  return NULL;
}

const Stmt *bugreporter::GetCalleeExpr(const ExplodedNode *N) {
  // Callee is checked as a PreVisit to the CallExpr.
  const Stmt *S = N->getLocationAs<PreStmt>()->getStmt();
  if (const CallExpr *CE = dyn_cast<CallExpr>(S))
    return CE->getCallee();
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

namespace {
class FindLastStoreBRVisitor : public BugReporterVisitor {
  const MemRegion *R;
  SVal V;
  bool satisfied;
  const ExplodedNode *StoreSite;
public:
  FindLastStoreBRVisitor(SVal v, const MemRegion *r)
  : R(r), V(v), satisfied(false), StoreSite(0) {}

  virtual void Profile(llvm::FoldingSetNodeID &ID) const {
    static int tag = 0;
    ID.AddPointer(&tag);
    ID.AddPointer(R);
    ID.Add(V);
  }

  PathDiagnosticPiece *VisitNode(const ExplodedNode *N,
                                 const ExplodedNode *PrevN,
                                 BugReporterContext &BRC) {

    if (satisfied)
      return NULL;

    if (!StoreSite) {
      const ExplodedNode *Node = N, *Last = NULL;

      for ( ; Node ; Last = Node, Node = Node->getFirstPred()) {

        if (const VarRegion *VR = dyn_cast<VarRegion>(R)) {
          if (const PostStmt *P = Node->getLocationAs<PostStmt>())
            if (const DeclStmt *DS = P->getStmtAs<DeclStmt>())
              if (DS->getSingleDecl() == VR->getDecl()) {
                Last = Node;
                break;
              }
        }

        if (Node->getState()->getSVal(R) != V)
          break;
      }

      if (!Node || !Last) {
        satisfied = true;
        return NULL;
      }

      StoreSite = Last;
    }

    if (StoreSite != N)
      return NULL;

    satisfied = true;
    llvm::SmallString<256> sbuf;
    llvm::raw_svector_ostream os(sbuf);

    if (const PostStmt *PS = N->getLocationAs<PostStmt>()) {
      if (const DeclStmt *DS = PS->getStmtAs<DeclStmt>()) {

        if (const VarRegion *VR = dyn_cast<VarRegion>(R)) {
          os << "Variable '" << VR->getDecl() << "' ";
        }
        else
          return NULL;

        if (isa<loc::ConcreteInt>(V)) {
          bool b = false;
          if (R->isBoundable()) {
            if (const TypedValueRegion *TR = dyn_cast<TypedValueRegion>(R)) {
              if (TR->getValueType()->isObjCObjectPointerType()) {
                os << "initialized to nil";
                b = true;
              }
            }
          }

          if (!b)
            os << "initialized to a null pointer value";
        }
        else if (isa<nonloc::ConcreteInt>(V)) {
          os << "initialized to " << cast<nonloc::ConcreteInt>(V).getValue();
        }
        else if (V.isUndef()) {
          if (isa<VarRegion>(R)) {
            const VarDecl *VD = cast<VarDecl>(DS->getSingleDecl());
            if (VD->getInit())
              os << "initialized to a garbage value";
            else
              os << "declared without an initial value";
          }
        }
      }
    }

    if (os.str().empty()) {
      if (isa<loc::ConcreteInt>(V)) {
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
      }
      else if (isa<nonloc::ConcreteInt>(V)) {
        os << "The value " << cast<nonloc::ConcreteInt>(V).getValue()
           << " is assigned to ";
      }
      else
        return NULL;

      if (const VarRegion *VR = dyn_cast<VarRegion>(R)) {
        os << '\'' << VR->getDecl() << '\'';
      }
      else
        return NULL;
    }

    // FIXME: Refactor this into BugReporterContext.
    const Stmt *S = 0;
    ProgramPoint P = N->getLocation();

    if (BlockEdge *BE = dyn_cast<BlockEdge>(&P)) {
      const CFGBlock *BSrc = BE->getSrc();
      S = BSrc->getTerminatorCondition();
    }
    else if (PostStmt *PS = dyn_cast<PostStmt>(&P)) {
      S = PS->getStmt();
    }

    if (!S)
      return NULL;

    // Construct a new PathDiagnosticPiece.
    PathDiagnosticLocation L(S, BRC.getSourceManager());
    return new PathDiagnosticEventPiece(L, os.str());
  }
};


static void registerFindLastStore(BugReporterContext &BRC, const MemRegion *R,
                                  SVal V) {
  BRC.addVisitor(new FindLastStoreBRVisitor(V, R));
}

class TrackConstraintBRVisitor : public BugReporterVisitor {
  DefinedSVal Constraint;
  const bool Assumption;
  bool isSatisfied;
public:
  TrackConstraintBRVisitor(DefinedSVal constraint, bool assumption)
  : Constraint(constraint), Assumption(assumption), isSatisfied(false) {}

  void Profile(llvm::FoldingSetNodeID &ID) const {
    static int tag = 0;
    ID.AddPointer(&tag);
    ID.AddBoolean(Assumption);
    ID.Add(Constraint);
  }

  PathDiagnosticPiece *VisitNode(const ExplodedNode *N,
                                 const ExplodedNode *PrevN,
                                 BugReporterContext &BRC) {
    if (isSatisfied)
      return NULL;

    // Check if in the previous state it was feasible for this constraint
    // to *not* be true.
    if (PrevN->getState()->assume(Constraint, !Assumption)) {

      isSatisfied = true;

      // As a sanity check, make sure that the negation of the constraint
      // was infeasible in the current state.  If it is feasible, we somehow
      // missed the transition point.
      if (N->getState()->assume(Constraint, !Assumption))
        return NULL;

      // We found the transition point for the constraint.  We now need to
      // pretty-print the constraint. (work-in-progress)
      std::string sbuf;
      llvm::raw_string_ostream os(sbuf);

      if (isa<Loc>(Constraint)) {
        os << "Assuming pointer value is ";
        os << (Assumption ? "non-null" : "null");
      }

      if (os.str().empty())
        return NULL;

      // FIXME: Refactor this into BugReporterContext.
      const Stmt *S = 0;
      ProgramPoint P = N->getLocation();

      if (BlockEdge *BE = dyn_cast<BlockEdge>(&P)) {
        const CFGBlock *BSrc = BE->getSrc();
        S = BSrc->getTerminatorCondition();
      }
      else if (PostStmt *PS = dyn_cast<PostStmt>(&P)) {
        S = PS->getStmt();
      }

      if (!S)
        return NULL;

      // Construct a new PathDiagnosticPiece.
      PathDiagnosticLocation L(S, BRC.getSourceManager());
      return new PathDiagnosticEventPiece(L, os.str());
    }

    return NULL;
  }
};
} // end anonymous namespace

static void registerTrackConstraint(BugReporterContext &BRC,
                                    DefinedSVal Constraint,
                                    bool Assumption) {
  BRC.addVisitor(new TrackConstraintBRVisitor(Constraint, Assumption));
}

void bugreporter::registerTrackNullOrUndefValue(BugReporterContext &BRC,
                                                const void *data,
                                                const ExplodedNode *N) {

  const Stmt *S = static_cast<const Stmt*>(data);

  if (!S)
    return;

  ProgramStateManager &StateMgr = BRC.getStateManager();
  
  // Walk through nodes until we get one that matches the statement
  // exactly.
  while (N) {
    const ProgramPoint &pp = N->getLocation();
    if (const PostStmt *ps = dyn_cast<PostStmt>(&pp)) {
      if (ps->getStmt() == S)
        break;
    }
    N = *N->pred_begin();
  }

  if (!N)
    return;
  
  const ProgramState *state = N->getState();

  // Walk through lvalue-to-rvalue conversions.  
  if (const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(S)) {
    if (const VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl())) {
      const VarRegion *R =
        StateMgr.getRegionManager().getVarRegion(VD, N->getLocationContext());

      // What did we load?
      SVal V = state->getSVal(loc::MemRegionVal(R));

      if (isa<loc::ConcreteInt>(V) || isa<nonloc::ConcreteInt>(V)
          || V.isUndef()) {
        ::registerFindLastStore(BRC, R, V);
      }
    }
  }

  SVal V = state->getSValAsScalarOrLoc(S);

  // Uncomment this to find cases where we aren't properly getting the
  // base value that was dereferenced.
  // assert(!V.isUnknownOrUndef());

  // Is it a symbolic value?
  if (loc::MemRegionVal *L = dyn_cast<loc::MemRegionVal>(&V)) {
    const SubRegion *R = cast<SubRegion>(L->getRegion());
    while (R && !isa<SymbolicRegion>(R)) {
      R = dyn_cast<SubRegion>(R->getSuperRegion());
    }

    if (R) {
      assert(isa<SymbolicRegion>(R));
      registerTrackConstraint(BRC, loc::MemRegionVal(R), false);
    }
  }
}

void bugreporter::registerFindLastStore(BugReporterContext &BRC,
                                        const void *data,
                                        const ExplodedNode *N) {

  const MemRegion *R = static_cast<const MemRegion*>(data);

  if (!R)
    return;

  const ProgramState *state = N->getState();
  SVal V = state->getSVal(R);

  if (V.isUnknown())
    return;

  BRC.addVisitor(new FindLastStoreBRVisitor(V, R));
}


namespace {
class NilReceiverVisitor : public BugReporterVisitor {
public:
  NilReceiverVisitor() {}

  void Profile(llvm::FoldingSetNodeID &ID) const {
    static int x = 0;
    ID.AddPointer(&x);
  }

  PathDiagnosticPiece *VisitNode(const ExplodedNode *N,
                                 const ExplodedNode *PrevN,
                                 BugReporterContext &BRC) {

    const PostStmt *P = N->getLocationAs<PostStmt>();
    if (!P)
      return 0;
    const ObjCMessageExpr *ME = P->getStmtAs<ObjCMessageExpr>();
    if (!ME)
      return 0;
    const Expr *Receiver = ME->getInstanceReceiver();
    if (!Receiver)
      return 0;
    const ProgramState *state = N->getState();
    const SVal &V = state->getSVal(Receiver);
    const DefinedOrUnknownSVal *DV = dyn_cast<DefinedOrUnknownSVal>(&V);
    if (!DV)
      return 0;
    state = state->assume(*DV, true);
    if (state)
      return 0;

    // The receiver was nil, and hence the method was skipped.
    // Register a BugReporterVisitor to issue a message telling us how
    // the receiver was null.
    bugreporter::registerTrackNullOrUndefValue(BRC, Receiver, N);
    // Issue a message saying that the method was skipped.
    PathDiagnosticLocation L(Receiver, BRC.getSourceManager());
    return new PathDiagnosticEventPiece(L, "No method actually called "
                                           "because the receiver is nil");
  }
};
} // end anonymous namespace

void bugreporter::registerNilReceiverVisitor(BugReporterContext &BRC) {
  BRC.addVisitor(new NilReceiverVisitor());
}

// Registers every VarDecl inside a Stmt with a last store vistor.
void bugreporter::registerVarDeclsLastStore(BugReporterContext &BRC,
                                                   const void *stmt,
                                                   const ExplodedNode *N) {
  const Stmt *S = static_cast<const Stmt *>(stmt);

  std::deque<const Stmt *> WorkList;

  WorkList.push_back(S);

  while (!WorkList.empty()) {
    const Stmt *Head = WorkList.front();
    WorkList.pop_front();

    ProgramStateManager &StateMgr = BRC.getStateManager();
    const ProgramState *state = N->getState();

    if (const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(Head)) {
      if (const VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl())) {
        const VarRegion *R =
        StateMgr.getRegionManager().getVarRegion(VD, N->getLocationContext());

        // What did we load?
        SVal V = state->getSVal(S);

        if (isa<loc::ConcreteInt>(V) || isa<nonloc::ConcreteInt>(V)) {
          ::registerFindLastStore(BRC, R, V);
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

namespace {
class ConditionVisitor : public BugReporterVisitor {
public:
  void Profile(llvm::FoldingSetNodeID &ID) const {
    static int x = 0;
    ID.AddPointer(&x);
  }

  virtual PathDiagnosticPiece *VisitNode(const ExplodedNode *N,
                                         const ExplodedNode *Prev,
                                         BugReporterContext &BRC);
  
  PathDiagnosticPiece *VisitTerminator(const Stmt *Term,
                                       const ProgramState *CurrentState,
                                       const ProgramState *PrevState,
                                       const CFGBlock *srcBlk,
                                       const CFGBlock *dstBlk,
                                       BugReporterContext &BRC);
  
  PathDiagnosticPiece *VisitTrueTest(const Expr *Cond,
                                     bool tookTrue,
                                     BugReporterContext &BRC);

  PathDiagnosticPiece *VisitTrueTest(const Expr *Cond,
                                     const DeclRefExpr *DR,
                                     const bool tookTrue,
                                     BugReporterContext &BRC);
  
  PathDiagnosticPiece *VisitTrueTest(const Expr *Cond,
                                     const BinaryOperator *BExpr,
                                     const bool tookTrue,
                                     BugReporterContext &BRC);
  
  void patternMatch(const Expr *Ex,
                    llvm::raw_ostream &Out,
                    BugReporterContext &BRC);
};
}

PathDiagnosticPiece *ConditionVisitor::VisitNode(const ExplodedNode *N,
                                                 const ExplodedNode *Prev,
                                                 BugReporterContext &BRC) {
  
  const ProgramPoint &progPoint = N->getLocation();

  const ProgramState *CurrentState = N->getState();
  const ProgramState *PrevState = Prev->getState();
  
  // Compare the GDMs of the state, because that is where constraints
  // are managed.  Note that ensure that we only look at nodes that
  // were generated by the analyzer engine proper, not checkers.
  if (CurrentState->getGDM().getRoot() ==
      PrevState->getGDM().getRoot())
    return 0;
  
  // If an assumption was made on a branch, it should be caught
  // here by looking at the state transition.
  if (const BlockEdge *BE = dyn_cast<BlockEdge>(&progPoint)) {
    const CFGBlock *srcBlk = BE->getSrc();    
    if (const Stmt *term = srcBlk->getTerminator())
      return VisitTerminator(term, CurrentState, PrevState,
                             srcBlk, BE->getDst(), BRC);
    return 0;
  }
  
  if (const PostStmt *PS = dyn_cast<PostStmt>(&progPoint)) {
    // FIXME: Assuming that BugReporter is a GRBugReporter is a layering
    // violation.
    const std::pair<const ProgramPointTag *, const ProgramPointTag *> &tags =      
      cast<GRBugReporter>(BRC.getBugReporter()).
        getEngine().getEagerlyAssumeTags();

    const ProgramPointTag *tag = PS->getTag();
    if (tag == tags.first)
      return VisitTrueTest(cast<Expr>(PS->getStmt()), true, BRC);
    if (tag == tags.second)
      return VisitTrueTest(cast<Expr>(PS->getStmt()), false, BRC);
                           
    return 0;
  }
    
  return 0;
}

PathDiagnosticPiece *
ConditionVisitor::VisitTerminator(const Stmt *Term,
                                  const ProgramState *CurrentState,
                                  const ProgramState *PrevState,
                                  const CFGBlock *srcBlk,
                                  const CFGBlock *dstBlk,
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
  return VisitTrueTest(Cond->IgnoreParenNoopCasts(BRC.getASTContext()),
                       tookTrue, BRC);
}

PathDiagnosticPiece *
ConditionVisitor::VisitTrueTest(const Expr *Cond,
                                bool tookTrue,
                                BugReporterContext &BRC) {
  
  const Expr *Ex = Cond;
  
  while (true) {
    Ex = Ex->IgnoreParens();
    switch (Ex->getStmtClass()) {
      default:
        return 0;
      case Stmt::BinaryOperatorClass:
        return VisitTrueTest(Cond, cast<BinaryOperator>(Cond), tookTrue, BRC);
      case Stmt::DeclRefExprClass:
        return VisitTrueTest(Cond, cast<DeclRefExpr>(Ex), tookTrue, BRC);
      case Stmt::UnaryOperatorClass: {
        const UnaryOperator *UO = cast<UnaryOperator>(Ex);
        if (UO->getOpcode() == UO_LNot) {
          tookTrue = !tookTrue;
          Ex = UO->getSubExpr()->IgnoreParenNoopCasts(BRC.getASTContext());
          continue;
        }
        return 0;
      }
    }
  }
}

void ConditionVisitor::patternMatch(const Expr *Ex, llvm::raw_ostream &Out,
                                    BugReporterContext &BRC) {
  const Expr *OriginalExpr = Ex;
  Ex = Ex->IgnoreParenCasts();

  if (const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(Ex)) {
    if (const VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl()))
      Out << VD->getDeclName().getAsString();
    return;
  }
  
  if (const IntegerLiteral *IL = dyn_cast<IntegerLiteral>(Ex)) {
    QualType OriginalTy = OriginalExpr->getType();
    if (OriginalTy->isPointerType()) {
      if (IL->getValue() == 0) {
        Out << "null";
        return;
      }
    }
    else if (OriginalTy->isObjCObjectPointerType()) {
      if (IL->getValue() == 0) {
        Out << "nil";
        return;
      }
    }
    
    Out << IL->getValue();
    return;
  }
}

PathDiagnosticPiece *
ConditionVisitor::VisitTrueTest(const Expr *Cond,
                                const BinaryOperator *BExpr,
                                const bool tookTrue,
                                BugReporterContext &BRC) {
  
  llvm::SmallString<128> LhsString, RhsString;
  {
    llvm::raw_svector_ostream OutLHS(LhsString), OutRHS(RhsString);  
    patternMatch(BExpr->getLHS(), OutLHS, BRC);
    patternMatch(BExpr->getRHS(), OutRHS, BRC);
  }
  
  if (LhsString.empty() || RhsString.empty())
    return 0;

  llvm::SmallString<256> buf;
  llvm::raw_svector_ostream Out(buf);
  Out << "Assuming " << LhsString << " is ";

  // Do we need to invert the opcode?
  BinaryOperator::Opcode Op = BExpr->getOpcode();
  
  if (!tookTrue)
    switch (Op) {
      case BO_EQ: Op = BO_NE; break;
      case BO_NE: Op = BO_EQ; break;
      case BO_LT: Op = BO_GE; break;
      case BO_GT: Op = BO_LE; break;
      case BO_LE: Op = BO_GT; break;
      case BO_GE: Op = BO_GE; break;
      default:
        return 0;
    }
  
  switch (BExpr->getOpcode()) {
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
  
  Out << RhsString;

  PathDiagnosticLocation Loc(Cond, BRC.getSourceManager());
  return new PathDiagnosticEventPiece(Loc, Out.str());
}
  
PathDiagnosticPiece *
ConditionVisitor::VisitTrueTest(const Expr *Cond,
                                const DeclRefExpr *DR,
                                const bool tookTrue,
                                BugReporterContext &BRC) {

  const VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl());
  if (!VD)
    return 0;
  
  llvm::SmallString<256> Buf;
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
  
  PathDiagnosticLocation Loc(Cond, BRC.getSourceManager());
  return new PathDiagnosticEventPiece(Loc, Out.str());
}

void bugreporter::registerConditionVisitor(BugReporterContext &BRC) {
  BRC.addVisitor(new ConditionVisitor());
}

