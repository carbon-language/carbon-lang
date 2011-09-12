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
  const ProgramPoint &PP = EndPathNode->getLocation();
  PathDiagnosticLocation L;

  if (const BlockEntrance *BE = dyn_cast<BlockEntrance>(&PP)) {
    const CFGBlock *block = BE->getBlock();
    if (block->getBlockID() == 0) {
      L = PathDiagnosticLocation(
          EndPathNode->getLocationContext()->getDecl()->getBodyRBrace(),
          BRC.getSourceManager());
    }
  }

  if (!L.isValid()) {
    const Stmt *S = BR.getStmt();

    if (!S)
      return NULL;

    L = PathDiagnosticLocation(S, BRC.getSourceManager());
  }

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


void FindLastStoreBRVisitor ::Profile(llvm::FoldingSetNodeID &ID) const {
  static int tag = 0;
  ID.AddPointer(&tag);
  ID.AddPointer(R);
  ID.Add(V);
}

PathDiagnosticPiece *FindLastStoreBRVisitor::VisitNode(const ExplodedNode *N,
                                                     const ExplodedNode *PrevN,
                                                     BugReporterContext &BRC,
                                                     BugReport &BR) {

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

void TrackConstraintBRVisitor::Profile(llvm::FoldingSetNodeID &ID) const {
  static int tag = 0;
  ID.AddPointer(&tag);
  ID.AddBoolean(Assumption);
  ID.Add(Constraint);
}

PathDiagnosticPiece *
TrackConstraintBRVisitor::VisitNode(const ExplodedNode *N,
                                    const ExplodedNode *PrevN,
                                    BugReporterContext &BRC,
                                    BugReport &BR) {
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

BugReporterVisitor *
bugreporter::getTrackNullOrUndefValueVisitor(const ExplodedNode *N,
                                             const Stmt *S) {
  if (!S || !N)
    return 0;

  ProgramStateManager &StateMgr = N->getState()->getStateManager();

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
    return 0;
  
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
        return new FindLastStoreBRVisitor(V, R);
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
      return new TrackConstraintBRVisitor(loc::MemRegionVal(R), false);
    }
  }

  return 0;
}

BugReporterVisitor *
FindLastStoreBRVisitor::createVisitorObject(const ExplodedNode *N,
                                            const MemRegion *R) {
  assert(R && "The memory region is null.");

  const ProgramState *state = N->getState();
  SVal V = state->getSVal(R);
  if (V.isUnknown())
    return 0;

  return new FindLastStoreBRVisitor(V, R);
}


PathDiagnosticPiece *NilReceiverBRVisitor::VisitNode(const ExplodedNode *N,
                                                     const ExplodedNode *PrevN,
                                                     BugReporterContext &BRC,
                                                     BugReport &BR) {
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
  BR.addVisitor(bugreporter::getTrackNullOrUndefValueVisitor(N, Receiver));
  // Issue a message saying that the method was skipped.
  PathDiagnosticLocation L(Receiver, BRC.getSourceManager());
  return new PathDiagnosticEventPiece(L, "No method actually called "
      "because the receiver is nil");
}

// Registers every VarDecl inside a Stmt with a last store visitor.
void FindLastStoreBRVisitor::registerStatementVarDecls(BugReport &BR,
                                                       const Stmt *S) {
  const ExplodedNode *N = BR.getErrorNode();
  std::deque<const Stmt *> WorkList;
  WorkList.push_back(S);

  while (!WorkList.empty()) {
    const Stmt *Head = WorkList.front();
    WorkList.pop_front();

    const ProgramState *state = N->getState();
    ProgramStateManager &StateMgr = state->getStateManager();

    if (const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(Head)) {
      if (const VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl())) {
        const VarRegion *R =
        StateMgr.getRegionManager().getVarRegion(VD, N->getLocationContext());

        // What did we load?
        SVal V = state->getSVal(S);

        if (isa<loc::ConcreteInt>(V) || isa<nonloc::ConcreteInt>(V)) {
          // Register a new visitor with the BugReport.
          BR.addVisitor(new FindLastStoreBRVisitor(V, R));
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
PathDiagnosticPiece *ConditionBRVisitor::VisitNode(const ExplodedNode *N,
                                                   const ExplodedNode *Prev,
                                                   BugReporterContext &BRC,
                                                   BugReport &BR) {
  
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
ConditionBRVisitor::VisitTerminator(const Stmt *Term,
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
ConditionBRVisitor::VisitTrueTest(const Expr *Cond,
                                  bool tookTrue,
                                  BugReporterContext &BRC) {
  
  const Expr *Ex = Cond;
  
  while (true) {
    Ex = Ex->IgnoreParens();
    switch (Ex->getStmtClass()) {
      default:
        return 0;
      case Stmt::BinaryOperatorClass:
        return VisitTrueTest(Cond, cast<BinaryOperator>(Ex), tookTrue, BRC);
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

bool ConditionBRVisitor::patternMatch(const Expr *Ex, llvm::raw_ostream &Out,
                                      BugReporterContext &BRC) {
  const Expr *OriginalExpr = Ex;
  Ex = Ex->IgnoreParenCasts();

  if (const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(Ex)) {
    const bool quotes = isa<VarDecl>(DR->getDecl());
    if (quotes)
      Out << '\'';
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
                                  BugReporterContext &BRC) {
  
  bool shouldInvert = false;
  
  llvm::SmallString<128> LhsString, RhsString;
  {
    llvm::raw_svector_ostream OutLHS(LhsString), OutRHS(RhsString);  
    const bool isVarLHS = patternMatch(BExpr->getLHS(), OutLHS, BRC);
    const bool isVarRHS = patternMatch(BExpr->getRHS(), OutRHS, BRC);
    
    shouldInvert = !isVarLHS && isVarRHS;    
  }
  
  if (LhsString.empty() || RhsString.empty())
    return 0;

  // Should we invert the strings if the LHS is not a variable name?
  
  llvm::SmallString<256> buf;
  llvm::raw_svector_ostream Out(buf);
  Out << "Assuming " << (shouldInvert ? RhsString : LhsString) << " is ";

  // Do we need to invert the opcode?
  BinaryOperator::Opcode Op = BExpr->getOpcode();
    
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
  
  Out << (shouldInvert ? LhsString : RhsString);

  PathDiagnosticLocation Loc(Cond, BRC.getSourceManager());
  return new PathDiagnosticEventPiece(Loc, Out.str());
}
  
PathDiagnosticPiece *
ConditionBRVisitor::VisitTrueTest(const Expr *Cond,
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
