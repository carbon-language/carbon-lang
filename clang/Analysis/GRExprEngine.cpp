//=-- GRExprEngine.cpp - Path-Sensitive Expression-Level Dataflow ---*- C++ -*-=
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a meta-engine for path-sensitive dataflow analysis that
//  is built on GREngine, but provides the boilerplate to execute transfer
//  functions and build the ExplodedGraph at the expression level.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/GRExprEngine.h"
#include "clang/Analysis/PathSensitive/GRTransferFuncs.h"

#include "llvm/Support/Streams.h"

using namespace clang;
using llvm::dyn_cast;
using llvm::cast;
using llvm::APSInt;

GRExprEngine::StateTy
GRExprEngine::SetRVal(StateTy St, Expr* Ex, const RVal& V) {

  if (!StateCleaned) {
    St = RemoveDeadBindings(CurrentStmt, St);
    StateCleaned = true;
  }

  bool isBlkExpr = false;
    
  if (Ex == CurrentStmt) {
    isBlkExpr = getCFG().isBlkExpr(Ex);
    
    if (!isBlkExpr)
      return St;
  }

  return StateMgr.SetRVal(St, Ex, isBlkExpr, V);
}

const GRExprEngine::StateTy::BufferTy&
GRExprEngine::SetRVal(StateTy St, Expr* Ex, const RVal::BufferTy& RB,
                      StateTy::BufferTy& RetBuf) {
  
  assert (RetBuf.empty());
  
  for (RVal::BufferTy::const_iterator I = RB.begin(), E = RB.end(); I!=E; ++I)
    RetBuf.push_back(SetRVal(St, Ex, *I));
                     
  return RetBuf;
}

GRExprEngine::StateTy
GRExprEngine::SetRVal(StateTy St, const LVal& LV, const RVal& RV) {
  
  if (!StateCleaned) {
    St = RemoveDeadBindings(CurrentStmt, St);
    StateCleaned = true;
  }
  
  return StateMgr.SetRVal(St, LV, RV);
}

void GRExprEngine::ProcessBranch(Expr* Condition, Stmt* Term,
                                 BranchNodeBuilder& builder) {

  // Remove old bindings for subexpressions.
  StateTy PrevState = StateMgr.RemoveSubExprBindings(builder.getState());
  
  // Check for NULL conditions; e.g. "for(;;)"
  if (!Condition) { 
    builder.markInfeasible(false);
    
    // Get the current block counter.
    GRBlockCounter BC = builder.getBlockCounter();
    unsigned BlockID = builder.getTargetBlock(true)->getBlockID();
    unsigned NumVisited = BC.getNumVisited(BlockID);
        
    if (NumVisited < 1) builder.generateNode(PrevState, true);
    else builder.markInfeasible(true);

    return;
  }
  
  RVal V = GetRVal(PrevState, Condition);
  
  switch (V.getBaseKind()) {
    default:
      break;

    case RVal::UnknownKind:
      builder.generateNode(PrevState, true);
      builder.generateNode(PrevState, false);
      return;
      
    case RVal::UninitializedKind: {      
      NodeTy* N = builder.generateNode(PrevState, true);

      if (N) {
        N->markAsSink();
        UninitBranches.insert(N);
      }
      
      builder.markInfeasible(false);
      return;
    }      
  }
  
  // Get the current block counter.
  GRBlockCounter BC = builder.getBlockCounter();
  unsigned BlockID = builder.getTargetBlock(true)->getBlockID();
  unsigned NumVisited = BC.getNumVisited(BlockID);
  
  if (isa<nonlval::ConcreteInt>(V) || 
      BC.getNumVisited(builder.getTargetBlock(true)->getBlockID()) < 1) {
    
    // Process the true branch.

    bool isFeasible = true;
    
    StateTy St = Assume(PrevState, V, true, isFeasible);

    if (isFeasible)
      builder.generateNode(St, true);
    else
      builder.markInfeasible(true);
  }
  else
    builder.markInfeasible(true);
  
  BlockID = builder.getTargetBlock(false)->getBlockID();
  NumVisited = BC.getNumVisited(BlockID);
  
  if (isa<nonlval::ConcreteInt>(V) || 
      BC.getNumVisited(builder.getTargetBlock(false)->getBlockID()) < 1) {
    
    // Process the false branch.  
    
    bool isFeasible = false;
    
    StateTy St = Assume(PrevState, V, false, isFeasible);
    
    if (isFeasible)
      builder.generateNode(St, false);
    else
      builder.markInfeasible(false);
  }
  else
    builder.markInfeasible(false);
}

/// ProcessIndirectGoto - Called by GRCoreEngine.  Used to generate successor
///  nodes by processing the 'effects' of a computed goto jump.
void GRExprEngine::ProcessIndirectGoto(IndirectGotoNodeBuilder& builder) {

  StateTy St = builder.getState();  
  RVal V = GetRVal(St, builder.getTarget());
  
  // Three possibilities:
  //
  //   (1) We know the computed label.
  //   (2) The label is NULL (or some other constant), or Uninitialized.
  //   (3) We have no clue about the label.  Dispatch to all targets.
  //
  
  typedef IndirectGotoNodeBuilder::iterator iterator;

  if (isa<lval::GotoLabel>(V)) {
    LabelStmt* L = cast<lval::GotoLabel>(V).getLabel();
    
    for (iterator I=builder.begin(), E=builder.end(); I != E; ++I) {
      if (I.getLabel() == L) {
        builder.generateNode(I, St);
        return;
      }
    }
    
    assert (false && "No block with label.");
    return;
  }

  if (isa<lval::ConcreteInt>(V) || isa<UninitializedVal>(V)) {
    // Dispatch to the first target and mark it as a sink.
    NodeTy* N = builder.generateNode(builder.begin(), St, true);
    UninitBranches.insert(N);
    return;
  }
  
  // This is really a catch-all.  We don't support symbolics yet.
  
  assert (V.isUnknown());
  
  for (iterator I=builder.begin(), E=builder.end(); I != E; ++I)
    builder.generateNode(I, St);
}

/// ProcessSwitch - Called by GRCoreEngine.  Used to generate successor
///  nodes by processing the 'effects' of a switch statement.
void GRExprEngine::ProcessSwitch(SwitchNodeBuilder& builder) {
  
  typedef SwitchNodeBuilder::iterator iterator;
  
  StateTy St = builder.getState();  
  Expr* CondE = builder.getCondition();
  RVal  CondV = GetRVal(St, CondE);

  if (CondV.isUninit()) {
    NodeTy* N = builder.generateDefaultCaseNode(St, true);
    UninitBranches.insert(N);
    return;
  }
  
  StateTy  DefaultSt = St;
  
  // While most of this can be assumed (such as the signedness), having it
  // just computed makes sure everything makes the same assumptions end-to-end.
  
  unsigned bits = getContext().getTypeSize(CondE->getType(),
                                           CondE->getExprLoc());

  APSInt V1(bits, false);
  APSInt V2 = V1;
  
  for (iterator I = builder.begin(), EI = builder.end(); I != EI; ++I) {

    CaseStmt* Case = cast<CaseStmt>(I.getCase());
    
    // Evaluate the case.
    if (!Case->getLHS()->isIntegerConstantExpr(V1, getContext(), 0, true)) {
      assert (false && "Case condition must evaluate to an integer constant.");
      return;
    }
    
    // Get the RHS of the case, if it exists.
    
    if (Expr* E = Case->getRHS()) {
      if (!E->isIntegerConstantExpr(V2, getContext(), 0, true)) {
        assert (false &&
                "Case condition (RHS) must evaluate to an integer constant.");
        return ;
      }
      
      assert (V1 <= V2);
    }
    else V2 = V1;
    
    // FIXME: Eventually we should replace the logic below with a range
    //  comparison, rather than concretize the values within the range.
    //  This should be easy once we have "ranges" for NonLVals.
        
    do {      
      nonlval::ConcreteInt CaseVal(ValMgr.getValue(V1));
      
      RVal Res = EvalBinOp(BinaryOperator::EQ, CondV, CaseVal);
      
      // Now "assume" that the case matches.
      bool isFeasible = false;
      
      StateTy StNew = Assume(St, Res, true, isFeasible);
      
      if (isFeasible) {
        builder.generateCaseStmtNode(I, StNew);
       
        // If CondV evaluates to a constant, then we know that this
        // is the *only* case that we can take, so stop evaluating the
        // others.
        if (isa<nonlval::ConcreteInt>(CondV))
          return;
      }
      
      // Now "assume" that the case doesn't match.  Add this state
      // to the default state (if it is feasible).
      
      StNew = Assume(DefaultSt, Res, false, isFeasible);
      
      if (isFeasible)
        DefaultSt = StNew;

      // Concretize the next value in the range.      
      ++V1;
      
    } while (V1 < V2);
  }
  
  // If we reach here, than we know that the default branch is
  // possible.  
  builder.generateDefaultCaseNode(DefaultSt);
}


void GRExprEngine::VisitLogicalExpr(BinaryOperator* B, NodeTy* Pred,
                                    NodeSet& Dst) {

  bool hasR2;
  StateTy PrevState = Pred->getState();

  RVal R1 = GetRVal(PrevState, B->getLHS());
  RVal R2 = GetRVal(PrevState, B->getRHS(), hasR2);
  
  if (hasR2) {
    if (R2.isUnknownOrUninit()) {
      Nodify(Dst, B, Pred, SetRVal(PrevState, B, R2));
      return;
    }
  }
  else if (R1.isUnknownOrUninit()) {
    Nodify(Dst, B, Pred, SetRVal(PrevState, B, R1));
    return;
  }

  // R1 is an expression that can evaluate to either 'true' or 'false'.
  if (B->getOpcode() == BinaryOperator::LAnd) {
    // hasR2 == 'false' means that LHS evaluated to 'false' and that
    // we short-circuited, leading to a value of '0' for the '&&' expression.
    if (hasR2 == false) { 
      Nodify(Dst, B, Pred, SetRVal(PrevState, B, MakeConstantVal(0U, B)));
      return;
    }
  }
  else {
    assert (B->getOpcode() == BinaryOperator::LOr);
    // hasR2 == 'false' means that the LHS evaluate to 'true' and that
    //  we short-circuited, leading to a value of '1' for the '||' expression.
    if (hasR2 == false) {
      Nodify(Dst, B, Pred, SetRVal(PrevState, B, MakeConstantVal(1U, B)));
      return;      
    }
  }
    
  // If we reach here we did not short-circuit.  Assume R2 == true and
  // R2 == false.
    
  bool isFeasible;
  StateTy St = Assume(PrevState, R2, true, isFeasible);
  
  if (isFeasible)
    Nodify(Dst, B, Pred, SetRVal(PrevState, B, MakeConstantVal(1U, B)));

  St = Assume(PrevState, R2, false, isFeasible);
  
  if (isFeasible)
    Nodify(Dst, B, Pred, SetRVal(PrevState, B, MakeConstantVal(0U, B)));  
}



void GRExprEngine::ProcessStmt(Stmt* S, StmtNodeBuilder& builder) {

  Builder = &builder;
  StmtEntryNode = builder.getLastNode();
  CurrentStmt = S;
  NodeSet Dst;
  StateCleaned = false;

  Visit(S, StmtEntryNode, Dst);

  // If no nodes were generated, generate a new node that has all the
  // dead mappings removed.
  
  if (Dst.size() == 1 && *Dst.begin() == StmtEntryNode) {
    StateTy St = RemoveDeadBindings(S, StmtEntryNode->getState());
    builder.generateNode(S, St, StmtEntryNode);
  }
  
  // For safety, NULL out these variables.
  
  CurrentStmt = NULL;
  StmtEntryNode = NULL;
  Builder = NULL;
}

GRExprEngine::NodeTy*
GRExprEngine::Nodify(NodeSet& Dst, Stmt* S, NodeTy* Pred, StateTy St) {
 
  // If the state hasn't changed, don't generate a new node.
  if (St == Pred->getState())
    return NULL;
  
  NodeTy* N = Builder->generateNode(S, St, Pred);
  Dst.Add(N);
  
  return N;
}

void GRExprEngine::Nodify(NodeSet& Dst, Stmt* S, NodeTy* Pred,
                         const StateTy::BufferTy& SB) {
  
  for (StateTy::BufferTy::const_iterator I=SB.begin(), E=SB.end(); I!=E; ++I)
    Nodify(Dst, S, Pred, *I);
}

void GRExprEngine::VisitDeclRefExpr(DeclRefExpr* D, NodeTy* Pred, NodeSet& Dst){

  if (D != CurrentStmt) {
    Dst.Add(Pred); // No-op. Simply propagate the current state unchanged.
    return;
  }
  
  // If we are here, we are loading the value of the decl and binding
  // it to the block-level expression.
  
  StateTy St = Pred->getState();  
  Nodify(Dst, D, Pred, SetRVal(St, D, GetRVal(St, D)));
}

void GRExprEngine::VisitCall(CallExpr* CE, NodeTy* Pred,
                             CallExpr::arg_iterator AI,
                             CallExpr::arg_iterator AE,
                             NodeSet& Dst) {
  
  // Process the arguments.
  
  if (AI != AE) {
    
    NodeSet DstTmp;  
    Visit(*AI, Pred, DstTmp);
    ++AI;
    
    for (NodeSet::iterator DI=DstTmp.begin(), DE=DstTmp.end(); DI != DE; ++DI)
      VisitCall(CE, *DI, AI, AE, Dst);
    
    return;
  }

  // If we reach here we have processed all of the arguments.  Evaluate
  // the callee expression.
  NodeSet DstTmp;  
  Visit(CE->getCallee(), Pred, DstTmp);
  
  // Finally, evaluate the function call.
  for (NodeSet::iterator DI = DstTmp.begin(), DE = DstTmp.end(); DI!=DE; ++DI) {

    StateTy St = (*DI)->getState();    
    RVal L = GetLVal(St, CE->getCallee());

    // Check for uninitialized control-flow.

    if (L.isUninit()) {
      
      NodeTy* N = Builder->generateNode(CE, St, *DI);
      N->markAsSink();
      UninitBranches.insert(N);
      continue;
    }
    
    // FIXME: EvalCall must handle the case where the callee is Unknown.
    assert (!L.isUnknown());    

    Nodify(Dst, CE, *DI, EvalCall(CE, cast<LVal>(L), (*DI)->getState()));
  }
}

void GRExprEngine::VisitCast(Expr* CastE, Expr* Ex, NodeTy* Pred, NodeSet& Dst){
  
  NodeSet S1;
  Visit(Ex, Pred, S1);

  QualType T = CastE->getType();
  
  // Check for redundant casts or casting to "void"
  if (T->isVoidType() ||
      Ex->getType() == T || 
      (T->isPointerType() && Ex->getType()->isFunctionType())) {
    
    for (NodeSet::iterator I1 = S1.begin(), E1 = S1.end(); I1 != E1; ++I1)
      Dst.Add(*I1);

    return;
  }
  
  for (NodeSet::iterator I1 = S1.begin(), E1 = S1.end(); I1 != E1; ++I1) {
    NodeTy* N = *I1;
    StateTy St = N->getState();
    RVal V = GetRVal(St, Ex);
    Nodify(Dst, CastE, N, SetRVal(St, CastE, EvalCast(V, CastE->getType())));
  }
}

void GRExprEngine::VisitDeclStmt(DeclStmt* DS, GRExprEngine::NodeTy* Pred,
                                 GRExprEngine::NodeSet& Dst) {
  
  StateTy St = Pred->getState();
  
  for (const ScopedDecl* D = DS->getDecl(); D; D = D->getNextDeclarator())
    if (const VarDecl* VD = dyn_cast<VarDecl>(D)) {
      
      // FIXME: Add support for local arrays.
      if (VD->getType()->isArrayType())
        continue;
      
      const Expr* Ex = VD->getInit(); 
      
      St = SetRVal(St, lval::DeclVal(VD),
                   Ex ? GetRVal(St, Ex) : UninitializedVal());
    }

  Nodify(Dst, DS, Pred, St);
  
  if (Dst.empty()) { Dst.Add(Pred); }
}


void GRExprEngine::VisitGuardedExpr(Expr* Ex, Expr* L, Expr* R,
                                   NodeTy* Pred, NodeSet& Dst) {
  
  StateTy St = Pred->getState();
  
  RVal V = GetRVal(St, L);
  if (isa<UnknownVal>(V)) V = GetRVal(St, R);
  
  Nodify(Dst, Ex, Pred, SetRVal(St, Ex, V));
}

/// VisitSizeOfAlignOfTypeExpr - Transfer function for sizeof(type).
void GRExprEngine::VisitSizeOfAlignOfTypeExpr(SizeOfAlignOfTypeExpr* Ex,
                                              NodeTy* Pred,
                                              NodeSet& Dst) {

  assert (Ex->isSizeOf() && "FIXME: AlignOf(Expr) not yet implemented.");
  
  // 6.5.3.4 sizeof: "The result type is an integer."
  
  QualType T = Ex->getArgumentType();


  // FIXME: Add support for VLAs.
  if (!T.getTypePtr()->isConstantSizeType())
    return;
  
  
  uint64_t size = 1;  // Handle sizeof(void)
  
  if (T != getContext().VoidTy) {
    SourceLocation Loc = Ex->getExprLoc();
    size = getContext().getTypeSize(T, Loc) / 8;
  }
  
  Nodify(Dst, Ex, Pred,
         SetRVal(Pred->getState(), Ex,
                  NonLVal::MakeVal(ValMgr, size, Ex->getType())));
  
}

void GRExprEngine::VisitDeref(UnaryOperator* U, NodeTy* Pred, NodeSet& Dst) {

  Expr* Ex = U->getSubExpr()->IgnoreParens();
    
  NodeSet DstTmp;
  
  if (!isa<DeclRefExpr>(Ex))
    DstTmp.Add(Pred);
  else
    Visit(Ex, Pred, DstTmp);
  
  for (NodeSet::iterator I = DstTmp.begin(), DE = DstTmp.end(); I != DE; ++I) {

    NodeTy* N = *I;
    StateTy St = N->getState();
    
    // FIXME: Bifurcate when dereferencing a symbolic with no constraints?
    
    RVal V = GetRVal(St, Ex);
    
    // Check for dereferences of uninitialized values.
    
    if (V.isUninit()) {
      
      NodeTy* Succ = Builder->generateNode(U, St, N);
      
      if (Succ) {
        Succ->markAsSink();
        UninitDeref.insert(Succ);
      }
      
      continue;
    }
    
    // Check for dereferences of unknown values.  Treat as No-Ops.
    
    if (V.isUnknown()) {
      Dst.Add(N);
      continue;
    }
    
    // After a dereference, one of two possible situations arise:
    //  (1) A crash, because the pointer was NULL.
    //  (2) The pointer is not NULL, and the dereference works.
    // 
    // We add these assumptions.
    
    LVal LV = cast<LVal>(V);    
    bool isFeasibleNotNull;
    
    // "Assume" that the pointer is Not-NULL.
    
    StateTy StNotNull = Assume(St, LV, true, isFeasibleNotNull);
    
    if (isFeasibleNotNull) {
      
      // FIXME: Currently symbolic analysis "generates" new symbols
      //  for the contents of values.  We need a better approach.
      
      Nodify(Dst, U, N, SetRVal(StNotNull, U,
                                GetRVal(StNotNull, LV, U->getType())));
    }
    
    bool isFeasibleNull;
    
    // Now "assume" that the pointer is NULL.
    
    StateTy StNull = Assume(St, LV, false, isFeasibleNull);
    
    if (isFeasibleNull) {
      
      // We don't use "Nodify" here because the node will be a sink
      // and we have no intention of processing it later.

      NodeTy* NullNode = Builder->generateNode(U, StNull, N);
      
      if (NullNode) {

        NullNode->markAsSink();
        
        if (isFeasibleNotNull) ImplicitNullDeref.insert(NullNode);
        else ExplicitNullDeref.insert(NullNode);
      }
    }    
  }
}

void GRExprEngine::VisitUnaryOperator(UnaryOperator* U, NodeTy* Pred,
                                      NodeSet& Dst) {
  
  NodeSet S1;
  
  assert (U->getOpcode() != UnaryOperator::Deref);
  assert (U->getOpcode() != UnaryOperator::SizeOf);
  assert (U->getOpcode() != UnaryOperator::AlignOf);
  
  bool use_GetLVal = false;
  
  switch (U->getOpcode()) {
    case UnaryOperator::PostInc:
    case UnaryOperator::PostDec:
    case UnaryOperator::PreInc:
    case UnaryOperator::PreDec:
    case UnaryOperator::AddrOf:
      // Evalue subexpression as an LVal.
      use_GetLVal = true;
      VisitLVal(U->getSubExpr(), Pred, S1);
      break;
      
    default:
      Visit(U->getSubExpr(), Pred, S1);
      break;
  }

  for (NodeSet::iterator I1 = S1.begin(), E1 = S1.end(); I1 != E1; ++I1) {

    NodeTy* N1 = *I1;
    StateTy St = N1->getState();
        
    RVal SubV = use_GetLVal ? GetLVal(St, U->getSubExpr()) : 
                              GetRVal(St, U->getSubExpr());
    
    if (SubV.isUnknown()) {
      Dst.Add(N1);
      continue;
    }

    if (SubV.isUninit()) {
      Nodify(Dst, U, N1, SetRVal(St, U, SubV));
      continue;
    }
    
    if (U->isIncrementDecrementOp()) {
      
      // Handle ++ and -- (both pre- and post-increment).
      
      LVal SubLV = cast<LVal>(SubV); 
      RVal V  = GetRVal(St, SubLV, U->getType());
      
      // An LVal should never bind to UnknownVal.      
      assert (!V.isUnknown());

      // Propagate uninitialized values.      
      if (V.isUninit()) {
        Nodify(Dst, U, N1, SetRVal(St, U, V));
        continue;
      }
      
      // Handle all NonLVals.
      
      BinaryOperator::Opcode Op = U->isIncrementOp() ? BinaryOperator::Add
                                                     : BinaryOperator::Sub;
      
      RVal Result = EvalBinOp(Op, cast<NonLVal>(V), MakeConstantVal(1U, U));
      
      if (U->isPostfix())
        St = SetRVal(SetRVal(St, U, V), SubLV, Result);
      else
        St = SetRVal(SetRVal(St, U, Result), SubLV, Result);
        
      Nodify(Dst, U, N1, St);
      continue;
    }    
    
    // Handle all other unary operators.
    
    switch (U->getOpcode()) {

      case UnaryOperator::Minus:
        St = SetRVal(St, U, EvalMinus(U, cast<NonLVal>(SubV)));
        break;
        
      case UnaryOperator::Not:
        St = SetRVal(St, U, EvalComplement(cast<NonLVal>(SubV)));
        break;
        
      case UnaryOperator::LNot:   
        
        // C99 6.5.3.3: "The expression !E is equivalent to (0==E)."
        //
        //  Note: technically we do "E == 0", but this is the same in the
        //    transfer functions as "0 == E".

        if (isa<LVal>(SubV)) {
          lval::ConcreteInt V(ValMgr.getZeroWithPtrWidth());
          RVal Result = EvalBinOp(BinaryOperator::EQ, cast<LVal>(SubV), V);
          St = SetRVal(St, U, Result);
        }
        else {
          nonlval::ConcreteInt V(ValMgr.getZeroWithPtrWidth());
          RVal Result = EvalBinOp(BinaryOperator::EQ, cast<NonLVal>(SubV), V);
          St = SetRVal(St, U, Result);
        }
        
        break;
        
      case UnaryOperator::AddrOf: {
        assert (isa<LVal>(SubV));
        St = SetRVal(St, U, SubV);
        break;
      }
                
      default: ;
        assert (false && "Not implemented.");
    } 
    
    Nodify(Dst, U, N1, St);
  }
}

void GRExprEngine::VisitSizeOfExpr(UnaryOperator* U, NodeTy* Pred,
                                   NodeSet& Dst) {
  
  QualType T = U->getSubExpr()->getType();
  
  // FIXME: Add support for VLAs.
  if (!T.getTypePtr()->isConstantSizeType())
    return;
  
  SourceLocation Loc = U->getExprLoc();
  uint64_t size = getContext().getTypeSize(T, Loc) / 8;                
  StateTy St = Pred->getState();
  St = SetRVal(St, U, NonLVal::MakeVal(ValMgr, size, U->getType(), Loc));

  Nodify(Dst, U, Pred, St);
}

void GRExprEngine::VisitLVal(Expr* Ex, NodeTy* Pred, NodeSet& Dst) {
  
  assert (Ex != CurrentStmt && !getCFG().isBlkExpr(Ex));
  
  Ex = Ex->IgnoreParens();
  
  if (isa<DeclRefExpr>(Ex)) {
    Dst.Add(Pred);
    return;
  }
  
  if (UnaryOperator* U = dyn_cast<UnaryOperator>(Ex)) {
    if (U->getOpcode() == UnaryOperator::Deref) {
      Ex = U->getSubExpr()->IgnoreParens();
      
      if (isa<DeclRefExpr>(Ex))
        Dst.Add(Pred);
      else
        Visit(Ex, Pred, Dst);
      
      return;
    }
  }
  
  Visit(Ex, Pred, Dst);
}

void GRExprEngine::VisitBinaryOperator(BinaryOperator* B,
                                       GRExprEngine::NodeTy* Pred,
                                       GRExprEngine::NodeSet& Dst) {
  NodeSet S1;
  
  if (B->isAssignmentOp())
    VisitLVal(B->getLHS(), Pred, S1);
  else
    Visit(B->getLHS(), Pred, S1);

  for (NodeSet::iterator I1=S1.begin(), E1=S1.end(); I1 != E1; ++I1) {

    NodeTy* N1 = *I1;
    
    // When getting the value for the LHS, check if we are in an assignment.
    // In such cases, we want to (initially) treat the LHS as an LVal,
    // so we use GetLVal instead of GetRVal so that DeclRefExpr's are
    // evaluated to LValDecl's instead of to an NonLVal.

    RVal LeftV = B->isAssignmentOp() ? GetLVal(N1->getState(), B->getLHS())
                                     : GetRVal(N1->getState(), B->getLHS());
    
    // Visit the RHS...
    
    NodeSet S2;    
    Visit(B->getRHS(), N1, S2);
    
    // Process the binary operator.
  
    for (NodeSet::iterator I2 = S2.begin(), E2 = S2.end(); I2 != E2; ++I2) {

      NodeTy* N2 = *I2;
      StateTy St = N2->getState();      
      RVal RightV = GetRVal(St, B->getRHS());

      BinaryOperator::Opcode Op = B->getOpcode();
      
      if (Op <= BinaryOperator::Or) {
        
        // Process non-assignements except commas or short-circuited
        // logical expressions (LAnd and LOr).
        
        RVal Result = EvalBinOp(Op, LeftV, RightV);
        
        if (Result.isUnknown()) {
          Dst.Add(N2);
          continue;
        }
        
        Nodify(Dst, B, N2, SetRVal(St, B, Result));
        continue;
      }
        
      // Process assignments.
    
      switch (Op) {        
          
        case BinaryOperator::Assign: {
          
          // Simple assignments.

          if (LeftV.isUninit()) {
            HandleUninitializedStore(B, N2);
            continue;
          }
          
          if (LeftV.isUnknown()) {
            St = SetRVal(St, B, RightV);
            break;
          }

          St = SetRVal(SetRVal(St, B, RightV), cast<LVal>(LeftV), RightV);
          break;
        }

          // Compound assignment operators.
          
        default: { 
          
          assert (B->isCompoundAssignmentOp());                                    
          
          if (LeftV.isUninit()) {
            HandleUninitializedStore(B, N2);
            continue;
          }
          
          if (LeftV.isUnknown()) {
            
            // While we do not know the location to store RightV,
            // the entire expression does evaluate to RightV.
            
            if (RightV.isUnknown()) {
              Dst.Add(N2);
              continue;
            }
            
            St = SetRVal(St, B, RightV);
            break;
          }
          
          // At this pointer we know that the LHS evaluates to an LVal
          // that is neither "Unknown" or "Unintialized."
          
          LVal LeftLV = cast<LVal>(LeftV);
          
          // Propagate uninitialized values (right-side).
          
          if (RightV.isUninit()) {
            St = SetRVal(SetRVal(St, B, RightV), LeftLV, RightV);
            break;
          }
          
          // Fetch the value of the LHS (the value of the variable, etc.).
          
          RVal V = GetRVal(N1->getState(), LeftLV, B->getLHS()->getType());
          
          // Propagate uninitialized value (left-side).
          
          if (V.isUninit()) {
            St = SetRVal(St, B, V);
            break;
          }
          
          // Propagate unknown values.
          
          assert (!V.isUnknown() &&
                  "An LVal should never bind to UnknownVal");
          
          if (RightV.isUnknown()) {
            St = SetRVal(SetRVal(St, LeftLV, RightV), B, RightV);
            break;
          }
            
          // Neither the LHS or the RHS have Unknown/Uninit values.  Process
          // the operation and store the result.
          
          if (Op >= BinaryOperator::AndAssign)
            ((int&) Op) -= (BinaryOperator::AndAssign - BinaryOperator::And);
          else
            ((int&) Op) -= BinaryOperator::MulAssign;          
          
          // Get the computation type.
          QualType CTy = cast<CompoundAssignOperator>(B)->getComputationType();
          
          // Perform promotions.
          V = EvalCast(V, CTy);
          RightV = EvalCast(RightV, CTy);
          
          // Evaluate operands and promote to result type.
          RVal Result = EvalCast(EvalBinOp(Op, V, RightV), B->getType());
          
          St = SetRVal(SetRVal(St, B, Result), LeftLV, Result);
        }
      }
    
      Nodify(Dst, B, N2, St);
    }
  }
}

void GRExprEngine::HandleUninitializedStore(Stmt* S, NodeTy* Pred) {  
  NodeTy* N = Builder->generateNode(S, Pred->getState(), Pred);
  N->markAsSink();
  UninitStores.insert(N);
}

void GRExprEngine::Visit(Stmt* S, NodeTy* Pred, NodeSet& Dst) {

  // FIXME: add metadata to the CFG so that we can disable
  //  this check when we KNOW that there is no block-level subexpression.
  //  The motivation is that this check requires a hashtable lookup.

  if (S != CurrentStmt && getCFG().isBlkExpr(S)) {
    Dst.Add(Pred);
    return;
  }

  switch (S->getStmtClass()) {
      
    default:
      // Cases we intentionally have "default" handle:
      //   AddrLabelExpr
      
      Dst.Add(Pred); // No-op. Simply propagate the current state unchanged.
      break;
                                                       
    case Stmt::BinaryOperatorClass: {
      BinaryOperator* B = cast<BinaryOperator>(S);
 
      if (B->isLogicalOp()) {
        VisitLogicalExpr(B, Pred, Dst);
        break;
      }
      else if (B->getOpcode() == BinaryOperator::Comma) {
        StateTy St = Pred->getState();
        Nodify(Dst, B, Pred, SetRVal(St, B, GetRVal(St, B->getRHS())));
        break;
      }
      
      VisitBinaryOperator(cast<BinaryOperator>(S), Pred, Dst);
      break;
    }
      
    case Stmt::CallExprClass: {
      CallExpr* C = cast<CallExpr>(S);
      VisitCall(C, Pred, C->arg_begin(), C->arg_end(), Dst);
      break;      
    }

    case Stmt::CastExprClass: {
      CastExpr* C = cast<CastExpr>(S);
      VisitCast(C, C->getSubExpr(), Pred, Dst);
      break;
    }
      
      // While explicitly creating a node+state for visiting a CharacterLiteral
      // seems wasteful, it also solves a bunch of problems when handling
      // the ?, &&, and ||.
      
    case Stmt::CharacterLiteralClass: {
      CharacterLiteral* C = cast<CharacterLiteral>(S);
      StateTy St = Pred->getState();
      NonLVal X = NonLVal::MakeVal(ValMgr, C->getValue(), C->getType(),
                                        C->getLoc());
      Nodify(Dst, C, Pred, SetRVal(St, C, X));
      break;      
    }
      
      // FIXME: ChooseExpr is really a constant.  We need to fix
      //        the CFG do not model them as explicit control-flow.
      
    case Stmt::ChooseExprClass: { // __builtin_choose_expr
      ChooseExpr* C = cast<ChooseExpr>(S);
      VisitGuardedExpr(C, C->getLHS(), C->getRHS(), Pred, Dst);
      break;
    }
      
    case Stmt::CompoundAssignOperatorClass:
      VisitBinaryOperator(cast<BinaryOperator>(S), Pred, Dst);
      break;
      
    case Stmt::ConditionalOperatorClass: { // '?' operator
      ConditionalOperator* C = cast<ConditionalOperator>(S);
      VisitGuardedExpr(C, C->getLHS(), C->getRHS(), Pred, Dst);
      break;
    }
      
    case Stmt::DeclRefExprClass:
      VisitDeclRefExpr(cast<DeclRefExpr>(S), Pred, Dst);
      break;
      
    case Stmt::DeclStmtClass:
      VisitDeclStmt(cast<DeclStmt>(S), Pred, Dst);
      break;
      
      // While explicitly creating a node+state for visiting an IntegerLiteral
      // seems wasteful, it also solves a bunch of problems when handling
      // the ?, &&, and ||.
      
    case Stmt::IntegerLiteralClass: {      
      StateTy St = Pred->getState();
      IntegerLiteral* I = cast<IntegerLiteral>(S);
      NonLVal X = NonLVal::MakeVal(ValMgr, I);
      Nodify(Dst, I, Pred, SetRVal(St, I, X));
      break;      
    }
      
    case Stmt::ImplicitCastExprClass: {
      ImplicitCastExpr* C = cast<ImplicitCastExpr>(S);
      VisitCast(C, C->getSubExpr(), Pred, Dst);
      break;
    }

    case Stmt::ParenExprClass:
      Visit(cast<ParenExpr>(S)->getSubExpr(), Pred, Dst);
      break;
      
    case Stmt::SizeOfAlignOfTypeExprClass:
      VisitSizeOfAlignOfTypeExpr(cast<SizeOfAlignOfTypeExpr>(S), Pred, Dst);
      break;
      
    case Stmt::StmtExprClass: {
      StmtExpr* SE = cast<StmtExpr>(S);
      
      StateTy St = Pred->getState();
      Expr* LastExpr = cast<Expr>(*SE->getSubStmt()->body_rbegin());
      Nodify(Dst, SE, Pred, SetRVal(St, SE, GetRVal(St, LastExpr)));
      break;      
    }
      
      // FIXME: We may wish to always bind state to ReturnStmts so
      //  that users can quickly query what was the state at the
      //  exit points of a function.
      
    case Stmt::ReturnStmtClass: {
      if (Expr* R = cast<ReturnStmt>(S)->getRetValue())
        Visit(R, Pred, Dst);
      else
        Dst.Add(Pred);
      
      break;
    }
      
    case Stmt::UnaryOperatorClass: {
      UnaryOperator* U = cast<UnaryOperator>(S);
      
      switch (U->getOpcode()) {
        case UnaryOperator::Deref: VisitDeref(U, Pred, Dst); break;
        case UnaryOperator::Plus:  Visit(U->getSubExpr(), Pred, Dst); break;
        case UnaryOperator::SizeOf: VisitSizeOfExpr(U, Pred, Dst); break;
        default: VisitUnaryOperator(U, Pred, Dst); break;
      }
      
      break;
    }
  }
}

//===----------------------------------------------------------------------===//
// "Assume" logic.
//===----------------------------------------------------------------------===//

GRExprEngine::StateTy GRExprEngine::Assume(StateTy St, LVal Cond,
                                           bool Assumption, 
                                           bool& isFeasible) {
  switch (Cond.getSubKind()) {
    default:
      assert (false && "'Assume' not implemented for this LVal.");
      return St;
      
    case lval::SymbolValKind:
      if (Assumption)
        return AssumeSymNE(St, cast<lval::SymbolVal>(Cond).getSymbol(),
                           ValMgr.getZeroWithPtrWidth(), isFeasible);
      else
        return AssumeSymEQ(St, cast<lval::SymbolVal>(Cond).getSymbol(),
                           ValMgr.getZeroWithPtrWidth(), isFeasible);
      
      
    case lval::DeclValKind:
      isFeasible = Assumption;
      return St;

    case lval::ConcreteIntKind: {
      bool b = cast<lval::ConcreteInt>(Cond).getValue() != 0;
      isFeasible = b ? Assumption : !Assumption;      
      return St;
    }
  }
}

GRExprEngine::StateTy GRExprEngine::Assume(StateTy St, NonLVal Cond,
                                         bool Assumption, 
                                         bool& isFeasible) {  
  switch (Cond.getSubKind()) {
    default:
      assert (false && "'Assume' not implemented for this NonLVal.");
      return St;
      
      
    case nonlval::SymbolValKind: {
      nonlval::SymbolVal& SV = cast<nonlval::SymbolVal>(Cond);
      SymbolID sym = SV.getSymbol();
      
      if (Assumption)
        return AssumeSymNE(St, sym, ValMgr.getValue(0, SymMgr.getType(sym)),
                           isFeasible);
      else
        return AssumeSymEQ(St, sym, ValMgr.getValue(0, SymMgr.getType(sym)),
                           isFeasible);
    }
      
    case nonlval::SymIntConstraintValKind:
      return
        AssumeSymInt(St, Assumption,
                     cast<nonlval::SymIntConstraintVal>(Cond).getConstraint(),
                     isFeasible);
      
    case nonlval::ConcreteIntKind: {
      bool b = cast<nonlval::ConcreteInt>(Cond).getValue() != 0;
      isFeasible = b ? Assumption : !Assumption;      
      return St;
    }
  }
}

GRExprEngine::StateTy
GRExprEngine::AssumeSymNE(StateTy St, SymbolID sym,
                         const llvm::APSInt& V, bool& isFeasible) {
  
  // First, determine if sym == X, where X != V.
  if (const llvm::APSInt* X = St.getSymVal(sym)) {
    isFeasible = *X != V;
    return St;
  }
  
  // Second, determine if sym != V.
  if (St.isNotEqual(sym, V)) {
    isFeasible = true;
    return St;
  }
      
  // If we reach here, sym is not a constant and we don't know if it is != V.
  // Make that assumption.
  
  isFeasible = true;
  return StateMgr.AddNE(St, sym, V);
}

GRExprEngine::StateTy
GRExprEngine::AssumeSymEQ(StateTy St, SymbolID sym,
                         const llvm::APSInt& V, bool& isFeasible) {
  
  // First, determine if sym == X, where X != V.
  if (const llvm::APSInt* X = St.getSymVal(sym)) {
    isFeasible = *X == V;
    return St;
  }
  
  // Second, determine if sym != V.
  if (St.isNotEqual(sym, V)) {
    isFeasible = false;
    return St;
  }
  
  // If we reach here, sym is not a constant and we don't know if it is == V.
  // Make that assumption.
  
  isFeasible = true;
  return StateMgr.AddEQ(St, sym, V);
}

GRExprEngine::StateTy
GRExprEngine::AssumeSymInt(StateTy St, bool Assumption,
                          const SymIntConstraint& C, bool& isFeasible) {
  
  switch (C.getOpcode()) {
    default:
      // No logic yet for other operators.
      return St;
      
    case BinaryOperator::EQ:
      if (Assumption)
        return AssumeSymEQ(St, C.getSymbol(), C.getInt(), isFeasible);
      else
        return AssumeSymNE(St, C.getSymbol(), C.getInt(), isFeasible);
      
    case BinaryOperator::NE:
      if (Assumption)
        return AssumeSymNE(St, C.getSymbol(), C.getInt(), isFeasible);
      else
        return AssumeSymEQ(St, C.getSymbol(), C.getInt(), isFeasible);
  }
}

//===----------------------------------------------------------------------===//
// Visualization.
//===----------------------------------------------------------------------===//

#ifndef NDEBUG
static GRExprEngine* GraphPrintCheckerState;

namespace llvm {
template<>
struct VISIBILITY_HIDDEN DOTGraphTraits<GRExprEngine::NodeTy*> :
  public DefaultDOTGraphTraits {
    
  static void PrintVarBindings(std::ostream& Out, GRExprEngine::StateTy St) {

    Out << "Variables:\\l";
    
    bool isFirst = true;
    
    for (GRExprEngine::StateTy::vb_iterator I=St.vb_begin(),
                                           E=St.vb_end(); I!=E;++I) {        

      if (isFirst)
        isFirst = false;
      else
        Out << "\\l";
      
      Out << ' ' << I.getKey()->getName() << " : ";
      I.getData().print(Out);
    }
    
  }
    
    
  static void PrintSubExprBindings(std::ostream& Out, GRExprEngine::StateTy St){
    
    bool isFirst = true;
    
    for (GRExprEngine::StateTy::seb_iterator I=St.seb_begin(), E=St.seb_end();
                                            I != E;++I) {        
      
      if (isFirst) {
        Out << "\\l\\lSub-Expressions:\\l";
        isFirst = false;
      }
      else
        Out << "\\l";
      
      Out << " (" << (void*) I.getKey() << ") ";
      I.getKey()->printPretty(Out);
      Out << " : ";
      I.getData().print(Out);
    }
  }
    
  static void PrintBlkExprBindings(std::ostream& Out, GRExprEngine::StateTy St){
        
    bool isFirst = true;

    for (GRExprEngine::StateTy::beb_iterator I=St.beb_begin(), E=St.beb_end();
                                            I != E; ++I) {      
      if (isFirst) {
        Out << "\\l\\lBlock-level Expressions:\\l";
        isFirst = false;
      }
      else
        Out << "\\l";

      Out << " (" << (void*) I.getKey() << ") ";
      I.getKey()->printPretty(Out);
      Out << " : ";
      I.getData().print(Out);
    }
  }
    
  static void PrintEQ(std::ostream& Out, GRExprEngine::StateTy St) {
    ValueState::ConstEqTy CE = St.getImpl()->ConstEq;
    
    if (CE.isEmpty())
      return;
    
    Out << "\\l\\|'==' constraints:";

    for (ValueState::ConstEqTy::iterator I=CE.begin(), E=CE.end(); I!=E;++I)
      Out << "\\l $" << I.getKey() << " : " << I.getData()->toString();
  }
    
  static void PrintNE(std::ostream& Out, GRExprEngine::StateTy St) {
    ValueState::ConstNotEqTy NE = St.getImpl()->ConstNotEq;
    
    if (NE.isEmpty())
      return;
    
    Out << "\\l\\|'!=' constraints:";
    
    for (ValueState::ConstNotEqTy::iterator I=NE.begin(), EI=NE.end();
         I != EI; ++I){
      
      Out << "\\l $" << I.getKey() << " : ";
      bool isFirst = true;
      
      ValueState::IntSetTy::iterator J=I.getData().begin(),
                                    EJ=I.getData().end();      
      for ( ; J != EJ; ++J) {        
        if (isFirst) isFirst = false;
        else Out << ", ";
        
        Out << (*J)->toString();
      }    
    }
  }
    
  static std::string getNodeAttributes(const GRExprEngine::NodeTy* N, void*) {
    
    if (GraphPrintCheckerState->isImplicitNullDeref(N) ||
        GraphPrintCheckerState->isExplicitNullDeref(N) ||
        GraphPrintCheckerState->isUninitDeref(N) ||
        GraphPrintCheckerState->isUninitStore(N) ||
        GraphPrintCheckerState->isUninitControlFlow(N))
      return "color=\"red\",style=\"filled\"";
    
    return "";
  }
    
  static std::string getNodeLabel(const GRExprEngine::NodeTy* N, void*) {
    std::ostringstream Out;

    // Program Location.
    ProgramPoint Loc = N->getLocation();
    
    switch (Loc.getKind()) {
      case ProgramPoint::BlockEntranceKind:
        Out << "Block Entrance: B" 
            << cast<BlockEntrance>(Loc).getBlock()->getBlockID();
        break;
      
      case ProgramPoint::BlockExitKind:
        assert (false);
        break;
        
      case ProgramPoint::PostStmtKind: {
        const PostStmt& L = cast<PostStmt>(Loc);
        Out << L.getStmt()->getStmtClassName() << ':' 
            << (void*) L.getStmt() << ' ';
        
        L.getStmt()->printPretty(Out);
        
        if (GraphPrintCheckerState->isImplicitNullDeref(N)) {
          Out << "\\|Implicit-Null Dereference.\\l";
        }
        else if (GraphPrintCheckerState->isExplicitNullDeref(N)) {
          Out << "\\|Explicit-Null Dereference.\\l";
        }
        else if (GraphPrintCheckerState->isUninitDeref(N)) {
          Out << "\\|Dereference of uninitialied value.\\l";
        }
        else if (GraphPrintCheckerState->isUninitStore(N)) {
          Out << "\\|Store to Uninitialized LVal.";
        }
        
        break;
      }
    
      default: {
        const BlockEdge& E = cast<BlockEdge>(Loc);
        Out << "Edge: (B" << E.getSrc()->getBlockID() << ", B"
            << E.getDst()->getBlockID()  << ')';
        
        if (Stmt* T = E.getSrc()->getTerminator()) {
          Out << "\\|Terminator: ";
          E.getSrc()->printTerminator(Out);
          
          if (isa<SwitchStmt>(T)) {
            Stmt* Label = E.getDst()->getLabel();
            
            if (Label) {                        
              if (CaseStmt* C = dyn_cast<CaseStmt>(Label)) {
                Out << "\\lcase ";
                C->getLHS()->printPretty(Out);
                
                if (Stmt* RHS = C->getRHS()) {
                  Out << " .. ";
                  RHS->printPretty(Out);
                }
                
                Out << ":";
              }
              else {
                assert (isa<DefaultStmt>(Label));
                Out << "\\ldefault:";
              }
            }
            else 
              Out << "\\l(implicit) default:";
          }
          else if (isa<IndirectGotoStmt>(T)) {
            // FIXME
          }
          else {
            Out << "\\lCondition: ";
            if (*E.getSrc()->succ_begin() == E.getDst())
              Out << "true";
            else
              Out << "false";                        
          }
          
          Out << "\\l";
        }
        
        if (GraphPrintCheckerState->isUninitControlFlow(N)) {
          Out << "\\|Control-flow based on\\lUninitialized value.\\l";
        }
      }
    }
    
    Out << "\\|StateID: " << (void*) N->getState().getImpl() << "\\|";

    N->getState().printDOT(Out);
      
    Out << "\\l";
    return Out.str();
  }
};
} // end llvm namespace    
#endif

void GRExprEngine::ViewGraph() {
#ifndef NDEBUG
  GraphPrintCheckerState = this;
  llvm::ViewGraph(*G.roots_begin(), "GRExprEngine");
  GraphPrintCheckerState = NULL;
#endif
}
