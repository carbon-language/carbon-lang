//===-- GRConstants.cpp - Simple, Path-Sens. Constant Prop. ------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//               Constant Propagation via Graph Reachability
//
//  This files defines a simple analysis that performs path-sensitive
//  constant propagation within a function.  An example use of this analysis
//  is to perform simple checks for NULL dereferences.
//
//===----------------------------------------------------------------------===//

#include "RValues.h"
#include "ValueState.h"

#include "clang/Analysis/PathSensitive/GREngine.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ASTContext.h"
#include "clang/Analysis/Analyses/LiveVariables.h"
#include "clang/Basic/Diagnostic.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Streams.h"

#include <functional>

#ifndef NDEBUG
#include "llvm/Support/GraphWriter.h"
#include <sstream>
#endif

using namespace clang;
using llvm::dyn_cast;
using llvm::cast;
using llvm::APSInt;

//===----------------------------------------------------------------------===//
// The Checker.
//
//  FIXME: This checker logic should be eventually broken into two components.
//         The first is the "meta"-level checking logic; the code that
//         does the Stmt visitation, fetching values from the map, etc.
//         The second part does the actual state manipulation.  This way we
//         get more of a separate of concerns of these two pieces, with the
//         latter potentially being refactored back into the main checking
//         logic.
//===----------------------------------------------------------------------===//

namespace {
  
class VISIBILITY_HIDDEN GRConstants {
    
public:
  typedef ValueStateManager::StateTy StateTy;
  typedef GRStmtNodeBuilder<GRConstants> StmtNodeBuilder;
  typedef GRBranchNodeBuilder<GRConstants> BranchNodeBuilder;
  typedef ExplodedGraph<GRConstants> GraphTy;
  typedef GraphTy::NodeTy NodeTy;
  
  class NodeSet {
    typedef llvm::SmallVector<NodeTy*,3> ImplTy;
    ImplTy Impl;
  public:
    
    NodeSet() {}
    NodeSet(NodeTy* N) { assert (N && !N->isSink()); Impl.push_back(N); }
    
    void Add(NodeTy* N) { if (N && !N->isSink()) Impl.push_back(N); }
    
    typedef ImplTy::iterator       iterator;
    typedef ImplTy::const_iterator const_iterator;
        
    unsigned size() const { return Impl.size(); }
    bool empty() const { return Impl.empty(); }
    
    iterator begin() { return Impl.begin(); }
    iterator end()   { return Impl.end(); }

    const_iterator begin() const { return Impl.begin(); }
    const_iterator end() const { return Impl.end(); }
  };
                                                                
protected:
  /// G - the simulation graph.
  GraphTy& G;
  
  /// Liveness - live-variables information the ValueDecl* and block-level
  ///  Expr* in the CFG.  Used to prune out dead state.
  LiveVariables Liveness;

  /// Builder - The current GRStmtNodeBuilder which is used when building the nodes
  ///  for a given statement.
  StmtNodeBuilder* Builder;
  
  /// StateMgr - Object that manages the data for all created states.
  ValueStateManager StateMgr;
  
  /// ValueMgr - Object that manages the data for all created RValues.
  ValueManager& ValMgr;
  
  /// SymMgr - Object that manages the symbol information.
  SymbolManager& SymMgr;
  
  /// StmtEntryNode - The immediate predecessor node.
  NodeTy* StmtEntryNode;
  
  /// CurrentStmt - The current block-level statement.
  Stmt* CurrentStmt;
  
  /// UninitBranches - Nodes in the ExplodedGraph that result from
  ///  taking a branch based on an uninitialized value.
  typedef llvm::SmallPtrSet<NodeTy*,5> UninitBranchesTy;
  UninitBranchesTy UninitBranches;
  
  /// ImplicitNullDeref - Nodes in the ExplodedGraph that result from
  ///  taking a dereference on a symbolic pointer that may be NULL.
  typedef llvm::SmallPtrSet<NodeTy*,5> NullDerefTy;
  NullDerefTy ImplicitNullDeref;
  NullDerefTy ExplicitNullDeref;
  
  
  bool StateCleaned;
  
public:
  GRConstants(GraphTy& g) : G(g), Liveness(G.getCFG(), G.getFunctionDecl()),
      Builder(NULL),
      StateMgr(G.getContext(), G.getAllocator()),
      ValMgr(StateMgr.getValueManager()),
      SymMgr(StateMgr.getSymbolManager()),
      StmtEntryNode(NULL), CurrentStmt(NULL) {
    
    // Compute liveness information.
    Liveness.runOnCFG(G.getCFG());
    Liveness.runOnAllBlocks(G.getCFG(), NULL, true);
  }
  
  /// getContext - Return the ASTContext associated with this analysis.
  ASTContext& getContext() const { return G.getContext(); }
  
  /// getCFG - Returns the CFG associated with this analysis.
  CFG& getCFG() { return G.getCFG(); }
  
  /// getInitialState - Return the initial state used for the root vertex
  ///  in the ExplodedGraph.
  StateTy getInitialState() {
    StateTy St = StateMgr.getInitialState();
    
    // Iterate the parameters.
    FunctionDecl& F = G.getFunctionDecl();
    
    for (FunctionDecl::param_iterator I=F.param_begin(), E=F.param_end(); 
          I!=E; ++I)
      St = SetValue(St, lval::DeclVal(*I), RValue::GetSymbolValue(SymMgr, *I));
    
    return St;
  }
  
  bool isUninitControlFlow(const NodeTy* N) const {
    return N->isSink() && UninitBranches.count(const_cast<NodeTy*>(N)) != 0;
  }
  
  bool isImplicitNullDeref(const NodeTy* N) const {
    return N->isSink() && ImplicitNullDeref.count(const_cast<NodeTy*>(N)) != 0;
  }
  
  bool isExplicitNullDeref(const NodeTy* N) const {
    return N->isSink() && ExplicitNullDeref.count(const_cast<NodeTy*>(N)) != 0;
  }
  
  typedef NullDerefTy::iterator null_iterator;
  null_iterator null_begin() { return ExplicitNullDeref.begin(); }
  null_iterator null_end() { return ExplicitNullDeref.end(); }

  /// ProcessStmt - Called by GREngine. Used to generate new successor
  ///  nodes by processing the 'effects' of a block-level statement.
  void ProcessStmt(Stmt* S, StmtNodeBuilder& builder);    
  
  /// ProcessBranch - Called by GREngine.  Used to generate successor
  ///  nodes by processing the 'effects' of a branch condition.
  void ProcessBranch(Expr* Condition, Stmt* Term, BranchNodeBuilder& builder);

  /// RemoveDeadBindings - Return a new state that is the same as 'M' except
  ///  that all subexpression mappings are removed and that any
  ///  block-level expressions that are not live at 'S' also have their
  ///  mappings removed.
  StateTy RemoveDeadBindings(Stmt* S, StateTy M);

  StateTy SetValue(StateTy St, Stmt* S, const RValue& V);

  StateTy SetValue(StateTy St, const Stmt* S, const RValue& V) {
    return SetValue(St, const_cast<Stmt*>(S), V);
  }
  
  /// SetValue - This version of SetValue is used to batch process a set
  ///  of different possible RValues and return a set of different states.
  const StateTy::BufferTy& SetValue(StateTy St, Stmt* S,
                                    const RValue::BufferTy& V,
                                    StateTy::BufferTy& RetBuf);
  
  StateTy SetValue(StateTy St, const LValue& LV, const RValue& V);
  
  inline RValue GetValue(const StateTy& St, Stmt* S) {
    return StateMgr.GetValue(St, S);
  }
  
  inline RValue GetValue(const StateTy& St, Stmt* S, bool& hasVal) {
    return StateMgr.GetValue(St, S, &hasVal);
  }
  
  inline RValue GetValue(const StateTy& St, const Stmt* S) {
    return GetValue(St, const_cast<Stmt*>(S));
  }
  
  inline RValue GetValue(const StateTy& St, const LValue& LV,
                         QualType* T = NULL) {
    
    return StateMgr.GetValue(St, LV, T);
  }
  
  inline LValue GetLValue(const StateTy& St, Stmt* S) {
    return StateMgr.GetLValue(St, S);
  }
  
  inline NonLValue GetRValueConstant(uint64_t X, Expr* E) {
    return NonLValue::GetValue(ValMgr, X, E->getType(), E->getLocStart());
  }
    
  /// Assume - Create new state by assuming that a given expression
  ///  is true or false.
  inline StateTy Assume(StateTy St, RValue Cond, bool Assumption, 
                        bool& isFeasible) {
    if (isa<LValue>(Cond))
      return Assume(St, cast<LValue>(Cond), Assumption, isFeasible);
    else
      return Assume(St, cast<NonLValue>(Cond), Assumption, isFeasible);
  }
  
  StateTy Assume(StateTy St, LValue Cond, bool Assumption, bool& isFeasible);
  StateTy Assume(StateTy St, NonLValue Cond, bool Assumption, bool& isFeasible);
  
  StateTy AssumeSymNE(StateTy St, SymbolID sym, const llvm::APSInt& V,
                      bool& isFeasible);

  StateTy AssumeSymEQ(StateTy St, SymbolID sym, const llvm::APSInt& V,
                      bool& isFeasible);
  
  StateTy AssumeSymInt(StateTy St, bool Assumption, const SymIntConstraint& C,
                       bool& isFeasible);
  
  NodeTy* Nodify(NodeSet& Dst, Stmt* S, NodeTy* Pred, StateTy St,
                 bool AlwaysMakeNode = false);
  
  /// Nodify - This version of Nodify is used to batch process a set of states.
  ///  The states are not guaranteed to be unique.
  void Nodify(NodeSet& Dst, Stmt* S, NodeTy* Pred, const StateTy::BufferTy& SB);
  
  /// Visit - Transfer function logic for all statements.  Dispatches to
  ///  other functions that handle specific kinds of statements.
  void Visit(Stmt* S, NodeTy* Pred, NodeSet& Dst);

  /// VisitCast - Transfer function logic for all casts (implicit and explicit).
  void VisitCast(Expr* CastE, Expr* E, NodeTy* Pred, NodeSet& Dst);
  
  /// VisitUnaryOperator - Transfer function logic for unary operators.
  void VisitUnaryOperator(UnaryOperator* B, NodeTy* Pred, NodeSet& Dst);
  
  /// VisitBinaryOperator - Transfer function logic for binary operators.
  void VisitBinaryOperator(BinaryOperator* B, NodeTy* Pred, NodeSet& Dst);
  
  void VisitAssignmentLHS(Expr* E, NodeTy* Pred, NodeSet& Dst);

  /// VisitDeclRefExpr - Transfer function logic for DeclRefExprs.
  void VisitDeclRefExpr(DeclRefExpr* DR, NodeTy* Pred, NodeSet& Dst); 

  /// VisitDeclStmt - Transfer function logic for DeclStmts.
  void VisitDeclStmt(DeclStmt* DS, NodeTy* Pred, NodeSet& Dst); 
  
  /// VisitGuardedExpr - Transfer function logic for ?, __builtin_choose
  void VisitGuardedExpr(Stmt* S, Stmt* LHS, Stmt* RHS,
                        NodeTy* Pred, NodeSet& Dst);
  
  /// VisitLogicalExpr - Transfer function logic for '&&', '||'
  void VisitLogicalExpr(BinaryOperator* B, NodeTy* Pred, NodeSet& Dst);
};
} // end anonymous namespace


GRConstants::StateTy
GRConstants::SetValue(StateTy St, Stmt* S, const RValue& V) {

  if (!StateCleaned) {
    St = RemoveDeadBindings(CurrentStmt, St);
    StateCleaned = true;
  }

  bool isBlkExpr = false;
    
  if (S == CurrentStmt) {
    isBlkExpr = getCFG().isBlkExpr(S);
    
    if (!isBlkExpr)
      return St;
  }

  return StateMgr.SetValue(St, S, isBlkExpr, V);
}

const GRConstants::StateTy::BufferTy&
GRConstants::SetValue(StateTy St, Stmt* S, const RValue::BufferTy& RB,
                      StateTy::BufferTy& RetBuf) {
  
  assert (RetBuf.empty());
  
  for (RValue::BufferTy::const_iterator I=RB.begin(), E=RB.end(); I!=E; ++I)
    RetBuf.push_back(SetValue(St, S, *I));
                     
  return RetBuf;
}

GRConstants::StateTy
GRConstants::SetValue(StateTy St, const LValue& LV, const RValue& V) {
  
  if (!LV.isValid())
    return St;
  
  if (!StateCleaned) {
    St = RemoveDeadBindings(CurrentStmt, St);
    StateCleaned = true;
  }
  
  return StateMgr.SetValue(St, LV, V);
}

void GRConstants::ProcessBranch(Expr* Condition, Stmt* Term,
                                BranchNodeBuilder& builder) {

  StateTy PrevState = builder.getState();
  
  // Remove old bindings for subexpressions.  
  for (StateTy::vb_iterator I=PrevState.begin(), E=PrevState.end(); I!=E; ++I)
    if (I.getKey().isSubExpr())
      PrevState = StateMgr.Remove(PrevState, I.getKey());
  
  // Remove terminator-specific bindings.
  switch (Term->getStmtClass()) {
    default: break;
      
    case Stmt::BinaryOperatorClass: { // '&&', '||'
      BinaryOperator* B = cast<BinaryOperator>(Term);
      // FIXME: Liveness analysis should probably remove these automatically.
      //   Verify later when we converge to an 'optimization' stage.
      PrevState = StateMgr.Remove(PrevState, B->getRHS());
      break;
    }
      
    case Stmt::ConditionalOperatorClass: { // '?' operator
      ConditionalOperator* C = cast<ConditionalOperator>(Term);
      // FIXME: Liveness analysis should probably remove these automatically.
      //   Verify later when we converge to an 'optimization' stage.
      if (Expr* L = C->getLHS()) PrevState = StateMgr.Remove(PrevState, L);
      PrevState = StateMgr.Remove(PrevState, C->getRHS());
      break;
    }
      
    case Stmt::ChooseExprClass: { // __builtin_choose_expr
      ChooseExpr* C = cast<ChooseExpr>(Term);
      // FIXME: Liveness analysis should probably remove these automatically.
      //   Verify later when we converge to an 'optimization' stage.
      PrevState = StateMgr.Remove(PrevState, C->getRHS());
      PrevState = StateMgr.Remove(PrevState, C->getRHS());
      break;   
    }
  }
  
  RValue V = GetValue(PrevState, Condition);
  
  switch (V.getBaseKind()) {
    default:
      break;

    case RValue::InvalidKind:
      builder.generateNode(PrevState, true);
      builder.generateNode(PrevState, false);
      return;
      
    case RValue::UninitializedKind: {      
      NodeTy* N = builder.generateNode(PrevState, true);

      if (N) {
        N->markAsSink();
        UninitBranches.insert(N);
      }
      
      builder.markInfeasible(false);
      return;
    }      
  }

  // Process the true branch.
  bool isFeasible = true;
  
  StateTy St = Assume(PrevState, V, true, isFeasible);

  if (isFeasible)
    builder.generateNode(St, true);
  else {
    builder.markInfeasible(true);
    isFeasible = true;
  }
  
  // Process the false branch.  
  St = Assume(PrevState, V, false, isFeasible);
  
  if (isFeasible)
    builder.generateNode(St, false);
  else
    builder.markInfeasible(false);
}


void GRConstants::VisitLogicalExpr(BinaryOperator* B, NodeTy* Pred,
                                   NodeSet& Dst) {

  bool hasR2;
  StateTy PrevState = Pred->getState();

  RValue R1 = GetValue(PrevState, B->getLHS());
  RValue R2 = GetValue(PrevState, B->getRHS(), hasR2);
    
  if (isa<InvalidValue>(R1) && 
       (isa<InvalidValue>(R2) ||
        isa<UninitializedValue>(R2))) {    

    Nodify(Dst, B, Pred, SetValue(PrevState, B, R2));
    return;
  }    
  else if (isa<UninitializedValue>(R1)) {
    Nodify(Dst, B, Pred, SetValue(PrevState, B, R1));
    return;
  }

  // R1 is an expression that can evaluate to either 'true' or 'false'.
  if (B->getOpcode() == BinaryOperator::LAnd) {
    // hasR2 == 'false' means that LHS evaluated to 'false' and that
    // we short-circuited, leading to a value of '0' for the '&&' expression.
    if (hasR2 == false) { 
      Nodify(Dst, B, Pred, SetValue(PrevState, B, GetRValueConstant(0U, B)));
      return;
    }
  }
  else {
    assert (B->getOpcode() == BinaryOperator::LOr);
    // hasR2 == 'false' means that the LHS evaluate to 'true' and that
    //  we short-circuited, leading to a value of '1' for the '||' expression.
    if (hasR2 == false) {
      Nodify(Dst, B, Pred, SetValue(PrevState, B, GetRValueConstant(1U, B)));
      return;      
    }
  }
    
  // If we reach here we did not short-circuit.  Assume R2 == true and
  // R2 == false.
    
  bool isFeasible;
  StateTy St = Assume(PrevState, R2, true, isFeasible);
  
  if (isFeasible)
    Nodify(Dst, B, Pred, SetValue(PrevState, B, GetRValueConstant(1U, B)));

  St = Assume(PrevState, R2, false, isFeasible);
  
  if (isFeasible)
    Nodify(Dst, B, Pred, SetValue(PrevState, B, GetRValueConstant(0U, B)));  
}



void GRConstants::ProcessStmt(Stmt* S, StmtNodeBuilder& builder) {
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
  
  CurrentStmt = NULL;
  StmtEntryNode = NULL;
  Builder = NULL;
}

GRConstants::StateTy GRConstants::RemoveDeadBindings(Stmt* Loc, StateTy M) {
  
  // This code essentially performs a "mark-and-sweep" of the VariableBindings.
  // The roots are any Block-level exprs and Decls that our liveness algorithm
  // tells us are live.  We then see what Decls they may reference, and keep
  // those around.  This code more than likely can be made faster, and the
  // frequency of which this method is called should be experimented with
  // for optimum performance.

  llvm::SmallVector<ValueDecl*, 10> WList;

  for (StateTy::vb_iterator I = M.begin(), E = M.end();
       I!=E && !I.getKey().isSymbol(); ++I) {

    // Remove old bindings for subexpressions.
    if (I.getKey().isSubExpr()) {
      M = StateMgr.Remove(M, I.getKey());
      continue;
    }
    
    if (I.getKey().isBlkExpr()) {
      if (Liveness.isLive(Loc, cast<Stmt>(I.getKey()))) {
        if (isa<lval::DeclVal>(I.getData())) {
          lval::DeclVal LV = cast<lval::DeclVal>(I.getData());
          WList.push_back(LV.getDecl());
        }
      }
      else
        M = StateMgr.Remove(M, I.getKey());
    
      continue;
    }
    
    assert (I.getKey().isDecl());

    if (VarDecl* V = dyn_cast<VarDecl>(cast<ValueDecl>(I.getKey())))
      if (Liveness.isLive(Loc, V))
        WList.push_back(V);
  }

  llvm::SmallPtrSet<ValueDecl*, 10> Marked;
  
  while (!WList.empty()) {
    ValueDecl* V = WList.back();
    WList.pop_back();
    
    if (Marked.count(V))
      continue;
    
    Marked.insert(V);
    
    if (V->getType()->isPointerType()) {
      const LValue& LV = cast<LValue>(GetValue(M, lval::DeclVal(V)));

      if (!isa<lval::DeclVal>(LV))
        continue;
      
      const lval::DeclVal& LVD = cast<lval::DeclVal>(LV);
      WList.push_back(LVD.getDecl());
    }    
  }

  for (StateTy::vb_iterator I = M.begin(), E = M.end(); I!=E ; ++I)
    if (I.getKey().isDecl())
      if (VarDecl* V = dyn_cast<VarDecl>(cast<ValueDecl>(I.getKey())))
        if (!Marked.count(V))
          M = StateMgr.Remove(M, V);

  return M;
}

GRConstants::NodeTy*
GRConstants::Nodify(NodeSet& Dst, Stmt* S, NodeTy* Pred, StateTy St,
                    bool AlwaysMakeNode) {
 
  // If the state hasn't changed, don't generate a new node.
  if (!AlwaysMakeNode && St == Pred->getState())
    return NULL;
  
  NodeTy* N = Builder->generateNode(S, St, Pred);
  Dst.Add(N);
  return N;
}

void GRConstants::Nodify(NodeSet& Dst, Stmt* S, NodeTy* Pred,
                         const StateTy::BufferTy& SB) {
  
  for (StateTy::BufferTy::const_iterator I=SB.begin(), E=SB.end(); I!=E; ++I)
    Nodify(Dst, S, Pred, *I);
}

void GRConstants::VisitDeclRefExpr(DeclRefExpr* D, NodeTy* Pred, NodeSet& Dst) {
  if (D != CurrentStmt) {
    Dst.Add(Pred); // No-op. Simply propagate the current state unchanged.
    return;
  }
  
  // If we are here, we are loading the value of the decl and binding
  // it to the block-level expression.
  
  StateTy St = Pred->getState();
  
  Nodify(Dst, D, Pred,
         SetValue(St, D, GetValue(St, lval::DeclVal(D->getDecl()))));
}

void GRConstants::VisitCast(Expr* CastE, Expr* E, NodeTy* Pred, NodeSet& Dst) {
  
  QualType T = CastE->getType();

  // Check for redundant casts.
  if (E->getType() == T) {
    Dst.Add(Pred);
    return;
  }
  
  NodeSet S1;
  Visit(E, Pred, S1);
  
  for (NodeSet::iterator I1=S1.begin(), E1=S1.end(); I1 != E1; ++I1) {
    NodeTy* N = *I1;
    StateTy St = N->getState();
    const RValue& V = GetValue(St, E);
    Nodify(Dst, CastE, N, SetValue(St, CastE, V.EvalCast(ValMgr, CastE)));
  }
}

void GRConstants::VisitDeclStmt(DeclStmt* DS, GRConstants::NodeTy* Pred,
                                GRConstants::NodeSet& Dst) {
  
  StateTy St = Pred->getState();
  
  for (const ScopedDecl* D = DS->getDecl(); D; D = D->getNextDeclarator())
    if (const VarDecl* VD = dyn_cast<VarDecl>(D)) {
      const Expr* E = VD->getInit();      
      St = SetValue(St, lval::DeclVal(VD),
                    E ? GetValue(St, E) : UninitializedValue());
    }

  Nodify(Dst, DS, Pred, St);
  
  if (Dst.empty())
    Dst.Add(Pred);  
}


void GRConstants::VisitGuardedExpr(Stmt* S, Stmt* LHS, Stmt* RHS,
                                   NodeTy* Pred, NodeSet& Dst) {
  
  StateTy St = Pred->getState();
  
  RValue R = GetValue(St, LHS);
  if (isa<InvalidValue>(R)) R = GetValue(St, RHS);
  
  Nodify(Dst, S, Pred, SetValue(St, S, R));
}

void GRConstants::VisitUnaryOperator(UnaryOperator* U,
                                     GRConstants::NodeTy* Pred,
                                     GRConstants::NodeSet& Dst) {
  
  NodeSet S1;
  UnaryOperator::Opcode Op = U->getOpcode();
  
  // FIXME: This is a hack so that for '*' and '&' we don't recurse
  //  on visiting the subexpression if it is a DeclRefExpr.  We should
  //  probably just handle AddrOf and Deref in their own methods to make
  //  this cleaner.
  if ((Op == UnaryOperator::Deref || Op == UnaryOperator::AddrOf) &&
      isa<DeclRefExpr>(U->getSubExpr()))
    S1.Add(Pred);
  else
    Visit(U->getSubExpr(), Pred, S1);
    
  for (NodeSet::iterator I1=S1.begin(), E1=S1.end(); I1 != E1; ++I1) {
    NodeTy* N1 = *I1;
    StateTy St = N1->getState();
    
    switch (U->getOpcode()) {
      case UnaryOperator::PostInc: {
        const LValue& L1 = GetLValue(St, U->getSubExpr());
        NonLValue R1 = cast<NonLValue>(GetValue(St, L1));
        
        NonLValue Result = R1.EvalBinaryOp(ValMgr, BinaryOperator::Add,
                                           GetRValueConstant(1U, U));
        
        Nodify(Dst, U, N1, SetValue(SetValue(St, U, R1), L1, Result));
        break;
      }
        
      case UnaryOperator::PostDec: {
        const LValue& L1 = GetLValue(St, U->getSubExpr());
        NonLValue R1 = cast<NonLValue>(GetValue(St, L1));
        
        NonLValue Result = R1.EvalBinaryOp(ValMgr, BinaryOperator::Sub,
                                           GetRValueConstant(1U, U));
        
        Nodify(Dst, U, N1, SetValue(SetValue(St, U, R1), L1, Result));
        break;
      }
        
      case UnaryOperator::PreInc: {
        const LValue& L1 = GetLValue(St, U->getSubExpr());
        NonLValue R1 = cast<NonLValue>(GetValue(St, L1));
        
        NonLValue Result = R1.EvalBinaryOp(ValMgr, BinaryOperator::Add,
                                           GetRValueConstant(1U, U));
        
        Nodify(Dst, U, N1, SetValue(SetValue(St, U, Result), L1, Result));
        break;
      }
        
      case UnaryOperator::PreDec: {
        const LValue& L1 = GetLValue(St, U->getSubExpr());
        NonLValue R1 = cast<NonLValue>(GetValue(St, L1));
        
        NonLValue Result = R1.EvalBinaryOp(ValMgr, BinaryOperator::Sub,
                                           GetRValueConstant(1U, U));

        Nodify(Dst, U, N1, SetValue(SetValue(St, U, Result), L1, Result));
        break;
      }
        
      case UnaryOperator::Minus: {
        const NonLValue& R1 = cast<NonLValue>(GetValue(St, U->getSubExpr()));
        Nodify(Dst, U, N1, SetValue(St, U, R1.EvalMinus(ValMgr, U)));
        break;
      }
        
      case UnaryOperator::Not: {
        const NonLValue& R1 = cast<NonLValue>(GetValue(St, U->getSubExpr()));
        Nodify(Dst, U, N1, SetValue(St, U, R1.EvalComplement(ValMgr)));
        break;
      }
        
      case UnaryOperator::LNot: {
        // C99 6.5.3.3: "The expression !E is equivalent to (0==E)."
        //
        //  Note: technically we do "E == 0", but this is the same in the
        //    transfer functions as "0 == E".
        
        RValue V1 = GetValue(St, U->getSubExpr());
        
        if (isa<LValue>(V1)) {
          const LValue& L1 = cast<LValue>(V1);
          lval::ConcreteInt V2(ValMgr.getZeroWithPtrWidth());
          Nodify(Dst, U, N1,
                 SetValue(St, U, L1.EvalBinaryOp(ValMgr, BinaryOperator::EQ,
                                                 V2)));
        }
        else {
          const NonLValue& R1 = cast<NonLValue>(V1);
          nonlval::ConcreteInt V2(ValMgr.getZeroWithPtrWidth());
          Nodify(Dst, U, N1,
                 SetValue(St, U, R1.EvalBinaryOp(ValMgr, BinaryOperator::EQ,
                                                 V2)));
        }
        
        break;
      }
        
      case UnaryOperator::AddrOf: {
        const LValue& L1 = GetLValue(St, U->getSubExpr());
        Nodify(Dst, U, N1, SetValue(St, U, L1));
        break;
      }
        
      case UnaryOperator::Deref: {
        // FIXME: Stop when dereferencing an uninitialized value.
        // FIXME: Bifurcate when dereferencing a symbolic with no constraints?
        
        const RValue& V = GetValue(St, U->getSubExpr());
        const LValue& L1 = cast<LValue>(V);
        
        // After a dereference, one of two possible situations arise:
        //  (1) A crash, because the pointer was NULL.
        //  (2) The pointer is not NULL, and the dereference works.
        // 
        // We add these assumptions.
                
        bool isFeasibleNotNull;
       
        // "Assume" that the pointer is Not-NULL.
        StateTy StNotNull = Assume(St, L1, true, isFeasibleNotNull);
        
        if (isFeasibleNotNull) {
          QualType T = U->getType();
          Nodify(Dst, U, N1, SetValue(StNotNull, U,
                                      GetValue(StNotNull, L1, &T)));
        }
        
        bool isFeasibleNull;
        
        // "Assume" that the pointer is NULL.
        StateTy StNull = Assume(St, L1, false, isFeasibleNull);
        
        if (isFeasibleNull) {
          NodeTy* NullNode = Nodify(Dst, U, N1, StNull, true);
          if (NullNode) {
            NullNode->markAsSink();
            
            if (isFeasibleNotNull)
              ImplicitNullDeref.insert(NullNode);
            else
              ExplicitNullDeref.insert(NullNode);
          }
        }
        
        break;
      }
        
      default: ;
        assert (false && "Not implemented.");
    }    
  }
}

void GRConstants::VisitAssignmentLHS(Expr* E, GRConstants::NodeTy* Pred,
                                     GRConstants::NodeSet& Dst) {

  if (isa<DeclRefExpr>(E)) {
    Dst.Add(Pred);
    return;
  }
  
  if (UnaryOperator* U = dyn_cast<UnaryOperator>(E)) {
    if (U->getOpcode() == UnaryOperator::Deref) {
      Visit(U->getSubExpr(), Pred, Dst);
      return;
    }
  }
  
  Visit(E, Pred, Dst);
}

void GRConstants::VisitBinaryOperator(BinaryOperator* B,
                                      GRConstants::NodeTy* Pred,
                                      GRConstants::NodeSet& Dst) {
  NodeSet S1;
  
  if (B->isAssignmentOp())
    VisitAssignmentLHS(B->getLHS(), Pred, S1);
  else
    Visit(B->getLHS(), Pred, S1);

  for (NodeSet::iterator I1=S1.begin(), E1=S1.end(); I1 != E1; ++I1) {
    NodeTy* N1 = *I1;
    
    // When getting the value for the LHS, check if we are in an assignment.
    // In such cases, we want to (initially) treat the LHS as an LValue,
    // so we use GetLValue instead of GetValue so that DeclRefExpr's are
    // evaluated to LValueDecl's instead of to an NonLValue.
    const RValue& V1 = 
      B->isAssignmentOp() ? GetLValue(N1->getState(), B->getLHS())
                          : GetValue(N1->getState(), B->getLHS());
    
    NodeSet S2;
    Visit(B->getRHS(), N1, S2);
  
    for (NodeSet::iterator I2=S2.begin(), E2=S2.end(); I2 != E2; ++I2) {

      NodeTy* N2 = *I2;
      StateTy St = N2->getState();
      const RValue& V2 = GetValue(St, B->getRHS());

      BinaryOperator::Opcode Op = B->getOpcode();
      
      if (Op <= BinaryOperator::Or) {
        
        if (isa<InvalidValue>(V1) || isa<UninitializedValue>(V1)) {
          Nodify(Dst, B, N2, SetValue(St, B, V1));
          continue;
        }
        
        if (isa<LValue>(V1)) {
          // FIXME: Add support for RHS being a non-lvalue.
          const LValue& L1 = cast<LValue>(V1);
          const LValue& L2 = cast<LValue>(V2);
          
          Nodify(Dst, B, N2, SetValue(St, B, L1.EvalBinaryOp(ValMgr, Op, L2)));
        }
        else {
          const NonLValue& R1 = cast<NonLValue>(V1);
          const NonLValue& R2 = cast<NonLValue>(V2);
            
          Nodify(Dst, B, N2, SetValue(St, B, R1.EvalBinaryOp(ValMgr, Op, R2)));
        }
        
        continue;
      
      }
      
      switch (Op) {
        case BinaryOperator::Assign: {
          const LValue& L1 = cast<LValue>(V1);
          Nodify(Dst, B, N2, SetValue(SetValue(St, B, V2), L1, V2));
          break;
        }

        default: { // Compound assignment operators.
          
          assert (B->isCompoundAssignmentOp());
                          
          const LValue& L1 = cast<LValue>(V1);
          RValue Result = cast<NonLValue>(InvalidValue());
          
          Op = (BinaryOperator::Opcode)
                  (((unsigned) Op) - ((unsigned) BinaryOperator::MulAssign));
          
          if (isa<LValue>(V2)) {
            // FIXME: Add support for Non-LValues on RHS.
            const LValue& L2 = cast<LValue>(V2);
            Result = L1.EvalBinaryOp(ValMgr, Op, L2);
          }
          else {
            const NonLValue& R1 = cast<NonLValue>(GetValue(N1->getState(), L1));
            const NonLValue& R2 = cast<NonLValue>(V2);
            Result = R1.EvalBinaryOp(ValMgr, Op, R2);
          }
          
          Nodify(Dst, B, N2, SetValue(SetValue(St, B, Result), L1, Result));
          break;
        }
      }
    }
  }
}


void GRConstants::Visit(Stmt* S, GRConstants::NodeTy* Pred,
                        GRConstants::NodeSet& Dst) {

  // FIXME: add metadata to the CFG so that we can disable
  //  this check when we KNOW that there is no block-level subexpression.
  //  The motivation is that this check requires a hashtable lookup.

  if (S != CurrentStmt && getCFG().isBlkExpr(S)) {
    Dst.Add(Pred);
    return;
  }

  switch (S->getStmtClass()) {
    case Stmt::BinaryOperatorClass:
 
      if (cast<BinaryOperator>(S)->isLogicalOp()) {
        VisitLogicalExpr(cast<BinaryOperator>(S), Pred, Dst);
        break;
      }
      
      // Fall-through.
      
    case Stmt::CompoundAssignOperatorClass:
      VisitBinaryOperator(cast<BinaryOperator>(S), Pred, Dst);
      break;
      
    case Stmt::UnaryOperatorClass:
      VisitUnaryOperator(cast<UnaryOperator>(S), Pred, Dst);
      break;
      
    case Stmt::ParenExprClass:
      Visit(cast<ParenExpr>(S)->getSubExpr(), Pred, Dst);
      break;
    
    case Stmt::DeclRefExprClass:
      VisitDeclRefExpr(cast<DeclRefExpr>(S), Pred, Dst);
      break;
      
    case Stmt::ImplicitCastExprClass: {
      ImplicitCastExpr* C = cast<ImplicitCastExpr>(S);
      VisitCast(C, C->getSubExpr(), Pred, Dst);
      break;
    }
      
    case Stmt::CastExprClass: {
      CastExpr* C = cast<CastExpr>(S);
      VisitCast(C, C->getSubExpr(), Pred, Dst);
      break;
    }
      
    case Stmt::ConditionalOperatorClass: { // '?' operator
      ConditionalOperator* C = cast<ConditionalOperator>(S);
      VisitGuardedExpr(S, C->getLHS(), C->getRHS(), Pred, Dst);
      break;
    }

    case Stmt::ChooseExprClass: { // __builtin_choose_expr
      ChooseExpr* C = cast<ChooseExpr>(S);
      VisitGuardedExpr(S, C->getLHS(), C->getRHS(), Pred, Dst);
      break;
    }
      
    case Stmt::ReturnStmtClass:
      if (Expr* R = cast<ReturnStmt>(S)->getRetValue())
        Visit(R, Pred, Dst);
      else
        Dst.Add(Pred);
      
      break;
      
    case Stmt::DeclStmtClass:
      VisitDeclStmt(cast<DeclStmt>(S), Pred, Dst);
      break;
      
    default:
      Dst.Add(Pred); // No-op. Simply propagate the current state unchanged.
      break;
  }
}

//===----------------------------------------------------------------------===//
// "Assume" logic.
//===----------------------------------------------------------------------===//

GRConstants::StateTy GRConstants::Assume(StateTy St, LValue Cond,
                                         bool Assumption, 
                                         bool& isFeasible) {    
  
  switch (Cond.getSubKind()) {
    default:
      assert (false && "'Assume' not implemented for this LValue.");
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

GRConstants::StateTy GRConstants::Assume(StateTy St, NonLValue Cond,
                                         bool Assumption, 
                                         bool& isFeasible) {
  
  switch (Cond.getSubKind()) {
    default:
      assert (false && "'Assume' not implemented for this NonLValue.");
      return St;
      
      
    case nonlval::SymbolValKind: {
      lval::SymbolVal& SV = cast<lval::SymbolVal>(Cond);
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

GRConstants::StateTy
GRConstants::AssumeSymNE(StateTy St, SymbolID sym,
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

GRConstants::StateTy
GRConstants::AssumeSymEQ(StateTy St, SymbolID sym,
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

GRConstants::StateTy
GRConstants::AssumeSymInt(StateTy St, bool Assumption,
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
// Driver.
//===----------------------------------------------------------------------===//

#ifndef NDEBUG
static GRConstants* GraphPrintCheckerState;

namespace llvm {
template<>
struct VISIBILITY_HIDDEN DOTGraphTraits<GRConstants::NodeTy*> :
  public DefaultDOTGraphTraits {

  static void PrintKindLabel(std::ostream& Out, VarBindKey::Kind kind) {
    switch (kind) {
      case VarBindKey::IsSubExpr:  Out << "Sub-Expressions:\\l"; break;
      case VarBindKey::IsDecl:    Out << "Variables:\\l"; break;
      case VarBindKey::IsBlkExpr: Out << "Block-level Expressions:\\l"; break;
      default: assert (false && "Unknown VarBindKey type.");
    }
  }
    
  static void PrintKind(std::ostream& Out, GRConstants::StateTy M,
                        VarBindKey::Kind kind, bool isFirstGroup = false) {
    bool isFirst = true;
    
    for (GRConstants::StateTy::vb_iterator I=M.begin(), E=M.end();I!=E;++I) {        
      if (I.getKey().getKind() != kind)
        continue;
    
      if (isFirst) {
        if (!isFirstGroup) Out << "\\l\\l";
        PrintKindLabel(Out, kind);
        isFirst = false;
      }
      else
        Out << "\\l";
      
      Out << ' ';
    
      if (ValueDecl* V = dyn_cast<ValueDecl>(I.getKey()))
        Out << V->getName();          
      else {
        Stmt* E = cast<Stmt>(I.getKey());
        Out << " (" << (void*) E << ") ";
        E->printPretty(Out);
      }
    
      Out << " : ";
      I.getData().print(Out);
    }
  }
    
  static void PrintEQ(std::ostream& Out, GRConstants::StateTy St) {
    ValueState::ConstantEqTy CE = St.getImpl()->ConstantEq;
    
    if (CE.isEmpty())
      return;
    
    Out << "\\l\\|'==' constraints:";

    for (ValueState::ConstantEqTy::iterator I=CE.begin(), E=CE.end(); I!=E;++I)
      Out << "\\l $" << I.getKey() << " : " << I.getData()->toString();
  }
    
  static void PrintNE(std::ostream& Out, GRConstants::StateTy St) {
    ValueState::ConstantNotEqTy NE = St.getImpl()->ConstantNotEq;
    
    if (NE.isEmpty())
      return;
    
    Out << "\\l\\|'!=' constraints:";
    
    for (ValueState::ConstantNotEqTy::iterator I=NE.begin(), EI=NE.end();
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
    
  static std::string getNodeLabel(const GRConstants::NodeTy* N, void*) {
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
    
    PrintKind(Out, N->getState(), VarBindKey::IsDecl, true);
    PrintKind(Out, N->getState(), VarBindKey::IsBlkExpr);
    PrintKind(Out, N->getState(), VarBindKey::IsSubExpr);
    
    PrintEQ(Out, N->getState());
    PrintNE(Out, N->getState());
      
    Out << "\\l";
    return Out.str();
  }
};
} // end llvm namespace    
#endif

namespace clang {
void RunGRConstants(CFG& cfg, FunctionDecl& FD, ASTContext& Ctx,
                    Diagnostic& Diag) {
  
  GREngine<GRConstants> Engine(cfg, FD, Ctx);
  Engine.ExecuteWorkList();
  
  // Look for explicit-Null dereferences and warn about them.
  GRConstants* CheckerState = &Engine.getCheckerState();
  
  for (GRConstants::null_iterator I=CheckerState->null_begin(),
                                  E=CheckerState->null_end(); I!=E; ++I) {
    
    const PostStmt& L = cast<PostStmt>((*I)->getLocation());
    Expr* E = cast<Expr>(L.getStmt());
    
    Diag.Report(FullSourceLoc(E->getExprLoc(), Ctx.getSourceManager()),
                diag::chkr_null_deref_after_check);
  }
  
  
#ifndef NDEBUG
  GraphPrintCheckerState = CheckerState;
  llvm::ViewGraph(*Engine.getGraph().roots_begin(),"GRConstants");
  GraphPrintCheckerState = NULL;
#endif  
}
} // end clang namespace
