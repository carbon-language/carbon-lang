//===-- GRExprEngine.h - Path-Sensitive Expression-Level Dataflow ---*- C++ -*-=
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

#include "ValueState.h"

#include "clang/Analysis/PathSensitive/GRCoreEngine.h"
#include "clang/Analysis/PathSensitive/GRTransferFuncs.h"

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

#include <functional>

#ifndef NDEBUG
#include "llvm/Support/GraphWriter.h"
#include <sstream>
#endif

namespace clang {
  
class GRExprEngine {
  
public:
  typedef ValueStateManager::StateTy StateTy;
  typedef ExplodedGraph<GRExprEngine> GraphTy;
  typedef GraphTy::NodeTy NodeTy;
  
  // Builders.
  typedef GRStmtNodeBuilder<GRExprEngine> StmtNodeBuilder;
  typedef GRBranchNodeBuilder<GRExprEngine> BranchNodeBuilder;
  typedef GRIndirectGotoNodeBuilder<GRExprEngine> IndirectGotoNodeBuilder;
  typedef GRSwitchNodeBuilder<GRExprEngine> SwitchNodeBuilder;
  
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
  
  /// Builder - The current GRStmtNodeBuilder which is used when building the
  ///  nodes for a given statement.
  StmtNodeBuilder* Builder;
  
  /// StateMgr - Object that manages the data for all created states.
  ValueStateManager StateMgr;
  
  /// ValueMgr - Object that manages the data for all created RValues.
  ValueManager& ValMgr;
  
  /// TF - Object that represents a bundle of transfer functions
  ///  for manipulating and creating RValues.
  GRTransferFuncs* TF;
  
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
  
  /// UninitStores - Sinks in the ExplodedGraph that result from
  ///  making a store to an uninitialized lvalue.
  typedef llvm::SmallPtrSet<NodeTy*,5> UninitStoresTy;
  UninitStoresTy UninitStores;
  
  /// ImplicitNullDeref - Nodes in the ExplodedGraph that result from
  ///  taking a dereference on a symbolic pointer that may be NULL.
  typedef llvm::SmallPtrSet<NodeTy*,5> BadDerefTy;
  BadDerefTy ImplicitNullDeref;
  BadDerefTy ExplicitNullDeref;
  BadDerefTy UninitDeref;
  
  bool StateCleaned;
  
public:
  GRExprEngine(GraphTy& g) : 
  G(g), Liveness(G.getCFG(), G.getFunctionDecl()),
  Builder(NULL),
  StateMgr(G.getContext(), G.getAllocator()),
  ValMgr(StateMgr.getValueManager()),
  TF(NULL), // FIXME.
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
  
  /// setTransferFunctions
  void setTransferFunctions(GRTransferFuncs* tf) { TF = tf; }
  void setTransferFunctions(GRTransferFuncs& tf) { TF = &tf; }
  
  /// ViewGraph - Visualize the ExplodedGraph created by executing the
  ///  simulation.
  void ViewGraph();
  
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
  
  bool isUninitStore(const NodeTy* N) const {
    return N->isSink() && UninitStores.count(const_cast<NodeTy*>(N)) != 0;
  }
  
  bool isImplicitNullDeref(const NodeTy* N) const {
    return N->isSink() && ImplicitNullDeref.count(const_cast<NodeTy*>(N)) != 0;
  }
  
  bool isExplicitNullDeref(const NodeTy* N) const {
    return N->isSink() && ExplicitNullDeref.count(const_cast<NodeTy*>(N)) != 0;
  }
  
  bool isUninitDeref(const NodeTy* N) const {
    return N->isSink() && UninitDeref.count(const_cast<NodeTy*>(N)) != 0;
  }
  
  typedef BadDerefTy::iterator null_iterator;
  null_iterator null_begin() { return ExplicitNullDeref.begin(); }
  null_iterator null_end() { return ExplicitNullDeref.end(); }
  
  /// ProcessStmt - Called by GRCoreEngine. Used to generate new successor
  ///  nodes by processing the 'effects' of a block-level statement.
  void ProcessStmt(Stmt* S, StmtNodeBuilder& builder);    
  
  /// ProcessBranch - Called by GRCoreEngine.  Used to generate successor
  ///  nodes by processing the 'effects' of a branch condition.
  void ProcessBranch(Expr* Condition, Stmt* Term, BranchNodeBuilder& builder);
  
  /// ProcessIndirectGoto - Called by GRCoreEngine.  Used to generate successor
  ///  nodes by processing the 'effects' of a computed goto jump.
  void ProcessIndirectGoto(IndirectGotoNodeBuilder& builder);
  
  /// ProcessSwitch - Called by GRCoreEngine.  Used to generate successor
  ///  nodes by processing the 'effects' of a switch statement.
  void ProcessSwitch(SwitchNodeBuilder& builder);  
  
  /// RemoveDeadBindings - Return a new state that is the same as 'St' except
  ///  that all subexpression mappings are removed and that any
  ///  block-level expressions that are not live at 'S' also have their
  ///  mappings removed.
  inline StateTy RemoveDeadBindings(Stmt* S, StateTy St) {
    return StateMgr.RemoveDeadBindings(St, S, Liveness);
  }
  
  StateTy SetValue(StateTy St, Expr* S, const RValue& V);
  
  StateTy SetValue(StateTy St, const Expr* S, const RValue& V) {
    return SetValue(St, const_cast<Expr*>(S), V);
  }
  
  /// SetValue - This version of SetValue is used to batch process a set
  ///  of different possible RValues and return a set of different states.
  const StateTy::BufferTy& SetValue(StateTy St, Expr* S,
                                    const RValue::BufferTy& V,
                                    StateTy::BufferTy& RetBuf);
  
  StateTy SetValue(StateTy St, const LValue& LV, const RValue& V);
  
  inline RValue GetValue(const StateTy& St, Expr* S) {
    return StateMgr.GetValue(St, S);
  }
  
  inline RValue GetValue(const StateTy& St, Expr* S, bool& hasVal) {
    return StateMgr.GetValue(St, S, &hasVal);
  }
  
  inline RValue GetValue(const StateTy& St, const Expr* S) {
    return GetValue(St, const_cast<Expr*>(S));
  }
  
  inline RValue GetValue(const StateTy& St, const LValue& LV,
                         QualType* T = NULL) {
    
    return StateMgr.GetValue(St, LV, T);
  }
  
  inline LValue GetLValue(const StateTy& St, Expr* S) {
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
  
  NodeTy* Nodify(NodeSet& Dst, Stmt* S, NodeTy* Pred, StateTy St);
  
  /// Nodify - This version of Nodify is used to batch process a set of states.
  ///  The states are not guaranteed to be unique.
  void Nodify(NodeSet& Dst, Stmt* S, NodeTy* Pred, const StateTy::BufferTy& SB);
  
  /// HandleUninitializedStore - Create the necessary sink node to represent
  ///  a store to an "uninitialized" LValue.
  void HandleUninitializedStore(Stmt* S, NodeTy* Pred);
  
  /// Visit - Transfer function logic for all statements.  Dispatches to
  ///  other functions that handle specific kinds of statements.
  void Visit(Stmt* S, NodeTy* Pred, NodeSet& Dst);
  
  /// VisitBinaryOperator - Transfer function logic for binary operators.
  void VisitBinaryOperator(BinaryOperator* B, NodeTy* Pred, NodeSet& Dst);
  
  void VisitLValue(Expr* E, NodeTy* Pred, NodeSet& Dst);
  
  /// VisitCall - Transfer function for function calls.
  void VisitCall(CallExpr* CE, NodeTy* Pred,
                 CallExpr::arg_iterator I, CallExpr::arg_iterator E,
                 NodeSet& Dst);
  
  /// VisitCast - Transfer function logic for all casts (implicit and explicit).
  void VisitCast(Expr* CastE, Expr* E, NodeTy* Pred, NodeSet& Dst);  
  
  /// VisitDeclRefExpr - Transfer function logic for DeclRefExprs.
  void VisitDeclRefExpr(DeclRefExpr* DR, NodeTy* Pred, NodeSet& Dst); 
  
  /// VisitDeclStmt - Transfer function logic for DeclStmts.
  void VisitDeclStmt(DeclStmt* DS, NodeTy* Pred, NodeSet& Dst); 
  
  /// VisitGuardedExpr - Transfer function logic for ?, __builtin_choose
  void VisitGuardedExpr(Expr* S, Expr* LHS, Expr* RHS,
                        NodeTy* Pred, NodeSet& Dst);
  
  /// VisitLogicalExpr - Transfer function logic for '&&', '||'
  void VisitLogicalExpr(BinaryOperator* B, NodeTy* Pred, NodeSet& Dst);
  
  /// VisitSizeOfAlignOfTypeExpr - Transfer function for sizeof(type).
  void VisitSizeOfAlignOfTypeExpr(SizeOfAlignOfTypeExpr* S, NodeTy* Pred,
                                  NodeSet& Dst);
  
  /// VisitUnaryOperator - Transfer function logic for unary operators.
  void VisitUnaryOperator(UnaryOperator* B, NodeTy* Pred, NodeSet& Dst);
  
  void VisitDeref(UnaryOperator* B, NodeTy* Pred, NodeSet& Dst);
  
  
  inline RValue EvalCast(ValueManager& ValMgr, RValue X, Expr* CastExpr) {
    if (isa<UnknownVal>(X) || isa<UninitializedVal>(X))
      return X;    
    
    return TF->EvalCast(ValMgr, X, CastExpr);
  }
  
  inline NonLValue EvalMinus(ValueManager& ValMgr, UnaryOperator* U,
                             NonLValue X) {
    if (isa<UnknownVal>(X) || isa<UninitializedVal>(X))
      return X;    
    
    return TF->EvalMinus(ValMgr, U, X);    
  }
  
  inline NonLValue EvalPlus(ValueManager& ValMgr, UnaryOperator* U,
                             NonLValue X) {
    if (isa<UnknownVal>(X) || isa<UninitializedVal>(X))
      return X;    
    
    return TF->EvalPlus(ValMgr, U, X);    
  }
  
  inline NonLValue EvalComplement(ValueManager& ValMgr, NonLValue X) {
    if (isa<UnknownVal>(X) || isa<UninitializedVal>(X))
      return X;    

    return TF->EvalComplement(ValMgr, X);
  }
  
  inline NonLValue EvalBinaryOp(BinaryOperator::Opcode Op,
                                NonLValue LHS, NonLValue RHS) {
    
    if (isa<UninitializedVal>(LHS) || isa<UninitializedVal>(RHS))
      return cast<NonLValue>(UninitializedVal());
    
    if (isa<UnknownVal>(LHS) || isa<UnknownVal>(RHS))
      return cast<NonLValue>(UnknownVal());
    
    return TF->EvalBinaryOp(ValMgr, Op, LHS, RHS);
  }    
  
  inline RValue EvalBinaryOp(BinaryOperator::Opcode Op,
                             LValue LHS, LValue RHS) {
    
    if (isa<UninitializedVal>(LHS) || isa<UninitializedVal>(RHS))
      return UninitializedVal();
    
    if (isa<UnknownVal>(LHS) || isa<UnknownVal>(RHS))
      return UnknownVal();
    
    return TF->EvalBinaryOp(ValMgr, Op, LHS, RHS);
  }
  
  inline RValue EvalBinaryOp(BinaryOperator::Opcode Op,
                             LValue LHS, NonLValue RHS) {
    
    if (isa<UninitializedVal>(LHS) || isa<UninitializedVal>(RHS))
      return UninitializedVal();
    
    if (isa<UnknownVal>(LHS) || isa<UnknownVal>(RHS))
      return UnknownVal();
    
    return TF->EvalBinaryOp(ValMgr, Op, LHS, RHS);
  }
  
  inline RValue EvalBinaryOp(BinaryOperator::Opcode Op,
                             RValue LHS, RValue RHS) {
    
    if (isa<UninitializedVal>(LHS) || isa<UninitializedVal>(RHS))
      return UninitializedVal();
    
    if (isa<UnknownVal>(LHS) || isa<UnknownVal>(RHS))
      return UnknownVal();
    
    return TF->EvalBinaryOp(ValMgr, Op, LHS, RHS);
  }
  
  StateTy EvalCall(CallExpr* CE, StateTy St) {
    return St;     
  }
};
} // end clang namespace