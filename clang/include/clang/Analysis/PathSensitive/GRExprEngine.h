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
  typedef ValueStateManager::StateTy  StateTy;
  typedef ExplodedGraph<GRExprEngine> GraphTy;
  typedef GraphTy::NodeTy             NodeTy;
  
  // Builders.
  typedef GRStmtNodeBuilder<GRExprEngine>          StmtNodeBuilder;
  typedef GRBranchNodeBuilder<GRExprEngine>        BranchNodeBuilder;
  typedef GRIndirectGotoNodeBuilder<GRExprEngine>  IndirectGotoNodeBuilder;
  typedef GRSwitchNodeBuilder<GRExprEngine>        SwitchNodeBuilder;
  
  class NodeSet {
    typedef llvm::SmallVector<NodeTy*,3> ImplTy;
    ImplTy Impl;
    
  public:

    NodeSet(NodeTy* N) { assert (N && !N->isSink()); Impl.push_back(N); }
    NodeSet() {}
    
    inline void Add(NodeTy* N) { if (N && !N->isSink()) Impl.push_back(N); }
    
    typedef ImplTy::iterator       iterator;
    typedef ImplTy::const_iterator const_iterator;
    
    inline unsigned size() const { return Impl.size();  }
    inline bool empty()    const { return Impl.empty(); }
    
    inline iterator begin() { return Impl.begin(); }
    inline iterator end()   { return Impl.end();   }
    
    inline const_iterator begin() const { return Impl.begin(); }
    inline const_iterator end()   const { return Impl.end();   }
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
  
  /// ValueMgr - Object that manages the data for all created RVals.
  ValueManager& ValMgr;
  
  /// TF - Object that represents a bundle of transfer functions
  ///  for manipulating and creating RVals.
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

  typedef llvm::SmallPtrSet<NodeTy*,5> UninitStoresTy;
  typedef llvm::SmallPtrSet<NodeTy*,5> BadDerefTy;
  typedef llvm::SmallPtrSet<NodeTy*,5> DivZerosTy;
  
  /// UninitStores - Sinks in the ExplodedGraph that result from
  ///  making a store to an uninitialized lvalue.
  UninitStoresTy UninitStores;
  
  /// ImplicitNullDeref - Nodes in the ExplodedGraph that result from
  ///  taking a dereference on a symbolic pointer that MAY be NULL.
  BadDerefTy ImplicitNullDeref;
    
  /// ExplicitNullDeref - Nodes in the ExplodedGraph that result from
  ///  taking a dereference on a symbolic pointer that MUST be NULL.
  BadDerefTy ExplicitNullDeref;
  
  /// UnitDeref - Nodes in the ExplodedGraph that result from
  ///  taking a dereference on an uninitialized value.
  BadDerefTy UninitDeref;

  /// BadDivides - Nodes in the ExplodedGraph that result from evaluating
  ///  a divide-by-zero or divide-by-uninitialized.
  DivZerosTy BadDivides;
  
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
  StateTy getInitialState() { return StateMgr.getInitialState(); }
  
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
  
  bool isBadDivide(const NodeTy* N) const {
    return N->isSink() && BadDivides.count(const_cast<NodeTy*>(N)) != 0; 
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
  
protected:
  
  /// RemoveDeadBindings - Return a new state that is the same as 'St' except
  ///  that all subexpression mappings are removed and that any
  ///  block-level expressions that are not live at 'S' also have their
  ///  mappings removed.
  inline StateTy RemoveDeadBindings(Stmt* S, StateTy St) {
    return StateMgr.RemoveDeadBindings(St, S, Liveness);
  }
  
  StateTy SetRVal(StateTy St, Expr* Ex, const RVal& V);
  
  StateTy SetRVal(StateTy St, const Expr* Ex, const RVal& V) {
    return SetRVal(St, const_cast<Expr*>(Ex), V);
  }
 
  StateTy SetBlkExprRVal(StateTy St, Expr* Ex, const RVal& V) {
    return StateMgr.SetRVal(St, Ex, true, V);
  }
  
  /// SetRVal - This version of SetRVal is used to batch process a set
  ///  of different possible RVals and return a set of different states.
  const StateTy::BufferTy& SetRVal(StateTy St, Expr* Ex,
                                   const RVal::BufferTy& V,
                                   StateTy::BufferTy& RetBuf);
  
  StateTy SetRVal(StateTy St, const LVal& LV, const RVal& V);
  
  RVal GetRVal(const StateTy& St, Expr* Ex) {
    return StateMgr.GetRVal(St, Ex);
  }
    
  RVal GetRVal(const StateTy& St, const Expr* Ex) {
    return GetRVal(St, const_cast<Expr*>(Ex));
  }
  
  RVal GetBlkExprRVal(const StateTy& St, Expr* Ex) {
    return StateMgr.GetBlkExprRVal(St, Ex);
  }  
    
  RVal GetRVal(const StateTy& St, const LVal& LV, QualType T = QualType()) {    
    return StateMgr.GetRVal(St, LV, T);
  }
  
  RVal GetLVal(const StateTy& St, Expr* Ex) {
    return StateMgr.GetLVal(St, Ex);
  }
  
  StateTy Symbolicate(StateTy St, VarDecl* VD) {
    lval::DeclVal X(VD);
    
    if (GetRVal(St, X).isUnknown()) {
      return SetRVal(St, lval::DeclVal(VD), RVal::GetSymbolValue(SymMgr, VD));
    }
    else return St;
  }
  
  inline NonLVal MakeConstantVal(uint64_t X, Expr* Ex) {
    return NonLVal::MakeVal(ValMgr, X, Ex->getType(), Ex->getLocStart());
  }
  
  /// Assume - Create new state by assuming that a given expression
  ///  is true or false.
  StateTy Assume(StateTy St, RVal Cond, bool Assumption, bool& isFeasible) {
    
    if (Cond.isUnknown()) {
      isFeasible = true;
      return St;
    }
    
    if (isa<LVal>(Cond))
      return Assume(St, cast<LVal>(Cond), Assumption, isFeasible);
    else
      return Assume(St, cast<NonLVal>(Cond), Assumption, isFeasible);
  }
  
  StateTy Assume(StateTy St, LVal Cond, bool Assumption, bool& isFeasible);
  StateTy Assume(StateTy St, NonLVal Cond, bool Assumption, bool& isFeasible);
  
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
  ///  a store to an "uninitialized" LVal.
  void HandleUninitializedStore(Stmt* S, NodeTy* Pred);
  
  /// Visit - Transfer function logic for all statements.  Dispatches to
  ///  other functions that handle specific kinds of statements.
  void Visit(Stmt* S, NodeTy* Pred, NodeSet& Dst);
  
  /// VisitBinaryOperator - Transfer function logic for binary operators.
  void VisitBinaryOperator(BinaryOperator* B, NodeTy* Pred, NodeSet& Dst);
  
  void VisitLVal(Expr* Ex, NodeTy* Pred, NodeSet& Dst);
  
  /// VisitCall - Transfer function for function calls.
  void VisitCall(CallExpr* CE, NodeTy* Pred,
                 CallExpr::arg_iterator AI, CallExpr::arg_iterator AE,
                 NodeSet& Dst);
  
  /// VisitCast - Transfer function logic for all casts (implicit and explicit).
  void VisitCast(Expr* CastE, Expr* Ex, NodeTy* Pred, NodeSet& Dst);  
  
  /// VisitDeclRefExpr - Transfer function logic for DeclRefExprs.
  void VisitDeclRefExpr(DeclRefExpr* DR, NodeTy* Pred, NodeSet& Dst); 
  
  /// VisitDeclStmt - Transfer function logic for DeclStmts.
  void VisitDeclStmt(DeclStmt* DS, NodeTy* Pred, NodeSet& Dst); 
  
  /// VisitGuardedExpr - Transfer function logic for ?, __builtin_choose
  void VisitGuardedExpr(Expr* Ex, Expr* L, Expr* R, NodeTy* Pred, NodeSet& Dst);
  
  /// VisitLogicalExpr - Transfer function logic for '&&', '||'
  void VisitLogicalExpr(BinaryOperator* B, NodeTy* Pred, NodeSet& Dst);
  
  /// VisitSizeOfAlignOfTypeExpr - Transfer function for sizeof(type).
  void VisitSizeOfAlignOfTypeExpr(SizeOfAlignOfTypeExpr* Ex, NodeTy* Pred,
                                  NodeSet& Dst);
  
  // VisitSizeOfExpr - Transfer function for sizeof(expr).
  void VisitSizeOfExpr(UnaryOperator* U, NodeTy* Pred, NodeSet& Dst);
  
  /// VisitUnaryOperator - Transfer function logic for unary operators.
  void VisitUnaryOperator(UnaryOperator* B, NodeTy* Pred, NodeSet& Dst);
  
  void VisitDeref(UnaryOperator* B, NodeTy* Pred, NodeSet& Dst);
  
  RVal EvalCast(RVal X, QualType CastT) {
    if (X.isUnknownOrUninit())
      return X;
    
    if (isa<LVal>(X))
      return TF->EvalCast(ValMgr, cast<LVal>(X), CastT);
    else
      return TF->EvalCast(ValMgr, cast<NonLVal>(X), CastT);
  }
  
  RVal EvalMinus(UnaryOperator* U, RVal X) {
    return X.isValid() ? TF->EvalMinus(ValMgr, U, cast<NonLVal>(X)) : X;
  }
  
  RVal EvalComplement(RVal X) {
    return X.isValid() ? TF->EvalComplement(ValMgr, cast<NonLVal>(X)) : X;
  }

  RVal EvalBinOp(BinaryOperator::Opcode Op, NonLVal L, RVal R) {
    return R.isValid() ? TF->EvalBinOp(ValMgr, Op, L, cast<NonLVal>(R)) : R;
  }
  
  RVal EvalBinOp(BinaryOperator::Opcode Op, NonLVal L, NonLVal R) {
    return R.isValid() ? TF->EvalBinOp(ValMgr, Op, L, R) : R;
  }
  
  RVal EvalBinOp(BinaryOperator::Opcode Op, RVal L, RVal R) {
    
    if (L.isUninit() || R.isUninit())
      return UninitializedVal();
    
    if (L.isUnknown() || R.isUnknown())
      return UnknownVal();
        
    if (isa<LVal>(L)) {
      if (isa<LVal>(R))
        return TF->EvalBinOp(ValMgr, Op, cast<LVal>(L), cast<LVal>(R));
      else
        return TF->EvalBinOp(ValMgr, Op, cast<LVal>(L), cast<NonLVal>(R));
    }
    
    return TF->EvalBinOp(ValMgr, Op, cast<NonLVal>(L), cast<NonLVal>(R));
  }
  
  StateTy EvalCall(CallExpr* CE, LVal L, StateTy St) {
    return St;     
  }
  
  StateTy MarkBranch(StateTy St, Stmt* Terminator, bool branchTaken);
};
} // end clang namespace