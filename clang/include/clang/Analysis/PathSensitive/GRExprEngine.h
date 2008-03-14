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

#include "clang/Analysis/PathSensitive/GRCoreEngine.h"
#include "clang/Analysis/PathSensitive/ValueState.h"
#include "clang/Analysis/PathSensitive/GRTransferFuncs.h"

namespace clang {
  
class GRExprEngine {
  
public:
  typedef ValueState                  StateTy;
  typedef ExplodedGraph<GRExprEngine> GraphTy;
  typedef GraphTy::NodeTy             NodeTy;
  
  // Builders.
  typedef GRStmtNodeBuilder<StateTy>               StmtNodeBuilder;
  typedef GRBranchNodeBuilder<GRExprEngine>        BranchNodeBuilder;
  typedef GRIndirectGotoNodeBuilder<GRExprEngine>  IndirectGotoNodeBuilder;
  typedef GRSwitchNodeBuilder<GRExprEngine>        SwitchNodeBuilder;
  typedef ExplodedNodeSet<StateTy>                 NodeSet;
    
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
  BasicValueFactory& BasicVals;
  
  /// TF - Object that represents a bundle of transfer functions
  ///  for manipulating and creating RVals.
  GRTransferFuncs* TF;
  
  /// SymMgr - Object that manages the symbol information.
  SymbolManager& SymMgr;
  
  /// StmtEntryNode - The immediate predecessor node.
  NodeTy* StmtEntryNode;
  
  /// CleanedState - The state for StmtEntryNode "cleaned" of all dead
  ///  variables and symbols (as determined by a liveness analysis).
  ValueState* CleanedState;
  
  /// CurrentStmt - The current block-level statement.
  Stmt* CurrentStmt;

  typedef llvm::SmallPtrSet<NodeTy*,2> UndefBranchesTy;  
  typedef llvm::SmallPtrSet<NodeTy*,2> UndefStoresTy;
  typedef llvm::SmallPtrSet<NodeTy*,2> BadDerefTy;
  typedef llvm::SmallPtrSet<NodeTy*,2> BadCallsTy;
  typedef llvm::DenseMap<NodeTy*, Expr*> UndefArgsTy;
  typedef llvm::SmallPtrSet<NodeTy*,2> BadDividesTy;
  typedef llvm::SmallPtrSet<NodeTy*,2> NoReturnCallsTy;  
  typedef llvm::SmallPtrSet<NodeTy*,2> UndefResultsTy;  

  
  /// UndefBranches - Nodes in the ExplodedGraph that result from
  ///  taking a branch based on an undefined value.
  UndefBranchesTy UndefBranches;
  
  /// UndefStores - Sinks in the ExplodedGraph that result from
  ///  making a store to an undefined lvalue.
  UndefStoresTy UndefStores;
  
  /// NoReturnCalls - Sinks in the ExplodedGraph that result from
  //  calling a function with the attribute "noreturn".
  NoReturnCallsTy NoReturnCalls;
  
  /// ImplicitNullDeref - Nodes in the ExplodedGraph that result from
  ///  taking a dereference on a symbolic pointer that MAY be NULL.
  BadDerefTy ImplicitNullDeref;
    
  /// ExplicitNullDeref - Nodes in the ExplodedGraph that result from
  ///  taking a dereference on a symbolic pointer that MUST be NULL.
  BadDerefTy ExplicitNullDeref;
  
  /// UnitDeref - Nodes in the ExplodedGraph that result from
  ///  taking a dereference on an undefined value.
  BadDerefTy UndefDeref;

  /// ImplicitBadDivides - Nodes in the ExplodedGraph that result from 
  ///  evaluating a divide or modulo operation where the denominator
  ///  MAY be zero.
  BadDividesTy ImplicitBadDivides;
  
  /// ExplicitBadDivides - Nodes in the ExplodedGraph that result from 
  ///  evaluating a divide or modulo operation where the denominator
  ///  MUST be zero or undefined.
  BadDividesTy ExplicitBadDivides;
  
  /// UndefResults - Nodes in the ExplodedGraph where the operands are defined
  ///  by the result is not.  Excludes divide-by-zero errors.
  UndefResultsTy UndefResults;
  
  /// BadCalls - Nodes in the ExplodedGraph resulting from calls to function
  ///  pointers that are NULL (or other constants) or Undefined.
  BadCallsTy BadCalls;
  
  /// UndefArg - Nodes in the ExplodedGraph resulting from calls to functions
  ///   where a pass-by-value argument has an undefined value.
  UndefArgsTy UndefArgs;
  
public:
  GRExprEngine(GraphTy& g) : 
    G(g), Liveness(G.getCFG()),
    Builder(NULL),
    StateMgr(G.getContext(), G.getAllocator()),
    BasicVals(StateMgr.getBasicValueFactory()),
    TF(NULL), // FIXME.
    SymMgr(StateMgr.getSymbolManager()),
    StmtEntryNode(NULL), CleanedState(NULL), CurrentStmt(NULL) {
    
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
  void ViewGraph(bool trim = false);
  
  void ViewGraph(NodeTy** Beg, NodeTy** End);
  
  /// getInitialState - Return the initial state used for the root vertex
  ///  in the ExplodedGraph.
  ValueState* getInitialState();
  
  bool isUndefControlFlow(const NodeTy* N) const {
    return N->isSink() && UndefBranches.count(const_cast<NodeTy*>(N)) != 0;
  }
  
  bool isUndefStore(const NodeTy* N) const {
    return N->isSink() && UndefStores.count(const_cast<NodeTy*>(N)) != 0;
  }
  
  bool isImplicitNullDeref(const NodeTy* N) const {
    return N->isSink() && ImplicitNullDeref.count(const_cast<NodeTy*>(N)) != 0;
  }
  
  bool isExplicitNullDeref(const NodeTy* N) const {
    return N->isSink() && ExplicitNullDeref.count(const_cast<NodeTy*>(N)) != 0;
  }
  
  bool isUndefDeref(const NodeTy* N) const {
    return N->isSink() && UndefDeref.count(const_cast<NodeTy*>(N)) != 0;
  }
  
  bool isImplicitBadDivide(const NodeTy* N) const {
    return N->isSink() && ImplicitBadDivides.count(const_cast<NodeTy*>(N)) != 0; 
  }
  
  bool isExplicitBadDivide(const NodeTy* N) const {
    return N->isSink() && ExplicitBadDivides.count(const_cast<NodeTy*>(N)) != 0; 
  }
  
  bool isNoReturnCall(const NodeTy* N) const {
    return N->isSink() && NoReturnCalls.count(const_cast<NodeTy*>(N)) != 0;
  }
  
  bool isUndefResult(const NodeTy* N) const {
    return N->isSink() && UndefResults.count(const_cast<NodeTy*>(N)) != 0;
  }
  
  bool isBadCall(const NodeTy* N) const {
    return N->isSink() && BadCalls.count(const_cast<NodeTy*>(N)) != 0;
  }
  
  bool isUndefArg(const NodeTy* N) const {
    return N->isSink() &&
           UndefArgs.find(const_cast<NodeTy*>(N)) != UndefArgs.end();
  }
  
  typedef UndefBranchesTy::iterator undef_branch_iterator;
  undef_branch_iterator undef_branches_begin() { return UndefBranches.begin(); }
  undef_branch_iterator undef_branches_end() { return UndefBranches.end(); }  
  
  typedef BadDerefTy::iterator null_deref_iterator;
  null_deref_iterator null_derefs_begin() { return ExplicitNullDeref.begin(); }
  null_deref_iterator null_derefs_end() { return ExplicitNullDeref.end(); }
  
  typedef BadDerefTy::iterator undef_deref_iterator;
  undef_deref_iterator undef_derefs_begin() { return UndefDeref.begin(); }
  undef_deref_iterator undef_derefs_end() { return UndefDeref.end(); }
  
  typedef BadDividesTy::iterator bad_divide_iterator;

  bad_divide_iterator explicit_bad_divides_begin() {
    return ExplicitBadDivides.begin();
  }
  
  bad_divide_iterator explicit_bad_divides_end() {
    return ExplicitBadDivides.end();
  }
  
  bad_divide_iterator implicit_bad_divides_begin() {
    return ImplicitBadDivides.begin();
  }
  
  bad_divide_iterator implicit_bad_divides_end() {
    return ImplicitBadDivides.end();
  }
  
  typedef UndefResultsTy::iterator undef_result_iterator;
  undef_result_iterator undef_results_begin() { return UndefResults.begin(); }
  undef_result_iterator undef_results_end() { return UndefResults.end(); }

  typedef BadCallsTy::iterator bad_calls_iterator;
  bad_calls_iterator bad_calls_begin() { return BadCalls.begin(); }
  bad_calls_iterator bad_calls_end() { return BadCalls.end(); }  
  
  typedef UndefArgsTy::iterator undef_arg_iterator;
  undef_arg_iterator undef_arg_begin() { return UndefArgs.begin(); }
  undef_arg_iterator undef_arg_end() { return UndefArgs.end(); }  
  
  /// ProcessStmt - Called by GRCoreEngine. Used to generate new successor
  ///  nodes by processing the 'effects' of a block-level statement.  
  void ProcessStmt(Stmt* S, StmtNodeBuilder& builder);    
  
  /// ProcessBlockEntrance - Called by GRCoreEngine when start processing
  ///  a CFGBlock.  This method returns true if the analysis should continue
  ///  exploring the given path, and false otherwise.
  bool ProcessBlockEntrance(CFGBlock* B, ValueState* St, GRBlockCounter BC);
  
  /// ProcessBranch - Called by GRCoreEngine.  Used to generate successor
  ///  nodes by processing the 'effects' of a branch condition.
  void ProcessBranch(Expr* Condition, Stmt* Term, BranchNodeBuilder& builder);
  
  /// ProcessIndirectGoto - Called by GRCoreEngine.  Used to generate successor
  ///  nodes by processing the 'effects' of a computed goto jump.
  void ProcessIndirectGoto(IndirectGotoNodeBuilder& builder);
  
  /// ProcessSwitch - Called by GRCoreEngine.  Used to generate successor
  ///  nodes by processing the 'effects' of a switch statement.
  void ProcessSwitch(SwitchNodeBuilder& builder);
  
  
  ValueStateManager& getStateManager() { return StateMgr; }
  const ValueStateManager& getStateManger() const { return StateMgr; }
  
  BasicValueFactory& getBasicVals() { return BasicVals; }
  const BasicValueFactory& getBasicVals() const { return BasicVals; }
  
  SymbolManager& getSymbolManager() { return SymMgr; }
  const SymbolManager& getSymbolManager() const { return SymMgr; }
  
protected:
  
  ValueState* GetState(NodeTy* N) {
    return N == StmtEntryNode ? CleanedState : N->getState();
  }
  
  ValueState* SetRVal(ValueState* St, Expr* Ex, RVal V);
  
  ValueState* SetRVal(ValueState* St, const Expr* Ex, RVal V) {
    return SetRVal(St, const_cast<Expr*>(Ex), V);
  }
 
  ValueState* SetBlkExprRVal(ValueState* St, Expr* Ex, RVal V) {
    return StateMgr.SetRVal(St, Ex, V, true, false);
  }
  
  ValueState* SetRVal(ValueState* St, LVal LV, RVal V) {
    return StateMgr.SetRVal(St, LV, V);
  }
  
  RVal GetRVal(ValueState* St, Expr* Ex) {
    return StateMgr.GetRVal(St, Ex);
  }
    
  RVal GetRVal(ValueState* St, const Expr* Ex) {
    return GetRVal(St, const_cast<Expr*>(Ex));
  }
  
  RVal GetBlkExprRVal(ValueState* St, Expr* Ex) {
    return StateMgr.GetBlkExprRVal(St, Ex);
  }  
    
  RVal GetRVal(ValueState* St, LVal LV, QualType T = QualType()) {    
    return StateMgr.GetRVal(St, LV, T);
  }
  
  RVal GetLVal(ValueState* St, Expr* Ex) {
    return StateMgr.GetLVal(St, Ex);
  }
  
  inline NonLVal MakeConstantVal(uint64_t X, Expr* Ex) {
    return NonLVal::MakeVal(BasicVals, X, Ex->getType());
  }
  
  /// Assume - Create new state by assuming that a given expression
  ///  is true or false.
  ValueState* Assume(ValueState* St, RVal Cond, bool Assumption,
                     bool& isFeasible) {
    
    if (Cond.isUnknown()) {
      isFeasible = true;
      return St;
    }
    
    if (isa<LVal>(Cond))
      return Assume(St, cast<LVal>(Cond), Assumption, isFeasible);
    else
      return Assume(St, cast<NonLVal>(Cond), Assumption, isFeasible);
  }
  
  ValueState* Assume(ValueState* St, LVal Cond, bool Assumption,
                     bool& isFeasible);
  
  ValueState* Assume(ValueState* St, NonLVal Cond, bool Assumption,
                     bool& isFeasible);
  
  ValueState* AssumeSymNE(ValueState* St, SymbolID sym, const llvm::APSInt& V,
                          bool& isFeasible);
  
  ValueState* AssumeSymEQ(ValueState* St, SymbolID sym, const llvm::APSInt& V,
                          bool& isFeasible);
  
  ValueState* AssumeSymInt(ValueState* St, bool Assumption,
                           const SymIntConstraint& C, bool& isFeasible);
  
  NodeTy* Nodify(NodeSet& Dst, Stmt* S, NodeTy* Pred, ValueState* St) {
    assert (Builder && "GRStmtNodeBuilder not present.");
    return Builder->Nodify(Dst, S, Pred, St);
  }
  
  /// HandleUndefinedStore - Create the necessary sink node to represent
  ///  a store to an "undefined" LVal.
  void HandleUndefinedStore(Stmt* S, NodeTy* Pred);
  
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
  
  void VisitDeref(UnaryOperator* U, NodeTy* Pred, NodeSet& Dst,
                  bool GetLVal = false);
  
  RVal EvalCast(RVal X, QualType CastT) {
    if (X.isUnknownOrUndef())
      return X;
    
    if (isa<LVal>(X))
      return TF->EvalCast(*this, cast<LVal>(X), CastT);
    else
      return TF->EvalCast(*this, cast<NonLVal>(X), CastT);
  }
  
  RVal EvalMinus(UnaryOperator* U, RVal X) {
    return X.isValid() ? TF->EvalMinus(*this, U, cast<NonLVal>(X)) : X;
  }
  
  RVal EvalComplement(RVal X) {
    return X.isValid() ? TF->EvalComplement(*this, cast<NonLVal>(X)) : X;
  }

  RVal EvalBinOp(BinaryOperator::Opcode Op, NonLVal L, RVal R) {
    return R.isValid() ? TF->EvalBinOp(*this, Op, L, cast<NonLVal>(R)) : R;
  }
  
  RVal EvalBinOp(BinaryOperator::Opcode Op, NonLVal L, NonLVal R) {
    return R.isValid() ? TF->EvalBinOp(*this, Op, L, R) : R;
  }
  
  RVal EvalBinOp(BinaryOperator::Opcode Op, RVal L, RVal R) {
    
    if (L.isUndef() || R.isUndef())
      return UndefinedVal();
    
    if (L.isUnknown() || R.isUnknown())
      return UnknownVal();
        
    if (isa<LVal>(L)) {
      if (isa<LVal>(R))
        return TF->EvalBinOp(*this, Op, cast<LVal>(L), cast<LVal>(R));
      else
        return TF->EvalBinOp(*this, Op, cast<LVal>(L), cast<NonLVal>(R));
    }
    
    if (isa<LVal>(R)) {
      // Support pointer arithmetic where the increment/decrement operand
      // is on the left and the pointer on the right.
      
      assert (Op == BinaryOperator::Add || Op == BinaryOperator::Sub);

      // Commute the operands.      
      return TF->EvalBinOp(*this, Op, cast<LVal>(R), cast<NonLVal>(L));
    }
    else
      return TF->EvalBinOp(*this, Op, cast<NonLVal>(L), cast<NonLVal>(R));
  }
  
  void EvalCall(NodeSet& Dst, CallExpr* CE, LVal L, NodeTy* Pred) {
    assert (Builder && "GRStmtNodeBuilder must be defined.");    
    return TF->EvalCall(Dst, *this, *Builder, CE, L, Pred);
  }
  
  ValueState* MarkBranch(ValueState* St, Stmt* Terminator, bool branchTaken);
};
} // end clang namespace

