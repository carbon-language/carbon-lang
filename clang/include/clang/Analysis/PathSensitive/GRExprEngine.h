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
  typedef ValueState*                 StateTy;
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

  /// BadDivides - Nodes in the ExplodedGraph that result from evaluating
  ///  a divide-by-zero or divide-by-undefined.
  BadDividesTy BadDivides;
  
  /// UndefResults - Nodes in the ExplodedGraph where the operands are defined
  ///  by the result is not.  Excludes divide-by-zero errors.
  UndefResultsTy UndefResults;
  
  /// BadCalls - Nodes in the ExplodedGraph resulting from calls to function
  ///  pointers that are NULL (or other constants) or Undefined.
  BadCallsTy BadCalls;
  
  /// UndefArg - Nodes in the ExplodedGraph resulting from calls to functions
  ///   where a pass-by-value argument has an undefined value.
  UndefArgsTy UndefArgs;
  
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
  
  bool isBadDivide(const NodeTy* N) const {
    return N->isSink() && BadDivides.count(const_cast<NodeTy*>(N)) != 0; 
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
  
  typedef BadDerefTy::iterator null_deref_iterator;
  null_deref_iterator null_derefs_begin() { return ExplicitNullDeref.begin(); }
  null_deref_iterator null_derefs_end() { return ExplicitNullDeref.end(); }
  
  typedef BadDerefTy::iterator undef_deref_iterator;
  undef_deref_iterator undef_derefs_begin() { return UndefDeref.begin(); }
  undef_deref_iterator undef_derefs_end() { return UndefDeref.end(); }
  
  typedef BadDividesTy::iterator bad_divide_iterator;
  bad_divide_iterator bad_divides_begin() { return BadDivides.begin(); }
  bad_divide_iterator bad_divides_end() { return BadDivides.end(); }
  
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
  
protected:
  
  /// RemoveDeadBindings - Return a new state that is the same as 'St' except
  ///  that all subexpression mappings are removed and that any
  ///  block-level expressions that are not live at 'S' also have their
  ///  mappings removed.
  inline ValueState* RemoveDeadBindings(Stmt* S, ValueState* St) {
    return StateMgr.RemoveDeadBindings(St, S, Liveness);
  }
  
  ValueState* SetRVal(ValueState* St, Expr* Ex, RVal V);
  
  ValueState* SetRVal(ValueState* St, const Expr* Ex, RVal V) {
    return SetRVal(St, const_cast<Expr*>(Ex), V);
  }
 
  ValueState* SetBlkExprRVal(ValueState* St, Expr* Ex, RVal V) {
    return StateMgr.SetRVal(St, Ex, V, true, false);
  }
  
#if 0
  /// SetRVal - This version of SetRVal is used to batch process a set
  ///  of different possible RVals and return a set of different states.
  const ValueState*::BufferTy& SetRVal(ValueState* St, Expr* Ex,
                                       const RVal::BufferTy& V,
                                       ValueState*::BufferTy& RetBuf);
#endif
  
  ValueState* SetRVal(ValueState* St, LVal LV, RVal V);
  
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
    return NonLVal::MakeVal(ValMgr, X, Ex->getType(), Ex->getLocStart());
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
  
  NodeTy* Nodify(NodeSet& Dst, Stmt* S, NodeTy* Pred, ValueState* St);
  
#if 0
  /// Nodify - This version of Nodify is used to batch process a set of states.
  ///  The states are not guaranteed to be unique.
  void Nodify(NodeSet& Dst, Stmt* S, NodeTy* Pred,
              const ValueState*::BufferTy& SB);
#endif
  
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
  
  void VisitDeref(UnaryOperator* B, NodeTy* Pred, NodeSet& Dst);
  
  RVal EvalCast(RVal X, QualType CastT) {
    if (X.isUnknownOrUndef())
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
    
    if (L.isUndef() || R.isUndef())
      return UndefinedVal();
    
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
  
  ValueState* EvalCall(CallExpr* CE, LVal L, ValueState* St) {
    return TF->EvalCall(StateMgr, ValMgr, CE, L, St);
  }
  
  ValueState* MarkBranch(ValueState* St, Stmt* Terminator, bool branchTaken);
};
} // end clang namespace

