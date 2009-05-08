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
//  is built on GRCoreEngine, but provides the boilerplate to execute transfer
//  functions and build the ExplodedGraph at the expression level.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_GREXPRENGINE
#define LLVM_CLANG_ANALYSIS_GREXPRENGINE

#include "clang/Analysis/PathSensitive/GRCoreEngine.h"
#include "clang/Analysis/PathSensitive/GRState.h"
#include "clang/Analysis/PathSensitive/GRSimpleAPICheck.h"
#include "clang/Analysis/PathSensitive/GRTransferFuncs.h"
#include "clang/Analysis/PathSensitive/BugReporter.h"
#include "clang/AST/Type.h"
#include "clang/AST/ExprObjC.h"

namespace clang {  
  
  class PathDiagnosticClient;
  class Diagnostic;
  class ObjCForCollectionStmt;

class GRExprEngine {  
public:
  typedef GRState                  StateTy;
  typedef ExplodedGraph<StateTy>      GraphTy;
  typedef GraphTy::NodeTy             NodeTy;
  
  // Builders.
  typedef GRStmtNodeBuilder<StateTy>          StmtNodeBuilder;
  typedef GRBranchNodeBuilder<StateTy>        BranchNodeBuilder;
  typedef GRIndirectGotoNodeBuilder<StateTy>  IndirectGotoNodeBuilder;
  typedef GRSwitchNodeBuilder<StateTy>        SwitchNodeBuilder;
  typedef GREndPathNodeBuilder<StateTy>       EndPathNodeBuilder;
  typedef ExplodedNodeSet<StateTy>            NodeSet;
    
protected:
  GRCoreEngine<GRExprEngine> CoreEngine;
  
  /// G - the simulation graph.
  GraphTy& G;
  
  /// Liveness - live-variables information the ValueDecl* and block-level
  ///  Expr* in the CFG.  Used to prune out dead state.
  LiveVariables& Liveness;

  /// Builder - The current GRStmtNodeBuilder which is used when building the
  ///  nodes for a given statement.
  StmtNodeBuilder* Builder;
  
  /// StateMgr - Object that manages the data for all created states.
  GRStateManager StateMgr;

  /// SymMgr - Object that manages the symbol information.
  SymbolManager& SymMgr;
  
  /// ValMgr - Object that manages/creates SVals.
  ValueManager &ValMgr;
  
  /// EntryNode - The immediate predecessor node.
  NodeTy* EntryNode;

  /// CleanedState - The state for EntryNode "cleaned" of all dead
  ///  variables and symbols (as determined by a liveness analysis).
  const GRState* CleanedState;  
  
  /// CurrentStmt - The current block-level statement.
  Stmt* CurrentStmt;
  
  // Obj-C Class Identifiers.
  IdentifierInfo* NSExceptionII;
  
  // Obj-C Selectors.
  Selector* NSExceptionInstanceRaiseSelectors;
  Selector RaiseSel;
  
  llvm::OwningPtr<GRSimpleAPICheck> BatchAuditor;

  /// PurgeDead - Remove dead bindings before processing a statement.
  bool PurgeDead;
  
  /// BR - The BugReporter associated with this engine.  It is important that
  //   this object be placed at the very end of member variables so that its
  //   destructor is called before the rest of the GRExprEngine is destroyed.
  GRBugReporter BR;
  
  /// EargerlyAssume - A flag indicating how the engine should handle
  //   expressions such as: 'x = (y != 0)'.  When this flag is true then
  //   the subexpression 'y != 0' will be eagerly assumed to be true or false,
  //   thus evaluating it to the integers 0 or 1 respectively.  The upside
  //   is that this can increase analysis precision until we have a better way
  //   to lazily evaluate such logic.  The downside is that it eagerly
  //   bifurcates paths.
  const bool EagerlyAssume;

public:
  typedef llvm::SmallPtrSet<NodeTy*,2> ErrorNodes;  
  typedef llvm::DenseMap<NodeTy*, Expr*> UndefArgsTy;
  
  /// NilReceiverStructRetExplicit - Nodes in the ExplodedGraph that resulted
  ///  from [x ...] with 'x' definitely being nil and the result was a 'struct'
  //  (an undefined value).
  ErrorNodes NilReceiverStructRetExplicit;
  
  /// NilReceiverStructRetImplicit - Nodes in the ExplodedGraph that resulted
  ///  from [x ...] with 'x' possibly being nil and the result was a 'struct'
  //  (an undefined value).
  ErrorNodes NilReceiverStructRetImplicit;
  
  /// NilReceiverLargerThanVoidPtrRetExplicit - Nodes in the ExplodedGraph that
  /// resulted from [x ...] with 'x' definitely being nil and the result's size
  // was larger than sizeof(void *) (an undefined value).
  ErrorNodes NilReceiverLargerThanVoidPtrRetExplicit;

  /// NilReceiverLargerThanVoidPtrRetImplicit - Nodes in the ExplodedGraph that
  /// resulted from [x ...] with 'x' possibly being nil and the result's size
  // was larger than sizeof(void *) (an undefined value).
  ErrorNodes NilReceiverLargerThanVoidPtrRetImplicit;
  
  /// RetsStackAddr - Nodes in the ExplodedGraph that result from returning
  ///  the address of a stack variable.
  ErrorNodes RetsStackAddr;

  /// RetsUndef - Nodes in the ExplodedGraph that result from returning
  ///  an undefined value.
  ErrorNodes RetsUndef;
  
  /// UndefBranches - Nodes in the ExplodedGraph that result from
  ///  taking a branch based on an undefined value.
  ErrorNodes UndefBranches;
  
  /// UndefStores - Sinks in the ExplodedGraph that result from
  ///  making a store to an undefined lvalue.
  ErrorNodes UndefStores;
  
  /// NoReturnCalls - Sinks in the ExplodedGraph that result from
  //  calling a function with the attribute "noreturn".
  ErrorNodes NoReturnCalls;
  
  /// ImplicitNullDeref - Nodes in the ExplodedGraph that result from
  ///  taking a dereference on a symbolic pointer that MAY be NULL.
  ErrorNodes ImplicitNullDeref;
    
  /// ExplicitNullDeref - Nodes in the ExplodedGraph that result from
  ///  taking a dereference on a symbolic pointer that MUST be NULL.
  ErrorNodes ExplicitNullDeref;
  
  /// UnitDeref - Nodes in the ExplodedGraph that result from
  ///  taking a dereference on an undefined value.
  ErrorNodes UndefDeref;

  /// ImplicitBadDivides - Nodes in the ExplodedGraph that result from 
  ///  evaluating a divide or modulo operation where the denominator
  ///  MAY be zero.
  ErrorNodes ImplicitBadDivides;
  
  /// ExplicitBadDivides - Nodes in the ExplodedGraph that result from 
  ///  evaluating a divide or modulo operation where the denominator
  ///  MUST be zero or undefined.
  ErrorNodes ExplicitBadDivides;
  
  /// ImplicitBadSizedVLA - Nodes in the ExplodedGraph that result from 
  ///  constructing a zero-sized VLA where the size may be zero.
  ErrorNodes ImplicitBadSizedVLA;
  
  /// ExplicitBadSizedVLA - Nodes in the ExplodedGraph that result from 
  ///  constructing a zero-sized VLA where the size must be zero.
  ErrorNodes ExplicitBadSizedVLA;
  
  /// UndefResults - Nodes in the ExplodedGraph where the operands are defined
  ///  by the result is not.  Excludes divide-by-zero errors.
  ErrorNodes UndefResults;
  
  /// BadCalls - Nodes in the ExplodedGraph resulting from calls to function
  ///  pointers that are NULL (or other constants) or Undefined.
  ErrorNodes BadCalls;
  
  /// UndefReceiver - Nodes in the ExplodedGraph resulting from message
  ///  ObjC message expressions where the receiver is undefined (uninitialized).
  ErrorNodes UndefReceivers;
  
  /// UndefArg - Nodes in the ExplodedGraph resulting from calls to functions
  ///   where a pass-by-value argument has an undefined value.
  UndefArgsTy UndefArgs;
  
  /// MsgExprUndefArgs - Nodes in the ExplodedGraph resulting from
  ///   message expressions where a pass-by-value argument has an undefined
  ///  value.
  UndefArgsTy MsgExprUndefArgs;

  /// OutOfBoundMemAccesses - Nodes in the ExplodedGraph resulting from
  /// out-of-bound memory accesses where the index MAY be out-of-bound.
  ErrorNodes ImplicitOOBMemAccesses;

  /// OutOfBoundMemAccesses - Nodes in the ExplodedGraph resulting from
  /// out-of-bound memory accesses where the index MUST be out-of-bound.
  ErrorNodes ExplicitOOBMemAccesses;
  
public:
  GRExprEngine(CFG& cfg, Decl& CD, ASTContext& Ctx, LiveVariables& L,
               BugReporterData& BRD,
               bool purgeDead, bool eagerlyAssume = true,
               StoreManagerCreator SMC = CreateBasicStoreManager,
               ConstraintManagerCreator CMC = CreateBasicConstraintManager);

  ~GRExprEngine();
  
  void ExecuteWorkList(unsigned Steps = 150000) {
    CoreEngine.ExecuteWorkList(Steps);
  }
  
  /// getContext - Return the ASTContext associated with this analysis.
  ASTContext& getContext() const { return G.getContext(); }
  
  /// getCFG - Returns the CFG associated with this analysis.
  CFG& getCFG() { return G.getCFG(); }
  
  GRTransferFuncs& getTF() { return *StateMgr.TF; }
  
  BugReporter& getBugReporter() { return BR; }
  
  /// setTransferFunctions
  void setTransferFunctions(GRTransferFuncs* tf);

  void setTransferFunctions(GRTransferFuncs& tf) {
    setTransferFunctions(&tf);
  }
  
  /// ViewGraph - Visualize the ExplodedGraph created by executing the
  ///  simulation.
  void ViewGraph(bool trim = false);
  
  void ViewGraph(NodeTy** Beg, NodeTy** End);
  
  /// getLiveness - Returned computed live-variables information for the
  ///  analyzed function.  
  const LiveVariables& getLiveness() const { return Liveness; }  
  LiveVariables& getLiveness() { return Liveness; }
  
  /// getInitialState - Return the initial state used for the root vertex
  ///  in the ExplodedGraph.
  const GRState* getInitialState();
  
  GraphTy& getGraph() { return G; }
  const GraphTy& getGraph() const { return G; }

  void RegisterInternalChecks();
  
  bool isRetStackAddr(const NodeTy* N) const {
    return N->isSink() && RetsStackAddr.count(const_cast<NodeTy*>(N)) != 0;
  }
  
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
      (UndefArgs.find(const_cast<NodeTy*>(N)) != UndefArgs.end() ||
       MsgExprUndefArgs.find(const_cast<NodeTy*>(N)) != MsgExprUndefArgs.end());
  }
  
  bool isUndefReceiver(const NodeTy* N) const {
    return N->isSink() && UndefReceivers.count(const_cast<NodeTy*>(N)) != 0;
  }
  
  typedef ErrorNodes::iterator ret_stackaddr_iterator;
  ret_stackaddr_iterator ret_stackaddr_begin() { return RetsStackAddr.begin(); }
  ret_stackaddr_iterator ret_stackaddr_end() { return RetsStackAddr.end(); }  
  
  typedef ErrorNodes::iterator ret_undef_iterator;
  ret_undef_iterator ret_undef_begin() { return RetsUndef.begin(); }
  ret_undef_iterator ret_undef_end() { return RetsUndef.end(); }
  
  typedef ErrorNodes::iterator undef_branch_iterator;
  undef_branch_iterator undef_branches_begin() { return UndefBranches.begin(); }
  undef_branch_iterator undef_branches_end() { return UndefBranches.end(); }  
  
  typedef ErrorNodes::iterator null_deref_iterator;
  null_deref_iterator null_derefs_begin() { return ExplicitNullDeref.begin(); }
  null_deref_iterator null_derefs_end() { return ExplicitNullDeref.end(); }
  
  null_deref_iterator implicit_null_derefs_begin() {
    return ImplicitNullDeref.begin();
  }
  null_deref_iterator implicit_null_derefs_end() {
    return ImplicitNullDeref.end();
  }
  
  typedef ErrorNodes::iterator nil_receiver_struct_ret_iterator;
  
  nil_receiver_struct_ret_iterator nil_receiver_struct_ret_begin() {
    return NilReceiverStructRetExplicit.begin();
  }

  nil_receiver_struct_ret_iterator nil_receiver_struct_ret_end() {
    return NilReceiverStructRetExplicit.end();
  }
  
  typedef ErrorNodes::iterator nil_receiver_larger_than_voidptr_ret_iterator;
  
  nil_receiver_larger_than_voidptr_ret_iterator
  nil_receiver_larger_than_voidptr_ret_begin() {
    return NilReceiverLargerThanVoidPtrRetExplicit.begin();
  }

  nil_receiver_larger_than_voidptr_ret_iterator
  nil_receiver_larger_than_voidptr_ret_end() {
    return NilReceiverLargerThanVoidPtrRetExplicit.end();
  }
  
  typedef ErrorNodes::iterator undef_deref_iterator;
  undef_deref_iterator undef_derefs_begin() { return UndefDeref.begin(); }
  undef_deref_iterator undef_derefs_end() { return UndefDeref.end(); }
  
  typedef ErrorNodes::iterator bad_divide_iterator;

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
  
  typedef ErrorNodes::iterator undef_result_iterator;
  undef_result_iterator undef_results_begin() { return UndefResults.begin(); }
  undef_result_iterator undef_results_end() { return UndefResults.end(); }

  typedef ErrorNodes::iterator bad_calls_iterator;
  bad_calls_iterator bad_calls_begin() { return BadCalls.begin(); }
  bad_calls_iterator bad_calls_end() { return BadCalls.end(); }  
  
  typedef UndefArgsTy::iterator undef_arg_iterator;
  undef_arg_iterator undef_arg_begin() { return UndefArgs.begin(); }
  undef_arg_iterator undef_arg_end() { return UndefArgs.end(); }  
  
  undef_arg_iterator msg_expr_undef_arg_begin() {
    return MsgExprUndefArgs.begin();
  }
  undef_arg_iterator msg_expr_undef_arg_end() {
    return MsgExprUndefArgs.end();
  }  
  
  typedef ErrorNodes::iterator undef_receivers_iterator;

  undef_receivers_iterator undef_receivers_begin() {
    return UndefReceivers.begin();
  }
  
  undef_receivers_iterator undef_receivers_end() {
    return UndefReceivers.end();
  }

  typedef ErrorNodes::iterator oob_memacc_iterator;
  oob_memacc_iterator implicit_oob_memacc_begin() { 
    return ImplicitOOBMemAccesses.begin();
  }
  oob_memacc_iterator implicit_oob_memacc_end() {
    return ImplicitOOBMemAccesses.end();
  }
  oob_memacc_iterator explicit_oob_memacc_begin() {
    return ExplicitOOBMemAccesses.begin();
  }
  oob_memacc_iterator explicit_oob_memacc_end() {
    return ExplicitOOBMemAccesses.end();
  }

  void AddCheck(GRSimpleAPICheck* A, Stmt::StmtClass C);
  void AddCheck(GRSimpleAPICheck* A);
  
  /// ProcessStmt - Called by GRCoreEngine. Used to generate new successor
  ///  nodes by processing the 'effects' of a block-level statement.  
  void ProcessStmt(Stmt* S, StmtNodeBuilder& builder);    
  
  /// ProcessBlockEntrance - Called by GRCoreEngine when start processing
  ///  a CFGBlock.  This method returns true if the analysis should continue
  ///  exploring the given path, and false otherwise.
  bool ProcessBlockEntrance(CFGBlock* B, const GRState* St,
                            GRBlockCounter BC);
  
  /// ProcessBranch - Called by GRCoreEngine.  Used to generate successor
  ///  nodes by processing the 'effects' of a branch condition.
  void ProcessBranch(Stmt* Condition, Stmt* Term, BranchNodeBuilder& builder);
  
  /// ProcessIndirectGoto - Called by GRCoreEngine.  Used to generate successor
  ///  nodes by processing the 'effects' of a computed goto jump.
  void ProcessIndirectGoto(IndirectGotoNodeBuilder& builder);
  
  /// ProcessSwitch - Called by GRCoreEngine.  Used to generate successor
  ///  nodes by processing the 'effects' of a switch statement.
  void ProcessSwitch(SwitchNodeBuilder& builder);
  
  /// ProcessEndPath - Called by GRCoreEngine.  Used to generate end-of-path
  ///  nodes when the control reaches the end of a function.
  void ProcessEndPath(EndPathNodeBuilder& builder) {
    getTF().EvalEndPath(*this, builder);
    StateMgr.EndPath(builder.getState());
  }
  
  GRStateManager& getStateManager() { return StateMgr; }
  const GRStateManager& getStateManager() const { return StateMgr; }

  StoreManager& getStoreManager() { return StateMgr.getStoreManager(); }
  
  ConstraintManager& getConstraintManager() {
    return StateMgr.getConstraintManager();
  }
  
  // FIXME: Remove when we migrate over to just using ValueManager.
  BasicValueFactory& getBasicVals() {
    return StateMgr.getBasicVals();
  }
  const BasicValueFactory& getBasicVals() const {
    return StateMgr.getBasicVals();
  }
  
  ValueManager &getValueManager() { return ValMgr; }  
  const ValueManager &getValueManager() const { return ValMgr; }
  
  // FIXME: Remove when we migrate over to just using ValueManager.
  SymbolManager& getSymbolManager() { return SymMgr; }
  const SymbolManager& getSymbolManager() const { return SymMgr; }
  
protected:
  
  const GRState* GetState(NodeTy* N) {
    return N == EntryNode ? CleanedState : N->getState();
  }
  
public:
  
  const GRState* BindExpr(const GRState* St, Expr* Ex, SVal V) {
    return StateMgr.BindExpr(St, Ex, V);
  }
  
  const GRState* BindExpr(const GRState* St, const Expr* Ex, SVal V) {
    return BindExpr(St, const_cast<Expr*>(Ex), V);
  }
    
protected:
 
  const GRState* BindBlkExpr(const GRState* St, Expr* Ex, SVal V) {
    return StateMgr.BindExpr(St, Ex, V, true, false);
  }
  
  const GRState* BindLoc(const GRState* St, Loc LV, SVal V) {
    return StateMgr.BindLoc(St, LV, V);
  }

  SVal GetSVal(const GRState* St, Stmt* Ex) {
    return StateMgr.GetSVal(St, Ex);
  }
    
  SVal GetSVal(const GRState* St, const Stmt* Ex) {
    return GetSVal(St, const_cast<Stmt*>(Ex));
  }
  
  SVal GetBlkExprSVal(const GRState* St, Stmt* Ex) {
    return StateMgr.GetBlkExprSVal(St, Ex);
  }
    
  SVal GetSVal(const GRState* St, Loc LV, QualType T = QualType()) {    
    return StateMgr.GetSVal(St, LV, T);
  }
  
  inline NonLoc MakeConstantVal(uint64_t X, Expr* Ex) {
    return NonLoc::MakeVal(getBasicVals(), X, Ex->getType());
  }
  
  /// Assume - Create new state by assuming that a given expression
  ///  is true or false.
  const GRState* Assume(const GRState* St, SVal Cond, bool Assumption,
                           bool& isFeasible) {
    return StateMgr.Assume(St, Cond, Assumption, isFeasible);
  }
  
  const GRState* Assume(const GRState* St, Loc Cond, bool Assumption,
                           bool& isFeasible) {
    return StateMgr.Assume(St, Cond, Assumption, isFeasible);
  }

  const GRState* AssumeInBound(const GRState* St, SVal Idx, SVal UpperBound,
                               bool Assumption, bool& isFeasible) {
    return StateMgr.AssumeInBound(St, Idx, UpperBound, Assumption, isFeasible);
  }

public:
  NodeTy* MakeNode(NodeSet& Dst, Stmt* S, NodeTy* Pred, const GRState* St,
                   ProgramPoint::Kind K = ProgramPoint::PostStmtKind,
                   const void *tag = 0);
protected:
    
  /// Visit - Transfer function logic for all statements.  Dispatches to
  ///  other functions that handle specific kinds of statements.
  void Visit(Stmt* S, NodeTy* Pred, NodeSet& Dst);
  
  /// VisitLValue - Evaluate the lvalue of the expression. For example, if Ex is
  /// a DeclRefExpr, it evaluates to the MemRegionVal which represents its
  /// storage location. Note that not all kinds of expressions has lvalue.
  void VisitLValue(Expr* Ex, NodeTy* Pred, NodeSet& Dst);
  
  /// VisitArraySubscriptExpr - Transfer function for array accesses.
  void VisitArraySubscriptExpr(ArraySubscriptExpr* Ex, NodeTy* Pred,
                               NodeSet& Dst, bool asLValue);
  
  /// VisitAsmStmt - Transfer function logic for inline asm.
  void VisitAsmStmt(AsmStmt* A, NodeTy* Pred, NodeSet& Dst);
  
  void VisitAsmStmtHelperOutputs(AsmStmt* A,
                                 AsmStmt::outputs_iterator I,
                                 AsmStmt::outputs_iterator E,
                                 NodeTy* Pred, NodeSet& Dst);
  
  void VisitAsmStmtHelperInputs(AsmStmt* A,
                                AsmStmt::inputs_iterator I,
                                AsmStmt::inputs_iterator E,
                                NodeTy* Pred, NodeSet& Dst);
  
  /// VisitBinaryOperator - Transfer function logic for binary operators.
  void VisitBinaryOperator(BinaryOperator* B, NodeTy* Pred, NodeSet& Dst);

  
  /// VisitCall - Transfer function for function calls.
  void VisitCall(CallExpr* CE, NodeTy* Pred,
                 CallExpr::arg_iterator AI, CallExpr::arg_iterator AE,
                 NodeSet& Dst);
  void VisitCallRec(CallExpr* CE, NodeTy* Pred,
                    CallExpr::arg_iterator AI, CallExpr::arg_iterator AE,
                    NodeSet& Dst, const FunctionProtoType *, 
                    unsigned ParamIdx = 0);
  
  /// VisitCast - Transfer function logic for all casts (implicit and explicit).
  void VisitCast(Expr* CastE, Expr* Ex, NodeTy* Pred, NodeSet& Dst);

  /// VisitCastPointerToInteger - Transfer function (called by VisitCast) that
  ///  handles pointer to integer casts and array to integer casts.
  void VisitCastPointerToInteger(SVal V, const GRState* state, QualType PtrTy,
                                 Expr* CastE, NodeTy* Pred, NodeSet& Dst);
  
  /// VisitCompoundLiteralExpr - Transfer function logic for compound literals.
  void VisitCompoundLiteralExpr(CompoundLiteralExpr* CL, NodeTy* Pred,
                                NodeSet& Dst, bool asLValue);
  
  /// VisitDeclRefExpr - Transfer function logic for DeclRefExprs.
  void VisitDeclRefExpr(DeclRefExpr* DR, NodeTy* Pred, NodeSet& Dst,
                        bool asLValue); 
  
  /// VisitDeclStmt - Transfer function logic for DeclStmts.
  void VisitDeclStmt(DeclStmt* DS, NodeTy* Pred, NodeSet& Dst); 
  
  /// VisitGuardedExpr - Transfer function logic for ?, __builtin_choose
  void VisitGuardedExpr(Expr* Ex, Expr* L, Expr* R, NodeTy* Pred, NodeSet& Dst);

  void VisitInitListExpr(InitListExpr* E, NodeTy* Pred, NodeSet& Dst);

  /// VisitLogicalExpr - Transfer function logic for '&&', '||'
  void VisitLogicalExpr(BinaryOperator* B, NodeTy* Pred, NodeSet& Dst);
  
  /// VisitMemberExpr - Transfer function for member expressions.
  void VisitMemberExpr(MemberExpr* M, NodeTy* Pred, NodeSet& Dst,bool asLValue);
  
  /// VisitObjCIvarRefExpr - Transfer function logic for ObjCIvarRefExprs.
  void VisitObjCIvarRefExpr(ObjCIvarRefExpr* DR, NodeTy* Pred, NodeSet& Dst,
                            bool asLValue); 

  /// VisitObjCForCollectionStmt - Transfer function logic for
  ///  ObjCForCollectionStmt.
  void VisitObjCForCollectionStmt(ObjCForCollectionStmt* S, NodeTy* Pred,
                                  NodeSet& Dst);
  
  void VisitObjCForCollectionStmtAux(ObjCForCollectionStmt* S, NodeTy* Pred,
                                     NodeSet& Dst, SVal ElementV);
  
  /// VisitObjCMessageExpr - Transfer function for ObjC message expressions.
  void VisitObjCMessageExpr(ObjCMessageExpr* ME, NodeTy* Pred, NodeSet& Dst);
  
  void VisitObjCMessageExprArgHelper(ObjCMessageExpr* ME,
                                     ObjCMessageExpr::arg_iterator I,
                                     ObjCMessageExpr::arg_iterator E,
                                     NodeTy* Pred, NodeSet& Dst);
  
  void VisitObjCMessageExprDispatchHelper(ObjCMessageExpr* ME, NodeTy* Pred,
                                          NodeSet& Dst);
  
  /// VisitReturnStmt - Transfer function logic for return statements.
  void VisitReturnStmt(ReturnStmt* R, NodeTy* Pred, NodeSet& Dst);
  
  /// VisitSizeOfAlignOfExpr - Transfer function for sizeof.
  void VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr* Ex, NodeTy* Pred,
                              NodeSet& Dst);
    
  /// VisitUnaryOperator - Transfer function logic for unary operators.
  void VisitUnaryOperator(UnaryOperator* B, NodeTy* Pred, NodeSet& Dst,
                          bool asLValue);
 
  const GRState* CheckDivideZero(Expr* Ex, const GRState* St, NodeTy* Pred,
                                 SVal Denom);  
  
  /// EvalEagerlyAssume - Given the nodes in 'Src', eagerly assume symbolic
  ///  expressions of the form 'x != 0' and generate new nodes (stored in Dst)
  ///  with those assumptions.
  void EvalEagerlyAssume(NodeSet& Dst, NodeSet& Src, Expr *Ex);
  
  SVal EvalCast(SVal X, QualType CastT) {
    if (X.isUnknownOrUndef())
      return X;
    
    if (isa<Loc>(X))
      return getTF().EvalCast(*this, cast<Loc>(X), CastT);
    else
      return getTF().EvalCast(*this, cast<NonLoc>(X), CastT);
  }
  
  SVal EvalMinus(UnaryOperator* U, SVal X) {
    return X.isValid() ? getTF().EvalMinus(*this, U, cast<NonLoc>(X)) : X;
  }
  
  SVal EvalComplement(SVal X) {
    return X.isValid() ? getTF().EvalComplement(*this, cast<NonLoc>(X)) : X;
  }
  
public:
  
  SVal EvalBinOp(BinaryOperator::Opcode Op, NonLoc L, NonLoc R, QualType T) {
    return R.isValid() ? getTF().DetermEvalBinOpNN(*this, Op, L, R, T)
                       : R;
  }

  SVal EvalBinOp(BinaryOperator::Opcode Op, NonLoc L, SVal R, QualType T) {
    return R.isValid() ? getTF().DetermEvalBinOpNN(*this, Op, L,
                                                   cast<NonLoc>(R), T) : R;
  }
  
  void EvalBinOp(ExplodedNodeSet<GRState>& Dst, Expr* Ex,
                 BinaryOperator::Opcode Op, NonLoc L, NonLoc R,
                 ExplodedNode<GRState>* Pred, QualType T);
  
  void EvalBinOp(GRStateSet& OStates, const GRState* St, Expr* Ex,
                 BinaryOperator::Opcode Op, NonLoc L, NonLoc R, QualType T);  
  
  SVal EvalBinOp(BinaryOperator::Opcode Op, SVal L, SVal R, QualType T);
  
protected:
  
  void EvalCall(NodeSet& Dst, CallExpr* CE, SVal L, NodeTy* Pred);
  
  void EvalObjCMessageExpr(NodeSet& Dst, ObjCMessageExpr* ME, NodeTy* Pred) {
    assert (Builder && "GRStmtNodeBuilder must be defined.");
    getTF().EvalObjCMessageExpr(Dst, *this, *Builder, ME, Pred);
  }

  void EvalReturn(NodeSet& Dst, ReturnStmt* s, NodeTy* Pred);
  
  const GRState* MarkBranch(const GRState* St, Stmt* Terminator, 
                            bool branchTaken);
  
  /// EvalBind - Handle the semantics of binding a value to a specific location.
  ///  This method is used by EvalStore, VisitDeclStmt, and others.
  void EvalBind(NodeSet& Dst, Expr* Ex, NodeTy* Pred,
                const GRState* St, SVal location, SVal Val);
  
public:
  void EvalLoad(NodeSet& Dst, Expr* Ex, NodeTy* Pred,
                const GRState* St, SVal location, const void *tag = 0);
  
  NodeTy* EvalLocation(Stmt* Ex, NodeTy* Pred,
                       const GRState* St, SVal location,
                       const void *tag = 0);

  
  void EvalStore(NodeSet& Dst, Expr* E, NodeTy* Pred, const GRState* St,
                 SVal TargetLV, SVal Val, const void *tag = 0);
  
  void EvalStore(NodeSet& Dst, Expr* E, Expr* StoreE, NodeTy* Pred,
                 const GRState* St, SVal TargetLV, SVal Val,
                 const void *tag = 0);
  
};
  
} // end clang namespace

#endif
