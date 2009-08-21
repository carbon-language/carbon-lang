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

#include "clang/Analysis/PathSensitive/AnalysisManager.h"
#include "clang/Analysis/PathSensitive/GRSubEngine.h"
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
  class Checker;

class GRExprEngine : public GRSubEngine {  
  AnalysisManager &AMgr;

  GRCoreEngine CoreEngine;
  
  /// G - the simulation graph.
  ExplodedGraph& G;
  
  /// Liveness - live-variables information the ValueDecl* and block-level
  ///  Expr* in the CFG.  Used to prune out dead state.
  LiveVariables& Liveness;

  /// Builder - The current GRStmtNodeBuilder which is used when building the
  ///  nodes for a given statement.
  GRStmtNodeBuilder* Builder;
  
  /// StateMgr - Object that manages the data for all created states.
  GRStateManager StateMgr;

  /// SymMgr - Object that manages the symbol information.
  SymbolManager& SymMgr;
  
  /// ValMgr - Object that manages/creates SVals.
  ValueManager &ValMgr;
  
  /// SVator - SValuator object that creates SVals from expressions.
  SValuator &SVator;
  
  /// EntryNode - The immediate predecessor node.
  ExplodedNode* EntryNode;

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
  std::vector<Checker*> Checkers;

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
  typedef llvm::SmallPtrSet<ExplodedNode*,2> ErrorNodes;  
  typedef llvm::DenseMap<ExplodedNode*, Expr*> UndefArgsTy;
  
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
  GRExprEngine(CFG& cfg, const Decl &CD, ASTContext& Ctx, LiveVariables& L,
               AnalysisManager &mgr,
               bool purgeDead, bool eagerlyAssume = true,
               StoreManagerCreator SMC = CreateBasicStoreManager,
               ConstraintManagerCreator CMC = CreateBasicConstraintManager);

  ~GRExprEngine();
  
  void ExecuteWorkList(const LocationContext *L, unsigned Steps = 150000) {
    CoreEngine.ExecuteWorkList(L, Steps);
  }
  
  /// getContext - Return the ASTContext associated with this analysis.
  ASTContext& getContext() const { return G.getContext(); }
  
  /// getCFG - Returns the CFG associated with this analysis.
  CFG& getCFG() { return G.getCFG(); }
  
  SValuator &getSValuator() { return SVator; }
  
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
  
  void ViewGraph(ExplodedNode** Beg, ExplodedNode** End);
  
  /// getLiveness - Returned computed live-variables information for the
  ///  analyzed function.  
  const LiveVariables& getLiveness() const { return Liveness; }  
  LiveVariables& getLiveness() { return Liveness; }
  
  /// getInitialState - Return the initial state used for the root vertex
  ///  in the ExplodedGraph.
  const GRState* getInitialState(const LocationContext *InitLoc);
  
  ExplodedGraph& getGraph() { return G; }
  const ExplodedGraph& getGraph() const { return G; }

  void RegisterInternalChecks();
  
  void registerCheck(Checker *check) {
    Checkers.push_back(check);
  }
  
  bool isRetStackAddr(const ExplodedNode* N) const {
    return N->isSink() && RetsStackAddr.count(const_cast<ExplodedNode*>(N)) != 0;
  }
  
  bool isUndefControlFlow(const ExplodedNode* N) const {
    return N->isSink() && UndefBranches.count(const_cast<ExplodedNode*>(N)) != 0;
  }
  
  bool isUndefStore(const ExplodedNode* N) const {
    return N->isSink() && UndefStores.count(const_cast<ExplodedNode*>(N)) != 0;
  }
  
  bool isImplicitNullDeref(const ExplodedNode* N) const {
    return N->isSink() && ImplicitNullDeref.count(const_cast<ExplodedNode*>(N)) != 0;
  }
  
  bool isExplicitNullDeref(const ExplodedNode* N) const {
    return N->isSink() && ExplicitNullDeref.count(const_cast<ExplodedNode*>(N)) != 0;
  }
  
  bool isUndefDeref(const ExplodedNode* N) const {
    return N->isSink() && UndefDeref.count(const_cast<ExplodedNode*>(N)) != 0;
  }
  
  bool isImplicitBadDivide(const ExplodedNode* N) const {
    return N->isSink() && ImplicitBadDivides.count(const_cast<ExplodedNode*>(N)) != 0;
  }
  
  bool isExplicitBadDivide(const ExplodedNode* N) const {
    return N->isSink() && ExplicitBadDivides.count(const_cast<ExplodedNode*>(N)) != 0;
  }
  
  bool isNoReturnCall(const ExplodedNode* N) const {
    return N->isSink() && NoReturnCalls.count(const_cast<ExplodedNode*>(N)) != 0;
  }
  
  bool isUndefResult(const ExplodedNode* N) const {
    return N->isSink() && UndefResults.count(const_cast<ExplodedNode*>(N)) != 0;
  }
  
  bool isBadCall(const ExplodedNode* N) const {
    return N->isSink() && BadCalls.count(const_cast<ExplodedNode*>(N)) != 0;
  }
  
  bool isUndefArg(const ExplodedNode* N) const {
    return N->isSink() &&
      (UndefArgs.find(const_cast<ExplodedNode*>(N)) != UndefArgs.end() ||
       MsgExprUndefArgs.find(const_cast<ExplodedNode*>(N)) != MsgExprUndefArgs.end());
  }
  
  bool isUndefReceiver(const ExplodedNode* N) const {
    return N->isSink() && UndefReceivers.count(const_cast<ExplodedNode*>(N)) != 0;
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
  void ProcessStmt(Stmt* S, GRStmtNodeBuilder& builder);    
  
  /// ProcessBlockEntrance - Called by GRCoreEngine when start processing
  ///  a CFGBlock.  This method returns true if the analysis should continue
  ///  exploring the given path, and false otherwise.
  bool ProcessBlockEntrance(CFGBlock* B, const GRState* St,
                            GRBlockCounter BC);
  
  /// ProcessBranch - Called by GRCoreEngine.  Used to generate successor
  ///  nodes by processing the 'effects' of a branch condition.
  void ProcessBranch(Stmt* Condition, Stmt* Term, GRBranchNodeBuilder& builder);
  
  /// ProcessIndirectGoto - Called by GRCoreEngine.  Used to generate successor
  ///  nodes by processing the 'effects' of a computed goto jump.
  void ProcessIndirectGoto(GRIndirectGotoNodeBuilder& builder);
  
  /// ProcessSwitch - Called by GRCoreEngine.  Used to generate successor
  ///  nodes by processing the 'effects' of a switch statement.
  void ProcessSwitch(GRSwitchNodeBuilder& builder);
  
  /// ProcessEndPath - Called by GRCoreEngine.  Used to generate end-of-path
  ///  nodes when the control reaches the end of a function.
  void ProcessEndPath(GREndPathNodeBuilder& builder) {
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
  const GRState* GetState(ExplodedNode* N) {
    return N == EntryNode ? CleanedState : N->getState();
  }
  
public:
  ExplodedNode* MakeNode(ExplodedNodeSet& Dst, Stmt* S, ExplodedNode* Pred, const GRState* St,
                   ProgramPoint::Kind K = ProgramPoint::PostStmtKind,
                   const void *tag = 0);
protected:
  /// CheckerVisit - Dispatcher for performing checker-specific logic
  ///  at specific statements.
  void CheckerVisit(Stmt *S, ExplodedNodeSet &Dst, ExplodedNodeSet &Src, bool isPrevisit);
  
  /// Visit - Transfer function logic for all statements.  Dispatches to
  ///  other functions that handle specific kinds of statements.
  void Visit(Stmt* S, ExplodedNode* Pred, ExplodedNodeSet& Dst);
  
  /// VisitLValue - Evaluate the lvalue of the expression. For example, if Ex is
  /// a DeclRefExpr, it evaluates to the MemRegionVal which represents its
  /// storage location. Note that not all kinds of expressions has lvalue.
  void VisitLValue(Expr* Ex, ExplodedNode* Pred, ExplodedNodeSet& Dst);
  
  /// VisitArraySubscriptExpr - Transfer function for array accesses.
  void VisitArraySubscriptExpr(ArraySubscriptExpr* Ex, ExplodedNode* Pred,
                               ExplodedNodeSet& Dst, bool asLValue);
  
  /// VisitAsmStmt - Transfer function logic for inline asm.
  void VisitAsmStmt(AsmStmt* A, ExplodedNode* Pred, ExplodedNodeSet& Dst);
  
  void VisitAsmStmtHelperOutputs(AsmStmt* A,
                                 AsmStmt::outputs_iterator I,
                                 AsmStmt::outputs_iterator E,
                                 ExplodedNode* Pred, ExplodedNodeSet& Dst);
  
  void VisitAsmStmtHelperInputs(AsmStmt* A,
                                AsmStmt::inputs_iterator I,
                                AsmStmt::inputs_iterator E,
                                ExplodedNode* Pred, ExplodedNodeSet& Dst);
  
  /// VisitBinaryOperator - Transfer function logic for binary operators.
  void VisitBinaryOperator(BinaryOperator* B, ExplodedNode* Pred, ExplodedNodeSet& Dst);

  
  /// VisitCall - Transfer function for function calls.
  void VisitCall(CallExpr* CE, ExplodedNode* Pred,
                 CallExpr::arg_iterator AI, CallExpr::arg_iterator AE,
                 ExplodedNodeSet& Dst);
  void VisitCallRec(CallExpr* CE, ExplodedNode* Pred,
                    CallExpr::arg_iterator AI, CallExpr::arg_iterator AE,
                    ExplodedNodeSet& Dst, const FunctionProtoType *, 
                    unsigned ParamIdx = 0);
  
  /// VisitCast - Transfer function logic for all casts (implicit and explicit).
  void VisitCast(Expr* CastE, Expr* Ex, ExplodedNode* Pred, ExplodedNodeSet& Dst);

  /// VisitCompoundLiteralExpr - Transfer function logic for compound literals.
  void VisitCompoundLiteralExpr(CompoundLiteralExpr* CL, ExplodedNode* Pred,
                                ExplodedNodeSet& Dst, bool asLValue);
  
  /// VisitDeclRefExpr - Transfer function logic for DeclRefExprs.
  void VisitDeclRefExpr(DeclRefExpr* DR, ExplodedNode* Pred, ExplodedNodeSet& Dst,
                        bool asLValue); 
  
  /// VisitDeclStmt - Transfer function logic for DeclStmts.
  void VisitDeclStmt(DeclStmt* DS, ExplodedNode* Pred, ExplodedNodeSet& Dst); 
  
  /// VisitGuardedExpr - Transfer function logic for ?, __builtin_choose
  void VisitGuardedExpr(Expr* Ex, Expr* L, Expr* R, ExplodedNode* Pred, ExplodedNodeSet& Dst);

  void VisitInitListExpr(InitListExpr* E, ExplodedNode* Pred, ExplodedNodeSet& Dst);

  /// VisitLogicalExpr - Transfer function logic for '&&', '||'
  void VisitLogicalExpr(BinaryOperator* B, ExplodedNode* Pred, ExplodedNodeSet& Dst);
  
  /// VisitMemberExpr - Transfer function for member expressions.
  void VisitMemberExpr(MemberExpr* M, ExplodedNode* Pred, ExplodedNodeSet& Dst,bool asLValue);
  
  /// VisitObjCIvarRefExpr - Transfer function logic for ObjCIvarRefExprs.
  void VisitObjCIvarRefExpr(ObjCIvarRefExpr* DR, ExplodedNode* Pred, ExplodedNodeSet& Dst,
                            bool asLValue); 

  /// VisitObjCForCollectionStmt - Transfer function logic for
  ///  ObjCForCollectionStmt.
  void VisitObjCForCollectionStmt(ObjCForCollectionStmt* S, ExplodedNode* Pred,
                                  ExplodedNodeSet& Dst);
  
  void VisitObjCForCollectionStmtAux(ObjCForCollectionStmt* S, ExplodedNode* Pred,
                                     ExplodedNodeSet& Dst, SVal ElementV);
  
  /// VisitObjCMessageExpr - Transfer function for ObjC message expressions.
  void VisitObjCMessageExpr(ObjCMessageExpr* ME, ExplodedNode* Pred, ExplodedNodeSet& Dst);
  
  void VisitObjCMessageExprArgHelper(ObjCMessageExpr* ME,
                                     ObjCMessageExpr::arg_iterator I,
                                     ObjCMessageExpr::arg_iterator E,
                                     ExplodedNode* Pred, ExplodedNodeSet& Dst);
  
  void VisitObjCMessageExprDispatchHelper(ObjCMessageExpr* ME, ExplodedNode* Pred,
                                          ExplodedNodeSet& Dst);
  
  /// VisitReturnStmt - Transfer function logic for return statements.
  void VisitReturnStmt(ReturnStmt* R, ExplodedNode* Pred, ExplodedNodeSet& Dst);
  
  /// VisitSizeOfAlignOfExpr - Transfer function for sizeof.
  void VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr* Ex, ExplodedNode* Pred,
                              ExplodedNodeSet& Dst);
    
  /// VisitUnaryOperator - Transfer function logic for unary operators.
  void VisitUnaryOperator(UnaryOperator* B, ExplodedNode* Pred, ExplodedNodeSet& Dst,
                          bool asLValue);
 
  const GRState* CheckDivideZero(Expr* Ex, const GRState* St, ExplodedNode* Pred,
                                 SVal Denom);  
  
  /// EvalEagerlyAssume - Given the nodes in 'Src', eagerly assume symbolic
  ///  expressions of the form 'x != 0' and generate new nodes (stored in Dst)
  ///  with those assumptions.
  void EvalEagerlyAssume(ExplodedNodeSet& Dst, ExplodedNodeSet& Src, Expr *Ex);
    
  SVal EvalMinus(SVal X) {
    return X.isValid() ? SVator.EvalMinus(cast<NonLoc>(X)) : X;
  }
  
  SVal EvalComplement(SVal X) {
    return X.isValid() ? SVator.EvalComplement(cast<NonLoc>(X)) : X;
  }
  
public:
  
  SVal EvalBinOp(BinaryOperator::Opcode op, NonLoc L, NonLoc R, QualType T) {
    return SVator.EvalBinOpNN(op, L, R, T);
  }

  SVal EvalBinOp(BinaryOperator::Opcode op, NonLoc L, SVal R, QualType T) {
    return R.isValid() ? SVator.EvalBinOpNN(op, L, cast<NonLoc>(R), T) : R;
  }
  
  SVal EvalBinOp(const GRState *state, BinaryOperator::Opcode op,
                 SVal lhs, SVal rhs, QualType T);

protected:
  
  void EvalCall(ExplodedNodeSet& Dst, CallExpr* CE, SVal L, ExplodedNode* Pred);
  
  void EvalObjCMessageExpr(ExplodedNodeSet& Dst, ObjCMessageExpr* ME, ExplodedNode* Pred) {
    assert (Builder && "GRStmtNodeBuilder must be defined.");
    getTF().EvalObjCMessageExpr(Dst, *this, *Builder, ME, Pred);
  }

  void EvalReturn(ExplodedNodeSet& Dst, ReturnStmt* s, ExplodedNode* Pred);
  
  const GRState* MarkBranch(const GRState* St, Stmt* Terminator, 
                            bool branchTaken);
  
  /// EvalBind - Handle the semantics of binding a value to a specific location.
  ///  This method is used by EvalStore, VisitDeclStmt, and others.
  void EvalBind(ExplodedNodeSet& Dst, Expr* Ex, ExplodedNode* Pred,
                const GRState* St, SVal location, SVal Val);
  
public:
  void EvalLoad(ExplodedNodeSet& Dst, Expr* Ex, ExplodedNode* Pred,
                const GRState* St, SVal location, const void *tag = 0);
  
  ExplodedNode* EvalLocation(Stmt* Ex, ExplodedNode* Pred,
                       const GRState* St, SVal location,
                       const void *tag = 0);

  
  void EvalStore(ExplodedNodeSet& Dst, Expr* E, ExplodedNode* Pred, const GRState* St,
                 SVal TargetLV, SVal Val, const void *tag = 0);
  
  void EvalStore(ExplodedNodeSet& Dst, Expr* E, Expr* StoreE, ExplodedNode* Pred,
                 const GRState* St, SVal TargetLV, SVal Val,
                 const void *tag = 0);
  
};
  
} // end clang namespace

#endif
