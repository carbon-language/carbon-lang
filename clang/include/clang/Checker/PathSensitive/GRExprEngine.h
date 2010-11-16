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

#include "clang/Checker/PathSensitive/AnalysisManager.h"
#include "clang/Checker/PathSensitive/GRSubEngine.h"
#include "clang/Checker/PathSensitive/GRCoreEngine.h"
#include "clang/Checker/PathSensitive/GRState.h"
#include "clang/Checker/PathSensitive/GRTransferFuncs.h"
#include "clang/Checker/BugReporter/BugReporter.h"
#include "clang/AST/Type.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtObjC.h"

namespace clang {
class AnalysisManager;
class Checker;
class ObjCForCollectionStmt;

class GRExprEngine : public GRSubEngine {
  AnalysisManager &AMgr;

  GRCoreEngine CoreEngine;

  /// G - the simulation graph.
  ExplodedGraph& G;

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
  const Stmt* CurrentStmt;

  // Obj-C Class Identifiers.
  IdentifierInfo* NSExceptionII;

  // Obj-C Selectors.
  Selector* NSExceptionInstanceRaiseSelectors;
  Selector RaiseSel;

  enum CallbackKind {
    PreVisitStmtCallback,
    PostVisitStmtCallback,
    ProcessAssumeCallback,
    EvalRegionChangesCallback
  };

  typedef uint32_t CallbackTag;

  /// GetCallbackTag - Create a tag for a certain kind of callback. The 'Sub'
  ///  argument can be used to differentiate callbacks that depend on another
  ///  value from a small set of possibilities, such as statement classes.
  static inline CallbackTag GetCallbackTag(CallbackKind K, uint32_t Sub = 0) {
    assert(Sub == ((Sub << 8) >> 8) && "Tag sub-kind must fit into 24 bits");
    return K | (Sub << 8);
  }

  typedef llvm::DenseMap<void *, unsigned> CheckerMap;
  typedef std::vector<std::pair<void *, Checker*> > CheckersOrdered;
  typedef llvm::DenseMap<CallbackTag, CheckersOrdered *> CheckersOrderedCache;
  
  /// A registration map from checker tag to the index into the
  ///  ordered checkers vector.
  CheckerMap CheckerM;

  /// An ordered vector of checkers that are called when evaluating
  ///  various expressions and statements.
  CheckersOrdered Checkers;

  /// A map used for caching the checkers that respond to the callback for
  ///  a particular callback tag.
  CheckersOrderedCache COCache;

  /// The BugReporter associated with this engine.  It is important that
  ///  this object be placed at the very end of member variables so that its
  ///  destructor is called before the rest of the GRExprEngine is destroyed.
  GRBugReporter BR;
  
  llvm::OwningPtr<GRTransferFuncs> TF;

public:
  GRExprEngine(AnalysisManager &mgr, GRTransferFuncs *tf);

  ~GRExprEngine();

  void ExecuteWorkList(const LocationContext *L, unsigned Steps = 150000) {
    CoreEngine.ExecuteWorkList(L, Steps, 0);
  }

  /// Execute the work list with an initial state. Nodes that reaches the exit
  /// of the function are added into the Dst set, which represent the exit
  /// state of the function call.
  void ExecuteWorkListWithInitialState(const LocationContext *L, unsigned Steps,
                                       const GRState *InitState, 
                                       ExplodedNodeSet &Dst) {
    CoreEngine.ExecuteWorkListWithInitialState(L, Steps, InitState, Dst);
  }

  /// getContext - Return the ASTContext associated with this analysis.
  ASTContext& getContext() const { return AMgr.getASTContext(); }

  virtual AnalysisManager &getAnalysisManager() { return AMgr; }

  SValuator &getSValuator() { return SVator; }

  GRTransferFuncs& getTF() { return *TF; }

  BugReporter& getBugReporter() { return BR; }

  GRStmtNodeBuilder &getBuilder() { assert(Builder); return *Builder; }

  // FIXME: Remove once GRTransferFuncs is no longer referenced.
  void setTransferFunction(GRTransferFuncs* tf);

  /// ViewGraph - Visualize the ExplodedGraph created by executing the
  ///  simulation.
  void ViewGraph(bool trim = false);

  void ViewGraph(ExplodedNode** Beg, ExplodedNode** End);

  /// getInitialState - Return the initial state used for the root vertex
  ///  in the ExplodedGraph.
  const GRState* getInitialState(const LocationContext *InitLoc);

  ExplodedGraph& getGraph() { return G; }
  const ExplodedGraph& getGraph() const { return G; }

  template <typename CHECKER>
  void registerCheck(CHECKER *check) {
    unsigned entry = Checkers.size();
    void *tag = CHECKER::getTag();
    Checkers.push_back(std::make_pair(tag, check));
    CheckerM[tag] = entry;
  }
  
  Checker *lookupChecker(void *tag) const;

  template <typename CHECKER>
  CHECKER *getChecker() const {
     return static_cast<CHECKER*>(lookupChecker(CHECKER::getTag()));
  }

  /// ProcessElement - Called by GRCoreEngine. Used to generate new successor
  ///  nodes by processing the 'effects' of a CFG element.
  void ProcessElement(const CFGElement E, GRStmtNodeBuilder& builder);

  void ProcessStmt(const CFGStmt S, GRStmtNodeBuilder &builder);

  void ProcessInitializer(const CFGInitializer I, GRStmtNodeBuilder &builder);

  void ProcessImplicitDtor(const CFGImplicitDtor D, GRStmtNodeBuilder &builder);

  /// ProcessBlockEntrance - Called by GRCoreEngine when start processing
  ///  a CFGBlock.  This method returns true if the analysis should continue
  ///  exploring the given path, and false otherwise.
  bool ProcessBlockEntrance(const CFGBlock* B, const ExplodedNode *Pred,
                            GRBlockCounter BC);

  /// ProcessBranch - Called by GRCoreEngine.  Used to generate successor
  ///  nodes by processing the 'effects' of a branch condition.
  void ProcessBranch(const Stmt* Condition, const Stmt* Term, 
                     GRBranchNodeBuilder& builder);

  /// ProcessIndirectGoto - Called by GRCoreEngine.  Used to generate successor
  ///  nodes by processing the 'effects' of a computed goto jump.
  void ProcessIndirectGoto(GRIndirectGotoNodeBuilder& builder);

  /// ProcessSwitch - Called by GRCoreEngine.  Used to generate successor
  ///  nodes by processing the 'effects' of a switch statement.
  void ProcessSwitch(GRSwitchNodeBuilder& builder);

  /// ProcessEndPath - Called by GRCoreEngine.  Used to generate end-of-path
  ///  nodes when the control reaches the end of a function.
  void ProcessEndPath(GREndPathNodeBuilder& builder);

  /// Generate the entry node of the callee.
  void ProcessCallEnter(GRCallEnterNodeBuilder &builder);

  /// Generate the first post callsite node.
  void ProcessCallExit(GRCallExitNodeBuilder &builder);

  /// Called by GRCoreEngine when the analysis worklist has terminated.
  void ProcessEndWorklist(bool hasWorkRemaining);

  /// EvalAssume - Callback function invoked by the ConstraintManager when
  ///  making assumptions about state values.
  const GRState *ProcessAssume(const GRState *state, SVal cond,bool assumption);

  /// WantsRegionChangeUpdate - Called by GRStateManager to determine if a
  ///  region change should trigger a ProcessRegionChanges update.
  bool WantsRegionChangeUpdate(const GRState* state);

  /// ProcessRegionChanges - Called by GRStateManager whenever a change is made
  ///  to the store. Used to update checkers that track region values.
  const GRState* ProcessRegionChanges(const GRState *state,
                                      const MemRegion * const *Begin,
                                      const MemRegion * const *End);

  virtual GRStateManager& getStateManager() { return StateMgr; }

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

  // Functions for external checking of whether we have unfinished work
  bool wasBlockAborted() const { return CoreEngine.wasBlockAborted(); }
  bool hasEmptyWorkList() const { return !CoreEngine.getWorkList()->hasWork(); }
  bool hasWorkRemaining() const {
    return wasBlockAborted() || CoreEngine.getWorkList()->hasWork();
  }

  const GRCoreEngine &getCoreEngine() const { return CoreEngine; }

protected:
  const GRState* GetState(ExplodedNode* N) {
    return N == EntryNode ? CleanedState : N->getState();
  }

public:
  ExplodedNode* MakeNode(ExplodedNodeSet& Dst, const Stmt* S, 
                         ExplodedNode* Pred, const GRState* St,
                         ProgramPoint::Kind K = ProgramPoint::PostStmtKind,
                         const void *tag = 0);

  /// CheckerVisit - Dispatcher for performing checker-specific logic
  ///  at specific statements.
  void CheckerVisit(const Stmt *S, ExplodedNodeSet &Dst, ExplodedNodeSet &Src, 
                    CallbackKind Kind);

  bool CheckerEvalCall(const CallExpr *CE, 
                       ExplodedNodeSet &Dst, 
                       ExplodedNode *Pred);

  void CheckerEvalNilReceiver(const ObjCMessageExpr *ME, 
                              ExplodedNodeSet &Dst,
                              const GRState *state,
                              ExplodedNode *Pred);
  
  void CheckerVisitBind(const Stmt *StoreE, ExplodedNodeSet &Dst,
                        ExplodedNodeSet &Src,  SVal location, SVal val,
                        bool isPrevisit);

  /// Visit - Transfer function logic for all statements.  Dispatches to
  ///  other functions that handle specific kinds of statements.
  void Visit(const Stmt* S, ExplodedNode* Pred, ExplodedNodeSet& Dst);

  /// VisitLValue - Evaluate the lvalue of the expression. For example, if Ex is
  /// a DeclRefExpr, it evaluates to the MemRegionVal which represents its
  /// storage location. Note that not all kinds of expressions has lvalue.
  void VisitLValue(const Expr* Ex, ExplodedNode* Pred, ExplodedNodeSet& Dst);

  /// VisitArraySubscriptExpr - Transfer function for array accesses.
  void VisitArraySubscriptExpr(const ArraySubscriptExpr* Ex, ExplodedNode* Pred,
                               ExplodedNodeSet& Dst, bool asLValue);

  /// VisitAsmStmt - Transfer function logic for inline asm.
  void VisitAsmStmt(const AsmStmt* A, ExplodedNode* Pred, ExplodedNodeSet& Dst);

  void VisitAsmStmtHelperOutputs(const AsmStmt* A,
                                 AsmStmt::const_outputs_iterator I,
                                 AsmStmt::const_outputs_iterator E,
                                 ExplodedNode* Pred, ExplodedNodeSet& Dst);

  void VisitAsmStmtHelperInputs(const AsmStmt* A,
                                AsmStmt::const_inputs_iterator I,
                                AsmStmt::const_inputs_iterator E,
                                ExplodedNode* Pred, ExplodedNodeSet& Dst);
  
  /// VisitBlockExpr - Transfer function logic for BlockExprs.
  void VisitBlockExpr(const BlockExpr *BE, ExplodedNode *Pred, 
                      ExplodedNodeSet &Dst);

  /// VisitBinaryOperator - Transfer function logic for binary operators.
  void VisitBinaryOperator(const BinaryOperator* B, ExplodedNode* Pred, 
                           ExplodedNodeSet& Dst, bool asLValue);


  /// VisitCall - Transfer function for function calls.
  void VisitCall(const CallExpr* CE, ExplodedNode* Pred,
                 CallExpr::const_arg_iterator AI, 
                 CallExpr::const_arg_iterator AE,
                 ExplodedNodeSet& Dst, bool asLValue);

  /// VisitCast - Transfer function logic for all casts (implicit and explicit).
  void VisitCast(const CastExpr *CastE, const Expr *Ex, ExplodedNode *Pred,
                 ExplodedNodeSet &Dst, bool asLValue);

  /// VisitCompoundLiteralExpr - Transfer function logic for compound literals.
  void VisitCompoundLiteralExpr(const CompoundLiteralExpr* CL, 
                                ExplodedNode* Pred, ExplodedNodeSet& Dst, 
                                bool asLValue);

  /// VisitDeclRefExpr - Transfer function logic for DeclRefExprs.
  void VisitDeclRefExpr(const DeclRefExpr* DR, ExplodedNode* Pred,
                        ExplodedNodeSet& Dst, bool asLValue);

  /// VisitBlockDeclRefExpr - Transfer function logic for BlockDeclRefExprs.
  void VisitBlockDeclRefExpr(const BlockDeclRefExpr* DR, ExplodedNode* Pred,
                             ExplodedNodeSet& Dst, bool asLValue);
  
  void VisitCommonDeclRefExpr(const Expr* DR, const NamedDecl *D,
                              ExplodedNode* Pred, ExplodedNodeSet& Dst, 
                              bool asLValue);  
  
  /// VisitDeclStmt - Transfer function logic for DeclStmts.
  void VisitDeclStmt(const DeclStmt* DS, ExplodedNode* Pred, 
                     ExplodedNodeSet& Dst);

  /// VisitGuardedExpr - Transfer function logic for ?, __builtin_choose
  void VisitGuardedExpr(const Expr* Ex, const Expr* L, const Expr* R, 
                        ExplodedNode* Pred, ExplodedNodeSet& Dst);

  /// VisitCondInit - Transfer function for handling the initialization
  ///  of a condition variable in an IfStmt, SwitchStmt, etc.
  void VisitCondInit(const VarDecl *VD, const Stmt *S, ExplodedNode *Pred,
                     ExplodedNodeSet& Dst);
  
  void VisitInitListExpr(const InitListExpr* E, ExplodedNode* Pred,
                         ExplodedNodeSet& Dst);

  /// VisitLogicalExpr - Transfer function logic for '&&', '||'
  void VisitLogicalExpr(const BinaryOperator* B, ExplodedNode* Pred,
                        ExplodedNodeSet& Dst);

  /// VisitMemberExpr - Transfer function for member expressions.
  void VisitMemberExpr(const MemberExpr* M, ExplodedNode* Pred, 
                       ExplodedNodeSet& Dst, bool asLValue);

  /// Transfer function logic for ObjCAtSynchronizedStmts.
  void VisitObjCAtSynchronizedStmt(const ObjCAtSynchronizedStmt *S,
                                   ExplodedNode *Pred, ExplodedNodeSet &Dst);

  /// VisitObjCIvarRefExpr - Transfer function logic for ObjCIvarRefExprs.
  void VisitObjCIvarRefExpr(const ObjCIvarRefExpr* DR, ExplodedNode* Pred,
                            ExplodedNodeSet& Dst, bool asLValue);

  /// VisitObjCForCollectionStmt - Transfer function logic for
  ///  ObjCForCollectionStmt.
  void VisitObjCForCollectionStmt(const ObjCForCollectionStmt* S, 
                                  ExplodedNode* Pred, ExplodedNodeSet& Dst);

  void VisitObjCForCollectionStmtAux(const ObjCForCollectionStmt* S, 
                                     ExplodedNode* Pred,
                                     ExplodedNodeSet& Dst, SVal ElementV);

  /// VisitObjCMessageExpr - Transfer function for ObjC message expressions.
  void VisitObjCMessageExpr(const ObjCMessageExpr* ME, ExplodedNode* Pred, 
                            ExplodedNodeSet& Dst, bool asLValue);

  /// VisitReturnStmt - Transfer function logic for return statements.
  void VisitReturnStmt(const ReturnStmt* R, ExplodedNode* Pred, 
                       ExplodedNodeSet& Dst);
  
  /// VisitOffsetOfExpr - Transfer function for offsetof.
  void VisitOffsetOfExpr(const OffsetOfExpr* Ex, ExplodedNode* Pred,
                         ExplodedNodeSet& Dst);

  /// VisitSizeOfAlignOfExpr - Transfer function for sizeof.
  void VisitSizeOfAlignOfExpr(const SizeOfAlignOfExpr* Ex, ExplodedNode* Pred,
                              ExplodedNodeSet& Dst);

  /// VisitUnaryOperator - Transfer function logic for unary operators.
  void VisitUnaryOperator(const UnaryOperator* B, ExplodedNode* Pred, 
                          ExplodedNodeSet& Dst, bool asLValue);

  void VisitCXXThisExpr(const CXXThisExpr *TE, ExplodedNode *Pred, 
                        ExplodedNodeSet & Dst);
  
  void VisitCXXConstructExpr(const CXXConstructExpr *E, const MemRegion *Dest,
                             ExplodedNode *Pred, ExplodedNodeSet &Dst);

  void VisitCXXMemberCallExpr(const CXXMemberCallExpr *MCE, ExplodedNode *Pred,
                              ExplodedNodeSet &Dst);

  void VisitCXXNewExpr(const CXXNewExpr *CNE, ExplodedNode *Pred,
                       ExplodedNodeSet &Dst);

  void VisitCXXDeleteExpr(const CXXDeleteExpr *CDE, ExplodedNode *Pred,
                          ExplodedNodeSet &Dst);

  void VisitAggExpr(const Expr *E, const MemRegion *Dest, ExplodedNode *Pred,
                    ExplodedNodeSet &Dst);

  /// Create a C++ temporary object for an rvalue.
  void CreateCXXTemporaryObject(const Expr *Ex, ExplodedNode *Pred, 
                                ExplodedNodeSet &Dst);

  /// Synthesize CXXThisRegion.
  const CXXThisRegion *getCXXThisRegion(const CXXRecordDecl *RD,
                                        const StackFrameContext *SFC);

  /// Evaluate arguments with a work list algorithm.
  void EvalArguments(ConstExprIterator AI, ConstExprIterator AE,
                     const FunctionProtoType *FnType, 
                     ExplodedNode *Pred, ExplodedNodeSet &Dst);

  /// EvalEagerlyAssume - Given the nodes in 'Src', eagerly assume symbolic
  ///  expressions of the form 'x != 0' and generate new nodes (stored in Dst)
  ///  with those assumptions.
  void EvalEagerlyAssume(ExplodedNodeSet& Dst, ExplodedNodeSet& Src, 
                         const Expr *Ex);

  SVal EvalMinus(SVal X) {
    return X.isValid() ? SVator.EvalMinus(cast<NonLoc>(X)) : X;
  }

  SVal EvalComplement(SVal X) {
    return X.isValid() ? SVator.EvalComplement(cast<NonLoc>(X)) : X;
  }

public:

  SVal EvalBinOp(const GRState *state, BinaryOperator::Opcode op,
                 NonLoc L, NonLoc R, QualType T) {
    return SVator.EvalBinOpNN(state, op, L, R, T);
  }

  SVal EvalBinOp(const GRState *state, BinaryOperator::Opcode op,
                 NonLoc L, SVal R, QualType T) {
    return R.isValid() ? SVator.EvalBinOpNN(state,op,L, cast<NonLoc>(R), T) : R;
  }

  SVal EvalBinOp(const GRState *ST, BinaryOperator::Opcode Op,
                 SVal LHS, SVal RHS, QualType T) {
    return SVator.EvalBinOp(ST, Op, LHS, RHS, T);
  }
  
protected:
  void EvalObjCMessageExpr(ExplodedNodeSet& Dst, const ObjCMessageExpr* ME, 
                           ExplodedNode* Pred, const GRState *state) {
    assert (Builder && "GRStmtNodeBuilder must be defined.");
    getTF().EvalObjCMessageExpr(Dst, *this, *Builder, ME, Pred, state);
  }

  const GRState* MarkBranch(const GRState* St, const Stmt* Terminator,
                            bool branchTaken);

  /// EvalBind - Handle the semantics of binding a value to a specific location.
  ///  This method is used by EvalStore, VisitDeclStmt, and others.
  void EvalBind(ExplodedNodeSet& Dst, const Stmt* StoreE, ExplodedNode* Pred,
                const GRState* St, SVal location, SVal Val,
                bool atDeclInit = false);

public:
  // FIXME: 'tag' should be removed, and a LocationContext should be used
  // instead.
  void EvalLoad(ExplodedNodeSet& Dst, const Expr* Ex, ExplodedNode* Pred,
                const GRState* St, SVal location, const void *tag = 0,
                QualType LoadTy = QualType());

  // FIXME: 'tag' should be removed, and a LocationContext should be used
  // instead.
  void EvalStore(ExplodedNodeSet& Dst, const Expr* AssignE, const Expr* StoreE,
                 ExplodedNode* Pred, const GRState* St, SVal TargetLV, SVal Val,
                 const void *tag = 0);
private:  
  void EvalLoadCommon(ExplodedNodeSet& Dst, const Expr* Ex, ExplodedNode* Pred,
                      const GRState* St, SVal location, const void *tag,
                      QualType LoadTy);

  // FIXME: 'tag' should be removed, and a LocationContext should be used
  // instead.
  void EvalLocation(ExplodedNodeSet &Dst, const Stmt *S, ExplodedNode* Pred,
                    const GRState* St, SVal location,
                    const void *tag, bool isLoad);

  bool InlineCall(ExplodedNodeSet &Dst, const CallExpr *CE, ExplodedNode *Pred);
};

} // end clang namespace

#endif
