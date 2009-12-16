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
#include "clang/AST/ExprCXX.h"

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

  typedef llvm::DenseMap<void *, unsigned> CheckerMap;
  CheckerMap CheckerM;
  
  typedef std::vector<std::pair<void *, Checker*> >CheckersOrdered;
  CheckersOrdered Checkers;

  /// BR - The BugReporter associated with this engine.  It is important that
  //   this object be placed at the very end of member variables so that its
  //   destructor is called before the rest of the GRExprEngine is destroyed.
  GRBugReporter BR;

public:
  GRExprEngine(AnalysisManager &mgr);

  ~GRExprEngine();

  void ExecuteWorkList(const LocationContext *L, unsigned Steps = 150000) {
    CoreEngine.ExecuteWorkList(L, Steps);
  }

  /// getContext - Return the ASTContext associated with this analysis.
  ASTContext& getContext() const { return G.getContext(); }

  AnalysisManager &getAnalysisManager() const { return AMgr; }

  SValuator &getSValuator() { return SVator; }

  GRTransferFuncs& getTF() { return *StateMgr.TF; }

  BugReporter& getBugReporter() { return BR; }

  GRStmtNodeBuilder &getBuilder() { assert(Builder); return *Builder; }

  /// setTransferFunctions
  void setTransferFunctions(GRTransferFuncs* tf);

  void setTransferFunctions(GRTransferFuncs& tf) {
    setTransferFunctions(&tf);
  }

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

  void AddCheck(GRSimpleAPICheck* A, Stmt::StmtClass C);
  void AddCheck(GRSimpleAPICheck* A);

  /// ProcessStmt - Called by GRCoreEngine. Used to generate new successor
  ///  nodes by processing the 'effects' of a block-level statement.
  void ProcessStmt(CFGElement E, GRStmtNodeBuilder& builder);

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
  void ProcessEndPath(GREndPathNodeBuilder& builder);

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
  ExplodedNode* MakeNode(ExplodedNodeSet& Dst, Stmt* S, ExplodedNode* Pred, 
                         const GRState* St,
                         ProgramPoint::Kind K = ProgramPoint::PostStmtKind,
                         const void *tag = 0);
protected:
  /// CheckerVisit - Dispatcher for performing checker-specific logic
  ///  at specific statements.
  void CheckerVisit(Stmt *S, ExplodedNodeSet &Dst, ExplodedNodeSet &Src, 
                    bool isPrevisit);

  bool CheckerEvalCall(const CallExpr *CE, 
                       ExplodedNodeSet &Dst, 
                       ExplodedNode *Pred);

  void CheckerEvalNilReceiver(const ObjCMessageExpr *ME, 
                              ExplodedNodeSet &Dst,
                              const GRState *state,
                              ExplodedNode *Pred);
  
  void CheckerVisitBind(const Stmt *AssignE, const Stmt *StoreE,
                        ExplodedNodeSet &Dst, ExplodedNodeSet &Src, 
                        SVal location, SVal val, bool isPrevisit);


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
  
  /// VisitBlockExpr - Transfer function logic for BlockExprs.
  void VisitBlockExpr(BlockExpr *BE, ExplodedNode *Pred, ExplodedNodeSet &Dst);

  /// VisitBinaryOperator - Transfer function logic for binary operators.
  void VisitBinaryOperator(BinaryOperator* B, ExplodedNode* Pred, 
                           ExplodedNodeSet& Dst, bool asLValue);


  /// VisitCall - Transfer function for function calls.
  void VisitCall(CallExpr* CE, ExplodedNode* Pred,
                 CallExpr::arg_iterator AI, CallExpr::arg_iterator AE,
                 ExplodedNodeSet& Dst);
  void VisitCallRec(CallExpr* CE, ExplodedNode* Pred,
                    CallExpr::arg_iterator AI, CallExpr::arg_iterator AE,
                    ExplodedNodeSet& Dst, const FunctionProtoType *,
                    unsigned ParamIdx = 0);

  /// VisitCast - Transfer function logic for all casts (implicit and explicit).
  void VisitCast(Expr* CastE, Expr* Ex, ExplodedNode* Pred,
                 ExplodedNodeSet& Dst);

  /// VisitCompoundLiteralExpr - Transfer function logic for compound literals.
  void VisitCompoundLiteralExpr(CompoundLiteralExpr* CL, ExplodedNode* Pred,
                                ExplodedNodeSet& Dst, bool asLValue);

  /// VisitDeclRefExpr - Transfer function logic for DeclRefExprs.
  void VisitDeclRefExpr(DeclRefExpr* DR, ExplodedNode* Pred,
                        ExplodedNodeSet& Dst, bool asLValue);

  /// VisitBlockDeclRefExpr - Transfer function logic for BlockDeclRefExprs.
  void VisitBlockDeclRefExpr(BlockDeclRefExpr* DR, ExplodedNode* Pred,
                             ExplodedNodeSet& Dst, bool asLValue);
  
  void VisitCommonDeclRefExpr(Expr* DR, const NamedDecl *D,ExplodedNode* Pred,
                             ExplodedNodeSet& Dst, bool asLValue);  
  
  /// VisitDeclStmt - Transfer function logic for DeclStmts.
  void VisitDeclStmt(DeclStmt* DS, ExplodedNode* Pred, ExplodedNodeSet& Dst);

  /// VisitGuardedExpr - Transfer function logic for ?, __builtin_choose
  void VisitGuardedExpr(Expr* Ex, Expr* L, Expr* R, ExplodedNode* Pred,
                        ExplodedNodeSet& Dst);

  void VisitInitListExpr(InitListExpr* E, ExplodedNode* Pred,
                         ExplodedNodeSet& Dst);

  /// VisitLogicalExpr - Transfer function logic for '&&', '||'
  void VisitLogicalExpr(BinaryOperator* B, ExplodedNode* Pred,
                        ExplodedNodeSet& Dst);

  /// VisitMemberExpr - Transfer function for member expressions.
  void VisitMemberExpr(MemberExpr* M, ExplodedNode* Pred, ExplodedNodeSet& Dst,
                       bool asLValue);

  /// VisitObjCIvarRefExpr - Transfer function logic for ObjCIvarRefExprs.
  void VisitObjCIvarRefExpr(ObjCIvarRefExpr* DR, ExplodedNode* Pred,
                            ExplodedNodeSet& Dst, bool asLValue);

  /// VisitObjCForCollectionStmt - Transfer function logic for
  ///  ObjCForCollectionStmt.
  void VisitObjCForCollectionStmt(ObjCForCollectionStmt* S, ExplodedNode* Pred,
                                  ExplodedNodeSet& Dst);

  void VisitObjCForCollectionStmtAux(ObjCForCollectionStmt* S, 
                                     ExplodedNode* Pred,
                                     ExplodedNodeSet& Dst, SVal ElementV);

  /// VisitObjCMessageExpr - Transfer function for ObjC message expressions.
  void VisitObjCMessageExpr(ObjCMessageExpr* ME, ExplodedNode* Pred, 
                            ExplodedNodeSet& Dst);

  void VisitObjCMessageExprArgHelper(ObjCMessageExpr* ME,
                                     ObjCMessageExpr::arg_iterator I,
                                     ObjCMessageExpr::arg_iterator E,
                                     ExplodedNode* Pred, ExplodedNodeSet& Dst);

  void VisitObjCMessageExprDispatchHelper(ObjCMessageExpr* ME, 
                                          ExplodedNode* Pred,
                                          ExplodedNodeSet& Dst);

  /// VisitReturnStmt - Transfer function logic for return statements.
  void VisitReturnStmt(ReturnStmt* R, ExplodedNode* Pred, ExplodedNodeSet& Dst);

  /// VisitSizeOfAlignOfExpr - Transfer function for sizeof.
  void VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr* Ex, ExplodedNode* Pred,
                              ExplodedNodeSet& Dst);

  /// VisitUnaryOperator - Transfer function logic for unary operators.
  void VisitUnaryOperator(UnaryOperator* B, ExplodedNode* Pred, 
                          ExplodedNodeSet& Dst, bool asLValue);

  void VisitCXXThisExpr(CXXThisExpr *TE, ExplodedNode *Pred, 
                        ExplodedNodeSet & Dst);

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
  void EvalObjCMessageExpr(ExplodedNodeSet& Dst, ObjCMessageExpr* ME, 
                           ExplodedNode* Pred, const GRState *state) {
    assert (Builder && "GRStmtNodeBuilder must be defined.");
    getTF().EvalObjCMessageExpr(Dst, *this, *Builder, ME, Pred, state);
  }

  const GRState* MarkBranch(const GRState* St, Stmt* Terminator,
                            bool branchTaken);

  /// EvalBind - Handle the semantics of binding a value to a specific location.
  ///  This method is used by EvalStore, VisitDeclStmt, and others.
  void EvalBind(ExplodedNodeSet& Dst, Stmt *AssignE,
                Stmt* StoreE, ExplodedNode* Pred,
                const GRState* St, SVal location, SVal Val,
                bool atDeclInit = false);

public:
  // FIXME: 'tag' should be removed, and a LocationContext should be used
  // instead.
  void EvalLoad(ExplodedNodeSet& Dst, Expr* Ex, ExplodedNode* Pred,
                const GRState* St, SVal location, const void *tag = 0,
                QualType LoadTy = QualType());

  // FIXME: 'tag' should be removed, and a LocationContext should be used
  // instead.
  void EvalStore(ExplodedNodeSet& Dst, Expr* AssignE, Expr* StoreE,
                 ExplodedNode* Pred, const GRState* St, SVal TargetLV, SVal Val,
                 const void *tag = 0);
private:  
  void EvalLoadCommon(ExplodedNodeSet& Dst, Expr* Ex, ExplodedNode* Pred,
                      const GRState* St, SVal location, const void *tag,
                      QualType LoadTy);

  // FIXME: 'tag' should be removed, and a LocationContext should be used
  // instead.
  void EvalLocation(ExplodedNodeSet &Dst, Stmt *S, ExplodedNode* Pred,
                    const GRState* St, SVal location,
                    const void *tag, bool isLoad);
};

} // end clang namespace

#endif
