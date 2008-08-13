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

#include "clang/Analysis/PathSensitive/BasicStore.h"
#include "clang/Analysis/PathSensitive/GRExprEngine.h"
#include "clang/Analysis/PathSensitive/BugReporter.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/Streams.h"
#include "llvm/ADT/ImmutableList.h"
#include "llvm/Support/Compiler.h"

#ifndef NDEBUG
#include "llvm/Support/GraphWriter.h"
#include <sstream>
#endif

using namespace clang;
using llvm::dyn_cast;
using llvm::cast;
using llvm::APSInt;

//===----------------------------------------------------------------------===//
// Engine construction and deletion.
//===----------------------------------------------------------------------===//

namespace {

class VISIBILITY_HIDDEN MappedBatchAuditor : public GRSimpleAPICheck {
  typedef llvm::ImmutableList<GRSimpleAPICheck*> Checks;
  typedef llvm::DenseMap<void*,Checks> MapTy;
  
  MapTy M;
  Checks::Factory F;

public:
  MappedBatchAuditor(llvm::BumpPtrAllocator& Alloc) : F(Alloc) {}
  
  virtual ~MappedBatchAuditor() {
    llvm::DenseSet<GRSimpleAPICheck*> AlreadyVisited;
    
    for (MapTy::iterator MI = M.begin(), ME = M.end(); MI != ME; ++MI)
      for (Checks::iterator I=MI->second.begin(), E=MI->second.end(); I!=E;++I){

        GRSimpleAPICheck* check = *I;
        
        if (AlreadyVisited.count(check))
          continue;
        
        AlreadyVisited.insert(check);
        delete check;
      }
  }

  void AddCheck(GRSimpleAPICheck* A, Stmt::StmtClass C) {
    assert (A && "Check cannot be null.");
    void* key = reinterpret_cast<void*>((uintptr_t) C);
    MapTy::iterator I = M.find(key);
    M[key] = F.Concat(A, I == M.end() ? F.GetEmptyList() : I->second);
  }
  
  virtual void EmitWarnings(BugReporter& BR) {
    llvm::DenseSet<GRSimpleAPICheck*> AlreadyVisited;
    
    for (MapTy::iterator MI = M.begin(), ME = M.end(); MI != ME; ++MI)
      for (Checks::iterator I=MI->second.begin(), E=MI->second.end(); I!=E;++I){
        
        GRSimpleAPICheck* check = *I;
        
        if (AlreadyVisited.count(check))
          continue;
        
        check->EmitWarnings(BR);
      }
  }
  
  virtual bool Audit(NodeTy* N, ValueStateManager& VMgr) {
    Stmt* S = cast<PostStmt>(N->getLocation()).getStmt();
    void* key = reinterpret_cast<void*>((uintptr_t) S->getStmtClass());
    MapTy::iterator MI = M.find(key);

    if (MI == M.end())
      return false;
    
    bool isSink = false;
    
    for (Checks::iterator I=MI->second.begin(), E=MI->second.end(); I!=E; ++I)
      isSink |= (*I)->Audit(N, VMgr);

    return isSink;    
  }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Engine construction and deletion.
//===----------------------------------------------------------------------===//

static inline Selector GetNullarySelector(const char* name, ASTContext& Ctx) {
  IdentifierInfo* II = &Ctx.Idents.get(name);
  return Ctx.Selectors.getSelector(0, &II);
}


GRExprEngine::GRExprEngine(CFG& cfg, Decl& CD, ASTContext& Ctx,
                           LiveVariables& L)
  : CoreEngine(cfg, CD, Ctx, *this), 
    G(CoreEngine.getGraph()),
    Liveness(L),
    Builder(NULL),
    StateMgr(G.getContext(), CreateBasicStoreManager(G.getAllocator()),
             G.getAllocator(), G.getCFG()),
    SymMgr(StateMgr.getSymbolManager()),
    CurrentStmt(NULL),
  NSExceptionII(NULL), NSExceptionInstanceRaiseSelectors(NULL),
  RaiseSel(GetNullarySelector("raise", G.getContext())) {}

GRExprEngine::~GRExprEngine() {    
  for (BugTypeSet::iterator I = BugTypes.begin(), E = BugTypes.end(); I!=E; ++I)
    delete *I;

  
  delete [] NSExceptionInstanceRaiseSelectors;
}

//===----------------------------------------------------------------------===//
// Utility methods.
//===----------------------------------------------------------------------===//

// SaveAndRestore - A utility class that uses RIIA to save and restore
//  the value of a variable.
template<typename T>
struct VISIBILITY_HIDDEN SaveAndRestore {
  SaveAndRestore(T& x) : X(x), old_value(x) {}
  ~SaveAndRestore() { X = old_value; }
  T get() { return old_value; }
  
  T& X;
  T old_value;
};

// SaveOr - Similar to SaveAndRestore.  Operates only on bools; the old
//  value of a variable is saved, and during the dstor the old value is
//  or'ed with the new value.
struct VISIBILITY_HIDDEN SaveOr {
  SaveOr(bool& x) : X(x), old_value(x) { x = false; }
  ~SaveOr() { X |= old_value; }
  
  bool& X;
  bool old_value;
};


void GRExprEngine::EmitWarnings(BugReporterData& BRData) {
  for (bug_type_iterator I = bug_types_begin(), E = bug_types_end(); I!=E; ++I){
    GRBugReporter BR(BRData, *this);
    (*I)->EmitWarnings(BR);
  }
  
  if (BatchAuditor) {
    GRBugReporter BR(BRData, *this);
    BatchAuditor->EmitWarnings(BR);
  }
}

void GRExprEngine::setTransferFunctions(GRTransferFuncs* tf) {
  StateMgr.TF = tf;
  getTF().RegisterChecks(*this);
}

void GRExprEngine::AddCheck(GRSimpleAPICheck* A, Stmt::StmtClass C) {
  if (!BatchAuditor)
    BatchAuditor.reset(new MappedBatchAuditor(getGraph().getAllocator()));
  
  ((MappedBatchAuditor*) BatchAuditor.get())->AddCheck(A, C);
}

const ValueState* GRExprEngine::getInitialState() {

  // The LiveVariables information already has a compilation of all VarDecls
  // used in the function.  Iterate through this set, and "symbolicate"
  // any VarDecl whose value originally comes from outside the function.
  
  typedef LiveVariables::AnalysisDataTy LVDataTy;
  LVDataTy& D = Liveness.getAnalysisData();
  
  ValueState StateImpl = *StateMgr.getInitialState();
  
  for (LVDataTy::decl_iterator I=D.begin_decl(), E=D.end_decl(); I != E; ++I) {
    
    ScopedDecl *SD = const_cast<ScopedDecl*>(I->first);
    if (VarDecl* VD = dyn_cast<VarDecl>(SD)) {
      // Initialize globals and parameters to symbolic values.
      // Initialize local variables to undefined.
      RVal X = (VD->hasGlobalStorage() || isa<ParmVarDecl>(VD) ||
                isa<ImplicitParamDecl>(VD))
             ? RVal::GetSymbolValue(SymMgr, VD)
             : UndefinedVal();
      
      StateMgr.SetRVal(StateImpl, lval::DeclVal(VD), X);
      
    } else if (ImplicitParamDecl *IPD = dyn_cast<ImplicitParamDecl>(SD)) {
        RVal X = RVal::GetSymbolValue(SymMgr, IPD);
      StateMgr.SetRVal(StateImpl, lval::DeclVal(IPD), X);
    }
      

  }
  
  return StateMgr.getPersistentState(StateImpl);
}

//===----------------------------------------------------------------------===//
// Top-level transfer function logic (Dispatcher).
//===----------------------------------------------------------------------===//

void GRExprEngine::ProcessStmt(Stmt* S, StmtNodeBuilder& builder) {
  
  Builder = &builder;
  EntryNode = builder.getLastNode();
  
  // FIXME: Consolidate.
  CurrentStmt = S;
  StateMgr.CurrentStmt = S;
  
  // Set up our simple checks.
  if (BatchAuditor)
    Builder->setAuditor(BatchAuditor.get());
  
  // Create the cleaned state.  
  CleanedState = StateMgr.RemoveDeadBindings(EntryNode->getState(), CurrentStmt,
                                             Liveness, DeadSymbols);
  
  // Process any special transfer function for dead symbols.
  NodeSet Tmp;
  
  if (DeadSymbols.empty())
    Tmp.Add(EntryNode);
  else {
    SaveAndRestore<bool> OldSink(Builder->BuildSinks);
    SaveOr OldHasGen(Builder->HasGeneratedNode);

    SaveAndRestore<bool> OldPurgeDeadSymbols(Builder->PurgingDeadSymbols);
    Builder->PurgingDeadSymbols = true;
    
    getTF().EvalDeadSymbols(Tmp, *this, *Builder, EntryNode, S, 
                        CleanedState, DeadSymbols);

    if (!Builder->BuildSinks && !Builder->HasGeneratedNode)
      Tmp.Add(EntryNode);
  }
  
  bool HasAutoGenerated = false;

  for (NodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {

    NodeSet Dst;
    
    // Set the cleaned state.  
    Builder->SetCleanedState(*I == EntryNode ? CleanedState : GetState(*I));
    
    // Visit the statement.  
    Visit(S, *I, Dst);

    // Do we need to auto-generate a node?  We only need to do this to generate
    // a node with a "cleaned" state; GRCoreEngine will actually handle
    // auto-transitions for other cases.    
    if (Dst.size() == 1 && *Dst.begin() == EntryNode
        && !Builder->HasGeneratedNode && !HasAutoGenerated) {
      HasAutoGenerated = true;
      builder.generateNode(S, GetState(EntryNode), *I);
    }
  }
  
  // NULL out these variables to cleanup.
  CleanedState = NULL;
  EntryNode = NULL;

  // FIXME: Consolidate.
  StateMgr.CurrentStmt = 0;
  CurrentStmt = 0;
  
  Builder = NULL;
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
      //   AddrLabelExpr, IntegerLiteral, CharacterLiteral
      
      Dst.Add(Pred); // No-op. Simply propagate the current state unchanged.
      break;
    
    case Stmt::ArraySubscriptExprClass:
      VisitArraySubscriptExpr(cast<ArraySubscriptExpr>(S), Pred, Dst, false);
      break;
      
    case Stmt::AsmStmtClass:
      VisitAsmStmt(cast<AsmStmt>(S), Pred, Dst);
      break;
      
    case Stmt::BinaryOperatorClass: {
      BinaryOperator* B = cast<BinaryOperator>(S);
      
      if (B->isLogicalOp()) {
        VisitLogicalExpr(B, Pred, Dst);
        break;
      }
      else if (B->getOpcode() == BinaryOperator::Comma) {
        const ValueState* St = GetState(Pred);
        MakeNode(Dst, B, Pred, SetRVal(St, B, GetRVal(St, B->getRHS())));
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
      VisitDeclRefExpr(cast<DeclRefExpr>(S), Pred, Dst, false);
      break;
      
    case Stmt::DeclStmtClass:
      VisitDeclStmt(cast<DeclStmt>(S), Pred, Dst);
      break;
      
    case Stmt::ImplicitCastExprClass: {
      ImplicitCastExpr* C = cast<ImplicitCastExpr>(S);
      VisitCast(C, C->getSubExpr(), Pred, Dst);
      break;
    }
      
    case Stmt::MemberExprClass: {
      VisitMemberExpr(cast<MemberExpr>(S), Pred, Dst, false);
      break;
    }
      
    case Stmt::ObjCMessageExprClass: {
      VisitObjCMessageExpr(cast<ObjCMessageExpr>(S), Pred, Dst);
      break;
    }
      
    case Stmt::ParenExprClass:
      Visit(cast<ParenExpr>(S)->getSubExpr()->IgnoreParens(), Pred, Dst);
      break;
      
    case Stmt::ReturnStmtClass:
      VisitReturnStmt(cast<ReturnStmt>(S), Pred, Dst);
      break;
      
    case Stmt::SizeOfAlignOfTypeExprClass:
      VisitSizeOfAlignOfTypeExpr(cast<SizeOfAlignOfTypeExpr>(S), Pred, Dst);
      break;
      
    case Stmt::StmtExprClass: {
      StmtExpr* SE = cast<StmtExpr>(S);
      
      const ValueState* St = GetState(Pred);
      
      // FIXME: Not certain if we can have empty StmtExprs.  If so, we should
      // probably just remove these from the CFG.
      assert (!SE->getSubStmt()->body_empty());
      
      if (Expr* LastExpr = dyn_cast<Expr>(*SE->getSubStmt()->body_rbegin()))
        MakeNode(Dst, SE, Pred, SetRVal(St, SE, GetRVal(St, LastExpr)));
      else
        Dst.Add(Pred);
      
      break;
    }
      
    case Stmt::UnaryOperatorClass:
      VisitUnaryOperator(cast<UnaryOperator>(S), Pred, Dst, false);
      break;
  }
}

void GRExprEngine::VisitLVal(Expr* Ex, NodeTy* Pred, NodeSet& Dst) {
  
  Ex = Ex->IgnoreParens();
  
  if (Ex != CurrentStmt && getCFG().isBlkExpr(Ex)) {
    Dst.Add(Pred);
    return;
  }
  
  switch (Ex->getStmtClass()) {
    default:
      Visit(Ex, Pred, Dst);
      return;
      
    case Stmt::ArraySubscriptExprClass:
      VisitArraySubscriptExpr(cast<ArraySubscriptExpr>(Ex), Pred, Dst, true);
      return;
      
    case Stmt::DeclRefExprClass:
      VisitDeclRefExpr(cast<DeclRefExpr>(Ex), Pred, Dst, true);
      return;
      
    case Stmt::UnaryOperatorClass:
      VisitUnaryOperator(cast<UnaryOperator>(Ex), Pred, Dst, true);
      return;
      
    case Stmt::MemberExprClass:
      VisitMemberExpr(cast<MemberExpr>(Ex), Pred, Dst, true);
      return;
  }
}

//===----------------------------------------------------------------------===//
// Block entrance.  (Update counters).
//===----------------------------------------------------------------------===//

bool GRExprEngine::ProcessBlockEntrance(CFGBlock* B, const ValueState*,
                                        GRBlockCounter BC) {
  
  return BC.getNumVisited(B->getBlockID()) < 3;
}

//===----------------------------------------------------------------------===//
// Branch processing.
//===----------------------------------------------------------------------===//

const ValueState* GRExprEngine::MarkBranch(const ValueState* St,
                                           Stmt* Terminator,
                                           bool branchTaken) {
  
  switch (Terminator->getStmtClass()) {
    default:
      return St;
      
    case Stmt::BinaryOperatorClass: { // '&&' and '||'
      
      BinaryOperator* B = cast<BinaryOperator>(Terminator);
      BinaryOperator::Opcode Op = B->getOpcode();
      
      assert (Op == BinaryOperator::LAnd || Op == BinaryOperator::LOr);
      
      // For &&, if we take the true branch, then the value of the whole
      // expression is that of the RHS expression.
      //
      // For ||, if we take the false branch, then the value of the whole
      // expression is that of the RHS expression.
      
      Expr* Ex = (Op == BinaryOperator::LAnd && branchTaken) ||
                 (Op == BinaryOperator::LOr && !branchTaken)  
               ? B->getRHS() : B->getLHS();
        
      return SetBlkExprRVal(St, B, UndefinedVal(Ex));
    }
      
    case Stmt::ConditionalOperatorClass: { // ?:
      
      ConditionalOperator* C = cast<ConditionalOperator>(Terminator);
      
      // For ?, if branchTaken == true then the value is either the LHS or
      // the condition itself. (GNU extension).
      
      Expr* Ex;      
      
      if (branchTaken)
        Ex = C->getLHS() ? C->getLHS() : C->getCond();        
      else
        Ex = C->getRHS();
      
      return SetBlkExprRVal(St, C, UndefinedVal(Ex));
    }
      
    case Stmt::ChooseExprClass: { // ?:
      
      ChooseExpr* C = cast<ChooseExpr>(Terminator);
      
      Expr* Ex = branchTaken ? C->getLHS() : C->getRHS();      
      return SetBlkExprRVal(St, C, UndefinedVal(Ex));
    }
  }
}

void GRExprEngine::ProcessBranch(Expr* Condition, Stmt* Term,
                                 BranchNodeBuilder& builder) {

  // Remove old bindings for subexpressions.
  const ValueState* PrevState =
    StateMgr.RemoveSubExprBindings(builder.getState());
  
  // Check for NULL conditions; e.g. "for(;;)"
  if (!Condition) { 
    builder.markInfeasible(false);
    return;
  }
  
  RVal V = GetRVal(PrevState, Condition);
  
  switch (V.getBaseKind()) {
    default:
      break;

    case RVal::UnknownKind:
      builder.generateNode(MarkBranch(PrevState, Term, true), true);
      builder.generateNode(MarkBranch(PrevState, Term, false), false);
      return;
      
    case RVal::UndefinedKind: {      
      NodeTy* N = builder.generateNode(PrevState, true);

      if (N) {
        N->markAsSink();
        UndefBranches.insert(N);
      }
      
      builder.markInfeasible(false);
      return;
    }      
  }
    
  // Process the true branch.

  bool isFeasible = false;  
  const ValueState* St = Assume(PrevState, V, true, isFeasible);

  if (isFeasible)
    builder.generateNode(MarkBranch(St, Term, true), true);
  else
    builder.markInfeasible(true);
      
  // Process the false branch.  
  
  isFeasible = false;
  St = Assume(PrevState, V, false, isFeasible);
  
  if (isFeasible)
    builder.generateNode(MarkBranch(St, Term, false), false);
  else
    builder.markInfeasible(false);
}

/// ProcessIndirectGoto - Called by GRCoreEngine.  Used to generate successor
///  nodes by processing the 'effects' of a computed goto jump.
void GRExprEngine::ProcessIndirectGoto(IndirectGotoNodeBuilder& builder) {

  const ValueState* St = builder.getState();  
  RVal V = GetRVal(St, builder.getTarget());
  
  // Three possibilities:
  //
  //   (1) We know the computed label.
  //   (2) The label is NULL (or some other constant), or Undefined.
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

  if (isa<lval::ConcreteInt>(V) || isa<UndefinedVal>(V)) {
    // Dispatch to the first target and mark it as a sink.
    NodeTy* N = builder.generateNode(builder.begin(), St, true);
    UndefBranches.insert(N);
    return;
  }
  
  // This is really a catch-all.  We don't support symbolics yet.
  
  assert (V.isUnknown());
  
  for (iterator I=builder.begin(), E=builder.end(); I != E; ++I)
    builder.generateNode(I, St);
}


void GRExprEngine::VisitGuardedExpr(Expr* Ex, Expr* L, Expr* R,
                                    NodeTy* Pred, NodeSet& Dst) {
  
  assert (Ex == CurrentStmt && getCFG().isBlkExpr(Ex));
  
  const ValueState* St = GetState(Pred);
  RVal X = GetBlkExprRVal(St, Ex);
  
  assert (X.isUndef());
  
  Expr* SE = (Expr*) cast<UndefinedVal>(X).getData();
  
  assert (SE);
  
  X = GetBlkExprRVal(St, SE);
  
  // Make sure that we invalidate the previous binding.
  MakeNode(Dst, Ex, Pred, StateMgr.SetRVal(St, Ex, X, true, true));
}

/// ProcessSwitch - Called by GRCoreEngine.  Used to generate successor
///  nodes by processing the 'effects' of a switch statement.
void GRExprEngine::ProcessSwitch(SwitchNodeBuilder& builder) {
  
  typedef SwitchNodeBuilder::iterator iterator;
  
  const ValueState* St = builder.getState();  
  Expr* CondE = builder.getCondition();
  RVal  CondV = GetRVal(St, CondE);

  if (CondV.isUndef()) {
    NodeTy* N = builder.generateDefaultCaseNode(St, true);
    UndefBranches.insert(N);
    return;
  }
  
  const ValueState*  DefaultSt = St;
  
  // While most of this can be assumed (such as the signedness), having it
  // just computed makes sure everything makes the same assumptions end-to-end.
  
  unsigned bits = getContext().getTypeSize(CondE->getType());

  APSInt V1(bits, false);
  APSInt V2 = V1;
  bool DefaultFeasible = false;
  
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
    else
      V2 = V1;
    
    // FIXME: Eventually we should replace the logic below with a range
    //  comparison, rather than concretize the values within the range.
    //  This should be easy once we have "ranges" for NonLVals.
        
    do {
      nonlval::ConcreteInt CaseVal(getBasicVals().getValue(V1));
      
      RVal Res = EvalBinOp(BinaryOperator::EQ, CondV, CaseVal);
      
      // Now "assume" that the case matches.
      
      bool isFeasible = false;      
      const ValueState* StNew = Assume(St, Res, true, isFeasible);
      
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
      
      isFeasible = false;
      StNew = Assume(DefaultSt, Res, false, isFeasible);
      
      if (isFeasible) {
        DefaultFeasible = true;
        DefaultSt = StNew;
      }

      // Concretize the next value in the range.
      if (V1 == V2)
        break;
      
      ++V1;
      assert (V1 <= V2);
      
    } while (true);
  }
  
  // If we reach here, than we know that the default branch is
  // possible.  
  if (DefaultFeasible) builder.generateDefaultCaseNode(DefaultSt);
}

//===----------------------------------------------------------------------===//
// Transfer functions: logical operations ('&&', '||').
//===----------------------------------------------------------------------===//

void GRExprEngine::VisitLogicalExpr(BinaryOperator* B, NodeTy* Pred,
                                    NodeSet& Dst) {
  
  assert (B->getOpcode() == BinaryOperator::LAnd ||
          B->getOpcode() == BinaryOperator::LOr);
  
  assert (B == CurrentStmt && getCFG().isBlkExpr(B));
  
  const ValueState* St = GetState(Pred);
  RVal X = GetBlkExprRVal(St, B);
  
  assert (X.isUndef());
  
  Expr* Ex = (Expr*) cast<UndefinedVal>(X).getData();
  
  assert (Ex);
  
  if (Ex == B->getRHS()) {
    
    X = GetBlkExprRVal(St, Ex);
    
    // Handle undefined values.
    
    if (X.isUndef()) {
      MakeNode(Dst, B, Pred, SetBlkExprRVal(St, B, X));
      return;
    }
    
    // We took the RHS.  Because the value of the '&&' or '||' expression must
    // evaluate to 0 or 1, we must assume the value of the RHS evaluates to 0
    // or 1.  Alternatively, we could take a lazy approach, and calculate this
    // value later when necessary.  We don't have the machinery in place for
    // this right now, and since most logical expressions are used for branches,
    // the payoff is not likely to be large.  Instead, we do eager evaluation.
        
    bool isFeasible = false;
    const ValueState* NewState = Assume(St, X, true, isFeasible);
    
    if (isFeasible)
      MakeNode(Dst, B, Pred,
               SetBlkExprRVal(NewState, B, MakeConstantVal(1U, B)));
      
    isFeasible = false;
    NewState = Assume(St, X, false, isFeasible);
    
    if (isFeasible)
      MakeNode(Dst, B, Pred,
               SetBlkExprRVal(NewState, B, MakeConstantVal(0U, B)));
  }
  else {
    // We took the LHS expression.  Depending on whether we are '&&' or
    // '||' we know what the value of the expression is via properties of
    // the short-circuiting.
    
    X = MakeConstantVal( B->getOpcode() == BinaryOperator::LAnd ? 0U : 1U, B);
    MakeNode(Dst, B, Pred, SetBlkExprRVal(St, B, X));
  }
}
 
//===----------------------------------------------------------------------===//
// Transfer functions: Loads and stores.
//===----------------------------------------------------------------------===//

void GRExprEngine::VisitDeclRefExpr(DeclRefExpr* D, NodeTy* Pred, NodeSet& Dst,
                                    bool asLVal) {
  
  const ValueState* St = GetState(Pred);
  RVal X = RVal::MakeVal(getBasicVals(), D);
  
  if (asLVal)
    MakeNode(Dst, D, Pred, SetRVal(St, D, cast<LVal>(X)));
  else {
    RVal V = isa<lval::DeclVal>(X) ? GetRVal(St, cast<LVal>(X)) : X;
    MakeNode(Dst, D, Pred, SetRVal(St, D, V));
  }
}

/// VisitArraySubscriptExpr - Transfer function for array accesses
void GRExprEngine::VisitArraySubscriptExpr(ArraySubscriptExpr* A, NodeTy* Pred,
                                           NodeSet& Dst, bool asLVal) {
  
  Expr* Base = A->getBase()->IgnoreParens();
  Expr* Idx  = A->getIdx()->IgnoreParens();
  
  // Always visit the base as an LVal expression.  This computes the
  // abstract address of the base object.
  NodeSet Tmp;
  
  if (LVal::IsLValType(Base->getType())) // Base always is an LVal.
    Visit(Base, Pred, Tmp);
  else  
    VisitLVal(Base, Pred, Tmp);
  
  for (NodeSet::iterator I1=Tmp.begin(), E1=Tmp.end(); I1!=E1; ++I1) {
    
    // Evaluate the index.

    NodeSet Tmp2;
    Visit(Idx, *I1, Tmp2);
      
    for (NodeSet::iterator I2=Tmp2.begin(), E2=Tmp2.end(); I2!=E2; ++I2) {

      const ValueState* St = GetState(*I2);
      RVal BaseV = GetRVal(St, Base);
      RVal IdxV  = GetRVal(St, Idx);      
      
      // If IdxV is 0, return just BaseV.
      
      bool useBase = false;
      
      if (nonlval::ConcreteInt* IdxInt = dyn_cast<nonlval::ConcreteInt>(&IdxV))        
        useBase = IdxInt->getValue() == 0;
      
      RVal V = useBase ? BaseV : lval::ArrayOffset::Make(getBasicVals(), BaseV,IdxV);

      if (asLVal)
        MakeNode(Dst, A, *I2, SetRVal(St, A, V));
      else
        EvalLoad(Dst, A, *I2, St, V);
    }
  }
}

/// VisitMemberExpr - Transfer function for member expressions.
void GRExprEngine::VisitMemberExpr(MemberExpr* M, NodeTy* Pred,
                                   NodeSet& Dst, bool asLVal) {
  
  Expr* Base = M->getBase()->IgnoreParens();

  // Always visit the base as an LVal expression.  This computes the
  // abstract address of the base object.
  NodeSet Tmp;
  
  if (asLVal) {
      
    if (LVal::IsLValType(Base->getType())) // Base always is an LVal.
      Visit(Base, Pred, Tmp);
    else  
      VisitLVal(Base, Pred, Tmp);
  
    for (NodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {
      const ValueState* St = GetState(*I);
      RVal BaseV = GetRVal(St, Base);      
      
      RVal V = lval::FieldOffset::Make(getBasicVals(), GetRVal(St, Base),
                                       M->getMemberDecl());
      
      MakeNode(Dst, M, *I, SetRVal(St, M, V));
    }
    
    return;
  }

  // Evaluate the base.  Can be an LVal or NonLVal (depends on whether
  //  or not isArrow() is true).
  Visit(Base, Pred, Tmp);
  
  for (NodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {

    const ValueState* St = GetState(*I);
    RVal BaseV = GetRVal(St, Base);
    
    if (LVal::IsLValType(Base->getType())) {
    
      assert (M->isArrow());
      
      RVal V = lval::FieldOffset::Make(getBasicVals(), GetRVal(St, Base),
                                       M->getMemberDecl());
    
      EvalLoad(Dst, M, *I, St, V);
    }
    else {
      
      assert (!M->isArrow());
      
      if (BaseV.isUnknownOrUndef()) {
        MakeNode(Dst, M, *I, SetRVal(St, M, BaseV));
        continue;
      }

      // FIXME: Implement nonlval objects representing struct temporaries.
      assert (isa<NonLVal>(BaseV));
      MakeNode(Dst, M, *I, SetRVal(St, M, UnknownVal()));
    }
  }
}

void GRExprEngine::EvalStore(NodeSet& Dst, Expr* Ex, NodeTy* Pred,
                             const ValueState* St, RVal location, RVal Val) {
  
  assert (Builder && "GRStmtNodeBuilder must be defined.");
  
  // Evaluate the location (checks for bad dereferences).
  St = EvalLocation(Ex, Pred, St, location);
  
  if (!St)
    return;
  
  // Proceed with the store.
  
  unsigned size = Dst.size();  

  SaveAndRestore<bool> OldSink(Builder->BuildSinks);
  SaveOr OldHasGen(Builder->HasGeneratedNode);

  assert (!location.isUndef());
  
  getTF().EvalStore(Dst, *this, *Builder, Ex, Pred, St, location, Val);
  
  // Handle the case where no nodes where generated.  Auto-generate that
  // contains the updated state if we aren't generating sinks.
  
  if (!Builder->BuildSinks && Dst.size() == size && !Builder->HasGeneratedNode)
    getTF().GRTransferFuncs::EvalStore(Dst, *this, *Builder, Ex, Pred, St,
                                   location, Val);
}

void GRExprEngine::EvalLoad(NodeSet& Dst, Expr* Ex, NodeTy* Pred,
                            const ValueState* St, RVal location,
                            bool CheckOnly) {

  // Evaluate the location (checks for bad dereferences).
  
  St = EvalLocation(Ex, Pred, St, location, true);
  
  if (!St)
    return;
  
  // Proceed with the load.

  // FIXME: Currently symbolic analysis "generates" new symbols
  //  for the contents of values.  We need a better approach.

  // FIXME: The "CheckOnly" option exists only because Array and Field
  //  loads aren't fully implemented.  Eventually this option will go away.
  
  if (CheckOnly)
    MakeNode(Dst, Ex, Pred, St);
  else if (location.isUnknown()) {
    // This is important.  We must nuke the old binding.
    MakeNode(Dst, Ex, Pred, SetRVal(St, Ex, UnknownVal()));
  }
  else    
    MakeNode(Dst, Ex, Pred, SetRVal(St, Ex, GetRVal(St, cast<LVal>(location),
                                                    Ex->getType())));  
}

const ValueState* GRExprEngine::EvalLocation(Expr* Ex, NodeTy* Pred,
                                             const ValueState* St,
                                             RVal location, bool isLoad) {
  
  // Check for loads/stores from/to undefined values.  
  if (location.isUndef()) {
    ProgramPoint::Kind K =
      isLoad ? ProgramPoint::PostLoadKind : ProgramPoint::PostStmtKind;
    
    if (NodeTy* Succ = Builder->generateNode(Ex, St, Pred, K)) {
      Succ->markAsSink();
      UndefDeref.insert(Succ);
    }
    
    return NULL;
  }
  
  // Check for loads/stores from/to unknown locations.  Treat as No-Ops.
  if (location.isUnknown())
    return St;
  
  // During a load, one of two possible situations arise:
  //  (1) A crash, because the location (pointer) was NULL.
  //  (2) The location (pointer) is not NULL, and the dereference works.
  // 
  // We add these assumptions.
  
  LVal LV = cast<LVal>(location);    
  
  // "Assume" that the pointer is not NULL.
  
  bool isFeasibleNotNull = false;
  const ValueState* StNotNull = Assume(St, LV, true, isFeasibleNotNull);
  
  // "Assume" that the pointer is NULL.
  
  bool isFeasibleNull = false;
  const ValueState* StNull = Assume(St, LV, false, isFeasibleNull);
  
  if (isFeasibleNull) {
    
    // We don't use "MakeNode" here because the node will be a sink
    // and we have no intention of processing it later.
    
    ProgramPoint::Kind K =
      isLoad ? ProgramPoint::PostLoadKind : ProgramPoint::PostStmtKind;

    NodeTy* NullNode = Builder->generateNode(Ex, StNull, Pred, K);
    
    if (NullNode) {
      
      NullNode->markAsSink();
      
      if (isFeasibleNotNull) ImplicitNullDeref.insert(NullNode);
      else ExplicitNullDeref.insert(NullNode);
    }
  }
  
  return isFeasibleNotNull ? StNotNull : NULL;
}

//===----------------------------------------------------------------------===//
// Transfer function: Function calls.
//===----------------------------------------------------------------------===//

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
  Expr* Callee = CE->getCallee()->IgnoreParens();

  VisitLVal(Callee, Pred, DstTmp);
  
  // Finally, evaluate the function call.
  for (NodeSet::iterator DI = DstTmp.begin(), DE = DstTmp.end(); DI!=DE; ++DI) {

    const ValueState* St = GetState(*DI);
    RVal L = GetRVal(St, Callee);

    // FIXME: Add support for symbolic function calls (calls involving
    //  function pointer values that are symbolic).
    
    // Check for undefined control-flow or calls to NULL.
    
    if (L.isUndef() || isa<lval::ConcreteInt>(L)) {      
      NodeTy* N = Builder->generateNode(CE, St, *DI);
      
      if (N) {
        N->markAsSink();
        BadCalls.insert(N);
      }
      
      continue;
    }
    
    // Check for the "noreturn" attribute.
    
    SaveAndRestore<bool> OldSink(Builder->BuildSinks);
    
    if (isa<lval::FuncVal>(L)) {      
      
      FunctionDecl* FD = cast<lval::FuncVal>(L).getDecl();
      
      if (FD->getAttr<NoReturnAttr>())
        Builder->BuildSinks = true;
      else {
        // HACK: Some functions are not marked noreturn, and don't return.
        //  Here are a few hardwired ones.  If this takes too long, we can
        //  potentially cache these results.
        const char* s = FD->getIdentifier()->getName();
        unsigned n = strlen(s);
        
        switch (n) {
          default:
            break;
            
          case 4:
            if (!memcmp(s, "exit", 4)) Builder->BuildSinks = true;
            break;

          case 5:
            if (!memcmp(s, "panic", 5)) Builder->BuildSinks = true;
            break;
          
          case 6:
            if (!memcmp(s, "Assert", 6)) {
              Builder->BuildSinks = true;
              break;
            }
            
            // FIXME: This is just a wrapper around throwing an exception.
            //  Eventually inter-procedural analysis should handle this easily.
            if (!memcmp(s, "ziperr", 6)) Builder->BuildSinks = true;

            break;
          
          case 7:
            if (!memcmp(s, "assfail", 7)) Builder->BuildSinks = true;
            break;
            
          case 8:
            if (!memcmp(s ,"db_error", 8)) Builder->BuildSinks = true;
            break;
          
          case 12:
            if (!memcmp(s, "__assert_rtn", 12)) Builder->BuildSinks = true;
            break;
            
          case 14:
            if (!memcmp(s, "dtrace_assfail", 14)) Builder->BuildSinks = true;
            break;
            
          case 26:
            if (!memcmp(s, "_XCAssertionFailureHandler", 26) ||
                !memcmp(s, "_DTAssertionFailureHandler", 26))
              Builder->BuildSinks = true;

            break;
        }
        
      }
    }
    
    // Evaluate the call.

    if (isa<lval::FuncVal>(L)) {
      
      IdentifierInfo* Info = cast<lval::FuncVal>(L).getDecl()->getIdentifier();
      
      if (unsigned id = Info->getBuiltinID())
        switch (id) {
          case Builtin::BI__builtin_expect: {
            // For __builtin_expect, just return the value of the subexpression.
            assert (CE->arg_begin() != CE->arg_end());            
            RVal X = GetRVal(St, *(CE->arg_begin()));
            MakeNode(Dst, CE, *DI, SetRVal(St, CE, X));
            continue;            
          }
            
          default:
            break;
        }
    }

    // Check any arguments passed-by-value against being undefined.

    bool badArg = false;
    
    for (CallExpr::arg_iterator I = CE->arg_begin(), E = CE->arg_end();
         I != E; ++I) {

      if (GetRVal(GetState(*DI), *I).isUndef()) {        
        NodeTy* N = Builder->generateNode(CE, GetState(*DI), *DI);
      
        if (N) {
          N->markAsSink();
          UndefArgs[N] = *I;
        }
        
        badArg = true;
        break;
      }
    }
    
    if (badArg)
      continue;        

    // Dispatch to the plug-in transfer function.      
    
    unsigned size = Dst.size();
    SaveOr OldHasGen(Builder->HasGeneratedNode);
    EvalCall(Dst, CE, L, *DI);
    
    // Handle the case where no nodes where generated.  Auto-generate that
    // contains the updated state if we aren't generating sinks.
    
    if (!Builder->BuildSinks && Dst.size() == size &&
        !Builder->HasGeneratedNode)
      MakeNode(Dst, CE, *DI, St);
  }
}

//===----------------------------------------------------------------------===//
// Transfer function: Objective-C message expressions.
//===----------------------------------------------------------------------===//

void GRExprEngine::VisitObjCMessageExpr(ObjCMessageExpr* ME, NodeTy* Pred,
                                        NodeSet& Dst){
  
  VisitObjCMessageExprArgHelper(ME, ME->arg_begin(), ME->arg_end(),
                                Pred, Dst);
}  

void GRExprEngine::VisitObjCMessageExprArgHelper(ObjCMessageExpr* ME,
                                                 ObjCMessageExpr::arg_iterator AI,
                                                 ObjCMessageExpr::arg_iterator AE,
                                                 NodeTy* Pred, NodeSet& Dst) {
  if (AI == AE) {
    
    // Process the receiver.
    
    if (Expr* Receiver = ME->getReceiver()) {
      NodeSet Tmp;
      Visit(Receiver, Pred, Tmp);
      
      for (NodeSet::iterator NI = Tmp.begin(), NE = Tmp.end(); NI != NE; ++NI)
        VisitObjCMessageExprDispatchHelper(ME, *NI, Dst);
      
      return;
    }
    
    VisitObjCMessageExprDispatchHelper(ME, Pred, Dst);
    return;
  }
  
  NodeSet Tmp;
  Visit(*AI, Pred, Tmp);
  
  ++AI;
  
  for (NodeSet::iterator NI = Tmp.begin(), NE = Tmp.end(); NI != NE; ++NI)
    VisitObjCMessageExprArgHelper(ME, AI, AE, *NI, Dst);
}

void GRExprEngine::VisitObjCMessageExprDispatchHelper(ObjCMessageExpr* ME,
                                                      NodeTy* Pred,
                                                      NodeSet& Dst) {
  
  // FIXME: More logic for the processing the method call. 
  
  const ValueState* St = GetState(Pred);
  bool RaisesException = false;
  
  
  if (Expr* Receiver = ME->getReceiver()) {
    
    RVal L = GetRVal(St, Receiver);
    
    // Check for undefined control-flow or calls to NULL.
    
    if (L.isUndef()) {
      NodeTy* N = Builder->generateNode(ME, St, Pred);
      
      if (N) {
        N->markAsSink();
        UndefReceivers.insert(N);
      }
      
      return;
    }
    
    // Check if the "raise" message was sent.
    if (ME->getSelector() == RaiseSel)
      RaisesException = true;
  }
  else {
    
    IdentifierInfo* ClsName = ME->getClassName();
    Selector S = ME->getSelector();
    
    // Check for special instance methods.
        
    if (!NSExceptionII) {      
      ASTContext& Ctx = getContext();
      
      NSExceptionII = &Ctx.Idents.get("NSException");
    }
    
    if (ClsName == NSExceptionII) {
        
      enum { NUM_RAISE_SELECTORS = 2 };
      
      // Lazily create a cache of the selectors.

      if (!NSExceptionInstanceRaiseSelectors) {
        
        ASTContext& Ctx = getContext();
        
        NSExceptionInstanceRaiseSelectors = new Selector[NUM_RAISE_SELECTORS];
      
        llvm::SmallVector<IdentifierInfo*, NUM_RAISE_SELECTORS> II;
        unsigned idx = 0;
        
        // raise:format:      
        II.push_back(&Ctx.Idents.get("raise"));
        II.push_back(&Ctx.Idents.get("format"));      
        NSExceptionInstanceRaiseSelectors[idx++] =
          Ctx.Selectors.getSelector(II.size(), &II[0]);      
        
        // raise:format::arguments:      
        II.push_back(&Ctx.Idents.get("arguments"));
        NSExceptionInstanceRaiseSelectors[idx++] =
          Ctx.Selectors.getSelector(II.size(), &II[0]);
      }
      
      for (unsigned i = 0; i < NUM_RAISE_SELECTORS; ++i)
        if (S == NSExceptionInstanceRaiseSelectors[i]) {
          RaisesException = true; break;
        }
    }
  }
  
  // Check for any arguments that are uninitialized/undefined.
  
  for (ObjCMessageExpr::arg_iterator I = ME->arg_begin(), E = ME->arg_end();
       I != E; ++I) {
    
    if (GetRVal(St, *I).isUndef()) {
      
      // Generate an error node for passing an uninitialized/undefined value
      // as an argument to a message expression.  This node is a sink.
      NodeTy* N = Builder->generateNode(ME, St, Pred);
      
      if (N) {
        N->markAsSink();
        MsgExprUndefArgs[N] = *I;
      }
      
      return;
    }    
  }
  
  // Check if we raise an exception.  For now treat these as sinks.  Eventually
  // we will want to handle exceptions properly.
  
  SaveAndRestore<bool> OldSink(Builder->BuildSinks);

  if (RaisesException)
    Builder->BuildSinks = true;
  
  // Dispatch to plug-in transfer function.
  
  unsigned size = Dst.size();
  SaveOr OldHasGen(Builder->HasGeneratedNode);
 
  EvalObjCMessageExpr(Dst, ME, Pred);
  
  // Handle the case where no nodes where generated.  Auto-generate that
  // contains the updated state if we aren't generating sinks.
  
  if (!Builder->BuildSinks && Dst.size() == size && !Builder->HasGeneratedNode)
    MakeNode(Dst, ME, Pred, St);
}

//===----------------------------------------------------------------------===//
// Transfer functions: Miscellaneous statements.
//===----------------------------------------------------------------------===//

void GRExprEngine::VisitCast(Expr* CastE, Expr* Ex, NodeTy* Pred, NodeSet& Dst){
  
  NodeSet S1;
  QualType T = CastE->getType();
  
  if (T->isReferenceType())
    VisitLVal(Ex, Pred, S1);
  else
    Visit(Ex, Pred, S1);
  
  // Check for casting to "void".
  if (T->isVoidType()) {
    
    for (NodeSet::iterator I1 = S1.begin(), E1 = S1.end(); I1 != E1; ++I1)
      Dst.Add(*I1);

    return;
  }
  
  // FIXME: The rest of this should probably just go into EvalCall, and
  //   let the transfer function object be responsible for constructing
  //   nodes.
  
  QualType ExTy = Ex->getType();
  
  for (NodeSet::iterator I1 = S1.begin(), E1 = S1.end(); I1 != E1; ++I1) {
    NodeTy* N = *I1;
    const ValueState* St = GetState(N);
    RVal V = GetRVal(St, Ex);

    // Unknown?
    
    if (V.isUnknown()) {
      Dst.Add(N);
      continue;
    }
    
    // Undefined?
    
    if (V.isUndef()) {
      MakeNode(Dst, CastE, N, SetRVal(St, CastE, V));
      continue;
    }
  
    // Check for casts from pointers to integers.
    if (T->isIntegerType() && LVal::IsLValType(ExTy)) {
      unsigned bits = getContext().getTypeSize(ExTy);
    
      // FIXME: Determine if the number of bits of the target type is 
      // equal or exceeds the number of bits to store the pointer value.
      // If not, flag an error.
      
      V = nonlval::LValAsInteger::Make(getBasicVals(), cast<LVal>(V), bits);
      MakeNode(Dst, CastE, N, SetRVal(St, CastE, V));
      continue;
    }
    
    // Check for casts from integers to pointers.
    if (LVal::IsLValType(T) && ExTy->isIntegerType())
      if (nonlval::LValAsInteger *LV = dyn_cast<nonlval::LValAsInteger>(&V)) {
        // Just unpackage the lval and return it.
        V = LV->getLVal();
        MakeNode(Dst, CastE, N, SetRVal(St, CastE, V));
        continue;
      }
    
    // All other cases.
    
    MakeNode(Dst, CastE, N, SetRVal(St, CastE, EvalCast(V, CastE->getType())));
  }
}

void GRExprEngine::VisitDeclStmt(DeclStmt* DS, NodeTy* Pred, NodeSet& Dst) {  
  VisitDeclStmtAux(DS, DS->getDecl(), Pred, Dst);
}

void GRExprEngine::VisitDeclStmtAux(DeclStmt* DS, ScopedDecl* D,
                                    NodeTy* Pred, NodeSet& Dst) {

  if (!D)
    return;
  
  if (!isa<VarDecl>(D)) {
    VisitDeclStmtAux(DS, D->getNextDeclarator(), Pred, Dst);
    return;
  }
  
  const VarDecl* VD = dyn_cast<VarDecl>(D);
  
  // FIXME: Add support for local arrays.
  if (VD->getType()->isArrayType()) {
    VisitDeclStmtAux(DS, D->getNextDeclarator(), Pred, Dst);
    return;
  }
  
  Expr* Ex = const_cast<Expr*>(VD->getInit());

  // FIXME: static variables may have an initializer, but the second
  //  time a function is called those values may not be current.
  NodeSet Tmp;

  if (Ex) Visit(Ex, Pred, Tmp);
  if (Tmp.empty()) Tmp.Add(Pred);
  
  for (NodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {
      
    const ValueState* St = GetState(*I);

    if (!Ex && VD->hasGlobalStorage()) {
      
      // Handle variables with global storage and no initializers.
      
      // FIXME: static variables may have an initializer, but the second
      //  time a function is called those values may not be current.
      
      
      // In this context, Static => Local variable.
      
      assert (!VD->getStorageClass() == VarDecl::Static ||
              !VD->isFileVarDecl());
      
      // If there is no initializer, set the value of the
      // variable to "Undefined".
      
      if (VD->getStorageClass() == VarDecl::Static) {
          
        // C99: 6.7.8 Initialization
        //  If an object that has static storage duration is not initialized
        //  explicitly, then: 
        //   —if it has pointer type, it is initialized to a null pointer; 
        //   —if it has arithmetic type, it is initialized to (positive or 
        //     unsigned) zero; 
        
        // FIXME: Handle structs.  Now we treat their values as unknown.

        QualType T = VD->getType();
        
        if (LVal::IsLValType(T))
          St = SetRVal(St, lval::DeclVal(VD),
                       lval::ConcreteInt(getBasicVals().getValue(0, T)));
        else if (T->isIntegerType())
          St = SetRVal(St, lval::DeclVal(VD),
                       nonlval::ConcreteInt(getBasicVals().getValue(0, T)));          
          
        // FIXME: Handle structs.  Now we treat them as unknown.  What
        //  we need to do is treat their members as unknown.
      }
    }
    else {
      
      // FIXME: Handle structs.  Now we treat them as unknown.  What
      //  we need to do is treat their members as unknown.

      QualType T = VD->getType();

      if (LVal::IsLValType(T) || T->isIntegerType()) {
        
        RVal V = Ex ? GetRVal(St, Ex) : UndefinedVal();
        
        if (Ex && V.isUnknown()) {
          
          // EXPERIMENTAL: "Conjured" symbols.

          unsigned Count = Builder->getCurrentBlockCount();
          SymbolID Sym = SymMgr.getConjuredSymbol(Ex, Count);
          
          V = LVal::IsLValType(Ex->getType())
            ? cast<RVal>(lval::SymbolVal(Sym)) 
            : cast<RVal>(nonlval::SymbolVal(Sym));            
        }
        
        St = SetRVal(St, lval::DeclVal(VD), V);        
      }
    }

    // Create a new node.  We don't really need to create a new NodeSet
    // here, but it simplifies things and doesn't cost much.
    NodeSet Tmp2;    
    MakeNode(Tmp2, DS, *I, St);
    if (Tmp2.empty()) Tmp2.Add(*I);
    
    for (NodeSet::iterator I2=Tmp2.begin(), E2=Tmp2.end(); I2!=E2; ++I2)
      VisitDeclStmtAux(DS, D->getNextDeclarator(), *I2, Dst);
  }
}


/// VisitSizeOfAlignOfTypeExpr - Transfer function for sizeof(type).
void GRExprEngine::VisitSizeOfAlignOfTypeExpr(SizeOfAlignOfTypeExpr* Ex,
                                              NodeTy* Pred,
                                              NodeSet& Dst) {
  QualType T = Ex->getArgumentType();
  uint64_t amt;  
  
  if (Ex->isSizeOf()) {

    // FIXME: Add support for VLAs.
    if (!T.getTypePtr()->isConstantSizeType())
      return;
    
    // Some code tries to take the sizeof an ObjCInterfaceType, relying that
    // the compiler has laid out its representation.  Just report Unknown
    // for these.
    if (T->isObjCInterfaceType())
      return;
    
    amt = 1;  // Handle sizeof(void)
    
    if (T != getContext().VoidTy)
      amt = getContext().getTypeSize(T) / 8;
    
  }
  else  // Get alignment of the type.
    amt = getContext().getTypeAlign(T) / 8;
  
  MakeNode(Dst, Ex, Pred,
           SetRVal(GetState(Pred), Ex,
                   NonLVal::MakeVal(getBasicVals(), amt, Ex->getType())));  
}


void GRExprEngine::VisitUnaryOperator(UnaryOperator* U, NodeTy* Pred,
                                      NodeSet& Dst, bool asLVal) {

  switch (U->getOpcode()) {
      
    default:
      break;
          
    case UnaryOperator::Deref: {
      
      Expr* Ex = U->getSubExpr()->IgnoreParens();
      NodeSet Tmp;
      Visit(Ex, Pred, Tmp);
      
      for (NodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {
        
        const ValueState* St = GetState(*I);
        RVal location = GetRVal(St, Ex);
        
        if (asLVal)
          MakeNode(Dst, U, *I, SetRVal(St, U, location));
        else
          EvalLoad(Dst, U, *I, St, location);
      } 

      return;
    }
      
    case UnaryOperator::Real: {
      
      Expr* Ex = U->getSubExpr()->IgnoreParens();
      NodeSet Tmp;
      Visit(Ex, Pred, Tmp);
      
      for (NodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {
        
        // FIXME: We don't have complex RValues yet.
        if (Ex->getType()->isAnyComplexType()) {
          // Just report "Unknown."
          Dst.Add(*I);
          continue;
        }
        
        // For all other types, UnaryOperator::Real is an identity operation.
        assert (U->getType() == Ex->getType());
        const ValueState* St = GetState(*I);
        MakeNode(Dst, U, *I, SetRVal(St, U, GetRVal(St, Ex)));
      } 
      
      return;
    }
      
    case UnaryOperator::Imag: {
      
      Expr* Ex = U->getSubExpr()->IgnoreParens();
      NodeSet Tmp;
      Visit(Ex, Pred, Tmp);
      
      for (NodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {
        // FIXME: We don't have complex RValues yet.
        if (Ex->getType()->isAnyComplexType()) {
          // Just report "Unknown."
          Dst.Add(*I);
          continue;
        }
        
        // For all other types, UnaryOperator::Float returns 0.
        assert (Ex->getType()->isIntegerType());
        const ValueState* St = GetState(*I);
        RVal X = NonLVal::MakeVal(getBasicVals(), 0, Ex->getType());
        MakeNode(Dst, U, *I, SetRVal(St, U, X));
      }
      
      return;
    }
      
      // FIXME: Just report "Unknown" for OffsetOf.      
    case UnaryOperator::OffsetOf:
      Dst.Add(Pred);
      return;
      
    case UnaryOperator::Plus: assert (!asLVal);  // FALL-THROUGH.
    case UnaryOperator::Extension: {
      
      // Unary "+" is a no-op, similar to a parentheses.  We still have places
      // where it may be a block-level expression, so we need to
      // generate an extra node that just propagates the value of the
      // subexpression.

      Expr* Ex = U->getSubExpr()->IgnoreParens();
      NodeSet Tmp;
      Visit(Ex, Pred, Tmp);
      
      for (NodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {        
        const ValueState* St = GetState(*I);
        MakeNode(Dst, U, *I, SetRVal(St, U, GetRVal(St, Ex)));
      }
      
      return;
    }
    
    case UnaryOperator::AddrOf: {
      
      assert (!asLVal);
      Expr* Ex = U->getSubExpr()->IgnoreParens();
      NodeSet Tmp;
      VisitLVal(Ex, Pred, Tmp);
     
      for (NodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {        
        const ValueState* St = GetState(*I);
        RVal V = GetRVal(St, Ex);
        St = SetRVal(St, U, V);
        MakeNode(Dst, U, *I, St);
      }

      return; 
    }
      
    case UnaryOperator::LNot:
    case UnaryOperator::Minus:
    case UnaryOperator::Not: {
      
      assert (!asLVal);
      Expr* Ex = U->getSubExpr()->IgnoreParens();
      NodeSet Tmp;
      Visit(Ex, Pred, Tmp);
      
      for (NodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {        
        const ValueState* St = GetState(*I);
        RVal V = GetRVal(St, Ex);
        
        if (V.isUnknownOrUndef()) {
          MakeNode(Dst, U, *I, SetRVal(St, U, V));
          continue;
        }
        
        switch (U->getOpcode()) {
          default:
            assert(false && "Invalid Opcode.");
            break;
            
          case UnaryOperator::Not:
            St = SetRVal(St, U, EvalComplement(cast<NonLVal>(V)));
            break;            
            
          case UnaryOperator::Minus:
            St = SetRVal(St, U, EvalMinus(U, cast<NonLVal>(V)));
            break;   
            
          case UnaryOperator::LNot:   
            
            // C99 6.5.3.3: "The expression !E is equivalent to (0==E)."
            //
            //  Note: technically we do "E == 0", but this is the same in the
            //    transfer functions as "0 == E".
            
            if (isa<LVal>(V)) {
              lval::ConcreteInt X(getBasicVals().getZeroWithPtrWidth());
              RVal Result = EvalBinOp(BinaryOperator::EQ, cast<LVal>(V), X);
              St = SetRVal(St, U, Result);
            }
            else {
              nonlval::ConcreteInt X(getBasicVals().getValue(0, Ex->getType()));
#if 0            
              RVal Result = EvalBinOp(BinaryOperator::EQ, cast<NonLVal>(V), X);
              St = SetRVal(St, U, Result);
#else
              EvalBinOp(Dst, U, BinaryOperator::EQ, cast<NonLVal>(V), X, *I);
              continue;
#endif
            }
            
            break;
        }
        
        MakeNode(Dst, U, *I, St);
      }
      
      return;
    }
      
    case UnaryOperator::SizeOf: {
            
      QualType T = U->getSubExpr()->getType();
        
      // FIXME: Add support for VLAs.
      
      if (!T.getTypePtr()->isConstantSizeType())
        return;
        
      uint64_t size = getContext().getTypeSize(T) / 8;                
      const ValueState* St = GetState(Pred);
      St = SetRVal(St, U, NonLVal::MakeVal(getBasicVals(), size, U->getType()));
        
      MakeNode(Dst, U, Pred, St);
      return;
    }
  }

  // Handle ++ and -- (both pre- and post-increment).

  assert (U->isIncrementDecrementOp());
  NodeSet Tmp;
  Expr* Ex = U->getSubExpr()->IgnoreParens();
  VisitLVal(Ex, Pred, Tmp);
  
  for (NodeSet::iterator I = Tmp.begin(), E = Tmp.end(); I!=E; ++I) {
    
    const ValueState* St = GetState(*I);
    RVal V1 = GetRVal(St, Ex);
    
    // Perform a load.      
    NodeSet Tmp2;
    EvalLoad(Tmp2, Ex, *I, St, V1);

    for (NodeSet::iterator I2 = Tmp2.begin(), E2 = Tmp2.end(); I2!=E2; ++I2) {
        
      St = GetState(*I2);
      RVal V2 = GetRVal(St, Ex);
        
      // Propagate unknown and undefined values.      
      if (V2.isUnknownOrUndef()) {
        MakeNode(Dst, U, *I2, SetRVal(St, U, V2));
        continue;
      }
      
      // Handle all other values.
      
      BinaryOperator::Opcode Op = U->isIncrementOp() ? BinaryOperator::Add
                                                     : BinaryOperator::Sub;
      
      RVal Result = EvalBinOp(Op, V2, MakeConstantVal(1U, U));      
      St = SetRVal(St, U, U->isPostfix() ? V2 : Result);

      // Perform the store.      
      EvalStore(Dst, U, *I2, St, V1, Result);
    }
  }
}

void GRExprEngine::VisitAsmStmt(AsmStmt* A, NodeTy* Pred, NodeSet& Dst) {
  VisitAsmStmtHelperOutputs(A, A->begin_outputs(), A->end_outputs(), Pred, Dst);
}  

void GRExprEngine::VisitAsmStmtHelperOutputs(AsmStmt* A,
                                             AsmStmt::outputs_iterator I,
                                             AsmStmt::outputs_iterator E,
                                             NodeTy* Pred, NodeSet& Dst) {
  if (I == E) {
    VisitAsmStmtHelperInputs(A, A->begin_inputs(), A->end_inputs(), Pred, Dst);
    return;
  }
  
  NodeSet Tmp;
  VisitLVal(*I, Pred, Tmp);
  
  ++I;
  
  for (NodeSet::iterator NI = Tmp.begin(), NE = Tmp.end(); NI != NE; ++NI)
    VisitAsmStmtHelperOutputs(A, I, E, *NI, Dst);
}

void GRExprEngine::VisitAsmStmtHelperInputs(AsmStmt* A,
                                            AsmStmt::inputs_iterator I,
                                            AsmStmt::inputs_iterator E,
                                            NodeTy* Pred, NodeSet& Dst) {
  if (I == E) {
    
    // We have processed both the inputs and the outputs.  All of the outputs
    // should evaluate to LVals.  Nuke all of their values.
    
    // FIXME: Some day in the future it would be nice to allow a "plug-in"
    // which interprets the inline asm and stores proper results in the
    // outputs.
    
    const ValueState* St = GetState(Pred);
    
    for (AsmStmt::outputs_iterator OI = A->begin_outputs(),
                                   OE = A->end_outputs(); OI != OE; ++OI) {
      
      RVal X = GetRVal(St, *OI);      
      assert (!isa<NonLVal>(X));  // Should be an Lval, or unknown, undef.
      
      if (isa<LVal>(X))
        St = SetRVal(St, cast<LVal>(X), UnknownVal());
    }
    
    MakeNode(Dst, A, Pred, St);
    return;
  }
  
  NodeSet Tmp;
  Visit(*I, Pred, Tmp);
  
  ++I;
  
  for (NodeSet::iterator NI = Tmp.begin(), NE = Tmp.end(); NI != NE; ++NI)
    VisitAsmStmtHelperInputs(A, I, E, *NI, Dst);
}

void GRExprEngine::EvalReturn(NodeSet& Dst, ReturnStmt* S, NodeTy* Pred) {
  assert (Builder && "GRStmtNodeBuilder must be defined.");
  
  unsigned size = Dst.size();  

  SaveAndRestore<bool> OldSink(Builder->BuildSinks);
  SaveOr OldHasGen(Builder->HasGeneratedNode);

  getTF().EvalReturn(Dst, *this, *Builder, S, Pred);
  
  // Handle the case where no nodes where generated.
  
  if (!Builder->BuildSinks && Dst.size() == size && !Builder->HasGeneratedNode)
    MakeNode(Dst, S, Pred, GetState(Pred));
}

void GRExprEngine::VisitReturnStmt(ReturnStmt* S, NodeTy* Pred, NodeSet& Dst) {

  Expr* R = S->getRetValue();
  
  if (!R) {
    EvalReturn(Dst, S, Pred);
    return;
  }

  NodeSet DstRet;
  QualType T = R->getType();
  
  if (T->isPointerLikeType()) {
    
    // Check if any of the return values return the address of a stack variable.
    
    NodeSet Tmp;
    Visit(R, Pred, Tmp);
    
    for (NodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {
      RVal X = GetRVal((*I)->getState(), R);

      if (isa<lval::DeclVal>(X)) {
        
        if (cast<lval::DeclVal>(X).getDecl()->hasLocalStorage()) {
        
          // Create a special node representing the v
          
          NodeTy* RetStackNode = Builder->generateNode(S, GetState(*I), *I);
          
          if (RetStackNode) {
            RetStackNode->markAsSink();
            RetsStackAddr.insert(RetStackNode);
          }
          
          continue;
        }
      }
      
      DstRet.Add(*I);
    }
  }
  else
    Visit(R, Pred, DstRet);
  
  for (NodeSet::iterator I=DstRet.begin(), E=DstRet.end(); I!=E; ++I)
    EvalReturn(Dst, S, *I);
}

//===----------------------------------------------------------------------===//
// Transfer functions: Binary operators.
//===----------------------------------------------------------------------===//

bool GRExprEngine::CheckDivideZero(Expr* Ex, const ValueState* St,
                                   NodeTy* Pred, RVal Denom) {
  
  // Divide by undefined? (potentially zero)
  
  if (Denom.isUndef()) {
    NodeTy* DivUndef = Builder->generateNode(Ex, St, Pred);
    
    if (DivUndef) {
      DivUndef->markAsSink();
      ExplicitBadDivides.insert(DivUndef);
    }
    
    return true;
  }
  
  // Check for divide/remainder-by-zero.
  // First, "assume" that the denominator is 0 or undefined.            
  
  bool isFeasibleZero = false;
  const ValueState* ZeroSt =  Assume(St, Denom, false, isFeasibleZero);
  
  // Second, "assume" that the denominator cannot be 0.            
  
  bool isFeasibleNotZero = false;
  St = Assume(St, Denom, true, isFeasibleNotZero);
  
  // Create the node for the divide-by-zero (if it occurred).
  
  if (isFeasibleZero)
    if (NodeTy* DivZeroNode = Builder->generateNode(Ex, ZeroSt, Pred)) {
      DivZeroNode->markAsSink();
      
      if (isFeasibleNotZero)
        ImplicitBadDivides.insert(DivZeroNode);
      else
        ExplicitBadDivides.insert(DivZeroNode);
      
    }
  
  return !isFeasibleNotZero;
}

void GRExprEngine::VisitBinaryOperator(BinaryOperator* B,
                                       GRExprEngine::NodeTy* Pred,
                                       GRExprEngine::NodeSet& Dst) {

  NodeSet Tmp1;
  Expr* LHS = B->getLHS()->IgnoreParens();
  Expr* RHS = B->getRHS()->IgnoreParens();
  
  if (B->isAssignmentOp())
    VisitLVal(LHS, Pred, Tmp1);
  else
    Visit(LHS, Pred, Tmp1);

  for (NodeSet::iterator I1=Tmp1.begin(), E1=Tmp1.end(); I1 != E1; ++I1) {

    RVal LeftV = GetRVal((*I1)->getState(), LHS);
    
    // Process the RHS.
    
    NodeSet Tmp2;
    Visit(RHS, *I1, Tmp2);
    
    // With both the LHS and RHS evaluated, process the operation itself.
    
    for (NodeSet::iterator I2=Tmp2.begin(), E2=Tmp2.end(); I2 != E2; ++I2) {

      const ValueState* St = GetState(*I2);
      RVal RightV = GetRVal(St, RHS);
      BinaryOperator::Opcode Op = B->getOpcode();
      
      switch (Op) {
          
        case BinaryOperator::Assign: {
          
          // EXPERIMENTAL: "Conjured" symbols.
          
          if (RightV.isUnknown()) {            
            unsigned Count = Builder->getCurrentBlockCount();
            SymbolID Sym = SymMgr.getConjuredSymbol(B->getRHS(), Count);
            
            RightV = LVal::IsLValType(B->getRHS()->getType()) 
                   ? cast<RVal>(lval::SymbolVal(Sym)) 
                   : cast<RVal>(nonlval::SymbolVal(Sym));            
          }
          
          // Simulate the effects of a "store":  bind the value of the RHS
          // to the L-Value represented by the LHS.
          
          EvalStore(Dst, B, *I2, SetRVal(St, B, RightV), LeftV, RightV);          
          continue;
        }
          
        case BinaryOperator::Div:
        case BinaryOperator::Rem:
          
          // Special checking for integer denominators.
          
          if (RHS->getType()->isIntegerType()
              && CheckDivideZero(B, St, *I2, RightV))
            continue;
          
          // FALL-THROUGH.

        default: {
      
          if (B->isAssignmentOp())
            break;
          
          // Process non-assignements except commas or short-circuited
          // logical expressions (LAnd and LOr).
          
          RVal Result = EvalBinOp(Op, LeftV, RightV);
          
          if (Result.isUnknown()) {
            Dst.Add(*I2);
            continue;
          }
          
          if (Result.isUndef() && !LeftV.isUndef() && !RightV.isUndef()) {
            
            // The operands were *not* undefined, but the result is undefined.
            // This is a special node that should be flagged as an error.
            
            if (NodeTy* UndefNode = Builder->generateNode(B, St, *I2)) {
              UndefNode->markAsSink();            
              UndefResults.insert(UndefNode);
            }
            
            continue;
          }
          
          // Otherwise, create a new node.
          
          MakeNode(Dst, B, *I2, SetRVal(St, B, Result));
          continue;
        }
      }
    
      assert (B->isCompoundAssignmentOp());

      if (Op >= BinaryOperator::AndAssign)
        ((int&) Op) -= (BinaryOperator::AndAssign - BinaryOperator::And);
      else
        ((int&) Op) -= BinaryOperator::MulAssign;  
          
      // Perform a load (the LHS).  This performs the checks for
      // null dereferences, and so on.
      NodeSet Tmp3;
      RVal location = GetRVal(St, LHS);
      EvalLoad(Tmp3, LHS, *I2, St, location);
      
      for (NodeSet::iterator I3=Tmp3.begin(), E3=Tmp3.end(); I3!=E3; ++I3) {
        
        St = GetState(*I3);
        RVal V = GetRVal(St, LHS);

        // Propagate undefined values (left-side).          
        if (V.isUndef()) {
          EvalStore(Dst, B, *I3, SetRVal(St, B, V), location, V);
          continue;
        }
        
        // Propagate unknown values (left and right-side).
        if (RightV.isUnknown() || V.isUnknown()) {
          EvalStore(Dst, B, *I3, SetRVal(St, B, UnknownVal()), location,
                    UnknownVal());
          continue;
        }

        // At this point:
        //
        //  The LHS is not Undef/Unknown.
        //  The RHS is not Unknown.
        
        // Get the computation type.
        QualType CTy = cast<CompoundAssignOperator>(B)->getComputationType();
          
        // Perform promotions.
        V = EvalCast(V, CTy);
        RightV = EvalCast(RightV, CTy);
          
        // Evaluate operands and promote to result type.                    

        if ((Op == BinaryOperator::Div || Op == BinaryOperator::Rem)
             && RHS->getType()->isIntegerType()) {
          
          if (CheckDivideZero(B, St, *I3, RightV))
            continue;
        }
        else if (RightV.isUndef()) {
            
          // Propagate undefined values (right-side).
          
          EvalStore(Dst, B, *I3, SetRVal(St, B, RightV), location, RightV);
          continue;
        }
      
        // Compute the result of the operation.
      
        RVal Result = EvalCast(EvalBinOp(Op, V, RightV), B->getType());
          
        if (Result.isUndef()) {
            
          // The operands were not undefined, but the result is undefined.
          
          if (NodeTy* UndefNode = Builder->generateNode(B, St, *I3)) {
            UndefNode->markAsSink();            
            UndefResults.insert(UndefNode);
          }
          
          continue;
        }
 
        EvalStore(Dst, B, *I3, SetRVal(St, B, Result), location, Result);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Transfer-function Helpers.
//===----------------------------------------------------------------------===//

void GRExprEngine::EvalBinOp(ExplodedNodeSet<ValueState>& Dst, Expr* Ex,
                             BinaryOperator::Opcode Op,
                             NonLVal L, NonLVal R,
                             ExplodedNode<ValueState>* Pred) {

  ValueStateSet OStates;
  EvalBinOp(OStates, GetState(Pred), Ex, Op, L, R);

  for (ValueStateSet::iterator I=OStates.begin(), E=OStates.end(); I!=E; ++I)
    MakeNode(Dst, Ex, Pred, *I);
}

void GRExprEngine::EvalBinOp(ValueStateSet& OStates, const ValueState* St,
                             Expr* Ex, BinaryOperator::Opcode Op,
                             NonLVal L, NonLVal R) {
  
  ValueStateSet::AutoPopulate AP(OStates, St);
  if (R.isValid()) getTF().EvalBinOpNN(OStates, StateMgr, St, Ex, Op, L, R);
}

//===----------------------------------------------------------------------===//
// Visualization.
//===----------------------------------------------------------------------===//

#ifndef NDEBUG
static GRExprEngine* GraphPrintCheckerState;
static SourceManager* GraphPrintSourceManager;
static ValueState::CheckerStatePrinter* GraphCheckerStatePrinter;

namespace llvm {
template<>
struct VISIBILITY_HIDDEN DOTGraphTraits<GRExprEngine::NodeTy*> :
  public DefaultDOTGraphTraits {
    
  static void PrintVarBindings(std::ostream& Out, ValueState* St) {

    Out << "Variables:\\l";
    
    bool isFirst = true;
    
    for (ValueState::vb_iterator I=St->vb_begin(), E=St->vb_end(); I!=E;++I) {        

      if (isFirst)
        isFirst = false;
      else
        Out << "\\l";
      
      Out << ' ' << I.getKey()->getName() << " : ";
      I.getData().print(Out);
    }
    
  }
    
    
  static void PrintSubExprBindings(std::ostream& Out, ValueState* St){
    
    bool isFirst = true;
    
    for (ValueState::seb_iterator I=St->seb_begin(), E=St->seb_end();I!=E;++I) {        
      
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
    
  static void PrintBlkExprBindings(std::ostream& Out, ValueState* St){
        
    bool isFirst = true;

    for (ValueState::beb_iterator I=St->beb_begin(), E=St->beb_end(); I!=E;++I){      
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
    
  static void PrintEQ(std::ostream& Out, ValueState* St) {
    ValueState::ConstEqTy CE = St->ConstEq;
    
    if (CE.isEmpty())
      return;
    
    Out << "\\l\\|'==' constraints:";

    for (ValueState::ConstEqTy::iterator I=CE.begin(), E=CE.end(); I!=E;++I)
      Out << "\\l $" << I.getKey() << " : " << I.getData()->toString();
  }
    
  static void PrintNE(std::ostream& Out, ValueState* St) {
    ValueState::ConstNotEqTy NE = St->ConstNotEq;
    
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
        GraphPrintCheckerState->isUndefDeref(N) ||
        GraphPrintCheckerState->isUndefStore(N) ||
        GraphPrintCheckerState->isUndefControlFlow(N) ||
        GraphPrintCheckerState->isExplicitBadDivide(N) ||
        GraphPrintCheckerState->isImplicitBadDivide(N) ||
        GraphPrintCheckerState->isUndefResult(N) ||
        GraphPrintCheckerState->isBadCall(N) ||
        GraphPrintCheckerState->isUndefArg(N))
      return "color=\"red\",style=\"filled\"";
    
    if (GraphPrintCheckerState->isNoReturnCall(N))
      return "color=\"blue\",style=\"filled\"";
    
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
        
      case ProgramPoint::PostLoadKind:
      case ProgramPoint::PostPurgeDeadSymbolsKind:
      case ProgramPoint::PostStmtKind: {
        const PostStmt& L = cast<PostStmt>(Loc);        
        Stmt* S = L.getStmt();
        SourceLocation SLoc = S->getLocStart();

        Out << S->getStmtClassName() << ' ' << (void*) S << ' ';        
        S->printPretty(Out);
        
        if (SLoc.isFileID()) {        
          Out << "\\lline="
            << GraphPrintSourceManager->getLineNumber(SLoc) << " col="
            << GraphPrintSourceManager->getColumnNumber(SLoc) << "\\l";          
        }
        
        if (GraphPrintCheckerState->isImplicitNullDeref(N))
          Out << "\\|Implicit-Null Dereference.\\l";
        else if (GraphPrintCheckerState->isExplicitNullDeref(N))
          Out << "\\|Explicit-Null Dereference.\\l";
        else if (GraphPrintCheckerState->isUndefDeref(N))
          Out << "\\|Dereference of undefialied value.\\l";
        else if (GraphPrintCheckerState->isUndefStore(N))
          Out << "\\|Store to Undefined LVal.";
        else if (GraphPrintCheckerState->isExplicitBadDivide(N))
          Out << "\\|Explicit divide-by zero or undefined value.";
        else if (GraphPrintCheckerState->isImplicitBadDivide(N))
          Out << "\\|Implicit divide-by zero or undefined value.";
        else if (GraphPrintCheckerState->isUndefResult(N))
          Out << "\\|Result of operation is undefined.";
        else if (GraphPrintCheckerState->isNoReturnCall(N))
          Out << "\\|Call to function marked \"noreturn\".";
        else if (GraphPrintCheckerState->isBadCall(N))
          Out << "\\|Call to NULL/Undefined.";
        else if (GraphPrintCheckerState->isUndefArg(N))
          Out << "\\|Argument in call is undefined";
        
        break;
      }
    
      default: {
        const BlockEdge& E = cast<BlockEdge>(Loc);
        Out << "Edge: (B" << E.getSrc()->getBlockID() << ", B"
            << E.getDst()->getBlockID()  << ')';
        
        if (Stmt* T = E.getSrc()->getTerminator()) {
          
          SourceLocation SLoc = T->getLocStart();
         
          Out << "\\|Terminator: ";
          
          E.getSrc()->printTerminator(Out);
          
          if (SLoc.isFileID()) {
            Out << "\\lline="
              << GraphPrintSourceManager->getLineNumber(SLoc) << " col="
              << GraphPrintSourceManager->getColumnNumber(SLoc);
          }
            
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
        
        if (GraphPrintCheckerState->isUndefControlFlow(N)) {
          Out << "\\|Control-flow based on\\lUndefined value.\\l";
        }
      }
    }
    
    Out << "\\|StateID: " << (void*) N->getState() << "\\|";

    N->getState()->printDOT(Out, GraphCheckerStatePrinter);
      
    Out << "\\l";
    return Out.str();
  }
};
} // end llvm namespace    
#endif

#ifndef NDEBUG

template <typename ITERATOR>
GRExprEngine::NodeTy* GetGraphNode(ITERATOR I) { return *I; }

template <>
GRExprEngine::NodeTy*
GetGraphNode<llvm::DenseMap<GRExprEngine::NodeTy*, Expr*>::iterator>
  (llvm::DenseMap<GRExprEngine::NodeTy*, Expr*>::iterator I) {
  return I->first;
}

template <typename ITERATOR>
static void AddSources(std::vector<GRExprEngine::NodeTy*>& Sources,
                       ITERATOR I, ITERATOR E) {
  
  llvm::SmallPtrSet<void*,10> CachedSources;
  
  for ( ; I != E; ++I ) {
    GRExprEngine::NodeTy* N = GetGraphNode(I);
    void* p = N->getLocation().getRawData();
    
    if (CachedSources.count(p))
      continue;
    
    CachedSources.insert(p);
    
    Sources.push_back(N);
  }
}
#endif

void GRExprEngine::ViewGraph(bool trim) {
#ifndef NDEBUG  
  if (trim) {
    std::vector<NodeTy*> Src;
    
    // Fixme: Migrate over to the new way of adding nodes.
    AddSources(Src, null_derefs_begin(), null_derefs_end());
    AddSources(Src, undef_derefs_begin(), undef_derefs_end());
    AddSources(Src, explicit_bad_divides_begin(), explicit_bad_divides_end());
    AddSources(Src, undef_results_begin(), undef_results_end());
    AddSources(Src, bad_calls_begin(), bad_calls_end());
    AddSources(Src, undef_arg_begin(), undef_arg_end());
    AddSources(Src, undef_branches_begin(), undef_branches_end());
    
    // The new way.
    for (BugTypeSet::iterator I=BugTypes.begin(), E=BugTypes.end(); I!=E; ++I)
      (*I)->GetErrorNodes(Src);
      
    
    ViewGraph(&Src[0], &Src[0]+Src.size());
  }
  else {
    GraphPrintCheckerState = this;
    GraphPrintSourceManager = &getContext().getSourceManager();
    GraphCheckerStatePrinter = getTF().getCheckerStatePrinter();

    llvm::ViewGraph(*G.roots_begin(), "GRExprEngine");
    
    GraphPrintCheckerState = NULL;
    GraphPrintSourceManager = NULL;
    GraphCheckerStatePrinter = NULL;
  }
#endif
}

void GRExprEngine::ViewGraph(NodeTy** Beg, NodeTy** End) {
#ifndef NDEBUG
  GraphPrintCheckerState = this;
  GraphPrintSourceManager = &getContext().getSourceManager();
  GraphCheckerStatePrinter = getTF().getCheckerStatePrinter();
  
  GRExprEngine::GraphTy* TrimmedG = G.Trim(Beg, End);

  if (!TrimmedG)
    llvm::cerr << "warning: Trimmed ExplodedGraph is empty.\n";
  else {
    llvm::ViewGraph(*TrimmedG->roots_begin(), "TrimmedGRExprEngine");    
    delete TrimmedG;
  }  
  
  GraphPrintCheckerState = NULL;
  GraphPrintSourceManager = NULL;
  GraphCheckerStatePrinter = NULL;
#endif
}
