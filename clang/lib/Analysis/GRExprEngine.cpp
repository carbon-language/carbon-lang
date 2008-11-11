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

#include "clang/Analysis/PathSensitive/GRExprEngine.h"
#include "clang/Analysis/PathSensitive/BugReporter.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/Streams.h"
#include "llvm/ADT/ImmutableList.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"

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
  
  virtual bool Audit(NodeTy* N, GRStateManager& VMgr) {
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
                           LiveVariables& L,
                           GRStateManager::StoreManagerCreator SMC)
  : CoreEngine(cfg, CD, Ctx, *this), 
    G(CoreEngine.getGraph()),
    Liveness(L),
    Builder(NULL),
    StateMgr(G.getContext(), SMC,
             CreateBasicConstraintManager, G.getAllocator(), cfg, CD, L),
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
  tf->RegisterChecks(*this);
  tf->RegisterPrinters(getStateManager().Printers);
}

void GRExprEngine::AddCheck(GRSimpleAPICheck* A, Stmt::StmtClass C) {
  if (!BatchAuditor)
    BatchAuditor.reset(new MappedBatchAuditor(getGraph().getAllocator()));
  
  ((MappedBatchAuditor*) BatchAuditor.get())->AddCheck(A, C);
}

const GRState* GRExprEngine::getInitialState() {
  return StateMgr.getInitialState();
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
        const GRState* St = GetState(Pred);
        MakeNode(Dst, B, Pred, BindExpr(St, B, GetSVal(St, B->getRHS())));
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

    case Stmt::CompoundLiteralExprClass:
      VisitCompoundLiteralExpr(cast<CompoundLiteralExpr>(S), Pred, Dst, false);
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
      
    case Stmt::ImplicitCastExprClass:
    case Stmt::CStyleCastExprClass: {
      CastExpr* C = cast<CastExpr>(S);
      VisitCast(C, C->getSubExpr(), Pred, Dst);
      break;
    }

    case Stmt::InitListExprClass:
      VisitInitListExpr(cast<InitListExpr>(S), Pred, Dst);
      break;
      
    case Stmt::MemberExprClass:
      VisitMemberExpr(cast<MemberExpr>(S), Pred, Dst, false);
      break;
      
    case Stmt::ObjCIvarRefExprClass:
      VisitObjCIvarRefExpr(cast<ObjCIvarRefExpr>(S), Pred, Dst, false);
      break;
      
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
      
    case Stmt::SizeOfAlignOfExprClass:
      VisitSizeOfAlignOfExpr(cast<SizeOfAlignOfExpr>(S), Pred, Dst);
      break;
      
    case Stmt::StmtExprClass: {
      StmtExpr* SE = cast<StmtExpr>(S);
      
      const GRState* St = GetState(Pred);
      
      // FIXME: Not certain if we can have empty StmtExprs.  If so, we should
      // probably just remove these from the CFG.
      assert (!SE->getSubStmt()->body_empty());
      
      if (Expr* LastExpr = dyn_cast<Expr>(*SE->getSubStmt()->body_rbegin()))
        MakeNode(Dst, SE, Pred, BindExpr(St, SE, GetSVal(St, LastExpr)));
      else
        Dst.Add(Pred);
      
      break;
    }
      
    case Stmt::UnaryOperatorClass:
      VisitUnaryOperator(cast<UnaryOperator>(S), Pred, Dst, false);
      break;
  }
}

void GRExprEngine::VisitLValue(Expr* Ex, NodeTy* Pred, NodeSet& Dst) {
  
  Ex = Ex->IgnoreParens();
  
  if (Ex != CurrentStmt && getCFG().isBlkExpr(Ex)) {
    Dst.Add(Pred);
    return;
  }
  
  switch (Ex->getStmtClass()) {
      
    case Stmt::ArraySubscriptExprClass:
      VisitArraySubscriptExpr(cast<ArraySubscriptExpr>(Ex), Pred, Dst, true);
      return;
      
    case Stmt::DeclRefExprClass:
      VisitDeclRefExpr(cast<DeclRefExpr>(Ex), Pred, Dst, true);
      return;
      
    case Stmt::ObjCIvarRefExprClass:
      VisitObjCIvarRefExpr(cast<ObjCIvarRefExpr>(Ex), Pred, Dst, true);
      return;
      
    case Stmt::UnaryOperatorClass:
      VisitUnaryOperator(cast<UnaryOperator>(Ex), Pred, Dst, true);
      return;
      
    case Stmt::MemberExprClass:
      VisitMemberExpr(cast<MemberExpr>(Ex), Pred, Dst, true);
      return;
      
    case Stmt::CompoundLiteralExprClass:
      VisitCompoundLiteralExpr(cast<CompoundLiteralExpr>(Ex), Pred, Dst, true);
      return;
      
    case Stmt::ObjCPropertyRefExprClass:
      // FIXME: Property assignments are lvalues, but not really "locations".
      //  e.g.:  self.x = something;
      //  Here the "self.x" really can translate to a method call (setter) when
      //  the assignment is made.  Moreover, the entire assignment expression
      //  evaluate to whatever "something" is, not calling the "getter" for
      //  the property (which would make sense since it can have side effects).
      //  We'll probably treat this as a location, but not one that we can
      //  take the address of.  Perhaps we need a new SVal class for cases
      //  like thsis?
      //  Note that we have a similar problem for bitfields, since they don't
      //  have "locations" in the sense that we can take their address.
      Dst.Add(Pred);
      return;

    case Stmt::StringLiteralClass: {
      const GRState* St = GetState(Pred);
      SVal V = StateMgr.GetLValue(St, cast<StringLiteral>(Ex));
      MakeNode(Dst, Ex, Pred, BindExpr(St, Ex, V));
      return;
    }
      
    default:
      // Arbitrary subexpressions can return aggregate temporaries that
      // can be used in a lvalue context.  We need to enhance our support
      // of such temporaries in both the environment and the store, so right
      // now we just do a regular visit.
      assert ((Ex->getType()->isAggregateType() || 
              Ex->getType()->isUnionType()) &&
              "Other kinds of expressions with non-aggregate/union types do"
              " not have lvalues.");
      
      Visit(Ex, Pred, Dst);
  }
}

//===----------------------------------------------------------------------===//
// Block entrance.  (Update counters).
//===----------------------------------------------------------------------===//

bool GRExprEngine::ProcessBlockEntrance(CFGBlock* B, const GRState*,
                                        GRBlockCounter BC) {
  
  return BC.getNumVisited(B->getBlockID()) < 3;
}

//===----------------------------------------------------------------------===//
// Branch processing.
//===----------------------------------------------------------------------===//

const GRState* GRExprEngine::MarkBranch(const GRState* St,
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
        
      return BindBlkExpr(St, B, UndefinedVal(Ex));
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
      
      return BindBlkExpr(St, C, UndefinedVal(Ex));
    }
      
    case Stmt::ChooseExprClass: { // ?:
      
      ChooseExpr* C = cast<ChooseExpr>(Terminator);
      
      Expr* Ex = branchTaken ? C->getLHS() : C->getRHS();      
      return BindBlkExpr(St, C, UndefinedVal(Ex));
    }
  }
}

void GRExprEngine::ProcessBranch(Expr* Condition, Stmt* Term,
                                 BranchNodeBuilder& builder) {

  // Remove old bindings for subexpressions.
  const GRState* PrevState =
    StateMgr.RemoveSubExprBindings(builder.getState());
  
  // Check for NULL conditions; e.g. "for(;;)"
  if (!Condition) { 
    builder.markInfeasible(false);
    return;
  }
  
  SVal V = GetSVal(PrevState, Condition);
  
  switch (V.getBaseKind()) {
    default:
      break;

    case SVal::UnknownKind:
      builder.generateNode(MarkBranch(PrevState, Term, true), true);
      builder.generateNode(MarkBranch(PrevState, Term, false), false);
      return;
      
    case SVal::UndefinedKind: {      
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
  const GRState* St = Assume(PrevState, V, true, isFeasible);

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

  const GRState* St = builder.getState();  
  SVal V = GetSVal(St, builder.getTarget());
  
  // Three possibilities:
  //
  //   (1) We know the computed label.
  //   (2) The label is NULL (or some other constant), or Undefined.
  //   (3) We have no clue about the label.  Dispatch to all targets.
  //
  
  typedef IndirectGotoNodeBuilder::iterator iterator;

  if (isa<loc::GotoLabel>(V)) {
    LabelStmt* L = cast<loc::GotoLabel>(V).getLabel();
    
    for (iterator I=builder.begin(), E=builder.end(); I != E; ++I) {
      if (I.getLabel() == L) {
        builder.generateNode(I, St);
        return;
      }
    }
    
    assert (false && "No block with label.");
    return;
  }

  if (isa<loc::ConcreteInt>(V) || isa<UndefinedVal>(V)) {
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
  
  const GRState* St = GetState(Pred);
  SVal X = GetBlkExprSVal(St, Ex);
  
  assert (X.isUndef());
  
  Expr* SE = (Expr*) cast<UndefinedVal>(X).getData();
  
  assert (SE);
  
  X = GetBlkExprSVal(St, SE);
  
  // Make sure that we invalidate the previous binding.
  MakeNode(Dst, Ex, Pred, StateMgr.BindExpr(St, Ex, X, true, true));
}

/// ProcessSwitch - Called by GRCoreEngine.  Used to generate successor
///  nodes by processing the 'effects' of a switch statement.
void GRExprEngine::ProcessSwitch(SwitchNodeBuilder& builder) {
  
  typedef SwitchNodeBuilder::iterator iterator;
  
  const GRState* St = builder.getState();  
  Expr* CondE = builder.getCondition();
  SVal  CondV = GetSVal(St, CondE);

  if (CondV.isUndef()) {
    NodeTy* N = builder.generateDefaultCaseNode(St, true);
    UndefBranches.insert(N);
    return;
  }
  
  const GRState*  DefaultSt = St;
  
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
      nonloc::ConcreteInt CaseVal(getBasicVals().getValue(V1));
      
      SVal Res = EvalBinOp(BinaryOperator::EQ, CondV, CaseVal);
      
      // Now "assume" that the case matches.
      
      bool isFeasible = false;      
      const GRState* StNew = Assume(St, Res, true, isFeasible);
      
      if (isFeasible) {
        builder.generateCaseStmtNode(I, StNew);
       
        // If CondV evaluates to a constant, then we know that this
        // is the *only* case that we can take, so stop evaluating the
        // others.
        if (isa<nonloc::ConcreteInt>(CondV))
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
  
  const GRState* St = GetState(Pred);
  SVal X = GetBlkExprSVal(St, B);
  
  assert (X.isUndef());
  
  Expr* Ex = (Expr*) cast<UndefinedVal>(X).getData();
  
  assert (Ex);
  
  if (Ex == B->getRHS()) {
    
    X = GetBlkExprSVal(St, Ex);
    
    // Handle undefined values.
    
    if (X.isUndef()) {
      MakeNode(Dst, B, Pred, BindBlkExpr(St, B, X));
      return;
    }
    
    // We took the RHS.  Because the value of the '&&' or '||' expression must
    // evaluate to 0 or 1, we must assume the value of the RHS evaluates to 0
    // or 1.  Alternatively, we could take a lazy approach, and calculate this
    // value later when necessary.  We don't have the machinery in place for
    // this right now, and since most logical expressions are used for branches,
    // the payoff is not likely to be large.  Instead, we do eager evaluation.
        
    bool isFeasible = false;
    const GRState* NewState = Assume(St, X, true, isFeasible);
    
    if (isFeasible)
      MakeNode(Dst, B, Pred,
               BindBlkExpr(NewState, B, MakeConstantVal(1U, B)));
      
    isFeasible = false;
    NewState = Assume(St, X, false, isFeasible);
    
    if (isFeasible)
      MakeNode(Dst, B, Pred,
               BindBlkExpr(NewState, B, MakeConstantVal(0U, B)));
  }
  else {
    // We took the LHS expression.  Depending on whether we are '&&' or
    // '||' we know what the value of the expression is via properties of
    // the short-circuiting.
    
    X = MakeConstantVal( B->getOpcode() == BinaryOperator::LAnd ? 0U : 1U, B);
    MakeNode(Dst, B, Pred, BindBlkExpr(St, B, X));
  }
}
 
//===----------------------------------------------------------------------===//
// Transfer functions: Loads and stores.
//===----------------------------------------------------------------------===//

void GRExprEngine::VisitDeclRefExpr(DeclRefExpr* Ex, NodeTy* Pred, NodeSet& Dst,
                                    bool asLValue) {
  
  const GRState* St = GetState(Pred);

  const NamedDecl* D = Ex->getDecl();

  if (const VarDecl* VD = dyn_cast<VarDecl>(D)) {

    SVal V = StateMgr.GetLValue(St, VD);

    if (asLValue)
      MakeNode(Dst, Ex, Pred, BindExpr(St, Ex, V));
    else
      EvalLoad(Dst, Ex, Pred, St, V);
    return;

  } else if (const EnumConstantDecl* ED = dyn_cast<EnumConstantDecl>(D)) {
    assert(!asLValue && "EnumConstantDecl does not have lvalue.");

    BasicValueFactory& BasicVals = StateMgr.getBasicVals();
    SVal V = nonloc::ConcreteInt(BasicVals.getValue(ED->getInitVal()));
    MakeNode(Dst, Ex, Pred, BindExpr(St, Ex, V));
    return;

  } else if (const FunctionDecl* FD = dyn_cast<FunctionDecl>(D)) {
    // FIXME: Does this need to be revised?  We were getting cases in
    //  real code that did this.

    // FIXME: This is not a valid assertion.  Produce a test case that
    // refutes it.
    // assert(asLValue); // Can we assume this?

    SVal V = loc::FuncVal(FD);
    MakeNode(Dst, Ex, Pred, BindExpr(St, Ex, V));
    return;
  }
  
  assert (false &&
          "ValueDecl support for this ValueDecl not implemented.");
}

/// VisitArraySubscriptExpr - Transfer function for array accesses
void GRExprEngine::VisitArraySubscriptExpr(ArraySubscriptExpr* A, NodeTy* Pred,
                                           NodeSet& Dst, bool asLValue) {
  
  Expr* Base = A->getBase()->IgnoreParens();
  Expr* Idx  = A->getIdx()->IgnoreParens();
  NodeSet Tmp;
  Visit(Base, Pred, Tmp);   // Get Base's rvalue, which should be an LocVal.
  
  for (NodeSet::iterator I1=Tmp.begin(), E1=Tmp.end(); I1!=E1; ++I1) {    
    NodeSet Tmp2;
    Visit(Idx, *I1, Tmp2);     // Evaluate the index.
      
    for (NodeSet::iterator I2=Tmp2.begin(), E2=Tmp2.end(); I2!=E2; ++I2) {
      const GRState* St = GetState(*I2);
      SVal V = StateMgr.GetLValue(St, GetSVal(St, Base), GetSVal(St, Idx));

      if (asLValue)
        MakeNode(Dst, A, *I2, BindExpr(St, A, V));
      else
        EvalLoad(Dst, A, *I2, St, V);
    }
  }
}

/// VisitMemberExpr - Transfer function for member expressions.
void GRExprEngine::VisitMemberExpr(MemberExpr* M, NodeTy* Pred,
                                   NodeSet& Dst, bool asLValue) {
  
  Expr* Base = M->getBase()->IgnoreParens();
  NodeSet Tmp;
  
  if (M->isArrow()) 
    Visit(Base, Pred, Tmp);        // p->f = ...  or   ... = p->f
  else
    VisitLValue(Base, Pred, Tmp);  // x.f = ...   or   ... = x.f
    
  for (NodeSet::iterator I = Tmp.begin(), E = Tmp.end(); I != E; ++I) {
    const GRState* St = GetState(*I);
    // FIXME: Should we insert some assumption logic in here to determine
    // if "Base" is a valid piece of memory?  Before we put this assumption
    // later when using FieldOffset lvals (which we no longer have).    
    SVal L = StateMgr.GetLValue(St, GetSVal(St, Base), M->getMemberDecl());

    if (asLValue)
      MakeNode(Dst, M, *I, BindExpr(St, M, L));
    else
      EvalLoad(Dst, M, *I, St, L);
  }
}

void GRExprEngine::EvalStore(NodeSet& Dst, Expr* Ex, NodeTy* Pred,
                             const GRState* St, SVal location, SVal Val) {
  
  assert (Builder && "GRStmtNodeBuilder must be defined.");
  
  // Evaluate the location (checks for bad dereferences).
  St = EvalLocation(Ex, Pred, St, location);
  
  if (!St)
    return;
  
  // Proceed with the store.
  
  unsigned size = Dst.size();  

  SaveAndRestore<bool> OldSink(Builder->BuildSinks);
  SaveAndRestore<ProgramPoint::Kind> OldSPointKind(Builder->PointKind);
  SaveOr OldHasGen(Builder->HasGeneratedNode);

  assert (!location.isUndef());
  Builder->PointKind = ProgramPoint::PostStoreKind;
  
  getTF().EvalStore(Dst, *this, *Builder, Ex, Pred, St, location, Val);
  
  // Handle the case where no nodes where generated.  Auto-generate that
  // contains the updated state if we aren't generating sinks.
  
  if (!Builder->BuildSinks && Dst.size() == size && !Builder->HasGeneratedNode)
    getTF().GRTransferFuncs::EvalStore(Dst, *this, *Builder, Ex, Pred, St,
                                   location, Val);
}

void GRExprEngine::EvalLoad(NodeSet& Dst, Expr* Ex, NodeTy* Pred,
                            const GRState* St, SVal location,
                            bool CheckOnly) {

  // Evaluate the location (checks for bad dereferences).
  
  St = EvalLocation(Ex, Pred, St, location, true);
  
  if (!St)
    return;
  
  // Proceed with the load.
  ProgramPoint::Kind K = ProgramPoint::PostLoadKind;

  // FIXME: Currently symbolic analysis "generates" new symbols
  //  for the contents of values.  We need a better approach.

  // FIXME: The "CheckOnly" option exists only because Array and Field
  //  loads aren't fully implemented.  Eventually this option will go away.

  if (CheckOnly)
    MakeNode(Dst, Ex, Pred, St, K);
  else if (location.isUnknown()) {
    // This is important.  We must nuke the old binding.
    MakeNode(Dst, Ex, Pred, BindExpr(St, Ex, UnknownVal()), K);
  }
  else    
    MakeNode(Dst, Ex, Pred, BindExpr(St, Ex, GetSVal(St, cast<Loc>(location),
                                                     Ex->getType())), K);  
}

void GRExprEngine::EvalStore(NodeSet& Dst, Expr* Ex, Expr* StoreE, NodeTy* Pred,
                             const GRState* St, SVal location, SVal Val) {
 
  NodeSet TmpDst;
  EvalStore(TmpDst, StoreE, Pred, St, location, Val);

  for (NodeSet::iterator I=TmpDst.begin(), E=TmpDst.end(); I!=E; ++I)
    MakeNode(Dst, Ex, *I, (*I)->getState());
}

const GRState* GRExprEngine::EvalLocation(Expr* Ex, NodeTy* Pred,
                                          const GRState* St,
                                          SVal location, bool isLoad) {
  
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
  
  Loc LV = cast<Loc>(location);    
  
  // "Assume" that the pointer is not NULL.
  
  bool isFeasibleNotNull = false;
  const GRState* StNotNull = Assume(St, LV, true, isFeasibleNotNull);
  
  // "Assume" that the pointer is NULL.
  
  bool isFeasibleNull = false;
  GRStateRef StNull = GRStateRef(Assume(St, LV, false, isFeasibleNull),
                                 getStateManager());
  
  if (isFeasibleNull) {
    
    // Use the Generic Data Map to mark in the state what lval was null.
    const SVal* PersistentLV = getBasicVals().getPersistentSVal(LV);
    StNull = StNull.set<GRState::NullDerefTag>(PersistentLV);
    
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

  // Check for out-of-bound array access.
  if (isFeasibleNotNull && isa<loc::MemRegionVal>(LV)) {
    const MemRegion* R = cast<loc::MemRegionVal>(LV).getRegion();
    if (const ElementRegion* ER = dyn_cast<ElementRegion>(R)) {
      // Get the index of the accessed element.
      SVal Idx = ER->getIndex();
      // Get the extent of the array.
      SVal NumElements = StateMgr.getStoreManager().getSizeInElements(StNotNull,
                                                       ER->getSuperRegion());

      bool isFeasibleInBound = false;
      const GRState* StInBound = AssumeInBound(StNotNull, Idx, NumElements, 
                                               true, isFeasibleInBound);

      bool isFeasibleOutBound = false;
      const GRState* StOutBound = AssumeInBound(StNotNull, Idx, NumElements, 
                                                false, isFeasibleOutBound);
      StInBound = StOutBound = 0; // FIXME: squeltch warning.

      // Report warnings ...
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
                             NodeSet& Dst)
{
  // Determine the type of function we're calling (if available).
  const FunctionTypeProto *Proto = NULL;
  QualType FnType = CE->getCallee()->IgnoreParens()->getType();
  if (const PointerType *FnTypePtr = FnType->getAsPointerType())
    Proto = FnTypePtr->getPointeeType()->getAsFunctionTypeProto();

  VisitCallRec(CE, Pred, AI, AE, Dst, Proto, /*ParamIdx=*/0);
}

void GRExprEngine::VisitCallRec(CallExpr* CE, NodeTy* Pred,
                                CallExpr::arg_iterator AI,
                                CallExpr::arg_iterator AE,
                                NodeSet& Dst, const FunctionTypeProto *Proto,
                                unsigned ParamIdx) {
  
  // Process the arguments.
  if (AI != AE) {
    // If the call argument is being bound to a reference parameter,
    // visit it as an lvalue, not an rvalue.
    bool VisitAsLvalue = false;
    if (Proto && ParamIdx < Proto->getNumArgs())
      VisitAsLvalue = Proto->getArgType(ParamIdx)->isReferenceType();

    NodeSet DstTmp;  
    if (VisitAsLvalue)
      VisitLValue(*AI, Pred, DstTmp);    
    else
      Visit(*AI, Pred, DstTmp);    
    ++AI;
    
    for (NodeSet::iterator DI=DstTmp.begin(), DE=DstTmp.end(); DI != DE; ++DI)
      VisitCallRec(CE, *DI, AI, AE, Dst, Proto, ParamIdx + 1);
    
    return;
  }

  // If we reach here we have processed all of the arguments.  Evaluate
  // the callee expression.
  
  NodeSet DstTmp;    
  Expr* Callee = CE->getCallee()->IgnoreParens();

  Visit(Callee, Pred, DstTmp);
  
  // Finally, evaluate the function call.
  for (NodeSet::iterator DI = DstTmp.begin(), DE = DstTmp.end(); DI!=DE; ++DI) {

    const GRState* St = GetState(*DI);
    SVal L = GetSVal(St, Callee);

    // FIXME: Add support for symbolic function calls (calls involving
    //  function pointer values that are symbolic).
    
    // Check for undefined control-flow or calls to NULL.
    
    if (L.isUndef() || isa<loc::ConcreteInt>(L)) {      
      NodeTy* N = Builder->generateNode(CE, St, *DI);
      
      if (N) {
        N->markAsSink();
        BadCalls.insert(N);
      }
      
      continue;
    }
    
    // Check for the "noreturn" attribute.
    
    SaveAndRestore<bool> OldSink(Builder->BuildSinks);
    
    if (isa<loc::FuncVal>(L)) {      
      
      FunctionDecl* FD = cast<loc::FuncVal>(L).getDecl();
      
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
            else if (!memcmp(s, "error", 5)) {
              if (CE->getNumArgs() > 0) {
                SVal X = GetSVal(St, *CE->arg_begin());
                // FIXME: use Assume to inspect the possible symbolic value of
                // X. Also check the specific signature of error().
                nonloc::ConcreteInt* CI = dyn_cast<nonloc::ConcreteInt>(&X);
                if (CI && CI->getValue() != 0)
                  Builder->BuildSinks = true;
              }
            }
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
            
          case 13:
            if (!memcmp(s, "__assert_fail", 13)) Builder->BuildSinks = true;
            break;
            
          case 14:
            if (!memcmp(s, "dtrace_assfail", 14) ||
                !memcmp(s, "yy_fatal_error", 14))
              Builder->BuildSinks = true;
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

    if (isa<loc::FuncVal>(L)) {
      
      IdentifierInfo* Info = cast<loc::FuncVal>(L).getDecl()->getIdentifier();
      
      if (unsigned id = Info->getBuiltinID())
        switch (id) {
          case Builtin::BI__builtin_expect: {
            // For __builtin_expect, just return the value of the subexpression.
            assert (CE->arg_begin() != CE->arg_end());            
            SVal X = GetSVal(St, *(CE->arg_begin()));
            MakeNode(Dst, CE, *DI, BindExpr(St, CE, X));
            continue;            
          }
            
          case Builtin::BI__builtin_alloca: {
            // FIXME: Handle size.
            // FIXME: Refactor into StoreManager itself?
            MemRegionManager& RM = getStateManager().getRegionManager();
            const MemRegion* R =
              RM.getAllocaRegion(CE, Builder->getCurrentBlockCount());            
            MakeNode(Dst, CE, *DI, BindExpr(St, CE, loc::MemRegionVal(R)));
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

      if (GetSVal(GetState(*DI), *I).isUndef()) {        
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
// Transfer function: Objective-C ivar references.
//===----------------------------------------------------------------------===//

void GRExprEngine::VisitObjCIvarRefExpr(ObjCIvarRefExpr* Ex,
                                            NodeTy* Pred, NodeSet& Dst,
                                            bool asLValue) {
  
  Expr* Base = cast<Expr>(Ex->getBase());
  NodeSet Tmp;
  Visit(Base, Pred, Tmp);
  
  for (NodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {
    const GRState* St = GetState(*I);
    SVal BaseVal = GetSVal(St, Base);
    SVal location = StateMgr.GetLValue(St, Ex->getDecl(), BaseVal);
    
    if (asLValue)
      MakeNode(Dst, Ex, *I, BindExpr(St, Ex, location));
    else
      EvalLoad(Dst, Ex, *I, St, location);
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
  
  const GRState* St = GetState(Pred);
  bool RaisesException = false;
  
  
  if (Expr* Receiver = ME->getReceiver()) {
    
    SVal L = GetSVal(St, Receiver);
    
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
    
    if (GetSVal(St, *I).isUndef()) {
      
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
  QualType ExTy = Ex->getType();

  if (const ExplicitCastExpr *ExCast=dyn_cast_or_null<ExplicitCastExpr>(CastE))
    T = ExCast->getTypeAsWritten();

  if (ExTy->isArrayType() || ExTy->isFunctionType() || T->isReferenceType())
    VisitLValue(Ex, Pred, S1);
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
  
  for (NodeSet::iterator I1 = S1.begin(), E1 = S1.end(); I1 != E1; ++I1) {
    NodeTy* N = *I1;
    const GRState* St = GetState(N);
    SVal V = GetSVal(St, Ex);

    // Unknown?
    
    if (V.isUnknown()) {
      Dst.Add(N);
      continue;
    }
    
    // Undefined?
    
    if (V.isUndef()) {
      MakeNode(Dst, CastE, N, BindExpr(St, CastE, V));
      continue;
    }
    
    // For const casts, just propagate the value.
    ASTContext& C = getContext();
    
    if (C.getCanonicalType(T).getUnqualifiedType() == 
        C.getCanonicalType(ExTy).getUnqualifiedType()) {
      MakeNode(Dst, CastE, N, BindExpr(St, CastE, V));
      continue;
    }
  
    // Check for casts from pointers to integers.
    if (T->isIntegerType() && Loc::IsLocType(ExTy)) {
      unsigned bits = getContext().getTypeSize(ExTy);
    
      // FIXME: Determine if the number of bits of the target type is 
      // equal or exceeds the number of bits to store the pointer value.
      // If not, flag an error.
      
      V = nonloc::LocAsInteger::Make(getBasicVals(), cast<Loc>(V), bits);
      MakeNode(Dst, CastE, N, BindExpr(St, CastE, V));
      continue;
    }
    
    // Check for casts from integers to pointers.
    if (Loc::IsLocType(T) && ExTy->isIntegerType())
      if (nonloc::LocAsInteger *LV = dyn_cast<nonloc::LocAsInteger>(&V)) {
        // Just unpackage the lval and return it.
        V = LV->getLoc();
        MakeNode(Dst, CastE, N, BindExpr(St, CastE, V));
        continue;
      }

    // StoreManager casts array to different values.
    if (ExTy->isArrayType()) {
      assert(T->isPointerType() || T->isReferenceType());

      V = StateMgr.ArrayToPointer(V);
      MakeNode(Dst, CastE, N, BindExpr(St, CastE, V));
      continue;
    }

    // All other cases.
    MakeNode(Dst, CastE, N, BindExpr(St, CastE, EvalCast(V, CastE->getType())));
  }
}

void GRExprEngine::VisitCompoundLiteralExpr(CompoundLiteralExpr* CL,
                                            NodeTy* Pred, NodeSet& Dst, 
                                            bool asLValue) {
  InitListExpr* ILE = cast<InitListExpr>(CL->getInitializer()->IgnoreParens());
  NodeSet Tmp;
  Visit(ILE, Pred, Tmp);
  
  for (NodeSet::iterator I = Tmp.begin(), EI = Tmp.end(); I!=EI; ++I) {
    const GRState* St = GetState(*I);
    SVal ILV = GetSVal(St, ILE);
    St = StateMgr.BindCompoundLiteral(St, CL, ILV);

    if (asLValue)
      MakeNode(Dst, CL, *I, BindExpr(St, CL, StateMgr.GetLValue(St, CL)));
    else
      MakeNode(Dst, CL, *I, BindExpr(St, CL, ILV));
  }
}

void GRExprEngine::VisitDeclStmt(DeclStmt* DS, NodeTy* Pred, NodeSet& Dst) {  

  // The CFG has one DeclStmt per Decl.  
  ScopedDecl* D = *DS->decl_begin();
  
  if (!D || !isa<VarDecl>(D))
    return;
  
  const VarDecl* VD = dyn_cast<VarDecl>(D);
  
  Expr* Ex = const_cast<Expr*>(VD->getInit());

  // FIXME: static variables may have an initializer, but the second
  //  time a function is called those values may not be current.
  NodeSet Tmp;

  if (Ex)
    Visit(Ex, Pred, Tmp);

  if (Tmp.empty())
    Tmp.Add(Pred);
  
  for (NodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {
    const GRState* St = GetState(*I);
    St = StateMgr.BindDecl(St, VD, Ex, Builder->getCurrentBlockCount());
    MakeNode(Dst, DS, *I, St);
  }
}

namespace {
  // This class is used by VisitInitListExpr as an item in a worklist
  // for processing the values contained in an InitListExpr.
class VISIBILITY_HIDDEN InitListWLItem {
public:
  llvm::ImmutableList<SVal> Vals;
  GRExprEngine::NodeTy* N;
  InitListExpr::reverse_iterator Itr;
  
  InitListWLItem(GRExprEngine::NodeTy* n, llvm::ImmutableList<SVal> vals,
         InitListExpr::reverse_iterator itr)
  : Vals(vals), N(n), Itr(itr) {}
};
}


void GRExprEngine::VisitInitListExpr(InitListExpr* E, NodeTy* Pred, 
                                     NodeSet& Dst) {

  const GRState* state = GetState(Pred);
  QualType T = E->getType();
  unsigned NumInitElements = E->getNumInits();  

  if (T->isArrayType() || T->isStructureType()) {

    llvm::ImmutableList<SVal> StartVals = getBasicVals().getEmptySValList();
    
    // Handle base case where the initializer has no elements.
    // e.g: static int* myArray[] = {};
    if (NumInitElements == 0) {
      SVal V = NonLoc::MakeCompoundVal(T, StartVals, getBasicVals());
      MakeNode(Dst, E, Pred, BindExpr(state, E, V));
      return;
    }      
    
    // Create a worklist to process the initializers.
    llvm::SmallVector<InitListWLItem, 10> WorkList;
    WorkList.reserve(NumInitElements);  
    WorkList.push_back(InitListWLItem(Pred, StartVals, E->rbegin()));    
    InitListExpr::reverse_iterator ItrEnd = E->rend();
    
    // Process the worklist until it is empty.
    while (!WorkList.empty()) {
      InitListWLItem X = WorkList.back();
      WorkList.pop_back();
      
      NodeSet Tmp;
      Visit(*X.Itr, X.N, Tmp);
      
      InitListExpr::reverse_iterator NewItr = X.Itr + 1;

      for (NodeSet::iterator NI=Tmp.begin(), NE=Tmp.end(); NI!=NE; ++NI) {
        // Get the last initializer value.
        state = GetState(*NI);
        SVal InitV = GetSVal(state, cast<Expr>(*X.Itr));
        
        // Construct the new list of values by prepending the new value to
        // the already constructed list.
        llvm::ImmutableList<SVal> NewVals =
          getBasicVals().consVals(InitV, X.Vals);
        
        if (NewItr == ItrEnd) {
          // Now we have a list holding all init values. Make CompoundValData.
          SVal V = NonLoc::MakeCompoundVal(T, NewVals, getBasicVals());

          // Make final state and node.
          MakeNode(Dst, E, *NI, BindExpr(state, E, V));
        }
        else {
          // Still some initializer values to go.  Push them onto the worklist.
          WorkList.push_back(InitListWLItem(*NI, NewVals, NewItr));
        }
      }
    }
    
    return;
  }

  if (Loc::IsLocType(T) || T->isIntegerType()) {
    assert (E->getNumInits() == 1);
    NodeSet Tmp;
    Expr* Init = E->getInit(0);
    Visit(Init, Pred, Tmp);
    for (NodeSet::iterator I = Tmp.begin(), EI = Tmp.end(); I != EI; ++I) {
      state = GetState(*I);
      MakeNode(Dst, E, *I, BindExpr(state, E, GetSVal(state, Init)));
    }
    return;
  }

  if (T->isUnionType()) {
    // FIXME: to be implemented.
    MakeNode(Dst, E, Pred, state);
    return;
  }

  printf("InitListExpr type = %s\n", T.getAsString().c_str());
  assert(0 && "unprocessed InitListExpr type");
}

/// VisitSizeOfAlignOfExpr - Transfer function for sizeof(type).
void GRExprEngine::VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr* Ex,
                                          NodeTy* Pred,
                                          NodeSet& Dst) {
  QualType T = Ex->getTypeOfArgument();
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
           BindExpr(GetState(Pred), Ex,
                    NonLoc::MakeVal(getBasicVals(), amt, Ex->getType())));  
}


void GRExprEngine::VisitUnaryOperator(UnaryOperator* U, NodeTy* Pred,
                                      NodeSet& Dst, bool asLValue) {

  switch (U->getOpcode()) {
      
    default:
      break;
          
    case UnaryOperator::Deref: {
      
      Expr* Ex = U->getSubExpr()->IgnoreParens();
      NodeSet Tmp;
      Visit(Ex, Pred, Tmp);
      
      for (NodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {
        
        const GRState* St = GetState(*I);
        SVal location = GetSVal(St, Ex);
        
        if (asLValue)
          MakeNode(Dst, U, *I, BindExpr(St, U, location));
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
        
        // FIXME: We don't have complex SValues yet.
        if (Ex->getType()->isAnyComplexType()) {
          // Just report "Unknown."
          Dst.Add(*I);
          continue;
        }
        
        // For all other types, UnaryOperator::Real is an identity operation.
        assert (U->getType() == Ex->getType());
        const GRState* St = GetState(*I);
        MakeNode(Dst, U, *I, BindExpr(St, U, GetSVal(St, Ex)));
      } 
      
      return;
    }
      
    case UnaryOperator::Imag: {
      
      Expr* Ex = U->getSubExpr()->IgnoreParens();
      NodeSet Tmp;
      Visit(Ex, Pred, Tmp);
      
      for (NodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {
        // FIXME: We don't have complex SValues yet.
        if (Ex->getType()->isAnyComplexType()) {
          // Just report "Unknown."
          Dst.Add(*I);
          continue;
        }
        
        // For all other types, UnaryOperator::Float returns 0.
        assert (Ex->getType()->isIntegerType());
        const GRState* St = GetState(*I);
        SVal X = NonLoc::MakeVal(getBasicVals(), 0, Ex->getType());
        MakeNode(Dst, U, *I, BindExpr(St, U, X));
      }
      
      return;
    }
      
      // FIXME: Just report "Unknown" for OffsetOf.      
    case UnaryOperator::OffsetOf:
      Dst.Add(Pred);
      return;
      
    case UnaryOperator::Plus: assert (!asLValue);  // FALL-THROUGH.
    case UnaryOperator::Extension: {
      
      // Unary "+" is a no-op, similar to a parentheses.  We still have places
      // where it may be a block-level expression, so we need to
      // generate an extra node that just propagates the value of the
      // subexpression.

      Expr* Ex = U->getSubExpr()->IgnoreParens();
      NodeSet Tmp;
      Visit(Ex, Pred, Tmp);
      
      for (NodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {        
        const GRState* St = GetState(*I);
        MakeNode(Dst, U, *I, BindExpr(St, U, GetSVal(St, Ex)));
      }
      
      return;
    }
    
    case UnaryOperator::AddrOf: {
      
      assert(!asLValue);
      Expr* Ex = U->getSubExpr()->IgnoreParens();
      NodeSet Tmp;
      VisitLValue(Ex, Pred, Tmp);
     
      for (NodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {        
        const GRState* St = GetState(*I);
        SVal V = GetSVal(St, Ex);
        St = BindExpr(St, U, V);
        MakeNode(Dst, U, *I, St);
      }

      return; 
    }
      
    case UnaryOperator::LNot:
    case UnaryOperator::Minus:
    case UnaryOperator::Not: {
      
      assert (!asLValue);
      Expr* Ex = U->getSubExpr()->IgnoreParens();
      NodeSet Tmp;
      Visit(Ex, Pred, Tmp);
      
      for (NodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {        
        const GRState* St = GetState(*I);
        
        // Get the value of the subexpression.
        SVal V = GetSVal(St, Ex);

        // Perform promotions.
        // FIXME: This is the right thing to do, but it currently breaks
        //  a bunch of tests.
        // V = EvalCast(V, U->getType()); 
        
        if (V.isUnknownOrUndef()) {
          MakeNode(Dst, U, *I, BindExpr(St, U, V));
          continue;
        }
        
        switch (U->getOpcode()) {
          default:
            assert(false && "Invalid Opcode.");
            break;
            
          case UnaryOperator::Not:
            // FIXME: Do we need to handle promotions?
            St = BindExpr(St, U, EvalComplement(cast<NonLoc>(V)));
            break;            
            
          case UnaryOperator::Minus:
            // FIXME: Do we need to handle promotions?
            St = BindExpr(St, U, EvalMinus(U, cast<NonLoc>(V)));
            break;   
            
          case UnaryOperator::LNot:   
            
            // C99 6.5.3.3: "The expression !E is equivalent to (0==E)."
            //
            //  Note: technically we do "E == 0", but this is the same in the
            //    transfer functions as "0 == E".
            
            if (isa<Loc>(V)) {
              loc::ConcreteInt X(getBasicVals().getZeroWithPtrWidth());
              SVal Result = EvalBinOp(BinaryOperator::EQ, cast<Loc>(V), X);
              St = BindExpr(St, U, Result);
            }
            else {
              nonloc::ConcreteInt X(getBasicVals().getValue(0, Ex->getType()));
#if 0            
              SVal Result = EvalBinOp(BinaryOperator::EQ, cast<NonLoc>(V), X);
              St = SetSVal(St, U, Result);
#else
              EvalBinOp(Dst, U, BinaryOperator::EQ, cast<NonLoc>(V), X, *I);
              continue;
#endif
            }
            
            break;
        }
        
        MakeNode(Dst, U, *I, St);
      }
      
      return;
    }
  }

  // Handle ++ and -- (both pre- and post-increment).

  assert (U->isIncrementDecrementOp());
  NodeSet Tmp;
  Expr* Ex = U->getSubExpr()->IgnoreParens();
  VisitLValue(Ex, Pred, Tmp);
  
  for (NodeSet::iterator I = Tmp.begin(), E = Tmp.end(); I!=E; ++I) {
    
    const GRState* St = GetState(*I);
    SVal V1 = GetSVal(St, Ex);
    
    // Perform a load.      
    NodeSet Tmp2;
    EvalLoad(Tmp2, Ex, *I, St, V1);

    for (NodeSet::iterator I2 = Tmp2.begin(), E2 = Tmp2.end(); I2!=E2; ++I2) {
        
      St = GetState(*I2);
      SVal V2 = GetSVal(St, Ex);
        
      // Propagate unknown and undefined values.      
      if (V2.isUnknownOrUndef()) {
        MakeNode(Dst, U, *I2, BindExpr(St, U, V2));
        continue;
      }
      
      // Handle all other values.
      
      BinaryOperator::Opcode Op = U->isIncrementOp() ? BinaryOperator::Add
                                                     : BinaryOperator::Sub;
      
      SVal Result = EvalBinOp(Op, V2, MakeConstantVal(1U, U));      
      St = BindExpr(St, U, U->isPostfix() ? V2 : Result);

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
  VisitLValue(*I, Pred, Tmp);
  
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
    // should evaluate to Locs.  Nuke all of their values.
    
    // FIXME: Some day in the future it would be nice to allow a "plug-in"
    // which interprets the inline asm and stores proper results in the
    // outputs.
    
    const GRState* St = GetState(Pred);
    
    for (AsmStmt::outputs_iterator OI = A->begin_outputs(),
                                   OE = A->end_outputs(); OI != OE; ++OI) {
      
      SVal X = GetSVal(St, *OI);      
      assert (!isa<NonLoc>(X));  // Should be an Lval, or unknown, undef.
      
      if (isa<Loc>(X))
        St = BindLoc(St, cast<Loc>(X), UnknownVal());
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
      SVal X = GetSVal((*I)->getState(), R);

      if (isa<loc::MemRegionVal>(X)) {
        
        // Determine if the value is on the stack.
        const MemRegion* R = cast<loc::MemRegionVal>(&X)->getRegion();

        if (R && getStateManager().hasStackStorage(R)) {
        
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

const GRState* GRExprEngine::CheckDivideZero(Expr* Ex, const GRState* St,
                                             NodeTy* Pred, SVal Denom) {
  
  // Divide by undefined? (potentially zero)
  
  if (Denom.isUndef()) {
    NodeTy* DivUndef = Builder->generateNode(Ex, St, Pred);
    
    if (DivUndef) {
      DivUndef->markAsSink();
      ExplicitBadDivides.insert(DivUndef);
    }
    
    return 0;
  }
  
  // Check for divide/remainder-by-zero.
  // First, "assume" that the denominator is 0 or undefined.            
  
  bool isFeasibleZero = false;
  const GRState* ZeroSt =  Assume(St, Denom, false, isFeasibleZero);
  
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
  
  return isFeasibleNotZero ? St : 0;
}

void GRExprEngine::VisitBinaryOperator(BinaryOperator* B,
                                       GRExprEngine::NodeTy* Pred,
                                       GRExprEngine::NodeSet& Dst) {

  NodeSet Tmp1;
  Expr* LHS = B->getLHS()->IgnoreParens();
  Expr* RHS = B->getRHS()->IgnoreParens();
  
  if (B->isAssignmentOp())
    VisitLValue(LHS, Pred, Tmp1);
  else
    Visit(LHS, Pred, Tmp1);

  for (NodeSet::iterator I1=Tmp1.begin(), E1=Tmp1.end(); I1 != E1; ++I1) {

    SVal LeftV = GetSVal((*I1)->getState(), LHS);
    
    // Process the RHS.
    
    NodeSet Tmp2;
    Visit(RHS, *I1, Tmp2);
    
    // With both the LHS and RHS evaluated, process the operation itself.
    
    for (NodeSet::iterator I2=Tmp2.begin(), E2=Tmp2.end(); I2 != E2; ++I2) {

      const GRState* St = GetState(*I2);
      const GRState* OldSt = St;

      SVal RightV = GetSVal(St, RHS);
      BinaryOperator::Opcode Op = B->getOpcode();
      
      switch (Op) {
          
        case BinaryOperator::Assign: {
          
          // EXPERIMENTAL: "Conjured" symbols.
          // FIXME: Handle structs.
          QualType T = RHS->getType();
          
          if (RightV.isUnknown() && (T->isIntegerType() || Loc::IsLocType(T))) {
            unsigned Count = Builder->getCurrentBlockCount();
            SymbolID Sym = SymMgr.getConjuredSymbol(B->getRHS(), Count);
            
            RightV = Loc::IsLocType(B->getRHS()->getType()) 
                   ? cast<SVal>(loc::SymbolVal(Sym)) 
                   : cast<SVal>(nonloc::SymbolVal(Sym));            
          }
          
          // Simulate the effects of a "store":  bind the value of the RHS
          // to the L-Value represented by the LHS.
          
          EvalStore(Dst, B, LHS, *I2, BindExpr(St, B, RightV), LeftV, RightV);
          continue;
        }
          
        case BinaryOperator::Div:
        case BinaryOperator::Rem:
          
          // Special checking for integer denominators.          
          if (RHS->getType()->isIntegerType()) {
            St = CheckDivideZero(B, St, *I2, RightV);
            if (!St) continue;
          }
          
          // FALL-THROUGH.

        default: {
      
          if (B->isAssignmentOp())
            break;
          
          // Process non-assignements except commas or short-circuited
          // logical expressions (LAnd and LOr).
          
          SVal Result = EvalBinOp(Op, LeftV, RightV);
          
          if (Result.isUnknown()) {
            if (OldSt != St) {
              // Generate a new node if we have already created a new state.
              MakeNode(Dst, B, *I2, St);
            }
            else
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
          
          MakeNode(Dst, B, *I2, BindExpr(St, B, Result));
          continue;
        }
      }
    
      assert (B->isCompoundAssignmentOp());

      if (Op >= BinaryOperator::AndAssign) {
        Op = (BinaryOperator::Opcode) (Op - (BinaryOperator::AndAssign - 
                                             BinaryOperator::And));
      }
      else {
        Op = (BinaryOperator::Opcode) (Op - BinaryOperator::MulAssign);
      }
          
      // Perform a load (the LHS).  This performs the checks for
      // null dereferences, and so on.
      NodeSet Tmp3;
      SVal location = GetSVal(St, LHS);
      EvalLoad(Tmp3, LHS, *I2, St, location);
      
      for (NodeSet::iterator I3=Tmp3.begin(), E3=Tmp3.end(); I3!=E3; ++I3) {
        
        St = GetState(*I3);
        SVal V = GetSVal(St, LHS);

        // Check for divide-by-zero.
        if ((Op == BinaryOperator::Div || Op == BinaryOperator::Rem)
            && RHS->getType()->isIntegerType()) {
          
          // CheckDivideZero returns a new state where the denominator
          // is assumed to be non-zero.
          St = CheckDivideZero(B, St, *I3, RightV);
          
          if (!St)
            continue;
        }
        
        // Propagate undefined values (left-side).          
        if (V.isUndef()) {
          EvalStore(Dst, B, LHS, *I3, BindExpr(St, B, V), location, V);
          continue;
        }
        
        // Propagate unknown values (left and right-side).
        if (RightV.isUnknown() || V.isUnknown()) {
          EvalStore(Dst, B, LHS, *I3, BindExpr(St, B, UnknownVal()), location,
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
        if (RightV.isUndef()) {            
          // Propagate undefined values (right-side).          
          EvalStore(Dst,B, LHS, *I3, BindExpr(St, B, RightV), location, RightV);
          continue;
        }
      
        // Compute the result of the operation.
      
        SVal Result = EvalCast(EvalBinOp(Op, V, RightV), B->getType());
          
        if (Result.isUndef()) {
            
          // The operands were not undefined, but the result is undefined.
          
          if (NodeTy* UndefNode = Builder->generateNode(B, St, *I3)) {
            UndefNode->markAsSink();            
            UndefResults.insert(UndefNode);
          }
          
          continue;
        }

        // EXPERIMENTAL: "Conjured" symbols.
        // FIXME: Handle structs.
        if (Result.isUnknown() &&
            (CTy->isIntegerType() || Loc::IsLocType(CTy))) {
          
          unsigned Count = Builder->getCurrentBlockCount();
          SymbolID Sym = SymMgr.getConjuredSymbol(B->getRHS(), Count);
          
          Result = Loc::IsLocType(CTy) 
                 ? cast<SVal>(loc::SymbolVal(Sym)) 
                 : cast<SVal>(nonloc::SymbolVal(Sym));            
        }
 
        EvalStore(Dst, B, LHS, *I3, BindExpr(St, B, Result), location, Result);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Transfer-function Helpers.
//===----------------------------------------------------------------------===//

void GRExprEngine::EvalBinOp(ExplodedNodeSet<GRState>& Dst, Expr* Ex,
                             BinaryOperator::Opcode Op,
                             NonLoc L, NonLoc R,
                             ExplodedNode<GRState>* Pred) {

  GRStateSet OStates;
  EvalBinOp(OStates, GetState(Pred), Ex, Op, L, R);

  for (GRStateSet::iterator I=OStates.begin(), E=OStates.end(); I!=E; ++I)
    MakeNode(Dst, Ex, Pred, *I);
}

void GRExprEngine::EvalBinOp(GRStateSet& OStates, const GRState* St,
                             Expr* Ex, BinaryOperator::Opcode Op,
                             NonLoc L, NonLoc R) {
  
  GRStateSet::AutoPopulate AP(OStates, St);
  if (R.isValid()) getTF().EvalBinOpNN(OStates, StateMgr, St, Ex, Op, L, R);
}

//===----------------------------------------------------------------------===//
// Visualization.
//===----------------------------------------------------------------------===//

#ifndef NDEBUG
static GRExprEngine* GraphPrintCheckerState;
static SourceManager* GraphPrintSourceManager;

namespace llvm {
template<>
struct VISIBILITY_HIDDEN DOTGraphTraits<GRExprEngine::NodeTy*> :
  public DefaultDOTGraphTraits {
    
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
        llvm::raw_os_ostream OutS(Out);
        S->printPretty(OutS);
        OutS.flush();
        
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
          Out << "\\|Store to Undefined Loc.";
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
          
          llvm::raw_os_ostream OutS(Out);
          E.getSrc()->printTerminator(OutS);
          OutS.flush();
          
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
                llvm::raw_os_ostream OutS(Out);
                C->getLHS()->printPretty(OutS);
                OutS.flush();
              
                if (Stmt* RHS = C->getRHS()) {
                  Out << " .. ";
                  RHS->printPretty(OutS);
                  OutS.flush();
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

    GRStateRef state(N->getState(), GraphPrintCheckerState->getStateManager());
    state.printDOT(Out);
      
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
  
  llvm::SmallSet<ProgramPoint,10> CachedSources;
  
  for ( ; I != E; ++I ) {
    GRExprEngine::NodeTy* N = GetGraphNode(I);
    ProgramPoint P = N->getLocation();
    
    if (CachedSources.count(P))
      continue;
    
    CachedSources.insert(P);    
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

    llvm::ViewGraph(*G.roots_begin(), "GRExprEngine");
    
    GraphPrintCheckerState = NULL;
    GraphPrintSourceManager = NULL;
  }
#endif
}

void GRExprEngine::ViewGraph(NodeTy** Beg, NodeTy** End) {
#ifndef NDEBUG
  GraphPrintCheckerState = this;
  GraphPrintSourceManager = &getContext().getSourceManager();
    
  GRExprEngine::GraphTy* TrimmedG = G.Trim(Beg, End);

  if (!TrimmedG)
    llvm::cerr << "warning: Trimmed ExplodedGraph is empty.\n";
  else {
    llvm::ViewGraph(*TrimmedG->roots_begin(), "TrimmedGRExprEngine");    
    delete TrimmedG;
  }  
  
  GraphPrintCheckerState = NULL;
  GraphPrintSourceManager = NULL;
#endif
}
