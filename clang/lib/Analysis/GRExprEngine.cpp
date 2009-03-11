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
#include "clang/Analysis/PathSensitive/GRExprEngineBuilders.h"

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
                           LiveVariables& L, BugReporterData& BRD,
                           bool purgeDead, bool eagerlyAssume,
                           StoreManagerCreator SMC,
                           ConstraintManagerCreator CMC)
  : CoreEngine(cfg, CD, Ctx, *this), 
    G(CoreEngine.getGraph()),
    Liveness(L),
    Builder(NULL),
    StateMgr(G.getContext(), SMC, CMC, G.getAllocator(), cfg, CD, L),
    SymMgr(StateMgr.getSymbolManager()),
    CurrentStmt(NULL),
    NSExceptionII(NULL), NSExceptionInstanceRaiseSelectors(NULL),
    RaiseSel(GetNullarySelector("raise", G.getContext())), 
    PurgeDead(purgeDead),
    BR(BRD, *this),
    EagerlyAssume(eagerlyAssume) {}

GRExprEngine::~GRExprEngine() {    
  BR.FlushReports();
  delete [] NSExceptionInstanceRaiseSelectors;
}

//===----------------------------------------------------------------------===//
// Utility methods.
//===----------------------------------------------------------------------===//


void GRExprEngine::setTransferFunctions(GRTransferFuncs* tf) {
  StateMgr.TF = tf;
  tf->RegisterChecks(getBugReporter());
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
  SymbolReaper SymReaper(Liveness, SymMgr);  
  CleanedState = PurgeDead ? StateMgr.RemoveDeadBindings(EntryNode->getState(), 
                                                         CurrentStmt, SymReaper)
                           : EntryNode->getState();

  // Process any special transfer function for dead symbols.
  NodeSet Tmp;
  
  if (!SymReaper.hasDeadSymbols())
    Tmp.Add(EntryNode);
  else {
    SaveAndRestore<bool> OldSink(Builder->BuildSinks);
    SaveOr OldHasGen(Builder->HasGeneratedNode);

    SaveAndRestore<bool> OldPurgeDeadSymbols(Builder->PurgingDeadSymbols);
    Builder->PurgingDeadSymbols = true;
    
    getTF().EvalDeadSymbols(Tmp, *this, *Builder, EntryNode, S, 
                            CleanedState, SymReaper);

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
        const GRState* state = GetState(Pred);
        MakeNode(Dst, B, Pred, BindExpr(state, B, GetSVal(state, B->getRHS())));
        break;
      }

      if (EagerlyAssume && (B->isRelationalOp() || B->isEqualityOp())) {
        NodeSet Tmp;
        VisitBinaryOperator(cast<BinaryOperator>(S), Pred, Tmp);
        EvalEagerlyAssume(Dst, Tmp, cast<Expr>(S));        
      }
      else
        VisitBinaryOperator(cast<BinaryOperator>(S), Pred, Dst);

      break;
    }

    case Stmt::CallExprClass:
    case Stmt::CXXOperatorCallExprClass: {
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
    case Stmt::QualifiedDeclRefExprClass:
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

    case Stmt::ObjCForCollectionStmtClass:
      VisitObjCForCollectionStmt(cast<ObjCForCollectionStmt>(S), Pred, Dst);
      break;
      
    case Stmt::ObjCMessageExprClass: {
      VisitObjCMessageExpr(cast<ObjCMessageExpr>(S), Pred, Dst);
      break;
    }
      
    case Stmt::ObjCAtThrowStmtClass: {
      // FIXME: This is not complete.  We basically treat @throw as
      // an abort.
      SaveAndRestore<bool> OldSink(Builder->BuildSinks);
      Builder->BuildSinks = true;
      MakeNode(Dst, S, Pred, GetState(Pred));
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

      if (SE->getSubStmt()->body_empty()) {
        // Empty statement expression.
        assert(SE->getType() == getContext().VoidTy
               && "Empty statement expression must have void type.");
        Dst.Add(Pred);
        break;
      }
               
      if (Expr* LastExpr = dyn_cast<Expr>(*SE->getSubStmt()->body_rbegin())) {
        const GRState* state = GetState(Pred);
        MakeNode(Dst, SE, Pred, BindExpr(state, SE, GetSVal(state, LastExpr)));
      }
      else
        Dst.Add(Pred);
      
      break;
    }

    case Stmt::StringLiteralClass:
      VisitLValue(cast<StringLiteral>(S), Pred, Dst);
      break;
      
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
    case Stmt::QualifiedDeclRefExprClass:
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
      const GRState* state = GetState(Pred);
      SVal V = StateMgr.GetLValue(state, cast<StringLiteral>(Ex));
      MakeNode(Dst, Ex, Pred, BindExpr(state, Ex, V));
      return;
    }
      
    default:
      // Arbitrary subexpressions can return aggregate temporaries that
      // can be used in a lvalue context.  We need to enhance our support
      // of such temporaries in both the environment and the store, so right
      // now we just do a regular visit.
      assert ((Ex->getType()->isAggregateType()) &&
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

const GRState* GRExprEngine::MarkBranch(const GRState* state,
                                           Stmt* Terminator,
                                           bool branchTaken) {
  
  switch (Terminator->getStmtClass()) {
    default:
      return state;
      
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
        
      return BindBlkExpr(state, B, UndefinedVal(Ex));
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
      
      return BindBlkExpr(state, C, UndefinedVal(Ex));
    }
      
    case Stmt::ChooseExprClass: { // ?:
      
      ChooseExpr* C = cast<ChooseExpr>(Terminator);
      
      Expr* Ex = branchTaken ? C->getLHS() : C->getRHS();      
      return BindBlkExpr(state, C, UndefinedVal(Ex));
    }
  }
}

void GRExprEngine::ProcessBranch(Stmt* Condition, Stmt* Term,
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
  const GRState* state = Assume(PrevState, V, true, isFeasible);

  if (isFeasible)
    builder.generateNode(MarkBranch(state, Term, true), true);
  else
    builder.markInfeasible(true);
      
  // Process the false branch.  
  
  isFeasible = false;
  state = Assume(PrevState, V, false, isFeasible);
  
  if (isFeasible)
    builder.generateNode(MarkBranch(state, Term, false), false);
  else
    builder.markInfeasible(false);
}

/// ProcessIndirectGoto - Called by GRCoreEngine.  Used to generate successor
///  nodes by processing the 'effects' of a computed goto jump.
void GRExprEngine::ProcessIndirectGoto(IndirectGotoNodeBuilder& builder) {

  const GRState* state = builder.getState();  
  SVal V = GetSVal(state, builder.getTarget());
  
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
        builder.generateNode(I, state);
        return;
      }
    }
    
    assert (false && "No block with label.");
    return;
  }

  if (isa<loc::ConcreteInt>(V) || isa<UndefinedVal>(V)) {
    // Dispatch to the first target and mark it as a sink.
    NodeTy* N = builder.generateNode(builder.begin(), state, true);
    UndefBranches.insert(N);
    return;
  }
  
  // This is really a catch-all.  We don't support symbolics yet.
  
  assert (V.isUnknown());
  
  for (iterator I=builder.begin(), E=builder.end(); I != E; ++I)
    builder.generateNode(I, state);
}


void GRExprEngine::VisitGuardedExpr(Expr* Ex, Expr* L, Expr* R,
                                    NodeTy* Pred, NodeSet& Dst) {
  
  assert (Ex == CurrentStmt && getCFG().isBlkExpr(Ex));
  
  const GRState* state = GetState(Pred);
  SVal X = GetBlkExprSVal(state, Ex);
  
  assert (X.isUndef());
  
  Expr* SE = (Expr*) cast<UndefinedVal>(X).getData();
  
  assert (SE);
  
  X = GetBlkExprSVal(state, SE);
  
  // Make sure that we invalidate the previous binding.
  MakeNode(Dst, Ex, Pred, StateMgr.BindExpr(state, Ex, X, true, true));
}

/// ProcessSwitch - Called by GRCoreEngine.  Used to generate successor
///  nodes by processing the 'effects' of a switch statement.
void GRExprEngine::ProcessSwitch(SwitchNodeBuilder& builder) {  
  typedef SwitchNodeBuilder::iterator iterator;  
  const GRState* state = builder.getState();  
  Expr* CondE = builder.getCondition();
  SVal  CondV = GetSVal(state, CondE);

  if (CondV.isUndef()) {
    NodeTy* N = builder.generateDefaultCaseNode(state, true);
    UndefBranches.insert(N);
    return;
  }

  const GRState*  DefaultSt = state;  
  bool DefaultFeasible = false;
  
  for (iterator I = builder.begin(), EI = builder.end(); I != EI; ++I) {
    CaseStmt* Case = cast<CaseStmt>(I.getCase());

    // Evaluate the LHS of the case value.
    Expr::EvalResult V1;
    bool b = Case->getLHS()->Evaluate(V1, getContext());    
    
    // Sanity checks.  These go away in Release builds.
    assert(b && V1.Val.isInt() && !V1.HasSideEffects 
             && "Case condition must evaluate to an integer constant.");
    b = b; // silence unused variable warning    
    assert(V1.Val.getInt().getBitWidth() == 
           getContext().getTypeSize(CondE->getType()));
           
    // Get the RHS of the case, if it exists.
    Expr::EvalResult V2;
    
    if (Expr* E = Case->getRHS()) {
      b = E->Evaluate(V2, getContext());
      assert(b && V2.Val.isInt() && !V2.HasSideEffects 
             && "Case condition must evaluate to an integer constant.");
      b = b; // silence unused variable warning
    }
    else
      V2 = V1;
    
    // FIXME: Eventually we should replace the logic below with a range
    //  comparison, rather than concretize the values within the range.
    //  This should be easy once we have "ranges" for NonLVals.
        
    do {
      nonloc::ConcreteInt CaseVal(getBasicVals().getValue(V1.Val.getInt()));      
      SVal Res = EvalBinOp(BinaryOperator::EQ, CondV, CaseVal);
      
      // Now "assume" that the case matches.      
      bool isFeasible = false;      
      const GRState* StNew = Assume(state, Res, true, isFeasible);
      
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
      if (V1.Val.getInt() == V2.Val.getInt())
        break;
      
      ++V1.Val.getInt();
      assert (V1.Val.getInt() <= V2.Val.getInt());
      
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
  
  const GRState* state = GetState(Pred);
  SVal X = GetBlkExprSVal(state, B);
  
  assert (X.isUndef());
  
  Expr* Ex = (Expr*) cast<UndefinedVal>(X).getData();
  
  assert (Ex);
  
  if (Ex == B->getRHS()) {
    
    X = GetBlkExprSVal(state, Ex);
    
    // Handle undefined values.
    
    if (X.isUndef()) {
      MakeNode(Dst, B, Pred, BindBlkExpr(state, B, X));
      return;
    }
    
    // We took the RHS.  Because the value of the '&&' or '||' expression must
    // evaluate to 0 or 1, we must assume the value of the RHS evaluates to 0
    // or 1.  Alternatively, we could take a lazy approach, and calculate this
    // value later when necessary.  We don't have the machinery in place for
    // this right now, and since most logical expressions are used for branches,
    // the payoff is not likely to be large.  Instead, we do eager evaluation.
        
    bool isFeasible = false;
    const GRState* NewState = Assume(state, X, true, isFeasible);
    
    if (isFeasible)
      MakeNode(Dst, B, Pred,
               BindBlkExpr(NewState, B, MakeConstantVal(1U, B)));
      
    isFeasible = false;
    NewState = Assume(state, X, false, isFeasible);
    
    if (isFeasible)
      MakeNode(Dst, B, Pred,
               BindBlkExpr(NewState, B, MakeConstantVal(0U, B)));
  }
  else {
    // We took the LHS expression.  Depending on whether we are '&&' or
    // '||' we know what the value of the expression is via properties of
    // the short-circuiting.
    
    X = MakeConstantVal( B->getOpcode() == BinaryOperator::LAnd ? 0U : 1U, B);
    MakeNode(Dst, B, Pred, BindBlkExpr(state, B, X));
  }
}
 
//===----------------------------------------------------------------------===//
// Transfer functions: Loads and stores.
//===----------------------------------------------------------------------===//

void GRExprEngine::VisitDeclRefExpr(DeclRefExpr* Ex, NodeTy* Pred, NodeSet& Dst,
                                    bool asLValue) {
  
  const GRState* state = GetState(Pred);

  const NamedDecl* D = Ex->getDecl();

  if (const VarDecl* VD = dyn_cast<VarDecl>(D)) {

    SVal V = StateMgr.GetLValue(state, VD);

    if (asLValue)
      MakeNode(Dst, Ex, Pred, BindExpr(state, Ex, V));
    else
      EvalLoad(Dst, Ex, Pred, state, V);
    return;

  } else if (const EnumConstantDecl* ED = dyn_cast<EnumConstantDecl>(D)) {
    assert(!asLValue && "EnumConstantDecl does not have lvalue.");

    BasicValueFactory& BasicVals = StateMgr.getBasicVals();
    SVal V = nonloc::ConcreteInt(BasicVals.getValue(ED->getInitVal()));
    MakeNode(Dst, Ex, Pred, BindExpr(state, Ex, V));
    return;

  } else if (const FunctionDecl* FD = dyn_cast<FunctionDecl>(D)) {
    assert(asLValue);
    SVal V = loc::FuncVal(FD);
    MakeNode(Dst, Ex, Pred, BindExpr(state, Ex, V));
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
  
  if (Base->getType()->isVectorType()) {
    // For vector types get its lvalue.
    // FIXME: This may not be correct.  Is the rvalue of a vector its location?
    //  In fact, I think this is just a hack.  We need to get the right
    // semantics.
    VisitLValue(Base, Pred, Tmp);
  }
  else  
    Visit(Base, Pred, Tmp);   // Get Base's rvalue, which should be an LocVal.
  
  for (NodeSet::iterator I1=Tmp.begin(), E1=Tmp.end(); I1!=E1; ++I1) {    
    NodeSet Tmp2;
    Visit(Idx, *I1, Tmp2);     // Evaluate the index.
      
    for (NodeSet::iterator I2=Tmp2.begin(), E2=Tmp2.end(); I2!=E2; ++I2) {
      const GRState* state = GetState(*I2);
      SVal V = StateMgr.GetLValue(state, GetSVal(state, Base),
                                  GetSVal(state, Idx));

      if (asLValue)
        MakeNode(Dst, A, *I2, BindExpr(state, A, V));
      else
        EvalLoad(Dst, A, *I2, state, V);
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
    
  FieldDecl *Field = dyn_cast<FieldDecl>(M->getMemberDecl());
  if (!Field) // FIXME: skipping member expressions for non-fields
    return;

  for (NodeSet::iterator I = Tmp.begin(), E = Tmp.end(); I != E; ++I) {
    const GRState* state = GetState(*I);
    // FIXME: Should we insert some assumption logic in here to determine
    // if "Base" is a valid piece of memory?  Before we put this assumption
    // later when using FieldOffset lvals (which we no longer have).
    SVal L = StateMgr.GetLValue(state, GetSVal(state, Base), Field);

    if (asLValue)
      MakeNode(Dst, M, *I, BindExpr(state, M, L));
    else
      EvalLoad(Dst, M, *I, state, L);
  }
}

/// EvalBind - Handle the semantics of binding a value to a specific location.
///  This method is used by EvalStore and (soon) VisitDeclStmt, and others.
void GRExprEngine::EvalBind(NodeSet& Dst, Expr* Ex, NodeTy* Pred,
                             const GRState* state, SVal location, SVal Val) {

  const GRState* newState = 0;
  
  if (location.isUnknown()) {
    // We know that the new state will be the same as the old state since
    // the location of the binding is "unknown".  Consequently, there
    // is no reason to just create a new node.
    newState = state;
  }
  else {
    // We are binding to a value other than 'unknown'.  Perform the binding
    // using the StoreManager.
    newState = StateMgr.BindLoc(state, cast<Loc>(location), Val);
  }

  // The next thing to do is check if the GRTransferFuncs object wants to
  // update the state based on the new binding.  If the GRTransferFunc object
  // doesn't do anything, just auto-propagate the current state.
  GRStmtNodeBuilderRef BuilderRef(Dst, *Builder, *this, Pred, newState, Ex,
                                  newState != state);
    
  getTF().EvalBind(BuilderRef, location, Val);
}

/// EvalStore - Handle the semantics of a store via an assignment.
///  @param Dst The node set to store generated state nodes
///  @param Ex The expression representing the location of the store
///  @param state The current simulation state
///  @param location The location to store the value
///  @param Val The value to be stored
void GRExprEngine::EvalStore(NodeSet& Dst, Expr* Ex, NodeTy* Pred,
                             const GRState* state, SVal location, SVal Val) {
  
  assert (Builder && "GRStmtNodeBuilder must be defined.");
  
  // Evaluate the location (checks for bad dereferences).
  Pred = EvalLocation(Ex, Pred, state, location);
  
  if (!Pred)
    return;

  assert (!location.isUndef());
  state = GetState(Pred);

  // Proceed with the store.  
  SaveAndRestore<ProgramPoint::Kind> OldSPointKind(Builder->PointKind);
  Builder->PointKind = ProgramPoint::PostStoreKind;  
  EvalBind(Dst, Ex, Pred, state, location, Val);
}

void GRExprEngine::EvalLoad(NodeSet& Dst, Expr* Ex, NodeTy* Pred,
                            const GRState* state, SVal location) {

  // Evaluate the location (checks for bad dereferences).  
  Pred = EvalLocation(Ex, Pred, state, location);
  
  if (!Pred)
    return;
  
  state = GetState(Pred);
  
  // Proceed with the load.
  ProgramPoint::Kind K = ProgramPoint::PostLoadKind;

  // FIXME: Currently symbolic analysis "generates" new symbols
  //  for the contents of values.  We need a better approach.

  if (location.isUnknown()) {
    // This is important.  We must nuke the old binding.
    MakeNode(Dst, Ex, Pred, BindExpr(state, Ex, UnknownVal()), K);
  }
  else {
    SVal V = GetSVal(state, cast<Loc>(location), Ex->getType());
    MakeNode(Dst, Ex, Pred, BindExpr(state, Ex, V), K);  
  }
}

void GRExprEngine::EvalStore(NodeSet& Dst, Expr* Ex, Expr* StoreE, NodeTy* Pred,
                             const GRState* state, SVal location, SVal Val) {
 
  NodeSet TmpDst;
  EvalStore(TmpDst, StoreE, Pred, state, location, Val);

  for (NodeSet::iterator I=TmpDst.begin(), E=TmpDst.end(); I!=E; ++I)
    MakeNode(Dst, Ex, *I, (*I)->getState());
}

GRExprEngine::NodeTy* GRExprEngine::EvalLocation(Stmt* Ex, NodeTy* Pred,
                                                 const GRState* state,
                                                 SVal location) {
  
  // Check for loads/stores from/to undefined values.  
  if (location.isUndef()) {
    NodeTy* N =
      Builder->generateNode(Ex, state, Pred,
                            ProgramPoint::PostUndefLocationCheckFailedKind);
    
    if (N) {
      N->markAsSink();
      UndefDeref.insert(N);
    }
    
    return 0;
  }
  
  // Check for loads/stores from/to unknown locations.  Treat as No-Ops.
  if (location.isUnknown())
    return Pred;
  
  // During a load, one of two possible situations arise:
  //  (1) A crash, because the location (pointer) was NULL.
  //  (2) The location (pointer) is not NULL, and the dereference works.
  // 
  // We add these assumptions.
  
  Loc LV = cast<Loc>(location);    
  
  // "Assume" that the pointer is not NULL.
  bool isFeasibleNotNull = false;
  const GRState* StNotNull = Assume(state, LV, true, isFeasibleNotNull);
  
  // "Assume" that the pointer is NULL.
  bool isFeasibleNull = false;
  GRStateRef StNull = GRStateRef(Assume(state, LV, false, isFeasibleNull),
                                 getStateManager());
  
  if (isFeasibleNull) {
    
    // Use the Generic Data Map to mark in the state what lval was null.
    const SVal* PersistentLV = getBasicVals().getPersistentSVal(LV);
    StNull = StNull.set<GRState::NullDerefTag>(PersistentLV);
    
    // We don't use "MakeNode" here because the node will be a sink
    // and we have no intention of processing it later.
    NodeTy* NullNode =
      Builder->generateNode(Ex, StNull, Pred, 
                            ProgramPoint::PostNullCheckFailedKind);

    if (NullNode) {
      
      NullNode->markAsSink();
      
      if (isFeasibleNotNull) ImplicitNullDeref.insert(NullNode);
      else ExplicitNullDeref.insert(NullNode);
    }
  }
  
  if (!isFeasibleNotNull)
    return 0;

  // Check for out-of-bound array access.
  if (isa<loc::MemRegionVal>(LV)) {
    const MemRegion* R = cast<loc::MemRegionVal>(LV).getRegion();
    if (const ElementRegion* ER = dyn_cast<ElementRegion>(R)) {
      // Get the index of the accessed element.
      SVal Idx = ER->getIndex();
      // Get the extent of the array.
      SVal NumElements = getStoreManager().getSizeInElements(StNotNull,
                                                          ER->getSuperRegion());

      bool isFeasibleInBound = false;
      const GRState* StInBound = AssumeInBound(StNotNull, Idx, NumElements, 
                                               true, isFeasibleInBound);

      bool isFeasibleOutBound = false;
      const GRState* StOutBound = AssumeInBound(StNotNull, Idx, NumElements, 
                                                false, isFeasibleOutBound);

      if (isFeasibleOutBound) {
        // Report warning.  Make sink node manually.
        NodeTy* OOBNode =
          Builder->generateNode(Ex, StOutBound, Pred,
                                ProgramPoint::PostOutOfBoundsCheckFailedKind);

        if (OOBNode) {
          OOBNode->markAsSink();

          if (isFeasibleInBound)
            ImplicitOOBMemAccesses.insert(OOBNode);
          else
            ExplicitOOBMemAccesses.insert(OOBNode);
        }
      }

      if (!isFeasibleInBound)
        return 0;
      
      StNotNull = StInBound;
    }
  }
  
  // Generate a new node indicating the checks succeed.
  return Builder->generateNode(Ex, StNotNull, Pred,
                               ProgramPoint::PostLocationChecksSucceedKind);
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
  const FunctionProtoType *Proto = NULL;
  QualType FnType = CE->getCallee()->IgnoreParens()->getType();
  if (const PointerType *FnTypePtr = FnType->getAsPointerType())
    Proto = FnTypePtr->getPointeeType()->getAsFunctionProtoType();

  VisitCallRec(CE, Pred, AI, AE, Dst, Proto, /*ParamIdx=*/0);
}

void GRExprEngine::VisitCallRec(CallExpr* CE, NodeTy* Pred,
                                CallExpr::arg_iterator AI,
                                CallExpr::arg_iterator AE,
                                NodeSet& Dst, const FunctionProtoType *Proto,
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

    const GRState* state = GetState(*DI);
    SVal L = GetSVal(state, Callee);

    // FIXME: Add support for symbolic function calls (calls involving
    //  function pointer values that are symbolic).
    
    // Check for undefined control-flow or calls to NULL.
    
    if (L.isUndef() || isa<loc::ConcreteInt>(L)) {      
      NodeTy* N = Builder->generateNode(CE, state, *DI);
      
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
                SVal X = GetSVal(state, *CE->arg_begin());
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
            if (!memcmp(s ,"db_error", 8) || 
                !memcmp(s, "__assert", 8))
              Builder->BuildSinks = true;
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
                !memcmp(s, "_DTAssertionFailureHandler", 26) ||
                !memcmp(s, "_TSAssertionFailureHandler", 26))
              Builder->BuildSinks = true;

            break;
        }
        
      }
    }
    
    // Evaluate the call.

    if (isa<loc::FuncVal>(L)) {
      
      if (unsigned id 
            = cast<loc::FuncVal>(L).getDecl()->getBuiltinID(getContext()))
        switch (id) {
          case Builtin::BI__builtin_expect: {
            // For __builtin_expect, just return the value of the subexpression.
            assert (CE->arg_begin() != CE->arg_end());            
            SVal X = GetSVal(state, *(CE->arg_begin()));
            MakeNode(Dst, CE, *DI, BindExpr(state, CE, X));
            continue;            
          }
            
          case Builtin::BI__builtin_alloca: {
            // FIXME: Refactor into StoreManager itself?
            MemRegionManager& RM = getStateManager().getRegionManager();
            const MemRegion* R =
              RM.getAllocaRegion(CE, Builder->getCurrentBlockCount());

            // Set the extent of the region in bytes. This enables us to use the
            // SVal of the argument directly. If we save the extent in bits, we
            // cannot represent values like symbol*8.
            SVal Extent = GetSVal(state, *(CE->arg_begin()));
            state = getStoreManager().setExtent(state, R, Extent);

            MakeNode(Dst, CE, *DI, BindExpr(state, CE, loc::MemRegionVal(R)));
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
      MakeNode(Dst, CE, *DI, state);
  }
}

//===----------------------------------------------------------------------===//
// Transfer function: Objective-C ivar references.
//===----------------------------------------------------------------------===//

static std::pair<const void*,const void*> EagerlyAssumeTag
  = std::pair<const void*,const void*>(&EagerlyAssumeTag,0);

void GRExprEngine::EvalEagerlyAssume(NodeSet &Dst, NodeSet &Src, Expr *Ex) {
  for (NodeSet::iterator I=Src.begin(), E=Src.end(); I!=E; ++I) {
    NodeTy *Pred = *I;
    
    // Test if the previous node was as the same expression.  This can happen
    // when the expression fails to evaluate to anything meaningful and
    // (as an optimization) we don't generate a node.
    ProgramPoint P = Pred->getLocation();    
    if (!isa<PostStmt>(P) || cast<PostStmt>(P).getStmt() != Ex) {
      Dst.Add(Pred);      
      continue;
    }    

    const GRState* state = Pred->getState();    
    SVal V = GetSVal(state, Ex);    
    if (isa<nonloc::SymIntConstraintVal>(V)) {
      // First assume that the condition is true.
      bool isFeasible = false;
      const GRState *stateTrue = Assume(state, V, true, isFeasible);
      if (isFeasible) {
        stateTrue = BindExpr(stateTrue, Ex, MakeConstantVal(1U, Ex));        
        Dst.Add(Builder->generateNode(PostStmtCustom(Ex, &EagerlyAssumeTag),
                                      stateTrue, Pred));
      }
        
      // Next, assume that the condition is false.
      isFeasible = false;
      const GRState *stateFalse = Assume(state, V, false, isFeasible);
      if (isFeasible) {
        stateFalse = BindExpr(stateFalse, Ex, MakeConstantVal(0U, Ex));
        Dst.Add(Builder->generateNode(PostStmtCustom(Ex, &EagerlyAssumeTag),
                                      stateFalse, Pred));
      }
    }
    else
      Dst.Add(Pred);
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
    const GRState* state = GetState(*I);
    SVal BaseVal = GetSVal(state, Base);
    SVal location = StateMgr.GetLValue(state, Ex->getDecl(), BaseVal);
    
    if (asLValue)
      MakeNode(Dst, Ex, *I, BindExpr(state, Ex, location));
    else
      EvalLoad(Dst, Ex, *I, state, location);
  }
}

//===----------------------------------------------------------------------===//
// Transfer function: Objective-C fast enumeration 'for' statements.
//===----------------------------------------------------------------------===//

void GRExprEngine::VisitObjCForCollectionStmt(ObjCForCollectionStmt* S,
                                              NodeTy* Pred, NodeSet& Dst) {
    
  // ObjCForCollectionStmts are processed in two places.  This method
  // handles the case where an ObjCForCollectionStmt* occurs as one of the
  // statements within a basic block.  This transfer function does two things:
  //
  //  (1) binds the next container value to 'element'.  This creates a new
  //      node in the ExplodedGraph.
  //
  //  (2) binds the value 0/1 to the ObjCForCollectionStmt* itself, indicating
  //      whether or not the container has any more elements.  This value
  //      will be tested in ProcessBranch.  We need to explicitly bind
  //      this value because a container can contain nil elements.
  //  
  // FIXME: Eventually this logic should actually do dispatches to
  //   'countByEnumeratingWithState:objects:count:' (NSFastEnumeration).
  //   This will require simulating a temporary NSFastEnumerationState, either
  //   through an SVal or through the use of MemRegions.  This value can
  //   be affixed to the ObjCForCollectionStmt* instead of 0/1; when the loop
  //   terminates we reclaim the temporary (it goes out of scope) and we
  //   we can test if the SVal is 0 or if the MemRegion is null (depending
  //   on what approach we take).
  //
  //  For now: simulate (1) by assigning either a symbol or nil if the
  //    container is empty.  Thus this transfer function will by default
  //    result in state splitting.
  
  Stmt* elem = S->getElement();
  SVal ElementV;
    
  if (DeclStmt* DS = dyn_cast<DeclStmt>(elem)) {
    VarDecl* ElemD = cast<VarDecl>(DS->getSolitaryDecl());
    assert (ElemD->getInit() == 0);
    ElementV = getStateManager().GetLValue(GetState(Pred), ElemD);
    VisitObjCForCollectionStmtAux(S, Pred, Dst, ElementV);
    return;
  }

  NodeSet Tmp;
  VisitLValue(cast<Expr>(elem), Pred, Tmp);
  
  for (NodeSet::iterator I = Tmp.begin(), E = Tmp.end(); I!=E; ++I) {
    const GRState* state = GetState(*I);
    VisitObjCForCollectionStmtAux(S, *I, Dst, GetSVal(state, elem));
  }
}

void GRExprEngine::VisitObjCForCollectionStmtAux(ObjCForCollectionStmt* S,
                                                 NodeTy* Pred, NodeSet& Dst,
                                                 SVal ElementV) {
    

  
  // Get the current state.  Use 'EvalLocation' to determine if it is a null
  // pointer, etc.
  Stmt* elem = S->getElement();
  
  Pred = EvalLocation(elem, Pred, GetState(Pred), ElementV);
  if (!Pred)
    return;
    
  GRStateRef state = GRStateRef(GetState(Pred), getStateManager());

  // Handle the case where the container still has elements.
  QualType IntTy = getContext().IntTy;
  SVal TrueV = NonLoc::MakeVal(getBasicVals(), 1, IntTy);
  GRStateRef hasElems = state.BindExpr(S, TrueV);
  
  // Handle the case where the container has no elements.
  SVal FalseV = NonLoc::MakeVal(getBasicVals(), 0, IntTy);
  GRStateRef noElems = state.BindExpr(S, FalseV);
  
  if (loc::MemRegionVal* MV = dyn_cast<loc::MemRegionVal>(&ElementV))
    if (const TypedRegion* R = dyn_cast<TypedRegion>(MV->getRegion())) {
      // FIXME: The proper thing to do is to really iterate over the
      //  container.  We will do this with dispatch logic to the store.
      //  For now, just 'conjure' up a symbolic value.
      QualType T = R->getRValueType(getContext());
      assert (Loc::IsLocType(T));
      unsigned Count = Builder->getCurrentBlockCount();
      loc::SymbolVal SymV(SymMgr.getConjuredSymbol(elem, T, Count));
      hasElems = hasElems.BindLoc(ElementV, SymV);

      // Bind the location to 'nil' on the false branch.
      SVal nilV = loc::ConcreteInt(getBasicVals().getValue(0, T));      
      noElems = noElems.BindLoc(ElementV, nilV);      
    }
  
  // Create the new nodes.
  MakeNode(Dst, S, Pred, hasElems);
  MakeNode(Dst, S, Pred, noElems);
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
  
  const GRState* state = GetState(Pred);
  bool RaisesException = false;
  
  
  if (Expr* Receiver = ME->getReceiver()) {
    
    SVal L = GetSVal(state, Receiver);
    
    // Check for undefined control-flow.    
    if (L.isUndef()) {
      NodeTy* N = Builder->generateNode(ME, state, Pred);
      
      if (N) {
        N->markAsSink();
        UndefReceivers.insert(N);
      }
      
      return;
    }
    
    // "Assume" that the receiver is not NULL.    
    bool isFeasibleNotNull = false;
    Assume(state, L, true, isFeasibleNotNull);
    
    // "Assume" that the receiver is NULL.    
    bool isFeasibleNull = false;
    const GRState *StNull = Assume(state, L, false, isFeasibleNull);
    
    if (isFeasibleNull) {
      // Check if the receiver was nil and the return value a struct.
      if (ME->getType()->isRecordType()) {
        // The [0 ...] expressions will return garbage.  Flag either an
        // explicit or implicit error.  Because of the structure of this
        // function we currently do not bifurfacte the state graph at
        // this point.
        // FIXME: We should bifurcate and fill the returned struct with
        //  garbage.                
        if (NodeTy* N = Builder->generateNode(ME, StNull, Pred)) {
          N->markAsSink();
          if (isFeasibleNotNull)
            NilReceiverStructRetImplicit.insert(N);
          else
            NilReceiverStructRetExplicit.insert(N);
        }
      }
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
    
    if (GetSVal(state, *I).isUndef()) {
      
      // Generate an error node for passing an uninitialized/undefined value
      // as an argument to a message expression.  This node is a sink.
      NodeTy* N = Builder->generateNode(ME, state, Pred);
      
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
    MakeNode(Dst, ME, Pred, state);
}

//===----------------------------------------------------------------------===//
// Transfer functions: Miscellaneous statements.
//===----------------------------------------------------------------------===//

void GRExprEngine::VisitCastPointerToInteger(SVal V, const GRState* state,
                                             QualType PtrTy,
                                             Expr* CastE, NodeTy* Pred,
                                             NodeSet& Dst) {
  if (!V.isUnknownOrUndef()) {
    // FIXME: Determine if the number of bits of the target type is 
    // equal or exceeds the number of bits to store the pointer value.
    // If not, flag an error.    
    MakeNode(Dst, CastE, Pred, BindExpr(state, CastE, EvalCast(cast<Loc>(V),
                                                               CastE->getType())));
  }
  else  
    MakeNode(Dst, CastE, Pred, BindExpr(state, CastE, V));
}

  
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
    const GRState* state = GetState(N);
    SVal V = GetSVal(state, Ex);
    ASTContext& C = getContext();

    // Unknown?
    if (V.isUnknown()) {
      Dst.Add(N);
      continue;
    }
    
    // Undefined?
    if (V.isUndef())
      goto PassThrough;
    
    // For const casts, just propagate the value.
    if (C.getCanonicalType(T).getUnqualifiedType() == 
        C.getCanonicalType(ExTy).getUnqualifiedType())
      goto PassThrough;
      
    // Check for casts from pointers to integers.
    if (T->isIntegerType() && Loc::IsLocType(ExTy)) {
      VisitCastPointerToInteger(V, state, ExTy, CastE, N, Dst);
      continue;
    }
    
    // Check for casts from integers to pointers.
    if (Loc::IsLocType(T) && ExTy->isIntegerType()) {
      if (nonloc::LocAsInteger *LV = dyn_cast<nonloc::LocAsInteger>(&V)) {
        // Just unpackage the lval and return it.
        V = LV->getLoc();
        MakeNode(Dst, CastE, N, BindExpr(state, CastE, V));
        continue;
      }
      
      goto DispatchCast;
    }
    
    // Just pass through function and block pointers.
    if (ExTy->isBlockPointerType() || ExTy->isFunctionPointerType()) {
      assert(Loc::IsLocType(T));
      goto PassThrough;
    }
    
    // Check for casts from array type to another type.
    if (ExTy->isArrayType()) {
      // We will always decay to a pointer.
      V = StateMgr.ArrayToPointer(V);
      
      // Are we casting from an array to a pointer?  If so just pass on
      // the decayed value.
      if (T->isPointerType())
        goto PassThrough;
      
      // Are we casting from an array to an integer?  If so, cast the decayed
      // pointer value to an integer.
      assert(T->isIntegerType());
      QualType ElemTy = cast<ArrayType>(ExTy)->getElementType();
      QualType PointerTy = getContext().getPointerType(ElemTy);
      VisitCastPointerToInteger(V, state, PointerTy, CastE, N, Dst);
      continue;
    }

    // Check for casts from a region to a specific type.
    if (loc::MemRegionVal *RV = dyn_cast<loc::MemRegionVal>(&V)) {      
      // FIXME: For TypedViewRegions, we should handle the case where the
      //  underlying symbolic pointer is a function pointer or
      //  block pointer.
      
      // FIXME: We should handle the case where we strip off view layers to get
      //  to a desugared type.
      
      assert(Loc::IsLocType(T));
      assert(Loc::IsLocType(ExTy));

      const MemRegion* R = RV->getRegion();
      StoreManager& StoreMgr = getStoreManager();
      
      // Delegate to store manager to get the result of casting a region
      // to a different type.
      const StoreManager::CastResult& Res = StoreMgr.CastRegion(state, R, T);
      
      // Inspect the result.  If the MemRegion* returned is NULL, this
      // expression evaluates to UnknownVal.
      R = Res.getRegion();
      if (R) { V = loc::MemRegionVal(R); } else { V = UnknownVal(); }
      
      // Generate the new node in the ExplodedGraph.
      MakeNode(Dst, CastE, N, BindExpr(Res.getState(), CastE, V));
      continue;
    }

    // If we are casting a symbolic value, make a symbolic region and a
    // TypedViewRegion subregion.
    if (loc::SymbolVal* SV = dyn_cast<loc::SymbolVal>(&V)) {
      SymbolRef Sym = SV->getSymbol();
      QualType SymTy = getSymbolManager().getType(Sym);

      // Just pass through symbols that are function or block pointers.
      if (SymTy->isFunctionPointerType() || SymTy->isBlockPointerType())
        goto PassThrough;
      
      // Are we casting to a function or block pointer?
      if (T->isFunctionPointerType() || T->isBlockPointerType()) {
        // FIXME: We should verify that the underlying type of the symbolic 
        // pointer is a void* (or maybe char*).  Other things are an abuse
        // of the type system.
        goto PassThrough;        
      }

      StoreManager& StoreMgr = getStoreManager();
      const MemRegion* R =
        StoreMgr.getRegionManager().getSymbolicRegion(Sym, getSymbolManager());
      
      // Delegate to store manager to get the result of casting a region
      // to a different type.
      const StoreManager::CastResult& Res = StoreMgr.CastRegion(state, R, T);
      
      // Inspect the result.  If the MemRegion* returned is NULL, this
      // expression evaluates to UnknownVal.
      R = Res.getRegion();
      if (R) { V = loc::MemRegionVal(R); } else { V = UnknownVal(); }
      
      // Generate the new node in the ExplodedGraph.
      MakeNode(Dst, CastE, N, BindExpr(Res.getState(), CastE, V));
      continue;
    }

        // All other cases.
    DispatchCast: {
      MakeNode(Dst, CastE, N, BindExpr(state, CastE,
                                       EvalCast(V, CastE->getType())));
      continue;
    }
    
    PassThrough: {
      MakeNode(Dst, CastE, N, BindExpr(state, CastE, V));
    }
  }
}

void GRExprEngine::VisitCompoundLiteralExpr(CompoundLiteralExpr* CL,
                                            NodeTy* Pred, NodeSet& Dst, 
                                            bool asLValue) {
  InitListExpr* ILE = cast<InitListExpr>(CL->getInitializer()->IgnoreParens());
  NodeSet Tmp;
  Visit(ILE, Pred, Tmp);
  
  for (NodeSet::iterator I = Tmp.begin(), EI = Tmp.end(); I!=EI; ++I) {
    const GRState* state = GetState(*I);
    SVal ILV = GetSVal(state, ILE);
    state = StateMgr.BindCompoundLiteral(state, CL, ILV);

    if (asLValue)
      MakeNode(Dst, CL, *I, BindExpr(state, CL, StateMgr.GetLValue(state, CL)));
    else
      MakeNode(Dst, CL, *I, BindExpr(state, CL, ILV));
  }
}

void GRExprEngine::VisitDeclStmt(DeclStmt* DS, NodeTy* Pred, NodeSet& Dst) {  

  // The CFG has one DeclStmt per Decl.  
  Decl* D = *DS->decl_begin();
  
  if (!D || !isa<VarDecl>(D))
    return;
  
  const VarDecl* VD = dyn_cast<VarDecl>(D);    
  Expr* InitEx = const_cast<Expr*>(VD->getInit());

  // FIXME: static variables may have an initializer, but the second
  //  time a function is called those values may not be current.
  NodeSet Tmp;

  if (InitEx)
    Visit(InitEx, Pred, Tmp);

  if (Tmp.empty())
    Tmp.Add(Pred);
  
  for (NodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {
    const GRState* state = GetState(*I);
    unsigned Count = Builder->getCurrentBlockCount();

    // Check if 'VD' is a VLA and if so check if has a non-zero size.
    QualType T = getContext().getCanonicalType(VD->getType());
    if (VariableArrayType* VLA = dyn_cast<VariableArrayType>(T)) {
      // FIXME: Handle multi-dimensional VLAs.
      
      Expr* SE = VLA->getSizeExpr();
      SVal Size = GetSVal(state, SE);
      
      if (Size.isUndef()) {
        if (NodeTy* N = Builder->generateNode(DS, state, Pred)) {
          N->markAsSink();          
          ExplicitBadSizedVLA.insert(N);
        }
        continue;
      }
      
      bool isFeasibleZero = false;
      const GRState* ZeroSt =  Assume(state, Size, false, isFeasibleZero);
      
      bool isFeasibleNotZero = false;
      state = Assume(state, Size, true, isFeasibleNotZero);
      
      if (isFeasibleZero) {
        if (NodeTy* N = Builder->generateNode(DS, ZeroSt, Pred)) {
          N->markAsSink();          
          if (isFeasibleNotZero) ImplicitBadSizedVLA.insert(N);
          else ExplicitBadSizedVLA.insert(N);
        }
      }
      
      if (!isFeasibleNotZero)
        continue;      
    }
    
    // Decls without InitExpr are not initialized explicitly.
    if (InitEx) {
      SVal InitVal = GetSVal(state, InitEx);
      QualType T = VD->getType();
      
      // Recover some path-sensitivity if a scalar value evaluated to
      // UnknownVal.
      if (InitVal.isUnknown()) {
        if (Loc::IsLocType(T)) {
          SymbolRef Sym = SymMgr.getConjuredSymbol(InitEx, Count);        
          InitVal = loc::SymbolVal(Sym);
        }
        else if (T->isIntegerType() && T->isScalarType()) {
          SymbolRef Sym = SymMgr.getConjuredSymbol(InitEx, Count);        
          InitVal = nonloc::SymbolVal(Sym);                    
        }
      }        
      
      state = StateMgr.BindDecl(state, VD, InitVal);
      
      // The next thing to do is check if the GRTransferFuncs object wants to
      // update the state based on the new binding.  If the GRTransferFunc
      // object doesn't do anything, just auto-propagate the current state.
      GRStmtNodeBuilderRef BuilderRef(Dst, *Builder, *this, *I, state, DS,true);
      getTF().EvalBind(BuilderRef, loc::MemRegionVal(StateMgr.getRegion(VD)),
                       InitVal);      
    } 
    else {
      state = StateMgr.BindDeclWithNoInit(state, VD);
      MakeNode(Dst, DS, *I, state);
    }
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
  QualType T = getContext().getCanonicalType(E->getType());
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

  if (T->isUnionType() || T->isVectorType()) {
    // FIXME: to be implemented.
    // Note: That vectors can return true for T->isIntegerType()
    MakeNode(Dst, E, Pred, state);
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
    if (T == getContext().VoidTy) {          
      // sizeof(void) == 1 byte.
      amt = 1;
    }
    else if (!T.getTypePtr()->isConstantSizeType()) {
      // FIXME: Add support for VLAs.
      return;
    }
    else if (T->isObjCInterfaceType()) {
      // Some code tries to take the sizeof an ObjCInterfaceType, relying that
      // the compiler has laid out its representation.  Just report Unknown
      // for these.      
      return;
    }
    else {
      // All other cases.
      amt = getContext().getTypeSize(T) / 8;
    }    
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
        
        const GRState* state = GetState(*I);
        SVal location = GetSVal(state, Ex);
        
        if (asLValue)
          MakeNode(Dst, U, *I, BindExpr(state, U, location));
        else
          EvalLoad(Dst, U, *I, state, location);
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
        const GRState* state = GetState(*I);
        MakeNode(Dst, U, *I, BindExpr(state, U, GetSVal(state, Ex)));
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
        const GRState* state = GetState(*I);
        SVal X = NonLoc::MakeVal(getBasicVals(), 0, Ex->getType());
        MakeNode(Dst, U, *I, BindExpr(state, U, X));
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
        const GRState* state = GetState(*I);
        MakeNode(Dst, U, *I, BindExpr(state, U, GetSVal(state, Ex)));
      }
      
      return;
    }
    
    case UnaryOperator::AddrOf: {
      
      assert(!asLValue);
      Expr* Ex = U->getSubExpr()->IgnoreParens();
      NodeSet Tmp;
      VisitLValue(Ex, Pred, Tmp);
     
      for (NodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {        
        const GRState* state = GetState(*I);
        SVal V = GetSVal(state, Ex);
        state = BindExpr(state, U, V);
        MakeNode(Dst, U, *I, state);
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
        const GRState* state = GetState(*I);
        
        // Get the value of the subexpression.
        SVal V = GetSVal(state, Ex);

        if (V.isUnknownOrUndef()) {
          MakeNode(Dst, U, *I, BindExpr(state, U, V));
          continue;
        }
        
//        QualType DstT = getContext().getCanonicalType(U->getType());
//        QualType SrcT = getContext().getCanonicalType(Ex->getType());
//        
//        if (DstT != SrcT) // Perform promotions.
//          V = EvalCast(V, DstT); 
//        
//        if (V.isUnknownOrUndef()) {
//          MakeNode(Dst, U, *I, BindExpr(St, U, V));
//          continue;
//        }
        
        switch (U->getOpcode()) {
          default:
            assert(false && "Invalid Opcode.");
            break;
            
          case UnaryOperator::Not:
            // FIXME: Do we need to handle promotions?
            state = BindExpr(state, U, EvalComplement(cast<NonLoc>(V)));
            break;            
            
          case UnaryOperator::Minus:
            // FIXME: Do we need to handle promotions?
            state = BindExpr(state, U, EvalMinus(U, cast<NonLoc>(V)));
            break;   
            
          case UnaryOperator::LNot:   
            
            // C99 6.5.3.3: "The expression !E is equivalent to (0==E)."
            //
            //  Note: technically we do "E == 0", but this is the same in the
            //    transfer functions as "0 == E".
            
            if (isa<Loc>(V)) {
              loc::ConcreteInt X(getBasicVals().getZeroWithPtrWidth());
              SVal Result = EvalBinOp(BinaryOperator::EQ, cast<Loc>(V), X);
              state = BindExpr(state, U, Result);
            }
            else {
              nonloc::ConcreteInt X(getBasicVals().getValue(0, Ex->getType()));
#if 0            
              SVal Result = EvalBinOp(BinaryOperator::EQ, cast<NonLoc>(V), X);
              state = SetSVal(state, U, Result);
#else
              EvalBinOp(Dst, U, BinaryOperator::EQ, cast<NonLoc>(V), X, *I);
              continue;
#endif
            }
            
            break;
        }
        
        MakeNode(Dst, U, *I, state);
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
    
    const GRState* state = GetState(*I);
    SVal V1 = GetSVal(state, Ex);
    
    // Perform a load.      
    NodeSet Tmp2;
    EvalLoad(Tmp2, Ex, *I, state, V1);

    for (NodeSet::iterator I2 = Tmp2.begin(), E2 = Tmp2.end(); I2!=E2; ++I2) {
        
      state = GetState(*I2);
      SVal V2 = GetSVal(state, Ex);
        
      // Propagate unknown and undefined values.      
      if (V2.isUnknownOrUndef()) {
        MakeNode(Dst, U, *I2, BindExpr(state, U, V2));
        continue;
      }
      
      // Handle all other values.
      
      BinaryOperator::Opcode Op = U->isIncrementOp() ? BinaryOperator::Add
                                                     : BinaryOperator::Sub;
      
      SVal Result = EvalBinOp(Op, V2, MakeConstantVal(1U, U));      
      state = BindExpr(state, U, U->isPostfix() ? V2 : Result);

      // Perform the store.      
      EvalStore(Dst, U, *I2, state, V1, Result);
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
    
    const GRState* state = GetState(Pred);
    
    for (AsmStmt::outputs_iterator OI = A->begin_outputs(),
                                   OE = A->end_outputs(); OI != OE; ++OI) {
      
      SVal X = GetSVal(state, *OI);      
      assert (!isa<NonLoc>(X));  // Should be an Lval, or unknown, undef.
      
      if (isa<Loc>(X))
        state = BindLoc(state, cast<Loc>(X), UnknownVal());
    }
    
    MakeNode(Dst, A, Pred, state);
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

  NodeSet Tmp;
  Visit(R, Pred, Tmp);

  for (NodeSet::iterator I = Tmp.begin(), E = Tmp.end(); I != E; ++I) {
    SVal X = GetSVal((*I)->getState(), R);
    
    // Check if we return the address of a stack variable.
    if (isa<loc::MemRegionVal>(X)) {
      // Determine if the value is on the stack.
      const MemRegion* R = cast<loc::MemRegionVal>(&X)->getRegion();
      
      if (R && getStateManager().hasStackStorage(R)) {
        // Create a special node representing the error.
        if (NodeTy* N = Builder->generateNode(S, GetState(*I), *I)) {
          N->markAsSink();
          RetsStackAddr.insert(N);
        }
        continue;
      }
    }
    // Check if we return an undefined value.
    else if (X.isUndef()) {
      if (NodeTy* N = Builder->generateNode(S, GetState(*I), *I)) {
        N->markAsSink();
        RetsUndef.insert(N);
      }
      continue;
    }
    
    EvalReturn(Dst, S, *I);
  }
}

//===----------------------------------------------------------------------===//
// Transfer functions: Binary operators.
//===----------------------------------------------------------------------===//

const GRState* GRExprEngine::CheckDivideZero(Expr* Ex, const GRState* state,
                                             NodeTy* Pred, SVal Denom) {
  
  // Divide by undefined? (potentially zero)
  
  if (Denom.isUndef()) {
    NodeTy* DivUndef = Builder->generateNode(Ex, state, Pred);
    
    if (DivUndef) {
      DivUndef->markAsSink();
      ExplicitBadDivides.insert(DivUndef);
    }
    
    return 0;
  }
  
  // Check for divide/remainder-by-zero.
  // First, "assume" that the denominator is 0 or undefined.            
  
  bool isFeasibleZero = false;
  const GRState* ZeroSt =  Assume(state, Denom, false, isFeasibleZero);
  
  // Second, "assume" that the denominator cannot be 0.            
  
  bool isFeasibleNotZero = false;
  state = Assume(state, Denom, true, isFeasibleNotZero);
  
  // Create the node for the divide-by-zero (if it occurred).
  
  if (isFeasibleZero)
    if (NodeTy* DivZeroNode = Builder->generateNode(Ex, ZeroSt, Pred)) {
      DivZeroNode->markAsSink();
      
      if (isFeasibleNotZero)
        ImplicitBadDivides.insert(DivZeroNode);
      else
        ExplicitBadDivides.insert(DivZeroNode);
      
    }
  
  return isFeasibleNotZero ? state : 0;
}

void GRExprEngine::VisitBinaryOperator(BinaryOperator* B,
                                       GRExprEngine::NodeTy* Pred,
                                       GRExprEngine::NodeSet& Dst) {

  NodeSet Tmp1;
  Expr* LHS = B->getLHS()->IgnoreParens();
  Expr* RHS = B->getRHS()->IgnoreParens();
  
  // FIXME: Add proper support for ObjCKVCRefExpr.
  if (isa<ObjCKVCRefExpr>(LHS)) {
    Visit(RHS, Pred, Dst);   
    return;
  }
  
  
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

      const GRState* state = GetState(*I2);
      const GRState* OldSt = state;

      SVal RightV = GetSVal(state, RHS);
      BinaryOperator::Opcode Op = B->getOpcode();
      
      switch (Op) {
          
        case BinaryOperator::Assign: {
          
          // EXPERIMENTAL: "Conjured" symbols.
          // FIXME: Handle structs.
          QualType T = RHS->getType();
          
          if (RightV.isUnknown() && (Loc::IsLocType(T) || 
                                  (T->isScalarType() && T->isIntegerType()))) {
            unsigned Count = Builder->getCurrentBlockCount();
            SymbolRef Sym = SymMgr.getConjuredSymbol(B->getRHS(), Count);
            
            RightV = Loc::IsLocType(T) 
                   ? cast<SVal>(loc::SymbolVal(Sym)) 
                   : cast<SVal>(nonloc::SymbolVal(Sym));            
          }
          
          // Simulate the effects of a "store":  bind the value of the RHS
          // to the L-Value represented by the LHS.
          
          EvalStore(Dst, B, LHS, *I2, BindExpr(state, B, RightV), LeftV,
                    RightV);
          continue;
        }
          
        case BinaryOperator::Div:
        case BinaryOperator::Rem:
          
          // Special checking for integer denominators.          
          if (RHS->getType()->isIntegerType() && 
              RHS->getType()->isScalarType()) {
            
            state = CheckDivideZero(B, state, *I2, RightV);
            if (!state) continue;
          }
          
          // FALL-THROUGH.

        default: {
      
          if (B->isAssignmentOp())
            break;
          
          // Process non-assignements except commas or short-circuited
          // logical expressions (LAnd and LOr).
          
          SVal Result = EvalBinOp(Op, LeftV, RightV);
          
          if (Result.isUnknown()) {
            if (OldSt != state) {
              // Generate a new node if we have already created a new state.
              MakeNode(Dst, B, *I2, state);
            }
            else
              Dst.Add(*I2);
            
            continue;
          }
          
          if (Result.isUndef() && !LeftV.isUndef() && !RightV.isUndef()) {
            
            // The operands were *not* undefined, but the result is undefined.
            // This is a special node that should be flagged as an error.
            
            if (NodeTy* UndefNode = Builder->generateNode(B, state, *I2)) {
              UndefNode->markAsSink();            
              UndefResults.insert(UndefNode);
            }
            
            continue;
          }
          
          // Otherwise, create a new node.
          
          MakeNode(Dst, B, *I2, BindExpr(state, B, Result));
          continue;
        }
      }
    
      assert (B->isCompoundAssignmentOp());

      switch (Op) {
        default:
          assert(0 && "Invalid opcode for compound assignment.");
        case BinaryOperator::MulAssign: Op = BinaryOperator::Mul; break;
        case BinaryOperator::DivAssign: Op = BinaryOperator::Div; break;
        case BinaryOperator::RemAssign: Op = BinaryOperator::Rem; break;
        case BinaryOperator::AddAssign: Op = BinaryOperator::Add; break;
        case BinaryOperator::SubAssign: Op = BinaryOperator::Sub; break;
        case BinaryOperator::ShlAssign: Op = BinaryOperator::Shl; break;
        case BinaryOperator::ShrAssign: Op = BinaryOperator::Shr; break;
        case BinaryOperator::AndAssign: Op = BinaryOperator::And; break;
        case BinaryOperator::XorAssign: Op = BinaryOperator::Xor; break;
        case BinaryOperator::OrAssign:  Op = BinaryOperator::Or;  break;
      }
          
      // Perform a load (the LHS).  This performs the checks for
      // null dereferences, and so on.
      NodeSet Tmp3;
      SVal location = GetSVal(state, LHS);
      EvalLoad(Tmp3, LHS, *I2, state, location);
      
      for (NodeSet::iterator I3=Tmp3.begin(), E3=Tmp3.end(); I3!=E3; ++I3) {
        
        state = GetState(*I3);
        SVal V = GetSVal(state, LHS);

        // Check for divide-by-zero.
        if ((Op == BinaryOperator::Div || Op == BinaryOperator::Rem)
            && RHS->getType()->isIntegerType()
            && RHS->getType()->isScalarType()) {
          
          // CheckDivideZero returns a new state where the denominator
          // is assumed to be non-zero.
          state = CheckDivideZero(B, state, *I3, RightV);
          
          if (!state)
            continue;
        }
        
        // Propagate undefined values (left-side).          
        if (V.isUndef()) {
          EvalStore(Dst, B, LHS, *I3, BindExpr(state, B, V), location, V);
          continue;
        }
        
        // Propagate unknown values (left and right-side).
        if (RightV.isUnknown() || V.isUnknown()) {
          EvalStore(Dst, B, LHS, *I3, BindExpr(state, B, UnknownVal()),
                    location, UnknownVal());
          continue;
        }

        // At this point:
        //
        //  The LHS is not Undef/Unknown.
        //  The RHS is not Unknown.
        
        // Get the computation type.
        QualType CTy = cast<CompoundAssignOperator>(B)->getComputationType();
        CTy = getContext().getCanonicalType(CTy);
        
        QualType LTy = getContext().getCanonicalType(LHS->getType());
        QualType RTy = getContext().getCanonicalType(RHS->getType());
          
        // Perform promotions.
        if (LTy != CTy) V = EvalCast(V, CTy);
        if (RTy != CTy) RightV = EvalCast(RightV, CTy);
          
        // Evaluate operands and promote to result type.                    
        if (RightV.isUndef()) {            
          // Propagate undefined values (right-side).          
          EvalStore(Dst, B, LHS, *I3, BindExpr(state, B, RightV), location,
                    RightV);
          continue;
        }
      
        // Compute the result of the operation.      
        SVal Result = EvalCast(EvalBinOp(Op, V, RightV), B->getType());
          
        if (Result.isUndef()) {
          // The operands were not undefined, but the result is undefined.
          if (NodeTy* UndefNode = Builder->generateNode(B, state, *I3)) {
            UndefNode->markAsSink();            
            UndefResults.insert(UndefNode);
          }
          continue;
        }

        // EXPERIMENTAL: "Conjured" symbols.
        // FIXME: Handle structs.
        
        SVal LHSVal;
        
        if (Result.isUnknown() && (Loc::IsLocType(CTy) 
                            || (CTy->isScalarType() && CTy->isIntegerType()))) {
          
          unsigned Count = Builder->getCurrentBlockCount();
          
          // The symbolic value is actually for the type of the left-hand side
          // expression, not the computation type, as this is the value the
          // LValue on the LHS will bind to.
          SymbolRef Sym = SymMgr.getConjuredSymbol(B->getRHS(), LTy, Count);
          LHSVal = Loc::IsLocType(LTy) 
                 ? cast<SVal>(loc::SymbolVal(Sym)) 
                 : cast<SVal>(nonloc::SymbolVal(Sym));
          
          // However, we need to convert the symbol to the computation type.
          Result = (LTy == CTy) ? LHSVal : EvalCast(LHSVal,CTy);
        }
        else {
          // The left-hand side may bind to a different value then the
          // computation type.
          LHSVal = (LTy == CTy) ? Result : EvalCast(Result,LTy);
        }
          
        EvalStore(Dst, B, LHS, *I3, BindExpr(state, B, Result), location,
                  LHSVal);
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

void GRExprEngine::EvalBinOp(GRStateSet& OStates, const GRState* state,
                             Expr* Ex, BinaryOperator::Opcode Op,
                             NonLoc L, NonLoc R) {
  
  GRStateSet::AutoPopulate AP(OStates, state);
  if (R.isValid()) getTF().EvalBinOpNN(OStates, *this, state, Ex, Op, L, R);
}

SVal GRExprEngine::EvalBinOp(BinaryOperator::Opcode Op, SVal L, SVal R) {
  
  if (L.isUndef() || R.isUndef())
    return UndefinedVal();
  
  if (L.isUnknown() || R.isUnknown())
    return UnknownVal();
  
  if (isa<Loc>(L)) {
    if (isa<Loc>(R))
      return getTF().EvalBinOp(*this, Op, cast<Loc>(L), cast<Loc>(R));
    else
      return getTF().EvalBinOp(*this, Op, cast<Loc>(L), cast<NonLoc>(R));
  }
  
  if (isa<Loc>(R)) {
    // Support pointer arithmetic where the increment/decrement operand
    // is on the left and the pointer on the right.
    
    assert (Op == BinaryOperator::Add || Op == BinaryOperator::Sub);
    
    // Commute the operands.
    return getTF().EvalBinOp(*this, Op, cast<Loc>(R),
                             cast<NonLoc>(L));
  }
  else
    return getTF().DetermEvalBinOpNN(*this, Op, cast<NonLoc>(L),
                                     cast<NonLoc>(R));
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
        
      default: {
        if (isa<PostStmt>(Loc)) {
          const PostStmt& L = cast<PostStmt>(Loc);        
          Stmt* S = L.getStmt();
          SourceLocation SLoc = S->getLocStart();

          Out << S->getStmtClassName() << ' ' << (void*) S << ' ';        
          llvm::raw_os_ostream OutS(Out);
          S->printPretty(OutS);
          OutS.flush();
          
          if (SLoc.isFileID()) {        
            Out << "\\lline="
              << GraphPrintSourceManager->getInstantiationLineNumber(SLoc)
              << " col="
              << GraphPrintSourceManager->getInstantiationColumnNumber(SLoc)
              << "\\l";
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
              << GraphPrintSourceManager->getInstantiationLineNumber(SLoc)
              << " col="
              << GraphPrintSourceManager->getInstantiationColumnNumber(SLoc);
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
#endif

void GRExprEngine::ViewGraph(bool trim) {
#ifndef NDEBUG  
  if (trim) {
    std::vector<NodeTy*> Src;

    // Flush any outstanding reports to make sure we cover all the nodes.
    // This does not cause them to get displayed.
    for (BugReporter::iterator I=BR.begin(), E=BR.end(); I!=E; ++I)
      const_cast<BugType*>(*I)->FlushReports(BR);

    // Iterate through the reports and get their nodes.
    for (BugReporter::iterator I=BR.begin(), E=BR.end(); I!=E; ++I) {
      for (BugType::const_iterator I2=(*I)->begin(), E2=(*I)->end(); I2!=E2; ++I2) {        
        const BugReportEquivClass& EQ = *I2;
        const BugReport &R = **EQ.begin();
        NodeTy *N = const_cast<NodeTy*>(R.getEndNode());
        if (N) Src.push_back(N);
      }
    }
    
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
    
  std::auto_ptr<GRExprEngine::GraphTy> TrimmedG(G.Trim(Beg, End).first);

  if (!TrimmedG.get())
    llvm::cerr << "warning: Trimmed ExplodedGraph is empty.\n";
  else
    llvm::ViewGraph(*TrimmedG->roots_begin(), "TrimmedGRExprEngine");    
  
  GraphPrintCheckerState = NULL;
  GraphPrintSourceManager = NULL;
#endif
}
