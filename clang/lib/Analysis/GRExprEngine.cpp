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
#include "clang/Analysis/PathSensitive/Checker.h"
#include "clang/AST/ParentMap.h"
#include "clang/AST/StmtObjC.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/PrettyStackTrace.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/ImmutableList.h"
#include "llvm/ADT/StringSwitch.h"

#ifndef NDEBUG
#include "llvm/Support/GraphWriter.h"
#endif

using namespace clang;
using llvm::dyn_cast;
using llvm::cast;
using llvm::APSInt;

//===----------------------------------------------------------------------===//
// Batch auditor.  DEPRECATED.
//===----------------------------------------------------------------------===//

namespace {

class VISIBILITY_HIDDEN MappedBatchAuditor : public GRSimpleAPICheck {
  typedef llvm::ImmutableList<GRSimpleAPICheck*> Checks;
  typedef llvm::DenseMap<void*,Checks> MapTy;

  MapTy M;
  Checks::Factory F;
  Checks AllStmts;

public:
  MappedBatchAuditor(llvm::BumpPtrAllocator& Alloc) :
    F(Alloc), AllStmts(F.GetEmptyList()) {}

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

  void AddCheck(GRSimpleAPICheck *A, Stmt::StmtClass C) {
    assert (A && "Check cannot be null.");
    void* key = reinterpret_cast<void*>((uintptr_t) C);
    MapTy::iterator I = M.find(key);
    M[key] = F.Concat(A, I == M.end() ? F.GetEmptyList() : I->second);
  }

  void AddCheck(GRSimpleAPICheck *A) {
    assert (A && "Check cannot be null.");
    AllStmts = F.Concat(A, AllStmts);
  }

  virtual bool Audit(ExplodedNode* N, GRStateManager& VMgr) {
    // First handle the auditors that accept all statements.
    bool isSink = false;
    for (Checks::iterator I = AllStmts.begin(), E = AllStmts.end(); I!=E; ++I)
      isSink |= (*I)->Audit(N, VMgr);

    // Next handle the auditors that accept only specific statements.
    const Stmt* S = cast<PostStmt>(N->getLocation()).getStmt();
    void* key = reinterpret_cast<void*>((uintptr_t) S->getStmtClass());
    MapTy::iterator MI = M.find(key);
    if (MI != M.end()) {
      for (Checks::iterator I=MI->second.begin(), E=MI->second.end(); I!=E; ++I)
        isSink |= (*I)->Audit(N, VMgr);
    }

    return isSink;
  }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Checker worklist routines.
//===----------------------------------------------------------------------===//

void GRExprEngine::CheckerVisit(Stmt *S, ExplodedNodeSet &Dst,
                                ExplodedNodeSet &Src, bool isPrevisit) {

  if (Checkers.empty()) {
    Dst = Src;
    return;
  }

  ExplodedNodeSet Tmp;
  ExplodedNodeSet *PrevSet = &Src;

  for (CheckersOrdered::iterator I=Checkers.begin(),E=Checkers.end(); I!=E; ++I)
  {
    ExplodedNodeSet *CurrSet = (I+1 == E) ? &Dst 
                                          : (PrevSet == &Tmp) ? &Src : &Tmp;

    CurrSet->clear();
    void *tag = I->first;
    Checker *checker = I->second;

    for (ExplodedNodeSet::iterator NI = PrevSet->begin(), NE = PrevSet->end();
         NI != NE; ++NI)
      checker->GR_Visit(*CurrSet, *Builder, *this, S, *NI, tag, isPrevisit);

    // Update which NodeSet is the current one.
    PrevSet = CurrSet;
  }

  // Don't autotransition.  The CheckerContext objects should do this
  // automatically.
}

//===----------------------------------------------------------------------===//
// Engine construction and deletion.
//===----------------------------------------------------------------------===//

static inline Selector GetNullarySelector(const char* name, ASTContext& Ctx) {
  IdentifierInfo* II = &Ctx.Idents.get(name);
  return Ctx.Selectors.getSelector(0, &II);
}


GRExprEngine::GRExprEngine(AnalysisManager &mgr)
  : AMgr(mgr),
    CoreEngine(mgr.getASTContext(), *this),
    G(CoreEngine.getGraph()),
    Builder(NULL),
    StateMgr(G.getContext(), mgr.getStoreManagerCreator(),
             mgr.getConstraintManagerCreator(), G.getAllocator()),
    SymMgr(StateMgr.getSymbolManager()),
    ValMgr(StateMgr.getValueManager()),
    SVator(ValMgr.getSValuator()),
    CurrentStmt(NULL),
    NSExceptionII(NULL), NSExceptionInstanceRaiseSelectors(NULL),
    RaiseSel(GetNullarySelector("raise", G.getContext())),
    BR(mgr, *this) {}

GRExprEngine::~GRExprEngine() {
  BR.FlushReports();
  delete [] NSExceptionInstanceRaiseSelectors;
  for (CheckersOrdered::iterator I=Checkers.begin(), E=Checkers.end(); I!=E;++I)
    delete I->second;
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

void GRExprEngine::AddCheck(GRSimpleAPICheck *A) {
  if (!BatchAuditor)
    BatchAuditor.reset(new MappedBatchAuditor(getGraph().getAllocator()));

  ((MappedBatchAuditor*) BatchAuditor.get())->AddCheck(A);
}

const GRState* GRExprEngine::getInitialState(const LocationContext *InitLoc) {
  const GRState *state = StateMgr.getInitialState(InitLoc);

  // Preconditions.

  // FIXME: It would be nice if we had a more general mechanism to add
  // such preconditions.  Some day.
  const Decl *D = InitLoc->getDecl();
  
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    // Precondition: the first argument of 'main' is an integer guaranteed
    //  to be > 0.
    if (FD->getIdentifier()->getName() == "main" &&
        FD->getNumParams() > 0) {
      const ParmVarDecl *PD = FD->getParamDecl(0);
      QualType T = PD->getType();
      if (T->isIntegerType())
        if (const MemRegion *R = state->getRegion(PD, InitLoc)) {
          SVal V = state->getSVal(loc::MemRegionVal(R));
          SVal Constraint_untested = EvalBinOp(state, BinaryOperator::GT, V,
                                               ValMgr.makeZeroVal(T),
                                               getContext().IntTy);

          if (DefinedOrUnknownSVal *Constraint =
              dyn_cast<DefinedOrUnknownSVal>(&Constraint_untested)) {
            if (const GRState *newState = state->Assume(*Constraint, true))
              state = newState;
          }
        }
    }
  }
  else if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D)) {    
    // Precondition: 'self' is always non-null upon entry to an Objective-C
    // method.
    const ImplicitParamDecl *SelfD = MD->getSelfDecl();
    const MemRegion *R = state->getRegion(SelfD, InitLoc);
    SVal V = state->getSVal(loc::MemRegionVal(R));
    
    if (const Loc *LV = dyn_cast<Loc>(&V)) {
      // Assume that the pointer value in 'self' is non-null.
      state = state->Assume(*LV, true);
      assert(state && "'self' cannot be null");
    }
  }
  
  return state;
}

//===----------------------------------------------------------------------===//
// Top-level transfer function logic (Dispatcher).
//===----------------------------------------------------------------------===//

void GRExprEngine::ProcessStmt(Stmt* S, GRStmtNodeBuilder& builder) {

  PrettyStackTraceLoc CrashInfo(getContext().getSourceManager(),
                                S->getLocStart(),
                                "Error evaluating statement");

  Builder = &builder;
  EntryNode = builder.getLastNode();

  // FIXME: Consolidate.
  CurrentStmt = S;
  StateMgr.CurrentStmt = S;

  // Set up our simple checks.
  if (BatchAuditor)
    Builder->setAuditor(BatchAuditor.get());

  // Create the cleaned state.
  SymbolReaper SymReaper(Builder->getBasePredecessor()->getLiveVariables(), 
                         SymMgr);
  CleanedState = AMgr.shouldPurgeDead()
    ? StateMgr.RemoveDeadBindings(EntryNode->getState(), CurrentStmt, SymReaper)
    : EntryNode->getState();

  // Process any special transfer function for dead symbols.
  ExplodedNodeSet Tmp;

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

  for (ExplodedNodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {

    ExplodedNodeSet Dst;

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

void GRExprEngine::Visit(Stmt* S, ExplodedNode* Pred, ExplodedNodeSet& Dst) {
  PrettyStackTraceLoc CrashInfo(getContext().getSourceManager(),
                                S->getLocStart(),
                                "Error evaluating statement");

  // FIXME: add metadata to the CFG so that we can disable
  //  this check when we KNOW that there is no block-level subexpression.
  //  The motivation is that this check requires a hashtable lookup.

  if (S != CurrentStmt && Pred->getLocationContext()->getCFG()->isBlkExpr(S)) {
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
        MakeNode(Dst, B, Pred, state->BindExpr(B, state->getSVal(B->getRHS())));
        break;
      }

      if (AMgr.shouldEagerlyAssume() && (B->isRelationalOp() || B->isEqualityOp())) {
        ExplodedNodeSet Tmp;
        VisitBinaryOperator(cast<BinaryOperator>(S), Pred, Tmp, false);
        EvalEagerlyAssume(Dst, Tmp, cast<Expr>(S));
      }
      else
        VisitBinaryOperator(cast<BinaryOperator>(S), Pred, Dst, false);

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
      VisitBinaryOperator(cast<BinaryOperator>(S), Pred, Dst, false);
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
        MakeNode(Dst, SE, Pred, state->BindExpr(SE, state->getSVal(LastExpr)));
      }
      else
        Dst.Add(Pred);

      break;
    }

    case Stmt::StringLiteralClass:
      VisitLValue(cast<StringLiteral>(S), Pred, Dst);
      break;

    case Stmt::UnaryOperatorClass: {
      UnaryOperator *U = cast<UnaryOperator>(S);
      if (AMgr.shouldEagerlyAssume() && (U->getOpcode() == UnaryOperator::LNot)) {
        ExplodedNodeSet Tmp;
        VisitUnaryOperator(U, Pred, Tmp, false);
        EvalEagerlyAssume(Dst, Tmp, U);
      }
      else
        VisitUnaryOperator(U, Pred, Dst, false);
      break;
    }
  }
}

void GRExprEngine::VisitLValue(Expr* Ex, ExplodedNode* Pred,
                               ExplodedNodeSet& Dst) {

  Ex = Ex->IgnoreParens();

  if (Ex != CurrentStmt && Pred->getLocationContext()->getCFG()->isBlkExpr(Ex)) {
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
    case Stmt::ObjCImplicitSetterGetterRefExprClass:
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
      SVal V = state->getLValue(cast<StringLiteral>(Ex));
      MakeNode(Dst, Ex, Pred, state->BindExpr(Ex, V));
      return;
    }

    case Stmt::BinaryOperatorClass:
    case Stmt::CompoundAssignOperatorClass:
      VisitBinaryOperator(cast<BinaryOperator>(Ex), Pred, Dst, true);
      return;

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
// Generic node creation.
//===----------------------------------------------------------------------===//

ExplodedNode* GRExprEngine::MakeNode(ExplodedNodeSet& Dst, Stmt* S,
                                     ExplodedNode* Pred, const GRState* St,
                                     ProgramPoint::Kind K, const void *tag) {
  assert (Builder && "GRStmtNodeBuilder not present.");
  SaveAndRestore<const void*> OldTag(Builder->Tag);
  Builder->Tag = tag;
  return Builder->MakeNode(Dst, S, Pred, St, K);
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

      return state->BindExpr(B, UndefinedVal(Ex));
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

      return state->BindExpr(C, UndefinedVal(Ex));
    }

    case Stmt::ChooseExprClass: { // ?:

      ChooseExpr* C = cast<ChooseExpr>(Terminator);

      Expr* Ex = branchTaken ? C->getLHS() : C->getRHS();
      return state->BindExpr(C, UndefinedVal(Ex));
    }
  }
}

/// RecoverCastedSymbol - A helper function for ProcessBranch that is used
/// to try to recover some path-sensitivity for casts of symbolic
/// integers that promote their values (which are currently not tracked well).
/// This function returns the SVal bound to Condition->IgnoreCasts if all the
//  cast(s) did was sign-extend the original value.
static SVal RecoverCastedSymbol(GRStateManager& StateMgr, const GRState* state,
                                Stmt* Condition, ASTContext& Ctx) {

  Expr *Ex = dyn_cast<Expr>(Condition);
  if (!Ex)
    return UnknownVal();

  uint64_t bits = 0;
  bool bitsInit = false;

  while (CastExpr *CE = dyn_cast<CastExpr>(Ex)) {
    QualType T = CE->getType();

    if (!T->isIntegerType())
      return UnknownVal();

    uint64_t newBits = Ctx.getTypeSize(T);
    if (!bitsInit || newBits < bits) {
      bitsInit = true;
      bits = newBits;
    }

    Ex = CE->getSubExpr();
  }

  // We reached a non-cast.  Is it a symbolic value?
  QualType T = Ex->getType();

  if (!bitsInit || !T->isIntegerType() || Ctx.getTypeSize(T) > bits)
    return UnknownVal();

  return state->getSVal(Ex);
}

void GRExprEngine::ProcessBranch(Stmt* Condition, Stmt* Term,
                                 GRBranchNodeBuilder& builder) {

  // Check for NULL conditions; e.g. "for(;;)"
  if (!Condition) {
    builder.markInfeasible(false);
    return;
  }

  PrettyStackTraceLoc CrashInfo(getContext().getSourceManager(),
                                Condition->getLocStart(),
                                "Error evaluating branch");

  const GRState* PrevState = builder.getState();
  SVal X = PrevState->getSVal(Condition);
  DefinedSVal *V = NULL;
  
  while (true) {
    V = dyn_cast<DefinedSVal>(&X);

    if (!V) {
      if (X.isUnknown()) {
        if (const Expr *Ex = dyn_cast<Expr>(Condition)) {
          if (Ex->getType()->isIntegerType()) {
            // Try to recover some path-sensitivity.  Right now casts of symbolic
            // integers that promote their values are currently not tracked well.
            // If 'Condition' is such an expression, try and recover the
            // underlying value and use that instead.
            SVal recovered = RecoverCastedSymbol(getStateManager(),
                                                 builder.getState(), Condition,
                                                 getContext());

            if (!recovered.isUnknown()) {
              X = recovered;
              continue;
            }
          }
        }    

        builder.generateNode(MarkBranch(PrevState, Term, true), true);
        builder.generateNode(MarkBranch(PrevState, Term, false), false);
        return;
      }

      assert(X.isUndef());
      ExplodedNode *N = builder.generateNode(PrevState, true);

      if (N) {
        N->markAsSink();
        UndefBranches.insert(N);
      }

      builder.markInfeasible(false);
      return;
    }
    
    break;
  }

  // Process the true branch.
  if (builder.isFeasible(true)) {
    if (const GRState *state = PrevState->Assume(*V, true))
      builder.generateNode(MarkBranch(state, Term, true), true);
    else
      builder.markInfeasible(true);
  }

  // Process the false branch.
  if (builder.isFeasible(false)) {
    if (const GRState *state = PrevState->Assume(*V, false))
      builder.generateNode(MarkBranch(state, Term, false), false);
    else
      builder.markInfeasible(false);
  }
}

/// ProcessIndirectGoto - Called by GRCoreEngine.  Used to generate successor
///  nodes by processing the 'effects' of a computed goto jump.
void GRExprEngine::ProcessIndirectGoto(GRIndirectGotoNodeBuilder& builder) {

  const GRState *state = builder.getState();
  SVal V = state->getSVal(builder.getTarget());

  // Three possibilities:
  //
  //   (1) We know the computed label.
  //   (2) The label is NULL (or some other constant), or Undefined.
  //   (3) We have no clue about the label.  Dispatch to all targets.
  //

  typedef GRIndirectGotoNodeBuilder::iterator iterator;

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
    ExplodedNode* N = builder.generateNode(builder.begin(), state, true);
    UndefBranches.insert(N);
    return;
  }

  // This is really a catch-all.  We don't support symbolics yet.
  // FIXME: Implement dispatch for symbolic pointers.

  for (iterator I=builder.begin(), E=builder.end(); I != E; ++I)
    builder.generateNode(I, state);
}


void GRExprEngine::VisitGuardedExpr(Expr* Ex, Expr* L, Expr* R,
                                    ExplodedNode* Pred, ExplodedNodeSet& Dst) {

  assert (Ex == CurrentStmt && Pred->getLocationContext()->getCFG()->isBlkExpr(Ex));

  const GRState* state = GetState(Pred);
  SVal X = state->getSVal(Ex);

  assert (X.isUndef());

  Expr *SE = (Expr*) cast<UndefinedVal>(X).getData();
  assert(SE);
  X = state->getSVal(SE);

  // Make sure that we invalidate the previous binding.
  MakeNode(Dst, Ex, Pred, state->BindExpr(Ex, X, true));
}

/// ProcessSwitch - Called by GRCoreEngine.  Used to generate successor
///  nodes by processing the 'effects' of a switch statement.
void GRExprEngine::ProcessSwitch(GRSwitchNodeBuilder& builder) {
  typedef GRSwitchNodeBuilder::iterator iterator;
  const GRState* state = builder.getState();
  Expr* CondE = builder.getCondition();
  SVal  CondV_untested = state->getSVal(CondE);

  if (CondV_untested.isUndef()) {
    ExplodedNode* N = builder.generateDefaultCaseNode(state, true);
    UndefBranches.insert(N);
    return;
  }
  DefinedOrUnknownSVal CondV = cast<DefinedOrUnknownSVal>(CondV_untested);

  const GRState *DefaultSt = state;
  bool defaultIsFeasible = false;

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
      DefinedOrUnknownSVal Res = SVator.EvalEQ(DefaultSt, CondV, CaseVal);
            
      // Now "assume" that the case matches.
      if (const GRState* stateNew = state->Assume(Res, true)) {
        builder.generateCaseStmtNode(I, stateNew);

        // If CondV evaluates to a constant, then we know that this
        // is the *only* case that we can take, so stop evaluating the
        // others.
        if (isa<nonloc::ConcreteInt>(CondV))
          return;
      }

      // Now "assume" that the case doesn't match.  Add this state
      // to the default state (if it is feasible).
      if (const GRState *stateNew = DefaultSt->Assume(Res, false)) {
        defaultIsFeasible = true;
        DefaultSt = stateNew;
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
  if (defaultIsFeasible) builder.generateDefaultCaseNode(DefaultSt);
}

//===----------------------------------------------------------------------===//
// Transfer functions: logical operations ('&&', '||').
//===----------------------------------------------------------------------===//

void GRExprEngine::VisitLogicalExpr(BinaryOperator* B, ExplodedNode* Pred,
                                    ExplodedNodeSet& Dst) {

  assert(B->getOpcode() == BinaryOperator::LAnd ||
         B->getOpcode() == BinaryOperator::LOr);

  assert(B == CurrentStmt && Pred->getLocationContext()->getCFG()->isBlkExpr(B));

  const GRState* state = GetState(Pred);
  SVal X = state->getSVal(B);
  assert(X.isUndef());

  const Expr *Ex = (const Expr*) cast<UndefinedVal>(X).getData();
  assert(Ex);

  if (Ex == B->getRHS()) {
    X = state->getSVal(Ex);

    // Handle undefined values.
    if (X.isUndef()) {
      MakeNode(Dst, B, Pred, state->BindExpr(B, X));
      return;
    }
    
    DefinedOrUnknownSVal XD = cast<DefinedOrUnknownSVal>(X);

    // We took the RHS.  Because the value of the '&&' or '||' expression must
    // evaluate to 0 or 1, we must assume the value of the RHS evaluates to 0
    // or 1.  Alternatively, we could take a lazy approach, and calculate this
    // value later when necessary.  We don't have the machinery in place for
    // this right now, and since most logical expressions are used for branches,
    // the payoff is not likely to be large.  Instead, we do eager evaluation.
    if (const GRState *newState = state->Assume(XD, true))
      MakeNode(Dst, B, Pred,
               newState->BindExpr(B, ValMgr.makeIntVal(1U, B->getType())));

    if (const GRState *newState = state->Assume(XD, false))
      MakeNode(Dst, B, Pred,
               newState->BindExpr(B, ValMgr.makeIntVal(0U, B->getType())));
  }
  else {
    // We took the LHS expression.  Depending on whether we are '&&' or
    // '||' we know what the value of the expression is via properties of
    // the short-circuiting.
    X = ValMgr.makeIntVal(B->getOpcode() == BinaryOperator::LAnd ? 0U : 1U,
                          B->getType());
    MakeNode(Dst, B, Pred, state->BindExpr(B, X));
  }
}

//===----------------------------------------------------------------------===//
// Transfer functions: Loads and stores.
//===----------------------------------------------------------------------===//

void GRExprEngine::VisitDeclRefExpr(DeclRefExpr *Ex, ExplodedNode *Pred,
                                    ExplodedNodeSet &Dst, bool asLValue) {

  const GRState *state = GetState(Pred);
  const NamedDecl *D = Ex->getDecl();

  if (const VarDecl* VD = dyn_cast<VarDecl>(D)) {

    SVal V = state->getLValue(VD, Pred->getLocationContext());

    if (asLValue)
      MakeNode(Dst, Ex, Pred, state->BindExpr(Ex, V),
               ProgramPoint::PostLValueKind);
    else
      EvalLoad(Dst, Ex, Pred, state, V);
    return;

  } else if (const EnumConstantDecl* ED = dyn_cast<EnumConstantDecl>(D)) {
    assert(!asLValue && "EnumConstantDecl does not have lvalue.");

    SVal V = ValMgr.makeIntVal(ED->getInitVal());
    MakeNode(Dst, Ex, Pred, state->BindExpr(Ex, V));
    return;

  } else if (const FunctionDecl* FD = dyn_cast<FunctionDecl>(D)) {
    // This code is valid regardless of the value of 'isLValue'.
    SVal V = ValMgr.getFunctionPointer(FD);
    MakeNode(Dst, Ex, Pred, state->BindExpr(Ex, V),
             ProgramPoint::PostLValueKind);
    return;
  }

  assert (false &&
          "ValueDecl support for this ValueDecl not implemented.");
}

/// VisitArraySubscriptExpr - Transfer function for array accesses
void GRExprEngine::VisitArraySubscriptExpr(ArraySubscriptExpr* A,
                                           ExplodedNode* Pred,
                                           ExplodedNodeSet& Dst, bool asLValue){

  Expr* Base = A->getBase()->IgnoreParens();
  Expr* Idx  = A->getIdx()->IgnoreParens();
  ExplodedNodeSet Tmp;

  if (Base->getType()->isVectorType()) {
    // For vector types get its lvalue.
    // FIXME: This may not be correct.  Is the rvalue of a vector its location?
    //  In fact, I think this is just a hack.  We need to get the right
    // semantics.
    VisitLValue(Base, Pred, Tmp);
  }
  else
    Visit(Base, Pred, Tmp);   // Get Base's rvalue, which should be an LocVal.

  for (ExplodedNodeSet::iterator I1=Tmp.begin(), E1=Tmp.end(); I1!=E1; ++I1) {
    ExplodedNodeSet Tmp2;
    Visit(Idx, *I1, Tmp2);     // Evaluate the index.

    for (ExplodedNodeSet::iterator I2=Tmp2.begin(),E2=Tmp2.end();I2!=E2; ++I2) {
      const GRState* state = GetState(*I2);
      SVal V = state->getLValue(A->getType(), state->getSVal(Idx),
                                state->getSVal(Base));

      if (asLValue)
        MakeNode(Dst, A, *I2, state->BindExpr(A, V),
                 ProgramPoint::PostLValueKind);
      else
        EvalLoad(Dst, A, *I2, state, V);
    }
  }
}

/// VisitMemberExpr - Transfer function for member expressions.
void GRExprEngine::VisitMemberExpr(MemberExpr* M, ExplodedNode* Pred,
                                   ExplodedNodeSet& Dst, bool asLValue) {

  Expr* Base = M->getBase()->IgnoreParens();
  ExplodedNodeSet Tmp;

  if (M->isArrow())
    Visit(Base, Pred, Tmp);        // p->f = ...  or   ... = p->f
  else
    VisitLValue(Base, Pred, Tmp);  // x.f = ...   or   ... = x.f

  FieldDecl *Field = dyn_cast<FieldDecl>(M->getMemberDecl());
  if (!Field) // FIXME: skipping member expressions for non-fields
    return;

  for (ExplodedNodeSet::iterator I = Tmp.begin(), E = Tmp.end(); I != E; ++I) {
    const GRState* state = GetState(*I);
    // FIXME: Should we insert some assumption logic in here to determine
    // if "Base" is a valid piece of memory?  Before we put this assumption
    // later when using FieldOffset lvals (which we no longer have).
    SVal L = state->getLValue(Field, state->getSVal(Base));

    if (asLValue)
      MakeNode(Dst, M, *I, state->BindExpr(M, L), ProgramPoint::PostLValueKind);
    else
      EvalLoad(Dst, M, *I, state, L);
  }
}

/// EvalBind - Handle the semantics of binding a value to a specific location.
///  This method is used by EvalStore and (soon) VisitDeclStmt, and others.
void GRExprEngine::EvalBind(ExplodedNodeSet& Dst, Expr* Ex, ExplodedNode* Pred,
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
    newState = state->bindLoc(cast<Loc>(location), Val);
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
void GRExprEngine::EvalStore(ExplodedNodeSet& Dst, Expr* Ex, ExplodedNode* Pred,
                             const GRState* state, SVal location, SVal Val,
                             const void *tag) {

  assert (Builder && "GRStmtNodeBuilder must be defined.");

  // Evaluate the location (checks for bad dereferences).
  Pred = EvalLocation(Ex, Pred, state, location, tag);

  if (!Pred)
    return;

  assert (!location.isUndef());
  state = GetState(Pred);

  // Proceed with the store.
  SaveAndRestore<ProgramPoint::Kind> OldSPointKind(Builder->PointKind);
  SaveAndRestore<const void*> OldTag(Builder->Tag);
  Builder->PointKind = ProgramPoint::PostStoreKind;
  Builder->Tag = tag;
  EvalBind(Dst, Ex, Pred, state, location, Val);
}

void GRExprEngine::EvalLoad(ExplodedNodeSet& Dst, Expr* Ex, ExplodedNode* Pred,
                            const GRState* state, SVal location,
                            const void *tag) {

  // Evaluate the location (checks for bad dereferences).
  Pred = EvalLocation(Ex, Pred, state, location, tag);

  if (!Pred)
    return;

  state = GetState(Pred);

  // Proceed with the load.
  ProgramPoint::Kind K = ProgramPoint::PostLoadKind;

  // FIXME: Currently symbolic analysis "generates" new symbols
  //  for the contents of values.  We need a better approach.

  if (location.isUnknown()) {
    // This is important.  We must nuke the old binding.
    MakeNode(Dst, Ex, Pred, state->BindExpr(Ex, UnknownVal()),
             K, tag);
  }
  else {
    SVal V = state->getSVal(cast<Loc>(location), Ex->getType());
    MakeNode(Dst, Ex, Pred, state->BindExpr(Ex, V), K, tag);
  }
}

void GRExprEngine::EvalStore(ExplodedNodeSet& Dst, Expr* Ex, Expr* StoreE,
                             ExplodedNode* Pred, const GRState* state,
                             SVal location, SVal Val, const void *tag) {

  ExplodedNodeSet TmpDst;
  EvalStore(TmpDst, StoreE, Pred, state, location, Val, tag);

  for (ExplodedNodeSet::iterator I=TmpDst.begin(), E=TmpDst.end(); I!=E; ++I)
    MakeNode(Dst, Ex, *I, (*I)->getState(), ProgramPoint::PostStmtKind, tag);
}

ExplodedNode* GRExprEngine::EvalLocation(Stmt* Ex, ExplodedNode* Pred,
                                         const GRState* state, SVal location,
                                         const void *tag) {

  SaveAndRestore<const void*> OldTag(Builder->Tag);
  Builder->Tag = tag;

  if (location.isUnknown() || Checkers.empty())
    return Pred;

  
  for (CheckersOrdered::iterator I=Checkers.begin(), E=Checkers.end(); I!=E;++I)
  {
    Pred = I->second->CheckLocation(Ex, Pred, state, location, *this);
    if (!Pred)
      break;
  }
  
  return Pred;

  // FIXME: Temporarily disable out-of-bounds checking until we make
  // the logic reflect recent changes to CastRegion and friends.
#if 0
  // Check for out-of-bound array access.
  if (isa<loc::MemRegionVal>(LV)) {
    const MemRegion* R = cast<loc::MemRegionVal>(LV).getRegion();
    if (const ElementRegion* ER = dyn_cast<ElementRegion>(R)) {
      // Get the index of the accessed element.
      SVal Idx = ER->getIndex();
      // Get the extent of the array.
      SVal NumElements = getStoreManager().getSizeInElements(StNotNull,
                                                          ER->getSuperRegion());

      const GRState * StInBound = StNotNull->AssumeInBound(Idx, NumElements,
                                                           true);
      const GRState* StOutBound = StNotNull->AssumeInBound(Idx, NumElements,
                                                           false);

      if (StOutBound) {
        // Report warning.  Make sink node manually.
        ExplodedNode* OOBNode =
          Builder->generateNode(Ex, StOutBound, Pred,
                                ProgramPoint::PostOutOfBoundsCheckFailedKind);

        if (OOBNode) {
          OOBNode->markAsSink();

          if (StInBound)
            ImplicitOOBMemAccesses.insert(OOBNode);
          else
            ExplicitOOBMemAccesses.insert(OOBNode);
        }
      }

      if (!StInBound)
        return NULL;

      StNotNull = StInBound;
    }
  }
#endif
}

//===----------------------------------------------------------------------===//
// Transfer function: OSAtomics.
//
// FIXME: Eventually refactor into a more "plugin" infrastructure.
//===----------------------------------------------------------------------===//

// Mac OS X:
// http://developer.apple.com/documentation/Darwin/Reference/Manpages/man3
// atomic.3.html
//
static bool EvalOSAtomicCompareAndSwap(ExplodedNodeSet& Dst,
                                       GRExprEngine& Engine,
                                       GRStmtNodeBuilder& Builder,
                                       CallExpr* CE, SVal L,
                                       ExplodedNode* Pred) {

  // Not enough arguments to match OSAtomicCompareAndSwap?
  if (CE->getNumArgs() != 3)
    return false;

  ASTContext &C = Engine.getContext();
  Expr *oldValueExpr = CE->getArg(0);
  QualType oldValueType = C.getCanonicalType(oldValueExpr->getType());

  Expr *newValueExpr = CE->getArg(1);
  QualType newValueType = C.getCanonicalType(newValueExpr->getType());

  // Do the types of 'oldValue' and 'newValue' match?
  if (oldValueType != newValueType)
    return false;

  Expr *theValueExpr = CE->getArg(2);
  const PointerType *theValueType =
    theValueExpr->getType()->getAs<PointerType>();

  // theValueType not a pointer?
  if (!theValueType)
    return false;

  QualType theValueTypePointee =
    C.getCanonicalType(theValueType->getPointeeType()).getUnqualifiedType();

  // The pointee must match newValueType and oldValueType.
  if (theValueTypePointee != newValueType)
    return false;

  static unsigned magic_load = 0;
  static unsigned magic_store = 0;

  const void *OSAtomicLoadTag = &magic_load;
  const void *OSAtomicStoreTag = &magic_store;

  // Load 'theValue'.
  const GRState *state = Pred->getState();
  ExplodedNodeSet Tmp;
  SVal location = state->getSVal(theValueExpr);
  Engine.EvalLoad(Tmp, theValueExpr, Pred, state, location, OSAtomicLoadTag);

  for (ExplodedNodeSet::iterator I = Tmp.begin(), E = Tmp.end();
       I != E; ++I) {

    ExplodedNode *N = *I;
    const GRState *stateLoad = N->getState();
    SVal theValueVal_untested = stateLoad->getSVal(theValueExpr);
    SVal oldValueVal_untested = stateLoad->getSVal(oldValueExpr);

    // FIXME: Issue an error.
    if (theValueVal_untested.isUndef() || oldValueVal_untested.isUndef()) {
      return false;
    }
    
    DefinedOrUnknownSVal theValueVal =
      cast<DefinedOrUnknownSVal>(theValueVal_untested);
    DefinedOrUnknownSVal oldValueVal =
      cast<DefinedOrUnknownSVal>(oldValueVal_untested);

    SValuator &SVator = Engine.getSValuator();

    // Perform the comparison.
    DefinedOrUnknownSVal Cmp = SVator.EvalEQ(stateLoad, theValueVal,
                                             oldValueVal);

    const GRState *stateEqual = stateLoad->Assume(Cmp, true);

    // Were they equal?
    if (stateEqual) {
      // Perform the store.
      ExplodedNodeSet TmpStore;
      SVal val = stateEqual->getSVal(newValueExpr);

      // Handle implicit value casts.
      if (const TypedRegion *R =
          dyn_cast_or_null<TypedRegion>(location.getAsRegion())) {
        llvm::tie(state, val) = SVator.EvalCast(val, state, R->getValueType(C),
                                                newValueExpr->getType());
      }

      Engine.EvalStore(TmpStore, theValueExpr, N, stateEqual, location,
                       val, OSAtomicStoreTag);

      // Now bind the result of the comparison.
      for (ExplodedNodeSet::iterator I2 = TmpStore.begin(),
           E2 = TmpStore.end(); I2 != E2; ++I2) {
        ExplodedNode *predNew = *I2;
        const GRState *stateNew = predNew->getState();
        SVal Res = Engine.getValueManager().makeTruthVal(true, CE->getType());
        Engine.MakeNode(Dst, CE, predNew, stateNew->BindExpr(CE, Res));
      }
    }

    // Were they not equal?
    if (const GRState *stateNotEqual = stateLoad->Assume(Cmp, false)) {
      SVal Res = Engine.getValueManager().makeTruthVal(false, CE->getType());
      Engine.MakeNode(Dst, CE, N, stateNotEqual->BindExpr(CE, Res));
    }
  }

  return true;
}

static bool EvalOSAtomic(ExplodedNodeSet& Dst,
                         GRExprEngine& Engine,
                         GRStmtNodeBuilder& Builder,
                         CallExpr* CE, SVal L,
                         ExplodedNode* Pred) {
  const FunctionDecl* FD = L.getAsFunctionDecl();
  if (!FD)
    return false;

  const char *FName = FD->getNameAsCString();

  // Check for compare and swap.
  if (strncmp(FName, "OSAtomicCompareAndSwap", 22) == 0 ||
      strncmp(FName, "objc_atomicCompareAndSwap", 25) == 0)
    return EvalOSAtomicCompareAndSwap(Dst, Engine, Builder, CE, L, Pred);

  // FIXME: Other atomics.
  return false;
}

//===----------------------------------------------------------------------===//
// Transfer function: Function calls.
//===----------------------------------------------------------------------===//
static void MarkNoReturnFunction(const FunctionDecl *FD, CallExpr *CE,
                                 const GRState *state,
                                 GRStmtNodeBuilder *Builder) {
  if (!FD)
    return;

  if (FD->getAttr<NoReturnAttr>() ||
      FD->getAttr<AnalyzerNoReturnAttr>())
    Builder->BuildSinks = true;
  else {
    // HACK: Some functions are not marked noreturn, and don't return.
    //  Here are a few hardwired ones.  If this takes too long, we can
    //  potentially cache these results.
    using llvm::StringRef;
    bool BuildSinks
      = llvm::StringSwitch<bool>(StringRef(FD->getIdentifier()->getName()))
          .Case("exit", true)
          .Case("panic", true)
          .Case("error", true)
          .Case("Assert", true)
          // FIXME: This is just a wrapper around throwing an exception.
          //  Eventually inter-procedural analysis should handle this easily.
          .Case("ziperr", true)
          .Case("assfail", true)
          .Case("db_error", true)
          .Case("__assert", true)
          .Case("__assert_rtn", true)
          .Case("__assert_fail", true)
          .Case("dtrace_assfail", true)
          .Case("yy_fatal_error", true)
          .Case("_XCAssertionFailureHandler", true)
          .Case("_DTAssertionFailureHandler", true)
          .Case("_TSAssertionFailureHandler", true)
          .Default(false);
    
    if (BuildSinks)
      Builder->BuildSinks = true;
  }
}

bool GRExprEngine::EvalBuiltinFunction(const FunctionDecl *FD, CallExpr *CE,
                                       ExplodedNode *Pred,
                                       ExplodedNodeSet &Dst) {
  if (!FD)
    return false;

  unsigned id = FD->getBuiltinID();
  if (!id)
    return false;

  const GRState *state = Pred->getState();

  switch (id) {
  case Builtin::BI__builtin_expect: {
    // For __builtin_expect, just return the value of the subexpression.
    assert (CE->arg_begin() != CE->arg_end());
    SVal X = state->getSVal(*(CE->arg_begin()));
    MakeNode(Dst, CE, Pred, state->BindExpr(CE, X));
    return true;
  }

  case Builtin::BI__builtin_alloca: {
    // FIXME: Refactor into StoreManager itself?
    MemRegionManager& RM = getStateManager().getRegionManager();
    const MemRegion* R =
      RM.getAllocaRegion(CE, Builder->getCurrentBlockCount());

    // Set the extent of the region in bytes. This enables us to use the
    // SVal of the argument directly. If we save the extent in bits, we
    // cannot represent values like symbol*8.
    SVal Extent = state->getSVal(*(CE->arg_begin()));
    state = getStoreManager().setExtent(state, R, Extent);
    MakeNode(Dst, CE, Pred, state->BindExpr(CE, loc::MemRegionVal(R)));
    return true;
  }
  }

  return false;
}

void GRExprEngine::EvalCall(ExplodedNodeSet& Dst, CallExpr* CE, SVal L,
                            ExplodedNode* Pred) {
  assert (Builder && "GRStmtNodeBuilder must be defined.");

  // FIXME: Allow us to chain together transfer functions.
  if (EvalOSAtomic(Dst, *this, *Builder, CE, L, Pred))
      return;

  getTF().EvalCall(Dst, *this, *Builder, CE, L, Pred);
}

void GRExprEngine::VisitCall(CallExpr* CE, ExplodedNode* Pred,
                             CallExpr::arg_iterator AI,
                             CallExpr::arg_iterator AE,
                             ExplodedNodeSet& Dst) {
  // Determine the type of function we're calling (if available).
  const FunctionProtoType *Proto = NULL;
  QualType FnType = CE->getCallee()->IgnoreParens()->getType();
  if (const PointerType *FnTypePtr = FnType->getAs<PointerType>())
    Proto = FnTypePtr->getPointeeType()->getAs<FunctionProtoType>();

  VisitCallRec(CE, Pred, AI, AE, Dst, Proto, /*ParamIdx=*/0);
}

void GRExprEngine::VisitCallRec(CallExpr* CE, ExplodedNode* Pred,
                                CallExpr::arg_iterator AI,
                                CallExpr::arg_iterator AE,
                                ExplodedNodeSet& Dst,
                                const FunctionProtoType *Proto,
                                unsigned ParamIdx) {

  // Process the arguments.
  if (AI != AE) {
    // If the call argument is being bound to a reference parameter,
    // visit it as an lvalue, not an rvalue.
    bool VisitAsLvalue = false;
    if (Proto && ParamIdx < Proto->getNumArgs())
      VisitAsLvalue = Proto->getArgType(ParamIdx)->isReferenceType();

    ExplodedNodeSet DstTmp;
    if (VisitAsLvalue)
      VisitLValue(*AI, Pred, DstTmp);
    else
      Visit(*AI, Pred, DstTmp);
    ++AI;

    for (ExplodedNodeSet::iterator DI=DstTmp.begin(), DE=DstTmp.end(); DI != DE;
         ++DI)
      VisitCallRec(CE, *DI, AI, AE, Dst, Proto, ParamIdx + 1);

    return;
  }

  // If we reach here we have processed all of the arguments.  Evaluate
  // the callee expression.
  ExplodedNodeSet DstTmp;
  Expr* Callee = CE->getCallee()->IgnoreParens();

  { // Enter new scope to make the lifetime of 'DstTmp2' bounded.
    ExplodedNodeSet DstTmp2;
    Visit(Callee, Pred, DstTmp2);

    // Perform the previsit of the CallExpr, storing the results in DstTmp.
    CheckerVisit(CE, DstTmp, DstTmp2, true);
  }

  // Finally, evaluate the function call.
  for (ExplodedNodeSet::iterator DI = DstTmp.begin(), DE = DstTmp.end();
       DI != DE; ++DI) {

    const GRState* state = GetState(*DI);
    SVal L = state->getSVal(Callee);

    // FIXME: Add support for symbolic function calls (calls involving
    //  function pointer values that are symbolic).

    // Check for the "noreturn" attribute.

    SaveAndRestore<bool> OldSink(Builder->BuildSinks);
    const FunctionDecl* FD = L.getAsFunctionDecl();

    MarkNoReturnFunction(FD, CE, state, Builder);

    // Evaluate the call.
    if (EvalBuiltinFunction(FD, CE, *DI, Dst))
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

void GRExprEngine::EvalEagerlyAssume(ExplodedNodeSet &Dst, ExplodedNodeSet &Src,
                                     Expr *Ex) {
  for (ExplodedNodeSet::iterator I=Src.begin(), E=Src.end(); I!=E; ++I) {
    ExplodedNode *Pred = *I;

    // Test if the previous node was as the same expression.  This can happen
    // when the expression fails to evaluate to anything meaningful and
    // (as an optimization) we don't generate a node.
    ProgramPoint P = Pred->getLocation();
    if (!isa<PostStmt>(P) || cast<PostStmt>(P).getStmt() != Ex) {
      Dst.Add(Pred);
      continue;
    }

    const GRState* state = Pred->getState();
    SVal V = state->getSVal(Ex);
    if (nonloc::SymExprVal *SEV = dyn_cast<nonloc::SymExprVal>(&V)) {
      // First assume that the condition is true.
      if (const GRState *stateTrue = state->Assume(*SEV, true)) {
        stateTrue = stateTrue->BindExpr(Ex,
                                        ValMgr.makeIntVal(1U, Ex->getType()));
        Dst.Add(Builder->generateNode(PostStmtCustom(Ex,
                                &EagerlyAssumeTag, Pred->getLocationContext()),
                                      stateTrue, Pred));
      }

      // Next, assume that the condition is false.
      if (const GRState *stateFalse = state->Assume(*SEV, false)) {
        stateFalse = stateFalse->BindExpr(Ex,
                                          ValMgr.makeIntVal(0U, Ex->getType()));
        Dst.Add(Builder->generateNode(PostStmtCustom(Ex, &EagerlyAssumeTag,
                                                   Pred->getLocationContext()),
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

void GRExprEngine::VisitObjCIvarRefExpr(ObjCIvarRefExpr* Ex, ExplodedNode* Pred,
                                        ExplodedNodeSet& Dst, bool asLValue) {

  Expr* Base = cast<Expr>(Ex->getBase());
  ExplodedNodeSet Tmp;
  Visit(Base, Pred, Tmp);

  for (ExplodedNodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {
    const GRState* state = GetState(*I);
    SVal BaseVal = state->getSVal(Base);
    SVal location = state->getLValue(Ex->getDecl(), BaseVal);

    if (asLValue)
      MakeNode(Dst, Ex, *I, state->BindExpr(Ex, location));
    else
      EvalLoad(Dst, Ex, *I, state, location);
  }
}

//===----------------------------------------------------------------------===//
// Transfer function: Objective-C fast enumeration 'for' statements.
//===----------------------------------------------------------------------===//

void GRExprEngine::VisitObjCForCollectionStmt(ObjCForCollectionStmt* S,
                                     ExplodedNode* Pred, ExplodedNodeSet& Dst) {

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
    VarDecl* ElemD = cast<VarDecl>(DS->getSingleDecl());
    assert (ElemD->getInit() == 0);
    ElementV = GetState(Pred)->getLValue(ElemD, Pred->getLocationContext());
    VisitObjCForCollectionStmtAux(S, Pred, Dst, ElementV);
    return;
  }

  ExplodedNodeSet Tmp;
  VisitLValue(cast<Expr>(elem), Pred, Tmp);

  for (ExplodedNodeSet::iterator I = Tmp.begin(), E = Tmp.end(); I!=E; ++I) {
    const GRState* state = GetState(*I);
    VisitObjCForCollectionStmtAux(S, *I, Dst, state->getSVal(elem));
  }
}

void GRExprEngine::VisitObjCForCollectionStmtAux(ObjCForCollectionStmt* S,
                                       ExplodedNode* Pred, ExplodedNodeSet& Dst,
                                                 SVal ElementV) {



  // Get the current state.  Use 'EvalLocation' to determine if it is a null
  // pointer, etc.
  Stmt* elem = S->getElement();

  Pred = EvalLocation(elem, Pred, GetState(Pred), ElementV);
  if (!Pred)
    return;

  const GRState *state = GetState(Pred);

  // Handle the case where the container still has elements.
  SVal TrueV = ValMgr.makeTruthVal(1);
  const GRState *hasElems = state->BindExpr(S, TrueV);

  // Handle the case where the container has no elements.
  SVal FalseV = ValMgr.makeTruthVal(0);
  const GRState *noElems = state->BindExpr(S, FalseV);

  if (loc::MemRegionVal* MV = dyn_cast<loc::MemRegionVal>(&ElementV))
    if (const TypedRegion* R = dyn_cast<TypedRegion>(MV->getRegion())) {
      // FIXME: The proper thing to do is to really iterate over the
      //  container.  We will do this with dispatch logic to the store.
      //  For now, just 'conjure' up a symbolic value.
      QualType T = R->getValueType(getContext());
      assert (Loc::IsLocType(T));
      unsigned Count = Builder->getCurrentBlockCount();
      SymbolRef Sym = SymMgr.getConjuredSymbol(elem, T, Count);
      SVal V = ValMgr.makeLoc(Sym);
      hasElems = hasElems->bindLoc(ElementV, V);

      // Bind the location to 'nil' on the false branch.
      SVal nilV = ValMgr.makeIntVal(0, T);
      noElems = noElems->bindLoc(ElementV, nilV);
    }

  // Create the new nodes.
  MakeNode(Dst, S, Pred, hasElems);
  MakeNode(Dst, S, Pred, noElems);
}

//===----------------------------------------------------------------------===//
// Transfer function: Objective-C message expressions.
//===----------------------------------------------------------------------===//

void GRExprEngine::VisitObjCMessageExpr(ObjCMessageExpr* ME, ExplodedNode* Pred,
                                        ExplodedNodeSet& Dst){

  VisitObjCMessageExprArgHelper(ME, ME->arg_begin(), ME->arg_end(),
                                Pred, Dst);
}

void GRExprEngine::VisitObjCMessageExprArgHelper(ObjCMessageExpr* ME,
                                              ObjCMessageExpr::arg_iterator AI,
                                              ObjCMessageExpr::arg_iterator AE,
                                     ExplodedNode* Pred, ExplodedNodeSet& Dst) {
  if (AI == AE) {

    // Process the receiver.

    if (Expr* Receiver = ME->getReceiver()) {
      ExplodedNodeSet Tmp;
      Visit(Receiver, Pred, Tmp);

      for (ExplodedNodeSet::iterator NI = Tmp.begin(), NE = Tmp.end(); NI != NE;
           ++NI)
        VisitObjCMessageExprDispatchHelper(ME, *NI, Dst);

      return;
    }

    VisitObjCMessageExprDispatchHelper(ME, Pred, Dst);
    return;
  }

  ExplodedNodeSet Tmp;
  Visit(*AI, Pred, Tmp);

  ++AI;

  for (ExplodedNodeSet::iterator NI = Tmp.begin(), NE = Tmp.end();NI != NE;++NI)
    VisitObjCMessageExprArgHelper(ME, AI, AE, *NI, Dst);
}

void GRExprEngine::VisitObjCMessageExprDispatchHelper(ObjCMessageExpr* ME,
                                                      ExplodedNode* Pred,
                                                      ExplodedNodeSet& Dst) {

  // FIXME: More logic for the processing the method call.

  const GRState* state = GetState(Pred);
  bool RaisesException = false;


  if (Expr* Receiver = ME->getReceiver()) {

    SVal L_untested = state->getSVal(Receiver);

    // Check for undefined control-flow.
    if (L_untested.isUndef()) {
      ExplodedNode* N = Builder->generateNode(ME, state, Pred);

      if (N) {
        N->markAsSink();
        UndefReceivers.insert(N);
      }

      return;
    }

    // "Assume" that the receiver is not NULL.
    DefinedOrUnknownSVal L = cast<DefinedOrUnknownSVal>(L_untested);
    const GRState *StNotNull = state->Assume(L, true);

    // "Assume" that the receiver is NULL.
    const GRState *StNull = state->Assume(L, false);

    if (StNull) {
      QualType RetTy = ME->getType();

      // Check if the receiver was nil and the return value a struct.
      if (RetTy->isRecordType()) {
        if (Pred->getParentMap().isConsumedExpr(ME)) {
          // The [0 ...] expressions will return garbage.  Flag either an
          // explicit or implicit error.  Because of the structure of this
          // function we currently do not bifurfacte the state graph at
          // this point.
          // FIXME: We should bifurcate and fill the returned struct with
          //  garbage.
          if (ExplodedNode* N = Builder->generateNode(ME, StNull, Pred)) {
            N->markAsSink();
            if (StNotNull)
              NilReceiverStructRetImplicit.insert(N);
            else
              NilReceiverStructRetExplicit.insert(N);
          }
        }
      }
      else {
        ASTContext& Ctx = getContext();
        if (RetTy != Ctx.VoidTy) {
          if (Pred->getParentMap().isConsumedExpr(ME)) {
            // sizeof(void *)
            const uint64_t voidPtrSize = Ctx.getTypeSize(Ctx.VoidPtrTy);
            // sizeof(return type)
            const uint64_t returnTypeSize = Ctx.getTypeSize(ME->getType());

            if (voidPtrSize < returnTypeSize) {
              if (ExplodedNode* N = Builder->generateNode(ME, StNull, Pred)) {
                N->markAsSink();
                if (StNotNull)
                  NilReceiverLargerThanVoidPtrRetImplicit.insert(N);
                else
                  NilReceiverLargerThanVoidPtrRetExplicit.insert(N);
              }
            }
            else if (!StNotNull) {
              // Handle the safe cases where the return value is 0 if the
              // receiver is nil.
              //
              // FIXME: For now take the conservative approach that we only
              // return null values if we *know* that the receiver is nil.
              // This is because we can have surprises like:
              //
              //   ... = [[NSScreens screens] objectAtIndex:0];
              //
              // What can happen is that [... screens] could return nil, but
              // it most likely isn't nil.  We should assume the semantics
              // of this case unless we have *a lot* more knowledge.
              //
              SVal V = ValMgr.makeZeroVal(ME->getType());
              MakeNode(Dst, ME, Pred, StNull->BindExpr(ME, V));
              return;
            }
          }
        }
      }
      // We have handled the cases where the receiver is nil.  The remainder
      // of this method should assume that the receiver is not nil.
      if (!StNotNull)
        return;

      state = StNotNull;
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

    if (state->getSVal(*I).isUndef()) {

      // Generate an error node for passing an uninitialized/undefined value
      // as an argument to a message expression.  This node is a sink.
      ExplodedNode* N = Builder->generateNode(ME, state, Pred);

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

void GRExprEngine::VisitCast(Expr* CastE, Expr* Ex, ExplodedNode* Pred, ExplodedNodeSet& Dst){
  ExplodedNodeSet S1;
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
    for (ExplodedNodeSet::iterator I1 = S1.begin(), E1 = S1.end(); I1 != E1; ++I1)
      Dst.Add(*I1);

    return;
  }

  for (ExplodedNodeSet::iterator I1 = S1.begin(), E1 = S1.end(); I1 != E1; ++I1) {
    ExplodedNode* N = *I1;
    const GRState* state = GetState(N);
    SVal V = state->getSVal(Ex);
    const SValuator::CastResult &Res = SVator.EvalCast(V, state, T, ExTy);
    state = Res.getState()->BindExpr(CastE, Res.getSVal());
    MakeNode(Dst, CastE, N, state);
  }
}

void GRExprEngine::VisitCompoundLiteralExpr(CompoundLiteralExpr* CL,
                                            ExplodedNode* Pred,
                                            ExplodedNodeSet& Dst,
                                            bool asLValue) {
  InitListExpr* ILE = cast<InitListExpr>(CL->getInitializer()->IgnoreParens());
  ExplodedNodeSet Tmp;
  Visit(ILE, Pred, Tmp);

  for (ExplodedNodeSet::iterator I = Tmp.begin(), EI = Tmp.end(); I!=EI; ++I) {
    const GRState* state = GetState(*I);
    SVal ILV = state->getSVal(ILE);
    state = state->bindCompoundLiteral(CL, ILV);

    if (asLValue)
      MakeNode(Dst, CL, *I, state->BindExpr(CL, state->getLValue(CL)));
    else
      MakeNode(Dst, CL, *I, state->BindExpr(CL, ILV));
  }
}

void GRExprEngine::VisitDeclStmt(DeclStmt *DS, ExplodedNode *Pred,
                                 ExplodedNodeSet& Dst) {

  // The CFG has one DeclStmt per Decl.
  Decl* D = *DS->decl_begin();

  if (!D || !isa<VarDecl>(D))
    return;

  const VarDecl* VD = dyn_cast<VarDecl>(D);
  Expr* InitEx = const_cast<Expr*>(VD->getInit());

  // FIXME: static variables may have an initializer, but the second
  //  time a function is called those values may not be current.
  ExplodedNodeSet Tmp;

  if (InitEx)
    Visit(InitEx, Pred, Tmp);
  else
    Tmp.Add(Pred);

  for (ExplodedNodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {
    const GRState* state = GetState(*I);
    unsigned Count = Builder->getCurrentBlockCount();

    // Check if 'VD' is a VLA and if so check if has a non-zero size.
    QualType T = getContext().getCanonicalType(VD->getType());
    if (VariableArrayType* VLA = dyn_cast<VariableArrayType>(T)) {
      // FIXME: Handle multi-dimensional VLAs.

      Expr* SE = VLA->getSizeExpr();
      SVal Size_untested = state->getSVal(SE);

      if (Size_untested.isUndef()) {
        if (ExplodedNode* N = Builder->generateNode(DS, state, Pred)) {
          N->markAsSink();
          ExplicitBadSizedVLA.insert(N);
        }
        continue;
      }

      DefinedOrUnknownSVal Size = cast<DefinedOrUnknownSVal>(Size_untested);
      const GRState *zeroState =  state->Assume(Size, false);
      state = state->Assume(Size, true);

      if (zeroState) {
        if (ExplodedNode* N = Builder->generateNode(DS, zeroState, Pred)) {
          N->markAsSink();
          if (state)
            ImplicitBadSizedVLA.insert(N);
          else
            ExplicitBadSizedVLA.insert(N);
        }
      }

      if (!state)
        continue;
    }

    // Decls without InitExpr are not initialized explicitly.
    const LocationContext *LC = (*I)->getLocationContext();

    if (InitEx) {
      SVal InitVal = state->getSVal(InitEx);
      QualType T = VD->getType();

      // Recover some path-sensitivity if a scalar value evaluated to
      // UnknownVal.
      if (InitVal.isUnknown() ||
          !getConstraintManager().canReasonAbout(InitVal)) {
        InitVal = ValMgr.getConjuredSymbolVal(NULL, InitEx, Count);
      }

      state = state->bindDecl(VD, LC, InitVal);

      // The next thing to do is check if the GRTransferFuncs object wants to
      // update the state based on the new binding.  If the GRTransferFunc
      // object doesn't do anything, just auto-propagate the current state.
      GRStmtNodeBuilderRef BuilderRef(Dst, *Builder, *this, *I, state, DS,true);
      getTF().EvalBind(BuilderRef, loc::MemRegionVal(state->getRegion(VD, LC)),
                       InitVal);
    }
    else {
      state = state->bindDeclWithNoInit(VD, LC);
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
  ExplodedNode* N;
  InitListExpr::reverse_iterator Itr;

  InitListWLItem(ExplodedNode* n, llvm::ImmutableList<SVal> vals,
                 InitListExpr::reverse_iterator itr)
  : Vals(vals), N(n), Itr(itr) {}
};
}


void GRExprEngine::VisitInitListExpr(InitListExpr* E, ExplodedNode* Pred,
                                     ExplodedNodeSet& Dst) {

  const GRState* state = GetState(Pred);
  QualType T = getContext().getCanonicalType(E->getType());
  unsigned NumInitElements = E->getNumInits();

  if (T->isArrayType() || T->isStructureType() ||
      T->isUnionType() || T->isVectorType()) {

    llvm::ImmutableList<SVal> StartVals = getBasicVals().getEmptySValList();

    // Handle base case where the initializer has no elements.
    // e.g: static int* myArray[] = {};
    if (NumInitElements == 0) {
      SVal V = ValMgr.makeCompoundVal(T, StartVals);
      MakeNode(Dst, E, Pred, state->BindExpr(E, V));
      return;
    }

    // Create a worklist to process the initializers.
    llvm::SmallVector<InitListWLItem, 10> WorkList;
    WorkList.reserve(NumInitElements);
    WorkList.push_back(InitListWLItem(Pred, StartVals, E->rbegin()));
    InitListExpr::reverse_iterator ItrEnd = E->rend();
    assert(!(E->rbegin() == E->rend()));

    // Process the worklist until it is empty.
    while (!WorkList.empty()) {
      InitListWLItem X = WorkList.back();
      WorkList.pop_back();

      ExplodedNodeSet Tmp;
      Visit(*X.Itr, X.N, Tmp);

      InitListExpr::reverse_iterator NewItr = X.Itr + 1;

      for (ExplodedNodeSet::iterator NI=Tmp.begin(), NE=Tmp.end(); NI!=NE; ++NI) {
        // Get the last initializer value.
        state = GetState(*NI);
        SVal InitV = state->getSVal(cast<Expr>(*X.Itr));

        // Construct the new list of values by prepending the new value to
        // the already constructed list.
        llvm::ImmutableList<SVal> NewVals =
          getBasicVals().consVals(InitV, X.Vals);

        if (NewItr == ItrEnd) {
          // Now we have a list holding all init values. Make CompoundValData.
          SVal V = ValMgr.makeCompoundVal(T, NewVals);

          // Make final state and node.
          MakeNode(Dst, E, *NI, state->BindExpr(E, V));
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
    ExplodedNodeSet Tmp;
    Expr* Init = E->getInit(0);
    Visit(Init, Pred, Tmp);
    for (ExplodedNodeSet::iterator I = Tmp.begin(), EI = Tmp.end(); I != EI; ++I) {
      state = GetState(*I);
      MakeNode(Dst, E, *I, state->BindExpr(E, state->getSVal(Init)));
    }
    return;
  }


  printf("InitListExpr type = %s\n", T.getAsString().c_str());
  assert(0 && "unprocessed InitListExpr type");
}

/// VisitSizeOfAlignOfExpr - Transfer function for sizeof(type).
void GRExprEngine::VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr* Ex,
                                          ExplodedNode* Pred,
                                          ExplodedNodeSet& Dst) {
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
           GetState(Pred)->BindExpr(Ex, ValMgr.makeIntVal(amt, Ex->getType())));
}


void GRExprEngine::VisitUnaryOperator(UnaryOperator* U, ExplodedNode* Pred,
                                      ExplodedNodeSet& Dst, bool asLValue) {

  switch (U->getOpcode()) {

    default:
      break;

    case UnaryOperator::Deref: {

      Expr* Ex = U->getSubExpr()->IgnoreParens();
      ExplodedNodeSet Tmp;
      Visit(Ex, Pred, Tmp);

      for (ExplodedNodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {

        const GRState* state = GetState(*I);
        SVal location = state->getSVal(Ex);

        if (asLValue)
          MakeNode(Dst, U, *I, state->BindExpr(U, location),
                   ProgramPoint::PostLValueKind);
        else
          EvalLoad(Dst, U, *I, state, location);
      }

      return;
    }

    case UnaryOperator::Real: {

      Expr* Ex = U->getSubExpr()->IgnoreParens();
      ExplodedNodeSet Tmp;
      Visit(Ex, Pred, Tmp);

      for (ExplodedNodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {

        // FIXME: We don't have complex SValues yet.
        if (Ex->getType()->isAnyComplexType()) {
          // Just report "Unknown."
          Dst.Add(*I);
          continue;
        }

        // For all other types, UnaryOperator::Real is an identity operation.
        assert (U->getType() == Ex->getType());
        const GRState* state = GetState(*I);
        MakeNode(Dst, U, *I, state->BindExpr(U, state->getSVal(Ex)));
      }

      return;
    }

    case UnaryOperator::Imag: {

      Expr* Ex = U->getSubExpr()->IgnoreParens();
      ExplodedNodeSet Tmp;
      Visit(Ex, Pred, Tmp);

      for (ExplodedNodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {
        // FIXME: We don't have complex SValues yet.
        if (Ex->getType()->isAnyComplexType()) {
          // Just report "Unknown."
          Dst.Add(*I);
          continue;
        }

        // For all other types, UnaryOperator::Float returns 0.
        assert (Ex->getType()->isIntegerType());
        const GRState* state = GetState(*I);
        SVal X = ValMgr.makeZeroVal(Ex->getType());
        MakeNode(Dst, U, *I, state->BindExpr(U, X));
      }

      return;
    }

    case UnaryOperator::OffsetOf: {
      Expr::EvalResult Res;
      if (U->Evaluate(Res, getContext()) && Res.Val.isInt()) {
        const APSInt &IV = Res.Val.getInt();
        assert(IV.getBitWidth() == getContext().getTypeSize(U->getType()));
        assert(U->getType()->isIntegerType());
        assert(IV.isSigned() == U->getType()->isSignedIntegerType());
        SVal X = ValMgr.makeIntVal(IV);      
        MakeNode(Dst, U, Pred, GetState(Pred)->BindExpr(U, X));
        return;
      }
      // FIXME: Handle the case where __builtin_offsetof is not a constant.
      Dst.Add(Pred);
      return;
    }

    case UnaryOperator::Plus: assert (!asLValue);  // FALL-THROUGH.
    case UnaryOperator::Extension: {

      // Unary "+" is a no-op, similar to a parentheses.  We still have places
      // where it may be a block-level expression, so we need to
      // generate an extra node that just propagates the value of the
      // subexpression.

      Expr* Ex = U->getSubExpr()->IgnoreParens();
      ExplodedNodeSet Tmp;
      Visit(Ex, Pred, Tmp);

      for (ExplodedNodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {
        const GRState* state = GetState(*I);
        MakeNode(Dst, U, *I, state->BindExpr(U, state->getSVal(Ex)));
      }

      return;
    }

    case UnaryOperator::AddrOf: {

      assert(!asLValue);
      Expr* Ex = U->getSubExpr()->IgnoreParens();
      ExplodedNodeSet Tmp;
      VisitLValue(Ex, Pred, Tmp);

      for (ExplodedNodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {
        const GRState* state = GetState(*I);
        SVal V = state->getSVal(Ex);
        state = state->BindExpr(U, V);
        MakeNode(Dst, U, *I, state);
      }

      return;
    }

    case UnaryOperator::LNot:
    case UnaryOperator::Minus:
    case UnaryOperator::Not: {

      assert (!asLValue);
      Expr* Ex = U->getSubExpr()->IgnoreParens();
      ExplodedNodeSet Tmp;
      Visit(Ex, Pred, Tmp);

      for (ExplodedNodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {
        const GRState* state = GetState(*I);

        // Get the value of the subexpression.
        SVal V = state->getSVal(Ex);

        if (V.isUnknownOrUndef()) {
          MakeNode(Dst, U, *I, state->BindExpr(U, V));
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
            state = state->BindExpr(U, EvalComplement(cast<NonLoc>(V)));
            break;

          case UnaryOperator::Minus:
            // FIXME: Do we need to handle promotions?
            state = state->BindExpr(U, EvalMinus(cast<NonLoc>(V)));
            break;

          case UnaryOperator::LNot:

            // C99 6.5.3.3: "The expression !E is equivalent to (0==E)."
            //
            //  Note: technically we do "E == 0", but this is the same in the
            //    transfer functions as "0 == E".
            SVal Result;

            if (isa<Loc>(V)) {
              Loc X = ValMgr.makeNull();
              Result = EvalBinOp(state, BinaryOperator::EQ, cast<Loc>(V), X,
                                 U->getType());
            }
            else {
              nonloc::ConcreteInt X(getBasicVals().getValue(0, Ex->getType()));
              Result = EvalBinOp(state, BinaryOperator::EQ, cast<NonLoc>(V), X,
                                 U->getType());
            }

            state = state->BindExpr(U, Result);

            break;
        }

        MakeNode(Dst, U, *I, state);
      }

      return;
    }
  }

  // Handle ++ and -- (both pre- and post-increment).

  assert (U->isIncrementDecrementOp());
  ExplodedNodeSet Tmp;
  Expr* Ex = U->getSubExpr()->IgnoreParens();
  VisitLValue(Ex, Pred, Tmp);

  for (ExplodedNodeSet::iterator I = Tmp.begin(), E = Tmp.end(); I!=E; ++I) {

    const GRState* state = GetState(*I);
    SVal V1 = state->getSVal(Ex);

    // Perform a load.
    ExplodedNodeSet Tmp2;
    EvalLoad(Tmp2, Ex, *I, state, V1);

    for (ExplodedNodeSet::iterator I2 = Tmp2.begin(), E2 = Tmp2.end(); I2!=E2; ++I2) {

      state = GetState(*I2);
      SVal V2_untested = state->getSVal(Ex);

      // Propagate unknown and undefined values.
      if (V2_untested.isUnknownOrUndef()) {
        MakeNode(Dst, U, *I2, state->BindExpr(U, V2_untested));
        continue;
      }      
      DefinedSVal V2 = cast<DefinedSVal>(V2_untested);

      // Handle all other values.
      BinaryOperator::Opcode Op = U->isIncrementOp() ? BinaryOperator::Add
                                                     : BinaryOperator::Sub;

      // If the UnaryOperator has non-location type, use its type to create the
      // constant value. If the UnaryOperator has location type, create the
      // constant with int type and pointer width.
      SVal RHS;

      if (U->getType()->isAnyPointerType())
        RHS = ValMgr.makeIntValWithPtrWidth(1, false);
      else
        RHS = ValMgr.makeIntVal(1, U->getType());

      SVal Result = EvalBinOp(state, Op, V2, RHS, U->getType());

      // Conjure a new symbol if necessary to recover precision.
      if (Result.isUnknown() || !getConstraintManager().canReasonAbout(Result)){
        DefinedOrUnknownSVal SymVal =
          ValMgr.getConjuredSymbolVal(NULL, Ex,
                                      Builder->getCurrentBlockCount());
        Result = SymVal;

        // If the value is a location, ++/-- should always preserve
        // non-nullness.  Check if the original value was non-null, and if so
        // propagate that constraint.
        if (Loc::IsLocType(U->getType())) {
          DefinedOrUnknownSVal Constraint =
            SVator.EvalEQ(state, V2, ValMgr.makeZeroVal(U->getType()));

          if (!state->Assume(Constraint, true)) {
            // It isn't feasible for the original value to be null.
            // Propagate this constraint.
            Constraint = SVator.EvalEQ(state, SymVal,
                                       ValMgr.makeZeroVal(U->getType()));


            state = state->Assume(Constraint, false);
            assert(state);
          }
        }
      }

      state = state->BindExpr(U, U->isPostfix() ? V2 : Result);

      // Perform the store.
      EvalStore(Dst, U, *I2, state, V1, Result);
    }
  }
}

void GRExprEngine::VisitAsmStmt(AsmStmt* A, ExplodedNode* Pred, ExplodedNodeSet& Dst) {
  VisitAsmStmtHelperOutputs(A, A->begin_outputs(), A->end_outputs(), Pred, Dst);
}

void GRExprEngine::VisitAsmStmtHelperOutputs(AsmStmt* A,
                                             AsmStmt::outputs_iterator I,
                                             AsmStmt::outputs_iterator E,
                                             ExplodedNode* Pred, ExplodedNodeSet& Dst) {
  if (I == E) {
    VisitAsmStmtHelperInputs(A, A->begin_inputs(), A->end_inputs(), Pred, Dst);
    return;
  }

  ExplodedNodeSet Tmp;
  VisitLValue(*I, Pred, Tmp);

  ++I;

  for (ExplodedNodeSet::iterator NI = Tmp.begin(), NE = Tmp.end(); NI != NE; ++NI)
    VisitAsmStmtHelperOutputs(A, I, E, *NI, Dst);
}

void GRExprEngine::VisitAsmStmtHelperInputs(AsmStmt* A,
                                            AsmStmt::inputs_iterator I,
                                            AsmStmt::inputs_iterator E,
                                            ExplodedNode* Pred, ExplodedNodeSet& Dst) {
  if (I == E) {

    // We have processed both the inputs and the outputs.  All of the outputs
    // should evaluate to Locs.  Nuke all of their values.

    // FIXME: Some day in the future it would be nice to allow a "plug-in"
    // which interprets the inline asm and stores proper results in the
    // outputs.

    const GRState* state = GetState(Pred);

    for (AsmStmt::outputs_iterator OI = A->begin_outputs(),
                                   OE = A->end_outputs(); OI != OE; ++OI) {

      SVal X = state->getSVal(*OI);
      assert (!isa<NonLoc>(X));  // Should be an Lval, or unknown, undef.

      if (isa<Loc>(X))
        state = state->bindLoc(cast<Loc>(X), UnknownVal());
    }

    MakeNode(Dst, A, Pred, state);
    return;
  }

  ExplodedNodeSet Tmp;
  Visit(*I, Pred, Tmp);

  ++I;

  for (ExplodedNodeSet::iterator NI = Tmp.begin(), NE = Tmp.end(); NI!=NE; ++NI)
    VisitAsmStmtHelperInputs(A, I, E, *NI, Dst);
}

void GRExprEngine::EvalReturn(ExplodedNodeSet& Dst, ReturnStmt* S,
                              ExplodedNode* Pred) {
  assert (Builder && "GRStmtNodeBuilder must be defined.");

  unsigned size = Dst.size();

  SaveAndRestore<bool> OldSink(Builder->BuildSinks);
  SaveOr OldHasGen(Builder->HasGeneratedNode);

  getTF().EvalReturn(Dst, *this, *Builder, S, Pred);

  // Handle the case where no nodes where generated.

  if (!Builder->BuildSinks && Dst.size() == size && !Builder->HasGeneratedNode)
    MakeNode(Dst, S, Pred, GetState(Pred));
}

void GRExprEngine::VisitReturnStmt(ReturnStmt* S, ExplodedNode* Pred,
                                   ExplodedNodeSet& Dst) {

  Expr* R = S->getRetValue();

  if (!R) {
    EvalReturn(Dst, S, Pred);
    return;
  }

  ExplodedNodeSet Tmp;
  Visit(R, Pred, Tmp);

  for (ExplodedNodeSet::iterator I = Tmp.begin(), E = Tmp.end(); I != E; ++I) {
    SVal X = (*I)->getState()->getSVal(R);

    // Check if we return the address of a stack variable.
    if (isa<loc::MemRegionVal>(X)) {
      // Determine if the value is on the stack.
      const MemRegion* R = cast<loc::MemRegionVal>(&X)->getRegion();

      if (R && R->hasStackStorage()) {
        // Create a special node representing the error.
        if (ExplodedNode* N = Builder->generateNode(S, GetState(*I), *I)) {
          N->markAsSink();
          RetsStackAddr.insert(N);
        }
        continue;
      }
    }
    // Check if we return an undefined value.
    else if (X.isUndef()) {
      if (ExplodedNode* N = Builder->generateNode(S, GetState(*I), *I)) {
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

void GRExprEngine::VisitBinaryOperator(BinaryOperator* B,
                                       ExplodedNode* Pred,
                                       ExplodedNodeSet& Dst, bool asLValue) {

  ExplodedNodeSet Tmp1;
  Expr* LHS = B->getLHS()->IgnoreParens();
  Expr* RHS = B->getRHS()->IgnoreParens();

  // FIXME: Add proper support for ObjCImplicitSetterGetterRefExpr.
  if (isa<ObjCImplicitSetterGetterRefExpr>(LHS)) {
    Visit(RHS, Pred, Dst);
    return;
  }

  if (B->isAssignmentOp())
    VisitLValue(LHS, Pred, Tmp1);
  else
    Visit(LHS, Pred, Tmp1);

  for (ExplodedNodeSet::iterator I1=Tmp1.begin(), E1=Tmp1.end(); I1!=E1; ++I1) {
    SVal LeftV = (*I1)->getState()->getSVal(LHS);
    ExplodedNodeSet Tmp2;
    Visit(RHS, *I1, Tmp2);

    ExplodedNodeSet CheckedSet;
    CheckerVisit(B, CheckedSet, Tmp2, true);

    // With both the LHS and RHS evaluated, process the operation itself.

    for (ExplodedNodeSet::iterator I2=CheckedSet.begin(), E2=CheckedSet.end();
         I2 != E2; ++I2) {

      const GRState *state = GetState(*I2);
      const GRState *OldSt = state;
      SVal RightV = state->getSVal(RHS);

      BinaryOperator::Opcode Op = B->getOpcode();

      if (Op == BinaryOperator::Assign) {
        // EXPERIMENTAL: "Conjured" symbols.
        // FIXME: Handle structs.
        QualType T = RHS->getType();
        
        if ((RightV.isUnknown()||!getConstraintManager().canReasonAbout(RightV))
            && (Loc::IsLocType(T) || (T->isScalarType()&&T->isIntegerType()))) {
          unsigned Count = Builder->getCurrentBlockCount();
          RightV = ValMgr.getConjuredSymbolVal(NULL, B->getRHS(), Count);
        }

        SVal ExprVal = asLValue ? LeftV : RightV;

        // Simulate the effects of a "store":  bind the value of the RHS
        // to the L-Value represented by the LHS.
        EvalStore(Dst, B, LHS, *I2, state->BindExpr(B, ExprVal), LeftV, RightV);
        continue;
      }
      
      if (!B->isAssignmentOp()) {
        // Process non-assignments except commas or short-circuited
        // logical expressions (LAnd and LOr).
        SVal Result = EvalBinOp(state, Op, LeftV, RightV, B->getType());
        
        if (Result.isUnknown()) {
          if (OldSt != state) {
            // Generate a new node if we have already created a new state.
            MakeNode(Dst, B, *I2, state);
          }
          else
            Dst.Add(*I2);
          
          continue;
        }
        
        state = state->BindExpr(B, Result);
        
        if (Result.isUndef()) {
          // The operands were *not* undefined, but the result is undefined.
          // This is a special node that should be flagged as an error.
          if (ExplodedNode *UndefNode = Builder->generateNode(B, state, *I2)){
            UndefNode->markAsSink();
            UndefResults.insert(UndefNode);
          }
          continue;
        }
        
        // Otherwise, create a new node.
        MakeNode(Dst, B, *I2, state);
        continue;
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
      ExplodedNodeSet Tmp3;
      SVal location = state->getSVal(LHS);
      EvalLoad(Tmp3, LHS, *I2, state, location);

      for (ExplodedNodeSet::iterator I3=Tmp3.begin(), E3=Tmp3.end(); I3!=E3;
           ++I3) {
        state = GetState(*I3);
        SVal V = state->getSVal(LHS);

        // Get the computation type.
        QualType CTy =
          cast<CompoundAssignOperator>(B)->getComputationResultType();
        CTy = getContext().getCanonicalType(CTy);

        QualType CLHSTy =
          cast<CompoundAssignOperator>(B)->getComputationLHSType();
        CLHSTy = getContext().getCanonicalType(CLHSTy);

        QualType LTy = getContext().getCanonicalType(LHS->getType());
        QualType RTy = getContext().getCanonicalType(RHS->getType());

        // Promote LHS.
        llvm::tie(state, V) = SVator.EvalCast(V, state, CLHSTy, LTy);

        // Compute the result of the operation.
        SVal Result;
        llvm::tie(state, Result) = SVator.EvalCast(EvalBinOp(state, Op, V,
                                                             RightV, CTy),
                                                   state, B->getType(), CTy);

        if (Result.isUndef()) {
          // The operands were not undefined, but the result is undefined.
          if (ExplodedNode* UndefNode = Builder->generateNode(B, state, *I3)) {
            UndefNode->markAsSink();
            UndefResults.insert(UndefNode);
          }
          continue;
        }

        // EXPERIMENTAL: "Conjured" symbols.
        // FIXME: Handle structs.

        SVal LHSVal;

        if ((Result.isUnknown() ||
             !getConstraintManager().canReasonAbout(Result))
            && (Loc::IsLocType(CTy)
                || (CTy->isScalarType() && CTy->isIntegerType()))) {

          unsigned Count = Builder->getCurrentBlockCount();

          // The symbolic value is actually for the type of the left-hand side
          // expression, not the computation type, as this is the value the
          // LValue on the LHS will bind to.
          LHSVal = ValMgr.getConjuredSymbolVal(NULL, B->getRHS(), LTy, Count);

          // However, we need to convert the symbol to the computation type.
          llvm::tie(state, Result) = SVator.EvalCast(LHSVal, state, CTy, LTy);
        }
        else {
          // The left-hand side may bind to a different value then the
          // computation type.
          llvm::tie(state, LHSVal) = SVator.EvalCast(Result, state, LTy, CTy);
        }

        EvalStore(Dst, B, LHS, *I3, state->BindExpr(B, Result),
                  location, LHSVal);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Checker registration/lookup.
//===----------------------------------------------------------------------===//

Checker *GRExprEngine::lookupChecker(void *tag) const {
  CheckerMap::iterator I = CheckerM.find(tag);
  return (I == CheckerM.end()) ? NULL : Checkers[I->second].second;
}

//===----------------------------------------------------------------------===//
// Visualization.
//===----------------------------------------------------------------------===//

#ifndef NDEBUG
static GRExprEngine* GraphPrintCheckerState;
static SourceManager* GraphPrintSourceManager;

namespace llvm {
template<>
struct VISIBILITY_HIDDEN DOTGraphTraits<ExplodedNode*> :
  public DefaultDOTGraphTraits {
  // FIXME: Since we do not cache error nodes in GRExprEngine now, this does not
  // work.
  static std::string getNodeAttributes(const ExplodedNode* N, void*) {

    if (GraphPrintCheckerState->isImplicitNullDeref(N) ||
        GraphPrintCheckerState->isExplicitNullDeref(N) ||
        GraphPrintCheckerState->isUndefDeref(N) ||
        GraphPrintCheckerState->isUndefStore(N) ||
        GraphPrintCheckerState->isUndefControlFlow(N) ||
        GraphPrintCheckerState->isUndefResult(N) ||
        GraphPrintCheckerState->isBadCall(N) ||
        GraphPrintCheckerState->isUndefArg(N))
      return "color=\"red\",style=\"filled\"";

    if (GraphPrintCheckerState->isNoReturnCall(N))
      return "color=\"blue\",style=\"filled\"";

    return "";
  }

  static std::string getNodeLabel(const ExplodedNode* N, void*,bool ShortNames){

    std::string sbuf;
    llvm::raw_string_ostream Out(sbuf);

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
        if (StmtPoint *L = dyn_cast<StmtPoint>(&Loc)) {
          const Stmt* S = L->getStmt();
          SourceLocation SLoc = S->getLocStart();

          Out << S->getStmtClassName() << ' ' << (void*) S << ' ';
          LangOptions LO; // FIXME.
          S->printPretty(Out, 0, PrintingPolicy(LO));

          if (SLoc.isFileID()) {
            Out << "\\lline="
              << GraphPrintSourceManager->getInstantiationLineNumber(SLoc)
              << " col="
              << GraphPrintSourceManager->getInstantiationColumnNumber(SLoc)
              << "\\l";
          }

          if (isa<PreStmt>(Loc))
            Out << "\\lPreStmt\\l;";
          else if (isa<PostLoad>(Loc))
            Out << "\\lPostLoad\\l;";
          else if (isa<PostStore>(Loc))
            Out << "\\lPostStore\\l";
          else if (isa<PostLValue>(Loc))
            Out << "\\lPostLValue\\l";
          else if (isa<PostLocationChecksSucceed>(Loc))
            Out << "\\lPostLocationChecksSucceed\\l";
          else if (isa<PostNullCheckFailed>(Loc))
            Out << "\\lPostNullCheckFailed\\l";

          if (GraphPrintCheckerState->isImplicitNullDeref(N))
            Out << "\\|Implicit-Null Dereference.\\l";
          else if (GraphPrintCheckerState->isExplicitNullDeref(N))
            Out << "\\|Explicit-Null Dereference.\\l";
          else if (GraphPrintCheckerState->isUndefDeref(N))
            Out << "\\|Dereference of undefialied value.\\l";
          else if (GraphPrintCheckerState->isUndefStore(N))
            Out << "\\|Store to Undefined Loc.";
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
          LangOptions LO; // FIXME.
          E.getSrc()->printTerminator(Out, LO);

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
                LangOptions LO; // FIXME.
                C->getLHS()->printPretty(Out, 0, PrintingPolicy(LO));

                if (Stmt* RHS = C->getRHS()) {
                  Out << " .. ";
                  RHS->printPretty(Out, 0, PrintingPolicy(LO));
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

    const GRState *state = N->getState();
    state->printDOT(Out);

    Out << "\\l";
    return Out.str();
  }
};
} // end llvm namespace
#endif

#ifndef NDEBUG
template <typename ITERATOR>
ExplodedNode* GetGraphNode(ITERATOR I) { return *I; }

template <> ExplodedNode*
GetGraphNode<llvm::DenseMap<ExplodedNode*, Expr*>::iterator>
  (llvm::DenseMap<ExplodedNode*, Expr*>::iterator I) {
  return I->first;
}
#endif

void GRExprEngine::ViewGraph(bool trim) {
#ifndef NDEBUG
  if (trim) {
    std::vector<ExplodedNode*> Src;

    // Flush any outstanding reports to make sure we cover all the nodes.
    // This does not cause them to get displayed.
    for (BugReporter::iterator I=BR.begin(), E=BR.end(); I!=E; ++I)
      const_cast<BugType*>(*I)->FlushReports(BR);

    // Iterate through the reports and get their nodes.
    for (BugReporter::iterator I=BR.begin(), E=BR.end(); I!=E; ++I) {
      for (BugType::const_iterator I2=(*I)->begin(), E2=(*I)->end();
           I2!=E2; ++I2) {
        const BugReportEquivClass& EQ = *I2;
        const BugReport &R = **EQ.begin();
        ExplodedNode *N = const_cast<ExplodedNode*>(R.getEndNode());
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

void GRExprEngine::ViewGraph(ExplodedNode** Beg, ExplodedNode** End) {
#ifndef NDEBUG
  GraphPrintCheckerState = this;
  GraphPrintSourceManager = &getContext().getSourceManager();

  std::auto_ptr<ExplodedGraph> TrimmedG(G.Trim(Beg, End).first);

  if (!TrimmedG.get())
    llvm::errs() << "warning: Trimmed ExplodedGraph is empty.\n";
  else
    llvm::ViewGraph(*TrimmedG->roots_begin(), "TrimmedGRExprEngine");

  GraphPrintCheckerState = NULL;
  GraphPrintSourceManager = NULL;
#endif
}
