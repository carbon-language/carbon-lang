//=-- ExprEngine.cpp - Path-Sensitive Expression-Level Dataflow ---*- C++ -*-=
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

// FIXME: Restructure checker registration.
#include "ExprEngineInternalChecks.h"

#include "clang/StaticAnalyzer/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/PathSensitive/AnalysisManager.h"
#include "clang/StaticAnalyzer/PathSensitive/ExprEngine.h"
#include "clang/StaticAnalyzer/PathSensitive/ExprEngineBuilders.h"
#include "clang/StaticAnalyzer/PathSensitive/Checker.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/ParentMap.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/DeclCXX.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/ImmutableList.h"

#ifndef NDEBUG
#include "llvm/Support/GraphWriter.h"
#endif

using namespace clang;
using namespace ento;
using llvm::dyn_cast;
using llvm::dyn_cast_or_null;
using llvm::cast;
using llvm::APSInt;

namespace {
  // Trait class for recording returned expression in the state.
  struct ReturnExpr {
    static int TagInt;
    typedef const Stmt *data_type;
  };
  int ReturnExpr::TagInt; 
}

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

static inline Selector GetNullarySelector(const char* name, ASTContext& Ctx) {
  IdentifierInfo* II = &Ctx.Idents.get(name);
  return Ctx.Selectors.getSelector(0, &II);
}

//===----------------------------------------------------------------------===//
// Checker worklist routines.
//===----------------------------------------------------------------------===//

void ExprEngine::CheckerVisit(const Stmt *S, ExplodedNodeSet &Dst,
                                ExplodedNodeSet &Src, CallbackKind Kind) {

  // Determine if we already have a cached 'CheckersOrdered' vector
  // specifically tailored for the provided <CallbackKind, Stmt kind>.  This
  // can reduce the number of checkers actually called.
  CheckersOrdered *CO = &Checkers;
  llvm::OwningPtr<CheckersOrdered> NewCO;

  // The cache key is made up of the and the callback kind (pre- or post-visit)
  // and the statement kind.
  CallbackTag K = GetCallbackTag(Kind, S->getStmtClass());

  CheckersOrdered *& CO_Ref = COCache[K];
  
  if (!CO_Ref) {
    // If we have no previously cached CheckersOrdered vector for this
    // statement kind, then create one.
    NewCO.reset(new CheckersOrdered);
  }
  else {
    // Use the already cached set.
    CO = CO_Ref;
  }
  
  if (CO->empty()) {
    // If there are no checkers, return early without doing any
    // more work.
    Dst.insert(Src);
    return;
  }

  ExplodedNodeSet Tmp;
  ExplodedNodeSet *PrevSet = &Src;
  unsigned checkersEvaluated = 0;

  for (CheckersOrdered::iterator I=CO->begin(), E=CO->end(); I!=E; ++I) {
    // If all nodes are sunk, bail out early.
    if (PrevSet->empty())
      break;
    ExplodedNodeSet *CurrSet = 0;
    if (I+1 == E)
      CurrSet = &Dst;
    else {
      CurrSet = (PrevSet == &Tmp) ? &Src : &Tmp;
      CurrSet->clear();
    }
    void *tag = I->first;
    Checker *checker = I->second;
    bool respondsToCallback = true;

    for (ExplodedNodeSet::iterator NI = PrevSet->begin(), NE = PrevSet->end();
         NI != NE; ++NI) {

      checker->GR_Visit(*CurrSet, *Builder, *this, S, *NI, tag,
                        Kind == PreVisitStmtCallback, respondsToCallback);
      
    }

    PrevSet = CurrSet;

    if (NewCO.get()) {
      ++checkersEvaluated;
      if (respondsToCallback)
        NewCO->push_back(*I);
    }    
  }
  
  // If we built NewCO, check if we called all the checkers.  This is important
  // so that we know that we accurately determined the entire set of checkers
  // that responds to this callback.  Note that 'checkersEvaluated' might
  // not be the same as Checkers.size() if one of the Checkers generates
  // a sink node.
  if (NewCO.get() && checkersEvaluated == Checkers.size())
    CO_Ref = NewCO.take();

  // Don't autotransition.  The CheckerContext objects should do this
  // automatically.
}

void ExprEngine::CheckerEvalNilReceiver(const ObjCMessageExpr *ME,
                                          ExplodedNodeSet &Dst,
                                          const GRState *state,
                                          ExplodedNode *Pred) {
  bool evaluated = false;
  ExplodedNodeSet DstTmp;

  for (CheckersOrdered::iterator I=Checkers.begin(),E=Checkers.end();I!=E;++I) {
    void *tag = I->first;
    Checker *checker = I->second;

    if (checker->GR_evalNilReceiver(DstTmp, *Builder, *this, ME, Pred, state,
                                    tag)) {
      evaluated = true;
      break;
    } else
      // The checker didn't evaluate the expr. Restore the Dst.
      DstTmp.clear();
  }

  if (evaluated)
    Dst.insert(DstTmp);
  else
    Dst.insert(Pred);
}

// CheckerEvalCall returns true if one of the checkers processed the node.
// This may return void when all call evaluation logic goes to some checker
// in the future.
bool ExprEngine::CheckerEvalCall(const CallExpr *CE,
                                   ExplodedNodeSet &Dst,
                                   ExplodedNode *Pred) {
  bool evaluated = false;
  ExplodedNodeSet DstTmp;

  for (CheckersOrdered::iterator I=Checkers.begin(),E=Checkers.end();I!=E;++I) {
    void *tag = I->first;
    Checker *checker = I->second;

    if (checker->GR_evalCallExpr(DstTmp, *Builder, *this, CE, Pred, tag)) {
      evaluated = true;
      break;
    } else
      // The checker didn't evaluate the expr. Restore the DstTmp set.
      DstTmp.clear();
  }

  if (evaluated)
    Dst.insert(DstTmp);
  else
    Dst.insert(Pred);

  return evaluated;
}

// FIXME: This is largely copy-paste from CheckerVisit().  Need to
// unify.
void ExprEngine::CheckerVisitBind(const Stmt *StoreE, ExplodedNodeSet &Dst,
                                    ExplodedNodeSet &Src, SVal location,
                                    SVal val, bool isPrevisit) {

  if (Checkers.empty()) {
    Dst.insert(Src);
    return;
  }

  ExplodedNodeSet Tmp;
  ExplodedNodeSet *PrevSet = &Src;

  for (CheckersOrdered::iterator I=Checkers.begin(),E=Checkers.end(); I!=E; ++I)
  {
    ExplodedNodeSet *CurrSet = 0;
    if (I+1 == E)
      CurrSet = &Dst;
    else {
      CurrSet = (PrevSet == &Tmp) ? &Src : &Tmp;
      CurrSet->clear();
    }

    void *tag = I->first;
    Checker *checker = I->second;

    for (ExplodedNodeSet::iterator NI = PrevSet->begin(), NE = PrevSet->end();
         NI != NE; ++NI)
      checker->GR_VisitBind(*CurrSet, *Builder, *this, StoreE,
                            *NI, tag, location, val, isPrevisit);

    // Update which NodeSet is the current one.
    PrevSet = CurrSet;
  }

  // Don't autotransition.  The CheckerContext objects should do this
  // automatically.
}
//===----------------------------------------------------------------------===//
// Engine construction and deletion.
//===----------------------------------------------------------------------===//

static void RegisterInternalChecks(ExprEngine &Eng) {
  // Register internal "built-in" BugTypes with the BugReporter. These BugTypes
  // are different than what probably many checks will do since they don't
  // create BugReports on-the-fly but instead wait until ExprEngine finishes
  // analyzing a function.  Generation of BugReport objects is done via a call
  // to 'FlushReports' from BugReporter.
  // The following checks do not need to have their associated BugTypes
  // explicitly registered with the BugReporter.  If they issue any BugReports,
  // their associated BugType will get registered with the BugReporter
  // automatically.  Note that the check itself is owned by the ExprEngine
  // object.
  RegisterAdjustedReturnValueChecker(Eng);
  // CallAndMessageChecker should be registered before AttrNonNullChecker,
  // where we assume arguments are not undefined.
  RegisterCallAndMessageChecker(Eng);
  RegisterAttrNonNullChecker(Eng);
  RegisterDereferenceChecker(Eng);
  RegisterVLASizeChecker(Eng);
  RegisterDivZeroChecker(Eng);
  RegisterReturnUndefChecker(Eng);
  RegisterUndefinedArraySubscriptChecker(Eng);
  RegisterUndefinedAssignmentChecker(Eng);
  RegisterUndefBranchChecker(Eng);
  RegisterUndefCapturedBlockVarChecker(Eng);
  RegisterUndefResultChecker(Eng);
  RegisterStackAddrLeakChecker(Eng);
  RegisterObjCAtSyncChecker(Eng);
  registerObjCSelfInitChecker(Eng);

  // This is not a checker yet.
  RegisterNoReturnFunctionChecker(Eng);
  RegisterBuiltinFunctionChecker(Eng);
  RegisterOSAtomicChecker(Eng);
  RegisterUnixAPIChecker(Eng);
  RegisterMacOSXAPIChecker(Eng);
}

ExprEngine::ExprEngine(AnalysisManager &mgr, TransferFuncs *tf)
  : AMgr(mgr),
    Engine(*this),
    G(Engine.getGraph()),
    Builder(NULL),
    StateMgr(getContext(), mgr.getStoreManagerCreator(),
             mgr.getConstraintManagerCreator(), G.getAllocator(),
             *this),
    SymMgr(StateMgr.getSymbolManager()),
    svalBuilder(StateMgr.getSValBuilder()),
    EntryNode(NULL), currentStmt(NULL),
    NSExceptionII(NULL), NSExceptionInstanceRaiseSelectors(NULL),
    RaiseSel(GetNullarySelector("raise", getContext())),
    BR(mgr, *this), TF(tf) {
  // Register internal checks.
  RegisterInternalChecks(*this);

  // FIXME: Eventually remove the TF object entirely.
  TF->RegisterChecks(*this);
  TF->RegisterPrinters(getStateManager().Printers);
}

ExprEngine::~ExprEngine() {
  BR.FlushReports();
  delete [] NSExceptionInstanceRaiseSelectors;
  
  // Delete the set of checkers.
  for (CheckersOrdered::iterator I=Checkers.begin(), E=Checkers.end(); I!=E;++I)
    delete I->second;
  
  for (CheckersOrderedCache::iterator I=COCache.begin(), E=COCache.end();
       I!=E;++I)
    delete I->second;
}

//===----------------------------------------------------------------------===//
// Utility methods.
//===----------------------------------------------------------------------===//

const GRState* ExprEngine::getInitialState(const LocationContext *InitLoc) {
  const GRState *state = StateMgr.getInitialState(InitLoc);

  // Preconditions.

  // FIXME: It would be nice if we had a more general mechanism to add
  // such preconditions.  Some day.
  do {
    const Decl *D = InitLoc->getDecl();
    if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
      // Precondition: the first argument of 'main' is an integer guaranteed
      //  to be > 0.
      const IdentifierInfo *II = FD->getIdentifier();
      if (!II || !(II->getName() == "main" && FD->getNumParams() > 0))
        break;

      const ParmVarDecl *PD = FD->getParamDecl(0);
      QualType T = PD->getType();
      if (!T->isIntegerType())
        break;

      const MemRegion *R = state->getRegion(PD, InitLoc);
      if (!R)
        break;

      SVal V = state->getSVal(loc::MemRegionVal(R));
      SVal Constraint_untested = evalBinOp(state, BO_GT, V,
                                           svalBuilder.makeZeroVal(T),
                                           getContext().IntTy);

      DefinedOrUnknownSVal *Constraint =
        dyn_cast<DefinedOrUnknownSVal>(&Constraint_untested);

      if (!Constraint)
        break;

      if (const GRState *newState = state->assume(*Constraint, true))
        state = newState;

      break;
    }

    if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D)) {
      // Precondition: 'self' is always non-null upon entry to an Objective-C
      // method.
      const ImplicitParamDecl *SelfD = MD->getSelfDecl();
      const MemRegion *R = state->getRegion(SelfD, InitLoc);
      SVal V = state->getSVal(loc::MemRegionVal(R));

      if (const Loc *LV = dyn_cast<Loc>(&V)) {
        // Assume that the pointer value in 'self' is non-null.
        state = state->assume(*LV, true);
        assert(state && "'self' cannot be null");
      }
    }
  } while (0);

  return state;
}

//===----------------------------------------------------------------------===//
// Top-level transfer function logic (Dispatcher).
//===----------------------------------------------------------------------===//

/// evalAssume - Called by ConstraintManager. Used to call checker-specific
///  logic for handling assumptions on symbolic values.
const GRState *ExprEngine::processAssume(const GRState *state, SVal cond,
                                           bool assumption) {
  // Determine if we already have a cached 'CheckersOrdered' vector
  // specifically tailored for processing assumptions.  This
  // can reduce the number of checkers actually called.
  CheckersOrdered *CO = &Checkers;
  llvm::OwningPtr<CheckersOrdered> NewCO;

  CallbackTag K = GetCallbackTag(processAssumeCallback);
  CheckersOrdered *& CO_Ref = COCache[K];

  if (!CO_Ref) {
    // If we have no previously cached CheckersOrdered vector for this
    // statement kind, then create one.
    NewCO.reset(new CheckersOrdered);
  }
  else {
    // Use the already cached set.
    CO = CO_Ref;
  }

  if (!CO->empty()) {
    // Let the checkers have a crack at the assume before the transfer functions
    // get their turn.
    for (CheckersOrdered::iterator I = CO->begin(), E = CO->end(); I!=E; ++I) {

      // If any checker declares the state infeasible (or if it starts that
      // way), bail out.
      if (!state)
        return NULL;

      Checker *C = I->second;
      bool respondsToCallback = true;

      state = C->evalAssume(state, cond, assumption, &respondsToCallback);

      // Check if we're building the cache of checkers that care about
      // assumptions.
      if (NewCO.get() && respondsToCallback)
        NewCO->push_back(*I);
    }

    // If we got through all the checkers, and we built a list of those that
    // care about assumptions, save it.
    if (NewCO.get())
      CO_Ref = NewCO.take();
  }

  // If the state is infeasible at this point, bail out.
  if (!state)
    return NULL;

  return TF->evalAssume(state, cond, assumption);
}

bool ExprEngine::wantsRegionChangeUpdate(const GRState* state) {
  CallbackTag K = GetCallbackTag(EvalRegionChangesCallback);
  CheckersOrdered *CO = COCache[K];

  if (!CO)
    CO = &Checkers;

  for (CheckersOrdered::iterator I = CO->begin(), E = CO->end(); I != E; ++I) {
    Checker *C = I->second;
    if (C->wantsRegionChangeUpdate(state))
      return true;
  }

  return false;
}

const GRState *
ExprEngine::processRegionChanges(const GRState *state,
                                   const MemRegion * const *Begin,
                                   const MemRegion * const *End) {
  // FIXME: Most of this method is copy-pasted from processAssume.

  // Determine if we already have a cached 'CheckersOrdered' vector
  // specifically tailored for processing region changes.  This
  // can reduce the number of checkers actually called.
  CheckersOrdered *CO = &Checkers;
  llvm::OwningPtr<CheckersOrdered> NewCO;

  CallbackTag K = GetCallbackTag(EvalRegionChangesCallback);
  CheckersOrdered *& CO_Ref = COCache[K];

  if (!CO_Ref) {
    // If we have no previously cached CheckersOrdered vector for this
    // callback, then create one.
    NewCO.reset(new CheckersOrdered);
  }
  else {
    // Use the already cached set.
    CO = CO_Ref;
  }

  // If there are no checkers, just return the state as is.
  if (CO->empty())
    return state;

  for (CheckersOrdered::iterator I = CO->begin(), E = CO->end(); I != E; ++I) {
    // If any checker declares the state infeasible (or if it starts that way),
    // bail out.
    if (!state)
      return NULL;

    Checker *C = I->second;
    bool respondsToCallback = true;

    state = C->EvalRegionChanges(state, Begin, End, &respondsToCallback);

    // See if we're building a cache of checkers that care about region changes.
    if (NewCO.get() && respondsToCallback)
      NewCO->push_back(*I);
  }

  // If we got through all the checkers, and we built a list of those that
  // care about region changes, save it.
  if (NewCO.get())
    CO_Ref = NewCO.take();

  return state;
}

void ExprEngine::processEndWorklist(bool hasWorkRemaining) {
  for (CheckersOrdered::iterator I = Checkers.begin(), E = Checkers.end();
       I != E; ++I) {
    I->second->VisitEndAnalysis(G, BR, *this);
  }
}

void ExprEngine::processCFGElement(const CFGElement E, 
                                  StmtNodeBuilder& builder) {
  switch (E.getKind()) {
  case CFGElement::Statement:
    ProcessStmt(E.getAs<CFGStmt>(), builder);
    break;
  case CFGElement::Initializer:
    ProcessInitializer(E.getAs<CFGInitializer>(), builder);
    break;
  case CFGElement::ImplicitDtor:
    ProcessImplicitDtor(E.getAs<CFGImplicitDtor>(), builder);
    break;
  default:
    // Suppress compiler warning.
    llvm_unreachable("Unexpected CFGElement kind.");
  }
}

void ExprEngine::ProcessStmt(const CFGStmt S, StmtNodeBuilder& builder) {
  currentStmt = S.getStmt();
  PrettyStackTraceLoc CrashInfo(getContext().getSourceManager(),
                                currentStmt->getLocStart(),
                                "Error evaluating statement");

  Builder = &builder;
  EntryNode = builder.getPredecessor();

  // Create the cleaned state.
  const LocationContext *LC = EntryNode->getLocationContext();
  SymbolReaper SymReaper(LC, currentStmt, SymMgr);

  if (AMgr.shouldPurgeDead()) {
    const GRState *St = EntryNode->getState();

    for (CheckersOrdered::iterator I = Checkers.begin(), E = Checkers.end();
         I != E; ++I) {
      Checker *checker = I->second;
      checker->MarkLiveSymbols(St, SymReaper);
    }

    const StackFrameContext *SFC = LC->getCurrentStackFrame();
    CleanedState = StateMgr.RemoveDeadBindings(St, SFC, SymReaper);
  } else {
    CleanedState = EntryNode->getState();
  }

  // Process any special transfer function for dead symbols.
  ExplodedNodeSet Tmp;

  if (!SymReaper.hasDeadSymbols())
    Tmp.Add(EntryNode);
  else {
    SaveAndRestore<bool> OldSink(Builder->BuildSinks);
    SaveOr OldHasGen(Builder->hasGeneratedNode);

    SaveAndRestore<bool> OldPurgeDeadSymbols(Builder->PurgingDeadSymbols);
    Builder->PurgingDeadSymbols = true;

    // FIXME: This should soon be removed.
    ExplodedNodeSet Tmp2;
    getTF().evalDeadSymbols(Tmp2, *this, *Builder, EntryNode,
                            CleanedState, SymReaper);

    if (Checkers.empty())
      Tmp.insert(Tmp2);
    else {
      ExplodedNodeSet Tmp3;
      ExplodedNodeSet *SrcSet = &Tmp2;
      for (CheckersOrdered::iterator I = Checkers.begin(), E = Checkers.end();
           I != E; ++I) {
        ExplodedNodeSet *DstSet = 0;
        if (I+1 == E)
          DstSet = &Tmp;
        else {
          DstSet = (SrcSet == &Tmp2) ? &Tmp3 : &Tmp2;
          DstSet->clear();
        }

        void *tag = I->first;
        Checker *checker = I->second;
        for (ExplodedNodeSet::iterator NI = SrcSet->begin(), NE = SrcSet->end();
             NI != NE; ++NI)
          checker->GR_evalDeadSymbols(*DstSet, *Builder, *this, currentStmt,
                                      *NI, SymReaper, tag);
        SrcSet = DstSet;
      }
    }

    if (!Builder->BuildSinks && !Builder->hasGeneratedNode)
      Tmp.Add(EntryNode);
  }

  bool HasAutoGenerated = false;

  for (ExplodedNodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {
    ExplodedNodeSet Dst;

    // Set the cleaned state.
    Builder->SetCleanedState(*I == EntryNode ? CleanedState : GetState(*I));

    // Visit the statement.
    Visit(currentStmt, *I, Dst);

    // Do we need to auto-generate a node?  We only need to do this to generate
    // a node with a "cleaned" state; CoreEngine will actually handle
    // auto-transitions for other cases.
    if (Dst.size() == 1 && *Dst.begin() == EntryNode
        && !Builder->hasGeneratedNode && !HasAutoGenerated) {
      HasAutoGenerated = true;
      builder.generateNode(currentStmt, GetState(EntryNode), *I);
    }
  }

  // NULL out these variables to cleanup.
  CleanedState = NULL;
  EntryNode = NULL;

  currentStmt = 0;

  Builder = NULL;
}

void ExprEngine::ProcessInitializer(const CFGInitializer Init,
                                    StmtNodeBuilder &builder) {
  // We don't set EntryNode and currentStmt. And we don't clean up state.
  const CXXCtorInitializer *BMI = Init.getInitializer();

  ExplodedNode *pred = builder.getPredecessor();

  const StackFrameContext *stackFrame = cast<StackFrameContext>(pred->getLocationContext());
  const CXXConstructorDecl *decl = cast<CXXConstructorDecl>(stackFrame->getDecl());
  const CXXThisRegion *thisReg = getCXXThisRegion(decl, stackFrame);

  SVal thisVal = pred->getState()->getSVal(thisReg);

  if (BMI->isAnyMemberInitializer()) {
    ExplodedNodeSet Dst;

    // Evaluate the initializer.
    Visit(BMI->getInit(), pred, Dst);

    for (ExplodedNodeSet::iterator I = Dst.begin(), E = Dst.end(); I != E; ++I){
      ExplodedNode *Pred = *I;
      const GRState *state = Pred->getState();

      const FieldDecl *FD = BMI->getAnyMember();

      SVal FieldLoc = state->getLValue(FD, thisVal);
      SVal InitVal = state->getSVal(BMI->getInit());
      state = state->bindLoc(FieldLoc, InitVal);

      // Use a custom node building process.
      PostInitializer PP(BMI, stackFrame);
      // Builder automatically add the generated node to the deferred set,
      // which are processed in the builder's dtor.
      builder.generateNode(PP, state, Pred);
    }
    return;
  }

  assert(BMI->isBaseInitializer());

  // Get the base class declaration.
  const CXXConstructExpr *ctorExpr = cast<CXXConstructExpr>(BMI->getInit());

  // Create the base object region.
  SVal baseVal = 
    getStoreManager().evalDerivedToBase(thisVal, ctorExpr->getType());
  const MemRegion *baseReg = baseVal.getAsRegion();
  assert(baseReg);
  Builder = &builder;
  ExplodedNodeSet dst;
  VisitCXXConstructExpr(ctorExpr, baseReg, pred, dst);
}

void ExprEngine::ProcessImplicitDtor(const CFGImplicitDtor D,
                                       StmtNodeBuilder &builder) {
  Builder = &builder;

  switch (D.getDtorKind()) {
  case CFGElement::AutomaticObjectDtor:
    ProcessAutomaticObjDtor(cast<CFGAutomaticObjDtor>(D), builder);
    break;
  case CFGElement::BaseDtor:
    ProcessBaseDtor(cast<CFGBaseDtor>(D), builder);
    break;
  case CFGElement::MemberDtor:
    ProcessMemberDtor(cast<CFGMemberDtor>(D), builder);
    break;
  case CFGElement::TemporaryDtor:
    ProcessTemporaryDtor(cast<CFGTemporaryDtor>(D), builder);
    break;
  default:
    llvm_unreachable("Unexpected dtor kind.");
  }
}

void ExprEngine::ProcessAutomaticObjDtor(const CFGAutomaticObjDtor dtor,
                                           StmtNodeBuilder &builder) {
  ExplodedNode *pred = builder.getPredecessor();
  const GRState *state = pred->getState();
  const VarDecl *varDecl = dtor.getVarDecl();

  QualType varType = varDecl->getType();

  if (const ReferenceType *refType = varType->getAs<ReferenceType>())
    varType = refType->getPointeeType();

  const CXXRecordDecl *recordDecl = varType->getAsCXXRecordDecl();
  assert(recordDecl && "get CXXRecordDecl fail");
  const CXXDestructorDecl *dtorDecl = recordDecl->getDestructor();

  Loc dest = state->getLValue(varDecl, pred->getLocationContext());

  ExplodedNodeSet dstSet;
  VisitCXXDestructor(dtorDecl, cast<loc::MemRegionVal>(dest).getRegion(),
                     dtor.getTriggerStmt(), pred, dstSet);
}

void ExprEngine::ProcessBaseDtor(const CFGBaseDtor D,
                                   StmtNodeBuilder &builder) {
}

void ExprEngine::ProcessMemberDtor(const CFGMemberDtor D,
                                     StmtNodeBuilder &builder) {
}

void ExprEngine::ProcessTemporaryDtor(const CFGTemporaryDtor D,
                                        StmtNodeBuilder &builder) {
}

void ExprEngine::Visit(const Stmt* S, ExplodedNode* Pred, 
                         ExplodedNodeSet& Dst) {
  PrettyStackTraceLoc CrashInfo(getContext().getSourceManager(),
                                S->getLocStart(),
                                "Error evaluating statement");

  // Expressions to ignore.
  if (const Expr *Ex = dyn_cast<Expr>(S))
    S = Ex->IgnoreParens();
  
  // FIXME: add metadata to the CFG so that we can disable
  //  this check when we KNOW that there is no block-level subexpression.
  //  The motivation is that this check requires a hashtable lookup.

  if (S != currentStmt && Pred->getLocationContext()->getCFG()->isBlkExpr(S)) {
    Dst.Add(Pred);
    return;
  }

  switch (S->getStmtClass()) {
    // C++ stuff we don't support yet.
    case Stmt::CXXBindTemporaryExprClass:
    case Stmt::CXXCatchStmtClass:
    case Stmt::CXXDefaultArgExprClass:
    case Stmt::CXXDependentScopeMemberExprClass:
    case Stmt::ExprWithCleanupsClass:
    case Stmt::CXXNullPtrLiteralExprClass:
    case Stmt::CXXPseudoDestructorExprClass:
    case Stmt::CXXTemporaryObjectExprClass:
    case Stmt::CXXThrowExprClass:
    case Stmt::CXXTryStmtClass:
    case Stmt::CXXTypeidExprClass:
    case Stmt::CXXUuidofExprClass:
    case Stmt::CXXUnresolvedConstructExprClass:
    case Stmt::CXXScalarValueInitExprClass:
    case Stmt::DependentScopeDeclRefExprClass:
    case Stmt::UnaryTypeTraitExprClass:
    case Stmt::BinaryTypeTraitExprClass:
    case Stmt::UnresolvedLookupExprClass:
    case Stmt::UnresolvedMemberExprClass:
    case Stmt::CXXNoexceptExprClass:
    case Stmt::PackExpansionExprClass:
    {
      SaveAndRestore<bool> OldSink(Builder->BuildSinks);
      Builder->BuildSinks = true;
      MakeNode(Dst, S, Pred, GetState(Pred));
      break;
    }
      
    case Stmt::ParenExprClass:
      llvm_unreachable("ParenExprs already handled.");
    // Cases that should never be evaluated simply because they shouldn't
    // appear in the CFG.
    case Stmt::BreakStmtClass:
    case Stmt::CaseStmtClass:
    case Stmt::CompoundStmtClass:
    case Stmt::ContinueStmtClass:
    case Stmt::DefaultStmtClass:
    case Stmt::DoStmtClass:
    case Stmt::GotoStmtClass:
    case Stmt::IndirectGotoStmtClass:
    case Stmt::LabelStmtClass:
    case Stmt::NoStmtClass:
    case Stmt::NullStmtClass:
    case Stmt::SwitchCaseClass:
    case Stmt::OpaqueValueExprClass:
      llvm_unreachable("Stmt should not be in analyzer evaluation loop");
      break;

    case Stmt::GNUNullExprClass: {
      MakeNode(Dst, S, Pred, GetState(Pred)->BindExpr(S, svalBuilder.makeNull()));
      break;
    }

    case Stmt::ObjCAtSynchronizedStmtClass:
      VisitObjCAtSynchronizedStmt(cast<ObjCAtSynchronizedStmt>(S), Pred, Dst);
      break;

    // Cases not handled yet; but will handle some day.
    case Stmt::DesignatedInitExprClass:
    case Stmt::ExtVectorElementExprClass:
    case Stmt::ImaginaryLiteralClass:
    case Stmt::ImplicitValueInitExprClass:
    case Stmt::ObjCAtCatchStmtClass:
    case Stmt::ObjCAtFinallyStmtClass:
    case Stmt::ObjCAtTryStmtClass:
    case Stmt::ObjCEncodeExprClass:
    case Stmt::ObjCIsaExprClass:
    case Stmt::ObjCPropertyRefExprClass:
    case Stmt::ObjCProtocolExprClass:
    case Stmt::ObjCSelectorExprClass:
    case Stmt::ObjCStringLiteralClass:
    case Stmt::ParenListExprClass:
    case Stmt::PredefinedExprClass:
    case Stmt::ShuffleVectorExprClass:
    case Stmt::VAArgExprClass:
        // Fall through.

    // Cases we intentionally don't evaluate, since they don't need
    // to be explicitly evaluated.
    case Stmt::AddrLabelExprClass:
    case Stmt::IntegerLiteralClass:
    case Stmt::CharacterLiteralClass:
    case Stmt::CXXBoolLiteralExprClass:
    case Stmt::FloatingLiteralClass:
    case Stmt::SizeOfPackExprClass:
      Dst.Add(Pred); // No-op. Simply propagate the current state unchanged.
      break;

    case Stmt::ArraySubscriptExprClass:
      VisitLvalArraySubscriptExpr(cast<ArraySubscriptExpr>(S), Pred, Dst);
      break;

    case Stmt::AsmStmtClass:
      VisitAsmStmt(cast<AsmStmt>(S), Pred, Dst);
      break;

    case Stmt::BlockDeclRefExprClass: {
      const BlockDeclRefExpr *BE = cast<BlockDeclRefExpr>(S);
      VisitCommonDeclRefExpr(BE, BE->getDecl(), Pred, Dst);
      break;
    }

    case Stmt::BlockExprClass:
      VisitBlockExpr(cast<BlockExpr>(S), Pred, Dst);
      break;

    case Stmt::BinaryOperatorClass: {
      const BinaryOperator* B = cast<BinaryOperator>(S);
      if (B->isLogicalOp()) {
        VisitLogicalExpr(B, Pred, Dst);
        break;
      }
      else if (B->getOpcode() == BO_Comma) {
        const GRState* state = GetState(Pred);
        MakeNode(Dst, B, Pred, state->BindExpr(B, state->getSVal(B->getRHS())));
        break;
      }

      if (AMgr.shouldEagerlyAssume() &&
          (B->isRelationalOp() || B->isEqualityOp())) {
        ExplodedNodeSet Tmp;
        VisitBinaryOperator(cast<BinaryOperator>(S), Pred, Tmp);
        evalEagerlyAssume(Dst, Tmp, cast<Expr>(S));
      }
      else
        VisitBinaryOperator(cast<BinaryOperator>(S), Pred, Dst);

      break;
    }

    case Stmt::CallExprClass: {
      const CallExpr* C = cast<CallExpr>(S);
      VisitCall(C, Pred, C->arg_begin(), C->arg_end(), Dst);
      break;
    }

    case Stmt::CXXConstructExprClass: {
      const CXXConstructExpr *C = cast<CXXConstructExpr>(S);
      // For block-level CXXConstructExpr, we don't have a destination region.
      // Let VisitCXXConstructExpr() create one.
      VisitCXXConstructExpr(C, 0, Pred, Dst);
      break;
    }

    case Stmt::CXXMemberCallExprClass: {
      const CXXMemberCallExpr *MCE = cast<CXXMemberCallExpr>(S);
      VisitCXXMemberCallExpr(MCE, Pred, Dst);
      break;
    }

    case Stmt::CXXOperatorCallExprClass: {
      const CXXOperatorCallExpr *C = cast<CXXOperatorCallExpr>(S);
      VisitCXXOperatorCallExpr(C, Pred, Dst);
      break;
    }

    case Stmt::CXXNewExprClass: {
      const CXXNewExpr *NE = cast<CXXNewExpr>(S);
      VisitCXXNewExpr(NE, Pred, Dst);
      break;
    }

    case Stmt::CXXDeleteExprClass: {
      const CXXDeleteExpr *CDE = cast<CXXDeleteExpr>(S);
      VisitCXXDeleteExpr(CDE, Pred, Dst);
      break;
    }
      // FIXME: ChooseExpr is really a constant.  We need to fix
      //        the CFG do not model them as explicit control-flow.

    case Stmt::ChooseExprClass: { // __builtin_choose_expr
      const ChooseExpr* C = cast<ChooseExpr>(S);
      VisitGuardedExpr(C, C->getLHS(), C->getRHS(), Pred, Dst);
      break;
    }

    case Stmt::CompoundAssignOperatorClass:
      VisitBinaryOperator(cast<BinaryOperator>(S), Pred, Dst);
      break;

    case Stmt::CompoundLiteralExprClass:
      VisitCompoundLiteralExpr(cast<CompoundLiteralExpr>(S), Pred, Dst);
      break;

    case Stmt::ConditionalOperatorClass: { // '?' operator
      const ConditionalOperator* C = cast<ConditionalOperator>(S);
      VisitGuardedExpr(C, C->getLHS(), C->getRHS(), Pred, Dst);
      break;
    }

    case Stmt::CXXThisExprClass:
      VisitCXXThisExpr(cast<CXXThisExpr>(S), Pred, Dst);
      break;

    case Stmt::DeclRefExprClass: {
      const DeclRefExpr *DE = cast<DeclRefExpr>(S);
      VisitCommonDeclRefExpr(DE, DE->getDecl(), Pred, Dst);
      break;
    }

    case Stmt::DeclStmtClass:
      VisitDeclStmt(cast<DeclStmt>(S), Pred, Dst);
      break;

    case Stmt::ForStmtClass:
      // This case isn't for branch processing, but for handling the
      // initialization of a condition variable.
      VisitCondInit(cast<ForStmt>(S)->getConditionVariable(), S, Pred, Dst);
      break;

    case Stmt::ImplicitCastExprClass:
    case Stmt::CStyleCastExprClass:
    case Stmt::CXXStaticCastExprClass:
    case Stmt::CXXDynamicCastExprClass:
    case Stmt::CXXReinterpretCastExprClass:
    case Stmt::CXXConstCastExprClass:
    case Stmt::CXXFunctionalCastExprClass: {
      const CastExpr* C = cast<CastExpr>(S);
      VisitCast(C, C->getSubExpr(), Pred, Dst);
      break;
    }

    case Stmt::IfStmtClass:
      // This case isn't for branch processing, but for handling the
      // initialization of a condition variable.
      VisitCondInit(cast<IfStmt>(S)->getConditionVariable(), S, Pred, Dst);
      break;

    case Stmt::InitListExprClass:
      VisitInitListExpr(cast<InitListExpr>(S), Pred, Dst);
      break;

    case Stmt::MemberExprClass:
      VisitMemberExpr(cast<MemberExpr>(S), Pred, Dst);
      break;
    case Stmt::ObjCIvarRefExprClass:
      VisitLvalObjCIvarRefExpr(cast<ObjCIvarRefExpr>(S), Pred, Dst);
      break;

    case Stmt::ObjCForCollectionStmtClass:
      VisitObjCForCollectionStmt(cast<ObjCForCollectionStmt>(S), Pred, Dst);
      break;

    case Stmt::ObjCMessageExprClass:
      VisitObjCMessageExpr(cast<ObjCMessageExpr>(S), Pred, Dst);
      break;

    case Stmt::ObjCAtThrowStmtClass: {
      // FIXME: This is not complete.  We basically treat @throw as
      // an abort.
      SaveAndRestore<bool> OldSink(Builder->BuildSinks);
      Builder->BuildSinks = true;
      MakeNode(Dst, S, Pred, GetState(Pred));
      break;
    }

    case Stmt::ReturnStmtClass:
      VisitReturnStmt(cast<ReturnStmt>(S), Pred, Dst);
      break;

    case Stmt::OffsetOfExprClass:
      VisitOffsetOfExpr(cast<OffsetOfExpr>(S), Pred, Dst);
      break;

    case Stmt::SizeOfAlignOfExprClass:
      VisitSizeOfAlignOfExpr(cast<SizeOfAlignOfExpr>(S), Pred, Dst);
      break;

    case Stmt::StmtExprClass: {
      const StmtExpr* SE = cast<StmtExpr>(S);

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

    case Stmt::StringLiteralClass: {
      const GRState* state = GetState(Pred);
      SVal V = state->getLValue(cast<StringLiteral>(S));
      MakeNode(Dst, S, Pred, state->BindExpr(S, V));
      return;
    }

    case Stmt::SwitchStmtClass:
      // This case isn't for branch processing, but for handling the
      // initialization of a condition variable.
      VisitCondInit(cast<SwitchStmt>(S)->getConditionVariable(), S, Pred, Dst);
      break;

    case Stmt::UnaryOperatorClass: {
      const UnaryOperator *U = cast<UnaryOperator>(S);
      if (AMgr.shouldEagerlyAssume()&&(U->getOpcode() == UO_LNot)) {
        ExplodedNodeSet Tmp;
        VisitUnaryOperator(U, Pred, Tmp);
        evalEagerlyAssume(Dst, Tmp, U);
      }
      else
        VisitUnaryOperator(U, Pred, Dst);
      break;
    }

    case Stmt::WhileStmtClass:
      // This case isn't for branch processing, but for handling the
      // initialization of a condition variable.
      VisitCondInit(cast<WhileStmt>(S)->getConditionVariable(), S, Pred, Dst);
      break;
  }
}

//===----------------------------------------------------------------------===//
// Block entrance.  (Update counters).
//===----------------------------------------------------------------------===//

void ExprEngine::processCFGBlockEntrance(ExplodedNodeSet &dstNodes,
                               GenericNodeBuilder<BlockEntrance> &nodeBuilder){
  
  // FIXME: Refactor this into a checker.
  const CFGBlock *block = nodeBuilder.getProgramPoint().getBlock();
  ExplodedNode *pred = nodeBuilder.getPredecessor();
  
  if (nodeBuilder.getBlockCounter().getNumVisited(
                       pred->getLocationContext()->getCurrentStackFrame(), 
                       block->getBlockID()) >= AMgr.getMaxVisit()) {

    static int tag = 0;
    nodeBuilder.generateNode(pred->getState(), pred, &tag, true);
  }
}

//===----------------------------------------------------------------------===//
// Generic node creation.
//===----------------------------------------------------------------------===//

ExplodedNode* ExprEngine::MakeNode(ExplodedNodeSet& Dst, const Stmt* S,
                                     ExplodedNode* Pred, const GRState* St,
                                     ProgramPoint::Kind K, const void *tag) {
  assert (Builder && "StmtNodeBuilder not present.");
  SaveAndRestore<const void*> OldTag(Builder->Tag);
  Builder->Tag = tag;
  return Builder->MakeNode(Dst, S, Pred, St, K);
}

//===----------------------------------------------------------------------===//
// Branch processing.
//===----------------------------------------------------------------------===//

const GRState* ExprEngine::MarkBranch(const GRState* state,
                                        const Stmt* Terminator,
                                        bool branchTaken) {

  switch (Terminator->getStmtClass()) {
    default:
      return state;

    case Stmt::BinaryOperatorClass: { // '&&' and '||'

      const BinaryOperator* B = cast<BinaryOperator>(Terminator);
      BinaryOperator::Opcode Op = B->getOpcode();

      assert (Op == BO_LAnd || Op == BO_LOr);

      // For &&, if we take the true branch, then the value of the whole
      // expression is that of the RHS expression.
      //
      // For ||, if we take the false branch, then the value of the whole
      // expression is that of the RHS expression.

      const Expr* Ex = (Op == BO_LAnd && branchTaken) ||
                       (Op == BO_LOr && !branchTaken)
                       ? B->getRHS() : B->getLHS();

      return state->BindExpr(B, UndefinedVal(Ex));
    }

    case Stmt::ConditionalOperatorClass: { // ?:

      const ConditionalOperator* C = cast<ConditionalOperator>(Terminator);

      // For ?, if branchTaken == true then the value is either the LHS or
      // the condition itself. (GNU extension).

      const Expr* Ex;

      if (branchTaken)
        Ex = C->getLHS() ? C->getLHS() : C->getCond();
      else
        Ex = C->getRHS();

      return state->BindExpr(C, UndefinedVal(Ex));
    }

    case Stmt::ChooseExprClass: { // ?:

      const ChooseExpr* C = cast<ChooseExpr>(Terminator);

      const Expr* Ex = branchTaken ? C->getLHS() : C->getRHS();
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
                                const Stmt* Condition, ASTContext& Ctx) {

  const Expr *Ex = dyn_cast<Expr>(Condition);
  if (!Ex)
    return UnknownVal();

  uint64_t bits = 0;
  bool bitsInit = false;

  while (const CastExpr *CE = dyn_cast<CastExpr>(Ex)) {
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

void ExprEngine::processBranch(const Stmt* Condition, const Stmt* Term,
                                 BranchNodeBuilder& builder) {

  // Check for NULL conditions; e.g. "for(;;)"
  if (!Condition) {
    builder.markInfeasible(false);
    return;
  }

  PrettyStackTraceLoc CrashInfo(getContext().getSourceManager(),
                                Condition->getLocStart(),
                                "Error evaluating branch");

  for (CheckersOrdered::iterator I=Checkers.begin(),E=Checkers.end();I!=E;++I) {
    void *tag = I->first;
    Checker *checker = I->second;
    checker->VisitBranchCondition(builder, *this, Condition, tag);
  }

  // If the branch condition is undefined, return;
  if (!builder.isFeasible(true) && !builder.isFeasible(false))
    return;

  const GRState* PrevState = builder.getState();
  SVal X = PrevState->getSVal(Condition);

  if (X.isUnknown()) {
    // Give it a chance to recover from unknown.
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
        }
      }
    }
    // If the condition is still unknown, give up.
    if (X.isUnknown()) {
      builder.generateNode(MarkBranch(PrevState, Term, true), true);
      builder.generateNode(MarkBranch(PrevState, Term, false), false);
      return;
    }
  }

  DefinedSVal V = cast<DefinedSVal>(X);

  // Process the true branch.
  if (builder.isFeasible(true)) {
    if (const GRState *state = PrevState->assume(V, true))
      builder.generateNode(MarkBranch(state, Term, true), true);
    else
      builder.markInfeasible(true);
  }

  // Process the false branch.
  if (builder.isFeasible(false)) {
    if (const GRState *state = PrevState->assume(V, false))
      builder.generateNode(MarkBranch(state, Term, false), false);
    else
      builder.markInfeasible(false);
  }
}

/// processIndirectGoto - Called by CoreEngine.  Used to generate successor
///  nodes by processing the 'effects' of a computed goto jump.
void ExprEngine::processIndirectGoto(IndirectGotoNodeBuilder& builder) {

  const GRState *state = builder.getState();
  SVal V = state->getSVal(builder.getTarget());

  // Three possibilities:
  //
  //   (1) We know the computed label.
  //   (2) The label is NULL (or some other constant), or Undefined.
  //   (3) We have no clue about the label.  Dispatch to all targets.
  //

  typedef IndirectGotoNodeBuilder::iterator iterator;

  if (isa<loc::GotoLabel>(V)) {
    const LabelStmt* L = cast<loc::GotoLabel>(V).getLabel();

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
    //ExplodedNode* N = builder.generateNode(builder.begin(), state, true);
    // FIXME: add checker visit.
    //    UndefBranches.insert(N);
    return;
  }

  // This is really a catch-all.  We don't support symbolics yet.
  // FIXME: Implement dispatch for symbolic pointers.

  for (iterator I=builder.begin(), E=builder.end(); I != E; ++I)
    builder.generateNode(I, state);
}


void ExprEngine::VisitGuardedExpr(const Expr* Ex, const Expr* L, 
                                    const Expr* R,
                                    ExplodedNode* Pred, ExplodedNodeSet& Dst) {

  assert(Ex == currentStmt &&
         Pred->getLocationContext()->getCFG()->isBlkExpr(Ex));

  const GRState* state = GetState(Pred);
  SVal X = state->getSVal(Ex);

  assert (X.isUndef());

  const Expr *SE = (Expr*) cast<UndefinedVal>(X).getData();
  assert(SE);
  X = state->getSVal(SE);

  // Make sure that we invalidate the previous binding.
  MakeNode(Dst, Ex, Pred, state->BindExpr(Ex, X, true));
}

/// ProcessEndPath - Called by CoreEngine.  Used to generate end-of-path
///  nodes when the control reaches the end of a function.
void ExprEngine::processEndOfFunction(EndOfFunctionNodeBuilder& builder) {
  getTF().evalEndPath(*this, builder);
  StateMgr.EndPath(builder.getState());
  for (CheckersOrdered::iterator I=Checkers.begin(),E=Checkers.end(); I!=E;++I){
    void *tag = I->first;
    Checker *checker = I->second;
    checker->evalEndPath(builder, tag, *this);
  }
}

/// ProcessSwitch - Called by CoreEngine.  Used to generate successor
///  nodes by processing the 'effects' of a switch statement.
void ExprEngine::processSwitch(SwitchNodeBuilder& builder) {
  typedef SwitchNodeBuilder::iterator iterator;
  const GRState* state = builder.getState();
  const Expr* CondE = builder.getCondition();
  SVal  CondV_untested = state->getSVal(CondE);

  if (CondV_untested.isUndef()) {
    //ExplodedNode* N = builder.generateDefaultCaseNode(state, true);
    // FIXME: add checker
    //UndefBranches.insert(N);

    return;
  }
  DefinedOrUnknownSVal CondV = cast<DefinedOrUnknownSVal>(CondV_untested);

  const GRState *DefaultSt = state;
  
  iterator I = builder.begin(), EI = builder.end();
  bool defaultIsFeasible = I == EI;

  for ( ; I != EI; ++I) {
    const CaseStmt* Case = I.getCase();

    // Evaluate the LHS of the case value.
    Expr::EvalResult V1;
    bool b = Case->getLHS()->Evaluate(V1, getContext());

    // Sanity checks.  These go away in Release builds.
    assert(b && V1.Val.isInt() && !V1.HasSideEffects
             && "Case condition must evaluate to an integer constant.");
    (void)b; // silence unused variable warning
    assert(V1.Val.getInt().getBitWidth() ==
           getContext().getTypeSize(CondE->getType()));

    // Get the RHS of the case, if it exists.
    Expr::EvalResult V2;

    if (const Expr* E = Case->getRHS()) {
      b = E->Evaluate(V2, getContext());
      assert(b && V2.Val.isInt() && !V2.HasSideEffects
             && "Case condition must evaluate to an integer constant.");
      (void)b; // silence unused variable warning
    }
    else
      V2 = V1;

    // FIXME: Eventually we should replace the logic below with a range
    //  comparison, rather than concretize the values within the range.
    //  This should be easy once we have "ranges" for NonLVals.

    do {
      nonloc::ConcreteInt CaseVal(getBasicVals().getValue(V1.Val.getInt()));
      DefinedOrUnknownSVal Res = svalBuilder.evalEQ(DefaultSt ? DefaultSt : state,
                                               CondV, CaseVal);

      // Now "assume" that the case matches.
      if (const GRState* stateNew = state->assume(Res, true)) {
        builder.generateCaseStmtNode(I, stateNew);

        // If CondV evaluates to a constant, then we know that this
        // is the *only* case that we can take, so stop evaluating the
        // others.
        if (isa<nonloc::ConcreteInt>(CondV))
          return;
      }

      // Now "assume" that the case doesn't match.  Add this state
      // to the default state (if it is feasible).
      if (DefaultSt) {
        if (const GRState *stateNew = DefaultSt->assume(Res, false)) {
          defaultIsFeasible = true;
          DefaultSt = stateNew;
        }
        else {
          defaultIsFeasible = false;
          DefaultSt = NULL;
        }
      }

      // Concretize the next value in the range.
      if (V1.Val.getInt() == V2.Val.getInt())
        break;

      ++V1.Val.getInt();
      assert (V1.Val.getInt() <= V2.Val.getInt());

    } while (true);
  }

  if (!defaultIsFeasible)
    return;

  // If we have switch(enum value), the default branch is not
  // feasible if all of the enum constants not covered by 'case:' statements
  // are not feasible values for the switch condition.
  //
  // Note that this isn't as accurate as it could be.  Even if there isn't
  // a case for a particular enum value as long as that enum value isn't
  // feasible then it shouldn't be considered for making 'default:' reachable.
  const SwitchStmt *SS = builder.getSwitch();
  const Expr *CondExpr = SS->getCond()->IgnoreParenImpCasts();
  if (CondExpr->getType()->getAs<EnumType>()) {
    if (SS->isAllEnumCasesCovered())
      return;
  }

  builder.generateDefaultCaseNode(DefaultSt);
}

void ExprEngine::processCallEnter(CallEnterNodeBuilder &B) {
  const GRState *state = B.getState()->EnterStackFrame(B.getCalleeContext());
  B.generateNode(state);
}

void ExprEngine::processCallExit(CallExitNodeBuilder &B) {
  const GRState *state = B.getState();
  const ExplodedNode *Pred = B.getPredecessor();
  const StackFrameContext *calleeCtx = 
                            cast<StackFrameContext>(Pred->getLocationContext());
  const Stmt *CE = calleeCtx->getCallSite();

  // If the callee returns an expression, bind its value to CallExpr.
  const Stmt *ReturnedExpr = state->get<ReturnExpr>();
  if (ReturnedExpr) {
    SVal RetVal = state->getSVal(ReturnedExpr);
    state = state->BindExpr(CE, RetVal);
    // Clear the return expr GDM.
    state = state->remove<ReturnExpr>();
  }

  // Bind the constructed object value to CXXConstructExpr.
  if (const CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(CE)) {
    const CXXThisRegion *ThisR =
      getCXXThisRegion(CCE->getConstructor()->getParent(), calleeCtx);

    SVal ThisV = state->getSVal(ThisR);
    // Always bind the region to the CXXConstructExpr.
    state = state->BindExpr(CCE, ThisV);
  }

  B.generateNode(state);
}

//===----------------------------------------------------------------------===//
// Transfer functions: logical operations ('&&', '||').
//===----------------------------------------------------------------------===//

void ExprEngine::VisitLogicalExpr(const BinaryOperator* B, ExplodedNode* Pred,
                                    ExplodedNodeSet& Dst) {

  assert(B->getOpcode() == BO_LAnd ||
         B->getOpcode() == BO_LOr);

  assert(B==currentStmt && Pred->getLocationContext()->getCFG()->isBlkExpr(B));

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
    if (const GRState *newState = state->assume(XD, true))
      MakeNode(Dst, B, Pred,
               newState->BindExpr(B, svalBuilder.makeIntVal(1U, B->getType())));

    if (const GRState *newState = state->assume(XD, false))
      MakeNode(Dst, B, Pred,
               newState->BindExpr(B, svalBuilder.makeIntVal(0U, B->getType())));
  }
  else {
    // We took the LHS expression.  Depending on whether we are '&&' or
    // '||' we know what the value of the expression is via properties of
    // the short-circuiting.
    X = svalBuilder.makeIntVal(B->getOpcode() == BO_LAnd ? 0U : 1U,
                          B->getType());
    MakeNode(Dst, B, Pred, state->BindExpr(B, X));
  }
}

//===----------------------------------------------------------------------===//
// Transfer functions: Loads and stores.
//===----------------------------------------------------------------------===//

void ExprEngine::VisitBlockExpr(const BlockExpr *BE, ExplodedNode *Pred,
                                  ExplodedNodeSet &Dst) {

  ExplodedNodeSet Tmp;

  CanQualType T = getContext().getCanonicalType(BE->getType());
  SVal V = svalBuilder.getBlockPointer(BE->getBlockDecl(), T,
                                  Pred->getLocationContext());

  MakeNode(Tmp, BE, Pred, GetState(Pred)->BindExpr(BE, V),
           ProgramPoint::PostLValueKind);

  // Post-visit the BlockExpr.
  CheckerVisit(BE, Dst, Tmp, PostVisitStmtCallback);
}

void ExprEngine::VisitCommonDeclRefExpr(const Expr *Ex, const NamedDecl *D,
                                        ExplodedNode *Pred,
                                        ExplodedNodeSet &Dst) {
  const GRState *state = GetState(Pred);

  if (const VarDecl* VD = dyn_cast<VarDecl>(D)) {
    assert(Ex->isLValue());
    SVal V = state->getLValue(VD, Pred->getLocationContext());

    // For references, the 'lvalue' is the pointer address stored in the
    // reference region.
    if (VD->getType()->isReferenceType()) {
      if (const MemRegion *R = V.getAsRegion())
        V = state->getSVal(R);
      else
        V = UnknownVal();
    }

    MakeNode(Dst, Ex, Pred, state->BindExpr(Ex, V),
             ProgramPoint::PostLValueKind);
    return;
  }
  if (const EnumConstantDecl* ED = dyn_cast<EnumConstantDecl>(D)) {
    assert(!Ex->isLValue());
    SVal V = svalBuilder.makeIntVal(ED->getInitVal());
    MakeNode(Dst, Ex, Pred, state->BindExpr(Ex, V));
    return;
  }
  if (const FunctionDecl* FD = dyn_cast<FunctionDecl>(D)) {
    SVal V = svalBuilder.getFunctionPointer(FD);
    MakeNode(Dst, Ex, Pred, state->BindExpr(Ex, V),
             ProgramPoint::PostLValueKind);
    return;
  }
  assert (false &&
          "ValueDecl support for this ValueDecl not implemented.");
}

/// VisitArraySubscriptExpr - Transfer function for array accesses
void ExprEngine::VisitLvalArraySubscriptExpr(const ArraySubscriptExpr* A,
                                             ExplodedNode* Pred,
                                             ExplodedNodeSet& Dst){

  const Expr* Base = A->getBase()->IgnoreParens();
  const Expr* Idx  = A->getIdx()->IgnoreParens();
  
  // Evaluate the base.
  ExplodedNodeSet Tmp;
  Visit(Base, Pred, Tmp);

  for (ExplodedNodeSet::iterator I1=Tmp.begin(), E1=Tmp.end(); I1!=E1; ++I1) {
    ExplodedNodeSet Tmp2;
    Visit(Idx, *I1, Tmp2);     // Evaluate the index.
    ExplodedNodeSet Tmp3;
    CheckerVisit(A, Tmp3, Tmp2, PreVisitStmtCallback);

    for (ExplodedNodeSet::iterator I2=Tmp3.begin(),E2=Tmp3.end();I2!=E2; ++I2) {
      const GRState* state = GetState(*I2);
      SVal V = state->getLValue(A->getType(), state->getSVal(Idx),
                                state->getSVal(Base));
      assert(A->isLValue());
      MakeNode(Dst, A, *I2, state->BindExpr(A, V), ProgramPoint::PostLValueKind);
    }
  }
}

/// VisitMemberExpr - Transfer function for member expressions.
void ExprEngine::VisitMemberExpr(const MemberExpr* M, ExplodedNode* Pred,
                                 ExplodedNodeSet& Dst) {

  Expr *baseExpr = M->getBase()->IgnoreParens();
  ExplodedNodeSet dstBase;
  Visit(baseExpr, Pred, dstBase);

  FieldDecl *field = dyn_cast<FieldDecl>(M->getMemberDecl());
  if (!field) // FIXME: skipping member expressions for non-fields
    return;

  for (ExplodedNodeSet::iterator I = dstBase.begin(), E = dstBase.end();
    I != E; ++I) {
    const GRState* state = GetState(*I);
    SVal baseExprVal = state->getSVal(baseExpr);
    if (isa<nonloc::LazyCompoundVal>(baseExprVal) ||
        isa<nonloc::CompoundVal>(baseExprVal)) {
      MakeNode(Dst, M, *I, state->BindExpr(M, UnknownVal()));
      continue;
    }

    // FIXME: Should we insert some assumption logic in here to determine
    // if "Base" is a valid piece of memory?  Before we put this assumption
    // later when using FieldOffset lvals (which we no longer have).

    // For all other cases, compute an lvalue.    
    SVal L = state->getLValue(field, baseExprVal);
    if (M->isLValue())
      MakeNode(Dst, M, *I, state->BindExpr(M, L), ProgramPoint::PostLValueKind);
    else
      evalLoad(Dst, M, *I, state, L);
  }
}

/// evalBind - Handle the semantics of binding a value to a specific location.
///  This method is used by evalStore and (soon) VisitDeclStmt, and others.
void ExprEngine::evalBind(ExplodedNodeSet& Dst, const Stmt* StoreE,
                            ExplodedNode* Pred, const GRState* state,
                            SVal location, SVal Val, bool atDeclInit) {


  // Do a previsit of the bind.
  ExplodedNodeSet CheckedSet, Src;
  Src.Add(Pred);
  CheckerVisitBind(StoreE, CheckedSet, Src, location, Val, true);

  for (ExplodedNodeSet::iterator I = CheckedSet.begin(), E = CheckedSet.end();
       I!=E; ++I) {

    if (Pred != *I)
      state = GetState(*I);

    const GRState* newState = 0;

    if (atDeclInit) {
      const VarRegion *VR =
        cast<VarRegion>(cast<loc::MemRegionVal>(location).getRegion());

      newState = state->bindDecl(VR, Val);
    }
    else {
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
    }

    // The next thing to do is check if the TransferFuncs object wants to
    // update the state based on the new binding.  If the GRTransferFunc object
    // doesn't do anything, just auto-propagate the current state.
    
    // NOTE: We use 'AssignE' for the location of the PostStore if 'AssignE'
    // is non-NULL.  Checkers typically care about 
    
    StmtNodeBuilderRef BuilderRef(Dst, *Builder, *this, *I, newState, StoreE,
                                    true);

    getTF().evalBind(BuilderRef, location, Val);
  }
}

/// evalStore - Handle the semantics of a store via an assignment.
///  @param Dst The node set to store generated state nodes
///  @param AssignE The assignment expression if the store happens in an 
///         assignment.
///  @param LocatioinE The location expression that is stored to.
///  @param state The current simulation state
///  @param location The location to store the value
///  @param Val The value to be stored
void ExprEngine::evalStore(ExplodedNodeSet& Dst, const Expr *AssignE,
                             const Expr* LocationE,
                             ExplodedNode* Pred,
                             const GRState* state, SVal location, SVal Val,
                             const void *tag) {

  assert(Builder && "StmtNodeBuilder must be defined.");

  // Evaluate the location (checks for bad dereferences).
  ExplodedNodeSet Tmp;
  evalLocation(Tmp, LocationE, Pred, state, location, tag, false);

  if (Tmp.empty())
    return;

  assert(!location.isUndef());

  SaveAndRestore<ProgramPoint::Kind> OldSPointKind(Builder->PointKind,
                                                   ProgramPoint::PostStoreKind);

  // Proceed with the store.  We use AssignE as the anchor for the PostStore
  // ProgramPoint if it is non-NULL, and LocationE otherwise.
  const Expr *StoreE = AssignE ? AssignE : LocationE;

  for (ExplodedNodeSet::iterator NI=Tmp.begin(), NE=Tmp.end(); NI!=NE; ++NI)
    evalBind(Dst, StoreE, *NI, GetState(*NI), location, Val);
}

void ExprEngine::evalLoad(ExplodedNodeSet& Dst, const Expr *Ex,
                            ExplodedNode* Pred,
                            const GRState* state, SVal location,
                            const void *tag, QualType LoadTy) {
  assert(!isa<NonLoc>(location) && "location cannot be a NonLoc.");

  // Are we loading from a region?  This actually results in two loads; one
  // to fetch the address of the referenced value and one to fetch the
  // referenced value.
  if (const TypedRegion *TR =
        dyn_cast_or_null<TypedRegion>(location.getAsRegion())) {

    QualType ValTy = TR->getValueType();
    if (const ReferenceType *RT = ValTy->getAs<ReferenceType>()) {
      static int loadReferenceTag = 0;
      ExplodedNodeSet Tmp;
      evalLoadCommon(Tmp, Ex, Pred, state, location, &loadReferenceTag,
                     getContext().getPointerType(RT->getPointeeType()));

      // Perform the load from the referenced value.
      for (ExplodedNodeSet::iterator I=Tmp.begin(), E=Tmp.end() ; I!=E; ++I) {
        state = GetState(*I);
        location = state->getSVal(Ex);
        evalLoadCommon(Dst, Ex, *I, state, location, tag, LoadTy);
      }
      return;
    }
  }

  evalLoadCommon(Dst, Ex, Pred, state, location, tag, LoadTy);
}

void ExprEngine::evalLoadCommon(ExplodedNodeSet& Dst, const Expr *Ex,
                                  ExplodedNode* Pred,
                                  const GRState* state, SVal location,
                                  const void *tag, QualType LoadTy) {

  // Evaluate the location (checks for bad dereferences).
  ExplodedNodeSet Tmp;
  evalLocation(Tmp, Ex, Pred, state, location, tag, true);

  if (Tmp.empty())
    return;

  assert(!location.isUndef());

  SaveAndRestore<ProgramPoint::Kind> OldSPointKind(Builder->PointKind);

  // Proceed with the load.
  for (ExplodedNodeSet::iterator NI=Tmp.begin(), NE=Tmp.end(); NI!=NE; ++NI) {
    state = GetState(*NI);

    if (location.isUnknown()) {
      // This is important.  We must nuke the old binding.
      MakeNode(Dst, Ex, *NI, state->BindExpr(Ex, UnknownVal()),
               ProgramPoint::PostLoadKind, tag);
    }
    else {
      if (LoadTy.isNull())
        LoadTy = Ex->getType();
      SVal V = state->getSVal(cast<Loc>(location), LoadTy);
      MakeNode(Dst, Ex, *NI, state->bindExprAndLocation(Ex, location, V),
               ProgramPoint::PostLoadKind, tag);
    }
  }
}

void ExprEngine::evalLocation(ExplodedNodeSet &Dst, const Stmt *S,
                                ExplodedNode* Pred,
                                const GRState* state, SVal location,
                                const void *tag, bool isLoad) {
  // Early checks for performance reason.
  if (location.isUnknown() || Checkers.empty()) {
    Dst.Add(Pred);
    return;
  }

  ExplodedNodeSet Src, Tmp;
  Src.Add(Pred);
  ExplodedNodeSet *PrevSet = &Src;

  for (CheckersOrdered::iterator I=Checkers.begin(),E=Checkers.end(); I!=E; ++I)
  {
    ExplodedNodeSet *CurrSet = 0;
    if (I+1 == E)
      CurrSet = &Dst;
    else {
      CurrSet = (PrevSet == &Tmp) ? &Src : &Tmp;
      CurrSet->clear();
    }

    void *tag = I->first;
    Checker *checker = I->second;

    for (ExplodedNodeSet::iterator NI = PrevSet->begin(), NE = PrevSet->end();
         NI != NE; ++NI) {
      // Use the 'state' argument only when the predecessor node is the
      // same as Pred.  This allows us to catch updates to the state.
      checker->GR_visitLocation(*CurrSet, *Builder, *this, S, *NI,
                                *NI == Pred ? state : GetState(*NI),
                                location, tag, isLoad);
    }

    // Update which NodeSet is the current one.
    PrevSet = CurrSet;
  }
}

bool ExprEngine::InlineCall(ExplodedNodeSet &Dst, const CallExpr *CE, 
                              ExplodedNode *Pred) {
  const GRState *state = GetState(Pred);
  const Expr *Callee = CE->getCallee();
  SVal L = state->getSVal(Callee);
  
  const FunctionDecl *FD = L.getAsFunctionDecl();
  if (!FD)
    return false;

  // Check if the function definition is in the same translation unit.
  if (FD->hasBody(FD)) {
    const StackFrameContext *stackFrame = 
      AMgr.getStackFrame(AMgr.getAnalysisContext(FD), 
                         Pred->getLocationContext(),
                         CE, Builder->getBlock(), Builder->getIndex());
    // Now we have the definition of the callee, create a CallEnter node.
    CallEnter Loc(CE, stackFrame, Pred->getLocationContext());

    ExplodedNode *N = Builder->generateNode(Loc, state, Pred);
    Dst.Add(N);
    return true;
  }

  // Check if we can find the function definition in other translation units.
  if (AMgr.hasIndexer()) {
    AnalysisContext *C = AMgr.getAnalysisContextInAnotherTU(FD);
    if (C == 0)
      return false;
    const StackFrameContext *stackFrame = 
      AMgr.getStackFrame(C, Pred->getLocationContext(),
                         CE, Builder->getBlock(), Builder->getIndex());
    CallEnter Loc(CE, stackFrame, Pred->getLocationContext());
    ExplodedNode *N = Builder->generateNode(Loc, state, Pred);
    Dst.Add(N);
    return true;
  }

  return false;
}

void ExprEngine::VisitCall(const CallExpr* CE, ExplodedNode* Pred,
                             CallExpr::const_arg_iterator AI,
                             CallExpr::const_arg_iterator AE,
                             ExplodedNodeSet& Dst) {

  // Determine the type of function we're calling (if available).
  const FunctionProtoType *Proto = NULL;
  QualType FnType = CE->getCallee()->IgnoreParens()->getType();
  if (const PointerType *FnTypePtr = FnType->getAs<PointerType>())
    Proto = FnTypePtr->getPointeeType()->getAs<FunctionProtoType>();

  // Evaluate the arguments.
  ExplodedNodeSet ArgsEvaluated;
  evalArguments(CE->arg_begin(), CE->arg_end(), Proto, Pred, ArgsEvaluated);

  // Now process the call itself.
  ExplodedNodeSet DstTmp;
  const Expr* Callee = CE->getCallee()->IgnoreParens();

  for (ExplodedNodeSet::iterator NI=ArgsEvaluated.begin(),
                                 NE=ArgsEvaluated.end(); NI != NE; ++NI) {
    // Evaluate the callee.
    ExplodedNodeSet DstTmp2;
    Visit(Callee, *NI, DstTmp2);
    // Perform the previsit of the CallExpr, storing the results in DstTmp.
    CheckerVisit(CE, DstTmp, DstTmp2, PreVisitStmtCallback);
  }

  // Finally, evaluate the function call.  We try each of the checkers
  // to see if the can evaluate the function call.
  ExplodedNodeSet DstTmp3;

  for (ExplodedNodeSet::iterator DI = DstTmp.begin(), DE = DstTmp.end();
       DI != DE; ++DI) {

    const GRState* state = GetState(*DI);
    SVal L = state->getSVal(Callee);

    // FIXME: Add support for symbolic function calls (calls involving
    //  function pointer values that are symbolic).
    SaveAndRestore<bool> OldSink(Builder->BuildSinks);
    ExplodedNodeSet DstChecker;

    // If the callee is processed by a checker, skip the rest logic.
    if (CheckerEvalCall(CE, DstChecker, *DI))
      DstTmp3.insert(DstChecker);
    else if (AMgr.shouldInlineCall() && InlineCall(Dst, CE, *DI)) {
      // Callee is inlined. We shouldn't do post call checking.
      return;
    }
    else {
      for (ExplodedNodeSet::iterator DI_Checker = DstChecker.begin(),
           DE_Checker = DstChecker.end();
           DI_Checker != DE_Checker; ++DI_Checker) {

        // Dispatch to the plug-in transfer function.
        unsigned oldSize = DstTmp3.size();
        SaveOr OldHasGen(Builder->hasGeneratedNode);
        Pred = *DI_Checker;

        // Dispatch to transfer function logic to handle the call itself.
        // FIXME: Allow us to chain together transfer functions.
        assert(Builder && "StmtNodeBuilder must be defined.");
        getTF().evalCall(DstTmp3, *this, *Builder, CE, L, Pred);

        // Handle the case where no nodes where generated.  Auto-generate that
        // contains the updated state if we aren't generating sinks.
        if (!Builder->BuildSinks && DstTmp3.size() == oldSize &&
            !Builder->hasGeneratedNode)
          MakeNode(DstTmp3, CE, Pred, state);
      }
    }
  }

  // Finally, perform the post-condition check of the CallExpr and store
  // the created nodes in 'Dst'.
  CheckerVisit(CE, Dst, DstTmp3, PostVisitStmtCallback);
}

//===----------------------------------------------------------------------===//
// Transfer function: Objective-C ivar references.
//===----------------------------------------------------------------------===//

static std::pair<const void*,const void*> EagerlyAssumeTag
  = std::pair<const void*,const void*>(&EagerlyAssumeTag,static_cast<void*>(0));

void ExprEngine::evalEagerlyAssume(ExplodedNodeSet &Dst, ExplodedNodeSet &Src,
                                     const Expr *Ex) {
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

    const GRState* state = GetState(Pred);
    SVal V = state->getSVal(Ex);
    if (nonloc::SymExprVal *SEV = dyn_cast<nonloc::SymExprVal>(&V)) {
      // First assume that the condition is true.
      if (const GRState *stateTrue = state->assume(*SEV, true)) {
        stateTrue = stateTrue->BindExpr(Ex,
                                        svalBuilder.makeIntVal(1U, Ex->getType()));
        Dst.Add(Builder->generateNode(PostStmtCustom(Ex,
                                &EagerlyAssumeTag, Pred->getLocationContext()),
                                      stateTrue, Pred));
      }

      // Next, assume that the condition is false.
      if (const GRState *stateFalse = state->assume(*SEV, false)) {
        stateFalse = stateFalse->BindExpr(Ex,
                                          svalBuilder.makeIntVal(0U, Ex->getType()));
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
// Transfer function: Objective-C @synchronized.
//===----------------------------------------------------------------------===//

void ExprEngine::VisitObjCAtSynchronizedStmt(const ObjCAtSynchronizedStmt *S,
                                               ExplodedNode *Pred,
                                               ExplodedNodeSet &Dst) {

  // The mutex expression is a CFGElement, so we don't need to explicitly
  // visit it since it will already be processed.

  // Pre-visit the ObjCAtSynchronizedStmt.
  ExplodedNodeSet Tmp;
  Tmp.Add(Pred);
  CheckerVisit(S, Dst, Tmp, PreVisitStmtCallback);
}

//===----------------------------------------------------------------------===//
// Transfer function: Objective-C ivar references.
//===----------------------------------------------------------------------===//

void ExprEngine::VisitLvalObjCIvarRefExpr(const ObjCIvarRefExpr* Ex, 
                                          ExplodedNode* Pred,
                                          ExplodedNodeSet& Dst) {

  // Visit the base expression, which is needed for computing the lvalue
  // of the ivar.
  ExplodedNodeSet dstBase;
  const Expr *baseExpr = Ex->getBase();
  Visit(baseExpr, Pred, dstBase);

  ExplodedNodeSet dstIvar;

  // Using the base, compute the lvalue of the instance variable.
  for (ExplodedNodeSet::iterator I = dstBase.begin(), E = dstBase.end();
       I!=E; ++I) {
    ExplodedNode *nodeBase = *I;
    const GRState *state = GetState(nodeBase);
    SVal baseVal = state->getSVal(baseExpr);
    SVal location = state->getLValue(Ex->getDecl(), baseVal);
    MakeNode(dstIvar, Ex, *I, state->BindExpr(Ex, location));
  }

  // Perform the post-condition check of the ObjCIvarRefExpr and store
  // the created nodes in 'Dst'.
  CheckerVisit(Ex, Dst, dstIvar, PostVisitStmtCallback);
}

//===----------------------------------------------------------------------===//
// Transfer function: Objective-C fast enumeration 'for' statements.
//===----------------------------------------------------------------------===//

void ExprEngine::VisitObjCForCollectionStmt(const ObjCForCollectionStmt* S,
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

  const Stmt* elem = S->getElement();
  SVal ElementV;

  if (const DeclStmt* DS = dyn_cast<DeclStmt>(elem)) {
    const VarDecl* ElemD = cast<VarDecl>(DS->getSingleDecl());
    assert (ElemD->getInit() == 0);
    ElementV = GetState(Pred)->getLValue(ElemD, Pred->getLocationContext());
    VisitObjCForCollectionStmtAux(S, Pred, Dst, ElementV);
    return;
  }

  ExplodedNodeSet Tmp;
  Visit(cast<Expr>(elem), Pred, Tmp);
  for (ExplodedNodeSet::iterator I = Tmp.begin(), E = Tmp.end(); I!=E; ++I) {
    const GRState* state = GetState(*I);
    VisitObjCForCollectionStmtAux(S, *I, Dst, state->getSVal(elem));
  }
}

void ExprEngine::VisitObjCForCollectionStmtAux(const ObjCForCollectionStmt* S,
                                       ExplodedNode* Pred, ExplodedNodeSet& Dst,
                                                 SVal ElementV) {

  // Check if the location we are writing back to is a null pointer.
  const Stmt* elem = S->getElement();
  ExplodedNodeSet Tmp;
  evalLocation(Tmp, elem, Pred, GetState(Pred), ElementV, NULL, false);

  if (Tmp.empty())
    return;

  for (ExplodedNodeSet::iterator NI=Tmp.begin(), NE=Tmp.end(); NI!=NE; ++NI) {
    Pred = *NI;
    const GRState *state = GetState(Pred);

    // Handle the case where the container still has elements.
    SVal TrueV = svalBuilder.makeTruthVal(1);
    const GRState *hasElems = state->BindExpr(S, TrueV);

    // Handle the case where the container has no elements.
    SVal FalseV = svalBuilder.makeTruthVal(0);
    const GRState *noElems = state->BindExpr(S, FalseV);

    if (loc::MemRegionVal* MV = dyn_cast<loc::MemRegionVal>(&ElementV))
      if (const TypedRegion* R = dyn_cast<TypedRegion>(MV->getRegion())) {
        // FIXME: The proper thing to do is to really iterate over the
        //  container.  We will do this with dispatch logic to the store.
        //  For now, just 'conjure' up a symbolic value.
        QualType T = R->getValueType();
        assert(Loc::IsLocType(T));
        unsigned Count = Builder->getCurrentBlockCount();
        SymbolRef Sym = SymMgr.getConjuredSymbol(elem, T, Count);
        SVal V = svalBuilder.makeLoc(Sym);
        hasElems = hasElems->bindLoc(ElementV, V);

        // Bind the location to 'nil' on the false branch.
        SVal nilV = svalBuilder.makeIntVal(0, T);
        noElems = noElems->bindLoc(ElementV, nilV);
      }

    // Create the new nodes.
    MakeNode(Dst, S, Pred, hasElems);
    MakeNode(Dst, S, Pred, noElems);
  }
}

//===----------------------------------------------------------------------===//
// Transfer function: Objective-C message expressions.
//===----------------------------------------------------------------------===//

namespace {
class ObjCMsgWLItem {
public:
  ObjCMessageExpr::const_arg_iterator I;
  ExplodedNode *N;

  ObjCMsgWLItem(const ObjCMessageExpr::const_arg_iterator &i, ExplodedNode *n)
    : I(i), N(n) {}
};
} // end anonymous namespace

void ExprEngine::VisitObjCMessageExpr(const ObjCMessageExpr* ME, 
                                        ExplodedNode* Pred,
                                        ExplodedNodeSet& Dst){

  // Create a worklist to process both the arguments.
  llvm::SmallVector<ObjCMsgWLItem, 20> WL;

  // But first evaluate the receiver (if any).
  ObjCMessageExpr::const_arg_iterator AI = ME->arg_begin(), AE = ME->arg_end();
  if (const Expr *Receiver = ME->getInstanceReceiver()) {
    ExplodedNodeSet Tmp;
    Visit(Receiver, Pred, Tmp);

    if (Tmp.empty())
      return;

    for (ExplodedNodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I)
      WL.push_back(ObjCMsgWLItem(AI, *I));
  }
  else
    WL.push_back(ObjCMsgWLItem(AI, Pred));

  // Evaluate the arguments.
  ExplodedNodeSet ArgsEvaluated;
  while (!WL.empty()) {
    ObjCMsgWLItem Item = WL.back();
    WL.pop_back();

    if (Item.I == AE) {
      ArgsEvaluated.insert(Item.N);
      continue;
    }

    // Evaluate the subexpression.
    ExplodedNodeSet Tmp;

    // FIXME: [Objective-C++] handle arguments that are references
    Visit(*Item.I, Item.N, Tmp);

    // Enqueue evaluating the next argument on the worklist.
    ++(Item.I);
    for (ExplodedNodeSet::iterator NI=Tmp.begin(), NE=Tmp.end(); NI!=NE; ++NI)
      WL.push_back(ObjCMsgWLItem(Item.I, *NI));
  }

  // Now that the arguments are processed, handle the previsits checks.
  ExplodedNodeSet DstPrevisit;
  CheckerVisit(ME, DstPrevisit, ArgsEvaluated, PreVisitStmtCallback);

  // Proceed with evaluate the message expression.
  ExplodedNodeSet dstEval;

  for (ExplodedNodeSet::iterator DI = DstPrevisit.begin(),
                                 DE = DstPrevisit.end(); DI != DE; ++DI) {

    Pred = *DI;
    bool RaisesException = false;
    unsigned oldSize = dstEval.size();
    SaveAndRestore<bool> OldSink(Builder->BuildSinks);
    SaveOr OldHasGen(Builder->hasGeneratedNode);

    if (const Expr *Receiver = ME->getInstanceReceiver()) {
      const GRState *state = GetState(Pred);

      // Bifurcate the state into nil and non-nil ones.
      DefinedOrUnknownSVal receiverVal =
        cast<DefinedOrUnknownSVal>(state->getSVal(Receiver));

      const GRState *notNilState, *nilState;
      llvm::tie(notNilState, nilState) = state->assume(receiverVal);

      // There are three cases: can be nil or non-nil, must be nil, must be
      // non-nil. We handle must be nil, and merge the rest two into non-nil.
      if (nilState && !notNilState) {
        CheckerEvalNilReceiver(ME, dstEval, nilState, Pred);
        continue;
      }

      // Check if the "raise" message was sent.
      assert(notNilState);
      if (ME->getSelector() == RaiseSel)
        RaisesException = true;

      // Check if we raise an exception.  For now treat these as sinks.
      // Eventually we will want to handle exceptions properly.
      if (RaisesException)
        Builder->BuildSinks = true;

      // Dispatch to plug-in transfer function.
      evalObjCMessageExpr(dstEval, ME, Pred, notNilState);
    }
    else if (ObjCInterfaceDecl *Iface = ME->getReceiverInterface()) {
      IdentifierInfo* ClsName = Iface->getIdentifier();
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
          NSExceptionInstanceRaiseSelectors =
            new Selector[NUM_RAISE_SELECTORS];
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
            RaisesException = true;
            break;
          }
      }

      // Check if we raise an exception.  For now treat these as sinks.
      // Eventually we will want to handle exceptions properly.
      if (RaisesException)
        Builder->BuildSinks = true;

      // Dispatch to plug-in transfer function.
      evalObjCMessageExpr(dstEval, ME, Pred, Builder->GetState(Pred));
    }

    // Handle the case where no nodes where generated.  Auto-generate that
    // contains the updated state if we aren't generating sinks.
    if (!Builder->BuildSinks && dstEval.size() == oldSize &&
        !Builder->hasGeneratedNode)
      MakeNode(dstEval, ME, Pred, GetState(Pred));
  }

  // Finally, perform the post-condition check of the ObjCMessageExpr and store
  // the created nodes in 'Dst'.
  CheckerVisit(ME, Dst, dstEval, PostVisitStmtCallback);
}

//===----------------------------------------------------------------------===//
// Transfer functions: Miscellaneous statements.
//===----------------------------------------------------------------------===//

void ExprEngine::VisitCast(const CastExpr *CastE, const Expr *Ex, 
                           ExplodedNode *Pred, ExplodedNodeSet &Dst) {
  
  ExplodedNodeSet S1;
  Visit(Ex, Pred, S1);
  ExplodedNodeSet S2;
  CheckerVisit(CastE, S2, S1, PreVisitStmtCallback);
  
  if (CastE->getCastKind() == CK_LValueToRValue) {
    for (ExplodedNodeSet::iterator I = S2.begin(), E = S2.end(); I!=E; ++I) {
      ExplodedNode *subExprNode = *I;
      const GRState *state = GetState(subExprNode);
      evalLoad(Dst, CastE, subExprNode, state, state->getSVal(Ex));
    }
    return;
  }
  
  // All other casts.  
  QualType T = CastE->getType();
  QualType ExTy = Ex->getType();

  if (const ExplicitCastExpr *ExCast=dyn_cast_or_null<ExplicitCastExpr>(CastE))
    T = ExCast->getTypeAsWritten();
 
#if 0
  // If we are evaluating the cast in an lvalue context, we implicitly want
  // the cast to evaluate to a location.
  if (asLValue) {
    ASTContext &Ctx = getContext();
    T = Ctx.getPointerType(Ctx.getCanonicalType(T));
    ExTy = Ctx.getPointerType(Ctx.getCanonicalType(ExTy));
  }
#endif

  switch (CastE->getCastKind()) {
  case CK_ToVoid:
    for (ExplodedNodeSet::iterator I = S2.begin(), E = S2.end(); I != E; ++I)
      Dst.Add(*I);
    return;

  case CK_LValueToRValue:
  case CK_NoOp:
  case CK_FunctionToPointerDecay:
    for (ExplodedNodeSet::iterator I = S2.begin(), E = S2.end(); I != E; ++I) {
      // Copy the SVal of Ex to CastE.
      ExplodedNode *N = *I;
      const GRState *state = GetState(N);
      SVal V = state->getSVal(Ex);
      state = state->BindExpr(CastE, V);
      MakeNode(Dst, CastE, N, state);
    }
    return;

  case CK_GetObjCProperty:
  case CK_Dependent:
  case CK_ArrayToPointerDecay:
  case CK_BitCast:
  case CK_LValueBitCast:
  case CK_IntegralCast:
  case CK_NullToPointer:
  case CK_IntegralToPointer:
  case CK_PointerToIntegral:
  case CK_PointerToBoolean:
  case CK_IntegralToBoolean:
  case CK_IntegralToFloating:
  case CK_FloatingToIntegral:
  case CK_FloatingToBoolean:
  case CK_FloatingCast:
  case CK_FloatingRealToComplex:
  case CK_FloatingComplexToReal:
  case CK_FloatingComplexToBoolean:
  case CK_FloatingComplexCast:
  case CK_FloatingComplexToIntegralComplex:
  case CK_IntegralRealToComplex:
  case CK_IntegralComplexToReal:
  case CK_IntegralComplexToBoolean:
  case CK_IntegralComplexCast:
  case CK_IntegralComplexToFloatingComplex:
  case CK_AnyPointerToObjCPointerCast:
  case CK_AnyPointerToBlockPointerCast:
  
  case CK_ObjCObjectLValueCast: {
    // Delegate to SValBuilder to process.
    for (ExplodedNodeSet::iterator I = S2.begin(), E = S2.end(); I != E; ++I) {
      ExplodedNode* N = *I;
      const GRState* state = GetState(N);
      SVal V = state->getSVal(Ex);
      V = svalBuilder.evalCast(V, T, ExTy);
      state = state->BindExpr(CastE, V);
      MakeNode(Dst, CastE, N, state);
    }
    return;
  }

  case CK_DerivedToBase:
  case CK_UncheckedDerivedToBase:
    // For DerivedToBase cast, delegate to the store manager.
    for (ExplodedNodeSet::iterator I = S2.begin(), E = S2.end(); I != E; ++I) {
      ExplodedNode *node = *I;
      const GRState *state = GetState(node);
      SVal val = state->getSVal(Ex);
      val = getStoreManager().evalDerivedToBase(val, T);
      state = state->BindExpr(CastE, val);
      MakeNode(Dst, CastE, node, state);
    }
    return;

  // Various C++ casts that are not handled yet.
  case CK_Dynamic:  
  case CK_ToUnion:
  case CK_BaseToDerived:
  case CK_NullToMemberPointer:
  case CK_BaseToDerivedMemberPointer:
  case CK_DerivedToBaseMemberPointer:
  case CK_UserDefinedConversion:
  case CK_ConstructorConversion:
  case CK_VectorSplat:
  case CK_MemberPointerToBoolean: {
    SaveAndRestore<bool> OldSink(Builder->BuildSinks);
    Builder->BuildSinks = true;
    MakeNode(Dst, CastE, Pred, GetState(Pred));
    return;
  }
  }
}

void ExprEngine::VisitCompoundLiteralExpr(const CompoundLiteralExpr* CL,
                                            ExplodedNode* Pred,
                                            ExplodedNodeSet& Dst) {
  const InitListExpr* ILE 
    = cast<InitListExpr>(CL->getInitializer()->IgnoreParens());
  ExplodedNodeSet Tmp;
  Visit(ILE, Pred, Tmp);

  for (ExplodedNodeSet::iterator I = Tmp.begin(), EI = Tmp.end(); I!=EI; ++I) {
    const GRState* state = GetState(*I);
    SVal ILV = state->getSVal(ILE);
    const LocationContext *LC = (*I)->getLocationContext();
    state = state->bindCompoundLiteral(CL, LC, ILV);

    if (CL->isLValue()) {
      MakeNode(Dst, CL, *I, state->BindExpr(CL, state->getLValue(CL, LC)));
    }
    else
      MakeNode(Dst, CL, *I, state->BindExpr(CL, ILV));
  }
}

void ExprEngine::VisitDeclStmt(const DeclStmt *DS, ExplodedNode *Pred,
                                 ExplodedNodeSet& Dst) {

  // The CFG has one DeclStmt per Decl.
  const Decl* D = *DS->decl_begin();

  if (!D || !isa<VarDecl>(D))
    return;

  const VarDecl* VD = dyn_cast<VarDecl>(D);
  const Expr* InitEx = VD->getInit();

  // FIXME: static variables may have an initializer, but the second
  //  time a function is called those values may not be current.
  ExplodedNodeSet Tmp;

  if (InitEx) {
    if (VD->getType()->isReferenceType() && !InitEx->isLValue()) {
      // If the initializer is C++ record type, it should already has a 
      // temp object.
      if (!InitEx->getType()->isRecordType())
        CreateCXXTemporaryObject(InitEx, Pred, Tmp);
      else
        Tmp.Add(Pred);
    } else
      Visit(InitEx, Pred, Tmp);
  } else
    Tmp.Add(Pred);

  ExplodedNodeSet Tmp2;
  CheckerVisit(DS, Tmp2, Tmp, PreVisitStmtCallback);

  for (ExplodedNodeSet::iterator I=Tmp2.begin(), E=Tmp2.end(); I!=E; ++I) {
    ExplodedNode *N = *I;
    const GRState *state = GetState(N);

    // Decls without InitExpr are not initialized explicitly.
    const LocationContext *LC = N->getLocationContext();

    if (InitEx) {
      SVal InitVal = state->getSVal(InitEx);

      // We bound the temp obj region to the CXXConstructExpr. Now recover
      // the lazy compound value when the variable is not a reference.
      if (AMgr.getLangOptions().CPlusPlus && VD->getType()->isRecordType() && 
          !VD->getType()->isReferenceType() && isa<loc::MemRegionVal>(InitVal)){
        InitVal = state->getSVal(cast<loc::MemRegionVal>(InitVal).getRegion());
        assert(isa<nonloc::LazyCompoundVal>(InitVal));
      }

      // Recover some path-sensitivity if a scalar value evaluated to
      // UnknownVal.
      if ((InitVal.isUnknown() ||
          !getConstraintManager().canReasonAbout(InitVal)) &&
          !VD->getType()->isReferenceType()) {
        InitVal = svalBuilder.getConjuredSymbolVal(NULL, InitEx,
                                               Builder->getCurrentBlockCount());
      }

      evalBind(Dst, DS, *I, state,
               loc::MemRegionVal(state->getRegion(VD, LC)), InitVal, true);
    }
    else {
      state = state->bindDeclWithNoInit(state->getRegion(VD, LC));
      MakeNode(Dst, DS, *I, state);
    }
  }
}

void ExprEngine::VisitCondInit(const VarDecl *VD, const Stmt *S,
                                 ExplodedNode *Pred, ExplodedNodeSet& Dst) {

  const Expr* InitEx = VD->getInit();
  ExplodedNodeSet Tmp;
  Visit(InitEx, Pred, Tmp);

  for (ExplodedNodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {
    ExplodedNode *N = *I;
    const GRState *state = GetState(N);

    const LocationContext *LC = N->getLocationContext();
    SVal InitVal = state->getSVal(InitEx);

    // Recover some path-sensitivity if a scalar value evaluated to
    // UnknownVal.
    if (InitVal.isUnknown() ||
        !getConstraintManager().canReasonAbout(InitVal)) {
      InitVal = svalBuilder.getConjuredSymbolVal(NULL, InitEx,
                                            Builder->getCurrentBlockCount());
    }

    evalBind(Dst, S, N, state,
             loc::MemRegionVal(state->getRegion(VD, LC)), InitVal, true);
  }
}

namespace {
  // This class is used by VisitInitListExpr as an item in a worklist
  // for processing the values contained in an InitListExpr.
class InitListWLItem {
public:
  llvm::ImmutableList<SVal> Vals;
  ExplodedNode* N;
  InitListExpr::const_reverse_iterator Itr;

  InitListWLItem(ExplodedNode* n, llvm::ImmutableList<SVal> vals,
                 InitListExpr::const_reverse_iterator itr)
  : Vals(vals), N(n), Itr(itr) {}
};
}


void ExprEngine::VisitInitListExpr(const InitListExpr* E, ExplodedNode* Pred,
                                     ExplodedNodeSet& Dst) {

  const GRState* state = GetState(Pred);
  QualType T = getContext().getCanonicalType(E->getType());
  unsigned NumInitElements = E->getNumInits();

  if (T->isArrayType() || T->isRecordType() || T->isVectorType()) {
    llvm::ImmutableList<SVal> StartVals = getBasicVals().getEmptySValList();

    // Handle base case where the initializer has no elements.
    // e.g: static int* myArray[] = {};
    if (NumInitElements == 0) {
      SVal V = svalBuilder.makeCompoundVal(T, StartVals);
      MakeNode(Dst, E, Pred, state->BindExpr(E, V));
      return;
    }

    // Create a worklist to process the initializers.
    llvm::SmallVector<InitListWLItem, 10> WorkList;
    WorkList.reserve(NumInitElements);
    WorkList.push_back(InitListWLItem(Pred, StartVals, E->rbegin()));
    InitListExpr::const_reverse_iterator ItrEnd = E->rend();
    assert(!(E->rbegin() == E->rend()));

    // Process the worklist until it is empty.
    while (!WorkList.empty()) {
      InitListWLItem X = WorkList.back();
      WorkList.pop_back();

      ExplodedNodeSet Tmp;
      Visit(*X.Itr, X.N, Tmp);

      InitListExpr::const_reverse_iterator NewItr = X.Itr + 1;

      for (ExplodedNodeSet::iterator NI=Tmp.begin(),NE=Tmp.end();NI!=NE;++NI) {
        // Get the last initializer value.
        state = GetState(*NI);
        SVal InitV = state->getSVal(cast<Expr>(*X.Itr));

        // Construct the new list of values by prepending the new value to
        // the already constructed list.
        llvm::ImmutableList<SVal> NewVals =
          getBasicVals().consVals(InitV, X.Vals);

        if (NewItr == ItrEnd) {
          // Now we have a list holding all init values. Make CompoundValData.
          SVal V = svalBuilder.makeCompoundVal(T, NewVals);

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
    const Expr* Init = E->getInit(0);
    Visit(Init, Pred, Tmp);
    for (ExplodedNodeSet::iterator I=Tmp.begin(), EI=Tmp.end(); I != EI; ++I) {
      state = GetState(*I);
      MakeNode(Dst, E, *I, state->BindExpr(E, state->getSVal(Init)));
    }
    return;
  }

  assert(0 && "unprocessed InitListExpr type");
}

/// VisitSizeOfAlignOfExpr - Transfer function for sizeof(type).
void ExprEngine::VisitSizeOfAlignOfExpr(const SizeOfAlignOfExpr* Ex,
                                          ExplodedNode* Pred,
                                          ExplodedNodeSet& Dst) {
  QualType T = Ex->getTypeOfArgument();
  CharUnits amt;

  if (Ex->isSizeOf()) {
    if (T == getContext().VoidTy) {
      // sizeof(void) == 1 byte.
      amt = CharUnits::One();
    }
    else if (!T->isConstantSizeType()) {
      assert(T->isVariableArrayType() && "Unknown non-constant-sized type.");

      // FIXME: Add support for VLA type arguments, not just VLA expressions.
      // When that happens, we should probably refactor VLASizeChecker's code.
      if (Ex->isArgumentType()) {
        Dst.Add(Pred);
        return;
      }

      // Get the size by getting the extent of the sub-expression.
      // First, visit the sub-expression to find its region.
      const Expr *Arg = Ex->getArgumentExpr();
      ExplodedNodeSet Tmp;
      Visit(Arg, Pred, Tmp);

      for (ExplodedNodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {
        const GRState* state = GetState(*I);
        const MemRegion *MR = state->getSVal(Arg).getAsRegion();

        // If the subexpression can't be resolved to a region, we don't know
        // anything about its size. Just leave the state as is and continue.
        if (!MR) {
          Dst.Add(*I);
          continue;
        }

        // The result is the extent of the VLA.
        SVal Extent = cast<SubRegion>(MR)->getExtent(svalBuilder);
        MakeNode(Dst, Ex, *I, state->BindExpr(Ex, Extent));
      }

      return;
    }
    else if (T->getAs<ObjCObjectType>()) {
      // Some code tries to take the sizeof an ObjCObjectType, relying that
      // the compiler has laid out its representation.  Just report Unknown
      // for these.
      Dst.Add(Pred);
      return;
    }
    else {
      // All other cases.
      amt = getContext().getTypeSizeInChars(T);
    }
  }
  else  // Get alignment of the type.
    amt = getContext().getTypeAlignInChars(T);

  MakeNode(Dst, Ex, Pred,
           GetState(Pred)->BindExpr(Ex,
              svalBuilder.makeIntVal(amt.getQuantity(), Ex->getType())));
}

void ExprEngine::VisitOffsetOfExpr(const OffsetOfExpr* OOE, 
                                     ExplodedNode* Pred, ExplodedNodeSet& Dst) {
  Expr::EvalResult Res;
  if (OOE->Evaluate(Res, getContext()) && Res.Val.isInt()) {
    const APSInt &IV = Res.Val.getInt();
    assert(IV.getBitWidth() == getContext().getTypeSize(OOE->getType()));
    assert(OOE->getType()->isIntegerType());
    assert(IV.isSigned() == OOE->getType()->isSignedIntegerType());
    SVal X = svalBuilder.makeIntVal(IV);
    MakeNode(Dst, OOE, Pred, GetState(Pred)->BindExpr(OOE, X));
    return;
  }
  // FIXME: Handle the case where __builtin_offsetof is not a constant.
  Dst.Add(Pred);
}

void ExprEngine::VisitUnaryOperator(const UnaryOperator* U, 
                                      ExplodedNode* Pred,
                                      ExplodedNodeSet& Dst) {

  switch (U->getOpcode()) {

    default:
      break;

    case UO_Real: {
      const Expr* Ex = U->getSubExpr()->IgnoreParens();
      ExplodedNodeSet Tmp;
      Visit(Ex, Pred, Tmp);

      for (ExplodedNodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {

        // FIXME: We don't have complex SValues yet.
        if (Ex->getType()->isAnyComplexType()) {
          // Just report "Unknown."
          Dst.Add(*I);
          continue;
        }

        // For all other types, UO_Real is an identity operation.
        assert (U->getType() == Ex->getType());
        const GRState* state = GetState(*I);
        MakeNode(Dst, U, *I, state->BindExpr(U, state->getSVal(Ex)));
      }

      return;
    }

    case UO_Imag: {

      const Expr* Ex = U->getSubExpr()->IgnoreParens();
      ExplodedNodeSet Tmp;
      Visit(Ex, Pred, Tmp);

      for (ExplodedNodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {
        // FIXME: We don't have complex SValues yet.
        if (Ex->getType()->isAnyComplexType()) {
          // Just report "Unknown."
          Dst.Add(*I);
          continue;
        }

        // For all other types, UO_Imag returns 0.
        const GRState* state = GetState(*I);
        SVal X = svalBuilder.makeZeroVal(Ex->getType());
        MakeNode(Dst, U, *I, state->BindExpr(U, X));
      }

      return;
    }
      
    case UO_Plus:
      assert(!U->isLValue());
      // FALL-THROUGH.
    case UO_Deref:
    case UO_AddrOf:
    case UO_Extension: {

      // Unary "+" is a no-op, similar to a parentheses.  We still have places
      // where it may be a block-level expression, so we need to
      // generate an extra node that just propagates the value of the
      // subexpression.

      const Expr* Ex = U->getSubExpr()->IgnoreParens();
      ExplodedNodeSet Tmp;
      Visit(Ex, Pred, Tmp);

      for (ExplodedNodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {
        const GRState* state = GetState(*I);
        MakeNode(Dst, U, *I, state->BindExpr(U, state->getSVal(Ex)));
      }

      return;
    }

    case UO_LNot:
    case UO_Minus:
    case UO_Not: {
      assert (!U->isLValue());
      const Expr* Ex = U->getSubExpr()->IgnoreParens();
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
//          V = evalCast(V, DstT);
//
//        if (V.isUnknownOrUndef()) {
//          MakeNode(Dst, U, *I, BindExpr(St, U, V));
//          continue;
//        }

        switch (U->getOpcode()) {
          default:
            assert(false && "Invalid Opcode.");
            break;

          case UO_Not:
            // FIXME: Do we need to handle promotions?
            state = state->BindExpr(U, evalComplement(cast<NonLoc>(V)));
            break;

          case UO_Minus:
            // FIXME: Do we need to handle promotions?
            state = state->BindExpr(U, evalMinus(cast<NonLoc>(V)));
            break;

          case UO_LNot:

            // C99 6.5.3.3: "The expression !E is equivalent to (0==E)."
            //
            //  Note: technically we do "E == 0", but this is the same in the
            //    transfer functions as "0 == E".
            SVal Result;

            if (isa<Loc>(V)) {
              Loc X = svalBuilder.makeNull();
              Result = evalBinOp(state, BO_EQ, cast<Loc>(V), X,
                                 U->getType());
            }
            else {
              nonloc::ConcreteInt X(getBasicVals().getValue(0, Ex->getType()));
              Result = evalBinOp(state, BO_EQ, cast<NonLoc>(V), X,
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
  const Expr* Ex = U->getSubExpr()->IgnoreParens();
  Visit(Ex, Pred, Tmp);

  for (ExplodedNodeSet::iterator I = Tmp.begin(), E = Tmp.end(); I!=E; ++I) {

    const GRState* state = GetState(*I);
    SVal loc = state->getSVal(Ex);

    // Perform a load.
    ExplodedNodeSet Tmp2;
    evalLoad(Tmp2, Ex, *I, state, loc);

    for (ExplodedNodeSet::iterator I2=Tmp2.begin(), E2=Tmp2.end();I2!=E2;++I2) {

      state = GetState(*I2);
      SVal V2_untested = state->getSVal(Ex);

      // Propagate unknown and undefined values.
      if (V2_untested.isUnknownOrUndef()) {
        MakeNode(Dst, U, *I2, state->BindExpr(U, V2_untested));
        continue;
      }
      DefinedSVal V2 = cast<DefinedSVal>(V2_untested);

      // Handle all other values.
      BinaryOperator::Opcode Op = U->isIncrementOp() ? BO_Add
                                                     : BO_Sub;

      // If the UnaryOperator has non-location type, use its type to create the
      // constant value. If the UnaryOperator has location type, create the
      // constant with int type and pointer width.
      SVal RHS;

      if (U->getType()->isAnyPointerType())
        RHS = svalBuilder.makeArrayIndex(1);
      else
        RHS = svalBuilder.makeIntVal(1, U->getType());

      SVal Result = evalBinOp(state, Op, V2, RHS, U->getType());

      // Conjure a new symbol if necessary to recover precision.
      if (Result.isUnknown() || !getConstraintManager().canReasonAbout(Result)){
        DefinedOrUnknownSVal SymVal =
          svalBuilder.getConjuredSymbolVal(NULL, Ex,
                                      Builder->getCurrentBlockCount());
        Result = SymVal;

        // If the value is a location, ++/-- should always preserve
        // non-nullness.  Check if the original value was non-null, and if so
        // propagate that constraint.
        if (Loc::IsLocType(U->getType())) {
          DefinedOrUnknownSVal Constraint =
            svalBuilder.evalEQ(state, V2,svalBuilder.makeZeroVal(U->getType()));

          if (!state->assume(Constraint, true)) {
            // It isn't feasible for the original value to be null.
            // Propagate this constraint.
            Constraint = svalBuilder.evalEQ(state, SymVal,
                                       svalBuilder.makeZeroVal(U->getType()));


            state = state->assume(Constraint, false);
            assert(state);
          }
        }
      }

      // Since the lvalue-to-rvalue conversion is explicit in the AST,
      // we bind an l-value if the operator is prefix and an lvalue (in C++).
      if (U->isLValue())
        state = state->BindExpr(U, loc);
      else
        state = state->BindExpr(U, V2);

      // Perform the store.
      evalStore(Dst, NULL, U, *I2, state, loc, Result);
    }
  }
}

void ExprEngine::VisitAsmStmt(const AsmStmt* A, ExplodedNode* Pred,
                                ExplodedNodeSet& Dst) {
  VisitAsmStmtHelperOutputs(A, A->begin_outputs(), A->end_outputs(), Pred, Dst);
}

void ExprEngine::VisitAsmStmtHelperOutputs(const AsmStmt* A,
                                             AsmStmt::const_outputs_iterator I,
                                             AsmStmt::const_outputs_iterator E,
                                     ExplodedNode* Pred, ExplodedNodeSet& Dst) {
  if (I == E) {
    VisitAsmStmtHelperInputs(A, A->begin_inputs(), A->end_inputs(), Pred, Dst);
    return;
  }

  ExplodedNodeSet Tmp;
  Visit(*I, Pred, Tmp);
  ++I;

  for (ExplodedNodeSet::iterator NI = Tmp.begin(), NE = Tmp.end();NI != NE;++NI)
    VisitAsmStmtHelperOutputs(A, I, E, *NI, Dst);
}

void ExprEngine::VisitAsmStmtHelperInputs(const AsmStmt* A,
                                            AsmStmt::const_inputs_iterator I,
                                            AsmStmt::const_inputs_iterator E,
                                            ExplodedNode* Pred,
                                            ExplodedNodeSet& Dst) {
  if (I == E) {

    // We have processed both the inputs and the outputs.  All of the outputs
    // should evaluate to Locs.  Nuke all of their values.

    // FIXME: Some day in the future it would be nice to allow a "plug-in"
    // which interprets the inline asm and stores proper results in the
    // outputs.

    const GRState* state = GetState(Pred);

    for (AsmStmt::const_outputs_iterator OI = A->begin_outputs(),
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

void ExprEngine::VisitReturnStmt(const ReturnStmt *RS, ExplodedNode *Pred,
                                   ExplodedNodeSet &Dst) {
  ExplodedNodeSet Src;
  if (const Expr *RetE = RS->getRetValue()) {
    // Record the returned expression in the state. It will be used in
    // processCallExit to bind the return value to the call expr.
    {
      static int tag = 0;
      const GRState *state = GetState(Pred);
      state = state->set<ReturnExpr>(RetE);
      Pred = Builder->generateNode(RetE, state, Pred, &tag);
    }
    // We may get a NULL Pred because we generated a cached node.
    if (Pred)
      Visit(RetE, Pred, Src);
  }
  else {
    Src.Add(Pred);
  }

  ExplodedNodeSet CheckedSet;
  CheckerVisit(RS, CheckedSet, Src, PreVisitStmtCallback);

  for (ExplodedNodeSet::iterator I = CheckedSet.begin(), E = CheckedSet.end();
       I != E; ++I) {

    assert(Builder && "StmtNodeBuilder must be defined.");

    Pred = *I;
    unsigned size = Dst.size();

    SaveAndRestore<bool> OldSink(Builder->BuildSinks);
    SaveOr OldHasGen(Builder->hasGeneratedNode);

    getTF().evalReturn(Dst, *this, *Builder, RS, Pred);

    // Handle the case where no nodes where generated.
    if (!Builder->BuildSinks && Dst.size() == size &&
        !Builder->hasGeneratedNode)
      MakeNode(Dst, RS, Pred, GetState(Pred));
  }
}

//===----------------------------------------------------------------------===//
// Transfer functions: Binary operators.
//===----------------------------------------------------------------------===//

void ExprEngine::VisitBinaryOperator(const BinaryOperator* B,
                                       ExplodedNode* Pred,
                                       ExplodedNodeSet& Dst) {
  ExplodedNodeSet Tmp1;
  Expr* LHS = B->getLHS()->IgnoreParens();
  Expr* RHS = B->getRHS()->IgnoreParens();

  Visit(LHS, Pred, Tmp1);
  ExplodedNodeSet Tmp3;

  for (ExplodedNodeSet::iterator I1=Tmp1.begin(), E1=Tmp1.end(); I1!=E1; ++I1) {
    SVal LeftV = GetState(*I1)->getSVal(LHS);
    ExplodedNodeSet Tmp2;
    Visit(RHS, *I1, Tmp2);

    ExplodedNodeSet CheckedSet;
    CheckerVisit(B, CheckedSet, Tmp2, PreVisitStmtCallback);

    // With both the LHS and RHS evaluated, process the operation itself.

    for (ExplodedNodeSet::iterator I2=CheckedSet.begin(), E2=CheckedSet.end();
         I2 != E2; ++I2) {

      const GRState *state = GetState(*I2);
      SVal RightV = state->getSVal(RHS);

      BinaryOperator::Opcode Op = B->getOpcode();

      if (Op == BO_Assign) {
        // EXPERIMENTAL: "Conjured" symbols.
        // FIXME: Handle structs.
        QualType T = RHS->getType();

        if (RightV.isUnknown() ||!getConstraintManager().canReasonAbout(RightV))
        {
          unsigned Count = Builder->getCurrentBlockCount();
          RightV = svalBuilder.getConjuredSymbolVal(NULL, B->getRHS(), Count);
        }

        SVal ExprVal = B->isLValue() ? LeftV : RightV;

        // Simulate the effects of a "store":  bind the value of the RHS
        // to the L-Value represented by the LHS.
        evalStore(Tmp3, B, LHS, *I2, state->BindExpr(B, ExprVal), LeftV,RightV);
        continue;
      }

      if (!B->isAssignmentOp()) {
        // Process non-assignments except commas or short-circuited
        // logical expressions (LAnd and LOr).
        SVal Result = evalBinOp(state, Op, LeftV, RightV, B->getType());

        if (Result.isUnknown()) {
          MakeNode(Tmp3, B, *I2, state);
          continue;
        }

        state = state->BindExpr(B, Result);

        MakeNode(Tmp3, B, *I2, state);
        continue;
      }

      assert (B->isCompoundAssignmentOp());

      switch (Op) {
        default:
          assert(0 && "Invalid opcode for compound assignment.");
        case BO_MulAssign: Op = BO_Mul; break;
        case BO_DivAssign: Op = BO_Div; break;
        case BO_RemAssign: Op = BO_Rem; break;
        case BO_AddAssign: Op = BO_Add; break;
        case BO_SubAssign: Op = BO_Sub; break;
        case BO_ShlAssign: Op = BO_Shl; break;
        case BO_ShrAssign: Op = BO_Shr; break;
        case BO_AndAssign: Op = BO_And; break;
        case BO_XorAssign: Op = BO_Xor; break;
        case BO_OrAssign:  Op = BO_Or;  break;
      }

      // Perform a load (the LHS).  This performs the checks for
      // null dereferences, and so on.
      ExplodedNodeSet Tmp4;
      SVal location = state->getSVal(LHS);
      evalLoad(Tmp4, LHS, *I2, state, location);

      for (ExplodedNodeSet::iterator I4=Tmp4.begin(), E4=Tmp4.end(); I4!=E4;
           ++I4) {
        state = GetState(*I4);
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
        V = svalBuilder.evalCast(V, CLHSTy, LTy);

        // Compute the result of the operation.
        SVal Result = svalBuilder.evalCast(evalBinOp(state, Op, V, RightV, CTy),
                                      B->getType(), CTy);

        // EXPERIMENTAL: "Conjured" symbols.
        // FIXME: Handle structs.

        SVal LHSVal;

        if (Result.isUnknown() ||
            !getConstraintManager().canReasonAbout(Result)) {

          unsigned Count = Builder->getCurrentBlockCount();

          // The symbolic value is actually for the type of the left-hand side
          // expression, not the computation type, as this is the value the
          // LValue on the LHS will bind to.
          LHSVal = svalBuilder.getConjuredSymbolVal(NULL, B->getRHS(), LTy, Count);

          // However, we need to convert the symbol to the computation type.
          Result = svalBuilder.evalCast(LHSVal, CTy, LTy);
        }
        else {
          // The left-hand side may bind to a different value then the
          // computation type.
          LHSVal = svalBuilder.evalCast(Result, LTy, CTy);
        }

        // In C++, assignment and compound assignment operators return an 
        // lvalue.
        if (B->isLValue())
          state = state->BindExpr(B, location);
        else
          state = state->BindExpr(B, Result);

        evalStore(Tmp3, B, LHS, *I4, state, location, LHSVal);
      }
    }
  }

  CheckerVisit(B, Dst, Tmp3, PostVisitStmtCallback);
}

//===----------------------------------------------------------------------===//
// Checker registration/lookup.
//===----------------------------------------------------------------------===//

Checker *ExprEngine::lookupChecker(void *tag) const {
  CheckerMap::const_iterator I = CheckerM.find(tag);
  return (I == CheckerM.end()) ? NULL : Checkers[I->second].second;
}

//===----------------------------------------------------------------------===//
// Visualization.
//===----------------------------------------------------------------------===//

#ifndef NDEBUG
static ExprEngine* GraphPrintCheckerState;
static SourceManager* GraphPrintSourceManager;

namespace llvm {
template<>
struct DOTGraphTraits<ExplodedNode*> :
  public DefaultDOTGraphTraits {

  DOTGraphTraits (bool isSimple=false) : DefaultDOTGraphTraits(isSimple) {}

  // FIXME: Since we do not cache error nodes in ExprEngine now, this does not
  // work.
  static std::string getNodeAttributes(const ExplodedNode* N, void*) {

#if 0
      // FIXME: Replace with a general scheme to tell if the node is
      // an error node.
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
#endif
    return "";
  }

  static std::string getNodeLabel(const ExplodedNode* N, void*){

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

      case ProgramPoint::CallEnterKind:
        Out << "CallEnter";
        break;

      case ProgramPoint::CallExitKind:
        Out << "CallExit";
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

#if 0
            // FIXME: Replace with a general scheme to determine
            // the name of the check.
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
#endif

          break;
        }

        const BlockEdge& E = cast<BlockEdge>(Loc);
        Out << "Edge: (B" << E.getSrc()->getBlockID() << ", B"
            << E.getDst()->getBlockID()  << ')';

        if (const Stmt* T = E.getSrc()->getTerminator()) {

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
            const Stmt* Label = E.getDst()->getLabel();

            if (Label) {
              if (const CaseStmt* C = dyn_cast<CaseStmt>(Label)) {
                Out << "\\lcase ";
                LangOptions LO; // FIXME.
                C->getLHS()->printPretty(Out, 0, PrintingPolicy(LO));

                if (const Stmt* RHS = C->getRHS()) {
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

#if 0
          // FIXME: Replace with a general scheme to determine
          // the name of the check.
        if (GraphPrintCheckerState->isUndefControlFlow(N)) {
          Out << "\\|Control-flow based on\\lUndefined value.\\l";
        }
#endif
      }
    }

    const GRState *state = N->getState();
    Out << "\\|StateID: " << (void*) state
        << " NodeID: " << (void*) N << "\\|";
    state->printDOT(Out, *N->getLocationContext()->getCFG());
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

void ExprEngine::ViewGraph(bool trim) {
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
        ExplodedNode *N = const_cast<ExplodedNode*>(R.getErrorNode());
        if (N) Src.push_back(N);
      }
    }

    ViewGraph(&Src[0], &Src[0]+Src.size());
  }
  else {
    GraphPrintCheckerState = this;
    GraphPrintSourceManager = &getContext().getSourceManager();

    llvm::ViewGraph(*G.roots_begin(), "ExprEngine");

    GraphPrintCheckerState = NULL;
    GraphPrintSourceManager = NULL;
  }
#endif
}

void ExprEngine::ViewGraph(ExplodedNode** Beg, ExplodedNode** End) {
#ifndef NDEBUG
  GraphPrintCheckerState = this;
  GraphPrintSourceManager = &getContext().getSourceManager();

  std::auto_ptr<ExplodedGraph> TrimmedG(G.Trim(Beg, End).first);

  if (!TrimmedG.get())
    llvm::errs() << "warning: Trimmed ExplodedGraph is empty.\n";
  else
    llvm::ViewGraph(*TrimmedG->roots_begin(), "TrimmedExprEngine");

  GraphPrintCheckerState = NULL;
  GraphPrintSourceManager = NULL;
#endif
}
