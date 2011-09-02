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

#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngineBuilders.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ObjCMessage.h"
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
using llvm::APSInt;

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

static inline Selector GetNullarySelector(const char* name, ASTContext &Ctx) {
  IdentifierInfo* II = &Ctx.Idents.get(name);
  return Ctx.Selectors.getSelector(0, &II);
}

//===----------------------------------------------------------------------===//
// Engine construction and deletion.
//===----------------------------------------------------------------------===//

ExprEngine::ExprEngine(AnalysisManager &mgr, bool gcEnabled)
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
    ObjCGCEnabled(gcEnabled), BR(mgr, *this) {
  
  if (mgr.shouldEagerlyTrimExplodedGraph()) {
    // Enable eager node reclaimation when constructing the ExplodedGraph.  
    G.enableNodeReclamation();
  }
}

ExprEngine::~ExprEngine() {
  BR.FlushReports();
  delete [] NSExceptionInstanceRaiseSelectors;
}

//===----------------------------------------------------------------------===//
// Utility methods.
//===----------------------------------------------------------------------===//

const ProgramState *ExprEngine::getInitialState(const LocationContext *InitLoc) {
  const ProgramState *state = StateMgr.getInitialState(InitLoc);

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

      if (const ProgramState *newState = state->assume(*Constraint, true))
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

bool
ExprEngine::doesInvalidateGlobals(const CallOrObjCMessage &callOrMessage) const
{
  if (callOrMessage.isFunctionCall() && !callOrMessage.isCXXCall()) {
    SVal calleeV = callOrMessage.getFunctionCallee();
    if (const FunctionTextRegion *codeR =
          dyn_cast_or_null<FunctionTextRegion>(calleeV.getAsRegion())) {
      
      const FunctionDecl *fd = codeR->getDecl();
      if (const IdentifierInfo *ii = fd->getIdentifier()) {
        StringRef fname = ii->getName();
        if (fname == "strlen")
          return false;
      }
    }
  }
  
  // The conservative answer: invalidates globals.
  return true;
}

//===----------------------------------------------------------------------===//
// Top-level transfer function logic (Dispatcher).
//===----------------------------------------------------------------------===//

/// evalAssume - Called by ConstraintManager. Used to call checker-specific
///  logic for handling assumptions on symbolic values.
const ProgramState *ExprEngine::processAssume(const ProgramState *state,
                                              SVal cond, bool assumption) {
  return getCheckerManager().runCheckersForEvalAssume(state, cond, assumption);
}

bool ExprEngine::wantsRegionChangeUpdate(const ProgramState *state) {
  return getCheckerManager().wantsRegionChangeUpdate(state);
}

const ProgramState *
ExprEngine::processRegionChanges(const ProgramState *state,
                            const StoreManager::InvalidatedSymbols *invalidated,
                                 ArrayRef<const MemRegion *> Explicits,
                                 ArrayRef<const MemRegion *> Regions) {
  return getCheckerManager().runCheckersForRegionChanges(state, invalidated,
                                                         Explicits, Regions);
}

void ExprEngine::printState(raw_ostream &Out, const ProgramState *State,
                            const char *NL, const char *Sep) {
  getCheckerManager().runCheckersForPrintState(Out, State, NL, Sep);
}

void ExprEngine::processEndWorklist(bool hasWorkRemaining) {
  getCheckerManager().runCheckersForEndAnalysis(G, BR, *this);
}

void ExprEngine::processCFGElement(const CFGElement E, 
                                  StmtNodeBuilder& builder) {
  switch (E.getKind()) {
    case CFGElement::Invalid:
      llvm_unreachable("Unexpected CFGElement kind.");
    case CFGElement::Statement:
      ProcessStmt(const_cast<Stmt*>(E.getAs<CFGStmt>()->getStmt()), builder);
      return;
    case CFGElement::Initializer:
      ProcessInitializer(E.getAs<CFGInitializer>()->getInitializer(), builder);
      return;
    case CFGElement::AutomaticObjectDtor:
    case CFGElement::BaseDtor:
    case CFGElement::MemberDtor:
    case CFGElement::TemporaryDtor:
      ProcessImplicitDtor(*E.getAs<CFGImplicitDtor>(), builder);
      return;
  }
}

void ExprEngine::ProcessStmt(const CFGStmt S, StmtNodeBuilder& builder) {
  // TODO: Use RAII to remove the unnecessary, tagged nodes.
  //RegisterCreatedNodes registerCreatedNodes(getGraph());

  // Reclaim any unnecessary nodes in the ExplodedGraph.
  G.reclaimRecentlyAllocatedNodes();
  // Recycle any unused states in the ProgramStateManager.
  StateMgr.recycleUnusedStates();
  
  currentStmt = S.getStmt();
  PrettyStackTraceLoc CrashInfo(getContext().getSourceManager(),
                                currentStmt->getLocStart(),
                                "Error evaluating statement");

  // A tag to track convenience transitions, which can be removed at cleanup.
  static SimpleProgramPointTag cleanupTag("ExprEngine : Clean Node");
  Builder = &builder;
  EntryNode = builder.getPredecessor();

  const ProgramState *EntryState = EntryNode->getState();
  CleanedState = EntryState;
  ExplodedNode *CleanedNode = 0;

  // Create the cleaned state.
  const LocationContext *LC = EntryNode->getLocationContext();
  SymbolReaper SymReaper(LC, currentStmt, SymMgr, getStoreManager());

  if (AMgr.shouldPurgeDead()) {
    getCheckerManager().runCheckersForLiveSymbols(CleanedState, SymReaper);

    const StackFrameContext *SFC = LC->getCurrentStackFrame();

    // Create a state in which dead bindings are removed from the environment
    // and the store. TODO: The function should just return new env and store,
    // not a new state.
    CleanedState = StateMgr.removeDeadBindings(CleanedState, SFC, SymReaper);
  }

  // Process any special transfer function for dead symbols.
  ExplodedNodeSet Tmp;
  if (!SymReaper.hasDeadSymbols()) {
    // Generate a CleanedNode that has the environment and store cleaned
    // up. Since no symbols are dead, we can optimize and not clean out
    // the constraint manager.
    CleanedNode =
      Builder->generateNode(currentStmt, CleanedState, EntryNode, &cleanupTag);
    Tmp.Add(CleanedNode);

  } else {
    SaveAndRestore<bool> OldSink(Builder->BuildSinks);
    SaveOr OldHasGen(Builder->hasGeneratedNode);

    SaveAndRestore<bool> OldPurgeDeadSymbols(Builder->PurgingDeadSymbols);
    Builder->PurgingDeadSymbols = true;

    // Call checkers with the non-cleaned state so that they could query the
    // values of the soon to be dead symbols.
    ExplodedNodeSet CheckedSet;
    getCheckerManager().runCheckersForDeadSymbols(CheckedSet, EntryNode,
                                                 SymReaper, currentStmt, *this);

    // For each node in CheckedSet, generate CleanedNodes that have the
    // environment, the store, and the constraints cleaned up but have the
    // user-supplied states as the predecessors.
    for (ExplodedNodeSet::const_iterator
          I = CheckedSet.begin(), E = CheckedSet.end(); I != E; ++I) {
      const ProgramState *CheckerState = (*I)->getState();

      // The constraint manager has not been cleaned up yet, so clean up now.
      CheckerState = getConstraintManager().removeDeadBindings(CheckerState,
                                                               SymReaper);

      assert(StateMgr.haveEqualEnvironments(CheckerState, EntryState) &&
        "Checkers are not allowed to modify the Environment as a part of "
        "checkDeadSymbols processing.");
      assert(StateMgr.haveEqualStores(CheckerState, EntryState) &&
        "Checkers are not allowed to modify the Store as a part of "
        "checkDeadSymbols processing.");

      // Create a state based on CleanedState with CheckerState GDM and
      // generate a transition to that state.
      const ProgramState *CleanedCheckerSt =
        StateMgr.getPersistentStateWithGDM(CleanedState, CheckerState);
      ExplodedNode *CleanedNode = Builder->generateNode(currentStmt,
                                                        CleanedCheckerSt, *I,
                                                        &cleanupTag);
      Tmp.Add(CleanedNode);
    }
  }

  for (ExplodedNodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {
    // TODO: Remove Dest set, it's no longer needed.
    ExplodedNodeSet Dst;
    // Visit the statement.
    Visit(currentStmt, *I, Dst);
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
      const ProgramState *state = Pred->getState();

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

  switch (D.getKind()) {
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
  const ProgramState *state = pred->getState();
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

void ExprEngine::Visit(const Stmt *S, ExplodedNode *Pred, 
                         ExplodedNodeSet &Dst) {
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
    // C++ and ARC stuff we don't support yet.
    case Expr::ObjCIndirectCopyRestoreExprClass:
    case Stmt::CXXBindTemporaryExprClass:
    case Stmt::CXXCatchStmtClass:
    case Stmt::CXXDependentScopeMemberExprClass:
    case Stmt::CXXForRangeStmtClass:
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
    case Stmt::ArrayTypeTraitExprClass:
    case Stmt::ExpressionTraitExprClass:
    case Stmt::UnresolvedLookupExprClass:
    case Stmt::UnresolvedMemberExprClass:
    case Stmt::CXXNoexceptExprClass:
    case Stmt::PackExpansionExprClass:
    case Stmt::SubstNonTypeTemplateParmPackExprClass:
    case Stmt::SEHTryStmtClass:
    case Stmt::SEHExceptStmtClass:
    case Stmt::SEHFinallyStmtClass:
    {
      SaveAndRestore<bool> OldSink(Builder->BuildSinks);
      Builder->BuildSinks = true;
      const ExplodedNode *node = MakeNode(Dst, S, Pred, Pred->getState());
      Engine.addAbortedBlock(node, Builder->getBlock());
      break;
    }
    
    // We don't handle default arguments either yet, but we can fake it
    // for now by just skipping them.
    case Stmt::SubstNonTypeTemplateParmExprClass:
    case Stmt::CXXDefaultArgExprClass: {
      Dst.Add(Pred);
      break;
    }

    case Stmt::ParenExprClass:
      llvm_unreachable("ParenExprs already handled.");
    case Stmt::GenericSelectionExprClass:
      llvm_unreachable("GenericSelectionExprs already handled.");
    // Cases that should never be evaluated simply because they shouldn't
    // appear in the CFG.
    case Stmt::BreakStmtClass:
    case Stmt::CaseStmtClass:
    case Stmt::CompoundStmtClass:
    case Stmt::ContinueStmtClass:
    case Stmt::DefaultStmtClass:
    case Stmt::DoStmtClass:
    case Stmt::ForStmtClass:
    case Stmt::GotoStmtClass:
    case Stmt::IfStmtClass:
    case Stmt::IndirectGotoStmtClass:
    case Stmt::LabelStmtClass:
    case Stmt::NoStmtClass:
    case Stmt::NullStmtClass:
    case Stmt::SwitchStmtClass:
    case Stmt::WhileStmtClass:
      llvm_unreachable("Stmt should not be in analyzer evaluation loop");
      break;

    case Stmt::GNUNullExprClass: {
      // GNU __null is a pointer-width integer, not an actual pointer.
      const ProgramState *state = Pred->getState();
      state = state->BindExpr(S, svalBuilder.makeIntValWithPtrWidth(0, false));
      MakeNode(Dst, S, Pred, state);
      break;
    }

    case Stmt::ObjCAtSynchronizedStmtClass:
      VisitObjCAtSynchronizedStmt(cast<ObjCAtSynchronizedStmt>(S), Pred, Dst);
      break;

    case Stmt::ObjCPropertyRefExprClass:
      // Implicitly handled by Environment::getSVal().
      Dst.Add(Pred);
      break;

    case Stmt::ImplicitValueInitExprClass: {
      const ProgramState *state = Pred->getState();
      QualType ty = cast<ImplicitValueInitExpr>(S)->getType();
      SVal val = svalBuilder.makeZeroVal(ty);
      MakeNode(Dst, S, Pred, state->BindExpr(S, val));
      break;
    }
      
    case Stmt::ExprWithCleanupsClass: {
      Visit(cast<ExprWithCleanups>(S)->getSubExpr(), Pred, Dst);
      break;
    }

    // Cases not handled yet; but will handle some day.
    case Stmt::DesignatedInitExprClass:
    case Stmt::ExtVectorElementExprClass:
    case Stmt::ImaginaryLiteralClass:
    case Stmt::ObjCAtCatchStmtClass:
    case Stmt::ObjCAtFinallyStmtClass:
    case Stmt::ObjCAtTryStmtClass:
    case Stmt::ObjCAutoreleasePoolStmtClass:
    case Stmt::ObjCEncodeExprClass:
    case Stmt::ObjCIsaExprClass:
    case Stmt::ObjCProtocolExprClass:
    case Stmt::ObjCSelectorExprClass:
    case Stmt::ObjCStringLiteralClass:
    case Stmt::ParenListExprClass:
    case Stmt::PredefinedExprClass:
    case Stmt::ShuffleVectorExprClass:
    case Stmt::VAArgExprClass:
    case Stmt::CUDAKernelCallExprClass:
    case Stmt::OpaqueValueExprClass:
    case Stmt::AsTypeExprClass:
        // Fall through.

    // Cases we intentionally don't evaluate, since they don't need
    // to be explicitly evaluated.
    case Stmt::AddrLabelExprClass:
    case Stmt::IntegerLiteralClass:
    case Stmt::CharacterLiteralClass:
    case Stmt::CXXBoolLiteralExprClass:
    case Stmt::FloatingLiteralClass:
    case Stmt::SizeOfPackExprClass:
    case Stmt::CXXNullPtrLiteralExprClass:
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
        const ProgramState *state = Pred->getState();
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

    case Stmt::CallExprClass:
    case Stmt::CXXOperatorCallExprClass:
    case Stmt::CXXMemberCallExprClass: {
      VisitCallExpr(cast<CallExpr>(S), Pred, Dst);
      break;
    }

    case Stmt::CXXConstructExprClass: {
      const CXXConstructExpr *C = cast<CXXConstructExpr>(S);
      // For block-level CXXConstructExpr, we don't have a destination region.
      // Let VisitCXXConstructExpr() create one.
      VisitCXXConstructExpr(C, 0, Pred, Dst);
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
      const ChooseExpr *C = cast<ChooseExpr>(S);
      VisitGuardedExpr(C, C->getLHS(), C->getRHS(), Pred, Dst);
      break;
    }

    case Stmt::CompoundAssignOperatorClass:
      VisitBinaryOperator(cast<BinaryOperator>(S), Pred, Dst);
      break;

    case Stmt::CompoundLiteralExprClass:
      VisitCompoundLiteralExpr(cast<CompoundLiteralExpr>(S), Pred, Dst);
      break;

    case Stmt::BinaryConditionalOperatorClass:
    case Stmt::ConditionalOperatorClass: { // '?' operator
      const AbstractConditionalOperator *C
        = cast<AbstractConditionalOperator>(S);
      VisitGuardedExpr(C, C->getTrueExpr(), C->getFalseExpr(), Pred, Dst);
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

    case Stmt::ImplicitCastExprClass:
    case Stmt::CStyleCastExprClass:
    case Stmt::CXXStaticCastExprClass:
    case Stmt::CXXDynamicCastExprClass:
    case Stmt::CXXReinterpretCastExprClass:
    case Stmt::CXXConstCastExprClass:
    case Stmt::CXXFunctionalCastExprClass: 
    case Stmt::ObjCBridgedCastExprClass: {
      const CastExpr *C = cast<CastExpr>(S);
      // Handle the previsit checks.
      ExplodedNodeSet dstPrevisit;
      getCheckerManager().runCheckersForPreStmt(dstPrevisit, Pred, C, *this);
      
      // Handle the expression itself.
      ExplodedNodeSet dstExpr;
      for (ExplodedNodeSet::iterator i = dstPrevisit.begin(),
                                     e = dstPrevisit.end(); i != e ; ++i) { 
        VisitCast(C, C->getSubExpr(), *i, dstExpr);
      }

      // Handle the postvisit checks.
      getCheckerManager().runCheckersForPostStmt(Dst, dstExpr, C, *this);
      break;
    }

    case Expr::MaterializeTemporaryExprClass: {
      const MaterializeTemporaryExpr *Materialize
                                            = cast<MaterializeTemporaryExpr>(S);
      if (!Materialize->getType()->isRecordType())
        CreateCXXTemporaryObject(Materialize, Pred, Dst);
      else
        Visit(Materialize->GetTemporaryExpr(), Pred, Dst);
      break;
    }
      
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
      VisitObjCMessage(cast<ObjCMessageExpr>(S), Pred, Dst);
      break;

    case Stmt::ObjCAtThrowStmtClass: {
      // FIXME: This is not complete.  We basically treat @throw as
      // an abort.
      SaveAndRestore<bool> OldSink(Builder->BuildSinks);
      Builder->BuildSinks = true;
      MakeNode(Dst, S, Pred, Pred->getState());
      break;
    }

    case Stmt::ReturnStmtClass:
      VisitReturnStmt(cast<ReturnStmt>(S), Pred, Dst);
      break;

    case Stmt::OffsetOfExprClass:
      VisitOffsetOfExpr(cast<OffsetOfExpr>(S), Pred, Dst);
      break;

    case Stmt::UnaryExprOrTypeTraitExprClass:
      VisitUnaryExprOrTypeTraitExpr(cast<UnaryExprOrTypeTraitExpr>(S),
                                    Pred, Dst);
      break;

    case Stmt::StmtExprClass: {
      const StmtExpr *SE = cast<StmtExpr>(S);

      if (SE->getSubStmt()->body_empty()) {
        // Empty statement expression.
        assert(SE->getType() == getContext().VoidTy
               && "Empty statement expression must have void type.");
        Dst.Add(Pred);
        break;
      }

      if (Expr *LastExpr = dyn_cast<Expr>(*SE->getSubStmt()->body_rbegin())) {
        const ProgramState *state = Pred->getState();
        MakeNode(Dst, SE, Pred, state->BindExpr(SE, state->getSVal(LastExpr)));
      }
      else
        Dst.Add(Pred);

      break;
    }

    case Stmt::StringLiteralClass: {
      const ProgramState *state = Pred->getState();
      SVal V = state->getLValue(cast<StringLiteral>(S));
      MakeNode(Dst, S, Pred, state->BindExpr(S, V));
      return;
    }

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
    static SimpleProgramPointTag tag("ExprEngine : Block count exceeded");
    nodeBuilder.generateNode(pred->getState(), pred, &tag, true);
  }
}

//===----------------------------------------------------------------------===//
// Generic node creation.
//===----------------------------------------------------------------------===//

ExplodedNode *ExprEngine::MakeNode(ExplodedNodeSet &Dst, const Stmt *S,
                                   ExplodedNode *Pred, const ProgramState *St,
                                   ProgramPoint::Kind K,
                                   const ProgramPointTag *tag) {
  assert (Builder && "StmtNodeBuilder not present.");
  SaveAndRestore<const ProgramPointTag*> OldTag(Builder->Tag);
  Builder->Tag = tag;
  return Builder->MakeNode(Dst, S, Pred, St, K);
}

//===----------------------------------------------------------------------===//
// Branch processing.
//===----------------------------------------------------------------------===//

const ProgramState *ExprEngine::MarkBranch(const ProgramState *state,
                                        const Stmt *Terminator,
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

      const Expr *Ex = (Op == BO_LAnd && branchTaken) ||
                       (Op == BO_LOr && !branchTaken)
                       ? B->getRHS() : B->getLHS();

      return state->BindExpr(B, UndefinedVal(Ex));
    }

    case Stmt::BinaryConditionalOperatorClass:
    case Stmt::ConditionalOperatorClass: { // ?:
      const AbstractConditionalOperator* C
        = cast<AbstractConditionalOperator>(Terminator);

      // For ?, if branchTaken == true then the value is either the LHS or
      // the condition itself. (GNU extension).

      const Expr *Ex;

      if (branchTaken)
        Ex = C->getTrueExpr();
      else
        Ex = C->getFalseExpr();

      return state->BindExpr(C, UndefinedVal(Ex));
    }

    case Stmt::ChooseExprClass: { // ?:

      const ChooseExpr *C = cast<ChooseExpr>(Terminator);

      const Expr *Ex = branchTaken ? C->getLHS() : C->getRHS();
      return state->BindExpr(C, UndefinedVal(Ex));
    }
  }
}

/// RecoverCastedSymbol - A helper function for ProcessBranch that is used
/// to try to recover some path-sensitivity for casts of symbolic
/// integers that promote their values (which are currently not tracked well).
/// This function returns the SVal bound to Condition->IgnoreCasts if all the
//  cast(s) did was sign-extend the original value.
static SVal RecoverCastedSymbol(ProgramStateManager& StateMgr,
                                const ProgramState *state,
                                const Stmt *Condition,
                                ASTContext &Ctx) {

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

void ExprEngine::processBranch(const Stmt *Condition, const Stmt *Term,
                                 BranchNodeBuilder& builder) {

  // Check for NULL conditions; e.g. "for(;;)"
  if (!Condition) {
    builder.markInfeasible(false);
    return;
  }

  PrettyStackTraceLoc CrashInfo(getContext().getSourceManager(),
                                Condition->getLocStart(),
                                "Error evaluating branch");

  getCheckerManager().runCheckersForBranchCondition(Condition, builder, *this);

  // If the branch condition is undefined, return;
  if (!builder.isFeasible(true) && !builder.isFeasible(false))
    return;

  const ProgramState *PrevState = builder.getState();
  SVal X = PrevState->getSVal(Condition);

  if (X.isUnknownOrUndef()) {
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
    if (X.isUnknownOrUndef()) {
      builder.generateNode(MarkBranch(PrevState, Term, true), true);
      builder.generateNode(MarkBranch(PrevState, Term, false), false);
      return;
    }
  }

  DefinedSVal V = cast<DefinedSVal>(X);

  // Process the true branch.
  if (builder.isFeasible(true)) {
    if (const ProgramState *state = PrevState->assume(V, true))
      builder.generateNode(MarkBranch(state, Term, true), true);
    else
      builder.markInfeasible(true);
  }

  // Process the false branch.
  if (builder.isFeasible(false)) {
    if (const ProgramState *state = PrevState->assume(V, false))
      builder.generateNode(MarkBranch(state, Term, false), false);
    else
      builder.markInfeasible(false);
  }
}

/// processIndirectGoto - Called by CoreEngine.  Used to generate successor
///  nodes by processing the 'effects' of a computed goto jump.
void ExprEngine::processIndirectGoto(IndirectGotoNodeBuilder &builder) {

  const ProgramState *state = builder.getState();
  SVal V = state->getSVal(builder.getTarget());

  // Three possibilities:
  //
  //   (1) We know the computed label.
  //   (2) The label is NULL (or some other constant), or Undefined.
  //   (3) We have no clue about the label.  Dispatch to all targets.
  //

  typedef IndirectGotoNodeBuilder::iterator iterator;

  if (isa<loc::GotoLabel>(V)) {
    const LabelDecl *L = cast<loc::GotoLabel>(V).getLabel();

    for (iterator I = builder.begin(), E = builder.end(); I != E; ++I) {
      if (I.getLabel() == L) {
        builder.generateNode(I, state);
        return;
      }
    }

    assert(false && "No block with label.");
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

/// ProcessEndPath - Called by CoreEngine.  Used to generate end-of-path
///  nodes when the control reaches the end of a function.
void ExprEngine::processEndOfFunction(EndOfFunctionNodeBuilder& builder) {
  StateMgr.EndPath(builder.getState());
  getCheckerManager().runCheckersForEndPath(builder, *this);
}

/// ProcessSwitch - Called by CoreEngine.  Used to generate successor
///  nodes by processing the 'effects' of a switch statement.
void ExprEngine::processSwitch(SwitchNodeBuilder& builder) {
  typedef SwitchNodeBuilder::iterator iterator;
  const ProgramState *state = builder.getState();
  const Expr *CondE = builder.getCondition();
  SVal  CondV_untested = state->getSVal(CondE);

  if (CondV_untested.isUndef()) {
    //ExplodedNode* N = builder.generateDefaultCaseNode(state, true);
    // FIXME: add checker
    //UndefBranches.insert(N);

    return;
  }
  DefinedOrUnknownSVal CondV = cast<DefinedOrUnknownSVal>(CondV_untested);

  const ProgramState *DefaultSt = state;
  
  iterator I = builder.begin(), EI = builder.end();
  bool defaultIsFeasible = I == EI;

  for ( ; I != EI; ++I) {
    // Successor may be pruned out during CFG construction.
    if (!I.getBlock())
      continue;
    
    const CaseStmt *Case = I.getCase();

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

    if (const Expr *E = Case->getRHS()) {
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
      if (const ProgramState *stateNew = state->assume(Res, true)) {
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
        if (const ProgramState *stateNew = DefaultSt->assume(Res, false)) {
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

//===----------------------------------------------------------------------===//
// Transfer functions: Loads and stores.
//===----------------------------------------------------------------------===//

void ExprEngine::VisitCommonDeclRefExpr(const Expr *Ex, const NamedDecl *D,
                                        ExplodedNode *Pred,
                                        ExplodedNodeSet &Dst) {
  const ProgramState *state = Pred->getState();

  if (const VarDecl *VD = dyn_cast<VarDecl>(D)) {
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
  if (const EnumConstantDecl *ED = dyn_cast<EnumConstantDecl>(D)) {
    assert(!Ex->isLValue());
    SVal V = svalBuilder.makeIntVal(ED->getInitVal());
    MakeNode(Dst, Ex, Pred, state->BindExpr(Ex, V));
    return;
  }
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    SVal V = svalBuilder.getFunctionPointer(FD);
    MakeNode(Dst, Ex, Pred, state->BindExpr(Ex, V),
             ProgramPoint::PostLValueKind);
    return;
  }
  assert (false &&
          "ValueDecl support for this ValueDecl not implemented.");
}

/// VisitArraySubscriptExpr - Transfer function for array accesses
void ExprEngine::VisitLvalArraySubscriptExpr(const ArraySubscriptExpr *A,
                                             ExplodedNode *Pred,
                                             ExplodedNodeSet &Dst){

  const Expr *Base = A->getBase()->IgnoreParens();
  const Expr *Idx  = A->getIdx()->IgnoreParens();
  

  ExplodedNodeSet checkerPreStmt;
  getCheckerManager().runCheckersForPreStmt(checkerPreStmt, Pred, A, *this);

  for (ExplodedNodeSet::iterator it = checkerPreStmt.begin(),
                                 ei = checkerPreStmt.end(); it != ei; ++it) {
    const ProgramState *state = (*it)->getState();
    SVal V = state->getLValue(A->getType(), state->getSVal(Idx),
                              state->getSVal(Base));
    assert(A->isLValue());
    MakeNode(Dst, A, *it, state->BindExpr(A, V), ProgramPoint::PostLValueKind);
  }
}

/// VisitMemberExpr - Transfer function for member expressions.
void ExprEngine::VisitMemberExpr(const MemberExpr *M, ExplodedNode *Pred,
                                 ExplodedNodeSet &Dst) {

  Decl *member = M->getMemberDecl();
  if (VarDecl *VD = dyn_cast<VarDecl>(member)) {
    assert(M->isLValue());
    VisitCommonDeclRefExpr(M, VD, Pred, Dst);
    return;
  }
  
  FieldDecl *field = dyn_cast<FieldDecl>(member);
  if (!field) // FIXME: skipping member expressions for non-fields
    return;

  Expr *baseExpr = M->getBase()->IgnoreParens();
  const ProgramState *state = Pred->getState();
  SVal baseExprVal = state->getSVal(baseExpr);
  if (isa<nonloc::LazyCompoundVal>(baseExprVal) ||
      isa<nonloc::CompoundVal>(baseExprVal) ||
      // FIXME: This can originate by conjuring a symbol for an unknown
      // temporary struct object, see test/Analysis/fields.c:
      // (p = getit()).x
      isa<nonloc::SymbolVal>(baseExprVal)) {
    MakeNode(Dst, M, Pred, state->BindExpr(M, UnknownVal()));
    return;
  }

  // FIXME: Should we insert some assumption logic in here to determine
  // if "Base" is a valid piece of memory?  Before we put this assumption
  // later when using FieldOffset lvals (which we no longer have).

  // For all other cases, compute an lvalue.    
  SVal L = state->getLValue(field, baseExprVal);
  if (M->isLValue())
    MakeNode(Dst, M, Pred, state->BindExpr(M, L), ProgramPoint::PostLValueKind);
  else
    evalLoad(Dst, M, Pred, state, L);
}

/// evalBind - Handle the semantics of binding a value to a specific location.
///  This method is used by evalStore and (soon) VisitDeclStmt, and others.
void ExprEngine::evalBind(ExplodedNodeSet &Dst, const Stmt *StoreE,
                          ExplodedNode *Pred,
                          SVal location, SVal Val, bool atDeclInit) {

  // Do a previsit of the bind.
  ExplodedNodeSet CheckedSet;
  getCheckerManager().runCheckersForBind(CheckedSet, Pred, location, Val,
                                         StoreE, *this);

  for (ExplodedNodeSet::iterator I = CheckedSet.begin(), E = CheckedSet.end();
       I!=E; ++I) {

    const ProgramState *state = (*I)->getState();

    if (atDeclInit) {
      const VarRegion *VR =
        cast<VarRegion>(cast<loc::MemRegionVal>(location).getRegion());

      state = state->bindDecl(VR, Val);
    } else {
      state = state->bindLoc(location, Val);
    }

    MakeNode(Dst, StoreE, *I, state);
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
void ExprEngine::evalStore(ExplodedNodeSet &Dst, const Expr *AssignE,
                             const Expr *LocationE,
                             ExplodedNode *Pred,
                             const ProgramState *state, SVal location, SVal Val,
                             const ProgramPointTag *tag) {

  assert(Builder && "StmtNodeBuilder must be defined.");

  // Proceed with the store.  We use AssignE as the anchor for the PostStore
  // ProgramPoint if it is non-NULL, and LocationE otherwise.
  const Expr *StoreE = AssignE ? AssignE : LocationE;

  if (isa<loc::ObjCPropRef>(location)) {
    loc::ObjCPropRef prop = cast<loc::ObjCPropRef>(location);
    return VisitObjCMessage(ObjCPropertySetter(prop.getPropRefExpr(),
                                               StoreE, Val), Pred, Dst);
  }

  // Evaluate the location (checks for bad dereferences).
  ExplodedNodeSet Tmp;
  evalLocation(Tmp, LocationE, Pred, state, location, tag, false);

  if (Tmp.empty())
    return;

  if (location.isUndef())
    return;

  SaveAndRestore<ProgramPoint::Kind> OldSPointKind(Builder->PointKind,
                                                   ProgramPoint::PostStoreKind);

  for (ExplodedNodeSet::iterator NI=Tmp.begin(), NE=Tmp.end(); NI!=NE; ++NI)
    evalBind(Dst, StoreE, *NI, location, Val);
}

void ExprEngine::evalLoad(ExplodedNodeSet &Dst, const Expr *Ex,
                            ExplodedNode *Pred,
                            const ProgramState *state, SVal location,
                            const ProgramPointTag *tag, QualType LoadTy) {
  assert(!isa<NonLoc>(location) && "location cannot be a NonLoc.");

  if (isa<loc::ObjCPropRef>(location)) {
    loc::ObjCPropRef prop = cast<loc::ObjCPropRef>(location);
    return VisitObjCMessage(ObjCPropertyGetter(prop.getPropRefExpr(), Ex),
                            Pred, Dst);
  }

  // Are we loading from a region?  This actually results in two loads; one
  // to fetch the address of the referenced value and one to fetch the
  // referenced value.
  if (const TypedValueRegion *TR =
        dyn_cast_or_null<TypedValueRegion>(location.getAsRegion())) {

    QualType ValTy = TR->getValueType();
    if (const ReferenceType *RT = ValTy->getAs<ReferenceType>()) {
      static SimpleProgramPointTag
             loadReferenceTag("ExprEngine : Load Reference");
      ExplodedNodeSet Tmp;
      evalLoadCommon(Tmp, Ex, Pred, state, location, &loadReferenceTag,
                     getContext().getPointerType(RT->getPointeeType()));

      // Perform the load from the referenced value.
      for (ExplodedNodeSet::iterator I=Tmp.begin(), E=Tmp.end() ; I!=E; ++I) {
        state = (*I)->getState();
        location = state->getSVal(Ex);
        evalLoadCommon(Dst, Ex, *I, state, location, tag, LoadTy);
      }
      return;
    }
  }

  evalLoadCommon(Dst, Ex, Pred, state, location, tag, LoadTy);
}

void ExprEngine::evalLoadCommon(ExplodedNodeSet &Dst, const Expr *Ex,
                                  ExplodedNode *Pred,
                                  const ProgramState *state, SVal location,
                                  const ProgramPointTag *tag, QualType LoadTy) {

  // Evaluate the location (checks for bad dereferences).
  ExplodedNodeSet Tmp;
  evalLocation(Tmp, Ex, Pred, state, location, tag, true);

  if (Tmp.empty())
    return;

  if (location.isUndef())
    return;

  SaveAndRestore<ProgramPoint::Kind> OldSPointKind(Builder->PointKind);

  // Proceed with the load.
  for (ExplodedNodeSet::iterator NI=Tmp.begin(), NE=Tmp.end(); NI!=NE; ++NI) {
    state = (*NI)->getState();

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
                                ExplodedNode *Pred,
                                const ProgramState *state, SVal location,
                                const ProgramPointTag *tag, bool isLoad) {
  // Early checks for performance reason.
  if (location.isUnknown()) {
    Dst.Add(Pred);
    return;
  }

  ExplodedNodeSet Src;
  if (Pred->getState() == state) {
    Src.Add(Pred);
  } else {
    // Associate this new state with an ExplodedNode.
    // FIXME: If I pass null tag, the graph is incorrect, e.g for
    //   int *p;
    //   p = 0;
    //   *p = 0xDEADBEEF;
    // "p = 0" is not noted as "Null pointer value stored to 'p'" but
    // instead "int *p" is noted as
    // "Variable 'p' initialized to a null pointer value"
    
    // FIXME: why is 'tag' not used instead of etag?
    static SimpleProgramPointTag etag("ExprEngine: Location");

    ExplodedNode *N = Builder->generateNode(S, state, Pred, &etag);
    Src.Add(N ? N : Pred);
  }
  getCheckerManager().runCheckersForLocation(Dst, Src, location, isLoad, S,
                                             *this);
}

bool ExprEngine::InlineCall(ExplodedNodeSet &Dst, const CallExpr *CE, 
                              ExplodedNode *Pred) {
  return false;
  
  // Inlining isn't correct right now because we:
  // (a) don't generate CallExit nodes.
  // (b) we need a way to postpone doing post-visits of CallExprs until
  // the CallExit.  This means we need CallExits for the non-inline
  // cases as well.
  
#if 0
  const ProgramState *state = Pred->getState();
  const Expr *Callee = CE->getCallee();
  SVal L = state->getSVal(Callee);
  
  const FunctionDecl *FD = L.getAsFunctionDecl();
  if (!FD)
    return false;

  // Specially handle CXXMethods.
  const CXXMethodDecl *methodDecl = 0;
  
  switch (CE->getStmtClass()) {
    default: break;
    case Stmt::CXXOperatorCallExprClass: {
      const CXXOperatorCallExpr *opCall = cast<CXXOperatorCallExpr>(CE);
      methodDecl = 
        dyn_cast_or_null<CXXMethodDecl>(opCall->getCalleeDecl());
      break;
    }
    case Stmt::CXXMemberCallExprClass: {
      const CXXMemberCallExpr *memberCall = cast<CXXMemberCallExpr>(CE);
      const MemberExpr *memberExpr = 
        cast<MemberExpr>(memberCall->getCallee()->IgnoreParens());
      methodDecl = cast<CXXMethodDecl>(memberExpr->getMemberDecl());
      break;
    }
  }
      
  
  
  
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
  
  // Generate the CallExit node.

  return false;
#endif
}

std::pair<const ProgramPointTag *, const ProgramPointTag*>
ExprEngine::getEagerlyAssumeTags() {
  static SimpleProgramPointTag
         EagerlyAssumeTrue("ExprEngine : Eagerly Assume True"),
         EagerlyAssumeFalse("ExprEngine : Eagerly Assume False");
  return std::make_pair(&EagerlyAssumeTrue, &EagerlyAssumeFalse);
}

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

    const ProgramState *state = Pred->getState();
    SVal V = state->getSVal(Ex);
    if (nonloc::SymExprVal *SEV = dyn_cast<nonloc::SymExprVal>(&V)) {
      const std::pair<const ProgramPointTag *, const ProgramPointTag*> &tags =
        getEagerlyAssumeTags();

      // First assume that the condition is true.
      if (const ProgramState *StateTrue = state->assume(*SEV, true)) {
        SVal Val = svalBuilder.makeIntVal(1U, Ex->getType());        
        StateTrue = StateTrue->BindExpr(Ex, Val);
        Dst.Add(Builder->generateNode(Ex, StateTrue, Pred, tags.first));
      }

      // Next, assume that the condition is false.
      if (const ProgramState *StateFalse = state->assume(*SEV, false)) {
        SVal Val = svalBuilder.makeIntVal(0U, Ex->getType());
        StateFalse = StateFalse->BindExpr(Ex, Val);
        Dst.Add(Builder->generateNode(Ex, StateFalse, Pred, tags.second));
      }
    }
    else
      Dst.Add(Pred);
  }
}

void ExprEngine::VisitAsmStmt(const AsmStmt *A, ExplodedNode *Pred,
                                ExplodedNodeSet &Dst) {
  VisitAsmStmtHelperOutputs(A, A->begin_outputs(), A->end_outputs(), Pred, Dst);
}

void ExprEngine::VisitAsmStmtHelperOutputs(const AsmStmt *A,
                                             AsmStmt::const_outputs_iterator I,
                                             AsmStmt::const_outputs_iterator E,
                                     ExplodedNode *Pred, ExplodedNodeSet &Dst) {
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

void ExprEngine::VisitAsmStmtHelperInputs(const AsmStmt *A,
                                            AsmStmt::const_inputs_iterator I,
                                            AsmStmt::const_inputs_iterator E,
                                            ExplodedNode *Pred,
                                            ExplodedNodeSet &Dst) {
  if (I == E) {

    // We have processed both the inputs and the outputs.  All of the outputs
    // should evaluate to Locs.  Nuke all of their values.

    // FIXME: Some day in the future it would be nice to allow a "plug-in"
    // which interprets the inline asm and stores proper results in the
    // outputs.

    const ProgramState *state = Pred->getState();

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
  static std::string getNodeAttributes(const ExplodedNode *N, void*) {

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

  static std::string getNodeLabel(const ExplodedNode *N, void*){

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
          const Stmt *S = L->getStmt();
          SourceLocation SLoc = S->getLocStart();

          Out << S->getStmtClassName() << ' ' << (void*) S << ' ';
          LangOptions LO; // FIXME.
          S->printPretty(Out, 0, PrintingPolicy(LO));

          if (SLoc.isFileID()) {
            Out << "\\lline="
              << GraphPrintSourceManager->getExpansionLineNumber(SLoc)
              << " col="
              << GraphPrintSourceManager->getExpansionColumnNumber(SLoc)
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

        const BlockEdge &E = cast<BlockEdge>(Loc);
        Out << "Edge: (B" << E.getSrc()->getBlockID() << ", B"
            << E.getDst()->getBlockID()  << ')';

        if (const Stmt *T = E.getSrc()->getTerminator()) {

          SourceLocation SLoc = T->getLocStart();

          Out << "\\|Terminator: ";
          LangOptions LO; // FIXME.
          E.getSrc()->printTerminator(Out, LO);

          if (SLoc.isFileID()) {
            Out << "\\lline="
              << GraphPrintSourceManager->getExpansionLineNumber(SLoc)
              << " col="
              << GraphPrintSourceManager->getExpansionColumnNumber(SLoc);
          }

          if (isa<SwitchStmt>(T)) {
            const Stmt *Label = E.getDst()->getLabel();

            if (Label) {
              if (const CaseStmt *C = dyn_cast<CaseStmt>(Label)) {
                Out << "\\lcase ";
                LangOptions LO; // FIXME.
                C->getLHS()->printPretty(Out, 0, PrintingPolicy(LO));

                if (const Stmt *RHS = C->getRHS()) {
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

    const ProgramState *state = N->getState();
    Out << "\\|StateID: " << (void*) state
        << " NodeID: " << (void*) N << "\\|";
    state->printDOT(Out, *N->getLocationContext()->getCFG());

    Out << "\\l";    

    if (const ProgramPointTag *tag = Loc.getTag()) {
      Out << "\\|Tag: " << tag->getTagDescription(); 
      Out << "\\l";
    }
    return Out.str();
  }
};
} // end llvm namespace
#endif

#ifndef NDEBUG
template <typename ITERATOR>
ExplodedNode *GetGraphNode(ITERATOR I) { return *I; }

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
    for (BugReporter::EQClasses_iterator
           EI = BR.EQClasses_begin(), EE = BR.EQClasses_end(); EI != EE; ++EI) {
      BugReportEquivClass& EQ = *EI;
      const BugReport &R = **EQ.begin();
      ExplodedNode *N = const_cast<ExplodedNode*>(R.getErrorNode());
      if (N) Src.push_back(N);
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
