//===--- CheckerManager.cpp - Static Analyzer Checker Manager -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Defines the Static Analyzer Checker Manager.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ObjCMessage.h"
#include "clang/Analysis/ProgramPoint.h"
#include "clang/AST/DeclBase.h"

using namespace clang;
using namespace ento;

bool CheckerManager::hasPathSensitiveCheckers() const {
  return !StmtCheckers.empty()              ||
         !PreObjCMessageCheckers.empty()    ||
         !PostObjCMessageCheckers.empty()   ||
         !LocationCheckers.empty()          ||
         !BindCheckers.empty()              ||
         !EndAnalysisCheckers.empty()       ||
         !EndPathCheckers.empty()           ||
         !BranchConditionCheckers.empty()   ||
         !LiveSymbolsCheckers.empty()       ||
         !DeadSymbolsCheckers.empty()       ||
         !RegionChangesCheckers.empty()     ||
         !EvalAssumeCheckers.empty()        ||
         !EvalCallCheckers.empty()          ||
         !InlineCallCheckers.empty();
}

void CheckerManager::finishedCheckerRegistration() {
#ifndef NDEBUG
  // Make sure that for every event that has listeners, there is at least
  // one dispatcher registered for it.
  for (llvm::DenseMap<EventTag, EventInfo>::iterator
         I = Events.begin(), E = Events.end(); I != E; ++I)
    assert(I->second.HasDispatcher && "No dispatcher registered for an event");
#endif
}

//===----------------------------------------------------------------------===//
// Functions for running checkers for AST traversing..
//===----------------------------------------------------------------------===//

void CheckerManager::runCheckersOnASTDecl(const Decl *D, AnalysisManager& mgr,
                                          BugReporter &BR) {
  assert(D);

  unsigned DeclKind = D->getKind();
  CachedDeclCheckers *checkers = 0;
  CachedDeclCheckersMapTy::iterator CCI = CachedDeclCheckersMap.find(DeclKind);
  if (CCI != CachedDeclCheckersMap.end()) {
    checkers = &(CCI->second);
  } else {
    // Find the checkers that should run for this Decl and cache them.
    checkers = &CachedDeclCheckersMap[DeclKind];
    for (unsigned i = 0, e = DeclCheckers.size(); i != e; ++i) {
      DeclCheckerInfo &info = DeclCheckers[i];
      if (info.IsForDeclFn(D))
        checkers->push_back(info.CheckFn);
    }
  }

  assert(checkers);
  for (CachedDeclCheckers::iterator
         I = checkers->begin(), E = checkers->end(); I != E; ++I)
    (*I)(D, mgr, BR);
}

void CheckerManager::runCheckersOnASTBody(const Decl *D, AnalysisManager& mgr,
                                          BugReporter &BR) {
  assert(D && D->hasBody());

  for (unsigned i = 0, e = BodyCheckers.size(); i != e; ++i)
    BodyCheckers[i](D, mgr, BR);
}

//===----------------------------------------------------------------------===//
// Functions for running checkers for path-sensitive checking.
//===----------------------------------------------------------------------===//

template <typename CHECK_CTX>
static void expandGraphWithCheckers(CHECK_CTX checkCtx,
                                    ExplodedNodeSet &Dst,
                                    const ExplodedNodeSet &Src) {
  const NodeBuilderContext &BldrCtx = checkCtx.Eng.getBuilderContext();
  if (Src.empty())
    return;

  typename CHECK_CTX::CheckersTy::const_iterator
      I = checkCtx.checkers_begin(), E = checkCtx.checkers_end();
  if (I == E) {
    Dst.insert(Src);
    return;
  }

  ExplodedNodeSet Tmp1, Tmp2;
  const ExplodedNodeSet *PrevSet = &Src;

  for (; I != E; ++I) {
    ExplodedNodeSet *CurrSet = 0;
    if (I+1 == E)
      CurrSet = &Dst;
    else {
      CurrSet = (PrevSet == &Tmp1) ? &Tmp2 : &Tmp1;
      CurrSet->clear();
    }

    NodeBuilder B(*PrevSet, *CurrSet, BldrCtx);
    for (ExplodedNodeSet::iterator NI = PrevSet->begin(), NE = PrevSet->end();
         NI != NE; ++NI) {
      checkCtx.runChecker(*I, B, *NI);
    }

    // If all the produced transitions are sinks, stop.
    if (CurrSet->empty())
      return;

    // Update which NodeSet is the current one.
    PrevSet = CurrSet;
  }
}

namespace {
  struct CheckStmtContext {
    typedef SmallVectorImpl<CheckerManager::CheckStmtFunc> CheckersTy;
    bool IsPreVisit;
    const CheckersTy &Checkers;
    const Stmt *S;
    ExprEngine &Eng;
    bool wasInlined;

    CheckersTy::const_iterator checkers_begin() { return Checkers.begin(); }
    CheckersTy::const_iterator checkers_end() { return Checkers.end(); }

    CheckStmtContext(bool isPreVisit, const CheckersTy &checkers,
                     const Stmt *s, ExprEngine &eng, bool wasInlined = false)
      : IsPreVisit(isPreVisit), Checkers(checkers), S(s), Eng(eng),
        wasInlined(wasInlined) {}

    void runChecker(CheckerManager::CheckStmtFunc checkFn,
                    NodeBuilder &Bldr, ExplodedNode *Pred) {
      // FIXME: Remove respondsToCallback from CheckerContext;
      ProgramPoint::Kind K =  IsPreVisit ? ProgramPoint::PreStmtKind :
                                           ProgramPoint::PostStmtKind;
      const ProgramPoint &L = ProgramPoint::getProgramPoint(S, K,
                                Pred->getLocationContext(), checkFn.Checker);
      CheckerContext C(Bldr, Eng, Pred, L, wasInlined);
      checkFn(S, C);
    }
  };
}

/// \brief Run checkers for visiting Stmts.
void CheckerManager::runCheckersForStmt(bool isPreVisit,
                                        ExplodedNodeSet &Dst,
                                        const ExplodedNodeSet &Src,
                                        const Stmt *S,
                                        ExprEngine &Eng,
                                        bool wasInlined) {
  CheckStmtContext C(isPreVisit, *getCachedStmtCheckersFor(S, isPreVisit),
                     S, Eng, wasInlined);
  expandGraphWithCheckers(C, Dst, Src);
}

namespace {
  struct CheckObjCMessageContext {
    typedef std::vector<CheckerManager::CheckObjCMessageFunc> CheckersTy;
    bool IsPreVisit;
    const CheckersTy &Checkers;
    const ObjCMessage &Msg;
    ExprEngine &Eng;

    CheckersTy::const_iterator checkers_begin() { return Checkers.begin(); }
    CheckersTy::const_iterator checkers_end() { return Checkers.end(); }

    CheckObjCMessageContext(bool isPreVisit, const CheckersTy &checkers,
                            const ObjCMessage &msg, ExprEngine &eng)
      : IsPreVisit(isPreVisit), Checkers(checkers), Msg(msg), Eng(eng) { }

    void runChecker(CheckerManager::CheckObjCMessageFunc checkFn,
                    NodeBuilder &Bldr, ExplodedNode *Pred) {
      ProgramPoint::Kind K =  IsPreVisit ? ProgramPoint::PreStmtKind :
                                           ProgramPoint::PostStmtKind;
      const ProgramPoint &L =
        ProgramPoint::getProgramPoint(Msg.getMessageExpr(),
                                      K, Pred->getLocationContext(),
                                      checkFn.Checker);
      CheckerContext C(Bldr, Eng, Pred, L);

      checkFn(Msg, C);
    }
  };
}

/// \brief Run checkers for visiting obj-c messages.
void CheckerManager::runCheckersForObjCMessage(bool isPreVisit,
                                               ExplodedNodeSet &Dst,
                                               const ExplodedNodeSet &Src,
                                               const ObjCMessage &msg,
                                               ExprEngine &Eng) {
  CheckObjCMessageContext C(isPreVisit,
                            isPreVisit ? PreObjCMessageCheckers
                                       : PostObjCMessageCheckers,
                            msg, Eng);
  expandGraphWithCheckers(C, Dst, Src);
}

namespace {
  struct CheckLocationContext {
    typedef std::vector<CheckerManager::CheckLocationFunc> CheckersTy;
    const CheckersTy &Checkers;
    SVal Loc;
    bool IsLoad;
    const Stmt *NodeEx; /* Will become a CFGStmt */
    const Stmt *BoundEx;
    ExprEngine &Eng;

    CheckersTy::const_iterator checkers_begin() { return Checkers.begin(); }
    CheckersTy::const_iterator checkers_end() { return Checkers.end(); }

    CheckLocationContext(const CheckersTy &checkers,
                         SVal loc, bool isLoad, const Stmt *NodeEx,
                         const Stmt *BoundEx,
                         ExprEngine &eng)
      : Checkers(checkers), Loc(loc), IsLoad(isLoad), NodeEx(NodeEx),
        BoundEx(BoundEx), Eng(eng) {}

    void runChecker(CheckerManager::CheckLocationFunc checkFn,
                    NodeBuilder &Bldr, ExplodedNode *Pred) {
      ProgramPoint::Kind K =  IsLoad ? ProgramPoint::PreLoadKind :
                                       ProgramPoint::PreStoreKind;
      const ProgramPoint &L =
        ProgramPoint::getProgramPoint(NodeEx, K,
                                      Pred->getLocationContext(),
                                      checkFn.Checker);
      CheckerContext C(Bldr, Eng, Pred, L);
      checkFn(Loc, IsLoad, BoundEx, C);
    }
  };
}

/// \brief Run checkers for load/store of a location.

void CheckerManager::runCheckersForLocation(ExplodedNodeSet &Dst,
                                            const ExplodedNodeSet &Src,
                                            SVal location, bool isLoad,
                                            const Stmt *NodeEx,
                                            const Stmt *BoundEx,
                                            ExprEngine &Eng) {
  CheckLocationContext C(LocationCheckers, location, isLoad, NodeEx,
                         BoundEx, Eng);
  expandGraphWithCheckers(C, Dst, Src);
}

namespace {
  struct CheckBindContext {
    typedef std::vector<CheckerManager::CheckBindFunc> CheckersTy;
    const CheckersTy &Checkers;
    SVal Loc;
    SVal Val;
    const Stmt *S;
    ExprEngine &Eng;
    ProgramPoint::Kind PointKind;

    CheckersTy::const_iterator checkers_begin() { return Checkers.begin(); }
    CheckersTy::const_iterator checkers_end() { return Checkers.end(); }

    CheckBindContext(const CheckersTy &checkers,
                     SVal loc, SVal val, const Stmt *s, ExprEngine &eng,
                     ProgramPoint::Kind PK)
      : Checkers(checkers), Loc(loc), Val(val), S(s), Eng(eng), PointKind(PK) {}

    void runChecker(CheckerManager::CheckBindFunc checkFn,
                    NodeBuilder &Bldr, ExplodedNode *Pred) {
      const ProgramPoint &L = ProgramPoint::getProgramPoint(S, PointKind,
                                Pred->getLocationContext(), checkFn.Checker);
      CheckerContext C(Bldr, Eng, Pred, L);

      checkFn(Loc, Val, S, C);
    }
  };
}

/// \brief Run checkers for binding of a value to a location.
void CheckerManager::runCheckersForBind(ExplodedNodeSet &Dst,
                                        const ExplodedNodeSet &Src,
                                        SVal location, SVal val,
                                        const Stmt *S, ExprEngine &Eng,
                                        ProgramPoint::Kind PointKind) {
  CheckBindContext C(BindCheckers, location, val, S, Eng, PointKind);
  expandGraphWithCheckers(C, Dst, Src);
}

void CheckerManager::runCheckersForEndAnalysis(ExplodedGraph &G,
                                               BugReporter &BR,
                                               ExprEngine &Eng) {
  for (unsigned i = 0, e = EndAnalysisCheckers.size(); i != e; ++i)
    EndAnalysisCheckers[i](G, BR, Eng);
}

/// \brief Run checkers for end of path.
// Note, We do not chain the checker output (like in expandGraphWithCheckers)
// for this callback since end of path nodes are expected to be final.
void CheckerManager::runCheckersForEndPath(NodeBuilderContext &BC,
                                           ExplodedNodeSet &Dst,
                                           ExprEngine &Eng) {
  ExplodedNode *Pred = BC.Pred;
  
  // We define the builder outside of the loop bacause if at least one checkers
  // creates a sucsessor for Pred, we do not need to generate an 
  // autotransition for it.
  NodeBuilder Bldr(Pred, Dst, BC);
  for (unsigned i = 0, e = EndPathCheckers.size(); i != e; ++i) {
    CheckEndPathFunc checkFn = EndPathCheckers[i];

    const ProgramPoint &L = BlockEntrance(BC.Block,
                                          Pred->getLocationContext(),
                                          checkFn.Checker);
    CheckerContext C(Bldr, Eng, Pred, L);
    checkFn(C);
  }
}

namespace {
  struct CheckBranchConditionContext {
    typedef std::vector<CheckerManager::CheckBranchConditionFunc> CheckersTy;
    const CheckersTy &Checkers;
    const Stmt *Condition;
    ExprEngine &Eng;

    CheckersTy::const_iterator checkers_begin() { return Checkers.begin(); }
    CheckersTy::const_iterator checkers_end() { return Checkers.end(); }

    CheckBranchConditionContext(const CheckersTy &checkers,
                                const Stmt *Cond, ExprEngine &eng)
      : Checkers(checkers), Condition(Cond), Eng(eng) {}

    void runChecker(CheckerManager::CheckBranchConditionFunc checkFn,
                    NodeBuilder &Bldr, ExplodedNode *Pred) {
      ProgramPoint L = PostCondition(Condition, Pred->getLocationContext(),
                                     checkFn.Checker);
      CheckerContext C(Bldr, Eng, Pred, L);
      checkFn(Condition, C);
    }
  };
}

/// \brief Run checkers for branch condition.
void CheckerManager::runCheckersForBranchCondition(const Stmt *Condition,
                                                   ExplodedNodeSet &Dst,
                                                   ExplodedNode *Pred,
                                                   ExprEngine &Eng) {
  ExplodedNodeSet Src;
  Src.insert(Pred);
  CheckBranchConditionContext C(BranchConditionCheckers, Condition, Eng);
  expandGraphWithCheckers(C, Dst, Src);
}

/// \brief Run checkers for live symbols.
void CheckerManager::runCheckersForLiveSymbols(ProgramStateRef state,
                                               SymbolReaper &SymReaper) {
  for (unsigned i = 0, e = LiveSymbolsCheckers.size(); i != e; ++i)
    LiveSymbolsCheckers[i](state, SymReaper);
}

namespace {
  struct CheckDeadSymbolsContext {
    typedef std::vector<CheckerManager::CheckDeadSymbolsFunc> CheckersTy;
    const CheckersTy &Checkers;
    SymbolReaper &SR;
    const Stmt *S;
    ExprEngine &Eng;

    CheckersTy::const_iterator checkers_begin() { return Checkers.begin(); }
    CheckersTy::const_iterator checkers_end() { return Checkers.end(); }

    CheckDeadSymbolsContext(const CheckersTy &checkers, SymbolReaper &sr,
                            const Stmt *s, ExprEngine &eng)
      : Checkers(checkers), SR(sr), S(s), Eng(eng) { }

    void runChecker(CheckerManager::CheckDeadSymbolsFunc checkFn,
                    NodeBuilder &Bldr, ExplodedNode *Pred) {
      ProgramPoint::Kind K = ProgramPoint::PostPurgeDeadSymbolsKind;
      const ProgramPoint &L = ProgramPoint::getProgramPoint(S, K,
                                Pred->getLocationContext(), checkFn.Checker);
      CheckerContext C(Bldr, Eng, Pred, L);

      checkFn(SR, C);
    }
  };
}

/// \brief Run checkers for dead symbols.
void CheckerManager::runCheckersForDeadSymbols(ExplodedNodeSet &Dst,
                                               const ExplodedNodeSet &Src,
                                               SymbolReaper &SymReaper,
                                               const Stmt *S,
                                               ExprEngine &Eng) {
  CheckDeadSymbolsContext C(DeadSymbolsCheckers, SymReaper, S, Eng);
  expandGraphWithCheckers(C, Dst, Src);
}

/// \brief True if at least one checker wants to check region changes.
bool CheckerManager::wantsRegionChangeUpdate(ProgramStateRef state) {
  for (unsigned i = 0, e = RegionChangesCheckers.size(); i != e; ++i)
    if (RegionChangesCheckers[i].WantUpdateFn(state))
      return true;

  return false;
}

/// \brief Run checkers for region changes.
ProgramStateRef 
CheckerManager::runCheckersForRegionChanges(ProgramStateRef state,
                            const StoreManager::InvalidatedSymbols *invalidated,
                                    ArrayRef<const MemRegion *> ExplicitRegions,
                                          ArrayRef<const MemRegion *> Regions,
                                          const CallOrObjCMessage *Call) {
  for (unsigned i = 0, e = RegionChangesCheckers.size(); i != e; ++i) {
    // If any checker declares the state infeasible (or if it starts that way),
    // bail out.
    if (!state)
      return NULL;
    state = RegionChangesCheckers[i].CheckFn(state, invalidated, 
                                             ExplicitRegions, Regions, Call);
  }
  return state;
}

/// \brief Run checkers for handling assumptions on symbolic values.
ProgramStateRef 
CheckerManager::runCheckersForEvalAssume(ProgramStateRef state,
                                         SVal Cond, bool Assumption) {
  for (unsigned i = 0, e = EvalAssumeCheckers.size(); i != e; ++i) {
    // If any checker declares the state infeasible (or if it starts that way),
    // bail out.
    if (!state)
      return NULL;
    state = EvalAssumeCheckers[i](state, Cond, Assumption);
  }
  return state;
}

/// \brief Run checkers for evaluating a call.
/// Only one checker will evaluate the call.
void CheckerManager::runCheckersForEvalCall(ExplodedNodeSet &Dst,
                                            const ExplodedNodeSet &Src,
                                            const CallExpr *CE,
                                            ExprEngine &Eng,
                                            GraphExpander *defaultEval) {
  if (EvalCallCheckers.empty()   &&
      InlineCallCheckers.empty() &&
      defaultEval == 0) {
    Dst.insert(Src);
    return;
  }

  for (ExplodedNodeSet::iterator
         NI = Src.begin(), NE = Src.end(); NI != NE; ++NI) {

    ExplodedNode *Pred = *NI;
    bool anyEvaluated = false;

    // First, check if any of the InlineCall callbacks can evaluate the call.
    assert(InlineCallCheckers.size() <= 1 &&
           "InlineCall is a special hacky callback to allow intrusive"
           "evaluation of the call (which simulates inlining). It is "
           "currently only used by OSAtomicChecker and should go away "
           "at some point.");
    for (std::vector<InlineCallFunc>::iterator
           EI = InlineCallCheckers.begin(), EE = InlineCallCheckers.end();
         EI != EE; ++EI) {
      ExplodedNodeSet checkDst;
      bool evaluated = (*EI)(CE, Eng, Pred, checkDst);
      assert(!(evaluated && anyEvaluated)
             && "There are more than one checkers evaluating the call");
      if (evaluated) {
        anyEvaluated = true;
        Dst.insert(checkDst);
#ifdef NDEBUG
        break; // on release don't check that no other checker also evals.
#endif
      }
    }

#ifdef NDEBUG // on release don't check that no other checker also evals.
    if (anyEvaluated) {
      break;
    }
#endif

    ExplodedNodeSet checkDst;
    NodeBuilder B(Pred, checkDst, Eng.getBuilderContext());
    // Next, check if any of the EvalCall callbacks can evaluate the call.
    for (std::vector<EvalCallFunc>::iterator
           EI = EvalCallCheckers.begin(), EE = EvalCallCheckers.end();
         EI != EE; ++EI) {
      ProgramPoint::Kind K = ProgramPoint::PostStmtKind;
      const ProgramPoint &L = ProgramPoint::getProgramPoint(CE, K,
                                Pred->getLocationContext(), EI->Checker);
      bool evaluated = false;
      { // CheckerContext generates transitions(populates checkDest) on
        // destruction, so introduce the scope to make sure it gets properly
        // populated.
        CheckerContext C(B, Eng, Pred, L);
        evaluated = (*EI)(CE, C);
      }
      assert(!(evaluated && anyEvaluated)
             && "There are more than one checkers evaluating the call");
      if (evaluated) {
        anyEvaluated = true;
        Dst.insert(checkDst);
#ifdef NDEBUG
        break; // on release don't check that no other checker also evals.
#endif
      }
    }
    
    // If none of the checkers evaluated the call, ask ExprEngine to handle it.
    if (!anyEvaluated) {
      if (defaultEval)
        defaultEval->expandGraph(Dst, Pred);
      else
        Dst.insert(Pred);
    }
  }
}

/// \brief Run checkers for the entire Translation Unit.
void CheckerManager::runCheckersOnEndOfTranslationUnit(
                                                  const TranslationUnitDecl *TU,
                                                  AnalysisManager &mgr,
                                                  BugReporter &BR) {
  for (unsigned i = 0, e = EndOfTranslationUnitCheckers.size(); i != e; ++i)
    EndOfTranslationUnitCheckers[i](TU, mgr, BR);
}

void CheckerManager::runCheckersForPrintState(raw_ostream &Out,
                                              ProgramStateRef State,
                                              const char *NL, const char *Sep) {
  for (llvm::DenseMap<CheckerTag, CheckerRef>::iterator
        I = CheckerTags.begin(), E = CheckerTags.end(); I != E; ++I)
    I->second->printState(Out, State, NL, Sep);
}

//===----------------------------------------------------------------------===//
// Internal registration functions for AST traversing.
//===----------------------------------------------------------------------===//

void CheckerManager::_registerForDecl(CheckDeclFunc checkfn,
                                      HandlesDeclFunc isForDeclFn) {
  DeclCheckerInfo info = { checkfn, isForDeclFn };
  DeclCheckers.push_back(info);
}

void CheckerManager::_registerForBody(CheckDeclFunc checkfn) {
  BodyCheckers.push_back(checkfn);
}

//===----------------------------------------------------------------------===//
// Internal registration functions for path-sensitive checking.
//===----------------------------------------------------------------------===//

void CheckerManager::_registerForPreStmt(CheckStmtFunc checkfn,
                                         HandlesStmtFunc isForStmtFn) {
  StmtCheckerInfo info = { checkfn, isForStmtFn, /*IsPreVisit*/true };
  StmtCheckers.push_back(info);
}
void CheckerManager::_registerForPostStmt(CheckStmtFunc checkfn,
                                          HandlesStmtFunc isForStmtFn) {
  StmtCheckerInfo info = { checkfn, isForStmtFn, /*IsPreVisit*/false };
  StmtCheckers.push_back(info);
}

void CheckerManager::_registerForPreObjCMessage(CheckObjCMessageFunc checkfn) {
  PreObjCMessageCheckers.push_back(checkfn);
}
void CheckerManager::_registerForPostObjCMessage(CheckObjCMessageFunc checkfn) {
  PostObjCMessageCheckers.push_back(checkfn);
}

void CheckerManager::_registerForLocation(CheckLocationFunc checkfn) {
  LocationCheckers.push_back(checkfn);
}

void CheckerManager::_registerForBind(CheckBindFunc checkfn) {
  BindCheckers.push_back(checkfn);
}

void CheckerManager::_registerForEndAnalysis(CheckEndAnalysisFunc checkfn) {
  EndAnalysisCheckers.push_back(checkfn);
}

void CheckerManager::_registerForEndPath(CheckEndPathFunc checkfn) {
  EndPathCheckers.push_back(checkfn);
}

void CheckerManager::_registerForBranchCondition(
                                             CheckBranchConditionFunc checkfn) {
  BranchConditionCheckers.push_back(checkfn);
}

void CheckerManager::_registerForLiveSymbols(CheckLiveSymbolsFunc checkfn) {
  LiveSymbolsCheckers.push_back(checkfn);
}

void CheckerManager::_registerForDeadSymbols(CheckDeadSymbolsFunc checkfn) {
  DeadSymbolsCheckers.push_back(checkfn);
}

void CheckerManager::_registerForRegionChanges(CheckRegionChangesFunc checkfn,
                                     WantsRegionChangeUpdateFunc wantUpdateFn) {
  RegionChangesCheckerInfo info = {checkfn, wantUpdateFn};
  RegionChangesCheckers.push_back(info);
}

void CheckerManager::_registerForEvalAssume(EvalAssumeFunc checkfn) {
  EvalAssumeCheckers.push_back(checkfn);
}

void CheckerManager::_registerForEvalCall(EvalCallFunc checkfn) {
  EvalCallCheckers.push_back(checkfn);
}

void CheckerManager::_registerForInlineCall(InlineCallFunc checkfn) {
  InlineCallCheckers.push_back(checkfn);
}

void CheckerManager::_registerForEndOfTranslationUnit(
                                            CheckEndOfTranslationUnit checkfn) {
  EndOfTranslationUnitCheckers.push_back(checkfn);
}

//===----------------------------------------------------------------------===//
// Implementation details.
//===----------------------------------------------------------------------===//

CheckerManager::CachedStmtCheckers *
CheckerManager::getCachedStmtCheckersFor(const Stmt *S, bool isPreVisit) {
  assert(S);

  CachedStmtCheckersKey key(S->getStmtClass(), isPreVisit);
  CachedStmtCheckers *checkers = 0;
  CachedStmtCheckersMapTy::iterator CCI = CachedStmtCheckersMap.find(key);
  if (CCI != CachedStmtCheckersMap.end()) {
    checkers = &(CCI->second);
  } else {
    // Find the checkers that should run for this Stmt and cache them.
    checkers = &CachedStmtCheckersMap[key];
    for (unsigned i = 0, e = StmtCheckers.size(); i != e; ++i) {
      StmtCheckerInfo &info = StmtCheckers[i];
      if (info.IsPreVisit == isPreVisit && info.IsForStmtFn(S))
        checkers->push_back(info.CheckFn);
    }
  }

  assert(checkers);
  return checkers;
}

CheckerManager::~CheckerManager() {
  for (unsigned i = 0, e = CheckerDtors.size(); i != e; ++i)
    CheckerDtors[i]();
}

// Anchor for the vtable.
GraphExpander::~GraphExpander() { }
