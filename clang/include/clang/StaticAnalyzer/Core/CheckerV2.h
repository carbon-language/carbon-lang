//== CheckerV2.h - Registration mechanism for checkers -----------*- C++ -*--=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines CheckerV2, used to create and register checkers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SA_CORE_CHECKERV2
#define LLVM_CLANG_SA_CORE_CHECKERV2

#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "llvm/Support/Casting.h"

namespace clang {
namespace ento {
  class BugReporter;

namespace check {

struct _VoidCheck {
  static void _register(void *checker, CheckerManager &mgr) { }
};

template <typename DECL>
class ASTDecl {
  template <typename CHECKER>
  static void _checkDecl(void *checker, const Decl *D, AnalysisManager& mgr,
                         BugReporter &BR) {
    ((const CHECKER *)checker)->checkASTDecl(llvm::cast<DECL>(D), mgr, BR);
  }

  static bool _handlesDecl(const Decl *D) {
    return llvm::isa<DECL>(D);
  }
public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForDecl(CheckerManager::CheckDeclFunc(checker,
                                                       _checkDecl<CHECKER>),
                         _handlesDecl);
  }
};

class ASTCodeBody {
  template <typename CHECKER>
  static void _checkBody(void *checker, const Decl *D, AnalysisManager& mgr,
                         BugReporter &BR) {
    ((const CHECKER *)checker)->checkASTCodeBody(D, mgr, BR);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForBody(CheckerManager::CheckDeclFunc(checker,
                                                       _checkBody<CHECKER>));
  }
};

template <typename STMT>
class PreStmt {
  template <typename CHECKER>
  static void _checkStmt(void *checker, const Stmt *S, CheckerContext &C) {
    ((const CHECKER *)checker)->checkPreStmt(llvm::cast<STMT>(S), C);
  }

  static bool _handlesStmt(const Stmt *S) {
    return llvm::isa<STMT>(S);
  }
public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForPreStmt(CheckerManager::CheckStmtFunc(checker,
                                                          _checkStmt<CHECKER>),
                            _handlesStmt);
  }
};

template <typename STMT>
class PostStmt {
  template <typename CHECKER>
  static void _checkStmt(void *checker, const Stmt *S, CheckerContext &C) {
    ((const CHECKER *)checker)->checkPostStmt(llvm::cast<STMT>(S), C);
  }

  static bool _handlesStmt(const Stmt *S) {
    return llvm::isa<STMT>(S);
  }
public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForPostStmt(CheckerManager::CheckStmtFunc(checker,
                                                           _checkStmt<CHECKER>),
                             _handlesStmt);
  }
};

class PreObjCMessage {
  template <typename CHECKER>
  static void _checkObjCMessage(void *checker, const ObjCMessage &msg,
                                CheckerContext &C) {
    ((const CHECKER *)checker)->checkPreObjCMessage(msg, C);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForPreObjCMessage(
     CheckerManager::CheckObjCMessageFunc(checker, _checkObjCMessage<CHECKER>));
  }
};

class PostObjCMessage {
  template <typename CHECKER>
  static void _checkObjCMessage(void *checker, const ObjCMessage &msg,
                                CheckerContext &C) {
    ((const CHECKER *)checker)->checkPostObjCMessage(msg, C);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForPostObjCMessage(
     CheckerManager::CheckObjCMessageFunc(checker, _checkObjCMessage<CHECKER>));
  }
};

class Location {
  template <typename CHECKER>
  static void _checkLocation(void *checker, const SVal &location, bool isLoad,
                             CheckerContext &C) {
    ((const CHECKER *)checker)->checkLocation(location, isLoad, C);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForLocation(
           CheckerManager::CheckLocationFunc(checker, _checkLocation<CHECKER>));
  }
};

class EndAnalysis {
  template <typename CHECKER>
  static void _checkEndAnalysis(void *checker, ExplodedGraph &G,
                                BugReporter &BR, ExprEngine &Eng) {
    ((const CHECKER *)checker)->checkEndAnalysis(G, BR, Eng);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForEndAnalysis(
     CheckerManager::CheckEndAnalysisFunc(checker, _checkEndAnalysis<CHECKER>));
  }
};

class EndPath {
  template <typename CHECKER>
  static void _checkEndPath(void *checker, EndOfFunctionNodeBuilder &B,
                            ExprEngine &Eng) {
    ((const CHECKER *)checker)->checkEndPath(B, Eng);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForEndPath(
     CheckerManager::CheckEndPathFunc(checker, _checkEndPath<CHECKER>));
  }
};

class LiveSymbols {
  template <typename CHECKER>
  static void _checkLiveSymbols(void *checker, const GRState *state,
                                SymbolReaper &SR) {
    ((const CHECKER *)checker)->checkLiveSymbols(state, SR);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForLiveSymbols(
     CheckerManager::CheckLiveSymbolsFunc(checker, _checkLiveSymbols<CHECKER>));
  }
};

class DeadSymbols {
  template <typename CHECKER>
  static void _checkDeadSymbols(void *checker,
                                SymbolReaper &SR, CheckerContext &C) {
    ((const CHECKER *)checker)->checkDeadSymbols(SR, C);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForDeadSymbols(
     CheckerManager::CheckDeadSymbolsFunc(checker, _checkDeadSymbols<CHECKER>));
  }
};

class RegionChanges {
  template <typename CHECKER>
  static const GRState *_checkRegionChanges(void *checker, const GRState *state,
                                            const MemRegion * const *Begin,
                                            const MemRegion * const *End) {
    return ((const CHECKER *)checker)->checkRegionChanges(state, Begin, End);
  }
  template <typename CHECKER>
  static bool _wantsRegionChangeUpdate(void *checker, const GRState *state) {
    return ((const CHECKER *)checker)->wantsRegionChangeUpdate(state);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForRegionChanges(
          CheckerManager::CheckRegionChangesFunc(checker,
                                                 _checkRegionChanges<CHECKER>),
          CheckerManager::WantsRegionChangeUpdateFunc(checker,
                                            _wantsRegionChangeUpdate<CHECKER>));
  }
};

} // end check namespace

namespace eval {

class Call {
  template <typename CHECKER>
  static bool _evalCall(void *checker, const CallExpr *CE, CheckerContext &C) {
    return ((const CHECKER *)checker)->evalCall(CE, C);
  }

public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    mgr._registerForEvalCall(
                     CheckerManager::EvalCallFunc(checker, _evalCall<CHECKER>));
  }
};

} // end eval namespace

template <typename CHECK1, typename CHECK2=check::_VoidCheck,
          typename CHECK3=check::_VoidCheck, typename CHECK4=check::_VoidCheck,
          typename CHECK5=check::_VoidCheck, typename CHECK6=check::_VoidCheck,
          typename CHECK7=check::_VoidCheck, typename CHECK8=check::_VoidCheck,
          typename CHECK9=check::_VoidCheck, typename CHECK10=check::_VoidCheck,
          typename CHECK11=check::_VoidCheck,typename CHECK12=check::_VoidCheck>
class CheckerV2 {
public:
  template <typename CHECKER>
  static void _register(CHECKER *checker, CheckerManager &mgr) {
    CHECK1::_register(checker, mgr);
    CHECK2::_register(checker, mgr);
    CHECK3::_register(checker, mgr);
    CHECK4::_register(checker, mgr);
    CHECK5::_register(checker, mgr);
    CHECK6::_register(checker, mgr);
    CHECK7::_register(checker, mgr);
    CHECK8::_register(checker, mgr);
    CHECK9::_register(checker, mgr);
    CHECK10::_register(checker, mgr);
    CHECK11::_register(checker, mgr);
    CHECK12::_register(checker, mgr);
  }
};

} // end ento namespace

} // end clang namespace

#endif
