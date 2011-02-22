//===--- CheckerManager.h - Static Analyzer Checker Manager -----*- C++ -*-===//
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

#ifndef LLVM_CLANG_SA_CORE_CHECKERMANAGER_H
#define LLVM_CLANG_SA_CORE_CHECKERMANAGER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include <vector>

namespace clang {
  class Decl;
  class Stmt;

namespace ento {
  class ExprEngine;
  class AnalysisManager;
  class BugReporter;
  class CheckerContext;
  class ObjCMessage;
  class SVal;
  class ExplodedNodeSet;
  class GRState;

struct VoidCheckerFnParm {};
template <typename P1=VoidCheckerFnParm, typename P2=VoidCheckerFnParm,
          typename P3=VoidCheckerFnParm, typename P4=VoidCheckerFnParm>
class CheckerFn {
  typedef void (*Func)(void *, P1, P2, P3, P4);
  Func Fn;
public:
  void *Checker;
  CheckerFn(void *checker, Func fn) : Fn(fn), Checker(checker) { }
  void operator()(P1 p1, P2 p2, P3 p3, P4 p4) { Fn(Checker, p1, p2, p3, p4); } 
};

template <typename P1, typename P2, typename P3>
class CheckerFn<P1, P2, P3, VoidCheckerFnParm> {
  typedef void (*Func)(void *, P1, P2, P3);
  Func Fn;
public:
  void *Checker;
  CheckerFn(void *checker, Func fn) : Fn(fn), Checker(checker) { }
  void operator()(P1 p1, P2 p2, P3 p3) { Fn(Checker, p1, p2, p3); } 
};

template <typename P1, typename P2>
class CheckerFn<P1, P2, VoidCheckerFnParm, VoidCheckerFnParm> {
  typedef void (*Func)(void *, P1, P2);
  Func Fn;
public:
  void *Checker;
  CheckerFn(void *checker, Func fn) : Fn(fn), Checker(checker) { }
  void operator()(P1 p1, P2 p2) { Fn(Checker, p1, p2); } 
};

template <>
class CheckerFn<VoidCheckerFnParm, VoidCheckerFnParm, VoidCheckerFnParm,
                VoidCheckerFnParm> {
  typedef void (*Func)(void *);
  Func Fn;
public:
  void *Checker;
  CheckerFn(void *checker, Func fn) : Fn(fn), Checker(checker) { }
  void operator()() { Fn(Checker); } 
};

class CheckerManager {
public:
  ~CheckerManager();

  typedef void *CheckerRef;
  typedef CheckerFn<> CheckerDtor;

//===----------------------------------------------------------------------===//
// registerChecker
//===----------------------------------------------------------------------===//

  /// \brief Used to register checkers.
  template <typename CHECKER>
  void registerChecker() {
    CHECKER *checker = new CHECKER();
    CheckerDtors.push_back(CheckerDtor(checker, destruct<CHECKER>));
    CHECKER::_register(checker, *this);
  }

  typedef void (*RegisterToEngFunc)(ExprEngine &Eng);
  void addCheckerRegisterFunction(RegisterToEngFunc fn) {
    Funcs.push_back(fn);
  }

//===----------------------------------------------------------------------===//
// Functions for running checkers for AST traversing..
//===----------------------------------------------------------------------===//

  /// \brief Run checkers handling Decls.
  void runCheckersOnASTDecl(const Decl *D, AnalysisManager& mgr,
                            BugReporter &BR);

  /// \brief Run checkers handling Decls containing a Stmt body.
  void runCheckersOnASTBody(const Decl *D, AnalysisManager& mgr,
                            BugReporter &BR);

//===----------------------------------------------------------------------===//
// Functions for running checkers for path-sensitive checking.
//===----------------------------------------------------------------------===//

  /// \brief Run checkers for pre-visiting Stmts.
  void runCheckersForPreStmt(ExplodedNodeSet &Dst,
                             ExplodedNodeSet &Src,
                             const Stmt *S,
                             ExprEngine &Eng) {
    runCheckersForStmt(/*isPreVisit=*/true, Dst, Src, S, Eng);
  }

  /// \brief Run checkers for post-visiting Stmts.
  void runCheckersForPostStmt(ExplodedNodeSet &Dst,
                              ExplodedNodeSet &Src,
                              const Stmt *S,
                              ExprEngine &Eng) {
    runCheckersForStmt(/*isPreVisit=*/false, Dst, Src, S, Eng);
  }

  /// \brief Run checkers for visiting Stmts.
  void runCheckersForStmt(bool isPreVisit,
                          ExplodedNodeSet &Dst, ExplodedNodeSet &Src,
                          const Stmt *S, ExprEngine &Eng);

  /// \brief Run checkers for pre-visiting obj-c messages.
  void runCheckersForPreObjCMessage(ExplodedNodeSet &Dst,
                                    ExplodedNodeSet &Src,
                                    const ObjCMessage &msg,
                                    ExprEngine &Eng) {
    runCheckersForObjCMessage(/*isPreVisit=*/true, Dst, Src, msg, Eng);
  }

  /// \brief Run checkers for post-visiting obj-c messages.
  void runCheckersForPostObjCMessage(ExplodedNodeSet &Dst,
                                     ExplodedNodeSet &Src,
                                     const ObjCMessage &msg,
                                     ExprEngine &Eng) {
    runCheckersForObjCMessage(/*isPreVisit=*/false, Dst, Src, msg, Eng);
  }

  /// \brief Run checkers for visiting obj-c messages.
  void runCheckersForObjCMessage(bool isPreVisit,
                                 ExplodedNodeSet &Dst, ExplodedNodeSet &Src,
                                 const ObjCMessage &msg, ExprEngine &Eng);

  /// \brief Run checkers for load/store of a location.
  void runCheckersForLocation(ExplodedNodeSet &Dst,
                              ExplodedNodeSet &Src,
                              SVal location, bool isLoad,
                              const Stmt *S,
                              const GRState *state,
                              ExprEngine &Eng);

  // FIXME: Temporary until checker running is moved completely into
  // CheckerManager.
  void registerCheckersToEngine(ExprEngine &eng);

//===----------------------------------------------------------------------===//
// Internal registration functions for AST traversing.
//===----------------------------------------------------------------------===//

  // Functions used by the registration mechanism, checkers should not touch
  // these directly.

  typedef CheckerFn<const Decl *, AnalysisManager&, BugReporter &>
      CheckDeclFunc;
  typedef CheckerFn<const Stmt *, CheckerContext &> CheckStmtFunc;

  typedef bool (*HandlesDeclFunc)(const Decl *D);
  void _registerForDecl(CheckDeclFunc checkfn, HandlesDeclFunc isForDeclFn);

  void _registerForBody(CheckDeclFunc checkfn);

//===----------------------------------------------------------------------===//
// Internal registration functions for path-sensitive checking.
//===----------------------------------------------------------------------===//

  typedef CheckerFn<const ObjCMessage &, CheckerContext &> CheckObjCMessageFunc;
  typedef CheckerFn<const SVal &/*location*/, bool/*isLoad*/, CheckerContext &>
      CheckLocationFunc;

  typedef bool (*HandlesStmtFunc)(const Stmt *D);
  void _registerForPreStmt(CheckStmtFunc checkfn,
                           HandlesStmtFunc isForStmtFn);
  void _registerForPostStmt(CheckStmtFunc checkfn,
                            HandlesStmtFunc isForStmtFn);

  void _registerForPreObjCMessage(CheckObjCMessageFunc checkfn);
  void _registerForPostObjCMessage(CheckObjCMessageFunc checkfn);

  void _registerForLocation(CheckLocationFunc checkfn);

//===----------------------------------------------------------------------===//
// Implementation details.
//===----------------------------------------------------------------------===//

private:
  template <typename CHECKER>
  static void destruct(void *obj) { delete static_cast<CHECKER *>(obj); }

  std::vector<CheckerDtor> CheckerDtors;

  std::vector<RegisterToEngFunc> Funcs;

  struct DeclCheckerInfo {
    CheckDeclFunc CheckFn;
    HandlesDeclFunc IsForDeclFn;
  };
  std::vector<DeclCheckerInfo> DeclCheckers;

  std::vector<CheckDeclFunc> BodyCheckers;

  typedef llvm::SmallVector<CheckDeclFunc, 4> CachedDeclCheckers;
  typedef llvm::DenseMap<unsigned, CachedDeclCheckers> CachedDeclCheckersMapTy;
  CachedDeclCheckersMapTy CachedDeclCheckersMap;

  struct StmtCheckerInfo {
    CheckStmtFunc CheckFn;
    HandlesStmtFunc IsForStmtFn;
    bool IsPreVisit;
  };
  std::vector<StmtCheckerInfo> StmtCheckers;

  struct CachedStmtCheckersKey {
    unsigned StmtKind;
    bool IsPreVisit;

    CachedStmtCheckersKey() : StmtKind(0), IsPreVisit(0) { }
    CachedStmtCheckersKey(unsigned stmtKind, bool isPreVisit)
      : StmtKind(stmtKind), IsPreVisit(isPreVisit) { }

    static CachedStmtCheckersKey getSentinel() {
      return CachedStmtCheckersKey(~0U, ~0U);
    }
    unsigned getHashValue() const {
      llvm::FoldingSetNodeID ID;
      ID.AddInteger(StmtKind);
      ID.AddBoolean(IsPreVisit);
      return ID.ComputeHash();
    }
    bool operator==(const CachedStmtCheckersKey &RHS) const {
      return StmtKind == RHS.StmtKind && IsPreVisit == RHS.IsPreVisit;
    }
  };
  friend struct llvm::DenseMapInfo<CachedStmtCheckersKey>;

  typedef llvm::SmallVector<CheckStmtFunc, 4> CachedStmtCheckers;
  typedef llvm::DenseMap<CachedStmtCheckersKey, CachedStmtCheckers>
      CachedStmtCheckersMapTy;
  CachedStmtCheckersMapTy CachedStmtCheckersMap;

  CachedStmtCheckers *getCachedStmtCheckersFor(const Stmt *S, bool isPreVisit);

  std::vector<CheckObjCMessageFunc> PreObjCMessageCheckers;
  std::vector<CheckObjCMessageFunc> PostObjCMessageCheckers;

  std::vector<CheckLocationFunc> LocationCheckers;
};

} // end ento namespace

} // end clang namespace

namespace llvm {
  /// Define DenseMapInfo so that CachedStmtCheckersKey can be used as key
  /// in DenseMap and DenseSets.
  template <>
  struct DenseMapInfo<clang::ento::CheckerManager::CachedStmtCheckersKey> {
    static inline clang::ento::CheckerManager::CachedStmtCheckersKey
        getEmptyKey() {
      return clang::ento::CheckerManager::CachedStmtCheckersKey();
    }
    static inline clang::ento::CheckerManager::CachedStmtCheckersKey
        getTombstoneKey() {
      return clang::ento::CheckerManager::CachedStmtCheckersKey::getSentinel();
    }

    static unsigned
        getHashValue(clang::ento::CheckerManager::CachedStmtCheckersKey S) {
      return S.getHashValue();
    }

    static bool isEqual(clang::ento::CheckerManager::CachedStmtCheckersKey LHS,
                       clang::ento::CheckerManager::CachedStmtCheckersKey RHS) {
      return LHS == RHS;
    }
  };
} // end namespace llvm

#endif
