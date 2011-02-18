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
#include <vector>

namespace clang {
  class Decl;

namespace ento {
  class ExprEngine;
  class AnalysisManager;
  class BugReporter;

class CheckerManager {
public:
  ~CheckerManager();

  typedef void *CheckerRef;

//===----------------------------------------------------------------------===//
// registerChecker
//===----------------------------------------------------------------------===//

  /// \brief Used to register checkers.
  template <typename CHECKER>
  void registerChecker() {
    CHECKER *checker = new CHECKER();
    Checkers.push_back(std::pair<CheckerRef, Dtor>(checker, destruct<CHECKER>));
    CHECKER::_register(checker, *this);
  }

  typedef void (*RegisterToEngFunc)(ExprEngine &Eng);
  void addCheckerRegisterFunction(RegisterToEngFunc fn) {
    Funcs.push_back(fn);
  }

//===----------------------------------------------------------------------===//
// Functions for running checkers.
//===----------------------------------------------------------------------===//

  /// \brief Run checkers handling Decls.
  void runCheckersOnASTDecl(const Decl *D, AnalysisManager& mgr,
                            BugReporter &BR);

  /// \brief Run checkers handling Decls containing a Stmt body.
  void runCheckersOnASTBody(const Decl *D, AnalysisManager& mgr,
                            BugReporter &BR);

//===----------------------------------------------------------------------===//
// Internal registration functions.
//===----------------------------------------------------------------------===//

  // Functions used by the registration mechanism, checkers should not touch
  // these directly.

  typedef void (*CheckDeclFunc)(CheckerRef checker, const Decl *D,
                                AnalysisManager& mgr, BugReporter &BR);
  typedef bool (*HandlesDeclFunc)(const Decl *D);
  void _registerForDecl(CheckerRef checker, CheckDeclFunc checkfn,
                        HandlesDeclFunc isForDeclFn);

  void _registerForBody(CheckerRef checker, CheckDeclFunc checkfn);
  
  void registerCheckersToEngine(ExprEngine &eng);

private:
  template <typename CHECKER>
  static void destruct(void *obj) { delete static_cast<CHECKER *>(obj); }

  std::vector<RegisterToEngFunc> Funcs;

  struct DeclCheckerInfo {
    CheckerRef Checker;
    CheckDeclFunc CheckFn;
    HandlesDeclFunc IsForDeclFn;
  };
  std::vector<DeclCheckerInfo> DeclCheckers;

  std::vector<std::pair<CheckerRef, CheckDeclFunc> > BodyCheckers;

  typedef void (*Dtor)(void *);
  std::vector<std::pair<CheckerRef, Dtor> > Checkers;

  typedef llvm::SmallVector<std::pair<CheckerRef, CheckDeclFunc>, 4>
      CachedDeclCheckers;
  typedef llvm::DenseMap<unsigned, CachedDeclCheckers> CachedDeclCheckersMapTy;
  CachedDeclCheckersMapTy CachedDeclCheckersMap;
};

} // end ento namespace

} // end clang namespace

#endif
