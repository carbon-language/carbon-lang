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
#include "clang/StaticAnalyzer/Core/CheckerProvider.h"
#include "clang/AST/DeclBase.h"

using namespace clang;
using namespace ento;

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
        checkers->push_back(std::make_pair(info.Checker, info.CheckFn));
    }
  }

  assert(checkers);
  for (CachedDeclCheckers::iterator
         I = checkers->begin(), E = checkers->end(); I != E; ++I) {
    CheckerRef checker = I->first;
    CheckDeclFunc fn = I->second;
    fn(checker, D, mgr, BR);
  }
}

void CheckerManager::runCheckersOnASTBody(const Decl *D, AnalysisManager& mgr,
                                          BugReporter &BR) {
  assert(D && D->hasBody());

  for (unsigned i = 0, e = BodyCheckers.size(); i != e; ++i) {
    CheckerRef checker = BodyCheckers[i].first;
    CheckDeclFunc fn = BodyCheckers[i].second;
    fn(checker, D, mgr, BR);
  }
}

void CheckerManager::_registerForDecl(CheckerRef checker, CheckDeclFunc checkfn,
                                      HandlesDeclFunc isForDeclFn) {
  DeclCheckerInfo info = { checker, checkfn, isForDeclFn };
  DeclCheckers.push_back(info);
}

void CheckerManager::_registerForBody(CheckerRef checker,
                                      CheckDeclFunc checkfn) {
  BodyCheckers.push_back(std::make_pair(checker, checkfn));
}

void CheckerManager::registerCheckersToEngine(ExprEngine &eng) {
  for (unsigned i = 0, e = Funcs.size(); i != e; ++i)
    Funcs[i](eng);
}

CheckerManager::~CheckerManager() {
  for (unsigned i = 0, e = Checkers.size(); i != e; ++i) {
    CheckerRef checker = Checkers[i].first;
    Dtor dtor = Checkers[i].second;
    dtor(checker);
  }
}

// Anchor for the vtable.
CheckerProvider::~CheckerProvider() { }
