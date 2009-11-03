//== NullDerefChecker.h - Null dereference checker --------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines NullDerefChecker and UndefDerefChecker, two builtin checks
// in GRExprEngine that check for null and undefined pointers at loads
// and stores.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DEREFCHECKER
#define LLVM_CLANG_DEREFCHECKER

#include "clang/Analysis/PathSensitive/Checker.h"
#include "clang/Analysis/PathSensitive/BugType.h"

namespace clang {

class ExplodedNode;

class NullDerefChecker : public Checker {
  BuiltinBug *BT;
  llvm::SmallVector<ExplodedNode*, 2> ImplicitNullDerefNodes;

public:
  NullDerefChecker() : BT(0) {}
  ExplodedNode *CheckLocation(const Stmt *S, ExplodedNode *Pred,
                              const GRState *state, SVal V,GRExprEngine &Eng);

  static void *getTag();
  typedef llvm::SmallVectorImpl<ExplodedNode*>::iterator iterator;
  iterator implicit_nodes_begin() { return ImplicitNullDerefNodes.begin(); }
  iterator implicit_nodes_end() { return ImplicitNullDerefNodes.end(); }
};

class UndefDerefChecker : public Checker {
  BuiltinBug *BT;
public:
  UndefDerefChecker() : BT(0) {}

  ExplodedNode *CheckLocation(const Stmt *S, ExplodedNode *Pred,
                              const GRState *state, SVal V, GRExprEngine &Eng);

  static void *getTag();
};

} // end clang namespace
#endif
