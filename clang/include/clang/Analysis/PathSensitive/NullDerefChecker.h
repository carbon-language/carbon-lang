//== NullDerefChecker - Null dereference checker ----------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_NULLDEREFCHECKER
#define LLVM_CLANG_NULLDEREFCHECKER

#include "clang/Analysis/PathSensitive/Checker.h"
#include "clang/Analysis/PathSensitive/BugType.h"

namespace clang {

class ExplodedNode;

class NullDeref : public BuiltinBug {
public:
  NullDeref() 
    : BuiltinBug(0, "Null dereference", "Dereference of null pointer") {}

  void registerInitialVisitors(BugReporterContext& BRC,
                               const ExplodedNode* N,
                               BuiltinBugReport *R);
};

class NullDerefChecker : public Checker {
  NullDeref *BT;
  llvm::SmallVector<ExplodedNode*, 2> ImplicitNullDerefNodes;

public:

  NullDerefChecker() : BT(0) {}
  ExplodedNode *CheckLocation(const Stmt *S, ExplodedNode *Pred,
                              const GRState *state, SVal V,GRExprEngine &Eng);

  static void *getTag() {
    static int x = 0;
    return &x;
  }

  typedef llvm::SmallVectorImpl<ExplodedNode*>::iterator iterator;
  iterator implicit_nodes_begin() { return ImplicitNullDerefNodes.begin(); }
  iterator implicit_nodes_end() { return ImplicitNullDerefNodes.end(); }
};

} // end clang namespace
#endif
