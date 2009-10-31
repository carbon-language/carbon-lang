//== UndefDerefChecker.h - Undefined dereference checker --------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines UndefDerefChecker, a builtin check in GRExprEngine that performs
// checks for defined pointers at loads and stores.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/Checker.h"
#include "clang/Analysis/PathSensitive/BugType.h"

namespace clang {

class UndefDerefChecker : public Checker {
  BuiltinBug *BT;
public:
  UndefDerefChecker() : BT(0) {}

  ExplodedNode *CheckLocation(const Stmt *S, ExplodedNode *Pred,
                              const GRState *state, SVal V, GRExprEngine &Eng);

  static void *getTag();
};

}
