//=== ZeroSizedVLAChecker.cpp - Undefined dereference checker ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines ZeorSizedVLAChecker, a builtin check in GRExprEngine that 
// performs checks for declaration of VLA of zero size.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/Checker.h"

namespace clang {

class ZeroSizedVLAChecker : public Checker {
  BugType *BT;

public:
  ZeroSizedVLAChecker() : BT(0) {}
  static void *getTag();
  ExplodedNode *CheckType(QualType T, ExplodedNode *Pred, 
                          const GRState *state, Stmt *S, GRExprEngine &Eng);
};

}
