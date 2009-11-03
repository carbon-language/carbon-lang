//===--- AttrNonNullChecker.h - Undefined arguments checker ----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines AttrNonNullChecker, a builtin check in GRExprEngine that 
// performs checks for arguments declared to have nonnull attribute.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/CheckerVisitor.h"

namespace clang {

class AttrNonNullChecker : public CheckerVisitor<AttrNonNullChecker> {
  BugType *BT;

public:
  AttrNonNullChecker() : BT(0) {}
  static void *getTag(); 
  void PreVisitCallExpr(CheckerContext &C, const CallExpr *CE);
};

}
