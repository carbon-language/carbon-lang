//===--- UndefinedArgChecker.h - Undefined arguments checker ----*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines BadCallChecker, a builtin check in GRExprEngine that performs
// checks for undefined arguments.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/CheckerVisitor.h"

namespace clang {

class UndefinedArgChecker : public CheckerVisitor<UndefinedArgChecker> {
  BugType *BT;

public:
  UndefinedArgChecker() : BT(0) {}

  static void *getTag();

  void PreVisitCallExpr(CheckerContext &C, const CallExpr *CE);
};

}
