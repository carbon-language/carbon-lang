//===--- BadCallChecker.h - Bad call checker --------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines BadCallChecker, a builtin check in GRExprEngine that performs
// checks for bad callee at call sites.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/CheckerVisitor.h"

namespace clang {

class BadCallChecker : public CheckerVisitor<BadCallChecker> {
  BuiltinBug *BT;

public:
  BadCallChecker() : BT(0) {}

  static void *getTag();

  void PreVisitCallExpr(CheckerContext &C, const CallExpr *CE);
};

}
