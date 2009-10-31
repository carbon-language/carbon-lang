//== DivZeroChecker.h - Division by zero checker ----------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines DivZeroChecker, a builtin check in GRExprEngine that performs
// checks for division by zeros.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/CheckerVisitor.h"

namespace clang {

class DivZeroChecker : public CheckerVisitor<DivZeroChecker> {
  BuiltinBug *BT;
public:
  DivZeroChecker() : BT(0) {}

  static void *getTag();
  void PreVisitBinaryOperator(CheckerContext &C, const BinaryOperator *B);
};

}
