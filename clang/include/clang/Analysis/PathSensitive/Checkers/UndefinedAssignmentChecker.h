//===--- UndefinedAssignmentChecker.h ---------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines UndefinedAssginmentChecker, a builtin check in GRExprEngine that
// checks for assigning undefined values.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNDEFASSIGNMENTCHECKER
#define LLVM_CLANG_UNDEFASSIGNMENTCHECKER

#include "clang/Analysis/PathSensitive/CheckerVisitor.h"

namespace clang {
class UndefinedAssignmentChecker
  : public CheckerVisitor<UndefinedAssignmentChecker> {
  BugType *BT;
public:
  UndefinedAssignmentChecker() : BT(0) {}
  static void *getTag();
  virtual void PreVisitBind(CheckerContext &C, const Stmt *AssignE,
                            const Stmt *StoreE, SVal location,
                            SVal val);
};
}
#endif

