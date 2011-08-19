//== ReturnUndefChecker.cpp -------------------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines ReturnUndefChecker, which is a path-sensitive
// check which looks for undefined or garbage values being returned to the
// caller.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"

using namespace clang;
using namespace ento;

namespace {
class ReturnUndefChecker : 
    public Checker< check::PreStmt<ReturnStmt> > {
  mutable llvm::OwningPtr<BuiltinBug> BT;
public:
  void checkPreStmt(const ReturnStmt *RS, CheckerContext &C) const;
};
}

void ReturnUndefChecker::checkPreStmt(const ReturnStmt *RS,
                                      CheckerContext &C) const {
 
  const Expr *RetE = RS->getRetValue();
  if (!RetE)
    return;
  
  if (!C.getState()->getSVal(RetE).isUndef())
    return;
  
  ExplodedNode *N = C.generateSink();

  if (!N)
    return;
  
  if (!BT)
    BT.reset(new BuiltinBug("Garbage return value",
                            "Undefined or garbage value returned to caller"));
    
  BugReport *report = 
    new BugReport(*BT, BT->getDescription(), N);

  report->addRange(RetE->getSourceRange());
  report->addVisitor(bugreporter::getTrackNullOrUndefValueVisitor(N, RetE));

  C.EmitReport(report);
}

void ento::registerReturnUndefChecker(CheckerManager &mgr) {
  mgr.registerChecker<ReturnUndefChecker>();
}
