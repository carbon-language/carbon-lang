//== TraversalChecker.cpp -------------------------------------- -*- C++ -*--=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This checker prints branch statements to llvm::outs as they are encountered.
// This lets us see exactly how the ExprEngine is traversing the graph.
//
//===----------------------------------------------------------------------===//
#include "ClangSACheckers.h"
#include "clang/AST/ParentMap.h"
#include "clang/AST/StmtObjC.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

using namespace clang;
using namespace ento;

namespace {
class TraversalDumper : public Checker< check::BranchCondition,
                                        check::EndPath > {
public:
  void checkBranchCondition(const Stmt *Condition, CheckerContext &C) const;
  void checkEndPath(CheckerContext &C) const;
};
}

void TraversalDumper::checkBranchCondition(const Stmt *Condition,
                                           CheckerContext &C) const {
  // Special-case Objective-C's for-in loop, which uses the entire loop as its
  // condition. We just print the collection expression.
  const Stmt *Parent = dyn_cast<ObjCForCollectionStmt>(Condition);
  if (!Parent) {
    const ParentMap &Parents = C.getLocationContext()->getParentMap();
    Parent = Parents.getParent(Condition);
  }

  // It is mildly evil to print directly to llvm::outs() rather than emitting
  // warnings, but this ensures things do not get filtered out by the rest of
  // the static analyzer machinery.
  SourceLocation Loc = Parent->getLocStart();
  llvm::outs() << C.getSourceManager().getSpellingLineNumber(Loc) << " "
               << Parent->getStmtClassName() << "\n";
}

void TraversalDumper::checkEndPath(CheckerContext &C) const {
  llvm::outs() << "--END PATH--\n";
}

void ento::registerTraversalDumper(CheckerManager &mgr) {
  mgr.registerChecker<TraversalDumper>();
}
