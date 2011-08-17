//=== UndefResultChecker.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines UndefResultChecker, a builtin check in ExprEngine that 
// performs checks for undefined results of non-assignment binary operators.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"

using namespace clang;
using namespace ento;

namespace {
class UndefResultChecker 
  : public Checker< check::PostStmt<BinaryOperator> > {

  mutable llvm::OwningPtr<BugType> BT;
  
public:
  void checkPostStmt(const BinaryOperator *B, CheckerContext &C) const;
};
} // end anonymous namespace

void UndefResultChecker::checkPostStmt(const BinaryOperator *B,
                                       CheckerContext &C) const {
  const ProgramState *state = C.getState();
  if (state->getSVal(B).isUndef()) {
    // Generate an error node.
    ExplodedNode *N = C.generateSink();
    if (!N)
      return;
    
    if (!BT)
      BT.reset(new BuiltinBug("Result of operation is garbage or undefined"));

    llvm::SmallString<256> sbuf;
    llvm::raw_svector_ostream OS(sbuf);
    const Expr *Ex = NULL;
    bool isLeft = true;
    
    if (state->getSVal(B->getLHS()).isUndef()) {
      Ex = B->getLHS()->IgnoreParenCasts();
      isLeft = true;
    }
    else if (state->getSVal(B->getRHS()).isUndef()) {
      Ex = B->getRHS()->IgnoreParenCasts();
      isLeft = false;
    }
    
    if (Ex) {
      OS << "The " << (isLeft ? "left" : "right")
         << " operand of '"
         << BinaryOperator::getOpcodeStr(B->getOpcode())
         << "' is a garbage value";
    }          
    else {
      // Neither operand was undefined, but the result is undefined.
      OS << "The result of the '"
         << BinaryOperator::getOpcodeStr(B->getOpcode())
         << "' expression is undefined";
    }
    BugReport *report = new BugReport(*BT, OS.str(), N);
    if (Ex) {
      report->addRange(Ex->getSourceRange());
      report->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue, Ex);
    }
    else
      report->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue, B);
    C.EmitReport(report);
  }
}

void ento::registerUndefResultChecker(CheckerManager &mgr) {
  mgr.registerChecker<UndefResultChecker>();
}
