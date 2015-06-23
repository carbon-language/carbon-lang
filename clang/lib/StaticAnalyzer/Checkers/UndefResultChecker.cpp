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
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;

namespace {
class UndefResultChecker 
  : public Checker< check::PostStmt<BinaryOperator> > {

  mutable std::unique_ptr<BugType> BT;

public:
  void checkPostStmt(const BinaryOperator *B, CheckerContext &C) const;
};
} // end anonymous namespace

void UndefResultChecker::checkPostStmt(const BinaryOperator *B,
                                       CheckerContext &C) const {
  ProgramStateRef state = C.getState();
  const LocationContext *LCtx = C.getLocationContext();
  if (state->getSVal(B, LCtx).isUndef()) {

    // Do not report assignments of uninitialized values inside swap functions.
    // This should allow to swap partially uninitialized structs
    // (radar://14129997)
    if (const FunctionDecl *EnclosingFunctionDecl =
        dyn_cast<FunctionDecl>(C.getStackFrame()->getDecl()))
      if (C.getCalleeName(EnclosingFunctionDecl) == "swap")
        return;

    // Generate an error node.
    ExplodedNode *N = C.generateSink();
    if (!N)
      return;
    
    if (!BT)
      BT.reset(
          new BuiltinBug(this, "Result of operation is garbage or undefined"));

    SmallString<256> sbuf;
    llvm::raw_svector_ostream OS(sbuf);
    const Expr *Ex = nullptr;
    bool isLeft = true;
    
    if (state->getSVal(B->getLHS(), LCtx).isUndef()) {
      Ex = B->getLHS()->IgnoreParenCasts();
      isLeft = true;
    }
    else if (state->getSVal(B->getRHS(), LCtx).isUndef()) {
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
    auto report = llvm::make_unique<BugReport>(*BT, OS.str(), N);
    if (Ex) {
      report->addRange(Ex->getSourceRange());
      bugreporter::trackNullOrUndefValue(N, Ex, *report);
    }
    else
      bugreporter::trackNullOrUndefValue(N, B, *report);
    
    C.emitReport(std::move(report));
  }
}

void ento::registerUndefResultChecker(CheckerManager &mgr) {
  mgr.registerChecker<UndefResultChecker>();
}
