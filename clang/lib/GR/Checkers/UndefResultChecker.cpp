//=== UndefResultChecker.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines UndefResultChecker, a builtin check in GRExprEngine that 
// performs checks for undefined results of non-assignment binary operators.
//
//===----------------------------------------------------------------------===//

#include "GRExprEngineInternalChecks.h"
#include "clang/GR/BugReporter/BugType.h"
#include "clang/GR/PathSensitive/CheckerVisitor.h"
#include "clang/GR/PathSensitive/GRExprEngine.h"

using namespace clang;

namespace {
class UndefResultChecker 
  : public CheckerVisitor<UndefResultChecker> {

  BugType *BT;
  
public:
  UndefResultChecker() : BT(0) {}
  static void *getTag() { static int tag = 0; return &tag; }
  void PostVisitBinaryOperator(CheckerContext &C, const BinaryOperator *B);
};
} // end anonymous namespace

void clang::RegisterUndefResultChecker(GRExprEngine &Eng) {
  Eng.registerCheck(new UndefResultChecker());
}

void UndefResultChecker::PostVisitBinaryOperator(CheckerContext &C, 
                                                 const BinaryOperator *B) {
  const GRState *state = C.getState();
  if (state->getSVal(B).isUndef()) {
    // Generate an error node.
    ExplodedNode *N = C.generateSink();
    if (!N)
      return;
    
    if (!BT)
      BT = new BuiltinBug("Result of operation is garbage or undefined");

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
    EnhancedBugReport *report = new EnhancedBugReport(*BT, OS.str(), N);
    if (Ex) {
      report->addRange(Ex->getSourceRange());
      report->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue, Ex);
    }
    else
      report->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue, B);
    C.EmitReport(report);
  }
}
