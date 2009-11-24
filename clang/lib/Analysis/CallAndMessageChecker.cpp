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
#include "clang/Analysis/PathSensitive/BugReporter.h"
#include "GRExprEngineInternalChecks.h"

using namespace clang;

namespace {
class VISIBILITY_HIDDEN UndefinedArgChecker
  : public CheckerVisitor<UndefinedArgChecker> {
  BugType *BT_call_null;
  BugType *BT_call_undef;  
  BugType *BT_call_arg;
  BugType *BT_msg_undef;
  BugType *BT_msg_arg;
public:
  UndefinedArgChecker() :
    BT_call_null(0), BT_call_undef(0), BT_call_arg(0),
    BT_msg_undef(0), BT_msg_arg(0) {}
  static void *getTag() {
    static int x = 0;
    return &x;
  }
  void PreVisitCallExpr(CheckerContext &C, const CallExpr *CE);
  void PreVisitObjCMessageExpr(CheckerContext &C, const ObjCMessageExpr *ME);
private:
  void EmitBadCall(BugType *BT, CheckerContext &C, const CallExpr *CE);
};
} // end anonymous namespace

void clang::RegisterUndefinedArgChecker(GRExprEngine &Eng) {
  Eng.registerCheck(new UndefinedArgChecker());
}

void UndefinedArgChecker::EmitBadCall(BugType *BT, CheckerContext &C,
                                      const CallExpr *CE) {
  ExplodedNode *N = C.GenerateSink();
  if (!N)
    return;
    
  EnhancedBugReport *R = new EnhancedBugReport(*BT, BT->getName(), N);
  R->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue,
                       bugreporter::GetCalleeExpr(N));
  C.EmitReport(R);
}

void UndefinedArgChecker::PreVisitCallExpr(CheckerContext &C, 
                                           const CallExpr *CE){
  
  const Expr *Callee = CE->getCallee()->IgnoreParens();
  SVal L = C.getState()->getSVal(Callee);
  
  if (L.isUndef()) {
    if (!BT_call_undef)
      BT_call_undef =
        new BuiltinBug("Called function pointer is an undefined pointer value");
    EmitBadCall(BT_call_undef, C, CE);
    return;
  }
  
  if (isa<loc::ConcreteInt>(L)) {
    if (!BT_call_null)
      BT_call_null =
        new BuiltinBug("Called function pointer is null (null dereference)");
    EmitBadCall(BT_call_null, C, CE);
  }  
  
  for (CallExpr::const_arg_iterator I = CE->arg_begin(), E = CE->arg_end();
       I != E; ++I) {
    if (C.getState()->getSVal(*I).isUndef()) {
      if (ExplodedNode *N = C.GenerateSink()) {
        if (!BT_call_arg)
          BT_call_arg = new BuiltinBug("Pass-by-value argument in function call"
                                       " is undefined");
        // Generate a report for this bug.
        EnhancedBugReport *R = new EnhancedBugReport(*BT_call_arg,
                                                     BT_call_arg->getName(), N);
        R->addRange((*I)->getSourceRange());
        R->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue, *I);
        C.EmitReport(R);
        return;
      }
    }
  }
}

void UndefinedArgChecker::PreVisitObjCMessageExpr(CheckerContext &C,
                                                  const ObjCMessageExpr *ME) {

  const GRState *state = C.getState();

  if (const Expr *receiver = ME->getReceiver())
    if (state->getSVal(receiver).isUndef()) {
      if (ExplodedNode *N = C.GenerateSink()) {
        if (!BT_msg_undef)
          BT_msg_undef =
            new BuiltinBug("Receiver in message expression is a garbage value");
        EnhancedBugReport *R =
          new EnhancedBugReport(*BT_msg_undef, BT_msg_undef->getName(), N);
        R->addRange(receiver->getSourceRange());
        R->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue,
                             receiver);
        C.EmitReport(R);
      }
      return;
    }

  // Check for any arguments that are uninitialized/undefined.
  for (ObjCMessageExpr::const_arg_iterator I = ME->arg_begin(), E = ME->arg_end();
       I != E; ++I) {
    if (state->getSVal(*I).isUndef()) {
      if (ExplodedNode *N = C.GenerateSink()) {
        if (!BT_msg_arg)
          BT_msg_arg =
            new BuiltinBug("Pass-by-value argument in message expression"
                           " is undefined");      
        // Generate a report for this bug.
        EnhancedBugReport *R = new EnhancedBugReport(*BT_msg_arg,
                                                     BT_msg_arg->getName(), N);
        R->addRange((*I)->getSourceRange());
        R->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue, *I);
        C.EmitReport(R);
        return;
      }
    }
  }
}
