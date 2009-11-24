//===--- CallAndMessageChecker.cpp ------------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines CallAndMessageChecker, a builtin checker that checks for various
// errors of call and objc message expressions.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/CheckerVisitor.h"
#include "clang/Analysis/PathSensitive/BugReporter.h"
#include "clang/AST/ParentMap.h"
#include "GRExprEngineInternalChecks.h"

using namespace clang;

namespace {
class VISIBILITY_HIDDEN CallAndMessageChecker
  : public CheckerVisitor<CallAndMessageChecker> {
  BugType *BT_call_null;
  BugType *BT_call_undef;  
  BugType *BT_call_arg;
  BugType *BT_msg_undef;
  BugType *BT_msg_arg;
  BugType *BT_struct_ret;
  BugType *BT_void_ptr;
public:
  CallAndMessageChecker() :
    BT_call_null(0), BT_call_undef(0), BT_call_arg(0),
    BT_msg_undef(0), BT_msg_arg(0), BT_struct_ret(0), BT_void_ptr(0) {}

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

void clang::RegisterCallAndMessageChecker(GRExprEngine &Eng) {
  Eng.registerCheck(new CallAndMessageChecker());
}

void CallAndMessageChecker::EmitBadCall(BugType *BT, CheckerContext &C,
                                        const CallExpr *CE) {
  ExplodedNode *N = C.GenerateSink();
  if (!N)
    return;
    
  EnhancedBugReport *R = new EnhancedBugReport(*BT, BT->getName(), N);
  R->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue,
                       bugreporter::GetCalleeExpr(N));
  C.EmitReport(R);
}

void CallAndMessageChecker::PreVisitCallExpr(CheckerContext &C, 
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

void CallAndMessageChecker::PreVisitObjCMessageExpr(CheckerContext &C,
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
  for (ObjCMessageExpr::const_arg_iterator I = ME->arg_begin(),
         E = ME->arg_end(); I != E; ++I) {
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

  // Check if the receiver was nil and then return value a struct.
  if (const Expr *Receiver = ME->getReceiver()) {
    SVal L_untested = state->getSVal(Receiver);
    // Assume that the receiver is not NULL.
    DefinedOrUnknownSVal L = cast<DefinedOrUnknownSVal>(L_untested);
    const GRState *StNotNull = state->Assume(L, true);

    // Assume that the receiver is NULL.
    const GRState *StNull = state->Assume(L, false);

    if (StNull) {
      QualType RetTy = ME->getType();
      if (RetTy->isRecordType()) {
        if (C.getPredecessor()->getParentMap().isConsumedExpr(ME)) {
          // The [0 ...] expressions will return garbage.  Flag either an
          // explicit or implicit error.  Because of the structure of this
          // function we currently do not bifurfacte the state graph at
          // this point.
          // FIXME: We should bifurcate and fill the returned struct with
          //  garbage.
          if (ExplodedNode* N = C.GenerateSink(StNull)) {
            if (!StNotNull) {
              if (!BT_struct_ret) {
                std::string sbuf;
                llvm::raw_string_ostream os(sbuf);
                os << "The receiver in the message expression is 'nil' and "
                  "results in the returned value (of type '"
                   << ME->getType().getAsString()
                   << "') to be garbage or otherwise undefined";
                BT_struct_ret = new BuiltinBug(os.str().c_str());
              }
              
              EnhancedBugReport *R = new EnhancedBugReport(*BT_struct_ret, 
                                                   BT_struct_ret->getName(), N);
              R->addRange(Receiver->getSourceRange());
              R->addVisitorCreator(bugreporter::registerTrackNullOrUndefValue, 
                                   Receiver);
              C.EmitReport(R);
              return;
            }
            else
              // Do not report implicit bug.
              return;
          }
        }
      } else {
        ASTContext &Ctx = C.getASTContext();
        if (RetTy != Ctx.VoidTy) {
          if (C.getPredecessor()->getParentMap().isConsumedExpr(ME)) {
            // sizeof(void *)
            const uint64_t voidPtrSize = Ctx.getTypeSize(Ctx.VoidPtrTy);
            // sizeof(return type)
            const uint64_t returnTypeSize = Ctx.getTypeSize(ME->getType());
            
            if (voidPtrSize < returnTypeSize) {
              if (ExplodedNode* N = C.GenerateSink(StNull)) {
                if (!StNotNull) {
                  if (!BT_struct_ret) {
                    std::string sbuf;
                    llvm::raw_string_ostream os(sbuf);
                    os << "The receiver in the message expression is 'nil' and "
                      "results in the returned value (of type '"
                       << ME->getType().getAsString()
                       << "' and of size "
                       << returnTypeSize / 8
                       << " bytes) to be garbage or otherwise undefined";
                    BT_void_ptr = new BuiltinBug(os.str().c_str());
                  }
              
                  EnhancedBugReport *R = new EnhancedBugReport(*BT_void_ptr, 
                                                     BT_void_ptr->getName(), N);
                  R->addRange(Receiver->getSourceRange());
                  R->addVisitorCreator(
                          bugreporter::registerTrackNullOrUndefValue, Receiver);
                  C.EmitReport(R);
                  return;
                } else
                  // Do not report implicit bug.
                  return;
              }
            }
            else if (!StNotNull) {
              // Handle the safe cases where the return value is 0 if the
              // receiver is nil.
              //
              // FIXME: For now take the conservative approach that we only
              // return null values if we *know* that the receiver is nil.
              // This is because we can have surprises like:
              //
              //   ... = [[NSScreens screens] objectAtIndex:0];
              //
              // What can happen is that [... screens] could return nil, but
              // it most likely isn't nil.  We should assume the semantics
              // of this case unless we have *a lot* more knowledge.
              //
              SVal V = C.getValueManager().makeZeroVal(ME->getType());
              C.GenerateNode(StNull->BindExpr(ME, V));
              return;
            }
          }
        }
      }
    }
    // Do not propagate null state.
    if (StNotNull)
      C.GenerateNode(StNotNull);
  }
}
