//=== StackAddrLeakChecker.cpp ------------------------------------*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines stack address leak checker, which checks if an invalid 
// stack address is stored into a global or heap location. See CERT DCL30-C.
//
//===----------------------------------------------------------------------===//

#include "GRExprEngineInternalChecks.h"
#include "clang/Checker/BugReporter/BugType.h"
#include "clang/Checker/PathSensitive/Checker.h"
#include "clang/Checker/PathSensitive/GRState.h"

using namespace clang;

namespace {
class StackAddrLeakChecker : public Checker {
  BuiltinBug *BT_stackleak;

public:
  StackAddrLeakChecker() : BT_stackleak(0) {}
  static void *getTag() {
    static int x;
    return &x;
  }

  void EvalEndPath(GREndPathNodeBuilder &B, void *tag, GRExprEngine &Eng);
};
}

void clang::RegisterStackAddrLeakChecker(GRExprEngine &Eng) {
  Eng.registerCheck(new StackAddrLeakChecker());
}

void StackAddrLeakChecker::EvalEndPath(GREndPathNodeBuilder &B, void *tag,
                                       GRExprEngine &Eng) {
  SaveAndRestore<bool> OldHasGen(B.HasGeneratedNode);
  const GRState *state = B.getState();
  TranslationUnitDecl *TU = Eng.getContext().getTranslationUnitDecl();

  // Check each global variable if it contains a MemRegionVal of a stack
  // variable declared in the function we are leaving.
  for (DeclContext::decl_iterator I = TU->decls_begin(), E = TU->decls_end();
       I != E; ++I) {
    if (VarDecl *VD = dyn_cast<VarDecl>(*I)) {
      const LocationContext *LCtx = B.getPredecessor()->getLocationContext();
      SVal L = state->getLValue(VD, LCtx);
      SVal V = state->getSVal(cast<Loc>(L));
      if (loc::MemRegionVal *RV = dyn_cast<loc::MemRegionVal>(&V)) {
        const MemRegion *R = RV->getRegion();

        if (const StackSpaceRegion *SSR = 
                              dyn_cast<StackSpaceRegion>(R->getMemorySpace())) {
          const StackFrameContext *ValSFC = SSR->getStackFrame();
          const StackFrameContext *CurSFC = LCtx->getCurrentStackFrame();
          // If the global variable holds a location in the current stack frame,
          // emit a warning.
          if (ValSFC == CurSFC) {
            // The variable is declared in the function scope which we are 
            // leaving. Keeping this variable's address in a global variable
            // is dangerous.

            // FIXME: better warning location.
            
            ExplodedNode *N = B.generateNode(state, tag, B.getPredecessor());
            if (N) {
              if (!BT_stackleak)
                BT_stackleak = new BuiltinBug("Stack address leak",
                        "Stack address was saved into a global variable. "
                        "is dangerous because the address will become invalid "
                        "after returning from the function.");
              BugReport *R = new BugReport(*BT_stackleak, 
                                           BT_stackleak->getDescription(), N);
              Eng.getBugReporter().EmitReport(R);
            }
          }
        }
      }
    }
  }
}
