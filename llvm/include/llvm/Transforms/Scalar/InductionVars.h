//===-- InductionVars.h - Induction Variable Recognition ---------*- C++ -*--=//
//
// This family of functions is useful for Induction variable recognition, 
// removal and optimizations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPT_INDUCTION_VARS_H
#define LLVM_OPT_INDUCTION_VARS_H

#include "llvm/Pass.h"

struct InductionVariableCannonicalize : public MethodPass {
  // doInductionVariableCannonicalize - Simplify induction variables in loops
  //
  static bool doIt(Method *M);

  virtual bool runOnMethod(Method *M) {
    return doIt(M);
  }
};

#endif
