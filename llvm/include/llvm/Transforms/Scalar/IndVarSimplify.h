//===- llvm/Transforms/Scalar/IndVarSimplify.h - IV Eliminate ----*- C++ -*--=//
//
// InductionVariableSimplify - Transform induction variables in a program
//   to all use a single cannonical induction variable per loop.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_INDVARSIMPLIFY_H
#define LLVM_TRANSFORMS_SCALAR_INDVARSIMPLIFY_H

#include "llvm/Pass.h"

struct InductionVariableSimplify : public Pass {
  static bool doit(Method *M);

  virtual bool doPerMethodWork(Method *M) { return doit(M); }
};

#endif
