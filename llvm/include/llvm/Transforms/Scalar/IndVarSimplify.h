//===- llvm/Transforms/Scalar/IndVarSimplify.h - IV Eliminate ----*- C++ -*--=//
//
// InductionVariableSimplify - Transform induction variables in a program
//   to all use a single cannonical induction variable per loop.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_INDVARSIMPLIFY_H
#define LLVM_TRANSFORMS_SCALAR_INDVARSIMPLIFY_H

#include "llvm/Pass.h"

namespace cfg { class LoopInfo; }

struct InductionVariableSimplify : public MethodPass {
  static bool doit(Method *M, cfg::LoopInfo &Loops);

  virtual bool runOnMethod(Method *M);

  virtual void getAnalysisUsageInfo(Pass::AnalysisSet &Required,
                                    Pass::AnalysisSet &Destroyed,
                                    Pass::AnalysisSet &Provided);
};

#endif
