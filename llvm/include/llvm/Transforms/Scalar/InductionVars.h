//===-- InductionVars.h - Induction Variable Recognition ---------*- C++ -*--=//
//
// This family of functions is useful for Induction variable recognition, 
// removal and optimizations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPT_INDUCTION_VARS_H
#define LLVM_OPT_INDUCTION_VARS_H

#include "llvm/Pass.h"
class IntervalPartition;

struct InductionVariableCannonicalize : public FunctionPass {
  // doInductionVariableCannonicalize - Simplify induction variables in loops
  //
  static bool doIt(Function *F, IntervalPartition &IP);

  virtual bool runOnFunction(Function *F);

  // getAnalysisUsage - Declare that we need IntervalPartitions
  void getAnalysisUsage(AnalysisUsage &AU) const;
};

#endif
