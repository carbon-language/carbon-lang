//===-- InductionVars.h - Induction Variable Recognition ---------*- C++ -*--=//
//
// This family of functions is useful for Induction variable recognition, 
// removal and optimizations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPT_INDUCTION_VARS_H
#define LLVM_OPT_INDUCTION_VARS_H

#include "llvm/Pass.h"
namespace cfg { class IntervalPartition; }

struct InductionVariableCannonicalize : public MethodPass {
  // doInductionVariableCannonicalize - Simplify induction variables in loops
  //
  static bool doIt(Function *M, cfg::IntervalPartition &IP);

  virtual bool runOnMethod(Function *M);

  // getAnalysisUsageInfo - Declare that we need IntervalPartitions
  void getAnalysisUsageInfo(Pass::AnalysisSet &Required,
                            Pass::AnalysisSet &Destroyed,
                            Pass::AnalysisSet &Provided);
};

#endif
