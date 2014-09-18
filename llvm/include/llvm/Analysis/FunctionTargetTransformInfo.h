//===- llvm/Analysis/FunctionTargetTransformInfo.h --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass wraps a TargetTransformInfo in a FunctionPass so that it can
// forward along the current Function so that we can make target specific
// decisions based on the particular subtarget specified for each Function.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_FUNCTIONTARGETTRANSFORMINFO_H
#define LLVM_ANALYSIS_FUNCTIONTARGETTRANSFORMINFO_H

#include "llvm/Pass.h"
#include "TargetTransformInfo.h"

namespace llvm {
class FunctionTargetTransformInfo final : public FunctionPass {
private:
  const Function *Fn;
  const TargetTransformInfo *TTI;

  FunctionTargetTransformInfo(const FunctionTargetTransformInfo &)
      LLVM_DELETED_FUNCTION;
  void operator=(const FunctionTargetTransformInfo &) LLVM_DELETED_FUNCTION;

public:
  static char ID;
  FunctionTargetTransformInfo();

  // Implementation boilerplate.
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  void releaseMemory() override;
  bool runOnFunction(Function &F) override;

  // Shimmed functions from TargetTransformInfo.
  void
  getUnrollingPreferences(Loop *L,
                          TargetTransformInfo::UnrollingPreferences &UP) const {
    TTI->getUnrollingPreferences(Fn, L, UP);
  }
};
}
#endif
