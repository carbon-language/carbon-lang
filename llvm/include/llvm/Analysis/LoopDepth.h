//===- llvm/Analysis/LoopDepth.h - Loop Depth Calculation --------*- C++ -*--=//
//
// This file provides a simple class to calculate the loop depth of a 
// BasicBlock.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LOOP_DEPTH_H
#define LLVM_ANALYSIS_LOOP_DEPTH_H

#include "llvm/Pass.h"
namespace cfg {
  class LoopInfo;

class LoopDepthCalculator : public MethodPass {
  std::map<const BasicBlock*, unsigned> LoopDepth;
  void calculate(Method *M, LoopInfo &Loops);
public:
  static AnalysisID ID;            // cfg::LoopDepth Analysis ID 

  LoopDepthCalculator(AnalysisID id) { assert(id == ID); }

  // This is a pass...
  bool runOnMethod(Method *M);

  inline unsigned getLoopDepth(const BasicBlock *BB) const { 
    std::map<const BasicBlock*,unsigned>::const_iterator I = LoopDepth.find(BB);
    return I != LoopDepth.end() ? I->second : 0;
  }

  // getAnalysisUsageInfo - Provide loop depth, require loop info
  //
  virtual void getAnalysisUsageInfo(Pass::AnalysisSet &Requires,
                                    Pass::AnalysisSet &Destroyed,
                                    Pass::AnalysisSet &Provided);
};

}  // end namespace cfg

#endif
