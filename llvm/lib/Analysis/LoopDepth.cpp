//===- LoopDepth.cpp - Loop Depth Calculation --------------------*- C++ -*--=//
//
// This file provides a simple class to calculate the loop depth of a 
// BasicBlock.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopDepth.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Method.h"
#include <algorithm>

AnalysisID cfg::LoopDepthCalculator::ID(AnalysisID::create<cfg::LoopDepthCalculator>());

bool cfg::LoopDepthCalculator::runOnMethod(Method *M) {
  calculate(M, getAnalysis<LoopInfo>());
  return false;
}

void cfg::LoopDepthCalculator::calculate(Method *M, LoopInfo &Loops) {
  for (Method::iterator I = M->begin(), E = M->end(); I != E; ++I)
    LoopDepth[*I] = Loops.getLoopDepth(*I);
}

// getAnalysisUsageInfo - Provide loop depth, require loop info
//
void cfg::LoopDepthCalculator::getAnalysisUsageInfo(Pass::AnalysisSet &Requires,
                                                  Pass::AnalysisSet &Destroyed,
                                                  Pass::AnalysisSet &Provided) {
  Provided.push_back(ID);
  Requires.push_back(LoopInfo::ID);
}

