//===- llvm/Analysis/LoopDepth.h - Loop Depth Calculation --------*- C++ -*--=//
//
// This file provides a simple class to calculate the loop depth of a 
// BasicBlock.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LOOP_DEPTH_H
#define LLVM_ANALYSIS_LOOP_DEPTH_H

#include <map>
class BasicBlock;
class Method;
namespace cfg {class Interval; }

class LoopDepthCalculator {
  map<const BasicBlock*, unsigned> LoopDepth;
  inline void AddBB(const BasicBlock *BB);    // Increment count for this block
  inline void ProcessInterval(cfg::Interval *I);
public:
  LoopDepthCalculator(Method *M);

  inline unsigned getLoopDepth(const BasicBlock *BB) const { 
    map<const BasicBlock*, unsigned>::const_iterator I = LoopDepth.find(BB);
    return I != LoopDepth.end() ? I->second : 0;
  }
};

#endif
