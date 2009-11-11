//===- LazyValueInfo.h - Value constraint analysis --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface for lazy computation of value constraint
// information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LIVEVALUES_H
#define LLVM_ANALYSIS_LIVEVALUES_H

#include "llvm/Pass.h"

namespace llvm {

/// LazyValueInfo - This pass computes, caches, and vends lazy value constraint
/// information.
class LazyValueInfo : public FunctionPass {
public:
  static char ID;
  LazyValueInfo();

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }
  virtual void releaseMemory();
  
  virtual bool runOnFunction(Function &F) {
    // Fully lazy.
    return false;
  }
};

}  // end namespace llvm

#endif

