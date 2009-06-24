//===- llvm/Analysis/LoopDependenceAnalysis.h --------------- -*- C++ -*---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// LoopDependenceAnalysis is an LLVM pass that analyses dependences in memory
// accesses in loops.
//
// Please note that this is work in progress and the interface is subject to
// change.
//
// TODO: adapt as interface progresses
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LOOP_DEPENDENCE_ANALYSIS_H
#define LLVM_ANALYSIS_LOOP_DEPENDENCE_ANALYSIS_H

#include "llvm/Analysis/LoopPass.h"

namespace llvm {

  class AnalysisUsage;
  class LoopPass;
  class ScalarEvolution;

  class LoopDependenceAnalysis : public LoopPass {
    Loop *L;
    ScalarEvolution *SE;

  public:
    static char ID; // Class identification, replacement for typeinfo
    LoopDependenceAnalysis() : LoopPass(&ID) {}

    bool runOnLoop(Loop*, LPPassManager&);

    virtual void getAnalysisUsage(AnalysisUsage&) const;
  }; // class LoopDependenceAnalysis


  // createLoopDependenceAnalysisPass - This creates an instance of the
  // LoopDependenceAnalysis pass.
  //
  LoopPass *createLoopDependenceAnalysisPass();

} // namespace llvm

#endif /* LLVM_ANALYSIS_LOOP_DEPENDENCE_ANALYSIS_H */
