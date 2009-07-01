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
#include "llvm/Support/raw_ostream.h"
#include <iosfwd>

namespace llvm {

  class AliasAnalysis;
  class AnalysisUsage;
  class ScalarEvolution;
  class Value;

  class LoopDependenceAnalysis : public LoopPass {
    Loop *L;
    AliasAnalysis *AA;
    ScalarEvolution *SE;

  public:
    static char ID; // Class identification, replacement for typeinfo
    LoopDependenceAnalysis() : LoopPass(&ID) {}

    /// TODO: docs
    bool isDependencePair(const Value*, const Value*) const;
    bool depends(Value*, Value*);

    bool runOnLoop(Loop*, LPPassManager&);

    virtual void getAnalysisUsage(AnalysisUsage&) const;

    void print(raw_ostream&, const Module* = 0) const;
    virtual void print(std::ostream&, const Module* = 0) const;
  }; // class LoopDependenceAnalysis


  // createLoopDependenceAnalysisPass - This creates an instance of the
  // LoopDependenceAnalysis pass.
  //
  LoopPass *createLoopDependenceAnalysisPass();

} // namespace llvm

#endif /* LLVM_ANALYSIS_LOOP_DEPENDENCE_ANALYSIS_H */
