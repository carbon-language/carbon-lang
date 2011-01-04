//===-------- EdgeBundles.h - Bundles of CFG edges --------------*- c++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The EdgeBundles analysis forms equivalence classes of CFG edges such that all
// edges leaving a machine basic block are in the same bundle, and all edges
// leaving a basic block are in the same bundle.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_EDGEBUNDLES_H
#define LLVM_CODEGEN_EDGEBUNDLES_H

#include "llvm/ADT/IntEqClasses.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

namespace llvm {

class EdgeBundles : public MachineFunctionPass {
  const MachineFunction *MF;

  /// EC - Each edge bundle is an equivalence class. The keys are:
  ///   2*BB->getNumber()   -> Ingoing bundle.
  ///   2*BB->getNumber()+1 -> Outgoing bundle.
  IntEqClasses EC;

public:
  static char ID;
  EdgeBundles() : MachineFunctionPass(ID) {}

  /// getBundle - Return the ingoing (Out = false) or outgoing (Out = true)
  /// bundle number for basic block #N
  unsigned getBundle(unsigned N, bool Out) const { return EC[2 * N + Out]; }

  /// getNumBundles - Return the total number of bundles in the CFG.
  unsigned getNumBundles() const { return EC.getNumClasses(); }

  /// getMachineFunction - Return the last machine function computed.
  const MachineFunction *getMachineFunction() const { return MF; }

  /// view - Visualize the annotated bipartite CFG with Graphviz.
  void view() const;

private:
  virtual bool runOnMachineFunction(MachineFunction&);
  virtual void getAnalysisUsage(AnalysisUsage&) const;
};

/// Specialize WriteGraph, the standard implementation won't work.
raw_ostream &WriteGraph(raw_ostream &O, const EdgeBundles &G,
                        bool ShortNames = false,
                        const std::string &Title = "");

} // end namespace llvm

#endif
