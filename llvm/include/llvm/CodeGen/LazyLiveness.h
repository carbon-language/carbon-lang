//===- LazyLiveness.h - Lazy, CFG-invariant liveness information ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass implements a lazy liveness analysis as per "Fast Liveness Checking
// for SSA-form Programs," by Boissinot, et al.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LAZYLIVENESS_H
#define LLVM_CODEGEN_LAZYLIVENESS_H

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SparseBitVector.h"
#include <vector>

namespace llvm {

class MachineRegisterInfo;

class LazyLiveness : public MachineFunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid
  LazyLiveness() : MachineFunctionPass(&ID) { }
  
  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<MachineDominatorTree>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  
  bool runOnMachineFunction(MachineFunction &mf);

  bool vregLiveIntoMBB(unsigned vreg, MachineBasicBlock* MBB);
  
private:
  void computeBackedgeChain(MachineFunction& mf, MachineBasicBlock* MBB);
  
  typedef std::pair<MachineBasicBlock*, MachineBasicBlock*> edge_t;
  
  MachineRegisterInfo* MRI;
  
  DenseMap<MachineBasicBlock*, unsigned> preorder;
  std::vector<MachineBasicBlock*> rev_preorder;
  DenseMap<MachineBasicBlock*, SparseBitVector<128> > rv;
  DenseMap<MachineBasicBlock*, SparseBitVector<128> > tv;
  DenseSet<edge_t> backedges;
  SparseBitVector<128> backedge_source;
  SparseBitVector<128> backedge_target;
  SparseBitVector<128> calculated;
};

}

#endif

