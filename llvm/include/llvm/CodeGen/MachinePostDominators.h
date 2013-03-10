//=- llvm/CodeGen/MachineDominators.h ----------------------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes interfaces to post dominance information for
// target-specific code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEPOSTDOMINATORS_H
#define LLVM_CODEGEN_MACHINEPOSTDOMINATORS_H

#include "llvm/Analysis/DominatorInternals.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

namespace llvm {

///
/// PostDominatorTree Class - Concrete subclass of DominatorTree that is used
/// to compute the a post-dominator tree.
///
struct MachinePostDominatorTree : public MachineFunctionPass {
private:
  DominatorTreeBase<MachineBasicBlock> *DT;

public:
  static char ID;

  MachinePostDominatorTree();

  ~MachinePostDominatorTree();

  FunctionPass *createMachinePostDominatorTreePass();

  const std::vector<MachineBasicBlock *> &getRoots() const {
    return DT->getRoots();
  }

  MachineDomTreeNode *getRootNode() const {
    return DT->getRootNode();
  }

  MachineDomTreeNode *operator[](MachineBasicBlock *BB) const {
    return DT->getNode(BB);
  }

  MachineDomTreeNode *getNode(MachineBasicBlock *BB) const {
    return DT->getNode(BB);
  }

  bool dominates(const MachineDomTreeNode *A,
                 const MachineDomTreeNode *B) const {
    return DT->dominates(A, B);
  }

  bool dominates(const MachineBasicBlock *A, const MachineBasicBlock *B) const {
    return DT->dominates(A, B);
  }

  bool properlyDominates(const MachineDomTreeNode *A,
                         const MachineDomTreeNode *B) const {
    return DT->properlyDominates(A, B);
  }

  bool properlyDominates(const MachineBasicBlock *A,
                         const MachineBasicBlock *B) const {
    return DT->properlyDominates(A, B);
  }

  MachineBasicBlock *findNearestCommonDominator(MachineBasicBlock *A,
                                                MachineBasicBlock *B) {
    return DT->findNearestCommonDominator(A, B);
  }

  virtual bool runOnMachineFunction(MachineFunction &MF);
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;
  virtual void print(llvm::raw_ostream &OS, const Module *M = 0) const;
};
} //end of namespace llvm

#endif
