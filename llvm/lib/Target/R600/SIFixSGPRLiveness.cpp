//===-- SIFixSGPRLiveness.cpp - SGPR liveness adjustment ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
///
/// SGPRs are not affected by control flow. This pass adjusts SGPR liveness in
/// so that the register allocator can still correctly allocate them.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

namespace {

class SIFixSGPRLiveness : public MachineFunctionPass {
private:
  static char ID;

  const TargetInstrInfo *TII;
  MachineRegisterInfo *MRI;
  MachineDominatorTree *MD;
  MachinePostDominatorTree *MPD;

  bool isSGPR(const TargetRegisterClass *RegClass) {
    return RegClass == &AMDGPU::SReg_1RegClass ||
           RegClass == &AMDGPU::SReg_32RegClass ||
           RegClass == &AMDGPU::SReg_64RegClass ||
           RegClass == &AMDGPU::SReg_128RegClass ||
           RegClass == &AMDGPU::SReg_256RegClass;
  }

  void addKill(MachineBasicBlock::iterator I, unsigned Reg);
  MachineBasicBlock *handleUses(unsigned VirtReg, MachineBasicBlock *Begin);
  void handlePreds(MachineBasicBlock *Begin, MachineBasicBlock *End,
                   unsigned VirtReg);

  bool handleVirtReg(unsigned VirtReg);

public:
  SIFixSGPRLiveness(TargetMachine &tm);

  virtual bool runOnMachineFunction(MachineFunction &MF);

  virtual const char *getPassName() const {
    return "SI fix SGPR liveness pass";
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const;
};

} // end anonymous namespace

char SIFixSGPRLiveness::ID = 0;

SIFixSGPRLiveness::SIFixSGPRLiveness(TargetMachine &tm):
  MachineFunctionPass(ID),
  TII(tm.getInstrInfo()) {
  initializeLiveIntervalsPass(*PassRegistry::getPassRegistry());
}

void SIFixSGPRLiveness::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineDominatorTree>();
  AU.addRequired<MachinePostDominatorTree>();
  AU.setPreservesCFG();
  MachineFunctionPass::getAnalysisUsage(AU);
}

void SIFixSGPRLiveness::addKill(MachineBasicBlock::iterator I, unsigned Reg) {
  MachineBasicBlock *MBB = I->getParent();

  BuildMI(*MBB, I, DebugLoc(), TII->get(TargetOpcode::KILL)).addReg(Reg);
}

// Find the common post dominator of all uses
MachineBasicBlock *SIFixSGPRLiveness::handleUses(unsigned VirtReg,
                                                 MachineBasicBlock *Begin) {
  MachineBasicBlock *LastUse = Begin, *End = Begin;
  bool EndUsesReg = true;

  MachineRegisterInfo::use_iterator i, e;
  for (i = MRI->use_begin(VirtReg), e = MRI->use_end(); i != e; ++i) {
    MachineBasicBlock *MBB = i->getParent();
    if (LastUse == MBB)
      continue;

    LastUse = MBB;
    MBB = MPD->findNearestCommonDominator(End, MBB);

    if (MBB == LastUse)
      EndUsesReg = true;
    else if (MBB != End)
      EndUsesReg = false;

    End = MBB;
  }

  return EndUsesReg ? Begin : End;
}

// Handles predecessors separately, only add KILLs to dominated ones
void SIFixSGPRLiveness::handlePreds(MachineBasicBlock *Begin,
                                    MachineBasicBlock *End,
                                    unsigned VirtReg) {
  MachineBasicBlock::pred_iterator i, e;
  for (i = End->pred_begin(), e = End->pred_end(); i != e; ++i) {

    if (MD->dominates(End, *i))
      continue; // ignore loops

    if (MD->dominates(*i, Begin))
      continue; // too far up, abort search

    if (MD->dominates(Begin, *i)) {
      // found end of livetime
      addKill((*i)->getFirstTerminator(), VirtReg);
      continue;
    }

    handlePreds(Begin, *i, VirtReg);
  }
}

bool SIFixSGPRLiveness::handleVirtReg(unsigned VirtReg) {

  MachineInstr *Def = MRI->getVRegDef(VirtReg);
  if (!Def || MRI->use_empty(VirtReg))
    return false; // No definition or not used

  MachineBasicBlock *Begin = Def->getParent();
  MachineBasicBlock *End = handleUses(VirtReg, Begin);
  if (Begin == End)
    return false; // Defined and only used in the same block

  if (MD->dominates(Begin, End)) {
    // Lifetime dominate the end node, just kill it here
    addKill(End->getFirstNonPHI(), VirtReg);
  } else {
    // only some predecessors are dominate, handle them separately
    handlePreds(Begin, End, VirtReg);
  }
  return true;
}

bool SIFixSGPRLiveness::runOnMachineFunction(MachineFunction &MF) {
  bool Changes = false;

  MRI = &MF.getRegInfo();
  MD = &getAnalysis<MachineDominatorTree>();
  MPD = &getAnalysis<MachinePostDominatorTree>();

  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    unsigned VirtReg = TargetRegisterInfo::index2VirtReg(i);

    const TargetRegisterClass *RegClass = MRI->getRegClass(VirtReg);
    if (!isSGPR(RegClass))
      continue;

    Changes |= handleVirtReg(VirtReg);
  }

  return Changes;
}

FunctionPass *llvm::createSIFixSGPRLivenessPass(TargetMachine &tm) {
  return new SIFixSGPRLiveness(tm);
}
