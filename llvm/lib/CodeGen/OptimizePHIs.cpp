//===-- OptimizePHIs.cpp - Optimize machine instruction PHIs --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass optimizes machine instruction PHIs to take advantage of
// opportunities created during DAG legalization.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "phi-opt"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Function.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumPHICycles, "Number of PHI cycles replaced");

namespace {
  class OptimizePHIs : public MachineFunctionPass {
    MachineRegisterInfo *MRI;
    const TargetInstrInfo *TII;

  public:
    static char ID; // Pass identification
    OptimizePHIs() : MachineFunctionPass(&ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

  private:
    bool IsSingleValuePHICycle(const MachineInstr *MI, unsigned &SingleValReg,
                               SmallSet<unsigned, 16> &RegsInCycle);
    bool ReplacePHICycles(MachineBasicBlock &MBB);
  };
}

char OptimizePHIs::ID = 0;
static RegisterPass<OptimizePHIs>
X("opt-phis", "Optimize machine instruction PHIs");

FunctionPass *llvm::createOptimizePHIsPass() { return new OptimizePHIs(); }

bool OptimizePHIs::runOnMachineFunction(MachineFunction &Fn) {
  MRI = &Fn.getRegInfo();
  TII = Fn.getTarget().getInstrInfo();

  // Find PHI cycles that can be replaced by a single value.  InstCombine
  // does this, but DAG legalization may introduce new opportunities, e.g.,
  // when i64 values are split up for 32-bit targets.
  bool Changed = false;
  for (MachineFunction::iterator I = Fn.begin(), E = Fn.end(); I != E; ++I)
    Changed |= ReplacePHICycles(*I);

  return Changed;
}

/// IsSingleValuePHICycle - Check if MI is a PHI where all the source operands
/// are copies of SingleValReg, possibly via copies through other PHIs.  If
/// SingleValReg is zero on entry, it is set to the register with the single
/// non-copy value.  RegsInCycle is a set used to keep track of the PHIs that
/// have been scanned.
bool OptimizePHIs::IsSingleValuePHICycle(const MachineInstr *MI,
                                         unsigned &SingleValReg,
                                         SmallSet<unsigned, 16> &RegsInCycle) {
  assert(MI->isPHI() && "IsSingleValuePHICycle expects a PHI instruction");
  unsigned DstReg = MI->getOperand(0).getReg();

  // See if we already saw this register.
  if (!RegsInCycle.insert(DstReg))
    return true;

  // Don't scan crazily complex things.
  if (RegsInCycle.size() == 16)
    return false;

  // Scan the PHI operands.
  for (unsigned i = 1; i != MI->getNumOperands(); i += 2) {
    unsigned SrcReg = MI->getOperand(i).getReg();
    if (SrcReg == DstReg)
      continue;
    const MachineInstr *SrcMI = MRI->getVRegDef(SrcReg);

    // Skip over register-to-register moves.
    unsigned MvSrcReg, MvDstReg, SrcSubIdx, DstSubIdx;
    if (SrcMI &&
        TII->isMoveInstr(*SrcMI, MvSrcReg, MvDstReg, SrcSubIdx, DstSubIdx) &&
        SrcSubIdx == 0 && DstSubIdx == 0 &&
        TargetRegisterInfo::isVirtualRegister(MvSrcReg))
      SrcMI = MRI->getVRegDef(MvSrcReg);
    if (!SrcMI)
      return false;

    if (SrcMI->isPHI()) {
      if (!IsSingleValuePHICycle(SrcMI, SingleValReg, RegsInCycle))
        return false;
    } else {
      // Fail if there is more than one non-phi/non-move register.
      if (SingleValReg != 0)
        return false;
      SingleValReg = SrcReg;
    }
  }
  return true;
}

/// ReplacePHICycles - Find PHI cycles that can be replaced by a single
/// value and remove them.
bool OptimizePHIs::ReplacePHICycles(MachineBasicBlock &MBB) {
  bool Changed = false;
  for (MachineBasicBlock::iterator
         MII = MBB.begin(), E = MBB.end(); MII != E; ) {
    MachineInstr *MI = &*MII++;
    if (!MI->isPHI())
      break;

    unsigned SingleValReg = 0;
    SmallSet<unsigned, 16> RegsInCycle;
    if (IsSingleValuePHICycle(MI, SingleValReg, RegsInCycle) &&
        SingleValReg != 0) {
      MRI->replaceRegWith(MI->getOperand(0).getReg(), SingleValReg);
      MI->eraseFromParent();
      ++NumPHICycles;
      Changed = true;
    }
  }
  return Changed;
}
