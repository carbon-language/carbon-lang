//===-- SIFixSGPRLiveRanges.cpp - Fix SGPR live ranges ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// SALU instructions ignore control flow, so we need to modify the live ranges
/// of the registers they define.
///
/// The strategy is to view the entire program as if it were a single basic
/// block and calculate the intervals accordingly.  We implement this
/// by walking this list of segments for each LiveRange and setting the
/// end of each segment equal to the start of the segment that immediately
/// follows it.

#include "AMDGPU.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "si-fix-sgpr-live-ranges"

namespace {

class SIFixSGPRLiveRanges : public MachineFunctionPass {
public:
  static char ID;

public:
  SIFixSGPRLiveRanges() : MachineFunctionPass(ID) {
    initializeSIFixSGPRLiveRangesPass(*PassRegistry::getPassRegistry());
  }

  virtual bool runOnMachineFunction(MachineFunction &MF) override;

  virtual const char *getPassName() const override {
    return "SI Fix SGPR live ranges";
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervals>();
    AU.addPreserved<LiveIntervals>();
    AU.addPreserved<SlotIndexes>();
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(SIFixSGPRLiveRanges, DEBUG_TYPE,
                      "SI Fix SGPR Live Ranges", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_END(SIFixSGPRLiveRanges, DEBUG_TYPE,
                    "SI Fix SGPR Live Ranges", false, false)

char SIFixSGPRLiveRanges::ID = 0;

char &llvm::SIFixSGPRLiveRangesID = SIFixSGPRLiveRanges::ID;

FunctionPass *llvm::createSIFixSGPRLiveRangesPass() {
  return new SIFixSGPRLiveRanges();
}

bool SIFixSGPRLiveRanges::runOnMachineFunction(MachineFunction &MF) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIRegisterInfo *TRI =
      static_cast<const SIRegisterInfo *>(MF.getSubtarget().getRegisterInfo());
  LiveIntervals *LIS = &getAnalysis<LiveIntervals>();

  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end();
                                                  BI != BE; ++BI) {

    MachineBasicBlock &MBB = *BI;
    for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end();
                                                      I != E; ++I) {
      MachineInstr &MI = *I;
      MachineOperand *ExecUse = MI.findRegisterUseOperand(AMDGPU::EXEC);
      if (ExecUse)
        continue;

      for (const MachineOperand &Def : MI.operands()) {
        if (!Def.isReg() || !Def.isDef() ||!TargetRegisterInfo::isVirtualRegister(Def.getReg()))
          continue;

        const TargetRegisterClass *RC = MRI.getRegClass(Def.getReg());

        if (!TRI->isSGPRClass(RC))
          continue;
        LiveInterval &LI = LIS->getInterval(Def.getReg());
        for (unsigned i = 0, e = LI.size() - 1; i != e; ++i) {
          LiveRange::Segment &Seg = LI.segments[i];
          LiveRange::Segment &Next = LI.segments[i + 1];
          Seg.end = Next.start;
        }
      }
    }
  }

  return false;
}
