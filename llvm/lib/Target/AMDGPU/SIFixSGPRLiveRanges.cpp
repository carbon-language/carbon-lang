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
/// of the registers they define in some cases.
///
/// The main case we need to handle is when a def is used in one side of a
/// branch and not another.  For example:
///
/// %def
/// IF
///   ...
///   ...
/// ELSE
///   %use
///   ...
/// ENDIF
///
/// Here we need the register allocator to avoid assigning any of the defs
/// inside of the IF to the same register as %def.  In traditional live
/// interval analysis %def is not live inside the IF branch, however, since
/// SALU instructions inside of IF will be executed even if the branch is not
/// taken, there is the chance that one of the instructions will overwrite the
/// value of %def, so the use in ELSE will see the wrong value.
///
/// The strategy we use for solving this is to add an extra use after the ENDIF:
///
/// %def
/// IF
///   ...
///   ...
/// ELSE
///   %use
///   ...
/// ENDIF
/// %use
///
/// Adding this use will make the def live throughout the IF branch, which is
/// what we want.

#include "AMDGPU.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
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

  bool runOnMachineFunction(MachineFunction &MF) override;

  const char *getPassName() const override {
    return "SI Fix SGPR live ranges";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervals>();
    AU.addRequired<MachinePostDominatorTree>();
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(SIFixSGPRLiveRanges, DEBUG_TYPE,
                      "SI Fix SGPR Live Ranges", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_DEPENDENCY(MachinePostDominatorTree)
INITIALIZE_PASS_END(SIFixSGPRLiveRanges, DEBUG_TYPE,
                    "SI Fix SGPR Live Ranges", false, false)

char SIFixSGPRLiveRanges::ID = 0;

char &llvm::SIFixSGPRLiveRangesID = SIFixSGPRLiveRanges::ID;

FunctionPass *llvm::createSIFixSGPRLiveRangesPass() {
  return new SIFixSGPRLiveRanges();
}

bool SIFixSGPRLiveRanges::runOnMachineFunction(MachineFunction &MF) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  const SIRegisterInfo *TRI = static_cast<const SIRegisterInfo *>(
      MF.getSubtarget().getRegisterInfo());
  LiveIntervals *LIS = &getAnalysis<LiveIntervals>();
 MachinePostDominatorTree *PDT = &getAnalysis<MachinePostDominatorTree>();
  std::vector<std::pair<unsigned, LiveRange *>> SGPRLiveRanges;

  // First pass, collect all live intervals for SGPRs
  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      for (const MachineOperand &MO : MI.defs()) {
        if (MO.isImplicit())
          continue;
        unsigned Def = MO.getReg();
        if (TargetRegisterInfo::isVirtualRegister(Def)) {
          if (TRI->isSGPRClass(MRI.getRegClass(Def)))
            SGPRLiveRanges.push_back(
                std::make_pair(Def, &LIS->getInterval(Def)));
        } else if (TRI->isSGPRClass(TRI->getPhysRegClass(Def))) {
            SGPRLiveRanges.push_back(
                std::make_pair(Def, &LIS->getRegUnit(Def)));
        }
      }
    }
  }

  // Second pass fix the intervals
  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end();
                                                  BI != BE; ++BI) {
    MachineBasicBlock &MBB = *BI;
    if (MBB.succ_size() < 2)
      continue;

    // We have structured control flow, so number of successors should be two.
    assert(MBB.succ_size() == 2);
    MachineBasicBlock *SuccA = *MBB.succ_begin();
    MachineBasicBlock *SuccB = *(++MBB.succ_begin());
    MachineBasicBlock *NCD = PDT->findNearestCommonDominator(SuccA, SuccB);

    if (!NCD)
      continue;

    MachineBasicBlock::iterator NCDTerm = NCD->getFirstTerminator();

    if (NCDTerm != NCD->end() && NCDTerm->getOpcode() == AMDGPU::SI_ELSE) {
      assert(NCD->succ_size() == 2);
      // We want to make sure we insert the Use after the ENDIF, not after
      // the ELSE.
      NCD = PDT->findNearestCommonDominator(*NCD->succ_begin(),
                                            *(++NCD->succ_begin()));
    }
    assert(SuccA && SuccB);
    for (std::pair<unsigned, LiveRange*> RegLR : SGPRLiveRanges) {
      unsigned Reg = RegLR.first;
      LiveRange *LR = RegLR.second;

      // FIXME: We could be smarter here.  If the register is Live-In to
      // one block, but the other doesn't have any SGPR defs, then there
      // won't be a conflict.  Also, if the branch decision is based on
      // a value in an SGPR, then there will be no conflict.
      bool LiveInToA = LIS->isLiveInToMBB(*LR, SuccA);
      bool LiveInToB = LIS->isLiveInToMBB(*LR, SuccB);

      if ((!LiveInToA && !LiveInToB) ||
          (LiveInToA && LiveInToB))
        continue;

      // This interval is live in to one successor, but not the other, so
      // we need to update its range so it is live in to both.
      DEBUG(dbgs() << "Possible SGPR conflict detected " <<  " in " << *LR <<
                      " BB#" << SuccA->getNumber() << ", BB#" <<
                      SuccB->getNumber() <<
                      " with NCD = " << NCD->getNumber() << '\n');

      // FIXME: Need to figure out how to update LiveRange here so this pass
      // will be able to preserve LiveInterval analysis.
      BuildMI(*NCD, NCD->getFirstNonPHI(), DebugLoc(),
              TII->get(AMDGPU::SGPR_USE))
              .addReg(Reg, RegState::Implicit);
      DEBUG(NCD->getFirstNonPHI()->dump());
    }
  }

  return false;
}
