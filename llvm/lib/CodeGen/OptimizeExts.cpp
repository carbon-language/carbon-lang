//===-- OptimizeExts.cpp - Optimize sign / zero extension instrs -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs optimization of sign / zero extension instructions. It
// may be extended to handle other instructions of similar property.
//
// On some targets, some instructions, e.g. X86 sign / zero extension, may
// leave the source value in the lower part of the result. This pass will
// replace (some) uses of the pre-extension value with uses of the sub-register
// of the results.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "ext-opt"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

static cl::opt<bool> Aggressive("aggressive-ext-opt", cl::Hidden,
                                cl::desc("Aggressive extension optimization"));

STATISTIC(NumReuse, "Number of extension results reused");

namespace {
  class OptimizeExts : public MachineFunctionPass {
    const TargetMachine   *TM;
    const TargetInstrInfo *TII;
    MachineRegisterInfo *MRI;
    MachineDominatorTree *DT;   // Machine dominator tree

  public:
    static char ID; // Pass identification
    OptimizeExts() : MachineFunctionPass(&ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      MachineFunctionPass::getAnalysisUsage(AU);
      if (Aggressive) {
        AU.addRequired<MachineDominatorTree>();
        AU.addPreserved<MachineDominatorTree>();
      }
    }

  private:
    bool OptimizeInstr(MachineInstr *MI, MachineBasicBlock *MBB,
                       SmallPtrSet<MachineInstr*, 8> &LocalMIs);
  };
}

char OptimizeExts::ID = 0;
static RegisterPass<OptimizeExts>
X("opt-exts", "Optimize sign / zero extensions");

FunctionPass *llvm::createOptimizeExtsPass() { return new OptimizeExts(); }

/// OptimizeInstr - If instruction is a copy-like instruction, i.e. it reads
/// a single register and writes a single register and it does not modify
/// the source, and if the source value is preserved as a sub-register of
/// the result, then replace all reachable uses of the source with the subreg
/// of the result.
/// Do not generate an EXTRACT that is used only in a debug use, as this
/// changes the code.  Since this code does not currently share EXTRACTs, just
/// ignore all debug uses.
bool OptimizeExts::OptimizeInstr(MachineInstr *MI, MachineBasicBlock *MBB,
                                 SmallPtrSet<MachineInstr*, 8> &LocalMIs) {
  bool Changed = false;
  LocalMIs.insert(MI);

  unsigned SrcReg, DstReg, SubIdx;
  if (TII->isCoalescableExtInstr(*MI, SrcReg, DstReg, SubIdx)) {
    if (TargetRegisterInfo::isPhysicalRegister(DstReg) ||
        TargetRegisterInfo::isPhysicalRegister(SrcReg))
      return false;

    MachineRegisterInfo::use_nodbg_iterator UI = MRI->use_nodbg_begin(SrcReg);
    if (++UI == MRI->use_nodbg_end())
      // No other uses.
      return false;

    // Ok, the source has other uses. See if we can replace the other uses
    // with use of the result of the extension.
    SmallPtrSet<MachineBasicBlock*, 4> ReachedBBs;
    UI = MRI->use_nodbg_begin(DstReg);
    for (MachineRegisterInfo::use_nodbg_iterator UE = MRI->use_nodbg_end();
         UI != UE; ++UI)
      ReachedBBs.insert(UI->getParent());

    bool ExtendLife = true;
    // Uses that are in the same BB of uses of the result of the instruction.
    SmallVector<MachineOperand*, 8> Uses;
    // Uses that the result of the instruction can reach.
    SmallVector<MachineOperand*, 8> ExtendedUses;

    UI = MRI->use_nodbg_begin(SrcReg);
    for (MachineRegisterInfo::use_nodbg_iterator UE = MRI->use_nodbg_end();
         UI != UE; ++UI) {
      MachineOperand &UseMO = UI.getOperand();
      MachineInstr *UseMI = &*UI;
      if (UseMI == MI)
        continue;
      if (UseMI->isPHI()) {
        ExtendLife = false;
        continue;
      }

      MachineBasicBlock *UseMBB = UseMI->getParent();
      if (UseMBB == MBB) {
        // Local uses that come after the extension.
        if (!LocalMIs.count(UseMI))
          Uses.push_back(&UseMO);
      } else if (ReachedBBs.count(UseMBB))
        // Non-local uses where the result of extension is used. Always
        // replace these unless it's a PHI.
        Uses.push_back(&UseMO);
      else if (Aggressive && DT->dominates(MBB, UseMBB))
        // We may want to extend live range of the extension result in order
        // to replace these uses.
        ExtendedUses.push_back(&UseMO);
      else {
        // Both will be live out of the def MBB anyway. Don't extend live
        // range of the extension result.
        ExtendLife = false;
        break;
      }
    }

    if (ExtendLife && !ExtendedUses.empty())
      // Ok, we'll extend the liveness of the extension result.
      std::copy(ExtendedUses.begin(), ExtendedUses.end(),
                std::back_inserter(Uses));

    // Now replace all uses.
    if (!Uses.empty()) {
      SmallPtrSet<MachineBasicBlock*, 4> PHIBBs;
      // Look for PHI uses of the extended result, we don't want to extend the
      // liveness of a PHI input. It breaks all kinds of assumptions down
      // stream. A PHI use is expected to be the kill of its source values.
      UI = MRI->use_nodbg_begin(DstReg);
      for (MachineRegisterInfo::use_nodbg_iterator UE = MRI->use_nodbg_end();
           UI != UE; ++UI)
        if (UI->isPHI())
          PHIBBs.insert(UI->getParent());

      const TargetRegisterClass *RC = MRI->getRegClass(SrcReg);
      for (unsigned i = 0, e = Uses.size(); i != e; ++i) {
        MachineOperand *UseMO = Uses[i];
        MachineInstr *UseMI = UseMO->getParent();
        MachineBasicBlock *UseMBB = UseMI->getParent();
        if (PHIBBs.count(UseMBB))
          continue;
        unsigned NewVR = MRI->createVirtualRegister(RC);
        BuildMI(*UseMBB, UseMI, UseMI->getDebugLoc(),
                TII->get(TargetOpcode::EXTRACT_SUBREG), NewVR)
          .addReg(DstReg).addImm(SubIdx);
        UseMO->setReg(NewVR);
        ++NumReuse;
        Changed = true;
      }
    }
  }

  return Changed;
}

bool OptimizeExts::runOnMachineFunction(MachineFunction &MF) {
  TM = &MF.getTarget();
  TII = TM->getInstrInfo();
  MRI = &MF.getRegInfo();
  DT = Aggressive ? &getAnalysis<MachineDominatorTree>() : 0;

  bool Changed = false;

  SmallPtrSet<MachineInstr*, 8> LocalMIs;
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    MachineBasicBlock *MBB = &*I;
    LocalMIs.clear();
    for (MachineBasicBlock::iterator MII = I->begin(), ME = I->end(); MII != ME;
         ++MII) {
      MachineInstr *MI = &*MII;
      Changed |= OptimizeInstr(MI, MBB, LocalMIs);
    }
  }

  return Changed;
}
