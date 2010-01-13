//===-- OptimizeExts.cpp - Optimize sign / zero extension instrs -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
      AU.addRequired<MachineDominatorTree>();
      AU.addPreserved<MachineDominatorTree>();
    }
  };
}

char OptimizeExts::ID = 0;
static RegisterPass<OptimizeExts>
X("opt-exts", "Optimize sign / zero extensions");

FunctionPass *llvm::createOptimizeExtsPass() { return new OptimizeExts(); }

bool OptimizeExts::runOnMachineFunction(MachineFunction &MF) {
  TM = &MF.getTarget();
  TII = TM->getInstrInfo();
  MRI = &MF.getRegInfo();
  DT = &getAnalysis<MachineDominatorTree>();

  bool Changed = false;

  SmallPtrSet<MachineInstr*, 8> LocalMIs;
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    MachineBasicBlock *MBB = &*I;
    for (MachineBasicBlock::iterator MII = I->begin(), ME = I->end(); MII != ME;
         ++MII) {
      MachineInstr *MI = &*MII;
      LocalMIs.insert(MI);

      unsigned SrcReg, DstReg, SubIdx;
      if (TII->isCoalescableExtInstr(*MI, SrcReg, DstReg, SubIdx)) {
        if (TargetRegisterInfo::isPhysicalRegister(DstReg) ||
            TargetRegisterInfo::isPhysicalRegister(SrcReg))
          continue;

        MachineRegisterInfo::use_iterator UI = MRI->use_begin(SrcReg);
        if (++UI == MRI->use_end())
          // No other uses.
          continue;

        // Ok, the source has other uses. See if we can replace the other uses
        // with use of the result of the extension.
        
        SmallPtrSet<MachineBasicBlock*, 4> ReachedBBs;
        UI = MRI->use_begin(DstReg);
        for (MachineRegisterInfo::use_iterator UE = MRI->use_end(); UI != UE;
             ++UI)
          ReachedBBs.insert(UI->getParent());

        bool ExtendLife = true;
        SmallVector<MachineOperand*, 8> Uses;
        SmallVector<MachineOperand*, 8> ExtendedUses;

        UI = MRI->use_begin(SrcReg);
        for (MachineRegisterInfo::use_iterator UE = MRI->use_end(); UI != UE;
             ++UI) {
          MachineOperand &UseMO = UI.getOperand();
          MachineInstr *UseMI = &*UI;
          if (UseMI == MI)
            continue;
          MachineBasicBlock *UseMBB = UseMI->getParent();
          if (UseMBB == MBB) {
            // Local uses that come after the extension.
            if (!LocalMIs.count(UseMI))
              Uses.push_back(&UseMO);
          } else if (ReachedBBs.count(UseMBB))
            // Non-local uses where the result of extension is used. Always
            // replace these.
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
          const TargetRegisterClass *RC = MRI->getRegClass(SrcReg);
          for (unsigned i = 0, e = Uses.size(); i != e; ++i) {
            MachineOperand *UseMO = Uses[i];
            MachineInstr *UseMI = UseMO->getParent();
            MachineBasicBlock *UseMBB = UseMI->getParent();
            unsigned NewVR = MRI->createVirtualRegister(RC);
            BuildMI(*UseMBB, UseMI, UseMI->getDebugLoc(),
                    TII->get(TargetInstrInfo::EXTRACT_SUBREG), NewVR)
              .addReg(DstReg).addImm(SubIdx);
            UseMO->setReg(NewVR);
            ++NumReuse;
            Changed = true;
          }
        }
      }
    }
  }

  return Changed;
}
