//===-- lib/CodeGen/MachineInstrBundle.cpp --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineInstrBundle.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
using namespace llvm;

namespace {
  class UnpackMachineBundles : public MachineFunctionPass {
  public:
    static char ID; // Pass identification
    UnpackMachineBundles() : MachineFunctionPass(ID) {
      initializeUnpackMachineBundlesPass(*PassRegistry::getPassRegistry());
    }

    virtual bool runOnMachineFunction(MachineFunction &MF);
  };
} // end anonymous namespace

char UnpackMachineBundles::ID = 0;
INITIALIZE_PASS(UnpackMachineBundles, "unpack-mi-bundle",
                "Unpack machine instruction bundles", false, false)

FunctionPass *llvm::createUnpackMachineBundlesPass() {
  return new UnpackMachineBundles();
}

bool UnpackMachineBundles::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    MachineBasicBlock *MBB = &*I;

    for (MachineBasicBlock::instr_iterator MII = MBB->instr_begin(),
           MIE = MBB->instr_end(); MII != MIE; ) {
      MachineInstr *MI = &*MII;

      // Remove BUNDLE instruction and the InsideBundle flags from bundled
      // instructions.
      if (MI->isBundle()) {
        while (++MII != MIE && MII->isInsideBundle()) {
          MII->setIsInsideBundle(false);
          for (unsigned i = 0, e = MII->getNumOperands(); i != e; ++i) {
            MachineOperand &MO = MII->getOperand(i);
            if (MO.isReg() && MO.isInternalRead())
              MO.setIsInternalRead(false);
          }
        }
        MI->eraseFromParent();

        Changed = true;
        continue;
      }

      ++MII;
    }
  }

  return Changed;
}

/// finalizeBundle - Finalize a machine instruction bundle which includes
/// a sequence of instructions starting from FirstMI to LastMI (inclusive).
/// This routine adds a BUNDLE instruction to represent the bundle, it adds
/// IsInternalRead markers to MachineOperands which are defined inside the
/// bundle, and it copies externally visible defs and uses to the BUNDLE
/// instruction.
void llvm::finalizeBundle(MachineBasicBlock &MBB,
                          MachineBasicBlock::instr_iterator FirstMI,
                          MachineBasicBlock::instr_iterator LastMI) {
  const TargetMachine &TM = MBB.getParent()->getTarget();
  const TargetInstrInfo *TII = TM.getInstrInfo();
  const TargetRegisterInfo *TRI = TM.getRegisterInfo();

  MachineInstrBuilder MIB = BuildMI(MBB, FirstMI, FirstMI->getDebugLoc(),
                                    TII->get(TargetOpcode::BUNDLE));

  SmallVector<unsigned, 8> LocalDefs;
  SmallSet<unsigned, 8> LocalDefSet;
  SmallSet<unsigned, 8> DeadDefSet;
  SmallSet<unsigned, 8> KilledDefSet;
  SmallVector<unsigned, 8> ExternUses;
  SmallSet<unsigned, 8> ExternUseSet;
  SmallSet<unsigned, 8> KilledUseSet;
  SmallSet<unsigned, 8> UndefUseSet;
  SmallVector<MachineOperand*, 4> Defs;
  do {
    for (unsigned i = 0, e = FirstMI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = FirstMI->getOperand(i);
      if (!MO.isReg())
        continue;
      if (MO.isDef()) {
        Defs.push_back(&MO);
        continue;
      }

      unsigned Reg = MO.getReg();
      if (!Reg)
        continue;
      assert(TargetRegisterInfo::isPhysicalRegister(Reg));
      if (LocalDefSet.count(Reg)) {
        MO.setIsInternalRead();
        if (MO.isKill())
          // Internal def is now killed.
          KilledDefSet.insert(Reg);
      } else {
        if (ExternUseSet.insert(Reg)) {
          ExternUses.push_back(Reg);
          if (MO.isUndef())
            UndefUseSet.insert(Reg);
        }
        if (MO.isKill())
          // External def is now killed.
          KilledUseSet.insert(Reg);
      }
    }

    for (unsigned i = 0, e = Defs.size(); i != e; ++i) {
      MachineOperand &MO = *Defs[i];
      unsigned Reg = MO.getReg();
      if (!Reg)
        continue;

      if (LocalDefSet.insert(Reg)) {
        LocalDefs.push_back(Reg);
        if (MO.isDead()) {
          DeadDefSet.insert(Reg);
        }
      } else {
        // Re-defined inside the bundle, it's no longer killed.
        KilledDefSet.erase(Reg);
        if (!MO.isDead())
          // Previously defined but dead.
          DeadDefSet.erase(Reg);
      }

      if (!MO.isDead()) {
        for (const unsigned *SubRegs = TRI->getSubRegisters(Reg);
             unsigned SubReg = *SubRegs; ++SubRegs) {
          if (LocalDefSet.insert(SubReg))
            LocalDefs.push_back(SubReg);
        }
      }
    }

    FirstMI->setIsInsideBundle();
    Defs.clear();
  } while (FirstMI++ != LastMI);

  SmallSet<unsigned, 8> Added;
  for (unsigned i = 0, e = LocalDefs.size(); i != e; ++i) {
    unsigned Reg = LocalDefs[i];
    if (Added.insert(Reg)) {
      // If it's not live beyond end of the bundle, mark it dead.
      bool isDead = DeadDefSet.count(Reg) || KilledDefSet.count(Reg);
      MIB.addReg(Reg, getDefRegState(true) | getDeadRegState(isDead) |
                 getImplRegState(true));
    }
  }

  for (unsigned i = 0, e = ExternUses.size(); i != e; ++i) {
    unsigned Reg = ExternUses[i];
    bool isKill = KilledUseSet.count(Reg);
    bool isUndef = UndefUseSet.count(Reg);
    MIB.addReg(Reg, getKillRegState(isKill) | getUndefRegState(isUndef) |
               getImplRegState(true));
  }
}
