//===-- LiveInterval.cpp - Live Interval Representation -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LiveRegUnits utility for tracking liveness of
// physical register units across machine instructions in forward or backward
// order.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/LiveRegUnits.h"
#include "llvm/CodeGen/MachineInstrBundle.h"
using namespace llvm;

/// Return true if the given MachineOperand clobbers the given register unit.
/// A register unit is only clobbered if all its super-registers are clobbered.
static bool operClobbersUnit(const MachineOperand *MO, unsigned Unit,
                             const MCRegisterInfo *MCRI) {
  for (MCRegUnitRootIterator RI(Unit, MCRI); RI.isValid(); ++RI) {
    for (MCSuperRegIterator SI(*RI, MCRI, true); SI.isValid(); ++SI) {
      if (!MO->clobbersPhysReg(*SI))
        return false;
    }
  }
  return true;
}

/// We assume the high bits of a physical super register are not preserved
/// unless the instruction has an implicit-use operand reading the
/// super-register or a register unit for the upper bits is available.
void LiveRegUnits::removeRegsInMask(const MachineOperand &Op,
                                    const MCRegisterInfo &MCRI) {
  SparseSet<unsigned>::iterator LUI = LiveUnits.begin();
  while (LUI != LiveUnits.end()) {
    if (operClobbersUnit(&Op, *LUI, &MCRI))
      LUI = LiveUnits.erase(LUI);
    else
      ++LUI;
  }
}

void LiveRegUnits::stepBackward(const MachineInstr &MI,
                                const MCRegisterInfo &MCRI) {
  // Remove defined registers and regmask kills from the set.
  for (ConstMIBundleOperands O(&MI); O.isValid(); ++O) {
    if (O->isReg()) {
      if (!O->isDef())
        continue;
      unsigned Reg = O->getReg();
      if (Reg == 0)
        continue;
      removeReg(Reg, MCRI);
    } else if (O->isRegMask()) {
      removeRegsInMask(*O, MCRI);
    }
  }
  // Add uses to the set.
  for (ConstMIBundleOperands O(&MI); O.isValid(); ++O) {
    if (!O->isReg() || !O->readsReg() || O->isUndef())
      continue;
    unsigned Reg = O->getReg();
    if (Reg == 0)
      continue;
    addReg(Reg, MCRI);
  }
}

/// Uses with kill flag get removed from the set, defs added. If possible
/// use StepBackward() instead of this function because some kill flags may
/// be missing.
void LiveRegUnits::stepForward(const MachineInstr &MI,
                               const MCRegisterInfo &MCRI) {
  SmallVector<unsigned, 4> Defs;
  // Remove killed registers from the set.
  for (ConstMIBundleOperands O(&MI); O.isValid(); ++O) {
    if (O->isReg()) {
      unsigned Reg = O->getReg();
      if (Reg == 0)
        continue;
      if (O->isDef()) {
        if (!O->isDead())
          Defs.push_back(Reg);
      } else {
        if (!O->isKill())
          continue;
        assert(O->isUse());
        removeReg(Reg, MCRI);
      }
    } else if (O->isRegMask()) {
      removeRegsInMask(*O, MCRI);
    }
  }
  // Add defs to the set.
  for (unsigned i = 0, e = Defs.size(); i != e; ++i) {
    addReg(Defs[i], MCRI);
  }
}

/// Adds all registers in the live-in list of block @p BB.
void LiveRegUnits::addLiveIns(const MachineBasicBlock *MBB,
                              const MCRegisterInfo &MCRI) {
  for (MachineBasicBlock::livein_iterator L = MBB->livein_begin(),
         LE = MBB->livein_end(); L != LE; ++L) {
    addReg(*L, MCRI);
  }
}
