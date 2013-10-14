//===-- llvm/CodeGen/LiveRegUnits.h - Live register unit set ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a Set of live register units. This can be used for ad
// hoc liveness tracking after register allocation. You can start with the
// live-ins/live-outs at the beginning/end of a block and update the information
// while walking the instructions inside the block.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LIVEREGUNITS_H
#define LLVM_CODEGEN_LIVEREGUNITS_H

#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/MC/MCRegisterInfo.h"
#include <cassert>

namespace llvm {

class MachineInstr;

/// A set of live register units with functions to track liveness when walking
/// backward/forward through a basic block.
class LiveRegUnits {
  SmallSet<unsigned, 32> LiveUnits;

public:
  /// Constructs a new empty LiveRegUnits set.
  LiveRegUnits() {
  }

  /// Constructs a new LiveRegUnits set by copying @p Other.
  LiveRegUnits(const LiveRegUnits &Other)
    : LiveUnits(Other.LiveUnits) {
  }

  /// Adds a register to the set.
  void addReg(unsigned Reg, const MCRegisterInfo &MCRI) {
    for (MCRegUnitIterator RUnits(Reg, &MCRI); RUnits.isValid(); ++RUnits)
      LiveUnits.insert(*RUnits);
  }

  /// Removes a register from the set.
  void removeReg(unsigned Reg, const MCRegisterInfo &MCRI) {
    for (MCRegUnitIterator RUnits(Reg, &MCRI); RUnits.isValid(); ++RUnits)
      LiveUnits.erase(*RUnits);
  }

  /// \brief Removes registers clobbered by the regmask operand @p Op.
  /// Note that we assume the high bits of a physical super register are not
  /// preserved unless the instruction has an implicit-use operand reading
  /// the super-register or a register unit for the upper bits is available.
  void removeRegsInMask(const MachineOperand &Op,
                        const MCRegisterInfo &MCRI) {
    const uint32_t *Mask = Op.getRegMask();
    unsigned Bit = 0;
    for (unsigned R = 0; R < MCRI.getNumRegs(); ++R) {
      if ((*Mask & (1u << Bit)) == 0)
        removeReg(R, MCRI);
      ++Bit;
      if (Bit >= 32) {
        Bit = 0;
        ++Mask;
      }
    }
  }

  /// Returns true if register @p Reg (or one of its super register) is
  /// contained in the set.
  bool contains(unsigned Reg, const MCRegisterInfo &MCRI) const {
    for (MCRegUnitIterator RUnits(Reg, &MCRI); RUnits.isValid(); ++RUnits) {
      if (LiveUnits.count(*RUnits))
        return true;
    }
    return false;
  }

  /// Simulates liveness when stepping backwards over an instruction(bundle):
  /// Defs are removed from the set, uses added.
  void stepBackward(const MachineInstr &MI, const MCRegisterInfo &MCRI) {
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

  /// \brief Simulates liveness when stepping forward over an
  /// instruction(bundle).
  ///
  /// Uses with kill flag get removed from the set, defs added. If possible
  /// use StepBackward() instead of this function because some kill flags may
  /// be missing.
  void stepForward(const MachineInstr &MI, const MCRegisterInfo &MCRI) {
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
  void addLiveIns(const MachineBasicBlock &BB, const MCRegisterInfo &MCRI) {
    for (MachineBasicBlock::livein_iterator L = BB.livein_begin(),
         LE = BB.livein_end(); L != LE; ++L) {
      addReg(*L, MCRI);
    }
  }
};

} // namespace llvm

#endif
