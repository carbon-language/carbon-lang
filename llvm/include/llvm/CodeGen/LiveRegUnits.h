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

#include "llvm/ADT/SparseSet.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include <cassert>

namespace llvm {

class MachineInstr;

/// A set of live register units with functions to track liveness when walking
/// backward/forward through a basic block.
class LiveRegUnits {
  SparseSet<unsigned> LiveUnits;

  LiveRegUnits(const LiveRegUnits&) LLVM_DELETED_FUNCTION;
  LiveRegUnits &operator=(const LiveRegUnits&) LLVM_DELETED_FUNCTION;
public:
  /// \brief Constructs a new empty LiveRegUnits set.
  LiveRegUnits() {}

  void init(const TargetRegisterInfo *TRI) {
    LiveUnits.clear();
    LiveUnits.setUniverse(TRI->getNumRegs());
  }

  void clear() { LiveUnits.clear(); }

  bool empty() const { return LiveUnits.empty(); }

  /// \brief Adds a register to the set.
  void addReg(unsigned Reg, const MCRegisterInfo &MCRI) {
    for (MCRegUnitIterator RUnits(Reg, &MCRI); RUnits.isValid(); ++RUnits)
      LiveUnits.insert(*RUnits);
  }

  /// \brief Removes a register from the set.
  void removeReg(unsigned Reg, const MCRegisterInfo &MCRI) {
    for (MCRegUnitIterator RUnits(Reg, &MCRI); RUnits.isValid(); ++RUnits)
      LiveUnits.erase(*RUnits);
  }

  /// \brief Removes registers clobbered by the regmask operand @p Op.
  void removeRegsInMask(const MachineOperand &Op, const MCRegisterInfo &MCRI);

  /// \brief Returns true if register @p Reg (or one of its super register) is
  /// contained in the set.
  bool contains(unsigned Reg, const MCRegisterInfo &MCRI) const {
    for (MCRegUnitIterator RUnits(Reg, &MCRI); RUnits.isValid(); ++RUnits) {
      if (LiveUnits.count(*RUnits))
        return true;
    }
    return false;
  }

  /// \brief Simulates liveness when stepping backwards over an
  /// instruction(bundle): Remove Defs, add uses.
  void stepBackward(const MachineInstr &MI, const MCRegisterInfo &MCRI);

  /// \brief Simulates liveness when stepping forward over an
  /// instruction(bundle): Remove killed-uses, add defs.
  void stepForward(const MachineInstr &MI, const MCRegisterInfo &MCRI);

  /// \brief Adds all registers in the live-in list of block @p BB.
  void addLiveIns(const MachineBasicBlock *MBB, const MCRegisterInfo &MCRI);
};

} // namespace llvm

#endif
