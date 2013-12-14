//===--- LivePhysRegs.cpp - Live Physical Register Set --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LivePhysRegs utility for tracking liveness of
// physical registers across machine instructions in forward or backward order.
// A more detailed description can be found in the corresponding header file.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineInstrBundle.h"
#include "llvm/Support/Debug.h"
using namespace llvm;


/// \brief Remove all registers from the set that get clobbered by the register
/// mask.
void LivePhysRegs::removeRegsInMask(const MachineOperand &MO) {
  SparseSet<unsigned>::iterator LRI = LiveRegs.begin();
  while (LRI != LiveRegs.end()) {
    if (MO.clobbersPhysReg(*LRI))
      LRI = LiveRegs.erase(LRI);
    else
      ++LRI;
  }
}

/// Simulates liveness when stepping backwards over an instruction(bundle):
/// Remove Defs, add uses. This is the recommended way of calculating liveness.
void LivePhysRegs::stepBackward(const MachineInstr &MI) {
  // Remove defined registers and regmask kills from the set.
  for (ConstMIBundleOperands O(&MI); O.isValid(); ++O) {
    if (O->isReg()) {
      if (!O->isDef())
        continue;
      unsigned Reg = O->getReg();
      if (Reg == 0)
        continue;
      removeReg(Reg);
    } else if (O->isRegMask())
      removeRegsInMask(*O);
  }

  // Add uses to the set.
  for (ConstMIBundleOperands O(&MI); O.isValid(); ++O) {
    if (!O->isReg() || !O->readsReg() || O->isUndef())
      continue;
    unsigned Reg = O->getReg();
    if (Reg == 0)
      continue;
    addReg(Reg);
  }
}

/// Simulates liveness when stepping forward over an instruction(bundle): Remove
/// killed-uses, add defs. This is the not recommended way, because it depends
/// on accurate kill flags. If possible use stepBackwards() instead of this
/// function.
void LivePhysRegs::stepForward(const MachineInstr &MI) {
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
        removeReg(Reg);
      }
    } else if (O->isRegMask())
      removeRegsInMask(*O);
  }

  // Add defs to the set.
  for (unsigned i = 0, e = Defs.size(); i != e; ++i)
    addReg(Defs[i]);
}

/// Prin the currently live registers to OS.
void LivePhysRegs::print(raw_ostream &OS) const {
  OS << "Live Registers:";
  if (!TRI) {
    OS << " (uninitialized)\n";
    return;
  }

  if (empty()) {
    OS << " (empty)\n";
    return;
  }

  for (const_iterator I = begin(), E = end(); I != E; ++I)
    OS << " " << PrintReg(*I, TRI);
  OS << "\n";
}

/// Dumps the currently live registers to the debug output.
void LivePhysRegs::dump() const {
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  dbgs() << "  " << *this;
#endif
}
