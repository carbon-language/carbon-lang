//===-- llvm/CodeGen/AllocationOrder.cpp - Allocation Order ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements an allocation order for virtual registers.
//
// The preferred allocation order for a virtual register depends on allocation
// hints and target hooks. The AllocationOrder class encapsulates all of that.
//
//===----------------------------------------------------------------------===//

#include "AllocationOrder.h"
#include "VirtRegMap.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

// Compare VirtRegMap::getRegAllocPref().
AllocationOrder::AllocationOrder(unsigned VirtReg,
                                 const VirtRegMap &VRM,
                                 const BitVector &ReservedRegs)
  : Pos(0), Reserved(ReservedRegs) {
  const TargetRegisterClass *RC = VRM.getRegInfo().getRegClass(VirtReg);
  std::pair<unsigned, unsigned> HintPair =
    VRM.getRegInfo().getRegAllocationHint(VirtReg);

  // HintPair.second is a register, phys or virt.
  Hint = HintPair.second;

  // Translate to physreg, or 0 if not assigned yet.
  if (Hint && TargetRegisterInfo::isVirtualRegister(Hint))
    Hint = VRM.getPhys(Hint);

  // Target-dependent hints require resolution.
  if (HintPair.first)
    Hint = VRM.getTargetRegInfo().ResolveRegAllocHint(HintPair.first, Hint,
                                                      VRM.getMachineFunction());

  // The hint must be a valid physreg for allocation.
  if (Hint && (!TargetRegisterInfo::isPhysicalRegister(Hint) ||
               !RC->contains(Hint) || ReservedRegs.test(Hint)))
    Hint = 0;

  // The remaining allocation order may also depend on the hint.
  tie(Begin, End) = VRM.getTargetRegInfo()
        .getAllocationOrder(RC, HintPair.first, Hint, VRM.getMachineFunction());
}

unsigned AllocationOrder::next() {
  // First take the hint.
  if (!Pos) {
    Pos = Begin;
    if (Hint)
      return Hint;
  }
  // Then look at the order from TRI.
  while(Pos != End) {
    unsigned Reg = *Pos++;
    if (Reg != Hint && !Reserved.test(Reg))
      return Reg;
  }
  return 0;
}
