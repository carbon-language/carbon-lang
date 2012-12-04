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

#define DEBUG_TYPE "regalloc"
#include "AllocationOrder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

// Compare VirtRegMap::getRegAllocPref().
AllocationOrder::AllocationOrder(unsigned VirtReg,
                                 const VirtRegMap &VRM,
                                 const RegisterClassInfo &RegClassInfo)
  : Pos(0) {
  const MachineFunction &MF = VRM.getMachineFunction();
  const TargetRegisterInfo *TRI = &VRM.getTargetRegInfo();
  Order = RegClassInfo.getOrder(MF.getRegInfo().getRegClass(VirtReg));
  TRI->getRegAllocationHints(VirtReg, Order, Hints, MF, &VRM);

  DEBUG({
    if (!Hints.empty()) {
      dbgs() << "hints:";
      for (unsigned I = 0, E = Hints.size(); I != E; ++I)
        dbgs() << ' ' << PrintReg(Hints[I], TRI);
      dbgs() << '\n';
    }
  });
}

bool AllocationOrder::isHint(unsigned PhysReg) const {
  return std::find(Hints.begin(), Hints.end(), PhysReg) != Hints.end();
}

unsigned AllocationOrder::next() {
  if (Pos < Hints.size())
    return Hints[Pos++];
  ArrayRef<MCPhysReg>::iterator I = Order.begin() + (Pos - Hints.size());
  ArrayRef<MCPhysReg>::iterator E = Order.end();
  while (I != E) {
    unsigned Reg = *I++;
    ++Pos;
    if (!isHint(Reg))
      return Reg;
  }
  return 0;
}
