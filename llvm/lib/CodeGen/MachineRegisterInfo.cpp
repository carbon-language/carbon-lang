//===-- lib/Codegen/MachineRegisterInfo.cpp -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of the MachineRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineRegisterInfo.h"
using namespace llvm;

MachineRegisterInfo::MachineRegisterInfo(const MRegisterInfo &MRI) {
  VRegInfo.reserve(256);
  UsedPhysRegs.resize(MRI.getNumRegs());
  
  // Create the physreg use/def lists.
  PhysRegUseDefLists = new MachineOperand*[MRI.getNumRegs()];
  memset(PhysRegUseDefLists, 0, sizeof(MachineOperand*)*MRI.getNumRegs());
}

MachineRegisterInfo::~MachineRegisterInfo() {
#ifndef NDEBUG
  for (unsigned i = 0, e = VRegInfo.size(); i != e; ++i)
    assert(VRegInfo[i].second == 0 && "Vreg use list non-empty still?");
#endif
  delete [] PhysRegUseDefLists;
}

/// HandleVRegListReallocation - We just added a virtual register to the
/// VRegInfo info list and it reallocated.  Update the use/def lists info
/// pointers.
void MachineRegisterInfo::HandleVRegListReallocation() {
  // The back pointers for the vreg lists point into the previous vector.
  // Update them to point to their correct slots.
  for (unsigned i = 0, e = VRegInfo.size(); i != e; ++i) {
    MachineOperand *List = VRegInfo[i].second;
    if (!List) continue;
    // Update the back-pointer to be accurate once more.
    List->Contents.Reg.Prev = &VRegInfo[i].second;
  }
}


/// getVRegDef - Return the machine instr that defines the specified virtual
/// register or null if none is found.  This assumes that the code is in SSA
/// form, so there should only be one definition.
MachineInstr *MachineRegisterInfo::getVRegDef(unsigned Reg) const {
  assert(Reg-MRegisterInfo::FirstVirtualRegister < VRegInfo.size() &&
         "Invalid vreg!");
  for (reg_iterator I = reg_begin(Reg), E = reg_end(); I != E; ++I) {
    // Since we are in SSA form, we can stop at the first definition.
    if (I->isDef())
      return I->getParent();
  }
  return 0;
}
