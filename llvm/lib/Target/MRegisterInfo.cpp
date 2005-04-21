//===- MRegisterInfo.cpp - Target Register Information Implementation -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MRegisterInfo interface.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/MRegisterInfo.h"

namespace llvm {

MRegisterInfo::MRegisterInfo(const MRegisterDesc *D, unsigned NR,
                             regclass_iterator RCB, regclass_iterator RCE,
			     int CFSO, int CFDO)
  : Desc(D), NumRegs(NR), RegClassBegin(RCB), RegClassEnd(RCE) {
  assert(NumRegs < FirstVirtualRegister &&
         "Target has too many physical registers!");

  CallFrameSetupOpcode   = CFSO;
  CallFrameDestroyOpcode = CFDO;
}

MRegisterInfo::~MRegisterInfo() {}

std::vector<bool> MRegisterInfo::getAllocatableSet(MachineFunction &MF) const {
  std::vector<bool> Allocatable(NumRegs);
  for (MRegisterInfo::regclass_iterator I = regclass_begin(),
         E = regclass_end(); I != E; ++I) {
    const TargetRegisterClass *RC = *I;
    for (TargetRegisterClass::iterator I = RC->allocation_order_begin(MF),
           E = RC->allocation_order_end(MF); I != E; ++I)
      Allocatable[*I] = true;
  }
  return Allocatable;
}

} // End llvm namespace
