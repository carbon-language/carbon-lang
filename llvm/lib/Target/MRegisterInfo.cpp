//===- MRegisterInfo.cpp - Target Register Information Implementation -----===//
//
// This file implements the MRegisterInfo interface.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/MRegisterInfo.h"

MRegisterInfo::MRegisterInfo(const MRegisterDesc *D, unsigned NR,
                             regclass_iterator RCB, regclass_iterator RCE,
			     int CFSO, int CFDO)
  : Desc(D), NumRegs(NR), RegClassBegin(RCB), RegClassEnd(RCE) {
  assert(NumRegs < FirstVirtualRegister &&
         "Target has too many physical registers!");

  PhysRegClasses = new const TargetRegisterClass*[NumRegs];
  for (unsigned i = 0; i != NumRegs; ++i)
    PhysRegClasses[i] = 0;

  // Fill in the PhysRegClasses map
  for (MRegisterInfo::regclass_iterator I = regclass_begin(),
         E = regclass_end(); I != E; ++I)
    for (unsigned i = 0, e = (*I)->getNumRegs(); i != e; ++i) {
      unsigned Reg = (*I)->getRegister(i);
      assert(PhysRegClasses[Reg] == 0 && "Register in more than one class?");
      PhysRegClasses[Reg] = *I;
    }

  CallFrameSetupOpcode   = CFSO;
  CallFrameDestroyOpcode = CFDO;
}


MRegisterInfo::~MRegisterInfo() {
  delete[] PhysRegClasses;
}
