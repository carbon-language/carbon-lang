//===-- llvm/CallingConvLower.cpp - Calling Conventions -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CCState class, used for lowering and implementing
// calling conventions.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

CCState::CCState(unsigned CC, const TargetMachine &tm,
                 SmallVector<CCValAssign, 16> &locs)
  : CallingConv(CC), TM(tm), MRI(*TM.getRegisterInfo()), Locs(locs) {
  // No stack is used.
  StackOffset = 0;
  
  UsedRegs.resize(MRI.getNumRegs());
}


/// MarkAllocated - Mark a register and all of its aliases as allocated.
void CCState::MarkAllocated(unsigned Reg) {
  UsedRegs[Reg/32] |= 1 << (Reg&31);
  
  if (const unsigned *RegAliases = MRI.getAliasSet(Reg))
    for (; (Reg = *RegAliases); ++RegAliases)
      UsedRegs[Reg/32] |= 1 << (Reg&31);
}
