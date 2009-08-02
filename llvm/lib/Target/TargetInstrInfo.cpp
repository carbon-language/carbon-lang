//===-- TargetInstrInfo.cpp - Target Instruction Information --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"
using namespace llvm;

TargetInstrInfo::TargetInstrInfo(const TargetInstrDesc* Desc,
                                 unsigned numOpcodes)
  : Descriptors(Desc), NumOpcodes(numOpcodes) {
}

TargetInstrInfo::~TargetInstrInfo() {
}

/// insertNoop - Insert a noop into the instruction stream at the specified
/// point.
void TargetInstrInfo::insertNoop(MachineBasicBlock &MBB, 
                                 MachineBasicBlock::iterator MI) const {
  llvm_unreachable("Target didn't implement insertNoop!");
}


bool TargetInstrInfo::isUnpredicatedTerminator(const MachineInstr *MI) const {
  const TargetInstrDesc &TID = MI->getDesc();
  if (!TID.isTerminator()) return false;
  
  // Conditional branch is a special case.
  if (TID.isBranch() && !TID.isBarrier())
    return true;
  if (!TID.isPredicable())
    return true;
  return !isPredicated(MI);
}

/// getRegClass - Get the register class for the operand, handling resolution
/// of "symbolic" pointer register classes etc.  If this is not a register
/// operand, this returns null.
const TargetRegisterClass *
TargetOperandInfo::getRegClass(const TargetRegisterInfo *TRI) const {
  if (isLookupPtrRegClass())
    return TRI->getPointerRegClass(RegClass);
  return TRI->getRegClass(RegClass);
}

