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
#include "llvm/Constant.h"
#include "llvm/DerivedTypes.h"
using namespace llvm;

TargetInstrInfo::TargetInstrInfo(const TargetInstrDesc* Desc,
                                 unsigned numOpcodes)
  : Descriptors(Desc), NumOpcodes(numOpcodes) {
}

TargetInstrInfo::~TargetInstrInfo() {
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

/// getInstrOperandRegClass - Return register class of the operand of an
/// instruction of the specified TargetInstrDesc.
const TargetRegisterClass*
llvm::getInstrOperandRegClass(const TargetRegisterInfo *TRI,
                        const TargetInstrDesc &II, unsigned Op) {
  if (Op >= II.getNumOperands())
    return NULL;
  if (II.OpInfo[Op].isLookupPtrRegClass())
    return TRI->getPointerRegClass();
  return TRI->getRegClass(II.OpInfo[Op].RegClass);
}
