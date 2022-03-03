//===- LoongArchRegisterInfo.cpp - LoongArch Register Information -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the LoongArch implementation of the TargetRegisterInfo
// class.
//
//===----------------------------------------------------------------------===//

#include "LoongArchRegisterInfo.h"
#include "LoongArch.h"
#include "LoongArchSubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define GET_REGINFO_TARGET_DESC
#include "LoongArchGenRegisterInfo.inc"

LoongArchRegisterInfo::LoongArchRegisterInfo(unsigned HwMode)
    : LoongArchGenRegisterInfo(LoongArch::R1, /*DwarfFlavour*/ 0,
                               /*EHFlavor*/ 0,
                               /*PC*/ 0, HwMode) {}

const MCPhysReg *
LoongArchRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  auto &Subtarget = MF->getSubtarget<LoongArchSubtarget>();

  switch (Subtarget.getTargetABI()) {
  default:
    llvm_unreachable("Unrecognized ABI");
  case LoongArchABI::ABI_ILP32S:
  case LoongArchABI::ABI_LP64S:
    return CSR_ILP32S_LP64S_SaveList;
  case LoongArchABI::ABI_ILP32F:
  case LoongArchABI::ABI_LP64F:
    return CSR_ILP32F_LP64F_SaveList;
  case LoongArchABI::ABI_ILP32D:
  case LoongArchABI::ABI_LP64D:
    return CSR_ILP32D_LP64D_SaveList;
  }
}

const uint32_t *
LoongArchRegisterInfo::getCallPreservedMask(const MachineFunction &MF,
                                            CallingConv::ID CC) const {
  auto &Subtarget = MF.getSubtarget<LoongArchSubtarget>();

  switch (Subtarget.getTargetABI()) {
  default:
    llvm_unreachable("Unrecognized ABI");
  case LoongArchABI::ABI_ILP32S:
  case LoongArchABI::ABI_LP64S:
    return CSR_ILP32S_LP64S_RegMask;
  case LoongArchABI::ABI_ILP32F:
  case LoongArchABI::ABI_LP64F:
    return CSR_ILP32F_LP64F_RegMask;
  case LoongArchABI::ABI_ILP32D:
  case LoongArchABI::ABI_LP64D:
    return CSR_ILP32D_LP64D_RegMask;
  }
}

const uint32_t *LoongArchRegisterInfo::getNoPreservedMask() const {
  return CSR_NoRegs_RegMask;
}

BitVector
LoongArchRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  const LoongArchFrameLowering *TFI = getFrameLowering(MF);
  BitVector Reserved(getNumRegs());

  // Use markSuperRegs to ensure any register aliases are also reserved
  markSuperRegs(Reserved, LoongArch::R0);  // zero
  markSuperRegs(Reserved, LoongArch::R2);  // tp
  markSuperRegs(Reserved, LoongArch::R3);  // sp
  markSuperRegs(Reserved, LoongArch::R21); // non-allocatable
  if (TFI->hasFP(MF))
    markSuperRegs(Reserved, LoongArch::R22); // fp
  // Reserve the base register if we need to realign the stack and allocate
  // variable-sized objects at runtime.
  if (TFI->hasBP(MF))
    markSuperRegs(Reserved, LoongArchABI::getBPReg()); // bp

  assert(checkAllSuperRegsMarked(Reserved));
  return Reserved;
}

bool LoongArchRegisterInfo::isConstantPhysReg(MCRegister PhysReg) const {
  return PhysReg == LoongArch::R0;
}

Register
LoongArchRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  const TargetFrameLowering *TFI = getFrameLowering(MF);
  return TFI->hasFP(MF) ? LoongArch::R22 : LoongArch::R3;
}

void LoongArchRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                                int SPAdj,
                                                unsigned FIOperandNum,
                                                RegScavenger *RS) const {
  assert(SPAdj == 0 && "Unexpected non-zero SPAdj value");
  // TODO: Implement this when we have function calls
}
