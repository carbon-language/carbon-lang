//=- LoongArchInstrInfo.cpp - LoongArch Instruction Information -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the LoongArch implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "LoongArchInstrInfo.h"
#include "LoongArch.h"

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#include "LoongArchGenInstrInfo.inc"

LoongArchInstrInfo::LoongArchInstrInfo(LoongArchSubtarget &STI)
    // FIXME: add CFSetup and CFDestroy Inst when we implement function call.
    : LoongArchGenInstrInfo() {}

void LoongArchInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator MBBI,
                                     const DebugLoc &DL, MCRegister DstReg,
                                     MCRegister SrcReg, bool KillSrc) const {
  if (LoongArch::GPRRegClass.contains(DstReg, SrcReg)) {
    BuildMI(MBB, MBBI, DL, get(LoongArch::OR), DstReg)
        .addReg(SrcReg, getKillRegState(KillSrc))
        .addReg(LoongArch::R0);
    return;
  }

  // TODO: Now, we only support GPR->GPR copies.
  llvm_unreachable("LoongArch didn't implement copyPhysReg");
}
