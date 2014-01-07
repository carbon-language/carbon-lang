//===- AArch64RegisterInfo.cpp - AArch64 Register Information -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the AArch64 implementation of the TargetRegisterInfo
// class.
//
//===----------------------------------------------------------------------===//


#include "AArch64RegisterInfo.h"
#include "AArch64FrameLowering.h"
#include "AArch64MachineFunctionInfo.h"
#include "AArch64TargetMachine.h"
#include "MCTargetDesc/AArch64MCTargetDesc.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"

#define GET_REGINFO_TARGET_DESC
#include "AArch64GenRegisterInfo.inc"

using namespace llvm;

AArch64RegisterInfo::AArch64RegisterInfo()
  : AArch64GenRegisterInfo(AArch64::X30) {
}

const uint16_t *
AArch64RegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  return CSR_PCS_SaveList;
}

const uint32_t*
AArch64RegisterInfo::getCallPreservedMask(CallingConv::ID) const {
  return CSR_PCS_RegMask;
}

const uint32_t *AArch64RegisterInfo::getTLSDescCallPreservedMask() const {
  return TLSDesc_RegMask;
}

const TargetRegisterClass *
AArch64RegisterInfo::getCrossCopyRegClass(const TargetRegisterClass *RC) const {
  if (RC == &AArch64::FlagClassRegClass)
    return &AArch64::GPR64RegClass;

  return RC;
}



BitVector
AArch64RegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();

  Reserved.set(AArch64::XSP);
  Reserved.set(AArch64::WSP);

  Reserved.set(AArch64::XZR);
  Reserved.set(AArch64::WZR);

  if (TFI->hasFP(MF)) {
    Reserved.set(AArch64::X29);
    Reserved.set(AArch64::W29);
  }

  return Reserved;
}

static bool hasFrameOffset(int opcode) {
  return opcode != AArch64::LD1x2_8B  && opcode != AArch64::LD1x3_8B  &&
         opcode != AArch64::LD1x4_8B  && opcode != AArch64::ST1x2_8B  &&
         opcode != AArch64::ST1x3_8B  && opcode != AArch64::ST1x4_8B  &&
         opcode != AArch64::LD1x2_16B && opcode != AArch64::LD1x3_16B &&
         opcode != AArch64::LD1x4_16B && opcode != AArch64::ST1x2_16B &&
         opcode != AArch64::ST1x3_16B && opcode != AArch64::ST1x4_16B;
}

void
AArch64RegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator MBBI,
                                         int SPAdj,
                                         unsigned FIOperandNum,
                                         RegScavenger *RS) const {
  assert(SPAdj == 0 && "Cannot deal with nonzero SPAdj yet");
  MachineInstr &MI = *MBBI;
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  const AArch64FrameLowering *TFI =
   static_cast<const AArch64FrameLowering *>(MF.getTarget().getFrameLowering());

  // In order to work out the base and offset for addressing, the FrameLowering
  // code needs to know (sometimes) whether the instruction is storing/loading a
  // callee-saved register, or whether it's a more generic
  // operation. Fortunately the frame indices are used *only* for that purpose
  // and are contiguous, so we can check here.
  const std::vector<CalleeSavedInfo> &CSI = MFI->getCalleeSavedInfo();
  int MinCSFI = 0;
  int MaxCSFI = -1;

  if (CSI.size()) {
    MinCSFI = CSI[0].getFrameIdx();
    MaxCSFI = CSI[CSI.size() - 1].getFrameIdx();
  }

  int FrameIndex = MI.getOperand(FIOperandNum).getIndex();
  bool IsCalleeSaveOp = FrameIndex >= MinCSFI && FrameIndex <= MaxCSFI;

  unsigned FrameReg;
  int64_t Offset;
  Offset = TFI->resolveFrameIndexReference(MF, FrameIndex, FrameReg, SPAdj,
                                           IsCalleeSaveOp);
  // A vector load/store instruction doesn't have an offset operand.
  bool HasOffsetOp = hasFrameOffset(MI.getOpcode());
  if (HasOffsetOp)
    Offset += MI.getOperand(FIOperandNum + 1).getImm();

  // DBG_VALUE instructions have no real restrictions so they can be handled
  // easily.
  if (MI.isDebugValue()) {
    MI.getOperand(FIOperandNum).ChangeToRegister(FrameReg, /*isDef=*/ false);
    MI.getOperand(FIOperandNum + 1).ChangeToImmediate(Offset);
    return;
  }

  const AArch64InstrInfo &TII =
    *static_cast<const AArch64InstrInfo*>(MF.getTarget().getInstrInfo());
  int MinOffset, MaxOffset, OffsetScale;
  if (MI.getOpcode() == AArch64::ADDxxi_lsl0_s || !HasOffsetOp) {
    MinOffset = 0;
    MaxOffset = 0xfff;
    OffsetScale = 1;
  } else {
    // Load/store of a stack object
    TII.getAddressConstraints(MI, OffsetScale, MinOffset, MaxOffset);
  }

  // There are two situations we don't use frame + offset directly in the
  // instruction:
  // (1) The offset can't really be scaled
  // (2) Can't encode offset as it doesn't have an offset operand
  if ((Offset % OffsetScale != 0 || Offset < MinOffset || Offset > MaxOffset) ||
      (!HasOffsetOp && Offset != 0)) {
    unsigned BaseReg =
      MF.getRegInfo().createVirtualRegister(&AArch64::GPR64RegClass);
    emitRegUpdate(MBB, MBBI, MBBI->getDebugLoc(), TII,
                  BaseReg, FrameReg, BaseReg, Offset);
    FrameReg = BaseReg;
    Offset = 0;
  }

  // Negative offsets are expected if we address from FP, but for
  // now this checks nothing has gone horribly wrong.
  assert(Offset >= 0 && "Unexpected negative offset from SP");

  MI.getOperand(FIOperandNum).ChangeToRegister(FrameReg, false, false, true);
  if (HasOffsetOp)
    MI.getOperand(FIOperandNum + 1).ChangeToImmediate(Offset / OffsetScale);
}

unsigned
AArch64RegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();

  if (TFI->hasFP(MF))
    return AArch64::X29;
  else
    return AArch64::XSP;
}

bool
AArch64RegisterInfo::useFPForScavengingIndex(const MachineFunction &MF) const {
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();
  const AArch64FrameLowering *AFI
    = static_cast<const AArch64FrameLowering*>(TFI);
  return AFI->useFPForAddressing(MF);
}
