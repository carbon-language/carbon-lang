//===-- MSP430RegisterInfo.cpp - MSP430 Register Information --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MSP430 implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "msp430-reg-info"

#include "MSP430RegisterInfo.h"
#include "MSP430.h"
#include "MSP430MachineFunctionInfo.h"
#include "MSP430TargetMachine.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

#define GET_REGINFO_TARGET_DESC
#include "MSP430GenRegisterInfo.inc"

using namespace llvm;

// FIXME: Provide proper call frame setup / destroy opcodes.
MSP430RegisterInfo::MSP430RegisterInfo(MSP430TargetMachine &tm,
                                       const TargetInstrInfo &tii)
  : MSP430GenRegisterInfo(MSP430::PCW), TM(tm), TII(tii) {
  StackAlign = TM.getFrameLowering()->getStackAlignment();
}

const uint16_t*
MSP430RegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  const TargetFrameLowering *TFI = MF->getTarget().getFrameLowering();
  const Function* F = MF->getFunction();
  static const uint16_t CalleeSavedRegs[] = {
    MSP430::FPW, MSP430::R5W, MSP430::R6W, MSP430::R7W,
    MSP430::R8W, MSP430::R9W, MSP430::R10W, MSP430::R11W,
    0
  };
  static const uint16_t CalleeSavedRegsFP[] = {
    MSP430::R5W, MSP430::R6W, MSP430::R7W,
    MSP430::R8W, MSP430::R9W, MSP430::R10W, MSP430::R11W,
    0
  };
  static const uint16_t CalleeSavedRegsIntr[] = {
    MSP430::FPW,  MSP430::R5W,  MSP430::R6W,  MSP430::R7W,
    MSP430::R8W,  MSP430::R9W,  MSP430::R10W, MSP430::R11W,
    MSP430::R12W, MSP430::R13W, MSP430::R14W, MSP430::R15W,
    0
  };
  static const uint16_t CalleeSavedRegsIntrFP[] = {
    MSP430::R5W,  MSP430::R6W,  MSP430::R7W,
    MSP430::R8W,  MSP430::R9W,  MSP430::R10W, MSP430::R11W,
    MSP430::R12W, MSP430::R13W, MSP430::R14W, MSP430::R15W,
    0
  };

  if (TFI->hasFP(*MF))
    return (F->getCallingConv() == CallingConv::MSP430_INTR ?
            CalleeSavedRegsIntrFP : CalleeSavedRegsFP);
  else
    return (F->getCallingConv() == CallingConv::MSP430_INTR ?
            CalleeSavedRegsIntr : CalleeSavedRegs);

}

BitVector MSP430RegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();

  // Mark 4 special registers with subregisters as reserved.
  Reserved.set(MSP430::PCB);
  Reserved.set(MSP430::SPB);
  Reserved.set(MSP430::SRB);
  Reserved.set(MSP430::CGB);
  Reserved.set(MSP430::PCW);
  Reserved.set(MSP430::SPW);
  Reserved.set(MSP430::SRW);
  Reserved.set(MSP430::CGW);

  // Mark frame pointer as reserved if needed.
  if (TFI->hasFP(MF))
    Reserved.set(MSP430::FPW);

  return Reserved;
}

const TargetRegisterClass *
MSP430RegisterInfo::getPointerRegClass(const MachineFunction &MF, unsigned Kind)
                                                                         const {
  return &MSP430::GR16RegClass;
}

void
MSP430RegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                        int SPAdj, unsigned FIOperandNum,
                                        RegScavenger *RS) const {
  assert(SPAdj == 0 && "Unexpected");

  MachineInstr &MI = *II;
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction &MF = *MBB.getParent();
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();
  DebugLoc dl = MI.getDebugLoc();
  int FrameIndex = MI.getOperand(FIOperandNum).getIndex();

  unsigned BasePtr = (TFI->hasFP(MF) ? MSP430::FPW : MSP430::SPW);
  int Offset = MF.getFrameInfo()->getObjectOffset(FrameIndex);

  // Skip the saved PC
  Offset += 2;

  if (!TFI->hasFP(MF))
    Offset += MF.getFrameInfo()->getStackSize();
  else
    Offset += 2; // Skip the saved FPW

  // Fold imm into offset
  Offset += MI.getOperand(FIOperandNum + 1).getImm();

  if (MI.getOpcode() == MSP430::ADD16ri) {
    // This is actually "load effective address" of the stack slot
    // instruction. We have only two-address instructions, thus we need to
    // expand it into mov + add

    MI.setDesc(TII.get(MSP430::MOV16rr));
    MI.getOperand(FIOperandNum).ChangeToRegister(BasePtr, false);

    if (Offset == 0)
      return;

    // We need to materialize the offset via add instruction.
    unsigned DstReg = MI.getOperand(0).getReg();
    if (Offset < 0)
      BuildMI(MBB, llvm::next(II), dl, TII.get(MSP430::SUB16ri), DstReg)
        .addReg(DstReg).addImm(-Offset);
    else
      BuildMI(MBB, llvm::next(II), dl, TII.get(MSP430::ADD16ri), DstReg)
        .addReg(DstReg).addImm(Offset);

    return;
  }

  MI.getOperand(FIOperandNum).ChangeToRegister(BasePtr, false);
  MI.getOperand(FIOperandNum + 1).ChangeToImmediate(Offset);
}

unsigned MSP430RegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();

  return TFI->hasFP(MF) ? MSP430::FPW : MSP430::SPW;
}
