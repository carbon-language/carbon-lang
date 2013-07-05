//===-- SystemZRegisterInfo.cpp - SystemZ register information ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SystemZRegisterInfo.h"
#include "SystemZTargetMachine.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

#define GET_REGINFO_TARGET_DESC
#include "SystemZGenRegisterInfo.inc"

using namespace llvm;

SystemZRegisterInfo::SystemZRegisterInfo(SystemZTargetMachine &tm)
  : SystemZGenRegisterInfo(SystemZ::R14D), TM(tm) {}

const uint16_t*
SystemZRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  static const uint16_t CalleeSavedRegs[] = {
    SystemZ::R6D,  SystemZ::R7D,  SystemZ::R8D,  SystemZ::R9D,
    SystemZ::R10D, SystemZ::R11D, SystemZ::R12D, SystemZ::R13D,
    SystemZ::R14D, SystemZ::R15D,
    SystemZ::F8D,  SystemZ::F9D,  SystemZ::F10D, SystemZ::F11D,
    SystemZ::F12D, SystemZ::F13D, SystemZ::F14D, SystemZ::F15D,
    0
  };

  return CalleeSavedRegs;
}

BitVector
SystemZRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();

  if (TFI->hasFP(MF)) {
    // R11D is the frame pointer.  Reserve all aliases.
    Reserved.set(SystemZ::R11D);
    Reserved.set(SystemZ::R11W);
    Reserved.set(SystemZ::R10Q);
  }

  // R15D is the stack pointer.  Reserve all aliases.
  Reserved.set(SystemZ::R15D);
  Reserved.set(SystemZ::R15W);
  Reserved.set(SystemZ::R14Q);
  return Reserved;
}

void
SystemZRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator MI,
                                         int SPAdj, unsigned FIOperandNum,
                                         RegScavenger *RS) const {
  assert(SPAdj == 0 && "Outgoing arguments should be part of the frame");

  MachineBasicBlock &MBB = *MI->getParent();
  MachineFunction &MF = *MBB.getParent();
  const SystemZInstrInfo &TII =
    *static_cast<const SystemZInstrInfo*>(TM.getInstrInfo());
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();
  DebugLoc DL = MI->getDebugLoc();

  // Decompose the frame index into a base and offset.
  int FrameIndex = MI->getOperand(FIOperandNum).getIndex();
  unsigned BasePtr = getFrameRegister(MF);
  int64_t Offset = (TFI->getFrameIndexOffset(MF, FrameIndex) +
                    MI->getOperand(FIOperandNum + 1).getImm());

  // Special handling of dbg_value instructions.
  if (MI->isDebugValue()) {
    MI->getOperand(FIOperandNum).ChangeToRegister(BasePtr, /*isDef*/ false);
    MI->getOperand(FIOperandNum + 1).ChangeToImmediate(Offset);
    return;
  }

  // See if the offset is in range, or if an equivalent instruction that
  // accepts the offset exists.
  unsigned Opcode = MI->getOpcode();
  unsigned OpcodeForOffset = TII.getOpcodeForOffset(Opcode, Offset);
  if (OpcodeForOffset)
    MI->getOperand(FIOperandNum).ChangeToRegister(BasePtr, false);
  else {
    // Create an anchor point that is in range.  Start at 0xffff so that
    // can use LLILH to load the immediate.
    int64_t OldOffset = Offset;
    int64_t Mask = 0xffff;
    do {
      Offset = OldOffset & Mask;
      OpcodeForOffset = TII.getOpcodeForOffset(Opcode, Offset);
      Mask >>= 1;
      assert(Mask && "One offset must be OK");
    } while (!OpcodeForOffset);

    unsigned ScratchReg =
      MF.getRegInfo().createVirtualRegister(&SystemZ::ADDR64BitRegClass);
    int64_t HighOffset = OldOffset - Offset;

    if (MI->getDesc().TSFlags & SystemZII::HasIndex
        && MI->getOperand(FIOperandNum + 2).getReg() == 0) {
      // Load the offset into the scratch register and use it as an index.
      // The scratch register then dies here.
      TII.loadImmediate(MBB, MI, ScratchReg, HighOffset);
      MI->getOperand(FIOperandNum).ChangeToRegister(BasePtr, false);
      MI->getOperand(FIOperandNum + 2).ChangeToRegister(ScratchReg,
                                                        false, false, true);
    } else {
      // Load the anchor address into a scratch register.
      unsigned LAOpcode = TII.getOpcodeForOffset(SystemZ::LA, HighOffset);
      if (LAOpcode)
        BuildMI(MBB, MI, DL, TII.get(LAOpcode),ScratchReg)
          .addReg(BasePtr).addImm(HighOffset).addReg(0);
      else {
        // Load the high offset into the scratch register and use it as
        // an index.
        TII.loadImmediate(MBB, MI, ScratchReg, HighOffset);
        BuildMI(MBB, MI, DL, TII.get(SystemZ::AGR),ScratchReg)
          .addReg(ScratchReg, RegState::Kill).addReg(BasePtr);
      }

      // Use the scratch register as the base.  It then dies here.
      MI->getOperand(FIOperandNum).ChangeToRegister(ScratchReg,
                                                    false, false, true);
    }
  }
  MI->setDesc(TII.get(OpcodeForOffset));
  MI->getOperand(FIOperandNum + 1).ChangeToImmediate(Offset);
}

unsigned
SystemZRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();
  return TFI->hasFP(MF) ? SystemZ::R11D : SystemZ::R15D;
}
