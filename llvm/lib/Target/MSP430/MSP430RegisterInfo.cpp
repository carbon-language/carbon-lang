//===- MSP430RegisterInfo.cpp - MSP430 Register Information ---------------===//
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

#include "MSP430.h"
#include "MSP430RegisterInfo.h"
#include "MSP430TargetMachine.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/BitVector.h"

using namespace llvm;

// FIXME: Provide proper call frame setup / destroy opcodes.
MSP430RegisterInfo::MSP430RegisterInfo(MSP430TargetMachine &tm,
                                       const TargetInstrInfo &tii)
  : MSP430GenRegisterInfo(MSP430::ADJCALLSTACKDOWN, MSP430::ADJCALLSTACKUP),
    TM(tm), TII(tii) {
  StackAlign = TM.getFrameInfo()->getStackAlignment();
}

const unsigned*
MSP430RegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  static const unsigned CalleeSavedRegs[] = {
    MSP430::FPW, MSP430::R5W, MSP430::R6W, MSP430::R7W,
    MSP430::R8W, MSP430::R9W, MSP430::R10W, MSP430::R11W,
    0
  };

  return CalleeSavedRegs;
}

const TargetRegisterClass* const*
MSP430RegisterInfo::getCalleeSavedRegClasses(const MachineFunction *MF) const {
  static const TargetRegisterClass * const CalleeSavedRegClasses[] = {
    &MSP430::GR16RegClass, &MSP430::GR16RegClass,
    &MSP430::GR16RegClass, &MSP430::GR16RegClass,
    &MSP430::GR16RegClass, &MSP430::GR16RegClass,
    &MSP430::GR16RegClass, &MSP430::GR16RegClass,
    0
  };

  return CalleeSavedRegClasses;
}

BitVector
MSP430RegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());

  // Mark 4 special registers as reserved.
  Reserved.set(MSP430::PCW);
  Reserved.set(MSP430::SPW);
  Reserved.set(MSP430::SRW);
  Reserved.set(MSP430::CGW);

  // Mark frame pointer as reserved if needed.
  if (hasFP(MF))
    Reserved.set(MSP430::FPW);

  return Reserved;
}

const TargetRegisterClass* MSP430RegisterInfo::getPointerRegClass() const {
  return &MSP430::GR16RegClass;
}


bool MSP430RegisterInfo::hasFP(const MachineFunction &MF) const {
  return NoFramePointerElim || MF.getFrameInfo()->hasVarSizedObjects();
}

bool MSP430RegisterInfo::hasReservedCallFrame(MachineFunction &MF) const {
  return !MF.getFrameInfo()->hasVarSizedObjects();
}

void MSP430RegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  if (!hasReservedCallFrame(MF)) {
    // If the stack pointer can be changed after prologue, turn the
    // adjcallstackup instruction into a 'sub SPW, <amt>' and the
    // adjcallstackdown instruction into 'add SPW, <amt>'
    // TODO: consider using push / pop instead of sub + store / add
    MachineInstr *Old = I;
    uint64_t Amount = Old->getOperand(0).getImm();
    if (Amount != 0) {
      // We need to keep the stack aligned properly.  To do this, we round the
      // amount of space needed for the outgoing arguments up to the next
      // alignment boundary.
      Amount = (Amount+StackAlign-1)/StackAlign*StackAlign;

      MachineInstr *New = 0;
      if (Old->getOpcode() == getCallFrameSetupOpcode()) {
        New = BuildMI(MF, Old->getDebugLoc(),
                      TII.get(MSP430::SUB16ri), MSP430::SPW)
          .addReg(MSP430::SPW).addImm(Amount);
      } else {
        assert(Old->getOpcode() == getCallFrameDestroyOpcode());
        // factor out the amount the callee already popped.
        uint64_t CalleeAmt = Old->getOperand(1).getImm();
        Amount -= CalleeAmt;
        if (Amount)
          New = BuildMI(MF, Old->getDebugLoc(),
                        TII.get(MSP430::ADD16ri), MSP430::SPW)
            .addReg(MSP430::SPW).addImm(Amount);
      }

      if (New) {
        // The SRW implicit def is dead.
        New->getOperand(3).setIsDead();

        // Replace the pseudo instruction with a new instruction...
        MBB.insert(I, New);
      }
    }
  } else if (I->getOpcode() == getCallFrameDestroyOpcode()) {
    // If we are performing frame pointer elimination and if the callee pops
    // something off the stack pointer, add it back.
    if (uint64_t CalleeAmt = I->getOperand(1).getImm()) {
      MachineInstr *Old = I;
      MachineInstr *New =
        BuildMI(MF, Old->getDebugLoc(), TII.get(MSP430::SUB16ri),
                MSP430::SPW).addReg(MSP430::SPW).addImm(CalleeAmt);
      // The SRW implicit def is dead.
      New->getOperand(3).setIsDead();

      MBB.insert(I, New);
    }
  }

  MBB.erase(I);
}

void
MSP430RegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                        int SPAdj, RegScavenger *RS) const {
  assert(SPAdj == 0 && "Unexpected");

  unsigned i = 0;
  MachineInstr &MI = *II;
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction &MF = *MBB.getParent();
  DebugLoc dl = MI.getDebugLoc();
  while (!MI.getOperand(i).isFI()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }

  int FrameIndex = MI.getOperand(i).getIndex();

  unsigned BasePtr = (hasFP(MF) ? MSP430::FPW : MSP430::SPW);
  int Offset = MF.getFrameInfo()->getObjectOffset(FrameIndex);

  if (!hasFP(MF))
    Offset += MF.getFrameInfo()->getStackSize();

  // Skip the saved PC
  Offset += 2;

  // Fold imm into offset
  Offset += MI.getOperand(i+1).getImm();

  if (MI.getOpcode() == MSP430::ADD16ri) {
    // This is actually "load effective address" of the stack slot
    // instruction. We have only two-address instructions, thus we need to
    // expand it into mov + add

    MI.setDesc(TII.get(MSP430::MOV16rr));
    MI.getOperand(i).ChangeToRegister(BasePtr, false);

    if (Offset == 0)
      return;

    // We need to materialize the offset via add instruction.
    unsigned DstReg = MI.getOperand(0).getReg();
    if (Offset < 0)
      BuildMI(MBB, next(II), dl, TII.get(MSP430::SUB16ri), DstReg)
        .addReg(DstReg).addImm(-Offset);
    else
      BuildMI(MBB, next(II), dl, TII.get(MSP430::ADD16ri), DstReg)
        .addReg(DstReg).addImm(Offset);

    return;
  }

  MI.getOperand(i).ChangeToRegister(BasePtr, false);
  MI.getOperand(i+1).ChangeToImmediate(Offset);
}

void MSP430RegisterInfo::emitPrologue(MachineFunction &MF) const {
  // Nothing here yet
}

void MSP430RegisterInfo::emitEpilogue(MachineFunction &MF,
                                      MachineBasicBlock &MBB) const {
  // Nothing here yet
}

unsigned MSP430RegisterInfo::getRARegister() const {
  assert(0 && "Not implemented yet!");
}

unsigned MSP430RegisterInfo::getFrameRegister(MachineFunction &MF) const {
  assert(0 && "Not implemented yet!");
}

int MSP430RegisterInfo::getDwarfRegNum(unsigned RegNum, bool isEH) const {
  assert(0 && "Not implemented yet!");
}

#include "MSP430GenRegisterInfo.inc"
