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
#include "MSP430MachineFunctionInfo.h"
#include "MSP430RegisterInfo.h"
#include "MSP430TargetMachine.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Support/ErrorHandling.h"

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
  const Function* F = MF->getFunction();
  static const unsigned CalleeSavedRegs[] = {
    MSP430::FPW, MSP430::R5W, MSP430::R6W, MSP430::R7W,
    MSP430::R8W, MSP430::R9W, MSP430::R10W, MSP430::R11W,
    0
  };
  static const unsigned CalleeSavedRegsIntr[] = {
    MSP430::FPW,  MSP430::R5W,  MSP430::R6W,  MSP430::R7W,
    MSP430::R8W,  MSP430::R9W,  MSP430::R10W, MSP430::R11W,
    MSP430::R12W, MSP430::R13W, MSP430::R14W, MSP430::R15W,
    0
  };

  return (F->getCallingConv() == CallingConv::MSP430_INTR ?
          CalleeSavedRegsIntr : CalleeSavedRegs);
}

const TargetRegisterClass *const *
MSP430RegisterInfo::getCalleeSavedRegClasses(const MachineFunction *MF) const {
  const Function* F = MF->getFunction();
  static const TargetRegisterClass * const CalleeSavedRegClasses[] = {
    &MSP430::GR16RegClass, &MSP430::GR16RegClass,
    &MSP430::GR16RegClass, &MSP430::GR16RegClass,
    &MSP430::GR16RegClass, &MSP430::GR16RegClass,
    &MSP430::GR16RegClass, &MSP430::GR16RegClass,
    0
  };
  static const TargetRegisterClass * const CalleeSavedRegClassesIntr[] = {
    &MSP430::GR16RegClass, &MSP430::GR16RegClass,
    &MSP430::GR16RegClass, &MSP430::GR16RegClass,
    &MSP430::GR16RegClass, &MSP430::GR16RegClass,
    &MSP430::GR16RegClass, &MSP430::GR16RegClass,
    &MSP430::GR16RegClass, &MSP430::GR16RegClass,
    &MSP430::GR16RegClass, &MSP430::GR16RegClass,
    0
  };

  return (F->getCallingConv() == CallingConv::MSP430_INTR ?
          CalleeSavedRegClassesIntr : CalleeSavedRegClasses);
}

BitVector MSP430RegisterInfo::getReservedRegs(const MachineFunction &MF) const {
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

const TargetRegisterClass *
MSP430RegisterInfo::getPointerRegClass(unsigned Kind) const {
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

unsigned
MSP430RegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                        int SPAdj, int *Value,
                                        RegScavenger *RS) const {
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

  // Skip the saved PC
  Offset += 2;

  if (!hasFP(MF))
    Offset += MF.getFrameInfo()->getStackSize();
  else
    Offset += 2; // Skip the saved FPW

  // Fold imm into offset
  Offset += MI.getOperand(i+1).getImm();

  if (MI.getOpcode() == MSP430::ADD16ri) {
    // This is actually "load effective address" of the stack slot
    // instruction. We have only two-address instructions, thus we need to
    // expand it into mov + add

    MI.setDesc(TII.get(MSP430::MOV16rr));
    MI.getOperand(i).ChangeToRegister(BasePtr, false);

    if (Offset == 0)
      return 0;

    // We need to materialize the offset via add instruction.
    unsigned DstReg = MI.getOperand(0).getReg();
    if (Offset < 0)
      BuildMI(MBB, llvm::next(II), dl, TII.get(MSP430::SUB16ri), DstReg)
        .addReg(DstReg).addImm(-Offset);
    else
      BuildMI(MBB, llvm::next(II), dl, TII.get(MSP430::ADD16ri), DstReg)
        .addReg(DstReg).addImm(Offset);

    return 0;
  }

  MI.getOperand(i).ChangeToRegister(BasePtr, false);
  MI.getOperand(i+1).ChangeToImmediate(Offset);
  return 0;
}

void
MSP430RegisterInfo::processFunctionBeforeFrameFinalized(MachineFunction &MF)
                                                                         const {
  // Create a frame entry for the FPW register that must be saved.
  if (hasFP(MF)) {
    int FrameIdx = MF.getFrameInfo()->CreateFixedObject(2, -4, true, false);
    assert(FrameIdx == MF.getFrameInfo()->getObjectIndexBegin() &&
           "Slot for FPW register must be last in order to be found!");
    FrameIdx = 0;
  }
}


void MSP430RegisterInfo::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();   // Prolog goes in entry BB
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MSP430MachineFunctionInfo *MSP430FI = MF.getInfo<MSP430MachineFunctionInfo>();
  MachineBasicBlock::iterator MBBI = MBB.begin();
  DebugLoc DL = (MBBI != MBB.end() ? MBBI->getDebugLoc() :
                 DebugLoc::getUnknownLoc());

  // Get the number of bytes to allocate from the FrameInfo.
  uint64_t StackSize = MFI->getStackSize();

  uint64_t NumBytes = 0;
  if (hasFP(MF)) {
    // Calculate required stack adjustment
    uint64_t FrameSize = StackSize - 2;
    NumBytes = FrameSize - MSP430FI->getCalleeSavedFrameSize();

    // Get the offset of the stack slot for the EBP register... which is
    // guaranteed to be the last slot by processFunctionBeforeFrameFinalized.
    // Update the frame offset adjustment.
    MFI->setOffsetAdjustment(-NumBytes);

    // Save FPW into the appropriate stack slot...
    BuildMI(MBB, MBBI, DL, TII.get(MSP430::PUSH16r))
      .addReg(MSP430::FPW, RegState::Kill);

    // Update FPW with the new base value...
    BuildMI(MBB, MBBI, DL, TII.get(MSP430::MOV16rr), MSP430::FPW)
      .addReg(MSP430::SPW);

    // Mark the FramePtr as live-in in every block except the entry.
    for (MachineFunction::iterator I = llvm::next(MF.begin()), E = MF.end();
         I != E; ++I)
      I->addLiveIn(MSP430::FPW);

  } else
    NumBytes = StackSize - MSP430FI->getCalleeSavedFrameSize();

  // Skip the callee-saved push instructions.
  while (MBBI != MBB.end() && (MBBI->getOpcode() == MSP430::PUSH16r))
    ++MBBI;

  if (MBBI != MBB.end())
    DL = MBBI->getDebugLoc();

  if (NumBytes) { // adjust stack pointer: SPW -= numbytes
    // If there is an SUB16ri of SPW immediately before this instruction, merge
    // the two.
    //NumBytes -= mergeSPUpdates(MBB, MBBI, true);
    // If there is an ADD16ri or SUB16ri of SPW immediately after this
    // instruction, merge the two instructions.
    // mergeSPUpdatesDown(MBB, MBBI, &NumBytes);

    if (NumBytes) {
      MachineInstr *MI =
        BuildMI(MBB, MBBI, DL, TII.get(MSP430::SUB16ri), MSP430::SPW)
        .addReg(MSP430::SPW).addImm(NumBytes);
      // The SRW implicit def is dead.
      MI->getOperand(3).setIsDead();
    }
  }
}

void MSP430RegisterInfo::emitEpilogue(MachineFunction &MF,
                                      MachineBasicBlock &MBB) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  MSP430MachineFunctionInfo *MSP430FI = MF.getInfo<MSP430MachineFunctionInfo>();
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  unsigned RetOpcode = MBBI->getOpcode();
  DebugLoc DL = MBBI->getDebugLoc();

  switch (RetOpcode) {
  case MSP430::RET:
  case MSP430::RETI: break;  // These are ok
  default:
    llvm_unreachable("Can only insert epilog into returning blocks");
  }

  // Get the number of bytes to allocate from the FrameInfo
  uint64_t StackSize = MFI->getStackSize();
  unsigned CSSize = MSP430FI->getCalleeSavedFrameSize();
  uint64_t NumBytes = 0;

  if (hasFP(MF)) {
    // Calculate required stack adjustment
    uint64_t FrameSize = StackSize - 2;
    NumBytes = FrameSize - CSSize;

    // pop FPW.
    BuildMI(MBB, MBBI, DL, TII.get(MSP430::POP16r), MSP430::FPW);
  } else
    NumBytes = StackSize - CSSize;

  // Skip the callee-saved pop instructions.
  while (MBBI != MBB.begin()) {
    MachineBasicBlock::iterator PI = prior(MBBI);
    unsigned Opc = PI->getOpcode();
    if (Opc != MSP430::POP16r && !PI->getDesc().isTerminator())
      break;
    --MBBI;
  }

  DL = MBBI->getDebugLoc();

  // If there is an ADD16ri or SUB16ri of SPW immediately before this
  // instruction, merge the two instructions.
  //if (NumBytes || MFI->hasVarSizedObjects())
  //  mergeSPUpdatesUp(MBB, MBBI, StackPtr, &NumBytes);

  if (MFI->hasVarSizedObjects()) {
    BuildMI(MBB, MBBI, DL,
            TII.get(MSP430::MOV16rr), MSP430::SPW).addReg(MSP430::FPW);
    if (CSSize) {
      MachineInstr *MI =
        BuildMI(MBB, MBBI, DL,
                TII.get(MSP430::SUB16ri), MSP430::SPW)
        .addReg(MSP430::SPW).addImm(CSSize);
      // The SRW implicit def is dead.
      MI->getOperand(3).setIsDead();
    }
  } else {
    // adjust stack pointer back: SPW += numbytes
    if (NumBytes) {
      MachineInstr *MI =
        BuildMI(MBB, MBBI, DL, TII.get(MSP430::ADD16ri), MSP430::SPW)
        .addReg(MSP430::SPW).addImm(NumBytes);
      // The SRW implicit def is dead.
      MI->getOperand(3).setIsDead();
    }
  }
}

unsigned MSP430RegisterInfo::getRARegister() const {
  return MSP430::PCW;
}

unsigned MSP430RegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  return hasFP(MF) ? MSP430::FPW : MSP430::SPW;
}

int MSP430RegisterInfo::getDwarfRegNum(unsigned RegNum, bool isEH) const {
  llvm_unreachable("Not implemented yet!");
  return 0;
}

#include "MSP430GenRegisterInfo.inc"
