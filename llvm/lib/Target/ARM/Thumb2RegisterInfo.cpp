//===- Thumb2RegisterInfo.cpp - Thumb-2 Register Information -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Thumb-2 implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMAddressingModes.h"
#include "ARMBaseInstrInfo.h"
#include "ARMMachineFunctionInfo.h"
#include "ARMSubtarget.h"
#include "Thumb2InstrInfo.h"
#include "Thumb2RegisterInfo.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/LLVMContext.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
using namespace llvm;

Thumb2RegisterInfo::Thumb2RegisterInfo(const ARMBaseInstrInfo &tii,
                                       const ARMSubtarget &sti)
  : ARMBaseRegisterInfo(tii, sti) {
}

/// emitLoadConstPool - Emits a load from constpool to materialize the
/// specified immediate.
void Thumb2RegisterInfo::emitLoadConstPool(MachineBasicBlock &MBB,
                                           MachineBasicBlock::iterator &MBBI,
                                           DebugLoc dl,
                                           unsigned DestReg, unsigned SubIdx,
                                           int Val,
                                           ARMCC::CondCodes Pred,
                                           unsigned PredReg) const {
  MachineFunction &MF = *MBB.getParent();
  MachineConstantPool *ConstantPool = MF.getConstantPool();
  Constant *C = 
             MF.getFunction()->getContext().getConstantInt(Type::Int32Ty, Val);
  unsigned Idx = ConstantPool->getConstantPoolIndex(C, 4);

  BuildMI(MBB, MBBI, dl, TII.get(ARM::t2LDRpci))
    .addReg(DestReg, getDefRegState(true), SubIdx)
    .addConstantPoolIndex(Idx).addImm((int64_t)ARMCC::AL).addReg(0);
}

static unsigned
negativeOffsetOpcode(unsigned opcode)
{
  switch (opcode) {
  case ARM::t2LDRi12:   return ARM::t2LDRi8;
  case ARM::t2LDRHi12:  return ARM::t2LDRHi8;
  case ARM::t2LDRBi12:  return ARM::t2LDRBi8;
  case ARM::t2LDRSHi12: return ARM::t2LDRSHi8;
  case ARM::t2LDRSBi12: return ARM::t2LDRSBi8;
  case ARM::t2STRi12:   return ARM::t2STRi8;
  case ARM::t2STRBi12:  return ARM::t2STRBi8;
  case ARM::t2STRHi12:  return ARM::t2STRHi8;

  case ARM::t2LDRi8:
  case ARM::t2LDRHi8:
  case ARM::t2LDRBi8:
  case ARM::t2LDRSHi8:
  case ARM::t2LDRSBi8:
  case ARM::t2STRi8:
  case ARM::t2STRBi8:
  case ARM::t2STRHi8:
    return opcode;

  default:
    break;
  }

  return 0;
}

static unsigned
positiveOffsetOpcode(unsigned opcode)
{
  switch (opcode) {
  case ARM::t2LDRi8:   return ARM::t2LDRi12;
  case ARM::t2LDRHi8:  return ARM::t2LDRHi12;
  case ARM::t2LDRBi8:  return ARM::t2LDRBi12;
  case ARM::t2LDRSHi8: return ARM::t2LDRSHi12;
  case ARM::t2LDRSBi8: return ARM::t2LDRSBi12;
  case ARM::t2STRi8:   return ARM::t2STRi12;
  case ARM::t2STRBi8:  return ARM::t2STRBi12;
  case ARM::t2STRHi8:  return ARM::t2STRHi12;

  case ARM::t2LDRi12:
  case ARM::t2LDRHi12:
  case ARM::t2LDRBi12:
  case ARM::t2LDRSHi12:
  case ARM::t2LDRSBi12:
  case ARM::t2STRi12:
  case ARM::t2STRBi12:
  case ARM::t2STRHi12:
    return opcode;

  default:
    break;
  }

  return 0;
}

static unsigned
immediateOffsetOpcode(unsigned opcode)
{
  switch (opcode) {
  case ARM::t2LDRs:   return ARM::t2LDRi12;
  case ARM::t2LDRHs:  return ARM::t2LDRHi12;
  case ARM::t2LDRBs:  return ARM::t2LDRBi12;
  case ARM::t2LDRSHs: return ARM::t2LDRSHi12;
  case ARM::t2LDRSBs: return ARM::t2LDRSBi12;
  case ARM::t2STRs:   return ARM::t2STRi12;
  case ARM::t2STRBs:  return ARM::t2STRBi12;
  case ARM::t2STRHs:  return ARM::t2STRHi12;

  case ARM::t2LDRi12:
  case ARM::t2LDRHi12:
  case ARM::t2LDRBi12:
  case ARM::t2LDRSHi12:
  case ARM::t2LDRSBi12:
  case ARM::t2STRi12:
  case ARM::t2STRBi12:
  case ARM::t2STRHi12:
  case ARM::t2LDRi8:
  case ARM::t2LDRHi8:
  case ARM::t2LDRBi8:
  case ARM::t2LDRSHi8:
  case ARM::t2LDRSBi8:
  case ARM::t2STRi8:
  case ARM::t2STRBi8:
  case ARM::t2STRHi8:
    return opcode;

  default:
    break;
  }

  return 0;
}

bool Thumb2RegisterInfo::
requiresRegisterScavenging(const MachineFunction &MF) const {
  // FIXME
  return false;
}

int Thumb2RegisterInfo::
rewriteFrameIndex(MachineInstr &MI, unsigned FrameRegIdx,
                  unsigned FrameReg, int Offset) const 
{
  unsigned Opcode = MI.getOpcode();
  const TargetInstrDesc &Desc = MI.getDesc();
  unsigned AddrMode = (Desc.TSFlags & ARMII::AddrModeMask);
  bool isSub = false;

  // Memory operands in inline assembly always use AddrModeT2_i12
  if (Opcode == ARM::INLINEASM)
    AddrMode = ARMII::AddrModeT2_i12; // FIXME. mode for thumb2?
  
  if (Opcode == getOpcode(ARMII::ADDri)) {
    Offset += MI.getOperand(FrameRegIdx+1).getImm();
    if (Offset == 0) {
      // Turn it into a move.
      MI.setDesc(TII.get(getOpcode(ARMII::MOVr)));
      MI.getOperand(FrameRegIdx).ChangeToRegister(FrameReg, false);
      MI.RemoveOperand(FrameRegIdx+1);
      return 0;
    } else if (Offset < 0) {
      Offset = -Offset;
      isSub = true;
      MI.setDesc(TII.get(getOpcode(ARMII::SUBri)));
    }

    // Common case: small offset, fits into instruction.
    if (ARM_AM::getT2SOImmVal(Offset) != -1) {
      // Replace the FrameIndex with sp / fp
      MI.getOperand(FrameRegIdx).ChangeToRegister(FrameReg, false);
      MI.getOperand(FrameRegIdx+1).ChangeToImmediate(Offset);
      return 0;
    }

    // Otherwise, extract 8 adjacent bits from the immediate into this
    // t2ADDri/t2SUBri.
    unsigned RotAmt = CountLeadingZeros_32(Offset);
    if (RotAmt > 24)
      RotAmt = 24;
    unsigned ThisImmVal = Offset & ARM_AM::rotr32(0xff000000U, RotAmt);

    // We will handle these bits from offset, clear them.
    Offset &= ~ThisImmVal;

    assert(ARM_AM::getT2SOImmVal(ThisImmVal) != -1 &&
           "Bit extraction didn't work?");
    MI.getOperand(FrameRegIdx+1).ChangeToImmediate(ThisImmVal);
  } else {
    // AddrModeT2_so cannot handle any offset. If there is no offset
    // register then we change to an immediate version.
    if (AddrMode == ARMII::AddrModeT2_so) {
      unsigned OffsetReg = MI.getOperand(FrameRegIdx+1).getReg();
      if (OffsetReg != 0) {
        MI.getOperand(FrameRegIdx).ChangeToRegister(FrameReg, false);
        return Offset;
      }
      
      MI.RemoveOperand(FrameRegIdx+1);
      MI.getOperand(FrameRegIdx+1).ChangeToImmediate(0);
      Opcode = immediateOffsetOpcode(Opcode);
      AddrMode = ARMII::AddrModeT2_i12;
    }

    // Neon and FP address modes are handled by the base ARM version...
    if ((AddrMode != ARMII::AddrModeT2_i8) &&
        (AddrMode != ARMII::AddrModeT2_i12)) {
      return ARMBaseRegisterInfo::rewriteFrameIndex(MI, FrameRegIdx,
                                                    FrameReg, Offset);
    }
    
    unsigned NumBits = 0;
    Offset += MI.getOperand(FrameRegIdx+1).getImm();

    // i8 supports only negative, and i12 supports only positive, so
    // based on Offset sign convert Opcode to the appropriate
    // instruction
    if (Offset < 0) {
      Opcode = negativeOffsetOpcode(Opcode);
      NumBits = 8;
      isSub = true;
      Offset = -Offset;
    }
    else {
      Opcode = positiveOffsetOpcode(Opcode);
      NumBits = 12;
    }

    if (Opcode) {
      MI.setDesc(TII.get(Opcode));
      MachineOperand &ImmOp = MI.getOperand(FrameRegIdx+1);

      // Attempt to fold address computation
      // Common case: small offset, fits into instruction.
      unsigned Mask = (1 << NumBits) - 1;
      if ((unsigned)Offset <= Mask) {
        // Replace the FrameIndex with fp/sp
        MI.getOperand(FrameRegIdx).ChangeToRegister(FrameReg, false);
        ImmOp.ChangeToImmediate((isSub) ? -Offset : Offset);
        return 0;
      }
      
      // Otherwise, offset doesn't fit. Pull in what we can to simplify
      unsigned ImmedOffset = Offset & Mask;
      ImmOp.ChangeToImmediate((isSub) ? -ImmedOffset : ImmedOffset);
      Offset &= ~Mask;
    }
  }

  return (isSub) ? -Offset : Offset;
}
