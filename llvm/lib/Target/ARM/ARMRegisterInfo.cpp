//===- ARMRegisterInfo.cpp - ARM Register Information -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the "Instituto Nokia de Tecnologia" and
// is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the ARM implementation of the MRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMAddressingModes.h"
#include "ARMInstrInfo.h"
#include "ARMMachineFunctionInfo.h"
#include "ARMRegisterInfo.h"
#include "ARMSubtarget.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Type.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <iostream>
using namespace llvm;

unsigned ARMRegisterInfo::getRegisterNumbering(unsigned RegEnum) {
  using namespace ARM;
  switch (RegEnum) {
  case R0:  case S0:  case D0:  return 0;
  case R1:  case S1:  case D1:  return 1;
  case R2:  case S2:  case D2:  return 2;
  case R3:  case S3:  case D3:  return 3;
  case R4:  case S4:  case D4:  return 4;
  case R5:  case S5:  case D5:  return 5;
  case R6:  case S6:  case D6:  return 6;
  case R7:  case S7:  case D7:  return 7;
  case R8:  case S8:  case D8:  return 8;
  case R9:  case S9:  case D9:  return 9;
  case R10: case S10: case D10: return 10;
  case R11: case S11: case D11: return 11;
  case R12: case S12: case D12: return 12;
  case SP:  case S13: case D13: return 13;
  case LR:  case S14: case D14: return 14;
  case PC:  case S15: case D15: return 15;
  case S16: return 16;
  case S17: return 17;
  case S18: return 18;
  case S19: return 19;
  case S20: return 20;
  case S21: return 21;
  case S22: return 22;
  case S23: return 23;
  case S24: return 24;
  case S25: return 25;
  case S26: return 26;
  case S27: return 27;
  case S28: return 28;
  case S29: return 29;
  case S30: return 30;
  case S31: return 31;
  default:
    std::cerr << "Unknown ARM register!\n";
    abort();
  }
}

ARMRegisterInfo::ARMRegisterInfo(const TargetInstrInfo &tii,
                                 const ARMSubtarget &sti)
  : ARMGenRegisterInfo(ARM::ADJCALLSTACKDOWN, ARM::ADJCALLSTACKUP),
    TII(tii), STI(sti),
    FramePtr(STI.useThumbBacktraces() ? ARM::R7 : ARM::R11) {
}

bool ARMRegisterInfo::spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                                MachineBasicBlock::iterator MI,
                                const std::vector<CalleeSavedInfo> &CSI) const {
  MachineFunction &MF = *MBB.getParent();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  if (!AFI->isThumbFunction() || CSI.empty())
    return false;

  MachineInstrBuilder MIB = BuildMI(MBB, MI, TII.get(ARM::tPUSH));
  for (unsigned i = CSI.size(); i != 0; --i)
    MIB.addReg(CSI[i-1].getReg());
  return true;
}

bool ARMRegisterInfo::restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                                                 MachineBasicBlock::iterator MI,
                                const std::vector<CalleeSavedInfo> &CSI) const {
  MachineFunction &MF = *MBB.getParent();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  if (!AFI->isThumbFunction() || CSI.empty())
    return false;

  MachineInstr *PopMI = new MachineInstr(TII.get(ARM::tPOP));
  MBB.insert(MI, PopMI);
  for (unsigned i = CSI.size(); i != 0; --i) {
    unsigned Reg = CSI[i-1].getReg();
    if (Reg == ARM::LR) {
      Reg = ARM::PC;
      PopMI->setInstrDescriptor(TII.get(ARM::tPOP_RET));
      MBB.erase(MI);
    }
    PopMI->addRegOperand(Reg, true);
  }
  return true;
}

void ARMRegisterInfo::
storeRegToStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                    unsigned SrcReg, int FI,
                    const TargetRegisterClass *RC) const {
  if (RC == ARM::GPRRegisterClass) {
    MachineFunction &MF = *MBB.getParent();
    ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
    if (AFI->isThumbFunction())
      BuildMI(MBB, I, TII.get(ARM::tSTRspi)).addReg(SrcReg)
        .addFrameIndex(FI).addImm(0);
    else
      BuildMI(MBB, I, TII.get(ARM::STR)).addReg(SrcReg)
          .addFrameIndex(FI).addReg(0).addImm(0);
  } else if (RC == ARM::DPRRegisterClass) {
    BuildMI(MBB, I, TII.get(ARM::FSTD)).addReg(SrcReg)
    .addFrameIndex(FI).addImm(0);
  } else {
    assert(RC == ARM::SPRRegisterClass && "Unknown regclass!");
    BuildMI(MBB, I, TII.get(ARM::FSTS)).addReg(SrcReg)
      .addFrameIndex(FI).addImm(0);
  }
}

void ARMRegisterInfo::
loadRegFromStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                     unsigned DestReg, int FI,
                     const TargetRegisterClass *RC) const {
  if (RC == ARM::GPRRegisterClass) {
    MachineFunction &MF = *MBB.getParent();
    ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
    if (AFI->isThumbFunction())
      BuildMI(MBB, I, TII.get(ARM::tLDRspi), DestReg)
        .addFrameIndex(FI).addImm(0);
    else
      BuildMI(MBB, I, TII.get(ARM::LDR), DestReg)
      .addFrameIndex(FI).addReg(0).addImm(0);
  } else if (RC == ARM::DPRRegisterClass) {
    BuildMI(MBB, I, TII.get(ARM::FLDD), DestReg)
      .addFrameIndex(FI).addImm(0);
  } else {
    assert(RC == ARM::SPRRegisterClass && "Unknown regclass!");
    BuildMI(MBB, I, TII.get(ARM::FLDS), DestReg)
      .addFrameIndex(FI).addImm(0);
  }
}

void ARMRegisterInfo::copyRegToReg(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator I,
                                   unsigned DestReg, unsigned SrcReg,
                                   const TargetRegisterClass *RC) const {
  if (RC == ARM::GPRRegisterClass) {
    MachineFunction &MF = *MBB.getParent();
    ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
    BuildMI(MBB, I, TII.get(AFI->isThumbFunction() ? ARM::tMOVrr : ARM::MOVrr),
            DestReg).addReg(SrcReg);
  } else if (RC == ARM::SPRRegisterClass)
    BuildMI(MBB, I, TII.get(ARM::FCPYS), DestReg).addReg(SrcReg);
  else if (RC == ARM::DPRRegisterClass)
    BuildMI(MBB, I, TII.get(ARM::FCPYD), DestReg).addReg(SrcReg);
  else
    abort();
}

MachineInstr *ARMRegisterInfo::foldMemoryOperand(MachineInstr *MI,
                                                 unsigned OpNum, int FI) const {
  unsigned Opc = MI->getOpcode();
  MachineInstr *NewMI = NULL;
  switch (Opc) {
  default: break;
  case ARM::MOVrr: {
    if (OpNum == 0) { // move -> store
      unsigned SrcReg = MI->getOperand(1).getReg();
      NewMI = BuildMI(TII.get(ARM::STR)).addReg(SrcReg).addFrameIndex(FI)
        .addReg(0).addImm(0);
    } else {          // move -> load
      unsigned DstReg = MI->getOperand(0).getReg();
      NewMI = BuildMI(TII.get(ARM::LDR), DstReg).addFrameIndex(FI).addReg(0)
        .addImm(0);
    }
    break;
  }
  case ARM::tMOVrr: {
    if (OpNum == 0) { // move -> store
      unsigned SrcReg = MI->getOperand(1).getReg();
      NewMI = BuildMI(TII.get(ARM::tSTRspi)).addReg(SrcReg).addFrameIndex(FI)
        .addImm(0);
    } else {          // move -> load
      unsigned DstReg = MI->getOperand(0).getReg();
      NewMI = BuildMI(TII.get(ARM::tLDRspi), DstReg).addFrameIndex(FI)
        .addImm(0);
    }
    break;
  }
  case ARM::FCPYS: {
    if (OpNum == 0) { // move -> store
      unsigned SrcReg = MI->getOperand(1).getReg();
      NewMI = BuildMI(TII.get(ARM::FSTS)).addReg(SrcReg).addFrameIndex(FI)
        .addImm(0);
    } else {          // move -> load
      unsigned DstReg = MI->getOperand(0).getReg();
      NewMI = BuildMI(TII.get(ARM::FLDS), DstReg).addFrameIndex(FI).addImm(0);
    }
    break;
  }
  case ARM::FCPYD: {
    if (OpNum == 0) { // move -> store
      unsigned SrcReg = MI->getOperand(1).getReg();
      NewMI = BuildMI(TII.get(ARM::FSTD)).addReg(SrcReg).addFrameIndex(FI)
        .addImm(0);
    } else {          // move -> load
      unsigned DstReg = MI->getOperand(0).getReg();
      NewMI = BuildMI(TII.get(ARM::FLDD), DstReg).addFrameIndex(FI).addImm(0);
    }
    break;
  }
  }

  if (NewMI)
    NewMI->copyKillDeadInfo(MI);
  return NewMI;
}

const unsigned* ARMRegisterInfo::getCalleeSavedRegs() const {
  static const unsigned CalleeSavedRegs[] = {
    ARM::LR, ARM::R11, ARM::R10, ARM::R9, ARM::R8,
    ARM::R7, ARM::R6,  ARM::R5,  ARM::R4,

    ARM::D15, ARM::D14, ARM::D13, ARM::D12,
    ARM::D11, ARM::D10, ARM::D9,  ARM::D8,
    0
  };

  static const unsigned DarwinCalleeSavedRegs[] = {
    ARM::LR,  ARM::R7,  ARM::R6, ARM::R5, ARM::R4,
    ARM::R11, ARM::R10, ARM::R9, ARM::R8,

    ARM::D15, ARM::D14, ARM::D13, ARM::D12,
    ARM::D11, ARM::D10, ARM::D9,  ARM::D8,
    0
  };
  return STI.isTargetDarwin() ? DarwinCalleeSavedRegs : CalleeSavedRegs;
}

const TargetRegisterClass* const *
ARMRegisterInfo::getCalleeSavedRegClasses() const {
  static const TargetRegisterClass * const CalleeSavedRegClasses[] = {
    &ARM::GPRRegClass, &ARM::GPRRegClass, &ARM::GPRRegClass,
    &ARM::GPRRegClass, &ARM::GPRRegClass, &ARM::GPRRegClass,
    &ARM::GPRRegClass, &ARM::GPRRegClass, &ARM::GPRRegClass,

    &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass,
    &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass,
    0
  };
  return CalleeSavedRegClasses;
}

/// hasFP - Return true if the specified function should have a dedicated frame
/// pointer register.  This is true if the function has variable sized allocas
/// or if frame pointer elimination is disabled.
///
static bool hasFP(const MachineFunction &MF) {
  return NoFramePointerElim || MF.getFrameInfo()->hasVarSizedObjects();
}

/// emitARMRegPlusImmediate - Emit a series of instructions to materialize
/// a destreg = basereg + immediate in ARM code.
static
void emitARMRegPlusImmediate(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator &MBBI,
                             unsigned DestReg, unsigned BaseReg,
                             int NumBytes, const TargetInstrInfo &TII) {
  bool isSub = NumBytes < 0;
  if (isSub) NumBytes = -NumBytes;

  while (NumBytes) {
    unsigned RotAmt = ARM_AM::getSOImmValRotate(NumBytes);
    unsigned ThisVal = NumBytes & ARM_AM::rotr32(0xFF, RotAmt);
    assert(ThisVal && "Didn't extract field correctly");
    
    // We will handle these bits from offset, clear them.
    NumBytes &= ~ThisVal;
    
    // Get the properly encoded SOImmVal field.
    int SOImmVal = ARM_AM::getSOImmVal(ThisVal);
    assert(SOImmVal != -1 && "Bit extraction didn't work?");
    
    // Build the new ADD / SUB.
    BuildMI(MBB, MBBI, TII.get(isSub ? ARM::SUBri : ARM::ADDri), DestReg)
      .addReg(BaseReg).addImm(SOImmVal);
    BaseReg = DestReg;
  }
}

/// isLowRegister - Returns true if the register is low register r0-r7.
///
static bool isLowRegister(unsigned Reg) {
  using namespace ARM;
  switch (Reg) {
  case R0:  case R1:  case R2:  case R3:
  case R4:  case R5:  case R6:  case R7:
    return true;
  default:
    return false;
  }
}

/// emitThumbRegPlusImmediate - Emit a series of instructions to materialize
/// a destreg = basereg + immediate in Thumb code.
static
void emitThumbRegPlusImmediate(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator &MBBI,
                               unsigned DestReg, unsigned BaseReg,
                               int NumBytes, const TargetInstrInfo &TII) {
  bool isSub = NumBytes < 0;
  unsigned Bytes = (unsigned)NumBytes;
  if (isSub) Bytes = -NumBytes;
  bool isMul4 = (Bytes & 3) == 0;
  bool isTwoAddr = false;
  unsigned NumBits = 1;
  unsigned Opc = 0;
  unsigned ExtraOpc = 0;

  if (DestReg == BaseReg && BaseReg == ARM::SP) {
    assert(isMul4 && "Thumb sp inc / dec size must be multiple of 4!");
    Bytes >>= 2;  // Implicitly multiplied by 4.
    NumBits = 7;
    Opc = isSub ? ARM::tSUBspi : ARM::tADDspi;
    isTwoAddr = true;
  } else if (!isSub && BaseReg == ARM::SP) {
    if (!isMul4) {
      Bytes &= ~3;
      ExtraOpc = ARM::tADDi3;
    }
    Bytes >>= 2;  // Implicitly multiplied by 4.
    NumBits = 8;
    Opc = ARM::tADDrSPi;
  } else {
    if (DestReg != BaseReg) {
      if (isLowRegister(DestReg) && isLowRegister(BaseReg)) {
        // If both are low registers, emit DestReg = add BaseReg, max(Imm, 7)
        unsigned Chunk = (1 << 3) - 1;
        unsigned ThisVal = (Bytes > Chunk) ? Chunk : Bytes;
        Bytes -= ThisVal;
        BuildMI(MBB, MBBI, TII.get(isSub ? ARM::tSUBi3 : ARM::tADDi3), DestReg)
          .addReg(BaseReg).addImm(ThisVal);
      } else {
        BuildMI(MBB, MBBI, TII.get(ARM::tMOVrr), DestReg).addReg(BaseReg);
      }
      BaseReg = DestReg;
    }
    NumBits = 8;
    Opc = isSub ? ARM::tSUBi8 : ARM::tADDi8;
    isTwoAddr = true;
  }

  unsigned Chunk = (1 << NumBits) - 1;
  while (Bytes) {
    unsigned ThisVal = (Bytes > Chunk) ? Chunk : Bytes;
    Bytes -= ThisVal;    
    // Build the new tADD / tSUB.
    if (isTwoAddr)
      BuildMI(MBB, MBBI, TII.get(Opc), DestReg).addImm(ThisVal);
    else {
      BuildMI(MBB, MBBI, TII.get(Opc), DestReg).addReg(BaseReg).addImm(ThisVal);
      BaseReg = DestReg;

      if (Opc == ARM::tADDrSPi) {
        // r4 = add sp, imm
        // r4 = add r4, imm
        // ...
        NumBits = 8;
        Opc = isSub ? ARM::tSUBi8 : ARM::tADDi8;
        isTwoAddr = true;
      }
    }
  }

  if (ExtraOpc)
    BuildMI(MBB, MBBI, TII.get(ExtraOpc), DestReg).addReg(DestReg)
      .addImm(((unsigned)NumBytes) & 3);
}

static
void emitSPUpdate(MachineBasicBlock &MBB, MachineBasicBlock::iterator &MBBI,
                  int NumBytes, bool isThumb, const TargetInstrInfo &TII) {
  if (isThumb)
    emitThumbRegPlusImmediate(MBB, MBBI, ARM::SP, ARM::SP, NumBytes, TII);
  else
    emitARMRegPlusImmediate(MBB, MBBI, ARM::SP, ARM::SP, NumBytes, TII);
}

void ARMRegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  if (hasFP(MF)) {
    // If we have alloca, convert as follows:
    // ADJCALLSTACKDOWN -> sub, sp, sp, amount
    // ADJCALLSTACKUP   -> add, sp, sp, amount
    MachineInstr *Old = I;
    unsigned Amount = Old->getOperand(0).getImmedValue();
    if (Amount != 0) {
      ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
      // We need to keep the stack aligned properly.  To do this, we round the
      // amount of space needed for the outgoing arguments up to the next
      // alignment boundary.
      unsigned Align = MF.getTarget().getFrameInfo()->getStackAlignment();
      Amount = (Amount+Align-1)/Align*Align;

      // Replace the pseudo instruction with a new instruction...
      if (Old->getOpcode() == ARM::ADJCALLSTACKDOWN) {
        emitSPUpdate(MBB, I, -Amount, AFI->isThumbFunction(), TII);
      } else {
        assert(Old->getOpcode() == ARM::ADJCALLSTACKUP);
        emitSPUpdate(MBB, I, Amount, AFI->isThumbFunction(), TII);
      }
    }
  }
  MBB.erase(I);
}

/// emitThumbConstant - Emit a series of instructions to materialize a
/// constant.
static void emitThumbConstant(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator &MBBI,
                              unsigned DestReg, int Imm,
                              const TargetInstrInfo &TII) {
  bool isSub = Imm < 0;
  if (isSub) Imm = -Imm;

  int Chunk = (1 << 8) - 1;
  int ThisVal = (Imm > Chunk) ? Chunk : Imm;
  Imm -= ThisVal;
  BuildMI(MBB, MBBI, TII.get(ARM::tMOVri8), DestReg).addImm(ThisVal);
  if (Imm > 0) 
    emitThumbRegPlusImmediate(MBB, MBBI, DestReg, DestReg, Imm, TII);
  if (isSub)
    BuildMI(MBB, MBBI, TII.get(ARM::tNEG), DestReg).addReg(DestReg);
}

void ARMRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II) const{
  unsigned i = 0;
  MachineInstr &MI = *II;
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction &MF = *MBB.getParent();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  bool isThumb = AFI->isThumbFunction();

  while (!MI.getOperand(i).isFrameIndex()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }
  
  unsigned FrameReg = ARM::SP;
  int FrameIndex = MI.getOperand(i).getFrameIndex();
  int Offset = MF.getFrameInfo()->getObjectOffset(FrameIndex) + 
               MF.getFrameInfo()->getStackSize();

  if (AFI->isGPRCalleeSavedArea1Frame(FrameIndex))
    Offset -= AFI->getGPRCalleeSavedArea1Offset();
  else if (AFI->isGPRCalleeSavedArea2Frame(FrameIndex))
    Offset -= AFI->getGPRCalleeSavedArea2Offset();
  else if (AFI->isDPRCalleeSavedAreaFrame(FrameIndex))
    Offset -= AFI->getDPRCalleeSavedAreaOffset();
  else if (hasFP(MF)) {
    // There is alloca()'s in this function, must reference off the frame
    // pointer instead.
    FrameReg = getFrameRegister(MF);
    Offset -= AFI->getFramePtrSpillOffset();
  }

  unsigned Opcode = MI.getOpcode();
  const TargetInstrDescriptor &Desc = TII.get(Opcode);
  unsigned AddrMode = (Desc.TSFlags & ARMII::AddrModeMask);
  bool isSub = false;
  
  if (Opcode == ARM::ADDri) {
    Offset += MI.getOperand(i+1).getImm();
    if (Offset == 0) {
      // Turn it into a move.
      MI.setInstrDescriptor(TII.get(ARM::MOVrr));
      MI.getOperand(i).ChangeToRegister(FrameReg, false);
      MI.RemoveOperand(i+1);
      return;
    } else if (Offset < 0) {
      Offset = -Offset;
      isSub = true;
      MI.setInstrDescriptor(TII.get(ARM::SUBri));
    }

    // Common case: small offset, fits into instruction.
    int ImmedOffset = ARM_AM::getSOImmVal(Offset);
    if (ImmedOffset != -1) {
      // Replace the FrameIndex with sp / fp
      MI.getOperand(i).ChangeToRegister(FrameReg, false);
      MI.getOperand(i+1).ChangeToImmediate(ImmedOffset);
      return;
    }
    
    // Otherwise, we fallback to common code below to form the imm offset with
    // a sequence of ADDri instructions.  First though, pull as much of the imm
    // into this ADDri as possible.
    unsigned RotAmt = ARM_AM::getSOImmValRotate(Offset);
    unsigned ThisImmVal = Offset & ARM_AM::rotr32(0xFF, (32-RotAmt) & 31);
    
    // We will handle these bits from offset, clear them.
    Offset &= ~ThisImmVal;
    
    // Get the properly encoded SOImmVal field.
    int ThisSOImmVal = ARM_AM::getSOImmVal(ThisImmVal);
    assert(ThisSOImmVal != -1 && "Bit extraction didn't work?");    
    MI.getOperand(i+1).ChangeToImmediate(ThisSOImmVal);
  } else if (Opcode == ARM::tADDrSPi) {
    Offset += MI.getOperand(i+1).getImm();
    assert((Offset & 3) == 0 &&
           "add/sub sp, #imm immediate must be multiple of 4!");
    Offset >>= 2;
    if (Offset == 0) {
      // Turn it into a move.
      MI.setInstrDescriptor(TII.get(ARM::tMOVrr));
      MI.getOperand(i).ChangeToRegister(FrameReg, false);
      MI.RemoveOperand(i+1);
      return;
    }

    // Common case: small offset, fits into instruction.
    if ((Offset & ~255U) == 0) {
      // Replace the FrameIndex with sp / fp
      MI.getOperand(i).ChangeToRegister(FrameReg, false);
      MI.getOperand(i+1).ChangeToImmediate(Offset);
      return;
    }

    unsigned DestReg = MI.getOperand(0).getReg();
    if (Offset > 0) {
      // Translate r0 = add sp, imm to
      // r0 = add sp, 255*4
      // r0 = add r0, (imm - 255*4)
      MI.getOperand(i).ChangeToRegister(FrameReg, false);
      MI.getOperand(i+1).ChangeToImmediate(255);
      Offset = (Offset - 255) << 2;
      MachineBasicBlock::iterator NII = next(II);
      emitThumbRegPlusImmediate(MBB, NII, DestReg, DestReg, Offset, TII);
    } else {
      // Translate r0 = add sp, -imm to
      // r0 = -imm (this is then translated into a series of instructons)
      // r0 = add r0, sp
      Offset <<= 2;
      emitThumbConstant(MBB, II, DestReg, Offset, TII);
      MI.setInstrDescriptor(TII.get(ARM::tADDhirr));
      MI.getOperand(i).ChangeToRegister(DestReg, false);
      MI.getOperand(i+1).ChangeToRegister(FrameReg, false);
    }
    return;
  } else {
    unsigned ImmIdx = 0;
    int InstrOffs = 0;
    unsigned NumBits = 0;
    unsigned Scale = 1;
    switch (AddrMode) {
    case ARMII::AddrMode2: {
      ImmIdx = i+2;
      InstrOffs = ARM_AM::getAM2Offset(MI.getOperand(ImmIdx).getImm());
      if (ARM_AM::getAM2Op(MI.getOperand(ImmIdx).getImm()) == ARM_AM::sub)
        InstrOffs *= -1;
      NumBits = 12;
      break;
    }
    case ARMII::AddrMode3: {
      ImmIdx = i+2;
      InstrOffs = ARM_AM::getAM3Offset(MI.getOperand(ImmIdx).getImm());
      if (ARM_AM::getAM3Op(MI.getOperand(ImmIdx).getImm()) == ARM_AM::sub)
        InstrOffs *= -1;
      NumBits = 8;
      break;
    }
    case ARMII::AddrMode5: {
      ImmIdx = i+1;
      InstrOffs = ARM_AM::getAM5Offset(MI.getOperand(ImmIdx).getImm());
      if (ARM_AM::getAM5Op(MI.getOperand(ImmIdx).getImm()) == ARM_AM::sub)
        InstrOffs *= -1;
      NumBits = 8;
      Scale = 4;
      break;
    }
    case ARMII::AddrModeTs: {
      ImmIdx = i+1;
      InstrOffs = MI.getOperand(ImmIdx).getImm();
      NumBits = 8;
      Scale = 4;
      break;
    }
    default:
      std::cerr << "Unsupported addressing mode!\n";
      abort();
      break;
    }

    Offset += InstrOffs * Scale;
    assert((Scale == 1 || (Offset & (Scale-1)) == 0) &&
           "Can't encode this offset!");
    if (Offset < 0) {
      Offset = -Offset;
      isSub = true;
    }

    MachineOperand &ImmOp = MI.getOperand(ImmIdx);
    int ImmedOffset = Offset / Scale;
    unsigned Mask = (1 << NumBits) - 1;
    if ((unsigned)Offset <= Mask * Scale) {
      // Replace the FrameIndex with sp
      MI.getOperand(i).ChangeToRegister(FrameReg, false);
      if (isSub)
        ImmedOffset |= 1 << NumBits;
      ImmOp.ChangeToImmediate(ImmedOffset);
      return;
    }

    // Otherwise, it didn't fit.  Pull in what we can to simplify the immediate.
    ImmedOffset = ImmedOffset & Mask;
    if (isSub)
      ImmedOffset |= 1 << NumBits;
    ImmOp.ChangeToImmediate(ImmedOffset);
    Offset &= ~(Mask*Scale);
  }
  
  // If we get here, the immediate doesn't fit into the instruction.  We folded
  // as much as possible above, handle the rest, providing a register that is
  // SP+LargeImm.
  assert(Offset && "This code isn't needed if offset already handled!");

  if (isThumb) {
    if (TII.isLoad(Opcode)) {
      // Use the destination register to materialize sp + offset.
      unsigned TmpReg = MI.getOperand(0).getReg();
      emitThumbRegPlusImmediate(MBB, II, TmpReg, FrameReg,
                                isSub ? -Offset : Offset, TII);
      MI.getOperand(i).ChangeToRegister(TmpReg, false);
    } else if (TII.isStore(Opcode)) {
      // FIXME! This is horrific!!! We need register scavenging.
      // Our temporary workaround has marked r3 unavailable. Of course, r3 is
      // also a ABI register so it's possible that is is the register that is
      // being storing here. If that's the case, we do the following:
      // r12 = r2
      // Use r2 to materialize sp + offset
      // str r12, r2
      // r2 = r12
      unsigned DestReg = MI.getOperand(0).getReg();
      unsigned TmpReg = ARM::R3;
      if (DestReg == ARM::R3) {
        BuildMI(MBB, II, TII.get(ARM::tMOVrr), ARM::R12).addReg(ARM::R2);
        TmpReg = ARM::R2;
      }
      emitThumbRegPlusImmediate(MBB, II, TmpReg, FrameReg,
                                isSub ? -Offset : Offset, TII);
      MI.getOperand(i).ChangeToRegister(DestReg, false);
      if (DestReg == ARM::R3)
        BuildMI(MBB, II, TII.get(ARM::tMOVrr), ARM::R2).addReg(ARM::R12);
    } else
      assert(false && "Unexpected opcode!");
  } else {
    // Insert a set of r12 with the full address: r12 = sp + offset
    // If the offset we have is too large to fit into the instruction, we need
    // to form it with a series of ADDri's.  Do this by taking 8-bit chunks
    // out of 'Offset'.
    emitARMRegPlusImmediate(MBB, II, ARM::R12, FrameReg,
                            isSub ? -Offset : Offset, TII);
    MI.getOperand(i).ChangeToRegister(ARM::R12, false);
  }
}

void ARMRegisterInfo::
processFunctionBeforeCalleeSavedScan(MachineFunction &MF) const {
  // This tells PEI to spill the FP as if it is any other callee-save register
  // to take advantage the eliminateFrameIndex machinery. This also ensures it
  // is spilled in the order specified by getCalleeSavedRegs() to make it easier
  // to combine multiple loads / stores.
  bool CanEliminateFrame = true;
  bool CS1Spilled = false;
  bool LRSpilled = false;
  unsigned NumGPRSpills = 0;
  SmallVector<unsigned, 4> UnspilledCS1GPRs;
  SmallVector<unsigned, 4> UnspilledCS2GPRs;

  // Don't spill FP if the frame can be eliminated. This is determined
  // by scanning the callee-save registers to see if any is used.
  const unsigned *CSRegs = getCalleeSavedRegs();
  const TargetRegisterClass* const *CSRegClasses = getCalleeSavedRegClasses();
  for (unsigned i = 0; CSRegs[i]; ++i) {
    unsigned Reg = CSRegs[i];
    bool Spilled = false;
    if (MF.isPhysRegUsed(Reg)) {
      Spilled = true;
      CanEliminateFrame = false;
    } else {
      // Check alias registers too.
      for (const unsigned *Aliases = getAliasSet(Reg); *Aliases; ++Aliases) {
        if (MF.isPhysRegUsed(*Aliases)) {
          Spilled = true;
          CanEliminateFrame = false;
        }
      }
    }

    if (CSRegClasses[i] == &ARM::GPRRegClass) {
      if (Spilled) {
        NumGPRSpills++;

        // Keep track if LR and any of R4, R5, R6, and R7 is spilled.
        switch (Reg) {
        case ARM::LR:
          LRSpilled = true;
          // Fallthrough
        case ARM::R4:
        case ARM::R5:
        case ARM::R6:
        case ARM::R7:
          CS1Spilled = true;
          break;
        default:
          break;
        }
      } else { 
        switch (Reg) {
        case ARM::R4:
        case ARM::R5:
        case ARM::R6:
        case ARM::R7:
        case ARM::LR:
          UnspilledCS1GPRs.push_back(Reg);
          break;
        default:
          UnspilledCS2GPRs.push_back(Reg);
          break;
        }
      }
    }
  }

  if (!CanEliminateFrame) {
    ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
    AFI->setHasStackFrame(true);

    // If LR is not spilled, but at least one of R4, R5, R6, and R7 is spilled.
    // Spill LR as well so we can fold BX_RET to the registers restore (LDM).
    if (!LRSpilled && CS1Spilled) {
      MF.changePhyRegUsed(ARM::LR, true);
      NumGPRSpills++;
      UnspilledCS1GPRs.erase(std::find(UnspilledCS1GPRs.begin(),
                                    UnspilledCS1GPRs.end(), (unsigned)ARM::LR));
    }

    // If stack and double are 8-byte aligned and we are spilling a odd number
    // of GPRs. Spill one extra callee save GPR so we won't have to pad between
    // the integer and double callee save areas.
    unsigned TargetAlign = MF.getTarget().getFrameInfo()->getStackAlignment();
    if (TargetAlign == 8 && (NumGPRSpills & 1)) {
      if (CS1Spilled && !UnspilledCS1GPRs.empty())
        MF.changePhyRegUsed(UnspilledCS1GPRs.front(), true);
      else
        MF.changePhyRegUsed(UnspilledCS2GPRs.front(), true);
    }
    MF.changePhyRegUsed(FramePtr, true);
  }
}

/// Move iterator pass the next bunch of callee save load / store ops for
/// the particular spill area (1: integer area 1, 2: integer area 2,
/// 3: fp area, 0: don't care).
static void movePastCSLoadStoreOps(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator &MBBI,
                                   int Opc, unsigned Area,
                                   const ARMSubtarget &STI) {
  while (MBBI != MBB.end() &&
         MBBI->getOpcode() == Opc && MBBI->getOperand(1).isFrameIndex()) {
    if (Area != 0) {
      bool Done = false;
      unsigned Category = 0;
      switch (MBBI->getOperand(0).getReg()) {
      case ARM::R4:  case ARM::R5:  case ARM::R6: case ARM::R7:
      case ARM::LR:
        Category = 1;
        break;
      case ARM::R8:  case ARM::R9:  case ARM::R10: case ARM::R11:
        Category = STI.isTargetDarwin() ? 2 : 1;
        break;
      case ARM::D8:  case ARM::D9:  case ARM::D10: case ARM::D11:
      case ARM::D12: case ARM::D13: case ARM::D14: case ARM::D15:
        Category = 3;
        break;
      default:
        Done = true;
        break;
      }
      if (Done || Category != Area)
        break;
    }

    ++MBBI;
  }
}

void ARMRegisterInfo::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();
  MachineBasicBlock::iterator MBBI = MBB.begin();
  MachineFrameInfo  *MFI = MF.getFrameInfo();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  bool isThumb = AFI->isThumbFunction();
  unsigned VARegSaveSize = AFI->getVarArgsRegSaveSize();
  unsigned Align = MF.getTarget().getFrameInfo()->getStackAlignment();
  unsigned NumBytes = MFI->getStackSize();
  const std::vector<CalleeSavedInfo> &CSI = MFI->getCalleeSavedInfo();

  // Determine the sizes of each callee-save spill areas and record which frame
  // belongs to which callee-save spill areas.
  unsigned GPRCS1Size = 0, GPRCS2Size = 0, DPRCSSize = 0;
  int FramePtrSpillFI = 0;
  if (AFI->hasStackFrame()) {
    if (VARegSaveSize)
      emitSPUpdate(MBB, MBBI, -VARegSaveSize, isThumb, TII);

    for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
      unsigned Reg = CSI[i].getReg();
      int FI = CSI[i].getFrameIdx();
      switch (Reg) {
      case ARM::R4:
      case ARM::R5:
      case ARM::R6:
      case ARM::R7:
      case ARM::LR:
        if (Reg == FramePtr)
          FramePtrSpillFI = FI;
        AFI->addGPRCalleeSavedArea1Frame(FI);
        GPRCS1Size += 4;
        break;
      case ARM::R8:
      case ARM::R9:
      case ARM::R10:
      case ARM::R11:
        if (Reg == FramePtr)
          FramePtrSpillFI = FI;
        if (STI.isTargetDarwin()) {
          AFI->addGPRCalleeSavedArea2Frame(FI);
          GPRCS2Size += 4;
        } else {
          AFI->addGPRCalleeSavedArea1Frame(FI);
          GPRCS1Size += 4;
        }
        break;
      default:
        AFI->addDPRCalleeSavedAreaFrame(FI);
        DPRCSSize += 8;
      }
    }

    if (!isThumb) {
      // Build the new SUBri to adjust SP for integer callee-save spill area 1.
      emitSPUpdate(MBB, MBBI, -GPRCS1Size, isThumb, TII);
      movePastCSLoadStoreOps(MBB, MBBI, ARM::STR, 1, STI);
    } else {
      if (MBBI != MBB.end() && MBBI->getOpcode() == ARM::tPUSH)
        ++MBBI;
    }

    // Point FP to the stack slot that contains the previous FP.
    BuildMI(MBB, MBBI, TII.get(isThumb ? ARM::tADDrSPi : ARM::ADDri), FramePtr)
      .addFrameIndex(FramePtrSpillFI).addImm(0);

    if (!isThumb) {
      // Build the new SUBri to adjust SP for integer callee-save spill area 2.
      emitSPUpdate(MBB, MBBI, -GPRCS2Size, false, TII);

      // Build the new SUBri to adjust SP for FP callee-save spill area.
      movePastCSLoadStoreOps(MBB, MBBI, ARM::STR, 2, STI);
      emitSPUpdate(MBB, MBBI, -DPRCSSize, false, TII);
    }
  }

  // If necessary, add one more SUBri to account for the call frame
  // and/or local storage, alloca area.
  if (MFI->hasCalls() && !hasFP(MF))
    // We reserve argument space for call sites in the function immediately on
    // entry to the current function.  This eliminates the need for add/sub
    // brackets around call sites.
    NumBytes += MFI->getMaxCallFrameSize();

  // Round the size to a multiple of the alignment.
  NumBytes = (NumBytes+Align-1)/Align*Align;
  MFI->setStackSize(NumBytes);

  // Determine starting offsets of spill areas.
  if (AFI->hasStackFrame()) {
    unsigned DPRCSOffset  = NumBytes - (GPRCS1Size + GPRCS2Size + DPRCSSize);
    unsigned GPRCS2Offset = DPRCSOffset + DPRCSSize;
    unsigned GPRCS1Offset = GPRCS2Offset + GPRCS2Size;
    AFI->setFramePtrSpillOffset(MFI->getObjectOffset(FramePtrSpillFI) + NumBytes);
    AFI->setGPRCalleeSavedArea1Offset(GPRCS1Offset);
    AFI->setGPRCalleeSavedArea2Offset(GPRCS2Offset);
    AFI->setDPRCalleeSavedAreaOffset(DPRCSOffset);
  
    NumBytes = DPRCSOffset;
    if (NumBytes) {
      // Insert it after all the callee-save spills.
      if (!isThumb)
        movePastCSLoadStoreOps(MBB, MBBI, ARM::FSTD, 3, STI);
      emitSPUpdate(MBB, MBBI, -NumBytes, isThumb, TII);
    }
  } else 
    emitSPUpdate(MBB, MBBI, -NumBytes, isThumb, TII);

  AFI->setGPRCalleeSavedArea1Size(GPRCS1Size);
  AFI->setGPRCalleeSavedArea2Size(GPRCS2Size);
  AFI->setDPRCalleeSavedAreaSize(DPRCSSize);
}

static bool isCalleeSavedRegister(unsigned Reg, const unsigned *CSRegs) {
  for (unsigned i = 0; CSRegs[i]; ++i)
    if (Reg == CSRegs[i])
      return true;
  return false;
}

static bool isCSRestore(MachineInstr *MI, const unsigned *CSRegs) {
  return ((MI->getOpcode() == ARM::FLDD ||
           MI->getOpcode() == ARM::LDR  ||
           MI->getOpcode() == ARM::tLDRspi) &&
          MI->getOperand(1).isFrameIndex() &&
          isCalleeSavedRegister(MI->getOperand(0).getReg(), CSRegs));
}

void ARMRegisterInfo::emitEpilogue(MachineFunction &MF,
				   MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  assert((MBBI->getOpcode() == ARM::BX_RET ||
          MBBI->getOpcode() == ARM::tBX_RET ||
          MBBI->getOpcode() == ARM::tPOP_RET) &&
         "Can only insert epilog into returning blocks");

  MachineFrameInfo *MFI = MF.getFrameInfo();
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();
  bool isThumb = AFI->isThumbFunction();
  unsigned VARegSaveSize = AFI->getVarArgsRegSaveSize();
  int NumBytes = (int)MFI->getStackSize();
  if (AFI->hasStackFrame()) {
    // Unwind MBBI to point to first LDR / FLDD.
    const unsigned *CSRegs = getCalleeSavedRegs();
    if (MBBI != MBB.begin()) {
      do
        --MBBI;
      while (MBBI != MBB.begin() && isCSRestore(MBBI, CSRegs));
      if (!isCSRestore(MBBI, CSRegs))
        ++MBBI;
    }

    // Move SP to start of FP callee save spill area.
    NumBytes -= (AFI->getGPRCalleeSavedArea1Size() +
                 AFI->getGPRCalleeSavedArea2Size() +
                 AFI->getDPRCalleeSavedAreaSize());
    if (isThumb)
      emitSPUpdate(MBB, MBBI, -NumBytes, isThumb, TII);
    else {
      NumBytes = AFI->getFramePtrSpillOffset() - NumBytes;
      // Reset SP based on frame pointer only if the stack frame extends beyond
      // frame pointer stack slot.
      if (AFI->getGPRCalleeSavedArea2Size() ||
          AFI->getDPRCalleeSavedAreaSize()  ||
          AFI->getDPRCalleeSavedAreaOffset())
        if (NumBytes)
          BuildMI(MBB, MBBI, TII.get(ARM::SUBri), ARM::SP).addReg(FramePtr)
            .addImm(NumBytes);
        else
          BuildMI(MBB, MBBI, TII.get(isThumb ? ARM::tMOVrr : ARM::MOVrr),
                  ARM::SP).addReg(FramePtr);

      // Move SP to start of integer callee save spill area 2.
      movePastCSLoadStoreOps(MBB, MBBI, ARM::FLDD, 3, STI);
      emitSPUpdate(MBB, MBBI, AFI->getDPRCalleeSavedAreaSize(), false, TII);

      // Move SP to start of integer callee save spill area 1.
      movePastCSLoadStoreOps(MBB, MBBI, ARM::LDR, 2, STI);
      emitSPUpdate(MBB, MBBI, AFI->getGPRCalleeSavedArea2Size(), false, TII);

      // Move SP to SP upon entry to the function.
      movePastCSLoadStoreOps(MBB, MBBI, ARM::LDR, 1, STI);
      emitSPUpdate(MBB, MBBI, AFI->getGPRCalleeSavedArea1Size(), false, TII);
    }

    if (VARegSaveSize)
      emitSPUpdate(MBB, MBBI, VARegSaveSize, isThumb, TII);
  } else if (NumBytes != 0) {
    emitSPUpdate(MBB, MBBI, NumBytes, isThumb, TII);
  }
}

unsigned ARMRegisterInfo::getRARegister() const {
  return ARM::LR;
}

unsigned ARMRegisterInfo::getFrameRegister(MachineFunction &MF) const {
  return STI.useThumbBacktraces() ? ARM::R7 : ARM::R11;
}

#include "ARMGenRegisterInfo.inc"

