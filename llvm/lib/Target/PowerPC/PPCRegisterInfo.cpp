//===- PPCRegisterInfo.cpp - PowerPC Register Information -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PowerPC implementation of the MRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "reginfo"
#include "PPC.h"
#include "PPCInstrBuilder.h"
#include "PPCRegisterInfo.h"
#include "llvm/Constants.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineDebugInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdlib>
#include <iostream>
using namespace llvm;

/// getRegisterNumbering - Given the enum value for some register, e.g.
/// PPC::F14, return the number that it corresponds to (e.g. 14).
unsigned PPCRegisterInfo::getRegisterNumbering(unsigned RegEnum) {
  switch (RegEnum) {
    case PPC::R0 :  case PPC::F0 :  case PPC::V0 : case PPC::CR0:  return  0;
    case PPC::R1 :  case PPC::F1 :  case PPC::V1 : case PPC::CR1:  return  1;
    case PPC::R2 :  case PPC::F2 :  case PPC::V2 : case PPC::CR2:  return  2;
    case PPC::R3 :  case PPC::F3 :  case PPC::V3 : case PPC::CR3:  return  3;
    case PPC::R4 :  case PPC::F4 :  case PPC::V4 : case PPC::CR4:  return  4;
    case PPC::R5 :  case PPC::F5 :  case PPC::V5 : case PPC::CR5:  return  5;
    case PPC::R6 :  case PPC::F6 :  case PPC::V6 : case PPC::CR6:  return  6;
    case PPC::R7 :  case PPC::F7 :  case PPC::V7 : case PPC::CR7:  return  7;
    case PPC::R8 :  case PPC::F8 :  case PPC::V8 : return  8;
    case PPC::R9 :  case PPC::F9 :  case PPC::V9 : return  9;
    case PPC::R10:  case PPC::F10:  case PPC::V10: return 10;
    case PPC::R11:  case PPC::F11:  case PPC::V11: return 11;
    case PPC::R12:  case PPC::F12:  case PPC::V12: return 12;
    case PPC::R13:  case PPC::F13:  case PPC::V13: return 13;
    case PPC::R14:  case PPC::F14:  case PPC::V14: return 14;
    case PPC::R15:  case PPC::F15:  case PPC::V15: return 15;
    case PPC::R16:  case PPC::F16:  case PPC::V16: return 16;
    case PPC::R17:  case PPC::F17:  case PPC::V17: return 17;
    case PPC::R18:  case PPC::F18:  case PPC::V18: return 18;
    case PPC::R19:  case PPC::F19:  case PPC::V19: return 19;
    case PPC::R20:  case PPC::F20:  case PPC::V20: return 20;
    case PPC::R21:  case PPC::F21:  case PPC::V21: return 21;
    case PPC::R22:  case PPC::F22:  case PPC::V22: return 22;
    case PPC::R23:  case PPC::F23:  case PPC::V23: return 23;
    case PPC::R24:  case PPC::F24:  case PPC::V24: return 24;
    case PPC::R25:  case PPC::F25:  case PPC::V25: return 25;
    case PPC::R26:  case PPC::F26:  case PPC::V26: return 26;
    case PPC::R27:  case PPC::F27:  case PPC::V27: return 27;
    case PPC::R28:  case PPC::F28:  case PPC::V28: return 28;
    case PPC::R29:  case PPC::F29:  case PPC::V29: return 29;
    case PPC::R30:  case PPC::F30:  case PPC::V30: return 30;
    case PPC::R31:  case PPC::F31:  case PPC::V31: return 31;
    default:
      std::cerr << "Unhandled reg in PPCRegisterInfo::getRegisterNumbering!\n";
      abort();
  }
}

PPCRegisterInfo::PPCRegisterInfo()
  : PPCGenRegisterInfo(PPC::ADJCALLSTACKDOWN, PPC::ADJCALLSTACKUP) {
  ImmToIdxMap[PPC::LD]   = PPC::LDX;    ImmToIdxMap[PPC::STD]  = PPC::STDX;
  ImmToIdxMap[PPC::LBZ]  = PPC::LBZX;   ImmToIdxMap[PPC::STB]  = PPC::STBX;
  ImmToIdxMap[PPC::LHZ]  = PPC::LHZX;   ImmToIdxMap[PPC::LHA]  = PPC::LHAX;
  ImmToIdxMap[PPC::LWZ]  = PPC::LWZX;   ImmToIdxMap[PPC::LWA]  = PPC::LWAX;
  ImmToIdxMap[PPC::LFS]  = PPC::LFSX;   ImmToIdxMap[PPC::LFD]  = PPC::LFDX;
  ImmToIdxMap[PPC::STH]  = PPC::STHX;   ImmToIdxMap[PPC::STW]  = PPC::STWX;
  ImmToIdxMap[PPC::STFS] = PPC::STFSX;  ImmToIdxMap[PPC::STFD] = PPC::STFDX;
  ImmToIdxMap[PPC::ADDI] = PPC::ADD4;
}

void
PPCRegisterInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator MI,
                                     unsigned SrcReg, int FrameIdx,
                                     const TargetRegisterClass *RC) const {
  if (SrcReg == PPC::LR) {
    // FIXME: this spills LR immediately to memory in one step.  To do this, we
    // use R11, which we know cannot be used in the prolog/epilog.  This is a
    // hack.
    BuildMI(MBB, MI, PPC::MFLR, 1, PPC::R11);
    addFrameReference(BuildMI(MBB, MI, PPC::STW, 3).addReg(PPC::R11), FrameIdx);
  } else if (RC == PPC::CRRCRegisterClass) {
    // FIXME: We use R0 here, because it isn't available for RA.
    // We need to store the CR in the low 4-bits of the saved value.  First,
    // issue a MFCR to save all of the CRBits.
    BuildMI(MBB, MI, PPC::MFCR, 0, PPC::R0);
    
    // If the saved register wasn't CR0, shift the bits left so that they are in
    // CR0's slot.
    if (SrcReg != PPC::CR0) {
      unsigned ShiftBits = PPCRegisterInfo::getRegisterNumbering(SrcReg)*4;
      // rlwinm r0, r0, ShiftBits, 0, 31.
      BuildMI(MBB, MI, PPC::RLWINM, 4, PPC::R0)
        .addReg(PPC::R0).addImm(ShiftBits).addImm(0).addImm(31);
    }
    
    addFrameReference(BuildMI(MBB, MI, PPC::STW, 3).addReg(PPC::R0), FrameIdx);
  } else if (RC == PPC::GPRCRegisterClass) {
    addFrameReference(BuildMI(MBB, MI, PPC::STW, 3).addReg(SrcReg),FrameIdx);
  } else if (RC == PPC::G8RCRegisterClass) {
    addFrameReference(BuildMI(MBB, MI, PPC::STD, 3).addReg(SrcReg),FrameIdx);
  } else if (RC == PPC::F8RCRegisterClass) {
    addFrameReference(BuildMI(MBB, MI, PPC::STFD, 3).addReg(SrcReg),FrameIdx);
  } else if (RC == PPC::F4RCRegisterClass) {
    addFrameReference(BuildMI(MBB, MI, PPC::STFS, 3).addReg(SrcReg),FrameIdx);
  } else if (RC == PPC::VRRCRegisterClass) {
    // We don't have indexed addressing for vector loads.  Emit:
    // R11 = ADDI FI#
    // Dest = LVX R0, R11
    // 
    // FIXME: We use R0 here, because it isn't available for RA.
    addFrameReference(BuildMI(MBB, MI, PPC::ADDI, 1, PPC::R0), FrameIdx, 0, 0);
    BuildMI(MBB, MI, PPC::STVX, 3)
      .addReg(SrcReg).addReg(PPC::R0).addReg(PPC::R0);
  } else {
    assert(0 && "Unknown regclass!");
    abort();
  }
}

void
PPCRegisterInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MI,
                                        unsigned DestReg, int FrameIdx,
                                        const TargetRegisterClass *RC) const {
  if (DestReg == PPC::LR) {
    addFrameReference(BuildMI(MBB, MI, PPC::LWZ, 2, PPC::R11), FrameIdx);
    BuildMI(MBB, MI, PPC::MTLR, 1).addReg(PPC::R11);
  } else if (RC == PPC::CRRCRegisterClass) {
    // FIXME: We use R0 here, because it isn't available for RA.
    addFrameReference(BuildMI(MBB, MI, PPC::LWZ, 2, PPC::R0), FrameIdx);
    
    // If the reloaded register isn't CR0, shift the bits right so that they are
    // in the right CR's slot.
    if (DestReg != PPC::CR0) {
      unsigned ShiftBits = PPCRegisterInfo::getRegisterNumbering(DestReg)*4;
      // rlwinm r11, r11, 32-ShiftBits, 0, 31.
      BuildMI(MBB, MI, PPC::RLWINM, 4, PPC::R0)
        .addReg(PPC::R0).addImm(32-ShiftBits).addImm(0).addImm(31);
    }
    
    BuildMI(MBB, MI, PPC::MTCRF, 1, DestReg).addReg(PPC::R0);
  } else if (RC == PPC::GPRCRegisterClass) {
    addFrameReference(BuildMI(MBB, MI, PPC::LWZ, 2, DestReg), FrameIdx);
  } else if (RC == PPC::G8RCRegisterClass) {
    addFrameReference(BuildMI(MBB, MI, PPC::LD, 2, DestReg), FrameIdx);
  } else if (RC == PPC::F8RCRegisterClass) {
    addFrameReference(BuildMI(MBB, MI, PPC::LFD, 2, DestReg), FrameIdx);
  } else if (RC == PPC::F4RCRegisterClass) {
    addFrameReference(BuildMI(MBB, MI, PPC::LFS, 2, DestReg), FrameIdx);
  } else if (RC == PPC::VRRCRegisterClass) {
    // We don't have indexed addressing for vector loads.  Emit:
    // R11 = ADDI FI#
    // Dest = LVX R0, R11
    // 
    // FIXME: We use R0 here, because it isn't available for RA.
    addFrameReference(BuildMI(MBB, MI, PPC::ADDI, 1, PPC::R0), FrameIdx, 0, 0);
    BuildMI(MBB, MI, PPC::LVX, 2, DestReg).addReg(PPC::R0).addReg(PPC::R0);
  } else {
    assert(0 && "Unknown regclass!");
    abort();
  }
}

void PPCRegisterInfo::copyRegToReg(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MI,
                                   unsigned DestReg, unsigned SrcReg,
                                   const TargetRegisterClass *RC) const {
  if (RC == PPC::GPRCRegisterClass) {
    BuildMI(MBB, MI, PPC::OR, 2, DestReg).addReg(SrcReg).addReg(SrcReg);
  } else if (RC == PPC::G8RCRegisterClass) {
    BuildMI(MBB, MI, PPC::OR8, 2, DestReg).addReg(SrcReg).addReg(SrcReg);
  } else if (RC == PPC::F4RCRegisterClass) {
    BuildMI(MBB, MI, PPC::FMRS, 1, DestReg).addReg(SrcReg);
  } else if (RC == PPC::F8RCRegisterClass) {
    BuildMI(MBB, MI, PPC::FMRD, 1, DestReg).addReg(SrcReg);
  } else if (RC == PPC::CRRCRegisterClass) {
    BuildMI(MBB, MI, PPC::MCRF, 1, DestReg).addReg(SrcReg);
  } else if (RC == PPC::VRRCRegisterClass) {
    BuildMI(MBB, MI, PPC::VOR, 2, DestReg).addReg(SrcReg).addReg(SrcReg);
  } else {
    std::cerr << "Attempt to copy register that is not GPR or FPR";
    abort();
  }
}

const unsigned* PPCRegisterInfo::getCalleeSaveRegs() const {
  static const unsigned CalleeSaveRegs[] = {
    PPC::R1, PPC::R13,
    PPC::R14, PPC::R15,
    PPC::R16, PPC::R17,
    PPC::R18, PPC::R19,
    PPC::R20, PPC::R21,
    PPC::R22, PPC::R23,
    PPC::R24, PPC::R25,
    PPC::R26, PPC::R27,
    PPC::R28, PPC::R29,
    PPC::R30, PPC::R31,
    PPC::F14, PPC::F15,
    PPC::F16, PPC::F17,
    PPC::F18, PPC::F19,
    PPC::F20, PPC::F21,
    PPC::F22, PPC::F23,
    PPC::F24, PPC::F25,
    PPC::F26, PPC::F27,
    PPC::F28, PPC::F29,
    PPC::F30, PPC::F31,
    PPC::CR2, PPC::CR3,
    PPC::CR4, PPC::V20,
    PPC::V21, PPC::V22,
    PPC::V23, PPC::V24,
    PPC::V25, PPC::V26,
    PPC::V27, PPC::V28,
    PPC::V29, PPC::V30,
    PPC::V31, PPC::LR,  0
  };
  return CalleeSaveRegs;
}

const TargetRegisterClass* const*
PPCRegisterInfo::getCalleeSaveRegClasses() const {
  static const TargetRegisterClass * const CalleeSaveRegClasses[] = {
    &PPC::GPRCRegClass, &PPC::GPRCRegClass,
    &PPC::GPRCRegClass, &PPC::GPRCRegClass,
    &PPC::GPRCRegClass, &PPC::GPRCRegClass,
    &PPC::GPRCRegClass, &PPC::GPRCRegClass,
    &PPC::GPRCRegClass, &PPC::GPRCRegClass,
    &PPC::GPRCRegClass, &PPC::GPRCRegClass,
    &PPC::GPRCRegClass, &PPC::GPRCRegClass,
    &PPC::GPRCRegClass, &PPC::GPRCRegClass,
    &PPC::GPRCRegClass, &PPC::GPRCRegClass,
    &PPC::GPRCRegClass, &PPC::GPRCRegClass,
    &PPC::F8RCRegClass, &PPC::F8RCRegClass,
    &PPC::F8RCRegClass, &PPC::F8RCRegClass,
    &PPC::F8RCRegClass, &PPC::F8RCRegClass,
    &PPC::F8RCRegClass, &PPC::F8RCRegClass,
    &PPC::F8RCRegClass, &PPC::F8RCRegClass,
    &PPC::F8RCRegClass, &PPC::F8RCRegClass,
    &PPC::F8RCRegClass, &PPC::F8RCRegClass,
    &PPC::F8RCRegClass, &PPC::F8RCRegClass,
    &PPC::F8RCRegClass, &PPC::F8RCRegClass,
    &PPC::CRRCRegClass, &PPC::CRRCRegClass,
    &PPC::CRRCRegClass, &PPC::VRRCRegClass,
    &PPC::VRRCRegClass, &PPC::VRRCRegClass,
    &PPC::VRRCRegClass, &PPC::VRRCRegClass,
    &PPC::VRRCRegClass, &PPC::VRRCRegClass,
    &PPC::VRRCRegClass, &PPC::VRRCRegClass,
    &PPC::VRRCRegClass, &PPC::VRRCRegClass,
    &PPC::VRRCRegClass, &PPC::GPRCRegClass,  0
  };
  return CalleeSaveRegClasses;
}

/// foldMemoryOperand - PowerPC (like most RISC's) can only fold spills into
/// copy instructions, turning them into load/store instructions.
MachineInstr *PPCRegisterInfo::foldMemoryOperand(MachineInstr *MI,
                                                 unsigned OpNum,
                                                 int FrameIndex) const {
  // Make sure this is a reg-reg copy.  Note that we can't handle MCRF, because
  // it takes more than one instruction to store it.
  unsigned Opc = MI->getOpcode();
  
  if ((Opc == PPC::OR &&
       MI->getOperand(1).getReg() == MI->getOperand(2).getReg())) {
    if (OpNum == 0) {  // move -> store
      unsigned InReg = MI->getOperand(1).getReg();
      return addFrameReference(BuildMI(PPC::STW,
                                       3).addReg(InReg), FrameIndex);
    } else {           // move -> load
      unsigned OutReg = MI->getOperand(0).getReg();
      return addFrameReference(BuildMI(PPC::LWZ, 2, OutReg), FrameIndex);
    }
  } else if ((Opc == PPC::OR8 &&
              MI->getOperand(1).getReg() == MI->getOperand(2).getReg())) {
    if (OpNum == 0) {  // move -> store
      unsigned InReg = MI->getOperand(1).getReg();
      return addFrameReference(BuildMI(PPC::STD,
                                       3).addReg(InReg), FrameIndex);
    } else {           // move -> load
      unsigned OutReg = MI->getOperand(0).getReg();
      return addFrameReference(BuildMI(PPC::LD, 2, OutReg), FrameIndex);
    }
  } else if (Opc == PPC::FMRD) {
    if (OpNum == 0) {  // move -> store
      unsigned InReg = MI->getOperand(1).getReg();
      return addFrameReference(BuildMI(PPC::STFD,
                                       3).addReg(InReg), FrameIndex);
    } else {           // move -> load
      unsigned OutReg = MI->getOperand(0).getReg();
      return addFrameReference(BuildMI(PPC::LFD, 2, OutReg), FrameIndex);
    }
  } else if (Opc == PPC::FMRS) {
    if (OpNum == 0) {  // move -> store
      unsigned InReg = MI->getOperand(1).getReg();
      return addFrameReference(BuildMI(PPC::STFS,
                                       3).addReg(InReg), FrameIndex);
    } else {           // move -> load
      unsigned OutReg = MI->getOperand(0).getReg();
      return addFrameReference(BuildMI(PPC::LFS, 2, OutReg), FrameIndex);
    }
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// Stack Frame Processing methods
//===----------------------------------------------------------------------===//

// hasFP - Return true if the specified function should have a dedicated frame
// pointer register.  This is true if the function has variable sized allocas or
// if frame pointer elimination is disabled.
//
static bool hasFP(const MachineFunction &MF) {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  unsigned TargetAlign = MF.getTarget().getFrameInfo()->getStackAlignment();

  // If frame pointers are forced, or if there are variable sized stack objects,
  // use a frame pointer.
  // 
  return NoFramePointerElim || MFI->hasVarSizedObjects();
}

void PPCRegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  if (hasFP(MF)) {
    // If we have a frame pointer, convert as follows:
    // ADJCALLSTACKDOWN -> addi, r1, r1, -amount
    // ADJCALLSTACKUP   -> addi, r1, r1, amount
    MachineInstr *Old = I;
    unsigned Amount = Old->getOperand(0).getImmedValue();
    if (Amount != 0) {
      // We need to keep the stack aligned properly.  To do this, we round the
      // amount of space needed for the outgoing arguments up to the next
      // alignment boundary.
      unsigned Align = MF.getTarget().getFrameInfo()->getStackAlignment();
      Amount = (Amount+Align-1)/Align*Align;

      // Replace the pseudo instruction with a new instruction...
      if (Old->getOpcode() == PPC::ADJCALLSTACKDOWN) {
        BuildMI(MBB, I, PPC::ADDI, 2, PPC::R1).addReg(PPC::R1).addImm(-Amount);
      } else {
        assert(Old->getOpcode() == PPC::ADJCALLSTACKUP);
        BuildMI(MBB, I, PPC::ADDI, 2, PPC::R1).addReg(PPC::R1).addImm(Amount);
      }
    }
  }
  MBB.erase(I);
}

void
PPCRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II) const {
  unsigned i = 0;
  MachineInstr &MI = *II;
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction &MF = *MBB.getParent();

  while (!MI.getOperand(i).isFrameIndex()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }

  int FrameIndex = MI.getOperand(i).getFrameIndex();

  // Replace the FrameIndex with base register with GPR1 (SP) or GPR31 (FP).
  MI.getOperand(i).ChangeToRegister(hasFP(MF) ? PPC::R31 : PPC::R1);

  // Take into account whether it's an add or mem instruction
  unsigned OffIdx = (i == 2) ? 1 : 2;

  // Now add the frame object offset to the offset from r1.
  int Offset = MF.getFrameInfo()->getObjectOffset(FrameIndex) +
               MI.getOperand(OffIdx).getImmedValue();

  // If we're not using a Frame Pointer that has been set to the value of the
  // SP before having the stack size subtracted from it, then add the stack size
  // to Offset to get the correct offset.
  Offset += MF.getFrameInfo()->getStackSize();

  if (Offset > 32767 || Offset < -32768) {
    // Insert a set of r0 with the full offset value before the ld, st, or add
    MachineBasicBlock *MBB = MI.getParent();
    BuildMI(*MBB, II, PPC::LIS, 1, PPC::R0).addImm(Offset >> 16);
    BuildMI(*MBB, II, PPC::ORI, 2, PPC::R0).addReg(PPC::R0).addImm(Offset);
    
    // convert into indexed form of the instruction
    // sth 0:rA, 1:imm 2:(rB) ==> sthx 0:rA, 2:rB, 1:r0
    // addi 0:rA 1:rB, 2, imm ==> add 0:rA, 1:rB, 2:r0
    assert(ImmToIdxMap.count(MI.getOpcode()) &&
           "No indexed form of load or store available!");
    unsigned NewOpcode = ImmToIdxMap.find(MI.getOpcode())->second;
    MI.setOpcode(NewOpcode);
    MI.getOperand(1).ChangeToRegister(MI.getOperand(i).getReg());
    MI.getOperand(2).ChangeToRegister(PPC::R0);
  } else {
    switch (MI.getOpcode()) {
    case PPC::LWA:
    case PPC::LD:
    case PPC::STD:
    case PPC::STD_32:
      assert((Offset & 3) == 0 && "Invalid frame offset!");
      Offset >>= 2;    // The actual encoded value has the low two bits zero.
      break;
    }
    MI.getOperand(OffIdx).ChangeToImmediate(Offset);
  }
}

/// VRRegNo - Map from a numbered VR register to its enum value.
///
static const unsigned short VRRegNo[] = {
 PPC::V0 , PPC::V1 , PPC::V2 , PPC::V3 , PPC::V4 , PPC::V5 , PPC::V6 , PPC::V7 ,
 PPC::V8 , PPC::V9 , PPC::V10, PPC::V11, PPC::V12, PPC::V13, PPC::V14, PPC::V15,
 PPC::V16, PPC::V17, PPC::V18, PPC::V19, PPC::V20, PPC::V21, PPC::V22, PPC::V23,
 PPC::V24, PPC::V25, PPC::V26, PPC::V27, PPC::V28, PPC::V29, PPC::V30, PPC::V31
};

/// RemoveVRSaveCode - We have found that this function does not need any code
/// to manipulate the VRSAVE register, even though it uses vector registers.
/// This can happen when the only registers used are known to be live in or out
/// of the function.  Remove all of the VRSAVE related code from the function.
static void RemoveVRSaveCode(MachineInstr *MI) {
  MachineBasicBlock *Entry = MI->getParent();
  MachineFunction *MF = Entry->getParent();

  // We know that the MTVRSAVE instruction immediately follows MI.  Remove it.
  MachineBasicBlock::iterator MBBI = MI;
  ++MBBI;
  assert(MBBI != Entry->end() && MBBI->getOpcode() == PPC::MTVRSAVE);
  MBBI->eraseFromParent();
  
  bool RemovedAllMTVRSAVEs = true;
  // See if we can find and remove the MTVRSAVE instruction from all of the
  // epilog blocks.
  const TargetInstrInfo &TII = *MF->getTarget().getInstrInfo();
  for (MachineFunction::iterator I = MF->begin(), E = MF->end(); I != E; ++I) {
    // If last instruction is a return instruction, add an epilogue
    if (!I->empty() && TII.isReturn(I->back().getOpcode())) {
      bool FoundIt = false;
      for (MBBI = I->end(); MBBI != I->begin(); ) {
        --MBBI;
        if (MBBI->getOpcode() == PPC::MTVRSAVE) {
          MBBI->eraseFromParent();  // remove it.
          FoundIt = true;
          break;
        }
      }
      RemovedAllMTVRSAVEs &= FoundIt;
    }
  }

  // If we found and removed all MTVRSAVE instructions, remove the read of
  // VRSAVE as well.
  if (RemovedAllMTVRSAVEs) {
    MBBI = MI;
    assert(MBBI != Entry->begin() && "UPDATE_VRSAVE is first instr in block?");
    --MBBI;
    assert(MBBI->getOpcode() == PPC::MFVRSAVE && "VRSAVE instrs wandered?");
    MBBI->eraseFromParent();
  }
  
  // Finally, nuke the UPDATE_VRSAVE.
  MI->eraseFromParent();
}

// HandleVRSaveUpdate - MI is the UPDATE_VRSAVE instruction introduced by the
// instruction selector.  Based on the vector registers that have been used,
// transform this into the appropriate ORI instruction.
static void HandleVRSaveUpdate(MachineInstr *MI, const bool *UsedRegs) {
  unsigned UsedRegMask = 0;
  for (unsigned i = 0; i != 32; ++i)
    if (UsedRegs[VRRegNo[i]])
      UsedRegMask |= 1 << (31-i);
  
  // Live in and live out values already must be in the mask, so don't bother
  // marking them.
  MachineFunction *MF = MI->getParent()->getParent();
  for (MachineFunction::livein_iterator I = 
       MF->livein_begin(), E = MF->livein_end(); I != E; ++I) {
    unsigned RegNo = PPCRegisterInfo::getRegisterNumbering(I->first);
    if (VRRegNo[RegNo] == I->first)        // If this really is a vector reg.
      UsedRegMask &= ~(1 << (31-RegNo));   // Doesn't need to be marked.
  }
  for (MachineFunction::liveout_iterator I = 
       MF->liveout_begin(), E = MF->liveout_end(); I != E; ++I) {
    unsigned RegNo = PPCRegisterInfo::getRegisterNumbering(*I);
    if (VRRegNo[RegNo] == *I)              // If this really is a vector reg.
      UsedRegMask &= ~(1 << (31-RegNo));   // Doesn't need to be marked.
  }
  
  unsigned SrcReg = MI->getOperand(1).getReg();
  unsigned DstReg = MI->getOperand(0).getReg();
  // If no registers are used, turn this into a copy.
  if (UsedRegMask == 0) {
    // Remove all VRSAVE code.
    RemoveVRSaveCode(MI);
    return;
  } else if ((UsedRegMask & 0xFFFF) == UsedRegMask) {
    BuildMI(*MI->getParent(), MI, PPC::ORI, 2, DstReg)
        .addReg(SrcReg).addImm(UsedRegMask);
  } else if ((UsedRegMask & 0xFFFF0000) == UsedRegMask) {
    BuildMI(*MI->getParent(), MI, PPC::ORIS, 2, DstReg)
        .addReg(SrcReg).addImm(UsedRegMask >> 16);
  } else {
    BuildMI(*MI->getParent(), MI, PPC::ORIS, 2, DstReg)
       .addReg(SrcReg).addImm(UsedRegMask >> 16);
    BuildMI(*MI->getParent(), MI, PPC::ORI, 2, DstReg)
      .addReg(DstReg).addImm(UsedRegMask & 0xFFFF);
  }
  
  // Remove the old UPDATE_VRSAVE instruction.
  MI->eraseFromParent();
}


void PPCRegisterInfo::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();   // Prolog goes in entry BB
  MachineBasicBlock::iterator MBBI = MBB.begin();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineDebugInfo *DebugInfo = MFI->getMachineDebugInfo();
  
  // Do we have a frame pointer for this function?
  bool HasFP = hasFP(MF);

  // Scan the prolog, looking for an UPDATE_VRSAVE instruction.  If we find it,
  // process it.
  for (unsigned i = 0; MBBI != MBB.end(); ++i, ++MBBI) {
    if (MBBI->getOpcode() == PPC::UPDATE_VRSAVE) {
      HandleVRSaveUpdate(MBBI, MF.getUsedPhysregs());
      break;
    }
  }
  
  // Move MBBI back to the beginning of the function.
  MBBI = MBB.begin();
  
  // Get the number of bytes to allocate from the FrameInfo
  unsigned NumBytes = MFI->getStackSize();
  
  // Get the alignments provided by the target, and the maximum alignment
  // (if any) of the fixed frame objects.
  unsigned TargetAlign = MF.getTarget().getFrameInfo()->getStackAlignment();
  unsigned MaxAlign = MFI->getMaxAlignment();

  // If we have calls, we cannot use the red zone to store callee save registers
  // and we must set up a stack frame, so calculate the necessary size here.
  if (MFI->hasCalls()) {
    // We reserve argument space for call sites in the function immediately on
    // entry to the current function.  This eliminates the need for add/sub
    // brackets around call sites.
    NumBytes += MFI->getMaxCallFrameSize();
  }

  // If we are a leaf function, and use up to 224 bytes of stack space,
  // and don't have a frame pointer, then we do not need to adjust the stack
  // pointer (we fit in the Red Zone).
  if ((NumBytes == 0) || (NumBytes <= 224 && !HasFP && !MFI->hasCalls() &&
                          MaxAlign <= TargetAlign)) {
    MFI->setStackSize(0);
    return;
  }

  // Add the size of R1 to  NumBytes size for the store of R1 to the bottom
  // of the stack and round the size to a multiple of the alignment.
  unsigned Align = std::max(TargetAlign, MaxAlign);
  unsigned GPRSize = 4;
  unsigned Size = HasFP ? GPRSize + GPRSize : GPRSize;
  NumBytes = (NumBytes+Size+Align-1)/Align*Align;

  // Update frame info to pretend that this is part of the stack...
  MFI->setStackSize(NumBytes);
  int NegNumbytes = -NumBytes;

  // Adjust stack pointer: r1 -= numbytes.
  // If there is a preferred stack alignment, align R1 now
  if (MaxAlign > TargetAlign) {
    assert(isPowerOf2_32(MaxAlign) && MaxAlign < 32767 && "Invalid alignment!");
    assert(isInt16(0-NumBytes) && "Unhandled stack size and alignment!");
    BuildMI(MBB, MBBI, PPC::RLWINM, 4, PPC::R0)
      .addReg(PPC::R1).addImm(0).addImm(32-Log2_32(MaxAlign)).addImm(31);
    BuildMI(MBB, MBBI, PPC::SUBFIC,2,PPC::R0).addReg(PPC::R0)
      .addImm(0-NumBytes);
    BuildMI(MBB, MBBI, PPC::STWUX, 3)
      .addReg(PPC::R1).addReg(PPC::R1).addReg(PPC::R0);
  } else if (NumBytes <= 32768) {
    BuildMI(MBB, MBBI, PPC::STWU, 3).addReg(PPC::R1).addImm(NegNumbytes)
      .addReg(PPC::R1);
  } else {
    BuildMI(MBB, MBBI, PPC::LIS, 1, PPC::R0).addImm(NegNumbytes >> 16);
    BuildMI(MBB, MBBI, PPC::ORI, 2, PPC::R0).addReg(PPC::R0)
      .addImm(NegNumbytes & 0xFFFF);
    BuildMI(MBB, MBBI, PPC::STWUX, 3).addReg(PPC::R1).addReg(PPC::R1)
      .addReg(PPC::R0);
  }
  
  if (DebugInfo && DebugInfo->hasInfo()) {
    std::vector<MachineMove *> &Moves = DebugInfo->getFrameMoves();
    unsigned LabelID = DebugInfo->NextLabelID();
    
    // Show update of SP.
    MachineLocation Dst(MachineLocation::VirtualFP);
    MachineLocation Src(MachineLocation::VirtualFP, NegNumbytes);
    Moves.push_back(new MachineMove(LabelID, Dst, Src));

    BuildMI(MBB, MBBI, PPC::DWARF_LABEL, 1).addImm(LabelID);
  }
  
  // If there is a frame pointer, copy R1 (SP) into R31 (FP)
  if (HasFP) {
    BuildMI(MBB, MBBI, PPC::STW, 3)
      .addReg(PPC::R31).addImm(GPRSize).addReg(PPC::R1);
    BuildMI(MBB, MBBI, PPC::OR, 2, PPC::R31).addReg(PPC::R1).addReg(PPC::R1);
  }
}

void PPCRegisterInfo::emitEpilogue(MachineFunction &MF,
                                   MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  assert(MBBI->getOpcode() == PPC::BLR &&
         "Can only insert epilog into returning blocks");

  // Get alignment info so we know how to restore r1
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  unsigned TargetAlign = MF.getTarget().getFrameInfo()->getStackAlignment();

  // Get the number of bytes allocated from the FrameInfo.
  unsigned NumBytes = MFI->getStackSize();
  unsigned GPRSize = 4; 

  if (NumBytes != 0) {
    // If this function has a frame pointer, load the saved stack pointer from
    // its stack slot.
    if (hasFP(MF)) {
      BuildMI(MBB, MBBI, PPC::LWZ, 2, PPC::R31)
          .addImm(GPRSize).addReg(PPC::R31);
    }
    
    // The loaded (or persistent) stack pointer value is offseted by the 'stwu'
    // on entry to the function.  Add this offset back now.
    if (NumBytes < 32768 && TargetAlign >= MFI->getMaxAlignment()) {
      BuildMI(MBB, MBBI, PPC::ADDI, 2, PPC::R1)
          .addReg(PPC::R1).addImm(NumBytes);
    } else {
      BuildMI(MBB, MBBI, PPC::LWZ, 2, PPC::R1).addImm(0).addReg(PPC::R1);
    }
  }
}

unsigned PPCRegisterInfo::getRARegister() const {
  return PPC::LR;
}

unsigned PPCRegisterInfo::getFrameRegister(MachineFunction &MF) const {
  return hasFP(MF) ? PPC::R31 : PPC::R1;
}

void PPCRegisterInfo::getInitialFrameState(std::vector<MachineMove *> &Moves)
                                                                         const {
  // Initial state is the frame pointer is R1.
  MachineLocation Dst(MachineLocation::VirtualFP);
  MachineLocation Src(PPC::R1, 0);
  Moves.push_back(new MachineMove(0, Dst, Src));
}

#include "PPCGenRegisterInfo.inc"

