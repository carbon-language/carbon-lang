//===- PPC32RegisterInfo.cpp - PowerPC32 Register Information ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PowerPC32 implementation of the MRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "reginfo"
#include "PowerPC.h"
#include "PowerPCInstrBuilder.h"
#include "PPC32RegisterInfo.h"
#include "llvm/Constants.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdlib>
#include <iostream>
using namespace llvm;

namespace llvm {
  // Switch toggling compilation for AIX
  extern cl::opt<bool> AIX;
}

PPC32RegisterInfo::PPC32RegisterInfo()
  : PPC32GenRegisterInfo(PPC::ADJCALLSTACKDOWN, PPC::ADJCALLSTACKUP) {
  ImmToIdxMap[PPC::LD]   = PPC::LDX;    ImmToIdxMap[PPC::STD]  = PPC::STDX;
  ImmToIdxMap[PPC::LBZ]  = PPC::LBZX;   ImmToIdxMap[PPC::STB]  = PPC::STBX;
  ImmToIdxMap[PPC::LHZ]  = PPC::LHZX;   ImmToIdxMap[PPC::LHA]  = PPC::LHAX;
  ImmToIdxMap[PPC::LWZ]  = PPC::LWZX;   ImmToIdxMap[PPC::LWA]  = PPC::LWAX;
  ImmToIdxMap[PPC::LFS]  = PPC::LFSX;   ImmToIdxMap[PPC::LFD]  = PPC::LFDX;
  ImmToIdxMap[PPC::STH]  = PPC::STHX;   ImmToIdxMap[PPC::STW]  = PPC::STWX;
  ImmToIdxMap[PPC::STFS] = PPC::STFSX;  ImmToIdxMap[PPC::STFD] = PPC::STFDX;
  ImmToIdxMap[PPC::ADDI] = PPC::ADD;
}

static const TargetRegisterClass *getClass(unsigned SrcReg) {
  if (PPC32::FPRCRegisterClass->contains(SrcReg))
    return PPC32::FPRCRegisterClass;
  assert(PPC32::GPRCRegisterClass->contains(SrcReg) && "Reg not FPR or GPR");
  return PPC32::GPRCRegisterClass;
}

static unsigned getIdx(const TargetRegisterClass *RC) {
  if (RC == PPC32::GPRCRegisterClass) {
    switch (RC->getSize()) {
      default: assert(0 && "Invalid data size!");
      case 1:  return 0;
      case 2:  return 1;
      case 4:  return 2;
    }
  } else if (RC == PPC32::FPRCRegisterClass) {
    switch (RC->getSize()) {
      default: assert(0 && "Invalid data size!");
      case 4:  return 3;
      case 8:  return 4;
    }
  } else if (RC == PPC32::CRRCRegisterClass) {
    switch (RC->getSize()) {
      default: assert(0 && "Invalid data size!");
      case 4:  return 2;
    }
  }
  std::cerr << "Invalid register class to getIdx()!\n";
  abort();
}

void
PPC32RegisterInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator MI,
                                       unsigned SrcReg, int FrameIdx) const {
  static const unsigned Opcode[] = {
    PPC::STB, PPC::STH, PPC::STW, PPC::STFS, PPC::STFD
  };
  unsigned OC = Opcode[getIdx(getClass(SrcReg))];
  if (SrcReg == PPC::LR) {
    BuildMI(MBB, MI, PPC::MFLR, 1, PPC::R11).addReg(PPC::LR);
    addFrameReference(BuildMI(MBB, MI, OC, 3).addReg(PPC::R11),FrameIdx);
  } else if (PPC32::CRRCRegisterClass == getClass(SrcReg)) {
    BuildMI(MBB, MI, PPC::MFCR, 0, PPC::R11);
    addFrameReference(BuildMI(MBB, MI, OC, 3).addReg(PPC::R11),FrameIdx);
  } else {
    addFrameReference(BuildMI(MBB, MI, OC, 3).addReg(SrcReg),FrameIdx);
  }
}

void
PPC32RegisterInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MI,
                                        unsigned DestReg, int FrameIdx) const{
  static const unsigned Opcode[] = {
    PPC::LBZ, PPC::LHZ, PPC::LWZ, PPC::LFS, PPC::LFD
  };
  unsigned OC = Opcode[getIdx(getClass(DestReg))];
  if (DestReg == PPC::LR) {
    addFrameReference(BuildMI(MBB, MI, OC, 2, PPC::R11), FrameIdx);
    BuildMI(MBB, MI, PPC::MTLR, 1).addReg(PPC::R11);
  } else if (PPC32::CRRCRegisterClass == getClass(DestReg)) {
    addFrameReference(BuildMI(MBB, MI, OC, 2, PPC::R11), FrameIdx);
    BuildMI(MBB, MI, PPC::MTCRF, 1, DestReg).addReg(PPC::R11);
  } else {
    addFrameReference(BuildMI(MBB, MI, OC, 2, DestReg), FrameIdx);
  }
}

void PPC32RegisterInfo::copyRegToReg(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator MI,
                                     unsigned DestReg, unsigned SrcReg,
                                     const TargetRegisterClass *RC) const {
  MachineInstr *I;

  if (RC == PPC32::GPRCRegisterClass) {
    BuildMI(MBB, MI, PPC::OR, 2, DestReg).addReg(SrcReg).addReg(SrcReg);
  } else if (RC == PPC32::FPRCRegisterClass) {
    BuildMI(MBB, MI, PPC::FMR, 1, DestReg).addReg(SrcReg);
  } else if (RC == PPC32::CRRCRegisterClass) {
    BuildMI(MBB, MI, PPC::MCRF, 1, DestReg).addReg(SrcReg);
  } else {
    std::cerr << "Attempt to copy register that is not GPR or FPR";
    abort();
  }
}

//===----------------------------------------------------------------------===//
// Stack Frame Processing methods
//===----------------------------------------------------------------------===//

// hasFP - Return true if the specified function should have a dedicated frame
// pointer register.  This is true if the function has variable sized allocas or
// if frame pointer elimination is disabled.
//
static bool hasFP(MachineFunction &MF) {
  MachineFrameInfo *MFI = MF.getFrameInfo();
  return MFI->hasVarSizedObjects();
}

void PPC32RegisterInfo::
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
        MBB.insert(I, BuildMI(PPC::ADDI, 2, PPC::R1).addReg(PPC::R1)
                .addSImm(-Amount));
      } else {
        assert(Old->getOpcode() == PPC::ADJCALLSTACKUP);
        MBB.insert(I, BuildMI(PPC::ADDI, 2, PPC::R1).addReg(PPC::R1)
                .addSImm(Amount));
      }
    }
  }
  MBB.erase(I);
}

void
PPC32RegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II) const {
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
  MI.SetMachineOperandReg(i, hasFP(MF) ? PPC::R31 : PPC::R1);

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
    MBB->insert(II, BuildMI(PPC::LIS, 1, PPC::R0).addSImm(Offset >> 16));
    MBB->insert(II, BuildMI(PPC::ORI, 2, PPC::R0).addReg(PPC::R0)
      .addImm(Offset));
    // convert into indexed form of the instruction
    // sth 0:rA, 1:imm 2:(rB) ==> sthx 0:rA, 2:rB, 1:r0
    // addi 0:rA 1:rB, 2, imm ==> add 0:rA, 1:rB, 2:r0
    unsigned NewOpcode = const_cast<std::map<unsigned, unsigned>& >(ImmToIdxMap)[MI.getOpcode()];
    assert(NewOpcode && "No indexed form of load or store available!");
    MI.setOpcode(NewOpcode);
    MI.SetMachineOperandReg(1, MI.getOperand(i).getReg());
    MI.SetMachineOperandReg(2, PPC::R0);
  } else {
    MI.SetMachineOperandConst(OffIdx,MachineOperand::MO_SignExtendedImmed,Offset);
  }
}


void PPC32RegisterInfo::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();   // Prolog goes in entry BB
  MachineBasicBlock::iterator MBBI = MBB.begin();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineInstr *MI;

  // Get the number of bytes to allocate from the FrameInfo
  unsigned NumBytes = MFI->getStackSize();

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
  if ((NumBytes == 0) || (NumBytes <= 224 && !hasFP(MF) && !MFI->hasCalls())) {
    MFI->setStackSize(0);
    return;
  }

  // Add the size of R1 to  NumBytes size for the store of R1 to the bottom
  // of the stack and round the size to a multiple of the alignment.
  unsigned Align = MF.getTarget().getFrameInfo()->getStackAlignment();
  unsigned GPRSize = getSpillSize(PPC::R1)/8;
  unsigned Size = hasFP(MF) ? GPRSize + GPRSize : GPRSize;
  NumBytes = (NumBytes+Size+Align-1)/Align*Align;

  // Update frame info to pretend that this is part of the stack...
  MFI->setStackSize(NumBytes);

  // Adjust stack pointer: r1 -= numbytes.
  if (NumBytes <= 32768) {
    MI=BuildMI(PPC::STWU,3).addReg(PPC::R1).addSImm(-NumBytes).addReg(PPC::R1);
    MBB.insert(MBBI, MI);
  } else {
    int NegNumbytes = -NumBytes;
    MI = BuildMI(PPC::LIS, 1, PPC::R0).addSImm(NegNumbytes >> 16);
    MBB.insert(MBBI, MI);
    MI = BuildMI(PPC::ORI, 2, PPC::R0).addReg(PPC::R0)
      .addImm(NegNumbytes & 0xFFFF);
    MBB.insert(MBBI, MI);
    MI = BuildMI(PPC::STWUX, 3).addReg(PPC::R1).addReg(PPC::R1).addReg(PPC::R0);
    MBB.insert(MBBI, MI);
  }

  if (hasFP(MF)) {
    MI = BuildMI(PPC::STW, 3).addReg(PPC::R31).addSImm(GPRSize).addReg(PPC::R1);
    MBB.insert(MBBI, MI);
    MI = BuildMI(PPC::OR, 2, PPC::R31).addReg(PPC::R1).addReg(PPC::R1);
    MBB.insert(MBBI, MI);
  }
}

void PPC32RegisterInfo::emitEpilogue(MachineFunction &MF,
                                     MachineBasicBlock &MBB) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  MachineInstr *MI;
  assert(MBBI->getOpcode() == PPC::BLR &&
         "Can only insert epilog into returning blocks");

  // Get the number of bytes allocated from the FrameInfo...
  unsigned NumBytes = MFI->getStackSize();
  unsigned GPRSize = getSpillSize(PPC::R31)/8;

  if (NumBytes != 0) {
    if (hasFP(MF)) {
      MI = BuildMI(PPC::LWZ, 2, PPC::R31).addSImm(GPRSize).addReg(PPC::R31);
      MBB.insert(MBBI, MI);
    }
    MI = BuildMI(PPC::LWZ, 2, PPC::R1).addSImm(0).addReg(PPC::R1);
    MBB.insert(MBBI, MI);
  }
}

#include "PPC32GenRegisterInfo.inc"

const TargetRegisterClass*
PPC32RegisterInfo::getRegClassForType(const Type* Ty) const {
  switch (Ty->getTypeID()) {
    default:              assert(0 && "Invalid type to getClass!");
    case Type::LongTyID:
    case Type::ULongTyID: assert(0 && "Long values can't fit in registers!");
    case Type::BoolTyID:
    case Type::SByteTyID:
    case Type::UByteTyID:
    case Type::ShortTyID:
    case Type::UShortTyID:
    case Type::IntTyID:
    case Type::UIntTyID:
    case Type::PointerTyID: return &GPRCInstance;

    case Type::FloatTyID:
    case Type::DoubleTyID: return &FPRCInstance;
  }
}

