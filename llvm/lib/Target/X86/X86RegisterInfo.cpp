//===- X86RegisterInfo.cpp - X86 Register Information -----------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains the X86 implementation of the MRegisterInfo class.  This
// file is responsible for the frame pointer elimination optimization on X86.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86RegisterInfo.h"
#include "X86InstrBuilder.h"
#include "llvm/Constants.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "Support/CommandLine.h"
#include "Support/STLExtras.h"
using namespace llvm;

namespace {
  cl::opt<bool>
  NoFPElim("disable-fp-elim",
	   cl::desc("Disable frame pointer elimination optimization"));
}

X86RegisterInfo::X86RegisterInfo()
  : X86GenRegisterInfo(X86::ADJCALLSTACKDOWN, X86::ADJCALLSTACKUP) {}

static unsigned getIdx(const TargetRegisterClass *RC) {
  switch (RC->getSize()) {
  default: assert(0 && "Invalid data size!");
  case 1:  return 0;
  case 2:  return 1;
  case 4:  return 2;
  case 10: return 3;
  }
}

int X86RegisterInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator MI,
                                         unsigned SrcReg, int FrameIdx,
                                         const TargetRegisterClass *RC) const {
  static const unsigned Opcode[] =
    { X86::MOVrm8, X86::MOVrm16, X86::MOVrm32, X86::FSTPr80 };
  MachineInstr *I = addFrameReference(BuildMI(Opcode[getIdx(RC)], 5),
				       FrameIdx).addReg(SrcReg);
  MBB.insert(MI, I);
  return 1;
}

int X86RegisterInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator MI,
                                          unsigned DestReg, int FrameIdx,
                                          const TargetRegisterClass *RC) const{
  static const unsigned Opcode[] =
    { X86::MOVmr8, X86::MOVmr16, X86::MOVmr32, X86::FLDr80 };
  unsigned OC = Opcode[getIdx(RC)];
  MBB.insert(MI, addFrameReference(BuildMI(OC, 4, DestReg), FrameIdx));
  return 1;
}

int X86RegisterInfo::copyRegToReg(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator MI,
                                  unsigned DestReg, unsigned SrcReg,
                                  const TargetRegisterClass *RC) const {
  static const unsigned Opcode[] =
    { X86::MOVrr8, X86::MOVrr16, X86::MOVrr32, X86::FpMOV };
  MBB.insert(MI, BuildMI(Opcode[getIdx(RC)],1,DestReg).addReg(SrcReg));
  return 1;
}

//===----------------------------------------------------------------------===//
// Stack Frame Processing methods
//===----------------------------------------------------------------------===//

// hasFP - Return true if the specified function should have a dedicated frame
// pointer register.  This is true if the function has variable sized allocas or
// if frame pointer elimination is disabled.
//
static bool hasFP(MachineFunction &MF) {
  return NoFPElim || MF.getFrameInfo()->hasVarSizedObjects();
}

void X86RegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  if (hasFP(MF)) {
    // If we have a frame pointer, turn the adjcallstackup instruction into a
    // 'sub ESP, <amt>' and the adjcallstackdown instruction into 'add ESP,
    // <amt>'
    MachineInstr *Old = I;
    unsigned Amount = Old->getOperand(0).getImmedValue();
    if (Amount != 0) {
      // We need to keep the stack aligned properly.  To do this, we round the
      // amount of space needed for the outgoing arguments up to the next
      // alignment boundary.
      unsigned Align = MF.getTarget().getFrameInfo().getStackAlignment();
      Amount = (Amount+Align-1)/Align*Align;

      MachineInstr *New;
      if (Old->getOpcode() == X86::ADJCALLSTACKDOWN) {
	New=BuildMI(X86::SUBri32, 1, X86::ESP, MOTy::UseAndDef).addZImm(Amount);
      } else {
	assert(Old->getOpcode() == X86::ADJCALLSTACKUP);
	New=BuildMI(X86::ADDri32, 1, X86::ESP, MOTy::UseAndDef).addZImm(Amount);
      }

      // Replace the pseudo instruction with a new instruction...
      MBB.insert(I, New);
    }
  }

  MBB.erase(I);
}

void X86RegisterInfo::eliminateFrameIndex(MachineFunction &MF,
                                         MachineBasicBlock::iterator II) const {
  unsigned i = 0;
  MachineInstr &MI = *II;
  while (!MI.getOperand(i).isFrameIndex()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }

  int FrameIndex = MI.getOperand(i).getFrameIndex();

  // This must be part of a four operand memory reference.  Replace the
  // FrameIndex with base register with EBP.  Add add an offset to the offset.
  MI.SetMachineOperandReg(i, hasFP(MF) ? X86::EBP : X86::ESP);

  // Now add the frame object offset to the offset from EBP.
  int Offset = MF.getFrameInfo()->getObjectOffset(FrameIndex) +
               MI.getOperand(i+3).getImmedValue()+4;

  if (!hasFP(MF))
    Offset += MF.getFrameInfo()->getStackSize();
  else
    Offset += 4;  // Skip the saved EBP

  MI.SetMachineOperandConst(i+3, MachineOperand::MO_SignExtendedImmed, Offset);
}

void
X86RegisterInfo::processFunctionBeforeFrameFinalized(MachineFunction &MF) const{
  if (hasFP(MF)) {
    // Create a frame entry for the EBP register that must be saved.
    int FrameIdx = MF.getFrameInfo()->CreateFixedObject(4, -8);
    assert(FrameIdx == MF.getFrameInfo()->getObjectIndexBegin() &&
           "Slot for EBP register must be last in order to be found!");
  }
}

void X86RegisterInfo::emitPrologue(MachineFunction &MF) const {
  MachineBasicBlock &MBB = MF.front();   // Prolog goes in entry BB
  MachineBasicBlock::iterator MBBI = MBB.begin();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineInstr *MI;

  // Get the number of bytes to allocate from the FrameInfo
  unsigned NumBytes = MFI->getStackSize();
  if (hasFP(MF)) {
    // Get the offset of the stack slot for the EBP register... which is
    // guaranteed to be the last slot by processFunctionBeforeFrameFinalized.
    int EBPOffset = MFI->getObjectOffset(MFI->getObjectIndexBegin())+4;

    if (NumBytes) {   // adjust stack pointer: ESP -= numbytes
      MI= BuildMI(X86::SUBri32, 1, X86::ESP, MOTy::UseAndDef).addZImm(NumBytes);
      MBB.insert(MBBI, MI);
    }

    // Save EBP into the appropriate stack slot...
    MI = addRegOffset(BuildMI(X86::MOVrm32, 5),    // mov [ESP-<offset>], EBP
		      X86::ESP, EBPOffset+NumBytes).addReg(X86::EBP);
    MBB.insert(MBBI, MI);

    // Update EBP with the new base value...
    if (NumBytes == 4)    // mov EBP, ESP
      MI = BuildMI(X86::MOVrr32, 2, X86::EBP).addReg(X86::ESP);
    else                  // lea EBP, [ESP+StackSize]
      MI = addRegOffset(BuildMI(X86::LEAr32, 5, X86::EBP), X86::ESP,NumBytes-4);

    MBB.insert(MBBI, MI);

  } else {
    if (MFI->hasCalls()) {
      // When we have no frame pointer, we reserve argument space for call sites
      // in the function immediately on entry to the current function.  This
      // eliminates the need for add/sub ESP brackets around call sites.
      //
      NumBytes += MFI->getMaxCallFrameSize();
      
      // Round the size to a multiple of the alignment (don't forget the 4 byte
      // offset though).
      unsigned Align = MF.getTarget().getFrameInfo().getStackAlignment();
      NumBytes = ((NumBytes+4)+Align-1)/Align*Align - 4;
    }

    // Update frame info to pretend that this is part of the stack...
    MFI->setStackSize(NumBytes);

    if (NumBytes) {
      // adjust stack pointer: ESP -= numbytes
      MI= BuildMI(X86::SUBri32, 1, X86::ESP, MOTy::UseAndDef).addZImm(NumBytes);
      MBB.insert(MBBI, MI);
    }
  }
}

void X86RegisterInfo::emitEpilogue(MachineFunction &MF,
                                   MachineBasicBlock &MBB) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  MachineInstr *MI;
  assert(MBBI->getOpcode() == X86::RET &&
         "Can only insert epilog into returning blocks");

  if (hasFP(MF)) {
    // Get the offset of the stack slot for the EBP register... which is
    // guaranteed to be the last slot by processFunctionBeforeFrameFinalized.
    int EBPOffset = MFI->getObjectOffset(MFI->getObjectIndexEnd()-1)+4;
    
    // mov ESP, EBP
    MI = BuildMI(X86::MOVrr32, 1,X86::ESP).addReg(X86::EBP);
    MBB.insert(MBBI, MI);

    // pop EBP
    MI = BuildMI(X86::POPr32, 0, X86::EBP);
    MBB.insert(MBBI, MI);
  } else {
    // Get the number of bytes allocated from the FrameInfo...
    unsigned NumBytes = MFI->getStackSize();

    if (NumBytes) {    // adjust stack pointer back: ESP += numbytes
      MI =BuildMI(X86::ADDri32, 1, X86::ESP, MOTy::UseAndDef).addZImm(NumBytes);
      MBB.insert(MBBI, MI);
    }
  }
}

#include "X86GenRegisterInfo.inc"

const TargetRegisterClass*
X86RegisterInfo::getRegClassForType(const Type* Ty) const {
  switch (Ty->getPrimitiveID()) {
  case Type::LongTyID:
  case Type::ULongTyID: assert(0 && "Long values can't fit in registers!");
  default:              assert(0 && "Invalid type to getClass!");
  case Type::BoolTyID:
  case Type::SByteTyID:
  case Type::UByteTyID:   return &R8Instance;
  case Type::ShortTyID:
  case Type::UShortTyID:  return &R16Instance;
  case Type::IntTyID:
  case Type::UIntTyID:
  case Type::PointerTyID: return &R32Instance;
    
  case Type::FloatTyID:
  case Type::DoubleTyID: return &RFPInstance;
  }
}
