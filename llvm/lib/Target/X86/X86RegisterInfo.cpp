//===- X86RegisterInfo.cpp - X86 Register Information -----------*- C++ -*-===//
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

void X86RegisterInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
					  MachineBasicBlock::iterator &MBBI,
					  unsigned SrcReg, int FrameIdx,
					  const TargetRegisterClass *RC) const {
  static const unsigned Opcode[] =
    { X86::MOVrm8, X86::MOVrm16, X86::MOVrm32, X86::FSTPr80 };
  MachineInstr *MI = addFrameReference(BuildMI(Opcode[getIdx(RC)], 5),
				       FrameIdx).addReg(SrcReg);
  MBBI = MBB.insert(MBBI, MI)+1;
}

void X86RegisterInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
					   MachineBasicBlock::iterator &MBBI,
					   unsigned DestReg, int FrameIdx,
					   const TargetRegisterClass *RC) const{
  static const unsigned Opcode[] =
    { X86::MOVmr8, X86::MOVmr16, X86::MOVmr32, X86::FLDr80 };
  MachineInstr *MI = addFrameReference(BuildMI(Opcode[getIdx(RC)], 4, DestReg),
				       FrameIdx);
  MBBI = MBB.insert(MBBI, MI)+1;
}

void X86RegisterInfo::copyRegToReg(MachineBasicBlock &MBB,
				   MachineBasicBlock::iterator &MBBI,
				   unsigned DestReg, unsigned SrcReg,
				   const TargetRegisterClass *RC) const {
  static const unsigned Opcode[] =
    { X86::MOVrr8, X86::MOVrr16, X86::MOVrr32, X86::FpMOV };
  MachineInstr *MI = BuildMI(Opcode[getIdx(RC)],1,DestReg).addReg(SrcReg);
  MBBI = MBB.insert(MBBI, MI)+1;
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

// hasSPAdjust - Return true if this function has ESP adjustment instructions in
// the prolog and epilog which allocate local stack space.  This is necessary
// because we elide these instructions if there are no function calls in the
// current function (ie, this is a leaf function).  In this case, we can refer
// beyond the stack pointer because we know that nothing will trample on that
// part of the stack.
//
static bool hasSPAdjust(MachineFunction &MF) {
  assert(!hasFP(MF) && "Can only eliminate SP adjustment if no frame-pointer!");
  return MF.getFrameInfo()->hasCalls();
}

void X86RegisterInfo::eliminateCallFramePseudoInstr(MachineFunction &MF,
						    MachineBasicBlock &MBB,
	                                 MachineBasicBlock::iterator &I) const {
  MachineInstr *New = 0, *Old = *I;;
  if (hasFP(MF)) {
    // If we have a frame pointer, turn the adjcallstackup instruction into a
    // 'sub ESP, <amt>' and the adjcallstackdown instruction into 'add ESP,
    // <amt>'
    unsigned Amount = Old->getOperand(0).getImmedValue();
    if (Amount != 0) {
      // We need to keep the stack aligned properly.  To do this, we round the
      // amount of space needed for the outgoing arguments up to the next
      // alignment boundary.
      unsigned Align = MF.getTarget().getFrameInfo().getStackAlignment();
      Amount = (Amount+Align-1)/Align*Align;

      if (Old->getOpcode() == X86::ADJCALLSTACKDOWN) {
	New=BuildMI(X86::SUBri32, 2, X86::ESP).addReg(X86::ESP).addZImm(Amount);
      } else {
	assert(Old->getOpcode() == X86::ADJCALLSTACKUP);
	New=BuildMI(X86::ADDri32, 2, X86::ESP).addReg(X86::ESP).addZImm(Amount);
      }
    }
  }

  if (New)
    *I = New;        // Replace the pseudo instruction with a new instruction...
  else
    I = MBB.erase(I);// Just delete the pseudo instruction...
  delete Old;
}

void X86RegisterInfo::eliminateFrameIndex(MachineFunction &MF,
					MachineBasicBlock::iterator &II) const {
  unsigned i = 0;
  MachineInstr &MI = **II;
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

  if (!hasFP(MF) && hasSPAdjust(MF)) {
    const MachineFrameInfo *MFI = MF.getFrameInfo();
    Offset += MFI->getStackSize();
  }

  MI.SetMachineOperandConst(i+3, MachineOperand::MO_SignExtendedImmed, Offset);
}

void X86RegisterInfo::processFunctionBeforeFrameFinalized(MachineFunction &MF)
  const {
  if (hasFP(MF)) {
    // Create a frame entry for the EBP register that must be saved.
    int FrameIdx = MF.getFrameInfo()->CreateStackObject(4, 4);
    assert(FrameIdx == MF.getFrameInfo()->getObjectIndexEnd()-1 &&
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
    int EBPOffset = MFI->getObjectOffset(MFI->getObjectIndexEnd()-1)+4;

    MI = addRegOffset(BuildMI(X86::MOVrm32, 5),    // mov [ESP-<offset>], EBP
		      X86::ESP, EBPOffset).addReg(X86::EBP);
    MBBI = MBB.insert(MBBI, MI)+1;
    
    MI = BuildMI(X86::MOVrr32, 2, X86::EBP).addReg(X86::ESP);
    MBBI = MBB.insert(MBBI, MI)+1;
  } else {
    // If we don't have a frame pointer, and the function contains no call sites
    // (it's a leaf function), we don't have to emit ANY stack adjustment
    // instructions at all, we can just refer to the area beyond the stack
    // pointer.  This can be important for small functions.
    //
    if (!hasSPAdjust(MF)) return;

    // When we have no frame pointer, we reserve argument space for call sites
    // in the function immediately on entry to the current function.  This
    // eliminates the need for add/sub ESP brackets around call sites.
    //
    NumBytes += MFI->getMaxCallFrameSize();

    // Round the size to a multiple of the alignment (don't forget the 4 byte
    // offset though).
    unsigned Align = MF.getTarget().getFrameInfo().getStackAlignment();
    NumBytes = ((NumBytes+4)+Align-1)/Align*Align - 4;

    // Update frame info to pretend that this is part of the stack...
    MFI->setStackSize(NumBytes);
  }

  if (NumBytes) {
    // adjust stack pointer: ESP -= numbytes
    MI  = BuildMI(X86::SUBri32, 2, X86::ESP).addReg(X86::ESP).addZImm(NumBytes);
    MBBI = 1+MBB.insert(MBBI, MI);
  }
}

void X86RegisterInfo::emitEpilogue(MachineFunction &MF,
				   MachineBasicBlock &MBB) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineBasicBlock::iterator MBBI = MBB.end()-1;
  MachineInstr *MI;
  assert((*MBBI)->getOpcode() == X86::RET &&
         "Can only insert epilog into returning blocks");

  if (hasFP(MF)) {
    // Get the offset of the stack slot for the EBP register... which is
    // guaranteed to be the last slot by processFunctionBeforeFrameFinalized.
    int EBPOffset = MFI->getObjectOffset(MFI->getObjectIndexEnd()-1)+4;
    
    // mov ESP, EBP
    MI = BuildMI(X86::MOVrr32, 1,X86::ESP).addReg(X86::EBP);
    MBBI = 1+MBB.insert(MBBI, MI);

    // mov EBP, [ESP-<offset>]
    MI = addRegOffset(BuildMI(X86::MOVmr32, 5, X86::EBP), X86::ESP, EBPOffset);
    MBBI = 1+MBB.insert(MBBI, MI);
  } else {
    if (!hasSPAdjust(MF)) return;

    // Get the number of bytes allocated from the FrameInfo...
    unsigned NumBytes = MFI->getStackSize();

    if (NumBytes) {    // adjust stack pointer back: ESP += numbytes
      MI =BuildMI(X86::ADDri32, 2, X86::ESP).addReg(X86::ESP).addZImm(NumBytes);
      MBBI = 1+MBB.insert(MBBI, MI);
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
