//===- PowerPCRegisterInfo.cpp - PowerPC Register Information ---*- C++ -*-===//
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

#include "PowerPC.h"
#include "PowerPCRegisterInfo.h"
#include "PowerPCInstrBuilder.h"
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
  NoFPElim("disable-fp-elim",cl::desc("Disable frame pointer elimination optimization"));
}

PowerPCRegisterInfo::PowerPCRegisterInfo()
  : PowerPCGenRegisterInfo(PPC32::ADJCALLSTACKDOWN,
                           PPC32::ADJCALLSTACKUP) {}

static unsigned getIdx(const TargetRegisterClass *RC) {
	if (RC == PowerPC::GPRCRegisterClass) {
		switch (RC->getSize()) {
			default: assert(0 && "Invalid data size!");
			case 1:  return 0;
			case 2:  return 1;
			case 4:  return 2;
		}
	}
	else if (RC == PowerPC::FPRCRegisterClass) {
		switch (RC->getSize()) {
			default: assert(0 && "Invalid data size!");
			case 4:  return 3;
			case 8:  return 4;
		}
	}
	abort();
}

int PowerPCRegisterInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator MI,
                                         unsigned SrcReg, int FrameIdx,
                                         const TargetRegisterClass *RC) const {
	static const unsigned Opcode[] = 
	{ PPC32::STB, PPC32::STH, PPC32::STW, PPC32::STFS, PPC32::STFD };

	unsigned OC = Opcode[getIdx(RC)];
	MBB.insert(MI, addFrameReference(BuildMI(OC, 3).addReg(SrcReg),FrameIdx));
	return 1;
}

int PowerPCRegisterInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator MI,
                                          unsigned DestReg, int FrameIdx,
                                          const TargetRegisterClass *RC) const{
	static const unsigned Opcode[] =

	{ PPC32::LBZ, PPC32::LHZ, PPC32::LWZ, PPC32::LFS, PPC32::LFD };
	unsigned OC = Opcode[getIdx(RC)];
	MBB.insert(MI, addFrameReference(BuildMI(OC, 2, DestReg), FrameIdx));
	return 1;
}

int PowerPCRegisterInfo::copyRegToReg(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator MI,
                                  unsigned DestReg, unsigned SrcReg,
                                  const TargetRegisterClass *RC) const {
	MachineInstr *I;

	if(RC == PowerPC::GPRCRegisterClass) {
		I = BuildMI(PPC32::OR, 2, DestReg).addReg(SrcReg).addReg(SrcReg);
	} else if (RC == PowerPC::FPRCRegisterClass) {
		I = BuildMI(PPC32::FMR, 1, DestReg).addReg(SrcReg);
	} else { 
		std::cerr << "Attempt to copy register that is not GPR or FPR";
		abort();
	}
	MBB.insert(MI, I);
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

void PowerPCRegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
	if (hasFP(MF)) {
		// If we have a frame pointer, turn the adjcallstackdown instruction into a
		// 'sub r1, r1, <amt>' and the adjcallstackup instruction into 'add r1, r1, <amt>'
		MachineInstr *Old = I;
		int Amount = Old->getOperand(0).getImmedValue();
		if (Amount != 0) {
			// We need to keep the stack aligned properly.  To do this, we round the
			// amount of space needed for the outgoing arguments up to the next
			// alignment boundary.
			unsigned Align = MF.getTarget().getFrameInfo()->getStackAlignment();
			Amount = (Amount+Align-1)/Align*Align;
			
			MachineInstr *New;
			if (Old->getOpcode() == PPC32::ADJCALLSTACKDOWN) {
				New=BuildMI(PPC32::ADDI, 2, PPC32::R1).addReg(PPC32::R1).addSImm(-Amount);
			} else {
				assert(Old->getOpcode() == PPC32::ADJCALLSTACKUP);
				New=BuildMI(PPC32::ADDI, 2, PPC32::R1).addReg(PPC32::R1).addSImm(Amount);
			}
			
			// Replace the pseudo instruction with a new instruction...
			MBB.insert(I, New);
		}
	}
	
	MBB.erase(I);
}

void
PowerPCRegisterInfo::eliminateFrameIndex(MachineFunction &MF,
                                         MachineBasicBlock::iterator II) const {
  unsigned i = 0;
  MachineInstr &MI = *II;
  while (!MI.getOperand(i).isFrameIndex()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }

  int FrameIndex = MI.getOperand(i).getFrameIndex();

  // This must be part of a four operand memory reference.  Replace the
  // FrameIndex with base register with GPR1.
  MI.SetMachineOperandReg(i, PPC32::R1);

  // Take into account whether its an add or mem instruction
  if (i == 2) i--;

  // Now add the frame object offset to the offset from r1.
  int Offset = MF.getFrameInfo()->getObjectOffset(FrameIndex) +
               MI.getOperand(i).getImmedValue()+4;

  if (!hasFP(MF))
    Offset += MF.getFrameInfo()->getStackSize();

  MI.SetMachineOperandConst(i-1, MachineOperand::MO_SignExtendedImmed, Offset);
  std::cout << "offset = " << Offset << std::endl;
}


void PowerPCRegisterInfo::processFunctionBeforeFrameFinalized(
    MachineFunction &MF) const {
	// Do Nothing
}

void PowerPCRegisterInfo::emitPrologue(MachineFunction &MF) const {
	MachineBasicBlock &MBB = MF.front();   // Prolog goes in entry BB
	MachineBasicBlock::iterator MBBI = MBB.begin();
	MachineFrameInfo *MFI = MF.getFrameInfo();
	MachineInstr *MI;
	
	// Get the number of bytes to allocate from the FrameInfo
	unsigned NumBytes = MFI->getStackSize();
	
    if (MFI->hasCalls()) {
		// When we have no frame pointer, we reserve argument space for call sites
		// in the function immediately on entry to the current function.  This
		// eliminates the need for add/sub brackets around call sites.
		//
		NumBytes += MFI->getMaxCallFrameSize();
		
		// Round the size to a multiple of the alignment (don't forget the 4 byte
		// offset though).
		unsigned Align = MF.getTarget().getFrameInfo()->getStackAlignment();
		NumBytes = ((NumBytes+4)+Align-1)/Align*Align - 4;

		// Store the incoming LR so it is preserved across calls
		MI= BuildMI(PPC32::MovePCtoLR, 0, PPC32::LR).addReg(PPC32::LR);
		MBB.insert(MBBI, MI);
		MI= BuildMI(PPC32::MFSPR, 1, PPC32::R0).addImm(8);
		MBB.insert(MBBI, MI);
		MI= BuildMI(PPC32::STW, 3).addReg(PPC32::R0).addSImm(8).addReg(PPC32::R1);
		MBB.insert(MBBI, MI);
    }
	
    // Update frame info to pretend that this is part of the stack...
    MFI->setStackSize(NumBytes);
	
	// adjust stack pointer: r1 -= numbytes
	if (NumBytes) {
		MI= BuildMI(PPC32::STWU, 2, PPC32::R1).addImm(-NumBytes).addReg(PPC32::R1);
		MBB.insert(MBBI, MI);
    }
}

void PowerPCRegisterInfo::emitEpilogue(MachineFunction &MF,
                                       MachineBasicBlock &MBB) const {
	const MachineFrameInfo *MFI = MF.getFrameInfo();
	MachineBasicBlock::iterator MBBI = prior(MBB.end());
	MachineInstr *MI;
	assert(MBBI->getOpcode() == PPC32::BLR &&
		   "Can only insert epilog into returning blocks");
	
	// Get the number of bytes allocated from the FrameInfo...
    unsigned NumBytes = MFI->getStackSize();

    // adjust stack pointer back: r1 += numbytes
    if (NumBytes) {
		MI =BuildMI(PPC32::ADDI, 2, PPC32::R1)
		.addReg(PPC32::R1)
		.addSImm(NumBytes);
		MBB.insert(MBBI, MI);
    }
	
	// If we have calls, restore the LR value before we branch to it
    if (MFI->hasCalls()) {
		MI = BuildMI(PPC32::LWZ, 2, PPC32::R0).addSImm(8).addReg(PPC32::R1);
		MBB.insert(MBBI, MI);
		MI = BuildMI(PPC32::MTLR, 1).addReg(PPC32::R0);
		MBB.insert(MBBI, MI);
	}
}


#include "PowerPCGenRegisterInfo.inc"

const TargetRegisterClass*
PowerPCRegisterInfo::getRegClassForType(const Type* Ty) const {
	switch (Ty->getPrimitiveID()) {
		case Type::LongTyID:
		case Type::ULongTyID: assert(0 && "Long values can't fit in registers!");
		default:              assert(0 && "Invalid type to getClass!");
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

