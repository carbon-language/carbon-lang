//===- PIC16RegisterInfo.cpp - PIC16 Register Information -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PIC16 implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "pic16-reg-info"

#include "PIC16.h"
#include "PIC16RegisterInfo.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Type.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;

// FIXME: add subtarget support.
PIC16RegisterInfo::PIC16RegisterInfo(const TargetInstrInfo &tii)
  : PIC16GenRegisterInfo(PIC16::ADJCALLSTACKDOWN, PIC16::ADJCALLSTACKUP),
  TII(tii) {}

/// getRegisterNumbering - Given the enum value for some register, e.g.
/// PIC16::RA, return the number that it corresponds to (e.g. 31).
unsigned PIC16RegisterInfo::
getRegisterNumbering(unsigned RegEnum) 
{
  assert (RegEnum <= 31 && "Unknown register number!");
  return RegEnum;
}

void PIC16RegisterInfo::
copyRegToReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
             unsigned DestReg, unsigned SrcReg,
             const TargetRegisterClass *RC) const 
{
  return;
}

void PIC16RegisterInfo::reMaterialize(MachineBasicBlock &MBB, 
                                      MachineBasicBlock::iterator I,
                                      unsigned DestReg, 
                                      const MachineInstr *Orig) const 
{
  MachineInstr *MI = MBB.getParent()->CloneMachineInstr(Orig);
  MI->getOperand(0).setReg(DestReg);
  MBB.insert(I, MI);
}

MachineInstr *PIC16RegisterInfo::
foldMemoryOperand(MachineInstr* MI, unsigned OpNum, int FI) const 
{
  MachineInstr *NewMI = NULL;
  return NewMI;
}

//===----------------------------------------------------------------------===//
//
// Callee Saved Registers methods 
//
//===----------------------------------------------------------------------===//

/// PIC16 Callee Saved Registers
const unsigned* PIC16RegisterInfo::
getCalleeSavedRegs(const MachineFunction *MF) const 
{
  // PIC16 calle-save register range is $16-$26(s0-s7)
  static const unsigned CalleeSavedRegs[] = { 0 };
  return CalleeSavedRegs;
}

/// PIC16 Callee Saved Register Classes
const TargetRegisterClass* const* 
PIC16RegisterInfo::getCalleeSavedRegClasses(const MachineFunction *MF) const 
{
  static const TargetRegisterClass * const CalleeSavedRegClasses[] = { 0 };
  return CalleeSavedRegClasses;
}

BitVector PIC16RegisterInfo::
getReservedRegs(const MachineFunction &MF) const
{
  BitVector Reserved(getNumRegs());
  return Reserved;
}

//===----------------------------------------------------------------------===//
//
// Stack Frame Processing methods
// +----------------------------+
//
// FIXME: Add stack layout description here.
//
//
//===----------------------------------------------------------------------===//

// hasFP - Return true if the specified function should have a dedicated frame
// pointer register.  This is true if the function has variable sized allocas or
// if frame pointer elimination is disabled.
bool PIC16RegisterInfo::
hasFP(const MachineFunction &MF) const {
  return false;
}

// This function eliminate ADJCALLSTACKDOWN, 
// ADJCALLSTACKUP pseudo instructions
void PIC16RegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  // Simply discard ADJCALLSTACKDOWN, ADJCALLSTACKUP instructions.
  MBB.erase(I);
}

// FrameIndex represent objects inside a abstract stack.
// We must replace FrameIndex with an stack/frame pointer
// direct reference.
void PIC16RegisterInfo::
eliminateFrameIndex(MachineBasicBlock::iterator II, int SPAdj, 
                    RegScavenger *RS) const 
{
  MachineInstr &MI    = *II;
  MachineFunction &MF = *MI.getParent()->getParent();

  unsigned i = 0;
  while (!MI.getOperand(i).isFI()) {
    ++i;
    assert(i < MI.getNumOperands() && 
           "Instr doesn't have FrameIndex operand!");
  }

  int FrameIndex = MI.getOperand(i).getIndex();
  int stackSize  = MF.getFrameInfo()->getStackSize();
  int spOffset   = MF.getFrameInfo()->getObjectOffset(FrameIndex);

  DOUT << "\nFunction : " << MF.getFunction()->getName() << "\n";
  DOUT << "<--------->\n";
#ifndef NDEBUG
  MI.print(DOUT);
#endif
  DOUT << "FrameIndex : " << FrameIndex << "\n";
  DOUT << "spOffset   : " << spOffset << "\n";
  DOUT << "stackSize  : " << stackSize << "\n";

  // As explained on LowerFORMAL_ARGUMENTS, detect negative offsets 
  // and adjust SPOffsets considering the final stack size.
  int Offset = ((spOffset < 0) ? (stackSize + (-(spOffset+4))) : (spOffset));

  DOUT << "Offset     : " << Offset << "\n";
  DOUT << "<--------->\n";

  // MI.getOperand(i+1).ChangeToImmediate(Offset);
  MI.getOperand(i).ChangeToRegister(getFrameRegister(MF), false);
}

void PIC16RegisterInfo::
emitPrologue(MachineFunction &MF) const 
{
}

void PIC16RegisterInfo::
emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const 
{
}

void PIC16RegisterInfo::
processFunctionBeforeFrameFinalized(MachineFunction &MF) const 
{
}

unsigned PIC16RegisterInfo::
getRARegister() const {
  assert(0 && "What is the return address register");
  return 0;
}

unsigned PIC16RegisterInfo::
getFrameRegister(MachineFunction &MF) const {
  return PIC16::STKPTR;
}

unsigned PIC16RegisterInfo::
getEHExceptionRegister() const {
  assert(0 && "What is the exception register");
  return 0;
}

unsigned PIC16RegisterInfo::
getEHHandlerRegister() const {
  assert(0 && "What is the exception handler register");
  return 0;
}

int PIC16RegisterInfo::
getDwarfRegNum(unsigned RegNum, bool isEH) const {
  assert(0 && "What is the dwarf register number");
  return -1;
}


#include "PIC16GenRegisterInfo.inc"

