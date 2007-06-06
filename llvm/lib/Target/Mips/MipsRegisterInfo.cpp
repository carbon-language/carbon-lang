//===- MipsRegisterInfo.cpp - MIPS Register Information -== -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Bruno Cardoso Lopes and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MIPS implementation of the MRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mips-reg-info"

#include "Mips.h"
#include "MipsRegisterInfo.h"
#include "llvm/Constants.h"
#include "llvm/Type.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
//#include "MipsSubtarget.h"

using namespace llvm;

// TODO: add subtarget support
MipsRegisterInfo::MipsRegisterInfo(const TargetInstrInfo &tii)
  : MipsGenRegisterInfo(Mips::ADJCALLSTACKDOWN, Mips::ADJCALLSTACKUP),
  TII(tii) {}

void MipsRegisterInfo::
storeRegToStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
          unsigned SrcReg, int FI, 
          const TargetRegisterClass *RC) const 
{
  if (RC == Mips::CPURegsRegisterClass)
    BuildMI(MBB, I, TII.get(Mips::SW)).addFrameIndex(FI)
          .addImm(0).addReg(SrcReg, false, false, true);
  else
    assert(0 && "Can't store this register to stack slot");
}

void MipsRegisterInfo::
loadRegFromStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                     unsigned DestReg, int FI,
                     const TargetRegisterClass *RC) const 
{
  if (RC == Mips::CPURegsRegisterClass)
    BuildMI(MBB, I, TII.get(Mips::LW), DestReg).addImm(0).addFrameIndex(FI);
  else
    assert(0 && "Can't load this register from stack slot");
}

void MipsRegisterInfo::
copyRegToReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
             unsigned DestReg, unsigned SrcReg,
             const TargetRegisterClass *RC) const 
{
  if (RC == Mips::CPURegsRegisterClass)
    BuildMI(MBB, I, TII.get(Mips::ADDu), DestReg).addReg(Mips::ZERO)
      .addReg(SrcReg);
  else
    assert (0 && "Can't copy this register");
}

void MipsRegisterInfo::reMaterialize(MachineBasicBlock &MBB, 
                                      MachineBasicBlock::iterator I,
                                      unsigned DestReg, 
                                      const MachineInstr *Orig) const 
{
    MachineInstr *MI = Orig->clone();
    MI->getOperand(0).setReg(DestReg);
    MBB.insert(I, MI);
}

MachineInstr *MipsRegisterInfo::
foldMemoryOperand(MachineInstr* MI, unsigned OpNum, int FI) const 
{
  MachineInstr *NewMI = NULL;

  switch (MI->getOpcode()) 
  {
    case Mips::ADDu:
      if ((MI->getOperand(0).isRegister()) &&
        (MI->getOperand(1).isRegister()) && 
        (MI->getOperand(1).getReg() == Mips::ZERO) &&
        (MI->getOperand(2).isRegister())) 
      {
        if (OpNum == 0)    // COPY -> STORE
          NewMI = BuildMI(TII.get(Mips::SW)).addFrameIndex(FI)
                  .addImm(0).addReg(MI->getOperand(2).getReg());
        else               // COPY -> LOAD
          NewMI = BuildMI(TII.get(Mips::LW), MI->getOperand(0)
                  .getReg()).addImm(0).addFrameIndex(FI);
      }
      break;
  }

  if (NewMI)
    NewMI->copyKillDeadInfo(MI);
  return NewMI;
}

/// Mips Callee Saved Registers
const unsigned* MipsRegisterInfo::
getCalleeSavedRegs() const 
{
  // Mips calle-save register range is $16-$26(s0-s7)
  static const unsigned CalleeSavedRegs[] = {  
    Mips::S0, Mips::S1, Mips::S2, Mips::S3, 
    Mips::S4, Mips::S5, Mips::S6, Mips::S7, 0
  };
  return CalleeSavedRegs;
}

/// Mips Callee Saved Register Classes
const TargetRegisterClass* const* 
MipsRegisterInfo::getCalleeSavedRegClasses() const 
{
  static const TargetRegisterClass * const CalleeSavedRegClasses[] = {
    &Mips::CPURegsRegClass, &Mips::CPURegsRegClass,
    &Mips::CPURegsRegClass, &Mips::CPURegsRegClass,
    &Mips::CPURegsRegClass, &Mips::CPURegsRegClass,
    &Mips::CPURegsRegClass, &Mips::CPURegsRegClass, 0 
  };
  return CalleeSavedRegClasses;
}

BitVector MipsRegisterInfo::
getReservedRegs(const MachineFunction &MF) const
{
  BitVector Reserved(getNumRegs());
  Reserved.set(Mips::ZERO);
  Reserved.set(Mips::AT);
  Reserved.set(Mips::K0);
  Reserved.set(Mips::K1);
  Reserved.set(Mips::GP);
  Reserved.set(Mips::SP);
  Reserved.set(Mips::FP);
  Reserved.set(Mips::RA);
  return Reserved;
}

//===----------------------------------------------------------------------===//
// Stack Frame Processing methods
//===----------------------------------------------------------------------===//

// True if target has frame pointer
bool MipsRegisterInfo::
hasFP(const MachineFunction &MF) const {
  return false;
}

// This function eliminate ADJCALLSTACKDOWN, 
// ADJCALLSTACKUP pseudo instructions
void MipsRegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  // Simply discard ADJCALLSTACKDOWN, ADJCALLSTACKUP instructions.
  MBB.erase(I);
}

// FrameIndex represent objects inside a abstract stack.
// We must replace FrameIndex with an stack/frame pointer
// direct reference.
void MipsRegisterInfo::
eliminateFrameIndex(MachineBasicBlock::iterator II, int SPAdj, 
                    RegScavenger *RS) const 
{
  unsigned i = 0;
  MachineInstr &MI    = *II;
  MachineFunction &MF = *MI.getParent()->getParent();

  while (!MI.getOperand(i).isFrameIndex()) {
    ++i;
    assert(i < MI.getNumOperands() && 
           "Instr doesn't have FrameIndex operand!");
  }

  // FrameInfo addressable stack objects are accessed 
  // using neg. offsets, so we must add with the stack
  // size to obtain $sp relative address.
  int FrameIndex = MI.getOperand(i).getFrameIndex();
  int stackSize  = MF.getFrameInfo()->getStackSize();
  int spOffset   = MF.getFrameInfo()->getObjectOffset(FrameIndex);

  #ifndef NDEBUG
  DOUT << "\n<--------->\n";
  MI.print(DOUT);
  DOUT << "FrameIndex : " << FrameIndex << "\n";
  DOUT << "spOffset   : " << spOffset << "\n";
  DOUT << "stackSize  : " << stackSize << "\n";
  #endif

  // If the FrameIndex points to a positive SPOffset this
  // means we are inside the callee and getting the arguments 
  // from the caller stack
  int Offset = (-(stackSize)) + spOffset; 

  #ifndef NDEBUG
  DOUT << "Offset     : " << Offset << "\n";
  DOUT << "<--------->\n";
  #endif

  MI.getOperand(i-1).ChangeToImmediate(Offset);
  MI.getOperand(i).ChangeToRegister(Mips::SP,false);
}

void MipsRegisterInfo::
emitPrologue(MachineFunction &MF) const 
{
  MachineBasicBlock &MBB = MF.front();
  MachineFrameInfo *MFI  = MF.getFrameInfo();

  // Get the number of bytes to allocate from the FrameInfo
  int NumBytes = (int) MFI->getStackSize();

  // Do we need to allocate space on the stack?
  if (NumBytes == 0) return;

  // FIXME: is Stack Align needed here ?? (maybe it's done before...)
  unsigned Align = MF.getTarget().getFrameInfo()->getStackAlignment();
  NumBytes = -((NumBytes+Align-1)/Align*Align);

  // Update frame info to pretend that this is part of the stack...
  MFI->setStackSize(NumBytes);

  // adjust stack : addi sp, sp, (-imm)
  BuildMI(MBB, MBB.begin(), TII.get(Mips::ADDi), Mips::SP)
      .addReg(Mips::SP).addImm(NumBytes);
}

void MipsRegisterInfo::
emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const 
{
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  MachineFrameInfo *MFI = MF.getFrameInfo();

  // Get the number of bytes from FrameInfo
  int NumBytes = (int) MFI->getStackSize();

  // adjust stack  : insert addi sp, sp, (imm)
  if (NumBytes) {
    BuildMI(MBB, MBBI, TII.get(Mips::ADDi), Mips::SP)
      .addReg(Mips::SP).addImm(-NumBytes);
  }
}

void MipsRegisterInfo::
processFunctionBeforeFrameFinalized(MachineFunction &MF) const {}

unsigned MipsRegisterInfo::
getRARegister() const {
  return Mips::RA;
}

unsigned MipsRegisterInfo::
getFrameRegister(MachineFunction &MF) const {
  assert(0 && "What is the frame register");
  return Mips::FP;
}

unsigned MipsRegisterInfo::
getEHExceptionRegister() const {
  assert(0 && "What is the exception register");
  return 0;
}

unsigned MipsRegisterInfo::
getEHHandlerRegister() const {
  assert(0 && "What is the exception handler register");
  return 0;
}

#include "MipsGenRegisterInfo.inc"

