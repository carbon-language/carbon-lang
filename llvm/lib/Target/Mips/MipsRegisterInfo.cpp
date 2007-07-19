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
#include "MipsMachineFunction.h"
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
    BuildMI(MBB, I, TII.get(Mips::SW)).addReg(SrcReg, false, false, true)
          .addImm(0).addFrameIndex(FI);
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
getCalleeSavedRegs(const MachineFunction *MF) const 
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
MipsRegisterInfo::getCalleeSavedRegClasses(const MachineFunction *MF) const 
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
//
// Stack Frame Processing methods
// +----------------------------+
//
// Too meet ABI, we construct the frame on the reverse
// of natural order.
//
// The LLVM Frame will look like this:
//
// As the stack grows down, we start at 0, and the reference
// is decrement.
//
//  0          ----------
// -4          Args to pass
//  .          saved "Callee Saved" Registers
//  .          Local Area
//  .          saved FP
//  .          saved RA
// -StackSize  -----------
//
// On the EliminateFrameIndex we just negate the address above
// and we get the stack frame required by the ABI, which is:
//
// sp + stacksize  -------------
//                 saved $RA  (only on non-leaf functions)
//                 saved $FP  (only with frame pointer)
//                 saved "Callee Saved" Registers
//                 Local Area
//                 saved $GP  (used in PIC - not supported yet)
//                 Args to pass area
// sp              -------------
//
// The sp is the stack pointer subtracted/added from the stack size
// at the Prologue/Epilogue
//
// References to the previous stack (to obtain arguments) are done
// with fixed location stack frames using positive stack offsets.
//
// Examples:
// - reference to the actual stack frame
//   for any local area var there is smt like : FI >= 0, StackOffset: -4
//     sw REGX, 4(REGY)
//
// - reference to previous stack frame
//   suppose there's a store to the 5th arguments : FI < 0, StackOffset: 16.
//   The emitted instruction will be something like:
//     sw REGX, 16+StackSize (REGY)
//
//===----------------------------------------------------------------------===//

// hasFP - Return true if the specified function should have a dedicated frame
// pointer register.  This is true if the function has variable sized allocas or
// if frame pointer elimination is disabled.
bool MipsRegisterInfo::
hasFP(const MachineFunction &MF) const {
  return (NoFramePointerElim || MF.getFrameInfo()->hasVarSizedObjects());
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
  MachineInstr &MI    = *II;
  MachineFunction &MF = *MI.getParent()->getParent();

  unsigned i = 0;
  while (!MI.getOperand(i).isFrameIndex()) {
    ++i;
    assert(i < MI.getNumOperands() && 
           "Instr doesn't have FrameIndex operand!");
  }

  int FrameIndex = MI.getOperand(i).getFrameIndex();
  int stackSize  = MF.getFrameInfo()->getStackSize();
  int spOffset   = MF.getFrameInfo()->getObjectOffset(FrameIndex);

  #ifndef NDEBUG
  DOUT << "\nFunction : " << MF.getFunction()->getName() << "\n";
  DOUT << "<--------->\n";
  MI.print(DOUT);
  DOUT << "FrameIndex : " << FrameIndex << "\n";
  DOUT << "spOffset   : " << spOffset << "\n";
  DOUT << "stackSize  : " << stackSize << "\n";
  #endif

  int Offset = ( (spOffset >= 0) ? (stackSize + spOffset) : (-spOffset));

  #ifndef NDEBUG
  DOUT << "Offset     : " << Offset << "\n";
  DOUT << "<--------->\n";
  #endif

  MI.getOperand(i-1).ChangeToImmediate(Offset);
  MI.getOperand(i).ChangeToRegister(getFrameRegister(MF),false);
}

void MipsRegisterInfo::
emitPrologue(MachineFunction &MF) const 
{
  MachineBasicBlock &MBB   = MF.front();
  MachineFrameInfo *MFI    = MF.getFrameInfo();
  MipsFunctionInfo *MipsFI = MF.getInfo<MipsFunctionInfo>();
  MachineBasicBlock::iterator MBBI = MBB.begin();

  // Get the number of bytes to allocate from the FrameInfo
  int NumBytes = (int) MFI->getStackSize();

  #ifndef NDEBUG
  DOUT << "\n<--- EMIT PROLOGUE --->";
  DOUT << "Stack size :" << NumBytes << "\n";
  #endif

  // Do we need to allocate space on the stack?
  if (NumBytes == 0) return;

  int FPOffset, RAOffset;
  
  // Always allocate space for saved RA and FP,
  // even if FramePointer is not used. When not
  // using FP, the last stack slot becomes empty
  // and RA is saved before it.
  if ((hasFP(MF)) && (MFI->hasCalls())) {
    FPOffset = NumBytes;
    RAOffset = (NumBytes+4);
  } else if ((!hasFP(MF)) && (MFI->hasCalls())) {
    FPOffset = 0;
    RAOffset = NumBytes;
  } else if ((hasFP(MF)) && (!MFI->hasCalls())) {
    FPOffset = NumBytes;
    RAOffset = 0;
  }

  MFI->setObjectOffset(MFI->CreateStackObject(4,4), -FPOffset);
  MFI->setObjectOffset(MFI->CreateStackObject(4,4), -RAOffset);
  MipsFI->setFPStackOffset(FPOffset);
  MipsFI->setRAStackOffset(RAOffset);

  #ifndef NDEBUG
  DOUT << "FPOffset :" << FPOffset << "\n";
  DOUT << "RAOffset :" << RAOffset << "\n";
  #endif

  // Align stack. 
  NumBytes += 8;
  unsigned Align = MF.getTarget().getFrameInfo()->getStackAlignment();
  NumBytes = ((NumBytes+Align-1)/Align*Align);

  #ifndef NDEBUG
  DOUT << "New stack size :" << NumBytes << "\n\n";
  #endif

  // Update frame info
  MFI->setStackSize(NumBytes);

  // Adjust stack : addi sp, sp, (-imm)
  BuildMI(MBB, MBBI, TII.get(Mips::ADDiu), Mips::SP)
      .addReg(Mips::SP).addImm(-NumBytes);

  // Save the return address only if the function isnt a leaf one.
  // sw  $ra, stack_loc($sp)
  if (MFI->hasCalls()) { 
    BuildMI(MBB, MBBI, TII.get(Mips::SW))
        .addReg(Mips::RA).addImm(RAOffset).addReg(Mips::SP);
  }

  // if framepointer enabled, save it and set it
  // to point to the stack pointer
  if (hasFP(MF)) {
    // sw  $fp,stack_loc($sp)
    BuildMI(MBB, MBBI, TII.get(Mips::SW))
      .addReg(Mips::FP).addImm(FPOffset).addReg(Mips::SP);

    // move $fp, $sp
    BuildMI(MBB, MBBI, TII.get(Mips::ADDu), Mips::FP)
      .addReg(Mips::SP).addReg(Mips::ZERO);
  }
}

void MipsRegisterInfo::
emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const 
{
  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  MachineFrameInfo *MFI            = MF.getFrameInfo();
  MipsFunctionInfo *MipsFI         = MF.getInfo<MipsFunctionInfo>();

  // Get the number of bytes from FrameInfo
  int NumBytes = (int) MFI->getStackSize();

  // Get the FI's where RA and FP are saved.
  int FPOffset = MipsFI->getFPStackOffset();
  int RAOffset = MipsFI->getRAStackOffset();

  #ifndef NDEBUG
  DOUT << "\n<--- EMIT EPILOGUE --->" << "\n";
  DOUT << "Stack size :" << NumBytes << "\n";
  DOUT << "FPOffset :" << FPOffset << "\n";
  DOUT << "RAOffset :" << RAOffset << "\n\n";
  #endif

  // if framepointer enabled, restore it and restore the
  // stack pointer
  if (hasFP(MF)) {
    // move $sp, $fp
    BuildMI(MBB, MBBI, TII.get(Mips::ADDu), Mips::SP)
      .addReg(Mips::FP).addReg(Mips::ZERO);

    // lw  $fp,stack_loc($sp)
    BuildMI(MBB, MBBI, TII.get(Mips::LW))
      .addReg(Mips::FP).addImm(FPOffset).addReg(Mips::SP);
  }

  // Restore the return address only if the function isnt a leaf one.
  // lw  $ra, stack_loc($sp)
  if (MFI->hasCalls()) { 
    BuildMI(MBB, MBBI, TII.get(Mips::LW))
        .addReg(Mips::RA).addImm(RAOffset).addReg(Mips::SP);
  }

  // adjust stack  : insert addi sp, sp, (imm)
  if (NumBytes) {
    BuildMI(MBB, MBBI, TII.get(Mips::ADDiu), Mips::SP)
      .addReg(Mips::SP).addImm(NumBytes);
  }
}

void MipsRegisterInfo::
processFunctionBeforeFrameFinalized(MachineFunction &MF) const {
}

unsigned MipsRegisterInfo::
getRARegister() const {
  return Mips::RA;
}

unsigned MipsRegisterInfo::
getFrameRegister(MachineFunction &MF) const {
  return hasFP(MF) ? Mips::FP : Mips::SP;
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

