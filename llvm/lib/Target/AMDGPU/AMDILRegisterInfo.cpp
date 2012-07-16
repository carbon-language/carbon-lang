//===- AMDILRegisterInfo.cpp - AMDIL Register Information -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the AMDIL implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "AMDILRegisterInfo.h"
#include "AMDIL.h"
#include "AMDILInstrInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

AMDILRegisterInfo::AMDILRegisterInfo(TargetMachine &tm,
    const TargetInstrInfo &tii)
: AMDILGenRegisterInfo(0), // RA???
  TM(tm), TII(tii)
{
  baseOffset = 0;
  nextFuncOffset = 0;
}

const uint16_t*
AMDILRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const
{
  static const uint16_t CalleeSavedRegs[] = { 0 };
  // TODO: Does IL need to actually have any callee saved regs?
  // I don't think we do since we can just use sequential registers
  // Maybe this would be easier if every function call was inlined first
  // and then there would be no callee issues to deal with
  //TODO(getCalleeSavedRegs);
  return CalleeSavedRegs;
}

BitVector
AMDILRegisterInfo::getReservedRegs(const MachineFunction &MF) const
{
  BitVector Reserved(getNumRegs());
  // We reserve the first getNumRegs() registers as they are the ones passed
  // in live-in/live-out
  // and therefor cannot be killed by the scheduler. This works around a bug
  // discovered
  // that was causing the linearscan register allocator to kill registers
  // inside of the
  // function that were also passed as LiveIn registers.
  for (unsigned int x = 0, y = 256; x < y; ++x) {
    Reserved.set(x);
  }
  return Reserved;
}

BitVector
AMDILRegisterInfo::getAllocatableSet(const MachineFunction &MF,
    const TargetRegisterClass *RC = NULL) const
{
  BitVector Allocatable(getNumRegs());
  Allocatable.clear();
  return Allocatable;
}

const TargetRegisterClass* const*
AMDILRegisterInfo::getCalleeSavedRegClasses(const MachineFunction *MF) const
{
  static const TargetRegisterClass * const CalleeSavedRegClasses[] = { 0 };
  // TODO: Keep in sync with getCalleeSavedRegs
  //TODO(getCalleeSavedRegClasses);
  return CalleeSavedRegClasses;
}
void
AMDILRegisterInfo::eliminateCallFramePseudoInstr(
    MachineFunction &MF,
    MachineBasicBlock &MBB,
    MachineBasicBlock::iterator I) const
{
  MBB.erase(I);
}

// For each frame index we find, we store the offset in the stack which is
// being pushed back into the global buffer. The offset into the stack where
// the value is stored is copied into a new register and the frame index is
// then replaced with that register.
void 
AMDILRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
    int SPAdj,
    RegScavenger *RS) const
{
  assert(!"Implement");
}

void
AMDILRegisterInfo::processFunctionBeforeFrameFinalized(
    MachineFunction &MF) const
{
  //TODO(processFunctionBeforeFrameFinalized);
  // Here we keep track of the amount of stack that the current function
  // uses so
  // that we can set the offset to the end of the stack and any other
  // function call
  // will not overwrite any stack variables.
  // baseOffset = nextFuncOffset;
  MachineFrameInfo *MFI = MF.getFrameInfo();

  for (uint32_t x = 0, y = MFI->getNumObjects(); x < y; ++x) {
    int64_t size = MFI->getObjectSize(x);
    if (!(size % 4) && size > 1) {
      nextFuncOffset += size;
    } else {
      nextFuncOffset += 16;
    }
  }
}
unsigned int
AMDILRegisterInfo::getRARegister() const
{
  return AMDGPU::RA;
}

unsigned int
AMDILRegisterInfo::getFrameRegister(const MachineFunction &MF) const
{
  return AMDGPU::FP;
}

unsigned int
AMDILRegisterInfo::getEHExceptionRegister() const
{
  assert(0 && "What is the exception register");
  return 0;
}

unsigned int
AMDILRegisterInfo::getEHHandlerRegister() const
{
  assert(0 && "What is the exception handler register");
  return 0;
}

int64_t
AMDILRegisterInfo::getStackSize() const
{
  return nextFuncOffset - baseOffset;
}

#define GET_REGINFO_TARGET_DESC
#include "AMDGPUGenRegisterInfo.inc"

