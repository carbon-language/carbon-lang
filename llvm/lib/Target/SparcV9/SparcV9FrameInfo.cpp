//===-- SparcV9FrameInfo.cpp - Stack frame layout info for SparcV9 --------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
// Interface to stack frame layout info for the UltraSPARC.  Starting offsets
// for each area of the stack frame are aligned at a multiple of
// getStackFrameSizeAlignment().
// 
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionInfo.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "SparcV9FrameInfo.h"

using namespace llvm;

int
SparcV9FrameInfo::getFirstAutomaticVarOffset(MachineFunction&, bool& pos) const {
  pos = false;                          // static stack area grows downwards
  return StaticAreaOffsetFromFP;
}

int
SparcV9FrameInfo::getRegSpillAreaOffset(MachineFunction& mcInfo, bool& pos) const 
{
  // ensure no more auto vars are added
  mcInfo.getInfo()->freezeAutomaticVarsArea();
  
  pos = false;                          // static stack area grows downwards
  unsigned autoVarsSize = mcInfo.getInfo()->getAutomaticVarsSize();
  return StaticAreaOffsetFromFP - autoVarsSize; 
}

int SparcV9FrameInfo::getTmpAreaOffset(MachineFunction& mcInfo, bool& pos) const {
  MachineFunctionInfo *MFI = mcInfo.getInfo();
  MFI->freezeAutomaticVarsArea();     // ensure no more auto vars are added
  MFI->freezeSpillsArea();            // ensure no more spill slots are added
  
  pos = false;                          // static stack area grows downwards
  unsigned autoVarsSize = MFI->getAutomaticVarsSize();
  unsigned spillAreaSize = MFI->getRegSpillsSize();
  int offset = autoVarsSize + spillAreaSize;
  return StaticAreaOffsetFromFP - offset;
}

int
SparcV9FrameInfo::getDynamicAreaOffset(MachineFunction& mcInfo, bool& pos) const {
  // Dynamic stack area grows downwards starting at top of opt-args area.
  // The opt-args, required-args, and register-save areas are empty except
  // during calls and traps, so they are shifted downwards on each
  // dynamic-size alloca.
  pos = false;
  unsigned optArgsSize = mcInfo.getInfo()->getMaxOptionalArgsSize();
  if (int extra = optArgsSize % getStackFrameSizeAlignment())
    optArgsSize += (getStackFrameSizeAlignment() - extra);
  int offset = optArgsSize + FirstOptionalOutgoingArgOffsetFromSP;
  assert((offset - OFFSET) % getStackFrameSizeAlignment() == 0);
  return offset;
}
