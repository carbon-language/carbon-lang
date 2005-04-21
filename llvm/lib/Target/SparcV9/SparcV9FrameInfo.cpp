//===-- SparcV9FrameInfo.cpp - Stack frame layout info for SparcV9 --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Interface to stack frame layout info for the UltraSPARC.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "MachineFunctionInfo.h"
#include "SparcV9FrameInfo.h"

using namespace llvm;

int
SparcV9FrameInfo::getRegSpillAreaOffset(MachineFunction& mcInfo, bool& pos) const
{
  // ensure no more auto vars are added
  mcInfo.getInfo<SparcV9FunctionInfo>()->freezeAutomaticVarsArea();

  pos = false;                          // static stack area grows downwards
  unsigned autoVarsSize = mcInfo.getInfo<SparcV9FunctionInfo>()->getAutomaticVarsSize();
  return StaticAreaOffsetFromFP - autoVarsSize;
}

int SparcV9FrameInfo::getTmpAreaOffset(MachineFunction& mcInfo, bool& pos) const {
  SparcV9FunctionInfo *MFI = mcInfo.getInfo<SparcV9FunctionInfo>();
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
  unsigned optArgsSize = mcInfo.getInfo<SparcV9FunctionInfo>()->getMaxOptionalArgsSize();
  if (int extra = optArgsSize % 16)
    optArgsSize += (16 - extra);
  int offset = optArgsSize + FirstOptionalOutgoingArgOffsetFromSP;
  assert((offset - OFFSET) % 16 == 0);
  return offset;
}
