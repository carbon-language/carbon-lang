//===-- TargetFrameInfo.cpp - Implement machine frame interface -*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Implements the layout of a stack frame on the target machine.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetFrameInfo.h"
#include <cstdlib>

using namespace llvm;

//===--------------------------------------------------------------------===//
// These methods provide details of the stack frame used by Sparc, thus they
// are Sparc specific.
//===--------------------------------------------------------------------===//

// This method adjusts a stack offset to meet alignment rules of target.
int 
TargetFrameInfo::adjustAlignment(int unalignedOffset, bool growUp,
                                 unsigned align) const {
  abort();
  return 0;
}

// These methods compute offsets using the frame contents for a particular
// function.  The frame contents are obtained from the MachineFunction object
// for the given function.  The rest must be implemented by the
// machine-specific subclass.
// 
int
TargetFrameInfo::getIncomingArgOffset(MachineFunction& mcInfo, unsigned argNum)
  const {
  abort();
  return 0;
}

int
TargetFrameInfo::getOutgoingArgOffset(MachineFunction& mcInfo,
                                      unsigned argNum) const {
  abort();
  return 0;
}

int
TargetFrameInfo::getFirstAutomaticVarOffset(MachineFunction& mcInfo,
                                            bool& growUp) const {
  abort();
  return 0;
}

int 
TargetFrameInfo::getRegSpillAreaOffset(MachineFunction& mcInfo, bool& growUp)
  const {
  abort();
  return 0;
}

int
TargetFrameInfo::getTmpAreaOffset(MachineFunction& mcInfo, bool& growUp) const {
  abort();
  return 0;
}

int 
TargetFrameInfo::getDynamicAreaOffset(MachineFunction& mcInfo, bool& growUp)
  const {
  abort();
  return 0;
}

