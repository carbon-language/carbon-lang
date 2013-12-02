//===-- XCoreMachineFuctionInfo.cpp - XCore machine function info ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "XCoreMachineFunctionInfo.h"

using namespace llvm;

void XCoreFunctionInfo::anchor() { }

bool XCoreFunctionInfo::isLargeFrame(const MachineFunction &MF) const {
  if (CachedEStackSize == -1) {
    CachedEStackSize = MF.getFrameInfo()->estimateStackSize(MF);
  }
  // isLargeFrame() is used when deciding if spill slots should be added to
  // allow eliminateFrameIndex() to scavenge registers.
  // This is only required when there is no FP and offsets are greater than
  // ~256KB (~64Kwords). Thus only for code run on the emulator!
  //
  // The arbitrary value of 0xf000 allows frames of up to ~240KB before spill
  // slots are added for the use of eliminateFrameIndex() register scavenging.
  // For frames less than 240KB, it is assumed that there will be less than
  // 16KB of function arguments.
  return CachedEStackSize > 0xf000;
}
