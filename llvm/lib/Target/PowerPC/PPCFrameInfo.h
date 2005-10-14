//===-- PowerPCFrameInfo.h - Define TargetFrameInfo for PowerPC -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//----------------------------------------------------------------------------

#ifndef POWERPC_FRAMEINFO_H
#define POWERPC_FRAMEINFO_H

#include "PPC.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class PowerPCFrameInfo: public TargetFrameInfo {
  const TargetMachine &TM;
  std::pair<unsigned, int> LR[1];

public:
  PowerPCFrameInfo(const TargetMachine &tm, bool LP64)
    : TargetFrameInfo(TargetFrameInfo::StackGrowsDown, 16, 0), TM(tm) {
    LR[0].first = PPC::LR;
    LR[0].second = LP64 ? 16 : 8;
  }

  const std::pair<unsigned, int> *
  getCalleeSaveSpillSlots(unsigned &NumEntries) const {
    NumEntries = 1;
    return &LR[0];
  }
};

} // End llvm namespace

#endif
