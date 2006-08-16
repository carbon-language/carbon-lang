//===-- ARMTargetFrameInfo.h - Define TargetFrameInfo for ARM ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the "Instituto Nokia de Tecnologia" and
// is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//
//===----------------------------------------------------------------------===//

#ifndef ARM_FRAMEINFO_H
#define ARM_FRAMEINFO_H

#include "ARM.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class ARMFrameInfo: public TargetFrameInfo {
  std::pair<unsigned, int> LR[1];

public:
  ARMFrameInfo()
    : TargetFrameInfo(TargetFrameInfo::StackGrowsDown, 8, 0) {
    LR[0].first = ARM::R14;
    LR[0].second = -4;
  }

  const std::pair<unsigned, int> *
  getCalleeSaveSpillSlots(unsigned &NumEntries) const {
    NumEntries = 1;
    return &LR[0];
  }
};

} // End llvm namespace

#endif
