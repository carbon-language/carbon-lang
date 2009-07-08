//===- ARMRegisterInfo.h - ARM Register Information Impl --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the ARM implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef ARMREGISTERINFO_H
#define ARMREGISTERINFO_H

#include "ARM.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "ARMBaseRegisterInfo.h"

namespace llvm {
  class ARMSubtarget;
  class ARMBaseInstrInfo;
  class Type;

struct ARMRegisterInfo : public ARMBaseRegisterInfo {
public:
  ARMRegisterInfo(const ARMBaseInstrInfo &tii, const ARMSubtarget &STI);
};

} // end namespace llvm

#endif
