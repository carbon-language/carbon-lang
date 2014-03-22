//===-- ARMRegisterInfo.h - ARM Register Information Impl -------*- C++ -*-===//
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

#include "ARMBaseRegisterInfo.h"

namespace llvm {

class ARMSubtarget;

struct ARMRegisterInfo : public ARMBaseRegisterInfo {
  virtual void anchor();
public:
  ARMRegisterInfo(const ARMSubtarget &STI);
};

} // end namespace llvm

#endif
