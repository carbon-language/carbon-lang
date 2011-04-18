//===- ARMRegisterInfo.cpp - ARM Register Information -----------*- C++ -*-===//
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

#include "ARM.h"
#include "ARMBaseInstrInfo.h"
#include "ARMRegisterInfo.h"
using namespace llvm;

ARMRegisterInfo::ARMRegisterInfo(const ARMBaseInstrInfo &tii,
                                 const ARMSubtarget &sti)
  : ARMBaseRegisterInfo(tii, sti) {
}
