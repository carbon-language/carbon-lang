//===- MSP430Subtarget.cpp - MSP430 Subtarget Information ---------*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MSP430 specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "MSP430Subtarget.h"
#include "MSP430.h"

#define GET_SUBTARGETINFO_CTOR
#define GET_SUBTARGETINFO_MC_DESC
#define GET_SUBTARGETINFO_TARGET_DESC
#include "MSP430GenSubtarget.inc"

using namespace llvm;

MSP430Subtarget::MSP430Subtarget(const std::string &TT,
                                 const std::string &CPUIgnored,
                                 const std::string &FS) {
  std::string CPU = "generic";

  // Parse features string.
  ParseSubtargetFeatures(FS, CPU);
}
