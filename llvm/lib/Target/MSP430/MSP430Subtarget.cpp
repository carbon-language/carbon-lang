//===-- MSP430Subtarget.cpp - MSP430 Subtarget Information ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MSP430 specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "MSP430Subtarget.h"
#include "MSP430.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define DEBUG_TYPE "msp430-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "MSP430GenSubtargetInfo.inc"

void MSP430Subtarget::anchor() { }

MSP430Subtarget &MSP430Subtarget::initializeSubtargetDependencies(StringRef CPU, StringRef FS) {
  ParseSubtargetFeatures("generic", FS);
  return *this;
}

MSP430Subtarget::MSP430Subtarget(const std::string &TT, const std::string &CPU,
                                 const std::string &FS, const TargetMachine &TM)
    : MSP430GenSubtargetInfo(TT, CPU, FS),
      // FIXME: Check DataLayout string.
      DL("e-m:e-p:16:16-i32:16:32-n8:16"), FrameLowering(),
      InstrInfo(initializeSubtargetDependencies(CPU, FS)), TLInfo(TM),
      TSInfo(DL) {}
