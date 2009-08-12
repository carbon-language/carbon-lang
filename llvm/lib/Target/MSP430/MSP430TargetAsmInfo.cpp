//===-- MSP430TargetAsmInfo.cpp - MSP430 asm properties -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the MSP430TargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "MSP430TargetAsmInfo.h"
using namespace llvm;

MSP430TargetAsmInfo::MSP430TargetAsmInfo(const Target &T, const StringRef &TT) {
  AlignmentIsInBytes = false;
}
