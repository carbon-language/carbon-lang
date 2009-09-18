//===-- MSP430MCAsmInfo.cpp - MSP430 asm properties -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the MSP430MCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "MSP430MCAsmInfo.h"
using namespace llvm;

MSP430MCAsmInfo::MSP430MCAsmInfo(const Target &T, const StringRef &TT) {
  AlignmentIsInBytes = false;
  AllowNameToStartWithDigit = true;
}
