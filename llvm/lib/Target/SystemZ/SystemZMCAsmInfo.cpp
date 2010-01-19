//===-- SystemZMCAsmInfo.cpp - SystemZ asm properties ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the SystemZMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "SystemZMCAsmInfo.h"
using namespace llvm;

SystemZMCAsmInfo::SystemZMCAsmInfo(const Target &T, const StringRef &TT)
: MCAsmInfo(false) {
  AlignmentIsInBytes = true;

  PrivateGlobalPrefix = ".L";
  WeakRefDirective = "\t.weak\t";
  SetDirective = "\t.set\t";
  PCSymbol = ".";

  NonexecutableStackDirective = "\t.section\t.note.GNU-stack,\"\",@progbits";
}
