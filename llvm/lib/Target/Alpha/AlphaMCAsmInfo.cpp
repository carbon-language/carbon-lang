//===-- AlphaMCAsmInfo.cpp - Alpha asm properties ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the AlphaMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "AlphaMCAsmInfo.h"
using namespace llvm;

AlphaMCAsmInfo::AlphaMCAsmInfo(const Target &T, const StringRef &TT) {
  AlignmentIsInBytes = false;
  PrivateGlobalPrefix = "$";
  PICJumpTableDirective = ".gprel32";
  WeakRefDirective = "\t.weak\t";
}
