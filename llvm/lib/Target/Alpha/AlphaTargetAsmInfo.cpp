//===-- AlphaTargetAsmInfo.cpp - Alpha asm properties -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the AlphaTargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "AlphaTargetMachine.h"
#include "AlphaTargetAsmInfo.h"

using namespace llvm;

AlphaTargetAsmInfo::AlphaTargetAsmInfo(const AlphaTargetMachine &TM) {
  AlignmentIsInBytes = false;
  PrivateGlobalPrefix = "$";
  JumpTableDirective = ".gprel32";
  WeakRefDirective = "\t.weak\t";
}
