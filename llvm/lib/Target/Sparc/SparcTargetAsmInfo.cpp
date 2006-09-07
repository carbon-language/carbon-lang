//===-- SparcTargetAsmInfo.cpp - Sparc asm properties -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the SparcTargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "SparcTargetAsmInfo.h"

using namespace llvm;

SparcTargetAsmInfo::SparcTargetAsmInfo(const SparcTargetMachine &TM) {
  Data16bitsDirective = "\t.half\t";
  Data32bitsDirective = "\t.word\t";
  Data64bitsDirective = 0;  // .xword is only supported by V9.
  ZeroDirective = "\t.skip\t";
  CommentString = "!";
  ConstantPoolSection = "\t.section \".rodata\",#alloc\n";
}
