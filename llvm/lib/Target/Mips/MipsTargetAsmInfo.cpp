//===-- MipsTargetAsmInfo.cpp - Mips asm properties -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Bruno Cardoso Lopes and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the MipsTargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "MipsTargetAsmInfo.h"

using namespace llvm;

MipsTargetAsmInfo::MipsTargetAsmInfo(const MipsTargetMachine &TM) {
  Data16bitsDirective = "\t.half\t";
  Data32bitsDirective = "\t.word\t";
  CommentString = "#";
}
