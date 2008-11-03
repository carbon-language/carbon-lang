//===-- PIC16TargetAsmInfo.cpp - PIC16 asm properties ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the PIC16TargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "PIC16TargetAsmInfo.h"
#include "PIC16TargetMachine.h"

using namespace llvm;

PIC16TargetAsmInfo::
PIC16TargetAsmInfo(const PIC16TargetMachine &TM)
  : TargetAsmInfo(TM) {
  Data16bitsDirective = "\t.half\t";
  Data32bitsDirective = "\t.word\t";
  CommentString = ";";
  COMMDirective = "\t";
  COMMDirectiveTakesAlignment = 0;
}
