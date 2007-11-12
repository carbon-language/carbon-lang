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
#include "MipsTargetMachine.h"

using namespace llvm;

MipsTargetAsmInfo::MipsTargetAsmInfo(const MipsTargetMachine &TM) {
  AlignmentIsInBytes   = false;
  Data16bitsDirective  = "\t.half\t";
  Data32bitsDirective  = "\t.word\t";
  PrivateGlobalPrefix  = "$";
  JumpTableDataSection = "\t.rdata";
  CommentString        = "#";
  ReadOnlySection      = "\t.rdata";
  ZeroDirective        = "\t.space\t";
  BSSSection           = "\t.section\t.bss";
  GlobalDirective      = "\t.globl\t";
  LCOMMDirective       = "\t.lcomm\t";

  if (TM.getRelocationModel() == Reloc::Static)
    JumpTableDirective = "\t.word\t";
  else    
    JumpTableDirective = "\t.gpword\t";

  COMMDirectiveTakesAlignment = true;
}
