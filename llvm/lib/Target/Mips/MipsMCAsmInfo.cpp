//===-- MipsMCAsmInfo.cpp - Mips asm properties ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the MipsMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "MipsMCAsmInfo.h"
using namespace llvm;

MipsMCAsmInfo::MipsMCAsmInfo(const Target &T, const StringRef &TT,
                             bool isLittleEndian) : MCAsmInfo(isLittleEndian) {
  AlignmentIsInBytes          = false;
  COMMDirectiveTakesAlignment = true;
  Data16bitsDirective         = "\t.half\t";
  Data32bitsDirective         = "\t.word\t";
  Data64bitsDirective         = 0;
  PrivateGlobalPrefix         = "$";
  CommentString               = "#";
  ZeroDirective               = "\t.space\t";
  PICJumpTableDirective       = "\t.gpword\t";
}
