//===-- MBlazeMCAsmInfo.cpp - MBlaze asm properties -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the MBlazeMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "MBlazeMCAsmInfo.h"
using namespace llvm;

MBlazeMCAsmInfo::MBlazeMCAsmInfo(const Target &T, const StringRef &TT) {
  AlignmentIsInBytes          = false;
  Data16bitsDirective         = "\t.half\t";
  Data32bitsDirective         = "\t.word\t";
  Data64bitsDirective         = 0;
  PrivateGlobalPrefix         = "$";
  CommentString               = "#";
  ZeroDirective               = "\t.space\t";
  GPRel32Directive            = "\t.gpword\t";
  HasSetDirective             = false;
}
