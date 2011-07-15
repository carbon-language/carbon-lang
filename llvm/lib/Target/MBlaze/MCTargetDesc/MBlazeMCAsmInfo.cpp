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

MBlazeMCAsmInfo::MBlazeMCAsmInfo() {
  IsLittleEndian              = false;
  StackGrowsUp                = false;
  SupportsDebugInformation    = true;
  AlignmentIsInBytes          = false;
  PrivateGlobalPrefix         = "$";
  GPRel32Directive            = "\t.gpword\t";
}
