//===-- SPUTargetAsmInfo.cpp - Cell SPU asm properties ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the SPUTargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "SPUTargetAsmInfo.h"
using namespace llvm;

SPULinuxTargetAsmInfo::SPULinuxTargetAsmInfo() {
  ZeroDirective = "\t.space\t";
  SetDirective = "\t.set";
  Data64bitsDirective = "\t.quad\t";
  AlignmentIsInBytes = false;
  LCOMMDirective = "\t.lcomm\t";
      
  PCSymbol = ".";
  CommentString = "#";
  GlobalPrefix = "";
  PrivateGlobalPrefix = ".L";

  // Has leb128, .loc and .file
  HasLEB128 = true;
  HasDotLocAndDotFile = true;

  SupportsDebugInformation = true;
  NeedsSet = true;

  // Exception handling is not supported on CellSPU (think about it: you only
  // have 256K for code+data. Would you support exception handling?)
  ExceptionsType = ExceptionHandling::None;
}

