//===-- SPUMCAsmInfo.cpp - Cell SPU asm properties ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the SPUMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "SPUMCAsmInfo.h"
using namespace llvm;

SPULinuxMCAsmInfo::SPULinuxMCAsmInfo(const Target &T, const StringRef &TT) {
  ZeroDirective = "\t.space\t";
  Data64bitsDirective = "\t.quad\t";
  AlignmentIsInBytes = false;
      
  PCSymbol = ".";
  CommentString = "#";
  GlobalPrefix = "";
  PrivateGlobalPrefix = ".L";

  // Has leb128, .loc and .file
  HasLEB128 = true;
  HasDotLocAndDotFile = true;

  SupportsDebugInformation = true;

  // Exception handling is not supported on CellSPU (think about it: you only
  // have 256K for code+data. Would you support exception handling?)
  ExceptionsType = ExceptionHandling::None;

  // SPU assembly requires ".section" before ".bss" 
  UsesELFSectionDirectiveForBSS = true;  
}

