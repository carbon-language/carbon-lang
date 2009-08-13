//===-- PPCTargetAsmInfo.cpp - PPC asm properties ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the DarwinTargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "PPCTargetAsmInfo.h"
#include "llvm/ADT/Triple.h"
using namespace llvm;

PPCDarwinTargetAsmInfo::PPCDarwinTargetAsmInfo(const Triple &TheTriple) 
  : DarwinTargetAsmInfo(TheTriple) {
  PCSymbol = ".";
  CommentString = ";";
  ExceptionsType = ExceptionHandling::Dwarf;

  if (TheTriple.getArch() != Triple::ppc64)
    Data64bitsDirective = 0;      // We can't emit a 64-bit unit in PPC32 mode.
  AssemblerDialect = 1;           // New-Style mnemonics.
}

PPCLinuxTargetAsmInfo::PPCLinuxTargetAsmInfo(const Triple &TheTriple) {
  CommentString = "#";
  GlobalPrefix = "";
  PrivateGlobalPrefix = ".L";
  UsedDirective = "\t# .no_dead_strip\t";
  WeakRefDirective = "\t.weak\t";

  // Debug Information
  AbsoluteDebugSectionOffsets = true;
  SupportsDebugInformation = true;

  PCSymbol = ".";

  // Set up DWARF directives
  HasLEB128 = true;  // Target asm supports leb128 directives (little-endian)

  // Exceptions handling
  if (TheTriple.getArch() != Triple::ppc64) {
    ExceptionsType = ExceptionHandling::Dwarf;
    Data64bitsDirective = 0;
  }
  AbsoluteEHSectionOffsets = false;
    
  ZeroDirective = "\t.space\t";
  SetDirective = "\t.set";
  
  AlignmentIsInBytes = false;
  LCOMMDirective = "\t.lcomm\t";
  AssemblerDialect = 0;           // Old-Style mnemonics.
}

