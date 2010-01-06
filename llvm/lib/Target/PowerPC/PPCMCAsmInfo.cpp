//===-- PPCMCAsmInfo.cpp - PPC asm properties -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the MCAsmInfoDarwin properties.
//
//===----------------------------------------------------------------------===//

#include "PPCMCAsmInfo.h"
using namespace llvm;

PPCMCAsmInfoDarwin::PPCMCAsmInfoDarwin(bool is64Bit) {
  PCSymbol = ".";
  CommentString = ";";
  ExceptionsType = ExceptionHandling::Dwarf;

  if (!is64Bit)
    Data64bitsDirective = 0;      // We can't emit a 64-bit unit in PPC32 mode.
  AssemblerDialect = 1;           // New-Style mnemonics.
  SupportsDebugInformation= true; // Debug information.
}

PPCLinuxMCAsmInfo::PPCLinuxMCAsmInfo(bool is64Bit) {
  CommentString = "#";
  GlobalPrefix = "";
  PrivateGlobalPrefix = ".L";
  UsedDirective = "\t# .no_dead_strip\t";
  WeakRefDirective = "\t.weak\t";
  
  // Uses '.section' before '.bss' directive
  UsesELFSectionDirectiveForBSS = true;  

  // Debug Information
  AbsoluteDebugSectionOffsets = true;
  SupportsDebugInformation = true;

  PCSymbol = ".";

  // Set up DWARF directives
  HasLEB128 = true;  // Target asm supports leb128 directives (little-endian)

  // Exceptions handling
  if (!is64Bit)
    ExceptionsType = ExceptionHandling::Dwarf;
  AbsoluteEHSectionOffsets = false;
    
  ZeroDirective = "\t.space\t";
  SetDirective = "\t.set";
  Data64bitsDirective = is64Bit ? "\t.quad\t" : 0;
  AlignmentIsInBytes = false;
  LCOMMDirective = "\t.lcomm\t";
  AssemblerDialect = 0;           // Old-Style mnemonics.
}

