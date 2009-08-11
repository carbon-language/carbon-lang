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
#include "PPCTargetMachine.h"
#include "llvm/Function.h"
#include "llvm/Support/Dwarf.h"

using namespace llvm;
using namespace llvm::dwarf;

PPCDarwinTargetAsmInfo::PPCDarwinTargetAsmInfo(const PPCTargetMachine &TM) {
  PCSymbol = ".";
  CommentString = ";";
  ExceptionsType = ExceptionHandling::Dwarf;

  const PPCSubtarget *Subtarget = &TM.getSubtarget<PPCSubtarget>();
  bool isPPC64 = Subtarget->isPPC64();
  
  if (!isPPC64)
    Data64bitsDirective = 0;      // we can't emit a 64-bit unit
  InlineAsmStart = "# InlineAsm Start";
  InlineAsmEnd = "# InlineAsm End";
  AssemblerDialect = Subtarget->getAsmFlavor();
}

PPCLinuxTargetAsmInfo::PPCLinuxTargetAsmInfo(const PPCTargetMachine &TM) {
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

  const PPCSubtarget *Subtarget = &TM.getSubtarget<PPCSubtarget>();
  bool isPPC64 = Subtarget->isPPC64();

  // Exceptions handling
  if (!isPPC64)
    ExceptionsType = ExceptionHandling::Dwarf;
  AbsoluteEHSectionOffsets = false;
    
  ZeroDirective = "\t.space\t";
  SetDirective = "\t.set";
  Data64bitsDirective = isPPC64 ? "\t.quad\t" : 0;
  AlignmentIsInBytes = false;
  LCOMMDirective = "\t.lcomm\t";
  InlineAsmStart = "# InlineAsm Start";
  InlineAsmEnd = "# InlineAsm End";
  AssemblerDialect = Subtarget->getAsmFlavor();
}

