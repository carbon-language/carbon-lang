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

PPCDarwinTargetAsmInfo::PPCDarwinTargetAsmInfo(const PPCTargetMachine &TM) :
  PPCTargetAsmInfo<DarwinTargetAsmInfo>(TM) {
  PCSymbol = ".";
  CommentString = ";";
  UsedDirective = "\t.no_dead_strip\t";
  ExceptionsType = ExceptionHandling::Dwarf;

  GlobalEHDirective = "\t.globl\t";
  SupportsWeakOmittedEHFrame = false;
}

PPCLinuxTargetAsmInfo::PPCLinuxTargetAsmInfo(const PPCTargetMachine &TM) :
  PPCTargetAsmInfo<TargetAsmInfo>(TM) {
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
  if (!TM.getSubtargetImpl()->isPPC64())
    ExceptionsType = ExceptionHandling::Dwarf;
  AbsoluteEHSectionOffsets = false;
}


// Instantiate default implementation.
TEMPLATE_INSTANTIATION(class PPCTargetAsmInfo<TargetAsmInfo>);
