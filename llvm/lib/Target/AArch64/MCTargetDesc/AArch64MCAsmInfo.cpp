//===-- AArch64MCAsmInfo.cpp - AArch64 asm properties ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the AArch64MCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "AArch64MCAsmInfo.h"

using namespace llvm;

AArch64ELFMCAsmInfo::AArch64ELFMCAsmInfo() {
  PointerSize = 8;

  // ".comm align is in bytes but .align is pow-2."
  AlignmentIsInBytes = false;

  CommentString = "//";
  Code32Directive = ".code\t32";

  Data16bitsDirective = "\t.hword\t";
  Data32bitsDirective = "\t.word\t";
  Data64bitsDirective = "\t.xword\t";

  UseDataRegionDirectives = true;

  HasLEB128 = true;
  SupportsDebugInformation = true;

  // Exceptions handling
  ExceptionsType = ExceptionHandling::DwarfCFI;
}

// Pin the vtable to this file.
void AArch64ELFMCAsmInfo::anchor() {}
