//===-- SparcMCAsmInfo.cpp - Sparc asm properties -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the SparcMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "SparcMCAsmInfo.h"
#include "llvm/ADT/Triple.h"

using namespace llvm;

void SparcELFMCAsmInfo::anchor() { }

SparcELFMCAsmInfo::SparcELFMCAsmInfo(StringRef TT) {
  IsLittleEndian = false;
  Triple TheTriple(TT);
  bool isV9 = (TheTriple.getArch() == Triple::sparcv9);

  if (isV9) {
    PointerSize = CalleeSaveStackSlotSize = 8;
  }

  Data16bitsDirective = "\t.half\t";
  Data32bitsDirective = "\t.word\t";
  // .xword is only supported by V9.
  Data64bitsDirective = (isV9) ? "\t.xword\t" : 0;
  ZeroDirective = "\t.skip\t";
  CommentString = "!";
  HasLEB128 = true;
  SupportsDebugInformation = true;

  ExceptionsType = ExceptionHandling::DwarfCFI;

  SunStyleELFSectionSwitchSyntax = true;
  UsesELFSectionDirectiveForBSS = true;

  PrivateGlobalPrefix = ".L";
}


