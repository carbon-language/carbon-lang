//===-- MCAsmInfo.cpp - Asm Info -------------------------------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines target asm properties related what form asm statements
// should take.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAsmInfo.h"
#include "llvm/System/DataTypes.h"
#include <cctype>
#include <cstring>
using namespace llvm;

MCAsmInfo::MCAsmInfo() {
  HasSubsectionsViaSymbols = false;
  HasMachoZeroFillDirective = false;
  HasMachoTBSSDirective = false;
  HasStaticCtorDtorReferenceInStaticMode = false;
  MaxInstLength = 4;
  PCSymbol = "$";
  SeparatorChar = ';';
  CommentColumn = 40;
  CommentString = "#";
  GlobalPrefix = "";
  PrivateGlobalPrefix = ".";
  LinkerPrivateGlobalPrefix = "";
  InlineAsmStart = "APP";
  InlineAsmEnd = "NO_APP";
  AssemblerDialect = 0;
  AllowQuotesInName = false;
  AllowNameToStartWithDigit = false;
  AllowPeriodsInName = true;
  ZeroDirective = "\t.zero\t";
  AsciiDirective = "\t.ascii\t";
  AscizDirective = "\t.asciz\t";
  Data8bitsDirective = "\t.byte\t";
  Data16bitsDirective = "\t.short\t";
  Data32bitsDirective = "\t.long\t";
  Data64bitsDirective = "\t.quad\t";
  SunStyleELFSectionSwitchSyntax = false;
  UsesELFSectionDirectiveForBSS = false;
  AlignDirective = "\t.align\t";
  AlignmentIsInBytes = true;
  TextAlignFillValue = 0;
  GPRel32Directive = 0;
  GlobalDirective = "\t.globl\t";
  HasSetDirective = true;
  HasLCOMMDirective = false;
  COMMDirectiveAlignmentIsInBytes = true;
  HasDotTypeDotSizeDirective = true;
  HasSingleParameterDotFile = true;
  HasNoDeadStrip = false;
  WeakRefDirective = 0;
  WeakDefDirective = 0;
  LinkOnceDirective = 0;
  HiddenVisibilityAttr = MCSA_Hidden;
  ProtectedVisibilityAttr = MCSA_Protected;
  HasLEB128 = false;
  HasDotLocAndDotFile = false;
  SupportsDebugInformation = false;
  ExceptionsType = ExceptionHandling::None;
  DwarfRequiresFrameSection = true;
  DwarfUsesInlineInfoSection = false;
  DwarfSectionOffsetDirective = 0;
  HasMicrosoftFastStdCallMangling = false;

  AsmTransCBE = 0;
}

MCAsmInfo::~MCAsmInfo() {
}


unsigned MCAsmInfo::getULEB128Size(unsigned Value) {
  unsigned Size = 0;
  do {
    Value >>= 7;
    Size += sizeof(int8_t);
  } while (Value);
  return Size;
}

unsigned MCAsmInfo::getSLEB128Size(int Value) {
  unsigned Size = 0;
  int Sign = Value >> (8 * sizeof(Value) - 1);
  bool IsMore;

  do {
    unsigned Byte = Value & 0x7f;
    Value >>= 7;
    IsMore = Value != Sign || ((Byte ^ Sign) & 0x40) != 0;
    Size += sizeof(int8_t);
  } while (IsMore);
  return Size;
}
