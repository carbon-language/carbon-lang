//===-- MCTargetDesc/AMDGPUMCAsmInfo.cpp - TODO: Add brief description -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// TODO: Add full description
//
//===----------------------------------------------------------------------===//

#include "AMDGPUMCAsmInfo.h"
#ifndef NULL
#define NULL 0
#endif

using namespace llvm;
AMDGPUMCAsmInfo::AMDGPUMCAsmInfo(const Target &T, StringRef &TT) : MCAsmInfo()
{
  //===------------------------------------------------------------------===//
  HasSubsectionsViaSymbols = true;
  HasMachoZeroFillDirective = false;
  HasMachoTBSSDirective = false;
  HasStaticCtorDtorReferenceInStaticMode = false;
  LinkerRequiresNonEmptyDwarfLines = true;
  MaxInstLength = 16;
  PCSymbol = "$";
  SeparatorString = "\n";
  CommentColumn = 40;
  CommentString = ";";
  LabelSuffix = ":";
  GlobalPrefix = "@";
  PrivateGlobalPrefix = ";.";
  LinkerPrivateGlobalPrefix = "!";
  InlineAsmStart = ";#ASMSTART";
  InlineAsmEnd = ";#ASMEND";
  AssemblerDialect = 0;
  AllowQuotesInName = false;
  AllowNameToStartWithDigit = false;
  AllowPeriodsInName = false;

  //===--- Data Emission Directives -------------------------------------===//
  ZeroDirective = ".zero";
  AsciiDirective = ".ascii\t";
  AscizDirective = ".asciz\t";
  Data8bitsDirective = ".byte\t";
  Data16bitsDirective = ".short\t";
  Data32bitsDirective = ".long\t";
  Data64bitsDirective = ".quad\t";
  GPRel32Directive = NULL;
  SunStyleELFSectionSwitchSyntax = true;
  UsesELFSectionDirectiveForBSS = true;
  HasMicrosoftFastStdCallMangling = false;

  //===--- Alignment Information ----------------------------------------===//
  AlignDirective = ".align\t";
  AlignmentIsInBytes = true;
  TextAlignFillValue = 0;

  //===--- Global Variable Emission Directives --------------------------===//
  GlobalDirective = ".global";
  ExternDirective = ".extern";
  HasSetDirective = false;
  HasAggressiveSymbolFolding = true;
  LCOMMDirectiveType = LCOMM::None;
  COMMDirectiveAlignmentIsInBytes = false;
  HasDotTypeDotSizeDirective = false;
  HasSingleParameterDotFile = true;
  HasNoDeadStrip = true;
  HasSymbolResolver = false;
  WeakRefDirective = ".weakref\t";
  WeakDefDirective = ".weakdef\t";
  LinkOnceDirective = NULL;
  HiddenVisibilityAttr = MCSA_Hidden;
  HiddenDeclarationVisibilityAttr = MCSA_Hidden;
  ProtectedVisibilityAttr = MCSA_Protected;

  //===--- Dwarf Emission Directives -----------------------------------===//
  HasLEB128 = true;
  SupportsDebugInformation = true;
  ExceptionsType = ExceptionHandling::None;
  DwarfUsesInlineInfoSection = false;
  DwarfSectionOffsetDirective = ".offset";

}
const char*
AMDGPUMCAsmInfo::getDataASDirective(unsigned int Size, unsigned int AS) const
{
  switch (AS) {
    default:
      return NULL;
    case 0:
      return NULL;
  };
  return NULL;
}

const MCSection*
AMDGPUMCAsmInfo::getNonexecutableStackSection(MCContext &CTX) const
{
  return NULL;
}
