//===-- XCoreTargetAsmInfo.cpp - XCore asm properties ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "XCoreTargetAsmInfo.h"
using namespace llvm;

XCoreTargetAsmInfo::XCoreTargetAsmInfo(const TargetMachine &TM)
  : ELFTargetAsmInfo(TM) {
  SupportsDebugInformation = true;
  Data16bitsDirective = "\t.short\t";
  Data32bitsDirective = "\t.long\t";
  Data64bitsDirective = 0;
  ZeroDirective = "\t.space\t";
  CommentString = "#";
    
  PrivateGlobalPrefix = ".L";
  AscizDirective = ".asciiz";
  WeakDefDirective = "\t.weak\t";
  WeakRefDirective = "\t.weak\t";
  SetDirective = "\t.set\t";

  // Debug
  HasLEB128 = true;
  AbsoluteDebugSectionOffsets = true;
  
  DwarfAbbrevSection = "\t.section\t.debug_abbrev,\"\",@progbits";
  DwarfInfoSection = "\t.section\t.debug_info,\"\",@progbits";
  DwarfLineSection = "\t.section\t.debug_line,\"\",@progbits";
  DwarfFrameSection = "\t.section\t.debug_frame,\"\",@progbits";
  DwarfPubNamesSection = "\t.section\t.debug_pubnames,\"\",@progbits";
  DwarfPubTypesSection = "\t.section\t.debug_pubtypes,\"\",@progbits";
  DwarfStrSection = "\t.section\t.debug_str,\"\",@progbits";
  DwarfLocSection = "\t.section\t.debug_loc,\"\",@progbits";
  DwarfARangesSection = "\t.section\t.debug_aranges,\"\",@progbits";
  DwarfRangesSection = "\t.section\t.debug_ranges,\"\",@progbits";
  DwarfMacroInfoSection = "\t.section\t.debug_macinfo,\"\",@progbits";
}

