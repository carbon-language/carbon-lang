//===-- COFFTargetAsmInfo.cpp - COFF asm properties -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines target asm properties related what form asm statements
// should take in general on COFF-based targets
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/COFFTargetAsmInfo.h"
#include "llvm/ADT/SmallVector.h"
using namespace llvm;

COFFTargetAsmInfo::COFFTargetAsmInfo() {
  GlobalPrefix = "_";
  LCOMMDirective = "\t.lcomm\t";
  COMMDirectiveTakesAlignment = false;
  HasDotTypeDotSizeDirective = false;
  HasSingleParameterDotFile = false;
  HiddenDirective = NULL;
  PrivateGlobalPrefix = "L";  // Prefix for private global symbols
  WeakRefDirective = "\t.weak\t";
  SetDirective = "\t.set\t";

  // Set up DWARF directives
  HasLEB128 = true;  // Target asm supports leb128 directives (little-endian)
  AbsoluteDebugSectionOffsets = true;
  AbsoluteEHSectionOffsets = false;
  SupportsDebugInformation = true;
  DwarfSectionOffsetDirective = "\t.secrel32\t";
  DwarfAbbrevSection =  "\t.section\t.debug_abbrev,\"dr\"";
  DwarfInfoSection =    "\t.section\t.debug_info,\"dr\"";
  DwarfLineSection =    "\t.section\t.debug_line,\"dr\"";
  DwarfFrameSection =   "\t.section\t.debug_frame,\"dr\"";
  DwarfPubNamesSection ="\t.section\t.debug_pubnames,\"dr\"";
  DwarfPubTypesSection ="\t.section\t.debug_pubtypes,\"dr\"";
  DwarfStrSection =     "\t.section\t.debug_str,\"dr\"";
  DwarfLocSection =     "\t.section\t.debug_loc,\"dr\"";
  DwarfARangesSection = "\t.section\t.debug_aranges,\"dr\"";
  DwarfRangesSection =  "\t.section\t.debug_ranges,\"dr\"";
  DwarfMacroInfoSection = "\t.section\t.debug_macinfo,\"dr\"";
}

