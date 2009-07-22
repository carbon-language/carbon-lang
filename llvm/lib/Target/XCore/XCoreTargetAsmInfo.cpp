//===-- XCoreTargetAsmInfo.cpp - XCore asm properties -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the XCoreTargetAsmInfo properties.
// We use the small section flag for the CP relative and DP relative
// flags. If a section is small and writable then it is DP relative. If a
// section is small and not writable then it is CP relative.
//
//===----------------------------------------------------------------------===//

#include "XCoreTargetAsmInfo.h"
#include "XCoreTargetMachine.h"
#include "llvm/GlobalVariable.h"
#include "llvm/ADT/StringExtras.h"

using namespace llvm;

XCoreTargetAsmInfo::XCoreTargetAsmInfo(const XCoreTargetMachine &TM)
  : ELFTargetAsmInfo(TM) {
  SupportsDebugInformation = true;
  TextSection = getUnnamedSection("\t.text", SectionFlags::Code);
  DataSection = getNamedSection("\t.dp.data", SectionFlags::Writeable |
                                SectionFlags::Small);
  BSSSection_  = getNamedSection("\t.dp.bss", SectionFlags::Writeable |
                                 SectionFlags::BSS | SectionFlags::Small);

  // TLS globals are lowered in the backend to arrays indexed by the current
  // thread id. After lowering they require no special handling by the linker
  // and can be placed in the standard data / bss sections.
  TLSDataSection = DataSection;
  TLSBSSSection = BSSSection_;

  if (TM.getSubtargetImpl()->isXS1A()) {
    ReadOnlySection = getNamedSection("\t.dp.rodata", SectionFlags::None |
                                      SectionFlags::Writeable |
                                      SectionFlags::Small);
  } else {
    ReadOnlySection = getNamedSection("\t.cp.rodata", SectionFlags::None |
                                      SectionFlags::Small);
  }
  Data16bitsDirective = "\t.short\t";
  Data32bitsDirective = "\t.long\t";
  Data64bitsDirective = 0;
  ZeroDirective = "\t.space\t";
  CommentString = "#";
  ConstantPoolSection = "\t.section\t.cp.rodata,\"ac\",@progbits";
  JumpTableDataSection = "\t.section\t.dp.data,\"awd\",@progbits";
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

