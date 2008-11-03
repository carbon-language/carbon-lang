//===-- SPUTargetAsmInfo.cpp - Cell SPU asm properties ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the SPUTargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "SPUTargetAsmInfo.h"
#include "SPUTargetMachine.h"
#include "llvm/Function.h"
using namespace llvm;

SPUTargetAsmInfo::SPUTargetAsmInfo(const SPUTargetMachine &TM)
  : TargetAsmInfo(TM) {
  PCSymbol = ".";
  CommentString = "#";
  GlobalPrefix = "";
  PrivateGlobalPrefix = ".L";
  ZeroDirective = "\t.space\t";
  SetDirective = "\t.set";
  Data64bitsDirective = "\t.quad\t";  
  AlignmentIsInBytes = false;
  SwitchToSectionDirective = ".section\t";
  ConstantPoolSection = "\t.const\t";
  JumpTableDataSection = ".const";
  CStringSection = "\t.cstring";
  StaticCtorsSection = ".mod_init_func";
  StaticDtorsSection = ".mod_term_func";
  InlineAsmStart = "# InlineAsm Start";
  InlineAsmEnd = "# InlineAsm End";
  
  NeedsSet = true;
  /* FIXME: Need actual assembler syntax for DWARF info: */
  DwarfAbbrevSection = ".section __DWARF,__debug_abbrev,regular,debug";
  DwarfInfoSection = ".section __DWARF,__debug_info,regular,debug";
  DwarfLineSection = ".section __DWARF,__debug_line,regular,debug";
  DwarfFrameSection = ".section __DWARF,__debug_frame,regular,debug";
  DwarfPubNamesSection = ".section __DWARF,__debug_pubnames,regular,debug";
  DwarfPubTypesSection = ".section __DWARF,__debug_pubtypes,regular,debug";
  DwarfStrSection = ".section __DWARF,__debug_str,regular,debug";
  DwarfLocSection = ".section __DWARF,__debug_loc,regular,debug";
  DwarfARangesSection = ".section __DWARF,__debug_aranges,regular,debug";
  DwarfRangesSection = ".section __DWARF,__debug_ranges,regular,debug";
  DwarfMacInfoSection = ".section __DWARF,__debug_macinfo,regular,debug";
}
