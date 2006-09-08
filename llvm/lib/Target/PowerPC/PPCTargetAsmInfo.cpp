//===-- PPCTargetAsmInfo.cpp - PPC asm properties ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the DarwinTargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "PPCTargetAsmInfo.h"
#include "PPCTargetMachine.h"

using namespace llvm;

DarwinTargetAsmInfo::DarwinTargetAsmInfo(const PPCTargetMachine &TM) {
  bool isPPC64 = TM.getSubtargetImpl()->isPPC64();

  CommentString = ";";
  GlobalPrefix = "_";
  PrivateGlobalPrefix = "L";
  ZeroDirective = "\t.space\t";
  SetDirective = "\t.set";
  Data64bitsDirective = isPPC64 ? "\t.quad\t" : 0;  
  AlignmentIsInBytes = false;
  ConstantPoolSection = "\t.const\t";
  JumpTableDataSection = ".const";
  JumpTableTextSection = "\t.text";
  LCOMMDirective = "\t.lcomm\t";
  StaticCtorsSection = ".mod_init_func";
  StaticDtorsSection = ".mod_term_func";
  InlineAsmStart = "# InlineAsm Start";
  InlineAsmEnd = "# InlineAsm End";
  
  NeedsSet = true;
  AddressSize = isPPC64 ? 8 : 4;
  DwarfAbbrevSection = ".section __DWARF,__debug_abbrev";
  DwarfInfoSection = ".section __DWARF,__debug_info";
  DwarfLineSection = ".section __DWARF,__debug_line";
  DwarfFrameSection = ".section __DWARF,__debug_frame";
  DwarfPubNamesSection = ".section __DWARF,__debug_pubnames";
  DwarfPubTypesSection = ".section __DWARF,__debug_pubtypes";
  DwarfStrSection = ".section __DWARF,__debug_str";
  DwarfLocSection = ".section __DWARF,__debug_loc";
  DwarfARangesSection = ".section __DWARF,__debug_aranges";
  DwarfRangesSection = ".section __DWARF,__debug_ranges";
  DwarfMacInfoSection = ".section __DWARF,__debug_macinfo";
}
