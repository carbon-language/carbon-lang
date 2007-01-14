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
#include "llvm/Function.h"
using namespace llvm;

PPCTargetAsmInfo::PPCTargetAsmInfo(const PPCTargetMachine &TM) {
  bool isPPC64 = TM.getSubtargetImpl()->isPPC64();
  
  ZeroDirective = "\t.space\t";
  SetDirective = "\t.set";
  Data64bitsDirective = isPPC64 ? "\t.quad\t" : 0;  
  AlignmentIsInBytes = false;
  LCOMMDirective = "\t.lcomm\t";
  InlineAsmStart = "# InlineAsm Start";
  InlineAsmEnd = "# InlineAsm End";
  
  NeedsSet = true;
  AddressSize = isPPC64 ? 8 : 4;
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

DarwinTargetAsmInfo::DarwinTargetAsmInfo(const PPCTargetMachine &TM)
: PPCTargetAsmInfo(TM)
{
  CommentString = ";";
  GlobalPrefix = "_";
  PrivateGlobalPrefix = "L";
  ConstantPoolSection = "\t.const\t";
  JumpTableDataSection = ".const";
  CStringSection = "\t.cstring";
  StaticCtorsSection = ".mod_init_func";
  StaticDtorsSection = ".mod_term_func";
  UsedDirective = "\t.no_dead_strip\t";
  WeakRefDirective = "\t.weak_reference\t";
  HiddenDirective = "\t.private_extern\t";
}

LinuxTargetAsmInfo::LinuxTargetAsmInfo(const PPCTargetMachine &TM)
: PPCTargetAsmInfo(TM)
{
  CommentString = "#";
  GlobalPrefix = "";
  PrivateGlobalPrefix = "";
  ConstantPoolSection = "\t.section .rodata.cst4\t";
  JumpTableDataSection = ".section .rodata.cst4";
  CStringSection = "\t.section\t.rodata";
  StaticCtorsSection = ".section\t.ctors,\"aw\",@progbits";
  StaticDtorsSection = ".section\t.dtors,\"aw\",@progbits";
  UsedDirective = "\t# .no_dead_strip\t";
  WeakRefDirective = "\t.weak\t";
}
