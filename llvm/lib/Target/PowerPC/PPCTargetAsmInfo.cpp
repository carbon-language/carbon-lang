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
  AssemblerDialect = TM.getSubtargetImpl()->getAsmFlavor();
  
}

DarwinTargetAsmInfo::DarwinTargetAsmInfo(const PPCTargetMachine &TM)
: PPCTargetAsmInfo(TM)
{
  PCSymbol = ".";
  CommentString = ";";
  GlobalPrefix = "_";
  PrivateGlobalPrefix = "L";
  ConstantPoolSection = "\t.const\t";
  JumpTableDataSection = ".const";
  CStringSection = "\t.cstring";
  FourByteConstantSection = "\t.literal4\n";
  EightByteConstantSection = "\t.literal8\n";
  ReadOnlySection = "\t.const\n";
  if (TM.getRelocationModel() == Reloc::Static) {
    StaticCtorsSection = ".constructor";
    StaticDtorsSection = ".destructor";
  } else {
    StaticCtorsSection = ".mod_init_func";
    StaticDtorsSection = ".mod_term_func";
  }
  UsedDirective = "\t.no_dead_strip\t";
  WeakDefDirective = "\t.weak_definition\t";
  WeakRefDirective = "\t.weak_reference\t";
  HiddenDirective = "\t.private_extern\t";
  SupportsExceptionHandling = true;
  NeedsIndirectEncoding = true;
  NeedsSet = true;
  BSSSection = 0;
  
  DwarfEHFrameSection =
  ".section __TEXT,__eh_frame,coalesced,no_toc+strip_static_syms+live_support";
  DwarfExceptionSection = ".section __DATA,__gcc_except_tab";

  DwarfAbbrevSection = ".section __DWARF,__debug_abbrev,regular,debug";
  DwarfInfoSection = ".section __DWARF,__debug_info,regular,debug";
  DwarfLineSection = ".section __DWARF,__debug_line,regular,debug";
  DwarfFrameSection = ".section __DWARF,__debug_frame,regular,debug";
  GlobalEHDirective = "\t.globl\t";
  DwarfPubNamesSection = ".section __DWARF,__debug_pubnames,regular,debug";
  DwarfPubTypesSection = ".section __DWARF,__debug_pubtypes,regular,debug";
  DwarfStrSection = ".section __DWARF,__debug_str,regular,debug";
  DwarfLocSection = ".section __DWARF,__debug_loc,regular,debug";
  DwarfARangesSection = ".section __DWARF,__debug_aranges,regular,debug";
  DwarfRangesSection = ".section __DWARF,__debug_ranges,regular,debug";
  DwarfMacInfoSection = ".section __DWARF,__debug_macinfo,regular,debug";
  
  // In non-PIC modes, emit a special label before jump tables so that the
  // linker can perform more accurate dead code stripping.
  if (TM.getRelocationModel() != Reloc::PIC_) {
    // Emit a local label that is preserved until the linker runs.
    JumpTableSpecialLabelPrefix = "l";
  }
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
  BSSSection = "\t.section\t\".sbss\",\"aw\",@nobits";

  // Debug Information
  AbsoluteDebugSectionOffsets = true;
  SupportsDebugInformation = true;
  DwarfAbbrevSection =  "\t.section\t.debug_abbrev,\"\",@progbits";
  DwarfInfoSection =    "\t.section\t.debug_info,\"\",@progbits";
  DwarfLineSection =    "\t.section\t.debug_line,\"\",@progbits";
  DwarfFrameSection =   "\t.section\t.debug_frame,\"\",@progbits";
  DwarfPubNamesSection ="\t.section\t.debug_pubnames,\"\",@progbits";
  DwarfPubTypesSection ="\t.section\t.debug_pubtypes,\"\",@progbits";
  DwarfStrSection =     "\t.section\t.debug_str,\"\",@progbits";
  DwarfLocSection =     "\t.section\t.debug_loc,\"\",@progbits";
  DwarfARangesSection = "\t.section\t.debug_aranges,\"\",@progbits";
  DwarfRangesSection =  "\t.section\t.debug_ranges,\"\",@progbits";
  DwarfMacInfoSection = "\t.section\t.debug_macinfo,\"\",@progbits";

  ReadOnlySection = "\t.section\t.rodata";
  FourByteConstantSection = "\t.section\t.rodata.cst4,\"aM\",@progbits,4";
  EightByteConstantSection = "\t.section\t.rodata.cst8,\"aM\",@progbits,8";
  SixteenByteConstantSection = "\t.section\t.rodata.cst16,\"aM\",@progbits,16";
  PCSymbol = ".";

  // Set up DWARF directives
  HasLEB128 = true;  // Target asm supports leb128 directives (little-endian)

  // Exceptions handling
  if (!TM.getSubtargetImpl()->isPPC64())
    SupportsExceptionHandling = true;
  AbsoluteEHSectionOffsets = false;
  DwarfEHFrameSection = "\t.section\t.eh_frame,\"aw\",@progbits";
  DwarfExceptionSection = "\t.section\t.gcc_except_table,\"a\",@progbits";
}
