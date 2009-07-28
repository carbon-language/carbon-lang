//===-- PPCTargetAsmInfo.cpp - PPC asm properties ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the DarwinTargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "PPCTargetAsmInfo.h"
#include "PPCTargetMachine.h"
#include "llvm/Function.h"
#include "llvm/Support/Dwarf.h"

using namespace llvm;
using namespace llvm::dwarf;

PPCDarwinTargetAsmInfo::PPCDarwinTargetAsmInfo(const PPCTargetMachine &TM) :
  PPCTargetAsmInfo<DarwinTargetAsmInfo>(TM) {
  PCSymbol = ".";
  CommentString = ";";
  ConstantPoolSection = "\t.const\t";
  UsedDirective = "\t.no_dead_strip\t";
  SupportsExceptionHandling = true;
  
  DwarfEHFrameSection =
   ".section __TEXT,__eh_frame,coalesced,no_toc+strip_static_syms+live_support";
  DwarfExceptionSection = ".section __DATA,__gcc_except_tab";
  GlobalEHDirective = "\t.globl\t";
  SupportsWeakOmittedEHFrame = false;
}

/// PreferredEHDataFormat - This hook allows the target to select data
/// format used for encoding pointers in exception handling data. Reason is
/// 0 for data, 1 for code labels, 2 for function pointers. Global is true
/// if the symbol can be relocated.
unsigned
PPCDarwinTargetAsmInfo::PreferredEHDataFormat(DwarfEncoding::Target Reason,
                                              bool Global) const {
  if ((Reason == DwarfEncoding::Data || Reason == DwarfEncoding::Functions)
      && Global)
    return DW_EH_PE_pcrel | DW_EH_PE_indirect | DW_EH_PE_sdata4;

  if (Reason == DwarfEncoding::CodeLabels || !Global)
    return DW_EH_PE_pcrel;

  return DW_EH_PE_absptr;
}

const char *PPCDarwinTargetAsmInfo::getEHGlobalPrefix() const {
  const PPCSubtarget* Subtarget = &TM.getSubtarget<PPCSubtarget>();
  if (Subtarget->getDarwinVers() > 9)
    return PrivateGlobalPrefix;
  return "";
}

PPCLinuxTargetAsmInfo::PPCLinuxTargetAsmInfo(const PPCTargetMachine &TM) :
  PPCTargetAsmInfo<ELFTargetAsmInfo>(TM) {
  CommentString = "#";
  GlobalPrefix = "";
  PrivateGlobalPrefix = ".L";
  ConstantPoolSection = "\t.section .rodata.cst4\t";
  JumpTableDataSection = ".section .rodata.cst4";
  CStringSection = ".rodata.str";
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
  DwarfMacroInfoSection = "\t.section\t.debug_macinfo,\"\",@progbits";

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

/// PreferredEHDataFormat - This hook allows the target to select data
/// format used for encoding pointers in exception handling data. Reason is
/// 0 for data, 1 for code labels, 2 for function pointers. Global is true
/// if the symbol can be relocated.
unsigned
PPCLinuxTargetAsmInfo::PreferredEHDataFormat(DwarfEncoding::Target Reason,
                                             bool Global) const {
  // We really need to write something here.
  return TargetAsmInfo::PreferredEHDataFormat(Reason, Global);
}

// Instantiate default implementation.
TEMPLATE_INSTANTIATION(class PPCTargetAsmInfo<TargetAsmInfo>);
