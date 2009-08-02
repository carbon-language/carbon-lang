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
  UsedDirective = "\t.no_dead_strip\t";
  SupportsExceptionHandling = true;
  
  DwarfEHFrameSection =
   ".section __TEXT,__eh_frame,coalesced,no_toc+strip_static_syms+live_support";
  GlobalEHDirective = "\t.globl\t";
  SupportsWeakOmittedEHFrame = false;
}

PPCLinuxTargetAsmInfo::PPCLinuxTargetAsmInfo(const PPCTargetMachine &TM) :
  PPCTargetAsmInfo<ELFTargetAsmInfo>(TM) {
  CommentString = "#";
  GlobalPrefix = "";
  PrivateGlobalPrefix = ".L";
  UsedDirective = "\t# .no_dead_strip\t";
  WeakRefDirective = "\t.weak\t";

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
}


// Instantiate default implementation.
TEMPLATE_INSTANTIATION(class PPCTargetAsmInfo<TargetAsmInfo>);
