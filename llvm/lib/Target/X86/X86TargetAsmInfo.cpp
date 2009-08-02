//===-- X86TargetAsmInfo.cpp - X86 asm properties ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the X86TargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "X86TargetAsmInfo.h"
#include "X86TargetMachine.h"
#include "X86Subtarget.h"
#include "llvm/DerivedTypes.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Module.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;
using namespace llvm::dwarf;

const char *const llvm::x86_asm_table[] = {
  "{si}", "S",
  "{di}", "D",
  "{ax}", "a",
  "{cx}", "c",
  "{memory}", "memory",
  "{flags}", "",
  "{dirflag}", "",
  "{fpsr}", "",
  "{cc}", "cc",
  0,0};

X86DarwinTargetAsmInfo::X86DarwinTargetAsmInfo(const X86TargetMachine &TM):
  X86TargetAsmInfo<DarwinTargetAsmInfo>(TM) {
  const X86Subtarget *Subtarget = &TM.getSubtarget<X86Subtarget>();
  bool is64Bit = Subtarget->is64Bit();

  AlignmentIsInBytes = false;
  TextAlignFillValue = 0x90;


  if (!is64Bit)
    Data64bitsDirective = 0;       // we can't emit a 64-bit unit
  ZeroDirective = "\t.space\t";  // ".space N" emits N zeros.
  ZeroFillDirective = "\t.zerofill\t";  // Uses .zerofill
  LCOMMDirective = "\t.lcomm\t";

  // Leopard and above support aligned common symbols.
  COMMDirectiveTakesAlignment = (Subtarget->getDarwinVers() >= 9);
  HasDotTypeDotSizeDirective = false;

  if (is64Bit) {
    PersonalityPrefix = "";
    PersonalitySuffix = "+4@GOTPCREL";
  } else {
    PersonalityPrefix = "L";
    PersonalitySuffix = "$non_lazy_ptr";
  }

  InlineAsmStart = "## InlineAsm Start";
  InlineAsmEnd = "## InlineAsm End";
  CommentString = "##";
  SetDirective = "\t.set";
  PCSymbol = ".";
  UsedDirective = "\t.no_dead_strip\t";
  ProtectedDirective = "\t.globl\t";

  SupportsDebugInformation = true;
  DwarfDebugInlineSection = ".section __DWARF,__debug_inlined,regular,debug";
  DwarfUsesInlineInfoSection = true;

  // Exceptions handling
  SupportsExceptionHandling = true;
  GlobalEHDirective = "\t.globl\t";
  SupportsWeakOmittedEHFrame = false;
  AbsoluteEHSectionOffsets = false;
  DwarfEHFrameSection =
  ".section __TEXT,__eh_frame,coalesced,no_toc+strip_static_syms+live_support";
}

const char *
X86DarwinTargetAsmInfo::getEHGlobalPrefix() const {
  const X86Subtarget* Subtarget = &TM.getSubtarget<X86Subtarget>();
  if (Subtarget->getDarwinVers() > 9)
    return PrivateGlobalPrefix;
  return "";
}

X86ELFTargetAsmInfo::X86ELFTargetAsmInfo(const X86TargetMachine &TM) :
  X86TargetAsmInfo<ELFTargetAsmInfo>(TM) {

  PrivateGlobalPrefix = ".L";
  WeakRefDirective = "\t.weak\t";
  SetDirective = "\t.set\t";
  PCSymbol = ".";

  // Set up DWARF directives
  HasLEB128 = true;  // Target asm supports leb128 directives (little-endian)

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

  // Exceptions handling
  SupportsExceptionHandling = true;
  AbsoluteEHSectionOffsets = false;
  DwarfEHFrameSection = "\t.section\t.eh_frame,\"aw\",@progbits";

  // On Linux we must declare when we can use a non-executable stack.
  if (TM.getSubtarget<X86Subtarget>().isLinux())
    NonexecutableStackDirective = "\t.section\t.note.GNU-stack,\"\",@progbits";
}


X86WinTargetAsmInfo::X86WinTargetAsmInfo(const X86TargetMachine &TM):
  X86TargetAsmInfo<TargetAsmInfo>(TM) {
  GlobalPrefix = "_";
  CommentString = ";";

  InlineAsmStart = "; InlineAsm Start";
  InlineAsmEnd   = "; InlineAsm End";

  PrivateGlobalPrefix = "$";
  AlignDirective = "\tALIGN\t";
  ZeroDirective = "\tdb\t";
  ZeroDirectiveSuffix = " dup(0)";
  AsciiDirective = "\tdb\t";
  AscizDirective = 0;
  Data8bitsDirective = "\tdb\t";
  Data16bitsDirective = "\tdw\t";
  Data32bitsDirective = "\tdd\t";
  Data64bitsDirective = "\tdq\t";
  HasDotTypeDotSizeDirective = false;
  HasSingleParameterDotFile = false;

  AlignmentIsInBytes = true;

  SwitchToSectionDirective = "";
  TextSectionStartSuffix = "\tSEGMENT PARA 'CODE'";
  DataSectionStartSuffix = "\tSEGMENT PARA 'DATA'";
  SectionEndDirectiveSuffix = "\tends\n";
}

// Instantiate default implementation.
TEMPLATE_INSTANTIATION(class X86TargetAsmInfo<TargetAsmInfo>);
