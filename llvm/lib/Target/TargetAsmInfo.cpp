//===-- TargetAsmInfo.cpp - Asm Info ---------------------------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines target asm properties related what form asm statements
// should take.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Support/Dwarf.h"
#include <cctype>
#include <cstring>

using namespace llvm;

TargetAsmInfo::TargetAsmInfo() :
  TextSection("\t.text"),
  DataSection("\t.data"),
  BSSSection("\t.bss"),
  TLSDataSection("\t.section .tdata,\"awT\",@progbits"),
  TLSBSSSection("\t.section .tbss,\"awT\",@nobits"),
  ZeroFillDirective(0),
  NonexecutableStackDirective(0),
  NeedsSet(false),
  MaxInstLength(4),
  PCSymbol("$"),
  SeparatorChar(';'),
  CommentString("#"),
  GlobalPrefix(""),
  PrivateGlobalPrefix("."),
  JumpTableSpecialLabelPrefix(0),
  GlobalVarAddrPrefix(""),
  GlobalVarAddrSuffix(""),
  FunctionAddrPrefix(""),
  FunctionAddrSuffix(""),
  PersonalityPrefix(""),
  PersonalitySuffix(""),
  NeedsIndirectEncoding(false),
  InlineAsmStart("#APP"),
  InlineAsmEnd("#NO_APP"),
  AssemblerDialect(0),
  StringConstantPrefix(".str"),
  ZeroDirective("\t.zero\t"),
  ZeroDirectiveSuffix(0),
  AsciiDirective("\t.ascii\t"),
  AscizDirective("\t.asciz\t"),
  Data8bitsDirective("\t.byte\t"),
  Data16bitsDirective("\t.short\t"),
  Data32bitsDirective("\t.long\t"),
  Data64bitsDirective("\t.quad\t"),
  AlignDirective("\t.align\t"),
  AlignmentIsInBytes(true),
  TextAlignFillValue(0),
  SwitchToSectionDirective("\t.section\t"),
  TextSectionStartSuffix(""),
  DataSectionStartSuffix(""),
  SectionEndDirectiveSuffix(0),
  ConstantPoolSection("\t.section .rodata"),
  JumpTableDataSection("\t.section .rodata"),
  JumpTableDirective(0),
  CStringSection(0),
  StaticCtorsSection("\t.section .ctors,\"aw\",@progbits"),
  StaticDtorsSection("\t.section .dtors,\"aw\",@progbits"),
  FourByteConstantSection(0),
  EightByteConstantSection(0),
  SixteenByteConstantSection(0),
  ReadOnlySection(0),
  GlobalDirective("\t.globl\t"),
  SetDirective(0),
  LCOMMDirective(0),
  COMMDirective("\t.comm\t"),
  COMMDirectiveTakesAlignment(true),
  HasDotTypeDotSizeDirective(true),
  UsedDirective(0),
  WeakRefDirective(0),
  WeakDefDirective(0),
  HiddenDirective("\t.hidden\t"),
  ProtectedDirective("\t.protected\t"),
  AbsoluteDebugSectionOffsets(false),
  AbsoluteEHSectionOffsets(false),
  HasLEB128(false),
  HasDotLocAndDotFile(false),
  SupportsDebugInformation(false),
  SupportsExceptionHandling(false),
  DwarfRequiresFrameSection(true),
  GlobalEHDirective(0),
  SupportsWeakOmittedEHFrame(true),
  DwarfSectionOffsetDirective(0),
  DwarfAbbrevSection(".debug_abbrev"),
  DwarfInfoSection(".debug_info"),
  DwarfLineSection(".debug_line"),
  DwarfFrameSection(".debug_frame"),
  DwarfPubNamesSection(".debug_pubnames"),
  DwarfPubTypesSection(".debug_pubtypes"),
  DwarfStrSection(".debug_str"),
  DwarfLocSection(".debug_loc"),
  DwarfARangesSection(".debug_aranges"),
  DwarfRangesSection(".debug_ranges"),
  DwarfMacInfoSection(".debug_macinfo"),
  DwarfEHFrameSection(".eh_frame"),
  DwarfExceptionSection(".gcc_except_table"),
  AsmTransCBE(0) {
}

TargetAsmInfo::~TargetAsmInfo() {
}

/// Measure the specified inline asm to determine an approximation of its
/// length.
/// Comments (which run till the next SeparatorChar or newline) do not
/// count as an instruction.
/// Any other non-whitespace text is considered an instruction, with
/// multiple instructions separated by SeparatorChar or newlines.
/// Variable-length instructions are not handled here; this function
/// may be overloaded in the target code to do that.
unsigned TargetAsmInfo::getInlineAsmLength(const char *Str) const {
  // Count the number of instructions in the asm.
  bool atInsnStart = true;
  unsigned Length = 0;
  for (; *Str; ++Str) {
    if (*Str == '\n' || *Str == SeparatorChar)
      atInsnStart = true;
    if (atInsnStart && !isspace(*Str)) {
      Length += MaxInstLength;
      atInsnStart = false;
    }
    if (atInsnStart && strncmp(Str, CommentString, strlen(CommentString))==0)
      atInsnStart = false;
  }

  return Length;
}

unsigned TargetAsmInfo::PreferredEHDataFormat(DwarfEncoding::Target Reason,
                                              bool Global) const {
  return dwarf::DW_EH_PE_absptr;
}

