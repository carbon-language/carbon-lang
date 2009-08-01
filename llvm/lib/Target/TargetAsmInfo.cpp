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

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ErrorHandling.h"
#include <cctype>
#include <cstring>
using namespace llvm;

TargetAsmInfo::TargetAsmInfo(const TargetMachine &tm) : TM(tm) {
  ZeroFillDirective = 0;
  NonexecutableStackDirective = 0;
  NeedsSet = false;
  MaxInstLength = 4;
  PCSymbol = "$";
  SeparatorChar = ';';
  CommentColumn = 60;
  CommentString = "#";
  FirstOperandColumn = 0;
  MaxOperandLength = 0;
  GlobalPrefix = "";
  PrivateGlobalPrefix = ".";
  LinkerPrivateGlobalPrefix = "";
  JumpTableSpecialLabelPrefix = 0;
  GlobalVarAddrPrefix = "";
  GlobalVarAddrSuffix = "";
  FunctionAddrPrefix = "";
  FunctionAddrSuffix = "";
  PersonalityPrefix = "";
  PersonalitySuffix = "";
  NeedsIndirectEncoding = false;
  InlineAsmStart = "#APP";
  InlineAsmEnd = "#NO_APP";
  AssemblerDialect = 0;
  AllowQuotesInName = false;
  ZeroDirective = "\t.zero\t";
  ZeroDirectiveSuffix = 0;
  AsciiDirective = "\t.ascii\t";
  AscizDirective = "\t.asciz\t";
  Data8bitsDirective = "\t.byte\t";
  Data16bitsDirective = "\t.short\t";
  Data32bitsDirective = "\t.long\t";
  Data64bitsDirective = "\t.quad\t";
  AlignDirective = "\t.align\t";
  AlignmentIsInBytes = true;
  TextAlignFillValue = 0;
  SwitchToSectionDirective = "\t.section\t";
  TextSectionStartSuffix = "";
  DataSectionStartSuffix = "";
  SectionEndDirectiveSuffix = 0;
  JumpTableDataSection = "\t.section .rodata";
  JumpTableDirective = 0;
  // FIXME: Flags are ELFish - replace with normal section stuff.
  StaticCtorsSection = "\t.section .ctors,\"aw\",@progbits";
  StaticDtorsSection = "\t.section .dtors,\"aw\",@progbits";
  GlobalDirective = "\t.globl\t";
  SetDirective = 0;
  LCOMMDirective = 0;
  COMMDirective = "\t.comm\t";
  COMMDirectiveTakesAlignment = true;
  HasDotTypeDotSizeDirective = true;
  HasSingleParameterDotFile = true;
  UsedDirective = 0;
  WeakRefDirective = 0;
  WeakDefDirective = 0;
  // FIXME: These are ELFish - move to ELFTAI.
  HiddenDirective = "\t.hidden\t";
  ProtectedDirective = "\t.protected\t";
  AbsoluteDebugSectionOffsets = false;
  AbsoluteEHSectionOffsets = false;
  HasLEB128 = false;
  HasDotLocAndDotFile = false;
  SupportsDebugInformation = false;
  SupportsExceptionHandling = false;
  DwarfRequiresFrameSection = true;
  DwarfUsesInlineInfoSection = false;
  Is_EHSymbolPrivate = true;
  GlobalEHDirective = 0;
  SupportsWeakOmittedEHFrame = true;
  DwarfSectionOffsetDirective = 0;
  DwarfAbbrevSection = ".debug_abbrev";
  DwarfInfoSection = ".debug_info";
  DwarfLineSection = ".debug_line";
  DwarfFrameSection = ".debug_frame";
  DwarfPubNamesSection = ".debug_pubnames";
  DwarfPubTypesSection = ".debug_pubtypes";
  DwarfDebugInlineSection = ".debug_inlined";
  DwarfStrSection = ".debug_str";
  DwarfLocSection = ".debug_loc";
  DwarfARangesSection = ".debug_aranges";
  DwarfRangesSection = ".debug_ranges";
  DwarfMacroInfoSection = ".debug_macinfo";
  DwarfEHFrameSection = ".eh_frame";
  DwarfExceptionSection = ".gcc_except_table";
  AsmTransCBE = 0;
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

unsigned TargetAsmInfo::PreferredEHDataFormat() const {
  return dwarf::DW_EH_PE_absptr;
}

unsigned TargetAsmInfo::getULEB128Size(unsigned Value) {
  unsigned Size = 0;
  do {
    Value >>= 7;
    Size += sizeof(int8_t);
  } while (Value);
  return Size;
}

unsigned TargetAsmInfo::getSLEB128Size(int Value) {
  unsigned Size = 0;
  int Sign = Value >> (8 * sizeof(Value) - 1);
  bool IsMore;

  do {
    unsigned Byte = Value & 0x7f;
    Value >>= 7;
    IsMore = Value != Sign || ((Byte ^ Sign) & 0x40) != 0;
    Size += sizeof(int8_t);
  } while (IsMore);
  return Size;
}
