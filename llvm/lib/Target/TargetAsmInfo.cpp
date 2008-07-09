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
#include "llvm/GlobalVariable.h"
#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetOptions.h"
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

static bool isSuitableForBSS(const GlobalVariable *GV) {
  if (!GV->hasInitializer())
    return true;

  // Leave constant zeros in readonly constant sections, so they can be shared
  Constant *C = GV->getInitializer();
  return (C->isNullValue() && !GV->isConstant() && !NoZerosInBSS);
}

SectionKind::Kind
TargetAsmInfo::SectionKindForGlobal(const GlobalValue *GV) const {
  // Early exit - functions should be always in text sections.
  if (isa<Function>(GV))
    return SectionKind::Text;

  const GlobalVariable* GVar = dyn_cast<GlobalVariable>(GV);
  bool isThreadLocal = GVar->isThreadLocal();
  assert(GVar && "Invalid global value for section selection");

  SectionKind::Kind kind;
  if (isSuitableForBSS(GVar)) {
    // Variable can be easily put to BSS section.
    return (isThreadLocal ? SectionKind::ThreadBSS : SectionKind::BSS);
  } else if (GVar->isConstant() && !isThreadLocal) {
    // Now we know, that varible has initializer and it is constant. We need to
    // check its initializer to decide, which section to output it into. Also
    // note, there is no thread-local r/o section.
    Constant *C = GVar->getInitializer();
    if (C->ContainsRelocations())
      kind = SectionKind::ROData;
    else {
      const ConstantArray *CVA = dyn_cast<ConstantArray>(C);
      // Check, if initializer is a null-terminated string
      if (CVA && CVA->isCString())
        kind = SectionKind::RODataMergeStr;
      else
        kind = SectionKind::RODataMergeConst;
    }
  }

  // Variable is not constant or thread-local - emit to generic data section.
  return (isThreadLocal ? SectionKind::ThreadData : SectionKind::Data);
}

unsigned
TargetAsmInfo::SectionFlagsForGlobal(const GlobalValue *GV,
                                     const char* name) const {
  unsigned flags = SectionFlags::None;

  // Decode flags from global itself.
  if (GV) {
    SectionKind::Kind kind = SectionKindForGlobal(GV);
    switch (kind) {
     case SectionKind::Text:
      flags |= SectionFlags::Code;
      break;
     case SectionKind::ThreadData:
      flags |= SectionFlags::TLS;
      // FALLS THROUGH
     case SectionKind::Data:
      flags |= SectionFlags::Writeable;
      break;
     case SectionKind::ThreadBSS:
      flags |= SectionFlags::TLS;
      // FALLS THROUGH
     case SectionKind::BSS:
      flags |= SectionFlags::BSS;
      break;
     case SectionKind::ROData:
      // No additional flags here
      break;
     case SectionKind::RODataMergeStr:
      flags |= SectionFlags::Strings;
      // FALLS THROUGH
     case SectionKind::RODataMergeConst:
      flags |= SectionFlags::Mergeable;
      break;
     default:
      assert(0 && "Unexpected section kind!");
    }

    if (GV->hasLinkOnceLinkage() ||
        GV->hasWeakLinkage() ||
        GV->hasCommonLinkage())
      flags |= SectionFlags::Linkonce;
  }

  // Add flags from sections, if any.
  if (name) {
    // Some lame default implementation
    if (strcmp(name, ".bss") == 0 ||
        strncmp(name, ".bss.", 5) == 0 ||
        strncmp(name, ".llvm.linkonce.b.", 17) == 0)
      flags |= SectionFlags::BSS;
    else if (strcmp(name, ".tdata") == 0 ||
             strncmp(name, ".tdata.", 7) == 0 ||
             strncmp(name, ".llvm.linkonce.td.", 18) == 0)
      flags |= SectionFlags::TLS;
    else if (strcmp(name, ".tbss") == 0 ||
             strncmp(name, ".tbss.", 6) == 0 ||
             strncmp(name, ".llvm.linkonce.tb.", 18) == 0)
      flags |= SectionFlags::BSS | SectionFlags::TLS;
  }

  return flags;
}

std::string
TargetAsmInfo::SectionForGlobal(const GlobalValue *GV) const {
  SectionKind::Kind kind = SectionKindForGlobal(GV);

  if (kind == SectionKind::Text)
    return getTextSection();
  else if (kind == SectionKind::BSS && getBSSSection())
    return getBSSSection();

  return getDataSection();
}

std::string
TargetAsmInfo::UniqueSectionForGlobal(const GlobalValue* GV,
                                      SectionKind::Kind kind) const {
  switch (kind) {
   case SectionKind::Text:
    return ".llvm.linkonce.t." + GV->getName();
   case SectionKind::Data:
    return ".llvm.linkonce.d." + GV->getName();
   case SectionKind::BSS:
    return ".llvm.linkonce.b." + GV->getName();
   case SectionKind::ROData:
   case SectionKind::RODataMergeConst:
   case SectionKind::RODataMergeStr:
    return ".llvm.linkonce.r." + GV->getName();
   case SectionKind::ThreadData:
    return ".llvm.linkonce.td." + GV->getName();
   case SectionKind::ThreadBSS:
    return ".llvm.linkonce.tb." + GV->getName();
   default:
    assert(0 && "Unknown section kind");
  }
}
