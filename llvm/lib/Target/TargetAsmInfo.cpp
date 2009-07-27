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
  BSSSection = "\t.bss";
  BSSSection_ = 0;
  ReadOnlySection = 0;
  TLSDataSection = 0;
  TLSBSSSection = 0;
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
  ConstantPoolSection = "\t.section .rodata";
  JumpTableDataSection = "\t.section .rodata";
  JumpTableDirective = 0;
  CStringSection = 0;
  CStringSection_ = 0;
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

unsigned TargetAsmInfo::PreferredEHDataFormat(DwarfEncoding::Target Reason,
                                              bool Global) const {
  return dwarf::DW_EH_PE_absptr;
}

static bool isSuitableForBSS(const GlobalVariable *GV) {
  Constant *C = GV->getInitializer();
  
  // Must have zero initializer.
  if (!C->isNullValue())
    return false;
  
  // Leave constant zeros in readonly constant sections, so they can be shared.
  if (GV->isConstant())
    return false;
  
  // If the global has an explicit section specified, don't put it in BSS.
  if (!GV->getSection().empty())
    return false;
  
  // If -nozero-initialized-in-bss is specified, don't ever use BSS.
  if (NoZerosInBSS)
    return false;
  
  // Otherwise, put it in BSS!
  return true;
}

static bool isConstantString(const Constant *C) {
  // First check: is we have constant array of i8 terminated with zero
  const ConstantArray *CVA = dyn_cast<ConstantArray>(C);
  // Check, if initializer is a null-terminated string
  if (CVA && CVA->isCString())
    return true;

  // Another possibility: [1 x i8] zeroinitializer
  if (isa<ConstantAggregateZero>(C))
    if (const ArrayType *Ty = dyn_cast<ArrayType>(C->getType()))
      return (Ty->getElementType() == Type::Int8Ty &&
              Ty->getNumElements() == 1);

  return false;
}

static SectionKind::Kind SectionKindForGlobal(const GlobalValue *GV,
                                              const TargetMachine &TM) {
  Reloc::Model ReloModel = TM.getRelocationModel();
  
  // Early exit - functions should be always in text sections.
  const GlobalVariable *GVar = dyn_cast<GlobalVariable>(GV);
  if (GVar == 0)
    return SectionKind::Text;

  
  // Handle thread-local data first.
  if (GVar->isThreadLocal()) {
    if (isSuitableForBSS(GVar))
      return SectionKind::ThreadBSS;
    return SectionKind::ThreadData;
  }

  // Variable can be easily put to BSS section.
  if (isSuitableForBSS(GVar))
    return SectionKind::BSS;

  Constant *C = GVar->getInitializer();
  
  // If the global is marked constant, we can put it into a mergable section,
  // a mergable string section, or general .data if it contains relocations.
  if (GVar->isConstant()) {
    // If the initializer for the global contains something that requires a
    // relocation, then we may have to drop this into a wriable data section
    // even though it is marked const.
    switch (C->getRelocationInfo()) {
    default: llvm_unreachable("unknown relocation info kind");
    case Constant::NoRelocation:
      // If initializer is a null-terminated string, put it in a "cstring"
      // section if the target has it.
      if (isConstantString(C))
        return SectionKind::MergeableCString;
      
      // Otherwise, just drop it into a mergable constant section.  If we have
      // a section for this size, use it, otherwise use the arbitrary sized
      // mergable section.
      switch (TM.getTargetData()->getTypeAllocSize(C->getType())) {
      case 4:  return SectionKind::MergeableConst4;
      case 8:  return SectionKind::MergeableConst8;
      case 16: return SectionKind::MergeableConst16;
      default: return SectionKind::MergeableConst;
      }
      
    case Constant::LocalRelocation:
      // In static relocation model, the linker will resolve all addresses, so
      // the relocation entries will actually be constants by the time the app
      // starts up.
      if (ReloModel == Reloc::Static)
        return SectionKind::ReadOnly;
              
      // Otherwise, the dynamic linker needs to fix it up, put it in the
      // writable data.rel.local section.
      return SectionKind::ReadOnlyWithRelLocal;
              
    case Constant::GlobalRelocations:
      // In static relocation model, the linker will resolve all addresses, so
      // the relocation entries will actually be constants by the time the app
      // starts up.
      if (ReloModel == Reloc::Static)
        return SectionKind::ReadOnly;
      
      // Otherwise, the dynamic linker needs to fix it up, put it in the
      // writable data.rel section.
      return SectionKind::ReadOnlyWithRel;
    }
  }

  // Okay, this isn't a constant.  If the initializer for the global is going
  // to require a runtime relocation by the dynamic linker, put it into a more
  // specific section to improve startup time of the app.  This coalesces these
  // globals together onto fewer pages, improving the locality of the dynamic
  // linker.
  if (ReloModel == Reloc::Static)
    return SectionKind::DataNoRel;

  switch (C->getRelocationInfo()) {
  default: llvm_unreachable("unknown relocation info kind");
  case Constant::NoRelocation:
    return SectionKind::DataNoRel;
  case Constant::LocalRelocation:
    return SectionKind::DataRelLocal;
  case Constant::GlobalRelocations:
    return SectionKind::DataRel;
  }
}

/// SectionForGlobal - This method computes the appropriate section to emit
/// the specified global variable or function definition.  This should not
/// be passed external (or available externally) globals.
const Section *TargetAsmInfo::SectionForGlobal(const GlobalValue *GV) const {
  assert(!GV->isDeclaration() && !GV->hasAvailableExternallyLinkage() &&
         "Can only be used for global definitions");
  
  SectionKind::Kind GVKind = SectionKindForGlobal(GV, TM);
  
  SectionKind Kind = SectionKind::get(GVKind, GV->isWeakForLinker(),
                                      GV->hasSection());


  // Select section name.
  if (GV->hasSection()) {
    // If the target has special section hacks for specifically named globals,
    // return them now.
    if (const Section *TS = getSpecialCasedSectionGlobals(GV, Kind))
      return TS;
    
    // If the target has magic semantics for certain section names, make sure to
    // pick up the flags.  This allows the user to write things with attribute
    // section and still get the appropriate section flags printed.
    GVKind = getKindForNamedSection(GV->getSection().c_str(), GVKind);
    
    return getOrCreateSection(GV->getSection().c_str(), false, GVKind);
  }

  // If this global is linkonce/weak and the target handles this by emitting it
  // into a 'uniqued' section name, create and return the section now.
  if (Kind.isWeak()) {
    if (const char *Prefix = getSectionPrefixForUniqueGlobal(Kind)) {
      // FIXME: Use mangler interface (PR4584).
      std::string Name = Prefix+GV->getNameStr();
      return getOrCreateSection(Name.c_str(), false, GVKind);
    }
  }
  
  // Use default section depending on the 'type' of global
  return SelectSectionForGlobal(GV, Kind);
}

// Lame default implementation. Calculate the section name for global.
const Section*
TargetAsmInfo::SelectSectionForGlobal(const GlobalValue *GV,
                                      SectionKind Kind) const {
  assert(!Kind.isThreadLocal() && "Doesn't support TLS");
  
  if (Kind.isText())
    return getTextSection();
  
  if (Kind.isBSS())
    if (const Section *S = getBSSSection_())
      return S;
  
  if (Kind.isReadOnly())
    if (const Section *S = getReadOnlySection())
      return S;

  return getDataSection();
}

/// getSectionForMergableConstant - Given a mergable constant with the
/// specified size and relocation information, return a section that it
/// should be placed in.
const Section *
TargetAsmInfo::getSectionForMergeableConstant(SectionKind Kind) const {
  if (Kind.isReadOnly())
    if (const Section *S = getReadOnlySection())
      return S;
  
  return getDataSection();
}


const Section *TargetAsmInfo::getOrCreateSection(const char *Name,
                                                 bool isDirective,
                                                 SectionKind::Kind Kind) const {
  Section &S = Sections[Name];

  // This is newly-created section, set it up properly.
  if (S.Name.empty()) {
    S.Kind = SectionKind::get(Kind, false /*weak*/, !isDirective);
    S.Name = Name;
  }

  return &S;
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
