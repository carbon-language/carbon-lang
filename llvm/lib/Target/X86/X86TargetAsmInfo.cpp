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
  if (TM.getRelocationModel() != Reloc::Static)
    ConstantPoolSection = "\t.const_data";
  else
    ConstantPoolSection = "\t.const\n";
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
  DwarfExceptionSection = ".section __DATA,__gcc_except_tab";
}

unsigned
X86DarwinTargetAsmInfo::PreferredEHDataFormat(DwarfEncoding::Target Reason,
                                              bool Global) const {
  if (Reason == DwarfEncoding::Functions && Global)
    return (DW_EH_PE_pcrel | DW_EH_PE_indirect | DW_EH_PE_sdata4);
  if (Reason == DwarfEncoding::CodeLabels || !Global)
    return DW_EH_PE_pcrel;
  return DW_EH_PE_absptr;
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

  CStringSection = ".rodata.str";
  PrivateGlobalPrefix = ".L";
  WeakRefDirective = "\t.weak\t";
  SetDirective = "\t.set\t";
  PCSymbol = ".";

  // Set up DWARF directives
  HasLEB128 = true;  // Target asm supports leb128 directives (little-endian)

  BSSSection_ = getOrCreateSection("\t.bss", true, SectionKind::BSS);

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
  DwarfExceptionSection = "\t.section\t.gcc_except_table,\"a\",@progbits";

  // On Linux we must declare when we can use a non-executable stack.
  if (TM.getSubtarget<X86Subtarget>().isLinux())
    NonexecutableStackDirective = "\t.section\t.note.GNU-stack,\"\",@progbits";
}

unsigned
X86ELFTargetAsmInfo::PreferredEHDataFormat(DwarfEncoding::Target Reason,
                                           bool Global) const {
  CodeModel::Model CM = TM.getCodeModel();
  bool is64Bit = TM.getSubtarget<X86Subtarget>().is64Bit();

  if (TM.getRelocationModel() == Reloc::PIC_) {
    unsigned Format = 0;

    if (!is64Bit)
      // 32 bit targets always encode pointers as 4 bytes
      Format = DW_EH_PE_sdata4;
    else {
      // 64 bit targets encode pointers in 4 bytes iff:
      // - code model is small OR
      // - code model is medium and we're emitting externally visible symbols
      //   or any code symbols
      if (CM == CodeModel::Small ||
          (CM == CodeModel::Medium && (Global ||
                                       Reason != DwarfEncoding::Data)))
        Format = DW_EH_PE_sdata4;
      else
        Format = DW_EH_PE_sdata8;
    }

    if (Global)
      Format |= DW_EH_PE_indirect;

    return (Format | DW_EH_PE_pcrel);
  } else {
    if (is64Bit &&
        (CM == CodeModel::Small ||
         (CM == CodeModel::Medium && Reason != DwarfEncoding::Data)))
      return DW_EH_PE_udata4;
    else
      return DW_EH_PE_absptr;
  }
}

X86COFFTargetAsmInfo::X86COFFTargetAsmInfo(const X86TargetMachine &TM):
  X86GenericTargetAsmInfo(TM) {

  GlobalPrefix = "_";
  LCOMMDirective = "\t.lcomm\t";
  COMMDirectiveTakesAlignment = false;
  HasDotTypeDotSizeDirective = false;
  HasSingleParameterDotFile = false;
  StaticCtorsSection = "\t.section .ctors,\"aw\"";
  StaticDtorsSection = "\t.section .dtors,\"aw\"";
  HiddenDirective = NULL;
  PrivateGlobalPrefix = "L";  // Prefix for private global symbols
  WeakRefDirective = "\t.weak\t";
  SetDirective = "\t.set\t";

  // Set up DWARF directives
  HasLEB128 = true;  // Target asm supports leb128 directives (little-endian)
  AbsoluteDebugSectionOffsets = true;
  AbsoluteEHSectionOffsets = false;
  SupportsDebugInformation = true;
  DwarfSectionOffsetDirective = "\t.secrel32\t";
  DwarfAbbrevSection =  "\t.section\t.debug_abbrev,\"dr\"";
  DwarfInfoSection =    "\t.section\t.debug_info,\"dr\"";
  DwarfLineSection =    "\t.section\t.debug_line,\"dr\"";
  DwarfFrameSection =   "\t.section\t.debug_frame,\"dr\"";
  DwarfPubNamesSection ="\t.section\t.debug_pubnames,\"dr\"";
  DwarfPubTypesSection ="\t.section\t.debug_pubtypes,\"dr\"";
  DwarfStrSection =     "\t.section\t.debug_str,\"dr\"";
  DwarfLocSection =     "\t.section\t.debug_loc,\"dr\"";
  DwarfARangesSection = "\t.section\t.debug_aranges,\"dr\"";
  DwarfRangesSection =  "\t.section\t.debug_ranges,\"dr\"";
  DwarfMacroInfoSection = "\t.section\t.debug_macinfo,\"dr\"";
}

unsigned
X86COFFTargetAsmInfo::PreferredEHDataFormat(DwarfEncoding::Target Reason,
                                            bool Global) const {
  CodeModel::Model CM = TM.getCodeModel();
  bool is64Bit = TM.getSubtarget<X86Subtarget>().is64Bit();

  if (TM.getRelocationModel() == Reloc::PIC_) {
    unsigned Format = 0;

    if (!is64Bit)
      // 32 bit targets always encode pointers as 4 bytes
      Format = DW_EH_PE_sdata4;
    else {
      // 64 bit targets encode pointers in 4 bytes iff:
      // - code model is small OR
      // - code model is medium and we're emitting externally visible symbols
      //   or any code symbols
      if (CM == CodeModel::Small ||
          (CM == CodeModel::Medium && (Global ||
                                       Reason != DwarfEncoding::Data)))
        Format = DW_EH_PE_sdata4;
      else
        Format = DW_EH_PE_sdata8;
    }

    if (Global)
      Format |= DW_EH_PE_indirect;

    return (Format | DW_EH_PE_pcrel);
  }
  
  if (is64Bit &&
      (CM == CodeModel::Small ||
       (CM == CodeModel::Medium && Reason != DwarfEncoding::Data)))
    return DW_EH_PE_udata4;
  return DW_EH_PE_absptr;
}

const char *X86COFFTargetAsmInfo::
getSectionPrefixForUniqueGlobal(SectionKind Kind) const {
  if (Kind.isText())
    return ".text$linkonce";
  if (Kind.isWriteable())
    return ".data$linkonce";
  return ".rdata$linkonce";
}



void X86COFFTargetAsmInfo::getSectionFlagsAsString(SectionKind Kind,
                                            SmallVectorImpl<char> &Str) const {
  // FIXME: Inefficient.
  std::string Res = ",\"";
  if (Kind.isText())
    Res += 'x';
  if (Kind.isWriteable())
    Res += 'w';
  Res += "\"";

  Str.append(Res.begin(), Res.end());
}

X86WinTargetAsmInfo::X86WinTargetAsmInfo(const X86TargetMachine &TM):
  X86GenericTargetAsmInfo(TM) {
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

  TextSection = getOrCreateSection("_text", true, SectionKind::Text);
  DataSection = getOrCreateSection("_data", true, SectionKind::DataRel);

  JumpTableDataSection = NULL;
  SwitchToSectionDirective = "";
  TextSectionStartSuffix = "\tSEGMENT PARA 'CODE'";
  DataSectionStartSuffix = "\tSEGMENT PARA 'DATA'";
  SectionEndDirectiveSuffix = "\tends\n";
}

// Instantiate default implementation.
TEMPLATE_INSTANTIATION(class X86TargetAsmInfo<TargetAsmInfo>);
