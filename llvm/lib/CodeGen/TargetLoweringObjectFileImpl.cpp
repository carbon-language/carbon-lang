//===-- llvm/CodeGen/TargetLoweringObjectFileImpl.cpp - Object File Info --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements classes used to handle lowerings specific to common
// object file formats.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSectionCOFF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/Mangler.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Triple.h"
using namespace llvm;
using namespace dwarf;

//===----------------------------------------------------------------------===//
//                                  ELF
//===----------------------------------------------------------------------===//

TargetLoweringObjectFileELF::TargetLoweringObjectFileELF()
  : TargetLoweringObjectFile(),
    TLSDataSection(0),
    TLSBSSSection(0),
    DataRelSection(0),
    DataRelLocalSection(0),
    DataRelROSection(0),
    DataRelROLocalSection(0),
    MergeableConst4Section(0),
    MergeableConst8Section(0),
    MergeableConst16Section(0) {
}

void TargetLoweringObjectFileELF::Initialize(MCContext &Ctx,
                                             const TargetMachine &TM) {
  TargetLoweringObjectFile::Initialize(Ctx, TM);

  BSSSection =
    getContext().getELFSection(".bss", ELF::SHT_NOBITS,
                               ELF::SHF_WRITE |ELF::SHF_ALLOC,
                               SectionKind::getBSS());

  TextSection =
    getContext().getELFSection(".text", ELF::SHT_PROGBITS,
                               ELF::SHF_EXECINSTR |
                               ELF::SHF_ALLOC,
                               SectionKind::getText());

  DataSection =
    getContext().getELFSection(".data", ELF::SHT_PROGBITS,
                               ELF::SHF_WRITE |ELF::SHF_ALLOC,
                               SectionKind::getDataRel());

  ReadOnlySection =
    getContext().getELFSection(".rodata", ELF::SHT_PROGBITS,
                               ELF::SHF_ALLOC,
                               SectionKind::getReadOnly());

  TLSDataSection =
    getContext().getELFSection(".tdata", ELF::SHT_PROGBITS,
                               ELF::SHF_ALLOC | ELF::SHF_TLS |
                               ELF::SHF_WRITE,
                               SectionKind::getThreadData());

  TLSBSSSection =
    getContext().getELFSection(".tbss", ELF::SHT_NOBITS,
                               ELF::SHF_ALLOC | ELF::SHF_TLS |
                               ELF::SHF_WRITE,
                               SectionKind::getThreadBSS());

  DataRelSection =
    getContext().getELFSection(".data.rel", ELF::SHT_PROGBITS,
                               ELF::SHF_ALLOC |ELF::SHF_WRITE,
                               SectionKind::getDataRel());

  DataRelLocalSection =
    getContext().getELFSection(".data.rel.local", ELF::SHT_PROGBITS,
                               ELF::SHF_ALLOC |ELF::SHF_WRITE,
                               SectionKind::getDataRelLocal());

  DataRelROSection =
    getContext().getELFSection(".data.rel.ro", ELF::SHT_PROGBITS,
                               ELF::SHF_ALLOC |ELF::SHF_WRITE,
                               SectionKind::getReadOnlyWithRel());

  DataRelROLocalSection =
    getContext().getELFSection(".data.rel.ro.local", ELF::SHT_PROGBITS,
                               ELF::SHF_ALLOC |ELF::SHF_WRITE,
                               SectionKind::getReadOnlyWithRelLocal());

  MergeableConst4Section =
    getContext().getELFSection(".rodata.cst4", ELF::SHT_PROGBITS,
                               ELF::SHF_ALLOC |ELF::SHF_MERGE,
                               SectionKind::getMergeableConst4());

  MergeableConst8Section =
    getContext().getELFSection(".rodata.cst8", ELF::SHT_PROGBITS,
                               ELF::SHF_ALLOC |ELF::SHF_MERGE,
                               SectionKind::getMergeableConst8());

  MergeableConst16Section =
    getContext().getELFSection(".rodata.cst16", ELF::SHT_PROGBITS,
                               ELF::SHF_ALLOC |ELF::SHF_MERGE,
                               SectionKind::getMergeableConst16());

  StaticCtorSection =
    getContext().getELFSection(".ctors", ELF::SHT_PROGBITS,
                               ELF::SHF_ALLOC |ELF::SHF_WRITE,
                               SectionKind::getDataRel());

  StaticDtorSection =
    getContext().getELFSection(".dtors", ELF::SHT_PROGBITS,
                               ELF::SHF_ALLOC |ELF::SHF_WRITE,
                               SectionKind::getDataRel());

  // Exception Handling Sections.

  // FIXME: We're emitting LSDA info into a readonly section on ELF, even though
  // it contains relocatable pointers.  In PIC mode, this is probably a big
  // runtime hit for C++ apps.  Either the contents of the LSDA need to be
  // adjusted or this should be a data section.
  LSDASection =
    getContext().getELFSection(".gcc_except_table", ELF::SHT_PROGBITS,
                               ELF::SHF_ALLOC,
                               SectionKind::getReadOnly());
  // Debug Info Sections.
  DwarfAbbrevSection =
    getContext().getELFSection(".debug_abbrev", ELF::SHT_PROGBITS, 0,
                               SectionKind::getMetadata());
  DwarfInfoSection =
    getContext().getELFSection(".debug_info", ELF::SHT_PROGBITS, 0,
                               SectionKind::getMetadata());
  DwarfLineSection =
    getContext().getELFSection(".debug_line", ELF::SHT_PROGBITS, 0,
                               SectionKind::getMetadata());
  DwarfFrameSection =
    getContext().getELFSection(".debug_frame", ELF::SHT_PROGBITS, 0,
                               SectionKind::getMetadata());
  DwarfPubNamesSection =
    getContext().getELFSection(".debug_pubnames", ELF::SHT_PROGBITS, 0,
                               SectionKind::getMetadata());
  DwarfPubTypesSection =
    getContext().getELFSection(".debug_pubtypes", ELF::SHT_PROGBITS, 0,
                               SectionKind::getMetadata());
  DwarfStrSection =
    getContext().getELFSection(".debug_str", ELF::SHT_PROGBITS, 0,
                               SectionKind::getMetadata());
  DwarfLocSection =
    getContext().getELFSection(".debug_loc", ELF::SHT_PROGBITS, 0,
                               SectionKind::getMetadata());
  DwarfARangesSection =
    getContext().getELFSection(".debug_aranges", ELF::SHT_PROGBITS, 0,
                               SectionKind::getMetadata());
  DwarfRangesSection =
    getContext().getELFSection(".debug_ranges", ELF::SHT_PROGBITS, 0,
                               SectionKind::getMetadata());
  DwarfMacroInfoSection =
    getContext().getELFSection(".debug_macinfo", ELF::SHT_PROGBITS, 0,
                               SectionKind::getMetadata());
}

const MCSection *TargetLoweringObjectFileELF::getEHFrameSection() const {
  return getContext().getELFSection(".eh_frame", ELF::SHT_PROGBITS,
                                    ELF::SHF_ALLOC,
                                    SectionKind::getDataRel());
}

MCSymbol *
TargetLoweringObjectFileELF::getCFIPersonalitySymbol(const GlobalValue *GV,
                                                     Mangler *Mang,
                                                MachineModuleInfo *MMI) const {
  unsigned Encoding = getPersonalityEncoding();
  switch (Encoding & 0x70) {
  default:
    report_fatal_error("We do not support this DWARF encoding yet!");
  case dwarf::DW_EH_PE_absptr:
    return  Mang->getSymbol(GV);
    break;
  case dwarf::DW_EH_PE_pcrel: {
    return getContext().GetOrCreateSymbol(StringRef("DW.ref.") +
                                          Mang->getSymbol(GV)->getName());
    break;
  }
  }
}

void TargetLoweringObjectFileELF::emitPersonalityValue(MCStreamer &Streamer,
                                                       const TargetMachine &TM,
                                                       const MCSymbol *Sym) const {
  SmallString<64> NameData("DW.ref.");
  NameData += Sym->getName();
  MCSymbol *Label = getContext().GetOrCreateSymbol(NameData);
  Streamer.EmitSymbolAttribute(Label, MCSA_Hidden);
  Streamer.EmitSymbolAttribute(Label, MCSA_Weak);
  StringRef Prefix = ".data.";
  NameData.insert(NameData.begin(), Prefix.begin(), Prefix.end());
  unsigned Flags = ELF::SHF_ALLOC | ELF::SHF_WRITE | ELF::SHF_GROUP;
  const MCSection *Sec = getContext().getELFSection(NameData,
                                                    ELF::SHT_PROGBITS,
                                                    Flags,
                                                    SectionKind::getDataRel(),
                                                    0, Label->getName());
  Streamer.SwitchSection(Sec);
  Streamer.EmitValueToAlignment(8);
  Streamer.EmitSymbolAttribute(Label, MCSA_ELF_TypeObject);
  const MCExpr *E = MCConstantExpr::Create(8, getContext());
  Streamer.EmitELFSize(Label, E);
  Streamer.EmitLabel(Label);

  unsigned Size = TM.getTargetData()->getPointerSize();
  Streamer.EmitSymbolValue(Sym, Size);
}

static SectionKind
getELFKindForNamedSection(StringRef Name, SectionKind K) {
  // N.B.: The defaults used in here are no the same ones used in MC.
  // We follow gcc, MC follows gas. For example, given ".section .eh_frame",
  // both gas and MC will produce a section with no flags. Given
  // section(".eh_frame") gcc will produce
  // .section	.eh_frame,"a",@progbits
  if (Name.empty() || Name[0] != '.') return K;

  // Some lame default implementation based on some magic section names.
  if (Name == ".bss" ||
      Name.startswith(".bss.") ||
      Name.startswith(".gnu.linkonce.b.") ||
      Name.startswith(".llvm.linkonce.b.") ||
      Name == ".sbss" ||
      Name.startswith(".sbss.") ||
      Name.startswith(".gnu.linkonce.sb.") ||
      Name.startswith(".llvm.linkonce.sb."))
    return SectionKind::getBSS();

  if (Name == ".tdata" ||
      Name.startswith(".tdata.") ||
      Name.startswith(".gnu.linkonce.td.") ||
      Name.startswith(".llvm.linkonce.td."))
    return SectionKind::getThreadData();

  if (Name == ".tbss" ||
      Name.startswith(".tbss.") ||
      Name.startswith(".gnu.linkonce.tb.") ||
      Name.startswith(".llvm.linkonce.tb."))
    return SectionKind::getThreadBSS();

  return K;
}


static unsigned getELFSectionType(StringRef Name, SectionKind K) {

  if (Name == ".init_array")
    return ELF::SHT_INIT_ARRAY;

  if (Name == ".fini_array")
    return ELF::SHT_FINI_ARRAY;

  if (Name == ".preinit_array")
    return ELF::SHT_PREINIT_ARRAY;

  if (K.isBSS() || K.isThreadBSS())
    return ELF::SHT_NOBITS;

  return ELF::SHT_PROGBITS;
}


static unsigned
getELFSectionFlags(SectionKind K) {
  unsigned Flags = 0;

  if (!K.isMetadata())
    Flags |= ELF::SHF_ALLOC;

  if (K.isText())
    Flags |= ELF::SHF_EXECINSTR;

  if (K.isWriteable())
    Flags |= ELF::SHF_WRITE;

  if (K.isThreadLocal())
    Flags |= ELF::SHF_TLS;

  // K.isMergeableConst() is left out to honour PR4650
  if (K.isMergeableCString() || K.isMergeableConst4() ||
      K.isMergeableConst8() || K.isMergeableConst16())
    Flags |= ELF::SHF_MERGE;

  if (K.isMergeableCString())
    Flags |= ELF::SHF_STRINGS;

  return Flags;
}


const MCSection *TargetLoweringObjectFileELF::
getExplicitSectionGlobal(const GlobalValue *GV, SectionKind Kind,
                         Mangler *Mang, const TargetMachine &TM) const {
  StringRef SectionName = GV->getSection();

  // Infer section flags from the section name if we can.
  Kind = getELFKindForNamedSection(SectionName, Kind);

  return getContext().getELFSection(SectionName,
                                    getELFSectionType(SectionName, Kind),
                                    getELFSectionFlags(Kind), Kind);
}

/// getSectionPrefixForGlobal - Return the section prefix name used by options
/// FunctionsSections and DataSections.
static const char *getSectionPrefixForGlobal(SectionKind Kind) {
  if (Kind.isText())                 return ".text.";
  if (Kind.isReadOnly())             return ".rodata.";

  if (Kind.isThreadData())           return ".tdata.";
  if (Kind.isThreadBSS())            return ".tbss.";

  if (Kind.isDataNoRel())            return ".data.";
  if (Kind.isDataRelLocal())         return ".data.rel.local.";
  if (Kind.isDataRel())              return ".data.rel.";
  if (Kind.isReadOnlyWithRelLocal()) return ".data.rel.ro.local.";

  assert(Kind.isReadOnlyWithRel() && "Unknown section kind");
  return ".data.rel.ro.";
}


const MCSection *TargetLoweringObjectFileELF::
SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                       Mangler *Mang, const TargetMachine &TM) const {
  // If we have -ffunction-section or -fdata-section then we should emit the
  // global value to a uniqued section specifically for it.
  bool EmitUniquedSection;
  if (Kind.isText())
    EmitUniquedSection = TM.getFunctionSections();
  else
    EmitUniquedSection = TM.getDataSections();

  // If this global is linkonce/weak and the target handles this by emitting it
  // into a 'uniqued' section name, create and return the section now.
  if ((GV->isWeakForLinker() || EmitUniquedSection) &&
      !Kind.isCommon() && !Kind.isBSS()) {
    const char *Prefix;
    Prefix = getSectionPrefixForGlobal(Kind);

    SmallString<128> Name(Prefix, Prefix+strlen(Prefix));
    MCSymbol *Sym = Mang->getSymbol(GV);
    Name.append(Sym->getName().begin(), Sym->getName().end());
    StringRef Group = "";
    unsigned Flags = getELFSectionFlags(Kind);
    if (GV->isWeakForLinker()) {
      Group = Sym->getName();
      Flags |= ELF::SHF_GROUP;
    }

    return getContext().getELFSection(Name.str(),
                                      getELFSectionType(Name.str(), Kind),
                                      Flags, Kind, 0, Group);
  }

  if (Kind.isText()) return TextSection;

  if (Kind.isMergeable1ByteCString() ||
      Kind.isMergeable2ByteCString() ||
      Kind.isMergeable4ByteCString()) {

    // We also need alignment here.
    // FIXME: this is getting the alignment of the character, not the
    // alignment of the global!
    unsigned Align =
      TM.getTargetData()->getPreferredAlignment(cast<GlobalVariable>(GV));

    const char *SizeSpec = ".rodata.str1.";
    if (Kind.isMergeable2ByteCString())
      SizeSpec = ".rodata.str2.";
    else if (Kind.isMergeable4ByteCString())
      SizeSpec = ".rodata.str4.";
    else
      assert(Kind.isMergeable1ByteCString() && "unknown string width");


    std::string Name = SizeSpec + utostr(Align);
    return getContext().getELFSection(Name, ELF::SHT_PROGBITS,
                                      ELF::SHF_ALLOC |
                                      ELF::SHF_MERGE |
                                      ELF::SHF_STRINGS,
                                      Kind);
  }

  if (Kind.isMergeableConst()) {
    if (Kind.isMergeableConst4() && MergeableConst4Section)
      return MergeableConst4Section;
    if (Kind.isMergeableConst8() && MergeableConst8Section)
      return MergeableConst8Section;
    if (Kind.isMergeableConst16() && MergeableConst16Section)
      return MergeableConst16Section;
    return ReadOnlySection;  // .const
  }

  if (Kind.isReadOnly())             return ReadOnlySection;

  if (Kind.isThreadData())           return TLSDataSection;
  if (Kind.isThreadBSS())            return TLSBSSSection;

  // Note: we claim that common symbols are put in BSSSection, but they are
  // really emitted with the magic .comm directive, which creates a symbol table
  // entry but not a section.
  if (Kind.isBSS() || Kind.isCommon()) return BSSSection;

  if (Kind.isDataNoRel())            return DataSection;
  if (Kind.isDataRelLocal())         return DataRelLocalSection;
  if (Kind.isDataRel())              return DataRelSection;
  if (Kind.isReadOnlyWithRelLocal()) return DataRelROLocalSection;

  assert(Kind.isReadOnlyWithRel() && "Unknown section kind");
  return DataRelROSection;
}

/// getSectionForConstant - Given a mergeable constant with the
/// specified size and relocation information, return a section that it
/// should be placed in.
const MCSection *TargetLoweringObjectFileELF::
getSectionForConstant(SectionKind Kind) const {
  if (Kind.isMergeableConst4() && MergeableConst4Section)
    return MergeableConst4Section;
  if (Kind.isMergeableConst8() && MergeableConst8Section)
    return MergeableConst8Section;
  if (Kind.isMergeableConst16() && MergeableConst16Section)
    return MergeableConst16Section;
  if (Kind.isReadOnly())
    return ReadOnlySection;

  if (Kind.isReadOnlyWithRelLocal()) return DataRelROLocalSection;
  assert(Kind.isReadOnlyWithRel() && "Unknown section kind");
  return DataRelROSection;
}

const MCExpr *TargetLoweringObjectFileELF::
getExprForDwarfGlobalReference(const GlobalValue *GV, Mangler *Mang,
                               MachineModuleInfo *MMI,
                               unsigned Encoding, MCStreamer &Streamer) const {

  if (Encoding & dwarf::DW_EH_PE_indirect) {
    MachineModuleInfoELF &ELFMMI = MMI->getObjFileInfo<MachineModuleInfoELF>();

    SmallString<128> Name;
    Mang->getNameWithPrefix(Name, GV, true);
    Name += ".DW.stub";

    // Add information about the stub reference to ELFMMI so that the stub
    // gets emitted by the asmprinter.
    MCSymbol *SSym = getContext().GetOrCreateSymbol(Name.str());
    MachineModuleInfoImpl::StubValueTy &StubSym = ELFMMI.getGVStubEntry(SSym);
    if (StubSym.getPointer() == 0) {
      MCSymbol *Sym = Mang->getSymbol(GV);
      StubSym = MachineModuleInfoImpl::StubValueTy(Sym, !GV->hasLocalLinkage());
    }

    return TargetLoweringObjectFile::
      getExprForDwarfReference(SSym, Encoding & ~dwarf::DW_EH_PE_indirect, Streamer);
  }

  return TargetLoweringObjectFile::
    getExprForDwarfGlobalReference(GV, Mang, MMI, Encoding, Streamer);
}

//===----------------------------------------------------------------------===//
//                                 MachO
//===----------------------------------------------------------------------===//

TargetLoweringObjectFileMachO::TargetLoweringObjectFileMachO()
  : TargetLoweringObjectFile(),
    TLSDataSection(0),
    TLSBSSSection(0),
    TLSTLVSection(0),
    TLSThreadInitSection(0),
    CStringSection(0),
    UStringSection(0),
    TextCoalSection(0),
    ConstTextCoalSection(0),
    ConstDataSection(0),
    DataCoalSection(0),
    DataCommonSection(0),
    DataBSSSection(0),
    FourByteConstantSection(0),
    EightByteConstantSection(0),
    SixteenByteConstantSection(0),
    LazySymbolPointerSection(0),
    NonLazySymbolPointerSection(0) {
}

void TargetLoweringObjectFileMachO::Initialize(MCContext &Ctx,
                                               const TargetMachine &TM) {
  IsFunctionEHFrameSymbolPrivate = false;
  SupportsWeakOmittedEHFrame = false;

  // .comm doesn't support alignment before Leopard.
  Triple T(((LLVMTargetMachine&)TM).getTargetTriple());
  if (T.isMacOSX() && T.isMacOSXVersionLT(10, 5))
    CommDirectiveSupportsAlignment = false;

  TargetLoweringObjectFile::Initialize(Ctx, TM);

  TextSection // .text
    = getContext().getMachOSection("__TEXT", "__text",
                                   MCSectionMachO::S_ATTR_PURE_INSTRUCTIONS,
                                   SectionKind::getText());
  DataSection // .data
    = getContext().getMachOSection("__DATA", "__data", 0,
                                   SectionKind::getDataRel());

  TLSDataSection // .tdata
    = getContext().getMachOSection("__DATA", "__thread_data",
                                   MCSectionMachO::S_THREAD_LOCAL_REGULAR,
                                   SectionKind::getDataRel());
  TLSBSSSection // .tbss
    = getContext().getMachOSection("__DATA", "__thread_bss",
                                   MCSectionMachO::S_THREAD_LOCAL_ZEROFILL,
                                   SectionKind::getThreadBSS());

  // TODO: Verify datarel below.
  TLSTLVSection // .tlv
    = getContext().getMachOSection("__DATA", "__thread_vars",
                                   MCSectionMachO::S_THREAD_LOCAL_VARIABLES,
                                   SectionKind::getDataRel());

  TLSThreadInitSection
    = getContext().getMachOSection("__DATA", "__thread_init",
                          MCSectionMachO::S_THREAD_LOCAL_INIT_FUNCTION_POINTERS,
                          SectionKind::getDataRel());

  CStringSection // .cstring
    = getContext().getMachOSection("__TEXT", "__cstring",
                                   MCSectionMachO::S_CSTRING_LITERALS,
                                   SectionKind::getMergeable1ByteCString());
  UStringSection
    = getContext().getMachOSection("__TEXT","__ustring", 0,
                                   SectionKind::getMergeable2ByteCString());
  FourByteConstantSection // .literal4
    = getContext().getMachOSection("__TEXT", "__literal4",
                                   MCSectionMachO::S_4BYTE_LITERALS,
                                   SectionKind::getMergeableConst4());
  EightByteConstantSection // .literal8
    = getContext().getMachOSection("__TEXT", "__literal8",
                                   MCSectionMachO::S_8BYTE_LITERALS,
                                   SectionKind::getMergeableConst8());

  // ld_classic doesn't support .literal16 in 32-bit mode, and ld64 falls back
  // to using it in -static mode.
  SixteenByteConstantSection = 0;
  if (TM.getRelocationModel() != Reloc::Static &&
      TM.getTargetData()->getPointerSize() == 32)
    SixteenByteConstantSection =   // .literal16
      getContext().getMachOSection("__TEXT", "__literal16",
                                   MCSectionMachO::S_16BYTE_LITERALS,
                                   SectionKind::getMergeableConst16());

  ReadOnlySection  // .const
    = getContext().getMachOSection("__TEXT", "__const", 0,
                                   SectionKind::getReadOnly());

  TextCoalSection
    = getContext().getMachOSection("__TEXT", "__textcoal_nt",
                                   MCSectionMachO::S_COALESCED |
                                   MCSectionMachO::S_ATTR_PURE_INSTRUCTIONS,
                                   SectionKind::getText());
  ConstTextCoalSection
    = getContext().getMachOSection("__TEXT", "__const_coal",
                                   MCSectionMachO::S_COALESCED,
                                   SectionKind::getReadOnly());
  ConstDataSection  // .const_data
    = getContext().getMachOSection("__DATA", "__const", 0,
                                   SectionKind::getReadOnlyWithRel());
  DataCoalSection
    = getContext().getMachOSection("__DATA","__datacoal_nt",
                                   MCSectionMachO::S_COALESCED,
                                   SectionKind::getDataRel());
  DataCommonSection
    = getContext().getMachOSection("__DATA","__common",
                                   MCSectionMachO::S_ZEROFILL,
                                   SectionKind::getBSS());
  DataBSSSection
    = getContext().getMachOSection("__DATA","__bss", MCSectionMachO::S_ZEROFILL,
                                   SectionKind::getBSS());


  LazySymbolPointerSection
    = getContext().getMachOSection("__DATA", "__la_symbol_ptr",
                                   MCSectionMachO::S_LAZY_SYMBOL_POINTERS,
                                   SectionKind::getMetadata());
  NonLazySymbolPointerSection
    = getContext().getMachOSection("__DATA", "__nl_symbol_ptr",
                                   MCSectionMachO::S_NON_LAZY_SYMBOL_POINTERS,
                                   SectionKind::getMetadata());

  if (TM.getRelocationModel() == Reloc::Static) {
    StaticCtorSection
      = getContext().getMachOSection("__TEXT", "__constructor", 0,
                                     SectionKind::getDataRel());
    StaticDtorSection
      = getContext().getMachOSection("__TEXT", "__destructor", 0,
                                     SectionKind::getDataRel());
  } else {
    StaticCtorSection
      = getContext().getMachOSection("__DATA", "__mod_init_func",
                                     MCSectionMachO::S_MOD_INIT_FUNC_POINTERS,
                                     SectionKind::getDataRel());
    StaticDtorSection
      = getContext().getMachOSection("__DATA", "__mod_term_func",
                                     MCSectionMachO::S_MOD_TERM_FUNC_POINTERS,
                                     SectionKind::getDataRel());
  }

  // Exception Handling.
  LSDASection = getContext().getMachOSection("__TEXT", "__gcc_except_tab", 0,
                                             SectionKind::getReadOnlyWithRel());

  if (T.isMacOSX() && !T.isMacOSXVersionLT(10, 6))
    CompactUnwindSection =
      getContext().getMachOSection("__LD", "__compact_unwind",
                                   MCSectionMachO::S_ATTR_DEBUG,
                                   SectionKind::getReadOnly());

  // Debug Information.
  DwarfAbbrevSection =
    getContext().getMachOSection("__DWARF", "__debug_abbrev",
                                 MCSectionMachO::S_ATTR_DEBUG,
                                 SectionKind::getMetadata());
  DwarfInfoSection =
    getContext().getMachOSection("__DWARF", "__debug_info",
                                 MCSectionMachO::S_ATTR_DEBUG,
                                 SectionKind::getMetadata());
  DwarfLineSection =
    getContext().getMachOSection("__DWARF", "__debug_line",
                                 MCSectionMachO::S_ATTR_DEBUG,
                                 SectionKind::getMetadata());
  DwarfFrameSection =
    getContext().getMachOSection("__DWARF", "__debug_frame",
                                 MCSectionMachO::S_ATTR_DEBUG,
                                 SectionKind::getMetadata());
  DwarfPubNamesSection =
    getContext().getMachOSection("__DWARF", "__debug_pubnames",
                                 MCSectionMachO::S_ATTR_DEBUG,
                                 SectionKind::getMetadata());
  DwarfPubTypesSection =
    getContext().getMachOSection("__DWARF", "__debug_pubtypes",
                                 MCSectionMachO::S_ATTR_DEBUG,
                                 SectionKind::getMetadata());
  DwarfStrSection =
    getContext().getMachOSection("__DWARF", "__debug_str",
                                 MCSectionMachO::S_ATTR_DEBUG,
                                 SectionKind::getMetadata());
  DwarfLocSection =
    getContext().getMachOSection("__DWARF", "__debug_loc",
                                 MCSectionMachO::S_ATTR_DEBUG,
                                 SectionKind::getMetadata());
  DwarfARangesSection =
    getContext().getMachOSection("__DWARF", "__debug_aranges",
                                 MCSectionMachO::S_ATTR_DEBUG,
                                 SectionKind::getMetadata());
  DwarfRangesSection =
    getContext().getMachOSection("__DWARF", "__debug_ranges",
                                 MCSectionMachO::S_ATTR_DEBUG,
                                 SectionKind::getMetadata());
  DwarfMacroInfoSection =
    getContext().getMachOSection("__DWARF", "__debug_macinfo",
                                 MCSectionMachO::S_ATTR_DEBUG,
                                 SectionKind::getMetadata());
  DwarfDebugInlineSection =
    getContext().getMachOSection("__DWARF", "__debug_inlined",
                                 MCSectionMachO::S_ATTR_DEBUG,
                                 SectionKind::getMetadata());

  TLSExtraDataSection = TLSTLVSection;
}

const MCSection *TargetLoweringObjectFileMachO::getEHFrameSection() const {
  return getContext().getMachOSection("__TEXT", "__eh_frame",
                                      MCSectionMachO::S_COALESCED |
                                      MCSectionMachO::S_ATTR_NO_TOC |
                                      MCSectionMachO::S_ATTR_STRIP_STATIC_SYMS |
                                      MCSectionMachO::S_ATTR_LIVE_SUPPORT,
                                      SectionKind::getReadOnly());
}

const MCSection *TargetLoweringObjectFileMachO::
getExplicitSectionGlobal(const GlobalValue *GV, SectionKind Kind,
                         Mangler *Mang, const TargetMachine &TM) const {
  // Parse the section specifier and create it if valid.
  StringRef Segment, Section;
  unsigned TAA = 0, StubSize = 0;
  bool TAAParsed;
  std::string ErrorCode =
    MCSectionMachO::ParseSectionSpecifier(GV->getSection(), Segment, Section,
                                          TAA, TAAParsed, StubSize);
  if (!ErrorCode.empty()) {
    // If invalid, report the error with report_fatal_error.
    report_fatal_error("Global variable '" + GV->getNameStr() +
                      "' has an invalid section specifier '" + GV->getSection()+
                      "': " + ErrorCode + ".");
    // Fall back to dropping it into the data section.
    return DataSection;
  }

  // Get the section.
  const MCSectionMachO *S =
    getContext().getMachOSection(Segment, Section, TAA, StubSize, Kind);

  // If TAA wasn't set by ParseSectionSpecifier() above,
  // use the value returned by getMachOSection() as a default.
  if (!TAAParsed)
    TAA = S->getTypeAndAttributes();

  // Okay, now that we got the section, verify that the TAA & StubSize agree.
  // If the user declared multiple globals with different section flags, we need
  // to reject it here.
  if (S->getTypeAndAttributes() != TAA || S->getStubSize() != StubSize) {
    // If invalid, report the error with report_fatal_error.
    report_fatal_error("Global variable '" + GV->getNameStr() +
                      "' section type or attributes does not match previous"
                      " section specifier");
  }

  return S;
}

const MCSection *TargetLoweringObjectFileMachO::
SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                       Mangler *Mang, const TargetMachine &TM) const {

  // Handle thread local data.
  if (Kind.isThreadBSS()) return TLSBSSSection;
  if (Kind.isThreadData()) return TLSDataSection;

  if (Kind.isText())
    return GV->isWeakForLinker() ? TextCoalSection : TextSection;

  // If this is weak/linkonce, put this in a coalescable section, either in text
  // or data depending on if it is writable.
  if (GV->isWeakForLinker()) {
    if (Kind.isReadOnly())
      return ConstTextCoalSection;
    return DataCoalSection;
  }

  // FIXME: Alignment check should be handled by section classifier.
  if (Kind.isMergeable1ByteCString() &&
      TM.getTargetData()->getPreferredAlignment(cast<GlobalVariable>(GV)) < 32)
    return CStringSection;

  // Do not put 16-bit arrays in the UString section if they have an
  // externally visible label, this runs into issues with certain linker
  // versions.
  if (Kind.isMergeable2ByteCString() && !GV->hasExternalLinkage() &&
      TM.getTargetData()->getPreferredAlignment(cast<GlobalVariable>(GV)) < 32)
    return UStringSection;

  if (Kind.isMergeableConst()) {
    if (Kind.isMergeableConst4())
      return FourByteConstantSection;
    if (Kind.isMergeableConst8())
      return EightByteConstantSection;
    if (Kind.isMergeableConst16() && SixteenByteConstantSection)
      return SixteenByteConstantSection;
  }

  // Otherwise, if it is readonly, but not something we can specially optimize,
  // just drop it in .const.
  if (Kind.isReadOnly())
    return ReadOnlySection;

  // If this is marked const, put it into a const section.  But if the dynamic
  // linker needs to write to it, put it in the data segment.
  if (Kind.isReadOnlyWithRel())
    return ConstDataSection;

  // Put zero initialized globals with strong external linkage in the
  // DATA, __common section with the .zerofill directive.
  if (Kind.isBSSExtern())
    return DataCommonSection;

  // Put zero initialized globals with local linkage in __DATA,__bss directive
  // with the .zerofill directive (aka .lcomm).
  if (Kind.isBSSLocal())
    return DataBSSSection;

  // Otherwise, just drop the variable in the normal data section.
  return DataSection;
}

const MCSection *
TargetLoweringObjectFileMachO::getSectionForConstant(SectionKind Kind) const {
  // If this constant requires a relocation, we have to put it in the data
  // segment, not in the text segment.
  if (Kind.isDataRel() || Kind.isReadOnlyWithRel())
    return ConstDataSection;

  if (Kind.isMergeableConst4())
    return FourByteConstantSection;
  if (Kind.isMergeableConst8())
    return EightByteConstantSection;
  if (Kind.isMergeableConst16() && SixteenByteConstantSection)
    return SixteenByteConstantSection;
  return ReadOnlySection;  // .const
}

/// shouldEmitUsedDirectiveFor - This hook allows targets to selectively decide
/// not to emit the UsedDirective for some symbols in llvm.used.
// FIXME: REMOVE this (rdar://7071300)
bool TargetLoweringObjectFileMachO::
shouldEmitUsedDirectiveFor(const GlobalValue *GV, Mangler *Mang) const {
  /// On Darwin, internally linked data beginning with "L" or "l" does not have
  /// the directive emitted (this occurs in ObjC metadata).
  if (!GV) return false;

  // Check whether the mangled name has the "Private" or "LinkerPrivate" prefix.
  if (GV->hasLocalLinkage() && !isa<Function>(GV)) {
    // FIXME: ObjC metadata is currently emitted as internal symbols that have
    // \1L and \0l prefixes on them.  Fix them to be Private/LinkerPrivate and
    // this horrible hack can go away.
    MCSymbol *Sym = Mang->getSymbol(GV);
    if (Sym->getName()[0] == 'L' || Sym->getName()[0] == 'l')
      return false;
  }

  return true;
}

const MCExpr *TargetLoweringObjectFileMachO::
getExprForDwarfGlobalReference(const GlobalValue *GV, Mangler *Mang,
                               MachineModuleInfo *MMI, unsigned Encoding,
                               MCStreamer &Streamer) const {
  // The mach-o version of this method defaults to returning a stub reference.

  if (Encoding & DW_EH_PE_indirect) {
    MachineModuleInfoMachO &MachOMMI =
      MMI->getObjFileInfo<MachineModuleInfoMachO>();

    SmallString<128> Name;
    Mang->getNameWithPrefix(Name, GV, true);
    Name += "$non_lazy_ptr";

    // Add information about the stub reference to MachOMMI so that the stub
    // gets emitted by the asmprinter.
    MCSymbol *SSym = getContext().GetOrCreateSymbol(Name.str());
    MachineModuleInfoImpl::StubValueTy &StubSym = MachOMMI.getGVStubEntry(SSym);
    if (StubSym.getPointer() == 0) {
      MCSymbol *Sym = Mang->getSymbol(GV);
      StubSym = MachineModuleInfoImpl::StubValueTy(Sym, !GV->hasLocalLinkage());
    }

    return TargetLoweringObjectFile::
      getExprForDwarfReference(SSym, Encoding & ~dwarf::DW_EH_PE_indirect, Streamer);
  }

  return TargetLoweringObjectFile::
    getExprForDwarfGlobalReference(GV, Mang, MMI, Encoding, Streamer);
}

MCSymbol *TargetLoweringObjectFileMachO::
getCFIPersonalitySymbol(const GlobalValue *GV, Mangler *Mang,
                        MachineModuleInfo *MMI) const {
  // The mach-o version of this method defaults to returning a stub reference.
  MachineModuleInfoMachO &MachOMMI =
    MMI->getObjFileInfo<MachineModuleInfoMachO>();

  SmallString<128> Name;
  Mang->getNameWithPrefix(Name, GV, true);
  Name += "$non_lazy_ptr";

  // Add information about the stub reference to MachOMMI so that the stub
  // gets emitted by the asmprinter.
  MCSymbol *SSym = getContext().GetOrCreateSymbol(Name.str());
  MachineModuleInfoImpl::StubValueTy &StubSym = MachOMMI.getGVStubEntry(SSym);
  if (StubSym.getPointer() == 0) {
    MCSymbol *Sym = Mang->getSymbol(GV);
    StubSym = MachineModuleInfoImpl::StubValueTy(Sym, !GV->hasLocalLinkage());
  }

  return SSym;
}

unsigned TargetLoweringObjectFileMachO::getPersonalityEncoding() const {
  return DW_EH_PE_indirect | DW_EH_PE_pcrel | DW_EH_PE_sdata4;
}

unsigned TargetLoweringObjectFileMachO::getLSDAEncoding() const {
  return DW_EH_PE_pcrel;
}

unsigned TargetLoweringObjectFileMachO::getFDEEncoding(bool CFI) const {
  return DW_EH_PE_pcrel;
}

unsigned TargetLoweringObjectFileMachO::getTTypeEncoding() const {
  return DW_EH_PE_indirect | DW_EH_PE_pcrel | DW_EH_PE_sdata4;
}

//===----------------------------------------------------------------------===//
//                                  COFF
//===----------------------------------------------------------------------===//

TargetLoweringObjectFileCOFF::TargetLoweringObjectFileCOFF()
  : TargetLoweringObjectFile(),
    DrectveSection(0),
    PDataSection(0),
    XDataSection(0) {
}

void TargetLoweringObjectFileCOFF::Initialize(MCContext &Ctx,
                                              const TargetMachine &TM) {
  TargetLoweringObjectFile::Initialize(Ctx, TM);
  TextSection =
    getContext().getCOFFSection(".text",
                                COFF::IMAGE_SCN_CNT_CODE |
                                COFF::IMAGE_SCN_MEM_EXECUTE |
                                COFF::IMAGE_SCN_MEM_READ,
                                SectionKind::getText());
  DataSection =
    getContext().getCOFFSection(".data",
                                COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                                COFF::IMAGE_SCN_MEM_READ |
                                COFF::IMAGE_SCN_MEM_WRITE,
                                SectionKind::getDataRel());
  ReadOnlySection =
    getContext().getCOFFSection(".rdata",
                                COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                                COFF::IMAGE_SCN_MEM_READ,
                                SectionKind::getReadOnly());
  StaticCtorSection =
    getContext().getCOFFSection(".ctors",
                                COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                                COFF::IMAGE_SCN_MEM_READ |
                                COFF::IMAGE_SCN_MEM_WRITE,
                                SectionKind::getDataRel());
  StaticDtorSection =
    getContext().getCOFFSection(".dtors",
                                COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                                COFF::IMAGE_SCN_MEM_READ |
                                COFF::IMAGE_SCN_MEM_WRITE,
                                SectionKind::getDataRel());

  // FIXME: We're emitting LSDA info into a readonly section on COFF, even
  // though it contains relocatable pointers.  In PIC mode, this is probably a
  // big runtime hit for C++ apps.  Either the contents of the LSDA need to be
  // adjusted or this should be a data section.
  LSDASection =
    getContext().getCOFFSection(".gcc_except_table",
                                COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                                COFF::IMAGE_SCN_MEM_READ,
                                SectionKind::getReadOnly());
  // Debug info.
  DwarfAbbrevSection =
    getContext().getCOFFSection(".debug_abbrev",
                                COFF::IMAGE_SCN_MEM_DISCARDABLE |
                                COFF::IMAGE_SCN_MEM_READ,
                                SectionKind::getMetadata());
  DwarfInfoSection =
    getContext().getCOFFSection(".debug_info",
                                COFF::IMAGE_SCN_MEM_DISCARDABLE |
                                COFF::IMAGE_SCN_MEM_READ,
                                SectionKind::getMetadata());
  DwarfLineSection =
    getContext().getCOFFSection(".debug_line",
                                COFF::IMAGE_SCN_MEM_DISCARDABLE |
                                COFF::IMAGE_SCN_MEM_READ,
                                SectionKind::getMetadata());
  DwarfFrameSection =
    getContext().getCOFFSection(".debug_frame",
                                COFF::IMAGE_SCN_MEM_DISCARDABLE |
                                COFF::IMAGE_SCN_MEM_READ,
                                SectionKind::getMetadata());
  DwarfPubNamesSection =
    getContext().getCOFFSection(".debug_pubnames",
                                COFF::IMAGE_SCN_MEM_DISCARDABLE |
                                COFF::IMAGE_SCN_MEM_READ,
                                SectionKind::getMetadata());
  DwarfPubTypesSection =
    getContext().getCOFFSection(".debug_pubtypes",
                                COFF::IMAGE_SCN_MEM_DISCARDABLE |
                                COFF::IMAGE_SCN_MEM_READ,
                                SectionKind::getMetadata());
  DwarfStrSection =
    getContext().getCOFFSection(".debug_str",
                                COFF::IMAGE_SCN_MEM_DISCARDABLE |
                                COFF::IMAGE_SCN_MEM_READ,
                                SectionKind::getMetadata());
  DwarfLocSection =
    getContext().getCOFFSection(".debug_loc",
                                COFF::IMAGE_SCN_MEM_DISCARDABLE |
                                COFF::IMAGE_SCN_MEM_READ,
                                SectionKind::getMetadata());
  DwarfARangesSection =
    getContext().getCOFFSection(".debug_aranges",
                                COFF::IMAGE_SCN_MEM_DISCARDABLE |
                                COFF::IMAGE_SCN_MEM_READ,
                                SectionKind::getMetadata());
  DwarfRangesSection =
    getContext().getCOFFSection(".debug_ranges",
                                COFF::IMAGE_SCN_MEM_DISCARDABLE |
                                COFF::IMAGE_SCN_MEM_READ,
                                SectionKind::getMetadata());
  DwarfMacroInfoSection =
    getContext().getCOFFSection(".debug_macinfo",
                                COFF::IMAGE_SCN_MEM_DISCARDABLE |
                                COFF::IMAGE_SCN_MEM_READ,
                                SectionKind::getMetadata());

  DrectveSection =
    getContext().getCOFFSection(".drectve",
                                COFF::IMAGE_SCN_LNK_INFO,
                                SectionKind::getMetadata());

  PDataSection =
    getContext().getCOFFSection(".pdata",
                                COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                                COFF::IMAGE_SCN_MEM_READ |
                                COFF::IMAGE_SCN_MEM_WRITE,
                                SectionKind::getDataRel());

  XDataSection =
    getContext().getCOFFSection(".xdata",
                                COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                                COFF::IMAGE_SCN_MEM_READ |
                                COFF::IMAGE_SCN_MEM_WRITE,
                                SectionKind::getDataRel());
}

const MCSection *TargetLoweringObjectFileCOFF::getEHFrameSection() const {
  return getContext().getCOFFSection(".eh_frame",
                                     COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                                     COFF::IMAGE_SCN_MEM_READ |
                                     COFF::IMAGE_SCN_MEM_WRITE,
                                     SectionKind::getDataRel());
}

const MCSection *TargetLoweringObjectFileCOFF::getWin64EHFuncTableSection(
                                                       StringRef suffix) const {
  if (suffix == "")
    return PDataSection;
  return getContext().getCOFFSection((".pdata"+suffix).str(),
                                     COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                                     COFF::IMAGE_SCN_MEM_READ |
                                     COFF::IMAGE_SCN_MEM_WRITE,
                                     SectionKind::getDataRel());
}

const MCSection *TargetLoweringObjectFileCOFF::getWin64EHTableSection(
                                                       StringRef suffix) const {
  if (suffix == "")
    return XDataSection;
  return getContext().getCOFFSection((".xdata"+suffix).str(),
                                     COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                                     COFF::IMAGE_SCN_MEM_READ |
                                     COFF::IMAGE_SCN_MEM_WRITE,
                                     SectionKind::getDataRel());
}


static unsigned
getCOFFSectionFlags(SectionKind K) {
  unsigned Flags = 0;

  if (K.isMetadata())
    Flags |=
      COFF::IMAGE_SCN_MEM_DISCARDABLE;
  else if (K.isText())
    Flags |=
      COFF::IMAGE_SCN_MEM_EXECUTE |
      COFF::IMAGE_SCN_MEM_READ |
      COFF::IMAGE_SCN_CNT_CODE;
  else if (K.isBSS ())
    Flags |=
      COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA |
      COFF::IMAGE_SCN_MEM_READ |
      COFF::IMAGE_SCN_MEM_WRITE;
  else if (K.isReadOnly())
    Flags |=
      COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
      COFF::IMAGE_SCN_MEM_READ;
  else if (K.isWriteable())
    Flags |=
      COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
      COFF::IMAGE_SCN_MEM_READ |
      COFF::IMAGE_SCN_MEM_WRITE;

  return Flags;
}

const MCSection *TargetLoweringObjectFileCOFF::
getExplicitSectionGlobal(const GlobalValue *GV, SectionKind Kind,
                         Mangler *Mang, const TargetMachine &TM) const {
  return getContext().getCOFFSection(GV->getSection(),
                                     getCOFFSectionFlags(Kind),
                                     Kind);
}

static const char *getCOFFSectionPrefixForUniqueGlobal(SectionKind Kind) {
  if (Kind.isText())
    return ".text$";
  if (Kind.isBSS ())
    return ".bss$";
  if (Kind.isWriteable())
    return ".data$";
  return ".rdata$";
}


const MCSection *TargetLoweringObjectFileCOFF::
SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                       Mangler *Mang, const TargetMachine &TM) const {
  assert(!Kind.isThreadLocal() && "Doesn't support TLS");

  // If this global is linkonce/weak and the target handles this by emitting it
  // into a 'uniqued' section name, create and return the section now.
  if (GV->isWeakForLinker()) {
    const char *Prefix = getCOFFSectionPrefixForUniqueGlobal(Kind);
    SmallString<128> Name(Prefix, Prefix+strlen(Prefix));
    MCSymbol *Sym = Mang->getSymbol(GV);
    Name.append(Sym->getName().begin() + 1, Sym->getName().end());

    unsigned Characteristics = getCOFFSectionFlags(Kind);

    Characteristics |= COFF::IMAGE_SCN_LNK_COMDAT;

    return getContext().getCOFFSection(Name.str(), Characteristics,
                          COFF::IMAGE_COMDAT_SELECT_ANY, Kind);
  }

  if (Kind.isText())
    return getTextSection();

  return getDataSection();
}

