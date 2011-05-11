//===- lib/MC/ELFObjectWriter.cpp - ELF File Writer -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements ELF object file writer information.
//
//===----------------------------------------------------------------------===//

#include "ELFObjectWriter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ELF.h"
#include "llvm/Target/TargetAsmBackend.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/Statistic.h"

#include "../Target/X86/X86FixupKinds.h"
#include "../Target/ARM/ARMFixupKinds.h"

#include <vector>
using namespace llvm;

#undef  DEBUG_TYPE
#define DEBUG_TYPE "reloc-info"

// Emulate the wierd behavior of GNU-as for relocation types
namespace llvm {
cl::opt<bool>
ForceARMElfPIC("arm-elf-force-pic", cl::Hidden, cl::init(false),
               cl::desc("Force ELF emitter to emit PIC style relocations"));
}

bool ELFObjectWriter::isFixupKindPCRel(const MCAssembler &Asm, unsigned Kind) {
  const MCFixupKindInfo &FKI =
    Asm.getBackend().getFixupKindInfo((MCFixupKind) Kind);

  return FKI.Flags & MCFixupKindInfo::FKF_IsPCRel;
}

bool ELFObjectWriter::RelocNeedsGOT(MCSymbolRefExpr::VariantKind Variant) {
  switch (Variant) {
  default:
    return false;
  case MCSymbolRefExpr::VK_GOT:
  case MCSymbolRefExpr::VK_PLT:
  case MCSymbolRefExpr::VK_GOTPCREL:
  case MCSymbolRefExpr::VK_TPOFF:
  case MCSymbolRefExpr::VK_TLSGD:
  case MCSymbolRefExpr::VK_GOTTPOFF:
  case MCSymbolRefExpr::VK_INDNTPOFF:
  case MCSymbolRefExpr::VK_NTPOFF:
  case MCSymbolRefExpr::VK_GOTNTPOFF:
  case MCSymbolRefExpr::VK_TLSLDM:
  case MCSymbolRefExpr::VK_DTPOFF:
  case MCSymbolRefExpr::VK_TLSLD:
    return true;
  }
}

ELFObjectWriter::~ELFObjectWriter()
{}

// Emit the ELF header.
void ELFObjectWriter::WriteHeader(uint64_t SectionDataSize,
                                  unsigned NumberOfSections) {
  // ELF Header
  // ----------
  //
  // Note
  // ----
  // emitWord method behaves differently for ELF32 and ELF64, writing
  // 4 bytes in the former and 8 in the latter.

  Write8(0x7f); // e_ident[EI_MAG0]
  Write8('E');  // e_ident[EI_MAG1]
  Write8('L');  // e_ident[EI_MAG2]
  Write8('F');  // e_ident[EI_MAG3]

  Write8(is64Bit() ? ELF::ELFCLASS64 : ELF::ELFCLASS32); // e_ident[EI_CLASS]

  // e_ident[EI_DATA]
  Write8(isLittleEndian() ? ELF::ELFDATA2LSB : ELF::ELFDATA2MSB);

  Write8(ELF::EV_CURRENT);        // e_ident[EI_VERSION]
  // e_ident[EI_OSABI]
  switch (TargetObjectWriter->getOSType()) {
    case Triple::FreeBSD:  Write8(ELF::ELFOSABI_FREEBSD); break;
    case Triple::Linux:    Write8(ELF::ELFOSABI_LINUX); break;
    default:               Write8(ELF::ELFOSABI_NONE); break;
  }
  Write8(0);                  // e_ident[EI_ABIVERSION]

  WriteZeros(ELF::EI_NIDENT - ELF::EI_PAD);

  Write16(ELF::ET_REL);             // e_type

  Write16(TargetObjectWriter->getEMachine()); // e_machine = target

  Write32(ELF::EV_CURRENT);         // e_version
  WriteWord(0);                    // e_entry, no entry point in .o file
  WriteWord(0);                    // e_phoff, no program header for .o
  WriteWord(SectionDataSize + (is64Bit() ? sizeof(ELF::Elf64_Ehdr) :
            sizeof(ELF::Elf32_Ehdr)));  // e_shoff = sec hdr table off in bytes

  // e_flags = whatever the target wants
  WriteEFlags();

  // e_ehsize = ELF header size
  Write16(is64Bit() ? sizeof(ELF::Elf64_Ehdr) : sizeof(ELF::Elf32_Ehdr));

  Write16(0);                  // e_phentsize = prog header entry size
  Write16(0);                  // e_phnum = # prog header entries = 0

  // e_shentsize = Section header entry size
  Write16(is64Bit() ? sizeof(ELF::Elf64_Shdr) : sizeof(ELF::Elf32_Shdr));

  // e_shnum     = # of section header ents
  if (NumberOfSections >= ELF::SHN_LORESERVE)
    Write16(0);
  else
    Write16(NumberOfSections);

  // e_shstrndx  = Section # of '.shstrtab'
  if (NumberOfSections >= ELF::SHN_LORESERVE)
    Write16(ELF::SHN_XINDEX);
  else
    Write16(ShstrtabIndex);
}

void ELFObjectWriter::WriteSymbolEntry(MCDataFragment *SymtabF,
                                       MCDataFragment *ShndxF,
                                       uint64_t name,
                                       uint8_t info, uint64_t value,
                                       uint64_t size, uint8_t other,
                                       uint32_t shndx,
                                       bool Reserved) {
  if (ShndxF) {
    if (shndx >= ELF::SHN_LORESERVE && !Reserved)
      String32(*ShndxF, shndx);
    else
      String32(*ShndxF, 0);
  }

  uint16_t Index = (shndx >= ELF::SHN_LORESERVE && !Reserved) ?
    uint16_t(ELF::SHN_XINDEX) : shndx;

  if (is64Bit()) {
    String32(*SymtabF, name);  // st_name
    String8(*SymtabF, info);   // st_info
    String8(*SymtabF, other);  // st_other
    String16(*SymtabF, Index); // st_shndx
    String64(*SymtabF, value); // st_value
    String64(*SymtabF, size);  // st_size
  } else {
    String32(*SymtabF, name);  // st_name
    String32(*SymtabF, value); // st_value
    String32(*SymtabF, size);  // st_size
    String8(*SymtabF, info);   // st_info
    String8(*SymtabF, other);  // st_other
    String16(*SymtabF, Index); // st_shndx
  }
}

uint64_t ELFObjectWriter::SymbolValue(MCSymbolData &Data,
                                      const MCAsmLayout &Layout) {
  if (Data.isCommon() && Data.isExternal())
    return Data.getCommonAlignment();

  const MCSymbol &Symbol = Data.getSymbol();

  if (Symbol.isAbsolute() && Symbol.isVariable()) {
    if (const MCExpr *Value = Symbol.getVariableValue()) {
      int64_t IntValue;
      if (Value->EvaluateAsAbsolute(IntValue, Layout))
	return (uint64_t)IntValue;
    }
  }

  if (!Symbol.isInSection())
    return 0;

  if (Data.getFragment())
    return Layout.getSymbolOffset(&Data);

  return 0;
}

void ELFObjectWriter::ExecutePostLayoutBinding(MCAssembler &Asm,
                                               const MCAsmLayout &Layout) {
  // The presence of symbol versions causes undefined symbols and
  // versions declared with @@@ to be renamed.

  for (MCAssembler::symbol_iterator it = Asm.symbol_begin(),
         ie = Asm.symbol_end(); it != ie; ++it) {
    const MCSymbol &Alias = it->getSymbol();
    const MCSymbol &Symbol = Alias.AliasedSymbol();
    MCSymbolData &SD = Asm.getSymbolData(Symbol);

    // Not an alias.
    if (&Symbol == &Alias)
      continue;

    StringRef AliasName = Alias.getName();
    size_t Pos = AliasName.find('@');
    if (Pos == StringRef::npos)
      continue;

    // Aliases defined with .symvar copy the binding from the symbol they alias.
    // This is the first place we are able to copy this information.
    it->setExternal(SD.isExternal());
    MCELF::SetBinding(*it, MCELF::GetBinding(SD));

    StringRef Rest = AliasName.substr(Pos);
    if (!Symbol.isUndefined() && !Rest.startswith("@@@"))
      continue;

    // FIXME: produce a better error message.
    if (Symbol.isUndefined() && Rest.startswith("@@") &&
        !Rest.startswith("@@@"))
      report_fatal_error("A @@ version cannot be undefined");

    Renames.insert(std::make_pair(&Symbol, &Alias));
  }
}

void ELFObjectWriter::WriteSymbol(MCDataFragment *SymtabF,
                                  MCDataFragment *ShndxF,
                                  ELFSymbolData &MSD,
                                  const MCAsmLayout &Layout) {
  MCSymbolData &OrigData = *MSD.SymbolData;
  MCSymbolData &Data =
    Layout.getAssembler().getSymbolData(OrigData.getSymbol().AliasedSymbol());

  bool IsReserved = Data.isCommon() || Data.getSymbol().isAbsolute() ||
    Data.getSymbol().isVariable();

  uint8_t Binding = MCELF::GetBinding(OrigData);
  uint8_t Visibility = MCELF::GetVisibility(OrigData);
  uint8_t Type = MCELF::GetType(Data);

  uint8_t Info = (Binding << ELF_STB_Shift) | (Type << ELF_STT_Shift);
  uint8_t Other = Visibility;

  uint64_t Value = SymbolValue(Data, Layout);
  uint64_t Size = 0;

  assert(!(Data.isCommon() && !Data.isExternal()));

  const MCExpr *ESize = Data.getSize();
  if (ESize) {
    int64_t Res;
    if (!ESize->EvaluateAsAbsolute(Res, Layout))
      report_fatal_error("Size expression must be absolute.");
    Size = Res;
  }

  // Write out the symbol table entry
  WriteSymbolEntry(SymtabF, ShndxF, MSD.StringIndex, Info, Value,
                   Size, Other, MSD.SectionIndex, IsReserved);
}

void ELFObjectWriter::WriteSymbolTable(MCDataFragment *SymtabF,
                                       MCDataFragment *ShndxF,
                                       const MCAssembler &Asm,
                                       const MCAsmLayout &Layout,
                                     const SectionIndexMapTy &SectionIndexMap) {
  // The string table must be emitted first because we need the index
  // into the string table for all the symbol names.
  assert(StringTable.size() && "Missing string table");

  // FIXME: Make sure the start of the symbol table is aligned.

  // The first entry is the undefined symbol entry.
  WriteSymbolEntry(SymtabF, ShndxF, 0, 0, 0, 0, 0, 0, false);

  // Write the symbol table entries.
  LastLocalSymbolIndex = LocalSymbolData.size() + 1;
  for (unsigned i = 0, e = LocalSymbolData.size(); i != e; ++i) {
    ELFSymbolData &MSD = LocalSymbolData[i];
    WriteSymbol(SymtabF, ShndxF, MSD, Layout);
  }

  // Write out a symbol table entry for each regular section.
  for (MCAssembler::const_iterator i = Asm.begin(), e = Asm.end(); i != e;
       ++i) {
    const MCSectionELF &Section =
      static_cast<const MCSectionELF&>(i->getSection());
    if (Section.getType() == ELF::SHT_RELA ||
        Section.getType() == ELF::SHT_REL ||
        Section.getType() == ELF::SHT_STRTAB ||
        Section.getType() == ELF::SHT_SYMTAB)
      continue;
    WriteSymbolEntry(SymtabF, ShndxF, 0, ELF::STT_SECTION, 0, 0,
                     ELF::STV_DEFAULT, SectionIndexMap.lookup(&Section), false);
    LastLocalSymbolIndex++;
  }

  for (unsigned i = 0, e = ExternalSymbolData.size(); i != e; ++i) {
    ELFSymbolData &MSD = ExternalSymbolData[i];
    MCSymbolData &Data = *MSD.SymbolData;
    assert(((Data.getFlags() & ELF_STB_Global) ||
            (Data.getFlags() & ELF_STB_Weak)) &&
           "External symbol requires STB_GLOBAL or STB_WEAK flag");
    WriteSymbol(SymtabF, ShndxF, MSD, Layout);
    if (MCELF::GetBinding(Data) == ELF::STB_LOCAL)
      LastLocalSymbolIndex++;
  }

  for (unsigned i = 0, e = UndefinedSymbolData.size(); i != e; ++i) {
    ELFSymbolData &MSD = UndefinedSymbolData[i];
    MCSymbolData &Data = *MSD.SymbolData;
    WriteSymbol(SymtabF, ShndxF, MSD, Layout);
    if (MCELF::GetBinding(Data) == ELF::STB_LOCAL)
      LastLocalSymbolIndex++;
  }
}

const MCSymbol *ELFObjectWriter::SymbolToReloc(const MCAssembler &Asm,
                                               const MCValue &Target,
                                               const MCFragment &F, 
                                               const MCFixup &Fixup,
                                               bool IsPCRel) const {
  const MCSymbol &Symbol = Target.getSymA()->getSymbol();
  const MCSymbol &ASymbol = Symbol.AliasedSymbol();
  const MCSymbol *Renamed = Renames.lookup(&Symbol);
  const MCSymbolData &SD = Asm.getSymbolData(Symbol);

  if (ASymbol.isUndefined()) {
    if (Renamed)
      return Renamed;
    return &ASymbol;
  }

  if (SD.isExternal()) {
    if (Renamed)
      return Renamed;
    return &Symbol;
  }

  const MCSectionELF &Section =
    static_cast<const MCSectionELF&>(ASymbol.getSection());
  const SectionKind secKind = Section.getKind();

  if (secKind.isBSS())
    return ExplicitRelSym(Asm, Target, F, Fixup, IsPCRel);

  if (secKind.isThreadLocal()) {
    if (Renamed)
      return Renamed;
    return &Symbol;
  }

  MCSymbolRefExpr::VariantKind Kind = Target.getSymA()->getKind();
  const MCSectionELF &Sec2 =
    static_cast<const MCSectionELF&>(F.getParent()->getSection());

  if (&Sec2 != &Section &&
      (Kind == MCSymbolRefExpr::VK_PLT ||
       Kind == MCSymbolRefExpr::VK_GOTPCREL ||
       Kind == MCSymbolRefExpr::VK_GOTOFF)) {
    if (Renamed)
      return Renamed;
    return &Symbol;
  }

  if (Section.getFlags() & ELF::SHF_MERGE) {
    if (Target.getConstant() == 0)
      return ExplicitRelSym(Asm, Target, F, Fixup, IsPCRel);
    if (Renamed)
      return Renamed;
    return &Symbol;
  }

  return ExplicitRelSym(Asm, Target, F, Fixup, IsPCRel);

}


void ELFObjectWriter::RecordRelocation(const MCAssembler &Asm,
                                       const MCAsmLayout &Layout,
                                       const MCFragment *Fragment,
                                       const MCFixup &Fixup,
                                       MCValue Target,
                                       uint64_t &FixedValue) {
  int64_t Addend = 0;
  int Index = 0;
  int64_t Value = Target.getConstant();
  const MCSymbol *RelocSymbol = NULL;

  bool IsPCRel = isFixupKindPCRel(Asm, Fixup.getKind());
  if (!Target.isAbsolute()) {
    const MCSymbol &Symbol = Target.getSymA()->getSymbol();
    const MCSymbol &ASymbol = Symbol.AliasedSymbol();
    RelocSymbol = SymbolToReloc(Asm, Target, *Fragment, Fixup, IsPCRel);

    if (const MCSymbolRefExpr *RefB = Target.getSymB()) {
      const MCSymbol &SymbolB = RefB->getSymbol();
      MCSymbolData &SDB = Asm.getSymbolData(SymbolB);
      IsPCRel = true;

      // Offset of the symbol in the section
      int64_t a = Layout.getSymbolOffset(&SDB);

      // Ofeset of the relocation in the section
      int64_t b = Layout.getFragmentOffset(Fragment) + Fixup.getOffset();
      Value += b - a;
    }

    if (!RelocSymbol) {
      MCSymbolData &SD = Asm.getSymbolData(ASymbol);
      MCFragment *F = SD.getFragment();

      Index = F->getParent()->getOrdinal() + 1;

      // Offset of the symbol in the section
      Value += Layout.getSymbolOffset(&SD);
    } else {
      if (Asm.getSymbolData(Symbol).getFlags() & ELF_Other_Weakref)
        WeakrefUsedInReloc.insert(RelocSymbol);
      else
        UsedInReloc.insert(RelocSymbol);
      Index = -1;
    }
    Addend = Value;
    // Compensate for the addend on i386.
    if (is64Bit())
      Value = 0;
  }

  FixedValue = Value;
  unsigned Type = GetRelocType(Target, Fixup, IsPCRel,
                               (RelocSymbol != 0), Addend);

  uint64_t RelocOffset = Layout.getFragmentOffset(Fragment) +
    Fixup.getOffset();

  if (!hasRelocationAddend())
    Addend = 0;
  ELFRelocationEntry ERE(RelocOffset, Index, Type, RelocSymbol, Addend);
  Relocations[Fragment->getParent()].push_back(ERE);
}


uint64_t
ELFObjectWriter::getSymbolIndexInSymbolTable(const MCAssembler &Asm,
                                             const MCSymbol *S) {
  MCSymbolData &SD = Asm.getSymbolData(*S);
  return SD.getIndex();
}

bool ELFObjectWriter::isInSymtab(const MCAssembler &Asm,
                                 const MCSymbolData &Data,
                                 bool Used, bool Renamed) {
  if (Data.getFlags() & ELF_Other_Weakref)
    return false;

  if (Used)
    return true;

  if (Renamed)
    return false;

  const MCSymbol &Symbol = Data.getSymbol();

  if (Symbol.getName() == "_GLOBAL_OFFSET_TABLE_")
    return true;

  const MCSymbol &A = Symbol.AliasedSymbol();
  if (Symbol.isVariable() && !A.isVariable() && A.isUndefined())
    return false;

  bool IsGlobal = MCELF::GetBinding(Data) == ELF::STB_GLOBAL;
  if (!Symbol.isVariable() && Symbol.isUndefined() && !IsGlobal)
    return false;

  if (!Asm.isSymbolLinkerVisible(Symbol) && !Symbol.isUndefined())
    return false;

  if (Symbol.isTemporary())
    return false;

  return true;
}

bool ELFObjectWriter::isLocal(const MCSymbolData &Data, bool isSignature,
                              bool isUsedInReloc) {
  if (Data.isExternal())
    return false;

  const MCSymbol &Symbol = Data.getSymbol();
  const MCSymbol &RefSymbol = Symbol.AliasedSymbol();

  if (RefSymbol.isUndefined() && !RefSymbol.isVariable()) {
    if (isSignature && !isUsedInReloc)
      return true;

    return false;
  }

  return true;
}

void ELFObjectWriter::ComputeIndexMap(MCAssembler &Asm,
                                      SectionIndexMapTy &SectionIndexMap,
                                      const RelMapTy &RelMap) {
  unsigned Index = 1;
  for (MCAssembler::iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it) {
    const MCSectionELF &Section =
      static_cast<const MCSectionELF &>(it->getSection());
    if (Section.getType() != ELF::SHT_GROUP)
      continue;
    SectionIndexMap[&Section] = Index++;
  }

  for (MCAssembler::iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it) {
    const MCSectionELF &Section =
      static_cast<const MCSectionELF &>(it->getSection());
    if (Section.getType() == ELF::SHT_GROUP ||
        Section.getType() == ELF::SHT_REL ||
        Section.getType() == ELF::SHT_RELA)
      continue;
    SectionIndexMap[&Section] = Index++;
    const MCSectionELF *RelSection = RelMap.lookup(&Section);
    if (RelSection)
      SectionIndexMap[RelSection] = Index++;
  }
}

void ELFObjectWriter::ComputeSymbolTable(MCAssembler &Asm,
                                      const SectionIndexMapTy &SectionIndexMap,
                                         RevGroupMapTy RevGroupMap,
                                         unsigned NumRegularSections) {
  // FIXME: Is this the correct place to do this?
  if (NeedsGOT) {
    llvm::StringRef Name = "_GLOBAL_OFFSET_TABLE_";
    MCSymbol *Sym = Asm.getContext().GetOrCreateSymbol(Name);
    MCSymbolData &Data = Asm.getOrCreateSymbolData(*Sym);
    Data.setExternal(true);
    MCELF::SetBinding(Data, ELF::STB_GLOBAL);
  }

  // Index 0 is always the empty string.
  StringMap<uint64_t> StringIndexMap;
  StringTable += '\x00';

  // FIXME: We could optimize suffixes in strtab in the same way we
  // optimize them in shstrtab.

  // Add the data for the symbols.
  for (MCAssembler::symbol_iterator it = Asm.symbol_begin(),
         ie = Asm.symbol_end(); it != ie; ++it) {
    const MCSymbol &Symbol = it->getSymbol();

    bool Used = UsedInReloc.count(&Symbol);
    bool WeakrefUsed = WeakrefUsedInReloc.count(&Symbol);
    bool isSignature = RevGroupMap.count(&Symbol);

    if (!isInSymtab(Asm, *it,
                    Used || WeakrefUsed || isSignature,
                    Renames.count(&Symbol)))
      continue;

    ELFSymbolData MSD;
    MSD.SymbolData = it;
    const MCSymbol &RefSymbol = Symbol.AliasedSymbol();

    // Undefined symbols are global, but this is the first place we
    // are able to set it.
    bool Local = isLocal(*it, isSignature, Used);
    if (!Local && MCELF::GetBinding(*it) == ELF::STB_LOCAL) {
      MCSymbolData &SD = Asm.getSymbolData(RefSymbol);
      MCELF::SetBinding(*it, ELF::STB_GLOBAL);
      MCELF::SetBinding(SD, ELF::STB_GLOBAL);
    }

    if (RefSymbol.isUndefined() && !Used && WeakrefUsed)
      MCELF::SetBinding(*it, ELF::STB_WEAK);

    if (it->isCommon()) {
      assert(!Local);
      MSD.SectionIndex = ELF::SHN_COMMON;
    } else if (Symbol.isAbsolute() || RefSymbol.isVariable()) {
      MSD.SectionIndex = ELF::SHN_ABS;
    } else if (RefSymbol.isUndefined()) {
      if (isSignature && !Used)
        MSD.SectionIndex = SectionIndexMap.lookup(RevGroupMap[&Symbol]);
      else
        MSD.SectionIndex = ELF::SHN_UNDEF;
    } else {
      const MCSectionELF &Section =
        static_cast<const MCSectionELF&>(RefSymbol.getSection());
      MSD.SectionIndex = SectionIndexMap.lookup(&Section);
      if (MSD.SectionIndex >= ELF::SHN_LORESERVE)
        NeedsSymtabShndx = true;
      assert(MSD.SectionIndex && "Invalid section index!");
    }

    // The @@@ in symbol version is replaced with @ in undefined symbols and
    // @@ in defined ones.
    StringRef Name = Symbol.getName();
    SmallString<32> Buf;

    size_t Pos = Name.find("@@@");
    if (Pos != StringRef::npos) {
      Buf += Name.substr(0, Pos);
      unsigned Skip = MSD.SectionIndex == ELF::SHN_UNDEF ? 2 : 1;
      Buf += Name.substr(Pos + Skip);
      Name = Buf;
    }

    uint64_t &Entry = StringIndexMap[Name];
    if (!Entry) {
      Entry = StringTable.size();
      StringTable += Name;
      StringTable += '\x00';
    }
    MSD.StringIndex = Entry;
    if (MSD.SectionIndex == ELF::SHN_UNDEF)
      UndefinedSymbolData.push_back(MSD);
    else if (Local)
      LocalSymbolData.push_back(MSD);
    else
      ExternalSymbolData.push_back(MSD);
  }

  // Symbols are required to be in lexicographic order.
  array_pod_sort(LocalSymbolData.begin(), LocalSymbolData.end());
  array_pod_sort(ExternalSymbolData.begin(), ExternalSymbolData.end());
  array_pod_sort(UndefinedSymbolData.begin(), UndefinedSymbolData.end());

  // Set the symbol indices. Local symbols must come before all other
  // symbols with non-local bindings.
  unsigned Index = 1;
  for (unsigned i = 0, e = LocalSymbolData.size(); i != e; ++i)
    LocalSymbolData[i].SymbolData->setIndex(Index++);

  Index += NumRegularSections;

  for (unsigned i = 0, e = ExternalSymbolData.size(); i != e; ++i)
    ExternalSymbolData[i].SymbolData->setIndex(Index++);
  for (unsigned i = 0, e = UndefinedSymbolData.size(); i != e; ++i)
    UndefinedSymbolData[i].SymbolData->setIndex(Index++);
}

void ELFObjectWriter::CreateRelocationSections(MCAssembler &Asm,
                                               MCAsmLayout &Layout,
                                               RelMapTy &RelMap) {
  for (MCAssembler::const_iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it) {
    const MCSectionData &SD = *it;
    if (Relocations[&SD].empty())
      continue;

    MCContext &Ctx = Asm.getContext();
    const MCSectionELF &Section =
      static_cast<const MCSectionELF&>(SD.getSection());

    const StringRef SectionName = Section.getSectionName();
    std::string RelaSectionName = hasRelocationAddend() ? ".rela" : ".rel";
    RelaSectionName += SectionName;

    unsigned EntrySize;
    if (hasRelocationAddend())
      EntrySize = is64Bit() ? sizeof(ELF::Elf64_Rela) : sizeof(ELF::Elf32_Rela);
    else
      EntrySize = is64Bit() ? sizeof(ELF::Elf64_Rel) : sizeof(ELF::Elf32_Rel);

    const MCSectionELF *RelaSection =
      Ctx.getELFSection(RelaSectionName, hasRelocationAddend() ?
                        ELF::SHT_RELA : ELF::SHT_REL, 0,
                        SectionKind::getReadOnly(),
                        EntrySize, "");
    RelMap[&Section] = RelaSection;
    Asm.getOrCreateSectionData(*RelaSection);
  }
}

void ELFObjectWriter::WriteRelocations(MCAssembler &Asm, MCAsmLayout &Layout,
                                       const RelMapTy &RelMap) {
  for (MCAssembler::const_iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it) {
    const MCSectionData &SD = *it;
    const MCSectionELF &Section =
      static_cast<const MCSectionELF&>(SD.getSection());

    const MCSectionELF *RelaSection = RelMap.lookup(&Section);
    if (!RelaSection)
      continue;
    MCSectionData &RelaSD = Asm.getOrCreateSectionData(*RelaSection);
    RelaSD.setAlignment(is64Bit() ? 8 : 4);

    MCDataFragment *F = new MCDataFragment(&RelaSD);
    WriteRelocationsFragment(Asm, F, &*it);
  }
}

void ELFObjectWriter::WriteSecHdrEntry(uint32_t Name, uint32_t Type,
                                       uint64_t Flags, uint64_t Address,
                                       uint64_t Offset, uint64_t Size,
                                       uint32_t Link, uint32_t Info,
                                       uint64_t Alignment,
                                       uint64_t EntrySize) {
  Write32(Name);        // sh_name: index into string table
  Write32(Type);        // sh_type
  WriteWord(Flags);     // sh_flags
  WriteWord(Address);   // sh_addr
  WriteWord(Offset);    // sh_offset
  WriteWord(Size);      // sh_size
  Write32(Link);        // sh_link
  Write32(Info);        // sh_info
  WriteWord(Alignment); // sh_addralign
  WriteWord(EntrySize); // sh_entsize
}

void ELFObjectWriter::WriteRelocationsFragment(const MCAssembler &Asm,
                                               MCDataFragment *F,
                                               const MCSectionData *SD) {
  std::vector<ELFRelocationEntry> &Relocs = Relocations[SD];
  // sort by the r_offset just like gnu as does
  array_pod_sort(Relocs.begin(), Relocs.end());

  for (unsigned i = 0, e = Relocs.size(); i != e; ++i) {
    ELFRelocationEntry entry = Relocs[e - i - 1];

    if (!entry.Index)
      ;
    else if (entry.Index < 0)
      entry.Index = getSymbolIndexInSymbolTable(Asm, entry.Symbol);
    else
      entry.Index += LocalSymbolData.size();
    if (is64Bit()) {
      String64(*F, entry.r_offset);

      struct ELF::Elf64_Rela ERE64;
      ERE64.setSymbolAndType(entry.Index, entry.Type);
      String64(*F, ERE64.r_info);

      if (hasRelocationAddend())
        String64(*F, entry.r_addend);
    } else {
      String32(*F, entry.r_offset);

      struct ELF::Elf32_Rela ERE32;
      ERE32.setSymbolAndType(entry.Index, entry.Type);
      String32(*F, ERE32.r_info);

      if (hasRelocationAddend())
        String32(*F, entry.r_addend);
    }
  }
}

static int compareBySuffix(const void *a, const void *b) {
  const MCSectionELF *secA = *static_cast<const MCSectionELF* const *>(a);
  const MCSectionELF *secB = *static_cast<const MCSectionELF* const *>(b);
  const StringRef &NameA = secA->getSectionName();
  const StringRef &NameB = secB->getSectionName();
  const unsigned sizeA = NameA.size();
  const unsigned sizeB = NameB.size();
  const unsigned len = std::min(sizeA, sizeB);
  for (unsigned int i = 0; i < len; ++i) {
    char ca = NameA[sizeA - i - 1];
    char cb = NameB[sizeB - i - 1];
    if (ca != cb)
      return cb - ca;
  }

  return sizeB - sizeA;
}

void ELFObjectWriter::CreateMetadataSections(MCAssembler &Asm,
                                             MCAsmLayout &Layout,
                                             SectionIndexMapTy &SectionIndexMap,
                                             const RelMapTy &RelMap) {
  MCContext &Ctx = Asm.getContext();
  MCDataFragment *F;

  unsigned EntrySize = is64Bit() ? ELF::SYMENTRY_SIZE64 : ELF::SYMENTRY_SIZE32;

  // We construct .shstrtab, .symtab and .strtab in this order to match gnu as.
  const MCSectionELF *ShstrtabSection =
    Ctx.getELFSection(".shstrtab", ELF::SHT_STRTAB, 0,
                      SectionKind::getReadOnly());
  MCSectionData &ShstrtabSD = Asm.getOrCreateSectionData(*ShstrtabSection);
  ShstrtabSD.setAlignment(1);

  const MCSectionELF *SymtabSection =
    Ctx.getELFSection(".symtab", ELF::SHT_SYMTAB, 0,
                      SectionKind::getReadOnly(),
                      EntrySize, "");
  MCSectionData &SymtabSD = Asm.getOrCreateSectionData(*SymtabSection);
  SymtabSD.setAlignment(is64Bit() ? 8 : 4);

  MCSectionData *SymtabShndxSD = NULL;

  if (NeedsSymtabShndx) {
    const MCSectionELF *SymtabShndxSection =
      Ctx.getELFSection(".symtab_shndx", ELF::SHT_SYMTAB_SHNDX, 0,
                        SectionKind::getReadOnly(), 4, "");
    SymtabShndxSD = &Asm.getOrCreateSectionData(*SymtabShndxSection);
    SymtabShndxSD->setAlignment(4);
  }

  const MCSectionELF *StrtabSection;
  StrtabSection = Ctx.getELFSection(".strtab", ELF::SHT_STRTAB, 0,
                                    SectionKind::getReadOnly());
  MCSectionData &StrtabSD = Asm.getOrCreateSectionData(*StrtabSection);
  StrtabSD.setAlignment(1);

  ComputeIndexMap(Asm, SectionIndexMap, RelMap);

  ShstrtabIndex = SectionIndexMap.lookup(ShstrtabSection);
  SymbolTableIndex = SectionIndexMap.lookup(SymtabSection);
  StringTableIndex = SectionIndexMap.lookup(StrtabSection);

  // Symbol table
  F = new MCDataFragment(&SymtabSD);
  MCDataFragment *ShndxF = NULL;
  if (NeedsSymtabShndx) {
    ShndxF = new MCDataFragment(SymtabShndxSD);
  }
  WriteSymbolTable(F, ShndxF, Asm, Layout, SectionIndexMap);

  F = new MCDataFragment(&StrtabSD);
  F->getContents().append(StringTable.begin(), StringTable.end());

  F = new MCDataFragment(&ShstrtabSD);

  std::vector<const MCSectionELF*> Sections;
  for (MCAssembler::const_iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it) {
    const MCSectionELF &Section =
      static_cast<const MCSectionELF&>(it->getSection());
    Sections.push_back(&Section);
  }
  array_pod_sort(Sections.begin(), Sections.end(), compareBySuffix);

  // Section header string table.
  //
  // The first entry of a string table holds a null character so skip
  // section 0.
  uint64_t Index = 1;
  F->getContents() += '\x00';

  for (unsigned int I = 0, E = Sections.size(); I != E; ++I) {
    const MCSectionELF &Section = *Sections[I];

    StringRef Name = Section.getSectionName();
    if (I != 0) {
      StringRef PreviousName = Sections[I - 1]->getSectionName();
      if (PreviousName.endswith(Name)) {
        SectionStringTableIndex[&Section] = Index - Name.size() - 1;
        continue;
      }
    }
    // Remember the index into the string table so we can write it
    // into the sh_name field of the section header table.
    SectionStringTableIndex[&Section] = Index;

    Index += Name.size() + 1;
    F->getContents() += Name;
    F->getContents() += '\x00';
  }
}

void ELFObjectWriter::CreateIndexedSections(MCAssembler &Asm,
                                            MCAsmLayout &Layout,
                                            GroupMapTy &GroupMap,
                                            RevGroupMapTy &RevGroupMap,
                                            SectionIndexMapTy &SectionIndexMap,
                                            const RelMapTy &RelMap) {
  // Create the .note.GNU-stack section if needed.
  MCContext &Ctx = Asm.getContext();
  if (Asm.getNoExecStack()) {
    const MCSectionELF *GnuStackSection =
      Ctx.getELFSection(".note.GNU-stack", ELF::SHT_PROGBITS, 0,
                        SectionKind::getReadOnly());
    Asm.getOrCreateSectionData(*GnuStackSection);
  }

  // Build the groups
  for (MCAssembler::const_iterator it = Asm.begin(), ie = Asm.end();
       it != ie; ++it) {
    const MCSectionELF &Section =
      static_cast<const MCSectionELF&>(it->getSection());
    if (!(Section.getFlags() & ELF::SHF_GROUP))
      continue;

    const MCSymbol *SignatureSymbol = Section.getGroup();
    Asm.getOrCreateSymbolData(*SignatureSymbol);
    const MCSectionELF *&Group = RevGroupMap[SignatureSymbol];
    if (!Group) {
      Group = Ctx.CreateELFGroupSection();
      MCSectionData &Data = Asm.getOrCreateSectionData(*Group);
      Data.setAlignment(4);
      MCDataFragment *F = new MCDataFragment(&Data);
      String32(*F, ELF::GRP_COMDAT);
    }
    GroupMap[Group] = SignatureSymbol;
  }

  ComputeIndexMap(Asm, SectionIndexMap, RelMap);

  // Add sections to the groups
  for (MCAssembler::const_iterator it = Asm.begin(), ie = Asm.end();
       it != ie; ++it) {
    const MCSectionELF &Section =
      static_cast<const MCSectionELF&>(it->getSection());
    if (!(Section.getFlags() & ELF::SHF_GROUP))
      continue;
    const MCSectionELF *Group = RevGroupMap[Section.getGroup()];
    MCSectionData &Data = Asm.getOrCreateSectionData(*Group);
    // FIXME: we could use the previous fragment
    MCDataFragment *F = new MCDataFragment(&Data);
    unsigned Index = SectionIndexMap.lookup(&Section);
    String32(*F, Index);
  }
}

void ELFObjectWriter::WriteSection(MCAssembler &Asm,
                                   const SectionIndexMapTy &SectionIndexMap,
                                   uint32_t GroupSymbolIndex,
                                   uint64_t Offset, uint64_t Size,
                                   uint64_t Alignment,
                                   const MCSectionELF &Section) {
  uint64_t sh_link = 0;
  uint64_t sh_info = 0;

  switch(Section.getType()) {
  case ELF::SHT_DYNAMIC:
    sh_link = SectionStringTableIndex[&Section];
    sh_info = 0;
    break;

  case ELF::SHT_REL:
  case ELF::SHT_RELA: {
    const MCSectionELF *SymtabSection;
    const MCSectionELF *InfoSection;
    SymtabSection = Asm.getContext().getELFSection(".symtab", ELF::SHT_SYMTAB,
                                                   0,
                                                   SectionKind::getReadOnly());
    sh_link = SectionIndexMap.lookup(SymtabSection);
    assert(sh_link && ".symtab not found");

    // Remove ".rel" and ".rela" prefixes.
    unsigned SecNameLen = (Section.getType() == ELF::SHT_REL) ? 4 : 5;
    StringRef SectionName = Section.getSectionName().substr(SecNameLen);

    InfoSection = Asm.getContext().getELFSection(SectionName,
                                                 ELF::SHT_PROGBITS, 0,
                                                 SectionKind::getReadOnly());
    sh_info = SectionIndexMap.lookup(InfoSection);
    break;
  }

  case ELF::SHT_SYMTAB:
  case ELF::SHT_DYNSYM:
    sh_link = StringTableIndex;
    sh_info = LastLocalSymbolIndex;
    break;

  case ELF::SHT_SYMTAB_SHNDX:
    sh_link = SymbolTableIndex;
    break;

  case ELF::SHT_PROGBITS:
  case ELF::SHT_STRTAB:
  case ELF::SHT_NOBITS:
  case ELF::SHT_NOTE:
  case ELF::SHT_NULL:
  case ELF::SHT_ARM_ATTRIBUTES:
  case ELF::SHT_INIT_ARRAY:
  case ELF::SHT_FINI_ARRAY:
  case ELF::SHT_PREINIT_ARRAY:
  case ELF::SHT_X86_64_UNWIND:
    // Nothing to do.
    break;

  case ELF::SHT_GROUP: {
    sh_link = SymbolTableIndex;
    sh_info = GroupSymbolIndex;
    break;
  }

  default:
    assert(0 && "FIXME: sh_type value not supported!");
    break;
  }

  WriteSecHdrEntry(SectionStringTableIndex[&Section], Section.getType(),
                   Section.getFlags(), 0, Offset, Size, sh_link, sh_info,
                   Alignment, Section.getEntrySize());
}

bool ELFObjectWriter::IsELFMetaDataSection(const MCSectionData &SD) {
  return SD.getOrdinal() == ~UINT32_C(0) &&
    !SD.getSection().isVirtualSection();
}

uint64_t ELFObjectWriter::DataSectionSize(const MCSectionData &SD) {
  uint64_t Ret = 0;
  for (MCSectionData::const_iterator i = SD.begin(), e = SD.end(); i != e;
       ++i) {
    const MCFragment &F = *i;
    assert(F.getKind() == MCFragment::FT_Data);
    Ret += cast<MCDataFragment>(F).getContents().size();
  }
  return Ret;
}

uint64_t ELFObjectWriter::GetSectionFileSize(const MCAsmLayout &Layout,
                                             const MCSectionData &SD) {
  if (IsELFMetaDataSection(SD))
    return DataSectionSize(SD);
  return Layout.getSectionFileSize(&SD);
}

uint64_t ELFObjectWriter::GetSectionAddressSize(const MCAsmLayout &Layout,
                                                const MCSectionData &SD) {
  if (IsELFMetaDataSection(SD))
    return DataSectionSize(SD);
  return Layout.getSectionAddressSize(&SD);
}

void ELFObjectWriter::WriteDataSectionData(MCAssembler &Asm,
                                           const MCAsmLayout &Layout,
                                           const MCSectionELF &Section) {
  uint64_t FileOff = OS.tell();
  const MCSectionData &SD = Asm.getOrCreateSectionData(Section);

  uint64_t Padding = OffsetToAlignment(FileOff, SD.getAlignment());
  WriteZeros(Padding);
  FileOff += Padding;

  FileOff += GetSectionFileSize(Layout, SD);

  if (IsELFMetaDataSection(SD)) {
    for (MCSectionData::const_iterator i = SD.begin(), e = SD.end(); i != e;
         ++i) {
      const MCFragment &F = *i;
      assert(F.getKind() == MCFragment::FT_Data);
      WriteBytes(cast<MCDataFragment>(F).getContents().str());
    }
  } else {
    Asm.WriteSectionData(&SD, Layout);
  }
}

void ELFObjectWriter::WriteSectionHeader(MCAssembler &Asm,
                                         const GroupMapTy &GroupMap,
                                         const MCAsmLayout &Layout,
                                      const SectionIndexMapTy &SectionIndexMap,
                                   const SectionOffsetMapTy &SectionOffsetMap) {
  const unsigned NumSections = Asm.size() + 1;

  std::vector<const MCSectionELF*> Sections;
  Sections.resize(NumSections - 1);

  for (SectionIndexMapTy::const_iterator i=
         SectionIndexMap.begin(), e = SectionIndexMap.end(); i != e; ++i) {
    const std::pair<const MCSectionELF*, uint32_t> &p = *i;
    Sections[p.second - 1] = p.first;
  }

  // Null section first.
  uint64_t FirstSectionSize =
    NumSections >= ELF::SHN_LORESERVE ? NumSections : 0;
  uint32_t FirstSectionLink =
    ShstrtabIndex >= ELF::SHN_LORESERVE ? ShstrtabIndex : 0;
  WriteSecHdrEntry(0, 0, 0, 0, 0, FirstSectionSize, FirstSectionLink, 0, 0, 0);

  for (unsigned i = 0; i < NumSections - 1; ++i) {
    const MCSectionELF &Section = *Sections[i];
    const MCSectionData &SD = Asm.getOrCreateSectionData(Section);
    uint32_t GroupSymbolIndex;
    if (Section.getType() != ELF::SHT_GROUP)
      GroupSymbolIndex = 0;
    else
      GroupSymbolIndex = getSymbolIndexInSymbolTable(Asm,
                                                     GroupMap.lookup(&Section));

    uint64_t Size = GetSectionAddressSize(Layout, SD);

    WriteSection(Asm, SectionIndexMap, GroupSymbolIndex,
                 SectionOffsetMap.lookup(&Section), Size,
                 SD.getAlignment(), Section);
  }
}

void ELFObjectWriter::ComputeSectionOrder(MCAssembler &Asm,
                                  std::vector<const MCSectionELF*> &Sections) {
  for (MCAssembler::iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it) {
    const MCSectionELF &Section =
      static_cast<const MCSectionELF &>(it->getSection());
    if (Section.getType() == ELF::SHT_GROUP)
      Sections.push_back(&Section);
  }

  for (MCAssembler::iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it) {
    const MCSectionELF &Section =
      static_cast<const MCSectionELF &>(it->getSection());
    if (Section.getType() != ELF::SHT_GROUP &&
        Section.getType() != ELF::SHT_REL &&
        Section.getType() != ELF::SHT_RELA)
      Sections.push_back(&Section);
  }

  for (MCAssembler::iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it) {
    const MCSectionELF &Section =
      static_cast<const MCSectionELF &>(it->getSection());
    if (Section.getType() == ELF::SHT_REL ||
        Section.getType() == ELF::SHT_RELA)
      Sections.push_back(&Section);
  }
}

void ELFObjectWriter::WriteObject(MCAssembler &Asm,
                                  const MCAsmLayout &Layout) {
  GroupMapTy GroupMap;
  RevGroupMapTy RevGroupMap;
  SectionIndexMapTy SectionIndexMap;

  unsigned NumUserSections = Asm.size();

  DenseMap<const MCSectionELF*, const MCSectionELF*> RelMap;
  CreateRelocationSections(Asm, const_cast<MCAsmLayout&>(Layout), RelMap);

  const unsigned NumUserAndRelocSections = Asm.size();
  CreateIndexedSections(Asm, const_cast<MCAsmLayout&>(Layout), GroupMap,
                        RevGroupMap, SectionIndexMap, RelMap);
  const unsigned AllSections = Asm.size();
  const unsigned NumIndexedSections = AllSections - NumUserAndRelocSections;

  unsigned NumRegularSections = NumUserSections + NumIndexedSections;

  // Compute symbol table information.
  ComputeSymbolTable(Asm, SectionIndexMap, RevGroupMap, NumRegularSections);


  WriteRelocations(Asm, const_cast<MCAsmLayout&>(Layout), RelMap);

  CreateMetadataSections(const_cast<MCAssembler&>(Asm),
                         const_cast<MCAsmLayout&>(Layout),
                         SectionIndexMap,
                         RelMap);

  uint64_t NaturalAlignment = is64Bit() ? 8 : 4;
  uint64_t HeaderSize = is64Bit() ? sizeof(ELF::Elf64_Ehdr) :
                                    sizeof(ELF::Elf32_Ehdr);
  uint64_t FileOff = HeaderSize;

  std::vector<const MCSectionELF*> Sections;
  ComputeSectionOrder(Asm, Sections);
  unsigned NumSections = Sections.size();
  SectionOffsetMapTy SectionOffsetMap;
  for (unsigned i = 0; i < NumRegularSections + 1; ++i) {
    const MCSectionELF &Section = *Sections[i];
    const MCSectionData &SD = Asm.getOrCreateSectionData(Section);

    FileOff = RoundUpToAlignment(FileOff, SD.getAlignment());

    // Remember the offset into the file for this section.
    SectionOffsetMap[&Section] = FileOff;

    // Get the size of the section in the output file (including padding).
    FileOff += GetSectionFileSize(Layout, SD);
  }

  FileOff = RoundUpToAlignment(FileOff, NaturalAlignment);

  const unsigned SectionHeaderOffset = FileOff - HeaderSize;

  uint64_t SectionHeaderEntrySize = is64Bit() ?
    sizeof(ELF::Elf64_Shdr) : sizeof(ELF::Elf32_Shdr);
  FileOff += (NumSections + 1) * SectionHeaderEntrySize;

  for (unsigned i = NumRegularSections + 1; i < NumSections; ++i) {
    const MCSectionELF &Section = *Sections[i];
    const MCSectionData &SD = Asm.getOrCreateSectionData(Section);

    FileOff = RoundUpToAlignment(FileOff, SD.getAlignment());

    // Remember the offset into the file for this section.
    SectionOffsetMap[&Section] = FileOff;

    // Get the size of the section in the output file (including padding).
    FileOff += GetSectionFileSize(Layout, SD);
  }

  // Write out the ELF header ...
  WriteHeader(SectionHeaderOffset, NumSections + 1);

  // ... then the regular sections ...
  // + because of .shstrtab
  for (unsigned i = 0; i < NumRegularSections + 1; ++i)
    WriteDataSectionData(Asm, Layout, *Sections[i]);

  FileOff = OS.tell();
  uint64_t Padding = OffsetToAlignment(FileOff, NaturalAlignment);
  WriteZeros(Padding);

  // ... then the section header table ...
  WriteSectionHeader(Asm, GroupMap, Layout, SectionIndexMap,
                     SectionOffsetMap);

  FileOff = OS.tell();

  // ... and then the remainting sections ...
  for (unsigned i = NumRegularSections + 1; i < NumSections; ++i)
    WriteDataSectionData(Asm, Layout, *Sections[i]);
}

bool
ELFObjectWriter::IsSymbolRefDifferenceFullyResolvedImpl(const MCAssembler &Asm,
                                                      const MCSymbolData &DataA,
                                                      const MCFragment &FB,
                                                      bool InSet,
                                                      bool IsPCRel) const {
  if (DataA.getFlags() & ELF_STB_Weak)
    return false;
  return MCObjectWriter::IsSymbolRefDifferenceFullyResolvedImpl(
                                                 Asm, DataA, FB,InSet, IsPCRel);
}

MCObjectWriter *llvm::createELFObjectWriter(MCELFObjectTargetWriter *MOTW,
                                            raw_ostream &OS,
                                            bool IsLittleEndian) {
  switch (MOTW->getEMachine()) {
    case ELF::EM_386:
    case ELF::EM_X86_64:
      return new X86ELFObjectWriter(MOTW, OS, IsLittleEndian); break;
    case ELF::EM_ARM:
      return new ARMELFObjectWriter(MOTW, OS, IsLittleEndian); break;
    case ELF::EM_MBLAZE:
      return new MBlazeELFObjectWriter(MOTW, OS, IsLittleEndian); break;
    default: llvm_unreachable("Unsupported architecture"); break;
  }
}


/// START OF SUBCLASSES for ELFObjectWriter
//===- ARMELFObjectWriter -------------------------------------------===//

ARMELFObjectWriter::ARMELFObjectWriter(MCELFObjectTargetWriter *MOTW,
                                       raw_ostream &_OS,
                                       bool IsLittleEndian)
  : ELFObjectWriter(MOTW, _OS, IsLittleEndian)
{}

ARMELFObjectWriter::~ARMELFObjectWriter()
{}

// FIXME: get the real EABI Version from the Triple.
void ARMELFObjectWriter::WriteEFlags() {
  Write32(ELF::EF_ARM_EABIMASK & DefaultEABIVersion);
}

// In ARM, _MergedGlobals and other most symbols get emitted directly.
// I.e. not as an offset to a section symbol.
// This code is an approximation of what ARM/gcc does.

STATISTIC(PCRelCount, "Total number of PIC Relocations");
STATISTIC(NonPCRelCount, "Total number of non-PIC relocations");

const MCSymbol *ARMELFObjectWriter::ExplicitRelSym(const MCAssembler &Asm,
                                                   const MCValue &Target,
                                                   const MCFragment &F,
                                                   const MCFixup &Fixup,
                                                   bool IsPCRel) const {
  const MCSymbol &Symbol = Target.getSymA()->getSymbol();
  bool EmitThisSym = false;

  const MCSectionELF &Section =
    static_cast<const MCSectionELF&>(Symbol.getSection());
  const SectionKind secKind = Section.getKind();
  const MCSymbolRefExpr::VariantKind Kind = Target.getSymA()->getKind();
  MCSymbolRefExpr::VariantKind Kind2; 
  Kind2 = Target.getSymB() ?  Target.getSymB()->getKind() :
    MCSymbolRefExpr::VK_None;
  bool InNormalSection = true;
  unsigned RelocType = 0;
  RelocType = GetRelocTypeInner(Target, Fixup, IsPCRel);

  DEBUG(dbgs() << "considering symbol "
        << Section.getSectionName() << "/"
        << Symbol.getName() << "/"
        << " Rel:" << (unsigned)RelocType
        << " Kind: " << (int)Kind << "/" << (int)Kind2
        << " Tmp:"
        << Symbol.isAbsolute() << "/" << Symbol.isDefined() << "/"
        << Symbol.isVariable() << "/" << Symbol.isTemporary()
        << " Counts:" << PCRelCount << "/" << NonPCRelCount << "\n");

  if (IsPCRel || ForceARMElfPIC) { ++PCRelCount;
    switch (RelocType) {
    default:
      // Most relocation types are emitted as explicit symbols
      InNormalSection =
        StringSwitch<bool>(Section.getSectionName())
        .Case(".data.rel.ro.local", false)
        .Case(".data.rel", false)
        .Case(".bss", false)
        .Default(true);
      EmitThisSym = true;
      break;
    case ELF::R_ARM_ABS32:
      // But things get strange with R_ARM_ABS32
      // In this case, most things that go in .rodata show up
      // as section relative relocations
      InNormalSection =
        StringSwitch<bool>(Section.getSectionName())
        .Case(".data.rel.ro.local", false)
        .Case(".data.rel", false)
        .Case(".rodata", false)
        .Case(".bss", false)
        .Default(true);
      EmitThisSym = false;
      break;
    }
  } else {
    NonPCRelCount++;
    InNormalSection =
      StringSwitch<bool>(Section.getSectionName())
      .Case(".data.rel.ro.local", false)
      .Case(".rodata", false)
      .Case(".data.rel", false)
      .Case(".bss", false)
      .Default(true);

    switch (RelocType) {
    default: EmitThisSym = true; break;
    case ELF::R_ARM_ABS32: EmitThisSym = false; break;
    }
  }

  if (EmitThisSym)
    return &Symbol;
  if (! Symbol.isTemporary() && InNormalSection) {
    return &Symbol;
  }
  return NULL;
}

// Need to examine the Fixup when determining whether to 
// emit the relocation as an explicit symbol or as a section relative
// offset
unsigned ARMELFObjectWriter::GetRelocType(const MCValue &Target,
                                          const MCFixup &Fixup,
                                          bool IsPCRel,
                                          bool IsRelocWithSymbol,
                                          int64_t Addend) {
  MCSymbolRefExpr::VariantKind Modifier = Target.isAbsolute() ?
    MCSymbolRefExpr::VK_None : Target.getSymA()->getKind();

  unsigned Type = GetRelocTypeInner(Target, Fixup, IsPCRel);

  if (RelocNeedsGOT(Modifier))
    NeedsGOT = true;
  
  return Type;
}

unsigned ARMELFObjectWriter::GetRelocTypeInner(const MCValue &Target,
                                               const MCFixup &Fixup,
                                               bool IsPCRel) const  {
  MCSymbolRefExpr::VariantKind Modifier = Target.isAbsolute() ?
    MCSymbolRefExpr::VK_None : Target.getSymA()->getKind();

  unsigned Type = 0;
  if (IsPCRel) {
    switch ((unsigned)Fixup.getKind()) {
    default: assert(0 && "Unimplemented");
    case FK_Data_4:
      switch (Modifier) {
      default: llvm_unreachable("Unsupported Modifier");
      case MCSymbolRefExpr::VK_None:
        Type = ELF::R_ARM_REL32;
        break;
      case MCSymbolRefExpr::VK_ARM_TLSGD:
        assert(0 && "unimplemented");
        break;
      case MCSymbolRefExpr::VK_ARM_GOTTPOFF:
        Type = ELF::R_ARM_TLS_IE32;
        break;
      }
      break;
    case ARM::fixup_arm_uncondbranch:
      switch (Modifier) {
      case MCSymbolRefExpr::VK_ARM_PLT:
        Type = ELF::R_ARM_PLT32;
        break;
      default:
        Type = ELF::R_ARM_CALL;
        break;
      }
      break;
    case ARM::fixup_arm_condbranch:
      Type = ELF::R_ARM_JUMP24;
      break;
    case ARM::fixup_arm_movt_hi16:
    case ARM::fixup_arm_movt_hi16_pcrel:
      Type = ELF::R_ARM_MOVT_PREL;
      break;
    case ARM::fixup_arm_movw_lo16:
    case ARM::fixup_arm_movw_lo16_pcrel:
      Type = ELF::R_ARM_MOVW_PREL_NC;
      break;
    case ARM::fixup_t2_movt_hi16:
    case ARM::fixup_t2_movt_hi16_pcrel:
      Type = ELF::R_ARM_THM_MOVT_PREL;
      break;
    case ARM::fixup_t2_movw_lo16:
    case ARM::fixup_t2_movw_lo16_pcrel:
      Type = ELF::R_ARM_THM_MOVW_PREL_NC;
      break;
    }
  } else {
    switch ((unsigned)Fixup.getKind()) {
    default: llvm_unreachable("invalid fixup kind!");
    case FK_Data_4:
      switch (Modifier) {
      default: llvm_unreachable("Unsupported Modifier"); break;
      case MCSymbolRefExpr::VK_ARM_GOT:
        Type = ELF::R_ARM_GOT_BREL;
        break;
      case MCSymbolRefExpr::VK_ARM_TLSGD:
        Type = ELF::R_ARM_TLS_GD32;
        break;
      case MCSymbolRefExpr::VK_ARM_TPOFF:
        Type = ELF::R_ARM_TLS_LE32;
        break;
      case MCSymbolRefExpr::VK_ARM_GOTTPOFF:
        Type = ELF::R_ARM_TLS_IE32;
        break;
      case MCSymbolRefExpr::VK_None:
        Type = ELF::R_ARM_ABS32;
        break;
      case MCSymbolRefExpr::VK_ARM_GOTOFF:
        Type = ELF::R_ARM_GOTOFF32;
        break;
      }
      break;
    case ARM::fixup_arm_ldst_pcrel_12:
    case ARM::fixup_arm_pcrel_10:
    case ARM::fixup_arm_adr_pcrel_12:
    case ARM::fixup_arm_thumb_bl:
    case ARM::fixup_arm_thumb_cb:
    case ARM::fixup_arm_thumb_cp:
    case ARM::fixup_arm_thumb_br:
      assert(0 && "Unimplemented");
      break;
    case ARM::fixup_arm_uncondbranch:
      Type = ELF::R_ARM_CALL;
      break;
    case ARM::fixup_arm_condbranch:
      Type = ELF::R_ARM_JUMP24;
      break;
    case ARM::fixup_arm_movt_hi16:
      Type = ELF::R_ARM_MOVT_ABS;
      break;
    case ARM::fixup_arm_movw_lo16:
      Type = ELF::R_ARM_MOVW_ABS_NC;
      break;
    case ARM::fixup_t2_movt_hi16:
      Type = ELF::R_ARM_THM_MOVT_ABS;
      break;
    case ARM::fixup_t2_movw_lo16:
      Type = ELF::R_ARM_THM_MOVW_ABS_NC;
      break;
    }
  }

  return Type;
}

//===- MBlazeELFObjectWriter -------------------------------------------===//

MBlazeELFObjectWriter::MBlazeELFObjectWriter(MCELFObjectTargetWriter *MOTW,
                                             raw_ostream &_OS,
                                             bool IsLittleEndian)
  : ELFObjectWriter(MOTW, _OS, IsLittleEndian) {
}

MBlazeELFObjectWriter::~MBlazeELFObjectWriter() {
}

unsigned MBlazeELFObjectWriter::GetRelocType(const MCValue &Target,
                                             const MCFixup &Fixup,
                                             bool IsPCRel,
                                             bool IsRelocWithSymbol,
                                             int64_t Addend) {
  // determine the type of the relocation
  unsigned Type;
  if (IsPCRel) {
    switch ((unsigned)Fixup.getKind()) {
    default:
      llvm_unreachable("Unimplemented");
    case FK_PCRel_4:
      Type = ELF::R_MICROBLAZE_64_PCREL;
      break;
    case FK_PCRel_2:
      Type = ELF::R_MICROBLAZE_32_PCREL;
      break;
    }
  } else {
    switch ((unsigned)Fixup.getKind()) {
    default: llvm_unreachable("invalid fixup kind!");
    case FK_Data_4:
      Type = ((IsRelocWithSymbol || Addend !=0)
              ? ELF::R_MICROBLAZE_32
              : ELF::R_MICROBLAZE_64);
      break;
    case FK_Data_2:
      Type = ELF::R_MICROBLAZE_32;
      break;
    }
  }
  return Type;
}

//===- X86ELFObjectWriter -------------------------------------------===//


X86ELFObjectWriter::X86ELFObjectWriter(MCELFObjectTargetWriter *MOTW,
                                       raw_ostream &_OS,
                                       bool IsLittleEndian)
  : ELFObjectWriter(MOTW, _OS, IsLittleEndian)
{}

X86ELFObjectWriter::~X86ELFObjectWriter()
{}

unsigned X86ELFObjectWriter::GetRelocType(const MCValue &Target,
                                          const MCFixup &Fixup,
                                          bool IsPCRel,
                                          bool IsRelocWithSymbol,
                                          int64_t Addend) {
  // determine the type of the relocation

  MCSymbolRefExpr::VariantKind Modifier = Target.isAbsolute() ?
    MCSymbolRefExpr::VK_None : Target.getSymA()->getKind();
  unsigned Type;
  if (is64Bit()) {
    if (IsPCRel) {
      switch ((unsigned)Fixup.getKind()) {
      default: llvm_unreachable("invalid fixup kind!");

      case FK_Data_8: Type = ELF::R_X86_64_PC64; break;
      case FK_Data_4: Type = ELF::R_X86_64_PC32; break;
      case FK_Data_2: Type = ELF::R_X86_64_PC16; break;

      case FK_PCRel_8:
        assert(Modifier == MCSymbolRefExpr::VK_None);
        Type = ELF::R_X86_64_PC64;
        break;
      case X86::reloc_signed_4byte:
      case X86::reloc_riprel_4byte_movq_load:
      case X86::reloc_riprel_4byte:
      case FK_PCRel_4:
        switch (Modifier) {
        default:
          llvm_unreachable("Unimplemented");
        case MCSymbolRefExpr::VK_None:
          Type = ELF::R_X86_64_PC32;
          break;
        case MCSymbolRefExpr::VK_PLT:
          Type = ELF::R_X86_64_PLT32;
          break;
        case MCSymbolRefExpr::VK_GOTPCREL:
          Type = ELF::R_X86_64_GOTPCREL;
          break;
        case MCSymbolRefExpr::VK_GOTTPOFF:
          Type = ELF::R_X86_64_GOTTPOFF;
        break;
        case MCSymbolRefExpr::VK_TLSGD:
          Type = ELF::R_X86_64_TLSGD;
          break;
        case MCSymbolRefExpr::VK_TLSLD:
          Type = ELF::R_X86_64_TLSLD;
          break;
        }
        break;
      case FK_PCRel_2:
        assert(Modifier == MCSymbolRefExpr::VK_None);
        Type = ELF::R_X86_64_PC16;
        break;
      case FK_PCRel_1:
        assert(Modifier == MCSymbolRefExpr::VK_None);
        Type = ELF::R_X86_64_PC8;
        break;
      }
    } else {
      switch ((unsigned)Fixup.getKind()) {
      default: llvm_unreachable("invalid fixup kind!");
      case FK_Data_8: Type = ELF::R_X86_64_64; break;
      case X86::reloc_signed_4byte:
        assert(isInt<32>(Target.getConstant()));
        switch (Modifier) {
        default:
          llvm_unreachable("Unimplemented");
        case MCSymbolRefExpr::VK_None:
          Type = ELF::R_X86_64_32S;
          break;
        case MCSymbolRefExpr::VK_GOT:
          Type = ELF::R_X86_64_GOT32;
          break;
        case MCSymbolRefExpr::VK_GOTPCREL:
          Type = ELF::R_X86_64_GOTPCREL;
          break;
        case MCSymbolRefExpr::VK_TPOFF:
          Type = ELF::R_X86_64_TPOFF32;
          break;
        case MCSymbolRefExpr::VK_DTPOFF:
          Type = ELF::R_X86_64_DTPOFF32;
          break;
        }
        break;
      case FK_Data_4:
        Type = ELF::R_X86_64_32;
        break;
      case FK_Data_2: Type = ELF::R_X86_64_16; break;
      case FK_PCRel_1:
      case FK_Data_1: Type = ELF::R_X86_64_8; break;
      }
    }
  } else {
    if (IsPCRel) {
      switch (Modifier) {
      default:
        llvm_unreachable("Unimplemented");
      case MCSymbolRefExpr::VK_None:
        Type = ELF::R_386_PC32;
        break;
      case MCSymbolRefExpr::VK_PLT:
        Type = ELF::R_386_PLT32;
        break;
      }
    } else {
      switch ((unsigned)Fixup.getKind()) {
      default: llvm_unreachable("invalid fixup kind!");

      case X86::reloc_global_offset_table:
        Type = ELF::R_386_GOTPC;
        break;

      // FIXME: Should we avoid selecting reloc_signed_4byte in 32 bit mode
      // instead?
      case X86::reloc_signed_4byte:
      case FK_PCRel_4:
      case FK_Data_4:
        switch (Modifier) {
        default:
          llvm_unreachable("Unimplemented");
        case MCSymbolRefExpr::VK_None:
          Type = ELF::R_386_32;
          break;
        case MCSymbolRefExpr::VK_GOT:
          Type = ELF::R_386_GOT32;
          break;
        case MCSymbolRefExpr::VK_GOTOFF:
          Type = ELF::R_386_GOTOFF;
          break;
        case MCSymbolRefExpr::VK_TLSGD:
          Type = ELF::R_386_TLS_GD;
          break;
        case MCSymbolRefExpr::VK_TPOFF:
          Type = ELF::R_386_TLS_LE_32;
          break;
        case MCSymbolRefExpr::VK_INDNTPOFF:
          Type = ELF::R_386_TLS_IE;
          break;
        case MCSymbolRefExpr::VK_NTPOFF:
          Type = ELF::R_386_TLS_LE;
          break;
        case MCSymbolRefExpr::VK_GOTNTPOFF:
          Type = ELF::R_386_TLS_GOTIE;
          break;
        case MCSymbolRefExpr::VK_TLSLDM:
          Type = ELF::R_386_TLS_LDM;
          break;
        case MCSymbolRefExpr::VK_DTPOFF:
          Type = ELF::R_386_TLS_LDO_32;
          break;
        }
        break;
      case FK_Data_2: Type = ELF::R_386_16; break;
      case FK_PCRel_1:
      case FK_Data_1: Type = ELF::R_386_8; break;
      }
    }
  }

  if (RelocNeedsGOT(Modifier))
    NeedsGOT = true;

  return Type;
}
