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

#include "llvm/MC/ELFObjectWriter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFSymbolFlags.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ELF.h"
#include "llvm/Target/TargetAsmBackend.h"

#include "../Target/X86/X86FixupKinds.h"

#include <vector>
using namespace llvm;

namespace {

  class ELFObjectWriterImpl {
    static bool isFixupKindX86PCRel(unsigned Kind) {
      switch (Kind) {
      default:
        return false;
      case X86::reloc_pcrel_1byte:
      case X86::reloc_pcrel_4byte:
      case X86::reloc_riprel_4byte:
      case X86::reloc_riprel_4byte_movq_load:
        return true;
      }
    }

    static bool isFixupKindX86RIPRel(unsigned Kind) {
      return Kind == X86::reloc_riprel_4byte ||
        Kind == X86::reloc_riprel_4byte_movq_load;
    }


    /// ELFSymbolData - Helper struct for containing some precomputed information
    /// on symbols.
    struct ELFSymbolData {
      MCSymbolData *SymbolData;
      uint64_t StringIndex;
      uint32_t SectionIndex;

      // Support lexicographic sorting.
      bool operator<(const ELFSymbolData &RHS) const {
        const std::string &Name = SymbolData->getSymbol().getName();
        return Name < RHS.SymbolData->getSymbol().getName();
      }
    };

    /// @name Relocation Data
    /// @{

    struct ELFRelocationEntry {
      // Make these big enough for both 32-bit and 64-bit
      uint64_t r_offset;
      uint64_t r_info;
      uint64_t r_addend;

      // Support lexicographic sorting.
      bool operator<(const ELFRelocationEntry &RE) const {
        return RE.r_offset < r_offset;
      }
    };

    llvm::DenseMap<const MCSectionData*,
                   std::vector<ELFRelocationEntry> > Relocations;
    DenseMap<const MCSection*, uint64_t> SectionStringTableIndex;

    /// @}
    /// @name Symbol Table Data
    /// @{

    SmallString<256> StringTable;
    std::vector<ELFSymbolData> LocalSymbolData;
    std::vector<ELFSymbolData> ExternalSymbolData;
    std::vector<ELFSymbolData> UndefinedSymbolData;

    /// @}

    ELFObjectWriter *Writer;

    raw_ostream &OS;

    // This holds the current offset into the object file.
    size_t FileOff;

    unsigned Is64Bit : 1;

    bool HasRelocationAddend;

    // This holds the symbol table index of the last local symbol.
    unsigned LastLocalSymbolIndex;
    // This holds the .strtab section index.
    unsigned StringTableIndex;

    unsigned ShstrtabIndex;

  public:
    ELFObjectWriterImpl(ELFObjectWriter *_Writer, bool _Is64Bit,
                        bool _HasRelAddend)
      : Writer(_Writer), OS(Writer->getStream()), FileOff(0),
        Is64Bit(_Is64Bit), HasRelocationAddend(_HasRelAddend) {
    }

    void Write8(uint8_t Value) { Writer->Write8(Value); }
    void Write16(uint16_t Value) { Writer->Write16(Value); }
    void Write32(uint32_t Value) { Writer->Write32(Value); }
    void Write64(uint64_t Value) { Writer->Write64(Value); }
    void WriteZeros(unsigned N) { Writer->WriteZeros(N); }
    void WriteBytes(StringRef Str, unsigned ZeroFillSize = 0) {
      Writer->WriteBytes(Str, ZeroFillSize);
    }

    void WriteWord(uint64_t W) {
      if (Is64Bit) {
        Writer->Write64(W);
      } else {
        Writer->Write32(W);
      }
    }

    void String8(char *buf, uint8_t Value) {
      buf[0] = Value;
    }

    void StringLE16(char *buf, uint16_t Value) {
      buf[0] = char(Value >> 0);
      buf[1] = char(Value >> 8);
    }

    void StringLE32(char *buf, uint32_t Value) {
      buf[0] = char(Value >> 0);
      buf[1] = char(Value >> 8);
      buf[2] = char(Value >> 16);
      buf[3] = char(Value >> 24);
    }

    void StringLE64(char *buf, uint64_t Value) {
      buf[0] = char(Value >> 0);
      buf[1] = char(Value >> 8);
      buf[2] = char(Value >> 16);
      buf[3] = char(Value >> 24);
      buf[4] = char(Value >> 32);
      buf[5] = char(Value >> 40);
      buf[6] = char(Value >> 48);
      buf[7] = char(Value >> 56);
    }

    void StringBE16(char *buf ,uint16_t Value) {
      buf[0] = char(Value >> 8);
      buf[1] = char(Value >> 0);
    }

    void StringBE32(char *buf, uint32_t Value) {
      buf[0] = char(Value >> 24);
      buf[1] = char(Value >> 16);
      buf[2] = char(Value >> 8);
      buf[3] = char(Value >> 0);
    }

    void StringBE64(char *buf, uint64_t Value) {
      buf[0] = char(Value >> 56);
      buf[1] = char(Value >> 48);
      buf[2] = char(Value >> 40);
      buf[3] = char(Value >> 32);
      buf[4] = char(Value >> 24);
      buf[5] = char(Value >> 16);
      buf[6] = char(Value >> 8);
      buf[7] = char(Value >> 0);
    }

    void String16(char *buf, uint16_t Value) {
      if (Writer->isLittleEndian())
        StringLE16(buf, Value);
      else
        StringBE16(buf, Value);
    }

    void String32(char *buf, uint32_t Value) {
      if (Writer->isLittleEndian())
        StringLE32(buf, Value);
      else
        StringBE32(buf, Value);
    }

    void String64(char *buf, uint64_t Value) {
      if (Writer->isLittleEndian())
        StringLE64(buf, Value);
      else
        StringBE64(buf, Value);
    }

    void WriteHeader(uint64_t SectionDataSize, unsigned NumberOfSections);

    void WriteSymbolEntry(MCDataFragment *F, uint64_t name, uint8_t info,
                          uint64_t value, uint64_t size,
                          uint8_t other, uint16_t shndx);

    void WriteSymbol(MCDataFragment *F, ELFSymbolData &MSD,
                     const MCAsmLayout &Layout);

    void WriteSymbolTable(MCDataFragment *F, const MCAssembler &Asm,
                          const MCAsmLayout &Layout);

    void RecordRelocation(const MCAssembler &Asm, const MCAsmLayout &Layout,
                          const MCFragment *Fragment, const MCFixup &Fixup,
                          MCValue Target, uint64_t &FixedValue);

    // XXX-PERF: this should be cached
    uint64_t getNumOfLocalSymbols(const MCAssembler &Asm) {
      std::vector<const MCSymbol*> Local;

      uint64_t Index = 0;
      for (MCAssembler::const_symbol_iterator it = Asm.symbol_begin(),
             ie = Asm.symbol_end(); it != ie; ++it) {
        const MCSymbol &Symbol = it->getSymbol();

        // Ignore non-linker visible symbols.
        if (!Asm.isSymbolLinkerVisible(Symbol))
          continue;

        if (it->isExternal() || Symbol.isUndefined())
          continue;

        Index++;
      }

      return Index;
    }

    uint64_t getSymbolIndexInSymbolTable(MCAssembler &Asm, const MCSymbol *S);

    /// ComputeSymbolTable - Compute the symbol table data
    ///
    /// \param StringTable [out] - The string table data.
    /// \param StringIndexMap [out] - Map from symbol names to offsets in the
    /// string table.
    void ComputeSymbolTable(MCAssembler &Asm);

    void WriteRelocation(MCAssembler &Asm, MCAsmLayout &Layout,
                         const MCSectionData &SD);

    void WriteRelocations(MCAssembler &Asm, MCAsmLayout &Layout) {
      for (MCAssembler::const_iterator it = Asm.begin(),
             ie = Asm.end(); it != ie; ++it) {
        WriteRelocation(Asm, Layout, *it);
      }
    }

    void CreateMetadataSections(MCAssembler &Asm, MCAsmLayout &Layout);

    void ExecutePostLayoutBinding(MCAssembler &Asm) {}

    void WriteSecHdrEntry(uint32_t Name, uint32_t Type, uint64_t Flags,
                          uint64_t Address, uint64_t Offset,
                          uint64_t Size, uint32_t Link, uint32_t Info,
                          uint64_t Alignment, uint64_t EntrySize);

    void WriteRelocationsFragment(const MCAssembler &Asm, MCDataFragment *F,
                                  const MCSectionData *SD);

    void WriteObject(const MCAssembler &Asm, const MCAsmLayout &Layout);
  };

}

// Emit the ELF header.
void ELFObjectWriterImpl::WriteHeader(uint64_t SectionDataSize,
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

  Write8(Is64Bit ? ELF::ELFCLASS64 : ELF::ELFCLASS32); // e_ident[EI_CLASS]

  // e_ident[EI_DATA]
  Write8(Writer->isLittleEndian() ? ELF::ELFDATA2LSB : ELF::ELFDATA2MSB);

  Write8(ELF::EV_CURRENT);        // e_ident[EI_VERSION]
  Write8(ELF::ELFOSABI_LINUX);    // e_ident[EI_OSABI]
  Write8(0);                  // e_ident[EI_ABIVERSION]

  WriteZeros(ELF::EI_NIDENT - ELF::EI_PAD);

  Write16(ELF::ET_REL);             // e_type

  // FIXME: Make this configurable
  Write16(Is64Bit ? ELF::EM_X86_64 : ELF::EM_386); // e_machine = target

  Write32(ELF::EV_CURRENT);         // e_version
  WriteWord(0);                    // e_entry, no entry point in .o file
  WriteWord(0);                    // e_phoff, no program header for .o
  WriteWord(SectionDataSize + (Is64Bit ? sizeof(ELF::Elf64_Ehdr) :
            sizeof(ELF::Elf32_Ehdr)));  // e_shoff = sec hdr table off in bytes

  // FIXME: Make this configurable.
  Write32(0);   // e_flags = whatever the target wants

  // e_ehsize = ELF header size
  Write16(Is64Bit ? sizeof(ELF::Elf64_Ehdr) : sizeof(ELF::Elf32_Ehdr));

  Write16(0);                  // e_phentsize = prog header entry size
  Write16(0);                  // e_phnum = # prog header entries = 0

  // e_shentsize = Section header entry size
  Write16(Is64Bit ? sizeof(ELF::Elf64_Shdr) : sizeof(ELF::Elf32_Shdr));

  // e_shnum     = # of section header ents
  Write16(NumberOfSections);

  // e_shstrndx  = Section # of '.shstrtab'
  Write16(ShstrtabIndex);
}

void ELFObjectWriterImpl::WriteSymbolEntry(MCDataFragment *F, uint64_t name,
                                           uint8_t info, uint64_t value,
                                           uint64_t size, uint8_t other,
                                           uint16_t shndx) {
  if (Is64Bit) {
    char buf[8];

    String32(buf, name);
    F->getContents() += StringRef(buf, 4); // st_name

    String8(buf, info);
    F->getContents() += StringRef(buf, 1);  // st_info

    String8(buf, other);
    F->getContents() += StringRef(buf, 1); // st_other

    String16(buf, shndx);
    F->getContents() += StringRef(buf, 2); // st_shndx

    String64(buf, value);
    F->getContents() += StringRef(buf, 8); // st_value

    String64(buf, size);
    F->getContents() += StringRef(buf, 8);  // st_size
  } else {
    char buf[4];

    String32(buf, name);
    F->getContents() += StringRef(buf, 4);  // st_name

    String32(buf, value);
    F->getContents() += StringRef(buf, 4); // st_value

    String32(buf, size);
    F->getContents() += StringRef(buf, 4);  // st_size

    String8(buf, info);
    F->getContents() += StringRef(buf, 1);  // st_info

    String8(buf, other);
    F->getContents() += StringRef(buf, 1); // st_other

    String16(buf, shndx);
    F->getContents() += StringRef(buf, 2); // st_shndx
  }
}

void ELFObjectWriterImpl::WriteSymbol(MCDataFragment *F, ELFSymbolData &MSD,
                                      const MCAsmLayout &Layout) {
  MCSymbolData &Data = *MSD.SymbolData;
  uint8_t Info = (Data.getFlags() & 0xff);
  uint8_t Other = ((Data.getFlags() & 0xf00) >> ELF_STV_Shift);
  uint64_t Value = 0;
  uint64_t Size = 0;
  const MCExpr *ESize;

  if (Data.isCommon() && Data.isExternal())
    Value = Data.getCommonAlignment();

  ESize = Data.getSize();
  if (Data.getSize()) {
    MCValue Res;
    if (ESize->getKind() == MCExpr::Binary) {
      const MCBinaryExpr *BE = static_cast<const MCBinaryExpr *>(ESize);

      if (BE->EvaluateAsRelocatable(Res, &Layout)) {
        MCSymbolData &A =
          Layout.getAssembler().getSymbolData(Res.getSymA()->getSymbol());
        MCSymbolData &B =
          Layout.getAssembler().getSymbolData(Res.getSymB()->getSymbol());

        Size = Layout.getSymbolAddress(&A) - Layout.getSymbolAddress(&B);
        Value = Layout.getSymbolAddress(&Data);
      }
    } else if (ESize->getKind() == MCExpr::Constant) {
      Size = static_cast<const MCConstantExpr *>(ESize)->getValue();
    } else {
      assert(0 && "Unsupported size expression");
    }
  }

  // Write out the symbol table entry
  WriteSymbolEntry(F, MSD.StringIndex, Info, Value,
                   Size, Other, MSD.SectionIndex);
}

void ELFObjectWriterImpl::WriteSymbolTable(MCDataFragment *F,
                                           const MCAssembler &Asm,
                                           const MCAsmLayout &Layout) {
  // The string table must be emitted first because we need the index
  // into the string table for all the symbol names.
  assert(StringTable.size() && "Missing string table");

  // FIXME: Make sure the start of the symbol table is aligned.

  // The first entry is the undefined symbol entry.
  unsigned EntrySize = Is64Bit ? ELF::SYMENTRY_SIZE64 : ELF::SYMENTRY_SIZE32;
  F->getContents().append(EntrySize, '\x00');

  // Write the symbol table entries.
  LastLocalSymbolIndex = LocalSymbolData.size() + 1;
  for (unsigned i = 0, e = LocalSymbolData.size(); i != e; ++i) {
    ELFSymbolData &MSD = LocalSymbolData[i];
    WriteSymbol(F, MSD, Layout);
  }

  // Write out a symbol table entry for each section.
  // leaving out the just added .symtab which is at
  // the very end
  unsigned Index = 1;
  for (MCAssembler::const_iterator it = Asm.begin(),
       ie = Asm.end(); it != ie; ++it, ++Index) {
    const MCSectionELF &Section =
      static_cast<const MCSectionELF&>(it->getSection());
    // Leave out relocations so we don't have indexes within
    // the relocations messed up
    if (Section.getType() == ELF::SHT_RELA || Section.getType() == ELF::SHT_REL)
      continue;
    if (Index == Asm.size())
      continue;
    WriteSymbolEntry(F, 0, ELF::STT_SECTION, 0, 0, ELF::STV_DEFAULT, Index);
    LastLocalSymbolIndex++;
  }

  for (unsigned i = 0, e = ExternalSymbolData.size(); i != e; ++i) {
    ELFSymbolData &MSD = ExternalSymbolData[i];
    MCSymbolData &Data = *MSD.SymbolData;
    assert((Data.getFlags() & ELF_STB_Global) &&
           "External symbol requires STB_GLOBAL flag");
    WriteSymbol(F, MSD, Layout);
    if (Data.getFlags() & ELF_STB_Local)
      LastLocalSymbolIndex++;
  }

  for (unsigned i = 0, e = UndefinedSymbolData.size(); i != e; ++i) {
    ELFSymbolData &MSD = UndefinedSymbolData[i];
    MCSymbolData &Data = *MSD.SymbolData;
    Data.setFlags(Data.getFlags() | ELF_STB_Global);
    WriteSymbol(F, MSD, Layout);
    if (Data.getFlags() & ELF_STB_Local)
      LastLocalSymbolIndex++;
  }
}

// FIXME: this is currently X86/X86_64 only
void ELFObjectWriterImpl::RecordRelocation(const MCAssembler &Asm,
                                           const MCAsmLayout &Layout,
                                           const MCFragment *Fragment,
                                           const MCFixup &Fixup,
                                           MCValue Target,
                                           uint64_t &FixedValue) {
  unsigned IsPCRel = isFixupKindX86PCRel(Fixup.getKind());

  uint64_t FixupOffset =
    Layout.getFragmentOffset(Fragment) + Fixup.getOffset();
  int64_t Value;
  int64_t Addend = 0;
  unsigned Index = 0;
  unsigned Type;

  Value = Target.getConstant();

  if (!Target.isAbsolute()) {
    const MCSymbol *Symbol = &Target.getSymA()->getSymbol();
    MCSymbolData &SD = Asm.getSymbolData(*Symbol);
    const MCSymbolData *Base = Asm.getAtom(Layout, &SD);

    if (Base) {
      if (MCFragment *F = SD.getFragment())
        Index = F->getParent()->getOrdinal() + getNumOfLocalSymbols(Asm) + 1;
      else
        Index = getSymbolIndexInSymbolTable(const_cast<MCAssembler &>(Asm), Symbol);
      if (Base != &SD)
        Value += Layout.getSymbolAddress(&SD) - Layout.getSymbolAddress(Base);
      Addend = Value;
      Value = 0;
    } else {
      MCFragment *F = SD.getFragment();
      if (F) {
        // Index of the section in .symtab against this symbol
        // is being relocated + 2 (empty section + abs. symbols).
        Index = F->getParent()->getOrdinal() + getNumOfLocalSymbols(Asm) + 1;

        MCSectionData *FSD = F->getParent();
        // Offset of the symbol in the section
        Addend = Layout.getSymbolAddress(&SD) - Layout.getSectionAddress(FSD);
      } else {
        FixedValue = Value;
        return;
      }
    }
  }

  // determine the type of the relocation
  if (Is64Bit) {
    if (IsPCRel) {
      Type = ELF::R_X86_64_PC32;
    } else {
      switch ((unsigned)Fixup.getKind()) {
      default: llvm_unreachable("invalid fixup kind!");
      case FK_Data_8: Type = ELF::R_X86_64_64; break;
      case X86::reloc_pcrel_4byte:
      case FK_Data_4:
        // check that the offset fits within a signed long
        if (isInt<32>(Target.getConstant()))
          Type = ELF::R_X86_64_32S;
        else
          Type = ELF::R_X86_64_32;
        break;
      case FK_Data_2: Type = ELF::R_X86_64_16; break;
      case X86::reloc_pcrel_1byte:
      case FK_Data_1: Type = ELF::R_X86_64_8; break;
      }
    }
  } else {
    if (IsPCRel) {
      Type = ELF::R_386_PC32;
    } else {
      switch ((unsigned)Fixup.getKind()) {
      default: llvm_unreachable("invalid fixup kind!");
      case X86::reloc_pcrel_4byte:
      case FK_Data_4: Type = ELF::R_386_32; break;
      case FK_Data_2: Type = ELF::R_386_16; break;
      case X86::reloc_pcrel_1byte:
      case FK_Data_1: Type = ELF::R_386_8; break;
      }
    }
  }

  FixedValue = Value;

  ELFRelocationEntry ERE;

  if (Is64Bit) {
    struct ELF::Elf64_Rela ERE64;
    ERE64.setSymbolAndType(Index, Type);
    ERE.r_info = ERE64.r_info;
  } else {
    struct ELF::Elf32_Rela ERE32;
    ERE32.setSymbolAndType(Index, Type);
    ERE.r_info = ERE32.r_info;
  }

  ERE.r_offset = FixupOffset;

  if (HasRelocationAddend)
    ERE.r_addend = Addend;
  else
    ERE.r_addend = 0; // Silence compiler warning.

  Relocations[Fragment->getParent()].push_back(ERE);
}

// XXX-PERF: this should be cached
uint64_t ELFObjectWriterImpl::getSymbolIndexInSymbolTable(MCAssembler &Asm,
                                                          const MCSymbol *S) {
  std::vector<ELFSymbolData> Local;
  std::vector<ELFSymbolData> External;
  std::vector<ELFSymbolData> Undefined;

  for (MCAssembler::symbol_iterator it = Asm.symbol_begin(),
         ie = Asm.symbol_end(); it != ie; ++it) {
    const MCSymbol &Symbol = it->getSymbol();

    // Ignore non-linker visible symbols.
    if (!Asm.isSymbolLinkerVisible(Symbol))
      continue;

    if (it->isExternal() || Symbol.isUndefined())
      continue;

    ELFSymbolData MSD;
    MSD.SymbolData = it;

    Local.push_back(MSD);
  }
  for (MCAssembler::symbol_iterator it = Asm.symbol_begin(),
         ie = Asm.symbol_end(); it != ie; ++it) {
    const MCSymbol &Symbol = it->getSymbol();

    // Ignore non-linker visible symbols.
    if (!Asm.isSymbolLinkerVisible(Symbol))
      continue;

    if (!it->isExternal() && !Symbol.isUndefined())
      continue;

    ELFSymbolData MSD;
    MSD.SymbolData = it;

    if (Symbol.isUndefined())
      Undefined.push_back(MSD);
    else
      External.push_back(MSD);
  }

  array_pod_sort(Local.begin(), Local.end());
  array_pod_sort(External.begin(), External.end());
  array_pod_sort(Undefined.begin(), Undefined.end());

  for (unsigned i = 0, e = Local.size(); i != e; ++i)
    if (&Local[i].SymbolData->getSymbol() == S)
      return i + /* empty symbol */ 1;
  for (unsigned i = 0, e = External.size(); i != e; ++i)
    if (&External[i].SymbolData->getSymbol() == S)
      return i + Local.size() + Asm.size() + /* empty symbol */ 1;
  for (unsigned i = 0, e = Undefined.size(); i != e; ++i)
    if (&Undefined[i].SymbolData->getSymbol() == S)
      return i + Local.size() + External.size() + Asm.size() + /* empty symbol */ 1;

  llvm_unreachable("Cannot find symbol which should exist!");
}

void ELFObjectWriterImpl::ComputeSymbolTable(MCAssembler &Asm) {
  // Build section lookup table.
  DenseMap<const MCSection*, uint8_t> SectionIndexMap;
  unsigned Index = 1;
  for (MCAssembler::iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it, ++Index)
    SectionIndexMap[&it->getSection()] = Index;

  // Index 0 is always the empty string.
  StringMap<uint64_t> StringIndexMap;
  StringTable += '\x00';

  // Add the data for local symbols.
  for (MCAssembler::symbol_iterator it = Asm.symbol_begin(),
         ie = Asm.symbol_end(); it != ie; ++it) {
    const MCSymbol &Symbol = it->getSymbol();

    // Ignore non-linker visible symbols.
    if (!Asm.isSymbolLinkerVisible(Symbol))
      continue;

    if (it->isExternal() || Symbol.isUndefined())
      continue;

    uint64_t &Entry = StringIndexMap[Symbol.getName()];
    if (!Entry) {
      Entry = StringTable.size();
      StringTable += Symbol.getName();
      StringTable += '\x00';
    }

    ELFSymbolData MSD;
    MSD.SymbolData = it;
    MSD.StringIndex = Entry;

    if (Symbol.isAbsolute()) {
      MSD.SectionIndex = ELF::SHN_ABS;
      LocalSymbolData.push_back(MSD);
    } else {
      MSD.SectionIndex = SectionIndexMap.lookup(&Symbol.getSection());
      assert(MSD.SectionIndex && "Invalid section index!");
      LocalSymbolData.push_back(MSD);
    }
  }

  // Now add non-local symbols.
  for (MCAssembler::symbol_iterator it = Asm.symbol_begin(),
         ie = Asm.symbol_end(); it != ie; ++it) {
    const MCSymbol &Symbol = it->getSymbol();

    // Ignore non-linker visible symbols.
    if (!Asm.isSymbolLinkerVisible(Symbol))
      continue;

    if (!it->isExternal() && !Symbol.isUndefined())
      continue;

    uint64_t &Entry = StringIndexMap[Symbol.getName()];
    if (!Entry) {
      Entry = StringTable.size();
      StringTable += Symbol.getName();
      StringTable += '\x00';
    }

    ELFSymbolData MSD;
    MSD.SymbolData = it;
    MSD.StringIndex = Entry;

    if (Symbol.isUndefined()) {
      MSD.SectionIndex = ELF::SHN_UNDEF;
      // XXX: for some reason we dont Emit* this
      it->setFlags(it->getFlags() | ELF_STB_Global);
      UndefinedSymbolData.push_back(MSD);
    } else if (Symbol.isAbsolute()) {
      MSD.SectionIndex = ELF::SHN_ABS;
      ExternalSymbolData.push_back(MSD);
    } else if (it->isCommon()) {
      MSD.SectionIndex = ELF::SHN_COMMON;
      ExternalSymbolData.push_back(MSD);
    } else {
      MSD.SectionIndex = SectionIndexMap.lookup(&Symbol.getSection());
      assert(MSD.SectionIndex && "Invalid section index!");
      ExternalSymbolData.push_back(MSD);
    }
  }

  // Symbols are required to be in lexicographic order.
  array_pod_sort(LocalSymbolData.begin(), LocalSymbolData.end());
  array_pod_sort(ExternalSymbolData.begin(), ExternalSymbolData.end());
  array_pod_sort(UndefinedSymbolData.begin(), UndefinedSymbolData.end());

  // Set the symbol indices. Local symbols must come before all other
  // symbols with non-local bindings.
  Index = 0;
  for (unsigned i = 0, e = LocalSymbolData.size(); i != e; ++i)
    LocalSymbolData[i].SymbolData->setIndex(Index++);
  for (unsigned i = 0, e = ExternalSymbolData.size(); i != e; ++i)
    ExternalSymbolData[i].SymbolData->setIndex(Index++);
  for (unsigned i = 0, e = UndefinedSymbolData.size(); i != e; ++i)
    UndefinedSymbolData[i].SymbolData->setIndex(Index++);
}

void ELFObjectWriterImpl::WriteRelocation(MCAssembler &Asm, MCAsmLayout &Layout,
                                          const MCSectionData &SD) {
  if (!Relocations[&SD].empty()) {
    MCContext &Ctx = Asm.getContext();
    const MCSection *RelaSection;
    const MCSectionELF &Section =
      static_cast<const MCSectionELF&>(SD.getSection());

    const StringRef SectionName = Section.getSectionName();
    std::string RelaSectionName = HasRelocationAddend ? ".rela" : ".rel";
    RelaSectionName += SectionName;

    unsigned EntrySize;
    if (HasRelocationAddend)
      EntrySize = Is64Bit ? sizeof(ELF::Elf64_Rela) : sizeof(ELF::Elf32_Rela);
    else
      EntrySize = Is64Bit ? sizeof(ELF::Elf64_Rel) : sizeof(ELF::Elf32_Rel);

    RelaSection = Ctx.getELFSection(RelaSectionName, HasRelocationAddend ?
                                    ELF::SHT_RELA : ELF::SHT_REL, 0,
                                    SectionKind::getReadOnly(),
                                    false, EntrySize);

    MCSectionData &RelaSD = Asm.getOrCreateSectionData(*RelaSection);
    RelaSD.setAlignment(1);

    MCDataFragment *F = new MCDataFragment(&RelaSD);

    WriteRelocationsFragment(Asm, F, &SD);

    Asm.AddSectionToTheEnd(RelaSD, Layout);
  }
}

void ELFObjectWriterImpl::WriteSecHdrEntry(uint32_t Name, uint32_t Type,
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

void ELFObjectWriterImpl::WriteRelocationsFragment(const MCAssembler &Asm,
                                                   MCDataFragment *F,
                                                   const MCSectionData *SD) {
  std::vector<ELFRelocationEntry> &Relocs = Relocations[SD];
  // sort by the r_offset just like gnu as does
  array_pod_sort(Relocs.begin(), Relocs.end());

  for (unsigned i = 0, e = Relocs.size(); i != e; ++i) {
    ELFRelocationEntry entry = Relocs[e - i - 1];

    unsigned WordSize = Is64Bit ? 8 : 4;
    F->getContents() += StringRef((const char *)&entry.r_offset, WordSize);
    F->getContents() += StringRef((const char *)&entry.r_info, WordSize);

    if (HasRelocationAddend)
      F->getContents() += StringRef((const char *)&entry.r_addend, WordSize);
  }
}

void ELFObjectWriterImpl::CreateMetadataSections(MCAssembler &Asm,
                                                 MCAsmLayout &Layout) {
  MCContext &Ctx = Asm.getContext();
  MCDataFragment *F;

  WriteRelocations(Asm, Layout);

  const MCSection *SymtabSection;
  unsigned EntrySize = Is64Bit ? ELF::SYMENTRY_SIZE64 : ELF::SYMENTRY_SIZE32;

  SymtabSection = Ctx.getELFSection(".symtab", ELF::SHT_SYMTAB, 0,
                                    SectionKind::getReadOnly(),
                                    false, EntrySize);

  MCSectionData &SymtabSD = Asm.getOrCreateSectionData(*SymtabSection);

  SymtabSD.setAlignment(Is64Bit ? 8 : 4);

  F = new MCDataFragment(&SymtabSD);

  // Symbol table
  WriteSymbolTable(F, Asm, Layout);
  Asm.AddSectionToTheEnd(SymtabSD, Layout);

  const MCSection *StrtabSection;
  StrtabSection = Ctx.getELFSection(".strtab", ELF::SHT_STRTAB, 0,
                                    SectionKind::getReadOnly(), false);

  MCSectionData &StrtabSD = Asm.getOrCreateSectionData(*StrtabSection);
  StrtabSD.setAlignment(1);

  // FIXME: This isn't right. If the sections get rearranged this will
  // be wrong. We need a proper lookup.
  StringTableIndex = Asm.size();

  F = new MCDataFragment(&StrtabSD);
  F->getContents().append(StringTable.begin(), StringTable.end());
  Asm.AddSectionToTheEnd(StrtabSD, Layout);

  const MCSection *ShstrtabSection;
  ShstrtabSection = Ctx.getELFSection(".shstrtab", ELF::SHT_STRTAB, 0,
                                      SectionKind::getReadOnly(), false);

  MCSectionData &ShstrtabSD = Asm.getOrCreateSectionData(*ShstrtabSection);
  ShstrtabSD.setAlignment(1);

  F = new MCDataFragment(&ShstrtabSD);

  // FIXME: This isn't right. If the sections get rearranged this will
  // be wrong. We need a proper lookup.
  ShstrtabIndex = Asm.size();

  // Section header string table.
  //
  // The first entry of a string table holds a null character so skip
  // section 0.
  uint64_t Index = 1;
  F->getContents() += '\x00';

  for (MCAssembler::const_iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it) {
    const MCSectionELF &Section =
      static_cast<const MCSectionELF&>(it->getSection());

    // Remember the index into the string table so we can write it
    // into the sh_name field of the section header table.
    SectionStringTableIndex[&it->getSection()] = Index;

    Index += Section.getSectionName().size() + 1;
    F->getContents() += Section.getSectionName();
    F->getContents() += '\x00';
  }

  Asm.AddSectionToTheEnd(ShstrtabSD, Layout);
}

void ELFObjectWriterImpl::WriteObject(const MCAssembler &Asm,
                                      const MCAsmLayout &Layout) {
  // Compute symbol table information.
  ComputeSymbolTable(const_cast<MCAssembler&>(Asm));

  CreateMetadataSections(const_cast<MCAssembler&>(Asm),
                         const_cast<MCAsmLayout&>(Layout));

  // Add 1 for the null section.
  unsigned NumSections = Asm.size() + 1;

  uint64_t SectionDataSize = 0;

  for (MCAssembler::const_iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it) {
    const MCSectionData &SD = *it;

    // Get the size of the section in the output file (including padding).
    uint64_t Size = Layout.getSectionFileSize(&SD);
    SectionDataSize += Size;
  }

  // Write out the ELF header ...
  WriteHeader(SectionDataSize, NumSections);
  FileOff = Is64Bit ? sizeof(ELF::Elf64_Ehdr) : sizeof(ELF::Elf32_Ehdr);

  // ... then all of the sections ...
  DenseMap<const MCSection*, uint64_t> SectionOffsetMap;

  DenseMap<const MCSection*, uint8_t> SectionIndexMap;

  unsigned Index = 1;
  for (MCAssembler::const_iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it) {
    // Remember the offset into the file for this section.
    SectionOffsetMap[&it->getSection()] = FileOff;

    SectionIndexMap[&it->getSection()] = Index++;

    const MCSectionData &SD = *it;
    FileOff += Layout.getSectionFileSize(&SD);

    Asm.WriteSectionData(it, Layout, Writer);
  }

  // ... and then the section header table.
  // Should we align the section header table?
  //
  // Null section first.
  WriteSecHdrEntry(0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

  for (MCAssembler::const_iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it) {
    const MCSectionData &SD = *it;
    const MCSectionELF &Section =
      static_cast<const MCSectionELF&>(SD.getSection());

    uint64_t sh_link = 0;
    uint64_t sh_info = 0;

    switch(Section.getType()) {
    case ELF::SHT_DYNAMIC:
      sh_link = SectionStringTableIndex[&it->getSection()];
      sh_info = 0;
      break;

    case ELF::SHT_REL:
    case ELF::SHT_RELA: {
      const MCSection *SymtabSection;
      const MCSection *InfoSection;

      SymtabSection = Asm.getContext().getELFSection(".symtab", ELF::SHT_SYMTAB, 0,
                                                     SectionKind::getReadOnly(),
                                                     false);
      sh_link = SectionIndexMap[SymtabSection];

      // Remove ".rel" and ".rela" prefixes.
      unsigned SecNameLen = (Section.getType() == ELF::SHT_REL) ? 4 : 5;
      StringRef SectionName = Section.getSectionName().substr(SecNameLen);

      InfoSection = Asm.getContext().getELFSection(SectionName,
                                                   ELF::SHT_PROGBITS, 0,
                                                   SectionKind::getReadOnly(),
                                                   false);
      sh_info = SectionIndexMap[InfoSection];
      break;
    }

    case ELF::SHT_SYMTAB:
    case ELF::SHT_DYNSYM:
      sh_link = StringTableIndex;
      sh_info = LastLocalSymbolIndex;
      break;

    case ELF::SHT_PROGBITS:
    case ELF::SHT_STRTAB:
    case ELF::SHT_NOBITS:
      // Nothing to do.
      break;

    case ELF::SHT_HASH:
    case ELF::SHT_GROUP:
    case ELF::SHT_SYMTAB_SHNDX:
    default:
      assert(0 && "FIXME: sh_type value not supported!");
      break;
    }

    WriteSecHdrEntry(SectionStringTableIndex[&it->getSection()],
                     Section.getType(), Section.getFlags(),
                     Layout.getSectionAddress(&SD),
                     SectionOffsetMap.lookup(&SD.getSection()),
                     Layout.getSectionSize(&SD), sh_link,
                     sh_info, SD.getAlignment(),
                     Section.getEntrySize());
  }
}

ELFObjectWriter::ELFObjectWriter(raw_ostream &OS,
                                 bool Is64Bit,
                                 bool IsLittleEndian,
                                 bool HasRelocationAddend)
  : MCObjectWriter(OS, IsLittleEndian)
{
  Impl = new ELFObjectWriterImpl(this, Is64Bit, HasRelocationAddend);
}

ELFObjectWriter::~ELFObjectWriter() {
  delete (ELFObjectWriterImpl*) Impl;
}

void ELFObjectWriter::ExecutePostLayoutBinding(MCAssembler &Asm) {
  ((ELFObjectWriterImpl*) Impl)->ExecutePostLayoutBinding(Asm);
}

void ELFObjectWriter::RecordRelocation(const MCAssembler &Asm,
                                       const MCAsmLayout &Layout,
                                       const MCFragment *Fragment,
                                       const MCFixup &Fixup, MCValue Target,
                                       uint64_t &FixedValue) {
  ((ELFObjectWriterImpl*) Impl)->RecordRelocation(Asm, Layout, Fragment, Fixup,
                                                  Target, FixedValue);
}

void ELFObjectWriter::WriteObject(const MCAssembler &Asm,
                                  const MCAsmLayout &Layout) {
  ((ELFObjectWriterImpl*) Impl)->WriteObject(Asm, Layout);
}
