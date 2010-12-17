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

#include "llvm/ADT/SmallPtrSet.h"
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
#include "../Target/ARM/ARMFixupKinds.h"

#include <vector>
using namespace llvm;

static unsigned GetType(const MCSymbolData &SD) {
  uint32_t Type = (SD.getFlags() & (0xf << ELF_STT_Shift)) >> ELF_STT_Shift;
  assert(Type == ELF::STT_NOTYPE || Type == ELF::STT_OBJECT ||
         Type == ELF::STT_FUNC || Type == ELF::STT_SECTION ||
         Type == ELF::STT_FILE || Type == ELF::STT_COMMON ||
         Type == ELF::STT_TLS);
  return Type;
}

static unsigned GetBinding(const MCSymbolData &SD) {
  uint32_t Binding = (SD.getFlags() & (0xf << ELF_STB_Shift)) >> ELF_STB_Shift;
  assert(Binding == ELF::STB_LOCAL || Binding == ELF::STB_GLOBAL ||
         Binding == ELF::STB_WEAK);
  return Binding;
}

static void SetBinding(MCSymbolData &SD, unsigned Binding) {
  assert(Binding == ELF::STB_LOCAL || Binding == ELF::STB_GLOBAL ||
         Binding == ELF::STB_WEAK);
  uint32_t OtherFlags = SD.getFlags() & ~(0xf << ELF_STB_Shift);
  SD.setFlags(OtherFlags | (Binding << ELF_STB_Shift));
}

static unsigned GetVisibility(MCSymbolData &SD) {
  unsigned Visibility =
    (SD.getFlags() & (0xf << ELF_STV_Shift)) >> ELF_STV_Shift;
  assert(Visibility == ELF::STV_DEFAULT || Visibility == ELF::STV_INTERNAL ||
         Visibility == ELF::STV_HIDDEN || Visibility == ELF::STV_PROTECTED);
  return Visibility;
}


static bool RelocNeedsGOT(MCSymbolRefExpr::VariantKind Variant) {
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

namespace {
  class ELFObjectWriter : public MCObjectWriter {
  protected:
    /*static bool isFixupKindX86RIPRel(unsigned Kind) {
      return Kind == X86::reloc_riprel_4byte ||
        Kind == X86::reloc_riprel_4byte_movq_load;
    }*/


    /// ELFSymbolData - Helper struct for containing some precomputed information
    /// on symbols.
    struct ELFSymbolData {
      MCSymbolData *SymbolData;
      uint64_t StringIndex;
      uint32_t SectionIndex;

      // Support lexicographic sorting.
      bool operator<(const ELFSymbolData &RHS) const {
        if (GetType(*SymbolData) == ELF::STT_FILE)
          return true;
        if (GetType(*RHS.SymbolData) == ELF::STT_FILE)
          return false;
        return SymbolData->getSymbol().getName() <
               RHS.SymbolData->getSymbol().getName();
      }
    };

    /// @name Relocation Data
    /// @{

    struct ELFRelocationEntry {
      // Make these big enough for both 32-bit and 64-bit
      uint64_t r_offset;
      int Index;
      unsigned Type;
      const MCSymbol *Symbol;
      uint64_t r_addend;

      ELFRelocationEntry()
        : r_offset(0), Index(0), Type(0), Symbol(0), r_addend(0) {}

      ELFRelocationEntry(uint64_t RelocOffset, int Idx,
                         unsigned RelType, const MCSymbol *Sym,
                         uint64_t Addend)
        : r_offset(RelocOffset), Index(Idx), Type(RelType),
          Symbol(Sym), r_addend(Addend) {}

      // Support lexicographic sorting.
      bool operator<(const ELFRelocationEntry &RE) const {
        return RE.r_offset < r_offset;
      }
    };

    SmallPtrSet<const MCSymbol *, 16> UsedInReloc;
    SmallPtrSet<const MCSymbol *, 16> WeakrefUsedInReloc;
    DenseMap<const MCSymbol *, const MCSymbol *> Renames;

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

    bool NeedsGOT;

    bool NeedsSymtabShndx;

    unsigned Is64Bit : 1;

    bool HasRelocationAddend;

    Triple::OSType OSType;

    uint16_t EMachine;

    // This holds the symbol table index of the last local symbol.
    unsigned LastLocalSymbolIndex;
    // This holds the .strtab section index.
    unsigned StringTableIndex;
    // This holds the .symtab section index.
    unsigned SymbolTableIndex;

    unsigned ShstrtabIndex;


    const MCSymbol *SymbolToReloc(const MCAssembler &Asm,
                                  const MCValue &Target,
                                  const MCFragment &F) const;

  public:
    ELFObjectWriter(raw_ostream &_OS, bool _Is64Bit, bool IsLittleEndian,
                    uint16_t _EMachine, bool _HasRelAddend,
                    Triple::OSType _OSType)
      : MCObjectWriter(_OS, IsLittleEndian),
        NeedsGOT(false), NeedsSymtabShndx(false),
        Is64Bit(_Is64Bit), HasRelocationAddend(_HasRelAddend),
        OSType(_OSType), EMachine(_EMachine) {
    }

    virtual ~ELFObjectWriter();

    void WriteWord(uint64_t W) {
      if (Is64Bit)
        Write64(W);
      else
        Write32(W);
    }

    void StringLE16(char *buf, uint16_t Value) {
      buf[0] = char(Value >> 0);
      buf[1] = char(Value >> 8);
    }

    void StringLE32(char *buf, uint32_t Value) {
      StringLE16(buf, uint16_t(Value >> 0));
      StringLE16(buf + 2, uint16_t(Value >> 16));
    }

    void StringLE64(char *buf, uint64_t Value) {
      StringLE32(buf, uint32_t(Value >> 0));
      StringLE32(buf + 4, uint32_t(Value >> 32));
    }

    void StringBE16(char *buf ,uint16_t Value) {
      buf[0] = char(Value >> 8);
      buf[1] = char(Value >> 0);
    }

    void StringBE32(char *buf, uint32_t Value) {
      StringBE16(buf, uint16_t(Value >> 16));
      StringBE16(buf + 2, uint16_t(Value >> 0));
    }

    void StringBE64(char *buf, uint64_t Value) {
      StringBE32(buf, uint32_t(Value >> 32));
      StringBE32(buf + 4, uint32_t(Value >> 0));
    }

    void String8(MCDataFragment &F, uint8_t Value) {
      char buf[1];
      buf[0] = Value;
      F.getContents() += StringRef(buf, 1);
    }

    void String16(MCDataFragment &F, uint16_t Value) {
      char buf[2];
      if (isLittleEndian())
        StringLE16(buf, Value);
      else
        StringBE16(buf, Value);
      F.getContents() += StringRef(buf, 2);
    }

    void String32(MCDataFragment &F, uint32_t Value) {
      char buf[4];
      if (isLittleEndian())
        StringLE32(buf, Value);
      else
        StringBE32(buf, Value);
      F.getContents() += StringRef(buf, 4);
    }

    void String64(MCDataFragment &F, uint64_t Value) {
      char buf[8];
      if (isLittleEndian())
        StringLE64(buf, Value);
      else
        StringBE64(buf, Value);
      F.getContents() += StringRef(buf, 8);
    }

    virtual void WriteHeader(uint64_t SectionDataSize, unsigned NumberOfSections);

    virtual void WriteSymbolEntry(MCDataFragment *SymtabF, MCDataFragment *ShndxF,
                          uint64_t name, uint8_t info,
                          uint64_t value, uint64_t size,
                          uint8_t other, uint32_t shndx,
                          bool Reserved);

    virtual void WriteSymbol(MCDataFragment *SymtabF,  MCDataFragment *ShndxF,
                     ELFSymbolData &MSD,
                     const MCAsmLayout &Layout);

    typedef DenseMap<const MCSectionELF*, uint32_t> SectionIndexMapTy;
    virtual void WriteSymbolTable(MCDataFragment *SymtabF, MCDataFragment *ShndxF,
                          const MCAssembler &Asm,
                          const MCAsmLayout &Layout,
                          const SectionIndexMapTy &SectionIndexMap);

    virtual void RecordRelocation(const MCAssembler &Asm, const MCAsmLayout &Layout,
                                  const MCFragment *Fragment, const MCFixup &Fixup,
                                  MCValue Target, uint64_t &FixedValue);

    virtual uint64_t getSymbolIndexInSymbolTable(const MCAssembler &Asm,
                                         const MCSymbol *S);

    // Map from a group section to the signature symbol
    typedef DenseMap<const MCSectionELF*, const MCSymbol*> GroupMapTy;
    // Map from a signature symbol to the group section
    typedef DenseMap<const MCSymbol*, const MCSectionELF*> RevGroupMapTy;

    /// ComputeSymbolTable - Compute the symbol table data
    ///
    /// \param StringTable [out] - The string table data.
    /// \param StringIndexMap [out] - Map from symbol names to offsets in the
    /// string table.
    virtual void ComputeSymbolTable(MCAssembler &Asm,
                            const SectionIndexMapTy &SectionIndexMap,
                            RevGroupMapTy RevGroupMap);

    virtual void ComputeIndexMap(MCAssembler &Asm,
                         SectionIndexMapTy &SectionIndexMap);

    virtual void WriteRelocation(MCAssembler &Asm, MCAsmLayout &Layout,
                         const MCSectionData &SD);

    virtual void WriteRelocations(MCAssembler &Asm, MCAsmLayout &Layout) {
      for (MCAssembler::const_iterator it = Asm.begin(),
             ie = Asm.end(); it != ie; ++it) {
        WriteRelocation(Asm, Layout, *it);
      }
    }

    virtual void CreateMetadataSections(MCAssembler &Asm, MCAsmLayout &Layout,
                                const SectionIndexMapTy &SectionIndexMap);

    virtual void CreateGroupSections(MCAssembler &Asm, MCAsmLayout &Layout,
                             GroupMapTy &GroupMap, RevGroupMapTy &RevGroupMap);

    virtual void ExecutePostLayoutBinding(MCAssembler &Asm,
                                          const MCAsmLayout &Layout);

    virtual void WriteSecHdrEntry(uint32_t Name, uint32_t Type, uint64_t Flags,
                          uint64_t Address, uint64_t Offset,
                          uint64_t Size, uint32_t Link, uint32_t Info,
                          uint64_t Alignment, uint64_t EntrySize);

    virtual void WriteRelocationsFragment(const MCAssembler &Asm, MCDataFragment *F,
                                  const MCSectionData *SD);

    virtual bool IsFixupFullyResolved(const MCAssembler &Asm,
                              const MCValue Target,
                              bool IsPCRel,
                              const MCFragment *DF) const;

    virtual void WriteObject(MCAssembler &Asm, const MCAsmLayout &Layout);
    virtual void WriteSection(MCAssembler &Asm,
                      const SectionIndexMapTy &SectionIndexMap,
                      uint32_t GroupSymbolIndex,
                      uint64_t Offset, uint64_t Size, uint64_t Alignment,
                      const MCSectionELF &Section);

  protected:
    virtual unsigned GetRelocType(const MCValue &Target, const MCFixup &Fixup,
                                  bool IsPCRel, bool IsRelocWithSymbol,
                                  int64_t Addend) = 0;

    virtual bool isFixupKindPCRel(unsigned Kind) const = 0;
  };

  //===- X86ELFObjectWriter -------------------------------------------===//

  class X86ELFObjectWriter : public ELFObjectWriter {
  public:
    X86ELFObjectWriter(raw_ostream &_OS, bool _Is64Bit, bool IsLittleEndian,
                       uint16_t _EMachine, bool _HasRelAddend,
                       Triple::OSType _OSType);

    virtual ~X86ELFObjectWriter();
  protected:
    virtual unsigned GetRelocType(const MCValue &Target, const MCFixup &Fixup,
                                  bool IsPCRel, bool IsRelocWithSymbol,
                                  int64_t Addend);

    virtual bool isFixupKindPCRel(unsigned Kind) const {
      switch (Kind) {
      default:
        return false;
      case FK_PCRel_1:
      case FK_PCRel_2:
      case FK_PCRel_4:
      case X86::reloc_riprel_4byte:
      case X86::reloc_riprel_4byte_movq_load:
        return true;
      }
    }
  };


  //===- ARMELFObjectWriter -------------------------------------------===//

  class ARMELFObjectWriter : public ELFObjectWriter {
  public:
    ARMELFObjectWriter(raw_ostream &_OS, bool _Is64Bit, bool IsLittleEndian,
                       uint16_t _EMachine, bool _HasRelAddend,
                       Triple::OSType _OSType);

    virtual ~ARMELFObjectWriter();
  protected:
    virtual unsigned GetRelocType(const MCValue &Target, const MCFixup &Fixup,
                                  bool IsPCRel, bool IsRelocWithSymbol,
                                  int64_t Addend);
    virtual bool isFixupKindPCRel(unsigned Kind) const {
      switch (Kind) {
      default:
        return false;
      case FK_PCRel_1:
      case FK_PCRel_2:
      case FK_PCRel_4:
      case ARM::fixup_arm_ldst_pcrel_12:
      case ARM::fixup_arm_pcrel_10:
      case ARM::fixup_arm_branch:
        return true;
      }
    }
  };

  //===- MBlazeELFObjectWriter -------------------------------------------===//

  class MBlazeELFObjectWriter : public ELFObjectWriter {
  public:
    MBlazeELFObjectWriter(raw_ostream &_OS, bool _Is64Bit, bool IsLittleEndian,
                          uint16_t _EMachine, bool _HasRelAddend,
                          Triple::OSType _OSType);

    virtual ~MBlazeELFObjectWriter();
  protected:
    virtual unsigned GetRelocType(const MCValue &Target, const MCFixup &Fixup,
                                  bool IsPCRel, bool IsRelocWithSymbol,
                                  int64_t Addend);

    virtual bool isFixupKindPCRel(unsigned Kind) const {
      switch (Kind) {
      default:
        return false;
      case FK_PCRel_1:
      case FK_PCRel_2:
      case FK_PCRel_4:
        return true;
      }
    }
  };
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

  Write8(Is64Bit ? ELF::ELFCLASS64 : ELF::ELFCLASS32); // e_ident[EI_CLASS]

  // e_ident[EI_DATA]
  Write8(isLittleEndian() ? ELF::ELFDATA2LSB : ELF::ELFDATA2MSB);

  Write8(ELF::EV_CURRENT);        // e_ident[EI_VERSION]
  // e_ident[EI_OSABI]
  switch (OSType) {
    case Triple::FreeBSD:  Write8(ELF::ELFOSABI_FREEBSD); break;
    case Triple::Linux:    Write8(ELF::ELFOSABI_LINUX); break;
    default:               Write8(ELF::ELFOSABI_NONE); break;
  }
  Write8(0);                  // e_ident[EI_ABIVERSION]

  WriteZeros(ELF::EI_NIDENT - ELF::EI_PAD);

  Write16(ELF::ET_REL);             // e_type

  Write16(EMachine); // e_machine = target

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

  if (Is64Bit) {
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

static uint64_t SymbolValue(MCSymbolData &Data, const MCAsmLayout &Layout) {
  if (Data.isCommon() && Data.isExternal())
    return Data.getCommonAlignment();

  const MCSymbol &Symbol = Data.getSymbol();
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
    SetBinding(*it, GetBinding(SD));

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

  uint8_t Binding = GetBinding(OrigData);
  uint8_t Visibility = GetVisibility(OrigData);
  uint8_t Type = GetType(Data);

  uint8_t Info = (Binding << ELF_STB_Shift) | (Type << ELF_STT_Shift);
  uint8_t Other = Visibility;

  uint64_t Value = SymbolValue(Data, Layout);
  uint64_t Size = 0;
  const MCExpr *ESize;

  assert(!(Data.isCommon() && !Data.isExternal()));

  ESize = Data.getSize();
  if (Data.getSize()) {
    MCValue Res;
    if (ESize->getKind() == MCExpr::Binary) {
      const MCBinaryExpr *BE = static_cast<const MCBinaryExpr *>(ESize);

      if (BE->EvaluateAsRelocatable(Res, &Layout)) {
        assert(!Res.getSymA() || !Res.getSymA()->getSymbol().isDefined());
        assert(!Res.getSymB() || !Res.getSymB()->getSymbol().isDefined());
        Size = Res.getConstant();
      }
    } else if (ESize->getKind() == MCExpr::Constant) {
      Size = static_cast<const MCConstantExpr *>(ESize)->getValue();
    } else {
      assert(0 && "Unsupported size expression");
    }
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
    if (GetBinding(Data) == ELF::STB_LOCAL)
      LastLocalSymbolIndex++;
  }

  for (unsigned i = 0, e = UndefinedSymbolData.size(); i != e; ++i) {
    ELFSymbolData &MSD = UndefinedSymbolData[i];
    MCSymbolData &Data = *MSD.SymbolData;
    WriteSymbol(SymtabF, ShndxF, MSD, Layout);
    if (GetBinding(Data) == ELF::STB_LOCAL)
      LastLocalSymbolIndex++;
  }
}

const MCSymbol *ELFObjectWriter::SymbolToReloc(const MCAssembler &Asm,
                                               const MCValue &Target,
                                               const MCFragment &F) const {
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
    return NULL;

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

  if (Section.getFlags() & MCSectionELF::SHF_MERGE) {
    if (Target.getConstant() == 0)
      return NULL;
    if (Renamed)
      return Renamed;
    return &Symbol;
  }

  return NULL;
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

  bool IsPCRel = isFixupKindPCRel(Fixup.getKind());
  if (!Target.isAbsolute()) {
    const MCSymbol &Symbol = Target.getSymA()->getSymbol();
    const MCSymbol &ASymbol = Symbol.AliasedSymbol();
    RelocSymbol = SymbolToReloc(Asm, Target, *Fragment);

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
    if (Is64Bit)
      Value = 0;
  }

  FixedValue = Value;
  unsigned Type = GetRelocType(Target, Fixup, IsPCRel,
                               (RelocSymbol != 0), Addend);
  
  uint64_t RelocOffset = Layout.getFragmentOffset(Fragment) +
    Fixup.getOffset();

  if (!HasRelocationAddend) Addend = 0;
  ELFRelocationEntry ERE(RelocOffset, Index, Type, RelocSymbol, Addend);
  Relocations[Fragment->getParent()].push_back(ERE);
}


uint64_t
ELFObjectWriter::getSymbolIndexInSymbolTable(const MCAssembler &Asm,
                                             const MCSymbol *S) {
  MCSymbolData &SD = Asm.getSymbolData(*S);
  return SD.getIndex();
}

static bool isInSymtab(const MCAssembler &Asm, const MCSymbolData &Data,
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
  if (!A.isVariable() && A.isUndefined() && !Data.isCommon())
    return false;

  if (!Asm.isSymbolLinkerVisible(Symbol) && !Symbol.isUndefined())
    return false;

  if (Symbol.isTemporary())
    return false;

  return true;
}

static bool isLocal(const MCSymbolData &Data, bool isSignature,
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
                                      SectionIndexMapTy &SectionIndexMap) {
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
    if (Section.getType() == ELF::SHT_GROUP)
      continue;
    SectionIndexMap[&Section] = Index++;
  }
}

void ELFObjectWriter::ComputeSymbolTable(MCAssembler &Asm,
                                      const SectionIndexMapTy &SectionIndexMap,
                                      RevGroupMapTy RevGroupMap) {
  // FIXME: Is this the correct place to do this?
  if (NeedsGOT) {
    llvm::StringRef Name = "_GLOBAL_OFFSET_TABLE_";
    MCSymbol *Sym = Asm.getContext().GetOrCreateSymbol(Name);
    MCSymbolData &Data = Asm.getOrCreateSymbolData(*Sym);
    Data.setExternal(true);
    SetBinding(Data, ELF::STB_GLOBAL);
  }

  // Build section lookup table.
  int NumRegularSections = Asm.size();

  // Index 0 is always the empty string.
  StringMap<uint64_t> StringIndexMap;
  StringTable += '\x00';

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
    if (!Local && GetBinding(*it) == ELF::STB_LOCAL) {
      MCSymbolData &SD = Asm.getSymbolData(RefSymbol);
      SetBinding(*it, ELF::STB_GLOBAL);
      SetBinding(SD, ELF::STB_GLOBAL);
    }

    if (RefSymbol.isUndefined() && !Used && WeakrefUsed)
      SetBinding(*it, ELF::STB_WEAK);

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

void ELFObjectWriter::WriteRelocation(MCAssembler &Asm, MCAsmLayout &Layout,
                                      const MCSectionData &SD) {
  if (!Relocations[&SD].empty()) {
    MCContext &Ctx = Asm.getContext();
    const MCSectionELF *RelaSection;
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
                                    EntrySize, "");

    MCSectionData &RelaSD = Asm.getOrCreateSectionData(*RelaSection);
    RelaSD.setAlignment(Is64Bit ? 8 : 4);

    MCDataFragment *F = new MCDataFragment(&RelaSD);

    WriteRelocationsFragment(Asm, F, &SD);
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
    if (Is64Bit) {
      String64(*F, entry.r_offset);

      struct ELF::Elf64_Rela ERE64;
      ERE64.setSymbolAndType(entry.Index, entry.Type);
      String64(*F, ERE64.r_info);

      if (HasRelocationAddend)
        String64(*F, entry.r_addend);
    } else {
      String32(*F, entry.r_offset);

      struct ELF::Elf32_Rela ERE32;
      ERE32.setSymbolAndType(entry.Index, entry.Type);
      String32(*F, ERE32.r_info);

      if (HasRelocationAddend)
        String32(*F, entry.r_addend);
    }
  }
}

void ELFObjectWriter::CreateMetadataSections(MCAssembler &Asm,
                                             MCAsmLayout &Layout,
                                    const SectionIndexMapTy &SectionIndexMap) {
  MCContext &Ctx = Asm.getContext();
  MCDataFragment *F;

  unsigned EntrySize = Is64Bit ? ELF::SYMENTRY_SIZE64 : ELF::SYMENTRY_SIZE32;

  // We construct .shstrtab, .symtab and .strtab in this order to match gnu as.
  const MCSectionELF *ShstrtabSection =
    Ctx.getELFSection(".shstrtab", ELF::SHT_STRTAB, 0,
                      SectionKind::getReadOnly());
  MCSectionData &ShstrtabSD = Asm.getOrCreateSectionData(*ShstrtabSection);
  ShstrtabSD.setAlignment(1);
  ShstrtabIndex = Asm.size();

  const MCSectionELF *SymtabSection =
    Ctx.getELFSection(".symtab", ELF::SHT_SYMTAB, 0,
                      SectionKind::getReadOnly(),
                      EntrySize, "");
  MCSectionData &SymtabSD = Asm.getOrCreateSectionData(*SymtabSection);
  SymtabSD.setAlignment(Is64Bit ? 8 : 4);
  SymbolTableIndex = Asm.size();

  MCSectionData *SymtabShndxSD = NULL;

  if (NeedsSymtabShndx) {
    const MCSectionELF *SymtabShndxSection =
      Ctx.getELFSection(".symtab_shndx", ELF::SHT_SYMTAB_SHNDX, 0,
                        SectionKind::getReadOnly(), 4, "");
    SymtabShndxSD = &Asm.getOrCreateSectionData(*SymtabShndxSection);
    SymtabShndxSD->setAlignment(4);
  }

  const MCSection *StrtabSection;
  StrtabSection = Ctx.getELFSection(".strtab", ELF::SHT_STRTAB, 0,
                                    SectionKind::getReadOnly());
  MCSectionData &StrtabSD = Asm.getOrCreateSectionData(*StrtabSection);
  StrtabSD.setAlignment(1);
  StringTableIndex = Asm.size();

  WriteRelocations(Asm, Layout);

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

  // Section header string table.
  //
  // The first entry of a string table holds a null character so skip
  // section 0.
  uint64_t Index = 1;
  F->getContents() += '\x00';

  StringMap<uint64_t> SecStringMap;
  for (MCAssembler::const_iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it) {
    const MCSectionELF &Section =
      static_cast<const MCSectionELF&>(it->getSection());
    // FIXME: We could merge suffixes like in .text and .rela.text.

    StringRef Name = Section.getSectionName();
    if (SecStringMap.count(Name)) {
      SectionStringTableIndex[&Section] =  SecStringMap[Name];
      continue;
    }
    // Remember the index into the string table so we can write it
    // into the sh_name field of the section header table.
    SectionStringTableIndex[&Section] = Index;
    SecStringMap[Name] = Index;

    Index += Name.size() + 1;
    F->getContents() += Name;
    F->getContents() += '\x00';
  }
}

bool ELFObjectWriter::IsFixupFullyResolved(const MCAssembler &Asm,
                                           const MCValue Target,
                                           bool IsPCRel,
                                           const MCFragment *DF) const {
  // If this is a PCrel relocation, find the section this fixup value is
  // relative to.
  const MCSection *BaseSection = 0;
  if (IsPCRel) {
    BaseSection = &DF->getParent()->getSection();
    assert(BaseSection);
  }

  const MCSection *SectionA = 0;
  const MCSymbol *SymbolA = 0;
  if (const MCSymbolRefExpr *A = Target.getSymA()) {
    SymbolA = &A->getSymbol();
    SectionA = &SymbolA->AliasedSymbol().getSection();
  }

  const MCSection *SectionB = 0;
  const MCSymbol *SymbolB = 0;
  if (const MCSymbolRefExpr *B = Target.getSymB()) {
    SymbolB = &B->getSymbol();
    SectionB = &SymbolB->AliasedSymbol().getSection();
  }

  if (!BaseSection)
    return SectionA == SectionB;

  if (SymbolB)
    return false;

  // Absolute address but PCrel instruction, so we need a relocation.
  if (!SymbolA)
    return false;

  // FIXME: This is in here just to match gnu as output. If the two ends
  // are in the same section, there is nothing that the linker can do to
  // break it.
  const MCSymbolData &DataA = Asm.getSymbolData(*SymbolA);
  if (DataA.isExternal())
    return false;

  return BaseSection == SectionA;
}

void ELFObjectWriter::CreateGroupSections(MCAssembler &Asm,
                                          MCAsmLayout &Layout,
                                          GroupMapTy &GroupMap,
                                          RevGroupMapTy &RevGroupMap) {
  // Build the groups
  for (MCAssembler::const_iterator it = Asm.begin(), ie = Asm.end();
       it != ie; ++it) {
    const MCSectionELF &Section =
      static_cast<const MCSectionELF&>(it->getSection());
    if (!(Section.getFlags() & MCSectionELF::SHF_GROUP))
      continue;

    const MCSymbol *SignatureSymbol = Section.getGroup();
    Asm.getOrCreateSymbolData(*SignatureSymbol);
    const MCSectionELF *&Group = RevGroupMap[SignatureSymbol];
    if (!Group) {
      Group = Asm.getContext().CreateELFGroupSection();
      MCSectionData &Data = Asm.getOrCreateSectionData(*Group);
      Data.setAlignment(4);
      MCDataFragment *F = new MCDataFragment(&Data);
      String32(*F, ELF::GRP_COMDAT);
    }
    GroupMap[Group] = SignatureSymbol;
  }

  // Add sections to the groups
  unsigned Index = 1;
  unsigned NumGroups = RevGroupMap.size();
  for (MCAssembler::const_iterator it = Asm.begin(), ie = Asm.end();
       it != ie; ++it, ++Index) {
    const MCSectionELF &Section =
      static_cast<const MCSectionELF&>(it->getSection());
    if (!(Section.getFlags() & MCSectionELF::SHF_GROUP))
      continue;
    const MCSectionELF *Group = RevGroupMap[Section.getGroup()];
    MCSectionData &Data = Asm.getOrCreateSectionData(*Group);
    // FIXME: we could use the previous fragment
    MCDataFragment *F = new MCDataFragment(&Data);
    String32(*F, NumGroups + Index);
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
  case ELF::SHT_NULL:
  case ELF::SHT_ARM_ATTRIBUTES:
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

static bool IsELFMetaDataSection(const MCSectionData &SD) {
  return SD.getOrdinal() == ~UINT32_C(0) &&
    !SD.getSection().isVirtualSection();
}

static uint64_t DataSectionSize(const MCSectionData &SD) {
  uint64_t Ret = 0;
  for (MCSectionData::const_iterator i = SD.begin(), e = SD.end(); i != e;
       ++i) {
    const MCFragment &F = *i;
    assert(F.getKind() == MCFragment::FT_Data);
    Ret += cast<MCDataFragment>(F).getContents().size();
  }
  return Ret;
}

static uint64_t GetSectionFileSize(const MCAsmLayout &Layout,
                                   const MCSectionData &SD) {
  if (IsELFMetaDataSection(SD))
    return DataSectionSize(SD);
  return Layout.getSectionFileSize(&SD);
}

static uint64_t GetSectionAddressSize(const MCAsmLayout &Layout,
                                      const MCSectionData &SD) {
  if (IsELFMetaDataSection(SD))
    return DataSectionSize(SD);
  return Layout.getSectionAddressSize(&SD);
}

static void WriteDataSectionData(ELFObjectWriter *W, const MCSectionData &SD) {
  for (MCSectionData::const_iterator i = SD.begin(), e = SD.end(); i != e;
       ++i) {
    const MCFragment &F = *i;
    assert(F.getKind() == MCFragment::FT_Data);
    W->WriteBytes(cast<MCDataFragment>(F).getContents().str());
  }
}

void ELFObjectWriter::WriteObject(MCAssembler &Asm,
                                  const MCAsmLayout &Layout) {
  GroupMapTy GroupMap;
  RevGroupMapTy RevGroupMap;
  CreateGroupSections(Asm, const_cast<MCAsmLayout&>(Layout), GroupMap,
                      RevGroupMap);

  SectionIndexMapTy SectionIndexMap;

  ComputeIndexMap(Asm, SectionIndexMap);

  // Compute symbol table information.
  ComputeSymbolTable(Asm, SectionIndexMap, RevGroupMap);

  CreateMetadataSections(const_cast<MCAssembler&>(Asm),
                         const_cast<MCAsmLayout&>(Layout),
                         SectionIndexMap);

  // Update to include the metadata sections.
  ComputeIndexMap(Asm, SectionIndexMap);

  // Add 1 for the null section.
  unsigned NumSections = Asm.size() + 1;
  uint64_t NaturalAlignment = Is64Bit ? 8 : 4;
  uint64_t HeaderSize = Is64Bit ? sizeof(ELF::Elf64_Ehdr) : sizeof(ELF::Elf32_Ehdr);
  uint64_t FileOff = HeaderSize;

  std::vector<const MCSectionELF*> Sections;
  Sections.resize(NumSections);

  for (SectionIndexMapTy::const_iterator i=
         SectionIndexMap.begin(), e = SectionIndexMap.end(); i != e; ++i) {
    const std::pair<const MCSectionELF*, uint32_t> &p = *i;
    Sections[p.second] = p.first;
  }

  for (unsigned i = 1; i < NumSections; ++i) {
    const MCSectionELF &Section = *Sections[i];
    const MCSectionData &SD = Asm.getOrCreateSectionData(Section);

    FileOff = RoundUpToAlignment(FileOff, SD.getAlignment());

    // Get the size of the section in the output file (including padding).
    FileOff += GetSectionFileSize(Layout, SD);
  }

  FileOff = RoundUpToAlignment(FileOff, NaturalAlignment);

  // Write out the ELF header ...
  WriteHeader(FileOff - HeaderSize, NumSections);

  FileOff = HeaderSize;

  // ... then all of the sections ...
  DenseMap<const MCSection*, uint64_t> SectionOffsetMap;

  for (unsigned i = 1; i < NumSections; ++i) {
    const MCSectionELF &Section = *Sections[i];
    const MCSectionData &SD = Asm.getOrCreateSectionData(Section);

    uint64_t Padding = OffsetToAlignment(FileOff, SD.getAlignment());
    WriteZeros(Padding);
    FileOff += Padding;

    // Remember the offset into the file for this section.
    SectionOffsetMap[&Section] = FileOff;

    FileOff += GetSectionFileSize(Layout, SD);

    if (IsELFMetaDataSection(SD))
      WriteDataSectionData(this, SD);
    else
      Asm.WriteSectionData(&SD, Layout);
  }

  uint64_t Padding = OffsetToAlignment(FileOff, NaturalAlignment);
  WriteZeros(Padding);
  FileOff += Padding;

  // ... and then the section header table.
  // Should we align the section header table?
  //
  // Null section first.
  uint64_t FirstSectionSize =
    NumSections >= ELF::SHN_LORESERVE ? NumSections : 0;
  uint32_t FirstSectionLink =
    ShstrtabIndex >= ELF::SHN_LORESERVE ? ShstrtabIndex : 0;
  WriteSecHdrEntry(0, 0, 0, 0, 0, FirstSectionSize, FirstSectionLink, 0, 0, 0);

  for (unsigned i = 1; i < NumSections; ++i) {
    const MCSectionELF &Section = *Sections[i];
    const MCSectionData &SD = Asm.getOrCreateSectionData(Section);
    uint32_t GroupSymbolIndex;
    if (Section.getType() != ELF::SHT_GROUP)
      GroupSymbolIndex = 0;
    else
      GroupSymbolIndex = getSymbolIndexInSymbolTable(Asm, GroupMap[&Section]);

    uint64_t Size = GetSectionAddressSize(Layout, SD);

    WriteSection(Asm, SectionIndexMap, GroupSymbolIndex,
                 SectionOffsetMap[&Section], Size,
                 SD.getAlignment(), Section);
  }
}

MCObjectWriter *llvm::createELFObjectWriter(raw_ostream &OS,
                                            bool Is64Bit,
                                            Triple::OSType OSType,
                                            uint16_t EMachine,
                                            bool IsLittleEndian,
                                            bool HasRelocationAddend) {
  switch (EMachine) {
    case ELF::EM_386:
    case ELF::EM_X86_64:
      return new X86ELFObjectWriter(OS, Is64Bit, IsLittleEndian, EMachine,
                                    HasRelocationAddend, OSType); break;
    case ELF::EM_ARM:
      return new ARMELFObjectWriter(OS, Is64Bit, IsLittleEndian, EMachine,
                                    HasRelocationAddend, OSType); break;
    case ELF::EM_MBLAZE:
      return new MBlazeELFObjectWriter(OS, Is64Bit, IsLittleEndian, EMachine,
                                       HasRelocationAddend, OSType); break;
    default: llvm_unreachable("Unsupported architecture"); break;
  }
}


/// START OF SUBCLASSES for ELFObjectWriter
//===- ARMELFObjectWriter -------------------------------------------===//

ARMELFObjectWriter::ARMELFObjectWriter(raw_ostream &_OS, bool _Is64Bit,
                                       bool _IsLittleEndian,
                                       uint16_t _EMachine, bool _HasRelocationAddend,
                                       Triple::OSType _OSType)
  : ELFObjectWriter(_OS, _Is64Bit, _IsLittleEndian, _EMachine,
                    _HasRelocationAddend, _OSType)
{}

ARMELFObjectWriter::~ARMELFObjectWriter()
{}

unsigned ARMELFObjectWriter::GetRelocType(const MCValue &Target,
                                          const MCFixup &Fixup,
                                          bool IsPCRel,
                                          bool IsRelocWithSymbol,
                                          int64_t Addend) {
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
        Type = ELF::R_ARM_BASE_PREL; break;
      case MCSymbolRefExpr::VK_ARM_TLSGD:
        assert(0 && "unimplemented"); break;
      case MCSymbolRefExpr::VK_ARM_GOTTPOFF:
        Type = ELF::R_ARM_TLS_IE32;
      } break;
    case ARM::fixup_arm_branch:
      switch (Modifier) {
      case MCSymbolRefExpr::VK_ARM_PLT:
        Type = ELF::R_ARM_PLT32; break;
      default:
        Type = ELF::R_ARM_CALL; break;
      } break;
    }
  } else {
    switch ((unsigned)Fixup.getKind()) {
    default: llvm_unreachable("invalid fixup kind!");
    case FK_Data_4:
      switch (Modifier) {
      default: llvm_unreachable("Unsupported Modifier"); break;
      case MCSymbolRefExpr::VK_ARM_GOT:
        Type = ELF::R_ARM_GOT_BREL; break;
      case MCSymbolRefExpr::VK_ARM_TLSGD:
        Type = ELF::R_ARM_TLS_GD32; break;
      case MCSymbolRefExpr::VK_ARM_TPOFF:
        Type = ELF::R_ARM_TLS_LE32; break;
      case MCSymbolRefExpr::VK_ARM_GOTTPOFF:
        Type = ELF::R_ARM_TLS_IE32; break;
      case MCSymbolRefExpr::VK_None:
        Type = ELF::R_ARM_ABS32; break;
      case MCSymbolRefExpr::VK_ARM_GOTOFF:
        Type = ELF::R_ARM_GOTOFF32; break;
      } break;
    case ARM::fixup_arm_ldst_pcrel_12:
    case ARM::fixup_arm_pcrel_10:
    case ARM::fixup_arm_adr_pcrel_12:
    case ARM::fixup_arm_thumb_bl:
    case ARM::fixup_arm_thumb_cb:
    case ARM::fixup_arm_thumb_cp:
    case ARM::fixup_arm_thumb_br:
      assert(0 && "Unimplemented"); break;
    case ARM::fixup_arm_branch:
      // FIXME: Differentiate between R_ARM_CALL and
      // R_ARM_JUMP24 (latter used for conditional jumps)
      Type = ELF::R_ARM_CALL; break;
    case ARM::fixup_arm_movt_hi16: 
      Type = ELF::R_ARM_MOVT_ABS; break;
    case ARM::fixup_arm_movw_lo16:
      Type = ELF::R_ARM_MOVW_ABS_NC; break;
    }
  }

  if (RelocNeedsGOT(Modifier))
    NeedsGOT = true;
  
  return Type;
}

//===- MBlazeELFObjectWriter -------------------------------------------===//

MBlazeELFObjectWriter::MBlazeELFObjectWriter(raw_ostream &_OS, bool _Is64Bit,
                                             bool _IsLittleEndian,
                                             uint16_t _EMachine,
                                             bool _HasRelocationAddend,
                                             Triple::OSType _OSType)
  : ELFObjectWriter(_OS, _Is64Bit, _IsLittleEndian, _EMachine,
                    _HasRelocationAddend, _OSType) {
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


X86ELFObjectWriter::X86ELFObjectWriter(raw_ostream &_OS, bool _Is64Bit,
                                       bool _IsLittleEndian,
                                       uint16_t _EMachine, bool _HasRelocationAddend,
                                       Triple::OSType _OSType)
  : ELFObjectWriter(_OS, _Is64Bit, _IsLittleEndian, _EMachine,
                    _HasRelocationAddend, _OSType)
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
  if (Is64Bit) {
    if (IsPCRel) {
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
    } else {
      switch ((unsigned)Fixup.getKind()) {
      default: llvm_unreachable("invalid fixup kind!");
      case FK_Data_8: Type = ELF::R_X86_64_64; break;
      case X86::reloc_signed_4byte:
      case FK_PCRel_4:
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
