//===- lib/MC/ELFObjectWriter.cpp - ELF File Writer -----------------------===//
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

#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELF.h"
#include "llvm/MC/MCELFSymbolFlags.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include <vector>
using namespace llvm;

#undef  DEBUG_TYPE
#define DEBUG_TYPE "reloc-info"

namespace {

typedef DenseMap<const MCSectionELF *, uint32_t> SectionIndexMapTy;

class ELFObjectWriter;

class SymbolTableWriter {
  ELFObjectWriter &EWriter;
  bool Is64Bit;

  // indexes we are going to write to .symtab_shndx.
  std::vector<uint32_t> ShndxIndexes;

  // The numbel of symbols written so far.
  unsigned NumWritten;

  void createSymtabShndx();

  template <typename T> void write(T Value);

public:
  SymbolTableWriter(ELFObjectWriter &EWriter, bool Is64Bit);

  void writeSymbol(uint32_t name, uint8_t info, uint64_t value, uint64_t size,
                   uint8_t other, uint32_t shndx, bool Reserved);

  ArrayRef<uint32_t> getShndxIndexes() const { return ShndxIndexes; }
};

class ELFObjectWriter : public MCObjectWriter {
    static bool isFixupKindPCRel(const MCAssembler &Asm, unsigned Kind);
    static bool RelocNeedsGOT(MCSymbolRefExpr::VariantKind Variant);
    static uint64_t SymbolValue(const MCSymbol &Sym, const MCAsmLayout &Layout);
    static bool isInSymtab(const MCAsmLayout &Layout, const MCSymbol &Symbol,
                           bool Used, bool Renamed);
    static bool isLocal(const MCSymbol &Symbol, bool isUsedInReloc);

    /// Helper struct for containing some precomputed information on symbols.
    struct ELFSymbolData {
      const MCSymbol *Symbol;
      uint64_t StringIndex;
      uint32_t SectionIndex;
      StringRef Name;

      // Support lexicographic sorting.
      bool operator<(const ELFSymbolData &RHS) const {
        unsigned LHSType = MCELF::GetType(Symbol->getData());
        unsigned RHSType = MCELF::GetType(RHS.Symbol->getData());
        if (LHSType == ELF::STT_SECTION && RHSType != ELF::STT_SECTION)
          return false;
        if (LHSType != ELF::STT_SECTION && RHSType == ELF::STT_SECTION)
          return true;
        if (LHSType == ELF::STT_SECTION && RHSType == ELF::STT_SECTION)
          return SectionIndex < RHS.SectionIndex;
        return Name < RHS.Name;
      }
    };

    /// The target specific ELF writer instance.
    std::unique_ptr<MCELFObjectTargetWriter> TargetObjectWriter;

    SmallPtrSet<const MCSymbol *, 16> UsedInReloc;
    SmallPtrSet<const MCSymbol *, 16> WeakrefUsedInReloc;
    DenseMap<const MCSymbol *, const MCSymbol *> Renames;

    llvm::DenseMap<const MCSectionELF *, std::vector<ELFRelocationEntry>>
        Relocations;
    StringTableBuilder ShStrTabBuilder;

    /// @}
    /// @name Symbol Table Data
    /// @{

    StringTableBuilder StrTabBuilder;
    std::vector<uint64_t> FileSymbolData;
    std::vector<ELFSymbolData> LocalSymbolData;
    std::vector<ELFSymbolData> ExternalSymbolData;
    std::vector<ELFSymbolData> UndefinedSymbolData;

    /// @}

    bool NeedsGOT;

    // This holds the symbol table index of the last local symbol.
    unsigned LastLocalSymbolIndex;
    // This holds the .strtab section index.
    unsigned StringTableIndex;
    // This holds the .symtab section index.
    unsigned SymbolTableIndex;

    unsigned ShstrtabIndex;

    // Sections in the order they are to be output in the section table.
    std::vector<const MCSectionELF *> SectionTable;
    unsigned addToSectionTable(const MCSectionELF *Sec);

    // TargetObjectWriter wrappers.
    bool is64Bit() const { return TargetObjectWriter->is64Bit(); }
    bool hasRelocationAddend() const {
      return TargetObjectWriter->hasRelocationAddend();
    }
    unsigned GetRelocType(const MCValue &Target, const MCFixup &Fixup,
                          bool IsPCRel) const {
      return TargetObjectWriter->GetRelocType(Target, Fixup, IsPCRel);
    }

  public:
    ELFObjectWriter(MCELFObjectTargetWriter *MOTW, raw_pwrite_stream &OS,
                    bool IsLittleEndian)
        : MCObjectWriter(OS, IsLittleEndian), TargetObjectWriter(MOTW),
          NeedsGOT(false) {}

    void reset() override {
      UsedInReloc.clear();
      WeakrefUsedInReloc.clear();
      Renames.clear();
      Relocations.clear();
      ShStrTabBuilder.clear();
      StrTabBuilder.clear();
      FileSymbolData.clear();
      LocalSymbolData.clear();
      ExternalSymbolData.clear();
      UndefinedSymbolData.clear();
      NeedsGOT = false;
      SectionTable.clear();
      MCObjectWriter::reset();
    }

    ~ELFObjectWriter() override;

    void WriteWord(uint64_t W) {
      if (is64Bit())
        Write64(W);
      else
        Write32(W);
    }

    template <typename T> void write(T Val) {
      if (IsLittleEndian)
        support::endian::Writer<support::little>(OS).write(Val);
      else
        support::endian::Writer<support::big>(OS).write(Val);
    }

    void writeHeader(const MCAssembler &Asm);

    void WriteSymbol(SymbolTableWriter &Writer, ELFSymbolData &MSD,
                     const MCAsmLayout &Layout);

    // Start and end offset of each section
    typedef std::map<const MCSectionELF *, std::pair<uint64_t, uint64_t>>
        SectionOffsetsTy;

    void WriteSymbolTable(MCAssembler &Asm, const MCAsmLayout &Layout,
                          SectionOffsetsTy &SectionOffsets);

    bool shouldRelocateWithSymbol(const MCAssembler &Asm,
                                  const MCSymbolRefExpr *RefA,
                                  const MCSymbol *Sym, uint64_t C,
                                  unsigned Type) const;

    void RecordRelocation(MCAssembler &Asm, const MCAsmLayout &Layout,
                          const MCFragment *Fragment, const MCFixup &Fixup,
                          MCValue Target, bool &IsPCRel,
                          uint64_t &FixedValue) override;

    uint64_t getSymbolIndexInSymbolTable(const MCAssembler &Asm,
                                         const MCSymbol *S);

    // Map from a signature symbol to the group section index
    typedef DenseMap<const MCSymbol *, unsigned> RevGroupMapTy;

    /// Compute the symbol table data
    ///
    /// \param Asm - The assembler.
    /// \param SectionIndexMap - Maps a section to its index.
    /// \param RevGroupMap - Maps a signature symbol to the group section.
    void computeSymbolTable(MCAssembler &Asm, const MCAsmLayout &Layout,
                            const SectionIndexMapTy &SectionIndexMap,
                            const RevGroupMapTy &RevGroupMap);

    const MCSectionELF *createRelocationSection(MCAssembler &Asm,
                                                const MCSectionELF &Sec);

    const MCSectionELF *createSectionHeaderStringTable();
    const MCSectionELF *createStringTable(MCContext &Ctx);

    void ExecutePostLayoutBinding(MCAssembler &Asm,
                                  const MCAsmLayout &Layout) override;

    void writeSectionHeader(MCAssembler &Asm, const MCAsmLayout &Layout,
                            const SectionIndexMapTy &SectionIndexMap,
                            const SectionOffsetsTy &SectionOffsets);

    void writeSectionData(const MCAssembler &Asm, const MCSectionData &SD,
                          const MCAsmLayout &Layout);

    void WriteSecHdrEntry(uint32_t Name, uint32_t Type, uint64_t Flags,
                          uint64_t Address, uint64_t Offset, uint64_t Size,
                          uint32_t Link, uint32_t Info, uint64_t Alignment,
                          uint64_t EntrySize);

    void writeRelocations(const MCAssembler &Asm, const MCSectionELF &Sec);

    bool IsSymbolRefDifferenceFullyResolvedImpl(const MCAssembler &Asm,
                                                const MCSymbol &SymA,
                                                const MCFragment &FB,
                                                bool InSet,
                                                bool IsPCRel) const override;

    bool isWeak(const MCSymbol &Sym) const override;

    void WriteObject(MCAssembler &Asm, const MCAsmLayout &Layout) override;
    void writeSection(MCAssembler &Asm,
                      const SectionIndexMapTy &SectionIndexMap,
                      uint32_t GroupSymbolIndex,
                      uint64_t Offset, uint64_t Size, uint64_t Alignment,
                      const MCSectionELF &Section);
  };
}

unsigned ELFObjectWriter::addToSectionTable(const MCSectionELF *Sec) {
  SectionTable.push_back(Sec);
  ShStrTabBuilder.add(Sec->getSectionName());
  return SectionTable.size();
}

void SymbolTableWriter::createSymtabShndx() {
  if (!ShndxIndexes.empty())
    return;

  ShndxIndexes.resize(NumWritten);
}

template <typename T> void SymbolTableWriter::write(T Value) {
  EWriter.write(Value);
}

SymbolTableWriter::SymbolTableWriter(ELFObjectWriter &EWriter, bool Is64Bit)
    : EWriter(EWriter), Is64Bit(Is64Bit), NumWritten(0) {}

void SymbolTableWriter::writeSymbol(uint32_t name, uint8_t info, uint64_t value,
                                    uint64_t size, uint8_t other,
                                    uint32_t shndx, bool Reserved) {
  bool LargeIndex = shndx >= ELF::SHN_LORESERVE && !Reserved;

  if (LargeIndex)
    createSymtabShndx();

  if (!ShndxIndexes.empty()) {
    if (LargeIndex)
      ShndxIndexes.push_back(shndx);
    else
      ShndxIndexes.push_back(0);
  }

  uint16_t Index = LargeIndex ? uint16_t(ELF::SHN_XINDEX) : shndx;

  if (Is64Bit) {
    write(name);  // st_name
    write(info);  // st_info
    write(other); // st_other
    write(Index); // st_shndx
    write(value); // st_value
    write(size);  // st_size
  } else {
    write(name);            // st_name
    write(uint32_t(value)); // st_value
    write(uint32_t(size));  // st_size
    write(info);            // st_info
    write(other);           // st_other
    write(Index);           // st_shndx
  }

  ++NumWritten;
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
  case MCSymbolRefExpr::VK_GOTOFF:
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
void ELFObjectWriter::writeHeader(const MCAssembler &Asm) {
  // ELF Header
  // ----------
  //
  // Note
  // ----
  // emitWord method behaves differently for ELF32 and ELF64, writing
  // 4 bytes in the former and 8 in the latter.

  WriteBytes(ELF::ElfMagic); // e_ident[EI_MAG0] to e_ident[EI_MAG3]

  Write8(is64Bit() ? ELF::ELFCLASS64 : ELF::ELFCLASS32); // e_ident[EI_CLASS]

  // e_ident[EI_DATA]
  Write8(isLittleEndian() ? ELF::ELFDATA2LSB : ELF::ELFDATA2MSB);

  Write8(ELF::EV_CURRENT);        // e_ident[EI_VERSION]
  // e_ident[EI_OSABI]
  Write8(TargetObjectWriter->getOSABI());
  Write8(0);                  // e_ident[EI_ABIVERSION]

  WriteZeros(ELF::EI_NIDENT - ELF::EI_PAD);

  Write16(ELF::ET_REL);             // e_type

  Write16(TargetObjectWriter->getEMachine()); // e_machine = target

  Write32(ELF::EV_CURRENT);         // e_version
  WriteWord(0);                    // e_entry, no entry point in .o file
  WriteWord(0);                    // e_phoff, no program header for .o
  WriteWord(0);                     // e_shoff = sec hdr table off in bytes

  // e_flags = whatever the target wants
  Write32(Asm.getELFHeaderEFlags());

  // e_ehsize = ELF header size
  Write16(is64Bit() ? sizeof(ELF::Elf64_Ehdr) : sizeof(ELF::Elf32_Ehdr));

  Write16(0);                  // e_phentsize = prog header entry size
  Write16(0);                  // e_phnum = # prog header entries = 0

  // e_shentsize = Section header entry size
  Write16(is64Bit() ? sizeof(ELF::Elf64_Shdr) : sizeof(ELF::Elf32_Shdr));

  // e_shnum     = # of section header ents
  Write16(0);

  // e_shstrndx  = Section # of '.shstrtab'
  assert(ShstrtabIndex < ELF::SHN_LORESERVE);
  Write16(ShstrtabIndex);
}

uint64_t ELFObjectWriter::SymbolValue(const MCSymbol &Sym,
                                      const MCAsmLayout &Layout) {
  MCSymbolData &Data = Sym.getData();
  if (Data.isCommon() && Data.isExternal())
    return Data.getCommonAlignment();

  uint64_t Res;
  if (!Layout.getSymbolOffset(Sym, Res))
    return 0;

  if (Layout.getAssembler().isThumbFunc(&Sym))
    Res |= 1;

  return Res;
}

void ELFObjectWriter::ExecutePostLayoutBinding(MCAssembler &Asm,
                                               const MCAsmLayout &Layout) {
  // The presence of symbol versions causes undefined symbols and
  // versions declared with @@@ to be renamed.

  for (const MCSymbol &Alias : Asm.symbols()) {
    MCSymbolData &OriginalData = Alias.getData();

    // Not an alias.
    if (!Alias.isVariable())
      continue;
    auto *Ref = dyn_cast<MCSymbolRefExpr>(Alias.getVariableValue());
    if (!Ref)
      continue;
    const MCSymbol &Symbol = Ref->getSymbol();
    MCSymbolData &SD = Asm.getSymbolData(Symbol);

    StringRef AliasName = Alias.getName();
    size_t Pos = AliasName.find('@');
    if (Pos == StringRef::npos)
      continue;

    // Aliases defined with .symvar copy the binding from the symbol they alias.
    // This is the first place we are able to copy this information.
    OriginalData.setExternal(SD.isExternal());
    MCELF::SetBinding(OriginalData, MCELF::GetBinding(SD));

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

static uint8_t mergeTypeForSet(uint8_t origType, uint8_t newType) {
  uint8_t Type = newType;

  // Propagation rules:
  // IFUNC > FUNC > OBJECT > NOTYPE
  // TLS_OBJECT > OBJECT > NOTYPE
  //
  // dont let the new type degrade the old type
  switch (origType) {
  default:
    break;
  case ELF::STT_GNU_IFUNC:
    if (Type == ELF::STT_FUNC || Type == ELF::STT_OBJECT ||
        Type == ELF::STT_NOTYPE || Type == ELF::STT_TLS)
      Type = ELF::STT_GNU_IFUNC;
    break;
  case ELF::STT_FUNC:
    if (Type == ELF::STT_OBJECT || Type == ELF::STT_NOTYPE ||
        Type == ELF::STT_TLS)
      Type = ELF::STT_FUNC;
    break;
  case ELF::STT_OBJECT:
    if (Type == ELF::STT_NOTYPE)
      Type = ELF::STT_OBJECT;
    break;
  case ELF::STT_TLS:
    if (Type == ELF::STT_OBJECT || Type == ELF::STT_NOTYPE ||
        Type == ELF::STT_GNU_IFUNC || Type == ELF::STT_FUNC)
      Type = ELF::STT_TLS;
    break;
  }

  return Type;
}

void ELFObjectWriter::WriteSymbol(SymbolTableWriter &Writer, ELFSymbolData &MSD,
                                  const MCAsmLayout &Layout) {
  MCSymbolData &OrigData = MSD.Symbol->getData();
  assert((!OrigData.getFragment() ||
          (&OrigData.getFragment()->getParent()->getSection() ==
           &MSD.Symbol->getSection())) &&
         "The symbol's section doesn't match the fragment's symbol");
  const MCSymbol *Base = Layout.getBaseSymbol(*MSD.Symbol);

  // This has to be in sync with when computeSymbolTable uses SHN_ABS or
  // SHN_COMMON.
  bool IsReserved = !Base || OrigData.isCommon();

  // Binding and Type share the same byte as upper and lower nibbles
  uint8_t Binding = MCELF::GetBinding(OrigData);
  uint8_t Type = MCELF::GetType(OrigData);
  MCSymbolData *BaseSD = nullptr;
  if (Base) {
    BaseSD = &Layout.getAssembler().getSymbolData(*Base);
    Type = mergeTypeForSet(Type, MCELF::GetType(*BaseSD));
  }
  uint8_t Info = (Binding << ELF_STB_Shift) | (Type << ELF_STT_Shift);

  // Other and Visibility share the same byte with Visibility using the lower
  // 2 bits
  uint8_t Visibility = MCELF::GetVisibility(OrigData);
  uint8_t Other = MCELF::getOther(OrigData) << (ELF_STO_Shift - ELF_STV_Shift);
  Other |= Visibility;

  uint64_t Value = SymbolValue(*MSD.Symbol, Layout);
  uint64_t Size = 0;

  const MCExpr *ESize = OrigData.getSize();
  if (!ESize && Base)
    ESize = BaseSD->getSize();

  if (ESize) {
    int64_t Res;
    if (!ESize->evaluateKnownAbsolute(Res, Layout))
      report_fatal_error("Size expression must be absolute.");
    Size = Res;
  }

  // Write out the symbol table entry
  Writer.writeSymbol(MSD.StringIndex, Info, Value, Size, Other,
                     MSD.SectionIndex, IsReserved);
}

void ELFObjectWriter::WriteSymbolTable(MCAssembler &Asm,
                                       const MCAsmLayout &Layout,
                                       SectionOffsetsTy &SectionOffsets) {

  MCContext &Ctx = Asm.getContext();

  unsigned EntrySize = is64Bit() ? ELF::SYMENTRY_SIZE64 : ELF::SYMENTRY_SIZE32;

  // Symbol table
  const MCSectionELF *SymtabSection =
      Ctx.getELFSection(".symtab", ELF::SHT_SYMTAB, 0, EntrySize, "");
  MCSectionData &SymtabSD = Asm.getOrCreateSectionData(*SymtabSection);
  SymtabSD.setAlignment(is64Bit() ? 8 : 4);
  SymbolTableIndex = addToSectionTable(SymtabSection);

  // The string table must be emitted first because we need the index
  // into the string table for all the symbol names.

  SymbolTableWriter Writer(*this, is64Bit());

  uint64_t Padding = OffsetToAlignment(OS.tell(), SymtabSD.getAlignment());
  WriteZeros(Padding);

  uint64_t SecStart = OS.tell();

  // The first entry is the undefined symbol entry.
  Writer.writeSymbol(0, 0, 0, 0, 0, 0, false);

  for (unsigned i = 0, e = FileSymbolData.size(); i != e; ++i) {
    Writer.writeSymbol(FileSymbolData[i], ELF::STT_FILE | ELF::STB_LOCAL, 0, 0,
                       ELF::STV_DEFAULT, ELF::SHN_ABS, true);
  }

  // Write the symbol table entries.
  LastLocalSymbolIndex = FileSymbolData.size() + LocalSymbolData.size() + 1;

  for (unsigned i = 0, e = LocalSymbolData.size(); i != e; ++i) {
    ELFSymbolData &MSD = LocalSymbolData[i];
    WriteSymbol(Writer, MSD, Layout);
  }

  for (unsigned i = 0, e = ExternalSymbolData.size(); i != e; ++i) {
    ELFSymbolData &MSD = ExternalSymbolData[i];
    MCSymbolData &Data = MSD.Symbol->getData();
    assert(((Data.getFlags() & ELF_STB_Global) ||
            (Data.getFlags() & ELF_STB_Weak)) &&
           "External symbol requires STB_GLOBAL or STB_WEAK flag");
    WriteSymbol(Writer, MSD, Layout);
    if (MCELF::GetBinding(Data) == ELF::STB_LOCAL)
      LastLocalSymbolIndex++;
  }

  for (unsigned i = 0, e = UndefinedSymbolData.size(); i != e; ++i) {
    ELFSymbolData &MSD = UndefinedSymbolData[i];
    MCSymbolData &Data = MSD.Symbol->getData();
    WriteSymbol(Writer, MSD, Layout);
    if (MCELF::GetBinding(Data) == ELF::STB_LOCAL)
      LastLocalSymbolIndex++;
  }

  uint64_t SecEnd = OS.tell();
  SectionOffsets[SymtabSection] = std::make_pair(SecStart, SecEnd);

  ArrayRef<uint32_t> ShndxIndexes = Writer.getShndxIndexes();
  if (ShndxIndexes.empty())
    return;

  SecStart = OS.tell();
  const MCSectionELF *SymtabShndxSection =
      Ctx.getELFSection(".symtab_shndxr", ELF::SHT_SYMTAB_SHNDX, 0, 4, "");
  addToSectionTable(SymtabShndxSection);
  MCSectionData *SymtabShndxSD =
      &Asm.getOrCreateSectionData(*SymtabShndxSection);
  SymtabShndxSD->setAlignment(4);
  for (uint32_t Index : ShndxIndexes)
    write(Index);
  SecEnd = OS.tell();
  SectionOffsets[SymtabShndxSection] = std::make_pair(SecStart, SecEnd);
}

// It is always valid to create a relocation with a symbol. It is preferable
// to use a relocation with a section if that is possible. Using the section
// allows us to omit some local symbols from the symbol table.
bool ELFObjectWriter::shouldRelocateWithSymbol(const MCAssembler &Asm,
                                               const MCSymbolRefExpr *RefA,
                                               const MCSymbol *Sym, uint64_t C,
                                               unsigned Type) const {
  MCSymbolData *SD = Sym ? &Sym->getData() : nullptr;

  // A PCRel relocation to an absolute value has no symbol (or section). We
  // represent that with a relocation to a null section.
  if (!RefA)
    return false;

  MCSymbolRefExpr::VariantKind Kind = RefA->getKind();
  switch (Kind) {
  default:
    break;
  // The .odp creation emits a relocation against the symbol ".TOC." which
  // create a R_PPC64_TOC relocation. However the relocation symbol name
  // in final object creation should be NULL, since the symbol does not
  // really exist, it is just the reference to TOC base for the current
  // object file. Since the symbol is undefined, returning false results
  // in a relocation with a null section which is the desired result.
  case MCSymbolRefExpr::VK_PPC_TOCBASE:
    return false;

  // These VariantKind cause the relocation to refer to something other than
  // the symbol itself, like a linker generated table. Since the address of
  // symbol is not relevant, we cannot replace the symbol with the
  // section and patch the difference in the addend.
  case MCSymbolRefExpr::VK_GOT:
  case MCSymbolRefExpr::VK_PLT:
  case MCSymbolRefExpr::VK_GOTPCREL:
  case MCSymbolRefExpr::VK_Mips_GOT:
  case MCSymbolRefExpr::VK_PPC_GOT_LO:
  case MCSymbolRefExpr::VK_PPC_GOT_HI:
  case MCSymbolRefExpr::VK_PPC_GOT_HA:
    return true;
  }

  // An undefined symbol is not in any section, so the relocation has to point
  // to the symbol itself.
  assert(Sym && "Expected a symbol");
  if (Sym->isUndefined())
    return true;

  unsigned Binding = MCELF::GetBinding(*SD);
  switch(Binding) {
  default:
    llvm_unreachable("Invalid Binding");
  case ELF::STB_LOCAL:
    break;
  case ELF::STB_WEAK:
    // If the symbol is weak, it might be overridden by a symbol in another
    // file. The relocation has to point to the symbol so that the linker
    // can update it.
    return true;
  case ELF::STB_GLOBAL:
    // Global ELF symbols can be preempted by the dynamic linker. The relocation
    // has to point to the symbol for a reason analogous to the STB_WEAK case.
    return true;
  }

  // If a relocation points to a mergeable section, we have to be careful.
  // If the offset is zero, a relocation with the section will encode the
  // same information. With a non-zero offset, the situation is different.
  // For example, a relocation can point 42 bytes past the end of a string.
  // If we change such a relocation to use the section, the linker would think
  // that it pointed to another string and subtracting 42 at runtime will
  // produce the wrong value.
  auto &Sec = cast<MCSectionELF>(Sym->getSection());
  unsigned Flags = Sec.getFlags();
  if (Flags & ELF::SHF_MERGE) {
    if (C != 0)
      return true;

    // It looks like gold has a bug (http://sourceware.org/PR16794) and can
    // only handle section relocations to mergeable sections if using RELA.
    if (!hasRelocationAddend())
      return true;
  }

  // Most TLS relocations use a got, so they need the symbol. Even those that
  // are just an offset (@tpoff), require a symbol in gold versions before
  // 5efeedf61e4fe720fd3e9a08e6c91c10abb66d42 (2014-09-26) which fixed
  // http://sourceware.org/PR16773.
  if (Flags & ELF::SHF_TLS)
    return true;

  // If the symbol is a thumb function the final relocation must set the lowest
  // bit. With a symbol that is done by just having the symbol have that bit
  // set, so we would lose the bit if we relocated with the section.
  // FIXME: We could use the section but add the bit to the relocation value.
  if (Asm.isThumbFunc(Sym))
    return true;

  if (TargetObjectWriter->needsRelocateWithSymbol(*SD, Type))
    return true;
  return false;
}

static const MCSymbol *getWeakRef(const MCSymbolRefExpr &Ref) {
  const MCSymbol &Sym = Ref.getSymbol();

  if (Ref.getKind() == MCSymbolRefExpr::VK_WEAKREF)
    return &Sym;

  if (!Sym.isVariable())
    return nullptr;

  const MCExpr *Expr = Sym.getVariableValue();
  const auto *Inner = dyn_cast<MCSymbolRefExpr>(Expr);
  if (!Inner)
    return nullptr;

  if (Inner->getKind() == MCSymbolRefExpr::VK_WEAKREF)
    return &Inner->getSymbol();
  return nullptr;
}

// True if the assembler knows nothing about the final value of the symbol.
// This doesn't cover the comdat issues, since in those cases the assembler
// can at least know that all symbols in the section will move together.
static bool isWeak(const MCSymbolData &D) {
  if (MCELF::GetType(D) == ELF::STT_GNU_IFUNC)
    return true;

  switch (MCELF::GetBinding(D)) {
  default:
    llvm_unreachable("Unknown binding");
  case ELF::STB_LOCAL:
    return false;
  case ELF::STB_GLOBAL:
    return false;
  case ELF::STB_WEAK:
  case ELF::STB_GNU_UNIQUE:
    return true;
  }
}

void ELFObjectWriter::RecordRelocation(MCAssembler &Asm,
                                       const MCAsmLayout &Layout,
                                       const MCFragment *Fragment,
                                       const MCFixup &Fixup, MCValue Target,
                                       bool &IsPCRel, uint64_t &FixedValue) {
  const MCSectionData *FixupSectionD = Fragment->getParent();
  const MCSectionELF &FixupSection =
      cast<MCSectionELF>(FixupSectionD->getSection());
  uint64_t C = Target.getConstant();
  uint64_t FixupOffset = Layout.getFragmentOffset(Fragment) + Fixup.getOffset();

  if (const MCSymbolRefExpr *RefB = Target.getSymB()) {
    assert(RefB->getKind() == MCSymbolRefExpr::VK_None &&
           "Should not have constructed this");

    // Let A, B and C being the components of Target and R be the location of
    // the fixup. If the fixup is not pcrel, we want to compute (A - B + C).
    // If it is pcrel, we want to compute (A - B + C - R).

    // In general, ELF has no relocations for -B. It can only represent (A + C)
    // or (A + C - R). If B = R + K and the relocation is not pcrel, we can
    // replace B to implement it: (A - R - K + C)
    if (IsPCRel)
      Asm.getContext().reportFatalError(
          Fixup.getLoc(),
          "No relocation available to represent this relative expression");

    const MCSymbol &SymB = RefB->getSymbol();

    if (SymB.isUndefined())
      Asm.getContext().reportFatalError(
          Fixup.getLoc(),
          Twine("symbol '") + SymB.getName() +
              "' can not be undefined in a subtraction expression");

    assert(!SymB.isAbsolute() && "Should have been folded");
    const MCSection &SecB = SymB.getSection();
    if (&SecB != &FixupSection)
      Asm.getContext().reportFatalError(
          Fixup.getLoc(), "Cannot represent a difference across sections");

    if (::isWeak(SymB.getData()))
      Asm.getContext().reportFatalError(
          Fixup.getLoc(), "Cannot represent a subtraction with a weak symbol");

    uint64_t SymBOffset = Layout.getSymbolOffset(SymB);
    uint64_t K = SymBOffset - FixupOffset;
    IsPCRel = true;
    C -= K;
  }

  // We either rejected the fixup or folded B into C at this point.
  const MCSymbolRefExpr *RefA = Target.getSymA();
  const MCSymbol *SymA = RefA ? &RefA->getSymbol() : nullptr;

  unsigned Type = GetRelocType(Target, Fixup, IsPCRel);
  bool RelocateWithSymbol = shouldRelocateWithSymbol(Asm, RefA, SymA, C, Type);
  if (!RelocateWithSymbol && SymA && !SymA->isUndefined())
    C += Layout.getSymbolOffset(*SymA);

  uint64_t Addend = 0;
  if (hasRelocationAddend()) {
    Addend = C;
    C = 0;
  }

  FixedValue = C;

  // FIXME: What is this!?!?
  MCSymbolRefExpr::VariantKind Modifier =
      RefA ? RefA->getKind() : MCSymbolRefExpr::VK_None;
  if (RelocNeedsGOT(Modifier))
    NeedsGOT = true;

  if (!RelocateWithSymbol) {
    const MCSection *SecA =
        (SymA && !SymA->isUndefined()) ? &SymA->getSection() : nullptr;
    auto *ELFSec = cast_or_null<MCSectionELF>(SecA);
    MCSymbol *SectionSymbol =
        ELFSec ? Asm.getContext().getOrCreateSectionSymbol(*ELFSec)
               : nullptr;
    ELFRelocationEntry Rec(FixupOffset, SectionSymbol, Type, Addend);
    Relocations[&FixupSection].push_back(Rec);
    return;
  }

  if (SymA) {
    if (const MCSymbol *R = Renames.lookup(SymA))
      SymA = R;

    if (const MCSymbol *WeakRef = getWeakRef(*RefA))
      WeakrefUsedInReloc.insert(WeakRef);
    else
      UsedInReloc.insert(SymA);
  }
  ELFRelocationEntry Rec(FixupOffset, SymA, Type, Addend);
  Relocations[&FixupSection].push_back(Rec);
  return;
}


uint64_t
ELFObjectWriter::getSymbolIndexInSymbolTable(const MCAssembler &Asm,
                                             const MCSymbol *S) {
  const MCSymbolData &SD = Asm.getSymbolData(*S);
  return SD.getIndex();
}

bool ELFObjectWriter::isInSymtab(const MCAsmLayout &Layout,
                                 const MCSymbol &Symbol, bool Used,
                                 bool Renamed) {
  const MCSymbolData &Data = Symbol.getData();
  if (Symbol.isVariable()) {
    const MCExpr *Expr = Symbol.getVariableValue();
    if (const MCSymbolRefExpr *Ref = dyn_cast<MCSymbolRefExpr>(Expr)) {
      if (Ref->getKind() == MCSymbolRefExpr::VK_WEAKREF)
        return false;
    }
  }

  if (Used)
    return true;

  if (Renamed)
    return false;

  if (Symbol.getName() == "_GLOBAL_OFFSET_TABLE_")
    return true;

  if (Symbol.isVariable()) {
    const MCSymbol *Base = Layout.getBaseSymbol(Symbol);
    if (Base && Base->isUndefined())
      return false;
  }

  bool IsGlobal = MCELF::GetBinding(Data) == ELF::STB_GLOBAL;
  if (!Symbol.isVariable() && Symbol.isUndefined() && !IsGlobal)
    return false;

  if (Symbol.isTemporary())
    return false;

  return true;
}

bool ELFObjectWriter::isLocal(const MCSymbol &Symbol, bool isUsedInReloc) {
  const MCSymbolData &Data = Symbol.getData();
  if (Data.isExternal())
    return false;

  if (Symbol.isDefined())
    return true;

  if (isUsedInReloc)
    return false;

  return true;
}

void ELFObjectWriter::computeSymbolTable(
    MCAssembler &Asm, const MCAsmLayout &Layout,
    const SectionIndexMapTy &SectionIndexMap,
    const RevGroupMapTy &RevGroupMap) {
  // FIXME: Is this the correct place to do this?
  // FIXME: Why is an undefined reference to _GLOBAL_OFFSET_TABLE_ needed?
  if (NeedsGOT) {
    StringRef Name = "_GLOBAL_OFFSET_TABLE_";
    MCSymbol *Sym = Asm.getContext().getOrCreateSymbol(Name);
    MCSymbolData &Data = Asm.getOrCreateSymbolData(*Sym);
    Data.setExternal(true);
    MCELF::SetBinding(Data, ELF::STB_GLOBAL);
  }

  // Add the data for the symbols.
  for (const MCSymbol &Symbol : Asm.symbols()) {
    MCSymbolData &SD = Symbol.getData();

    bool Used = UsedInReloc.count(&Symbol);
    bool WeakrefUsed = WeakrefUsedInReloc.count(&Symbol);
    bool isSignature = RevGroupMap.count(&Symbol);

    if (!isInSymtab(Layout, Symbol, Used || WeakrefUsed || isSignature,
                    Renames.count(&Symbol)))
      continue;

    ELFSymbolData MSD;
    MSD.Symbol = &Symbol;
    const MCSymbol *BaseSymbol = Layout.getBaseSymbol(Symbol);

    // Undefined symbols are global, but this is the first place we
    // are able to set it.
    bool Local = isLocal(Symbol, Used);
    if (!Local && MCELF::GetBinding(SD) == ELF::STB_LOCAL) {
      assert(BaseSymbol);
      MCSymbolData &BaseData = Asm.getSymbolData(*BaseSymbol);
      MCELF::SetBinding(SD, ELF::STB_GLOBAL);
      MCELF::SetBinding(BaseData, ELF::STB_GLOBAL);
    }

    if (!BaseSymbol) {
      MSD.SectionIndex = ELF::SHN_ABS;
    } else if (SD.isCommon()) {
      assert(!Local);
      MSD.SectionIndex = ELF::SHN_COMMON;
    } else if (BaseSymbol->isUndefined()) {
      if (isSignature && !Used)
        MSD.SectionIndex = RevGroupMap.lookup(&Symbol);
      else
        MSD.SectionIndex = ELF::SHN_UNDEF;
      if (!Used && WeakrefUsed)
        MCELF::SetBinding(SD, ELF::STB_WEAK);
    } else {
      const MCSectionELF &Section =
        static_cast<const MCSectionELF&>(BaseSymbol->getSection());
      MSD.SectionIndex = SectionIndexMap.lookup(&Section);
      assert(MSD.SectionIndex && "Invalid section index!");
    }

    // The @@@ in symbol version is replaced with @ in undefined symbols and @@
    // in defined ones.
    //
    // FIXME: All name handling should be done before we get to the writer,
    // including dealing with GNU-style version suffixes.  Fixing this isn't
    // trivial.
    //
    // We thus have to be careful to not perform the symbol version replacement
    // blindly:
    //
    // The ELF format is used on Windows by the MCJIT engine.  Thus, on
    // Windows, the ELFObjectWriter can encounter symbols mangled using the MS
    // Visual Studio C++ name mangling scheme. Symbols mangled using the MSVC
    // C++ name mangling can legally have "@@@" as a sub-string. In that case,
    // the EFLObjectWriter should not interpret the "@@@" sub-string as
    // specifying GNU-style symbol versioning. The ELFObjectWriter therefore
    // checks for the MSVC C++ name mangling prefix which is either "?", "@?",
    // "__imp_?" or "__imp_@?".
    //
    // It would have been interesting to perform the MS mangling prefix check
    // only when the target triple is of the form *-pc-windows-elf. But, it
    // seems that this information is not easily accessible from the
    // ELFObjectWriter.
    StringRef Name = Symbol.getName();
    if (!Name.startswith("?") && !Name.startswith("@?") &&
        !Name.startswith("__imp_?") && !Name.startswith("__imp_@?")) {
      // This symbol isn't following the MSVC C++ name mangling convention. We
      // can thus safely interpret the @@@ in symbol names as specifying symbol
      // versioning.
      SmallString<32> Buf;
      size_t Pos = Name.find("@@@");
      if (Pos != StringRef::npos) {
        Buf += Name.substr(0, Pos);
        unsigned Skip = MSD.SectionIndex == ELF::SHN_UNDEF ? 2 : 1;
        Buf += Name.substr(Pos + Skip);
        Name = Buf;
      }
    }

    // Sections have their own string table
    if (MCELF::GetType(SD) != ELF::STT_SECTION)
      MSD.Name = StrTabBuilder.add(Name);

    if (MSD.SectionIndex == ELF::SHN_UNDEF)
      UndefinedSymbolData.push_back(MSD);
    else if (Local)
      LocalSymbolData.push_back(MSD);
    else
      ExternalSymbolData.push_back(MSD);
  }

  for (auto i = Asm.file_names_begin(), e = Asm.file_names_end(); i != e; ++i)
    StrTabBuilder.add(*i);

  StrTabBuilder.finalize(StringTableBuilder::ELF);

  for (auto i = Asm.file_names_begin(), e = Asm.file_names_end(); i != e; ++i)
    FileSymbolData.push_back(StrTabBuilder.getOffset(*i));

  for (ELFSymbolData &MSD : LocalSymbolData)
    MSD.StringIndex = MCELF::GetType(MSD.Symbol->getData()) == ELF::STT_SECTION
                          ? 0
                          : StrTabBuilder.getOffset(MSD.Name);
  for (ELFSymbolData &MSD : ExternalSymbolData)
    MSD.StringIndex = StrTabBuilder.getOffset(MSD.Name);
  for (ELFSymbolData& MSD : UndefinedSymbolData)
    MSD.StringIndex = StrTabBuilder.getOffset(MSD.Name);

  // Symbols are required to be in lexicographic order.
  array_pod_sort(LocalSymbolData.begin(), LocalSymbolData.end());
  array_pod_sort(ExternalSymbolData.begin(), ExternalSymbolData.end());
  array_pod_sort(UndefinedSymbolData.begin(), UndefinedSymbolData.end());

  // Set the symbol indices. Local symbols must come before all other
  // symbols with non-local bindings.
  unsigned Index = FileSymbolData.size() + 1;
  for (unsigned i = 0, e = LocalSymbolData.size(); i != e; ++i)
    LocalSymbolData[i].Symbol->getData().setIndex(Index++);

  for (unsigned i = 0, e = ExternalSymbolData.size(); i != e; ++i)
    ExternalSymbolData[i].Symbol->getData().setIndex(Index++);
  for (unsigned i = 0, e = UndefinedSymbolData.size(); i != e; ++i)
    UndefinedSymbolData[i].Symbol->getData().setIndex(Index++);
}

const MCSectionELF *
ELFObjectWriter::createRelocationSection(MCAssembler &Asm,
                                         const MCSectionELF &Sec) {
  if (Relocations[&Sec].empty())
    return nullptr;

  MCContext &Ctx = Asm.getContext();
  const StringRef SectionName = Sec.getSectionName();
  std::string RelaSectionName = hasRelocationAddend() ? ".rela" : ".rel";
  RelaSectionName += SectionName;

  unsigned EntrySize;
  if (hasRelocationAddend())
    EntrySize = is64Bit() ? sizeof(ELF::Elf64_Rela) : sizeof(ELF::Elf32_Rela);
  else
    EntrySize = is64Bit() ? sizeof(ELF::Elf64_Rel) : sizeof(ELF::Elf32_Rel);

  unsigned Flags = 0;
  if (Sec.getFlags() & ELF::SHF_GROUP)
    Flags = ELF::SHF_GROUP;

  const MCSectionELF *RelaSection = Ctx.createELFRelSection(
      RelaSectionName, hasRelocationAddend() ? ELF::SHT_RELA : ELF::SHT_REL,
      Flags, EntrySize, Sec.getGroup(), &Sec);
  MCSectionData &RelSD = Asm.getOrCreateSectionData(*RelaSection);
  RelSD.setAlignment(is64Bit() ? 8 : 4);
  return RelaSection;
}

static SmallVector<char, 128>
getUncompressedData(const MCAsmLayout &Layout,
                    const MCSectionData::FragmentListType &Fragments) {
  SmallVector<char, 128> UncompressedData;
  for (const MCFragment &F : Fragments) {
    const SmallVectorImpl<char> *Contents;
    switch (F.getKind()) {
    case MCFragment::FT_Data:
      Contents = &cast<MCDataFragment>(F).getContents();
      break;
    case MCFragment::FT_Dwarf:
      Contents = &cast<MCDwarfLineAddrFragment>(F).getContents();
      break;
    case MCFragment::FT_DwarfFrame:
      Contents = &cast<MCDwarfCallFrameFragment>(F).getContents();
      break;
    default:
      llvm_unreachable(
          "Not expecting any other fragment types in a debug_* section");
    }
    UncompressedData.append(Contents->begin(), Contents->end());
  }
  return UncompressedData;
}

// Include the debug info compression header:
// "ZLIB" followed by 8 bytes representing the uncompressed size of the section,
// useful for consumers to preallocate a buffer to decompress into.
static bool
prependCompressionHeader(uint64_t Size,
                         SmallVectorImpl<char> &CompressedContents) {
  const StringRef Magic = "ZLIB";
  if (Size <= Magic.size() + sizeof(Size) + CompressedContents.size())
    return false;
  if (sys::IsLittleEndianHost)
    sys::swapByteOrder(Size);
  CompressedContents.insert(CompressedContents.begin(),
                            Magic.size() + sizeof(Size), 0);
  std::copy(Magic.begin(), Magic.end(), CompressedContents.begin());
  std::copy(reinterpret_cast<char *>(&Size),
            reinterpret_cast<char *>(&Size + 1),
            CompressedContents.begin() + Magic.size());
  return true;
}

void ELFObjectWriter::writeSectionData(const MCAssembler &Asm,
                                       const MCSectionData &SD,
                                       const MCAsmLayout &Layout) {
  const MCSectionELF &Section =
      static_cast<const MCSectionELF &>(SD.getSection());
  StringRef SectionName = Section.getSectionName();

  // Compressing debug_frame requires handling alignment fragments which is
  // more work (possibly generalizing MCAssembler.cpp:writeFragment to allow
  // for writing to arbitrary buffers) for little benefit.
  if (!Asm.getContext().getAsmInfo()->compressDebugSections() ||
      !SectionName.startswith(".debug_") || SectionName == ".debug_frame") {
    Asm.writeSectionData(&SD, Layout);
    return;
  }

  // Gather the uncompressed data from all the fragments.
  const MCSectionData::FragmentListType &Fragments = SD.getFragmentList();
  SmallVector<char, 128> UncompressedData =
      getUncompressedData(Layout, Fragments);

  SmallVector<char, 128> CompressedContents;
  zlib::Status Success = zlib::compress(
      StringRef(UncompressedData.data(), UncompressedData.size()),
      CompressedContents);
  if (Success != zlib::StatusOK) {
    Asm.writeSectionData(&SD, Layout);
    return;
  }

  if (!prependCompressionHeader(UncompressedData.size(), CompressedContents)) {
    Asm.writeSectionData(&SD, Layout);
    return;
  }
  Asm.getContext().renameELFSection(&Section,
                                    (".z" + SectionName.drop_front(1)).str());
  OS << CompressedContents;
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

void ELFObjectWriter::writeRelocations(const MCAssembler &Asm,
                                       const MCSectionELF &Sec) {
  std::vector<ELFRelocationEntry> &Relocs = Relocations[&Sec];

  // Sort the relocation entries. Most targets just sort by Offset, but some
  // (e.g., MIPS) have additional constraints.
  TargetObjectWriter->sortRelocs(Asm, Relocs);

  for (unsigned i = 0, e = Relocs.size(); i != e; ++i) {
    const ELFRelocationEntry &Entry = Relocs[e - i - 1];
    unsigned Index =
        Entry.Symbol ? getSymbolIndexInSymbolTable(Asm, Entry.Symbol) : 0;

    if (is64Bit()) {
      write(Entry.Offset);
      if (TargetObjectWriter->isN64()) {
        write(uint32_t(Index));

        write(TargetObjectWriter->getRSsym(Entry.Type));
        write(TargetObjectWriter->getRType3(Entry.Type));
        write(TargetObjectWriter->getRType2(Entry.Type));
        write(TargetObjectWriter->getRType(Entry.Type));
      } else {
        struct ELF::Elf64_Rela ERE64;
        ERE64.setSymbolAndType(Index, Entry.Type);
        write(ERE64.r_info);
      }
      if (hasRelocationAddend())
        write(Entry.Addend);
    } else {
      write(uint32_t(Entry.Offset));

      struct ELF::Elf32_Rela ERE32;
      ERE32.setSymbolAndType(Index, Entry.Type);
      write(ERE32.r_info);

      if (hasRelocationAddend())
        write(uint32_t(Entry.Addend));
    }
  }
}

const MCSectionELF *ELFObjectWriter::createSectionHeaderStringTable() {
  const MCSectionELF *ShstrtabSection = SectionTable[ShstrtabIndex - 1];
  ShStrTabBuilder.finalize(StringTableBuilder::ELF);
  OS << ShStrTabBuilder.data();
  return ShstrtabSection;
}

const MCSectionELF *ELFObjectWriter::createStringTable(MCContext &Ctx) {
  const MCSectionELF *StrtabSection =
      Ctx.getELFSection(".strtab", ELF::SHT_STRTAB, 0);
  StringTableIndex = addToSectionTable(StrtabSection);
  OS << StrTabBuilder.data();
  return StrtabSection;
}

void ELFObjectWriter::writeSection(MCAssembler &Asm,
                                   const SectionIndexMapTy &SectionIndexMap,
                                   uint32_t GroupSymbolIndex,
                                   uint64_t Offset, uint64_t Size,
                                   uint64_t Alignment,
                                   const MCSectionELF &Section) {
  uint64_t sh_link = 0;
  uint64_t sh_info = 0;

  switch(Section.getType()) {
  default:
    // Nothing to do.
    break;

  case ELF::SHT_DYNAMIC:
    llvm_unreachable("SHT_DYNAMIC in a relocatable object");

  case ELF::SHT_REL:
  case ELF::SHT_RELA: {
    sh_link = SymbolTableIndex;
    assert(sh_link && ".symtab not found");
    const MCSectionELF *InfoSection = Section.getAssociatedSection();
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

  case ELF::SHT_GROUP:
    sh_link = SymbolTableIndex;
    sh_info = GroupSymbolIndex;
    break;
  }

  if (TargetObjectWriter->getEMachine() == ELF::EM_ARM &&
      Section.getType() == ELF::SHT_ARM_EXIDX)
    sh_link = SectionIndexMap.lookup(Section.getAssociatedSection());

  WriteSecHdrEntry(ShStrTabBuilder.getOffset(Section.getSectionName()),
                   Section.getType(),
                   Section.getFlags(), 0, Offset, Size, sh_link, sh_info,
                   Alignment, Section.getEntrySize());
}

void ELFObjectWriter::writeSectionHeader(
    MCAssembler &Asm, const MCAsmLayout &Layout,
    const SectionIndexMapTy &SectionIndexMap,
    const SectionOffsetsTy &SectionOffsets) {
  const unsigned NumSections = SectionTable.size();

  // Null section first.
  uint64_t FirstSectionSize =
      (NumSections + 1) >= ELF::SHN_LORESERVE ? NumSections + 1 : 0;
  WriteSecHdrEntry(0, 0, 0, 0, 0, FirstSectionSize, 0, 0, 0, 0);

  for (const MCSectionELF *Section : SectionTable) {
    const MCSectionData &SD = Asm.getOrCreateSectionData(*Section);
    uint32_t GroupSymbolIndex;
    unsigned Type = Section->getType();
    if (Type != ELF::SHT_GROUP)
      GroupSymbolIndex = 0;
    else
      GroupSymbolIndex = getSymbolIndexInSymbolTable(Asm, Section->getGroup());

    const std::pair<uint64_t, uint64_t> &Offsets =
        SectionOffsets.find(Section)->second;
    uint64_t Size = Type == ELF::SHT_NOBITS ? Layout.getSectionAddressSize(&SD)
                                            : Offsets.second - Offsets.first;

    writeSection(Asm, SectionIndexMap, GroupSymbolIndex, Offsets.first, Size,
                 SD.getAlignment(), *Section);
  }
}

void ELFObjectWriter::WriteObject(MCAssembler &Asm,
                                  const MCAsmLayout &Layout) {
  MCContext &Ctx = Asm.getContext();
  const MCSectionELF *ShstrtabSection =
      Ctx.getELFSection(".shstrtab", ELF::SHT_STRTAB, 0);
  ShstrtabIndex = addToSectionTable(ShstrtabSection);

  RevGroupMapTy RevGroupMap;
  SectionIndexMapTy SectionIndexMap;

  std::map<const MCSymbol *, std::vector<const MCSectionELF *>> GroupMembers;

  // Write out the ELF header ...
  writeHeader(Asm);

  // ... then the sections ...
  SectionOffsetsTy SectionOffsets;
  bool ComputedSymtab = false;
  for (const MCSectionData &SD : Asm) {
    const MCSectionELF &Section =
        static_cast<const MCSectionELF &>(SD.getSection());

    uint64_t Padding = OffsetToAlignment(OS.tell(), SD.getAlignment());
    WriteZeros(Padding);

    // Remember the offset into the file for this section.
    uint64_t SecStart = OS.tell();

    const MCSymbol *SignatureSymbol = Section.getGroup();
    unsigned Type = Section.getType();
    if (Type == ELF::SHT_GROUP) {
      assert(SignatureSymbol);
      write(uint32_t(ELF::GRP_COMDAT));
      for (const MCSectionELF *Member : GroupMembers[SignatureSymbol]) {
        uint32_t SecIndex = SectionIndexMap.lookup(Member);
        write(SecIndex);
      }
    } else if (Type == ELF::SHT_REL || Type == ELF::SHT_RELA) {
      if (!ComputedSymtab) {
        // Compute symbol table information.
        computeSymbolTable(Asm, Layout, SectionIndexMap, RevGroupMap);
        ComputedSymtab = true;
      }
      writeRelocations(Asm, *Section.getAssociatedSection());
    } else {
      writeSectionData(Asm, SD, Layout);
    }

    uint64_t SecEnd = OS.tell();
    SectionOffsets[&Section] = std::make_pair(SecStart, SecEnd);

    if (Type == ELF::SHT_GROUP || Type == ELF::SHT_REL || Type == ELF::SHT_RELA)
      continue;

    const MCSectionELF *RelSection = createRelocationSection(Asm, Section);

    if (SignatureSymbol) {
      Asm.getOrCreateSymbolData(*SignatureSymbol);
      unsigned &GroupIdx = RevGroupMap[SignatureSymbol];
      if (!GroupIdx) {
        const MCSectionELF *Group = Ctx.createELFGroupSection(SignatureSymbol);
        GroupIdx = addToSectionTable(Group);
        MCSectionData *GroupD = &Asm.getOrCreateSectionData(*Group);
        GroupD->setAlignment(4);
      }
      GroupMembers[SignatureSymbol].push_back(&Section);
      if (RelSection)
        GroupMembers[SignatureSymbol].push_back(RelSection);
    }

    SectionIndexMap[&Section] = addToSectionTable(&Section);
    if (RelSection)
      SectionIndexMap[RelSection] = addToSectionTable(RelSection);
  }

  if (!ComputedSymtab) {
    // Compute symbol table information.
    computeSymbolTable(Asm, Layout, SectionIndexMap, RevGroupMap);
    ComputedSymtab = true;
  }

  WriteSymbolTable(Asm, Layout, SectionOffsets);

  {
    uint64_t SecStart = OS.tell();
    const MCSectionELF *Sec = createStringTable(Ctx);
    uint64_t SecEnd = OS.tell();
    SectionOffsets[Sec] = std::make_pair(SecStart, SecEnd);
  }

  {
    uint64_t SecStart = OS.tell();
    const MCSectionELF *Sec = createSectionHeaderStringTable();
    uint64_t SecEnd = OS.tell();
    SectionOffsets[Sec] = std::make_pair(SecStart, SecEnd);
  }

  uint64_t NaturalAlignment = is64Bit() ? 8 : 4;
  uint64_t Padding = OffsetToAlignment(OS.tell(), NaturalAlignment);
  WriteZeros(Padding);

  const unsigned SectionHeaderOffset = OS.tell();

  // ... then the section header table ...
  writeSectionHeader(Asm, Layout, SectionIndexMap, SectionOffsets);

  uint16_t NumSections = (SectionTable.size() + 1 >= ELF::SHN_LORESERVE)
                             ? (uint16_t)ELF::SHN_UNDEF
                             : SectionTable.size() + 1;
  if (sys::IsLittleEndianHost != IsLittleEndian)
    sys::swapByteOrder(NumSections);
  unsigned NumSectionsOffset;

  if (is64Bit()) {
    uint64_t Val = SectionHeaderOffset;
    if (sys::IsLittleEndianHost != IsLittleEndian)
      sys::swapByteOrder(Val);
    OS.pwrite(reinterpret_cast<char *>(&Val), sizeof(Val),
              offsetof(ELF::Elf64_Ehdr, e_shoff));
    NumSectionsOffset = offsetof(ELF::Elf64_Ehdr, e_shnum);
  } else {
    uint32_t Val = SectionHeaderOffset;
    if (sys::IsLittleEndianHost != IsLittleEndian)
      sys::swapByteOrder(Val);
    OS.pwrite(reinterpret_cast<char *>(&Val), sizeof(Val),
              offsetof(ELF::Elf32_Ehdr, e_shoff));
    NumSectionsOffset = offsetof(ELF::Elf32_Ehdr, e_shnum);
  }
  OS.pwrite(reinterpret_cast<char *>(&NumSections), sizeof(NumSections),
            NumSectionsOffset);
}

bool ELFObjectWriter::IsSymbolRefDifferenceFullyResolvedImpl(
    const MCAssembler &Asm, const MCSymbol &SymA, const MCFragment &FB,
    bool InSet, bool IsPCRel) const {
  if (IsPCRel) {
    assert(!InSet);
    if (::isWeak(SymA.getData()))
      return false;
  }
  return MCObjectWriter::IsSymbolRefDifferenceFullyResolvedImpl(Asm, SymA, FB,
                                                                InSet, IsPCRel);
}

bool ELFObjectWriter::isWeak(const MCSymbol &Sym) const {
  const MCSymbolData &SD = Sym.getData();
  if (::isWeak(SD))
    return true;

  // It is invalid to replace a reference to a global in a comdat
  // with a reference to a local since out of comdat references
  // to a local are forbidden.
  // We could try to return false for more cases, like the reference
  // being in the same comdat or Sym being an alias to another global,
  // but it is not clear if it is worth the effort.
  if (MCELF::GetBinding(SD) != ELF::STB_GLOBAL)
    return false;

  if (!Sym.isInSection())
    return false;

  const auto &Sec = cast<MCSectionELF>(Sym.getSection());
  return Sec.getGroup();
}

MCObjectWriter *llvm::createELFObjectWriter(MCELFObjectTargetWriter *MOTW,
                                            raw_pwrite_stream &OS,
                                            bool IsLittleEndian) {
  return new ELFObjectWriter(MOTW, OS, IsLittleEndian);
}
