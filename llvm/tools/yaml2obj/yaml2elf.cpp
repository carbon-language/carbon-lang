//===- yaml2elf - Convert YAML to a ELF object file -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief The ELF component of yaml2obj.
///
//===----------------------------------------------------------------------===//

#include "yaml2obj.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFYAML.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

// There is similar code in yaml2coff, but with some slight COFF-specific
// variations like different initial state. Might be able to deduplicate
// some day, but also want to make sure that the Mach-O use case is served.
//
// This class has a deliberately small interface, since a lot of
// implementation variation is possible.
//
// TODO: Use an ordered container with a suffix-based comparison in order
// to deduplicate suffixes. std::map<> with a custom comparator is likely
// to be the simplest implementation, but a suffix trie could be more
// suitable for the job.
namespace {
class StringTableBuilder {
  /// \brief Indices of strings currently present in `Buf`.
  StringMap<unsigned> StringIndices;
  /// \brief The contents of the string table as we build it.
  std::string Buf;
public:
  StringTableBuilder() {
    Buf.push_back('\0');
  }
  /// \returns Index of string in string table.
  unsigned addString(StringRef S) {
    StringMapEntry<unsigned> &Entry = StringIndices.GetOrCreateValue(S);
    unsigned &I = Entry.getValue();
    if (I != 0)
      return I;
    I = Buf.size();
    Buf.append(S.begin(), S.end());
    Buf.push_back('\0');
    return I;
  }
  size_t size() const {
    return Buf.size();
  }
  void writeToStream(raw_ostream &OS) {
    OS.write(Buf.data(), Buf.size());
  }
};
} // end anonymous namespace

// This class is used to build up a contiguous binary blob while keeping
// track of an offset in the output (which notionally begins at
// `InitialOffset`).
namespace {
class ContiguousBlobAccumulator {
  const uint64_t InitialOffset;
  SmallVector<char, 128> Buf;
  raw_svector_ostream OS;

  /// \returns The new offset.
  uint64_t padToAlignment(unsigned Align) {
    uint64_t CurrentOffset = InitialOffset + OS.tell();
    uint64_t AlignedOffset = RoundUpToAlignment(CurrentOffset, Align);
    for (; CurrentOffset != AlignedOffset; ++CurrentOffset)
      OS.write('\0');
    return AlignedOffset; // == CurrentOffset;
  }

public:
  ContiguousBlobAccumulator(uint64_t InitialOffset_)
      : InitialOffset(InitialOffset_), Buf(), OS(Buf) {}
  template <class Integer>
  raw_ostream &getOSAndAlignedOffset(Integer &Offset, unsigned Align = 16) {
    Offset = padToAlignment(Align);
    return OS;
  }
  void writeBlobToStream(raw_ostream &Out) { Out << OS.str(); }
};
} // end anonymous namespace

// Used to keep track of section and symbol names, so that in the YAML file
// sections and symbols can be referenced by name instead of by index.
namespace {
class NameToIdxMap {
  StringMap<int> Map;
public:
  /// \returns true if name is already present in the map.
  bool addName(StringRef Name, unsigned i) {
    StringMapEntry<int> &Entry = Map.GetOrCreateValue(Name, -1);
    if (Entry.getValue() != -1)
      return true;
    Entry.setValue((int)i);
    return false;
  }
  /// \returns true if name is not present in the map
  bool lookup(StringRef Name, unsigned &Idx) const {
    StringMap<int>::const_iterator I = Map.find(Name);
    if (I == Map.end())
      return true;
    Idx = I->getValue();
    return false;
  }
};
} // end anonymous namespace

template <class T>
static size_t arrayDataSize(ArrayRef<T> A) {
  return A.size() * sizeof(T);
}

template <class T>
static void writeArrayData(raw_ostream &OS, ArrayRef<T> A) {
  OS.write((const char *)A.data(), arrayDataSize(A));
}

template <class T>
static void zero(T &Obj) {
  memset(&Obj, 0, sizeof(Obj));
}

namespace {
/// \brief "Single point of truth" for the ELF file construction.
/// TODO: This class still has a ways to go before it is truly a "single
/// point of truth".
template <class ELFT>
class ELFState {
  typedef typename object::ELFFile<ELFT>::Elf_Ehdr Elf_Ehdr;
  typedef typename object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename object::ELFFile<ELFT>::Elf_Rel Elf_Rel;
  typedef typename object::ELFFile<ELFT>::Elf_Rela Elf_Rela;

  /// \brief The future ".strtab" section.
  StringTableBuilder DotStrtab;

  /// \brief The future ".shstrtab" section.
  StringTableBuilder DotShStrtab;

  NameToIdxMap SN2I;
  NameToIdxMap SymN2I;
  const ELFYAML::Object &Doc;

  bool buildSectionIndex();
  bool buildSymbolIndex(std::size_t &StartIndex,
                        const std::vector<ELFYAML::Symbol> &Symbols);
  void initELFHeader(Elf_Ehdr &Header);
  bool initSectionHeaders(std::vector<Elf_Shdr> &SHeaders,
                          ContiguousBlobAccumulator &CBA);
  void initSymtabSectionHeader(Elf_Shdr &SHeader,
                               ContiguousBlobAccumulator &CBA);
  void initStrtabSectionHeader(Elf_Shdr &SHeader, StringRef Name,
                               StringTableBuilder &STB,
                               ContiguousBlobAccumulator &CBA);
  void addSymbols(const std::vector<ELFYAML::Symbol> &Symbols,
                  std::vector<Elf_Sym> &Syms, unsigned SymbolBinding);
  void writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::RawContentSection &Section,
                           ContiguousBlobAccumulator &CBA);
  bool writeSectionContent(Elf_Shdr &SHeader,
                           const ELFYAML::RelocationSection &Section,
                           ContiguousBlobAccumulator &CBA);

  // - SHT_NULL entry (placed first, i.e. 0'th entry)
  // - symbol table (.symtab) (placed third to last)
  // - string table (.strtab) (placed second to last)
  // - section header string table (.shstrtab) (placed last)
  unsigned getDotSymTabSecNo() const { return Doc.Sections.size() + 1; }
  unsigned getDotStrTabSecNo() const { return Doc.Sections.size() + 2; }
  unsigned getDotShStrTabSecNo() const { return Doc.Sections.size() + 3; }
  unsigned getSectionCount() const { return Doc.Sections.size() + 4; }

  ELFState(const ELFYAML::Object &D) : Doc(D) {}

public:
  static int writeELF(raw_ostream &OS, const ELFYAML::Object &Doc);
};
} // end anonymous namespace

template <class ELFT>
void ELFState<ELFT>::initELFHeader(Elf_Ehdr &Header) {
  using namespace llvm::ELF;
  zero(Header);
  Header.e_ident[EI_MAG0] = 0x7f;
  Header.e_ident[EI_MAG1] = 'E';
  Header.e_ident[EI_MAG2] = 'L';
  Header.e_ident[EI_MAG3] = 'F';
  Header.e_ident[EI_CLASS] = ELFT::Is64Bits ? ELFCLASS64 : ELFCLASS32;
  bool IsLittleEndian = ELFT::TargetEndianness == support::little;
  Header.e_ident[EI_DATA] = IsLittleEndian ? ELFDATA2LSB : ELFDATA2MSB;
  Header.e_ident[EI_VERSION] = EV_CURRENT;
  Header.e_ident[EI_OSABI] = Doc.Header.OSABI;
  Header.e_ident[EI_ABIVERSION] = 0;
  Header.e_type = Doc.Header.Type;
  Header.e_machine = Doc.Header.Machine;
  Header.e_version = EV_CURRENT;
  Header.e_entry = Doc.Header.Entry;
  Header.e_flags = Doc.Header.Flags;
  Header.e_ehsize = sizeof(Elf_Ehdr);
  Header.e_shentsize = sizeof(Elf_Shdr);
  // Immediately following the ELF header.
  Header.e_shoff = sizeof(Header);
  Header.e_shnum = getSectionCount();
  Header.e_shstrndx = getDotShStrTabSecNo();
}

template <class ELFT>
bool ELFState<ELFT>::initSectionHeaders(std::vector<Elf_Shdr> &SHeaders,
                                        ContiguousBlobAccumulator &CBA) {
  // Ensure SHN_UNDEF entry is present. An all-zero section header is a
  // valid SHN_UNDEF entry since SHT_NULL == 0.
  Elf_Shdr SHeader;
  zero(SHeader);
  SHeaders.push_back(SHeader);

  for (const auto &Sec : Doc.Sections) {
    zero(SHeader);
    SHeader.sh_name = DotShStrtab.addString(Sec->Name);
    SHeader.sh_type = Sec->Type;
    SHeader.sh_flags = Sec->Flags;
    SHeader.sh_addr = Sec->Address;
    SHeader.sh_addralign = Sec->AddressAlign;

    if (!Sec->Link.empty()) {
      unsigned Index;
      if (SN2I.lookup(Sec->Link, Index)) {
        errs() << "error: Unknown section referenced: '" << Sec->Link
               << "' at YAML section '" << Sec->Name << "'.\n";
        return false;
      }
      SHeader.sh_link = Index;
    }

    if (auto S = dyn_cast<ELFYAML::RawContentSection>(Sec.get()))
      writeSectionContent(SHeader, *S, CBA);
    else if (auto S = dyn_cast<ELFYAML::RelocationSection>(Sec.get())) {
      if (S->Link.empty())
        // For relocation section set link to .symtab by default.
        SHeader.sh_link = getDotSymTabSecNo();

      unsigned Index;
      if (SN2I.lookup(S->Info, Index)) {
        errs() << "error: Unknown section referenced: '" << S->Info
               << "' at YAML section '" << S->Name << "'.\n";
        return false;
      }
      SHeader.sh_info = Index;

      if (!writeSectionContent(SHeader, *S, CBA))
        return false;
    } else
      llvm_unreachable("Unknown section type");

    SHeaders.push_back(SHeader);
  }
  return true;
}

template <class ELFT>
void ELFState<ELFT>::initSymtabSectionHeader(Elf_Shdr &SHeader,
                                             ContiguousBlobAccumulator &CBA) {
  zero(SHeader);
  SHeader.sh_name = DotShStrtab.addString(StringRef(".symtab"));
  SHeader.sh_type = ELF::SHT_SYMTAB;
  SHeader.sh_link = getDotStrTabSecNo();
  // One greater than symbol table index of the last local symbol.
  SHeader.sh_info = Doc.Symbols.Local.size() + 1;
  SHeader.sh_entsize = sizeof(Elf_Sym);

  std::vector<Elf_Sym> Syms;
  {
    // Ensure STN_UNDEF is present
    Elf_Sym Sym;
    zero(Sym);
    Syms.push_back(Sym);
  }
  addSymbols(Doc.Symbols.Local, Syms, ELF::STB_LOCAL);
  addSymbols(Doc.Symbols.Global, Syms, ELF::STB_GLOBAL);
  addSymbols(Doc.Symbols.Weak, Syms, ELF::STB_WEAK);

  writeArrayData(CBA.getOSAndAlignedOffset(SHeader.sh_offset),
                 makeArrayRef(Syms));
  SHeader.sh_size = arrayDataSize(makeArrayRef(Syms));
}

template <class ELFT>
void ELFState<ELFT>::initStrtabSectionHeader(Elf_Shdr &SHeader, StringRef Name,
                                             StringTableBuilder &STB,
                                             ContiguousBlobAccumulator &CBA) {
  zero(SHeader);
  SHeader.sh_name = DotShStrtab.addString(Name);
  SHeader.sh_type = ELF::SHT_STRTAB;
  STB.writeToStream(CBA.getOSAndAlignedOffset(SHeader.sh_offset));
  SHeader.sh_size = STB.size();
  SHeader.sh_addralign = 1;
}

template <class ELFT>
void ELFState<ELFT>::addSymbols(const std::vector<ELFYAML::Symbol> &Symbols,
                                std::vector<Elf_Sym> &Syms,
                                unsigned SymbolBinding) {
  for (const auto &Sym : Symbols) {
    Elf_Sym Symbol;
    zero(Symbol);
    if (!Sym.Name.empty())
      Symbol.st_name = DotStrtab.addString(Sym.Name);
    Symbol.setBindingAndType(SymbolBinding, Sym.Type);
    if (!Sym.Section.empty()) {
      unsigned Index;
      if (SN2I.lookup(Sym.Section, Index)) {
        errs() << "error: Unknown section referenced: '" << Sym.Section
               << "' by YAML symbol " << Sym.Name << ".\n";
        exit(1);
      }
      Symbol.st_shndx = Index;
    } // else Symbol.st_shndex == SHN_UNDEF (== 0), since it was zero'd earlier.
    Symbol.st_value = Sym.Value;
    Symbol.st_size = Sym.Size;
    Syms.push_back(Symbol);
  }
}

template <class ELFT>
void
ELFState<ELFT>::writeSectionContent(Elf_Shdr &SHeader,
                                    const ELFYAML::RawContentSection &Section,
                                    ContiguousBlobAccumulator &CBA) {
  Section.Content.writeAsBinary(CBA.getOSAndAlignedOffset(SHeader.sh_offset));
  SHeader.sh_entsize = 0;
  SHeader.sh_size = Section.Content.binary_size();
}

template <class ELFT>
bool
ELFState<ELFT>::writeSectionContent(Elf_Shdr &SHeader,
                                    const ELFYAML::RelocationSection &Section,
                                    ContiguousBlobAccumulator &CBA) {
  if (Section.Type != llvm::ELF::SHT_REL &&
      Section.Type != llvm::ELF::SHT_RELA) {
    errs() << "error: Invalid relocation section type.\n";
    return false;
  }

  bool IsRela = Section.Type == llvm::ELF::SHT_RELA;
  SHeader.sh_entsize = IsRela ? sizeof(Elf_Rela) : sizeof(Elf_Rel);
  SHeader.sh_size = SHeader.sh_entsize * Section.Relocations.size();

  auto &OS = CBA.getOSAndAlignedOffset(SHeader.sh_offset);

  for (const auto &Rel : Section.Relocations) {
    unsigned SymIdx;
    if (SymN2I.lookup(Rel.Symbol, SymIdx)) {
      errs() << "error: Unknown symbol referenced: '" << Rel.Symbol
             << "' at YAML relocation.\n";
      return false;
    }

    if (IsRela) {
      Elf_Rela REntry;
      zero(REntry);
      REntry.r_offset = Rel.Offset;
      REntry.r_addend = Rel.Addend;
      REntry.setSymbolAndType(SymIdx, Rel.Type);
      OS.write((const char *)&REntry, sizeof(REntry));
    } else {
      Elf_Rel REntry;
      zero(REntry);
      REntry.r_offset = Rel.Offset;
      REntry.setSymbolAndType(SymIdx, Rel.Type);
      OS.write((const char *)&REntry, sizeof(REntry));
    }
  }
  return true;
}

template <class ELFT> bool ELFState<ELFT>::buildSectionIndex() {
  SN2I.addName(".symtab", getDotSymTabSecNo());
  SN2I.addName(".strtab", getDotStrTabSecNo());
  SN2I.addName(".shstrtab", getDotShStrTabSecNo());

  for (unsigned i = 0, e = Doc.Sections.size(); i != e; ++i) {
    StringRef Name = Doc.Sections[i]->Name;
    if (Name.empty())
      continue;
    // "+ 1" to take into account the SHT_NULL entry.
    if (SN2I.addName(Name, i + 1)) {
      errs() << "error: Repeated section name: '" << Name
             << "' at YAML section number " << i << ".\n";
      return false;
    }
  }
  return true;
}

template <class ELFT>
bool
ELFState<ELFT>::buildSymbolIndex(std::size_t &StartIndex,
                                 const std::vector<ELFYAML::Symbol> &Symbols) {
  for (const auto &Sym : Symbols) {
    ++StartIndex;
    if (Sym.Name.empty())
      continue;
    if (SymN2I.addName(Sym.Name, StartIndex)) {
      errs() << "error: Repeated symbol name: '" << Sym.Name << "'.\n";
      return false;
    }
  }
  return true;
}

template <class ELFT>
int ELFState<ELFT>::writeELF(raw_ostream &OS, const ELFYAML::Object &Doc) {
  ELFState<ELFT> State(Doc);
  if (!State.buildSectionIndex())
    return 1;

  std::size_t StartSymIndex = 0;
  if (!State.buildSymbolIndex(StartSymIndex, Doc.Symbols.Local) ||
      !State.buildSymbolIndex(StartSymIndex, Doc.Symbols.Global) ||
      !State.buildSymbolIndex(StartSymIndex, Doc.Symbols.Weak))
    return 1;

  Elf_Ehdr Header;
  State.initELFHeader(Header);

  // TODO: Flesh out section header support.
  // TODO: Program headers.

  // XXX: This offset is tightly coupled with the order that we write
  // things to `OS`.
  const size_t SectionContentBeginOffset =
      Header.e_ehsize + Header.e_shentsize * Header.e_shnum;
  ContiguousBlobAccumulator CBA(SectionContentBeginOffset);

  std::vector<Elf_Shdr> SHeaders;
  if(!State.initSectionHeaders(SHeaders, CBA))
    return 1;

  // .symtab section.
  Elf_Shdr SymtabSHeader;
  State.initSymtabSectionHeader(SymtabSHeader, CBA);
  SHeaders.push_back(SymtabSHeader);

  // .strtab string table header.
  Elf_Shdr DotStrTabSHeader;
  State.initStrtabSectionHeader(DotStrTabSHeader, ".strtab", State.DotStrtab,
                                CBA);
  SHeaders.push_back(DotStrTabSHeader);

  // .shstrtab string table header.
  Elf_Shdr ShStrTabSHeader;
  State.initStrtabSectionHeader(ShStrTabSHeader, ".shstrtab", State.DotShStrtab,
                                CBA);
  SHeaders.push_back(ShStrTabSHeader);

  OS.write((const char *)&Header, sizeof(Header));
  writeArrayData(OS, makeArrayRef(SHeaders));
  CBA.writeBlobToStream(OS);
  return 0;
}

static bool is64Bit(const ELFYAML::Object &Doc) {
  return Doc.Header.Class == ELFYAML::ELF_ELFCLASS(ELF::ELFCLASS64);
}

static bool isLittleEndian(const ELFYAML::Object &Doc) {
  return Doc.Header.Data == ELFYAML::ELF_ELFDATA(ELF::ELFDATA2LSB);
}

int yaml2elf(llvm::raw_ostream &Out, llvm::MemoryBuffer *Buf) {
  yaml::Input YIn(Buf->getBuffer());
  ELFYAML::Object Doc;
  YIn >> Doc;
  if (YIn.error()) {
    errs() << "yaml2obj: Failed to parse YAML file!\n";
    return 1;
  }
  using object::ELFType;
  typedef ELFType<support::little, 8, true> LE64;
  typedef ELFType<support::big, 8, true> BE64;
  typedef ELFType<support::little, 4, false> LE32;
  typedef ELFType<support::big, 4, false> BE32;
  if (is64Bit(Doc)) {
    if (isLittleEndian(Doc))
      return ELFState<LE64>::writeELF(outs(), Doc);
    else
      return ELFState<BE64>::writeELF(outs(), Doc);
  } else {
    if (isLittleEndian(Doc))
      return ELFState<LE32>::writeELF(outs(), Doc);
    else
      return ELFState<BE32>::writeELF(outs(), Doc);
  }
}
