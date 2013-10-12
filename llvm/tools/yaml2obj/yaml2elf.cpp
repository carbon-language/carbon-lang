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

// Used to keep track of section names, so that in the YAML file sections
// can be referenced by name instead of by index.
namespace {
class SectionNameToIdxMap {
  StringMap<int> Map;
public:
  /// \returns true if name is already present in the map.
  bool addName(StringRef SecName, unsigned i) {
    StringMapEntry<int> &Entry = Map.GetOrCreateValue(SecName, -1);
    if (Entry.getValue() != -1)
      return true;
    Entry.setValue((int)i);
    return false;
  }
  /// \returns true if name is not present in the map
  bool lookupSection(StringRef SecName, unsigned &Idx) const {
    StringMap<int>::const_iterator I = Map.find(SecName);
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

/// \brief Create a string table in `SHeader`, which we assume is already
/// zero'd.
template <class Elf_Shdr>
static void createStringTableSectionHeader(Elf_Shdr &SHeader,
                                           StringTableBuilder &STB,
                                           ContiguousBlobAccumulator &CBA) {
  SHeader.sh_type = ELF::SHT_STRTAB;
  STB.writeToStream(CBA.getOSAndAlignedOffset(SHeader.sh_offset));
  SHeader.sh_size = STB.size();
  SHeader.sh_addralign = 1;
}

namespace {
/// \brief "Single point of truth" for the ELF file construction.
/// TODO: This class still has a ways to go before it is truly a "single
/// point of truth".
template <class ELFT>
class ELFState {
  /// \brief The future ".strtab" section.
  StringTableBuilder DotStrtab;
  /// \brief The section number of the ".strtab" section.
  unsigned DotStrtabSecNo;
  /// \brief The accumulated contents of all sections so far.
  ContiguousBlobAccumulator &SectionContentAccum;
  typedef typename object::ELFFile<ELFT>::Elf_Ehdr Elf_Ehdr;
  /// \brief The ELF file header.
  Elf_Ehdr &Header;

  SectionNameToIdxMap &SN2I;

public:

  ELFState(Elf_Ehdr &Header_, ContiguousBlobAccumulator &Accum,
           unsigned DotStrtabSecNo_, SectionNameToIdxMap &SN2I_)
      : DotStrtab(), DotStrtabSecNo(DotStrtabSecNo_),
        SectionContentAccum(Accum), Header(Header_), SN2I(SN2I_) {}

  unsigned getDotStrTabSecNo() const { return DotStrtabSecNo; }
  StringTableBuilder &getStringTable() { return DotStrtab; }
  ContiguousBlobAccumulator &getSectionContentAccum() {
    return SectionContentAccum;
  }
  SectionNameToIdxMap &getSN2I() { return SN2I; }
};
} // end anonymous namespace

// FIXME: At this point it is fairly clear that we need to refactor these
// static functions into methods of a class sharing some typedefs. These
// ELF type names are insane.
template <class ELFT>
static void
addSymbols(const std::vector<ELFYAML::Symbol> &Symbols, ELFState<ELFT> &State,
           std::vector<typename object::ELFFile<ELFT>::Elf_Sym> &Syms,
           unsigned SymbolBinding) {
  typedef typename object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  for (unsigned i = 0, e = Symbols.size(); i != e; ++i) {
    const ELFYAML::Symbol &Sym = Symbols[i];
    Elf_Sym Symbol;
    zero(Symbol);
    if (!Sym.Name.empty())
      Symbol.st_name = State.getStringTable().addString(Sym.Name);
    Symbol.setBindingAndType(SymbolBinding, Sym.Type);
    if (!Sym.Section.empty()) {
      unsigned Index;
      if (State.getSN2I().lookupSection(Sym.Section, Index)) {
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
static void
handleSymtabSectionHeader(const ELFYAML::LocalGlobalWeakSymbols &Symbols,
                          ELFState<ELFT> &State,
                          typename object::ELFFile<ELFT>::Elf_Shdr &SHeader) {

  typedef typename object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  SHeader.sh_type = ELF::SHT_SYMTAB;
  SHeader.sh_link = State.getDotStrTabSecNo();
  // One greater than symbol table index of the last local symbol.
  SHeader.sh_info = Symbols.Local.size() + 1;
  SHeader.sh_entsize = sizeof(Elf_Sym);

  std::vector<Elf_Sym> Syms;
  {
    // Ensure STN_UNDEF is present
    Elf_Sym Sym;
    zero(Sym);
    Syms.push_back(Sym);
  }
  addSymbols(Symbols.Local, State, Syms, ELF::STB_LOCAL);
  addSymbols(Symbols.Global, State, Syms, ELF::STB_GLOBAL);
  addSymbols(Symbols.Weak, State, Syms, ELF::STB_WEAK);

  ContiguousBlobAccumulator &CBA = State.getSectionContentAccum();
  writeArrayData(CBA.getOSAndAlignedOffset(SHeader.sh_offset),
                 makeArrayRef(Syms));
  SHeader.sh_size = arrayDataSize(makeArrayRef(Syms));
}

template <class ELFT>
static int writeELF(raw_ostream &OS, const ELFYAML::Object &Doc) {
  using namespace llvm::ELF;
  typedef typename object::ELFFile<ELFT>::Elf_Ehdr Elf_Ehdr;
  typedef typename object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;

  const ELFYAML::FileHeader &Hdr = Doc.Header;

  Elf_Ehdr Header;
  zero(Header);
  Header.e_ident[EI_MAG0] = 0x7f;
  Header.e_ident[EI_MAG1] = 'E';
  Header.e_ident[EI_MAG2] = 'L';
  Header.e_ident[EI_MAG3] = 'F';
  Header.e_ident[EI_CLASS] = ELFT::Is64Bits ? ELFCLASS64 : ELFCLASS32;
  bool IsLittleEndian = ELFT::TargetEndianness == support::little;
  Header.e_ident[EI_DATA] = IsLittleEndian ? ELFDATA2LSB : ELFDATA2MSB;
  Header.e_ident[EI_VERSION] = EV_CURRENT;
  Header.e_ident[EI_OSABI] = Hdr.OSABI;
  Header.e_ident[EI_ABIVERSION] = 0;
  Header.e_type = Hdr.Type;
  Header.e_machine = Hdr.Machine;
  Header.e_version = EV_CURRENT;
  Header.e_entry = Hdr.Entry;
  Header.e_ehsize = sizeof(Elf_Ehdr);

  // TODO: Flesh out section header support.
  // TODO: Program headers.

  Header.e_shentsize = sizeof(Elf_Shdr);
  // Immediately following the ELF header.
  Header.e_shoff = sizeof(Header);
  const std::vector<ELFYAML::Section> &Sections = Doc.Sections;
  // "+ 4" for
  // - SHT_NULL entry (placed first, i.e. 0'th entry)
  // - symbol table (.symtab) (placed third to last)
  // - string table (.strtab) (placed second to last)
  // - section header string table. (placed last)
  Header.e_shnum = Sections.size() + 4;
  // Place section header string table last.
  Header.e_shstrndx = Header.e_shnum - 1;
  const unsigned DotStrtabSecNo = Header.e_shnum - 2;

  // XXX: This offset is tightly coupled with the order that we write
  // things to `OS`.
  const size_t SectionContentBeginOffset =
      Header.e_ehsize + Header.e_shentsize * Header.e_shnum;
  ContiguousBlobAccumulator CBA(SectionContentBeginOffset);
  SectionNameToIdxMap SN2I;
  for (unsigned i = 0, e = Sections.size(); i != e; ++i) {
    StringRef Name = Sections[i].Name;
    if (Name.empty())
      continue;
    // "+ 1" to take into account the SHT_NULL entry.
    if (SN2I.addName(Name, i + 1)) {
      errs() << "error: Repeated section name: '" << Name
             << "' at YAML section number " << i << ".\n";
      return 1;
    }
  }

  ELFState<ELFT> State(Header, CBA, DotStrtabSecNo, SN2I);

  StringTableBuilder SHStrTab;
  std::vector<Elf_Shdr> SHeaders;
  {
    // Ensure SHN_UNDEF entry is present. An all-zero section header is a
    // valid SHN_UNDEF entry since SHT_NULL == 0.
    Elf_Shdr SHdr;
    zero(SHdr);
    SHeaders.push_back(SHdr);
  }
  for (unsigned i = 0, e = Sections.size(); i != e; ++i) {
    const ELFYAML::Section &Sec = Sections[i];
    Elf_Shdr SHeader;
    zero(SHeader);
    SHeader.sh_name = SHStrTab.addString(Sec.Name);
    SHeader.sh_type = Sec.Type;
    SHeader.sh_flags = Sec.Flags;
    SHeader.sh_addr = Sec.Address;

    Sec.Content.writeAsBinary(CBA.getOSAndAlignedOffset(SHeader.sh_offset));
    SHeader.sh_size = Sec.Content.binary_size();

    if (!Sec.Link.empty()) {
      unsigned Index;
      if (SN2I.lookupSection(Sec.Link, Index)) {
        errs() << "error: Unknown section referenced: '" << Sec.Link
               << "' at YAML section number " << i << ".\n";
        return 1;
      }
      SHeader.sh_link = Index;
    }
    SHeader.sh_info = 0;
    SHeader.sh_addralign = Sec.AddressAlign;
    SHeader.sh_entsize = 0;
    SHeaders.push_back(SHeader);
  }

  // .symtab section.
  Elf_Shdr SymtabSHeader;
  zero(SymtabSHeader);
  SymtabSHeader.sh_name = SHStrTab.addString(StringRef(".symtab"));
  handleSymtabSectionHeader<ELFT>(Doc.Symbols, State, SymtabSHeader);
  SHeaders.push_back(SymtabSHeader);

  // .strtab string table header.
  Elf_Shdr DotStrTabSHeader;
  zero(DotStrTabSHeader);
  DotStrTabSHeader.sh_name = SHStrTab.addString(StringRef(".strtab"));
  createStringTableSectionHeader(DotStrTabSHeader, State.getStringTable(), CBA);
  SHeaders.push_back(DotStrTabSHeader);

  // Section header string table header.
  Elf_Shdr SHStrTabSHeader;
  zero(SHStrTabSHeader);
  createStringTableSectionHeader(SHStrTabSHeader, SHStrTab, CBA);
  SHeaders.push_back(SHStrTabSHeader);

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
      return writeELF<LE64>(outs(), Doc);
    else
      return writeELF<BE64>(outs(), Doc);
  } else {
    if (isLittleEndian(Doc))
      return writeELF<LE32>(outs(), Doc);
    else
      return writeELF<BE32>(outs(), Doc);
  }
}
