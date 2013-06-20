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
#include "llvm/Object/ELF.h"
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

public:
  ContiguousBlobAccumulator(uint64_t InitialOffset_)
      : InitialOffset(InitialOffset_), Buf(), OS(Buf) {}
  raw_ostream &getOS() { return OS; }
  uint64_t currentOffset() const { return InitialOffset + OS.tell(); }
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
static size_t vectorDataSize(const std::vector<T> &Vec) {
  return Vec.size() * sizeof(T);
}

template <class T>
static void writeVectorData(raw_ostream &OS, const std::vector<T> &Vec) {
  OS.write((const char *)Vec.data(), vectorDataSize(Vec));
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
  SHeader.sh_offset = CBA.currentOffset();
  SHeader.sh_size = STB.size();
  STB.writeToStream(CBA.getOS());
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
  typedef typename object::ELFObjectFile<ELFT>::Elf_Ehdr Elf_Ehdr;
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

// FIXME: This function is hideous. The hideous ELF type names are hideous.
// Factor the ELF output into a class (templated on ELFT) and share some
// typedefs.
template <class ELFT>
static void handleSymtabSectionHeader(
    const ELFYAML::Section &Sec, ELFState<ELFT> &State,
    typename object::ELFObjectFile<ELFT>::Elf_Shdr &SHeader) {

  typedef typename object::ELFObjectFile<ELFT>::Elf_Sym Elf_Sym;
  // TODO: Ensure that a manually specified `Link` field is diagnosed as an
  // error for SHT_SYMTAB.
  SHeader.sh_link = State.getDotStrTabSecNo();
  // TODO: Once we handle symbol binding, this should be one greater than
  // symbol table index of the last local symbol.
  SHeader.sh_info = 0;
  SHeader.sh_entsize = sizeof(Elf_Sym);

  std::vector<Elf_Sym> Syms;
  {
    // Ensure STN_UNDEF is present
    Elf_Sym Sym;
    zero(Sym);
    Syms.push_back(Sym);
  }
  for (unsigned i = 0, e = Sec.Symbols.size(); i != e; ++i) {
    const ELFYAML::Symbol &Sym = Sec.Symbols[i];
    Elf_Sym Symbol;
    zero(Symbol);
    if (!Sym.Name.empty())
      Symbol.st_name = State.getStringTable().addString(Sym.Name);
    Symbol.setBindingAndType(Sym.Binding, Sym.Type);
    unsigned Index;
    if (State.getSN2I().lookupSection(Sym.Section, Index)) {
      errs() << "error: Unknown section referenced: '" << Sym.Section
             << "' by YAML symbol " << Sym.Name << ".\n";
      exit(1);
    }
    Symbol.st_shndx = Index;
    Symbol.st_value = Sym.Value;
    Symbol.st_size = Sym.Size;
    Syms.push_back(Symbol);
  }

  ContiguousBlobAccumulator &CBA = State.getSectionContentAccum();
  SHeader.sh_offset = CBA.currentOffset();
  SHeader.sh_size = vectorDataSize(Syms);
  writeVectorData(CBA.getOS(), Syms);
}

template <class ELFT>
static int writeELF(raw_ostream &OS, const ELFYAML::Object &Doc) {
  using namespace llvm::ELF;
  typedef typename object::ELFObjectFile<ELFT>::Elf_Ehdr Elf_Ehdr;
  typedef typename object::ELFObjectFile<ELFT>::Elf_Shdr Elf_Shdr;

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
  // "+ 3" for
  // - SHT_NULL entry (placed first, i.e. 0'th entry)
  // - string table (.strtab) (placed second to last)
  // - section header string table. (placed last)
  Header.e_shnum = Sections.size() + 3;
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

    SHeader.sh_offset = CBA.currentOffset();
    SHeader.sh_size = Sec.Content.binary_size();
    Sec.Content.writeAsBinary(CBA.getOS());

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
    // XXX: Really ugly right now. Should not be writing to `CBA` above
    // (and setting sh_offset and sh_size) when going through this branch
    // here.
    if (Sec.Type == ELFYAML::ELF_SHT(SHT_SYMTAB))
      handleSymtabSectionHeader<ELFT>(Sec, State, SHeader);
    SHeaders.push_back(SHeader);
  }

  // .strtab string table header.
  Elf_Shdr DotStrTabSHeader;
  zero(DotStrTabSHeader);
  DotStrTabSHeader.sh_name = SHStrTab.addString(StringRef(".strtab"));
  createStringTableSectionHeader(DotStrTabSHeader, State.getStringTable(), CBA);

  // Section header string table header.
  Elf_Shdr SHStrTabSHeader;
  zero(SHStrTabSHeader);
  createStringTableSectionHeader(SHStrTabSHeader, SHStrTab, CBA);

  OS.write((const char *)&Header, sizeof(Header));
  writeVectorData(OS, SHeaders);
  OS.write((const char *)&DotStrTabSHeader, sizeof(DotStrTabSHeader));
  OS.write((const char *)&SHStrTabSHeader, sizeof(SHStrTabSHeader));
  CBA.writeBlobToStream(OS);
  return 0;
}

int yaml2elf(llvm::raw_ostream &Out, llvm::MemoryBuffer *Buf) {
  yaml::Input YIn(Buf->getBuffer());
  ELFYAML::Object Doc;
  YIn >> Doc;
  if (YIn.error()) {
    errs() << "yaml2obj: Failed to parse YAML file!\n";
    return 1;
  }
  if (Doc.Header.Class == ELFYAML::ELF_ELFCLASS(ELF::ELFCLASS64)) {
    if (Doc.Header.Data == ELFYAML::ELF_ELFDATA(ELF::ELFDATA2LSB))
      return writeELF<object::ELFType<support::little, 8, true> >(outs(), Doc);
    else
      return writeELF<object::ELFType<support::big, 8, true> >(outs(), Doc);
  } else {
    if (Doc.Header.Data == ELFYAML::ELF_ELFDATA(ELF::ELFDATA2LSB))
      return writeELF<object::ELFType<support::little, 4, false> >(outs(), Doc);
    else
      return writeELF<object::ELFType<support::big, 4, false> >(outs(), Doc);
  }
}
