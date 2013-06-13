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

// This class is used to build up a contiguous binary blob while keeping
// track of an offset in the output (which notionally begins at
// `InitialOffset`).
class ContiguousBlobAccumulator {
  const uint64_t InitialOffset;
  raw_svector_ostream OS;

public:
  ContiguousBlobAccumulator(uint64_t InitialOffset_, SmallVectorImpl<char> &Buf)
      : InitialOffset(InitialOffset_), OS(Buf) {}
  raw_ostream &getOS() { return OS; }
  uint64_t currentOffset() const { return InitialOffset + OS.tell(); }
  void writeBlobToStream(raw_ostream &Out) { Out << OS.str(); }
};

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

template <class ELFT>
static void writeELF(raw_ostream &OS, const ELFYAML::Object &Doc) {
  using namespace llvm::ELF;
  using namespace llvm::object;
  typedef typename ELFObjectFile<ELFT>::Elf_Ehdr Elf_Ehdr;
  typedef typename ELFObjectFile<ELFT>::Elf_Shdr Elf_Shdr;

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
  // TODO: Implement ELF_ELFOSABI enum.
  Header.e_ident[EI_OSABI] = ELFOSABI_NONE;
  // TODO: Implement ELF_ABIVERSION enum.
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
  std::vector<ELFYAML::Section> Sections = Doc.Sections;
  if (Sections.empty() || Sections.front().Type != SHT_NULL) {
    ELFYAML::Section S;
    S.Type = SHT_NULL;
    zero(S.Flags);
    Sections.insert(Sections.begin(), S);
  }
  // "+ 1" for string table.
  Header.e_shnum = Sections.size() + 1;
  // Place section header string table last.
  Header.e_shstrndx = Sections.size();

  StringTableBuilder StrTab;
  SmallVector<char, 128> Buf;
  // XXX: This offset is tightly coupled with the order that we write
  // things to `OS`.
  const size_t SectionContentBeginOffset =
      Header.e_ehsize + Header.e_shentsize * Header.e_shnum;
  ContiguousBlobAccumulator CBA(SectionContentBeginOffset, Buf);
  std::vector<Elf_Shdr> SHeaders;
  for (unsigned i = 0, e = Sections.size(); i != e; ++i) {
    const ELFYAML::Section &Sec = Sections[i];
    Elf_Shdr SHeader;
    zero(SHeader);
    SHeader.sh_name = StrTab.addString(Sec.Name);
    SHeader.sh_type = Sec.Type;
    SHeader.sh_flags = Sec.Flags;
    SHeader.sh_addr = Sec.Address;

    SHeader.sh_offset = CBA.currentOffset();
    SHeader.sh_size = Sec.Content.binary_size();
    Sec.Content.writeAsBinary(CBA.getOS());

    SHeader.sh_link = 0;
    SHeader.sh_info = 0;
    SHeader.sh_addralign = 1;
    SHeader.sh_entsize = 0;
    SHeaders.push_back(SHeader);
  }

  // String table header.
  Elf_Shdr StrTabSHeader;
  zero(StrTabSHeader);
  StrTabSHeader.sh_name = 0;
  StrTabSHeader.sh_type = SHT_STRTAB;
  StrTabSHeader.sh_flags = 0;
  StrTabSHeader.sh_addr = 0;
  StrTabSHeader.sh_offset = CBA.currentOffset();
  StrTabSHeader.sh_size = StrTab.size();
  StrTab.writeToStream(CBA.getOS());
  StrTabSHeader.sh_link = 0;
  StrTabSHeader.sh_info = 0;
  StrTabSHeader.sh_addralign = 1;
  StrTabSHeader.sh_entsize = 0;

  OS.write((const char *)&Header, sizeof(Header));
  writeVectorData(OS, SHeaders);
  OS.write((const char *)&StrTabSHeader, sizeof(StrTabSHeader));
  CBA.writeBlobToStream(OS);
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
      writeELF<object::ELFType<support::little, 8, true> >(outs(), Doc);
    else
      writeELF<object::ELFType<support::big, 8, true> >(outs(), Doc);
  } else {
    if (Doc.Header.Data == ELFYAML::ELF_ELFDATA(ELF::ELFDATA2LSB))
      writeELF<object::ELFType<support::little, 4, false> >(outs(), Doc);
    else
      writeELF<object::ELFType<support::big, 4, false> >(outs(), Doc);
  }

  return 0;
}
