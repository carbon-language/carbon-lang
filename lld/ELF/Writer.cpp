//===- Writer.cpp ---------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Writer.h"
#include "Chunks.h"
#include "Config.h"
#include "Error.h"
#include "Symbols.h"
#include "SymbolTable.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;

using namespace lld;
using namespace lld::elf2;

static const int PageSize = 4096;

// On freebsd x86_64 the first page cannot be mmaped.
// On linux that is controled by vm.mmap_min_addr. At least on some x86_64
// installs that is 65536, so the first 15 pages cannot be used.
// Given that, the smallest value that can be used in here is 0x10000.
// If using 2MB pages, the smallest page aligned address that works is
// 0x200000, but it looks like every OS uses 4k pages for executables.
// FIXME: This is architecture and OS dependent.
static const int VAStart = 0x10000;

namespace {
// OutputSection represents a section in an output file. It's a
// container of chunks. OutputSection and Chunk are 1:N relationship.
// Chunks cannot belong to more than one OutputSections. The writer
// creates multiple OutputSections and assign them unique,
// non-overlapping file offsets and VAs.
template <bool Is64Bits> class OutputSectionBase {
public:
  typedef
      typename std::conditional<Is64Bits, Elf64_Dyn, Elf32_Dyn>::type Elf_Dyn;
  typedef typename std::conditional<Is64Bits, uint64_t, uint32_t>::type uintX_t;
  typedef
      typename std::conditional<Is64Bits, Elf64_Shdr, Elf32_Shdr>::type HeaderT;

  OutputSectionBase(StringRef Name, uint32_t sh_type, uintX_t sh_flags)
      : Name(Name) {
    memset(&Header, 0, sizeof(HeaderT));
    Header.sh_type = sh_type;
    Header.sh_flags = sh_flags;
  }
  void setVA(uintX_t VA) { Header.sh_addr = VA; }
  uintX_t getVA() const { return Header.sh_addr; }
  void setFileOffset(uintX_t Off) { Header.sh_offset = Off; }
  template <endianness E>
  void writeHeaderTo(typename ELFFile<ELFType<E, Is64Bits>>::Elf_Shdr *SHdr);
  StringRef getName() { return Name; }
  void setNameOffset(uintX_t Offset) { Header.sh_name = Offset; }

  unsigned getSectionIndex() const { return SectionIndex; }
  void setSectionIndex(unsigned I) { SectionIndex = I; }

  // Returns the size of the section in the output file.
  uintX_t getSize() { return Header.sh_size; }
  void setSize(uintX_t Val) { Header.sh_size = Val; }
  uintX_t getFlags() { return Header.sh_flags; }
  uintX_t getFileOff() { return Header.sh_offset; }
  uintX_t getAlign() {
    // The ELF spec states that a value of 0 means the section has no alignment
    // constraits.
    return std::max<uintX_t>(Header.sh_addralign, 1);
  }
  uint32_t getType() { return Header.sh_type; }

  static unsigned getAddrSize() { return Is64Bits ? 8 : 4; }

  virtual void finalize() {}
  virtual void writeTo(uint8_t *Buf) = 0;

protected:
  StringRef Name;
  HeaderT Header;
  unsigned SectionIndex;
  ~OutputSectionBase() = default;
};
template <class ELFT> class SymbolTableSection;

template <class ELFT> struct DynamicReloc {
  typedef typename ELFFile<ELFT>::Elf_Rel Elf_Rel;
  const SectionChunk<ELFT> &C;
  const Elf_Rel &RI;
};

static bool relocNeedsGOT(uint32_t Type) {
  switch (Type) {
  default:
    return false;
  case R_X86_64_GOTPCREL:
    return true;
  }
}

template <class ELFT>
class GotSection final : public OutputSectionBase<ELFT::Is64Bits> {
  typedef OutputSectionBase<ELFT::Is64Bits> Base;
  typedef typename Base::uintX_t uintX_t;

public:
  GotSection()
      : OutputSectionBase<ELFT::Is64Bits>(".got", SHT_PROGBITS,
                                          SHF_ALLOC | SHF_WRITE) {
    this->Header.sh_addralign = this->getAddrSize();
  }
  void finalize() override {
    this->Header.sh_size = Entries.size() * this->getAddrSize();
  }
  void writeTo(uint8_t *Buf) override {}
  void addEntry(SymbolBody *Sym) {
    Sym->setGotIndex(Entries.size());
    Entries.push_back(Sym);
  }
  bool empty() const { return Entries.empty(); }
  uintX_t getEntryAddr(const SymbolBody &B) const {
    return this->getVA() + B.getGotIndex() * this->getAddrSize();
  }

private:
  std::vector<const SymbolBody *> Entries;
};

template <class ELFT>
class RelocationSection final : public OutputSectionBase<ELFT::Is64Bits> {
  typedef typename ELFFile<ELFT>::Elf_Rel Elf_Rel;
  typedef typename ELFFile<ELFT>::Elf_Rela Elf_Rela;

public:
  RelocationSection(SymbolTableSection<ELFT> &DynSymSec,
                    const GotSection<ELFT> &GotSec, bool IsRela)
      : OutputSectionBase<ELFT::Is64Bits>(IsRela ? ".rela.dyn" : ".rel.dyn",
                                          IsRela ? SHT_RELA : SHT_REL,
                                          SHF_ALLOC),
        DynSymSec(DynSymSec), GotSec(GotSec), IsRela(IsRela) {
    this->Header.sh_entsize = IsRela ? sizeof(Elf_Rela) : sizeof(Elf_Rel);
    this->Header.sh_addralign = ELFT::Is64Bits ? 8 : 4;
  }

  void addReloc(const DynamicReloc<ELFT> &Reloc) { Relocs.push_back(Reloc); }
  void finalize() override {
    this->Header.sh_link = DynSymSec.getSectionIndex();
    this->Header.sh_size = Relocs.size() * this->Header.sh_entsize;
  }
  void writeTo(uint8_t *Buf) override {
    auto *P = reinterpret_cast<Elf_Rela *>(Buf);
    bool IsMips64EL = Relocs[0].C.getFile()->getObj()->isMips64EL();
    for (const DynamicReloc<ELFT> &Rel : Relocs) {
      const SectionChunk<ELFT> &C = Rel.C;
      const Elf_Rel &RI = Rel.RI;
      OutputSection<ELFT> *Out = C.getOutputSection();
      uint32_t SymIndex = RI.getSymbol(IsMips64EL);
      const SymbolBody *Body = C.getFile()->getSymbolBody(SymIndex);
      uint32_t Type = RI.getType(IsMips64EL);
      if (relocNeedsGOT(Type)) {
        P->r_offset = GotSec.getEntryAddr(*Body);
        P->setSymbolAndType(Body->getDynamicSymbolTableIndex(),
                            R_X86_64_GLOB_DAT, IsMips64EL);
      } else {
        P->r_offset = RI.r_offset + C.getOutputSectionOff() + Out->getVA();
        P->setSymbolAndType(Body->getDynamicSymbolTableIndex(), Type,
                            IsMips64EL);
        if (IsRela)
          P->r_addend = static_cast<const Elf_Rela &>(RI).r_addend;
      }

      ++P;
    }
  }
  bool hasRelocs() const { return !Relocs.empty(); }
  bool isRela() const { return IsRela; }

private:
  std::vector<DynamicReloc<ELFT>> Relocs;
  SymbolTableSection<ELFT> &DynSymSec;
  const GotSection<ELFT> &GotSec;
  const bool IsRela;
};
}

template <class ELFT>
class lld::elf2::OutputSection final
    : public OutputSectionBase<ELFT::Is64Bits> {
public:
  typedef typename OutputSectionBase<ELFT::Is64Bits>::uintX_t uintX_t;
  typedef typename ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename ELFFile<ELFT>::Elf_Rel Elf_Rel;
  typedef typename ELFFile<ELFT>::Elf_Rela Elf_Rela;
  OutputSection(const GotSection<ELFT> &GotSec, StringRef Name,
                uint32_t sh_type, uintX_t sh_flags)
      : OutputSectionBase<ELFT::Is64Bits>(Name, sh_type, sh_flags),
        GotSec(GotSec) {}

  void addChunk(SectionChunk<ELFT> *C);
  void writeTo(uint8_t *Buf) override;

  template <bool isRela>
  void relocate(uint8_t *Buf,
                iterator_range<const Elf_Rel_Impl<ELFT, isRela> *> Rels,
                const ObjectFile<ELFT> &File, uintX_t BaseAddr);

  void relocateOne(uint8_t *Buf, const Elf_Rela &Rel, uint32_t Type,
                   uintX_t BaseAddr, uintX_t SymVA);
  void relocateOne(uint8_t *Buf, const Elf_Rel &Rel, uint32_t Type,
                   uintX_t BaseAddr, uintX_t SymVA);

private:
  std::vector<SectionChunk<ELFT> *> Chunks;
  const GotSection<ELFT> &GotSec;
};

namespace {
template <bool Is64Bits>
class InterpSection final : public OutputSectionBase<Is64Bits> {
public:
  InterpSection()
      : OutputSectionBase<Is64Bits>(".interp", SHT_PROGBITS, SHF_ALLOC) {
    this->Header.sh_size = Config->DynamicLinker.size() + 1;
    this->Header.sh_addralign = 1;
  }

  void writeTo(uint8_t *Buf) override {
    memcpy(Buf, Config->DynamicLinker.data(), Config->DynamicLinker.size());
  }
};

template <bool Is64Bits>
class StringTableSection final : public OutputSectionBase<Is64Bits> {
public:
  typedef typename OutputSectionBase<Is64Bits>::uintX_t uintX_t;
  StringTableSection(bool Dynamic)
      : OutputSectionBase<Is64Bits>(Dynamic ? ".dynstr" : ".strtab", SHT_STRTAB,
                                    Dynamic ? (uintX_t)SHF_ALLOC : 0),
        Dynamic(Dynamic) {
    this->Header.sh_addralign = 1;
  }

  void add(StringRef S) { StrTabBuilder.add(S); }
  size_t getFileOff(StringRef S) const { return StrTabBuilder.getOffset(S); }
  StringRef data() const { return StrTabBuilder.data(); }
  void writeTo(uint8_t *Buf) override;

  void finalize() override {
    StrTabBuilder.finalize(StringTableBuilder::ELF);
    this->Header.sh_size = StrTabBuilder.data().size();
  }

  bool isDynamic() const { return Dynamic; }

private:
  const bool Dynamic;
  llvm::StringTableBuilder StrTabBuilder;
};

template <class ELFT> class Writer;

template <class ELFT>
class SymbolTableSection final : public OutputSectionBase<ELFT::Is64Bits> {
public:
  typedef typename ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename ELFFile<ELFT>::Elf_Sym_Range Elf_Sym_Range;
  typedef typename OutputSectionBase<ELFT::Is64Bits>::uintX_t uintX_t;
  SymbolTableSection(Writer<ELFT> &W, SymbolTable &Table,
                     StringTableSection<ELFT::Is64Bits> &StrTabSec)
      : OutputSectionBase<ELFT::Is64Bits>(
            StrTabSec.isDynamic() ? ".dynsym" : ".symtab",
            StrTabSec.isDynamic() ? SHT_DYNSYM : SHT_SYMTAB,
            StrTabSec.isDynamic() ? (uintX_t)SHF_ALLOC : 0),
        Table(Table), StrTabSec(StrTabSec), W(W) {
    typedef OutputSectionBase<ELFT::Is64Bits> Base;
    typename Base::HeaderT &Header = this->Header;

    Header.sh_entsize = sizeof(Elf_Sym);
    Header.sh_addralign = ELFT::Is64Bits ? 8 : 4;
  }

  void finalize() override {
    this->Header.sh_size = getNumSymbols() * sizeof(Elf_Sym);
    this->Header.sh_link = StrTabSec.getSectionIndex();
    this->Header.sh_info = NumLocals + 1;
  }

  void writeTo(uint8_t *Buf) override;

  const SymbolTable &getSymTable() const { return Table; }

  void addSymbol(StringRef Name, bool isLocal = false) {
    StrTabSec.add(Name);
    ++NumVisible;
    if (isLocal)
      ++NumLocals;
  }

  StringTableSection<ELFT::Is64Bits> &getStrTabSec() { return StrTabSec; }
  unsigned getNumSymbols() const { return NumVisible + 1; }

private:
  SymbolTable &Table;
  StringTableSection<ELFT::Is64Bits> &StrTabSec;
  unsigned NumVisible = 0;
  unsigned NumLocals = 0;
  const Writer<ELFT> &W;
};

template <class ELFT>
class HashTableSection final : public OutputSectionBase<ELFT::Is64Bits> {
  typedef typename ELFFile<ELFT>::Elf_Word Elf_Word;

public:
  HashTableSection(SymbolTableSection<ELFT> &DynSymSec)
      : OutputSectionBase<ELFT::Is64Bits>(".hash", SHT_HASH, SHF_ALLOC),
        DynSymSec(DynSymSec) {
    this->Header.sh_entsize = sizeof(Elf_Word);
    this->Header.sh_addralign = sizeof(Elf_Word);
  }

  void addSymbol(SymbolBody *S) {
    StringRef Name = S->getName();
    DynSymSec.addSymbol(Name);
    Hashes.push_back(hash(Name));
    S->setDynamicSymbolTableIndex(Hashes.size());
  }

  void finalize() override {
    this->Header.sh_link = DynSymSec.getSectionIndex();

    assert(DynSymSec.getNumSymbols() == Hashes.size() + 1);
    unsigned NumEntries = 2;                 // nbucket and nchain.
    NumEntries += DynSymSec.getNumSymbols(); // The chain entries.

    // Create as many buckets as there are symbols.
    // FIXME: This is simplistic. We can try to optimize it, but implementing
    // support for SHT_GNU_HASH is probably even more profitable.
    NumEntries += DynSymSec.getNumSymbols();
    this->Header.sh_size = NumEntries * sizeof(Elf_Word);
  }

  void writeTo(uint8_t *Buf) override {
    unsigned NumSymbols = DynSymSec.getNumSymbols();
    auto *P = reinterpret_cast<Elf_Word *>(Buf);
    *P++ = NumSymbols; // nbucket
    *P++ = NumSymbols; // nchain

    Elf_Word *Buckets = P;
    Elf_Word *Chains = P + NumSymbols;

    for (unsigned I = 1; I < NumSymbols; ++I) {
      uint32_t Hash = Hashes[I - 1] % NumSymbols;
      Chains[I] = Buckets[Hash];
      Buckets[Hash] = I;
    }
  }

  SymbolTableSection<ELFT> &getDynSymSec() { return DynSymSec; }

private:
  uint32_t hash(StringRef Name) {
    uint32_t H = 0;
    for (char C : Name) {
      H = (H << 4) + C;
      uint32_t G = H & 0xf0000000;
      if (G)
        H ^= G >> 24;
      H &= ~G;
    }
    return H;
  }
  SymbolTableSection<ELFT> &DynSymSec;
  std::vector<uint32_t> Hashes;
};

template <class ELFT>
class DynamicSection final : public OutputSectionBase<ELFT::Is64Bits> {
  typedef OutputSectionBase<ELFT::Is64Bits> Base;
  typedef typename Base::HeaderT HeaderT;
  typedef typename Base::Elf_Dyn Elf_Dyn;

public:
  DynamicSection(SymbolTable &SymTab, HashTableSection<ELFT> &HashSec,
                 RelocationSection<ELFT> &RelaDynSec)
      : OutputSectionBase<ELFT::Is64Bits>(".dynamic", SHT_DYNAMIC,
                                          SHF_ALLOC | SHF_WRITE),
        HashSec(HashSec), DynSymSec(HashSec.getDynSymSec()),
        DynStrSec(DynSymSec.getStrTabSec()), RelaDynSec(RelaDynSec),
        SymTab(SymTab) {
    typename Base::HeaderT &Header = this->Header;
    Header.sh_addralign = ELFT::Is64Bits ? 8 : 4;
    Header.sh_entsize = ELFT::Is64Bits ? 16 : 8;
  }

  void finalize() override {
    typename Base::HeaderT &Header = this->Header;
    Header.sh_link = DynStrSec.getSectionIndex();

    unsigned NumEntries = 0;
    if (RelaDynSec.hasRelocs()) {
      ++NumEntries; // DT_RELA / DT_REL
      ++NumEntries; // DT_RELASZ / DTRELSZ
    }
    ++NumEntries; // DT_SYMTAB
    ++NumEntries; // DT_STRTAB
    ++NumEntries; // DT_STRSZ
    ++NumEntries; // DT_HASH

    StringRef RPath = Config->RPath;
    if (!RPath.empty()) {
      ++NumEntries; // DT_RUNPATH
      DynStrSec.add(RPath);
    }

    const std::vector<std::unique_ptr<SharedFileBase>> &SharedFiles =
        SymTab.getSharedFiles();
    for (const std::unique_ptr<SharedFileBase> &File : SharedFiles)
      DynStrSec.add(File->getName());
    NumEntries += SharedFiles.size();

    ++NumEntries; // DT_NULL

    Header.sh_size = NumEntries * Header.sh_entsize;
  }

  void writeTo(uint8_t *Buf) override {
    auto *P = reinterpret_cast<Elf_Dyn *>(Buf);

    if (RelaDynSec.hasRelocs()) {
      bool IsRela = RelaDynSec.isRela();
      P->d_tag = IsRela ? DT_RELA : DT_REL;
      P->d_un.d_ptr = RelaDynSec.getVA();
      ++P;

      P->d_tag = IsRela ? DT_RELASZ : DT_RELSZ;
      P->d_un.d_val = RelaDynSec.getSize();
      ++P;
    }

    P->d_tag = DT_SYMTAB;
    P->d_un.d_ptr = DynSymSec.getVA();
    ++P;

    P->d_tag = DT_STRTAB;
    P->d_un.d_ptr = DynStrSec.getVA();
    ++P;

    P->d_tag = DT_STRSZ;
    P->d_un.d_val = DynStrSec.data().size();
    ++P;

    P->d_tag = DT_HASH;
    P->d_un.d_ptr = HashSec.getVA();
    ++P;

    StringRef RPath = Config->RPath;
    if (!RPath.empty()) {
      P->d_tag = DT_RUNPATH;
      P->d_un.d_val = DynStrSec.getFileOff(RPath);
      ++P;
    }

    const std::vector<std::unique_ptr<SharedFileBase>> &SharedFiles =
        SymTab.getSharedFiles();
    for (const std::unique_ptr<SharedFileBase> &File : SharedFiles) {
      P->d_tag = DT_NEEDED;
      P->d_un.d_val = DynStrSec.getFileOff(File->getName());
      ++P;
    }

    P->d_tag = DT_NULL;
    P->d_un.d_val = 0;
    ++P;
  }

private:
  HashTableSection<ELFT> &HashSec;
  SymbolTableSection<ELFT> &DynSymSec;
  StringTableSection<ELFT::Is64Bits> &DynStrSec;
  RelocationSection<ELFT> &RelaDynSec;
  SymbolTable &SymTab;
};

static uint32_t convertSectionFlagsToPHDRFlags(uint64_t Flags) {
  uint32_t Ret = PF_R;
  if (Flags & SHF_WRITE)
    Ret |= PF_W;

  if (Flags & SHF_EXECINSTR)
    Ret |= PF_X;

  return Ret;
}

template <bool Is64Bits>
class ProgramHeader {
public:
  typedef typename std::conditional<Is64Bits, uint64_t, uint32_t>::type uintX_t;
  typedef
    typename std::conditional<Is64Bits, Elf64_Phdr, Elf32_Phdr>::type HeaderT;

  ProgramHeader(uintX_t p_type, uintX_t p_flags) {
    std::memset(&Header, 0, sizeof(HeaderT));
    Header.p_type = p_type;
    Header.p_flags = p_flags;
    Header.p_align = PageSize;
  }

  void setValuesFromSection(OutputSectionBase<Is64Bits> &Sec) {
    Header.p_flags = convertSectionFlagsToPHDRFlags(Sec.getFlags());
    Header.p_offset = Sec.getFileOff();
    Header.p_vaddr = Sec.getVA();
    Header.p_paddr = Header.p_vaddr;
    Header.p_filesz = Sec.getSize();
    Header.p_memsz = Header.p_filesz;
    Header.p_align = Sec.getAlign();
  }

  template <endianness E>
  void writeHeaderTo(typename ELFFile<ELFType<E, Is64Bits>>::Elf_Phdr *PHDR) {
    PHDR->p_type = Header.p_type;
    PHDR->p_flags = Header.p_flags;
    PHDR->p_offset = Header.p_offset;
    PHDR->p_vaddr = Header.p_vaddr;
    PHDR->p_paddr = Header.p_paddr;
    PHDR->p_filesz = Header.p_filesz;
    PHDR->p_memsz = Header.p_memsz;
    PHDR->p_align = Header.p_align;
  }

  HeaderT Header;
  bool Closed = false;
};

// The writer writes a SymbolTable result to a file.
template <class ELFT> class Writer {
public:
  typedef typename ELFFile<ELFT>::uintX_t uintX_t;
  typedef typename ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename ELFFile<ELFT>::Elf_Ehdr Elf_Ehdr;
  typedef typename ELFFile<ELFT>::Elf_Phdr Elf_Phdr;
  typedef typename ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename ELFFile<ELFT>::Elf_Sym_Range Elf_Sym_Range;
  typedef typename ELFFile<ELFT>::Elf_Rela Elf_Rela;
  Writer(SymbolTable *T)
      : SymTabSec(*this, *T, StrTabSec), DynSymSec(*this, *T, DynStrSec),
        RelaDynSec(DynSymSec, GotSec, T->shouldUseRela()), HashSec(DynSymSec),
        DynamicSec(*T, HashSec, RelaDynSec) {}
  void run();

  const OutputSection<ELFT> &getBSS() const {
    assert(BSSSec);
    return *BSSSec;
  }

private:
  void createSections();
  template <bool isRela>
  void scanRelocs(const SectionChunk<ELFT> &C,
                  iterator_range<const Elf_Rel_Impl<ELFT, isRela> *> Rels);
  void scanRelocs(const SectionChunk<ELFT> &C);
  void assignAddresses();
  void openFile(StringRef OutputPath);
  void writeHeader();
  void writeSections();
  bool needsInterpSection() const {
    return !SymTabSec.getSymTable().getSharedFiles().empty() &&
           !Config->DynamicLinker.empty();
  }
  bool needsDynamicSections() const {
    return !SymTabSec.getSymTable().getSharedFiles().empty() || Config->Shared;
  }
  unsigned getVAStart() const { return Config->Shared ? 0 : VAStart; }

  std::unique_ptr<llvm::FileOutputBuffer> Buffer;

  llvm::SpecificBumpPtrAllocator<OutputSection<ELFT>> CAlloc;
  std::vector<OutputSectionBase<ELFT::Is64Bits> *> OutputSections;
  unsigned getNumSections() const { return OutputSections.size() + 1; }

  llvm::BumpPtrAllocator PAlloc;
  std::vector<ProgramHeader<ELFT::Is64Bits> *> PHDRs;
  ProgramHeader<ELFT::Is64Bits> FileHeaderPHDR{PT_LOAD, PF_R};
  ProgramHeader<ELFT::Is64Bits> InterpPHDR{PT_INTERP, 0};
  ProgramHeader<ELFT::Is64Bits> DynamicPHDR{PT_DYNAMIC, 0};

  uintX_t FileSize;
  uintX_t ProgramHeaderOff;
  uintX_t SectionHeaderOff;

  StringTableSection<ELFT::Is64Bits> StrTabSec = { /*dynamic=*/false };
  StringTableSection<ELFT::Is64Bits> DynStrSec = { /*dynamic=*/true };

  SymbolTableSection<ELFT> SymTabSec;
  SymbolTableSection<ELFT> DynSymSec;

  RelocationSection<ELFT> RelaDynSec;

  GotSection<ELFT> GotSec;

  HashTableSection<ELFT> HashSec;

  DynamicSection<ELFT> DynamicSec;

  InterpSection<ELFT::Is64Bits> InterpSec;

  OutputSection<ELFT> *BSSSec = nullptr;
};
} // anonymous namespace

namespace lld {
namespace elf2 {

template <class ELFT>
void writeResult(SymbolTable *Symtab) { Writer<ELFT>(Symtab).run(); }

template void writeResult<ELF32LE>(SymbolTable *);
template void writeResult<ELF32BE>(SymbolTable *);
template void writeResult<ELF64LE>(SymbolTable *);
template void writeResult<ELF64BE>(SymbolTable *);

} // namespace elf2
} // namespace lld

// The main function of the writer.
template <class ELFT> void Writer<ELFT>::run() {
  createSections();
  assignAddresses();
  openFile(Config->OutputFile);
  writeHeader();
  writeSections();
  error(Buffer->commit());
}

template <class ELFT>
void OutputSection<ELFT>::addChunk(SectionChunk<ELFT> *C) {
  Chunks.push_back(C);
  C->setOutputSection(this);
  uint32_t Align = C->getAlign();
  if (Align > this->Header.sh_addralign)
    this->Header.sh_addralign = Align;

  uintX_t Off = this->Header.sh_size;
  Off = RoundUpToAlignment(Off, Align);
  C->setOutputSectionOff(Off);
  Off += C->getSize();
  this->Header.sh_size = Off;
}

template <class ELFT>
static typename ELFFile<ELFT>::uintX_t
getSymVA(const DefinedRegular<ELFT> *DR) {
  const SectionChunk<ELFT> *SC = &DR->Section;
  OutputSection<ELFT> *OS = SC->getOutputSection();
  return OS->getVA() + SC->getOutputSectionOff() + DR->Sym.st_value;
}

template <class ELFT>
void OutputSection<ELFT>::relocateOne(uint8_t *Buf, const Elf_Rel &Rel,
                                      uint32_t Type, uintX_t BaseAddr,
                                      uintX_t SymVA) {
  uintX_t Offset = Rel.r_offset;
  uint8_t *Location = Buf + Offset;
  switch (Type) {
  case R_386_32:
    support::endian::write32le(Location, SymVA);
    break;
  default:
    llvm::errs() << Twine("unrecognized reloc ") + Twine(Type) << '\n';
    break;
  }
}

template <class ELFT>
void OutputSection<ELFT>::relocateOne(uint8_t *Buf, const Elf_Rela &Rel,
                                      uint32_t Type, uintX_t BaseAddr,
                                      uintX_t SymVA) {
  uintX_t Offset = Rel.r_offset;
  uint8_t *Location = Buf + Offset;
  switch (Type) {
  case R_X86_64_PC32:
    support::endian::write32le(Location,
                               SymVA + (Rel.r_addend - (BaseAddr + Offset)));
    break;
  case R_X86_64_64:
    support::endian::write64le(Location, SymVA + Rel.r_addend);
    break;
  case R_X86_64_32: {
  case R_X86_64_32S:
    uint64_t VA = SymVA + Rel.r_addend;
    if (Type == R_X86_64_32 && !isUInt<32>(VA))
      error("R_X86_64_32 out of range");
    else if (!isInt<32>(VA))
      error("R_X86_64_32S out of range");

    support::endian::write32le(Location, VA);
    break;
  }
  default:
    llvm::errs() << Twine("unrecognized reloc ") + Twine(Type) << '\n';
    break;
  }
}

template <class ELFT>
template <bool isRela>
void OutputSection<ELFT>::relocate(
    uint8_t *Buf, iterator_range<const Elf_Rel_Impl<ELFT, isRela> *> Rels,
    const ObjectFile<ELFT> &File, uintX_t BaseAddr) {
  typedef Elf_Rel_Impl<ELFT, isRela> RelType;
  bool IsMips64EL = File.getObj()->isMips64EL();
  for (const RelType &RI : Rels) {
    uint32_t SymIndex = RI.getSymbol(IsMips64EL);
    const SymbolBody *Body = File.getSymbolBody(SymIndex);
    if (!Body)
      continue;

    uint32_t Type = RI.getType(IsMips64EL);
    uintX_t SymVA;

    switch (Body->kind()) {
    case SymbolBody::DefinedRegularKind:
      SymVA = getSymVA<ELFT>(cast<DefinedRegular<ELFT>>(Body));
      break;
    case SymbolBody::DefinedAbsoluteKind:
      SymVA = cast<DefinedAbsolute<ELFT>>(Body)->Sym.st_value;
      break;
    case SymbolBody::DefinedCommonKind: {
      auto *DC = cast<DefinedCommon<ELFT>>(Body);
      SymVA = DC->OutputSec->getVA() + DC->OffsetInBSS;
      break;
    }
    case SymbolBody::SharedKind:
      if (!relocNeedsGOT(Type))
        continue;
      SymVA = GotSec.getEntryAddr(*Body);
      Type = R_X86_64_PC32;
      break;
    case SymbolBody::UndefinedKind:
      assert(Body->isWeak() && "Undefined symbol reached writer");
      SymVA = 0;
      break;
    case SymbolBody::LazyKind:
      llvm_unreachable("Lazy symbol reached writer");
    }

    relocateOne(Buf, RI, Type, BaseAddr, SymVA);
  }
}

template <class ELFT> void OutputSection<ELFT>::writeTo(uint8_t *Buf) {
  for (SectionChunk<ELFT> *C : Chunks) {
    C->writeTo(Buf);
    const ObjectFile<ELFT> *File = C->getFile();
    ELFFile<ELFT> *EObj = File->getObj();
    uint8_t *Base = Buf + C->getOutputSectionOff();
    uintX_t BaseAddr = this->getVA() + C->getOutputSectionOff();
    // Iterate over all relocation sections that apply to this section.
    for (const Elf_Shdr *RelSec : C->RelocSections) {
      if (RelSec->sh_type == SHT_RELA)
        relocate(Base, EObj->relas(RelSec), *File, BaseAddr);
      else
        relocate(Base, EObj->rels(RelSec), *File, BaseAddr);
    }
  }
}

template <bool Is64Bits>
void StringTableSection<Is64Bits>::writeTo(uint8_t *Buf) {
  StringRef Data = StrTabBuilder.data();
  memcpy(Buf, Data.data(), Data.size());
}

template <class ELFT>
static int compareSym(const typename ELFFile<ELFT>::Elf_Sym *A,
                      const typename ELFFile<ELFT>::Elf_Sym *B) {
  uint32_t AN = A->st_name;
  uint32_t BN = B->st_name;
  assert(AN != BN);
  return AN - BN;
}

static bool includeInSymtab(const SymbolBody &B) {
  if (B.isLazy())
    return false;
  if (!B.isUsedInRegularObj())
    return false;
  uint8_t V = B.getMostConstrainingVisibility();
  if (V != STV_DEFAULT && V != STV_PROTECTED)
    return false;
  return true;
}

template <class ELFT> void SymbolTableSection<ELFT>::writeTo(uint8_t *Buf) {
  const OutputSection<ELFT> *Out = nullptr;
  const SectionChunk<ELFT> *Section = nullptr;
  Buf += sizeof(Elf_Sym);

  // All symbols with STB_LOCAL binding precede the weak and global symbols.
  // .dynsym only contains global symbols.
  if (!Config->DiscardAll && !StrTabSec.isDynamic()) {
    for (const std::unique_ptr<ObjectFileBase> &FileB :
         Table.getObjectFiles()) {
      auto &File = cast<ObjectFile<ELFT>>(*FileB);
      Elf_Sym_Range Syms = File.getLocalSymbols();
      for (const Elf_Sym &Sym : Syms) {
        auto *ESym = reinterpret_cast<Elf_Sym *>(Buf);
        uint32_t SecIndex = Sym.st_shndx;
        ErrorOr<StringRef> SymName = Sym.getName(File.getStringTable());
        if (Config->DiscardLocals && SymName->startswith(".L"))
          continue;
        ESym->st_name = (SymName) ? StrTabSec.getFileOff(*SymName) : 0;
        ESym->st_size = Sym.st_size;
        ESym->setBindingAndType(Sym.getBinding(), Sym.getType());
        if (SecIndex == SHN_XINDEX)
          SecIndex = File.getObj()->getExtendedSymbolTableIndex(
              &Sym, File.getSymbolTable(), File.getSymbolTableShndx());
        ArrayRef<SectionChunk<ELFT> *> Chunks = File.getChunks();
        Section = Chunks[SecIndex];
        assert(Section != nullptr);
        Out = Section->getOutputSection();
        assert(Out != nullptr);
        ESym->st_shndx = Out->getSectionIndex();
        ESym->st_value =
            Out->getVA() + Section->getOutputSectionOff() + Sym.st_value;
        Buf += sizeof(Elf_Sym);
      }
    }
  }

  for (auto &P : Table.getSymbols()) {
    StringRef Name = P.first;
    Symbol *Sym = P.second;
    SymbolBody *Body = Sym->Body;
    if (!includeInSymtab(*Body))
      continue;
    const Elf_Sym &InputSym = cast<ELFSymbolBody<ELFT>>(Body)->Sym;

    auto *ESym = reinterpret_cast<Elf_Sym *>(Buf);
    ESym->st_name = StrTabSec.getFileOff(Name);

    Out = nullptr;
    Section = nullptr;

    switch (Body->kind()) {
    case SymbolBody::DefinedRegularKind:
      Section = &cast<DefinedRegular<ELFT>>(Body)->Section;
      break;
    case SymbolBody::DefinedCommonKind:
      Out = &W.getBSS();
      break;
    case SymbolBody::UndefinedKind:
    case SymbolBody::DefinedAbsoluteKind:
    case SymbolBody::SharedKind:
      break;
    case SymbolBody::LazyKind:
      llvm_unreachable("Lazy symbol got to output symbol table!");
    }

    ESym->setBindingAndType(InputSym.getBinding(), InputSym.getType());
    ESym->st_size = InputSym.st_size;
    ESym->setVisibility(Body->getMostConstrainingVisibility());
    if (InputSym.isAbsolute()) {
      ESym->st_shndx = SHN_ABS;
      ESym->st_value = InputSym.st_value;
    }

    if (Section)
      Out = Section->getOutputSection();

    if (Out) {
      ESym->st_shndx = Out->getSectionIndex();
      uintX_t VA = Out->getVA();
      if (Section)
        VA += Section->getOutputSectionOff();
      if (auto *C = dyn_cast<DefinedCommon<ELFT>>(Body))
        VA += C->OffsetInBSS;
      else
        VA += InputSym.st_value;
      ESym->st_value = VA;
    }

    Buf += sizeof(Elf_Sym);
  }
}

template <bool Is64Bits>
template <endianness E>
void OutputSectionBase<Is64Bits>::writeHeaderTo(
    typename ELFFile<ELFType<E, Is64Bits>>::Elf_Shdr *SHdr) {
  SHdr->sh_name = Header.sh_name;
  SHdr->sh_type = Header.sh_type;
  SHdr->sh_flags = Header.sh_flags;
  SHdr->sh_addr = Header.sh_addr;
  SHdr->sh_offset = Header.sh_offset;
  SHdr->sh_size = Header.sh_size;
  SHdr->sh_link = Header.sh_link;
  SHdr->sh_info = Header.sh_info;
  SHdr->sh_addralign = Header.sh_addralign;
  SHdr->sh_entsize = Header.sh_entsize;
}

namespace {
template <bool Is64Bits> struct SectionKey {
  typedef typename std::conditional<Is64Bits, uint64_t, uint32_t>::type uintX_t;
  StringRef Name;
  uint32_t sh_type;
  uintX_t sh_flags;
};
}
namespace llvm {
template <bool Is64Bits> struct DenseMapInfo<SectionKey<Is64Bits>> {
  static SectionKey<Is64Bits> getEmptyKey() {
    return SectionKey<Is64Bits>{DenseMapInfo<StringRef>::getEmptyKey(), 0, 0};
  }
  static SectionKey<Is64Bits> getTombstoneKey() {
    return SectionKey<Is64Bits>{DenseMapInfo<StringRef>::getTombstoneKey(), 0,
                                0};
  }
  static unsigned getHashValue(const SectionKey<Is64Bits> &Val) {
    return hash_combine(Val.Name, Val.sh_type, Val.sh_flags);
  }
  static bool isEqual(const SectionKey<Is64Bits> &LHS,
                      const SectionKey<Is64Bits> &RHS) {
    return DenseMapInfo<StringRef>::isEqual(LHS.Name, RHS.Name) &&
           LHS.sh_type == RHS.sh_type && LHS.sh_flags == RHS.sh_flags;
  }
};
}

template <class ELFT>
static bool cmpAlign(const DefinedCommon<ELFT> *A,
                     const DefinedCommon<ELFT> *B) {
  return A->MaxAlignment > B->MaxAlignment;
}

template <bool Is64Bits>
static bool compSec(OutputSectionBase<Is64Bits> *A,
                    OutputSectionBase<Is64Bits> *B) {
  // Place SHF_ALLOC sections first.
  return (A->getFlags() & SHF_ALLOC) && !(B->getFlags() & SHF_ALLOC);
}

// The reason we have to do this early scan is as follows
// * To mmap the output file, we need to know the size
// * For that, we need to know how many dynamic relocs we will have.
// It might be possible to avoid this by outputting the file with write:
// * Write the allocated output sections, computing addresses.
// * Apply relocations, recording which ones require a dynamic reloc.
// * Write the dynamic relocations.
// * Write the rest of the file.
template <class ELFT>
template <bool isRela>
void Writer<ELFT>::scanRelocs(
    const SectionChunk<ELFT> &C,
    iterator_range<const Elf_Rel_Impl<ELFT, isRela> *> Rels) {
  typedef Elf_Rel_Impl<ELFT, isRela> RelType;
  const ObjectFile<ELFT> &File = *C.getFile();
  bool IsMips64EL = File.getObj()->isMips64EL();
  for (const RelType &RI : Rels) {
    uint32_t SymIndex = RI.getSymbol(IsMips64EL);
    SymbolBody *Body = File.getSymbolBody(SymIndex);
    if (!Body)
      continue;
    auto *S = dyn_cast<SharedSymbol<ELFT>>(Body);
    if (!S)
      continue;
    if (relocNeedsGOT(RI.getType(IsMips64EL))) {
      if (Body->isInGot())
        continue;
      GotSec.addEntry(Body);
    }
    RelaDynSec.addReloc({C, RI});
  }
}

template <class ELFT>
void Writer<ELFT>::scanRelocs(const SectionChunk<ELFT> &C) {
  const ObjectFile<ELFT> *File = C.getFile();
  ELFFile<ELFT> *EObj = File->getObj();

  if (!(C.getSectionHdr()->sh_flags & SHF_ALLOC))
    return;

  for (const Elf_Shdr *RelSec : C.RelocSections) {
    if (RelSec->sh_type == SHT_RELA)
      scanRelocs(C, EObj->relas(RelSec));
    else
      scanRelocs(C, EObj->rels(RelSec));
  }
}

// Create output section objects and add them to OutputSections.
template <class ELFT> void Writer<ELFT>::createSections() {
  SmallDenseMap<SectionKey<ELFT::Is64Bits>, OutputSection<ELFT> *> Map;
  auto getSection = [&](StringRef Name, uint32_t sh_type,
                        uintX_t sh_flags) -> OutputSection<ELFT> * {
    SectionKey<ELFT::Is64Bits> Key{Name, sh_type, sh_flags};
    OutputSection<ELFT> *&Sec = Map[Key];
    if (!Sec) {
      Sec = new (CAlloc.Allocate())
          OutputSection<ELFT>(GotSec, Key.Name, Key.sh_type, Key.sh_flags);
      OutputSections.push_back(Sec);
    }
    return Sec;
  };

  // FIXME: Try to avoid the extra walk over all global symbols.
  const SymbolTable &Symtab = SymTabSec.getSymTable();
  std::vector<DefinedCommon<ELFT> *> CommonSymbols;
  for (auto &P : Symtab.getSymbols()) {
    StringRef Name = P.first;
    SymbolBody *Body = P.second->Body;
    if (Body->isStrongUndefined())
      error(Twine("undefined symbol: ") + Name);

    if (auto *C = dyn_cast<DefinedCommon<ELFT>>(Body))
      CommonSymbols.push_back(C);
    if (!includeInSymtab(*Body))
      continue;
    SymTabSec.addSymbol(Name);

    // FIXME: This adds way too much to the dynamic symbol table. We only
    // need to add the symbols use by dynamic relocations when producing
    // an executable (ignoring --export-dynamic).
    if (needsDynamicSections())
      HashSec.addSymbol(Body);
  }

  for (const std::unique_ptr<ObjectFileBase> &FileB : Symtab.getObjectFiles()) {
    auto &File = cast<ObjectFile<ELFT>>(*FileB);
    if (!Config->DiscardAll) {
      Elf_Sym_Range Syms = File.getLocalSymbols();
      for (const Elf_Sym &Sym : Syms) {
        ErrorOr<StringRef> SymName = Sym.getName(File.getStringTable());
        if (SymName && !(Config->DiscardLocals && SymName->startswith(".L")))
          SymTabSec.addSymbol(*SymName, true);
      }
    }
    for (SectionChunk<ELFT> *C : File.getChunks()) {
      if (!C)
        continue;
      const Elf_Shdr *H = C->getSectionHdr();
      OutputSection<ELFT> *Sec =
          getSection(C->getSectionName(), H->sh_type, H->sh_flags);
      Sec->addChunk(C);
      scanRelocs(*C);
    }
  }

  BSSSec = getSection(".bss", SHT_NOBITS, SHF_ALLOC | SHF_WRITE);
  // Sort the common symbols by alignment as an heuristic to pack them better.
  std::stable_sort(CommonSymbols.begin(), CommonSymbols.end(), cmpAlign<ELFT>);
  uintX_t Off = BSSSec->getSize();
  for (DefinedCommon<ELFT> *C : CommonSymbols) {
    const Elf_Sym &Sym = C->Sym;
    uintX_t Align = C->MaxAlignment;
    Off = RoundUpToAlignment(Off, Align);
    C->OffsetInBSS = Off;
    C->OutputSec = BSSSec;
    Off += Sym.st_size;
  }

  BSSSec->setSize(Off);

  OutputSections.push_back(&SymTabSec);
  OutputSections.push_back(&StrTabSec);

  if (needsDynamicSections()) {
    if (needsInterpSection())
      OutputSections.push_back(&InterpSec);
    OutputSections.push_back(&DynSymSec);
    OutputSections.push_back(&HashSec);
    OutputSections.push_back(&DynamicSec);
    OutputSections.push_back(&DynStrSec);
    if (RelaDynSec.hasRelocs())
      OutputSections.push_back(&RelaDynSec);
    if (!GotSec.empty())
      OutputSections.push_back(&GotSec);
  }

  std::stable_sort(OutputSections.begin(), OutputSections.end(),
                   compSec<ELFT::Is64Bits>);
  for (unsigned I = 0, N = OutputSections.size(); I < N; ++I)
    OutputSections[I]->setSectionIndex(I + 1);
}

template <class ELFT>
static bool outputSectionHasPHDR(OutputSectionBase<ELFT::Is64Bits> *Sec) {
  return Sec->getFlags() & SHF_ALLOC;
}

// Visits all sections to assign incremental, non-overlapping RVAs and
// file offsets.
template <class ELFT> void Writer<ELFT>::assignAddresses() {
  assert(!OutputSections.empty() && "No output sections to layout!");
  uintX_t VA = getVAStart();
  uintX_t FileOff = 0;

  FileOff += sizeof(Elf_Ehdr);
  VA += sizeof(Elf_Ehdr);

  // Reserve space for PHDRs.
  ProgramHeaderOff = FileOff;
  FileOff = RoundUpToAlignment(FileOff, PageSize);
  VA = RoundUpToAlignment(VA, PageSize);

  if (needsInterpSection())
    PHDRs.push_back(&InterpPHDR);

  ProgramHeader<ELFT::Is64Bits> *LastPHDR = &FileHeaderPHDR;
  // Create a PHDR for the file header.
  PHDRs.push_back(&FileHeaderPHDR);
  FileHeaderPHDR.Header.p_vaddr = getVAStart();
  FileHeaderPHDR.Header.p_paddr = getVAStart();
  FileHeaderPHDR.Header.p_align = PageSize;

  for (OutputSectionBase<ELFT::Is64Bits> *Sec : OutputSections) {
    StrTabSec.add(Sec->getName());
    Sec->finalize();

    if (Sec->getSize()) {
      uintX_t Flags = convertSectionFlagsToPHDRFlags(Sec->getFlags());
      if (LastPHDR->Header.p_flags != Flags ||
          !outputSectionHasPHDR<ELFT>(Sec)) {
        // Flags changed. End current PHDR and potentially create a new one.
        if (!LastPHDR->Closed) {
          LastPHDR->Header.p_filesz = FileOff - LastPHDR->Header.p_offset;
          LastPHDR->Header.p_memsz = VA - LastPHDR->Header.p_vaddr;
          LastPHDR->Closed = true;
        }

        if (outputSectionHasPHDR<ELFT>(Sec)) {
          LastPHDR = new (PAlloc) ProgramHeader<ELFT::Is64Bits>(PT_LOAD, Flags);
          PHDRs.push_back(LastPHDR);
          VA = RoundUpToAlignment(VA, PageSize);
          FileOff = RoundUpToAlignment(FileOff, PageSize);
          LastPHDR->Header.p_offset = FileOff;
          LastPHDR->Header.p_vaddr = VA;
          LastPHDR->Header.p_paddr = VA;
        }
      }
    }

    uintX_t Align = Sec->getAlign();
    uintX_t Size = Sec->getSize();
    if (Sec->getFlags() & SHF_ALLOC) {
      VA = RoundUpToAlignment(VA, Align);
      Sec->setVA(VA);
      VA += Size;
    }
    FileOff = RoundUpToAlignment(FileOff, Align);
    Sec->setFileOffset(FileOff);
    if (Sec->getType() != SHT_NOBITS)
      FileOff += Size;
  }

  // Add a PHDR for the dynamic table.
  if (needsDynamicSections())
    PHDRs.push_back(&DynamicPHDR);

  FileOff += OffsetToAlignment(FileOff, ELFT::Is64Bits ? 8 : 4);

  // Add space for section headers.
  SectionHeaderOff = FileOff;
  FileOff += getNumSections() * sizeof(Elf_Shdr);
  FileSize = FileOff;
}

template <class ELFT> void Writer<ELFT>::writeHeader() {
  uint8_t *Buf = Buffer->getBufferStart();
  auto *EHdr = reinterpret_cast<Elf_Ehdr *>(Buf);
  EHdr->e_ident[EI_MAG0] = 0x7F;
  EHdr->e_ident[EI_MAG1] = 0x45;
  EHdr->e_ident[EI_MAG2] = 0x4C;
  EHdr->e_ident[EI_MAG3] = 0x46;
  EHdr->e_ident[EI_CLASS] = ELFT::Is64Bits ? ELFCLASS64 : ELFCLASS32;
  EHdr->e_ident[EI_DATA] = ELFT::TargetEndianness == llvm::support::little
                               ? ELFDATA2LSB
                               : ELFDATA2MSB;
  EHdr->e_ident[EI_VERSION] = EV_CURRENT;
  EHdr->e_ident[EI_OSABI] = ELFOSABI_NONE;

  // FIXME: Generalize the segment construction similar to how we create
  // output sections.
  const SymbolTable &Symtab = SymTabSec.getSymTable();

  EHdr->e_type = Config->Shared ? ET_DYN : ET_EXEC;
  auto &FirstObj = cast<ObjectFile<ELFT>>(*Symtab.getFirstELF());
  EHdr->e_machine = FirstObj.getEMachine();
  EHdr->e_version = EV_CURRENT;
  SymbolBody *Entry = Symtab.getEntrySym();
  EHdr->e_entry = Entry ? getSymVA(cast<DefinedRegular<ELFT>>(Entry)) : 0;
  EHdr->e_phoff = ProgramHeaderOff;
  EHdr->e_shoff = SectionHeaderOff;
  EHdr->e_ehsize = sizeof(Elf_Ehdr);
  EHdr->e_phentsize = sizeof(Elf_Phdr);
  EHdr->e_phnum = PHDRs.size();
  EHdr->e_shentsize = sizeof(Elf_Shdr);
  EHdr->e_shnum = getNumSections();
  EHdr->e_shstrndx = StrTabSec.getSectionIndex();

  // If nothing was merged into the file header PT_LOAD, set the size correctly.
  if (FileHeaderPHDR.Header.p_filesz == PageSize)
    FileHeaderPHDR.Header.p_filesz = FileHeaderPHDR.Header.p_memsz =
        sizeof(Elf_Ehdr) + sizeof(Elf_Phdr) * PHDRs.size();

  if (needsInterpSection())
    InterpPHDR.setValuesFromSection(InterpSec);
  if (needsDynamicSections())
    DynamicPHDR.setValuesFromSection(DynamicSec);

  auto PHdrs = reinterpret_cast<Elf_Phdr *>(Buf + EHdr->e_phoff);
  for (ProgramHeader<ELFT::Is64Bits> *PHDR : PHDRs)
    PHDR->template writeHeaderTo<ELFT::TargetEndianness>(PHdrs++);

  auto SHdrs = reinterpret_cast<Elf_Shdr *>(Buf + EHdr->e_shoff);
  // First entry is null.
  ++SHdrs;
  for (OutputSectionBase<ELFT::Is64Bits> *Sec : OutputSections) {
    Sec->setNameOffset(StrTabSec.getFileOff(Sec->getName()));
    Sec->template writeHeaderTo<ELFT::TargetEndianness>(SHdrs++);
  }
}

template <class ELFT> void Writer<ELFT>::openFile(StringRef Path) {
  ErrorOr<std::unique_ptr<FileOutputBuffer>> BufferOrErr =
      FileOutputBuffer::create(Path, FileSize, FileOutputBuffer::F_executable);
  error(BufferOrErr, Twine("failed to open ") + Path);
  Buffer = std::move(*BufferOrErr);
}

// Write section contents to a mmap'ed file.
template <class ELFT> void Writer<ELFT>::writeSections() {
  uint8_t *Buf = Buffer->getBufferStart();
  for (OutputSectionBase<ELFT::Is64Bits> *Sec : OutputSections)
    Sec->writeTo(Buf + Sec->getFileOff());
}
