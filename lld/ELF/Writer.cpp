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

  virtual void finalize() {}
  virtual void writeTo(uint8_t *Buf) = 0;

protected:
  StringRef Name;
  HeaderT Header;
  unsigned SectionIndex;
  ~OutputSectionBase() = default;
};
}

template <class ELFT>
class lld::elf2::OutputSection final
    : public OutputSectionBase<ELFT::Is64Bits> {
public:
  typedef typename OutputSectionBase<ELFT::Is64Bits>::uintX_t uintX_t;
  typedef typename ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename ELFFile<ELFT>::Elf_Rela Elf_Rela;
  OutputSection(StringRef Name, uint32_t sh_type, uintX_t sh_flags)
      : OutputSectionBase<ELFT::Is64Bits>(Name, sh_type, sh_flags) {}

  void addChunk(SectionChunk<ELFT> *C);
  void writeTo(uint8_t *Buf) override;

private:
  std::vector<SectionChunk<ELFT> *> Chunks;
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
  typedef typename ELFFile<ELFT>::Elf_Sym Elf_Sym;
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

    // For now the only local symbol is going to be the one at index 0
    Header.sh_info = 1;

    Header.sh_entsize = sizeof(Elf_Sym);
    Header.sh_addralign = ELFT::Is64Bits ? 8 : 4;
  }

  void finalize() override {
    this->Header.sh_size = (NumVisible + 1) * sizeof(Elf_Sym);
    this->Header.sh_link = StrTabSec.getSectionIndex();
  }

  void writeTo(uint8_t *Buf) override;

  const SymbolTable &getSymTable() const { return Table; }

  void addSymbol(StringRef Name) {
    StrTabSec.add(Name);
    ++NumVisible;
  }

  StringTableSection<ELFT::Is64Bits> &getStrTabSec() { return StrTabSec; }

private:
  SymbolTable &Table;
  StringTableSection<ELFT::Is64Bits> &StrTabSec;
  unsigned NumVisible = 0;
  const Writer<ELFT> &W;
};

template <class ELFT>
class DynamicSection final : public OutputSectionBase<ELFT::Is64Bits> {
  typedef OutputSectionBase<ELFT::Is64Bits> Base;
  typedef typename Base::HeaderT HeaderT;
  typedef typename Base::Elf_Dyn Elf_Dyn;

public:
  DynamicSection(SymbolTable &SymTab, SymbolTableSection<ELFT> &DynSymSec)
      : OutputSectionBase<ELFT::Is64Bits>(".dynamic", SHT_DYNAMIC,
                                          SHF_ALLOC | SHF_WRITE),
        DynStrSec(DynSymSec.getStrTabSec()), DynSymSec(DynSymSec),
        SymTab(SymTab) {
    typename Base::HeaderT &Header = this->Header;
    Header.sh_addralign = ELFT::Is64Bits ? 8 : 4;
    Header.sh_entsize = ELFT::Is64Bits ? 16 : 8;

    unsigned NumEntries = 0;

    ++NumEntries; // DT_SYMTAB
    ++NumEntries; // DT_STRTAB
    ++NumEntries; // DT_STRSZ

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

  void finalize() override {
    this->Header.sh_link = DynStrSec.getSectionIndex();
  }

  void writeTo(uint8_t *Buf) override {
    auto *P = reinterpret_cast<Elf_Dyn *>(Buf);

    P->d_tag = DT_SYMTAB;
    P->d_un.d_ptr = DynSymSec.getVA();
    ++P;

    P->d_tag = DT_STRTAB;
    P->d_un.d_ptr = DynStrSec.getVA();
    ++P;

    P->d_tag = DT_STRSZ;
    P->d_un.d_val = DynStrSec.data().size();
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
  StringTableSection<ELFT::Is64Bits> &DynStrSec;
  SymbolTableSection<ELFT> &DynSymSec;
  SymbolTable &SymTab;
};

// The writer writes a SymbolTable result to a file.
template <class ELFT> class Writer {
public:
  typedef typename ELFFile<ELFT>::uintX_t uintX_t;
  typedef typename ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename ELFFile<ELFT>::Elf_Ehdr Elf_Ehdr;
  typedef typename ELFFile<ELFT>::Elf_Phdr Elf_Phdr;
  typedef typename ELFFile<ELFT>::Elf_Sym Elf_Sym;
  Writer(SymbolTable *T)
      : StrTabSec(false), DynStrSec(true), SymTabSec(*this, *T, StrTabSec),
        DynSymSec(*this, *T, DynStrSec), DynamicSec(*T, DynSymSec) {}
  void run();

  const OutputSection<ELFT> &getBSS() const {
    assert(BSSSec);
    return *BSSSec;
  }

private:
  void createSections();
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

  uintX_t FileSize;
  uintX_t ProgramHeaderOff;
  uintX_t SectionHeaderOff;
  unsigned NumPhdrs;

  StringTableSection<ELFT::Is64Bits> StrTabSec;
  StringTableSection<ELFT::Is64Bits> DynStrSec;

  SymbolTableSection<ELFT> SymTabSec;
  SymbolTableSection<ELFT> DynSymSec;

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
static typename ELFFile<ELFT>::uintX_t getSymVA(DefinedRegular<ELFT> *DR) {
  const SectionChunk<ELFT> *SC = &DR->Section;
  OutputSection<ELFT> *OS = SC->getOutputSection();
  return OS->getVA() + SC->getOutputSectionOff() + DR->Sym.st_value;
}

template <class ELFT> void OutputSection<ELFT>::writeTo(uint8_t *Buf) {
  for (SectionChunk<ELFT> *C : Chunks) {
    C->writeTo(Buf);
    ObjectFile<ELFT> *File = C->getFile();
    ELFFile<ELFT> *EObj = File->getObj();
    uint8_t *Base = Buf + C->getOutputSectionOff();

    // Iterate over all relocation sections that apply to this section.
    for (const Elf_Shdr *RelSec : C->RelocSections) {
      // Only support RELA for now.
      if (RelSec->sh_type != SHT_RELA)
        continue;
      for (const Elf_Rela &RI : EObj->relas(RelSec)) {
        uint32_t SymIndex = RI.getSymbol(EObj->isMips64EL());
        SymbolBody *Body = File->getSymbolBody(SymIndex);
        if (!Body)
          continue;
        // Skip unsupported for now.
        if (!isa<DefinedRegular<ELFT>>(Body))
          continue;
        uintX_t Offset = RI.r_offset;
        uint32_t Type = RI.getType(EObj->isMips64EL());
        uintX_t P = this->getVA() + C->getOutputSectionOff();
        uintX_t SymVA = getSymVA<ELFT>(cast<DefinedRegular<ELFT>>(Body));
        uint8_t *Location = Base + Offset;
        switch (Type) {
        case llvm::ELF::R_X86_64_PC32:
          support::endian::write32le(Location,
                                     SymVA + (RI.r_addend - (P + Offset)));
          break;
        case llvm::ELF::R_X86_64_32:
          support::endian::write32le(Location, SymVA + RI.r_addend);
          break;
        default:
          llvm::errs() << Twine("unrecognized reloc ") + Twine(Type) << '\n';
          break;
        }
      }
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
  uint8_t *BufStart = Buf;

  Buf += sizeof(Elf_Sym);
  for (auto &P : Table.getSymbols()) {
    StringRef Name = P.first;
    Symbol *Sym = P.second;
    SymbolBody *Body = Sym->Body;
    if (!includeInSymtab(*Body))
      continue;
    const Elf_Sym &InputSym = cast<ELFSymbolBody<ELFT>>(Body)->Sym;

    auto *ESym = reinterpret_cast<Elf_Sym *>(Buf);
    ESym->st_name = StrTabSec.getFileOff(Name);

    const SectionChunk<ELFT> *Section = nullptr;
    const OutputSection<ELFT> *Out = nullptr;

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

  // The order the global symbols are in is not defined. We can use an arbitrary
  // order, but it has to be reproducible. That is true even when cross linking.
  // The default hashing of StringRef produces different results on 32 and 64
  // bit systems so we sort by st_name. That is arbitrary but deterministic.
  // FIXME: Experiment with passing in a custom hashing instead.
  auto *Syms = reinterpret_cast<Elf_Sym *>(BufStart);
  ++Syms;
  array_pod_sort(Syms, Syms + NumVisible, compareSym<ELFT>);
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

// Create output section objects and add them to OutputSections.
template <class ELFT> void Writer<ELFT>::createSections() {
  SmallDenseMap<SectionKey<ELFT::Is64Bits>, OutputSection<ELFT> *> Map;
  auto getSection = [&](StringRef Name, uint32_t sh_type,
                        uintX_t sh_flags) -> OutputSection<ELFT> * {
    SectionKey<ELFT::Is64Bits> Key{Name, sh_type, sh_flags};
    OutputSection<ELFT> *&Sec = Map[Key];
    if (!Sec) {
      Sec = new (CAlloc.Allocate())
          OutputSection<ELFT>(Key.Name, Key.sh_type, Key.sh_flags);
      OutputSections.push_back(Sec);
    }
    return Sec;
  };

  const SymbolTable &Symtab = SymTabSec.getSymTable();
  for (const std::unique_ptr<ObjectFileBase> &FileB : Symtab.getObjectFiles()) {
    auto &File = cast<ObjectFile<ELFT>>(*FileB);
    for (SectionChunk<ELFT> *C : File.getChunks()) {
      if (!C)
        continue;
      const Elf_Shdr *H = C->getSectionHdr();
      OutputSection<ELFT> *Sec =
          getSection(C->getSectionName(), H->sh_type, H->sh_flags);
      Sec->addChunk(C);
    }
  }

  BSSSec = getSection(".bss", SHT_NOBITS, SHF_ALLOC | SHF_WRITE);
  // FIXME: Try to avoid the extra walk over all global symbols.
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
      DynSymSec.addSymbol(Name);
  }

  // Sort the common symbols by alignment as an heuristic to pack them better.
  std::stable_sort(CommonSymbols.begin(), CommonSymbols.end(), cmpAlign<ELFT>);
  uintX_t Off = BSSSec->getSize();
  for (DefinedCommon<ELFT> *C : CommonSymbols) {
    const Elf_Sym &Sym = C->Sym;
    uintX_t Align = C->MaxAlignment;
    Off = RoundUpToAlignment(Off, Align);
    C->OffsetInBSS = Off;
    Off += Sym.st_size;
  }

  BSSSec->setSize(Off);

  OutputSections.push_back(&SymTabSec);
  OutputSections.push_back(&StrTabSec);

  if (needsDynamicSections()) {
    if (needsInterpSection())
      OutputSections.push_back(&InterpSec);
    OutputSections.push_back(&DynSymSec);
    OutputSections.push_back(&DynamicSec);
    OutputSections.push_back(&DynStrSec);
  }

  std::stable_sort(OutputSections.begin(), OutputSections.end(),
                   compSec<ELFT::Is64Bits>);
  for (unsigned I = 0, N = OutputSections.size(); I < N; ++I)
    OutputSections[I]->setSectionIndex(I + 1);
}

template <class ELFT>
static bool outputSectionHasPHDR(OutputSectionBase<ELFT::Is64Bits> *Sec) {
  return (Sec->getSize() != 0) && (Sec->getFlags() & SHF_ALLOC);
}

// Visits all sections to assign incremental, non-overlapping RVAs and
// file offsets.
template <class ELFT> void Writer<ELFT>::assignAddresses() {
  uintX_t VA = getVAStart();
  uintX_t FileOff = 0;

  FileOff += sizeof(Elf_Ehdr);
  VA += sizeof(Elf_Ehdr);

  // Reserve space for PHDRs.
  ProgramHeaderOff = FileOff;
  FileOff = RoundUpToAlignment(FileOff, PageSize);
  VA = RoundUpToAlignment(VA, PageSize);

  NumPhdrs = 0;

  // Add a PHDR for PT_INTERP.
  if (needsInterpSection())
    ++NumPhdrs;

  // Add a PHDR for the elf header and program headers. Some dynamic linkers
  // (musl at least) require them to be covered by a PT_LOAD.
  ++NumPhdrs;

  for (OutputSectionBase<ELFT::Is64Bits> *Sec : OutputSections) {
    StrTabSec.add(Sec->getName());
    Sec->finalize();

    // Since each output section gets its own PHDR, align each output section to
    // a page.
    if (outputSectionHasPHDR<ELFT>(Sec)) {
      ++NumPhdrs;
      VA = RoundUpToAlignment(VA, PageSize);
      FileOff = RoundUpToAlignment(FileOff, PageSize);
    }

    uintX_t Align = Sec->getAlign();
    uintX_t Size = Sec->getSize();
    if (Sec->getFlags() & SHF_ALLOC) {
      Sec->setVA(VA);
      VA += RoundUpToAlignment(Size, Align);
    }
    Sec->setFileOffset(FileOff);
    if (Sec->getType() != SHT_NOBITS)
      FileOff += RoundUpToAlignment(Size, Align);
  }

  // Add a PHDR for the dynamic table.
  if (needsDynamicSections())
    ++NumPhdrs;

  FileOff += OffsetToAlignment(FileOff, ELFT::Is64Bits ? 8 : 4);

  // Add space for section headers.
  SectionHeaderOff = FileOff;
  FileOff += getNumSections() * sizeof(Elf_Shdr);
  FileSize = FileOff;
}

static uint32_t convertSectionFlagsToPHDRFlags(uint64_t Flags) {
  uint32_t Ret = PF_R;
  if (Flags & SHF_WRITE)
    Ret |= PF_W;

  if (Flags & SHF_EXECINSTR)
    Ret |= PF_X;

  return Ret;
}

template <class ELFT>
static void setValuesFromSection(typename ELFFile<ELFT>::Elf_Phdr &P,
                                 OutputSectionBase<ELFT::Is64Bits> &S) {
  P.p_flags = convertSectionFlagsToPHDRFlags(S.getFlags());
  P.p_offset = S.getFileOff();
  P.p_vaddr = S.getVA();
  P.p_paddr = P.p_vaddr;
  P.p_filesz = S.getSize();
  P.p_memsz = P.p_filesz;
  P.p_align = S.getAlign();
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
  EHdr->e_phnum = NumPhdrs;
  EHdr->e_shentsize = sizeof(Elf_Shdr);
  EHdr->e_shnum = getNumSections();
  EHdr->e_shstrndx = StrTabSec.getSectionIndex();

  auto PHdrs = reinterpret_cast<Elf_Phdr *>(Buf + EHdr->e_phoff);
  if (needsInterpSection()) {
    PHdrs->p_type = PT_INTERP;
    setValuesFromSection<ELFT>(*PHdrs, InterpSec);
    ++PHdrs;
  }

  PHdrs->p_type = PT_LOAD;
  PHdrs->p_flags = PF_R;
  PHdrs->p_offset = 0;
  PHdrs->p_vaddr = getVAStart();
  PHdrs->p_paddr = PHdrs->p_vaddr;
  PHdrs->p_filesz = ProgramHeaderOff + NumPhdrs * sizeof(Elf_Phdr);
  PHdrs->p_memsz = PHdrs->p_filesz;
  PHdrs->p_align = PageSize;
  ++PHdrs;

  for (OutputSectionBase<ELFT::Is64Bits> *Sec : OutputSections) {
    if (!outputSectionHasPHDR<ELFT>(Sec))
      continue;
    PHdrs->p_type = PT_LOAD;
    PHdrs->p_flags = convertSectionFlagsToPHDRFlags(Sec->getFlags());
    PHdrs->p_offset = Sec->getFileOff();
    PHdrs->p_vaddr = Sec->getVA();
    PHdrs->p_paddr = PHdrs->p_vaddr;
    PHdrs->p_filesz = Sec->getType() == SHT_NOBITS ? 0 : Sec->getSize();
    PHdrs->p_memsz = Sec->getSize();
    PHdrs->p_align = PageSize;
    ++PHdrs;
  }

  if (needsDynamicSections()) {
    PHdrs->p_type = PT_DYNAMIC;
    setValuesFromSection<ELFT>(*PHdrs, DynamicSec);
  }

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
