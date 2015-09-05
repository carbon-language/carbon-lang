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

namespace {
// OutputSection represents a section in an output file. It's a
// container of chunks. OutputSection and Chunk are 1:N relationship.
// Chunks cannot belong to more than one OutputSections. The writer
// creates multiple OutputSections and assign them unique,
// non-overlapping file offsets and VAs.
template <bool Is64Bits> class OutputSectionBase {
public:
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
  uintX_t getVA() { return Header.sh_addr; }
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
class StringTableSection final : public OutputSectionBase<Is64Bits> {
public:
  llvm::StringTableBuilder StrTabBuilder;

  typedef typename OutputSectionBase<Is64Bits>::uintX_t uintX_t;
  StringTableSection() : OutputSectionBase<Is64Bits>(".strtab", SHT_STRTAB, 0) {
    this->Header.sh_addralign = 1;
  }

  void add(StringRef S) { StrTabBuilder.add(S); }
  size_t getFileOff(StringRef S) { return StrTabBuilder.getOffset(S); }
  void writeTo(uint8_t *Buf) override;

  void finalize() override {
    StrTabBuilder.finalize(StringTableBuilder::ELF);
    this->Header.sh_size = StrTabBuilder.data().size();
  }
};

template <class ELFT>
class SymbolTableSection final : public OutputSectionBase<ELFT::Is64Bits> {
public:
  typedef typename ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename OutputSectionBase<ELFT::Is64Bits>::uintX_t uintX_t;
  SymbolTableSection(SymbolTable &Table, llvm::StringTableBuilder &Builder)
      : OutputSectionBase<ELFT::Is64Bits>(".symtab", SHT_SYMTAB, 0),
        Table(Table), Builder(Builder) {
    typedef OutputSectionBase<ELFT::Is64Bits> Base;
    typename Base::HeaderT &Header = this->Header;

    // For now the only local symbol is going to be the one at index 0
    Header.sh_info = 1;

    Header.sh_entsize = sizeof(Elf_Sym);
    Header.sh_addralign = ELFT::Is64Bits ? 8 : 4;
  }

  void finalize() override {
    this->Header.sh_size = (NumVisible + 1) * sizeof(Elf_Sym);
  }

  void setStringTableIndex(uint32_t Index) { this->Header.sh_link = Index; }

  void writeTo(uint8_t *Buf) override;

  const SymbolTable &getSymTable() { return Table; }

  OutputSection<ELFT> *BSSSec = nullptr;
  unsigned NumVisible = 0;

private:
  SymbolTable &Table;
  llvm::StringTableBuilder &Builder;
};

// The writer writes a SymbolTable result to a file.
template <class ELFT> class Writer {
public:
  typedef typename llvm::object::ELFFile<ELFT>::uintX_t uintX_t;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename llvm::object::ELFFile<ELFT>::Elf_Sym Elf_Sym;
  Writer(SymbolTable *T) : SymTable(*T, StringTable.StrTabBuilder) {}
  void run();

private:
  void createSections();
  void assignAddresses();
  void openFile(StringRef OutputPath);
  void writeHeader();
  void writeSections();

  std::unique_ptr<llvm::FileOutputBuffer> Buffer;
  llvm::SpecificBumpPtrAllocator<OutputSection<ELFT>> CAlloc;
  std::vector<OutputSectionBase<ELFT::Is64Bits> *> OutputSections;

  uintX_t FileSize;
  uintX_t SizeOfHeaders;
  uintX_t SectionHeaderOff;

  unsigned StringTableIndex;
  StringTableSection<ELFT::Is64Bits> StringTable;

  SymbolTableSection<ELFT> SymTable;

  unsigned NumSections;

  void addOutputSection(OutputSectionBase<ELFT::Is64Bits> *Sec) {
    OutputSections.push_back(Sec);
    Sec->setSectionIndex(OutputSections.size());
  }
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
static typename llvm::object::ELFFile<ELFT>::uintX_t
getSymVA(DefinedRegular<ELFT> *DR) {
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
        case llvm::ELF::R_X86_64_32S:
          support::endian::write32le(Location, SymVA + RI.r_addend);
          break;
        case llvm::ELF::R_X86_64_64:
          support::endian::write64le(Location, SymVA + RI.r_addend);
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

template <class ELFT> void SymbolTableSection<ELFT>::writeTo(uint8_t *Buf) {
  uint8_t *BufStart = Buf;

  Buf += sizeof(Elf_Sym);
  for (auto &P : Table.getSymbols()) {
    StringRef Name = P.first;
    Symbol *Sym = P.second;
    SymbolBody *Body = Sym->Body;
    if (Body->isLazy())
      continue;
    const Elf_Sym &InputSym = cast<ELFSymbolBody<ELFT>>(Body)->Sym;

    uint8_t V = Body->getMostConstrainingVisibility();
    if (V != STV_DEFAULT && V != STV_PROTECTED)
      continue;

    auto *ESym = reinterpret_cast<Elf_Sym *>(Buf);
    ESym->st_name = Builder.getOffset(Name);

    const SectionChunk<ELFT> *Section = nullptr;
    OutputSection<ELFT> *Out = nullptr;

    switch (Body->kind()) {
    case SymbolBody::DefinedRegularKind:
      Section = &cast<DefinedRegular<ELFT>>(Body)->Section;
      break;
    case SymbolBody::DefinedCommonKind:
      Out = BSSSec;
      break;
    case SymbolBody::UndefinedKind:
      assert(Body->isWeak() && "Should be defined by now");
    case SymbolBody::DefinedAbsoluteKind:
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
      addOutputSection(Sec);
    }
    return Sec;
  };

  const SymbolTable &Symtab = SymTable.getSymTable();
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

  SymTable.BSSSec = getSection(".bss", SHT_NOBITS, SHF_ALLOC | SHF_WRITE);
  OutputSection<ELFT> *BSSSec = SymTable.BSSSec;
  // FIXME: Try to avoid the extra walk over all global symbols.
  unsigned &NumVisible = SymTable.NumVisible;
  llvm::StringTableBuilder &Builder = StringTable.StrTabBuilder;
  std::vector<DefinedCommon<ELFT> *> CommonSymbols;
  for (auto &P : Symtab.getSymbols()) {
    StringRef Name = P.first;
    SymbolBody *Body = P.second->Body;
    if (Body->isLazy())
      continue;
    if (auto *C = dyn_cast<DefinedCommon<ELFT>>(Body))
      CommonSymbols.push_back(C);
    uint8_t V = Body->getMostConstrainingVisibility();
    if (V != STV_DEFAULT && V != STV_PROTECTED)
      continue;
    NumVisible++;
    Builder.add(Name);
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
}

template <bool Is64Bits>
static bool compSec(OutputSectionBase<Is64Bits> *A,
                    OutputSectionBase<Is64Bits> *B) {
  // Place SHF_ALLOC sections first.
  return (A->getFlags() & SHF_ALLOC) && !(B->getFlags() & SHF_ALLOC);
}

// Visits all sections to assign incremental, non-overlapping RVAs and
// file offsets.
template <class ELFT> void Writer<ELFT>::assignAddresses() {
  SizeOfHeaders = RoundUpToAlignment(sizeof(Elf_Ehdr_Impl<ELFT>), PageSize);
  uintX_t VA = 0x1000; // The first page is kept unmapped.
  uintX_t FileOff = SizeOfHeaders;

  std::stable_sort(OutputSections.begin(), OutputSections.end(),
                   compSec<ELFT::Is64Bits>);

  addOutputSection(&SymTable);
  addOutputSection(&StringTable);
  StringTableIndex = OutputSections.size();
  SymTable.setStringTableIndex(StringTableIndex);

  for (OutputSectionBase<ELFT::Is64Bits> *Sec : OutputSections) {
    StringTable.add(Sec->getName());
    Sec->finalize();

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

  // Regular sections.
  NumSections = OutputSections.size();

  // First dummy section.
  NumSections++;

  FileOff += OffsetToAlignment(FileOff, ELFT::Is64Bits ? 8 : 4);

  // Add space for section headers.
  SectionHeaderOff = FileOff;
  FileOff += NumSections * sizeof(Elf_Shdr_Impl<ELFT>);
  FileSize = SizeOfHeaders + RoundUpToAlignment(FileOff - SizeOfHeaders, 8);
}

template <class ELFT> void Writer<ELFT>::writeHeader() {
  uint8_t *Buf = Buffer->getBufferStart();
  auto *EHdr = reinterpret_cast<Elf_Ehdr_Impl<ELFT> *>(Buf);
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

  EHdr->e_type = ET_EXEC;
  const SymbolTable &Symtab = SymTable.getSymTable();
  auto &FirstObj = cast<ObjectFile<ELFT>>(*Symtab.getFirstELF());
  EHdr->e_machine = FirstObj.getEMachine();
  EHdr->e_version = EV_CURRENT;
  EHdr->e_entry = 0x401000;
  EHdr->e_phoff = sizeof(Elf_Ehdr_Impl<ELFT>);
  EHdr->e_shoff = SectionHeaderOff;
  EHdr->e_ehsize = sizeof(Elf_Ehdr_Impl<ELFT>);
  EHdr->e_phentsize = sizeof(Elf_Phdr_Impl<ELFT>);
  EHdr->e_phnum = 1;
  EHdr->e_shentsize = sizeof(Elf_Shdr_Impl<ELFT>);
  EHdr->e_shnum = NumSections;
  EHdr->e_shstrndx = StringTableIndex;

  auto PHdrs = reinterpret_cast<Elf_Phdr_Impl<ELFT> *>(Buf + EHdr->e_phoff);
  PHdrs->p_type = PT_LOAD;
  PHdrs->p_flags = PF_R | PF_X;
  PHdrs->p_offset = 0x0000;
  PHdrs->p_vaddr = 0x400000;
  PHdrs->p_paddr = PHdrs->p_vaddr;
  PHdrs->p_filesz = FileSize;
  PHdrs->p_memsz = FileSize;
  PHdrs->p_align = 0x4000;

  auto SHdrs = reinterpret_cast<Elf_Shdr_Impl<ELFT> *>(Buf + EHdr->e_shoff);
  // First entry is null.
  ++SHdrs;
  for (OutputSectionBase<ELFT::Is64Bits> *Sec : OutputSections) {
    Sec->setNameOffset(StringTable.getFileOff(Sec->getName()));
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
