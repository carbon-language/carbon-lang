//===- Writer.cpp ---------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Writer.h"
#include "Config.h"
#include "OutputSections.h"
#include "SymbolTable.h"
#include "Target.h"

#include "llvm/Support/FileOutputBuffer.h"

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
      : SymTabSec(*T, StrTabSec, BssSec), DynSymSec(*T, DynStrSec, BssSec),
        RelaDynSec(DynSymSec, GotSec, T->shouldUseRela()), PltSec(GotSec),
        HashSec(DynSymSec), DynamicSec(*T, HashSec, RelaDynSec),
        BssSec(PltSec, GotSec, BssSec, ".bss", SHT_NOBITS,
               SHF_ALLOC | SHF_WRITE) {}
  void run();

private:
  void createSections();
  template <bool isRela>
  void scanRelocs(const InputSection<ELFT> &C,
                  iterator_range<const Elf_Rel_Impl<ELFT, isRela> *> Rels);
  void scanRelocs(const InputSection<ELFT> &C);
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

  lld::elf2::SymbolTableSection<ELFT> SymTabSec;
  lld::elf2::SymbolTableSection<ELFT> DynSymSec;

  RelocationSection<ELFT> RelaDynSec;

  GotSection<ELFT> GotSec;
  PltSection<ELFT> PltSec;

  HashTableSection<ELFT> HashSec;

  DynamicSection<ELFT> DynamicSec;

  InterpSection<ELFT::Is64Bits> InterpSec;

  OutputSection<ELFT> BssSec;
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
    const InputSection<ELFT> &C,
    iterator_range<const Elf_Rel_Impl<ELFT, isRela> *> Rels) {
  typedef Elf_Rel_Impl<ELFT, isRela> RelType;
  const ObjectFile<ELFT> &File = *C.getFile();
  bool IsMips64EL = File.getObj().isMips64EL();
  for (const RelType &RI : Rels) {
    uint32_t SymIndex = RI.getSymbol(IsMips64EL);
    SymbolBody *Body = File.getSymbolBody(SymIndex);
    if (!Body)
      continue;
    uint32_t Type = RI.getType(IsMips64EL);
    if (Target->relocNeedsPlt(Type, *Body)) {
      if (Body->isInPlt())
        continue;
      PltSec.addEntry(Body);
    }
    if (Target->relocNeedsGot(Type, *Body)) {
      if (Body->isInGot())
        continue;
      GotSec.addEntry(Body);
    } else if (!isa<SharedSymbol<ELFT>>(Body))
      continue;
    Body->setUsedInDynamicReloc();
    RelaDynSec.addReloc({C, RI});
  }
}

template <class ELFT>
void Writer<ELFT>::scanRelocs(const InputSection<ELFT> &C) {
  ObjectFile<ELFT> *File = C.getFile();
  ELFFile<ELFT> &EObj = File->getObj();

  if (!(C.getSectionHdr()->sh_flags & SHF_ALLOC))
    return;

  for (const Elf_Shdr *RelSec : C.RelocSections) {
    if (RelSec->sh_type == SHT_RELA)
      scanRelocs(C, EObj.relas(RelSec));
    else
      scanRelocs(C, EObj.rels(RelSec));
  }
}

template <class ELFT>
static void undefError(const SymbolTable &S, const SymbolBody &Sym) {
  typedef typename ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename ELFFile<ELFT>::Elf_Sym_Range Elf_Sym_Range;

  const Elf_Sym &SymE = cast<ELFSymbolBody<ELFT>>(Sym).Sym;
  ELFFileBase *SymFile = nullptr;

  for (const std::unique_ptr<ObjectFileBase> &F : S.getObjectFiles()) {
    const auto &File = cast<ObjectFile<ELFT>>(*F);
    Elf_Sym_Range Syms = File.getObj().symbols(File.getSymbolTable());
    if (&SymE > Syms.begin() && &SymE < Syms.end())
      SymFile = F.get();
  }

  std::string Message = "undefined symbol: " + Sym.getName().str();
  if (SymFile)
    Message += " in " + SymFile->getName().str();
  if (Config->NoInhibitExec)
    warning(Message);
  else
    error(Message);
}

// Create output section objects and add them to OutputSections.
template <class ELFT> void Writer<ELFT>::createSections() {
  SmallDenseMap<SectionKey<ELFT::Is64Bits>, OutputSection<ELFT> *> Map;

  OutputSections.push_back(&BssSec);
  Map[{BssSec.getName(), BssSec.getType(), BssSec.getFlags()}] = &BssSec;

  SymbolTable &Symtab = SymTabSec.getSymTable();
  for (const std::unique_ptr<ObjectFileBase> &FileB : Symtab.getObjectFiles()) {
    auto &File = cast<ObjectFile<ELFT>>(*FileB);
    if (!Config->DiscardAll) {
      Elf_Sym_Range Syms = File.getLocalSymbols();
      for (const Elf_Sym &Sym : Syms) {
        ErrorOr<StringRef> SymName = Sym.getName(File.getStringTable());
        if (SymName && shouldKeepInSymtab(*SymName))
          SymTabSec.addSymbol(*SymName, true);
      }
    }
    for (InputSection<ELFT> *C : File.getSections()) {
      if (!C)
        continue;
      const Elf_Shdr *H = C->getSectionHdr();
      SectionKey<ELFT::Is64Bits> Key{C->getSectionName(), H->sh_type,
                                     H->sh_flags};
      OutputSection<ELFT> *&Sec = Map[Key];
      if (!Sec) {
        Sec = new (CAlloc.Allocate()) OutputSection<ELFT>(
            PltSec, GotSec, BssSec, Key.Name, Key.sh_type, Key.sh_flags);
        OutputSections.push_back(Sec);
      }
      Sec->addSection(C);
      scanRelocs(*C);
    }
  }

  if (OutputSection<ELFT> *OS =
          Map.lookup({".init_array", SHT_INIT_ARRAY, SHF_WRITE | SHF_ALLOC})) {
    Symtab.addSyntheticSym<ELFT>("__init_array_start", *OS, 0);
    Symtab.addSyntheticSym<ELFT>("__init_array_end", *OS, OS->getSize());
  }

  // FIXME: Try to avoid the extra walk over all global symbols.
  std::vector<DefinedCommon<ELFT> *> CommonSymbols;
  for (auto &P : Symtab.getSymbols()) {
    StringRef Name = P.first;
    SymbolBody *Body = P.second->Body;
    if (Body->isStrongUndefined())
      undefError<ELFT>(Symtab, *Body);

    if (auto *C = dyn_cast<DefinedCommon<ELFT>>(Body))
      CommonSymbols.push_back(C);
    if (!includeInSymtab(*Body))
      continue;
    SymTabSec.addSymbol(Name);

    if (needsDynamicSections() && includeInDynamicSymtab(*Body))
      HashSec.addSymbol(Body);
  }

  // Sort the common symbols by alignment as an heuristic to pack them better.
  std::stable_sort(
      CommonSymbols.begin(), CommonSymbols.end(),
      [](const DefinedCommon<ELFT> *A, const DefinedCommon<ELFT> *B) {
        return A->MaxAlignment > B->MaxAlignment;
      });

  uintX_t Off = BssSec.getSize();
  for (DefinedCommon<ELFT> *C : CommonSymbols) {
    const Elf_Sym &Sym = C->Sym;
    uintX_t Align = C->MaxAlignment;
    Off = RoundUpToAlignment(Off, Align);
    C->OffsetInBSS = Off;
    Off += Sym.st_size;
  }

  BssSec.setSize(Off);

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
  }
  if (!GotSec.empty())
    OutputSections.push_back(&GotSec);
  if (!PltSec.empty())
    OutputSections.push_back(&PltSec);

  std::stable_sort(
      OutputSections.begin(), OutputSections.end(),
      [](OutputSectionBase<ELFT::Is64Bits> *A,
         OutputSectionBase<ELFT::Is64Bits> *B) {
        // Place SHF_ALLOC sections first.
        return (A->getFlags() & SHF_ALLOC) && !(B->getFlags() & SHF_ALLOC);
      });

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

  const SymbolTable &Symtab = SymTabSec.getSymTable();
  auto &FirstObj = cast<ObjectFile<ELFT>>(*Symtab.getFirstELF());
  EHdr->e_ident[EI_OSABI] = FirstObj.getOSABI();

  // FIXME: Generalize the segment construction similar to how we create
  // output sections.

  EHdr->e_type = Config->Shared ? ET_DYN : ET_EXEC;
  EHdr->e_machine = FirstObj.getEMachine();
  EHdr->e_version = EV_CURRENT;
  SymbolBody *Entry = Symtab.getEntrySym();
  EHdr->e_entry =
      Entry ? getSymVA(cast<ELFSymbolBody<ELFT>>(*Entry), BssSec) : 0;
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
