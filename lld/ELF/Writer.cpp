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
#include "Driver.h"
#include "llvm/ADT/DenseMap.h"

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;

using namespace lld;
using namespace lld::elf2;

static const int PageSize = 4096;

template <class ELFT>
Writer<ELFT>::Writer(SymbolTable *Symtab)
    : Symtab(Symtab) {}
template <class ELFT> Writer<ELFT>::~Writer() {}

// The main function of the writer.
template <class ELFT> void Writer<ELFT>::write(StringRef OutputPath) {
  createSections();
  assignAddresses();
  openFile(OutputPath);
  writeHeader();
  writeSections();
  error(Buffer->commit());
}

void OutputSection::setVA(uint64_t VA) {
  Header.sh_addr = VA;
  for (Chunk *C : Chunks)
    C->setVA(C->getVA() + VA);
}

void OutputSection::setFileOffset(uint64_t Off) {
  if (Header.sh_size == 0)
    return;
  Header.sh_offset = Off;
  for (Chunk *C : Chunks)
    C->setFileOff(C->getFileOff() + Off);
}

template <class ELFT>
void OutputSection::addChunk(Chunk *C) {
  Chunks.push_back(C);
  C->setOutputSection(this);
  uint64_t Off = Header.sh_size;
  Off = RoundUpToAlignment(Off, C->getAlign());
  C->setVA(Off);
  C->setFileOff(Off);
  Off += C->getSize();
  Header.sh_size = Off;
  if (auto SC = dyn_cast<SectionChunk<ELFT>>(C)) {
    Header.sh_type = SC->getSectionHdr()->sh_type;
    Header.sh_flags |= SC->getSectionHdr()->sh_flags;
  }
}

template <class ELFT>
void OutputSection::writeHeaderTo(Elf_Shdr_Impl<ELFT> *SHdr) {
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

// Create output section objects and add them to OutputSections.
template <class ELFT> void Writer<ELFT>::createSections() {
  SmallDenseMap<StringRef, OutputSection *> Map;
  for (std::unique_ptr<ObjectFileBase> &File : Symtab->ObjectFiles) {
    for (Chunk *C : File->getChunks()) {
      OutputSection *&Sec = Map[C->getSectionName()];
      if (!Sec) {
        Sec = new (CAlloc.Allocate()) OutputSection(C->getSectionName());
        OutputSections.push_back(Sec);
      }
      Sec->addChunk<ELFT>(C);
    }
  }
}

// Visits all sections to assign incremental, non-overlapping RVAs and
// file offsets.
template <class ELFT> void Writer<ELFT>::assignAddresses() {
  SizeOfHeaders = RoundUpToAlignment(sizeof(Elf_Ehdr_Impl<ELFT>), PageSize);
  uint64_t VA = 0x1000; // The first page is kept unmapped.
  uint64_t FileOff = SizeOfHeaders;
  for (OutputSection *Sec : OutputSections) {
    Sec->setVA(VA);
    Sec->setFileOffset(FileOff);
    VA += RoundUpToAlignment(Sec->getSize(), PageSize);
    FileOff += RoundUpToAlignment(Sec->getSize(), 8);
  }
  // Add space for section headers.
  SectionHeaderOff = FileOff;
  FileOff += (OutputSections.size() + 1) * sizeof(Elf_Shdr_Impl<ELFT>);
  FileSize = SizeOfHeaders + RoundUpToAlignment(FileOff - SizeOfHeaders, 8);
}

template <class ELFT> void Writer<ELFT>::writeHeader() {
  uint8_t *Buf = Buffer->getBufferStart();
  auto *EHdr = reinterpret_cast<Elf_Ehdr_Impl<ELFT> *>(Buf);
  EHdr->e_ident[EI_MAG0] = 0x7F;
  EHdr->e_ident[EI_MAG1] = 0x45;
  EHdr->e_ident[EI_MAG2] = 0x4C;
  EHdr->e_ident[EI_MAG3] = 0x46;
  EHdr->e_ident[EI_CLASS] = ELFCLASS64;
  EHdr->e_ident[EI_DATA] = ELFDATA2LSB;
  EHdr->e_ident[EI_VERSION] = EV_CURRENT;
  EHdr->e_ident[EI_OSABI] = ELFOSABI_NONE;

  EHdr->e_type = ET_EXEC;
  EHdr->e_machine = EM_X86_64;
  EHdr->e_version = EV_CURRENT;
  EHdr->e_entry = 0x401000;
  EHdr->e_phoff = sizeof(Elf_Ehdr_Impl<ELFT>);
  EHdr->e_shoff = SectionHeaderOff;
  EHdr->e_ehsize = sizeof(Elf_Ehdr_Impl<ELFT>);
  EHdr->e_phentsize = sizeof(Elf_Phdr_Impl<ELFT>);
  EHdr->e_phnum = 1;
  EHdr->e_shentsize = sizeof(Elf_Shdr_Impl<ELFT>);
  EHdr->e_shnum = OutputSections.size() + 1;
  EHdr->e_shstrndx = 0;

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
  for (OutputSection *Sec : OutputSections)
    Sec->writeHeaderTo<ELFT>(SHdrs++);
}

template <class ELFT> void Writer<ELFT>::openFile(StringRef Path) {
  std::error_code EC = FileOutputBuffer::create(Path, FileSize, Buffer,
                                                FileOutputBuffer::F_executable);
  error(EC, Twine("failed to open ") + Path);
}

// Write section contents to a mmap'ed file.
template <class ELFT> void Writer<ELFT>::writeSections() {
  uint8_t *Buf = Buffer->getBufferStart();
  for (OutputSection *Sec : OutputSections) {
    for (Chunk *C : Sec->getChunks())
      C->writeTo(Buf);
  }
}

namespace lld {
namespace elf2 {
template class Writer<ELF32LE>;
template class Writer<ELF32BE>;
template class Writer<ELF64LE>;
template class Writer<ELF64BE>;

template void OutputSection::addChunk<ELF32LE>(Chunk *);
template void OutputSection::addChunk<ELF32BE>(Chunk *);
template void OutputSection::addChunk<ELF64LE>(Chunk *);
template void OutputSection::addChunk<ELF64BE>(Chunk *);
}
}
