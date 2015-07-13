//===- Writer.cpp ---------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Config.h"
#include "Writer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdio>
#include <functional>
#include <unordered_map>
#include <utility>

#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;
using namespace llvm::support;
using namespace llvm::support::endian;

static const int PageSize = 4096;

struct SectionTraits {
  uint64_t Type;
  uint64_t Flags;
  StringRef Name;
};

bool operator==(const SectionTraits &A, const SectionTraits &B) {
  return A.Type == B.Type && A.Flags == B.Flags && A.Name == B.Name;
}

namespace std {
template <> struct hash<SectionTraits> {
  size_t operator()(const SectionTraits &ST) const {
    return hash_combine(ST.Type, ST.Flags, ST.Name);
  }
};
}

using namespace lld;
using namespace lld::elfv2;

// The main function of the writer.
template <class ELFT>
std::error_code Writer<ELFT>::write(StringRef OutputPath) {
  markLive();
  createSections();
  assignAddresses();
  removeEmptySections();
  if (auto EC = openFile(OutputPath))
    return EC;
  writeHeader();
  writeSections();
  return Buffer->commit();
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

void OutputSection::addChunk(Chunk *C) {
  Chunks.push_back(C);
  C->setOutputSection(this);
  uint64_t Off = Header.sh_size;
  Off = RoundUpToAlignment(Off, C->getAlign());
  C->setVA(Off);
  C->setFileOff(Off);
  Off += C->getSize();
  Header.sh_size = Off;
}

void OutputSection::addPermissions(uint32_t C) {
  // Header.Characteristics |= C & PermMask;
}

// Write the section header to a given buffer.
void OutputSection::writeHeaderTo(uint8_t *Buf) {}

// Set live bit on for each reachable chunk. Unmarked (unreachable)
// COMDAT chunks will be ignored in the next step, so that they don't
// come to the final output file.
template <class ELFT> void Writer<ELFT>::markLive() {
  if (!Config->DoGC)
    return;
  for (StringRef Name : Config->GCRoots)
    cast<Defined>(Symtab->find(Name))->markLive();
  for (Chunk *C : Symtab->getChunks())
    if (C->isRoot())
      C->markLive();
}

static SectionTraits getChunkTraits(Chunk *C) {
  return {0, C->getFlags(), C->getSectionName()};
}

// Create output section objects and add them to OutputSections.
template <class ELFT> void Writer<ELFT>::createSections() {
  std::unordered_map<SectionTraits, std::vector<Chunk *>> Map;
  for (Chunk *C : Symtab->getChunks()) {
    if (Config->DoGC && !C->isLive()) {
      if (Config->Verbose)
        C->printDiscardedMessage();
      continue;
    }
    Map[getChunkTraits(C)].push_back(C);
  }

  for (auto &P : Map) {
    auto Sec = new (CAlloc.Allocate())
        OutputSection(P.first.Name, OutputSections.size());
    OutputSections.push_back(Sec);
    for (Chunk *C : P.second) {
      Sec->addChunk(C);
      Sec->addPermissions(C->getFlags());
    }
  }
}

template <class ELFT> void Writer<ELFT>::removeEmptySections() {
  auto IsEmpty = [](OutputSection *S) { return S->getSize() == 0; };
  OutputSections.erase(
      std::remove_if(OutputSections.begin(), OutputSections.end(), IsEmpty),
      OutputSections.end());
}

// Visits all sections to assign incremental, non-overlapping RVAs and
// file offsets.
template <class ELFT> void Writer<ELFT>::assignAddresses() {
  SizeOfHeaders = RoundUpToAlignment(sizeof(Elf_Ehdr_Impl<ELFT>) +
                                         sizeof(Elf_Shdr_Impl<ELFT>) *
                                             OutputSections.size(),
                                     PageSize);
  uint64_t VA = 0x1000; // The first page is kept unmapped.
  uint64_t FileOff = SizeOfHeaders;
  for (OutputSection *Sec : OutputSections) {
    Sec->setVA(VA);
    Sec->setFileOffset(FileOff);
    VA += RoundUpToAlignment(Sec->getSize(), PageSize);
    FileOff += RoundUpToAlignment(Sec->getSize(), 8);
  }
  SizeOfImage = SizeOfHeaders + RoundUpToAlignment(VA - 0x1000, PageSize);
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
  EHdr->e_ident[EI_OSABI] = ELFOSABI_GNU;

  EHdr->e_type = ET_EXEC;
  EHdr->e_machine = EM_X86_64;
  EHdr->e_version = EV_CURRENT;
  EHdr->e_entry = 0x401000;
  EHdr->e_phoff = sizeof(Elf_Ehdr_Impl<ELFT>);
  EHdr->e_shoff = 0;
  EHdr->e_ehsize = sizeof(Elf_Ehdr_Impl<ELFT>);
  EHdr->e_phentsize = sizeof(Elf_Phdr_Impl<ELFT>);
  EHdr->e_phnum = 1;
  EHdr->e_shentsize = sizeof(Elf_Shdr_Impl<ELFT>);
  EHdr->e_shnum = 0;
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
}

template <class ELFT> std::error_code Writer<ELFT>::openFile(StringRef Path) {
  if (auto EC = FileOutputBuffer::create(Path, FileSize, Buffer,
                                         FileOutputBuffer::F_executable)) {
    llvm::errs() << "failed to open " << Path << ": " << EC.message() << "\n";
    return EC;
  }
  return std::error_code();
}

// Write section contents to a mmap'ed file.
template <class ELFT> void Writer<ELFT>::writeSections() {
  uint8_t *Buf = Buffer->getBufferStart();
  for (OutputSection *Sec : OutputSections) {
    // Fill gaps between functions in .text with nop instructions instead of
    // leaving as null bytes (which can be interpreted as ADD instructions).
    if (Sec->getPermissions() & PF_X)
      memset(Buf + Sec->getFileOff(), 0x90, Sec->getSize());
    for (Chunk *C : Sec->getChunks())
      C->writeTo(Buf);
  }
}

template <class ELFT> OutputSection *Writer<ELFT>::findSection(StringRef Name) {
  for (OutputSection *Sec : OutputSections)
    if (Sec->getName() == Name)
      return Sec;
  return nullptr;
}

template class Writer<ELF32LE>;
template class Writer<ELF32BE>;
template class Writer<ELF64LE>;
template class Writer<ELF64BE>;
