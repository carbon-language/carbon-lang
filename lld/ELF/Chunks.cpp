//===- Chunks.cpp ---------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Chunks.h"
#include "InputFiles.h"
#include "Writer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::support::endian;
using namespace llvm::ELF;

using namespace lld;
using namespace lld::elfv2;

template <class ELFT>
SectionChunk<ELFT>::SectionChunk(elfv2::ObjectFile<ELFT> *F, const Elf_Shdr *H,
                                 uint32_t SI)
    : File(F), Header(H), SectionIndex(SI) {
  // Initialize SectionName.
  SectionName = *File->getObj()->getSectionName(Header);

  Align = Header->sh_addralign;

  // When a new chunk is created, we don't if if it's going to make it
  // to the final output. Initially all sections are unmarked in terms
  // of garbage collection. The writer will call markLive() to mark
  // all reachable section chunks.
  Live = false;

  Root = true;
}

template <class ELFT> void SectionChunk<ELFT>::writeTo(uint8_t *Buf) {
  if (!hasData())
    return;
  // Copy section contents from source object file to output file.
  ArrayRef<uint8_t> Data = *File->getObj()->getSectionContents(Header);
  memcpy(Buf + FileOff, Data.data(), Data.size());

  // FIXME: Relocations
}

template <class ELFT> void SectionChunk<ELFT>::mark() {
  assert(!Live);
  Live = true;

  // Mark all symbols listed in the relocation table for this section.
  // FIXME: Relocations
}

static void add16(uint8_t *P, int16_t V) { write16le(P, read16le(P) + V); }
static void add32(uint8_t *P, int32_t V) { write32le(P, read32le(P) + V); }
static void add64(uint8_t *P, int64_t V) { write64le(P, read64le(P) + V); }

template <class ELFT>
void SectionChunk<ELFT>::applyReloc(uint8_t *Buf, const Elf_Rela *Rel) {
  // FIXME: Relocations
}

template <class ELFT>
void SectionChunk<ELFT>::applyReloc(uint8_t *Buf, const Elf_Rel *Rel) {}

template <class ELFT> bool SectionChunk<ELFT>::hasData() const {
  return Header->sh_type != SHT_NOBITS;
}

template <class ELFT> uint32_t SectionChunk<ELFT>::getFlags() const {
  return Header->sh_flags;
}

// Prints "Discarded <symbol>" for all external function symbols.
template <class ELFT> void SectionChunk<ELFT>::printDiscardedMessage() {
  auto Obj = File->getObj();

  for (auto &&Sym : Obj->symbols()) {
    auto Sec = Obj->getSection(&Sym);
    if (Sec && *Sec != Header)
      continue;
    if (Sym.getType() != STT_FUNC)
      continue;
    if (auto Name = Obj->getStaticSymbolName(&Sym)) {
      llvm::outs() << "Discarded " << *Name << " from " << File->getShortName()
                   << "\n";
    }
  }
}

template <class ELFT>
const llvm::object::Elf_Shdr_Impl<ELFT> *SectionChunk<ELFT>::getSectionHdr() {
  return Header;
}

template <class ELFT>
CommonChunk<ELFT>::CommonChunk(const Elf_Sym *S)
    : Sym(S) {
  // Alignment is a section attribute, but common symbols don't
  // belong to any section. How do we know common data alignments?
  // Needs investigating. For now, we set a large number as an alignment.
  Align = 16;
}

template <class ELFT> uint32_t CommonChunk<ELFT>::getFlags() const {
  return PF_R | PF_W;
}

namespace lld {
namespace elfv2 {
template class SectionChunk<llvm::object::ELF32LE>;
template class SectionChunk<llvm::object::ELF32BE>;
template class SectionChunk<llvm::object::ELF64LE>;
template class SectionChunk<llvm::object::ELF64BE>;

template class CommonChunk<llvm::object::ELF32LE>;
template class CommonChunk<llvm::object::ELF32BE>;
template class CommonChunk<llvm::object::ELF64LE>;
template class CommonChunk<llvm::object::ELF64BE>;
}
}
