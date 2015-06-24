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
#include "llvm/Object/COFF.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace llvm;
using namespace llvm::object;
using namespace llvm::support::endian;
using namespace llvm::COFF;

namespace lld {
namespace coff {

SectionChunk::SectionChunk(ObjectFile *F, const coff_section *H, uint32_t SI)
    : File(F), Header(H), SectionIndex(SI) {
  // Initialize SectionName.
  File->getCOFFObj()->getSectionName(Header, SectionName);

  // Bit [20:24] contains section alignment. Both 0 and 1 mean alignment 1.
  unsigned Shift = (Header->Characteristics >> 20) & 0xF;
  if (Shift > 0)
    Align = uint32_t(1) << (Shift - 1);

  // When a new chunk is created, we don't if if it's going to make it
  // to the final output. Initially all sections are unmarked in terms
  // of garbage collection. The writer will call markLive() to mark
  // all reachable section chunks.
  Live = false;

  // COMDAT sections are not GC root. Non-text sections are not
  // subject of garbage collection (thus they are root).
  if (!isCOMDAT() && !(Header->Characteristics & IMAGE_SCN_CNT_CODE))
    Root = true;
}

void SectionChunk::writeTo(uint8_t *Buf) {
  if (!hasData())
    return;
  // Copy section contents from source object file to output file.
  ArrayRef<uint8_t> Data;
  File->getCOFFObj()->getSectionContents(Header, Data);
  memcpy(Buf + FileOff, Data.data(), Data.size());

  // Apply relocations.
  for (const auto &I : getSectionRef().relocations()) {
    const coff_relocation *Rel = File->getCOFFObj()->getCOFFRelocation(I);
    applyReloc(Buf, Rel);
  }
}

void SectionChunk::mark() {
  assert(!Live);
  Live = true;

  // Mark all symbols listed in the relocation table for this section.
  for (const auto &I : getSectionRef().relocations()) {
    const coff_relocation *Rel = File->getCOFFObj()->getCOFFRelocation(I);
    SymbolBody *B = File->getSymbolBody(Rel->SymbolTableIndex);
    if (auto *Def = dyn_cast<Defined>(B))
      Def->markLive();
  }

  // Mark associative sections if any.
  for (Chunk *C : AssocChildren)
    C->markLive();
}

void SectionChunk::addAssociative(SectionChunk *Child) {
  AssocChildren.push_back(Child);
  // Associative sections are live if their parent COMDATs are live,
  // and vice versa, so they are not considered live by themselves.
  Child->Root = false;
}

static void add16(uint8_t *P, int16_t V) { write16le(P, read16le(P) + V); }
static void add32(uint8_t *P, int32_t V) { write32le(P, read32le(P) + V); }
static void add64(uint8_t *P, int64_t V) { write64le(P, read64le(P) + V); }

// Implements x64 PE/COFF relocations.
void SectionChunk::applyReloc(uint8_t *Buf, const coff_relocation *Rel) {
  uint8_t *Off = Buf + FileOff + Rel->VirtualAddress;
  SymbolBody *Body = File->getSymbolBody(Rel->SymbolTableIndex);
  uint64_t S = cast<Defined>(Body)->getRVA();
  uint64_t P = RVA + Rel->VirtualAddress;
  switch (Rel->Type) {
  case IMAGE_REL_AMD64_ADDR32:   add32(Off, S + Config->ImageBase); break;
  case IMAGE_REL_AMD64_ADDR64:   add64(Off, S + Config->ImageBase); break;
  case IMAGE_REL_AMD64_ADDR32NB: add32(Off, S); break;
  case IMAGE_REL_AMD64_REL32:    add32(Off, S - P - 4); break;
  case IMAGE_REL_AMD64_REL32_1:  add32(Off, S - P - 5); break;
  case IMAGE_REL_AMD64_REL32_2:  add32(Off, S - P - 6); break;
  case IMAGE_REL_AMD64_REL32_3:  add32(Off, S - P - 7); break;
  case IMAGE_REL_AMD64_REL32_4:  add32(Off, S - P - 8); break;
  case IMAGE_REL_AMD64_REL32_5:  add32(Off, S - P - 9); break;
  case IMAGE_REL_AMD64_SECTION:  add16(Off, Out->getSectionIndex()); break;
  case IMAGE_REL_AMD64_SECREL:   add32(Off, S - Out->getRVA()); break;
  default:
    llvm::report_fatal_error("Unsupported relocation type");
  }
}

// Windows-specific.
// Collect all locations that contain absolute 64-bit addresses,
// which need to be fixed by the loader if load-time relocation is needed.
// Only called when base relocation is enabled.
void SectionChunk::getBaserels(std::vector<uint32_t> *Res, Defined *ImageBase) {
  for (const auto &I : getSectionRef().relocations()) {
    // ADDR64 relocations contain absolute addresses.
    // Symbol __ImageBase is special -- it's an absolute symbol, but its
    // address never changes even if image is relocated.
    const coff_relocation *Rel = File->getCOFFObj()->getCOFFRelocation(I);
    if (Rel->Type != IMAGE_REL_AMD64_ADDR64)
      continue;
    SymbolBody *Body = File->getSymbolBody(Rel->SymbolTableIndex);
    if (Body == ImageBase)
      continue;
    Res->push_back(RVA + Rel->VirtualAddress);
  }
}

bool SectionChunk::hasData() const {
  return !(Header->Characteristics & IMAGE_SCN_CNT_UNINITIALIZED_DATA);
}

uint32_t SectionChunk::getPermissions() const {
  return Header->Characteristics & PermMask;
}

bool SectionChunk::isCOMDAT() const {
  return Header->Characteristics & IMAGE_SCN_LNK_COMDAT;
}

void SectionChunk::printDiscardedMessage() {
  llvm::dbgs() << "Discarded " << Sym->getName() << "\n";
}

SectionRef SectionChunk::getSectionRef() {
  DataRefImpl Ref;
  Ref.p = uintptr_t(Header);
  return SectionRef(Ref, File->getCOFFObj());
}

StringRef SectionChunk::getDebugName() {
  return Sym->getName();
}

CommonChunk::CommonChunk(const COFFSymbolRef S) : Sym(S) {
  // Common symbols are aligned on natural boundaries up to 32 bytes.
  // This is what MSVC link.exe does.
  Align = std::min(uint64_t(32), NextPowerOf2(Sym.getValue()));
}

uint32_t CommonChunk::getPermissions() const {
  return IMAGE_SCN_CNT_UNINITIALIZED_DATA | IMAGE_SCN_MEM_READ |
         IMAGE_SCN_MEM_WRITE;
}

void StringChunk::writeTo(uint8_t *Buf) {
  memcpy(Buf + FileOff, Str.data(), Str.size());
}

void ImportThunkChunk::writeTo(uint8_t *Buf) {
  memcpy(Buf + FileOff, ImportThunkData, sizeof(ImportThunkData));
  // The first two bytes is a JMP instruction. Fill its operand.
  uint32_t Operand = ImpSymbol->getRVA() - RVA - getSize();
  write32le(Buf + FileOff + 2, Operand);
}

// Windows-specific.
// This class represents a block in .reloc section.
BaserelChunk::BaserelChunk(uint32_t Page, uint32_t *Begin, uint32_t *End) {
  // Block header consists of 4 byte page RVA and 4 byte block size.
  // Each entry is 2 byte. Last entry may be padding.
  Data.resize(RoundUpToAlignment((End - Begin) * 2 + 8, 4));
  uint8_t *P = Data.data();
  write32le(P, Page);
  write32le(P + 4, Data.size());
  P += 8;
  for (uint32_t *I = Begin; I != End; ++I) {
    write16le(P, (IMAGE_REL_BASED_DIR64 << 12) | (*I - Page));
    P += 2;
  }
}

void BaserelChunk::writeTo(uint8_t *Buf) {
  memcpy(Buf + FileOff, Data.data(), Data.size());
}

} // namespace coff
} // namespace lld
