//===- ICF.cpp ------------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements ICF (Identical COMDAT Folding)
//
//===----------------------------------------------------------------------===//

#include "Chunks.h"
#include "Symbols.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include <tuple>
#include <unordered_set>
#include <vector>

using namespace llvm;

namespace lld {
namespace coff {
namespace {

struct Hasher {
  size_t operator()(const SectionChunk *C) const { return C->getHash(); }
};

struct Equals {
  bool operator()(const SectionChunk *A, const SectionChunk *B) const {
    return A->equals(B);
  }
};

} // anonymous namespace

uint64_t SectionChunk::getHash() const {
  ArrayRef<uint8_t> A = getContents();
  return hash_combine(getPermissions(),
                      hash_value(SectionName),
                      NumRelocs,
                      uint32_t(Header->SizeOfRawData),
                      std::distance(Relocs.end(), Relocs.begin()),
                      hash_combine_range(A.data(), A.data() + A.size()));
}

// Returns true if this and a given chunk are identical COMDAT sections.
bool SectionChunk::equals(const SectionChunk *X) const {
  // Compare headers
  if (getPermissions() != X->getPermissions())
    return false;
  if (SectionName != X->SectionName)
    return false;
  if (Header->SizeOfRawData != X->Header->SizeOfRawData)
    return false;
  if (NumRelocs != X->NumRelocs)
    return false;

  // Compare data
  if (getContents() != X->getContents())
    return false;

  // Compare associative sections
  if (AssocChildren.size() != X->AssocChildren.size())
    return false;
  for (size_t I = 0, E = AssocChildren.size(); I != E; ++I)
    if (AssocChildren[I]->Ptr != X->AssocChildren[I]->Ptr)
      return false;

  // Compare relocations
  auto Eq = [&](const coff_relocation &R1, const coff_relocation &R2) {
    if (R1.Type != R2.Type)
      return false;
    if (R1.VirtualAddress != R2.VirtualAddress)
      return false;
    SymbolBody *B1 = File->getSymbolBody(R1.SymbolTableIndex)->repl();
    SymbolBody *B2 = X->File->getSymbolBody(R2.SymbolTableIndex)->repl();
    if (B1 == B2)
      return true;
    auto *D1 = dyn_cast<DefinedRegular>(B1);
    auto *D2 = dyn_cast<DefinedRegular>(B2);
    return (D1 && D2 &&
            D1->getValue() == D2->getValue() &&
            D1->getChunk() == D2->getChunk());
  };
  return std::equal(Relocs.begin(), Relocs.end(), X->Relocs.begin(), Eq);
}

// Merge identical COMDAT sections.
// Two sections are considered as identical when their section headers,
// contents and relocations are all the same.
void doICF(const std::vector<Chunk *> &Chunks) {
  std::unordered_set<SectionChunk *, Hasher, Equals> Set;
  bool Redo;
  do {
    Set.clear();
    Redo = false;
    for (Chunk *C : Chunks) {
      auto *SC = dyn_cast<SectionChunk>(C);
      if (!SC || !SC->isCOMDAT() || !SC->isLive())
        continue;
      auto P = Set.insert(SC);
      bool Inserted = P.second;
      if (Inserted)
        continue;
      SectionChunk *Existing = *P.first;
      SC->replaceWith(Existing);
      // By merging sections, two relocations that originally pointed to
      // different locations can now point to the same location.
      // So, repeat the process until a convegence is obtained.
      Redo = true;
    }
  } while (Redo);
}

} // namespace coff
} // namespace lld
