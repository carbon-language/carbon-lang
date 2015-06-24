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
#include <tuple>
#include <unordered_set>
#include <vector>

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

// Merge identical COMDAT sections.
// Two sections are considered as identical when their section headers,
// contents and relocations are all the same.
void doICF(const std::vector<Chunk *> &Chunks) {
  std::unordered_set<SectionChunk *, Hasher, Equals> Set;
  bool removed = false;
  for (Chunk *C : Chunks) {
    if (!C->isCOMDAT() || !C->isLive())
      continue;
    auto *SC = reinterpret_cast<SectionChunk *>(C);
    auto P = Set.insert(SC);
    bool Inserted = P.second;
    if (Inserted)
      continue;
    SectionChunk *Existing = *P.first;
    SC->replaceWith(Existing);
    removed = true;
  }
  // By merging sections, two relocations that originally pointed to
  // different locations can now point to the same location.
  // So, repeat the process until a convegence is obtained.
  if (removed)
    doICF(Chunks);
}

} // namespace coff
} // namespace lld
