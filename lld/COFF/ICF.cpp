//===- ICF.cpp ------------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Identical COMDAT Folding is a feature to merge COMDAT sections not by
// name (which is regular COMDAT handling) but by contents. If two COMDAT
// sections have the same data, relocations, attributes, etc., then the two
// are considered identical and merged by the linker. This optimization
// makes outputs smaller.
//
// ICF is theoretically a problem of reducing graphs by merging as many
// identical subgraphs as possible, if we consider sections as vertices and
// relocations as edges. This may be a bit more complicated problem than you
// might think. The order of processing sections matters since merging two
// sections can make other sections, whose relocations now point to the same
// section, mergeable. Graphs may contain cycles, which is common in COFF.
// We need a sophisticated algorithm to do this properly and efficiently.
//
// What we do in this file is this. We split sections into groups. Sections
// in the same group are considered identical.
//
// First, all sections are grouped by their "constant" values. Constant
// values are values that never change by ICF, such as section contents,
// section name, number of relocations, type and offset of each relocation,
// etc. Because we do not care about relocation targets in this spep, two
// sections in the same group may be actually not identical, but at least
// two sections in different groups can never be identical.
//
// Then, we try to split each group by relocation targets. Relocations are
// considered identical if and only if the relocation targets are in the
// same group. Splitting a group may make more groups to be splittable,
// because two relocations that were previously considered identical might
// now point to different groups. We repeat this step until the convergence
// is obtained.
//
// This algorithm is so-called "optimistic" algorithm described in
// http://research.google.com/pubs/pub36912.html.
//
//===----------------------------------------------------------------------===//

#include "Chunks.h"
#include "Symbols.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <functional>
#include <tuple>
#include <unordered_set>
#include <vector>

using namespace llvm;

namespace lld {
namespace coff {

uint64_t SectionChunk::getHash() const {
  if (Hash == 0) {
    Hash = hash_combine(getPermissions(),
                        hash_value(SectionName),
                        NumRelocs,
                        uint32_t(Header->SizeOfRawData),
                        std::distance(Relocs.end(), Relocs.begin()),
                        Checksum);
  }
  return Hash;
}

void SectionChunk::initSuccessors() {
  Successors = AssocChildren;
  for (const coff_relocation &R : Relocs) {
    SymbolBody *B = File->getSymbolBody(R.SymbolTableIndex)->repl();
    if (auto D = dyn_cast<DefinedRegular>(B))
      Successors.push_back(D->getChunk());
  }
}

bool SectionChunk::equalsVertex(const SectionChunk *A, const SectionChunk *B) {
  if (A->getPermissions() != B->getPermissions() ||
      A->SectionName != B->SectionName ||
      A->Header->SizeOfRawData != B->Header->SizeOfRawData ||
      A->NumRelocs != B->NumRelocs ||
      A->Checksum != B->Checksum ||
      A->AssocChildren.size() != B->AssocChildren.size() ||
      A->getContents() != B->getContents()) {
    return false;
  }

  for (size_t I = 0, E = A->AssocChildren.size(); I != E; ++I)
    if (A->AssocChildren[I]->GroupID != B->AssocChildren[I]->GroupID)
      return false;

  // Compare relocations
  auto Eq = [&](const coff_relocation &R1, const coff_relocation &R2) {
    if (R1.Type != R2.Type ||
        R1.VirtualAddress != R2.VirtualAddress) {
      return false;
    }
    SymbolBody *B1 = A->File->getSymbolBody(R1.SymbolTableIndex)->repl();
    SymbolBody *B2 = B->File->getSymbolBody(R2.SymbolTableIndex)->repl();
    if (B1 == B2)
      return true;
    auto *D1 = dyn_cast<DefinedRegular>(B1);
    auto *D2 = dyn_cast<DefinedRegular>(B2);
    return D1 && D2 &&
           D1->getValue() == D2->getValue() &&
           D1->getChunk()->GroupID == D2->getChunk()->GroupID;
  };
  return std::equal(A->Relocs.begin(), A->Relocs.end(), B->Relocs.begin(), Eq);
}

bool SectionChunk::equalsEdge(const SectionChunk *A, const SectionChunk *B) {
  assert(A->Successors.size() == B->Successors.size());
  return std::equal(A->Successors.begin(), A->Successors.end(),
                    B->Successors.begin(),
                    [](const SectionChunk *X, const SectionChunk *Y) {
                      return X->GroupID == Y->GroupID;
                    });
}

typedef std::vector<SectionChunk *>::iterator ChunkIterator;
typedef bool (*Comparator)(const SectionChunk *, const SectionChunk *);
static uint64_t NextID = 0;

static bool partition(ChunkIterator Begin, ChunkIterator End, Comparator Eq) {
  bool R = false;
  for (auto It = Begin;;) {
    SectionChunk *Head = *It;
    auto Bound = std::partition(It + 1, End, [&](SectionChunk *SC) {
      return Eq(Head, SC);
    });
    if (Bound == End)
      return R;
    size_t ID = NextID++;
    std::for_each(It, Bound, [&](SectionChunk *SC) { SC->GroupID = ID; });
    It = Bound;
    R = true;
  }
}

static bool forEachGroup(std::vector<SectionChunk *> &SChunks,
                         Comparator Eq) {
  bool R = false;
  for (auto It = SChunks.begin(), End = SChunks.end(); It != End;) {
    SectionChunk *Head = *It;
    auto Bound = std::find_if(It + 1, End, [&](SectionChunk *SC) {
      return SC->GroupID != Head->GroupID;
    });
    if (partition(It, Bound, Eq))
      R = true;
    It = Bound;
  }
  return R;
}

// Merge identical COMDAT sections.
// Two sections are considered the same if their section headers,
// contents and relocations are all the same.
void doICF(const std::vector<Chunk *> &Chunks) {
  if (Config->Verbose)
    llvm::outs() << "\nICF\n";

  // Collect only mergeable sections.
  std::vector<SectionChunk *> SChunks;
  for (Chunk *C : Chunks) {
    if (auto *SC = dyn_cast<SectionChunk>(C)) {
      bool Writable = SC->getPermissions() & llvm::COFF::IMAGE_SCN_MEM_WRITE;
      if (SC->isCOMDAT() && SC->isLive() && !Writable) {
        SChunks.push_back(SC);
        SC->GroupID = SC->getHash() | (uint64_t(1) << 63);
      } else {
        SC->GroupID = NextID++;
      }
    }
  }

  std::sort(SChunks.begin(), SChunks.end(),
            [](SectionChunk *A, SectionChunk *B) {
              return A->GroupID < B->GroupID;
            });

  // Split groups until we get a convergence.
  for (SectionChunk *SC : SChunks)
    SC->initSuccessors();
  forEachGroup(SChunks, SectionChunk::equalsVertex);
  while (forEachGroup(SChunks, SectionChunk::equalsEdge));

  // Merge sections in the same group.
  for (auto It = SChunks.begin(), End = SChunks.end(); It != End;) {
    SectionChunk *Head = *It;
    auto Bound = std::find_if(It + 1, End, [&](SectionChunk *SC) {
      return Head->GroupID != SC->GroupID;
    });
    if (std::distance(It, Bound) == 1) {
      It = Bound;
      continue;
    }
    if (Config->Verbose)
      llvm::outs() << "Selected " << Head->getDebugName() << "\n";
    for (++It; It != Bound; ++It) {
      SectionChunk *SC = *It;
      if (Config->Verbose)
        llvm::outs() << "  Removed " << SC->getDebugName() << "\n";
      SC->replaceWith(Head);
    }
  }
}

} // namespace coff
} // namespace lld
