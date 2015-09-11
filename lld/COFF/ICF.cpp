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
// isomorphic subgraphs as possible, if we consider sections as vertices and
// relocations as edges. This may be a bit more complicated problem than you
// might think. The order of processing sections matters since merging two
// sections can make other sections, whose relocations now point to the
// section, mergeable. Graphs may contain cycles, which is common in COFF.
// We need a sophisticated algorithm to do this properly and efficiently.
//
// What we do in this file is this. We first compute strongly connected
// components of the graphs to get acyclic graphs. Then, we remove SCCs whose
// outdegree is zero from the graphs and try to merge them. This operation
// makes other SCCs to have outdegree zero, so we repeat the process until
// all SCCs are removed.
//
// This algorithm is different from what GNU gold does which is described in
// http://research.google.com/pubs/pub36912.html. I don't know which is
// faster, this or Gold's, in practice. It'd be interesting to implement the
// other algorithm to compare. Note that the gold's algorithm cannot handle
// cycles, so we need to tweak it, though.
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

// Invoke Fn for each live COMDAT successor sections of SC.
static void forEach(SectionChunk *SC, std::function<void(SectionChunk *)> Fn) {
  for (SectionChunk *C : SC->children())
    Fn(C);
  for (SymbolBody *B : SC->symbols()) {
    if (auto *D = dyn_cast<DefinedRegular>(B)) {
      SectionChunk *C = D->getChunk();
      if (C->isCOMDAT() && C->isLive())
        Fn(C);
    }
  }
}

typedef std::vector<Component *>::iterator ComponentIterator;

// Try to merge two SCCs, A and B. A and B are likely to be isomorphic
// because all sections have the same hash values.
static void tryMerge(std::vector<SectionChunk *> &A,
                     std::vector<SectionChunk *> &B) {
  // Assume that relocation targets are the same.
  size_t End = A.size();
  for (size_t I = 0; I != End; ++I) {
    assert(B[I] == B[I]->Ptr);
    B[I]->Ptr = A[I];
  }
  for (size_t I = 0; I != End; ++I) {
    if (A[I]->equals(B[I]))
      continue;
    // If we reach here, the assumption was wrong. Reset the pointers
    // to the original values and terminate the comparison.
    for (size_t I = 0; I != End; ++I)
      B[I]->Ptr = B[I];
    return;
  }
  // If we reach here, the assumption was correct. Actually replace them.
  for (size_t I = 0; I != End; ++I)
    B[I]->replaceWith(A[I]);
}

// Try to merge components. All components given to this function are
// guaranteed to have the same number of members.
static void doUniquefy(ComponentIterator Begin, ComponentIterator End) {
  // Sort component members by hash value.
  for (auto It = Begin; It != End; ++It) {
    Component *SCC = *It;
    auto Comp = [](SectionChunk *A, SectionChunk *B) {
      return A->getHash() < B->getHash();
    };
    std::sort(SCC->Members.begin(), SCC->Members.end(), Comp);
  }

  // Merge as much component members as possible.
  for (auto It = Begin; It != End;) {
    Component *SCC = *It;
    auto Bound = std::partition(It + 1, End, [&](Component *C) {
      for (size_t I = 0, E = SCC->Members.size(); I != E; ++I)
        if (SCC->Members[I]->getHash() != C->Members[I]->getHash())
          return false;
      return true;
    });

    // Components [I, Bound) are likely to have the same members
    // because all members have the same hash values. Verify that.
    for (auto I = It + 1; I != Bound; ++I)
      tryMerge(SCC->Members, (*I)->Members);
    It = Bound;
  }
}

static void uniquefy(ComponentIterator Begin, ComponentIterator End) {
  for (auto It = Begin; It != End;) {
    Component *SCC = *It;
    size_t Size = SCC->Members.size();
    auto Bound = std::partition(It + 1, End, [&](Component *C) {
      return C->Members.size() == Size;
    });
    doUniquefy(It, Bound);
    It = Bound;
  }
}

// Returns strongly connected components of the graph formed by Chunks.
// Chunks (a list of Live COMDAT sections) are considred as vertices,
// and their relocations or association are considered as edges.
static std::vector<Component *>
getSCC(const std::vector<SectionChunk *> &Chunks) {
  std::vector<Component *> Ret;
  std::vector<SectionChunk *> V;
  uint32_t Idx;

  std::function<void(SectionChunk *)> StrongConnect = [&](SectionChunk *SC) {
    SC->Index = SC->LowLink = Idx++;
    size_t Curr = V.size();
    V.push_back(SC);
    SC->OnStack = true;

    forEach(SC, [&](SectionChunk *C) {
      if (C->Index == 0) {
        StrongConnect(C);
        SC->LowLink = std::min(SC->LowLink, C->LowLink);
      } else if (C->OnStack) {
        SC->LowLink = std::min(SC->LowLink, C->Index);
      }
    });

    if (SC->LowLink != SC->Index)
      return;
    auto *SCC = new Component(
        std::vector<SectionChunk *>(V.begin() + Curr, V.end()));
    for (size_t I = Curr, E = V.size(); I != E; ++I) {
      V[I]->OnStack = false;
      V[I]->SCC = SCC;
    }
    Ret.push_back(SCC);
    V.erase(V.begin() + Curr, V.end());
  };

  for (SectionChunk *SC : Chunks) {
    if (SC->Index == 0) {
      Idx = 1;
      StrongConnect(SC);
    }
  }

  for (Component *SCC : Ret) {
    for (SectionChunk *SC : SCC->Members) {
      forEach(SC, [&](SectionChunk *C) {
        if (SCC == C->SCC)
          return;
        ++SCC->Outdegree;
        C->SCC->Predecessors.push_back(SCC);
      });
    }
  }
  return Ret;
}

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
  if (Checksum != X->Checksum)
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
// Two sections are considered the same if their section headers,
// contents and relocations are all the same.
void doICF(const std::vector<Chunk *> &Chunks) {
  std::vector<SectionChunk *> SChunks;
  for (Chunk *C : Chunks)
    if (auto *SC = dyn_cast<SectionChunk>(C))
      if (SC->isCOMDAT() && SC->isLive())
        SChunks.push_back(SC);

  std::vector<Component *> Components = getSCC(SChunks);

  while (Components.size() > 0) {
    auto Bound = std::partition(Components.begin(), Components.end(),
                                [](Component *SCC) { return SCC->Outdegree > 0; });
    uniquefy(Bound, Components.end());

    for (auto It = Bound, E = Components.end(); It != E; ++It) {
      Component *SCC = *It;
      for (Component *Pred : SCC->Predecessors)
        --Pred->Outdegree;
    }
    Components.erase(Bound, Components.end());
  }
}

} // namespace coff
} // namespace lld
