//===- lib/Support/IntervalMap.cpp - A sorted interval map ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the few non-templated functions in IntervalMap.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/IntervalMap.h"

namespace llvm {
namespace IntervalMapImpl {

IdxPair distribute(unsigned Nodes, unsigned Elements, unsigned Capacity,
                   const unsigned *CurSize, unsigned NewSize[],
                   unsigned Position, bool Grow) {
  assert(Elements + Grow <= Nodes * Capacity && "Not enough room for elements");
  assert(Position <= Elements && "Invalid position");
  if (!Nodes)
    return IdxPair();

  // Trivial algorithm: left-leaning even distribution.
  const unsigned PerNode = (Elements + Grow) / Nodes;
  const unsigned Extra = (Elements + Grow) % Nodes;
  IdxPair PosPair = IdxPair(Nodes, 0);
  unsigned Sum = 0;
  for (unsigned n = 0; n != Nodes; ++n) {
    Sum += NewSize[n] = PerNode + (n < Extra);
    if (PosPair.first == Nodes && Sum > Position)
      PosPair = IdxPair(n, Position - (Sum - NewSize[n]));
  }
  assert(Sum == Elements + Grow && "Bad distribution sum");

  // Subtract the Grow element that was added.
  if (Grow) {
    assert(PosPair.first < Nodes && "Bad algebra");
    assert(NewSize[PosPair.first] && "Too few elements to need Grow");
    --NewSize[PosPair.first];
  }

#ifndef NDEBUG
  Sum = 0;
  for (unsigned n = 0; n != Nodes; ++n) {
    assert(NewSize[n] <= Capacity && "Overallocated node");
    Sum += NewSize[n];
  }
  assert(Sum == Elements && "Bad distribution sum");
#endif

  return PosPair;
}

} // namespace IntervalMapImpl
} // namespace llvm 

