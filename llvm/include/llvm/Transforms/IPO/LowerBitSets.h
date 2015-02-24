//===- LowerBitSets.h - Bitset lowering pass --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines parts of the bitset lowering pass implementation that may
// be usefully unit tested.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_LOWERBITSETS_H
#define LLVM_TRANSFORMS_IPO_LOWERBITSETS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <stdint.h>
#include <limits>
#include <set>
#include <vector>

namespace llvm {

class DataLayout;
class GlobalVariable;
class Value;

struct BitSetInfo {
  // The actual bitset.
  std::vector<uint8_t> Bits;

  // The byte offset into the combined global represented by the bitset.
  uint64_t ByteOffset;

  // The size of the bitset in bits.
  uint64_t BitSize;

  // Log2 alignment of the bit set relative to the combined global.
  // For example, a log2 alignment of 3 means that bits in the bitset
  // represent addresses 8 bytes apart.
  unsigned AlignLog2;

  bool isSingleOffset() const {
    return Bits.size() == 1 && Bits[0] == 1;
  }

  bool containsGlobalOffset(uint64_t Offset) const;

  bool containsValue(const DataLayout *DL,
                     const DenseMap<GlobalVariable *, uint64_t> &GlobalLayout,
                     Value *V, uint64_t COffset = 0) const;

};

struct BitSetBuilder {
  SmallVector<uint64_t, 16> Offsets;
  uint64_t Min, Max;

  BitSetBuilder() : Min(std::numeric_limits<uint64_t>::max()), Max(0) {}

  void addOffset(uint64_t Offset) {
    if (Min > Offset)
      Min = Offset;
    if (Max < Offset)
      Max = Offset;

    Offsets.push_back(Offset);
  }

  BitSetInfo build();
};

/// This class implements a layout algorithm for globals referenced by bit sets
/// that tries to keep members of small bit sets together. This can
/// significantly reduce bit set sizes in many cases.
///
/// It works by assembling fragments of layout from sets of referenced globals.
/// Each set of referenced globals causes the algorithm to create a new
/// fragment, which is assembled by appending each referenced global in the set
/// into the fragment. If a referenced global has already been referenced by an
/// fragment created earlier, we instead delete that fragment and append its
/// contents into the fragment we are assembling.
///
/// By starting with the smallest fragments, we minimize the size of the
/// fragments that are copied into larger fragments. This is most intuitively
/// thought about when considering the case where the globals are virtual tables
/// and the bit sets represent their derived classes: in a single inheritance
/// hierarchy, the optimum layout would involve a depth-first search of the
/// class hierarchy (and in fact the computed layout ends up looking a lot like
/// a DFS), but a naive DFS would not work well in the presence of multiple
/// inheritance. This aspect of the algorithm ends up fitting smaller
/// hierarchies inside larger ones where that would be beneficial.
///
/// For example, consider this class hierarchy:
///
/// A       B
///   \   / | \
///     C   D   E
///
/// We have five bit sets: bsA (A, C), bsB (B, C, D, E), bsC (C), bsD (D) and
/// bsE (E). If we laid out our objects by DFS traversing B followed by A, our
/// layout would be {B, C, D, E, A}. This is optimal for bsB as it needs to
/// cover the only 4 objects in its hierarchy, but not for bsA as it needs to
/// cover 5 objects, i.e. the entire layout. Our algorithm proceeds as follows:
///
/// Add bsC, fragments {{C}}
/// Add bsD, fragments {{C}, {D}}
/// Add bsE, fragments {{C}, {D}, {E}}
/// Add bsA, fragments {{A, C}, {D}, {E}}
/// Add bsB, fragments {{B, A, C, D, E}}
///
/// This layout is optimal for bsA, as it now only needs to cover two (i.e. 3
/// fewer) objects, at the cost of bsB needing to cover 1 more object.
///
/// The bit set lowering pass assigns an object index to each object that needs
/// to be laid out, and calls addFragment for each bit set passing the object
/// indices of its referenced globals. It then assembles a layout from the
/// computed layout in the Fragments field.
struct GlobalLayoutBuilder {
  /// The computed layout. Each element of this vector contains a fragment of
  /// layout (which may be empty) consisting of object indices.
  std::vector<std::vector<uint64_t>> Fragments;

  /// Mapping from object index to fragment index.
  std::vector<uint64_t> FragmentMap;

  GlobalLayoutBuilder(uint64_t NumObjects)
      : Fragments(1), FragmentMap(NumObjects) {}

  /// Add \param F to the layout while trying to keep its indices contiguous.
  /// If a previously seen fragment uses any of \param F's indices, that
  /// fragment will be laid out inside \param F.
  void addFragment(const std::set<uint64_t> &F);
};

} // namespace llvm

#endif
