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

} // namespace llvm

#endif
