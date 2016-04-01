//===--- BasicBlockOffsetRanges.h - list of address ranges relative to BBs ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Represents a list of address ranges where addresses are relative to the
// beginning of basic blocks. Useful for converting address ranges in the input
// binary to equivalent ranges after optimizations take place.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_BASIC_BLOCK_OFFSET_RANGES_H
#define LLVM_TOOLS_LLVM_BOLT_BASIC_BLOCK_OFFSET_RANGES_H

#include "llvm/ADT/SmallVector.h"
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace llvm {
namespace bolt {

class BinaryFunction;
class BinaryBasicBlock;

class BasicBlockOffsetRanges {
public:
  typedef SmallVectorImpl<unsigned char> BinaryData;
  struct AbsoluteRange {
    uint64_t Begin;
    uint64_t End;
    const BinaryData *Data;
  };

  /// Add range [BeginAddress, EndAddress) to the address ranges list.
  /// \p Function is the function that contains the given address range.
  void addAddressRange(BinaryFunction &Function,
                       uint64_t BeginAddress,
                       uint64_t EndAddress,
                       const BinaryData *Data = nullptr);

  /// Returns the list of absolute addresses calculated using the output address
  /// of the basic blocks, i.e. the input ranges updated after basic block
  /// addresses might have changed, together with the data associated to them.
  std::vector<AbsoluteRange> getAbsoluteAddressRanges() const;

private:
  /// An address range inside one basic block.
  struct BBAddressRange {
    const BinaryBasicBlock *BasicBlock;
    /// Beginning of the range counting from BB's start address.
    uint16_t RangeBeginOffset;
    /// (Exclusive) end of the range counting from BB's start address.
    uint16_t RangeEndOffset;
    /// Binary data associated with this range.
    const BinaryData *Data;
  };

  std::vector<BBAddressRange> AddressRanges;
};

} // namespace bolt
} // namespace llvm

#endif
