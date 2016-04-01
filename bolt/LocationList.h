//===--- LocationList.h - DWARF location lists ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Represents DWARF location lists, maintaining their list of location
// expressions and the address ranges in which they are valid to be updated in
// the output debugging information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_LOCATION_LIST_H
#define LLVM_TOOLS_LLVM_BOLT_LOCATION_LIST_H

#include "BasicBlockOffsetRanges.h"

namespace llvm {

class DWARFCompileUnit;
class DWARFDebugInfoEntryMinimal;

namespace bolt {

class BinaryBasicBlock;

class LocationList {
public:
  LocationList(uint32_t Offset) : DebugLocOffset(Offset) { }

  /// Add a location expression that is valid in [BeginAddress, EndAddress)
  /// within Function to location list.
  void addLocation(const BasicBlockOffsetRanges::BinaryData *Expression,
                   BinaryFunction &Function,
                   uint64_t BeginAddress,
                   uint64_t EndAddress) {
    BBOffsetRanges.addAddressRange(Function, BeginAddress, EndAddress,
                                   Expression);
  }

  std::vector<BasicBlockOffsetRanges::AbsoluteRange>
  getAbsoluteAddressRanges() const {
    return BBOffsetRanges.getAbsoluteAddressRanges();
  }

  uint32_t getOriginalOffset() const { return DebugLocOffset; }

private:
  BasicBlockOffsetRanges BBOffsetRanges;

  // Offset of this location list in the input .debug_loc section.
  uint32_t DebugLocOffset;
};

} // namespace bolt
} // namespace llvm

#endif
