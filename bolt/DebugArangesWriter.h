//===--- DebugArangesWriter.h - Writes the .debug_aranges DWARF section ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Class that serializes a .debug_aranges section of a binary.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_DEBUGARANGESWRITER_H
#define LLVM_TOOLS_LLVM_BOLT_DEBUGARANGESWRITER_H

#include <map>
#include <vector>
#include <utility>

namespace llvm {

class MCObjectWriter;

namespace bolt {

class DebugArangesWriter {
public:
  DebugArangesWriter() = default;

  /// Adds a range to the .debug_arange section.
  void AddRange(uint32_t CompileUnitOffset, uint64_t Address, uint64_t Size);

  using RangesCUMapType = std::map<uint32_t, uint32_t>;

  /// Writes .debug_aranges with the added ranges to the MCObjectWriter.
  void WriteArangesSection(MCObjectWriter *Writer) const;

  /// Writes .debug_ranges with the added ranges to the MCObjectWriter.
  void WriteRangesSection(MCObjectWriter *Writer);

  /// Return mapping of CUs to offsets in .debug_ranges.
  const RangesCUMapType &getRangesOffsetCUMap() const {
    return RangesSectionOffsetCUMap;
  }

private:
  // Map from compile unit offset to the list of address intervals that belong
  // to that compile unit. Each interval is a pair
  // (first address, interval size).
  std::map<uint32_t, std::vector<std::pair<uint64_t, uint64_t>>> CUAddressRanges;

  /// When writing data to .debug_ranges remember offset per CU.
  RangesCUMapType RangesSectionOffsetCUMap;
};

} // namespace bolt
} // namespace llvm

#endif
