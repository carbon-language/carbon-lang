//===-- DebugRangesSectionsWriter.h - Writes DWARF address ranges sections -==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Class that serializes the .debug_ranges and .debug_aranges sections.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_DEBUG_RANGES_SECTIONS_WRITER_H
#define LLVM_TOOLS_LLVM_BOLT_DEBUG_RANGES_SECTIONS_WRITER_H

#include <map>
#include <vector>
#include <utility>

namespace llvm {

class MCObjectWriter;

namespace bolt {

/// Abstract interface for classes that represent objects that have
/// associated address ranges in .debug_ranges. These address ranges can
/// be serialized by DebugRangesSectionsWriter which notifies the object
/// of where in the section its address ranges list was written.
class AddressRangesOwner {
public:
  virtual void setAddressRangesOffset(uint32_t Offset) = 0;
};

class DebugRangesSectionsWriter {
public:
  DebugRangesSectionsWriter() = default;

  /// Adds a range to the .debug_arange section.
  void AddRange(uint32_t CompileUnitOffset, uint64_t Address, uint64_t Size);

  /// Adds an address range that belongs to a given object.
  /// When .debug_ranges is written, the offset of the range corresponding
  /// to the function will be set using BF->setAddressRangesOffset().
  void AddRange(AddressRangesOwner *ARO, uint64_t Address, uint64_t Size);

  using RangesCUMapType = std::map<uint32_t, uint32_t>;

  /// Writes .debug_aranges with the added ranges to the MCObjectWriter.
  void WriteArangesSection(MCObjectWriter *Writer) const;

  /// Writes .debug_ranges with the added ranges to the MCObjectWriter.
  void WriteRangesSection(MCObjectWriter *Writer);

  /// Resets the writer to a clear state.
  void reset() {
    CUAddressRanges.clear();
    ObjectAddressRanges.clear();
    RangesSectionOffsetCUMap.clear();
  }

  /// Return mapping of CUs to offsets in .debug_ranges.
  const RangesCUMapType &getRangesOffsetCUMap() const {
    return RangesSectionOffsetCUMap;
  }

  /// Returns an offset of an empty address ranges list that is always written
  /// to .debug_ranges
  uint32_t getEmptyRangesListOffset() const { return EmptyRangesListOffset; }

private:
  // Map from compile unit offset to the list of address intervals that belong
  // to that compile unit. Each interval is a pair
  // (first address, interval size).
  std::map<uint32_t, std::vector<std::pair<uint64_t, uint64_t>>>
      CUAddressRanges;

  // Map from BinaryFunction to the list of address intervals that belong
  // to that function, represented like CUAddressRanges.
  std::map<AddressRangesOwner *, std::vector<std::pair<uint64_t, uint64_t>>>
      ObjectAddressRanges;

  // Offset of an empty address ranges list.
  uint32_t EmptyRangesListOffset;

  /// When writing data to .debug_ranges remember offset per CU.
  RangesCUMapType RangesSectionOffsetCUMap;
};

} // namespace bolt
} // namespace llvm

#endif
