//===-- DebugLocWriter.h - Writes the DWARF .debug_loc section -------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Class that serializes the .debug_loc section given LocationLists.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_DEBUG_LOC_WRITER_H
#define LLVM_TOOLS_LLVM_BOLT_DEBUG_LOC_WRITER_H

#include "LocationList.h"
#include <map>
#include <vector>

namespace llvm {

class MCObjectWriter;

namespace bolt {

class DebugLocWriter {
public:
  /// Writes the given location list to the writer.
  void write(const LocationList &LocList, MCObjectWriter *Writer);

  using UpdatedOffsetMapType = std::map<uint32_t, uint32_t>;

  /// Returns mapping from offsets in the input .debug_loc to offsets in the
  /// output .debug_loc section with the corresponding updated location list
  /// entry.
  const UpdatedOffsetMapType &getUpdatedLocationListOffsets() const {
    return UpdatedOffsets;
  }

private:
  /// Current offset in the section (updated as new entries are written).
  uint32_t SectionOffset{0};

  /// Map from input offsets to output offsets for location lists that were
  /// updated, generated after write().
  UpdatedOffsetMapType UpdatedOffsets;
};

} // namespace bolt
} // namespace llvm

#endif
