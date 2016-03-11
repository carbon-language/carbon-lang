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

  /// Writes .debug_aranges with the added ranges to the MCObjectWriter.
  void Write(MCObjectWriter *Writer) const;

private:
  // Map from compile unit offset to the list of address intervals that belong
  // to that compile unit. Each interval is a pair
  // (first address, interval size).
  std::map<uint32_t, std::vector<std::pair<uint64_t, uint64_t>>> CUAddressRanges;
};

} // namespace bolt
} // namespace llvm

#endif
