//===-- DebugLocWriter.cpp - Writes the DWARF .debug_loc section. ----------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "DebugLocWriter.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCObjectWriter.h"
#include <algorithm>

namespace llvm {
namespace bolt {

void DebugLocWriter::write(const LocationList &LocList,
                           MCObjectWriter *Writer) {
  // Reference: DWARF 4 specification section 7.7.3.
  UpdatedOffsets[LocList.getOriginalOffset()] = SectionOffset;
  auto AbsoluteRanges = LocList.getAbsoluteAddressRanges();

  for (const auto &Entry : LocList.getAbsoluteAddressRanges()) {
    Writer->writeLE64(Entry.Begin);
    Writer->writeLE64(Entry.End);
    assert(Entry.Data && "Entry with null location expression.");
    Writer->writeLE16(Entry.Data->size());

    // Need to convert binary data from unsigned char to char.
    Writer->writeBytes(
        StringRef(reinterpret_cast<const char *>(Entry.Data->data()),
                  Entry.Data->size()));

    SectionOffset += 2 * 8 + 2 + Entry.Data->size();
  }
  Writer->writeLE64(0);
  Writer->writeLE64(0);
  SectionOffset += 2 * 8;
}

} // namespace bolt
} // namespace llvm
