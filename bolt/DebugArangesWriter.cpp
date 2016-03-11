//===--- DebugArangesWriter.h - Writes the .debug_aranges DWARF section ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DebugArangesWriter.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCObjectWriter.h"


namespace llvm {
namespace bolt {

void DebugArangesWriter::AddRange(uint32_t CompileUnitOffset,
                                  uint64_t Address,
                                  uint64_t Size) {
  CUAddressRanges[CompileUnitOffset].push_back(std::make_pair(Address, Size));
}

void DebugArangesWriter::Write(MCObjectWriter *Writer) const {
  // For reference on the format of the .debug_aranges section, see the DWARF4
  // specification, section 6.1.4 Lookup by Address
  // http://www.dwarfstd.org/doc/DWARF4.pdf
  for (const auto &CUOffsetAddressRangesPair : CUAddressRanges) {
    uint64_t Offset = CUOffsetAddressRangesPair.first;
    const auto &AddressRanges = CUOffsetAddressRangesPair.second;

    // Emit header.

    // Size of this set: 8 (size of the header) + 4 (padding after header)
    // + 2*sizeof(uint64_t) bytes for each of the ranges, plus an extra
    // pair of uint64_t's for the terminating, zero-length range.
    // Does not include size field itself.
    uint64_t Size = 8 + 4 + 2*sizeof(uint64_t) * (AddressRanges.size() + 1);

    // Header field #1: set size.
    Writer->writeLE32(Size);

    // Header field #2: version number, 2 as per the specification.
    Writer->writeLE16(2);

    // Header field #3: debug info offset of the correspondent compile unit.
    Writer->writeLE32(Offset);

    // Header field #4: address size.
    // 8 since we only write ELF64 binaries for now.
    Writer->write8(8);

    // Header field #5: segment size of target architecture.
    Writer->write8(0);

    // Padding before address table - 4 bytes in the 64-bit-pointer case.
    Writer->writeLE32(0);

    // Emit address ranges.
    for (const auto &Range : AddressRanges) {
      Writer->writeLE64(Range.first);
      Writer->writeLE64(Range.second);
    }

    // Emit terminating address range (offset 0, length 0).
    Writer->writeLE64(0);
    Writer->writeLE64(0);
  }
}

} // namespace bolt
} // namespace llvm
