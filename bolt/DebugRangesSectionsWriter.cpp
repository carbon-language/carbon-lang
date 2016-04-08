//===-- DebugRangesSectionsWriter.h - Writes DWARF address ranges sections -==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DebugRangesSectionsWriter.h"
#include "BinaryFunction.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCObjectWriter.h"

namespace llvm {
namespace bolt {

void DebugRangesSectionsWriter::AddRange(uint32_t CompileUnitOffset,
                                         uint64_t Address,
                                         uint64_t Size) {
  CUAddressRanges[CompileUnitOffset].push_back(std::make_pair(Address, Size));
}

void DebugRangesSectionsWriter::AddRange(AddressRangesOwner *BF,
                                         uint64_t Address,
                                         uint64_t Size) {
  ObjectAddressRanges[BF].push_back(std::make_pair(Address, Size));
}

namespace {

// Writes address ranges to Writer as pairs of 64-bit (address, size).
// If RelativeRange is true, assumes the address range to be written must be of
// the form (begin address, range size), otherwise (begin address, end address).
// Terminates the list by writing a pair of two zeroes.
// Returns the number of written bytes.
uint32_t WriteAddressRanges(
    MCObjectWriter *Writer,
    const std::vector<std::pair<uint64_t, uint64_t>> &AddressRanges,
    bool RelativeRange) {
  // Write entries.
  for (auto &Range : AddressRanges) {
    Writer->writeLE64(Range.first);
    Writer->writeLE64((!RelativeRange) * Range.first + Range.second);
  }
  // Finish with 0 entries.
  Writer->writeLE64(0);
  Writer->writeLE64(0);
  return AddressRanges.size() * 16 + 16;
}

} // namespace

void DebugRangesSectionsWriter::WriteRangesSection(MCObjectWriter *Writer) {
  uint32_t SectionOffset = 0;
  for (const auto &CUOffsetAddressRangesPair : CUAddressRanges) {
    uint64_t CUOffset = CUOffsetAddressRangesPair.first;
    RangesSectionOffsetCUMap[CUOffset] = SectionOffset;
    const auto &AddressRanges = CUOffsetAddressRangesPair.second;
    SectionOffset += WriteAddressRanges(Writer, AddressRanges, false);
  }

  for (const auto &BFAddressRangesPair : ObjectAddressRanges) {
    BFAddressRangesPair.first->setAddressRangesOffset(SectionOffset);
    const auto &AddressRanges = BFAddressRangesPair.second;
    SectionOffset += WriteAddressRanges(Writer, AddressRanges, false);
  }

  // Write an empty address list to be used for objects with unknown address
  // ranges.
  EmptyRangesListOffset = SectionOffset;
  SectionOffset += WriteAddressRanges(
      Writer,
      std::vector<std::pair<uint64_t, uint64_t>>{},
      false);
}

void
DebugRangesSectionsWriter::WriteArangesSection(MCObjectWriter *Writer) const {
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

    WriteAddressRanges(Writer, AddressRanges, true);
  }
}

} // namespace bolt
} // namespace llvm
