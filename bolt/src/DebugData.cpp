//===- DebugData.cpp - Representation and writing of debugging information. ==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "DebugData.h"
#include "BinaryBasicBlock.h"
#include "BinaryFunction.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/LEB128.h"
#include <algorithm>
#include <cassert>

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt-debug-info"

namespace opts {
extern llvm::cl::opt<unsigned> Verbosity;
}

namespace llvm {
namespace bolt {

const DebugLineTableRowRef DebugLineTableRowRef::NULL_ROW{0, 0};

namespace {

// Writes address ranges to Writer as pairs of 64-bit (address, size).
// If RelativeRange is true, assumes the address range to be written must be of
// the form (begin address, range size), otherwise (begin address, end address).
// Terminates the list by writing a pair of two zeroes.
// Returns the number of written bytes.
uint64_t writeAddressRanges(
    MCObjectWriter *Writer,
    const DebugAddressRangesVector &AddressRanges,
    const bool WriteRelativeRanges = false) {
  for (auto &Range : AddressRanges) {
    Writer->writeLE64(Range.LowPC);
    Writer->writeLE64(WriteRelativeRanges ? Range.HighPC - Range.LowPC
                                          : Range.HighPC);
  }
  // Finish with 0 entries.
  Writer->writeLE64(0);
  Writer->writeLE64(0);
  return AddressRanges.size() * 16 + 16;
}

} // namespace

DebugRangesSectionsWriter::DebugRangesSectionsWriter(BinaryContext *BC) {
  RangesBuffer = llvm::make_unique<SmallVector<char, 16>>();
  RangesStream = llvm::make_unique<raw_svector_ostream>(*RangesBuffer);
  Writer =
    std::unique_ptr<MCObjectWriter>(BC->createObjectWriter(*RangesStream));

  // Add an empty range as the first entry;
  SectionOffset += writeAddressRanges(Writer.get(), DebugAddressRangesVector{});
}

uint64_t DebugRangesSectionsWriter::addCURanges(
    uint64_t CUOffset,
    DebugAddressRangesVector &&Ranges) {
  const auto RangesOffset = addRanges(Ranges);
  CUAddressRanges.emplace(CUOffset, std::move(Ranges));

  return RangesOffset;
}

uint64_t
DebugRangesSectionsWriter::addRanges(const BinaryFunction *Function,
                                     DebugAddressRangesVector &&Ranges) {
  if (Ranges.empty())
    return getEmptyRangesOffset();

  static const BinaryFunction *CachedFunction;

  if (Function == CachedFunction) {
    const auto RI = CachedRanges.find(Ranges);
    if (RI != CachedRanges.end())
      return RI->second;
  } else {
    CachedRanges.clear();
    CachedFunction = Function;
  }

  const auto EntryOffset = addRanges(Ranges);
  CachedRanges.emplace(std::move(Ranges), EntryOffset);

  return EntryOffset;
}

uint64_t
DebugRangesSectionsWriter::addRanges(const DebugAddressRangesVector &Ranges) {
  if (Ranges.empty())
    return getEmptyRangesOffset();

  const auto EntryOffset = SectionOffset;
  SectionOffset += writeAddressRanges(Writer.get(), Ranges);

  return EntryOffset;
}

void
DebugRangesSectionsWriter::writeArangesSection(MCObjectWriter *Writer) const {
  // For reference on the format of the .debug_aranges section, see the DWARF4
  // specification, section 6.1.4 Lookup by Address
  // http://www.dwarfstd.org/doc/DWARF4.pdf
  for (const auto &CUOffsetAddressRangesPair : CUAddressRanges) {
    const auto Offset = CUOffsetAddressRangesPair.first;
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

    writeAddressRanges(Writer, AddressRanges, true);
  }
}

DebugLocWriter::DebugLocWriter(BinaryContext *BC) {
  LocBuffer = llvm::make_unique<SmallVector<char, 16>>();
  LocStream = llvm::make_unique<raw_svector_ostream>(*LocBuffer);
  Writer =
    std::unique_ptr<MCObjectWriter>(BC->createObjectWriter(*LocStream));

  // Add an empty list as the first entry;
  Writer->writeLE64(0);
  Writer->writeLE64(0);
  SectionOffset += 2 * 8;
}

// DWARF 4: 2.6.2
uint64_t DebugLocWriter::addList(const DWARFDebugLoc::LocationList &LocList) {
  if (LocList.Entries.empty())
    return getEmptyListOffset();

  const auto EntryOffset = SectionOffset;
  for (const auto &Entry : LocList.Entries) {
    Writer->writeLE64(Entry.Begin);
    Writer->writeLE64(Entry.End);
    Writer->writeLE16(Entry.Loc.size());
    Writer->writeBytes(
        StringRef(reinterpret_cast<const char *>(Entry.Loc.data()),
                  Entry.Loc.size()));
    SectionOffset += 2 * 8 + 2 + Entry.Loc.size();
  }
  Writer->writeLE64(0);
  Writer->writeLE64(0);
  SectionOffset += 2 * 8;

  return EntryOffset;
}

void SimpleBinaryPatcher::addBinaryPatch(uint32_t Offset,
                                         const std::string &NewValue) {
  Patches.emplace_back(std::make_pair(Offset, NewValue));
}

void SimpleBinaryPatcher::addBytePatch(uint32_t Offset, uint8_t Value) {
  Patches.emplace_back(std::make_pair(Offset, std::string(1, Value)));
}

void SimpleBinaryPatcher::addLEPatch(uint32_t Offset, uint64_t NewValue,
                                     size_t ByteSize) {
  std::string LE64(ByteSize, 0);
  for (size_t I = 0; I < ByteSize; ++I) {
    LE64[I] = NewValue & 0xff;
    NewValue >>= 8;
  }
  Patches.emplace_back(std::make_pair(Offset, LE64));
}

void SimpleBinaryPatcher::addUDataPatch(uint32_t Offset, uint64_t Value, uint64_t Size) {
  std::string Buff;
  raw_string_ostream OS(Buff);
  encodeULEB128(Value, OS, Size);

  Patches.emplace_back(Offset, OS.str());
}

void SimpleBinaryPatcher::addLE64Patch(uint32_t Offset, uint64_t NewValue) {
  addLEPatch(Offset, NewValue, 8);
}

void SimpleBinaryPatcher::addLE32Patch(uint32_t Offset, uint32_t NewValue) {
  addLEPatch(Offset, NewValue, 4);
}

void SimpleBinaryPatcher::patchBinary(std::string &BinaryContents) {
  for (const auto &Patch : Patches) {
    uint32_t Offset = Patch.first;
    const std::string &ByteSequence = Patch.second;
    assert(Offset + ByteSequence.size() <= BinaryContents.size() &&
        "Applied patch runs over binary size.");
    for (uint64_t I = 0, Size = ByteSequence.size(); I < Size; ++I) {
      BinaryContents[Offset + I] = ByteSequence[I];
    }
  }
}

void DebugAbbrevPatcher::addAttributePatch(const DWARFUnit *Unit,
                                           uint32_t AbbrevCode,
                                           dwarf::Attribute AttrTag,
                                           uint8_t NewAttrTag,
                                           uint8_t NewAttrForm) {
  assert(Unit && "No compile unit specified.");
  AbbrevPatches.emplace(
      AbbrevAttrPatch{Unit, AbbrevCode, AttrTag, NewAttrTag, NewAttrForm});
}

void DebugAbbrevPatcher::patchBinary(std::string &Contents) {
  SimpleBinaryPatcher Patcher;

  for (const auto &Patch : AbbrevPatches) {
    const auto *UnitAbbreviations = Patch.Unit->getAbbreviations();
    assert(UnitAbbreviations &&
           "Compile unit doesn't have associated abbreviations.");
    const auto *AbbreviationDeclaration =
      UnitAbbreviations->getAbbreviationDeclaration(Patch.Code);
    assert(AbbreviationDeclaration && "No abbreviation with given code.");
    const auto Attribute =
        AbbreviationDeclaration->findAttribute(Patch.Attr);

    assert(Attribute && "Specified attribute doesn't occur in abbreviation.");
    // Because we're only handling standard values (i.e. no DW_FORM_GNU_* or
    // DW_AT_APPLE_*), they are all small (< 128) and encoded in a single
    // byte in ULEB128, otherwise it'll be more tricky as we may need to
    // grow or shrink the section.
    Patcher.addBytePatch(Attribute->AttrOffset, Patch.NewAttr);
    Patcher.addBytePatch(Attribute->FormOffset, Patch.NewForm);
  }
  Patcher.patchBinary(Contents);
}



} // namespace bolt
} // namespace llvm
