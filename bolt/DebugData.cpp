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
#include <algorithm>
#include <cassert>

namespace llvm {
namespace bolt {

const DebugLineTableRowRef DebugLineTableRowRef::NULL_ROW{0, 0};

void BasicBlockOffsetRanges::addAddressRange(BinaryFunction &Function,
                                             uint64_t BeginAddress,
                                             uint64_t EndAddress,
                                             const BinaryData *Data) {
  auto FirstBB = Function.getBasicBlockContainingOffset(
      BeginAddress - Function.getAddress());
  if (!FirstBB) {
    errs() << "BOLT-WARNING: no basic blocks in function "
           << Function.getName() << " intersect with debug range [0x"
           << Twine::utohexstr(BeginAddress) << ", 0x"
           << Twine::utohexstr(EndAddress) << ")\n";
    return;
  }

  for (auto I = Function.getIndex(FirstBB), S = Function.size(); I != S; ++I) {
    auto BB = Function.getBasicBlockAtIndex(I);
    uint64_t BBAddress = Function.getAddress() + BB->getOffset();
    // Note the special handling for [a, a) address range.
    if (BBAddress >= EndAddress && BeginAddress != EndAddress)
      break;

    uint64_t InternalAddressRangeBegin = std::max(BBAddress, BeginAddress);
    assert(BB->getFunction() == &Function &&
           "Mismatching functions.\n");
    uint64_t InternalAddressRangeEnd =
      std::min(BBAddress + Function.getBasicBlockOriginalSize(BB),
               EndAddress);

    AddressRanges.push_back(
        BBAddressRange{
            BB,
            static_cast<uint16_t>(InternalAddressRangeBegin - BBAddress),
            static_cast<uint16_t>(InternalAddressRangeEnd - BBAddress),
            Data});
  }
}

std::vector<BasicBlockOffsetRanges::AbsoluteRange>
BasicBlockOffsetRanges::getAbsoluteAddressRanges() const {
  std::vector<AbsoluteRange> AbsoluteRanges;
  for (const auto &BBAddressRange : AddressRanges) {
    auto BBOutputAddressRange =
        BBAddressRange.BasicBlock->getOutputAddressRange();
    uint64_t NewRangeBegin = BBOutputAddressRange.first +
        BBAddressRange.RangeBeginOffset;
    // If the end offset pointed to the end of the basic block, then we set
    // the new end range to cover the whole basic block as the BB's size
    // might have increased.
    auto BBFunction = BBAddressRange.BasicBlock->getFunction();
    uint64_t NewRangeEnd =
        (BBAddressRange.RangeEndOffset ==
         BBFunction->getBasicBlockOriginalSize(BBAddressRange.BasicBlock))
        ? BBOutputAddressRange.second
        : (BBOutputAddressRange.first + BBAddressRange.RangeEndOffset);
    AbsoluteRanges.emplace_back(AbsoluteRange{NewRangeBegin, NewRangeEnd,
                                              BBAddressRange.Data});
  }
  if (AbsoluteRanges.empty()) {
    return AbsoluteRanges;
  }
  // Merge adjacent ranges that have the same data.
  std::sort(AbsoluteRanges.begin(), AbsoluteRanges.end(),
            [](const AbsoluteRange &A, const AbsoluteRange &B) {
                return A.Begin < B.Begin;
            });
  decltype(AbsoluteRanges) MergedRanges;

  MergedRanges.emplace_back(AbsoluteRanges[0]);
  for (unsigned I = 1, S = AbsoluteRanges.size(); I != S; ++I) {
    // If this range complements the last one and they point to the same
    // (possibly null) data, merge them instead of creating another one.
    if (AbsoluteRanges[I].Begin == MergedRanges.back().End &&
        AbsoluteRanges[I].Data == MergedRanges.back().Data) {
      MergedRanges.back().End = AbsoluteRanges[I].End;
    } else {
      MergedRanges.emplace_back(AbsoluteRanges[I]);
    }
  }

  return MergedRanges;
}

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
                                           uint16_t AttrTag,
                                           uint8_t NewAttrTag,
                                           uint8_t NewAttrForm) {
  assert(Unit && "No compile unit specified.");
  Patches[Unit].push_back(
      AbbrevAttrPatch{AbbrevCode, AttrTag, NewAttrTag, NewAttrForm});
}

void DebugAbbrevPatcher::patchBinary(std::string &Contents) {
  SimpleBinaryPatcher Patcher;

  for (const auto &UnitPatchesPair : Patches) {
    const auto *Unit = UnitPatchesPair.first;
    const auto *UnitAbbreviations = Unit->getAbbreviations();
    assert(UnitAbbreviations &&
           "Compile unit doesn't have associated abbreviations.");
    const auto &UnitPatches = UnitPatchesPair.second;
    for (const auto &AttrPatch : UnitPatches) {
      const auto *AbbreviationDeclaration =
        UnitAbbreviations->getAbbreviationDeclaration(AttrPatch.Code);
      assert(AbbreviationDeclaration && "No abbreviation with given code.");
      const auto *Attribute = AbbreviationDeclaration->findAttribute(
          AttrPatch.Attr);

      assert(Attribute && "Specified attribute doesn't occur in abbreviation.");
      // Because we're only handling standard values (i.e. no DW_FORM_GNU_* or
      // DW_AT_APPLE_*), they are all small (< 128) and encoded in a single
      // byte in ULEB128, otherwise it'll be more tricky as we may need to
      // grow or shrink the section.
      Patcher.addBytePatch(Attribute->AttrOffset,
          AttrPatch.NewAttr);
      Patcher.addBytePatch(Attribute->FormOffset,
          AttrPatch.NewForm);
    }
  }
  Patcher.patchBinary(Contents);
}



} // namespace bolt
} // namespace llvm
