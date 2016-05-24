//===-- DebugData.h - Representation and writing of debugging information. -==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Classes that represent and serialize DWARF-related entities.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_DEBUG_DATA_H
#define LLVM_TOOLS_LLVM_BOLT_DEBUG_DATA_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/Support/SMLoc.h"
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace llvm {

class DWARFCompileUnit;
class DWARFDebugInfoEntryMinimal;
class MCObjectWriter;

namespace bolt {

class BasicBlockTable;
class BinaryBasicBlock;
class BinaryFunction;

/// Eeferences a row in a DWARFDebugLine::LineTable by the DWARF
/// Context index of the DWARF Compile Unit that owns the Line Table and the row
/// index. This is tied to our IR during disassembly so that we can later update
/// .debug_line information. RowIndex has a base of 1, which means a RowIndex
/// of 1 maps to the first row of the line table and a RowIndex of 0 is invalid.
struct DebugLineTableRowRef {
  uint32_t DwCompileUnitIndex;
  uint32_t RowIndex;

  const static DebugLineTableRowRef NULL_ROW;

  bool operator==(const DebugLineTableRowRef &Rhs) const {
    return DwCompileUnitIndex == Rhs.DwCompileUnitIndex &&
      RowIndex == Rhs.RowIndex;
  }

  bool operator!=(const DebugLineTableRowRef &Rhs) const {
    return !(*this == Rhs);
  }

  static DebugLineTableRowRef fromSMLoc(const SMLoc &Loc) {
    union {
      decltype(Loc.getPointer()) Ptr;
      DebugLineTableRowRef Ref;
    } U;
    U.Ptr = Loc.getPointer();
    return U.Ref;
  }

  SMLoc toSMLoc() const {
    union {
      decltype(SMLoc().getPointer()) Ptr;
      DebugLineTableRowRef Ref;
    } U;
    U.Ref = *this;
    return SMLoc::getFromPointer(U.Ptr);
  }
};

/// Represents a list of address ranges where addresses are relative to the
/// beginning of basic blocks. Useful for converting address ranges in the input
/// binary to equivalent ranges after optimizations take place.
class BasicBlockOffsetRanges {
public:
  typedef SmallVectorImpl<unsigned char> BinaryData;
  struct AbsoluteRange {
    uint64_t Begin;
    uint64_t End;
    const BinaryData *Data;
  };

  /// Add range [BeginAddress, EndAddress) to the address ranges list.
  /// \p Function is the function that contains the given address range.
  void addAddressRange(BinaryFunction &Function,
                       uint64_t BeginAddress,
                       uint64_t EndAddress,
                       const BinaryData *Data = nullptr);

  /// Returns the list of absolute addresses calculated using the output address
  /// of the basic blocks, i.e. the input ranges updated after basic block
  /// addresses might have changed, together with the data associated to them.
  std::vector<AbsoluteRange> getAbsoluteAddressRanges() const;

private:
  /// An address range inside one basic block.
  struct BBAddressRange {
    const BinaryBasicBlock *BasicBlock;
    /// Beginning of the range counting from BB's start address.
    uint16_t RangeBeginOffset;
    /// (Exclusive) end of the range counting from BB's start address.
    uint16_t RangeEndOffset;
    /// Binary data associated with this range.
    const BinaryData *Data;
  };

  std::vector<BBAddressRange> AddressRanges;
};

/// Abstract interface for classes that represent objects that have
/// associated address ranges in .debug_ranges. These address ranges can
/// be serialized by DebugRangesSectionsWriter which notifies the object
/// of where in the section its address ranges list was written.
class AddressRangesOwner {
public:
  virtual void setAddressRangesOffset(uint32_t Offset) = 0;
};

/// Represents DWARF entities that have generic address ranges, maintaining
/// their address ranges to be updated on the output debugging information.
class AddressRangesDWARFObject : public AddressRangesOwner {
public:
  AddressRangesDWARFObject(const DWARFCompileUnit *CU,
                           const DWARFDebugInfoEntryMinimal *DIE)
      : CU(CU), DIE(DIE) { }

  /// Add range [BeginAddress, EndAddress) to this object.
  void addAddressRange(BinaryFunction &Function,
                       uint64_t BeginAddress,
                       uint64_t EndAddress) {
    BBOffsetRanges.addAddressRange(Function, BeginAddress, EndAddress);
  }

  /// Add range that is guaranteed to not change.
  void addAbsoluteRange(uint64_t BeginAddress,
                        uint64_t EndAddress) {
    AbsoluteRanges.emplace_back(std::make_pair(BeginAddress, EndAddress));
  }

  std::vector<std::pair<uint64_t, uint64_t>> getAbsoluteAddressRanges() const {
    auto AddressRangesWithData = BBOffsetRanges.getAbsoluteAddressRanges();
    std::vector<std::pair<uint64_t, uint64_t>>
        AddressRanges(AddressRangesWithData.size());
    for (unsigned I = 0, S = AddressRanges.size(); I != S; ++I) {
      AddressRanges[I] = std::make_pair(AddressRangesWithData[I].Begin,
                                        AddressRangesWithData[I].End);
    }
    std::move(AbsoluteRanges.begin(),
              AbsoluteRanges.end(),
              std::back_inserter(AddressRanges));
    return AddressRanges;
  }

  void setAddressRangesOffset(uint32_t Offset) { AddressRangesOffset = Offset; }

  uint32_t getAddressRangesOffset() const { return AddressRangesOffset; }

  const DWARFCompileUnit *getCompileUnit() const { return CU; }
  const DWARFDebugInfoEntryMinimal *getDIE() const { return DIE; }

private:
  const DWARFCompileUnit *CU;
  const DWARFDebugInfoEntryMinimal *DIE;

  BasicBlockOffsetRanges BBOffsetRanges;

  std::vector<std::pair<uint64_t, uint64_t>> AbsoluteRanges;

  /// Offset of the address ranges of this object in the output .debug_ranges.
  uint32_t AddressRangesOffset{-1U};
};



/// Represents DWARF location lists, maintaining their list of location
/// expressions and the address ranges in which they are valid to be updated in
/// the output debugging information.
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

  /// Offset of this location list in the input .debug_loc section.
  uint32_t DebugLocOffset;
};

/// Serializes the .debug_ranges and .debug_aranges DWARF sections.
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

  using CUAddressRangesType =
    std::map<uint32_t, std::vector<std::pair<uint64_t, uint64_t>>>;

  /// Return ranges for a given CU.
  const CUAddressRangesType &getCUAddressRanges() const {
    return CUAddressRanges;
  }

private:
  /// Map from compile unit offset to the list of address intervals that belong
  /// to that compile unit. Each interval is a pair
  /// (first address, interval size).
  CUAddressRangesType CUAddressRanges;

  /// Map from BinaryFunction to the list of address intervals that belong
  /// to that function, represented like CUAddressRanges.
  std::map<AddressRangesOwner *, std::vector<std::pair<uint64_t, uint64_t>>>
      ObjectAddressRanges;

  /// Offset of an empty address ranges list.
  uint32_t EmptyRangesListOffset;

  /// When writing data to .debug_ranges remember offset per CU.
  RangesCUMapType RangesSectionOffsetCUMap;
};

/// Serializes the .debug_loc DWARF section with LocationLists.
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

/// Abstract interface for classes that apply modifications to a binary string.
class BinaryPatcher {
public:
  virtual ~BinaryPatcher() {}
  /// Applies in-place modifications to the binary string \p BinaryContents .
  virtual void patchBinary(std::string &BinaryContents) = 0;
};

/// Applies simple modifications to a binary string, such as directly replacing
/// the contents of a certain portion with a string or an integer.
class SimpleBinaryPatcher : public BinaryPatcher {
private:
  std::vector<std::pair<uint32_t, std::string>> Patches;

  /// Adds a patch to replace the contents of \p ByteSize bytes with the integer
  /// \p NewValue encoded in little-endian, with the least-significant byte
  /// being written at the offset \p Offset .
  void addLEPatch(uint32_t Offset, uint64_t NewValue, size_t ByteSize);

public:
  ~SimpleBinaryPatcher() {}

  /// Adds a patch to replace the contents of the binary string starting at the
  /// specified \p Offset with the string \p NewValue.
  void addBinaryPatch(uint32_t Offset, const std::string &NewValue);

  /// Adds a patch to replace the contents of a single byte of the string, at
  /// the offset \p Offset, with the value \Value .
  void addBytePatch(uint32_t Offset, uint8_t Value);

  /// Adds a patch to put the integer \p NewValue encoded as a 64-bit
  /// little-endian value at offset \p Offset.
  void addLE64Patch(uint32_t Offset, uint64_t NewValue);

  /// Adds a patch to put the integer \p NewValue encoded as a 32-bit
  /// little-endian value at offset \p Offset.
  void addLE32Patch(uint32_t Offset, uint32_t NewValue);

  void patchBinary(std::string &BinaryContents) override;
};

/// Apply small modifications to the .debug_abbrev DWARF section.
class DebugAbbrevPatcher : public BinaryPatcher {
private:
  /// Patch of changing one attribute to another.
  struct AbbrevAttrPatch {
    uint32_t Code;    // Code of abbreviation to be modified.
    uint16_t Attr;    // ID of attribute to be replaced.
    uint8_t NewAttr;  // ID of the new attribute.
    uint8_t NewForm;  // Form of the new attribute.
  };

  std::map<const DWARFUnit *, std::vector<AbbrevAttrPatch>> Patches;

public:
  ~DebugAbbrevPatcher() { }
  /// Adds a patch to change an attribute of an abbreviation that belongs to
  /// \p Unit to another attribute.
  /// \p AbbrevCode code of the abbreviation to be modified.
  /// \p AttrTag ID of the attribute to be replaced.
  /// \p NewAttrTag ID of the new attribute.
  /// \p NewAttrForm Form of the new attribute.
  /// We only handle standard forms, that are encoded in a single byte.
  void addAttributePatch(const DWARFUnit *Unit,
                         uint32_t AbbrevCode,
                         uint16_t AttrTag,
                         uint8_t NewAttrTag,
                         uint8_t NewAttrForm);

  void patchBinary(std::string &Contents) override;
};

} // namespace bolt
} // namespace llvm

#endif
