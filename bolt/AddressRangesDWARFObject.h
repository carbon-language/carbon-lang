//===--- AddressRangesDWARFObject.h - DWARF Entities with address ranges --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Represents DWARF lexical blocks, maintaining their list of address ranges to
// be updated in the output debugging information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_ADDRESS_RANGES_DWARF_OBJECT_H
#define LLVM_TOOLS_LLVM_BOLT_ADDRESS_RANGES_DWARF_OBJECT_H

#include "DebugRangesSectionsWriter.h"
#include "BasicBlockOffsetRanges.h"

namespace llvm {

class DWARFCompileUnit;
class DWARFDebugInfoEntryMinimal;

namespace bolt {

class BasicBlockTable;
class BinaryBasicBlock;
class BinaryFunction;

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

  std::vector<std::pair<uint64_t, uint64_t>> getAbsoluteAddressRanges() const {
    auto AddressRangesWithData = BBOffsetRanges.getAbsoluteAddressRanges();
    std::vector<std::pair<uint64_t, uint64_t>> AddressRanges(
        AddressRangesWithData.size());
    for (unsigned I = 0, S = AddressRanges.size(); I != S; ++I) {
      AddressRanges[I] = std::make_pair(AddressRangesWithData[I].Begin,
                                        AddressRangesWithData[I].End);
    }
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

  // Offset of the address ranges of this object in the output .debug_ranges.
  uint32_t AddressRangesOffset;
};

} // namespace bolt
} // namespace llvm

#endif
