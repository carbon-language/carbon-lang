//===--- LexicalBlock.h - DWARF lexical blocks ----------------------------===//
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

#ifndef LLVM_TOOLS_LLVM_BOLT_LEXICAL_BLOCK_H
#define LLVM_TOOLS_LLVM_BOLT_LEXICAL_BLOCK_H

#include "DebugRangesSectionsWriter.h"
#include "BasicBlockOffsetRanges.h"

namespace llvm {

class DWARFCompileUnit;
class DWARFDebugInfoEntryMinimal;

namespace bolt {

class BasicBlockTable;
class BinaryBasicBlock;
class BinaryFunction;

class LexicalBlock : public AddressRangesOwner {
public:
  LexicalBlock(const DWARFCompileUnit *CU,
               const DWARFDebugInfoEntryMinimal *DIE)
      : CU(CU), DIE(DIE) { }

  // Add range [BeginAddress, EndAddress) to lexical block.
  void addAddressRange(BinaryFunction &Function,
                       uint64_t BeginAddress,
                       uint64_t EndAddress) {
    BBOffsetRanges.addAddressRange(Function, BeginAddress, EndAddress);
  }

  std::vector<std::pair<uint64_t, uint64_t>> getAbsoluteAddressRanges() const {
    return BBOffsetRanges.getAbsoluteAddressRanges();
  }

  void setAddressRangesOffset(uint32_t Offset) { AddressRangesOffset = Offset; }

  uint32_t getAddressRangesOffset() const { return AddressRangesOffset; }

  const DWARFCompileUnit *getCompileUnit() const { return CU; }
  const DWARFDebugInfoEntryMinimal *getDIE() const { return DIE; }

private:
  const DWARFCompileUnit *CU;
  const DWARFDebugInfoEntryMinimal *DIE;

  BasicBlockOffsetRanges BBOffsetRanges;

  // Offset of the address ranges of this block in the output .debug_ranges.
  uint32_t AddressRangesOffset;
};

} // namespace bolt
} // namespace llvm

#endif
