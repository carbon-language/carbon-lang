//===-- DWARFDebugRangeList.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARFDEBUGRANGELIST_H
#define LLVM_DEBUGINFO_DWARFDEBUGRANGELIST_H

#include "llvm/Support/DataExtractor.h"
#include <vector>

namespace llvm {

class raw_ostream;

class DWARFDebugRangeList {
public:
  struct RangeListEntry {
    // A beginning address offset. This address offset has the size of an
    // address and is relative to the applicable base address of the
    // compilation unit referencing this range list. It marks the beginning
    // of an address range.
    uint64_t StartAddress;
    // An ending address offset. This address offset again has the size of
    // an address and is relative to the applicable base address of the
    // compilation unit referencing this range list. It marks the first
    // address past the end of the address range. The ending address must
    // be greater than or equal to the beginning address.
    uint64_t EndAddress;
  };

private:
  // Offset in .debug_ranges section.
  uint32_t Offset;
  uint8_t AddressSize;
  std::vector<RangeListEntry> Entries;

public:
  DWARFDebugRangeList() { clear(); }
  void clear();
  void dump(raw_ostream &OS) const;
  bool extract(DataExtractor data, uint32_t *offset_ptr);
};

}  // namespace llvm

#endif  // LLVM_DEBUGINFO_DWARFDEBUGRANGELIST_H
