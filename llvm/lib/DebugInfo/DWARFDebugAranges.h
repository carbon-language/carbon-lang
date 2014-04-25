//===-- DWARFDebugAranges.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARFDEBUGARANGES_H
#define LLVM_DEBUGINFO_DWARFDEBUGARANGES_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/DataExtractor.h"

namespace llvm {

class DWARFContext;

class DWARFDebugAranges {
public:
  void generate(DWARFContext *CTX);
  uint32_t findAddress(uint64_t Address) const;

private:
  void clear();
  void extract(DataExtractor DebugArangesData);

  // Use appendRange multiple times and then call sortAndMinimize.
  void appendRange(uint32_t CUOffset, uint64_t LowPC, uint64_t HighPC);
  void sortAndMinimize();

  struct Range {
    explicit Range(uint64_t LowPC = -1ULL, uint64_t HighPC = -1ULL,
                   uint32_t CUOffset = -1U)
      : LowPC(LowPC), Length(HighPC - LowPC), CUOffset(CUOffset) {}

    void setHighPC(uint64_t HighPC) {
      if (HighPC == -1ULL || HighPC <= LowPC)
        Length = 0;
      else
        Length = HighPC - LowPC;
    }
    uint64_t HighPC() const {
      if (Length)
        return LowPC + Length;
      return -1ULL;
    }
    bool containsAddress(uint64_t Address) const {
      return LowPC <= Address && Address < HighPC();
    }

    bool operator <(const Range &other) const {
      return LowPC < other.LowPC;
    }

    static bool SortedOverlapCheck(const Range &Left, const Range &Right) {
      if (Left.CUOffset != Right.CUOffset)
        return false;
      return Left.HighPC() >= Right.LowPC;
    }

    uint64_t LowPC; // Start of address range.
    uint32_t Length; // End of address range (not including this address).
    uint32_t CUOffset; // Offset of the compile unit or die.
  };

  typedef std::vector<Range>              RangeColl;
  typedef RangeColl::const_iterator       RangeCollIterator;
  typedef DenseSet<uint32_t>              ParsedCUOffsetColl;

  RangeColl Aranges;
  ParsedCUOffsetColl ParsedCUOffsets;
};

}

#endif
