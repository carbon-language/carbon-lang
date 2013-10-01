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

#include "DWARFDebugArangeSet.h"
#include "llvm/ADT/DenseSet.h"
#include <list>

namespace llvm {

class DWARFContext;

class DWARFDebugAranges {
public:
  struct Range {
    explicit Range(uint64_t lo = -1ULL, uint64_t hi = -1ULL,
                   uint32_t off = -1U)
      : LoPC(lo), Length(hi-lo), Offset(off) {}

    void clear() {
      LoPC = -1ULL;
      Length = 0;
      Offset = -1U;
    }

    void setHiPC(uint64_t HiPC) {
      if (HiPC == -1ULL || HiPC <= LoPC)
        Length = 0;
      else
        Length = HiPC - LoPC;
    }
    uint64_t HiPC() const {
      if (Length)
        return LoPC + Length;
      return -1ULL;
    }
    bool isValidRange() const { return Length > 0; }

    static bool SortedOverlapCheck(const Range &curr_range,
                                   const Range &next_range, uint32_t n) {
      if (curr_range.Offset != next_range.Offset)
        return false;
      return curr_range.HiPC() + n >= next_range.LoPC;
    }

    bool contains(const Range &range) const {
      return LoPC <= range.LoPC && range.HiPC() <= HiPC();
    }

    void dump(raw_ostream &OS) const;
    uint64_t LoPC; // Start of address range
    uint32_t Length; // End of address range (not including this address)
    uint32_t Offset; // Offset of the compile unit or die
  };

  void clear() {
    Aranges.clear();
    ParsedCUOffsets.clear();
  }
  void extract(DataExtractor DebugArangesData);
  void generate(DWARFContext *CTX);

  // Use appendRange multiple times and then call sort.
  void appendRange(uint32_t CUOffset, uint64_t LowPC, uint64_t HighPC);
  void sort(bool Minimize, uint32_t OverlapSize);

  void dump(raw_ostream &OS) const;
  uint32_t findAddress(uint64_t Address) const;

  typedef std::vector<Range>              RangeColl;
  typedef RangeColl::const_iterator       RangeCollIterator;
  typedef DenseSet<uint32_t>              ParsedCUOffsetColl;

private:
  RangeColl Aranges;
  ParsedCUOffsetColl ParsedCUOffsets;
};

}

#endif
