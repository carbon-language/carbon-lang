//===-- llvm/CodeGen/LiveInterval.h - Interval representation ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LiveRange and LiveInterval classes.  Given some
// numbering of each the machine instructions an interval [i, j) is said to be a
// live interval for register v if there is no instruction with number j' > j
// such that v is live at j' abd there is no instruction with number i' < i such
// that v is live at i'. In this implementation intervals can have holes,
// i.e. an interval might look like [1,20), [50,65), [1000,1001).  Each
// individual range is represented as an instance of LiveRange, and the whole
// interval is represented as an instance of LiveInterval.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LIVEINTERVAL_H
#define LLVM_CODEGEN_LIVEINTERVAL_H

#include <iosfwd>
#include <vector>
#include <cassert>

namespace llvm {
  /// LiveRange structure - This represents a simple register range in the
  /// program, with an inclusive start point and an exclusive end point.
  /// These ranges are rendered as [start,end).
  struct LiveRange {
    unsigned start;  // Start point of the interval (inclusive)
    unsigned end;  // End point of the interval (exclusive)
    LiveRange(unsigned S, unsigned E) : start(S), end(E) {
      assert(S < E && "Cannot create empty or backwards range");
    }

    /// contains - Return true if the index is covered by this range.
    ///
    bool contains(unsigned I) const {
      return start <= I && I < end;
    }

    bool operator<(const LiveRange &LR) const {
      return start < LR.start || (start == LR.start && end < LR.end);
    }
    bool operator==(const LiveRange &LR) const {
      return start == LR.start && end == LR.end;
    }
  private:
    LiveRange(); // DO NOT IMPLEMENT
  };
  std::ostream& operator<<(std::ostream& os, const LiveRange &LR);

  inline bool operator<(unsigned V, const LiveRange &LR) {
    return V < LR.start;
  }


  /// LiveInterval - This class represents some number of live ranges for a
  /// register or value.  This class also contains a bit of register allocator
  /// state.
  struct LiveInterval {
    typedef std::vector<LiveRange> Ranges;
    unsigned reg;        // the register of this interval
    float weight;        // weight of this interval
    Ranges ranges;       // the ranges in which this register is live
    bool isDefinedOnce;  // True if this interval contains one value.

    LiveInterval(unsigned Reg, float Weight)
      : reg(Reg), weight(Weight), isDefinedOnce(false) {
    }

    bool containsOneValue() const { return isDefinedOnce; }

    bool empty() const { return ranges.empty(); }

    /// start - Return the lowest numbered slot covered by interval.
    unsigned start() const {
      assert(!empty() && "empty interval for register");
      return ranges.front().start;
    }

    /// end - return the maximum point of the interval of the whole,
    /// exclusive.
    unsigned end() const {
      assert(!empty() && "empty interval for register");
      return ranges.back().end;
    }

    bool expiredAt(unsigned index) const {
      return end() <= (index + 1);
    }

    bool liveAt(unsigned index) const;

    bool overlaps(const LiveInterval& other) const;

    /// addRange - Add the specified LiveRange to this interval, merging
    /// intervals as appropriate.  This returns an iterator to the inserted live
    /// range (which may have grown since it was inserted.
    void addRange(LiveRange LR) {
      addRangeFrom(LR, ranges.begin());
    }

    void join(const LiveInterval& other);

    bool operator<(const LiveInterval& other) const {
      return start() < other.start();
    }

    bool operator==(const LiveInterval& other) const {
      return reg == other.reg;
    }

  private:
    Ranges::iterator addRangeFrom(LiveRange LR, Ranges::iterator From);
    void extendIntervalEndTo(Ranges::iterator I, unsigned NewEnd);
    Ranges::iterator extendIntervalStartTo(Ranges::iterator I, unsigned NewStr);
  };

  std::ostream& operator<<(std::ostream& os, const LiveInterval& li);
}

#endif
