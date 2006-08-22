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
// such that v is live at j' and there is no instruction with number i' < i such
// that v is live at i'. In this implementation intervals can have holes,
// i.e. an interval might look like [1,20), [50,65), [1000,1001).  Each
// individual range is represented as an instance of LiveRange, and the whole
// interval is represented as an instance of LiveInterval.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LIVEINTERVAL_H
#define LLVM_CODEGEN_LIVEINTERVAL_H

#include "llvm/ADT/SmallVector.h"
#include <iosfwd>
#include <vector>
#include <cassert>

namespace llvm {
  class MRegisterInfo;

  /// LiveRange structure - This represents a simple register range in the
  /// program, with an inclusive start point and an exclusive end point.
  /// These ranges are rendered as [start,end).
  struct LiveRange {
    unsigned start;  // Start point of the interval (inclusive)
    unsigned end;    // End point of the interval (exclusive)
    unsigned ValId;  // identifier for the value contained in this interval.

    LiveRange(unsigned S, unsigned E, unsigned V) : start(S), end(E), ValId(V) {
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

    void dump() const;

  private:
    LiveRange(); // DO NOT IMPLEMENT
  };
  std::ostream& operator<<(std::ostream& os, const LiveRange &LR);

  inline bool operator<(unsigned V, const LiveRange &LR) {
    return V < LR.start;
  }

  inline bool operator<(const LiveRange &LR, unsigned V) {
    return LR.start < V;
  }

  /// LiveInterval - This class represents some number of live ranges for a
  /// register or value.  This class also contains a bit of register allocator
  /// state.
  struct LiveInterval {
    typedef SmallVector<LiveRange,4> Ranges;
    unsigned reg;        // the register of this interval
    float weight;        // weight of this interval
    Ranges ranges;       // the ranges in which this register is live
  private:
    unsigned NumValues;  // the number of distinct values in this interval.
    
    /// InstDefiningValue - This tracks the def index of the instruction that
    /// defines a particular value number in the interval.  This may be ~0,
    /// which is treated as unknown.
    SmallVector<unsigned, 4> InstDefiningValue;
  public:

    LiveInterval(unsigned Reg, float Weight)
      : reg(Reg), weight(Weight), NumValues(0) {
    }

    typedef Ranges::iterator iterator;
    iterator begin() { return ranges.begin(); }
    iterator end()   { return ranges.end(); }

    typedef Ranges::const_iterator const_iterator;
    const_iterator begin() const { return ranges.begin(); }
    const_iterator end() const  { return ranges.end(); }


    /// advanceTo - Advance the specified iterator to point to the LiveRange
    /// containing the specified position, or end() if the position is past the
    /// end of the interval.  If no LiveRange contains this position, but the
    /// position is in a hole, this method returns an iterator pointing the the
    /// LiveRange immediately after the hole.
    iterator advanceTo(iterator I, unsigned Pos) {
      if (Pos >= endNumber())
        return end();
      while (I->end <= Pos) ++I;
      return I;
    }

    void swap(LiveInterval& other) {
      std::swap(reg, other.reg);
      std::swap(weight, other.weight);
      std::swap(ranges, other.ranges);
      std::swap(NumValues, other.NumValues);
    }

    bool containsOneValue() const { return NumValues == 1; }

    /// getNextValue - Create a new value number and return it.  MIIdx specifies
    /// the instruction that defines the value number.
    unsigned getNextValue(unsigned MIIdx) {
      InstDefiningValue.push_back(MIIdx);
      return NumValues++;
    }
    
    /// getInstForValNum - Return the machine instruction index that defines the
    /// specified value number.
    unsigned getInstForValNum(unsigned ValNo) const {
      return InstDefiningValue[ValNo];
    }
    
    /// setInstDefiningValNum - Change the instruction defining the specified
    /// value number to the specified instruction.
    void setInstDefiningValNum(unsigned ValNo, unsigned MIIdx) {
      InstDefiningValue[ValNo] = MIIdx;
    }


    bool empty() const { return ranges.empty(); }

    /// beginNumber - Return the lowest numbered slot covered by interval.
    unsigned beginNumber() const {
      assert(!empty() && "empty interval for register");
      return ranges.front().start;
    }

    /// endNumber - return the maximum point of the interval of the whole,
    /// exclusive.
    unsigned endNumber() const {
      assert(!empty() && "empty interval for register");
      return ranges.back().end;
    }

    bool expiredAt(unsigned index) const {
      return index >= endNumber();
    }

    bool liveAt(unsigned index) const;

    /// getLiveRangeContaining - Return the live range that contains the
    /// specified index, or null if there is none.
    const LiveRange *getLiveRangeContaining(unsigned Idx) const;


    /// joinable - Two intervals are joinable if the either don't overlap at all
    /// or if the destination of the copy is a single assignment value, and it
    /// only overlaps with one value in the source interval.
    bool joinable(const LiveInterval& other, unsigned CopyIdx) const;

    /// getOverlapingRanges - Given another live interval which is defined as a
    /// copy from this one, return a list of all of the live ranges where the
    /// two overlap and have different value numbers.
    void getOverlapingRanges(const LiveInterval &Other, unsigned CopyIdx,
                             std::vector<LiveRange*> &Ranges);

    /// overlaps - Return true if the intersection of the two live intervals is
    /// not empty.
    bool overlaps(const LiveInterval& other) const {
      return overlapsFrom(other, other.begin());
    }

    /// overlapsFrom - Return true if the intersection of the two live intervals
    /// is not empty.  The specified iterator is a hint that we can begin
    /// scanning the Other interval starting at I.
    bool overlapsFrom(const LiveInterval& other, const_iterator I) const;

    /// addRange - Add the specified LiveRange to this interval, merging
    /// intervals as appropriate.  This returns an iterator to the inserted live
    /// range (which may have grown since it was inserted.
    void addRange(LiveRange LR) {
      addRangeFrom(LR, ranges.begin());
    }

    /// join - Join two live intervals (this, and other) together.  This
    /// operation is the result of a copy instruction in the source program,
    /// that occurs at index 'CopyIdx' that copies from 'other' to 'this'.  This
    /// destroys 'other'.
    void join(LiveInterval& other, unsigned CopyIdx);


    /// removeRange - Remove the specified range from this interval.  Note that
    /// the range must already be in this interval in its entirety.
    void removeRange(unsigned Start, unsigned End);

    bool operator<(const LiveInterval& other) const {
      return beginNumber() < other.beginNumber();
    }

    void print(std::ostream &OS, const MRegisterInfo *MRI = 0) const;
    void dump() const;

  private:
    Ranges::iterator addRangeFrom(LiveRange LR, Ranges::iterator From);
    void extendIntervalEndTo(Ranges::iterator I, unsigned NewEnd);
    Ranges::iterator extendIntervalStartTo(Ranges::iterator I, unsigned NewStr);
    LiveInterval& operator=(const LiveInterval& rhs); // DO NOT IMPLEMENT
  };

  inline std::ostream &operator<<(std::ostream &OS, const LiveInterval &LI) {
    LI.print(OS);
    return OS;
  }
}

#endif
