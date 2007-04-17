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
#include "llvm/Support/Streams.h"
#include <iosfwd>
#include <vector>
#include <cassert>

namespace llvm {
  class MachineInstr;
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
    void print(std::ostream &os) const;
    void print(std::ostream *os) const { if (os) print(*os); }

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
    unsigned preference; // preferred register to allocate for this interval
    float weight;        // weight of this interval
    MachineInstr* remat; // definition if the definition rematerializable
    Ranges ranges;       // the ranges in which this register is live
  private:
    /// ValueNumberInfo - If this value number is not defined by a copy, this
    /// holds ~0,x.  If the value number is not in use, it contains ~1,x to
    /// indicate that the value # is not used.  If the val# is defined by a
    /// copy, the first entry is the instruction # of the copy, and the second
    /// is the register number copied from.
    SmallVector<std::pair<unsigned,unsigned>, 4> ValueNumberInfo;
  public:

    LiveInterval(unsigned Reg, float Weight)
      : reg(Reg), preference(0), weight(Weight), remat(NULL) {
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
      std::swap(remat, other.remat);
      std::swap(ranges, other.ranges);
      std::swap(ValueNumberInfo, other.ValueNumberInfo);
    }

    bool containsOneValue() const { return ValueNumberInfo.size() == 1; }

    unsigned getNumValNums() const { return ValueNumberInfo.size(); }
    
    /// getNextValue - Create a new value number and return it.  MIIdx specifies
    /// the instruction that defines the value number.
    unsigned getNextValue(unsigned MIIdx, unsigned SrcReg) {
      ValueNumberInfo.push_back(std::make_pair(MIIdx, SrcReg));
      return ValueNumberInfo.size()-1;
    }
    
    /// getInstForValNum - Return the machine instruction index that defines the
    /// specified value number.
    unsigned getInstForValNum(unsigned ValNo) const {
      //assert(ValNo < ValueNumberInfo.size());
      return ValueNumberInfo[ValNo].first;
    }
    
    unsigned getSrcRegForValNum(unsigned ValNo) const {
      //assert(ValNo < ValueNumberInfo.size());
      if (ValueNumberInfo[ValNo].first < ~2U)
        return ValueNumberInfo[ValNo].second;
      return 0;
    }
    
    std::pair<unsigned, unsigned> getValNumInfo(unsigned ValNo) const {
      //assert(ValNo < ValueNumberInfo.size());
      return ValueNumberInfo[ValNo];
    }
    
    /// setValueNumberInfo - Change the value number info for the specified
    /// value number.
    void setValueNumberInfo(unsigned ValNo,
                            const std::pair<unsigned, unsigned> &I){
      ValueNumberInfo[ValNo] = I;
    }
    
    /// MergeValueNumberInto - This method is called when two value nubmers
    /// are found to be equivalent.  This eliminates V1, replacing all
    /// LiveRanges with the V1 value number with the V2 value number.  This can
    /// cause merging of V1/V2 values numbers and compaction of the value space.
    void MergeValueNumberInto(unsigned V1, unsigned V2);

    /// MergeInClobberRanges - For any live ranges that are not defined in the
    /// current interval, but are defined in the Clobbers interval, mark them
    /// used with an unknown definition value.
    void MergeInClobberRanges(const LiveInterval &Clobbers);

    
    /// MergeRangesInAsValue - Merge all of the intervals in RHS into this live
    /// interval as the specified value number.  The LiveRanges in RHS are
    /// allowed to overlap with LiveRanges in the current interval, but only if
    /// the overlapping LiveRanges have the specified value number.
    void MergeRangesInAsValue(const LiveInterval &RHS, unsigned LHSValNo);
    
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
    const LiveRange *getLiveRangeContaining(unsigned Idx) const {
      const_iterator I = FindLiveRangeContaining(Idx);
      return I == end() ? 0 : &*I;
    }

    /// FindLiveRangeContaining - Return an iterator to the live range that
    /// contains the specified index, or end() if there is none.
    const_iterator FindLiveRangeContaining(unsigned Idx) const;

    /// FindLiveRangeContaining - Return an iterator to the live range that
    /// contains the specified index, or end() if there is none.
    iterator FindLiveRangeContaining(unsigned Idx);
    
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

    /// join - Join two live intervals (this, and other) together.  This applies
    /// mappings to the value numbers in the LHS/RHS intervals as specified.  If
    /// the intervals are not joinable, this aborts.
    void join(LiveInterval &Other, int *ValNoAssignments,
              int *RHSValNoAssignments,
              SmallVector<std::pair<unsigned,unsigned>,16> &NewValueNumberInfo);

    /// removeRange - Remove the specified range from this interval.  Note that
    /// the range must already be in this interval in its entirety.
    void removeRange(unsigned Start, unsigned End);

    void removeRange(LiveRange LR) {
      removeRange(LR.start, LR.end);
    }

    /// getSize - Returns the sum of sizes of all the LiveRange's.
    ///
    unsigned getSize() const;

    bool operator<(const LiveInterval& other) const {
      return beginNumber() < other.beginNumber();
    }

    void print(std::ostream &OS, const MRegisterInfo *MRI = 0) const;
    void print(std::ostream *OS, const MRegisterInfo *MRI = 0) const {
      if (OS) print(*OS, MRI);
    }
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
