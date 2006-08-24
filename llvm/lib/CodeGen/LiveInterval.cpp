//===-- LiveInterval.cpp - Live Interval Representation -------------------===//
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

#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Target/MRegisterInfo.h"
#include <algorithm>
#include <iostream>
#include <map>
using namespace llvm;

// An example for liveAt():
//
// this = [1,4), liveAt(0) will return false. The instruction defining this
// spans slots [0,3]. The interval belongs to an spilled definition of the
// variable it represents. This is because slot 1 is used (def slot) and spans
// up to slot 3 (store slot).
//
bool LiveInterval::liveAt(unsigned I) const {
  Ranges::const_iterator r = std::upper_bound(ranges.begin(), ranges.end(), I);

  if (r == ranges.begin())
    return false;

  --r;
  return r->contains(I);
}

// overlaps - Return true if the intersection of the two live intervals is
// not empty.
//
// An example for overlaps():
//
// 0: A = ...
// 4: B = ...
// 8: C = A + B ;; last use of A
//
// The live intervals should look like:
//
// A = [3, 11)
// B = [7, x)
// C = [11, y)
//
// A->overlaps(C) should return false since we want to be able to join
// A and C.
//
bool LiveInterval::overlapsFrom(const LiveInterval& other,
                                const_iterator StartPos) const {
  const_iterator i = begin();
  const_iterator ie = end();
  const_iterator j = StartPos;
  const_iterator je = other.end();

  assert((StartPos->start <= i->start || StartPos == other.begin()) &&
         StartPos != other.end() && "Bogus start position hint!");

  if (i->start < j->start) {
    i = std::upper_bound(i, ie, j->start);
    if (i != ranges.begin()) --i;
  } else if (j->start < i->start) {
    ++StartPos;
    if (StartPos != other.end() && StartPos->start <= i->start) {
      assert(StartPos < other.end() && i < end());
      j = std::upper_bound(j, je, i->start);
      if (j != other.ranges.begin()) --j;
    }
  } else {
    return true;
  }

  if (j == je) return false;

  while (i != ie) {
    if (i->start > j->start) {
      std::swap(i, j);
      std::swap(ie, je);
    }

    if (i->end > j->start)
      return true;
    ++i;
  }

  return false;
}

/// NontrivialOverlap - Check to see if the two live ranges specified by i and j
/// overlap.  If so, check to see if they have value numbers that are not 
/// iIdx/jIdx respectively.  If both conditions are true, return true.
static inline bool NontrivialOverlap(const LiveRange &I, const LiveRange &J,
                                     unsigned iIdx, unsigned jIdx) {
  if (I.start == J.start) {
    // If this is not the allowed value merge, we cannot join.
    if (I.ValId != iIdx || J.ValId != jIdx)
      return true;
  } else if (I.start < J.start) {
    if (I.end > J.start && (I.ValId != iIdx || J.ValId != jIdx)) {
      return true;
    }
  } else {
    if (J.end > I.start && (I.ValId != iIdx || J.ValId != jIdx))
      return true;
  }
  
  return false;
}

/// joinable - Two intervals are joinable if the either don't overlap at all
/// or if the destination of the copy is a single assignment value, and it
/// only overlaps with one value in the source interval.
bool LiveInterval::joinable(const LiveInterval &other, unsigned CopyIdx) const {
  const LiveRange *SourceLR = other.getLiveRangeContaining(CopyIdx-1);
  const LiveRange *DestLR = getLiveRangeContaining(CopyIdx);
  assert(SourceLR && DestLR && "Not joining due to a copy?");
  unsigned OtherValIdx = SourceLR->ValId;
  unsigned ThisValIdx = DestLR->ValId;

  Ranges::const_iterator i = ranges.begin();
  Ranges::const_iterator ie = ranges.end();
  Ranges::const_iterator j = other.ranges.begin();
  Ranges::const_iterator je = other.ranges.end();

  if (i->start < j->start) {
    i = std::upper_bound(i, ie, j->start);
    if (i != ranges.begin()) --i;
  } else if (j->start < i->start) {
    j = std::upper_bound(j, je, i->start);
    if (j != other.ranges.begin()) --j;
  }

  while (i != ie && j != je) {
    if (NontrivialOverlap(*i, *j, ThisValIdx, OtherValIdx))
      return false;
    
    if (i->end < j->end)
      ++i;
    else
      ++j;
  }

  return true;
}

/// getOverlapingRanges - Given another live interval which is defined as a
/// copy from this one, return a list of all of the live ranges where the
/// two overlap and have different value numbers.
void LiveInterval::getOverlapingRanges(const LiveInterval &other, 
                                       unsigned CopyIdx,
                                       std::vector<LiveRange*> &Ranges) {
  const LiveRange *SourceLR = getLiveRangeContaining(CopyIdx-1);
  const LiveRange *DestLR = other.getLiveRangeContaining(CopyIdx);
  assert(SourceLR && DestLR && "Not joining due to a copy?");
  unsigned OtherValIdx = SourceLR->ValId;
  unsigned ThisValIdx = DestLR->ValId;
  
  Ranges::iterator i = ranges.begin();
  Ranges::iterator ie = ranges.end();
  Ranges::const_iterator j = other.ranges.begin();
  Ranges::const_iterator je = other.ranges.end();
  
  if (i->start < j->start) {
    i = std::upper_bound(i, ie, j->start);
    if (i != ranges.begin()) --i;
  } else if (j->start < i->start) {
    j = std::upper_bound(j, je, i->start);
    if (j != other.ranges.begin()) --j;
  }
  
  while (i != ie && j != je) {
    if (NontrivialOverlap(*i, *j, ThisValIdx, OtherValIdx))
      Ranges.push_back(&*i);
    
    if (i->end < j->end)
      ++i;
    else
      ++j;
  }
}



/// extendIntervalEndTo - This method is used when we want to extend the range
/// specified by I to end at the specified endpoint.  To do this, we should
/// merge and eliminate all ranges that this will overlap with.  The iterator is
/// not invalidated.
void LiveInterval::extendIntervalEndTo(Ranges::iterator I, unsigned NewEnd) {
  assert(I != ranges.end() && "Not a valid interval!");
  unsigned ValId = I->ValId;

  // Search for the first interval that we can't merge with.
  Ranges::iterator MergeTo = next(I);
  for (; MergeTo != ranges.end() && NewEnd >= MergeTo->end; ++MergeTo) {
    assert(MergeTo->ValId == ValId && "Cannot merge with differing values!");
  }

  // If NewEnd was in the middle of an interval, make sure to get its endpoint.
  I->end = std::max(NewEnd, prior(MergeTo)->end);

  // Erase any dead ranges.
  ranges.erase(next(I), MergeTo);
  
  // If the newly formed range now touches the range after it and if they have
  // the same value number, merge the two ranges into one range.
  Ranges::iterator Next = next(I);
  if (Next != ranges.end() && Next->start <= I->end && Next->ValId == ValId) {
    I->end = Next->end;
    ranges.erase(Next);
  }
}


/// extendIntervalStartTo - This method is used when we want to extend the range
/// specified by I to start at the specified endpoint.  To do this, we should
/// merge and eliminate all ranges that this will overlap with.
LiveInterval::Ranges::iterator
LiveInterval::extendIntervalStartTo(Ranges::iterator I, unsigned NewStart) {
  assert(I != ranges.end() && "Not a valid interval!");
  unsigned ValId = I->ValId;

  // Search for the first interval that we can't merge with.
  Ranges::iterator MergeTo = I;
  do {
    if (MergeTo == ranges.begin()) {
      I->start = NewStart;
      ranges.erase(MergeTo, I);
      return I;
    }
    assert(MergeTo->ValId == ValId && "Cannot merge with differing values!");
    --MergeTo;
  } while (NewStart <= MergeTo->start);

  // If we start in the middle of another interval, just delete a range and
  // extend that interval.
  if (MergeTo->end >= NewStart && MergeTo->ValId == ValId) {
    MergeTo->end = I->end;
  } else {
    // Otherwise, extend the interval right after.
    ++MergeTo;
    MergeTo->start = NewStart;
    MergeTo->end = I->end;
  }

  ranges.erase(next(MergeTo), next(I));
  return MergeTo;
}

LiveInterval::Ranges::iterator
LiveInterval::addRangeFrom(LiveRange LR, Ranges::iterator From) {
  unsigned Start = LR.start, End = LR.end;
  Ranges::iterator it = std::upper_bound(From, ranges.end(), Start);

  // If the inserted interval starts in the middle or right at the end of
  // another interval, just extend that interval to contain the range of LR.
  if (it != ranges.begin()) {
    Ranges::iterator B = prior(it);
    if (LR.ValId == B->ValId) {
      if (B->start <= Start && B->end >= Start) {
        extendIntervalEndTo(B, End);
        return B;
      }
    } else {
      // Check to make sure that we are not overlapping two live ranges with
      // different ValId's.
      assert(B->end <= Start &&
             "Cannot overlap two LiveRanges with differing ValID's"
             " (did you def the same reg twice in a MachineInstr?)");
    }
  }

  // Otherwise, if this range ends in the middle of, or right next to, another
  // interval, merge it into that interval.
  if (it != ranges.end())
    if (LR.ValId == it->ValId) {
      if (it->start <= End) {
        it = extendIntervalStartTo(it, Start);

        // If LR is a complete superset of an interval, we may need to grow its
        // endpoint as well.
        if (End > it->end)
          extendIntervalEndTo(it, End);
        return it;
      }
    } else {
      // Check to make sure that we are not overlapping two live ranges with
      // different ValId's.
      assert(it->start >= End &&
             "Cannot overlap two LiveRanges with differing ValID's");
    }

  // Otherwise, this is just a new range that doesn't interact with anything.
  // Insert it.
  return ranges.insert(it, LR);
}


/// removeRange - Remove the specified range from this interval.  Note that
/// the range must already be in this interval in its entirety.
void LiveInterval::removeRange(unsigned Start, unsigned End) {
  // Find the LiveRange containing this span.
  Ranges::iterator I = std::upper_bound(ranges.begin(), ranges.end(), Start);
  assert(I != ranges.begin() && "Range is not in interval!");
  --I;
  assert(I->contains(Start) && I->contains(End-1) &&
         "Range is not entirely in interval!");

  // If the span we are removing is at the start of the LiveRange, adjust it.
  if (I->start == Start) {
    if (I->end == End)
      ranges.erase(I);  // Removed the whole LiveRange.
    else
      I->start = End;
    return;
  }

  // Otherwise if the span we are removing is at the end of the LiveRange,
  // adjust the other way.
  if (I->end == End) {
    I->end = Start;
    return;
  }

  // Otherwise, we are splitting the LiveRange into two pieces.
  unsigned OldEnd = I->end;
  I->end = Start;   // Trim the old interval.

  // Insert the new one.
  ranges.insert(next(I), LiveRange(End, OldEnd, I->ValId));
}

/// getLiveRangeContaining - Return the live range that contains the
/// specified index, or null if there is none.
LiveInterval::const_iterator 
LiveInterval::FindLiveRangeContaining(unsigned Idx) const {
  const_iterator It = std::upper_bound(begin(), end(), Idx);
  if (It != ranges.begin()) {
    --It;
    if (It->contains(Idx))
      return It;
  }

  return end();
}

LiveInterval::iterator 
LiveInterval::FindLiveRangeContaining(unsigned Idx) {
  iterator It = std::upper_bound(begin(), end(), Idx);
  if (It != ranges.begin()) {
    --It;
    if (It->contains(Idx))
      return It;
  }
  
  return end();
}

/// join - Join two live intervals (this, and other) together.  This operation
/// is the result of a copy instruction in the source program, that occurs at
/// index 'CopyIdx' that copies from 'Other' to 'this'.
void LiveInterval::join(LiveInterval &Other, unsigned CopyIdx) {
  const LiveRange *SourceLR = Other.getLiveRangeContaining(CopyIdx-1);
  const LiveRange *DestLR = getLiveRangeContaining(CopyIdx);
  assert(SourceLR && DestLR && "Not joining due to a copy?");
  unsigned MergedSrcValIdx = SourceLR->ValId;
  unsigned MergedDstValIdx = DestLR->ValId;

  // Try to do the least amount of work possible.  In particular, if there are
  // more liverange chunks in the other set than there are in the 'this' set,
  // swap sets to merge the fewest chunks in possible.
  if (Other.ranges.size() > ranges.size()) {
    std::swap(MergedSrcValIdx, MergedDstValIdx);
    std::swap(ranges, Other.ranges);
    std::swap(NumValues, Other.NumValues);
    std::swap(InstDefiningValue, Other.InstDefiningValue);
  }

  // Join the ranges of other into the ranges of this interval.
  Ranges::iterator InsertPos = ranges.begin();
  std::map<unsigned, unsigned> Dst2SrcIdxMap;
  for (Ranges::iterator I = Other.ranges.begin(),
         E = Other.ranges.end(); I != E; ++I) {
    // Map the ValId in the other live range to the current live range.
    if (I->ValId == MergedSrcValIdx)
      I->ValId = MergedDstValIdx;
    else {
      unsigned &NV = Dst2SrcIdxMap[I->ValId];
      if (NV == 0) NV = getNextValue(Other.getInstForValNum(I->ValId));
      I->ValId = NV;
    }

    InsertPos = addRangeFrom(*I, InsertPos);
  }
  
  // Update the value number information for the value number defined by the
  // copy.  The copy is about to be removed, so ensure that the value is defined
  // by whatever the other value is defined by.
  if (InstDefiningValue[MergedDstValIdx] == CopyIdx) {
    InstDefiningValue[MergedDstValIdx] =
      Other.InstDefiningValue[MergedSrcValIdx];
  }
  
  weight += Other.weight;
}

/// MergeValueNumberInto - This method is called when two value nubmers
/// are found to be equivalent.  This eliminates V1, replacing all
/// LiveRanges with the V1 value number with the V2 value number.  This can
/// cause merging of V1/V2 values numbers and compaction of the value space.
void LiveInterval::MergeValueNumberInto(unsigned V1, unsigned V2) {
  assert(V1 != V2 && "Identical value#'s are always equivalent!");

  // This code actually merges the (numerically) larger value number into the
  // smaller value number, which is likely to allow us to compactify the value
  // space.  The only thing we have to be careful of is to preserve the
  // instruction that defines the result value.

  // Make sure V2 is smaller than V1.
  if (V1 < V2) {
    setInstDefiningValNum(V1, getInstForValNum(V2));
    std::swap(V1, V2);
  }

  // Merge V1 live ranges into V2.
  for (iterator I = begin(); I != end(); ) {
    iterator LR = I++;
    if (LR->ValId != V1) continue;  // Not a V1 LiveRange.
    
    // Okay, we found a V1 live range.  If it had a previous, touching, V2 live
    // range, extend it.
    if (LR != begin()) {
      iterator Prev = LR-1;
      if (Prev->ValId == V2 && Prev->end == LR->start) {
        Prev->end = LR->end;

        // Erase this live-range.
        ranges.erase(LR);
        I = Prev+1;
        LR = Prev;
      }
    }
    
    // Okay, now we have a V1 or V2 live range that is maximally merged forward.
    // Ensure that it is a V2 live-range.
    LR->ValId = V2;
    
    // If we can merge it into later V2 live ranges, do so now.  We ignore any
    // following V1 live ranges, as they will be merged in subsequent iterations
    // of the loop.
    if (I != end()) {
      if (I->start == LR->end && I->ValId == V2) {
        LR->end = I->end;
        ranges.erase(I);
        I = LR+1;
      }
    }
  }
}


std::ostream& llvm::operator<<(std::ostream& os, const LiveRange &LR) {
  return os << '[' << LR.start << ',' << LR.end << ':' << LR.ValId << ")";
}

void LiveRange::dump() const {
  std::cerr << *this << "\n";
}

void LiveInterval::print(std::ostream &OS, const MRegisterInfo *MRI) const {
  if (MRI && MRegisterInfo::isPhysicalRegister(reg))
    OS << MRI->getName(reg);
  else
    OS << "%reg" << reg;

  OS << ',' << weight;

  if (empty())
    OS << "EMPTY";
  else {
    OS << " = ";
    for (LiveInterval::Ranges::const_iterator I = ranges.begin(),
           E = ranges.end(); I != E; ++I)
    OS << *I;
  }
  
  // Print value number info.
  if (NumValues) {
    OS << "  ";
    for (unsigned i = 0; i != NumValues; ++i) {
      if (i) OS << " ";
      OS << i << "@";
      if (InstDefiningValue[i] == ~0U) {
        OS << "?";
      } else {
        OS << InstDefiningValue[i];
      }
    }
  }
}

void LiveInterval::dump() const {
  std::cerr << *this << "\n";
}
