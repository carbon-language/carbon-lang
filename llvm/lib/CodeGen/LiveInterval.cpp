//===-- LiveInterval.cpp - Live Interval Representation -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include <algorithm>
using namespace llvm;

// An example for liveAt():
//
// this = [1,4), liveAt(0) will return false. The instruction defining this
// spans slots [0,3]. The interval belongs to an spilled definition of the
// variable it represents. This is because slot 1 is used (def slot) and spans
// up to slot 3 (store slot).
//
bool LiveInterval::liveAt(SlotIndex I) const {
  Ranges::const_iterator r = std::upper_bound(ranges.begin(), ranges.end(), I);

  if (r == ranges.begin())
    return false;

  --r;
  return r->contains(I);
}

// liveBeforeAndAt - Check if the interval is live at the index and the index
// just before it. If index is liveAt, check if it starts a new live range.
// If it does, then check if the previous live range ends at index-1.
bool LiveInterval::liveBeforeAndAt(SlotIndex I) const {
  Ranges::const_iterator r = std::upper_bound(ranges.begin(), ranges.end(), I);

  if (r == ranges.begin())
    return false;

  --r;
  if (!r->contains(I))
    return false;
  if (I != r->start)
    return true;
  // I is the start of a live range. Check if the previous live range ends
  // at I-1.
  if (r == ranges.begin())
    return false;
  return r->end == I;
}

/// killedAt - Return true if a live range ends at index. Note that the kill
/// point is not contained in the half-open live range. It is usually the
/// getDefIndex() slot following its last use.
bool LiveInterval::killedAt(SlotIndex I) const {
  Ranges::const_iterator r = std::lower_bound(ranges.begin(), ranges.end(), I);

  // Now r points to the first interval with start >= I, or ranges.end().
  if (r == ranges.begin())
    return false;

  --r;
  // Now r points to the last interval with end <= I.
  // r->end is the kill point.
  return r->end == I;
}

/// killedInRange - Return true if the interval has kills in [Start,End).
bool LiveInterval::killedInRange(SlotIndex Start, SlotIndex End) const {
  Ranges::const_iterator r =
    std::lower_bound(ranges.begin(), ranges.end(), End);

  // Now r points to the first interval with start >= End, or ranges.end().
  if (r == ranges.begin())
    return false;

  --r;
  // Now r points to the last interval with end <= End.
  // r->end is the kill point.
  return r->end >= Start && r->end < End;
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
  assert(!empty() && "empty interval");
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

/// overlaps - Return true if the live interval overlaps a range specified
/// by [Start, End).
bool LiveInterval::overlaps(SlotIndex Start, SlotIndex End) const {
  assert(Start < End && "Invalid range");
  const_iterator I = std::lower_bound(begin(), end(), End);
  return I != begin() && (--I)->end > Start;
}


/// ValNo is dead, remove it.  If it is the largest value number, just nuke it
/// (and any other deleted values neighboring it), otherwise mark it as ~1U so
/// it can be nuked later.
void LiveInterval::markValNoForDeletion(VNInfo *ValNo) {
  if (ValNo->id == getNumValNums()-1) {
    do {
      valnos.pop_back();
    } while (!valnos.empty() && valnos.back()->isUnused());
  } else {
    ValNo->setIsUnused(true);
  }
}

/// extendIntervalEndTo - This method is used when we want to extend the range
/// specified by I to end at the specified endpoint.  To do this, we should
/// merge and eliminate all ranges that this will overlap with.  The iterator is
/// not invalidated.
void LiveInterval::extendIntervalEndTo(Ranges::iterator I, SlotIndex NewEnd) {
  assert(I != ranges.end() && "Not a valid interval!");
  VNInfo *ValNo = I->valno;

  // Search for the first interval that we can't merge with.
  Ranges::iterator MergeTo = llvm::next(I);
  for (; MergeTo != ranges.end() && NewEnd >= MergeTo->end; ++MergeTo) {
    assert(MergeTo->valno == ValNo && "Cannot merge with differing values!");
  }

  // If NewEnd was in the middle of an interval, make sure to get its endpoint.
  I->end = std::max(NewEnd, prior(MergeTo)->end);

  // Erase any dead ranges.
  ranges.erase(llvm::next(I), MergeTo);

  // If the newly formed range now touches the range after it and if they have
  // the same value number, merge the two ranges into one range.
  Ranges::iterator Next = llvm::next(I);
  if (Next != ranges.end() && Next->start <= I->end && Next->valno == ValNo) {
    I->end = Next->end;
    ranges.erase(Next);
  }
}


/// extendIntervalStartTo - This method is used when we want to extend the range
/// specified by I to start at the specified endpoint.  To do this, we should
/// merge and eliminate all ranges that this will overlap with.
LiveInterval::Ranges::iterator
LiveInterval::extendIntervalStartTo(Ranges::iterator I, SlotIndex NewStart) {
  assert(I != ranges.end() && "Not a valid interval!");
  VNInfo *ValNo = I->valno;

  // Search for the first interval that we can't merge with.
  Ranges::iterator MergeTo = I;
  do {
    if (MergeTo == ranges.begin()) {
      I->start = NewStart;
      ranges.erase(MergeTo, I);
      return I;
    }
    assert(MergeTo->valno == ValNo && "Cannot merge with differing values!");
    --MergeTo;
  } while (NewStart <= MergeTo->start);

  // If we start in the middle of another interval, just delete a range and
  // extend that interval.
  if (MergeTo->end >= NewStart && MergeTo->valno == ValNo) {
    MergeTo->end = I->end;
  } else {
    // Otherwise, extend the interval right after.
    ++MergeTo;
    MergeTo->start = NewStart;
    MergeTo->end = I->end;
  }

  ranges.erase(llvm::next(MergeTo), llvm::next(I));
  return MergeTo;
}

LiveInterval::iterator
LiveInterval::addRangeFrom(LiveRange LR, iterator From) {
  SlotIndex Start = LR.start, End = LR.end;
  iterator it = std::upper_bound(From, ranges.end(), Start);

  // If the inserted interval starts in the middle or right at the end of
  // another interval, just extend that interval to contain the range of LR.
  if (it != ranges.begin()) {
    iterator B = prior(it);
    if (LR.valno == B->valno) {
      if (B->start <= Start && B->end >= Start) {
        extendIntervalEndTo(B, End);
        return B;
      }
    } else {
      // Check to make sure that we are not overlapping two live ranges with
      // different valno's.
      assert(B->end <= Start &&
             "Cannot overlap two LiveRanges with differing ValID's"
             " (did you def the same reg twice in a MachineInstr?)");
    }
  }

  // Otherwise, if this range ends in the middle of, or right next to, another
  // interval, merge it into that interval.
  if (it != ranges.end()) {
    if (LR.valno == it->valno) {
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
      // different valno's.
      assert(it->start >= End &&
             "Cannot overlap two LiveRanges with differing ValID's");
    }
  }

  // Otherwise, this is just a new range that doesn't interact with anything.
  // Insert it.
  return ranges.insert(it, LR);
}

/// isInOneLiveRange - Return true if the range specified is entirely in 
/// a single LiveRange of the live interval.
bool LiveInterval::isInOneLiveRange(SlotIndex Start, SlotIndex End) {
  Ranges::iterator I = std::upper_bound(ranges.begin(), ranges.end(), Start);
  if (I == ranges.begin())
    return false;
  --I;
  return I->containsRange(Start, End);
}


/// removeRange - Remove the specified range from this interval.  Note that
/// the range must be in a single LiveRange in its entirety.
void LiveInterval::removeRange(SlotIndex Start, SlotIndex End,
                               bool RemoveDeadValNo) {
  // Find the LiveRange containing this span.
  Ranges::iterator I = std::upper_bound(ranges.begin(), ranges.end(), Start);
  assert(I != ranges.begin() && "Range is not in interval!");
  --I;
  assert(I->containsRange(Start, End) && "Range is not entirely in interval!");

  // If the span we are removing is at the start of the LiveRange, adjust it.
  VNInfo *ValNo = I->valno;
  if (I->start == Start) {
    if (I->end == End) {
      if (RemoveDeadValNo) {
        // Check if val# is dead.
        bool isDead = true;
        for (const_iterator II = begin(), EE = end(); II != EE; ++II)
          if (II != I && II->valno == ValNo) {
            isDead = false;
            break;
          }
        if (isDead) {
          // Now that ValNo is dead, remove it.
          markValNoForDeletion(ValNo);
        }
      }

      ranges.erase(I);  // Removed the whole LiveRange.
    } else
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
  SlotIndex OldEnd = I->end;
  I->end = Start;   // Trim the old interval.

  // Insert the new one.
  ranges.insert(llvm::next(I), LiveRange(End, OldEnd, ValNo));
}

/// removeValNo - Remove all the ranges defined by the specified value#.
/// Also remove the value# from value# list.
void LiveInterval::removeValNo(VNInfo *ValNo) {
  if (empty()) return;
  Ranges::iterator I = ranges.end();
  Ranges::iterator E = ranges.begin();
  do {
    --I;
    if (I->valno == ValNo)
      ranges.erase(I);
  } while (I != E);
  // Now that ValNo is dead, remove it.
  markValNoForDeletion(ValNo);
}

/// getLiveRangeContaining - Return the live range that contains the
/// specified index, or null if there is none.
LiveInterval::const_iterator 
LiveInterval::FindLiveRangeContaining(SlotIndex Idx) const {
  const_iterator It = std::upper_bound(begin(), end(), Idx);
  if (It != ranges.begin()) {
    --It;
    if (It->contains(Idx))
      return It;
  }

  return end();
}

LiveInterval::iterator 
LiveInterval::FindLiveRangeContaining(SlotIndex Idx) {
  iterator It = std::upper_bound(begin(), end(), Idx);
  if (It != begin()) {
    --It;
    if (It->contains(Idx))
      return It;
  }
  
  return end();
}

/// findDefinedVNInfo - Find the VNInfo defined by the specified
/// index (register interval).
VNInfo *LiveInterval::findDefinedVNInfoForRegInt(SlotIndex Idx) const {
  for (LiveInterval::const_vni_iterator i = vni_begin(), e = vni_end();
       i != e; ++i) {
    if ((*i)->def == Idx)
      return *i;
  }

  return 0;
}

/// findDefinedVNInfo - Find the VNInfo defined by the specified
/// register (stack inteval).
VNInfo *LiveInterval::findDefinedVNInfoForStackInt(unsigned reg) const {
  for (LiveInterval::const_vni_iterator i = vni_begin(), e = vni_end();
       i != e; ++i) {
    if ((*i)->getReg() == reg)
      return *i;
  }
  return 0;
}

/// join - Join two live intervals (this, and other) together.  This applies
/// mappings to the value numbers in the LHS/RHS intervals as specified.  If
/// the intervals are not joinable, this aborts.
void LiveInterval::join(LiveInterval &Other,
                        const int *LHSValNoAssignments,
                        const int *RHSValNoAssignments, 
                        SmallVector<VNInfo*, 16> &NewVNInfo,
                        MachineRegisterInfo *MRI) {
  // Determine if any of our live range values are mapped.  This is uncommon, so
  // we want to avoid the interval scan if not. 
  bool MustMapCurValNos = false;
  unsigned NumVals = getNumValNums();
  unsigned NumNewVals = NewVNInfo.size();
  for (unsigned i = 0; i != NumVals; ++i) {
    unsigned LHSValID = LHSValNoAssignments[i];
    if (i != LHSValID ||
        (NewVNInfo[LHSValID] && NewVNInfo[LHSValID] != getValNumInfo(i)))
      MustMapCurValNos = true;
  }

  // If we have to apply a mapping to our base interval assignment, rewrite it
  // now.
  if (MustMapCurValNos) {
    // Map the first live range.
    iterator OutIt = begin();
    OutIt->valno = NewVNInfo[LHSValNoAssignments[OutIt->valno->id]];
    ++OutIt;
    for (iterator I = OutIt, E = end(); I != E; ++I) {
      OutIt->valno = NewVNInfo[LHSValNoAssignments[I->valno->id]];
      
      // If this live range has the same value # as its immediate predecessor,
      // and if they are neighbors, remove one LiveRange.  This happens when we
      // have [0,3:0)[4,7:1) and map 0/1 onto the same value #.
      if (OutIt->valno == (OutIt-1)->valno && (OutIt-1)->end == OutIt->start) {
        (OutIt-1)->end = OutIt->end;
      } else {
        if (I != OutIt) {
          OutIt->start = I->start;
          OutIt->end = I->end;
        }
        
        // Didn't merge, on to the next one.
        ++OutIt;
      }
    }
    
    // If we merge some live ranges, chop off the end.
    ranges.erase(OutIt, end());
  }

  // Remember assignements because val# ids are changing.
  SmallVector<unsigned, 16> OtherAssignments;
  for (iterator I = Other.begin(), E = Other.end(); I != E; ++I)
    OtherAssignments.push_back(RHSValNoAssignments[I->valno->id]);

  // Update val# info. Renumber them and make sure they all belong to this
  // LiveInterval now. Also remove dead val#'s.
  unsigned NumValNos = 0;
  for (unsigned i = 0; i < NumNewVals; ++i) {
    VNInfo *VNI = NewVNInfo[i];
    if (VNI) {
      if (NumValNos >= NumVals)
        valnos.push_back(VNI);
      else 
        valnos[NumValNos] = VNI;
      VNI->id = NumValNos++;  // Renumber val#.
    }
  }
  if (NumNewVals < NumVals)
    valnos.resize(NumNewVals);  // shrinkify

  // Okay, now insert the RHS live ranges into the LHS.
  iterator InsertPos = begin();
  unsigned RangeNo = 0;
  for (iterator I = Other.begin(), E = Other.end(); I != E; ++I, ++RangeNo) {
    // Map the valno in the other live range to the current live range.
    I->valno = NewVNInfo[OtherAssignments[RangeNo]];
    assert(I->valno && "Adding a dead range?");
    InsertPos = addRangeFrom(*I, InsertPos);
  }

  ComputeJoinedWeight(Other);

  // Update regalloc hint if currently there isn't one.
  if (TargetRegisterInfo::isVirtualRegister(reg) &&
      TargetRegisterInfo::isVirtualRegister(Other.reg)) {
    std::pair<unsigned, unsigned> Hint = MRI->getRegAllocationHint(reg);
    if (Hint.first == 0 && Hint.second == 0) {
      std::pair<unsigned, unsigned> OtherHint =
        MRI->getRegAllocationHint(Other.reg);
      if (OtherHint.first || OtherHint.second)
        MRI->setRegAllocationHint(reg, OtherHint.first, OtherHint.second);
    }
  }
}

/// MergeRangesInAsValue - Merge all of the intervals in RHS into this live
/// interval as the specified value number.  The LiveRanges in RHS are
/// allowed to overlap with LiveRanges in the current interval, but only if
/// the overlapping LiveRanges have the specified value number.
void LiveInterval::MergeRangesInAsValue(const LiveInterval &RHS, 
                                        VNInfo *LHSValNo) {
  // TODO: Make this more efficient.
  iterator InsertPos = begin();
  for (const_iterator I = RHS.begin(), E = RHS.end(); I != E; ++I) {
    // Map the valno in the other live range to the current live range.
    LiveRange Tmp = *I;
    Tmp.valno = LHSValNo;
    InsertPos = addRangeFrom(Tmp, InsertPos);
  }
}


/// MergeValueInAsValue - Merge all of the live ranges of a specific val#
/// in RHS into this live interval as the specified value number.
/// The LiveRanges in RHS are allowed to overlap with LiveRanges in the
/// current interval, it will replace the value numbers of the overlaped
/// live ranges with the specified value number.
void LiveInterval::MergeValueInAsValue(
                                    const LiveInterval &RHS,
                                    const VNInfo *RHSValNo, VNInfo *LHSValNo) {
  SmallVector<VNInfo*, 4> ReplacedValNos;
  iterator IP = begin();
  for (const_iterator I = RHS.begin(), E = RHS.end(); I != E; ++I) {
    assert(I->valno == RHS.getValNumInfo(I->valno->id) && "Bad VNInfo");
    if (I->valno != RHSValNo)
      continue;
    SlotIndex Start = I->start, End = I->end;
    IP = std::upper_bound(IP, end(), Start);
    // If the start of this range overlaps with an existing liverange, trim it.
    if (IP != begin() && IP[-1].end > Start) {
      if (IP[-1].valno != LHSValNo) {
        ReplacedValNos.push_back(IP[-1].valno);
        IP[-1].valno = LHSValNo; // Update val#.
      }
      Start = IP[-1].end;
      // Trimmed away the whole range?
      if (Start >= End) continue;
    }
    // If the end of this range overlaps with an existing liverange, trim it.
    if (IP != end() && End > IP->start) {
      if (IP->valno != LHSValNo) {
        ReplacedValNos.push_back(IP->valno);
        IP->valno = LHSValNo;  // Update val#.
      }
      End = IP->start;
      // If this trimmed away the whole range, ignore it.
      if (Start == End) continue;
    }
    
    // Map the valno in the other live range to the current live range.
    IP = addRangeFrom(LiveRange(Start, End, LHSValNo), IP);
  }


  SmallSet<VNInfo*, 4> Seen;
  for (unsigned i = 0, e = ReplacedValNos.size(); i != e; ++i) {
    VNInfo *V1 = ReplacedValNos[i];
    if (Seen.insert(V1)) {
      bool isDead = true;
      for (const_iterator I = begin(), E = end(); I != E; ++I)
        if (I->valno == V1) {
          isDead = false;
          break;
        }          
      if (isDead) {
        // Now that V1 is dead, remove it.
        markValNoForDeletion(V1);
      }
    }
  }
}


/// MergeInClobberRanges - For any live ranges that are not defined in the
/// current interval, but are defined in the Clobbers interval, mark them
/// used with an unknown definition value.
void LiveInterval::MergeInClobberRanges(LiveIntervals &li_,
                                        const LiveInterval &Clobbers,
                                        VNInfo::Allocator &VNInfoAllocator) {
  if (Clobbers.empty()) return;
  
  DenseMap<VNInfo*, VNInfo*> ValNoMaps;
  VNInfo *UnusedValNo = 0;
  iterator IP = begin();
  for (const_iterator I = Clobbers.begin(), E = Clobbers.end(); I != E; ++I) {
    // For every val# in the Clobbers interval, create a new "unknown" val#.
    VNInfo *ClobberValNo = 0;
    DenseMap<VNInfo*, VNInfo*>::iterator VI = ValNoMaps.find(I->valno);
    if (VI != ValNoMaps.end())
      ClobberValNo = VI->second;
    else if (UnusedValNo)
      ClobberValNo = UnusedValNo;
    else {
      UnusedValNo = ClobberValNo =
        getNextValue(li_.getInvalidIndex(), 0, false, VNInfoAllocator);
      ValNoMaps.insert(std::make_pair(I->valno, ClobberValNo));
    }

    bool Done = false;
    SlotIndex Start = I->start, End = I->end;
    // If a clobber range starts before an existing range and ends after
    // it, the clobber range will need to be split into multiple ranges.
    // Loop until the entire clobber range is handled.
    while (!Done) {
      Done = true;
      IP = std::upper_bound(IP, end(), Start);
      SlotIndex SubRangeStart = Start;
      SlotIndex SubRangeEnd = End;

      // If the start of this range overlaps with an existing liverange, trim it.
      if (IP != begin() && IP[-1].end > SubRangeStart) {
        SubRangeStart = IP[-1].end;
        // Trimmed away the whole range?
        if (SubRangeStart >= SubRangeEnd) continue;
      }
      // If the end of this range overlaps with an existing liverange, trim it.
      if (IP != end() && SubRangeEnd > IP->start) {
        // If the clobber live range extends beyond the existing live range,
        // it'll need at least another live range, so set the flag to keep
        // iterating.
        if (SubRangeEnd > IP->end) {
          Start = IP->end;
          Done = false;
        }
        SubRangeEnd = IP->start;
        // If this trimmed away the whole range, ignore it.
        if (SubRangeStart == SubRangeEnd) continue;
      }

      // Insert the clobber interval.
      IP = addRangeFrom(LiveRange(SubRangeStart, SubRangeEnd, ClobberValNo),
                        IP);
      UnusedValNo = 0;
    }
  }

  if (UnusedValNo) {
    // Delete the last unused val#.
    valnos.pop_back();
  }
}

void LiveInterval::MergeInClobberRange(LiveIntervals &li_,
                                       SlotIndex Start,
                                       SlotIndex End,
                                       VNInfo::Allocator &VNInfoAllocator) {
  // Find a value # to use for the clobber ranges.  If there is already a value#
  // for unknown values, use it.
  VNInfo *ClobberValNo =
    getNextValue(li_.getInvalidIndex(), 0, false, VNInfoAllocator);
  
  iterator IP = begin();
  IP = std::upper_bound(IP, end(), Start);
    
  // If the start of this range overlaps with an existing liverange, trim it.
  if (IP != begin() && IP[-1].end > Start) {
    Start = IP[-1].end;
    // Trimmed away the whole range?
    if (Start >= End) return;
  }
  // If the end of this range overlaps with an existing liverange, trim it.
  if (IP != end() && End > IP->start) {
    End = IP->start;
    // If this trimmed away the whole range, ignore it.
    if (Start == End) return;
  }
    
  // Insert the clobber interval.
  addRangeFrom(LiveRange(Start, End, ClobberValNo), IP);
}

/// MergeValueNumberInto - This method is called when two value nubmers
/// are found to be equivalent.  This eliminates V1, replacing all
/// LiveRanges with the V1 value number with the V2 value number.  This can
/// cause merging of V1/V2 values numbers and compaction of the value space.
VNInfo* LiveInterval::MergeValueNumberInto(VNInfo *V1, VNInfo *V2) {
  assert(V1 != V2 && "Identical value#'s are always equivalent!");

  // This code actually merges the (numerically) larger value number into the
  // smaller value number, which is likely to allow us to compactify the value
  // space.  The only thing we have to be careful of is to preserve the
  // instruction that defines the result value.

  // Make sure V2 is smaller than V1.
  if (V1->id < V2->id) {
    V1->copyFrom(*V2);
    std::swap(V1, V2);
  }

  // Merge V1 live ranges into V2.
  for (iterator I = begin(); I != end(); ) {
    iterator LR = I++;
    if (LR->valno != V1) continue;  // Not a V1 LiveRange.
    
    // Okay, we found a V1 live range.  If it had a previous, touching, V2 live
    // range, extend it.
    if (LR != begin()) {
      iterator Prev = LR-1;
      if (Prev->valno == V2 && Prev->end == LR->start) {
        Prev->end = LR->end;

        // Erase this live-range.
        ranges.erase(LR);
        I = Prev+1;
        LR = Prev;
      }
    }
    
    // Okay, now we have a V1 or V2 live range that is maximally merged forward.
    // Ensure that it is a V2 live-range.
    LR->valno = V2;
    
    // If we can merge it into later V2 live ranges, do so now.  We ignore any
    // following V1 live ranges, as they will be merged in subsequent iterations
    // of the loop.
    if (I != end()) {
      if (I->start == LR->end && I->valno == V2) {
        LR->end = I->end;
        ranges.erase(I);
        I = LR+1;
      }
    }
  }
  
  // Now that V1 is dead, remove it.
  markValNoForDeletion(V1);
  
  return V2;
}

void LiveInterval::Copy(const LiveInterval &RHS,
                        MachineRegisterInfo *MRI,
                        VNInfo::Allocator &VNInfoAllocator) {
  ranges.clear();
  valnos.clear();
  std::pair<unsigned, unsigned> Hint = MRI->getRegAllocationHint(RHS.reg);
  MRI->setRegAllocationHint(reg, Hint.first, Hint.second);

  weight = RHS.weight;
  for (unsigned i = 0, e = RHS.getNumValNums(); i != e; ++i) {
    const VNInfo *VNI = RHS.getValNumInfo(i);
    createValueCopy(VNI, VNInfoAllocator);
  }
  for (unsigned i = 0, e = RHS.ranges.size(); i != e; ++i) {
    const LiveRange &LR = RHS.ranges[i];
    addRange(LiveRange(LR.start, LR.end, getValNumInfo(LR.valno->id)));
  }
}

unsigned LiveInterval::getSize() const {
  unsigned Sum = 0;
  for (const_iterator I = begin(), E = end(); I != E; ++I)
    Sum += I->start.distance(I->end);
  return Sum;
}

/// ComputeJoinedWeight - Set the weight of a live interval Joined
/// after Other has been merged into it.
void LiveInterval::ComputeJoinedWeight(const LiveInterval &Other) {
  // If either of these intervals was spilled, the weight is the
  // weight of the non-spilled interval.  This can only happen with
  // iterative coalescers.

  if (Other.weight != HUGE_VALF) {
    weight += Other.weight;
  }
  else if (weight == HUGE_VALF &&
      !TargetRegisterInfo::isPhysicalRegister(reg)) {
    // Remove this assert if you have an iterative coalescer
    assert(0 && "Joining to spilled interval");
    weight = Other.weight;
  }
  else {
    // Otherwise the weight stays the same
    // Remove this assert if you have an iterative coalescer
    assert(0 && "Joining from spilled interval");
  }
}

raw_ostream& llvm::operator<<(raw_ostream& os, const LiveRange &LR) {
  return os << '[' << LR.start << ',' << LR.end << ':' << LR.valno->id << ")";
}

void LiveRange::dump() const {
  dbgs() << *this << "\n";
}

void LiveInterval::print(raw_ostream &OS, const TargetRegisterInfo *TRI) const {
  if (isStackSlot())
    OS << "SS#" << getStackSlotIndex();
  else if (TRI && TargetRegisterInfo::isPhysicalRegister(reg))
    OS << TRI->getName(reg);
  else
    OS << "%reg" << reg;

  OS << ',' << weight;

  if (empty())
    OS << " EMPTY";
  else {
    OS << " = ";
    for (LiveInterval::Ranges::const_iterator I = ranges.begin(),
           E = ranges.end(); I != E; ++I) {
      OS << *I;
      assert(I->valno == getValNumInfo(I->valno->id) && "Bad VNInfo");
    }
  }

  // Print value number info.
  if (getNumValNums()) {
    OS << "  ";
    unsigned vnum = 0;
    for (const_vni_iterator i = vni_begin(), e = vni_end(); i != e;
         ++i, ++vnum) {
      const VNInfo *vni = *i;
      if (vnum) OS << " ";
      OS << vnum << "@";
      if (vni->isUnused()) {
        OS << "x";
      } else {
        if (!vni->isDefAccurate() && !vni->isPHIDef())
          OS << "?";
        else
          OS << vni->def;
        if (vni->hasPHIKill())
          OS << "-phikill";
        if (vni->hasRedefByEC())
          OS << "-ec";
      }
    }
  }
}

void LiveInterval::dump() const {
  dbgs() << *this << "\n";
}


void LiveRange::print(raw_ostream &os) const {
  os << *this;
}
