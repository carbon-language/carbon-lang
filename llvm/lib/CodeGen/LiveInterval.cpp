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

#include "LiveInterval.h"
#include "Support/STLExtras.h"
#include <ostream>
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
bool LiveInterval::overlaps(const LiveInterval& other) const {
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
  } else {
    return true;
  }

  while (i != ie && j != je) {
    if (i->start == j->start)
      return true;

    if (i->start > j->start) {
      swap(i, j);
      swap(ie, je);
    }
    assert(i->start < j->start);

    if (i->end > j->start)
      return true;
    ++i;
  }

  return false;
}

/// extendIntervalEndTo - This method is used when we want to extend the range
/// specified by I to end at the specified endpoint.  To do this, we should
/// merge and eliminate all ranges that this will overlap with.  The iterator is
/// not invalidated.
void LiveInterval::extendIntervalEndTo(Ranges::iterator I, unsigned NewEnd) {
  assert(I != ranges.end() && "Not a valid interval!");

  // Search for the first interval that we can't merge with.
  Ranges::iterator MergeTo = next(I);
  for (; MergeTo != ranges.end() && NewEnd >= MergeTo->end; ++MergeTo)
    /*empty*/;

  // If NewEnd was in the middle of an interval, make sure to get its endpoint.
  I->end = std::max(NewEnd, prior(MergeTo)->end);

  // Erase any dead ranges
  ranges.erase(next(I), MergeTo);
}


/// extendIntervalStartTo - This method is used when we want to extend the range
/// specified by I to start at the specified endpoint.  To do this, we should
/// merge and eliminate all ranges that this will overlap with.
LiveInterval::Ranges::iterator 
LiveInterval::extendIntervalStartTo(Ranges::iterator I, unsigned NewStart) {
  assert(I != ranges.end() && "Not a valid interval!");

  // Search for the first interval that we can't merge with.
  Ranges::iterator MergeTo = I;
  do {
    if (MergeTo == ranges.begin()) {
      I->start = NewStart;
      return I;
    }
    --MergeTo;
  } while (NewStart <= MergeTo->start);

  // If we start in the middle of another interval, just delete a range and
  // extend that interval.
  if (MergeTo->end >= NewStart) {
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
    if (B->start <= Start && B->end >= Start) {
      extendIntervalEndTo(B, End);
      return B;
    }
  }

  // Otherwise, if this range ends in the middle of, or right next to, another
  // interval, merge it into that interval.
  if (it != ranges.end() && it->start <= End)
    return extendIntervalStartTo(it, Start);

  // Otherwise, this is just a new range that doesn't interact with anything.
  // Insert it.
  return ranges.insert(it, LR);
}

void LiveInterval::join(const LiveInterval& other) {
  isDefinedOnce &= other.isDefinedOnce;

  // Join the ranges of other into the ranges of this interval.
  Ranges::iterator cur = ranges.begin();
  for (Ranges::const_iterator i = other.ranges.begin(),
         e = other.ranges.end(); i != e; ++i)
    cur = addRangeFrom(*i, cur);

  weight += other.weight;
}

std::ostream& llvm::operator<<(std::ostream& os, const LiveRange &LR) {
  return os << "[" << LR.start << "," << LR.end << ")";
}

std::ostream& llvm::operator<<(std::ostream& os, const LiveInterval& li) {
  os << "%reg" << li.reg << ',' << li.weight;
  if (li.empty())
    return os << "EMPTY";

  os << " = ";
  for (LiveInterval::Ranges::const_iterator i = li.ranges.begin(),
         e = li.ranges.end(); i != e; ++i)
    os << *i;
  return os;
}
