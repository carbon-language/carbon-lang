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

void LiveInterval::addRange(LiveRange LR) {
  Ranges::iterator it =
    ranges.insert(std::upper_bound(ranges.begin(), ranges.end(), LR.start), LR);

  mergeRangesBackward(mergeRangesForward(it));
}

void LiveInterval::join(const LiveInterval& other) {
  Ranges::iterator cur = ranges.begin();
  isDefinedOnce &= other.isDefinedOnce;

  for (Ranges::const_iterator i = other.ranges.begin(),
         e = other.ranges.end(); i != e; ++i) {
    cur = ranges.insert(std::upper_bound(cur, ranges.end(), i->start), *i);
    cur = mergeRangesBackward(mergeRangesForward(cur));
  }
  weight += other.weight;
}

LiveInterval::Ranges::iterator
LiveInterval::mergeRangesForward(Ranges::iterator it) {
  Ranges::iterator n;
  while ((n = next(it)) != ranges.end()) {
    if (n->start > it->end)
      break;
    it->end = std::max(it->end, n->end);
    n = ranges.erase(n);
  }
  return it;
}

LiveInterval::Ranges::iterator
LiveInterval::mergeRangesBackward(Ranges::iterator it) {
  while (it != ranges.begin()) {
    Ranges::iterator p = prior(it);
    if (it->start > p->end)
      break;

    it->start = std::min(it->start, p->start);
    it->end = std::max(it->end, p->end);
    it = ranges.erase(p);
  }

  return it;
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
