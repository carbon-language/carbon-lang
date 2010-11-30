//===-- LiveIntervalUnion.cpp - Live interval union data structure --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// LiveIntervalUnion represents a coalesced set of live intervals. This may be
// used during coalescing to represent a congruence class, or during register
// allocation to model liveness of a physical register.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "regalloc"
#include "LiveIntervalUnion.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
using namespace llvm;

// Find the first segment in the range [SegBegin,Segments.end()) that
// intersects with LS. If no intersection is found, return the first SI
// such that SI.start >= LS.End.
//
// This logic is tied to the underlying LiveSegments data structure. For now, we
// use set::upper_bound to find the nearest starting position,
// then reverse iterate to find the first overlap.
//
// Upon entry we have SegBegin.Start < LS.End
// SegBegin   |--...
//             \   .
//       LS ...-|
//
// After set::upper_bound, we have SI.start >= LS.start:
// SI   |--...
//     /
// LS |--...
//
// Assuming intervals are disjoint, if an intersection exists, it must be the
// segment found or the one immediately preceeding it. We continue reverse
// iterating to return the first overlapping segment.
LiveIntervalUnion::SegmentIter
LiveIntervalUnion::upperBound(SegmentIter SegBegin,
                              const LiveSegment &LS) {
  assert(LS.End > SegBegin->Start && "segment iterator precondition");

  // Get the next LIU segment such that segI->Start is not less than seg.Start
  //
  // FIXME: Once we have a B+tree, we can make good use of SegBegin as a hint to
  // upper_bound. For now, we're forced to search again from the root each time.
  SegmentIter SI = Segments.upper_bound(LS);
  while (SI != SegBegin) {
    --SI;
    if (LS.Start >= SI->End)
      return ++SI;
  }
  return SI;
}

// Merge a LiveInterval's segments. Guarantee no overlaps.
//
// After implementing B+tree, segments will be coalesced.
void LiveIntervalUnion::unify(LiveInterval &VirtReg) {

  // Insert each of the virtual register's live segments into the map.
  SegmentIter SegPos = Segments.begin();
  for (LiveInterval::iterator VirtRegI = VirtReg.begin(),
         VirtRegEnd = VirtReg.end();
       VirtRegI != VirtRegEnd; ++VirtRegI ) {

    LiveSegment Seg(*VirtRegI, &VirtReg);
    SegPos = Segments.insert(SegPos, Seg);

    assert(*SegPos == Seg && "need equal val for equal key");
#ifndef NDEBUG
    // Check for overlap (inductively).
    if (SegPos != Segments.begin()) {
      assert(llvm::prior(SegPos)->End <= Seg.Start && "overlapping segments" );
    }
    SegmentIter NextPos = llvm::next(SegPos);
    if (NextPos != Segments.end())
      assert(Seg.End <= NextPos->Start && "overlapping segments" );
#endif // NDEBUG
  }
}

// Remove a live virtual register's segments from this union.
void LiveIntervalUnion::extract(const LiveInterval &VirtReg) {

  // Remove each of the virtual register's live segments from the map.
  SegmentIter SegPos = Segments.begin();
  for (LiveInterval::const_iterator VirtRegI = VirtReg.begin(),
         VirtRegEnd = VirtReg.end();
       VirtRegI != VirtRegEnd; ++VirtRegI) {

    LiveSegment Seg(*VirtRegI, const_cast<LiveInterval*>(&VirtReg));
    SegPos = upperBound(SegPos, Seg);
    assert(SegPos != Segments.end() && "missing VirtReg segment");

    Segments.erase(SegPos++);
  }
}

raw_ostream& llvm::operator<<(raw_ostream& OS, const LiveSegment &LS) {
  return OS << '[' << LS.Start << ',' << LS.End << ':' <<
    LS.VirtReg->reg << ")";
}

void LiveSegment::dump() const {
  dbgs() << *this << "\n";
}

void
LiveIntervalUnion::print(raw_ostream &OS,
                         const AbstractRegisterDescription *RegDesc) const {
  OS << "LIU ";
  if (RegDesc != NULL)
    OS << RegDesc->getName(RepReg);
  else {
    OS << RepReg;
  }
  for (LiveSegments::const_iterator SI = Segments.begin(),
         SegEnd = Segments.end(); SI != SegEnd; ++SI) {
    dbgs() << " " << *SI;
  }
  OS << "\n";
}

void LiveIntervalUnion::dump(const AbstractRegisterDescription *RegDesc) const {
  print(dbgs(), RegDesc);
}

#ifndef NDEBUG
// Verify the live intervals in this union and add them to the visited set.
void LiveIntervalUnion::verify(LiveVirtRegBitSet& VisitedVRegs) {
  SegmentIter SI = Segments.begin();
  SegmentIter SegEnd = Segments.end();
  if (SI == SegEnd) return;
  VisitedVRegs.set(SI->VirtReg->reg);
  for (++SI; SI != SegEnd; ++SI) {
    VisitedVRegs.set(SI->VirtReg->reg);
    assert(llvm::prior(SI)->End <= SI->Start && "overlapping segments" );
  }
}
#endif //!NDEBUG

// Private interface accessed by Query.
//
// Find a pair of segments that intersect, one in the live virtual register
// (LiveInterval), and the other in this LiveIntervalUnion. The caller (Query)
// is responsible for advancing the LiveIntervalUnion segments to find a
// "notable" intersection, which requires query-specific logic.
//
// This design assumes only a fast mechanism for intersecting a single live
// virtual register segment with a set of LiveIntervalUnion segments.  This may
// be ok since most VIRTREGs have very few segments.  If we had a data
// structure that optimizd MxN intersection of segments, then we would bypass
// the loop that advances within the LiveInterval.
//
// If no intersection exists, set VirtRegI = VirtRegEnd, and set SI to the first
// segment whose start point is greater than LiveInterval's end point.
//
// Assumes that segments are sorted by start position in both
// LiveInterval and LiveSegments.
void LiveIntervalUnion::Query::findIntersection(InterferenceResult &IR) const {

  // Search until reaching the end of the LiveUnion segments.
  LiveInterval::iterator VirtRegEnd = VirtReg->end();
  SegmentIter LiveUnionEnd = LiveUnion->end();
  while (IR.LiveUnionI != LiveUnionEnd) {

    // Slowly advance the live virtual reg iterator until we surpass the next
    // segment in LiveUnion.
    //
    // Note: If this is ever used for coalescing of fixed registers and we have
    // a live vreg with thousands of segments, then change this code to use
    // upperBound instead.
    while (IR.VirtRegI != VirtRegEnd &&
           IR.VirtRegI->end <= IR.LiveUnionI->Start)
      ++IR.VirtRegI;
    if (IR.VirtRegI == VirtRegEnd)
      break; // Retain current (nonoverlapping) LiveUnionI

    // VirtRegI may have advanced far beyond LiveUnionI,
    // do a fast intersection test to "catch up"
    LiveSegment Seg(*IR.VirtRegI, VirtReg);
    IR.LiveUnionI = LiveUnion->upperBound(IR.LiveUnionI, Seg);

    // Check if no LiveUnionI exists with VirtRegI->Start < LiveUnionI.end
    if (IR.LiveUnionI == LiveUnionEnd)
      break;
    if (IR.LiveUnionI->Start < IR.VirtRegI->end) {
      assert(overlap(*IR.VirtRegI, *IR.LiveUnionI) &&
             "upperBound postcondition");
      break;
    }
  }
  if (IR.LiveUnionI == LiveUnionEnd)
    IR.VirtRegI = VirtRegEnd;
}

// Find the first intersection, and cache interference info
// (retain segment iterators into both VirtReg and LiveUnion).
LiveIntervalUnion::InterferenceResult
LiveIntervalUnion::Query::firstInterference() {
  if (FirstInterference != LiveIntervalUnion::InterferenceResult()) {
    return FirstInterference;
  }
  FirstInterference = InterferenceResult(VirtReg->begin(), LiveUnion->begin());
  findIntersection(FirstInterference);
  return FirstInterference;
}

// Treat the result as an iterator and advance to the next interfering pair
// of segments. This is a plain iterator with no filter.
bool LiveIntervalUnion::Query::nextInterference(InterferenceResult &IR) const {
  assert(isInterference(IR) && "iteration past end of interferences");

  // Advance either the VirtReg or LiveUnion segment to ensure that we visit all
  // unique overlapping pairs.
  if (IR.VirtRegI->end < IR.LiveUnionI->End) {
    if (++IR.VirtRegI == VirtReg->end())
      return false;
  }
  else {
    if (++IR.LiveUnionI == LiveUnion->end()) {
      IR.VirtRegI = VirtReg->end();
      return false;
    }
  }
  // Short-circuit findIntersection() if possible.
  if (overlap(*IR.VirtRegI, *IR.LiveUnionI))
    return true;

  // Find the next intersection.
  findIntersection(IR);
  return isInterference(IR);
}

// Scan the vector of interfering virtual registers in this union. Assume it's
// quite small.
bool LiveIntervalUnion::Query::isSeenInterference(LiveInterval *VirtReg) const {
  SmallVectorImpl<LiveInterval*>::const_iterator I =
    std::find(InterferingVRegs.begin(), InterferingVRegs.end(), VirtReg);
  return I != InterferingVRegs.end();
}

// Count the number of virtual registers in this union that interfere with this
// query's live virtual register.
//
// The number of times that we either advance IR.VirtRegI or call
// LiveUnion.upperBound() will be no more than the number of holes in
// VirtReg. So each invocation of collectInterferingVRegs() takes
// time proportional to |VirtReg Holes| * time(LiveUnion.upperBound()).
//
// For comments on how to speed it up, see Query::findIntersection().
unsigned LiveIntervalUnion::Query::
collectInterferingVRegs(unsigned MaxInterferingRegs) {
  InterferenceResult IR = firstInterference();
  LiveInterval::iterator VirtRegEnd = VirtReg->end();
  SegmentIter LiveUnionEnd = LiveUnion->end();
  LiveInterval *RecentInterferingVReg = NULL;
  while (IR.LiveUnionI != LiveUnionEnd) {
    // Advance the union's iterator to reach an unseen interfering vreg.
    do {
      if (IR.LiveUnionI->VirtReg == RecentInterferingVReg)
        continue;

      if (!isSeenInterference(IR.LiveUnionI->VirtReg))
        break;

      // Cache the most recent interfering vreg to bypass isSeenInterference.
      RecentInterferingVReg = IR.LiveUnionI->VirtReg;

    } while( ++IR.LiveUnionI != LiveUnionEnd);
    if (IR.LiveUnionI == LiveUnionEnd)
      break;

    // Advance the VirtReg iterator until surpassing the next segment in
    // LiveUnion.
    //
    // Note: If this is ever used for coalescing of fixed registers and we have
    // a live virtual register with thousands of segments, then use upperBound
    // instead.
    while (IR.VirtRegI != VirtRegEnd &&
           IR.VirtRegI->end <= IR.LiveUnionI->Start)
      ++IR.VirtRegI;
    if (IR.VirtRegI == VirtRegEnd)
      break;

    // Check for intersection with the union's segment.
    if (overlap(*IR.VirtRegI, *IR.LiveUnionI)) {

      if (!IR.LiveUnionI->VirtReg->isSpillable())
        SeenUnspillableVReg = true;

      InterferingVRegs.push_back(IR.LiveUnionI->VirtReg);
      if (InterferingVRegs.size() == MaxInterferingRegs)
        return MaxInterferingRegs;

      // Cache the most recent interfering vreg to bypass isSeenInterference.
      RecentInterferingVReg = IR.LiveUnionI->VirtReg;
      ++IR.LiveUnionI;
      continue;
    }
    // VirtRegI may have advanced far beyond LiveUnionI,
    // do a fast intersection test to "catch up"
    LiveSegment Seg(*IR.VirtRegI, VirtReg);
    IR.LiveUnionI = LiveUnion->upperBound(IR.LiveUnionI, Seg);
  }
  SeenAllInterferences = true;
  return InterferingVRegs.size();
}
