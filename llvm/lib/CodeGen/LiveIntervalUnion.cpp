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
#include "llvm/Target/TargetRegisterInfo.h"

#include <algorithm>

using namespace llvm;


// Merge a LiveInterval's segments. Guarantee no overlaps.
void LiveIntervalUnion::unify(LiveInterval &VirtReg) {
  if (VirtReg.empty())
    return;

  // Insert each of the virtual register's live segments into the map.
  LiveInterval::iterator RegPos = VirtReg.begin();
  LiveInterval::iterator RegEnd = VirtReg.end();
  SegmentIter SegPos = Segments.find(RegPos->start);

  for (;;) {
    SegPos.insert(RegPos->start, RegPos->end, &VirtReg);
    if (++RegPos == RegEnd)
      return;
    SegPos.advanceTo(RegPos->start);
  }
}

// Remove a live virtual register's segments from this union.
void LiveIntervalUnion::extract(LiveInterval &VirtReg) {
  if (VirtReg.empty())
    return;

  // Remove each of the virtual register's live segments from the map.
  LiveInterval::iterator RegPos = VirtReg.begin();
  LiveInterval::iterator RegEnd = VirtReg.end();
  SegmentIter SegPos = Segments.find(RegPos->start);

  for (;;) {
    assert(SegPos.value() == &VirtReg && "Inconsistent LiveInterval");
    SegPos.erase();
    if (!SegPos.valid())
      return;

    // Skip all segments that may have been coalesced.
    RegPos = VirtReg.advanceTo(RegPos, SegPos.start());
    if (RegPos == RegEnd)
      return;

    SegPos.advanceTo(RegPos->start);
  }
}

void
LiveIntervalUnion::print(raw_ostream &OS, const TargetRegisterInfo *TRI) const {
  OS << "LIU ";
  TRI->printReg(RepReg, OS);
  for (LiveSegments::const_iterator SI = Segments.begin(); SI.valid(); ++SI) {
    OS << " [" << SI.start() << ' ' << SI.stop() << "):";
    TRI->printReg(SI.value()->reg, OS);
  }
  OS << "\n";
}

#ifndef NDEBUG
// Verify the live intervals in this union and add them to the visited set.
void LiveIntervalUnion::verify(LiveVirtRegBitSet& VisitedVRegs) {
  for (SegmentIter SI = Segments.begin(); SI.valid(); ++SI)
    VisitedVRegs.set(SI.value()->reg);
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
// be ok since most virtual registers have very few segments.  If we had a data
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
  if (IR.VirtRegI == VirtRegEnd)
    return;
  while (IR.LiveUnionI.valid()) {
    // Slowly advance the live virtual reg iterator until we surpass the next
    // segment in LiveUnion.
    //
    // Note: If this is ever used for coalescing of fixed registers and we have
    // a live vreg with thousands of segments, then change this code to use
    // upperBound instead.
    IR.VirtRegI = VirtReg->advanceTo(IR.VirtRegI, IR.LiveUnionI.start());
    if (IR.VirtRegI == VirtRegEnd)
      break; // Retain current (nonoverlapping) LiveUnionI

    // VirtRegI may have advanced far beyond LiveUnionI, catch up.
    IR.LiveUnionI.advanceTo(IR.VirtRegI->start);

    // Check if no LiveUnionI exists with VirtRegI->Start < LiveUnionI.end
    if (!IR.LiveUnionI.valid())
      break;
    if (IR.LiveUnionI.start() < IR.VirtRegI->end) {
      assert(overlap(*IR.VirtRegI, IR.LiveUnionI) &&
             "upperBound postcondition");
      break;
    }
  }
  if (!IR.LiveUnionI.valid())
    IR.VirtRegI = VirtRegEnd;
}

// Find the first intersection, and cache interference info
// (retain segment iterators into both VirtReg and LiveUnion).
const LiveIntervalUnion::InterferenceResult &
LiveIntervalUnion::Query::firstInterference() {
  if (CheckedFirstInterference)
    return FirstInterference;
  CheckedFirstInterference = true;
  InterferenceResult &IR = FirstInterference;

  // Quickly skip interference check for empty sets.
  if (VirtReg->empty() || LiveUnion->empty()) {
    IR.VirtRegI = VirtReg->end();
  } else if (VirtReg->beginIndex() < LiveUnion->startIndex()) {
    // VirtReg starts first, perform double binary search.
    IR.VirtRegI = VirtReg->find(LiveUnion->startIndex());
    if (IR.VirtRegI != VirtReg->end())
      IR.LiveUnionI = LiveUnion->find(IR.VirtRegI->start);
  } else {
    // LiveUnion starts first, perform double binary search.
    IR.LiveUnionI = LiveUnion->find(VirtReg->beginIndex());
    if (IR.LiveUnionI.valid())
      IR.VirtRegI = VirtReg->find(IR.LiveUnionI.start());
    else
      IR.VirtRegI = VirtReg->end();
  }
  findIntersection(FirstInterference);
  assert((IR.VirtRegI == VirtReg->end() || IR.LiveUnionI.valid())
         && "Uninitialized iterator");
  return FirstInterference;
}

// Treat the result as an iterator and advance to the next interfering pair
// of segments. This is a plain iterator with no filter.
bool LiveIntervalUnion::Query::nextInterference(InterferenceResult &IR) const {
  assert(isInterference(IR) && "iteration past end of interferences");

  // Advance either the VirtReg or LiveUnion segment to ensure that we visit all
  // unique overlapping pairs.
  if (IR.VirtRegI->end < IR.LiveUnionI.stop()) {
    if (++IR.VirtRegI == VirtReg->end())
      return false;
  }
  else {
    if (!(++IR.LiveUnionI).valid()) {
      IR.VirtRegI = VirtReg->end();
      return false;
    }
  }
  // Short-circuit findIntersection() if possible.
  if (overlap(*IR.VirtRegI, IR.LiveUnionI))
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
  LiveInterval *RecentInterferingVReg = NULL;
  while (IR.LiveUnionI.valid()) {
    // Advance the union's iterator to reach an unseen interfering vreg.
    do {
      if (IR.LiveUnionI.value() == RecentInterferingVReg)
        continue;

      if (!isSeenInterference(IR.LiveUnionI.value()))
        break;

      // Cache the most recent interfering vreg to bypass isSeenInterference.
      RecentInterferingVReg = IR.LiveUnionI.value();

    } while ((++IR.LiveUnionI).valid());
    if (!IR.LiveUnionI.valid())
      break;

    // Advance the VirtReg iterator until surpassing the next segment in
    // LiveUnion.
    IR.VirtRegI = VirtReg->advanceTo(IR.VirtRegI, IR.LiveUnionI.start());
    if (IR.VirtRegI == VirtRegEnd)
      break;

    // Check for intersection with the union's segment.
    if (overlap(*IR.VirtRegI, IR.LiveUnionI)) {

      if (!IR.LiveUnionI.value()->isSpillable())
        SeenUnspillableVReg = true;

      if (InterferingVRegs.size() == MaxInterferingRegs)
        // Leave SeenAllInterferences set to false to indicate that at least one
        // interference exists beyond those we collected.
        return MaxInterferingRegs;

      InterferingVRegs.push_back(IR.LiveUnionI.value());

      // Cache the most recent interfering vreg to bypass isSeenInterference.
      RecentInterferingVReg = IR.LiveUnionI.value();
      ++IR.LiveUnionI;
      continue;
    }
    // VirtRegI may have advanced far beyond LiveUnionI,
    // do a fast intersection test to "catch up"
    IR.LiveUnionI.advanceTo(IR.VirtRegI->start);
  }
  SeenAllInterferences = true;
  return InterferingVRegs.size();
}
