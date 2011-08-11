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
#include "llvm/CodeGen/MachineLoopRanges.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetRegisterInfo.h"

using namespace llvm;


// Merge a LiveInterval's segments. Guarantee no overlaps.
void LiveIntervalUnion::unify(LiveInterval &VirtReg) {
  if (VirtReg.empty())
    return;
  ++Tag;

  // Insert each of the virtual register's live segments into the map.
  LiveInterval::iterator RegPos = VirtReg.begin();
  LiveInterval::iterator RegEnd = VirtReg.end();
  SegmentIter SegPos = Segments.find(RegPos->start);

  while (SegPos.valid()) {
    SegPos.insert(RegPos->start, RegPos->end, &VirtReg);
    if (++RegPos == RegEnd)
      return;
    SegPos.advanceTo(RegPos->start);
  }

  // We have reached the end of Segments, so it is no longer necessary to search
  // for the insertion position.
  // It is faster to insert the end first.
  --RegEnd;
  SegPos.insert(RegEnd->start, RegEnd->end, &VirtReg);
  for (; RegPos != RegEnd; ++RegPos, ++SegPos)
    SegPos.insert(RegPos->start, RegPos->end, &VirtReg);
}

// Remove a live virtual register's segments from this union.
void LiveIntervalUnion::extract(LiveInterval &VirtReg) {
  if (VirtReg.empty())
    return;
  ++Tag;

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
  OS << "LIU " << PrintReg(RepReg, TRI);
  if (empty()) {
    OS << " empty\n";
    return;
  }
  for (LiveSegments::const_iterator SI = Segments.begin(); SI.valid(); ++SI) {
    OS << " [" << SI.start() << ' ' << SI.stop() << "):"
       << PrintReg(SI.value()->reg, TRI);
  }
  OS << '\n';
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
void LiveIntervalUnion::Query::findIntersection() {
  // Search until reaching the end of the LiveUnion segments.
  LiveInterval::iterator VirtRegEnd = VirtReg->end();
  if (VirtRegI == VirtRegEnd)
    return;
  while (LiveUnionI.valid()) {
    // Slowly advance the live virtual reg iterator until we surpass the next
    // segment in LiveUnion.
    //
    // Note: If this is ever used for coalescing of fixed registers and we have
    // a live vreg with thousands of segments, then change this code to use
    // upperBound instead.
    VirtRegI = VirtReg->advanceTo(VirtRegI, LiveUnionI.start());
    if (VirtRegI == VirtRegEnd)
      break; // Retain current (nonoverlapping) LiveUnionI

    // VirtRegI may have advanced far beyond LiveUnionI, catch up.
    LiveUnionI.advanceTo(VirtRegI->start);

    // Check if no LiveUnionI exists with VirtRegI->Start < LiveUnionI.end
    if (!LiveUnionI.valid())
      break;
    if (LiveUnionI.start() < VirtRegI->end) {
      assert(overlap(*VirtRegI, LiveUnionI) && "upperBound postcondition");
      break;
    }
  }
  if (!LiveUnionI.valid())
    VirtRegI = VirtRegEnd;
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
  // Fast path return if we already have the desired information.
  if (SeenAllInterferences || InterferingVRegs.size() >= MaxInterferingRegs)
    return InterferingVRegs.size();

  // Set up iterators on the first call.
  if (!CheckedFirstInterference) {
    CheckedFirstInterference = true;
    LiveUnionI.setMap(LiveUnion->getMap());

    // Quickly skip interference check for empty sets.
    if (VirtReg->empty() || LiveUnion->empty()) {
      VirtRegI = VirtReg->end();
    } else if (VirtReg->beginIndex() < LiveUnion->startIndex()) {
      // VirtReg starts first, perform double binary search.
      VirtRegI = VirtReg->find(LiveUnion->startIndex());
      if (VirtRegI != VirtReg->end())
        LiveUnionI.find(VirtRegI->start);
    } else {
      // LiveUnion starts first, perform double binary search.
      LiveUnionI.find(VirtReg->beginIndex());
      if (LiveUnionI.valid())
        VirtRegI = VirtReg->find(LiveUnionI.start());
      else
        VirtRegI = VirtReg->end();
    }
    findIntersection();
    assert((VirtRegI == VirtReg->end() || LiveUnionI.valid())
           && "Uninitialized iterator");
  }

  LiveInterval::iterator VirtRegEnd = VirtReg->end();
  LiveInterval *RecentInterferingVReg = NULL;
  if (VirtRegI != VirtRegEnd) while (LiveUnionI.valid()) {
    // Advance the union's iterator to reach an unseen interfering vreg.
    do {
      if (LiveUnionI.value() == RecentInterferingVReg)
        continue;

      if (!isSeenInterference(LiveUnionI.value()))
        break;

      // Cache the most recent interfering vreg to bypass isSeenInterference.
      RecentInterferingVReg = LiveUnionI.value();

    } while ((++LiveUnionI).valid());
    if (!LiveUnionI.valid())
      break;

    // Advance the VirtReg iterator until surpassing the next segment in
    // LiveUnion.
    VirtRegI = VirtReg->advanceTo(VirtRegI, LiveUnionI.start());
    if (VirtRegI == VirtRegEnd)
      break;

    // Check for intersection with the union's segment.
    if (overlap(*VirtRegI, LiveUnionI)) {

      if (!LiveUnionI.value()->isSpillable())
        SeenUnspillableVReg = true;

      if (InterferingVRegs.size() == MaxInterferingRegs)
        // Leave SeenAllInterferences set to false to indicate that at least one
        // interference exists beyond those we collected.
        return MaxInterferingRegs;

      InterferingVRegs.push_back(LiveUnionI.value());

      // Cache the most recent interfering vreg to bypass isSeenInterference.
      RecentInterferingVReg = LiveUnionI.value();
      ++LiveUnionI;

      continue;
    }
    // VirtRegI may have advanced far beyond LiveUnionI,
    // do a fast intersection test to "catch up"
    LiveUnionI.advanceTo(VirtRegI->start);
  }
  SeenAllInterferences = true;
  return InterferingVRegs.size();
}

bool LiveIntervalUnion::Query::checkLoopInterference(MachineLoopRange *Loop) {
  // VirtReg is likely live throughout the loop, so start by checking LIU-Loop
  // overlaps.
  IntervalMapOverlaps<LiveIntervalUnion::Map, MachineLoopRange::Map>
    Overlaps(LiveUnion->getMap(), Loop->getMap());
  if (!Overlaps.valid())
    return false;

  // The loop is overlapping an LIU assignment. Check VirtReg as well.
  LiveInterval::iterator VRI = VirtReg->find(Overlaps.start());

  for (;;) {
    if (VRI == VirtReg->end())
      return false;
    if (VRI->start < Overlaps.stop())
      return true;

    Overlaps.advanceTo(VRI->start);
    if (!Overlaps.valid())
      return false;
    if (Overlaps.start() < VRI->end)
      return true;

    VRI = VirtReg->advanceTo(VRI, Overlaps.start());
  }
}
