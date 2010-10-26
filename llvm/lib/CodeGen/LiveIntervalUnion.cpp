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
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
using namespace llvm;

// Merge a LiveInterval's segments. Guarantee no overlaps.
//
// Consider coalescing adjacent segments to save space, even though it makes
// extraction more complicated.
void LiveIntervalUnion::unify(LiveInterval &lvr) {
  // Insert each of the virtual register's live segments into the map
  SegmentIter segPos = segments_.begin();
  for (LiveInterval::iterator lvrI = lvr.begin(), lvrEnd = lvr.end();
       lvrI != lvrEnd; ++lvrI ) {
    LiveSegment segment(lvrI->start, lvrI->end, lvr);
    segPos = segments_.insert(segPos, segment);
    assert(*segPos == segment && "need equal val for equal key");
#ifndef NDEBUG
    // check for overlap (inductively)
    if (segPos != segments_.begin()) {
      SegmentIter prevPos = segPos;
      --prevPos;
      assert(prevPos->end <= segment.start && "overlapping segments" );
    }
    SegmentIter nextPos = segPos;
    ++nextPos;
    if (nextPos != segments_.end())
      assert(segment.end <= nextPos->start && "overlapping segments" );
#endif // NDEBUG
  }
}

// Low-level helper to find the first segment in the range [segI,segEnd) that
// intersects with a live virtual register segment, or segI.start >= lvr.end
//
// This logic is tied to the underlying LiveSegments data structure. For now, we
// use a binary search within the vector to find the nearest starting position,
// then reverse iterate to find the first overlap.
//
// Upon entry we have segI.start < lvrSeg.end
// seg   |--...
//        \   .
// lvr ...-|
// 
// After binary search, we have segI.start >= lvrSeg.start:
// seg   |--...
//      /
// lvr |--...
//
// Assuming intervals are disjoint, if an intersection exists, it must be the
// segment found or immediately behind it. We continue reverse iterating to
// return the first overlap.
typedef LiveIntervalUnion::SegmentIter SegmentIter;
static SegmentIter upperBound(SegmentIter segBegin,
                       SegmentIter segEnd,
                       const LiveRange &lvrSeg) {
  assert(lvrSeg.end > segBegin->start && "segment iterator precondition");
  // get the next LIU segment such that setg.start is not less than
  // lvrSeg.start
  SegmentIter segI = std::upper_bound(segBegin, segEnd, lvrSeg.start);
  while (segI != segBegin) {
    --segI;
    if (lvrSeg.start >= segI->end)
      return ++segI;
  }
  return segI;
}

// Private interface accessed by Query.
//
// Find a pair of segments that intersect, one in the live virtual register
// (LiveInterval), and the other in this LiveIntervalUnion. The caller (Query)
// is responsible for advancing the LiveIntervalUnion segments to find a
// "notable" intersection, which requires query-specific logic.
// 
// This design assumes only a fast mechanism for intersecting a single live
// virtual register segment with a set of LiveIntervalUnion segments.  This may
// be ok since most LVRs have very few segments.  If we had a data
// structure that optimizd MxN intersection of segments, then we would bypass
// the loop that advances within the LiveInterval.
//
// If no intersection exists, set lvrI = lvrEnd, and set segI to the first
// segment whose start point is greater than LiveInterval's end point.
//
// Assumes that segments are sorted by start position in both
// LiveInterval and LiveSegments.
void LiveIntervalUnion::Query::findIntersection(InterferenceResult &ir) const {
  LiveInterval::iterator lvrEnd = lvr_.end();
  SegmentIter liuEnd = liu_.end();
  while (ir.liuSegI_ != liuEnd) {
    // Slowly advance the live virtual reg iterator until we surpass the next
    // segment in this union. If this is ever used for coalescing of fixed
    // registers and we have a LiveInterval with thousands of segments, then use
    // upper bound instead.
    while (ir.lvrSegI_ != lvrEnd && ir.lvrSegI_->end <= ir.liuSegI_->start)
      ++ir.lvrSegI_;
    if (ir.lvrSegI_ == lvrEnd)
      break;
    // lvrSegI_ may have advanced far beyond liuSegI_,
    // do a fast intersection test to "catch up"
    ir.liuSegI_ = upperBound(ir.liuSegI_, liuEnd, *ir.lvrSegI_);
    // Check if no liuSegI_ exists with lvrSegI_->start < liuSegI_.end
    if (ir.liuSegI_ == liuEnd)
      break;
    if (ir.liuSegI_->start < ir.lvrSegI_->end) {
      assert(overlap(*ir.lvrSegI_, *ir.liuSegI_) && "upperBound postcondition");
      break;
    }
  }
  if (ir.liuSegI_ == liuEnd)
    ir.lvrSegI_ = lvrEnd;
}

// Find the first intersection, and cache interference info
// (retain segment iterators into both lvr_ and liu_).
LiveIntervalUnion::InterferenceResult
LiveIntervalUnion::Query::firstInterference() {
  if (firstInterference_ != LiveIntervalUnion::InterferenceResult()) {
    return firstInterference_;
  }
  firstInterference_ = InterferenceResult(lvr_.begin(), liu_.begin());
  findIntersection(firstInterference_);
  return firstInterference_;
}

// Treat the result as an iterator and advance to the next interfering pair
// of segments. This is a plain iterator with no filter.
bool LiveIntervalUnion::Query::nextInterference(InterferenceResult &ir) const {
  assert(isInterference(ir) && "iteration past end of interferences");
  // Advance either the lvr or liu segment to ensure that we visit all unique
  // overlapping pairs.
  if (ir.lvrSegI_->end < ir.liuSegI_->end) {
    if (++ir.lvrSegI_ == lvr_.end())
      return false;
  }
  else {
    if (++ir.liuSegI_ == liu_.end()) {
      ir.lvrSegI_ = lvr_.end();
      return false;
    }
  }
  if (overlap(*ir.lvrSegI_, *ir.liuSegI_))
    return true;
  // find the next intersection
  findIntersection(ir);
  return isInterference(ir);
}
