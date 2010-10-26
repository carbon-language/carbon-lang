//===-- LiveIntervalUnion.h - Live interval union data struct --*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// LiveIntervalUnion is a union of live segments across multiple live virtual
// registers. This may be used during coalescing to represent a congruence
// class, or during register allocation to model liveness of a physical
// register.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LIVEINTERVALUNION
#define LLVM_CODEGEN_LIVEINTERVALUNION

#include "llvm/CodeGen/LiveInterval.h"
#include <vector>
#include <set>

namespace llvm {

/// A LiveSegment is a copy of a LiveRange object used within
/// LiveIntervalUnion. LiveSegment additionally contains a pointer to its
/// original live virtual register (LiveInterval). This allows quick lookup of
/// the live virtual register as we iterate over live segments in a union. Note
/// that LiveRange is misnamed and actually represents only a single contiguous
/// interval within a virtual register's liveness. To limit confusion, in this
/// file we refer it as a live segment.
///
/// Note: This currently represents a half-open interval [start,end).
/// If LiveRange is modified to represent a closed interval, so should this.
struct LiveSegment {
  SlotIndex start;
  SlotIndex end;
  LiveInterval *liveVirtReg;

  LiveSegment(SlotIndex s, SlotIndex e, LiveInterval &lvr)
    : start(s), end(e), liveVirtReg(&lvr) {}

  bool operator==(const LiveSegment &ls) const {
    return start == ls.start && end == ls.end && liveVirtReg == ls.liveVirtReg;
  }

  bool operator!=(const LiveSegment &ls) const {
    return !operator==(ls);
  }

  // Order segments by starting point only--we expect them to be disjoint.
  bool operator<(const LiveSegment &ls) const { return start < ls.start; }
};

inline bool operator<(SlotIndex V, const LiveSegment &ls) {
  return V < ls.start;
}

inline bool operator<(const LiveSegment &ls, SlotIndex V) {
  return ls.start < V;
}

/// Compare a live virtual register segment to a LiveIntervalUnion segment.
inline bool overlap(const LiveRange &lvrSeg, const LiveSegment &liuSeg) {
  return lvrSeg.start < liuSeg.end && liuSeg.start < lvrSeg.end;
}

/// Union of live intervals that are strong candidates for coalescing into a
/// single register (either physical or virtual depending on the context).  We
/// expect the constituent live intervals to be disjoint, although we may
/// eventually make exceptions to handle value-based interference.
class LiveIntervalUnion {
  // A set of live virtual register segments that supports fast insertion,
  // intersection, and removal. 
  //
  // FIXME: std::set is a placeholder until we decide how to
  // efficiently represent it. Probably need to roll our own B-tree.
  typedef std::set<LiveSegment> LiveSegments;

  // A set of live virtual registers. Elements have type LiveInterval, where
  // each element represents the liveness of a single live virtual register.
  // This is traditionally known as a live range, but we refer is as a live
  // virtual register to avoid confusing it with the misnamed LiveRange
  // class.
  typedef std::vector<LiveInterval*> LiveVirtRegs;

public:
  // SegmentIter can advance to the next segment ordered by starting position
  // which may belong to a different live virtual register. We also must be able
  // to reach the current segment's containing virtual register.
  typedef LiveSegments::iterator SegmentIter;

  class InterferenceResult;
  class Query;

private:
  unsigned repReg_;        // representative register number
  LiveSegments segments_;  // union of virtual reg segements

public:
  // default ctor avoids placement new
  LiveIntervalUnion() : repReg_(0) {}

  // Initialize the union by associating it with a representative register
  // number.
  void init(unsigned repReg) { repReg_ = repReg; }

  // Iterate over all segments in the union of live virtual registers ordered
  // by their starting position.
  SegmentIter begin() { return segments_.begin(); }
  SegmentIter end() { return segments_.end(); }

  // Add a live virtual register to this union and merge its segments.
  // Holds a nonconst reference to the LVR for later maniplution.
  void unify(LiveInterval &lvr);

  // FIXME: needed by RegAllocGreedy
  //void extract(const LiveInterval &li);

  /// Cache a single interference test result in the form of two intersecting
  /// segments. This allows efficiently iterating over the interferences. The
  /// iteration logic is handled by LiveIntervalUnion::Query which may
  /// filter interferences depending on the type of query.
  class InterferenceResult {
    friend class Query;

    LiveInterval::iterator lvrSegI_; // current position in _lvr
    SegmentIter liuSegI_;            // current position in _liu
    
    // Internal ctor.
    InterferenceResult(LiveInterval::iterator lvrSegI, SegmentIter liuSegI)
      : lvrSegI_(lvrSegI), liuSegI_(liuSegI) {}

  public:
    // Public default ctor.
    InterferenceResult(): lvrSegI_(), liuSegI_() {}

    // Note: this interface provides raw access to the iterators because the
    // result has no way to tell if it's valid to dereference them.

    // Access the lvr segment. 
    const LiveInterval::iterator &lvrSegPos() const { return lvrSegI_; }

    // Access the liu segment.
    const SegmentIter &liuSeg() const { return liuSegI_; }

    bool operator==(const InterferenceResult &ir) const {
      return lvrSegI_ == ir.lvrSegI_ && liuSegI_ == ir.liuSegI_;
    }
    bool operator!=(const InterferenceResult &ir) const {
      return !operator==(ir);
    }
  };

  /// Query interferences between a single live virtual register and a live
  /// interval union.
  class Query {
    LiveIntervalUnion &liu_;
    LiveInterval &lvr_;
    InterferenceResult firstInterference_;
    // TBD: interfering vregs

  public:
    Query(LiveInterval &lvr, LiveIntervalUnion &liu): liu_(liu), lvr_(lvr) {}

    LiveInterval &lvr() const { return lvr_; }

    bool isInterference(const InterferenceResult &ir) const {
      if (ir.lvrSegI_ != lvr_.end()) {
        assert(overlap(*ir.lvrSegI_, *ir.liuSegI_) &&
               "invalid segment iterators");
        return true;
      }
      return false;
    }

    // Does this live virtual register interfere with the union.
    bool checkInterference() { return isInterference(firstInterference()); }

    // First pair of interfering segments, or a noninterfering result.
    InterferenceResult firstInterference();

    // Treat the result as an iterator and advance to the next interfering pair
    // of segments. Visiting each unique interfering pairs means that the same
    // lvr or liu segment may be visited multiple times.
    bool nextInterference(InterferenceResult &ir) const;
        
    // TBD: bool collectInterferingVirtRegs(unsigned maxInterference)

  private:
    // Private interface for queries
    void findIntersection(InterferenceResult &ir) const;
  };
};

} // end namespace llvm

#endif // !defined(LLVM_CODEGEN_LIVEINTERVALUNION)
