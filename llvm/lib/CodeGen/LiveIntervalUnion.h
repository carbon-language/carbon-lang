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

#ifndef NDEBUG
// forward declaration
template <unsigned Element> class SparseBitVector;
typedef SparseBitVector<128> LvrBitSet;
#endif

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

  LiveSegment(SlotIndex s, SlotIndex e, LiveInterval *lvr)
    : start(s), end(e), liveVirtReg(lvr) {}

  bool operator==(const LiveSegment &ls) const {
    return start == ls.start && end == ls.end && liveVirtReg == ls.liveVirtReg;
  }

  bool operator!=(const LiveSegment &ls) const {
    return !operator==(ls);
  }

  // Order segments by starting point only--we expect them to be disjoint.
  bool operator<(const LiveSegment &ls) const { return start < ls.start; }

  void dump() const;
  void print(raw_ostream &os) const;
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

template <> struct isPodLike<LiveSegment> { static const bool value = true; };

raw_ostream& operator<<(raw_ostream& os, const LiveSegment &ls);

/// Abstraction to provide info for the representative register.
class AbstractRegisterDescription {
public:
  virtual const char *getName(unsigned reg) const = 0;
  virtual ~AbstractRegisterDescription() {}
};

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

  // Return an iterator to the first segment after or including begin that
  // intersects with lvrSeg.
  SegmentIter upperBound(SegmentIter begin, const LiveSegment &seg);

  // Add a live virtual register to this union and merge its segments.
  // Holds a nonconst reference to the LVR for later maniplution.
  void unify(LiveInterval &lvr);

  // Remove a live virtual register's segments from this union.
  void extract(const LiveInterval &lvr);

  void dump(const AbstractRegisterDescription *regInfo) const;

  // If tri != NULL, use it to decode repReg_
  void print(raw_ostream &os, const AbstractRegisterDescription *rdesc) const;
  
#ifndef NDEBUG
  // Verify the live intervals in this union and add them to the visited set.
  void verify(LvrBitSet& visitedVRegs);
#endif

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
    LiveInterval::iterator lvrSegPos() const { return lvrSegI_; }

    // Access the liu segment.
    SegmentIter liuSegPos() const { return liuSegI_; }

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
    LiveIntervalUnion *liu_;
    LiveInterval *lvr_;
    InterferenceResult firstInterference_;
    SmallVector<LiveInterval*,4> interferingVRegs_;
    bool seenUnspillableVReg_;

  public:
    Query(): liu_(), lvr_() {}

    Query(LiveInterval *lvr, LiveIntervalUnion *liu):
      liu_(liu), lvr_(lvr), seenUnspillableVReg_(false) {}

    void clear() {
      liu_ = NULL;
      lvr_ = NULL;
      firstInterference_ = InterferenceResult();
      interferingVRegs_.clear();
      seenUnspillableVReg_ = false;
    }
    
    void init(LiveInterval *lvr, LiveIntervalUnion *liu) {
      if (lvr_ == lvr) {
        // We currently allow query objects to be reused acrossed live virtual
        // registers, but always for the same live interval union.
        assert(liu_ == liu && "inconsistent initialization");
        // Retain cached results, e.g. firstInterference.
        return;
      }
      liu_ = liu;
      lvr_ = lvr;
      // Clear cached results.
      firstInterference_ = InterferenceResult();
      interferingVRegs_.clear();
      seenUnspillableVReg_ = false;
    }

    LiveInterval &lvr() const { assert(lvr_ && "uninitialized"); return *lvr_; }

    bool isInterference(const InterferenceResult &ir) const {
      if (ir.lvrSegI_ != lvr_->end()) {
        assert(overlap(*ir.lvrSegI_, *ir.liuSegI_) &&
               "invalid segment iterators");
        return true;
      }
      return false;
    }

    // Does this live virtual register interfere with the union.
    bool checkInterference() { return isInterference(firstInterference()); }

    // Get the first pair of interfering segments, or a noninterfering result.
    // This initializes the firstInterference_ cache.
    InterferenceResult firstInterference();

    // Treat the result as an iterator and advance to the next interfering pair
    // of segments. Visiting each unique interfering pairs means that the same
    // lvr or liu segment may be visited multiple times.
    bool nextInterference(InterferenceResult &ir) const;

    // Count the virtual registers in this union that interfere with this
    // query's live virtual register, up to maxInterferingRegs.
    unsigned collectInterferingVRegs(unsigned maxInterferingRegs = UINT_MAX);

    // Was this virtual register visited during collectInterferingVRegs?
    bool isSeenInterference(LiveInterval *lvr) const;

    // Did collectInterferingVRegs encounter an unspillable vreg?
    bool seenUnspillableVReg() const {
      return seenUnspillableVReg_;
    }

    // Vector generated by collectInterferingVRegs.
    const SmallVectorImpl<LiveInterval*> &interferingVRegs() const {
      return interferingVRegs_;
    }
    
  private:
    Query(const Query&);          // DO NOT IMPLEMENT
    void operator=(const Query&); // DO NOT IMPLEMENT
    
    // Private interface for queries
    void findIntersection(InterferenceResult &ir) const;
  };
};

} // end namespace llvm

#endif // !defined(LLVM_CODEGEN_LIVEINTERVALUNION)
