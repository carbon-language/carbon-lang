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
#include <set>

namespace llvm {

#ifndef NDEBUG
// forward declaration
template <unsigned Element> class SparseBitVector;
typedef SparseBitVector<128> LiveVirtRegBitSet;
#endif

/// A LiveSegment is a copy of a LiveRange object used within
/// LiveIntervalUnion. LiveSegment additionally contains a pointer to its
/// original live virtual register (LiveInterval). This allows quick lookup of
/// the live virtual register as we iterate over live segments in a union. Note
/// that LiveRange is misnamed and actually represents only a single contiguous
/// interval within a virtual register's liveness. To limit confusion, in this
/// file we refer it as a live segment.
///
/// Note: This currently represents a half-open interval [Start,End).
/// If LiveRange is modified to represent a closed interval, so should this.
struct LiveSegment {
  SlotIndex Start;
  SlotIndex End;
  LiveInterval *VirtReg;

  LiveSegment(const LiveRange& LR, LiveInterval *VReg)
    : Start(LR.start), End(LR.end), VirtReg(VReg) {}

  bool operator==(const LiveSegment &LS) const {
    return Start == LS.Start && End == LS.End && VirtReg == LS.VirtReg;
  }

  bool operator!=(const LiveSegment &LS) const {
    return !operator==(LS);
  }

  // Order segments by starting point only--we expect them to be disjoint.
  bool operator<(const LiveSegment &LS) const { return Start < LS.Start; }

  void dump() const;
  void print(raw_ostream &OS) const;
};

inline bool operator<(SlotIndex Idx, const LiveSegment &LS) {
  return Idx < LS.Start;
}

inline bool operator<(const LiveSegment &LS, SlotIndex Idx) {
  return LS.Start < Idx;
}

/// Compare a live virtual register segment to a LiveIntervalUnion segment.
inline bool overlap(const LiveRange &VirtRegSegment,
                    const LiveSegment &LiveUnionSegment) {
  return VirtRegSegment.start < LiveUnionSegment.End &&
    LiveUnionSegment.Start < VirtRegSegment.end;
}

template <> struct isPodLike<LiveSegment> { static const bool value = true; };

raw_ostream& operator<<(raw_ostream& OS, const LiveSegment &LS);

/// Abstraction to provide info for the representative register.
class AbstractRegisterDescription {
public:
  virtual const char *getName(unsigned Reg) const = 0;
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

public:
  // SegmentIter can advance to the next segment ordered by starting position
  // which may belong to a different live virtual register. We also must be able
  // to reach the current segment's containing virtual register.
  typedef LiveSegments::iterator SegmentIter;

  class InterferenceResult;
  class Query;

private:
  unsigned RepReg;        // representative register number
  LiveSegments Segments;  // union of virtual reg segements

public:
  // default ctor avoids placement new
  LiveIntervalUnion() : RepReg(0) {}

  // Initialize the union by associating it with a representative register
  // number.
  void init(unsigned Reg) { RepReg = Reg; }

  // Iterate over all segments in the union of live virtual registers ordered
  // by their starting position.
  SegmentIter begin() { return Segments.begin(); }
  SegmentIter end() { return Segments.end(); }

  // Return an iterator to the first segment after or including begin that
  // intersects with LS.
  SegmentIter upperBound(SegmentIter SegBegin, const LiveSegment &LS);

  // Add a live virtual register to this union and merge its segments.
  // Holds a nonconst reference to the VirtReg for later maniplution.
  void unify(LiveInterval &VirtReg);

  // Remove a live virtual register's segments from this union.
  void extract(const LiveInterval &VirtReg);

  void dump(const AbstractRegisterDescription *RegDesc) const;

  // If tri != NULL, use it to decode RepReg
  void print(raw_ostream &OS, const AbstractRegisterDescription *RegDesc) const;

#ifndef NDEBUG
  // Verify the live intervals in this union and add them to the visited set.
  void verify(LiveVirtRegBitSet& VisitedVRegs);
#endif

  /// Cache a single interference test result in the form of two intersecting
  /// segments. This allows efficiently iterating over the interferences. The
  /// iteration logic is handled by LiveIntervalUnion::Query which may
  /// filter interferences depending on the type of query.
  class InterferenceResult {
    friend class Query;

    LiveInterval::iterator VirtRegI; // current position in VirtReg
    SegmentIter LiveUnionI;          // current position in LiveUnion

    // Internal ctor.
    InterferenceResult(LiveInterval::iterator VRegI, SegmentIter UnionI)
      : VirtRegI(VRegI), LiveUnionI(UnionI) {}

  public:
    // Public default ctor.
    InterferenceResult(): VirtRegI(), LiveUnionI() {}

    // Note: this interface provides raw access to the iterators because the
    // result has no way to tell if it's valid to dereference them.

    // Access the VirtReg segment.
    LiveInterval::iterator virtRegPos() const { return VirtRegI; }

    // Access the LiveUnion segment.
    SegmentIter liveUnionPos() const { return LiveUnionI; }

    bool operator==(const InterferenceResult &IR) const {
      return VirtRegI == IR.VirtRegI && LiveUnionI == IR.LiveUnionI;
    }
    bool operator!=(const InterferenceResult &IR) const {
      return !operator==(IR);
    }
  };

  /// Query interferences between a single live virtual register and a live
  /// interval union.
  class Query {
    LiveIntervalUnion *LiveUnion;
    LiveInterval *VirtReg;
    InterferenceResult FirstInterference;
    SmallVector<LiveInterval*,4> InterferingVRegs;
    bool SeenAllInterferences;
    bool SeenUnspillableVReg;

  public:
    Query(): LiveUnion(), VirtReg() {}

    Query(LiveInterval *VReg, LiveIntervalUnion *LIU):
      LiveUnion(LIU), VirtReg(VReg), SeenAllInterferences(false),
      SeenUnspillableVReg(false)
    {}

    void clear() {
      LiveUnion = NULL;
      VirtReg = NULL;
      FirstInterference = InterferenceResult();
      InterferingVRegs.clear();
      SeenAllInterferences = false;
      SeenUnspillableVReg = false;
    }

    void init(LiveInterval *VReg, LiveIntervalUnion *LIU) {
      if (VirtReg == VReg) {
        // We currently allow query objects to be reused acrossed live virtual
        // registers, but always for the same live interval union.
        assert(LiveUnion == LIU && "inconsistent initialization");
        // Retain cached results, e.g. firstInterference.
        return;
      }
      clear();
      LiveUnion = LIU;
      VirtReg = VReg;
    }

    LiveInterval &virtReg() const {
      assert(VirtReg && "uninitialized");
      return *VirtReg;
    }

    bool isInterference(const InterferenceResult &IR) const {
      if (IR.VirtRegI != VirtReg->end()) {
        assert(overlap(*IR.VirtRegI, *IR.LiveUnionI) &&
               "invalid segment iterators");
        return true;
      }
      return false;
    }

    // Does this live virtual register interfere with the union?
    bool checkInterference() { return isInterference(firstInterference()); }

    // Get the first pair of interfering segments, or a noninterfering result.
    // This initializes the firstInterference_ cache.
    InterferenceResult firstInterference();

    // Treat the result as an iterator and advance to the next interfering pair
    // of segments. Visiting each unique interfering pairs means that the same
    // VirtReg or LiveUnion segment may be visited multiple times.
    bool nextInterference(InterferenceResult &IR) const;

    // Count the virtual registers in this union that interfere with this
    // query's live virtual register, up to maxInterferingRegs.
    unsigned collectInterferingVRegs(unsigned MaxInterferingRegs = UINT_MAX);

    // Was this virtual register visited during collectInterferingVRegs?
    bool isSeenInterference(LiveInterval *VReg) const;

    // Did collectInterferingVRegs collect all interferences?
    bool seenAllInterferences() const { return SeenAllInterferences; }

    // Did collectInterferingVRegs encounter an unspillable vreg?
    bool seenUnspillableVReg() const { return SeenUnspillableVReg; }

    // Vector generated by collectInterferingVRegs.
    const SmallVectorImpl<LiveInterval*> &interferingVRegs() const {
      return InterferingVRegs;
    }

  private:
    Query(const Query&);          // DO NOT IMPLEMENT
    void operator=(const Query&); // DO NOT IMPLEMENT

    // Private interface for queries
    void findIntersection(InterferenceResult &IR) const;
  };
};

} // end namespace llvm

#endif // !defined(LLVM_CODEGEN_LIVEINTERVALUNION)
