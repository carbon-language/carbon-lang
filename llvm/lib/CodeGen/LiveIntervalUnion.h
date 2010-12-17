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

#include "llvm/ADT/IntervalMap.h"
#include "llvm/CodeGen/LiveInterval.h"

#include <algorithm>

namespace llvm {

class MachineLoopRange;
class TargetRegisterInfo;

#ifndef NDEBUG
// forward declaration
template <unsigned Element> class SparseBitVector;
typedef SparseBitVector<128> LiveVirtRegBitSet;
#endif

/// Compare a live virtual register segment to a LiveIntervalUnion segment.
inline bool
overlap(const LiveRange &VRSeg,
        const IntervalMap<SlotIndex, LiveInterval*>::const_iterator &LUSeg) {
  return VRSeg.start < LUSeg.stop() && LUSeg.start() < VRSeg.end;
}

/// Union of live intervals that are strong candidates for coalescing into a
/// single register (either physical or virtual depending on the context).  We
/// expect the constituent live intervals to be disjoint, although we may
/// eventually make exceptions to handle value-based interference.
class LiveIntervalUnion {
  // A set of live virtual register segments that supports fast insertion,
  // intersection, and removal.
  // Mapping SlotIndex intervals to virtual register numbers.
  typedef IntervalMap<SlotIndex, LiveInterval*> LiveSegments;

public:
  // SegmentIter can advance to the next segment ordered by starting position
  // which may belong to a different live virtual register. We also must be able
  // to reach the current segment's containing virtual register.
  typedef LiveSegments::iterator SegmentIter;

  // LiveIntervalUnions share an external allocator.
  typedef LiveSegments::Allocator Allocator;

  class InterferenceResult;
  class Query;

private:
  const unsigned RepReg;  // representative register number
  LiveSegments Segments;  // union of virtual reg segments

public:
  LiveIntervalUnion(unsigned r, Allocator &a) : RepReg(r), Segments(a) {}

  // Iterate over all segments in the union of live virtual registers ordered
  // by their starting position.
  SegmentIter begin() { return Segments.begin(); }
  SegmentIter end() { return Segments.end(); }
  SegmentIter find(SlotIndex x) { return Segments.find(x); }
  bool empty() const { return Segments.empty(); }
  SlotIndex startIndex() const { return Segments.start(); }

  // Provide public access to the underlying map to allow overlap iteration.
  typedef LiveSegments Map;
  const Map &getMap() { return Segments; }

  // Add a live virtual register to this union and merge its segments.
  void unify(LiveInterval &VirtReg);

  // Remove a live virtual register's segments from this union.
  void extract(LiveInterval &VirtReg);

  // Print union, using TRI to translate register names
  void print(raw_ostream &OS, const TargetRegisterInfo *TRI) const;

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

    /// start - Return the start of the current overlap.
    SlotIndex start() const {
      return std::max(VirtRegI->start, LiveUnionI.start());
    }

    /// stop - Return the end of the current overlap.
    SlotIndex stop() const {
      return std::min(VirtRegI->end, LiveUnionI.stop());
    }

    /// interference - Return the register that is interfering here.
    LiveInterval *interference() const { return LiveUnionI.value(); }

    // Note: this interface provides raw access to the iterators because the
    // result has no way to tell if it's valid to dereference them.

    // Access the VirtReg segment.
    LiveInterval::iterator virtRegPos() const { return VirtRegI; }

    // Access the LiveUnion segment.
    const SegmentIter &liveUnionPos() const { return LiveUnionI; }

    bool operator==(const InterferenceResult &IR) const {
      return VirtRegI == IR.VirtRegI && LiveUnionI == IR.LiveUnionI;
    }
    bool operator!=(const InterferenceResult &IR) const {
      return !operator==(IR);
    }

    void print(raw_ostream &OS, const TargetRegisterInfo *TRI) const;
  };

  /// Query interferences between a single live virtual register and a live
  /// interval union.
  class Query {
    LiveIntervalUnion *LiveUnion;
    LiveInterval *VirtReg;
    InterferenceResult FirstInterference;
    SmallVector<LiveInterval*,4> InterferingVRegs;
    bool CheckedFirstInterference;
    bool SeenAllInterferences;
    bool SeenUnspillableVReg;

  public:
    Query(): LiveUnion(), VirtReg() {}

    Query(LiveInterval *VReg, LiveIntervalUnion *LIU):
      LiveUnion(LIU), VirtReg(VReg), CheckedFirstInterference(false),
      SeenAllInterferences(false), SeenUnspillableVReg(false)
    {}

    void clear() {
      LiveUnion = NULL;
      VirtReg = NULL;
      InterferingVRegs.clear();
      CheckedFirstInterference = false;
      SeenAllInterferences = false;
      SeenUnspillableVReg = false;
    }

    void init(LiveInterval *VReg, LiveIntervalUnion *LIU) {
      assert(VReg && LIU && "Invalid arguments");
      if (VirtReg == VReg && LiveUnion == LIU) {
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
        assert(overlap(*IR.VirtRegI, IR.LiveUnionI) &&
               "invalid segment iterators");
        return true;
      }
      return false;
    }

    // Does this live virtual register interfere with the union?
    bool checkInterference() { return isInterference(firstInterference()); }

    // Get the first pair of interfering segments, or a noninterfering result.
    // This initializes the firstInterference_ cache.
    const InterferenceResult &firstInterference();

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

    /// checkLoopInterference - Return true if there is interference overlapping
    /// Loop.
    bool checkLoopInterference(MachineLoopRange*);

    void print(raw_ostream &OS, const TargetRegisterInfo *TRI);
  private:
    Query(const Query&);          // DO NOT IMPLEMENT
    void operator=(const Query&); // DO NOT IMPLEMENT

    // Private interface for queries
    void findIntersection(InterferenceResult &IR) const;
  };
};

} // end namespace llvm

#endif // !defined(LLVM_CODEGEN_LIVEINTERVALUNION)
