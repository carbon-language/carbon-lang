//===-- llvm/CodeGen/LiveInterval.h - Interval representation ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LiveRange and LiveInterval classes.  Given some
// numbering of each the machine instructions an interval [i, j) is said to be a
// live range for register v if there is no instruction with number j' >= j
// such that v is live at j' and there is no instruction with number i' < i such
// that v is live at i'. In this implementation ranges can have holes,
// i.e. a range might look like [1,20), [50,65), [1000,1001).  Each
// individual segment is represented as an instance of LiveRange::Segment,
// and the whole range is represented as an instance of LiveRange.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LIVEINTERVAL_H
#define LLVM_CODEGEN_LIVEINTERVAL_H

#include "llvm/ADT/IntEqClasses.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/Support/AlignOf.h"
#include "llvm/Support/Allocator.h"
#include <cassert>
#include <climits>

namespace llvm {
  class CoalescerPair;
  class LiveIntervals;
  class MachineInstr;
  class MachineRegisterInfo;
  class TargetRegisterInfo;
  class raw_ostream;
  template <typename T, unsigned Small> class SmallPtrSet;

  /// VNInfo - Value Number Information.
  /// This class holds information about a machine level values, including
  /// definition and use points.
  ///
  class VNInfo {
  public:
    typedef BumpPtrAllocator Allocator;

    /// The ID number of this value.
    unsigned id;

    /// The index of the defining instruction.
    SlotIndex def;

    /// VNInfo constructor.
    VNInfo(unsigned i, SlotIndex d)
      : id(i), def(d)
    { }

    /// VNInfo construtor, copies values from orig, except for the value number.
    VNInfo(unsigned i, const VNInfo &orig)
      : id(i), def(orig.def)
    { }

    /// Copy from the parameter into this VNInfo.
    void copyFrom(VNInfo &src) {
      def = src.def;
    }

    /// Returns true if this value is defined by a PHI instruction (or was,
    /// PHI instructions may have been eliminated).
    /// PHI-defs begin at a block boundary, all other defs begin at register or
    /// EC slots.
    bool isPHIDef() const { return def.isBlock(); }

    /// Returns true if this value is unused.
    bool isUnused() const { return !def.isValid(); }

    /// Mark this value as unused.
    void markUnused() { def = SlotIndex(); }
  };

  /// Result of a LiveRange query. This class hides the implementation details
  /// of live ranges, and it should be used as the primary interface for
  /// examining live ranges around instructions.
  class LiveQueryResult {
    VNInfo *const EarlyVal;
    VNInfo *const LateVal;
    const SlotIndex EndPoint;
    const bool Kill;

  public:
    LiveQueryResult(VNInfo *EarlyVal, VNInfo *LateVal, SlotIndex EndPoint,
                    bool Kill)
      : EarlyVal(EarlyVal), LateVal(LateVal), EndPoint(EndPoint), Kill(Kill)
    {}

    /// Return the value that is live-in to the instruction. This is the value
    /// that will be read by the instruction's use operands. Return NULL if no
    /// value is live-in.
    VNInfo *valueIn() const {
      return EarlyVal;
    }

    /// Return true if the live-in value is killed by this instruction. This
    /// means that either the live range ends at the instruction, or it changes
    /// value.
    bool isKill() const {
      return Kill;
    }

    /// Return true if this instruction has a dead def.
    bool isDeadDef() const {
      return EndPoint.isDead();
    }

    /// Return the value leaving the instruction, if any. This can be a
    /// live-through value, or a live def. A dead def returns NULL.
    VNInfo *valueOut() const {
      return isDeadDef() ? 0 : LateVal;
    }

    /// Return the value defined by this instruction, if any. This includes
    /// dead defs, it is the value created by the instruction's def operands.
    VNInfo *valueDefined() const {
      return EarlyVal == LateVal ? 0 : LateVal;
    }

    /// Return the end point of the last live range segment to interact with
    /// the instruction, if any.
    ///
    /// The end point is an invalid SlotIndex only if the live range doesn't
    /// intersect the instruction at all.
    ///
    /// The end point may be at or past the end of the instruction's basic
    /// block. That means the value was live out of the block.
    SlotIndex endPoint() const {
      return EndPoint;
    }
  };

  /// This class represents the liveness of a register, stack slot, etc.
  /// It manages an ordered list of Segment objects.
  /// The Segments are organized in a static single assignment form: At places
  /// where a new value is defined or different values reach a CFG join a new
  /// segment with a new value number is used.
  class LiveRange {
  public:

    /// This represents a simple continuous liveness interval for a value.
    /// The start point is inclusive, the end point exclusive. These intervals
    /// are rendered as [start,end).
    struct Segment {
      SlotIndex start;  // Start point of the interval (inclusive)
      SlotIndex end;    // End point of the interval (exclusive)
      VNInfo *valno;    // identifier for the value contained in this segment.

      Segment() : valno(0) {}

      Segment(SlotIndex S, SlotIndex E, VNInfo *V)
        : start(S), end(E), valno(V) {
        assert(S < E && "Cannot create empty or backwards segment");
      }

      /// Return true if the index is covered by this segment.
      bool contains(SlotIndex I) const {
        return start <= I && I < end;
      }

      /// Return true if the given interval, [S, E), is covered by this segment.
      bool containsInterval(SlotIndex S, SlotIndex E) const {
        assert((S < E) && "Backwards interval?");
        return (start <= S && S < end) && (start < E && E <= end);
      }

      bool operator<(const Segment &Other) const {
        return start < Other.start || (start == Other.start && end < Other.end);
      }
      bool operator==(const Segment &Other) const {
        return start == Other.start && end == Other.end;
      }

      void dump() const;
    };

    typedef SmallVector<Segment,4> Segments;
    typedef SmallVector<VNInfo*,4> VNInfoList;

    Segments segments;   // the liveness segments
    VNInfoList valnos;   // value#'s

    typedef Segments::iterator iterator;
    iterator begin() { return segments.begin(); }
    iterator end()   { return segments.end(); }

    typedef Segments::const_iterator const_iterator;
    const_iterator begin() const { return segments.begin(); }
    const_iterator end() const  { return segments.end(); }

    typedef VNInfoList::iterator vni_iterator;
    vni_iterator vni_begin() { return valnos.begin(); }
    vni_iterator vni_end()   { return valnos.end(); }

    typedef VNInfoList::const_iterator const_vni_iterator;
    const_vni_iterator vni_begin() const { return valnos.begin(); }
    const_vni_iterator vni_end() const   { return valnos.end(); }

    /// advanceTo - Advance the specified iterator to point to the Segment
    /// containing the specified position, or end() if the position is past the
    /// end of the range.  If no Segment contains this position, but the
    /// position is in a hole, this method returns an iterator pointing to the
    /// Segment immediately after the hole.
    iterator advanceTo(iterator I, SlotIndex Pos) {
      assert(I != end());
      if (Pos >= endIndex())
        return end();
      while (I->end <= Pos) ++I;
      return I;
    }

    /// find - Return an iterator pointing to the first segment that ends after
    /// Pos, or end(). This is the same as advanceTo(begin(), Pos), but faster
    /// when searching large ranges.
    ///
    /// If Pos is contained in a Segment, that segment is returned.
    /// If Pos is in a hole, the following Segment is returned.
    /// If Pos is beyond endIndex, end() is returned.
    iterator find(SlotIndex Pos);

    const_iterator find(SlotIndex Pos) const {
      return const_cast<LiveRange*>(this)->find(Pos);
    }

    void clear() {
      valnos.clear();
      segments.clear();
    }

    size_t size() const {
      return segments.size();
    }

    bool hasAtLeastOneValue() const { return !valnos.empty(); }

    bool containsOneValue() const { return valnos.size() == 1; }

    unsigned getNumValNums() const { return (unsigned)valnos.size(); }

    /// getValNumInfo - Returns pointer to the specified val#.
    ///
    inline VNInfo *getValNumInfo(unsigned ValNo) {
      return valnos[ValNo];
    }
    inline const VNInfo *getValNumInfo(unsigned ValNo) const {
      return valnos[ValNo];
    }

    /// containsValue - Returns true if VNI belongs to this range.
    bool containsValue(const VNInfo *VNI) const {
      return VNI && VNI->id < getNumValNums() && VNI == getValNumInfo(VNI->id);
    }

    /// getNextValue - Create a new value number and return it.  MIIdx specifies
    /// the instruction that defines the value number.
    VNInfo *getNextValue(SlotIndex def, VNInfo::Allocator &VNInfoAllocator) {
      VNInfo *VNI =
        new (VNInfoAllocator) VNInfo((unsigned)valnos.size(), def);
      valnos.push_back(VNI);
      return VNI;
    }

    /// createDeadDef - Make sure the range has a value defined at Def.
    /// If one already exists, return it. Otherwise allocate a new value and
    /// add liveness for a dead def.
    VNInfo *createDeadDef(SlotIndex Def, VNInfo::Allocator &VNInfoAllocator);

    /// Create a copy of the given value. The new value will be identical except
    /// for the Value number.
    VNInfo *createValueCopy(const VNInfo *orig,
                            VNInfo::Allocator &VNInfoAllocator) {
      VNInfo *VNI =
        new (VNInfoAllocator) VNInfo((unsigned)valnos.size(), *orig);
      valnos.push_back(VNI);
      return VNI;
    }

    /// RenumberValues - Renumber all values in order of appearance and remove
    /// unused values.
    void RenumberValues();

    /// MergeValueNumberInto - This method is called when two value numbers
    /// are found to be equivalent.  This eliminates V1, replacing all
    /// segments with the V1 value number with the V2 value number.  This can
    /// cause merging of V1/V2 values numbers and compaction of the value space.
    VNInfo* MergeValueNumberInto(VNInfo *V1, VNInfo *V2);

    /// Merge all of the live segments of a specific val# in RHS into this live
    /// range as the specified value number. The segments in RHS are allowed
    /// to overlap with segments in the current range, it will replace the
    /// value numbers of the overlaped live segments with the specified value
    /// number.
    void MergeSegmentsInAsValue(const LiveRange &RHS, VNInfo *LHSValNo);

    /// MergeValueInAsValue - Merge all of the segments of a specific val#
    /// in RHS into this live range as the specified value number.
    /// The segments in RHS are allowed to overlap with segments in the
    /// current range, but only if the overlapping segments have the
    /// specified value number.
    void MergeValueInAsValue(const LiveRange &RHS,
                             const VNInfo *RHSValNo, VNInfo *LHSValNo);

    bool empty() const { return segments.empty(); }

    /// beginIndex - Return the lowest numbered slot covered.
    SlotIndex beginIndex() const {
      assert(!empty() && "Call to beginIndex() on empty range.");
      return segments.front().start;
    }

    /// endNumber - return the maximum point of the range of the whole,
    /// exclusive.
    SlotIndex endIndex() const {
      assert(!empty() && "Call to endIndex() on empty range.");
      return segments.back().end;
    }

    bool expiredAt(SlotIndex index) const {
      return index >= endIndex();
    }

    bool liveAt(SlotIndex index) const {
      const_iterator r = find(index);
      return r != end() && r->start <= index;
    }

    /// Return the segment that contains the specified index, or null if there
    /// is none.
    const Segment *getSegmentContaining(SlotIndex Idx) const {
      const_iterator I = FindSegmentContaining(Idx);
      return I == end() ? 0 : &*I;
    }

    /// Return the live segment that contains the specified index, or null if
    /// there is none.
    Segment *getSegmentContaining(SlotIndex Idx) {
      iterator I = FindSegmentContaining(Idx);
      return I == end() ? 0 : &*I;
    }

    /// getVNInfoAt - Return the VNInfo that is live at Idx, or NULL.
    VNInfo *getVNInfoAt(SlotIndex Idx) const {
      const_iterator I = FindSegmentContaining(Idx);
      return I == end() ? 0 : I->valno;
    }

    /// getVNInfoBefore - Return the VNInfo that is live up to but not
    /// necessarilly including Idx, or NULL. Use this to find the reaching def
    /// used by an instruction at this SlotIndex position.
    VNInfo *getVNInfoBefore(SlotIndex Idx) const {
      const_iterator I = FindSegmentContaining(Idx.getPrevSlot());
      return I == end() ? 0 : I->valno;
    }

    /// Return an iterator to the segment that contains the specified index, or
    /// end() if there is none.
    iterator FindSegmentContaining(SlotIndex Idx) {
      iterator I = find(Idx);
      return I != end() && I->start <= Idx ? I : end();
    }

    const_iterator FindSegmentContaining(SlotIndex Idx) const {
      const_iterator I = find(Idx);
      return I != end() && I->start <= Idx ? I : end();
    }

    /// overlaps - Return true if the intersection of the two live ranges is
    /// not empty.
    bool overlaps(const LiveRange &other) const {
      if (other.empty())
        return false;
      return overlapsFrom(other, other.begin());
    }

    /// overlaps - Return true if the two ranges have overlapping segments
    /// that are not coalescable according to CP.
    ///
    /// Overlapping segments where one range is defined by a coalescable
    /// copy are allowed.
    bool overlaps(const LiveRange &Other, const CoalescerPair &CP,
                  const SlotIndexes&) const;

    /// overlaps - Return true if the live range overlaps an interval specified
    /// by [Start, End).
    bool overlaps(SlotIndex Start, SlotIndex End) const;

    /// overlapsFrom - Return true if the intersection of the two live ranges
    /// is not empty.  The specified iterator is a hint that we can begin
    /// scanning the Other range starting at I.
    bool overlapsFrom(const LiveRange &Other, const_iterator I) const;

    /// Add the specified Segment to this range, merging segments as
    /// appropriate.  This returns an iterator to the inserted segment (which
    /// may have grown since it was inserted).
    iterator addSegment(Segment S) {
      return addSegmentFrom(S, segments.begin());
    }

    /// extendInBlock - If this range is live before Kill in the basic block
    /// that starts at StartIdx, extend it to be live up to Kill, and return
    /// the value. If there is no segment before Kill, return NULL.
    VNInfo *extendInBlock(SlotIndex StartIdx, SlotIndex Kill);

    /// join - Join two live ranges (this, and other) together.  This applies
    /// mappings to the value numbers in the LHS/RHS ranges as specified.  If
    /// the ranges are not joinable, this aborts.
    void join(LiveRange &Other,
              const int *ValNoAssignments,
              const int *RHSValNoAssignments,
              SmallVectorImpl<VNInfo *> &NewVNInfo);

    /// True iff this segment is a single segment that lies between the
    /// specified boundaries, exclusively. Vregs live across a backedge are not
    /// considered local. The boundaries are expected to lie within an extended
    /// basic block, so vregs that are not live out should contain no holes.
    bool isLocal(SlotIndex Start, SlotIndex End) const {
      return beginIndex() > Start.getBaseIndex() &&
        endIndex() < End.getBoundaryIndex();
    }

    /// Remove the specified segment from this range.  Note that the segment
    /// must be a single Segment in its entirety.
    void removeSegment(SlotIndex Start, SlotIndex End,
                       bool RemoveDeadValNo = false);

    void removeSegment(Segment S, bool RemoveDeadValNo = false) {
      removeSegment(S.start, S.end, RemoveDeadValNo);
    }

    /// Query Liveness at Idx.
    /// The sub-instruction slot of Idx doesn't matter, only the instruction
    /// it refers to is considered.
    LiveQueryResult Query(SlotIndex Idx) const {
      // Find the segment that enters the instruction.
      const_iterator I = find(Idx.getBaseIndex());
      const_iterator E = end();
      if (I == E)
        return LiveQueryResult(0, 0, SlotIndex(), false);

      // Is this an instruction live-in segment?
      // If Idx is the start index of a basic block, include live-in segments
      // that start at Idx.getBaseIndex().
      VNInfo *EarlyVal = 0;
      VNInfo *LateVal  = 0;
      SlotIndex EndPoint;
      bool Kill = false;
      if (I->start <= Idx.getBaseIndex()) {
        EarlyVal = I->valno;
        EndPoint = I->end;
        // Move to the potentially live-out segment.
        if (SlotIndex::isSameInstr(Idx, I->end)) {
          Kill = true;
          if (++I == E)
            return LiveQueryResult(EarlyVal, LateVal, EndPoint, Kill);
        }
        // Special case: A PHIDef value can have its def in the middle of a
        // segment if the value happens to be live out of the layout
        // predecessor.
        // Such a value is not live-in.
        if (EarlyVal->def == Idx.getBaseIndex())
          EarlyVal = 0;
      }
      // I now points to the segment that may be live-through, or defined by
      // this instr. Ignore segments starting after the current instr.
      if (!SlotIndex::isEarlierInstr(Idx, I->start)) {
        LateVal = I->valno;
        EndPoint = I->end;
      }
      return LiveQueryResult(EarlyVal, LateVal, EndPoint, Kill);
    }

    /// removeValNo - Remove all the segments defined by the specified value#.
    /// Also remove the value# from value# list.
    void removeValNo(VNInfo *ValNo);

    /// Returns true if the live range is zero length, i.e. no live segments
    /// span instructions. It doesn't pay to spill such a range.
    bool isZeroLength(SlotIndexes *Indexes) const {
      for (const_iterator i = begin(), e = end(); i != e; ++i)
        if (Indexes->getNextNonNullIndex(i->start).getBaseIndex() <
            i->end.getBaseIndex())
          return false;
      return true;
    }

    bool operator<(const LiveRange& other) const {
      const SlotIndex &thisIndex = beginIndex();
      const SlotIndex &otherIndex = other.beginIndex();
      return thisIndex < otherIndex;
    }

    void print(raw_ostream &OS) const;
    void dump() const;

    /// \brief Walk the range and assert if any invariants fail to hold.
    ///
    /// Note that this is a no-op when asserts are disabled.
#ifdef NDEBUG
    void verify() const {}
#else
    void verify() const;
#endif

  private:

    iterator addSegmentFrom(Segment S, iterator From);
    void extendSegmentEndTo(iterator I, SlotIndex NewEnd);
    iterator extendSegmentStartTo(iterator I, SlotIndex NewStr);
    void markValNoForDeletion(VNInfo *V);

  };

  inline raw_ostream &operator<<(raw_ostream &OS, const LiveRange &LR) {
    LR.print(OS);
    return OS;
  }

  /// LiveInterval - This class represents the liveness of a register,
  /// or stack slot.
  class LiveInterval : public LiveRange {
  public:
    typedef LiveRange super;

    const unsigned reg;  // the register or stack slot of this interval.
    float weight;        // weight of this interval

    LiveInterval(unsigned Reg, float Weight)
      : reg(Reg), weight(Weight) {}

    /// getSize - Returns the sum of sizes of all the LiveRange's.
    ///
    unsigned getSize() const;

    /// isSpillable - Can this interval be spilled?
    bool isSpillable() const {
      return weight != llvm::huge_valf;
    }

    /// markNotSpillable - Mark interval as not spillable
    void markNotSpillable() {
      weight = llvm::huge_valf;
    }

    bool operator<(const LiveInterval& other) const {
      const SlotIndex &thisIndex = beginIndex();
      const SlotIndex &otherIndex = other.beginIndex();
      return thisIndex < otherIndex ||
              (thisIndex == otherIndex && reg < other.reg);
    }

    void print(raw_ostream &OS) const;
    void dump() const;

  private:
    LiveInterval& operator=(const LiveInterval& rhs) LLVM_DELETED_FUNCTION;

  };

  inline raw_ostream &operator<<(raw_ostream &OS, const LiveInterval &LI) {
    LI.print(OS);
    return OS;
  }

  raw_ostream &operator<<(raw_ostream &OS, const LiveRange::Segment &S);

  inline bool operator<(SlotIndex V, const LiveRange::Segment &S) {
    return V < S.start;
  }

  inline bool operator<(const LiveRange::Segment &S, SlotIndex V) {
    return S.start < V;
  }

  /// Helper class for performant LiveRange bulk updates.
  ///
  /// Calling LiveRange::addSegment() repeatedly can be expensive on large
  /// live ranges because segments after the insertion point may need to be
  /// shifted. The LiveRangeUpdater class can defer the shifting when adding
  /// many segments in order.
  ///
  /// The LiveRange will be in an invalid state until flush() is called.
  class LiveRangeUpdater {
    LiveRange *LR;
    SlotIndex LastStart;
    LiveRange::iterator WriteI;
    LiveRange::iterator ReadI;
    SmallVector<LiveRange::Segment, 16> Spills;
    void mergeSpills();

  public:
    /// Create a LiveRangeUpdater for adding segments to LR.
    /// LR will temporarily be in an invalid state until flush() is called.
    LiveRangeUpdater(LiveRange *lr = 0) : LR(lr) {}

    ~LiveRangeUpdater() { flush(); }

    /// Add a segment to LR and coalesce when possible, just like
    /// LR.addSegment(). Segments should be added in increasing start order for
    /// best performance.
    void add(LiveRange::Segment);

    void add(SlotIndex Start, SlotIndex End, VNInfo *VNI) {
      add(LiveRange::Segment(Start, End, VNI));
    }

    /// Return true if the LR is currently in an invalid state, and flush()
    /// needs to be called.
    bool isDirty() const { return LastStart.isValid(); }

    /// Flush the updater state to LR so it is valid and contains all added
    /// segments.
    void flush();

    /// Select a different destination live range.
    void setDest(LiveRange *lr) {
      if (LR != lr && isDirty())
        flush();
      LR = lr;
    }

    /// Get the current destination live range.
    LiveRange *getDest() const { return LR; }

    void dump() const;
    void print(raw_ostream&) const;
  };

  inline raw_ostream &operator<<(raw_ostream &OS, const LiveRangeUpdater &X) {
    X.print(OS);
    return OS;
  }

  /// ConnectedVNInfoEqClasses - Helper class that can divide VNInfos in a
  /// LiveInterval into equivalence clases of connected components. A
  /// LiveInterval that has multiple connected components can be broken into
  /// multiple LiveIntervals.
  ///
  /// Given a LiveInterval that may have multiple connected components, run:
  ///
  ///   unsigned numComps = ConEQ.Classify(LI);
  ///   if (numComps > 1) {
  ///     // allocate numComps-1 new LiveIntervals into LIS[1..]
  ///     ConEQ.Distribute(LIS);
  /// }

  class ConnectedVNInfoEqClasses {
    LiveIntervals &LIS;
    IntEqClasses EqClass;

    // Note that values a and b are connected.
    void Connect(unsigned a, unsigned b);

    unsigned Renumber();

  public:
    explicit ConnectedVNInfoEqClasses(LiveIntervals &lis) : LIS(lis) {}

    /// Classify - Classify the values in LI into connected components.
    /// Return the number of connected components.
    unsigned Classify(const LiveInterval *LI);

    /// getEqClass - Classify creates equivalence classes numbered 0..N. Return
    /// the equivalence class assigned the VNI.
    unsigned getEqClass(const VNInfo *VNI) const { return EqClass[VNI->id]; }

    /// Distribute - Distribute values in LIV[0] into a separate LiveInterval
    /// for each connected component. LIV must have a LiveInterval for each
    /// connected component. The LiveIntervals in Liv[1..] must be empty.
    /// Instructions using LIV[0] are rewritten.
    void Distribute(LiveInterval *LIV[], MachineRegisterInfo &MRI);

  };

}
#endif
