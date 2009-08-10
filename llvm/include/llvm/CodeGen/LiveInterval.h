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
// live interval for register v if there is no instruction with number j' >= j
// such that v is live at j' and there is no instruction with number i' < i such
// that v is live at i'. In this implementation intervals can have holes,
// i.e. an interval might look like [1,20), [50,65), [1000,1001).  Each
// individual range is represented as an instance of LiveRange, and the whole
// interval is represented as an instance of LiveInterval.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LIVEINTERVAL_H
#define LLVM_CODEGEN_LIVEINTERVAL_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/AlignOf.h"
#include <iosfwd>
#include <cassert>
#include <climits>

namespace llvm {
  class MachineInstr;
  class MachineRegisterInfo;
  class TargetRegisterInfo;
  class raw_ostream;

  /// VNInfo - Value Number Information.
  /// This class holds information about a machine level values, including
  /// definition and use points.
  ///
  /// Care must be taken in interpreting the def index of the value. The 
  /// following rules apply:
  ///
  /// If the isDefAccurate() method returns false then def does not contain the
  /// index of the defining MachineInstr, or even (necessarily) to a
  /// MachineInstr at all. In general such a def index is not meaningful
  /// and should not be used. The exception is that, for values originally
  /// defined by PHI instructions, after PHI elimination def will contain the
  /// index of the MBB in which the PHI originally existed. This can be used
  /// to insert code (spills or copies) which deals with the value, which will
  /// be live in to the block.

  class VNInfo {
  private:
    enum {
      HAS_PHI_KILL    = 1,                         
      REDEF_BY_EC     = 1 << 1,
      IS_PHI_DEF      = 1 << 2,
      IS_UNUSED       = 1 << 3,
      IS_DEF_ACCURATE = 1 << 4
    };

    unsigned char flags;
    union {
      MachineInstr *copy;
      unsigned reg;
    } cr;

  public:
    /// Holds information about individual kills.
    struct KillInfo {
      bool isPHIKill : 1;
      unsigned killIdx : 31;

      KillInfo(bool isPHIKill, unsigned killIdx)
        : isPHIKill(isPHIKill), killIdx(killIdx) {

        assert(killIdx != 0 && "Zero kill indices are no longer permitted.");
      }

    };

    typedef SmallVector<KillInfo, 4> KillSet;

    /// The ID number of this value.
    unsigned id;
    
    /// The index of the defining instruction (if isDefAccurate() returns true).
    unsigned def;

    KillSet kills;

    VNInfo()
      : flags(IS_UNUSED), id(~1U), def(0) { cr.copy = 0; }

    /// VNInfo constructor.
    /// d is presumed to point to the actual defining instr. If it doesn't
    /// setIsDefAccurate(false) should be called after construction.
    VNInfo(unsigned i, unsigned d, MachineInstr *c)
      : flags(IS_DEF_ACCURATE), id(i), def(d) { cr.copy = c; }

    /// VNInfo construtor, copies values from orig, except for the value number.
    VNInfo(unsigned i, const VNInfo &orig)
      : flags(orig.flags), cr(orig.cr), id(i), def(orig.def), kills(orig.kills)
    { }

    /// Copy from the parameter into this VNInfo.
    void copyFrom(VNInfo &src) {
      flags = src.flags;
      cr = src.cr;
      def = src.def;
      kills = src.kills;
    }

    /// Used for copying value number info.
    unsigned getFlags() const { return flags; }
    void setFlags(unsigned flags) { this->flags = flags; }

    /// For a register interval, if this VN was definied by a copy instr
    /// getCopy() returns a pointer to it, otherwise returns 0.
    /// For a stack interval the behaviour of this method is undefined.
    MachineInstr* getCopy() const { return cr.copy; }
    /// For a register interval, set the copy member.
    /// This method should not be called on stack intervals as it may lead to
    /// undefined behavior.
    void setCopy(MachineInstr *c) { cr.copy = c; }
    
    /// For a stack interval, returns the reg which this stack interval was
    /// defined from.
    /// For a register interval the behaviour of this method is undefined. 
    unsigned getReg() const { return cr.reg; }
    /// For a stack interval, set the defining register.
    /// This method should not be called on register intervals as it may lead
    /// to undefined behaviour.
    void setReg(unsigned reg) { cr.reg = reg; }

    /// Returns true if one or more kills are PHI nodes.
    bool hasPHIKill() const { return flags & HAS_PHI_KILL; }
    void setHasPHIKill(bool hasKill) {
      if (hasKill)
        flags |= HAS_PHI_KILL;
      else
        flags &= ~HAS_PHI_KILL;
    }

    /// Returns true if this value is re-defined by an early clobber somewhere
    /// during the live range.
    bool hasRedefByEC() const { return flags & REDEF_BY_EC; }
    void setHasRedefByEC(bool hasRedef) {
      if (hasRedef)
        flags |= REDEF_BY_EC;
      else
        flags &= ~REDEF_BY_EC;
    }
   
    /// Returns true if this value is defined by a PHI instruction (or was,
    /// PHI instrucions may have been eliminated).
    bool isPHIDef() const { return flags & IS_PHI_DEF; }
    void setIsPHIDef(bool phiDef) {
      if (phiDef)
        flags |= IS_PHI_DEF;
      else
        flags &= ~IS_PHI_DEF;
    }

    /// Returns true if this value is unused.
    bool isUnused() const { return flags & IS_UNUSED; }
    void setIsUnused(bool unused) {
      if (unused)
        flags |= IS_UNUSED;
      else
        flags &= ~IS_UNUSED;
    }

    /// Returns true if the def is accurate.
    bool isDefAccurate() const { return flags & IS_DEF_ACCURATE; }
    void setIsDefAccurate(bool defAccurate) {
      if (defAccurate)
        flags |= IS_DEF_ACCURATE;
      else 
        flags &= ~IS_DEF_ACCURATE;
    }

  };

  inline bool operator<(const VNInfo::KillInfo &k1, const VNInfo::KillInfo &k2) {
    return k1.killIdx < k2.killIdx;
  }
  
  inline bool operator<(const VNInfo::KillInfo &k, unsigned idx) {
    return k.killIdx < idx;
  }

  inline bool operator<(unsigned idx, const VNInfo::KillInfo &k) {
    return idx < k.killIdx;
  }

  /// LiveRange structure - This represents a simple register range in the
  /// program, with an inclusive start point and an exclusive end point.
  /// These ranges are rendered as [start,end).
  struct LiveRange {
    unsigned start;  // Start point of the interval (inclusive)
    unsigned end;    // End point of the interval (exclusive)
    VNInfo *valno;   // identifier for the value contained in this interval.

    LiveRange(unsigned S, unsigned E, VNInfo *V) : start(S), end(E), valno(V) {
      assert(S < E && "Cannot create empty or backwards range");
    }

    /// contains - Return true if the index is covered by this range.
    ///
    bool contains(unsigned I) const {
      return start <= I && I < end;
    }

    bool operator<(const LiveRange &LR) const {
      return start < LR.start || (start == LR.start && end < LR.end);
    }
    bool operator==(const LiveRange &LR) const {
      return start == LR.start && end == LR.end;
    }

    void dump() const;
    void print(std::ostream &os) const;
    void print(std::ostream *os) const { if (os) print(*os); }
    void print(raw_ostream &os) const;
    void print(raw_ostream *os) const { if (os) print(*os); }

  private:
    LiveRange(); // DO NOT IMPLEMENT
  };

  std::ostream& operator<<(std::ostream& os, const LiveRange &LR);
  raw_ostream& operator<<(raw_ostream& os, const LiveRange &LR);


  inline bool operator<(unsigned V, const LiveRange &LR) {
    return V < LR.start;
  }

  inline bool operator<(const LiveRange &LR, unsigned V) {
    return LR.start < V;
  }

  /// LiveInterval - This class represents some number of live ranges for a
  /// register or value.  This class also contains a bit of register allocator
  /// state.
  class LiveInterval {
  public:

    typedef SmallVector<LiveRange,4> Ranges;
    typedef SmallVector<VNInfo*,4> VNInfoList;

    unsigned reg;        // the register or stack slot of this interval
                         // if the top bits is set, it represents a stack slot.
    float weight;        // weight of this interval
    Ranges ranges;       // the ranges in which this register is live
    VNInfoList valnos;   // value#'s
    
    struct InstrSlots {
      enum {
        LOAD  = 0,
        USE   = 1,
        DEF   = 2,
        STORE = 3,
        NUM   = 4
      };

      static unsigned scale(unsigned slot, unsigned factor) {
        unsigned index = slot / NUM,
                 offset = slot % NUM;
        assert(index <= ~0U / (factor * NUM) &&
               "Rescaled interval would overflow");
        return index * NUM * factor + offset;
      }

    };

    LiveInterval(unsigned Reg, float Weight, bool IsSS = false)
      : reg(Reg), weight(Weight) {
      if (IsSS)
        reg = reg | (1U << (sizeof(unsigned)*CHAR_BIT-1));
    }

    typedef Ranges::iterator iterator;
    iterator begin() { return ranges.begin(); }
    iterator end()   { return ranges.end(); }

    typedef Ranges::const_iterator const_iterator;
    const_iterator begin() const { return ranges.begin(); }
    const_iterator end() const  { return ranges.end(); }

    typedef VNInfoList::iterator vni_iterator;
    vni_iterator vni_begin() { return valnos.begin(); }
    vni_iterator vni_end() { return valnos.end(); }

    typedef VNInfoList::const_iterator const_vni_iterator;
    const_vni_iterator vni_begin() const { return valnos.begin(); }
    const_vni_iterator vni_end() const { return valnos.end(); }

    /// advanceTo - Advance the specified iterator to point to the LiveRange
    /// containing the specified position, or end() if the position is past the
    /// end of the interval.  If no LiveRange contains this position, but the
    /// position is in a hole, this method returns an iterator pointing the the
    /// LiveRange immediately after the hole.
    iterator advanceTo(iterator I, unsigned Pos) {
      if (Pos >= endNumber())
        return end();
      while (I->end <= Pos) ++I;
      return I;
    }
    
    void clear() {
      while (!valnos.empty()) {
        VNInfo *VNI = valnos.back();
        valnos.pop_back();
        VNI->~VNInfo();
      }
      
      ranges.clear();
    }

    /// isStackSlot - Return true if this is a stack slot interval.
    ///
    bool isStackSlot() const {
      return reg & (1U << (sizeof(unsigned)*CHAR_BIT-1));
    }

    /// getStackSlotIndex - Return stack slot index if this is a stack slot
    /// interval.
    int getStackSlotIndex() const {
      assert(isStackSlot() && "Interval is not a stack slot interval!");
      return reg & ~(1U << (sizeof(unsigned)*CHAR_BIT-1));
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

    /// getNextValue - Create a new value number and return it.  MIIdx specifies
    /// the instruction that defines the value number.
    VNInfo *getNextValue(unsigned MIIdx, MachineInstr *CopyMI,
                         bool isDefAccurate, BumpPtrAllocator &VNInfoAllocator) {

      assert(MIIdx != ~0u && MIIdx != ~1u &&
             "PHI def / unused flags should now be passed explicitly.");
      VNInfo *VNI =
        static_cast<VNInfo*>(VNInfoAllocator.Allocate((unsigned)sizeof(VNInfo),
                                                      alignof<VNInfo>()));
      new (VNI) VNInfo((unsigned)valnos.size(), MIIdx, CopyMI);
      VNI->setIsDefAccurate(isDefAccurate);
      valnos.push_back(VNI);
      return VNI;
    }

    /// Create a copy of the given value. The new value will be identical except
    /// for the Value number.
    VNInfo *createValueCopy(const VNInfo *orig, BumpPtrAllocator &VNInfoAllocator) {

      VNInfo *VNI =
        static_cast<VNInfo*>(VNInfoAllocator.Allocate((unsigned)sizeof(VNInfo),
                                                      alignof<VNInfo>()));
    
      new (VNI) VNInfo((unsigned)valnos.size(), *orig);
      valnos.push_back(VNI);
      return VNI;
    }

    /// addKill - Add a kill instruction index to the specified value
    /// number.
    static void addKill(VNInfo *VNI, unsigned KillIdx, bool phiKill) {
      VNInfo::KillSet &kills = VNI->kills;
      VNInfo::KillInfo newKill(phiKill, KillIdx);
      if (kills.empty()) {
        kills.push_back(newKill);
      } else {
        VNInfo::KillSet::iterator
          I = std::lower_bound(kills.begin(), kills.end(), newKill);
        kills.insert(I, newKill);
      }
    }

    /// addKills - Add a number of kills into the VNInfo kill vector. If this
    /// interval is live at a kill point, then the kill is not added.
    void addKills(VNInfo *VNI, const VNInfo::KillSet &kills) {
      for (unsigned i = 0, e = static_cast<unsigned>(kills.size());
           i != e; ++i) {
        const VNInfo::KillInfo &Kill = kills[i];
        if (!liveBeforeAndAt(Kill.killIdx)) {
          VNInfo::KillSet::iterator
            I = std::lower_bound(VNI->kills.begin(), VNI->kills.end(), Kill);
          VNI->kills.insert(I, Kill);
        }
      }
    }

    /// removeKill - Remove the specified kill from the list of kills of
    /// the specified val#.
    static bool removeKill(VNInfo *VNI, unsigned KillIdx) {
      VNInfo::KillSet &kills = VNI->kills;
      VNInfo::KillSet::iterator
        I = std::lower_bound(kills.begin(), kills.end(), KillIdx);
      if (I != kills.end() && I->killIdx == KillIdx) {
        kills.erase(I);
        return true;
      }
      return false;
    }

    /// removeKills - Remove all the kills in specified range
    /// [Start, End] of the specified val#.
    static void removeKills(VNInfo *VNI, unsigned Start, unsigned End) {
      VNInfo::KillSet &kills = VNI->kills;

      VNInfo::KillSet::iterator
        I = std::lower_bound(kills.begin(), kills.end(), Start);
      VNInfo::KillSet::iterator
        E = std::upper_bound(kills.begin(), kills.end(), End);
      kills.erase(I, E);
    }

    /// isKill - Return true if the specified index is a kill of the
    /// specified val#.
    static bool isKill(const VNInfo *VNI, unsigned KillIdx) {
      const VNInfo::KillSet &kills = VNI->kills;
      VNInfo::KillSet::const_iterator
        I = std::lower_bound(kills.begin(), kills.end(), KillIdx);
      return I != kills.end() && I->killIdx == KillIdx;
    }

    /// isOnlyLROfValNo - Return true if the specified live range is the only
    /// one defined by the its val#.
    bool isOnlyLROfValNo(const LiveRange *LR) {
      for (const_iterator I = begin(), E = end(); I != E; ++I) {
        const LiveRange *Tmp = I;
        if (Tmp != LR && Tmp->valno == LR->valno)
          return false;
      }
      return true;
    }
    
    /// MergeValueNumberInto - This method is called when two value nubmers
    /// are found to be equivalent.  This eliminates V1, replacing all
    /// LiveRanges with the V1 value number with the V2 value number.  This can
    /// cause merging of V1/V2 values numbers and compaction of the value space.
    VNInfo* MergeValueNumberInto(VNInfo *V1, VNInfo *V2);

    /// MergeInClobberRanges - For any live ranges that are not defined in the
    /// current interval, but are defined in the Clobbers interval, mark them
    /// used with an unknown definition value. Caller must pass in reference to
    /// VNInfoAllocator since it will create a new val#.
    void MergeInClobberRanges(const LiveInterval &Clobbers,
                              BumpPtrAllocator &VNInfoAllocator);

    /// MergeInClobberRange - Same as MergeInClobberRanges except it merge in a
    /// single LiveRange only.
    void MergeInClobberRange(unsigned Start, unsigned End,
                             BumpPtrAllocator &VNInfoAllocator);

    /// MergeValueInAsValue - Merge all of the live ranges of a specific val#
    /// in RHS into this live interval as the specified value number.
    /// The LiveRanges in RHS are allowed to overlap with LiveRanges in the
    /// current interval, it will replace the value numbers of the overlaped
    /// live ranges with the specified value number.
    void MergeRangesInAsValue(const LiveInterval &RHS, VNInfo *LHSValNo);

    /// MergeValueInAsValue - Merge all of the live ranges of a specific val#
    /// in RHS into this live interval as the specified value number.
    /// The LiveRanges in RHS are allowed to overlap with LiveRanges in the
    /// current interval, but only if the overlapping LiveRanges have the
    /// specified value number.
    void MergeValueInAsValue(const LiveInterval &RHS,
                             const VNInfo *RHSValNo, VNInfo *LHSValNo);

    /// Copy - Copy the specified live interval. This copies all the fields
    /// except for the register of the interval.
    void Copy(const LiveInterval &RHS, MachineRegisterInfo *MRI,
              BumpPtrAllocator &VNInfoAllocator);
    
    bool empty() const { return ranges.empty(); }

    /// beginNumber - Return the lowest numbered slot covered by interval.
    unsigned beginNumber() const {
      if (empty())
        return 0;
      return ranges.front().start;
    }

    /// endNumber - return the maximum point of the interval of the whole,
    /// exclusive.
    unsigned endNumber() const {
      if (empty())
        return 0;
      return ranges.back().end;
    }

    bool expiredAt(unsigned index) const {
      return index >= endNumber();
    }

    bool liveAt(unsigned index) const;

    // liveBeforeAndAt - Check if the interval is live at the index and the
    // index just before it. If index is liveAt, check if it starts a new live
    // range.If it does, then check if the previous live range ends at index-1.
    bool liveBeforeAndAt(unsigned index) const;

    /// getLiveRangeContaining - Return the live range that contains the
    /// specified index, or null if there is none.
    const LiveRange *getLiveRangeContaining(unsigned Idx) const {
      const_iterator I = FindLiveRangeContaining(Idx);
      return I == end() ? 0 : &*I;
    }

    /// getLiveRangeContaining - Return the live range that contains the
    /// specified index, or null if there is none.
    LiveRange *getLiveRangeContaining(unsigned Idx) {
      iterator I = FindLiveRangeContaining(Idx);
      return I == end() ? 0 : &*I;
    }

    /// FindLiveRangeContaining - Return an iterator to the live range that
    /// contains the specified index, or end() if there is none.
    const_iterator FindLiveRangeContaining(unsigned Idx) const;

    /// FindLiveRangeContaining - Return an iterator to the live range that
    /// contains the specified index, or end() if there is none.
    iterator FindLiveRangeContaining(unsigned Idx);

    /// findDefinedVNInfo - Find the VNInfo that's defined at the specified
    /// index (register interval) or defined by the specified register (stack
    /// inteval).
    VNInfo *findDefinedVNInfo(unsigned DefIdxOrReg) const;
    
    /// overlaps - Return true if the intersection of the two live intervals is
    /// not empty.
    bool overlaps(const LiveInterval& other) const {
      return overlapsFrom(other, other.begin());
    }

    /// overlaps - Return true if the live interval overlaps a range specified
    /// by [Start, End).
    bool overlaps(unsigned Start, unsigned End) const;

    /// overlapsFrom - Return true if the intersection of the two live intervals
    /// is not empty.  The specified iterator is a hint that we can begin
    /// scanning the Other interval starting at I.
    bool overlapsFrom(const LiveInterval& other, const_iterator I) const;

    /// addRange - Add the specified LiveRange to this interval, merging
    /// intervals as appropriate.  This returns an iterator to the inserted live
    /// range (which may have grown since it was inserted.
    void addRange(LiveRange LR) {
      addRangeFrom(LR, ranges.begin());
    }

    /// join - Join two live intervals (this, and other) together.  This applies
    /// mappings to the value numbers in the LHS/RHS intervals as specified.  If
    /// the intervals are not joinable, this aborts.
    void join(LiveInterval &Other, const int *ValNoAssignments,
              const int *RHSValNoAssignments,
              SmallVector<VNInfo*, 16> &NewVNInfo,
              MachineRegisterInfo *MRI);

    /// isInOneLiveRange - Return true if the range specified is entirely in the
    /// a single LiveRange of the live interval.
    bool isInOneLiveRange(unsigned Start, unsigned End);

    /// removeRange - Remove the specified range from this interval.  Note that
    /// the range must be a single LiveRange in its entirety.
    void removeRange(unsigned Start, unsigned End, bool RemoveDeadValNo = false);

    void removeRange(LiveRange LR, bool RemoveDeadValNo = false) {
      removeRange(LR.start, LR.end, RemoveDeadValNo);
    }

    /// removeValNo - Remove all the ranges defined by the specified value#.
    /// Also remove the value# from value# list.
    void removeValNo(VNInfo *ValNo);

    /// scaleNumbering - Renumber VNI and ranges to provide gaps for new
    /// instructions.
    void scaleNumbering(unsigned factor);

    /// getSize - Returns the sum of sizes of all the LiveRange's.
    ///
    unsigned getSize() const;

    /// ComputeJoinedWeight - Set the weight of a live interval after
    /// Other has been merged into it.
    void ComputeJoinedWeight(const LiveInterval &Other);

    bool operator<(const LiveInterval& other) const {
      return beginNumber() < other.beginNumber();
    }

    void print(std::ostream &OS, const TargetRegisterInfo *TRI = 0) const;
    void print(std::ostream *OS, const TargetRegisterInfo *TRI = 0) const {
      if (OS) print(*OS, TRI);
    }
    void print(raw_ostream &OS, const TargetRegisterInfo *TRI = 0) const;
    void print(raw_ostream *OS, const TargetRegisterInfo *TRI = 0) const {
      if (OS) print(*OS, TRI);
    }
    void dump() const;

  private:

    Ranges::iterator addRangeFrom(LiveRange LR, Ranges::iterator From);
    void extendIntervalEndTo(Ranges::iterator I, unsigned NewEnd);
    Ranges::iterator extendIntervalStartTo(Ranges::iterator I, unsigned NewStr);
    LiveInterval& operator=(const LiveInterval& rhs); // DO NOT IMPLEMENT

  };

  inline std::ostream &operator<<(std::ostream &OS, const LiveInterval &LI) {
    LI.print(OS);
    return OS;
  }
  inline raw_ostream &operator<<(raw_ostream &OS, const LiveInterval &LI) {
    LI.print(OS);
    return OS;
  }
}

#endif
