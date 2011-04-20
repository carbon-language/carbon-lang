//===---- LiveRangeEdit.h - Basic tools for split and spill -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The LiveRangeEdit class represents changes done to a virtual register when it
// is spilled or split.
//
// The parent register is never changed. Instead, a number of new virtual
// registers are created and added to the newRegs vector.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LIVERANGEEDIT_H
#define LLVM_CODEGEN_LIVERANGEEDIT_H

#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace llvm {

class AliasAnalysis;
class LiveIntervals;
class MachineLoopInfo;
class MachineRegisterInfo;
class VirtRegMap;

class LiveRangeEdit {
public:
  /// Callback methods for LiveRangeEdit owners.
  struct Delegate {
    /// Called immediately before erasing a dead machine instruction.
    virtual void LRE_WillEraseInstruction(MachineInstr *MI) {}

    /// Called when a virtual register is no longer used. Return false to defer
    /// its deletion from LiveIntervals.
    virtual bool LRE_CanEraseVirtReg(unsigned) { return true; }

    /// Called before shrinking the live range of a virtual register.
    virtual void LRE_WillShrinkVirtReg(unsigned) {}

    /// Called after cloning a virtual register.
    /// This is used for new registers representing connected components of Old.
    virtual void LRE_DidCloneVirtReg(unsigned New, unsigned Old) {}

    virtual ~Delegate() {}
  };

private:
  LiveInterval &parent_;
  SmallVectorImpl<LiveInterval*> &newRegs_;
  Delegate *const delegate_;
  const SmallVectorImpl<LiveInterval*> *uselessRegs_;

  /// firstNew_ - Index of the first register added to newRegs_.
  const unsigned firstNew_;

  /// scannedRemattable_ - true when remattable values have been identified.
  bool scannedRemattable_;

  /// remattable_ - Values defined by remattable instructions as identified by
  /// tii.isTriviallyReMaterializable().
  SmallPtrSet<const VNInfo*,4> remattable_;

  /// rematted_ - Values that were actually rematted, and so need to have their
  /// live range trimmed or entirely removed.
  SmallPtrSet<const VNInfo*,4> rematted_;

  /// scanRemattable - Identify the parent_ values that may rematerialize.
  void scanRemattable(LiveIntervals &lis,
                      const TargetInstrInfo &tii,
                      AliasAnalysis *aa);

  /// allUsesAvailableAt - Return true if all registers used by OrigMI at
  /// OrigIdx are also available with the same value at UseIdx.
  bool allUsesAvailableAt(const MachineInstr *OrigMI, SlotIndex OrigIdx,
                          SlotIndex UseIdx, LiveIntervals &lis);

  /// foldAsLoad - If LI has a single use and a single def that can be folded as
  /// a load, eliminate the register by folding the def into the use.
  bool foldAsLoad(LiveInterval *LI, SmallVectorImpl<MachineInstr*> &Dead,
                  MachineRegisterInfo&, LiveIntervals&, const TargetInstrInfo&);

public:
  /// Create a LiveRangeEdit for breaking down parent into smaller pieces.
  /// @param parent The register being spilled or split.
  /// @param newRegs List to receive any new registers created. This needn't be
  ///                empty initially, any existing registers are ignored.
  /// @param uselessRegs List of registers that can't be used when
  ///        rematerializing values because they are about to be removed.
  LiveRangeEdit(LiveInterval &parent,
                SmallVectorImpl<LiveInterval*> &newRegs,
                Delegate *delegate = 0,
                const SmallVectorImpl<LiveInterval*> *uselessRegs = 0)
    : parent_(parent), newRegs_(newRegs),
      delegate_(delegate),
      uselessRegs_(uselessRegs),
      firstNew_(newRegs.size()),
      scannedRemattable_(false) {}

  LiveInterval &getParent() const { return parent_; }
  unsigned getReg() const { return parent_.reg; }

  /// Iterator for accessing the new registers added by this edit.
  typedef SmallVectorImpl<LiveInterval*>::const_iterator iterator;
  iterator begin() const { return newRegs_.begin()+firstNew_; }
  iterator end() const { return newRegs_.end(); }
  unsigned size() const { return newRegs_.size()-firstNew_; }
  bool empty() const { return size() == 0; }
  LiveInterval *get(unsigned idx) const { return newRegs_[idx+firstNew_]; }

  /// FIXME: Temporary accessors until we can get rid of
  /// LiveIntervals::AddIntervalsForSpills
  SmallVectorImpl<LiveInterval*> *getNewVRegs() { return &newRegs_; }
  const SmallVectorImpl<LiveInterval*> *getUselessVRegs() {
    return uselessRegs_;
  }

  /// createFrom - Create a new virtual register based on OldReg.
  LiveInterval &createFrom(unsigned OldReg, LiveIntervals&, VirtRegMap&);

  /// create - Create a new register with the same class and original slot as
  /// parent.
  LiveInterval &create(LiveIntervals &LIS, VirtRegMap &VRM) {
    return createFrom(getReg(), LIS, VRM);
  }

  /// anyRematerializable - Return true if any parent values may be
  /// rematerializable.
  /// This function must be called before any rematerialization is attempted.
  bool anyRematerializable(LiveIntervals&, const TargetInstrInfo&,
                           AliasAnalysis*);

  /// checkRematerializable - Manually add VNI to the list of rematerializable
  /// values if DefMI may be rematerializable.
  bool checkRematerializable(VNInfo *VNI, const MachineInstr *DefMI,
                             const TargetInstrInfo&, AliasAnalysis*);

  /// Remat - Information needed to rematerialize at a specific location.
  struct Remat {
    VNInfo *ParentVNI;      // parent_'s value at the remat location.
    MachineInstr *OrigMI;   // Instruction defining ParentVNI.
    explicit Remat(VNInfo *ParentVNI) : ParentVNI(ParentVNI), OrigMI(0) {}
  };

  /// canRematerializeAt - Determine if ParentVNI can be rematerialized at
  /// UseIdx. It is assumed that parent_.getVNINfoAt(UseIdx) == ParentVNI.
  /// When cheapAsAMove is set, only cheap remats are allowed.
  bool canRematerializeAt(Remat &RM,
                          SlotIndex UseIdx,
                          bool cheapAsAMove,
                          LiveIntervals &lis);

  /// rematerializeAt - Rematerialize RM.ParentVNI into DestReg by inserting an
  /// instruction into MBB before MI. The new instruction is mapped, but
  /// liveness is not updated.
  /// Return the SlotIndex of the new instruction.
  SlotIndex rematerializeAt(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MI,
                            unsigned DestReg,
                            const Remat &RM,
                            LiveIntervals&,
                            const TargetInstrInfo&,
                            const TargetRegisterInfo&);

  /// markRematerialized - explicitly mark a value as rematerialized after doing
  /// it manually.
  void markRematerialized(const VNInfo *ParentVNI) {
    rematted_.insert(ParentVNI);
  }

  /// didRematerialize - Return true if ParentVNI was rematerialized anywhere.
  bool didRematerialize(const VNInfo *ParentVNI) const {
    return rematted_.count(ParentVNI);
  }

  /// eraseVirtReg - Notify the delegate that Reg is no longer in use, and try
  /// to erase it from LIS.
  void eraseVirtReg(unsigned Reg, LiveIntervals &LIS);

  /// eliminateDeadDefs - Try to delete machine instructions that are now dead
  /// (allDefsAreDead returns true). This may cause live intervals to be trimmed
  /// and further dead efs to be eliminated.
  void eliminateDeadDefs(SmallVectorImpl<MachineInstr*> &Dead,
                         LiveIntervals&, VirtRegMap&,
                         const TargetInstrInfo&);

  /// calculateRegClassAndHint - Recompute register class and hint for each new
  /// register.
  void calculateRegClassAndHint(MachineFunction&, LiveIntervals&,
                                const MachineLoopInfo&);
};

}

#endif
