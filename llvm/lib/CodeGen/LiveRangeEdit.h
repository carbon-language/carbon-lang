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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class AliasAnalysis;
class LiveIntervals;
class MachineLoopInfo;
class MachineRegisterInfo;
class VirtRegMap;

class LiveRangeEdit {
public:
  /// Callback methods for LiveRangeEdit owners.
  class Delegate {
    virtual void anchor();
  public:
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
  MachineRegisterInfo &MRI;
  LiveIntervals &LIS;
  VirtRegMap *VRM;
  const TargetInstrInfo &TII;
  Delegate *const delegate_;

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
  void scanRemattable(AliasAnalysis *aa);

  /// allUsesAvailableAt - Return true if all registers used by OrigMI at
  /// OrigIdx are also available with the same value at UseIdx.
  bool allUsesAvailableAt(const MachineInstr *OrigMI, SlotIndex OrigIdx,
                          SlotIndex UseIdx);

  /// foldAsLoad - If LI has a single use and a single def that can be folded as
  /// a load, eliminate the register by folding the def into the use.
  bool foldAsLoad(LiveInterval *LI, SmallVectorImpl<MachineInstr*> &Dead);

public:
  /// Create a LiveRangeEdit for breaking down parent into smaller pieces.
  /// @param parent The register being spilled or split.
  /// @param newRegs List to receive any new registers created. This needn't be
  ///                empty initially, any existing registers are ignored.
  LiveRangeEdit(LiveInterval &parent,
                SmallVectorImpl<LiveInterval*> &newRegs,
                MachineFunction &MF,
                LiveIntervals &lis,
                VirtRegMap *vrm,
                Delegate *delegate = 0)
    : parent_(parent), newRegs_(newRegs),
      MRI(MF.getRegInfo()), LIS(lis), VRM(vrm),
      TII(*MF.getTarget().getInstrInfo()),
      delegate_(delegate),
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

  ArrayRef<LiveInterval*> regs() const {
    return makeArrayRef(newRegs_).slice(firstNew_);
  }

  /// createFrom - Create a new virtual register based on OldReg.
  LiveInterval &createFrom(unsigned OldReg);

  /// create - Create a new register with the same class and original slot as
  /// parent.
  LiveInterval &create() {
    return createFrom(getReg());
  }

  /// anyRematerializable - Return true if any parent values may be
  /// rematerializable.
  /// This function must be called before any rematerialization is attempted.
  bool anyRematerializable(AliasAnalysis*);

  /// checkRematerializable - Manually add VNI to the list of rematerializable
  /// values if DefMI may be rematerializable.
  bool checkRematerializable(VNInfo *VNI, const MachineInstr *DefMI,
                             AliasAnalysis*);

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
                          bool cheapAsAMove);

  /// rematerializeAt - Rematerialize RM.ParentVNI into DestReg by inserting an
  /// instruction into MBB before MI. The new instruction is mapped, but
  /// liveness is not updated.
  /// Return the SlotIndex of the new instruction.
  SlotIndex rematerializeAt(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MI,
                            unsigned DestReg,
                            const Remat &RM,
                            const TargetRegisterInfo&,
                            bool Late = false);

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
  void eraseVirtReg(unsigned Reg);

  /// eliminateDeadDefs - Try to delete machine instructions that are now dead
  /// (allDefsAreDead returns true). This may cause live intervals to be trimmed
  /// and further dead efs to be eliminated.
  /// RegsBeingSpilled lists registers currently being spilled by the register
  /// allocator.  These registers should not be split into new intervals
  /// as currently those new intervals are not guaranteed to spill.
  void eliminateDeadDefs(SmallVectorImpl<MachineInstr*> &Dead,
                         ArrayRef<unsigned> RegsBeingSpilled 
                          = ArrayRef<unsigned>());

  /// calculateRegClassAndHint - Recompute register class and hint for each new
  /// register.
  void calculateRegClassAndHint(MachineFunction&,
                                const MachineLoopInfo&);
};

}

#endif
