//===-- SimpleRegisterCoalescing.h - Register Coalescing --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple register copy coalescing phase.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SIMPLE_REGISTER_COALESCING_H
#define LLVM_CODEGEN_SIMPLE_REGISTER_COALESCING_H

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/RegisterCoalescer.h"
#include "llvm/ADT/BitVector.h"

namespace llvm {
  class SimpleRegisterCoalescing;
  class LiveVariables;
  class TargetRegisterInfo;
  class TargetInstrInfo;
  class VirtRegMap;
  class MachineLoopInfo;

  /// CopyRec - Representation for copy instructions in coalescer queue.
  ///
  struct CopyRec {
    MachineInstr *MI;
    unsigned LoopDepth;
    CopyRec(MachineInstr *mi, unsigned depth)
      : MI(mi), LoopDepth(depth) {}
  };

  class SimpleRegisterCoalescing : public MachineFunctionPass,
                                   public RegisterCoalescer {
    MachineFunction* mf_;
    MachineRegisterInfo* mri_;
    const TargetMachine* tm_;
    const TargetRegisterInfo* tri_;
    const TargetInstrInfo* tii_;
    LiveIntervals *li_;
    const MachineLoopInfo* loopInfo;
    AliasAnalysis *AA;
    
    DenseMap<const TargetRegisterClass*, BitVector> allocatableRCRegs_;

    /// JoinedCopies - Keep track of copies eliminated due to coalescing.
    ///
    SmallPtrSet<MachineInstr*, 32> JoinedCopies;

    /// ReMatCopies - Keep track of copies eliminated due to remat.
    ///
    SmallPtrSet<MachineInstr*, 32> ReMatCopies;

    /// ReMatDefs - Keep track of definition instructions which have
    /// been remat'ed.
    SmallPtrSet<MachineInstr*, 8> ReMatDefs;

  public:
    static char ID; // Pass identifcation, replacement for typeid
    SimpleRegisterCoalescing() : MachineFunctionPass(ID) {
      initializeSimpleRegisterCoalescingPass(*PassRegistry::getPassRegistry());
    }

    struct InstrSlots {
      enum {
        LOAD  = 0,
        USE   = 1,
        DEF   = 2,
        STORE = 3,
        NUM   = 4
      };
    };
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    virtual void releaseMemory();

    /// runOnMachineFunction - pass entry point
    virtual bool runOnMachineFunction(MachineFunction&);

    bool coalesceFunction(MachineFunction &mf, RegallocQuery &) {
      // This runs as an independent pass, so don't do anything.
      return false;
    }

    /// print - Implement the dump method.
    virtual void print(raw_ostream &O, const Module* = 0) const;

  private:
    /// joinIntervals - join compatible live intervals
    void joinIntervals();

    /// CopyCoalesceInMBB - Coalesce copies in the specified MBB, putting
    /// copies that cannot yet be coalesced into the "TryAgain" list.
    void CopyCoalesceInMBB(MachineBasicBlock *MBB,
                           std::vector<CopyRec> &TryAgain);

    /// JoinCopy - Attempt to join intervals corresponding to SrcReg/DstReg,
    /// which are the src/dst of the copy instruction CopyMI.  This returns true
    /// if the copy was successfully coalesced away. If it is not currently
    /// possible to coalesce this interval, but it may be possible if other
    /// things get coalesced, then it returns true by reference in 'Again'.
    bool JoinCopy(CopyRec &TheCopy, bool &Again);

    /// JoinIntervals - Attempt to join these two intervals.  On failure, this
    /// returns false.  The output "SrcInt" will not have been modified, so we can
    /// use this information below to update aliases.
    bool JoinIntervals(CoalescerPair &CP);

    /// Return true if the two specified registers belong to different register
    /// classes.  The registers may be either phys or virt regs.
    bool differingRegisterClasses(unsigned RegA, unsigned RegB) const;

    /// AdjustCopiesBackFrom - We found a non-trivially-coalescable copy. If
    /// the source value number is defined by a copy from the destination reg
    /// see if we can merge these two destination reg valno# into a single
    /// value number, eliminating a copy.
    bool AdjustCopiesBackFrom(const CoalescerPair &CP, MachineInstr *CopyMI);

    /// HasOtherReachingDefs - Return true if there are definitions of IntB
    /// other than BValNo val# that can reach uses of AValno val# of IntA.
    bool HasOtherReachingDefs(LiveInterval &IntA, LiveInterval &IntB,
                              VNInfo *AValNo, VNInfo *BValNo);

    /// RemoveCopyByCommutingDef - We found a non-trivially-coalescable copy.
    /// If the source value number is defined by a commutable instruction and
    /// its other operand is coalesced to the copy dest register, see if we
    /// can transform the copy into a noop by commuting the definition.
    bool RemoveCopyByCommutingDef(const CoalescerPair &CP,MachineInstr *CopyMI);

    /// TrimLiveIntervalToLastUse - If there is a last use in the same basic
    /// block as the copy instruction, trim the ive interval to the last use
    /// and return true.
    bool TrimLiveIntervalToLastUse(SlotIndex CopyIdx,
                                   MachineBasicBlock *CopyMBB,
                                   LiveInterval &li, const LiveRange *LR);

    /// ReMaterializeTrivialDef - If the source of a copy is defined by a trivial
    /// computation, replace the copy by rematerialize the definition.
    bool ReMaterializeTrivialDef(LiveInterval &SrcInt, unsigned DstReg,
                                 unsigned DstSubIdx, MachineInstr *CopyMI);

    /// isWinToJoinCrossClass - Return true if it's profitable to coalesce
    /// two virtual registers from different register classes.
    bool isWinToJoinCrossClass(unsigned SrcReg,
                               unsigned DstReg,
                               const TargetRegisterClass *SrcRC,
                               const TargetRegisterClass *DstRC,
                               const TargetRegisterClass *NewRC);

    /// UpdateRegDefsUses - Replace all defs and uses of SrcReg to DstReg and
    /// update the subregister number if it is not zero. If DstReg is a
    /// physical register and the existing subregister number of the def / use
    /// being updated is not zero, make sure to set it to the correct physical
    /// subregister.
    void UpdateRegDefsUses(const CoalescerPair &CP);

    /// ShortenDeadCopyLiveRange - Shorten a live range defined by a dead copy.
    /// Return true if live interval is removed.
    bool ShortenDeadCopyLiveRange(LiveInterval &li, MachineInstr *CopyMI);

    /// ShortenDeadCopyLiveRange - Shorten a live range as it's artificially
    /// extended by a dead copy. Mark the last use (if any) of the val# as kill
    /// as ends the live range there. If there isn't another use, then this
    /// live range is dead. Return true if live interval is removed.
    bool ShortenDeadCopySrcLiveRange(LiveInterval &li, MachineInstr *CopyMI);

    /// RemoveDeadDef - If a def of a live interval is now determined dead,
    /// remove the val# it defines. If the live interval becomes empty, remove
    /// it as well.
    bool RemoveDeadDef(LiveInterval &li, MachineInstr *DefMI);

    /// RemoveCopyFlag - If DstReg is no longer defined by CopyMI, clear the
    /// VNInfo copy flag for DstReg and all aliases.
    void RemoveCopyFlag(unsigned DstReg, const MachineInstr *CopyMI);

    /// lastRegisterUse - Returns the last use of the specific register between
    /// cycles Start and End or NULL if there are no uses.
    MachineOperand *lastRegisterUse(SlotIndex Start, SlotIndex End,
                                    unsigned Reg, SlotIndex &LastUseIdx) const;
  };

} // End llvm namespace

#endif
