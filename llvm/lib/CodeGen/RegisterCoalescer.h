//===-- RegisterCoalescer.h - Register Coalescing Interface ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the abstract interface for register coalescers, 
// allowing them to interact with and query register allocators.
//
//===----------------------------------------------------------------------===//

#include "RegisterClassInfo.h"
#include "llvm/Support/IncludeFile.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/ADT/SmallPtrSet.h"

#ifndef LLVM_CODEGEN_REGISTER_COALESCER_H
#define LLVM_CODEGEN_REGISTER_COALESCER_H

namespace llvm {

  class MachineFunction;
  class RegallocQuery;
  class AnalysisUsage;
  class MachineInstr;
  class TargetRegisterInfo;
  class TargetRegisterClass;
  class TargetInstrInfo;
  class LiveDebugVariables;
  class VirtRegMap;
  class MachineLoopInfo;

  class CoalescerPair;

  /// An abstract interface for register coalescers.  Coalescers must
  /// implement this interface to be part of the coalescer analysis
  /// group.
  class RegisterCoalescer : public MachineFunctionPass {
    MachineFunction* mf_;
    MachineRegisterInfo* mri_;
    const TargetMachine* tm_;
    const TargetRegisterInfo* tri_;
    const TargetInstrInfo* tii_;
    LiveIntervals *li_;
    LiveDebugVariables *ldv_;
    const MachineLoopInfo* loopInfo;
    AliasAnalysis *AA;
    RegisterClassInfo RegClassInfo;

    /// JoinedCopies - Keep track of copies eliminated due to coalescing.
    ///
    SmallPtrSet<MachineInstr*, 32> JoinedCopies;

    /// ReMatCopies - Keep track of copies eliminated due to remat.
    ///
    SmallPtrSet<MachineInstr*, 32> ReMatCopies;

    /// ReMatDefs - Keep track of definition instructions which have
    /// been remat'ed.
    SmallPtrSet<MachineInstr*, 8> ReMatDefs;

    /// joinIntervals - join compatible live intervals
    void joinIntervals();

    /// CopyCoalesceInMBB - Coalesce copies in the specified MBB, putting
    /// copies that cannot yet be coalesced into the "TryAgain" list.
    void CopyCoalesceInMBB(MachineBasicBlock *MBB,
                           std::vector<MachineInstr*> &TryAgain);

    /// JoinCopy - Attempt to join intervals corresponding to SrcReg/DstReg,
    /// which are the src/dst of the copy instruction CopyMI.  This returns true
    /// if the copy was successfully coalesced away. If it is not currently
    /// possible to coalesce this interval, but it may be possible if other
    /// things get coalesced, then it returns true by reference in 'Again'.
    bool JoinCopy(MachineInstr *TheCopy, bool &Again);

    /// JoinIntervals - Attempt to join these two intervals.  On failure, this
    /// returns false.  The output "SrcInt" will not have been modified, so we can
    /// use this information below to update aliases.
    bool JoinIntervals(CoalescerPair &CP);

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

    /// ReMaterializeTrivialDef - If the source of a copy is defined by a trivial
    /// computation, replace the copy by rematerialize the definition.
    /// If PreserveSrcInt is true, make sure SrcInt is valid after the call.
    bool ReMaterializeTrivialDef(LiveInterval &SrcInt, bool PreserveSrcInt,
                                 unsigned DstReg, unsigned DstSubIdx,
                                 MachineInstr *CopyMI);

    /// shouldJoinPhys - Return true if a physreg copy should be joined.
    bool shouldJoinPhys(CoalescerPair &CP);

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

    /// RemoveDeadDef - If a def of a live interval is now determined dead,
    /// remove the val# it defines. If the live interval becomes empty, remove
    /// it as well.
    bool RemoveDeadDef(LiveInterval &li, MachineInstr *DefMI);

    /// RemoveCopyFlag - If DstReg is no longer defined by CopyMI, clear the
    /// VNInfo copy flag for DstReg and all aliases.
    void RemoveCopyFlag(unsigned DstReg, const MachineInstr *CopyMI);

    /// markAsJoined - Remember that CopyMI has already been joined.
    void markAsJoined(MachineInstr *CopyMI);

  public:
    static char ID; // Class identification, replacement for typeinfo
    RegisterCoalescer() : MachineFunctionPass(ID) {
      initializeRegisterCoalescerPass(*PassRegistry::getPassRegistry());
    }

    /// Register allocators must call this from their own
    /// getAnalysisUsage to cover the case where the coalescer is not
    /// a Pass in the proper sense and isn't managed by PassManager.
    /// PassManager needs to know which analyses to make available and
    /// which to invalidate when running the register allocator or any
    /// pass that might call coalescing.  The long-term solution is to
    /// allow hierarchies of PassManagers.
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;

    virtual void releaseMemory();

    /// runOnMachineFunction - pass entry point
    virtual bool runOnMachineFunction(MachineFunction&);

    /// print - Implement the dump method.
    virtual void print(raw_ostream &O, const Module* = 0) const;
  };

  /// CoalescerPair - A helper class for register coalescers. When deciding if
  /// two registers can be coalesced, CoalescerPair can determine if a copy
  /// instruction would become an identity copy after coalescing.
  class CoalescerPair {
    const TargetInstrInfo &tii_;
    const TargetRegisterInfo &tri_;

    /// dstReg_ - The register that will be left after coalescing. It can be a
    /// virtual or physical register.
    unsigned dstReg_;

    /// srcReg_ - the virtual register that will be coalesced into dstReg.
    unsigned srcReg_;

    /// subReg_ - The subregister index of srcReg in dstReg_. It is possible the
    /// coalesce srcReg_ into a subreg of the larger dstReg_ when dstReg_ is a
    /// virtual register.
    unsigned subIdx_;

    /// partial_ - True when the original copy was a partial subregister copy.
    bool partial_;

    /// crossClass_ - True when both regs are virtual, and newRC is constrained.
    bool crossClass_;

    /// flipped_ - True when DstReg and SrcReg are reversed from the oriignal copy
    /// instruction.
    bool flipped_;

    /// newRC_ - The register class of the coalesced register, or NULL if dstReg_
    /// is a physreg.
    const TargetRegisterClass *newRC_;

  public:
    CoalescerPair(const TargetInstrInfo &tii, const TargetRegisterInfo &tri)
      : tii_(tii), tri_(tri), dstReg_(0), srcReg_(0), subIdx_(0),
        partial_(false), crossClass_(false), flipped_(false), newRC_(0) {}

    /// setRegisters - set registers to match the copy instruction MI. Return
    /// false if MI is not a coalescable copy instruction.
    bool setRegisters(const MachineInstr*);

    /// flip - Swap srcReg_ and dstReg_. Return false if swapping is impossible
    /// because dstReg_ is a physical register, or subIdx_ is set.
    bool flip();

    /// isCoalescable - Return true if MI is a copy instruction that will become
    /// an identity copy after coalescing.
    bool isCoalescable(const MachineInstr*) const;

    /// isPhys - Return true if DstReg is a physical register.
    bool isPhys() const { return !newRC_; }

    /// isPartial - Return true if the original copy instruction did not copy the
    /// full register, but was a subreg operation.
    bool isPartial() const { return partial_; }

    /// isCrossClass - Return true if DstReg is virtual and NewRC is a smaller register class than DstReg's.
    bool isCrossClass() const { return crossClass_; }

    /// isFlipped - Return true when getSrcReg is the register being defined by
    /// the original copy instruction.
    bool isFlipped() const { return flipped_; }

    /// getDstReg - Return the register (virtual or physical) that will remain
    /// after coalescing.
    unsigned getDstReg() const { return dstReg_; }

    /// getSrcReg - Return the virtual register that will be coalesced away.
    unsigned getSrcReg() const { return srcReg_; }

    /// getSubIdx - Return the subregister index in DstReg that SrcReg will be
    /// coalesced into, or 0.
    unsigned getSubIdx() const { return subIdx_; }

    /// getNewRC - Return the register class of the coalesced register.
    const TargetRegisterClass *getNewRC() const { return newRC_; }
  };
} // End llvm namespace

#endif
