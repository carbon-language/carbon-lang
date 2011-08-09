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

#ifndef LLVM_CODEGEN_REGISTER_COALESCER_H
#define LLVM_CODEGEN_REGISTER_COALESCER_H

namespace llvm {

  class MachineInstr;
  class TargetRegisterInfo;
  class TargetRegisterClass;
  class TargetInstrInfo;

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
