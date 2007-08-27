//===-- SimpleRegisterCoalescing.h - Register Coalescing --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple register copy coalescing phase.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SIMPLE_REGISTER_COALESCING_H
#define LLVM_CODEGEN_SIMPLE_REGISTER_COALESCING_H

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/IndexedMap.h"

namespace llvm {

  class LiveVariables;
  class MRegisterInfo;
  class TargetInstrInfo;
  class VirtRegMap;

  class SimpleRegisterCoalescing : public MachineFunctionPass {
    MachineFunction* mf_;
    const TargetMachine* tm_;
    const MRegisterInfo* mri_;
    const TargetInstrInfo* tii_;
    LiveIntervals *li_;
    LiveVariables *lv_;
    
    typedef IndexedMap<unsigned> Reg2RegMap;
    Reg2RegMap r2rMap_;

    BitVector allocatableRegs_;
    DenseMap<const TargetRegisterClass*, BitVector> allocatableRCRegs_;

    /// JoinedLIs - Keep track which register intervals have been coalesced
    /// with other intervals.
    BitVector JoinedLIs;

  public:
    static char ID; // Pass identifcation, replacement for typeid
    SimpleRegisterCoalescing() : MachineFunctionPass((intptr_t)&ID) {}

    struct CopyRec {
      MachineInstr *MI;
      unsigned SrcReg, DstReg;
    };
    CopyRec getCopyRec(MachineInstr *MI, unsigned SrcReg, unsigned DstReg) {
      CopyRec R;
      R.MI = MI;
      R.SrcReg = SrcReg;
      R.DstReg = DstReg;
      return R;
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

    /// print - Implement the dump method.
    virtual void print(std::ostream &O, const Module* = 0) const;
    void print(std::ostream *O, const Module* M = 0) const {
      if (O) print(*O, M);
    }

  private:      
    /// joinIntervals - join compatible live intervals
    void joinIntervals();

    /// CopyCoalesceInMBB - Coalesce copies in the specified MBB, putting
    /// copies that cannot yet be coalesced into the "TryAgain" list.
    void CopyCoalesceInMBB(MachineBasicBlock *MBB,
                         std::vector<CopyRec> *TryAgain, bool PhysOnly = false);

    /// JoinCopy - Attempt to join intervals corresponding to SrcReg/DstReg,
    /// which are the src/dst of the copy instruction CopyMI.  This returns true
    /// if the copy was successfully coalesced away, or if it is never possible
    /// to coalesce these this copy, due to register constraints.  It returns
    /// false if it is not currently possible to coalesce this interval, but
    /// it may be possible if other things get coalesced.
    bool JoinCopy(MachineInstr *CopyMI, unsigned SrcReg, unsigned DstReg,
                  bool PhysOnly = false);
    
    /// JoinIntervals - Attempt to join these two intervals.  On failure, this
    /// returns false.  Otherwise, if one of the intervals being joined is a
    /// physreg, this method always canonicalizes DestInt to be it.  The output
    /// "SrcInt" will not have been modified, so we can use this information
    /// below to update aliases.
    bool JoinIntervals(LiveInterval &LHS, LiveInterval &RHS);
    
    /// SimpleJoin - Attempt to join the specified interval into this one. The
    /// caller of this method must guarantee that the RHS only contains a single
    /// value number and that the RHS is not defined by a copy from this
    /// interval.  This returns false if the intervals are not joinable, or it
    /// joins them and returns true.
    bool SimpleJoin(LiveInterval &LHS, LiveInterval &RHS);
    
    /// Return true if the two specified registers belong to different
    /// register classes.  The registers may be either phys or virt regs.
    bool differingRegisterClasses(unsigned RegA, unsigned RegB) const;


    bool AdjustCopiesBackFrom(LiveInterval &IntA, LiveInterval &IntB,
                              MachineInstr *CopyMI);

    /// lastRegisterUse - Returns the last use of the specific register between
    /// cycles Start and End. It also returns the use operand by reference. It
    /// returns NULL if there are no uses.
     MachineInstr *lastRegisterUse(unsigned Start, unsigned End, unsigned Reg,
                                  MachineOperand *&MOU);

    /// findDefOperand - Returns the MachineOperand that is a def of the specific
    /// register. It returns NULL if the def is not found.
    MachineOperand *findDefOperand(MachineInstr *MI, unsigned Reg);

    /// unsetRegisterKill - Unset IsKill property of all uses of the specific
    /// register of the specific instruction.
    void unsetRegisterKill(MachineInstr *MI, unsigned Reg);

    /// unsetRegisterKills - Unset IsKill property of all uses of specific register
    /// between cycles Start and End.
    void unsetRegisterKills(unsigned Start, unsigned End, unsigned Reg);

    /// hasRegisterDef - True if the instruction defines the specific register.
    ///
    bool hasRegisterDef(MachineInstr *MI, unsigned Reg);

    /// rep - returns the representative of this register
    unsigned rep(unsigned Reg) {
      unsigned Rep = r2rMap_[Reg];
      if (Rep)
        return r2rMap_[Reg] = rep(Rep);
      return Reg;
    }

    void printRegName(unsigned reg) const;
  };

} // End llvm namespace

#endif
