//===-- LiveIntervalAnalysis.h - Live Interval Analysis ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LiveInterval analysis pass.  Given some numbering of
// each the machine instructions (in this implemention depth-first order) an
// interval [i, j) is said to be a live interval for register v if there is no
// instruction with number j' > j such that v is live at j' abd there is no
// instruction with number i' < i such that v is live at i'. In this
// implementation intervals can have holes, i.e. an interval might look like
// [1,20), [50,65), [1000,1001).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LIVEINTERVAL_ANALYSIS_H
#define LLVM_CODEGEN_LIVEINTERVAL_ANALYSIS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/LiveInterval.h"

namespace llvm {

  class LiveVariables;
  class MRegisterInfo;
  class TargetInstrInfo;
  class VirtRegMap;

  class LiveIntervals : public MachineFunctionPass {
    MachineFunction* mf_;
    const TargetMachine* tm_;
    const MRegisterInfo* mri_;
    const TargetInstrInfo* tii_;
    LiveVariables* lv_;

    typedef std::map<MachineInstr*, unsigned> Mi2IndexMap;
    Mi2IndexMap mi2iMap_;

    typedef std::vector<MachineInstr*> Index2MiMap;
    Index2MiMap i2miMap_;

    typedef std::map<unsigned, LiveInterval> Reg2IntervalMap;
    Reg2IntervalMap r2iMap_;

    typedef DenseMap<unsigned> Reg2RegMap;
    Reg2RegMap r2rMap_;

    std::vector<bool> allocatableRegs_;

  public:
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

    static unsigned getBaseIndex(unsigned index) {
      return index - (index % InstrSlots::NUM);
    }
    static unsigned getBoundaryIndex(unsigned index) {
      return getBaseIndex(index + InstrSlots::NUM - 1);
    }
    static unsigned getLoadIndex(unsigned index) {
      return getBaseIndex(index) + InstrSlots::LOAD;
    }
    static unsigned getUseIndex(unsigned index) {
      return getBaseIndex(index) + InstrSlots::USE;
    }
    static unsigned getDefIndex(unsigned index) {
      return getBaseIndex(index) + InstrSlots::DEF;
    }
    static unsigned getStoreIndex(unsigned index) {
      return getBaseIndex(index) + InstrSlots::STORE;
    }

    typedef Reg2IntervalMap::iterator iterator;
    typedef Reg2IntervalMap::const_iterator const_iterator;
    const_iterator begin() const { return r2iMap_.begin(); }
    const_iterator end() const { return r2iMap_.end(); }
    iterator begin() { return r2iMap_.begin(); }
    iterator end() { return r2iMap_.end(); }
    unsigned getNumIntervals() const { return r2iMap_.size(); }

    LiveInterval &getInterval(unsigned reg) {
      Reg2IntervalMap::iterator I = r2iMap_.find(reg);
      assert(I != r2iMap_.end() && "Interval does not exist for register");
      return I->second;
    }

    const LiveInterval &getInterval(unsigned reg) const {
      Reg2IntervalMap::const_iterator I = r2iMap_.find(reg);
      assert(I != r2iMap_.end() && "Interval does not exist for register");
      return I->second;
    }

    /// getInstructionIndex - returns the base index of instr
    unsigned getInstructionIndex(MachineInstr* instr) const {
      Mi2IndexMap::const_iterator it = mi2iMap_.find(instr);
      assert(it != mi2iMap_.end() && "Invalid instruction!");
      return it->second;
    }

    /// getInstructionFromIndex - given an index in any slot of an
    /// instruction return a pointer the instruction
    MachineInstr* getInstructionFromIndex(unsigned index) const {
      index /= InstrSlots::NUM; // convert index to vector index
      assert(index < i2miMap_.size() &&
             "index does not correspond to an instruction");
      return i2miMap_[index];
    }

    std::vector<LiveInterval*> addIntervalsForSpills(const LiveInterval& i,
                                                     VirtRegMap& vrm,
                                                     int slot);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    virtual void releaseMemory();

    /// runOnMachineFunction - pass entry point
    virtual bool runOnMachineFunction(MachineFunction&);

    /// print - Implement the dump method.
    virtual void print(std::ostream &O, const Module* = 0) const;

  private:
    /// RemoveMachineInstrFromMaps - This marks the specified machine instr as
    /// deleted.
    void RemoveMachineInstrFromMaps(MachineInstr *MI) {
      // remove index -> MachineInstr and
      // MachineInstr -> index mappings
      Mi2IndexMap::iterator mi2i = mi2iMap_.find(MI);
      if (mi2i != mi2iMap_.end()) {
        i2miMap_[mi2i->second/InstrSlots::NUM] = 0;
        mi2iMap_.erase(mi2i);
      }
    }
      
    /// computeIntervals - compute live intervals
    void computeIntervals();

    /// joinIntervals - join compatible live intervals
    void joinIntervals();

    /// CopyCoallesceInMBB - Coallsece copies in the specified MBB, putting
    /// copies that cannot yet be coallesced into the "TryAgain" list.
    void CopyCoallesceInMBB(MachineBasicBlock *MBB,
                            std::vector<CopyRec> &TryAgain);

    /// JoinCopy - Attempt to join intervals corresponding to SrcReg/DstReg,
    /// which are the src/dst of the copy instruction CopyMI.  This returns true
    /// if the copy was successfully coallesced away, or if it is never possible
    /// to coallesce these this copy, due to register constraints.  It returns
    /// false if it is not currently possible to coallesce this interval, but
    /// it may be possible if other things get coallesced.
    bool JoinCopy(MachineInstr *CopyMI, unsigned SrcReg, unsigned DstReg);
    
    /// handleRegisterDef - update intervals for a register def
    /// (calls handlePhysicalRegisterDef and
    /// handleVirtualRegisterDef)
    void handleRegisterDef(MachineBasicBlock* mbb,
                           MachineBasicBlock::iterator mi,
                           unsigned reg);

    /// handleVirtualRegisterDef - update intervals for a virtual
    /// register def
    void handleVirtualRegisterDef(MachineBasicBlock* mbb,
                                  MachineBasicBlock::iterator mi,
                                  LiveInterval& interval);

    /// handlePhysicalRegisterDef - update intervals for a physical register
    /// def.
    void handlePhysicalRegisterDef(MachineBasicBlock* mbb,
                                   MachineBasicBlock::iterator mi,
                                   LiveInterval& interval,
                                   bool isLiveIn = false);

    /// Return true if the two specified registers belong to different
    /// register classes.  The registers may be either phys or virt regs.
    bool differingRegisterClasses(unsigned RegA, unsigned RegB) const;


    bool AdjustCopiesBackFrom(LiveInterval &IntA, LiveInterval &IntB,
                              MachineInstr *CopyMI, unsigned CopyIdx);

    bool overlapsAliases(const LiveInterval *lhs,
                         const LiveInterval *rhs) const;

    static LiveInterval createInterval(unsigned Reg);

    LiveInterval &getOrCreateInterval(unsigned reg) {
      Reg2IntervalMap::iterator I = r2iMap_.find(reg);
      if (I == r2iMap_.end())
        I = r2iMap_.insert(I, std::make_pair(reg, createInterval(reg)));
      return I->second;
    }

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
