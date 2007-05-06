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

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IndexedMap.h"

namespace llvm {

  class LiveVariables;
  class MRegisterInfo;
  class TargetInstrInfo;
  class TargetRegisterClass;
  class VirtRegMap;

  class LiveIntervals : public MachineFunctionPass {
    MachineFunction* mf_;
    const TargetMachine* tm_;
    const MRegisterInfo* mri_;
    const TargetInstrInfo* tii_;
    LiveVariables* lv_;

    /// MBB2IdxMap - The index of the first instruction in the specified basic
    /// block.
    std::vector<unsigned> MBB2IdxMap;
    
    typedef std::map<MachineInstr*, unsigned> Mi2IndexMap;
    Mi2IndexMap mi2iMap_;

    typedef std::vector<MachineInstr*> Index2MiMap;
    Index2MiMap i2miMap_;

    typedef std::map<unsigned, LiveInterval> Reg2IntervalMap;
    Reg2IntervalMap r2iMap_;

    typedef IndexedMap<unsigned> Reg2RegMap;
    Reg2RegMap r2rMap_;

    BitVector allocatableRegs_;
    DenseMap<const TargetRegisterClass*, BitVector> allocatableRCRegs_;

    /// JoinedLIs - Keep track which register intervals have been coalesced
    /// with other intervals.
    BitVector JoinedLIs;

  public:
    static char ID; // Pass identification, replacement for typeid
    LiveIntervals() : MachineFunctionPass((intptr_t)&ID) {}

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

    bool hasInterval(unsigned reg) const {
      return r2iMap_.count(reg);
    }

    /// getMBBStartIdx - Return the base index of the first instruction in the
    /// specified MachineBasicBlock.
    unsigned getMBBStartIdx(MachineBasicBlock *MBB) const {
      return getMBBStartIdx(MBB->getNumber());
    }
    
    unsigned getMBBStartIdx(unsigned MBBNo) const {
      assert(MBBNo < MBB2IdxMap.size() && "Invalid MBB number!");
      return MBB2IdxMap[MBBNo];
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

    /// CreateNewLiveInterval - Create a new live interval with the given live
    /// ranges. The new live interval will have an infinite spill weight.
    LiveInterval &CreateNewLiveInterval(const LiveInterval *LI,
                                        const std::vector<LiveRange> &LRs);

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
    /// isRemoved - returns true if the specified machine instr has been
    /// removed.
    bool isRemoved(MachineInstr* instr) const {
      return !mi2iMap_.count(instr);
    }

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
      
    /// computeIntervals - Compute live intervals.
    void computeIntervals();

    /// joinIntervals - join compatible live intervals
    void joinIntervals();

    /// CopyCoallesceInMBB - Coallsece copies in the specified MBB, putting
    /// copies that cannot yet be coallesced into the "TryAgain" list.
    void CopyCoallesceInMBB(MachineBasicBlock *MBB,
                         std::vector<CopyRec> *TryAgain, bool PhysOnly = false);

    /// JoinCopy - Attempt to join intervals corresponding to SrcReg/DstReg,
    /// which are the src/dst of the copy instruction CopyMI.  This returns true
    /// if the copy was successfully coallesced away, or if it is never possible
    /// to coallesce these this copy, due to register constraints.  It returns
    /// false if it is not currently possible to coallesce this interval, but
    /// it may be possible if other things get coallesced.
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
    
    /// handleRegisterDef - update intervals for a register def
    /// (calls handlePhysicalRegisterDef and
    /// handleVirtualRegisterDef)
    void handleRegisterDef(MachineBasicBlock *MBB,
                           MachineBasicBlock::iterator MI, unsigned MIIdx,
                           unsigned reg);

    /// handleVirtualRegisterDef - update intervals for a virtual
    /// register def
    void handleVirtualRegisterDef(MachineBasicBlock *MBB,
                                  MachineBasicBlock::iterator MI,
                                  unsigned MIIdx,
                                  LiveInterval& interval);

    /// handlePhysicalRegisterDef - update intervals for a physical register
    /// def.
    void handlePhysicalRegisterDef(MachineBasicBlock* mbb,
                                   MachineBasicBlock::iterator mi,
                                   unsigned MIIdx,
                                   LiveInterval &interval,
                                   unsigned SrcReg);

    /// handleLiveInRegister - Create interval for a livein register.
    void handleLiveInRegister(MachineBasicBlock* mbb,
                              unsigned MIIdx,
                              LiveInterval &interval, bool isAlias = false);

    /// Return true if the two specified registers belong to different
    /// register classes.  The registers may be either phys or virt regs.
    bool differingRegisterClasses(unsigned RegA, unsigned RegB) const;


    bool AdjustCopiesBackFrom(LiveInterval &IntA, LiveInterval &IntB,
                              MachineInstr *CopyMI);

    /// lastRegisterUse - Returns the last use of the specific register between
    /// cycles Start and End. It also returns the use operand by reference. It
    /// returns NULL if there are no uses.
    MachineInstr *lastRegisterUse(unsigned Reg, unsigned Start, unsigned End,
                                  MachineOperand *&MOU);

    /// findDefOperand - Returns the MachineOperand that is a def of the specific
    /// register. It returns NULL if the def is not found.
    MachineOperand *findDefOperand(MachineInstr *MI, unsigned Reg);

    /// unsetRegisterKill - Unset IsKill property of all uses of the specific
    /// register of the specific instruction.
    void unsetRegisterKill(MachineInstr *MI, unsigned Reg);

    /// hasRegisterDef - True if the instruction defines the specific register.
    ///
    bool hasRegisterDef(MachineInstr *MI, unsigned Reg);

    static LiveInterval createInterval(unsigned Reg);

    void removeInterval(unsigned Reg) {
      r2iMap_.erase(Reg);
    }

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
