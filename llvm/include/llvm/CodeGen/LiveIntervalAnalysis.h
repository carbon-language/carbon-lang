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
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

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

    /// MBB2IdxMap - The indexes of the first and last instructions in the
    /// specified basic block.
    std::vector<std::pair<unsigned, unsigned> > MBB2IdxMap;

    typedef std::map<MachineInstr*, unsigned> Mi2IndexMap;
    Mi2IndexMap mi2iMap_;

    typedef std::vector<MachineInstr*> Index2MiMap;
    Index2MiMap i2miMap_;

    typedef std::map<unsigned, LiveInterval> Reg2IntervalMap;
    Reg2IntervalMap r2iMap_;

    BitVector allocatableRegs_;

    std::vector<MachineInstr*> ClonedMIs;

  public:
    static char ID; // Pass identification, replacement for typeid
    LiveIntervals() : MachineFunctionPass((intptr_t)&ID) {}

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
      return MBB2IdxMap[MBBNo].first;
    }

    /// getMBBEndIdx - Return the store index of the last instruction in the
    /// specified MachineBasicBlock.
    unsigned getMBBEndIdx(MachineBasicBlock *MBB) const {
      return getMBBEndIdx(MBB->getNumber());
    }
    unsigned getMBBEndIdx(unsigned MBBNo) const {
      assert(MBBNo < MBB2IdxMap.size() && "Invalid MBB number!");
      return MBB2IdxMap[MBBNo].second;
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

    // Interval creation

    LiveInterval &getOrCreateInterval(unsigned reg) {
      Reg2IntervalMap::iterator I = r2iMap_.find(reg);
      if (I == r2iMap_.end())
        I = r2iMap_.insert(I, std::make_pair(reg, createInterval(reg)));
      return I->second;
    }

    /// CreateNewLiveInterval - Create a new live interval with the given live
    /// ranges. The new live interval will have an infinite spill weight.
    LiveInterval &CreateNewLiveInterval(const LiveInterval *LI,
                                        const std::vector<LiveRange> &LRs);

    std::vector<LiveInterval*> addIntervalsForSpills(const LiveInterval& i,
                                                 VirtRegMap& vrm, unsigned reg);

    // Interval removal

    void removeInterval(unsigned Reg) {
      r2iMap_.erase(Reg);
    }

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
    /// computeIntervals - Compute live intervals.
    void computeIntervals();
    
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

    /// isReMaterializable - Returns true if the definition MI of the specified
    /// val# of the specified interval is re-materializable.
    bool isReMaterializable(const LiveInterval &li, const VNInfo *ValNo,
                            MachineInstr *MI);

    /// tryFoldMemoryOperand - Attempts to fold either a spill / restore from
    /// slot / to reg or any rematerialized load into ith operand of specified
    /// MI. If it is successul, MI is updated with the newly created MI and
    /// returns true.
    bool tryFoldMemoryOperand(MachineInstr* &MI, VirtRegMap &vrm,
                              unsigned index, unsigned i, bool isSS,
                              MachineInstr *DefMI, int slot, unsigned reg);

    static LiveInterval createInterval(unsigned Reg);

    void printRegName(unsigned reg) const;
  };

} // End llvm namespace

#endif
