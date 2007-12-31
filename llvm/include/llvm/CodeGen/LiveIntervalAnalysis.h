//===-- LiveIntervalAnalysis.h - Live Interval Analysis ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"
#include <cmath>

namespace llvm {

  class LiveVariables;
  class MachineLoopInfo;
  class MRegisterInfo;
  class MachineRegisterInfo;
  class TargetInstrInfo;
  class TargetRegisterClass;
  class VirtRegMap;
  typedef std::pair<unsigned, MachineBasicBlock*> IdxMBBPair;

  class LiveIntervals : public MachineFunctionPass {
    MachineFunction* mf_;
    const TargetMachine* tm_;
    const MRegisterInfo* mri_;
    const TargetInstrInfo* tii_;
    LiveVariables* lv_;

    /// Special pool allocator for VNInfo's (LiveInterval val#).
    ///
    BumpPtrAllocator VNInfoAllocator;

    /// MBB2IdxMap - The indexes of the first and last instructions in the
    /// specified basic block.
    std::vector<std::pair<unsigned, unsigned> > MBB2IdxMap;

    /// Idx2MBBMap - Sorted list of pairs of index of first instruction
    /// and MBB id.
    std::vector<IdxMBBPair> Idx2MBBMap;

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

    static float getSpillWeight(bool isDef, bool isUse, unsigned loopDepth) {
      return (isDef + isUse) * powf(10.0F, (float)loopDepth);
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

    /// conflictsWithPhysRegDef - Returns true if the specified register
    /// is defined during the duration of the specified interval.
    bool conflictsWithPhysRegDef(const LiveInterval &li, VirtRegMap &vrm,
                                 unsigned reg);

    /// findLiveInMBBs - Given a live range, if the value of the range
    /// is live in any MBB returns true as well as the list of basic blocks
    /// where the value is live in.
    bool findLiveInMBBs(const LiveRange &LR,
                        SmallVectorImpl<MachineBasicBlock*> &MBBs) const;

    // Interval creation

    LiveInterval &getOrCreateInterval(unsigned reg) {
      Reg2IntervalMap::iterator I = r2iMap_.find(reg);
      if (I == r2iMap_.end())
        I = r2iMap_.insert(I, std::make_pair(reg, createInterval(reg)));
      return I->second;
    }

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

    BumpPtrAllocator& getVNInfoAllocator() { return VNInfoAllocator; }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    virtual void releaseMemory();

    /// runOnMachineFunction - pass entry point
    virtual bool runOnMachineFunction(MachineFunction&);

    /// print - Implement the dump method.
    virtual void print(std::ostream &O, const Module* = 0) const;
    void print(std::ostream *O, const Module* M = 0) const {
      if (O) print(*O, M);
    }

    /// addIntervalsForSpills - Create new intervals for spilled defs / uses of
    /// the given interval.
    std::vector<LiveInterval*>
    addIntervalsForSpills(const LiveInterval& i,
                          const MachineLoopInfo *loopInfo, VirtRegMap& vrm);

    /// isReMaterializable - Returns true if every definition of MI of every
    /// val# of the specified interval is re-materializable. Also returns true
    /// by reference if all of the defs are load instructions.
    bool isReMaterializable(const LiveInterval &li, bool &isLoad);

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
    /// val# of the specified interval is re-materializable. Also returns true
    /// by reference if the def is a load.
    bool isReMaterializable(const LiveInterval &li, const VNInfo *ValNo,
                            MachineInstr *MI, bool &isLoad);

    /// tryFoldMemoryOperand - Attempts to fold either a spill / restore from
    /// slot / to reg or any rematerialized load into ith operand of specified
    /// MI. If it is successul, MI is updated with the newly created MI and
    /// returns true.
    bool tryFoldMemoryOperand(MachineInstr* &MI, VirtRegMap &vrm,
                              MachineInstr *DefMI, unsigned InstrIdx,
                              SmallVector<unsigned, 2> &Ops,
                              bool isSS, int Slot, unsigned Reg);

    /// canFoldMemoryOperand - Returns true if the specified load / store
    /// folding is possible.
    bool canFoldMemoryOperand(MachineInstr *MI,
                              SmallVector<unsigned, 2> &Ops) const;

    /// anyKillInMBBAfterIdx - Returns true if there is a kill of the specified
    /// VNInfo that's after the specified index but is within the basic block.
    bool anyKillInMBBAfterIdx(const LiveInterval &li, const VNInfo *VNI,
                              MachineBasicBlock *MBB, unsigned Idx) const;

    /// intervalIsInOneMBB - Returns true if the specified interval is entirely
    /// within a single basic block.
    bool intervalIsInOneMBB(const LiveInterval &li) const;

    /// SRInfo - Spill / restore info.
    struct SRInfo {
      int index;
      unsigned vreg;
      bool canFold;
      SRInfo(int i, unsigned vr, bool f) : index(i), vreg(vr), canFold(f) {};
    };

    bool alsoFoldARestore(int Id, int index, unsigned vr,
                          BitVector &RestoreMBBs,
                          std::map<unsigned,std::vector<SRInfo> >&RestoreIdxes);
    void eraseRestoreInfo(int Id, int index, unsigned vr,
                          BitVector &RestoreMBBs,
                          std::map<unsigned,std::vector<SRInfo> >&RestoreIdxes);

    /// rewriteInstructionForSpills, rewriteInstructionsForSpills - Helper
    /// functions for addIntervalsForSpills to rewrite uses / defs for the given
    /// live range.
    bool rewriteInstructionForSpills(const LiveInterval &li, bool TrySplit,
        unsigned id, unsigned index, unsigned end, MachineInstr *MI,
        MachineInstr *OrigDefMI, MachineInstr *DefMI, unsigned Slot, int LdSlot,
        bool isLoad, bool isLoadSS, bool DefIsReMat, bool CanDelete,
        VirtRegMap &vrm, MachineRegisterInfo &RegMap, 
                                     const TargetRegisterClass* rc,
        SmallVector<int, 4> &ReMatIds,
        unsigned &NewVReg, bool &HasDef, bool &HasUse,
        const MachineLoopInfo *loopInfo,
        std::map<unsigned,unsigned> &MBBVRegsMap,
        std::vector<LiveInterval*> &NewLIs);
    void rewriteInstructionsForSpills(const LiveInterval &li, bool TrySplit,
        LiveInterval::Ranges::const_iterator &I,
        MachineInstr *OrigDefMI, MachineInstr *DefMI, unsigned Slot, int LdSlot,
        bool isLoad, bool isLoadSS, bool DefIsReMat, bool CanDelete,
        VirtRegMap &vrm, MachineRegisterInfo &RegMap,
                                      const TargetRegisterClass* rc,
        SmallVector<int, 4> &ReMatIds, const MachineLoopInfo *loopInfo,
        BitVector &SpillMBBs,
        std::map<unsigned,std::vector<SRInfo> > &SpillIdxes,
        BitVector &RestoreMBBs,
        std::map<unsigned,std::vector<SRInfo> > &RestoreIdxes,
        std::map<unsigned,unsigned> &MBBVRegsMap,
        std::vector<LiveInterval*> &NewLIs);

    static LiveInterval createInterval(unsigned Reg);

    void printRegName(unsigned reg) const;
  };

} // End llvm namespace

#endif
