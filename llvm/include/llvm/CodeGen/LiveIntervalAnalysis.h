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
// instruction with number j' > j such that v is live at j' and there is no
// instruction with number i' < i such that v is live at i'. In this
// implementation intervals can have holes, i.e. an interval might look like
// [1,20), [50,65), [1000,1001).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LIVEINTERVAL_ANALYSIS_H
#define LLVM_CODEGEN_LIVEINTERVAL_ANALYSIS_H

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"
#include <cmath>
#include <iterator>

namespace llvm {

  class AliasAnalysis;
  class LiveVariables;
  class MachineLoopInfo;
  class TargetRegisterInfo;
  class MachineRegisterInfo;
  class TargetInstrInfo;
  class TargetRegisterClass;
  class VirtRegMap;

  class LiveIntervals : public MachineFunctionPass {
    MachineFunction* mf_;
    MachineRegisterInfo* mri_;
    const TargetMachine* tm_;
    const TargetRegisterInfo* tri_;
    const TargetInstrInfo* tii_;
    AliasAnalysis *aa_;
    LiveVariables* lv_;
    SlotIndexes* indexes_;

    /// Special pool allocator for VNInfo's (LiveInterval val#).
    ///
    VNInfo::Allocator VNInfoAllocator;

    typedef DenseMap<unsigned, LiveInterval*> Reg2IntervalMap;
    Reg2IntervalMap r2iMap_;

    /// allocatableRegs_ - A bit vector of allocatable registers.
    BitVector allocatableRegs_;

    /// CloneMIs - A list of clones as result of re-materialization.
    std::vector<MachineInstr*> CloneMIs;

  public:
    static char ID; // Pass identification, replacement for typeid
    LiveIntervals() : MachineFunctionPass(ID) {
      initializeLiveIntervalsPass(*PassRegistry::getPassRegistry());
    }

    // Calculate the spill weight to assign to a single instruction.
    static float getSpillWeight(bool isDef, bool isUse, unsigned loopDepth);

    // After summing the spill weights of all defs and uses, the final weight
    // should be normalized, dividing the weight of the interval by its size.
    // This encourages spilling of intervals that are large and have few uses,
    // and discourages spilling of small intervals with many uses.
    void normalizeSpillWeight(LiveInterval &li) {
      li.weight /= getApproximateInstructionCount(li) + 25;
    }

    typedef Reg2IntervalMap::iterator iterator;
    typedef Reg2IntervalMap::const_iterator const_iterator;
    const_iterator begin() const { return r2iMap_.begin(); }
    const_iterator end() const { return r2iMap_.end(); }
    iterator begin() { return r2iMap_.begin(); }
    iterator end() { return r2iMap_.end(); }
    unsigned getNumIntervals() const { return (unsigned)r2iMap_.size(); }

    LiveInterval &getInterval(unsigned reg) {
      Reg2IntervalMap::iterator I = r2iMap_.find(reg);
      assert(I != r2iMap_.end() && "Interval does not exist for register");
      return *I->second;
    }

    const LiveInterval &getInterval(unsigned reg) const {
      Reg2IntervalMap::const_iterator I = r2iMap_.find(reg);
      assert(I != r2iMap_.end() && "Interval does not exist for register");
      return *I->second;
    }

    bool hasInterval(unsigned reg) const {
      return r2iMap_.count(reg);
    }

    /// isAllocatable - is the physical register reg allocatable in the current
    /// function?
    bool isAllocatable(unsigned reg) const {
      return allocatableRegs_.test(reg);
    }

    /// getScaledIntervalSize - get the size of an interval in "units,"
    /// where every function is composed of one thousand units.  This
    /// measure scales properly with empty index slots in the function.
    double getScaledIntervalSize(LiveInterval& I) {
      return (1000.0 * I.getSize()) / indexes_->getIndexesLength();
    }

    /// getFuncInstructionCount - Return the number of instructions in the
    /// current function.
    unsigned getFuncInstructionCount() {
      return indexes_->getFunctionSize();
    }

    /// getApproximateInstructionCount - computes an estimate of the number
    /// of instructions in a given LiveInterval.
    unsigned getApproximateInstructionCount(LiveInterval& I) {
      double IntervalPercentage = getScaledIntervalSize(I) / 1000.0;
      return (unsigned)(IntervalPercentage * indexes_->getFunctionSize());
    }

    /// conflictsWithPhysReg - Returns true if the specified register is used or
    /// defined during the duration of the specified interval. Copies to and
    /// from li.reg are allowed. This method is only able to analyze simple
    /// ranges that stay within a single basic block. Anything else is
    /// considered a conflict.
    bool conflictsWithPhysReg(const LiveInterval &li, VirtRegMap &vrm,
                              unsigned reg);

    /// conflictsWithAliasRef - Similar to conflictsWithPhysRegRef except
    /// it checks for alias uses and defs.
    bool conflictsWithAliasRef(LiveInterval &li, unsigned Reg,
                                   SmallPtrSet<MachineInstr*,32> &JoinedCopies);

    // Interval creation
    LiveInterval &getOrCreateInterval(unsigned reg) {
      Reg2IntervalMap::iterator I = r2iMap_.find(reg);
      if (I == r2iMap_.end())
        I = r2iMap_.insert(std::make_pair(reg, createInterval(reg))).first;
      return *I->second;
    }

    /// dupInterval - Duplicate a live interval. The caller is responsible for
    /// managing the allocated memory.
    LiveInterval *dupInterval(LiveInterval *li);

    /// addLiveRangeToEndOfBlock - Given a register and an instruction,
    /// adds a live range from that instruction to the end of its MBB.
    LiveRange addLiveRangeToEndOfBlock(unsigned reg,
                                       MachineInstr* startInst);

    // Interval removal

    void removeInterval(unsigned Reg) {
      DenseMap<unsigned, LiveInterval*>::iterator I = r2iMap_.find(Reg);
      delete I->second;
      r2iMap_.erase(I);
    }

    SlotIndexes *getSlotIndexes() const {
      return indexes_;
    }

    SlotIndex getZeroIndex() const {
      return indexes_->getZeroIndex();
    }

    SlotIndex getInvalidIndex() const {
      return indexes_->getInvalidIndex();
    }

    /// isNotInMIMap - returns true if the specified machine instr has been
    /// removed or was never entered in the map.
    bool isNotInMIMap(const MachineInstr* Instr) const {
      return !indexes_->hasIndex(Instr);
    }

    /// Returns the base index of the given instruction.
    SlotIndex getInstructionIndex(const MachineInstr *instr) const {
      return indexes_->getInstructionIndex(instr);
    }

    /// Returns the instruction associated with the given index.
    MachineInstr* getInstructionFromIndex(SlotIndex index) const {
      return indexes_->getInstructionFromIndex(index);
    }

    /// Return the first index in the given basic block.
    SlotIndex getMBBStartIdx(const MachineBasicBlock *mbb) const {
      return indexes_->getMBBStartIdx(mbb);
    }

    /// Return the last index in the given basic block.
    SlotIndex getMBBEndIdx(const MachineBasicBlock *mbb) const {
      return indexes_->getMBBEndIdx(mbb);
    }

    bool isLiveInToMBB(const LiveInterval &li,
                       const MachineBasicBlock *mbb) const {
      return li.liveAt(getMBBStartIdx(mbb));
    }

    LiveRange* findEnteringRange(LiveInterval &li,
                                 const MachineBasicBlock *mbb) {
      return li.getLiveRangeContaining(getMBBStartIdx(mbb));
    }

    bool isLiveOutOfMBB(const LiveInterval &li,
                        const MachineBasicBlock *mbb) const {
      return li.liveAt(getMBBEndIdx(mbb).getPrevSlot());
    }

    LiveRange* findExitingRange(LiveInterval &li,
                                const MachineBasicBlock *mbb) {
      return li.getLiveRangeContaining(getMBBEndIdx(mbb).getPrevSlot());
    }

    MachineBasicBlock* getMBBFromIndex(SlotIndex index) const {
      return indexes_->getMBBFromIndex(index);
    }

    SlotIndex InsertMachineInstrInMaps(MachineInstr *MI) {
      return indexes_->insertMachineInstrInMaps(MI);
    }

    void RemoveMachineInstrFromMaps(MachineInstr *MI) {
      indexes_->removeMachineInstrFromMaps(MI);
    }

    void ReplaceMachineInstrInMaps(MachineInstr *MI, MachineInstr *NewMI) {
      indexes_->replaceMachineInstrInMaps(MI, NewMI);
    }

    void InsertMBBInMaps(MachineBasicBlock *MBB) {
      indexes_->insertMBBInMaps(MBB);
    }

    bool findLiveInMBBs(SlotIndex Start, SlotIndex End,
                        SmallVectorImpl<MachineBasicBlock*> &MBBs) const {
      return indexes_->findLiveInMBBs(Start, End, MBBs);
    }

    void renumber() {
      indexes_->renumberIndexes();
    }

    VNInfo::Allocator& getVNInfoAllocator() { return VNInfoAllocator; }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    virtual void releaseMemory();

    /// runOnMachineFunction - pass entry point
    virtual bool runOnMachineFunction(MachineFunction&);

    /// print - Implement the dump method.
    virtual void print(raw_ostream &O, const Module* = 0) const;

    /// addIntervalsForSpills - Create new intervals for spilled defs / uses of
    /// the given interval. FIXME: It also returns the weight of the spill slot
    /// (if any is created) by reference. This is temporary.
    std::vector<LiveInterval*>
    addIntervalsForSpills(const LiveInterval& i,
                          const SmallVectorImpl<LiveInterval*> &SpillIs,
                          const MachineLoopInfo *loopInfo, VirtRegMap& vrm);

    /// spillPhysRegAroundRegDefsUses - Spill the specified physical register
    /// around all defs and uses of the specified interval. Return true if it
    /// was able to cut its interval.
    bool spillPhysRegAroundRegDefsUses(const LiveInterval &li,
                                       unsigned PhysReg, VirtRegMap &vrm);

    /// isReMaterializable - Returns true if every definition of MI of every
    /// val# of the specified interval is re-materializable. Also returns true
    /// by reference if all of the defs are load instructions.
    bool isReMaterializable(const LiveInterval &li,
                            const SmallVectorImpl<LiveInterval*> &SpillIs,
                            bool &isLoad);

    /// isReMaterializable - Returns true if the definition MI of the specified
    /// val# of the specified interval is re-materializable.
    bool isReMaterializable(const LiveInterval &li, const VNInfo *ValNo,
                            MachineInstr *MI);

    /// getRepresentativeReg - Find the largest super register of the specified
    /// physical register.
    unsigned getRepresentativeReg(unsigned Reg) const;

    /// getNumConflictsWithPhysReg - Return the number of uses and defs of the
    /// specified interval that conflicts with the specified physical register.
    unsigned getNumConflictsWithPhysReg(const LiveInterval &li,
                                        unsigned PhysReg) const;

    /// intervalIsInOneMBB - Returns true if the specified interval is entirely
    /// within a single basic block.
    bool intervalIsInOneMBB(const LiveInterval &li) const;

  private:
    /// computeIntervals - Compute live intervals.
    void computeIntervals();

    /// handleRegisterDef - update intervals for a register def
    /// (calls handlePhysicalRegisterDef and
    /// handleVirtualRegisterDef)
    void handleRegisterDef(MachineBasicBlock *MBB,
                           MachineBasicBlock::iterator MI,
                           SlotIndex MIIdx,
                           MachineOperand& MO, unsigned MOIdx);

    /// isPartialRedef - Return true if the specified def at the specific index
    /// is partially re-defining the specified live interval. A common case of
    /// this is a definition of the sub-register.
    bool isPartialRedef(SlotIndex MIIdx, MachineOperand &MO,
                        LiveInterval &interval);

    /// handleVirtualRegisterDef - update intervals for a virtual
    /// register def
    void handleVirtualRegisterDef(MachineBasicBlock *MBB,
                                  MachineBasicBlock::iterator MI,
                                  SlotIndex MIIdx, MachineOperand& MO,
                                  unsigned MOIdx,
                                  LiveInterval& interval);

    /// handlePhysicalRegisterDef - update intervals for a physical register
    /// def.
    void handlePhysicalRegisterDef(MachineBasicBlock* mbb,
                                   MachineBasicBlock::iterator mi,
                                   SlotIndex MIIdx, MachineOperand& MO,
                                   LiveInterval &interval,
                                   MachineInstr *CopyMI);

    /// handleLiveInRegister - Create interval for a livein register.
    void handleLiveInRegister(MachineBasicBlock* mbb,
                              SlotIndex MIIdx,
                              LiveInterval &interval, bool isAlias = false);

    /// getReMatImplicitUse - If the remat definition MI has one (for now, we
    /// only allow one) virtual register operand, then its uses are implicitly
    /// using the register. Returns the virtual register.
    unsigned getReMatImplicitUse(const LiveInterval &li,
                                 MachineInstr *MI) const;

    /// isValNoAvailableAt - Return true if the val# of the specified interval
    /// which reaches the given instruction also reaches the specified use
    /// index.
    bool isValNoAvailableAt(const LiveInterval &li, MachineInstr *MI,
                            SlotIndex UseIdx) const;

    /// isReMaterializable - Returns true if the definition MI of the specified
    /// val# of the specified interval is re-materializable. Also returns true
    /// by reference if the def is a load.
    bool isReMaterializable(const LiveInterval &li, const VNInfo *ValNo,
                            MachineInstr *MI,
                            const SmallVectorImpl<LiveInterval*> &SpillIs,
                            bool &isLoad);

    /// tryFoldMemoryOperand - Attempts to fold either a spill / restore from
    /// slot / to reg or any rematerialized load into ith operand of specified
    /// MI. If it is successul, MI is updated with the newly created MI and
    /// returns true.
    bool tryFoldMemoryOperand(MachineInstr* &MI, VirtRegMap &vrm,
                              MachineInstr *DefMI, SlotIndex InstrIdx,
                              SmallVector<unsigned, 2> &Ops,
                              bool isSS, int FrameIndex, unsigned Reg);

    /// canFoldMemoryOperand - Return true if the specified load / store
    /// folding is possible.
    bool canFoldMemoryOperand(MachineInstr *MI,
                              SmallVector<unsigned, 2> &Ops,
                              bool ReMatLoadSS) const;

    /// anyKillInMBBAfterIdx - Returns true if there is a kill of the specified
    /// VNInfo that's after the specified index but is within the basic block.
    bool anyKillInMBBAfterIdx(const LiveInterval &li, const VNInfo *VNI,
                              MachineBasicBlock *MBB,
                              SlotIndex Idx) const;

    /// hasAllocatableSuperReg - Return true if the specified physical register
    /// has any super register that's allocatable.
    bool hasAllocatableSuperReg(unsigned Reg) const;

    /// SRInfo - Spill / restore info.
    struct SRInfo {
      SlotIndex index;
      unsigned vreg;
      bool canFold;
      SRInfo(SlotIndex i, unsigned vr, bool f)
        : index(i), vreg(vr), canFold(f) {}
    };

    bool alsoFoldARestore(int Id, SlotIndex index, unsigned vr,
                          BitVector &RestoreMBBs,
                          DenseMap<unsigned,std::vector<SRInfo> >&RestoreIdxes);
    void eraseRestoreInfo(int Id, SlotIndex index, unsigned vr,
                          BitVector &RestoreMBBs,
                          DenseMap<unsigned,std::vector<SRInfo> >&RestoreIdxes);

    /// handleSpilledImpDefs - Remove IMPLICIT_DEF instructions which are being
    /// spilled and create empty intervals for their uses.
    void handleSpilledImpDefs(const LiveInterval &li, VirtRegMap &vrm,
                              const TargetRegisterClass* rc,
                              std::vector<LiveInterval*> &NewLIs);

    /// rewriteImplicitOps - Rewrite implicit use operands of MI (i.e. uses of
    /// interval on to-be re-materialized operands of MI) with new register.
    void rewriteImplicitOps(const LiveInterval &li,
                           MachineInstr *MI, unsigned NewVReg, VirtRegMap &vrm);

    /// rewriteInstructionForSpills, rewriteInstructionsForSpills - Helper
    /// functions for addIntervalsForSpills to rewrite uses / defs for the given
    /// live range.
    bool rewriteInstructionForSpills(const LiveInterval &li, const VNInfo *VNI,
        bool TrySplit, SlotIndex index, SlotIndex end,
        MachineInstr *MI, MachineInstr *OrigDefMI, MachineInstr *DefMI,
        unsigned Slot, int LdSlot,
        bool isLoad, bool isLoadSS, bool DefIsReMat, bool CanDelete,
        VirtRegMap &vrm, const TargetRegisterClass* rc,
        SmallVector<int, 4> &ReMatIds, const MachineLoopInfo *loopInfo,
        unsigned &NewVReg, unsigned ImpUse, bool &HasDef, bool &HasUse,
        DenseMap<unsigned,unsigned> &MBBVRegsMap,
        std::vector<LiveInterval*> &NewLIs);
    void rewriteInstructionsForSpills(const LiveInterval &li, bool TrySplit,
        LiveInterval::Ranges::const_iterator &I,
        MachineInstr *OrigDefMI, MachineInstr *DefMI, unsigned Slot, int LdSlot,
        bool isLoad, bool isLoadSS, bool DefIsReMat, bool CanDelete,
        VirtRegMap &vrm, const TargetRegisterClass* rc,
        SmallVector<int, 4> &ReMatIds, const MachineLoopInfo *loopInfo,
        BitVector &SpillMBBs,
        DenseMap<unsigned,std::vector<SRInfo> > &SpillIdxes,
        BitVector &RestoreMBBs,
        DenseMap<unsigned,std::vector<SRInfo> > &RestoreIdxes,
        DenseMap<unsigned,unsigned> &MBBVRegsMap,
        std::vector<LiveInterval*> &NewLIs);

    // Normalize the spill weight of all the intervals in NewLIs.
    void normalizeSpillWeights(std::vector<LiveInterval*> &NewLIs);

    static LiveInterval* createInterval(unsigned Reg);

    void printInstrs(raw_ostream &O) const;
    void dumpInstrs() const;
  };
} // End llvm namespace

#endif
