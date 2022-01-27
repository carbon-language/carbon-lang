//==- RegAllocGreedy.h ------- greedy register allocator  ----------*-C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file defines the RAGreedy function pass for register allocation in
// optimized builds.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_REGALLOCGREEDY_H_
#define LLVM_CODEGEN_REGALLOCGREEDY_H_

#include "AllocationOrder.h"
#include "InterferenceCache.h"
#include "LiveDebugVariables.h"
#include "RegAllocBase.h"
#include "RegAllocEvictionAdvisor.h"
#include "SpillPlacement.h"
#include "SplitKit.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/EdgeBundles.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervalUnion.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRangeEdit.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/LiveStacks.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/Spiller.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Pass.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Target/TargetMachine.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <queue>
#include <tuple>
#include <utility>

namespace llvm {
class LLVM_LIBRARY_VISIBILITY RAGreedy : public MachineFunctionPass,
                                         public RegAllocBase,
                                         private LiveRangeEdit::Delegate {
  // Interface to eviction advisers
public:
  /// Track allocation stage and eviction loop prevention during allocation.
  class ExtraRegInfo final {
    // RegInfo - Keep additional information about each live range.
    struct RegInfo {
      LiveRangeStage Stage = RS_New;

      // Cascade - Eviction loop prevention. See
      // canEvictInterferenceBasedOnCost().
      unsigned Cascade = 0;

      RegInfo() = default;
    };

    IndexedMap<RegInfo, VirtReg2IndexFunctor> Info;
    unsigned NextCascade = 1;

  public:
    ExtraRegInfo() = default;
    ExtraRegInfo(const ExtraRegInfo &) = delete;

    LiveRangeStage getStage(Register Reg) const { return Info[Reg].Stage; }

    LiveRangeStage getStage(const LiveInterval &VirtReg) const {
      return getStage(VirtReg.reg());
    }

    void setStage(Register Reg, LiveRangeStage Stage) {
      Info.grow(Reg.id());
      Info[Reg].Stage = Stage;
    }

    void setStage(const LiveInterval &VirtReg, LiveRangeStage Stage) {
      setStage(VirtReg.reg(), Stage);
    }

    /// Return the current stage of the register, if present, otherwise
    /// initialize it and return that.
    LiveRangeStage getOrInitStage(Register Reg) {
      Info.grow(Reg.id());
      return getStage(Reg);
    }

    unsigned getCascade(Register Reg) const { return Info[Reg].Cascade; }

    void setCascade(Register Reg, unsigned Cascade) {
      Info.grow(Reg.id());
      Info[Reg].Cascade = Cascade;
    }

    unsigned getOrAssignNewCascade(Register Reg) {
      unsigned Cascade = getCascade(Reg);
      if (!Cascade) {
        Cascade = NextCascade++;
        setCascade(Reg, Cascade);
      }
      return Cascade;
    }

    unsigned getCascadeOrCurrentNext(Register Reg) const {
      unsigned Cascade = getCascade(Reg);
      if (!Cascade)
        Cascade = NextCascade;
      return Cascade;
    }

    template <typename Iterator>
    void setStage(Iterator Begin, Iterator End, LiveRangeStage NewStage) {
      for (; Begin != End; ++Begin) {
        Register Reg = *Begin;
        Info.grow(Reg.id());
        if (Info[Reg].Stage == RS_New)
          Info[Reg].Stage = NewStage;
      }
    }
    void LRE_DidCloneVirtReg(Register New, Register Old);
  };

  LiveRegMatrix *getInterferenceMatrix() const { return Matrix; }
  LiveIntervals *getLiveIntervals() const { return LIS; }
  VirtRegMap *getVirtRegMap() const { return VRM; }
  const RegisterClassInfo &getRegClassInfo() const { return RegClassInfo; }
  const ExtraRegInfo &getExtraInfo() const { return *ExtraInfo; }
  // end (interface to eviction advisers)

private:
  // Convenient shortcuts.
  using PQueue = std::priority_queue<std::pair<unsigned, unsigned>>;
  using SmallLISet = SmallPtrSet<LiveInterval *, 4>;

  // context
  MachineFunction *MF;

  // Shortcuts to some useful interface.
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;
  RegisterClassInfo RCI;

  // analyses
  SlotIndexes *Indexes;
  MachineBlockFrequencyInfo *MBFI;
  MachineDominatorTree *DomTree;
  MachineLoopInfo *Loops;
  MachineOptimizationRemarkEmitter *ORE;
  EdgeBundles *Bundles;
  SpillPlacement *SpillPlacer;
  LiveDebugVariables *DebugVars;
  AliasAnalysis *AA;

  // state
  std::unique_ptr<Spiller> SpillerInstance;
  PQueue Queue;
  std::unique_ptr<VirtRegAuxInfo> VRAI;
  Optional<ExtraRegInfo> ExtraInfo;
  std::unique_ptr<RegAllocEvictionAdvisor> EvictAdvisor;

  // Enum CutOffStage to keep a track whether the register allocation failed
  // because of the cutoffs encountered in last chance recoloring.
  // Note: This is used as bitmask. New value should be next power of 2.
  enum CutOffStage {
    // No cutoffs encountered
    CO_None = 0,

    // lcr-max-depth cutoff encountered
    CO_Depth = 1,

    // lcr-max-interf cutoff encountered
    CO_Interf = 2
  };

  uint8_t CutOffInfo;

#ifndef NDEBUG
  static const char *const StageName[];
#endif

  /// EvictionTrack - Keeps track of past evictions in order to optimize region
  /// split decision.
  class EvictionTrack {

  public:
    using EvictorInfo =
        std::pair<Register /* evictor */, MCRegister /* physreg */>;
    using EvicteeInfo = llvm::DenseMap<Register /* evictee */, EvictorInfo>;

  private:
    /// Each Vreg that has been evicted in the last stage of selectOrSplit will
    /// be mapped to the evictor Vreg and the PhysReg it was evicted from.
    EvicteeInfo Evictees;

  public:
    /// Clear all eviction information.
    void clear() { Evictees.clear(); }

    ///  Clear eviction information for the given evictee Vreg.
    /// E.g. when Vreg get's a new allocation, the old eviction info is no
    /// longer relevant.
    /// \param Evictee The evictee Vreg for whom we want to clear collected
    /// eviction info.
    void clearEvicteeInfo(Register Evictee) { Evictees.erase(Evictee); }

    /// Track new eviction.
    /// The Evictor vreg has evicted the Evictee vreg from Physreg.
    /// \param PhysReg The physical register Evictee was evicted from.
    /// \param Evictor The evictor Vreg that evicted Evictee.
    /// \param Evictee The evictee Vreg.
    void addEviction(MCRegister PhysReg, Register Evictor, Register Evictee) {
      Evictees[Evictee].first = Evictor;
      Evictees[Evictee].second = PhysReg;
    }

    /// Return the Evictor Vreg which evicted Evictee Vreg from PhysReg.
    /// \param Evictee The evictee vreg.
    /// \return The Evictor vreg which evicted Evictee vreg from PhysReg. 0 if
    /// nobody has evicted Evictee from PhysReg.
    EvictorInfo getEvictor(Register Evictee) {
      if (Evictees.count(Evictee)) {
        return Evictees[Evictee];
      }

      return EvictorInfo(0, 0);
    }
  };

  // Keeps track of past evictions in order to optimize region split decision.
  EvictionTrack LastEvicted;

  // splitting state.
  std::unique_ptr<SplitAnalysis> SA;
  std::unique_ptr<SplitEditor> SE;

  /// Cached per-block interference maps
  InterferenceCache IntfCache;

  /// All basic blocks where the current register has uses.
  SmallVector<SpillPlacement::BlockConstraint, 8> SplitConstraints;

  /// Global live range splitting candidate info.
  struct GlobalSplitCandidate {
    // Register intended for assignment, or 0.
    MCRegister PhysReg;

    // SplitKit interval index for this candidate.
    unsigned IntvIdx;

    // Interference for PhysReg.
    InterferenceCache::Cursor Intf;

    // Bundles where this candidate should be live.
    BitVector LiveBundles;
    SmallVector<unsigned, 8> ActiveBlocks;

    void reset(InterferenceCache &Cache, MCRegister Reg) {
      PhysReg = Reg;
      IntvIdx = 0;
      Intf.setPhysReg(Cache, Reg);
      LiveBundles.clear();
      ActiveBlocks.clear();
    }

    // Set B[I] = C for every live bundle where B[I] was NoCand.
    unsigned getBundles(SmallVectorImpl<unsigned> &B, unsigned C) {
      unsigned Count = 0;
      for (unsigned I : LiveBundles.set_bits())
        if (B[I] == NoCand) {
          B[I] = C;
          Count++;
        }
      return Count;
    }
  };

  /// Candidate info for each PhysReg in AllocationOrder.
  /// This vector never shrinks, but grows to the size of the largest register
  /// class.
  SmallVector<GlobalSplitCandidate, 32> GlobalCand;

  enum : unsigned { NoCand = ~0u };

  /// Candidate map. Each edge bundle is assigned to a GlobalCand entry, or to
  /// NoCand which indicates the stack interval.
  SmallVector<unsigned, 32> BundleCand;

  /// Callee-save register cost, calculated once per machine function.
  BlockFrequency CSRCost;

  /// Enable or not the consideration of the cost of local intervals created
  /// by a split candidate when choosing the best split candidate.
  bool EnableAdvancedRASplitCost;

  /// Set of broken hints that may be reconciled later because of eviction.
  SmallSetVector<LiveInterval *, 8> SetOfBrokenHints;

  /// The register cost values. This list will be recreated for each Machine
  /// Function
  ArrayRef<uint8_t> RegCosts;

public:
  RAGreedy(const RegClassFilterFunc F = allocateAllRegClasses);

  /// Return the pass name.
  StringRef getPassName() const override { return "Greedy Register Allocator"; }

  /// RAGreedy analysis usage.
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  void releaseMemory() override;
  Spiller &spiller() override { return *SpillerInstance; }
  void enqueueImpl(LiveInterval *LI) override;
  LiveInterval *dequeue() override;
  MCRegister selectOrSplit(LiveInterval &,
                           SmallVectorImpl<Register> &) override;
  void aboutToRemoveInterval(LiveInterval &) override;

  /// Perform register allocation.
  bool runOnMachineFunction(MachineFunction &mf) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoPHIs);
  }

  MachineFunctionProperties getClearedProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::IsSSA);
  }

  static char ID;

private:
  MCRegister selectOrSplitImpl(LiveInterval &, SmallVectorImpl<Register> &,
                               SmallVirtRegSet &, unsigned = 0);

  bool LRE_CanEraseVirtReg(Register) override;
  void LRE_WillShrinkVirtReg(Register) override;
  void LRE_DidCloneVirtReg(Register, Register) override;
  void enqueue(PQueue &CurQueue, LiveInterval *LI);
  LiveInterval *dequeue(PQueue &CurQueue);

  BlockFrequency calcSpillCost();
  bool addSplitConstraints(InterferenceCache::Cursor, BlockFrequency &);
  bool addThroughConstraints(InterferenceCache::Cursor, ArrayRef<unsigned>);
  bool growRegion(GlobalSplitCandidate &Cand);
  bool splitCanCauseEvictionChain(Register Evictee, GlobalSplitCandidate &Cand,
                                  unsigned BBNumber,
                                  const AllocationOrder &Order);
  bool splitCanCauseLocalSpill(unsigned VirtRegToSplit,
                               GlobalSplitCandidate &Cand, unsigned BBNumber,
                               const AllocationOrder &Order);
  BlockFrequency calcGlobalSplitCost(GlobalSplitCandidate &,
                                     const AllocationOrder &Order,
                                     bool *CanCauseEvictionChain);
  bool calcCompactRegion(GlobalSplitCandidate &);
  void splitAroundRegion(LiveRangeEdit &, ArrayRef<unsigned>);
  void calcGapWeights(MCRegister, SmallVectorImpl<float> &);
  bool canEvictInterferenceInRange(const LiveInterval &VirtReg,
                                   MCRegister PhysReg, SlotIndex Start,
                                   SlotIndex End, EvictionCost &MaxCost) const;
  MCRegister getCheapestEvicteeWeight(const AllocationOrder &Order,
                                      const LiveInterval &VirtReg,
                                      SlotIndex Start, SlotIndex End,
                                      float *BestEvictWeight) const;
  void evictInterference(LiveInterval &, MCRegister,
                         SmallVectorImpl<Register> &);
  bool mayRecolorAllInterferences(MCRegister PhysReg, LiveInterval &VirtReg,
                                  SmallLISet &RecoloringCandidates,
                                  const SmallVirtRegSet &FixedRegisters);

  MCRegister tryAssign(LiveInterval &, AllocationOrder &,
                       SmallVectorImpl<Register> &, const SmallVirtRegSet &);
  MCRegister tryEvict(LiveInterval &, AllocationOrder &,
                      SmallVectorImpl<Register> &, uint8_t,
                      const SmallVirtRegSet &);
  MCRegister tryRegionSplit(LiveInterval &, AllocationOrder &,
                            SmallVectorImpl<Register> &);
  /// Calculate cost of region splitting.
  unsigned calculateRegionSplitCost(LiveInterval &VirtReg,
                                    AllocationOrder &Order,
                                    BlockFrequency &BestCost,
                                    unsigned &NumCands, bool IgnoreCSR,
                                    bool *CanCauseEvictionChain = nullptr);
  /// Perform region splitting.
  unsigned doRegionSplit(LiveInterval &VirtReg, unsigned BestCand,
                         bool HasCompact, SmallVectorImpl<Register> &NewVRegs);
  /// Check other options before using a callee-saved register for the first
  /// time.
  MCRegister tryAssignCSRFirstTime(LiveInterval &VirtReg,
                                   AllocationOrder &Order, MCRegister PhysReg,
                                   uint8_t &CostPerUseLimit,
                                   SmallVectorImpl<Register> &NewVRegs);
  void initializeCSRCost();
  unsigned tryBlockSplit(LiveInterval &, AllocationOrder &,
                         SmallVectorImpl<Register> &);
  unsigned tryInstructionSplit(LiveInterval &, AllocationOrder &,
                               SmallVectorImpl<Register> &);
  unsigned tryLocalSplit(LiveInterval &, AllocationOrder &,
                         SmallVectorImpl<Register> &);
  unsigned trySplit(LiveInterval &, AllocationOrder &,
                    SmallVectorImpl<Register> &, const SmallVirtRegSet &);
  unsigned tryLastChanceRecoloring(LiveInterval &, AllocationOrder &,
                                   SmallVectorImpl<Register> &,
                                   SmallVirtRegSet &, unsigned);
  bool tryRecoloringCandidates(PQueue &, SmallVectorImpl<Register> &,
                               SmallVirtRegSet &, unsigned);
  void tryHintRecoloring(LiveInterval &);
  void tryHintsRecoloring();

  /// Model the information carried by one end of a copy.
  struct HintInfo {
    /// The frequency of the copy.
    BlockFrequency Freq;
    /// The virtual register or physical register.
    Register Reg;
    /// Its currently assigned register.
    /// In case of a physical register Reg == PhysReg.
    MCRegister PhysReg;

    HintInfo(BlockFrequency Freq, Register Reg, MCRegister PhysReg)
        : Freq(Freq), Reg(Reg), PhysReg(PhysReg) {}
  };
  using HintsInfo = SmallVector<HintInfo, 4>;

  BlockFrequency getBrokenHintFreq(const HintsInfo &, MCRegister);
  void collectHintInfo(Register, HintsInfo &);

  /// Greedy RA statistic to remark.
  struct RAGreedyStats {
    unsigned Reloads = 0;
    unsigned FoldedReloads = 0;
    unsigned ZeroCostFoldedReloads = 0;
    unsigned Spills = 0;
    unsigned FoldedSpills = 0;
    unsigned Copies = 0;
    float ReloadsCost = 0.0f;
    float FoldedReloadsCost = 0.0f;
    float SpillsCost = 0.0f;
    float FoldedSpillsCost = 0.0f;
    float CopiesCost = 0.0f;

    bool isEmpty() {
      return !(Reloads || FoldedReloads || Spills || FoldedSpills ||
               ZeroCostFoldedReloads || Copies);
    }

    void add(RAGreedyStats other) {
      Reloads += other.Reloads;
      FoldedReloads += other.FoldedReloads;
      ZeroCostFoldedReloads += other.ZeroCostFoldedReloads;
      Spills += other.Spills;
      FoldedSpills += other.FoldedSpills;
      Copies += other.Copies;
      ReloadsCost += other.ReloadsCost;
      FoldedReloadsCost += other.FoldedReloadsCost;
      SpillsCost += other.SpillsCost;
      FoldedSpillsCost += other.FoldedSpillsCost;
      CopiesCost += other.CopiesCost;
    }

    void report(MachineOptimizationRemarkMissed &R);
  };

  /// Compute statistic for a basic block.
  RAGreedyStats computeStats(MachineBasicBlock &MBB);

  /// Compute and report statistic through a remark.
  RAGreedyStats reportStats(MachineLoop *L);

  /// Report the statistic for each loop.
  void reportStats();
};
} // namespace llvm
#endif // #ifndef LLVM_CODEGEN_REGALLOCGREEDY_H_
