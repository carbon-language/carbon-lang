//===-------- SplitKit.h - Toolkit for splitting live ranges ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the SplitAnalysis class as well as mutator functions for
// live range splitting.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/CodeGen/SlotIndexes.h"

namespace llvm {

class LiveInterval;
class LiveIntervals;
class LiveRangeEdit;
class MachineInstr;
class MachineLoop;
class MachineLoopInfo;
class MachineRegisterInfo;
class TargetInstrInfo;
class TargetRegisterInfo;
class VirtRegMap;
class VNInfo;
class raw_ostream;

/// At some point we should just include MachineDominators.h:
class MachineDominatorTree;
template <class NodeT> class DomTreeNodeBase;
typedef DomTreeNodeBase<MachineBasicBlock> MachineDomTreeNode;


/// SplitAnalysis - Analyze a LiveInterval, looking for live range splitting
/// opportunities.
class SplitAnalysis {
public:
  const MachineFunction &MF;
  const LiveIntervals &LIS;
  const MachineLoopInfo &Loops;
  const TargetInstrInfo &TII;

  // Instructions using the the current register.
  typedef SmallPtrSet<const MachineInstr*, 16> InstrPtrSet;
  InstrPtrSet UsingInstrs;

  // Sorted slot indexes of using instructions.
  SmallVector<SlotIndex, 8> UseSlots;

  // The number of instructions using CurLI in each basic block.
  typedef DenseMap<const MachineBasicBlock*, unsigned> BlockCountMap;
  BlockCountMap UsingBlocks;

  // The number of basic block using CurLI in each loop.
  typedef DenseMap<const MachineLoop*, unsigned> LoopCountMap;
  LoopCountMap UsingLoops;

private:
  // Current live interval.
  const LiveInterval *CurLI;

  // Sumarize statistics by counting instructions using CurLI.
  void analyzeUses();

  /// canAnalyzeBranch - Return true if MBB ends in a branch that can be
  /// analyzed.
  bool canAnalyzeBranch(const MachineBasicBlock *MBB);

public:
  SplitAnalysis(const MachineFunction &mf, const LiveIntervals &lis,
                const MachineLoopInfo &mli);

  /// analyze - set CurLI to the specified interval, and analyze how it may be
  /// split.
  void analyze(const LiveInterval *li);

  /// clear - clear all data structures so SplitAnalysis is ready to analyze a
  /// new interval.
  void clear();

  /// hasUses - Return true if MBB has any uses of CurLI.
  bool hasUses(const MachineBasicBlock *MBB) const {
    return UsingBlocks.lookup(MBB);
  }

  typedef SmallPtrSet<const MachineBasicBlock*, 16> BlockPtrSet;
  typedef SmallPtrSet<const MachineLoop*, 16> LoopPtrSet;

  // Print a set of blocks with use counts.
  void print(const BlockPtrSet&, raw_ostream&) const;

  // Sets of basic blocks surrounding a machine loop.
  struct LoopBlocks {
    BlockPtrSet Loop;  // Blocks in the loop.
    BlockPtrSet Preds; // Loop predecessor blocks.
    BlockPtrSet Exits; // Loop exit blocks.

    void clear() {
      Loop.clear();
      Preds.clear();
      Exits.clear();
    }
  };

  // Print loop blocks with use counts.
  void print(const LoopBlocks&, raw_ostream&) const;

  // Calculate the block sets surrounding the loop.
  void getLoopBlocks(const MachineLoop *Loop, LoopBlocks &Blocks);

  /// LoopPeripheralUse - how is a variable used in and around a loop?
  /// Peripheral blocks are the loop predecessors and exit blocks.
  enum LoopPeripheralUse {
    ContainedInLoop,  // All uses are inside the loop.
    SinglePeripheral, // At most one instruction per peripheral block.
    MultiPeripheral,  // Multiple instructions in some peripheral blocks.
    OutsideLoop       // Uses outside loop periphery.
  };

  /// analyzeLoopPeripheralUse - Return an enum describing how CurLI is used in
  /// and around the Loop.
  LoopPeripheralUse analyzeLoopPeripheralUse(const LoopBlocks&);

  /// getCriticalExits - It may be necessary to partially break critical edges
  /// leaving the loop if an exit block has phi uses of CurLI. Collect the exit
  /// blocks that need special treatment into CriticalExits.
  void getCriticalExits(const LoopBlocks &Blocks, BlockPtrSet &CriticalExits);

  /// canSplitCriticalExits - Return true if it is possible to insert new exit
  /// blocks before the blocks in CriticalExits.
  bool canSplitCriticalExits(const LoopBlocks &Blocks,
                             BlockPtrSet &CriticalExits);

  /// getCriticalPreds - Get the set of loop predecessors with critical edges to
  /// blocks outside the loop that have CurLI live in. We don't have to break
  /// these edges, but they do require special treatment.
  void getCriticalPreds(const LoopBlocks &Blocks, BlockPtrSet &CriticalPreds);

  /// getSplitLoops - Get the set of loops that have CurLI uses and would be
  /// profitable to split.
  void getSplitLoops(LoopPtrSet&);

  /// getBestSplitLoop - Return the loop where CurLI may best be split to a
  /// separate register, or NULL.
  const MachineLoop *getBestSplitLoop();

  /// isBypassLoop - Return true if CurLI is live through Loop and has no uses
  /// inside the loop. Bypass loops are candidates for splitting because it can
  /// prevent interference inside the loop.
  bool isBypassLoop(const MachineLoop *Loop);

  /// getBypassLoops - Get all the maximal bypass loops. These are the bypass
  /// loops whose parent is not a bypass loop.
  void getBypassLoops(LoopPtrSet&);

  /// getMultiUseBlocks - Add basic blocks to Blocks that may benefit from
  /// having CurLI split to a new live interval. Return true if Blocks can be
  /// passed to SplitEditor::splitSingleBlocks.
  bool getMultiUseBlocks(BlockPtrSet &Blocks);

  /// getBlockForInsideSplit - If CurLI is contained inside a single basic
  /// block, and it would pay to subdivide the interval inside that block,
  /// return it. Otherwise return NULL. The returned block can be passed to
  /// SplitEditor::splitInsideBlock.
  const MachineBasicBlock *getBlockForInsideSplit();
};


/// LiveIntervalMap - Map values from a large LiveInterval into a small
/// interval that is a subset. Insert phi-def values as needed. This class is
/// used by SplitEditor to create new smaller LiveIntervals.
///
/// ParentLI is the larger interval, LI is the subset interval. Every value
/// in LI corresponds to exactly one value in ParentLI, and the live range
/// of the value is contained within the live range of the ParentLI value.
/// Values in ParentLI may map to any number of OpenLI values, including 0.
class LiveIntervalMap {
  LiveIntervals &LIS;
  MachineDominatorTree &MDT;

  // The parent interval is never changed.
  const LiveInterval &ParentLI;

  // The child interval's values are fully contained inside ParentLI values.
  LiveInterval *LI;

  typedef DenseMap<const VNInfo*, VNInfo*> ValueMap;

  // Map ParentLI values to simple values in LI that are defined at the same
  // SlotIndex, or NULL for ParentLI values that have complex LI defs.
  // Note there is a difference between values mapping to NULL (complex), and
  // values not present (unknown/unmapped).
  ValueMap Values;

  typedef std::pair<VNInfo*, MachineDomTreeNode*> LiveOutPair;
  typedef DenseMap<MachineBasicBlock*,LiveOutPair> LiveOutMap;

  // LiveOutCache - Map each basic block where LI is live out to the live-out
  // value and its defining block. One of these conditions shall be true:
  //
  //  1. !LiveOutCache.count(MBB)
  //  2. LiveOutCache[MBB].second.getNode() == MBB
  //  3. forall P in preds(MBB): LiveOutCache[P] == LiveOutCache[MBB]
  //
  // This is only a cache, the values can be computed as:
  //
  //  VNI = LI->getVNInfoAt(LIS.getMBBEndIdx(MBB))
  //  Node = mbt_[LIS.getMBBFromIndex(VNI->def)]
  //
  // The cache is also used as a visiteed set by mapValue().
  LiveOutMap LiveOutCache;

  // Dump the live-out cache to dbgs().
  void dumpCache();

public:
  LiveIntervalMap(LiveIntervals &lis,
                  MachineDominatorTree &mdt,
                  const LiveInterval &parentli)
    : LIS(lis), MDT(mdt), ParentLI(parentli), LI(0) {}

  /// reset - clear all data structures and start a new live interval.
  void reset(LiveInterval *);

  /// getLI - return the current live interval.
  LiveInterval *getLI() const { return LI; }

  /// defValue - define a value in LI from the ParentLI value VNI and Idx.
  /// Idx does not have to be ParentVNI->def, but it must be contained within
  /// ParentVNI's live range in ParentLI.
  /// Return the new LI value.
  VNInfo *defValue(const VNInfo *ParentVNI, SlotIndex Idx);

  /// mapValue - map ParentVNI to the corresponding LI value at Idx. It is
  /// assumed that ParentVNI is live at Idx.
  /// If ParentVNI has not been defined by defValue, it is assumed that
  /// ParentVNI->def dominates Idx.
  /// If ParentVNI has been defined by defValue one or more times, a value that
  /// dominates Idx will be returned. This may require creating extra phi-def
  /// values and adding live ranges to LI.
  /// If simple is not NULL, *simple will indicate if ParentVNI is a simply
  /// mapped value.
  VNInfo *mapValue(const VNInfo *ParentVNI, SlotIndex Idx, bool *simple = 0);

  // extendTo - Find the last LI value defined in MBB at or before Idx. The
  // parentli is assumed to be live at Idx. Extend the live range to include
  // Idx. Return the found VNInfo, or NULL.
  VNInfo *extendTo(const MachineBasicBlock *MBB, SlotIndex Idx);

  /// isMapped - Return true is ParentVNI is a known mapped value. It may be a
  /// simple 1-1 mapping or a complex mapping to later defs.
  bool isMapped(const VNInfo *ParentVNI) const {
    return Values.count(ParentVNI);
  }

  /// isComplexMapped - Return true if ParentVNI has received new definitions
  /// with defValue.
  bool isComplexMapped(const VNInfo *ParentVNI) const;

  // addSimpleRange - Add a simple range from ParentLI to LI.
  // ParentVNI must be live in the [Start;End) interval.
  void addSimpleRange(SlotIndex Start, SlotIndex End, const VNInfo *ParentVNI);

  /// addRange - Add live ranges to LI where [Start;End) intersects ParentLI.
  /// All needed values whose def is not inside [Start;End) must be defined
  /// beforehand so mapValue will work.
  void addRange(SlotIndex Start, SlotIndex End);
};


/// SplitEditor - Edit machine code and LiveIntervals for live range
/// splitting.
///
/// - Create a SplitEditor from a SplitAnalysis.
/// - Start a new live interval with openIntv.
/// - Mark the places where the new interval is entered using enterIntv*
/// - Mark the ranges where the new interval is used with useIntv* 
/// - Mark the places where the interval is exited with exitIntv*.
/// - Finish the current interval with closeIntv and repeat from 2.
/// - Rewrite instructions with finish().
///
class SplitEditor {
  SplitAnalysis &sa_;
  LiveIntervals &LIS;
  VirtRegMap &VRM;
  MachineRegisterInfo &MRI;
  const TargetInstrInfo &TII;
  const TargetRegisterInfo &TRI;

  /// Edit - The current parent register and new intervals created.
  LiveRangeEdit &Edit;

  /// DupLI - Created as a copy of CurLI, ranges are carved out as new
  /// intervals get added through openIntv / closeIntv. This is used to avoid
  /// editing CurLI.
  LiveIntervalMap DupLI;

  /// Currently open LiveInterval.
  LiveIntervalMap OpenLI;

  /// defFromParent - Define Reg from ParentVNI at UseIdx using either
  /// rematerialization or a COPY from parent. Return the new value.
  VNInfo *defFromParent(LiveIntervalMap &Reg,
                        VNInfo *ParentVNI,
                        SlotIndex UseIdx,
                        MachineBasicBlock &MBB,
                        MachineBasicBlock::iterator I);

  /// intervalsLiveAt - Return true if any member of intervals_ is live at Idx.
  bool intervalsLiveAt(SlotIndex Idx) const;

  /// Values in CurLI whose live range has been truncated when entering an open
  /// li.
  SmallPtrSet<const VNInfo*, 8> truncatedValues;

  /// addTruncSimpleRange - Add the given simple range to DupLI after
  /// truncating any overlap with intervals_.
  void addTruncSimpleRange(SlotIndex Start, SlotIndex End, VNInfo *VNI);

  /// criticalPreds_ - Set of basic blocks where both dupli and OpenLI should be
  /// live out because of a critical edge.
  SplitAnalysis::BlockPtrSet criticalPreds_;

  /// computeRemainder - Compute the dupli liveness as the complement of all the
  /// new intervals.
  void computeRemainder();

  /// rewrite - Rewrite all uses of reg to use the new registers.
  void rewrite(unsigned reg);

public:
  /// Create a new SplitEditor for editing the LiveInterval analyzed by SA.
  /// Newly created intervals will be appended to newIntervals.
  SplitEditor(SplitAnalysis &SA, LiveIntervals&, VirtRegMap&,
              MachineDominatorTree&, LiveRangeEdit&);

  /// getAnalysis - Get the corresponding analysis.
  SplitAnalysis &getAnalysis() { return sa_; }

  /// Create a new virtual register and live interval.
  void openIntv();

  /// enterIntvBefore - Enter OpenLI before the instruction at Idx. If CurLI is
  /// not live before Idx, a COPY is not inserted.
  void enterIntvBefore(SlotIndex Idx);

  /// enterIntvAtEnd - Enter OpenLI at the end of MBB.
  void enterIntvAtEnd(MachineBasicBlock &MBB);

  /// useIntv - indicate that all instructions in MBB should use OpenLI.
  void useIntv(const MachineBasicBlock &MBB);

  /// useIntv - indicate that all instructions in range should use OpenLI.
  void useIntv(SlotIndex Start, SlotIndex End);

  /// leaveIntvAfter - Leave OpenLI after the instruction at Idx.
  void leaveIntvAfter(SlotIndex Idx);

  /// leaveIntvAtTop - Leave the interval at the top of MBB.
  /// Currently, only one value can leave the interval.
  void leaveIntvAtTop(MachineBasicBlock &MBB);

  /// closeIntv - Indicate that we are done editing the currently open
  /// LiveInterval, and ranges can be trimmed.
  void closeIntv();

  /// finish - after all the new live ranges have been created, compute the
  /// remaining live range, and rewrite instructions to use the new registers.
  void finish();

  // ===--- High level methods ---===

  /// splitAroundLoop - Split CurLI into a separate live interval inside
  /// the loop.
  void splitAroundLoop(const MachineLoop*);

  /// splitSingleBlocks - Split CurLI into a separate live interval inside each
  /// basic block in Blocks.
  void splitSingleBlocks(const SplitAnalysis::BlockPtrSet &Blocks);

  /// splitInsideBlock - Split CurLI into multiple intervals inside MBB.
  void splitInsideBlock(const MachineBasicBlock *);
};

}
