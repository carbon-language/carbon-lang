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
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/CodeGen/SlotIndexes.h"

namespace llvm {

class ConnectedVNInfoEqClasses;
class LiveInterval;
class LiveIntervals;
class LiveRangeEdit;
class MachineInstr;
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
  const VirtRegMap &VRM;
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

  /// Additional information about basic blocks where the current variable is
  /// live. Such a block will look like one of these templates:
  ///
  ///  1. |   o---x   | Internal to block. Variable is only live in this block.
  ///  2. |---x       | Live-in, kill.
  ///  3. |       o---| Def, live-out.
  ///  4. |---x   o---| Live-in, kill, def, live-out.
  ///  5. |---o---o---| Live-through with uses or defs.
  ///  6. |-----------| Live-through without uses. Transparent.
  ///
  struct BlockInfo {
    MachineBasicBlock *MBB;
    SlotIndex FirstUse;   ///< First instr using current reg.
    SlotIndex LastUse;    ///< Last instr using current reg.
    SlotIndex Kill;       ///< Interval end point inside block.
    SlotIndex Def;        ///< Interval start point inside block.
    /// Last possible point for splitting live ranges.
    SlotIndex LastSplitPoint;
    bool Uses;            ///< Current reg has uses or defs in block.
    bool LiveThrough;     ///< Live in whole block (Templ 5. or 6. above).
    bool LiveIn;          ///< Current reg is live in.
    bool LiveOut;         ///< Current reg is live out.

    // Per-interference pattern scratch data.
    bool OverlapEntry;    ///< Interference overlaps entering interval.
    bool OverlapExit;     ///< Interference overlaps exiting interval.
  };

  /// Basic blocks where var is live. This array is parallel to
  /// SpillConstraints.
  SmallVector<BlockInfo, 8> LiveBlocks;

private:
  // Current live interval.
  const LiveInterval *CurLI;

  // Sumarize statistics by counting instructions using CurLI.
  void analyzeUses();

  /// calcLiveBlockInfo - Compute per-block information about CurLI.
  void calcLiveBlockInfo();

  /// canAnalyzeBranch - Return true if MBB ends in a branch that can be
  /// analyzed.
  bool canAnalyzeBranch(const MachineBasicBlock *MBB);

public:
  SplitAnalysis(const VirtRegMap &vrm, const LiveIntervals &lis,
                const MachineLoopInfo &mli);

  /// analyze - set CurLI to the specified interval, and analyze how it may be
  /// split.
  void analyze(const LiveInterval *li);

  /// clear - clear all data structures so SplitAnalysis is ready to analyze a
  /// new interval.
  void clear();

  /// getParent - Return the last analyzed interval.
  const LiveInterval &getParent() const { return *CurLI; }

  /// hasUses - Return true if MBB has any uses of CurLI.
  bool hasUses(const MachineBasicBlock *MBB) const {
    return UsingBlocks.lookup(MBB);
  }

  typedef SmallPtrSet<const MachineBasicBlock*, 16> BlockPtrSet;

  // Print a set of blocks with use counts.
  void print(const BlockPtrSet&, raw_ostream&) const;

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

  /// markComplexMapped - Mark ParentVNI as complex mapped regardless of the
  /// number of definitions.
  void markComplexMapped(const VNInfo *ParentVNI) { Values[ParentVNI] = 0; }

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
  SplitAnalysis &SA;
  LiveIntervals &LIS;
  VirtRegMap &VRM;
  MachineRegisterInfo &MRI;
  MachineDominatorTree &MDT;
  const TargetInstrInfo &TII;
  const TargetRegisterInfo &TRI;

  /// Edit - The current parent register and new intervals created.
  LiveRangeEdit &Edit;

  /// Index into Edit of the currently open interval.
  /// The index 0 is used for the complement, so the first interval started by
  /// openIntv will be 1.
  unsigned OpenIdx;

  typedef IntervalMap<SlotIndex, unsigned> RegAssignMap;

  /// Allocator for the interval map. This will eventually be shared with
  /// SlotIndexes and LiveIntervals.
  RegAssignMap::Allocator Allocator;

  /// RegAssign - Map of the assigned register indexes.
  /// Edit.get(RegAssign.lookup(Idx)) is the register that should be live at
  /// Idx.
  RegAssignMap RegAssign;

  /// LIMappers - One LiveIntervalMap or each interval in Edit.
  SmallVector<LiveIntervalMap, 4> LIMappers;

  /// defFromParent - Define Reg from ParentVNI at UseIdx using either
  /// rematerialization or a COPY from parent. Return the new value.
  VNInfo *defFromParent(unsigned RegIdx,
                        VNInfo *ParentVNI,
                        SlotIndex UseIdx,
                        MachineBasicBlock &MBB,
                        MachineBasicBlock::iterator I);

  /// rewriteAssigned - Rewrite all uses of Edit.getReg() to assigned registers.
  void rewriteAssigned();

  /// rewriteComponents - Rewrite all uses of Intv[0] according to the eq
  /// classes in ConEQ.
  /// This must be done when Intvs[0] is styill live at all uses, before calling
  /// ConEq.Distribute().
  void rewriteComponents(const SmallVectorImpl<LiveInterval*> &Intvs,
                         const ConnectedVNInfoEqClasses &ConEq);

public:
  /// Create a new SplitEditor for editing the LiveInterval analyzed by SA.
  /// Newly created intervals will be appended to newIntervals.
  SplitEditor(SplitAnalysis &SA, LiveIntervals&, VirtRegMap&,
              MachineDominatorTree&, LiveRangeEdit&);

  /// getAnalysis - Get the corresponding analysis.
  SplitAnalysis &getAnalysis() { return SA; }

  /// Create a new virtual register and live interval.
  void openIntv();

  /// enterIntvBefore - Enter the open interval before the instruction at Idx.
  /// If the parent interval is not live before Idx, a COPY is not inserted.
  /// Return the beginning of the new live range.
  SlotIndex enterIntvBefore(SlotIndex Idx);

  /// enterIntvAtEnd - Enter the open interval at the end of MBB.
  /// Use the open interval from he inserted copy to the MBB end.
  /// Return the beginning of the new live range.
  SlotIndex enterIntvAtEnd(MachineBasicBlock &MBB);

  /// useIntv - indicate that all instructions in MBB should use OpenLI.
  void useIntv(const MachineBasicBlock &MBB);

  /// useIntv - indicate that all instructions in range should use OpenLI.
  void useIntv(SlotIndex Start, SlotIndex End);

  /// leaveIntvAfter - Leave the open interval after the instruction at Idx.
  /// Return the end of the live range.
  SlotIndex leaveIntvAfter(SlotIndex Idx);

  /// leaveIntvBefore - Leave the open interval before the instruction at Idx.
  /// Return the end of the live range.
  SlotIndex leaveIntvBefore(SlotIndex Idx);

  /// leaveIntvAtTop - Leave the interval at the top of MBB.
  /// Add liveness from the MBB top to the copy.
  /// Return the end of the live range.
  SlotIndex leaveIntvAtTop(MachineBasicBlock &MBB);

  /// overlapIntv - Indicate that all instructions in range should use the open
  /// interval, but also let the complement interval be live.
  ///
  /// This doubles the register pressure, but is sometimes required to deal with
  /// register uses after the last valid split point.
  ///
  /// The Start index should be a return value from a leaveIntv* call, and End
  /// should be in the same basic block. The parent interval must have the same
  /// value across the range.
  ///
  void overlapIntv(SlotIndex Start, SlotIndex End);

  /// closeIntv - Indicate that we are done editing the currently open
  /// LiveInterval, and ranges can be trimmed.
  void closeIntv();

  /// finish - after all the new live ranges have been created, compute the
  /// remaining live range, and rewrite instructions to use the new registers.
  void finish();

  /// dump - print the current interval maping to dbgs().
  void dump() const;

  // ===--- High level methods ---===

  /// splitSingleBlocks - Split CurLI into a separate live interval inside each
  /// basic block in Blocks.
  void splitSingleBlocks(const SplitAnalysis::BlockPtrSet &Blocks);

  /// splitInsideBlock - Split CurLI into multiple intervals inside MBB.
  void splitInsideBlock(const MachineBasicBlock *);
};

}
