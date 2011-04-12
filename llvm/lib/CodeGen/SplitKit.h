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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IndexedMap.h"
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

  // Sorted slot indexes of using instructions.
  SmallVector<SlotIndex, 8> UseSlots;

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
    bool LiveThrough;     ///< Live in whole block (Templ 5. or 6. above).
    bool LiveIn;          ///< Current reg is live in.
    bool LiveOut;         ///< Current reg is live out.
  };

private:
  // Current live interval.
  const LiveInterval *CurLI;

  /// LastSplitPoint - Last legal split point in each basic block in the current
  /// function. The first entry is the first terminator, the second entry is the
  /// last valid split point for a variable that is live in to a landing pad
  /// successor.
  SmallVector<std::pair<SlotIndex, SlotIndex>, 8> LastSplitPoint;

  /// UseBlocks - Blocks where CurLI has uses.
  SmallVector<BlockInfo, 8> UseBlocks;

  /// ThroughBlocks - Block numbers where CurLI is live through without uses.
  BitVector ThroughBlocks;

  /// NumThroughBlocks - Number of live-through blocks.
  unsigned NumThroughBlocks;

  SlotIndex computeLastSplitPoint(unsigned Num);

  // Sumarize statistics by counting instructions using CurLI.
  void analyzeUses();

  /// calcLiveBlockInfo - Compute per-block information about CurLI.
  bool calcLiveBlockInfo();

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

  /// getLastSplitPoint - Return that base index of the last valid split point
  /// in the basic block numbered Num.
  SlotIndex getLastSplitPoint(unsigned Num) {
    // Inline the common simple case.
    if (LastSplitPoint[Num].first.isValid() &&
        !LastSplitPoint[Num].second.isValid())
      return LastSplitPoint[Num].first;
    return computeLastSplitPoint(Num);
  }

  /// isOriginalEndpoint - Return true if the original live range was killed or
  /// (re-)defined at Idx. Idx should be the 'def' slot for a normal kill/def,
  /// and 'use' for an early-clobber def.
  /// This can be used to recognize code inserted by earlier live range
  /// splitting.
  bool isOriginalEndpoint(SlotIndex Idx) const;

  /// getUseBlocks - Return an array of BlockInfo objects for the basic blocks
  /// where CurLI has uses.
  ArrayRef<BlockInfo> getUseBlocks() { return UseBlocks; }

  /// getNumThroughBlocks - Return the number of through blocks.
  unsigned getNumThroughBlocks() const { return NumThroughBlocks; }

  /// isThroughBlock - Return true if CurLI is live through MBB without uses.
  bool isThroughBlock(unsigned MBB) const { return ThroughBlocks.test(MBB); }

  /// getThroughBlocks - Return the set of through blocks.
  const BitVector &getThroughBlocks() const { return ThroughBlocks; }

  typedef SmallPtrSet<const MachineBasicBlock*, 16> BlockPtrSet;

  /// getMultiUseBlocks - Add basic blocks to Blocks that may benefit from
  /// having CurLI split to a new live interval. Return true if Blocks can be
  /// passed to SplitEditor::splitSingleBlocks.
  bool getMultiUseBlocks(BlockPtrSet &Blocks);
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
  LiveRangeEdit *Edit;

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

  typedef DenseMap<std::pair<unsigned, unsigned>, VNInfo*> ValueMap;

  /// Values - keep track of the mapping from parent values to values in the new
  /// intervals. Given a pair (RegIdx, ParentVNI->id), Values contains:
  ///
  /// 1. No entry - the value is not mapped to Edit.get(RegIdx).
  /// 2. Null - the value is mapped to multiple values in Edit.get(RegIdx).
  ///    Each value is represented by a minimal live range at its def.
  /// 3. A non-null VNInfo - the value is mapped to a single new value.
  ///    The new value has no live ranges anywhere.
  ValueMap Values;

  typedef std::pair<VNInfo*, MachineDomTreeNode*> LiveOutPair;
  typedef IndexedMap<LiveOutPair, MBB2NumberFunctor> LiveOutMap;

  // LiveOutCache - Map each basic block where a new register is live out to the
  // live-out value and its defining block.
  // One of these conditions shall be true:
  //
  //  1. !LiveOutCache.count(MBB)
  //  2. LiveOutCache[MBB].second.getNode() == MBB
  //  3. forall P in preds(MBB): LiveOutCache[P] == LiveOutCache[MBB]
  //
  // This is only a cache, the values can be computed as:
  //
  //  VNI = Edit.get(RegIdx)->getVNInfoAt(LIS.getMBBEndIdx(MBB))
  //  Node = mbt_[LIS.getMBBFromIndex(VNI->def)]
  //
  // The cache is also used as a visited set by extendRange(). It can be shared
  // by all the new registers because at most one is live out of each block.
  LiveOutMap LiveOutCache;

  // LiveOutSeen - Indexed by MBB->getNumber(), a bit is set for each valid
  // entry in LiveOutCache.
  BitVector LiveOutSeen;

  /// defValue - define a value in RegIdx from ParentVNI at Idx.
  /// Idx does not have to be ParentVNI->def, but it must be contained within
  /// ParentVNI's live range in ParentLI. The new value is added to the value
  /// map.
  /// Return the new LI value.
  VNInfo *defValue(unsigned RegIdx, const VNInfo *ParentVNI, SlotIndex Idx);

  /// markComplexMapped - Mark ParentVNI as complex mapped in RegIdx regardless
  /// of the number of defs.
  void markComplexMapped(unsigned RegIdx, const VNInfo *ParentVNI);

  /// defFromParent - Define Reg from ParentVNI at UseIdx using either
  /// rematerialization or a COPY from parent. Return the new value.
  VNInfo *defFromParent(unsigned RegIdx,
                        VNInfo *ParentVNI,
                        SlotIndex UseIdx,
                        MachineBasicBlock &MBB,
                        MachineBasicBlock::iterator I);

  /// extendRange - Extend the live range of Edit.get(RegIdx) so it reaches Idx.
  /// Insert PHIDefs as needed to preserve SSA form.
  void extendRange(unsigned RegIdx, SlotIndex Idx);

  /// updateSSA - Insert PHIDefs as necessary and update LiveOutCache such that
  /// Edit.get(RegIdx) is live-in to all the blocks in LiveIn.
  /// Return the value that is eventually live-in to IdxMBB.
  VNInfo *updateSSA(unsigned RegIdx,
                    SmallVectorImpl<MachineDomTreeNode*> &LiveIn,
                    SlotIndex Idx,
                    const MachineBasicBlock *IdxMBB);

  /// transferSimpleValues - Transfer simply defined values to the new ranges.
  /// Return true if any complex ranges were skipped.
  bool transferSimpleValues();

  /// extendPHIKillRanges - Extend the ranges of all values killed by original
  /// parent PHIDefs.
  void extendPHIKillRanges();

  /// rewriteAssigned - Rewrite all uses of Edit.getReg() to assigned registers.
  void rewriteAssigned(bool ExtendRanges);

  /// deleteRematVictims - Delete defs that are dead after rematerializing.
  void deleteRematVictims();

public:
  /// Create a new SplitEditor for editing the LiveInterval analyzed by SA.
  /// Newly created intervals will be appended to newIntervals.
  SplitEditor(SplitAnalysis &SA, LiveIntervals&, VirtRegMap&,
              MachineDominatorTree&);

  /// reset - Prepare for a new split.
  void reset(LiveRangeEdit&);

  /// Create a new virtual register and live interval.
  /// Return the interval index, starting from 1. Interval index 0 is the
  /// implicit complement interval.
  unsigned openIntv();

  /// currentIntv - Return the current interval index.
  unsigned currentIntv() const { return OpenIdx; }

  /// selectIntv - Select a previously opened interval index.
  void selectIntv(unsigned Idx);

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

  /// splitSingleBlock - Split CurLI into a separate live interval around the
  /// uses in a single block. This is intended to be used as part of a larger
  /// split, and doesn't call finish().
  void splitSingleBlock(const SplitAnalysis::BlockInfo &BI);

  /// splitSingleBlocks - Split CurLI into a separate live interval inside each
  /// basic block in Blocks.
  void splitSingleBlocks(const SplitAnalysis::BlockPtrSet &Blocks);
};

}
