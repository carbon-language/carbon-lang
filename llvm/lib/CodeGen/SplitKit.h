//===---------- SplitKit.cpp - Toolkit for splitting live ranges ----------===//
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

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/SlotIndexes.h"

namespace llvm {

class LiveInterval;
class LiveIntervals;
class MachineInstr;
class MachineLoop;
class MachineLoopInfo;
class MachineRegisterInfo;
class TargetInstrInfo;
class VirtRegMap;
class VNInfo;

/// SplitAnalysis - Analyze a LiveInterval, looking for live range splitting
/// opportunities.
class SplitAnalysis {
public:
  const MachineFunction &mf_;
  const LiveIntervals &lis_;
  const MachineLoopInfo &loops_;
  const TargetInstrInfo &tii_;

  // Instructions using the the current register.
  typedef SmallPtrSet<const MachineInstr*, 16> InstrPtrSet;
  InstrPtrSet usingInstrs_;

  // The number of instructions using curli in each basic block.
  typedef DenseMap<const MachineBasicBlock*, unsigned> BlockCountMap;
  BlockCountMap usingBlocks_;

  // The number of basic block using curli in each loop.
  typedef DenseMap<const MachineLoop*, unsigned> LoopCountMap;
  LoopCountMap usingLoops_;

private:
  // Current live interval.
  const LiveInterval *curli_;

  // Sumarize statistics by counting instructions using curli_.
  void analyzeUses();

  /// canAnalyzeBranch - Return true if MBB ends in a branch that can be
  /// analyzed.
  bool canAnalyzeBranch(const MachineBasicBlock *MBB);

public:
  SplitAnalysis(const MachineFunction &mf, const LiveIntervals &lis,
                const MachineLoopInfo &mli);

  /// analyze - set curli to the specified interval, and analyze how it may be
  /// split.
  void analyze(const LiveInterval *li);

  /// removeUse - Update statistics by noting that mi no longer uses curli.
  void removeUse(const MachineInstr *mi);

  const LiveInterval *getCurLI() { return curli_; }

  /// clear - clear all data structures so SplitAnalysis is ready to analyze a
  /// new interval.
  void clear();

  typedef SmallPtrSet<const MachineBasicBlock*, 16> BlockPtrSet;
  typedef SmallPtrSet<const MachineLoop*, 16> LoopPtrSet;

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

  /// analyzeLoopPeripheralUse - Return an enum describing how curli_ is used in
  /// and around the Loop.
  LoopPeripheralUse analyzeLoopPeripheralUse(const LoopBlocks&);

  /// getCriticalExits - It may be necessary to partially break critical edges
  /// leaving the loop if an exit block has phi uses of curli. Collect the exit
  /// blocks that need special treatment into CriticalExits.
  void getCriticalExits(const LoopBlocks &Blocks, BlockPtrSet &CriticalExits);

  /// canSplitCriticalExits - Return true if it is possible to insert new exit
  /// blocks before the blocks in CriticalExits.
  bool canSplitCriticalExits(const LoopBlocks &Blocks,
                             BlockPtrSet &CriticalExits);

  /// getBestSplitLoop - Return the loop where curli may best be split to a
  /// separate register, or NULL.
  const MachineLoop *getBestSplitLoop();

  /// getMultiUseBlocks - Add basic blocks to Blocks that may benefit from
  /// having curli split to a new live interval. Return true if Blocks can be
  /// passed to SplitEditor::splitSingleBlocks.
  bool getMultiUseBlocks(BlockPtrSet &Blocks);

  /// getBlockForInsideSplit - If curli is contained inside a single basic block,
  /// and it wou pay to subdivide the interval inside that block, return it.
  /// Otherwise return NULL. The returned block can be passed to
  /// SplitEditor::splitInsideBlock.
  const MachineBasicBlock *getBlockForInsideSplit();
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
/// - Rewrite instructions with rewrite().
///
class SplitEditor {
  SplitAnalysis &sa_;
  LiveIntervals &lis_;
  VirtRegMap &vrm_;
  MachineRegisterInfo &mri_;
  const TargetInstrInfo &tii_;

  /// curli_ - The immutable interval we are currently splitting.
  const LiveInterval *const curli_;

  /// dupli_ - Created as a copy of curli_, ranges are carved out as new
  /// intervals get added through openIntv / closeIntv. This is used to avoid
  /// editing curli_.
  LiveInterval *dupli_;

  /// Currently open LiveInterval.
  LiveInterval *openli_;

  /// createInterval - Create a new virtual register and LiveInterval with same
  /// register class and spill slot as curli.
  LiveInterval *createInterval();

  /// getDupLI - Ensure dupli is created and return it.
  LiveInterval *getDupLI();

  /// valueMap_ - Map values in cupli to values in openli. These are direct 1-1
  /// mappings, and do not include values created by inserted copies.
  DenseMap<const VNInfo*, VNInfo*> valueMap_;

  /// mapValue - Return the openIntv value that corresponds to the given curli
  /// value.
  VNInfo *mapValue(const VNInfo *curliVNI);

  /// A dupli value is live through openIntv.
  bool liveThrough_;

  /// All the new intervals created for this split are added to intervals_.
  SmallVectorImpl<LiveInterval*> &intervals_;

  /// The index into intervals_ of the first interval we added. There may be
  /// others from before we got it.
  unsigned firstInterval;

  /// Insert a COPY instruction curli -> li. Allocate a new value from li
  /// defined by the COPY
  VNInfo *insertCopy(LiveInterval &LI,
                     MachineBasicBlock &MBB,
                     MachineBasicBlock::iterator I);

public:
  /// Create a new SplitEditor for editing the LiveInterval analyzed by SA.
  /// Newly created intervals will be appended to newIntervals.
  SplitEditor(SplitAnalysis &SA, LiveIntervals&, VirtRegMap&,
              SmallVectorImpl<LiveInterval*> &newIntervals);

  /// getAnalysis - Get the corresponding analysis.
  SplitAnalysis &getAnalysis() { return sa_; }

  /// Create a new virtual register and live interval.
  void openIntv();

  /// enterIntvBefore - Enter openli before the instruction at Idx. If curli is
  /// not live before Idx, a COPY is not inserted.
  void enterIntvBefore(SlotIndex Idx);

  /// enterIntvAtEnd - Enter openli at the end of MBB.
  /// PhiMBB is a successor inside openli where a PHI value is created.
  /// Currently, all entries must share the same PhiMBB.
  void enterIntvAtEnd(MachineBasicBlock &MBB, MachineBasicBlock &PhiMBB);

  /// useIntv - indicate that all instructions in MBB should use openli.
  void useIntv(const MachineBasicBlock &MBB);

  /// useIntv - indicate that all instructions in range should use openli.
  void useIntv(SlotIndex Start, SlotIndex End);

  /// leaveIntvAfter - Leave openli after the instruction at Idx.
  void leaveIntvAfter(SlotIndex Idx);

  /// leaveIntvAtTop - Leave the interval at the top of MBB.
  /// Currently, only one value can leave the interval.
  void leaveIntvAtTop(MachineBasicBlock &MBB);

  /// closeIntv - Indicate that we are done editing the currently open
  /// LiveInterval, and ranges can be trimmed.
  void closeIntv();

  /// rewrite - after all the new live ranges have been created, rewrite
  /// instructions using curli to use the new intervals.
  void rewrite();

  // ===--- High level methods ---===

  /// splitAroundLoop - Split curli into a separate live interval inside
  /// the loop. Return true if curli has been completely replaced, false if
  /// curli is still intact, and needs to be spilled or split further.
  bool splitAroundLoop(const MachineLoop*);

  /// splitSingleBlocks - Split curli into a separate live interval inside each
  /// basic block in Blocks. Return true if curli has been completely replaced,
  /// false if curli is still intact, and needs to be spilled or split further.
  bool splitSingleBlocks(const SplitAnalysis::BlockPtrSet &Blocks);

  /// splitInsideBlock - Split curli into multiple intervals inside MBB. Return
  /// true if curli has been completely replaced, false if curli is still
  /// intact, and needs to be spilled or split further.
  bool splitInsideBlock(const MachineBasicBlock *);
};

}
