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
  const MachineFunction &mf_;
  const LiveIntervals &lis_;
  const MachineLoopInfo &loops_;
  const TargetInstrInfo &tii_;

  // Current live interval.
  const LiveInterval *curli_;

  // Instructions using the the current register.
  typedef SmallPtrSet<const MachineInstr*, 16> InstrPtrSet;
  InstrPtrSet usingInstrs_;

  // The number of instructions using curli in each basic block.
  typedef DenseMap<const MachineBasicBlock*, unsigned> BlockCountMap;
  BlockCountMap usingBlocks_;

  // Loops where the curent interval is used.
  typedef SmallPtrSet<const MachineLoop*, 16> LoopPtrSet;
  LoopPtrSet usingLoops_;

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

  const LiveInterval *getCurLI() { return curli_; }

  /// clear - clear all data structures so SplitAnalysis is ready to analyze a
  /// new interval.
  void clear();

  typedef SmallPtrSet<const MachineBasicBlock*, 16> BlockPtrSet;

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
};

/// SplitEditor - Edit machine code and LiveIntervals for live range
/// splitting.
///
/// 1. Create a SplitEditor from a SplitAnalysis. This will create a new
///    LiveInterval, dupli, that is identical to SA.curli.
/// 2. Start a new live interval with openLI.
/// 3. Insert copies to the new interval with copyTo* and mark the ranges where
///    it should be used with use*.
/// 4. Insert back-copies with copyFromLI.
/// 5. Finish the current LI with closeLI and repeat from 2.
/// 6. Rewrite instructions with rewrite().
///
class SplitEditor {
  SplitAnalysis &sa_;
  LiveIntervals &lis_;
  VirtRegMap &vrm_;
  MachineRegisterInfo &mri_;
  const TargetInstrInfo &tii_;

  /// dupli_ - Created as a copy of sa_.curli_, ranges are carved out as new
  /// intervals get added through openLI / closeLI.
  LiveInterval *dupli_;

  /// Currently open LiveInterval.
  LiveInterval *openli_;

  /// createInterval - Create a new virtual register and LiveInterval with same
  /// register class and spill slot as curli.
  LiveInterval *createInterval();

	/// valueMap_ - Map values in dupli to values in openli. These are direct 1-1
	/// mappings, and do not include values created by inserted copies.
	DenseMap<VNInfo*,VNInfo*> valueMap_;

	/// mapValue - Return the openli value that corresponds to the given dupli
	/// value.
	VNInfo *mapValue(VNInfo *dupliVNI);	

public:
  /// Create a new SplitEditor for editing the LiveInterval analyzed by SA.
  SplitEditor(SplitAnalysis&, LiveIntervals&, VirtRegMap&);

	/// getAnalysis - Get the corresponding analysis.
	SplitAnalysis &getAnalysis() { return sa_; }

  /// Create a new virtual register and live interval to be used by following
  /// use* and copy* calls.
  void openLI();

  /// copyToPHI - Insert a copy to openli at the end of A, and catch it with a
  /// PHI def at the beginning of the successor B. This call is ignored if dupli
  /// is not live out of A.
  void copyToPHI(MachineBasicBlock &A, MachineBasicBlock &B);

  /// useLI - indicate that all instructions in MBB should use openli.
  void useLI(const MachineBasicBlock &MBB);

  /// useLI - indicate that all instructions in range should use openli.
  void useLI(SlotIndex Start, SlotIndex End);

  /// copyFromLI - Insert a copy back to dupli from openli at position I.
	/// This also marks the remainder of MBB as not used by openli.
  SlotIndex copyFromLI(MachineBasicBlock &MBB, MachineBasicBlock::iterator I);

  /// closeLI - Indicate that we are done editing the currently open
  /// LiveInterval, and ranges can be trimmed.
  void closeLI();

  /// rewrite - after all the new live ranges have been created, rewrite
  /// instructions using curli to use the new intervals.
  void rewrite();

  // ===--- High level methods ---===

  /// splitAroundLoop - Split curli into a separate live interval inside
  /// the loop.
  void splitAroundLoop(const MachineLoop*);

};


}
