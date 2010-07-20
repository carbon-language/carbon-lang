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

namespace llvm {

class LiveInterval;
class LiveIntervals;
class MachineBasicBlock;
class MachineInstr;
class MachineFunction;
class MachineFunctionPass;
class MachineLoop;
class MachineLoopInfo;
class TargetInstrInfo;

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

/// splitAroundLoop - Try to split curli into a separate live interval inside
/// the loop. Retun true on success.
bool splitAroundLoop(SplitAnalysis&, const MachineLoop*);

}
