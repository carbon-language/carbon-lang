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

class SplitAnalysis {
  const MachineFunction &mf_;
  const LiveIntervals &lis_;
  const MachineLoopInfo &loops_;

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

public:
  SplitAnalysis(const MachineFunction *mf, const LiveIntervals *lis,
                const MachineLoopInfo *mli);

  /// analyze - set curli to the specified interval, and analyze how it may be
  /// split.
  void analyze(const LiveInterval *li);

  /// clear - clear all data structures so SplitAnalysis is ready to analyze a
  /// new interval.
  void clear();

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
  LoopPeripheralUse analyzeLoopPeripheralUse(const MachineLoop*);

  /// getBestSplitLoop - Return the loop where curli may best be split to a
  /// separate register, or NULL.
  const MachineLoop *getBestSplitLoop();
};

/// splitAroundLoop - Try to split curli into a separate live interval inside
/// the loop. Retun true on success.
bool splitAroundLoop(SplitAnalysis&, const MachineLoop*);

}
