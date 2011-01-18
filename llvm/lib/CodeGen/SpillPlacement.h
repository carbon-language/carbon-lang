//===-- SpillPlacement.h - Optimal Spill Code Placement --------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This analysis computes the optimal spill code placement between basic blocks.
//
// The runOnMachineFunction() method only precomputes some profiling information
// about the CFG. The real work is done by placeSpills() which is called by the
// register allocator.
//
// Given a variable that is live across multiple basic blocks, and given
// constraints on the basic blocks where the variable is live, determine which
// edge bundles should have the variable in a register and which edge bundles
// should have the variable in a stack slot.
//
// The returned bit vector can be used to place optimal spill code at basic
// block entries and exits. Spill code placement inside a basic block is not
// considered.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SPILLPLACEMENT_H
#define LLVM_CODEGEN_SPILLPLACEMENT_H

#include "llvm/CodeGen/MachineFunctionPass.h"

namespace llvm {

class BitVector;
class EdgeBundles;
class MachineBasicBlock;
class MachineLoopInfo;
template <typename> class SmallVectorImpl;

class SpillPlacement  : public MachineFunctionPass {
  struct Node;
  const MachineFunction *MF;
  const EdgeBundles *bundles;
  const MachineLoopInfo *loops;
  Node *nodes;

  // Nodes that are active in the current computation. Owned by the placeSpills
  // caller.
  BitVector *ActiveNodes;

public:
  static char ID; // Pass identification, replacement for typeid.

  SpillPlacement() : MachineFunctionPass(ID), nodes(0) {}
  ~SpillPlacement() { releaseMemory(); }

  /// BorderConstraint - A basic block has separate constraints for entry and
  /// exit.
  enum BorderConstraint {
    DontCare,  ///< Block doesn't care / variable not live.
    PrefReg,   ///< Block entry/exit prefers a register.
    PrefSpill, ///< Block entry/exit prefers a stack slot.
    MustSpill  ///< A register is impossible, variable must be spilled.
  };

  /// BlockConstraint - Entry and exit constraints for a basic block.
  struct BlockConstraint {
    unsigned Number;            ///< Basic block number (from MBB::getNumber()).
    BorderConstraint Entry : 8; ///< Constraint on block entry.
    BorderConstraint Exit : 8;  ///< Constraint on block exit.
  };

  /// placeSpills - Compute the optimal spill code placement given the
  /// constraints. No MustSpill constraints will be violated, and the smallest
  /// possible number of PrefX constraints will be violated, weighted by
  /// expected execution frequencies.
  /// @param LiveBlocks Constraints for blocks that have the variable live in or
  ///                   live out. DontCare/DontCare means the variable is live
  ///                   through the block. DontCare/X means the variable is live
  ///                   out, but not live in.
  /// @param RegBundles Bit vector to receive the edge bundles where the
  ///                   variable should be kept in a register. Each bit
  ///                   corresponds to an edge bundle, a set bit means the
  ///                   variable should be kept in a register through the
  ///                   bundle. A clear bit means the variable should be
  ///                   spilled.
  /// @return True if a perfect solution was found, allowing the variable to be
  ///         in a register through all relevant bundles.
  bool placeSpills(const SmallVectorImpl<BlockConstraint> &LiveBlocks,
                   BitVector &RegBundles);

  /// getBlockFrequency - Return the estimated block execution frequency per
  /// function invocation.
  float getBlockFrequency(const MachineBasicBlock*);

private:
  virtual bool runOnMachineFunction(MachineFunction&);
  virtual void getAnalysisUsage(AnalysisUsage&) const;
  virtual void releaseMemory();

  void activate(unsigned);
  void prepareNodes(const SmallVectorImpl<BlockConstraint>&);
  void iterate(const SmallVectorImpl<unsigned>&);
};

} // end namespace llvm

#endif
