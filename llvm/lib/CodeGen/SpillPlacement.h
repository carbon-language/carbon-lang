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
// about the CFG. The real work is done by prepare(), addConstraints(), and
// finish() which are called by the register allocator.
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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

namespace llvm {

class BitVector;
class EdgeBundles;
class MachineBasicBlock;
class MachineLoopInfo;

class SpillPlacement  : public MachineFunctionPass {
  struct Node;
  const MachineFunction *MF;
  const EdgeBundles *bundles;
  const MachineLoopInfo *loops;
  Node *nodes;

  // Nodes that are active in the current computation. Owned by the prepare()
  // caller.
  BitVector *ActiveNodes;

  // The number of active nodes with a positive bias.
  unsigned PositiveNodes;

  // Block frequencies are computed once. Indexed by block number.
  SmallVector<float, 4> BlockFrequency;

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

  /// prepare - Reset state and prepare for a new spill placement computation.
  /// @param RegBundles Bit vector to receive the edge bundles where the
  ///                   variable should be kept in a register. Each bit
  ///                   corresponds to an edge bundle, a set bit means the
  ///                   variable should be kept in a register through the
  ///                   bundle. A clear bit means the variable should be
  ///                   spilled. This vector is retained.
  void prepare(BitVector &RegBundles);

  /// addConstraints - Add constraints and biases. This method may be called
  /// more than once to accumulate constraints.
  /// @param LiveBlocks Constraints for blocks that have the variable live in or
  ///                   live out. DontCare/DontCare means the variable is live
  ///                   through the block. DontCare/X means the variable is live
  ///                   out, but not live in.
  void addConstraints(ArrayRef<BlockConstraint> LiveBlocks);

  /// getPositiveNodes - Return the total number of graph nodes with a positive
  /// bias after adding constraints.
  unsigned getPositiveNodes() const { return PositiveNodes; }

  /// finish - Compute the optimal spill code placement given the
  /// constraints. No MustSpill constraints will be violated, and the smallest
  /// possible number of PrefX constraints will be violated, weighted by
  /// expected execution frequencies.
  /// The selected bundles are returned in the bitvector passed to prepare().
  /// @return True if a perfect solution was found, allowing the variable to be
  ///         in a register through all relevant bundles.
  bool finish();

  /// getBlockFrequency - Return the estimated block execution frequency per
  /// function invocation.
  float getBlockFrequency(unsigned Number) const {
    return BlockFrequency[Number];
  }

private:
  virtual bool runOnMachineFunction(MachineFunction&);
  virtual void getAnalysisUsage(AnalysisUsage&) const;
  virtual void releaseMemory();

  void activate(unsigned);
  void iterate(const SmallVectorImpl<unsigned>&);
};

} // end namespace llvm

#endif
