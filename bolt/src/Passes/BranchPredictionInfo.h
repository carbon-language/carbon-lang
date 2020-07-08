//===------ Passes/BranchPredictionInfo.h ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is an auxiliary class to the feature miner, static branch probability
// and frequency passes. This class is responsible for finding loop info (loop
// back edges, loop exit edges and loop headers) of a function. It also finds
// basic block info (if a block contains store and call instructions) and if a
// basic block contains a call to the exit.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_PASSES_BRANCHPREDICTIONINFO_H_
#define LLVM_TOOLS_LLVM_BOLT_PASSES_BRANCHPREDICTIONINFO_H_

#include "BinaryContext.h"
#include "BinaryFunction.h"
#include "BinaryLoop.h"
#include "llvm/MC/MCSymbol.h"

namespace llvm {
namespace bolt {

class BranchPredictionInfo {

public:
  /// An edge indicates that a control flow may go from a basic block (source)
  /// to an other one (destination), and this pair of basic blocks will be used
  /// to index maps and retrieve content of sets.
  typedef std::pair<const MCSymbol *, const MCSymbol *> Edge;

private:
  /// Holds the loop headers of a given function.
  DenseSet<const BinaryBasicBlock *> LoopHeaders;

  /// Holds the loop backedges of a given function.
  DenseSet<Edge> BackEdges;

  /// Holds the loop exit edges of a given function.
  DenseSet<BinaryLoop::Edge> ExitEdges;

  /// Holds the basic blocks of a given function
  /// that contains at least one call instructions.
  DenseSet<const BinaryBasicBlock *> CallSet;

  /// Holds the basic blocks of a given function
  /// that contains at least one store instructions.
  DenseSet<const BinaryBasicBlock *> StoreSet;

  unsigned NumLoads;
  unsigned NumStores;

public:
  unsigned getNumLoads() { return NumLoads; }

  unsigned getNumStores() { return NumStores; }

  /// findLoopEdgesInfo - Finds all loop back edges, loop exit eges
  /// and loop headers within the function.
  void findLoopEdgesInfo(const BinaryLoopInfo &LoopsInfo);

  /// findBasicBlockInfo - Finds all call and store instructions within
  /// the basic blocks of a given function.
  void findBasicBlockInfo(const BinaryFunction &Function, BinaryContext &BC);

  /// isBackEdge - Checks if the edge is a loop back edge.
  bool isBackEdge(const Edge &CFGEdge) const;

  /// isBackEdge - Checks if the edge is a loop back edge.
  bool isBackEdge(const BinaryBasicBlock *SrcBB,
                  const BinaryBasicBlock *DstBB) const;

  /// isExitEdge - Checks if the edge is a loop exit edge.
  bool isExitEdge(const BinaryLoop::Edge &CFGEdge) const;

  /// isExitEdge - Checks if the edge is a loop exit edge.
  bool isExitEdge(const BinaryBasicBlock *SrcBB,
                  const BinaryBasicBlock *DstBB) const;

  /// isLoopHeader - Checks if the basic block is a loop header.
  bool isLoopHeader(const BinaryBasicBlock *BB) const;

  /// hasCallInst - Checks if the basic block has a call instruction.
  bool hasCallInst(const BinaryBasicBlock *BB) const;

  /// hasStoreInst - Checks if the basic block has a store instruction.
  bool hasStoreInst(const BinaryBasicBlock *BB) const;

  /// callToExit - Checks if a basic block invokes exit function.
  bool callToExit(BinaryBasicBlock *BB, BinaryContext &BC) const;

  /// countBackEdges - Compute the number of BB's successor that are back edges.
  unsigned countBackEdges(BinaryBasicBlock *BB) const;

  /// countExitEdges - Compute the number of BB's successor that are exit edges.
  unsigned countExitEdges(BinaryBasicBlock *BB) const;

  /// clear - Cleans up all the content from the data structs used.
  void clear();
};

} // namespace bolt
} // namespace llvm

#endif /* LLVM_TOOLS_LLVM_BOLT_PASSES_BRANCHPREDICTIONINFO_H_ */
