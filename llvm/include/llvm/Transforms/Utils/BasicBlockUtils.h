//===-- Transform/Utils/BasicBlockUtils.h - BasicBlock Utils ----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This family of functions perform manipulations on basic blocks, and
// instructions contained within basic blocks.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_BASICBLOCK_H
#define LLVM_TRANSFORMS_UTILS_BASICBLOCK_H

// FIXME: Move to this file: BasicBlock::removePredecessor, BB::splitBasicBlock

#include "llvm/BasicBlock.h"
#include "llvm/Support/CFG.h"
class Instruction;
class Pass;

// ReplaceInstWithValue - Replace all uses of an instruction (specified by BI)
// with a value, then remove and delete the original instruction.
//
void ReplaceInstWithValue(BasicBlock::InstListType &BIL,
                          BasicBlock::iterator &BI, Value *V);

// ReplaceInstWithInst - Replace the instruction specified by BI with the
// instruction specified by I.  The original instruction is deleted and BI is
// updated to point to the new instruction.
//
void ReplaceInstWithInst(BasicBlock::InstListType &BIL,
                         BasicBlock::iterator &BI, Instruction *I);

// ReplaceInstWithInst - Replace the instruction specified by From with the
// instruction specified by To.
//
void ReplaceInstWithInst(Instruction *From, Instruction *To);


// RemoveSuccessor - Change the specified terminator instruction such that its
// successor #SuccNum no longer exists.  Because this reduces the outgoing
// degree of the current basic block, the actual terminator instruction itself
// may have to be changed.  In the case where the last successor of the block is
// deleted, a return instruction is inserted in its place which can cause a
// suprising change in program behavior if it is not expected.
//
void RemoveSuccessor(TerminatorInst *TI, unsigned SuccNum);


/// isCriticalEdge - Return true if the specified edge is a critical edge.
/// Critical edges are edges from a block with multiple successors to a block
/// with multiple predecessors.
///
bool isCriticalEdge(const TerminatorInst *TI, unsigned SuccNum);

/// SplitCriticalEdge - If this edge is a critical edge, insert a new node to
/// split the critical edge.  This will update DominatorSet, ImmediateDominator,
/// DominatorTree, and DominatorFrontier information if it is available, thus
/// calling this pass will not invalidate either of them.  This returns true if
/// the edge was split, false otherwise.
///
bool SplitCriticalEdge(TerminatorInst *TI, unsigned SuccNum, Pass *P = 0);

inline bool SplitCriticalEdge(BasicBlock *BB, succ_iterator SI, Pass *P = 0) {
  return SplitCriticalEdge(BB->getTerminator(), SI.getSuccessorIndex(), P);
}

/// SplitCriticalEdge - If the edge from *PI to BB is not critical, return
/// false.  Otherwise, split all edges between the two blocks and return true.
/// This updates all of the same analyses as the other SplitCriticalEdge
/// function.
inline bool SplitCriticalEdge(BasicBlock *Succ, pred_iterator PI, Pass *P = 0) {
  BasicBlock *Pred = *PI;
  bool MadeChange = false;
  for (succ_iterator SI = succ_begin(Pred), E = succ_end(Pred); SI != E; ++SI)
    if (*SI == Succ)
      MadeChange |= SplitCriticalEdge(Pred, SI, P);
  return MadeChange;
}


#endif
