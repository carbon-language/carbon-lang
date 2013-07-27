//===-- Analysis/CFG.h - BasicBlock Analyses --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This family of functions performs analyses on basic blocks, and instructions
// contained within basic blocks.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CFG_H
#define LLVM_ANALYSIS_CFG_H

#include "llvm/IR/BasicBlock.h"
#include "llvm/Support/CFG.h"

namespace llvm {

class BasicBlock;
class DominatorTree;
class Function;
class Instruction;
class LoopInfo;
class TerminatorInst;

/// Analyze the specified function to find all of the loop backedges in the
/// function and return them.  This is a relatively cheap (compared to
/// computing dominators and loop info) analysis.
///
/// The output is added to Result, as pairs of <from,to> edge info.
void FindFunctionBackedges(
    const Function &F,
    SmallVectorImpl<std::pair<const BasicBlock *, const BasicBlock *> > &
        Result);

/// Search for the specified successor of basic block BB and return its position
/// in the terminator instruction's list of successors.  It is an error to call
/// this with a block that is not a successor.
unsigned GetSuccessorNumber(BasicBlock *BB, BasicBlock *Succ);

/// Return true if the specified edge is a critical edge. Critical edges are
/// edges from a block with multiple successors to a block with multiple
/// predecessors.
///
bool isCriticalEdge(const TerminatorInst *TI, unsigned SuccNum,
                    bool AllowIdenticalEdges = false);

/// Determine whether there is a path from From to To within a single function.
/// Returns false only if we can prove that once 'From' has been executed then
/// 'To' can not be executed. Conservatively returns true.
///
/// This function is linear with respect to the number of blocks in the CFG,
/// walking down successors from From to reach To, with a fixed threshold.
/// Using DT or LI allows us to answer more quickly. LI reduces the cost of
/// an entire loop of any number of blocsk to be the same as the cost of a
/// single block. DT reduces the cost by allowing the search to terminate when
/// we find a block that dominates the block containing 'To'. DT is most useful
/// on branchy code but not loops, and LI is most useful on code with loops but
/// does not help on branchy code outside loops.
bool isPotentiallyReachable(const Instruction *From, const Instruction *To,
                            DominatorTree *DT = 0, LoopInfo *LI = 0);

} // End llvm namespace

#endif
