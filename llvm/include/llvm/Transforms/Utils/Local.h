//===-- Local.h - Functions to perform local transformations -----*- C++ -*--=//
//
// This family of functions perform various local transformations to the
// program.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_LOCAL_H
#define LLVM_TRANSFORMS_UTILS_LOCAL_H

#include "llvm/Function.h"
class Pass;

//===----------------------------------------------------------------------===//
//  Local constant propogation...
//

/// doConstantPropogation - Constant prop a specific instruction.  Returns true
/// and potentially moves the iterator if constant propogation was performed.
///
bool doConstantPropogation(BasicBlock::iterator &I);

/// ConstantFoldTerminator - If a terminator instruction is predicated on a
/// constant value, convert it into an unconditional branch to the constant
/// destination.  This is a nontrivial operation because the successors of this
/// basic block must have their PHI nodes updated.
///
bool ConstantFoldTerminator(BasicBlock *BB);


//===----------------------------------------------------------------------===//
//  Local dead code elimination...
//

/// isInstructionTriviallyDead - Return true if the result produced by the
/// instruction is not used, and the instruction has no side effects.
///
bool isInstructionTriviallyDead(Instruction *I);


/// dceInstruction - Inspect the instruction at *BBI and figure out if it
/// isTriviallyDead.  If so, remove the instruction and update the iterator to
/// point to the instruction that immediately succeeded the original
/// instruction.
///
bool dceInstruction(BasicBlock::iterator &BBI);


//===----------------------------------------------------------------------===//
//  Control Flow Graph Restructuring...
//

/// SimplifyCFG - This function is used to do simplification of a CFG.  For
/// example, it adjusts branches to branches to eliminate the extra hop, it
/// eliminates unreachable basic blocks, and does other "peephole" optimization
/// of the CFG.  It returns true if a modification was made, possibly deleting
/// the basic block that was pointed to.
///
/// WARNING:  The entry node of a method may not be simplified.
///
bool SimplifyCFG(BasicBlock *BB);


/// isCriticalEdge - Return true if the specified edge is a critical edge.
/// Critical edges are edges from a block with multiple successors to a block
/// with multiple predecessors.
///
///
bool isCriticalEdge(const TerminatorInst *TI, unsigned SuccNum);

/// SplitCriticalEdge - Insert a new node node to split the critical edge.  This
/// will update DominatorSet, ImmediateDominator and DominatorTree information
/// if it is available, thus calling this pass will not invalidate either of
/// them.
///
void SplitCriticalEdge(TerminatorInst *TI, unsigned SuccNum, Pass *P = 0);

#endif
