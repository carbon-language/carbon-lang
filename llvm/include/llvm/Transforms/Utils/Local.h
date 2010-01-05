//===-- Local.h - Functions to perform local transformations ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This family of functions perform various local transformations to the
// program.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_LOCAL_H
#define LLVM_TRANSFORMS_UTILS_LOCAL_H

namespace llvm {

class User;
class BasicBlock;
class BranchInst;
class Instruction;
class Value;
class Pass;
class PHINode;
class AllocaInst;
class ConstantExpr;
class TargetData;
class DbgInfoIntrinsic;

template<typename T> class SmallVectorImpl;
  
//===----------------------------------------------------------------------===//
//  Local analysis.
//

/// isSafeToLoadUnconditionally - Return true if we know that executing a load
/// from this value cannot trap.  If it is not obviously safe to load from the
/// specified pointer, we do a quick local scan of the basic block containing
/// ScanFrom, to determine if the address is already accessed.
bool isSafeToLoadUnconditionally(Value *V, Instruction *ScanFrom);

//===----------------------------------------------------------------------===//
//  Local constant propagation.
//

/// ConstantFoldTerminator - If a terminator instruction is predicated on a
/// constant value, convert it into an unconditional branch to the constant
/// destination.  This is a nontrivial operation because the successors of this
/// basic block must have their PHI nodes updated.
///
bool ConstantFoldTerminator(BasicBlock *BB);

//===----------------------------------------------------------------------===//
//  Local dead code elimination.
//

/// isInstructionTriviallyDead - Return true if the result produced by the
/// instruction is not used, and the instruction has no side effects.
///
bool isInstructionTriviallyDead(Instruction *I);

/// RecursivelyDeleteTriviallyDeadInstructions - If the specified value is a
/// trivially dead instruction, delete it.  If that makes any of its operands
/// trivially dead, delete them too, recursively.  Return true if any
/// instructions were deleted.
bool RecursivelyDeleteTriviallyDeadInstructions(Value *V);

/// RecursivelyDeleteDeadPHINode - If the specified value is an effectively
/// dead PHI node, due to being a def-use chain of single-use nodes that
/// either forms a cycle or is terminated by a trivially dead instruction,
/// delete it.  If that makes any of its operands trivially dead, delete them
/// too, recursively.  Return true if the PHI node is actually deleted.
bool RecursivelyDeleteDeadPHINode(PHINode *PN);

//===----------------------------------------------------------------------===//
//  Control Flow Graph Restructuring.
//

/// RemovePredecessorAndSimplify - Like BasicBlock::removePredecessor, this
/// method is called when we're about to delete Pred as a predecessor of BB.  If
/// BB contains any PHI nodes, this drops the entries in the PHI nodes for Pred.
///
/// Unlike the removePredecessor method, this attempts to simplify uses of PHI
/// nodes that collapse into identity values.  For example, if we have:
///   x = phi(1, 0, 0, 0)
///   y = and x, z
///
/// .. and delete the predecessor corresponding to the '1', this will attempt to
/// recursively fold the 'and' to 0.
void RemovePredecessorAndSimplify(BasicBlock *BB, BasicBlock *Pred,
                                  TargetData *TD = 0);
    
  
/// MergeBasicBlockIntoOnlyPred - BB is a block with one predecessor and its
/// predecessor is known to have one successor (BB!).  Eliminate the edge
/// between them, moving the instructions in the predecessor into BB.  This
/// deletes the predecessor block.
///
void MergeBasicBlockIntoOnlyPred(BasicBlock *BB, Pass *P = 0);
    

/// TryToSimplifyUncondBranchFromEmptyBlock - BB is known to contain an
/// unconditional branch, and contains no instructions other than PHI nodes,
/// potential debug intrinsics and the branch.  If possible, eliminate BB by
/// rewriting all the predecessors to branch to the successor block and return
/// true.  If we can't transform, return false.
bool TryToSimplifyUncondBranchFromEmptyBlock(BasicBlock *BB);

/// EliminateDuplicatePHINodes - Check for and eliminate duplicate PHI
/// nodes in this block. This doesn't try to be clever about PHI nodes
/// which differ only in the order of the incoming values, but instcombine
/// orders them so it usually won't matter.
///
bool EliminateDuplicatePHINodes(BasicBlock *BB);

/// SimplifyCFG - This function is used to do simplification of a CFG.  For
/// example, it adjusts branches to branches to eliminate the extra hop, it
/// eliminates unreachable basic blocks, and does other "peephole" optimization
/// of the CFG.  It returns true if a modification was made, possibly deleting
/// the basic block that was pointed to.
///
/// WARNING:  The entry node of a method may not be simplified.
///
bool SimplifyCFG(BasicBlock *BB);

/// FoldBranchToCommonDest - If this basic block is ONLY a setcc and a branch,
/// and if a predecessor branches to us and one of our successors, fold the
/// setcc into the predecessor and use logical operations to pick the right
/// destination.
bool FoldBranchToCommonDest(BranchInst *BI);

/// DemoteRegToStack - This function takes a virtual register computed by an
/// Instruction and replaces it with a slot in the stack frame, allocated via
/// alloca.  This allows the CFG to be changed around without fear of
/// invalidating the SSA information for the value.  It returns the pointer to
/// the alloca inserted to create a stack slot for X.
///
AllocaInst *DemoteRegToStack(Instruction &X,
                             bool VolatileLoads = false,
                             Instruction *AllocaPoint = 0);

/// DemotePHIToStack - This function takes a virtual register computed by a phi
/// node and replaces it with a slot in the stack frame, allocated via alloca.
/// The phi node is deleted and it returns the pointer to the alloca inserted. 
AllocaInst *DemotePHIToStack(PHINode *P, Instruction *AllocaPoint = 0);

/// OnlyUsedByDbgIntrinsics - Return true if the instruction I is only used
/// by DbgIntrinsics. If DbgInUses is specified then the vector is filled 
/// with DbgInfoIntrinsic that use the instruction I.
bool OnlyUsedByDbgInfoIntrinsics(Instruction *I, 
                           SmallVectorImpl<DbgInfoIntrinsic *> *DbgInUses = 0);

} // End llvm namespace

#endif
