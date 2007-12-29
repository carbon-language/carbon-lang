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

#include "llvm/Function.h"

namespace llvm {

class Pass;
class PHINode;
class AllocaInst;
class ConstantExpr;
class TargetData;

//===----------------------------------------------------------------------===//
//  Local constant propagation...
//

/// doConstantPropagation - Constant prop a specific instruction.  Returns true
/// and potentially moves the iterator if constant propagation was performed.
///
bool doConstantPropagation(BasicBlock::iterator &I, const TargetData *TD = 0);

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

/// DemoteRegToStack - This function takes a virtual register computed by an
/// Instruction and replaces it with a slot in the stack frame, allocated via
/// alloca.  This allows the CFG to be changed around without fear of
/// invalidating the SSA information for the value.  It returns the pointer to
/// the alloca inserted to create a stack slot for X.
///
AllocaInst *DemoteRegToStack(Instruction &X, bool VolatileLoads = false,
                             Instruction *AllocaPoint = NULL);

/// DemotePHIToStack - This function takes a virtual register computed by a phi
/// node and replaces it with a slot in the stack frame, allocated via alloca.
/// The phi node is deleted and it returns the pointer to the alloca inserted. 
AllocaInst *DemotePHIToStack(PHINode *P, Instruction *AllocaPoint = NULL);

} // End llvm namespace

#endif
