//===-- Local.h - Functions to perform local transformations ----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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

//===----------------------------------------------------------------------===//
//  Local constant propagation...
//

/// doConstantPropagation - Constant prop a specific instruction.  Returns true
/// and potentially moves the iterator if constant propagation was performed.
///
bool doConstantPropagation(BasicBlock::iterator &I);

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
//  PHI Instruction Simplification
//

/// hasConstantValue - If the specified PHI node always merges together the same
/// value, return the value, otherwise return null.
///
Value *hasConstantValue(PHINode *PN);


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

} // End llvm namespace

#endif
