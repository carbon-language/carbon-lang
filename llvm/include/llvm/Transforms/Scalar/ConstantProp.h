//===-- ConstantProp.h - Functions for Constant Propogation ------*- C++ -*--=//
//
// This family of functions are useful for performing constant propogation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_CONSTANT_PROPOGATION_H
#define LLVM_TRANSFORMS_SCALAR_CONSTANT_PROPOGATION_H

#include "llvm/BasicBlock.h"
class TerminatorInst;
class Pass;

//===----------------------------------------------------------------------===//
// Normal Constant Propogation Pass
//
Pass *createConstantPropogationPass();

// doConstantPropogation - Constant prop a specific instruction.  Returns true
// and potentially moves the iterator if constant propogation was performed.
//
bool doConstantPropogation(BasicBlock *BB, BasicBlock::iterator &I);

// ConstantFoldTerminator - If a terminator instruction is predicated on a
// constant value, convert it into an unconditional branch to the constant
// destination.  This is a nontrivial operation because the successors of this
// basic block must have their PHI nodes updated.
//
bool ConstantFoldTerminator(BasicBlock *BB, BasicBlock::iterator &I,
                            TerminatorInst *T);


//===----------------------------------------------------------------------===//
// Sparse Conditional Constant Propogation Pass
//
Pass *createSCCPPass();

#endif
