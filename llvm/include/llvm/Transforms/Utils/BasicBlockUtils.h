//===-- Transform/Utils/BasicBlockUtils.h - BasicBlock Utilities -*- C++ -*-==//
//
// This family of functions perform manipulations on basic blocks, and
// instructions contained within basic blocks.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_BASICBLOCK_H
#define LLVM_TRANSFORMS_UTILS_BASICBLOCK_H

// FIXME: Move to this file: BasicBlock::removePredecessor, BB::splitBasicBlock

#include "llvm/BasicBlock.h"
class Instruction;


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
// instruction specified by To.  Note that this is slower than providing an
// iterator directly, because the basic block containing From must be searched
// for the instruction.
//
void ReplaceInstWithInst(Instruction *From, Instruction *To);

#endif
