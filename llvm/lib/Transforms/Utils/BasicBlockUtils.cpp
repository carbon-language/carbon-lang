//===-- BasicBlockUtils.cpp - BasicBlock Utilities -------------------------==//
//
// This family of functions perform manipulations on basic blocks, and
// instructions contained within basic blocks.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Function.h"
#include "llvm/Instruction.h"
#include <algorithm>

// ReplaceInstWithValue - Replace all uses of an instruction (specified by BI)
// with a value, then remove and delete the original instruction.
//
void ReplaceInstWithValue(BasicBlock::InstListType &BIL,
                          BasicBlock::iterator &BI, Value *V) {
  Instruction *I = *BI;
  // Replaces all of the uses of the instruction with uses of the value
  I->replaceAllUsesWith(V);

  // Remove the unneccesary instruction now...
  BIL.remove(BI);

  // Make sure to propogate a name if there is one already...
  if (I->hasName() && !V->hasName())
    V->setName(I->getName(), BIL.getParent()->getSymbolTable());

  // Remove the dead instruction now...
  delete I;
}


// ReplaceInstWithInst - Replace the instruction specified by BI with the
// instruction specified by I.  The original instruction is deleted and BI is
// updated to point to the new instruction.
//
void ReplaceInstWithInst(BasicBlock::InstListType &BIL,
                         BasicBlock::iterator &BI, Instruction *I) {
  assert(I->getParent() == 0 &&
         "ReplaceInstWithInst: Instruction already inserted into basic block!");

  // Insert the new instruction into the basic block...
  BI = BIL.insert(BI, I)+1;  // Increment BI to point to instruction to delete

  // Replace all uses of the old instruction, and delete it.
  ReplaceInstWithValue(BIL, BI, I);

  // Move BI back to point to the newly inserted instruction
  --BI;
}

// ReplaceInstWithInst - Replace the instruction specified by From with the
// instruction specified by To.  Note that this is slower than providing an
// iterator directly, because the basic block containing From must be searched
// for the instruction.
//
void ReplaceInstWithInst(Instruction *From, Instruction *To) {
  BasicBlock *BB = From->getParent();
  BasicBlock::InstListType &BIL = BB->getInstList();
  BasicBlock::iterator BI = find(BIL.begin(), BIL.end(), From);
  assert(BI != BIL.end() && "Inst not in it's parents BB!");
  ReplaceInstWithInst(BIL, BI, To);
}

// InsertInstBeforeInst - Insert 'NewInst' into the basic block that 'Existing'
// is already in, and put it right before 'Existing'.  This instruction should
// only be used when there is no iterator to Existing already around.  The 
// returned iterator points to the new instruction.
//
BasicBlock::iterator InsertInstBeforeInst(Instruction *NewInst,
                                          Instruction *Existing) {
  BasicBlock *BB = Existing->getParent();
  BasicBlock::InstListType &BIL = BB->getInstList();
  BasicBlock::iterator BI = find(BIL.begin(), BIL.end(), Existing);
  assert(BI != BIL.end() && "Inst not in it's parents BB!");
  return BIL.insert(BI, NewInst);
}

