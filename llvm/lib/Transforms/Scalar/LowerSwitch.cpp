//===- LowerSwitch.cpp - Eliminate Switch instructions --------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// The LowerSwitch transformation rewrites switch statements with a sequence of
// branches, which allows targets to get away with not implementing the switch
// statement until it is convenient.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/iTerminators.h"
#include "llvm/iOperators.h"
#include "llvm/iPHINode.h"
#include "llvm/Pass.h"
#include "Support/Debug.h"
#include "Support/Statistic.h"
using namespace llvm;

namespace {
  Statistic<> NumLowered("lowerswitch", "Number of SwitchInst's replaced");

  /// LowerSwitch Pass - Replace all SwitchInst instructions with chained branch
  /// instructions.  Note that this cannot be a BasicBlock pass because it
  /// modifies the CFG!
  class LowerSwitch : public FunctionPass {
  public:
    bool runOnFunction(Function &F);
    typedef std::pair<Constant*, BasicBlock*> Case;
    typedef std::vector<Case>::iterator       CaseItr;
  private:
    void processSwitchInst(SwitchInst *SI);

    BasicBlock* switchConvert(CaseItr Begin, CaseItr End, Value* Val,
                              BasicBlock* OrigBlock, BasicBlock* Default);
    BasicBlock* newLeafBlock(Case& Leaf, Value* Val,
                             BasicBlock* OrigBlock, BasicBlock* Default);
  };

  /// The comparison function for sorting the switch case values in the vector.
  struct CaseCmp {
    bool operator () (const LowerSwitch::Case& C1,
                      const LowerSwitch::Case& C2) {
      if (const ConstantUInt* U1 = dyn_cast<const ConstantUInt>(C1.first))
        return U1->getValue() < cast<const ConstantUInt>(C2.first)->getValue();

      const ConstantSInt* S1 = dyn_cast<const ConstantSInt>(C1.first);
      return S1->getValue() < cast<const ConstantSInt>(C2.first)->getValue();
    }
  };

  RegisterOpt<LowerSwitch>
  X("lowerswitch", "Lower SwitchInst's to branches");
}

// createLowerSwitchPass - Interface to this file...
FunctionPass *llvm::createLowerSwitchPass() {
  return new LowerSwitch();
}

bool LowerSwitch::runOnFunction(Function &F) {
  bool Changed = false;

  for (Function::iterator I = F.begin(), E = F.end(); I != E; ) {
    BasicBlock *Cur = I++; // Advance over block so we don't traverse new blocks

    if (SwitchInst *SI = dyn_cast<SwitchInst>(Cur->getTerminator())) {
      Changed = true;
      processSwitchInst(SI);
    }
  }

  return Changed;
}

// operator<< - Used for debugging purposes.
//
std::ostream& operator<<(std::ostream &O,
                         const std::vector<LowerSwitch::Case> &C) {
  O << "[";

  for (std::vector<LowerSwitch::Case>::const_iterator B = C.begin(),
         E = C.end(); B != E; ) {
    O << *B->first;
    if (++B != E) O << ", ";
  }

  return O << "]";
}

// switchConvert - Convert the switch statement into a binary lookup of
// the case values. The function recursively builds this tree.
//
BasicBlock* LowerSwitch::switchConvert(CaseItr Begin, CaseItr End,
                                       Value* Val, BasicBlock* OrigBlock,
                                       BasicBlock* Default)
{
  unsigned Size = End - Begin;

  if (Size == 1)
    return newLeafBlock(*Begin, Val, OrigBlock, Default);

  unsigned Mid = Size / 2;
  std::vector<Case> LHS(Begin, Begin + Mid);
  DEBUG(std::cerr << "LHS: " << LHS << "\n");
  std::vector<Case> RHS(Begin + Mid, End);
  DEBUG(std::cerr << "RHS: " << RHS << "\n");

  Case& Pivot = *(Begin + Mid);
  DEBUG(std::cerr << "Pivot ==> "
                  << cast<ConstantUInt>(Pivot.first)->getValue() << "\n");

  BasicBlock* LBranch = switchConvert(LHS.begin(), LHS.end(), Val,
                                      OrigBlock, Default);
  BasicBlock* RBranch = switchConvert(RHS.begin(), RHS.end(), Val,
                                      OrigBlock, Default);

  // Create a new node that checks if the value is < pivot. Go to the
  // left branch if it is and right branch if not.
  Function* F = OrigBlock->getParent();
  BasicBlock* NewNode = new BasicBlock("NodeBlock");
  F->getBasicBlockList().insert(OrigBlock->getNext(), NewNode);

  SetCondInst* Comp = new SetCondInst(Instruction::SetLT, Val, Pivot.first,
                                      "Pivot");
  NewNode->getInstList().push_back(Comp);
  new BranchInst(LBranch, RBranch, Comp, NewNode);
  return NewNode;
}

// newLeafBlock - Create a new leaf block for the binary lookup tree. It
// checks if the switch's value == the case's value. If not, then it
// jumps to the default branch. At this point in the tree, the value
// can't be another valid case value, so the jump to the "default" branch
// is warranted.
//
BasicBlock* LowerSwitch::newLeafBlock(Case& Leaf, Value* Val,
                                      BasicBlock* OrigBlock,
                                      BasicBlock* Default)
{
  Function* F = OrigBlock->getParent();
  BasicBlock* NewLeaf = new BasicBlock("LeafBlock");
  F->getBasicBlockList().insert(OrigBlock->getNext(), NewLeaf);

  // Make the seteq instruction...
  SetCondInst* Comp = new SetCondInst(Instruction::SetEQ, Val,
                                      Leaf.first, "SwitchLeaf");
  NewLeaf->getInstList().push_back(Comp);

  // Make the conditional branch...
  BasicBlock* Succ = Leaf.second;
  new BranchInst(Succ, Default, Comp, NewLeaf);

  // If there were any PHI nodes in this successor, rewrite one entry
  // from OrigBlock to come from NewLeaf.
  for (BasicBlock::iterator I = Succ->begin();
       PHINode* PN = dyn_cast<PHINode>(I); ++I) {
    int BlockIdx = PN->getBasicBlockIndex(OrigBlock);
    assert(BlockIdx != -1 && "Switch didn't go to this successor??");
    PN->setIncomingBlock((unsigned)BlockIdx, NewLeaf);
  }

  return NewLeaf;
}

// processSwitchInst - Replace the specified switch instruction with a sequence
// of chained if-then insts in a balanced binary search.
//
void LowerSwitch::processSwitchInst(SwitchInst *SI) {
  BasicBlock *CurBlock = SI->getParent();
  BasicBlock *OrigBlock = CurBlock;
  Function *F = CurBlock->getParent();
  Value *Val = SI->getOperand(0);  // The value we are switching on...
  BasicBlock* Default = SI->getDefaultDest();

  // Unlink the switch instruction from it's block.
  CurBlock->getInstList().remove(SI);

  // If there is only the default destination, don't bother with the code below.
  if (SI->getNumOperands() == 2) {
    new BranchInst(SI->getDefaultDest(), CurBlock);
    delete SI;
    return;
  }

  // Create a new, empty default block so that the new hierarchy of
  // if-then statements go to this and the PHI nodes are happy.
  BasicBlock* NewDefault = new BasicBlock("NewDefault");
  F->getBasicBlockList().insert(Default, NewDefault);

  new BranchInst(Default, NewDefault);

  // If there is an entry in any PHI nodes for the default edge, make sure
  // to update them as well.
  for (BasicBlock::iterator I = Default->begin();
       PHINode *PN = dyn_cast<PHINode>(I); ++I) {
    int BlockIdx = PN->getBasicBlockIndex(OrigBlock);
    assert(BlockIdx != -1 && "Switch didn't go to this successor??");
    PN->setIncomingBlock((unsigned)BlockIdx, NewDefault);
  }

  std::vector<Case> Cases;

  // Expand comparisons for all of the non-default cases...
  for (unsigned i = 1; i < SI->getNumSuccessors(); ++i)
    Cases.push_back(Case(SI->getSuccessorValue(i), SI->getSuccessor(i)));

  std::sort(Cases.begin(), Cases.end(), CaseCmp());
  DEBUG(std::cerr << "Cases: " << Cases << "\n");
  BasicBlock* SwitchBlock = switchConvert(Cases.begin(), Cases.end(), Val,
                                          OrigBlock, NewDefault);

  // Branch to our shiny new if-then stuff...
  new BranchInst(SwitchBlock, OrigBlock);

  // We are now done with the switch instruction, delete it.
  delete SI;
}
