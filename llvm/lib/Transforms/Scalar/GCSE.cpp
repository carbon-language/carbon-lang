//===-- GCSE.cpp - SSA based Global Common Subexpr Elimination ------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This pass is designed to be a very quick global transformation that
// eliminates global common subexpressions from a function.  It does this by
// using an existing value numbering implementation to identify the common
// subexpressions, eliminating them when possible.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/BasicBlock.h"
#include "llvm/Constant.h"
#include "llvm/Instructions.h"
#include "llvm/Type.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/ValueNumbering.h"
#include "llvm/Transforms/Utils/Local.h"
#include "Support/DepthFirstIterator.h"
#include "Support/Statistic.h"
#include <algorithm>
using namespace llvm;

namespace {
  Statistic<> NumInstRemoved("gcse", "Number of instructions removed");
  Statistic<> NumLoadRemoved("gcse", "Number of loads removed");
  Statistic<> NumCallRemoved("gcse", "Number of calls removed");
  Statistic<> NumNonInsts   ("gcse", "Number of instructions removed due "
                             "to non-instruction values");

  struct GCSE : public FunctionPass {
    virtual bool runOnFunction(Function &F);

  private:
    void ReplaceInstructionWith(Instruction *I, Value *V);

    // This transformation requires dominator and immediate dominator info
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<DominatorSet>();
      AU.addRequired<DominatorTree>();
      AU.addRequired<ValueNumbering>();
    }
  };

  RegisterOpt<GCSE> X("gcse", "Global Common Subexpression Elimination");
}

// createGCSEPass - The public interface to this file...
FunctionPass *llvm::createGCSEPass() { return new GCSE(); }

// GCSE::runOnFunction - This is the main transformation entry point for a
// function.
//
bool GCSE::runOnFunction(Function &F) {
  bool Changed = false;

  // Get pointers to the analysis results that we will be using...
  DominatorSet &DS = getAnalysis<DominatorSet>();
  ValueNumbering &VN = getAnalysis<ValueNumbering>();
  DominatorTree &DT = getAnalysis<DominatorTree>();

  std::vector<Value*> EqualValues;

  // Traverse the CFG of the function in dominator order, so that we see each
  // instruction after we see its operands.
  for (df_iterator<DominatorTree::Node*> DI = df_begin(DT.getRootNode()),
         E = df_end(DT.getRootNode()); DI != E; ++DI) {
    BasicBlock *BB = DI->getBlock();

    // Remember which instructions we've seen in this basic block as we scan.
    std::set<Instruction*> BlockInsts;

    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ) {
      Instruction *Inst = I++;

      // If this instruction computes a value, try to fold together common
      // instructions that compute it.
      //
      if (Inst->getType() != Type::VoidTy) {
        VN.getEqualNumberNodes(Inst, EqualValues);

        // If this instruction computes a value that is already computed
        // elsewhere, try to recycle the old value.
        if (!EqualValues.empty()) {
          if (Inst == &*BB->begin())
            I = BB->end();
          else {
            I = Inst; --I;
          }
          
          // First check to see if we were able to value number this instruction
          // to a non-instruction value.  If so, prefer that value over other
          // instructions which may compute the same thing.
          for (unsigned i = 0, e = EqualValues.size(); i != e; ++i)
            if (!isa<Instruction>(EqualValues[i])) {
              ++NumNonInsts;      // Keep track of # of insts repl with values

              // Change all users of Inst to use the replacement and remove it
              // from the program.
              ReplaceInstructionWith(Inst, EqualValues[i]);
              Inst = 0;
              EqualValues.clear();  // don't enter the next loop
              break;
            }

          // If there were no non-instruction values that this instruction
          // produces, find a dominating instruction that produces the same
          // value.  If we find one, use it's value instead of ours.
          for (unsigned i = 0, e = EqualValues.size(); i != e; ++i) {
            Instruction *OtherI = cast<Instruction>(EqualValues[i]);
            bool Dominates = false;
            if (OtherI->getParent() == BB)
              Dominates = BlockInsts.count(OtherI);
            else
              Dominates = DS.dominates(OtherI->getParent(), BB);

            if (Dominates) {
              // Okay, we found an instruction with the same value as this one
              // and that dominates this one.  Replace this instruction with the
              // specified one.
              ReplaceInstructionWith(Inst, OtherI);
              Inst = 0;
              break;
            }
          }

          EqualValues.clear();

          if (Inst) {
            I = Inst; ++I;             // Deleted no instructions
          } else if (I == BB->end()) { // Deleted first instruction
            I = BB->begin();
          } else {                     // Deleted inst in middle of block.
            ++I;
          }
        }

        if (Inst)
          BlockInsts.insert(Inst);
      }
    }
  }

  // When the worklist is empty, return whether or not we changed anything...
  return Changed;
}


void GCSE::ReplaceInstructionWith(Instruction *I, Value *V) {
  if (isa<LoadInst>(I))
    ++NumLoadRemoved; // Keep track of loads eliminated
  if (isa<CallInst>(I))
    ++NumCallRemoved; // Keep track of calls eliminated
  ++NumInstRemoved;   // Keep track of number of insts eliminated

  // Update value numbering
  getAnalysis<ValueNumbering>().deleteInstruction(I);

  // If we are not replacing the instruction with a constant, we cannot do
  // anything special.
  if (!isa<Constant>(V)) {
    I->replaceAllUsesWith(V);
    
    // Erase the instruction from the program.
    I->getParent()->getInstList().erase(I);
    return;
  }

  Constant *C = cast<Constant>(V);
  std::vector<User*> Users(I->use_begin(), I->use_end());

  // Perform the replacement.
  I->replaceAllUsesWith(C);

  // Erase the instruction from the program.
  I->getParent()->getInstList().erase(I);
  
  // Check each user to see if we can constant fold it.
  while (!Users.empty()) {
    Instruction *U = cast<Instruction>(Users.back());
    Users.pop_back();

    if (Constant *C = ConstantFoldInstruction(U)) {
      ReplaceInstructionWith(U, C);

      // If the instruction used I more than once, it could be on the user list
      // multiple times.  Make sure we don't reprocess it.
      std::vector<User*>::iterator It = std::find(Users.begin(), Users.end(),U);
      while (It != Users.end()) {
        Users.erase(It);
        It = std::find(Users.begin(), Users.end(), U);
      }
    }
  }
}
