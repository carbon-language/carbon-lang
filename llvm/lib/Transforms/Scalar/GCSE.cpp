//===-- GCSE.cpp - SSA based Global Common Subexpr Elimination ------------===//
//
// This pass is designed to be a very quick global transformation that
// eliminates global common subexpressions from a function.  It does this by
// using an existing value numbering implementation to identify the common
// subexpressions, eliminating them when possible.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/iMemory.h"
#include "llvm/Type.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/ValueNumbering.h"
#include "llvm/Support/InstIterator.h"
#include "Support/Statistic.h"
#include <algorithm>

namespace {
  Statistic<> NumInstRemoved("gcse", "Number of instructions removed");
  Statistic<> NumLoadRemoved("gcse", "Number of loads removed");
  Statistic<> NumNonInsts   ("gcse", "Number of instructions removed due "
                             "to non-instruction values");

  class GCSE : public FunctionPass {
    std::set<Instruction*>  WorkList;
    DominatorSet           *DomSetInfo;
    ValueNumbering         *VN;
  public:
    virtual bool runOnFunction(Function &F);

  private:
    bool EliminateRedundancies(Instruction *I,std::vector<Value*> &EqualValues);
    Instruction *EliminateCSE(Instruction *I, Instruction *Other);
    void ReplaceInstWithInst(Instruction *First, BasicBlock::iterator SI);

    // This transformation requires dominator and immediate dominator info
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<DominatorSet>();
      AU.addRequired<ImmediateDominators>();
      AU.addRequired<ValueNumbering>();
    }
  };

  RegisterOpt<GCSE> X("gcse", "Global Common Subexpression Elimination");
}

// createGCSEPass - The public interface to this file...
Pass *createGCSEPass() { return new GCSE(); }


// GCSE::runOnFunction - This is the main transformation entry point for a
// function.
//
bool GCSE::runOnFunction(Function &F) {
  bool Changed = false;

  // Get pointers to the analysis results that we will be using...
  DomSetInfo = &getAnalysis<DominatorSet>();
  VN = &getAnalysis<ValueNumbering>();

  // Step #1: Add all instructions in the function to the worklist for
  // processing.  All of the instructions are considered to be our
  // subexpressions to eliminate if possible.
  //
  WorkList.insert(inst_begin(F), inst_end(F));

  // Step #2: WorkList processing.  Iterate through all of the instructions,
  // checking to see if there are any additionally defined subexpressions in the
  // program.  If so, eliminate them!
  //
  while (!WorkList.empty()) {
    Instruction &I = **WorkList.begin(); // Get an instruction from the worklist
    WorkList.erase(WorkList.begin());

    // If this instruction computes a value, try to fold together common
    // instructions that compute it.
    //
    if (I.getType() != Type::VoidTy) {
      std::vector<Value*> EqualValues;
      VN->getEqualNumberNodes(&I, EqualValues);

      if (!EqualValues.empty())
        Changed |= EliminateRedundancies(&I, EqualValues);
    }
  }

  // When the worklist is empty, return whether or not we changed anything...
  return Changed;
}

bool GCSE::EliminateRedundancies(Instruction *I,
                                 std::vector<Value*> &EqualValues) {
  // If the EqualValues set contains any non-instruction values, then we know
  // that all of the instructions can be replaced with the non-instruction value
  // because it is guaranteed to dominate all of the instructions in the
  // function.  We only have to do hard work if all we have are instructions.
  //
  for (unsigned i = 0, e = EqualValues.size(); i != e; ++i)
    if (!isa<Instruction>(EqualValues[i])) {
      // Found a non-instruction.  Replace all instructions with the
      // non-instruction.
      //
      Value *Replacement = EqualValues[i];

      // Make sure we get I as well...
      EqualValues[i] = I;

      // Replace all instructions with the Replacement value.
      for (i = 0; i != e; ++i)
        if (Instruction *I = dyn_cast<Instruction>(EqualValues[i])) {
          // Change all users of I to use Replacement.
          I->replaceAllUsesWith(Replacement);

          if (isa<LoadInst>(I))
            ++NumLoadRemoved; // Keep track of loads eliminated
          ++NumInstRemoved;   // Keep track of number of instructions eliminated
          ++NumNonInsts;      // Keep track of number of insts repl with values

          // Erase the instruction from the program.
          I->getParent()->getInstList().erase(I);
          WorkList.erase(I);
        }
      
      return true;
    }
  
  // Remove duplicate entries from EqualValues...
  std::sort(EqualValues.begin(), EqualValues.end());
  EqualValues.erase(std::unique(EqualValues.begin(), EqualValues.end()),
                    EqualValues.end());

  // From this point on, EqualValues is logically a vector of instructions.
  //
  bool Changed = false;
  EqualValues.push_back(I); // Make sure I is included...
  while (EqualValues.size() > 1) {
    // FIXME, this could be done better than simple iteration!
    Instruction *Test = cast<Instruction>(EqualValues.back());
    EqualValues.pop_back();
    
    for (unsigned i = 0, e = EqualValues.size(); i != e; ++i)
      if (Instruction *Ret = EliminateCSE(Test,
                                          cast<Instruction>(EqualValues[i]))) {
        if (Ret == Test)          // Eliminated EqualValues[i]
          EqualValues[i] = Test;  // Make sure that we reprocess I at some point
        Changed = true;
        break;
      }
  }
  return Changed;
}


// ReplaceInstWithInst - Destroy the instruction pointed to by SI, making all
// uses of the instruction use First now instead.
//
void GCSE::ReplaceInstWithInst(Instruction *First, BasicBlock::iterator SI) {
  Instruction &Second = *SI;
  
  //cerr << "DEL " << (void*)Second << Second;

  // Add the first instruction back to the worklist
  WorkList.insert(First);

  // Add all uses of the second instruction to the worklist
  for (Value::use_iterator UI = Second.use_begin(), UE = Second.use_end();
       UI != UE; ++UI)
    WorkList.insert(cast<Instruction>(*UI));
    
  // Make all users of 'Second' now use 'First'
  Second.replaceAllUsesWith(First);

  // Erase the second instruction from the program
  Second.getParent()->getInstList().erase(SI);
}

// EliminateCSE - The two instruction I & Other have been found to be common
// subexpressions.  This function is responsible for eliminating one of them,
// and for fixing the worklist to be correct.  The instruction that is preserved
// is returned from the function if the other is eliminated, otherwise null is
// returned.
//
Instruction *GCSE::EliminateCSE(Instruction *I, Instruction *Other) {
  assert(I != Other);

  WorkList.erase(I);
  WorkList.erase(Other); // Other may not actually be on the worklist anymore...

  // Handle the easy case, where both instructions are in the same basic block
  BasicBlock *BB1 = I->getParent(), *BB2 = Other->getParent();
  Instruction *Ret = 0;

  if (BB1 == BB2) {
    // Eliminate the second occuring instruction.  Add all uses of the second
    // instruction to the worklist.
    //
    // Scan the basic block looking for the "first" instruction
    BasicBlock::iterator BI = BB1->begin();
    while (&*BI != I && &*BI != Other) {
      ++BI;
      assert(BI != BB1->end() && "Instructions not found in parent BB!");
    }

    // Keep track of which instructions occurred first & second
    Instruction *First = BI;
    Instruction *Second = I != First ? I : Other; // Get iterator to second inst
    BI = Second;

    // Destroy Second, using First instead.
    ReplaceInstWithInst(First, BI);
    Ret = First;

    // Otherwise, the two instructions are in different basic blocks.  If one
    // dominates the other instruction, we can simply use it
    //
  } else if (DomSetInfo->dominates(BB1, BB2)) {    // I dom Other?
    ReplaceInstWithInst(I, Other);
    Ret = I;
  } else if (DomSetInfo->dominates(BB2, BB1)) {    // Other dom I?
    ReplaceInstWithInst(Other, I);
    Ret = Other;
  } else {
    // This code is disabled because it has several problems:
    // One, the actual assumption is wrong, as shown by this code:
    // int "test"(int %X, int %Y) {
    //         %Z = add int %X, %Y
    //         ret int %Z
    // Unreachable:
    //         %Q = add int %X, %Y
    //         ret int %Q
    // }
    //
    // Here there are no shared dominators.  Additionally, this had the habit of
    // moving computations where they were not always computed.  For example, in
    // a case like this:
    //  if (c) {
    //    if (d)  ...
    //    else ... X+Y ...
    //  } else {
    //    ... X+Y ...
    //  }
    // 
    // In thiscase, the expression would be hoisted to outside the 'if' stmt,
    // causing the expression to be evaluated, even for the if (d) path, which
    // could cause problems, if, for example, it caused a divide by zero.  In
    // general the problem this case is trying to solve is better addressed with
    // PRE than GCSE.
    //
    return 0;
  }

  if (isa<LoadInst>(Ret))
    ++NumLoadRemoved;  // Keep track of loads eliminated
  ++NumInstRemoved;   // Keep track of number of instructions eliminated

  // Add all users of Ret to the worklist...
  for (Value::use_iterator I = Ret->use_begin(), E = Ret->use_end(); I != E;++I)
    if (Instruction *Inst = dyn_cast<Instruction>(*I))
      WorkList.insert(Inst);

  return Ret;
}
