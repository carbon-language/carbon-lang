//===- TailRecursionElimination.cpp - Eliminate Tail Calls ----------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements tail recursion elimination.
//
// Caveats: The algorithm implemented is trivially simple.  There are several
// improvements that could be made:
//
//  1. If the function has any alloca instructions, these instructions will not
//     remain in the entry block of the function.  Doing this requires analysis
//     to prove that the alloca is not reachable by the recursively invoked
//     function call.
//  2. Tail recursion is only performed if the call immediately preceeds the
//     return instruction.  Would it be useful to generalize this somehow?
//  3. TRE is only performed if the function returns void or if the return
//     returns the result returned by the call.  It is possible, but unlikely,
//     that the return returns something else (like constant 0), and can still
//     be TRE'd.  It can be TRE'd if ALL OTHER return instructions in the
//     function return the exact same value.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "Support/Statistic.h"

using namespace llvm;

namespace {
  Statistic<> NumEliminated("tailcallelim", "Number of tail calls removed");

  struct TailCallElim : public FunctionPass {
    virtual bool runOnFunction(Function &F);
  };
  RegisterOpt<TailCallElim> X("tailcallelim", "Tail Call Elimination");
}

// Public interface to the TailCallElimination pass
FunctionPass *llvm::createTailCallEliminationPass() {
  return new TailCallElim();
}


bool TailCallElim::runOnFunction(Function &F) {
  // If this function is a varargs function, we won't be able to PHI the args
  // right, so don't even try to convert it...
  if (F.getFunctionType()->isVarArg()) return false;

  BasicBlock *OldEntry = 0;
  std::vector<PHINode*> ArgumentPHIs;
  bool MadeChange = false;

  // Loop over the function, looking for any returning blocks...
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    if (ReturnInst *Ret = dyn_cast<ReturnInst>(BB->getTerminator()))
      if (Ret != BB->begin())  // Make sure there is something before the ret...
        if (CallInst *CI = dyn_cast<CallInst>(Ret->getPrev()))
          // Make sure the tail call is to the current function, and that the
          // return either returns void or returns the value computed by the
          // call.
          if (CI->getCalledFunction() == &F &&
              (Ret->getNumOperands() == 0 || Ret->getReturnValue() == CI)) {
            // Ohh, it looks like we found a tail call, is this the first?
            if (!OldEntry) {
              // Ok, so this is the first tail call we have found in this
              // function.  Insert a new entry block into the function, allowing
              // us to branch back to the old entry block.
              OldEntry = &F.getEntryBlock();
              BasicBlock *NewEntry = new BasicBlock("tailrecurse", OldEntry);
              new BranchInst(OldEntry, NewEntry);
              
              // Now that we have created a new block, which jumps to the entry
              // block, insert a PHI node for each argument of the function.
              // For now, we initialize each PHI to only have the real arguments
              // which are passed in.
              Instruction *InsertPos = OldEntry->begin();
              for (Function::aiterator I = F.abegin(), E = F.aend(); I!=E; ++I){
                PHINode *PN = new PHINode(I->getType(), I->getName()+".tr",
                                          InsertPos);
                I->replaceAllUsesWith(PN); // Everyone use the PHI node now!
                PN->addIncoming(I, NewEntry);
                ArgumentPHIs.push_back(PN);
              }
            }
            
            // Ok, now that we know we have a pseudo-entry block WITH all of the
            // required PHI nodes, add entries into the PHI node for the actual
            // parameters passed into the tail-recursive call.
            for (unsigned i = 0, e = CI->getNumOperands()-1; i != e; ++i)
              ArgumentPHIs[i]->addIncoming(CI->getOperand(i+1), BB);

            // Now that all of the PHI nodes are in place, remove the call and
            // ret instructions, replacing them with an unconditional branch.
            new BranchInst(OldEntry, CI);
            BB->getInstList().pop_back();  // Remove return.
            BB->getInstList().pop_back();  // Remove call.
            MadeChange = true;
            NumEliminated++;
          }
  
  return MadeChange;
}
