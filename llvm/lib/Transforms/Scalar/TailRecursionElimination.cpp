//===- TailRecursionElimination.cpp - Eliminate Tail Calls ----------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file transforms calls of the current function (self recursion) followed
// by a return instruction with a branch to the entry of the function, creating
// a loop.  This pass also implements the following extensions to the basic
// algorithm:
//
//  1. Trivial instructions between the call and return do not prevent the
//     transformation from taking place, though currently the analysis cannot
//     support moving any really useful instructions (only dead ones).
//
// There are several improvements that could be made:
//
//  1. If the function has any alloca instructions, these instructions will be
//     moved out of the entry block of the function, causing them to be
//     evaluated each time through the tail recursion.  Safely keeping allocas
//     in the entry block requires analysis to proves that the tail-called
//     function does not read or write the stack object.
//  2. Tail recursion is only performed if the call immediately preceeds the
//     return instruction.  It's possible that there could be a jump between
//     the call and the return.
//  3. TRE is only performed if the function returns void or if the return
//     returns the result returned by the call.  It is possible, but unlikely,
//     that the return returns something else (like constant 0), and can still
//     be TRE'd.  It can be TRE'd if ALL OTHER return instructions in the
//     function return the exact same value.
//  4. There can be intervening operations between the call and the return that
//     prevent the TRE from occurring.  For example, there could be GEP's and
//     stores to memory that will not be read or written by the call.  This
//     requires some substantial analysis (such as with DSA) to prove safe to
//     move ahead of the call, but doing so could allow many more TREs to be
//     performed, for example in TreeAdd/TreeAlloc from the treeadd benchmark.
//  5. This pass could transform functions that are prevented from being tail
//     recursive by a commutative expression to use an accumulator helper
//     function, thus compiling the typical naive factorial or 'fib'
//     implementation into efficient code.
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

  private:
    bool ProcessReturningBlock(ReturnInst *RI, BasicBlock *&OldEntry,
                               std::vector<PHINode*> &ArgumentPHIs);
    bool CanMoveAboveCall(Instruction *I, CallInst *CI);
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
      MadeChange |= ProcessReturningBlock(Ret, OldEntry, ArgumentPHIs);
  
  return MadeChange;
}


// CanMoveAboveCall - Return true if it is safe to move the specified
// instruction from after the call to before the call, assuming that all
// instructions between the call and this instruction are movable.
//
bool TailCallElim::CanMoveAboveCall(Instruction *I, CallInst *CI) {
  // FIXME: We can move load/store/call/free instructions above the call if the
  // call does not mod/ref the memory location being processed.
  if (I->mayWriteToMemory() || isa<LoadInst>(I))
    return false;

  // Otherwise, if this is a side-effect free instruction, check to make sure
  // that it does not use the return value of the call.  If it doesn't use the
  // return value of the call, it must only use things that are defined before
  // the call, or movable instructions between the call and the instruction
  // itself.
  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
    if (I->getOperand(i) == CI)
      return false;
  return true;
}


bool TailCallElim::ProcessReturningBlock(ReturnInst *Ret, BasicBlock *&OldEntry,
                                         std::vector<PHINode*> &ArgumentPHIs) {
  BasicBlock *BB = Ret->getParent();
  Function *F = BB->getParent();

  if (&BB->front() == Ret) // Make sure there is something before the ret...
    return false;

  // Scan backwards from the return, checking to see if there is a tail call in
  // this block.  If so, set CI to it.
  CallInst *CI;
  BasicBlock::iterator BBI = Ret;
  while (1) {
    CI = dyn_cast<CallInst>(BBI);
    if (CI && CI->getCalledFunction() == F)
      break;

    if (BBI == BB->begin())
      return false;          // Didn't find a potential tail call.
    --BBI;
  }

  // Ok, we found a potential tail call.  We can currently only transform the
  // tail call if all of the instructions between the call and the return are
  // movable to above the call itself, leaving the call next to the return.
  // Check that this is the case now.
  for (BBI = CI, ++BBI; &*BBI != Ret; ++BBI)
    if (!CanMoveAboveCall(BBI, CI))
      return false;   // Cannot move this instruction out of the way.

  // We can only transform call/return pairs that either ignore the return value
  // of the call and return void, or return the value returned by the tail call.
  if (Ret->getNumOperands() != 0 && Ret->getReturnValue() != CI)
    return false;

  // OK! We can transform this tail call.  If this is the first one found,
  // create the new entry block, allowing us to branch back to the old entry.
  if (OldEntry == 0) {
    OldEntry = &F->getEntryBlock();
    std::string OldName = OldEntry->getName(); OldEntry->setName("tailrecurse");
    BasicBlock *NewEntry = new BasicBlock(OldName, OldEntry);
    new BranchInst(OldEntry, NewEntry);
    
    // Now that we have created a new block, which jumps to the entry
    // block, insert a PHI node for each argument of the function.
    // For now, we initialize each PHI to only have the real arguments
    // which are passed in.
    Instruction *InsertPos = OldEntry->begin();
    for (Function::aiterator I = F->abegin(), E = F->aend(); I != E; ++I) {
      PHINode *PN = new PHINode(I->getType(), I->getName()+".tr", InsertPos);
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
  new BranchInst(OldEntry, Ret);
  BB->getInstList().erase(Ret);  // Remove return.
  BB->getInstList().erase(CI);   // Remove call.
  NumEliminated++;
  return true;
}
