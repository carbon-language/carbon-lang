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
//  2. This pass transforms functions that are prevented from being tail
//     recursive by an associative expression to use an accumulator variable,
//     thus compiling the typical naive factorial or 'fib' implementation into
//     efficient code.
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
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/CFG.h"
#include "Support/Statistic.h"
using namespace llvm;

namespace {
  Statistic<> NumEliminated("tailcallelim", "Number of tail calls removed");
  Statistic<> NumAccumAdded("tailcallelim","Number of accumulators introduced");

  struct TailCallElim : public FunctionPass {
    virtual bool runOnFunction(Function &F);

  private:
    bool ProcessReturningBlock(ReturnInst *RI, BasicBlock *&OldEntry,
                               std::vector<PHINode*> &ArgumentPHIs);
    bool CanMoveAboveCall(Instruction *I, CallInst *CI);
    Value *CanTransformAccumulatorRecursion(Instruction *I, CallInst *CI);
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
  
  // If we eliminated any tail recursions, it's possible that we inserted some
  // silly PHI nodes which just merge an initial value (the incoming operand)
  // with themselves.  Check to see if we did and clean up our mess if so.  This
  // occurs when a function passes an argument straight through to its tail
  // call.
  if (!ArgumentPHIs.empty()) {
    unsigned NumIncoming = ArgumentPHIs[0]->getNumIncomingValues();
    for (unsigned i = 0, e = ArgumentPHIs.size(); i != e; ++i) {
      PHINode *PN = ArgumentPHIs[i];
      Value *V = 0;
      for (unsigned op = 0, e = NumIncoming; op != e; ++op) {
        Value *Op = PN->getIncomingValue(op);
        if (Op != PN) {
          if (V == 0) {
            V = Op;     // First value seen?
          } else if (V != Op) {
            V = 0;
            break;
          }
        }
      }

      // If the PHI Node is a dynamic constant, replace it with the value it is.
      if (V) {
        PN->replaceAllUsesWith(V);
        PN->getParent()->getInstList().erase(PN);
      }
    }
  }

  return MadeChange;
}


/// CanMoveAboveCall - Return true if it is safe to move the specified
/// instruction from after the call to before the call, assuming that all
/// instructions between the call and this instruction are movable.
///
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


/// CanTransformAccumulatorRecursion - If the specified instruction can be
/// transformed using accumulator recursion elimination, return the constant
/// which is the start of the accumulator value.  Otherwise return null.
///
Value *TailCallElim::CanTransformAccumulatorRecursion(Instruction *I,
                                                      CallInst *CI) {
  if (!I->isAssociative()) return 0;
  assert(I->getNumOperands() == 2 &&
         "Associative operations should have 2 args!");

  // Exactly one operand should be the result of the call instruction...
  if (I->getOperand(0) == CI && I->getOperand(1) == CI ||
      I->getOperand(0) != CI && I->getOperand(1) != CI)
    return 0;

  // The only user of this instruction we allow is a single return instruction.
  if (!I->hasOneUse() || !isa<ReturnInst>(I->use_back()))
    return 0;

  // Ok, now we have to check all of the other return instructions in this
  // function.  If they return non-constants or differing values, then we cannot
  // transform the function safely.
  Value *ReturnedValue = 0;
  Function *F = CI->getParent()->getParent();

  for (Function::iterator BBI = F->begin(), E = F->end(); BBI != E; ++BBI)
    if (ReturnInst *RI = dyn_cast<ReturnInst>(BBI->getTerminator())) {
      Value *RetOp = RI->getOperand(0);
      if (RetOp != I) { // Ignore the one returning I.
        // We can only perform this transformation if the value returned is
        // evaluatable at the start of the initial invocation of the function,
        // instead of at the end of the evaluation.
        //
        // We currently handle static constants and arguments that are not
        // modified as part of the recursion.
        if (!isa<Constant>(RetOp)) {    // Constants are always ok
          // Check to see if this is an immutable argument, if so, the value
          // will be available to initialize the accumulator.
          if (Argument *Arg = dyn_cast<Argument>(RetOp)) {
            // Figure out which argument number this is...
            unsigned ArgNo = 0;
            for (Function::aiterator AI = F->abegin(); &*AI != Arg; ++AI)
              ++ArgNo;
            
            // If we are passing this argument into call as the corresponding
            // argument operand, then the argument is dynamically constant.
            // Otherwise, we cannot transform this function safely.
            if (CI->getOperand(ArgNo+1) != Arg)
              return 0;

          } else {
            // Not a constant or immutable argument, we can't safely transform.
            return 0;
          }
        }
        
        if (ReturnedValue && RetOp != ReturnedValue)
          return 0;     // Cannot transform if differing values are returned.
        ReturnedValue = RetOp;
      }
    }
  
  // Ok, if we passed this battery of tests, we can perform accumulator
  // recursion elimination.
  return ReturnedValue;
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

  // If we are introducing accumulator recursion to eliminate associative
  // operations after the call instruction, this variable contains the initial
  // value for the accumulator.  If this value is set, we actually perform
  // accumulator recursion elimination instead of simple tail recursion
  // elimination.
  Value *AccumulatorRecursionEliminationInitVal = 0;
  Instruction *AccumulatorRecursionInstr = 0;

  // Ok, we found a potential tail call.  We can currently only transform the
  // tail call if all of the instructions between the call and the return are
  // movable to above the call itself, leaving the call next to the return.
  // Check that this is the case now.
  for (BBI = CI, ++BBI; &*BBI != Ret; ++BBI)
    if (!CanMoveAboveCall(BBI, CI)) {
      // If we can't move the instruction above the call, it might be because it
      // is an associative operation that could be tranformed using accumulator
      // recursion elimination.  Check to see if this is the case, and if so,
      // remember the initial accumulator value for later.
      if ((AccumulatorRecursionEliminationInitVal =
                             CanTransformAccumulatorRecursion(BBI, CI))) {
        // Yes, this is accumulator recursion.  Remember which instruction
        // accumulates.
        AccumulatorRecursionInstr = BBI;
      } else {
        return false;   // Otherwise, we cannot eliminate the tail recursion!
      }
    }

  // We can only transform call/return pairs that either ignore the return value
  // of the call and return void, or return the value returned by the tail call.
  if (Ret->getNumOperands() != 0 && Ret->getReturnValue() != CI &&
      AccumulatorRecursionEliminationInitVal == 0)
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
  
  // If we are introducing an accumulator variable to eliminate the recursion,
  // do so now.  Note that we _know_ that no subsequent tail recursion
  // eliminations will happen on this function because of the way the
  // accumulator recursion predicate is set up.
  //
  if (AccumulatorRecursionEliminationInitVal) {
    Instruction *AccRecInstr = AccumulatorRecursionInstr;
    // Start by inserting a new PHI node for the accumulator.
    PHINode *AccPN = new PHINode(AccRecInstr->getType(), "accumulator.tr",
                                 OldEntry->begin());

    // Loop over all of the predecessors of the tail recursion block.  For the
    // real entry into the function we seed the PHI with the initial value,
    // computed earlier.  For any other existing branches to this block (due to
    // other tail recursions eliminated) the accumulator is not modified.
    // Because we haven't added the branch in the current block to OldEntry yet,
    // it will not show up as a predecessor.
    for (pred_iterator PI = pred_begin(OldEntry), PE = pred_end(OldEntry);
         PI != PE; ++PI) {
      if (*PI == &F->getEntryBlock())
        AccPN->addIncoming(AccumulatorRecursionEliminationInitVal, *PI);
      else
        AccPN->addIncoming(AccPN, *PI);
    }

    // Add an incoming argument for the current block, which is computed by our
    // associative accumulator instruction.
    AccPN->addIncoming(AccRecInstr, BB);

    // Next, rewrite the accumulator recursion instruction so that it does not
    // use the result of the call anymore, instead, use the PHI node we just
    // inserted.
    AccRecInstr->setOperand(AccRecInstr->getOperand(0) != CI, AccPN);

    // Finally, rewrite any return instructions in the program to return the PHI
    // node instead of the "initval" that they do currently.  This loop will
    // actually rewrite the return value we are destroying, but that's ok.
    for (Function::iterator BBI = F->begin(), E = F->end(); BBI != E; ++BBI)
      if (ReturnInst *RI = dyn_cast<ReturnInst>(BBI->getTerminator()))
        RI->setOperand(0, AccPN);
    ++NumAccumAdded;
  }

  // Now that all of the PHI nodes are in place, remove the call and
  // ret instructions, replacing them with an unconditional branch.
  new BranchInst(OldEntry, Ret);
  BB->getInstList().erase(Ret);  // Remove return.
  BB->getInstList().erase(CI);   // Remove call.
  ++NumEliminated;
  return true;
}
