//===- InlineFunction.cpp - Code to perform function inlining -------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements inlining of a function into a call site, resolving
// parameters and the return value as appropriate.
//
// FIXME: This pass should transform alloca instructions in the called function
// into alloca/dealloca pairs!  Or perhaps it should refuse to inline them!
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Constant.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Transforms/Utils/Local.h"
using namespace llvm;

bool llvm::InlineFunction(CallInst *CI) { return InlineFunction(CallSite(CI)); }
bool llvm::InlineFunction(InvokeInst *II) {return InlineFunction(CallSite(II));}

// InlineFunction - This function inlines the called function into the basic
// block of the caller.  This returns false if it is not possible to inline this
// call.  The program is still in a well defined state if this occurs though.
//
// Note that this only does one level of inlining.  For example, if the 
// instruction 'call B' is inlined, and 'B' calls 'C', then the call to 'C' now 
// exists in the instruction stream.  Similiarly this will inline a recursive
// function by one level.
//
bool llvm::InlineFunction(CallSite CS) {
  Instruction *TheCall = CS.getInstruction();
  assert(TheCall->getParent() && TheCall->getParent()->getParent() &&
         "Instruction not in function!");

  const Function *CalledFunc = CS.getCalledFunction();
  if (CalledFunc == 0 ||          // Can't inline external function or indirect
      CalledFunc->isExternal() || // call, or call to a vararg function!
      CalledFunc->getFunctionType()->isVarArg()) return false;

  BasicBlock *OrigBB = TheCall->getParent();
  Function *Caller = OrigBB->getParent();

  // Get an iterator to the last basic block in the function, which will have
  // the new function inlined after it.
  //
  Function::iterator LastBlock = &Caller->back();

  // Make sure to capture all of the return instructions from the cloned
  // function.
  std::vector<ReturnInst*> Returns;
  { // Scope to destroy ValueMap after cloning.
    // Calculate the vector of arguments to pass into the function cloner...
    std::map<const Value*, Value*> ValueMap;
    assert(std::distance(CalledFunc->abegin(), CalledFunc->aend()) == 
           std::distance(CS.arg_begin(), CS.arg_end()) &&
           "No varargs calls can be inlined!");
    
    CallSite::arg_iterator AI = CS.arg_begin();
    for (Function::const_aiterator I = CalledFunc->abegin(),
           E = CalledFunc->aend(); I != E; ++I, ++AI)
      ValueMap[I] = *AI;
    
    // Clone the entire body of the callee into the caller.  
    CloneFunctionInto(Caller, CalledFunc, ValueMap, Returns, ".i");
  }    
  
  // Remember the first block that is newly cloned over.
  Function::iterator FirstNewBlock = LastBlock; ++FirstNewBlock;

  // If there are any alloca instructions in the block that used to be the entry
  // block for the callee, move them to the entry block of the caller.  First
  // calculate which instruction they should be inserted before.  We insert the
  // instructions at the end of the current alloca list.
  //
  if (isa<AllocaInst>(FirstNewBlock->begin())) {
    BasicBlock::iterator InsertPoint = Caller->begin()->begin();
    while (isa<AllocaInst>(InsertPoint)) ++InsertPoint;
    
    for (BasicBlock::iterator I = FirstNewBlock->begin(),
           E = FirstNewBlock->end(); I != E; )
      if (AllocaInst *AI = dyn_cast<AllocaInst>(I++))
        if (isa<Constant>(AI->getArraySize())) {
          FirstNewBlock->getInstList().remove(AI);
          Caller->front().getInstList().insert(InsertPoint, AI);      
        }
  }

  // If we are inlining for an invoke instruction, we must make sure to rewrite
  // any inlined 'unwind' instructions into branches to the invoke exception
  // destination, and call instructions into invoke instructions.
  if (InvokeInst *II = dyn_cast<InvokeInst>(TheCall)) {
    BasicBlock *InvokeDest = II->getExceptionalDest();
    std::vector<Value*> InvokeDestPHIValues;

    // If there are PHI nodes in the exceptional destination block, we need to
    // keep track of which values came into them from this invoke, then remove
    // the entry for this block.
    for (BasicBlock::iterator I = InvokeDest->begin();
         PHINode *PN = dyn_cast<PHINode>(I); ++I)
      // Save the value to use for this edge...
      InvokeDestPHIValues.push_back(PN->getIncomingValueForBlock(OrigBB));

    for (Function::iterator BB = FirstNewBlock, E = Caller->end();
         BB != E; ++BB) {
      for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ) {
        // We only need to check for function calls: inlined invoke instructions
        // require no special handling...
        if (CallInst *CI = dyn_cast<CallInst>(I)) {
          // Convert this function call into an invoke instruction...

          // First, split the basic block...
          BasicBlock *Split = BB->splitBasicBlock(CI, CI->getName()+".noexc");
          
          // Next, create the new invoke instruction, inserting it at the end
          // of the old basic block.
          InvokeInst *II =
            new InvokeInst(CI->getCalledValue(), Split, InvokeDest, 
                           std::vector<Value*>(CI->op_begin()+1, CI->op_end()),
                           CI->getName(), BB->getTerminator());

          // Make sure that anything using the call now uses the invoke!
          CI->replaceAllUsesWith(II);

          // Delete the unconditional branch inserted by splitBasicBlock
          BB->getInstList().pop_back();
          Split->getInstList().pop_front();  // Delete the original call
          
          // Update any PHI nodes in the exceptional block to indicate that
          // there is now a new entry in them.
          unsigned i = 0;
          for (BasicBlock::iterator I = InvokeDest->begin();
               PHINode *PN = dyn_cast<PHINode>(I); ++I, ++i)
            PN->addIncoming(InvokeDestPHIValues[i], BB);

          // This basic block is now complete, start scanning the next one.
          break;
        } else {
          ++I;
        }
      }

      if (UnwindInst *UI = dyn_cast<UnwindInst>(BB->getTerminator())) {
        // An UnwindInst requires special handling when it gets inlined into an
        // invoke site.  Once this happens, we know that the unwind would cause
        // a control transfer to the invoke exception destination, so we can
        // transform it into a direct branch to the exception destination.
        new BranchInst(InvokeDest, UI);

        // Delete the unwind instruction!
        UI->getParent()->getInstList().pop_back();

        // Update any PHI nodes in the exceptional block to indicate that
        // there is now a new entry in them.
        unsigned i = 0;
        for (BasicBlock::iterator I = InvokeDest->begin();
             PHINode *PN = dyn_cast<PHINode>(I); ++I, ++i)
          PN->addIncoming(InvokeDestPHIValues[i], BB);
      }
    }

    // Now that everything is happy, we have one final detail.  The PHI nodes in
    // the exception destination block still have entries due to the original
    // invoke instruction.  Eliminate these entries (which might even delete the
    // PHI node) now.
    InvokeDest->removePredecessor(II->getParent());
  }

  // If we cloned in _exactly one_ basic block, and if that block ends in a
  // return instruction, we splice the body of the inlined callee directly into
  // the calling basic block.
  if (Returns.size() == 1 && std::distance(FirstNewBlock, Caller->end()) == 1) {
    // Move all of the instructions right before the call.
    OrigBB->getInstList().splice(TheCall, FirstNewBlock->getInstList(),
                                 FirstNewBlock->begin(), FirstNewBlock->end());
    // Remove the cloned basic block.
    Caller->getBasicBlockList().pop_back();
    
    // If the call site was an invoke instruction, add a branch to the normal
    // destination.
    if (InvokeInst *II = dyn_cast<InvokeInst>(TheCall))
      new BranchInst(II->getNormalDest(), TheCall);

    // If the return instruction returned a value, replace uses of the call with
    // uses of the returned value.
    if (!TheCall->use_empty())
      TheCall->replaceAllUsesWith(Returns[0]->getReturnValue());

    // Since we are now done with the Call/Invoke, we can delete it.
    TheCall->getParent()->getInstList().erase(TheCall);

    // Since we are now done with the return instruction, delete it also.
    Returns[0]->getParent()->getInstList().erase(Returns[0]);

    // We are now done with the inlining.
    return true;
  }

  // Otherwise, we have the normal case, of more than one block to inline or
  // multiple return sites.

  // We want to clone the entire callee function into the hole between the
  // "starter" and "ender" blocks.  How we accomplish this depends on whether
  // this is an invoke instruction or a call instruction.
  BasicBlock *AfterCallBB;
  if (InvokeInst *II = dyn_cast<InvokeInst>(TheCall)) {
    
    // Add an unconditional branch to make this look like the CallInst case...
    BranchInst *NewBr = new BranchInst(II->getNormalDest(), TheCall);
    
    // Split the basic block.  This guarantees that no PHI nodes will have to be
    // updated due to new incoming edges, and make the invoke case more
    // symmetric to the call case.
    AfterCallBB = OrigBB->splitBasicBlock(NewBr,
                                          CalledFunc->getName()+".entry");
    
  } else {  // It's a call
    // If this is a call instruction, we need to split the basic block that
    // the call lives in.
    //
    AfterCallBB = OrigBB->splitBasicBlock(TheCall,
                                          CalledFunc->getName()+".entry");
  }

  // Change the branch that used to go to AfterCallBB to branch to the first
  // basic block of the inlined function.
  //
  TerminatorInst *Br = OrigBB->getTerminator();
  assert(Br && Br->getOpcode() == Instruction::Br && 
         "splitBasicBlock broken!");
  Br->setOperand(0, FirstNewBlock);


  // Now that the function is correct, make it a little bit nicer.  In
  // particular, move the basic blocks inserted from the end of the function
  // into the space made by splitting the source basic block.
  //
  Caller->getBasicBlockList().splice(AfterCallBB, Caller->getBasicBlockList(),
                                     FirstNewBlock, Caller->end());

  // Handle all of the return instructions that we just cloned in, and eliminate
  // any users of the original call/invoke instruction.
  if (Returns.size() > 1) {
    // The PHI node should go at the front of the new basic block to merge all
    // possible incoming values.
    //
    PHINode *PHI = 0;
    if (!TheCall->use_empty()) {
      PHI = new PHINode(CalledFunc->getReturnType(),
                        TheCall->getName(), AfterCallBB->begin());
        
      // Anything that used the result of the function call should now use the
      // PHI node as their operand.
      //
      TheCall->replaceAllUsesWith(PHI);
    }
      
    // Loop over all of the return instructions, turning them into unconditional
    // branches to the merge point now, and adding entries to the PHI node as
    // appropriate.
    for (unsigned i = 0, e = Returns.size(); i != e; ++i) {
      ReturnInst *RI = Returns[i];
        
      if (PHI) {
        assert(RI->getReturnValue() && "Ret should have value!");
        assert(RI->getReturnValue()->getType() == PHI->getType() && 
               "Ret value not consistent in function!");
        PHI->addIncoming(RI->getReturnValue(), RI->getParent());
      }
        
      // Add a branch to the merge point where the PHI node lives if it exists.
      new BranchInst(AfterCallBB, RI);
        
      // Delete the return instruction now
      RI->getParent()->getInstList().erase(RI);
    }
      
  } else if (!Returns.empty()) {
    // Otherwise, if there is exactly one return value, just replace anything
    // using the return value of the call with the computed value.
    if (!TheCall->use_empty())
      TheCall->replaceAllUsesWith(Returns[0]->getReturnValue());
      
    // Add a branch to the merge point where the PHI node lives if it exists.
    new BranchInst(AfterCallBB, Returns[0]);
      
    // Delete the return instruction now
    Returns[0]->getParent()->getInstList().erase(Returns[0]);
  }
    
  // Since we are now done with the Call/Invoke, we can delete it.
  TheCall->getParent()->getInstList().erase(TheCall);

  // We should always be able to fold the entry block of the function into the
  // single predecessor of the block...
  assert(cast<BranchInst>(Br)->isUnconditional() &&"splitBasicBlock broken!");
  BasicBlock *CalleeEntry = cast<BranchInst>(Br)->getSuccessor(0);
  SimplifyCFG(CalleeEntry);
    
  // Okay, continue the CFG cleanup.  It's often the case that there is only a
  // single return instruction in the callee function.  If this is the case,
  // then we have an unconditional branch from the return block to the
  // 'AfterCallBB'.  Check for this case, and eliminate the branch is possible.
  SimplifyCFG(AfterCallBB);

  return true;
}
