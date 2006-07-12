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
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Support/CallSite.h"
using namespace llvm;

bool llvm::InlineFunction(CallInst *CI, CallGraph *CG) {
  return InlineFunction(CallSite(CI), CG);
}
bool llvm::InlineFunction(InvokeInst *II, CallGraph *CG) {
  return InlineFunction(CallSite(II), CG);
}

/// HandleInlinedInvoke - If we inlined an invoke site, we need to convert calls
/// in the body of the inlined function into invokes and turn unwind
/// instructions into branches to the invoke unwind dest.
///
/// II is the invoke instruction begin inlined.  FirstNewBlock is the first
/// block of the inlined code (the last block is the end of the function),
/// and InlineCodeInfo is information about the code that got inlined.
static void HandleInlinedInvoke(InvokeInst *II, BasicBlock *FirstNewBlock,
                                ClonedCodeInfo &InlinedCodeInfo) {
  BasicBlock *InvokeDest = II->getUnwindDest();
  std::vector<Value*> InvokeDestPHIValues;

  // If there are PHI nodes in the unwind destination block, we need to
  // keep track of which values came into them from this invoke, then remove
  // the entry for this block.
  BasicBlock *InvokeBlock = II->getParent();
  for (BasicBlock::iterator I = InvokeDest->begin(); isa<PHINode>(I); ++I) {
    PHINode *PN = cast<PHINode>(I);
    // Save the value to use for this edge.
    InvokeDestPHIValues.push_back(PN->getIncomingValueForBlock(InvokeBlock));
  }

  Function *Caller = FirstNewBlock->getParent();
  
  // The inlined code is currently at the end of the function, scan from the
  // start of the inlined code to its end, checking for stuff we need to
  // rewrite.
  if (InlinedCodeInfo.ContainsCalls || InlinedCodeInfo.ContainsUnwinds) {
    for (Function::iterator BB = FirstNewBlock, E = Caller->end();
         BB != E; ++BB) {
      if (InlinedCodeInfo.ContainsCalls) {
        for (BasicBlock::iterator BBI = BB->begin(), E = BB->end(); BBI != E; ){
          Instruction *I = BBI++;
          
          // We only need to check for function calls: inlined invoke
          // instructions require no special handling.
          if (!isa<CallInst>(I)) continue;
          CallInst *CI = cast<CallInst>(I);

          // If this is an intrinsic function call, don't convert it to an
          // invoke.
          if (CI->getCalledFunction() &&
              CI->getCalledFunction()->getIntrinsicID())
            continue;
          
          // Convert this function call into an invoke instruction.
          // First, split the basic block.
          BasicBlock *Split = BB->splitBasicBlock(CI, CI->getName()+".noexc");
          
          // Next, create the new invoke instruction, inserting it at the end
          // of the old basic block.
          InvokeInst *II =
            new InvokeInst(CI->getCalledValue(), Split, InvokeDest,
                           std::vector<Value*>(CI->op_begin()+1, CI->op_end()),
                           CI->getName(), BB->getTerminator());
          II->setCallingConv(CI->getCallingConv());
          
          // Make sure that anything using the call now uses the invoke!
          CI->replaceAllUsesWith(II);
          
          // Delete the unconditional branch inserted by splitBasicBlock
          BB->getInstList().pop_back();
          Split->getInstList().pop_front();  // Delete the original call
          
          // Update any PHI nodes in the exceptional block to indicate that
          // there is now a new entry in them.
          unsigned i = 0;
          for (BasicBlock::iterator I = InvokeDest->begin();
               isa<PHINode>(I); ++I, ++i) {
            PHINode *PN = cast<PHINode>(I);
            PN->addIncoming(InvokeDestPHIValues[i], BB);
          }
            
          // This basic block is now complete, start scanning the next one.
          break;
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
             isa<PHINode>(I); ++I, ++i) {
          PHINode *PN = cast<PHINode>(I);
          PN->addIncoming(InvokeDestPHIValues[i], BB);
        }
      }
    }
  }

  // Now that everything is happy, we have one final detail.  The PHI nodes in
  // the exception destination block still have entries due to the original
  // invoke instruction.  Eliminate these entries (which might even delete the
  // PHI node) now.
  InvokeDest->removePredecessor(II->getParent());
}

/// UpdateCallGraphAfterInlining - Once we have cloned code over from a callee
/// into the caller, update the specified callgraph to reflect the changes we
/// made.  Note that it's possible that not all code was copied over, so only
/// some edges of the callgraph will be remain.
static void UpdateCallGraphAfterInlining(const Function *Caller,
                                         const Function *Callee,
                                         Function::iterator FirstNewBlock,
                                       std::map<const Value*, Value*> &ValueMap,
                                         CallGraph &CG) {
  // Update the call graph by deleting the edge from Callee to Caller
  CallGraphNode *CalleeNode = CG[Callee];
  CallGraphNode *CallerNode = CG[Caller];
  CallerNode->removeCallEdgeTo(CalleeNode);
  
  // Since we inlined some uninlined call sites in the callee into the caller,
  // add edges from the caller to all of the callees of the callee.
  for (CallGraphNode::iterator I = CalleeNode->begin(),
       E = CalleeNode->end(); I != E; ++I) {
    const Instruction *OrigCall = I->first.getInstruction();
    
    std::map<const Value*, Value*>::iterator VMI = ValueMap.find(OrigCall);
    if (VMI != ValueMap.end()) { // Only copy the edge if the call was inlined!
      // If the call was inlined, but then constant folded, there is no edge to
      // add.  Check for this case.
      if (Instruction *NewCall = dyn_cast<Instruction>(VMI->second))
        CallerNode->addCalledFunction(CallSite::get(NewCall), I->second);
    }
  }
}


// InlineFunction - This function inlines the called function into the basic
// block of the caller.  This returns false if it is not possible to inline this
// call.  The program is still in a well defined state if this occurs though.
//
// Note that this only does one level of inlining.  For example, if the
// instruction 'call B' is inlined, and 'B' calls 'C', then the call to 'C' now
// exists in the instruction stream.  Similiarly this will inline a recursive
// function by one level.
//
bool llvm::InlineFunction(CallSite CS, CallGraph *CG) {
  Instruction *TheCall = CS.getInstruction();
  assert(TheCall->getParent() && TheCall->getParent()->getParent() &&
         "Instruction not in function!");

  const Function *CalledFunc = CS.getCalledFunction();
  if (CalledFunc == 0 ||          // Can't inline external function or indirect
      CalledFunc->isExternal() || // call, or call to a vararg function!
      CalledFunc->getFunctionType()->isVarArg()) return false;


  // If the call to the callee is a non-tail call, we must clear the 'tail'
  // flags on any calls that we inline.
  bool MustClearTailCallFlags =
    isa<CallInst>(TheCall) && !cast<CallInst>(TheCall)->isTailCall();

  BasicBlock *OrigBB = TheCall->getParent();
  Function *Caller = OrigBB->getParent();

  // Get an iterator to the last basic block in the function, which will have
  // the new function inlined after it.
  //
  Function::iterator LastBlock = &Caller->back();

  // Make sure to capture all of the return instructions from the cloned
  // function.
  std::vector<ReturnInst*> Returns;
  ClonedCodeInfo InlinedFunctionInfo;
  Function::iterator FirstNewBlock;
  
  { // Scope to destroy ValueMap after cloning.
    std::map<const Value*, Value*> ValueMap;

    // Calculate the vector of arguments to pass into the function cloner, which
    // matches up the formal to the actual argument values.
    assert(std::distance(CalledFunc->arg_begin(), CalledFunc->arg_end()) ==
           std::distance(CS.arg_begin(), CS.arg_end()) &&
           "No varargs calls can be inlined!");
    CallSite::arg_iterator AI = CS.arg_begin();
    for (Function::const_arg_iterator I = CalledFunc->arg_begin(),
           E = CalledFunc->arg_end(); I != E; ++I, ++AI)
      ValueMap[I] = *AI;

    // We want the inliner to prune the code as it copies.  We would LOVE to
    // have no dead or constant instructions leftover after inlining occurs
    // (which can happen, e.g., because an argument was constant), but we'll be
    // happy with whatever the cloner can do.
    CloneAndPruneFunctionInto(Caller, CalledFunc, ValueMap, Returns, ".i",
                              &InlinedFunctionInfo);
    
    // Remember the first block that is newly cloned over.
    FirstNewBlock = LastBlock; ++FirstNewBlock;
    
    // Update the callgraph if requested.
    if (CG)
      UpdateCallGraphAfterInlining(Caller, CalledFunc, FirstNewBlock, ValueMap,
                                   *CG);
  }
 
  // If there are any alloca instructions in the block that used to be the entry
  // block for the callee, move them to the entry block of the caller.  First
  // calculate which instruction they should be inserted before.  We insert the
  // instructions at the end of the current alloca list.
  //
  {
    BasicBlock::iterator InsertPoint = Caller->begin()->begin();
    for (BasicBlock::iterator I = FirstNewBlock->begin(),
           E = FirstNewBlock->end(); I != E; )
      if (AllocaInst *AI = dyn_cast<AllocaInst>(I++))
        if (isa<Constant>(AI->getArraySize())) {
          // Scan for the block of allocas that we can move over, and move them
          // all at once.
          while (isa<AllocaInst>(I) &&
                 isa<Constant>(cast<AllocaInst>(I)->getArraySize()))
            ++I;

          // Transfer all of the allocas over in a block.  Using splice means
          // that they instructions aren't removed from the symbol table, then
          // reinserted.
          Caller->front().getInstList().splice(InsertPoint,
                                               FirstNewBlock->getInstList(),
                                               AI, I);
        }
  }

  // If the inlined code contained dynamic alloca instructions, wrap the inlined
  // code with llvm.stacksave/llvm.stackrestore intrinsics.
  if (InlinedFunctionInfo.ContainsDynamicAllocas) {
    Module *M = Caller->getParent();
    const Type *SBytePtr = PointerType::get(Type::SByteTy);
    // Get the two intrinsics we care about.
    Function *StackSave, *StackRestore;
    StackSave    = M->getOrInsertFunction("llvm.stacksave", SBytePtr, NULL);
    StackRestore = M->getOrInsertFunction("llvm.stackrestore", Type::VoidTy,
                                          SBytePtr, NULL);

    // If we are preserving the callgraph, add edges to the stacksave/restore
    // functions for the calls we insert.
    CallGraphNode *StackSaveCGN, *StackRestoreCGN, *CallerNode;
    if (CG) {
      StackSaveCGN    = CG->getOrInsertFunction(StackSave);
      StackRestoreCGN = CG->getOrInsertFunction(StackRestore);
      CallerNode = (*CG)[Caller];
    }
      
    // Insert the llvm.stacksave.
    CallInst *SavedPtr = new CallInst(StackSave, "savedstack", 
                                      FirstNewBlock->begin());
    if (CG) CallerNode->addCalledFunction(SavedPtr, StackSaveCGN);
      
    // Insert a call to llvm.stackrestore before any return instructions in the
    // inlined function.
    for (unsigned i = 0, e = Returns.size(); i != e; ++i) {
      CallInst *CI = new CallInst(StackRestore, SavedPtr, "", Returns[i]);
      if (CG) CallerNode->addCalledFunction(CI, StackRestoreCGN);
    }

    // Count the number of StackRestore calls we insert.
    unsigned NumStackRestores = Returns.size();
    
    // If we are inlining an invoke instruction, insert restores before each
    // unwind.  These unwinds will be rewritten into branches later.
    if (InlinedFunctionInfo.ContainsUnwinds && isa<InvokeInst>(TheCall)) {
      for (Function::iterator BB = FirstNewBlock, E = Caller->end();
           BB != E; ++BB)
        if (UnwindInst *UI = dyn_cast<UnwindInst>(BB->getTerminator())) {
          new CallInst(StackRestore, SavedPtr, "", UI);
          ++NumStackRestores;
        }
    }
  }

  // If we are inlining tail call instruction through a call site that isn't 
  // marked 'tail', we must remove the tail marker for any calls in the inlined
  // code.
  if (MustClearTailCallFlags && InlinedFunctionInfo.ContainsCalls) {
    for (Function::iterator BB = FirstNewBlock, E = Caller->end();
         BB != E; ++BB)
      for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
        if (CallInst *CI = dyn_cast<CallInst>(I))
          CI->setTailCall(false);
  }

  // If we are inlining for an invoke instruction, we must make sure to rewrite
  // any inlined 'unwind' instructions into branches to the invoke exception
  // destination, and call instructions into invoke instructions.
  if (InvokeInst *II = dyn_cast<InvokeInst>(TheCall))
    HandleInlinedInvoke(II, FirstNewBlock, InlinedFunctionInfo);

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
                                          CalledFunc->getName()+".exit");

  } else {  // It's a call
    // If this is a call instruction, we need to split the basic block that
    // the call lives in.
    //
    AfterCallBB = OrigBB->splitBasicBlock(TheCall,
                                          CalledFunc->getName()+".exit");
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

    // Splice the code from the return block into the block that it will return
    // to, which contains the code that was after the call.
    BasicBlock *ReturnBB = Returns[0]->getParent();
    AfterCallBB->getInstList().splice(AfterCallBB->begin(),
                                      ReturnBB->getInstList());

    // Update PHI nodes that use the ReturnBB to use the AfterCallBB.
    ReturnBB->replaceAllUsesWith(AfterCallBB);

    // Delete the return instruction now and empty ReturnBB now.
    Returns[0]->eraseFromParent();
    ReturnBB->eraseFromParent();
  } else if (!TheCall->use_empty()) {
    // No returns, but something is using the return value of the call.  Just
    // nuke the result.
    TheCall->replaceAllUsesWith(UndefValue::get(TheCall->getType()));
  }

  // Since we are now done with the Call/Invoke, we can delete it.
  TheCall->eraseFromParent();

  // We should always be able to fold the entry block of the function into the
  // single predecessor of the block...
  assert(cast<BranchInst>(Br)->isUnconditional() && "splitBasicBlock broken!");
  BasicBlock *CalleeEntry = cast<BranchInst>(Br)->getSuccessor(0);

  // Splice the code entry block into calling block, right before the
  // unconditional branch.
  OrigBB->getInstList().splice(Br, CalleeEntry->getInstList());
  CalleeEntry->replaceAllUsesWith(OrigBB);  // Update PHI nodes

  // Remove the unconditional branch.
  OrigBB->getInstList().erase(Br);

  // Now we can remove the CalleeEntry block, which is now empty.
  Caller->getBasicBlockList().erase(CalleeEntry);
  
  return true;
}
