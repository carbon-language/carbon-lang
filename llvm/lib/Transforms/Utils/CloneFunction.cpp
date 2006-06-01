//===- CloneFunction.cpp - Clone a function into another function ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CloneFunctionInto interface, which is used as the
// low-level function cloner.  This is used by the CloneFunction and function
// inliner to do the dirty work of copying the body of a function around.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Function.h"
#include "llvm/Support/CFG.h"
#include "ValueMapper.h"
#include "llvm/Transforms/Utils/Local.h"
using namespace llvm;

// CloneBasicBlock - See comments in Cloning.h
BasicBlock *llvm::CloneBasicBlock(const BasicBlock *BB,
                                  std::map<const Value*, Value*> &ValueMap,
                                  const char *NameSuffix, Function *F,
                                  ClonedCodeInfo *CodeInfo) {
  BasicBlock *NewBB = new BasicBlock("", F);
  if (BB->hasName()) NewBB->setName(BB->getName()+NameSuffix);

  bool hasCalls = false, hasDynamicAllocas = false, hasStaticAllocas = false;
  
  // Loop over all instructions, and copy them over.
  for (BasicBlock::const_iterator II = BB->begin(), IE = BB->end();
       II != IE; ++II) {
    Instruction *NewInst = II->clone();
    if (II->hasName())
      NewInst->setName(II->getName()+NameSuffix);
    NewBB->getInstList().push_back(NewInst);
    ValueMap[II] = NewInst;                // Add instruction map to value.
    
    hasCalls |= isa<CallInst>(II);
    if (const AllocaInst *AI = dyn_cast<AllocaInst>(II)) {
      if (isa<ConstantInt>(AI->getArraySize()))
        hasStaticAllocas = true;
      else
        hasDynamicAllocas = true;
    }
  }
  
  if (CodeInfo) {
    CodeInfo->ContainsCalls          |= hasCalls;
    CodeInfo->ContainsUnwinds        |= isa<UnwindInst>(BB->getTerminator());
    CodeInfo->ContainsDynamicAllocas |= hasDynamicAllocas;
    CodeInfo->ContainsDynamicAllocas |= hasStaticAllocas && 
                                        BB != &BB->getParent()->front();
  }
  return NewBB;
}

// Clone OldFunc into NewFunc, transforming the old arguments into references to
// ArgMap values.
//
void llvm::CloneFunctionInto(Function *NewFunc, const Function *OldFunc,
                             std::map<const Value*, Value*> &ValueMap,
                             std::vector<ReturnInst*> &Returns,
                             const char *NameSuffix, ClonedCodeInfo *CodeInfo) {
  assert(NameSuffix && "NameSuffix cannot be null!");

#ifndef NDEBUG
  for (Function::const_arg_iterator I = OldFunc->arg_begin(), 
       E = OldFunc->arg_end(); I != E; ++I)
    assert(ValueMap.count(I) && "No mapping from source argument specified!");
#endif

  // Loop over all of the basic blocks in the function, cloning them as
  // appropriate.  Note that we save BE this way in order to handle cloning of
  // recursive functions into themselves.
  //
  for (Function::const_iterator BI = OldFunc->begin(), BE = OldFunc->end();
       BI != BE; ++BI) {
    const BasicBlock &BB = *BI;

    // Create a new basic block and copy instructions into it!
    BasicBlock *CBB = CloneBasicBlock(&BB, ValueMap, NameSuffix, NewFunc,
                                      CodeInfo);
    ValueMap[&BB] = CBB;                       // Add basic block mapping.

    if (ReturnInst *RI = dyn_cast<ReturnInst>(CBB->getTerminator()))
      Returns.push_back(RI);
  }

  // Loop over all of the instructions in the function, fixing up operand
  // references as we go.  This uses ValueMap to do all the hard work.
  //
  for (Function::iterator BB = cast<BasicBlock>(ValueMap[OldFunc->begin()]),
         BE = NewFunc->end(); BB != BE; ++BB)
    // Loop over all instructions, fixing each one as we find it...
    for (BasicBlock::iterator II = BB->begin(); II != BB->end(); ++II)
      RemapInstruction(II, ValueMap);
}

/// CloneFunction - Return a copy of the specified function, but without
/// embedding the function into another module.  Also, any references specified
/// in the ValueMap are changed to refer to their mapped value instead of the
/// original one.  If any of the arguments to the function are in the ValueMap,
/// the arguments are deleted from the resultant function.  The ValueMap is
/// updated to include mappings from all of the instructions and basicblocks in
/// the function from their old to new values.
///
Function *llvm::CloneFunction(const Function *F,
                              std::map<const Value*, Value*> &ValueMap,
                              ClonedCodeInfo *CodeInfo) {
  std::vector<const Type*> ArgTypes;

  // The user might be deleting arguments to the function by specifying them in
  // the ValueMap.  If so, we need to not add the arguments to the arg ty vector
  //
  for (Function::const_arg_iterator I = F->arg_begin(), E = F->arg_end();
       I != E; ++I)
    if (ValueMap.count(I) == 0)  // Haven't mapped the argument to anything yet?
      ArgTypes.push_back(I->getType());

  // Create a new function type...
  FunctionType *FTy = FunctionType::get(F->getFunctionType()->getReturnType(),
                                    ArgTypes, F->getFunctionType()->isVarArg());

  // Create the new function...
  Function *NewF = new Function(FTy, F->getLinkage(), F->getName());

  // Loop over the arguments, copying the names of the mapped arguments over...
  Function::arg_iterator DestI = NewF->arg_begin();
  for (Function::const_arg_iterator I = F->arg_begin(), E = F->arg_end();
       I != E; ++I)
    if (ValueMap.count(I) == 0) {   // Is this argument preserved?
      DestI->setName(I->getName()); // Copy the name over...
      ValueMap[I] = DestI++;        // Add mapping to ValueMap
    }

  std::vector<ReturnInst*> Returns;  // Ignore returns cloned...
  CloneFunctionInto(NewF, F, ValueMap, Returns, "", CodeInfo);
  return NewF;
}



namespace {
  /// PruningFunctionCloner - This class is a private class used to implement
  /// the CloneAndPruneFunctionInto method.
  struct PruningFunctionCloner {
    Function *NewFunc;
    const Function *OldFunc;
    std::map<const Value*, Value*> &ValueMap;
    std::vector<ReturnInst*> &Returns;
    const char *NameSuffix;
    ClonedCodeInfo *CodeInfo;

  public:
    PruningFunctionCloner(Function *newFunc, const Function *oldFunc,
                          std::map<const Value*, Value*> &valueMap,
                          std::vector<ReturnInst*> &returns,
                          const char *nameSuffix, 
                          ClonedCodeInfo *codeInfo)
    : NewFunc(newFunc), OldFunc(oldFunc), ValueMap(valueMap), Returns(returns),
      NameSuffix(nameSuffix), CodeInfo(codeInfo) {
    }

    /// CloneBlock - The specified block is found to be reachable, clone it and
    /// anything that it can reach.
    void CloneBlock(const BasicBlock *BB);
    
  public:
    /// ConstantFoldMappedInstruction - Constant fold the specified instruction,
    /// mapping its operands through ValueMap if they are available.
    Constant *ConstantFoldMappedInstruction(const Instruction *I);
  };
}

/// CloneBlock - The specified block is found to be reachable, clone it and
/// anything that it can reach.
void PruningFunctionCloner::CloneBlock(const BasicBlock *BB) {
  Value *&BBEntry = ValueMap[BB];

  // Have we already cloned this block?
  if (BBEntry) return;
  
  // Nope, clone it now.
  BasicBlock *NewBB;
  BBEntry = NewBB = new BasicBlock();
  if (BB->hasName()) NewBB->setName(BB->getName()+NameSuffix);

  bool hasCalls = false, hasDynamicAllocas = false, hasStaticAllocas = false;
  
  // Loop over all instructions, and copy them over, DCE'ing as we go.  This
  // loop doesn't include the terminator.
  for (BasicBlock::const_iterator II = BB->begin(), IE = --BB->end();
       II != IE; ++II) {
    // If this instruction constant folds, don't bother cloning the instruction,
    // instead, just add the constant to the value map.
    if (Constant *C = ConstantFoldMappedInstruction(II)) {
      ValueMap[II] = C;
      continue;
    }
    
    Instruction *NewInst = II->clone();
    if (II->hasName())
      NewInst->setName(II->getName()+NameSuffix);
    NewBB->getInstList().push_back(NewInst);
    ValueMap[II] = NewInst;                // Add instruction map to value.
    
    hasCalls |= isa<CallInst>(II);
    if (const AllocaInst *AI = dyn_cast<AllocaInst>(II)) {
      if (isa<ConstantInt>(AI->getArraySize()))
        hasStaticAllocas = true;
      else
        hasDynamicAllocas = true;
    }
  }
  
  // Finally, clone over the terminator.
  const TerminatorInst *OldTI = BB->getTerminator();
  bool TerminatorDone = false;
  if (const BranchInst *BI = dyn_cast<BranchInst>(OldTI)) {
    if (BI->isConditional()) {
      // If the condition was a known constant in the callee...
      ConstantBool *Cond = dyn_cast<ConstantBool>(BI->getCondition());
      if (Cond == 0)  // Or is a known constant in the caller...
        Cond = dyn_cast_or_null<ConstantBool>(ValueMap[BI->getCondition()]);
      if (Cond) {     // Constant fold to uncond branch!
        BasicBlock *Dest = BI->getSuccessor(!Cond->getValue());
        ValueMap[OldTI] = new BranchInst(Dest, NewBB);
        CloneBlock(Dest);
        TerminatorDone = true;
      }
    }
  } else if (const SwitchInst *SI = dyn_cast<SwitchInst>(OldTI)) {
    // If switching on a value known constant in the caller.
    ConstantInt *Cond = dyn_cast<ConstantInt>(SI->getCondition());
    if (Cond == 0)  // Or known constant after constant prop in the callee...
      Cond = dyn_cast_or_null<ConstantInt>(ValueMap[SI->getCondition()]);
    if (Cond) {     // Constant fold to uncond branch!
      BasicBlock *Dest = SI->getSuccessor(SI->findCaseValue(Cond));
      ValueMap[OldTI] = new BranchInst(Dest, NewBB);
      CloneBlock(Dest);
      TerminatorDone = true;
    }
  }
  
  if (!TerminatorDone) {
    Instruction *NewInst = OldTI->clone();
    if (OldTI->hasName())
      NewInst->setName(OldTI->getName()+NameSuffix);
    NewBB->getInstList().push_back(NewInst);
    ValueMap[OldTI] = NewInst;             // Add instruction map to value.
    
    // Recursively clone any reachable successor blocks.
    const TerminatorInst *TI = BB->getTerminator();
    for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i)
      CloneBlock(TI->getSuccessor(i));
  }
  
  if (CodeInfo) {
    CodeInfo->ContainsCalls          |= hasCalls;
    CodeInfo->ContainsUnwinds        |= isa<UnwindInst>(OldTI);
    CodeInfo->ContainsDynamicAllocas |= hasDynamicAllocas;
    CodeInfo->ContainsDynamicAllocas |= hasStaticAllocas && 
      BB != &BB->getParent()->front();
  }
  
  if (ReturnInst *RI = dyn_cast<ReturnInst>(NewBB->getTerminator()))
    Returns.push_back(RI);
}

/// ConstantFoldMappedInstruction - Constant fold the specified instruction,
/// mapping its operands through ValueMap if they are available.
Constant *PruningFunctionCloner::
ConstantFoldMappedInstruction(const Instruction *I) {
  if (isa<BinaryOperator>(I) || isa<ShiftInst>(I)) {
    if (Constant *Op0 = dyn_cast_or_null<Constant>(MapValue(I->getOperand(0),
                                                            ValueMap)))
      if (Constant *Op1 = dyn_cast_or_null<Constant>(MapValue(I->getOperand(1),
                                                              ValueMap)))
        return ConstantExpr::get(I->getOpcode(), Op0, Op1);
    return 0;
  }

  std::vector<Constant*> Ops;
  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
    if (Constant *Op = dyn_cast_or_null<Constant>(MapValue(I->getOperand(i),
                                                           ValueMap)))
      Ops.push_back(Op);
    else
      return 0;  // All operands not constant!

  return ConstantFoldInstOperands(I->getOpcode(), I->getType(), Ops);
}

/// CloneAndPruneFunctionInto - This works exactly like CloneFunctionInto,
/// except that it does some simple constant prop and DCE on the fly.  The
/// effect of this is to copy significantly less code in cases where (for
/// example) a function call with constant arguments is inlined, and those
/// constant arguments cause a significant amount of code in the callee to be
/// dead.  Since this doesn't produce an exactly copy of the input, it can't be
/// used for things like CloneFunction or CloneModule.
void llvm::CloneAndPruneFunctionInto(Function *NewFunc, const Function *OldFunc,
                                     std::map<const Value*, Value*> &ValueMap,
                                     std::vector<ReturnInst*> &Returns,
                                     const char *NameSuffix, 
                                     ClonedCodeInfo *CodeInfo) {
  assert(NameSuffix && "NameSuffix cannot be null!");
  
#ifndef NDEBUG
  for (Function::const_arg_iterator I = OldFunc->arg_begin(), 
       E = OldFunc->arg_end(); I != E; ++I)
    assert(ValueMap.count(I) && "No mapping from source argument specified!");
#endif
  
  PruningFunctionCloner PFC(NewFunc, OldFunc, ValueMap, Returns, 
                            NameSuffix, CodeInfo);

  // Clone the entry block, and anything recursively reachable from it.
  PFC.CloneBlock(&OldFunc->getEntryBlock());
  
  // Loop over all of the basic blocks in the old function.  If the block was
  // reachable, we have cloned it and the old block is now in the value map:
  // insert it into the new function in the right order.  If not, ignore it.
  //
  // Defer PHI resolution until rest of function is resolved.
  std::vector<const PHINode*> PHIToResolve;
  for (Function::const_iterator BI = OldFunc->begin(), BE = OldFunc->end();
       BI != BE; ++BI) {
    BasicBlock *NewBB = cast_or_null<BasicBlock>(ValueMap[BI]);
    if (NewBB == 0) continue;  // Dead block.

    // Add the new block to the new function.
    NewFunc->getBasicBlockList().push_back(NewBB);
    
    // Loop over all of the instructions in the block, fixing up operand
    // references as we go.  This uses ValueMap to do all the hard work.
    //
    BasicBlock::iterator I = NewBB->begin();
    
    // Handle PHI nodes specially, as we have to remove references to dead
    // blocks.
    if (PHINode *PN = dyn_cast<PHINode>(I)) {
      // Skip over all PHI nodes, remembering them for later.
      BasicBlock::const_iterator OldI = BI->begin();
      for (; (PN = dyn_cast<PHINode>(I)); ++I, ++OldI)
        PHIToResolve.push_back(cast<PHINode>(OldI));
    }
    
    // Otherwise, remap the rest of the instructions normally.
    for (; I != NewBB->end(); ++I)
      RemapInstruction(I, ValueMap);
  }
  
  // Defer PHI resolution until rest of function is resolved, PHI resolution
  // requires the CFG to be up-to-date.
  for (unsigned phino = 0, e = PHIToResolve.size(); phino != e; ) {
    const PHINode *OPN = PHIToResolve[phino];
    
    unsigned NumPreds = OPN->getNumIncomingValues();
    
    unsigned BBPHIStart = phino;
    const BasicBlock *OldBB = OPN->getParent();
    BasicBlock *NewBB = cast<BasicBlock>(ValueMap[OldBB]);

    // Map operands for blocks that are live and remove operands for blocks
    // that are dead.
    for (; phino != PHIToResolve.size() &&
         PHIToResolve[phino]->getParent() == OldBB; ++phino) {
      OPN = PHIToResolve[phino];
      PHINode *PN = cast<PHINode>(ValueMap[OPN]);
      for (unsigned pred = 0, e = NumPreds; pred != e; ++pred) {
        if (BasicBlock *MappedBlock = 
            cast_or_null<BasicBlock>(ValueMap[PN->getIncomingBlock(pred)])) {
          Value *InVal = MapValue(PN->getIncomingValue(pred), ValueMap);
          assert(InVal && "Unknown input value?");
          PN->setIncomingValue(pred, InVal);
          PN->setIncomingBlock(pred, MappedBlock);
        } else {
          PN->removeIncomingValue(pred, false);
          --pred, --e;  // Revisit the next entry.
        }
      } 
    }
    
    // The loop above has removed PHI entries for those blocks that are dead
    // and has updated others.  However, if a block is live (i.e. copied over)
    // but its terminator has been changed to not go to this block, then our
    // phi nodes will have invalid entries.  Update the PHI nodes in this
    // case.
    PHINode *PN = cast<PHINode>(NewBB->begin());
    NumPreds = std::distance(pred_begin(NewBB), pred_end(NewBB));
    if (NumPreds != PN->getNumIncomingValues()) {
      assert(NumPreds < PN->getNumIncomingValues());
      // Count how many times each predecessor comes to this block.
      std::map<BasicBlock*, unsigned> PredCount;
      for (pred_iterator PI = pred_begin(NewBB), E = pred_end(NewBB);
           PI != E; ++PI)
        --PredCount[*PI];
      
      // Figure out how many entries to remove from each PHI.
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
        ++PredCount[PN->getIncomingBlock(i)];
      
      // At this point, the excess predecessor entries are positive in the
      // map.  Loop over all of the PHIs and remove excess predecessor
      // entries.
      BasicBlock::iterator I = NewBB->begin();
      for (; (PN = dyn_cast<PHINode>(I)); ++I) {
        for (std::map<BasicBlock*, unsigned>::iterator PCI =PredCount.begin(),
             E = PredCount.end(); PCI != E; ++PCI) {
          BasicBlock *Pred     = PCI->first;
          for (unsigned NumToRemove = PCI->second; NumToRemove; --NumToRemove)
            PN->removeIncomingValue(Pred, false);
        }
      }
    }
    
    // If the loops above have made these phi nodes have 0 or 1 operand,
    // replace them with undef or the input value.  We must do this for
    // correctness, because 0-operand phis are not valid.
    PN = cast<PHINode>(NewBB->begin());
    if (PN->getNumIncomingValues() == 0) {
      BasicBlock::iterator I = NewBB->begin();
      BasicBlock::const_iterator OldI = OldBB->begin();
      while ((PN = dyn_cast<PHINode>(I++))) {
        Value *NV = UndefValue::get(PN->getType());
        PN->replaceAllUsesWith(NV);
        assert(ValueMap[OldI] == PN && "ValueMap mismatch");
        ValueMap[OldI] = NV;
        PN->eraseFromParent();
        ++OldI;
      }
    } else if (PN->getNumIncomingValues() == 1) {
      BasicBlock::iterator I = NewBB->begin();
      BasicBlock::const_iterator OldI = OldBB->begin();
      while ((PN = dyn_cast<PHINode>(I++))) {
        Value *NV = PN->getIncomingValue(0);
        PN->replaceAllUsesWith(NV);
        assert(ValueMap[OldI] == PN && "ValueMap mismatch");
        ValueMap[OldI] = NV;
        PN->eraseFromParent();
        ++OldI;
      }
    }
  }
}


