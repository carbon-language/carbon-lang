//===- CloneFunction.cpp - Clone a function into another function ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/IntrinsicInst.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Function.h"
#include "llvm/LLVMContext.h"
#include "llvm/Metadata.h"
#include "llvm/Support/CFG.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/DebugInfo.h"
#include "llvm/ADT/SmallVector.h"
#include <map>
using namespace llvm;

// CloneBasicBlock - See comments in Cloning.h
BasicBlock *llvm::CloneBasicBlock(const BasicBlock *BB,
                                  ValueToValueMapTy &VMap,
                                  const Twine &NameSuffix, Function *F,
                                  ClonedCodeInfo *CodeInfo) {
  BasicBlock *NewBB = BasicBlock::Create(BB->getContext(), "", F);
  if (BB->hasName()) NewBB->setName(BB->getName()+NameSuffix);

  bool hasCalls = false, hasDynamicAllocas = false, hasStaticAllocas = false;
  
  // Loop over all instructions, and copy them over.
  for (BasicBlock::const_iterator II = BB->begin(), IE = BB->end();
       II != IE; ++II) {
    Instruction *NewInst = II->clone();
    if (II->hasName())
      NewInst->setName(II->getName()+NameSuffix);
    NewBB->getInstList().push_back(NewInst);
    VMap[II] = NewInst;                // Add instruction map to value.
    
    hasCalls |= (isa<CallInst>(II) && !isa<DbgInfoIntrinsic>(II));
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
                                        BB != &BB->getParent()->getEntryBlock();
  }
  return NewBB;
}

// Clone OldFunc into NewFunc, transforming the old arguments into references to
// VMap values.
//
void llvm::CloneFunctionInto(Function *NewFunc, const Function *OldFunc,
                             ValueToValueMapTy &VMap,
                             bool ModuleLevelChanges,
                             SmallVectorImpl<ReturnInst*> &Returns,
                             const char *NameSuffix, ClonedCodeInfo *CodeInfo) {
  assert(NameSuffix && "NameSuffix cannot be null!");

#ifndef NDEBUG
  for (Function::const_arg_iterator I = OldFunc->arg_begin(), 
       E = OldFunc->arg_end(); I != E; ++I)
    assert(VMap.count(I) && "No mapping from source argument specified!");
#endif

  // Clone any attributes.
  if (NewFunc->arg_size() == OldFunc->arg_size())
    NewFunc->copyAttributesFrom(OldFunc);
  else {
    //Some arguments were deleted with the VMap. Copy arguments one by one
    for (Function::const_arg_iterator I = OldFunc->arg_begin(), 
           E = OldFunc->arg_end(); I != E; ++I)
      if (Argument* Anew = dyn_cast<Argument>(VMap[I]))
        Anew->addAttr( OldFunc->getAttributes()
                       .getParamAttributes(I->getArgNo() + 1));
    NewFunc->setAttributes(NewFunc->getAttributes()
                           .addAttr(0, OldFunc->getAttributes()
                                     .getRetAttributes()));
    NewFunc->setAttributes(NewFunc->getAttributes()
                           .addAttr(~0, OldFunc->getAttributes()
                                     .getFnAttributes()));

  }

  // Loop over all of the basic blocks in the function, cloning them as
  // appropriate.  Note that we save BE this way in order to handle cloning of
  // recursive functions into themselves.
  //
  for (Function::const_iterator BI = OldFunc->begin(), BE = OldFunc->end();
       BI != BE; ++BI) {
    const BasicBlock &BB = *BI;

    // Create a new basic block and copy instructions into it!
    BasicBlock *CBB = CloneBasicBlock(&BB, VMap, NameSuffix, NewFunc, CodeInfo);
    VMap[&BB] = CBB;                       // Add basic block mapping.

    if (ReturnInst *RI = dyn_cast<ReturnInst>(CBB->getTerminator()))
      Returns.push_back(RI);
  }

  // Loop over all of the instructions in the function, fixing up operand
  // references as we go.  This uses VMap to do all the hard work.
  for (Function::iterator BB = cast<BasicBlock>(VMap[OldFunc->begin()]),
         BE = NewFunc->end(); BB != BE; ++BB)
    // Loop over all instructions, fixing each one as we find it...
    for (BasicBlock::iterator II = BB->begin(); II != BB->end(); ++II)
      RemapInstruction(II, VMap,
                       ModuleLevelChanges ? RF_None : RF_NoModuleLevelChanges);
}

/// CloneFunction - Return a copy of the specified function, but without
/// embedding the function into another module.  Also, any references specified
/// in the VMap are changed to refer to their mapped value instead of the
/// original one.  If any of the arguments to the function are in the VMap,
/// the arguments are deleted from the resultant function.  The VMap is
/// updated to include mappings from all of the instructions and basicblocks in
/// the function from their old to new values.
///
Function *llvm::CloneFunction(const Function *F, ValueToValueMapTy &VMap,
                              bool ModuleLevelChanges,
                              ClonedCodeInfo *CodeInfo) {
  std::vector<Type*> ArgTypes;

  // The user might be deleting arguments to the function by specifying them in
  // the VMap.  If so, we need to not add the arguments to the arg ty vector
  //
  for (Function::const_arg_iterator I = F->arg_begin(), E = F->arg_end();
       I != E; ++I)
    if (VMap.count(I) == 0)  // Haven't mapped the argument to anything yet?
      ArgTypes.push_back(I->getType());

  // Create a new function type...
  FunctionType *FTy = FunctionType::get(F->getFunctionType()->getReturnType(),
                                    ArgTypes, F->getFunctionType()->isVarArg());

  // Create the new function...
  Function *NewF = Function::Create(FTy, F->getLinkage(), F->getName());

  // Loop over the arguments, copying the names of the mapped arguments over...
  Function::arg_iterator DestI = NewF->arg_begin();
  for (Function::const_arg_iterator I = F->arg_begin(), E = F->arg_end();
       I != E; ++I)
    if (VMap.count(I) == 0) {   // Is this argument preserved?
      DestI->setName(I->getName()); // Copy the name over...
      VMap[I] = DestI++;        // Add mapping to VMap
    }

  SmallVector<ReturnInst*, 8> Returns;  // Ignore returns cloned.
  CloneFunctionInto(NewF, F, VMap, ModuleLevelChanges, Returns, "", CodeInfo);
  return NewF;
}



namespace {
  /// PruningFunctionCloner - This class is a private class used to implement
  /// the CloneAndPruneFunctionInto method.
  struct PruningFunctionCloner {
    Function *NewFunc;
    const Function *OldFunc;
    ValueToValueMapTy &VMap;
    bool ModuleLevelChanges;
    SmallVectorImpl<ReturnInst*> &Returns;
    const char *NameSuffix;
    ClonedCodeInfo *CodeInfo;
    const TargetData *TD;
  public:
    PruningFunctionCloner(Function *newFunc, const Function *oldFunc,
                          ValueToValueMapTy &valueMap,
                          bool moduleLevelChanges,
                          SmallVectorImpl<ReturnInst*> &returns,
                          const char *nameSuffix, 
                          ClonedCodeInfo *codeInfo,
                          const TargetData *td)
    : NewFunc(newFunc), OldFunc(oldFunc),
      VMap(valueMap), ModuleLevelChanges(moduleLevelChanges),
      Returns(returns), NameSuffix(nameSuffix), CodeInfo(codeInfo), TD(td) {
    }

    /// CloneBlock - The specified block is found to be reachable, clone it and
    /// anything that it can reach.
    void CloneBlock(const BasicBlock *BB,
                    std::vector<const BasicBlock*> &ToClone);
    
  public:
    /// ConstantFoldMappedInstruction - Constant fold the specified instruction,
    /// mapping its operands through VMap if they are available.
    Constant *ConstantFoldMappedInstruction(const Instruction *I);
  };
}

/// CloneBlock - The specified block is found to be reachable, clone it and
/// anything that it can reach.
void PruningFunctionCloner::CloneBlock(const BasicBlock *BB,
                                       std::vector<const BasicBlock*> &ToClone){
  TrackingVH<Value> &BBEntry = VMap[BB];

  // Have we already cloned this block?
  if (BBEntry) return;
  
  // Nope, clone it now.
  BasicBlock *NewBB;
  BBEntry = NewBB = BasicBlock::Create(BB->getContext());
  if (BB->hasName()) NewBB->setName(BB->getName()+NameSuffix);

  bool hasCalls = false, hasDynamicAllocas = false, hasStaticAllocas = false;
  
  // Loop over all instructions, and copy them over, DCE'ing as we go.  This
  // loop doesn't include the terminator.
  for (BasicBlock::const_iterator II = BB->begin(), IE = --BB->end();
       II != IE; ++II) {
    // If this instruction constant folds, don't bother cloning the instruction,
    // instead, just add the constant to the value map.
    if (Constant *C = ConstantFoldMappedInstruction(II)) {
      VMap[II] = C;
      continue;
    }

    Instruction *NewInst = II->clone();
    if (II->hasName())
      NewInst->setName(II->getName()+NameSuffix);
    NewBB->getInstList().push_back(NewInst);
    VMap[II] = NewInst;                // Add instruction map to value.
    
    hasCalls |= (isa<CallInst>(II) && !isa<DbgInfoIntrinsic>(II));
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
      ConstantInt *Cond = dyn_cast<ConstantInt>(BI->getCondition());
      // Or is a known constant in the caller...
      if (Cond == 0) {
        Value *V = VMap[BI->getCondition()];
        Cond = dyn_cast_or_null<ConstantInt>(V);
      }

      // Constant fold to uncond branch!
      if (Cond) {
        BasicBlock *Dest = BI->getSuccessor(!Cond->getZExtValue());
        VMap[OldTI] = BranchInst::Create(Dest, NewBB);
        ToClone.push_back(Dest);
        TerminatorDone = true;
      }
    }
  } else if (const SwitchInst *SI = dyn_cast<SwitchInst>(OldTI)) {
    // If switching on a value known constant in the caller.
    ConstantInt *Cond = dyn_cast<ConstantInt>(SI->getCondition());
    if (Cond == 0) { // Or known constant after constant prop in the callee...
      Value *V = VMap[SI->getCondition()];
      Cond = dyn_cast_or_null<ConstantInt>(V);
    }
    if (Cond) {     // Constant fold to uncond branch!
      BasicBlock *Dest = SI->getSuccessor(SI->findCaseValue(Cond));
      VMap[OldTI] = BranchInst::Create(Dest, NewBB);
      ToClone.push_back(Dest);
      TerminatorDone = true;
    }
  }
  
  if (!TerminatorDone) {
    Instruction *NewInst = OldTI->clone();
    if (OldTI->hasName())
      NewInst->setName(OldTI->getName()+NameSuffix);
    NewBB->getInstList().push_back(NewInst);
    VMap[OldTI] = NewInst;             // Add instruction map to value.
    
    // Recursively clone any reachable successor blocks.
    const TerminatorInst *TI = BB->getTerminator();
    for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i)
      ToClone.push_back(TI->getSuccessor(i));
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
/// mapping its operands through VMap if they are available.
Constant *PruningFunctionCloner::
ConstantFoldMappedInstruction(const Instruction *I) {
  SmallVector<Constant*, 8> Ops;
  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
    if (Constant *Op = dyn_cast_or_null<Constant>(MapValue(I->getOperand(i),
                                                           VMap,
                  ModuleLevelChanges ? RF_None : RF_NoModuleLevelChanges)))
      Ops.push_back(Op);
    else
      return 0;  // All operands not constant!

  if (const CmpInst *CI = dyn_cast<CmpInst>(I))
    return ConstantFoldCompareInstOperands(CI->getPredicate(), Ops[0], Ops[1],
                                           TD);

  if (const LoadInst *LI = dyn_cast<LoadInst>(I))
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Ops[0]))
      if (!LI->isVolatile() && CE->getOpcode() == Instruction::GetElementPtr)
        if (GlobalVariable *GV = dyn_cast<GlobalVariable>(CE->getOperand(0)))
          if (GV->isConstant() && GV->hasDefinitiveInitializer())
            return ConstantFoldLoadThroughGEPConstantExpr(GV->getInitializer(),
                                                          CE);

  return ConstantFoldInstOperands(I->getOpcode(), I->getType(), Ops, TD);
}

/// CloneAndPruneFunctionInto - This works exactly like CloneFunctionInto,
/// except that it does some simple constant prop and DCE on the fly.  The
/// effect of this is to copy significantly less code in cases where (for
/// example) a function call with constant arguments is inlined, and those
/// constant arguments cause a significant amount of code in the callee to be
/// dead.  Since this doesn't produce an exact copy of the input, it can't be
/// used for things like CloneFunction or CloneModule.
void llvm::CloneAndPruneFunctionInto(Function *NewFunc, const Function *OldFunc,
                                     ValueToValueMapTy &VMap,
                                     bool ModuleLevelChanges,
                                     SmallVectorImpl<ReturnInst*> &Returns,
                                     const char *NameSuffix, 
                                     ClonedCodeInfo *CodeInfo,
                                     const TargetData *TD,
                                     Instruction *TheCall) {
  assert(NameSuffix && "NameSuffix cannot be null!");
  
#ifndef NDEBUG
  for (Function::const_arg_iterator II = OldFunc->arg_begin(), 
       E = OldFunc->arg_end(); II != E; ++II)
    assert(VMap.count(II) && "No mapping from source argument specified!");
#endif

  PruningFunctionCloner PFC(NewFunc, OldFunc, VMap, ModuleLevelChanges,
                            Returns, NameSuffix, CodeInfo, TD);

  // Clone the entry block, and anything recursively reachable from it.
  std::vector<const BasicBlock*> CloneWorklist;
  CloneWorklist.push_back(&OldFunc->getEntryBlock());
  while (!CloneWorklist.empty()) {
    const BasicBlock *BB = CloneWorklist.back();
    CloneWorklist.pop_back();
    PFC.CloneBlock(BB, CloneWorklist);
  }
  
  // Loop over all of the basic blocks in the old function.  If the block was
  // reachable, we have cloned it and the old block is now in the value map:
  // insert it into the new function in the right order.  If not, ignore it.
  //
  // Defer PHI resolution until rest of function is resolved.
  SmallVector<const PHINode*, 16> PHIToResolve;
  for (Function::const_iterator BI = OldFunc->begin(), BE = OldFunc->end();
       BI != BE; ++BI) {
    Value *V = VMap[BI];
    BasicBlock *NewBB = cast_or_null<BasicBlock>(V);
    if (NewBB == 0) continue;  // Dead block.

    // Add the new block to the new function.
    NewFunc->getBasicBlockList().push_back(NewBB);
    
    // Loop over all of the instructions in the block, fixing up operand
    // references as we go.  This uses VMap to do all the hard work.
    //
    BasicBlock::iterator I = NewBB->begin();

    DebugLoc TheCallDL;
    if (TheCall) 
      TheCallDL = TheCall->getDebugLoc();
    
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
      RemapInstruction(I, VMap,
                       ModuleLevelChanges ? RF_None : RF_NoModuleLevelChanges);
  }
  
  // Defer PHI resolution until rest of function is resolved, PHI resolution
  // requires the CFG to be up-to-date.
  for (unsigned phino = 0, e = PHIToResolve.size(); phino != e; ) {
    const PHINode *OPN = PHIToResolve[phino];
    unsigned NumPreds = OPN->getNumIncomingValues();
    const BasicBlock *OldBB = OPN->getParent();
    BasicBlock *NewBB = cast<BasicBlock>(VMap[OldBB]);

    // Map operands for blocks that are live and remove operands for blocks
    // that are dead.
    for (; phino != PHIToResolve.size() &&
         PHIToResolve[phino]->getParent() == OldBB; ++phino) {
      OPN = PHIToResolve[phino];
      PHINode *PN = cast<PHINode>(VMap[OPN]);
      for (unsigned pred = 0, e = NumPreds; pred != e; ++pred) {
        Value *V = VMap[PN->getIncomingBlock(pred)];
        if (BasicBlock *MappedBlock = cast_or_null<BasicBlock>(V)) {
          Value *InVal = MapValue(PN->getIncomingValue(pred),
                                  VMap, 
                        ModuleLevelChanges ? RF_None : RF_NoModuleLevelChanges);
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
        assert(VMap[OldI] == PN && "VMap mismatch");
        VMap[OldI] = NV;
        PN->eraseFromParent();
        ++OldI;
      }
    }
    // NOTE: We cannot eliminate single entry phi nodes here, because of
    // VMap.  Single entry phi nodes can have multiple VMap entries
    // pointing at them.  Thus, deleting one would require scanning the VMap
    // to update any entries in it that would require that.  This would be
    // really slow.
  }
  
  // Now that the inlined function body has been fully constructed, go through
  // and zap unconditional fall-through branches.  This happen all the time when
  // specializing code: code specialization turns conditional branches into
  // uncond branches, and this code folds them.
  Function::iterator I = cast<BasicBlock>(VMap[&OldFunc->getEntryBlock()]);
  while (I != NewFunc->end()) {
    BranchInst *BI = dyn_cast<BranchInst>(I->getTerminator());
    if (!BI || BI->isConditional()) { ++I; continue; }
    
    // Note that we can't eliminate uncond branches if the destination has
    // single-entry PHI nodes.  Eliminating the single-entry phi nodes would
    // require scanning the VMap to update any entries that point to the phi
    // node.
    BasicBlock *Dest = BI->getSuccessor(0);
    if (!Dest->getSinglePredecessor() || isa<PHINode>(Dest->begin())) {
      ++I; continue;
    }
    
    // We know all single-entry PHI nodes in the inlined function have been
    // removed, so we just need to splice the blocks.
    BI->eraseFromParent();
    
    // Make all PHI nodes that referred to Dest now refer to I as their source.
    Dest->replaceAllUsesWith(I);

    // Move all the instructions in the succ to the pred.
    I->getInstList().splice(I->end(), Dest->getInstList());
    
    // Remove the dest block.
    Dest->eraseFromParent();
    
    // Do not increment I, iteratively merge all things this block branches to.
  }
}
