//===- TailDuplication.cpp - Simplify CFG through tail duplication --------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This pass performs a limited form of tail duplication, intended to simplify
// CFGs by removing some unconditional branches.  This pass is necessary to
// straighten out loops created by the C front-end, but also is capable of
// making other code nicer.  After this pass is run, the CFG simplify pass
// should be run to clean up the mess.
//
// This pass could be enhanced in the future to use profile information to be
// more aggressive.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Constant.h"
#include "llvm/Function.h"
#include "llvm/iPHINode.h"
#include "llvm/iTerminators.h"
#include "llvm/Pass.h"
#include "llvm/Type.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/ValueHolder.h"
#include "llvm/Transforms/Utils/Local.h"
#include "Support/Debug.h"
#include "Support/Statistic.h"
using namespace llvm;

namespace {
  Statistic<> NumEliminated("tailduplicate",
                            "Number of unconditional branches eliminated");
  Statistic<> NumPHINodes("tailduplicate", "Number of phi nodes inserted");

  class TailDup : public FunctionPass {
    bool runOnFunction(Function &F);
  private:
    inline bool shouldEliminateUnconditionalBranch(TerminatorInst *TI);
    inline bool canEliminateUnconditionalBranch(TerminatorInst *TI);
    inline void eliminateUnconditionalBranch(BranchInst *BI);
    inline void InsertPHINodesIfNecessary(Instruction *OrigInst, Value *NewInst,
                                          BasicBlock *NewBlock);
    inline Value *GetValueInBlock(BasicBlock *BB, Value *OrigVal,
                                  std::map<BasicBlock*, ValueHolder> &ValueMap,
                              std::map<BasicBlock*, ValueHolder> &OutValueMap);
    inline Value *GetValueOutBlock(BasicBlock *BB, Value *OrigVal,
                                   std::map<BasicBlock*, ValueHolder> &ValueMap,
                               std::map<BasicBlock*, ValueHolder> &OutValueMap);
  };
  RegisterOpt<TailDup> X("tailduplicate", "Tail Duplication");
}

// Public interface to the Tail Duplication pass
Pass *llvm::createTailDuplicationPass() { return new TailDup(); }

/// runOnFunction - Top level algorithm - Loop over each unconditional branch in
/// the function, eliminating it if it looks attractive enough.
///
bool TailDup::runOnFunction(Function &F) {
  bool Changed = false;
  for (Function::iterator I = F.begin(), E = F.end(); I != E; )
    if (shouldEliminateUnconditionalBranch(I->getTerminator()) &&
        canEliminateUnconditionalBranch(I->getTerminator())) {
      eliminateUnconditionalBranch(cast<BranchInst>(I->getTerminator()));
      Changed = true;
    } else {
      ++I;
    }
  return Changed;
}

/// shouldEliminateUnconditionalBranch - Return true if this branch looks
/// attractive to eliminate.  We eliminate the branch if the destination basic
/// block has <= 5 instructions in it, not counting PHI nodes.  In practice,
/// since one of these is a terminator instruction, this means that we will add
/// up to 4 instructions to the new block.
///
/// We don't count PHI nodes in the count since they will be removed when the
/// contents of the block are copied over.
///
bool TailDup::shouldEliminateUnconditionalBranch(TerminatorInst *TI) {
  BranchInst *BI = dyn_cast<BranchInst>(TI);
  if (!BI || !BI->isUnconditional()) return false;  // Not an uncond branch!

  BasicBlock *Dest = BI->getSuccessor(0);
  if (Dest == BI->getParent()) return false;        // Do not loop infinitely!

  // Do not inline a block if we will just get another branch to the same block!
  if (BranchInst *DBI = dyn_cast<BranchInst>(Dest->getTerminator()))
    if (DBI->isUnconditional() && DBI->getSuccessor(0) == Dest)
      return false;                                 // Do not loop infinitely!

  // Do not bother working on dead blocks...
  pred_iterator PI = pred_begin(Dest), PE = pred_end(Dest);
  if (PI == PE && Dest != Dest->getParent()->begin())
    return false;   // It's just a dead block, ignore it...

  // Also, do not bother with blocks with only a single predecessor: simplify
  // CFG will fold these two blocks together!
  ++PI;
  if (PI == PE) return false;  // Exactly one predecessor!

  BasicBlock::iterator I = Dest->begin();
  while (isa<PHINode>(*I)) ++I;

  for (unsigned Size = 0; I != Dest->end(); ++Size, ++I)
    if (Size == 6) return false;  // The block is too large...
  return true;  
}

/// canEliminateUnconditionalBranch - Unfortunately, the general form of tail
/// duplication can do very bad things to SSA form, by destroying arbitrary
/// relationships between dominators and dominator frontiers as it processes the
/// program.  The right solution for this is to have an incrementally updating
/// dominator data structure, which can gracefully react to arbitrary
/// "addEdge/removeEdge" changes to the CFG.  Implementing this is nontrivial,
/// however, so we just disable the transformation in cases where it is not
/// currently safe.
///
bool TailDup::canEliminateUnconditionalBranch(TerminatorInst *TI) {
  // Basically, we refuse to make the transformation if any of the values
  // computed in the 'tail' are used in any other basic blocks.
  BasicBlock *Tail = TI->getSuccessor(0);
  assert(isa<BranchInst>(TI) && cast<BranchInst>(TI)->isUnconditional());
  
  for (BasicBlock::iterator I = Tail->begin(), E = Tail->end(); I != E; ++I)
    for (Value::use_iterator UI = I->use_begin(), E = I->use_end(); UI != E;
         ++UI) {
      Instruction *User = cast<Instruction>(*UI);
      if (User->getParent() != Tail || isa<PHINode>(User))
        return false;
    }
  return true;
}


/// eliminateUnconditionalBranch - Clone the instructions from the destination
/// block into the source block, eliminating the specified unconditional branch.
/// If the destination block defines values used by successors of the dest
/// block, we may need to insert PHI nodes.
///
void TailDup::eliminateUnconditionalBranch(BranchInst *Branch) {
  BasicBlock *SourceBlock = Branch->getParent();
  BasicBlock *DestBlock = Branch->getSuccessor(0);
  assert(SourceBlock != DestBlock && "Our predicate is broken!");

  DEBUG(std::cerr << "TailDuplication[" << SourceBlock->getParent()->getName()
                  << "]: Eliminating branch: " << *Branch);

  // We are going to have to map operands from the original block B to the new
  // copy of the block B'.  If there are PHI nodes in the DestBlock, these PHI
  // nodes also define part of this mapping.  Loop over these PHI nodes, adding
  // them to our mapping.
  //
  std::map<Value*, Value*> ValueMapping;

  BasicBlock::iterator BI = DestBlock->begin();
  bool HadPHINodes = isa<PHINode>(BI);
  for (; PHINode *PN = dyn_cast<PHINode>(BI); ++BI)
    ValueMapping[PN] = PN->getIncomingValueForBlock(SourceBlock);

  // Clone the non-phi instructions of the dest block into the source block,
  // keeping track of the mapping...
  //
  for (; BI != DestBlock->end(); ++BI) {
    Instruction *New = BI->clone();
    New->setName(BI->getName());
    SourceBlock->getInstList().push_back(New);
    ValueMapping[BI] = New;
  }

  // Now that we have built the mapping information and cloned all of the
  // instructions (giving us a new terminator, among other things), walk the new
  // instructions, rewriting references of old instructions to use new
  // instructions.
  //
  BI = Branch; ++BI;  // Get an iterator to the first new instruction
  for (; BI != SourceBlock->end(); ++BI)
    for (unsigned i = 0, e = BI->getNumOperands(); i != e; ++i)
      if (Value *Remapped = ValueMapping[BI->getOperand(i)])
        BI->setOperand(i, Remapped);

  // Next we check to see if any of the successors of DestBlock had PHI nodes.
  // If so, we need to add entries to the PHI nodes for SourceBlock now.
  for (succ_iterator SI = succ_begin(DestBlock), SE = succ_end(DestBlock);
       SI != SE; ++SI) {
    BasicBlock *Succ = *SI;
    for (BasicBlock::iterator PNI = Succ->begin();
         PHINode *PN = dyn_cast<PHINode>(PNI); ++PNI) {
      // Ok, we have a PHI node.  Figure out what the incoming value was for the
      // DestBlock.
      Value *IV = PN->getIncomingValueForBlock(DestBlock);
      
      // Remap the value if necessary...
      if (Value *MappedIV = ValueMapping[IV])
        IV = MappedIV;
      PN->addIncoming(IV, SourceBlock);
    }
  }
  
  // Now that all of the instructions are correctly copied into the SourceBlock,
  // we have one more minor problem: the successors of the original DestBB may
  // use the values computed in DestBB either directly (if DestBB dominated the
  // block), or through a PHI node.  In either case, we need to insert PHI nodes
  // into any successors of DestBB (which are now our successors) for each value
  // that is computed in DestBB, but is used outside of it.  All of these uses
  // we have to rewrite with the new PHI node.
  //
  if (succ_begin(SourceBlock) != succ_end(SourceBlock)) // Avoid wasting time...
    for (BI = DestBlock->begin(); BI != DestBlock->end(); ++BI)
      if (BI->getType() != Type::VoidTy)
        InsertPHINodesIfNecessary(BI, ValueMapping[BI], SourceBlock);

  // Final step: now that we have finished everything up, walk the cloned
  // instructions one last time, constant propagating and DCE'ing them, because
  // they may not be needed anymore.
  //
  BI = Branch; ++BI;  // Get an iterator to the first new instruction
  if (HadPHINodes)
    while (BI != SourceBlock->end())
      if (!dceInstruction(BI) && !doConstantPropagation(BI))
        ++BI;

  DestBlock->removePredecessor(SourceBlock); // Remove entries in PHI nodes...
  SourceBlock->getInstList().erase(Branch);  // Destroy the uncond branch...
  
  ++NumEliminated;  // We just killed a branch!
}

/// InsertPHINodesIfNecessary - So at this point, we cloned the OrigInst
/// instruction into the NewBlock with the value of NewInst.  If OrigInst was
/// used outside of its defining basic block, we need to insert a PHI nodes into
/// the successors.
///
void TailDup::InsertPHINodesIfNecessary(Instruction *OrigInst, Value *NewInst,
                                        BasicBlock *NewBlock) {
  // Loop over all of the uses of OrigInst, rewriting them to be newly inserted
  // PHI nodes, unless they are in the same basic block as OrigInst.
  BasicBlock *OrigBlock = OrigInst->getParent();
  std::vector<Instruction*> Users;
  Users.reserve(OrigInst->use_size());
  for (Value::use_iterator I = OrigInst->use_begin(), E = OrigInst->use_end();
       I != E; ++I) {
    Instruction *In = cast<Instruction>(*I);
    if (In->getParent() != OrigBlock ||  // Don't modify uses in the orig block!
        isa<PHINode>(In))
      Users.push_back(In);
  }

  // The common case is that the instruction is only used within the block that
  // defines it.  If we have this case, quick exit.
  //
  if (Users.empty()) return; 

  // Otherwise, we have a more complex case, handle it now.  This requires the
  // construction of a mapping between a basic block and the value to use when
  // in the scope of that basic block.  This map will map to the original and
  // new values when in the original or new block, but will map to inserted PHI
  // nodes when in other blocks.
  //
  std::map<BasicBlock*, ValueHolder> ValueMap;
  std::map<BasicBlock*, ValueHolder> OutValueMap;   // The outgoing value map
  OutValueMap[OrigBlock] = OrigInst;
  OutValueMap[NewBlock ] = NewInst;    // Seed the initial values...

  DEBUG(std::cerr << "  ** Inserting PHI nodes for " << OrigInst);
  while (!Users.empty()) {
    Instruction *User = Users.back(); Users.pop_back();

    if (PHINode *PN = dyn_cast<PHINode>(User)) {
      // PHI nodes must be handled specially here, because their operands are
      // actually defined in predecessor basic blocks, NOT in the block that the
      // PHI node lives in.  Note that we have already added entries to PHI nods
      // which are in blocks that are immediate successors of OrigBlock, so
      // don't modify them again.
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
        if (PN->getIncomingValue(i) == OrigInst &&
            PN->getIncomingBlock(i) != OrigBlock) {
          Value *V = GetValueOutBlock(PN->getIncomingBlock(i), OrigInst,
                                      ValueMap, OutValueMap);
          PN->setIncomingValue(i, V);
        }
      
    } else {
      // Any other user of the instruction can just replace any uses with the
      // new value defined in the block it resides in.
      Value *V = GetValueInBlock(User->getParent(), OrigInst, ValueMap,
                                 OutValueMap);
      User->replaceUsesOfWith(OrigInst, V);
    }
  }
}

/// GetValueInBlock - This is a recursive method which inserts PHI nodes into
/// the function until there is a value available in basic block BB.
///
Value *TailDup::GetValueInBlock(BasicBlock *BB, Value *OrigVal,
                                std::map<BasicBlock*, ValueHolder> &ValueMap,
                                std::map<BasicBlock*,ValueHolder> &OutValueMap){
  ValueHolder &BBVal = ValueMap[BB];
  if (BBVal) return BBVal;       // Value already computed for this block?

  // If this block has no predecessors, then it must be unreachable, thus, it
  // doesn't matter which value we use.
  if (pred_begin(BB) == pred_end(BB))
    return BBVal = Constant::getNullValue(OrigVal->getType());

  // If there is no value already available in this basic block, we need to
  // either reuse a value from an incoming, dominating, basic block, or we need
  // to create a new PHI node to merge in different incoming values.  Because we
  // don't know if we're part of a loop at this point or not, we create a PHI
  // node, even if we will ultimately eliminate it.
  PHINode *PN = new PHINode(OrigVal->getType(), OrigVal->getName()+".pn",
                            BB->begin());
  BBVal = PN;   // Insert this into the BBVal slot in case of cycles...

  ValueHolder &BBOutVal = OutValueMap[BB];
  if (BBOutVal == 0) BBOutVal = PN;

  // Now that we have created the PHI node, loop over all of the predecessors of
  // this block, computing an incoming value for the predecessor.
  std::vector<BasicBlock*> Preds(pred_begin(BB), pred_end(BB));
  for (unsigned i = 0, e = Preds.size(); i != e; ++i)
    PN->addIncoming(GetValueOutBlock(Preds[i], OrigVal, ValueMap, OutValueMap),
                    Preds[i]);

  // The PHI node is complete.  In many cases, however the PHI node was
  // ultimately unnecessary: we could have just reused a dominating incoming
  // value.  If this is the case, nuke the PHI node and replace the map entry
  // with the dominating value.
  //
  assert(PN->getNumIncomingValues() > 0 && "No predecessors?");

  // Check to see if all of the elements in the PHI node are either the PHI node
  // itself or ONE particular value.
  unsigned i = 0;
  Value *ReplVal = PN->getIncomingValue(i);
  for (; ReplVal == PN && i != PN->getNumIncomingValues(); ++i)
    ReplVal = PN->getIncomingValue(i);  // Skip values equal to the PN

  for (; i != PN->getNumIncomingValues(); ++i)
    if (PN->getIncomingValue(i) != PN && PN->getIncomingValue(i) != ReplVal) {
      ReplVal = 0;
      break;
    }

  // Found a value to replace the PHI node with?
  if (ReplVal && ReplVal != PN) {
    PN->replaceAllUsesWith(ReplVal);
    BB->getInstList().erase(PN);   // Erase the PHI node...
  } else {
    ++NumPHINodes;
  }

  return BBVal;
}

Value *TailDup::GetValueOutBlock(BasicBlock *BB, Value *OrigVal,
                                 std::map<BasicBlock*, ValueHolder> &ValueMap,
                              std::map<BasicBlock*, ValueHolder> &OutValueMap) {
  ValueHolder &BBVal = OutValueMap[BB];
  if (BBVal) return BBVal;       // Value already computed for this block?

  return GetValueInBlock(BB, OrigVal, ValueMap, OutValueMap);
}
