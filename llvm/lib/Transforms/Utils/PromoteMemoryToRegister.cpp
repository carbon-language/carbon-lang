//===- PromoteMemoryToRegister.cpp - Convert memory refs to regs ----------===//
//
// This file is used to promote memory references to be register references.  A
// simple example of the transformation performed by this function is:
//
//        FROM CODE                           TO CODE
//   %X = alloca int, uint 1                 ret int 42
//   store int 42, int *%X
//   %Y = load int* %X
//   ret int %Y
//
// The code is transformed by looping over all of the alloca instruction,
// calculating dominator frontiers, then inserting phi-nodes following the usual
// SSA construction algorithm.  This code does not modify the CFG of the
// function.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/iMemory.h"
#include "llvm/iPHINode.h"
#include "llvm/iTerminators.h"
#include "llvm/Function.h"
#include "llvm/Constant.h"
#include "llvm/Type.h"
#include "llvm/Support/CFG.h"
#include "Support/StringExtras.h"

/// isAllocaPromotable - Return true if this alloca is legal for promotion.
/// This is true if there are only loads and stores to the alloca...
///
bool isAllocaPromotable(const AllocaInst *AI, const TargetData &TD) {
  // FIXME: If the memory unit is of pointer or integer type, we can permit
  // assignments to subsections of the memory unit.

  // Only allow direct loads and stores...
  for (Value::use_const_iterator UI = AI->use_begin(), UE = AI->use_end();
       UI != UE; ++UI)     // Loop over all of the uses of the alloca
    if (!isa<LoadInst>(*UI))
      if (const StoreInst *SI = dyn_cast<StoreInst>(*UI)) {
        if (SI->getOperand(0) == AI)
          return false;   // Don't allow a store of the AI, only INTO the AI.
      } else {
        return false;   // Not a load or store?
      }
  
  return true;
}


namespace {
  struct PromoteMem2Reg {
    const std::vector<AllocaInst*>   &Allocas;      // the alloca instructions..
    std::vector<unsigned> VersionNumbers;           // Current version counters
    DominanceFrontier &DF;
    const TargetData &TD;

    std::map<Instruction*, unsigned>  AllocaLookup; // reverse mapping of above
    
    std::vector<std::vector<BasicBlock*> > PhiNodes;// Idx corresponds 2 Allocas
    
    // List of instructions to remove at end of pass
    std::vector<Instruction *>        KillList;
    
    std::map<BasicBlock*,
             std::vector<PHINode*> >  NewPhiNodes; // the PhiNodes we're adding

  public:
    PromoteMem2Reg(const std::vector<AllocaInst*> &A, DominanceFrontier &df,
                   const TargetData &td)
      : Allocas(A), DF(df), TD(td) {}

    void run();

  private:
    void RenamePass(BasicBlock *BB, BasicBlock *Pred,
                    std::vector<Value*> &IncVals,
                    std::set<BasicBlock*> &Visited);
    bool QueuePhiNode(BasicBlock *BB, unsigned AllocaIdx);
  };
}  // end of anonymous namespace


void PromoteMem2Reg::run() {
  // If there is nothing to do, bail out...
  if (Allocas.empty()) return;

  Function &F = *DF.getRoot()->getParent();
  VersionNumbers.resize(Allocas.size());

  for (unsigned i = 0, e = Allocas.size(); i != e; ++i) {
    assert(isAllocaPromotable(Allocas[i], TD) &&
           "Cannot promote non-promotable alloca!");
    assert(Allocas[i]->getParent()->getParent() == &F &&
           "All allocas should be in the same function, which is same as DF!");
    AllocaLookup[Allocas[i]] = i;
  }


  // Add each alloca to the KillList.  Note: KillList is destroyed MOST recently
  // added to least recently.
  KillList.assign(Allocas.begin(), Allocas.end());

  // Calculate the set of write-locations for each alloca.  This is analogous to
  // counting the number of 'redefinitions' of each variable.
  std::vector<std::vector<BasicBlock*> > WriteSets;// Idx corresponds to Allocas
  WriteSets.resize(Allocas.size());
  for (unsigned i = 0; i != Allocas.size(); ++i) {
    AllocaInst *AI = Allocas[i];
    for (Value::use_iterator U =AI->use_begin(), E = AI->use_end(); U != E; ++U)
      if (StoreInst *SI = dyn_cast<StoreInst>(*U))
        // jot down the basic-block it came from
        WriteSets[i].push_back(SI->getParent());
  }

  // Compute the locations where PhiNodes need to be inserted.  Look at the
  // dominance frontier of EACH basic-block we have a write in
  //
  PhiNodes.resize(Allocas.size());
  for (unsigned i = 0; i != Allocas.size(); ++i) {
    for (unsigned j = 0; j != WriteSets[i].size(); j++) {
      // Look up the DF for this write, add it to PhiNodes
      DominanceFrontier::const_iterator it = DF.find(WriteSets[i][j]);
      if (it != DF.end()) {
        const DominanceFrontier::DomSetType &S = it->second;
        for (DominanceFrontier::DomSetType::iterator P = S.begin(),PE = S.end();
             P != PE; ++P)
          QueuePhiNode(*P, i);
      }
    }
    
    // Perform iterative step
    for (unsigned k = 0; k != PhiNodes[i].size(); k++) {
      DominanceFrontier::const_iterator it = DF.find(PhiNodes[i][k]);
      if (it != DF.end()) {
        const DominanceFrontier::DomSetType     &S = it->second;
        for (DominanceFrontier::DomSetType::iterator P = S.begin(),PE = S.end();
             P != PE; ++P)
          QueuePhiNode(*P, i);
      }
    }
  }

  // Set the incoming values for the basic block to be null values for all of
  // the alloca's.  We do this in case there is a load of a value that has not
  // been stored yet.  In this case, it will get this null value.
  //
  std::vector<Value *> Values(Allocas.size());
  for (unsigned i = 0, e = Allocas.size(); i != e; ++i)
    Values[i] = Constant::getNullValue(Allocas[i]->getAllocatedType());

  // Walks all basic blocks in the function performing the SSA rename algorithm
  // and inserting the phi nodes we marked as necessary
  //
  std::set<BasicBlock*> Visited;      // The basic blocks we've already visited
  RenamePass(F.begin(), 0, Values, Visited);

  // Remove all instructions marked by being placed in the KillList...
  //
  while (!KillList.empty()) {
    Instruction *I = KillList.back();
    KillList.pop_back();

    // If there are any uses of these instructions left, they must be in
    // sections of dead code that were not processed on the dominance frontier.
    // Just delete the users now.
    //
    while (!I->use_empty()) {
      Instruction *U = cast<Instruction>(I->use_back());
      U->getParent()->getInstList().erase(U);
    }

    I->getParent()->getInstList().erase(I);
  }
}


// QueuePhiNode - queues a phi-node to be added to a basic-block for a specific
// Alloca returns true if there wasn't already a phi-node for that variable
//
bool PromoteMem2Reg::QueuePhiNode(BasicBlock *BB, unsigned AllocaNo) {
  // Look up the basic-block in question
  std::vector<PHINode*> &BBPNs = NewPhiNodes[BB];
  if (BBPNs.empty()) BBPNs.resize(Allocas.size());

  // If the BB already has a phi node added for the i'th alloca then we're done!
  if (BBPNs[AllocaNo]) return false;

  // Create a PhiNode using the dereferenced type... and add the phi-node to the
  // BasicBlock.
  PHINode *PN = new PHINode(Allocas[AllocaNo]->getAllocatedType(),
                            Allocas[AllocaNo]->getName() + "." +
                                      utostr(VersionNumbers[AllocaNo]++),
                            BB->begin());

  // Add null incoming values for all predecessors.  This ensures that if one of
  // the predecessors is not found in the depth-first traversal of the CFG (ie,
  // because it is an unreachable predecessor), that all PHI nodes will have the
  // correct number of entries for their predecessors.
  Value *NullVal = Constant::getNullValue(PN->getType());
  for (pred_iterator PI = pred_begin(BB), PE = pred_end(BB); PI != PE; ++PI)
    PN->addIncoming(NullVal, *PI);

  BBPNs[AllocaNo] = PN;
  PhiNodes[AllocaNo].push_back(BB);
  return true;
}

void PromoteMem2Reg::RenamePass(BasicBlock *BB, BasicBlock *Pred,
                             std::vector<Value*> &IncomingVals,
                             std::set<BasicBlock*> &Visited) {
  // If this is a BB needing a phi node, lookup/create the phinode for each
  // variable we need phinodes for.
  std::vector<PHINode *> &BBPNs = NewPhiNodes[BB];
  for (unsigned k = 0; k != BBPNs.size(); ++k)
    if (PHINode *PN = BBPNs[k]) {
      int BBI = PN->getBasicBlockIndex(Pred);
      assert(BBI >= 0 && "Predecessor not in basic block yet!");

      // At this point we can assume that the array has phi nodes.. let's update
      // the incoming data.
      PN->setIncomingValue(BBI, IncomingVals[k]);

      // also note that the active variable IS designated by the phi node
      IncomingVals[k] = PN;
    }

  // don't revisit nodes
  if (Visited.count(BB)) return;
  
  // mark as visited
  Visited.insert(BB);

  // keep track of the value of each variable we're watching.. how?
  for (BasicBlock::iterator II = BB->begin(); II != BB->end(); ++II) {
    Instruction *I = II; // get the instruction

    if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
      if (AllocaInst *Src = dyn_cast<AllocaInst>(LI->getPointerOperand())) {
        std::map<Instruction*, unsigned>::iterator AI = AllocaLookup.find(Src);
        if (AI != AllocaLookup.end()) {
          Value *V = IncomingVals[AI->second];

          // walk the use list of this load and replace all uses with r
          LI->replaceAllUsesWith(V);
          KillList.push_back(LI); // Mark the load to be deleted
        }
      }
    } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
      // Delete this instruction and mark the name as the current holder of the
      // value
      if (AllocaInst *Dest = dyn_cast<AllocaInst>(SI->getPointerOperand())) {
        std::map<Instruction *, unsigned>::iterator ai =AllocaLookup.find(Dest);
        if (ai != AllocaLookup.end()) {
          // what value were we writing?
          IncomingVals[ai->second] = SI->getOperand(0);
          KillList.push_back(SI);  // Mark the store to be deleted
        }
      }
      
    } else if (TerminatorInst *TI = dyn_cast<TerminatorInst>(I)) {
      // Recurse across our successors
      for (unsigned i = 0; i != TI->getNumSuccessors(); i++) {
        std::vector<Value*> OutgoingVals(IncomingVals);
        RenamePass(TI->getSuccessor(i), BB, OutgoingVals, Visited);
      }
    }
  }
}

/// PromoteMemToReg - Promote the specified list of alloca instructions into
/// scalar registers, inserting PHI nodes as appropriate.  This function makes
/// use of DominanceFrontier information.  This function does not modify the CFG
/// of the function at all.  All allocas must be from the same function.
///
void PromoteMemToReg(const std::vector<AllocaInst*> &Allocas,
                     DominanceFrontier &DF, const TargetData &TD) {
  PromoteMem2Reg(Allocas, DF, TD).run();
}
