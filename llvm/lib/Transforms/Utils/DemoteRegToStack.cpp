//===- DemoteRegToStack.cpp - Move a virtual register to the stack --------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
// This file provide the function DemoteRegToStack().  This function takes a
// virtual register computed by an Instruction& X and replaces it with a slot in
// the stack frame, allocated via alloca. It returns the pointer to the
// AllocaInst inserted.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Function.h"
#include "llvm/iMemory.h"
#include "llvm/iPHINode.h"
#include "llvm/iTerminators.h"
#include "llvm/Type.h"
#include "Support/hash_set"
using namespace llvm;

typedef hash_set<PHINode*>           PhiSet;
typedef hash_set<PHINode*>::iterator PhiSetIterator;

// Helper function to push a phi *and* all its operands to the worklist!
// Do not push an instruction if it is already in the result set of Phis to go.
static inline void PushOperandsOnWorkList(std::vector<Instruction*>& workList,
                                          PhiSet& phisToGo, PHINode* phiN) {
  for (User::op_iterator OI = phiN->op_begin(), OE = phiN->op_end();
       OI != OE; ++OI) {
    Instruction* opI = cast<Instruction>(OI);
    if (!isa<PHINode>(opI) || !phisToGo.count(cast<PHINode>(opI)))
      workList.push_back(opI);
  }
}

static void FindPhis(Instruction& X, PhiSet& phisToGo) {
  std::vector<Instruction*> workList;
  workList.push_back(&X);

  // Handle the case that X itself is a Phi!
  if (PHINode* phiX = dyn_cast<PHINode>(&X)) {
    phisToGo.insert(phiX);
    PushOperandsOnWorkList(workList, phisToGo, phiX);
  }

  // Now use a worklist to find all phis reachable from X, and
  // (recursively) all phis reachable from operands of such phis.
  while (!workList.empty()) {
    Instruction *I = workList.back();
    workList.pop_back();
    for (Value::use_iterator UI = I->use_begin(), E = I->use_end(); UI!=E; ++UI)
      if (PHINode* phiN = dyn_cast<PHINode>(*UI))
        if (phisToGo.find(phiN) == phisToGo.end()) {
          // Seeing this phi for the first time: it must go!
          phisToGo.insert(phiN);
          workList.push_back(phiN);
          PushOperandsOnWorkList(workList, phisToGo, phiN);
        }
  }
}


// Insert loads before all uses of I, except uses in Phis
// since all such Phis *must* be deleted.
static void LoadBeforeUses(Instruction* def, AllocaInst* XSlot) {
  for (unsigned nPhis = 0; def->use_size() - nPhis > 0; ) {
      Instruction* useI = cast<Instruction>(def->use_back());
      if (!isa<PHINode>(useI)) {
        LoadInst* loadI =
          new LoadInst(XSlot, std::string("Load")+XSlot->getName(), useI);
        useI->replaceUsesOfWith(def, loadI);
      } else
        ++nPhis;
  }
}

static void AddLoadsAndStores(AllocaInst* XSlot, Instruction& X,
                              PhiSet& phisToGo) {
  for (PhiSetIterator PI=phisToGo.begin(), PE=phisToGo.end(); PI != PE; ++PI) {
    PHINode* pn = *PI;

    // First, insert loads before all uses except uses in Phis.
    // Do this first because new stores will appear as uses also!
    LoadBeforeUses(pn, XSlot);

    // For every incoming operand of the Phi, insert a store either
    // just after the instruction defining the value or just before the
    // predecessor of the Phi if the value is a formal, not an instruction.
    // 
    for (unsigned i=0, N=pn->getNumIncomingValues(); i < N; ++i) {
      Value* phiOp = pn->getIncomingValue(i);
      if (phiOp != &X &&
          (!isa<PHINode>(phiOp) || !phisToGo.count(cast<PHINode>(phiOp)))) {
        // This operand is not a phi that will be deleted: need to store.
        assert(!isa<TerminatorInst>(phiOp));

        Instruction* storeBefore;
        if (Instruction* I = dyn_cast<Instruction>(phiOp)) {
          // phiOp is an instruction, store its result right after it.
          assert(I->getNext() && "Non-terminator without successor?");
          storeBefore = I->getNext();
        } else {
          // If not, it must be a formal: store it at the end of the
          // predecessor block of the Phi (*not* at function entry!).
          storeBefore = pn->getIncomingBlock(i)->getTerminator();
        }
              
        // Create instr. to store the value of phiOp before `insertBefore'
        StoreInst* storeI = new StoreInst(phiOp, XSlot, storeBefore);
      }
    }
  }
}

//---------------------------------------------------------------------------- 
// function DemoteRegToStack()
// 
// This function takes a virtual register computed by an
// Instruction& X and replaces it with a slot in the stack frame,
// allocated via alloca.  It has to:
// (1) Identify all Phi operations that have X as an operand and
//     transitively other Phis that use such Phis; 
// (2) Store all values merged with X via Phi operations to the stack slot;
// (3) Load the value from the stack slot just before any use of X or any
//     of the Phis that were eliminated; and
// (4) Delete all the Phis, which should all now be dead.
//
// Returns the pointer to the alloca inserted to create a stack slot for X.
//
AllocaInst* llvm::DemoteRegToStack(Instruction& X) {
  if (X.getType() == Type::VoidTy)
    return 0;                             // nothing to do!

  // Find all Phis involving X or recursively using such Phis or Phis
  // involving operands of such Phis (essentially all Phis in the "web" of X)
  PhiSet phisToGo;
  FindPhis(X, phisToGo);

  // Create a stack slot to hold X
  Function* parentFunc = X.getParent()->getParent();
  AllocaInst *XSlot = new AllocaInst(X.getType(), 0, X.getName(),
                                     parentFunc->getEntryBlock().begin());


  // Insert loads before all uses of X and (*only then*) insert store after X
  assert(X.getNext() && "Non-terminator (since non-void) with no successor?");
  LoadBeforeUses(&X, XSlot);
  StoreInst* storeI = new StoreInst(&X, XSlot, X.getNext());

  // Do the same for all the phis that will be deleted
  AddLoadsAndStores(XSlot, X, phisToGo);

  // Delete the phis and return the alloca instruction
  for (PhiSetIterator PI = phisToGo.begin(), E = phisToGo.end(); PI != E; ++PI)
    (*PI)->getParent()->getInstList().erase(*PI);

  return XSlot;
}
