//===-- LICM.cpp - Loop Invariant Code Motion Pass ------------------------===//
//
// This pass is a simple loop invariant code motion pass.
//
// Note that this pass does NOT require pre-headers to exist on loops in the
// CFG, but if there is not distinct preheader for a loop, the hoisted code will
// be *DUPLICATED* in every basic block, outside of the loop, that preceeds the
// loop header.  Additionally, any use of one of these hoisted expressions
// cannot be loop invariant itself, because the expression hoisted gets a PHI
// node that is loop variant.
//
// For these reasons, and many more, it makes sense to run a pass before this
// that ensures that there are preheaders on all loops.  That said, we don't
// REQUIRE it. :)
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/iOperators.h"
#include "llvm/iPHINode.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Support/CFG.h"
#include "Support/STLExtras.h"
#include "Support/StatisticReporter.h"
#include <algorithm>
using std::string;

static Statistic<> NumHoistedNPH("licm\t\t- Number of insts hoisted to multiple"
                                 " loop preds (bad, no loop pre-header)");
static Statistic<> NumHoistedPH("licm\t\t- Number of insts hoisted to a loop "
                                "pre-header");

namespace {
  struct LICM : public FunctionPass, public InstVisitor<LICM> {
    virtual bool runOnFunction(Function &F);

    // This transformation requires natural loop information...
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.preservesCFG();
      AU.addRequired<LoopInfo>();
    }

  private:
    // List of predecessor blocks for the current loop - These blocks are where
    // we hoist loop invariants to for the current loop.
    //
    std::vector<BasicBlock*> LoopPreds, LoopBackEdges;

    Loop *CurLoop;  // The current loop we are working on...
    bool Changed;   // Set to true when we change anything.

    // visitLoop - Hoist expressions out of the specified loop...    
    void visitLoop(Loop *L);

    // notInCurrentLoop - Little predicate that returns true if the specified
    // basic block is in a subloop of the current one, not the current one
    // itself.
    //
    bool notInCurrentLoop(BasicBlock *BB) {
      for (unsigned i = 0, e = CurLoop->getSubLoops().size(); i != e; ++i)
        if (CurLoop->getSubLoops()[i]->contains(BB))
          return true;  // A subloop actually contains this block!
      return false;      
    }

    // hoist - When an instruction is found to only use loop invariant operands
    // that is safe to hoist, this instruction is called to do the dirty work.
    //
    void hoist(Instruction &I);

    // isLoopInvariant - Return true if the specified value is loop invariant
    inline bool isLoopInvariant(Value *V) {
      if (Instruction *I = dyn_cast<Instruction>(V))
        return !CurLoop->contains(I->getParent());
      return true;  // All non-instructions are loop invariant
    }

    // visitBasicBlock - Run LICM on a particular block.
    void visitBasicBlock(BasicBlock *BB);

    // Instruction visitation handlers... these basically control whether or not
    // the specified instruction types are hoisted.
    //
    friend class InstVisitor<LICM>;
    void visitBinaryOperator(Instruction &I) {
      if (isLoopInvariant(I.getOperand(0)) && isLoopInvariant(I.getOperand(1)))
        hoist(I);
    }
    void visitCastInst(CastInst &I) {
      if (isLoopInvariant(I.getOperand(0))) hoist((Instruction&)I);
    }
    void visitShiftInst(ShiftInst &I) { visitBinaryOperator((Instruction&)I); }

    void visitGetElementPtrInst(GetElementPtrInst &GEPI) {
      Instruction &I = (Instruction&)GEPI;
      for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i)
        if (!isLoopInvariant(I.getOperand(i))) return;
      hoist(I);
    }
  };

  RegisterOpt<LICM> X("licm", "Loop Invariant Code Motion");
}

Pass *createLICMPass() { return new LICM(); }

bool LICM::runOnFunction(Function &) {
  // get our loop information...
  const std::vector<Loop*> &TopLevelLoops =
    getAnalysis<LoopInfo>().getTopLevelLoops();

  // Traverse loops in postorder, hoisting expressions out of the deepest loops
  // first.
  //
  Changed = false;
  std::for_each(TopLevelLoops.begin(), TopLevelLoops.end(),
                bind_obj(this, &LICM::visitLoop));
  return Changed;
}

void LICM::visitLoop(Loop *L) {
  // Recurse through all subloops before we process this loop...
  std::for_each(L->getSubLoops().begin(), L->getSubLoops().end(),
                bind_obj(this, &LICM::visitLoop));
  CurLoop = L;

  // Calculate the set of predecessors for this loop.  The predecessors for this
  // loop are equal to the predecessors for the header node of the loop that are
  // not themselves in the loop.
  //
  BasicBlock *Header = L->getHeader();

  // Calculate the sets of predecessors and backedges of the loop...
  LoopBackEdges.insert(LoopBackEdges.end(),pred_begin(Header),pred_end(Header));

  std::vector<BasicBlock*>::iterator LPI =
    std::partition(LoopBackEdges.begin(), LoopBackEdges.end(),
                   bind_obj(CurLoop, &Loop::contains));

  // Move all predecessors to the LoopPreds vector...
  LoopPreds.insert(LoopPreds.end(), LPI, LoopBackEdges.end());

  // Remove predecessors from backedges list...
  LoopBackEdges.erase(LPI, LoopBackEdges.end());
 

  // The only way that there could be no predecessors to a loop is if the loop
  // is not reachable.  Since we don't care about optimizing dead loops,
  // summarily ignore them.
  //
  if (LoopPreds.empty()) return;
  
  // We want to visit all of the instructions in this loop... that are not parts
  // of our subloops (they have already had their invariants hoisted out of
  // their loop, into this loop, so there is no need to process the BODIES of
  // the subloops).
  //
  std::vector<BasicBlock*> BBs(L->getBlocks().begin(), L->getBlocks().end());

  // Remove blocks that are actually in subloops...
  BBs.erase(std::remove_if(BBs.begin(), BBs.end(), 
                           bind_obj(this, &LICM::notInCurrentLoop)), BBs.end());

  // Visit all of the basic blocks we have chosen, hoisting out the instructions
  // as neccesary.  This leaves dead copies of the instruction in the loop
  // unfortunately...
  //
  for_each(BBs.begin(), BBs.end(), bind_obj(this, &LICM::visitBasicBlock));

  // Clear out loops state information for the next iteration
  CurLoop = 0;
  LoopPreds.clear();
  LoopBackEdges.clear();
}

void LICM::visitBasicBlock(BasicBlock *BB) {
  for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ) {
    visit(*I);
    
    if (dceInstruction(I))
      Changed = true;
    else
      ++I;
  }
}


void LICM::hoist(Instruction &Inst) {
  if (Inst.use_empty()) return;  // Don't (re) hoist dead instructions!
  //cerr << "Hoisting " << Inst;

  BasicBlock *Header = CurLoop->getHeader();

  // Old instruction will be removed, so take it's name...
  string InstName = Inst.getName();
  Inst.setName("");

  // The common case is that we have a pre-header.  Generate special case code
  // that is faster if that is the case.
  //
  if (LoopPreds.size() == 1) {
    BasicBlock *Pred = LoopPreds[0];

    // Create a new copy of the instruction, for insertion into Pred.
    Instruction *New = Inst.clone();
    New->setName(InstName);

    // Insert the new node in Pred, before the terminator.
    Pred->getInstList().insert(--Pred->end(), New);

    // Kill the old instruction...
    Inst.replaceAllUsesWith(New);
    ++NumHoistedPH;

  } else {
    // No loop pre-header, insert a PHI node into header to capture all of the
    // incoming versions of the value.
    //
    PHINode *LoopVal = new PHINode(Inst.getType(), InstName+".phi");

    // Insert the new PHI node into the loop header...
    Header->getInstList().push_front(LoopVal);

    // Insert cloned versions of the instruction into all of the loop preds.
    for (unsigned i = 0, e = LoopPreds.size(); i != e; ++i) {
      BasicBlock *Pred = LoopPreds[i];
      
      // Create a new copy of the instruction, for insertion into Pred.
      Instruction *New = Inst.clone();
      New->setName(InstName);

      // Insert the new node in Pred, before the terminator.
      Pred->getInstList().insert(--Pred->end(), New);

      // Add the incoming value to the PHI node.
      LoopVal->addIncoming(New, Pred);
    }

    // Add incoming values to the PHI node for all backedges in the loop...
    for (unsigned i = 0, e = LoopBackEdges.size(); i != e; ++i)
      LoopVal->addIncoming(LoopVal, LoopBackEdges[i]);

    // Replace all uses of the old version of the instruction in the loop with
    // the new version that is out of the loop.  We know that this is ok,
    // because the new definition is in the loop header, which dominates the
    // entire loop body.  The old definition was defined _inside_ of the loop,
    // so the scope cannot extend outside of the loop, so we're ok.
    //
    Inst.replaceAllUsesWith(LoopVal);
    ++NumHoistedNPH;
  }

  Changed = true;
}

