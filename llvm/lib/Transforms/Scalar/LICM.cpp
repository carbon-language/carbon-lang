//===-- LICM.cpp - Loop Invariant Code Motion Pass ------------------------===//
//
// This pass is a simple loop invariant code motion pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/iOperators.h"
#include "llvm/iMemory.h"
#include "llvm/Support/InstVisitor.h"
#include "Support/STLExtras.h"
#include "Support/StatisticReporter.h"
#include <algorithm>
using std::string;

namespace {
  Statistic<>NumHoisted("licm\t\t- Number of instructions hoisted out of loop");
  Statistic<> NumHoistedLoads("licm\t\t- Number of load insts hoisted");

  struct LICM : public FunctionPass, public InstVisitor<LICM> {
    virtual bool runOnFunction(Function &F);

    /// This transformation requires natural loop information & requires that
    /// loop preheaders be inserted into the CFG...
    ///
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.preservesCFG();
      AU.addRequiredID(LoopPreheadersID);
      AU.addRequired<LoopInfo>();
      AU.addRequired<DominatorTree>();
      AU.addRequired<AliasAnalysis>();
    }

  private:
    Loop *CurLoop;         // The current loop we are working on...
    BasicBlock *Preheader; // The preheader block of the current loop...
    bool Changed;          // Set to true when we change anything.
    AliasAnalysis *AA;     // Currently AliasAnalysis information

    /// visitLoop - Hoist expressions out of the specified loop...    
    ///
    void visitLoop(Loop *L);

    /// HoistRegion - Walk the specified region of the CFG (defined by all
    /// blocks dominated by the specified block, and that are in the current
    /// loop) in depth first order w.r.t the DominatorTree.  This allows us to
    /// visit defintions before uses, allowing us to hoist a loop body in one
    /// pass without iteration.
    ///
    void HoistRegion(DominatorTree::Node *N);

    /// inCurrentLoop - Little predicate that returns false if the specified
    /// basic block is in a subloop of the current one, not the current one
    /// itself.
    ///
    bool inCurrentLoop(BasicBlock *BB) {
      for (unsigned i = 0, e = CurLoop->getSubLoops().size(); i != e; ++i)
        if (CurLoop->getSubLoops()[i]->contains(BB))
          return false;  // A subloop actually contains this block!
      return true;
    }

    /// hoist - When an instruction is found to only use loop invariant operands
    /// that is safe to hoist, this instruction is called to do the dirty work.
    ///
    void hoist(Instruction &I);

    /// pointerInvalidatedByLoop - Return true if the body of this loop may
    /// store into the memory location pointed to by V.
    /// 
    bool pointerInvalidatedByLoop(Value *V);

    /// isLoopInvariant - Return true if the specified value is loop invariant
    ///
    inline bool isLoopInvariant(Value *V) {
      if (Instruction *I = dyn_cast<Instruction>(V))
        return !CurLoop->contains(I->getParent());
      return true;  // All non-instructions are loop invariant
    }

    /// Instruction visitation handlers... these basically control whether or
    /// not the specified instruction types are hoisted.
    ///
    friend class InstVisitor<LICM>;
    void visitBinaryOperator(Instruction &I) {
      if (isLoopInvariant(I.getOperand(0)) && isLoopInvariant(I.getOperand(1)))
        hoist(I);
    }
    void visitCastInst(CastInst &CI) {
      Instruction &I = (Instruction&)CI;
      if (isLoopInvariant(I.getOperand(0))) hoist(I);
    }
    void visitShiftInst(ShiftInst &I) { visitBinaryOperator((Instruction&)I); }

    void visitLoadInst(LoadInst &LI);

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

/// runOnFunction - For LICM, this simply traverses the loop structure of the
/// function, hoisting expressions out of loops if possible.
///
bool LICM::runOnFunction(Function &) {
  // Get information about the top level loops in the function...
  const std::vector<Loop*> &TopLevelLoops =
    getAnalysis<LoopInfo>().getTopLevelLoops();

  // Get our alias analysis information...
  AA = &getAnalysis<AliasAnalysis>();

  // Traverse loops in postorder, hoisting expressions out of the deepest loops
  // first.
  //
  Changed = false;
  std::for_each(TopLevelLoops.begin(), TopLevelLoops.end(),
                bind_obj(this, &LICM::visitLoop));
  return Changed;
}


/// visitLoop - Hoist expressions out of the specified loop...    
///
void LICM::visitLoop(Loop *L) {
  // Recurse through all subloops before we process this loop...
  std::for_each(L->getSubLoops().begin(), L->getSubLoops().end(),
                bind_obj(this, &LICM::visitLoop));
  CurLoop = L;

  // Get the preheader block to move instructions into...
  Preheader = L->getLoopPreheader();
  assert(Preheader&&"Preheader insertion pass guarantees we have a preheader!");

  // We want to visit all of the instructions in this loop... that are not parts
  // of our subloops (they have already had their invariants hoisted out of
  // their loop, into this loop, so there is no need to process the BODIES of
  // the subloops).
  //
  // Traverse the body of the loop in depth first order on the dominator tree so
  // that we are guaranteed to see definitions before we see uses.  This allows
  // us to perform the LICM transformation in one pass, without iteration.
  //
  HoistRegion(getAnalysis<DominatorTree>()[L->getHeader()]);

  // Clear out loops state information for the next iteration
  CurLoop = 0;
  Preheader = 0;
}

/// HoistRegion - Walk the specified region of the CFG (defined by all blocks
/// dominated by the specified block, and that are in the current loop) in depth
/// first order w.r.t the DominatorTree.  This allows us to visit defintions
/// before uses, allowing us to hoist a loop body in one pass without iteration.
///
void LICM::HoistRegion(DominatorTree::Node *N) {
  assert(N != 0 && "Null dominator tree node?");

  // This subregion is not in the loop, it has already been already been hoisted
  if (!inCurrentLoop(N->getNode()))
    return;

  visit(*N->getNode());

  const std::vector<DominatorTree::Node*> &Children = N->getChildren();
  for (unsigned i = 0, e = Children.size(); i != e; ++i)
    HoistRegion(Children[i]);
}


/// hoist - When an instruction is found to only use loop invariant operands
/// that is safe to hoist, this instruction is called to do the dirty work.
///
void LICM::hoist(Instruction &Inst) {
  DEBUG(std::cerr << "LICM hoisting: " << Inst);

  BasicBlock *Header = CurLoop->getHeader();

  // Remove the instruction from its current basic block... but don't delete the
  // instruction.
  Inst.getParent()->getInstList().remove(&Inst);

  // Insert the new node in Preheader, before the terminator.
  Preheader->getInstList().insert(Preheader->getTerminator(), &Inst);
  
  ++NumHoisted;
  Changed = true;
}


void LICM::visitLoadInst(LoadInst &LI) {
  if (isLoopInvariant(LI.getOperand(0)) &&
      !pointerInvalidatedByLoop(LI.getOperand(0))) {
    hoist(LI);
    ++NumHoistedLoads;
  }
}

/// pointerInvalidatedByLoop - Return true if the body of this loop may store
/// into the memory location pointed to by V.
/// 
bool LICM::pointerInvalidatedByLoop(Value *V) {
  // Check to see if any of the basic blocks in CurLoop invalidate V.
  for (unsigned i = 0, e = CurLoop->getBlocks().size(); i != e; ++i)
    if (AA->canBasicBlockModify(*CurLoop->getBlocks()[i], V))
      return true;
  return false;
}
