//===-- LICM.cpp - Loop Invariant Code Motion Pass ------------------------===//
//
// This pass is a simple loop invariant code motion pass.  An interesting aspect
// of this pass is that it uses alias analysis for two purposes:
//
//  1. Moving loop invariant loads out of loops.  If we can determine that a
//     load inside of a loop never aliases anything stored to, we can hoist it
//     like any other instruction.
//  2. Scalar Promotion of Memory - If there is a store instruction inside of
//     the loop, we try to move the store to happen AFTER the loop instead of
//     inside of the loop.  This can only happen if a few conditions are true:
//       A. The pointer stored through is loop invariant
//       B. There are no stores or loads in the loop which _may_ alias the
//          pointer.  There are no calls in the loop which mod/ref the pointer.
//     If these conditions are true, we can promote the loads and stores in the
//     loop of the pointer to use a temporary alloca'd variable.  We then use
//     the mem2reg functionality to construct the appropriate SSA form for the
//     variable.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Instructions.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Support/CFG.h"
#include "Support/Statistic.h"
#include "Support/CommandLine.h"
#include "llvm/Assembly/Writer.h"
#include <algorithm>

namespace {
  cl::opt<bool> DisablePromotion("disable-licm-promotion", cl::Hidden,
                             cl::desc("Disable memory promotion in LICM pass"));

  Statistic<> NumHoisted("licm", "Number of instructions hoisted out of loop");
  Statistic<> NumHoistedLoads("licm", "Number of load insts hoisted");
  Statistic<> NumPromoted("licm", "Number of memory locations promoted to registers");

  /// LoopBodyInfo - We recursively traverse loops from most-deeply-nested to
  /// least-deeply-nested.  For all of the loops nested within the current one,
  /// we keep track of information so that we don't have to repeat queries.
  ///
  struct LoopBodyInfo {
    std::vector<CallInst*> Calls;          // Call instructions in loop
    std::vector<InvokeInst*> Invokes;      // Invoke instructions in loop

    // StoredPointers - Targets of store instructions...
    std::set<Value*> StoredPointers;

    // LoadedPointers - Source pointers for load instructions...
    std::set<Value*> LoadedPointers;

    enum PointerClass {
      PointerUnknown = 0, // Nothing is known about this pointer yet
      PointerMustStore,   // Memory is stored to ONLY through this pointer
      PointerMayStore,    // Memory is stored to through this or other pointers
      PointerNoStore      // Memory is not modified in this loop
    };

    // PointerIsModified - Keep track of information as we find out about it in
    // the loop body...
    //
    std::map<Value*, enum PointerClass> PointerIsModified;

    /// CantModifyAnyPointers - Return true if no memory modifying instructions
    /// occur in this loop.  This is just a conservative approximation, because
    /// a call may not actually store anything.
    bool CantModifyAnyPointers() const {
      return Calls.empty() && Invokes.empty() && StoredPointers.empty();
    }

    /// incorporate - Incorporate information about a subloop into the current
    /// loop.
    void incorporate(const LoopBodyInfo &OtherLBI);
    void incorporate(BasicBlock &BB);  // do the same for a basic block

    PointerClass getPointerInfo(Value *V, AliasAnalysis &AA) {
      PointerClass &VInfo = PointerIsModified[V];
      if (VInfo == PointerUnknown)
        VInfo = calculatePointerInfo(V, AA);
      return VInfo;
    }
  private:
    /// calculatePointerInfo - Calculate information about the specified
    /// pointer.
    PointerClass calculatePointerInfo(Value *V, AliasAnalysis &AA) const;
  };
}

/// incorporate - Incorporate information about a subloop into the current loop.
void LoopBodyInfo::incorporate(const LoopBodyInfo &OtherLBI) {
  // Do not incorporate NonModifiedPointers (which is just a cache) because it
  // is too much trouble to make sure it's still valid.
  Calls.insert  (Calls.end(),  OtherLBI.Calls.begin(),  OtherLBI.Calls.end());
  Invokes.insert(Invokes.end(),OtherLBI.Invokes.begin(),OtherLBI.Invokes.end());
  StoredPointers.insert(OtherLBI.StoredPointers.begin(),
                        OtherLBI.StoredPointers.end());
  LoadedPointers.insert(OtherLBI.LoadedPointers.begin(),
                        OtherLBI.LoadedPointers.end());
}

void LoopBodyInfo::incorporate(BasicBlock &BB) {
  for (BasicBlock::iterator I = BB.begin(), E = --BB.end(); I != E; ++I)
    if (CallInst *CI = dyn_cast<CallInst>(&*I))
      Calls.push_back(CI);
    else if (StoreInst *SI = dyn_cast<StoreInst>(&*I))
      StoredPointers.insert(SI->getOperand(1));
    else if (LoadInst *LI = dyn_cast<LoadInst>(&*I))
      LoadedPointers.insert(LI->getOperand(0));

  if (InvokeInst *II = dyn_cast<InvokeInst>(BB.getTerminator()))
    Invokes.push_back(II);
}


// calculatePointerInfo - Calculate information about the specified pointer.
LoopBodyInfo::PointerClass LoopBodyInfo::calculatePointerInfo(Value *V,
                                                      AliasAnalysis &AA) const {
  for (unsigned i = 0, e = Calls.size(); i != e; ++i)
    if (AA.getModRefInfo(Calls[i], V, ~0))
      return PointerMayStore;

  for (unsigned i = 0, e = Invokes.size(); i != e; ++i)
    if (AA.getModRefInfo(Invokes[i], V, ~0))
      return PointerMayStore;

  PointerClass Result = PointerNoStore;
  for (std::set<Value*>::const_iterator I = StoredPointers.begin(),
         E = StoredPointers.end(); I != E; ++I)
    if (AA.alias(V, ~0, *I, ~0))
      if (V == *I)
        Result = PointerMustStore;   // If this is the only alias, return must
      else
        return PointerMayStore;      // We have to return may
  return Result;
}

namespace {
  struct LICM : public FunctionPass, public InstVisitor<LICM> {
    virtual bool runOnFunction(Function &F);

    /// This transformation requires natural loop information & requires that
    /// loop preheaders be inserted into the CFG...
    ///
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequiredID(LoopPreheadersID);
      AU.addRequired<LoopInfo>();
      AU.addRequired<DominatorTree>();
      AU.addRequired<DominanceFrontier>();
      AU.addRequired<AliasAnalysis>();
    }

  private:
    LoopInfo      *LI;       // Current LoopInfo
    AliasAnalysis *AA;       // Current AliasAnalysis information
    bool Changed;            // Set to true when we change anything.
    BasicBlock *Preheader;   // The preheader block of the current loop...
    Loop *CurLoop;           // The current loop we are working on...
    LoopBodyInfo *CurLBI;    // Information about the current loop...

    /// visitLoop - Hoist expressions out of the specified loop...    
    ///
    void visitLoop(Loop *L, LoopBodyInfo &LBI);

    /// HoistRegion - Walk the specified region of the CFG (defined by all
    /// blocks dominated by the specified block, and that are in the current
    /// loop) in depth first order w.r.t the DominatorTree.  This allows us to
    /// visit defintions before uses, allowing us to hoist a loop body in one
    /// pass without iteration.
    ///
    void HoistRegion(DominatorTree::Node *N);

    /// inSubLoop - Little predicate that returns true if the specified basic
    /// block is in a subloop of the current one, not the current one itself.
    ///
    bool inSubLoop(BasicBlock *BB) {
      assert(CurLoop->contains(BB) && "Only valid if BB is IN the loop");
      for (unsigned i = 0, e = CurLoop->getSubLoops().size(); i != e; ++i)
        if (CurLoop->getSubLoops()[i]->contains(BB))
          return true;  // A subloop actually contains this block!
      return false;
    }

    /// hoist - When an instruction is found to only use loop invariant operands
    /// that is safe to hoist, this instruction is called to do the dirty work.
    ///
    void hoist(Instruction &I);

    /// pointerInvalidatedByLoop - Return true if the body of this loop may
    /// store into the memory location pointed to by V.
    /// 
    bool pointerInvalidatedByLoop(Value *V) {
      // Check to see if any of the basic blocks in CurLoop invalidate V.
      return CurLBI->getPointerInfo(V, *AA) != LoopBodyInfo::PointerNoStore;
    }

    /// isLoopInvariant - Return true if the specified value is loop invariant
    ///
    inline bool isLoopInvariant(Value *V) {
      if (Instruction *I = dyn_cast<Instruction>(V))
        return !CurLoop->contains(I->getParent());
      return true;  // All non-instructions are loop invariant
    }

    /// PromoteValuesInLoop - Look at the stores in the loop and promote as many
    /// to scalars as we can.
    ///
    void PromoteValuesInLoop();

    /// findPromotableValuesInLoop - Check the current loop for stores to
    /// definate pointers, which are not loaded and stored through may aliases.
    /// If these are found, create an alloca for the value, add it to the
    /// PromotedValues list, and keep track of the mapping from value to
    /// alloca...
    ///
    void findPromotableValuesInLoop(
                   std::vector<std::pair<AllocaInst*, Value*> > &PromotedValues,
                                    std::map<Value*, AllocaInst*> &Val2AlMap);
    

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
  Changed = false;

  // Get our Loop and Alias Analysis information...
  LI = &getAnalysis<LoopInfo>();
  AA = &getAnalysis<AliasAnalysis>();

  // Hoist expressions out of all of the top-level loops.
  const std::vector<Loop*> &TopLevelLoops = LI->getTopLevelLoops();
  for (std::vector<Loop*>::const_iterator I = TopLevelLoops.begin(),
         E = TopLevelLoops.end(); I != E; ++I) {
    LoopBodyInfo LBI;
    LICM::visitLoop(*I, LBI);
  }
  return Changed;
}


/// visitLoop - Hoist expressions out of the specified loop...    
///
void LICM::visitLoop(Loop *L, LoopBodyInfo &LBI) {
  // Recurse through all subloops before we process this loop...
  for (std::vector<Loop*>::const_iterator I = L->getSubLoops().begin(),
         E = L->getSubLoops().end(); I != E; ++I) {
    LoopBodyInfo SubLBI;
    LICM::visitLoop(*I, SubLBI);

    // Incorporate information about the subloops into this loop...
    LBI.incorporate(SubLBI);
  }
  CurLoop = L;
  CurLBI = &LBI;

  // Get the preheader block to move instructions into...
  Preheader = L->getLoopPreheader();
  assert(Preheader&&"Preheader insertion pass guarantees we have a preheader!");

  // Loop over the body of this loop, looking for calls, invokes, and stores.
  // Because subloops have already been incorporated into LBI, we skip blocks in
  // subloops.
  //
  const std::vector<BasicBlock*> &LoopBBs = L->getBlocks();
  for (std::vector<BasicBlock*>::const_iterator I = LoopBBs.begin(),
         E = LoopBBs.end(); I != E; ++I)
    if (LI->getLoopFor(*I) == L)        // Ignore blocks in subloops...
      LBI.incorporate(**I);             // Incorporate the specified basic block

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

  // Now that all loop invariants have been removed from the loop, promote any
  // memory references to scalars that we can...
  if (!DisablePromotion)
    PromoteValuesInLoop();

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

  // If this subregion is not in the top level loop at all, exit.
  if (!CurLoop->contains(N->getNode())) return;

  // Only need to hoist the contents of this block if it is not part of a
  // subloop (which would already have been hoisted)
  if (!inSubLoop(N->getNode()))
    visit(*N->getNode());

  const std::vector<DominatorTree::Node*> &Children = N->getChildren();
  for (unsigned i = 0, e = Children.size(); i != e; ++i)
    HoistRegion(Children[i]);
}


/// hoist - When an instruction is found to only use loop invariant operands
/// that is safe to hoist, this instruction is called to do the dirty work.
///
void LICM::hoist(Instruction &Inst) {
  DEBUG(std::cerr << "LICM hoisting to";
        WriteAsOperand(std::cerr, Preheader, false);
        std::cerr << ": " << Inst);

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

/// PromoteValuesInLoop - Try to promote memory values to scalars by sinking
/// stores out of the loop and moving loads to before the loop.  We do this by
/// looping over the stores in the loop, looking for stores to Must pointers
/// which are loop invariant.  We promote these memory locations to use allocas
/// instead.  These allocas can easily be raised to register values by the
/// PromoteMem2Reg functionality.
///
void LICM::PromoteValuesInLoop() {
  // PromotedValues - List of values that are promoted out of the loop.  Each
  // value has an alloca instruction for it, and a cannonical version of the
  // pointer.
  std::vector<std::pair<AllocaInst*, Value*> > PromotedValues;
  std::map<Value*, AllocaInst*> ValueToAllocaMap; // Map of ptr to alloca

  findPromotableValuesInLoop(PromotedValues, ValueToAllocaMap);
  if (ValueToAllocaMap.empty()) return;   // If there are values to promote...

  Changed = true;
  NumPromoted += PromotedValues.size();

  // Emit a copy from the value into the alloca'd value in the loop preheader
  TerminatorInst *LoopPredInst = Preheader->getTerminator();
  for (unsigned i = 0, e = PromotedValues.size(); i != e; ++i) {
    // Load from the memory we are promoting...
    LoadInst *LI = new LoadInst(PromotedValues[i].second, 
                                PromotedValues[i].second->getName()+".promoted",
                                LoopPredInst);
    // Store into the temporary alloca...
    new StoreInst(LI, PromotedValues[i].first, LoopPredInst);
  }
  
  // Scan the basic blocks in the loop, replacing uses of our pointers with
  // uses of the allocas in question.  If we find a branch that exits the
  // loop, make sure to put reload code into all of the successors of the
  // loop.
  //
  const std::vector<BasicBlock*> &LoopBBs = CurLoop->getBlocks();
  for (std::vector<BasicBlock*>::const_iterator I = LoopBBs.begin(),
         E = LoopBBs.end(); I != E; ++I) {
    // Rewrite all loads and stores in the block of the pointer...
    for (BasicBlock::iterator II = (*I)->begin(), E = (*I)->end();
         II != E; ++II) {
      if (LoadInst *L = dyn_cast<LoadInst>(&*II)) {
        std::map<Value*, AllocaInst*>::iterator
          I = ValueToAllocaMap.find(L->getOperand(0));
        if (I != ValueToAllocaMap.end())
          L->setOperand(0, I->second);    // Rewrite load instruction...
      } else if (StoreInst *S = dyn_cast<StoreInst>(&*II)) {
        std::map<Value*, AllocaInst*>::iterator
          I = ValueToAllocaMap.find(S->getOperand(1));
        if (I != ValueToAllocaMap.end())
          S->setOperand(1, I->second);    // Rewrite store instruction...
      }
    }

    // Check to see if any successors of this block are outside of the loop.
    // If so, we need to copy the value from the alloca back into the memory
    // location...
    //
    for (succ_iterator SI = succ_begin(*I), SE = succ_end(*I); SI != SE; ++SI)
      if (!CurLoop->contains(*SI)) {
        // Copy all of the allocas into their memory locations...
        BasicBlock::iterator BI = (*SI)->begin();
        while (isa<PHINode>(*BI))
          ++BI;             // Skip over all of the phi nodes in the block...
        Instruction *InsertPos = BI;
        for (unsigned i = 0, e = PromotedValues.size(); i != e; ++i) {
          // Load from the alloca...
          LoadInst *LI = new LoadInst(PromotedValues[i].first, "", InsertPos);
          // Store into the memory we promoted...
          new StoreInst(LI, PromotedValues[i].second, InsertPos);
        }
      }
  }

  // Now that we have done the deed, use the mem2reg functionality to promote
  // all of the new allocas we just created into real SSA registers...
  //
  std::vector<AllocaInst*> PromotedAllocas;
  PromotedAllocas.reserve(PromotedValues.size());
  for (unsigned i = 0, e = PromotedValues.size(); i != e; ++i)
    PromotedAllocas.push_back(PromotedValues[i].first);
  PromoteMemToReg(PromotedAllocas, getAnalysis<DominanceFrontier>());
}

/// findPromotableValuesInLoop - Check the current loop for stores to definate
/// pointers, which are not loaded and stored through may aliases.  If these are
/// found, create an alloca for the value, add it to the PromotedValues list,
/// and keep track of the mapping from value to alloca...
///
void LICM::findPromotableValuesInLoop(
                   std::vector<std::pair<AllocaInst*, Value*> > &PromotedValues,
                             std::map<Value*, AllocaInst*> &ValueToAllocaMap) {
  Instruction *FnStart = CurLoop->getHeader()->getParent()->begin()->begin();

  for (std::set<Value*>::iterator I = CurLBI->StoredPointers.begin(),
         E = CurLBI->StoredPointers.end(); I != E; ++I) {
    Value *V = *I;
    if (isLoopInvariant(V) &&
        CurLBI->getPointerInfo(V, *AA) == LoopBodyInfo::PointerMustStore) {

      // Don't add a new entry for this stored pointer if it aliases something
      // we have already processed.
      std::map<Value*, AllocaInst*>::iterator V2AMI = 
        ValueToAllocaMap.lower_bound(V);
      if (V2AMI == ValueToAllocaMap.end() || V2AMI->first != V) {
        // Check to make sure that any loads in the loop are either NO or MUST
        // aliases.  We cannot rewrite loads that _might_ come from this memory
        // location.

        bool PointerOk = true;
        for (std::set<Value*>::const_iterator I =CurLBI->LoadedPointers.begin(),
               E = CurLBI->LoadedPointers.end(); I != E; ++I)
          if (AA->alias(V, ~0, *I, ~0) == AliasAnalysis::MayAlias) {
            PointerOk = false;
            break;
          }

        if (PointerOk) {
          const Type *Ty = cast<PointerType>(V->getType())->getElementType();
          AllocaInst *AI = new AllocaInst(Ty, 0, V->getName()+".tmp", FnStart);
          PromotedValues.push_back(std::make_pair(AI, V));
          ValueToAllocaMap.insert(V2AMI, std::make_pair(V, AI));

          DEBUG(std::cerr << "LICM: Promoting value: " << *V << "\n");

          // Loop over all of the loads and stores that alias this pointer,
          // adding them to the Value2AllocaMap as well...
          for (std::set<Value*>::const_iterator
                 I = CurLBI->LoadedPointers.begin(),
                 E = CurLBI->LoadedPointers.end(); I != E; ++I)
            if (AA->alias(V, ~0, *I, ~0) == AliasAnalysis::MustAlias)
              ValueToAllocaMap[*I] = AI;

          for (std::set<Value*>::const_iterator
                 I = CurLBI->StoredPointers.begin(),
                 E = CurLBI->StoredPointers.end(); I != E; ++I)
            if (AA->alias(V, ~0, *I, ~0) == AliasAnalysis::MustAlias)
              ValueToAllocaMap[*I] = AI;
        }
      }
    }
  }
}
