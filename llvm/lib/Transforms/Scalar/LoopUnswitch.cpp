//===-- LoopUnswitch.cpp - Hoist loop-invariant conditionals in loop ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass transforms loops that contain branches on loop-invariant conditions
// to have multiple loops.  For example, it turns the left into the right code:
//
//  for (...)                  if (lic)
//    A                          for (...)
//    if (lic)                     A; B; C
//      B                      else
//    C                          for (...)
//                                 A; C
//
// This can increase the size of the code exponentially (doubling it every time
// a loop is unswitched) so we only unswitch if the resultant code will be
// smaller than a threshold.
//
// This pass expects LICM to be run before it to hoist invariant conditions out
// of the loop, to make the unswitching opportunity obvious.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loop-unswitch"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"
#include <algorithm>
#include <iostream>
#include <set>
using namespace llvm;

namespace {
  Statistic<> NumUnswitched("loop-unswitch", "Number of loops unswitched");
  cl::opt<unsigned>
  Threshold("loop-unswitch-threshold", cl::desc("Max loop size to unswitch"),
            cl::init(10), cl::Hidden);
  
  class LoopUnswitch : public FunctionPass {
    LoopInfo *LI;  // Loop information
  public:
    virtual bool runOnFunction(Function &F);
    bool visitLoop(Loop *L);

    /// This transformation requires natural loop information & requires that
    /// loop preheaders be inserted into the CFG...
    ///
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequiredID(LoopSimplifyID);
      AU.addPreservedID(LoopSimplifyID);
      AU.addRequired<LoopInfo>();
      AU.addPreserved<LoopInfo>();
    }

  private:
    unsigned getLoopUnswitchCost(Loop *L, Value *LIC);
    void VersionLoop(Value *LIC, Loop *L, Loop *&Out1, Loop *&Out2);
    BasicBlock *SplitBlock(BasicBlock *BB, bool SplitAtTop);
    void RewriteLoopBodyWithConditionConstant(Loop *L, Value *LIC, bool Val);
    void UnswitchTrivialCondition(Loop *L, Value *Cond, ConstantBool *LoopCond);
  };
  RegisterOpt<LoopUnswitch> X("loop-unswitch", "Unswitch loops");
}

FunctionPass *llvm::createLoopUnswitchPass() { return new LoopUnswitch(); }

bool LoopUnswitch::runOnFunction(Function &F) {
  bool Changed = false;
  LI = &getAnalysis<LoopInfo>();

  // Transform all the top-level loops.  Copy the loop list so that the child
  // can update the loop tree if it needs to delete the loop.
  std::vector<Loop*> SubLoops(LI->begin(), LI->end());
  for (unsigned i = 0, e = SubLoops.size(); i != e; ++i)
    Changed |= visitLoop(SubLoops[i]);

  return Changed;
}


/// LoopValuesUsedOutsideLoop - Return true if there are any values defined in
/// the loop that are used by instructions outside of it.
static bool LoopValuesUsedOutsideLoop(Loop *L) {
  // We will be doing lots of "loop contains block" queries.  Loop::contains is
  // linear time, use a set to speed this up.
  std::set<BasicBlock*> LoopBlocks;

  for (Loop::block_iterator BB = L->block_begin(), E = L->block_end();
       BB != E; ++BB)
    LoopBlocks.insert(*BB);
  
  for (Loop::block_iterator BB = L->block_begin(), E = L->block_end();
       BB != E; ++BB) {
    for (BasicBlock::iterator I = (*BB)->begin(), E = (*BB)->end(); I != E; ++I)
      for (Value::use_iterator UI = I->use_begin(), E = I->use_end(); UI != E;
           ++UI) {
        BasicBlock *UserBB = cast<Instruction>(*UI)->getParent();
        if (!LoopBlocks.count(UserBB))
          return true;
      }
  }
  return false;
}

/// IsTrivialUnswitchCondition - Check to see if this unswitch condition is
/// trivial: that is, that the condition controls whether or not the loop does
/// anything at all.  If this is a trivial condition, unswitching produces no
/// code duplications (equivalently, it produces a simpler loop and a new empty
/// loop, which gets deleted).
///
/// If this is a trivial condition, return ConstantBool::True if the loop body
/// runs when the condition is true, False if the loop body executes when the
/// condition is false.  Otherwise, return null to indicate a complex condition.
static ConstantBool *IsTrivialUnswitchCondition(Loop *L, Value *Cond) {
  BasicBlock *Header = L->getHeader();
  BranchInst *HeaderTerm = dyn_cast<BranchInst>(Header->getTerminator());
  ConstantBool *RetVal = 0;
  
  // If the header block doesn't end with a conditional branch on Cond, we can't
  // handle it.
  if (!HeaderTerm || !HeaderTerm->isConditional() ||
      HeaderTerm->getCondition() != Cond)
    return 0;
  
  // Check to see if the conditional branch goes to the latch block.  If not,
  // it's not trivial.  This also determines the value of Cond that will execute
  // the loop.
  BasicBlock *Latch = L->getLoopLatch();
  if (HeaderTerm->getSuccessor(1) == Latch)
    RetVal = ConstantBool::True;
  else if (HeaderTerm->getSuccessor(0) == Latch)
    RetVal = ConstantBool::False;
  else
    return 0;  // Doesn't branch to latch block.
  
  // The latch block must end with a conditional branch where one edge goes to
  // the header (this much we know) and one edge goes OUT of the loop.
  BranchInst *LatchBranch = dyn_cast<BranchInst>(Latch->getTerminator());
  if (!LatchBranch || !LatchBranch->isConditional()) return 0;

  if (LatchBranch->getSuccessor(0) == Header) {
    if (L->contains(LatchBranch->getSuccessor(1))) return 0;
  } else {
    assert(LatchBranch->getSuccessor(1) == Header);
    if (L->contains(LatchBranch->getSuccessor(0))) return 0;
  }
  
  // We already know that nothing uses any scalar values defined inside of this
  // loop.  As such, we just have to check to see if this loop will execute any
  // side-effecting instructions (e.g. stores, calls, volatile loads) in the
  // part of the loop that the code *would* execute.
  for (BasicBlock::iterator I = Header->begin(), E = Header->end(); I != E; ++I)
    if (I->mayWriteToMemory())
      return 0;
  for (BasicBlock::iterator I = Latch->begin(), E = Latch->end(); I != E; ++I)
    if (I->mayWriteToMemory())
      return 0;
  return RetVal;
}

/// getLoopUnswitchCost - Return the cost (code size growth) that will happen if
/// we choose to unswitch the specified loop on the specified value.
///
unsigned LoopUnswitch::getLoopUnswitchCost(Loop *L, Value *LIC) {
  // If the condition is trivial, always unswitch.  There is no code growth for
  // this case.
  if (IsTrivialUnswitchCondition(L, LIC))
    return 0;
  
  unsigned Cost = 0;
  // FIXME: this is brain dead.  It should take into consideration code
  // shrinkage.
  for (Loop::block_iterator I = L->block_begin(), E = L->block_end();
       I != E; ++I) {
    BasicBlock *BB = *I;
    // Do not include empty blocks in the cost calculation.  This happen due to
    // loop canonicalization and will be removed.
    if (BB->begin() == BasicBlock::iterator(BB->getTerminator()))
      continue;
    
    // Count basic blocks.
    ++Cost;
  }

  return Cost;
}

bool LoopUnswitch::visitLoop(Loop *L) {
  bool Changed = false;

  // Recurse through all subloops before we process this loop.  Copy the loop
  // list so that the child can update the loop tree if it needs to delete the
  // loop.
  std::vector<Loop*> SubLoops(L->begin(), L->end());
  for (unsigned i = 0, e = SubLoops.size(); i != e; ++i)
    Changed |= visitLoop(SubLoops[i]);

  // Loop over all of the basic blocks in the loop.  If we find an interior
  // block that is branching on a loop-invariant condition, we can unswitch this
  // loop.
  for (Loop::block_iterator I = L->block_begin(), E = L->block_end();
       I != E; ++I) {
    TerminatorInst *TI = (*I)->getTerminator();
    if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
      if (!isa<Constant>(SI) && L->isLoopInvariant(SI->getCondition()))
        DEBUG(std::cerr << "TODO: Implement unswitching 'switch' loop %"
              << L->getHeader()->getName() << ", cost = "
              << L->getBlocks().size() << "\n" << **I);
      continue;
    }
    
    BranchInst *BI = dyn_cast<BranchInst>(TI);
    if (!BI) continue;
    
    // If this isn't branching on an invariant condition, we can't unswitch it.
    if (!BI->isConditional() || isa<Constant>(BI->getCondition()) ||
        !L->isLoopInvariant(BI->getCondition()))
      continue;
    
    // Check to see if it would be profitable to unswitch this loop.
    if (getLoopUnswitchCost(L, BI->getCondition()) > Threshold) {
      // FIXME: this should estimate growth by the amount of code shared by the
      // resultant unswitched loops.  This should have no code growth:
      //    for () { if (iv) {...} }
      // as one copy of the loop will be empty.
      //
      DEBUG(std::cerr << "NOT unswitching loop %"
            << L->getHeader()->getName() << ", cost too high: "
            << L->getBlocks().size() << "\n");
      continue;
    }
    
    // If this loop has live-out values, we can't unswitch it. We need something
    // like loop-closed SSA form in order to know how to insert PHI nodes for
    // these values.
    if (LoopValuesUsedOutsideLoop(L)) {
      DEBUG(std::cerr << "NOT unswitching loop %"
                      << L->getHeader()->getName()
                      << ", a loop value is used outside loop!\n");
      continue;
    }
      
    //std::cerr << "BEFORE:\n"; LI->dump();
    Loop *NewLoop1 = 0, *NewLoop2 = 0;
 
    // If this is a trivial condition to unswitch (which results in no code
    // duplication), do it now.
    if (ConstantBool *V = IsTrivialUnswitchCondition(L, BI->getCondition())) {
      UnswitchTrivialCondition(L, BI->getCondition(), V);
      NewLoop1 = L;
    } else {
      VersionLoop(BI->getCondition(), L, NewLoop1, NewLoop2);
    }
    
    //std::cerr << "AFTER:\n"; LI->dump();
    
    // Try to unswitch each of our new loops now!
    if (NewLoop1) visitLoop(NewLoop1);
    if (NewLoop2) visitLoop(NewLoop2);
    return true;
  }

  return Changed;
}

/// SplitBlock - Split the specified basic block into two pieces.  If SplitAtTop
/// is false, this splits the block so the second half only has an unconditional
/// branch.  If SplitAtTop is true, it makes it so the first half of the block
/// only has an unconditional branch in it.
///
/// This method updates the LoopInfo for this function to correctly reflect the
/// CFG changes made.
///
/// This routine returns the new basic block that was inserted, which is always
/// the later part of the block.
BasicBlock *LoopUnswitch::SplitBlock(BasicBlock *BB, bool SplitAtTop) {
  BasicBlock::iterator SplitPoint;
  if (!SplitAtTop)
    SplitPoint = BB->getTerminator();
  else {
    SplitPoint = BB->begin();
    while (isa<PHINode>(SplitPoint)) ++SplitPoint;
  }
  
  BasicBlock *New = BB->splitBasicBlock(SplitPoint, BB->getName()+".tail");
  // New now lives in whichever loop that BB used to.
  if (Loop *L = LI->getLoopFor(BB))
    L->addBasicBlockToLoop(New, *LI);
  return New;
}


// RemapInstruction - Convert the instruction operands from referencing the
// current values into those specified by ValueMap.
//
static inline void RemapInstruction(Instruction *I,
                                    std::map<const Value *, Value*> &ValueMap) {
  for (unsigned op = 0, E = I->getNumOperands(); op != E; ++op) {
    Value *Op = I->getOperand(op);
    std::map<const Value *, Value*>::iterator It = ValueMap.find(Op);
    if (It != ValueMap.end()) Op = It->second;
    I->setOperand(op, Op);
  }
}

/// CloneLoop - Recursively clone the specified loop and all of its children,
/// mapping the blocks with the specified map.
static Loop *CloneLoop(Loop *L, Loop *PL, std::map<const Value*, Value*> &VM,
                       LoopInfo *LI) {
  Loop *New = new Loop();

  if (PL)
    PL->addChildLoop(New);
  else
    LI->addTopLevelLoop(New);

  // Add all of the blocks in L to the new loop.
  for (Loop::block_iterator I = L->block_begin(), E = L->block_end();
       I != E; ++I)
    if (LI->getLoopFor(*I) == L)
      New->addBasicBlockToLoop(cast<BasicBlock>(VM[*I]), *LI);

  // Add all of the subloops to the new loop.
  for (Loop::iterator I = L->begin(), E = L->end(); I != E; ++I)
    CloneLoop(*I, New, VM, LI);

  return New;
}

/// UnswitchTrivialCondition - Given a loop that has a trivial unswitchable
/// condition in it (a cond branch from its header block to its latch block,
/// where the path through the loop that doesn't execute its body has no 
/// side-effects), unswitch it.  This doesn't involve any code duplication, just
/// moving the conditional branch outside of the loop and updating loop info.
void LoopUnswitch::UnswitchTrivialCondition(Loop *L, Value *Cond, 
                                            ConstantBool *LoopCond) {
  DEBUG(std::cerr << "loop-unswitch: Trivial-Unswitch loop %"
        << L->getHeader()->getName() << " [" << L->getBlocks().size()
        << " blocks] in Function " << L->getHeader()->getParent()->getName()
        << " on cond:" << *Cond << "\n");
  
  // First step, split the preahder, so that we know that there is a safe place
  // to insert the conditional branch.  We will change 'OrigPH' to have a
  // conditional branch on Cond.
  BasicBlock *OrigPH = L->getLoopPreheader();
  BasicBlock *NewPH = SplitBlock(OrigPH, false);

  // Now that we have a place to insert the conditional branch, create a place
  // to branch to: this is the non-header successor of the latch block.
  BranchInst *LatchBranch =cast<BranchInst>(L->getLoopLatch()->getTerminator());
  BasicBlock *ExitBlock = 
    LatchBranch->getSuccessor(LatchBranch->getSuccessor(0) == L->getHeader());
  assert(!L->contains(ExitBlock) && "Exit block is in the loop?");
  
  // Split this block now, so that the loop maintains its exit block.
  BasicBlock *NewExit = SplitBlock(ExitBlock, true);
  
  // Okay, now we have a position to branch from and a position to branch to, 
  // insert the new conditional branch.
  bool EnterOnTrue = LoopCond->getValue();
  new BranchInst(EnterOnTrue ? NewPH : NewExit, EnterOnTrue ? NewExit : NewPH,
                 Cond, OrigPH->getTerminator());
  OrigPH->getTerminator()->eraseFromParent();

  // Now that we know that the loop is never entered when this condition is a
  // particular value, rewrite the loop with this info.  We know that this will
  // at least eliminate the old branch.
  RewriteLoopBodyWithConditionConstant(L, Cond, EnterOnTrue);
  
  ++NumUnswitched;
}


/// VersionLoop - We determined that the loop is profitable to unswitch and
/// contains a branch on a loop invariant condition.  Split it into loop
/// versions and test the condition outside of either loop.  Return the loops
/// created as Out1/Out2.
void LoopUnswitch::VersionLoop(Value *LIC, Loop *L, Loop *&Out1, Loop *&Out2) {
  Function *F = L->getHeader()->getParent();
  
  DEBUG(std::cerr << "loop-unswitch: Unswitching loop %"
        << L->getHeader()->getName() << " [" << L->getBlocks().size()
        << " blocks] in Function " << F->getName()
        << " on cond:" << *LIC << "\n");

  std::vector<BasicBlock*> LoopBlocks;

  // First step, split the preheader and exit blocks, and add these blocks to
  // the LoopBlocks list.
  BasicBlock *OrigPreheader = L->getLoopPreheader();
  LoopBlocks.push_back(SplitBlock(OrigPreheader, false));

  // We want the loop to come after the preheader, but before the exit blocks.
  LoopBlocks.insert(LoopBlocks.end(), L->block_begin(), L->block_end());

  std::vector<BasicBlock*> ExitBlocks;
  L->getExitBlocks(ExitBlocks);
  std::sort(ExitBlocks.begin(), ExitBlocks.end());
  ExitBlocks.erase(std::unique(ExitBlocks.begin(), ExitBlocks.end()),
                   ExitBlocks.end());
  for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i) {
    SplitBlock(ExitBlocks[i], true);
    LoopBlocks.push_back(ExitBlocks[i]);
  }

  // Next step, clone all of the basic blocks that make up the loop (including
  // the loop preheader and exit blocks), keeping track of the mapping between
  // the instructions and blocks.
  std::vector<BasicBlock*> NewBlocks;
  NewBlocks.reserve(LoopBlocks.size());
  std::map<const Value*, Value*> ValueMap;
  for (unsigned i = 0, e = LoopBlocks.size(); i != e; ++i) {
    NewBlocks.push_back(CloneBasicBlock(LoopBlocks[i], ValueMap, ".us", F));
    ValueMap[LoopBlocks[i]] = NewBlocks.back();  // Keep the BB mapping.
  }

  // Splice the newly inserted blocks into the function right before the
  // original preheader.
  F->getBasicBlockList().splice(LoopBlocks[0], F->getBasicBlockList(),
                                NewBlocks[0], F->end());

  // Now we create the new Loop object for the versioned loop.
  Loop *NewLoop = CloneLoop(L, L->getParentLoop(), ValueMap, LI);
  if (Loop *Parent = L->getParentLoop()) {
    // Make sure to add the cloned preheader and exit blocks to the parent loop
    // as well.
    Parent->addBasicBlockToLoop(NewBlocks[0], *LI);
    for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i)
      Parent->addBasicBlockToLoop(cast<BasicBlock>(ValueMap[ExitBlocks[i]]),
                                  *LI);
  }

  // Rewrite the code to refer to itself.
  for (unsigned i = 0, e = NewBlocks.size(); i != e; ++i)
    for (BasicBlock::iterator I = NewBlocks[i]->begin(),
           E = NewBlocks[i]->end(); I != E; ++I)
      RemapInstruction(I, ValueMap);
  
  // Rewrite the original preheader to select between versions of the loop.
  assert(isa<BranchInst>(OrigPreheader->getTerminator()) &&
         cast<BranchInst>(OrigPreheader->getTerminator())->isUnconditional() &&
         OrigPreheader->getTerminator()->getSuccessor(0) == LoopBlocks[0] &&
         "Preheader splitting did not work correctly!");
  // Remove the unconditional branch to LoopBlocks[0].
  OrigPreheader->getInstList().pop_back();

  // Insert a conditional branch on LIC to the two preheaders.  The original
  // code is the true version and the new code is the false version.
  new BranchInst(LoopBlocks[0], NewBlocks[0], LIC, OrigPreheader);

  // Now we rewrite the original code to know that the condition is true and the
  // new code to know that the condition is false.
  RewriteLoopBodyWithConditionConstant(L, LIC, true);
  RewriteLoopBodyWithConditionConstant(NewLoop, LIC, false);
  ++NumUnswitched;
  Out1 = L;
  Out2 = NewLoop;
}

// RewriteLoopBodyWithConditionConstant - We know that the boolean value LIC has
// the value specified by Val in the specified loop.  Rewrite any uses of LIC or
// of properties correlated to it.
void LoopUnswitch::RewriteLoopBodyWithConditionConstant(Loop *L, Value *LIC,
                                                        bool Val) {
  assert(!isa<Constant>(LIC) && "Why are we unswitching on a constant?");
  // FIXME: Support correlated properties, like:
  //  for (...)
  //    if (li1 < li2)
  //      ...
  //    if (li1 > li2)
  //      ...
  ConstantBool *BoolVal = ConstantBool::get(Val);

  std::vector<User*> Users(LIC->use_begin(), LIC->use_end());
  for (unsigned i = 0, e = Users.size(); i != e; ++i)
    if (Instruction *U = cast<Instruction>(Users[i]))
      if (L->contains(U->getParent()))
        U->replaceUsesOfWith(LIC, BoolVal);
}
