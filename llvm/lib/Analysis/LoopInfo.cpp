//===- LoopInfo.cpp - Natural Loop Calculator -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the LoopInfo class that is used to identify natural loops
// and determine the loop depth of various nodes of the CFG.  Note that the
// loops identified may actually be several natural loops that share the same
// header node... not just a single natural loop.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <algorithm>
using namespace llvm;

// Always verify loopinfo if expensive checking is enabled.
#ifdef XDEBUG
static bool VerifyLoopInfo = true;
#else
static bool VerifyLoopInfo = false;
#endif
static cl::opt<bool,true>
VerifyLoopInfoX("verify-loop-info", cl::location(VerifyLoopInfo),
                cl::desc("Verify loop info (time consuming)"));

char LoopInfo::ID = 0;
static RegisterPass<LoopInfo>
X("loops", "Natural Loop Information", true, true);

//===----------------------------------------------------------------------===//
// Loop implementation
//

/// isLoopInvariant - Return true if the specified value is loop invariant
///
bool Loop::isLoopInvariant(Value *V) const {
  if (Instruction *I = dyn_cast<Instruction>(V))
    return isLoopInvariant(I);
  return true;  // All non-instructions are loop invariant
}

/// isLoopInvariant - Return true if the specified instruction is
/// loop-invariant.
///
bool Loop::isLoopInvariant(Instruction *I) const {
  return !contains(I);
}

/// makeLoopInvariant - If the given value is an instruciton inside of the
/// loop and it can be hoisted, do so to make it trivially loop-invariant.
/// Return true if the value after any hoisting is loop invariant. This
/// function can be used as a slightly more aggressive replacement for
/// isLoopInvariant.
///
/// If InsertPt is specified, it is the point to hoist instructions to.
/// If null, the terminator of the loop preheader is used.
///
bool Loop::makeLoopInvariant(Value *V, bool &Changed,
                             Instruction *InsertPt) const {
  if (Instruction *I = dyn_cast<Instruction>(V))
    return makeLoopInvariant(I, Changed, InsertPt);
  return true;  // All non-instructions are loop-invariant.
}

/// makeLoopInvariant - If the given instruction is inside of the
/// loop and it can be hoisted, do so to make it trivially loop-invariant.
/// Return true if the instruction after any hoisting is loop invariant. This
/// function can be used as a slightly more aggressive replacement for
/// isLoopInvariant.
///
/// If InsertPt is specified, it is the point to hoist instructions to.
/// If null, the terminator of the loop preheader is used.
///
bool Loop::makeLoopInvariant(Instruction *I, bool &Changed,
                             Instruction *InsertPt) const {
  // Test if the value is already loop-invariant.
  if (isLoopInvariant(I))
    return true;
  if (!I->isSafeToSpeculativelyExecute())
    return false;
  if (I->mayReadFromMemory())
    return false;
  // Determine the insertion point, unless one was given.
  if (!InsertPt) {
    BasicBlock *Preheader = getLoopPreheader();
    // Without a preheader, hoisting is not feasible.
    if (!Preheader)
      return false;
    InsertPt = Preheader->getTerminator();
  }
  // Don't hoist instructions with loop-variant operands.
  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
    if (!makeLoopInvariant(I->getOperand(i), Changed, InsertPt))
      return false;
  // Hoist.
  I->moveBefore(InsertPt);
  Changed = true;
  return true;
}

/// getCanonicalInductionVariable - Check to see if the loop has a canonical
/// induction variable: an integer recurrence that starts at 0 and increments
/// by one each time through the loop.  If so, return the phi node that
/// corresponds to it.
///
/// The IndVarSimplify pass transforms loops to have a canonical induction
/// variable.
///
PHINode *Loop::getCanonicalInductionVariable() const {
  BasicBlock *H = getHeader();

  BasicBlock *Incoming = 0, *Backedge = 0;
  typedef GraphTraits<Inverse<BasicBlock*> > InvBlockTraits;
  InvBlockTraits::ChildIteratorType PI = InvBlockTraits::child_begin(H);
  assert(PI != InvBlockTraits::child_end(H) &&
         "Loop must have at least one backedge!");
  Backedge = *PI++;
  if (PI == InvBlockTraits::child_end(H)) return 0;  // dead loop
  Incoming = *PI++;
  if (PI != InvBlockTraits::child_end(H)) return 0;  // multiple backedges?

  if (contains(Incoming)) {
    if (contains(Backedge))
      return 0;
    std::swap(Incoming, Backedge);
  } else if (!contains(Backedge))
    return 0;

  // Loop over all of the PHI nodes, looking for a canonical indvar.
  for (BasicBlock::iterator I = H->begin(); isa<PHINode>(I); ++I) {
    PHINode *PN = cast<PHINode>(I);
    if (ConstantInt *CI =
        dyn_cast<ConstantInt>(PN->getIncomingValueForBlock(Incoming)))
      if (CI->isNullValue())
        if (Instruction *Inc =
            dyn_cast<Instruction>(PN->getIncomingValueForBlock(Backedge)))
          if (Inc->getOpcode() == Instruction::Add &&
                Inc->getOperand(0) == PN)
            if (ConstantInt *CI = dyn_cast<ConstantInt>(Inc->getOperand(1)))
              if (CI->equalsInt(1))
                return PN;
  }
  return 0;
}

/// getCanonicalInductionVariableIncrement - Return the LLVM value that holds
/// the canonical induction variable value for the "next" iteration of the
/// loop.  This always succeeds if getCanonicalInductionVariable succeeds.
///
Instruction *Loop::getCanonicalInductionVariableIncrement() const {
  if (PHINode *PN = getCanonicalInductionVariable()) {
    bool P1InLoop = contains(PN->getIncomingBlock(1));
    return cast<Instruction>(PN->getIncomingValue(P1InLoop));
  }
  return 0;
}

/// getTripCount - Return a loop-invariant LLVM value indicating the number of
/// times the loop will be executed.  Note that this means that the backedge
/// of the loop executes N-1 times.  If the trip-count cannot be determined,
/// this returns null.
///
/// The IndVarSimplify pass transforms loops to have a form that this
/// function easily understands.
///
Value *Loop::getTripCount() const {
  // Canonical loops will end with a 'cmp ne I, V', where I is the incremented
  // canonical induction variable and V is the trip count of the loop.
  Instruction *Inc = getCanonicalInductionVariableIncrement();
  if (Inc == 0) return 0;
  PHINode *IV = cast<PHINode>(Inc->getOperand(0));

  BasicBlock *BackedgeBlock =
    IV->getIncomingBlock(contains(IV->getIncomingBlock(1)));

  if (BranchInst *BI = dyn_cast<BranchInst>(BackedgeBlock->getTerminator()))
    if (BI->isConditional()) {
      if (ICmpInst *ICI = dyn_cast<ICmpInst>(BI->getCondition())) {
        if (ICI->getOperand(0) == Inc) {
          if (BI->getSuccessor(0) == getHeader()) {
            if (ICI->getPredicate() == ICmpInst::ICMP_NE)
              return ICI->getOperand(1);
          } else if (ICI->getPredicate() == ICmpInst::ICMP_EQ) {
            return ICI->getOperand(1);
          }
        }
      }
    }

  return 0;
}

/// getSmallConstantTripCount - Returns the trip count of this loop as a
/// normal unsigned value, if possible. Returns 0 if the trip count is unknown
/// of not constant. Will also return 0 if the trip count is very large
/// (>= 2^32)
unsigned Loop::getSmallConstantTripCount() const {
  Value* TripCount = this->getTripCount();
  if (TripCount) {
    if (ConstantInt *TripCountC = dyn_cast<ConstantInt>(TripCount)) {
      // Guard against huge trip counts.
      if (TripCountC->getValue().getActiveBits() <= 32) {
        return (unsigned)TripCountC->getZExtValue();
      }
    }
  }
  return 0;
}

/// getSmallConstantTripMultiple - Returns the largest constant divisor of the
/// trip count of this loop as a normal unsigned value, if possible. This
/// means that the actual trip count is always a multiple of the returned
/// value (don't forget the trip count could very well be zero as well!).
///
/// Returns 1 if the trip count is unknown or not guaranteed to be the
/// multiple of a constant (which is also the case if the trip count is simply
/// constant, use getSmallConstantTripCount for that case), Will also return 1
/// if the trip count is very large (>= 2^32).
unsigned Loop::getSmallConstantTripMultiple() const {
  Value* TripCount = this->getTripCount();
  // This will hold the ConstantInt result, if any
  ConstantInt *Result = NULL;
  if (TripCount) {
    // See if the trip count is constant itself
    Result = dyn_cast<ConstantInt>(TripCount);
    // if not, see if it is a multiplication
    if (!Result)
      if (BinaryOperator *BO = dyn_cast<BinaryOperator>(TripCount)) {
        switch (BO->getOpcode()) {
        case BinaryOperator::Mul:
          Result = dyn_cast<ConstantInt>(BO->getOperand(1));
          break;
        case BinaryOperator::Shl:
          if (ConstantInt *CI = dyn_cast<ConstantInt>(BO->getOperand(1)))
            if (CI->getValue().getActiveBits() <= 5)
              return 1u << CI->getZExtValue();
          break;
        default:
          break;
        }
      }
  }
  // Guard against huge trip counts.
  if (Result && Result->getValue().getActiveBits() <= 32) {
    return (unsigned)Result->getZExtValue();
  } else {
    return 1;
  }
}

/// isLCSSAForm - Return true if the Loop is in LCSSA form
bool Loop::isLCSSAForm(DominatorTree &DT) const {
  // Sort the blocks vector so that we can use binary search to do quick
  // lookups.
  SmallPtrSet<BasicBlock *, 16> LoopBBs(block_begin(), block_end());

  for (block_iterator BI = block_begin(), E = block_end(); BI != E; ++BI) {
    BasicBlock *BB = *BI;
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E;++I)
      for (Value::use_iterator UI = I->use_begin(), E = I->use_end(); UI != E;
           ++UI) {
        BasicBlock *UserBB = cast<Instruction>(*UI)->getParent();
        if (PHINode *P = dyn_cast<PHINode>(*UI))
          UserBB = P->getIncomingBlock(UI);

        // Check the current block, as a fast-path, before checking whether
        // the use is anywhere in the loop.  Most values are used in the same
        // block they are defined in.  Also, blocks not reachable from the
        // entry are special; uses in them don't need to go through PHIs.
        if (UserBB != BB &&
            !LoopBBs.count(UserBB) &&
            DT.isReachableFromEntry(UserBB))
          return false;
      }
  }

  return true;
}

/// isLoopSimplifyForm - Return true if the Loop is in the form that
/// the LoopSimplify form transforms loops to, which is sometimes called
/// normal form.
bool Loop::isLoopSimplifyForm() const {
  // Normal-form loops have a preheader, a single backedge, and all of their
  // exits have all their predecessors inside the loop.
  return getLoopPreheader() && getLoopLatch() && hasDedicatedExits();
}

/// hasDedicatedExits - Return true if no exit block for the loop
/// has a predecessor that is outside the loop.
bool Loop::hasDedicatedExits() const {
  // Sort the blocks vector so that we can use binary search to do quick
  // lookups.
  SmallPtrSet<BasicBlock *, 16> LoopBBs(block_begin(), block_end());
  // Each predecessor of each exit block of a normal loop is contained
  // within the loop.
  SmallVector<BasicBlock *, 4> ExitBlocks;
  getExitBlocks(ExitBlocks);
  for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i)
    for (pred_iterator PI = pred_begin(ExitBlocks[i]),
         PE = pred_end(ExitBlocks[i]); PI != PE; ++PI)
      if (!LoopBBs.count(*PI))
        return false;
  // All the requirements are met.
  return true;
}

/// getUniqueExitBlocks - Return all unique successor blocks of this loop.
/// These are the blocks _outside of the current loop_ which are branched to.
/// This assumes that loop exits are in canonical form.
///
void
Loop::getUniqueExitBlocks(SmallVectorImpl<BasicBlock *> &ExitBlocks) const {
  assert(hasDedicatedExits() &&
         "getUniqueExitBlocks assumes the loop has canonical form exits!");

  // Sort the blocks vector so that we can use binary search to do quick
  // lookups.
  SmallVector<BasicBlock *, 128> LoopBBs(block_begin(), block_end());
  std::sort(LoopBBs.begin(), LoopBBs.end());

  SmallVector<BasicBlock *, 32> switchExitBlocks;

  for (block_iterator BI = block_begin(), BE = block_end(); BI != BE; ++BI) {

    BasicBlock *current = *BI;
    switchExitBlocks.clear();

    typedef GraphTraits<BasicBlock *> BlockTraits;
    typedef GraphTraits<Inverse<BasicBlock *> > InvBlockTraits;
    for (BlockTraits::ChildIteratorType I =
         BlockTraits::child_begin(*BI), E = BlockTraits::child_end(*BI);
         I != E; ++I) {
      // If block is inside the loop then it is not a exit block.
      if (std::binary_search(LoopBBs.begin(), LoopBBs.end(), *I))
        continue;

      InvBlockTraits::ChildIteratorType PI = InvBlockTraits::child_begin(*I);
      BasicBlock *firstPred = *PI;

      // If current basic block is this exit block's first predecessor
      // then only insert exit block in to the output ExitBlocks vector.
      // This ensures that same exit block is not inserted twice into
      // ExitBlocks vector.
      if (current != firstPred)
        continue;

      // If a terminator has more then two successors, for example SwitchInst,
      // then it is possible that there are multiple edges from current block
      // to one exit block.
      if (std::distance(BlockTraits::child_begin(current),
                        BlockTraits::child_end(current)) <= 2) {
        ExitBlocks.push_back(*I);
        continue;
      }

      // In case of multiple edges from current block to exit block, collect
      // only one edge in ExitBlocks. Use switchExitBlocks to keep track of
      // duplicate edges.
      if (std::find(switchExitBlocks.begin(), switchExitBlocks.end(), *I)
          == switchExitBlocks.end()) {
        switchExitBlocks.push_back(*I);
        ExitBlocks.push_back(*I);
      }
    }
  }
}

/// getUniqueExitBlock - If getUniqueExitBlocks would return exactly one
/// block, return that block. Otherwise return null.
BasicBlock *Loop::getUniqueExitBlock() const {
  SmallVector<BasicBlock *, 8> UniqueExitBlocks;
  getUniqueExitBlocks(UniqueExitBlocks);
  if (UniqueExitBlocks.size() == 1)
    return UniqueExitBlocks[0];
  return 0;
}

void Loop::dump() const {
  print(dbgs());
}

//===----------------------------------------------------------------------===//
// LoopInfo implementation
//
bool LoopInfo::runOnFunction(Function &) {
  releaseMemory();
  LI.Calculate(getAnalysis<DominatorTree>().getBase());    // Update
  return false;
}

void LoopInfo::verifyAnalysis() const {
  // LoopInfo is a FunctionPass, but verifying every loop in the function
  // each time verifyAnalysis is called is very expensive. The
  // -verify-loop-info option can enable this. In order to perform some
  // checking by default, LoopPass has been taught to call verifyLoop
  // manually during loop pass sequences.

  if (!VerifyLoopInfo) return;

  for (iterator I = begin(), E = end(); I != E; ++I) {
    assert(!(*I)->getParentLoop() && "Top-level loop has a parent!");
    (*I)->verifyLoopNest();
  }

  // TODO: check BBMap consistency.
}

void LoopInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<DominatorTree>();
}

void LoopInfo::print(raw_ostream &OS, const Module*) const {
  LI.print(OS);
}

