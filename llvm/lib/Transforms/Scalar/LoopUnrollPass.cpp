//===-- LoopUnroll.cpp - Loop unroller pass -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass implements a simple loop unroller.  It works best when loops have
// been canonicalized by the -indvars pass, allowing it to determine the trip
// counts of loops easily.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loop-unroll"
#include "llvm/IntrinsicInst.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"
#include <climits>

using namespace llvm;

static cl::opt<unsigned>
UnrollThreshold("unroll-threshold", cl::init(150), cl::Hidden,
  cl::desc("The cut-off point for automatic loop unrolling"));

static cl::opt<unsigned>
UnrollCount("unroll-count", cl::init(0), cl::Hidden,
  cl::desc("Use this unroll count for all loops, for testing purposes"));

static cl::opt<bool>
UnrollAllowPartial("unroll-allow-partial", cl::init(false), cl::Hidden,
  cl::desc("Allows loops to be partially unrolled until "
           "-unroll-threshold loop size is reached."));

namespace {
  class LoopUnroll : public LoopPass {
  public:
    static char ID; // Pass ID, replacement for typeid
    LoopUnroll() : LoopPass(ID) {
      initializeLoopUnrollPass(*PassRegistry::getPassRegistry());
    }

    /// A magic value for use with the Threshold parameter to indicate
    /// that the loop unroll should be performed regardless of how much
    /// code expansion would result.
    static const unsigned NoThreshold = UINT_MAX;
    
    // Threshold to use when optsize is specified (and there is no
    // explicit -unroll-threshold).
    static const unsigned OptSizeUnrollThreshold = 50;
    
    unsigned CurrentThreshold;

    bool runOnLoop(Loop *L, LPPassManager &LPM);

    /// This transformation requires natural loop information & requires that
    /// loop preheaders be inserted into the CFG...
    ///
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<LoopInfo>();
      AU.addPreserved<LoopInfo>();
      AU.addRequiredID(LoopSimplifyID);
      AU.addPreservedID(LoopSimplifyID);
      AU.addRequiredID(LCSSAID);
      AU.addPreservedID(LCSSAID);
      AU.addPreserved<ScalarEvolution>();
      // FIXME: Loop unroll requires LCSSA. And LCSSA requires dom info.
      // If loop unroll does not preserve dom info then LCSSA pass on next
      // loop will receive invalid dom info.
      // For now, recreate dom info, if loop is unrolled.
      AU.addPreserved<DominatorTree>();
    }
  };
}

char LoopUnroll::ID = 0;
INITIALIZE_PASS_BEGIN(LoopUnroll, "loop-unroll", "Unroll loops", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfo)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(LCSSA)
INITIALIZE_PASS_END(LoopUnroll, "loop-unroll", "Unroll loops", false, false)

Pass *llvm::createLoopUnrollPass() { return new LoopUnroll(); }

/// ApproximateLoopSize - Approximate the size of the loop.
static unsigned ApproximateLoopSize(const Loop *L, unsigned &NumCalls) {
  CodeMetrics Metrics;
  for (Loop::block_iterator I = L->block_begin(), E = L->block_end();
       I != E; ++I)
    Metrics.analyzeBasicBlock(*I);
  NumCalls = Metrics.NumInlineCandidates;
  
  unsigned LoopSize = Metrics.NumInsts;
  
  // If we can identify the induction variable, we know that it will become
  // constant when we unroll the loop, so factor that into our loop size 
  // estimate.
  // FIXME: We have to divide by InlineConstants::InstrCost because the
  // measure returned by CountCodeReductionForConstant is not an instruction
  // count, but rather a weight as defined by InlineConstants.  It would 
  // probably be a good idea to standardize on a single weighting scheme by
  // pushing more of the logic for weighting into CodeMetrics.
  if (PHINode *IndVar = L->getCanonicalInductionVariable()) {
    unsigned SizeDecrease = Metrics.CountCodeReductionForConstant(IndVar);
    // NOTE: Because SizeDecrease is a fuzzy estimate, we don't want to allow
    // it to totally negate the cost of unrolling a loop.
    SizeDecrease = SizeDecrease > LoopSize / 2 ? LoopSize / 2 : SizeDecrease;
  }
  
  // Don't allow an estimate of size zero.  This would allows unrolling of loops
  // with huge iteration counts, which is a compile time problem even if it's
  // not a problem for code quality.
  if (LoopSize == 0) LoopSize = 1;
  
  return LoopSize;
}

bool LoopUnroll::runOnLoop(Loop *L, LPPassManager &LPM) {
  LoopInfo *LI = &getAnalysis<LoopInfo>();

  BasicBlock *Header = L->getHeader();
  DEBUG(dbgs() << "Loop Unroll: F[" << Header->getParent()->getName()
        << "] Loop %" << Header->getName() << "\n");
  (void)Header;
  
  // Determine the current unrolling threshold.  While this is normally set
  // from UnrollThreshold, it is overridden to a smaller value if the current
  // function is marked as optimize-for-size, and the unroll threshold was
  // not user specified.
  CurrentThreshold = UnrollThreshold;
  if (Header->getParent()->hasFnAttr(Attribute::OptimizeForSize) &&
      UnrollThreshold.getNumOccurrences() == 0)
    CurrentThreshold = OptSizeUnrollThreshold;

  // Find trip count
  unsigned TripCount = L->getSmallConstantTripCount();
  unsigned Count = UnrollCount;

  // Automatically select an unroll count.
  if (Count == 0) {
    // Conservative heuristic: if we know the trip count, see if we can
    // completely unroll (subject to the threshold, checked below); otherwise
    // try to find greatest modulo of the trip count which is still under
    // threshold value.
    if (TripCount == 0)
      return false;
    Count = TripCount;
  }

  // Enforce the threshold.
  if (CurrentThreshold != NoThreshold) {
    unsigned NumInlineCandidates;
    unsigned LoopSize = ApproximateLoopSize(L, NumInlineCandidates);
    DEBUG(dbgs() << "  Loop Size = " << LoopSize << "\n");
    if (NumInlineCandidates != 0) {
      DEBUG(dbgs() << "  Not unrolling loop with inlinable calls.\n");
      return false;
    }
    uint64_t Size = (uint64_t)LoopSize*Count;
    if (TripCount != 1 && Size > CurrentThreshold) {
      DEBUG(dbgs() << "  Too large to fully unroll with count: " << Count
            << " because size: " << Size << ">" << CurrentThreshold << "\n");
      if (!UnrollAllowPartial) {
        DEBUG(dbgs() << "  will not try to unroll partially because "
              << "-unroll-allow-partial not given\n");
        return false;
      }
      // Reduce unroll count to be modulo of TripCount for partial unrolling
      Count = CurrentThreshold / LoopSize;
      while (Count != 0 && TripCount%Count != 0) {
        Count--;
      }
      if (Count < 2) {
        DEBUG(dbgs() << "  could not unroll partially\n");
        return false;
      }
      DEBUG(dbgs() << "  partially unrolling with count: " << Count << "\n");
    }
  }

  // Unroll the loop.
  Function *F = L->getHeader()->getParent();
  if (!UnrollLoop(L, Count, LI, &LPM))
    return false;

  // FIXME: Reconstruct dom info, because it is not preserved properly.
  if (DominatorTree *DT = getAnalysisIfAvailable<DominatorTree>())
    DT->runOnFunction(*F);
  return true;
}
