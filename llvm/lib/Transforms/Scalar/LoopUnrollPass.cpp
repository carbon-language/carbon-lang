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
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"
#include <climits>

using namespace llvm;

static cl::opt<unsigned>
UnrollThreshold("unroll-threshold", cl::init(100), cl::Hidden,
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
    LoopUnroll() : LoopPass(&ID) {}

    /// A magic value for use with the Threshold parameter to indicate
    /// that the loop unroll should be performed regardless of how much
    /// code expansion would result.
    static const unsigned NoThreshold = UINT_MAX;

    bool runOnLoop(Loop *L, LPPassManager &LPM);

    /// This transformation requires natural loop information & requires that
    /// loop preheaders be inserted into the CFG...
    ///
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequiredID(LoopSimplifyID);
      AU.addRequiredID(LCSSAID);
      AU.addRequired<LoopInfo>();
      AU.addPreservedID(LCSSAID);
      AU.addPreserved<LoopInfo>();
      // FIXME: Loop unroll requires LCSSA. And LCSSA requires dom info.
      // If loop unroll does not preserve dom info then LCSSA pass on next
      // loop will receive invalid dom info.
      // For now, recreate dom info, if loop is unrolled.
      AU.addPreserved<DominatorTree>();
      AU.addPreserved<DominanceFrontier>();
    }
  };
}

char LoopUnroll::ID = 0;
static RegisterPass<LoopUnroll> X("loop-unroll", "Unroll loops");

Pass *llvm::createLoopUnrollPass() { return new LoopUnroll(); }

/// ApproximateLoopSize - Approximate the size of the loop.
static unsigned ApproximateLoopSize(const Loop *L) {
  CodeMetrics Metrics;
  for (Loop::block_iterator I = L->block_begin(), E = L->block_end();
       I != E; ++I)
    Metrics.analyzeBasicBlock(*I);
  return Metrics.NumInsts;
}

bool LoopUnroll::runOnLoop(Loop *L, LPPassManager &LPM) {
  assert(L->isLCSSAForm());
  LoopInfo *LI = &getAnalysis<LoopInfo>();

  BasicBlock *Header = L->getHeader();
  DEBUG(errs() << "Loop Unroll: F[" << Header->getParent()->getName()
        << "] Loop %" << Header->getName() << "\n");
  (void)Header;

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
  if (UnrollThreshold != NoThreshold) {
    unsigned LoopSize = ApproximateLoopSize(L);
    DEBUG(errs() << "  Loop Size = " << LoopSize << "\n");
    uint64_t Size = (uint64_t)LoopSize*Count;
    if (TripCount != 1 && Size > UnrollThreshold) {
      DEBUG(errs() << "  Too large to fully unroll with count: " << Count
            << " because size: " << Size << ">" << UnrollThreshold << "\n");
      if (!UnrollAllowPartial) {
        DEBUG(errs() << "  will not try to unroll partially because "
              << "-unroll-allow-partial not given\n");
        return false;
      }
      // Reduce unroll count to be modulo of TripCount for partial unrolling
      Count = UnrollThreshold / LoopSize;
      while (Count != 0 && TripCount%Count != 0) {
        Count--;
      }
      if (Count < 2) {
        DEBUG(errs() << "  could not unroll partially\n");
        return false;
      }
      DEBUG(errs() << "  partially unrolling with count: " << Count << "\n");
    }
  }

  // Unroll the loop.
  Function *F = L->getHeader()->getParent();
  if (!UnrollLoop(L, Count, LI, &LPM))
    return false;

  // FIXME: Reconstruct dom info, because it is not preserved properly.
  DominatorTree *DT = getAnalysisIfAvailable<DominatorTree>();
  if (DT) {
    DT->runOnFunction(*F);
    DominanceFrontier *DF = getAnalysisIfAvailable<DominanceFrontier>();
    if (DF)
      DF->runOnFunction(*F);
  }
  return true;
}
