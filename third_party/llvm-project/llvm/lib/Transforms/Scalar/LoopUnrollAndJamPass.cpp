//===- LoopUnrollAndJam.cpp - Loop unroll and jam pass --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements an unroll and jam pass. Most of the work is done by
// Utils/UnrollLoopAndJam.cpp.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LoopUnrollAndJamPass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/PriorityWorklist.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopNestAnalysis.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Utils/LoopPeel.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"
#include <cassert>
#include <cstdint>

namespace llvm {
class Instruction;
class Value;
} // namespace llvm

using namespace llvm;

#define DEBUG_TYPE "loop-unroll-and-jam"

/// @{
/// Metadata attribute names
static const char *const LLVMLoopUnrollAndJamFollowupAll =
    "llvm.loop.unroll_and_jam.followup_all";
static const char *const LLVMLoopUnrollAndJamFollowupInner =
    "llvm.loop.unroll_and_jam.followup_inner";
static const char *const LLVMLoopUnrollAndJamFollowupOuter =
    "llvm.loop.unroll_and_jam.followup_outer";
static const char *const LLVMLoopUnrollAndJamFollowupRemainderInner =
    "llvm.loop.unroll_and_jam.followup_remainder_inner";
static const char *const LLVMLoopUnrollAndJamFollowupRemainderOuter =
    "llvm.loop.unroll_and_jam.followup_remainder_outer";
/// @}

static cl::opt<bool>
    AllowUnrollAndJam("allow-unroll-and-jam", cl::Hidden,
                      cl::desc("Allows loops to be unroll-and-jammed."));

static cl::opt<unsigned> UnrollAndJamCount(
    "unroll-and-jam-count", cl::Hidden,
    cl::desc("Use this unroll count for all loops including those with "
             "unroll_and_jam_count pragma values, for testing purposes"));

static cl::opt<unsigned> UnrollAndJamThreshold(
    "unroll-and-jam-threshold", cl::init(60), cl::Hidden,
    cl::desc("Threshold to use for inner loop when doing unroll and jam."));

static cl::opt<unsigned> PragmaUnrollAndJamThreshold(
    "pragma-unroll-and-jam-threshold", cl::init(1024), cl::Hidden,
    cl::desc("Unrolled size limit for loops with an unroll_and_jam(full) or "
             "unroll_count pragma."));

// Returns the loop hint metadata node with the given name (for example,
// "llvm.loop.unroll.count").  If no such metadata node exists, then nullptr is
// returned.
static MDNode *getUnrollMetadataForLoop(const Loop *L, StringRef Name) {
  if (MDNode *LoopID = L->getLoopID())
    return GetUnrollMetadata(LoopID, Name);
  return nullptr;
}

// Returns true if the loop has any metadata starting with Prefix. For example a
// Prefix of "llvm.loop.unroll." returns true if we have any unroll metadata.
static bool hasAnyUnrollPragma(const Loop *L, StringRef Prefix) {
  if (MDNode *LoopID = L->getLoopID()) {
    // First operand should refer to the loop id itself.
    assert(LoopID->getNumOperands() > 0 && "requires at least one operand");
    assert(LoopID->getOperand(0) == LoopID && "invalid loop id");

    for (unsigned I = 1, E = LoopID->getNumOperands(); I < E; ++I) {
      MDNode *MD = dyn_cast<MDNode>(LoopID->getOperand(I));
      if (!MD)
        continue;

      MDString *S = dyn_cast<MDString>(MD->getOperand(0));
      if (!S)
        continue;

      if (S->getString().startswith(Prefix))
        return true;
    }
  }
  return false;
}

// Returns true if the loop has an unroll_and_jam(enable) pragma.
static bool hasUnrollAndJamEnablePragma(const Loop *L) {
  return getUnrollMetadataForLoop(L, "llvm.loop.unroll_and_jam.enable");
}

// If loop has an unroll_and_jam_count pragma return the (necessarily
// positive) value from the pragma.  Otherwise return 0.
static unsigned unrollAndJamCountPragmaValue(const Loop *L) {
  MDNode *MD = getUnrollMetadataForLoop(L, "llvm.loop.unroll_and_jam.count");
  if (MD) {
    assert(MD->getNumOperands() == 2 &&
           "Unroll count hint metadata should have two operands.");
    unsigned Count =
        mdconst::extract<ConstantInt>(MD->getOperand(1))->getZExtValue();
    assert(Count >= 1 && "Unroll count must be positive.");
    return Count;
  }
  return 0;
}

// Returns loop size estimation for unrolled loop.
static uint64_t
getUnrollAndJammedLoopSize(unsigned LoopSize,
                           TargetTransformInfo::UnrollingPreferences &UP) {
  assert(LoopSize >= UP.BEInsns && "LoopSize should not be less than BEInsns!");
  return static_cast<uint64_t>(LoopSize - UP.BEInsns) * UP.Count + UP.BEInsns;
}

// Calculates unroll and jam count and writes it to UP.Count. Returns true if
// unroll count was set explicitly.
static bool computeUnrollAndJamCount(
    Loop *L, Loop *SubLoop, const TargetTransformInfo &TTI, DominatorTree &DT,
    LoopInfo *LI, ScalarEvolution &SE,
    const SmallPtrSetImpl<const Value *> &EphValues,
    OptimizationRemarkEmitter *ORE, unsigned OuterTripCount,
    unsigned OuterTripMultiple, unsigned OuterLoopSize, unsigned InnerTripCount,
    unsigned InnerLoopSize, TargetTransformInfo::UnrollingPreferences &UP,
    TargetTransformInfo::PeelingPreferences &PP) {
  // First up use computeUnrollCount from the loop unroller to get a count
  // for unrolling the outer loop, plus any loops requiring explicit
  // unrolling we leave to the unroller. This uses UP.Threshold /
  // UP.PartialThreshold / UP.MaxCount to come up with sensible loop values.
  // We have already checked that the loop has no unroll.* pragmas.
  unsigned MaxTripCount = 0;
  bool UseUpperBound = false;
  bool ExplicitUnroll = computeUnrollCount(
      L, TTI, DT, LI, SE, EphValues, ORE, OuterTripCount, MaxTripCount,
      /*MaxOrZero*/ false, OuterTripMultiple, OuterLoopSize, UP, PP,
      UseUpperBound);
  if (ExplicitUnroll || UseUpperBound) {
    // If the user explicitly set the loop as unrolled, dont UnJ it. Leave it
    // for the unroller instead.
    LLVM_DEBUG(dbgs() << "Won't unroll-and-jam; explicit count set by "
                         "computeUnrollCount\n");
    UP.Count = 0;
    return false;
  }

  // Override with any explicit Count from the "unroll-and-jam-count" option.
  bool UserUnrollCount = UnrollAndJamCount.getNumOccurrences() > 0;
  if (UserUnrollCount) {
    UP.Count = UnrollAndJamCount;
    UP.Force = true;
    if (UP.AllowRemainder &&
        getUnrollAndJammedLoopSize(OuterLoopSize, UP) < UP.Threshold &&
        getUnrollAndJammedLoopSize(InnerLoopSize, UP) <
            UP.UnrollAndJamInnerLoopThreshold)
      return true;
  }

  // Check for unroll_and_jam pragmas
  unsigned PragmaCount = unrollAndJamCountPragmaValue(L);
  if (PragmaCount > 0) {
    UP.Count = PragmaCount;
    UP.Runtime = true;
    UP.Force = true;
    if ((UP.AllowRemainder || (OuterTripMultiple % PragmaCount == 0)) &&
        getUnrollAndJammedLoopSize(OuterLoopSize, UP) < UP.Threshold &&
        getUnrollAndJammedLoopSize(InnerLoopSize, UP) <
            UP.UnrollAndJamInnerLoopThreshold)
      return true;
  }

  bool PragmaEnableUnroll = hasUnrollAndJamEnablePragma(L);
  bool ExplicitUnrollAndJamCount = PragmaCount > 0 || UserUnrollCount;
  bool ExplicitUnrollAndJam = PragmaEnableUnroll || ExplicitUnrollAndJamCount;

  // If the loop has an unrolling pragma, we want to be more aggressive with
  // unrolling limits.
  if (ExplicitUnrollAndJam)
    UP.UnrollAndJamInnerLoopThreshold = PragmaUnrollAndJamThreshold;

  if (!UP.AllowRemainder && getUnrollAndJammedLoopSize(InnerLoopSize, UP) >=
                                UP.UnrollAndJamInnerLoopThreshold) {
    LLVM_DEBUG(dbgs() << "Won't unroll-and-jam; can't create remainder and "
                         "inner loop too large\n");
    UP.Count = 0;
    return false;
  }

  // We have a sensible limit for the outer loop, now adjust it for the inner
  // loop and UP.UnrollAndJamInnerLoopThreshold. If the outer limit was set
  // explicitly, we want to stick to it.
  if (!ExplicitUnrollAndJamCount && UP.AllowRemainder) {
    while (UP.Count != 0 && getUnrollAndJammedLoopSize(InnerLoopSize, UP) >=
                                UP.UnrollAndJamInnerLoopThreshold)
      UP.Count--;
  }

  // If we are explicitly unroll and jamming, we are done. Otherwise there are a
  // number of extra performance heuristics to check.
  if (ExplicitUnrollAndJam)
    return true;

  // If the inner loop count is known and small, leave the entire loop nest to
  // be the unroller
  if (InnerTripCount && InnerLoopSize * InnerTripCount < UP.Threshold) {
    LLVM_DEBUG(dbgs() << "Won't unroll-and-jam; small inner loop count is "
                         "being left for the unroller\n");
    UP.Count = 0;
    return false;
  }

  // Check for situations where UnJ is likely to be unprofitable. Including
  // subloops with more than 1 block.
  if (SubLoop->getBlocks().size() != 1) {
    LLVM_DEBUG(
        dbgs() << "Won't unroll-and-jam; More than one inner loop block\n");
    UP.Count = 0;
    return false;
  }

  // Limit to loops where there is something to gain from unrolling and
  // jamming the loop. In this case, look for loads that are invariant in the
  // outer loop and can become shared.
  unsigned NumInvariant = 0;
  for (BasicBlock *BB : SubLoop->getBlocks()) {
    for (Instruction &I : *BB) {
      if (auto *Ld = dyn_cast<LoadInst>(&I)) {
        Value *V = Ld->getPointerOperand();
        const SCEV *LSCEV = SE.getSCEVAtScope(V, L);
        if (SE.isLoopInvariant(LSCEV, L))
          NumInvariant++;
      }
    }
  }
  if (NumInvariant == 0) {
    LLVM_DEBUG(dbgs() << "Won't unroll-and-jam; No loop invariant loads\n");
    UP.Count = 0;
    return false;
  }

  return false;
}

static LoopUnrollResult
tryToUnrollAndJamLoop(Loop *L, DominatorTree &DT, LoopInfo *LI,
                      ScalarEvolution &SE, const TargetTransformInfo &TTI,
                      AssumptionCache &AC, DependenceInfo &DI,
                      OptimizationRemarkEmitter &ORE, int OptLevel) {
  TargetTransformInfo::UnrollingPreferences UP =
      gatherUnrollingPreferences(L, SE, TTI, nullptr, nullptr, ORE, OptLevel,
                                 None, None, None, None, None, None);
  TargetTransformInfo::PeelingPreferences PP =
      gatherPeelingPreferences(L, SE, TTI, None, None);

  TransformationMode EnableMode = hasUnrollAndJamTransformation(L);
  if (EnableMode & TM_Disable)
    return LoopUnrollResult::Unmodified;
  if (EnableMode & TM_ForcedByUser)
    UP.UnrollAndJam = true;

  if (AllowUnrollAndJam.getNumOccurrences() > 0)
    UP.UnrollAndJam = AllowUnrollAndJam;
  if (UnrollAndJamThreshold.getNumOccurrences() > 0)
    UP.UnrollAndJamInnerLoopThreshold = UnrollAndJamThreshold;
  // Exit early if unrolling is disabled.
  if (!UP.UnrollAndJam || UP.UnrollAndJamInnerLoopThreshold == 0)
    return LoopUnrollResult::Unmodified;

  LLVM_DEBUG(dbgs() << "Loop Unroll and Jam: F["
                    << L->getHeader()->getParent()->getName() << "] Loop %"
                    << L->getHeader()->getName() << "\n");

  // A loop with any unroll pragma (enabling/disabling/count/etc) is left for
  // the unroller, so long as it does not explicitly have unroll_and_jam
  // metadata. This means #pragma nounroll will disable unroll and jam as well
  // as unrolling
  if (hasAnyUnrollPragma(L, "llvm.loop.unroll.") &&
      !hasAnyUnrollPragma(L, "llvm.loop.unroll_and_jam.")) {
    LLVM_DEBUG(dbgs() << "  Disabled due to pragma.\n");
    return LoopUnrollResult::Unmodified;
  }

  if (!isSafeToUnrollAndJam(L, SE, DT, DI, *LI)) {
    LLVM_DEBUG(dbgs() << "  Disabled due to not being safe.\n");
    return LoopUnrollResult::Unmodified;
  }

  // Approximate the loop size and collect useful info
  unsigned NumInlineCandidates;
  bool NotDuplicatable;
  bool Convergent;
  SmallPtrSet<const Value *, 32> EphValues;
  CodeMetrics::collectEphemeralValues(L, &AC, EphValues);
  Loop *SubLoop = L->getSubLoops()[0];
  InstructionCost InnerLoopSizeIC =
      ApproximateLoopSize(SubLoop, NumInlineCandidates, NotDuplicatable,
                          Convergent, TTI, EphValues, UP.BEInsns);
  InstructionCost OuterLoopSizeIC =
      ApproximateLoopSize(L, NumInlineCandidates, NotDuplicatable, Convergent,
                          TTI, EphValues, UP.BEInsns);
  LLVM_DEBUG(dbgs() << "  Outer Loop Size: " << OuterLoopSizeIC << "\n");
  LLVM_DEBUG(dbgs() << "  Inner Loop Size: " << InnerLoopSizeIC << "\n");

  if (!InnerLoopSizeIC.isValid() || !OuterLoopSizeIC.isValid()) {
    LLVM_DEBUG(dbgs() << "  Not unrolling loop which contains instructions"
                      << " with invalid cost.\n");
    return LoopUnrollResult::Unmodified;
  }
  unsigned InnerLoopSize = *InnerLoopSizeIC.getValue();
  unsigned OuterLoopSize = *OuterLoopSizeIC.getValue();

  if (NotDuplicatable) {
    LLVM_DEBUG(dbgs() << "  Not unrolling loop which contains non-duplicatable "
                         "instructions.\n");
    return LoopUnrollResult::Unmodified;
  }
  if (NumInlineCandidates != 0) {
    LLVM_DEBUG(dbgs() << "  Not unrolling loop with inlinable calls.\n");
    return LoopUnrollResult::Unmodified;
  }
  if (Convergent) {
    LLVM_DEBUG(
        dbgs() << "  Not unrolling loop with convergent instructions.\n");
    return LoopUnrollResult::Unmodified;
  }

  // Save original loop IDs for after the transformation.
  MDNode *OrigOuterLoopID = L->getLoopID();
  MDNode *OrigSubLoopID = SubLoop->getLoopID();

  // To assign the loop id of the epilogue, assign it before unrolling it so it
  // is applied to every inner loop of the epilogue. We later apply the loop ID
  // for the jammed inner loop.
  Optional<MDNode *> NewInnerEpilogueLoopID = makeFollowupLoopID(
      OrigOuterLoopID, {LLVMLoopUnrollAndJamFollowupAll,
                        LLVMLoopUnrollAndJamFollowupRemainderInner});
  if (NewInnerEpilogueLoopID.hasValue())
    SubLoop->setLoopID(NewInnerEpilogueLoopID.getValue());

  // Find trip count and trip multiple
  BasicBlock *Latch = L->getLoopLatch();
  BasicBlock *SubLoopLatch = SubLoop->getLoopLatch();
  unsigned OuterTripCount = SE.getSmallConstantTripCount(L, Latch);
  unsigned OuterTripMultiple = SE.getSmallConstantTripMultiple(L, Latch);
  unsigned InnerTripCount = SE.getSmallConstantTripCount(SubLoop, SubLoopLatch);

  // Decide if, and by how much, to unroll
  bool IsCountSetExplicitly = computeUnrollAndJamCount(
      L, SubLoop, TTI, DT, LI, SE, EphValues, &ORE, OuterTripCount,
      OuterTripMultiple, OuterLoopSize, InnerTripCount, InnerLoopSize, UP, PP);
  if (UP.Count <= 1)
    return LoopUnrollResult::Unmodified;
  // Unroll factor (Count) must be less or equal to TripCount.
  if (OuterTripCount && UP.Count > OuterTripCount)
    UP.Count = OuterTripCount;

  Loop *EpilogueOuterLoop = nullptr;
  LoopUnrollResult UnrollResult = UnrollAndJamLoop(
      L, UP.Count, OuterTripCount, OuterTripMultiple, UP.UnrollRemainder, LI,
      &SE, &DT, &AC, &TTI, &ORE, &EpilogueOuterLoop);

  // Assign new loop attributes.
  if (EpilogueOuterLoop) {
    Optional<MDNode *> NewOuterEpilogueLoopID = makeFollowupLoopID(
        OrigOuterLoopID, {LLVMLoopUnrollAndJamFollowupAll,
                          LLVMLoopUnrollAndJamFollowupRemainderOuter});
    if (NewOuterEpilogueLoopID.hasValue())
      EpilogueOuterLoop->setLoopID(NewOuterEpilogueLoopID.getValue());
  }

  Optional<MDNode *> NewInnerLoopID =
      makeFollowupLoopID(OrigOuterLoopID, {LLVMLoopUnrollAndJamFollowupAll,
                                           LLVMLoopUnrollAndJamFollowupInner});
  if (NewInnerLoopID.hasValue())
    SubLoop->setLoopID(NewInnerLoopID.getValue());
  else
    SubLoop->setLoopID(OrigSubLoopID);

  if (UnrollResult == LoopUnrollResult::PartiallyUnrolled) {
    Optional<MDNode *> NewOuterLoopID = makeFollowupLoopID(
        OrigOuterLoopID,
        {LLVMLoopUnrollAndJamFollowupAll, LLVMLoopUnrollAndJamFollowupOuter});
    if (NewOuterLoopID.hasValue()) {
      L->setLoopID(NewOuterLoopID.getValue());

      // Do not setLoopAlreadyUnrolled if a followup was given.
      return UnrollResult;
    }
  }

  // If loop has an unroll count pragma or unrolled by explicitly set count
  // mark loop as unrolled to prevent unrolling beyond that requested.
  if (UnrollResult != LoopUnrollResult::FullyUnrolled && IsCountSetExplicitly)
    L->setLoopAlreadyUnrolled();

  return UnrollResult;
}

static bool tryToUnrollAndJamLoop(LoopNest &LN, DominatorTree &DT, LoopInfo &LI,
                                  ScalarEvolution &SE,
                                  const TargetTransformInfo &TTI,
                                  AssumptionCache &AC, DependenceInfo &DI,
                                  OptimizationRemarkEmitter &ORE, int OptLevel,
                                  LPMUpdater &U) {
  bool DidSomething = false;
  ArrayRef<Loop *> Loops = LN.getLoops();
  Loop *OutmostLoop = &LN.getOutermostLoop();

  // Add the loop nests in the reverse order of LN. See method
  // declaration.
  SmallPriorityWorklist<Loop *, 4> Worklist;
  appendLoopsToWorklist(Loops, Worklist);
  while (!Worklist.empty()) {
    Loop *L = Worklist.pop_back_val();
    std::string LoopName = std::string(L->getName());
    LoopUnrollResult Result =
        tryToUnrollAndJamLoop(L, DT, &LI, SE, TTI, AC, DI, ORE, OptLevel);
    if (Result != LoopUnrollResult::Unmodified)
      DidSomething = true;
    if (L == OutmostLoop && Result == LoopUnrollResult::FullyUnrolled)
      U.markLoopAsDeleted(*L, LoopName);
  }

  return DidSomething;
}

namespace {

class LoopUnrollAndJam : public LoopPass {
public:
  static char ID; // Pass ID, replacement for typeid
  unsigned OptLevel;

  LoopUnrollAndJam(int OptLevel = 2) : LoopPass(ID), OptLevel(OptLevel) {
    initializeLoopUnrollAndJamPass(*PassRegistry::getPassRegistry());
  }

  bool runOnLoop(Loop *L, LPPassManager &LPM) override {
    if (skipLoop(L))
      return false;

    auto *F = L->getHeader()->getParent();
    auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    auto *LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    auto &DI = getAnalysis<DependenceAnalysisWrapperPass>().getDI();
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    auto &TTI = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(*F);
    auto &ORE = getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();
    auto &AC = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(*F);

    LoopUnrollResult Result =
        tryToUnrollAndJamLoop(L, DT, LI, SE, TTI, AC, DI, ORE, OptLevel);

    if (Result == LoopUnrollResult::FullyUnrolled)
      LPM.markLoopAsDeleted(*L);

    return Result != LoopUnrollResult::Unmodified;
  }

  /// This transformation requires natural loop information & requires that
  /// loop preheaders be inserted into the CFG...
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DependenceAnalysisWrapperPass>();
    AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
    getLoopAnalysisUsage(AU);
  }
};

} // end anonymous namespace

char LoopUnrollAndJam::ID = 0;

INITIALIZE_PASS_BEGIN(LoopUnrollAndJam, "loop-unroll-and-jam",
                      "Unroll and Jam loops", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(LCSSAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DependenceAnalysisWrapperPass)
INITIALIZE_PASS_DEPENDENCY(OptimizationRemarkEmitterWrapperPass)
INITIALIZE_PASS_END(LoopUnrollAndJam, "loop-unroll-and-jam",
                    "Unroll and Jam loops", false, false)

Pass *llvm::createLoopUnrollAndJamPass(int OptLevel) {
  return new LoopUnrollAndJam(OptLevel);
}

PreservedAnalyses LoopUnrollAndJamPass::run(LoopNest &LN,
                                            LoopAnalysisManager &AM,
                                            LoopStandardAnalysisResults &AR,
                                            LPMUpdater &U) {
  Function &F = *LN.getParent();

  DependenceInfo DI(&F, &AR.AA, &AR.SE, &AR.LI);
  OptimizationRemarkEmitter ORE(&F);

  if (!tryToUnrollAndJamLoop(LN, AR.DT, AR.LI, AR.SE, AR.TTI, AR.AC, DI, ORE,
                             OptLevel, U))
    return PreservedAnalyses::all();

  auto PA = getLoopPassPreservedAnalyses();
  PA.preserve<LoopNestAnalysis>();
  return PA;
}
