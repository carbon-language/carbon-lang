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

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"
#include <climits>

using namespace llvm;

#define DEBUG_TYPE "loop-unroll"

static cl::opt<unsigned>
UnrollThreshold("unroll-threshold", cl::init(150), cl::Hidden,
  cl::desc("The cut-off point for automatic loop unrolling"));

static cl::opt<unsigned> UnrollMaxIterationsCountToAnalyze(
    "unroll-max-iteration-count-to-analyze", cl::init(0), cl::Hidden,
    cl::desc("Don't allow loop unrolling to simulate more than this number of"
             "iterations when checking full unroll profitability"));

static cl::opt<unsigned> UnrollMinPercentOfOptimized(
    "unroll-percent-of-optimized-for-complete-unroll", cl::init(20), cl::Hidden,
    cl::desc("If complete unrolling could trigger further optimizations, and, "
             "by that, remove the given percent of instructions, perform the "
             "complete unroll even if it's beyond the threshold"));

static cl::opt<unsigned> UnrollAbsoluteThreshold(
    "unroll-absolute-threshold", cl::init(2000), cl::Hidden,
    cl::desc("Don't unroll if the unrolled size is bigger than this threshold,"
             " even if we can remove big portion of instructions later."));

static cl::opt<unsigned>
UnrollCount("unroll-count", cl::init(0), cl::Hidden,
  cl::desc("Use this unroll count for all loops including those with "
           "unroll_count pragma values, for testing purposes"));

static cl::opt<bool>
UnrollAllowPartial("unroll-allow-partial", cl::init(false), cl::Hidden,
  cl::desc("Allows loops to be partially unrolled until "
           "-unroll-threshold loop size is reached."));

static cl::opt<bool>
UnrollRuntime("unroll-runtime", cl::ZeroOrMore, cl::init(false), cl::Hidden,
  cl::desc("Unroll loops with run-time trip counts"));

static cl::opt<unsigned>
PragmaUnrollThreshold("pragma-unroll-threshold", cl::init(16 * 1024), cl::Hidden,
  cl::desc("Unrolled size limit for loops with an unroll(full) or "
           "unroll_count pragma."));

namespace {
  class LoopUnroll : public LoopPass {
  public:
    static char ID; // Pass ID, replacement for typeid
    LoopUnroll(int T = -1, int C = -1, int P = -1, int R = -1) : LoopPass(ID) {
      CurrentThreshold = (T == -1) ? UnrollThreshold : unsigned(T);
      CurrentAbsoluteThreshold = UnrollAbsoluteThreshold;
      CurrentMinPercentOfOptimized = UnrollMinPercentOfOptimized;
      CurrentCount = (C == -1) ? UnrollCount : unsigned(C);
      CurrentAllowPartial = (P == -1) ? UnrollAllowPartial : (bool)P;
      CurrentRuntime = (R == -1) ? UnrollRuntime : (bool)R;

      UserThreshold = (T != -1) || (UnrollThreshold.getNumOccurrences() > 0);
      UserAbsoluteThreshold = (UnrollAbsoluteThreshold.getNumOccurrences() > 0);
      UserPercentOfOptimized =
          (UnrollMinPercentOfOptimized.getNumOccurrences() > 0);
      UserAllowPartial = (P != -1) ||
                         (UnrollAllowPartial.getNumOccurrences() > 0);
      UserRuntime = (R != -1) || (UnrollRuntime.getNumOccurrences() > 0);
      UserCount = (C != -1) || (UnrollCount.getNumOccurrences() > 0);

      initializeLoopUnrollPass(*PassRegistry::getPassRegistry());
    }

    /// A magic value for use with the Threshold parameter to indicate
    /// that the loop unroll should be performed regardless of how much
    /// code expansion would result.
    static const unsigned NoThreshold = UINT_MAX;

    // Threshold to use when optsize is specified (and there is no
    // explicit -unroll-threshold).
    static const unsigned OptSizeUnrollThreshold = 50;

    // Default unroll count for loops with run-time trip count if
    // -unroll-count is not set
    static const unsigned UnrollRuntimeCount = 8;

    unsigned CurrentCount;
    unsigned CurrentThreshold;
    unsigned CurrentAbsoluteThreshold;
    unsigned CurrentMinPercentOfOptimized;
    bool     CurrentAllowPartial;
    bool     CurrentRuntime;
    bool     UserCount;            // CurrentCount is user-specified.
    bool     UserThreshold;        // CurrentThreshold is user-specified.
    bool UserAbsoluteThreshold;    // CurrentAbsoluteThreshold is
                                   // user-specified.
    bool UserPercentOfOptimized;   // CurrentMinPercentOfOptimized is
                                   // user-specified.
    bool     UserAllowPartial;     // CurrentAllowPartial is user-specified.
    bool     UserRuntime;          // CurrentRuntime is user-specified.

    bool runOnLoop(Loop *L, LPPassManager &LPM) override;

    /// This transformation requires natural loop information & requires that
    /// loop preheaders be inserted into the CFG...
    ///
    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<AssumptionCacheTracker>();
      AU.addRequired<LoopInfoWrapperPass>();
      AU.addPreserved<LoopInfoWrapperPass>();
      AU.addRequiredID(LoopSimplifyID);
      AU.addPreservedID(LoopSimplifyID);
      AU.addRequiredID(LCSSAID);
      AU.addPreservedID(LCSSAID);
      AU.addRequired<ScalarEvolution>();
      AU.addPreserved<ScalarEvolution>();
      AU.addRequired<TargetTransformInfoWrapperPass>();
      // FIXME: Loop unroll requires LCSSA. And LCSSA requires dom info.
      // If loop unroll does not preserve dom info then LCSSA pass on next
      // loop will receive invalid dom info.
      // For now, recreate dom info, if loop is unrolled.
      AU.addPreserved<DominatorTreeWrapperPass>();
    }

    // Fill in the UnrollingPreferences parameter with values from the
    // TargetTransformationInfo.
    void getUnrollingPreferences(Loop *L, const TargetTransformInfo &TTI,
                                 TargetTransformInfo::UnrollingPreferences &UP) {
      UP.Threshold = CurrentThreshold;
      UP.AbsoluteThreshold = CurrentAbsoluteThreshold;
      UP.MinPercentOfOptimized = CurrentMinPercentOfOptimized;
      UP.OptSizeThreshold = OptSizeUnrollThreshold;
      UP.PartialThreshold = CurrentThreshold;
      UP.PartialOptSizeThreshold = OptSizeUnrollThreshold;
      UP.Count = CurrentCount;
      UP.MaxCount = UINT_MAX;
      UP.Partial = CurrentAllowPartial;
      UP.Runtime = CurrentRuntime;
      UP.AllowExpensiveTripCount = false;
      TTI.getUnrollingPreferences(L, UP);
    }

    // Select and return an unroll count based on parameters from
    // user, unroll preferences, unroll pragmas, or a heuristic.
    // SetExplicitly is set to true if the unroll count is is set by
    // the user or a pragma rather than selected heuristically.
    unsigned
    selectUnrollCount(const Loop *L, unsigned TripCount, bool PragmaFullUnroll,
                      unsigned PragmaCount,
                      const TargetTransformInfo::UnrollingPreferences &UP,
                      bool &SetExplicitly);

    // Select threshold values used to limit unrolling based on a
    // total unrolled size.  Parameters Threshold and PartialThreshold
    // are set to the maximum unrolled size for fully and partially
    // unrolled loops respectively.
    void selectThresholds(const Loop *L, bool HasPragma,
                          const TargetTransformInfo::UnrollingPreferences &UP,
                          unsigned &Threshold, unsigned &PartialThreshold,
                          unsigned &AbsoluteThreshold,
                          unsigned &PercentOfOptimizedForCompleteUnroll) {
      // Determine the current unrolling threshold.  While this is
      // normally set from UnrollThreshold, it is overridden to a
      // smaller value if the current function is marked as
      // optimize-for-size, and the unroll threshold was not user
      // specified.
      Threshold = UserThreshold ? CurrentThreshold : UP.Threshold;
      PartialThreshold = UserThreshold ? CurrentThreshold : UP.PartialThreshold;
      AbsoluteThreshold = UserAbsoluteThreshold ? CurrentAbsoluteThreshold
                                                : UP.AbsoluteThreshold;
      PercentOfOptimizedForCompleteUnroll = UserPercentOfOptimized
                                                ? CurrentMinPercentOfOptimized
                                                : UP.MinPercentOfOptimized;

      if (!UserThreshold &&
          L->getHeader()->getParent()->hasFnAttribute(
              Attribute::OptimizeForSize)) {
        Threshold = UP.OptSizeThreshold;
        PartialThreshold = UP.PartialOptSizeThreshold;
      }
      if (HasPragma) {
        // If the loop has an unrolling pragma, we want to be more
        // aggressive with unrolling limits.  Set thresholds to at
        // least the PragmaTheshold value which is larger than the
        // default limits.
        if (Threshold != NoThreshold)
          Threshold = std::max<unsigned>(Threshold, PragmaUnrollThreshold);
        if (PartialThreshold != NoThreshold)
          PartialThreshold =
              std::max<unsigned>(PartialThreshold, PragmaUnrollThreshold);
      }
    }
    bool canUnrollCompletely(Loop *L, unsigned Threshold,
                             unsigned AbsoluteThreshold, uint64_t UnrolledSize,
                             unsigned NumberOfOptimizedInstructions,
                             unsigned PercentOfOptimizedForCompleteUnroll);
  };
}

char LoopUnroll::ID = 0;
INITIALIZE_PASS_BEGIN(LoopUnroll, "loop-unroll", "Unroll loops", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(LCSSA)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_PASS_END(LoopUnroll, "loop-unroll", "Unroll loops", false, false)

Pass *llvm::createLoopUnrollPass(int Threshold, int Count, int AllowPartial,
                                 int Runtime) {
  return new LoopUnroll(Threshold, Count, AllowPartial, Runtime);
}

Pass *llvm::createSimpleLoopUnrollPass() {
  return llvm::createLoopUnrollPass(-1, -1, 0, 0);
}

namespace {
/// \brief SCEV expressions visitor used for finding expressions that would
/// become constants if the loop L is unrolled.
struct FindConstantPointers {
  /// \brief Shows whether the expression is ConstAddress+Constant or not.
  bool IndexIsConstant;

  /// \brief Used for filtering out SCEV expressions with two or more AddRec
  /// subexpressions.
  ///
  /// Used to filter out complicated SCEV expressions, having several AddRec
  /// sub-expressions. We don't handle them, because unrolling one loop
  /// would help to replace only one of these inductions with a constant, and
  /// consequently, the expression would remain non-constant.
  bool HaveSeenAR;

  /// \brief If the SCEV expression becomes ConstAddress+Constant, this value
  /// holds ConstAddress. Otherwise, it's nullptr.
  Value *BaseAddress;

  /// \brief The loop, which we try to completely unroll.
  const Loop *L;

  ScalarEvolution &SE;

  FindConstantPointers(const Loop *L, ScalarEvolution &SE)
      : IndexIsConstant(true), HaveSeenAR(false), BaseAddress(nullptr),
        L(L), SE(SE) {}

  /// Examine the given expression S and figure out, if it can be a part of an
  /// expression, that could become a constant after the loop is unrolled.
  /// The routine sets IndexIsConstant and HaveSeenAR according to the analysis
  /// results.
  /// \returns true if we need to examine subexpressions, and false otherwise.
  bool follow(const SCEV *S) {
    if (const SCEVUnknown *SC = dyn_cast<SCEVUnknown>(S)) {
      // We've reached the leaf node of SCEV, it's most probably just a
      // variable.
      // If it's the only one SCEV-subexpression, then it might be a base
      // address of an index expression.
      // If we've already recorded base address, then just give up on this SCEV
      // - it's too complicated.
      if (BaseAddress) {
        IndexIsConstant = false;
        return false;
      }
      BaseAddress = SC->getValue();
      return false;
    }
    if (isa<SCEVConstant>(S))
      return false;
    if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(S)) {
      // If the current SCEV expression is AddRec, and its loop isn't the loop
      // we are about to unroll, then we won't get a constant address after
      // unrolling, and thus, won't be able to eliminate the load.
      if (AR->getLoop() != L) {
        IndexIsConstant = false;
        return false;
      }
      // We don't handle multiple AddRecs here, so give up in this case.
      if (HaveSeenAR) {
        IndexIsConstant = false;
        return false;
      }
      HaveSeenAR = true;
    }

    // Continue traversal.
    return true;
  }
  bool isDone() const { return !IndexIsConstant; }
};

// This class is used to get an estimate of the optimization effects that we
// could get from complete loop unrolling. It comes from the fact that some
// loads might be replaced with concrete constant values and that could trigger
// a chain of instruction simplifications.
//
// E.g. we might have:
//   int a[] = {0, 1, 0};
//   v = 0;
//   for (i = 0; i < 3; i ++)
//     v += b[i]*a[i];
// If we completely unroll the loop, we would get:
//   v = b[0]*a[0] + b[1]*a[1] + b[2]*a[2]
// Which then will be simplified to:
//   v = b[0]* 0 + b[1]* 1 + b[2]* 0
// And finally:
//   v = b[1]
class UnrollAnalyzer : public InstVisitor<UnrollAnalyzer, bool> {
  typedef InstVisitor<UnrollAnalyzer, bool> Base;
  friend class InstVisitor<UnrollAnalyzer, bool>;

  struct SCEVGEPDescriptor {
    Value *BaseAddr;
    unsigned Start;
    unsigned Step;
  };

  /// \brief The loop we're going to analyze.
  const Loop *L;

  /// \brief TripCount of the given loop.
  unsigned TripCount;

  ScalarEvolution &SE;

  const TargetTransformInfo &TTI;

  // While we walk the loop instructions, we we build up and maintain a mapping
  // of simplified values specific to this iteration.  The idea is to propagate
  // any special information we have about loads that can be replaced with
  // constants after complete unrolling, and account for likely simplifications
  // post-unrolling.
  DenseMap<Value *, Constant *> SimplifiedValues;

  // To avoid requesting SCEV info on every iteration, request it once, and
  // for each value that would become ConstAddress+Constant after loop
  // unrolling, save the corresponding data.
  SmallDenseMap<Value *, SCEVGEPDescriptor> SCEVCache;

  /// \brief Number of currently simulated iteration.
  ///
  /// If an expression is ConstAddress+Constant, then the Constant is
  /// Start + Iteration*Step, where Start and Step could be obtained from
  /// SCEVCache.
  unsigned Iteration;

  /// \brief Upper threshold for complete unrolling.
  unsigned MaxUnrolledLoopSize;

  /// Base case for the instruction visitor.
  bool visitInstruction(Instruction &I) { return false; };

  /// TODO: Add visitors for other instruction types, e.g. ZExt, SExt.

  /// Try to simplify binary operator I.
  ///
  /// TODO: Probaly it's worth to hoist the code for estimating the
  /// simplifications effects to a separate class, since we have a very similar
  /// code in InlineCost already.
  bool visitBinaryOperator(BinaryOperator &I) {
    Value *LHS = I.getOperand(0), *RHS = I.getOperand(1);
    if (!isa<Constant>(LHS))
      if (Constant *SimpleLHS = SimplifiedValues.lookup(LHS))
        LHS = SimpleLHS;
    if (!isa<Constant>(RHS))
      if (Constant *SimpleRHS = SimplifiedValues.lookup(RHS))
        RHS = SimpleRHS;
    Value *SimpleV = nullptr;
    const DataLayout &DL = I.getModule()->getDataLayout();
    if (auto FI = dyn_cast<FPMathOperator>(&I))
      SimpleV =
          SimplifyFPBinOp(I.getOpcode(), LHS, RHS, FI->getFastMathFlags(), DL);
    else
      SimpleV = SimplifyBinOp(I.getOpcode(), LHS, RHS, DL);

    if (SimpleV)
      NumberOfOptimizedInstructions += TTI.getUserCost(&I);

    if (Constant *C = dyn_cast_or_null<Constant>(SimpleV)) {
      SimplifiedValues[&I] = C;
      return true;
    }
    return false;
  }

  /// Try to fold load I.
  bool visitLoad(LoadInst &I) {
    Value *AddrOp = I.getPointerOperand();
    if (!isa<Constant>(AddrOp))
      if (Constant *SimplifiedAddrOp = SimplifiedValues.lookup(AddrOp))
        AddrOp = SimplifiedAddrOp;

    auto It = SCEVCache.find(AddrOp);
    if (It == SCEVCache.end())
      return false;
    SCEVGEPDescriptor GEPDesc = It->second;

    auto GV = dyn_cast<GlobalVariable>(GEPDesc.BaseAddr);
    // We're only interested in loads that can be completely folded to a
    // constant.
    if (!GV || !GV->hasInitializer())
      return false;

    ConstantDataSequential *CDS =
        dyn_cast<ConstantDataSequential>(GV->getInitializer());
    if (!CDS)
      return false;

    // This calculation should never overflow because we bound Iteration quite
    // low and both the start and step are 32-bit integers. We use signed
    // integers so that UBSan will catch if a bug sneaks into the code.
    int ElemSize = CDS->getElementType()->getPrimitiveSizeInBits() / 8U;
    int64_t Index = ((int64_t)GEPDesc.Start +
                      (int64_t)GEPDesc.Step * (int64_t)Iteration) /
                     ElemSize;
    if (Index >= CDS->getNumElements()) {
      // FIXME: For now we conservatively ignore out of bound accesses, but
      // we're allowed to perform the optimization in this case.
      return false;
    }

    Constant *CV = CDS->getElementAsConstant(Index);
    assert(CV && "Constant expected.");
    SimplifiedValues[&I] = CV;

    NumberOfOptimizedInstructions += TTI.getUserCost(&I);
    return true;
  }

  /// Visit all GEPs in the loop and find those which after complete loop
  /// unrolling would become a constant, or BaseAddress+Constant.
  ///
  /// Such GEPs could allow to evaluate a load to a constant later - for now we
  /// just store the corresponding BaseAddress and StartValue with StepValue in
  /// the SCEVCache.
  void cacheSCEVResults() {
    for (auto BB : L->getBlocks()) {
      for (Instruction &I : *BB) {
        if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(&I)) {
          Value *V = cast<Value>(GEP);
          if (!SE.isSCEVable(V->getType()))
              continue;
          const SCEV *S = SE.getSCEV(V);
          // FIXME: Hoist the initialization out of the loop.
          FindConstantPointers Visitor(L, SE);
          SCEVTraversal<FindConstantPointers> T(Visitor);
          // Try to find (BaseAddress+Step+Offset) tuple.
          // If succeeded, save it to the cache - it might help in folding
          // loads.
          T.visitAll(S);
          if (!Visitor.IndexIsConstant || !Visitor.BaseAddress)
            continue;

          const SCEV *BaseAddrSE = SE.getSCEV(Visitor.BaseAddress);
          if (BaseAddrSE->getType() != S->getType())
            continue;
          const SCEV *OffSE = SE.getMinusSCEV(S, BaseAddrSE);
          const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(OffSE);

          if (!AR)
            continue;

          const SCEVConstant *StepSE =
              dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE));
          const SCEVConstant *StartSE = dyn_cast<SCEVConstant>(AR->getStart());
          if (!StepSE || !StartSE)
            continue;

          // Check and skip caching if doing so would require lots of bits to
          // avoid overflow.
          APInt Start = StartSE->getValue()->getValue();
          APInt Step = StepSE->getValue()->getValue();
          if (Start.getActiveBits() > 32 || Step.getActiveBits() > 32)
            continue;

          // We found a cacheable SCEV model for the GEP.
          SCEVCache[V] = {Visitor.BaseAddress,
                          (unsigned)Start.getLimitedValue(),
                          (unsigned)Step.getLimitedValue()};
        }
      }
    }
  }

public:
  UnrollAnalyzer(const Loop *L, unsigned TripCount, ScalarEvolution &SE,
                 const TargetTransformInfo &TTI, unsigned MaxUnrolledLoopSize)
      : L(L), TripCount(TripCount), SE(SE), TTI(TTI),
        MaxUnrolledLoopSize(MaxUnrolledLoopSize),
        NumberOfOptimizedInstructions(0), UnrolledLoopSize(0) {}

  /// \brief Count the number of optimized instructions.
  unsigned NumberOfOptimizedInstructions;

  /// \brief Count the total number of instructions.
  unsigned UnrolledLoopSize;

  /// \brief Figure out if the loop is worth full unrolling.
  ///
  /// Complete loop unrolling can make some loads constant, and we need to know
  /// if that would expose any further optimization opportunities.  This routine
  /// estimates this optimization.  It assigns computed number of instructions,
  /// that potentially might be optimized away, to
  /// NumberOfOptimizedInstructions, and total number of instructions to
  /// UnrolledLoopSize (not counting blocks that won't be reached, if we were
  /// able to compute the condition).
  /// \returns false if we can't analyze the loop, or if we discovered that
  /// unrolling won't give anything. Otherwise, returns true.
  bool analyzeLoop() {
    SmallSetVector<BasicBlock *, 16> BBWorklist;

    // We want to be able to scale offsets by the trip count and add more
    // offsets to them without checking for overflows, and we already don't want
    // to analyze *massive* trip counts, so we force the max to be reasonably
    // small.
    assert(UnrollMaxIterationsCountToAnalyze < (INT_MAX / 2) &&
           "The unroll iterations max is too large!");

    // Don't simulate loops with a big or unknown tripcount
    if (!UnrollMaxIterationsCountToAnalyze || !TripCount ||
        TripCount > UnrollMaxIterationsCountToAnalyze)
      return false;

    // To avoid compute SCEV-expressions on every iteration, compute them once
    // and store interesting to us in SCEVCache.
    cacheSCEVResults();

    // Simulate execution of each iteration of the loop counting instructions,
    // which would be simplified.
    // Since the same load will take different values on different iterations,
    // we literally have to go through all loop's iterations.
    for (Iteration = 0; Iteration < TripCount; ++Iteration) {
      SimplifiedValues.clear();
      BBWorklist.clear();
      BBWorklist.insert(L->getHeader());
      // Note that we *must not* cache the size, this loop grows the worklist.
      for (unsigned Idx = 0; Idx != BBWorklist.size(); ++Idx) {
        BasicBlock *BB = BBWorklist[Idx];

        // Visit all instructions in the given basic block and try to simplify
        // it.  We don't change the actual IR, just count optimization
        // opportunities.
        for (Instruction &I : *BB) {
          UnrolledLoopSize += TTI.getUserCost(&I);
          Base::visit(I);
          // If unrolled body turns out to be too big, bail out.
          if (UnrolledLoopSize - NumberOfOptimizedInstructions >
              MaxUnrolledLoopSize)
            return false;
        }

        // Add BB's successors to the worklist.
        for (BasicBlock *Succ : successors(BB))
          if (L->contains(Succ))
            BBWorklist.insert(Succ);
      }

      // If we found no optimization opportunities on the first iteration, we
      // won't find them on later ones too.
      if (!NumberOfOptimizedInstructions)
        return false;
    }
    return true;
  }
};
} // namespace

/// ApproximateLoopSize - Approximate the size of the loop.
static unsigned ApproximateLoopSize(const Loop *L, unsigned &NumCalls,
                                    bool &NotDuplicatable,
                                    const TargetTransformInfo &TTI,
                                    AssumptionCache *AC) {
  SmallPtrSet<const Value *, 32> EphValues;
  CodeMetrics::collectEphemeralValues(L, AC, EphValues);

  CodeMetrics Metrics;
  for (Loop::block_iterator I = L->block_begin(), E = L->block_end();
       I != E; ++I)
    Metrics.analyzeBasicBlock(*I, TTI, EphValues);
  NumCalls = Metrics.NumInlineCandidates;
  NotDuplicatable = Metrics.notDuplicatable;

  unsigned LoopSize = Metrics.NumInsts;

  // Don't allow an estimate of size zero.  This would allows unrolling of loops
  // with huge iteration counts, which is a compile time problem even if it's
  // not a problem for code quality. Also, the code using this size may assume
  // that each loop has at least three instructions (likely a conditional
  // branch, a comparison feeding that branch, and some kind of loop increment
  // feeding that comparison instruction).
  LoopSize = std::max(LoopSize, 3u);

  return LoopSize;
}

// Returns the loop hint metadata node with the given name (for example,
// "llvm.loop.unroll.count").  If no such metadata node exists, then nullptr is
// returned.
static MDNode *GetUnrollMetadataForLoop(const Loop *L, StringRef Name) {
  if (MDNode *LoopID = L->getLoopID())
    return GetUnrollMetadata(LoopID, Name);
  return nullptr;
}

// Returns true if the loop has an unroll(full) pragma.
static bool HasUnrollFullPragma(const Loop *L) {
  return GetUnrollMetadataForLoop(L, "llvm.loop.unroll.full");
}

// Returns true if the loop has an unroll(disable) pragma.
static bool HasUnrollDisablePragma(const Loop *L) {
  return GetUnrollMetadataForLoop(L, "llvm.loop.unroll.disable");
}

// Returns true if the loop has an runtime unroll(disable) pragma.
static bool HasRuntimeUnrollDisablePragma(const Loop *L) {
  return GetUnrollMetadataForLoop(L, "llvm.loop.unroll.runtime.disable");
}

// If loop has an unroll_count pragma return the (necessarily
// positive) value from the pragma.  Otherwise return 0.
static unsigned UnrollCountPragmaValue(const Loop *L) {
  MDNode *MD = GetUnrollMetadataForLoop(L, "llvm.loop.unroll.count");
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

// Remove existing unroll metadata and add unroll disable metadata to
// indicate the loop has already been unrolled.  This prevents a loop
// from being unrolled more than is directed by a pragma if the loop
// unrolling pass is run more than once (which it generally is).
static void SetLoopAlreadyUnrolled(Loop *L) {
  MDNode *LoopID = L->getLoopID();
  if (!LoopID) return;

  // First remove any existing loop unrolling metadata.
  SmallVector<Metadata *, 4> MDs;
  // Reserve first location for self reference to the LoopID metadata node.
  MDs.push_back(nullptr);
  for (unsigned i = 1, ie = LoopID->getNumOperands(); i < ie; ++i) {
    bool IsUnrollMetadata = false;
    MDNode *MD = dyn_cast<MDNode>(LoopID->getOperand(i));
    if (MD) {
      const MDString *S = dyn_cast<MDString>(MD->getOperand(0));
      IsUnrollMetadata = S && S->getString().startswith("llvm.loop.unroll.");
    }
    if (!IsUnrollMetadata)
      MDs.push_back(LoopID->getOperand(i));
  }

  // Add unroll(disable) metadata to disable future unrolling.
  LLVMContext &Context = L->getHeader()->getContext();
  SmallVector<Metadata *, 1> DisableOperands;
  DisableOperands.push_back(MDString::get(Context, "llvm.loop.unroll.disable"));
  MDNode *DisableNode = MDNode::get(Context, DisableOperands);
  MDs.push_back(DisableNode);

  MDNode *NewLoopID = MDNode::get(Context, MDs);
  // Set operand 0 to refer to the loop id itself.
  NewLoopID->replaceOperandWith(0, NewLoopID);
  L->setLoopID(NewLoopID);
}

bool LoopUnroll::canUnrollCompletely(
    Loop *L, unsigned Threshold, unsigned AbsoluteThreshold,
    uint64_t UnrolledSize, unsigned NumberOfOptimizedInstructions,
    unsigned PercentOfOptimizedForCompleteUnroll) {

  if (Threshold == NoThreshold) {
    DEBUG(dbgs() << "  Can fully unroll, because no threshold is set.\n");
    return true;
  }

  if (UnrolledSize <= Threshold) {
    DEBUG(dbgs() << "  Can fully unroll, because unrolled size: "
                 << UnrolledSize << "<" << Threshold << "\n");
    return true;
  }

  assert(UnrolledSize && "UnrolledSize can't be 0 at this point.");
  unsigned PercentOfOptimizedInstructions =
      (uint64_t)NumberOfOptimizedInstructions * 100ull / UnrolledSize;

  if (UnrolledSize <= AbsoluteThreshold &&
      PercentOfOptimizedInstructions >= PercentOfOptimizedForCompleteUnroll) {
    DEBUG(dbgs() << "  Can fully unroll, because unrolling will help removing "
                 << PercentOfOptimizedInstructions
                 << "% instructions (threshold: "
                 << PercentOfOptimizedForCompleteUnroll << "%)\n");
    DEBUG(dbgs() << "  Unrolled size (" << UnrolledSize
                 << ") is less than the threshold (" << AbsoluteThreshold
                 << ").\n");
    return true;
  }

  DEBUG(dbgs() << "  Too large to fully unroll:\n");
  DEBUG(dbgs() << "    Unrolled size: " << UnrolledSize << "\n");
  DEBUG(dbgs() << "    Estimated number of optimized instructions: "
               << NumberOfOptimizedInstructions << "\n");
  DEBUG(dbgs() << "    Absolute threshold: " << AbsoluteThreshold << "\n");
  DEBUG(dbgs() << "    Minimum percent of removed instructions: "
               << PercentOfOptimizedForCompleteUnroll << "\n");
  DEBUG(dbgs() << "    Threshold for small loops: " << Threshold << "\n");
  return false;
}

unsigned LoopUnroll::selectUnrollCount(
    const Loop *L, unsigned TripCount, bool PragmaFullUnroll,
    unsigned PragmaCount, const TargetTransformInfo::UnrollingPreferences &UP,
    bool &SetExplicitly) {
  SetExplicitly = true;

  // User-specified count (either as a command-line option or
  // constructor parameter) has highest precedence.
  unsigned Count = UserCount ? CurrentCount : 0;

  // If there is no user-specified count, unroll pragmas have the next
  // highest precendence.
  if (Count == 0) {
    if (PragmaCount) {
      Count = PragmaCount;
    } else if (PragmaFullUnroll) {
      Count = TripCount;
    }
  }

  if (Count == 0)
    Count = UP.Count;

  if (Count == 0) {
    SetExplicitly = false;
    if (TripCount == 0)
      // Runtime trip count.
      Count = UnrollRuntimeCount;
    else
      // Conservative heuristic: if we know the trip count, see if we can
      // completely unroll (subject to the threshold, checked below); otherwise
      // try to find greatest modulo of the trip count which is still under
      // threshold value.
      Count = TripCount;
  }
  if (TripCount && Count > TripCount)
    return TripCount;
  return Count;
}

bool LoopUnroll::runOnLoop(Loop *L, LPPassManager &LPM) {
  if (skipOptnoneFunction(L))
    return false;

  Function &F = *L->getHeader()->getParent();

  LoopInfo *LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  ScalarEvolution *SE = &getAnalysis<ScalarEvolution>();
  const TargetTransformInfo &TTI =
      getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
  auto &AC = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);

  BasicBlock *Header = L->getHeader();
  DEBUG(dbgs() << "Loop Unroll: F[" << Header->getParent()->getName()
        << "] Loop %" << Header->getName() << "\n");

  if (HasUnrollDisablePragma(L)) {
    return false;
  }
  bool PragmaFullUnroll = HasUnrollFullPragma(L);
  unsigned PragmaCount = UnrollCountPragmaValue(L);
  bool HasPragma = PragmaFullUnroll || PragmaCount > 0;

  TargetTransformInfo::UnrollingPreferences UP;
  getUnrollingPreferences(L, TTI, UP);

  // Find trip count and trip multiple if count is not available
  unsigned TripCount = 0;
  unsigned TripMultiple = 1;
  // If there are multiple exiting blocks but one of them is the latch, use the
  // latch for the trip count estimation. Otherwise insist on a single exiting
  // block for the trip count estimation.
  BasicBlock *ExitingBlock = L->getLoopLatch();
  if (!ExitingBlock || !L->isLoopExiting(ExitingBlock))
    ExitingBlock = L->getExitingBlock();
  if (ExitingBlock) {
    TripCount = SE->getSmallConstantTripCount(L, ExitingBlock);
    TripMultiple = SE->getSmallConstantTripMultiple(L, ExitingBlock);
  }

  // Select an initial unroll count.  This may be reduced later based
  // on size thresholds.
  bool CountSetExplicitly;
  unsigned Count = selectUnrollCount(L, TripCount, PragmaFullUnroll,
                                     PragmaCount, UP, CountSetExplicitly);

  unsigned NumInlineCandidates;
  bool notDuplicatable;
  unsigned LoopSize =
      ApproximateLoopSize(L, NumInlineCandidates, notDuplicatable, TTI, &AC);
  DEBUG(dbgs() << "  Loop Size = " << LoopSize << "\n");

  // When computing the unrolled size, note that the conditional branch on the
  // backedge and the comparison feeding it are not replicated like the rest of
  // the loop body (which is why 2 is subtracted).
  uint64_t UnrolledSize = (uint64_t)(LoopSize-2) * Count + 2;
  if (notDuplicatable) {
    DEBUG(dbgs() << "  Not unrolling loop which contains non-duplicatable"
                 << " instructions.\n");
    return false;
  }
  if (NumInlineCandidates != 0) {
    DEBUG(dbgs() << "  Not unrolling loop with inlinable calls.\n");
    return false;
  }

  unsigned Threshold, PartialThreshold;
  unsigned AbsoluteThreshold, PercentOfOptimizedForCompleteUnroll;
  selectThresholds(L, HasPragma, UP, Threshold, PartialThreshold,
                   AbsoluteThreshold, PercentOfOptimizedForCompleteUnroll);

  // Given Count, TripCount and thresholds determine the type of
  // unrolling which is to be performed.
  enum { Full = 0, Partial = 1, Runtime = 2 };
  int Unrolling;
  if (TripCount && Count == TripCount) {
    Unrolling = Partial;
    // If the loop is really small, we don't need to run an expensive analysis.
    if (canUnrollCompletely(
            L, Threshold, AbsoluteThreshold,
            UnrolledSize, 0, 100)) {
      Unrolling = Full;
    } else {
      // The loop isn't that small, but we still can fully unroll it if that
      // helps to remove a significant number of instructions.
      // To check that, run additional analysis on the loop.
      UnrollAnalyzer UA(L, TripCount, *SE, TTI, AbsoluteThreshold);
      if (UA.analyzeLoop() &&
          canUnrollCompletely(L, Threshold, AbsoluteThreshold,
                              UA.UnrolledLoopSize,
                              UA.NumberOfOptimizedInstructions,
                              PercentOfOptimizedForCompleteUnroll)) {
        Unrolling = Full;
      }
    }
  } else if (TripCount && Count < TripCount) {
    Unrolling = Partial;
  } else {
    Unrolling = Runtime;
  }

  // Reduce count based on the type of unrolling and the threshold values.
  unsigned OriginalCount = Count;
  bool AllowRuntime = UserRuntime ? CurrentRuntime : UP.Runtime;
  if (HasRuntimeUnrollDisablePragma(L)) {
    AllowRuntime = false;
  }
  if (Unrolling == Partial) {
    bool AllowPartial = UserAllowPartial ? CurrentAllowPartial : UP.Partial;
    if (!AllowPartial && !CountSetExplicitly) {
      DEBUG(dbgs() << "  will not try to unroll partially because "
                   << "-unroll-allow-partial not given\n");
      return false;
    }
    if (PartialThreshold != NoThreshold && UnrolledSize > PartialThreshold) {
      // Reduce unroll count to be modulo of TripCount for partial unrolling.
      Count = (std::max(PartialThreshold, 3u)-2) / (LoopSize-2);
      while (Count != 0 && TripCount % Count != 0)
        Count--;
    }
  } else if (Unrolling == Runtime) {
    if (!AllowRuntime && !CountSetExplicitly) {
      DEBUG(dbgs() << "  will not try to unroll loop with runtime trip count "
                   << "-unroll-runtime not given\n");
      return false;
    }
    // Reduce unroll count to be the largest power-of-two factor of
    // the original count which satisfies the threshold limit.
    while (Count != 0 && UnrolledSize > PartialThreshold) {
      Count >>= 1;
      UnrolledSize = (LoopSize-2) * Count + 2;
    }
    if (Count > UP.MaxCount)
      Count = UP.MaxCount;
    DEBUG(dbgs() << "  partially unrolling with count: " << Count << "\n");
  }

  if (HasPragma) {
    if (PragmaCount != 0)
      // If loop has an unroll count pragma mark loop as unrolled to prevent
      // unrolling beyond that requested by the pragma.
      SetLoopAlreadyUnrolled(L);

    // Emit optimization remarks if we are unable to unroll the loop
    // as directed by a pragma.
    DebugLoc LoopLoc = L->getStartLoc();
    Function *F = Header->getParent();
    LLVMContext &Ctx = F->getContext();
    if (PragmaFullUnroll && PragmaCount == 0) {
      if (TripCount && Count != TripCount) {
        emitOptimizationRemarkMissed(
            Ctx, DEBUG_TYPE, *F, LoopLoc,
            "Unable to fully unroll loop as directed by unroll(full) pragma "
            "because unrolled size is too large.");
      } else if (!TripCount) {
        emitOptimizationRemarkMissed(
            Ctx, DEBUG_TYPE, *F, LoopLoc,
            "Unable to fully unroll loop as directed by unroll(full) pragma "
            "because loop has a runtime trip count.");
      }
    } else if (PragmaCount > 0 && Count != OriginalCount) {
      emitOptimizationRemarkMissed(
          Ctx, DEBUG_TYPE, *F, LoopLoc,
          "Unable to unroll loop the number of times directed by "
          "unroll_count pragma because unrolled size is too large.");
    }
  }

  if (Unrolling != Full && Count < 2) {
    // Partial unrolling by 1 is a nop.  For full unrolling, a factor
    // of 1 makes sense because loop control can be eliminated.
    return false;
  }

  // Unroll the loop.
  if (!UnrollLoop(L, Count, TripCount, AllowRuntime, UP.AllowExpensiveTripCount,
                  TripMultiple, LI, this, &LPM, &AC))
    return false;

  return true;
}
