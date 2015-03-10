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
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/Analysis/InstructionSimplify.h"
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
                          unsigned NumberOfOptimizedInstructions) {
      // Determine the current unrolling threshold.  While this is
      // normally set from UnrollThreshold, it is overridden to a
      // smaller value if the current function is marked as
      // optimize-for-size, and the unroll threshold was not user
      // specified.
      Threshold = UserThreshold ? CurrentThreshold : UP.Threshold;

      // If we are allowed to completely unroll if we can remove M% of
      // instructions, and we know that with complete unrolling we'll be able
      // to kill N instructions, then we can afford to completely unroll loops
      // with unrolled size up to N*100/M.
      // Adjust the threshold according to that:
      unsigned PercentOfOptimizedForCompleteUnroll =
          UserPercentOfOptimized ? CurrentMinPercentOfOptimized
                                 : UP.MinPercentOfOptimized;
      unsigned AbsoluteThreshold = UserAbsoluteThreshold
                                       ? CurrentAbsoluteThreshold
                                       : UP.AbsoluteThreshold;
      if (PercentOfOptimizedForCompleteUnroll)
        Threshold = std::max<unsigned>(Threshold,
                                       NumberOfOptimizedInstructions * 100 /
                                           PercentOfOptimizedForCompleteUnroll);
      // But don't allow unrolling loops bigger than absolute threshold.
      Threshold = std::min<unsigned>(Threshold, AbsoluteThreshold);

      PartialThreshold = UserThreshold ? CurrentThreshold : UP.PartialThreshold;
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

static bool isLoadFromConstantInitializer(Value *V) {
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V))
    if (GV->isConstant() && GV->hasDefinitiveInitializer())
      return GV->getInitializer();
  return false;
}

struct FindConstantPointers {
  bool LoadCanBeConstantFolded;
  bool IndexIsConstant;
  APInt Step;
  APInt StartValue;
  Value *BaseAddress;
  const Loop *L;
  ScalarEvolution &SE;
  FindConstantPointers(const Loop *loop, ScalarEvolution &SE)
      : LoadCanBeConstantFolded(true), IndexIsConstant(true), L(loop), SE(SE) {}

  bool follow(const SCEV *S) {
    if (const SCEVUnknown *SC = dyn_cast<SCEVUnknown>(S)) {
      // We've reached the leaf node of SCEV, it's most probably just a
      // variable. Now it's time to see if it corresponds to a global constant
      // global (in which case we can eliminate the load), or not.
      BaseAddress = SC->getValue();
      LoadCanBeConstantFolded =
          IndexIsConstant && isLoadFromConstantInitializer(BaseAddress);
      return false;
    }
    if (isa<SCEVConstant>(S))
      return true;
    if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(S)) {
      // If the current SCEV expression is AddRec, and its loop isn't the loop
      // we are about to unroll, then we won't get a constant address after
      // unrolling, and thus, won't be able to eliminate the load.
      if (AR->getLoop() != L)
        return IndexIsConstant = false;
      // If the step isn't constant, we won't get constant addresses in unrolled
      // version. Bail out.
      if (const SCEVConstant *StepSE =
              dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE)))
        Step = StepSE->getValue()->getValue();
      else
        return IndexIsConstant = false;

      return IndexIsConstant;
    }
    // If Result is true, continue traversal.
    // Otherwise, we have found something that prevents us from (possible) load
    // elimination.
    return IndexIsConstant;
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

  const Loop *L;
  unsigned TripCount;
  ScalarEvolution &SE;
  const TargetTransformInfo &TTI;

  DenseMap<Value *, Constant *> SimplifiedValues;
  DenseMap<LoadInst *, Value *> LoadBaseAddresses;
  SmallPtrSet<Instruction *, 32> CountedInstructions;

  /// \brief Count the number of optimized instructions.
  unsigned NumberOfOptimizedInstructions;

  // Provide base case for our instruction visit.
  bool visitInstruction(Instruction &I) { return false; };
  // TODO: We should also visit ICmp, FCmp, GetElementPtr, Trunc, ZExt, SExt,
  // FPTrunc, FPExt, FPToUI, FPToSI, UIToFP, SIToFP, BitCast, Select,
  // ExtractElement, InsertElement, ShuffleVector, ExtractValue, InsertValue.
  //
  // Probaly it's worth to hoist the code for estimating the simplifications
  // effects to a separate class, since we have a very similar code in
  // InlineCost already.
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

    if (SimpleV && CountedInstructions.insert(&I).second)
      NumberOfOptimizedInstructions += TTI.getUserCost(&I);

    if (Constant *C = dyn_cast_or_null<Constant>(SimpleV)) {
      SimplifiedValues[&I] = C;
      return true;
    }
    return false;
  }

  Constant *computeLoadValue(LoadInst *LI, unsigned Iteration) {
    if (!LI)
      return nullptr;
    Value *BaseAddr = LoadBaseAddresses[LI];
    if (!BaseAddr)
      return nullptr;

    auto GV = dyn_cast<GlobalVariable>(BaseAddr);
    if (!GV)
      return nullptr;

    ConstantDataSequential *CDS =
        dyn_cast<ConstantDataSequential>(GV->getInitializer());
    if (!CDS)
      return nullptr;

    const SCEV *BaseAddrSE = SE.getSCEV(BaseAddr);
    const SCEV *S = SE.getSCEV(LI->getPointerOperand());
    const SCEV *OffSE = SE.getMinusSCEV(S, BaseAddrSE);

    APInt StepC, StartC;
    const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(OffSE);
    if (!AR)
      return nullptr;

    if (const SCEVConstant *StepSE =
            dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE)))
      StepC = StepSE->getValue()->getValue();
    else
      return nullptr;

    if (const SCEVConstant *StartSE = dyn_cast<SCEVConstant>(AR->getStart()))
      StartC = StartSE->getValue()->getValue();
    else
      return nullptr;

    unsigned ElemSize = CDS->getElementType()->getPrimitiveSizeInBits() / 8U;
    unsigned Start = StartC.getLimitedValue();
    unsigned Step = StepC.getLimitedValue();

    unsigned Index = (Start + Step * Iteration) / ElemSize;
    if (Index >= CDS->getNumElements())
      return nullptr;

    Constant *CV = CDS->getElementAsConstant(Index);

    return CV;
  }

public:
  UnrollAnalyzer(const Loop *L, unsigned TripCount, ScalarEvolution &SE,
                 const TargetTransformInfo &TTI)
      : L(L), TripCount(TripCount), SE(SE), TTI(TTI),
        NumberOfOptimizedInstructions(0) {}

  // Visit all loads the loop L, and for those that, after complete loop
  // unrolling, would have a constant address and it will point to a known
  // constant initializer, record its base address for future use.  It is used
  // when we estimate number of potentially simplified instructions.
  void findConstFoldableLoads() {
    for (auto BB : L->getBlocks()) {
      for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
        if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
          if (!LI->isSimple())
            continue;
          Value *AddrOp = LI->getPointerOperand();
          const SCEV *S = SE.getSCEV(AddrOp);
          FindConstantPointers Visitor(L, SE);
          SCEVTraversal<FindConstantPointers> T(Visitor);
          T.visitAll(S);
          if (Visitor.IndexIsConstant && Visitor.LoadCanBeConstantFolded) {
            LoadBaseAddresses[LI] = Visitor.BaseAddress;
          }
        }
      }
    }
  }

  // Given a list of loads that could be constant-folded (LoadBaseAddresses),
  // estimate number of optimized instructions after substituting the concrete
  // values for the given Iteration. Also track how many instructions become
  // dead through this process.
  unsigned estimateNumberOfOptimizedInstructions(unsigned Iteration) {
    // We keep a set vector for the worklist so that we don't wast space in the
    // worklist queuing up the same instruction repeatedly. This can happen due
    // to multiple operands being the same instruction or due to the same
    // instruction being an operand of lots of things that end up dead or
    // simplified.
    SmallSetVector<Instruction *, 8> Worklist;

    // Clear the simplified values and counts for this iteration.
    SimplifiedValues.clear();
    CountedInstructions.clear();
    NumberOfOptimizedInstructions = 0;

    // We start by adding all loads to the worklist.
    for (auto &LoadDescr : LoadBaseAddresses) {
      LoadInst *LI = LoadDescr.first;
      SimplifiedValues[LI] = computeLoadValue(LI, Iteration);
      if (CountedInstructions.insert(LI).second)
        NumberOfOptimizedInstructions += TTI.getUserCost(LI);

      for (User *U : LI->users())
        Worklist.insert(cast<Instruction>(U));
    }

    // And then we try to simplify every user of every instruction from the
    // worklist. If we do simplify a user, add it to the worklist to process
    // its users as well.
    while (!Worklist.empty()) {
      Instruction *I = Worklist.pop_back_val();
      if (!L->contains(I))
        continue;
      if (!visit(I))
        continue;
      for (User *U : I->users())
        Worklist.insert(cast<Instruction>(U));
    }

    // Now that we know the potentially simplifed instructions, estimate number
    // of instructions that would become dead if we do perform the
    // simplification.

    // The dead instructions are held in a separate set. This is used to
    // prevent us from re-examining instructions and make sure we only count
    // the benifit once. The worklist's internal set handles insertion
    // deduplication.
    SmallPtrSet<Instruction *, 16> DeadInstructions;

    // Lambda to enque operands onto the worklist.
    auto EnqueueOperands = [&](Instruction &I) {
      for (auto *Op : I.operand_values())
        if (auto *OpI = dyn_cast<Instruction>(Op))
          if (!OpI->use_empty())
            Worklist.insert(OpI);
    };

    // Start by initializing worklist with simplified instructions.
    for (auto &FoldedKeyValue : SimplifiedValues)
      if (auto *FoldedInst = dyn_cast<Instruction>(FoldedKeyValue.first)) {
        DeadInstructions.insert(FoldedInst);

        // Add each instruction operand of this dead instruction to the
        // worklist.
        EnqueueOperands(*FoldedInst);
      }

    // If a definition of an insn is only used by simplified or dead
    // instructions, it's also dead. Check defs of all instructions from the
    // worklist.
    while (!Worklist.empty()) {
      Instruction *I = Worklist.pop_back_val();
      if (!L->contains(I))
        continue;
      if (DeadInstructions.count(I))
        continue;

      if (std::all_of(I->user_begin(), I->user_end(), [&](User *U) {
            return DeadInstructions.count(cast<Instruction>(U));
          })) {
        NumberOfOptimizedInstructions += TTI.getUserCost(I);
        DeadInstructions.insert(I);
        EnqueueOperands(*I);
      }
    }
    return NumberOfOptimizedInstructions;
  }
};

// Complete loop unrolling can make some loads constant, and we need to know if
// that would expose any further optimization opportunities.
// This routine estimates this optimization effect and returns the number of
// instructions, that potentially might be optimized away.
static unsigned
approximateNumberOfOptimizedInstructions(const Loop *L, ScalarEvolution &SE,
                                         unsigned TripCount,
                                         const TargetTransformInfo &TTI) {
  if (!TripCount || !UnrollMaxIterationsCountToAnalyze)
    return 0;

  UnrollAnalyzer UA(L, TripCount, SE, TTI);
  UA.findConstFoldableLoads();

  // Estimate number of instructions, that could be simplified if we replace a
  // load with the corresponding constant. Since the same load will take
  // different values on different iterations, we have to go through all loop's
  // iterations here. To limit ourselves here, we check only first N
  // iterations, and then scale the found number, if necessary.
  unsigned IterationsNumberForEstimate =
      std::min<unsigned>(UnrollMaxIterationsCountToAnalyze, TripCount);
  unsigned NumberOfOptimizedInstructions = 0;
  for (unsigned i = 0; i < IterationsNumberForEstimate; ++i)
    NumberOfOptimizedInstructions +=
        UA.estimateNumberOfOptimizedInstructions(i);

  NumberOfOptimizedInstructions *= TripCount / IterationsNumberForEstimate;

  return NumberOfOptimizedInstructions;
}

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

  unsigned NumberOfOptimizedInstructions =
      approximateNumberOfOptimizedInstructions(L, *SE, TripCount, TTI);
  DEBUG(dbgs() << "  Complete unrolling could save: "
               << NumberOfOptimizedInstructions << "\n");

  unsigned Threshold, PartialThreshold;
  selectThresholds(L, HasPragma, UP, Threshold, PartialThreshold,
                   NumberOfOptimizedInstructions);

  // Given Count, TripCount and thresholds determine the type of
  // unrolling which is to be performed.
  enum { Full = 0, Partial = 1, Runtime = 2 };
  int Unrolling;
  if (TripCount && Count == TripCount) {
    if (Threshold != NoThreshold && UnrolledSize > Threshold) {
      DEBUG(dbgs() << "  Too large to fully unroll with count: " << Count
                   << " because size: " << UnrolledSize << ">" << Threshold
                   << "\n");
      Unrolling = Partial;
    } else {
      Unrolling = Full;
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
  if (!UnrollLoop(L, Count, TripCount, AllowRuntime, TripMultiple, LI, this,
                  &LPM, &AC))
    return false;

  return true;
}
