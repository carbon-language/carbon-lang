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
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
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
#include <climits>

using namespace llvm;

#define DEBUG_TYPE "loop-unroll"

static cl::opt<unsigned>
UnrollThreshold("unroll-threshold", cl::init(150), cl::Hidden,
  cl::desc("The cut-off point for automatic loop unrolling"));

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
  cl::desc("Unrolled size limit for loops with an unroll(enable) or "
           "unroll_count pragma."));

namespace {
  class LoopUnroll : public LoopPass {
  public:
    static char ID; // Pass ID, replacement for typeid
    LoopUnroll(int T = -1, int C = -1, int P = -1, int R = -1) : LoopPass(ID) {
      CurrentThreshold = (T == -1) ? UnrollThreshold : unsigned(T);
      CurrentCount = (C == -1) ? UnrollCount : unsigned(C);
      CurrentAllowPartial = (P == -1) ? UnrollAllowPartial : (bool)P;
      CurrentRuntime = (R == -1) ? UnrollRuntime : (bool)R;

      UserThreshold = (T != -1) || (UnrollThreshold.getNumOccurrences() > 0);
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
    bool     CurrentAllowPartial;
    bool     CurrentRuntime;
    bool     UserCount;            // CurrentCount is user-specified.
    bool     UserThreshold;        // CurrentThreshold is user-specified.
    bool     UserAllowPartial;     // CurrentAllowPartial is user-specified.
    bool     UserRuntime;          // CurrentRuntime is user-specified.

    bool runOnLoop(Loop *L, LPPassManager &LPM) override;

    /// This transformation requires natural loop information & requires that
    /// loop preheaders be inserted into the CFG...
    ///
    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<LoopInfo>();
      AU.addPreserved<LoopInfo>();
      AU.addRequiredID(LoopSimplifyID);
      AU.addPreservedID(LoopSimplifyID);
      AU.addRequiredID(LCSSAID);
      AU.addPreservedID(LCSSAID);
      AU.addRequired<ScalarEvolution>();
      AU.addPreserved<ScalarEvolution>();
      AU.addRequired<TargetTransformInfo>();
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
    selectUnrollCount(const Loop *L, unsigned TripCount, bool HasEnablePragma,
                      unsigned PragmaCount,
                      const TargetTransformInfo::UnrollingPreferences &UP,
                      bool &SetExplicitly);


    // Select threshold values used to limit unrolling based on a
    // total unrolled size.  Parameters Threshold and PartialThreshold
    // are set to the maximum unrolled size for fully and partially
    // unrolled loops respectively.
    void selectThresholds(const Loop *L, bool HasPragma,
                          const TargetTransformInfo::UnrollingPreferences &UP,
                          unsigned &Threshold, unsigned &PartialThreshold) {
      // Determine the current unrolling threshold.  While this is
      // normally set from UnrollThreshold, it is overridden to a
      // smaller value if the current function is marked as
      // optimize-for-size, and the unroll threshold was not user
      // specified.
      Threshold = UserThreshold ? CurrentThreshold : UP.Threshold;
      PartialThreshold = UserThreshold ? CurrentThreshold : UP.PartialThreshold;
      if (!UserThreshold &&
          L->getHeader()->getParent()->getAttributes().
              hasAttribute(AttributeSet::FunctionIndex,
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
INITIALIZE_AG_DEPENDENCY(TargetTransformInfo)
INITIALIZE_PASS_DEPENDENCY(LoopInfo)
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

/// ApproximateLoopSize - Approximate the size of the loop.
static unsigned ApproximateLoopSize(const Loop *L, unsigned &NumCalls,
                                    bool &NotDuplicatable,
                                    const TargetTransformInfo &TTI) {
  CodeMetrics Metrics;
  for (Loop::block_iterator I = L->block_begin(), E = L->block_end();
       I != E; ++I)
    Metrics.analyzeBasicBlock(*I, TTI);
  NumCalls = Metrics.NumInlineCandidates;
  NotDuplicatable = Metrics.notDuplicatable;

  unsigned LoopSize = Metrics.NumInsts;

  // Don't allow an estimate of size zero.  This would allows unrolling of loops
  // with huge iteration counts, which is a compile time problem even if it's
  // not a problem for code quality.
  if (LoopSize == 0) LoopSize = 1;

  return LoopSize;
}

// Returns the value associated with the given metadata node name (for
// example, "llvm.loop.unroll.count").  If no such named metadata node
// exists, then nullptr is returned.
static const ConstantInt *GetUnrollMetadataValue(const Loop *L,
                                                 StringRef Name) {
  MDNode *LoopID = L->getLoopID();
  if (!LoopID) return nullptr;

  // First operand should refer to the loop id itself.
  assert(LoopID->getNumOperands() > 0 && "requires at least one operand");
  assert(LoopID->getOperand(0) == LoopID && "invalid loop id");

  for (unsigned i = 1, e = LoopID->getNumOperands(); i < e; ++i) {
    const MDNode *MD = dyn_cast<MDNode>(LoopID->getOperand(i));
    if (!MD) continue;

    const MDString *S = dyn_cast<MDString>(MD->getOperand(0));
    if (!S) continue;

    if (Name.equals(S->getString())) {
      assert(MD->getNumOperands() == 2 &&
             "Unroll hint metadata should have two operands.");
      return cast<ConstantInt>(MD->getOperand(1));
    }
  }
  return nullptr;
}

// Returns true if the loop has an unroll(enable) pragma.
static bool HasUnrollEnablePragma(const Loop *L) {
  const ConstantInt *EnableValue =
      GetUnrollMetadataValue(L, "llvm.loop.unroll.enable");
  return (EnableValue && EnableValue->getZExtValue());
}

// Returns true if the loop has an unroll(disable) pragma.
static bool HasUnrollDisablePragma(const Loop *L) {
  const ConstantInt *EnableValue =
      GetUnrollMetadataValue(L, "llvm.loop.unroll.enable");
  return (EnableValue && !EnableValue->getZExtValue());
}

// If loop has an unroll_count pragma return the (necessarily
// positive) value from the pragma.  Otherwise return 0.
static unsigned UnrollCountPragmaValue(const Loop *L) {
  const ConstantInt *CountValue =
      GetUnrollMetadataValue(L, "llvm.loop.unroll.count");
  if (CountValue) {
    unsigned Count = CountValue->getZExtValue();
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
  SmallVector<Value *, 4> Vals;
  // Reserve first location for self reference to the LoopID metadata node.
  Vals.push_back(nullptr);
  for (unsigned i = 1, ie = LoopID->getNumOperands(); i < ie; ++i) {
    bool IsUnrollMetadata = false;
    MDNode *MD = dyn_cast<MDNode>(LoopID->getOperand(i));
    if (MD) {
      const MDString *S = dyn_cast<MDString>(MD->getOperand(0));
      IsUnrollMetadata = S && S->getString().startswith("llvm.loop.unroll.");
    }
    if (!IsUnrollMetadata) Vals.push_back(LoopID->getOperand(i));
  }

  // Add unroll(disable) metadata to disable future unrolling.
  LLVMContext &Context = L->getHeader()->getContext();
  SmallVector<Value *, 2> DisableOperands;
  DisableOperands.push_back(MDString::get(Context, "llvm.loop.unroll.enable"));
  DisableOperands.push_back(ConstantInt::get(Type::getInt1Ty(Context), 0));
  MDNode *DisableNode = MDNode::get(Context, DisableOperands);
  Vals.push_back(DisableNode);

  MDNode *NewLoopID = MDNode::get(Context, Vals);
  // Set operand 0 to refer to the loop id itself.
  NewLoopID->replaceOperandWith(0, NewLoopID);
  L->setLoopID(NewLoopID);
  LoopID->replaceAllUsesWith(NewLoopID);
}

unsigned LoopUnroll::selectUnrollCount(
    const Loop *L, unsigned TripCount, bool HasEnablePragma,
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
    } else if (HasEnablePragma) {
      // unroll(enable) pragma without an unroll_count pragma
      // indicates to unroll loop fully.
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

  LoopInfo *LI = &getAnalysis<LoopInfo>();
  ScalarEvolution *SE = &getAnalysis<ScalarEvolution>();
  const TargetTransformInfo &TTI = getAnalysis<TargetTransformInfo>();

  BasicBlock *Header = L->getHeader();
  DEBUG(dbgs() << "Loop Unroll: F[" << Header->getParent()->getName()
        << "] Loop %" << Header->getName() << "\n");

  if (HasUnrollDisablePragma(L)) {
    return false;
  }
  bool HasEnablePragma = HasUnrollEnablePragma(L);
  unsigned PragmaCount = UnrollCountPragmaValue(L);
  bool HasPragma = HasEnablePragma || PragmaCount > 0;

  TargetTransformInfo::UnrollingPreferences UP;
  getUnrollingPreferences(L, TTI, UP);

  // Find trip count and trip multiple if count is not available
  unsigned TripCount = 0;
  unsigned TripMultiple = 1;
  // Find "latch trip count". UnrollLoop assumes that control cannot exit
  // via the loop latch on any iteration prior to TripCount. The loop may exit
  // early via an earlier branch.
  BasicBlock *LatchBlock = L->getLoopLatch();
  if (LatchBlock) {
    TripCount = SE->getSmallConstantTripCount(L, LatchBlock);
    TripMultiple = SE->getSmallConstantTripMultiple(L, LatchBlock);
  }

  // Select an initial unroll count.  This may be reduced later based
  // on size thresholds.
  bool CountSetExplicitly;
  unsigned Count = selectUnrollCount(L, TripCount, HasEnablePragma, PragmaCount,
                                     UP, CountSetExplicitly);

  unsigned NumInlineCandidates;
  bool notDuplicatable;
  unsigned LoopSize =
      ApproximateLoopSize(L, NumInlineCandidates, notDuplicatable, TTI);
  DEBUG(dbgs() << "  Loop Size = " << LoopSize << "\n");
  uint64_t UnrolledSize = (uint64_t)LoopSize * Count;
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
  selectThresholds(L, HasPragma, UP, Threshold, PartialThreshold);

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
  if (Unrolling == Partial) {
    bool AllowPartial = UserAllowPartial ? CurrentAllowPartial : UP.Partial;
    if (!AllowPartial && !CountSetExplicitly) {
      DEBUG(dbgs() << "  will not try to unroll partially because "
                   << "-unroll-allow-partial not given\n");
      return false;
    }
    if (PartialThreshold != NoThreshold && UnrolledSize > PartialThreshold) {
      // Reduce unroll count to be modulo of TripCount for partial unrolling.
      Count = PartialThreshold / LoopSize;
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
      UnrolledSize = LoopSize * Count;
    }
    if (Count > UP.MaxCount)
      Count = UP.MaxCount;
    DEBUG(dbgs() << "  partially unrolling with count: " << Count << "\n");
  }

  if (HasPragma) {
    // Mark loop as unrolled to prevent unrolling beyond that
    // requested by the pragma.
    SetLoopAlreadyUnrolled(L);

    // Emit optimization remarks if we are unable to unroll the loop
    // as directed by a pragma.
    DebugLoc LoopLoc = L->getStartLoc();
    Function *F = Header->getParent();
    LLVMContext &Ctx = F->getContext();
    if (HasEnablePragma && PragmaCount == 0) {
      if (TripCount && Count != TripCount) {
        emitOptimizationRemarkMissed(
            Ctx, DEBUG_TYPE, *F, LoopLoc,
            "Unable to fully unroll loop as directed by unroll(enable) pragma "
            "because unrolled size is too large.");
      } else if (!TripCount) {
        emitOptimizationRemarkMissed(
            Ctx, DEBUG_TYPE, *F, LoopLoc,
            "Unable to fully unroll loop as directed by unroll(enable) pragma "
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
  if (!UnrollLoop(L, Count, TripCount, AllowRuntime, TripMultiple, LI, this, &LPM))
    return false;

  return true;
}
