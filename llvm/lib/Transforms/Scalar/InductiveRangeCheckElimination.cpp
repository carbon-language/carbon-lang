//===-- InductiveRangeCheckElimination.cpp - ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// The InductiveRangeCheckElimination pass splits a loop's iteration space into
// three disjoint ranges.  It does that in a way such that the loop running in
// the middle loop provably does not need range checks. As an example, it will
// convert
//
//   len = < known positive >
//   for (i = 0; i < n; i++) {
//     if (0 <= i && i < len) {
//       do_something();
//     } else {
//       throw_out_of_bounds();
//     }
//   }
//
// to
//
//   len = < known positive >
//   limit = smin(n, len)
//   // no first segment
//   for (i = 0; i < limit; i++) {
//     if (0 <= i && i < len) { // this check is fully redundant
//       do_something();
//     } else {
//       throw_out_of_bounds();
//     }
//   }
//   for (i = limit; i < n; i++) {
//     if (0 <= i && i < len) {
//       do_something();
//     } else {
//       throw_out_of_bounds();
//     }
//   }
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Optional.h"

#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"

#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/IR/Verifier.h"

#include "llvm/Support/Debug.h"

#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/SimplifyIndVar.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"

#include "llvm/Pass.h"

#include <array>

using namespace llvm;

cl::opt<unsigned> LoopSizeCutoff("irce-loop-size-cutoff", cl::Hidden,
                                 cl::init(64));

cl::opt<bool> PrintChangedLoops("irce-print-changed-loops", cl::Hidden,
                                cl::init(false));

#define DEBUG_TYPE "irce"

namespace {

/// An inductive range check is conditional branch in a loop with
///
///  1. a very cold successor (i.e. the branch jumps to that successor very
///     rarely)
///
///  and
///
///  2. a condition that is provably true for some range of values taken by the
///     containing loop's induction variable.
///
/// Currently all inductive range checks are branches conditional on an
/// expression of the form
///
///   0 <= (Offset + Scale * I) < Length
///
/// where `I' is the canonical induction variable of a loop to which Offset and
/// Scale are loop invariant, and Length is >= 0.  Currently the 'false' branch
/// is considered cold, looking at profiling data to verify that is a TODO.

class InductiveRangeCheck {
  const SCEV *Offset;
  const SCEV *Scale;
  Value *Length;
  BranchInst *Branch;

  InductiveRangeCheck() :
    Offset(nullptr), Scale(nullptr), Length(nullptr), Branch(nullptr) { }

public:
  const SCEV *getOffset() const { return Offset; }
  const SCEV *getScale() const { return Scale; }
  Value *getLength() const { return Length; }

  void print(raw_ostream &OS) const {
    OS << "InductiveRangeCheck:\n";
    OS << "  Offset: ";
    Offset->print(OS);
    OS << "  Scale: ";
    Scale->print(OS);
    OS << "  Length: ";
    Length->print(OS);
    OS << "  Branch: ";
    getBranch()->print(OS);
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void dump() {
    print(dbgs());
  }
#endif

  BranchInst *getBranch() const { return Branch; }

  /// Represents an signed integer range [Range.getBegin(), Range.getEnd()).  If
  /// R.getEnd() sle R.getBegin(), then R denotes the empty range.

  class Range {
    Value *Begin;
    Value *End;

  public:
    Range(Value *Begin, Value *End) : Begin(Begin), End(End) {
      assert(Begin->getType() == End->getType() && "ill-typed range!");
    }

    Type *getType() const { return Begin->getType(); }
    Value *getBegin() const { return Begin; }
    Value *getEnd() const { return End; }
  };

  typedef SpecificBumpPtrAllocator<InductiveRangeCheck> AllocatorTy;

  /// This is the value the condition of the branch needs to evaluate to for the
  /// branch to take the hot successor (see (1) above).
  bool getPassingDirection() { return true; }

  /// Computes a range for the induction variable in which the range check is
  /// redundant and can be constant-folded away.
  Optional<Range> computeSafeIterationSpace(ScalarEvolution &SE,
                                            IRBuilder<> &B) const;

  /// Create an inductive range check out of BI if possible, else return
  /// nullptr.
  static InductiveRangeCheck *create(AllocatorTy &Alloc, BranchInst *BI,
                                     Loop *L, ScalarEvolution &SE,
                                     BranchProbabilityInfo &BPI);
};

class InductiveRangeCheckElimination : public LoopPass {
  InductiveRangeCheck::AllocatorTy Allocator;

public:
  static char ID;
  InductiveRangeCheckElimination() : LoopPass(ID) {
    initializeInductiveRangeCheckEliminationPass(
        *PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequiredID(LoopSimplifyID);
    AU.addRequiredID(LCSSAID);
    AU.addRequired<ScalarEvolution>();
    AU.addRequired<BranchProbabilityInfo>();
  }

  bool runOnLoop(Loop *L, LPPassManager &LPM) override;
};

char InductiveRangeCheckElimination::ID = 0;
}

INITIALIZE_PASS(InductiveRangeCheckElimination, "irce",
                "Inductive range check elimination", false, false)

static bool IsLowerBoundCheck(Value *Check, Value *&IndexV) {
  using namespace llvm::PatternMatch;

  ICmpInst::Predicate Pred = ICmpInst::BAD_ICMP_PREDICATE;
  Value *LHS = nullptr, *RHS = nullptr;

  if (!match(Check, m_ICmp(Pred, m_Value(LHS), m_Value(RHS))))
    return false;

  switch (Pred) {
  default:
    return false;

  case ICmpInst::ICMP_SLE:
    std::swap(LHS, RHS);
  // fallthrough
  case ICmpInst::ICMP_SGE:
    if (!match(RHS, m_ConstantInt<0>()))
      return false;
    IndexV = LHS;
    return true;

  case ICmpInst::ICMP_SLT:
    std::swap(LHS, RHS);
  // fallthrough
  case ICmpInst::ICMP_SGT:
    if (!match(RHS, m_ConstantInt<-1>()))
      return false;
    IndexV = LHS;
    return true;
  }
}

static bool IsUpperBoundCheck(Value *Check, Value *Index, Value *&UpperLimit) {
  using namespace llvm::PatternMatch;

  ICmpInst::Predicate Pred = ICmpInst::BAD_ICMP_PREDICATE;
  Value *LHS = nullptr, *RHS = nullptr;

  if (!match(Check, m_ICmp(Pred, m_Value(LHS), m_Value(RHS))))
    return false;

  switch (Pred) {
  default:
    return false;

  case ICmpInst::ICMP_SGT:
    std::swap(LHS, RHS);
  // fallthrough
  case ICmpInst::ICMP_SLT:
    if (LHS != Index)
      return false;
    UpperLimit = RHS;
    return true;

  case ICmpInst::ICMP_UGT:
    std::swap(LHS, RHS);
  // fallthrough
  case ICmpInst::ICMP_ULT:
    if (LHS != Index)
      return false;
    UpperLimit = RHS;
    return true;
  }
}

/// Split a condition into something semantically equivalent to (0 <= I <
/// Limit), both comparisons signed and Len loop invariant on L and positive.
/// On success, return true and set Index to I and UpperLimit to Limit.  Return
/// false on failure (we may still write to UpperLimit and Index on failure).
/// It does not try to interpret I as a loop index.
///
static bool SplitRangeCheckCondition(Loop *L, ScalarEvolution &SE,
                                     Value *Condition, const SCEV *&Index,
                                     Value *&UpperLimit) {

  // TODO: currently this catches some silly cases like comparing "%idx slt 1".
  // Our transformations are still correct, but less likely to be profitable in
  // those cases.  We have to come up with some heuristics that pick out the
  // range checks that are more profitable to clone a loop for.  This function
  // in general can be made more robust.

  using namespace llvm::PatternMatch;

  Value *A = nullptr;
  Value *B = nullptr;
  ICmpInst::Predicate Pred = ICmpInst::BAD_ICMP_PREDICATE;

  // In these early checks we assume that the matched UpperLimit is positive.
  // We'll verify that fact later, before returning true.

  if (match(Condition, m_And(m_Value(A), m_Value(B)))) {
    Value *IndexV = nullptr;
    Value *ExpectedUpperBoundCheck = nullptr;

    if (IsLowerBoundCheck(A, IndexV))
      ExpectedUpperBoundCheck = B;
    else if (IsLowerBoundCheck(B, IndexV))
      ExpectedUpperBoundCheck = A;
    else
      return false;

    if (!IsUpperBoundCheck(ExpectedUpperBoundCheck, IndexV, UpperLimit))
      return false;

    Index = SE.getSCEV(IndexV);

    if (isa<SCEVCouldNotCompute>(Index))
      return false;

  } else if (match(Condition, m_ICmp(Pred, m_Value(A), m_Value(B)))) {
    switch (Pred) {
    default:
      return false;

    case ICmpInst::ICMP_SGT:
      std::swap(A, B);
    // fall through
    case ICmpInst::ICMP_SLT:
      UpperLimit = B;
      Index = SE.getSCEV(A);
      if (isa<SCEVCouldNotCompute>(Index) || !SE.isKnownNonNegative(Index))
        return false;
      break;

    case ICmpInst::ICMP_UGT:
      std::swap(A, B);
    // fall through
    case ICmpInst::ICMP_ULT:
      UpperLimit = B;
      Index = SE.getSCEV(A);
      if (isa<SCEVCouldNotCompute>(Index))
        return false;
      break;
    }
  } else {
    return false;
  }

  const SCEV *UpperLimitSCEV = SE.getSCEV(UpperLimit);
  if (isa<SCEVCouldNotCompute>(UpperLimitSCEV) ||
      !SE.isKnownNonNegative(UpperLimitSCEV))
    return false;

  if (SE.getLoopDisposition(UpperLimitSCEV, L) !=
      ScalarEvolution::LoopInvariant) {
    DEBUG(dbgs() << " in function: " << L->getHeader()->getParent()->getName()
                 << " ";
          dbgs() << " UpperLimit is not loop invariant: "
                 << UpperLimit->getName() << "\n";);
    return false;
  }

  return true;
}


InductiveRangeCheck *
InductiveRangeCheck::create(InductiveRangeCheck::AllocatorTy &A, BranchInst *BI,
                            Loop *L, ScalarEvolution &SE,
                            BranchProbabilityInfo &BPI) {

  if (BI->isUnconditional() || BI->getParent() == L->getLoopLatch())
    return nullptr;

  BranchProbability LikelyTaken(15, 16);

  if (BPI.getEdgeProbability(BI->getParent(), (unsigned) 0) < LikelyTaken)
    return nullptr;

  Value *Length = nullptr;
  const SCEV *IndexSCEV = nullptr;

  if (!SplitRangeCheckCondition(L, SE, BI->getCondition(), IndexSCEV, Length))
    return nullptr;

  assert(IndexSCEV && Length && "contract with SplitRangeCheckCondition!");

  const SCEVAddRecExpr *IndexAddRec = dyn_cast<SCEVAddRecExpr>(IndexSCEV);
  bool IsAffineIndex =
      IndexAddRec && (IndexAddRec->getLoop() == L) && IndexAddRec->isAffine();

  if (!IsAffineIndex)
    return nullptr;

  InductiveRangeCheck *IRC = new (A.Allocate()) InductiveRangeCheck;
  IRC->Length = Length;
  IRC->Offset = IndexAddRec->getStart();
  IRC->Scale = IndexAddRec->getStepRecurrence(SE);
  IRC->Branch = BI;
  return IRC;
}

static Value *MaybeSimplify(Value *V) {
  if (Instruction *I = dyn_cast<Instruction>(V))
    if (Value *Simplified = SimplifyInstruction(I))
      return Simplified;
  return V;
}

static Value *ConstructSMinOf(Value *X, Value *Y, IRBuilder<> &B) {
  return MaybeSimplify(B.CreateSelect(B.CreateICmpSLT(X, Y), X, Y));
}

static Value *ConstructSMaxOf(Value *X, Value *Y, IRBuilder<> &B) {
  return MaybeSimplify(B.CreateSelect(B.CreateICmpSGT(X, Y), X, Y));
}

namespace {

/// This class is used to constrain loops to run within a given iteration space.
/// The algorithm this class implements is given a Loop and a range [Begin,
/// End).  The algorithm then tries to break out a "main loop" out of the loop
/// it is given in a way that the "main loop" runs with the induction variable
/// in a subset of [Begin, End).  The algorithm emits appropriate pre and post
/// loops to run any remaining iterations.  The pre loop runs any iterations in
/// which the induction variable is < Begin, and the post loop runs any
/// iterations in which the induction variable is >= End.
///
class LoopConstrainer {

  // Keeps track of the structure of a loop.  This is similar to llvm::Loop,
  // except that it is more lightweight and can track the state of a loop
  // through changing and potentially invalid IR.  This structure also
  // formalizes the kinds of loops we can deal with -- ones that have a single
  // latch that is also an exiting block *and* have a canonical induction
  // variable.
  struct LoopStructure {
    const char *Tag;

    BasicBlock *Header;
    BasicBlock *Latch;

    // `Latch's terminator instruction is `LatchBr', and it's `LatchBrExitIdx'th
    // successor is `LatchExit', the exit block of the loop.
    BranchInst *LatchBr;
    BasicBlock *LatchExit;
    unsigned LatchBrExitIdx;

    // The canonical induction variable.  It's value is `CIVStart` on the 0th
    // itertion and `CIVNext` for all iterations after that.
    PHINode *CIV;
    Value *CIVStart;
    Value *CIVNext;

    LoopStructure() : Tag(""), Header(nullptr), Latch(nullptr),
                      LatchBr(nullptr), LatchExit(nullptr),
                      LatchBrExitIdx(-1), CIV(nullptr),
                      CIVStart(nullptr), CIVNext(nullptr) { }

    template <typename M> LoopStructure map(M Map) const {
      LoopStructure Result;
      Result.Tag = Tag;
      Result.Header = cast<BasicBlock>(Map(Header));
      Result.Latch = cast<BasicBlock>(Map(Latch));
      Result.LatchBr = cast<BranchInst>(Map(LatchBr));
      Result.LatchExit = cast<BasicBlock>(Map(LatchExit));
      Result.LatchBrExitIdx = LatchBrExitIdx;
      Result.CIV = cast<PHINode>(Map(CIV));
      Result.CIVNext = Map(CIVNext);
      Result.CIVStart = Map(CIVStart);
      return Result;
    }
  };

  // The representation of a clone of the original loop we started out with.
  struct ClonedLoop {
    // The cloned blocks
    std::vector<BasicBlock *> Blocks;

    // `Map` maps values in the clonee into values in the cloned version
    ValueToValueMapTy Map;

    // An instance of `LoopStructure` for the cloned loop
    LoopStructure Structure;
  };

  // Result of rewriting the range of a loop.  See changeIterationSpaceEnd for
  // more details on what these fields mean.
  struct RewrittenRangeInfo {
    BasicBlock *PseudoExit;
    BasicBlock *ExitSelector;
    std::vector<PHINode *> PHIValuesAtPseudoExit;

    RewrittenRangeInfo() : PseudoExit(nullptr), ExitSelector(nullptr) { }
  };

  // Calculated subranges we restrict the iteration space of the main loop to.
  // See the implementation of `calculateSubRanges' for more details on how
  // these fields are computed.  `ExitPreLoopAt' is `None' if we don't need a
  // pre loop.  `ExitMainLoopAt' is `None' if we don't need a post loop.
  struct SubRanges {
    Optional<Value *> ExitPreLoopAt;
    Optional<Value *> ExitMainLoopAt;
  };

  // A utility function that does a `replaceUsesOfWith' on the incoming block
  // set of a `PHINode' -- replaces instances of `Block' in the `PHINode's
  // incoming block list with `ReplaceBy'.
  static void replacePHIBlock(PHINode *PN, BasicBlock *Block,
                              BasicBlock *ReplaceBy);

  // Try to "parse" `OriginalLoop' and populate the various out parameters.
  // Returns true on success, false on failure.
  //
  bool recognizeLoop(LoopStructure &LoopStructureOut,
                     const SCEV *&LatchCountOut, BasicBlock *&PreHeaderOut,
                     const char *&FailureReasonOut) const;

  // Compute a safe set of limits for the main loop to run in -- effectively the
  // intersection of `Range' and the iteration space of the original loop.
  // Return the header count (1 + the latch taken count) in `HeaderCount'.
  // Return None if unable to compute the set of subranges.
  //
  Optional<SubRanges> calculateSubRanges(Value *&HeaderCount) const;

  // Clone `OriginalLoop' and return the result in CLResult.  The IR after
  // running `cloneLoop' is well formed except for the PHI nodes in CLResult --
  // the PHI nodes say that there is an incoming edge from `OriginalPreheader`
  // but there is no such edge.
  //
  void cloneLoop(ClonedLoop &CLResult, const char *Tag) const;

  // Rewrite the iteration space of the loop denoted by (LS, Preheader). The
  // iteration space of the rewritten loop ends at ExitLoopAt.  The start of the
  // iteration space is not changed.  `ExitLoopAt' is assumed to be slt
  // `OriginalHeaderCount'.
  //
  // If there are iterations left to execute, control is made to jump to
  // `ContinuationBlock', otherwise they take the normal loop exit.  The
  // returned `RewrittenRangeInfo' object is populated as follows:
  //
  //  .PseudoExit is a basic block that unconditionally branches to
  //      `ContinuationBlock'.
  //
  //  .ExitSelector is a basic block that decides, on exit from the loop,
  //      whether to branch to the "true" exit or to `PseudoExit'.
  //
  //  .PHIValuesAtPseudoExit are PHINodes in `PseudoExit' that compute the value
  //      for each PHINode in the loop header on taking the pseudo exit.
  //
  // After changeIterationSpaceEnd, `Preheader' is no longer a legitimate
  // preheader because it is made to branch to the loop header only
  // conditionally.
  //
  RewrittenRangeInfo
  changeIterationSpaceEnd(const LoopStructure &LS, BasicBlock *Preheader,
                          Value *ExitLoopAt,
                          BasicBlock *ContinuationBlock) const;

  // The loop denoted by `LS' has `OldPreheader' as its preheader.  This
  // function creates a new preheader for `LS' and returns it.
  //
  BasicBlock *createPreheader(const LoopConstrainer::LoopStructure &LS,
                              BasicBlock *OldPreheader, const char *Tag) const;

  // `ContinuationBlockAndPreheader' was the continuation block for some call to
  // `changeIterationSpaceEnd' and is the preheader to the loop denoted by `LS'.
  // This function rewrites the PHI nodes in `LS.Header' to start with the
  // correct value.
  void rewriteIncomingValuesForPHIs(
      LoopConstrainer::LoopStructure &LS,
      BasicBlock *ContinuationBlockAndPreheader,
      const LoopConstrainer::RewrittenRangeInfo &RRI) const;

  // Even though we do not preserve any passes at this time, we at least need to
  // keep the parent loop structure consistent.  The `LPPassManager' seems to
  // verify this after running a loop pass.  This function adds the list of
  // blocks denoted by the iterator range [BlocksBegin, BlocksEnd) to this loops
  // parent loop if required.
  template<typename IteratorTy>
  void addToParentLoopIfNeeded(IteratorTy BlocksBegin, IteratorTy BlocksEnd);

  // Some global state.
  Function &F;
  LLVMContext &Ctx;
  ScalarEvolution &SE;

  // Information about the original loop we started out with.
  Loop &OriginalLoop;
  LoopInfo &OriginalLoopInfo;
  const SCEV *LatchTakenCount;
  BasicBlock *OriginalPreheader;
  Value *OriginalHeaderCount;

  // The preheader of the main loop.  This may or may not be different from
  // `OriginalPreheader'.
  BasicBlock *MainLoopPreheader;

  // The range we need to run the main loop in.
  InductiveRangeCheck::Range Range;

  // The structure of the main loop (see comment at the beginning of this class
  // for a definition)
  LoopStructure MainLoopStructure;

public:
  LoopConstrainer(Loop &L, LoopInfo &LI, ScalarEvolution &SE,
                  InductiveRangeCheck::Range R)
    : F(*L.getHeader()->getParent()), Ctx(L.getHeader()->getContext()), SE(SE),
      OriginalLoop(L), OriginalLoopInfo(LI), LatchTakenCount(nullptr),
      OriginalPreheader(nullptr), OriginalHeaderCount(nullptr),
      MainLoopPreheader(nullptr), Range(R) { }

  // Entry point for the algorithm.  Returns true on success.
  bool run();
};

}

void LoopConstrainer::replacePHIBlock(PHINode *PN, BasicBlock *Block,
                                      BasicBlock *ReplaceBy) {
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
    if (PN->getIncomingBlock(i) == Block)
      PN->setIncomingBlock(i, ReplaceBy);
}

bool LoopConstrainer::recognizeLoop(LoopStructure &LoopStructureOut,
                                    const SCEV *&LatchCountOut,
                                    BasicBlock *&PreheaderOut,
                                    const char *&FailureReason) const {
  using namespace llvm::PatternMatch;

  assert(OriginalLoop.isLoopSimplifyForm() &&
         "should follow from addRequired<>");

  BasicBlock *Latch = OriginalLoop.getLoopLatch();
  if (!OriginalLoop.isLoopExiting(Latch)) {
    FailureReason = "no loop latch";
    return false;
  }

  PHINode *CIV = OriginalLoop.getCanonicalInductionVariable();
  if (!CIV) {
    FailureReason = "no CIV";
    return false;
  }

  BasicBlock *Header = OriginalLoop.getHeader();
  BasicBlock *Preheader = OriginalLoop.getLoopPreheader();
  if (!Preheader) {
    FailureReason = "no preheader";
    return false;
  }

  Value *CIVNext = CIV->getIncomingValueForBlock(Latch);
  Value *CIVStart = CIV->getIncomingValueForBlock(Preheader);

  const SCEV *LatchCount = SE.getExitCount(&OriginalLoop, Latch);
  if (isa<SCEVCouldNotCompute>(LatchCount)) {
    FailureReason = "could not compute latch count";
    return false;
  }

  // While SCEV does most of the analysis for us, we still have to
  // modify the latch; and currently we can only deal with certain
  // kinds of latches.  This can be made more sophisticated as needed.

  BranchInst *LatchBr = dyn_cast<BranchInst>(&*Latch->rbegin());

  if (!LatchBr || LatchBr->isUnconditional()) {
    FailureReason = "latch terminator not conditional branch";
    return false;
  }

  // Currently we only support a latch condition of the form:
  //
  //  %condition = icmp slt %civNext, %limit
  //  br i1 %condition, label %header, label %exit

  if (LatchBr->getSuccessor(0) != Header) {
    FailureReason = "unknown latch form (header not first successor)";
    return false;
  }

  Value *CIVComparedTo = nullptr;
  ICmpInst::Predicate Pred = ICmpInst::BAD_ICMP_PREDICATE;
  if (!(match(LatchBr->getCondition(),
              m_ICmp(Pred, m_Specific(CIVNext), m_Value(CIVComparedTo))) &&
        Pred == ICmpInst::ICMP_SLT)) {
    FailureReason = "unknown latch form (not slt)";
    return false;
  }

  // IndVarSimplify will sometimes leave behind (in SCEV's cache) backedge-taken
  // counts that are narrower than the canonical induction variable.  These
  // values are still accurate, and we could probably use them after sign/zero
  // extension; but for now we just bail out of the transformation to keep
  // things simple.
  const SCEV *CIVComparedToSCEV = SE.getSCEV(CIVComparedTo);
  if (isa<SCEVCouldNotCompute>(CIVComparedToSCEV) ||
      CIVComparedToSCEV->getType() != LatchCount->getType()) {
    FailureReason = "could not relate CIV to latch expression";
    return false;
  }

  const SCEV *ShouldBeOne = SE.getMinusSCEV(CIVComparedToSCEV, LatchCount);
  const SCEVConstant *SCEVOne = dyn_cast<SCEVConstant>(ShouldBeOne);
  if (!SCEVOne || SCEVOne->getValue()->getValue() != 1) {
    FailureReason = "unexpected header count in latch";
    return false;
  }

  unsigned LatchBrExitIdx = 1;
  BasicBlock *LatchExit = LatchBr->getSuccessor(LatchBrExitIdx);

  assert(SE.getLoopDisposition(LatchCount, &OriginalLoop) ==
             ScalarEvolution::LoopInvariant &&
         "loop variant exit count doesn't make sense!");

  assert(!OriginalLoop.contains(LatchExit) && "expected an exit block!");

  LoopStructureOut.Tag = "main";
  LoopStructureOut.Header = Header;
  LoopStructureOut.Latch = Latch;
  LoopStructureOut.LatchBr = LatchBr;
  LoopStructureOut.LatchExit = LatchExit;
  LoopStructureOut.LatchBrExitIdx = LatchBrExitIdx;
  LoopStructureOut.CIV = CIV;
  LoopStructureOut.CIVNext = CIVNext;
  LoopStructureOut.CIVStart = CIVStart;

  LatchCountOut = LatchCount;
  PreheaderOut = Preheader;
  FailureReason = nullptr;

  return true;
}

Optional<LoopConstrainer::SubRanges>
LoopConstrainer::calculateSubRanges(Value *&HeaderCountOut) const {
  IntegerType *Ty = cast<IntegerType>(LatchTakenCount->getType());

  if (Range.getType() != Ty)
    return None;

  SCEVExpander Expander(SE, "irce");
  Instruction *InsertPt = OriginalPreheader->getTerminator();

  Value *LatchCountV =
      MaybeSimplify(Expander.expandCodeFor(LatchTakenCount, Ty, InsertPt));

  IRBuilder<> B(InsertPt);

  LoopConstrainer::SubRanges Result;

  // I think we can be more aggressive here and make this nuw / nsw if the
  // addition that feeds into the icmp for the latch's terminating branch is nuw
  // / nsw.  In any case, a wrapping 2's complement addition is safe.
  ConstantInt *One = ConstantInt::get(Ty, 1);
  HeaderCountOut = MaybeSimplify(B.CreateAdd(LatchCountV, One, "header.count"));

  const SCEV *RangeBegin = SE.getSCEV(Range.getBegin());
  const SCEV *RangeEnd = SE.getSCEV(Range.getEnd());
  const SCEV *HeaderCountSCEV = SE.getSCEV(HeaderCountOut);
  const SCEV *Zero = SE.getConstant(Ty, 0);

  // In some cases we can prove that we don't need a pre or post loop

  bool ProvablyNoPreloop =
      SE.isKnownPredicate(ICmpInst::ICMP_SLE, RangeBegin, Zero);
  if (!ProvablyNoPreloop)
    Result.ExitPreLoopAt = ConstructSMinOf(HeaderCountOut, Range.getBegin(), B);

  bool ProvablyNoPostLoop =
      SE.isKnownPredicate(ICmpInst::ICMP_SLE, HeaderCountSCEV, RangeEnd);
  if (!ProvablyNoPostLoop)
    Result.ExitMainLoopAt = ConstructSMinOf(HeaderCountOut, Range.getEnd(), B);

  return Result;
}

void LoopConstrainer::cloneLoop(LoopConstrainer::ClonedLoop &Result,
                                const char *Tag) const {
  for (BasicBlock *BB : OriginalLoop.getBlocks()) {
    BasicBlock *Clone = CloneBasicBlock(BB, Result.Map, Twine(".") + Tag, &F);
    Result.Blocks.push_back(Clone);
    Result.Map[BB] = Clone;
  }

  auto GetClonedValue = [&Result](Value *V) {
    assert(V && "null values not in domain!");
    auto It = Result.Map.find(V);
    if (It == Result.Map.end())
      return V;
    return static_cast<Value *>(It->second);
  };

  Result.Structure = MainLoopStructure.map(GetClonedValue);
  Result.Structure.Tag = Tag;

  for (unsigned i = 0, e = Result.Blocks.size(); i != e; ++i) {
    BasicBlock *ClonedBB = Result.Blocks[i];
    BasicBlock *OriginalBB = OriginalLoop.getBlocks()[i];

    assert(Result.Map[OriginalBB] == ClonedBB && "invariant!");

    for (Instruction &I : *ClonedBB)
      RemapInstruction(&I, Result.Map,
                       RF_NoModuleLevelChanges | RF_IgnoreMissingEntries);

    // Exit blocks will now have one more predecessor and their PHI nodes need
    // to be edited to reflect that.  No phi nodes need to be introduced because
    // the loop is in LCSSA.

    for (auto SBBI = succ_begin(OriginalBB), SBBE = succ_end(OriginalBB);
         SBBI != SBBE; ++SBBI) {

      if (OriginalLoop.contains(*SBBI))
        continue; // not an exit block

      for (Instruction &I : **SBBI) {
        if (!isa<PHINode>(&I))
          break;

        PHINode *PN = cast<PHINode>(&I);
        Value *OldIncoming = PN->getIncomingValueForBlock(OriginalBB);
        PN->addIncoming(GetClonedValue(OldIncoming), ClonedBB);
      }
    }
  }
}

LoopConstrainer::RewrittenRangeInfo LoopConstrainer::changeIterationSpaceEnd(
    const LoopStructure &LS, BasicBlock *Preheader, Value *ExitLoopAt,
    BasicBlock *ContinuationBlock) const {

  // We start with a loop with a single latch:
  //
  //    +--------------------+
  //    |                    |
  //    |     preheader      |
  //    |                    |
  //    +--------+-----------+
  //             |      ----------------\
  //             |     /                |
  //    +--------v----v------+          |
  //    |                    |          |
  //    |      header        |          |
  //    |                    |          |
  //    +--------------------+          |
  //                                    |
  //            .....                   |
  //                                    |
  //    +--------------------+          |
  //    |                    |          |
  //    |       latch        >----------/
  //    |                    |
  //    +-------v------------+
  //            |
  //            |
  //            |   +--------------------+
  //            |   |                    |
  //            +--->   original exit    |
  //                |                    |
  //                +--------------------+
  //
  // We change the control flow to look like
  //
  //
  //    +--------------------+
  //    |                    |
  //    |     preheader      >-------------------------+
  //    |                    |                         |
  //    +--------v-----------+                         |
  //             |    /-------------+                  |
  //             |   /              |                  |
  //    +--------v--v--------+      |                  |
  //    |                    |      |                  |
  //    |      header        |      |   +--------+     |
  //    |                    |      |   |        |     |
  //    +--------------------+      |   |  +-----v-----v-----------+
  //                                |   |  |                       |
  //                                |   |  |     .pseudo.exit      |
  //                                |   |  |                       |
  //                                |   |  +-----------v-----------+
  //                                |   |              |
  //            .....               |   |              |
  //                                |   |     +--------v-------------+
  //    +--------------------+      |   |     |                      |
  //    |                    |      |   |     |   ContinuationBlock  |
  //    |       latch        >------+   |     |                      |
  //    |                    |          |     +----------------------+
  //    +---------v----------+          |
  //              |                     |
  //              |                     |
  //              |     +---------------^-----+
  //              |     |                     |
  //              +----->    .exit.selector   |
  //                    |                     |
  //                    +----------v----------+
  //                               |
  //     +--------------------+    |
  //     |                    |    |
  //     |   original exit    <----+
  //     |                    |
  //     +--------------------+
  //

  RewrittenRangeInfo RRI;

  auto BBInsertLocation = std::next(Function::iterator(LS.Latch));
  RRI.ExitSelector = BasicBlock::Create(Ctx, Twine(LS.Tag) + ".exit.selector",
                                        &F, BBInsertLocation);
  RRI.PseudoExit = BasicBlock::Create(Ctx, Twine(LS.Tag) + ".pseudo.exit", &F,
                                      BBInsertLocation);

  BranchInst *PreheaderJump = cast<BranchInst>(&*Preheader->rbegin());

  IRBuilder<> B(PreheaderJump);

  // EnterLoopCond - is it okay to start executing this `LS'?
  Value *EnterLoopCond = B.CreateICmpSLT(LS.CIVStart, ExitLoopAt);
  B.CreateCondBr(EnterLoopCond, LS.Header, RRI.PseudoExit);
  PreheaderJump->eraseFromParent();

  assert(LS.LatchBrExitIdx == 1 && "generalize this as needed!");

  B.SetInsertPoint(LS.LatchBr);

  // ContinueCond - is it okay to execute the next iteration in `LS'?
  Value *ContinueCond = B.CreateICmpSLT(LS.CIVNext, ExitLoopAt);

  LS.LatchBr->setCondition(ContinueCond);
  assert(LS.LatchBr->getSuccessor(LS.LatchBrExitIdx) == LS.LatchExit &&
         "invariant!");
  LS.LatchBr->setSuccessor(LS.LatchBrExitIdx, RRI.ExitSelector);

  B.SetInsertPoint(RRI.ExitSelector);

  // IterationsLeft - are there any more iterations left, given the original
  // upper bound on the induction variable?  If not, we branch to the "real"
  // exit.
  Value *IterationsLeft = B.CreateICmpSLT(LS.CIVNext, OriginalHeaderCount);
  B.CreateCondBr(IterationsLeft, RRI.PseudoExit, LS.LatchExit);

  BranchInst *BranchToContinuation =
      BranchInst::Create(ContinuationBlock, RRI.PseudoExit);

  // We emit PHI nodes into `RRI.PseudoExit' that compute the "latest" value of
  // each of the PHI nodes in the loop header.  This feeds into the initial
  // value of the same PHI nodes if/when we continue execution.
  for (Instruction &I : *LS.Header) {
    if (!isa<PHINode>(&I))
      break;

    PHINode *PN = cast<PHINode>(&I);

    PHINode *NewPHI = PHINode::Create(PN->getType(), 2, PN->getName() + ".copy",
                                      BranchToContinuation);

    NewPHI->addIncoming(PN->getIncomingValueForBlock(Preheader), Preheader);
    NewPHI->addIncoming(PN->getIncomingValueForBlock(LS.Latch),
                        RRI.ExitSelector);
    RRI.PHIValuesAtPseudoExit.push_back(NewPHI);
  }

  // The latch exit now has a branch from `RRI.ExitSelector' instead of
  // `LS.Latch'.  The PHI nodes need to be updated to reflect that.
  for (Instruction &I : *LS.LatchExit) {
    if (PHINode *PN = dyn_cast<PHINode>(&I))
      replacePHIBlock(PN, LS.Latch, RRI.ExitSelector);
    else
      break;
  }

  return RRI;
}

void LoopConstrainer::rewriteIncomingValuesForPHIs(
    LoopConstrainer::LoopStructure &LS, BasicBlock *ContinuationBlock,
    const LoopConstrainer::RewrittenRangeInfo &RRI) const {

  unsigned PHIIndex = 0;
  for (Instruction &I : *LS.Header) {
    if (!isa<PHINode>(&I))
      break;

    PHINode *PN = cast<PHINode>(&I);

    for (unsigned i = 0, e = PN->getNumIncomingValues(); i < e; ++i)
      if (PN->getIncomingBlock(i) == ContinuationBlock)
        PN->setIncomingValue(i, RRI.PHIValuesAtPseudoExit[PHIIndex++]);
  }

  LS.CIVStart = LS.CIV->getIncomingValueForBlock(ContinuationBlock);
}

BasicBlock *
LoopConstrainer::createPreheader(const LoopConstrainer::LoopStructure &LS,
                                 BasicBlock *OldPreheader,
                                 const char *Tag) const {

  BasicBlock *Preheader = BasicBlock::Create(Ctx, Tag, &F, LS.Header);
  BranchInst::Create(LS.Header, Preheader);

  for (Instruction &I : *LS.Header) {
    if (!isa<PHINode>(&I))
      break;

    PHINode *PN = cast<PHINode>(&I);
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i < e; ++i)
      replacePHIBlock(PN, OldPreheader, Preheader);
  }

  return Preheader;
}

template<typename IteratorTy>
void LoopConstrainer::addToParentLoopIfNeeded(IteratorTy Begin,
                                              IteratorTy End) {
  Loop *ParentLoop = OriginalLoop.getParentLoop();
  if (!ParentLoop)
    return;

  for (; Begin != End; Begin++)
    ParentLoop->addBasicBlockToLoop(*Begin, OriginalLoopInfo);
}

bool LoopConstrainer::run() {
  BasicBlock *Preheader = nullptr;
  const char *CouldNotProceedBecause = nullptr;
  if (!recognizeLoop(MainLoopStructure, LatchTakenCount, Preheader,
                     CouldNotProceedBecause)) {
    DEBUG(dbgs() << "irce: could not recognize loop, " << CouldNotProceedBecause
                 << "\n";);
    return false;
  }

  OriginalPreheader = Preheader;
  MainLoopPreheader = Preheader;

  Optional<SubRanges> MaybeSR = calculateSubRanges(OriginalHeaderCount);
  if (!MaybeSR.hasValue()) {
    DEBUG(dbgs() << "irce: could not compute subranges\n");
    return false;
  }
  SubRanges SR = MaybeSR.getValue();

  // It would have been better to make `PreLoop' and `PostLoop'
  // `Optional<ClonedLoop>'s, but `ValueToValueMapTy' does not have a copy
  // constructor.
  ClonedLoop PreLoop, PostLoop;
  bool NeedsPreLoop = SR.ExitPreLoopAt.hasValue();
  bool NeedsPostLoop = SR.ExitMainLoopAt.hasValue();

  // We clone these ahead of time so that we don't have to deal with changing
  // and temporarily invalid IR as we transform the loops.
  if (NeedsPreLoop)
    cloneLoop(PreLoop, "preloop");
  if (NeedsPostLoop)
    cloneLoop(PostLoop, "postloop");

  RewrittenRangeInfo PreLoopRRI;

  if (NeedsPreLoop) {
    Preheader->getTerminator()->replaceUsesOfWith(MainLoopStructure.Header,
                                                  PreLoop.Structure.Header);

    MainLoopPreheader =
        createPreheader(MainLoopStructure, Preheader, "mainloop");
    PreLoopRRI =
        changeIterationSpaceEnd(PreLoop.Structure, Preheader,
                                SR.ExitPreLoopAt.getValue(), MainLoopPreheader);
    rewriteIncomingValuesForPHIs(MainLoopStructure, MainLoopPreheader,
                                 PreLoopRRI);
  }

  BasicBlock *PostLoopPreheader = nullptr;
  RewrittenRangeInfo PostLoopRRI;

  if (NeedsPostLoop) {
    PostLoopPreheader =
        createPreheader(PostLoop.Structure, Preheader, "postloop");
    PostLoopRRI = changeIterationSpaceEnd(MainLoopStructure, MainLoopPreheader,
                                          SR.ExitMainLoopAt.getValue(),
                                          PostLoopPreheader);
    rewriteIncomingValuesForPHIs(PostLoop.Structure, PostLoopPreheader,
                                 PostLoopRRI);
  }

  SmallVector<BasicBlock *, 6> NewBlocks;
  NewBlocks.push_back(PostLoopPreheader);
  NewBlocks.push_back(PreLoopRRI.PseudoExit);
  NewBlocks.push_back(PreLoopRRI.ExitSelector);
  NewBlocks.push_back(PostLoopRRI.PseudoExit);
  NewBlocks.push_back(PostLoopRRI.ExitSelector);
  if (MainLoopPreheader != Preheader)
    NewBlocks.push_back(MainLoopPreheader);

  // Some of the above may be nullptr, filter them out before passing to
  // addToParentLoopIfNeeded.
  auto NewBlocksEnd = std::remove(NewBlocks.begin(), NewBlocks.end(), nullptr);

  typedef SmallVector<BasicBlock *, 6>::iterator SmallVectItTy;
  typedef std::vector<BasicBlock *>::iterator StdVectItTy;

  addToParentLoopIfNeeded<SmallVectItTy>(NewBlocks.begin(), NewBlocksEnd);
  addToParentLoopIfNeeded<StdVectItTy>(PreLoop.Blocks.begin(),
                                       PreLoop.Blocks.end());
  addToParentLoopIfNeeded<StdVectItTy>(PostLoop.Blocks.begin(),
                                       PostLoop.Blocks.end());

  return true;
}

/// Computes and returns a range of values for the induction variable in which
/// the range check can be safely elided.  If it cannot compute such a range,
/// returns None.
Optional<InductiveRangeCheck::Range>
InductiveRangeCheck::computeSafeIterationSpace(ScalarEvolution &SE,
                                               IRBuilder<> &B) const {

  // Currently we support inequalities of the form:
  //
  //   0 <= Offset + 1 * CIV < L given L >= 0
  //
  // The inequality is satisfied by -Offset <= CIV < (L - Offset) [^1].  All
  // additions and subtractions are twos-complement wrapping and comparisons are
  // signed.
  //
  // Proof:
  //
  //   If there exists CIV such that -Offset <= CIV < (L - Offset) then it
  //   follows that -Offset <= (-Offset + L) [== Eq. 1].  Since L >= 0, if
  //   (-Offset + L) sign-overflows then (-Offset + L) < (-Offset).  Hence by
  //   [Eq. 1], (-Offset + L) could not have overflown.
  //
  //   This means CIV = t + (-Offset) for t in [0, L).  Hence (CIV + Offset) =
  //   t.  Hence 0 <= (CIV + Offset) < L

  // [^1]: Note that the solution does _not_ apply if L < 0; consider values
  // Offset = 127, CIV = 126 and L = -2 in an i8 world.

  const SCEVConstant *ScaleC = dyn_cast<SCEVConstant>(getScale());
  if (!(ScaleC && ScaleC->getValue()->getValue() == 1)) {
    DEBUG(dbgs() << "irce: could not compute safe iteration space for:\n";
          print(dbgs()));
    return None;
  }

  Value *OffsetV = SCEVExpander(SE, "safe.itr.space").expandCodeFor(
      getOffset(), getOffset()->getType(), B.GetInsertPoint());
  OffsetV = MaybeSimplify(OffsetV);

  Value *Begin = MaybeSimplify(B.CreateNeg(OffsetV));
  Value *End = MaybeSimplify(B.CreateSub(getLength(), OffsetV));

  return InductiveRangeCheck::Range(Begin, End);
}

static Optional<InductiveRangeCheck::Range>
IntersectRange(const Optional<InductiveRangeCheck::Range> &R1,
               const InductiveRangeCheck::Range &R2, IRBuilder<> &B) {
  if (!R1.hasValue())
    return R2;
  auto &R1Value = R1.getValue();

  // TODO: we could widen the smaller range and have this work; but for now we
  // bail out to keep things simple.
  if (R1Value.getType() != R2.getType())
    return None;

  Value *NewMin = ConstructSMaxOf(R1Value.getBegin(), R2.getBegin(), B);
  Value *NewMax = ConstructSMinOf(R1Value.getEnd(), R2.getEnd(), B);
  return InductiveRangeCheck::Range(NewMin, NewMax);
}

bool InductiveRangeCheckElimination::runOnLoop(Loop *L, LPPassManager &LPM) {
  if (L->getBlocks().size() >= LoopSizeCutoff) {
    DEBUG(dbgs() << "irce: giving up constraining loop, too large\n";);
    return false;
  }

  BasicBlock *Preheader = L->getLoopPreheader();
  if (!Preheader) {
    DEBUG(dbgs() << "irce: loop has no preheader, leaving\n");
    return false;
  }

  LLVMContext &Context = Preheader->getContext();
  InductiveRangeCheck::AllocatorTy IRCAlloc;
  SmallVector<InductiveRangeCheck *, 16> RangeChecks;
  ScalarEvolution &SE = getAnalysis<ScalarEvolution>();
  BranchProbabilityInfo &BPI = getAnalysis<BranchProbabilityInfo>();

  for (auto BBI : L->getBlocks())
    if (BranchInst *TBI = dyn_cast<BranchInst>(BBI->getTerminator()))
      if (InductiveRangeCheck *IRC =
          InductiveRangeCheck::create(IRCAlloc, TBI, L, SE, BPI))
        RangeChecks.push_back(IRC);

  if (RangeChecks.empty())
    return false;

  DEBUG(dbgs() << "irce: looking at loop "; L->print(dbgs());
        dbgs() << "irce: loop has " << RangeChecks.size()
               << " inductive range checks: \n";
        for (InductiveRangeCheck *IRC : RangeChecks)
          IRC->print(dbgs());
    );

  Optional<InductiveRangeCheck::Range> SafeIterRange;
  Instruction *ExprInsertPt = Preheader->getTerminator();

  SmallVector<InductiveRangeCheck *, 4> RangeChecksToEliminate;

  IRBuilder<> B(ExprInsertPt);
  for (InductiveRangeCheck *IRC : RangeChecks) {
    auto Result = IRC->computeSafeIterationSpace(SE, B);
    if (Result.hasValue()) {
      auto MaybeSafeIterRange =
        IntersectRange(SafeIterRange, Result.getValue(), B);
      if (MaybeSafeIterRange.hasValue()) {
        RangeChecksToEliminate.push_back(IRC);
        SafeIterRange = MaybeSafeIterRange.getValue();
      }
    }
  }

  if (!SafeIterRange.hasValue())
    return false;

  LoopConstrainer LC(*L, getAnalysis<LoopInfoWrapperPass>().getLoopInfo(), SE,
                     SafeIterRange.getValue());
  bool Changed = LC.run();

  if (Changed) {
    auto PrintConstrainedLoopInfo = [L]() {
      dbgs() << "irce: in function ";
      dbgs() << L->getHeader()->getParent()->getName() << ": ";
      dbgs() << "constrained ";
      L->print(dbgs());
    };

    DEBUG(PrintConstrainedLoopInfo());

    if (PrintChangedLoops)
      PrintConstrainedLoopInfo();

    // Optimize away the now-redundant range checks.

    for (InductiveRangeCheck *IRC : RangeChecksToEliminate) {
      ConstantInt *FoldedRangeCheck = IRC->getPassingDirection()
                                          ? ConstantInt::getTrue(Context)
                                          : ConstantInt::getFalse(Context);
      IRC->getBranch()->setCondition(FoldedRangeCheck);
    }
  }

  return Changed;
}

Pass *llvm::createInductiveRangeCheckEliminationPass() {
  return new InductiveRangeCheckElimination;
}
