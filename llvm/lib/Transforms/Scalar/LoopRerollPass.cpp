//===-- LoopReroll.cpp - Loop rerolling pass ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass implements a simple loop reroller.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

using namespace llvm;

#define DEBUG_TYPE "loop-reroll"

STATISTIC(NumRerolledLoops, "Number of rerolled loops");

static cl::opt<unsigned>
MaxInc("max-reroll-increment", cl::init(2048), cl::Hidden,
  cl::desc("The maximum increment for loop rerolling"));

// This loop re-rolling transformation aims to transform loops like this:
//
// int foo(int a);
// void bar(int *x) {
//   for (int i = 0; i < 500; i += 3) {
//     foo(i);
//     foo(i+1);
//     foo(i+2);
//   }
// }
//
// into a loop like this:
//
// void bar(int *x) {
//   for (int i = 0; i < 500; ++i)
//     foo(i);
// }
//
// It does this by looking for loops that, besides the latch code, are composed
// of isomorphic DAGs of instructions, with each DAG rooted at some increment
// to the induction variable, and where each DAG is isomorphic to the DAG
// rooted at the induction variable (excepting the sub-DAGs which root the
// other induction-variable increments). In other words, we're looking for loop
// bodies of the form:
//
// %iv = phi [ (preheader, ...), (body, %iv.next) ]
// f(%iv)
// %iv.1 = add %iv, 1                <-- a root increment
// f(%iv.1)
// %iv.2 = add %iv, 2                <-- a root increment
// f(%iv.2)
// %iv.scale_m_1 = add %iv, scale-1  <-- a root increment
// f(%iv.scale_m_1)
// ...
// %iv.next = add %iv, scale
// %cmp = icmp(%iv, ...)
// br %cmp, header, exit
//
// where each f(i) is a set of instructions that, collectively, are a function
// only of i (and other loop-invariant values).
//
// As a special case, we can also reroll loops like this:
//
// int foo(int);
// void bar(int *x) {
//   for (int i = 0; i < 500; ++i) {
//     x[3*i] = foo(0);
//     x[3*i+1] = foo(0);
//     x[3*i+2] = foo(0);
//   }
// }
//
// into this:
//
// void bar(int *x) {
//   for (int i = 0; i < 1500; ++i)
//     x[i] = foo(0);
// }
//
// in which case, we're looking for inputs like this:
//
// %iv = phi [ (preheader, ...), (body, %iv.next) ]
// %scaled.iv = mul %iv, scale
// f(%scaled.iv)
// %scaled.iv.1 = add %scaled.iv, 1
// f(%scaled.iv.1)
// %scaled.iv.2 = add %scaled.iv, 2
// f(%scaled.iv.2)
// %scaled.iv.scale_m_1 = add %scaled.iv, scale-1
// f(%scaled.iv.scale_m_1)
// ...
// %iv.next = add %iv, 1
// %cmp = icmp(%iv, ...)
// br %cmp, header, exit

namespace {
  enum IterationLimits {
    /// The maximum number of iterations that we'll try and reroll. This
    /// has to be less than 25 in order to fit into a SmallBitVector.
    IL_MaxRerollIterations = 16,
    /// The bitvector index used by loop induction variables and other
    /// instructions that belong to no one particular iteration.
    IL_LoopIncIdx,
    IL_End
  };

  class LoopReroll : public LoopPass {
  public:
    static char ID; // Pass ID, replacement for typeid
    LoopReroll() : LoopPass(ID) {
      initializeLoopRerollPass(*PassRegistry::getPassRegistry());
    }

    bool runOnLoop(Loop *L, LPPassManager &LPM) override;

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<AliasAnalysis>();
      AU.addRequired<LoopInfoWrapperPass>();
      AU.addPreserved<LoopInfoWrapperPass>();
      AU.addRequired<DominatorTreeWrapperPass>();
      AU.addPreserved<DominatorTreeWrapperPass>();
      AU.addRequired<ScalarEvolution>();
      AU.addRequired<TargetLibraryInfoWrapperPass>();
    }

  protected:
    AliasAnalysis *AA;
    LoopInfo *LI;
    ScalarEvolution *SE;
    const DataLayout *DL;
    TargetLibraryInfo *TLI;
    DominatorTree *DT;

    typedef SmallVector<Instruction *, 16> SmallInstructionVector;
    typedef SmallSet<Instruction *, 16>   SmallInstructionSet;

    // A chain of isomorphic instructions, indentified by a single-use PHI,
    // representing a reduction. Only the last value may be used outside the
    // loop.
    struct SimpleLoopReduction {
      SimpleLoopReduction(Instruction *P, Loop *L)
        : Valid(false), Instructions(1, P) {
        assert(isa<PHINode>(P) && "First reduction instruction must be a PHI");
        add(L);
      }

      bool valid() const {
        return Valid;
      }

      Instruction *getPHI() const {
        assert(Valid && "Using invalid reduction");
        return Instructions.front();
      }

      Instruction *getReducedValue() const {
        assert(Valid && "Using invalid reduction");
        return Instructions.back();
      }

      Instruction *get(size_t i) const {
        assert(Valid && "Using invalid reduction");
        return Instructions[i+1];
      }

      Instruction *operator [] (size_t i) const { return get(i); }

      // The size, ignoring the initial PHI.
      size_t size() const {
        assert(Valid && "Using invalid reduction");
        return Instructions.size()-1;
      }

      typedef SmallInstructionVector::iterator iterator;
      typedef SmallInstructionVector::const_iterator const_iterator;

      iterator begin() {
        assert(Valid && "Using invalid reduction");
        return std::next(Instructions.begin());
      }

      const_iterator begin() const {
        assert(Valid && "Using invalid reduction");
        return std::next(Instructions.begin());
      }

      iterator end() { return Instructions.end(); }
      const_iterator end() const { return Instructions.end(); }

    protected:
      bool Valid;
      SmallInstructionVector Instructions;

      void add(Loop *L);
    };

    // The set of all reductions, and state tracking of possible reductions
    // during loop instruction processing.
    struct ReductionTracker {
      typedef SmallVector<SimpleLoopReduction, 16> SmallReductionVector;

      // Add a new possible reduction.
      void addSLR(SimpleLoopReduction &SLR) { PossibleReds.push_back(SLR); }

      // Setup to track possible reductions corresponding to the provided
      // rerolling scale. Only reductions with a number of non-PHI instructions
      // that is divisible by the scale are considered. Three instructions sets
      // are filled in:
      //   - A set of all possible instructions in eligible reductions.
      //   - A set of all PHIs in eligible reductions
      //   - A set of all reduced values (last instructions) in eligible
      //     reductions.
      void restrictToScale(uint64_t Scale,
                           SmallInstructionSet &PossibleRedSet,
                           SmallInstructionSet &PossibleRedPHISet,
                           SmallInstructionSet &PossibleRedLastSet) {
        PossibleRedIdx.clear();
        PossibleRedIter.clear();
        Reds.clear();

        for (unsigned i = 0, e = PossibleReds.size(); i != e; ++i)
          if (PossibleReds[i].size() % Scale == 0) {
            PossibleRedLastSet.insert(PossibleReds[i].getReducedValue());
            PossibleRedPHISet.insert(PossibleReds[i].getPHI());

            PossibleRedSet.insert(PossibleReds[i].getPHI());
            PossibleRedIdx[PossibleReds[i].getPHI()] = i;
            for (Instruction *J : PossibleReds[i]) {
              PossibleRedSet.insert(J);
              PossibleRedIdx[J] = i;
            }
          }
      }

      // The functions below are used while processing the loop instructions.

      // Are the two instructions both from reductions, and furthermore, from
      // the same reduction?
      bool isPairInSame(Instruction *J1, Instruction *J2) {
        DenseMap<Instruction *, int>::iterator J1I = PossibleRedIdx.find(J1);
        if (J1I != PossibleRedIdx.end()) {
          DenseMap<Instruction *, int>::iterator J2I = PossibleRedIdx.find(J2);
          if (J2I != PossibleRedIdx.end() && J1I->second == J2I->second)
            return true;
        }

        return false;
      }

      // The two provided instructions, the first from the base iteration, and
      // the second from iteration i, form a matched pair. If these are part of
      // a reduction, record that fact.
      void recordPair(Instruction *J1, Instruction *J2, unsigned i) {
        if (PossibleRedIdx.count(J1)) {
          assert(PossibleRedIdx.count(J2) &&
                 "Recording reduction vs. non-reduction instruction?");

          PossibleRedIter[J1] = 0;
          PossibleRedIter[J2] = i;

          int Idx = PossibleRedIdx[J1];
          assert(Idx == PossibleRedIdx[J2] &&
                 "Recording pair from different reductions?");
          Reds.insert(Idx);
        }
      }

      // The functions below can be called after we've finished processing all
      // instructions in the loop, and we know which reductions were selected.

      // Is the provided instruction the PHI of a reduction selected for
      // rerolling?
      bool isSelectedPHI(Instruction *J) {
        if (!isa<PHINode>(J))
          return false;

        for (DenseSet<int>::iterator RI = Reds.begin(), RIE = Reds.end();
             RI != RIE; ++RI) {
          int i = *RI;
          if (cast<Instruction>(J) == PossibleReds[i].getPHI())
            return true;
        }

        return false;
      }

      bool validateSelected();
      void replaceSelected();

    protected:
      // The vector of all possible reductions (for any scale).
      SmallReductionVector PossibleReds;

      DenseMap<Instruction *, int> PossibleRedIdx;
      DenseMap<Instruction *, int> PossibleRedIter;
      DenseSet<int> Reds;
    };

    // The set of all DAG roots, and state tracking of all roots
    // for a particular induction variable.
    struct DAGRootTracker {
      DAGRootTracker(LoopReroll *Parent, Loop *L, Instruction *IV,
                     ScalarEvolution *SE, AliasAnalysis *AA,
                     TargetLibraryInfo *TLI, const DataLayout *DL)
        : Parent(Parent), L(L), SE(SE), AA(AA), TLI(TLI),
          DL(DL), IV(IV) {
      }

      /// Stage 1: Find all the DAG roots for the induction variable.
      bool findRoots();
      /// Stage 2: Validate if the found roots are valid.
      bool validate(ReductionTracker &Reductions);
      /// Stage 3: Assuming validate() returned true, perform the
      /// replacement.
      /// @param IterCount The maximum iteration count of L.
      void replace(const SCEV *IterCount);

    protected:
      typedef MapVector<Instruction*, SmallBitVector> UsesTy;

      bool findScaleFromMul();
      bool collectAllRoots();

      bool collectUsedInstructions(SmallInstructionSet &PossibleRedSet);
      void collectInLoopUserSet(const SmallInstructionVector &Roots,
                                const SmallInstructionSet &Exclude,
                                const SmallInstructionSet &Final,
                                DenseSet<Instruction *> &Users);
      void collectInLoopUserSet(Instruction *Root,
                                const SmallInstructionSet &Exclude,
                                const SmallInstructionSet &Final,
                                DenseSet<Instruction *> &Users);

      UsesTy::iterator nextInstr(int Val, UsesTy &In, UsesTy::iterator I);

      LoopReroll *Parent;

      // Members of Parent, replicated here for brevity.
      Loop *L;
      ScalarEvolution *SE;
      AliasAnalysis *AA;
      TargetLibraryInfo *TLI;
      const DataLayout *DL;

      // The loop induction variable.
      Instruction *IV;
      // Loop step amount.
      uint64_t Inc;
      // Loop reroll count; if Inc == 1, this records the scaling applied
      // to the indvar: a[i*2+0] = ...; a[i*2+1] = ... ;
      // If Inc is not 1, Scale = Inc.
      uint64_t Scale;
      // If Scale != Inc, then RealIV is IV after its multiplication.
      Instruction *RealIV;
      // The roots themselves.
      SmallInstructionVector Roots;
      // All increment instructions for IV.
      SmallInstructionVector LoopIncs;
      // Map of all instructions in the loop (in order) to the iterations
      // they are used in (or specially, IL_LoopIncIdx for instructions
      // used in the loop increment mechanism).
      UsesTy Uses;
    };

    void collectPossibleIVs(Loop *L, SmallInstructionVector &PossibleIVs);
    void collectPossibleReductions(Loop *L,
           ReductionTracker &Reductions);
    bool reroll(Instruction *IV, Loop *L, BasicBlock *Header, const SCEV *IterCount,
                ReductionTracker &Reductions);
  };
}

char LoopReroll::ID = 0;
INITIALIZE_PASS_BEGIN(LoopReroll, "loop-reroll", "Reroll loops", false, false)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(LoopReroll, "loop-reroll", "Reroll loops", false, false)

Pass *llvm::createLoopRerollPass() {
  return new LoopReroll;
}

// Returns true if the provided instruction is used outside the given loop.
// This operates like Instruction::isUsedOutsideOfBlock, but considers PHIs in
// non-loop blocks to be outside the loop.
static bool hasUsesOutsideLoop(Instruction *I, Loop *L) {
  for (User *U : I->users()) {
    if (!L->contains(cast<Instruction>(U)))
      return true;
  }
  return false;
}

// Collect the list of loop induction variables with respect to which it might
// be possible to reroll the loop.
void LoopReroll::collectPossibleIVs(Loop *L,
                                    SmallInstructionVector &PossibleIVs) {
  BasicBlock *Header = L->getHeader();
  for (BasicBlock::iterator I = Header->begin(),
       IE = Header->getFirstInsertionPt(); I != IE; ++I) {
    if (!isa<PHINode>(I))
      continue;
    if (!I->getType()->isIntegerTy())
      continue;

    if (const SCEVAddRecExpr *PHISCEV =
        dyn_cast<SCEVAddRecExpr>(SE->getSCEV(I))) {
      if (PHISCEV->getLoop() != L)
        continue;
      if (!PHISCEV->isAffine())
        continue;
      if (const SCEVConstant *IncSCEV =
          dyn_cast<SCEVConstant>(PHISCEV->getStepRecurrence(*SE))) {
        if (!IncSCEV->getValue()->getValue().isStrictlyPositive())
          continue;
        if (IncSCEV->getValue()->uge(MaxInc))
          continue;

        DEBUG(dbgs() << "LRR: Possible IV: " << *I << " = " <<
              *PHISCEV << "\n");
        PossibleIVs.push_back(I);
      }
    }
  }
}

// Add the remainder of the reduction-variable chain to the instruction vector
// (the initial PHINode has already been added). If successful, the object is
// marked as valid.
void LoopReroll::SimpleLoopReduction::add(Loop *L) {
  assert(!Valid && "Cannot add to an already-valid chain");

  // The reduction variable must be a chain of single-use instructions
  // (including the PHI), except for the last value (which is used by the PHI
  // and also outside the loop).
  Instruction *C = Instructions.front();

  do {
    C = cast<Instruction>(*C->user_begin());
    if (C->hasOneUse()) {
      if (!C->isBinaryOp())
        return;

      if (!(isa<PHINode>(Instructions.back()) ||
            C->isSameOperationAs(Instructions.back())))
        return;

      Instructions.push_back(C);
    }
  } while (C->hasOneUse());

  if (Instructions.size() < 2 ||
      !C->isSameOperationAs(Instructions.back()) ||
      C->use_empty())
    return;

  // C is now the (potential) last instruction in the reduction chain.
  for (User *U : C->users()) {
    // The only in-loop user can be the initial PHI.
    if (L->contains(cast<Instruction>(U)))
      if (cast<Instruction>(U) != Instructions.front())
        return;
  }

  Instructions.push_back(C);
  Valid = true;
}

// Collect the vector of possible reduction variables.
void LoopReroll::collectPossibleReductions(Loop *L,
  ReductionTracker &Reductions) {
  BasicBlock *Header = L->getHeader();
  for (BasicBlock::iterator I = Header->begin(),
       IE = Header->getFirstInsertionPt(); I != IE; ++I) {
    if (!isa<PHINode>(I))
      continue;
    if (!I->getType()->isSingleValueType())
      continue;

    SimpleLoopReduction SLR(I, L);
    if (!SLR.valid())
      continue;

    DEBUG(dbgs() << "LRR: Possible reduction: " << *I << " (with " <<
          SLR.size() << " chained instructions)\n");
    Reductions.addSLR(SLR);
  }
}

// Collect the set of all users of the provided root instruction. This set of
// users contains not only the direct users of the root instruction, but also
// all users of those users, and so on. There are two exceptions:
//
//   1. Instructions in the set of excluded instructions are never added to the
//   use set (even if they are users). This is used, for example, to exclude
//   including root increments in the use set of the primary IV.
//
//   2. Instructions in the set of final instructions are added to the use set
//   if they are users, but their users are not added. This is used, for
//   example, to prevent a reduction update from forcing all later reduction
//   updates into the use set.
void LoopReroll::DAGRootTracker::collectInLoopUserSet(
  Instruction *Root, const SmallInstructionSet &Exclude,
  const SmallInstructionSet &Final,
  DenseSet<Instruction *> &Users) {
  SmallInstructionVector Queue(1, Root);
  while (!Queue.empty()) {
    Instruction *I = Queue.pop_back_val();
    if (!Users.insert(I).second)
      continue;

    if (!Final.count(I))
      for (Use &U : I->uses()) {
        Instruction *User = cast<Instruction>(U.getUser());
        if (PHINode *PN = dyn_cast<PHINode>(User)) {
          // Ignore "wrap-around" uses to PHIs of this loop's header.
          if (PN->getIncomingBlock(U) == L->getHeader())
            continue;
        }

        if (L->contains(User) && !Exclude.count(User)) {
          Queue.push_back(User);
        }
      }

    // We also want to collect single-user "feeder" values.
    for (User::op_iterator OI = I->op_begin(),
         OIE = I->op_end(); OI != OIE; ++OI) {
      if (Instruction *Op = dyn_cast<Instruction>(*OI))
        if (Op->hasOneUse() && L->contains(Op) && !Exclude.count(Op) &&
            !Final.count(Op))
          Queue.push_back(Op);
    }
  }
}

// Collect all of the users of all of the provided root instructions (combined
// into a single set).
void LoopReroll::DAGRootTracker::collectInLoopUserSet(
  const SmallInstructionVector &Roots,
  const SmallInstructionSet &Exclude,
  const SmallInstructionSet &Final,
  DenseSet<Instruction *> &Users) {
  for (SmallInstructionVector::const_iterator I = Roots.begin(),
       IE = Roots.end(); I != IE; ++I)
    collectInLoopUserSet(*I, Exclude, Final, Users);
}

static bool isSimpleLoadStore(Instruction *I) {
  if (LoadInst *LI = dyn_cast<LoadInst>(I))
    return LI->isSimple();
  if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return SI->isSimple();
  if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(I))
    return !MI->isVolatile();
  return false;
}

bool LoopReroll::DAGRootTracker::findRoots() {

  const SCEVAddRecExpr *RealIVSCEV = cast<SCEVAddRecExpr>(SE->getSCEV(IV));
  Inc = cast<SCEVConstant>(RealIVSCEV->getOperand(1))->
    getValue()->getZExtValue();

  // The effective induction variable, IV, is normally also the real induction
  // variable. When we're dealing with a loop like:
  //   for (int i = 0; i < 500; ++i)
  //     x[3*i] = ...;
  //     x[3*i+1] = ...;
  //     x[3*i+2] = ...;
  // then the real IV is still i, but the effective IV is (3*i).
  Scale = Inc;
  RealIV = IV;
  if (Inc == 1 && !findScaleFromMul())
    return false;

  // The set of increment instructions for each increment value.
  if (!collectAllRoots())
    return false;

  if (Roots.size() > IL_MaxRerollIterations) {
    DEBUG(dbgs() << "LRR: Aborting - too many iterations found. "
          << "#Found=" << Roots.size() << ", #Max=" << IL_MaxRerollIterations
          << "\n");
    return false;
  }

  return true;
}

// Recognize loops that are setup like this:
//
// %iv = phi [ (preheader, ...), (body, %iv.next) ]
// %scaled.iv = mul %iv, scale
// f(%scaled.iv)
// %scaled.iv.1 = add %scaled.iv, 1
// f(%scaled.iv.1)
// %scaled.iv.2 = add %scaled.iv, 2
// f(%scaled.iv.2)
// %scaled.iv.scale_m_1 = add %scaled.iv, scale-1
// f(%scaled.iv.scale_m_1)
// ...
// %iv.next = add %iv, 1
// %cmp = icmp(%iv, ...)
// br %cmp, header, exit
//
// and, if found, set IV = %scaled.iv, and add %iv.next to LoopIncs.
bool LoopReroll::DAGRootTracker::findScaleFromMul() {

  // This is a special case: here we're looking for all uses (except for
  // the increment) to be multiplied by a common factor. The increment must
  // be by one. This is to capture loops like:
  //   for (int i = 0; i < 500; ++i) {
  //     foo(3*i); foo(3*i+1); foo(3*i+2);
  //   }
  if (RealIV->getNumUses() != 2)
    return false;
  const SCEVAddRecExpr *RealIVSCEV = cast<SCEVAddRecExpr>(SE->getSCEV(RealIV));
  Instruction *User1 = cast<Instruction>(*RealIV->user_begin()),
              *User2 = cast<Instruction>(*std::next(RealIV->user_begin()));
  if (!SE->isSCEVable(User1->getType()) || !SE->isSCEVable(User2->getType()))
    return false;
  const SCEVAddRecExpr *User1SCEV =
                         dyn_cast<SCEVAddRecExpr>(SE->getSCEV(User1)),
                       *User2SCEV =
                         dyn_cast<SCEVAddRecExpr>(SE->getSCEV(User2));
  if (!User1SCEV || !User1SCEV->isAffine() ||
      !User2SCEV || !User2SCEV->isAffine())
    return false;

  // We assume below that User1 is the scale multiply and User2 is the
  // increment. If this can't be true, then swap them.
  if (User1SCEV == RealIVSCEV->getPostIncExpr(*SE)) {
    std::swap(User1, User2);
    std::swap(User1SCEV, User2SCEV);
  }

  if (User2SCEV != RealIVSCEV->getPostIncExpr(*SE))
    return false;
  assert(User2SCEV->getStepRecurrence(*SE)->isOne() &&
         "Invalid non-unit step for multiplicative scaling");
  LoopIncs.push_back(User2);

  if (const SCEVConstant *MulScale =
      dyn_cast<SCEVConstant>(User1SCEV->getStepRecurrence(*SE))) {
    // Make sure that both the start and step have the same multiplier.
    if (RealIVSCEV->getStart()->getType() != MulScale->getType())
      return false;
    if (SE->getMulExpr(RealIVSCEV->getStart(), MulScale) !=
        User1SCEV->getStart())
      return false;

    ConstantInt *MulScaleCI = MulScale->getValue();
    if (!MulScaleCI->uge(2) || MulScaleCI->uge(MaxInc))
      return false;
    Scale = MulScaleCI->getZExtValue();
    IV = User1;
  } else
    return false;

  DEBUG(dbgs() << "LRR: Found possible scaling " << *User1 << "\n");

  assert(Scale <= MaxInc && "Scale is too large");
  assert(Scale > 1 && "Scale must be at least 2");

  return true;
}

// Collect all root increments with respect to the provided induction variable
// (normally the PHI, but sometimes a multiply). A root increment is an
// instruction, normally an add, with a positive constant less than Scale. In a
// rerollable loop, each of these increments is the root of an instruction
// graph isomorphic to the others. Also, we collect the final induction
// increment (the increment equal to the Scale), and its users in LoopIncs.
bool LoopReroll::DAGRootTracker::collectAllRoots() {
  Roots.resize(Scale-1);
  
  for (User *U : IV->users()) {
    Instruction *UI = cast<Instruction>(U);
    if (!SE->isSCEVable(UI->getType()))
      continue;
    if (UI->getType() != IV->getType())
      continue;
    if (!L->contains(UI))
      continue;
    if (hasUsesOutsideLoop(UI, L))
      continue;

    if (const SCEVConstant *Diff = dyn_cast<SCEVConstant>(SE->getMinusSCEV(
          SE->getSCEV(UI), SE->getSCEV(IV)))) {
      uint64_t Idx = Diff->getValue()->getValue().getZExtValue();
      if (Idx > 0 && Idx < Scale) {
        if (Roots[Idx-1])
          // No duplicates allowed.
          return false;
        Roots[Idx-1] = UI;
      } else if (Idx == Scale && Inc > 1) {
        LoopIncs.push_back(UI);
      }
    }
  }

  for (unsigned i = 0; i < Scale-1; ++i) {
    if (!Roots[i])
      return false;
  }

  return true;
}

bool LoopReroll::DAGRootTracker::collectUsedInstructions(SmallInstructionSet &PossibleRedSet) {
  // Populate the MapVector with all instructions in the block, in order first,
  // so we can iterate over the contents later in perfect order.
  for (auto &I : *L->getHeader()) {
    Uses[&I].resize(IL_End);
  }

  SmallInstructionSet Exclude;
  Exclude.insert(Roots.begin(), Roots.end());
  Exclude.insert(LoopIncs.begin(), LoopIncs.end());

  DenseSet<Instruction*> VBase;
  collectInLoopUserSet(IV, Exclude, PossibleRedSet, VBase);
  for (auto *I : VBase) {
    Uses[I].set(0);
  }

  unsigned Idx = 1;
  for (auto *Root : Roots) {
    DenseSet<Instruction*> V;
    collectInLoopUserSet(Root, Exclude, PossibleRedSet, V);

    // While we're here, check the use sets are the same size.
    if (V.size() != VBase.size()) {
      DEBUG(dbgs() << "LRR: Aborting - use sets are different sizes\n");
      return false;
    }

    for (auto *I : V) {
      Uses[I].set(Idx);
    }
    ++Idx;
  }

  // Make sure the loop increments are also accounted for.
  Exclude.clear();
  Exclude.insert(Roots.begin(), Roots.end());

  DenseSet<Instruction*> V;
  collectInLoopUserSet(LoopIncs, Exclude, PossibleRedSet, V);
  for (auto *I : V) {
    Uses[I].set(IL_LoopIncIdx);
  }
  if (IV != RealIV)
    Uses[RealIV].set(IL_LoopIncIdx);

  return true;

}

LoopReroll::DAGRootTracker::UsesTy::iterator
LoopReroll::DAGRootTracker::nextInstr(int Val, UsesTy &In,
                                      UsesTy::iterator I) {
  while (I != In.end() && I->second.test(Val) == 0)
    ++I;
  return I;
}

bool LoopReroll::DAGRootTracker::validate(ReductionTracker &Reductions) {
  // We now need to check for equivalence of the use graph of each root with
  // that of the primary induction variable (excluding the roots). Our goal
  // here is not to solve the full graph isomorphism problem, but rather to
  // catch common cases without a lot of work. As a result, we will assume
  // that the relative order of the instructions in each unrolled iteration
  // is the same (although we will not make an assumption about how the
  // different iterations are intermixed). Note that while the order must be
  // the same, the instructions may not be in the same basic block.

  // An array of just the possible reductions for this scale factor. When we
  // collect the set of all users of some root instructions, these reduction
  // instructions are treated as 'final' (their uses are not considered).
  // This is important because we don't want the root use set to search down
  // the reduction chain.
  SmallInstructionSet PossibleRedSet;
  SmallInstructionSet PossibleRedLastSet;
  SmallInstructionSet PossibleRedPHISet;
  Reductions.restrictToScale(Scale, PossibleRedSet,
                             PossibleRedPHISet, PossibleRedLastSet);

  // Populate "Uses" with where each instruction is used.
  if (!collectUsedInstructions(PossibleRedSet))
    return false;

  // Make sure we mark the reduction PHIs as used in all iterations.
  for (auto *I : PossibleRedPHISet) {
    Uses[I].set(IL_LoopIncIdx);
  }

  // Make sure all instructions in the loop are in one and only one
  // set.
  for (auto &KV : Uses) {
    if (KV.second.count() != 1) {
      DEBUG(dbgs() << "LRR: Aborting - instruction is not used in 1 iteration: "
            << *KV.first << " (#uses=" << KV.second.count() << ")\n");
      return false;
    }
  }

  DEBUG(
    for (auto &KV : Uses) {
      dbgs() << "LRR: " << KV.second.find_first() << "\t" << *KV.first << "\n";
    }
    );

  for (unsigned Iter = 1; Iter < Scale; ++Iter) {
    // In addition to regular aliasing information, we need to look for
    // instructions from later (future) iterations that have side effects
    // preventing us from reordering them past other instructions with side
    // effects.
    bool FutureSideEffects = false;
    AliasSetTracker AST(*AA);
    // The map between instructions in f(%iv.(i+1)) and f(%iv).
    DenseMap<Value *, Value *> BaseMap;

    // Compare iteration Iter to the base.
    auto BaseIt = nextInstr(0, Uses, Uses.begin());
    auto RootIt = nextInstr(Iter, Uses, Uses.begin());
    auto LastRootIt = Uses.begin();

    while (BaseIt != Uses.end() && RootIt != Uses.end()) {
      Instruction *BaseInst = BaseIt->first;
      Instruction *RootInst = RootIt->first;

      // Skip over the IV or root instructions; only match their users.
      bool Continue = false;
      if (BaseInst == RealIV || BaseInst == IV) {
        BaseIt = nextInstr(0, Uses, ++BaseIt);
        Continue = true;
      }
      if (std::find(Roots.begin(), Roots.end(), RootInst) != Roots.end()) {
        LastRootIt = RootIt;
        RootIt = nextInstr(Iter, Uses, ++RootIt);
        Continue = true;
      }
      if (Continue) continue;

      // All instructions between the last root and this root
      // belong to some other iteration. If they belong to a 
      // future iteration, then they're dangerous to alias with.
      for (; LastRootIt != RootIt; ++LastRootIt) {
        Instruction *I = LastRootIt->first;
        if (LastRootIt->second.find_first() < (int)Iter)
          continue;
        if (I->mayWriteToMemory())
          AST.add(I);
        // Note: This is specifically guarded by a check on isa<PHINode>,
        // which while a valid (somewhat arbitrary) micro-optimization, is
        // needed because otherwise isSafeToSpeculativelyExecute returns
        // false on PHI nodes.
        if (!isa<PHINode>(I) && !isSimpleLoadStore(I) &&
            !isSafeToSpeculativelyExecute(I, DL))
          // Intervening instructions cause side effects.
          FutureSideEffects = true;
      }

      if (!BaseInst->isSameOperationAs(RootInst)) {
        DEBUG(dbgs() << "LRR: iteration root match failed at " << *BaseInst <<
              " vs. " << *RootInst << "\n");
        return false;
      }

      // Make sure that this instruction, which is in the use set of this
      // root instruction, does not also belong to the base set or the set of
      // some other root instruction.
      if (RootIt->second.count() > 1) {
        DEBUG(dbgs() << "LRR: iteration root match failed at " << *BaseInst <<
                        " vs. " << *RootInst << " (prev. case overlap)\n");
        return false;
      }

      // Make sure that we don't alias with any instruction in the alias set
      // tracker. If we do, then we depend on a future iteration, and we
      // can't reroll.
      if (RootInst->mayReadFromMemory())
        for (auto &K : AST) {
          if (K.aliasesUnknownInst(RootInst, *AA)) {
            DEBUG(dbgs() << "LRR: iteration root match failed at " << *BaseInst <<
                            " vs. " << *RootInst << " (depends on future store)\n");
            return false;
          }
        }

      // If we've past an instruction from a future iteration that may have
      // side effects, and this instruction might also, then we can't reorder
      // them, and this matching fails. As an exception, we allow the alias
      // set tracker to handle regular (simple) load/store dependencies.
      if (FutureSideEffects &&
            ((!isSimpleLoadStore(BaseInst) &&
              !isSafeToSpeculativelyExecute(BaseInst, DL)) ||
             (!isSimpleLoadStore(RootInst) &&
              !isSafeToSpeculativelyExecute(RootInst, DL)))) {
        DEBUG(dbgs() << "LRR: iteration root match failed at " << *BaseInst <<
                        " vs. " << *RootInst <<
                        " (side effects prevent reordering)\n");
        return false;
      }

      // For instructions that are part of a reduction, if the operation is
      // associative, then don't bother matching the operands (because we
      // already know that the instructions are isomorphic, and the order
      // within the iteration does not matter). For non-associative reductions,
      // we do need to match the operands, because we need to reject
      // out-of-order instructions within an iteration!
      // For example (assume floating-point addition), we need to reject this:
      //   x += a[i]; x += b[i];
      //   x += a[i+1]; x += b[i+1];
      //   x += b[i+2]; x += a[i+2];
      bool InReduction = Reductions.isPairInSame(BaseInst, RootInst);

      if (!(InReduction && BaseInst->isAssociative())) {
        bool Swapped = false, SomeOpMatched = false;
        for (unsigned j = 0; j < BaseInst->getNumOperands(); ++j) {
          Value *Op2 = RootInst->getOperand(j);

          // If this is part of a reduction (and the operation is not
          // associatve), then we match all operands, but not those that are
          // part of the reduction.
          if (InReduction)
            if (Instruction *Op2I = dyn_cast<Instruction>(Op2))
              if (Reductions.isPairInSame(RootInst, Op2I))
                continue;

          DenseMap<Value *, Value *>::iterator BMI = BaseMap.find(Op2);
          if (BMI != BaseMap.end())
            Op2 = BMI->second;
          else if (Roots[Iter-1] == (Instruction*) Op2)
            Op2 = IV;

          if (BaseInst->getOperand(Swapped ? unsigned(!j) : j) != Op2) {
            // If we've not already decided to swap the matched operands, and
            // we've not already matched our first operand (note that we could
            // have skipped matching the first operand because it is part of a
            // reduction above), and the instruction is commutative, then try
            // the swapped match.
            if (!Swapped && BaseInst->isCommutative() && !SomeOpMatched &&
                BaseInst->getOperand(!j) == Op2) {
              Swapped = true;
            } else {
              DEBUG(dbgs() << "LRR: iteration root match failed at " << *BaseInst
                    << " vs. " << *RootInst << " (operand " << j << ")\n");
              return false;
            }
          }

          SomeOpMatched = true;
        }
      }

      if ((!PossibleRedLastSet.count(BaseInst) &&
           hasUsesOutsideLoop(BaseInst, L)) ||
          (!PossibleRedLastSet.count(RootInst) &&
           hasUsesOutsideLoop(RootInst, L))) {
        DEBUG(dbgs() << "LRR: iteration root match failed at " << *BaseInst <<
                        " vs. " << *RootInst << " (uses outside loop)\n");
        return false;
      }

      Reductions.recordPair(BaseInst, RootInst, Iter);
      BaseMap.insert(std::make_pair(RootInst, BaseInst));

      LastRootIt = RootIt;
      BaseIt = nextInstr(0, Uses, ++BaseIt);
      RootIt = nextInstr(Iter, Uses, ++RootIt);
    }
    assert (BaseIt == Uses.end() && RootIt == Uses.end() &&
            "Mismatched set sizes!");
  }

  DEBUG(dbgs() << "LRR: Matched all iteration increments for " <<
                  *RealIV << "\n");

  return true;
}

void LoopReroll::DAGRootTracker::replace(const SCEV *IterCount) {
  BasicBlock *Header = L->getHeader();
  // Remove instructions associated with non-base iterations.
  for (BasicBlock::reverse_iterator J = Header->rbegin();
       J != Header->rend();) {
    unsigned I = Uses[&*J].find_first();
    if (I > 0 && I < IL_LoopIncIdx) {
      Instruction *D = &*J;
      DEBUG(dbgs() << "LRR: removing: " << *D << "\n");
      D->eraseFromParent();
      continue;
    }

    ++J;
  }

  // Insert the new induction variable.
  const SCEVAddRecExpr *RealIVSCEV = cast<SCEVAddRecExpr>(SE->getSCEV(RealIV));
  const SCEV *Start = RealIVSCEV->getStart();
  if (Inc == 1)
    Start = SE->getMulExpr(Start,
                           SE->getConstant(Start->getType(), Scale));
  const SCEVAddRecExpr *H =
    cast<SCEVAddRecExpr>(SE->getAddRecExpr(Start,
                           SE->getConstant(RealIVSCEV->getType(), 1),
                           L, SCEV::FlagAnyWrap));
  { // Limit the lifetime of SCEVExpander.
    SCEVExpander Expander(*SE, "reroll");
    Value *NewIV = Expander.expandCodeFor(H, IV->getType(), Header->begin());

    for (auto &KV : Uses) {
      if (KV.second.find_first() == 0)
        KV.first->replaceUsesOfWith(IV, NewIV);
    }

    if (BranchInst *BI = dyn_cast<BranchInst>(Header->getTerminator())) {
      // FIXME: Why do we need this check?
      if (Uses[BI].find_first() == IL_LoopIncIdx) {
        const SCEV *ICSCEV = RealIVSCEV->evaluateAtIteration(IterCount, *SE);
        if (Inc == 1)
          ICSCEV =
            SE->getMulExpr(ICSCEV, SE->getConstant(ICSCEV->getType(), Scale));
        // Iteration count SCEV minus 1
        const SCEV *ICMinus1SCEV =
          SE->getMinusSCEV(ICSCEV, SE->getConstant(ICSCEV->getType(), 1));

        Value *ICMinus1; // Iteration count minus 1
        if (isa<SCEVConstant>(ICMinus1SCEV)) {
          ICMinus1 = Expander.expandCodeFor(ICMinus1SCEV, NewIV->getType(), BI);
        } else {
          BasicBlock *Preheader = L->getLoopPreheader();
          if (!Preheader)
            Preheader = InsertPreheaderForLoop(L, Parent);

          ICMinus1 = Expander.expandCodeFor(ICMinus1SCEV, NewIV->getType(),
                                            Preheader->getTerminator());
        }

        Value *Cond =
            new ICmpInst(BI, CmpInst::ICMP_EQ, NewIV, ICMinus1, "exitcond");
        BI->setCondition(Cond);

        if (BI->getSuccessor(1) != Header)
          BI->swapSuccessors();
      }
    }
  }

  SimplifyInstructionsInBlock(Header, DL, TLI);
  DeleteDeadPHIs(Header, TLI);
}

// Validate the selected reductions. All iterations must have an isomorphic
// part of the reduction chain and, for non-associative reductions, the chain
// entries must appear in order.
bool LoopReroll::ReductionTracker::validateSelected() {
  // For a non-associative reduction, the chain entries must appear in order.
  for (DenseSet<int>::iterator RI = Reds.begin(), RIE = Reds.end();
       RI != RIE; ++RI) {
    int i = *RI;
    int PrevIter = 0, BaseCount = 0, Count = 0;
    for (Instruction *J : PossibleReds[i]) {
      // Note that all instructions in the chain must have been found because
      // all instructions in the function must have been assigned to some
      // iteration.
      int Iter = PossibleRedIter[J];
      if (Iter != PrevIter && Iter != PrevIter + 1 &&
          !PossibleReds[i].getReducedValue()->isAssociative()) {
        DEBUG(dbgs() << "LRR: Out-of-order non-associative reduction: " <<
                        J << "\n");
        return false;
      }

      if (Iter != PrevIter) {
        if (Count != BaseCount) {
          DEBUG(dbgs() << "LRR: Iteration " << PrevIter <<
                " reduction use count " << Count <<
                " is not equal to the base use count " <<
                BaseCount << "\n");
          return false;
        }

        Count = 0;
      }

      ++Count;
      if (Iter == 0)
        ++BaseCount;

      PrevIter = Iter;
    }
  }

  return true;
}

// For all selected reductions, remove all parts except those in the first
// iteration (and the PHI). Replace outside uses of the reduced value with uses
// of the first-iteration reduced value (in other words, reroll the selected
// reductions).
void LoopReroll::ReductionTracker::replaceSelected() {
  // Fixup reductions to refer to the last instruction associated with the
  // first iteration (not the last).
  for (DenseSet<int>::iterator RI = Reds.begin(), RIE = Reds.end();
       RI != RIE; ++RI) {
    int i = *RI;
    int j = 0;
    for (int e = PossibleReds[i].size(); j != e; ++j)
      if (PossibleRedIter[PossibleReds[i][j]] != 0) {
        --j;
        break;
      }

    // Replace users with the new end-of-chain value.
    SmallInstructionVector Users;
    for (User *U : PossibleReds[i].getReducedValue()->users()) {
      Users.push_back(cast<Instruction>(U));
    }

    for (SmallInstructionVector::iterator J = Users.begin(),
         JE = Users.end(); J != JE; ++J)
      (*J)->replaceUsesOfWith(PossibleReds[i].getReducedValue(),
                              PossibleReds[i][j]);
  }
}

// Reroll the provided loop with respect to the provided induction variable.
// Generally, we're looking for a loop like this:
//
// %iv = phi [ (preheader, ...), (body, %iv.next) ]
// f(%iv)
// %iv.1 = add %iv, 1                <-- a root increment
// f(%iv.1)
// %iv.2 = add %iv, 2                <-- a root increment
// f(%iv.2)
// %iv.scale_m_1 = add %iv, scale-1  <-- a root increment
// f(%iv.scale_m_1)
// ...
// %iv.next = add %iv, scale
// %cmp = icmp(%iv, ...)
// br %cmp, header, exit
//
// Notably, we do not require that f(%iv), f(%iv.1), etc. be isolated groups of
// instructions. In other words, the instructions in f(%iv), f(%iv.1), etc. can
// be intermixed with eachother. The restriction imposed by this algorithm is
// that the relative order of the isomorphic instructions in f(%iv), f(%iv.1),
// etc. be the same.
//
// First, we collect the use set of %iv, excluding the other increment roots.
// This gives us f(%iv). Then we iterate over the loop instructions (scale-1)
// times, having collected the use set of f(%iv.(i+1)), during which we:
//   - Ensure that the next unmatched instruction in f(%iv) is isomorphic to
//     the next unmatched instruction in f(%iv.(i+1)).
//   - Ensure that both matched instructions don't have any external users
//     (with the exception of last-in-chain reduction instructions).
//   - Track the (aliasing) write set, and other side effects, of all
//     instructions that belong to future iterations that come before the matched
//     instructions. If the matched instructions read from that write set, then
//     f(%iv) or f(%iv.(i+1)) has some dependency on instructions in
//     f(%iv.(j+1)) for some j > i, and we cannot reroll the loop. Similarly,
//     if any of these future instructions had side effects (could not be
//     speculatively executed), and so do the matched instructions, when we
//     cannot reorder those side-effect-producing instructions, and rerolling
//     fails.
//
// Finally, we make sure that all loop instructions are either loop increment
// roots, belong to simple latch code, parts of validated reductions, part of
// f(%iv) or part of some f(%iv.i). If all of that is true (and all reductions
// have been validated), then we reroll the loop.
bool LoopReroll::reroll(Instruction *IV, Loop *L, BasicBlock *Header,
                        const SCEV *IterCount,
                        ReductionTracker &Reductions) {
  DAGRootTracker DAGRoots(this, L, IV, SE, AA, TLI, DL);

  if (!DAGRoots.findRoots())
    return false;
  DEBUG(dbgs() << "LRR: Found all root induction increments for: " <<
                  *IV << "\n");
  
  if (!DAGRoots.validate(Reductions))
    return false;
  if (!Reductions.validateSelected())
    return false;
  // At this point, we've validated the rerolling, and we're committed to
  // making changes!

  Reductions.replaceSelected();
  DAGRoots.replace(IterCount);

  ++NumRerolledLoops;
  return true;
}

bool LoopReroll::runOnLoop(Loop *L, LPPassManager &LPM) {
  if (skipOptnoneFunction(L))
    return false;

  AA = &getAnalysis<AliasAnalysis>();
  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  SE = &getAnalysis<ScalarEvolution>();
  TLI = &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
  DataLayoutPass *DLP = getAnalysisIfAvailable<DataLayoutPass>();
  DL = DLP ? &DLP->getDataLayout() : nullptr;
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();

  BasicBlock *Header = L->getHeader();
  DEBUG(dbgs() << "LRR: F[" << Header->getParent()->getName() <<
        "] Loop %" << Header->getName() << " (" <<
        L->getNumBlocks() << " block(s))\n");

  bool Changed = false;

  // For now, we'll handle only single BB loops.
  if (L->getNumBlocks() > 1)
    return Changed;

  if (!SE->hasLoopInvariantBackedgeTakenCount(L))
    return Changed;

  const SCEV *LIBETC = SE->getBackedgeTakenCount(L);
  const SCEV *IterCount =
    SE->getAddExpr(LIBETC, SE->getConstant(LIBETC->getType(), 1));
  DEBUG(dbgs() << "LRR: iteration count = " << *IterCount << "\n");

  // First, we need to find the induction variable with respect to which we can
  // reroll (there may be several possible options).
  SmallInstructionVector PossibleIVs;
  collectPossibleIVs(L, PossibleIVs);

  if (PossibleIVs.empty()) {
    DEBUG(dbgs() << "LRR: No possible IVs found\n");
    return Changed;
  }

  ReductionTracker Reductions;
  collectPossibleReductions(L, Reductions);

  // For each possible IV, collect the associated possible set of 'root' nodes
  // (i+1, i+2, etc.).
  for (SmallInstructionVector::iterator I = PossibleIVs.begin(),
       IE = PossibleIVs.end(); I != IE; ++I)
    if (reroll(*I, L, Header, IterCount, Reductions)) {
      Changed = true;
      break;
    }

  return Changed;
}
