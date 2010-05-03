//===- IndVarSimplify.cpp - Induction Variable Elimination ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This transformation analyzes and transforms the induction variables (and
// computations derived from them) into simpler forms suitable for subsequent
// analysis and transformation.
//
// This transformation makes the following changes to each loop with an
// identifiable induction variable:
//   1. All loops are transformed to have a SINGLE canonical induction variable
//      which starts at zero and steps by one.
//   2. The canonical induction variable is guaranteed to be the first PHI node
//      in the loop header block.
//   3. The canonical induction variable is guaranteed to be in a wide enough
//      type so that IV expressions need not be (directly) zero-extended or
//      sign-extended.
//   4. Any pointer arithmetic recurrences are raised to use array subscripts.
//
// If the trip count of a loop is computable, this pass also makes the following
// changes:
//   1. The exit condition for the loop is canonicalized to compare the
//      induction value against the exit value.  This turns loops like:
//        'for (i = 7; i*i < 1000; ++i)' into 'for (i = 0; i != 25; ++i)'
//   2. Any use outside of the loop of an expression derived from the indvar
//      is changed to compute the derived value outside of the loop, eliminating
//      the dependence on the exit value of the induction variable.  If the only
//      purpose of the loop is to compute the exit value of some derived
//      expression, this transformation will make the loop dead.
//
// This transformation should be followed by strength reduction after all of the
// desired loop transformations have been performed.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "indvars"
#include "llvm/Transforms/Scalar.h"
#include "llvm/BasicBlock.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/LLVMContext.h"
#include "llvm/Type.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/IVUsers.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
using namespace llvm;

STATISTIC(NumRemoved , "Number of aux indvars removed");
STATISTIC(NumInserted, "Number of canonical indvars added");
STATISTIC(NumReplaced, "Number of exit values replaced");
STATISTIC(NumLFTR    , "Number of loop exit tests replaced");

namespace {
  class IndVarSimplify : public LoopPass {
    IVUsers         *IU;
    LoopInfo        *LI;
    ScalarEvolution *SE;
    DominatorTree   *DT;
    bool Changed;
  public:

    static char ID; // Pass identification, replacement for typeid
    IndVarSimplify() : LoopPass(&ID) {}

    virtual bool runOnLoop(Loop *L, LPPassManager &LPM);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<DominatorTree>();
      AU.addRequired<LoopInfo>();
      AU.addRequired<ScalarEvolution>();
      AU.addRequiredID(LoopSimplifyID);
      AU.addRequiredID(LCSSAID);
      AU.addRequired<IVUsers>();
      AU.addPreserved<ScalarEvolution>();
      AU.addPreservedID(LoopSimplifyID);
      AU.addPreservedID(LCSSAID);
      AU.addPreserved<IVUsers>();
      AU.setPreservesCFG();
    }

  private:

    void EliminateIVComparisons();
    void EliminateIVRemainders();
    void RewriteNonIntegerIVs(Loop *L);

    ICmpInst *LinearFunctionTestReplace(Loop *L, const SCEV *BackedgeTakenCount,
                                   Value *IndVar,
                                   BasicBlock *ExitingBlock,
                                   BranchInst *BI,
                                   SCEVExpander &Rewriter);
    void RewriteLoopExitValues(Loop *L, SCEVExpander &Rewriter);

    void RewriteIVExpressions(Loop *L, SCEVExpander &Rewriter);

    void SinkUnusedInvariants(Loop *L);

    void HandleFloatingPointIV(Loop *L, PHINode *PH);
  };
}

char IndVarSimplify::ID = 0;
static RegisterPass<IndVarSimplify>
X("indvars", "Canonicalize Induction Variables");

Pass *llvm::createIndVarSimplifyPass() {
  return new IndVarSimplify();
}

/// LinearFunctionTestReplace - This method rewrites the exit condition of the
/// loop to be a canonical != comparison against the incremented loop induction
/// variable.  This pass is able to rewrite the exit tests of any loop where the
/// SCEV analysis can determine a loop-invariant trip count of the loop, which
/// is actually a much broader range than just linear tests.
ICmpInst *IndVarSimplify::LinearFunctionTestReplace(Loop *L,
                                   const SCEV *BackedgeTakenCount,
                                   Value *IndVar,
                                   BasicBlock *ExitingBlock,
                                   BranchInst *BI,
                                   SCEVExpander &Rewriter) {
  // Special case: If the backedge-taken count is a UDiv, it's very likely a
  // UDiv that ScalarEvolution produced in order to compute a precise
  // expression, rather than a UDiv from the user's code. If we can't find a
  // UDiv in the code with some simple searching, assume the former and forego
  // rewriting the loop.
  if (isa<SCEVUDivExpr>(BackedgeTakenCount)) {
    ICmpInst *OrigCond = dyn_cast<ICmpInst>(BI->getCondition());
    if (!OrigCond) return 0;
    const SCEV *R = SE->getSCEV(OrigCond->getOperand(1));
    R = SE->getMinusSCEV(R, SE->getConstant(R->getType(), 1));
    if (R != BackedgeTakenCount) {
      const SCEV *L = SE->getSCEV(OrigCond->getOperand(0));
      L = SE->getMinusSCEV(L, SE->getConstant(L->getType(), 1));
      if (L != BackedgeTakenCount)
        return 0;
    }
  }

  // If the exiting block is not the same as the backedge block, we must compare
  // against the preincremented value, otherwise we prefer to compare against
  // the post-incremented value.
  Value *CmpIndVar;
  const SCEV *RHS = BackedgeTakenCount;
  if (ExitingBlock == L->getLoopLatch()) {
    // Add one to the "backedge-taken" count to get the trip count.
    // If this addition may overflow, we have to be more pessimistic and
    // cast the induction variable before doing the add.
    const SCEV *Zero = SE->getConstant(BackedgeTakenCount->getType(), 0);
    const SCEV *N =
      SE->getAddExpr(BackedgeTakenCount,
                     SE->getConstant(BackedgeTakenCount->getType(), 1));
    if ((isa<SCEVConstant>(N) && !N->isZero()) ||
        SE->isLoopEntryGuardedByCond(L, ICmpInst::ICMP_NE, N, Zero)) {
      // No overflow. Cast the sum.
      RHS = SE->getTruncateOrZeroExtend(N, IndVar->getType());
    } else {
      // Potential overflow. Cast before doing the add.
      RHS = SE->getTruncateOrZeroExtend(BackedgeTakenCount,
                                        IndVar->getType());
      RHS = SE->getAddExpr(RHS,
                           SE->getConstant(IndVar->getType(), 1));
    }

    // The BackedgeTaken expression contains the number of times that the
    // backedge branches to the loop header.  This is one less than the
    // number of times the loop executes, so use the incremented indvar.
    CmpIndVar = L->getCanonicalInductionVariableIncrement();
  } else {
    // We have to use the preincremented value...
    RHS = SE->getTruncateOrZeroExtend(BackedgeTakenCount,
                                      IndVar->getType());
    CmpIndVar = IndVar;
  }

  // Expand the code for the iteration count.
  assert(RHS->isLoopInvariant(L) &&
         "Computed iteration count is not loop invariant!");
  Value *ExitCnt = Rewriter.expandCodeFor(RHS, IndVar->getType(), BI);

  // Insert a new icmp_ne or icmp_eq instruction before the branch.
  ICmpInst::Predicate Opcode;
  if (L->contains(BI->getSuccessor(0)))
    Opcode = ICmpInst::ICMP_NE;
  else
    Opcode = ICmpInst::ICMP_EQ;

  DEBUG(dbgs() << "INDVARS: Rewriting loop exit condition to:\n"
               << "      LHS:" << *CmpIndVar << '\n'
               << "       op:\t"
               << (Opcode == ICmpInst::ICMP_NE ? "!=" : "==") << "\n"
               << "      RHS:\t" << *RHS << "\n");

  ICmpInst *Cond = new ICmpInst(BI, Opcode, CmpIndVar, ExitCnt, "exitcond");

  Value *OrigCond = BI->getCondition();
  // It's tempting to use replaceAllUsesWith here to fully replace the old
  // comparison, but that's not immediately safe, since users of the old
  // comparison may not be dominated by the new comparison. Instead, just
  // update the branch to use the new comparison; in the common case this
  // will make old comparison dead.
  BI->setCondition(Cond);
  RecursivelyDeleteTriviallyDeadInstructions(OrigCond);

  ++NumLFTR;
  Changed = true;
  return Cond;
}

/// RewriteLoopExitValues - Check to see if this loop has a computable
/// loop-invariant execution count.  If so, this means that we can compute the
/// final value of any expressions that are recurrent in the loop, and
/// substitute the exit values from the loop into any instructions outside of
/// the loop that use the final values of the current expressions.
///
/// This is mostly redundant with the regular IndVarSimplify activities that
/// happen later, except that it's more powerful in some cases, because it's
/// able to brute-force evaluate arbitrary instructions as long as they have
/// constant operands at the beginning of the loop.
void IndVarSimplify::RewriteLoopExitValues(Loop *L,
                                           SCEVExpander &Rewriter) {
  // Verify the input to the pass in already in LCSSA form.
  assert(L->isLCSSAForm(*DT));

  SmallVector<BasicBlock*, 8> ExitBlocks;
  L->getUniqueExitBlocks(ExitBlocks);

  // Find all values that are computed inside the loop, but used outside of it.
  // Because of LCSSA, these values will only occur in LCSSA PHI Nodes.  Scan
  // the exit blocks of the loop to find them.
  for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i) {
    BasicBlock *ExitBB = ExitBlocks[i];

    // If there are no PHI nodes in this exit block, then no values defined
    // inside the loop are used on this path, skip it.
    PHINode *PN = dyn_cast<PHINode>(ExitBB->begin());
    if (!PN) continue;

    unsigned NumPreds = PN->getNumIncomingValues();

    // Iterate over all of the PHI nodes.
    BasicBlock::iterator BBI = ExitBB->begin();
    while ((PN = dyn_cast<PHINode>(BBI++))) {
      if (PN->use_empty())
        continue; // dead use, don't replace it

      // SCEV only supports integer expressions for now.
      if (!PN->getType()->isIntegerTy() && !PN->getType()->isPointerTy())
        continue;

      // It's necessary to tell ScalarEvolution about this explicitly so that
      // it can walk the def-use list and forget all SCEVs, as it may not be
      // watching the PHI itself. Once the new exit value is in place, there
      // may not be a def-use connection between the loop and every instruction
      // which got a SCEVAddRecExpr for that loop.
      SE->forgetValue(PN);

      // Iterate over all of the values in all the PHI nodes.
      for (unsigned i = 0; i != NumPreds; ++i) {
        // If the value being merged in is not integer or is not defined
        // in the loop, skip it.
        Value *InVal = PN->getIncomingValue(i);
        if (!isa<Instruction>(InVal))
          continue;

        // If this pred is for a subloop, not L itself, skip it.
        if (LI->getLoopFor(PN->getIncomingBlock(i)) != L)
          continue; // The Block is in a subloop, skip it.

        // Check that InVal is defined in the loop.
        Instruction *Inst = cast<Instruction>(InVal);
        if (!L->contains(Inst))
          continue;

        // Okay, this instruction has a user outside of the current loop
        // and varies predictably *inside* the loop.  Evaluate the value it
        // contains when the loop exits, if possible.
        const SCEV *ExitValue = SE->getSCEVAtScope(Inst, L->getParentLoop());
        if (!ExitValue->isLoopInvariant(L))
          continue;

        Changed = true;
        ++NumReplaced;

        Value *ExitVal = Rewriter.expandCodeFor(ExitValue, PN->getType(), Inst);

        DEBUG(dbgs() << "INDVARS: RLEV: AfterLoopVal = " << *ExitVal << '\n'
                     << "  LoopVal = " << *Inst << "\n");

        PN->setIncomingValue(i, ExitVal);

        // If this instruction is dead now, delete it.
        RecursivelyDeleteTriviallyDeadInstructions(Inst);

        if (NumPreds == 1) {
          // Completely replace a single-pred PHI. This is safe, because the
          // NewVal won't be variant in the loop, so we don't need an LCSSA phi
          // node anymore.
          PN->replaceAllUsesWith(ExitVal);
          RecursivelyDeleteTriviallyDeadInstructions(PN);
        }
      }
      if (NumPreds != 1) {
        // Clone the PHI and delete the original one. This lets IVUsers and
        // any other maps purge the original user from their records.
        PHINode *NewPN = cast<PHINode>(PN->clone());
        NewPN->takeName(PN);
        NewPN->insertBefore(PN);
        PN->replaceAllUsesWith(NewPN);
        PN->eraseFromParent();
      }
    }
  }

  // The insertion point instruction may have been deleted; clear it out
  // so that the rewriter doesn't trip over it later.
  Rewriter.clearInsertPoint();
}

void IndVarSimplify::RewriteNonIntegerIVs(Loop *L) {
  // First step.  Check to see if there are any floating-point recurrences.
  // If there are, change them into integer recurrences, permitting analysis by
  // the SCEV routines.
  //
  BasicBlock *Header    = L->getHeader();

  SmallVector<WeakVH, 8> PHIs;
  for (BasicBlock::iterator I = Header->begin();
       PHINode *PN = dyn_cast<PHINode>(I); ++I)
    PHIs.push_back(PN);

  for (unsigned i = 0, e = PHIs.size(); i != e; ++i)
    if (PHINode *PN = dyn_cast_or_null<PHINode>(PHIs[i]))
      HandleFloatingPointIV(L, PN);

  // If the loop previously had floating-point IV, ScalarEvolution
  // may not have been able to compute a trip count. Now that we've done some
  // re-writing, the trip count may be computable.
  if (Changed)
    SE->forgetLoop(L);
}

void IndVarSimplify::EliminateIVComparisons() {
  SmallVector<WeakVH, 16> DeadInsts;

  // Look for ICmp users.
  for (IVUsers::iterator I = IU->begin(), E = IU->end(); I != E; ++I) {
    IVStrideUse &UI = *I;
    ICmpInst *ICmp = dyn_cast<ICmpInst>(UI.getUser());
    if (!ICmp) continue;

    bool Swapped = UI.getOperandValToReplace() == ICmp->getOperand(1);
    ICmpInst::Predicate Pred = ICmp->getPredicate();
    if (Swapped) Pred = ICmpInst::getSwappedPredicate(Pred);

    // Get the SCEVs for the ICmp operands.
    const SCEV *S = IU->getReplacementExpr(UI);
    const SCEV *X = SE->getSCEV(ICmp->getOperand(!Swapped));

    // Simplify unnecessary loops away.
    const Loop *ICmpLoop = LI->getLoopFor(ICmp->getParent());
    S = SE->getSCEVAtScope(S, ICmpLoop);
    X = SE->getSCEVAtScope(X, ICmpLoop);

    // If the condition is always true or always false, replace it with
    // a constant value.
    if (SE->isKnownPredicate(Pred, S, X))
      ICmp->replaceAllUsesWith(ConstantInt::getTrue(ICmp->getContext()));
    else if (SE->isKnownPredicate(ICmpInst::getInversePredicate(Pred), S, X))
      ICmp->replaceAllUsesWith(ConstantInt::getFalse(ICmp->getContext()));
    else
      continue;

    DEBUG(dbgs() << "INDVARS: Eliminated comparison: " << *ICmp << '\n');
    DeadInsts.push_back(ICmp);
  }

  // Now that we're done iterating through lists, clean up any instructions
  // which are now dead.
  while (!DeadInsts.empty())
    if (Instruction *Inst =
          dyn_cast_or_null<Instruction>(DeadInsts.pop_back_val()))
      RecursivelyDeleteTriviallyDeadInstructions(Inst);
}

void IndVarSimplify::EliminateIVRemainders() {
  SmallVector<WeakVH, 16> DeadInsts;

  // Look for SRem and URem users.
  for (IVUsers::iterator I = IU->begin(), E = IU->end(); I != E; ++I) {
    IVStrideUse &UI = *I;
    BinaryOperator *Rem = dyn_cast<BinaryOperator>(UI.getUser());
    if (!Rem) continue;

    bool isSigned = Rem->getOpcode() == Instruction::SRem;
    if (!isSigned && Rem->getOpcode() != Instruction::URem)
      continue;

    // We're only interested in the case where we know something about
    // the numerator.
    if (UI.getOperandValToReplace() != Rem->getOperand(0))
      continue;

    // Get the SCEVs for the ICmp operands.
    const SCEV *S = SE->getSCEV(Rem->getOperand(0));
    const SCEV *X = SE->getSCEV(Rem->getOperand(1));

    // Simplify unnecessary loops away.
    const Loop *ICmpLoop = LI->getLoopFor(Rem->getParent());
    S = SE->getSCEVAtScope(S, ICmpLoop);
    X = SE->getSCEVAtScope(X, ICmpLoop);

    // i % n  -->  i  if i is in [0,n).
    if ((!isSigned || SE->isKnownNonNegative(S)) &&
        SE->isKnownPredicate(isSigned ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT,
                             S, X))
      Rem->replaceAllUsesWith(Rem->getOperand(0));
    else {
      // (i+1) % n  -->  (i+1)==n?0:(i+1)  if i is in [0,n).
      const SCEV *LessOne =
        SE->getMinusSCEV(S, SE->getConstant(S->getType(), 1));
      if ((!isSigned || SE->isKnownNonNegative(LessOne)) &&
          SE->isKnownPredicate(isSigned ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT,
                               LessOne, X)) {
        ICmpInst *ICmp = new ICmpInst(Rem, ICmpInst::ICMP_EQ,
                                      Rem->getOperand(0), Rem->getOperand(1),
                                      "tmp");
        SelectInst *Sel =
          SelectInst::Create(ICmp,
                             ConstantInt::get(Rem->getType(), 0),
                             Rem->getOperand(0), "tmp", Rem);
        Rem->replaceAllUsesWith(Sel);
      } else
        continue;
    }

    // Inform IVUsers about the new users.
    if (Instruction *I = dyn_cast<Instruction>(Rem->getOperand(0)))
      IU->AddUsersIfInteresting(I);

    DEBUG(dbgs() << "INDVARS: Simplified rem: " << *Rem << '\n');
    DeadInsts.push_back(Rem);
  }

  // Now that we're done iterating through lists, clean up any instructions
  // which are now dead.
  while (!DeadInsts.empty())
    if (Instruction *Inst =
          dyn_cast_or_null<Instruction>(DeadInsts.pop_back_val()))
      RecursivelyDeleteTriviallyDeadInstructions(Inst);
}

bool IndVarSimplify::runOnLoop(Loop *L, LPPassManager &LPM) {
  IU = &getAnalysis<IVUsers>();
  LI = &getAnalysis<LoopInfo>();
  SE = &getAnalysis<ScalarEvolution>();
  DT = &getAnalysis<DominatorTree>();
  Changed = false;

  // If there are any floating-point recurrences, attempt to
  // transform them to use integer recurrences.
  RewriteNonIntegerIVs(L);

  BasicBlock *ExitingBlock = L->getExitingBlock(); // may be null
  const SCEV *BackedgeTakenCount = SE->getBackedgeTakenCount(L);

  // Create a rewriter object which we'll use to transform the code with.
  SCEVExpander Rewriter(*SE);

  // Check to see if this loop has a computable loop-invariant execution count.
  // If so, this means that we can compute the final value of any expressions
  // that are recurrent in the loop, and substitute the exit values from the
  // loop into any instructions outside of the loop that use the final values of
  // the current expressions.
  //
  if (!isa<SCEVCouldNotCompute>(BackedgeTakenCount))
    RewriteLoopExitValues(L, Rewriter);

  // Simplify ICmp IV users.
  EliminateIVComparisons();

  // Simplify SRem and URem IV users.
  EliminateIVRemainders();

  // Compute the type of the largest recurrence expression, and decide whether
  // a canonical induction variable should be inserted.
  const Type *LargestType = 0;
  bool NeedCannIV = false;
  if (!isa<SCEVCouldNotCompute>(BackedgeTakenCount)) {
    LargestType = BackedgeTakenCount->getType();
    LargestType = SE->getEffectiveSCEVType(LargestType);
    // If we have a known trip count and a single exit block, we'll be
    // rewriting the loop exit test condition below, which requires a
    // canonical induction variable.
    if (ExitingBlock)
      NeedCannIV = true;
  }
  for (IVUsers::const_iterator I = IU->begin(), E = IU->end(); I != E; ++I) {
    const Type *Ty =
      SE->getEffectiveSCEVType(I->getOperandValToReplace()->getType());
    if (!LargestType ||
        SE->getTypeSizeInBits(Ty) >
          SE->getTypeSizeInBits(LargestType))
      LargestType = Ty;
    NeedCannIV = true;
  }

  // Now that we know the largest of the induction variable expressions
  // in this loop, insert a canonical induction variable of the largest size.
  Value *IndVar = 0;
  if (NeedCannIV) {
    // Check to see if the loop already has any canonical-looking induction
    // variables. If any are present and wider than the planned canonical
    // induction variable, temporarily remove them, so that the Rewriter
    // doesn't attempt to reuse them.
    SmallVector<PHINode *, 2> OldCannIVs;
    while (PHINode *OldCannIV = L->getCanonicalInductionVariable()) {
      if (SE->getTypeSizeInBits(OldCannIV->getType()) >
          SE->getTypeSizeInBits(LargestType))
        OldCannIV->removeFromParent();
      else
        break;
      OldCannIVs.push_back(OldCannIV);
    }

    IndVar = Rewriter.getOrInsertCanonicalInductionVariable(L, LargestType);

    ++NumInserted;
    Changed = true;
    DEBUG(dbgs() << "INDVARS: New CanIV: " << *IndVar << '\n');

    // Now that the official induction variable is established, reinsert
    // any old canonical-looking variables after it so that the IR remains
    // consistent. They will be deleted as part of the dead-PHI deletion at
    // the end of the pass.
    while (!OldCannIVs.empty()) {
      PHINode *OldCannIV = OldCannIVs.pop_back_val();
      OldCannIV->insertBefore(L->getHeader()->getFirstNonPHI());
    }
  }

  // If we have a trip count expression, rewrite the loop's exit condition
  // using it.  We can currently only handle loops with a single exit.
  ICmpInst *NewICmp = 0;
  if (!isa<SCEVCouldNotCompute>(BackedgeTakenCount) &&
      !BackedgeTakenCount->isZero() &&
      ExitingBlock) {
    assert(NeedCannIV &&
           "LinearFunctionTestReplace requires a canonical induction variable");
    // Can't rewrite non-branch yet.
    if (BranchInst *BI = dyn_cast<BranchInst>(ExitingBlock->getTerminator()))
      NewICmp = LinearFunctionTestReplace(L, BackedgeTakenCount, IndVar,
                                          ExitingBlock, BI, Rewriter);
  }

  // Rewrite IV-derived expressions. Clears the rewriter cache.
  RewriteIVExpressions(L, Rewriter);

  // The Rewriter may not be used from this point on.

  // Loop-invariant instructions in the preheader that aren't used in the
  // loop may be sunk below the loop to reduce register pressure.
  SinkUnusedInvariants(L);

  // For completeness, inform IVUsers of the IV use in the newly-created
  // loop exit test instruction.
  if (NewICmp)
    IU->AddUsersIfInteresting(cast<Instruction>(NewICmp->getOperand(0)));

  // Clean up dead instructions.
  Changed |= DeleteDeadPHIs(L->getHeader());
  // Check a post-condition.
  assert(L->isLCSSAForm(*DT) && "Indvars did not leave the loop in lcssa form!");
  return Changed;
}

// FIXME: It is an extremely bad idea to indvar substitute anything more
// complex than affine induction variables.  Doing so will put expensive
// polynomial evaluations inside of the loop, and the str reduction pass
// currently can only reduce affine polynomials.  For now just disable
// indvar subst on anything more complex than an affine addrec, unless
// it can be expanded to a trivial value.
static bool isSafe(const SCEV *S, const Loop *L) {
  // Loop-invariant values are safe.
  if (S->isLoopInvariant(L)) return true;

  // Affine addrecs are safe. Non-affine are not, because LSR doesn't know how
  // to transform them into efficient code.
  if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(S))
    return AR->isAffine();

  // An add is safe it all its operands are safe.
  if (const SCEVCommutativeExpr *Commutative = dyn_cast<SCEVCommutativeExpr>(S)) {
    for (SCEVCommutativeExpr::op_iterator I = Commutative->op_begin(),
         E = Commutative->op_end(); I != E; ++I)
      if (!isSafe(*I, L)) return false;
    return true;
  }
  
  // A cast is safe if its operand is.
  if (const SCEVCastExpr *C = dyn_cast<SCEVCastExpr>(S))
    return isSafe(C->getOperand(), L);

  // A udiv is safe if its operands are.
  if (const SCEVUDivExpr *UD = dyn_cast<SCEVUDivExpr>(S))
    return isSafe(UD->getLHS(), L) &&
           isSafe(UD->getRHS(), L);

  // SCEVUnknown is always safe.
  if (isa<SCEVUnknown>(S))
    return true;

  // Nothing else is safe.
  return false;
}

void IndVarSimplify::RewriteIVExpressions(Loop *L, SCEVExpander &Rewriter) {
  SmallVector<WeakVH, 16> DeadInsts;

  // Rewrite all induction variable expressions in terms of the canonical
  // induction variable.
  //
  // If there were induction variables of other sizes or offsets, manually
  // add the offsets to the primary induction variable and cast, avoiding
  // the need for the code evaluation methods to insert induction variables
  // of different sizes.
  for (IVUsers::iterator UI = IU->begin(), E = IU->end(); UI != E; ++UI) {
    Value *Op = UI->getOperandValToReplace();
    const Type *UseTy = Op->getType();
    Instruction *User = UI->getUser();

    // Compute the final addrec to expand into code.
    const SCEV *AR = IU->getReplacementExpr(*UI);

    // Evaluate the expression out of the loop, if possible.
    if (!L->contains(UI->getUser())) {
      const SCEV *ExitVal = SE->getSCEVAtScope(AR, L->getParentLoop());
      if (ExitVal->isLoopInvariant(L))
        AR = ExitVal;
    }

    // FIXME: It is an extremely bad idea to indvar substitute anything more
    // complex than affine induction variables.  Doing so will put expensive
    // polynomial evaluations inside of the loop, and the str reduction pass
    // currently can only reduce affine polynomials.  For now just disable
    // indvar subst on anything more complex than an affine addrec, unless
    // it can be expanded to a trivial value.
    if (!isSafe(AR, L))
      continue;

    // Determine the insertion point for this user. By default, insert
    // immediately before the user. The SCEVExpander class will automatically
    // hoist loop invariants out of the loop. For PHI nodes, there may be
    // multiple uses, so compute the nearest common dominator for the
    // incoming blocks.
    Instruction *InsertPt = User;
    if (PHINode *PHI = dyn_cast<PHINode>(InsertPt))
      for (unsigned i = 0, e = PHI->getNumIncomingValues(); i != e; ++i)
        if (PHI->getIncomingValue(i) == Op) {
          if (InsertPt == User)
            InsertPt = PHI->getIncomingBlock(i)->getTerminator();
          else
            InsertPt =
              DT->findNearestCommonDominator(InsertPt->getParent(),
                                             PHI->getIncomingBlock(i))
                    ->getTerminator();
        }

    // Now expand it into actual Instructions and patch it into place.
    Value *NewVal = Rewriter.expandCodeFor(AR, UseTy, InsertPt);

    // Inform ScalarEvolution that this value is changing. The change doesn't
    // affect its value, but it does potentially affect which use lists the
    // value will be on after the replacement, which affects ScalarEvolution's
    // ability to walk use lists and drop dangling pointers when a value is
    // deleted.
    SE->forgetValue(User);

    // Patch the new value into place.
    if (Op->hasName())
      NewVal->takeName(Op);
    User->replaceUsesOfWith(Op, NewVal);
    UI->setOperandValToReplace(NewVal);
    DEBUG(dbgs() << "INDVARS: Rewrote IV '" << *AR << "' " << *Op << '\n'
                 << "   into = " << *NewVal << "\n");
    ++NumRemoved;
    Changed = true;

    // The old value may be dead now.
    DeadInsts.push_back(Op);
  }

  // Clear the rewriter cache, because values that are in the rewriter's cache
  // can be deleted in the loop below, causing the AssertingVH in the cache to
  // trigger.
  Rewriter.clear();
  // Now that we're done iterating through lists, clean up any instructions
  // which are now dead.
  while (!DeadInsts.empty())
    if (Instruction *Inst =
          dyn_cast_or_null<Instruction>(DeadInsts.pop_back_val()))
      RecursivelyDeleteTriviallyDeadInstructions(Inst);
}

/// If there's a single exit block, sink any loop-invariant values that
/// were defined in the preheader but not used inside the loop into the
/// exit block to reduce register pressure in the loop.
void IndVarSimplify::SinkUnusedInvariants(Loop *L) {
  BasicBlock *ExitBlock = L->getExitBlock();
  if (!ExitBlock) return;

  BasicBlock *Preheader = L->getLoopPreheader();
  if (!Preheader) return;

  Instruction *InsertPt = ExitBlock->getFirstNonPHI();
  BasicBlock::iterator I = Preheader->getTerminator();
  while (I != Preheader->begin()) {
    --I;
    // New instructions were inserted at the end of the preheader.
    if (isa<PHINode>(I))
      break;

    // Don't move instructions which might have side effects, since the side
    // effects need to complete before instructions inside the loop.  Also don't
    // move instructions which might read memory, since the loop may modify
    // memory. Note that it's okay if the instruction might have undefined
    // behavior: LoopSimplify guarantees that the preheader dominates the exit
    // block.
    if (I->mayHaveSideEffects() || I->mayReadFromMemory())
      continue;

    // Skip debug info intrinsics.
    if (isa<DbgInfoIntrinsic>(I))
      continue;

    // Don't sink static AllocaInsts out of the entry block, which would
    // turn them into dynamic allocas!
    if (AllocaInst *AI = dyn_cast<AllocaInst>(I))
      if (AI->isStaticAlloca())
        continue;

    // Determine if there is a use in or before the loop (direct or
    // otherwise).
    bool UsedInLoop = false;
    for (Value::use_iterator UI = I->use_begin(), UE = I->use_end();
         UI != UE; ++UI) {
      BasicBlock *UseBB = cast<Instruction>(UI)->getParent();
      if (PHINode *P = dyn_cast<PHINode>(UI)) {
        unsigned i =
          PHINode::getIncomingValueNumForOperand(UI.getOperandNo());
        UseBB = P->getIncomingBlock(i);
      }
      if (UseBB == Preheader || L->contains(UseBB)) {
        UsedInLoop = true;
        break;
      }
    }

    // If there is, the def must remain in the preheader.
    if (UsedInLoop)
      continue;

    // Otherwise, sink it to the exit block.
    Instruction *ToMove = I;
    bool Done = false;

    if (I != Preheader->begin()) {
      // Skip debug info intrinsics.
      do {
        --I;
      } while (isa<DbgInfoIntrinsic>(I) && I != Preheader->begin());

      if (isa<DbgInfoIntrinsic>(I) && I == Preheader->begin())
        Done = true;
    } else {
      Done = true;
    }

    ToMove->moveBefore(InsertPt);
    if (Done) break;
    InsertPt = ToMove;
  }
}

/// ConvertToSInt - Convert APF to an integer, if possible.
static bool ConvertToSInt(const APFloat &APF, int64_t &IntVal) {
  bool isExact = false;
  if (&APF.getSemantics() == &APFloat::PPCDoubleDouble)
    return false;
  // See if we can convert this to an int64_t
  uint64_t UIntVal;
  if (APF.convertToInteger(&UIntVal, 64, true, APFloat::rmTowardZero,
                           &isExact) != APFloat::opOK || !isExact)
    return false;
  IntVal = UIntVal;
  return true;
}

/// HandleFloatingPointIV - If the loop has floating induction variable
/// then insert corresponding integer induction variable if possible.
/// For example,
/// for(double i = 0; i < 10000; ++i)
///   bar(i)
/// is converted into
/// for(int i = 0; i < 10000; ++i)
///   bar((double)i);
///
void IndVarSimplify::HandleFloatingPointIV(Loop *L, PHINode *PN) {
  unsigned IncomingEdge = L->contains(PN->getIncomingBlock(0));
  unsigned BackEdge     = IncomingEdge^1;

  // Check incoming value.
  ConstantFP *InitValueVal =
    dyn_cast<ConstantFP>(PN->getIncomingValue(IncomingEdge));

  int64_t InitValue;
  if (!InitValueVal || !ConvertToSInt(InitValueVal->getValueAPF(), InitValue))
    return;

  // Check IV increment. Reject this PN if increment operation is not
  // an add or increment value can not be represented by an integer.
  BinaryOperator *Incr =
    dyn_cast<BinaryOperator>(PN->getIncomingValue(BackEdge));
  if (Incr == 0 || Incr->getOpcode() != Instruction::FAdd) return;
  
  // If this is not an add of the PHI with a constantfp, or if the constant fp
  // is not an integer, bail out.
  ConstantFP *IncValueVal = dyn_cast<ConstantFP>(Incr->getOperand(1));
  int64_t IncValue;
  if (IncValueVal == 0 || Incr->getOperand(0) != PN ||
      !ConvertToSInt(IncValueVal->getValueAPF(), IncValue))
    return;

  // Check Incr uses. One user is PN and the other user is an exit condition
  // used by the conditional terminator.
  Value::use_iterator IncrUse = Incr->use_begin();
  Instruction *U1 = cast<Instruction>(IncrUse++);
  if (IncrUse == Incr->use_end()) return;
  Instruction *U2 = cast<Instruction>(IncrUse++);
  if (IncrUse != Incr->use_end()) return;

  // Find exit condition, which is an fcmp.  If it doesn't exist, or if it isn't
  // only used by a branch, we can't transform it.
  FCmpInst *Compare = dyn_cast<FCmpInst>(U1);
  if (!Compare)
    Compare = dyn_cast<FCmpInst>(U2);
  if (Compare == 0 || !Compare->hasOneUse() ||
      !isa<BranchInst>(Compare->use_back()))
    return;
  
  BranchInst *TheBr = cast<BranchInst>(Compare->use_back());

  // We need to verify that the branch actually controls the iteration count
  // of the loop.  If not, the new IV can overflow and no one will notice.
  // The branch block must be in the loop and one of the successors must be out
  // of the loop.
  assert(TheBr->isConditional() && "Can't use fcmp if not conditional");
  if (!L->contains(TheBr->getParent()) ||
      (L->contains(TheBr->getSuccessor(0)) &&
       L->contains(TheBr->getSuccessor(1))))
    return;
  
  
  // If it isn't a comparison with an integer-as-fp (the exit value), we can't
  // transform it.
  ConstantFP *ExitValueVal = dyn_cast<ConstantFP>(Compare->getOperand(1));
  int64_t ExitValue;
  if (ExitValueVal == 0 ||
      !ConvertToSInt(ExitValueVal->getValueAPF(), ExitValue))
    return;
  
  // Find new predicate for integer comparison.
  CmpInst::Predicate NewPred = CmpInst::BAD_ICMP_PREDICATE;
  switch (Compare->getPredicate()) {
  default: return;  // Unknown comparison.
  case CmpInst::FCMP_OEQ:
  case CmpInst::FCMP_UEQ: NewPred = CmpInst::ICMP_EQ; break;
  case CmpInst::FCMP_ONE:
  case CmpInst::FCMP_UNE: NewPred = CmpInst::ICMP_NE; break;
  case CmpInst::FCMP_OGT:
  case CmpInst::FCMP_UGT: NewPred = CmpInst::ICMP_SGT; break;
  case CmpInst::FCMP_OGE:
  case CmpInst::FCMP_UGE: NewPred = CmpInst::ICMP_SGE; break;
  case CmpInst::FCMP_OLT:
  case CmpInst::FCMP_ULT: NewPred = CmpInst::ICMP_SLT; break;
  case CmpInst::FCMP_OLE:
  case CmpInst::FCMP_ULE: NewPred = CmpInst::ICMP_SLE; break;
  }
  
  // We convert the floating point induction variable to a signed i32 value if
  // we can.  This is only safe if the comparison will not overflow in a way
  // that won't be trapped by the integer equivalent operations.  Check for this
  // now.
  // TODO: We could use i64 if it is native and the range requires it.
  
  // The start/stride/exit values must all fit in signed i32.
  if (!isInt<32>(InitValue) || !isInt<32>(IncValue) || !isInt<32>(ExitValue))
    return;

  // If not actually striding (add x, 0.0), avoid touching the code.
  if (IncValue == 0)
    return;

  // Positive and negative strides have different safety conditions.
  if (IncValue > 0) {
    // If we have a positive stride, we require the init to be less than the
    // exit value and an equality or less than comparison.
    if (InitValue >= ExitValue ||
        NewPred == CmpInst::ICMP_SGT || NewPred == CmpInst::ICMP_SGE)
      return;
    
    uint32_t Range = uint32_t(ExitValue-InitValue);
    if (NewPred == CmpInst::ICMP_SLE) {
      // Normalize SLE -> SLT, check for infinite loop.
      if (++Range == 0) return;  // Range overflows.
    }
    
    unsigned Leftover = Range % uint32_t(IncValue);
    
    // If this is an equality comparison, we require that the strided value
    // exactly land on the exit value, otherwise the IV condition will wrap
    // around and do things the fp IV wouldn't.
    if ((NewPred == CmpInst::ICMP_EQ || NewPred == CmpInst::ICMP_NE) &&
        Leftover != 0)
      return;
    
    // If the stride would wrap around the i32 before exiting, we can't
    // transform the IV.
    if (Leftover != 0 && int32_t(ExitValue+IncValue) < ExitValue)
      return;
    
  } else {
    // If we have a negative stride, we require the init to be greater than the
    // exit value and an equality or greater than comparison.
    if (InitValue >= ExitValue ||
        NewPred == CmpInst::ICMP_SLT || NewPred == CmpInst::ICMP_SLE)
      return;
    
    uint32_t Range = uint32_t(InitValue-ExitValue);
    if (NewPred == CmpInst::ICMP_SGE) {
      // Normalize SGE -> SGT, check for infinite loop.
      if (++Range == 0) return;  // Range overflows.
    }
    
    unsigned Leftover = Range % uint32_t(-IncValue);
    
    // If this is an equality comparison, we require that the strided value
    // exactly land on the exit value, otherwise the IV condition will wrap
    // around and do things the fp IV wouldn't.
    if ((NewPred == CmpInst::ICMP_EQ || NewPred == CmpInst::ICMP_NE) &&
        Leftover != 0)
      return;
    
    // If the stride would wrap around the i32 before exiting, we can't
    // transform the IV.
    if (Leftover != 0 && int32_t(ExitValue+IncValue) > ExitValue)
      return;
  }
  
  const IntegerType *Int32Ty = Type::getInt32Ty(PN->getContext());

  // Insert new integer induction variable.
  PHINode *NewPHI = PHINode::Create(Int32Ty, PN->getName()+".int", PN);
  NewPHI->addIncoming(ConstantInt::get(Int32Ty, InitValue),
                      PN->getIncomingBlock(IncomingEdge));

  Value *NewAdd =
    BinaryOperator::CreateAdd(NewPHI, ConstantInt::get(Int32Ty, IncValue),
                              Incr->getName()+".int", Incr);
  NewPHI->addIncoming(NewAdd, PN->getIncomingBlock(BackEdge));

  ICmpInst *NewCompare = new ICmpInst(TheBr, NewPred, NewAdd,
                                      ConstantInt::get(Int32Ty, ExitValue),
                                      Compare->getName());

  // In the following deletions, PN may become dead and may be deleted.
  // Use a WeakVH to observe whether this happens.
  WeakVH WeakPH = PN;

  // Delete the old floating point exit comparison.  The branch starts using the
  // new comparison.
  NewCompare->takeName(Compare);
  Compare->replaceAllUsesWith(NewCompare);
  RecursivelyDeleteTriviallyDeadInstructions(Compare);

  // Delete the old floating point increment.
  Incr->replaceAllUsesWith(UndefValue::get(Incr->getType()));
  RecursivelyDeleteTriviallyDeadInstructions(Incr);

  // If the FP induction variable still has uses, this is because something else
  // in the loop uses its value.  In order to canonicalize the induction
  // variable, we chose to eliminate the IV and rewrite it in terms of an
  // int->fp cast.
  //
  // We give preference to sitofp over uitofp because it is faster on most
  // platforms.
  if (WeakPH) {
    Value *Conv = new SIToFPInst(NewPHI, PN->getType(), "indvar.conv",
                                 PN->getParent()->getFirstNonPHI());
    PN->replaceAllUsesWith(Conv);
    RecursivelyDeleteTriviallyDeadInstructions(PN);
  }

  // Add a new IVUsers entry for the newly-created integer PHI.
  IU->AddUsersIfInteresting(NewPHI);
}
