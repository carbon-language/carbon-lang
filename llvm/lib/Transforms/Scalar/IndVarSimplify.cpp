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
//   3. Any pointer arithmetic recurrences are raised to use array subscripts.
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
// desired loop transformations have been performed.  Additionally, on targets
// where it is profitable, the loop could be transformed to count down to zero
// (the "do loop" optimization).
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "indvars"
#include "llvm/Transforms/Scalar.h"
#include "llvm/BasicBlock.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Type.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/IVUsers.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
using namespace llvm;

STATISTIC(NumRemoved , "Number of aux indvars removed");
STATISTIC(NumInserted, "Number of canonical indvars added");
STATISTIC(NumReplaced, "Number of exit values replaced");
STATISTIC(NumLFTR    , "Number of loop exit tests replaced");

namespace {
  class VISIBILITY_HIDDEN IndVarSimplify : public LoopPass {
    IVUsers         *IU;
    LoopInfo        *LI;
    ScalarEvolution *SE;
    bool Changed;
  public:

   static char ID; // Pass identification, replacement for typeid
   IndVarSimplify() : LoopPass(&ID) {}

   virtual bool runOnLoop(Loop *L, LPPassManager &LPM);

   virtual void getAnalysisUsage(AnalysisUsage &AU) const {
     AU.addRequired<DominatorTree>();
     AU.addRequired<ScalarEvolution>();
     AU.addRequiredID(LCSSAID);
     AU.addRequiredID(LoopSimplifyID);
     AU.addRequired<LoopInfo>();
     AU.addRequired<IVUsers>();
     AU.addPreserved<ScalarEvolution>();
     AU.addPreservedID(LoopSimplifyID);
     AU.addPreserved<IVUsers>();
     AU.addPreservedID(LCSSAID);
     AU.setPreservesCFG();
   }

  private:

    void RewriteNonIntegerIVs(Loop *L);

    ICmpInst *LinearFunctionTestReplace(Loop *L, SCEVHandle BackedgeTakenCount,
                                   Value *IndVar,
                                   BasicBlock *ExitingBlock,
                                   BranchInst *BI,
                                   SCEVExpander &Rewriter);
    void RewriteLoopExitValues(Loop *L, const SCEV *BackedgeTakenCount);

    void RewriteIVExpressions(Loop *L, const Type *LargestType,
                              SCEVExpander &Rewriter);

    void SinkUnusedInvariants(Loop *L, SCEVExpander &Rewriter);

    void FixUsesBeforeDefs(Loop *L, SCEVExpander &Rewriter);

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
                                   SCEVHandle BackedgeTakenCount,
                                   Value *IndVar,
                                   BasicBlock *ExitingBlock,
                                   BranchInst *BI,
                                   SCEVExpander &Rewriter) {
  // If the exiting block is not the same as the backedge block, we must compare
  // against the preincremented value, otherwise we prefer to compare against
  // the post-incremented value.
  Value *CmpIndVar;
  SCEVHandle RHS = BackedgeTakenCount;
  if (ExitingBlock == L->getLoopLatch()) {
    // Add one to the "backedge-taken" count to get the trip count.
    // If this addition may overflow, we have to be more pessimistic and
    // cast the induction variable before doing the add.
    SCEVHandle Zero = SE->getIntegerSCEV(0, BackedgeTakenCount->getType());
    SCEVHandle N =
      SE->getAddExpr(BackedgeTakenCount,
                     SE->getIntegerSCEV(1, BackedgeTakenCount->getType()));
    if ((isa<SCEVConstant>(N) && !N->isZero()) ||
        SE->isLoopGuardedByCond(L, ICmpInst::ICMP_NE, N, Zero)) {
      // No overflow. Cast the sum.
      RHS = SE->getTruncateOrZeroExtend(N, IndVar->getType());
    } else {
      // Potential overflow. Cast before doing the add.
      RHS = SE->getTruncateOrZeroExtend(BackedgeTakenCount,
                                        IndVar->getType());
      RHS = SE->getAddExpr(RHS,
                           SE->getIntegerSCEV(1, IndVar->getType()));
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

  // Expand the code for the iteration count into the preheader of the loop.
  BasicBlock *Preheader = L->getLoopPreheader();
  Value *ExitCnt = Rewriter.expandCodeFor(RHS, IndVar->getType(),
                                          Preheader->getTerminator());

  // Insert a new icmp_ne or icmp_eq instruction before the branch.
  ICmpInst::Predicate Opcode;
  if (L->contains(BI->getSuccessor(0)))
    Opcode = ICmpInst::ICMP_NE;
  else
    Opcode = ICmpInst::ICMP_EQ;

  DOUT << "INDVARS: Rewriting loop exit condition to:\n"
       << "      LHS:" << *CmpIndVar // includes a newline
       << "       op:\t"
       << (Opcode == ICmpInst::ICMP_NE ? "!=" : "==") << "\n"
       << "      RHS:\t" << *RHS << "\n";

  ICmpInst *Cond = new ICmpInst(Opcode, CmpIndVar, ExitCnt, "exitcond", BI);

  Instruction *OrigCond = cast<Instruction>(BI->getCondition());
  OrigCond->replaceAllUsesWith(Cond);
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
                                           const SCEV *BackedgeTakenCount) {
  // Verify the input to the pass in already in LCSSA form.
  assert(L->isLCSSAForm());

  BasicBlock *Preheader = L->getLoopPreheader();

  // Scan all of the instructions in the loop, looking at those that have
  // extra-loop users and which are recurrences.
  SCEVExpander Rewriter(*SE, *LI);

  // We insert the code into the preheader of the loop if the loop contains
  // multiple exit blocks, or in the exit block if there is exactly one.
  BasicBlock *BlockToInsertInto;
  SmallVector<BasicBlock*, 8> ExitBlocks;
  L->getUniqueExitBlocks(ExitBlocks);
  if (ExitBlocks.size() == 1)
    BlockToInsertInto = ExitBlocks[0];
  else
    BlockToInsertInto = Preheader;
  BasicBlock::iterator InsertPt = BlockToInsertInto->getFirstNonPHI();

  std::map<Instruction*, Value*> ExitValues;

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

      // Iterate over all of the values in all the PHI nodes.
      for (unsigned i = 0; i != NumPreds; ++i) {
        // If the value being merged in is not integer or is not defined
        // in the loop, skip it.
        Value *InVal = PN->getIncomingValue(i);
        if (!isa<Instruction>(InVal) ||
            // SCEV only supports integer expressions for now.
            (!isa<IntegerType>(InVal->getType()) &&
             !isa<PointerType>(InVal->getType())))
          continue;

        // If this pred is for a subloop, not L itself, skip it.
        if (LI->getLoopFor(PN->getIncomingBlock(i)) != L)
          continue; // The Block is in a subloop, skip it.

        // Check that InVal is defined in the loop.
        Instruction *Inst = cast<Instruction>(InVal);
        if (!L->contains(Inst->getParent()))
          continue;

        // Okay, this instruction has a user outside of the current loop
        // and varies predictably *inside* the loop.  Evaluate the value it
        // contains when the loop exits, if possible.
        SCEVHandle SH = SE->getSCEV(Inst);
        SCEVHandle ExitValue = SE->getSCEVAtScope(SH, L->getParentLoop());
        if (isa<SCEVCouldNotCompute>(ExitValue) ||
            !ExitValue->isLoopInvariant(L))
          continue;

        Changed = true;
        ++NumReplaced;

        // See if we already computed the exit value for the instruction, if so,
        // just reuse it.
        Value *&ExitVal = ExitValues[Inst];
        if (!ExitVal)
          ExitVal = Rewriter.expandCodeFor(ExitValue, PN->getType(), InsertPt);

        DOUT << "INDVARS: RLEV: AfterLoopVal = " << *ExitVal
             << "  LoopVal = " << *Inst << "\n";

        PN->setIncomingValue(i, ExitVal);

        // If this instruction is dead now, delete it.
        RecursivelyDeleteTriviallyDeadInstructions(Inst);

        // See if this is a single-entry LCSSA PHI node.  If so, we can (and
        // have to) remove
        // the PHI entirely.  This is safe, because the NewVal won't be variant
        // in the loop, so we don't need an LCSSA phi node anymore.
        if (NumPreds == 1) {
          PN->replaceAllUsesWith(ExitVal);
          RecursivelyDeleteTriviallyDeadInstructions(PN);
          break;
        }
      }
    }
  }
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
    SE->forgetLoopBackedgeTakenCount(L);
}

bool IndVarSimplify::runOnLoop(Loop *L, LPPassManager &LPM) {
  IU = &getAnalysis<IVUsers>();
  LI = &getAnalysis<LoopInfo>();
  SE = &getAnalysis<ScalarEvolution>();
  Changed = false;

  // If there are any floating-point recurrences, attempt to
  // transform them to use integer recurrences.
  RewriteNonIntegerIVs(L);

  BasicBlock *Header       = L->getHeader();
  BasicBlock *ExitingBlock = L->getExitingBlock(); // may be null
  SCEVHandle BackedgeTakenCount = SE->getBackedgeTakenCount(L);

  // Check to see if this loop has a computable loop-invariant execution count.
  // If so, this means that we can compute the final value of any expressions
  // that are recurrent in the loop, and substitute the exit values from the
  // loop into any instructions outside of the loop that use the final values of
  // the current expressions.
  //
  if (!isa<SCEVCouldNotCompute>(BackedgeTakenCount))
    RewriteLoopExitValues(L, BackedgeTakenCount);

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
  for (unsigned i = 0, e = IU->StrideOrder.size(); i != e; ++i) {
    SCEVHandle Stride = IU->StrideOrder[i];
    const Type *Ty = SE->getEffectiveSCEVType(Stride->getType());
    if (!LargestType ||
        SE->getTypeSizeInBits(Ty) >
          SE->getTypeSizeInBits(LargestType))
      LargestType = Ty;

    std::map<SCEVHandle, IVUsersOfOneStride *>::iterator SI =
      IU->IVUsesByStride.find(IU->StrideOrder[i]);
    assert(SI != IU->IVUsesByStride.end() && "Stride doesn't exist!");

    if (!SI->second->Users.empty())
      NeedCannIV = true;
  }

  // Create a rewriter object which we'll use to transform the code with.
  SCEVExpander Rewriter(*SE, *LI);

  // Now that we know the largest of of the induction variable expressions
  // in this loop, insert a canonical induction variable of the largest size.
  Value *IndVar = 0;
  if (NeedCannIV) {
    IndVar = Rewriter.getOrInsertCanonicalInductionVariable(L,LargestType);
    ++NumInserted;
    Changed = true;
    DOUT << "INDVARS: New CanIV: " << *IndVar;
  }

  // If we have a trip count expression, rewrite the loop's exit condition
  // using it.  We can currently only handle loops with a single exit.
  ICmpInst *NewICmp = 0;
  if (!isa<SCEVCouldNotCompute>(BackedgeTakenCount) && ExitingBlock) {
    assert(NeedCannIV &&
           "LinearFunctionTestReplace requires a canonical induction variable");
    // Can't rewrite non-branch yet.
    if (BranchInst *BI = dyn_cast<BranchInst>(ExitingBlock->getTerminator()))
      NewICmp = LinearFunctionTestReplace(L, BackedgeTakenCount, IndVar,
                                          ExitingBlock, BI, Rewriter);
  }

  Rewriter.setInsertionPoint(Header->getFirstNonPHI());

  // Rewrite IV-derived expressions.
  RewriteIVExpressions(L, LargestType, Rewriter);

  // Loop-invariant instructions in the preheader that aren't used in the
  // loop may be sunk below the loop to reduce register pressure.
  SinkUnusedInvariants(L, Rewriter);

  // Reorder instructions to avoid use-before-def conditions.
  FixUsesBeforeDefs(L, Rewriter);

  // For completeness, inform IVUsers of the IV use in the newly-created
  // loop exit test instruction.
  if (NewICmp)
    IU->AddUsersIfInteresting(cast<Instruction>(NewICmp->getOperand(0)));

  // Clean up dead instructions.
  DeleteDeadPHIs(L->getHeader());
  // Check a post-condition.
  assert(L->isLCSSAForm() && "Indvars did not leave the loop in lcssa form!");
  return Changed;
}

void IndVarSimplify::RewriteIVExpressions(Loop *L, const Type *LargestType,
                                          SCEVExpander &Rewriter) {
  SmallVector<WeakVH, 16> DeadInsts;

  // Rewrite all induction variable expressions in terms of the canonical
  // induction variable.
  //
  // If there were induction variables of other sizes or offsets, manually
  // add the offsets to the primary induction variable and cast, avoiding
  // the need for the code evaluation methods to insert induction variables
  // of different sizes.
  for (unsigned i = 0, e = IU->StrideOrder.size(); i != e; ++i) {
    SCEVHandle Stride = IU->StrideOrder[i];

    std::map<SCEVHandle, IVUsersOfOneStride *>::iterator SI =
      IU->IVUsesByStride.find(IU->StrideOrder[i]);
    assert(SI != IU->IVUsesByStride.end() && "Stride doesn't exist!");
    ilist<IVStrideUse> &List = SI->second->Users;
    for (ilist<IVStrideUse>::iterator UI = List.begin(),
         E = List.end(); UI != E; ++UI) {
      SCEVHandle Offset = UI->getOffset();
      Value *Op = UI->getOperandValToReplace();
      Instruction *User = UI->getUser();
      bool isSigned = UI->isSigned();

      // Compute the final addrec to expand into code.
      SCEVHandle AR = IU->getReplacementExpr(*UI);

      // FIXME: It is an extremely bad idea to indvar substitute anything more
      // complex than affine induction variables.  Doing so will put expensive
      // polynomial evaluations inside of the loop, and the str reduction pass
      // currently can only reduce affine polynomials.  For now just disable
      // indvar subst on anything more complex than an affine addrec, unless
      // it can be expanded to a trivial value.
      if (!Stride->isLoopInvariant(L) &&
          !isa<SCEVConstant>(AR) &&
          L->contains(User->getParent()))
        continue;

      Value *NewVal = 0;
      if (AR->isLoopInvariant(L)) {
        BasicBlock::iterator I = Rewriter.getInsertionPoint();
        // Expand loop-invariant values in the loop preheader. They will
        // be sunk to the exit block later, if possible.
          NewVal =
          Rewriter.expandCodeFor(AR, LargestType,
                                 L->getLoopPreheader()->getTerminator());
        Rewriter.setInsertionPoint(I);
        ++NumReplaced;
      } else {
        const Type *IVTy = Offset->getType();
        const Type *UseTy = Op->getType();

        // Promote the Offset and Stride up to the canonical induction
        // variable's bit width.
        SCEVHandle PromotedOffset = Offset;
        SCEVHandle PromotedStride = Stride;
        if (SE->getTypeSizeInBits(IVTy) != SE->getTypeSizeInBits(LargestType)) {
          // It doesn't matter for correctness whether zero or sign extension
          // is used here, since the value is truncated away below, but if the
          // value is signed, sign extension is more likely to be folded.
          if (isSigned) {
            PromotedOffset = SE->getSignExtendExpr(PromotedOffset, LargestType);
            PromotedStride = SE->getSignExtendExpr(PromotedStride, LargestType);
          } else {
            PromotedOffset = SE->getZeroExtendExpr(PromotedOffset, LargestType);
            // If the stride is obviously negative, use sign extension to
            // produce things like x-1 instead of x+255.
            if (isa<SCEVConstant>(PromotedStride) &&
                cast<SCEVConstant>(PromotedStride)
                  ->getValue()->getValue().isNegative())
              PromotedStride = SE->getSignExtendExpr(PromotedStride,
                                                     LargestType);
            else
              PromotedStride = SE->getZeroExtendExpr(PromotedStride,
                                                     LargestType);
          }
        }

        // Create the SCEV representing the offset from the canonical
        // induction variable, still in the canonical induction variable's
        // type, so that all expanded arithmetic is done in the same type.
        SCEVHandle NewAR = SE->getAddRecExpr(SE->getIntegerSCEV(0, LargestType),
                                           PromotedStride, L);
        // Add the PromotedOffset as a separate step, because it may not be
        // loop-invariant.
        NewAR = SE->getAddExpr(NewAR, PromotedOffset);

        // Expand the addrec into instructions.
        Value *V = Rewriter.expandCodeFor(NewAR, LargestType);

        // Insert an explicit cast if necessary to truncate the value
        // down to the original stride type. This is done outside of
        // SCEVExpander because in SCEV expressions, a truncate of an
        // addrec is always folded.
        if (LargestType != IVTy) {
          if (SE->getTypeSizeInBits(IVTy) != SE->getTypeSizeInBits(LargestType))
            NewAR = SE->getTruncateExpr(NewAR, IVTy);
          if (Rewriter.isInsertedExpression(NewAR))
            V = Rewriter.expandCodeFor(NewAR, IVTy);
          else {
            V = Rewriter.InsertCastOfTo(CastInst::getCastOpcode(V, false,
                                                                IVTy, false),
                                        V, IVTy);
            assert(!isa<SExtInst>(V) && !isa<ZExtInst>(V) &&
                   "LargestType wasn't actually the largest type!");
            // Force the rewriter to use this trunc whenever this addrec
            // appears so that it doesn't insert new phi nodes or
            // arithmetic in a different type.
            Rewriter.addInsertedValue(V, NewAR);
          }
        }

        DOUT << "INDVARS: Made offset-and-trunc IV for offset "
             << *IVTy << " " << *Offset << ": ";
        DEBUG(WriteAsOperand(*DOUT, V, false));
        DOUT << "\n";

        // Now expand it into actual Instructions and patch it into place.
        NewVal = Rewriter.expandCodeFor(AR, UseTy);
      }

      // Patch the new value into place.
      if (Op->hasName())
        NewVal->takeName(Op);
      User->replaceUsesOfWith(Op, NewVal);
      UI->setOperandValToReplace(NewVal);
      DOUT << "INDVARS: Rewrote IV '" << *AR << "' " << *Op
           << "   into = " << *NewVal << "\n";
      ++NumRemoved;
      Changed = true;

      // The old value may be dead now.
      DeadInsts.push_back(Op);
    }
  }

  // Now that we're done iterating through lists, clean up any instructions
  // which are now dead.
  while (!DeadInsts.empty()) {
    Instruction *Inst = dyn_cast_or_null<Instruction>(DeadInsts.pop_back_val());
    if (Inst)
      RecursivelyDeleteTriviallyDeadInstructions(Inst);
  }
}

/// If there's a single exit block, sink any loop-invariant values that
/// were defined in the preheader but not used inside the loop into the
/// exit block to reduce register pressure in the loop.
void IndVarSimplify::SinkUnusedInvariants(Loop *L, SCEVExpander &Rewriter) {
  BasicBlock *ExitBlock = L->getExitBlock();
  if (!ExitBlock) return;

  Instruction *NonPHI = ExitBlock->getFirstNonPHI();
  BasicBlock *Preheader = L->getLoopPreheader();
  BasicBlock::iterator I = Preheader->getTerminator();
  while (I != Preheader->begin()) {
    --I;
    // New instructions were inserted at the end of the preheader. Only
    // consider those new instructions.
    if (!Rewriter.isInsertedInstruction(I))
      break;
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
    if (I != Preheader->begin())
      --I;
    else
      Done = true;
    ToMove->moveBefore(NonPHI);
    if (Done)
      break;
  }
}

/// Re-schedule the inserted instructions to put defs before uses. This
/// fixes problems that arrise when SCEV expressions contain loop-variant
/// values unrelated to the induction variable which are defined inside the
/// loop. FIXME: It would be better to insert instructions in the right
/// place so that this step isn't needed.
void IndVarSimplify::FixUsesBeforeDefs(Loop *L, SCEVExpander &Rewriter) {
  // Visit all the blocks in the loop in pre-order dom-tree dfs order.
  DominatorTree *DT = &getAnalysis<DominatorTree>();
  std::map<Instruction *, unsigned> NumPredsLeft;
  SmallVector<DomTreeNode *, 16> Worklist;
  Worklist.push_back(DT->getNode(L->getHeader()));
  do {
    DomTreeNode *Node = Worklist.pop_back_val();
    for (DomTreeNode::iterator I = Node->begin(), E = Node->end(); I != E; ++I)
      if (L->contains((*I)->getBlock()))
        Worklist.push_back(*I);
    BasicBlock *BB = Node->getBlock();
    // Visit all the instructions in the block top down.
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
      // Count the number of operands that aren't properly dominating.
      unsigned NumPreds = 0;
      if (Rewriter.isInsertedInstruction(I) && !isa<PHINode>(I))
        for (User::op_iterator OI = I->op_begin(), OE = I->op_end();
             OI != OE; ++OI)
          if (Instruction *Inst = dyn_cast<Instruction>(OI))
            if (L->contains(Inst->getParent()) && !NumPredsLeft.count(Inst))
              ++NumPreds;
      NumPredsLeft[I] = NumPreds;
      // Notify uses of the position of this instruction, and move the
      // users (and their dependents, recursively) into place after this
      // instruction if it is their last outstanding operand.
      for (Value::use_iterator UI = I->use_begin(), UE = I->use_end();
           UI != UE; ++UI) {
        Instruction *Inst = cast<Instruction>(UI);
        std::map<Instruction *, unsigned>::iterator Z = NumPredsLeft.find(Inst);
        if (Z != NumPredsLeft.end() && Z->second != 0 && --Z->second == 0) {
          SmallVector<Instruction *, 4> UseWorkList;
          UseWorkList.push_back(Inst);
          BasicBlock::iterator InsertPt = next(I);
          while (isa<PHINode>(InsertPt)) ++InsertPt;
          do {
            Instruction *Use = UseWorkList.pop_back_val();
            Use->moveBefore(InsertPt);
            NumPredsLeft.erase(Use);
            for (Value::use_iterator IUI = Use->use_begin(),
                 IUE = Use->use_end(); IUI != IUE; ++IUI) {
              Instruction *IUIInst = cast<Instruction>(IUI);
              if (L->contains(IUIInst->getParent()) &&
                  Rewriter.isInsertedInstruction(IUIInst) &&
                  !isa<PHINode>(IUIInst))
                UseWorkList.push_back(IUIInst);
            }
          } while (!UseWorkList.empty());
        }
      }
    }
  } while (!Worklist.empty());
}

/// Return true if it is OK to use SIToFPInst for an inducation variable
/// with given inital and exit values.
static bool useSIToFPInst(ConstantFP &InitV, ConstantFP &ExitV,
                          uint64_t intIV, uint64_t intEV) {

  if (InitV.getValueAPF().isNegative() || ExitV.getValueAPF().isNegative())
    return true;

  // If the iteration range can be handled by SIToFPInst then use it.
  APInt Max = APInt::getSignedMaxValue(32);
  if (Max.getZExtValue() > static_cast<uint64_t>(abs64(intEV - intIV)))
    return true;

  return false;
}

/// convertToInt - Convert APF to an integer, if possible.
static bool convertToInt(const APFloat &APF, uint64_t *intVal) {

  bool isExact = false;
  if (&APF.getSemantics() == &APFloat::PPCDoubleDouble)
    return false;
  if (APF.convertToInteger(intVal, 32, APF.isNegative(),
                           APFloat::rmTowardZero, &isExact)
      != APFloat::opOK)
    return false;
  if (!isExact)
    return false;
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
void IndVarSimplify::HandleFloatingPointIV(Loop *L, PHINode *PH) {

  unsigned IncomingEdge = L->contains(PH->getIncomingBlock(0));
  unsigned BackEdge     = IncomingEdge^1;

  // Check incoming value.
  ConstantFP *InitValue = dyn_cast<ConstantFP>(PH->getIncomingValue(IncomingEdge));
  if (!InitValue) return;
  uint64_t newInitValue = Type::Int32Ty->getPrimitiveSizeInBits();
  if (!convertToInt(InitValue->getValueAPF(), &newInitValue))
    return;

  // Check IV increment. Reject this PH if increement operation is not
  // an add or increment value can not be represented by an integer.
  BinaryOperator *Incr =
    dyn_cast<BinaryOperator>(PH->getIncomingValue(BackEdge));
  if (!Incr) return;
  if (Incr->getOpcode() != Instruction::Add) return;
  ConstantFP *IncrValue = NULL;
  unsigned IncrVIndex = 1;
  if (Incr->getOperand(1) == PH)
    IncrVIndex = 0;
  IncrValue = dyn_cast<ConstantFP>(Incr->getOperand(IncrVIndex));
  if (!IncrValue) return;
  uint64_t newIncrValue = Type::Int32Ty->getPrimitiveSizeInBits();
  if (!convertToInt(IncrValue->getValueAPF(), &newIncrValue))
    return;

  // Check Incr uses. One user is PH and the other users is exit condition used
  // by the conditional terminator.
  Value::use_iterator IncrUse = Incr->use_begin();
  Instruction *U1 = cast<Instruction>(IncrUse++);
  if (IncrUse == Incr->use_end()) return;
  Instruction *U2 = cast<Instruction>(IncrUse++);
  if (IncrUse != Incr->use_end()) return;

  // Find exit condition.
  FCmpInst *EC = dyn_cast<FCmpInst>(U1);
  if (!EC)
    EC = dyn_cast<FCmpInst>(U2);
  if (!EC) return;

  if (BranchInst *BI = dyn_cast<BranchInst>(EC->getParent()->getTerminator())) {
    if (!BI->isConditional()) return;
    if (BI->getCondition() != EC) return;
  }

  // Find exit value. If exit value can not be represented as an interger then
  // do not handle this floating point PH.
  ConstantFP *EV = NULL;
  unsigned EVIndex = 1;
  if (EC->getOperand(1) == Incr)
    EVIndex = 0;
  EV = dyn_cast<ConstantFP>(EC->getOperand(EVIndex));
  if (!EV) return;
  uint64_t intEV = Type::Int32Ty->getPrimitiveSizeInBits();
  if (!convertToInt(EV->getValueAPF(), &intEV))
    return;

  // Find new predicate for integer comparison.
  CmpInst::Predicate NewPred = CmpInst::BAD_ICMP_PREDICATE;
  switch (EC->getPredicate()) {
  case CmpInst::FCMP_OEQ:
  case CmpInst::FCMP_UEQ:
    NewPred = CmpInst::ICMP_EQ;
    break;
  case CmpInst::FCMP_OGT:
  case CmpInst::FCMP_UGT:
    NewPred = CmpInst::ICMP_UGT;
    break;
  case CmpInst::FCMP_OGE:
  case CmpInst::FCMP_UGE:
    NewPred = CmpInst::ICMP_UGE;
    break;
  case CmpInst::FCMP_OLT:
  case CmpInst::FCMP_ULT:
    NewPred = CmpInst::ICMP_ULT;
    break;
  case CmpInst::FCMP_OLE:
  case CmpInst::FCMP_ULE:
    NewPred = CmpInst::ICMP_ULE;
    break;
  default:
    break;
  }
  if (NewPred == CmpInst::BAD_ICMP_PREDICATE) return;

  // Insert new integer induction variable.
  PHINode *NewPHI = PHINode::Create(Type::Int32Ty,
                                    PH->getName()+".int", PH);
  NewPHI->addIncoming(ConstantInt::get(Type::Int32Ty, newInitValue),
                      PH->getIncomingBlock(IncomingEdge));

  Value *NewAdd = BinaryOperator::CreateAdd(NewPHI,
                                            ConstantInt::get(Type::Int32Ty,
                                                             newIncrValue),
                                            Incr->getName()+".int", Incr);
  NewPHI->addIncoming(NewAdd, PH->getIncomingBlock(BackEdge));

  // The back edge is edge 1 of newPHI, whatever it may have been in the
  // original PHI.
  ConstantInt *NewEV = ConstantInt::get(Type::Int32Ty, intEV);
  Value *LHS = (EVIndex == 1 ? NewPHI->getIncomingValue(1) : NewEV);
  Value *RHS = (EVIndex == 1 ? NewEV : NewPHI->getIncomingValue(1));
  ICmpInst *NewEC = new ICmpInst(NewPred, LHS, RHS, EC->getNameStart(),
                                 EC->getParent()->getTerminator());

  // In the following deltions, PH may become dead and may be deleted.
  // Use a WeakVH to observe whether this happens.
  WeakVH WeakPH = PH;

  // Delete old, floating point, exit comparision instruction.
  EC->replaceAllUsesWith(NewEC);
  RecursivelyDeleteTriviallyDeadInstructions(EC);

  // Delete old, floating point, increment instruction.
  Incr->replaceAllUsesWith(UndefValue::get(Incr->getType()));
  RecursivelyDeleteTriviallyDeadInstructions(Incr);

  // Replace floating induction variable, if it isn't already deleted.
  // Give SIToFPInst preference over UIToFPInst because it is faster on
  // platforms that are widely used.
  if (WeakPH && !PH->use_empty()) {
    if (useSIToFPInst(*InitValue, *EV, newInitValue, intEV)) {
      SIToFPInst *Conv = new SIToFPInst(NewPHI, PH->getType(), "indvar.conv",
                                        PH->getParent()->getFirstNonPHI());
      PH->replaceAllUsesWith(Conv);
    } else {
      UIToFPInst *Conv = new UIToFPInst(NewPHI, PH->getType(), "indvar.conv",
                                        PH->getParent()->getFirstNonPHI());
      PH->replaceAllUsesWith(Conv);
    }
    RecursivelyDeleteTriviallyDeadInstructions(PH);
  }

  // Add a new IVUsers entry for the newly-created integer PHI.
  IU->AddUsersIfInteresting(NewPHI);
}
