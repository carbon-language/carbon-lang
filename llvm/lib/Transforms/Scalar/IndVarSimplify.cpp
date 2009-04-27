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
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumRemoved , "Number of aux indvars removed");
STATISTIC(NumInserted, "Number of canonical indvars added");
STATISTIC(NumReplaced, "Number of exit values replaced");
STATISTIC(NumLFTR    , "Number of loop exit tests replaced");

namespace {
  class VISIBILITY_HIDDEN IndVarSimplify : public LoopPass {
    LoopInfo        *LI;
    ScalarEvolution *SE;
    bool Changed;
  public:

   static char ID; // Pass identification, replacement for typeid
   IndVarSimplify() : LoopPass(&ID) {}

   virtual bool runOnLoop(Loop *L, LPPassManager &LPM);

   virtual void getAnalysisUsage(AnalysisUsage &AU) const {
     AU.addRequired<ScalarEvolution>();
     AU.addRequiredID(LCSSAID);
     AU.addRequiredID(LoopSimplifyID);
     AU.addRequired<LoopInfo>();
     AU.addPreserved<ScalarEvolution>();
     AU.addPreservedID(LoopSimplifyID);
     AU.addPreservedID(LCSSAID);
     AU.setPreservesCFG();
   }

  private:

    void RewriteNonIntegerIVs(Loop *L);

    void LinearFunctionTestReplace(Loop *L, SCEVHandle BackedgeTakenCount,
                                   Value *IndVar,
                                   BasicBlock *ExitingBlock,
                                   BranchInst *BI,
                                   SCEVExpander &Rewriter);
    void RewriteLoopExitValues(Loop *L, const SCEV *BackedgeTakenCount);

    void DeleteTriviallyDeadInstructions(SmallPtrSet<Instruction*, 16> &Insts);

    void HandleFloatingPointIV(Loop *L, PHINode *PH,
                               SmallPtrSet<Instruction*, 16> &DeadInsts);
  };
}

char IndVarSimplify::ID = 0;
static RegisterPass<IndVarSimplify>
X("indvars", "Canonicalize Induction Variables");

Pass *llvm::createIndVarSimplifyPass() {
  return new IndVarSimplify();
}

/// DeleteTriviallyDeadInstructions - If any of the instructions is the
/// specified set are trivially dead, delete them and see if this makes any of
/// their operands subsequently dead.
void IndVarSimplify::
DeleteTriviallyDeadInstructions(SmallPtrSet<Instruction*, 16> &Insts) {
  while (!Insts.empty()) {
    Instruction *I = *Insts.begin();
    Insts.erase(I);
    if (isInstructionTriviallyDead(I)) {
      for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
        if (Instruction *U = dyn_cast<Instruction>(I->getOperand(i)))
          Insts.insert(U);
      SE->deleteValueFromRecords(I);
      DOUT << "INDVARS: Deleting: " << *I;
      I->eraseFromParent();
      Changed = true;
    }
  }
}

/// LinearFunctionTestReplace - This method rewrites the exit condition of the
/// loop to be a canonical != comparison against the incremented loop induction
/// variable.  This pass is able to rewrite the exit tests of any loop where the
/// SCEV analysis can determine a loop-invariant trip count of the loop, which
/// is actually a much broader range than just linear tests.
void IndVarSimplify::LinearFunctionTestReplace(Loop *L,
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

  Value *Cond = new ICmpInst(Opcode, CmpIndVar, ExitCnt, "exitcond", BI);
  BI->setCondition(Cond);
  ++NumLFTR;
  Changed = true;
}

/// RewriteLoopExitValues - Check to see if this loop has a computable
/// loop-invariant execution count.  If so, this means that we can compute the
/// final value of any expressions that are recurrent in the loop, and
/// substitute the exit values from the loop into any instructions outside of
/// the loop that use the final values of the current expressions.
void IndVarSimplify::RewriteLoopExitValues(Loop *L,
                                           const SCEV *BackedgeTakenCount) {
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

  bool HasConstantItCount = isa<SCEVConstant>(BackedgeTakenCount);

  SmallPtrSet<Instruction*, 16> InstructionsToDelete;
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

        // We require that this value either have a computable evolution or that
        // the loop have a constant iteration count.  In the case where the loop
        // has a constant iteration count, we can sometimes force evaluation of
        // the exit value through brute force.
        SCEVHandle SH = SE->getSCEV(Inst);
        if (!SH->hasComputableLoopEvolution(L) && !HasConstantItCount)
          continue;          // Cannot get exit evolution for the loop value.

        // Okay, this instruction has a user outside of the current loop
        // and varies predictably *inside* the loop.  Evaluate the value it
        // contains when the loop exits, if possible.
        SCEVHandle ExitValue = SE->getSCEVAtScope(Inst, L->getParentLoop());
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

        // If this instruction is dead now, schedule it to be removed.
        if (Inst->use_empty())
          InstructionsToDelete.insert(Inst);

        // See if this is a single-entry LCSSA PHI node.  If so, we can (and
        // have to) remove
        // the PHI entirely.  This is safe, because the NewVal won't be variant
        // in the loop, so we don't need an LCSSA phi node anymore.
        if (NumPreds == 1) {
          SE->deleteValueFromRecords(PN);
          PN->replaceAllUsesWith(ExitVal);
          PN->eraseFromParent();
          break;
        }
      }
    }
  }

  DeleteTriviallyDeadInstructions(InstructionsToDelete);
}

void IndVarSimplify::RewriteNonIntegerIVs(Loop *L) {
  // First step.  Check to see if there are any floating-point recurrences.
  // If there are, change them into integer recurrences, permitting analysis by
  // the SCEV routines.
  //
  BasicBlock *Header    = L->getHeader();

  SmallPtrSet<Instruction*, 16> DeadInsts;
  for (BasicBlock::iterator I = Header->begin(); isa<PHINode>(I); ++I) {
    PHINode *PN = cast<PHINode>(I);
    HandleFloatingPointIV(L, PN, DeadInsts);
  }

  // If the loop previously had floating-point IV, ScalarEvolution
  // may not have been able to compute a trip count. Now that we've done some
  // re-writing, the trip count may be computable.
  if (Changed)
    SE->forgetLoopBackedgeTakenCount(L);

  if (!DeadInsts.empty())
    DeleteTriviallyDeadInstructions(DeadInsts);
}

/// getEffectiveIndvarType - Determine the widest type that the
/// induction-variable PHINode Phi is cast to.
///
static const Type *getEffectiveIndvarType(const PHINode *Phi,
                                          const ScalarEvolution *SE) {
  const Type *Ty = Phi->getType();

  for (Value::use_const_iterator UI = Phi->use_begin(), UE = Phi->use_end();
       UI != UE; ++UI) {
    const Type *CandidateType = NULL;
    if (const ZExtInst *ZI = dyn_cast<ZExtInst>(UI))
      CandidateType = ZI->getDestTy();
    else if (const SExtInst *SI = dyn_cast<SExtInst>(UI))
      CandidateType = SI->getDestTy();
    else if (const IntToPtrInst *IP = dyn_cast<IntToPtrInst>(UI))
      CandidateType = IP->getDestTy();
    else if (const PtrToIntInst *PI = dyn_cast<PtrToIntInst>(UI))
      CandidateType = PI->getDestTy();
    if (CandidateType &&
        SE->isSCEVable(CandidateType) &&
        SE->getTypeSizeInBits(CandidateType) > SE->getTypeSizeInBits(Ty))
      Ty = CandidateType;
  }

  return Ty;
}

/// TestOrigIVForWrap - Analyze the original induction variable
/// that controls the loop's iteration to determine whether it
/// would ever undergo signed or unsigned overflow. Also, check
/// whether an induction variable in the same type that starts
/// at 0 would undergo signed overflow.
///
/// In addition to setting the NoSignedWrap and NoUnsignedWrap
/// variables to true when appropriate (they are not set to false here),
/// return the PHI for this induction variable.  Also record the initial
/// and final values and the increment; these are not meaningful unless
/// either NoSignedWrap or NoUnsignedWrap is true, and are always meaningful
/// in that case, although the final value may be 0 indicating a nonconstant.
///
/// TODO: This duplicates a fair amount of ScalarEvolution logic.
/// Perhaps this can be merged with
/// ScalarEvolution::getBackedgeTakenCount
/// and/or ScalarEvolution::get{Sign,Zero}ExtendExpr.
///
static const PHINode *TestOrigIVForWrap(const Loop *L,
                                        const BranchInst *BI,
                                        const Instruction *OrigCond,
                                        const ScalarEvolution &SE,
                                        bool &NoSignedWrap,
                                        bool &NoUnsignedWrap,
                                        const ConstantInt* &InitialVal,
                                        const ConstantInt* &IncrVal,
                                        const ConstantInt* &LimitVal) {
  // Verify that the loop is sane and find the exit condition.
  const ICmpInst *Cmp = dyn_cast<ICmpInst>(OrigCond);
  if (!Cmp) return 0;

  const Value *CmpLHS = Cmp->getOperand(0);
  const Value *CmpRHS = Cmp->getOperand(1);
  const BasicBlock *TrueBB = BI->getSuccessor(0);
  const BasicBlock *FalseBB = BI->getSuccessor(1);
  ICmpInst::Predicate Pred = Cmp->getPredicate();

  // Canonicalize a constant to the RHS.
  if (isa<ConstantInt>(CmpLHS)) {
    Pred = ICmpInst::getSwappedPredicate(Pred);
    std::swap(CmpLHS, CmpRHS);
  }
  // Canonicalize SLE to SLT.
  if (Pred == ICmpInst::ICMP_SLE)
    if (const ConstantInt *CI = dyn_cast<ConstantInt>(CmpRHS))
      if (!CI->getValue().isMaxSignedValue()) {
        CmpRHS = ConstantInt::get(CI->getValue() + 1);
        Pred = ICmpInst::ICMP_SLT;
      }
  // Canonicalize SGT to SGE.
  if (Pred == ICmpInst::ICMP_SGT)
    if (const ConstantInt *CI = dyn_cast<ConstantInt>(CmpRHS))
      if (!CI->getValue().isMaxSignedValue()) {
        CmpRHS = ConstantInt::get(CI->getValue() + 1);
        Pred = ICmpInst::ICMP_SGE;
      }
  // Canonicalize SGE to SLT.
  if (Pred == ICmpInst::ICMP_SGE) {
    std::swap(TrueBB, FalseBB);
    Pred = ICmpInst::ICMP_SLT;
  }
  // Canonicalize ULE to ULT.
  if (Pred == ICmpInst::ICMP_ULE)
    if (const ConstantInt *CI = dyn_cast<ConstantInt>(CmpRHS))
      if (!CI->getValue().isMaxValue()) {
        CmpRHS = ConstantInt::get(CI->getValue() + 1);
        Pred = ICmpInst::ICMP_ULT;
      }
  // Canonicalize UGT to UGE.
  if (Pred == ICmpInst::ICMP_UGT)
    if (const ConstantInt *CI = dyn_cast<ConstantInt>(CmpRHS))
      if (!CI->getValue().isMaxValue()) {
        CmpRHS = ConstantInt::get(CI->getValue() + 1);
        Pred = ICmpInst::ICMP_UGE;
      }
  // Canonicalize UGE to ULT.
  if (Pred == ICmpInst::ICMP_UGE) {
    std::swap(TrueBB, FalseBB);
    Pred = ICmpInst::ICMP_ULT;
  }
  // For now, analyze only LT loops for signed overflow.
  if (Pred != ICmpInst::ICMP_SLT && Pred != ICmpInst::ICMP_ULT)
    return 0;

  bool isSigned = Pred == ICmpInst::ICMP_SLT;

  // Get the increment instruction. Look past casts if we will
  // be able to prove that the original induction variable doesn't
  // undergo signed or unsigned overflow, respectively.
  const Value *IncrInst = CmpLHS;
  if (isSigned) {
    if (const SExtInst *SI = dyn_cast<SExtInst>(CmpLHS)) {
      if (!isa<ConstantInt>(CmpRHS) ||
          !cast<ConstantInt>(CmpRHS)->getValue()
            .isSignedIntN(SE.getTypeSizeInBits(IncrInst->getType())))
        return 0;
      IncrInst = SI->getOperand(0);
    }
  } else {
    if (const ZExtInst *ZI = dyn_cast<ZExtInst>(CmpLHS)) {
      if (!isa<ConstantInt>(CmpRHS) ||
          !cast<ConstantInt>(CmpRHS)->getValue()
            .isIntN(SE.getTypeSizeInBits(IncrInst->getType())))
        return 0;
      IncrInst = ZI->getOperand(0);
    }
  }

  // For now, only analyze induction variables that have simple increments.
  const BinaryOperator *IncrOp = dyn_cast<BinaryOperator>(IncrInst);
  if (!IncrOp || IncrOp->getOpcode() != Instruction::Add)
    return 0;
  IncrVal = dyn_cast<ConstantInt>(IncrOp->getOperand(1));
  if (!IncrVal)
    return 0;

  // Make sure the PHI looks like a normal IV.
  const PHINode *PN = dyn_cast<PHINode>(IncrOp->getOperand(0));
  if (!PN || PN->getNumIncomingValues() != 2)
    return 0;
  unsigned IncomingEdge = L->contains(PN->getIncomingBlock(0));
  unsigned BackEdge = !IncomingEdge;
  if (!L->contains(PN->getIncomingBlock(BackEdge)) ||
      PN->getIncomingValue(BackEdge) != IncrOp)
    return 0;
  if (!L->contains(TrueBB))
    return 0;

  // For now, only analyze loops with a constant start value, so that
  // we can easily determine if the start value is not a maximum value
  // which would wrap on the first iteration.
  InitialVal = dyn_cast<ConstantInt>(PN->getIncomingValue(IncomingEdge));
  if (!InitialVal)
    return 0;

  // The upper limit need not be a constant; we'll check later.
  LimitVal = dyn_cast<ConstantInt>(CmpRHS);

  // We detect the impossibility of wrapping in two cases, both of
  // which require starting with a non-max value:
  // - The IV counts up by one, and the loop iterates only while it remains
  // less than a limiting value (any) in the same type.
  // - The IV counts up by a positive increment other than 1, and the
  // constant limiting value + the increment is less than the max value
  // (computed as max-increment to avoid overflow)
  if (isSigned && !InitialVal->getValue().isMaxSignedValue()) {
    if (IncrVal->equalsInt(1))
      NoSignedWrap = true;    // LimitVal need not be constant
    else if (LimitVal) {
      uint64_t numBits = LimitVal->getValue().getBitWidth();
      if (IncrVal->getValue().sgt(APInt::getNullValue(numBits)) &&
          (APInt::getSignedMaxValue(numBits) - IncrVal->getValue())
            .sgt(LimitVal->getValue()))
        NoSignedWrap = true;
    }
  } else if (!isSigned && !InitialVal->getValue().isMaxValue()) {
    if (IncrVal->equalsInt(1))
      NoUnsignedWrap = true;  // LimitVal need not be constant
    else if (LimitVal) {
      uint64_t numBits = LimitVal->getValue().getBitWidth();
      if (IncrVal->getValue().ugt(APInt::getNullValue(numBits)) &&
          (APInt::getMaxValue(numBits) - IncrVal->getValue())
            .ugt(LimitVal->getValue()))
        NoUnsignedWrap = true;
    }
  }
  return PN;
}

static Value *getSignExtendedTruncVar(const SCEVAddRecExpr *AR,
                                      ScalarEvolution *SE,
                                      const Type *LargestType, Loop *L, 
                                      const Type *myType,
                                      SCEVExpander &Rewriter) {
  SCEVHandle ExtendedStart =
    SE->getSignExtendExpr(AR->getStart(), LargestType);
  SCEVHandle ExtendedStep =
    SE->getSignExtendExpr(AR->getStepRecurrence(*SE), LargestType);
  SCEVHandle ExtendedAddRec =
    SE->getAddRecExpr(ExtendedStart, ExtendedStep, L);
  if (LargestType != myType)
    ExtendedAddRec = SE->getTruncateExpr(ExtendedAddRec, myType);
  return Rewriter.expandCodeFor(ExtendedAddRec, myType);
}

static Value *getZeroExtendedTruncVar(const SCEVAddRecExpr *AR,
                                      ScalarEvolution *SE,
                                      const Type *LargestType, Loop *L, 
                                      const Type *myType,
                                      SCEVExpander &Rewriter) {
  SCEVHandle ExtendedStart =
    SE->getZeroExtendExpr(AR->getStart(), LargestType);
  SCEVHandle ExtendedStep =
    SE->getZeroExtendExpr(AR->getStepRecurrence(*SE), LargestType);
  SCEVHandle ExtendedAddRec =
    SE->getAddRecExpr(ExtendedStart, ExtendedStep, L);
  if (LargestType != myType)
    ExtendedAddRec = SE->getTruncateExpr(ExtendedAddRec, myType);
  return Rewriter.expandCodeFor(ExtendedAddRec, myType);
}

/// allUsesAreSameTyped - See whether all Uses of I are instructions
/// with the same Opcode and the same type.
static bool allUsesAreSameTyped(unsigned int Opcode, Instruction *I) {
  const Type* firstType = NULL;
  for (Value::use_iterator UI = I->use_begin(), UE = I->use_end();
       UI != UE; ++UI) {
    Instruction *II = dyn_cast<Instruction>(*UI);
    if (!II || II->getOpcode() != Opcode)
      return false;
    if (!firstType)
      firstType = II->getType();
    else if (firstType != II->getType())
      return false;
  }
  return true;
}

bool IndVarSimplify::runOnLoop(Loop *L, LPPassManager &LPM) {
  LI = &getAnalysis<LoopInfo>();
  SE = &getAnalysis<ScalarEvolution>();
  Changed = false;

  // If there are any floating-point recurrences, attempt to
  // transform them to use integer recurrences.
  RewriteNonIntegerIVs(L);

  BasicBlock *Header       = L->getHeader();
  BasicBlock *ExitingBlock = L->getExitingBlock();
  SmallPtrSet<Instruction*, 16> DeadInsts;

  // Verify the input to the pass in already in LCSSA form.
  assert(L->isLCSSAForm());

  // Check to see if this loop has a computable loop-invariant execution count.
  // If so, this means that we can compute the final value of any expressions
  // that are recurrent in the loop, and substitute the exit values from the
  // loop into any instructions outside of the loop that use the final values of
  // the current expressions.
  //
  SCEVHandle BackedgeTakenCount = SE->getBackedgeTakenCount(L);
  if (!isa<SCEVCouldNotCompute>(BackedgeTakenCount))
    RewriteLoopExitValues(L, BackedgeTakenCount);

  // Next, analyze all of the induction variables in the loop, canonicalizing
  // auxillary induction variables.
  std::vector<std::pair<PHINode*, SCEVHandle> > IndVars;

  for (BasicBlock::iterator I = Header->begin(); isa<PHINode>(I); ++I) {
    PHINode *PN = cast<PHINode>(I);
    if (SE->isSCEVable(PN->getType())) {
      SCEVHandle SCEV = SE->getSCEV(PN);
      // FIXME: It is an extremely bad idea to indvar substitute anything more
      // complex than affine induction variables.  Doing so will put expensive
      // polynomial evaluations inside of the loop, and the str reduction pass
      // currently can only reduce affine polynomials.  For now just disable
      // indvar subst on anything more complex than an affine addrec.
      if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(SCEV))
        if (AR->getLoop() == L && AR->isAffine())
          IndVars.push_back(std::make_pair(PN, SCEV));
    }
  }

  // Compute the type of the largest recurrence expression, and collect
  // the set of the types of the other recurrence expressions.
  const Type *LargestType = 0;
  SmallSetVector<const Type *, 4> SizesToInsert;
  if (!isa<SCEVCouldNotCompute>(BackedgeTakenCount)) {
    LargestType = BackedgeTakenCount->getType();
    LargestType = SE->getEffectiveSCEVType(LargestType);
    SizesToInsert.insert(LargestType);
  }
  for (unsigned i = 0, e = IndVars.size(); i != e; ++i) {
    const PHINode *PN = IndVars[i].first;
    const Type *PNTy = PN->getType();
    PNTy = SE->getEffectiveSCEVType(PNTy);
    SizesToInsert.insert(PNTy);
    const Type *EffTy = getEffectiveIndvarType(PN, SE);
    EffTy = SE->getEffectiveSCEVType(EffTy);
    SizesToInsert.insert(EffTy);
    if (!LargestType ||
        SE->getTypeSizeInBits(EffTy) >
          SE->getTypeSizeInBits(LargestType))
      LargestType = EffTy;
  }

  // Create a rewriter object which we'll use to transform the code with.
  SCEVExpander Rewriter(*SE, *LI);

  // Now that we know the largest of of the induction variables in this loop,
  // insert a canonical induction variable of the largest size.
  Value *IndVar = 0;
  if (!SizesToInsert.empty()) {
    IndVar = Rewriter.getOrInsertCanonicalInductionVariable(L,LargestType);
    ++NumInserted;
    Changed = true;
    DOUT << "INDVARS: New CanIV: " << *IndVar;
  }

  // If we have a trip count expression, rewrite the loop's exit condition
  // using it.  We can currently only handle loops with a single exit.
  bool NoSignedWrap = false;
  bool NoUnsignedWrap = false;
  const ConstantInt* InitialVal, * IncrVal, * LimitVal;
  const PHINode *OrigControllingPHI = 0;
  if (!isa<SCEVCouldNotCompute>(BackedgeTakenCount) && ExitingBlock)
    // Can't rewrite non-branch yet.
    if (BranchInst *BI = dyn_cast<BranchInst>(ExitingBlock->getTerminator())) {
      if (Instruction *OrigCond = dyn_cast<Instruction>(BI->getCondition())) {
        // Determine if the OrigIV will ever undergo overflow.
        OrigControllingPHI =
          TestOrigIVForWrap(L, BI, OrigCond, *SE,
                            NoSignedWrap, NoUnsignedWrap,
                            InitialVal, IncrVal, LimitVal);

        // We'll be replacing the original condition, so it'll be dead.
        DeadInsts.insert(OrigCond);
      }

      LinearFunctionTestReplace(L, BackedgeTakenCount, IndVar,
                                ExitingBlock, BI, Rewriter);
    }

  // Now that we have a canonical induction variable, we can rewrite any
  // recurrences in terms of the induction variable.  Start with the auxillary
  // induction variables, and recursively rewrite any of their uses.
  BasicBlock::iterator InsertPt = Header->getFirstNonPHI();
  Rewriter.setInsertionPoint(InsertPt);

  // If there were induction variables of other sizes, cast the primary
  // induction variable to the right size for them, avoiding the need for the
  // code evaluation methods to insert induction variables of different sizes.
  for (unsigned i = 0, e = SizesToInsert.size(); i != e; ++i) {
    const Type *Ty = SizesToInsert[i];
    if (Ty != LargestType) {
      Instruction *New = new TruncInst(IndVar, Ty, "indvar", InsertPt);
      Rewriter.addInsertedValue(New, SE->getSCEV(New));
      DOUT << "INDVARS: Made trunc IV for type " << *Ty << ": "
           << *New << "\n";
    }
  }

  // Rewrite all induction variables in terms of the canonical induction
  // variable.
  while (!IndVars.empty()) {
    PHINode *PN = IndVars.back().first;
    const SCEVAddRecExpr *AR = cast<SCEVAddRecExpr>(IndVars.back().second);
    Value *NewVal = Rewriter.expandCodeFor(AR, PN->getType());
    DOUT << "INDVARS: Rewrote IV '" << *AR << "' " << *PN
         << "   into = " << *NewVal << "\n";
    NewVal->takeName(PN);

    /// If the new canonical induction variable is wider than the original,
    /// and the original has uses that are casts to wider types, see if the
    /// truncate and extend can be omitted.
    if (PN == OrigControllingPHI && PN->getType() != LargestType)
      for (Value::use_iterator UI = PN->use_begin(), UE = PN->use_end();
           UI != UE; ++UI) {
        Instruction *UInst = dyn_cast<Instruction>(*UI);
        if (UInst && isa<SExtInst>(UInst) && NoSignedWrap) {
          Value *TruncIndVar = getSignExtendedTruncVar(AR, SE, LargestType, L, 
                                         UInst->getType(), Rewriter);
          UInst->replaceAllUsesWith(TruncIndVar);
          DeadInsts.insert(UInst);
        }
        // See if we can figure out sext(i+constant) doesn't wrap, so we can
        // use a larger add.  This is common in subscripting.
        if (UInst && UInst->getOpcode()==Instruction::Add &&
            !UInst->use_empty() &&
            allUsesAreSameTyped(Instruction::SExt, UInst) &&
            isa<ConstantInt>(UInst->getOperand(1)) &&
            NoSignedWrap && LimitVal) {
          uint64_t oldBitSize = LimitVal->getValue().getBitWidth();
          uint64_t newBitSize = LargestType->getPrimitiveSizeInBits();
          ConstantInt* AddRHS = dyn_cast<ConstantInt>(UInst->getOperand(1));
          if (((APInt::getSignedMaxValue(oldBitSize) - IncrVal->getValue()) -
                AddRHS->getValue()).sgt(LimitVal->getValue())) {
            // We've determined this is (i+constant) and it won't overflow.
            if (isa<SExtInst>(UInst->use_begin())) {
              SExtInst* oldSext = dyn_cast<SExtInst>(UInst->use_begin());
              uint64_t truncSize = oldSext->getType()->getPrimitiveSizeInBits();
              Value *TruncIndVar = getSignExtendedTruncVar(AR, SE, LargestType,
                                                L, oldSext->getType(), Rewriter);
              APInt APnewAddRHS = APInt(AddRHS->getValue()).sext(newBitSize);
              if (newBitSize > truncSize)
                APnewAddRHS = APnewAddRHS.trunc(truncSize);
              ConstantInt* newAddRHS =ConstantInt::get(APnewAddRHS);
              Value *NewAdd = 
                    BinaryOperator::CreateAdd(TruncIndVar, newAddRHS,
                                              UInst->getName()+".nosex", UInst);
              for (Value::use_iterator UI2 = UInst->use_begin(), 
                    UE2 = UInst->use_end(); UI2 != UE2; ++UI2) {
                Instruction *II = dyn_cast<Instruction>(UI2);
                II->replaceAllUsesWith(NewAdd);
                DeadInsts.insert(II);
              }
              DeadInsts.insert(UInst);
            }
          }
        }
        // Try for sext(i | constant).  This is safe as long as the
        // high bit of the constant is not set.
        if (UInst && UInst->getOpcode()==Instruction::Or &&
            !UInst->use_empty() &&
            allUsesAreSameTyped(Instruction::SExt, UInst) && NoSignedWrap &&
            isa<ConstantInt>(UInst->getOperand(1))) {
          ConstantInt* RHS = dyn_cast<ConstantInt>(UInst->getOperand(1));
          if (!RHS->getValue().isNegative()) {
            uint64_t newBitSize = LargestType->getPrimitiveSizeInBits();
            SExtInst* oldSext = dyn_cast<SExtInst>(UInst->use_begin());
            uint64_t truncSize = oldSext->getType()->getPrimitiveSizeInBits();
            Value *TruncIndVar = getSignExtendedTruncVar(AR, SE, LargestType,
                                              L, oldSext->getType(), Rewriter);
            APInt APnewOrRHS = APInt(RHS->getValue()).sext(newBitSize);
            if (newBitSize > truncSize)
              APnewOrRHS = APnewOrRHS.trunc(truncSize);
            ConstantInt* newOrRHS =ConstantInt::get(APnewOrRHS);
            Value *NewOr = 
                  BinaryOperator::CreateOr(TruncIndVar, newOrRHS,
                                            UInst->getName()+".nosex", UInst);
            for (Value::use_iterator UI2 = UInst->use_begin(), 
                  UE2 = UInst->use_end(); UI2 != UE2; ++UI2) {
              Instruction *II = dyn_cast<Instruction>(UI2);
              II->replaceAllUsesWith(NewOr);
              DeadInsts.insert(II);
            }
            DeadInsts.insert(UInst);
          }
        }
        // A zext of a signed variable known not to overflow is still safe.
        if (UInst && isa<ZExtInst>(UInst) && (NoUnsignedWrap || NoSignedWrap)) {
          Value *TruncIndVar = getZeroExtendedTruncVar(AR, SE, LargestType, L, 
                                         UInst->getType(), Rewriter);
          UInst->replaceAllUsesWith(TruncIndVar);
          DeadInsts.insert(UInst);
        }
        // If we have zext(i&constant), it's always safe to use the larger
        // variable.  This is not common but is a bottleneck in Openssl.
        // (RHS doesn't have to be constant.  There should be a better approach
        // than bottom-up pattern matching for this...)
        if (UInst && UInst->getOpcode()==Instruction::And &&
            !UInst->use_empty() &&
            allUsesAreSameTyped(Instruction::ZExt, UInst) &&
            isa<ConstantInt>(UInst->getOperand(1))) {
          uint64_t newBitSize = LargestType->getPrimitiveSizeInBits();
          ConstantInt* AndRHS = dyn_cast<ConstantInt>(UInst->getOperand(1));
          ZExtInst* oldZext = dyn_cast<ZExtInst>(UInst->use_begin());
          uint64_t truncSize = oldZext->getType()->getPrimitiveSizeInBits();
          Value *TruncIndVar = getSignExtendedTruncVar(AR, SE, LargestType,
                                  L, oldZext->getType(), Rewriter);
          APInt APnewAndRHS = APInt(AndRHS->getValue()).zext(newBitSize);
          if (newBitSize > truncSize)
            APnewAndRHS = APnewAndRHS.trunc(truncSize);
          ConstantInt* newAndRHS = ConstantInt::get(APnewAndRHS);
          Value *NewAnd = 
                BinaryOperator::CreateAnd(TruncIndVar, newAndRHS,
                                          UInst->getName()+".nozex", UInst);
          for (Value::use_iterator UI2 = UInst->use_begin(), 
                UE2 = UInst->use_end(); UI2 != UE2; ++UI2) {
            Instruction *II = dyn_cast<Instruction>(UI2);
            II->replaceAllUsesWith(NewAnd);
            DeadInsts.insert(II);
          }
          DeadInsts.insert(UInst);
        }
        // If we have zext((i+constant)&constant), we can use the larger
        // variable even if the add does overflow.  This works whenever the
        // constant being ANDed is the same size as i, which it presumably is.
        // We don't need to restrict the expression being and'ed to i+const,
        // but we have to promote everything in it, so it's convenient.
        // zext((i | constant)&constant) is also valid and accepted here.
        if (UInst && (UInst->getOpcode()==Instruction::Add ||
                      UInst->getOpcode()==Instruction::Or) &&
            UInst->hasOneUse() &&
            isa<ConstantInt>(UInst->getOperand(1))) {
          uint64_t newBitSize = LargestType->getPrimitiveSizeInBits();
          ConstantInt* AddRHS = dyn_cast<ConstantInt>(UInst->getOperand(1));
          Instruction *UInst2 = dyn_cast<Instruction>(UInst->use_begin());
          if (UInst2 && UInst2->getOpcode() == Instruction::And &&
              !UInst2->use_empty() &&
              allUsesAreSameTyped(Instruction::ZExt, UInst2) &&
              isa<ConstantInt>(UInst2->getOperand(1))) {
            ZExtInst* oldZext = dyn_cast<ZExtInst>(UInst2->use_begin());
            uint64_t truncSize = oldZext->getType()->getPrimitiveSizeInBits();
            Value *TruncIndVar = getSignExtendedTruncVar(AR, SE, LargestType,
                                    L, oldZext->getType(), Rewriter);
            ConstantInt* AndRHS = dyn_cast<ConstantInt>(UInst2->getOperand(1));
            APInt APnewAddRHS = APInt(AddRHS->getValue()).zext(newBitSize);
            if (newBitSize > truncSize)
              APnewAddRHS = APnewAddRHS.trunc(truncSize);
            ConstantInt* newAddRHS = ConstantInt::get(APnewAddRHS);
            Value *NewAdd = ((UInst->getOpcode()==Instruction::Add) ?
                  BinaryOperator::CreateAdd(TruncIndVar, newAddRHS,
                                            UInst->getName()+".nozex", UInst2) :
                  BinaryOperator::CreateOr(TruncIndVar, newAddRHS,
                                            UInst->getName()+".nozex", UInst2));
            APInt APcopy2 = APInt(AndRHS->getValue());
            ConstantInt* newAndRHS = ConstantInt::get(APcopy2.zext(newBitSize));
            Value *NewAnd = 
                  BinaryOperator::CreateAnd(NewAdd, newAndRHS,
                                            UInst->getName()+".nozex", UInst2);
            for (Value::use_iterator UI2 = UInst2->use_begin(), 
                  UE2 = UInst2->use_end(); UI2 != UE2; ++UI2) {
              Instruction *II = dyn_cast<Instruction>(UI2);
              II->replaceAllUsesWith(NewAnd);
              DeadInsts.insert(II);
            }
            DeadInsts.insert(UInst);
            DeadInsts.insert(UInst2);
          }
        }
      }

    // Replace the old PHI Node with the inserted computation.
    PN->replaceAllUsesWith(NewVal);
    DeadInsts.insert(PN);
    IndVars.pop_back();
    ++NumRemoved;
    Changed = true;
  }

  DeleteTriviallyDeadInstructions(DeadInsts);
  assert(L->isLCSSAForm());
  return Changed;
}

/// Return true if it is OK to use SIToFPInst for an inducation variable
/// with given inital and exit values.
static bool useSIToFPInst(ConstantFP &InitV, ConstantFP &ExitV,
                          uint64_t intIV, uint64_t intEV) {

  if (InitV.getValueAPF().isNegative() || ExitV.getValueAPF().isNegative())
    return true;

  // If the iteration range can be handled by SIToFPInst then use it.
  APInt Max = APInt::getSignedMaxValue(32);
  if (Max.getZExtValue() > static_cast<uint64_t>(abs(intEV - intIV)))
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
void IndVarSimplify::HandleFloatingPointIV(Loop *L, PHINode *PH,
                                   SmallPtrSet<Instruction*, 16> &DeadInsts) {

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

  // Delete old, floating point, exit comparision instruction.
  EC->replaceAllUsesWith(NewEC);
  DeadInsts.insert(EC);

  // Delete old, floating point, increment instruction.
  Incr->replaceAllUsesWith(UndefValue::get(Incr->getType()));
  DeadInsts.insert(Incr);

  // Replace floating induction variable. Give SIToFPInst preference over
  // UIToFPInst because it is faster on platforms that are widely used.
  if (useSIToFPInst(*InitValue, *EV, newInitValue, intEV)) {
    SIToFPInst *Conv = new SIToFPInst(NewPHI, PH->getType(), "indvar.conv",
                                      PH->getParent()->getFirstNonPHI());
    PH->replaceAllUsesWith(Conv);
  } else {
    UIToFPInst *Conv = new UIToFPInst(NewPHI, PH->getType(), "indvar.conv",
                                      PH->getParent()->getFirstNonPHI());
    PH->replaceAllUsesWith(Conv);
  }
  DeadInsts.insert(PH);
}

