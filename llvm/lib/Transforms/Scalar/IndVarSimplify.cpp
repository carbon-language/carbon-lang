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
#include "llvm/Target/TargetData.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
using namespace llvm;

STATISTIC(NumRemoved     , "Number of aux indvars removed");
STATISTIC(NumWidened     , "Number of indvars widened");
STATISTIC(NumInserted    , "Number of canonical indvars added");
STATISTIC(NumReplaced    , "Number of exit values replaced");
STATISTIC(NumLFTR        , "Number of loop exit tests replaced");
STATISTIC(NumElimIdentity, "Number of IV identities eliminated");
STATISTIC(NumElimExt     , "Number of IV sign/zero extends eliminated");
STATISTIC(NumElimRem     , "Number of IV remainder operations eliminated");
STATISTIC(NumElimCmp     , "Number of IV comparisons eliminated");

static cl::opt<bool> DisableIVRewrite(
  "disable-iv-rewrite", cl::Hidden,
  cl::desc("Disable canonical induction variable rewriting"));

namespace {
  class IndVarSimplify : public LoopPass {
    IVUsers         *IU;
    LoopInfo        *LI;
    ScalarEvolution *SE;
    DominatorTree   *DT;
    TargetData      *TD;

    SmallVector<WeakVH, 16> DeadInsts;
    bool Changed;
  public:

    static char ID; // Pass identification, replacement for typeid
    IndVarSimplify() : LoopPass(ID), IU(0), LI(0), SE(0), DT(0), TD(0),
                       Changed(false) {
      initializeIndVarSimplifyPass(*PassRegistry::getPassRegistry());
    }

    virtual bool runOnLoop(Loop *L, LPPassManager &LPM);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<DominatorTree>();
      AU.addRequired<LoopInfo>();
      AU.addRequired<ScalarEvolution>();
      AU.addRequiredID(LoopSimplifyID);
      AU.addRequiredID(LCSSAID);
      if (!DisableIVRewrite)
        AU.addRequired<IVUsers>();
      AU.addPreserved<ScalarEvolution>();
      AU.addPreservedID(LoopSimplifyID);
      AU.addPreservedID(LCSSAID);
      if (!DisableIVRewrite)
        AU.addPreserved<IVUsers>();
      AU.setPreservesCFG();
    }

  private:
    bool isValidRewrite(Value *FromVal, Value *ToVal);

    void SimplifyIVUsers(SCEVExpander &Rewriter);
    void SimplifyIVUsersNoRewrite(Loop *L, SCEVExpander &Rewriter);

    bool EliminateIVUser(Instruction *UseInst, Instruction *IVOperand);
    void EliminateIVComparison(ICmpInst *ICmp, Value *IVOperand);
    void EliminateIVRemainder(BinaryOperator *Rem,
                              Value *IVOperand,
                              bool IsSigned);
    bool isSimpleIVUser(Instruction *I, const Loop *L);
    void RewriteNonIntegerIVs(Loop *L);

    ICmpInst *LinearFunctionTestReplace(Loop *L, const SCEV *BackedgeTakenCount,
                                        PHINode *IndVar,
                                        SCEVExpander &Rewriter);

    void RewriteLoopExitValues(Loop *L, SCEVExpander &Rewriter);

    void RewriteIVExpressions(Loop *L, SCEVExpander &Rewriter);

    void SinkUnusedInvariants(Loop *L);

    void HandleFloatingPointIV(Loop *L, PHINode *PH);
  };
}

char IndVarSimplify::ID = 0;
INITIALIZE_PASS_BEGIN(IndVarSimplify, "indvars",
                "Induction Variable Simplification", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
INITIALIZE_PASS_DEPENDENCY(LoopInfo)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(LCSSA)
INITIALIZE_PASS_DEPENDENCY(IVUsers)
INITIALIZE_PASS_END(IndVarSimplify, "indvars",
                "Induction Variable Simplification", false, false)

Pass *llvm::createIndVarSimplifyPass() {
  return new IndVarSimplify();
}

/// isValidRewrite - Return true if the SCEV expansion generated by the
/// rewriter can replace the original value. SCEV guarantees that it
/// produces the same value, but the way it is produced may be illegal IR.
/// Ideally, this function will only be called for verification.
bool IndVarSimplify::isValidRewrite(Value *FromVal, Value *ToVal) {
  // If an SCEV expression subsumed multiple pointers, its expansion could
  // reassociate the GEP changing the base pointer. This is illegal because the
  // final address produced by a GEP chain must be inbounds relative to its
  // underlying object. Otherwise basic alias analysis, among other things,
  // could fail in a dangerous way. Ultimately, SCEV will be improved to avoid
  // producing an expression involving multiple pointers. Until then, we must
  // bail out here.
  //
  // Retrieve the pointer operand of the GEP. Don't use GetUnderlyingObject
  // because it understands lcssa phis while SCEV does not.
  Value *FromPtr = FromVal;
  Value *ToPtr = ToVal;
  if (GEPOperator *GEP = dyn_cast<GEPOperator>(FromVal)) {
    FromPtr = GEP->getPointerOperand();
  }
  if (GEPOperator *GEP = dyn_cast<GEPOperator>(ToVal)) {
    ToPtr = GEP->getPointerOperand();
  }
  if (FromPtr != FromVal || ToPtr != ToVal) {
    // Quickly check the common case
    if (FromPtr == ToPtr)
      return true;

    // SCEV may have rewritten an expression that produces the GEP's pointer
    // operand. That's ok as long as the pointer operand has the same base
    // pointer. Unlike GetUnderlyingObject(), getPointerBase() will find the
    // base of a recurrence. This handles the case in which SCEV expansion
    // converts a pointer type recurrence into a nonrecurrent pointer base
    // indexed by an integer recurrence.
    const SCEV *FromBase = SE->getPointerBase(SE->getSCEV(FromPtr));
    const SCEV *ToBase = SE->getPointerBase(SE->getSCEV(ToPtr));
    if (FromBase == ToBase)
      return true;

    DEBUG(dbgs() << "INDVARS: GEP rewrite bail out "
          << *FromBase << " != " << *ToBase << "\n");

    return false;
  }
  return true;
}

/// canExpandBackedgeTakenCount - Return true if this loop's backedge taken
/// count expression can be safely and cheaply expanded into an instruction
/// sequence that can be used by LinearFunctionTestReplace.
static bool canExpandBackedgeTakenCount(Loop *L, ScalarEvolution *SE) {
  const SCEV *BackedgeTakenCount = SE->getBackedgeTakenCount(L);
  if (isa<SCEVCouldNotCompute>(BackedgeTakenCount) ||
      BackedgeTakenCount->isZero())
    return false;

  if (!L->getExitingBlock())
    return false;

  // Can't rewrite non-branch yet.
  BranchInst *BI = dyn_cast<BranchInst>(L->getExitingBlock()->getTerminator());
  if (!BI)
    return false;

  // Special case: If the backedge-taken count is a UDiv, it's very likely a
  // UDiv that ScalarEvolution produced in order to compute a precise
  // expression, rather than a UDiv from the user's code. If we can't find a
  // UDiv in the code with some simple searching, assume the former and forego
  // rewriting the loop.
  if (isa<SCEVUDivExpr>(BackedgeTakenCount)) {
    ICmpInst *OrigCond = dyn_cast<ICmpInst>(BI->getCondition());
    if (!OrigCond) return false;
    const SCEV *R = SE->getSCEV(OrigCond->getOperand(1));
    R = SE->getMinusSCEV(R, SE->getConstant(R->getType(), 1));
    if (R != BackedgeTakenCount) {
      const SCEV *L = SE->getSCEV(OrigCond->getOperand(0));
      L = SE->getMinusSCEV(L, SE->getConstant(L->getType(), 1));
      if (L != BackedgeTakenCount)
        return false;
    }
  }
  return true;
}

/// getBackedgeIVType - Get the widest type used by the loop test after peeking
/// through Truncs.
///
/// TODO: Unnecessary once LinearFunctionTestReplace is removed.
static const Type *getBackedgeIVType(Loop *L) {
  if (!L->getExitingBlock())
    return 0;

  // Can't rewrite non-branch yet.
  BranchInst *BI = dyn_cast<BranchInst>(L->getExitingBlock()->getTerminator());
  if (!BI)
    return 0;

  ICmpInst *Cond = dyn_cast<ICmpInst>(BI->getCondition());
  if (!Cond)
    return 0;

  const Type *Ty = 0;
  for(User::op_iterator OI = Cond->op_begin(), OE = Cond->op_end();
      OI != OE; ++OI) {
    assert((!Ty || Ty == (*OI)->getType()) && "bad icmp operand types");
    TruncInst *Trunc = dyn_cast<TruncInst>(*OI);
    if (!Trunc)
      continue;

    return Trunc->getSrcTy();
  }
  return Ty;
}

/// LinearFunctionTestReplace - This method rewrites the exit condition of the
/// loop to be a canonical != comparison against the incremented loop induction
/// variable.  This pass is able to rewrite the exit tests of any loop where the
/// SCEV analysis can determine a loop-invariant trip count of the loop, which
/// is actually a much broader range than just linear tests.
ICmpInst *IndVarSimplify::
LinearFunctionTestReplace(Loop *L,
                          const SCEV *BackedgeTakenCount,
                          PHINode *IndVar,
                          SCEVExpander &Rewriter) {
  assert(canExpandBackedgeTakenCount(L, SE) && "precondition");
  BranchInst *BI = cast<BranchInst>(L->getExitingBlock()->getTerminator());

  // If the exiting block is not the same as the backedge block, we must compare
  // against the preincremented value, otherwise we prefer to compare against
  // the post-incremented value.
  Value *CmpIndVar;
  const SCEV *RHS = BackedgeTakenCount;
  if (L->getExitingBlock() == L->getLoopLatch()) {
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
    CmpIndVar = IndVar->getIncomingValueForBlock(L->getExitingBlock());
  } else {
    // We have to use the preincremented value...
    RHS = SE->getTruncateOrZeroExtend(BackedgeTakenCount,
                                      IndVar->getType());
    CmpIndVar = IndVar;
  }

  // Expand the code for the iteration count.
  assert(SE->isLoopInvariant(RHS, L) &&
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
  DeadInsts.push_back(OrigCond);

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
void IndVarSimplify::RewriteLoopExitValues(Loop *L, SCEVExpander &Rewriter) {
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
        if (!SE->isLoopInvariant(ExitValue, L))
          continue;

        Value *ExitVal = Rewriter.expandCodeFor(ExitValue, PN->getType(), Inst);

        DEBUG(dbgs() << "INDVARS: RLEV: AfterLoopVal = " << *ExitVal << '\n'
                     << "  LoopVal = " << *Inst << "\n");

        if (!isValidRewrite(Inst, ExitVal)) {
          DeadInsts.push_back(ExitVal);
          continue;
        }
        Changed = true;
        ++NumReplaced;

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
  BasicBlock *Header = L->getHeader();

  SmallVector<WeakVH, 8> PHIs;
  for (BasicBlock::iterator I = Header->begin();
       PHINode *PN = dyn_cast<PHINode>(I); ++I)
    PHIs.push_back(PN);

  for (unsigned i = 0, e = PHIs.size(); i != e; ++i)
    if (PHINode *PN = dyn_cast_or_null<PHINode>(&*PHIs[i]))
      HandleFloatingPointIV(L, PN);

  // If the loop previously had floating-point IV, ScalarEvolution
  // may not have been able to compute a trip count. Now that we've done some
  // re-writing, the trip count may be computable.
  if (Changed)
    SE->forgetLoop(L);
}

/// SimplifyIVUsers - Iteratively perform simplification on IVUsers within this
/// loop. IVUsers is treated as a worklist. Each successive simplification may
/// push more users which may themselves be candidates for simplification.
///
/// This is the old approach to IV simplification to be replaced by
/// SimplifyIVUsersNoRewrite.
///
void IndVarSimplify::SimplifyIVUsers(SCEVExpander &Rewriter) {
  // Each round of simplification involves a round of eliminating operations
  // followed by a round of widening IVs. A single IVUsers worklist is used
  // across all rounds. The inner loop advances the user. If widening exposes
  // more uses, then another pass through the outer loop is triggered.
  for (IVUsers::iterator I = IU->begin(); I != IU->end(); ++I) {
    Instruction *UseInst = I->getUser();
    Value *IVOperand = I->getOperandValToReplace();

    if (ICmpInst *ICmp = dyn_cast<ICmpInst>(UseInst)) {
      EliminateIVComparison(ICmp, IVOperand);
      continue;
    }
    if (BinaryOperator *Rem = dyn_cast<BinaryOperator>(UseInst)) {
      bool IsSigned = Rem->getOpcode() == Instruction::SRem;
      if (IsSigned || Rem->getOpcode() == Instruction::URem) {
        EliminateIVRemainder(Rem, IVOperand, IsSigned);
        continue;
      }
    }
  }
}

namespace {
  // Collect information about induction variables that are used by sign/zero
  // extend operations. This information is recorded by CollectExtend and
  // provides the input to WidenIV.
  struct WideIVInfo {
    const Type *WidestNativeType; // Widest integer type created [sz]ext
    bool IsSigned;                // Was an sext user seen before a zext?

    WideIVInfo() : WidestNativeType(0), IsSigned(false) {}
  };
}

/// CollectExtend - Update information about the induction variable that is
/// extended by this sign or zero extend operation. This is used to determine
/// the final width of the IV before actually widening it.
static void CollectExtend(CastInst *Cast, bool IsSigned, WideIVInfo &WI,
                          ScalarEvolution *SE, const TargetData *TD) {
  const Type *Ty = Cast->getType();
  uint64_t Width = SE->getTypeSizeInBits(Ty);
  if (TD && !TD->isLegalInteger(Width))
    return;

  if (!WI.WidestNativeType) {
    WI.WidestNativeType = SE->getEffectiveSCEVType(Ty);
    WI.IsSigned = IsSigned;
    return;
  }

  // We extend the IV to satisfy the sign of its first user, arbitrarily.
  if (WI.IsSigned != IsSigned)
    return;

  if (Width > SE->getTypeSizeInBits(WI.WidestNativeType))
    WI.WidestNativeType = SE->getEffectiveSCEVType(Ty);
}

namespace {
/// WidenIV - The goal of this transform is to remove sign and zero extends
/// without creating any new induction variables. To do this, it creates a new
/// phi of the wider type and redirects all users, either removing extends or
/// inserting truncs whenever we stop propagating the type.
///
class WidenIV {
  // Parameters
  PHINode *OrigPhi;
  const Type *WideType;
  bool IsSigned;

  // Context
  LoopInfo        *LI;
  Loop            *L;
  ScalarEvolution *SE;
  DominatorTree   *DT;

  // Result
  PHINode *WidePhi;
  Instruction *WideInc;
  const SCEV *WideIncExpr;
  SmallVectorImpl<WeakVH> &DeadInsts;

  SmallPtrSet<Instruction*,16> Widened;

public:
  WidenIV(PHINode *PN, const WideIVInfo &WI, LoopInfo *LInfo,
          ScalarEvolution *SEv, DominatorTree *DTree,
          SmallVectorImpl<WeakVH> &DI) :
    OrigPhi(PN),
    WideType(WI.WidestNativeType),
    IsSigned(WI.IsSigned),
    LI(LInfo),
    L(LI->getLoopFor(OrigPhi->getParent())),
    SE(SEv),
    DT(DTree),
    WidePhi(0),
    WideInc(0),
    WideIncExpr(0),
    DeadInsts(DI) {
    assert(L->getHeader() == OrigPhi->getParent() && "Phi must be an IV");
  }

  PHINode *CreateWideIV(SCEVExpander &Rewriter);

protected:
  Instruction *CloneIVUser(Instruction *NarrowUse,
                           Instruction *NarrowDef,
                           Instruction *WideDef);

  const SCEVAddRecExpr *GetWideRecurrence(Instruction *NarrowUse);

  Instruction *WidenIVUse(Instruction *NarrowUse,
                          Instruction *NarrowDef,
                          Instruction *WideDef);
};
} // anonymous namespace

static Value *getExtend( Value *NarrowOper, const Type *WideType,
                               bool IsSigned, IRBuilder<> &Builder) {
  return IsSigned ? Builder.CreateSExt(NarrowOper, WideType) :
                    Builder.CreateZExt(NarrowOper, WideType);
}

/// CloneIVUser - Instantiate a wide operation to replace a narrow
/// operation. This only needs to handle operations that can evaluation to
/// SCEVAddRec. It can safely return 0 for any operation we decide not to clone.
Instruction *WidenIV::CloneIVUser(Instruction *NarrowUse,
                                  Instruction *NarrowDef,
                                  Instruction *WideDef) {
  unsigned Opcode = NarrowUse->getOpcode();
  switch (Opcode) {
  default:
    return 0;
  case Instruction::Add:
  case Instruction::Mul:
  case Instruction::UDiv:
  case Instruction::Sub:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
    DEBUG(dbgs() << "Cloning IVUser: " << *NarrowUse << "\n");

    IRBuilder<> Builder(NarrowUse);

    // Replace NarrowDef operands with WideDef. Otherwise, we don't know
    // anything about the narrow operand yet so must insert a [sz]ext. It is
    // probably loop invariant and will be folded or hoisted. If it actually
    // comes from a widened IV, it should be removed during a future call to
    // WidenIVUse.
    Value *LHS = (NarrowUse->getOperand(0) == NarrowDef) ? WideDef :
      getExtend(NarrowUse->getOperand(0), WideType, IsSigned, Builder);
    Value *RHS = (NarrowUse->getOperand(1) == NarrowDef) ? WideDef :
      getExtend(NarrowUse->getOperand(1), WideType, IsSigned, Builder);

    BinaryOperator *NarrowBO = cast<BinaryOperator>(NarrowUse);
    BinaryOperator *WideBO = BinaryOperator::Create(NarrowBO->getOpcode(),
                                                    LHS, RHS,
                                                    NarrowBO->getName());
    Builder.Insert(WideBO);
    if (NarrowBO->hasNoUnsignedWrap()) WideBO->setHasNoUnsignedWrap();
    if (NarrowBO->hasNoSignedWrap()) WideBO->setHasNoSignedWrap();

    return WideBO;
  }
  llvm_unreachable(0);
}

// GetWideRecurrence - Is this instruction potentially interesting from IVUsers'
// perspective after widening it's type? In other words, can the extend be
// safely hoisted out of the loop with SCEV reducing the value to a recurrence
// on the same loop. If so, return the sign or zero extended
// recurrence. Otherwise return NULL.
const SCEVAddRecExpr *WidenIV::GetWideRecurrence(Instruction *NarrowUse) {
  if (!SE->isSCEVable(NarrowUse->getType()))
    return 0;

  const SCEV *NarrowExpr = SE->getSCEV(NarrowUse);
  const SCEV *WideExpr = IsSigned ?
    SE->getSignExtendExpr(NarrowExpr, WideType) :
    SE->getZeroExtendExpr(NarrowExpr, WideType);
  const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(WideExpr);
  if (!AddRec || AddRec->getLoop() != L)
    return 0;

  return AddRec;
}

/// HoistStep - Attempt to hoist an IV increment above a potential use.
///
/// To successfully hoist, two criteria must be met:
/// - IncV operands dominate InsertPos and
/// - InsertPos dominates IncV
///
/// Meeting the second condition means that we don't need to check all of IncV's
/// existing uses (it's moving up in the domtree).
///
/// This does not yet recursively hoist the operands, although that would
/// not be difficult.
static bool HoistStep(Instruction *IncV, Instruction *InsertPos,
                      const DominatorTree *DT)
{
  if (DT->dominates(IncV, InsertPos))
    return true;

  if (!DT->dominates(InsertPos->getParent(), IncV->getParent()))
    return false;

  if (IncV->mayHaveSideEffects())
    return false;

  // Attempt to hoist IncV
  for (User::op_iterator OI = IncV->op_begin(), OE = IncV->op_end();
       OI != OE; ++OI) {
    Instruction *OInst = dyn_cast<Instruction>(OI);
    if (OInst && !DT->dominates(OInst, InsertPos))
      return false;
  }
  IncV->moveBefore(InsertPos);
  return true;
}

/// WidenIVUse - Determine whether an individual user of the narrow IV can be
/// widened. If so, return the wide clone of the user.
Instruction *WidenIV::WidenIVUse(Instruction *NarrowUse,
                                 Instruction *NarrowDef,
                                 Instruction *WideDef) {
  // To be consistent with IVUsers, stop traversing the def-use chain at
  // inner-loop phis or post-loop phis.
  if (isa<PHINode>(NarrowUse) && LI->getLoopFor(NarrowUse->getParent()) != L)
    return 0;

  // Handle data flow merges and bizarre phi cycles.
  if (!Widened.insert(NarrowUse))
    return 0;

  // Our raison d'etre! Eliminate sign and zero extension.
  if (IsSigned ? isa<SExtInst>(NarrowUse) : isa<ZExtInst>(NarrowUse)) {
    Value *NewDef = WideDef;
    if (NarrowUse->getType() != WideType) {
      unsigned CastWidth = SE->getTypeSizeInBits(NarrowUse->getType());
      unsigned IVWidth = SE->getTypeSizeInBits(WideType);
      if (CastWidth < IVWidth) {
        // The cast isn't as wide as the IV, so insert a Trunc.
        IRBuilder<> Builder(NarrowUse);
        NewDef = Builder.CreateTrunc(WideDef, NarrowUse->getType());
      }
      else {
        // A wider extend was hidden behind a narrower one. This may induce
        // another round of IV widening in which the intermediate IV becomes
        // dead. It should be very rare.
        DEBUG(dbgs() << "INDVARS: New IV " << *WidePhi
              << " not wide enough to subsume " << *NarrowUse << "\n");
        NarrowUse->replaceUsesOfWith(NarrowDef, WideDef);
        NewDef = NarrowUse;
      }
    }
    if (NewDef != NarrowUse) {
      DEBUG(dbgs() << "INDVARS: eliminating " << *NarrowUse
            << " replaced by " << *WideDef << "\n");
      ++NumElimExt;
      NarrowUse->replaceAllUsesWith(NewDef);
      DeadInsts.push_back(NarrowUse);
    }
    // Now that the extend is gone, we want to expose it's uses for potential
    // further simplification. We don't need to directly inform SimplifyIVUsers
    // of the new users, because their parent IV will be processed later as a
    // new loop phi. If we preserved IVUsers analysis, we would also want to
    // push the uses of WideDef here.

    // No further widening is needed. The deceased [sz]ext had done it for us.
    return 0;
  }
  const SCEVAddRecExpr *WideAddRec = GetWideRecurrence(NarrowUse);
  if (!WideAddRec) {
    // This user does not evaluate to a recurence after widening, so don't
    // follow it. Instead insert a Trunc to kill off the original use,
    // eventually isolating the original narrow IV so it can be removed.
    IRBuilder<> Builder(NarrowUse);
    Value *Trunc = Builder.CreateTrunc(WideDef, NarrowDef->getType());
    NarrowUse->replaceUsesOfWith(NarrowDef, Trunc);
    return 0;
  }
  // Reuse the IV increment that SCEVExpander created as long as it dominates
  // NarrowUse.
  Instruction *WideUse = 0;
  if (WideAddRec == WideIncExpr && HoistStep(WideInc, NarrowUse, DT)) {
    WideUse = WideInc;
  }
  else {
    WideUse = CloneIVUser(NarrowUse, NarrowDef, WideDef);
    if (!WideUse)
      return 0;
  }
  // GetWideRecurrence ensured that the narrow expression could be extended
  // outside the loop without overflow. This suggests that the wide use
  // evaluates to the same expression as the extended narrow use, but doesn't
  // absolutely guarantee it. Hence the following failsafe check. In rare cases
  // where it fails, we simply throw away the newly created wide use.
  if (WideAddRec != SE->getSCEV(WideUse)) {
    DEBUG(dbgs() << "Wide use expression mismatch: " << *WideUse
          << ": " << *SE->getSCEV(WideUse) << " != " << *WideAddRec << "\n");
    DeadInsts.push_back(WideUse);
    return 0;
  }

  // Returning WideUse pushes it on the worklist.
  return WideUse;
}

/// CreateWideIV - Process a single induction variable. First use the
/// SCEVExpander to create a wide induction variable that evaluates to the same
/// recurrence as the original narrow IV. Then use a worklist to forward
/// traverse the narrow IV's def-use chain. After WidenIVUse has processed all
/// interesting IV users, the narrow IV will be isolated for removal by
/// DeleteDeadPHIs.
///
/// It would be simpler to delete uses as they are processed, but we must avoid
/// invalidating SCEV expressions.
///
PHINode *WidenIV::CreateWideIV(SCEVExpander &Rewriter) {
  // Is this phi an induction variable?
  const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(SE->getSCEV(OrigPhi));
  if (!AddRec)
    return NULL;

  // Widen the induction variable expression.
  const SCEV *WideIVExpr = IsSigned ?
    SE->getSignExtendExpr(AddRec, WideType) :
    SE->getZeroExtendExpr(AddRec, WideType);

  assert(SE->getEffectiveSCEVType(WideIVExpr->getType()) == WideType &&
         "Expect the new IV expression to preserve its type");

  // Can the IV be extended outside the loop without overflow?
  AddRec = dyn_cast<SCEVAddRecExpr>(WideIVExpr);
  if (!AddRec || AddRec->getLoop() != L)
    return NULL;

  // An AddRec must have loop-invariant operands. Since this AddRec is
  // materialized by a loop header phi, the expression cannot have any post-loop
  // operands, so they must dominate the loop header.
  assert(SE->properlyDominates(AddRec->getStart(), L->getHeader()) &&
         SE->properlyDominates(AddRec->getStepRecurrence(*SE), L->getHeader())
         && "Loop header phi recurrence inputs do not dominate the loop");

  // The rewriter provides a value for the desired IV expression. This may
  // either find an existing phi or materialize a new one. Either way, we
  // expect a well-formed cyclic phi-with-increments. i.e. any operand not part
  // of the phi-SCC dominates the loop entry.
  Instruction *InsertPt = L->getHeader()->begin();
  WidePhi = cast<PHINode>(Rewriter.expandCodeFor(AddRec, WideType, InsertPt));

  // Remembering the WideIV increment generated by SCEVExpander allows
  // WidenIVUse to reuse it when widening the narrow IV's increment. We don't
  // employ a general reuse mechanism because the call above is the only call to
  // SCEVExpander. Henceforth, we produce 1-to-1 narrow to wide uses.
  if (BasicBlock *LatchBlock = L->getLoopLatch()) {
    WideInc =
      cast<Instruction>(WidePhi->getIncomingValueForBlock(LatchBlock));
    WideIncExpr = SE->getSCEV(WideInc);
  }

  DEBUG(dbgs() << "Wide IV: " << *WidePhi << "\n");
  ++NumWidened;

  // Traverse the def-use chain using a worklist starting at the original IV.
  assert(Widened.empty() && "expect initial state" );

  // Each worklist entry has a Narrow def-use link and Wide def.
  SmallVector<std::pair<Use *, Instruction *>, 8> NarrowIVUsers;
  for (Value::use_iterator UI = OrigPhi->use_begin(),
         UE = OrigPhi->use_end(); UI != UE; ++UI) {
    NarrowIVUsers.push_back(std::make_pair(&UI.getUse(), WidePhi));
  }
  while (!NarrowIVUsers.empty()) {
    Use *NarrowDefUse;
    Instruction *WideDef;
    tie(NarrowDefUse, WideDef) = NarrowIVUsers.pop_back_val();

    // Process a def-use edge. This may replace the use, so don't hold a
    // use_iterator across it.
    Instruction *NarrowDef = cast<Instruction>(NarrowDefUse->get());
    Instruction *NarrowUse = cast<Instruction>(NarrowDefUse->getUser());
    Instruction *WideUse = WidenIVUse(NarrowUse, NarrowDef, WideDef);

    // Follow all def-use edges from the previous narrow use.
    if (WideUse) {
      for (Value::use_iterator UI = NarrowUse->use_begin(),
             UE = NarrowUse->use_end(); UI != UE; ++UI) {
        NarrowIVUsers.push_back(std::make_pair(&UI.getUse(), WideUse));
      }
    }
    // WidenIVUse may have removed the def-use edge.
    if (NarrowDef->use_empty())
      DeadInsts.push_back(NarrowDef);
  }
  return WidePhi;
}

void IndVarSimplify::EliminateIVComparison(ICmpInst *ICmp, Value *IVOperand) {
  unsigned IVOperIdx = 0;
  ICmpInst::Predicate Pred = ICmp->getPredicate();
  if (IVOperand != ICmp->getOperand(0)) {
    // Swapped
    assert(IVOperand == ICmp->getOperand(1) && "Can't find IVOperand");
    IVOperIdx = 1;
    Pred = ICmpInst::getSwappedPredicate(Pred);
  }

  // Get the SCEVs for the ICmp operands.
  const SCEV *S = SE->getSCEV(ICmp->getOperand(IVOperIdx));
  const SCEV *X = SE->getSCEV(ICmp->getOperand(1 - IVOperIdx));

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
    return;

  DEBUG(dbgs() << "INDVARS: Eliminated comparison: " << *ICmp << '\n');
  ++NumElimCmp;
  Changed = true;
  DeadInsts.push_back(ICmp);
}

void IndVarSimplify::EliminateIVRemainder(BinaryOperator *Rem,
                                          Value *IVOperand,
                                          bool IsSigned) {
  // We're only interested in the case where we know something about
  // the numerator.
  if (IVOperand != Rem->getOperand(0))
    return;

  // Get the SCEVs for the ICmp operands.
  const SCEV *S = SE->getSCEV(Rem->getOperand(0));
  const SCEV *X = SE->getSCEV(Rem->getOperand(1));

  // Simplify unnecessary loops away.
  const Loop *ICmpLoop = LI->getLoopFor(Rem->getParent());
  S = SE->getSCEVAtScope(S, ICmpLoop);
  X = SE->getSCEVAtScope(X, ICmpLoop);

  // i % n  -->  i  if i is in [0,n).
  if ((!IsSigned || SE->isKnownNonNegative(S)) &&
      SE->isKnownPredicate(IsSigned ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT,
                           S, X))
    Rem->replaceAllUsesWith(Rem->getOperand(0));
  else {
    // (i+1) % n  -->  (i+1)==n?0:(i+1)  if i is in [0,n).
    const SCEV *LessOne =
      SE->getMinusSCEV(S, SE->getConstant(S->getType(), 1));
    if (IsSigned && !SE->isKnownNonNegative(LessOne))
      return;

    if (!SE->isKnownPredicate(IsSigned ?
                              ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT,
                              LessOne, X))
      return;

    ICmpInst *ICmp = new ICmpInst(Rem, ICmpInst::ICMP_EQ,
                                  Rem->getOperand(0), Rem->getOperand(1),
                                  "tmp");
    SelectInst *Sel =
      SelectInst::Create(ICmp,
                         ConstantInt::get(Rem->getType(), 0),
                         Rem->getOperand(0), "tmp", Rem);
    Rem->replaceAllUsesWith(Sel);
  }

  // Inform IVUsers about the new users.
  if (IU) {
    if (Instruction *I = dyn_cast<Instruction>(Rem->getOperand(0)))
      IU->AddUsersIfInteresting(I);
  }
  DEBUG(dbgs() << "INDVARS: Simplified rem: " << *Rem << '\n');
  ++NumElimRem;
  Changed = true;
  DeadInsts.push_back(Rem);
}

/// EliminateIVUser - Eliminate an operation that consumes a simple IV and has
/// no observable side-effect given the range of IV values.
bool IndVarSimplify::EliminateIVUser(Instruction *UseInst,
                                     Instruction *IVOperand) {
  if (ICmpInst *ICmp = dyn_cast<ICmpInst>(UseInst)) {
    EliminateIVComparison(ICmp, IVOperand);
    return true;
  }
  if (BinaryOperator *Rem = dyn_cast<BinaryOperator>(UseInst)) {
    bool IsSigned = Rem->getOpcode() == Instruction::SRem;
    if (IsSigned || Rem->getOpcode() == Instruction::URem) {
      EliminateIVRemainder(Rem, IVOperand, IsSigned);
      return true;
    }
  }

  // Eliminate any operation that SCEV can prove is an identity function.
  if (!SE->isSCEVable(UseInst->getType()) ||
      (UseInst->getType() != IVOperand->getType()) ||
      (SE->getSCEV(UseInst) != SE->getSCEV(IVOperand)))
    return false;

  UseInst->replaceAllUsesWith(IVOperand);

  DEBUG(dbgs() << "INDVARS: Eliminated identity: " << *UseInst << '\n');
  ++NumElimIdentity;
  Changed = true;
  DeadInsts.push_back(UseInst);
  return true;
}

/// pushIVUsers - Add all uses of Def to the current IV's worklist.
///
static void pushIVUsers(
  Instruction *Def,
  SmallPtrSet<Instruction*,16> &Simplified,
  SmallVectorImpl< std::pair<Instruction*,Instruction*> > &SimpleIVUsers) {

  for (Value::use_iterator UI = Def->use_begin(), E = Def->use_end();
       UI != E; ++UI) {
    Instruction *User = cast<Instruction>(*UI);

    // Avoid infinite or exponential worklist processing.
    // Also ensure unique worklist users.
    if (Simplified.insert(User))
      SimpleIVUsers.push_back(std::make_pair(User, Def));
  }
}

/// isSimpleIVUser - Return true if this instruction generates a simple SCEV
/// expression in terms of that IV.
///
/// This is similar to IVUsers' isInsteresting() but processes each instruction
/// non-recursively when the operand is already known to be a simpleIVUser.
///
bool IndVarSimplify::isSimpleIVUser(Instruction *I, const Loop *L) {
  if (!SE->isSCEVable(I->getType()))
    return false;

  // Get the symbolic expression for this instruction.
  const SCEV *S = SE->getSCEV(I);

  // Only consider affine recurrences.
  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(S);
  if (AR && AR->getLoop() == L)
    return true;

  return false;
}

/// SimplifyIVUsersNoRewrite - Iteratively perform simplification on a worklist
/// of IV users. Each successive simplification may push more users which may
/// themselves be candidates for simplification.
///
/// The "NoRewrite" algorithm does not require IVUsers analysis. Instead, it
/// simplifies instructions in-place during analysis. Rather than rewriting
/// induction variables bottom-up from their users, it transforms a chain of
/// IVUsers top-down, updating the IR only when it encouters a clear
/// optimization opportunitiy. A SCEVExpander "Rewriter" instance is still
/// needed, but only used to generate a new IV (phi) of wider type for sign/zero
/// extend elimination.
///
/// Once DisableIVRewrite is default, LSR will be the only client of IVUsers.
///
void IndVarSimplify::SimplifyIVUsersNoRewrite(Loop *L, SCEVExpander &Rewriter) {
  std::map<PHINode *, WideIVInfo> WideIVMap;

  SmallVector<PHINode*, 8> LoopPhis;
  for (BasicBlock::iterator I = L->getHeader()->begin(); isa<PHINode>(I); ++I) {
    LoopPhis.push_back(cast<PHINode>(I));
  }
  // Each round of simplification iterates through the SimplifyIVUsers worklist
  // for all current phis, then determines whether any IVs can be
  // widened. Widening adds new phis to LoopPhis, inducing another round of
  // simplification on the wide IVs.
  while (!LoopPhis.empty()) {
    // Evaluate as many IV expressions as possible before widening any IVs. This
    // forces SCEV to set no-wrap flags before evaluating sign/zero
    // extension. The first time SCEV attempts to normalize sign/zero extension,
    // the result becomes final. So for the most predictable results, we delay
    // evaluation of sign/zero extend evaluation until needed, and avoid running
    // other SCEV based analysis prior to SimplifyIVUsersNoRewrite.
    do {
      PHINode *CurrIV = LoopPhis.pop_back_val();

      // Information about sign/zero extensions of CurrIV.
      WideIVInfo WI;

      // Instructions processed by SimplifyIVUsers for CurrIV.
      SmallPtrSet<Instruction*,16> Simplified;

      // Use-def pairs if IVUsers waiting to be processed for CurrIV.
      SmallVector<std::pair<Instruction*, Instruction*>, 8> SimpleIVUsers;

      pushIVUsers(CurrIV, Simplified, SimpleIVUsers);

      while (!SimpleIVUsers.empty()) {
        Instruction *UseInst, *Operand;
        tie(UseInst, Operand) = SimpleIVUsers.pop_back_val();

        if (EliminateIVUser(UseInst, Operand)) {
          pushIVUsers(Operand, Simplified, SimpleIVUsers);
          continue;
        }
        if (CastInst *Cast = dyn_cast<CastInst>(UseInst)) {
          bool IsSigned = Cast->getOpcode() == Instruction::SExt;
          if (IsSigned || Cast->getOpcode() == Instruction::ZExt) {
            CollectExtend(Cast, IsSigned, WI, SE, TD);
          }
          continue;
        }
        if (isSimpleIVUser(UseInst, L)) {
          pushIVUsers(UseInst, Simplified, SimpleIVUsers);
        }
      }
      if (WI.WidestNativeType) {
        WideIVMap[CurrIV] = WI;
      }
    } while(!LoopPhis.empty());

    for (std::map<PHINode *, WideIVInfo>::const_iterator I = WideIVMap.begin(),
           E = WideIVMap.end(); I != E; ++I) {
      WidenIV Widener(I->first, I->second, LI, SE, DT, DeadInsts);
      if (PHINode *WidePhi = Widener.CreateWideIV(Rewriter)) {
        Changed = true;
        LoopPhis.push_back(WidePhi);
      }
    }
    WideIVMap.clear();
  }
}

bool IndVarSimplify::runOnLoop(Loop *L, LPPassManager &LPM) {
  // If LoopSimplify form is not available, stay out of trouble. Some notes:
  //  - LSR currently only supports LoopSimplify-form loops. Indvars'
  //    canonicalization can be a pessimization without LSR to "clean up"
  //    afterwards.
  //  - We depend on having a preheader; in particular,
  //    Loop::getCanonicalInductionVariable only supports loops with preheaders,
  //    and we're in trouble if we can't find the induction variable even when
  //    we've manually inserted one.
  if (!L->isLoopSimplifyForm())
    return false;

  if (!DisableIVRewrite)
    IU = &getAnalysis<IVUsers>();
  LI = &getAnalysis<LoopInfo>();
  SE = &getAnalysis<ScalarEvolution>();
  DT = &getAnalysis<DominatorTree>();
  TD = getAnalysisIfAvailable<TargetData>();

  DeadInsts.clear();
  Changed = false;

  // If there are any floating-point recurrences, attempt to
  // transform them to use integer recurrences.
  RewriteNonIntegerIVs(L);

  const SCEV *BackedgeTakenCount = SE->getBackedgeTakenCount(L);

  // Create a rewriter object which we'll use to transform the code with.
  SCEVExpander Rewriter(*SE, "indvars");

  // Eliminate redundant IV users.
  //
  // Simplification works best when run before other consumers of SCEV. We
  // attempt to avoid evaluating SCEVs for sign/zero extend operations until
  // other expressions involving loop IVs have been evaluated. This helps SCEV
  // set no-wrap flags before normalizing sign/zero extension.
  if (DisableIVRewrite) {
    Rewriter.disableCanonicalMode();
    SimplifyIVUsersNoRewrite(L, Rewriter);
  }

  // Check to see if this loop has a computable loop-invariant execution count.
  // If so, this means that we can compute the final value of any expressions
  // that are recurrent in the loop, and substitute the exit values from the
  // loop into any instructions outside of the loop that use the final values of
  // the current expressions.
  //
  if (!isa<SCEVCouldNotCompute>(BackedgeTakenCount))
    RewriteLoopExitValues(L, Rewriter);

  // Eliminate redundant IV users.
  if (!DisableIVRewrite)
    SimplifyIVUsers(Rewriter);

  // Compute the type of the largest recurrence expression, and decide whether
  // a canonical induction variable should be inserted.
  const Type *LargestType = 0;
  bool NeedCannIV = false;
  bool ExpandBECount = canExpandBackedgeTakenCount(L, SE);
  if (ExpandBECount) {
    // If we have a known trip count and a single exit block, we'll be
    // rewriting the loop exit test condition below, which requires a
    // canonical induction variable.
    NeedCannIV = true;
    const Type *Ty = BackedgeTakenCount->getType();
    if (DisableIVRewrite) {
      // In this mode, SimplifyIVUsers may have already widened the IV used by
      // the backedge test and inserted a Trunc on the compare's operand. Get
      // the wider type to avoid creating a redundant narrow IV only used by the
      // loop test.
      LargestType = getBackedgeIVType(L);
    }
    if (!LargestType ||
        SE->getTypeSizeInBits(Ty) >
        SE->getTypeSizeInBits(LargestType))
      LargestType = SE->getEffectiveSCEVType(Ty);
  }
  if (!DisableIVRewrite) {
    for (IVUsers::const_iterator I = IU->begin(), E = IU->end(); I != E; ++I) {
      NeedCannIV = true;
      const Type *Ty =
        SE->getEffectiveSCEVType(I->getOperandValToReplace()->getType());
      if (!LargestType ||
          SE->getTypeSizeInBits(Ty) >
          SE->getTypeSizeInBits(LargestType))
        LargestType = Ty;
    }
  }

  // Now that we know the largest of the induction variable expressions
  // in this loop, insert a canonical induction variable of the largest size.
  PHINode *IndVar = 0;
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
  if (ExpandBECount) {
    assert(canExpandBackedgeTakenCount(L, SE) &&
           "canonical IV disrupted BackedgeTaken expansion");
    assert(NeedCannIV &&
           "LinearFunctionTestReplace requires a canonical induction variable");
    NewICmp = LinearFunctionTestReplace(L, BackedgeTakenCount, IndVar,
                                        Rewriter);
  }
  // Rewrite IV-derived expressions.
  if (!DisableIVRewrite)
    RewriteIVExpressions(L, Rewriter);

  // Clear the rewriter cache, because values that are in the rewriter's cache
  // can be deleted in the loop below, causing the AssertingVH in the cache to
  // trigger.
  Rewriter.clear();

  // Now that we're done iterating through lists, clean up any instructions
  // which are now dead.
  while (!DeadInsts.empty())
    if (Instruction *Inst =
          dyn_cast_or_null<Instruction>(&*DeadInsts.pop_back_val()))
      RecursivelyDeleteTriviallyDeadInstructions(Inst);

  // The Rewriter may not be used from this point on.

  // Loop-invariant instructions in the preheader that aren't used in the
  // loop may be sunk below the loop to reduce register pressure.
  SinkUnusedInvariants(L);

  // For completeness, inform IVUsers of the IV use in the newly-created
  // loop exit test instruction.
  if (NewICmp && IU)
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
static bool isSafe(const SCEV *S, const Loop *L, ScalarEvolution *SE) {
  // Loop-invariant values are safe.
  if (SE->isLoopInvariant(S, L)) return true;

  // Affine addrecs are safe. Non-affine are not, because LSR doesn't know how
  // to transform them into efficient code.
  if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(S))
    return AR->isAffine();

  // An add is safe it all its operands are safe.
  if (const SCEVCommutativeExpr *Commutative = dyn_cast<SCEVCommutativeExpr>(S)) {
    for (SCEVCommutativeExpr::op_iterator I = Commutative->op_begin(),
         E = Commutative->op_end(); I != E; ++I)
      if (!isSafe(*I, L, SE)) return false;
    return true;
  }

  // A cast is safe if its operand is.
  if (const SCEVCastExpr *C = dyn_cast<SCEVCastExpr>(S))
    return isSafe(C->getOperand(), L, SE);

  // A udiv is safe if its operands are.
  if (const SCEVUDivExpr *UD = dyn_cast<SCEVUDivExpr>(S))
    return isSafe(UD->getLHS(), L, SE) &&
           isSafe(UD->getRHS(), L, SE);

  // SCEVUnknown is always safe.
  if (isa<SCEVUnknown>(S))
    return true;

  // Nothing else is safe.
  return false;
}

void IndVarSimplify::RewriteIVExpressions(Loop *L, SCEVExpander &Rewriter) {
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
      if (SE->isLoopInvariant(ExitVal, L))
        AR = ExitVal;
    }

    // FIXME: It is an extremely bad idea to indvar substitute anything more
    // complex than affine induction variables.  Doing so will put expensive
    // polynomial evaluations inside of the loop, and the str reduction pass
    // currently can only reduce affine polynomials.  For now just disable
    // indvar subst on anything more complex than an affine addrec, unless
    // it can be expanded to a trivial value.
    if (!isSafe(AR, L, SE))
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

    DEBUG(dbgs() << "INDVARS: Rewrote IV '" << *AR << "' " << *Op << '\n'
                 << "   into = " << *NewVal << "\n");

    if (!isValidRewrite(Op, NewVal)) {
      DeadInsts.push_back(NewVal);
      continue;
    }
    // Inform ScalarEvolution that this value is changing. The change doesn't
    // affect its value, but it does potentially affect which use lists the
    // value will be on after the replacement, which affects ScalarEvolution's
    // ability to walk use lists and drop dangling pointers when a value is
    // deleted.
    SE->forgetValue(User);

    // Patch the new value into place.
    if (Op->hasName())
      NewVal->takeName(Op);
    if (Instruction *NewValI = dyn_cast<Instruction>(NewVal))
      NewValI->setDebugLoc(User->getDebugLoc());
    User->replaceUsesOfWith(Op, NewVal);
    UI->setOperandValToReplace(NewVal);

    ++NumRemoved;
    Changed = true;

    // The old value may be dead now.
    DeadInsts.push_back(Op);
  }
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
      User *U = *UI;
      BasicBlock *UseBB = cast<Instruction>(U)->getParent();
      if (PHINode *P = dyn_cast<PHINode>(U)) {
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
  Instruction *U1 = cast<Instruction>(*IncrUse++);
  if (IncrUse == Incr->use_end()) return;
  Instruction *U2 = cast<Instruction>(*IncrUse++);
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
  PHINode *NewPHI = PHINode::Create(Int32Ty, 2, PN->getName()+".int", PN);
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
  if (IU)
    IU->AddUsersIfInteresting(NewPHI);
}
