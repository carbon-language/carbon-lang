//===- LoopStrengthReduce.cpp - Strength Reduce IVs in Loops --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This transformation analyzes and transforms the induction variables (and
// computations derived from them) into forms suitable for efficient execution
// on the target.
//
// This pass performs a strength reduction on array references inside loops that
// have as one or more of their components the loop induction variable, it
// rewrites expressions to take advantage of scaled-index addressing modes
// available on the target, and it performs a variety of other optimizations
// related to loop induction variables.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loop-reduce"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Type.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/IVUsers.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Transforms/Utils/AddrModeMatcher.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetLowering.h"
#include <algorithm>
using namespace llvm;

STATISTIC(NumReduced ,    "Number of IV uses strength reduced");
STATISTIC(NumInserted,    "Number of PHIs inserted");
STATISTIC(NumVariable,    "Number of PHIs with variable strides");
STATISTIC(NumEliminated,  "Number of strides eliminated");
STATISTIC(NumShadow,      "Number of Shadow IVs optimized");
STATISTIC(NumImmSunk,     "Number of common expr immediates sunk into uses");
STATISTIC(NumLoopCond,    "Number of loop terminating conds optimized");

static cl::opt<bool> EnableFullLSRMode("enable-full-lsr",
                                       cl::init(false),
                                       cl::Hidden);

namespace {

  struct BasedUser;

  /// IVInfo - This structure keeps track of one IV expression inserted during
  /// StrengthReduceStridedIVUsers. It contains the stride, the common base, as
  /// well as the PHI node and increment value created for rewrite.
  struct IVExpr {
    const SCEV *Stride;
    const SCEV *Base;
    PHINode    *PHI;

    IVExpr(const SCEV *const stride, const SCEV *const base, PHINode *phi)
      : Stride(stride), Base(base), PHI(phi) {}
  };

  /// IVsOfOneStride - This structure keeps track of all IV expression inserted
  /// during StrengthReduceStridedIVUsers for a particular stride of the IV.
  struct IVsOfOneStride {
    std::vector<IVExpr> IVs;

    void addIV(const SCEV *const Stride, const SCEV *const Base, PHINode *PHI) {
      IVs.push_back(IVExpr(Stride, Base, PHI));
    }
  };

  class LoopStrengthReduce : public LoopPass {
    IVUsers *IU;
    LoopInfo *LI;
    DominatorTree *DT;
    ScalarEvolution *SE;
    bool Changed;

    /// IVsByStride - Keep track of all IVs that have been inserted for a
    /// particular stride.
    std::map<const SCEV *, IVsOfOneStride> IVsByStride;

    /// StrideNoReuse - Keep track of all the strides whose ivs cannot be
    /// reused (nor should they be rewritten to reuse other strides).
    SmallSet<const SCEV *, 4> StrideNoReuse;

    /// DeadInsts - Keep track of instructions we may have made dead, so that
    /// we can remove them after we are done working.
    SmallVector<WeakVH, 16> DeadInsts;

    /// TLI - Keep a pointer of a TargetLowering to consult for determining
    /// transformation profitability.
    const TargetLowering *TLI;

  public:
    static char ID; // Pass ID, replacement for typeid
    explicit LoopStrengthReduce(const TargetLowering *tli = NULL) : 
      LoopPass(&ID), TLI(tli) {
    }

    bool runOnLoop(Loop *L, LPPassManager &LPM);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      // We split critical edges, so we change the CFG.  However, we do update
      // many analyses if they are around.
      AU.addPreservedID(LoopSimplifyID);
      AU.addPreserved<LoopInfo>();
      AU.addPreserved<DominanceFrontier>();
      AU.addPreserved<DominatorTree>();

      AU.addRequiredID(LoopSimplifyID);
      AU.addRequired<LoopInfo>();
      AU.addRequired<DominatorTree>();
      AU.addRequired<ScalarEvolution>();
      AU.addPreserved<ScalarEvolution>();
      AU.addRequired<IVUsers>();
      AU.addPreserved<IVUsers>();
    }

  private:
    ICmpInst *ChangeCompareStride(Loop *L, ICmpInst *Cond,
                                  IVStrideUse* &CondUse,
                                  const SCEV *const *  &CondStride);

    void OptimizeIndvars(Loop *L);
    void OptimizeLoopCountIV(Loop *L);
    void OptimizeLoopTermCond(Loop *L);

    /// OptimizeShadowIV - If IV is used in a int-to-float cast
    /// inside the loop then try to eliminate the cast opeation.
    void OptimizeShadowIV(Loop *L);

    /// OptimizeMax - Rewrite the loop's terminating condition
    /// if it uses a max computation.
    ICmpInst *OptimizeMax(Loop *L, ICmpInst *Cond,
                          IVStrideUse* &CondUse);

    bool FindIVUserForCond(ICmpInst *Cond, IVStrideUse *&CondUse,
                           const SCEV *const * &CondStride);
    bool RequiresTypeConversion(const Type *Ty, const Type *NewTy);
    const SCEV *CheckForIVReuse(bool, bool, bool, const SCEV *const&,
                             IVExpr&, const Type*,
                             const std::vector<BasedUser>& UsersToProcess);
    bool ValidScale(bool, int64_t,
                    const std::vector<BasedUser>& UsersToProcess);
    bool ValidOffset(bool, int64_t, int64_t,
                     const std::vector<BasedUser>& UsersToProcess);
    const SCEV *CollectIVUsers(const SCEV *const &Stride,
                              IVUsersOfOneStride &Uses,
                              Loop *L,
                              bool &AllUsesAreAddresses,
                              bool &AllUsesAreOutsideLoop,
                              std::vector<BasedUser> &UsersToProcess);
    bool ShouldUseFullStrengthReductionMode(
                                const std::vector<BasedUser> &UsersToProcess,
                                const Loop *L,
                                bool AllUsesAreAddresses,
                                const SCEV *Stride);
    void PrepareToStrengthReduceFully(
                             std::vector<BasedUser> &UsersToProcess,
                             const SCEV *Stride,
                             const SCEV *CommonExprs,
                             const Loop *L,
                             SCEVExpander &PreheaderRewriter);
    void PrepareToStrengthReduceFromSmallerStride(
                                         std::vector<BasedUser> &UsersToProcess,
                                         Value *CommonBaseV,
                                         const IVExpr &ReuseIV,
                                         Instruction *PreInsertPt);
    void PrepareToStrengthReduceWithNewPhi(
                                  std::vector<BasedUser> &UsersToProcess,
                                  const SCEV *Stride,
                                  const SCEV *CommonExprs,
                                  Value *CommonBaseV,
                                  Instruction *IVIncInsertPt,
                                  const Loop *L,
                                  SCEVExpander &PreheaderRewriter);
    void StrengthReduceStridedIVUsers(const SCEV *const &Stride,
                                      IVUsersOfOneStride &Uses,
                                      Loop *L);
    void DeleteTriviallyDeadInstructions();
  };
}

char LoopStrengthReduce::ID = 0;
static RegisterPass<LoopStrengthReduce>
X("loop-reduce", "Loop Strength Reduction");

Pass *llvm::createLoopStrengthReducePass(const TargetLowering *TLI) {
  return new LoopStrengthReduce(TLI);
}

/// DeleteTriviallyDeadInstructions - If any of the instructions is the
/// specified set are trivially dead, delete them and see if this makes any of
/// their operands subsequently dead.
void LoopStrengthReduce::DeleteTriviallyDeadInstructions() {
  if (DeadInsts.empty()) return;
  
  while (!DeadInsts.empty()) {
    Instruction *I = dyn_cast_or_null<Instruction>(DeadInsts.back());
    DeadInsts.pop_back();
    
    if (I == 0 || !isInstructionTriviallyDead(I))
      continue;

    for (User::op_iterator OI = I->op_begin(), E = I->op_end(); OI != E; ++OI) {
      if (Instruction *U = dyn_cast<Instruction>(*OI)) {
        *OI = 0;
        if (U->use_empty())
          DeadInsts.push_back(U);
      }
    }
    
    I->eraseFromParent();
    Changed = true;
  }
}

/// containsAddRecFromDifferentLoop - Determine whether expression S involves a 
/// subexpression that is an AddRec from a loop other than L.  An outer loop 
/// of L is OK, but not an inner loop nor a disjoint loop.
static bool containsAddRecFromDifferentLoop(const SCEV *S, Loop *L) {
  // This is very common, put it first.
  if (isa<SCEVConstant>(S))
    return false;
  if (const SCEVCommutativeExpr *AE = dyn_cast<SCEVCommutativeExpr>(S)) {
    for (unsigned int i=0; i< AE->getNumOperands(); i++)
      if (containsAddRecFromDifferentLoop(AE->getOperand(i), L))
        return true;
    return false;
  }
  if (const SCEVAddRecExpr *AE = dyn_cast<SCEVAddRecExpr>(S)) {
    if (const Loop *newLoop = AE->getLoop()) {
      if (newLoop == L)
        return false;
      // if newLoop is an outer loop of L, this is OK.
      if (!LoopInfo::isNotAlreadyContainedIn(L, newLoop))
        return false;
    }
    return true;
  }
  if (const SCEVUDivExpr *DE = dyn_cast<SCEVUDivExpr>(S))
    return containsAddRecFromDifferentLoop(DE->getLHS(), L) ||
           containsAddRecFromDifferentLoop(DE->getRHS(), L);
#if 0
  // SCEVSDivExpr has been backed out temporarily, but will be back; we'll 
  // need this when it is.
  if (const SCEVSDivExpr *DE = dyn_cast<SCEVSDivExpr>(S))
    return containsAddRecFromDifferentLoop(DE->getLHS(), L) ||
           containsAddRecFromDifferentLoop(DE->getRHS(), L);
#endif
  if (const SCEVCastExpr *CE = dyn_cast<SCEVCastExpr>(S))
    return containsAddRecFromDifferentLoop(CE->getOperand(), L);
  return false;
}

/// isAddressUse - Returns true if the specified instruction is using the
/// specified value as an address.
static bool isAddressUse(Instruction *Inst, Value *OperandVal) {
  bool isAddress = isa<LoadInst>(Inst);
  if (StoreInst *SI = dyn_cast<StoreInst>(Inst)) {
    if (SI->getOperand(1) == OperandVal)
      isAddress = true;
  } else if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(Inst)) {
    // Addressing modes can also be folded into prefetches and a variety
    // of intrinsics.
    switch (II->getIntrinsicID()) {
      default: break;
      case Intrinsic::prefetch:
      case Intrinsic::x86_sse2_loadu_dq:
      case Intrinsic::x86_sse2_loadu_pd:
      case Intrinsic::x86_sse_loadu_ps:
      case Intrinsic::x86_sse_storeu_ps:
      case Intrinsic::x86_sse2_storeu_pd:
      case Intrinsic::x86_sse2_storeu_dq:
      case Intrinsic::x86_sse2_storel_dq:
        if (II->getOperand(1) == OperandVal)
          isAddress = true;
        break;
    }
  }
  return isAddress;
}

/// getAccessType - Return the type of the memory being accessed.
static const Type *getAccessType(const Instruction *Inst) {
  const Type *AccessTy = Inst->getType();
  if (const StoreInst *SI = dyn_cast<StoreInst>(Inst))
    AccessTy = SI->getOperand(0)->getType();
  else if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(Inst)) {
    // Addressing modes can also be folded into prefetches and a variety
    // of intrinsics.
    switch (II->getIntrinsicID()) {
    default: break;
    case Intrinsic::x86_sse_storeu_ps:
    case Intrinsic::x86_sse2_storeu_pd:
    case Intrinsic::x86_sse2_storeu_dq:
    case Intrinsic::x86_sse2_storel_dq:
      AccessTy = II->getOperand(1)->getType();
      break;
    }
  }
  return AccessTy;
}

namespace {
  /// BasedUser - For a particular base value, keep information about how we've
  /// partitioned the expression so far.
  struct BasedUser {
    /// SE - The current ScalarEvolution object.
    ScalarEvolution *SE;

    /// Base - The Base value for the PHI node that needs to be inserted for
    /// this use.  As the use is processed, information gets moved from this
    /// field to the Imm field (below).  BasedUser values are sorted by this
    /// field.
    const SCEV *Base;
    
    /// Inst - The instruction using the induction variable.
    Instruction *Inst;

    /// OperandValToReplace - The operand value of Inst to replace with the
    /// EmittedBase.
    Value *OperandValToReplace;

    /// Imm - The immediate value that should be added to the base immediately
    /// before Inst, because it will be folded into the imm field of the
    /// instruction.  This is also sometimes used for loop-variant values that
    /// must be added inside the loop.
    const SCEV *Imm;

    /// Phi - The induction variable that performs the striding that
    /// should be used for this user.
    PHINode *Phi;

    // isUseOfPostIncrementedValue - True if this should use the
    // post-incremented version of this IV, not the preincremented version.
    // This can only be set in special cases, such as the terminating setcc
    // instruction for a loop and uses outside the loop that are dominated by
    // the loop.
    bool isUseOfPostIncrementedValue;
    
    BasedUser(IVStrideUse &IVSU, ScalarEvolution *se)
      : SE(se), Base(IVSU.getOffset()), Inst(IVSU.getUser()),
        OperandValToReplace(IVSU.getOperandValToReplace()),
        Imm(SE->getIntegerSCEV(0, Base->getType())), 
        isUseOfPostIncrementedValue(IVSU.isUseOfPostIncrementedValue()) {}

    // Once we rewrite the code to insert the new IVs we want, update the
    // operands of Inst to use the new expression 'NewBase', with 'Imm' added
    // to it.
    void RewriteInstructionToUseNewBase(const SCEV *const &NewBase,
                                        Instruction *InsertPt,
                                       SCEVExpander &Rewriter, Loop *L, Pass *P,
                                        LoopInfo &LI,
                                        SmallVectorImpl<WeakVH> &DeadInsts);
    
    Value *InsertCodeForBaseAtPosition(const SCEV *const &NewBase, 
                                       const Type *Ty,
                                       SCEVExpander &Rewriter,
                                       Instruction *IP, Loop *L,
                                       LoopInfo &LI);
    void dump() const;
  };
}

void BasedUser::dump() const {
  errs() << " Base=" << *Base;
  errs() << " Imm=" << *Imm;
  errs() << "   Inst: " << *Inst;
}

Value *BasedUser::InsertCodeForBaseAtPosition(const SCEV *const &NewBase, 
                                              const Type *Ty,
                                              SCEVExpander &Rewriter,
                                              Instruction *IP, Loop *L,
                                              LoopInfo &LI) {
  // Figure out where we *really* want to insert this code.  In particular, if
  // the user is inside of a loop that is nested inside of L, we really don't
  // want to insert this expression before the user, we'd rather pull it out as
  // many loops as possible.
  Instruction *BaseInsertPt = IP;
  
  // Figure out the most-nested loop that IP is in.
  Loop *InsertLoop = LI.getLoopFor(IP->getParent());
  
  // If InsertLoop is not L, and InsertLoop is nested inside of L, figure out
  // the preheader of the outer-most loop where NewBase is not loop invariant.
  if (L->contains(IP->getParent()))
    while (InsertLoop && NewBase->isLoopInvariant(InsertLoop)) {
      BaseInsertPt = InsertLoop->getLoopPreheader()->getTerminator();
      InsertLoop = InsertLoop->getParentLoop();
    }
  
  Value *Base = Rewriter.expandCodeFor(NewBase, 0, BaseInsertPt);

  const SCEV *NewValSCEV = SE->getUnknown(Base);

  // Always emit the immediate into the same block as the user.
  NewValSCEV = SE->getAddExpr(NewValSCEV, Imm);

  return Rewriter.expandCodeFor(NewValSCEV, Ty, IP);
}


// Once we rewrite the code to insert the new IVs we want, update the
// operands of Inst to use the new expression 'NewBase', with 'Imm' added
// to it. NewBasePt is the last instruction which contributes to the
// value of NewBase in the case that it's a diffferent instruction from
// the PHI that NewBase is computed from, or null otherwise.
//
void BasedUser::RewriteInstructionToUseNewBase(const SCEV *const &NewBase,
                                               Instruction *NewBasePt,
                                      SCEVExpander &Rewriter, Loop *L, Pass *P,
                                      LoopInfo &LI,
                                      SmallVectorImpl<WeakVH> &DeadInsts) {
  if (!isa<PHINode>(Inst)) {
    // By default, insert code at the user instruction.
    BasicBlock::iterator InsertPt = Inst;
    
    // However, if the Operand is itself an instruction, the (potentially
    // complex) inserted code may be shared by many users.  Because of this, we
    // want to emit code for the computation of the operand right before its old
    // computation.  This is usually safe, because we obviously used to use the
    // computation when it was computed in its current block.  However, in some
    // cases (e.g. use of a post-incremented induction variable) the NewBase
    // value will be pinned to live somewhere after the original computation.
    // In this case, we have to back off.
    //
    // If this is a use outside the loop (which means after, since it is based
    // on a loop indvar) we use the post-incremented value, so that we don't
    // artificially make the preinc value live out the bottom of the loop. 
    if (!isUseOfPostIncrementedValue && L->contains(Inst->getParent())) {
      if (NewBasePt && isa<PHINode>(OperandValToReplace)) {
        InsertPt = NewBasePt;
        ++InsertPt;
      } else if (Instruction *OpInst
                 = dyn_cast<Instruction>(OperandValToReplace)) {
        InsertPt = OpInst;
        while (isa<PHINode>(InsertPt)) ++InsertPt;
      }
    }
    Value *NewVal = InsertCodeForBaseAtPosition(NewBase,
                                                OperandValToReplace->getType(),
                                                Rewriter, InsertPt, L, LI);
    // Replace the use of the operand Value with the new Phi we just created.
    Inst->replaceUsesOfWith(OperandValToReplace, NewVal);

    DEBUG(errs() << "      Replacing with ");
    DEBUG(WriteAsOperand(errs(), NewVal, /*PrintType=*/false));
    DEBUG(errs() << ", which has value " << *NewBase << " plus IMM "
                 << *Imm << "\n");
    return;
  }

  // PHI nodes are more complex.  We have to insert one copy of the NewBase+Imm
  // expression into each operand block that uses it.  Note that PHI nodes can
  // have multiple entries for the same predecessor.  We use a map to make sure
  // that a PHI node only has a single Value* for each predecessor (which also
  // prevents us from inserting duplicate code in some blocks).
  DenseMap<BasicBlock*, Value*> InsertedCode;
  PHINode *PN = cast<PHINode>(Inst);
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
    if (PN->getIncomingValue(i) == OperandValToReplace) {
      // If the original expression is outside the loop, put the replacement
      // code in the same place as the original expression,
      // which need not be an immediate predecessor of this PHI.  This way we 
      // need only one copy of it even if it is referenced multiple times in
      // the PHI.  We don't do this when the original expression is inside the
      // loop because multiple copies sometimes do useful sinking of code in
      // that case(?).
      Instruction *OldLoc = dyn_cast<Instruction>(OperandValToReplace);
      BasicBlock *PHIPred = PN->getIncomingBlock(i);
      if (L->contains(OldLoc->getParent())) {
        // If this is a critical edge, split the edge so that we do not insert
        // the code on all predecessor/successor paths.  We do this unless this
        // is the canonical backedge for this loop, as this can make some
        // inserted code be in an illegal position.
        if (e != 1 && PHIPred->getTerminator()->getNumSuccessors() > 1 &&
            !isa<IndirectBrInst>(PHIPred->getTerminator()) &&
            (PN->getParent() != L->getHeader() || !L->contains(PHIPred))) {

          // First step, split the critical edge.
          BasicBlock *NewBB = SplitCriticalEdge(PHIPred, PN->getParent(),
                                                P, false);

          // Next step: move the basic block.  In particular, if the PHI node
          // is outside of the loop, and PredTI is in the loop, we want to
          // move the block to be immediately before the PHI block, not
          // immediately after PredTI.
          if (L->contains(PHIPred) && !L->contains(PN->getParent()))
            NewBB->moveBefore(PN->getParent());

          // Splitting the edge can reduce the number of PHI entries we have.
          e = PN->getNumIncomingValues();
          PHIPred = NewBB;
          i = PN->getBasicBlockIndex(PHIPred);
        }
      }
      Value *&Code = InsertedCode[PHIPred];
      if (!Code) {
        // Insert the code into the end of the predecessor block.
        Instruction *InsertPt = (L->contains(OldLoc->getParent())) ?
                                PHIPred->getTerminator() :
                                OldLoc->getParent()->getTerminator();
        Code = InsertCodeForBaseAtPosition(NewBase, PN->getType(),
                                           Rewriter, InsertPt, L, LI);

        DEBUG(errs() << "      Changing PHI use to ");
        DEBUG(WriteAsOperand(errs(), Code, /*PrintType=*/false));
        DEBUG(errs() << ", which has value " << *NewBase << " plus IMM "
                     << *Imm << "\n");
      }

      // Replace the use of the operand Value with the new Phi we just created.
      PN->setIncomingValue(i, Code);
      Rewriter.clear();
    }
  }

  // PHI node might have become a constant value after SplitCriticalEdge.
  DeadInsts.push_back(Inst);
}


/// fitsInAddressMode - Return true if V can be subsumed within an addressing
/// mode, and does not need to be put in a register first.
static bool fitsInAddressMode(const SCEV *const &V, const Type *AccessTy,
                             const TargetLowering *TLI, bool HasBaseReg) {
  if (const SCEVConstant *SC = dyn_cast<SCEVConstant>(V)) {
    int64_t VC = SC->getValue()->getSExtValue();
    if (TLI) {
      TargetLowering::AddrMode AM;
      AM.BaseOffs = VC;
      AM.HasBaseReg = HasBaseReg;
      return TLI->isLegalAddressingMode(AM, AccessTy);
    } else {
      // Defaults to PPC. PPC allows a sign-extended 16-bit immediate field.
      return (VC > -(1 << 16) && VC < (1 << 16)-1);
    }
  }

  if (const SCEVUnknown *SU = dyn_cast<SCEVUnknown>(V))
    if (GlobalValue *GV = dyn_cast<GlobalValue>(SU->getValue())) {
      if (TLI) {
        TargetLowering::AddrMode AM;
        AM.BaseGV = GV;
        AM.HasBaseReg = HasBaseReg;
        return TLI->isLegalAddressingMode(AM, AccessTy);
      } else {
        // Default: assume global addresses are not legal.
      }
    }

  return false;
}

/// MoveLoopVariantsToImmediateField - Move any subexpressions from Val that are
/// loop varying to the Imm operand.
static void MoveLoopVariantsToImmediateField(const SCEV *&Val, const SCEV *&Imm,
                                             Loop *L, ScalarEvolution *SE) {
  if (Val->isLoopInvariant(L)) return;  // Nothing to do.
  
  if (const SCEVAddExpr *SAE = dyn_cast<SCEVAddExpr>(Val)) {
    SmallVector<const SCEV *, 4> NewOps;
    NewOps.reserve(SAE->getNumOperands());
    
    for (unsigned i = 0; i != SAE->getNumOperands(); ++i)
      if (!SAE->getOperand(i)->isLoopInvariant(L)) {
        // If this is a loop-variant expression, it must stay in the immediate
        // field of the expression.
        Imm = SE->getAddExpr(Imm, SAE->getOperand(i));
      } else {
        NewOps.push_back(SAE->getOperand(i));
      }

    if (NewOps.empty())
      Val = SE->getIntegerSCEV(0, Val->getType());
    else
      Val = SE->getAddExpr(NewOps);
  } else if (const SCEVAddRecExpr *SARE = dyn_cast<SCEVAddRecExpr>(Val)) {
    // Try to pull immediates out of the start value of nested addrec's.
    const SCEV *Start = SARE->getStart();
    MoveLoopVariantsToImmediateField(Start, Imm, L, SE);
    
    SmallVector<const SCEV *, 4> Ops(SARE->op_begin(), SARE->op_end());
    Ops[0] = Start;
    Val = SE->getAddRecExpr(Ops, SARE->getLoop());
  } else {
    // Otherwise, all of Val is variant, move the whole thing over.
    Imm = SE->getAddExpr(Imm, Val);
    Val = SE->getIntegerSCEV(0, Val->getType());
  }
}


/// MoveImmediateValues - Look at Val, and pull out any additions of constants
/// that can fit into the immediate field of instructions in the target.
/// Accumulate these immediate values into the Imm value.
static void MoveImmediateValues(const TargetLowering *TLI,
                                const Type *AccessTy,
                                const SCEV *&Val, const SCEV *&Imm,
                                bool isAddress, Loop *L,
                                ScalarEvolution *SE) {
  if (const SCEVAddExpr *SAE = dyn_cast<SCEVAddExpr>(Val)) {
    SmallVector<const SCEV *, 4> NewOps;
    NewOps.reserve(SAE->getNumOperands());
    
    for (unsigned i = 0; i != SAE->getNumOperands(); ++i) {
      const SCEV *NewOp = SAE->getOperand(i);
      MoveImmediateValues(TLI, AccessTy, NewOp, Imm, isAddress, L, SE);
      
      if (!NewOp->isLoopInvariant(L)) {
        // If this is a loop-variant expression, it must stay in the immediate
        // field of the expression.
        Imm = SE->getAddExpr(Imm, NewOp);
      } else {
        NewOps.push_back(NewOp);
      }
    }

    if (NewOps.empty())
      Val = SE->getIntegerSCEV(0, Val->getType());
    else
      Val = SE->getAddExpr(NewOps);
    return;
  } else if (const SCEVAddRecExpr *SARE = dyn_cast<SCEVAddRecExpr>(Val)) {
    // Try to pull immediates out of the start value of nested addrec's.
    const SCEV *Start = SARE->getStart();
    MoveImmediateValues(TLI, AccessTy, Start, Imm, isAddress, L, SE);
    
    if (Start != SARE->getStart()) {
      SmallVector<const SCEV *, 4> Ops(SARE->op_begin(), SARE->op_end());
      Ops[0] = Start;
      Val = SE->getAddRecExpr(Ops, SARE->getLoop());
    }
    return;
  } else if (const SCEVMulExpr *SME = dyn_cast<SCEVMulExpr>(Val)) {
    // Transform "8 * (4 + v)" -> "32 + 8*V" if "32" fits in the immed field.
    if (isAddress &&
        fitsInAddressMode(SME->getOperand(0), AccessTy, TLI, false) &&
        SME->getNumOperands() == 2 && SME->isLoopInvariant(L)) {

      const SCEV *SubImm = SE->getIntegerSCEV(0, Val->getType());
      const SCEV *NewOp = SME->getOperand(1);
      MoveImmediateValues(TLI, AccessTy, NewOp, SubImm, isAddress, L, SE);
      
      // If we extracted something out of the subexpressions, see if we can 
      // simplify this!
      if (NewOp != SME->getOperand(1)) {
        // Scale SubImm up by "8".  If the result is a target constant, we are
        // good.
        SubImm = SE->getMulExpr(SubImm, SME->getOperand(0));
        if (fitsInAddressMode(SubImm, AccessTy, TLI, false)) {
          // Accumulate the immediate.
          Imm = SE->getAddExpr(Imm, SubImm);
          
          // Update what is left of 'Val'.
          Val = SE->getMulExpr(SME->getOperand(0), NewOp);
          return;
        }
      }
    }
  }

  // Loop-variant expressions must stay in the immediate field of the
  // expression.
  if ((isAddress && fitsInAddressMode(Val, AccessTy, TLI, false)) ||
      !Val->isLoopInvariant(L)) {
    Imm = SE->getAddExpr(Imm, Val);
    Val = SE->getIntegerSCEV(0, Val->getType());
    return;
  }

  // Otherwise, no immediates to move.
}

static void MoveImmediateValues(const TargetLowering *TLI,
                                Instruction *User,
                                const SCEV *&Val, const SCEV *&Imm,
                                bool isAddress, Loop *L,
                                ScalarEvolution *SE) {
  const Type *AccessTy = getAccessType(User);
  MoveImmediateValues(TLI, AccessTy, Val, Imm, isAddress, L, SE);
}

/// SeparateSubExprs - Decompose Expr into all of the subexpressions that are
/// added together.  This is used to reassociate common addition subexprs
/// together for maximal sharing when rewriting bases.
static void SeparateSubExprs(SmallVector<const SCEV *, 16> &SubExprs,
                             const SCEV *Expr,
                             ScalarEvolution *SE) {
  if (const SCEVAddExpr *AE = dyn_cast<SCEVAddExpr>(Expr)) {
    for (unsigned j = 0, e = AE->getNumOperands(); j != e; ++j)
      SeparateSubExprs(SubExprs, AE->getOperand(j), SE);
  } else if (const SCEVAddRecExpr *SARE = dyn_cast<SCEVAddRecExpr>(Expr)) {
    const SCEV *Zero = SE->getIntegerSCEV(0, Expr->getType());
    if (SARE->getOperand(0) == Zero) {
      SubExprs.push_back(Expr);
    } else {
      // Compute the addrec with zero as its base.
      SmallVector<const SCEV *, 4> Ops(SARE->op_begin(), SARE->op_end());
      Ops[0] = Zero;   // Start with zero base.
      SubExprs.push_back(SE->getAddRecExpr(Ops, SARE->getLoop()));
      

      SeparateSubExprs(SubExprs, SARE->getOperand(0), SE);
    }
  } else if (!Expr->isZero()) {
    // Do not add zero.
    SubExprs.push_back(Expr);
  }
}

// This is logically local to the following function, but C++ says we have 
// to make it file scope.
struct SubExprUseData { unsigned Count; bool notAllUsesAreFree; };

/// RemoveCommonExpressionsFromUseBases - Look through all of the Bases of all
/// the Uses, removing any common subexpressions, except that if all such
/// subexpressions can be folded into an addressing mode for all uses inside
/// the loop (this case is referred to as "free" in comments herein) we do
/// not remove anything.  This looks for things like (a+b+c) and
/// (a+c+d) and computes the common (a+c) subexpression.  The common expression
/// is *removed* from the Bases and returned.
static const SCEV *
RemoveCommonExpressionsFromUseBases(std::vector<BasedUser> &Uses,
                                    ScalarEvolution *SE, Loop *L,
                                    const TargetLowering *TLI) {
  unsigned NumUses = Uses.size();

  // Only one use?  This is a very common case, so we handle it specially and
  // cheaply.
  const SCEV *Zero = SE->getIntegerSCEV(0, Uses[0].Base->getType());
  const SCEV *Result = Zero;
  const SCEV *FreeResult = Zero;
  if (NumUses == 1) {
    // If the use is inside the loop, use its base, regardless of what it is:
    // it is clearly shared across all the IV's.  If the use is outside the loop
    // (which means after it) we don't want to factor anything *into* the loop,
    // so just use 0 as the base.
    if (L->contains(Uses[0].Inst->getParent()))
      std::swap(Result, Uses[0].Base);
    return Result;
  }

  // To find common subexpressions, count how many of Uses use each expression.
  // If any subexpressions are used Uses.size() times, they are common.
  // Also track whether all uses of each expression can be moved into an
  // an addressing mode "for free"; such expressions are left within the loop.
  // struct SubExprUseData { unsigned Count; bool notAllUsesAreFree; };
  std::map<const SCEV *, SubExprUseData> SubExpressionUseData;
  
  // UniqueSubExprs - Keep track of all of the subexpressions we see in the
  // order we see them.
  SmallVector<const SCEV *, 16> UniqueSubExprs;

  SmallVector<const SCEV *, 16> SubExprs;
  unsigned NumUsesInsideLoop = 0;
  for (unsigned i = 0; i != NumUses; ++i) {
    // If the user is outside the loop, just ignore it for base computation.
    // Since the user is outside the loop, it must be *after* the loop (if it
    // were before, it could not be based on the loop IV).  We don't want users
    // after the loop to affect base computation of values *inside* the loop,
    // because we can always add their offsets to the result IV after the loop
    // is done, ensuring we get good code inside the loop.
    if (!L->contains(Uses[i].Inst->getParent()))
      continue;
    NumUsesInsideLoop++;
    
    // If the base is zero (which is common), return zero now, there are no
    // CSEs we can find.
    if (Uses[i].Base == Zero) return Zero;

    // If this use is as an address we may be able to put CSEs in the addressing
    // mode rather than hoisting them.
    bool isAddrUse = isAddressUse(Uses[i].Inst, Uses[i].OperandValToReplace);
    // We may need the AccessTy below, but only when isAddrUse, so compute it
    // only in that case.
    const Type *AccessTy = 0;
    if (isAddrUse)
      AccessTy = getAccessType(Uses[i].Inst);

    // Split the expression into subexprs.
    SeparateSubExprs(SubExprs, Uses[i].Base, SE);
    // Add one to SubExpressionUseData.Count for each subexpr present, and
    // if the subexpr is not a valid immediate within an addressing mode use,
    // set SubExpressionUseData.notAllUsesAreFree.  We definitely want to
    // hoist these out of the loop (if they are common to all uses).
    for (unsigned j = 0, e = SubExprs.size(); j != e; ++j) {
      if (++SubExpressionUseData[SubExprs[j]].Count == 1)
        UniqueSubExprs.push_back(SubExprs[j]);
      if (!isAddrUse || !fitsInAddressMode(SubExprs[j], AccessTy, TLI, false))
        SubExpressionUseData[SubExprs[j]].notAllUsesAreFree = true;
    }
    SubExprs.clear();
  }

  // Now that we know how many times each is used, build Result.  Iterate over
  // UniqueSubexprs so that we have a stable ordering.
  for (unsigned i = 0, e = UniqueSubExprs.size(); i != e; ++i) {
    std::map<const SCEV *, SubExprUseData>::iterator I = 
       SubExpressionUseData.find(UniqueSubExprs[i]);
    assert(I != SubExpressionUseData.end() && "Entry not found?");
    if (I->second.Count == NumUsesInsideLoop) { // Found CSE! 
      if (I->second.notAllUsesAreFree)
        Result = SE->getAddExpr(Result, I->first);
      else 
        FreeResult = SE->getAddExpr(FreeResult, I->first);
    } else
      // Remove non-cse's from SubExpressionUseData.
      SubExpressionUseData.erase(I);
  }

  if (FreeResult != Zero) {
    // We have some subexpressions that can be subsumed into addressing
    // modes in every use inside the loop.  However, it's possible that
    // there are so many of them that the combined FreeResult cannot
    // be subsumed, or that the target cannot handle both a FreeResult
    // and a Result in the same instruction (for example because it would
    // require too many registers).  Check this.
    for (unsigned i=0; i<NumUses; ++i) {
      if (!L->contains(Uses[i].Inst->getParent()))
        continue;
      // We know this is an addressing mode use; if there are any uses that
      // are not, FreeResult would be Zero.
      const Type *AccessTy = getAccessType(Uses[i].Inst);
      if (!fitsInAddressMode(FreeResult, AccessTy, TLI, Result!=Zero)) {
        // FIXME:  could split up FreeResult into pieces here, some hoisted
        // and some not.  There is no obvious advantage to this.
        Result = SE->getAddExpr(Result, FreeResult);
        FreeResult = Zero;
        break;
      }
    }
  }

  // If we found no CSE's, return now.
  if (Result == Zero) return Result;
  
  // If we still have a FreeResult, remove its subexpressions from
  // SubExpressionUseData.  This means they will remain in the use Bases.
  if (FreeResult != Zero) {
    SeparateSubExprs(SubExprs, FreeResult, SE);
    for (unsigned j = 0, e = SubExprs.size(); j != e; ++j) {
      std::map<const SCEV *, SubExprUseData>::iterator I = 
         SubExpressionUseData.find(SubExprs[j]);
      SubExpressionUseData.erase(I);
    }
    SubExprs.clear();
  }

  // Otherwise, remove all of the CSE's we found from each of the base values.
  for (unsigned i = 0; i != NumUses; ++i) {
    // Uses outside the loop don't necessarily include the common base, but
    // the final IV value coming into those uses does.  Instead of trying to
    // remove the pieces of the common base, which might not be there,
    // subtract off the base to compensate for this.
    if (!L->contains(Uses[i].Inst->getParent())) {
      Uses[i].Base = SE->getMinusSCEV(Uses[i].Base, Result);
      continue;
    }

    // Split the expression into subexprs.
    SeparateSubExprs(SubExprs, Uses[i].Base, SE);

    // Remove any common subexpressions.
    for (unsigned j = 0, e = SubExprs.size(); j != e; ++j)
      if (SubExpressionUseData.count(SubExprs[j])) {
        SubExprs.erase(SubExprs.begin()+j);
        --j; --e;
      }
    
    // Finally, add the non-shared expressions together.
    if (SubExprs.empty())
      Uses[i].Base = Zero;
    else
      Uses[i].Base = SE->getAddExpr(SubExprs);
    SubExprs.clear();
  }
 
  return Result;
}

/// ValidScale - Check whether the given Scale is valid for all loads and 
/// stores in UsersToProcess.
///
bool LoopStrengthReduce::ValidScale(bool HasBaseReg, int64_t Scale,
                               const std::vector<BasedUser>& UsersToProcess) {
  if (!TLI)
    return true;

  for (unsigned i = 0, e = UsersToProcess.size(); i!=e; ++i) {
    // If this is a load or other access, pass the type of the access in.
    const Type *AccessTy =
        Type::getVoidTy(UsersToProcess[i].Inst->getContext());
    if (isAddressUse(UsersToProcess[i].Inst,
                     UsersToProcess[i].OperandValToReplace))
      AccessTy = getAccessType(UsersToProcess[i].Inst);
    else if (isa<PHINode>(UsersToProcess[i].Inst))
      continue;
    
    TargetLowering::AddrMode AM;
    if (const SCEVConstant *SC = dyn_cast<SCEVConstant>(UsersToProcess[i].Imm))
      AM.BaseOffs = SC->getValue()->getSExtValue();
    AM.HasBaseReg = HasBaseReg || !UsersToProcess[i].Base->isZero();
    AM.Scale = Scale;

    // If load[imm+r*scale] is illegal, bail out.
    if (!TLI->isLegalAddressingMode(AM, AccessTy))
      return false;
  }
  return true;
}

/// ValidOffset - Check whether the given Offset is valid for all loads and
/// stores in UsersToProcess.
///
bool LoopStrengthReduce::ValidOffset(bool HasBaseReg,
                               int64_t Offset,
                               int64_t Scale,
                               const std::vector<BasedUser>& UsersToProcess) {
  if (!TLI)
    return true;

  for (unsigned i=0, e = UsersToProcess.size(); i!=e; ++i) {
    // If this is a load or other access, pass the type of the access in.
    const Type *AccessTy =
        Type::getVoidTy(UsersToProcess[i].Inst->getContext());
    if (isAddressUse(UsersToProcess[i].Inst,
                     UsersToProcess[i].OperandValToReplace))
      AccessTy = getAccessType(UsersToProcess[i].Inst);
    else if (isa<PHINode>(UsersToProcess[i].Inst))
      continue;

    TargetLowering::AddrMode AM;
    if (const SCEVConstant *SC = dyn_cast<SCEVConstant>(UsersToProcess[i].Imm))
      AM.BaseOffs = SC->getValue()->getSExtValue();
    AM.BaseOffs = (uint64_t)AM.BaseOffs + (uint64_t)Offset;
    AM.HasBaseReg = HasBaseReg || !UsersToProcess[i].Base->isZero();
    AM.Scale = Scale;

    // If load[imm+r*scale] is illegal, bail out.
    if (!TLI->isLegalAddressingMode(AM, AccessTy))
      return false;
  }
  return true;
}

/// RequiresTypeConversion - Returns true if converting Ty1 to Ty2 is not
/// a nop.
bool LoopStrengthReduce::RequiresTypeConversion(const Type *Ty1,
                                                const Type *Ty2) {
  if (Ty1 == Ty2)
    return false;
  Ty1 = SE->getEffectiveSCEVType(Ty1);
  Ty2 = SE->getEffectiveSCEVType(Ty2);
  if (Ty1 == Ty2)
    return false;
  if (Ty1->canLosslesslyBitCastTo(Ty2))
    return false;
  if (TLI && TLI->isTruncateFree(Ty1, Ty2))
    return false;
  return true;
}

/// CheckForIVReuse - Returns the multiple if the stride is the multiple
/// of a previous stride and it is a legal value for the target addressing
/// mode scale component and optional base reg. This allows the users of
/// this stride to be rewritten as prev iv * factor. It returns 0 if no
/// reuse is possible.  Factors can be negative on same targets, e.g. ARM.
///
/// If all uses are outside the loop, we don't require that all multiplies
/// be folded into the addressing mode, nor even that the factor be constant; 
/// a multiply (executed once) outside the loop is better than another IV 
/// within.  Well, usually.
const SCEV *LoopStrengthReduce::CheckForIVReuse(bool HasBaseReg,
                                bool AllUsesAreAddresses,
                                bool AllUsesAreOutsideLoop,
                                const SCEV *const &Stride, 
                                IVExpr &IV, const Type *Ty,
                                const std::vector<BasedUser>& UsersToProcess) {
  if (StrideNoReuse.count(Stride))
    return SE->getIntegerSCEV(0, Stride->getType());

  if (const SCEVConstant *SC = dyn_cast<SCEVConstant>(Stride)) {
    int64_t SInt = SC->getValue()->getSExtValue();
    for (unsigned NewStride = 0, e = IU->StrideOrder.size();
         NewStride != e; ++NewStride) {
      std::map<const SCEV *, IVsOfOneStride>::iterator SI = 
                IVsByStride.find(IU->StrideOrder[NewStride]);
      if (SI == IVsByStride.end() || !isa<SCEVConstant>(SI->first) ||
          StrideNoReuse.count(SI->first))
        continue;
      int64_t SSInt = cast<SCEVConstant>(SI->first)->getValue()->getSExtValue();
      if (SI->first != Stride &&
          (unsigned(abs64(SInt)) < SSInt || (SInt % SSInt) != 0))
        continue;
      int64_t Scale = SInt / SSInt;
      // Check that this stride is valid for all the types used for loads and
      // stores; if it can be used for some and not others, we might as well use
      // the original stride everywhere, since we have to create the IV for it
      // anyway. If the scale is 1, then we don't need to worry about folding
      // multiplications.
      if (Scale == 1 ||
          (AllUsesAreAddresses &&
           ValidScale(HasBaseReg, Scale, UsersToProcess))) {
        // Prefer to reuse an IV with a base of zero.
        for (std::vector<IVExpr>::iterator II = SI->second.IVs.begin(),
               IE = SI->second.IVs.end(); II != IE; ++II)
          // Only reuse previous IV if it would not require a type conversion
          // and if the base difference can be folded.
          if (II->Base->isZero() &&
              !RequiresTypeConversion(II->Base->getType(), Ty)) {
            IV = *II;
            return SE->getIntegerSCEV(Scale, Stride->getType());
          }
        // Otherwise, settle for an IV with a foldable base.
        if (AllUsesAreAddresses)
          for (std::vector<IVExpr>::iterator II = SI->second.IVs.begin(),
                 IE = SI->second.IVs.end(); II != IE; ++II)
            // Only reuse previous IV if it would not require a type conversion
            // and if the base difference can be folded.
            if (SE->getEffectiveSCEVType(II->Base->getType()) ==
                SE->getEffectiveSCEVType(Ty) &&
                isa<SCEVConstant>(II->Base)) {
              int64_t Base =
                cast<SCEVConstant>(II->Base)->getValue()->getSExtValue();
              if (Base > INT32_MIN && Base <= INT32_MAX &&
                  ValidOffset(HasBaseReg, -Base * Scale,
                              Scale, UsersToProcess)) {
                IV = *II;
                return SE->getIntegerSCEV(Scale, Stride->getType());
              }
            }
      }
    }
  } else if (AllUsesAreOutsideLoop) {
    // Accept nonconstant strides here; it is really really right to substitute
    // an existing IV if we can.
    for (unsigned NewStride = 0, e = IU->StrideOrder.size();
         NewStride != e; ++NewStride) {
      std::map<const SCEV *, IVsOfOneStride>::iterator SI = 
                IVsByStride.find(IU->StrideOrder[NewStride]);
      if (SI == IVsByStride.end() || !isa<SCEVConstant>(SI->first))
        continue;
      int64_t SSInt = cast<SCEVConstant>(SI->first)->getValue()->getSExtValue();
      if (SI->first != Stride && SSInt != 1)
        continue;
      for (std::vector<IVExpr>::iterator II = SI->second.IVs.begin(),
             IE = SI->second.IVs.end(); II != IE; ++II)
        // Accept nonzero base here.
        // Only reuse previous IV if it would not require a type conversion.
        if (!RequiresTypeConversion(II->Base->getType(), Ty)) {
          IV = *II;
          return Stride;
        }
    }
    // Special case, old IV is -1*x and this one is x.  Can treat this one as
    // -1*old.
    for (unsigned NewStride = 0, e = IU->StrideOrder.size();
         NewStride != e; ++NewStride) {
      std::map<const SCEV *, IVsOfOneStride>::iterator SI = 
                IVsByStride.find(IU->StrideOrder[NewStride]);
      if (SI == IVsByStride.end()) 
        continue;
      if (const SCEVMulExpr *ME = dyn_cast<SCEVMulExpr>(SI->first))
        if (const SCEVConstant *SC = dyn_cast<SCEVConstant>(ME->getOperand(0)))
          if (Stride == ME->getOperand(1) &&
              SC->getValue()->getSExtValue() == -1LL)
            for (std::vector<IVExpr>::iterator II = SI->second.IVs.begin(),
                   IE = SI->second.IVs.end(); II != IE; ++II)
              // Accept nonzero base here.
              // Only reuse previous IV if it would not require type conversion.
              if (!RequiresTypeConversion(II->Base->getType(), Ty)) {
                IV = *II;
                return SE->getIntegerSCEV(-1LL, Stride->getType());
              }
    }
  }
  return SE->getIntegerSCEV(0, Stride->getType());
}

/// PartitionByIsUseOfPostIncrementedValue - Simple boolean predicate that
/// returns true if Val's isUseOfPostIncrementedValue is true.
static bool PartitionByIsUseOfPostIncrementedValue(const BasedUser &Val) {
  return Val.isUseOfPostIncrementedValue;
}

/// isNonConstantNegative - Return true if the specified scev is negated, but
/// not a constant.
static bool isNonConstantNegative(const SCEV *const &Expr) {
  const SCEVMulExpr *Mul = dyn_cast<SCEVMulExpr>(Expr);
  if (!Mul) return false;
  
  // If there is a constant factor, it will be first.
  const SCEVConstant *SC = dyn_cast<SCEVConstant>(Mul->getOperand(0));
  if (!SC) return false;
  
  // Return true if the value is negative, this matches things like (-42 * V).
  return SC->getValue()->getValue().isNegative();
}

/// CollectIVUsers - Transform our list of users and offsets to a bit more
/// complex table. In this new vector, each 'BasedUser' contains 'Base', the base
/// of the strided accesses, as well as the old information from Uses. We
/// progressively move information from the Base field to the Imm field, until
/// we eventually have the full access expression to rewrite the use.
const SCEV *LoopStrengthReduce::CollectIVUsers(const SCEV *const &Stride,
                                              IVUsersOfOneStride &Uses,
                                              Loop *L,
                                              bool &AllUsesAreAddresses,
                                              bool &AllUsesAreOutsideLoop,
                                       std::vector<BasedUser> &UsersToProcess) {
  // FIXME: Generalize to non-affine IV's.
  if (!Stride->isLoopInvariant(L))
    return SE->getIntegerSCEV(0, Stride->getType());

  UsersToProcess.reserve(Uses.Users.size());
  for (ilist<IVStrideUse>::iterator I = Uses.Users.begin(),
       E = Uses.Users.end(); I != E; ++I) {
    UsersToProcess.push_back(BasedUser(*I, SE));

    // Move any loop variant operands from the offset field to the immediate
    // field of the use, so that we don't try to use something before it is
    // computed.
    MoveLoopVariantsToImmediateField(UsersToProcess.back().Base,
                                     UsersToProcess.back().Imm, L, SE);
    assert(UsersToProcess.back().Base->isLoopInvariant(L) &&
           "Base value is not loop invariant!");
  }

  // We now have a whole bunch of uses of like-strided induction variables, but
  // they might all have different bases.  We want to emit one PHI node for this
  // stride which we fold as many common expressions (between the IVs) into as
  // possible.  Start by identifying the common expressions in the base values 
  // for the strides (e.g. if we have "A+C+B" and "A+B+D" as our bases, find
  // "A+B"), emit it to the preheader, then remove the expression from the
  // UsersToProcess base values.
  const SCEV *CommonExprs =
    RemoveCommonExpressionsFromUseBases(UsersToProcess, SE, L, TLI);

  // Next, figure out what we can represent in the immediate fields of
  // instructions.  If we can represent anything there, move it to the imm
  // fields of the BasedUsers.  We do this so that it increases the commonality
  // of the remaining uses.
  unsigned NumPHI = 0;
  bool HasAddress = false;
  for (unsigned i = 0, e = UsersToProcess.size(); i != e; ++i) {
    // If the user is not in the current loop, this means it is using the exit
    // value of the IV.  Do not put anything in the base, make sure it's all in
    // the immediate field to allow as much factoring as possible.
    if (!L->contains(UsersToProcess[i].Inst->getParent())) {
      UsersToProcess[i].Imm = SE->getAddExpr(UsersToProcess[i].Imm,
                                             UsersToProcess[i].Base);
      UsersToProcess[i].Base = 
        SE->getIntegerSCEV(0, UsersToProcess[i].Base->getType());
    } else {
      // Not all uses are outside the loop.
      AllUsesAreOutsideLoop = false; 

      // Addressing modes can be folded into loads and stores.  Be careful that
      // the store is through the expression, not of the expression though.
      bool isPHI = false;
      bool isAddress = isAddressUse(UsersToProcess[i].Inst,
                                    UsersToProcess[i].OperandValToReplace);
      if (isa<PHINode>(UsersToProcess[i].Inst)) {
        isPHI = true;
        ++NumPHI;
      }

      if (isAddress)
        HasAddress = true;
     
      // If this use isn't an address, then not all uses are addresses.
      if (!isAddress && !isPHI)
        AllUsesAreAddresses = false;
      
      MoveImmediateValues(TLI, UsersToProcess[i].Inst, UsersToProcess[i].Base,
                          UsersToProcess[i].Imm, isAddress, L, SE);
    }
  }

  // If one of the use is a PHI node and all other uses are addresses, still
  // allow iv reuse. Essentially we are trading one constant multiplication
  // for one fewer iv.
  if (NumPHI > 1)
    AllUsesAreAddresses = false;
    
  // There are no in-loop address uses.
  if (AllUsesAreAddresses && (!HasAddress && !AllUsesAreOutsideLoop))
    AllUsesAreAddresses = false;

  return CommonExprs;
}

/// ShouldUseFullStrengthReductionMode - Test whether full strength-reduction
/// is valid and profitable for the given set of users of a stride. In
/// full strength-reduction mode, all addresses at the current stride are
/// strength-reduced all the way down to pointer arithmetic.
///
bool LoopStrengthReduce::ShouldUseFullStrengthReductionMode(
                                   const std::vector<BasedUser> &UsersToProcess,
                                   const Loop *L,
                                   bool AllUsesAreAddresses,
                                   const SCEV *Stride) {
  if (!EnableFullLSRMode)
    return false;

  // The heuristics below aim to avoid increasing register pressure, but
  // fully strength-reducing all the addresses increases the number of
  // add instructions, so don't do this when optimizing for size.
  // TODO: If the loop is large, the savings due to simpler addresses
  // may oughtweight the costs of the extra increment instructions.
  if (L->getHeader()->getParent()->hasFnAttr(Attribute::OptimizeForSize))
    return false;

  // TODO: For now, don't do full strength reduction if there could
  // potentially be greater-stride multiples of the current stride
  // which could reuse the current stride IV.
  if (IU->StrideOrder.back() != Stride)
    return false;

  // Iterate through the uses to find conditions that automatically rule out
  // full-lsr mode.
  for (unsigned i = 0, e = UsersToProcess.size(); i != e; ) {
    const SCEV *Base = UsersToProcess[i].Base;
    const SCEV *Imm = UsersToProcess[i].Imm;
    // If any users have a loop-variant component, they can't be fully
    // strength-reduced.
    if (Imm && !Imm->isLoopInvariant(L))
      return false;
    // If there are to users with the same base and the difference between
    // the two Imm values can't be folded into the address, full
    // strength reduction would increase register pressure.
    do {
      const SCEV *CurImm = UsersToProcess[i].Imm;
      if ((CurImm || Imm) && CurImm != Imm) {
        if (!CurImm) CurImm = SE->getIntegerSCEV(0, Stride->getType());
        if (!Imm)       Imm = SE->getIntegerSCEV(0, Stride->getType());
        const Instruction *Inst = UsersToProcess[i].Inst;
        const Type *AccessTy = getAccessType(Inst);
        const SCEV *Diff = SE->getMinusSCEV(UsersToProcess[i].Imm, Imm);
        if (!Diff->isZero() &&
            (!AllUsesAreAddresses ||
             !fitsInAddressMode(Diff, AccessTy, TLI, /*HasBaseReg=*/true)))
          return false;
      }
    } while (++i != e && Base == UsersToProcess[i].Base);
  }

  // If there's exactly one user in this stride, fully strength-reducing it
  // won't increase register pressure. If it's starting from a non-zero base,
  // it'll be simpler this way.
  if (UsersToProcess.size() == 1 && !UsersToProcess[0].Base->isZero())
    return true;

  // Otherwise, if there are any users in this stride that don't require
  // a register for their base, full strength-reduction will increase
  // register pressure.
  for (unsigned i = 0, e = UsersToProcess.size(); i != e; ++i)
    if (UsersToProcess[i].Base->isZero())
      return false;

  // Otherwise, go for it.
  return true;
}

/// InsertAffinePhi Create and insert a PHI node for an induction variable
/// with the specified start and step values in the specified loop.
///
/// If NegateStride is true, the stride should be negated by using a
/// subtract instead of an add.
///
/// Return the created phi node.
///
static PHINode *InsertAffinePhi(const SCEV *Start, const SCEV *Step,
                                Instruction *IVIncInsertPt,
                                const Loop *L,
                                SCEVExpander &Rewriter) {
  assert(Start->isLoopInvariant(L) && "New PHI start is not loop invariant!");
  assert(Step->isLoopInvariant(L) && "New PHI stride is not loop invariant!");

  BasicBlock *Header = L->getHeader();
  BasicBlock *Preheader = L->getLoopPreheader();
  BasicBlock *LatchBlock = L->getLoopLatch();
  const Type *Ty = Start->getType();
  Ty = Rewriter.SE.getEffectiveSCEVType(Ty);

  PHINode *PN = PHINode::Create(Ty, "lsr.iv", Header->begin());
  PN->addIncoming(Rewriter.expandCodeFor(Start, Ty, Preheader->getTerminator()),
                  Preheader);

  // If the stride is negative, insert a sub instead of an add for the
  // increment.
  bool isNegative = isNonConstantNegative(Step);
  const SCEV *IncAmount = Step;
  if (isNegative)
    IncAmount = Rewriter.SE.getNegativeSCEV(Step);

  // Insert an add instruction right before the terminator corresponding
  // to the back-edge or just before the only use. The location is determined
  // by the caller and passed in as IVIncInsertPt.
  Value *StepV = Rewriter.expandCodeFor(IncAmount, Ty,
                                        Preheader->getTerminator());
  Instruction *IncV;
  if (isNegative) {
    IncV = BinaryOperator::CreateSub(PN, StepV, "lsr.iv.next",
                                     IVIncInsertPt);
  } else {
    IncV = BinaryOperator::CreateAdd(PN, StepV, "lsr.iv.next",
                                     IVIncInsertPt);
  }
  if (!isa<ConstantInt>(StepV)) ++NumVariable;

  PN->addIncoming(IncV, LatchBlock);

  ++NumInserted;
  return PN;
}

static void SortUsersToProcess(std::vector<BasedUser> &UsersToProcess) {
  // We want to emit code for users inside the loop first.  To do this, we
  // rearrange BasedUser so that the entries at the end have
  // isUseOfPostIncrementedValue = false, because we pop off the end of the
  // vector (so we handle them first).
  std::partition(UsersToProcess.begin(), UsersToProcess.end(),
                 PartitionByIsUseOfPostIncrementedValue);

  // Sort this by base, so that things with the same base are handled
  // together.  By partitioning first and stable-sorting later, we are
  // guaranteed that within each base we will pop off users from within the
  // loop before users outside of the loop with a particular base.
  //
  // We would like to use stable_sort here, but we can't.  The problem is that
  // const SCEV *'s don't have a deterministic ordering w.r.t to each other, so
  // we don't have anything to do a '<' comparison on.  Because we think the
  // number of uses is small, do a horrible bubble sort which just relies on
  // ==.
  for (unsigned i = 0, e = UsersToProcess.size(); i != e; ++i) {
    // Get a base value.
    const SCEV *Base = UsersToProcess[i].Base;

    // Compact everything with this base to be consecutive with this one.
    for (unsigned j = i+1; j != e; ++j) {
      if (UsersToProcess[j].Base == Base) {
        std::swap(UsersToProcess[i+1], UsersToProcess[j]);
        ++i;
      }
    }
  }
}

/// PrepareToStrengthReduceFully - Prepare to fully strength-reduce
/// UsersToProcess, meaning lowering addresses all the way down to direct
/// pointer arithmetic.
///
void
LoopStrengthReduce::PrepareToStrengthReduceFully(
                                        std::vector<BasedUser> &UsersToProcess,
                                        const SCEV *Stride,
                                        const SCEV *CommonExprs,
                                        const Loop *L,
                                        SCEVExpander &PreheaderRewriter) {
  DEBUG(errs() << "  Fully reducing all users\n");

  // Rewrite the UsersToProcess records, creating a separate PHI for each
  // unique Base value.
  Instruction *IVIncInsertPt = L->getLoopLatch()->getTerminator();
  for (unsigned i = 0, e = UsersToProcess.size(); i != e; ) {
    // TODO: The uses are grouped by base, but not sorted. We arbitrarily
    // pick the first Imm value here to start with, and adjust it for the
    // other uses.
    const SCEV *Imm = UsersToProcess[i].Imm;
    const SCEV *Base = UsersToProcess[i].Base;
    const SCEV *Start = SE->getAddExpr(CommonExprs, Base, Imm);
    PHINode *Phi = InsertAffinePhi(Start, Stride, IVIncInsertPt, L,
                                   PreheaderRewriter);
    // Loop over all the users with the same base.
    do {
      UsersToProcess[i].Base = SE->getIntegerSCEV(0, Stride->getType());
      UsersToProcess[i].Imm = SE->getMinusSCEV(UsersToProcess[i].Imm, Imm);
      UsersToProcess[i].Phi = Phi;
      assert(UsersToProcess[i].Imm->isLoopInvariant(L) &&
             "ShouldUseFullStrengthReductionMode should reject this!");
    } while (++i != e && Base == UsersToProcess[i].Base);
  }
}

/// FindIVIncInsertPt - Return the location to insert the increment instruction.
/// If the only use if a use of postinc value, (must be the loop termination
/// condition), then insert it just before the use.
static Instruction *FindIVIncInsertPt(std::vector<BasedUser> &UsersToProcess,
                                      const Loop *L) {
  if (UsersToProcess.size() == 1 &&
      UsersToProcess[0].isUseOfPostIncrementedValue &&
      L->contains(UsersToProcess[0].Inst->getParent()))
    return UsersToProcess[0].Inst;
  return L->getLoopLatch()->getTerminator();
}

/// PrepareToStrengthReduceWithNewPhi - Insert a new induction variable for the
/// given users to share.
///
void
LoopStrengthReduce::PrepareToStrengthReduceWithNewPhi(
                                         std::vector<BasedUser> &UsersToProcess,
                                         const SCEV *Stride,
                                         const SCEV *CommonExprs,
                                         Value *CommonBaseV,
                                         Instruction *IVIncInsertPt,
                                         const Loop *L,
                                         SCEVExpander &PreheaderRewriter) {
  DEBUG(errs() << "  Inserting new PHI:\n");

  PHINode *Phi = InsertAffinePhi(SE->getUnknown(CommonBaseV),
                                 Stride, IVIncInsertPt, L,
                                 PreheaderRewriter);

  // Remember this in case a later stride is multiple of this.
  IVsByStride[Stride].addIV(Stride, CommonExprs, Phi);

  // All the users will share this new IV.
  for (unsigned i = 0, e = UsersToProcess.size(); i != e; ++i)
    UsersToProcess[i].Phi = Phi;

  DEBUG(errs() << "    IV=");
  DEBUG(WriteAsOperand(errs(), Phi, /*PrintType=*/false));
  DEBUG(errs() << "\n");
}

/// PrepareToStrengthReduceFromSmallerStride - Prepare for the given users to
/// reuse an induction variable with a stride that is a factor of the current
/// induction variable.
///
void
LoopStrengthReduce::PrepareToStrengthReduceFromSmallerStride(
                                         std::vector<BasedUser> &UsersToProcess,
                                         Value *CommonBaseV,
                                         const IVExpr &ReuseIV,
                                         Instruction *PreInsertPt) {
  DEBUG(errs() << "  Rewriting in terms of existing IV of STRIDE "
               << *ReuseIV.Stride << " and BASE " << *ReuseIV.Base << "\n");

  // All the users will share the reused IV.
  for (unsigned i = 0, e = UsersToProcess.size(); i != e; ++i)
    UsersToProcess[i].Phi = ReuseIV.PHI;

  Constant *C = dyn_cast<Constant>(CommonBaseV);
  if (C &&
      (!C->isNullValue() &&
       !fitsInAddressMode(SE->getUnknown(CommonBaseV), CommonBaseV->getType(),
                         TLI, false)))
    // We want the common base emitted into the preheader! This is just
    // using cast as a copy so BitCast (no-op cast) is appropriate
    CommonBaseV = new BitCastInst(CommonBaseV, CommonBaseV->getType(),
                                  "commonbase", PreInsertPt);
}

static bool IsImmFoldedIntoAddrMode(GlobalValue *GV, int64_t Offset,
                                    const Type *AccessTy,
                                   std::vector<BasedUser> &UsersToProcess,
                                   const TargetLowering *TLI) {
  SmallVector<Instruction*, 16> AddrModeInsts;
  for (unsigned i = 0, e = UsersToProcess.size(); i != e; ++i) {
    if (UsersToProcess[i].isUseOfPostIncrementedValue)
      continue;
    ExtAddrMode AddrMode =
      AddressingModeMatcher::Match(UsersToProcess[i].OperandValToReplace,
                                   AccessTy, UsersToProcess[i].Inst,
                                   AddrModeInsts, *TLI);
    if (GV && GV != AddrMode.BaseGV)
      return false;
    if (Offset && !AddrMode.BaseOffs)
      // FIXME: How to accurate check it's immediate offset is folded.
      return false;
    AddrModeInsts.clear();
  }
  return true;
}

/// StrengthReduceStridedIVUsers - Strength reduce all of the users of a single
/// stride of IV.  All of the users may have different starting values, and this
/// may not be the only stride.
void LoopStrengthReduce::StrengthReduceStridedIVUsers(const SCEV *const &Stride,
                                                      IVUsersOfOneStride &Uses,
                                                      Loop *L) {
  // If all the users are moved to another stride, then there is nothing to do.
  if (Uses.Users.empty())
    return;

  // Keep track if every use in UsersToProcess is an address. If they all are,
  // we may be able to rewrite the entire collection of them in terms of a
  // smaller-stride IV.
  bool AllUsesAreAddresses = true;

  // Keep track if every use of a single stride is outside the loop.  If so,
  // we want to be more aggressive about reusing a smaller-stride IV; a
  // multiply outside the loop is better than another IV inside.  Well, usually.
  bool AllUsesAreOutsideLoop = true;

  // Transform our list of users and offsets to a bit more complex table.  In
  // this new vector, each 'BasedUser' contains 'Base' the base of the
  // strided accessas well as the old information from Uses.  We progressively
  // move information from the Base field to the Imm field, until we eventually
  // have the full access expression to rewrite the use.
  std::vector<BasedUser> UsersToProcess;
  const SCEV *CommonExprs = CollectIVUsers(Stride, Uses, L, AllUsesAreAddresses,
                                          AllUsesAreOutsideLoop,
                                          UsersToProcess);

  // Sort the UsersToProcess array so that users with common bases are
  // next to each other.
  SortUsersToProcess(UsersToProcess);

  // If we managed to find some expressions in common, we'll need to carry
  // their value in a register and add it in for each use. This will take up
  // a register operand, which potentially restricts what stride values are
  // valid.
  bool HaveCommonExprs = !CommonExprs->isZero();
  const Type *ReplacedTy = CommonExprs->getType();

  // If all uses are addresses, consider sinking the immediate part of the
  // common expression back into uses if they can fit in the immediate fields.
  if (TLI && HaveCommonExprs && AllUsesAreAddresses) {
    const SCEV *NewCommon = CommonExprs;
    const SCEV *Imm = SE->getIntegerSCEV(0, ReplacedTy);
    MoveImmediateValues(TLI, Type::getVoidTy(
                        L->getLoopPreheader()->getContext()),
                        NewCommon, Imm, true, L, SE);
    if (!Imm->isZero()) {
      bool DoSink = true;

      // If the immediate part of the common expression is a GV, check if it's
      // possible to fold it into the target addressing mode.
      GlobalValue *GV = 0;
      if (const SCEVUnknown *SU = dyn_cast<SCEVUnknown>(Imm))
        GV = dyn_cast<GlobalValue>(SU->getValue());
      int64_t Offset = 0;
      if (const SCEVConstant *SC = dyn_cast<SCEVConstant>(Imm))
        Offset = SC->getValue()->getSExtValue();
      if (GV || Offset)
        // Pass VoidTy as the AccessTy to be conservative, because
        // there could be multiple access types among all the uses.
        DoSink = IsImmFoldedIntoAddrMode(GV, Offset,
                          Type::getVoidTy(L->getLoopPreheader()->getContext()),
                                         UsersToProcess, TLI);

      if (DoSink) {
        DEBUG(errs() << "  Sinking " << *Imm << " back down into uses\n");
        for (unsigned i = 0, e = UsersToProcess.size(); i != e; ++i)
          UsersToProcess[i].Imm = SE->getAddExpr(UsersToProcess[i].Imm, Imm);
        CommonExprs = NewCommon;
        HaveCommonExprs = !CommonExprs->isZero();
        ++NumImmSunk;
      }
    }
  }

  // Now that we know what we need to do, insert the PHI node itself.
  //
  DEBUG(errs() << "LSR: Examining IVs of TYPE " << *ReplacedTy << " of STRIDE "
               << *Stride << ":\n"
               << "  Common base: " << *CommonExprs << "\n");

  SCEVExpander Rewriter(*SE);
  SCEVExpander PreheaderRewriter(*SE);

  BasicBlock  *Preheader = L->getLoopPreheader();
  Instruction *PreInsertPt = Preheader->getTerminator();
  BasicBlock *LatchBlock = L->getLoopLatch();
  Instruction *IVIncInsertPt = LatchBlock->getTerminator();

  Value *CommonBaseV = Constant::getNullValue(ReplacedTy);

  const SCEV *RewriteFactor = SE->getIntegerSCEV(0, ReplacedTy);
  IVExpr   ReuseIV(SE->getIntegerSCEV(0,
                                    Type::getInt32Ty(Preheader->getContext())),
                   SE->getIntegerSCEV(0, 
                                    Type::getInt32Ty(Preheader->getContext())),
                   0);

  /// Choose a strength-reduction strategy and prepare for it by creating
  /// the necessary PHIs and adjusting the bookkeeping.
  if (ShouldUseFullStrengthReductionMode(UsersToProcess, L,
                                         AllUsesAreAddresses, Stride)) {
    PrepareToStrengthReduceFully(UsersToProcess, Stride, CommonExprs, L,
                                 PreheaderRewriter);
  } else {
    // Emit the initial base value into the loop preheader.
    CommonBaseV = PreheaderRewriter.expandCodeFor(CommonExprs, ReplacedTy,
                                                  PreInsertPt);

    // If all uses are addresses, check if it is possible to reuse an IV.  The
    // new IV must have a stride that is a multiple of the old stride; the
    // multiple must be a number that can be encoded in the scale field of the
    // target addressing mode; and we must have a valid instruction after this 
    // substitution, including the immediate field, if any.
    RewriteFactor = CheckForIVReuse(HaveCommonExprs, AllUsesAreAddresses,
                                    AllUsesAreOutsideLoop,
                                    Stride, ReuseIV, ReplacedTy,
                                    UsersToProcess);
    if (!RewriteFactor->isZero())
      PrepareToStrengthReduceFromSmallerStride(UsersToProcess, CommonBaseV,
                                               ReuseIV, PreInsertPt);
    else {
      IVIncInsertPt = FindIVIncInsertPt(UsersToProcess, L);
      PrepareToStrengthReduceWithNewPhi(UsersToProcess, Stride, CommonExprs,
                                        CommonBaseV, IVIncInsertPt,
                                        L, PreheaderRewriter);
    }
  }

  // Process all the users now, replacing their strided uses with
  // strength-reduced forms.  This outer loop handles all bases, the inner
  // loop handles all users of a particular base.
  while (!UsersToProcess.empty()) {
    const SCEV *Base = UsersToProcess.back().Base;
    Instruction *Inst = UsersToProcess.back().Inst;

    // Emit the code for Base into the preheader.
    Value *BaseV = 0;
    if (!Base->isZero()) {
      BaseV = PreheaderRewriter.expandCodeFor(Base, 0, PreInsertPt);

      DEBUG(errs() << "  INSERTING code for BASE = " << *Base << ":");
      if (BaseV->hasName())
        DEBUG(errs() << " Result value name = %" << BaseV->getName());
      DEBUG(errs() << "\n");

      // If BaseV is a non-zero constant, make sure that it gets inserted into
      // the preheader, instead of being forward substituted into the uses.  We
      // do this by forcing a BitCast (noop cast) to be inserted into the
      // preheader in this case.
      if (!fitsInAddressMode(Base, getAccessType(Inst), TLI, false) &&
          isa<Constant>(BaseV)) {
        // We want this constant emitted into the preheader! This is just
        // using cast as a copy so BitCast (no-op cast) is appropriate
        BaseV = new BitCastInst(BaseV, BaseV->getType(), "preheaderinsert",
                                PreInsertPt);       
      }
    }

    // Emit the code to add the immediate offset to the Phi value, just before
    // the instructions that we identified as using this stride and base.
    do {
      // FIXME: Use emitted users to emit other users.
      BasedUser &User = UsersToProcess.back();

      DEBUG(errs() << "    Examining ");
      if (User.isUseOfPostIncrementedValue)
        DEBUG(errs() << "postinc");
      else
        DEBUG(errs() << "preinc");
      DEBUG(errs() << " use ");
      DEBUG(WriteAsOperand(errs(), UsersToProcess.back().OperandValToReplace,
                           /*PrintType=*/false));
      DEBUG(errs() << " in Inst: " << *User.Inst);

      // If this instruction wants to use the post-incremented value, move it
      // after the post-inc and use its value instead of the PHI.
      Value *RewriteOp = User.Phi;
      if (User.isUseOfPostIncrementedValue) {
        RewriteOp = User.Phi->getIncomingValueForBlock(LatchBlock);
        // If this user is in the loop, make sure it is the last thing in the
        // loop to ensure it is dominated by the increment. In case it's the
        // only use of the iv, the increment instruction is already before the
        // use.
        if (L->contains(User.Inst->getParent()) && User.Inst != IVIncInsertPt)
          User.Inst->moveBefore(IVIncInsertPt);
      }

      const SCEV *RewriteExpr = SE->getUnknown(RewriteOp);

      if (SE->getEffectiveSCEVType(RewriteOp->getType()) !=
          SE->getEffectiveSCEVType(ReplacedTy)) {
        assert(SE->getTypeSizeInBits(RewriteOp->getType()) >
               SE->getTypeSizeInBits(ReplacedTy) &&
               "Unexpected widening cast!");
        RewriteExpr = SE->getTruncateExpr(RewriteExpr, ReplacedTy);
      }

      // If we had to insert new instructions for RewriteOp, we have to
      // consider that they may not have been able to end up immediately
      // next to RewriteOp, because non-PHI instructions may never precede
      // PHI instructions in a block. In this case, remember where the last
      // instruction was inserted so that if we're replacing a different
      // PHI node, we can use the later point to expand the final
      // RewriteExpr.
      Instruction *NewBasePt = dyn_cast<Instruction>(RewriteOp);
      if (RewriteOp == User.Phi) NewBasePt = 0;

      // Clear the SCEVExpander's expression map so that we are guaranteed
      // to have the code emitted where we expect it.
      Rewriter.clear();

      // If we are reusing the iv, then it must be multiplied by a constant
      // factor to take advantage of the addressing mode scale component.
      if (!RewriteFactor->isZero()) {
        // If we're reusing an IV with a nonzero base (currently this happens
        // only when all reuses are outside the loop) subtract that base here.
        // The base has been used to initialize the PHI node but we don't want
        // it here.
        if (!ReuseIV.Base->isZero()) {
          const SCEV *typedBase = ReuseIV.Base;
          if (SE->getEffectiveSCEVType(RewriteExpr->getType()) !=
              SE->getEffectiveSCEVType(ReuseIV.Base->getType())) {
            // It's possible the original IV is a larger type than the new IV,
            // in which case we have to truncate the Base.  We checked in
            // RequiresTypeConversion that this is valid.
            assert(SE->getTypeSizeInBits(RewriteExpr->getType()) <
                   SE->getTypeSizeInBits(ReuseIV.Base->getType()) &&
                   "Unexpected lengthening conversion!");
            typedBase = SE->getTruncateExpr(ReuseIV.Base, 
                                            RewriteExpr->getType());
          }
          RewriteExpr = SE->getMinusSCEV(RewriteExpr, typedBase);
        }

        // Multiply old variable, with base removed, by new scale factor.
        RewriteExpr = SE->getMulExpr(RewriteFactor,
                                     RewriteExpr);

        // The common base is emitted in the loop preheader. But since we
        // are reusing an IV, it has not been used to initialize the PHI node.
        // Add it to the expression used to rewrite the uses.
        // When this use is outside the loop, we earlier subtracted the
        // common base, and are adding it back here.  Use the same expression
        // as before, rather than CommonBaseV, so DAGCombiner will zap it.
        if (!CommonExprs->isZero()) {
          if (L->contains(User.Inst->getParent()))
            RewriteExpr = SE->getAddExpr(RewriteExpr,
                                       SE->getUnknown(CommonBaseV));
          else
            RewriteExpr = SE->getAddExpr(RewriteExpr, CommonExprs);
        }
      }

      // Now that we know what we need to do, insert code before User for the
      // immediate and any loop-variant expressions.
      if (BaseV)
        // Add BaseV to the PHI value if needed.
        RewriteExpr = SE->getAddExpr(RewriteExpr, SE->getUnknown(BaseV));

      User.RewriteInstructionToUseNewBase(RewriteExpr, NewBasePt,
                                          Rewriter, L, this, *LI,
                                          DeadInsts);

      // Mark old value we replaced as possibly dead, so that it is eliminated
      // if we just replaced the last use of that value.
      DeadInsts.push_back(User.OperandValToReplace);

      UsersToProcess.pop_back();
      ++NumReduced;

      // If there are any more users to process with the same base, process them
      // now.  We sorted by base above, so we just have to check the last elt.
    } while (!UsersToProcess.empty() && UsersToProcess.back().Base == Base);
    // TODO: Next, find out which base index is the most common, pull it out.
  }

  // IMPORTANT TODO: Figure out how to partition the IV's with this stride, but
  // different starting values, into different PHIs.
}

/// FindIVUserForCond - If Cond has an operand that is an expression of an IV,
/// set the IV user and stride information and return true, otherwise return
/// false.
bool LoopStrengthReduce::FindIVUserForCond(ICmpInst *Cond, IVStrideUse *&CondUse,
                                       const SCEV *const * &CondStride) {
  for (unsigned Stride = 0, e = IU->StrideOrder.size();
       Stride != e && !CondUse; ++Stride) {
    std::map<const SCEV *, IVUsersOfOneStride *>::iterator SI =
      IU->IVUsesByStride.find(IU->StrideOrder[Stride]);
    assert(SI != IU->IVUsesByStride.end() && "Stride doesn't exist!");

    for (ilist<IVStrideUse>::iterator UI = SI->second->Users.begin(),
         E = SI->second->Users.end(); UI != E; ++UI)
      if (UI->getUser() == Cond) {
        // NOTE: we could handle setcc instructions with multiple uses here, but
        // InstCombine does it as well for simple uses, it's not clear that it
        // occurs enough in real life to handle.
        CondUse = UI;
        CondStride = &SI->first;
        return true;
      }
  }
  return false;
}    

namespace {
  // Constant strides come first which in turns are sorted by their absolute
  // values. If absolute values are the same, then positive strides comes first.
  // e.g.
  // 4, -1, X, 1, 2 ==> 1, -1, 2, 4, X
  struct StrideCompare {
    const ScalarEvolution *SE;
    explicit StrideCompare(const ScalarEvolution *se) : SE(se) {}

    bool operator()(const SCEV *const &LHS, const SCEV *const &RHS) {
      const SCEVConstant *LHSC = dyn_cast<SCEVConstant>(LHS);
      const SCEVConstant *RHSC = dyn_cast<SCEVConstant>(RHS);
      if (LHSC && RHSC) {
        int64_t  LV = LHSC->getValue()->getSExtValue();
        int64_t  RV = RHSC->getValue()->getSExtValue();
        uint64_t ALV = (LV < 0) ? -LV : LV;
        uint64_t ARV = (RV < 0) ? -RV : RV;
        if (ALV == ARV) {
          if (LV != RV)
            return LV > RV;
        } else {
          return ALV < ARV;
        }

        // If it's the same value but different type, sort by bit width so
        // that we emit larger induction variables before smaller
        // ones, letting the smaller be re-written in terms of larger ones.
        return SE->getTypeSizeInBits(RHS->getType()) <
               SE->getTypeSizeInBits(LHS->getType());
      }
      return LHSC && !RHSC;
    }
  };
}

/// ChangeCompareStride - If a loop termination compare instruction is the
/// only use of its stride, and the compaison is against a constant value,
/// try eliminate the stride by moving the compare instruction to another
/// stride and change its constant operand accordingly. e.g.
///
/// loop:
/// ...
/// v1 = v1 + 3
/// v2 = v2 + 1
/// if (v2 < 10) goto loop
/// =>
/// loop:
/// ...
/// v1 = v1 + 3
/// if (v1 < 30) goto loop
ICmpInst *LoopStrengthReduce::ChangeCompareStride(Loop *L, ICmpInst *Cond,
                                                IVStrideUse* &CondUse,
                                              const SCEV *const* &CondStride) {
  // If there's only one stride in the loop, there's nothing to do here.
  if (IU->StrideOrder.size() < 2)
    return Cond;
  // If there are other users of the condition's stride, don't bother
  // trying to change the condition because the stride will still
  // remain.
  std::map<const SCEV *, IVUsersOfOneStride *>::iterator I =
    IU->IVUsesByStride.find(*CondStride);
  if (I == IU->IVUsesByStride.end() ||
      I->second->Users.size() != 1)
    return Cond;
  // Only handle constant strides for now.
  const SCEVConstant *SC = dyn_cast<SCEVConstant>(*CondStride);
  if (!SC) return Cond;

  ICmpInst::Predicate Predicate = Cond->getPredicate();
  int64_t CmpSSInt = SC->getValue()->getSExtValue();
  unsigned BitWidth = SE->getTypeSizeInBits((*CondStride)->getType());
  uint64_t SignBit = 1ULL << (BitWidth-1);
  const Type *CmpTy = Cond->getOperand(0)->getType();
  const Type *NewCmpTy = NULL;
  unsigned TyBits = SE->getTypeSizeInBits(CmpTy);
  unsigned NewTyBits = 0;
  const SCEV **NewStride = NULL;
  Value *NewCmpLHS = NULL;
  Value *NewCmpRHS = NULL;
  int64_t Scale = 1;
  const SCEV *NewOffset = SE->getIntegerSCEV(0, CmpTy);

  if (ConstantInt *C = dyn_cast<ConstantInt>(Cond->getOperand(1))) {
    int64_t CmpVal = C->getValue().getSExtValue();

    // Check stride constant and the comparision constant signs to detect
    // overflow.
    if ((CmpVal & SignBit) != (CmpSSInt & SignBit))
      return Cond;

    // Look for a suitable stride / iv as replacement.
    for (unsigned i = 0, e = IU->StrideOrder.size(); i != e; ++i) {
      std::map<const SCEV *, IVUsersOfOneStride *>::iterator SI =
        IU->IVUsesByStride.find(IU->StrideOrder[i]);
      if (!isa<SCEVConstant>(SI->first))
        continue;
      int64_t SSInt = cast<SCEVConstant>(SI->first)->getValue()->getSExtValue();
      if (SSInt == CmpSSInt ||
          abs64(SSInt) < abs64(CmpSSInt) ||
          (SSInt % CmpSSInt) != 0)
        continue;

      Scale = SSInt / CmpSSInt;
      int64_t NewCmpVal = CmpVal * Scale;
      APInt Mul = APInt(BitWidth*2, CmpVal, true);
      Mul = Mul * APInt(BitWidth*2, Scale, true);
      // Check for overflow.
      if (!Mul.isSignedIntN(BitWidth))
        continue;
      // Check for overflow in the stride's type too.
      if (!Mul.isSignedIntN(SE->getTypeSizeInBits(SI->first->getType())))
        continue;

      // Watch out for overflow.
      if (ICmpInst::isSigned(Predicate) &&
          (CmpVal & SignBit) != (NewCmpVal & SignBit))
        continue;

      if (NewCmpVal == CmpVal)
        continue;
      // Pick the best iv to use trying to avoid a cast.
      NewCmpLHS = NULL;
      for (ilist<IVStrideUse>::iterator UI = SI->second->Users.begin(),
             E = SI->second->Users.end(); UI != E; ++UI) {
        Value *Op = UI->getOperandValToReplace();

        // If the IVStrideUse implies a cast, check for an actual cast which
        // can be used to find the original IV expression.
        if (SE->getEffectiveSCEVType(Op->getType()) !=
            SE->getEffectiveSCEVType(SI->first->getType())) {
          CastInst *CI = dyn_cast<CastInst>(Op);
          // If it's not a simple cast, it's complicated.
          if (!CI)
            continue;
          // If it's a cast from a type other than the stride type,
          // it's complicated.
          if (CI->getOperand(0)->getType() != SI->first->getType())
            continue;
          // Ok, we found the IV expression in the stride's type.
          Op = CI->getOperand(0);
        }

        NewCmpLHS = Op;
        if (NewCmpLHS->getType() == CmpTy)
          break;
      }
      if (!NewCmpLHS)
        continue;

      NewCmpTy = NewCmpLHS->getType();
      NewTyBits = SE->getTypeSizeInBits(NewCmpTy);
      const Type *NewCmpIntTy = IntegerType::get(Cond->getContext(), NewTyBits);
      if (RequiresTypeConversion(NewCmpTy, CmpTy)) {
        // Check if it is possible to rewrite it using
        // an iv / stride of a smaller integer type.
        unsigned Bits = NewTyBits;
        if (ICmpInst::isSigned(Predicate))
          --Bits;
        uint64_t Mask = (1ULL << Bits) - 1;
        if (((uint64_t)NewCmpVal & Mask) != (uint64_t)NewCmpVal)
          continue;
      }

      // Don't rewrite if use offset is non-constant and the new type is
      // of a different type.
      // FIXME: too conservative?
      if (NewTyBits != TyBits && !isa<SCEVConstant>(CondUse->getOffset()))
        continue;

      bool AllUsesAreAddresses = true;
      bool AllUsesAreOutsideLoop = true;
      std::vector<BasedUser> UsersToProcess;
      const SCEV *CommonExprs = CollectIVUsers(SI->first, *SI->second, L,
                                              AllUsesAreAddresses,
                                              AllUsesAreOutsideLoop,
                                              UsersToProcess);
      // Avoid rewriting the compare instruction with an iv of new stride
      // if it's likely the new stride uses will be rewritten using the
      // stride of the compare instruction.
      if (AllUsesAreAddresses &&
          ValidScale(!CommonExprs->isZero(), Scale, UsersToProcess))
        continue;

      // Avoid rewriting the compare instruction with an iv which has
      // implicit extension or truncation built into it.
      // TODO: This is over-conservative.
      if (SE->getTypeSizeInBits(CondUse->getOffset()->getType()) != TyBits)
        continue;

      // If scale is negative, use swapped predicate unless it's testing
      // for equality.
      if (Scale < 0 && !Cond->isEquality())
        Predicate = ICmpInst::getSwappedPredicate(Predicate);

      NewStride = &IU->StrideOrder[i];
      if (!isa<PointerType>(NewCmpTy))
        NewCmpRHS = ConstantInt::get(NewCmpTy, NewCmpVal);
      else {
        Constant *CI = ConstantInt::get(NewCmpIntTy, NewCmpVal);
        NewCmpRHS = ConstantExpr::getIntToPtr(CI, NewCmpTy);
      }
      NewOffset = TyBits == NewTyBits
        ? SE->getMulExpr(CondUse->getOffset(),
                         SE->getConstant(CmpTy, Scale))
        : SE->getConstant(NewCmpIntTy,
          cast<SCEVConstant>(CondUse->getOffset())->getValue()
            ->getSExtValue()*Scale);
      break;
    }
  }

  // Forgo this transformation if it the increment happens to be
  // unfortunately positioned after the condition, and the condition
  // has multiple uses which prevent it from being moved immediately
  // before the branch. See
  // test/Transforms/LoopStrengthReduce/change-compare-stride-trickiness-*.ll
  // for an example of this situation.
  if (!Cond->hasOneUse()) {
    for (BasicBlock::iterator I = Cond, E = Cond->getParent()->end();
         I != E; ++I)
      if (I == NewCmpLHS)
        return Cond;
  }

  if (NewCmpRHS) {
    // Create a new compare instruction using new stride / iv.
    ICmpInst *OldCond = Cond;
    // Insert new compare instruction.
    Cond = new ICmpInst(OldCond, Predicate, NewCmpLHS, NewCmpRHS,
                        L->getHeader()->getName() + ".termcond");

    // Remove the old compare instruction. The old indvar is probably dead too.
    DeadInsts.push_back(CondUse->getOperandValToReplace());
    OldCond->replaceAllUsesWith(Cond);
    OldCond->eraseFromParent();

    IU->IVUsesByStride[*NewStride]->addUser(NewOffset, Cond, NewCmpLHS);
    CondUse = &IU->IVUsesByStride[*NewStride]->Users.back();
    CondStride = NewStride;
    ++NumEliminated;
    Changed = true;
  }

  return Cond;
}

/// OptimizeMax - Rewrite the loop's terminating condition if it uses
/// a max computation.
///
/// This is a narrow solution to a specific, but acute, problem. For loops
/// like this:
///
///   i = 0;
///   do {
///     p[i] = 0.0;
///   } while (++i < n);
///
/// the trip count isn't just 'n', because 'n' might not be positive. And
/// unfortunately this can come up even for loops where the user didn't use
/// a C do-while loop. For example, seemingly well-behaved top-test loops
/// will commonly be lowered like this:
//
///   if (n > 0) {
///     i = 0;
///     do {
///       p[i] = 0.0;
///     } while (++i < n);
///   }
///
/// and then it's possible for subsequent optimization to obscure the if
/// test in such a way that indvars can't find it.
///
/// When indvars can't find the if test in loops like this, it creates a
/// max expression, which allows it to give the loop a canonical
/// induction variable:
///
///   i = 0;
///   max = n < 1 ? 1 : n;
///   do {
///     p[i] = 0.0;
///   } while (++i != max);
///
/// Canonical induction variables are necessary because the loop passes
/// are designed around them. The most obvious example of this is the
/// LoopInfo analysis, which doesn't remember trip count values. It
/// expects to be able to rediscover the trip count each time it is
/// needed, and it does this using a simple analyis that only succeeds if
/// the loop has a canonical induction variable.
///
/// However, when it comes time to generate code, the maximum operation
/// can be quite costly, especially if it's inside of an outer loop.
///
/// This function solves this problem by detecting this type of loop and
/// rewriting their conditions from ICMP_NE back to ICMP_SLT, and deleting
/// the instructions for the maximum computation.
///
ICmpInst *LoopStrengthReduce::OptimizeMax(Loop *L, ICmpInst *Cond,
                                          IVStrideUse* &CondUse) {
  // Check that the loop matches the pattern we're looking for.
  if (Cond->getPredicate() != CmpInst::ICMP_EQ &&
      Cond->getPredicate() != CmpInst::ICMP_NE)
    return Cond;

  SelectInst *Sel = dyn_cast<SelectInst>(Cond->getOperand(1));
  if (!Sel || !Sel->hasOneUse()) return Cond;

  const SCEV *BackedgeTakenCount = SE->getBackedgeTakenCount(L);
  if (isa<SCEVCouldNotCompute>(BackedgeTakenCount))
    return Cond;
  const SCEV *One = SE->getIntegerSCEV(1, BackedgeTakenCount->getType());

  // Add one to the backedge-taken count to get the trip count.
  const SCEV *IterationCount = SE->getAddExpr(BackedgeTakenCount, One);

  // Check for a max calculation that matches the pattern.
  if (!isa<SCEVSMaxExpr>(IterationCount) && !isa<SCEVUMaxExpr>(IterationCount))
    return Cond;
  const SCEVNAryExpr *Max = cast<SCEVNAryExpr>(IterationCount);
  if (Max != SE->getSCEV(Sel)) return Cond;

  // To handle a max with more than two operands, this optimization would
  // require additional checking and setup.
  if (Max->getNumOperands() != 2)
    return Cond;

  const SCEV *MaxLHS = Max->getOperand(0);
  const SCEV *MaxRHS = Max->getOperand(1);
  if (!MaxLHS || MaxLHS != One) return Cond;

  // Check the relevant induction variable for conformance to
  // the pattern.
  const SCEV *IV = SE->getSCEV(Cond->getOperand(0));
  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(IV);
  if (!AR || !AR->isAffine() ||
      AR->getStart() != One ||
      AR->getStepRecurrence(*SE) != One)
    return Cond;

  assert(AR->getLoop() == L &&
         "Loop condition operand is an addrec in a different loop!");

  // Check the right operand of the select, and remember it, as it will
  // be used in the new comparison instruction.
  Value *NewRHS = 0;
  if (SE->getSCEV(Sel->getOperand(1)) == MaxRHS)
    NewRHS = Sel->getOperand(1);
  else if (SE->getSCEV(Sel->getOperand(2)) == MaxRHS)
    NewRHS = Sel->getOperand(2);
  if (!NewRHS) return Cond;

  // Determine the new comparison opcode. It may be signed or unsigned,
  // and the original comparison may be either equality or inequality.
  CmpInst::Predicate Pred =
    isa<SCEVSMaxExpr>(Max) ? CmpInst::ICMP_SLT : CmpInst::ICMP_ULT;
  if (Cond->getPredicate() == CmpInst::ICMP_EQ)
    Pred = CmpInst::getInversePredicate(Pred);

  // Ok, everything looks ok to change the condition into an SLT or SGE and
  // delete the max calculation.
  ICmpInst *NewCond =
    new ICmpInst(Cond, Pred, Cond->getOperand(0), NewRHS, "scmp");

  // Delete the max calculation instructions.
  Cond->replaceAllUsesWith(NewCond);
  CondUse->setUser(NewCond);
  Instruction *Cmp = cast<Instruction>(Sel->getOperand(0));
  Cond->eraseFromParent();
  Sel->eraseFromParent();
  if (Cmp->use_empty())
    Cmp->eraseFromParent();
  return NewCond;
}

/// OptimizeShadowIV - If IV is used in a int-to-float cast
/// inside the loop then try to eliminate the cast opeation.
void LoopStrengthReduce::OptimizeShadowIV(Loop *L) {

  const SCEV *BackedgeTakenCount = SE->getBackedgeTakenCount(L);
  if (isa<SCEVCouldNotCompute>(BackedgeTakenCount))
    return;
    
  for (unsigned Stride = 0, e = IU->StrideOrder.size(); Stride != e;
       ++Stride) {
    std::map<const SCEV *, IVUsersOfOneStride *>::iterator SI =
      IU->IVUsesByStride.find(IU->StrideOrder[Stride]);
    assert(SI != IU->IVUsesByStride.end() && "Stride doesn't exist!");
    if (!isa<SCEVConstant>(SI->first))
      continue;

    for (ilist<IVStrideUse>::iterator UI = SI->second->Users.begin(),
           E = SI->second->Users.end(); UI != E; /* empty */) {
      ilist<IVStrideUse>::iterator CandidateUI = UI;
      ++UI;
      Instruction *ShadowUse = CandidateUI->getUser();
      const Type *DestTy = NULL;

      /* If shadow use is a int->float cast then insert a second IV
         to eliminate this cast.

           for (unsigned i = 0; i < n; ++i) 
             foo((double)i);

         is transformed into

           double d = 0.0;
           for (unsigned i = 0; i < n; ++i, ++d) 
             foo(d);
      */
      if (UIToFPInst *UCast = dyn_cast<UIToFPInst>(CandidateUI->getUser()))
        DestTy = UCast->getDestTy();
      else if (SIToFPInst *SCast = dyn_cast<SIToFPInst>(CandidateUI->getUser()))
        DestTy = SCast->getDestTy();
      if (!DestTy) continue;

      if (TLI) {
        // If target does not support DestTy natively then do not apply
        // this transformation.
        EVT DVT = TLI->getValueType(DestTy);
        if (!TLI->isTypeLegal(DVT)) continue;
      }

      PHINode *PH = dyn_cast<PHINode>(ShadowUse->getOperand(0));
      if (!PH) continue;
      if (PH->getNumIncomingValues() != 2) continue;

      const Type *SrcTy = PH->getType();
      int Mantissa = DestTy->getFPMantissaWidth();
      if (Mantissa == -1) continue; 
      if ((int)SE->getTypeSizeInBits(SrcTy) > Mantissa)
        continue;

      unsigned Entry, Latch;
      if (PH->getIncomingBlock(0) == L->getLoopPreheader()) {
        Entry = 0;
        Latch = 1;
      } else {
        Entry = 1;
        Latch = 0;
      }
        
      ConstantInt *Init = dyn_cast<ConstantInt>(PH->getIncomingValue(Entry));
      if (!Init) continue;
      Constant *NewInit = ConstantFP::get(DestTy, Init->getZExtValue());

      BinaryOperator *Incr = 
        dyn_cast<BinaryOperator>(PH->getIncomingValue(Latch));
      if (!Incr) continue;
      if (Incr->getOpcode() != Instruction::Add
          && Incr->getOpcode() != Instruction::Sub)
        continue;

      /* Initialize new IV, double d = 0.0 in above example. */
      ConstantInt *C = NULL;
      if (Incr->getOperand(0) == PH)
        C = dyn_cast<ConstantInt>(Incr->getOperand(1));
      else if (Incr->getOperand(1) == PH)
        C = dyn_cast<ConstantInt>(Incr->getOperand(0));
      else
        continue;

      if (!C) continue;

      // Ignore negative constants, as the code below doesn't handle them
      // correctly. TODO: Remove this restriction.
      if (!C->getValue().isStrictlyPositive()) continue;

      /* Add new PHINode. */
      PHINode *NewPH = PHINode::Create(DestTy, "IV.S.", PH);

      /* create new increment. '++d' in above example. */
      Constant *CFP = ConstantFP::get(DestTy, C->getZExtValue());
      BinaryOperator *NewIncr = 
        BinaryOperator::Create(Incr->getOpcode() == Instruction::Add ?
                                 Instruction::FAdd : Instruction::FSub,
                               NewPH, CFP, "IV.S.next.", Incr);

      NewPH->addIncoming(NewInit, PH->getIncomingBlock(Entry));
      NewPH->addIncoming(NewIncr, PH->getIncomingBlock(Latch));

      /* Remove cast operation */
      ShadowUse->replaceAllUsesWith(NewPH);
      ShadowUse->eraseFromParent();
      NumShadow++;
      break;
    }
  }
}

/// OptimizeIndvars - Now that IVUsesByStride is set up with all of the indvar
/// uses in the loop, look to see if we can eliminate some, in favor of using
/// common indvars for the different uses.
void LoopStrengthReduce::OptimizeIndvars(Loop *L) {
  // TODO: implement optzns here.

  OptimizeShadowIV(L);
}

/// OptimizeLoopTermCond - Change loop terminating condition to use the 
/// postinc iv when possible.
void LoopStrengthReduce::OptimizeLoopTermCond(Loop *L) {
  // Finally, get the terminating condition for the loop if possible.  If we
  // can, we want to change it to use a post-incremented version of its
  // induction variable, to allow coalescing the live ranges for the IV into
  // one register value.
  BasicBlock *LatchBlock = L->getLoopLatch();
  BasicBlock *ExitingBlock = L->getExitingBlock();
  
  if (!ExitingBlock)
    // Multiple exits, just look at the exit in the latch block if there is one.
    ExitingBlock = LatchBlock;
  BranchInst *TermBr = dyn_cast<BranchInst>(ExitingBlock->getTerminator());
  if (!TermBr)
    return;
  if (TermBr->isUnconditional() || !isa<ICmpInst>(TermBr->getCondition()))
    return;

  // Search IVUsesByStride to find Cond's IVUse if there is one.
  IVStrideUse *CondUse = 0;
  const SCEV *const *CondStride = 0;
  ICmpInst *Cond = cast<ICmpInst>(TermBr->getCondition());
  if (!FindIVUserForCond(Cond, CondUse, CondStride))
    return; // setcc doesn't use the IV.

  if (ExitingBlock != LatchBlock) {
    if (!Cond->hasOneUse())
      // See below, we don't want the condition to be cloned.
      return;

    // If exiting block is the latch block, we know it's safe and profitable to
    // transform the icmp to use post-inc iv. Otherwise do so only if it would
    // not reuse another iv and its iv would be reused by other uses. We are
    // optimizing for the case where the icmp is the only use of the iv.
    IVUsersOfOneStride &StrideUses = *IU->IVUsesByStride[*CondStride];
    for (ilist<IVStrideUse>::iterator I = StrideUses.Users.begin(),
         E = StrideUses.Users.end(); I != E; ++I) {
      if (I->getUser() == Cond)
        continue;
      if (!I->isUseOfPostIncrementedValue())
        return;
    }

    // FIXME: This is expensive, and worse still ChangeCompareStride does a
    // similar check. Can we perform all the icmp related transformations after
    // StrengthReduceStridedIVUsers?
    if (const SCEVConstant *SC = dyn_cast<SCEVConstant>(*CondStride)) {
      int64_t SInt = SC->getValue()->getSExtValue();
      for (unsigned NewStride = 0, ee = IU->StrideOrder.size(); NewStride != ee;
           ++NewStride) {
        std::map<const SCEV *, IVUsersOfOneStride *>::iterator SI =
          IU->IVUsesByStride.find(IU->StrideOrder[NewStride]);
        if (!isa<SCEVConstant>(SI->first) || SI->first == *CondStride)
          continue;
        int64_t SSInt =
          cast<SCEVConstant>(SI->first)->getValue()->getSExtValue();
        if (SSInt == SInt)
          return; // This can definitely be reused.
        if (unsigned(abs64(SSInt)) < SInt || (SSInt % SInt) != 0)
          continue;
        int64_t Scale = SSInt / SInt;
        bool AllUsesAreAddresses = true;
        bool AllUsesAreOutsideLoop = true;
        std::vector<BasedUser> UsersToProcess;
        const SCEV *CommonExprs = CollectIVUsers(SI->first, *SI->second, L,
                                                AllUsesAreAddresses,
                                                AllUsesAreOutsideLoop,
                                                UsersToProcess);
        // Avoid rewriting the compare instruction with an iv of new stride
        // if it's likely the new stride uses will be rewritten using the
        // stride of the compare instruction.
        if (AllUsesAreAddresses &&
            ValidScale(!CommonExprs->isZero(), Scale, UsersToProcess))
          return;
      }
    }

    StrideNoReuse.insert(*CondStride);
  }

  // If the trip count is computed in terms of a max (due to ScalarEvolution
  // being unable to find a sufficient guard, for example), change the loop
  // comparison to use SLT or ULT instead of NE.
  Cond = OptimizeMax(L, Cond, CondUse);

  // If possible, change stride and operands of the compare instruction to
  // eliminate one stride.
  if (ExitingBlock == LatchBlock)
    Cond = ChangeCompareStride(L, Cond, CondUse, CondStride);

  // It's possible for the setcc instruction to be anywhere in the loop, and
  // possible for it to have multiple users.  If it is not immediately before
  // the latch block branch, move it.
  if (&*++BasicBlock::iterator(Cond) != (Instruction*)TermBr) {
    if (Cond->hasOneUse()) {   // Condition has a single use, just move it.
      Cond->moveBefore(TermBr);
    } else {
      // Otherwise, clone the terminating condition and insert into the loopend.
      Cond = cast<ICmpInst>(Cond->clone());
      Cond->setName(L->getHeader()->getName() + ".termcond");
      LatchBlock->getInstList().insert(TermBr, Cond);
      
      // Clone the IVUse, as the old use still exists!
      IU->IVUsesByStride[*CondStride]->addUser(CondUse->getOffset(), Cond,
                                             CondUse->getOperandValToReplace());
      CondUse = &IU->IVUsesByStride[*CondStride]->Users.back();
    }
  }

  // If we get to here, we know that we can transform the setcc instruction to
  // use the post-incremented version of the IV, allowing us to coalesce the
  // live ranges for the IV correctly.
  CondUse->setOffset(SE->getMinusSCEV(CondUse->getOffset(), *CondStride));
  CondUse->setIsUseOfPostIncrementedValue(true);
  Changed = true;

  ++NumLoopCond;
}

/// OptimizeLoopCountIV - If, after all sharing of IVs, the IV used for deciding
/// when to exit the loop is used only for that purpose, try to rearrange things
/// so it counts down to a test against zero.
void LoopStrengthReduce::OptimizeLoopCountIV(Loop *L) {

  // If the number of times the loop is executed isn't computable, give up.
  const SCEV *BackedgeTakenCount = SE->getBackedgeTakenCount(L);
  if (isa<SCEVCouldNotCompute>(BackedgeTakenCount))
    return;

  // Get the terminating condition for the loop if possible (this isn't
  // necessarily in the latch, or a block that's a predecessor of the header).
  if (!L->getExitBlock())
    return; // More than one loop exit blocks.

  // Okay, there is one exit block.  Try to find the condition that causes the
  // loop to be exited.
  BasicBlock *ExitingBlock = L->getExitingBlock();
  if (!ExitingBlock)
    return; // More than one block exiting!

  // Okay, we've computed the exiting block.  See what condition causes us to
  // exit.
  //
  // FIXME: we should be able to handle switch instructions (with a single exit)
  BranchInst *TermBr = dyn_cast<BranchInst>(ExitingBlock->getTerminator());
  if (TermBr == 0) return;
  assert(TermBr->isConditional() && "If unconditional, it can't be in loop!");
  if (!isa<ICmpInst>(TermBr->getCondition()))
    return;
  ICmpInst *Cond = cast<ICmpInst>(TermBr->getCondition());

  // Handle only tests for equality for the moment, and only stride 1.
  if (Cond->getPredicate() != CmpInst::ICMP_EQ)
    return;
  const SCEV *IV = SE->getSCEV(Cond->getOperand(0));
  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(IV);
  const SCEV *One = SE->getIntegerSCEV(1, BackedgeTakenCount->getType());
  if (!AR || !AR->isAffine() || AR->getStepRecurrence(*SE) != One)
    return;
  // If the RHS of the comparison is defined inside the loop, the rewrite
  // cannot be done.
  if (Instruction *CR = dyn_cast<Instruction>(Cond->getOperand(1)))
    if (L->contains(CR->getParent()))
      return;

  // Make sure the IV is only used for counting.  Value may be preinc or
  // postinc; 2 uses in either case.
  if (!Cond->getOperand(0)->hasNUses(2))
    return;
  PHINode *phi = dyn_cast<PHINode>(Cond->getOperand(0));
  Instruction *incr;
  if (phi && phi->getParent()==L->getHeader()) {
    // value tested is preinc.  Find the increment.
    // A CmpInst is not a BinaryOperator; we depend on this.
    Instruction::use_iterator UI = phi->use_begin();
    incr = dyn_cast<BinaryOperator>(UI);
    if (!incr)
      incr = dyn_cast<BinaryOperator>(++UI);
    // 1 use for postinc value, the phi.  Unnecessarily conservative?
    if (!incr || !incr->hasOneUse() || incr->getOpcode()!=Instruction::Add)
      return;
  } else {
    // Value tested is postinc.  Find the phi node.
    incr = dyn_cast<BinaryOperator>(Cond->getOperand(0));
    if (!incr || incr->getOpcode()!=Instruction::Add)
      return;

    Instruction::use_iterator UI = Cond->getOperand(0)->use_begin();
    phi = dyn_cast<PHINode>(UI);
    if (!phi)
      phi = dyn_cast<PHINode>(++UI);
    // 1 use for preinc value, the increment.
    if (!phi || phi->getParent()!=L->getHeader() || !phi->hasOneUse())
      return;
  }

  // Replace the increment with a decrement.
  BinaryOperator *decr = 
    BinaryOperator::Create(Instruction::Sub, incr->getOperand(0),
                           incr->getOperand(1), "tmp", incr);
  incr->replaceAllUsesWith(decr);
  incr->eraseFromParent();

  // Substitute endval-startval for the original startval, and 0 for the
  // original endval.  Since we're only testing for equality this is OK even 
  // if the computation wraps around.
  BasicBlock  *Preheader = L->getLoopPreheader();
  Instruction *PreInsertPt = Preheader->getTerminator();
  int inBlock = L->contains(phi->getIncomingBlock(0)) ? 1 : 0;
  Value *startVal = phi->getIncomingValue(inBlock);
  Value *endVal = Cond->getOperand(1);
  // FIXME check for case where both are constant
  Constant* Zero = ConstantInt::get(Cond->getOperand(1)->getType(), 0);
  BinaryOperator *NewStartVal = 
    BinaryOperator::Create(Instruction::Sub, endVal, startVal,
                           "tmp", PreInsertPt);
  phi->setIncomingValue(inBlock, NewStartVal);
  Cond->setOperand(1, Zero);

  Changed = true;
}

bool LoopStrengthReduce::runOnLoop(Loop *L, LPPassManager &LPM) {

  IU = &getAnalysis<IVUsers>();
  LI = &getAnalysis<LoopInfo>();
  DT = &getAnalysis<DominatorTree>();
  SE = &getAnalysis<ScalarEvolution>();
  Changed = false;

  // If LoopSimplify form is not available, stay out of trouble.
  if (!L->getLoopPreheader() || !L->getLoopLatch())
    return false;

  if (!IU->IVUsesByStride.empty()) {
    DEBUG(errs() << "\nLSR on \"" << L->getHeader()->getParent()->getName()
          << "\" ";
          L->dump());

    // Sort the StrideOrder so we process larger strides first.
    std::stable_sort(IU->StrideOrder.begin(), IU->StrideOrder.end(),
                     StrideCompare(SE));

    // Optimize induction variables.  Some indvar uses can be transformed to use
    // strides that will be needed for other purposes.  A common example of this
    // is the exit test for the loop, which can often be rewritten to use the
    // computation of some other indvar to decide when to terminate the loop.
    OptimizeIndvars(L);

    // Change loop terminating condition to use the postinc iv when possible
    // and optimize loop terminating compare. FIXME: Move this after
    // StrengthReduceStridedIVUsers?
    OptimizeLoopTermCond(L);

    // FIXME: We can shrink overlarge IV's here.  e.g. if the code has
    // computation in i64 values and the target doesn't support i64, demote
    // the computation to 32-bit if safe.

    // FIXME: Attempt to reuse values across multiple IV's.  In particular, we
    // could have something like "for(i) { foo(i*8); bar(i*16) }", which should
    // be codegened as "for (j = 0;; j+=8) { foo(j); bar(j+j); }" on X86/PPC.
    // Need to be careful that IV's are all the same type.  Only works for
    // intptr_t indvars.

    // IVsByStride keeps IVs for one particular loop.
    assert(IVsByStride.empty() && "Stale entries in IVsByStride?");

    // Note: this processes each stride/type pair individually.  All users
    // passed into StrengthReduceStridedIVUsers have the same type AND stride.
    // Also, note that we iterate over IVUsesByStride indirectly by using
    // StrideOrder. This extra layer of indirection makes the ordering of
    // strides deterministic - not dependent on map order.
    for (unsigned Stride = 0, e = IU->StrideOrder.size();
         Stride != e; ++Stride) {
      std::map<const SCEV *, IVUsersOfOneStride *>::iterator SI =
        IU->IVUsesByStride.find(IU->StrideOrder[Stride]);
      assert(SI != IU->IVUsesByStride.end() && "Stride doesn't exist!");
      // FIXME: Generalize to non-affine IV's.
      if (!SI->first->isLoopInvariant(L))
        continue;
      StrengthReduceStridedIVUsers(SI->first, *SI->second, L);
    }
  }

  // After all sharing is done, see if we can adjust the loop to test against
  // zero instead of counting up to a maximum.  This is usually faster.
  OptimizeLoopCountIV(L);

  // We're done analyzing this loop; release all the state we built up for it.
  IVsByStride.clear();
  StrideNoReuse.clear();

  // Clean up after ourselves
  if (!DeadInsts.empty())
    DeleteTriviallyDeadInstructions();

  // At this point, it is worth checking to see if any recurrence PHIs are also
  // dead, so that we can remove them as well.
  DeleteDeadPHIs(L->getHeader());

  return Changed;
}
