//===- InstructionCombining.cpp - Combine multiple instructions -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// InstructionCombining - Combine instructions to form fewer, simple
// instructions.  This pass does not modify the CFG.  This pass is where
// algebraic simplification happens.
//
// This pass combines things like:
//    %Y = add i32 %X, 1
//    %Z = add i32 %Y, 1
// into:
//    %Z = add i32 %X, 2
//
// This is a simple worklist driven algorithm.
//
// This pass guarantees that the following canonicalizations are performed on
// the program:
//    1. If a binary operator has a constant operand, it is moved to the RHS
//    2. Bitwise operators with constant operands are always grouped so that
//       shifts are performed first, then or's, then and's, then xor's.
//    3. Compare instructions are converted from <,>,<=,>= to ==,!= if possible
//    4. All cmp instructions on boolean values are replaced with logical ops
//    5. add X, X is represented as (X*2) => (X << 1)
//    6. Multiplies with a power-of-two constant argument are transformed into
//       shifts.
//   ... etc.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "instcombine"
#include "llvm/Transforms/Scalar.h"
#include "InstCombine.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/LLVMContext.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Operator.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/PatternMatch.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <climits>
using namespace llvm;
using namespace llvm::PatternMatch;

STATISTIC(NumCombined , "Number of insts combined");
STATISTIC(NumConstProp, "Number of constant folds");
STATISTIC(NumDeadInst , "Number of dead inst eliminated");
STATISTIC(NumDeadStore, "Number of dead stores eliminated");
STATISTIC(NumSunkInst , "Number of instructions sunk");


char InstCombiner::ID = 0;
static RegisterPass<InstCombiner>
X("instcombine", "Combine redundant instructions");

void InstCombiner::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addPreservedID(LCSSAID);
  AU.setPreservesCFG();
}


// isOnlyUse - Return true if this instruction will be deleted if we stop using
// it.
static bool isOnlyUse(Value *V) {
  return V->hasOneUse() || isa<Constant>(V);
}

// getPromotedType - Return the specified type promoted as it would be to pass
// though a va_arg area...
static const Type *getPromotedType(const Type *Ty) {
  if (const IntegerType* ITy = dyn_cast<IntegerType>(Ty)) {
    if (ITy->getBitWidth() < 32)
      return Type::getInt32Ty(Ty->getContext());
  }
  return Ty;
}

/// ShouldChangeType - Return true if it is desirable to convert a computation
/// from 'From' to 'To'.  We don't want to convert from a legal to an illegal
/// type for example, or from a smaller to a larger illegal type.
bool InstCombiner::ShouldChangeType(const Type *From, const Type *To) const {
  assert(isa<IntegerType>(From) && isa<IntegerType>(To));
  
  // If we don't have TD, we don't know if the source/dest are legal.
  if (!TD) return false;
  
  unsigned FromWidth = From->getPrimitiveSizeInBits();
  unsigned ToWidth = To->getPrimitiveSizeInBits();
  bool FromLegal = TD->isLegalInteger(FromWidth);
  bool ToLegal = TD->isLegalInteger(ToWidth);
  
  // If this is a legal integer from type, and the result would be an illegal
  // type, don't do the transformation.
  if (FromLegal && !ToLegal)
    return false;
  
  // Otherwise, if both are illegal, do not increase the size of the result. We
  // do allow things like i160 -> i64, but not i64 -> i160.
  if (!FromLegal && !ToLegal && ToWidth > FromWidth)
    return false;
  
  return true;
}

/// getBitCastOperand - If the specified operand is a CastInst, a constant
/// expression bitcast, or a GetElementPtrInst with all zero indices, return the
/// operand value, otherwise return null.
static Value *getBitCastOperand(Value *V) {
  if (Operator *O = dyn_cast<Operator>(V)) {
    if (O->getOpcode() == Instruction::BitCast)
      return O->getOperand(0);
    if (GEPOperator *GEP = dyn_cast<GEPOperator>(V))
      if (GEP->hasAllZeroIndices())
        return GEP->getPointerOperand();
  }
  return 0;
}



// SimplifyCommutative - This performs a few simplifications for commutative
// operators:
//
//  1. Order operands such that they are listed from right (least complex) to
//     left (most complex).  This puts constants before unary operators before
//     binary operators.
//
//  2. Transform: (op (op V, C1), C2) ==> (op V, (op C1, C2))
//  3. Transform: (op (op V1, C1), (op V2, C2)) ==> (op (op V1, V2), (op C1,C2))
//
bool InstCombiner::SimplifyCommutative(BinaryOperator &I) {
  bool Changed = false;
  if (getComplexity(I.getOperand(0)) < getComplexity(I.getOperand(1)))
    Changed = !I.swapOperands();

  if (!I.isAssociative()) return Changed;
  Instruction::BinaryOps Opcode = I.getOpcode();
  if (BinaryOperator *Op = dyn_cast<BinaryOperator>(I.getOperand(0)))
    if (Op->getOpcode() == Opcode && isa<Constant>(Op->getOperand(1))) {
      if (isa<Constant>(I.getOperand(1))) {
        Constant *Folded = ConstantExpr::get(I.getOpcode(),
                                             cast<Constant>(I.getOperand(1)),
                                             cast<Constant>(Op->getOperand(1)));
        I.setOperand(0, Op->getOperand(0));
        I.setOperand(1, Folded);
        return true;
      } else if (BinaryOperator *Op1=dyn_cast<BinaryOperator>(I.getOperand(1)))
        if (Op1->getOpcode() == Opcode && isa<Constant>(Op1->getOperand(1)) &&
            isOnlyUse(Op) && isOnlyUse(Op1)) {
          Constant *C1 = cast<Constant>(Op->getOperand(1));
          Constant *C2 = cast<Constant>(Op1->getOperand(1));

          // Fold (op (op V1, C1), (op V2, C2)) ==> (op (op V1, V2), (op C1,C2))
          Constant *Folded = ConstantExpr::get(I.getOpcode(), C1, C2);
          Instruction *New = BinaryOperator::Create(Opcode, Op->getOperand(0),
                                                    Op1->getOperand(0),
                                                    Op1->getName(), &I);
          Worklist.Add(New);
          I.setOperand(0, New);
          I.setOperand(1, Folded);
          return true;
        }
    }
  return Changed;
}

// dyn_castNegVal - Given a 'sub' instruction, return the RHS of the instruction
// if the LHS is a constant zero (which is the 'negate' form).
//
Value *InstCombiner::dyn_castNegVal(Value *V) const {
  if (BinaryOperator::isNeg(V))
    return BinaryOperator::getNegArgument(V);

  // Constants can be considered to be negated values if they can be folded.
  if (ConstantInt *C = dyn_cast<ConstantInt>(V))
    return ConstantExpr::getNeg(C);

  if (ConstantVector *C = dyn_cast<ConstantVector>(V))
    if (C->getType()->getElementType()->isInteger())
      return ConstantExpr::getNeg(C);

  return 0;
}

// dyn_castFNegVal - Given a 'fsub' instruction, return the RHS of the
// instruction if the LHS is a constant negative zero (which is the 'negate'
// form).
//
static inline Value *dyn_castFNegVal(Value *V) {
  if (BinaryOperator::isFNeg(V))
    return BinaryOperator::getFNegArgument(V);

  // Constants can be considered to be negated values if they can be folded.
  if (ConstantFP *C = dyn_cast<ConstantFP>(V))
    return ConstantExpr::getFNeg(C);

  if (ConstantVector *C = dyn_cast<ConstantVector>(V))
    if (C->getType()->getElementType()->isFloatingPoint())
      return ConstantExpr::getFNeg(C);

  return 0;
}

/// MatchSelectPattern - Pattern match integer [SU]MIN, [SU]MAX, and ABS idioms,
/// returning the kind and providing the out parameter results if we
/// successfully match.
static SelectPatternFlavor
MatchSelectPattern(Value *V, Value *&LHS, Value *&RHS) {
  SelectInst *SI = dyn_cast<SelectInst>(V);
  if (SI == 0) return SPF_UNKNOWN;
  
  ICmpInst *ICI = dyn_cast<ICmpInst>(SI->getCondition());
  if (ICI == 0) return SPF_UNKNOWN;
  
  LHS = ICI->getOperand(0);
  RHS = ICI->getOperand(1);
  
  // (icmp X, Y) ? X : Y 
  if (SI->getTrueValue() == ICI->getOperand(0) &&
      SI->getFalseValue() == ICI->getOperand(1)) {
    switch (ICI->getPredicate()) {
    default: return SPF_UNKNOWN; // Equality.
    case ICmpInst::ICMP_UGT:
    case ICmpInst::ICMP_UGE: return SPF_UMAX;
    case ICmpInst::ICMP_SGT:
    case ICmpInst::ICMP_SGE: return SPF_SMAX;
    case ICmpInst::ICMP_ULT:
    case ICmpInst::ICMP_ULE: return SPF_UMIN;
    case ICmpInst::ICMP_SLT:
    case ICmpInst::ICMP_SLE: return SPF_SMIN;
    }
  }
  
  // (icmp X, Y) ? Y : X 
  if (SI->getTrueValue() == ICI->getOperand(1) &&
      SI->getFalseValue() == ICI->getOperand(0)) {
    switch (ICI->getPredicate()) {
      default: return SPF_UNKNOWN; // Equality.
      case ICmpInst::ICMP_UGT:
      case ICmpInst::ICMP_UGE: return SPF_UMIN;
      case ICmpInst::ICMP_SGT:
      case ICmpInst::ICMP_SGE: return SPF_SMIN;
      case ICmpInst::ICMP_ULT:
      case ICmpInst::ICMP_ULE: return SPF_UMAX;
      case ICmpInst::ICMP_SLT:
      case ICmpInst::ICMP_SLE: return SPF_SMAX;
    }
  }
  
  // TODO: (X > 4) ? X : 5   -->  (X >= 5) ? X : 5  -->  MAX(X, 5)
  
  return SPF_UNKNOWN;
}

/// isFreeToInvert - Return true if the specified value is free to invert (apply
/// ~ to).  This happens in cases where the ~ can be eliminated.
static inline bool isFreeToInvert(Value *V) {
  // ~(~(X)) -> X.
  if (BinaryOperator::isNot(V))
    return true;
  
  // Constants can be considered to be not'ed values.
  if (isa<ConstantInt>(V))
    return true;
  
  // Compares can be inverted if they have a single use.
  if (CmpInst *CI = dyn_cast<CmpInst>(V))
    return CI->hasOneUse();
  
  return false;
}

static inline Value *dyn_castNotVal(Value *V) {
  // If this is not(not(x)) don't return that this is a not: we want the two
  // not's to be folded first.
  if (BinaryOperator::isNot(V)) {
    Value *Operand = BinaryOperator::getNotArgument(V);
    if (!isFreeToInvert(Operand))
      return Operand;
  }

  // Constants can be considered to be not'ed values...
  if (ConstantInt *C = dyn_cast<ConstantInt>(V))
    return ConstantInt::get(C->getType(), ~C->getValue());
  return 0;
}



// dyn_castFoldableMul - If this value is a multiply that can be folded into
// other computations (because it has a constant operand), return the
// non-constant operand of the multiply, and set CST to point to the multiplier.
// Otherwise, return null.
//
static inline Value *dyn_castFoldableMul(Value *V, ConstantInt *&CST) {
  if (V->hasOneUse() && V->getType()->isInteger())
    if (Instruction *I = dyn_cast<Instruction>(V)) {
      if (I->getOpcode() == Instruction::Mul)
        if ((CST = dyn_cast<ConstantInt>(I->getOperand(1))))
          return I->getOperand(0);
      if (I->getOpcode() == Instruction::Shl)
        if ((CST = dyn_cast<ConstantInt>(I->getOperand(1)))) {
          // The multiplier is really 1 << CST.
          uint32_t BitWidth = cast<IntegerType>(V->getType())->getBitWidth();
          uint32_t CSTVal = CST->getLimitedValue(BitWidth);
          CST = ConstantInt::get(V->getType()->getContext(),
                                 APInt(BitWidth, 1).shl(CSTVal));
          return I->getOperand(0);
        }
    }
  return 0;
}

/// AddOne - Add one to a ConstantInt
static Constant *AddOne(Constant *C) {
  return ConstantExpr::getAdd(C, ConstantInt::get(C->getType(), 1));
}
/// SubOne - Subtract one from a ConstantInt
static Constant *SubOne(ConstantInt *C) {
  return ConstantExpr::getSub(C,  ConstantInt::get(C->getType(), 1));
}
/// MultiplyOverflows - True if the multiply can not be expressed in an int
/// this size.
static bool MultiplyOverflows(ConstantInt *C1, ConstantInt *C2, bool sign) {
  uint32_t W = C1->getBitWidth();
  APInt LHSExt = C1->getValue(), RHSExt = C2->getValue();
  if (sign) {
    LHSExt.sext(W * 2);
    RHSExt.sext(W * 2);
  } else {
    LHSExt.zext(W * 2);
    RHSExt.zext(W * 2);
  }

  APInt MulExt = LHSExt * RHSExt;

  if (!sign)
    return MulExt.ugt(APInt::getLowBitsSet(W * 2, W));
  
  APInt Min = APInt::getSignedMinValue(W).sext(W * 2);
  APInt Max = APInt::getSignedMaxValue(W).sext(W * 2);
  return MulExt.slt(Min) || MulExt.sgt(Max);
}



/// AssociativeOpt - Perform an optimization on an associative operator.  This
/// function is designed to check a chain of associative operators for a
/// potential to apply a certain optimization.  Since the optimization may be
/// applicable if the expression was reassociated, this checks the chain, then
/// reassociates the expression as necessary to expose the optimization
/// opportunity.  This makes use of a special Functor, which must define
/// 'shouldApply' and 'apply' methods.
///
template<typename Functor>
static Instruction *AssociativeOpt(BinaryOperator &Root, const Functor &F) {
  unsigned Opcode = Root.getOpcode();
  Value *LHS = Root.getOperand(0);

  // Quick check, see if the immediate LHS matches...
  if (F.shouldApply(LHS))
    return F.apply(Root);

  // Otherwise, if the LHS is not of the same opcode as the root, return.
  Instruction *LHSI = dyn_cast<Instruction>(LHS);
  while (LHSI && LHSI->getOpcode() == Opcode && LHSI->hasOneUse()) {
    // Should we apply this transform to the RHS?
    bool ShouldApply = F.shouldApply(LHSI->getOperand(1));

    // If not to the RHS, check to see if we should apply to the LHS...
    if (!ShouldApply && F.shouldApply(LHSI->getOperand(0))) {
      cast<BinaryOperator>(LHSI)->swapOperands();   // Make the LHS the RHS
      ShouldApply = true;
    }

    // If the functor wants to apply the optimization to the RHS of LHSI,
    // reassociate the expression from ((? op A) op B) to (? op (A op B))
    if (ShouldApply) {
      // Now all of the instructions are in the current basic block, go ahead
      // and perform the reassociation.
      Instruction *TmpLHSI = cast<Instruction>(Root.getOperand(0));

      // First move the selected RHS to the LHS of the root...
      Root.setOperand(0, LHSI->getOperand(1));

      // Make what used to be the LHS of the root be the user of the root...
      Value *ExtraOperand = TmpLHSI->getOperand(1);
      if (&Root == TmpLHSI) {
        Root.replaceAllUsesWith(Constant::getNullValue(TmpLHSI->getType()));
        return 0;
      }
      Root.replaceAllUsesWith(TmpLHSI);          // Users now use TmpLHSI
      TmpLHSI->setOperand(1, &Root);             // TmpLHSI now uses the root
      BasicBlock::iterator ARI = &Root; ++ARI;
      TmpLHSI->moveBefore(ARI);                  // Move TmpLHSI to after Root
      ARI = Root;

      // Now propagate the ExtraOperand down the chain of instructions until we
      // get to LHSI.
      while (TmpLHSI != LHSI) {
        Instruction *NextLHSI = cast<Instruction>(TmpLHSI->getOperand(0));
        // Move the instruction to immediately before the chain we are
        // constructing to avoid breaking dominance properties.
        NextLHSI->moveBefore(ARI);
        ARI = NextLHSI;

        Value *NextOp = NextLHSI->getOperand(1);
        NextLHSI->setOperand(1, ExtraOperand);
        TmpLHSI = NextLHSI;
        ExtraOperand = NextOp;
      }

      // Now that the instructions are reassociated, have the functor perform
      // the transformation...
      return F.apply(Root);
    }

    LHSI = dyn_cast<Instruction>(LHSI->getOperand(0));
  }
  return 0;
}

namespace {

// AddRHS - Implements: X + X --> X << 1
struct AddRHS {
  Value *RHS;
  explicit AddRHS(Value *rhs) : RHS(rhs) {}
  bool shouldApply(Value *LHS) const { return LHS == RHS; }
  Instruction *apply(BinaryOperator &Add) const {
    return BinaryOperator::CreateShl(Add.getOperand(0),
                                     ConstantInt::get(Add.getType(), 1));
  }
};

// AddMaskingAnd - Implements (A & C1)+(B & C2) --> (A & C1)|(B & C2)
//                 iff C1&C2 == 0
struct AddMaskingAnd {
  Constant *C2;
  explicit AddMaskingAnd(Constant *c) : C2(c) {}
  bool shouldApply(Value *LHS) const {
    ConstantInt *C1;
    return match(LHS, m_And(m_Value(), m_ConstantInt(C1))) &&
           ConstantExpr::getAnd(C1, C2)->isNullValue();
  }
  Instruction *apply(BinaryOperator &Add) const {
    return BinaryOperator::CreateOr(Add.getOperand(0), Add.getOperand(1));
  }
};

}

static Value *FoldOperationIntoSelectOperand(Instruction &I, Value *SO,
                                             InstCombiner *IC) {
  if (CastInst *CI = dyn_cast<CastInst>(&I))
    return IC->Builder->CreateCast(CI->getOpcode(), SO, I.getType());

  // Figure out if the constant is the left or the right argument.
  bool ConstIsRHS = isa<Constant>(I.getOperand(1));
  Constant *ConstOperand = cast<Constant>(I.getOperand(ConstIsRHS));

  if (Constant *SOC = dyn_cast<Constant>(SO)) {
    if (ConstIsRHS)
      return ConstantExpr::get(I.getOpcode(), SOC, ConstOperand);
    return ConstantExpr::get(I.getOpcode(), ConstOperand, SOC);
  }

  Value *Op0 = SO, *Op1 = ConstOperand;
  if (!ConstIsRHS)
    std::swap(Op0, Op1);
  
  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(&I))
    return IC->Builder->CreateBinOp(BO->getOpcode(), Op0, Op1,
                                    SO->getName()+".op");
  if (ICmpInst *CI = dyn_cast<ICmpInst>(&I))
    return IC->Builder->CreateICmp(CI->getPredicate(), Op0, Op1,
                                   SO->getName()+".cmp");
  if (FCmpInst *CI = dyn_cast<FCmpInst>(&I))
    return IC->Builder->CreateICmp(CI->getPredicate(), Op0, Op1,
                                   SO->getName()+".cmp");
  llvm_unreachable("Unknown binary instruction type!");
}

// FoldOpIntoSelect - Given an instruction with a select as one operand and a
// constant as the other operand, try to fold the binary operator into the
// select arguments.  This also works for Cast instructions, which obviously do
// not have a second operand.
Instruction *InstCombiner::FoldOpIntoSelect(Instruction &Op, SelectInst *SI) {
  // Don't modify shared select instructions
  if (!SI->hasOneUse()) return 0;
  Value *TV = SI->getOperand(1);
  Value *FV = SI->getOperand(2);

  if (isa<Constant>(TV) || isa<Constant>(FV)) {
    // Bool selects with constant operands can be folded to logical ops.
    if (SI->getType() == Type::getInt1Ty(SI->getContext())) return 0;

    Value *SelectTrueVal = FoldOperationIntoSelectOperand(Op, TV, this);
    Value *SelectFalseVal = FoldOperationIntoSelectOperand(Op, FV, this);

    return SelectInst::Create(SI->getCondition(), SelectTrueVal,
                              SelectFalseVal);
  }
  return 0;
}


/// FoldOpIntoPhi - Given a binary operator, cast instruction, or select which
/// has a PHI node as operand #0, see if we can fold the instruction into the
/// PHI (which is only possible if all operands to the PHI are constants).
///
/// If AllowAggressive is true, FoldOpIntoPhi will allow certain transforms
/// that would normally be unprofitable because they strongly encourage jump
/// threading.
Instruction *InstCombiner::FoldOpIntoPhi(Instruction &I,
                                         bool AllowAggressive) {
  AllowAggressive = false;
  PHINode *PN = cast<PHINode>(I.getOperand(0));
  unsigned NumPHIValues = PN->getNumIncomingValues();
  if (NumPHIValues == 0 ||
      // We normally only transform phis with a single use, unless we're trying
      // hard to make jump threading happen.
      (!PN->hasOneUse() && !AllowAggressive))
    return 0;
  
  
  // Check to see if all of the operands of the PHI are simple constants
  // (constantint/constantfp/undef).  If there is one non-constant value,
  // remember the BB it is in.  If there is more than one or if *it* is a PHI,
  // bail out.  We don't do arbitrary constant expressions here because moving
  // their computation can be expensive without a cost model.
  BasicBlock *NonConstBB = 0;
  for (unsigned i = 0; i != NumPHIValues; ++i)
    if (!isa<Constant>(PN->getIncomingValue(i)) ||
        isa<ConstantExpr>(PN->getIncomingValue(i))) {
      if (NonConstBB) return 0;  // More than one non-const value.
      if (isa<PHINode>(PN->getIncomingValue(i))) return 0;  // Itself a phi.
      NonConstBB = PN->getIncomingBlock(i);
      
      // If the incoming non-constant value is in I's block, we have an infinite
      // loop.
      if (NonConstBB == I.getParent())
        return 0;
    }
  
  // If there is exactly one non-constant value, we can insert a copy of the
  // operation in that block.  However, if this is a critical edge, we would be
  // inserting the computation one some other paths (e.g. inside a loop).  Only
  // do this if the pred block is unconditionally branching into the phi block.
  if (NonConstBB != 0 && !AllowAggressive) {
    BranchInst *BI = dyn_cast<BranchInst>(NonConstBB->getTerminator());
    if (!BI || !BI->isUnconditional()) return 0;
  }

  // Okay, we can do the transformation: create the new PHI node.
  PHINode *NewPN = PHINode::Create(I.getType(), "");
  NewPN->reserveOperandSpace(PN->getNumOperands()/2);
  InsertNewInstBefore(NewPN, *PN);
  NewPN->takeName(PN);

  // Next, add all of the operands to the PHI.
  if (SelectInst *SI = dyn_cast<SelectInst>(&I)) {
    // We only currently try to fold the condition of a select when it is a phi,
    // not the true/false values.
    Value *TrueV = SI->getTrueValue();
    Value *FalseV = SI->getFalseValue();
    BasicBlock *PhiTransBB = PN->getParent();
    for (unsigned i = 0; i != NumPHIValues; ++i) {
      BasicBlock *ThisBB = PN->getIncomingBlock(i);
      Value *TrueVInPred = TrueV->DoPHITranslation(PhiTransBB, ThisBB);
      Value *FalseVInPred = FalseV->DoPHITranslation(PhiTransBB, ThisBB);
      Value *InV = 0;
      if (Constant *InC = dyn_cast<Constant>(PN->getIncomingValue(i))) {
        InV = InC->isNullValue() ? FalseVInPred : TrueVInPred;
      } else {
        assert(PN->getIncomingBlock(i) == NonConstBB);
        InV = SelectInst::Create(PN->getIncomingValue(i), TrueVInPred,
                                 FalseVInPred,
                                 "phitmp", NonConstBB->getTerminator());
        Worklist.Add(cast<Instruction>(InV));
      }
      NewPN->addIncoming(InV, ThisBB);
    }
  } else if (I.getNumOperands() == 2) {
    Constant *C = cast<Constant>(I.getOperand(1));
    for (unsigned i = 0; i != NumPHIValues; ++i) {
      Value *InV = 0;
      if (Constant *InC = dyn_cast<Constant>(PN->getIncomingValue(i))) {
        if (CmpInst *CI = dyn_cast<CmpInst>(&I))
          InV = ConstantExpr::getCompare(CI->getPredicate(), InC, C);
        else
          InV = ConstantExpr::get(I.getOpcode(), InC, C);
      } else {
        assert(PN->getIncomingBlock(i) == NonConstBB);
        if (BinaryOperator *BO = dyn_cast<BinaryOperator>(&I)) 
          InV = BinaryOperator::Create(BO->getOpcode(),
                                       PN->getIncomingValue(i), C, "phitmp",
                                       NonConstBB->getTerminator());
        else if (CmpInst *CI = dyn_cast<CmpInst>(&I))
          InV = CmpInst::Create(CI->getOpcode(),
                                CI->getPredicate(),
                                PN->getIncomingValue(i), C, "phitmp",
                                NonConstBB->getTerminator());
        else
          llvm_unreachable("Unknown binop!");
        
        Worklist.Add(cast<Instruction>(InV));
      }
      NewPN->addIncoming(InV, PN->getIncomingBlock(i));
    }
  } else { 
    CastInst *CI = cast<CastInst>(&I);
    const Type *RetTy = CI->getType();
    for (unsigned i = 0; i != NumPHIValues; ++i) {
      Value *InV;
      if (Constant *InC = dyn_cast<Constant>(PN->getIncomingValue(i))) {
        InV = ConstantExpr::getCast(CI->getOpcode(), InC, RetTy);
      } else {
        assert(PN->getIncomingBlock(i) == NonConstBB);
        InV = CastInst::Create(CI->getOpcode(), PN->getIncomingValue(i), 
                               I.getType(), "phitmp", 
                               NonConstBB->getTerminator());
        Worklist.Add(cast<Instruction>(InV));
      }
      NewPN->addIncoming(InV, PN->getIncomingBlock(i));
    }
  }
  return ReplaceInstUsesWith(I, NewPN);
}


/// WillNotOverflowSignedAdd - Return true if we can prove that:
///    (sext (add LHS, RHS))  === (add (sext LHS), (sext RHS))
/// This basically requires proving that the add in the original type would not
/// overflow to change the sign bit or have a carry out.
bool InstCombiner::WillNotOverflowSignedAdd(Value *LHS, Value *RHS) {
  // There are different heuristics we can use for this.  Here are some simple
  // ones.
  
  // Add has the property that adding any two 2's complement numbers can only 
  // have one carry bit which can change a sign.  As such, if LHS and RHS each
  // have at least two sign bits, we know that the addition of the two values
  // will sign extend fine.
  if (ComputeNumSignBits(LHS) > 1 && ComputeNumSignBits(RHS) > 1)
    return true;
  
  
  // If one of the operands only has one non-zero bit, and if the other operand
  // has a known-zero bit in a more significant place than it (not including the
  // sign bit) the ripple may go up to and fill the zero, but won't change the
  // sign.  For example, (X & ~4) + 1.
  
  // TODO: Implement.
  
  return false;
}


Instruction *InstCombiner::visitAdd(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *LHS = I.getOperand(0), *RHS = I.getOperand(1);

  if (Value *V = SimplifyAddInst(LHS, RHS, I.hasNoSignedWrap(),
                                 I.hasNoUnsignedWrap(), TD))
    return ReplaceInstUsesWith(I, V);

  
  if (Constant *RHSC = dyn_cast<Constant>(RHS)) {
    if (ConstantInt *CI = dyn_cast<ConstantInt>(RHSC)) {
      // X + (signbit) --> X ^ signbit
      const APInt& Val = CI->getValue();
      uint32_t BitWidth = Val.getBitWidth();
      if (Val == APInt::getSignBit(BitWidth))
        return BinaryOperator::CreateXor(LHS, RHS);
      
      // See if SimplifyDemandedBits can simplify this.  This handles stuff like
      // (X & 254)+1 -> (X&254)|1
      if (SimplifyDemandedInstructionBits(I))
        return &I;

      // zext(bool) + C -> bool ? C + 1 : C
      if (ZExtInst *ZI = dyn_cast<ZExtInst>(LHS))
        if (ZI->getSrcTy() == Type::getInt1Ty(I.getContext()))
          return SelectInst::Create(ZI->getOperand(0), AddOne(CI), CI);
    }

    if (isa<PHINode>(LHS))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;
    
    ConstantInt *XorRHS = 0;
    Value *XorLHS = 0;
    if (isa<ConstantInt>(RHSC) &&
        match(LHS, m_Xor(m_Value(XorLHS), m_ConstantInt(XorRHS)))) {
      uint32_t TySizeBits = I.getType()->getScalarSizeInBits();
      const APInt& RHSVal = cast<ConstantInt>(RHSC)->getValue();
      
      uint32_t Size = TySizeBits / 2;
      APInt C0080Val(APInt(TySizeBits, 1ULL).shl(Size - 1));
      APInt CFF80Val(-C0080Val);
      do {
        if (TySizeBits > Size) {
          // If we have ADD(XOR(AND(X, 0xFF), 0x80), 0xF..F80), it's a sext.
          // If we have ADD(XOR(AND(X, 0xFF), 0xF..F80), 0x80), it's a sext.
          if ((RHSVal == CFF80Val && XorRHS->getValue() == C0080Val) ||
              (RHSVal == C0080Val && XorRHS->getValue() == CFF80Val)) {
            // This is a sign extend if the top bits are known zero.
            if (!MaskedValueIsZero(XorLHS, 
                   APInt::getHighBitsSet(TySizeBits, TySizeBits - Size)))
              Size = 0;  // Not a sign ext, but can't be any others either.
            break;
          }
        }
        Size >>= 1;
        C0080Val = APIntOps::lshr(C0080Val, Size);
        CFF80Val = APIntOps::ashr(CFF80Val, Size);
      } while (Size >= 1);
      
      // FIXME: This shouldn't be necessary. When the backends can handle types
      // with funny bit widths then this switch statement should be removed. It
      // is just here to get the size of the "middle" type back up to something
      // that the back ends can handle.
      const Type *MiddleType = 0;
      switch (Size) {
        default: break;
        case 32:
        case 16:
        case  8: MiddleType = IntegerType::get(I.getContext(), Size); break;
      }
      if (MiddleType) {
        Value *NewTrunc = Builder->CreateTrunc(XorLHS, MiddleType, "sext");
        return new SExtInst(NewTrunc, I.getType(), I.getName());
      }
    }
  }

  if (I.getType() == Type::getInt1Ty(I.getContext()))
    return BinaryOperator::CreateXor(LHS, RHS);

  // X + X --> X << 1
  if (I.getType()->isInteger()) {
    if (Instruction *Result = AssociativeOpt(I, AddRHS(RHS)))
      return Result;

    if (Instruction *RHSI = dyn_cast<Instruction>(RHS)) {
      if (RHSI->getOpcode() == Instruction::Sub)
        if (LHS == RHSI->getOperand(1))                   // A + (B - A) --> B
          return ReplaceInstUsesWith(I, RHSI->getOperand(0));
    }
    if (Instruction *LHSI = dyn_cast<Instruction>(LHS)) {
      if (LHSI->getOpcode() == Instruction::Sub)
        if (RHS == LHSI->getOperand(1))                   // (B - A) + A --> B
          return ReplaceInstUsesWith(I, LHSI->getOperand(0));
    }
  }

  // -A + B  -->  B - A
  // -A + -B  -->  -(A + B)
  if (Value *LHSV = dyn_castNegVal(LHS)) {
    if (LHS->getType()->isIntOrIntVector()) {
      if (Value *RHSV = dyn_castNegVal(RHS)) {
        Value *NewAdd = Builder->CreateAdd(LHSV, RHSV, "sum");
        return BinaryOperator::CreateNeg(NewAdd);
      }
    }
    
    return BinaryOperator::CreateSub(RHS, LHSV);
  }

  // A + -B  -->  A - B
  if (!isa<Constant>(RHS))
    if (Value *V = dyn_castNegVal(RHS))
      return BinaryOperator::CreateSub(LHS, V);


  ConstantInt *C2;
  if (Value *X = dyn_castFoldableMul(LHS, C2)) {
    if (X == RHS)   // X*C + X --> X * (C+1)
      return BinaryOperator::CreateMul(RHS, AddOne(C2));

    // X*C1 + X*C2 --> X * (C1+C2)
    ConstantInt *C1;
    if (X == dyn_castFoldableMul(RHS, C1))
      return BinaryOperator::CreateMul(X, ConstantExpr::getAdd(C1, C2));
  }

  // X + X*C --> X * (C+1)
  if (dyn_castFoldableMul(RHS, C2) == LHS)
    return BinaryOperator::CreateMul(LHS, AddOne(C2));

  // X + ~X --> -1   since   ~X = -X-1
  if (dyn_castNotVal(LHS) == RHS ||
      dyn_castNotVal(RHS) == LHS)
    return ReplaceInstUsesWith(I, Constant::getAllOnesValue(I.getType()));
  

  // (A & C1)+(B & C2) --> (A & C1)|(B & C2) iff C1&C2 == 0
  if (match(RHS, m_And(m_Value(), m_ConstantInt(C2))))
    if (Instruction *R = AssociativeOpt(I, AddMaskingAnd(C2)))
      return R;
  
  // A+B --> A|B iff A and B have no bits set in common.
  if (const IntegerType *IT = dyn_cast<IntegerType>(I.getType())) {
    APInt Mask = APInt::getAllOnesValue(IT->getBitWidth());
    APInt LHSKnownOne(IT->getBitWidth(), 0);
    APInt LHSKnownZero(IT->getBitWidth(), 0);
    ComputeMaskedBits(LHS, Mask, LHSKnownZero, LHSKnownOne);
    if (LHSKnownZero != 0) {
      APInt RHSKnownOne(IT->getBitWidth(), 0);
      APInt RHSKnownZero(IT->getBitWidth(), 0);
      ComputeMaskedBits(RHS, Mask, RHSKnownZero, RHSKnownOne);
      
      // No bits in common -> bitwise or.
      if ((LHSKnownZero|RHSKnownZero).isAllOnesValue())
        return BinaryOperator::CreateOr(LHS, RHS);
    }
  }

  // W*X + Y*Z --> W * (X+Z)  iff W == Y
  if (I.getType()->isIntOrIntVector()) {
    Value *W, *X, *Y, *Z;
    if (match(LHS, m_Mul(m_Value(W), m_Value(X))) &&
        match(RHS, m_Mul(m_Value(Y), m_Value(Z)))) {
      if (W != Y) {
        if (W == Z) {
          std::swap(Y, Z);
        } else if (Y == X) {
          std::swap(W, X);
        } else if (X == Z) {
          std::swap(Y, Z);
          std::swap(W, X);
        }
      }

      if (W == Y) {
        Value *NewAdd = Builder->CreateAdd(X, Z, LHS->getName());
        return BinaryOperator::CreateMul(W, NewAdd);
      }
    }
  }

  if (ConstantInt *CRHS = dyn_cast<ConstantInt>(RHS)) {
    Value *X = 0;
    if (match(LHS, m_Not(m_Value(X))))    // ~X + C --> (C-1) - X
      return BinaryOperator::CreateSub(SubOne(CRHS), X);

    // (X & FF00) + xx00  -> (X+xx00) & FF00
    if (LHS->hasOneUse() &&
        match(LHS, m_And(m_Value(X), m_ConstantInt(C2)))) {
      Constant *Anded = ConstantExpr::getAnd(CRHS, C2);
      if (Anded == CRHS) {
        // See if all bits from the first bit set in the Add RHS up are included
        // in the mask.  First, get the rightmost bit.
        const APInt& AddRHSV = CRHS->getValue();

        // Form a mask of all bits from the lowest bit added through the top.
        APInt AddRHSHighBits(~((AddRHSV & -AddRHSV)-1));

        // See if the and mask includes all of these bits.
        APInt AddRHSHighBitsAnd(AddRHSHighBits & C2->getValue());

        if (AddRHSHighBits == AddRHSHighBitsAnd) {
          // Okay, the xform is safe.  Insert the new add pronto.
          Value *NewAdd = Builder->CreateAdd(X, CRHS, LHS->getName());
          return BinaryOperator::CreateAnd(NewAdd, C2);
        }
      }
    }

    // Try to fold constant add into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(LHS))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;
  }

  // add (select X 0 (sub n A)) A  -->  select X A n
  {
    SelectInst *SI = dyn_cast<SelectInst>(LHS);
    Value *A = RHS;
    if (!SI) {
      SI = dyn_cast<SelectInst>(RHS);
      A = LHS;
    }
    if (SI && SI->hasOneUse()) {
      Value *TV = SI->getTrueValue();
      Value *FV = SI->getFalseValue();
      Value *N;

      // Can we fold the add into the argument of the select?
      // We check both true and false select arguments for a matching subtract.
      if (match(FV, m_Zero()) &&
          match(TV, m_Sub(m_Value(N), m_Specific(A))))
        // Fold the add into the true select value.
        return SelectInst::Create(SI->getCondition(), N, A);
      if (match(TV, m_Zero()) &&
          match(FV, m_Sub(m_Value(N), m_Specific(A))))
        // Fold the add into the false select value.
        return SelectInst::Create(SI->getCondition(), A, N);
    }
  }

  // Check for (add (sext x), y), see if we can merge this into an
  // integer add followed by a sext.
  if (SExtInst *LHSConv = dyn_cast<SExtInst>(LHS)) {
    // (add (sext x), cst) --> (sext (add x, cst'))
    if (ConstantInt *RHSC = dyn_cast<ConstantInt>(RHS)) {
      Constant *CI = 
        ConstantExpr::getTrunc(RHSC, LHSConv->getOperand(0)->getType());
      if (LHSConv->hasOneUse() &&
          ConstantExpr::getSExt(CI, I.getType()) == RHSC &&
          WillNotOverflowSignedAdd(LHSConv->getOperand(0), CI)) {
        // Insert the new, smaller add.
        Value *NewAdd = Builder->CreateNSWAdd(LHSConv->getOperand(0), 
                                              CI, "addconv");
        return new SExtInst(NewAdd, I.getType());
      }
    }
    
    // (add (sext x), (sext y)) --> (sext (add int x, y))
    if (SExtInst *RHSConv = dyn_cast<SExtInst>(RHS)) {
      // Only do this if x/y have the same type, if at last one of them has a
      // single use (so we don't increase the number of sexts), and if the
      // integer add will not overflow.
      if (LHSConv->getOperand(0)->getType()==RHSConv->getOperand(0)->getType()&&
          (LHSConv->hasOneUse() || RHSConv->hasOneUse()) &&
          WillNotOverflowSignedAdd(LHSConv->getOperand(0),
                                   RHSConv->getOperand(0))) {
        // Insert the new integer add.
        Value *NewAdd = Builder->CreateNSWAdd(LHSConv->getOperand(0), 
                                              RHSConv->getOperand(0), "addconv");
        return new SExtInst(NewAdd, I.getType());
      }
    }
  }

  return Changed ? &I : 0;
}

Instruction *InstCombiner::visitFAdd(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *LHS = I.getOperand(0), *RHS = I.getOperand(1);

  if (Constant *RHSC = dyn_cast<Constant>(RHS)) {
    // X + 0 --> X
    if (ConstantFP *CFP = dyn_cast<ConstantFP>(RHSC)) {
      if (CFP->isExactlyValue(ConstantFP::getNegativeZero
                              (I.getType())->getValueAPF()))
        return ReplaceInstUsesWith(I, LHS);
    }

    if (isa<PHINode>(LHS))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;
  }

  // -A + B  -->  B - A
  // -A + -B  -->  -(A + B)
  if (Value *LHSV = dyn_castFNegVal(LHS))
    return BinaryOperator::CreateFSub(RHS, LHSV);

  // A + -B  -->  A - B
  if (!isa<Constant>(RHS))
    if (Value *V = dyn_castFNegVal(RHS))
      return BinaryOperator::CreateFSub(LHS, V);

  // Check for X+0.0.  Simplify it to X if we know X is not -0.0.
  if (ConstantFP *CFP = dyn_cast<ConstantFP>(RHS))
    if (CFP->getValueAPF().isPosZero() && CannotBeNegativeZero(LHS))
      return ReplaceInstUsesWith(I, LHS);

  // Check for (add double (sitofp x), y), see if we can merge this into an
  // integer add followed by a promotion.
  if (SIToFPInst *LHSConv = dyn_cast<SIToFPInst>(LHS)) {
    // (add double (sitofp x), fpcst) --> (sitofp (add int x, intcst))
    // ... if the constant fits in the integer value.  This is useful for things
    // like (double)(x & 1234) + 4.0 -> (double)((X & 1234)+4) which no longer
    // requires a constant pool load, and generally allows the add to be better
    // instcombined.
    if (ConstantFP *CFP = dyn_cast<ConstantFP>(RHS)) {
      Constant *CI = 
      ConstantExpr::getFPToSI(CFP, LHSConv->getOperand(0)->getType());
      if (LHSConv->hasOneUse() &&
          ConstantExpr::getSIToFP(CI, I.getType()) == CFP &&
          WillNotOverflowSignedAdd(LHSConv->getOperand(0), CI)) {
        // Insert the new integer add.
        Value *NewAdd = Builder->CreateNSWAdd(LHSConv->getOperand(0),
                                              CI, "addconv");
        return new SIToFPInst(NewAdd, I.getType());
      }
    }
    
    // (add double (sitofp x), (sitofp y)) --> (sitofp (add int x, y))
    if (SIToFPInst *RHSConv = dyn_cast<SIToFPInst>(RHS)) {
      // Only do this if x/y have the same type, if at last one of them has a
      // single use (so we don't increase the number of int->fp conversions),
      // and if the integer add will not overflow.
      if (LHSConv->getOperand(0)->getType()==RHSConv->getOperand(0)->getType()&&
          (LHSConv->hasOneUse() || RHSConv->hasOneUse()) &&
          WillNotOverflowSignedAdd(LHSConv->getOperand(0),
                                   RHSConv->getOperand(0))) {
        // Insert the new integer add.
        Value *NewAdd = Builder->CreateNSWAdd(LHSConv->getOperand(0), 
                                              RHSConv->getOperand(0),"addconv");
        return new SIToFPInst(NewAdd, I.getType());
      }
    }
  }
  
  return Changed ? &I : 0;
}


/// EmitGEPOffset - Given a getelementptr instruction/constantexpr, emit the
/// code necessary to compute the offset from the base pointer (without adding
/// in the base pointer).  Return the result as a signed integer of intptr size.
Value *InstCombiner::EmitGEPOffset(User *GEP) {
  TargetData &TD = *getTargetData();
  gep_type_iterator GTI = gep_type_begin(GEP);
  const Type *IntPtrTy = TD.getIntPtrType(GEP->getContext());
  Value *Result = Constant::getNullValue(IntPtrTy);

  // Build a mask for high order bits.
  unsigned IntPtrWidth = TD.getPointerSizeInBits();
  uint64_t PtrSizeMask = ~0ULL >> (64-IntPtrWidth);

  for (User::op_iterator i = GEP->op_begin() + 1, e = GEP->op_end(); i != e;
       ++i, ++GTI) {
    Value *Op = *i;
    uint64_t Size = TD.getTypeAllocSize(GTI.getIndexedType()) & PtrSizeMask;
    if (ConstantInt *OpC = dyn_cast<ConstantInt>(Op)) {
      if (OpC->isZero()) continue;
      
      // Handle a struct index, which adds its field offset to the pointer.
      if (const StructType *STy = dyn_cast<StructType>(*GTI)) {
        Size = TD.getStructLayout(STy)->getElementOffset(OpC->getZExtValue());
        
        Result = Builder->CreateAdd(Result,
                                    ConstantInt::get(IntPtrTy, Size),
                                    GEP->getName()+".offs");
        continue;
      }
      
      Constant *Scale = ConstantInt::get(IntPtrTy, Size);
      Constant *OC =
              ConstantExpr::getIntegerCast(OpC, IntPtrTy, true /*SExt*/);
      Scale = ConstantExpr::getMul(OC, Scale);
      // Emit an add instruction.
      Result = Builder->CreateAdd(Result, Scale, GEP->getName()+".offs");
      continue;
    }
    // Convert to correct type.
    if (Op->getType() != IntPtrTy)
      Op = Builder->CreateIntCast(Op, IntPtrTy, true, Op->getName()+".c");
    if (Size != 1) {
      Constant *Scale = ConstantInt::get(IntPtrTy, Size);
      // We'll let instcombine(mul) convert this to a shl if possible.
      Op = Builder->CreateMul(Op, Scale, GEP->getName()+".idx");
    }

    // Emit an add instruction.
    Result = Builder->CreateAdd(Op, Result, GEP->getName()+".offs");
  }
  return Result;
}




/// Optimize pointer differences into the same array into a size.  Consider:
///  &A[10] - &A[0]: we should compile this to "10".  LHS/RHS are the pointer
/// operands to the ptrtoint instructions for the LHS/RHS of the subtract.
///
Value *InstCombiner::OptimizePointerDifference(Value *LHS, Value *RHS,
                                               const Type *Ty) {
  assert(TD && "Must have target data info for this");
  
  // If LHS is a gep based on RHS or RHS is a gep based on LHS, we can optimize
  // this.
  bool Swapped = false;
  GetElementPtrInst *GEP = 0;
  ConstantExpr *CstGEP = 0;
  
  // TODO: Could also optimize &A[i] - &A[j] -> "i-j", and "&A.foo[i] - &A.foo".
  // For now we require one side to be the base pointer "A" or a constant
  // expression derived from it.
  if (GetElementPtrInst *LHSGEP = dyn_cast<GetElementPtrInst>(LHS)) {
    // (gep X, ...) - X
    if (LHSGEP->getOperand(0) == RHS) {
      GEP = LHSGEP;
      Swapped = false;
    } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(RHS)) {
      // (gep X, ...) - (ce_gep X, ...)
      if (CE->getOpcode() == Instruction::GetElementPtr &&
          LHSGEP->getOperand(0) == CE->getOperand(0)) {
        CstGEP = CE;
        GEP = LHSGEP;
        Swapped = false;
      }
    }
  }
  
  if (GetElementPtrInst *RHSGEP = dyn_cast<GetElementPtrInst>(RHS)) {
    // X - (gep X, ...)
    if (RHSGEP->getOperand(0) == LHS) {
      GEP = RHSGEP;
      Swapped = true;
    } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(LHS)) {
      // (ce_gep X, ...) - (gep X, ...)
      if (CE->getOpcode() == Instruction::GetElementPtr &&
          RHSGEP->getOperand(0) == CE->getOperand(0)) {
        CstGEP = CE;
        GEP = RHSGEP;
        Swapped = true;
      }
    }
  }
  
  if (GEP == 0)
    return 0;
  
  // Emit the offset of the GEP and an intptr_t.
  Value *Result = EmitGEPOffset(GEP);
  
  // If we had a constant expression GEP on the other side offsetting the
  // pointer, subtract it from the offset we have.
  if (CstGEP) {
    Value *CstOffset = EmitGEPOffset(CstGEP);
    Result = Builder->CreateSub(Result, CstOffset);
  }
  

  // If we have p - gep(p, ...)  then we have to negate the result.
  if (Swapped)
    Result = Builder->CreateNeg(Result, "diff.neg");

  return Builder->CreateIntCast(Result, Ty, true);
}


Instruction *InstCombiner::visitSub(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Op0 == Op1)                        // sub X, X  -> 0
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

  // If this is a 'B = x-(-A)', change to B = x+A.  This preserves NSW/NUW.
  if (Value *V = dyn_castNegVal(Op1)) {
    BinaryOperator *Res = BinaryOperator::CreateAdd(Op0, V);
    Res->setHasNoSignedWrap(I.hasNoSignedWrap());
    Res->setHasNoUnsignedWrap(I.hasNoUnsignedWrap());
    return Res;
  }

  if (isa<UndefValue>(Op0))
    return ReplaceInstUsesWith(I, Op0);    // undef - X -> undef
  if (isa<UndefValue>(Op1))
    return ReplaceInstUsesWith(I, Op1);    // X - undef -> undef
  if (I.getType() == Type::getInt1Ty(I.getContext()))
    return BinaryOperator::CreateXor(Op0, Op1);
  
  if (ConstantInt *C = dyn_cast<ConstantInt>(Op0)) {
    // Replace (-1 - A) with (~A).
    if (C->isAllOnesValue())
      return BinaryOperator::CreateNot(Op1);

    // C - ~X == X + (1+C)
    Value *X = 0;
    if (match(Op1, m_Not(m_Value(X))))
      return BinaryOperator::CreateAdd(X, AddOne(C));

    // -(X >>u 31) -> (X >>s 31)
    // -(X >>s 31) -> (X >>u 31)
    if (C->isZero()) {
      if (BinaryOperator *SI = dyn_cast<BinaryOperator>(Op1)) {
        if (SI->getOpcode() == Instruction::LShr) {
          if (ConstantInt *CU = dyn_cast<ConstantInt>(SI->getOperand(1))) {
            // Check to see if we are shifting out everything but the sign bit.
            if (CU->getLimitedValue(SI->getType()->getPrimitiveSizeInBits()) ==
                SI->getType()->getPrimitiveSizeInBits()-1) {
              // Ok, the transformation is safe.  Insert AShr.
              return BinaryOperator::Create(Instruction::AShr, 
                                          SI->getOperand(0), CU, SI->getName());
            }
          }
        } else if (SI->getOpcode() == Instruction::AShr) {
          if (ConstantInt *CU = dyn_cast<ConstantInt>(SI->getOperand(1))) {
            // Check to see if we are shifting out everything but the sign bit.
            if (CU->getLimitedValue(SI->getType()->getPrimitiveSizeInBits()) ==
                SI->getType()->getPrimitiveSizeInBits()-1) {
              // Ok, the transformation is safe.  Insert LShr. 
              return BinaryOperator::CreateLShr(
                                          SI->getOperand(0), CU, SI->getName());
            }
          }
        }
      }
    }

    // Try to fold constant sub into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op1))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;

    // C - zext(bool) -> bool ? C - 1 : C
    if (ZExtInst *ZI = dyn_cast<ZExtInst>(Op1))
      if (ZI->getSrcTy() == Type::getInt1Ty(I.getContext()))
        return SelectInst::Create(ZI->getOperand(0), SubOne(C), C);
  }

  if (BinaryOperator *Op1I = dyn_cast<BinaryOperator>(Op1)) {
    if (Op1I->getOpcode() == Instruction::Add) {
      if (Op1I->getOperand(0) == Op0)              // X-(X+Y) == -Y
        return BinaryOperator::CreateNeg(Op1I->getOperand(1),
                                         I.getName());
      else if (Op1I->getOperand(1) == Op0)         // X-(Y+X) == -Y
        return BinaryOperator::CreateNeg(Op1I->getOperand(0),
                                         I.getName());
      else if (ConstantInt *CI1 = dyn_cast<ConstantInt>(I.getOperand(0))) {
        if (ConstantInt *CI2 = dyn_cast<ConstantInt>(Op1I->getOperand(1)))
          // C1-(X+C2) --> (C1-C2)-X
          return BinaryOperator::CreateSub(
            ConstantExpr::getSub(CI1, CI2), Op1I->getOperand(0));
      }
    }

    if (Op1I->hasOneUse()) {
      // Replace (x - (y - z)) with (x + (z - y)) if the (y - z) subexpression
      // is not used by anyone else...
      //
      if (Op1I->getOpcode() == Instruction::Sub) {
        // Swap the two operands of the subexpr...
        Value *IIOp0 = Op1I->getOperand(0), *IIOp1 = Op1I->getOperand(1);
        Op1I->setOperand(0, IIOp1);
        Op1I->setOperand(1, IIOp0);

        // Create the new top level add instruction...
        return BinaryOperator::CreateAdd(Op0, Op1);
      }

      // Replace (A - (A & B)) with (A & ~B) if this is the only use of (A&B)...
      //
      if (Op1I->getOpcode() == Instruction::And &&
          (Op1I->getOperand(0) == Op0 || Op1I->getOperand(1) == Op0)) {
        Value *OtherOp = Op1I->getOperand(Op1I->getOperand(0) == Op0);

        Value *NewNot = Builder->CreateNot(OtherOp, "B.not");
        return BinaryOperator::CreateAnd(Op0, NewNot);
      }

      // 0 - (X sdiv C)  -> (X sdiv -C)
      if (Op1I->getOpcode() == Instruction::SDiv)
        if (ConstantInt *CSI = dyn_cast<ConstantInt>(Op0))
          if (CSI->isZero())
            if (Constant *DivRHS = dyn_cast<Constant>(Op1I->getOperand(1)))
              return BinaryOperator::CreateSDiv(Op1I->getOperand(0),
                                          ConstantExpr::getNeg(DivRHS));

      // X - X*C --> X * (1-C)
      ConstantInt *C2 = 0;
      if (dyn_castFoldableMul(Op1I, C2) == Op0) {
        Constant *CP1 = 
          ConstantExpr::getSub(ConstantInt::get(I.getType(), 1),
                                             C2);
        return BinaryOperator::CreateMul(Op0, CP1);
      }
    }
  }

  if (BinaryOperator *Op0I = dyn_cast<BinaryOperator>(Op0)) {
    if (Op0I->getOpcode() == Instruction::Add) {
      if (Op0I->getOperand(0) == Op1)             // (Y+X)-Y == X
        return ReplaceInstUsesWith(I, Op0I->getOperand(1));
      else if (Op0I->getOperand(1) == Op1)        // (X+Y)-Y == X
        return ReplaceInstUsesWith(I, Op0I->getOperand(0));
    } else if (Op0I->getOpcode() == Instruction::Sub) {
      if (Op0I->getOperand(0) == Op1)             // (X-Y)-X == -Y
        return BinaryOperator::CreateNeg(Op0I->getOperand(1),
                                         I.getName());
    }
  }

  ConstantInt *C1;
  if (Value *X = dyn_castFoldableMul(Op0, C1)) {
    if (X == Op1)  // X*C - X --> X * (C-1)
      return BinaryOperator::CreateMul(Op1, SubOne(C1));

    ConstantInt *C2;   // X*C1 - X*C2 -> X * (C1-C2)
    if (X == dyn_castFoldableMul(Op1, C2))
      return BinaryOperator::CreateMul(X, ConstantExpr::getSub(C1, C2));
  }
  
  // Optimize pointer differences into the same array into a size.  Consider:
  //  &A[10] - &A[0]: we should compile this to "10".
  if (TD) {
    Value *LHSOp, *RHSOp;
    if (match(Op0, m_PtrToInt(m_Value(LHSOp))) &&
        match(Op1, m_PtrToInt(m_Value(RHSOp))))
      if (Value *Res = OptimizePointerDifference(LHSOp, RHSOp, I.getType()))
        return ReplaceInstUsesWith(I, Res);
    
    // trunc(p)-trunc(q) -> trunc(p-q)
    if (match(Op0, m_Trunc(m_PtrToInt(m_Value(LHSOp)))) &&
        match(Op1, m_Trunc(m_PtrToInt(m_Value(RHSOp)))))
      if (Value *Res = OptimizePointerDifference(LHSOp, RHSOp, I.getType()))
        return ReplaceInstUsesWith(I, Res);
  }
  
  return 0;
}

Instruction *InstCombiner::visitFSub(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // If this is a 'B = x-(-A)', change to B = x+A...
  if (Value *V = dyn_castFNegVal(Op1))
    return BinaryOperator::CreateFAdd(Op0, V);

  if (BinaryOperator *Op1I = dyn_cast<BinaryOperator>(Op1)) {
    if (Op1I->getOpcode() == Instruction::FAdd) {
      if (Op1I->getOperand(0) == Op0)              // X-(X+Y) == -Y
        return BinaryOperator::CreateFNeg(Op1I->getOperand(1),
                                          I.getName());
      else if (Op1I->getOperand(1) == Op0)         // X-(Y+X) == -Y
        return BinaryOperator::CreateFNeg(Op1I->getOperand(0),
                                          I.getName());
    }
  }

  return 0;
}

Instruction *InstCombiner::visitMul(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (isa<UndefValue>(Op1))              // undef * X -> 0
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

  // Simplify mul instructions with a constant RHS.
  if (Constant *Op1C = dyn_cast<Constant>(Op1)) {
    if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1C)) {

      // ((X << C1)*C2) == (X * (C2 << C1))
      if (BinaryOperator *SI = dyn_cast<BinaryOperator>(Op0))
        if (SI->getOpcode() == Instruction::Shl)
          if (Constant *ShOp = dyn_cast<Constant>(SI->getOperand(1)))
            return BinaryOperator::CreateMul(SI->getOperand(0),
                                        ConstantExpr::getShl(CI, ShOp));

      if (CI->isZero())
        return ReplaceInstUsesWith(I, Op1C);  // X * 0  == 0
      if (CI->equalsInt(1))                  // X * 1  == X
        return ReplaceInstUsesWith(I, Op0);
      if (CI->isAllOnesValue())              // X * -1 == 0 - X
        return BinaryOperator::CreateNeg(Op0, I.getName());

      const APInt& Val = cast<ConstantInt>(CI)->getValue();
      if (Val.isPowerOf2()) {          // Replace X*(2^C) with X << C
        return BinaryOperator::CreateShl(Op0,
                 ConstantInt::get(Op0->getType(), Val.logBase2()));
      }
    } else if (isa<VectorType>(Op1C->getType())) {
      if (Op1C->isNullValue())
        return ReplaceInstUsesWith(I, Op1C);

      if (ConstantVector *Op1V = dyn_cast<ConstantVector>(Op1C)) {
        if (Op1V->isAllOnesValue())              // X * -1 == 0 - X
          return BinaryOperator::CreateNeg(Op0, I.getName());

        // As above, vector X*splat(1.0) -> X in all defined cases.
        if (Constant *Splat = Op1V->getSplatValue()) {
          if (ConstantInt *CI = dyn_cast<ConstantInt>(Splat))
            if (CI->equalsInt(1))
              return ReplaceInstUsesWith(I, Op0);
        }
      }
    }
    
    if (BinaryOperator *Op0I = dyn_cast<BinaryOperator>(Op0))
      if (Op0I->getOpcode() == Instruction::Add && Op0I->hasOneUse() &&
          isa<ConstantInt>(Op0I->getOperand(1)) && isa<ConstantInt>(Op1C)) {
        // Canonicalize (X+C1)*C2 -> X*C2+C1*C2.
        Value *Add = Builder->CreateMul(Op0I->getOperand(0), Op1C, "tmp");
        Value *C1C2 = Builder->CreateMul(Op1C, Op0I->getOperand(1));
        return BinaryOperator::CreateAdd(Add, C1C2);
        
      }

    // Try to fold constant mul into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;

    if (isa<PHINode>(Op0))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;
  }

  if (Value *Op0v = dyn_castNegVal(Op0))     // -X * -Y = X*Y
    if (Value *Op1v = dyn_castNegVal(Op1))
      return BinaryOperator::CreateMul(Op0v, Op1v);

  // (X / Y) *  Y = X - (X % Y)
  // (X / Y) * -Y = (X % Y) - X
  {
    Value *Op1C = Op1;
    BinaryOperator *BO = dyn_cast<BinaryOperator>(Op0);
    if (!BO ||
        (BO->getOpcode() != Instruction::UDiv && 
         BO->getOpcode() != Instruction::SDiv)) {
      Op1C = Op0;
      BO = dyn_cast<BinaryOperator>(Op1);
    }
    Value *Neg = dyn_castNegVal(Op1C);
    if (BO && BO->hasOneUse() &&
        (BO->getOperand(1) == Op1C || BO->getOperand(1) == Neg) &&
        (BO->getOpcode() == Instruction::UDiv ||
         BO->getOpcode() == Instruction::SDiv)) {
      Value *Op0BO = BO->getOperand(0), *Op1BO = BO->getOperand(1);

      // If the division is exact, X % Y is zero.
      if (SDivOperator *SDiv = dyn_cast<SDivOperator>(BO))
        if (SDiv->isExact()) {
          if (Op1BO == Op1C)
            return ReplaceInstUsesWith(I, Op0BO);
          return BinaryOperator::CreateNeg(Op0BO);
        }

      Value *Rem;
      if (BO->getOpcode() == Instruction::UDiv)
        Rem = Builder->CreateURem(Op0BO, Op1BO);
      else
        Rem = Builder->CreateSRem(Op0BO, Op1BO);
      Rem->takeName(BO);

      if (Op1BO == Op1C)
        return BinaryOperator::CreateSub(Op0BO, Rem);
      return BinaryOperator::CreateSub(Rem, Op0BO);
    }
  }

  /// i1 mul -> i1 and.
  if (I.getType() == Type::getInt1Ty(I.getContext()))
    return BinaryOperator::CreateAnd(Op0, Op1);

  // X*(1 << Y) --> X << Y
  // (1 << Y)*X --> X << Y
  {
    Value *Y;
    if (match(Op0, m_Shl(m_One(), m_Value(Y))))
      return BinaryOperator::CreateShl(Op1, Y);
    if (match(Op1, m_Shl(m_One(), m_Value(Y))))
      return BinaryOperator::CreateShl(Op0, Y);
  }
  
  // If one of the operands of the multiply is a cast from a boolean value, then
  // we know the bool is either zero or one, so this is a 'masking' multiply.
  //   X * Y (where Y is 0 or 1) -> X & (0-Y)
  if (!isa<VectorType>(I.getType())) {
    // -2 is "-1 << 1" so it is all bits set except the low one.
    APInt Negative2(I.getType()->getPrimitiveSizeInBits(), (uint64_t)-2, true);
    
    Value *BoolCast = 0, *OtherOp = 0;
    if (MaskedValueIsZero(Op0, Negative2))
      BoolCast = Op0, OtherOp = Op1;
    else if (MaskedValueIsZero(Op1, Negative2))
      BoolCast = Op1, OtherOp = Op0;

    if (BoolCast) {
      Value *V = Builder->CreateSub(Constant::getNullValue(I.getType()),
                                    BoolCast, "tmp");
      return BinaryOperator::CreateAnd(V, OtherOp);
    }
  }

  return Changed ? &I : 0;
}

Instruction *InstCombiner::visitFMul(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // Simplify mul instructions with a constant RHS...
  if (Constant *Op1C = dyn_cast<Constant>(Op1)) {
    if (ConstantFP *Op1F = dyn_cast<ConstantFP>(Op1C)) {
      // "In IEEE floating point, x*1 is not equivalent to x for nans.  However,
      // ANSI says we can drop signals, so we can do this anyway." (from GCC)
      if (Op1F->isExactlyValue(1.0))
        return ReplaceInstUsesWith(I, Op0);  // Eliminate 'mul double %X, 1.0'
    } else if (isa<VectorType>(Op1C->getType())) {
      if (ConstantVector *Op1V = dyn_cast<ConstantVector>(Op1C)) {
        // As above, vector X*splat(1.0) -> X in all defined cases.
        if (Constant *Splat = Op1V->getSplatValue()) {
          if (ConstantFP *F = dyn_cast<ConstantFP>(Splat))
            if (F->isExactlyValue(1.0))
              return ReplaceInstUsesWith(I, Op0);
        }
      }
    }

    // Try to fold constant mul into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;

    if (isa<PHINode>(Op0))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;
  }

  if (Value *Op0v = dyn_castFNegVal(Op0))     // -X * -Y = X*Y
    if (Value *Op1v = dyn_castFNegVal(Op1))
      return BinaryOperator::CreateFMul(Op0v, Op1v);

  return Changed ? &I : 0;
}

/// SimplifyDivRemOfSelect - Try to fold a divide or remainder of a select
/// instruction.
bool InstCombiner::SimplifyDivRemOfSelect(BinaryOperator &I) {
  SelectInst *SI = cast<SelectInst>(I.getOperand(1));
  
  // div/rem X, (Cond ? 0 : Y) -> div/rem X, Y
  int NonNullOperand = -1;
  if (Constant *ST = dyn_cast<Constant>(SI->getOperand(1)))
    if (ST->isNullValue())
      NonNullOperand = 2;
  // div/rem X, (Cond ? Y : 0) -> div/rem X, Y
  if (Constant *ST = dyn_cast<Constant>(SI->getOperand(2)))
    if (ST->isNullValue())
      NonNullOperand = 1;
  
  if (NonNullOperand == -1)
    return false;
  
  Value *SelectCond = SI->getOperand(0);
  
  // Change the div/rem to use 'Y' instead of the select.
  I.setOperand(1, SI->getOperand(NonNullOperand));
  
  // Okay, we know we replace the operand of the div/rem with 'Y' with no
  // problem.  However, the select, or the condition of the select may have
  // multiple uses.  Based on our knowledge that the operand must be non-zero,
  // propagate the known value for the select into other uses of it, and
  // propagate a known value of the condition into its other users.
  
  // If the select and condition only have a single use, don't bother with this,
  // early exit.
  if (SI->use_empty() && SelectCond->hasOneUse())
    return true;
  
  // Scan the current block backward, looking for other uses of SI.
  BasicBlock::iterator BBI = &I, BBFront = I.getParent()->begin();
  
  while (BBI != BBFront) {
    --BBI;
    // If we found a call to a function, we can't assume it will return, so
    // information from below it cannot be propagated above it.
    if (isa<CallInst>(BBI) && !isa<IntrinsicInst>(BBI))
      break;
    
    // Replace uses of the select or its condition with the known values.
    for (Instruction::op_iterator I = BBI->op_begin(), E = BBI->op_end();
         I != E; ++I) {
      if (*I == SI) {
        *I = SI->getOperand(NonNullOperand);
        Worklist.Add(BBI);
      } else if (*I == SelectCond) {
        *I = NonNullOperand == 1 ? ConstantInt::getTrue(BBI->getContext()) :
                                   ConstantInt::getFalse(BBI->getContext());
        Worklist.Add(BBI);
      }
    }
    
    // If we past the instruction, quit looking for it.
    if (&*BBI == SI)
      SI = 0;
    if (&*BBI == SelectCond)
      SelectCond = 0;
    
    // If we ran out of things to eliminate, break out of the loop.
    if (SelectCond == 0 && SI == 0)
      break;
    
  }
  return true;
}


/// This function implements the transforms on div instructions that work
/// regardless of the kind of div instruction it is (udiv, sdiv, or fdiv). It is
/// used by the visitors to those instructions.
/// @brief Transforms common to all three div instructions
Instruction *InstCombiner::commonDivTransforms(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // undef / X -> 0        for integer.
  // undef / X -> undef    for FP (the undef could be a snan).
  if (isa<UndefValue>(Op0)) {
    if (Op0->getType()->isFPOrFPVector())
      return ReplaceInstUsesWith(I, Op0);
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
  }

  // X / undef -> undef
  if (isa<UndefValue>(Op1))
    return ReplaceInstUsesWith(I, Op1);

  return 0;
}

/// This function implements the transforms common to both integer division
/// instructions (udiv and sdiv). It is called by the visitors to those integer
/// division instructions.
/// @brief Common integer divide transforms
Instruction *InstCombiner::commonIDivTransforms(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // (sdiv X, X) --> 1     (udiv X, X) --> 1
  if (Op0 == Op1) {
    if (const VectorType *Ty = dyn_cast<VectorType>(I.getType())) {
      Constant *CI = ConstantInt::get(Ty->getElementType(), 1);
      std::vector<Constant*> Elts(Ty->getNumElements(), CI);
      return ReplaceInstUsesWith(I, ConstantVector::get(Elts));
    }

    Constant *CI = ConstantInt::get(I.getType(), 1);
    return ReplaceInstUsesWith(I, CI);
  }
  
  if (Instruction *Common = commonDivTransforms(I))
    return Common;
  
  // Handle cases involving: [su]div X, (select Cond, Y, Z)
  // This does not apply for fdiv.
  if (isa<SelectInst>(Op1) && SimplifyDivRemOfSelect(I))
    return &I;

  if (ConstantInt *RHS = dyn_cast<ConstantInt>(Op1)) {
    // div X, 1 == X
    if (RHS->equalsInt(1))
      return ReplaceInstUsesWith(I, Op0);

    // (X / C1) / C2  -> X / (C1*C2)
    if (Instruction *LHS = dyn_cast<Instruction>(Op0))
      if (Instruction::BinaryOps(LHS->getOpcode()) == I.getOpcode())
        if (ConstantInt *LHSRHS = dyn_cast<ConstantInt>(LHS->getOperand(1))) {
          if (MultiplyOverflows(RHS, LHSRHS,
                                I.getOpcode()==Instruction::SDiv))
            return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
          else 
            return BinaryOperator::Create(I.getOpcode(), LHS->getOperand(0),
                                      ConstantExpr::getMul(RHS, LHSRHS));
        }

    if (!RHS->isZero()) { // avoid X udiv 0
      if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
        if (Instruction *R = FoldOpIntoSelect(I, SI))
          return R;
      if (isa<PHINode>(Op0))
        if (Instruction *NV = FoldOpIntoPhi(I))
          return NV;
    }
  }

  // 0 / X == 0, we don't need to preserve faults!
  if (ConstantInt *LHS = dyn_cast<ConstantInt>(Op0))
    if (LHS->equalsInt(0))
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

  // It can't be division by zero, hence it must be division by one.
  if (I.getType() == Type::getInt1Ty(I.getContext()))
    return ReplaceInstUsesWith(I, Op0);

  if (ConstantVector *Op1V = dyn_cast<ConstantVector>(Op1)) {
    if (ConstantInt *X = cast_or_null<ConstantInt>(Op1V->getSplatValue()))
      // div X, 1 == X
      if (X->isOne())
        return ReplaceInstUsesWith(I, Op0);
  }

  return 0;
}

Instruction *InstCombiner::visitUDiv(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // Handle the integer div common cases
  if (Instruction *Common = commonIDivTransforms(I))
    return Common;

  if (ConstantInt *C = dyn_cast<ConstantInt>(Op1)) {
    // X udiv C^2 -> X >> C
    // Check to see if this is an unsigned division with an exact power of 2,
    // if so, convert to a right shift.
    if (C->getValue().isPowerOf2())  // 0 not included in isPowerOf2
      return BinaryOperator::CreateLShr(Op0, 
            ConstantInt::get(Op0->getType(), C->getValue().logBase2()));

    // X udiv C, where C >= signbit
    if (C->getValue().isNegative()) {
      Value *IC = Builder->CreateICmpULT( Op0, C);
      return SelectInst::Create(IC, Constant::getNullValue(I.getType()),
                                ConstantInt::get(I.getType(), 1));
    }
  }

  // X udiv (C1 << N), where C1 is "1<<C2"  -->  X >> (N+C2)
  if (BinaryOperator *RHSI = dyn_cast<BinaryOperator>(I.getOperand(1))) {
    if (RHSI->getOpcode() == Instruction::Shl &&
        isa<ConstantInt>(RHSI->getOperand(0))) {
      const APInt& C1 = cast<ConstantInt>(RHSI->getOperand(0))->getValue();
      if (C1.isPowerOf2()) {
        Value *N = RHSI->getOperand(1);
        const Type *NTy = N->getType();
        if (uint32_t C2 = C1.logBase2())
          N = Builder->CreateAdd(N, ConstantInt::get(NTy, C2), "tmp");
        return BinaryOperator::CreateLShr(Op0, N);
      }
    }
  }
  
  // udiv X, (Select Cond, C1, C2) --> Select Cond, (shr X, C1), (shr X, C2)
  // where C1&C2 are powers of two.
  if (SelectInst *SI = dyn_cast<SelectInst>(Op1)) 
    if (ConstantInt *STO = dyn_cast<ConstantInt>(SI->getOperand(1)))
      if (ConstantInt *SFO = dyn_cast<ConstantInt>(SI->getOperand(2)))  {
        const APInt &TVA = STO->getValue(), &FVA = SFO->getValue();
        if (TVA.isPowerOf2() && FVA.isPowerOf2()) {
          // Compute the shift amounts
          uint32_t TSA = TVA.logBase2(), FSA = FVA.logBase2();
          // Construct the "on true" case of the select
          Constant *TC = ConstantInt::get(Op0->getType(), TSA);
          Value *TSI = Builder->CreateLShr(Op0, TC, SI->getName()+".t");
  
          // Construct the "on false" case of the select
          Constant *FC = ConstantInt::get(Op0->getType(), FSA); 
          Value *FSI = Builder->CreateLShr(Op0, FC, SI->getName()+".f");

          // construct the select instruction and return it.
          return SelectInst::Create(SI->getOperand(0), TSI, FSI, SI->getName());
        }
      }
  return 0;
}

Instruction *InstCombiner::visitSDiv(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // Handle the integer div common cases
  if (Instruction *Common = commonIDivTransforms(I))
    return Common;

  if (ConstantInt *RHS = dyn_cast<ConstantInt>(Op1)) {
    // sdiv X, -1 == -X
    if (RHS->isAllOnesValue())
      return BinaryOperator::CreateNeg(Op0);

    // sdiv X, C  -->  ashr X, log2(C)
    if (cast<SDivOperator>(&I)->isExact() &&
        RHS->getValue().isNonNegative() &&
        RHS->getValue().isPowerOf2()) {
      Value *ShAmt = llvm::ConstantInt::get(RHS->getType(),
                                            RHS->getValue().exactLogBase2());
      return BinaryOperator::CreateAShr(Op0, ShAmt, I.getName());
    }

    // -X/C  -->  X/-C  provided the negation doesn't overflow.
    if (SubOperator *Sub = dyn_cast<SubOperator>(Op0))
      if (isa<Constant>(Sub->getOperand(0)) &&
          cast<Constant>(Sub->getOperand(0))->isNullValue() &&
          Sub->hasNoSignedWrap())
        return BinaryOperator::CreateSDiv(Sub->getOperand(1),
                                          ConstantExpr::getNeg(RHS));
  }

  // If the sign bits of both operands are zero (i.e. we can prove they are
  // unsigned inputs), turn this into a udiv.
  if (I.getType()->isInteger()) {
    APInt Mask(APInt::getSignBit(I.getType()->getPrimitiveSizeInBits()));
    if (MaskedValueIsZero(Op0, Mask)) {
      if (MaskedValueIsZero(Op1, Mask)) {
        // X sdiv Y -> X udiv Y, iff X and Y don't have sign bit set
        return BinaryOperator::CreateUDiv(Op0, Op1, I.getName());
      }
      ConstantInt *ShiftedInt;
      if (match(Op1, m_Shl(m_ConstantInt(ShiftedInt), m_Value())) &&
          ShiftedInt->getValue().isPowerOf2()) {
        // X sdiv (1 << Y) -> X udiv (1 << Y) ( -> X u>> Y)
        // Safe because the only negative value (1 << Y) can take on is
        // INT_MIN, and X sdiv INT_MIN == X udiv INT_MIN == 0 if X doesn't have
        // the sign bit set.
        return BinaryOperator::CreateUDiv(Op0, Op1, I.getName());
      }
    }
  }
  
  return 0;
}

Instruction *InstCombiner::visitFDiv(BinaryOperator &I) {
  return commonDivTransforms(I);
}

/// This function implements the transforms on rem instructions that work
/// regardless of the kind of rem instruction it is (urem, srem, or frem). It 
/// is used by the visitors to those instructions.
/// @brief Transforms common to all three rem instructions
Instruction *InstCombiner::commonRemTransforms(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (isa<UndefValue>(Op0)) {             // undef % X -> 0
    if (I.getType()->isFPOrFPVector())
      return ReplaceInstUsesWith(I, Op0);  // X % undef -> undef (could be SNaN)
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
  }
  if (isa<UndefValue>(Op1))
    return ReplaceInstUsesWith(I, Op1);  // X % undef -> undef

  // Handle cases involving: rem X, (select Cond, Y, Z)
  if (isa<SelectInst>(Op1) && SimplifyDivRemOfSelect(I))
    return &I;

  return 0;
}

/// This function implements the transforms common to both integer remainder
/// instructions (urem and srem). It is called by the visitors to those integer
/// remainder instructions.
/// @brief Common integer remainder transforms
Instruction *InstCombiner::commonIRemTransforms(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Instruction *common = commonRemTransforms(I))
    return common;

  // 0 % X == 0 for integer, we don't need to preserve faults!
  if (Constant *LHS = dyn_cast<Constant>(Op0))
    if (LHS->isNullValue())
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

  if (ConstantInt *RHS = dyn_cast<ConstantInt>(Op1)) {
    // X % 0 == undef, we don't need to preserve faults!
    if (RHS->equalsInt(0))
      return ReplaceInstUsesWith(I, UndefValue::get(I.getType()));
    
    if (RHS->equalsInt(1))  // X % 1 == 0
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

    if (Instruction *Op0I = dyn_cast<Instruction>(Op0)) {
      if (SelectInst *SI = dyn_cast<SelectInst>(Op0I)) {
        if (Instruction *R = FoldOpIntoSelect(I, SI))
          return R;
      } else if (isa<PHINode>(Op0I)) {
        if (Instruction *NV = FoldOpIntoPhi(I))
          return NV;
      }

      // See if we can fold away this rem instruction.
      if (SimplifyDemandedInstructionBits(I))
        return &I;
    }
  }

  return 0;
}

Instruction *InstCombiner::visitURem(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Instruction *common = commonIRemTransforms(I))
    return common;
  
  if (ConstantInt *RHS = dyn_cast<ConstantInt>(Op1)) {
    // X urem C^2 -> X and C
    // Check to see if this is an unsigned remainder with an exact power of 2,
    // if so, convert to a bitwise and.
    if (ConstantInt *C = dyn_cast<ConstantInt>(RHS))
      if (C->getValue().isPowerOf2())
        return BinaryOperator::CreateAnd(Op0, SubOne(C));
  }

  if (Instruction *RHSI = dyn_cast<Instruction>(I.getOperand(1))) {
    // Turn A % (C << N), where C is 2^k, into A & ((C << N)-1)  
    if (RHSI->getOpcode() == Instruction::Shl &&
        isa<ConstantInt>(RHSI->getOperand(0))) {
      if (cast<ConstantInt>(RHSI->getOperand(0))->getValue().isPowerOf2()) {
        Constant *N1 = Constant::getAllOnesValue(I.getType());
        Value *Add = Builder->CreateAdd(RHSI, N1, "tmp");
        return BinaryOperator::CreateAnd(Op0, Add);
      }
    }
  }

  // urem X, (select Cond, 2^C1, 2^C2) --> select Cond, (and X, C1), (and X, C2)
  // where C1&C2 are powers of two.
  if (SelectInst *SI = dyn_cast<SelectInst>(Op1)) {
    if (ConstantInt *STO = dyn_cast<ConstantInt>(SI->getOperand(1)))
      if (ConstantInt *SFO = dyn_cast<ConstantInt>(SI->getOperand(2))) {
        // STO == 0 and SFO == 0 handled above.
        if ((STO->getValue().isPowerOf2()) && 
            (SFO->getValue().isPowerOf2())) {
          Value *TrueAnd = Builder->CreateAnd(Op0, SubOne(STO),
                                              SI->getName()+".t");
          Value *FalseAnd = Builder->CreateAnd(Op0, SubOne(SFO),
                                               SI->getName()+".f");
          return SelectInst::Create(SI->getOperand(0), TrueAnd, FalseAnd);
        }
      }
  }
  
  return 0;
}

Instruction *InstCombiner::visitSRem(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // Handle the integer rem common cases
  if (Instruction *Common = commonIRemTransforms(I))
    return Common;
  
  if (Value *RHSNeg = dyn_castNegVal(Op1))
    if (!isa<Constant>(RHSNeg) ||
        (isa<ConstantInt>(RHSNeg) &&
         cast<ConstantInt>(RHSNeg)->getValue().isStrictlyPositive())) {
      // X % -Y -> X % Y
      Worklist.AddValue(I.getOperand(1));
      I.setOperand(1, RHSNeg);
      return &I;
    }

  // If the sign bits of both operands are zero (i.e. we can prove they are
  // unsigned inputs), turn this into a urem.
  if (I.getType()->isInteger()) {
    APInt Mask(APInt::getSignBit(I.getType()->getPrimitiveSizeInBits()));
    if (MaskedValueIsZero(Op1, Mask) && MaskedValueIsZero(Op0, Mask)) {
      // X srem Y -> X urem Y, iff X and Y don't have sign bit set
      return BinaryOperator::CreateURem(Op0, Op1, I.getName());
    }
  }

  // If it's a constant vector, flip any negative values positive.
  if (ConstantVector *RHSV = dyn_cast<ConstantVector>(Op1)) {
    unsigned VWidth = RHSV->getNumOperands();

    bool hasNegative = false;
    for (unsigned i = 0; !hasNegative && i != VWidth; ++i)
      if (ConstantInt *RHS = dyn_cast<ConstantInt>(RHSV->getOperand(i)))
        if (RHS->getValue().isNegative())
          hasNegative = true;

    if (hasNegative) {
      std::vector<Constant *> Elts(VWidth);
      for (unsigned i = 0; i != VWidth; ++i) {
        if (ConstantInt *RHS = dyn_cast<ConstantInt>(RHSV->getOperand(i))) {
          if (RHS->getValue().isNegative())
            Elts[i] = cast<ConstantInt>(ConstantExpr::getNeg(RHS));
          else
            Elts[i] = RHS;
        }
      }

      Constant *NewRHSV = ConstantVector::get(Elts);
      if (NewRHSV != RHSV) {
        Worklist.AddValue(I.getOperand(1));
        I.setOperand(1, NewRHSV);
        return &I;
      }
    }
  }

  return 0;
}

Instruction *InstCombiner::visitFRem(BinaryOperator &I) {
  return commonRemTransforms(I);
}

// isOneBitSet - Return true if there is exactly one bit set in the specified
// constant.
static bool isOneBitSet(const ConstantInt *CI) {
  return CI->getValue().isPowerOf2();
}

/// getICmpCode - Encode a icmp predicate into a three bit mask.  These bits
/// are carefully arranged to allow folding of expressions such as:
///
///      (A < B) | (A > B) --> (A != B)
///
/// Note that this is only valid if the first and second predicates have the
/// same sign. Is illegal to do: (A u< B) | (A s> B) 
///
/// Three bits are used to represent the condition, as follows:
///   0  A > B
///   1  A == B
///   2  A < B
///
/// <=>  Value  Definition
/// 000     0   Always false
/// 001     1   A >  B
/// 010     2   A == B
/// 011     3   A >= B
/// 100     4   A <  B
/// 101     5   A != B
/// 110     6   A <= B
/// 111     7   Always true
///  
static unsigned getICmpCode(const ICmpInst *ICI) {
  switch (ICI->getPredicate()) {
    // False -> 0
  case ICmpInst::ICMP_UGT: return 1;  // 001
  case ICmpInst::ICMP_SGT: return 1;  // 001
  case ICmpInst::ICMP_EQ:  return 2;  // 010
  case ICmpInst::ICMP_UGE: return 3;  // 011
  case ICmpInst::ICMP_SGE: return 3;  // 011
  case ICmpInst::ICMP_ULT: return 4;  // 100
  case ICmpInst::ICMP_SLT: return 4;  // 100
  case ICmpInst::ICMP_NE:  return 5;  // 101
  case ICmpInst::ICMP_ULE: return 6;  // 110
  case ICmpInst::ICMP_SLE: return 6;  // 110
    // True -> 7
  default:
    llvm_unreachable("Invalid ICmp predicate!");
    return 0;
  }
}

/// getFCmpCode - Similar to getICmpCode but for FCmpInst. This encodes a fcmp
/// predicate into a three bit mask. It also returns whether it is an ordered
/// predicate by reference.
static unsigned getFCmpCode(FCmpInst::Predicate CC, bool &isOrdered) {
  isOrdered = false;
  switch (CC) {
  case FCmpInst::FCMP_ORD: isOrdered = true; return 0;  // 000
  case FCmpInst::FCMP_UNO:                   return 0;  // 000
  case FCmpInst::FCMP_OGT: isOrdered = true; return 1;  // 001
  case FCmpInst::FCMP_UGT:                   return 1;  // 001
  case FCmpInst::FCMP_OEQ: isOrdered = true; return 2;  // 010
  case FCmpInst::FCMP_UEQ:                   return 2;  // 010
  case FCmpInst::FCMP_OGE: isOrdered = true; return 3;  // 011
  case FCmpInst::FCMP_UGE:                   return 3;  // 011
  case FCmpInst::FCMP_OLT: isOrdered = true; return 4;  // 100
  case FCmpInst::FCMP_ULT:                   return 4;  // 100
  case FCmpInst::FCMP_ONE: isOrdered = true; return 5;  // 101
  case FCmpInst::FCMP_UNE:                   return 5;  // 101
  case FCmpInst::FCMP_OLE: isOrdered = true; return 6;  // 110
  case FCmpInst::FCMP_ULE:                   return 6;  // 110
    // True -> 7
  default:
    // Not expecting FCMP_FALSE and FCMP_TRUE;
    llvm_unreachable("Unexpected FCmp predicate!");
    return 0;
  }
}

/// getICmpValue - This is the complement of getICmpCode, which turns an
/// opcode and two operands into either a constant true or false, or a brand 
/// new ICmp instruction. The sign is passed in to determine which kind
/// of predicate to use in the new icmp instruction.
static Value *getICmpValue(bool sign, unsigned code, Value *LHS, Value *RHS) {
  switch (code) {
  default: llvm_unreachable("Illegal ICmp code!");
  case  0: return ConstantInt::getFalse(LHS->getContext());
  case  1: 
    if (sign)
      return new ICmpInst(ICmpInst::ICMP_SGT, LHS, RHS);
    else
      return new ICmpInst(ICmpInst::ICMP_UGT, LHS, RHS);
  case  2: return new ICmpInst(ICmpInst::ICMP_EQ,  LHS, RHS);
  case  3: 
    if (sign)
      return new ICmpInst(ICmpInst::ICMP_SGE, LHS, RHS);
    else
      return new ICmpInst(ICmpInst::ICMP_UGE, LHS, RHS);
  case  4: 
    if (sign)
      return new ICmpInst(ICmpInst::ICMP_SLT, LHS, RHS);
    else
      return new ICmpInst(ICmpInst::ICMP_ULT, LHS, RHS);
  case  5: return new ICmpInst(ICmpInst::ICMP_NE,  LHS, RHS);
  case  6: 
    if (sign)
      return new ICmpInst(ICmpInst::ICMP_SLE, LHS, RHS);
    else
      return new ICmpInst(ICmpInst::ICMP_ULE, LHS, RHS);
  case  7: return ConstantInt::getTrue(LHS->getContext());
  }
}

/// getFCmpValue - This is the complement of getFCmpCode, which turns an
/// opcode and two operands into either a FCmp instruction. isordered is passed
/// in to determine which kind of predicate to use in the new fcmp instruction.
static Value *getFCmpValue(bool isordered, unsigned code,
                           Value *LHS, Value *RHS) {
  switch (code) {
  default: llvm_unreachable("Illegal FCmp code!");
  case  0:
    if (isordered)
      return new FCmpInst(FCmpInst::FCMP_ORD, LHS, RHS);
    else
      return new FCmpInst(FCmpInst::FCMP_UNO, LHS, RHS);
  case  1: 
    if (isordered)
      return new FCmpInst(FCmpInst::FCMP_OGT, LHS, RHS);
    else
      return new FCmpInst(FCmpInst::FCMP_UGT, LHS, RHS);
  case  2: 
    if (isordered)
      return new FCmpInst(FCmpInst::FCMP_OEQ, LHS, RHS);
    else
      return new FCmpInst(FCmpInst::FCMP_UEQ, LHS, RHS);
  case  3: 
    if (isordered)
      return new FCmpInst(FCmpInst::FCMP_OGE, LHS, RHS);
    else
      return new FCmpInst(FCmpInst::FCMP_UGE, LHS, RHS);
  case  4: 
    if (isordered)
      return new FCmpInst(FCmpInst::FCMP_OLT, LHS, RHS);
    else
      return new FCmpInst(FCmpInst::FCMP_ULT, LHS, RHS);
  case  5: 
    if (isordered)
      return new FCmpInst(FCmpInst::FCMP_ONE, LHS, RHS);
    else
      return new FCmpInst(FCmpInst::FCMP_UNE, LHS, RHS);
  case  6: 
    if (isordered)
      return new FCmpInst(FCmpInst::FCMP_OLE, LHS, RHS);
    else
      return new FCmpInst(FCmpInst::FCMP_ULE, LHS, RHS);
  case  7: return ConstantInt::getTrue(LHS->getContext());
  }
}

/// PredicatesFoldable - Return true if both predicates match sign or if at
/// least one of them is an equality comparison (which is signless).
static bool PredicatesFoldable(ICmpInst::Predicate p1, ICmpInst::Predicate p2) {
  return (CmpInst::isSigned(p1) == CmpInst::isSigned(p2)) ||
         (CmpInst::isSigned(p1) && ICmpInst::isEquality(p2)) ||
         (CmpInst::isSigned(p2) && ICmpInst::isEquality(p1));
}

namespace { 
// FoldICmpLogical - Implements (icmp1 A, B) & (icmp2 A, B) --> (icmp3 A, B)
struct FoldICmpLogical {
  InstCombiner &IC;
  Value *LHS, *RHS;
  ICmpInst::Predicate pred;
  FoldICmpLogical(InstCombiner &ic, ICmpInst *ICI)
    : IC(ic), LHS(ICI->getOperand(0)), RHS(ICI->getOperand(1)),
      pred(ICI->getPredicate()) {}
  bool shouldApply(Value *V) const {
    if (ICmpInst *ICI = dyn_cast<ICmpInst>(V))
      if (PredicatesFoldable(pred, ICI->getPredicate()))
        return ((ICI->getOperand(0) == LHS && ICI->getOperand(1) == RHS) ||
                (ICI->getOperand(0) == RHS && ICI->getOperand(1) == LHS));
    return false;
  }
  Instruction *apply(Instruction &Log) const {
    ICmpInst *ICI = cast<ICmpInst>(Log.getOperand(0));
    if (ICI->getOperand(0) != LHS) {
      assert(ICI->getOperand(1) == LHS);
      ICI->swapOperands();  // Swap the LHS and RHS of the ICmp
    }

    ICmpInst *RHSICI = cast<ICmpInst>(Log.getOperand(1));
    unsigned LHSCode = getICmpCode(ICI);
    unsigned RHSCode = getICmpCode(RHSICI);
    unsigned Code;
    switch (Log.getOpcode()) {
    case Instruction::And: Code = LHSCode & RHSCode; break;
    case Instruction::Or:  Code = LHSCode | RHSCode; break;
    case Instruction::Xor: Code = LHSCode ^ RHSCode; break;
    default: llvm_unreachable("Illegal logical opcode!"); return 0;
    }

    bool isSigned = RHSICI->isSigned() || ICI->isSigned();
    Value *RV = getICmpValue(isSigned, Code, LHS, RHS);
    if (Instruction *I = dyn_cast<Instruction>(RV))
      return I;
    // Otherwise, it's a constant boolean value...
    return IC.ReplaceInstUsesWith(Log, RV);
  }
};
} // end anonymous namespace

// OptAndOp - This handles expressions of the form ((val OP C1) & C2).  Where
// the Op parameter is 'OP', OpRHS is 'C1', and AndRHS is 'C2'.  Op is
// guaranteed to be a binary operator.
Instruction *InstCombiner::OptAndOp(Instruction *Op,
                                    ConstantInt *OpRHS,
                                    ConstantInt *AndRHS,
                                    BinaryOperator &TheAnd) {
  Value *X = Op->getOperand(0);
  Constant *Together = 0;
  if (!Op->isShift())
    Together = ConstantExpr::getAnd(AndRHS, OpRHS);

  switch (Op->getOpcode()) {
  case Instruction::Xor:
    if (Op->hasOneUse()) {
      // (X ^ C1) & C2 --> (X & C2) ^ (C1&C2)
      Value *And = Builder->CreateAnd(X, AndRHS);
      And->takeName(Op);
      return BinaryOperator::CreateXor(And, Together);
    }
    break;
  case Instruction::Or:
    if (Together == AndRHS) // (X | C) & C --> C
      return ReplaceInstUsesWith(TheAnd, AndRHS);

    if (Op->hasOneUse() && Together != OpRHS) {
      // (X | C1) & C2 --> (X | (C1&C2)) & C2
      Value *Or = Builder->CreateOr(X, Together);
      Or->takeName(Op);
      return BinaryOperator::CreateAnd(Or, AndRHS);
    }
    break;
  case Instruction::Add:
    if (Op->hasOneUse()) {
      // Adding a one to a single bit bit-field should be turned into an XOR
      // of the bit.  First thing to check is to see if this AND is with a
      // single bit constant.
      const APInt& AndRHSV = cast<ConstantInt>(AndRHS)->getValue();

      // If there is only one bit set...
      if (isOneBitSet(cast<ConstantInt>(AndRHS))) {
        // Ok, at this point, we know that we are masking the result of the
        // ADD down to exactly one bit.  If the constant we are adding has
        // no bits set below this bit, then we can eliminate the ADD.
        const APInt& AddRHS = cast<ConstantInt>(OpRHS)->getValue();

        // Check to see if any bits below the one bit set in AndRHSV are set.
        if ((AddRHS & (AndRHSV-1)) == 0) {
          // If not, the only thing that can effect the output of the AND is
          // the bit specified by AndRHSV.  If that bit is set, the effect of
          // the XOR is to toggle the bit.  If it is clear, then the ADD has
          // no effect.
          if ((AddRHS & AndRHSV) == 0) { // Bit is not set, noop
            TheAnd.setOperand(0, X);
            return &TheAnd;
          } else {
            // Pull the XOR out of the AND.
            Value *NewAnd = Builder->CreateAnd(X, AndRHS);
            NewAnd->takeName(Op);
            return BinaryOperator::CreateXor(NewAnd, AndRHS);
          }
        }
      }
    }
    break;

  case Instruction::Shl: {
    // We know that the AND will not produce any of the bits shifted in, so if
    // the anded constant includes them, clear them now!
    //
    uint32_t BitWidth = AndRHS->getType()->getBitWidth();
    uint32_t OpRHSVal = OpRHS->getLimitedValue(BitWidth);
    APInt ShlMask(APInt::getHighBitsSet(BitWidth, BitWidth-OpRHSVal));
    ConstantInt *CI = ConstantInt::get(AndRHS->getContext(),
                                       AndRHS->getValue() & ShlMask);

    if (CI->getValue() == ShlMask) { 
    // Masking out bits that the shift already masks
      return ReplaceInstUsesWith(TheAnd, Op);   // No need for the and.
    } else if (CI != AndRHS) {                  // Reducing bits set in and.
      TheAnd.setOperand(1, CI);
      return &TheAnd;
    }
    break;
  }
  case Instruction::LShr: {
    // We know that the AND will not produce any of the bits shifted in, so if
    // the anded constant includes them, clear them now!  This only applies to
    // unsigned shifts, because a signed shr may bring in set bits!
    //
    uint32_t BitWidth = AndRHS->getType()->getBitWidth();
    uint32_t OpRHSVal = OpRHS->getLimitedValue(BitWidth);
    APInt ShrMask(APInt::getLowBitsSet(BitWidth, BitWidth - OpRHSVal));
    ConstantInt *CI = ConstantInt::get(Op->getContext(),
                                       AndRHS->getValue() & ShrMask);

    if (CI->getValue() == ShrMask) {   
    // Masking out bits that the shift already masks.
      return ReplaceInstUsesWith(TheAnd, Op);
    } else if (CI != AndRHS) {
      TheAnd.setOperand(1, CI);  // Reduce bits set in and cst.
      return &TheAnd;
    }
    break;
  }
  case Instruction::AShr:
    // Signed shr.
    // See if this is shifting in some sign extension, then masking it out
    // with an and.
    if (Op->hasOneUse()) {
      uint32_t BitWidth = AndRHS->getType()->getBitWidth();
      uint32_t OpRHSVal = OpRHS->getLimitedValue(BitWidth);
      APInt ShrMask(APInt::getLowBitsSet(BitWidth, BitWidth - OpRHSVal));
      Constant *C = ConstantInt::get(Op->getContext(),
                                     AndRHS->getValue() & ShrMask);
      if (C == AndRHS) {          // Masking out bits shifted in.
        // (Val ashr C1) & C2 -> (Val lshr C1) & C2
        // Make the argument unsigned.
        Value *ShVal = Op->getOperand(0);
        ShVal = Builder->CreateLShr(ShVal, OpRHS, Op->getName());
        return BinaryOperator::CreateAnd(ShVal, AndRHS, TheAnd.getName());
      }
    }
    break;
  }
  return 0;
}


/// InsertRangeTest - Emit a computation of: (V >= Lo && V < Hi) if Inside is
/// true, otherwise (V < Lo || V >= Hi).  In pratice, we emit the more efficient
/// (V-Lo) <u Hi-Lo.  This method expects that Lo <= Hi. isSigned indicates
/// whether to treat the V, Lo and HI as signed or not. IB is the location to
/// insert new instructions.
Instruction *InstCombiner::InsertRangeTest(Value *V, Constant *Lo, Constant *Hi,
                                           bool isSigned, bool Inside, 
                                           Instruction &IB) {
  assert(cast<ConstantInt>(ConstantExpr::getICmp((isSigned ? 
            ICmpInst::ICMP_SLE:ICmpInst::ICMP_ULE), Lo, Hi))->getZExtValue() &&
         "Lo is not <= Hi in range emission code!");
    
  if (Inside) {
    if (Lo == Hi)  // Trivially false.
      return new ICmpInst(ICmpInst::ICMP_NE, V, V);

    // V >= Min && V < Hi --> V < Hi
    if (cast<ConstantInt>(Lo)->isMinValue(isSigned)) {
      ICmpInst::Predicate pred = (isSigned ? 
        ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT);
      return new ICmpInst(pred, V, Hi);
    }

    // Emit V-Lo <u Hi-Lo
    Constant *NegLo = ConstantExpr::getNeg(Lo);
    Value *Add = Builder->CreateAdd(V, NegLo, V->getName()+".off");
    Constant *UpperBound = ConstantExpr::getAdd(NegLo, Hi);
    return new ICmpInst(ICmpInst::ICMP_ULT, Add, UpperBound);
  }

  if (Lo == Hi)  // Trivially true.
    return new ICmpInst(ICmpInst::ICMP_EQ, V, V);

  // V < Min || V >= Hi -> V > Hi-1
  Hi = SubOne(cast<ConstantInt>(Hi));
  if (cast<ConstantInt>(Lo)->isMinValue(isSigned)) {
    ICmpInst::Predicate pred = (isSigned ? 
        ICmpInst::ICMP_SGT : ICmpInst::ICMP_UGT);
    return new ICmpInst(pred, V, Hi);
  }

  // Emit V-Lo >u Hi-1-Lo
  // Note that Hi has already had one subtracted from it, above.
  ConstantInt *NegLo = cast<ConstantInt>(ConstantExpr::getNeg(Lo));
  Value *Add = Builder->CreateAdd(V, NegLo, V->getName()+".off");
  Constant *LowerBound = ConstantExpr::getAdd(NegLo, Hi);
  return new ICmpInst(ICmpInst::ICMP_UGT, Add, LowerBound);
}

// isRunOfOnes - Returns true iff Val consists of one contiguous run of 1s with
// any number of 0s on either side.  The 1s are allowed to wrap from LSB to
// MSB, so 0x000FFF0, 0x0000FFFF, and 0xFF0000FF are all runs.  0x0F0F0000 is
// not, since all 1s are not contiguous.
static bool isRunOfOnes(ConstantInt *Val, uint32_t &MB, uint32_t &ME) {
  const APInt& V = Val->getValue();
  uint32_t BitWidth = Val->getType()->getBitWidth();
  if (!APIntOps::isShiftedMask(BitWidth, V)) return false;

  // look for the first zero bit after the run of ones
  MB = BitWidth - ((V - 1) ^ V).countLeadingZeros();
  // look for the first non-zero bit
  ME = V.getActiveBits(); 
  return true;
}

/// FoldLogicalPlusAnd - This is part of an expression (LHS +/- RHS) & Mask,
/// where isSub determines whether the operator is a sub.  If we can fold one of
/// the following xforms:
/// 
/// ((A & N) +/- B) & Mask -> (A +/- B) & Mask iff N&Mask == Mask
/// ((A | N) +/- B) & Mask -> (A +/- B) & Mask iff N&Mask == 0
/// ((A ^ N) +/- B) & Mask -> (A +/- B) & Mask iff N&Mask == 0
///
/// return (A +/- B).
///
Value *InstCombiner::FoldLogicalPlusAnd(Value *LHS, Value *RHS,
                                        ConstantInt *Mask, bool isSub,
                                        Instruction &I) {
  Instruction *LHSI = dyn_cast<Instruction>(LHS);
  if (!LHSI || LHSI->getNumOperands() != 2 ||
      !isa<ConstantInt>(LHSI->getOperand(1))) return 0;

  ConstantInt *N = cast<ConstantInt>(LHSI->getOperand(1));

  switch (LHSI->getOpcode()) {
  default: return 0;
  case Instruction::And:
    if (ConstantExpr::getAnd(N, Mask) == Mask) {
      // If the AndRHS is a power of two minus one (0+1+), this is simple.
      if ((Mask->getValue().countLeadingZeros() + 
           Mask->getValue().countPopulation()) == 
          Mask->getValue().getBitWidth())
        break;

      // Otherwise, if Mask is 0+1+0+, and if B is known to have the low 0+
      // part, we don't need any explicit masks to take them out of A.  If that
      // is all N is, ignore it.
      uint32_t MB = 0, ME = 0;
      if (isRunOfOnes(Mask, MB, ME)) {  // begin/end bit of run, inclusive
        uint32_t BitWidth = cast<IntegerType>(RHS->getType())->getBitWidth();
        APInt Mask(APInt::getLowBitsSet(BitWidth, MB-1));
        if (MaskedValueIsZero(RHS, Mask))
          break;
      }
    }
    return 0;
  case Instruction::Or:
  case Instruction::Xor:
    // If the AndRHS is a power of two minus one (0+1+), and N&Mask == 0
    if ((Mask->getValue().countLeadingZeros() + 
         Mask->getValue().countPopulation()) == Mask->getValue().getBitWidth()
        && ConstantExpr::getAnd(N, Mask)->isNullValue())
      break;
    return 0;
  }
  
  if (isSub)
    return Builder->CreateSub(LHSI->getOperand(0), RHS, "fold");
  return Builder->CreateAdd(LHSI->getOperand(0), RHS, "fold");
}

/// FoldAndOfICmps - Fold (icmp)&(icmp) if possible.
Instruction *InstCombiner::FoldAndOfICmps(Instruction &I,
                                          ICmpInst *LHS, ICmpInst *RHS) {
  Value *Val, *Val2;
  ConstantInt *LHSCst, *RHSCst;
  ICmpInst::Predicate LHSCC, RHSCC;
  
  // This only handles icmp of constants: (icmp1 A, C1) & (icmp2 B, C2).
  if (!match(LHS, m_ICmp(LHSCC, m_Value(Val),
                         m_ConstantInt(LHSCst))) ||
      !match(RHS, m_ICmp(RHSCC, m_Value(Val2),
                         m_ConstantInt(RHSCst))))
    return 0;
  
  if (LHSCst == RHSCst && LHSCC == RHSCC) {
    // (icmp ult A, C) & (icmp ult B, C) --> (icmp ult (A|B), C)
    // where C is a power of 2
    if (LHSCC == ICmpInst::ICMP_ULT &&
        LHSCst->getValue().isPowerOf2()) {
      Value *NewOr = Builder->CreateOr(Val, Val2);
      return new ICmpInst(LHSCC, NewOr, LHSCst);
    }
    
    // (icmp eq A, 0) & (icmp eq B, 0) --> (icmp eq (A|B), 0)
    if (LHSCC == ICmpInst::ICMP_EQ && LHSCst->isZero()) {
      Value *NewOr = Builder->CreateOr(Val, Val2);
      return new ICmpInst(LHSCC, NewOr, LHSCst);
    }
  }
  
  // From here on, we only handle:
  //    (icmp1 A, C1) & (icmp2 A, C2) --> something simpler.
  if (Val != Val2) return 0;
  
  // ICMP_[US][GL]E X, CST is folded to ICMP_[US][GL]T elsewhere.
  if (LHSCC == ICmpInst::ICMP_UGE || LHSCC == ICmpInst::ICMP_ULE ||
      RHSCC == ICmpInst::ICMP_UGE || RHSCC == ICmpInst::ICMP_ULE ||
      LHSCC == ICmpInst::ICMP_SGE || LHSCC == ICmpInst::ICMP_SLE ||
      RHSCC == ICmpInst::ICMP_SGE || RHSCC == ICmpInst::ICMP_SLE)
    return 0;
  
  // We can't fold (ugt x, C) & (sgt x, C2).
  if (!PredicatesFoldable(LHSCC, RHSCC))
    return 0;
    
  // Ensure that the larger constant is on the RHS.
  bool ShouldSwap;
  if (CmpInst::isSigned(LHSCC) ||
      (ICmpInst::isEquality(LHSCC) && 
       CmpInst::isSigned(RHSCC)))
    ShouldSwap = LHSCst->getValue().sgt(RHSCst->getValue());
  else
    ShouldSwap = LHSCst->getValue().ugt(RHSCst->getValue());
    
  if (ShouldSwap) {
    std::swap(LHS, RHS);
    std::swap(LHSCst, RHSCst);
    std::swap(LHSCC, RHSCC);
  }

  // At this point, we know we have have two icmp instructions
  // comparing a value against two constants and and'ing the result
  // together.  Because of the above check, we know that we only have
  // icmp eq, icmp ne, icmp [su]lt, and icmp [SU]gt here. We also know 
  // (from the FoldICmpLogical check above), that the two constants 
  // are not equal and that the larger constant is on the RHS
  assert(LHSCst != RHSCst && "Compares not folded above?");

  switch (LHSCC) {
  default: llvm_unreachable("Unknown integer condition code!");
  case ICmpInst::ICMP_EQ:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:         // (X == 13 & X == 15) -> false
    case ICmpInst::ICMP_UGT:        // (X == 13 & X >  15) -> false
    case ICmpInst::ICMP_SGT:        // (X == 13 & X >  15) -> false
      return ReplaceInstUsesWith(I, ConstantInt::getFalse(I.getContext()));
    case ICmpInst::ICMP_NE:         // (X == 13 & X != 15) -> X == 13
    case ICmpInst::ICMP_ULT:        // (X == 13 & X <  15) -> X == 13
    case ICmpInst::ICMP_SLT:        // (X == 13 & X <  15) -> X == 13
      return ReplaceInstUsesWith(I, LHS);
    }
  case ICmpInst::ICMP_NE:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_ULT:
      if (LHSCst == SubOne(RHSCst)) // (X != 13 & X u< 14) -> X < 13
        return new ICmpInst(ICmpInst::ICMP_ULT, Val, LHSCst);
      break;                        // (X != 13 & X u< 15) -> no change
    case ICmpInst::ICMP_SLT:
      if (LHSCst == SubOne(RHSCst)) // (X != 13 & X s< 14) -> X < 13
        return new ICmpInst(ICmpInst::ICMP_SLT, Val, LHSCst);
      break;                        // (X != 13 & X s< 15) -> no change
    case ICmpInst::ICMP_EQ:         // (X != 13 & X == 15) -> X == 15
    case ICmpInst::ICMP_UGT:        // (X != 13 & X u> 15) -> X u> 15
    case ICmpInst::ICMP_SGT:        // (X != 13 & X s> 15) -> X s> 15
      return ReplaceInstUsesWith(I, RHS);
    case ICmpInst::ICMP_NE:
      if (LHSCst == SubOne(RHSCst)){// (X != 13 & X != 14) -> X-13 >u 1
        Constant *AddCST = ConstantExpr::getNeg(LHSCst);
        Value *Add = Builder->CreateAdd(Val, AddCST, Val->getName()+".off");
        return new ICmpInst(ICmpInst::ICMP_UGT, Add,
                            ConstantInt::get(Add->getType(), 1));
      }
      break;                        // (X != 13 & X != 15) -> no change
    }
    break;
  case ICmpInst::ICMP_ULT:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:         // (X u< 13 & X == 15) -> false
    case ICmpInst::ICMP_UGT:        // (X u< 13 & X u> 15) -> false
      return ReplaceInstUsesWith(I, ConstantInt::getFalse(I.getContext()));
    case ICmpInst::ICMP_SGT:        // (X u< 13 & X s> 15) -> no change
      break;
    case ICmpInst::ICMP_NE:         // (X u< 13 & X != 15) -> X u< 13
    case ICmpInst::ICMP_ULT:        // (X u< 13 & X u< 15) -> X u< 13
      return ReplaceInstUsesWith(I, LHS);
    case ICmpInst::ICMP_SLT:        // (X u< 13 & X s< 15) -> no change
      break;
    }
    break;
  case ICmpInst::ICMP_SLT:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:         // (X s< 13 & X == 15) -> false
    case ICmpInst::ICMP_SGT:        // (X s< 13 & X s> 15) -> false
      return ReplaceInstUsesWith(I, ConstantInt::getFalse(I.getContext()));
    case ICmpInst::ICMP_UGT:        // (X s< 13 & X u> 15) -> no change
      break;
    case ICmpInst::ICMP_NE:         // (X s< 13 & X != 15) -> X < 13
    case ICmpInst::ICMP_SLT:        // (X s< 13 & X s< 15) -> X < 13
      return ReplaceInstUsesWith(I, LHS);
    case ICmpInst::ICMP_ULT:        // (X s< 13 & X u< 15) -> no change
      break;
    }
    break;
  case ICmpInst::ICMP_UGT:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:         // (X u> 13 & X == 15) -> X == 15
    case ICmpInst::ICMP_UGT:        // (X u> 13 & X u> 15) -> X u> 15
      return ReplaceInstUsesWith(I, RHS);
    case ICmpInst::ICMP_SGT:        // (X u> 13 & X s> 15) -> no change
      break;
    case ICmpInst::ICMP_NE:
      if (RHSCst == AddOne(LHSCst)) // (X u> 13 & X != 14) -> X u> 14
        return new ICmpInst(LHSCC, Val, RHSCst);
      break;                        // (X u> 13 & X != 15) -> no change
    case ICmpInst::ICMP_ULT:        // (X u> 13 & X u< 15) -> (X-14) <u 1
      return InsertRangeTest(Val, AddOne(LHSCst),
                             RHSCst, false, true, I);
    case ICmpInst::ICMP_SLT:        // (X u> 13 & X s< 15) -> no change
      break;
    }
    break;
  case ICmpInst::ICMP_SGT:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:         // (X s> 13 & X == 15) -> X == 15
    case ICmpInst::ICMP_SGT:        // (X s> 13 & X s> 15) -> X s> 15
      return ReplaceInstUsesWith(I, RHS);
    case ICmpInst::ICMP_UGT:        // (X s> 13 & X u> 15) -> no change
      break;
    case ICmpInst::ICMP_NE:
      if (RHSCst == AddOne(LHSCst)) // (X s> 13 & X != 14) -> X s> 14
        return new ICmpInst(LHSCC, Val, RHSCst);
      break;                        // (X s> 13 & X != 15) -> no change
    case ICmpInst::ICMP_SLT:        // (X s> 13 & X s< 15) -> (X-14) s< 1
      return InsertRangeTest(Val, AddOne(LHSCst),
                             RHSCst, true, true, I);
    case ICmpInst::ICMP_ULT:        // (X s> 13 & X u< 15) -> no change
      break;
    }
    break;
  }
 
  return 0;
}

Instruction *InstCombiner::FoldAndOfFCmps(Instruction &I, FCmpInst *LHS,
                                          FCmpInst *RHS) {
  
  if (LHS->getPredicate() == FCmpInst::FCMP_ORD &&
      RHS->getPredicate() == FCmpInst::FCMP_ORD) {
    // (fcmp ord x, c) & (fcmp ord y, c)  -> (fcmp ord x, y)
    if (ConstantFP *LHSC = dyn_cast<ConstantFP>(LHS->getOperand(1)))
      if (ConstantFP *RHSC = dyn_cast<ConstantFP>(RHS->getOperand(1))) {
        // If either of the constants are nans, then the whole thing returns
        // false.
        if (LHSC->getValueAPF().isNaN() || RHSC->getValueAPF().isNaN())
          return ReplaceInstUsesWith(I, ConstantInt::getFalse(I.getContext()));
        return new FCmpInst(FCmpInst::FCMP_ORD,
                            LHS->getOperand(0), RHS->getOperand(0));
      }
    
    // Handle vector zeros.  This occurs because the canonical form of
    // "fcmp ord x,x" is "fcmp ord x, 0".
    if (isa<ConstantAggregateZero>(LHS->getOperand(1)) &&
        isa<ConstantAggregateZero>(RHS->getOperand(1)))
      return new FCmpInst(FCmpInst::FCMP_ORD,
                          LHS->getOperand(0), RHS->getOperand(0));
    return 0;
  }
  
  Value *Op0LHS = LHS->getOperand(0), *Op0RHS = LHS->getOperand(1);
  Value *Op1LHS = RHS->getOperand(0), *Op1RHS = RHS->getOperand(1);
  FCmpInst::Predicate Op0CC = LHS->getPredicate(), Op1CC = RHS->getPredicate();
  
  
  if (Op0LHS == Op1RHS && Op0RHS == Op1LHS) {
    // Swap RHS operands to match LHS.
    Op1CC = FCmpInst::getSwappedPredicate(Op1CC);
    std::swap(Op1LHS, Op1RHS);
  }
  
  if (Op0LHS == Op1LHS && Op0RHS == Op1RHS) {
    // Simplify (fcmp cc0 x, y) & (fcmp cc1 x, y).
    if (Op0CC == Op1CC)
      return new FCmpInst((FCmpInst::Predicate)Op0CC, Op0LHS, Op0RHS);
    
    if (Op0CC == FCmpInst::FCMP_FALSE || Op1CC == FCmpInst::FCMP_FALSE)
      return ReplaceInstUsesWith(I, ConstantInt::getFalse(I.getContext()));
    if (Op0CC == FCmpInst::FCMP_TRUE)
      return ReplaceInstUsesWith(I, RHS);
    if (Op1CC == FCmpInst::FCMP_TRUE)
      return ReplaceInstUsesWith(I, LHS);
    
    bool Op0Ordered;
    bool Op1Ordered;
    unsigned Op0Pred = getFCmpCode(Op0CC, Op0Ordered);
    unsigned Op1Pred = getFCmpCode(Op1CC, Op1Ordered);
    if (Op1Pred == 0) {
      std::swap(LHS, RHS);
      std::swap(Op0Pred, Op1Pred);
      std::swap(Op0Ordered, Op1Ordered);
    }
    if (Op0Pred == 0) {
      // uno && ueq -> uno && (uno || eq) -> ueq
      // ord && olt -> ord && (ord && lt) -> olt
      if (Op0Ordered == Op1Ordered)
        return ReplaceInstUsesWith(I, RHS);
      
      // uno && oeq -> uno && (ord && eq) -> false
      // uno && ord -> false
      if (!Op0Ordered)
        return ReplaceInstUsesWith(I, ConstantInt::getFalse(I.getContext()));
      // ord && ueq -> ord && (uno || eq) -> oeq
      return cast<Instruction>(getFCmpValue(true, Op1Pred, Op0LHS, Op0RHS));
    }
  }

  return 0;
}


Instruction *InstCombiner::visitAnd(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifyAndInst(Op0, Op1, TD))
    return ReplaceInstUsesWith(I, V);

  // See if we can simplify any instructions used by the instruction whose sole 
  // purpose is to compute bits we don't care about.
  if (SimplifyDemandedInstructionBits(I))
    return &I;  

  if (ConstantInt *AndRHS = dyn_cast<ConstantInt>(Op1)) {
    const APInt &AndRHSMask = AndRHS->getValue();
    APInt NotAndRHS(~AndRHSMask);

    // Optimize a variety of ((val OP C1) & C2) combinations...
    if (BinaryOperator *Op0I = dyn_cast<BinaryOperator>(Op0)) {
      Value *Op0LHS = Op0I->getOperand(0);
      Value *Op0RHS = Op0I->getOperand(1);
      switch (Op0I->getOpcode()) {
      default: break;
      case Instruction::Xor:
      case Instruction::Or:
        // If the mask is only needed on one incoming arm, push it up.
        if (!Op0I->hasOneUse()) break;
          
        if (MaskedValueIsZero(Op0LHS, NotAndRHS)) {
          // Not masking anything out for the LHS, move to RHS.
          Value *NewRHS = Builder->CreateAnd(Op0RHS, AndRHS,
                                             Op0RHS->getName()+".masked");
          return BinaryOperator::Create(Op0I->getOpcode(), Op0LHS, NewRHS);
        }
        if (!isa<Constant>(Op0RHS) &&
            MaskedValueIsZero(Op0RHS, NotAndRHS)) {
          // Not masking anything out for the RHS, move to LHS.
          Value *NewLHS = Builder->CreateAnd(Op0LHS, AndRHS,
                                             Op0LHS->getName()+".masked");
          return BinaryOperator::Create(Op0I->getOpcode(), NewLHS, Op0RHS);
        }

        break;
      case Instruction::Add:
        // ((A & N) + B) & AndRHS -> (A + B) & AndRHS iff N&AndRHS == AndRHS.
        // ((A | N) + B) & AndRHS -> (A + B) & AndRHS iff N&AndRHS == 0
        // ((A ^ N) + B) & AndRHS -> (A + B) & AndRHS iff N&AndRHS == 0
        if (Value *V = FoldLogicalPlusAnd(Op0LHS, Op0RHS, AndRHS, false, I))
          return BinaryOperator::CreateAnd(V, AndRHS);
        if (Value *V = FoldLogicalPlusAnd(Op0RHS, Op0LHS, AndRHS, false, I))
          return BinaryOperator::CreateAnd(V, AndRHS);  // Add commutes
        break;

      case Instruction::Sub:
        // ((A & N) - B) & AndRHS -> (A - B) & AndRHS iff N&AndRHS == AndRHS.
        // ((A | N) - B) & AndRHS -> (A - B) & AndRHS iff N&AndRHS == 0
        // ((A ^ N) - B) & AndRHS -> (A - B) & AndRHS iff N&AndRHS == 0
        if (Value *V = FoldLogicalPlusAnd(Op0LHS, Op0RHS, AndRHS, true, I))
          return BinaryOperator::CreateAnd(V, AndRHS);

        // (A - N) & AndRHS -> -N & AndRHS iff A&AndRHS==0 and AndRHS
        // has 1's for all bits that the subtraction with A might affect.
        if (Op0I->hasOneUse()) {
          uint32_t BitWidth = AndRHSMask.getBitWidth();
          uint32_t Zeros = AndRHSMask.countLeadingZeros();
          APInt Mask = APInt::getLowBitsSet(BitWidth, BitWidth - Zeros);

          ConstantInt *A = dyn_cast<ConstantInt>(Op0LHS);
          if (!(A && A->isZero()) &&               // avoid infinite recursion.
              MaskedValueIsZero(Op0LHS, Mask)) {
            Value *NewNeg = Builder->CreateNeg(Op0RHS);
            return BinaryOperator::CreateAnd(NewNeg, AndRHS);
          }
        }
        break;

      case Instruction::Shl:
      case Instruction::LShr:
        // (1 << x) & 1 --> zext(x == 0)
        // (1 >> x) & 1 --> zext(x == 0)
        if (AndRHSMask == 1 && Op0LHS == AndRHS) {
          Value *NewICmp =
            Builder->CreateICmpEQ(Op0RHS, Constant::getNullValue(I.getType()));
          return new ZExtInst(NewICmp, I.getType());
        }
        break;
      }

      if (ConstantInt *Op0CI = dyn_cast<ConstantInt>(Op0I->getOperand(1)))
        if (Instruction *Res = OptAndOp(Op0I, Op0CI, AndRHS, I))
          return Res;
    } else if (CastInst *CI = dyn_cast<CastInst>(Op0)) {
      // If this is an integer truncation or change from signed-to-unsigned, and
      // if the source is an and/or with immediate, transform it.  This
      // frequently occurs for bitfield accesses.
      if (Instruction *CastOp = dyn_cast<Instruction>(CI->getOperand(0))) {
        if ((isa<TruncInst>(CI) || isa<BitCastInst>(CI)) &&
            CastOp->getNumOperands() == 2)
          if (ConstantInt *AndCI =dyn_cast<ConstantInt>(CastOp->getOperand(1))){
            if (CastOp->getOpcode() == Instruction::And) {
              // Change: and (cast (and X, C1) to T), C2
              // into  : and (cast X to T), trunc_or_bitcast(C1)&C2
              // This will fold the two constants together, which may allow 
              // other simplifications.
              Value *NewCast = Builder->CreateTruncOrBitCast(
                CastOp->getOperand(0), I.getType(), 
                CastOp->getName()+".shrunk");
              // trunc_or_bitcast(C1)&C2
              Constant *C3 = ConstantExpr::getTruncOrBitCast(AndCI,I.getType());
              C3 = ConstantExpr::getAnd(C3, AndRHS);
              return BinaryOperator::CreateAnd(NewCast, C3);
            } else if (CastOp->getOpcode() == Instruction::Or) {
              // Change: and (cast (or X, C1) to T), C2
              // into  : trunc(C1)&C2 iff trunc(C1)&C2 == C2
              Constant *C3 = ConstantExpr::getTruncOrBitCast(AndCI,I.getType());
              if (ConstantExpr::getAnd(C3, AndRHS) == AndRHS)
                // trunc(C1)&C2
                return ReplaceInstUsesWith(I, AndRHS);
            }
          }
      }
    }

    // Try to fold constant and into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;
    if (isa<PHINode>(Op0))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;
  }


  // (~A & ~B) == (~(A | B)) - De Morgan's Law
  if (Value *Op0NotVal = dyn_castNotVal(Op0))
    if (Value *Op1NotVal = dyn_castNotVal(Op1))
      if (Op0->hasOneUse() && Op1->hasOneUse()) {
        Value *Or = Builder->CreateOr(Op0NotVal, Op1NotVal,
                                      I.getName()+".demorgan");
        return BinaryOperator::CreateNot(Or);
      }

  {
    Value *A = 0, *B = 0, *C = 0, *D = 0;
    // (A|B) & ~(A&B) -> A^B
    if (match(Op0, m_Or(m_Value(A), m_Value(B))) &&
        match(Op1, m_Not(m_And(m_Value(C), m_Value(D)))) &&
        ((A == C && B == D) || (A == D && B == C)))
      return BinaryOperator::CreateXor(A, B);
    
    // ~(A&B) & (A|B) -> A^B
    if (match(Op1, m_Or(m_Value(A), m_Value(B))) &&
        match(Op0, m_Not(m_And(m_Value(C), m_Value(D)))) &&
        ((A == C && B == D) || (A == D && B == C)))
      return BinaryOperator::CreateXor(A, B);
    
    if (Op0->hasOneUse() &&
        match(Op0, m_Xor(m_Value(A), m_Value(B)))) {
      if (A == Op1) {                                // (A^B)&A -> A&(A^B)
        I.swapOperands();     // Simplify below
        std::swap(Op0, Op1);
      } else if (B == Op1) {                         // (A^B)&B -> B&(B^A)
        cast<BinaryOperator>(Op0)->swapOperands();
        I.swapOperands();     // Simplify below
        std::swap(Op0, Op1);
      }
    }

    if (Op1->hasOneUse() &&
        match(Op1, m_Xor(m_Value(A), m_Value(B)))) {
      if (B == Op0) {                                // B&(A^B) -> B&(B^A)
        cast<BinaryOperator>(Op1)->swapOperands();
        std::swap(A, B);
      }
      if (A == Op0)                                // A&(A^B) -> A & ~B
        return BinaryOperator::CreateAnd(A, Builder->CreateNot(B, "tmp"));
    }

    // (A&((~A)|B)) -> A&B
    if (match(Op0, m_Or(m_Not(m_Specific(Op1)), m_Value(A))) ||
        match(Op0, m_Or(m_Value(A), m_Not(m_Specific(Op1)))))
      return BinaryOperator::CreateAnd(A, Op1);
    if (match(Op1, m_Or(m_Not(m_Specific(Op0)), m_Value(A))) ||
        match(Op1, m_Or(m_Value(A), m_Not(m_Specific(Op0)))))
      return BinaryOperator::CreateAnd(A, Op0);
  }
  
  if (ICmpInst *RHS = dyn_cast<ICmpInst>(Op1)) {
    // (icmp1 A, B) & (icmp2 A, B) --> (icmp3 A, B)
    if (Instruction *R = AssociativeOpt(I, FoldICmpLogical(*this, RHS)))
      return R;

    if (ICmpInst *LHS = dyn_cast<ICmpInst>(Op0))
      if (Instruction *Res = FoldAndOfICmps(I, LHS, RHS))
        return Res;
  }

  // fold (and (cast A), (cast B)) -> (cast (and A, B))
  if (CastInst *Op0C = dyn_cast<CastInst>(Op0))
    if (CastInst *Op1C = dyn_cast<CastInst>(Op1))
      if (Op0C->getOpcode() == Op1C->getOpcode()) { // same cast kind ?
        const Type *SrcTy = Op0C->getOperand(0)->getType();
        if (SrcTy == Op1C->getOperand(0)->getType() &&
            SrcTy->isIntOrIntVector() &&
            // Only do this if the casts both really cause code to be generated.
            ValueRequiresCast(Op0C->getOpcode(), Op0C->getOperand(0),
                              I.getType()) &&
            ValueRequiresCast(Op1C->getOpcode(), Op1C->getOperand(0), 
                              I.getType())) {
          Value *NewOp = Builder->CreateAnd(Op0C->getOperand(0),
                                            Op1C->getOperand(0), I.getName());
          return CastInst::Create(Op0C->getOpcode(), NewOp, I.getType());
        }
      }
    
  // (X >> Z) & (Y >> Z)  -> (X&Y) >> Z  for all shifts.
  if (BinaryOperator *SI1 = dyn_cast<BinaryOperator>(Op1)) {
    if (BinaryOperator *SI0 = dyn_cast<BinaryOperator>(Op0))
      if (SI0->isShift() && SI0->getOpcode() == SI1->getOpcode() && 
          SI0->getOperand(1) == SI1->getOperand(1) &&
          (SI0->hasOneUse() || SI1->hasOneUse())) {
        Value *NewOp =
          Builder->CreateAnd(SI0->getOperand(0), SI1->getOperand(0),
                             SI0->getName());
        return BinaryOperator::Create(SI1->getOpcode(), NewOp, 
                                      SI1->getOperand(1));
      }
  }

  // If and'ing two fcmp, try combine them into one.
  if (FCmpInst *LHS = dyn_cast<FCmpInst>(I.getOperand(0))) {
    if (FCmpInst *RHS = dyn_cast<FCmpInst>(I.getOperand(1)))
      if (Instruction *Res = FoldAndOfFCmps(I, LHS, RHS))
        return Res;
  }

  return Changed ? &I : 0;
}

/// CollectBSwapParts - Analyze the specified subexpression and see if it is
/// capable of providing pieces of a bswap.  The subexpression provides pieces
/// of a bswap if it is proven that each of the non-zero bytes in the output of
/// the expression came from the corresponding "byte swapped" byte in some other
/// value.  For example, if the current subexpression is "(shl i32 %X, 24)" then
/// we know that the expression deposits the low byte of %X into the high byte
/// of the bswap result and that all other bytes are zero.  This expression is
/// accepted, the high byte of ByteValues is set to X to indicate a correct
/// match.
///
/// This function returns true if the match was unsuccessful and false if so.
/// On entry to the function the "OverallLeftShift" is a signed integer value
/// indicating the number of bytes that the subexpression is later shifted.  For
/// example, if the expression is later right shifted by 16 bits, the
/// OverallLeftShift value would be -2 on entry.  This is used to specify which
/// byte of ByteValues is actually being set.
///
/// Similarly, ByteMask is a bitmask where a bit is clear if its corresponding
/// byte is masked to zero by a user.  For example, in (X & 255), X will be
/// processed with a bytemask of 1.  Because bytemask is 32-bits, this limits
/// this function to working on up to 32-byte (256 bit) values.  ByteMask is
/// always in the local (OverallLeftShift) coordinate space.
///
static bool CollectBSwapParts(Value *V, int OverallLeftShift, uint32_t ByteMask,
                              SmallVector<Value*, 8> &ByteValues) {
  if (Instruction *I = dyn_cast<Instruction>(V)) {
    // If this is an or instruction, it may be an inner node of the bswap.
    if (I->getOpcode() == Instruction::Or) {
      return CollectBSwapParts(I->getOperand(0), OverallLeftShift, ByteMask,
                               ByteValues) ||
             CollectBSwapParts(I->getOperand(1), OverallLeftShift, ByteMask,
                               ByteValues);
    }
  
    // If this is a logical shift by a constant multiple of 8, recurse with
    // OverallLeftShift and ByteMask adjusted.
    if (I->isLogicalShift() && isa<ConstantInt>(I->getOperand(1))) {
      unsigned ShAmt = 
        cast<ConstantInt>(I->getOperand(1))->getLimitedValue(~0U);
      // Ensure the shift amount is defined and of a byte value.
      if ((ShAmt & 7) || (ShAmt > 8*ByteValues.size()))
        return true;

      unsigned ByteShift = ShAmt >> 3;
      if (I->getOpcode() == Instruction::Shl) {
        // X << 2 -> collect(X, +2)
        OverallLeftShift += ByteShift;
        ByteMask >>= ByteShift;
      } else {
        // X >>u 2 -> collect(X, -2)
        OverallLeftShift -= ByteShift;
        ByteMask <<= ByteShift;
        ByteMask &= (~0U >> (32-ByteValues.size()));
      }

      if (OverallLeftShift >= (int)ByteValues.size()) return true;
      if (OverallLeftShift <= -(int)ByteValues.size()) return true;

      return CollectBSwapParts(I->getOperand(0), OverallLeftShift, ByteMask, 
                               ByteValues);
    }

    // If this is a logical 'and' with a mask that clears bytes, clear the
    // corresponding bytes in ByteMask.
    if (I->getOpcode() == Instruction::And &&
        isa<ConstantInt>(I->getOperand(1))) {
      // Scan every byte of the and mask, seeing if the byte is either 0 or 255.
      unsigned NumBytes = ByteValues.size();
      APInt Byte(I->getType()->getPrimitiveSizeInBits(), 255);
      const APInt &AndMask = cast<ConstantInt>(I->getOperand(1))->getValue();
      
      for (unsigned i = 0; i != NumBytes; ++i, Byte <<= 8) {
        // If this byte is masked out by a later operation, we don't care what
        // the and mask is.
        if ((ByteMask & (1 << i)) == 0)
          continue;
        
        // If the AndMask is all zeros for this byte, clear the bit.
        APInt MaskB = AndMask & Byte;
        if (MaskB == 0) {
          ByteMask &= ~(1U << i);
          continue;
        }
        
        // If the AndMask is not all ones for this byte, it's not a bytezap.
        if (MaskB != Byte)
          return true;

        // Otherwise, this byte is kept.
      }

      return CollectBSwapParts(I->getOperand(0), OverallLeftShift, ByteMask, 
                               ByteValues);
    }
  }
  
  // Okay, we got to something that isn't a shift, 'or' or 'and'.  This must be
  // the input value to the bswap.  Some observations: 1) if more than one byte
  // is demanded from this input, then it could not be successfully assembled
  // into a byteswap.  At least one of the two bytes would not be aligned with
  // their ultimate destination.
  if (!isPowerOf2_32(ByteMask)) return true;
  unsigned InputByteNo = CountTrailingZeros_32(ByteMask);
  
  // 2) The input and ultimate destinations must line up: if byte 3 of an i32
  // is demanded, it needs to go into byte 0 of the result.  This means that the
  // byte needs to be shifted until it lands in the right byte bucket.  The
  // shift amount depends on the position: if the byte is coming from the high
  // part of the value (e.g. byte 3) then it must be shifted right.  If from the
  // low part, it must be shifted left.
  unsigned DestByteNo = InputByteNo + OverallLeftShift;
  if (InputByteNo < ByteValues.size()/2) {
    if (ByteValues.size()-1-DestByteNo != InputByteNo)
      return true;
  } else {
    if (ByteValues.size()-1-DestByteNo != InputByteNo)
      return true;
  }
  
  // If the destination byte value is already defined, the values are or'd
  // together, which isn't a bswap (unless it's an or of the same bits).
  if (ByteValues[DestByteNo] && ByteValues[DestByteNo] != V)
    return true;
  ByteValues[DestByteNo] = V;
  return false;
}

/// MatchBSwap - Given an OR instruction, check to see if this is a bswap idiom.
/// If so, insert the new bswap intrinsic and return it.
Instruction *InstCombiner::MatchBSwap(BinaryOperator &I) {
  const IntegerType *ITy = dyn_cast<IntegerType>(I.getType());
  if (!ITy || ITy->getBitWidth() % 16 || 
      // ByteMask only allows up to 32-byte values.
      ITy->getBitWidth() > 32*8) 
    return 0;   // Can only bswap pairs of bytes.  Can't do vectors.
  
  /// ByteValues - For each byte of the result, we keep track of which value
  /// defines each byte.
  SmallVector<Value*, 8> ByteValues;
  ByteValues.resize(ITy->getBitWidth()/8);
    
  // Try to find all the pieces corresponding to the bswap.
  uint32_t ByteMask = ~0U >> (32-ByteValues.size());
  if (CollectBSwapParts(&I, 0, ByteMask, ByteValues))
    return 0;
  
  // Check to see if all of the bytes come from the same value.
  Value *V = ByteValues[0];
  if (V == 0) return 0;  // Didn't find a byte?  Must be zero.
  
  // Check to make sure that all of the bytes come from the same value.
  for (unsigned i = 1, e = ByteValues.size(); i != e; ++i)
    if (ByteValues[i] != V)
      return 0;
  const Type *Tys[] = { ITy };
  Module *M = I.getParent()->getParent()->getParent();
  Function *F = Intrinsic::getDeclaration(M, Intrinsic::bswap, Tys, 1);
  return CallInst::Create(F, V);
}

/// MatchSelectFromAndOr - We have an expression of the form (A&C)|(B&D).  Check
/// If A is (cond?-1:0) and either B or D is ~(cond?-1,0) or (cond?0,-1), then
/// we can simplify this expression to "cond ? C : D or B".
static Instruction *MatchSelectFromAndOr(Value *A, Value *B,
                                         Value *C, Value *D) {
  // If A is not a select of -1/0, this cannot match.
  Value *Cond = 0;
  if (!match(A, m_SelectCst<-1, 0>(m_Value(Cond))))
    return 0;

  // ((cond?-1:0)&C) | (B&(cond?0:-1)) -> cond ? C : B.
  if (match(D, m_SelectCst<0, -1>(m_Specific(Cond))))
    return SelectInst::Create(Cond, C, B);
  if (match(D, m_Not(m_SelectCst<-1, 0>(m_Specific(Cond)))))
    return SelectInst::Create(Cond, C, B);
  // ((cond?-1:0)&C) | ((cond?0:-1)&D) -> cond ? C : D.
  if (match(B, m_SelectCst<0, -1>(m_Specific(Cond))))
    return SelectInst::Create(Cond, C, D);
  if (match(B, m_Not(m_SelectCst<-1, 0>(m_Specific(Cond)))))
    return SelectInst::Create(Cond, C, D);
  return 0;
}

/// FoldOrOfICmps - Fold (icmp)|(icmp) if possible.
Instruction *InstCombiner::FoldOrOfICmps(Instruction &I,
                                         ICmpInst *LHS, ICmpInst *RHS) {
  Value *Val, *Val2;
  ConstantInt *LHSCst, *RHSCst;
  ICmpInst::Predicate LHSCC, RHSCC;
  
  // This only handles icmp of constants: (icmp1 A, C1) | (icmp2 B, C2).
  if (!match(LHS, m_ICmp(LHSCC, m_Value(Val), m_ConstantInt(LHSCst))) ||
      !match(RHS, m_ICmp(RHSCC, m_Value(Val2), m_ConstantInt(RHSCst))))
    return 0;

  
  // (icmp ne A, 0) | (icmp ne B, 0) --> (icmp ne (A|B), 0)
  if (LHSCst == RHSCst && LHSCC == RHSCC &&
      LHSCC == ICmpInst::ICMP_NE && LHSCst->isZero()) {
    Value *NewOr = Builder->CreateOr(Val, Val2);
    return new ICmpInst(LHSCC, NewOr, LHSCst);
  }
  
  // From here on, we only handle:
  //    (icmp1 A, C1) | (icmp2 A, C2) --> something simpler.
  if (Val != Val2) return 0;
  
  // ICMP_[US][GL]E X, CST is folded to ICMP_[US][GL]T elsewhere.
  if (LHSCC == ICmpInst::ICMP_UGE || LHSCC == ICmpInst::ICMP_ULE ||
      RHSCC == ICmpInst::ICMP_UGE || RHSCC == ICmpInst::ICMP_ULE ||
      LHSCC == ICmpInst::ICMP_SGE || LHSCC == ICmpInst::ICMP_SLE ||
      RHSCC == ICmpInst::ICMP_SGE || RHSCC == ICmpInst::ICMP_SLE)
    return 0;
  
  // We can't fold (ugt x, C) | (sgt x, C2).
  if (!PredicatesFoldable(LHSCC, RHSCC))
    return 0;
  
  // Ensure that the larger constant is on the RHS.
  bool ShouldSwap;
  if (CmpInst::isSigned(LHSCC) ||
      (ICmpInst::isEquality(LHSCC) && 
       CmpInst::isSigned(RHSCC)))
    ShouldSwap = LHSCst->getValue().sgt(RHSCst->getValue());
  else
    ShouldSwap = LHSCst->getValue().ugt(RHSCst->getValue());
  
  if (ShouldSwap) {
    std::swap(LHS, RHS);
    std::swap(LHSCst, RHSCst);
    std::swap(LHSCC, RHSCC);
  }
  
  // At this point, we know we have have two icmp instructions
  // comparing a value against two constants and or'ing the result
  // together.  Because of the above check, we know that we only have
  // ICMP_EQ, ICMP_NE, ICMP_LT, and ICMP_GT here. We also know (from the
  // FoldICmpLogical check above), that the two constants are not
  // equal.
  assert(LHSCst != RHSCst && "Compares not folded above?");

  switch (LHSCC) {
  default: llvm_unreachable("Unknown integer condition code!");
  case ICmpInst::ICMP_EQ:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:
      if (LHSCst == SubOne(RHSCst)) {
        // (X == 13 | X == 14) -> X-13 <u 2
        Constant *AddCST = ConstantExpr::getNeg(LHSCst);
        Value *Add = Builder->CreateAdd(Val, AddCST, Val->getName()+".off");
        AddCST = ConstantExpr::getSub(AddOne(RHSCst), LHSCst);
        return new ICmpInst(ICmpInst::ICMP_ULT, Add, AddCST);
      }
      break;                         // (X == 13 | X == 15) -> no change
    case ICmpInst::ICMP_UGT:         // (X == 13 | X u> 14) -> no change
    case ICmpInst::ICMP_SGT:         // (X == 13 | X s> 14) -> no change
      break;
    case ICmpInst::ICMP_NE:          // (X == 13 | X != 15) -> X != 15
    case ICmpInst::ICMP_ULT:         // (X == 13 | X u< 15) -> X u< 15
    case ICmpInst::ICMP_SLT:         // (X == 13 | X s< 15) -> X s< 15
      return ReplaceInstUsesWith(I, RHS);
    }
    break;
  case ICmpInst::ICMP_NE:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:          // (X != 13 | X == 15) -> X != 13
    case ICmpInst::ICMP_UGT:         // (X != 13 | X u> 15) -> X != 13
    case ICmpInst::ICMP_SGT:         // (X != 13 | X s> 15) -> X != 13
      return ReplaceInstUsesWith(I, LHS);
    case ICmpInst::ICMP_NE:          // (X != 13 | X != 15) -> true
    case ICmpInst::ICMP_ULT:         // (X != 13 | X u< 15) -> true
    case ICmpInst::ICMP_SLT:         // (X != 13 | X s< 15) -> true
      return ReplaceInstUsesWith(I, ConstantInt::getTrue(I.getContext()));
    }
    break;
  case ICmpInst::ICMP_ULT:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:         // (X u< 13 | X == 14) -> no change
      break;
    case ICmpInst::ICMP_UGT:        // (X u< 13 | X u> 15) -> (X-13) u> 2
      // If RHSCst is [us]MAXINT, it is always false.  Not handling
      // this can cause overflow.
      if (RHSCst->isMaxValue(false))
        return ReplaceInstUsesWith(I, LHS);
      return InsertRangeTest(Val, LHSCst, AddOne(RHSCst),
                             false, false, I);
    case ICmpInst::ICMP_SGT:        // (X u< 13 | X s> 15) -> no change
      break;
    case ICmpInst::ICMP_NE:         // (X u< 13 | X != 15) -> X != 15
    case ICmpInst::ICMP_ULT:        // (X u< 13 | X u< 15) -> X u< 15
      return ReplaceInstUsesWith(I, RHS);
    case ICmpInst::ICMP_SLT:        // (X u< 13 | X s< 15) -> no change
      break;
    }
    break;
  case ICmpInst::ICMP_SLT:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:         // (X s< 13 | X == 14) -> no change
      break;
    case ICmpInst::ICMP_SGT:        // (X s< 13 | X s> 15) -> (X-13) s> 2
      // If RHSCst is [us]MAXINT, it is always false.  Not handling
      // this can cause overflow.
      if (RHSCst->isMaxValue(true))
        return ReplaceInstUsesWith(I, LHS);
      return InsertRangeTest(Val, LHSCst, AddOne(RHSCst),
                             true, false, I);
    case ICmpInst::ICMP_UGT:        // (X s< 13 | X u> 15) -> no change
      break;
    case ICmpInst::ICMP_NE:         // (X s< 13 | X != 15) -> X != 15
    case ICmpInst::ICMP_SLT:        // (X s< 13 | X s< 15) -> X s< 15
      return ReplaceInstUsesWith(I, RHS);
    case ICmpInst::ICMP_ULT:        // (X s< 13 | X u< 15) -> no change
      break;
    }
    break;
  case ICmpInst::ICMP_UGT:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:         // (X u> 13 | X == 15) -> X u> 13
    case ICmpInst::ICMP_UGT:        // (X u> 13 | X u> 15) -> X u> 13
      return ReplaceInstUsesWith(I, LHS);
    case ICmpInst::ICMP_SGT:        // (X u> 13 | X s> 15) -> no change
      break;
    case ICmpInst::ICMP_NE:         // (X u> 13 | X != 15) -> true
    case ICmpInst::ICMP_ULT:        // (X u> 13 | X u< 15) -> true
      return ReplaceInstUsesWith(I, ConstantInt::getTrue(I.getContext()));
    case ICmpInst::ICMP_SLT:        // (X u> 13 | X s< 15) -> no change
      break;
    }
    break;
  case ICmpInst::ICMP_SGT:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:         // (X s> 13 | X == 15) -> X > 13
    case ICmpInst::ICMP_SGT:        // (X s> 13 | X s> 15) -> X > 13
      return ReplaceInstUsesWith(I, LHS);
    case ICmpInst::ICMP_UGT:        // (X s> 13 | X u> 15) -> no change
      break;
    case ICmpInst::ICMP_NE:         // (X s> 13 | X != 15) -> true
    case ICmpInst::ICMP_SLT:        // (X s> 13 | X s< 15) -> true
      return ReplaceInstUsesWith(I, ConstantInt::getTrue(I.getContext()));
    case ICmpInst::ICMP_ULT:        // (X s> 13 | X u< 15) -> no change
      break;
    }
    break;
  }
  return 0;
}

Instruction *InstCombiner::FoldOrOfFCmps(Instruction &I, FCmpInst *LHS,
                                         FCmpInst *RHS) {
  if (LHS->getPredicate() == FCmpInst::FCMP_UNO &&
      RHS->getPredicate() == FCmpInst::FCMP_UNO && 
      LHS->getOperand(0)->getType() == RHS->getOperand(0)->getType()) {
    if (ConstantFP *LHSC = dyn_cast<ConstantFP>(LHS->getOperand(1)))
      if (ConstantFP *RHSC = dyn_cast<ConstantFP>(RHS->getOperand(1))) {
        // If either of the constants are nans, then the whole thing returns
        // true.
        if (LHSC->getValueAPF().isNaN() || RHSC->getValueAPF().isNaN())
          return ReplaceInstUsesWith(I, ConstantInt::getTrue(I.getContext()));
        
        // Otherwise, no need to compare the two constants, compare the
        // rest.
        return new FCmpInst(FCmpInst::FCMP_UNO,
                            LHS->getOperand(0), RHS->getOperand(0));
      }
    
    // Handle vector zeros.  This occurs because the canonical form of
    // "fcmp uno x,x" is "fcmp uno x, 0".
    if (isa<ConstantAggregateZero>(LHS->getOperand(1)) &&
        isa<ConstantAggregateZero>(RHS->getOperand(1)))
      return new FCmpInst(FCmpInst::FCMP_UNO,
                          LHS->getOperand(0), RHS->getOperand(0));
    
    return 0;
  }
  
  Value *Op0LHS = LHS->getOperand(0), *Op0RHS = LHS->getOperand(1);
  Value *Op1LHS = RHS->getOperand(0), *Op1RHS = RHS->getOperand(1);
  FCmpInst::Predicate Op0CC = LHS->getPredicate(), Op1CC = RHS->getPredicate();
  
  if (Op0LHS == Op1RHS && Op0RHS == Op1LHS) {
    // Swap RHS operands to match LHS.
    Op1CC = FCmpInst::getSwappedPredicate(Op1CC);
    std::swap(Op1LHS, Op1RHS);
  }
  if (Op0LHS == Op1LHS && Op0RHS == Op1RHS) {
    // Simplify (fcmp cc0 x, y) | (fcmp cc1 x, y).
    if (Op0CC == Op1CC)
      return new FCmpInst((FCmpInst::Predicate)Op0CC,
                          Op0LHS, Op0RHS);
    if (Op0CC == FCmpInst::FCMP_TRUE || Op1CC == FCmpInst::FCMP_TRUE)
      return ReplaceInstUsesWith(I, ConstantInt::getTrue(I.getContext()));
    if (Op0CC == FCmpInst::FCMP_FALSE)
      return ReplaceInstUsesWith(I, RHS);
    if (Op1CC == FCmpInst::FCMP_FALSE)
      return ReplaceInstUsesWith(I, LHS);
    bool Op0Ordered;
    bool Op1Ordered;
    unsigned Op0Pred = getFCmpCode(Op0CC, Op0Ordered);
    unsigned Op1Pred = getFCmpCode(Op1CC, Op1Ordered);
    if (Op0Ordered == Op1Ordered) {
      // If both are ordered or unordered, return a new fcmp with
      // or'ed predicates.
      Value *RV = getFCmpValue(Op0Ordered, Op0Pred|Op1Pred, Op0LHS, Op0RHS);
      if (Instruction *I = dyn_cast<Instruction>(RV))
        return I;
      // Otherwise, it's a constant boolean value...
      return ReplaceInstUsesWith(I, RV);
    }
  }
  return 0;
}

/// FoldOrWithConstants - This helper function folds:
///
///     ((A | B) & C1) | (B & C2)
///
/// into:
/// 
///     (A & C1) | B
///
/// when the XOR of the two constants is "all ones" (-1).
Instruction *InstCombiner::FoldOrWithConstants(BinaryOperator &I, Value *Op,
                                               Value *A, Value *B, Value *C) {
  ConstantInt *CI1 = dyn_cast<ConstantInt>(C);
  if (!CI1) return 0;

  Value *V1 = 0;
  ConstantInt *CI2 = 0;
  if (!match(Op, m_And(m_Value(V1), m_ConstantInt(CI2)))) return 0;

  APInt Xor = CI1->getValue() ^ CI2->getValue();
  if (!Xor.isAllOnesValue()) return 0;

  if (V1 == A || V1 == B) {
    Value *NewOp = Builder->CreateAnd((V1 == A) ? B : A, CI1);
    return BinaryOperator::CreateOr(NewOp, V1);
  }

  return 0;
}

Instruction *InstCombiner::visitOr(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifyOrInst(Op0, Op1, TD))
    return ReplaceInstUsesWith(I, V);
  
  
  // See if we can simplify any instructions used by the instruction whose sole 
  // purpose is to compute bits we don't care about.
  if (SimplifyDemandedInstructionBits(I))
    return &I;

  if (ConstantInt *RHS = dyn_cast<ConstantInt>(Op1)) {
    ConstantInt *C1 = 0; Value *X = 0;
    // (X & C1) | C2 --> (X | C2) & (C1|C2)
    if (match(Op0, m_And(m_Value(X), m_ConstantInt(C1))) &&
        isOnlyUse(Op0)) {
      Value *Or = Builder->CreateOr(X, RHS);
      Or->takeName(Op0);
      return BinaryOperator::CreateAnd(Or, 
                         ConstantInt::get(I.getContext(),
                                          RHS->getValue() | C1->getValue()));
    }

    // (X ^ C1) | C2 --> (X | C2) ^ (C1&~C2)
    if (match(Op0, m_Xor(m_Value(X), m_ConstantInt(C1))) &&
        isOnlyUse(Op0)) {
      Value *Or = Builder->CreateOr(X, RHS);
      Or->takeName(Op0);
      return BinaryOperator::CreateXor(Or,
                 ConstantInt::get(I.getContext(),
                                  C1->getValue() & ~RHS->getValue()));
    }

    // Try to fold constant and into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;
    if (isa<PHINode>(Op0))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;
  }

  Value *A = 0, *B = 0;
  ConstantInt *C1 = 0, *C2 = 0;

  // (A | B) | C  and  A | (B | C)                  -> bswap if possible.
  // (A >> B) | (C << D)  and  (A << B) | (B >> C)  -> bswap if possible.
  if (match(Op0, m_Or(m_Value(), m_Value())) ||
      match(Op1, m_Or(m_Value(), m_Value())) ||
      (match(Op0, m_Shift(m_Value(), m_Value())) &&
       match(Op1, m_Shift(m_Value(), m_Value())))) {
    if (Instruction *BSwap = MatchBSwap(I))
      return BSwap;
  }
  
  // (X^C)|Y -> (X|Y)^C iff Y&C == 0
  if (Op0->hasOneUse() &&
      match(Op0, m_Xor(m_Value(A), m_ConstantInt(C1))) &&
      MaskedValueIsZero(Op1, C1->getValue())) {
    Value *NOr = Builder->CreateOr(A, Op1);
    NOr->takeName(Op0);
    return BinaryOperator::CreateXor(NOr, C1);
  }

  // Y|(X^C) -> (X|Y)^C iff Y&C == 0
  if (Op1->hasOneUse() &&
      match(Op1, m_Xor(m_Value(A), m_ConstantInt(C1))) &&
      MaskedValueIsZero(Op0, C1->getValue())) {
    Value *NOr = Builder->CreateOr(A, Op0);
    NOr->takeName(Op0);
    return BinaryOperator::CreateXor(NOr, C1);
  }

  // (A & C)|(B & D)
  Value *C = 0, *D = 0;
  if (match(Op0, m_And(m_Value(A), m_Value(C))) &&
      match(Op1, m_And(m_Value(B), m_Value(D)))) {
    Value *V1 = 0, *V2 = 0, *V3 = 0;
    C1 = dyn_cast<ConstantInt>(C);
    C2 = dyn_cast<ConstantInt>(D);
    if (C1 && C2) {  // (A & C1)|(B & C2)
      // If we have: ((V + N) & C1) | (V & C2)
      // .. and C2 = ~C1 and C2 is 0+1+ and (N & C2) == 0
      // replace with V+N.
      if (C1->getValue() == ~C2->getValue()) {
        if ((C2->getValue() & (C2->getValue()+1)) == 0 && // C2 == 0+1+
            match(A, m_Add(m_Value(V1), m_Value(V2)))) {
          // Add commutes, try both ways.
          if (V1 == B && MaskedValueIsZero(V2, C2->getValue()))
            return ReplaceInstUsesWith(I, A);
          if (V2 == B && MaskedValueIsZero(V1, C2->getValue()))
            return ReplaceInstUsesWith(I, A);
        }
        // Or commutes, try both ways.
        if ((C1->getValue() & (C1->getValue()+1)) == 0 &&
            match(B, m_Add(m_Value(V1), m_Value(V2)))) {
          // Add commutes, try both ways.
          if (V1 == A && MaskedValueIsZero(V2, C1->getValue()))
            return ReplaceInstUsesWith(I, B);
          if (V2 == A && MaskedValueIsZero(V1, C1->getValue()))
            return ReplaceInstUsesWith(I, B);
        }
      }
      
      // ((V | N) & C1) | (V & C2) --> (V|N) & (C1|C2)
      // iff (C1&C2) == 0 and (N&~C1) == 0
      if ((C1->getValue() & C2->getValue()) == 0) {
        if (match(A, m_Or(m_Value(V1), m_Value(V2))) &&
            ((V1 == B && MaskedValueIsZero(V2, ~C1->getValue())) ||  // (V|N)
             (V2 == B && MaskedValueIsZero(V1, ~C1->getValue()))))   // (N|V)
          return BinaryOperator::CreateAnd(A,
                               ConstantInt::get(A->getContext(),
                                                C1->getValue()|C2->getValue()));
        // Or commutes, try both ways.
        if (match(B, m_Or(m_Value(V1), m_Value(V2))) &&
            ((V1 == A && MaskedValueIsZero(V2, ~C2->getValue())) ||  // (V|N)
             (V2 == A && MaskedValueIsZero(V1, ~C2->getValue()))))   // (N|V)
          return BinaryOperator::CreateAnd(B,
                               ConstantInt::get(B->getContext(),
                                                C1->getValue()|C2->getValue()));
      }
    }
    
    // Check to see if we have any common things being and'ed.  If so, find the
    // terms for V1 & (V2|V3).
    if (isOnlyUse(Op0) || isOnlyUse(Op1)) {
      V1 = 0;
      if (A == B)      // (A & C)|(A & D) == A & (C|D)
        V1 = A, V2 = C, V3 = D;
      else if (A == D) // (A & C)|(B & A) == A & (B|C)
        V1 = A, V2 = B, V3 = C;
      else if (C == B) // (A & C)|(C & D) == C & (A|D)
        V1 = C, V2 = A, V3 = D;
      else if (C == D) // (A & C)|(B & C) == C & (A|B)
        V1 = C, V2 = A, V3 = B;
      
      if (V1) {
        Value *Or = Builder->CreateOr(V2, V3, "tmp");
        return BinaryOperator::CreateAnd(V1, Or);
      }
    }

    // (A & (C0?-1:0)) | (B & ~(C0?-1:0)) ->  C0 ? A : B, and commuted variants
    if (Instruction *Match = MatchSelectFromAndOr(A, B, C, D))
      return Match;
    if (Instruction *Match = MatchSelectFromAndOr(B, A, D, C))
      return Match;
    if (Instruction *Match = MatchSelectFromAndOr(C, B, A, D))
      return Match;
    if (Instruction *Match = MatchSelectFromAndOr(D, A, B, C))
      return Match;

    // ((A&~B)|(~A&B)) -> A^B
    if ((match(C, m_Not(m_Specific(D))) &&
         match(B, m_Not(m_Specific(A)))))
      return BinaryOperator::CreateXor(A, D);
    // ((~B&A)|(~A&B)) -> A^B
    if ((match(A, m_Not(m_Specific(D))) &&
         match(B, m_Not(m_Specific(C)))))
      return BinaryOperator::CreateXor(C, D);
    // ((A&~B)|(B&~A)) -> A^B
    if ((match(C, m_Not(m_Specific(B))) &&
         match(D, m_Not(m_Specific(A)))))
      return BinaryOperator::CreateXor(A, B);
    // ((~B&A)|(B&~A)) -> A^B
    if ((match(A, m_Not(m_Specific(B))) &&
         match(D, m_Not(m_Specific(C)))))
      return BinaryOperator::CreateXor(C, B);
  }
  
  // (X >> Z) | (Y >> Z)  -> (X|Y) >> Z  for all shifts.
  if (BinaryOperator *SI1 = dyn_cast<BinaryOperator>(Op1)) {
    if (BinaryOperator *SI0 = dyn_cast<BinaryOperator>(Op0))
      if (SI0->isShift() && SI0->getOpcode() == SI1->getOpcode() && 
          SI0->getOperand(1) == SI1->getOperand(1) &&
          (SI0->hasOneUse() || SI1->hasOneUse())) {
        Value *NewOp = Builder->CreateOr(SI0->getOperand(0), SI1->getOperand(0),
                                         SI0->getName());
        return BinaryOperator::Create(SI1->getOpcode(), NewOp, 
                                      SI1->getOperand(1));
      }
  }

  // ((A|B)&1)|(B&-2) -> (A&1) | B
  if (match(Op0, m_And(m_Or(m_Value(A), m_Value(B)), m_Value(C))) ||
      match(Op0, m_And(m_Value(C), m_Or(m_Value(A), m_Value(B))))) {
    Instruction *Ret = FoldOrWithConstants(I, Op1, A, B, C);
    if (Ret) return Ret;
  }
  // (B&-2)|((A|B)&1) -> (A&1) | B
  if (match(Op1, m_And(m_Or(m_Value(A), m_Value(B)), m_Value(C))) ||
      match(Op1, m_And(m_Value(C), m_Or(m_Value(A), m_Value(B))))) {
    Instruction *Ret = FoldOrWithConstants(I, Op0, A, B, C);
    if (Ret) return Ret;
  }

  // (~A | ~B) == (~(A & B)) - De Morgan's Law
  if (Value *Op0NotVal = dyn_castNotVal(Op0))
    if (Value *Op1NotVal = dyn_castNotVal(Op1))
      if (Op0->hasOneUse() && Op1->hasOneUse()) {
        Value *And = Builder->CreateAnd(Op0NotVal, Op1NotVal,
                                        I.getName()+".demorgan");
        return BinaryOperator::CreateNot(And);
      }

  // (icmp1 A, B) | (icmp2 A, B) --> (icmp3 A, B)
  if (ICmpInst *RHS = dyn_cast<ICmpInst>(I.getOperand(1))) {
    if (Instruction *R = AssociativeOpt(I, FoldICmpLogical(*this, RHS)))
      return R;

    if (ICmpInst *LHS = dyn_cast<ICmpInst>(I.getOperand(0)))
      if (Instruction *Res = FoldOrOfICmps(I, LHS, RHS))
        return Res;
  }
    
  // fold (or (cast A), (cast B)) -> (cast (or A, B))
  if (CastInst *Op0C = dyn_cast<CastInst>(Op0)) {
    if (CastInst *Op1C = dyn_cast<CastInst>(Op1))
      if (Op0C->getOpcode() == Op1C->getOpcode()) {// same cast kind ?
        if (!isa<ICmpInst>(Op0C->getOperand(0)) ||
            !isa<ICmpInst>(Op1C->getOperand(0))) {
          const Type *SrcTy = Op0C->getOperand(0)->getType();
          if (SrcTy == Op1C->getOperand(0)->getType() &&
              SrcTy->isIntOrIntVector() &&
              // Only do this if the casts both really cause code to be
              // generated.
              ValueRequiresCast(Op0C->getOpcode(), Op0C->getOperand(0), 
                                I.getType()) &&
              ValueRequiresCast(Op1C->getOpcode(), Op1C->getOperand(0), 
                                I.getType())) {
            Value *NewOp = Builder->CreateOr(Op0C->getOperand(0),
                                             Op1C->getOperand(0), I.getName());
            return CastInst::Create(Op0C->getOpcode(), NewOp, I.getType());
          }
        }
      }
  }
  
    
  // (fcmp uno x, c) | (fcmp uno y, c)  -> (fcmp uno x, y)
  if (FCmpInst *LHS = dyn_cast<FCmpInst>(I.getOperand(0))) {
    if (FCmpInst *RHS = dyn_cast<FCmpInst>(I.getOperand(1)))
      if (Instruction *Res = FoldOrOfFCmps(I, LHS, RHS))
        return Res;
  }

  return Changed ? &I : 0;
}

namespace {

// XorSelf - Implements: X ^ X --> 0
struct XorSelf {
  Value *RHS;
  XorSelf(Value *rhs) : RHS(rhs) {}
  bool shouldApply(Value *LHS) const { return LHS == RHS; }
  Instruction *apply(BinaryOperator &Xor) const {
    return &Xor;
  }
};

}

Instruction *InstCombiner::visitXor(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (isa<UndefValue>(Op1)) {
    if (isa<UndefValue>(Op0))
      // Handle undef ^ undef -> 0 special case. This is a common
      // idiom (misuse).
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
    return ReplaceInstUsesWith(I, Op1);  // X ^ undef -> undef
  }

  // xor X, X = 0, even if X is nested in a sequence of Xor's.
  if (Instruction *Result = AssociativeOpt(I, XorSelf(Op1))) {
    assert(Result == &I && "AssociativeOpt didn't work?"); Result=Result;
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
  }
  
  // See if we can simplify any instructions used by the instruction whose sole 
  // purpose is to compute bits we don't care about.
  if (SimplifyDemandedInstructionBits(I))
    return &I;
  if (isa<VectorType>(I.getType()))
    if (isa<ConstantAggregateZero>(Op1))
      return ReplaceInstUsesWith(I, Op0);  // X ^ <0,0> -> X

  // Is this a ~ operation?
  if (Value *NotOp = dyn_castNotVal(&I)) {
    if (BinaryOperator *Op0I = dyn_cast<BinaryOperator>(NotOp)) {
      if (Op0I->getOpcode() == Instruction::And || 
          Op0I->getOpcode() == Instruction::Or) {
        // ~(~X & Y) --> (X | ~Y) - De Morgan's Law
        // ~(~X | Y) === (X & ~Y) - De Morgan's Law
        if (dyn_castNotVal(Op0I->getOperand(1)))
          Op0I->swapOperands();
        if (Value *Op0NotVal = dyn_castNotVal(Op0I->getOperand(0))) {
          Value *NotY =
            Builder->CreateNot(Op0I->getOperand(1),
                               Op0I->getOperand(1)->getName()+".not");
          if (Op0I->getOpcode() == Instruction::And)
            return BinaryOperator::CreateOr(Op0NotVal, NotY);
          return BinaryOperator::CreateAnd(Op0NotVal, NotY);
        }
        
        // ~(X & Y) --> (~X | ~Y) - De Morgan's Law
        // ~(X | Y) === (~X & ~Y) - De Morgan's Law
        if (isFreeToInvert(Op0I->getOperand(0)) && 
            isFreeToInvert(Op0I->getOperand(1))) {
          Value *NotX =
            Builder->CreateNot(Op0I->getOperand(0), "notlhs");
          Value *NotY =
            Builder->CreateNot(Op0I->getOperand(1), "notrhs");
          if (Op0I->getOpcode() == Instruction::And)
            return BinaryOperator::CreateOr(NotX, NotY);
          return BinaryOperator::CreateAnd(NotX, NotY);
        }
      }
    }
  }
  
  
  if (ConstantInt *RHS = dyn_cast<ConstantInt>(Op1)) {
    if (RHS->isOne() && Op0->hasOneUse()) {
      // xor (cmp A, B), true = not (cmp A, B) = !cmp A, B
      if (ICmpInst *ICI = dyn_cast<ICmpInst>(Op0))
        return new ICmpInst(ICI->getInversePredicate(),
                            ICI->getOperand(0), ICI->getOperand(1));

      if (FCmpInst *FCI = dyn_cast<FCmpInst>(Op0))
        return new FCmpInst(FCI->getInversePredicate(),
                            FCI->getOperand(0), FCI->getOperand(1));
    }

    // fold (xor(zext(cmp)), 1) and (xor(sext(cmp)), -1) to ext(!cmp).
    if (CastInst *Op0C = dyn_cast<CastInst>(Op0)) {
      if (CmpInst *CI = dyn_cast<CmpInst>(Op0C->getOperand(0))) {
        if (CI->hasOneUse() && Op0C->hasOneUse()) {
          Instruction::CastOps Opcode = Op0C->getOpcode();
          if ((Opcode == Instruction::ZExt || Opcode == Instruction::SExt) &&
              (RHS == ConstantExpr::getCast(Opcode, 
                                           ConstantInt::getTrue(I.getContext()),
                                            Op0C->getDestTy()))) {
            CI->setPredicate(CI->getInversePredicate());
            return CastInst::Create(Opcode, CI, Op0C->getType());
          }
        }
      }
    }

    if (BinaryOperator *Op0I = dyn_cast<BinaryOperator>(Op0)) {
      // ~(c-X) == X-c-1 == X+(-c-1)
      if (Op0I->getOpcode() == Instruction::Sub && RHS->isAllOnesValue())
        if (Constant *Op0I0C = dyn_cast<Constant>(Op0I->getOperand(0))) {
          Constant *NegOp0I0C = ConstantExpr::getNeg(Op0I0C);
          Constant *ConstantRHS = ConstantExpr::getSub(NegOp0I0C,
                                      ConstantInt::get(I.getType(), 1));
          return BinaryOperator::CreateAdd(Op0I->getOperand(1), ConstantRHS);
        }
          
      if (ConstantInt *Op0CI = dyn_cast<ConstantInt>(Op0I->getOperand(1))) {
        if (Op0I->getOpcode() == Instruction::Add) {
          // ~(X-c) --> (-c-1)-X
          if (RHS->isAllOnesValue()) {
            Constant *NegOp0CI = ConstantExpr::getNeg(Op0CI);
            return BinaryOperator::CreateSub(
                           ConstantExpr::getSub(NegOp0CI,
                                      ConstantInt::get(I.getType(), 1)),
                                      Op0I->getOperand(0));
          } else if (RHS->getValue().isSignBit()) {
            // (X + C) ^ signbit -> (X + C + signbit)
            Constant *C = ConstantInt::get(I.getContext(),
                                           RHS->getValue() + Op0CI->getValue());
            return BinaryOperator::CreateAdd(Op0I->getOperand(0), C);

          }
        } else if (Op0I->getOpcode() == Instruction::Or) {
          // (X|C1)^C2 -> X^(C1|C2) iff X&~C1 == 0
          if (MaskedValueIsZero(Op0I->getOperand(0), Op0CI->getValue())) {
            Constant *NewRHS = ConstantExpr::getOr(Op0CI, RHS);
            // Anything in both C1 and C2 is known to be zero, remove it from
            // NewRHS.
            Constant *CommonBits = ConstantExpr::getAnd(Op0CI, RHS);
            NewRHS = ConstantExpr::getAnd(NewRHS, 
                                       ConstantExpr::getNot(CommonBits));
            Worklist.Add(Op0I);
            I.setOperand(0, Op0I->getOperand(0));
            I.setOperand(1, NewRHS);
            return &I;
          }
        }
      }
    }

    // Try to fold constant and into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;
    if (isa<PHINode>(Op0))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;
  }

  if (Value *X = dyn_castNotVal(Op0))   // ~A ^ A == -1
    if (X == Op1)
      return ReplaceInstUsesWith(I, Constant::getAllOnesValue(I.getType()));

  if (Value *X = dyn_castNotVal(Op1))   // A ^ ~A == -1
    if (X == Op0)
      return ReplaceInstUsesWith(I, Constant::getAllOnesValue(I.getType()));

  
  BinaryOperator *Op1I = dyn_cast<BinaryOperator>(Op1);
  if (Op1I) {
    Value *A, *B;
    if (match(Op1I, m_Or(m_Value(A), m_Value(B)))) {
      if (A == Op0) {              // B^(B|A) == (A|B)^B
        Op1I->swapOperands();
        I.swapOperands();
        std::swap(Op0, Op1);
      } else if (B == Op0) {       // B^(A|B) == (A|B)^B
        I.swapOperands();     // Simplified below.
        std::swap(Op0, Op1);
      }
    } else if (match(Op1I, m_Xor(m_Specific(Op0), m_Value(B)))) {
      return ReplaceInstUsesWith(I, B);                      // A^(A^B) == B
    } else if (match(Op1I, m_Xor(m_Value(A), m_Specific(Op0)))) {
      return ReplaceInstUsesWith(I, A);                      // A^(B^A) == B
    } else if (match(Op1I, m_And(m_Value(A), m_Value(B))) && 
               Op1I->hasOneUse()){
      if (A == Op0) {                                      // A^(A&B) -> A^(B&A)
        Op1I->swapOperands();
        std::swap(A, B);
      }
      if (B == Op0) {                                      // A^(B&A) -> (B&A)^A
        I.swapOperands();     // Simplified below.
        std::swap(Op0, Op1);
      }
    }
  }
  
  BinaryOperator *Op0I = dyn_cast<BinaryOperator>(Op0);
  if (Op0I) {
    Value *A, *B;
    if (match(Op0I, m_Or(m_Value(A), m_Value(B))) &&
        Op0I->hasOneUse()) {
      if (A == Op1)                                  // (B|A)^B == (A|B)^B
        std::swap(A, B);
      if (B == Op1)                                  // (A|B)^B == A & ~B
        return BinaryOperator::CreateAnd(A, Builder->CreateNot(Op1, "tmp"));
    } else if (match(Op0I, m_Xor(m_Specific(Op1), m_Value(B)))) {
      return ReplaceInstUsesWith(I, B);                      // (A^B)^A == B
    } else if (match(Op0I, m_Xor(m_Value(A), m_Specific(Op1)))) {
      return ReplaceInstUsesWith(I, A);                      // (B^A)^A == B
    } else if (match(Op0I, m_And(m_Value(A), m_Value(B))) && 
               Op0I->hasOneUse()){
      if (A == Op1)                                        // (A&B)^A -> (B&A)^A
        std::swap(A, B);
      if (B == Op1 &&                                      // (B&A)^A == ~B & A
          !isa<ConstantInt>(Op1)) {  // Canonical form is (B&C)^C
        return BinaryOperator::CreateAnd(Builder->CreateNot(A, "tmp"), Op1);
      }
    }
  }
  
  // (X >> Z) ^ (Y >> Z)  -> (X^Y) >> Z  for all shifts.
  if (Op0I && Op1I && Op0I->isShift() && 
      Op0I->getOpcode() == Op1I->getOpcode() && 
      Op0I->getOperand(1) == Op1I->getOperand(1) &&
      (Op1I->hasOneUse() || Op1I->hasOneUse())) {
    Value *NewOp =
      Builder->CreateXor(Op0I->getOperand(0), Op1I->getOperand(0),
                         Op0I->getName());
    return BinaryOperator::Create(Op1I->getOpcode(), NewOp, 
                                  Op1I->getOperand(1));
  }
    
  if (Op0I && Op1I) {
    Value *A, *B, *C, *D;
    // (A & B)^(A | B) -> A ^ B
    if (match(Op0I, m_And(m_Value(A), m_Value(B))) &&
        match(Op1I, m_Or(m_Value(C), m_Value(D)))) {
      if ((A == C && B == D) || (A == D && B == C)) 
        return BinaryOperator::CreateXor(A, B);
    }
    // (A | B)^(A & B) -> A ^ B
    if (match(Op0I, m_Or(m_Value(A), m_Value(B))) &&
        match(Op1I, m_And(m_Value(C), m_Value(D)))) {
      if ((A == C && B == D) || (A == D && B == C)) 
        return BinaryOperator::CreateXor(A, B);
    }
    
    // (A & B)^(C & D)
    if ((Op0I->hasOneUse() || Op1I->hasOneUse()) &&
        match(Op0I, m_And(m_Value(A), m_Value(B))) &&
        match(Op1I, m_And(m_Value(C), m_Value(D)))) {
      // (X & Y)^(X & Y) -> (Y^Z) & X
      Value *X = 0, *Y = 0, *Z = 0;
      if (A == C)
        X = A, Y = B, Z = D;
      else if (A == D)
        X = A, Y = B, Z = C;
      else if (B == C)
        X = B, Y = A, Z = D;
      else if (B == D)
        X = B, Y = A, Z = C;
      
      if (X) {
        Value *NewOp = Builder->CreateXor(Y, Z, Op0->getName());
        return BinaryOperator::CreateAnd(NewOp, X);
      }
    }
  }
    
  // (icmp1 A, B) ^ (icmp2 A, B) --> (icmp3 A, B)
  if (ICmpInst *RHS = dyn_cast<ICmpInst>(I.getOperand(1)))
    if (Instruction *R = AssociativeOpt(I, FoldICmpLogical(*this, RHS)))
      return R;

  // fold (xor (cast A), (cast B)) -> (cast (xor A, B))
  if (CastInst *Op0C = dyn_cast<CastInst>(Op0)) {
    if (CastInst *Op1C = dyn_cast<CastInst>(Op1))
      if (Op0C->getOpcode() == Op1C->getOpcode()) { // same cast kind?
        const Type *SrcTy = Op0C->getOperand(0)->getType();
        if (SrcTy == Op1C->getOperand(0)->getType() && SrcTy->isInteger() &&
            // Only do this if the casts both really cause code to be generated.
            ValueRequiresCast(Op0C->getOpcode(), Op0C->getOperand(0), 
                              I.getType()) &&
            ValueRequiresCast(Op1C->getOpcode(), Op1C->getOperand(0), 
                              I.getType())) {
          Value *NewOp = Builder->CreateXor(Op0C->getOperand(0),
                                            Op1C->getOperand(0), I.getName());
          return CastInst::Create(Op0C->getOpcode(), NewOp, I.getType());
        }
      }
  }

  return Changed ? &I : 0;
}


Instruction *InstCombiner::visitShl(BinaryOperator &I) {
  return commonShiftTransforms(I);
}

Instruction *InstCombiner::visitLShr(BinaryOperator &I) {
  return commonShiftTransforms(I);
}

Instruction *InstCombiner::visitAShr(BinaryOperator &I) {
  if (Instruction *R = commonShiftTransforms(I))
    return R;
  
  Value *Op0 = I.getOperand(0);
  
  // ashr int -1, X = -1   (for any arithmetic shift rights of ~0)
  if (ConstantInt *CSI = dyn_cast<ConstantInt>(Op0))
    if (CSI->isAllOnesValue())
      return ReplaceInstUsesWith(I, CSI);

  // See if we can turn a signed shr into an unsigned shr.
  if (MaskedValueIsZero(Op0,
                        APInt::getSignBit(I.getType()->getScalarSizeInBits())))
    return BinaryOperator::CreateLShr(Op0, I.getOperand(1));

  // Arithmetic shifting an all-sign-bit value is a no-op.
  unsigned NumSignBits = ComputeNumSignBits(Op0);
  if (NumSignBits == Op0->getType()->getScalarSizeInBits())
    return ReplaceInstUsesWith(I, Op0);

  return 0;
}

Instruction *InstCombiner::commonShiftTransforms(BinaryOperator &I) {
  assert(I.getOperand(1)->getType() == I.getOperand(0)->getType());
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // shl X, 0 == X and shr X, 0 == X
  // shl 0, X == 0 and shr 0, X == 0
  if (Op1 == Constant::getNullValue(Op1->getType()) ||
      Op0 == Constant::getNullValue(Op0->getType()))
    return ReplaceInstUsesWith(I, Op0);
  
  if (isa<UndefValue>(Op0)) {            
    if (I.getOpcode() == Instruction::AShr) // undef >>s X -> undef
      return ReplaceInstUsesWith(I, Op0);
    else                                    // undef << X -> 0, undef >>u X -> 0
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
  }
  if (isa<UndefValue>(Op1)) {
    if (I.getOpcode() == Instruction::AShr)  // X >>s undef -> X
      return ReplaceInstUsesWith(I, Op0);          
    else                                     // X << undef, X >>u undef -> 0
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
  }

  // See if we can fold away this shift.
  if (SimplifyDemandedInstructionBits(I))
    return &I;

  // Try to fold constant and into select arguments.
  if (isa<Constant>(Op0))
    if (SelectInst *SI = dyn_cast<SelectInst>(Op1))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;

  if (ConstantInt *CUI = dyn_cast<ConstantInt>(Op1))
    if (Instruction *Res = FoldShiftByConstant(Op0, CUI, I))
      return Res;
  return 0;
}

Instruction *InstCombiner::FoldShiftByConstant(Value *Op0, ConstantInt *Op1,
                                               BinaryOperator &I) {
  bool isLeftShift = I.getOpcode() == Instruction::Shl;

  // See if we can simplify any instructions used by the instruction whose sole 
  // purpose is to compute bits we don't care about.
  uint32_t TypeBits = Op0->getType()->getScalarSizeInBits();
  
  // shl i32 X, 32 = 0 and srl i8 Y, 9 = 0, ... just don't eliminate
  // a signed shift.
  //
  if (Op1->uge(TypeBits)) {
    if (I.getOpcode() != Instruction::AShr)
      return ReplaceInstUsesWith(I, Constant::getNullValue(Op0->getType()));
    else {
      I.setOperand(1, ConstantInt::get(I.getType(), TypeBits-1));
      return &I;
    }
  }
  
  // ((X*C1) << C2) == (X * (C1 << C2))
  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(Op0))
    if (BO->getOpcode() == Instruction::Mul && isLeftShift)
      if (Constant *BOOp = dyn_cast<Constant>(BO->getOperand(1)))
        return BinaryOperator::CreateMul(BO->getOperand(0),
                                        ConstantExpr::getShl(BOOp, Op1));
  
  // Try to fold constant and into select arguments.
  if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
    if (Instruction *R = FoldOpIntoSelect(I, SI))
      return R;
  if (isa<PHINode>(Op0))
    if (Instruction *NV = FoldOpIntoPhi(I))
      return NV;
  
  // Fold shift2(trunc(shift1(x,c1)), c2) -> trunc(shift2(shift1(x,c1),c2))
  if (TruncInst *TI = dyn_cast<TruncInst>(Op0)) {
    Instruction *TrOp = dyn_cast<Instruction>(TI->getOperand(0));
    // If 'shift2' is an ashr, we would have to get the sign bit into a funny
    // place.  Don't try to do this transformation in this case.  Also, we
    // require that the input operand is a shift-by-constant so that we have
    // confidence that the shifts will get folded together.  We could do this
    // xform in more cases, but it is unlikely to be profitable.
    if (TrOp && I.isLogicalShift() && TrOp->isShift() && 
        isa<ConstantInt>(TrOp->getOperand(1))) {
      // Okay, we'll do this xform.  Make the shift of shift.
      Constant *ShAmt = ConstantExpr::getZExt(Op1, TrOp->getType());
      // (shift2 (shift1 & 0x00FF), c2)
      Value *NSh = Builder->CreateBinOp(I.getOpcode(), TrOp, ShAmt,I.getName());

      // For logical shifts, the truncation has the effect of making the high
      // part of the register be zeros.  Emulate this by inserting an AND to
      // clear the top bits as needed.  This 'and' will usually be zapped by
      // other xforms later if dead.
      unsigned SrcSize = TrOp->getType()->getScalarSizeInBits();
      unsigned DstSize = TI->getType()->getScalarSizeInBits();
      APInt MaskV(APInt::getLowBitsSet(SrcSize, DstSize));
      
      // The mask we constructed says what the trunc would do if occurring
      // between the shifts.  We want to know the effect *after* the second
      // shift.  We know that it is a logical shift by a constant, so adjust the
      // mask as appropriate.
      if (I.getOpcode() == Instruction::Shl)
        MaskV <<= Op1->getZExtValue();
      else {
        assert(I.getOpcode() == Instruction::LShr && "Unknown logical shift");
        MaskV = MaskV.lshr(Op1->getZExtValue());
      }

      // shift1 & 0x00FF
      Value *And = Builder->CreateAnd(NSh,
                                      ConstantInt::get(I.getContext(), MaskV),
                                      TI->getName());

      // Return the value truncated to the interesting size.
      return new TruncInst(And, I.getType());
    }
  }
  
  if (Op0->hasOneUse()) {
    if (BinaryOperator *Op0BO = dyn_cast<BinaryOperator>(Op0)) {
      // Turn ((X >> C) + Y) << C  ->  (X + (Y << C)) & (~0 << C)
      Value *V1, *V2;
      ConstantInt *CC;
      switch (Op0BO->getOpcode()) {
        default: break;
        case Instruction::Add:
        case Instruction::And:
        case Instruction::Or:
        case Instruction::Xor: {
          // These operators commute.
          // Turn (Y + (X >> C)) << C  ->  (X + (Y << C)) & (~0 << C)
          if (isLeftShift && Op0BO->getOperand(1)->hasOneUse() &&
              match(Op0BO->getOperand(1), m_Shr(m_Value(V1),
                    m_Specific(Op1)))) {
            Value *YS =         // (Y << C)
              Builder->CreateShl(Op0BO->getOperand(0), Op1, Op0BO->getName());
            // (X + (Y << C))
            Value *X = Builder->CreateBinOp(Op0BO->getOpcode(), YS, V1,
                                            Op0BO->getOperand(1)->getName());
            uint32_t Op1Val = Op1->getLimitedValue(TypeBits);
            return BinaryOperator::CreateAnd(X, ConstantInt::get(I.getContext(),
                       APInt::getHighBitsSet(TypeBits, TypeBits-Op1Val)));
          }
          
          // Turn (Y + ((X >> C) & CC)) << C  ->  ((X & (CC << C)) + (Y << C))
          Value *Op0BOOp1 = Op0BO->getOperand(1);
          if (isLeftShift && Op0BOOp1->hasOneUse() &&
              match(Op0BOOp1, 
                    m_And(m_Shr(m_Value(V1), m_Specific(Op1)),
                          m_ConstantInt(CC))) &&
              cast<BinaryOperator>(Op0BOOp1)->getOperand(0)->hasOneUse()) {
            Value *YS =   // (Y << C)
              Builder->CreateShl(Op0BO->getOperand(0), Op1,
                                           Op0BO->getName());
            // X & (CC << C)
            Value *XM = Builder->CreateAnd(V1, ConstantExpr::getShl(CC, Op1),
                                           V1->getName()+".mask");
            return BinaryOperator::Create(Op0BO->getOpcode(), YS, XM);
          }
        }
          
        // FALL THROUGH.
        case Instruction::Sub: {
          // Turn ((X >> C) + Y) << C  ->  (X + (Y << C)) & (~0 << C)
          if (isLeftShift && Op0BO->getOperand(0)->hasOneUse() &&
              match(Op0BO->getOperand(0), m_Shr(m_Value(V1),
                    m_Specific(Op1)))) {
            Value *YS =  // (Y << C)
              Builder->CreateShl(Op0BO->getOperand(1), Op1, Op0BO->getName());
            // (X + (Y << C))
            Value *X = Builder->CreateBinOp(Op0BO->getOpcode(), V1, YS,
                                            Op0BO->getOperand(0)->getName());
            uint32_t Op1Val = Op1->getLimitedValue(TypeBits);
            return BinaryOperator::CreateAnd(X, ConstantInt::get(I.getContext(),
                       APInt::getHighBitsSet(TypeBits, TypeBits-Op1Val)));
          }
          
          // Turn (((X >> C)&CC) + Y) << C  ->  (X + (Y << C)) & (CC << C)
          if (isLeftShift && Op0BO->getOperand(0)->hasOneUse() &&
              match(Op0BO->getOperand(0),
                    m_And(m_Shr(m_Value(V1), m_Value(V2)),
                          m_ConstantInt(CC))) && V2 == Op1 &&
              cast<BinaryOperator>(Op0BO->getOperand(0))
                  ->getOperand(0)->hasOneUse()) {
            Value *YS = // (Y << C)
              Builder->CreateShl(Op0BO->getOperand(1), Op1, Op0BO->getName());
            // X & (CC << C)
            Value *XM = Builder->CreateAnd(V1, ConstantExpr::getShl(CC, Op1),
                                           V1->getName()+".mask");
            
            return BinaryOperator::Create(Op0BO->getOpcode(), XM, YS);
          }
          
          break;
        }
      }
      
      
      // If the operand is an bitwise operator with a constant RHS, and the
      // shift is the only use, we can pull it out of the shift.
      if (ConstantInt *Op0C = dyn_cast<ConstantInt>(Op0BO->getOperand(1))) {
        bool isValid = true;     // Valid only for And, Or, Xor
        bool highBitSet = false; // Transform if high bit of constant set?
        
        switch (Op0BO->getOpcode()) {
          default: isValid = false; break;   // Do not perform transform!
          case Instruction::Add:
            isValid = isLeftShift;
            break;
          case Instruction::Or:
          case Instruction::Xor:
            highBitSet = false;
            break;
          case Instruction::And:
            highBitSet = true;
            break;
        }
        
        // If this is a signed shift right, and the high bit is modified
        // by the logical operation, do not perform the transformation.
        // The highBitSet boolean indicates the value of the high bit of
        // the constant which would cause it to be modified for this
        // operation.
        //
        if (isValid && I.getOpcode() == Instruction::AShr)
          isValid = Op0C->getValue()[TypeBits-1] == highBitSet;
        
        if (isValid) {
          Constant *NewRHS = ConstantExpr::get(I.getOpcode(), Op0C, Op1);
          
          Value *NewShift =
            Builder->CreateBinOp(I.getOpcode(), Op0BO->getOperand(0), Op1);
          NewShift->takeName(Op0BO);
          
          return BinaryOperator::Create(Op0BO->getOpcode(), NewShift,
                                        NewRHS);
        }
      }
    }
  }
  
  // Find out if this is a shift of a shift by a constant.
  BinaryOperator *ShiftOp = dyn_cast<BinaryOperator>(Op0);
  if (ShiftOp && !ShiftOp->isShift())
    ShiftOp = 0;
  
  if (ShiftOp && isa<ConstantInt>(ShiftOp->getOperand(1))) {
    ConstantInt *ShiftAmt1C = cast<ConstantInt>(ShiftOp->getOperand(1));
    uint32_t ShiftAmt1 = ShiftAmt1C->getLimitedValue(TypeBits);
    uint32_t ShiftAmt2 = Op1->getLimitedValue(TypeBits);
    assert(ShiftAmt2 != 0 && "Should have been simplified earlier");
    if (ShiftAmt1 == 0) return 0;  // Will be simplified in the future.
    Value *X = ShiftOp->getOperand(0);
    
    uint32_t AmtSum = ShiftAmt1+ShiftAmt2;   // Fold into one big shift.
    
    const IntegerType *Ty = cast<IntegerType>(I.getType());
    
    // Check for (X << c1) << c2  and  (X >> c1) >> c2
    if (I.getOpcode() == ShiftOp->getOpcode()) {
      // If this is oversized composite shift, then unsigned shifts get 0, ashr
      // saturates.
      if (AmtSum >= TypeBits) {
        if (I.getOpcode() != Instruction::AShr)
          return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
        AmtSum = TypeBits-1;  // Saturate to 31 for i32 ashr.
      }
      
      return BinaryOperator::Create(I.getOpcode(), X,
                                    ConstantInt::get(Ty, AmtSum));
    }
    
    if (ShiftOp->getOpcode() == Instruction::LShr &&
        I.getOpcode() == Instruction::AShr) {
      if (AmtSum >= TypeBits)
        return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
      
      // ((X >>u C1) >>s C2) -> (X >>u (C1+C2))  since C1 != 0.
      return BinaryOperator::CreateLShr(X, ConstantInt::get(Ty, AmtSum));
    }
    
    if (ShiftOp->getOpcode() == Instruction::AShr &&
        I.getOpcode() == Instruction::LShr) {
      // ((X >>s C1) >>u C2) -> ((X >>s (C1+C2)) & mask) since C1 != 0.
      if (AmtSum >= TypeBits)
        AmtSum = TypeBits-1;
      
      Value *Shift = Builder->CreateAShr(X, ConstantInt::get(Ty, AmtSum));

      APInt Mask(APInt::getLowBitsSet(TypeBits, TypeBits - ShiftAmt2));
      return BinaryOperator::CreateAnd(Shift,
                                       ConstantInt::get(I.getContext(), Mask));
    }
    
    // Okay, if we get here, one shift must be left, and the other shift must be
    // right.  See if the amounts are equal.
    if (ShiftAmt1 == ShiftAmt2) {
      // If we have ((X >>? C) << C), turn this into X & (-1 << C).
      if (I.getOpcode() == Instruction::Shl) {
        APInt Mask(APInt::getHighBitsSet(TypeBits, TypeBits - ShiftAmt1));
        return BinaryOperator::CreateAnd(X,
                                         ConstantInt::get(I.getContext(),Mask));
      }
      // If we have ((X << C) >>u C), turn this into X & (-1 >>u C).
      if (I.getOpcode() == Instruction::LShr) {
        APInt Mask(APInt::getLowBitsSet(TypeBits, TypeBits - ShiftAmt1));
        return BinaryOperator::CreateAnd(X,
                                        ConstantInt::get(I.getContext(), Mask));
      }
      // We can simplify ((X << C) >>s C) into a trunc + sext.
      // NOTE: we could do this for any C, but that would make 'unusual' integer
      // types.  For now, just stick to ones well-supported by the code
      // generators.
      const Type *SExtType = 0;
      switch (Ty->getBitWidth() - ShiftAmt1) {
      case 1  :
      case 8  :
      case 16 :
      case 32 :
      case 64 :
      case 128:
        SExtType = IntegerType::get(I.getContext(),
                                    Ty->getBitWidth() - ShiftAmt1);
        break;
      default: break;
      }
      if (SExtType)
        return new SExtInst(Builder->CreateTrunc(X, SExtType, "sext"), Ty);
      // Otherwise, we can't handle it yet.
    } else if (ShiftAmt1 < ShiftAmt2) {
      uint32_t ShiftDiff = ShiftAmt2-ShiftAmt1;
      
      // (X >>? C1) << C2 --> X << (C2-C1) & (-1 << C2)
      if (I.getOpcode() == Instruction::Shl) {
        assert(ShiftOp->getOpcode() == Instruction::LShr ||
               ShiftOp->getOpcode() == Instruction::AShr);
        Value *Shift = Builder->CreateShl(X, ConstantInt::get(Ty, ShiftDiff));
        
        APInt Mask(APInt::getHighBitsSet(TypeBits, TypeBits - ShiftAmt2));
        return BinaryOperator::CreateAnd(Shift,
                                         ConstantInt::get(I.getContext(),Mask));
      }
      
      // (X << C1) >>u C2  --> X >>u (C2-C1) & (-1 >> C2)
      if (I.getOpcode() == Instruction::LShr) {
        assert(ShiftOp->getOpcode() == Instruction::Shl);
        Value *Shift = Builder->CreateLShr(X, ConstantInt::get(Ty, ShiftDiff));
        
        APInt Mask(APInt::getLowBitsSet(TypeBits, TypeBits - ShiftAmt2));
        return BinaryOperator::CreateAnd(Shift,
                                         ConstantInt::get(I.getContext(),Mask));
      }
      
      // We can't handle (X << C1) >>s C2, it shifts arbitrary bits in.
    } else {
      assert(ShiftAmt2 < ShiftAmt1);
      uint32_t ShiftDiff = ShiftAmt1-ShiftAmt2;

      // (X >>? C1) << C2 --> X >>? (C1-C2) & (-1 << C2)
      if (I.getOpcode() == Instruction::Shl) {
        assert(ShiftOp->getOpcode() == Instruction::LShr ||
               ShiftOp->getOpcode() == Instruction::AShr);
        Value *Shift = Builder->CreateBinOp(ShiftOp->getOpcode(), X,
                                            ConstantInt::get(Ty, ShiftDiff));
        
        APInt Mask(APInt::getHighBitsSet(TypeBits, TypeBits - ShiftAmt2));
        return BinaryOperator::CreateAnd(Shift,
                                         ConstantInt::get(I.getContext(),Mask));
      }
      
      // (X << C1) >>u C2  --> X << (C1-C2) & (-1 >> C2)
      if (I.getOpcode() == Instruction::LShr) {
        assert(ShiftOp->getOpcode() == Instruction::Shl);
        Value *Shift = Builder->CreateShl(X, ConstantInt::get(Ty, ShiftDiff));
        
        APInt Mask(APInt::getLowBitsSet(TypeBits, TypeBits - ShiftAmt2));
        return BinaryOperator::CreateAnd(Shift,
                                         ConstantInt::get(I.getContext(),Mask));
      }
      
      // We can't handle (X << C1) >>a C2, it shifts arbitrary bits in.
    }
  }
  return 0;
}



/// FindElementAtOffset - Given a type and a constant offset, determine whether
/// or not there is a sequence of GEP indices into the type that will land us at
/// the specified offset.  If so, fill them into NewIndices and return the
/// resultant element type, otherwise return null.
const Type *InstCombiner::FindElementAtOffset(const Type *Ty, int64_t Offset, 
                                          SmallVectorImpl<Value*> &NewIndices) {
  if (!TD) return 0;
  if (!Ty->isSized()) return 0;
  
  // Start with the index over the outer type.  Note that the type size
  // might be zero (even if the offset isn't zero) if the indexed type
  // is something like [0 x {int, int}]
  const Type *IntPtrTy = TD->getIntPtrType(Ty->getContext());
  int64_t FirstIdx = 0;
  if (int64_t TySize = TD->getTypeAllocSize(Ty)) {
    FirstIdx = Offset/TySize;
    Offset -= FirstIdx*TySize;
    
    // Handle hosts where % returns negative instead of values [0..TySize).
    if (Offset < 0) {
      --FirstIdx;
      Offset += TySize;
      assert(Offset >= 0);
    }
    assert((uint64_t)Offset < (uint64_t)TySize && "Out of range offset");
  }
  
  NewIndices.push_back(ConstantInt::get(IntPtrTy, FirstIdx));
    
  // Index into the types.  If we fail, set OrigBase to null.
  while (Offset) {
    // Indexing into tail padding between struct/array elements.
    if (uint64_t(Offset*8) >= TD->getTypeSizeInBits(Ty))
      return 0;
    
    if (const StructType *STy = dyn_cast<StructType>(Ty)) {
      const StructLayout *SL = TD->getStructLayout(STy);
      assert(Offset < (int64_t)SL->getSizeInBytes() &&
             "Offset must stay within the indexed type");
      
      unsigned Elt = SL->getElementContainingOffset(Offset);
      NewIndices.push_back(ConstantInt::get(Type::getInt32Ty(Ty->getContext()),
                                            Elt));
      
      Offset -= SL->getElementOffset(Elt);
      Ty = STy->getElementType(Elt);
    } else if (const ArrayType *AT = dyn_cast<ArrayType>(Ty)) {
      uint64_t EltSize = TD->getTypeAllocSize(AT->getElementType());
      assert(EltSize && "Cannot index into a zero-sized array");
      NewIndices.push_back(ConstantInt::get(IntPtrTy,Offset/EltSize));
      Offset %= EltSize;
      Ty = AT->getElementType();
    } else {
      // Otherwise, we can't index into the middle of this atomic type, bail.
      return 0;
    }
  }
  
  return Ty;
}


/// GetSelectFoldableOperands - We want to turn code that looks like this:
///   %C = or %A, %B
///   %D = select %cond, %C, %A
/// into:
///   %C = select %cond, %B, 0
///   %D = or %A, %C
///
/// Assuming that the specified instruction is an operand to the select, return
/// a bitmask indicating which operands of this instruction are foldable if they
/// equal the other incoming value of the select.
///
static unsigned GetSelectFoldableOperands(Instruction *I) {
  switch (I->getOpcode()) {
  case Instruction::Add:
  case Instruction::Mul:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    return 3;              // Can fold through either operand.
  case Instruction::Sub:   // Can only fold on the amount subtracted.
  case Instruction::Shl:   // Can only fold on the shift amount.
  case Instruction::LShr:
  case Instruction::AShr:
    return 1;
  default:
    return 0;              // Cannot fold
  }
}

/// GetSelectFoldableConstant - For the same transformation as the previous
/// function, return the identity constant that goes into the select.
static Constant *GetSelectFoldableConstant(Instruction *I) {
  switch (I->getOpcode()) {
  default: llvm_unreachable("This cannot happen!");
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
    return Constant::getNullValue(I->getType());
  case Instruction::And:
    return Constant::getAllOnesValue(I->getType());
  case Instruction::Mul:
    return ConstantInt::get(I->getType(), 1);
  }
}

/// FoldSelectOpOp - Here we have (select c, TI, FI), and we know that TI and FI
/// have the same opcode and only one use each.  Try to simplify this.
Instruction *InstCombiner::FoldSelectOpOp(SelectInst &SI, Instruction *TI,
                                          Instruction *FI) {
  if (TI->getNumOperands() == 1) {
    // If this is a non-volatile load or a cast from the same type,
    // merge.
    if (TI->isCast()) {
      if (TI->getOperand(0)->getType() != FI->getOperand(0)->getType())
        return 0;
    } else {
      return 0;  // unknown unary op.
    }

    // Fold this by inserting a select from the input values.
    SelectInst *NewSI = SelectInst::Create(SI.getCondition(), TI->getOperand(0),
                                          FI->getOperand(0), SI.getName()+".v");
    InsertNewInstBefore(NewSI, SI);
    return CastInst::Create(Instruction::CastOps(TI->getOpcode()), NewSI, 
                            TI->getType());
  }

  // Only handle binary operators here.
  if (!isa<BinaryOperator>(TI))
    return 0;

  // Figure out if the operations have any operands in common.
  Value *MatchOp, *OtherOpT, *OtherOpF;
  bool MatchIsOpZero;
  if (TI->getOperand(0) == FI->getOperand(0)) {
    MatchOp  = TI->getOperand(0);
    OtherOpT = TI->getOperand(1);
    OtherOpF = FI->getOperand(1);
    MatchIsOpZero = true;
  } else if (TI->getOperand(1) == FI->getOperand(1)) {
    MatchOp  = TI->getOperand(1);
    OtherOpT = TI->getOperand(0);
    OtherOpF = FI->getOperand(0);
    MatchIsOpZero = false;
  } else if (!TI->isCommutative()) {
    return 0;
  } else if (TI->getOperand(0) == FI->getOperand(1)) {
    MatchOp  = TI->getOperand(0);
    OtherOpT = TI->getOperand(1);
    OtherOpF = FI->getOperand(0);
    MatchIsOpZero = true;
  } else if (TI->getOperand(1) == FI->getOperand(0)) {
    MatchOp  = TI->getOperand(1);
    OtherOpT = TI->getOperand(0);
    OtherOpF = FI->getOperand(1);
    MatchIsOpZero = true;
  } else {
    return 0;
  }

  // If we reach here, they do have operations in common.
  SelectInst *NewSI = SelectInst::Create(SI.getCondition(), OtherOpT,
                                         OtherOpF, SI.getName()+".v");
  InsertNewInstBefore(NewSI, SI);

  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(TI)) {
    if (MatchIsOpZero)
      return BinaryOperator::Create(BO->getOpcode(), MatchOp, NewSI);
    else
      return BinaryOperator::Create(BO->getOpcode(), NewSI, MatchOp);
  }
  llvm_unreachable("Shouldn't get here");
  return 0;
}

static bool isSelect01(Constant *C1, Constant *C2) {
  ConstantInt *C1I = dyn_cast<ConstantInt>(C1);
  if (!C1I)
    return false;
  ConstantInt *C2I = dyn_cast<ConstantInt>(C2);
  if (!C2I)
    return false;
  return (C1I->isZero() || C1I->isOne()) && (C2I->isZero() || C2I->isOne());
}

/// FoldSelectIntoOp - Try fold the select into one of the operands to
/// facilitate further optimization.
Instruction *InstCombiner::FoldSelectIntoOp(SelectInst &SI, Value *TrueVal,
                                            Value *FalseVal) {
  // See the comment above GetSelectFoldableOperands for a description of the
  // transformation we are doing here.
  if (Instruction *TVI = dyn_cast<Instruction>(TrueVal)) {
    if (TVI->hasOneUse() && TVI->getNumOperands() == 2 &&
        !isa<Constant>(FalseVal)) {
      if (unsigned SFO = GetSelectFoldableOperands(TVI)) {
        unsigned OpToFold = 0;
        if ((SFO & 1) && FalseVal == TVI->getOperand(0)) {
          OpToFold = 1;
        } else  if ((SFO & 2) && FalseVal == TVI->getOperand(1)) {
          OpToFold = 2;
        }

        if (OpToFold) {
          Constant *C = GetSelectFoldableConstant(TVI);
          Value *OOp = TVI->getOperand(2-OpToFold);
          // Avoid creating select between 2 constants unless it's selecting
          // between 0 and 1.
          if (!isa<Constant>(OOp) || isSelect01(C, cast<Constant>(OOp))) {
            Instruction *NewSel = SelectInst::Create(SI.getCondition(), OOp, C);
            InsertNewInstBefore(NewSel, SI);
            NewSel->takeName(TVI);
            if (BinaryOperator *BO = dyn_cast<BinaryOperator>(TVI))
              return BinaryOperator::Create(BO->getOpcode(), FalseVal, NewSel);
            llvm_unreachable("Unknown instruction!!");
          }
        }
      }
    }
  }

  if (Instruction *FVI = dyn_cast<Instruction>(FalseVal)) {
    if (FVI->hasOneUse() && FVI->getNumOperands() == 2 &&
        !isa<Constant>(TrueVal)) {
      if (unsigned SFO = GetSelectFoldableOperands(FVI)) {
        unsigned OpToFold = 0;
        if ((SFO & 1) && TrueVal == FVI->getOperand(0)) {
          OpToFold = 1;
        } else  if ((SFO & 2) && TrueVal == FVI->getOperand(1)) {
          OpToFold = 2;
        }

        if (OpToFold) {
          Constant *C = GetSelectFoldableConstant(FVI);
          Value *OOp = FVI->getOperand(2-OpToFold);
          // Avoid creating select between 2 constants unless it's selecting
          // between 0 and 1.
          if (!isa<Constant>(OOp) || isSelect01(C, cast<Constant>(OOp))) {
            Instruction *NewSel = SelectInst::Create(SI.getCondition(), C, OOp);
            InsertNewInstBefore(NewSel, SI);
            NewSel->takeName(FVI);
            if (BinaryOperator *BO = dyn_cast<BinaryOperator>(FVI))
              return BinaryOperator::Create(BO->getOpcode(), TrueVal, NewSel);
            llvm_unreachable("Unknown instruction!!");
          }
        }
      }
    }
  }

  return 0;
}

/// visitSelectInstWithICmp - Visit a SelectInst that has an
/// ICmpInst as its first operand.
///
Instruction *InstCombiner::visitSelectInstWithICmp(SelectInst &SI,
                                                   ICmpInst *ICI) {
  bool Changed = false;
  ICmpInst::Predicate Pred = ICI->getPredicate();
  Value *CmpLHS = ICI->getOperand(0);
  Value *CmpRHS = ICI->getOperand(1);
  Value *TrueVal = SI.getTrueValue();
  Value *FalseVal = SI.getFalseValue();

  // Check cases where the comparison is with a constant that
  // can be adjusted to fit the min/max idiom. We may edit ICI in
  // place here, so make sure the select is the only user.
  if (ICI->hasOneUse())
    if (ConstantInt *CI = dyn_cast<ConstantInt>(CmpRHS)) {
      switch (Pred) {
      default: break;
      case ICmpInst::ICMP_ULT:
      case ICmpInst::ICMP_SLT: {
        // X < MIN ? T : F  -->  F
        if (CI->isMinValue(Pred == ICmpInst::ICMP_SLT))
          return ReplaceInstUsesWith(SI, FalseVal);
        // X < C ? X : C-1  -->  X > C-1 ? C-1 : X
        Constant *AdjustedRHS = SubOne(CI);
        if ((CmpLHS == TrueVal && AdjustedRHS == FalseVal) ||
            (CmpLHS == FalseVal && AdjustedRHS == TrueVal)) {
          Pred = ICmpInst::getSwappedPredicate(Pred);
          CmpRHS = AdjustedRHS;
          std::swap(FalseVal, TrueVal);
          ICI->setPredicate(Pred);
          ICI->setOperand(1, CmpRHS);
          SI.setOperand(1, TrueVal);
          SI.setOperand(2, FalseVal);
          Changed = true;
        }
        break;
      }
      case ICmpInst::ICMP_UGT:
      case ICmpInst::ICMP_SGT: {
        // X > MAX ? T : F  -->  F
        if (CI->isMaxValue(Pred == ICmpInst::ICMP_SGT))
          return ReplaceInstUsesWith(SI, FalseVal);
        // X > C ? X : C+1  -->  X < C+1 ? C+1 : X
        Constant *AdjustedRHS = AddOne(CI);
        if ((CmpLHS == TrueVal && AdjustedRHS == FalseVal) ||
            (CmpLHS == FalseVal && AdjustedRHS == TrueVal)) {
          Pred = ICmpInst::getSwappedPredicate(Pred);
          CmpRHS = AdjustedRHS;
          std::swap(FalseVal, TrueVal);
          ICI->setPredicate(Pred);
          ICI->setOperand(1, CmpRHS);
          SI.setOperand(1, TrueVal);
          SI.setOperand(2, FalseVal);
          Changed = true;
        }
        break;
      }
      }

      // (x <s 0) ? -1 : 0 -> ashr x, 31   -> all ones if signed
      // (x >s -1) ? -1 : 0 -> ashr x, 31  -> all ones if not signed
      CmpInst::Predicate Pred = CmpInst::BAD_ICMP_PREDICATE;
      if (match(TrueVal, m_ConstantInt<-1>()) &&
          match(FalseVal, m_ConstantInt<0>()))
        Pred = ICI->getPredicate();
      else if (match(TrueVal, m_ConstantInt<0>()) &&
               match(FalseVal, m_ConstantInt<-1>()))
        Pred = CmpInst::getInversePredicate(ICI->getPredicate());
      
      if (Pred != CmpInst::BAD_ICMP_PREDICATE) {
        // If we are just checking for a icmp eq of a single bit and zext'ing it
        // to an integer, then shift the bit to the appropriate place and then
        // cast to integer to avoid the comparison.
        const APInt &Op1CV = CI->getValue();
    
        // sext (x <s  0) to i32 --> x>>s31      true if signbit set.
        // sext (x >s -1) to i32 --> (x>>s31)^-1  true if signbit clear.
        if ((Pred == ICmpInst::ICMP_SLT && Op1CV == 0) ||
            (Pred == ICmpInst::ICMP_SGT && Op1CV.isAllOnesValue())) {
          Value *In = ICI->getOperand(0);
          Value *Sh = ConstantInt::get(In->getType(),
                                       In->getType()->getScalarSizeInBits()-1);
          In = InsertNewInstBefore(BinaryOperator::CreateAShr(In, Sh,
                                                        In->getName()+".lobit"),
                                   *ICI);
          if (In->getType() != SI.getType())
            In = CastInst::CreateIntegerCast(In, SI.getType(),
                                             true/*SExt*/, "tmp", ICI);
    
          if (Pred == ICmpInst::ICMP_SGT)
            In = InsertNewInstBefore(BinaryOperator::CreateNot(In,
                                       In->getName()+".not"), *ICI);
    
          return ReplaceInstUsesWith(SI, In);
        }
      }
    }

  if (CmpLHS == TrueVal && CmpRHS == FalseVal) {
    // Transform (X == Y) ? X : Y  -> Y
    if (Pred == ICmpInst::ICMP_EQ)
      return ReplaceInstUsesWith(SI, FalseVal);
    // Transform (X != Y) ? X : Y  -> X
    if (Pred == ICmpInst::ICMP_NE)
      return ReplaceInstUsesWith(SI, TrueVal);
    /// NOTE: if we wanted to, this is where to detect integer MIN/MAX

  } else if (CmpLHS == FalseVal && CmpRHS == TrueVal) {
    // Transform (X == Y) ? Y : X  -> X
    if (Pred == ICmpInst::ICMP_EQ)
      return ReplaceInstUsesWith(SI, FalseVal);
    // Transform (X != Y) ? Y : X  -> Y
    if (Pred == ICmpInst::ICMP_NE)
      return ReplaceInstUsesWith(SI, TrueVal);
    /// NOTE: if we wanted to, this is where to detect integer MIN/MAX
  }
  return Changed ? &SI : 0;
}


/// CanSelectOperandBeMappingIntoPredBlock - SI is a select whose condition is a
/// PHI node (but the two may be in different blocks).  See if the true/false
/// values (V) are live in all of the predecessor blocks of the PHI.  For
/// example, cases like this cannot be mapped:
///
///   X = phi [ C1, BB1], [C2, BB2]
///   Y = add
///   Z = select X, Y, 0
///
/// because Y is not live in BB1/BB2.
///
static bool CanSelectOperandBeMappingIntoPredBlock(const Value *V,
                                                   const SelectInst &SI) {
  // If the value is a non-instruction value like a constant or argument, it
  // can always be mapped.
  const Instruction *I = dyn_cast<Instruction>(V);
  if (I == 0) return true;
  
  // If V is a PHI node defined in the same block as the condition PHI, we can
  // map the arguments.
  const PHINode *CondPHI = cast<PHINode>(SI.getCondition());
  
  if (const PHINode *VP = dyn_cast<PHINode>(I))
    if (VP->getParent() == CondPHI->getParent())
      return true;
  
  // Otherwise, if the PHI and select are defined in the same block and if V is
  // defined in a different block, then we can transform it.
  if (SI.getParent() == CondPHI->getParent() &&
      I->getParent() != CondPHI->getParent())
    return true;
  
  // Otherwise we have a 'hard' case and we can't tell without doing more
  // detailed dominator based analysis, punt.
  return false;
}

/// FoldSPFofSPF - We have an SPF (e.g. a min or max) of an SPF of the form:
///   SPF2(SPF1(A, B), C) 
Instruction *InstCombiner::FoldSPFofSPF(Instruction *Inner,
                                        SelectPatternFlavor SPF1,
                                        Value *A, Value *B,
                                        Instruction &Outer,
                                        SelectPatternFlavor SPF2, Value *C) {
  if (C == A || C == B) {
    // MAX(MAX(A, B), B) -> MAX(A, B)
    // MIN(MIN(a, b), a) -> MIN(a, b)
    if (SPF1 == SPF2)
      return ReplaceInstUsesWith(Outer, Inner);
    
    // MAX(MIN(a, b), a) -> a
    // MIN(MAX(a, b), a) -> a
    if ((SPF1 == SPF_SMIN && SPF2 == SPF_SMAX) ||
        (SPF1 == SPF_SMAX && SPF2 == SPF_SMIN) ||
        (SPF1 == SPF_UMIN && SPF2 == SPF_UMAX) ||
        (SPF1 == SPF_UMAX && SPF2 == SPF_UMIN))
      return ReplaceInstUsesWith(Outer, C);
  }
  
  // TODO: MIN(MIN(A, 23), 97)
  return 0;
}




Instruction *InstCombiner::visitSelectInst(SelectInst &SI) {
  Value *CondVal = SI.getCondition();
  Value *TrueVal = SI.getTrueValue();
  Value *FalseVal = SI.getFalseValue();

  // select true, X, Y  -> X
  // select false, X, Y -> Y
  if (ConstantInt *C = dyn_cast<ConstantInt>(CondVal))
    return ReplaceInstUsesWith(SI, C->getZExtValue() ? TrueVal : FalseVal);

  // select C, X, X -> X
  if (TrueVal == FalseVal)
    return ReplaceInstUsesWith(SI, TrueVal);

  if (isa<UndefValue>(TrueVal))   // select C, undef, X -> X
    return ReplaceInstUsesWith(SI, FalseVal);
  if (isa<UndefValue>(FalseVal))   // select C, X, undef -> X
    return ReplaceInstUsesWith(SI, TrueVal);
  if (isa<UndefValue>(CondVal)) {  // select undef, X, Y -> X or Y
    if (isa<Constant>(TrueVal))
      return ReplaceInstUsesWith(SI, TrueVal);
    else
      return ReplaceInstUsesWith(SI, FalseVal);
  }

  if (SI.getType() == Type::getInt1Ty(SI.getContext())) {
    if (ConstantInt *C = dyn_cast<ConstantInt>(TrueVal)) {
      if (C->getZExtValue()) {
        // Change: A = select B, true, C --> A = or B, C
        return BinaryOperator::CreateOr(CondVal, FalseVal);
      } else {
        // Change: A = select B, false, C --> A = and !B, C
        Value *NotCond =
          InsertNewInstBefore(BinaryOperator::CreateNot(CondVal,
                                             "not."+CondVal->getName()), SI);
        return BinaryOperator::CreateAnd(NotCond, FalseVal);
      }
    } else if (ConstantInt *C = dyn_cast<ConstantInt>(FalseVal)) {
      if (C->getZExtValue() == false) {
        // Change: A = select B, C, false --> A = and B, C
        return BinaryOperator::CreateAnd(CondVal, TrueVal);
      } else {
        // Change: A = select B, C, true --> A = or !B, C
        Value *NotCond =
          InsertNewInstBefore(BinaryOperator::CreateNot(CondVal,
                                             "not."+CondVal->getName()), SI);
        return BinaryOperator::CreateOr(NotCond, TrueVal);
      }
    }
    
    // select a, b, a  -> a&b
    // select a, a, b  -> a|b
    if (CondVal == TrueVal)
      return BinaryOperator::CreateOr(CondVal, FalseVal);
    else if (CondVal == FalseVal)
      return BinaryOperator::CreateAnd(CondVal, TrueVal);
  }

  // Selecting between two integer constants?
  if (ConstantInt *TrueValC = dyn_cast<ConstantInt>(TrueVal))
    if (ConstantInt *FalseValC = dyn_cast<ConstantInt>(FalseVal)) {
      // select C, 1, 0 -> zext C to int
      if (FalseValC->isZero() && TrueValC->getValue() == 1) {
        return CastInst::Create(Instruction::ZExt, CondVal, SI.getType());
      } else if (TrueValC->isZero() && FalseValC->getValue() == 1) {
        // select C, 0, 1 -> zext !C to int
        Value *NotCond =
          InsertNewInstBefore(BinaryOperator::CreateNot(CondVal,
                                               "not."+CondVal->getName()), SI);
        return CastInst::Create(Instruction::ZExt, NotCond, SI.getType());
      }

      if (ICmpInst *IC = dyn_cast<ICmpInst>(SI.getCondition())) {
        // If one of the constants is zero (we know they can't both be) and we
        // have an icmp instruction with zero, and we have an 'and' with the
        // non-constant value, eliminate this whole mess.  This corresponds to
        // cases like this: ((X & 27) ? 27 : 0)
        if (TrueValC->isZero() || FalseValC->isZero())
          if (IC->isEquality() && isa<ConstantInt>(IC->getOperand(1)) &&
              cast<Constant>(IC->getOperand(1))->isNullValue())
            if (Instruction *ICA = dyn_cast<Instruction>(IC->getOperand(0)))
              if (ICA->getOpcode() == Instruction::And &&
                  isa<ConstantInt>(ICA->getOperand(1)) &&
                  (ICA->getOperand(1) == TrueValC ||
                   ICA->getOperand(1) == FalseValC) &&
                  isOneBitSet(cast<ConstantInt>(ICA->getOperand(1)))) {
                // Okay, now we know that everything is set up, we just don't
                // know whether we have a icmp_ne or icmp_eq and whether the 
                // true or false val is the zero.
                bool ShouldNotVal = !TrueValC->isZero();
                ShouldNotVal ^= IC->getPredicate() == ICmpInst::ICMP_NE;
                Value *V = ICA;
                if (ShouldNotVal)
                  V = InsertNewInstBefore(BinaryOperator::Create(
                                  Instruction::Xor, V, ICA->getOperand(1)), SI);
                return ReplaceInstUsesWith(SI, V);
              }
      }
    }

  // See if we are selecting two values based on a comparison of the two values.
  if (FCmpInst *FCI = dyn_cast<FCmpInst>(CondVal)) {
    if (FCI->getOperand(0) == TrueVal && FCI->getOperand(1) == FalseVal) {
      // Transform (X == Y) ? X : Y  -> Y
      if (FCI->getPredicate() == FCmpInst::FCMP_OEQ) {
        // This is not safe in general for floating point:  
        // consider X== -0, Y== +0.
        // It becomes safe if either operand is a nonzero constant.
        ConstantFP *CFPt, *CFPf;
        if (((CFPt = dyn_cast<ConstantFP>(TrueVal)) &&
              !CFPt->getValueAPF().isZero()) ||
            ((CFPf = dyn_cast<ConstantFP>(FalseVal)) &&
             !CFPf->getValueAPF().isZero()))
        return ReplaceInstUsesWith(SI, FalseVal);
      }
      // Transform (X != Y) ? X : Y  -> X
      if (FCI->getPredicate() == FCmpInst::FCMP_ONE)
        return ReplaceInstUsesWith(SI, TrueVal);
      // NOTE: if we wanted to, this is where to detect MIN/MAX

    } else if (FCI->getOperand(0) == FalseVal && FCI->getOperand(1) == TrueVal){
      // Transform (X == Y) ? Y : X  -> X
      if (FCI->getPredicate() == FCmpInst::FCMP_OEQ) {
        // This is not safe in general for floating point:  
        // consider X== -0, Y== +0.
        // It becomes safe if either operand is a nonzero constant.
        ConstantFP *CFPt, *CFPf;
        if (((CFPt = dyn_cast<ConstantFP>(TrueVal)) &&
              !CFPt->getValueAPF().isZero()) ||
            ((CFPf = dyn_cast<ConstantFP>(FalseVal)) &&
             !CFPf->getValueAPF().isZero()))
          return ReplaceInstUsesWith(SI, FalseVal);
      }
      // Transform (X != Y) ? Y : X  -> Y
      if (FCI->getPredicate() == FCmpInst::FCMP_ONE)
        return ReplaceInstUsesWith(SI, TrueVal);
      // NOTE: if we wanted to, this is where to detect MIN/MAX
    }
    // NOTE: if we wanted to, this is where to detect ABS
  }

  // See if we are selecting two values based on a comparison of the two values.
  if (ICmpInst *ICI = dyn_cast<ICmpInst>(CondVal))
    if (Instruction *Result = visitSelectInstWithICmp(SI, ICI))
      return Result;

  if (Instruction *TI = dyn_cast<Instruction>(TrueVal))
    if (Instruction *FI = dyn_cast<Instruction>(FalseVal))
      if (TI->hasOneUse() && FI->hasOneUse()) {
        Instruction *AddOp = 0, *SubOp = 0;

        // Turn (select C, (op X, Y), (op X, Z)) -> (op X, (select C, Y, Z))
        if (TI->getOpcode() == FI->getOpcode())
          if (Instruction *IV = FoldSelectOpOp(SI, TI, FI))
            return IV;

        // Turn select C, (X+Y), (X-Y) --> (X+(select C, Y, (-Y))).  This is
        // even legal for FP.
        if ((TI->getOpcode() == Instruction::Sub &&
             FI->getOpcode() == Instruction::Add) ||
            (TI->getOpcode() == Instruction::FSub &&
             FI->getOpcode() == Instruction::FAdd)) {
          AddOp = FI; SubOp = TI;
        } else if ((FI->getOpcode() == Instruction::Sub &&
                    TI->getOpcode() == Instruction::Add) ||
                   (FI->getOpcode() == Instruction::FSub &&
                    TI->getOpcode() == Instruction::FAdd)) {
          AddOp = TI; SubOp = FI;
        }

        if (AddOp) {
          Value *OtherAddOp = 0;
          if (SubOp->getOperand(0) == AddOp->getOperand(0)) {
            OtherAddOp = AddOp->getOperand(1);
          } else if (SubOp->getOperand(0) == AddOp->getOperand(1)) {
            OtherAddOp = AddOp->getOperand(0);
          }

          if (OtherAddOp) {
            // So at this point we know we have (Y -> OtherAddOp):
            //        select C, (add X, Y), (sub X, Z)
            Value *NegVal;  // Compute -Z
            if (Constant *C = dyn_cast<Constant>(SubOp->getOperand(1))) {
              NegVal = ConstantExpr::getNeg(C);
            } else {
              NegVal = InsertNewInstBefore(
                    BinaryOperator::CreateNeg(SubOp->getOperand(1),
                                              "tmp"), SI);
            }

            Value *NewTrueOp = OtherAddOp;
            Value *NewFalseOp = NegVal;
            if (AddOp != TI)
              std::swap(NewTrueOp, NewFalseOp);
            Instruction *NewSel =
              SelectInst::Create(CondVal, NewTrueOp,
                                 NewFalseOp, SI.getName() + ".p");

            NewSel = InsertNewInstBefore(NewSel, SI);
            return BinaryOperator::CreateAdd(SubOp->getOperand(0), NewSel);
          }
        }
      }

  // See if we can fold the select into one of our operands.
  if (SI.getType()->isInteger()) {
    if (Instruction *FoldI = FoldSelectIntoOp(SI, TrueVal, FalseVal))
      return FoldI;
    
    // MAX(MAX(a, b), a) -> MAX(a, b)
    // MIN(MIN(a, b), a) -> MIN(a, b)
    // MAX(MIN(a, b), a) -> a
    // MIN(MAX(a, b), a) -> a
    Value *LHS, *RHS, *LHS2, *RHS2;
    if (SelectPatternFlavor SPF = MatchSelectPattern(&SI, LHS, RHS)) {
      if (SelectPatternFlavor SPF2 = MatchSelectPattern(LHS, LHS2, RHS2))
        if (Instruction *R = FoldSPFofSPF(cast<Instruction>(LHS),SPF2,LHS2,RHS2, 
                                          SI, SPF, RHS))
          return R;
      if (SelectPatternFlavor SPF2 = MatchSelectPattern(RHS, LHS2, RHS2))
        if (Instruction *R = FoldSPFofSPF(cast<Instruction>(RHS),SPF2,LHS2,RHS2,
                                          SI, SPF, LHS))
          return R;
    }

    // TODO.
    // ABS(-X) -> ABS(X)
    // ABS(ABS(X)) -> ABS(X)
  }

  // See if we can fold the select into a phi node if the condition is a select.
  if (isa<PHINode>(SI.getCondition())) 
    // The true/false values have to be live in the PHI predecessor's blocks.
    if (CanSelectOperandBeMappingIntoPredBlock(TrueVal, SI) &&
        CanSelectOperandBeMappingIntoPredBlock(FalseVal, SI))
      if (Instruction *NV = FoldOpIntoPhi(SI))
        return NV;

  if (BinaryOperator::isNot(CondVal)) {
    SI.setOperand(0, BinaryOperator::getNotArgument(CondVal));
    SI.setOperand(1, FalseVal);
    SI.setOperand(2, TrueVal);
    return &SI;
  }

  return 0;
}

/// EnforceKnownAlignment - If the specified pointer points to an object that
/// we control, modify the object's alignment to PrefAlign. This isn't
/// often possible though. If alignment is important, a more reliable approach
/// is to simply align all global variables and allocation instructions to
/// their preferred alignment from the beginning.
///
static unsigned EnforceKnownAlignment(Value *V,
                                      unsigned Align, unsigned PrefAlign) {

  User *U = dyn_cast<User>(V);
  if (!U) return Align;

  switch (Operator::getOpcode(U)) {
  default: break;
  case Instruction::BitCast:
    return EnforceKnownAlignment(U->getOperand(0), Align, PrefAlign);
  case Instruction::GetElementPtr: {
    // If all indexes are zero, it is just the alignment of the base pointer.
    bool AllZeroOperands = true;
    for (User::op_iterator i = U->op_begin() + 1, e = U->op_end(); i != e; ++i)
      if (!isa<Constant>(*i) ||
          !cast<Constant>(*i)->isNullValue()) {
        AllZeroOperands = false;
        break;
      }

    if (AllZeroOperands) {
      // Treat this like a bitcast.
      return EnforceKnownAlignment(U->getOperand(0), Align, PrefAlign);
    }
    break;
  }
  }

  if (GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
    // If there is a large requested alignment and we can, bump up the alignment
    // of the global.
    if (!GV->isDeclaration()) {
      if (GV->getAlignment() >= PrefAlign)
        Align = GV->getAlignment();
      else {
        GV->setAlignment(PrefAlign);
        Align = PrefAlign;
      }
    }
  } else if (AllocaInst *AI = dyn_cast<AllocaInst>(V)) {
    // If there is a requested alignment and if this is an alloca, round up.
    if (AI->getAlignment() >= PrefAlign)
      Align = AI->getAlignment();
    else {
      AI->setAlignment(PrefAlign);
      Align = PrefAlign;
    }
  }

  return Align;
}

/// GetOrEnforceKnownAlignment - If the specified pointer has an alignment that
/// we can determine, return it, otherwise return 0.  If PrefAlign is specified,
/// and it is more than the alignment of the ultimate object, see if we can
/// increase the alignment of the ultimate object, making this check succeed.
unsigned InstCombiner::GetOrEnforceKnownAlignment(Value *V,
                                                  unsigned PrefAlign) {
  unsigned BitWidth = TD ? TD->getTypeSizeInBits(V->getType()) :
                      sizeof(PrefAlign) * CHAR_BIT;
  APInt Mask = APInt::getAllOnesValue(BitWidth);
  APInt KnownZero(BitWidth, 0), KnownOne(BitWidth, 0);
  ComputeMaskedBits(V, Mask, KnownZero, KnownOne);
  unsigned TrailZ = KnownZero.countTrailingOnes();
  unsigned Align = 1u << std::min(BitWidth - 1, TrailZ);

  if (PrefAlign > Align)
    Align = EnforceKnownAlignment(V, Align, PrefAlign);
  
    // We don't need to make any adjustment.
  return Align;
}

Instruction *InstCombiner::SimplifyMemTransfer(MemIntrinsic *MI) {
  unsigned DstAlign = GetOrEnforceKnownAlignment(MI->getOperand(1));
  unsigned SrcAlign = GetOrEnforceKnownAlignment(MI->getOperand(2));
  unsigned MinAlign = std::min(DstAlign, SrcAlign);
  unsigned CopyAlign = MI->getAlignment();

  if (CopyAlign < MinAlign) {
    MI->setAlignment(ConstantInt::get(MI->getAlignmentType(), 
                                             MinAlign, false));
    return MI;
  }
  
  // If MemCpyInst length is 1/2/4/8 bytes then replace memcpy with
  // load/store.
  ConstantInt *MemOpLength = dyn_cast<ConstantInt>(MI->getOperand(3));
  if (MemOpLength == 0) return 0;
  
  // Source and destination pointer types are always "i8*" for intrinsic.  See
  // if the size is something we can handle with a single primitive load/store.
  // A single load+store correctly handles overlapping memory in the memmove
  // case.
  unsigned Size = MemOpLength->getZExtValue();
  if (Size == 0) return MI;  // Delete this mem transfer.
  
  if (Size > 8 || (Size&(Size-1)))
    return 0;  // If not 1/2/4/8 bytes, exit.
  
  // Use an integer load+store unless we can find something better.
  Type *NewPtrTy =
            PointerType::getUnqual(IntegerType::get(MI->getContext(), Size<<3));
  
  // Memcpy forces the use of i8* for the source and destination.  That means
  // that if you're using memcpy to move one double around, you'll get a cast
  // from double* to i8*.  We'd much rather use a double load+store rather than
  // an i64 load+store, here because this improves the odds that the source or
  // dest address will be promotable.  See if we can find a better type than the
  // integer datatype.
  if (Value *Op = getBitCastOperand(MI->getOperand(1))) {
    const Type *SrcETy = cast<PointerType>(Op->getType())->getElementType();
    if (TD && SrcETy->isSized() && TD->getTypeStoreSize(SrcETy) == Size) {
      // The SrcETy might be something like {{{double}}} or [1 x double].  Rip
      // down through these levels if so.
      while (!SrcETy->isSingleValueType()) {
        if (const StructType *STy = dyn_cast<StructType>(SrcETy)) {
          if (STy->getNumElements() == 1)
            SrcETy = STy->getElementType(0);
          else
            break;
        } else if (const ArrayType *ATy = dyn_cast<ArrayType>(SrcETy)) {
          if (ATy->getNumElements() == 1)
            SrcETy = ATy->getElementType();
          else
            break;
        } else
          break;
      }
      
      if (SrcETy->isSingleValueType())
        NewPtrTy = PointerType::getUnqual(SrcETy);
    }
  }
  
  
  // If the memcpy/memmove provides better alignment info than we can
  // infer, use it.
  SrcAlign = std::max(SrcAlign, CopyAlign);
  DstAlign = std::max(DstAlign, CopyAlign);
  
  Value *Src = Builder->CreateBitCast(MI->getOperand(2), NewPtrTy);
  Value *Dest = Builder->CreateBitCast(MI->getOperand(1), NewPtrTy);
  Instruction *L = new LoadInst(Src, "tmp", false, SrcAlign);
  InsertNewInstBefore(L, *MI);
  InsertNewInstBefore(new StoreInst(L, Dest, false, DstAlign), *MI);

  // Set the size of the copy to 0, it will be deleted on the next iteration.
  MI->setOperand(3, Constant::getNullValue(MemOpLength->getType()));
  return MI;
}

Instruction *InstCombiner::SimplifyMemSet(MemSetInst *MI) {
  unsigned Alignment = GetOrEnforceKnownAlignment(MI->getDest());
  if (MI->getAlignment() < Alignment) {
    MI->setAlignment(ConstantInt::get(MI->getAlignmentType(),
                                             Alignment, false));
    return MI;
  }
  
  // Extract the length and alignment and fill if they are constant.
  ConstantInt *LenC = dyn_cast<ConstantInt>(MI->getLength());
  ConstantInt *FillC = dyn_cast<ConstantInt>(MI->getValue());
  if (!LenC || !FillC || FillC->getType() != Type::getInt8Ty(MI->getContext()))
    return 0;
  uint64_t Len = LenC->getZExtValue();
  Alignment = MI->getAlignment();
  
  // If the length is zero, this is a no-op
  if (Len == 0) return MI; // memset(d,c,0,a) -> noop
  
  // memset(s,c,n) -> store s, c (for n=1,2,4,8)
  if (Len <= 8 && isPowerOf2_32((uint32_t)Len)) {
    const Type *ITy = IntegerType::get(MI->getContext(), Len*8);  // n=1 -> i8.
    
    Value *Dest = MI->getDest();
    Dest = Builder->CreateBitCast(Dest, PointerType::getUnqual(ITy));

    // Alignment 0 is identity for alignment 1 for memset, but not store.
    if (Alignment == 0) Alignment = 1;
    
    // Extract the fill value and store.
    uint64_t Fill = FillC->getZExtValue()*0x0101010101010101ULL;
    InsertNewInstBefore(new StoreInst(ConstantInt::get(ITy, Fill),
                                      Dest, false, Alignment), *MI);
    
    // Set the size of the copy to 0, it will be deleted on the next iteration.
    MI->setLength(Constant::getNullValue(LenC->getType()));
    return MI;
  }

  return 0;
}


/// visitCallInst - CallInst simplification.  This mostly only handles folding 
/// of intrinsic instructions.  For normal calls, it allows visitCallSite to do
/// the heavy lifting.
///
Instruction *InstCombiner::visitCallInst(CallInst &CI) {
  if (isFreeCall(&CI))
    return visitFree(CI);

  // If the caller function is nounwind, mark the call as nounwind, even if the
  // callee isn't.
  if (CI.getParent()->getParent()->doesNotThrow() &&
      !CI.doesNotThrow()) {
    CI.setDoesNotThrow();
    return &CI;
  }
  
  IntrinsicInst *II = dyn_cast<IntrinsicInst>(&CI);
  if (!II) return visitCallSite(&CI);
  
  // Intrinsics cannot occur in an invoke, so handle them here instead of in
  // visitCallSite.
  if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(II)) {
    bool Changed = false;

    // memmove/cpy/set of zero bytes is a noop.
    if (Constant *NumBytes = dyn_cast<Constant>(MI->getLength())) {
      if (NumBytes->isNullValue()) return EraseInstFromFunction(CI);

      if (ConstantInt *CI = dyn_cast<ConstantInt>(NumBytes))
        if (CI->getZExtValue() == 1) {
          // Replace the instruction with just byte operations.  We would
          // transform other cases to loads/stores, but we don't know if
          // alignment is sufficient.
        }
    }

    // If we have a memmove and the source operation is a constant global,
    // then the source and dest pointers can't alias, so we can change this
    // into a call to memcpy.
    if (MemMoveInst *MMI = dyn_cast<MemMoveInst>(MI)) {
      if (GlobalVariable *GVSrc = dyn_cast<GlobalVariable>(MMI->getSource()))
        if (GVSrc->isConstant()) {
          Module *M = CI.getParent()->getParent()->getParent();
          Intrinsic::ID MemCpyID = Intrinsic::memcpy;
          const Type *Tys[1];
          Tys[0] = CI.getOperand(3)->getType();
          CI.setOperand(0, 
                        Intrinsic::getDeclaration(M, MemCpyID, Tys, 1));
          Changed = true;
        }
    }

    if (MemTransferInst *MTI = dyn_cast<MemTransferInst>(MI)) {
      // memmove(x,x,size) -> noop.
      if (MTI->getSource() == MTI->getDest())
        return EraseInstFromFunction(CI);
    }

    // If we can determine a pointer alignment that is bigger than currently
    // set, update the alignment.
    if (isa<MemTransferInst>(MI)) {
      if (Instruction *I = SimplifyMemTransfer(MI))
        return I;
    } else if (MemSetInst *MSI = dyn_cast<MemSetInst>(MI)) {
      if (Instruction *I = SimplifyMemSet(MSI))
        return I;
    }
          
    if (Changed) return II;
  }
  
  switch (II->getIntrinsicID()) {
  default: break;
  case Intrinsic::bswap:
    // bswap(bswap(x)) -> x
    if (IntrinsicInst *Operand = dyn_cast<IntrinsicInst>(II->getOperand(1)))
      if (Operand->getIntrinsicID() == Intrinsic::bswap)
        return ReplaceInstUsesWith(CI, Operand->getOperand(1));
      
    // bswap(trunc(bswap(x))) -> trunc(lshr(x, c))
    if (TruncInst *TI = dyn_cast<TruncInst>(II->getOperand(1))) {
      if (IntrinsicInst *Operand = dyn_cast<IntrinsicInst>(TI->getOperand(0)))
        if (Operand->getIntrinsicID() == Intrinsic::bswap) {
          unsigned C = Operand->getType()->getPrimitiveSizeInBits() -
                       TI->getType()->getPrimitiveSizeInBits();
          Value *CV = ConstantInt::get(Operand->getType(), C);
          Value *V = Builder->CreateLShr(Operand->getOperand(1), CV);
          return new TruncInst(V, TI->getType());
        }
    }
      
    break;
  case Intrinsic::powi:
    if (ConstantInt *Power = dyn_cast<ConstantInt>(II->getOperand(2))) {
      // powi(x, 0) -> 1.0
      if (Power->isZero())
        return ReplaceInstUsesWith(CI, ConstantFP::get(CI.getType(), 1.0));
      // powi(x, 1) -> x
      if (Power->isOne())
        return ReplaceInstUsesWith(CI, II->getOperand(1));
      // powi(x, -1) -> 1/x
      if (Power->isAllOnesValue())
        return BinaryOperator::CreateFDiv(ConstantFP::get(CI.getType(), 1.0),
                                          II->getOperand(1));
    }
    break;
      
  case Intrinsic::uadd_with_overflow: {
    Value *LHS = II->getOperand(1), *RHS = II->getOperand(2);
    const IntegerType *IT = cast<IntegerType>(II->getOperand(1)->getType());
    uint32_t BitWidth = IT->getBitWidth();
    APInt Mask = APInt::getSignBit(BitWidth);
    APInt LHSKnownZero(BitWidth, 0);
    APInt LHSKnownOne(BitWidth, 0);
    ComputeMaskedBits(LHS, Mask, LHSKnownZero, LHSKnownOne);
    bool LHSKnownNegative = LHSKnownOne[BitWidth - 1];
    bool LHSKnownPositive = LHSKnownZero[BitWidth - 1];

    if (LHSKnownNegative || LHSKnownPositive) {
      APInt RHSKnownZero(BitWidth, 0);
      APInt RHSKnownOne(BitWidth, 0);
      ComputeMaskedBits(RHS, Mask, RHSKnownZero, RHSKnownOne);
      bool RHSKnownNegative = RHSKnownOne[BitWidth - 1];
      bool RHSKnownPositive = RHSKnownZero[BitWidth - 1];
      if (LHSKnownNegative && RHSKnownNegative) {
        // The sign bit is set in both cases: this MUST overflow.
        // Create a simple add instruction, and insert it into the struct.
        Instruction *Add = BinaryOperator::CreateAdd(LHS, RHS, "", &CI);
        Worklist.Add(Add);
        Constant *V[] = {
          UndefValue::get(LHS->getType()),ConstantInt::getTrue(II->getContext())
        };
        Constant *Struct = ConstantStruct::get(II->getContext(), V, 2, false);
        return InsertValueInst::Create(Struct, Add, 0);
      }
      
      if (LHSKnownPositive && RHSKnownPositive) {
        // The sign bit is clear in both cases: this CANNOT overflow.
        // Create a simple add instruction, and insert it into the struct.
        Instruction *Add = BinaryOperator::CreateNUWAdd(LHS, RHS, "", &CI);
        Worklist.Add(Add);
        Constant *V[] = {
          UndefValue::get(LHS->getType()),
          ConstantInt::getFalse(II->getContext())
        };
        Constant *Struct = ConstantStruct::get(II->getContext(), V, 2, false);
        return InsertValueInst::Create(Struct, Add, 0);
      }
    }
  }
  // FALL THROUGH uadd into sadd
  case Intrinsic::sadd_with_overflow:
    // Canonicalize constants into the RHS.
    if (isa<Constant>(II->getOperand(1)) &&
        !isa<Constant>(II->getOperand(2))) {
      Value *LHS = II->getOperand(1);
      II->setOperand(1, II->getOperand(2));
      II->setOperand(2, LHS);
      return II;
    }

    // X + undef -> undef
    if (isa<UndefValue>(II->getOperand(2)))
      return ReplaceInstUsesWith(CI, UndefValue::get(II->getType()));
      
    if (ConstantInt *RHS = dyn_cast<ConstantInt>(II->getOperand(2))) {
      // X + 0 -> {X, false}
      if (RHS->isZero()) {
        Constant *V[] = {
          UndefValue::get(II->getOperand(0)->getType()),
          ConstantInt::getFalse(II->getContext())
        };
        Constant *Struct = ConstantStruct::get(II->getContext(), V, 2, false);
        return InsertValueInst::Create(Struct, II->getOperand(1), 0);
      }
    }
    break;
  case Intrinsic::usub_with_overflow:
  case Intrinsic::ssub_with_overflow:
    // undef - X -> undef
    // X - undef -> undef
    if (isa<UndefValue>(II->getOperand(1)) ||
        isa<UndefValue>(II->getOperand(2)))
      return ReplaceInstUsesWith(CI, UndefValue::get(II->getType()));
      
    if (ConstantInt *RHS = dyn_cast<ConstantInt>(II->getOperand(2))) {
      // X - 0 -> {X, false}
      if (RHS->isZero()) {
        Constant *V[] = {
          UndefValue::get(II->getOperand(1)->getType()),
          ConstantInt::getFalse(II->getContext())
        };
        Constant *Struct = ConstantStruct::get(II->getContext(), V, 2, false);
        return InsertValueInst::Create(Struct, II->getOperand(1), 0);
      }
    }
    break;
  case Intrinsic::umul_with_overflow:
  case Intrinsic::smul_with_overflow:
    // Canonicalize constants into the RHS.
    if (isa<Constant>(II->getOperand(1)) &&
        !isa<Constant>(II->getOperand(2))) {
      Value *LHS = II->getOperand(1);
      II->setOperand(1, II->getOperand(2));
      II->setOperand(2, LHS);
      return II;
    }

    // X * undef -> undef
    if (isa<UndefValue>(II->getOperand(2)))
      return ReplaceInstUsesWith(CI, UndefValue::get(II->getType()));
      
    if (ConstantInt *RHSI = dyn_cast<ConstantInt>(II->getOperand(2))) {
      // X*0 -> {0, false}
      if (RHSI->isZero())
        return ReplaceInstUsesWith(CI, Constant::getNullValue(II->getType()));
      
      // X * 1 -> {X, false}
      if (RHSI->equalsInt(1)) {
        Constant *V[] = {
          UndefValue::get(II->getOperand(1)->getType()),
          ConstantInt::getFalse(II->getContext())
        };
        Constant *Struct = ConstantStruct::get(II->getContext(), V, 2, false);
        return InsertValueInst::Create(Struct, II->getOperand(1), 0);
      }
    }
    break;
  case Intrinsic::ppc_altivec_lvx:
  case Intrinsic::ppc_altivec_lvxl:
  case Intrinsic::x86_sse_loadu_ps:
  case Intrinsic::x86_sse2_loadu_pd:
  case Intrinsic::x86_sse2_loadu_dq:
    // Turn PPC lvx     -> load if the pointer is known aligned.
    // Turn X86 loadups -> load if the pointer is known aligned.
    if (GetOrEnforceKnownAlignment(II->getOperand(1), 16) >= 16) {
      Value *Ptr = Builder->CreateBitCast(II->getOperand(1),
                                         PointerType::getUnqual(II->getType()));
      return new LoadInst(Ptr);
    }
    break;
  case Intrinsic::ppc_altivec_stvx:
  case Intrinsic::ppc_altivec_stvxl:
    // Turn stvx -> store if the pointer is known aligned.
    if (GetOrEnforceKnownAlignment(II->getOperand(2), 16) >= 16) {
      const Type *OpPtrTy = 
        PointerType::getUnqual(II->getOperand(1)->getType());
      Value *Ptr = Builder->CreateBitCast(II->getOperand(2), OpPtrTy);
      return new StoreInst(II->getOperand(1), Ptr);
    }
    break;
  case Intrinsic::x86_sse_storeu_ps:
  case Intrinsic::x86_sse2_storeu_pd:
  case Intrinsic::x86_sse2_storeu_dq:
    // Turn X86 storeu -> store if the pointer is known aligned.
    if (GetOrEnforceKnownAlignment(II->getOperand(1), 16) >= 16) {
      const Type *OpPtrTy = 
        PointerType::getUnqual(II->getOperand(2)->getType());
      Value *Ptr = Builder->CreateBitCast(II->getOperand(1), OpPtrTy);
      return new StoreInst(II->getOperand(2), Ptr);
    }
    break;
    
  case Intrinsic::x86_sse_cvttss2si: {
    // These intrinsics only demands the 0th element of its input vector.  If
    // we can simplify the input based on that, do so now.
    unsigned VWidth =
      cast<VectorType>(II->getOperand(1)->getType())->getNumElements();
    APInt DemandedElts(VWidth, 1);
    APInt UndefElts(VWidth, 0);
    if (Value *V = SimplifyDemandedVectorElts(II->getOperand(1), DemandedElts,
                                              UndefElts)) {
      II->setOperand(1, V);
      return II;
    }
    break;
  }
    
  case Intrinsic::ppc_altivec_vperm:
    // Turn vperm(V1,V2,mask) -> shuffle(V1,V2,mask) if mask is a constant.
    if (ConstantVector *Mask = dyn_cast<ConstantVector>(II->getOperand(3))) {
      assert(Mask->getNumOperands() == 16 && "Bad type for intrinsic!");
      
      // Check that all of the elements are integer constants or undefs.
      bool AllEltsOk = true;
      for (unsigned i = 0; i != 16; ++i) {
        if (!isa<ConstantInt>(Mask->getOperand(i)) && 
            !isa<UndefValue>(Mask->getOperand(i))) {
          AllEltsOk = false;
          break;
        }
      }
      
      if (AllEltsOk) {
        // Cast the input vectors to byte vectors.
        Value *Op0 = Builder->CreateBitCast(II->getOperand(1), Mask->getType());
        Value *Op1 = Builder->CreateBitCast(II->getOperand(2), Mask->getType());
        Value *Result = UndefValue::get(Op0->getType());
        
        // Only extract each element once.
        Value *ExtractedElts[32];
        memset(ExtractedElts, 0, sizeof(ExtractedElts));
        
        for (unsigned i = 0; i != 16; ++i) {
          if (isa<UndefValue>(Mask->getOperand(i)))
            continue;
          unsigned Idx=cast<ConstantInt>(Mask->getOperand(i))->getZExtValue();
          Idx &= 31;  // Match the hardware behavior.
          
          if (ExtractedElts[Idx] == 0) {
            ExtractedElts[Idx] = 
              Builder->CreateExtractElement(Idx < 16 ? Op0 : Op1, 
                  ConstantInt::get(Type::getInt32Ty(II->getContext()),
                                   Idx&15, false), "tmp");
          }
        
          // Insert this value into the result vector.
          Result = Builder->CreateInsertElement(Result, ExtractedElts[Idx],
                         ConstantInt::get(Type::getInt32Ty(II->getContext()),
                                          i, false), "tmp");
        }
        return CastInst::Create(Instruction::BitCast, Result, CI.getType());
      }
    }
    break;

  case Intrinsic::stackrestore: {
    // If the save is right next to the restore, remove the restore.  This can
    // happen when variable allocas are DCE'd.
    if (IntrinsicInst *SS = dyn_cast<IntrinsicInst>(II->getOperand(1))) {
      if (SS->getIntrinsicID() == Intrinsic::stacksave) {
        BasicBlock::iterator BI = SS;
        if (&*++BI == II)
          return EraseInstFromFunction(CI);
      }
    }
    
    // Scan down this block to see if there is another stack restore in the
    // same block without an intervening call/alloca.
    BasicBlock::iterator BI = II;
    TerminatorInst *TI = II->getParent()->getTerminator();
    bool CannotRemove = false;
    for (++BI; &*BI != TI; ++BI) {
      if (isa<AllocaInst>(BI) || isMalloc(BI)) {
        CannotRemove = true;
        break;
      }
      if (CallInst *BCI = dyn_cast<CallInst>(BI)) {
        if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(BCI)) {
          // If there is a stackrestore below this one, remove this one.
          if (II->getIntrinsicID() == Intrinsic::stackrestore)
            return EraseInstFromFunction(CI);
          // Otherwise, ignore the intrinsic.
        } else {
          // If we found a non-intrinsic call, we can't remove the stack
          // restore.
          CannotRemove = true;
          break;
        }
      }
    }
    
    // If the stack restore is in a return/unwind block and if there are no
    // allocas or calls between the restore and the return, nuke the restore.
    if (!CannotRemove && (isa<ReturnInst>(TI) || isa<UnwindInst>(TI)))
      return EraseInstFromFunction(CI);
    break;
  }
  }

  return visitCallSite(II);
}

// InvokeInst simplification
//
Instruction *InstCombiner::visitInvokeInst(InvokeInst &II) {
  return visitCallSite(&II);
}

/// isSafeToEliminateVarargsCast - If this cast does not affect the value 
/// passed through the varargs area, we can eliminate the use of the cast.
static bool isSafeToEliminateVarargsCast(const CallSite CS,
                                         const CastInst * const CI,
                                         const TargetData * const TD,
                                         const int ix) {
  if (!CI->isLosslessCast())
    return false;

  // The size of ByVal arguments is derived from the type, so we
  // can't change to a type with a different size.  If the size were
  // passed explicitly we could avoid this check.
  if (!CS.paramHasAttr(ix, Attribute::ByVal))
    return true;

  const Type* SrcTy = 
            cast<PointerType>(CI->getOperand(0)->getType())->getElementType();
  const Type* DstTy = cast<PointerType>(CI->getType())->getElementType();
  if (!SrcTy->isSized() || !DstTy->isSized())
    return false;
  if (!TD || TD->getTypeAllocSize(SrcTy) != TD->getTypeAllocSize(DstTy))
    return false;
  return true;
}

// visitCallSite - Improvements for call and invoke instructions.
//
Instruction *InstCombiner::visitCallSite(CallSite CS) {
  bool Changed = false;

  // If the callee is a constexpr cast of a function, attempt to move the cast
  // to the arguments of the call/invoke.
  if (transformConstExprCastCall(CS)) return 0;

  Value *Callee = CS.getCalledValue();

  if (Function *CalleeF = dyn_cast<Function>(Callee))
    if (CalleeF->getCallingConv() != CS.getCallingConv()) {
      Instruction *OldCall = CS.getInstruction();
      // If the call and callee calling conventions don't match, this call must
      // be unreachable, as the call is undefined.
      new StoreInst(ConstantInt::getTrue(Callee->getContext()),
                UndefValue::get(Type::getInt1PtrTy(Callee->getContext())), 
                                  OldCall);
      // If OldCall dues not return void then replaceAllUsesWith undef.
      // This allows ValueHandlers and custom metadata to adjust itself.
      if (!OldCall->getType()->isVoidTy())
        OldCall->replaceAllUsesWith(UndefValue::get(OldCall->getType()));
      if (isa<CallInst>(OldCall))   // Not worth removing an invoke here.
        return EraseInstFromFunction(*OldCall);
      return 0;
    }

  if (isa<ConstantPointerNull>(Callee) || isa<UndefValue>(Callee)) {
    // This instruction is not reachable, just remove it.  We insert a store to
    // undef so that we know that this code is not reachable, despite the fact
    // that we can't modify the CFG here.
    new StoreInst(ConstantInt::getTrue(Callee->getContext()),
               UndefValue::get(Type::getInt1PtrTy(Callee->getContext())),
                  CS.getInstruction());

    // If CS dues not return void then replaceAllUsesWith undef.
    // This allows ValueHandlers and custom metadata to adjust itself.
    if (!CS.getInstruction()->getType()->isVoidTy())
      CS.getInstruction()->
        replaceAllUsesWith(UndefValue::get(CS.getInstruction()->getType()));

    if (InvokeInst *II = dyn_cast<InvokeInst>(CS.getInstruction())) {
      // Don't break the CFG, insert a dummy cond branch.
      BranchInst::Create(II->getNormalDest(), II->getUnwindDest(),
                         ConstantInt::getTrue(Callee->getContext()), II);
    }
    return EraseInstFromFunction(*CS.getInstruction());
  }

  if (BitCastInst *BC = dyn_cast<BitCastInst>(Callee))
    if (IntrinsicInst *In = dyn_cast<IntrinsicInst>(BC->getOperand(0)))
      if (In->getIntrinsicID() == Intrinsic::init_trampoline)
        return transformCallThroughTrampoline(CS);

  const PointerType *PTy = cast<PointerType>(Callee->getType());
  const FunctionType *FTy = cast<FunctionType>(PTy->getElementType());
  if (FTy->isVarArg()) {
    int ix = FTy->getNumParams() + (isa<InvokeInst>(Callee) ? 3 : 1);
    // See if we can optimize any arguments passed through the varargs area of
    // the call.
    for (CallSite::arg_iterator I = CS.arg_begin()+FTy->getNumParams(),
           E = CS.arg_end(); I != E; ++I, ++ix) {
      CastInst *CI = dyn_cast<CastInst>(*I);
      if (CI && isSafeToEliminateVarargsCast(CS, CI, TD, ix)) {
        *I = CI->getOperand(0);
        Changed = true;
      }
    }
  }

  if (isa<InlineAsm>(Callee) && !CS.doesNotThrow()) {
    // Inline asm calls cannot throw - mark them 'nounwind'.
    CS.setDoesNotThrow();
    Changed = true;
  }

  return Changed ? CS.getInstruction() : 0;
}

// transformConstExprCastCall - If the callee is a constexpr cast of a function,
// attempt to move the cast to the arguments of the call/invoke.
//
bool InstCombiner::transformConstExprCastCall(CallSite CS) {
  if (!isa<ConstantExpr>(CS.getCalledValue())) return false;
  ConstantExpr *CE = cast<ConstantExpr>(CS.getCalledValue());
  if (CE->getOpcode() != Instruction::BitCast || 
      !isa<Function>(CE->getOperand(0)))
    return false;
  Function *Callee = cast<Function>(CE->getOperand(0));
  Instruction *Caller = CS.getInstruction();
  const AttrListPtr &CallerPAL = CS.getAttributes();

  // Okay, this is a cast from a function to a different type.  Unless doing so
  // would cause a type conversion of one of our arguments, change this call to
  // be a direct call with arguments casted to the appropriate types.
  //
  const FunctionType *FT = Callee->getFunctionType();
  const Type *OldRetTy = Caller->getType();
  const Type *NewRetTy = FT->getReturnType();

  if (isa<StructType>(NewRetTy))
    return false; // TODO: Handle multiple return values.

  // Check to see if we are changing the return type...
  if (OldRetTy != NewRetTy) {
    if (Callee->isDeclaration() &&
        // Conversion is ok if changing from one pointer type to another or from
        // a pointer to an integer of the same size.
        !((isa<PointerType>(OldRetTy) || !TD ||
           OldRetTy == TD->getIntPtrType(Caller->getContext())) &&
          (isa<PointerType>(NewRetTy) || !TD ||
           NewRetTy == TD->getIntPtrType(Caller->getContext()))))
      return false;   // Cannot transform this return value.

    if (!Caller->use_empty() &&
        // void -> non-void is handled specially
        !NewRetTy->isVoidTy() && !CastInst::isCastable(NewRetTy, OldRetTy))
      return false;   // Cannot transform this return value.

    if (!CallerPAL.isEmpty() && !Caller->use_empty()) {
      Attributes RAttrs = CallerPAL.getRetAttributes();
      if (RAttrs & Attribute::typeIncompatible(NewRetTy))
        return false;   // Attribute not compatible with transformed value.
    }

    // If the callsite is an invoke instruction, and the return value is used by
    // a PHI node in a successor, we cannot change the return type of the call
    // because there is no place to put the cast instruction (without breaking
    // the critical edge).  Bail out in this case.
    if (!Caller->use_empty())
      if (InvokeInst *II = dyn_cast<InvokeInst>(Caller))
        for (Value::use_iterator UI = II->use_begin(), E = II->use_end();
             UI != E; ++UI)
          if (PHINode *PN = dyn_cast<PHINode>(*UI))
            if (PN->getParent() == II->getNormalDest() ||
                PN->getParent() == II->getUnwindDest())
              return false;
  }

  unsigned NumActualArgs = unsigned(CS.arg_end()-CS.arg_begin());
  unsigned NumCommonArgs = std::min(FT->getNumParams(), NumActualArgs);

  CallSite::arg_iterator AI = CS.arg_begin();
  for (unsigned i = 0, e = NumCommonArgs; i != e; ++i, ++AI) {
    const Type *ParamTy = FT->getParamType(i);
    const Type *ActTy = (*AI)->getType();

    if (!CastInst::isCastable(ActTy, ParamTy))
      return false;   // Cannot transform this parameter value.

    if (CallerPAL.getParamAttributes(i + 1) 
        & Attribute::typeIncompatible(ParamTy))
      return false;   // Attribute not compatible with transformed value.

    // Converting from one pointer type to another or between a pointer and an
    // integer of the same size is safe even if we do not have a body.
    bool isConvertible = ActTy == ParamTy ||
      (TD && ((isa<PointerType>(ParamTy) ||
      ParamTy == TD->getIntPtrType(Caller->getContext())) &&
              (isa<PointerType>(ActTy) ||
              ActTy == TD->getIntPtrType(Caller->getContext()))));
    if (Callee->isDeclaration() && !isConvertible) return false;
  }

  if (FT->getNumParams() < NumActualArgs && !FT->isVarArg() &&
      Callee->isDeclaration())
    return false;   // Do not delete arguments unless we have a function body.

  if (FT->getNumParams() < NumActualArgs && FT->isVarArg() &&
      !CallerPAL.isEmpty())
    // In this case we have more arguments than the new function type, but we
    // won't be dropping them.  Check that these extra arguments have attributes
    // that are compatible with being a vararg call argument.
    for (unsigned i = CallerPAL.getNumSlots(); i; --i) {
      if (CallerPAL.getSlot(i - 1).Index <= FT->getNumParams())
        break;
      Attributes PAttrs = CallerPAL.getSlot(i - 1).Attrs;
      if (PAttrs & Attribute::VarArgsIncompatible)
        return false;
    }

  // Okay, we decided that this is a safe thing to do: go ahead and start
  // inserting cast instructions as necessary...
  std::vector<Value*> Args;
  Args.reserve(NumActualArgs);
  SmallVector<AttributeWithIndex, 8> attrVec;
  attrVec.reserve(NumCommonArgs);

  // Get any return attributes.
  Attributes RAttrs = CallerPAL.getRetAttributes();

  // If the return value is not being used, the type may not be compatible
  // with the existing attributes.  Wipe out any problematic attributes.
  RAttrs &= ~Attribute::typeIncompatible(NewRetTy);

  // Add the new return attributes.
  if (RAttrs)
    attrVec.push_back(AttributeWithIndex::get(0, RAttrs));

  AI = CS.arg_begin();
  for (unsigned i = 0; i != NumCommonArgs; ++i, ++AI) {
    const Type *ParamTy = FT->getParamType(i);
    if ((*AI)->getType() == ParamTy) {
      Args.push_back(*AI);
    } else {
      Instruction::CastOps opcode = CastInst::getCastOpcode(*AI,
          false, ParamTy, false);
      Args.push_back(Builder->CreateCast(opcode, *AI, ParamTy, "tmp"));
    }

    // Add any parameter attributes.
    if (Attributes PAttrs = CallerPAL.getParamAttributes(i + 1))
      attrVec.push_back(AttributeWithIndex::get(i + 1, PAttrs));
  }

  // If the function takes more arguments than the call was taking, add them
  // now.
  for (unsigned i = NumCommonArgs; i != FT->getNumParams(); ++i)
    Args.push_back(Constant::getNullValue(FT->getParamType(i)));

  // If we are removing arguments to the function, emit an obnoxious warning.
  if (FT->getNumParams() < NumActualArgs) {
    if (!FT->isVarArg()) {
      errs() << "WARNING: While resolving call to function '"
             << Callee->getName() << "' arguments were dropped!\n";
    } else {
      // Add all of the arguments in their promoted form to the arg list.
      for (unsigned i = FT->getNumParams(); i != NumActualArgs; ++i, ++AI) {
        const Type *PTy = getPromotedType((*AI)->getType());
        if (PTy != (*AI)->getType()) {
          // Must promote to pass through va_arg area!
          Instruction::CastOps opcode =
            CastInst::getCastOpcode(*AI, false, PTy, false);
          Args.push_back(Builder->CreateCast(opcode, *AI, PTy, "tmp"));
        } else {
          Args.push_back(*AI);
        }

        // Add any parameter attributes.
        if (Attributes PAttrs = CallerPAL.getParamAttributes(i + 1))
          attrVec.push_back(AttributeWithIndex::get(i + 1, PAttrs));
      }
    }
  }

  if (Attributes FnAttrs =  CallerPAL.getFnAttributes())
    attrVec.push_back(AttributeWithIndex::get(~0, FnAttrs));

  if (NewRetTy->isVoidTy())
    Caller->setName("");   // Void type should not have a name.

  const AttrListPtr &NewCallerPAL = AttrListPtr::get(attrVec.begin(),
                                                     attrVec.end());

  Instruction *NC;
  if (InvokeInst *II = dyn_cast<InvokeInst>(Caller)) {
    NC = InvokeInst::Create(Callee, II->getNormalDest(), II->getUnwindDest(),
                            Args.begin(), Args.end(),
                            Caller->getName(), Caller);
    cast<InvokeInst>(NC)->setCallingConv(II->getCallingConv());
    cast<InvokeInst>(NC)->setAttributes(NewCallerPAL);
  } else {
    NC = CallInst::Create(Callee, Args.begin(), Args.end(),
                          Caller->getName(), Caller);
    CallInst *CI = cast<CallInst>(Caller);
    if (CI->isTailCall())
      cast<CallInst>(NC)->setTailCall();
    cast<CallInst>(NC)->setCallingConv(CI->getCallingConv());
    cast<CallInst>(NC)->setAttributes(NewCallerPAL);
  }

  // Insert a cast of the return type as necessary.
  Value *NV = NC;
  if (OldRetTy != NV->getType() && !Caller->use_empty()) {
    if (!NV->getType()->isVoidTy()) {
      Instruction::CastOps opcode = CastInst::getCastOpcode(NC, false, 
                                                            OldRetTy, false);
      NV = NC = CastInst::Create(opcode, NC, OldRetTy, "tmp");

      // If this is an invoke instruction, we should insert it after the first
      // non-phi, instruction in the normal successor block.
      if (InvokeInst *II = dyn_cast<InvokeInst>(Caller)) {
        BasicBlock::iterator I = II->getNormalDest()->getFirstNonPHI();
        InsertNewInstBefore(NC, *I);
      } else {
        // Otherwise, it's a call, just insert cast right after the call instr
        InsertNewInstBefore(NC, *Caller);
      }
      Worklist.AddUsersToWorkList(*Caller);
    } else {
      NV = UndefValue::get(Caller->getType());
    }
  }


  if (!Caller->use_empty())
    Caller->replaceAllUsesWith(NV);
  
  EraseInstFromFunction(*Caller);
  return true;
}

// transformCallThroughTrampoline - Turn a call to a function created by the
// init_trampoline intrinsic into a direct call to the underlying function.
//
Instruction *InstCombiner::transformCallThroughTrampoline(CallSite CS) {
  Value *Callee = CS.getCalledValue();
  const PointerType *PTy = cast<PointerType>(Callee->getType());
  const FunctionType *FTy = cast<FunctionType>(PTy->getElementType());
  const AttrListPtr &Attrs = CS.getAttributes();

  // If the call already has the 'nest' attribute somewhere then give up -
  // otherwise 'nest' would occur twice after splicing in the chain.
  if (Attrs.hasAttrSomewhere(Attribute::Nest))
    return 0;

  IntrinsicInst *Tramp =
    cast<IntrinsicInst>(cast<BitCastInst>(Callee)->getOperand(0));

  Function *NestF = cast<Function>(Tramp->getOperand(2)->stripPointerCasts());
  const PointerType *NestFPTy = cast<PointerType>(NestF->getType());
  const FunctionType *NestFTy = cast<FunctionType>(NestFPTy->getElementType());

  const AttrListPtr &NestAttrs = NestF->getAttributes();
  if (!NestAttrs.isEmpty()) {
    unsigned NestIdx = 1;
    const Type *NestTy = 0;
    Attributes NestAttr = Attribute::None;

    // Look for a parameter marked with the 'nest' attribute.
    for (FunctionType::param_iterator I = NestFTy->param_begin(),
         E = NestFTy->param_end(); I != E; ++NestIdx, ++I)
      if (NestAttrs.paramHasAttr(NestIdx, Attribute::Nest)) {
        // Record the parameter type and any other attributes.
        NestTy = *I;
        NestAttr = NestAttrs.getParamAttributes(NestIdx);
        break;
      }

    if (NestTy) {
      Instruction *Caller = CS.getInstruction();
      std::vector<Value*> NewArgs;
      NewArgs.reserve(unsigned(CS.arg_end()-CS.arg_begin())+1);

      SmallVector<AttributeWithIndex, 8> NewAttrs;
      NewAttrs.reserve(Attrs.getNumSlots() + 1);

      // Insert the nest argument into the call argument list, which may
      // mean appending it.  Likewise for attributes.

      // Add any result attributes.
      if (Attributes Attr = Attrs.getRetAttributes())
        NewAttrs.push_back(AttributeWithIndex::get(0, Attr));

      {
        unsigned Idx = 1;
        CallSite::arg_iterator I = CS.arg_begin(), E = CS.arg_end();
        do {
          if (Idx == NestIdx) {
            // Add the chain argument and attributes.
            Value *NestVal = Tramp->getOperand(3);
            if (NestVal->getType() != NestTy)
              NestVal = new BitCastInst(NestVal, NestTy, "nest", Caller);
            NewArgs.push_back(NestVal);
            NewAttrs.push_back(AttributeWithIndex::get(NestIdx, NestAttr));
          }

          if (I == E)
            break;

          // Add the original argument and attributes.
          NewArgs.push_back(*I);
          if (Attributes Attr = Attrs.getParamAttributes(Idx))
            NewAttrs.push_back
              (AttributeWithIndex::get(Idx + (Idx >= NestIdx), Attr));

          ++Idx, ++I;
        } while (1);
      }

      // Add any function attributes.
      if (Attributes Attr = Attrs.getFnAttributes())
        NewAttrs.push_back(AttributeWithIndex::get(~0, Attr));

      // The trampoline may have been bitcast to a bogus type (FTy).
      // Handle this by synthesizing a new function type, equal to FTy
      // with the chain parameter inserted.

      std::vector<const Type*> NewTypes;
      NewTypes.reserve(FTy->getNumParams()+1);

      // Insert the chain's type into the list of parameter types, which may
      // mean appending it.
      {
        unsigned Idx = 1;
        FunctionType::param_iterator I = FTy->param_begin(),
          E = FTy->param_end();

        do {
          if (Idx == NestIdx)
            // Add the chain's type.
            NewTypes.push_back(NestTy);

          if (I == E)
            break;

          // Add the original type.
          NewTypes.push_back(*I);

          ++Idx, ++I;
        } while (1);
      }

      // Replace the trampoline call with a direct call.  Let the generic
      // code sort out any function type mismatches.
      FunctionType *NewFTy = FunctionType::get(FTy->getReturnType(), NewTypes, 
                                                FTy->isVarArg());
      Constant *NewCallee =
        NestF->getType() == PointerType::getUnqual(NewFTy) ?
        NestF : ConstantExpr::getBitCast(NestF, 
                                         PointerType::getUnqual(NewFTy));
      const AttrListPtr &NewPAL = AttrListPtr::get(NewAttrs.begin(),
                                                   NewAttrs.end());

      Instruction *NewCaller;
      if (InvokeInst *II = dyn_cast<InvokeInst>(Caller)) {
        NewCaller = InvokeInst::Create(NewCallee,
                                       II->getNormalDest(), II->getUnwindDest(),
                                       NewArgs.begin(), NewArgs.end(),
                                       Caller->getName(), Caller);
        cast<InvokeInst>(NewCaller)->setCallingConv(II->getCallingConv());
        cast<InvokeInst>(NewCaller)->setAttributes(NewPAL);
      } else {
        NewCaller = CallInst::Create(NewCallee, NewArgs.begin(), NewArgs.end(),
                                     Caller->getName(), Caller);
        if (cast<CallInst>(Caller)->isTailCall())
          cast<CallInst>(NewCaller)->setTailCall();
        cast<CallInst>(NewCaller)->
          setCallingConv(cast<CallInst>(Caller)->getCallingConv());
        cast<CallInst>(NewCaller)->setAttributes(NewPAL);
      }
      if (!Caller->getType()->isVoidTy())
        Caller->replaceAllUsesWith(NewCaller);
      Caller->eraseFromParent();
      Worklist.Remove(Caller);
      return 0;
    }
  }

  // Replace the trampoline call with a direct call.  Since there is no 'nest'
  // parameter, there is no need to adjust the argument list.  Let the generic
  // code sort out any function type mismatches.
  Constant *NewCallee =
    NestF->getType() == PTy ? NestF : 
                              ConstantExpr::getBitCast(NestF, PTy);
  CS.setCalledFunction(NewCallee);
  return CS.getInstruction();
}

/// FoldPHIArgBinOpIntoPHI - If we have something like phi [add (a,b), add(a,c)]
/// and if a/b/c and the add's all have a single use, turn this into a phi
/// and a single binop.
Instruction *InstCombiner::FoldPHIArgBinOpIntoPHI(PHINode &PN) {
  Instruction *FirstInst = cast<Instruction>(PN.getIncomingValue(0));
  assert(isa<BinaryOperator>(FirstInst) || isa<CmpInst>(FirstInst));
  unsigned Opc = FirstInst->getOpcode();
  Value *LHSVal = FirstInst->getOperand(0);
  Value *RHSVal = FirstInst->getOperand(1);
    
  const Type *LHSType = LHSVal->getType();
  const Type *RHSType = RHSVal->getType();
  
  // Scan to see if all operands are the same opcode, and all have one use.
  for (unsigned i = 1; i != PN.getNumIncomingValues(); ++i) {
    Instruction *I = dyn_cast<Instruction>(PN.getIncomingValue(i));
    if (!I || I->getOpcode() != Opc || !I->hasOneUse() ||
        // Verify type of the LHS matches so we don't fold cmp's of different
        // types or GEP's with different index types.
        I->getOperand(0)->getType() != LHSType ||
        I->getOperand(1)->getType() != RHSType)
      return 0;

    // If they are CmpInst instructions, check their predicates
    if (Opc == Instruction::ICmp || Opc == Instruction::FCmp)
      if (cast<CmpInst>(I)->getPredicate() !=
          cast<CmpInst>(FirstInst)->getPredicate())
        return 0;
    
    // Keep track of which operand needs a phi node.
    if (I->getOperand(0) != LHSVal) LHSVal = 0;
    if (I->getOperand(1) != RHSVal) RHSVal = 0;
  }

  // If both LHS and RHS would need a PHI, don't do this transformation,
  // because it would increase the number of PHIs entering the block,
  // which leads to higher register pressure. This is especially
  // bad when the PHIs are in the header of a loop.
  if (!LHSVal && !RHSVal)
    return 0;
  
  // Otherwise, this is safe to transform!
  
  Value *InLHS = FirstInst->getOperand(0);
  Value *InRHS = FirstInst->getOperand(1);
  PHINode *NewLHS = 0, *NewRHS = 0;
  if (LHSVal == 0) {
    NewLHS = PHINode::Create(LHSType,
                             FirstInst->getOperand(0)->getName() + ".pn");
    NewLHS->reserveOperandSpace(PN.getNumOperands()/2);
    NewLHS->addIncoming(InLHS, PN.getIncomingBlock(0));
    InsertNewInstBefore(NewLHS, PN);
    LHSVal = NewLHS;
  }
  
  if (RHSVal == 0) {
    NewRHS = PHINode::Create(RHSType,
                             FirstInst->getOperand(1)->getName() + ".pn");
    NewRHS->reserveOperandSpace(PN.getNumOperands()/2);
    NewRHS->addIncoming(InRHS, PN.getIncomingBlock(0));
    InsertNewInstBefore(NewRHS, PN);
    RHSVal = NewRHS;
  }
  
  // Add all operands to the new PHIs.
  if (NewLHS || NewRHS) {
    for (unsigned i = 1, e = PN.getNumIncomingValues(); i != e; ++i) {
      Instruction *InInst = cast<Instruction>(PN.getIncomingValue(i));
      if (NewLHS) {
        Value *NewInLHS = InInst->getOperand(0);
        NewLHS->addIncoming(NewInLHS, PN.getIncomingBlock(i));
      }
      if (NewRHS) {
        Value *NewInRHS = InInst->getOperand(1);
        NewRHS->addIncoming(NewInRHS, PN.getIncomingBlock(i));
      }
    }
  }
    
  if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(FirstInst))
    return BinaryOperator::Create(BinOp->getOpcode(), LHSVal, RHSVal);
  CmpInst *CIOp = cast<CmpInst>(FirstInst);
  return CmpInst::Create(CIOp->getOpcode(), CIOp->getPredicate(),
                         LHSVal, RHSVal);
}

Instruction *InstCombiner::FoldPHIArgGEPIntoPHI(PHINode &PN) {
  GetElementPtrInst *FirstInst =cast<GetElementPtrInst>(PN.getIncomingValue(0));
  
  SmallVector<Value*, 16> FixedOperands(FirstInst->op_begin(), 
                                        FirstInst->op_end());
  // This is true if all GEP bases are allocas and if all indices into them are
  // constants.
  bool AllBasePointersAreAllocas = true;

  // We don't want to replace this phi if the replacement would require
  // more than one phi, which leads to higher register pressure. This is
  // especially bad when the PHIs are in the header of a loop.
  bool NeededPhi = false;
  
  // Scan to see if all operands are the same opcode, and all have one use.
  for (unsigned i = 1; i != PN.getNumIncomingValues(); ++i) {
    GetElementPtrInst *GEP= dyn_cast<GetElementPtrInst>(PN.getIncomingValue(i));
    if (!GEP || !GEP->hasOneUse() || GEP->getType() != FirstInst->getType() ||
      GEP->getNumOperands() != FirstInst->getNumOperands())
      return 0;

    // Keep track of whether or not all GEPs are of alloca pointers.
    if (AllBasePointersAreAllocas &&
        (!isa<AllocaInst>(GEP->getOperand(0)) ||
         !GEP->hasAllConstantIndices()))
      AllBasePointersAreAllocas = false;
    
    // Compare the operand lists.
    for (unsigned op = 0, e = FirstInst->getNumOperands(); op != e; ++op) {
      if (FirstInst->getOperand(op) == GEP->getOperand(op))
        continue;
      
      // Don't merge two GEPs when two operands differ (introducing phi nodes)
      // if one of the PHIs has a constant for the index.  The index may be
      // substantially cheaper to compute for the constants, so making it a
      // variable index could pessimize the path.  This also handles the case
      // for struct indices, which must always be constant.
      if (isa<ConstantInt>(FirstInst->getOperand(op)) ||
          isa<ConstantInt>(GEP->getOperand(op)))
        return 0;
      
      if (FirstInst->getOperand(op)->getType() !=GEP->getOperand(op)->getType())
        return 0;

      // If we already needed a PHI for an earlier operand, and another operand
      // also requires a PHI, we'd be introducing more PHIs than we're
      // eliminating, which increases register pressure on entry to the PHI's
      // block.
      if (NeededPhi)
        return 0;

      FixedOperands[op] = 0;  // Needs a PHI.
      NeededPhi = true;
    }
  }
  
  // If all of the base pointers of the PHI'd GEPs are from allocas, don't
  // bother doing this transformation.  At best, this will just save a bit of
  // offset calculation, but all the predecessors will have to materialize the
  // stack address into a register anyway.  We'd actually rather *clone* the
  // load up into the predecessors so that we have a load of a gep of an alloca,
  // which can usually all be folded into the load.
  if (AllBasePointersAreAllocas)
    return 0;
  
  // Otherwise, this is safe to transform.  Insert PHI nodes for each operand
  // that is variable.
  SmallVector<PHINode*, 16> OperandPhis(FixedOperands.size());
  
  bool HasAnyPHIs = false;
  for (unsigned i = 0, e = FixedOperands.size(); i != e; ++i) {
    if (FixedOperands[i]) continue;  // operand doesn't need a phi.
    Value *FirstOp = FirstInst->getOperand(i);
    PHINode *NewPN = PHINode::Create(FirstOp->getType(),
                                     FirstOp->getName()+".pn");
    InsertNewInstBefore(NewPN, PN);
    
    NewPN->reserveOperandSpace(e);
    NewPN->addIncoming(FirstOp, PN.getIncomingBlock(0));
    OperandPhis[i] = NewPN;
    FixedOperands[i] = NewPN;
    HasAnyPHIs = true;
  }

  
  // Add all operands to the new PHIs.
  if (HasAnyPHIs) {
    for (unsigned i = 1, e = PN.getNumIncomingValues(); i != e; ++i) {
      GetElementPtrInst *InGEP =cast<GetElementPtrInst>(PN.getIncomingValue(i));
      BasicBlock *InBB = PN.getIncomingBlock(i);
      
      for (unsigned op = 0, e = OperandPhis.size(); op != e; ++op)
        if (PHINode *OpPhi = OperandPhis[op])
          OpPhi->addIncoming(InGEP->getOperand(op), InBB);
    }
  }
  
  Value *Base = FixedOperands[0];
  return cast<GEPOperator>(FirstInst)->isInBounds() ?
    GetElementPtrInst::CreateInBounds(Base, FixedOperands.begin()+1,
                                      FixedOperands.end()) :
    GetElementPtrInst::Create(Base, FixedOperands.begin()+1,
                              FixedOperands.end());
}


/// isSafeAndProfitableToSinkLoad - Return true if we know that it is safe to
/// sink the load out of the block that defines it.  This means that it must be
/// obvious the value of the load is not changed from the point of the load to
/// the end of the block it is in.
///
/// Finally, it is safe, but not profitable, to sink a load targetting a
/// non-address-taken alloca.  Doing so will cause us to not promote the alloca
/// to a register.
static bool isSafeAndProfitableToSinkLoad(LoadInst *L) {
  BasicBlock::iterator BBI = L, E = L->getParent()->end();
  
  for (++BBI; BBI != E; ++BBI)
    if (BBI->mayWriteToMemory())
      return false;
  
  // Check for non-address taken alloca.  If not address-taken already, it isn't
  // profitable to do this xform.
  if (AllocaInst *AI = dyn_cast<AllocaInst>(L->getOperand(0))) {
    bool isAddressTaken = false;
    for (Value::use_iterator UI = AI->use_begin(), E = AI->use_end();
         UI != E; ++UI) {
      if (isa<LoadInst>(UI)) continue;
      if (StoreInst *SI = dyn_cast<StoreInst>(*UI)) {
        // If storing TO the alloca, then the address isn't taken.
        if (SI->getOperand(1) == AI) continue;
      }
      isAddressTaken = true;
      break;
    }
    
    if (!isAddressTaken && AI->isStaticAlloca())
      return false;
  }
  
  // If this load is a load from a GEP with a constant offset from an alloca,
  // then we don't want to sink it.  In its present form, it will be
  // load [constant stack offset].  Sinking it will cause us to have to
  // materialize the stack addresses in each predecessor in a register only to
  // do a shared load from register in the successor.
  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(L->getOperand(0)))
    if (AllocaInst *AI = dyn_cast<AllocaInst>(GEP->getOperand(0)))
      if (AI->isStaticAlloca() && GEP->hasAllConstantIndices())
        return false;
  
  return true;
}

Instruction *InstCombiner::FoldPHIArgLoadIntoPHI(PHINode &PN) {
  LoadInst *FirstLI = cast<LoadInst>(PN.getIncomingValue(0));
  
  // When processing loads, we need to propagate two bits of information to the
  // sunk load: whether it is volatile, and what its alignment is.  We currently
  // don't sink loads when some have their alignment specified and some don't.
  // visitLoadInst will propagate an alignment onto the load when TD is around,
  // and if TD isn't around, we can't handle the mixed case.
  bool isVolatile = FirstLI->isVolatile();
  unsigned LoadAlignment = FirstLI->getAlignment();
  
  // We can't sink the load if the loaded value could be modified between the
  // load and the PHI.
  if (FirstLI->getParent() != PN.getIncomingBlock(0) ||
      !isSafeAndProfitableToSinkLoad(FirstLI))
    return 0;
  
  // If the PHI is of volatile loads and the load block has multiple
  // successors, sinking it would remove a load of the volatile value from
  // the path through the other successor.
  if (isVolatile && 
      FirstLI->getParent()->getTerminator()->getNumSuccessors() != 1)
    return 0;
  
  // Check to see if all arguments are the same operation.
  for (unsigned i = 1, e = PN.getNumIncomingValues(); i != e; ++i) {
    LoadInst *LI = dyn_cast<LoadInst>(PN.getIncomingValue(i));
    if (!LI || !LI->hasOneUse())
      return 0;
    
    // We can't sink the load if the loaded value could be modified between 
    // the load and the PHI.
    if (LI->isVolatile() != isVolatile ||
        LI->getParent() != PN.getIncomingBlock(i) ||
        !isSafeAndProfitableToSinkLoad(LI))
      return 0;
      
    // If some of the loads have an alignment specified but not all of them,
    // we can't do the transformation.
    if ((LoadAlignment != 0) != (LI->getAlignment() != 0))
      return 0;
    
    LoadAlignment = std::min(LoadAlignment, LI->getAlignment());
    
    // If the PHI is of volatile loads and the load block has multiple
    // successors, sinking it would remove a load of the volatile value from
    // the path through the other successor.
    if (isVolatile &&
        LI->getParent()->getTerminator()->getNumSuccessors() != 1)
      return 0;
  }
  
  // Okay, they are all the same operation.  Create a new PHI node of the
  // correct type, and PHI together all of the LHS's of the instructions.
  PHINode *NewPN = PHINode::Create(FirstLI->getOperand(0)->getType(),
                                   PN.getName()+".in");
  NewPN->reserveOperandSpace(PN.getNumOperands()/2);
  
  Value *InVal = FirstLI->getOperand(0);
  NewPN->addIncoming(InVal, PN.getIncomingBlock(0));
  
  // Add all operands to the new PHI.
  for (unsigned i = 1, e = PN.getNumIncomingValues(); i != e; ++i) {
    Value *NewInVal = cast<LoadInst>(PN.getIncomingValue(i))->getOperand(0);
    if (NewInVal != InVal)
      InVal = 0;
    NewPN->addIncoming(NewInVal, PN.getIncomingBlock(i));
  }
  
  Value *PhiVal;
  if (InVal) {
    // The new PHI unions all of the same values together.  This is really
    // common, so we handle it intelligently here for compile-time speed.
    PhiVal = InVal;
    delete NewPN;
  } else {
    InsertNewInstBefore(NewPN, PN);
    PhiVal = NewPN;
  }
  
  // If this was a volatile load that we are merging, make sure to loop through
  // and mark all the input loads as non-volatile.  If we don't do this, we will
  // insert a new volatile load and the old ones will not be deletable.
  if (isVolatile)
    for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i)
      cast<LoadInst>(PN.getIncomingValue(i))->setVolatile(false);
  
  return new LoadInst(PhiVal, "", isVolatile, LoadAlignment);
}



/// FoldPHIArgOpIntoPHI - If all operands to a PHI node are the same "unary"
/// operator and they all are only used by the PHI, PHI together their
/// inputs, and do the operation once, to the result of the PHI.
Instruction *InstCombiner::FoldPHIArgOpIntoPHI(PHINode &PN) {
  Instruction *FirstInst = cast<Instruction>(PN.getIncomingValue(0));

  if (isa<GetElementPtrInst>(FirstInst))
    return FoldPHIArgGEPIntoPHI(PN);
  if (isa<LoadInst>(FirstInst))
    return FoldPHIArgLoadIntoPHI(PN);
  
  // Scan the instruction, looking for input operations that can be folded away.
  // If all input operands to the phi are the same instruction (e.g. a cast from
  // the same type or "+42") we can pull the operation through the PHI, reducing
  // code size and simplifying code.
  Constant *ConstantOp = 0;
  const Type *CastSrcTy = 0;
  
  if (isa<CastInst>(FirstInst)) {
    CastSrcTy = FirstInst->getOperand(0)->getType();

    // Be careful about transforming integer PHIs.  We don't want to pessimize
    // the code by turning an i32 into an i1293.
    if (isa<IntegerType>(PN.getType()) && isa<IntegerType>(CastSrcTy)) {
      if (!ShouldChangeType(PN.getType(), CastSrcTy))
        return 0;
    }
  } else if (isa<BinaryOperator>(FirstInst) || isa<CmpInst>(FirstInst)) {
    // Can fold binop, compare or shift here if the RHS is a constant, 
    // otherwise call FoldPHIArgBinOpIntoPHI.
    ConstantOp = dyn_cast<Constant>(FirstInst->getOperand(1));
    if (ConstantOp == 0)
      return FoldPHIArgBinOpIntoPHI(PN);
  } else {
    return 0;  // Cannot fold this operation.
  }

  // Check to see if all arguments are the same operation.
  for (unsigned i = 1, e = PN.getNumIncomingValues(); i != e; ++i) {
    Instruction *I = dyn_cast<Instruction>(PN.getIncomingValue(i));
    if (I == 0 || !I->hasOneUse() || !I->isSameOperationAs(FirstInst))
      return 0;
    if (CastSrcTy) {
      if (I->getOperand(0)->getType() != CastSrcTy)
        return 0;  // Cast operation must match.
    } else if (I->getOperand(1) != ConstantOp) {
      return 0;
    }
  }

  // Okay, they are all the same operation.  Create a new PHI node of the
  // correct type, and PHI together all of the LHS's of the instructions.
  PHINode *NewPN = PHINode::Create(FirstInst->getOperand(0)->getType(),
                                   PN.getName()+".in");
  NewPN->reserveOperandSpace(PN.getNumOperands()/2);

  Value *InVal = FirstInst->getOperand(0);
  NewPN->addIncoming(InVal, PN.getIncomingBlock(0));

  // Add all operands to the new PHI.
  for (unsigned i = 1, e = PN.getNumIncomingValues(); i != e; ++i) {
    Value *NewInVal = cast<Instruction>(PN.getIncomingValue(i))->getOperand(0);
    if (NewInVal != InVal)
      InVal = 0;
    NewPN->addIncoming(NewInVal, PN.getIncomingBlock(i));
  }

  Value *PhiVal;
  if (InVal) {
    // The new PHI unions all of the same values together.  This is really
    // common, so we handle it intelligently here for compile-time speed.
    PhiVal = InVal;
    delete NewPN;
  } else {
    InsertNewInstBefore(NewPN, PN);
    PhiVal = NewPN;
  }

  // Insert and return the new operation.
  if (CastInst *FirstCI = dyn_cast<CastInst>(FirstInst))
    return CastInst::Create(FirstCI->getOpcode(), PhiVal, PN.getType());
  
  if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(FirstInst))
    return BinaryOperator::Create(BinOp->getOpcode(), PhiVal, ConstantOp);
  
  CmpInst *CIOp = cast<CmpInst>(FirstInst);
  return CmpInst::Create(CIOp->getOpcode(), CIOp->getPredicate(),
                         PhiVal, ConstantOp);
}

/// DeadPHICycle - Return true if this PHI node is only used by a PHI node cycle
/// that is dead.
static bool DeadPHICycle(PHINode *PN,
                         SmallPtrSet<PHINode*, 16> &PotentiallyDeadPHIs) {
  if (PN->use_empty()) return true;
  if (!PN->hasOneUse()) return false;

  // Remember this node, and if we find the cycle, return.
  if (!PotentiallyDeadPHIs.insert(PN))
    return true;
  
  // Don't scan crazily complex things.
  if (PotentiallyDeadPHIs.size() == 16)
    return false;

  if (PHINode *PU = dyn_cast<PHINode>(PN->use_back()))
    return DeadPHICycle(PU, PotentiallyDeadPHIs);

  return false;
}

/// PHIsEqualValue - Return true if this phi node is always equal to
/// NonPhiInVal.  This happens with mutually cyclic phi nodes like:
///   z = some value; x = phi (y, z); y = phi (x, z)
static bool PHIsEqualValue(PHINode *PN, Value *NonPhiInVal, 
                           SmallPtrSet<PHINode*, 16> &ValueEqualPHIs) {
  // See if we already saw this PHI node.
  if (!ValueEqualPHIs.insert(PN))
    return true;
  
  // Don't scan crazily complex things.
  if (ValueEqualPHIs.size() == 16)
    return false;
 
  // Scan the operands to see if they are either phi nodes or are equal to
  // the value.
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
    Value *Op = PN->getIncomingValue(i);
    if (PHINode *OpPN = dyn_cast<PHINode>(Op)) {
      if (!PHIsEqualValue(OpPN, NonPhiInVal, ValueEqualPHIs))
        return false;
    } else if (Op != NonPhiInVal)
      return false;
  }
  
  return true;
}


namespace {
struct PHIUsageRecord {
  unsigned PHIId;     // The ID # of the PHI (something determinstic to sort on)
  unsigned Shift;     // The amount shifted.
  Instruction *Inst;  // The trunc instruction.
  
  PHIUsageRecord(unsigned pn, unsigned Sh, Instruction *User)
    : PHIId(pn), Shift(Sh), Inst(User) {}
  
  bool operator<(const PHIUsageRecord &RHS) const {
    if (PHIId < RHS.PHIId) return true;
    if (PHIId > RHS.PHIId) return false;
    if (Shift < RHS.Shift) return true;
    if (Shift > RHS.Shift) return false;
    return Inst->getType()->getPrimitiveSizeInBits() <
           RHS.Inst->getType()->getPrimitiveSizeInBits();
  }
};
  
struct LoweredPHIRecord {
  PHINode *PN;        // The PHI that was lowered.
  unsigned Shift;     // The amount shifted.
  unsigned Width;     // The width extracted.
  
  LoweredPHIRecord(PHINode *pn, unsigned Sh, const Type *Ty)
    : PN(pn), Shift(Sh), Width(Ty->getPrimitiveSizeInBits()) {}
  
  // Ctor form used by DenseMap.
  LoweredPHIRecord(PHINode *pn, unsigned Sh)
    : PN(pn), Shift(Sh), Width(0) {}
};
}

namespace llvm {
  template<>
  struct DenseMapInfo<LoweredPHIRecord> {
    static inline LoweredPHIRecord getEmptyKey() {
      return LoweredPHIRecord(0, 0);
    }
    static inline LoweredPHIRecord getTombstoneKey() {
      return LoweredPHIRecord(0, 1);
    }
    static unsigned getHashValue(const LoweredPHIRecord &Val) {
      return DenseMapInfo<PHINode*>::getHashValue(Val.PN) ^ (Val.Shift>>3) ^
             (Val.Width>>3);
    }
    static bool isEqual(const LoweredPHIRecord &LHS,
                        const LoweredPHIRecord &RHS) {
      return LHS.PN == RHS.PN && LHS.Shift == RHS.Shift &&
             LHS.Width == RHS.Width;
    }
  };
  template <>
  struct isPodLike<LoweredPHIRecord> { static const bool value = true; };
}


/// SliceUpIllegalIntegerPHI - This is an integer PHI and we know that it has an
/// illegal type: see if it is only used by trunc or trunc(lshr) operations.  If
/// so, we split the PHI into the various pieces being extracted.  This sort of
/// thing is introduced when SROA promotes an aggregate to large integer values.
///
/// TODO: The user of the trunc may be an bitcast to float/double/vector or an
/// inttoptr.  We should produce new PHIs in the right type.
///
Instruction *InstCombiner::SliceUpIllegalIntegerPHI(PHINode &FirstPhi) {
  // PHIUsers - Keep track of all of the truncated values extracted from a set
  // of PHIs, along with their offset.  These are the things we want to rewrite.
  SmallVector<PHIUsageRecord, 16> PHIUsers;
  
  // PHIs are often mutually cyclic, so we keep track of a whole set of PHI
  // nodes which are extracted from. PHIsToSlice is a set we use to avoid
  // revisiting PHIs, PHIsInspected is a ordered list of PHIs that we need to
  // check the uses of (to ensure they are all extracts).
  SmallVector<PHINode*, 8> PHIsToSlice;
  SmallPtrSet<PHINode*, 8> PHIsInspected;
  
  PHIsToSlice.push_back(&FirstPhi);
  PHIsInspected.insert(&FirstPhi);
  
  for (unsigned PHIId = 0; PHIId != PHIsToSlice.size(); ++PHIId) {
    PHINode *PN = PHIsToSlice[PHIId];
    
    // Scan the input list of the PHI.  If any input is an invoke, and if the
    // input is defined in the predecessor, then we won't be split the critical
    // edge which is required to insert a truncate.  Because of this, we have to
    // bail out.
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
      InvokeInst *II = dyn_cast<InvokeInst>(PN->getIncomingValue(i));
      if (II == 0) continue;
      if (II->getParent() != PN->getIncomingBlock(i))
        continue;
     
      // If we have a phi, and if it's directly in the predecessor, then we have
      // a critical edge where we need to put the truncate.  Since we can't
      // split the edge in instcombine, we have to bail out.
      return 0;
    }
      
    
    for (Value::use_iterator UI = PN->use_begin(), E = PN->use_end();
         UI != E; ++UI) {
      Instruction *User = cast<Instruction>(*UI);
      
      // If the user is a PHI, inspect its uses recursively.
      if (PHINode *UserPN = dyn_cast<PHINode>(User)) {
        if (PHIsInspected.insert(UserPN))
          PHIsToSlice.push_back(UserPN);
        continue;
      }
      
      // Truncates are always ok.
      if (isa<TruncInst>(User)) {
        PHIUsers.push_back(PHIUsageRecord(PHIId, 0, User));
        continue;
      }
      
      // Otherwise it must be a lshr which can only be used by one trunc.
      if (User->getOpcode() != Instruction::LShr ||
          !User->hasOneUse() || !isa<TruncInst>(User->use_back()) ||
          !isa<ConstantInt>(User->getOperand(1)))
        return 0;
      
      unsigned Shift = cast<ConstantInt>(User->getOperand(1))->getZExtValue();
      PHIUsers.push_back(PHIUsageRecord(PHIId, Shift, User->use_back()));
    }
  }
  
  // If we have no users, they must be all self uses, just nuke the PHI.
  if (PHIUsers.empty())
    return ReplaceInstUsesWith(FirstPhi, UndefValue::get(FirstPhi.getType()));
  
  // If this phi node is transformable, create new PHIs for all the pieces
  // extracted out of it.  First, sort the users by their offset and size.
  array_pod_sort(PHIUsers.begin(), PHIUsers.end());
  
  DEBUG(errs() << "SLICING UP PHI: " << FirstPhi << '\n';
            for (unsigned i = 1, e = PHIsToSlice.size(); i != e; ++i)
              errs() << "AND USER PHI #" << i << ": " << *PHIsToSlice[i] <<'\n';
        );
  
  // PredValues - This is a temporary used when rewriting PHI nodes.  It is
  // hoisted out here to avoid construction/destruction thrashing.
  DenseMap<BasicBlock*, Value*> PredValues;
  
  // ExtractedVals - Each new PHI we introduce is saved here so we don't
  // introduce redundant PHIs.
  DenseMap<LoweredPHIRecord, PHINode*> ExtractedVals;
  
  for (unsigned UserI = 0, UserE = PHIUsers.size(); UserI != UserE; ++UserI) {
    unsigned PHIId = PHIUsers[UserI].PHIId;
    PHINode *PN = PHIsToSlice[PHIId];
    unsigned Offset = PHIUsers[UserI].Shift;
    const Type *Ty = PHIUsers[UserI].Inst->getType();
    
    PHINode *EltPHI;
    
    // If we've already lowered a user like this, reuse the previously lowered
    // value.
    if ((EltPHI = ExtractedVals[LoweredPHIRecord(PN, Offset, Ty)]) == 0) {
      
      // Otherwise, Create the new PHI node for this user.
      EltPHI = PHINode::Create(Ty, PN->getName()+".off"+Twine(Offset), PN);
      assert(EltPHI->getType() != PN->getType() &&
             "Truncate didn't shrink phi?");
    
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
        BasicBlock *Pred = PN->getIncomingBlock(i);
        Value *&PredVal = PredValues[Pred];
        
        // If we already have a value for this predecessor, reuse it.
        if (PredVal) {
          EltPHI->addIncoming(PredVal, Pred);
          continue;
        }

        // Handle the PHI self-reuse case.
        Value *InVal = PN->getIncomingValue(i);
        if (InVal == PN) {
          PredVal = EltPHI;
          EltPHI->addIncoming(PredVal, Pred);
          continue;
        }
        
        if (PHINode *InPHI = dyn_cast<PHINode>(PN)) {
          // If the incoming value was a PHI, and if it was one of the PHIs we
          // already rewrote it, just use the lowered value.
          if (Value *Res = ExtractedVals[LoweredPHIRecord(InPHI, Offset, Ty)]) {
            PredVal = Res;
            EltPHI->addIncoming(PredVal, Pred);
            continue;
          }
        }
        
        // Otherwise, do an extract in the predecessor.
        Builder->SetInsertPoint(Pred, Pred->getTerminator());
        Value *Res = InVal;
        if (Offset)
          Res = Builder->CreateLShr(Res, ConstantInt::get(InVal->getType(),
                                                          Offset), "extract");
        Res = Builder->CreateTrunc(Res, Ty, "extract.t");
        PredVal = Res;
        EltPHI->addIncoming(Res, Pred);
        
        // If the incoming value was a PHI, and if it was one of the PHIs we are
        // rewriting, we will ultimately delete the code we inserted.  This
        // means we need to revisit that PHI to make sure we extract out the
        // needed piece.
        if (PHINode *OldInVal = dyn_cast<PHINode>(PN->getIncomingValue(i)))
          if (PHIsInspected.count(OldInVal)) {
            unsigned RefPHIId = std::find(PHIsToSlice.begin(),PHIsToSlice.end(),
                                          OldInVal)-PHIsToSlice.begin();
            PHIUsers.push_back(PHIUsageRecord(RefPHIId, Offset, 
                                              cast<Instruction>(Res)));
            ++UserE;
          }
      }
      PredValues.clear();
      
      DEBUG(errs() << "  Made element PHI for offset " << Offset << ": "
                   << *EltPHI << '\n');
      ExtractedVals[LoweredPHIRecord(PN, Offset, Ty)] = EltPHI;
    }
    
    // Replace the use of this piece with the PHI node.
    ReplaceInstUsesWith(*PHIUsers[UserI].Inst, EltPHI);
  }
  
  // Replace all the remaining uses of the PHI nodes (self uses and the lshrs)
  // with undefs.
  Value *Undef = UndefValue::get(FirstPhi.getType());
  for (unsigned i = 1, e = PHIsToSlice.size(); i != e; ++i)
    ReplaceInstUsesWith(*PHIsToSlice[i], Undef);
  return ReplaceInstUsesWith(FirstPhi, Undef);
}

// PHINode simplification
//
Instruction *InstCombiner::visitPHINode(PHINode &PN) {
  // If LCSSA is around, don't mess with Phi nodes
  if (MustPreserveLCSSA) return 0;
  
  if (Value *V = PN.hasConstantValue())
    return ReplaceInstUsesWith(PN, V);

  // If all PHI operands are the same operation, pull them through the PHI,
  // reducing code size.
  if (isa<Instruction>(PN.getIncomingValue(0)) &&
      isa<Instruction>(PN.getIncomingValue(1)) &&
      cast<Instruction>(PN.getIncomingValue(0))->getOpcode() ==
      cast<Instruction>(PN.getIncomingValue(1))->getOpcode() &&
      // FIXME: The hasOneUse check will fail for PHIs that use the value more
      // than themselves more than once.
      PN.getIncomingValue(0)->hasOneUse())
    if (Instruction *Result = FoldPHIArgOpIntoPHI(PN))
      return Result;

  // If this is a trivial cycle in the PHI node graph, remove it.  Basically, if
  // this PHI only has a single use (a PHI), and if that PHI only has one use (a
  // PHI)... break the cycle.
  if (PN.hasOneUse()) {
    Instruction *PHIUser = cast<Instruction>(PN.use_back());
    if (PHINode *PU = dyn_cast<PHINode>(PHIUser)) {
      SmallPtrSet<PHINode*, 16> PotentiallyDeadPHIs;
      PotentiallyDeadPHIs.insert(&PN);
      if (DeadPHICycle(PU, PotentiallyDeadPHIs))
        return ReplaceInstUsesWith(PN, UndefValue::get(PN.getType()));
    }
   
    // If this phi has a single use, and if that use just computes a value for
    // the next iteration of a loop, delete the phi.  This occurs with unused
    // induction variables, e.g. "for (int j = 0; ; ++j);".  Detecting this
    // common case here is good because the only other things that catch this
    // are induction variable analysis (sometimes) and ADCE, which is only run
    // late.
    if (PHIUser->hasOneUse() &&
        (isa<BinaryOperator>(PHIUser) || isa<GetElementPtrInst>(PHIUser)) &&
        PHIUser->use_back() == &PN) {
      return ReplaceInstUsesWith(PN, UndefValue::get(PN.getType()));
    }
  }

  // We sometimes end up with phi cycles that non-obviously end up being the
  // same value, for example:
  //   z = some value; x = phi (y, z); y = phi (x, z)
  // where the phi nodes don't necessarily need to be in the same block.  Do a
  // quick check to see if the PHI node only contains a single non-phi value, if
  // so, scan to see if the phi cycle is actually equal to that value.
  {
    unsigned InValNo = 0, NumOperandVals = PN.getNumIncomingValues();
    // Scan for the first non-phi operand.
    while (InValNo != NumOperandVals && 
           isa<PHINode>(PN.getIncomingValue(InValNo)))
      ++InValNo;

    if (InValNo != NumOperandVals) {
      Value *NonPhiInVal = PN.getOperand(InValNo);
      
      // Scan the rest of the operands to see if there are any conflicts, if so
      // there is no need to recursively scan other phis.
      for (++InValNo; InValNo != NumOperandVals; ++InValNo) {
        Value *OpVal = PN.getIncomingValue(InValNo);
        if (OpVal != NonPhiInVal && !isa<PHINode>(OpVal))
          break;
      }
      
      // If we scanned over all operands, then we have one unique value plus
      // phi values.  Scan PHI nodes to see if they all merge in each other or
      // the value.
      if (InValNo == NumOperandVals) {
        SmallPtrSet<PHINode*, 16> ValueEqualPHIs;
        if (PHIsEqualValue(&PN, NonPhiInVal, ValueEqualPHIs))
          return ReplaceInstUsesWith(PN, NonPhiInVal);
      }
    }
  }

  // If there are multiple PHIs, sort their operands so that they all list
  // the blocks in the same order. This will help identical PHIs be eliminated
  // by other passes. Other passes shouldn't depend on this for correctness
  // however.
  PHINode *FirstPN = cast<PHINode>(PN.getParent()->begin());
  if (&PN != FirstPN)
    for (unsigned i = 0, e = FirstPN->getNumIncomingValues(); i != e; ++i) {
      BasicBlock *BBA = PN.getIncomingBlock(i);
      BasicBlock *BBB = FirstPN->getIncomingBlock(i);
      if (BBA != BBB) {
        Value *VA = PN.getIncomingValue(i);
        unsigned j = PN.getBasicBlockIndex(BBB);
        Value *VB = PN.getIncomingValue(j);
        PN.setIncomingBlock(i, BBB);
        PN.setIncomingValue(i, VB);
        PN.setIncomingBlock(j, BBA);
        PN.setIncomingValue(j, VA);
        // NOTE: Instcombine normally would want us to "return &PN" if we
        // modified any of the operands of an instruction.  However, since we
        // aren't adding or removing uses (just rearranging them) we don't do
        // this in this case.
      }
    }

  // If this is an integer PHI and we know that it has an illegal type, see if
  // it is only used by trunc or trunc(lshr) operations.  If so, we split the
  // PHI into the various pieces being extracted.  This sort of thing is
  // introduced when SROA promotes an aggregate to a single large integer type.
  if (isa<IntegerType>(PN.getType()) && TD &&
      !TD->isLegalInteger(PN.getType()->getPrimitiveSizeInBits()))
    if (Instruction *Res = SliceUpIllegalIntegerPHI(PN))
      return Res;
  
  return 0;
}

Instruction *InstCombiner::visitGetElementPtrInst(GetElementPtrInst &GEP) {
  SmallVector<Value*, 8> Ops(GEP.op_begin(), GEP.op_end());

  if (Value *V = SimplifyGEPInst(&Ops[0], Ops.size(), TD))
    return ReplaceInstUsesWith(GEP, V);

  Value *PtrOp = GEP.getOperand(0);

  if (isa<UndefValue>(GEP.getOperand(0)))
    return ReplaceInstUsesWith(GEP, UndefValue::get(GEP.getType()));

  // Eliminate unneeded casts for indices.
  if (TD) {
    bool MadeChange = false;
    unsigned PtrSize = TD->getPointerSizeInBits();
    
    gep_type_iterator GTI = gep_type_begin(GEP);
    for (User::op_iterator I = GEP.op_begin() + 1, E = GEP.op_end();
         I != E; ++I, ++GTI) {
      if (!isa<SequentialType>(*GTI)) continue;
      
      // If we are using a wider index than needed for this platform, shrink it
      // to what we need.  If narrower, sign-extend it to what we need.  This
      // explicit cast can make subsequent optimizations more obvious.
      unsigned OpBits = cast<IntegerType>((*I)->getType())->getBitWidth();
      if (OpBits == PtrSize)
        continue;
      
      *I = Builder->CreateIntCast(*I, TD->getIntPtrType(GEP.getContext()),true);
      MadeChange = true;
    }
    if (MadeChange) return &GEP;
  }

  // Combine Indices - If the source pointer to this getelementptr instruction
  // is a getelementptr instruction, combine the indices of the two
  // getelementptr instructions into a single instruction.
  //
  if (GEPOperator *Src = dyn_cast<GEPOperator>(PtrOp)) {
    // Note that if our source is a gep chain itself that we wait for that
    // chain to be resolved before we perform this transformation.  This
    // avoids us creating a TON of code in some cases.
    //
    if (GetElementPtrInst *SrcGEP =
          dyn_cast<GetElementPtrInst>(Src->getOperand(0)))
      if (SrcGEP->getNumOperands() == 2)
        return 0;   // Wait until our source is folded to completion.

    SmallVector<Value*, 8> Indices;

    // Find out whether the last index in the source GEP is a sequential idx.
    bool EndsWithSequential = false;
    for (gep_type_iterator I = gep_type_begin(*Src), E = gep_type_end(*Src);
         I != E; ++I)
      EndsWithSequential = !isa<StructType>(*I);

    // Can we combine the two pointer arithmetics offsets?
    if (EndsWithSequential) {
      // Replace: gep (gep %P, long B), long A, ...
      // With:    T = long A+B; gep %P, T, ...
      //
      Value *Sum;
      Value *SO1 = Src->getOperand(Src->getNumOperands()-1);
      Value *GO1 = GEP.getOperand(1);
      if (SO1 == Constant::getNullValue(SO1->getType())) {
        Sum = GO1;
      } else if (GO1 == Constant::getNullValue(GO1->getType())) {
        Sum = SO1;
      } else {
        // If they aren't the same type, then the input hasn't been processed
        // by the loop above yet (which canonicalizes sequential index types to
        // intptr_t).  Just avoid transforming this until the input has been
        // normalized.
        if (SO1->getType() != GO1->getType())
          return 0;
        Sum = Builder->CreateAdd(SO1, GO1, PtrOp->getName()+".sum");
      }

      // Update the GEP in place if possible.
      if (Src->getNumOperands() == 2) {
        GEP.setOperand(0, Src->getOperand(0));
        GEP.setOperand(1, Sum);
        return &GEP;
      }
      Indices.append(Src->op_begin()+1, Src->op_end()-1);
      Indices.push_back(Sum);
      Indices.append(GEP.op_begin()+2, GEP.op_end());
    } else if (isa<Constant>(*GEP.idx_begin()) &&
               cast<Constant>(*GEP.idx_begin())->isNullValue() &&
               Src->getNumOperands() != 1) {
      // Otherwise we can do the fold if the first index of the GEP is a zero
      Indices.append(Src->op_begin()+1, Src->op_end());
      Indices.append(GEP.idx_begin()+1, GEP.idx_end());
    }

    if (!Indices.empty())
      return (cast<GEPOperator>(&GEP)->isInBounds() &&
              Src->isInBounds()) ?
        GetElementPtrInst::CreateInBounds(Src->getOperand(0), Indices.begin(),
                                          Indices.end(), GEP.getName()) :
        GetElementPtrInst::Create(Src->getOperand(0), Indices.begin(),
                                  Indices.end(), GEP.getName());
  }
  
  // Handle gep(bitcast x) and gep(gep x, 0, 0, 0).
  if (Value *X = getBitCastOperand(PtrOp)) {
    assert(isa<PointerType>(X->getType()) && "Must be cast from pointer");

    // If the input bitcast is actually "bitcast(bitcast(x))", then we don't 
    // want to change the gep until the bitcasts are eliminated.
    if (getBitCastOperand(X)) {
      Worklist.AddValue(PtrOp);
      return 0;
    }
    
    bool HasZeroPointerIndex = false;
    if (ConstantInt *C = dyn_cast<ConstantInt>(GEP.getOperand(1)))
      HasZeroPointerIndex = C->isZero();
    
    // Transform: GEP (bitcast [10 x i8]* X to [0 x i8]*), i32 0, ...
    // into     : GEP [10 x i8]* X, i32 0, ...
    //
    // Likewise, transform: GEP (bitcast i8* X to [0 x i8]*), i32 0, ...
    //           into     : GEP i8* X, ...
    // 
    // This occurs when the program declares an array extern like "int X[];"
    if (HasZeroPointerIndex) {
      const PointerType *CPTy = cast<PointerType>(PtrOp->getType());
      const PointerType *XTy = cast<PointerType>(X->getType());
      if (const ArrayType *CATy =
          dyn_cast<ArrayType>(CPTy->getElementType())) {
        // GEP (bitcast i8* X to [0 x i8]*), i32 0, ... ?
        if (CATy->getElementType() == XTy->getElementType()) {
          // -> GEP i8* X, ...
          SmallVector<Value*, 8> Indices(GEP.idx_begin()+1, GEP.idx_end());
          return cast<GEPOperator>(&GEP)->isInBounds() ?
            GetElementPtrInst::CreateInBounds(X, Indices.begin(), Indices.end(),
                                              GEP.getName()) :
            GetElementPtrInst::Create(X, Indices.begin(), Indices.end(),
                                      GEP.getName());
        }
        
        if (const ArrayType *XATy = dyn_cast<ArrayType>(XTy->getElementType())){
          // GEP (bitcast [10 x i8]* X to [0 x i8]*), i32 0, ... ?
          if (CATy->getElementType() == XATy->getElementType()) {
            // -> GEP [10 x i8]* X, i32 0, ...
            // At this point, we know that the cast source type is a pointer
            // to an array of the same type as the destination pointer
            // array.  Because the array type is never stepped over (there
            // is a leading zero) we can fold the cast into this GEP.
            GEP.setOperand(0, X);
            return &GEP;
          }
        }
      }
    } else if (GEP.getNumOperands() == 2) {
      // Transform things like:
      // %t = getelementptr i32* bitcast ([2 x i32]* %str to i32*), i32 %V
      // into:  %t1 = getelementptr [2 x i32]* %str, i32 0, i32 %V; bitcast
      const Type *SrcElTy = cast<PointerType>(X->getType())->getElementType();
      const Type *ResElTy=cast<PointerType>(PtrOp->getType())->getElementType();
      if (TD && isa<ArrayType>(SrcElTy) &&
          TD->getTypeAllocSize(cast<ArrayType>(SrcElTy)->getElementType()) ==
          TD->getTypeAllocSize(ResElTy)) {
        Value *Idx[2];
        Idx[0] = Constant::getNullValue(Type::getInt32Ty(GEP.getContext()));
        Idx[1] = GEP.getOperand(1);
        Value *NewGEP = cast<GEPOperator>(&GEP)->isInBounds() ?
          Builder->CreateInBoundsGEP(X, Idx, Idx + 2, GEP.getName()) :
          Builder->CreateGEP(X, Idx, Idx + 2, GEP.getName());
        // V and GEP are both pointer types --> BitCast
        return new BitCastInst(NewGEP, GEP.getType());
      }
      
      // Transform things like:
      // getelementptr i8* bitcast ([100 x double]* X to i8*), i32 %tmp
      //   (where tmp = 8*tmp2) into:
      // getelementptr [100 x double]* %arr, i32 0, i32 %tmp2; bitcast
      
      if (TD && isa<ArrayType>(SrcElTy) &&
          ResElTy == Type::getInt8Ty(GEP.getContext())) {
        uint64_t ArrayEltSize =
            TD->getTypeAllocSize(cast<ArrayType>(SrcElTy)->getElementType());
        
        // Check to see if "tmp" is a scale by a multiple of ArrayEltSize.  We
        // allow either a mul, shift, or constant here.
        Value *NewIdx = 0;
        ConstantInt *Scale = 0;
        if (ArrayEltSize == 1) {
          NewIdx = GEP.getOperand(1);
          Scale = ConstantInt::get(cast<IntegerType>(NewIdx->getType()), 1);
        } else if (ConstantInt *CI = dyn_cast<ConstantInt>(GEP.getOperand(1))) {
          NewIdx = ConstantInt::get(CI->getType(), 1);
          Scale = CI;
        } else if (Instruction *Inst =dyn_cast<Instruction>(GEP.getOperand(1))){
          if (Inst->getOpcode() == Instruction::Shl &&
              isa<ConstantInt>(Inst->getOperand(1))) {
            ConstantInt *ShAmt = cast<ConstantInt>(Inst->getOperand(1));
            uint32_t ShAmtVal = ShAmt->getLimitedValue(64);
            Scale = ConstantInt::get(cast<IntegerType>(Inst->getType()),
                                     1ULL << ShAmtVal);
            NewIdx = Inst->getOperand(0);
          } else if (Inst->getOpcode() == Instruction::Mul &&
                     isa<ConstantInt>(Inst->getOperand(1))) {
            Scale = cast<ConstantInt>(Inst->getOperand(1));
            NewIdx = Inst->getOperand(0);
          }
        }
        
        // If the index will be to exactly the right offset with the scale taken
        // out, perform the transformation. Note, we don't know whether Scale is
        // signed or not. We'll use unsigned version of division/modulo
        // operation after making sure Scale doesn't have the sign bit set.
        if (ArrayEltSize && Scale && Scale->getSExtValue() >= 0LL &&
            Scale->getZExtValue() % ArrayEltSize == 0) {
          Scale = ConstantInt::get(Scale->getType(),
                                   Scale->getZExtValue() / ArrayEltSize);
          if (Scale->getZExtValue() != 1) {
            Constant *C = ConstantExpr::getIntegerCast(Scale, NewIdx->getType(),
                                                       false /*ZExt*/);
            NewIdx = Builder->CreateMul(NewIdx, C, "idxscale");
          }

          // Insert the new GEP instruction.
          Value *Idx[2];
          Idx[0] = Constant::getNullValue(Type::getInt32Ty(GEP.getContext()));
          Idx[1] = NewIdx;
          Value *NewGEP = cast<GEPOperator>(&GEP)->isInBounds() ?
            Builder->CreateInBoundsGEP(X, Idx, Idx + 2, GEP.getName()) :
            Builder->CreateGEP(X, Idx, Idx + 2, GEP.getName());
          // The NewGEP must be pointer typed, so must the old one -> BitCast
          return new BitCastInst(NewGEP, GEP.getType());
        }
      }
    }
  }
  
  /// See if we can simplify:
  ///   X = bitcast A* to B*
  ///   Y = gep X, <...constant indices...>
  /// into a gep of the original struct.  This is important for SROA and alias
  /// analysis of unions.  If "A" is also a bitcast, wait for A/X to be merged.
  if (BitCastInst *BCI = dyn_cast<BitCastInst>(PtrOp)) {
    if (TD &&
        !isa<BitCastInst>(BCI->getOperand(0)) && GEP.hasAllConstantIndices()) {
      // Determine how much the GEP moves the pointer.  We are guaranteed to get
      // a constant back from EmitGEPOffset.
      ConstantInt *OffsetV = cast<ConstantInt>(EmitGEPOffset(&GEP));
      int64_t Offset = OffsetV->getSExtValue();
      
      // If this GEP instruction doesn't move the pointer, just replace the GEP
      // with a bitcast of the real input to the dest type.
      if (Offset == 0) {
        // If the bitcast is of an allocation, and the allocation will be
        // converted to match the type of the cast, don't touch this.
        if (isa<AllocaInst>(BCI->getOperand(0)) ||
            isMalloc(BCI->getOperand(0))) {
          // See if the bitcast simplifies, if so, don't nuke this GEP yet.
          if (Instruction *I = visitBitCast(*BCI)) {
            if (I != BCI) {
              I->takeName(BCI);
              BCI->getParent()->getInstList().insert(BCI, I);
              ReplaceInstUsesWith(*BCI, I);
            }
            return &GEP;
          }
        }
        return new BitCastInst(BCI->getOperand(0), GEP.getType());
      }
      
      // Otherwise, if the offset is non-zero, we need to find out if there is a
      // field at Offset in 'A's type.  If so, we can pull the cast through the
      // GEP.
      SmallVector<Value*, 8> NewIndices;
      const Type *InTy =
        cast<PointerType>(BCI->getOperand(0)->getType())->getElementType();
      if (FindElementAtOffset(InTy, Offset, NewIndices)) {
        Value *NGEP = cast<GEPOperator>(&GEP)->isInBounds() ?
          Builder->CreateInBoundsGEP(BCI->getOperand(0), NewIndices.begin(),
                                     NewIndices.end()) :
          Builder->CreateGEP(BCI->getOperand(0), NewIndices.begin(),
                             NewIndices.end());
        
        if (NGEP->getType() == GEP.getType())
          return ReplaceInstUsesWith(GEP, NGEP);
        NGEP->takeName(&GEP);
        return new BitCastInst(NGEP, GEP.getType());
      }
    }
  }    
    
  return 0;
}

Instruction *InstCombiner::visitAllocaInst(AllocaInst &AI) {
  // Convert: alloca Ty, C - where C is a constant != 1 into: alloca [C x Ty], 1
  if (AI.isArrayAllocation()) {  // Check C != 1
    if (const ConstantInt *C = dyn_cast<ConstantInt>(AI.getArraySize())) {
      const Type *NewTy = 
        ArrayType::get(AI.getAllocatedType(), C->getZExtValue());
      assert(isa<AllocaInst>(AI) && "Unknown type of allocation inst!");
      AllocaInst *New = Builder->CreateAlloca(NewTy, 0, AI.getName());
      New->setAlignment(AI.getAlignment());

      // Scan to the end of the allocation instructions, to skip over a block of
      // allocas if possible...also skip interleaved debug info
      //
      BasicBlock::iterator It = New;
      while (isa<AllocaInst>(*It) || isa<DbgInfoIntrinsic>(*It)) ++It;

      // Now that I is pointing to the first non-allocation-inst in the block,
      // insert our getelementptr instruction...
      //
      Value *NullIdx =Constant::getNullValue(Type::getInt32Ty(AI.getContext()));
      Value *Idx[2];
      Idx[0] = NullIdx;
      Idx[1] = NullIdx;
      Value *V = GetElementPtrInst::CreateInBounds(New, Idx, Idx + 2,
                                                   New->getName()+".sub", It);

      // Now make everything use the getelementptr instead of the original
      // allocation.
      return ReplaceInstUsesWith(AI, V);
    } else if (isa<UndefValue>(AI.getArraySize())) {
      return ReplaceInstUsesWith(AI, Constant::getNullValue(AI.getType()));
    }
  }

  if (TD && isa<AllocaInst>(AI) && AI.getAllocatedType()->isSized()) {
    // If alloca'ing a zero byte object, replace the alloca with a null pointer.
    // Note that we only do this for alloca's, because malloc should allocate
    // and return a unique pointer, even for a zero byte allocation.
    if (TD->getTypeAllocSize(AI.getAllocatedType()) == 0)
      return ReplaceInstUsesWith(AI, Constant::getNullValue(AI.getType()));

    // If the alignment is 0 (unspecified), assign it the preferred alignment.
    if (AI.getAlignment() == 0)
      AI.setAlignment(TD->getPrefTypeAlignment(AI.getAllocatedType()));
  }

  return 0;
}

Instruction *InstCombiner::visitFree(Instruction &FI) {
  Value *Op = FI.getOperand(1);

  // free undef -> unreachable.
  if (isa<UndefValue>(Op)) {
    // Insert a new store to null because we cannot modify the CFG here.
    new StoreInst(ConstantInt::getTrue(FI.getContext()),
           UndefValue::get(Type::getInt1PtrTy(FI.getContext())), &FI);
    return EraseInstFromFunction(FI);
  }
  
  // If we have 'free null' delete the instruction.  This can happen in stl code
  // when lots of inlining happens.
  if (isa<ConstantPointerNull>(Op))
    return EraseInstFromFunction(FI);

  // If we have a malloc call whose only use is a free call, delete both.
  if (isMalloc(Op)) {
    if (CallInst* CI = extractMallocCallFromBitCast(Op)) {
      if (Op->hasOneUse() && CI->hasOneUse()) {
        EraseInstFromFunction(FI);
        EraseInstFromFunction(*CI);
        return EraseInstFromFunction(*cast<Instruction>(Op));
      }
    } else {
      // Op is a call to malloc
      if (Op->hasOneUse()) {
        EraseInstFromFunction(FI);
        return EraseInstFromFunction(*cast<Instruction>(Op));
      }
    }
  }

  return 0;
}

/// InstCombineLoadCast - Fold 'load (cast P)' -> cast (load P)' when possible.
static Instruction *InstCombineLoadCast(InstCombiner &IC, LoadInst &LI,
                                        const TargetData *TD) {
  User *CI = cast<User>(LI.getOperand(0));
  Value *CastOp = CI->getOperand(0);

  const PointerType *DestTy = cast<PointerType>(CI->getType());
  const Type *DestPTy = DestTy->getElementType();
  if (const PointerType *SrcTy = dyn_cast<PointerType>(CastOp->getType())) {

    // If the address spaces don't match, don't eliminate the cast.
    if (DestTy->getAddressSpace() != SrcTy->getAddressSpace())
      return 0;

    const Type *SrcPTy = SrcTy->getElementType();

    if (DestPTy->isInteger() || isa<PointerType>(DestPTy) || 
         isa<VectorType>(DestPTy)) {
      // If the source is an array, the code below will not succeed.  Check to
      // see if a trivial 'gep P, 0, 0' will help matters.  Only do this for
      // constants.
      if (const ArrayType *ASrcTy = dyn_cast<ArrayType>(SrcPTy))
        if (Constant *CSrc = dyn_cast<Constant>(CastOp))
          if (ASrcTy->getNumElements() != 0) {
            Value *Idxs[2];
            Idxs[0] = Constant::getNullValue(Type::getInt32Ty(LI.getContext()));
            Idxs[1] = Idxs[0];
            CastOp = ConstantExpr::getGetElementPtr(CSrc, Idxs, 2);
            SrcTy = cast<PointerType>(CastOp->getType());
            SrcPTy = SrcTy->getElementType();
          }

      if (IC.getTargetData() &&
          (SrcPTy->isInteger() || isa<PointerType>(SrcPTy) || 
            isa<VectorType>(SrcPTy)) &&
          // Do not allow turning this into a load of an integer, which is then
          // casted to a pointer, this pessimizes pointer analysis a lot.
          (isa<PointerType>(SrcPTy) == isa<PointerType>(LI.getType())) &&
          IC.getTargetData()->getTypeSizeInBits(SrcPTy) ==
               IC.getTargetData()->getTypeSizeInBits(DestPTy)) {

        // Okay, we are casting from one integer or pointer type to another of
        // the same size.  Instead of casting the pointer before the load, cast
        // the result of the loaded value.
        Value *NewLoad = 
          IC.Builder->CreateLoad(CastOp, LI.isVolatile(), CI->getName());
        // Now cast the result of the load.
        return new BitCastInst(NewLoad, LI.getType());
      }
    }
  }
  return 0;
}

Instruction *InstCombiner::visitLoadInst(LoadInst &LI) {
  Value *Op = LI.getOperand(0);

  // Attempt to improve the alignment.
  if (TD) {
    unsigned KnownAlign =
      GetOrEnforceKnownAlignment(Op, TD->getPrefTypeAlignment(LI.getType()));
    if (KnownAlign >
        (LI.getAlignment() == 0 ? TD->getABITypeAlignment(LI.getType()) :
                                  LI.getAlignment()))
      LI.setAlignment(KnownAlign);
  }

  // load (cast X) --> cast (load X) iff safe.
  if (isa<CastInst>(Op))
    if (Instruction *Res = InstCombineLoadCast(*this, LI, TD))
      return Res;

  // None of the following transforms are legal for volatile loads.
  if (LI.isVolatile()) return 0;
  
  // Do really simple store-to-load forwarding and load CSE, to catch cases
  // where there are several consequtive memory accesses to the same location,
  // separated by a few arithmetic operations.
  BasicBlock::iterator BBI = &LI;
  if (Value *AvailableVal = FindAvailableLoadedValue(Op, LI.getParent(), BBI,6))
    return ReplaceInstUsesWith(LI, AvailableVal);

  // load(gep null, ...) -> unreachable
  if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(Op)) {
    const Value *GEPI0 = GEPI->getOperand(0);
    // TODO: Consider a target hook for valid address spaces for this xform.
    if (isa<ConstantPointerNull>(GEPI0) && GEPI->getPointerAddressSpace() == 0){
      // Insert a new store to null instruction before the load to indicate
      // that this code is not reachable.  We do this instead of inserting
      // an unreachable instruction directly because we cannot modify the
      // CFG.
      new StoreInst(UndefValue::get(LI.getType()),
                    Constant::getNullValue(Op->getType()), &LI);
      return ReplaceInstUsesWith(LI, UndefValue::get(LI.getType()));
    }
  } 

  // load null/undef -> unreachable
  // TODO: Consider a target hook for valid address spaces for this xform.
  if (isa<UndefValue>(Op) ||
      (isa<ConstantPointerNull>(Op) && LI.getPointerAddressSpace() == 0)) {
    // Insert a new store to null instruction before the load to indicate that
    // this code is not reachable.  We do this instead of inserting an
    // unreachable instruction directly because we cannot modify the CFG.
    new StoreInst(UndefValue::get(LI.getType()),
                  Constant::getNullValue(Op->getType()), &LI);
    return ReplaceInstUsesWith(LI, UndefValue::get(LI.getType()));
  }

  // Instcombine load (constantexpr_cast global) -> cast (load global)
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Op))
    if (CE->isCast())
      if (Instruction *Res = InstCombineLoadCast(*this, LI, TD))
        return Res;
  
  if (Op->hasOneUse()) {
    // Change select and PHI nodes to select values instead of addresses: this
    // helps alias analysis out a lot, allows many others simplifications, and
    // exposes redundancy in the code.
    //
    // Note that we cannot do the transformation unless we know that the
    // introduced loads cannot trap!  Something like this is valid as long as
    // the condition is always false: load (select bool %C, int* null, int* %G),
    // but it would not be valid if we transformed it to load from null
    // unconditionally.
    //
    if (SelectInst *SI = dyn_cast<SelectInst>(Op)) {
      // load (select (Cond, &V1, &V2))  --> select(Cond, load &V1, load &V2).
      if (isSafeToLoadUnconditionally(SI->getOperand(1), SI) &&
          isSafeToLoadUnconditionally(SI->getOperand(2), SI)) {
        Value *V1 = Builder->CreateLoad(SI->getOperand(1),
                                        SI->getOperand(1)->getName()+".val");
        Value *V2 = Builder->CreateLoad(SI->getOperand(2),
                                        SI->getOperand(2)->getName()+".val");
        return SelectInst::Create(SI->getCondition(), V1, V2);
      }

      // load (select (cond, null, P)) -> load P
      if (Constant *C = dyn_cast<Constant>(SI->getOperand(1)))
        if (C->isNullValue()) {
          LI.setOperand(0, SI->getOperand(2));
          return &LI;
        }

      // load (select (cond, P, null)) -> load P
      if (Constant *C = dyn_cast<Constant>(SI->getOperand(2)))
        if (C->isNullValue()) {
          LI.setOperand(0, SI->getOperand(1));
          return &LI;
        }
    }
  }
  return 0;
}

/// InstCombineStoreToCast - Fold store V, (cast P) -> store (cast V), P
/// when possible.  This makes it generally easy to do alias analysis and/or
/// SROA/mem2reg of the memory object.
static Instruction *InstCombineStoreToCast(InstCombiner &IC, StoreInst &SI) {
  User *CI = cast<User>(SI.getOperand(1));
  Value *CastOp = CI->getOperand(0);

  const Type *DestPTy = cast<PointerType>(CI->getType())->getElementType();
  const PointerType *SrcTy = dyn_cast<PointerType>(CastOp->getType());
  if (SrcTy == 0) return 0;
  
  const Type *SrcPTy = SrcTy->getElementType();

  if (!DestPTy->isInteger() && !isa<PointerType>(DestPTy))
    return 0;
  
  /// NewGEPIndices - If SrcPTy is an aggregate type, we can emit a "noop gep"
  /// to its first element.  This allows us to handle things like:
  ///   store i32 xxx, (bitcast {foo*, float}* %P to i32*)
  /// on 32-bit hosts.
  SmallVector<Value*, 4> NewGEPIndices;
  
  // If the source is an array, the code below will not succeed.  Check to
  // see if a trivial 'gep P, 0, 0' will help matters.  Only do this for
  // constants.
  if (isa<ArrayType>(SrcPTy) || isa<StructType>(SrcPTy)) {
    // Index through pointer.
    Constant *Zero = Constant::getNullValue(Type::getInt32Ty(SI.getContext()));
    NewGEPIndices.push_back(Zero);
    
    while (1) {
      if (const StructType *STy = dyn_cast<StructType>(SrcPTy)) {
        if (!STy->getNumElements()) /* Struct can be empty {} */
          break;
        NewGEPIndices.push_back(Zero);
        SrcPTy = STy->getElementType(0);
      } else if (const ArrayType *ATy = dyn_cast<ArrayType>(SrcPTy)) {
        NewGEPIndices.push_back(Zero);
        SrcPTy = ATy->getElementType();
      } else {
        break;
      }
    }
    
    SrcTy = PointerType::get(SrcPTy, SrcTy->getAddressSpace());
  }

  if (!SrcPTy->isInteger() && !isa<PointerType>(SrcPTy))
    return 0;
  
  // If the pointers point into different address spaces or if they point to
  // values with different sizes, we can't do the transformation.
  if (!IC.getTargetData() ||
      SrcTy->getAddressSpace() != 
        cast<PointerType>(CI->getType())->getAddressSpace() ||
      IC.getTargetData()->getTypeSizeInBits(SrcPTy) !=
      IC.getTargetData()->getTypeSizeInBits(DestPTy))
    return 0;

  // Okay, we are casting from one integer or pointer type to another of
  // the same size.  Instead of casting the pointer before 
  // the store, cast the value to be stored.
  Value *NewCast;
  Value *SIOp0 = SI.getOperand(0);
  Instruction::CastOps opcode = Instruction::BitCast;
  const Type* CastSrcTy = SIOp0->getType();
  const Type* CastDstTy = SrcPTy;
  if (isa<PointerType>(CastDstTy)) {
    if (CastSrcTy->isInteger())
      opcode = Instruction::IntToPtr;
  } else if (isa<IntegerType>(CastDstTy)) {
    if (isa<PointerType>(SIOp0->getType()))
      opcode = Instruction::PtrToInt;
  }
  
  // SIOp0 is a pointer to aggregate and this is a store to the first field,
  // emit a GEP to index into its first field.
  if (!NewGEPIndices.empty())
    CastOp = IC.Builder->CreateInBoundsGEP(CastOp, NewGEPIndices.begin(),
                                           NewGEPIndices.end());
  
  NewCast = IC.Builder->CreateCast(opcode, SIOp0, CastDstTy,
                                   SIOp0->getName()+".c");
  return new StoreInst(NewCast, CastOp);
}

/// equivalentAddressValues - Test if A and B will obviously have the same
/// value. This includes recognizing that %t0 and %t1 will have the same
/// value in code like this:
///   %t0 = getelementptr \@a, 0, 3
///   store i32 0, i32* %t0
///   %t1 = getelementptr \@a, 0, 3
///   %t2 = load i32* %t1
///
static bool equivalentAddressValues(Value *A, Value *B) {
  // Test if the values are trivially equivalent.
  if (A == B) return true;
  
  // Test if the values come form identical arithmetic instructions.
  // This uses isIdenticalToWhenDefined instead of isIdenticalTo because
  // its only used to compare two uses within the same basic block, which
  // means that they'll always either have the same value or one of them
  // will have an undefined value.
  if (isa<BinaryOperator>(A) ||
      isa<CastInst>(A) ||
      isa<PHINode>(A) ||
      isa<GetElementPtrInst>(A))
    if (Instruction *BI = dyn_cast<Instruction>(B))
      if (cast<Instruction>(A)->isIdenticalToWhenDefined(BI))
        return true;
  
  // Otherwise they may not be equivalent.
  return false;
}

// If this instruction has two uses, one of which is a llvm.dbg.declare,
// return the llvm.dbg.declare.
DbgDeclareInst *InstCombiner::hasOneUsePlusDeclare(Value *V) {
  if (!V->hasNUses(2))
    return 0;
  for (Value::use_iterator UI = V->use_begin(), E = V->use_end();
       UI != E; ++UI) {
    if (DbgDeclareInst *DI = dyn_cast<DbgDeclareInst>(UI))
      return DI;
    if (isa<BitCastInst>(UI) && UI->hasOneUse()) {
      if (DbgDeclareInst *DI = dyn_cast<DbgDeclareInst>(UI->use_begin()))
        return DI;
      }
  }
  return 0;
}

Instruction *InstCombiner::visitStoreInst(StoreInst &SI) {
  Value *Val = SI.getOperand(0);
  Value *Ptr = SI.getOperand(1);

  // If the RHS is an alloca with a single use, zapify the store, making the
  // alloca dead.
  // If the RHS is an alloca with a two uses, the other one being a 
  // llvm.dbg.declare, zapify the store and the declare, making the
  // alloca dead.  We must do this to prevent declare's from affecting
  // codegen.
  if (!SI.isVolatile()) {
    if (Ptr->hasOneUse()) {
      if (isa<AllocaInst>(Ptr)) {
        EraseInstFromFunction(SI);
        ++NumCombined;
        return 0;
      }
      if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
        if (isa<AllocaInst>(GEP->getOperand(0))) {
          if (GEP->getOperand(0)->hasOneUse()) {
            EraseInstFromFunction(SI);
            ++NumCombined;
            return 0;
          }
          if (DbgDeclareInst *DI = hasOneUsePlusDeclare(GEP->getOperand(0))) {
            EraseInstFromFunction(*DI);
            EraseInstFromFunction(SI);
            ++NumCombined;
            return 0;
          }
        }
      }
    }
    if (DbgDeclareInst *DI = hasOneUsePlusDeclare(Ptr)) {
      EraseInstFromFunction(*DI);
      EraseInstFromFunction(SI);
      ++NumCombined;
      return 0;
    }
  }

  // Attempt to improve the alignment.
  if (TD) {
    unsigned KnownAlign =
      GetOrEnforceKnownAlignment(Ptr, TD->getPrefTypeAlignment(Val->getType()));
    if (KnownAlign >
        (SI.getAlignment() == 0 ? TD->getABITypeAlignment(Val->getType()) :
                                  SI.getAlignment()))
      SI.setAlignment(KnownAlign);
  }

  // Do really simple DSE, to catch cases where there are several consecutive
  // stores to the same location, separated by a few arithmetic operations. This
  // situation often occurs with bitfield accesses.
  BasicBlock::iterator BBI = &SI;
  for (unsigned ScanInsts = 6; BBI != SI.getParent()->begin() && ScanInsts;
       --ScanInsts) {
    --BBI;
    // Don't count debug info directives, lest they affect codegen,
    // and we skip pointer-to-pointer bitcasts, which are NOPs.
    // It is necessary for correctness to skip those that feed into a
    // llvm.dbg.declare, as these are not present when debugging is off.
    if (isa<DbgInfoIntrinsic>(BBI) ||
        (isa<BitCastInst>(BBI) && isa<PointerType>(BBI->getType()))) {
      ScanInsts++;
      continue;
    }    
    
    if (StoreInst *PrevSI = dyn_cast<StoreInst>(BBI)) {
      // Prev store isn't volatile, and stores to the same location?
      if (!PrevSI->isVolatile() &&equivalentAddressValues(PrevSI->getOperand(1),
                                                          SI.getOperand(1))) {
        ++NumDeadStore;
        ++BBI;
        EraseInstFromFunction(*PrevSI);
        continue;
      }
      break;
    }
    
    // If this is a load, we have to stop.  However, if the loaded value is from
    // the pointer we're loading and is producing the pointer we're storing,
    // then *this* store is dead (X = load P; store X -> P).
    if (LoadInst *LI = dyn_cast<LoadInst>(BBI)) {
      if (LI == Val && equivalentAddressValues(LI->getOperand(0), Ptr) &&
          !SI.isVolatile()) {
        EraseInstFromFunction(SI);
        ++NumCombined;
        return 0;
      }
      // Otherwise, this is a load from some other location.  Stores before it
      // may not be dead.
      break;
    }
    
    // Don't skip over loads or things that can modify memory.
    if (BBI->mayWriteToMemory() || BBI->mayReadFromMemory())
      break;
  }
  
  
  if (SI.isVolatile()) return 0;  // Don't hack volatile stores.

  // store X, null    -> turns into 'unreachable' in SimplifyCFG
  if (isa<ConstantPointerNull>(Ptr) && SI.getPointerAddressSpace() == 0) {
    if (!isa<UndefValue>(Val)) {
      SI.setOperand(0, UndefValue::get(Val->getType()));
      if (Instruction *U = dyn_cast<Instruction>(Val))
        Worklist.Add(U);  // Dropped a use.
      ++NumCombined;
    }
    return 0;  // Do not modify these!
  }

  // store undef, Ptr -> noop
  if (isa<UndefValue>(Val)) {
    EraseInstFromFunction(SI);
    ++NumCombined;
    return 0;
  }

  // If the pointer destination is a cast, see if we can fold the cast into the
  // source instead.
  if (isa<CastInst>(Ptr))
    if (Instruction *Res = InstCombineStoreToCast(*this, SI))
      return Res;
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Ptr))
    if (CE->isCast())
      if (Instruction *Res = InstCombineStoreToCast(*this, SI))
        return Res;

  
  // If this store is the last instruction in the basic block (possibly
  // excepting debug info instructions and the pointer bitcasts that feed
  // into them), and if the block ends with an unconditional branch, try
  // to move it to the successor block.
  BBI = &SI; 
  do {
    ++BBI;
  } while (isa<DbgInfoIntrinsic>(BBI) ||
           (isa<BitCastInst>(BBI) && isa<PointerType>(BBI->getType())));
  if (BranchInst *BI = dyn_cast<BranchInst>(BBI))
    if (BI->isUnconditional())
      if (SimplifyStoreAtEndOfBlock(SI))
        return 0;  // xform done!
  
  return 0;
}

/// SimplifyStoreAtEndOfBlock - Turn things like:
///   if () { *P = v1; } else { *P = v2 }
/// into a phi node with a store in the successor.
///
/// Simplify things like:
///   *P = v1; if () { *P = v2; }
/// into a phi node with a store in the successor.
///
bool InstCombiner::SimplifyStoreAtEndOfBlock(StoreInst &SI) {
  BasicBlock *StoreBB = SI.getParent();
  
  // Check to see if the successor block has exactly two incoming edges.  If
  // so, see if the other predecessor contains a store to the same location.
  // if so, insert a PHI node (if needed) and move the stores down.
  BasicBlock *DestBB = StoreBB->getTerminator()->getSuccessor(0);
  
  // Determine whether Dest has exactly two predecessors and, if so, compute
  // the other predecessor.
  pred_iterator PI = pred_begin(DestBB);
  BasicBlock *OtherBB = 0;
  if (*PI != StoreBB)
    OtherBB = *PI;
  ++PI;
  if (PI == pred_end(DestBB))
    return false;
  
  if (*PI != StoreBB) {
    if (OtherBB)
      return false;
    OtherBB = *PI;
  }
  if (++PI != pred_end(DestBB))
    return false;

  // Bail out if all the relevant blocks aren't distinct (this can happen,
  // for example, if SI is in an infinite loop)
  if (StoreBB == DestBB || OtherBB == DestBB)
    return false;

  // Verify that the other block ends in a branch and is not otherwise empty.
  BasicBlock::iterator BBI = OtherBB->getTerminator();
  BranchInst *OtherBr = dyn_cast<BranchInst>(BBI);
  if (!OtherBr || BBI == OtherBB->begin())
    return false;
  
  // If the other block ends in an unconditional branch, check for the 'if then
  // else' case.  there is an instruction before the branch.
  StoreInst *OtherStore = 0;
  if (OtherBr->isUnconditional()) {
    --BBI;
    // Skip over debugging info.
    while (isa<DbgInfoIntrinsic>(BBI) ||
           (isa<BitCastInst>(BBI) && isa<PointerType>(BBI->getType()))) {
      if (BBI==OtherBB->begin())
        return false;
      --BBI;
    }
    // If this isn't a store, isn't a store to the same location, or if the
    // alignments differ, bail out.
    OtherStore = dyn_cast<StoreInst>(BBI);
    if (!OtherStore || OtherStore->getOperand(1) != SI.getOperand(1) ||
        OtherStore->getAlignment() != SI.getAlignment())
      return false;
  } else {
    // Otherwise, the other block ended with a conditional branch. If one of the
    // destinations is StoreBB, then we have the if/then case.
    if (OtherBr->getSuccessor(0) != StoreBB && 
        OtherBr->getSuccessor(1) != StoreBB)
      return false;
    
    // Okay, we know that OtherBr now goes to Dest and StoreBB, so this is an
    // if/then triangle.  See if there is a store to the same ptr as SI that
    // lives in OtherBB.
    for (;; --BBI) {
      // Check to see if we find the matching store.
      if ((OtherStore = dyn_cast<StoreInst>(BBI))) {
        if (OtherStore->getOperand(1) != SI.getOperand(1) ||
            OtherStore->getAlignment() != SI.getAlignment())
          return false;
        break;
      }
      // If we find something that may be using or overwriting the stored
      // value, or if we run out of instructions, we can't do the xform.
      if (BBI->mayReadFromMemory() || BBI->mayWriteToMemory() ||
          BBI == OtherBB->begin())
        return false;
    }
    
    // In order to eliminate the store in OtherBr, we have to
    // make sure nothing reads or overwrites the stored value in
    // StoreBB.
    for (BasicBlock::iterator I = StoreBB->begin(); &*I != &SI; ++I) {
      // FIXME: This should really be AA driven.
      if (I->mayReadFromMemory() || I->mayWriteToMemory())
        return false;
    }
  }
  
  // Insert a PHI node now if we need it.
  Value *MergedVal = OtherStore->getOperand(0);
  if (MergedVal != SI.getOperand(0)) {
    PHINode *PN = PHINode::Create(MergedVal->getType(), "storemerge");
    PN->reserveOperandSpace(2);
    PN->addIncoming(SI.getOperand(0), SI.getParent());
    PN->addIncoming(OtherStore->getOperand(0), OtherBB);
    MergedVal = InsertNewInstBefore(PN, DestBB->front());
  }
  
  // Advance to a place where it is safe to insert the new store and
  // insert it.
  BBI = DestBB->getFirstNonPHI();
  InsertNewInstBefore(new StoreInst(MergedVal, SI.getOperand(1),
                                    OtherStore->isVolatile(),
                                    SI.getAlignment()), *BBI);
  
  // Nuke the old stores.
  EraseInstFromFunction(SI);
  EraseInstFromFunction(*OtherStore);
  ++NumCombined;
  return true;
}


Instruction *InstCombiner::visitBranchInst(BranchInst &BI) {
  // Change br (not X), label True, label False to: br X, label False, True
  Value *X = 0;
  BasicBlock *TrueDest;
  BasicBlock *FalseDest;
  if (match(&BI, m_Br(m_Not(m_Value(X)), TrueDest, FalseDest)) &&
      !isa<Constant>(X)) {
    // Swap Destinations and condition...
    BI.setCondition(X);
    BI.setSuccessor(0, FalseDest);
    BI.setSuccessor(1, TrueDest);
    return &BI;
  }

  // Cannonicalize fcmp_one -> fcmp_oeq
  FCmpInst::Predicate FPred; Value *Y;
  if (match(&BI, m_Br(m_FCmp(FPred, m_Value(X), m_Value(Y)), 
                             TrueDest, FalseDest)) &&
      BI.getCondition()->hasOneUse())
    if (FPred == FCmpInst::FCMP_ONE || FPred == FCmpInst::FCMP_OLE ||
        FPred == FCmpInst::FCMP_OGE) {
      FCmpInst *Cond = cast<FCmpInst>(BI.getCondition());
      Cond->setPredicate(FCmpInst::getInversePredicate(FPred));
      
      // Swap Destinations and condition.
      BI.setSuccessor(0, FalseDest);
      BI.setSuccessor(1, TrueDest);
      Worklist.Add(Cond);
      return &BI;
    }

  // Cannonicalize icmp_ne -> icmp_eq
  ICmpInst::Predicate IPred;
  if (match(&BI, m_Br(m_ICmp(IPred, m_Value(X), m_Value(Y)),
                      TrueDest, FalseDest)) &&
      BI.getCondition()->hasOneUse())
    if (IPred == ICmpInst::ICMP_NE  || IPred == ICmpInst::ICMP_ULE ||
        IPred == ICmpInst::ICMP_SLE || IPred == ICmpInst::ICMP_UGE ||
        IPred == ICmpInst::ICMP_SGE) {
      ICmpInst *Cond = cast<ICmpInst>(BI.getCondition());
      Cond->setPredicate(ICmpInst::getInversePredicate(IPred));
      // Swap Destinations and condition.
      BI.setSuccessor(0, FalseDest);
      BI.setSuccessor(1, TrueDest);
      Worklist.Add(Cond);
      return &BI;
    }

  return 0;
}

Instruction *InstCombiner::visitSwitchInst(SwitchInst &SI) {
  Value *Cond = SI.getCondition();
  if (Instruction *I = dyn_cast<Instruction>(Cond)) {
    if (I->getOpcode() == Instruction::Add)
      if (ConstantInt *AddRHS = dyn_cast<ConstantInt>(I->getOperand(1))) {
        // change 'switch (X+4) case 1:' into 'switch (X) case -3'
        for (unsigned i = 2, e = SI.getNumOperands(); i != e; i += 2)
          SI.setOperand(i,
                   ConstantExpr::getSub(cast<Constant>(SI.getOperand(i)),
                                                AddRHS));
        SI.setOperand(0, I->getOperand(0));
        Worklist.Add(I);
        return &SI;
      }
  }
  return 0;
}

Instruction *InstCombiner::visitExtractValueInst(ExtractValueInst &EV) {
  Value *Agg = EV.getAggregateOperand();

  if (!EV.hasIndices())
    return ReplaceInstUsesWith(EV, Agg);

  if (Constant *C = dyn_cast<Constant>(Agg)) {
    if (isa<UndefValue>(C))
      return ReplaceInstUsesWith(EV, UndefValue::get(EV.getType()));
      
    if (isa<ConstantAggregateZero>(C))
      return ReplaceInstUsesWith(EV, Constant::getNullValue(EV.getType()));

    if (isa<ConstantArray>(C) || isa<ConstantStruct>(C)) {
      // Extract the element indexed by the first index out of the constant
      Value *V = C->getOperand(*EV.idx_begin());
      if (EV.getNumIndices() > 1)
        // Extract the remaining indices out of the constant indexed by the
        // first index
        return ExtractValueInst::Create(V, EV.idx_begin() + 1, EV.idx_end());
      else
        return ReplaceInstUsesWith(EV, V);
    }
    return 0; // Can't handle other constants
  } 
  if (InsertValueInst *IV = dyn_cast<InsertValueInst>(Agg)) {
    // We're extracting from an insertvalue instruction, compare the indices
    const unsigned *exti, *exte, *insi, *inse;
    for (exti = EV.idx_begin(), insi = IV->idx_begin(),
         exte = EV.idx_end(), inse = IV->idx_end();
         exti != exte && insi != inse;
         ++exti, ++insi) {
      if (*insi != *exti)
        // The insert and extract both reference distinctly different elements.
        // This means the extract is not influenced by the insert, and we can
        // replace the aggregate operand of the extract with the aggregate
        // operand of the insert. i.e., replace
        // %I = insertvalue { i32, { i32 } } %A, { i32 } { i32 42 }, 1
        // %E = extractvalue { i32, { i32 } } %I, 0
        // with
        // %E = extractvalue { i32, { i32 } } %A, 0
        return ExtractValueInst::Create(IV->getAggregateOperand(),
                                        EV.idx_begin(), EV.idx_end());
    }
    if (exti == exte && insi == inse)
      // Both iterators are at the end: Index lists are identical. Replace
      // %B = insertvalue { i32, { i32 } } %A, i32 42, 1, 0
      // %C = extractvalue { i32, { i32 } } %B, 1, 0
      // with "i32 42"
      return ReplaceInstUsesWith(EV, IV->getInsertedValueOperand());
    if (exti == exte) {
      // The extract list is a prefix of the insert list. i.e. replace
      // %I = insertvalue { i32, { i32 } } %A, i32 42, 1, 0
      // %E = extractvalue { i32, { i32 } } %I, 1
      // with
      // %X = extractvalue { i32, { i32 } } %A, 1
      // %E = insertvalue { i32 } %X, i32 42, 0
      // by switching the order of the insert and extract (though the
      // insertvalue should be left in, since it may have other uses).
      Value *NewEV = Builder->CreateExtractValue(IV->getAggregateOperand(),
                                                 EV.idx_begin(), EV.idx_end());
      return InsertValueInst::Create(NewEV, IV->getInsertedValueOperand(),
                                     insi, inse);
    }
    if (insi == inse)
      // The insert list is a prefix of the extract list
      // We can simply remove the common indices from the extract and make it
      // operate on the inserted value instead of the insertvalue result.
      // i.e., replace
      // %I = insertvalue { i32, { i32 } } %A, { i32 } { i32 42 }, 1
      // %E = extractvalue { i32, { i32 } } %I, 1, 0
      // with
      // %E extractvalue { i32 } { i32 42 }, 0
      return ExtractValueInst::Create(IV->getInsertedValueOperand(), 
                                      exti, exte);
  }
  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(Agg)) {
    // We're extracting from an intrinsic, see if we're the only user, which
    // allows us to simplify multiple result intrinsics to simpler things that
    // just get one value..
    if (II->hasOneUse()) {
      // Check if we're grabbing the overflow bit or the result of a 'with
      // overflow' intrinsic.  If it's the latter we can remove the intrinsic
      // and replace it with a traditional binary instruction.
      switch (II->getIntrinsicID()) {
      case Intrinsic::uadd_with_overflow:
      case Intrinsic::sadd_with_overflow:
        if (*EV.idx_begin() == 0) {  // Normal result.
          Value *LHS = II->getOperand(1), *RHS = II->getOperand(2);
          II->replaceAllUsesWith(UndefValue::get(II->getType()));
          EraseInstFromFunction(*II);
          return BinaryOperator::CreateAdd(LHS, RHS);
        }
        break;
      case Intrinsic::usub_with_overflow:
      case Intrinsic::ssub_with_overflow:
        if (*EV.idx_begin() == 0) {  // Normal result.
          Value *LHS = II->getOperand(1), *RHS = II->getOperand(2);
          II->replaceAllUsesWith(UndefValue::get(II->getType()));
          EraseInstFromFunction(*II);
          return BinaryOperator::CreateSub(LHS, RHS);
        }
        break;
      case Intrinsic::umul_with_overflow:
      case Intrinsic::smul_with_overflow:
        if (*EV.idx_begin() == 0) {  // Normal result.
          Value *LHS = II->getOperand(1), *RHS = II->getOperand(2);
          II->replaceAllUsesWith(UndefValue::get(II->getType()));
          EraseInstFromFunction(*II);
          return BinaryOperator::CreateMul(LHS, RHS);
        }
        break;
      default:
        break;
      }
    }
  }
  // Can't simplify extracts from other values. Note that nested extracts are
  // already simplified implicitely by the above (extract ( extract (insert) )
  // will be translated into extract ( insert ( extract ) ) first and then just
  // the value inserted, if appropriate).
  return 0;
}

/// CheapToScalarize - Return true if the value is cheaper to scalarize than it
/// is to leave as a vector operation.
static bool CheapToScalarize(Value *V, bool isConstant) {
  if (isa<ConstantAggregateZero>(V)) 
    return true;
  if (ConstantVector *C = dyn_cast<ConstantVector>(V)) {
    if (isConstant) return true;
    // If all elts are the same, we can extract.
    Constant *Op0 = C->getOperand(0);
    for (unsigned i = 1; i < C->getNumOperands(); ++i)
      if (C->getOperand(i) != Op0)
        return false;
    return true;
  }
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I) return false;
  
  // Insert element gets simplified to the inserted element or is deleted if
  // this is constant idx extract element and its a constant idx insertelt.
  if (I->getOpcode() == Instruction::InsertElement && isConstant &&
      isa<ConstantInt>(I->getOperand(2)))
    return true;
  if (I->getOpcode() == Instruction::Load && I->hasOneUse())
    return true;
  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(I))
    if (BO->hasOneUse() &&
        (CheapToScalarize(BO->getOperand(0), isConstant) ||
         CheapToScalarize(BO->getOperand(1), isConstant)))
      return true;
  if (CmpInst *CI = dyn_cast<CmpInst>(I))
    if (CI->hasOneUse() &&
        (CheapToScalarize(CI->getOperand(0), isConstant) ||
         CheapToScalarize(CI->getOperand(1), isConstant)))
      return true;
  
  return false;
}

/// Read and decode a shufflevector mask.
///
/// It turns undef elements into values that are larger than the number of
/// elements in the input.
static std::vector<unsigned> getShuffleMask(const ShuffleVectorInst *SVI) {
  unsigned NElts = SVI->getType()->getNumElements();
  if (isa<ConstantAggregateZero>(SVI->getOperand(2)))
    return std::vector<unsigned>(NElts, 0);
  if (isa<UndefValue>(SVI->getOperand(2)))
    return std::vector<unsigned>(NElts, 2*NElts);

  std::vector<unsigned> Result;
  const ConstantVector *CP = cast<ConstantVector>(SVI->getOperand(2));
  for (User::const_op_iterator i = CP->op_begin(), e = CP->op_end(); i!=e; ++i)
    if (isa<UndefValue>(*i))
      Result.push_back(NElts*2);  // undef -> 8
    else
      Result.push_back(cast<ConstantInt>(*i)->getZExtValue());
  return Result;
}

/// FindScalarElement - Given a vector and an element number, see if the scalar
/// value is already around as a register, for example if it were inserted then
/// extracted from the vector.
static Value *FindScalarElement(Value *V, unsigned EltNo) {
  assert(isa<VectorType>(V->getType()) && "Not looking at a vector?");
  const VectorType *PTy = cast<VectorType>(V->getType());
  unsigned Width = PTy->getNumElements();
  if (EltNo >= Width)  // Out of range access.
    return UndefValue::get(PTy->getElementType());
  
  if (isa<UndefValue>(V))
    return UndefValue::get(PTy->getElementType());
  else if (isa<ConstantAggregateZero>(V))
    return Constant::getNullValue(PTy->getElementType());
  else if (ConstantVector *CP = dyn_cast<ConstantVector>(V))
    return CP->getOperand(EltNo);
  else if (InsertElementInst *III = dyn_cast<InsertElementInst>(V)) {
    // If this is an insert to a variable element, we don't know what it is.
    if (!isa<ConstantInt>(III->getOperand(2))) 
      return 0;
    unsigned IIElt = cast<ConstantInt>(III->getOperand(2))->getZExtValue();
    
    // If this is an insert to the element we are looking for, return the
    // inserted value.
    if (EltNo == IIElt) 
      return III->getOperand(1);
    
    // Otherwise, the insertelement doesn't modify the value, recurse on its
    // vector input.
    return FindScalarElement(III->getOperand(0), EltNo);
  } else if (ShuffleVectorInst *SVI = dyn_cast<ShuffleVectorInst>(V)) {
    unsigned LHSWidth =
      cast<VectorType>(SVI->getOperand(0)->getType())->getNumElements();
    unsigned InEl = getShuffleMask(SVI)[EltNo];
    if (InEl < LHSWidth)
      return FindScalarElement(SVI->getOperand(0), InEl);
    else if (InEl < LHSWidth*2)
      return FindScalarElement(SVI->getOperand(1), InEl - LHSWidth);
    else
      return UndefValue::get(PTy->getElementType());
  }
  
  // Otherwise, we don't know.
  return 0;
}

Instruction *InstCombiner::visitExtractElementInst(ExtractElementInst &EI) {
  // If vector val is undef, replace extract with scalar undef.
  if (isa<UndefValue>(EI.getOperand(0)))
    return ReplaceInstUsesWith(EI, UndefValue::get(EI.getType()));

  // If vector val is constant 0, replace extract with scalar 0.
  if (isa<ConstantAggregateZero>(EI.getOperand(0)))
    return ReplaceInstUsesWith(EI, Constant::getNullValue(EI.getType()));
  
  if (ConstantVector *C = dyn_cast<ConstantVector>(EI.getOperand(0))) {
    // If vector val is constant with all elements the same, replace EI with
    // that element. When the elements are not identical, we cannot replace yet
    // (we do that below, but only when the index is constant).
    Constant *op0 = C->getOperand(0);
    for (unsigned i = 1; i != C->getNumOperands(); ++i)
      if (C->getOperand(i) != op0) {
        op0 = 0; 
        break;
      }
    if (op0)
      return ReplaceInstUsesWith(EI, op0);
  }
  
  // If extracting a specified index from the vector, see if we can recursively
  // find a previously computed scalar that was inserted into the vector.
  if (ConstantInt *IdxC = dyn_cast<ConstantInt>(EI.getOperand(1))) {
    unsigned IndexVal = IdxC->getZExtValue();
    unsigned VectorWidth = EI.getVectorOperandType()->getNumElements();
      
    // If this is extracting an invalid index, turn this into undef, to avoid
    // crashing the code below.
    if (IndexVal >= VectorWidth)
      return ReplaceInstUsesWith(EI, UndefValue::get(EI.getType()));
    
    // This instruction only demands the single element from the input vector.
    // If the input vector has a single use, simplify it based on this use
    // property.
    if (EI.getOperand(0)->hasOneUse() && VectorWidth != 1) {
      APInt UndefElts(VectorWidth, 0);
      APInt DemandedMask(VectorWidth, 1 << IndexVal);
      if (Value *V = SimplifyDemandedVectorElts(EI.getOperand(0),
                                                DemandedMask, UndefElts)) {
        EI.setOperand(0, V);
        return &EI;
      }
    }
    
    if (Value *Elt = FindScalarElement(EI.getOperand(0), IndexVal))
      return ReplaceInstUsesWith(EI, Elt);
    
    // If the this extractelement is directly using a bitcast from a vector of
    // the same number of elements, see if we can find the source element from
    // it.  In this case, we will end up needing to bitcast the scalars.
    if (BitCastInst *BCI = dyn_cast<BitCastInst>(EI.getOperand(0))) {
      if (const VectorType *VT = 
              dyn_cast<VectorType>(BCI->getOperand(0)->getType()))
        if (VT->getNumElements() == VectorWidth)
          if (Value *Elt = FindScalarElement(BCI->getOperand(0), IndexVal))
            return new BitCastInst(Elt, EI.getType());
    }
  }
  
  if (Instruction *I = dyn_cast<Instruction>(EI.getOperand(0))) {
    // Push extractelement into predecessor operation if legal and
    // profitable to do so
    if (BinaryOperator *BO = dyn_cast<BinaryOperator>(I)) {
      if (I->hasOneUse() &&
          CheapToScalarize(BO, isa<ConstantInt>(EI.getOperand(1)))) {
        Value *newEI0 =
          Builder->CreateExtractElement(BO->getOperand(0), EI.getOperand(1),
                                        EI.getName()+".lhs");
        Value *newEI1 =
          Builder->CreateExtractElement(BO->getOperand(1), EI.getOperand(1),
                                        EI.getName()+".rhs");
        return BinaryOperator::Create(BO->getOpcode(), newEI0, newEI1);
      }
    } else if (InsertElementInst *IE = dyn_cast<InsertElementInst>(I)) {
      // Extracting the inserted element?
      if (IE->getOperand(2) == EI.getOperand(1))
        return ReplaceInstUsesWith(EI, IE->getOperand(1));
      // If the inserted and extracted elements are constants, they must not
      // be the same value, extract from the pre-inserted value instead.
      if (isa<Constant>(IE->getOperand(2)) && isa<Constant>(EI.getOperand(1))) {
        Worklist.AddValue(EI.getOperand(0));
        EI.setOperand(0, IE->getOperand(0));
        return &EI;
      }
    } else if (ShuffleVectorInst *SVI = dyn_cast<ShuffleVectorInst>(I)) {
      // If this is extracting an element from a shufflevector, figure out where
      // it came from and extract from the appropriate input element instead.
      if (ConstantInt *Elt = dyn_cast<ConstantInt>(EI.getOperand(1))) {
        unsigned SrcIdx = getShuffleMask(SVI)[Elt->getZExtValue()];
        Value *Src;
        unsigned LHSWidth =
          cast<VectorType>(SVI->getOperand(0)->getType())->getNumElements();

        if (SrcIdx < LHSWidth)
          Src = SVI->getOperand(0);
        else if (SrcIdx < LHSWidth*2) {
          SrcIdx -= LHSWidth;
          Src = SVI->getOperand(1);
        } else {
          return ReplaceInstUsesWith(EI, UndefValue::get(EI.getType()));
        }
        return ExtractElementInst::Create(Src,
                         ConstantInt::get(Type::getInt32Ty(EI.getContext()),
                                          SrcIdx, false));
      }
    }
    // FIXME: Canonicalize extractelement(bitcast) -> bitcast(extractelement)
  }
  return 0;
}

/// CollectSingleShuffleElements - If V is a shuffle of values that ONLY returns
/// elements from either LHS or RHS, return the shuffle mask and true. 
/// Otherwise, return false.
static bool CollectSingleShuffleElements(Value *V, Value *LHS, Value *RHS,
                                         std::vector<Constant*> &Mask) {
  assert(V->getType() == LHS->getType() && V->getType() == RHS->getType() &&
         "Invalid CollectSingleShuffleElements");
  unsigned NumElts = cast<VectorType>(V->getType())->getNumElements();

  if (isa<UndefValue>(V)) {
    Mask.assign(NumElts, UndefValue::get(Type::getInt32Ty(V->getContext())));
    return true;
  }
  
  if (V == LHS) {
    for (unsigned i = 0; i != NumElts; ++i)
      Mask.push_back(ConstantInt::get(Type::getInt32Ty(V->getContext()), i));
    return true;
  }
  
  if (V == RHS) {
    for (unsigned i = 0; i != NumElts; ++i)
      Mask.push_back(ConstantInt::get(Type::getInt32Ty(V->getContext()),
                                      i+NumElts));
    return true;
  }
  
  if (InsertElementInst *IEI = dyn_cast<InsertElementInst>(V)) {
    // If this is an insert of an extract from some other vector, include it.
    Value *VecOp    = IEI->getOperand(0);
    Value *ScalarOp = IEI->getOperand(1);
    Value *IdxOp    = IEI->getOperand(2);
    
    if (!isa<ConstantInt>(IdxOp))
      return false;
    unsigned InsertedIdx = cast<ConstantInt>(IdxOp)->getZExtValue();
    
    if (isa<UndefValue>(ScalarOp)) {  // inserting undef into vector.
      // Okay, we can handle this if the vector we are insertinting into is
      // transitively ok.
      if (CollectSingleShuffleElements(VecOp, LHS, RHS, Mask)) {
        // If so, update the mask to reflect the inserted undef.
        Mask[InsertedIdx] = UndefValue::get(Type::getInt32Ty(V->getContext()));
        return true;
      }      
    } else if (ExtractElementInst *EI = dyn_cast<ExtractElementInst>(ScalarOp)){
      if (isa<ConstantInt>(EI->getOperand(1)) &&
          EI->getOperand(0)->getType() == V->getType()) {
        unsigned ExtractedIdx =
          cast<ConstantInt>(EI->getOperand(1))->getZExtValue();
        
        // This must be extracting from either LHS or RHS.
        if (EI->getOperand(0) == LHS || EI->getOperand(0) == RHS) {
          // Okay, we can handle this if the vector we are insertinting into is
          // transitively ok.
          if (CollectSingleShuffleElements(VecOp, LHS, RHS, Mask)) {
            // If so, update the mask to reflect the inserted value.
            if (EI->getOperand(0) == LHS) {
              Mask[InsertedIdx % NumElts] = 
                 ConstantInt::get(Type::getInt32Ty(V->getContext()),
                                  ExtractedIdx);
            } else {
              assert(EI->getOperand(0) == RHS);
              Mask[InsertedIdx % NumElts] = 
                ConstantInt::get(Type::getInt32Ty(V->getContext()),
                                 ExtractedIdx+NumElts);
              
            }
            return true;
          }
        }
      }
    }
  }
  // TODO: Handle shufflevector here!
  
  return false;
}

/// CollectShuffleElements - We are building a shuffle of V, using RHS as the
/// RHS of the shuffle instruction, if it is not null.  Return a shuffle mask
/// that computes V and the LHS value of the shuffle.
static Value *CollectShuffleElements(Value *V, std::vector<Constant*> &Mask,
                                     Value *&RHS) {
  assert(isa<VectorType>(V->getType()) && 
         (RHS == 0 || V->getType() == RHS->getType()) &&
         "Invalid shuffle!");
  unsigned NumElts = cast<VectorType>(V->getType())->getNumElements();

  if (isa<UndefValue>(V)) {
    Mask.assign(NumElts, UndefValue::get(Type::getInt32Ty(V->getContext())));
    return V;
  } else if (isa<ConstantAggregateZero>(V)) {
    Mask.assign(NumElts, ConstantInt::get(Type::getInt32Ty(V->getContext()),0));
    return V;
  } else if (InsertElementInst *IEI = dyn_cast<InsertElementInst>(V)) {
    // If this is an insert of an extract from some other vector, include it.
    Value *VecOp    = IEI->getOperand(0);
    Value *ScalarOp = IEI->getOperand(1);
    Value *IdxOp    = IEI->getOperand(2);
    
    if (ExtractElementInst *EI = dyn_cast<ExtractElementInst>(ScalarOp)) {
      if (isa<ConstantInt>(EI->getOperand(1)) && isa<ConstantInt>(IdxOp) &&
          EI->getOperand(0)->getType() == V->getType()) {
        unsigned ExtractedIdx =
          cast<ConstantInt>(EI->getOperand(1))->getZExtValue();
        unsigned InsertedIdx = cast<ConstantInt>(IdxOp)->getZExtValue();
        
        // Either the extracted from or inserted into vector must be RHSVec,
        // otherwise we'd end up with a shuffle of three inputs.
        if (EI->getOperand(0) == RHS || RHS == 0) {
          RHS = EI->getOperand(0);
          Value *V = CollectShuffleElements(VecOp, Mask, RHS);
          Mask[InsertedIdx % NumElts] = 
            ConstantInt::get(Type::getInt32Ty(V->getContext()),
                             NumElts+ExtractedIdx);
          return V;
        }
        
        if (VecOp == RHS) {
          Value *V = CollectShuffleElements(EI->getOperand(0), Mask, RHS);
          // Everything but the extracted element is replaced with the RHS.
          for (unsigned i = 0; i != NumElts; ++i) {
            if (i != InsertedIdx)
              Mask[i] = ConstantInt::get(Type::getInt32Ty(V->getContext()),
                                         NumElts+i);
          }
          return V;
        }
        
        // If this insertelement is a chain that comes from exactly these two
        // vectors, return the vector and the effective shuffle.
        if (CollectSingleShuffleElements(IEI, EI->getOperand(0), RHS, Mask))
          return EI->getOperand(0);
      }
    }
  }
  // TODO: Handle shufflevector here!
  
  // Otherwise, can't do anything fancy.  Return an identity vector.
  for (unsigned i = 0; i != NumElts; ++i)
    Mask.push_back(ConstantInt::get(Type::getInt32Ty(V->getContext()), i));
  return V;
}

Instruction *InstCombiner::visitInsertElementInst(InsertElementInst &IE) {
  Value *VecOp    = IE.getOperand(0);
  Value *ScalarOp = IE.getOperand(1);
  Value *IdxOp    = IE.getOperand(2);
  
  // Inserting an undef or into an undefined place, remove this.
  if (isa<UndefValue>(ScalarOp) || isa<UndefValue>(IdxOp))
    ReplaceInstUsesWith(IE, VecOp);
  
  // If the inserted element was extracted from some other vector, and if the 
  // indexes are constant, try to turn this into a shufflevector operation.
  if (ExtractElementInst *EI = dyn_cast<ExtractElementInst>(ScalarOp)) {
    if (isa<ConstantInt>(EI->getOperand(1)) && isa<ConstantInt>(IdxOp) &&
        EI->getOperand(0)->getType() == IE.getType()) {
      unsigned NumVectorElts = IE.getType()->getNumElements();
      unsigned ExtractedIdx =
        cast<ConstantInt>(EI->getOperand(1))->getZExtValue();
      unsigned InsertedIdx = cast<ConstantInt>(IdxOp)->getZExtValue();
      
      if (ExtractedIdx >= NumVectorElts) // Out of range extract.
        return ReplaceInstUsesWith(IE, VecOp);
      
      if (InsertedIdx >= NumVectorElts)  // Out of range insert.
        return ReplaceInstUsesWith(IE, UndefValue::get(IE.getType()));
      
      // If we are extracting a value from a vector, then inserting it right
      // back into the same place, just use the input vector.
      if (EI->getOperand(0) == VecOp && ExtractedIdx == InsertedIdx)
        return ReplaceInstUsesWith(IE, VecOp);      
      
      // If this insertelement isn't used by some other insertelement, turn it
      // (and any insertelements it points to), into one big shuffle.
      if (!IE.hasOneUse() || !isa<InsertElementInst>(IE.use_back())) {
        std::vector<Constant*> Mask;
        Value *RHS = 0;
        Value *LHS = CollectShuffleElements(&IE, Mask, RHS);
        if (RHS == 0) RHS = UndefValue::get(LHS->getType());
        // We now have a shuffle of LHS, RHS, Mask.
        return new ShuffleVectorInst(LHS, RHS,
                                     ConstantVector::get(Mask));
      }
    }
  }

  unsigned VWidth = cast<VectorType>(VecOp->getType())->getNumElements();
  APInt UndefElts(VWidth, 0);
  APInt AllOnesEltMask(APInt::getAllOnesValue(VWidth));
  if (SimplifyDemandedVectorElts(&IE, AllOnesEltMask, UndefElts))
    return &IE;

  return 0;
}


Instruction *InstCombiner::visitShuffleVectorInst(ShuffleVectorInst &SVI) {
  Value *LHS = SVI.getOperand(0);
  Value *RHS = SVI.getOperand(1);
  std::vector<unsigned> Mask = getShuffleMask(&SVI);

  bool MadeChange = false;

  // Undefined shuffle mask -> undefined value.
  if (isa<UndefValue>(SVI.getOperand(2)))
    return ReplaceInstUsesWith(SVI, UndefValue::get(SVI.getType()));

  unsigned VWidth = cast<VectorType>(SVI.getType())->getNumElements();

  if (VWidth != cast<VectorType>(LHS->getType())->getNumElements())
    return 0;

  APInt UndefElts(VWidth, 0);
  APInt AllOnesEltMask(APInt::getAllOnesValue(VWidth));
  if (SimplifyDemandedVectorElts(&SVI, AllOnesEltMask, UndefElts)) {
    LHS = SVI.getOperand(0);
    RHS = SVI.getOperand(1);
    MadeChange = true;
  }
  
  // Canonicalize shuffle(x    ,x,mask) -> shuffle(x, undef,mask')
  // Canonicalize shuffle(undef,x,mask) -> shuffle(x, undef,mask').
  if (LHS == RHS || isa<UndefValue>(LHS)) {
    if (isa<UndefValue>(LHS) && LHS == RHS) {
      // shuffle(undef,undef,mask) -> undef.
      return ReplaceInstUsesWith(SVI, LHS);
    }
    
    // Remap any references to RHS to use LHS.
    std::vector<Constant*> Elts;
    for (unsigned i = 0, e = Mask.size(); i != e; ++i) {
      if (Mask[i] >= 2*e)
        Elts.push_back(UndefValue::get(Type::getInt32Ty(SVI.getContext())));
      else {
        if ((Mask[i] >= e && isa<UndefValue>(RHS)) ||
            (Mask[i] <  e && isa<UndefValue>(LHS))) {
          Mask[i] = 2*e;     // Turn into undef.
          Elts.push_back(UndefValue::get(Type::getInt32Ty(SVI.getContext())));
        } else {
          Mask[i] = Mask[i] % e;  // Force to LHS.
          Elts.push_back(ConstantInt::get(Type::getInt32Ty(SVI.getContext()),
                                          Mask[i]));
        }
      }
    }
    SVI.setOperand(0, SVI.getOperand(1));
    SVI.setOperand(1, UndefValue::get(RHS->getType()));
    SVI.setOperand(2, ConstantVector::get(Elts));
    LHS = SVI.getOperand(0);
    RHS = SVI.getOperand(1);
    MadeChange = true;
  }
  
  // Analyze the shuffle, are the LHS or RHS and identity shuffles?
  bool isLHSID = true, isRHSID = true;
    
  for (unsigned i = 0, e = Mask.size(); i != e; ++i) {
    if (Mask[i] >= e*2) continue;  // Ignore undef values.
    // Is this an identity shuffle of the LHS value?
    isLHSID &= (Mask[i] == i);
      
    // Is this an identity shuffle of the RHS value?
    isRHSID &= (Mask[i]-e == i);
  }

  // Eliminate identity shuffles.
  if (isLHSID) return ReplaceInstUsesWith(SVI, LHS);
  if (isRHSID) return ReplaceInstUsesWith(SVI, RHS);
  
  // If the LHS is a shufflevector itself, see if we can combine it with this
  // one without producing an unusual shuffle.  Here we are really conservative:
  // we are absolutely afraid of producing a shuffle mask not in the input
  // program, because the code gen may not be smart enough to turn a merged
  // shuffle into two specific shuffles: it may produce worse code.  As such,
  // we only merge two shuffles if the result is one of the two input shuffle
  // masks.  In this case, merging the shuffles just removes one instruction,
  // which we know is safe.  This is good for things like turning:
  // (splat(splat)) -> splat.
  if (ShuffleVectorInst *LHSSVI = dyn_cast<ShuffleVectorInst>(LHS)) {
    if (isa<UndefValue>(RHS)) {
      std::vector<unsigned> LHSMask = getShuffleMask(LHSSVI);

      if (LHSMask.size() == Mask.size()) {
        std::vector<unsigned> NewMask;
        for (unsigned i = 0, e = Mask.size(); i != e; ++i)
          if (Mask[i] >= e)
            NewMask.push_back(2*e);
          else
            NewMask.push_back(LHSMask[Mask[i]]);
      
        // If the result mask is equal to the src shuffle or this
        // shuffle mask, do the replacement.
        if (NewMask == LHSMask || NewMask == Mask) {
          unsigned LHSInNElts =
            cast<VectorType>(LHSSVI->getOperand(0)->getType())->
            getNumElements();
          std::vector<Constant*> Elts;
          for (unsigned i = 0, e = NewMask.size(); i != e; ++i) {
            if (NewMask[i] >= LHSInNElts*2) {
              Elts.push_back(UndefValue::get(
                                           Type::getInt32Ty(SVI.getContext())));
            } else {
              Elts.push_back(ConstantInt::get(
                                           Type::getInt32Ty(SVI.getContext()),
                                              NewMask[i]));
            }
          }
          return new ShuffleVectorInst(LHSSVI->getOperand(0),
                                       LHSSVI->getOperand(1),
                                       ConstantVector::get(Elts));
        }
      }
    }
  }

  return MadeChange ? &SVI : 0;
}




/// TryToSinkInstruction - Try to move the specified instruction from its
/// current block into the beginning of DestBlock, which can only happen if it's
/// safe to move the instruction past all of the instructions between it and the
/// end of its block.
static bool TryToSinkInstruction(Instruction *I, BasicBlock *DestBlock) {
  assert(I->hasOneUse() && "Invariants didn't hold!");

  // Cannot move control-flow-involving, volatile loads, vaarg, etc.
  if (isa<PHINode>(I) || I->mayHaveSideEffects() || isa<TerminatorInst>(I))
    return false;

  // Do not sink alloca instructions out of the entry block.
  if (isa<AllocaInst>(I) && I->getParent() ==
        &DestBlock->getParent()->getEntryBlock())
    return false;

  // We can only sink load instructions if there is nothing between the load and
  // the end of block that could change the value.
  if (I->mayReadFromMemory()) {
    for (BasicBlock::iterator Scan = I, E = I->getParent()->end();
         Scan != E; ++Scan)
      if (Scan->mayWriteToMemory())
        return false;
  }

  BasicBlock::iterator InsertPos = DestBlock->getFirstNonPHI();

  I->moveBefore(InsertPos);
  ++NumSunkInst;
  return true;
}


/// AddReachableCodeToWorklist - Walk the function in depth-first order, adding
/// all reachable code to the worklist.
///
/// This has a couple of tricks to make the code faster and more powerful.  In
/// particular, we constant fold and DCE instructions as we go, to avoid adding
/// them to the worklist (this significantly speeds up instcombine on code where
/// many instructions are dead or constant).  Additionally, if we find a branch
/// whose condition is a known constant, we only visit the reachable successors.
///
static bool AddReachableCodeToWorklist(BasicBlock *BB, 
                                       SmallPtrSet<BasicBlock*, 64> &Visited,
                                       InstCombiner &IC,
                                       const TargetData *TD) {
  bool MadeIRChange = false;
  SmallVector<BasicBlock*, 256> Worklist;
  Worklist.push_back(BB);
  
  std::vector<Instruction*> InstrsForInstCombineWorklist;
  InstrsForInstCombineWorklist.reserve(128);

  SmallPtrSet<ConstantExpr*, 64> FoldedConstants;
  
  while (!Worklist.empty()) {
    BB = Worklist.back();
    Worklist.pop_back();
    
    // We have now visited this block!  If we've already been here, ignore it.
    if (!Visited.insert(BB)) continue;

    for (BasicBlock::iterator BBI = BB->begin(), E = BB->end(); BBI != E; ) {
      Instruction *Inst = BBI++;
      
      // DCE instruction if trivially dead.
      if (isInstructionTriviallyDead(Inst)) {
        ++NumDeadInst;
        DEBUG(errs() << "IC: DCE: " << *Inst << '\n');
        Inst->eraseFromParent();
        continue;
      }
      
      // ConstantProp instruction if trivially constant.
      if (!Inst->use_empty() && isa<Constant>(Inst->getOperand(0)))
        if (Constant *C = ConstantFoldInstruction(Inst, TD)) {
          DEBUG(errs() << "IC: ConstFold to: " << *C << " from: "
                       << *Inst << '\n');
          Inst->replaceAllUsesWith(C);
          ++NumConstProp;
          Inst->eraseFromParent();
          continue;
        }
      
      
      
      if (TD) {
        // See if we can constant fold its operands.
        for (User::op_iterator i = Inst->op_begin(), e = Inst->op_end();
             i != e; ++i) {
          ConstantExpr *CE = dyn_cast<ConstantExpr>(i);
          if (CE == 0) continue;
          
          // If we already folded this constant, don't try again.
          if (!FoldedConstants.insert(CE))
            continue;
          
          Constant *NewC = ConstantFoldConstantExpression(CE, TD);
          if (NewC && NewC != CE) {
            *i = NewC;
            MadeIRChange = true;
          }
        }
      }
      

      InstrsForInstCombineWorklist.push_back(Inst);
    }

    // Recursively visit successors.  If this is a branch or switch on a
    // constant, only visit the reachable successor.
    TerminatorInst *TI = BB->getTerminator();
    if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
      if (BI->isConditional() && isa<ConstantInt>(BI->getCondition())) {
        bool CondVal = cast<ConstantInt>(BI->getCondition())->getZExtValue();
        BasicBlock *ReachableBB = BI->getSuccessor(!CondVal);
        Worklist.push_back(ReachableBB);
        continue;
      }
    } else if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
      if (ConstantInt *Cond = dyn_cast<ConstantInt>(SI->getCondition())) {
        // See if this is an explicit destination.
        for (unsigned i = 1, e = SI->getNumSuccessors(); i != e; ++i)
          if (SI->getCaseValue(i) == Cond) {
            BasicBlock *ReachableBB = SI->getSuccessor(i);
            Worklist.push_back(ReachableBB);
            continue;
          }
        
        // Otherwise it is the default destination.
        Worklist.push_back(SI->getSuccessor(0));
        continue;
      }
    }
    
    for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i)
      Worklist.push_back(TI->getSuccessor(i));
  }
  
  // Once we've found all of the instructions to add to instcombine's worklist,
  // add them in reverse order.  This way instcombine will visit from the top
  // of the function down.  This jives well with the way that it adds all uses
  // of instructions to the worklist after doing a transformation, thus avoiding
  // some N^2 behavior in pathological cases.
  IC.Worklist.AddInitialGroup(&InstrsForInstCombineWorklist[0],
                              InstrsForInstCombineWorklist.size());
  
  return MadeIRChange;
}

bool InstCombiner::DoOneIteration(Function &F, unsigned Iteration) {
  MadeIRChange = false;
  
  DEBUG(errs() << "\n\nINSTCOMBINE ITERATION #" << Iteration << " on "
        << F.getNameStr() << "\n");

  {
    // Do a depth-first traversal of the function, populate the worklist with
    // the reachable instructions.  Ignore blocks that are not reachable.  Keep
    // track of which blocks we visit.
    SmallPtrSet<BasicBlock*, 64> Visited;
    MadeIRChange |= AddReachableCodeToWorklist(F.begin(), Visited, *this, TD);

    // Do a quick scan over the function.  If we find any blocks that are
    // unreachable, remove any instructions inside of them.  This prevents
    // the instcombine code from having to deal with some bad special cases.
    for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
      if (!Visited.count(BB)) {
        Instruction *Term = BB->getTerminator();
        while (Term != BB->begin()) {   // Remove instrs bottom-up
          BasicBlock::iterator I = Term; --I;

          DEBUG(errs() << "IC: DCE: " << *I << '\n');
          // A debug intrinsic shouldn't force another iteration if we weren't
          // going to do one without it.
          if (!isa<DbgInfoIntrinsic>(I)) {
            ++NumDeadInst;
            MadeIRChange = true;
          }

          // If I is not void type then replaceAllUsesWith undef.
          // This allows ValueHandlers and custom metadata to adjust itself.
          if (!I->getType()->isVoidTy())
            I->replaceAllUsesWith(UndefValue::get(I->getType()));
          I->eraseFromParent();
        }
      }
  }

  while (!Worklist.isEmpty()) {
    Instruction *I = Worklist.RemoveOne();
    if (I == 0) continue;  // skip null values.

    // Check to see if we can DCE the instruction.
    if (isInstructionTriviallyDead(I)) {
      DEBUG(errs() << "IC: DCE: " << *I << '\n');
      EraseInstFromFunction(*I);
      ++NumDeadInst;
      MadeIRChange = true;
      continue;
    }

    // Instruction isn't dead, see if we can constant propagate it.
    if (!I->use_empty() && isa<Constant>(I->getOperand(0)))
      if (Constant *C = ConstantFoldInstruction(I, TD)) {
        DEBUG(errs() << "IC: ConstFold to: " << *C << " from: " << *I << '\n');

        // Add operands to the worklist.
        ReplaceInstUsesWith(*I, C);
        ++NumConstProp;
        EraseInstFromFunction(*I);
        MadeIRChange = true;
        continue;
      }

    // See if we can trivially sink this instruction to a successor basic block.
    if (I->hasOneUse()) {
      BasicBlock *BB = I->getParent();
      Instruction *UserInst = cast<Instruction>(I->use_back());
      BasicBlock *UserParent;
      
      // Get the block the use occurs in.
      if (PHINode *PN = dyn_cast<PHINode>(UserInst))
        UserParent = PN->getIncomingBlock(I->use_begin().getUse());
      else
        UserParent = UserInst->getParent();
      
      if (UserParent != BB) {
        bool UserIsSuccessor = false;
        // See if the user is one of our successors.
        for (succ_iterator SI = succ_begin(BB), E = succ_end(BB); SI != E; ++SI)
          if (*SI == UserParent) {
            UserIsSuccessor = true;
            break;
          }

        // If the user is one of our immediate successors, and if that successor
        // only has us as a predecessors (we'd have to split the critical edge
        // otherwise), we can keep going.
        if (UserIsSuccessor && UserParent->getSinglePredecessor())
          // Okay, the CFG is simple enough, try to sink this instruction.
          MadeIRChange |= TryToSinkInstruction(I, UserParent);
      }
    }

    // Now that we have an instruction, try combining it to simplify it.
    Builder->SetInsertPoint(I->getParent(), I);
    
#ifndef NDEBUG
    std::string OrigI;
#endif
    DEBUG(raw_string_ostream SS(OrigI); I->print(SS); OrigI = SS.str(););
    DEBUG(errs() << "IC: Visiting: " << OrigI << '\n');

    if (Instruction *Result = visit(*I)) {
      ++NumCombined;
      // Should we replace the old instruction with a new one?
      if (Result != I) {
        DEBUG(errs() << "IC: Old = " << *I << '\n'
                     << "    New = " << *Result << '\n');

        // Everything uses the new instruction now.
        I->replaceAllUsesWith(Result);

        // Push the new instruction and any users onto the worklist.
        Worklist.Add(Result);
        Worklist.AddUsersToWorkList(*Result);

        // Move the name to the new instruction first.
        Result->takeName(I);

        // Insert the new instruction into the basic block...
        BasicBlock *InstParent = I->getParent();
        BasicBlock::iterator InsertPos = I;

        if (!isa<PHINode>(Result))        // If combining a PHI, don't insert
          while (isa<PHINode>(InsertPos)) // middle of a block of PHIs.
            ++InsertPos;

        InstParent->getInstList().insert(InsertPos, Result);

        EraseInstFromFunction(*I);
      } else {
#ifndef NDEBUG
        DEBUG(errs() << "IC: Mod = " << OrigI << '\n'
                     << "    New = " << *I << '\n');
#endif

        // If the instruction was modified, it's possible that it is now dead.
        // if so, remove it.
        if (isInstructionTriviallyDead(I)) {
          EraseInstFromFunction(*I);
        } else {
          Worklist.Add(I);
          Worklist.AddUsersToWorkList(*I);
        }
      }
      MadeIRChange = true;
    }
  }

  Worklist.Zap();
  return MadeIRChange;
}


bool InstCombiner::runOnFunction(Function &F) {
  MustPreserveLCSSA = mustPreserveAnalysisID(LCSSAID);
  TD = getAnalysisIfAvailable<TargetData>();

  
  /// Builder - This is an IRBuilder that automatically inserts new
  /// instructions into the worklist when they are created.
  IRBuilder<true, TargetFolder, InstCombineIRInserter> 
    TheBuilder(F.getContext(), TargetFolder(TD),
               InstCombineIRInserter(Worklist));
  Builder = &TheBuilder;
  
  bool EverMadeChange = false;

  // Iterate while there is work to do.
  unsigned Iteration = 0;
  while (DoOneIteration(F, Iteration++))
    EverMadeChange = true;
  
  Builder = 0;
  return EverMadeChange;
}

FunctionPass *llvm::createInstructionCombiningPass() {
  return new InstCombiner();
}
