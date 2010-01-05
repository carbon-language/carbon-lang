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
STATISTIC(NumSunkInst , "Number of instructions sunk");


char InstCombiner::ID = 0;
static RegisterPass<InstCombiner>
X("instcombine", "Combine redundant instructions");

void InstCombiner::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addPreservedID(LCSSAID);
  AU.setPreservesCFG();
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

// FIXME: Value::stripPointerCasts
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
      }
      
      if (BinaryOperator *Op1 = dyn_cast<BinaryOperator>(I.getOperand(1)))
        if (Op1->getOpcode() == Opcode && isa<Constant>(Op1->getOperand(1)) &&
            Op->hasOneUse() && Op1->hasOneUse()) {
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
Value *InstCombiner::dyn_castFNegVal(Value *V) const {
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



/// AddOne - Add one to a ConstantInt.
static Constant *AddOne(Constant *C) {
  return ConstantExpr::getAdd(C, ConstantInt::get(C->getType(), 1));
}
/// SubOne - Subtract one from a ConstantInt.
static Constant *SubOne(ConstantInt *C) {
  return ConstantInt::get(C->getContext(), C->getValue()-1);
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
static Value *getICmpValue(bool Sign, unsigned Code, Value *LHS, Value *RHS) {
  switch (Code) {
  default: assert(0 && "Illegal ICmp code!");
  case 0:
    return ConstantInt::getFalse(LHS->getContext());
  case 1: 
    if (Sign)
      return new ICmpInst(ICmpInst::ICMP_SGT, LHS, RHS);
    return new ICmpInst(ICmpInst::ICMP_UGT, LHS, RHS);
  case 2:
    return new ICmpInst(ICmpInst::ICMP_EQ,  LHS, RHS);
  case 3: 
    if (Sign)
      return new ICmpInst(ICmpInst::ICMP_SGE, LHS, RHS);
    return new ICmpInst(ICmpInst::ICMP_UGE, LHS, RHS);
  case 4: 
    if (Sign)
      return new ICmpInst(ICmpInst::ICMP_SLT, LHS, RHS);
    return new ICmpInst(ICmpInst::ICMP_ULT, LHS, RHS);
  case 5:
    return new ICmpInst(ICmpInst::ICMP_NE,  LHS, RHS);
  case 6: 
    if (Sign)
      return new ICmpInst(ICmpInst::ICMP_SLE, LHS, RHS);
    return new ICmpInst(ICmpInst::ICMP_ULE, LHS, RHS);
  case 7:
    return ConstantInt::getTrue(LHS->getContext());
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
      const APInt &AndRHSV = cast<ConstantInt>(AndRHS)->getValue();

      // If there is only one bit set.
      if (AndRHSV.isPowerOf2()) {
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
  ICmpInst::Predicate LHSCC = LHS->getPredicate(), RHSCC = RHS->getPredicate();

  // (icmp1 A, B) & (icmp2 A, B) --> (icmp3 A, B)
  if (PredicatesFoldable(LHSCC, RHSCC)) {
    if (LHS->getOperand(0) == RHS->getOperand(1) &&
        LHS->getOperand(1) == RHS->getOperand(0))
      LHS->swapOperands();
    if (LHS->getOperand(0) == RHS->getOperand(0) &&
        LHS->getOperand(1) == RHS->getOperand(1)) {
      Value *Op0 = LHS->getOperand(0), *Op1 = LHS->getOperand(1);
      unsigned Code = getICmpCode(LHS) & getICmpCode(RHS);
      bool isSigned = LHS->isSigned() || RHS->isSigned();
      Value *RV = getICmpValue(isSigned, Code, Op0, Op1);
      if (Instruction *I = dyn_cast<Instruction>(RV))
        return I;
      // Otherwise, it's a constant boolean value.
      return ReplaceInstUsesWith(I, RV);
    }
  }
  
  // This only handles icmp of constants: (icmp1 A, C1) & (icmp2 B, C2).
  Value *Val = LHS->getOperand(0), *Val2 = RHS->getOperand(0);
  ConstantInt *LHSCst = dyn_cast<ConstantInt>(LHS->getOperand(1));
  ConstantInt *RHSCst = dyn_cast<ConstantInt>(RHS->getOperand(1));
  if (LHSCst == 0 || RHSCst == 0) return 0;
  
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
  // (from the icmp folding check above), that the two constants 
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
  
  if (ICmpInst *RHS = dyn_cast<ICmpInst>(Op1))
    if (ICmpInst *LHS = dyn_cast<ICmpInst>(Op0))
      if (Instruction *Res = FoldAndOfICmps(I, LHS, RHS))
        return Res;

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
  ICmpInst::Predicate LHSCC = LHS->getPredicate(), RHSCC = RHS->getPredicate();

  // (icmp1 A, B) | (icmp2 A, B) --> (icmp3 A, B)
  if (PredicatesFoldable(LHSCC, RHSCC)) {
    if (LHS->getOperand(0) == RHS->getOperand(1) &&
        LHS->getOperand(1) == RHS->getOperand(0))
      LHS->swapOperands();
    if (LHS->getOperand(0) == RHS->getOperand(0) &&
        LHS->getOperand(1) == RHS->getOperand(1)) {
      Value *Op0 = LHS->getOperand(0), *Op1 = LHS->getOperand(1);
      unsigned Code = getICmpCode(LHS) | getICmpCode(RHS);
      bool isSigned = LHS->isSigned() || RHS->isSigned();
      Value *RV = getICmpValue(isSigned, Code, Op0, Op1);
      if (Instruction *I = dyn_cast<Instruction>(RV))
        return I;
      // Otherwise, it's a constant boolean value.
      return ReplaceInstUsesWith(I, RV);
    }
  }
  
  // This only handles icmp of constants: (icmp1 A, C1) | (icmp2 B, C2).
  Value *Val = LHS->getOperand(0), *Val2 = RHS->getOperand(0);
  ConstantInt *LHSCst = dyn_cast<ConstantInt>(LHS->getOperand(1));
  ConstantInt *RHSCst = dyn_cast<ConstantInt>(RHS->getOperand(1));
  if (LHSCst == 0 || RHSCst == 0) return 0;

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
  // icmp folding check above), that the two constants are not
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
        Op0->hasOneUse()) {
      Value *Or = Builder->CreateOr(X, RHS);
      Or->takeName(Op0);
      return BinaryOperator::CreateAnd(Or, 
                         ConstantInt::get(I.getContext(),
                                          RHS->getValue() | C1->getValue()));
    }

    // (X ^ C1) | C2 --> (X | C2) ^ (C1&~C2)
    if (match(Op0, m_Xor(m_Value(X), m_ConstantInt(C1))) &&
        Op0->hasOneUse()) {
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
    if (Op0->hasOneUse() || Op1->hasOneUse()) {
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

  if (ICmpInst *RHS = dyn_cast<ICmpInst>(I.getOperand(1)))
    if (ICmpInst *LHS = dyn_cast<ICmpInst>(I.getOperand(0)))
      if (Instruction *Res = FoldOrOfICmps(I, LHS, RHS))
        return Res;
    
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

  // xor X, X = 0
  if (Op0 == Op1)
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
  
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
    if (ICmpInst *LHS = dyn_cast<ICmpInst>(I.getOperand(0)))
      if (PredicatesFoldable(LHS->getPredicate(), RHS->getPredicate())) {
        if (LHS->getOperand(0) == RHS->getOperand(1) &&
            LHS->getOperand(1) == RHS->getOperand(0))
          LHS->swapOperands();
        if (LHS->getOperand(0) == RHS->getOperand(0) &&
            LHS->getOperand(1) == RHS->getOperand(1)) {
          Value *Op0 = LHS->getOperand(0), *Op1 = LHS->getOperand(1);
          unsigned Code = getICmpCode(LHS) ^ getICmpCode(RHS);
          bool isSigned = LHS->isSigned() || RHS->isSigned();
          Value *RV = getICmpValue(isSigned, Code, Op0, Op1);
          if (Instruction *I = dyn_cast<Instruction>(RV))
            return I;
          // Otherwise, it's a constant boolean value.
          return ReplaceInstUsesWith(I, RV);
        }
      }

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
