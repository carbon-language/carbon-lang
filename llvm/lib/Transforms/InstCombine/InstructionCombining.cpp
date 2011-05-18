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
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/PatternMatch.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm-c/Initialization.h"
#include <algorithm>
#include <climits>
using namespace llvm;
using namespace llvm::PatternMatch;

STATISTIC(NumCombined , "Number of insts combined");
STATISTIC(NumConstProp, "Number of constant folds");
STATISTIC(NumDeadInst , "Number of dead inst eliminated");
STATISTIC(NumSunkInst , "Number of instructions sunk");
STATISTIC(NumExpand,    "Number of expansions");
STATISTIC(NumFactor   , "Number of factorizations");
STATISTIC(NumReassoc  , "Number of reassociations");

// Initialization Routines
void llvm::initializeInstCombine(PassRegistry &Registry) {
  initializeInstCombinerPass(Registry);
}

void LLVMInitializeInstCombine(LLVMPassRegistryRef R) {
  initializeInstCombine(*unwrap(R));
}

char InstCombiner::ID = 0;
INITIALIZE_PASS(InstCombiner, "instcombine",
                "Combine redundant instructions", false, false)

void InstCombiner::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
}


/// ShouldChangeType - Return true if it is desirable to convert a computation
/// from 'From' to 'To'.  We don't want to convert from a legal to an illegal
/// type for example, or from a smaller to a larger illegal type.
bool InstCombiner::ShouldChangeType(const Type *From, const Type *To) const {
  assert(From->isIntegerTy() && To->isIntegerTy());
  
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


/// SimplifyAssociativeOrCommutative - This performs a few simplifications for
/// operators which are associative or commutative:
//
//  Commutative operators:
//
//  1. Order operands such that they are listed from right (least complex) to
//     left (most complex).  This puts constants before unary operators before
//     binary operators.
//
//  Associative operators:
//
//  2. Transform: "(A op B) op C" ==> "A op (B op C)" if "B op C" simplifies.
//  3. Transform: "A op (B op C)" ==> "(A op B) op C" if "A op B" simplifies.
//
//  Associative and commutative operators:
//
//  4. Transform: "(A op B) op C" ==> "(C op A) op B" if "C op A" simplifies.
//  5. Transform: "A op (B op C)" ==> "B op (C op A)" if "C op A" simplifies.
//  6. Transform: "(A op C1) op (B op C2)" ==> "(A op B) op (C1 op C2)"
//     if C1 and C2 are constants.
//
bool InstCombiner::SimplifyAssociativeOrCommutative(BinaryOperator &I) {
  Instruction::BinaryOps Opcode = I.getOpcode();
  bool Changed = false;

  do {
    // Order operands such that they are listed from right (least complex) to
    // left (most complex).  This puts constants before unary operators before
    // binary operators.
    if (I.isCommutative() && getComplexity(I.getOperand(0)) <
        getComplexity(I.getOperand(1)))
      Changed = !I.swapOperands();

    BinaryOperator *Op0 = dyn_cast<BinaryOperator>(I.getOperand(0));
    BinaryOperator *Op1 = dyn_cast<BinaryOperator>(I.getOperand(1));

    if (I.isAssociative()) {
      // Transform: "(A op B) op C" ==> "A op (B op C)" if "B op C" simplifies.
      if (Op0 && Op0->getOpcode() == Opcode) {
        Value *A = Op0->getOperand(0);
        Value *B = Op0->getOperand(1);
        Value *C = I.getOperand(1);

        // Does "B op C" simplify?
        if (Value *V = SimplifyBinOp(Opcode, B, C, TD)) {
          // It simplifies to V.  Form "A op V".
          I.setOperand(0, A);
          I.setOperand(1, V);
          // Conservatively clear the optional flags, since they may not be
          // preserved by the reassociation.
          I.clearSubclassOptionalData();
          Changed = true;
          ++NumReassoc;
          continue;
        }
      }

      // Transform: "A op (B op C)" ==> "(A op B) op C" if "A op B" simplifies.
      if (Op1 && Op1->getOpcode() == Opcode) {
        Value *A = I.getOperand(0);
        Value *B = Op1->getOperand(0);
        Value *C = Op1->getOperand(1);

        // Does "A op B" simplify?
        if (Value *V = SimplifyBinOp(Opcode, A, B, TD)) {
          // It simplifies to V.  Form "V op C".
          I.setOperand(0, V);
          I.setOperand(1, C);
          // Conservatively clear the optional flags, since they may not be
          // preserved by the reassociation.
          I.clearSubclassOptionalData();
          Changed = true;
          ++NumReassoc;
          continue;
        }
      }
    }

    if (I.isAssociative() && I.isCommutative()) {
      // Transform: "(A op B) op C" ==> "(C op A) op B" if "C op A" simplifies.
      if (Op0 && Op0->getOpcode() == Opcode) {
        Value *A = Op0->getOperand(0);
        Value *B = Op0->getOperand(1);
        Value *C = I.getOperand(1);

        // Does "C op A" simplify?
        if (Value *V = SimplifyBinOp(Opcode, C, A, TD)) {
          // It simplifies to V.  Form "V op B".
          I.setOperand(0, V);
          I.setOperand(1, B);
          // Conservatively clear the optional flags, since they may not be
          // preserved by the reassociation.
          I.clearSubclassOptionalData();
          Changed = true;
          ++NumReassoc;
          continue;
        }
      }

      // Transform: "A op (B op C)" ==> "B op (C op A)" if "C op A" simplifies.
      if (Op1 && Op1->getOpcode() == Opcode) {
        Value *A = I.getOperand(0);
        Value *B = Op1->getOperand(0);
        Value *C = Op1->getOperand(1);

        // Does "C op A" simplify?
        if (Value *V = SimplifyBinOp(Opcode, C, A, TD)) {
          // It simplifies to V.  Form "B op V".
          I.setOperand(0, B);
          I.setOperand(1, V);
          // Conservatively clear the optional flags, since they may not be
          // preserved by the reassociation.
          I.clearSubclassOptionalData();
          Changed = true;
          ++NumReassoc;
          continue;
        }
      }

      // Transform: "(A op C1) op (B op C2)" ==> "(A op B) op (C1 op C2)"
      // if C1 and C2 are constants.
      if (Op0 && Op1 &&
          Op0->getOpcode() == Opcode && Op1->getOpcode() == Opcode &&
          isa<Constant>(Op0->getOperand(1)) &&
          isa<Constant>(Op1->getOperand(1)) &&
          Op0->hasOneUse() && Op1->hasOneUse()) {
        Value *A = Op0->getOperand(0);
        Constant *C1 = cast<Constant>(Op0->getOperand(1));
        Value *B = Op1->getOperand(0);
        Constant *C2 = cast<Constant>(Op1->getOperand(1));

        Constant *Folded = ConstantExpr::get(Opcode, C1, C2);
        Instruction *New = BinaryOperator::Create(Opcode, A, B, Op1->getName(),
                                                  &I);
        Worklist.Add(New);
        I.setOperand(0, New);
        I.setOperand(1, Folded);
        // Conservatively clear the optional flags, since they may not be
        // preserved by the reassociation.
        I.clearSubclassOptionalData();
        Changed = true;
        continue;
      }
    }

    // No further simplifications.
    return Changed;
  } while (1);
}

/// LeftDistributesOverRight - Whether "X LOp (Y ROp Z)" is always equal to
/// "(X LOp Y) ROp (X LOp Z)".
static bool LeftDistributesOverRight(Instruction::BinaryOps LOp,
                                     Instruction::BinaryOps ROp) {
  switch (LOp) {
  default:
    return false;

  case Instruction::And:
    // And distributes over Or and Xor.
    switch (ROp) {
    default:
      return false;
    case Instruction::Or:
    case Instruction::Xor:
      return true;
    }

  case Instruction::Mul:
    // Multiplication distributes over addition and subtraction.
    switch (ROp) {
    default:
      return false;
    case Instruction::Add:
    case Instruction::Sub:
      return true;
    }

  case Instruction::Or:
    // Or distributes over And.
    switch (ROp) {
    default:
      return false;
    case Instruction::And:
      return true;
    }
  }
}

/// RightDistributesOverLeft - Whether "(X LOp Y) ROp Z" is always equal to
/// "(X ROp Z) LOp (Y ROp Z)".
static bool RightDistributesOverLeft(Instruction::BinaryOps LOp,
                                     Instruction::BinaryOps ROp) {
  if (Instruction::isCommutative(ROp))
    return LeftDistributesOverRight(ROp, LOp);
  // TODO: It would be nice to handle division, aka "(X + Y)/Z = X/Z + Y/Z",
  // but this requires knowing that the addition does not overflow and other
  // such subtleties.
  return false;
}

/// SimplifyUsingDistributiveLaws - This tries to simplify binary operations
/// which some other binary operation distributes over either by factorizing
/// out common terms (eg "(A*B)+(A*C)" -> "A*(B+C)") or expanding out if this
/// results in simplifications (eg: "A & (B | C) -> (A&B) | (A&C)" if this is
/// a win).  Returns the simplified value, or null if it didn't simplify.
Value *InstCombiner::SimplifyUsingDistributiveLaws(BinaryOperator &I) {
  Value *LHS = I.getOperand(0), *RHS = I.getOperand(1);
  BinaryOperator *Op0 = dyn_cast<BinaryOperator>(LHS);
  BinaryOperator *Op1 = dyn_cast<BinaryOperator>(RHS);
  Instruction::BinaryOps TopLevelOpcode = I.getOpcode(); // op

  // Factorization.
  if (Op0 && Op1 && Op0->getOpcode() == Op1->getOpcode()) {
    // The instruction has the form "(A op' B) op (C op' D)".  Try to factorize
    // a common term.
    Value *A = Op0->getOperand(0), *B = Op0->getOperand(1);
    Value *C = Op1->getOperand(0), *D = Op1->getOperand(1);
    Instruction::BinaryOps InnerOpcode = Op0->getOpcode(); // op'

    // Does "X op' Y" always equal "Y op' X"?
    bool InnerCommutative = Instruction::isCommutative(InnerOpcode);

    // Does "X op' (Y op Z)" always equal "(X op' Y) op (X op' Z)"?
    if (LeftDistributesOverRight(InnerOpcode, TopLevelOpcode))
      // Does the instruction have the form "(A op' B) op (A op' D)" or, in the
      // commutative case, "(A op' B) op (C op' A)"?
      if (A == C || (InnerCommutative && A == D)) {
        if (A != C)
          std::swap(C, D);
        // Consider forming "A op' (B op D)".
        // If "B op D" simplifies then it can be formed with no cost.
        Value *V = SimplifyBinOp(TopLevelOpcode, B, D, TD);
        // If "B op D" doesn't simplify then only go on if both of the existing
        // operations "A op' B" and "C op' D" will be zapped as no longer used.
        if (!V && Op0->hasOneUse() && Op1->hasOneUse())
          V = Builder->CreateBinOp(TopLevelOpcode, B, D, Op1->getName());
        if (V) {
          ++NumFactor;
          V = Builder->CreateBinOp(InnerOpcode, A, V);
          V->takeName(&I);
          return V;
        }
      }

    // Does "(X op Y) op' Z" always equal "(X op' Z) op (Y op' Z)"?
    if (RightDistributesOverLeft(TopLevelOpcode, InnerOpcode))
      // Does the instruction have the form "(A op' B) op (C op' B)" or, in the
      // commutative case, "(A op' B) op (B op' D)"?
      if (B == D || (InnerCommutative && B == C)) {
        if (B != D)
          std::swap(C, D);
        // Consider forming "(A op C) op' B".
        // If "A op C" simplifies then it can be formed with no cost.
        Value *V = SimplifyBinOp(TopLevelOpcode, A, C, TD);
        // If "A op C" doesn't simplify then only go on if both of the existing
        // operations "A op' B" and "C op' D" will be zapped as no longer used.
        if (!V && Op0->hasOneUse() && Op1->hasOneUse())
          V = Builder->CreateBinOp(TopLevelOpcode, A, C, Op0->getName());
        if (V) {
          ++NumFactor;
          V = Builder->CreateBinOp(InnerOpcode, V, B);
          V->takeName(&I);
          return V;
        }
      }
  }

  // Expansion.
  if (Op0 && RightDistributesOverLeft(Op0->getOpcode(), TopLevelOpcode)) {
    // The instruction has the form "(A op' B) op C".  See if expanding it out
    // to "(A op C) op' (B op C)" results in simplifications.
    Value *A = Op0->getOperand(0), *B = Op0->getOperand(1), *C = RHS;
    Instruction::BinaryOps InnerOpcode = Op0->getOpcode(); // op'

    // Do "A op C" and "B op C" both simplify?
    if (Value *L = SimplifyBinOp(TopLevelOpcode, A, C, TD))
      if (Value *R = SimplifyBinOp(TopLevelOpcode, B, C, TD)) {
        // They do! Return "L op' R".
        ++NumExpand;
        // If "L op' R" equals "A op' B" then "L op' R" is just the LHS.
        if ((L == A && R == B) ||
            (Instruction::isCommutative(InnerOpcode) && L == B && R == A))
          return Op0;
        // Otherwise return "L op' R" if it simplifies.
        if (Value *V = SimplifyBinOp(InnerOpcode, L, R, TD))
          return V;
        // Otherwise, create a new instruction.
        C = Builder->CreateBinOp(InnerOpcode, L, R);
        C->takeName(&I);
        return C;
      }
  }

  if (Op1 && LeftDistributesOverRight(TopLevelOpcode, Op1->getOpcode())) {
    // The instruction has the form "A op (B op' C)".  See if expanding it out
    // to "(A op B) op' (A op C)" results in simplifications.
    Value *A = LHS, *B = Op1->getOperand(0), *C = Op1->getOperand(1);
    Instruction::BinaryOps InnerOpcode = Op1->getOpcode(); // op'

    // Do "A op B" and "A op C" both simplify?
    if (Value *L = SimplifyBinOp(TopLevelOpcode, A, B, TD))
      if (Value *R = SimplifyBinOp(TopLevelOpcode, A, C, TD)) {
        // They do! Return "L op' R".
        ++NumExpand;
        // If "L op' R" equals "B op' C" then "L op' R" is just the RHS.
        if ((L == B && R == C) ||
            (Instruction::isCommutative(InnerOpcode) && L == C && R == B))
          return Op1;
        // Otherwise return "L op' R" if it simplifies.
        if (Value *V = SimplifyBinOp(InnerOpcode, L, R, TD))
          return V;
        // Otherwise, create a new instruction.
        A = Builder->CreateBinOp(InnerOpcode, L, R);
        A->takeName(&I);
        return A;
      }
  }

  return 0;
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
    if (C->getType()->getElementType()->isIntegerTy())
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
    if (C->getType()->getElementType()->isFloatingPointTy())
      return ConstantExpr::getFNeg(C);

  return 0;
}

static Value *FoldOperationIntoSelectOperand(Instruction &I, Value *SO,
                                             InstCombiner *IC) {
  if (CastInst *CI = dyn_cast<CastInst>(&I)) {
    return IC->Builder->CreateCast(CI->getOpcode(), SO, I.getType());
  }

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
    if (SI->getType()->isIntegerTy(1)) return 0;

    // If it's a bitcast involving vectors, make sure it has the same number of
    // elements on both sides.
    if (BitCastInst *BC = dyn_cast<BitCastInst>(&Op)) {
      const VectorType *DestTy = dyn_cast<VectorType>(BC->getDestTy());
      const VectorType *SrcTy = dyn_cast<VectorType>(BC->getSrcTy());

      // Verify that either both or neither are vectors.
      if ((SrcTy == NULL) != (DestTy == NULL)) return 0;
      // If vectors, verify that they have the same number of elements.
      if (SrcTy && SrcTy->getNumElements() != DestTy->getNumElements())
        return 0;
    }
    
    Value *SelectTrueVal = FoldOperationIntoSelectOperand(Op, TV, this);
    Value *SelectFalseVal = FoldOperationIntoSelectOperand(Op, FV, this);

    return SelectInst::Create(SI->getCondition(),
                              SelectTrueVal, SelectFalseVal);
  }
  return 0;
}


/// FoldOpIntoPhi - Given a binary operator, cast instruction, or select which
/// has a PHI node as operand #0, see if we can fold the instruction into the
/// PHI (which is only possible if all operands to the PHI are constants).
///
Instruction *InstCombiner::FoldOpIntoPhi(Instruction &I) {
  PHINode *PN = cast<PHINode>(I.getOperand(0));
  unsigned NumPHIValues = PN->getNumIncomingValues();
  if (NumPHIValues == 0)
    return 0;
  
  // We normally only transform phis with a single use.  However, if a PHI has
  // multiple uses and they are all the same operation, we can fold *all* of the
  // uses into the PHI.
  if (!PN->hasOneUse()) {
    // Walk the use list for the instruction, comparing them to I.
    for (Value::use_iterator UI = PN->use_begin(), E = PN->use_end();
         UI != E; ++UI) {
      Instruction *User = cast<Instruction>(*UI);
      if (User != &I && !I.isIdenticalTo(User))
        return 0;
    }
    // Otherwise, we can replace *all* users with the new PHI we form.
  }
  
  // Check to see if all of the operands of the PHI are simple constants
  // (constantint/constantfp/undef).  If there is one non-constant value,
  // remember the BB it is in.  If there is more than one or if *it* is a PHI,
  // bail out.  We don't do arbitrary constant expressions here because moving
  // their computation can be expensive without a cost model.
  BasicBlock *NonConstBB = 0;
  for (unsigned i = 0; i != NumPHIValues; ++i) {
    Value *InVal = PN->getIncomingValue(i);
    if (isa<Constant>(InVal) && !isa<ConstantExpr>(InVal))
      continue;

    if (isa<PHINode>(InVal)) return 0;  // Itself a phi.
    if (NonConstBB) return 0;  // More than one non-const value.
    
    NonConstBB = PN->getIncomingBlock(i);

    // If the InVal is an invoke at the end of the pred block, then we can't
    // insert a computation after it without breaking the edge.
    if (InvokeInst *II = dyn_cast<InvokeInst>(InVal))
      if (II->getParent() == NonConstBB)
        return 0;
    
    // If the incoming non-constant value is in I's block, we will remove one
    // instruction, but insert another equivalent one, leading to infinite
    // instcombine.
    if (NonConstBB == I.getParent())
      return 0;
  }
  
  // If there is exactly one non-constant value, we can insert a copy of the
  // operation in that block.  However, if this is a critical edge, we would be
  // inserting the computation one some other paths (e.g. inside a loop).  Only
  // do this if the pred block is unconditionally branching into the phi block.
  if (NonConstBB != 0) {
    BranchInst *BI = dyn_cast<BranchInst>(NonConstBB->getTerminator());
    if (!BI || !BI->isUnconditional()) return 0;
  }

  // Okay, we can do the transformation: create the new PHI node.
  PHINode *NewPN = PHINode::Create(I.getType(), PN->getNumIncomingValues(), "");
  InsertNewInstBefore(NewPN, *PN);
  NewPN->takeName(PN);
  
  // If we are going to have to insert a new computation, do so right before the
  // predecessors terminator.
  if (NonConstBB)
    Builder->SetInsertPoint(NonConstBB->getTerminator());
  
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
      if (Constant *InC = dyn_cast<Constant>(PN->getIncomingValue(i)))
        InV = InC->isNullValue() ? FalseVInPred : TrueVInPred;
      else
        InV = Builder->CreateSelect(PN->getIncomingValue(i),
                                    TrueVInPred, FalseVInPred, "phitmp");
      NewPN->addIncoming(InV, ThisBB);
    }
  } else if (CmpInst *CI = dyn_cast<CmpInst>(&I)) {
    Constant *C = cast<Constant>(I.getOperand(1));
    for (unsigned i = 0; i != NumPHIValues; ++i) {
      Value *InV = 0;
      if (Constant *InC = dyn_cast<Constant>(PN->getIncomingValue(i)))
        InV = ConstantExpr::getCompare(CI->getPredicate(), InC, C);
      else if (isa<ICmpInst>(CI))
        InV = Builder->CreateICmp(CI->getPredicate(), PN->getIncomingValue(i),
                                  C, "phitmp");
      else
        InV = Builder->CreateFCmp(CI->getPredicate(), PN->getIncomingValue(i),
                                  C, "phitmp");
      NewPN->addIncoming(InV, PN->getIncomingBlock(i));
    }
  } else if (I.getNumOperands() == 2) {
    Constant *C = cast<Constant>(I.getOperand(1));
    for (unsigned i = 0; i != NumPHIValues; ++i) {
      Value *InV = 0;
      if (Constant *InC = dyn_cast<Constant>(PN->getIncomingValue(i)))
        InV = ConstantExpr::get(I.getOpcode(), InC, C);
      else
        InV = Builder->CreateBinOp(cast<BinaryOperator>(I).getOpcode(),
                                   PN->getIncomingValue(i), C, "phitmp");
      NewPN->addIncoming(InV, PN->getIncomingBlock(i));
    }
  } else { 
    CastInst *CI = cast<CastInst>(&I);
    const Type *RetTy = CI->getType();
    for (unsigned i = 0; i != NumPHIValues; ++i) {
      Value *InV;
      if (Constant *InC = dyn_cast<Constant>(PN->getIncomingValue(i)))
        InV = ConstantExpr::getCast(CI->getOpcode(), InC, RetTy);
      else 
        InV = Builder->CreateCast(CI->getOpcode(),
                                PN->getIncomingValue(i), I.getType(), "phitmp");
      NewPN->addIncoming(InV, PN->getIncomingBlock(i));
    }
  }
  
  for (Value::use_iterator UI = PN->use_begin(), E = PN->use_end();
       UI != E; ) {
    Instruction *User = cast<Instruction>(*UI++);
    if (User == &I) continue;
    ReplaceInstUsesWith(*User, NewPN);
    EraseInstFromFunction(*User);
  }
  return ReplaceInstUsesWith(I, NewPN);
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

  // Eliminate unneeded casts for indices, and replace indices which displace
  // by multiples of a zero size type with zero.
  if (TD) {
    bool MadeChange = false;
    const Type *IntPtrTy = TD->getIntPtrType(GEP.getContext());

    gep_type_iterator GTI = gep_type_begin(GEP);
    for (User::op_iterator I = GEP.op_begin() + 1, E = GEP.op_end();
         I != E; ++I, ++GTI) {
      // Skip indices into struct types.
      const SequentialType *SeqTy = dyn_cast<SequentialType>(*GTI);
      if (!SeqTy) continue;

      // If the element type has zero size then any index over it is equivalent
      // to an index of zero, so replace it with zero if it is not zero already.
      if (SeqTy->getElementType()->isSized() &&
          TD->getTypeAllocSize(SeqTy->getElementType()) == 0)
        if (!isa<Constant>(*I) || !cast<Constant>(*I)->isNullValue()) {
          *I = Constant::getNullValue(IntPtrTy);
          MadeChange = true;
        }

      if ((*I)->getType() != IntPtrTy) {
        // If we are using a wider index than needed for this platform, shrink
        // it to what we need.  If narrower, sign-extend it to what we need.
        // This explicit cast can make subsequent optimizations more obvious.
        *I = Builder->CreateIntCast(*I, IntPtrTy, true);
        MadeChange = true;
      }
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
      EndsWithSequential = !(*I)->isStructTy();

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
      return (GEP.isInBounds() && Src->isInBounds()) ?
        GetElementPtrInst::CreateInBounds(Src->getOperand(0), Indices.begin(),
                                          Indices.end(), GEP.getName()) :
        GetElementPtrInst::Create(Src->getOperand(0), Indices.begin(),
                                  Indices.end(), GEP.getName());
  }

  // Handle gep(bitcast x) and gep(gep x, 0, 0, 0).
  Value *StrippedPtr = PtrOp->stripPointerCasts();
  const PointerType *StrippedPtrTy =cast<PointerType>(StrippedPtr->getType());
  if (StrippedPtr != PtrOp &&
    StrippedPtrTy->getAddressSpace() == GEP.getPointerAddressSpace()) {

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
      if (const ArrayType *CATy =
          dyn_cast<ArrayType>(CPTy->getElementType())) {
        // GEP (bitcast i8* X to [0 x i8]*), i32 0, ... ?
        if (CATy->getElementType() == StrippedPtrTy->getElementType()) {
          // -> GEP i8* X, ...
          SmallVector<Value*, 8> Idx(GEP.idx_begin()+1, GEP.idx_end());
          GetElementPtrInst *Res =
            GetElementPtrInst::Create(StrippedPtr, Idx.begin(),
                                      Idx.end(), GEP.getName());
          Res->setIsInBounds(GEP.isInBounds());
          return Res;
        }
        
        if (const ArrayType *XATy =
              dyn_cast<ArrayType>(StrippedPtrTy->getElementType())){
          // GEP (bitcast [10 x i8]* X to [0 x i8]*), i32 0, ... ?
          if (CATy->getElementType() == XATy->getElementType()) {
            // -> GEP [10 x i8]* X, i32 0, ...
            // At this point, we know that the cast source type is a pointer
            // to an array of the same type as the destination pointer
            // array.  Because the array type is never stepped over (there
            // is a leading zero) we can fold the cast into this GEP.
            GEP.setOperand(0, StrippedPtr);
            return &GEP;
          }
        }
      }
    } else if (GEP.getNumOperands() == 2) {
      // Transform things like:
      // %t = getelementptr i32* bitcast ([2 x i32]* %str to i32*), i32 %V
      // into:  %t1 = getelementptr [2 x i32]* %str, i32 0, i32 %V; bitcast
      const Type *SrcElTy = StrippedPtrTy->getElementType();
      const Type *ResElTy=cast<PointerType>(PtrOp->getType())->getElementType();
      if (TD && SrcElTy->isArrayTy() &&
          TD->getTypeAllocSize(cast<ArrayType>(SrcElTy)->getElementType()) ==
          TD->getTypeAllocSize(ResElTy)) {
        Value *Idx[2];
        Idx[0] = Constant::getNullValue(Type::getInt32Ty(GEP.getContext()));
        Idx[1] = GEP.getOperand(1);
        Value *NewGEP = GEP.isInBounds() ?
          Builder->CreateInBoundsGEP(StrippedPtr, Idx, Idx + 2, GEP.getName()) :
          Builder->CreateGEP(StrippedPtr, Idx, Idx + 2, GEP.getName());
        // V and GEP are both pointer types --> BitCast
        return new BitCastInst(NewGEP, GEP.getType());
      }
      
      // Transform things like:
      // getelementptr i8* bitcast ([100 x double]* X to i8*), i32 %tmp
      //   (where tmp = 8*tmp2) into:
      // getelementptr [100 x double]* %arr, i32 0, i32 %tmp2; bitcast
      
      if (TD && SrcElTy->isArrayTy() && ResElTy->isIntegerTy(8)) {
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
          Value *NewGEP = GEP.isInBounds() ?
            Builder->CreateInBoundsGEP(StrippedPtr, Idx, Idx + 2,GEP.getName()):
            Builder->CreateGEP(StrippedPtr, Idx, Idx + 2, GEP.getName());
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
        !isa<BitCastInst>(BCI->getOperand(0)) && GEP.hasAllConstantIndices() &&
        StrippedPtrTy->getAddressSpace() == GEP.getPointerAddressSpace()) {

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
        Value *NGEP = GEP.isInBounds() ?
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



static bool IsOnlyNullComparedAndFreed(const Value &V) {
  for (Value::const_use_iterator UI = V.use_begin(), UE = V.use_end();
       UI != UE; ++UI) {
    const User *U = *UI;
    if (isFreeCall(U))
      continue;
    if (const ICmpInst *ICI = dyn_cast<ICmpInst>(U))
      if (ICI->isEquality() && isa<ConstantPointerNull>(ICI->getOperand(1)))
        continue;
    return false;
  }
  return true;
}

Instruction *InstCombiner::visitMalloc(Instruction &MI) {
  // If we have a malloc call which is only used in any amount of comparisons
  // to null and free calls, delete the calls and replace the comparisons with
  // true or false as appropriate.
  if (IsOnlyNullComparedAndFreed(MI)) {
    for (Value::use_iterator UI = MI.use_begin(), UE = MI.use_end();
         UI != UE;) {
      // We can assume that every remaining use is a free call or an icmp eq/ne
      // to null, so the cast is safe.
      Instruction *I = cast<Instruction>(*UI);

      // Early increment here, as we're about to get rid of the user.
      ++UI;

      if (isFreeCall(I)) {
        EraseInstFromFunction(*cast<CallInst>(I));
        continue;
      }
      // Again, the cast is safe.
      ICmpInst *C = cast<ICmpInst>(I);
      ReplaceInstUsesWith(*C, ConstantInt::get(Type::getInt1Ty(C->getContext()),
                                               C->isFalseWhenEqual()));
      EraseInstFromFunction(*C);
    }
    return EraseInstFromFunction(MI);
  }
  return 0;
}



Instruction *InstCombiner::visitFree(CallInst &FI) {
  Value *Op = FI.getArgOperand(0);

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
    // just get one value.
    if (II->hasOneUse()) {
      // Check if we're grabbing the overflow bit or the result of a 'with
      // overflow' intrinsic.  If it's the latter we can remove the intrinsic
      // and replace it with a traditional binary instruction.
      switch (II->getIntrinsicID()) {
      case Intrinsic::uadd_with_overflow:
      case Intrinsic::sadd_with_overflow:
        if (*EV.idx_begin() == 0) {  // Normal result.
          Value *LHS = II->getArgOperand(0), *RHS = II->getArgOperand(1);
          ReplaceInstUsesWith(*II, UndefValue::get(II->getType()));
          EraseInstFromFunction(*II);
          return BinaryOperator::CreateAdd(LHS, RHS);
        }
          
        // If the normal result of the add is dead, and the RHS is a constant,
        // we can transform this into a range comparison.
        // overflow = uadd a, -4  -->  overflow = icmp ugt a, 3
        if (II->getIntrinsicID() == Intrinsic::uadd_with_overflow)
          if (ConstantInt *CI = dyn_cast<ConstantInt>(II->getArgOperand(1)))
            return new ICmpInst(ICmpInst::ICMP_UGT, II->getArgOperand(0),
                                ConstantExpr::getNot(CI));
        break;
      case Intrinsic::usub_with_overflow:
      case Intrinsic::ssub_with_overflow:
        if (*EV.idx_begin() == 0) {  // Normal result.
          Value *LHS = II->getArgOperand(0), *RHS = II->getArgOperand(1);
          ReplaceInstUsesWith(*II, UndefValue::get(II->getType()));
          EraseInstFromFunction(*II);
          return BinaryOperator::CreateSub(LHS, RHS);
        }
        break;
      case Intrinsic::umul_with_overflow:
      case Intrinsic::smul_with_overflow:
        if (*EV.idx_begin() == 0) {  // Normal result.
          Value *LHS = II->getArgOperand(0), *RHS = II->getArgOperand(1);
          ReplaceInstUsesWith(*II, UndefValue::get(II->getType()));
          EraseInstFromFunction(*II);
          return BinaryOperator::CreateMul(LHS, RHS);
        }
        break;
      default:
        break;
      }
    }
  }
  if (LoadInst *L = dyn_cast<LoadInst>(Agg))
    // If the (non-volatile) load only has one use, we can rewrite this to a
    // load from a GEP. This reduces the size of the load.
    // FIXME: If a load is used only by extractvalue instructions then this
    //        could be done regardless of having multiple uses.
    if (!L->isVolatile() && L->hasOneUse()) {
      // extractvalue has integer indices, getelementptr has Value*s. Convert.
      SmallVector<Value*, 4> Indices;
      // Prefix an i32 0 since we need the first element.
      Indices.push_back(Builder->getInt32(0));
      for (ExtractValueInst::idx_iterator I = EV.idx_begin(), E = EV.idx_end();
            I != E; ++I)
        Indices.push_back(Builder->getInt32(*I));

      // We need to insert these at the location of the old load, not at that of
      // the extractvalue.
      Builder->SetInsertPoint(L->getParent(), L);
      Value *GEP = Builder->CreateInBoundsGEP(L->getPointerOperand(),
                                              Indices.begin(), Indices.end());
      // Returning the load directly will cause the main loop to insert it in
      // the wrong spot, so use ReplaceInstUsesWith().
      return ReplaceInstUsesWith(EV, Builder->CreateLoad(GEP));
    }
  // We could simplify extracts from other values. Note that nested extracts may
  // already be simplified implicitly by the above: extract (extract (insert) )
  // will be translated into extract ( insert ( extract ) ) first and then just
  // the value inserted, if appropriate. Similarly for extracts from single-use
  // loads: extract (extract (load)) will be translated to extract (load (gep))
  // and if again single-use then via load (gep (gep)) to load (gep).
  // However, double extracts from e.g. function arguments or return values
  // aren't handled yet.
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

  SmallVector<Instruction*, 128> InstrsForInstCombineWorklist;
  SmallPtrSet<ConstantExpr*, 64> FoldedConstants;
  
  do {
    BB = Worklist.pop_back_val();
    
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
  } while (!Worklist.empty());
  
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
    Builder->SetCurrentDebugLocation(I->getDebugLoc());
    
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

        Result->setDebugLoc(I->getDebugLoc());
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
  TD = getAnalysisIfAvailable<TargetData>();

  
  /// Builder - This is an IRBuilder that automatically inserts new
  /// instructions into the worklist when they are created.
  IRBuilder<true, TargetFolder, InstCombineIRInserter> 
    TheBuilder(F.getContext(), TargetFolder(TD),
               InstCombineIRInserter(Worklist));
  Builder = &TheBuilder;
  
  bool EverMadeChange = false;

  // Lower dbg.declare intrinsics otherwise their value may be clobbered
  // by instcombiner.
  EverMadeChange = LowerDbgDeclare(F);

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
