//===- InstructionCombining.cpp - Combine multiple instructions -----------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// InstructionCombining - Combine instructions to form fewer, simple
// instructions.  This pass does not modify the CFG This pass is where algebraic
// simplification happens.
//
// This pass combines things like:
//    %Y = add int 1, %X
//    %Z = add int 1, %Y
// into:
//    %Z = add int 2, %X
//
// This is a simple worklist driven algorithm.
//
// This pass guarantees that the following canonicalizations are performed on
// the program:
//    1. If a binary operator has a constant operand, it is moved to the RHS
//    2. Bitwise operators with constant operands are always grouped so that
//       shifts are performed first, then or's, then and's, then xor's.
//    3. SetCC instructions are converted from <,>,<=,>= to ==,!= if possible
//    4. All SetCC instructions on boolean values are replaced with logical ops
//    5. add X, X is represented as (X*2) => (X << 1)
//    6. Multiplies with a power-of-two constant argument are transformed into
//       shifts.
//    N. This list is incomplete
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "instcombine"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Pass.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Support/CallSite.h"
#include "Support/Debug.h"
#include "Support/Statistic.h"
#include <algorithm>
using namespace llvm;

namespace {
  Statistic<> NumCombined ("instcombine", "Number of insts combined");
  Statistic<> NumConstProp("instcombine", "Number of constant folds");
  Statistic<> NumDeadInst ("instcombine", "Number of dead inst eliminated");

  class InstCombiner : public FunctionPass,
                       public InstVisitor<InstCombiner, Instruction*> {
    // Worklist of all of the instructions that need to be simplified.
    std::vector<Instruction*> WorkList;
    TargetData *TD;

    /// AddUsersToWorkList - When an instruction is simplified, add all users of
    /// the instruction to the work lists because they might get more simplified
    /// now.
    ///
    void AddUsersToWorkList(Instruction &I) {
      for (Value::use_iterator UI = I.use_begin(), UE = I.use_end();
           UI != UE; ++UI)
        WorkList.push_back(cast<Instruction>(*UI));
    }

    /// AddUsesToWorkList - When an instruction is simplified, add operands to
    /// the work lists because they might get more simplified now.
    ///
    void AddUsesToWorkList(Instruction &I) {
      for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i)
        if (Instruction *Op = dyn_cast<Instruction>(I.getOperand(i)))
          WorkList.push_back(Op);
    }

    // removeFromWorkList - remove all instances of I from the worklist.
    void removeFromWorkList(Instruction *I);
  public:
    virtual bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<TargetData>();
      AU.setPreservesCFG();
    }

    // Visitation implementation - Implement instruction combining for different
    // instruction types.  The semantics are as follows:
    // Return Value:
    //    null        - No change was made
    //     I          - Change was made, I is still valid, I may be dead though
    //   otherwise    - Change was made, replace I with returned instruction
    //   
    Instruction *visitAdd(BinaryOperator &I);
    Instruction *visitSub(BinaryOperator &I);
    Instruction *visitMul(BinaryOperator &I);
    Instruction *visitDiv(BinaryOperator &I);
    Instruction *visitRem(BinaryOperator &I);
    Instruction *visitAnd(BinaryOperator &I);
    Instruction *visitOr (BinaryOperator &I);
    Instruction *visitXor(BinaryOperator &I);
    Instruction *visitSetCondInst(BinaryOperator &I);
    Instruction *visitShiftInst(ShiftInst &I);
    Instruction *visitCastInst(CastInst &CI);
    Instruction *visitSelectInst(SelectInst &CI);
    Instruction *visitCallInst(CallInst &CI);
    Instruction *visitInvokeInst(InvokeInst &II);
    Instruction *visitPHINode(PHINode &PN);
    Instruction *visitGetElementPtrInst(GetElementPtrInst &GEP);
    Instruction *visitAllocationInst(AllocationInst &AI);
    Instruction *visitFreeInst(FreeInst &FI);
    Instruction *visitLoadInst(LoadInst &LI);
    Instruction *visitBranchInst(BranchInst &BI);

    // visitInstruction - Specify what to return for unhandled instructions...
    Instruction *visitInstruction(Instruction &I) { return 0; }

  private:
    Instruction *visitCallSite(CallSite CS);
    bool transformConstExprCastCall(CallSite CS);

    // InsertNewInstBefore - insert an instruction New before instruction Old
    // in the program.  Add the new instruction to the worklist.
    //
    Value *InsertNewInstBefore(Instruction *New, Instruction &Old) {
      assert(New && New->getParent() == 0 &&
             "New instruction already inserted into a basic block!");
      BasicBlock *BB = Old.getParent();
      BB->getInstList().insert(&Old, New);  // Insert inst
      WorkList.push_back(New);              // Add to worklist
      return New;
    }

  public:
    // ReplaceInstUsesWith - This method is to be used when an instruction is
    // found to be dead, replacable with another preexisting expression.  Here
    // we add all uses of I to the worklist, replace all uses of I with the new
    // value, then return I, so that the inst combiner will know that I was
    // modified.
    //
    Instruction *ReplaceInstUsesWith(Instruction &I, Value *V) {
      AddUsersToWorkList(I);         // Add all modified instrs to worklist
      I.replaceAllUsesWith(V);
      return &I;
    }

    // EraseInstFromFunction - When dealing with an instruction that has side
    // effects or produces a void value, we can't rely on DCE to delete the
    // instruction.  Instead, visit methods should return the value returned by
    // this function.
    Instruction *EraseInstFromFunction(Instruction &I) {
      assert(I.use_empty() && "Cannot erase instruction that is used!");
      AddUsesToWorkList(I);
      removeFromWorkList(&I);
      I.getParent()->getInstList().erase(&I);
      return 0;  // Don't do anything with FI
    }


  private:
    /// InsertOperandCastBefore - This inserts a cast of V to DestTy before the
    /// InsertBefore instruction.  This is specialized a bit to avoid inserting
    /// casts that are known to not do anything...
    ///
    Value *InsertOperandCastBefore(Value *V, const Type *DestTy,
                                   Instruction *InsertBefore);

    // SimplifyCommutative - This performs a few simplifications for commutative
    // operators...
    bool SimplifyCommutative(BinaryOperator &I);

    Instruction *OptAndOp(Instruction *Op, ConstantIntegral *OpRHS,
                          ConstantIntegral *AndRHS, BinaryOperator &TheAnd);
  };

  RegisterOpt<InstCombiner> X("instcombine", "Combine redundant instructions");
}

// getComplexity:  Assign a complexity or rank value to LLVM Values...
//   0 -> Constant, 1 -> Other, 2 -> Argument, 2 -> Unary, 3 -> OtherInst
static unsigned getComplexity(Value *V) {
  if (isa<Instruction>(V)) {
    if (BinaryOperator::isNeg(V) || BinaryOperator::isNot(V))
      return 2;
    return 3;
  }
  if (isa<Argument>(V)) return 2;
  return isa<Constant>(V) ? 0 : 1;
}

// isOnlyUse - Return true if this instruction will be deleted if we stop using
// it.
static bool isOnlyUse(Value *V) {
  return V->hasOneUse() || isa<Constant>(V);
}

// getSignedIntegralType - Given an unsigned integral type, return the signed
// version of it that has the same size.
static const Type *getSignedIntegralType(const Type *Ty) {
  switch (Ty->getPrimitiveID()) {
  default: assert(0 && "Invalid unsigned integer type!"); abort();
  case Type::UByteTyID:  return Type::SByteTy;
  case Type::UShortTyID: return Type::ShortTy;
  case Type::UIntTyID:   return Type::IntTy;
  case Type::ULongTyID:  return Type::LongTy;
  }
}

// getUnsignedIntegralType - Given an signed integral type, return the unsigned
// version of it that has the same size.
static const Type *getUnsignedIntegralType(const Type *Ty) {
  switch (Ty->getPrimitiveID()) {
  default: assert(0 && "Invalid signed integer type!"); abort();
  case Type::SByteTyID: return Type::UByteTy;
  case Type::ShortTyID: return Type::UShortTy;
  case Type::IntTyID:   return Type::UIntTy;
  case Type::LongTyID:  return Type::ULongTy;
  }
}

// getPromotedType - Return the specified type promoted as it would be to pass
// though a va_arg area...
static const Type *getPromotedType(const Type *Ty) {
  switch (Ty->getPrimitiveID()) {
  case Type::SByteTyID:
  case Type::ShortTyID:  return Type::IntTy;
  case Type::UByteTyID:
  case Type::UShortTyID: return Type::UIntTy;
  case Type::FloatTyID:  return Type::DoubleTy;
  default:               return Ty;
  }
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
          Instruction *New = BinaryOperator::create(Opcode, Op->getOperand(0),
                                                    Op1->getOperand(0),
                                                    Op1->getName(), &I);
          WorkList.push_back(New);
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
static inline Value *dyn_castNegVal(Value *V) {
  if (BinaryOperator::isNeg(V))
    return BinaryOperator::getNegArgument(cast<BinaryOperator>(V));

  // Constants can be considered to be negated values if they can be folded...
  if (Constant *C = dyn_cast<Constant>(V))
    return ConstantExpr::get(Instruction::Sub,
                             Constant::getNullValue(V->getType()), C);
  return 0;
}

static Constant *NotConstant(Constant *C) {
  return ConstantExpr::get(Instruction::Xor, C,
                           ConstantIntegral::getAllOnesValue(C->getType()));
}

static inline Value *dyn_castNotVal(Value *V) {
  if (BinaryOperator::isNot(V))
    return BinaryOperator::getNotArgument(cast<BinaryOperator>(V));

  // Constants can be considered to be not'ed values...
  if (ConstantIntegral *C = dyn_cast<ConstantIntegral>(V))
    return NotConstant(C);
  return 0;
}

// dyn_castFoldableMul - If this value is a multiply that can be folded into
// other computations (because it has a constant operand), return the
// non-constant operand of the multiply.
//
static inline Value *dyn_castFoldableMul(Value *V) {
  if (V->hasOneUse() && V->getType()->isInteger())
    if (Instruction *I = dyn_cast<Instruction>(V))
      if (I->getOpcode() == Instruction::Mul)
        if (isa<Constant>(I->getOperand(1)))
          return I->getOperand(0);
  return 0;
}

// dyn_castMaskingAnd - If this value is an And instruction masking a value with
// a constant, return the constant being anded with.
//
template<class ValueType>
static inline Constant *dyn_castMaskingAnd(ValueType *V) {
  if (Instruction *I = dyn_cast<Instruction>(V))
    if (I->getOpcode() == Instruction::And)
      return dyn_cast<Constant>(I->getOperand(1));

  // If this is a constant, it acts just like we were masking with it.
  return dyn_cast<Constant>(V);
}

// Log2 - Calculate the log base 2 for the specified value if it is exactly a
// power of 2.
static unsigned Log2(uint64_t Val) {
  assert(Val > 1 && "Values 0 and 1 should be handled elsewhere!");
  unsigned Count = 0;
  while (Val != 1) {
    if (Val & 1) return 0;    // Multiple bits set?
    Val >>= 1;
    ++Count;
  }
  return Count;
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
Instruction *AssociativeOpt(BinaryOperator &Root, const Functor &F) {
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
      BasicBlock *BB = Root.getParent();
      // All of the instructions have a single use and have no side-effects,
      // because of this, we can pull them all into the current basic block.
      if (LHSI->getParent() != BB) {
        // Move all of the instructions from root to LHSI into the current
        // block.
        Instruction *TmpLHSI = cast<Instruction>(Root.getOperand(0));
        Instruction *LastUse = &Root;
        while (TmpLHSI->getParent() == BB) {
          LastUse = TmpLHSI;
          TmpLHSI = cast<Instruction>(TmpLHSI->getOperand(0));
        }
        
        // Loop over all of the instructions in other blocks, moving them into
        // the current one.
        Value *TmpLHS = TmpLHSI;
        do {
          TmpLHSI = cast<Instruction>(TmpLHS);
          // Remove from current block...
          TmpLHSI->getParent()->getInstList().remove(TmpLHSI);
          // Insert before the last instruction...
          BB->getInstList().insert(LastUse, TmpLHSI);
          TmpLHS = TmpLHSI->getOperand(0);
        } while (TmpLHSI != LHSI);
      }
      
      // Now all of the instructions are in the current basic block, go ahead
      // and perform the reassociation.
      Instruction *TmpLHSI = cast<Instruction>(Root.getOperand(0));

      // First move the selected RHS to the LHS of the root...
      Root.setOperand(0, LHSI->getOperand(1));

      // Make what used to be the LHS of the root be the user of the root...
      Value *ExtraOperand = TmpLHSI->getOperand(1);
      Root.replaceAllUsesWith(TmpLHSI);          // Users now use TmpLHSI
      TmpLHSI->setOperand(1, &Root);             // TmpLHSI now uses the root
      BB->getInstList().remove(&Root);           // Remove root from the BB
      BB->getInstList().insert(TmpLHSI, &Root);  // Insert root before TmpLHSI

      // Now propagate the ExtraOperand down the chain of instructions until we
      // get to LHSI.
      while (TmpLHSI != LHSI) {
        Instruction *NextLHSI = cast<Instruction>(TmpLHSI->getOperand(0));
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


// AddRHS - Implements: X + X --> X << 1
struct AddRHS {
  Value *RHS;
  AddRHS(Value *rhs) : RHS(rhs) {}
  bool shouldApply(Value *LHS) const { return LHS == RHS; }
  Instruction *apply(BinaryOperator &Add) const {
    return new ShiftInst(Instruction::Shl, Add.getOperand(0),
                         ConstantInt::get(Type::UByteTy, 1));
  }
};

// AddMaskingAnd - Implements (A & C1)+(B & C2) --> (A & C1)|(B & C2)
//                 iff C1&C2 == 0
struct AddMaskingAnd {
  Constant *C2;
  AddMaskingAnd(Constant *c) : C2(c) {}
  bool shouldApply(Value *LHS) const {
    if (Constant *C1 = dyn_castMaskingAnd(LHS))
      return ConstantExpr::get(Instruction::And, C1, C2)->isNullValue();
    return false;
  }
  Instruction *apply(BinaryOperator &Add) const {
    return BinaryOperator::create(Instruction::Or, Add.getOperand(0),
                                  Add.getOperand(1));
  }
};



Instruction *InstCombiner::visitAdd(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *LHS = I.getOperand(0), *RHS = I.getOperand(1);

  // X + 0 --> X
  if (!I.getType()->isFloatingPoint() &&    // -0 + +0 = +0, so it's not a noop
      RHS == Constant::getNullValue(I.getType()))
    return ReplaceInstUsesWith(I, LHS);

  // X + X --> X << 1
  if (I.getType()->isInteger())
    if (Instruction *Result = AssociativeOpt(I, AddRHS(RHS))) return Result;

  // -A + B  -->  B - A
  if (Value *V = dyn_castNegVal(LHS))
    return BinaryOperator::create(Instruction::Sub, RHS, V);

  // A + -B  -->  A - B
  if (!isa<Constant>(RHS))
    if (Value *V = dyn_castNegVal(RHS))
      return BinaryOperator::create(Instruction::Sub, LHS, V);

  // X*C + X --> X * (C+1)
  if (dyn_castFoldableMul(LHS) == RHS) {
    Constant *CP1 =
      ConstantExpr::get(Instruction::Add, 
                        cast<Constant>(cast<Instruction>(LHS)->getOperand(1)),
                        ConstantInt::get(I.getType(), 1));
    return BinaryOperator::create(Instruction::Mul, RHS, CP1);
  }

  // X + X*C --> X * (C+1)
  if (dyn_castFoldableMul(RHS) == LHS) {
    Constant *CP1 =
      ConstantExpr::get(Instruction::Add,
                        cast<Constant>(cast<Instruction>(RHS)->getOperand(1)),
                        ConstantInt::get(I.getType(), 1));
    return BinaryOperator::create(Instruction::Mul, LHS, CP1);
  }

  // (A & C1)+(B & C2) --> (A & C1)|(B & C2) iff C1&C2 == 0
  if (Constant *C2 = dyn_castMaskingAnd(RHS))
    if (Instruction *R = AssociativeOpt(I, AddMaskingAnd(C2))) return R;

  if (ConstantInt *CRHS = dyn_cast<ConstantInt>(RHS)) {
    if (Instruction *ILHS = dyn_cast<Instruction>(LHS)) {
      switch (ILHS->getOpcode()) {
      case Instruction::Xor:
        // ~X + C --> (C-1) - X
        if (ConstantInt *XorRHS = dyn_cast<ConstantInt>(ILHS->getOperand(1)))
          if (XorRHS->isAllOnesValue())
            return BinaryOperator::create(Instruction::Sub,
                                          ConstantExpr::get(Instruction::Sub,
                                    CRHS, ConstantInt::get(I.getType(), 1)),
                                          ILHS->getOperand(0));
        break;
      default: break;
      }
    }
  }

  return Changed ? &I : 0;
}

// isSignBit - Return true if the value represented by the constant only has the
// highest order bit set.
static bool isSignBit(ConstantInt *CI) {
  unsigned NumBits = CI->getType()->getPrimitiveSize()*8;
  return (CI->getRawValue() & ~(-1LL << NumBits)) == (1ULL << (NumBits-1));
}

static unsigned getTypeSizeInBits(const Type *Ty) {
  return Ty == Type::BoolTy ? 1 : Ty->getPrimitiveSize()*8;
}

/// RemoveNoopCast - Strip off nonconverting casts from the value.
///
static Value *RemoveNoopCast(Value *V) {
  if (CastInst *CI = dyn_cast<CastInst>(V)) {
    const Type *CTy = CI->getType();
    const Type *OpTy = CI->getOperand(0)->getType();
    if (CTy->isInteger() && OpTy->isInteger()) {
      if (CTy->getPrimitiveSize() == OpTy->getPrimitiveSize())
        return RemoveNoopCast(CI->getOperand(0));
    } else if (isa<PointerType>(CTy) && isa<PointerType>(OpTy))
      return RemoveNoopCast(CI->getOperand(0));
  }
  return V;
}

Instruction *InstCombiner::visitSub(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Op0 == Op1)         // sub X, X  -> 0
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

  // If this is a 'B = x-(-A)', change to B = x+A...
  if (Value *V = dyn_castNegVal(Op1))
    return BinaryOperator::create(Instruction::Add, Op0, V);

  if (ConstantInt *C = dyn_cast<ConstantInt>(Op0)) {
    // Replace (-1 - A) with (~A)...
    if (C->isAllOnesValue())
      return BinaryOperator::createNot(Op1);

    // C - ~X == X + (1+C)
    if (BinaryOperator::isNot(Op1))
      return BinaryOperator::create(Instruction::Add,
               BinaryOperator::getNotArgument(cast<BinaryOperator>(Op1)),
                    ConstantExpr::get(Instruction::Add, C,
                                      ConstantInt::get(I.getType(), 1)));
    // -((uint)X >> 31) -> ((int)X >> 31)
    // -((int)X >> 31) -> ((uint)X >> 31)
    if (C->isNullValue()) {
      Value *NoopCastedRHS = RemoveNoopCast(Op1);
      if (ShiftInst *SI = dyn_cast<ShiftInst>(NoopCastedRHS))
        if (SI->getOpcode() == Instruction::Shr)
          if (ConstantUInt *CU = dyn_cast<ConstantUInt>(SI->getOperand(1))) {
            const Type *NewTy;
            if (SI->getType()->isSigned())
              NewTy = getUnsignedIntegralType(SI->getType());
            else
              NewTy = getSignedIntegralType(SI->getType());
            // Check to see if we are shifting out everything but the sign bit.
            if (CU->getValue() == SI->getType()->getPrimitiveSize()*8-1) {
              // Ok, the transformation is safe.  Insert a cast of the incoming
              // value, then the new shift, then the new cast.
              Instruction *FirstCast = new CastInst(SI->getOperand(0), NewTy,
                                                 SI->getOperand(0)->getName());
              Value *InV = InsertNewInstBefore(FirstCast, I);
              Instruction *NewShift = new ShiftInst(Instruction::Shr, FirstCast,
                                                    CU, SI->getName());
              if (NewShift->getType() == I.getType())
                return NewShift;
              else {
                InV = InsertNewInstBefore(NewShift, I);
                return new CastInst(NewShift, I.getType());
              }
            }
          }
    }
  }

  if (BinaryOperator *Op1I = dyn_cast<BinaryOperator>(Op1))
    if (Op1I->hasOneUse()) {
      // Replace (x - (y - z)) with (x + (z - y)) if the (y - z) subexpression
      // is not used by anyone else...
      //
      if (Op1I->getOpcode() == Instruction::Sub &&
          !Op1I->getType()->isFloatingPoint()) {
        // Swap the two operands of the subexpr...
        Value *IIOp0 = Op1I->getOperand(0), *IIOp1 = Op1I->getOperand(1);
        Op1I->setOperand(0, IIOp1);
        Op1I->setOperand(1, IIOp0);
        
        // Create the new top level add instruction...
        return BinaryOperator::create(Instruction::Add, Op0, Op1);
      }

      // Replace (A - (A & B)) with (A & ~B) if this is the only use of (A&B)...
      //
      if (Op1I->getOpcode() == Instruction::And &&
          (Op1I->getOperand(0) == Op0 || Op1I->getOperand(1) == Op0)) {
        Value *OtherOp = Op1I->getOperand(Op1I->getOperand(0) == Op0);

        Instruction *NewNot = BinaryOperator::createNot(OtherOp, "B.not", &I);
        return BinaryOperator::create(Instruction::And, Op0, NewNot);
      }

      // X - X*C --> X * (1-C)
      if (dyn_castFoldableMul(Op1I) == Op0) {
        Constant *CP1 =
          ConstantExpr::get(Instruction::Sub,
                            ConstantInt::get(I.getType(), 1),
                         cast<Constant>(cast<Instruction>(Op1)->getOperand(1)));
        assert(CP1 && "Couldn't constant fold 1-C?");
        return BinaryOperator::create(Instruction::Mul, Op0, CP1);
      }
    }

  // X*C - X --> X * (C-1)
  if (dyn_castFoldableMul(Op0) == Op1) {
    Constant *CP1 =
      ConstantExpr::get(Instruction::Sub,
                        cast<Constant>(cast<Instruction>(Op0)->getOperand(1)),
                        ConstantInt::get(I.getType(), 1));
    assert(CP1 && "Couldn't constant fold C - 1?");
    return BinaryOperator::create(Instruction::Mul, Op1, CP1);
  }

  return 0;
}

/// isSignBitCheck - Given an exploded setcc instruction, return true if it is
/// really just returns true if the most significant (sign) bit is set.
static bool isSignBitCheck(unsigned Opcode, Value *LHS, ConstantInt *RHS) {
  if (RHS->getType()->isSigned()) {
    // True if source is LHS < 0 or LHS <= -1
    return Opcode == Instruction::SetLT && RHS->isNullValue() ||
           Opcode == Instruction::SetLE && RHS->isAllOnesValue();
  } else {
    ConstantUInt *RHSC = cast<ConstantUInt>(RHS);
    // True if source is LHS > 127 or LHS >= 128, where the constants depend on
    // the size of the integer type.
    if (Opcode == Instruction::SetGE)
      return RHSC->getValue() == 1ULL<<(RHS->getType()->getPrimitiveSize()*8-1);
    if (Opcode == Instruction::SetGT)
      return RHSC->getValue() ==
        (1ULL << (RHS->getType()->getPrimitiveSize()*8-1))-1;
  }
  return false;
}

Instruction *InstCombiner::visitMul(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *Op0 = I.getOperand(0);

  // Simplify mul instructions with a constant RHS...
  if (Constant *Op1 = dyn_cast<Constant>(I.getOperand(1))) {
    if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1)) {

      // ((X << C1)*C2) == (X * (C2 << C1))
      if (ShiftInst *SI = dyn_cast<ShiftInst>(Op0))
        if (SI->getOpcode() == Instruction::Shl)
          if (Constant *ShOp = dyn_cast<Constant>(SI->getOperand(1)))
            return BinaryOperator::create(Instruction::Mul, SI->getOperand(0),
                                 ConstantExpr::get(Instruction::Shl, CI, ShOp));
      
      if (CI->isNullValue())
        return ReplaceInstUsesWith(I, Op1);  // X * 0  == 0
      if (CI->equalsInt(1))                  // X * 1  == X
        return ReplaceInstUsesWith(I, Op0);
      if (CI->isAllOnesValue())              // X * -1 == 0 - X
        return BinaryOperator::createNeg(Op0, I.getName());

      int64_t Val = (int64_t)cast<ConstantInt>(CI)->getRawValue();
      if (uint64_t C = Log2(Val))            // Replace X*(2^C) with X << C
        return new ShiftInst(Instruction::Shl, Op0,
                             ConstantUInt::get(Type::UByteTy, C));
    } else {
      ConstantFP *Op1F = cast<ConstantFP>(Op1);
      if (Op1F->isNullValue())
        return ReplaceInstUsesWith(I, Op1);

      // "In IEEE floating point, x*1 is not equivalent to x for nans.  However,
      // ANSI says we can drop signals, so we can do this anyway." (from GCC)
      if (Op1F->getValue() == 1.0)
        return ReplaceInstUsesWith(I, Op0);  // Eliminate 'mul double %X, 1.0'
    }
  }

  if (Value *Op0v = dyn_castNegVal(Op0))     // -X * -Y = X*Y
    if (Value *Op1v = dyn_castNegVal(I.getOperand(1)))
      return BinaryOperator::create(Instruction::Mul, Op0v, Op1v);

  // If one of the operands of the multiply is a cast from a boolean value, then
  // we know the bool is either zero or one, so this is a 'masking' multiply.
  // See if we can simplify things based on how the boolean was originally
  // formed.
  CastInst *BoolCast = 0;
  if (CastInst *CI = dyn_cast<CastInst>(I.getOperand(0)))
    if (CI->getOperand(0)->getType() == Type::BoolTy)
      BoolCast = CI;
  if (!BoolCast)
    if (CastInst *CI = dyn_cast<CastInst>(I.getOperand(1)))
      if (CI->getOperand(0)->getType() == Type::BoolTy)
        BoolCast = CI;
  if (BoolCast) {
    if (SetCondInst *SCI = dyn_cast<SetCondInst>(BoolCast->getOperand(0))) {
      Value *SCIOp0 = SCI->getOperand(0), *SCIOp1 = SCI->getOperand(1);
      const Type *SCOpTy = SCIOp0->getType();

      // If the setcc is true iff the sign bit of X is set, then convert this
      // multiply into a shift/and combination.
      if (isa<ConstantInt>(SCIOp1) &&
          isSignBitCheck(SCI->getOpcode(), SCIOp0, cast<ConstantInt>(SCIOp1))) {
        // Shift the X value right to turn it into "all signbits".
        Constant *Amt = ConstantUInt::get(Type::UByteTy,
                                          SCOpTy->getPrimitiveSize()*8-1);
        if (SCIOp0->getType()->isUnsigned()) {
          const Type *NewTy = getSignedIntegralType(SCIOp0->getType());
          SCIOp0 = InsertNewInstBefore(new CastInst(SCIOp0, NewTy,
                                                    SCIOp0->getName()), I);
        }

        Value *V =
          InsertNewInstBefore(new ShiftInst(Instruction::Shr, SCIOp0, Amt,
                                            BoolCast->getOperand(0)->getName()+
                                            ".mask"), I);

        // If the multiply type is not the same as the source type, sign extend
        // or truncate to the multiply type.
        if (I.getType() != V->getType())
          V = InsertNewInstBefore(new CastInst(V, I.getType(), V->getName()),I);
        
        Value *OtherOp = Op0 == BoolCast ? I.getOperand(1) : Op0;
        return BinaryOperator::create(Instruction::And, V, OtherOp);
      }
    }
  }

  return Changed ? &I : 0;
}

Instruction *InstCombiner::visitDiv(BinaryOperator &I) {
  // div X, 1 == X
  if (ConstantInt *RHS = dyn_cast<ConstantInt>(I.getOperand(1))) {
    if (RHS->equalsInt(1))
      return ReplaceInstUsesWith(I, I.getOperand(0));

    // Check to see if this is an unsigned division with an exact power of 2,
    // if so, convert to a right shift.
    if (ConstantUInt *C = dyn_cast<ConstantUInt>(RHS))
      if (uint64_t Val = C->getValue())    // Don't break X / 0
        if (uint64_t C = Log2(Val))
          return new ShiftInst(Instruction::Shr, I.getOperand(0),
                               ConstantUInt::get(Type::UByteTy, C));
  }

  // 0 / X == 0, we don't need to preserve faults!
  if (ConstantInt *LHS = dyn_cast<ConstantInt>(I.getOperand(0)))
    if (LHS->equalsInt(0))
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

  return 0;
}


Instruction *InstCombiner::visitRem(BinaryOperator &I) {
  if (ConstantInt *RHS = dyn_cast<ConstantInt>(I.getOperand(1))) {
    if (RHS->equalsInt(1))  // X % 1 == 0
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
    if (RHS->isAllOnesValue())  // X % -1 == 0
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

    // Check to see if this is an unsigned remainder with an exact power of 2,
    // if so, convert to a bitwise and.
    if (ConstantUInt *C = dyn_cast<ConstantUInt>(RHS))
      if (uint64_t Val = C->getValue())    // Don't break X % 0 (divide by zero)
        if (Log2(Val))
          return BinaryOperator::create(Instruction::And, I.getOperand(0),
                                        ConstantUInt::get(I.getType(), Val-1));
  }

  // 0 % X == 0, we don't need to preserve faults!
  if (ConstantInt *LHS = dyn_cast<ConstantInt>(I.getOperand(0)))
    if (LHS->equalsInt(0))
      return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

  return 0;
}

// isMaxValueMinusOne - return true if this is Max-1
static bool isMaxValueMinusOne(const ConstantInt *C) {
  if (const ConstantUInt *CU = dyn_cast<ConstantUInt>(C)) {
    // Calculate -1 casted to the right type...
    unsigned TypeBits = C->getType()->getPrimitiveSize()*8;
    uint64_t Val = ~0ULL;                // All ones
    Val >>= 64-TypeBits;                 // Shift out unwanted 1 bits...
    return CU->getValue() == Val-1;
  }

  const ConstantSInt *CS = cast<ConstantSInt>(C);
  
  // Calculate 0111111111..11111
  unsigned TypeBits = C->getType()->getPrimitiveSize()*8;
  int64_t Val = INT64_MAX;             // All ones
  Val >>= 64-TypeBits;                 // Shift out unwanted 1 bits...
  return CS->getValue() == Val-1;
}

// isMinValuePlusOne - return true if this is Min+1
static bool isMinValuePlusOne(const ConstantInt *C) {
  if (const ConstantUInt *CU = dyn_cast<ConstantUInt>(C))
    return CU->getValue() == 1;

  const ConstantSInt *CS = cast<ConstantSInt>(C);
  
  // Calculate 1111111111000000000000 
  unsigned TypeBits = C->getType()->getPrimitiveSize()*8;
  int64_t Val = -1;                    // All ones
  Val <<= TypeBits-1;                  // Shift over to the right spot
  return CS->getValue() == Val+1;
}

/// getSetCondCode - Encode a setcc opcode into a three bit mask.  These bits
/// are carefully arranged to allow folding of expressions such as:
///
///      (A < B) | (A > B) --> (A != B)
///
/// Bit value '4' represents that the comparison is true if A > B, bit value '2'
/// represents that the comparison is true if A == B, and bit value '1' is true
/// if A < B.
///
static unsigned getSetCondCode(const SetCondInst *SCI) {
  switch (SCI->getOpcode()) {
    // False -> 0
  case Instruction::SetGT: return 1;
  case Instruction::SetEQ: return 2;
  case Instruction::SetGE: return 3;
  case Instruction::SetLT: return 4;
  case Instruction::SetNE: return 5;
  case Instruction::SetLE: return 6;
    // True -> 7
  default:
    assert(0 && "Invalid SetCC opcode!");
    return 0;
  }
}

/// getSetCCValue - This is the complement of getSetCondCode, which turns an
/// opcode and two operands into either a constant true or false, or a brand new
/// SetCC instruction.
static Value *getSetCCValue(unsigned Opcode, Value *LHS, Value *RHS) {
  switch (Opcode) {
  case 0: return ConstantBool::False;
  case 1: return new SetCondInst(Instruction::SetGT, LHS, RHS);
  case 2: return new SetCondInst(Instruction::SetEQ, LHS, RHS);
  case 3: return new SetCondInst(Instruction::SetGE, LHS, RHS);
  case 4: return new SetCondInst(Instruction::SetLT, LHS, RHS);
  case 5: return new SetCondInst(Instruction::SetNE, LHS, RHS);
  case 6: return new SetCondInst(Instruction::SetLE, LHS, RHS);
  case 7: return ConstantBool::True;
  default: assert(0 && "Illegal SetCCCode!"); return 0;
  }
}

// FoldSetCCLogical - Implements (setcc1 A, B) & (setcc2 A, B) --> (setcc3 A, B)
struct FoldSetCCLogical {
  InstCombiner &IC;
  Value *LHS, *RHS;
  FoldSetCCLogical(InstCombiner &ic, SetCondInst *SCI)
    : IC(ic), LHS(SCI->getOperand(0)), RHS(SCI->getOperand(1)) {}
  bool shouldApply(Value *V) const {
    if (SetCondInst *SCI = dyn_cast<SetCondInst>(V))
      return (SCI->getOperand(0) == LHS && SCI->getOperand(1) == RHS ||
              SCI->getOperand(0) == RHS && SCI->getOperand(1) == LHS);
    return false;
  }
  Instruction *apply(BinaryOperator &Log) const {
    SetCondInst *SCI = cast<SetCondInst>(Log.getOperand(0));
    if (SCI->getOperand(0) != LHS) {
      assert(SCI->getOperand(1) == LHS);
      SCI->swapOperands();  // Swap the LHS and RHS of the SetCC
    }

    unsigned LHSCode = getSetCondCode(SCI);
    unsigned RHSCode = getSetCondCode(cast<SetCondInst>(Log.getOperand(1)));
    unsigned Code;
    switch (Log.getOpcode()) {
    case Instruction::And: Code = LHSCode & RHSCode; break;
    case Instruction::Or:  Code = LHSCode | RHSCode; break;
    case Instruction::Xor: Code = LHSCode ^ RHSCode; break;
    default: assert(0 && "Illegal logical opcode!"); return 0;
    }

    Value *RV = getSetCCValue(Code, LHS, RHS);
    if (Instruction *I = dyn_cast<Instruction>(RV))
      return I;
    // Otherwise, it's a constant boolean value...
    return IC.ReplaceInstUsesWith(Log, RV);
  }
};


// OptAndOp - This handles expressions of the form ((val OP C1) & C2).  Where
// the Op parameter is 'OP', OpRHS is 'C1', and AndRHS is 'C2'.  Op is
// guaranteed to be either a shift instruction or a binary operator.
Instruction *InstCombiner::OptAndOp(Instruction *Op,
                                    ConstantIntegral *OpRHS,
                                    ConstantIntegral *AndRHS,
                                    BinaryOperator &TheAnd) {
  Value *X = Op->getOperand(0);
  Constant *Together = 0;
  if (!isa<ShiftInst>(Op))
    Together = ConstantExpr::get(Instruction::And, AndRHS, OpRHS);

  switch (Op->getOpcode()) {
  case Instruction::Xor:
    if (Together->isNullValue()) {
      // (X ^ C1) & C2 --> (X & C2) iff (C1&C2) == 0
      return BinaryOperator::create(Instruction::And, X, AndRHS);
    } else if (Op->hasOneUse()) {
      // (X ^ C1) & C2 --> (X & C2) ^ (C1&C2)
      std::string OpName = Op->getName(); Op->setName("");
      Instruction *And = BinaryOperator::create(Instruction::And,
                                                X, AndRHS, OpName);
      InsertNewInstBefore(And, TheAnd);
      return BinaryOperator::create(Instruction::Xor, And, Together);
    }
    break;
  case Instruction::Or:
    // (X | C1) & C2 --> X & C2 iff C1 & C1 == 0
    if (Together->isNullValue())
      return BinaryOperator::create(Instruction::And, X, AndRHS);
    else {
      if (Together == AndRHS) // (X | C) & C --> C
        return ReplaceInstUsesWith(TheAnd, AndRHS);
      
      if (Op->hasOneUse() && Together != OpRHS) {
        // (X | C1) & C2 --> (X | (C1&C2)) & C2
        std::string Op0Name = Op->getName(); Op->setName("");
        Instruction *Or = BinaryOperator::create(Instruction::Or, X,
                                                 Together, Op0Name);
        InsertNewInstBefore(Or, TheAnd);
        return BinaryOperator::create(Instruction::And, Or, AndRHS);
      }
    }
    break;
  case Instruction::Add:
    if (Op->hasOneUse()) {
      // Adding a one to a single bit bit-field should be turned into an XOR
      // of the bit.  First thing to check is to see if this AND is with a
      // single bit constant.
      unsigned long long AndRHSV = cast<ConstantInt>(AndRHS)->getRawValue();

      // Clear bits that are not part of the constant.
      AndRHSV &= (1ULL << AndRHS->getType()->getPrimitiveSize()*8)-1;

      // If there is only one bit set...
      if ((AndRHSV & (AndRHSV-1)) == 0) {
        // Ok, at this point, we know that we are masking the result of the
        // ADD down to exactly one bit.  If the constant we are adding has
        // no bits set below this bit, then we can eliminate the ADD.
        unsigned long long AddRHS = cast<ConstantInt>(OpRHS)->getRawValue();
            
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
            std::string Name = Op->getName(); Op->setName("");
            // Pull the XOR out of the AND.
            Instruction *NewAnd =
              BinaryOperator::create(Instruction::And, X, AndRHS, Name);
            InsertNewInstBefore(NewAnd, TheAnd);
            return BinaryOperator::create(Instruction::Xor, NewAnd, AndRHS);
          }
        }
      }
    }
    break;

  case Instruction::Shl: {
    // We know that the AND will not produce any of the bits shifted in, so if
    // the anded constant includes them, clear them now!
    //
    Constant *AllOne = ConstantIntegral::getAllOnesValue(AndRHS->getType());
    Constant *CI = ConstantExpr::get(Instruction::And, AndRHS,
                            ConstantExpr::get(Instruction::Shl, AllOne, OpRHS));
    if (CI != AndRHS) {
      TheAnd.setOperand(1, CI);
      return &TheAnd;
    }
    break;
  } 
  case Instruction::Shr:
    // We know that the AND will not produce any of the bits shifted in, so if
    // the anded constant includes them, clear them now!  This only applies to
    // unsigned shifts, because a signed shr may bring in set bits!
    //
    if (AndRHS->getType()->isUnsigned()) {
      Constant *AllOne = ConstantIntegral::getAllOnesValue(AndRHS->getType());
      Constant *CI = ConstantExpr::get(Instruction::And, AndRHS,
                            ConstantExpr::get(Instruction::Shr, AllOne, OpRHS));
      if (CI != AndRHS) {
        TheAnd.setOperand(1, CI);
        return &TheAnd;
      }
    }
    break;
  }
  return 0;
}


Instruction *InstCombiner::visitAnd(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // and X, X = X   and X, 0 == 0
  if (Op0 == Op1 || Op1 == Constant::getNullValue(I.getType()))
    return ReplaceInstUsesWith(I, Op1);

  // and X, -1 == X
  if (ConstantIntegral *RHS = dyn_cast<ConstantIntegral>(Op1)) {
    if (RHS->isAllOnesValue())
      return ReplaceInstUsesWith(I, Op0);

    // Optimize a variety of ((val OP C1) & C2) combinations...
    if (isa<BinaryOperator>(Op0) || isa<ShiftInst>(Op0)) {
      Instruction *Op0I = cast<Instruction>(Op0);
      Value *X = Op0I->getOperand(0);
      if (ConstantInt *Op0CI = dyn_cast<ConstantInt>(Op0I->getOperand(1)))
        if (Instruction *Res = OptAndOp(Op0I, Op0CI, RHS, I))
          return Res;
    }
  }

  Value *Op0NotVal = dyn_castNotVal(Op0);
  Value *Op1NotVal = dyn_castNotVal(Op1);

  // (~A & ~B) == (~(A | B)) - Demorgan's Law
  if (Op0NotVal && Op1NotVal && isOnlyUse(Op0) && isOnlyUse(Op1)) {
    Instruction *Or = BinaryOperator::create(Instruction::Or, Op0NotVal,
                                             Op1NotVal,I.getName()+".demorgan");
    InsertNewInstBefore(Or, I);
    return BinaryOperator::createNot(Or);
  }

  if (Op0NotVal == Op1 || Op1NotVal == Op0)  // A & ~A  == ~A & A == 0
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

  // (setcc1 A, B) & (setcc2 A, B) --> (setcc3 A, B)
  if (SetCondInst *RHS = dyn_cast<SetCondInst>(I.getOperand(1)))
    if (Instruction *R = AssociativeOpt(I, FoldSetCCLogical(*this, RHS)))
      return R;

  return Changed ? &I : 0;
}



Instruction *InstCombiner::visitOr(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // or X, X = X   or X, 0 == X
  if (Op0 == Op1 || Op1 == Constant::getNullValue(I.getType()))
    return ReplaceInstUsesWith(I, Op0);

  // or X, -1 == -1
  if (ConstantIntegral *RHS = dyn_cast<ConstantIntegral>(Op1)) {
    if (RHS->isAllOnesValue())
      return ReplaceInstUsesWith(I, Op1);

    if (Instruction *Op0I = dyn_cast<Instruction>(Op0)) {
      // (X & C1) | C2 --> (X | C2) & (C1|C2)
      if (Op0I->getOpcode() == Instruction::And && isOnlyUse(Op0))
        if (ConstantInt *Op0CI = dyn_cast<ConstantInt>(Op0I->getOperand(1))) {
          std::string Op0Name = Op0I->getName(); Op0I->setName("");
          Instruction *Or = BinaryOperator::create(Instruction::Or,
                                                   Op0I->getOperand(0), RHS,
                                                   Op0Name);
          InsertNewInstBefore(Or, I);
          return BinaryOperator::create(Instruction::And, Or,
                             ConstantExpr::get(Instruction::Or, RHS, Op0CI));
        }

      // (X ^ C1) | C2 --> (X | C2) ^ (C1&~C2)
      if (Op0I->getOpcode() == Instruction::Xor && isOnlyUse(Op0))
        if (ConstantInt *Op0CI = dyn_cast<ConstantInt>(Op0I->getOperand(1))) {
          std::string Op0Name = Op0I->getName(); Op0I->setName("");
          Instruction *Or = BinaryOperator::create(Instruction::Or,
                                                   Op0I->getOperand(0), RHS,
                                                   Op0Name);
          InsertNewInstBefore(Or, I);
          return BinaryOperator::create(Instruction::Xor, Or,
                            ConstantExpr::get(Instruction::And, Op0CI,
                                              NotConstant(RHS)));
        }
    }
  }

  // (A & C1)|(A & C2) == A & (C1|C2)
  if (Instruction *LHS = dyn_cast<BinaryOperator>(Op0))
    if (Instruction *RHS = dyn_cast<BinaryOperator>(Op1))
      if (LHS->getOperand(0) == RHS->getOperand(0))
        if (Constant *C0 = dyn_castMaskingAnd(LHS))
          if (Constant *C1 = dyn_castMaskingAnd(RHS))
            return BinaryOperator::create(Instruction::And, LHS->getOperand(0),
                                    ConstantExpr::get(Instruction::Or, C0, C1));

  Value *Op0NotVal = dyn_castNotVal(Op0);
  Value *Op1NotVal = dyn_castNotVal(Op1);

  if (Op1 == Op0NotVal)   // ~A | A == -1
    return ReplaceInstUsesWith(I, 
                               ConstantIntegral::getAllOnesValue(I.getType()));

  if (Op0 == Op1NotVal)   // A | ~A == -1
    return ReplaceInstUsesWith(I, 
                               ConstantIntegral::getAllOnesValue(I.getType()));

  // (~A | ~B) == (~(A & B)) - Demorgan's Law
  if (Op0NotVal && Op1NotVal && isOnlyUse(Op0) && isOnlyUse(Op1)) {
    Instruction *And = BinaryOperator::create(Instruction::And, Op0NotVal,
                                              Op1NotVal,I.getName()+".demorgan",
                                              &I);
    WorkList.push_back(And);
    return BinaryOperator::createNot(And);
  }

  // (setcc1 A, B) | (setcc2 A, B) --> (setcc3 A, B)
  if (SetCondInst *RHS = dyn_cast<SetCondInst>(I.getOperand(1)))
    if (Instruction *R = AssociativeOpt(I, FoldSetCCLogical(*this, RHS)))
      return R;

  return Changed ? &I : 0;
}

// XorSelf - Implements: X ^ X --> 0
struct XorSelf {
  Value *RHS;
  XorSelf(Value *rhs) : RHS(rhs) {}
  bool shouldApply(Value *LHS) const { return LHS == RHS; }
  Instruction *apply(BinaryOperator &Xor) const {
    return &Xor;
  }
};


Instruction *InstCombiner::visitXor(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // xor X, X = 0, even if X is nested in a sequence of Xor's.
  if (Instruction *Result = AssociativeOpt(I, XorSelf(Op1))) {
    assert(Result == &I && "AssociativeOpt didn't work?");
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));
  }

  if (ConstantIntegral *RHS = dyn_cast<ConstantIntegral>(Op1)) {
    // xor X, 0 == X
    if (RHS->isNullValue())
      return ReplaceInstUsesWith(I, Op0);

    if (BinaryOperator *Op0I = dyn_cast<BinaryOperator>(Op0)) {
      // xor (setcc A, B), true = not (setcc A, B) = setncc A, B
      if (SetCondInst *SCI = dyn_cast<SetCondInst>(Op0I))
        if (RHS == ConstantBool::True && SCI->hasOneUse())
          return new SetCondInst(SCI->getInverseCondition(),
                                 SCI->getOperand(0), SCI->getOperand(1));

      // ~(c-X) == X-c-1 == X+(-c-1)
      if (Op0I->getOpcode() == Instruction::Sub && RHS->isAllOnesValue())
        if (Constant *Op0I0C = dyn_cast<Constant>(Op0I->getOperand(0))) {
          Constant *NegOp0I0C = ConstantExpr::get(Instruction::Sub,
                             Constant::getNullValue(Op0I0C->getType()), Op0I0C);
          Constant *ConstantRHS = ConstantExpr::get(Instruction::Sub, NegOp0I0C,
                                              ConstantInt::get(I.getType(), 1));
          return BinaryOperator::create(Instruction::Add, Op0I->getOperand(1),
                                        ConstantRHS);
        }
          
      if (ConstantInt *Op0CI = dyn_cast<ConstantInt>(Op0I->getOperand(1)))
        switch (Op0I->getOpcode()) {
        case Instruction::Add:
          // ~(X-c) --> (-c-1)-X
          if (RHS->isAllOnesValue()) {
            Constant *NegOp0CI = ConstantExpr::get(Instruction::Sub,
                               Constant::getNullValue(Op0CI->getType()), Op0CI);
            return BinaryOperator::create(Instruction::Sub,
                           ConstantExpr::get(Instruction::Sub, NegOp0CI,
                                             ConstantInt::get(I.getType(), 1)),
                                          Op0I->getOperand(0));
          }
          break;
        case Instruction::And:
          // (X & C1) ^ C2 --> (X & C1) | C2 iff (C1&C2) == 0
          if (ConstantExpr::get(Instruction::And, RHS, Op0CI)->isNullValue())
            return BinaryOperator::create(Instruction::Or, Op0, RHS);
          break;
        case Instruction::Or:
          // (X | C1) ^ C2 --> (X | C1) & ~C2 iff (C1&C2) == C2
          if (ConstantExpr::get(Instruction::And, RHS, Op0CI) == RHS)
            return BinaryOperator::create(Instruction::And, Op0,
                                          NotConstant(RHS));
          break;
        default: break;
        }
    }
  }

  if (Value *X = dyn_castNotVal(Op0))   // ~A ^ A == -1
    if (X == Op1)
      return ReplaceInstUsesWith(I,
                                ConstantIntegral::getAllOnesValue(I.getType()));

  if (Value *X = dyn_castNotVal(Op1))   // A ^ ~A == -1
    if (X == Op0)
      return ReplaceInstUsesWith(I,
                                ConstantIntegral::getAllOnesValue(I.getType()));

  if (Instruction *Op1I = dyn_cast<Instruction>(Op1))
    if (Op1I->getOpcode() == Instruction::Or) {
      if (Op1I->getOperand(0) == Op0) {              // B^(B|A) == (A|B)^B
        cast<BinaryOperator>(Op1I)->swapOperands();
        I.swapOperands();
        std::swap(Op0, Op1);
      } else if (Op1I->getOperand(1) == Op0) {       // B^(A|B) == (A|B)^B
        I.swapOperands();
        std::swap(Op0, Op1);
      }      
    } else if (Op1I->getOpcode() == Instruction::Xor) {
      if (Op0 == Op1I->getOperand(0))                        // A^(A^B) == B
        return ReplaceInstUsesWith(I, Op1I->getOperand(1));
      else if (Op0 == Op1I->getOperand(1))                   // A^(B^A) == B
        return ReplaceInstUsesWith(I, Op1I->getOperand(0));
    }

  if (Instruction *Op0I = dyn_cast<Instruction>(Op0))
    if (Op0I->getOpcode() == Instruction::Or && Op0I->hasOneUse()) {
      if (Op0I->getOperand(0) == Op1)                // (B|A)^B == (A|B)^B
        cast<BinaryOperator>(Op0I)->swapOperands();
      if (Op0I->getOperand(1) == Op1) {              // (A|B)^B == A & ~B
        Value *NotB = BinaryOperator::createNot(Op1, Op1->getName()+".not", &I);
        WorkList.push_back(cast<Instruction>(NotB));
        return BinaryOperator::create(Instruction::And, Op0I->getOperand(0),
                                      NotB);
      }
    } else if (Op0I->getOpcode() == Instruction::Xor) {
      if (Op1 == Op0I->getOperand(0))                        // (A^B)^A == B
        return ReplaceInstUsesWith(I, Op0I->getOperand(1));
      else if (Op1 == Op0I->getOperand(1))                   // (B^A)^A == B
        return ReplaceInstUsesWith(I, Op0I->getOperand(0));
    }

  // (A & C1)^(B & C2) -> (A & C1)|(B & C2) iff C1^C2 == 0
  if (Constant *C1 = dyn_castMaskingAnd(Op0))
    if (Constant *C2 = dyn_castMaskingAnd(Op1))
      if (ConstantExpr::get(Instruction::And, C1, C2)->isNullValue())
        return BinaryOperator::create(Instruction::Or, Op0, Op1);

  // (setcc1 A, B) ^ (setcc2 A, B) --> (setcc3 A, B)
  if (SetCondInst *RHS = dyn_cast<SetCondInst>(I.getOperand(1)))
    if (Instruction *R = AssociativeOpt(I, FoldSetCCLogical(*this, RHS)))
      return R;

  return Changed ? &I : 0;
}

// AddOne, SubOne - Add or subtract a constant one from an integer constant...
static Constant *AddOne(ConstantInt *C) {
  Constant *Result = ConstantExpr::get(Instruction::Add, C,
                                       ConstantInt::get(C->getType(), 1));
  assert(Result && "Constant folding integer addition failed!");
  return Result;
}
static Constant *SubOne(ConstantInt *C) {
  Constant *Result = ConstantExpr::get(Instruction::Sub, C,
                                       ConstantInt::get(C->getType(), 1));
  assert(Result && "Constant folding integer addition failed!");
  return Result;
}

// isTrueWhenEqual - Return true if the specified setcondinst instruction is
// true when both operands are equal...
//
static bool isTrueWhenEqual(Instruction &I) {
  return I.getOpcode() == Instruction::SetEQ ||
         I.getOpcode() == Instruction::SetGE ||
         I.getOpcode() == Instruction::SetLE;
}

Instruction *InstCombiner::visitSetCondInst(BinaryOperator &I) {
  bool Changed = SimplifyCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  const Type *Ty = Op0->getType();

  // setcc X, X
  if (Op0 == Op1)
    return ReplaceInstUsesWith(I, ConstantBool::get(isTrueWhenEqual(I)));

  // setcc <global/alloca*>, 0 - Global/Stack value addresses are never null!
  if (isa<ConstantPointerNull>(Op1) && 
      (isa<GlobalValue>(Op0) || isa<AllocaInst>(Op0)))
    return ReplaceInstUsesWith(I, ConstantBool::get(!isTrueWhenEqual(I)));


  // setcc's with boolean values can always be turned into bitwise operations
  if (Ty == Type::BoolTy) {
    // If this is <, >, or !=, we can change this into a simple xor instruction
    if (!isTrueWhenEqual(I))
      return BinaryOperator::create(Instruction::Xor, Op0, Op1);

    // Otherwise we need to make a temporary intermediate instruction and insert
    // it into the instruction stream.  This is what we are after:
    //
    //  seteq bool %A, %B -> ~(A^B)
    //  setle bool %A, %B -> ~A | B
    //  setge bool %A, %B -> A | ~B
    //
    if (I.getOpcode() == Instruction::SetEQ) {  // seteq case
      Instruction *Xor = BinaryOperator::create(Instruction::Xor, Op0, Op1,
                                                I.getName()+"tmp");
      InsertNewInstBefore(Xor, I);
      return BinaryOperator::createNot(Xor);
    }

    // Handle the setXe cases...
    assert(I.getOpcode() == Instruction::SetGE ||
           I.getOpcode() == Instruction::SetLE);

    if (I.getOpcode() == Instruction::SetGE)
      std::swap(Op0, Op1);                   // Change setge -> setle

    // Now we just have the SetLE case.
    Instruction *Not = BinaryOperator::createNot(Op0, I.getName()+"tmp");
    InsertNewInstBefore(Not, I);
    return BinaryOperator::create(Instruction::Or, Not, Op1);
  }

  // Check to see if we are doing one of many comparisons against constant
  // integers at the end of their ranges...
  //
  if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1)) {
    // Simplify seteq and setne instructions...
    if (I.getOpcode() == Instruction::SetEQ ||
        I.getOpcode() == Instruction::SetNE) {
      bool isSetNE = I.getOpcode() == Instruction::SetNE;

      // If the first operand is (and|or|xor) with a constant, and the second
      // operand is a constant, simplify a bit.
      if (BinaryOperator *BO = dyn_cast<BinaryOperator>(Op0)) {
        switch (BO->getOpcode()) {
        case Instruction::Add:
          if (CI->isNullValue()) {
            // Replace ((add A, B) != 0) with (A != -B) if A or B is
            // efficiently invertible, or if the add has just this one use.
            Value *BOp0 = BO->getOperand(0), *BOp1 = BO->getOperand(1);
            if (Value *NegVal = dyn_castNegVal(BOp1))
              return new SetCondInst(I.getOpcode(), BOp0, NegVal);
            else if (Value *NegVal = dyn_castNegVal(BOp0))
              return new SetCondInst(I.getOpcode(), NegVal, BOp1);
            else if (BO->hasOneUse()) {
              Instruction *Neg = BinaryOperator::createNeg(BOp1, BO->getName());
              BO->setName("");
              InsertNewInstBefore(Neg, I);
              return new SetCondInst(I.getOpcode(), BOp0, Neg);
            }
          }
          break;
        case Instruction::Xor:
          // For the xor case, we can xor two constants together, eliminating
          // the explicit xor.
          if (Constant *BOC = dyn_cast<Constant>(BO->getOperand(1)))
            return BinaryOperator::create(I.getOpcode(), BO->getOperand(0),
                                  ConstantExpr::get(Instruction::Xor, CI, BOC));

          // FALLTHROUGH
        case Instruction::Sub:
          // Replace (([sub|xor] A, B) != 0) with (A != B)
          if (CI->isNullValue())
            return new SetCondInst(I.getOpcode(), BO->getOperand(0),
                                   BO->getOperand(1));
          break;

        case Instruction::Or:
          // If bits are being or'd in that are not present in the constant we
          // are comparing against, then the comparison could never succeed!
          if (Constant *BOC = dyn_cast<Constant>(BO->getOperand(1))) {
            Constant *NotCI = NotConstant(CI);
            if (!ConstantExpr::get(Instruction::And, BOC, NotCI)->isNullValue())
              return ReplaceInstUsesWith(I, ConstantBool::get(isSetNE));
          }
          break;

        case Instruction::And:
          if (ConstantInt *BOC = dyn_cast<ConstantInt>(BO->getOperand(1))) {
            // If bits are being compared against that are and'd out, then the
            // comparison can never succeed!
            if (!ConstantExpr::get(Instruction::And, CI,
                                   NotConstant(BOC))->isNullValue())
              return ReplaceInstUsesWith(I, ConstantBool::get(isSetNE));

            // Replace (and X, (1 << size(X)-1) != 0) with x < 0, converting X
            // to be a signed value as appropriate.
            if (isSignBit(BOC)) {
              Value *X = BO->getOperand(0);
              // If 'X' is not signed, insert a cast now...
              if (!BOC->getType()->isSigned()) {
                const Type *DestTy = getSignedIntegralType(BOC->getType());
                CastInst *NewCI = new CastInst(X,DestTy,X->getName()+".signed");
                InsertNewInstBefore(NewCI, I);
                X = NewCI;
              }
              return new SetCondInst(isSetNE ? Instruction::SetLT :
                                         Instruction::SetGE, X,
                                     Constant::getNullValue(X->getType()));
            }
          }
        default: break;
        }
      }
    } else {  // Not a SetEQ/SetNE
      // If the LHS is a cast from an integral value of the same size, 
      if (CastInst *Cast = dyn_cast<CastInst>(Op0)) {
        Value *CastOp = Cast->getOperand(0);
        const Type *SrcTy = CastOp->getType();
        unsigned SrcTySize = SrcTy->getPrimitiveSize();
        if (SrcTy != Cast->getType() && SrcTy->isInteger() &&
            SrcTySize == Cast->getType()->getPrimitiveSize()) {
          assert((SrcTy->isSigned() ^ Cast->getType()->isSigned()) && 
                 "Source and destination signednesses should differ!");
          if (Cast->getType()->isSigned()) {
            // If this is a signed comparison, check for comparisons in the
            // vicinity of zero.
            if (I.getOpcode() == Instruction::SetLT && CI->isNullValue())
              // X < 0  => x > 127
              return BinaryOperator::create(Instruction::SetGT, CastOp,
                         ConstantUInt::get(SrcTy, (1ULL << (SrcTySize*8-1))-1));
            else if (I.getOpcode() == Instruction::SetGT &&
                     cast<ConstantSInt>(CI)->getValue() == -1)
              // X > -1  => x < 128
              return BinaryOperator::create(Instruction::SetLT, CastOp,
                         ConstantUInt::get(SrcTy, 1ULL << (SrcTySize*8-1)));
          } else {
            ConstantUInt *CUI = cast<ConstantUInt>(CI);
            if (I.getOpcode() == Instruction::SetLT &&
                CUI->getValue() == 1ULL << (SrcTySize*8-1))
              // X < 128 => X > -1
              return BinaryOperator::create(Instruction::SetGT, CastOp,
                                            ConstantSInt::get(SrcTy, -1));
            else if (I.getOpcode() == Instruction::SetGT &&
                     CUI->getValue() == (1ULL << (SrcTySize*8-1))-1)
              // X > 127 => X < 0
              return BinaryOperator::create(Instruction::SetLT, CastOp,
                                            Constant::getNullValue(SrcTy));
          }
        }
      }
    }

    // Check to see if we are comparing against the minimum or maximum value...
    if (CI->isMinValue()) {
      if (I.getOpcode() == Instruction::SetLT)       // A < MIN -> FALSE
        return ReplaceInstUsesWith(I, ConstantBool::False);
      if (I.getOpcode() == Instruction::SetGE)       // A >= MIN -> TRUE
        return ReplaceInstUsesWith(I, ConstantBool::True);
      if (I.getOpcode() == Instruction::SetLE)       // A <= MIN -> A == MIN
        return BinaryOperator::create(Instruction::SetEQ, Op0, Op1);
      if (I.getOpcode() == Instruction::SetGT)       // A > MIN -> A != MIN
        return BinaryOperator::create(Instruction::SetNE, Op0, Op1);

    } else if (CI->isMaxValue()) {
      if (I.getOpcode() == Instruction::SetGT)       // A > MAX -> FALSE
        return ReplaceInstUsesWith(I, ConstantBool::False);
      if (I.getOpcode() == Instruction::SetLE)       // A <= MAX -> TRUE
        return ReplaceInstUsesWith(I, ConstantBool::True);
      if (I.getOpcode() == Instruction::SetGE)       // A >= MAX -> A == MAX
        return BinaryOperator::create(Instruction::SetEQ, Op0, Op1);
      if (I.getOpcode() == Instruction::SetLT)       // A < MAX -> A != MAX
        return BinaryOperator::create(Instruction::SetNE, Op0, Op1);

      // Comparing against a value really close to min or max?
    } else if (isMinValuePlusOne(CI)) {
      if (I.getOpcode() == Instruction::SetLT)       // A < MIN+1 -> A == MIN
        return BinaryOperator::create(Instruction::SetEQ, Op0, SubOne(CI));
      if (I.getOpcode() == Instruction::SetGE)       // A >= MIN-1 -> A != MIN
        return BinaryOperator::create(Instruction::SetNE, Op0, SubOne(CI));

    } else if (isMaxValueMinusOne(CI)) {
      if (I.getOpcode() == Instruction::SetGT)       // A > MAX-1 -> A == MAX
        return BinaryOperator::create(Instruction::SetEQ, Op0, AddOne(CI));
      if (I.getOpcode() == Instruction::SetLE)       // A <= MAX-1 -> A != MAX
        return BinaryOperator::create(Instruction::SetNE, Op0, AddOne(CI));
    }

    // If we still have a setle or setge instruction, turn it into the
    // appropriate setlt or setgt instruction.  Since the border cases have
    // already been handled above, this requires little checking.
    //
    if (I.getOpcode() == Instruction::SetLE)
      return BinaryOperator::create(Instruction::SetLT, Op0, AddOne(CI));
    if (I.getOpcode() == Instruction::SetGE)
      return BinaryOperator::create(Instruction::SetGT, Op0, SubOne(CI));
  }

  // Test to see if the operands of the setcc are casted versions of other
  // values.  If the cast can be stripped off both arguments, we do so now.
  if (CastInst *CI = dyn_cast<CastInst>(Op0)) {
    Value *CastOp0 = CI->getOperand(0);
    if (CastOp0->getType()->isLosslesslyConvertibleTo(CI->getType()) &&
        (isa<Constant>(Op1) || isa<CastInst>(Op1)) &&
        (I.getOpcode() == Instruction::SetEQ ||
         I.getOpcode() == Instruction::SetNE)) {
      // We keep moving the cast from the left operand over to the right
      // operand, where it can often be eliminated completely.
      Op0 = CastOp0;
      
      // If operand #1 is a cast instruction, see if we can eliminate it as
      // well.
      if (CastInst *CI2 = dyn_cast<CastInst>(Op1))
        if (CI2->getOperand(0)->getType()->isLosslesslyConvertibleTo(
                                                               Op0->getType()))
          Op1 = CI2->getOperand(0);
      
      // If Op1 is a constant, we can fold the cast into the constant.
      if (Op1->getType() != Op0->getType())
        if (Constant *Op1C = dyn_cast<Constant>(Op1)) {
          Op1 = ConstantExpr::getCast(Op1C, Op0->getType());
        } else {
          // Otherwise, cast the RHS right before the setcc
          Op1 = new CastInst(Op1, Op0->getType(), Op1->getName());
          InsertNewInstBefore(cast<Instruction>(Op1), I);
        }
      return BinaryOperator::create(I.getOpcode(), Op0, Op1);
    }

    // Handle the special case of: setcc (cast bool to X), <cst>
    // This comes up when you have code like
    //   int X = A < B;
    //   if (X) ...
    // For generality, we handle any zero-extension of any operand comparison
    // with a constant.
    if (ConstantInt *ConstantRHS = dyn_cast<ConstantInt>(Op1)) {
      const Type *SrcTy = CastOp0->getType();
      const Type *DestTy = Op0->getType();
      if (SrcTy->getPrimitiveSize() < DestTy->getPrimitiveSize() &&
          (SrcTy->isUnsigned() || SrcTy == Type::BoolTy)) {
        // Ok, we have an expansion of operand 0 into a new type.  Get the
        // constant value, masink off bits which are not set in the RHS.  These
        // could be set if the destination value is signed.
        uint64_t ConstVal = ConstantRHS->getRawValue();
        ConstVal &= (1ULL << DestTy->getPrimitiveSize()*8)-1;

        // If the constant we are comparing it with has high bits set, which
        // don't exist in the original value, the values could never be equal,
        // because the source would be zero extended.
        unsigned SrcBits =
          SrcTy == Type::BoolTy ? 1 : SrcTy->getPrimitiveSize()*8;
        bool HasSignBit = ConstVal & (1ULL << (DestTy->getPrimitiveSize()*8-1));
        if (ConstVal & ~((1ULL << SrcBits)-1)) {
          switch (I.getOpcode()) {
          default: assert(0 && "Unknown comparison type!");
          case Instruction::SetEQ:
            return ReplaceInstUsesWith(I, ConstantBool::False);
          case Instruction::SetNE:
            return ReplaceInstUsesWith(I, ConstantBool::True);
          case Instruction::SetLT:
          case Instruction::SetLE:
            if (DestTy->isSigned() && HasSignBit)
              return ReplaceInstUsesWith(I, ConstantBool::False);
            return ReplaceInstUsesWith(I, ConstantBool::True);
          case Instruction::SetGT:
          case Instruction::SetGE:
            if (DestTy->isSigned() && HasSignBit)
              return ReplaceInstUsesWith(I, ConstantBool::True);
            return ReplaceInstUsesWith(I, ConstantBool::False);
          }
        }
        
        // Otherwise, we can replace the setcc with a setcc of the smaller
        // operand value.
        Op1 = ConstantExpr::getCast(cast<Constant>(Op1), SrcTy);
        return BinaryOperator::create(I.getOpcode(), CastOp0, Op1);
      }
    }
  }
  return Changed ? &I : 0;
}



Instruction *InstCombiner::visitShiftInst(ShiftInst &I) {
  assert(I.getOperand(1)->getType() == Type::UByteTy);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  bool isLeftShift = I.getOpcode() == Instruction::Shl;

  // shl X, 0 == X and shr X, 0 == X
  // shl 0, X == 0 and shr 0, X == 0
  if (Op1 == Constant::getNullValue(Type::UByteTy) ||
      Op0 == Constant::getNullValue(Op0->getType()))
    return ReplaceInstUsesWith(I, Op0);

  // shr int -1, X = -1   (for any arithmetic shift rights of ~0)
  if (!isLeftShift)
    if (ConstantSInt *CSI = dyn_cast<ConstantSInt>(Op0))
      if (CSI->isAllOnesValue())
        return ReplaceInstUsesWith(I, CSI);

  if (ConstantUInt *CUI = dyn_cast<ConstantUInt>(Op1)) {
    // shl uint X, 32 = 0 and shr ubyte Y, 9 = 0, ... just don't eliminate shr
    // of a signed value.
    //
    unsigned TypeBits = Op0->getType()->getPrimitiveSize()*8;
    if (CUI->getValue() >= TypeBits) {
      if (!Op0->getType()->isSigned() || isLeftShift)
        return ReplaceInstUsesWith(I, Constant::getNullValue(Op0->getType()));
      else {
        I.setOperand(1, ConstantUInt::get(Type::UByteTy, TypeBits-1));
        return &I;
      }
    }

    // ((X*C1) << C2) == (X * (C1 << C2))
    if (BinaryOperator *BO = dyn_cast<BinaryOperator>(Op0))
      if (BO->getOpcode() == Instruction::Mul && isLeftShift)
        if (Constant *BOOp = dyn_cast<Constant>(BO->getOperand(1)))
          return BinaryOperator::create(Instruction::Mul, BO->getOperand(0),
                                ConstantExpr::get(Instruction::Shl, BOOp, CUI));
    

    // If the operand is an bitwise operator with a constant RHS, and the
    // shift is the only use, we can pull it out of the shift.
    if (Op0->hasOneUse())
      if (BinaryOperator *Op0BO = dyn_cast<BinaryOperator>(Op0))
        if (ConstantInt *Op0C = dyn_cast<ConstantInt>(Op0BO->getOperand(1))) {
          bool isValid = true;     // Valid only for And, Or, Xor
          bool highBitSet = false; // Transform if high bit of constant set?

          switch (Op0BO->getOpcode()) {
          default: isValid = false; break;   // Do not perform transform!
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
          if (isValid && !isLeftShift && !I.getType()->isUnsigned()) {
            uint64_t Val = Op0C->getRawValue();
            isValid = ((Val & (1 << (TypeBits-1))) != 0) == highBitSet;
          }

          if (isValid) {
            Constant *NewRHS = ConstantExpr::get(I.getOpcode(), Op0C, CUI);

            Instruction *NewShift =
              new ShiftInst(I.getOpcode(), Op0BO->getOperand(0), CUI,
                            Op0BO->getName());
            Op0BO->setName("");
            InsertNewInstBefore(NewShift, I);

            return BinaryOperator::create(Op0BO->getOpcode(), NewShift,
                                          NewRHS);
          }
        }

    // If this is a shift of a shift, see if we can fold the two together...
    if (ShiftInst *Op0SI = dyn_cast<ShiftInst>(Op0))
      if (ConstantUInt *ShiftAmt1C =
                                 dyn_cast<ConstantUInt>(Op0SI->getOperand(1))) {
        unsigned ShiftAmt1 = ShiftAmt1C->getValue();
        unsigned ShiftAmt2 = CUI->getValue();
        
        // Check for (A << c1) << c2   and   (A >> c1) >> c2
        if (I.getOpcode() == Op0SI->getOpcode()) {
          unsigned Amt = ShiftAmt1+ShiftAmt2;   // Fold into one big shift...
          if (Op0->getType()->getPrimitiveSize()*8 < Amt)
            Amt = Op0->getType()->getPrimitiveSize()*8;
          return new ShiftInst(I.getOpcode(), Op0SI->getOperand(0),
                               ConstantUInt::get(Type::UByteTy, Amt));
        }
        
        // Check for (A << c1) >> c2 or visaversa.  If we are dealing with
        // signed types, we can only support the (A >> c1) << c2 configuration,
        // because it can not turn an arbitrary bit of A into a sign bit.
        if (I.getType()->isUnsigned() || isLeftShift) {
          // Calculate bitmask for what gets shifted off the edge...
          Constant *C = ConstantIntegral::getAllOnesValue(I.getType());
          if (isLeftShift)
            C = ConstantExpr::get(Instruction::Shl, C, ShiftAmt1C);
          else
            C = ConstantExpr::get(Instruction::Shr, C, ShiftAmt1C);
          
          Instruction *Mask =
            BinaryOperator::create(Instruction::And, Op0SI->getOperand(0),
                                   C, Op0SI->getOperand(0)->getName()+".mask");
          InsertNewInstBefore(Mask, I);
          
          // Figure out what flavor of shift we should use...
          if (ShiftAmt1 == ShiftAmt2)
            return ReplaceInstUsesWith(I, Mask);  // (A << c) >> c  === A & c2
          else if (ShiftAmt1 < ShiftAmt2) {
            return new ShiftInst(I.getOpcode(), Mask,
                         ConstantUInt::get(Type::UByteTy, ShiftAmt2-ShiftAmt1));
          } else {
            return new ShiftInst(Op0SI->getOpcode(), Mask,
                         ConstantUInt::get(Type::UByteTy, ShiftAmt1-ShiftAmt2));
          }
        }
      }
  }

  return 0;
}


// isEliminableCastOfCast - Return true if it is valid to eliminate the CI
// instruction.
//
static inline bool isEliminableCastOfCast(const Type *SrcTy, const Type *MidTy,
                                          const Type *DstTy) {

  // It is legal to eliminate the instruction if casting A->B->A if the sizes
  // are identical and the bits don't get reinterpreted (for example 
  // int->float->int would not be allowed)
  if (SrcTy == DstTy && SrcTy->isLosslesslyConvertibleTo(MidTy))
    return true;

  // Allow free casting and conversion of sizes as long as the sign doesn't
  // change...
  if (SrcTy->isIntegral() && MidTy->isIntegral() && DstTy->isIntegral()) {
    unsigned SrcSize = SrcTy->getPrimitiveSize();
    unsigned MidSize = MidTy->getPrimitiveSize();
    unsigned DstSize = DstTy->getPrimitiveSize();

    // Cases where we are monotonically decreasing the size of the type are
    // always ok, regardless of what sign changes are going on.
    //
    if (SrcSize >= MidSize && MidSize >= DstSize)
      return true;

    // Cases where the source and destination type are the same, but the middle
    // type is bigger are noops.
    //
    if (SrcSize == DstSize && MidSize > SrcSize)
      return true;

    // If we are monotonically growing, things are more complex.
    //
    if (SrcSize <= MidSize && MidSize <= DstSize) {
      // We have eight combinations of signedness to worry about. Here's the
      // table:
      static const int SignTable[8] = {
        // CODE, SrcSigned, MidSigned, DstSigned, Comment
        1,     //   U          U          U       Always ok
        1,     //   U          U          S       Always ok
        3,     //   U          S          U       Ok iff SrcSize != MidSize
        3,     //   U          S          S       Ok iff SrcSize != MidSize
        0,     //   S          U          U       Never ok
        2,     //   S          U          S       Ok iff MidSize == DstSize
        1,     //   S          S          U       Always ok
        1,     //   S          S          S       Always ok
      };

      // Choose an action based on the current entry of the signtable that this
      // cast of cast refers to...
      unsigned Row = SrcTy->isSigned()*4+MidTy->isSigned()*2+DstTy->isSigned();
      switch (SignTable[Row]) {
      case 0: return false;              // Never ok
      case 1: return true;               // Always ok
      case 2: return MidSize == DstSize; // Ok iff MidSize == DstSize
      case 3:                            // Ok iff SrcSize != MidSize
        return SrcSize != MidSize || SrcTy == Type::BoolTy;
      default: assert(0 && "Bad entry in sign table!");
      }
    }
  }

  // Otherwise, we cannot succeed.  Specifically we do not want to allow things
  // like:  short -> ushort -> uint, because this can create wrong results if
  // the input short is negative!
  //
  return false;
}

static bool ValueRequiresCast(const Value *V, const Type *Ty) {
  if (V->getType() == Ty || isa<Constant>(V)) return false;
  if (const CastInst *CI = dyn_cast<CastInst>(V))
    if (isEliminableCastOfCast(CI->getOperand(0)->getType(), CI->getType(), Ty))
      return false;
  return true;
}

/// InsertOperandCastBefore - This inserts a cast of V to DestTy before the
/// InsertBefore instruction.  This is specialized a bit to avoid inserting
/// casts that are known to not do anything...
///
Value *InstCombiner::InsertOperandCastBefore(Value *V, const Type *DestTy,
                                             Instruction *InsertBefore) {
  if (V->getType() == DestTy) return V;
  if (Constant *C = dyn_cast<Constant>(V))
    return ConstantExpr::getCast(C, DestTy);

  CastInst *CI = new CastInst(V, DestTy, V->getName());
  InsertNewInstBefore(CI, *InsertBefore);
  return CI;
}

// CastInst simplification
//
Instruction *InstCombiner::visitCastInst(CastInst &CI) {
  Value *Src = CI.getOperand(0);

  // If the user is casting a value to the same type, eliminate this cast
  // instruction...
  if (CI.getType() == Src->getType())
    return ReplaceInstUsesWith(CI, Src);

  // If casting the result of another cast instruction, try to eliminate this
  // one!
  //
  if (CastInst *CSrc = dyn_cast<CastInst>(Src)) {
    if (isEliminableCastOfCast(CSrc->getOperand(0)->getType(),
                               CSrc->getType(), CI.getType())) {
      // This instruction now refers directly to the cast's src operand.  This
      // has a good chance of making CSrc dead.
      CI.setOperand(0, CSrc->getOperand(0));
      return &CI;
    }

    // If this is an A->B->A cast, and we are dealing with integral types, try
    // to convert this into a logical 'and' instruction.
    //
    if (CSrc->getOperand(0)->getType() == CI.getType() &&
        CI.getType()->isInteger() && CSrc->getType()->isInteger() &&
        CI.getType()->isUnsigned() && CSrc->getType()->isUnsigned() &&
        CSrc->getType()->getPrimitiveSize() < CI.getType()->getPrimitiveSize()){
      assert(CSrc->getType() != Type::ULongTy &&
             "Cannot have type bigger than ulong!");
      uint64_t AndValue = (1ULL << CSrc->getType()->getPrimitiveSize()*8)-1;
      Constant *AndOp = ConstantUInt::get(CI.getType(), AndValue);
      return BinaryOperator::create(Instruction::And, CSrc->getOperand(0),
                                    AndOp);
    }
  }

  // If casting the result of a getelementptr instruction with no offset, turn
  // this into a cast of the original pointer!
  //
  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Src)) {
    bool AllZeroOperands = true;
    for (unsigned i = 1, e = GEP->getNumOperands(); i != e; ++i)
      if (!isa<Constant>(GEP->getOperand(i)) ||
          !cast<Constant>(GEP->getOperand(i))->isNullValue()) {
        AllZeroOperands = false;
        break;
      }
    if (AllZeroOperands) {
      CI.setOperand(0, GEP->getOperand(0));
      return &CI;
    }
  }

  // If we are casting a malloc or alloca to a pointer to a type of the same
  // size, rewrite the allocation instruction to allocate the "right" type.
  //
  if (AllocationInst *AI = dyn_cast<AllocationInst>(Src))
    if (AI->hasOneUse() && !AI->isArrayAllocation())
      if (const PointerType *PTy = dyn_cast<PointerType>(CI.getType())) {
        // Get the type really allocated and the type casted to...
        const Type *AllocElTy = AI->getAllocatedType();
        unsigned AllocElTySize = TD->getTypeSize(AllocElTy);
        const Type *CastElTy = PTy->getElementType();
        unsigned CastElTySize = TD->getTypeSize(CastElTy);

        // If the allocation is for an even multiple of the cast type size
        if (CastElTySize && (AllocElTySize % CastElTySize == 0)) {
          Value *Amt = ConstantUInt::get(Type::UIntTy, 
                                         AllocElTySize/CastElTySize);
          std::string Name = AI->getName(); AI->setName("");
          AllocationInst *New;
          if (isa<MallocInst>(AI))
            New = new MallocInst(CastElTy, Amt, Name);
          else
            New = new AllocaInst(CastElTy, Amt, Name);
          InsertNewInstBefore(New, CI);
          return ReplaceInstUsesWith(CI, New);
        }
      }

  // If the source value is an instruction with only this use, we can attempt to
  // propagate the cast into the instruction.  Also, only handle integral types
  // for now.
  if (Instruction *SrcI = dyn_cast<Instruction>(Src))
    if (SrcI->hasOneUse() && Src->getType()->isIntegral() &&
        CI.getType()->isInteger()) {  // Don't mess with casts to bool here
      const Type *DestTy = CI.getType();
      unsigned SrcBitSize = getTypeSizeInBits(Src->getType());
      unsigned DestBitSize = getTypeSizeInBits(DestTy);

      Value *Op0 = SrcI->getNumOperands() > 0 ? SrcI->getOperand(0) : 0;
      Value *Op1 = SrcI->getNumOperands() > 1 ? SrcI->getOperand(1) : 0;

      switch (SrcI->getOpcode()) {
      case Instruction::Add:
      case Instruction::Mul:
      case Instruction::And:
      case Instruction::Or:
      case Instruction::Xor:
        // If we are discarding information, or just changing the sign, rewrite.
        if (DestBitSize <= SrcBitSize && DestBitSize != 1) {
          // Don't insert two casts if they cannot be eliminated.  We allow two
          // casts to be inserted if the sizes are the same.  This could only be
          // converting signedness, which is a noop.
          if (DestBitSize == SrcBitSize || !ValueRequiresCast(Op1, DestTy) ||
              !ValueRequiresCast(Op0, DestTy)) {
            Value *Op0c = InsertOperandCastBefore(Op0, DestTy, SrcI);
            Value *Op1c = InsertOperandCastBefore(Op1, DestTy, SrcI);
            return BinaryOperator::create(cast<BinaryOperator>(SrcI)
                             ->getOpcode(), Op0c, Op1c);
          }
        }
        break;
      case Instruction::Shl:
        // Allow changing the sign of the source operand.  Do not allow changing
        // the size of the shift, UNLESS the shift amount is a constant.  We
        // mush not change variable sized shifts to a smaller size, because it
        // is undefined to shift more bits out than exist in the value.
        if (DestBitSize == SrcBitSize ||
            (DestBitSize < SrcBitSize && isa<Constant>(Op1))) {
          Value *Op0c = InsertOperandCastBefore(Op0, DestTy, SrcI);
          return new ShiftInst(Instruction::Shl, Op0c, Op1);
        }
        break;
      }
    }
  
  return 0;
}

Instruction *InstCombiner::visitSelectInst(SelectInst &SI) {
  if (ConstantBool *C = dyn_cast<ConstantBool>(SI.getCondition()))
    if (C == ConstantBool::True)
      return ReplaceInstUsesWith(SI, SI.getTrueValue());
    else {
      assert(C == ConstantBool::False);
      return ReplaceInstUsesWith(SI, SI.getFalseValue());
    }
  // Other transformations are possible!

  return 0;
}


// CallInst simplification
//
Instruction *InstCombiner::visitCallInst(CallInst &CI) {
  // Intrinsics cannot occur in an invoke, so handle them here instead of in
  // visitCallSite.
  if (Function *F = CI.getCalledFunction())
    switch (F->getIntrinsicID()) {
    case Intrinsic::memmove:
    case Intrinsic::memcpy:
    case Intrinsic::memset:
      // memmove/cpy/set of zero bytes is a noop.
      if (Constant *NumBytes = dyn_cast<Constant>(CI.getOperand(3))) {
        if (NumBytes->isNullValue())
          return EraseInstFromFunction(CI);
      }
      break;
    default:
      break;
    }

  return visitCallSite(&CI);
}

// InvokeInst simplification
//
Instruction *InstCombiner::visitInvokeInst(InvokeInst &II) {
  return visitCallSite(&II);
}

// visitCallSite - Improvements for call and invoke instructions.
//
Instruction *InstCombiner::visitCallSite(CallSite CS) {
  bool Changed = false;

  // If the callee is a constexpr cast of a function, attempt to move the cast
  // to the arguments of the call/invoke.
  if (transformConstExprCastCall(CS)) return 0;

  Value *Callee = CS.getCalledValue();
  const PointerType *PTy = cast<PointerType>(Callee->getType());
  const FunctionType *FTy = cast<FunctionType>(PTy->getElementType());
  if (FTy->isVarArg()) {
    // See if we can optimize any arguments passed through the varargs area of
    // the call.
    for (CallSite::arg_iterator I = CS.arg_begin()+FTy->getNumParams(),
           E = CS.arg_end(); I != E; ++I)
      if (CastInst *CI = dyn_cast<CastInst>(*I)) {
        // If this cast does not effect the value passed through the varargs
        // area, we can eliminate the use of the cast.
        Value *Op = CI->getOperand(0);
        if (CI->getType()->isLosslesslyConvertibleTo(Op->getType())) {
          *I = Op;
          Changed = true;
        }
      }
  }
  
  return Changed ? CS.getInstruction() : 0;
}

// transformConstExprCastCall - If the callee is a constexpr cast of a function,
// attempt to move the cast to the arguments of the call/invoke.
//
bool InstCombiner::transformConstExprCastCall(CallSite CS) {
  if (!isa<ConstantExpr>(CS.getCalledValue())) return false;
  ConstantExpr *CE = cast<ConstantExpr>(CS.getCalledValue());
  if (CE->getOpcode() != Instruction::Cast ||
      !isa<ConstantPointerRef>(CE->getOperand(0)))
    return false;
  ConstantPointerRef *CPR = cast<ConstantPointerRef>(CE->getOperand(0));
  if (!isa<Function>(CPR->getValue())) return false;
  Function *Callee = cast<Function>(CPR->getValue());
  Instruction *Caller = CS.getInstruction();

  // Okay, this is a cast from a function to a different type.  Unless doing so
  // would cause a type conversion of one of our arguments, change this call to
  // be a direct call with arguments casted to the appropriate types.
  //
  const FunctionType *FT = Callee->getFunctionType();
  const Type *OldRetTy = Caller->getType();

  // Check to see if we are changing the return type...
  if (OldRetTy != FT->getReturnType()) {
    if (Callee->isExternal() &&
        !OldRetTy->isLosslesslyConvertibleTo(FT->getReturnType()) &&
        !Caller->use_empty())
      return false;   // Cannot transform this return value...

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
    bool isConvertible = (*AI)->getType()->isLosslesslyConvertibleTo(ParamTy);
    if (Callee->isExternal() && !isConvertible) return false;    
  }

  if (FT->getNumParams() < NumActualArgs && !FT->isVarArg() &&
      Callee->isExternal())
    return false;   // Do not delete arguments unless we have a function body...

  // Okay, we decided that this is a safe thing to do: go ahead and start
  // inserting cast instructions as necessary...
  std::vector<Value*> Args;
  Args.reserve(NumActualArgs);

  AI = CS.arg_begin();
  for (unsigned i = 0; i != NumCommonArgs; ++i, ++AI) {
    const Type *ParamTy = FT->getParamType(i);
    if ((*AI)->getType() == ParamTy) {
      Args.push_back(*AI);
    } else {
      Instruction *Cast = new CastInst(*AI, ParamTy, "tmp");
      InsertNewInstBefore(Cast, *Caller);
      Args.push_back(Cast);
    }
  }

  // If the function takes more arguments than the call was taking, add them
  // now...
  for (unsigned i = NumCommonArgs; i != FT->getNumParams(); ++i)
    Args.push_back(Constant::getNullValue(FT->getParamType(i)));

  // If we are removing arguments to the function, emit an obnoxious warning...
  if (FT->getNumParams() < NumActualArgs)
    if (!FT->isVarArg()) {
      std::cerr << "WARNING: While resolving call to function '"
                << Callee->getName() << "' arguments were dropped!\n";
    } else {
      // Add all of the arguments in their promoted form to the arg list...
      for (unsigned i = FT->getNumParams(); i != NumActualArgs; ++i, ++AI) {
        const Type *PTy = getPromotedType((*AI)->getType());
        if (PTy != (*AI)->getType()) {
          // Must promote to pass through va_arg area!
          Instruction *Cast = new CastInst(*AI, PTy, "tmp");
          InsertNewInstBefore(Cast, *Caller);
          Args.push_back(Cast);
        } else {
          Args.push_back(*AI);
        }
      }
    }

  if (FT->getReturnType() == Type::VoidTy)
    Caller->setName("");   // Void type should not have a name...

  Instruction *NC;
  if (InvokeInst *II = dyn_cast<InvokeInst>(Caller)) {
    NC = new InvokeInst(Callee, II->getNormalDest(), II->getUnwindDest(),
                        Args, Caller->getName(), Caller);
  } else {
    NC = new CallInst(Callee, Args, Caller->getName(), Caller);
  }

  // Insert a cast of the return type as necessary...
  Value *NV = NC;
  if (Caller->getType() != NV->getType() && !Caller->use_empty()) {
    if (NV->getType() != Type::VoidTy) {
      NV = NC = new CastInst(NC, Caller->getType(), "tmp");

      // If this is an invoke instruction, we should insert it after the first
      // non-phi, instruction in the normal successor block.
      if (InvokeInst *II = dyn_cast<InvokeInst>(Caller)) {
        BasicBlock::iterator I = II->getNormalDest()->begin();
        while (isa<PHINode>(I)) ++I;
        InsertNewInstBefore(NC, *I);
      } else {
        // Otherwise, it's a call, just insert cast right after the call instr
        InsertNewInstBefore(NC, *Caller);
      }
      AddUsersToWorkList(*Caller);
    } else {
      NV = Constant::getNullValue(Caller->getType());
    }
  }

  if (Caller->getType() != Type::VoidTy && !Caller->use_empty())
    Caller->replaceAllUsesWith(NV);
  Caller->getParent()->getInstList().erase(Caller);
  removeFromWorkList(Caller);
  return true;
}



// PHINode simplification
//
Instruction *InstCombiner::visitPHINode(PHINode &PN) {
  if (Value *V = hasConstantValue(&PN))
    return ReplaceInstUsesWith(PN, V);

  // If the only user of this instruction is a cast instruction, and all of the
  // incoming values are constants, change this PHI to merge together the casted
  // constants.
  if (PN.hasOneUse())
    if (CastInst *CI = dyn_cast<CastInst>(PN.use_back()))
      if (CI->getType() != PN.getType()) {  // noop casts will be folded
        bool AllConstant = true;
        for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i)
          if (!isa<Constant>(PN.getIncomingValue(i))) {
            AllConstant = false;
            break;
          }
        if (AllConstant) {
          // Make a new PHI with all casted values.
          PHINode *New = new PHINode(CI->getType(), PN.getName(), &PN);
          for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i) {
            Constant *OldArg = cast<Constant>(PN.getIncomingValue(i));
            New->addIncoming(ConstantExpr::getCast(OldArg, New->getType()),
                             PN.getIncomingBlock(i));
          }

          // Update the cast instruction.
          CI->setOperand(0, New);
          WorkList.push_back(CI);    // revisit the cast instruction to fold.
          WorkList.push_back(New);   // Make sure to revisit the new Phi
          return &PN;                // PN is now dead!
        }
      }
  return 0;
}


Instruction *InstCombiner::visitGetElementPtrInst(GetElementPtrInst &GEP) {
  // Is it 'getelementptr %P, long 0'  or 'getelementptr %P'
  // If so, eliminate the noop.
  if (GEP.getNumOperands() == 1)
    return ReplaceInstUsesWith(GEP, GEP.getOperand(0));

  bool HasZeroPointerIndex = false;
  if (Constant *C = dyn_cast<Constant>(GEP.getOperand(1)))
    HasZeroPointerIndex = C->isNullValue();

  if (GEP.getNumOperands() == 2 && HasZeroPointerIndex)
    return ReplaceInstUsesWith(GEP, GEP.getOperand(0));

  // Combine Indices - If the source pointer to this getelementptr instruction
  // is a getelementptr instruction, combine the indices of the two
  // getelementptr instructions into a single instruction.
  //
  std::vector<Value*> SrcGEPOperands;
  if (GetElementPtrInst *Src = dyn_cast<GetElementPtrInst>(GEP.getOperand(0))) {
    SrcGEPOperands.assign(Src->op_begin(), Src->op_end());
  } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(GEP.getOperand(0))) {
    if (CE->getOpcode() == Instruction::GetElementPtr)
      SrcGEPOperands.assign(CE->op_begin(), CE->op_end());
  }

  if (!SrcGEPOperands.empty()) {
    std::vector<Value *> Indices;
  
    // Can we combine the two pointer arithmetics offsets?
    if (SrcGEPOperands.size() == 2 && isa<Constant>(SrcGEPOperands[1]) &&
        isa<Constant>(GEP.getOperand(1))) {
      // Replace: gep (gep %P, long C1), long C2, ...
      // With:    gep %P, long (C1+C2), ...
      Value *Sum = ConstantExpr::get(Instruction::Add,
                                     cast<Constant>(SrcGEPOperands[1]),
                                     cast<Constant>(GEP.getOperand(1)));
      assert(Sum && "Constant folding of longs failed!?");
      GEP.setOperand(0, SrcGEPOperands[0]);
      GEP.setOperand(1, Sum);
      if (Instruction *I = dyn_cast<Instruction>(GEP.getOperand(0)))
        AddUsersToWorkList(*I);   // Reduce use count of Src
      return &GEP;
    } else if (SrcGEPOperands.size() == 2) {
      // Replace: gep (gep %P, long B), long A, ...
      // With:    T = long A+B; gep %P, T, ...
      //
      // Note that if our source is a gep chain itself that we wait for that
      // chain to be resolved before we perform this transformation.  This
      // avoids us creating a TON of code in some cases.
      //
      if (isa<GetElementPtrInst>(SrcGEPOperands[0]) &&
          cast<Instruction>(SrcGEPOperands[0])->getNumOperands() == 2)
        return 0;   // Wait until our source is folded to completion.

      Value *Sum = BinaryOperator::create(Instruction::Add, SrcGEPOperands[1],
                                          GEP.getOperand(1),
                                          GEP.getOperand(0)->getName()+".sum",
                                          &GEP);
      GEP.setOperand(0, SrcGEPOperands[0]);
      GEP.setOperand(1, Sum);
      WorkList.push_back(cast<Instruction>(Sum));
      return &GEP;
    } else if (*GEP.idx_begin() == Constant::getNullValue(Type::LongTy) &&
               SrcGEPOperands.size() != 1) { 
      // Otherwise we can do the fold if the first index of the GEP is a zero
      Indices.insert(Indices.end(), SrcGEPOperands.begin()+1,
                     SrcGEPOperands.end());
      Indices.insert(Indices.end(), GEP.idx_begin()+1, GEP.idx_end());
    } else if (SrcGEPOperands.back() == Constant::getNullValue(Type::LongTy)) {
      // FIXME: when we allow indices to be non-long values, support this for
      // other types!

      // If the src gep ends with a constant array index, merge this get into
      // it, even if we have a non-zero array index.
      Indices.insert(Indices.end(), SrcGEPOperands.begin()+1,
                     SrcGEPOperands.end()-1);
      Indices.insert(Indices.end(), GEP.idx_begin(), GEP.idx_end());
    }

    if (!Indices.empty())
      return new GetElementPtrInst(SrcGEPOperands[0], Indices, GEP.getName());

  } else if (GlobalValue *GV = dyn_cast<GlobalValue>(GEP.getOperand(0))) {
    // GEP of global variable.  If all of the indices for this GEP are
    // constants, we can promote this to a constexpr instead of an instruction.

    // Scan for nonconstants...
    std::vector<Constant*> Indices;
    User::op_iterator I = GEP.idx_begin(), E = GEP.idx_end();
    for (; I != E && isa<Constant>(*I); ++I)
      Indices.push_back(cast<Constant>(*I));

    if (I == E) {  // If they are all constants...
      Constant *CE =
        ConstantExpr::getGetElementPtr(ConstantPointerRef::get(GV), Indices);

      // Replace all uses of the GEP with the new constexpr...
      return ReplaceInstUsesWith(GEP, CE);
    }
  } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(GEP.getOperand(0))) {
    if (CE->getOpcode() == Instruction::Cast) {
      if (HasZeroPointerIndex) {
        // transform: GEP (cast [10 x ubyte]* X to [0 x ubyte]*), long 0, ...
        // into     : GEP [10 x ubyte]* X, long 0, ...
        //
        // This occurs when the program declares an array extern like "int X[];"
        //
        Constant *X = CE->getOperand(0);
        const PointerType *CPTy = cast<PointerType>(CE->getType());
        if (const PointerType *XTy = dyn_cast<PointerType>(X->getType()))
          if (const ArrayType *XATy =
              dyn_cast<ArrayType>(XTy->getElementType()))
            if (const ArrayType *CATy =
                dyn_cast<ArrayType>(CPTy->getElementType()))
              if (CATy->getElementType() == XATy->getElementType()) {
                // At this point, we know that the cast source type is a pointer
                // to an array of the same type as the destination pointer
                // array.  Because the array type is never stepped over (there
                // is a leading zero) we can fold the cast into this GEP.
                GEP.setOperand(0, X);
                return &GEP;
              }
      }
    }
  }

  return 0;
}

Instruction *InstCombiner::visitAllocationInst(AllocationInst &AI) {
  // Convert: malloc Ty, C - where C is a constant != 1 into: malloc [C x Ty], 1
  if (AI.isArrayAllocation())    // Check C != 1
    if (const ConstantUInt *C = dyn_cast<ConstantUInt>(AI.getArraySize())) {
      const Type *NewTy = ArrayType::get(AI.getAllocatedType(), C->getValue());
      AllocationInst *New = 0;

      // Create and insert the replacement instruction...
      if (isa<MallocInst>(AI))
        New = new MallocInst(NewTy, 0, AI.getName());
      else {
        assert(isa<AllocaInst>(AI) && "Unknown type of allocation inst!");
        New = new AllocaInst(NewTy, 0, AI.getName());
      }

      InsertNewInstBefore(New, AI);
      
      // Scan to the end of the allocation instructions, to skip over a block of
      // allocas if possible...
      //
      BasicBlock::iterator It = New;
      while (isa<AllocationInst>(*It)) ++It;

      // Now that I is pointing to the first non-allocation-inst in the block,
      // insert our getelementptr instruction...
      //
      std::vector<Value*> Idx(2, Constant::getNullValue(Type::LongTy));
      Value *V = new GetElementPtrInst(New, Idx, New->getName()+".sub", It);

      // Now make everything use the getelementptr instead of the original
      // allocation.
      return ReplaceInstUsesWith(AI, V);
    }

  // If alloca'ing a zero byte object, replace the alloca with a null pointer.
  // Note that we only do this for alloca's, because malloc should allocate and
  // return a unique pointer, even for a zero byte allocation.
  if (isa<AllocaInst>(AI) && TD->getTypeSize(AI.getAllocatedType()) == 0)
    return ReplaceInstUsesWith(AI, Constant::getNullValue(AI.getType()));

  return 0;
}

Instruction *InstCombiner::visitFreeInst(FreeInst &FI) {
  Value *Op = FI.getOperand(0);

  // Change free <ty>* (cast <ty2>* X to <ty>*) into free <ty2>* X
  if (CastInst *CI = dyn_cast<CastInst>(Op))
    if (isa<PointerType>(CI->getOperand(0)->getType())) {
      FI.setOperand(0, CI->getOperand(0));
      return &FI;
    }

  // If we have 'free null' delete the instruction.  This can happen in stl code
  // when lots of inlining happens.
  if (isa<ConstantPointerNull>(Op))
    return EraseInstFromFunction(FI);

  return 0;
}


/// GetGEPGlobalInitializer - Given a constant, and a getelementptr
/// constantexpr, return the constant value being addressed by the constant
/// expression, or null if something is funny.
///
static Constant *GetGEPGlobalInitializer(Constant *C, ConstantExpr *CE) {
  if (CE->getOperand(1) != Constant::getNullValue(Type::LongTy))
    return 0;  // Do not allow stepping over the value!

  // Loop over all of the operands, tracking down which value we are
  // addressing...
  for (unsigned i = 2, e = CE->getNumOperands(); i != e; ++i)
    if (ConstantUInt *CU = dyn_cast<ConstantUInt>(CE->getOperand(i))) {
      ConstantStruct *CS = dyn_cast<ConstantStruct>(C);
      if (CS == 0) return 0;
      if (CU->getValue() >= CS->getValues().size()) return 0;
      C = cast<Constant>(CS->getValues()[CU->getValue()]);
    } else if (ConstantSInt *CS = dyn_cast<ConstantSInt>(CE->getOperand(i))) {
      ConstantArray *CA = dyn_cast<ConstantArray>(C);
      if (CA == 0) return 0;
      if ((uint64_t)CS->getValue() >= CA->getValues().size()) return 0;
      C = cast<Constant>(CA->getValues()[CS->getValue()]);
    } else 
      return 0;
  return C;
}

Instruction *InstCombiner::visitLoadInst(LoadInst &LI) {
  Value *Op = LI.getOperand(0);
  if (LI.isVolatile()) return 0;

  if (ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(Op))
    Op = CPR->getValue();

  // Instcombine load (constant global) into the value loaded...
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Op))
    if (GV->isConstant() && !GV->isExternal())
      return ReplaceInstUsesWith(LI, GV->getInitializer());

  // Instcombine load (constantexpr_GEP global, 0, ...) into the value loaded...
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Op))
    if (CE->getOpcode() == Instruction::GetElementPtr)
      if (ConstantPointerRef *G=dyn_cast<ConstantPointerRef>(CE->getOperand(0)))
        if (GlobalVariable *GV = dyn_cast<GlobalVariable>(G->getValue()))
          if (GV->isConstant() && !GV->isExternal())
            if (Constant *V = GetGEPGlobalInitializer(GV->getInitializer(), CE))
              return ReplaceInstUsesWith(LI, V);
  return 0;
}


Instruction *InstCombiner::visitBranchInst(BranchInst &BI) {
  // Change br (not X), label True, label False to: br X, label False, True
  if (BI.isConditional() && !isa<Constant>(BI.getCondition())) {
    if (Value *V = dyn_castNotVal(BI.getCondition())) {
      BasicBlock *TrueDest = BI.getSuccessor(0);
      BasicBlock *FalseDest = BI.getSuccessor(1);
      // Swap Destinations and condition...
      BI.setCondition(V);
      BI.setSuccessor(0, FalseDest);
      BI.setSuccessor(1, TrueDest);
      return &BI;
    } else if (SetCondInst *I = dyn_cast<SetCondInst>(BI.getCondition())) {
      // Cannonicalize setne -> seteq
      if ((I->getOpcode() == Instruction::SetNE ||
           I->getOpcode() == Instruction::SetLE ||
           I->getOpcode() == Instruction::SetGE) && I->hasOneUse()) {
        std::string Name = I->getName(); I->setName("");
        Instruction::BinaryOps NewOpcode =
          SetCondInst::getInverseCondition(I->getOpcode());
        Value *NewSCC =  BinaryOperator::create(NewOpcode, I->getOperand(0),
                                                I->getOperand(1), Name, I);
        BasicBlock *TrueDest = BI.getSuccessor(0);
        BasicBlock *FalseDest = BI.getSuccessor(1);
        // Swap Destinations and condition...
        BI.setCondition(NewSCC);
        BI.setSuccessor(0, FalseDest);
        BI.setSuccessor(1, TrueDest);
        removeFromWorkList(I);
        I->getParent()->getInstList().erase(I);
        WorkList.push_back(cast<Instruction>(NewSCC));
        return &BI;
      }
    }
  }
  return 0;
}


void InstCombiner::removeFromWorkList(Instruction *I) {
  WorkList.erase(std::remove(WorkList.begin(), WorkList.end(), I),
                 WorkList.end());
}

bool InstCombiner::runOnFunction(Function &F) {
  bool Changed = false;
  TD = &getAnalysis<TargetData>();

  WorkList.insert(WorkList.end(), inst_begin(F), inst_end(F));

  while (!WorkList.empty()) {
    Instruction *I = WorkList.back();  // Get an instruction from the worklist
    WorkList.pop_back();

    // Check to see if we can DCE or ConstantPropagate the instruction...
    // Check to see if we can DIE the instruction...
    if (isInstructionTriviallyDead(I)) {
      // Add operands to the worklist...
      if (I->getNumOperands() < 4)
        AddUsesToWorkList(*I);
      ++NumDeadInst;

      I->getParent()->getInstList().erase(I);
      removeFromWorkList(I);
      continue;
    }

    // Instruction isn't dead, see if we can constant propagate it...
    if (Constant *C = ConstantFoldInstruction(I)) {
      // Add operands to the worklist...
      AddUsesToWorkList(*I);
      ReplaceInstUsesWith(*I, C);

      ++NumConstProp;
      I->getParent()->getInstList().erase(I);
      removeFromWorkList(I);
      continue;
    }

    // Check to see if any of the operands of this instruction are a
    // ConstantPointerRef.  Since they sneak in all over the place and inhibit
    // optimization, we want to strip them out unconditionally!
    for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
      if (ConstantPointerRef *CPR =
          dyn_cast<ConstantPointerRef>(I->getOperand(i))) {
        I->setOperand(i, CPR->getValue());
        Changed = true;
      }

    // Now that we have an instruction, try combining it to simplify it...
    if (Instruction *Result = visit(*I)) {
      ++NumCombined;
      // Should we replace the old instruction with a new one?
      if (Result != I) {
        DEBUG(std::cerr << "IC: Old = " << *I
                        << "    New = " << *Result);

        // Instructions can end up on the worklist more than once.  Make sure
        // we do not process an instruction that has been deleted.
        removeFromWorkList(I);

        // Move the name to the new instruction first...
        std::string OldName = I->getName(); I->setName("");
        Result->setName(OldName);

        // Insert the new instruction into the basic block...
        BasicBlock *InstParent = I->getParent();
        InstParent->getInstList().insert(I, Result);

        // Everything uses the new instruction now...
        I->replaceAllUsesWith(Result);

        // Erase the old instruction.
        InstParent->getInstList().erase(I);
      } else {
        DEBUG(std::cerr << "IC: MOD = " << *I);

        BasicBlock::iterator II = I;

        // If the instruction was modified, it's possible that it is now dead.
        // if so, remove it.
        if (dceInstruction(II)) {
          // Instructions may end up in the worklist more than once.  Erase them
          // all.
          removeFromWorkList(I);
          Result = 0;
        }
      }

      if (Result) {
        WorkList.push_back(Result);
        AddUsersToWorkList(*Result);
      }
      Changed = true;
    }
  }

  return Changed;
}

Pass *llvm::createInstructionCombiningPass() {
  return new InstCombiner();
}

