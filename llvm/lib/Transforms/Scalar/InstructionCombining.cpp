//===- InstructionCombining.cpp - Combine multiple instructions -----------===//
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
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/ConstantHandling.h"
#include "llvm/iMemory.h"
#include "llvm/iOther.h"
#include "llvm/iPHINode.h"
#include "llvm/iOperators.h"
#include "llvm/Pass.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Support/InstVisitor.h"
#include "Support/Statistic.h"
#include <algorithm>

namespace {
  Statistic<> NumCombined ("instcombine", "Number of insts combined");
  Statistic<> NumConstProp("instcombine", "Number of constant folds");
  Statistic<> NumDeadInst ("instcombine", "Number of dead inst eliminated");

  class InstCombiner : public FunctionPass,
                       public InstVisitor<InstCombiner, Instruction*> {
    // Worklist of all of the instructions that need to be simplified.
    std::vector<Instruction*> WorkList;

    void AddUsesToWorkList(Instruction &I) {
      // The instruction was simplified, add all users of the instruction to
      // the work lists because they might get more simplified now...
      //
      for (Value::use_iterator UI = I.use_begin(), UE = I.use_end();
           UI != UE; ++UI)
        WorkList.push_back(cast<Instruction>(*UI));
    }

    // removeFromWorkList - remove all instances of I from the worklist.
    void removeFromWorkList(Instruction *I);
  public:
    virtual bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
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
    Instruction *visitPHINode(PHINode &PN);
    Instruction *visitGetElementPtrInst(GetElementPtrInst &GEP);
    Instruction *visitAllocationInst(AllocationInst &AI);

    // visitInstruction - Specify what to return for unhandled instructions...
    Instruction *visitInstruction(Instruction &I) { return 0; }

    // InsertNewInstBefore - insert an instruction New before instruction Old
    // in the program.  Add the new instruction to the worklist.
    //
    void InsertNewInstBefore(Instruction *New, Instruction &Old) {
      assert(New && New->getParent() == 0 &&
             "New instruction already inserted into a basic block!");
      BasicBlock *BB = Old.getParent();
      BB->getInstList().insert(&Old, New);  // Insert inst
      WorkList.push_back(New);              // Add to worklist
    }

    // ReplaceInstUsesWith - This method is to be used when an instruction is
    // found to be dead, replacable with another preexisting expression.  Here
    // we add all uses of I to the worklist, replace all uses of I with the new
    // value, then return I, so that the inst combiner will know that I was
    // modified.
    //
    Instruction *ReplaceInstUsesWith(Instruction &I, Value *V) {
      AddUsesToWorkList(I);         // Add all modified instrs to worklist
      I.replaceAllUsesWith(V);
      return &I;
    }
  };

  RegisterOpt<InstCombiner> X("instcombine", "Combine redundant instructions");
}


// Make sure that this instruction has a constant on the right hand side if it
// has any constant arguments.  If not, fix it an return true.
//
static bool SimplifyBinOp(BinaryOperator &I) {
  if (isa<Constant>(I.getOperand(0)) && !isa<Constant>(I.getOperand(1)))
    return !I.swapOperands();
  return false;
}

// dyn_castNegInst - Given a 'sub' instruction, return the RHS of the
// instruction if the LHS is a constant zero (which is the 'negate' form).
//
static inline Value *dyn_castNegInst(Value *V) {
  return BinaryOperator::isNeg(V) ?
    BinaryOperator::getNegArgument(cast<BinaryOperator>(V)) : 0;
}

static inline Value *dyn_castNotInst(Value *V) {
  return BinaryOperator::isNot(V) ?
    BinaryOperator::getNotArgument(cast<BinaryOperator>(V)) : 0;
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

static inline Value *dyn_castFoldableMul(Value *V) {
  if (V->use_size() == 1 && V->getType()->isInteger())
    if (Instruction *I = dyn_cast<Instruction>(V))
      if (I->getOpcode() == Instruction::Mul)
        if (isa<Constant>(I->getOperand(1)))
          return I->getOperand(0);
  return 0;
}


Instruction *InstCombiner::visitAdd(BinaryOperator &I) {
  bool Changed = SimplifyBinOp(I);
  Value *LHS = I.getOperand(0), *RHS = I.getOperand(1);

  // Eliminate 'add int %X, 0'
  if (RHS == Constant::getNullValue(I.getType()))
    return ReplaceInstUsesWith(I, LHS);

  // -A + B  -->  B - A
  if (Value *V = dyn_castNegInst(LHS))
    return BinaryOperator::create(Instruction::Sub, RHS, V);

  // A + -B  -->  A - B
  if (Value *V = dyn_castNegInst(RHS))
    return BinaryOperator::create(Instruction::Sub, LHS, V);

  // Simplify add instructions with a constant RHS...
  if (Constant *Op2 = dyn_cast<Constant>(RHS)) {
    if (BinaryOperator *ILHS = dyn_cast<BinaryOperator>(LHS)) {
      if (ILHS->getOpcode() == Instruction::Add &&
          isa<Constant>(ILHS->getOperand(1))) {
        // Fold:
        //    %Y = add int %X, 1
        //    %Z = add int %Y, 1
        // into:
        //    %Z = add int %X, 2
        //
        if (Constant *Val = *Op2 + *cast<Constant>(ILHS->getOperand(1))) {
          I.setOperand(0, ILHS->getOperand(0));
          I.setOperand(1, Val);
          return &I;
        }
      }
    }
  }

  // X*C + X --> X * (C+1)
  if (dyn_castFoldableMul(LHS) == RHS) {
    Constant *CP1 = *cast<Constant>(cast<Instruction>(LHS)->getOperand(1)) +
                    *ConstantInt::get(I.getType(), 1);
    assert(CP1 && "Couldn't constant fold C + 1?");
    return BinaryOperator::create(Instruction::Mul, RHS, CP1);
  }

  // X + X*C --> X * (C+1)
  if (dyn_castFoldableMul(RHS) == LHS) {
    Constant *CP1 = *cast<Constant>(cast<Instruction>(RHS)->getOperand(1)) +
                    *ConstantInt::get(I.getType(), 1);
    assert(CP1 && "Couldn't constant fold C + 1?");
    return BinaryOperator::create(Instruction::Mul, LHS, CP1);
  }

  return Changed ? &I : 0;
}

Instruction *InstCombiner::visitSub(BinaryOperator &I) {
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Op0 == Op1)         // sub X, X  -> 0
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

  // If this is a subtract instruction with a constant RHS, convert it to an add
  // instruction of a negative constant
  //
  if (Constant *Op2 = dyn_cast<Constant>(Op1))
    if (Constant *RHS = *Constant::getNullValue(I.getType()) - *Op2) // 0 - RHS
      return BinaryOperator::create(Instruction::Add, Op0, RHS, I.getName());

  // If this is a 'B = x-(-A)', change to B = x+A...
  if (Value *V = dyn_castNegInst(Op1))
    return BinaryOperator::create(Instruction::Add, Op0, V);

  // Replace (-1 - A) with (~A)...
  if (ConstantInt *C = dyn_cast<ConstantInt>(Op0))
    if (C->isAllOnesValue())
      return BinaryOperator::createNot(Op1);

  if (BinaryOperator *Op1I = dyn_cast<BinaryOperator>(Op1))
    if (Op1I->use_size() == 1) {
      // Replace (x - (y - z)) with (x + (z - y)) if the (y - z) subexpression
      // is not used by anyone else...
      //
      if (Op1I->getOpcode() == Instruction::Sub) {
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
        Constant *CP1 = *ConstantInt::get(I.getType(), 1) -
                        *cast<Constant>(cast<Instruction>(Op1)->getOperand(1));
        assert(CP1 && "Couldn't constant fold 1-C?");
        return BinaryOperator::create(Instruction::Mul, Op0, CP1);
      }
    }

  // X*C - X --> X * (C-1)
  if (dyn_castFoldableMul(Op0) == Op1) {
    Constant *CP1 = *cast<Constant>(cast<Instruction>(Op0)->getOperand(1)) -
                    *ConstantInt::get(I.getType(), 1);
    assert(CP1 && "Couldn't constant fold C - 1?");
    return BinaryOperator::create(Instruction::Mul, Op1, CP1);
  }

  return 0;
}

Instruction *InstCombiner::visitMul(BinaryOperator &I) {
  bool Changed = SimplifyBinOp(I);
  Value *Op0 = I.getOperand(0);

  // Simplify mul instructions with a constant RHS...
  if (Constant *Op1 = dyn_cast<Constant>(I.getOperand(1))) {
    if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1)) {
      const Type *Ty = CI->getType();
      uint64_t Val = Ty->isSigned() ?
                          (uint64_t)cast<ConstantSInt>(CI)->getValue() : 
                                    cast<ConstantUInt>(CI)->getValue();
      switch (Val) {
      case 0:
        return ReplaceInstUsesWith(I, Op1);  // Eliminate 'mul double %X, 0'
      case 1:
        return ReplaceInstUsesWith(I, Op0);  // Eliminate 'mul int %X, 1'
      case 2:                     // Convert 'mul int %X, 2' to 'add int %X, %X'
        return BinaryOperator::create(Instruction::Add, Op0, Op0, I.getName());
      }

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


Instruction *InstCombiner::visitAnd(BinaryOperator &I) {
  bool Changed = SimplifyBinOp(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // and X, X = X   and X, 0 == 0
  if (Op0 == Op1 || Op1 == Constant::getNullValue(I.getType()))
    return ReplaceInstUsesWith(I, Op1);

  // and X, -1 == X
  if (ConstantIntegral *RHS = dyn_cast<ConstantIntegral>(Op1))
    if (RHS->isAllOnesValue())
      return ReplaceInstUsesWith(I, Op0);

  Value *Op0NotVal = dyn_castNotInst(Op0);
  Value *Op1NotVal = dyn_castNotInst(Op1);

  // (~A & ~B) == (~(A | B)) - Demorgan's Law
  if (Op0->use_size() == 1 && Op1->use_size() == 1 && Op0NotVal && Op1NotVal) {
    Instruction *Or = BinaryOperator::create(Instruction::Or, Op0NotVal,
                                             Op1NotVal,I.getName()+".demorgan",
                                             &I);
    return BinaryOperator::createNot(Or);
  }

  if (Op0NotVal == Op1 || Op1NotVal == Op0)  // A & ~A  == ~A & A == 0
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

  return Changed ? &I : 0;
}



Instruction *InstCombiner::visitOr(BinaryOperator &I) {
  bool Changed = SimplifyBinOp(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // or X, X = X   or X, 0 == X
  if (Op0 == Op1 || Op1 == Constant::getNullValue(I.getType()))
    return ReplaceInstUsesWith(I, Op0);

  // or X, -1 == -1
  if (ConstantIntegral *RHS = dyn_cast<ConstantIntegral>(Op1))
    if (RHS->isAllOnesValue())
      return ReplaceInstUsesWith(I, Op1);

  if (Value *X = dyn_castNotInst(Op0))   // ~A | A == -1
    if (X == Op1)
      return ReplaceInstUsesWith(I, 
                            ConstantIntegral::getAllOnesValue(I.getType()));

  if (Value *X = dyn_castNotInst(Op1))   // A | ~A == -1
    if (X == Op0)
      return ReplaceInstUsesWith(I, 
                            ConstantIntegral::getAllOnesValue(I.getType()));

  return Changed ? &I : 0;
}



Instruction *InstCombiner::visitXor(BinaryOperator &I) {
  bool Changed = SimplifyBinOp(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // xor X, X = 0
  if (Op0 == Op1)
    return ReplaceInstUsesWith(I, Constant::getNullValue(I.getType()));

  if (ConstantIntegral *Op1C = dyn_cast<ConstantIntegral>(Op1)) {
    // xor X, 0 == X
    if (Op1C->isNullValue())
      return ReplaceInstUsesWith(I, Op0);

    // Is this a "NOT" instruction?
    if (Op1C->isAllOnesValue()) {
      // xor (xor X, -1), -1 = not (not X) = X
      if (Value *X = dyn_castNotInst(Op0))
        return ReplaceInstUsesWith(I, X);

      // xor (setcc A, B), true = not (setcc A, B) = setncc A, B
      if (SetCondInst *SCI = dyn_cast<SetCondInst>(Op0))
        if (SCI->use_size() == 1)
          return new SetCondInst(SCI->getInverseCondition(),
                                 SCI->getOperand(0), SCI->getOperand(1));
    }
  }

  if (Value *X = dyn_castNotInst(Op0))   // ~A ^ A == -1
    if (X == Op1)
      return ReplaceInstUsesWith(I,
                                ConstantIntegral::getAllOnesValue(I.getType()));

  if (Value *X = dyn_castNotInst(Op1))   // A ^ ~A == -1
    if (X == Op0)
      return ReplaceInstUsesWith(I,
                                ConstantIntegral::getAllOnesValue(I.getType()));



  if (Instruction *Op1I = dyn_cast<Instruction>(Op1))
    if (Op1I->getOpcode() == Instruction::Or)
      if (Op1I->getOperand(0) == Op0) {              // B^(B|A) == (A|B)^B
        cast<BinaryOperator>(Op1I)->swapOperands();
        I.swapOperands();
        std::swap(Op0, Op1);
      } else if (Op1I->getOperand(1) == Op0) {       // B^(A|B) == (A|B)^B
        I.swapOperands();
        std::swap(Op0, Op1);
      }

  if (Instruction *Op0I = dyn_cast<Instruction>(Op0))
    if (Op0I->getOpcode() == Instruction::Or && Op0I->use_size() == 1) {
      if (Op0I->getOperand(0) == Op1)                // (B|A)^B == (A|B)^B
        cast<BinaryOperator>(Op0I)->swapOperands();
      if (Op0I->getOperand(1) == Op1) {              // (A|B)^B == A & ~B
        Value *NotB = BinaryOperator::createNot(Op1, Op1->getName()+".not", &I);
        WorkList.push_back(cast<Instruction>(NotB));
        return BinaryOperator::create(Instruction::And, Op0I->getOperand(0),
                                      NotB);
      }
    }

  return Changed ? &I : 0;
}

// AddOne, SubOne - Add or subtract a constant one from an integer constant...
static Constant *AddOne(ConstantInt *C) {
  Constant *Result = *C + *ConstantInt::get(C->getType(), 1);
  assert(Result && "Constant folding integer addition failed!");
  return Result;
}
static Constant *SubOne(ConstantInt *C) {
  Constant *Result = *C - *ConstantInt::get(C->getType(), 1);
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
  bool Changed = SimplifyBinOp(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);
  const Type *Ty = Op0->getType();

  // setcc X, X
  if (Op0 == Op1)
    return ReplaceInstUsesWith(I, ConstantBool::get(isTrueWhenEqual(I)));

  // setcc <global*>, 0 - Global value addresses are never null!
  if (isa<GlobalValue>(Op0) && isa<ConstantPointerNull>(Op1))
    return ReplaceInstUsesWith(I, ConstantBool::get(!isTrueWhenEqual(I)));

  // setcc's with boolean values can always be turned into bitwise operations
  if (Ty == Type::BoolTy) {
    // If this is <, >, or !=, we can change this into a simple xor instruction
    if (!isTrueWhenEqual(I))
      return BinaryOperator::create(Instruction::Xor, Op0, Op1, I.getName());

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
      return BinaryOperator::createNot(Xor, I.getName());
    }

    // Handle the setXe cases...
    assert(I.getOpcode() == Instruction::SetGE ||
           I.getOpcode() == Instruction::SetLE);

    if (I.getOpcode() == Instruction::SetGE)
      std::swap(Op0, Op1);                   // Change setge -> setle

    // Now we just have the SetLE case.
    Instruction *Not = BinaryOperator::createNot(Op0, I.getName()+"tmp");
    InsertNewInstBefore(Not, I);
    return BinaryOperator::create(Instruction::Or, Not, Op1, I.getName());
  }

  // Check to see if we are doing one of many comparisons against constant
  // integers at the end of their ranges...
  //
  if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1)) {
    // Check to see if we are comparing against the minimum or maximum value...
    if (CI->isMinValue()) {
      if (I.getOpcode() == Instruction::SetLT)       // A < MIN -> FALSE
        return ReplaceInstUsesWith(I, ConstantBool::False);
      if (I.getOpcode() == Instruction::SetGE)       // A >= MIN -> TRUE
        return ReplaceInstUsesWith(I, ConstantBool::True);
      if (I.getOpcode() == Instruction::SetLE)       // A <= MIN -> A == MIN
        return BinaryOperator::create(Instruction::SetEQ, Op0,Op1, I.getName());
      if (I.getOpcode() == Instruction::SetGT)       // A > MIN -> A != MIN
        return BinaryOperator::create(Instruction::SetNE, Op0,Op1, I.getName());

    } else if (CI->isMaxValue()) {
      if (I.getOpcode() == Instruction::SetGT)       // A > MAX -> FALSE
        return ReplaceInstUsesWith(I, ConstantBool::False);
      if (I.getOpcode() == Instruction::SetLE)       // A <= MAX -> TRUE
        return ReplaceInstUsesWith(I, ConstantBool::True);
      if (I.getOpcode() == Instruction::SetGE)       // A >= MAX -> A == MAX
        return BinaryOperator::create(Instruction::SetEQ, Op0,Op1, I.getName());
      if (I.getOpcode() == Instruction::SetLT)       // A < MAX -> A != MAX
        return BinaryOperator::create(Instruction::SetNE, Op0,Op1, I.getName());

      // Comparing against a value really close to min or max?
    } else if (isMinValuePlusOne(CI)) {
      if (I.getOpcode() == Instruction::SetLT)       // A < MIN+1 -> A == MIN
        return BinaryOperator::create(Instruction::SetEQ, Op0,
                                      SubOne(CI), I.getName());
      if (I.getOpcode() == Instruction::SetGE)       // A >= MIN-1 -> A != MIN
        return BinaryOperator::create(Instruction::SetNE, Op0,
                                      SubOne(CI), I.getName());

    } else if (isMaxValueMinusOne(CI)) {
      if (I.getOpcode() == Instruction::SetGT)       // A > MAX-1 -> A == MAX
        return BinaryOperator::create(Instruction::SetEQ, Op0,
                                      AddOne(CI), I.getName());
      if (I.getOpcode() == Instruction::SetLE)       // A <= MAX-1 -> A != MAX
        return BinaryOperator::create(Instruction::SetNE, Op0,
                                      AddOne(CI), I.getName());
    }
  }

  return Changed ? &I : 0;
}



Instruction *InstCombiner::visitShiftInst(ShiftInst &I) {
  assert(I.getOperand(1)->getType() == Type::UByteTy);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  // shl X, 0 == X and shr X, 0 == X
  // shl 0, X == 0 and shr 0, X == 0
  if (Op1 == Constant::getNullValue(Type::UByteTy) ||
      Op0 == Constant::getNullValue(Op0->getType()))
    return ReplaceInstUsesWith(I, Op0);

  // If this is a shift of a shift, see if we can fold the two together...
  if (ShiftInst *Op0SI = dyn_cast<ShiftInst>(Op0)) {
    if (isa<Constant>(Op1) && isa<Constant>(Op0SI->getOperand(1))) {
      ConstantUInt *ShiftAmt1C = cast<ConstantUInt>(Op0SI->getOperand(1));
      unsigned ShiftAmt1 = ShiftAmt1C->getValue();
      unsigned ShiftAmt2 = cast<ConstantUInt>(Op1)->getValue();

      // Check for (A << c1) << c2   and   (A >> c1) >> c2
      if (I.getOpcode() == Op0SI->getOpcode()) {
        unsigned Amt = ShiftAmt1+ShiftAmt2;   // Fold into one big shift...
        return new ShiftInst(I.getOpcode(), Op0SI->getOperand(0),
                             ConstantUInt::get(Type::UByteTy, Amt));
      }

      if (I.getType()->isUnsigned()) { // Check for (A << c1) >> c2 or visaversa
        // Calculate bitmask for what gets shifted off the edge...
        Constant *C = ConstantIntegral::getAllOnesValue(I.getType());
        if (I.getOpcode() == Instruction::Shr)
          C = *C >> *ShiftAmt1C;
        else
          C = *C << *ShiftAmt1C;
        assert(C && "Couldn't constant fold shift expression?");
          
        Instruction *Mask =
          BinaryOperator::create(Instruction::And, Op0SI->getOperand(0),
                                 C, Op0SI->getOperand(0)->getName()+".mask",&I);
        WorkList.push_back(Mask);
          
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

  // shl uint X, 32 = 0 and shr ubyte Y, 9 = 0, ... just don't eliminate shr of
  // a signed value.
  //
  if (ConstantUInt *CUI = dyn_cast<ConstantUInt>(Op1)) {
    unsigned TypeBits = Op0->getType()->getPrimitiveSize()*8;
    if (CUI->getValue() >= TypeBits &&
        (!Op0->getType()->isSigned() || I.getOpcode() == Instruction::Shl))
      return ReplaceInstUsesWith(I, Constant::getNullValue(Op0->getType()));

    // Check to see if we are shifting left by 1.  If so, turn it into an add
    // instruction.
    if (I.getOpcode() == Instruction::Shl && CUI->equalsInt(1))
      // Convert 'shl int %X, 1' to 'add int %X, %X'
      return BinaryOperator::create(Instruction::Add, Op0, Op0, I.getName());

  }

  // shr int -1, X = -1   (for any arithmetic shift rights of ~0)
  if (ConstantSInt *CSI = dyn_cast<ConstantSInt>(Op0))
    if (I.getOpcode() == Instruction::Shr && CSI->isAllOnesValue())
      return ReplaceInstUsesWith(I, CSI);
  
  return 0;
}


// isEliminableCastOfCast - Return true if it is valid to eliminate the CI
// instruction.
//
static inline bool isEliminableCastOfCast(const CastInst &CI,
                                          const CastInst *CSrc) {
  assert(CI.getOperand(0) == CSrc);
  const Type *SrcTy = CSrc->getOperand(0)->getType();
  const Type *MidTy = CSrc->getType();
  const Type *DstTy = CI.getType();

  // It is legal to eliminate the instruction if casting A->B->A if the sizes
  // are identical and the bits don't get reinterpreted (for example 
  // int->float->int would not be allowed)
  if (SrcTy == DstTy && SrcTy->isLosslesslyConvertableTo(MidTy))
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


// CastInst simplification
//
Instruction *InstCombiner::visitCastInst(CastInst &CI) {
  // If the user is casting a value to the same type, eliminate this cast
  // instruction...
  if (CI.getType() == CI.getOperand(0)->getType())
    return ReplaceInstUsesWith(CI, CI.getOperand(0));

  // If casting the result of another cast instruction, try to eliminate this
  // one!
  //
  if (CastInst *CSrc = dyn_cast<CastInst>(CI.getOperand(0))) {
    if (isEliminableCastOfCast(CI, CSrc)) {
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
      unsigned AndValue = (1U << CSrc->getType()->getPrimitiveSize()*8)-1;
      Constant *AndOp = ConstantUInt::get(CI.getType(), AndValue);
      return BinaryOperator::create(Instruction::And, CSrc->getOperand(0),
                                    AndOp);
    }
  }

  return 0;
}


// PHINode simplification
//
Instruction *InstCombiner::visitPHINode(PHINode &PN) {
  // If the PHI node only has one incoming value, eliminate the PHI node...
  if (PN.getNumIncomingValues() == 1)
    return ReplaceInstUsesWith(PN, PN.getIncomingValue(0));
  
  // Otherwise if all of the incoming values are the same for the PHI, replace
  // the PHI node with the incoming value.
  //
  Value *InVal = 0;
  for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i)
    if (PN.getIncomingValue(i) != &PN)  // Not the PHI node itself...
      if (InVal && PN.getIncomingValue(i) != InVal)
        return 0;  // Not the same, bail out.
      else
        InVal = PN.getIncomingValue(i);

  // The only case that could cause InVal to be null is if we have a PHI node
  // that only has entries for itself.  In this case, there is no entry into the
  // loop, so kill the PHI.
  //
  if (InVal == 0) InVal = Constant::getNullValue(PN.getType());

  // All of the incoming values are the same, replace the PHI node now.
  return ReplaceInstUsesWith(PN, InVal);
}


Instruction *InstCombiner::visitGetElementPtrInst(GetElementPtrInst &GEP) {
  // Is it 'getelementptr %P, uint 0'  or 'getelementptr %P'
  // If so, eliminate the noop.
  if ((GEP.getNumOperands() == 2 &&
       GEP.getOperand(1) == Constant::getNullValue(Type::LongTy)) ||
      GEP.getNumOperands() == 1)
    return ReplaceInstUsesWith(GEP, GEP.getOperand(0));

  // Combine Indices - If the source pointer to this getelementptr instruction
  // is a getelementptr instruction, combine the indices of the two
  // getelementptr instructions into a single instruction.
  //
  if (GetElementPtrInst *Src = dyn_cast<GetElementPtrInst>(GEP.getOperand(0))) {
    std::vector<Value *> Indices;
  
    // Can we combine the two pointer arithmetics offsets?
     if (Src->getNumOperands() == 2 && isa<Constant>(Src->getOperand(1)) &&
         isa<Constant>(GEP.getOperand(1))) {
      // Replace: gep (gep %P, long C1), long C2, ...
      // With:    gep %P, long (C1+C2), ...
      Value *Sum = *cast<Constant>(Src->getOperand(1)) +
                   *cast<Constant>(GEP.getOperand(1));
      assert(Sum && "Constant folding of longs failed!?");
      GEP.setOperand(0, Src->getOperand(0));
      GEP.setOperand(1, Sum);
      AddUsesToWorkList(*Src);   // Reduce use count of Src
      return &GEP;
    } else if (Src->getNumOperands() == 2 && Src->use_size() == 1) {
      // Replace: gep (gep %P, long B), long A, ...
      // With:    T = long A+B; gep %P, T, ...
      //
      Value *Sum = BinaryOperator::create(Instruction::Add, Src->getOperand(1),
                                          GEP.getOperand(1),
                                          Src->getName()+".sum", &GEP);
      GEP.setOperand(0, Src->getOperand(0));
      GEP.setOperand(1, Sum);
      WorkList.push_back(cast<Instruction>(Sum));
      return &GEP;
    } else if (*GEP.idx_begin() == Constant::getNullValue(Type::LongTy) &&
               Src->getNumOperands() != 1) { 
      // Otherwise we can do the fold if the first index of the GEP is a zero
      Indices.insert(Indices.end(), Src->idx_begin(), Src->idx_end());
      Indices.insert(Indices.end(), GEP.idx_begin()+1, GEP.idx_end());
    } else if (Src->getOperand(Src->getNumOperands()-1) == 
               Constant::getNullValue(Type::LongTy)) {
      // If the src gep ends with a constant array index, merge this get into
      // it, even if we have a non-zero array index.
      Indices.insert(Indices.end(), Src->idx_begin(), Src->idx_end()-1);
      Indices.insert(Indices.end(), GEP.idx_begin(), GEP.idx_end());
    }

    if (!Indices.empty())
      return new GetElementPtrInst(Src->getOperand(0), Indices, GEP.getName());

  } else if (GlobalValue *GV = dyn_cast<GlobalValue>(GEP.getOperand(0))) {
    // GEP of global variable.  If all of the indices for this GEP are
    // constants, we can promote this to a constexpr instead of an instruction.

    // Scan for nonconstants...
    std::vector<Constant*> Indices;
    User::op_iterator I = GEP.idx_begin(), E = GEP.idx_end();
    for (; I != E && isa<Constant>(*I); ++I)
      Indices.push_back(cast<Constant>(*I));

    if (I == E) {  // If they are all constants...
      ConstantExpr *CE =
        ConstantExpr::getGetElementPtr(ConstantPointerRef::get(GV), Indices);

      // Replace all uses of the GEP with the new constexpr...
      return ReplaceInstUsesWith(GEP, CE);
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
        New = new MallocInst(NewTy, 0, AI.getName(), &AI);
      else {
        assert(isa<AllocaInst>(AI) && "Unknown type of allocation inst!");
        New = new AllocaInst(NewTy, 0, AI.getName(), &AI);
      }
      
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
      ReplaceInstUsesWith(AI, V);
      return &AI;
    }
  return 0;
}



void InstCombiner::removeFromWorkList(Instruction *I) {
  WorkList.erase(std::remove(WorkList.begin(), WorkList.end(), I),
                 WorkList.end());
}

bool InstCombiner::runOnFunction(Function &F) {
  bool Changed = false;

  WorkList.insert(WorkList.end(), inst_begin(F), inst_end(F));

  while (!WorkList.empty()) {
    Instruction *I = WorkList.back();  // Get an instruction from the worklist
    WorkList.pop_back();

    // Check to see if we can DCE or ConstantPropagate the instruction...
    // Check to see if we can DIE the instruction...
    if (isInstructionTriviallyDead(I)) {
      // Add operands to the worklist...
      for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
        if (Instruction *Op = dyn_cast<Instruction>(I->getOperand(i)))
          WorkList.push_back(Op);

      ++NumDeadInst;
      BasicBlock::iterator BBI = I;
      if (dceInstruction(BBI)) {
        removeFromWorkList(I);
        continue;
      }
    } 

    // Instruction isn't dead, see if we can constant propagate it...
    if (Constant *C = ConstantFoldInstruction(I)) {
      // Add operands to the worklist...
      for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
        if (Instruction *Op = dyn_cast<Instruction>(I->getOperand(i)))
          WorkList.push_back(Op);
      ReplaceInstUsesWith(*I, C);

      ++NumConstProp;
      BasicBlock::iterator BBI = I;
      if (dceInstruction(BBI)) {
        removeFromWorkList(I);
        continue;
      }
    }
    
    // Now that we have an instruction, try combining it to simplify it...
    if (Instruction *Result = visit(*I)) {
      ++NumCombined;
      // Should we replace the old instruction with a new one?
      if (Result != I) {
        // Instructions can end up on the worklist more than once.  Make sure
        // we do not process an instruction that has been deleted.
        removeFromWorkList(I);
        ReplaceInstWithInst(I, Result);
      } else {
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
        AddUsesToWorkList(*Result);
      }
      Changed = true;
    }
  }

  return Changed;
}

Pass *createInstructionCombiningPass() {
  return new InstCombiner();
}
