//===- Expressions.cpp - Expression Analysis Utilities --------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines a package of expression analysis utilties:
//
// ClassifyExpression: Analyze an expression to determine the complexity of the
//   expression, and which other variables it depends on.  
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Expressions.h"
#include "llvm/ConstantHandling.h"
#include "llvm/Function.h"
using namespace llvm;

ExprType::ExprType(Value *Val) {
  if (Val) 
    if (ConstantInt *CPI = dyn_cast<ConstantInt>(Val)) {
      Offset = CPI;
      Var = 0;
      ExprTy = Constant;
      Scale = 0;
      return;
    }

  Var = Val; Offset = 0;
  ExprTy = Var ? Linear : Constant;
  Scale = 0;
}

ExprType::ExprType(const ConstantInt *scale, Value *var, 
		   const ConstantInt *offset) {
  Scale = var ? scale : 0; Var = var; Offset = offset;
  ExprTy = Scale ? ScaledLinear : (Var ? Linear : Constant);
  if (Scale && Scale->isNullValue()) {  // Simplify 0*Var + const
    Scale = 0; Var = 0;
    ExprTy = Constant;
  }
}


const Type *ExprType::getExprType(const Type *Default) const {
  if (Offset) return Offset->getType();
  if (Scale) return Scale->getType();
  return Var ? Var->getType() : Default;
}


namespace {
  class DefVal {
    const ConstantInt * const Val;
    const Type * const Ty;
  protected:
    inline DefVal(const ConstantInt *val, const Type *ty) : Val(val), Ty(ty) {}
  public:
    inline const Type *getType() const { return Ty; }
    inline const ConstantInt *getVal() const { return Val; }
    inline operator const ConstantInt * () const { return Val; }
    inline const ConstantInt *operator->() const { return Val; }
  };
  
  struct DefZero : public DefVal {
    inline DefZero(const ConstantInt *val, const Type *ty) : DefVal(val, ty) {}
    inline DefZero(const ConstantInt *val) : DefVal(val, val->getType()) {}
  };
  
  struct DefOne : public DefVal {
    inline DefOne(const ConstantInt *val, const Type *ty) : DefVal(val, ty) {}
  };
}


// getUnsignedConstant - Return a constant value of the specified type.  If the
// constant value is not valid for the specified type, return null.  This cannot
// happen for values in the range of 0 to 127.
//
static ConstantInt *getUnsignedConstant(uint64_t V, const Type *Ty) {
  if (isa<PointerType>(Ty)) Ty = Type::ULongTy;
  if (Ty->isSigned()) {
    // If this value is not a valid unsigned value for this type, return null!
    if (V > 127 && ((int64_t)V < 0 ||
                    !ConstantSInt::isValueValidForType(Ty, (int64_t)V)))
      return 0;
    return ConstantSInt::get(Ty, V);
  } else {
    // If this value is not a valid unsigned value for this type, return null!
    if (V > 255 && !ConstantUInt::isValueValidForType(Ty, V))
      return 0;
    return ConstantUInt::get(Ty, V);
  }
}

// Add - Helper function to make later code simpler.  Basically it just adds
// the two constants together, inserts the result into the constant pool, and
// returns it.  Of course life is not simple, and this is no exception.  Factors
// that complicate matters:
//   1. Either argument may be null.  If this is the case, the null argument is
//      treated as either 0 (if DefOne = false) or 1 (if DefOne = true)
//   2. Types get in the way.  We want to do arithmetic operations without
//      regard for the underlying types.  It is assumed that the constants are
//      integral constants.  The new value takes the type of the left argument.
//   3. If DefOne is true, a null return value indicates a value of 1, if DefOne
//      is false, a null return value indicates a value of 0.
//
static const ConstantInt *Add(const ConstantInt *Arg1,
                              const ConstantInt *Arg2, bool DefOne) {
  assert(Arg1 && Arg2 && "No null arguments should exist now!");
  assert(Arg1->getType() == Arg2->getType() && "Types must be compatible!");

  // Actually perform the computation now!
  Constant *Result = *Arg1 + *Arg2;
  assert(Result && Result->getType() == Arg1->getType() &&
	 "Couldn't perform addition!");
  ConstantInt *ResultI = cast<ConstantInt>(Result);

  // Check to see if the result is one of the special cases that we want to
  // recognize...
  if (ResultI->equalsInt(DefOne ? 1 : 0))
    return 0;  // Yes it is, simply return null.

  return ResultI;
}

static inline const ConstantInt *operator+(const DefZero &L, const DefZero &R) {
  if (L == 0) return R;
  if (R == 0) return L;
  return Add(L, R, false);
}

static inline const ConstantInt *operator+(const DefOne &L, const DefOne &R) {
  if (L == 0) {
    if (R == 0)
      return getUnsignedConstant(2, L.getType());
    else
      return Add(getUnsignedConstant(1, L.getType()), R, true);
  } else if (R == 0) {
    return Add(L, getUnsignedConstant(1, L.getType()), true);
  }
  return Add(L, R, true);
}


// Mul - Helper function to make later code simpler.  Basically it just
// multiplies the two constants together, inserts the result into the constant
// pool, and returns it.  Of course life is not simple, and this is no
// exception.  Factors that complicate matters:
//   1. Either argument may be null.  If this is the case, the null argument is
//      treated as either 0 (if DefOne = false) or 1 (if DefOne = true)
//   2. Types get in the way.  We want to do arithmetic operations without
//      regard for the underlying types.  It is assumed that the constants are
//      integral constants.
//   3. If DefOne is true, a null return value indicates a value of 1, if DefOne
//      is false, a null return value indicates a value of 0.
//
static inline const ConstantInt *Mul(const ConstantInt *Arg1, 
                                     const ConstantInt *Arg2, bool DefOne) {
  assert(Arg1 && Arg2 && "No null arguments should exist now!");
  assert(Arg1->getType() == Arg2->getType() && "Types must be compatible!");

  // Actually perform the computation now!
  Constant *Result = *Arg1 * *Arg2;
  assert(Result && Result->getType() == Arg1->getType() && 
	 "Couldn't perform multiplication!");
  ConstantInt *ResultI = cast<ConstantInt>(Result);

  // Check to see if the result is one of the special cases that we want to
  // recognize...
  if (ResultI->equalsInt(DefOne ? 1 : 0))
    return 0; // Yes it is, simply return null.

  return ResultI;
}

namespace {
  inline const ConstantInt *operator*(const DefZero &L, const DefZero &R) {
    if (L == 0 || R == 0) return 0;
    return Mul(L, R, false);
  }
  inline const ConstantInt *operator*(const DefOne &L, const DefZero &R) {
    if (R == 0) return getUnsignedConstant(0, L.getType());
    if (L == 0) return R->equalsInt(1) ? 0 : R.getVal();
    return Mul(L, R, true);
  }
  inline const ConstantInt *operator*(const DefZero &L, const DefOne &R) {
    if (L == 0 || R == 0) return L.getVal();
    return Mul(R, L, false);
  }
}

// handleAddition - Add two expressions together, creating a new expression that
// represents the composite of the two...
//
static ExprType handleAddition(ExprType Left, ExprType Right, Value *V) {
  const Type *Ty = V->getType();
  if (Left.ExprTy > Right.ExprTy)
    std::swap(Left, Right);   // Make left be simpler than right

  switch (Left.ExprTy) {
  case ExprType::Constant:
        return ExprType(Right.Scale, Right.Var,
			DefZero(Right.Offset, Ty) + DefZero(Left.Offset, Ty));
  case ExprType::Linear:              // RHS side must be linear or scaled
  case ExprType::ScaledLinear:        // RHS must be scaled
    if (Left.Var != Right.Var)        // Are they the same variables?
      return V;                       //   if not, we don't know anything!

    return ExprType(DefOne(Left.Scale  , Ty) + DefOne(Right.Scale , Ty),
		    Right.Var,
		    DefZero(Left.Offset, Ty) + DefZero(Right.Offset, Ty));
  default:
    assert(0 && "Dont' know how to handle this case!");
    return ExprType();
  }
}

// negate - Negate the value of the specified expression...
//
static inline ExprType negate(const ExprType &E, Value *V) {
  const Type *Ty = V->getType();
  ConstantInt *Zero   = getUnsignedConstant(0, Ty);
  ConstantInt *One    = getUnsignedConstant(1, Ty);
  ConstantInt *NegOne = cast<ConstantInt>(*Zero - *One);
  if (NegOne == 0) return V;  // Couldn't subtract values...

  return ExprType(DefOne (E.Scale , Ty) * NegOne, E.Var,
		  DefZero(E.Offset, Ty) * NegOne);
}


// ClassifyExpr: Analyze an expression to determine the complexity of the
// expression, and which other values it depends on.
//
// Note that this analysis cannot get into infinite loops because it treats PHI
// nodes as being an unknown linear expression.
//
ExprType llvm::ClassifyExpr(Value *Expr) {
  assert(Expr != 0 && "Can't classify a null expression!");
  if (Expr->getType() == Type::FloatTy || Expr->getType() == Type::DoubleTy)
    return Expr;   // FIXME: Can't handle FP expressions

  switch (Expr->getValueType()) {
  case Value::InstructionVal: break;    // Instruction... hmmm... investigate.
  case Value::TypeVal:   case Value::BasicBlockVal:
  case Value::FunctionVal: default:
    //assert(0 && "Unexpected expression type to classify!");
    std::cerr << "Bizarre thing to expr classify: " << Expr << "\n";
    return Expr;
  case Value::GlobalVariableVal:        // Global Variable & Function argument:
  case Value::ArgumentVal:              // nothing known, return variable itself
    return Expr;
  case Value::ConstantVal:              // Constant value, just return constant
    if (ConstantInt *CPI = dyn_cast<ConstantInt>(cast<Constant>(Expr)))
      // It's an integral constant!
      return ExprType(CPI->isNullValue() ? 0 : CPI);
    return Expr;
  }
  
  Instruction *I = cast<Instruction>(Expr);
  const Type *Ty = I->getType();

  switch (I->getOpcode()) {       // Handle each instruction type separately
  case Instruction::Add: {
    ExprType Left (ClassifyExpr(I->getOperand(0)));
    ExprType Right(ClassifyExpr(I->getOperand(1)));
    return handleAddition(Left, Right, I);
  }  // end case Instruction::Add

  case Instruction::Sub: {
    ExprType Left (ClassifyExpr(I->getOperand(0)));
    ExprType Right(ClassifyExpr(I->getOperand(1)));
    ExprType RightNeg = negate(Right, I);
    if (RightNeg.Var == I && !RightNeg.Offset && !RightNeg.Scale)
      return I;   // Could not negate value...
    return handleAddition(Left, RightNeg, I);
  }  // end case Instruction::Sub

  case Instruction::Shl: { 
    ExprType Right(ClassifyExpr(I->getOperand(1)));
    if (Right.ExprTy != ExprType::Constant) break;
    ExprType Left(ClassifyExpr(I->getOperand(0)));
    if (Right.Offset == 0) return Left;   // shl x, 0 = x
    assert(Right.Offset->getType() == Type::UByteTy &&
	   "Shift amount must always be a unsigned byte!");
    uint64_t ShiftAmount = cast<ConstantUInt>(Right.Offset)->getValue();
    ConstantInt *Multiplier = getUnsignedConstant(1ULL << ShiftAmount, Ty);

    // We don't know how to classify it if they are shifting by more than what
    // is reasonable.  In most cases, the result will be zero, but there is one
    // class of cases where it is not, so we cannot optimize without checking
    // for it.  The case is when you are shifting a signed value by 1 less than
    // the number of bits in the value.  For example:
    //    %X = shl sbyte %Y, ubyte 7
    // will try to form an sbyte multiplier of 128, which will give a null
    // multiplier, even though the result is not 0.  Until we can check for this
    // case, be conservative.  TODO.
    //
    if (Multiplier == 0)
      return Expr;

    return ExprType(DefOne(Left.Scale, Ty) * Multiplier, Left.Var,
		    DefZero(Left.Offset, Ty) * Multiplier);
  }  // end case Instruction::Shl

  case Instruction::Mul: {
    ExprType Left (ClassifyExpr(I->getOperand(0)));
    ExprType Right(ClassifyExpr(I->getOperand(1)));
    if (Left.ExprTy > Right.ExprTy)
      std::swap(Left, Right);   // Make left be simpler than right

    if (Left.ExprTy != ExprType::Constant)  // RHS must be > constant
      return I;         // Quadratic eqn! :(

    const ConstantInt *Offs = Left.Offset;
    if (Offs == 0) return ExprType();
    return ExprType( DefOne(Right.Scale , Ty) * Offs, Right.Var,
		    DefZero(Right.Offset, Ty) * Offs);
  } // end case Instruction::Mul

  case Instruction::Cast: {
    ExprType Src(ClassifyExpr(I->getOperand(0)));
    const Type *DestTy = I->getType();
    if (isa<PointerType>(DestTy))
      DestTy = Type::ULongTy;  // Pointer types are represented as ulong

    const Type *SrcValTy = Src.getExprType(0);
    if (!SrcValTy) return I;
    if (!SrcValTy->isLosslesslyConvertibleTo(DestTy)) {
      if (Src.ExprTy != ExprType::Constant)
        return I;  // Converting cast, and not a constant value...
    }

    const ConstantInt *Offset = Src.Offset;
    const ConstantInt *Scale  = Src.Scale;
    if (Offset) {
      const Constant *CPV = ConstantFoldCastInstruction(Offset, DestTy);
      if (!CPV) return I;
      Offset = cast<ConstantInt>(CPV);
    }
    if (Scale) {
      const Constant *CPV = ConstantFoldCastInstruction(Scale, DestTy);
      if (!CPV) return I;
      Scale = cast<ConstantInt>(CPV);
    }
    return ExprType(Scale, Src.Var, Offset);
  } // end case Instruction::Cast
    // TODO: Handle SUB, SHR?

  }  // end switch

  // Otherwise, I don't know anything about this value!
  return I;
}
