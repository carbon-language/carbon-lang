//===- Expressions.cpp - Expression Analysis Utilities ----------------------=//
//
// This file defines a package of expression analysis utilties:
//
// ClassifyExpression: Analyze an expression to determine the complexity of the
//   expression, and which other variables it depends on.  
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Expressions.h"
#include "llvm/Optimizations/ConstantHandling.h"
#include "llvm/Method.h"
#include "llvm/BasicBlock.h"

using namespace opt;  // Get all the constant handling stuff
using namespace analysis;

ExprType::ExprType(Value *Val) {
  if (Val) 
    if (ConstPoolInt *CPI = dyn_cast<ConstPoolInt>(Val)) {
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

ExprType::ExprType(const ConstPoolInt *scale, Value *var, 
		   const ConstPoolInt *offset) {
  Scale = scale; Var = var; Offset = offset;
  ExprTy = Scale ? ScaledLinear : (Var ? Linear : Constant);
  if (Scale && Scale->equalsInt(0)) {  // Simplify 0*Var + const
    Scale = 0; Var = 0;
    ExprTy = Constant;
  }
}


const Type *ExprType::getExprType(const Type *Default) const {
  if (Offset) return Offset->getType();
  if (Scale) return Scale->getType();
  return Var ? Var->getType() : Default;
}



class DefVal {
  const ConstPoolInt * const Val;
  const Type * const Ty;
protected:
  inline DefVal(const ConstPoolInt *val, const Type *ty) : Val(val), Ty(ty) {}
public:
  inline const Type *getType() const { return Ty; }
  inline const ConstPoolInt *getVal() const { return Val; }
  inline operator const ConstPoolInt * () const { return Val; }
  inline const ConstPoolInt *operator->() const { return Val; }
};

struct DefZero : public DefVal {
  inline DefZero(const ConstPoolInt *val, const Type *ty) : DefVal(val, ty) {}
  inline DefZero(const ConstPoolInt *val) : DefVal(val, val->getType()) {}
};

struct DefOne : public DefVal {
  inline DefOne(const ConstPoolInt *val, const Type *ty) : DefVal(val, ty) {}
};


static ConstPoolInt *getUnsignedConstant(uint64_t V, const Type *Ty) {
  if (Ty->isPointerType()) Ty = Type::ULongTy;
  return Ty->isSigned() ? ConstPoolSInt::get(Ty, V) : ConstPoolUInt::get(Ty, V);
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
static const ConstPoolInt *Add(const ConstPoolInt *Arg1,
			       const ConstPoolInt *Arg2, bool DefOne) {
  assert(Arg1 && Arg2 && "No null arguments should exist now!");
  assert(Arg1->getType() == Arg2->getType() && "Types must be compatible!");

  // Actually perform the computation now!
  ConstPoolVal *Result = *Arg1 + *Arg2;
  assert(Result && Result->getType() == Arg1->getType() &&
	 "Couldn't perform addition!");
  ConstPoolInt *ResultI = (ConstPoolInt*)Result;

  // Check to see if the result is one of the special cases that we want to
  // recognize...
  if (ResultI->equalsInt(DefOne ? 1 : 0))
    return 0;  // Yes it is, simply return null.

  return ResultI;
}

inline const ConstPoolInt *operator+(const DefZero &L, const DefZero &R) {
  if (L == 0) return R;
  if (R == 0) return L;
  return Add(L, R, false);
}

inline const ConstPoolInt *operator+(const DefOne &L, const DefOne &R) {
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
inline const ConstPoolInt *Mul(const ConstPoolInt *Arg1, 
			       const ConstPoolInt *Arg2, bool DefOne = false) {
  assert(Arg1 && Arg2 && "No null arguments should exist now!");
  assert(Arg1->getType() == Arg2->getType() && "Types must be compatible!");

  // Actually perform the computation now!
  ConstPoolVal *Result = *Arg1 * *Arg2;
  assert(Result && Result->getType() == Arg1->getType() && 
	 "Couldn't perform mult!");
  ConstPoolInt *ResultI = (ConstPoolInt*)Result;

  // Check to see if the result is one of the special cases that we want to
  // recognize...
  if (ResultI->equalsInt(DefOne ? 1 : 0))
    return 0; // Yes it is, simply return null.

  return ResultI;
}

inline const ConstPoolInt *operator*(const DefZero &L, const DefZero &R) {
  if (L == 0 || R == 0) return 0;
  return Mul(L, R, false);
}
inline const ConstPoolInt *operator*(const DefOne &L, const DefZero &R) {
  if (R == 0) return getUnsignedConstant(0, L.getType());
  if (L == 0) return R->equalsInt(1) ? 0 : R.getVal();
  return Mul(L, R, false);
}
inline const ConstPoolInt *operator*(const DefZero &L, const DefOne &R) {
  return R*L;
}

// handleAddition - Add two expressions together, creating a new expression that
// represents the composite of the two...
//
static ExprType handleAddition(ExprType Left, ExprType Right, Value *V) {
  const Type *Ty = V->getType();
  if (Left.ExprTy > Right.ExprTy)
    swap(Left, Right);   // Make left be simpler than right

  switch (Left.ExprTy) {
  case ExprType::Constant:
    return ExprType(Right.Scale, Right.Var,
		    DefZero(Right.Offset, Ty) + DefZero(Left.Offset, Ty));
  case ExprType::Linear:              // RHS side must be linear or scaled
  case ExprType::ScaledLinear:        // RHS must be scaled
    if (Left.Var != Right.Var)        // Are they the same variables?
      return ExprType(V);             //   if not, we don't know anything!

    return ExprType(DefOne(Left.Scale  , Ty) + DefOne(Right.Scale , Ty),
		    Left.Var,
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
  const Type *ETy = E.getExprType(Ty);
  ConstPoolInt *Zero   = getUnsignedConstant(0, ETy);
  ConstPoolInt *One    = getUnsignedConstant(1, ETy);
  ConstPoolInt *NegOne = (ConstPoolInt*)(*Zero - *One);
  if (NegOne == 0) return V;  // Couldn't subtract values...

  return ExprType(DefOne (E.Scale , Ty) * NegOne, E.Var,
		  DefZero(E.Offset, Ty) * NegOne);
}


// ClassifyExpression: Analyze an expression to determine the complexity of the
// expression, and which other values it depends on.  
//
// Note that this analysis cannot get into infinite loops because it treats PHI
// nodes as being an unknown linear expression.
//
ExprType analysis::ClassifyExpression(Value *Expr) {
  assert(Expr != 0 && "Can't classify a null expression!");
  switch (Expr->getValueType()) {
  case Value::InstructionVal: break;    // Instruction... hmmm... investigate.
  case Value::TypeVal:   case Value::BasicBlockVal:
  case Value::MethodVal: case Value::ModuleVal: default:
    assert(0 && "Unexpected expression type to classify!");
  case Value::GlobalVal:                // Global Variable & Method argument:
  case Value::MethodArgumentVal:        // nothing known, return variable itself
    return Expr;
  case Value::ConstantVal:              // Constant value, just return constant
    ConstPoolVal *CPV = cast<ConstPoolVal>(Expr);
    if (CPV->getType()->isIntegral()) { // It's an integral constant!
      ConstPoolInt *CPI = (ConstPoolInt*)Expr;
      return ExprType(CPI->equalsInt(0) ? 0 : CPI);
    }
    return Expr;
  }
  
  Instruction *I = cast<Instruction>(Expr);
  const Type *Ty = I->getType();

  switch (I->getOpcode()) {       // Handle each instruction type seperately
  case Instruction::Add: {
    ExprType Left (ClassifyExpression(I->getOperand(0)));
    ExprType Right(ClassifyExpression(I->getOperand(1)));
    return handleAddition(Left, Right, I);
  }  // end case Instruction::Add

  case Instruction::Sub: {
    ExprType Left (ClassifyExpression(I->getOperand(0)));
    ExprType Right(ClassifyExpression(I->getOperand(1)));
    return handleAddition(Left, negate(Right, I), I);
  }  // end case Instruction::Sub

  case Instruction::Shl: { 
    ExprType Right(ClassifyExpression(I->getOperand(1)));
    if (Right.ExprTy != ExprType::Constant) break;
    ExprType Left(ClassifyExpression(I->getOperand(0)));
    if (Right.Offset == 0) return Left;   // shl x, 0 = x
    assert(Right.Offset->getType() == Type::UByteTy &&
	   "Shift amount must always be a unsigned byte!");
    uint64_t ShiftAmount = ((ConstPoolUInt*)Right.Offset)->getValue();
    ConstPoolInt *Multiplier = getUnsignedConstant(1ULL << ShiftAmount, Ty);
    
    return ExprType(DefOne(Left.Scale, Ty) * Multiplier, Left.Var,
		    DefZero(Left.Offset, Ty) * Multiplier);
  }  // end case Instruction::Shl

  case Instruction::Mul: {
    ExprType Left (ClassifyExpression(I->getOperand(0)));
    ExprType Right(ClassifyExpression(I->getOperand(1)));
    if (Left.ExprTy > Right.ExprTy)
      swap(Left, Right);   // Make left be simpler than right

    if (Left.ExprTy != ExprType::Constant)  // RHS must be > constant
      return I;         // Quadratic eqn! :(

    const ConstPoolInt *Offs = Left.Offset;
    if (Offs == 0) return ExprType();
    return ExprType( DefOne(Right.Scale , Ty) * Offs, Right.Var,
		    DefZero(Right.Offset, Ty) * Offs);
  } // end case Instruction::Mul

  case Instruction::Cast: {
    ExprType Src(ClassifyExpression(I->getOperand(0)));
    if (Src.ExprTy != ExprType::Constant)
      return I;
    const ConstPoolInt *Offs = Src.Offset;
    if (Offs == 0) return ExprType();

    const Type *DestTy = I->getType();
    if (DestTy->isPointerType())
      DestTy = Type::ULongTy;  // Pointer types are represented as ulong

    assert(DestTy->isIntegral() && "Can only handle integral types!");

    const ConstPoolVal *CPV =ConstRules::get(*Offs)->castTo(Offs, DestTy);
    if (!CPV) return I;
    assert(CPV->getType()->isIntegral() && "Must have an integral type!");
    return (ConstPoolInt*)CPV;
  } // end case Instruction::Cast
    // TODO: Handle SUB, SHR?

  }  // end switch

  // Otherwise, I don't know anything about this value!
  return I;
}
