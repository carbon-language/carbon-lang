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
#include "llvm/ConstantPool.h"
#include "llvm/Method.h"
#include "llvm/BasicBlock.h"

using namespace opt;  // Get all the constant handling stuff
using namespace analysis;

class DefVal {
  const ConstPoolInt * const Val;
  ConstantPool &CP;
  const Type * const Ty;
protected:
  inline DefVal(const ConstPoolInt *val, ConstantPool &cp, const Type *ty)
    : Val(val), CP(cp), Ty(ty) {}
public:
  inline const Type *getType() const { return Ty; }
  inline ConstantPool &getCP() const { return CP; }
  inline const ConstPoolInt *getVal() const { return Val; }
  inline operator const ConstPoolInt * () const { return Val; }
  inline const ConstPoolInt *operator->() const { return Val; }
};

struct DefZero : public DefVal {
  inline DefZero(const ConstPoolInt *val, ConstantPool &cp, const Type *ty)
    : DefVal(val, cp, ty) {}
  inline DefZero(const ConstPoolInt *val)
    : DefVal(val, (ConstantPool&)val->getParent()->getConstantPool(),
	     val->getType()) {}
};

struct DefOne : public DefVal {
  inline DefOne(const ConstPoolInt *val, ConstantPool &cp, const Type *ty)
    : DefVal(val, cp, ty) {}
};


// getIntegralConstant - Wrapper around the ConstPoolInt member of the same
// name.  This method first checks to see if the desired constant is already in
// the constant pool.  If it is, it is quickly recycled, otherwise a new one
// is allocated and added to the constant pool.
//
static ConstPoolInt *getIntegralConstant(ConstantPool &CP, unsigned char V,
					 const Type *Ty) {
  // FIXME: Lookup prexisting constant in table!

  ConstPoolInt *CPI = ConstPoolInt::get(Ty, V);
  CP.insert(CPI);
  return CPI;
}

static ConstPoolInt *getUnsignedConstant(ConstantPool &CP, uint64_t V,
					 const Type *Ty) {
  // FIXME: Lookup prexisting constant in table!
  if (Ty->isPointerType()) Ty = Type::ULongTy;

  ConstPoolInt *CPI;
  CPI = Ty->isSigned() ? new ConstPoolSInt(Ty, V) : new ConstPoolUInt(Ty, V);
  CP.insert(CPI);
  return CPI;
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
static const ConstPoolInt *Add(ConstantPool &CP, const ConstPoolInt *Arg1,
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
  if (ResultI->equalsInt(DefOne ? 1 : 0)) {
    // Yes it is, simply delete the constant and return null.
    delete ResultI;
    return 0;
  }

  CP.insert(ResultI);
  return ResultI;
}

inline const ConstPoolInt *operator+(const DefZero &L, const DefZero &R) {
  if (L == 0) return R;
  if (R == 0) return L;
  return Add(L.getCP(), L, R, false);
}

inline const ConstPoolInt *operator+(const DefOne &L, const DefOne &R) {
  if (L == 0) {
    if (R == 0)
      return getIntegralConstant(L.getCP(), 2, L.getType());
    else
      return Add(L.getCP(), getIntegralConstant(L.getCP(), 1, L.getType()),
		 R, true);
  } else if (R == 0) {
    return Add(L.getCP(), L,
	       getIntegralConstant(L.getCP(), 1, L.getType()), true);
  }
  return Add(L.getCP(), L, R, true);
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
inline const ConstPoolInt *Mul(ConstantPool &CP, const ConstPoolInt *Arg1, 
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
  if (ResultI->equalsInt(DefOne ? 1 : 0)) {
    // Yes it is, simply delete the constant and return null.
    delete ResultI;
    return 0;
  }

  CP.insert(ResultI);
  return ResultI;
}

inline const ConstPoolInt *operator*(const DefZero &L, const DefZero &R) {
  if (L == 0 || R == 0) return 0;
  return Mul(L.getCP(), L, R, false);
}
inline const ConstPoolInt *operator*(const DefOne &L, const DefZero &R) {
  if (R == 0) return getIntegralConstant(L.getCP(), 0, L.getType());
  if (L == 0) return R->equalsInt(1) ? 0 : R.getVal();
  return Mul(L.getCP(), L, R, false);
}
inline const ConstPoolInt *operator*(const DefZero &L, const DefOne &R) {
  return R*L;
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
  case Value::MethodVal: case Value::ModuleVal:
    assert(0 && "Unexpected expression type to classify!");
  case Value::MethodArgumentVal:        // Method arg: nothing known, return var
    return Expr;
  case Value::ConstantVal:              // Constant value, just return constant
    ConstPoolVal *CPV = Expr->castConstantAsserting();
    if (CPV->getType()->isIntegral()) { // It's an integral constant!
      ConstPoolInt *CPI = (ConstPoolInt*)Expr;
      return ExprType(CPI->equalsInt(0) ? 0 : (ConstPoolInt*)Expr);
    }
    return Expr;
  }
  
  Instruction *I = Expr->castInstructionAsserting();
  ConstantPool &CP = I->getParent()->getParent()->getConstantPool();
  const Type *Ty = I->getType();

  switch (I->getOpcode()) {       // Handle each instruction type seperately
  case Instruction::Add: {
    ExprType Left (ClassifyExpression(I->getOperand(0)));
    ExprType Right(ClassifyExpression(I->getOperand(1)));
    if (Left.ExprTy > Right.ExprTy)
      swap(Left, Right);   // Make left be simpler than right

    switch (Left.ExprTy) {
    case ExprType::Constant:
      return ExprType(Right.Scale, Right.Var,
		      DefZero(Right.Offset,CP,Ty) + DefZero(Left.Offset, CP,Ty));
    case ExprType::Linear:        // RHS side must be linear or scaled
    case ExprType::ScaledLinear:  // RHS must be scaled
      if (Left.Var != Right.Var)        // Are they the same variables?
	return ExprType(I);       //   if not, we don't know anything!

      return ExprType(DefOne(Left.Scale  ,CP,Ty) + DefOne(Right.Scale  , CP,Ty),
		      Left.Var,
	              DefZero(Left.Offset,CP,Ty) + DefZero(Right.Offset, CP,Ty));
    }
  }  // end case Instruction::Add

  case Instruction::Shl: { 
    ExprType Right(ClassifyExpression(I->getOperand(1)));
    if (Right.ExprTy != ExprType::Constant) break;
    ExprType Left(ClassifyExpression(I->getOperand(0)));
    if (Right.Offset == 0) return Left;   // shl x, 0 = x
    assert(Right.Offset->getType() == Type::UByteTy &&
	   "Shift amount must always be a unsigned byte!");
    uint64_t ShiftAmount = ((ConstPoolUInt*)Right.Offset)->getValue();
    ConstPoolInt *Multiplier = getUnsignedConstant(CP, 1ULL << ShiftAmount, Ty);
    
    return ExprType(DefOne(Left.Scale, CP, Ty) * Multiplier,
		    Left.Var,
		    DefZero(Left.Offset, CP, Ty) * Multiplier);
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
    return ExprType(DefOne(Right.Scale, CP, Ty) * Offs,
		    Right.Var,
		    DefZero(Right.Offset, CP, Ty) * Offs);
  } // end case Instruction::Mul

  case Instruction::Cast: {
    ExprType Src(ClassifyExpression(I->getOperand(0)));
    if (Src.ExprTy != ExprType::Constant)
      return I;
    const ConstPoolInt *Offs = Src.Offset;
    if (Offs == 0) return ExprType();

    if (I->getType()->isPointerType())
      return Offs;  // Pointer types do not lose precision

    assert(I->getType()->isIntegral() && "Can only handle integral types!");

    const ConstPoolVal *CPV = ConstRules::get(*Offs)->castTo(Offs, I->getType());
    if (!CPV) return I;
    assert(CPV->getType()->isIntegral() && "Must have an integral type!");
    return (ConstPoolInt*)CPV;
  } // end case Instruction::Cast
    // TODO: Handle SUB, SHR?

  }  // end switch

  // Otherwise, I don't know anything about this value!
  return I;
}
