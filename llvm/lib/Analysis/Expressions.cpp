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

static ConstPoolUInt *getUnsignedConstant(ConstantPool &CP, uint64_t V) {
  // FIXME: Lookup prexisting constant in table!

  ConstPoolUInt *CPUI = new ConstPoolUInt(Type::ULongTy, V);
  CP.insert(CPUI);
  return CPUI;
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
inline const ConstPoolInt *Add(ConstantPool &CP, const ConstPoolInt *Arg1, 
			       const ConstPoolInt *Arg2, bool DefOne = false) {
  if (DefOne == false) { // Handle degenerate cases first...
    if (Arg1 == 0) return Arg2; // Also handles case of Arg1 == Arg2 == 0
    if (Arg2 == 0) return Arg1;
  } else {               // These aren't degenerate... :(
    if (Arg1 == 0 && Arg2 == 0) return getIntegralConstant(CP, 2, Type::UIntTy);
    if (Arg1 == 0) Arg1 = getIntegralConstant(CP, 1, Arg2->getType());
    if (Arg2 == 0) Arg2 = getIntegralConstant(CP, 1, Arg2->getType());
  }

  assert(Arg1 && Arg2 && "No null arguments should exist now!");

  // FIXME: Make types compatible!

  // Actually perform the computation now!
  ConstPoolVal *Result = *Arg1 + *Arg2;
  assert(Result && Result->getType()->isIntegral() && "Couldn't perform add!");
  ConstPoolInt *ResultI = (ConstPoolInt*)Result;

  // Check to see if the result is one of the special cases that we want to
  // recognize...
  if (ResultI->equals(DefOne ? 1 : 0)) {
    // Yes it is, simply delete the constant and return null.
    delete ResultI;
    return 0;
  }

  CP.insert(ResultI);
  return ResultI;
}


ExprAnalysisResult ExprAnalysisResult::operator+(const ConstPoolInt *NewOff) {
  if (NewOff == 0) return *this;   // No change!

  ConstantPool &CP = (ConstantPool&)NewOff->getParent()->getConstantPool();
  return ExprAnalysisResult(Scale, Var, Add(CP, Offset, NewOff));
}


// Mult - Helper function to make later code simpler.  Basically it just
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
inline const ConstPoolInt *Mult(ConstantPool &CP, const ConstPoolInt *Arg1, 
				const ConstPoolInt *Arg2, bool DefOne = false) {
  if (DefOne == false) { // Handle degenerate cases first...
    if (Arg1 == 0 || Arg2 == 0) return 0;  // 0 * x == 0
  } else {               // These aren't degenerate... :(
    if (Arg1 == 0) return Arg2; // Also handles case of Arg1 == Arg2 == 0
    if (Arg2 == 0) return Arg1;
  }
  assert(Arg1 && Arg2 && "No null arguments should exist now!");

  // FIXME: Make types compatible!

  // Actually perform the computation now!
  ConstPoolVal *Result = *Arg1 * *Arg2;
  assert(Result && Result->getType()->isIntegral() && "Couldn't perform mult!");
  ConstPoolInt *ResultI = (ConstPoolInt*)Result;

  // Check to see if the result is one of the special cases that we want to
  // recognize...
  if (ResultI->equals(DefOne ? 1 : 0)) {
    // Yes it is, simply delete the constant and return null.
    delete ResultI;
    return 0;
  }

  CP.insert(ResultI);
  return ResultI;
}


// ClassifyExpression: Analyze an expression to determine the complexity of the
// expression, and which other values it depends on.  
//
// Note that this analysis cannot get into infinite loops because it treats PHI
// nodes as being an unknown linear expression.
//
ExprAnalysisResult ClassifyExpression(Value *Expr) {
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
      return ExprAnalysisResult(CPI->equals(0) ? 0 : (ConstPoolInt*)Expr);
    }
    return Expr;
  }
  
  Instruction *I = Expr->castInstructionAsserting();
  ConstantPool &CP = I->getParent()->getParent()->getConstantPool();

  switch (I->getOpcode()) {       // Handle each instruction type seperately
  case Instruction::Add: {
    ExprAnalysisResult LeftTy (ClassifyExpression(I->getOperand(0)));
    ExprAnalysisResult RightTy(ClassifyExpression(I->getOperand(1)));
    if (LeftTy.ExprType > RightTy.ExprType)
      swap(LeftTy, RightTy);   // Make left be simpler than right

    switch (LeftTy.ExprType) {
    case ExprAnalysisResult::Constant:
      return RightTy + LeftTy.Offset;
    case ExprAnalysisResult::Linear:        // RHS side must be linear or scaled
    case ExprAnalysisResult::ScaledLinear:  // RHS must be scaled
      if (LeftTy.Var != RightTy.Var)        // Are they the same variables?
	return ExprAnalysisResult(I);       //   if not, we don't know anything!

      const ConstPoolInt *NewScale  = Add(CP, LeftTy.Scale, RightTy.Scale,true);
      const ConstPoolInt *NewOffset = Add(CP, LeftTy.Offset, RightTy.Offset);
      return ExprAnalysisResult(NewScale, LeftTy.Var, NewOffset);
    }
  }  // end case Instruction::Add

  case Instruction::Shl: { 
    ExprAnalysisResult RightTy(ClassifyExpression(I->getOperand(1)));
    if (RightTy.ExprType != ExprAnalysisResult::Constant)
      break;  // TODO: Can get some info if it's (<unsigned> X + <offset>)

    ExprAnalysisResult LeftTy (ClassifyExpression(I->getOperand(0)));
    if (RightTy.Offset == 0) return LeftTy;   // shl x, 0 = x
    assert(RightTy.Offset->getType() == Type::UByteTy &&
	   "Shift amount must always be a unsigned byte!");
    uint64_t ShiftAmount = ((ConstPoolUInt*)RightTy.Offset)->getValue();
    ConstPoolUInt *Multiplier = getUnsignedConstant(CP, 1ULL << ShiftAmount);
    
    return ExprAnalysisResult(Mult(CP, LeftTy.Scale, Multiplier, true),
			      LeftTy.Var,
			      Mult(CP, LeftTy.Offset, Multiplier));
  }  // end case Instruction::Shl

    // TODO: Handle CAST, SUB, MULT (at least!)

  }  // end switch

  // Otherwise, I don't know anything about this value!
  return ExprAnalysisResult(I);
}
