//===--- CGComplexExpr.cpp - Emit LLVM Code for Complex Exprs -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Expr nodes with complex types as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/AST.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Support/Compiler.h"
using namespace clang;
using namespace CodeGen;

//===----------------------------------------------------------------------===//
//                        Complex Expression Emitter
//===----------------------------------------------------------------------===//

typedef CodeGenFunction::ComplexPairTy ComplexPairTy;

namespace  {
class VISIBILITY_HIDDEN ComplexExprEmitter
  : public StmtVisitor<ComplexExprEmitter, ComplexPairTy> {
  CodeGenFunction &CGF;
  llvm::LLVMBuilder &Builder;
public:
  ComplexExprEmitter(CodeGenFunction &cgf) : CGF(cgf), Builder(CGF.Builder) {
  }

  
  //===--------------------------------------------------------------------===//
  //                               Utilities
  //===--------------------------------------------------------------------===//

  /// EmitLoadOfLValue - Given an expression with complex type that represents a
  /// value l-value, this method emits the address of the l-value, then loads
  /// and returns the result.
  ComplexPairTy EmitLoadOfLValue(const Expr *E) {
    LValue LV = CGF.EmitLValue(E);
    // FIXME: Volatile
    return EmitLoadOfComplex(LV.getAddress(), false);
  }
  
  /// EmitLoadOfComplex - Given a pointer to a complex value, emit code to load
  /// the real and imaginary pieces.
  ComplexPairTy EmitLoadOfComplex(llvm::Value *SrcPtr, bool isVolatile);
  
  /// EmitStoreOfComplex - Store the specified real/imag parts into the
  /// specified value pointer.
  void EmitStoreOfComplex(ComplexPairTy Val, llvm::Value *ResPtr, bool isVol);
  
  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  ComplexPairTy VisitStmt(Stmt *S) {
    S->dump();
    assert(0 && "Stmt can't have complex result type!");
    return ComplexPairTy();
  }
  ComplexPairTy VisitExpr(Expr *S);
  ComplexPairTy VisitParenExpr(ParenExpr *PE) { return Visit(PE->getSubExpr());}

  // l-values.
  ComplexPairTy VisitDeclRefExpr(Expr *E) { return EmitLoadOfLValue(E); }
  ComplexPairTy VisitArraySubscriptExpr(Expr *E) { return EmitLoadOfLValue(E); }

  // Operators.
  //  case Expr::UnaryOperatorClass:
  //  case Expr::ImplicitCastExprClass:
  //  case Expr::CastExprClass: 
  //  case Expr::CallExprClass:
  ComplexPairTy VisitBinMul        (const BinaryOperator *E);
  ComplexPairTy VisitBinAdd        (const BinaryOperator *E);
  // FIXME: div/rem
  // GCC rejects and/or/xor for integer complex.
  // Logical and/or always return int, never complex.

  // No comparisons produce a complex result.
  ComplexPairTy VisitBinAssign     (const BinaryOperator *E);

  ComplexPairTy VisitBinComma      (const BinaryOperator *E);

  
  ComplexPairTy VisitConditionalOperator(const ConditionalOperator *CO);
  //  case Expr::ChooseExprClass:
};
}  // end anonymous namespace.

//===----------------------------------------------------------------------===//
//                                Utilities
//===----------------------------------------------------------------------===//

/// EmitLoadOfComplex - Given an RValue reference for a complex, emit code to
/// load the real and imaginary pieces, returning them as Real/Imag.
ComplexPairTy ComplexExprEmitter::EmitLoadOfComplex(llvm::Value *SrcPtr,
                                                    bool isVolatile) {
  llvm::Constant *Zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0);
  llvm::Constant *One  = llvm::ConstantInt::get(llvm::Type::Int32Ty, 1);
  // FIXME: It would be nice to make this "Ptr->getName()+realp"
  llvm::Value *RealPtr = Builder.CreateGEP(SrcPtr, Zero, Zero, "realp");
  llvm::Value *ImagPtr = Builder.CreateGEP(SrcPtr, Zero, One, "imagp");
  
  // FIXME: It would be nice to make this "Ptr->getName()+real"
  llvm::Value *Real = Builder.CreateLoad(RealPtr, isVolatile, "real");
  llvm::Value *Imag = Builder.CreateLoad(ImagPtr, isVolatile, "imag");
  return ComplexPairTy(Real, Imag);
}

/// EmitStoreOfComplex - Store the specified real/imag parts into the
/// specified value pointer.
void ComplexExprEmitter::EmitStoreOfComplex(ComplexPairTy Val, llvm::Value *Ptr,
                                            bool isVolatile) {
  llvm::Constant *Zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0);
  llvm::Constant *One  = llvm::ConstantInt::get(llvm::Type::Int32Ty, 1);
  llvm::Value *RealPtr = Builder.CreateGEP(Ptr, Zero, Zero, "real");
  llvm::Value *ImagPtr = Builder.CreateGEP(Ptr, Zero, One, "imag");
  
  Builder.CreateStore(Val.first, RealPtr, isVolatile);
  Builder.CreateStore(Val.second, ImagPtr, isVolatile);
}



//===----------------------------------------------------------------------===//
//                            Visitor Methods
//===----------------------------------------------------------------------===//

ComplexPairTy ComplexExprEmitter::VisitExpr(Expr *E) {
  fprintf(stderr, "Unimplemented complex expr!\n");
  E->dump();
  const llvm::Type *EltTy = 
    CGF.ConvertType(E->getType()->getAsComplexType()->getElementType());
  llvm::Value *U = llvm::UndefValue::get(EltTy);
  return ComplexPairTy(U, U);
}

ComplexPairTy ComplexExprEmitter::VisitBinAdd(const BinaryOperator *E) {
  ComplexPairTy LHS = Visit(E->getLHS());
  ComplexPairTy RHS = Visit(E->getRHS());
  
  llvm::Value *ResR = Builder.CreateAdd(LHS.first,  RHS.first,  "add.r");
  llvm::Value *ResI = Builder.CreateAdd(LHS.second, RHS.second, "add.i");

  return ComplexPairTy(ResR, ResI);
}

ComplexPairTy ComplexExprEmitter::VisitBinMul(const BinaryOperator *E) {
  ComplexPairTy LHS = Visit(E->getLHS());
  ComplexPairTy RHS = Visit(E->getRHS());
  
  llvm::Value *ResRl = Builder.CreateMul(LHS.first, RHS.first, "mul.rl");
  llvm::Value *ResRr = Builder.CreateMul(LHS.second, RHS.second, "mul.rr");
  llvm::Value *ResR  = Builder.CreateSub(ResRl, ResRr, "mul.r");
  
  llvm::Value *ResIl = Builder.CreateMul(LHS.second, RHS.first, "mul.il");
  llvm::Value *ResIr = Builder.CreateMul(LHS.first, RHS.second, "mul.ir");
  llvm::Value *ResI  = Builder.CreateAdd(ResIl, ResIr, "mul.i");

  return ComplexPairTy(ResR, ResI);
}

ComplexPairTy ComplexExprEmitter::VisitBinAssign(const BinaryOperator *E) {
  assert(E->getLHS()->getType().getCanonicalType() ==
         E->getRHS()->getType().getCanonicalType() && "Invalid assignment");
  // Emit the RHS.
  ComplexPairTy Val = Visit(E->getRHS());

  // Compute the address to store into.
  LValue LHS = CGF.EmitLValue(E->getLHS());
  
  // Store into it.
  // FIXME: Volatility!
  EmitStoreOfComplex(Val, LHS.getAddress(), false);
  return Val;
}

ComplexPairTy ComplexExprEmitter::VisitBinComma(const BinaryOperator *E) {
  CGF.EmitStmt(E->getLHS());
  return Visit(E->getRHS());
}

ComplexPairTy ComplexExprEmitter::
VisitConditionalOperator(const ConditionalOperator *E) {
  llvm::BasicBlock *LHSBlock = new llvm::BasicBlock("cond.?");
  llvm::BasicBlock *RHSBlock = new llvm::BasicBlock("cond.:");
  llvm::BasicBlock *ContBlock = new llvm::BasicBlock("cond.cont");
  
  llvm::Value *Cond = CGF.EvaluateExprAsBool(E->getCond());
  Builder.CreateCondBr(Cond, LHSBlock, RHSBlock);
  
  CGF.EmitBlock(LHSBlock);
  
  // Handle the GNU extension for missing LHS.
  assert(E->getLHS() && "Must have LHS for complex value");

  ComplexPairTy LHS = Visit(E->getLHS());
  Builder.CreateBr(ContBlock);
  LHSBlock = Builder.GetInsertBlock();
  
  CGF.EmitBlock(RHSBlock);
  
  ComplexPairTy RHS = Visit(E->getRHS());
  Builder.CreateBr(ContBlock);
  RHSBlock = Builder.GetInsertBlock();
  
  CGF.EmitBlock(ContBlock);
  
  // Create a PHI node for the real part.
  llvm::PHINode *RealPN = Builder.CreatePHI(LHS.first->getType(), "cond.r");
  RealPN->reserveOperandSpace(2);
  RealPN->addIncoming(LHS.first, LHSBlock);
  RealPN->addIncoming(RHS.first, RHSBlock);

  // Create a PHI node for the imaginary part.
  llvm::PHINode *ImagPN = Builder.CreatePHI(LHS.first->getType(), "cond.i");
  ImagPN->reserveOperandSpace(2);
  ImagPN->addIncoming(LHS.second, LHSBlock);
  ImagPN->addIncoming(RHS.second, RHSBlock);
  
  return ComplexPairTy(RealPN, ImagPN);
}

//===----------------------------------------------------------------------===//
//                         Entry Point into this File
//===----------------------------------------------------------------------===//

/// EmitComplexExpr - Emit the computation of the specified expression of
/// complex type, ignoring the result.
ComplexPairTy CodeGenFunction::EmitComplexExpr(const Expr *E) {
  assert(E && E->getType()->isComplexType() &&
         "Invalid complex expression to emit");
  
  return ComplexExprEmitter(*this).Visit(const_cast<Expr*>(E));
}

