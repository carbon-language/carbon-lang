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
public:
  ComplexExprEmitter(CodeGenFunction &cgf) : CGF(cgf) {
  }

  
  //===--------------------------------------------------------------------===//
  //                               Utilities
  //===--------------------------------------------------------------------===//

  /// EmitLoadOfLValue - Given an expression with complex type that represents a
  /// value l-value, this method emits the address of the l-value, then loads
  /// and returns the result.
  ComplexPairTy EmitLoadOfLValue(const Expr *E) {
    return EmitLoadOfComplex(CGF.EmitLValue(E).getAddress());
  }
  
  /// EmitLoadOfComplex - Given a pointer to a complex value, emit code to load
  /// the real and imaginary pieces.
  ComplexPairTy EmitLoadOfComplex(llvm::Value *SrcPtr);
  
  /// EmitStoreOfComplex - Store the specified real/imag parts into the
  /// specified value pointer.
  void EmitStoreOfComplex(ComplexPairTy Val, llvm::Value *ResPtr);
  
  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//
  
  ComplexPairTy VisitStmt(Stmt *S) {
    fprintf(stderr, "Unimplemented agg expr!\n");
    S->dump();
    return ComplexPairTy();
  }
  ComplexPairTy VisitParenExpr(ParenExpr *PE) { return Visit(PE->getSubExpr());}

  // l-values.
  ComplexPairTy VisitDeclRefExpr(DeclRefExpr *DRE) {
    return EmitLoadOfLValue(DRE);
  }
  //  case Expr::ArraySubscriptExprClass:

  // Operators.
  //  case Expr::UnaryOperatorClass:
  //  case Expr::ImplicitCastExprClass:
  //  case Expr::CastExprClass: 
  //  case Expr::CallExprClass:
  ComplexPairTy VisitBinaryOperator(const BinaryOperator *BO);
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
ComplexPairTy ComplexExprEmitter::EmitLoadOfComplex(llvm::Value *SrcPtr) {
  llvm::Constant *Zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0);
  llvm::Constant *One  = llvm::ConstantInt::get(llvm::Type::Int32Ty, 1);
  // FIXME: It would be nice to make this "Ptr->getName()+realp"
  llvm::Value *RealPtr = CGF.Builder.CreateGEP(SrcPtr, Zero, Zero, "realp");
  llvm::Value *ImagPtr = CGF.Builder.CreateGEP(SrcPtr, Zero, One, "imagp");
  
  // FIXME: Handle volatility.
  // FIXME: It would be nice to make this "Ptr->getName()+real"
  llvm::Value *Real = CGF.Builder.CreateLoad(RealPtr, "real");
  llvm::Value *Imag = CGF.Builder.CreateLoad(ImagPtr, "imag");
  return ComplexPairTy(Real, Imag);
}

/// EmitStoreOfComplex - Store the specified real/imag parts into the
/// specified value pointer.
void ComplexExprEmitter::EmitStoreOfComplex(ComplexPairTy V, llvm::Value *Ptr) {
  llvm::Constant *Zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0);
  llvm::Constant *One  = llvm::ConstantInt::get(llvm::Type::Int32Ty, 1);
  llvm::Value *RealPtr = CGF.Builder.CreateGEP(Ptr, Zero, Zero, "real");
  llvm::Value *ImagPtr = CGF.Builder.CreateGEP(Ptr, Zero, One, "imag");
  
  // FIXME: Handle volatility.
  CGF.Builder.CreateStore(V.first, RealPtr);
  CGF.Builder.CreateStore(V.second, ImagPtr);
}



//===----------------------------------------------------------------------===//
//                            Visitor Methods
//===----------------------------------------------------------------------===//

ComplexPairTy ComplexExprEmitter::VisitBinaryOperator(const BinaryOperator *E) {
  fprintf(stderr, "Unimplemented complex binary expr!\n");
  E->dump();
  return ComplexPairTy();
}

ComplexPairTy ComplexExprEmitter::VisitBinAdd(const BinaryOperator *E) {
  ComplexPairTy LHS = Visit(E->getLHS());
  ComplexPairTy RHS = Visit(E->getRHS());
  
  llvm::Value *ResR = CGF.Builder.CreateAdd(LHS.first,  RHS.first,  "add.r");
  llvm::Value *ResI = CGF.Builder.CreateAdd(LHS.second, RHS.second, "add.i");

  return ComplexPairTy(ResR, ResI);
}

ComplexPairTy ComplexExprEmitter::VisitBinMul(const BinaryOperator *E) {
  ComplexPairTy LHS = Visit(E->getLHS());
  ComplexPairTy RHS = Visit(E->getRHS());
  
  llvm::Value *ResRl = CGF.Builder.CreateMul(LHS.first, RHS.first, "mul.rl");
  llvm::Value *ResRr = CGF.Builder.CreateMul(LHS.second, RHS.second, "mul.rr");
  llvm::Value *ResR  = CGF.Builder.CreateSub(ResRl, ResRr, "mul.r");
  
  llvm::Value *ResIl = CGF.Builder.CreateMul(LHS.second, RHS.first, "mul.il");
  llvm::Value *ResIr = CGF.Builder.CreateMul(LHS.first, RHS.second, "mul.ir");
  llvm::Value *ResI  = CGF.Builder.CreateAdd(ResIl, ResIr, "mul.i");

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
  EmitStoreOfComplex(Val, LHS.getAddress());
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
  CGF.Builder.CreateCondBr(Cond, LHSBlock, RHSBlock);
  
  CGF.EmitBlock(LHSBlock);
  
  // Handle the GNU extension for missing LHS.
  assert(E->getLHS() && "Must have LHS for complex value");

  ComplexPairTy LHS = Visit(E->getLHS());
  CGF.Builder.CreateBr(ContBlock);
  LHSBlock = CGF.Builder.GetInsertBlock();
  
  CGF.EmitBlock(RHSBlock);
  
  ComplexPairTy RHS = Visit(E->getRHS());
  CGF.Builder.CreateBr(ContBlock);
  RHSBlock = CGF.Builder.GetInsertBlock();
  
  CGF.EmitBlock(ContBlock);
  
  // Create a PHI node for the real part.
  llvm::PHINode *RealPN = CGF.Builder.CreatePHI(LHS.first->getType(), "cond.r");
  RealPN->reserveOperandSpace(2);
  RealPN->addIncoming(LHS.first, LHSBlock);
  RealPN->addIncoming(RHS.first, RHSBlock);

  // Create a PHI node for the imaginary part.
  llvm::PHINode *ImagPN = CGF.Builder.CreatePHI(LHS.first->getType(), "cond.i");
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

