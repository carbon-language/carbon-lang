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
  ComplexPairTy EmitLoadOfLValue(const Expr *E);
  
  
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

  // No comparisons produce a complex result.
  ComplexPairTy VisitBinAssign     (const BinaryOperator *E);

  
  ComplexPairTy VisitConditionalOperator(const ConditionalOperator *CO);
  //  case Expr::ChooseExprClass:
};
}  // end anonymous namespace.

//===----------------------------------------------------------------------===//
//                                Utilities
//===----------------------------------------------------------------------===//

/// EmitLoadOfLValue - Given an expression with complex type that represents a
/// value l-value, this method emits the address of the l-value, then loads
/// and returns the result.
ComplexPairTy ComplexExprEmitter::EmitLoadOfLValue(const Expr *E) {
  LValue LV = CGF.EmitLValue(E);
  assert(LV.isSimple() && "Can't have complex bitfield, vector, etc");
  
  // Load the real/imag values.
  llvm::Value *Real, *Imag;
  CGF.EmitLoadOfComplex(LV.getAddress(), Real, Imag);
  return ComplexPairTy(Real, Imag);
}

//===----------------------------------------------------------------------===//
//                            Visitor Methods
//===----------------------------------------------------------------------===//

ComplexPairTy ComplexExprEmitter::VisitBinaryOperator(const BinaryOperator *E) {
  fprintf(stderr, "Unimplemented complex binary expr!\n");
  E->dump();
  return ComplexPairTy();
#if 0
  switch (E->getOpcode()) {
  default:
    return;
  case BinaryOperator::Mul:
    LHS = EmitExpr(E->getLHS());
    RHS = EmitExpr(E->getRHS());
    return EmitMul(LHS, RHS, E->getType());
  case BinaryOperator::Div:
    LHS = EmitExpr(E->getLHS());
    RHS = EmitExpr(E->getRHS());
    return EmitDiv(LHS, RHS, E->getType());
  case BinaryOperator::Rem:
    LHS = EmitExpr(E->getLHS());
    RHS = EmitExpr(E->getRHS());
    return EmitRem(LHS, RHS, E->getType());
  case BinaryOperator::Add:
    LHS = EmitExpr(E->getLHS());
    RHS = EmitExpr(E->getRHS());
    if (!E->getType()->isPointerType())
      return EmitAdd(LHS, RHS, E->getType());
      
      return EmitPointerAdd(LHS, E->getLHS()->getType(),
                            RHS, E->getRHS()->getType(), E->getType());
  case BinaryOperator::Sub:
    LHS = EmitExpr(E->getLHS());
    RHS = EmitExpr(E->getRHS());
    
    if (!E->getLHS()->getType()->isPointerType())
      return EmitSub(LHS, RHS, E->getType());
      
      return EmitPointerSub(LHS, E->getLHS()->getType(),
                            RHS, E->getRHS()->getType(), E->getType());
  case BinaryOperator::Shl:
    LHS = EmitExpr(E->getLHS());
    RHS = EmitExpr(E->getRHS());
    return EmitShl(LHS, RHS, E->getType());
  case BinaryOperator::Shr:
    LHS = EmitExpr(E->getLHS());
    RHS = EmitExpr(E->getRHS());
    return EmitShr(LHS, RHS, E->getType());
  case BinaryOperator::And:
    LHS = EmitExpr(E->getLHS());
    RHS = EmitExpr(E->getRHS());
    return EmitAnd(LHS, RHS, E->getType());
  case BinaryOperator::Xor:
    LHS = EmitExpr(E->getLHS());
    RHS = EmitExpr(E->getRHS());
    return EmitXor(LHS, RHS, E->getType());
  case BinaryOperator::Or :
    LHS = EmitExpr(E->getLHS());
    RHS = EmitExpr(E->getRHS());
    return EmitOr(LHS, RHS, E->getType());
  case BinaryOperator::MulAssign: {
    const CompoundAssignOperator *CAO = cast<CompoundAssignOperator>(E);
    LValue LHSLV;
    EmitCompoundAssignmentOperands(CAO, LHSLV, LHS, RHS);
    LHS = EmitMul(LHS, RHS, CAO->getComputationType());
    return EmitCompoundAssignmentResult(CAO, LHSLV, LHS);
  }
  case BinaryOperator::DivAssign: {
    const CompoundAssignOperator *CAO = cast<CompoundAssignOperator>(E);
    LValue LHSLV;
    EmitCompoundAssignmentOperands(CAO, LHSLV, LHS, RHS);
    LHS = EmitDiv(LHS, RHS, CAO->getComputationType());
    return EmitCompoundAssignmentResult(CAO, LHSLV, LHS);
  }
  case BinaryOperator::RemAssign: {
    const CompoundAssignOperator *CAO = cast<CompoundAssignOperator>(E);
    LValue LHSLV;
    EmitCompoundAssignmentOperands(CAO, LHSLV, LHS, RHS);
    LHS = EmitRem(LHS, RHS, CAO->getComputationType());
    return EmitCompoundAssignmentResult(CAO, LHSLV, LHS);
  }
  case BinaryOperator::AddAssign: {
    const CompoundAssignOperator *CAO = cast<CompoundAssignOperator>(E);
    LValue LHSLV;
    EmitCompoundAssignmentOperands(CAO, LHSLV, LHS, RHS);
    LHS = EmitAdd(LHS, RHS, CAO->getComputationType());
    return EmitCompoundAssignmentResult(CAO, LHSLV, LHS);
  }
  case BinaryOperator::SubAssign: {
    const CompoundAssignOperator *CAO = cast<CompoundAssignOperator>(E);
    LValue LHSLV;
    EmitCompoundAssignmentOperands(CAO, LHSLV, LHS, RHS);
    LHS = EmitSub(LHS, RHS, CAO->getComputationType());
    return EmitCompoundAssignmentResult(CAO, LHSLV, LHS);
  }
  case BinaryOperator::ShlAssign: {
    const CompoundAssignOperator *CAO = cast<CompoundAssignOperator>(E);
    LValue LHSLV;
    EmitCompoundAssignmentOperands(CAO, LHSLV, LHS, RHS);
    LHS = EmitShl(LHS, RHS, CAO->getComputationType());
    return EmitCompoundAssignmentResult(CAO, LHSLV, LHS);
  }
  case BinaryOperator::ShrAssign: {
    const CompoundAssignOperator *CAO = cast<CompoundAssignOperator>(E);
    LValue LHSLV;
    EmitCompoundAssignmentOperands(CAO, LHSLV, LHS, RHS);
    LHS = EmitShr(LHS, RHS, CAO->getComputationType());
    return EmitCompoundAssignmentResult(CAO, LHSLV, LHS);
  }
  case BinaryOperator::AndAssign: {
    const CompoundAssignOperator *CAO = cast<CompoundAssignOperator>(E);
    LValue LHSLV;
    EmitCompoundAssignmentOperands(CAO, LHSLV, LHS, RHS);
    LHS = EmitAnd(LHS, RHS, CAO->getComputationType());
    return EmitCompoundAssignmentResult(CAO, LHSLV, LHS);
  }
  case BinaryOperator::OrAssign: {
    const CompoundAssignOperator *CAO = cast<CompoundAssignOperator>(E);
    LValue LHSLV;
    EmitCompoundAssignmentOperands(CAO, LHSLV, LHS, RHS);
    LHS = EmitOr(LHS, RHS, CAO->getComputationType());
    return EmitCompoundAssignmentResult(CAO, LHSLV, LHS);
  }
  case BinaryOperator::XorAssign: {
    const CompoundAssignOperator *CAO = cast<CompoundAssignOperator>(E);
    LValue LHSLV;
    EmitCompoundAssignmentOperands(CAO, LHSLV, LHS, RHS);
    LHS = EmitXor(LHS, RHS, CAO->getComputationType());
    return EmitCompoundAssignmentResult(CAO, LHSLV, LHS);
  }
  case BinaryOperator::Comma: return EmitBinaryComma(E);
  }
#endif
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
  CGF.EmitStoreOfComplex(Val.first, Val.second, LHS.getAddress());
  return Val;
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

