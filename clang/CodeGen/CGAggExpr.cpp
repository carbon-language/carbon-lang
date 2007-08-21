//===--- CGAggExpr.cpp - Emit LLVM Code from Aggregate Expressions --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Aggregate Expr nodes as LLVM code.
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

// FIXME: Handle volatility!
void CodeGenFunction::EmitAggregateCopy(llvm::Value *DestPtr,
                                        llvm::Value *SrcPtr, QualType Ty) {
  // Don't use memcpy for complex numbers.
  if (Ty->isComplexType()) {
    llvm::Value *Real, *Imag;
    EmitLoadOfComplex(RValue::getAggregate(SrcPtr), Real, Imag);
    EmitStoreOfComplex(Real, Imag, DestPtr);
    return;
  }
  
  // Aggregate assignment turns into llvm.memcpy.
  const llvm::Type *BP = llvm::PointerType::get(llvm::Type::Int8Ty);
  if (DestPtr->getType() != BP)
    DestPtr = Builder.CreateBitCast(DestPtr, BP, "tmp");
  if (SrcPtr->getType() != BP)
    SrcPtr = Builder.CreateBitCast(SrcPtr, BP, "tmp");
  
  // Get size and alignment info for this aggregate.
  std::pair<uint64_t, unsigned> TypeInfo =
    getContext().getTypeInfo(Ty, SourceLocation());
  
  // FIXME: Handle variable sized types.
  const llvm::Type *IntPtr = llvm::IntegerType::get(LLVMPointerWidth);
  
  llvm::Value *MemCpyOps[4] = {
    DestPtr, SrcPtr,
    llvm::ConstantInt::get(IntPtr, TypeInfo.first),
    llvm::ConstantInt::get(llvm::Type::Int32Ty, TypeInfo.second)
  };
  
  Builder.CreateCall(CGM.getMemCpyFn(), MemCpyOps, MemCpyOps+4);
}

//===----------------------------------------------------------------------===//
//                        Aggregate Expression Emitter
//===----------------------------------------------------------------------===//

namespace  {
class VISIBILITY_HIDDEN AggExprEmitter : public StmtVisitor<AggExprEmitter> {
  CodeGenFunction &CGF;
  llvm::Value *DestPtr;
  bool VolatileDest;
public:
  AggExprEmitter(CodeGenFunction &cgf, llvm::Value *destPtr, bool volatileDest)
    : CGF(cgf), DestPtr(destPtr), VolatileDest(volatileDest) {
  }

  /// EmitAggLoadOfLValue - Given an expression with aggregate type that
  /// represents a value lvalue, this method emits the address of the lvalue,
  /// then loads the result into DestPtr.
  void EmitAggLoadOfLValue(const Expr *E);
  
  void VisitStmt(Stmt *S) {
    fprintf(stderr, "Unimplemented agg expr!\n");
    S->dump();
  }
  void VisitParenExpr(ParenExpr *PE) { Visit(PE->getSubExpr()); }

  // l-values.
  void VisitDeclRefExpr(DeclRefExpr *DRE) { return EmitAggLoadOfLValue(DRE); }
  //  case Expr::ArraySubscriptExprClass:

  // Operators.
  //  case Expr::UnaryOperatorClass:
  //  case Expr::ImplicitCastExprClass:
  //  case Expr::CastExprClass: 
  //  case Expr::CallExprClass:
  void VisitBinaryOperator(const BinaryOperator *BO);
  void VisitBinAssign(const BinaryOperator *E);

  
  void VisitConditionalOperator(const ConditionalOperator *CO);
  //  case Expr::ChooseExprClass:
};
}  // end anonymous namespace.




/// EmitAggExpr - Emit the computation of the specified expression of
/// aggregate type.  The result is computed into DestPtr.  Note that if
/// DestPtr is null, the value of the aggregate expression is not needed.
void CodeGenFunction::EmitAggExpr(const Expr *E, llvm::Value *DestPtr,
                                  bool VolatileDest) {
  assert(E && hasAggregateLLVMType(E->getType()) &&
         "Invalid aggregate expression to emit");
  
  AggExprEmitter(*this, DestPtr, VolatileDest).Visit(const_cast<Expr*>(E));
}

/// EmitAggLoadOfLValue - Given an expression with aggregate type that
/// represents a value lvalue, this method emits the address of the lvalue,
/// then loads the result into DestPtr.
void AggExprEmitter::EmitAggLoadOfLValue(const Expr *E) {
  LValue LV = CGF.EmitLValue(E);
  assert(LV.isSimple() && "Can't have aggregate bitfield, vector, etc");
  llvm::Value *SrcPtr = LV.getAddress();
  
  // If the result is ignored, don't copy from the value.
  if (DestPtr == 0)
    // FIXME: If the source is volatile, we must read from it.
    return;

  CGF.EmitAggregateCopy(DestPtr, SrcPtr, E->getType());
}

void AggExprEmitter::VisitBinaryOperator(const BinaryOperator *E) {
  fprintf(stderr, "Unimplemented aggregate binary expr!\n");
  E->dump();
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

void AggExprEmitter::VisitBinAssign(const BinaryOperator *E) {
  assert(E->getLHS()->getType().getCanonicalType() ==
         E->getRHS()->getType().getCanonicalType() && "Invalid assignment");
  LValue LHS = CGF.EmitLValue(E->getLHS());

  // Codegen the RHS so that it stores directly into the LHS.
  CGF.EmitAggExpr(E->getRHS(), LHS.getAddress(), false /*FIXME: VOLATILE LHS*/);

  // If the result of the assignment is used, copy the RHS there also.
  if (DestPtr) {
    assert(0 && "FIXME: Chained agg assignment not implemented yet");
  }
}


void AggExprEmitter::VisitConditionalOperator(const ConditionalOperator *E) {
  llvm::BasicBlock *LHSBlock = new llvm::BasicBlock("cond.?");
  llvm::BasicBlock *RHSBlock = new llvm::BasicBlock("cond.:");
  llvm::BasicBlock *ContBlock = new llvm::BasicBlock("cond.cont");
  
  llvm::Value *Cond = CGF.EvaluateExprAsBool(E->getCond());
  CGF.Builder.CreateCondBr(Cond, LHSBlock, RHSBlock);
  
  CGF.EmitBlock(LHSBlock);
  
  // Handle the GNU extension for missing LHS.
  assert(E->getLHS() && "Must have LHS for aggregate value");

  CGF.EmitAggExpr(E->getLHS(), DestPtr, VolatileDest);
  CGF.Builder.CreateBr(ContBlock);
  LHSBlock =CGF. Builder.GetInsertBlock();
  
  CGF.EmitBlock(RHSBlock);
  
  CGF.EmitAggExpr(E->getRHS(), DestPtr, VolatileDest);
  CGF.Builder.CreateBr(ContBlock);
  RHSBlock = CGF.Builder.GetInsertBlock();
  
  CGF.EmitBlock(ContBlock);
}
