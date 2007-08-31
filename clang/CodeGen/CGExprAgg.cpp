//===--- CGExprAgg.cpp - Emit LLVM Code from Aggregate Expressions --------===//
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

//===----------------------------------------------------------------------===//
//                        Aggregate Expression Emitter
//===----------------------------------------------------------------------===//

namespace  {
class VISIBILITY_HIDDEN AggExprEmitter : public StmtVisitor<AggExprEmitter> {
  CodeGenFunction &CGF;
  llvm::LLVMBuilder &Builder;
  llvm::Value *DestPtr;
  bool VolatileDest;
public:
  AggExprEmitter(CodeGenFunction &cgf, llvm::Value *destPtr, bool volatileDest)
    : CGF(cgf), Builder(CGF.Builder),
      DestPtr(destPtr), VolatileDest(volatileDest) {
  }

  //===--------------------------------------------------------------------===//
  //                               Utilities
  //===--------------------------------------------------------------------===//

  /// EmitAggLoadOfLValue - Given an expression with aggregate type that
  /// represents a value lvalue, this method emits the address of the lvalue,
  /// then loads the result into DestPtr.
  void EmitAggLoadOfLValue(const Expr *E);
  
  void EmitAggregateCopy(llvm::Value *DestPtr, llvm::Value *SrcPtr,
                         QualType EltTy);
  
  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//
  
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
  void VisitStmtExpr(const StmtExpr *E);
  void VisitBinaryOperator(const BinaryOperator *BO);
  void VisitBinAssign(const BinaryOperator *E);

  
  void VisitConditionalOperator(const ConditionalOperator *CO);
  //  case Expr::ChooseExprClass:
};
}  // end anonymous namespace.

//===----------------------------------------------------------------------===//
//                                Utilities
//===----------------------------------------------------------------------===//

void AggExprEmitter::EmitAggregateCopy(llvm::Value *DestPtr,
                                       llvm::Value *SrcPtr, QualType Ty) {
  assert(!Ty->isComplexType() && "Shouldn't happen for complex");
  
  // Aggregate assignment turns into llvm.memcpy.
  const llvm::Type *BP = llvm::PointerType::get(llvm::Type::Int8Ty);
  if (DestPtr->getType() != BP)
    DestPtr = Builder.CreateBitCast(DestPtr, BP, "tmp");
  if (SrcPtr->getType() != BP)
    SrcPtr = Builder.CreateBitCast(SrcPtr, BP, "tmp");
  
  // Get size and alignment info for this aggregate.
  std::pair<uint64_t, unsigned> TypeInfo =
    CGF.getContext().getTypeInfo(Ty, SourceLocation());
  
  // FIXME: Handle variable sized types.
  const llvm::Type *IntPtr = llvm::IntegerType::get(CGF.LLVMPointerWidth);
  
  llvm::Value *MemCpyOps[4] = {
    DestPtr, SrcPtr,
    llvm::ConstantInt::get(IntPtr, TypeInfo.first),
    llvm::ConstantInt::get(llvm::Type::Int32Ty, TypeInfo.second)
  };
  
  Builder.CreateCall(CGF.CGM.getMemCpyFn(), MemCpyOps, MemCpyOps+4);
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

  EmitAggregateCopy(DestPtr, SrcPtr, E->getType());
}

//===----------------------------------------------------------------------===//
//                            Visitor Methods
//===----------------------------------------------------------------------===//

void AggExprEmitter::VisitStmtExpr(const StmtExpr *E) {
  CGF.EmitCompoundStmt(*E->getSubStmt(), true, DestPtr, VolatileDest);
}

void AggExprEmitter::VisitBinaryOperator(const BinaryOperator *E) {
  fprintf(stderr, "Unimplemented aggregate binary expr!\n");
  E->dump();
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
  Builder.CreateCondBr(Cond, LHSBlock, RHSBlock);
  
  CGF.EmitBlock(LHSBlock);
  
  // Handle the GNU extension for missing LHS.
  assert(E->getLHS() && "Must have LHS for aggregate value");

  Visit(E->getLHS());
  Builder.CreateBr(ContBlock);
  LHSBlock = Builder.GetInsertBlock();
  
  CGF.EmitBlock(RHSBlock);
  
  Visit(E->getRHS());
  Builder.CreateBr(ContBlock);
  RHSBlock = Builder.GetInsertBlock();
  
  CGF.EmitBlock(ContBlock);
}

//===----------------------------------------------------------------------===//
//                        Entry Points into this File
//===----------------------------------------------------------------------===//

/// EmitAggExpr - Emit the computation of the specified expression of
/// aggregate type.  The result is computed into DestPtr.  Note that if
/// DestPtr is null, the value of the aggregate expression is not needed.
void CodeGenFunction::EmitAggExpr(const Expr *E, llvm::Value *DestPtr,
                                  bool VolatileDest) {
  assert(E && hasAggregateLLVMType(E->getType()) &&
         "Invalid aggregate expression to emit");
  
  AggExprEmitter(*this, DestPtr, VolatileDest).Visit(const_cast<Expr*>(E));
}
