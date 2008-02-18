//===--- CGExprAgg.cpp - Emit LLVM Code from Aggregate Expressions --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/GlobalVariable.h"
#include "llvm/Support/Compiler.h"
using namespace clang;
using namespace CodeGen;

//===----------------------------------------------------------------------===//
//                        Aggregate Expression Emitter
//===----------------------------------------------------------------------===//

namespace  {
class VISIBILITY_HIDDEN AggExprEmitter : public StmtVisitor<AggExprEmitter> {
  CodeGenFunction &CGF;
  llvm::LLVMFoldingBuilder &Builder;
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
    CGF.WarnUnsupported(S, "aggregate expression");
  }
  void VisitParenExpr(ParenExpr *PE) { Visit(PE->getSubExpr()); }

  // l-values.
  void VisitDeclRefExpr(DeclRefExpr *DRE) { EmitAggLoadOfLValue(DRE); }
  void VisitMemberExpr(MemberExpr *ME) { EmitAggLoadOfLValue(ME); }
  void VisitUnaryDeref(UnaryOperator *E) { EmitAggLoadOfLValue(E); }
  void VisitStringLiteral(StringLiteral *E) { EmitAggLoadOfLValue(E); }

  void VisitArraySubscriptExpr(ArraySubscriptExpr *E) {
    EmitAggLoadOfLValue(E);
  }

  // Operators.
  //  case Expr::UnaryOperatorClass:
  //  case Expr::CastExprClass: 
  void VisitImplicitCastExpr(ImplicitCastExpr *E);
  void VisitCallExpr(const CallExpr *E);
  void VisitStmtExpr(const StmtExpr *E);
  void VisitBinaryOperator(const BinaryOperator *BO);
  void VisitBinAssign(const BinaryOperator *E);
  void VisitOverloadExpr(const OverloadExpr *E);

  
  void VisitConditionalOperator(const ConditionalOperator *CO);
  void VisitInitListExpr(InitListExpr *E);
  //  case Expr::ChooseExprClass:

private:

  llvm::Constant *GetConstantInit(InitListExpr *E,
                                  const llvm::ArrayType *AType);
  void EmitNonConstInit(Expr *E, llvm::Value *Dest, const llvm::Type *DestType);
};
}  // end anonymous namespace.

//===----------------------------------------------------------------------===//
//                                Utilities
//===----------------------------------------------------------------------===//

void AggExprEmitter::EmitAggregateCopy(llvm::Value *DestPtr,
                                       llvm::Value *SrcPtr, QualType Ty) {
  assert(!Ty->isComplexType() && "Shouldn't happen for complex");
  
  // Aggregate assignment turns into llvm.memcpy.
  const llvm::Type *BP = llvm::PointerType::getUnqual(llvm::Type::Int8Ty);
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
    // TypeInfo.first describes size in bits.
    llvm::ConstantInt::get(IntPtr, TypeInfo.first/8),
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

void AggExprEmitter::VisitImplicitCastExpr(ImplicitCastExpr *E)
{
  QualType STy = E->getSubExpr()->getType().getCanonicalType();
  QualType Ty = E->getType().getCanonicalType();

  assert(CGF.getContext().typesAreCompatible(
             STy.getUnqualifiedType(), Ty.getUnqualifiedType())
         && "Implicit cast types must be compatible");
  
  Visit(E->getSubExpr());
}

void AggExprEmitter::VisitCallExpr(const CallExpr *E)
{
  RValue RV = CGF.EmitCallExpr(E);
  assert(RV.isAggregate() && "Return value must be aggregate value!");
  
  // If the result is ignored, don't copy from the value.
  if (DestPtr == 0)
    // FIXME: If the source is volatile, we must read from it.
    return;
  
  EmitAggregateCopy(DestPtr, RV.getAggregateAddr(), E->getType());
}

void AggExprEmitter::VisitOverloadExpr(const OverloadExpr *E)
{
  RValue RV = CGF.EmitCallExpr(E->getFn(), E->arg_begin(),
                               E->getNumArgs(CGF.getContext()));
  assert(RV.isAggregate() && "Return value must be aggregate value!");
  
  // If the result is ignored, don't copy from the value.
  if (DestPtr == 0)
    // FIXME: If the source is volatile, we must read from it.
    return;
  
  EmitAggregateCopy(DestPtr, RV.getAggregateAddr(), E->getType());
}

void AggExprEmitter::VisitStmtExpr(const StmtExpr *E) {
  CGF.EmitCompoundStmt(*E->getSubStmt(), true, DestPtr, VolatileDest);
}

void AggExprEmitter::VisitBinaryOperator(const BinaryOperator *E) {
  CGF.WarnUnsupported(E, "aggregate binary expression");
}

void AggExprEmitter::VisitBinAssign(const BinaryOperator *E) {
  // For an assignment to work, the value on the right has
  // to be compatible with the value on the left.
  assert(CGF.getContext().typesAreCompatible(
             E->getLHS()->getType().getUnqualifiedType(),
             E->getRHS()->getType().getUnqualifiedType())
         && "Invalid assignment");
  LValue LHS = CGF.EmitLValue(E->getLHS());

  // Codegen the RHS so that it stores directly into the LHS.
  CGF.EmitAggExpr(E->getRHS(), LHS.getAddress(), false /*FIXME: VOLATILE LHS*/);

  if (DestPtr == 0)
    return;

  // If the result of the assignment is used, copy the RHS there also.
  EmitAggregateCopy(DestPtr, LHS.getAddress(), E->getType());
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

llvm::Constant *AggExprEmitter::GetConstantInit(InitListExpr *E,
                                                const llvm::ArrayType *AType) {
  std::vector<llvm::Constant*> ArrayElts;
  unsigned NumInitElements = E->getNumInits();
  const llvm::Type *ElementType = AType->getElementType();
  unsigned i;

  for (i = 0; i != NumInitElements; ++i) {
    if (InitListExpr *InitList = dyn_cast<InitListExpr>(E->getInit(i))) {
      assert(isa<llvm::ArrayType>(ElementType) && "Invalid initilizer");
      llvm::Constant *C =
        GetConstantInit(InitList, cast<llvm::ArrayType>(ElementType));
      if (!C) return NULL;
      ArrayElts.push_back(C);
    } else if (llvm::Constant *C =
        dyn_cast<llvm::Constant>(CGF.EmitScalarExpr(E->getInit(i))))
      ArrayElts.push_back(C);
    else
      return NULL;
  }

  // Remaining default initializers
  unsigned NumArrayElements = AType->getNumElements();
  for (/*Do not initialize i*/; i < NumArrayElements; ++i)
      ArrayElts.push_back(llvm::Constant::getNullValue(ElementType));

  return llvm::ConstantArray::get(AType, ArrayElts);
}

void AggExprEmitter::EmitNonConstInit(Expr *E, llvm::Value *DestPtr,
                                      const llvm::Type *DestType) {

  if (const llvm::ArrayType *AType = dyn_cast<llvm::ArrayType>(DestType)) {
    unsigned NumInitElements = 0;
    InitListExpr *InitList = NULL;

    if (E) {
      InitList = cast<InitListExpr>(E);
      NumInitElements = InitList->getNumInits();
    }

    llvm::Value *Idxs[] = {
      llvm::Constant::getNullValue(llvm::Type::Int32Ty),
      NULL
    };
    llvm::Value *NextVal = NULL;
    unsigned i;
    for (i = 0; i != NumInitElements; ++i) {
      Idxs[1] = llvm::ConstantInt::get(llvm::Type::Int32Ty, i);
      NextVal = Builder.CreateGEP(DestPtr, Idxs, Idxs + 2,".array");
      EmitNonConstInit(InitList->getInit(i), NextVal, AType->getElementType());
    }

    // Emit remaining default initializers
    unsigned NumArrayElements = AType->getNumElements();
    for (/*Do not initialize i*/; i < NumArrayElements; ++i) {
      Idxs[1] = llvm::ConstantInt::get(llvm::Type::Int32Ty, i);
      NextVal = Builder.CreateGEP(DestPtr, Idxs, Idxs + 2,".array");
      EmitNonConstInit(NULL, NextVal, AType->getElementType());
    }

  } else {
    llvm::Value *V;
    if (E)
      V = CGF.EmitScalarExpr(E);
    else
      V = llvm::Constant::getNullValue(DestType);
    Builder.CreateStore(V, DestPtr);
  }
}

void AggExprEmitter::VisitInitListExpr(InitListExpr *E) {

  if (!E->getType()->isArrayType()) {
    CGF.WarnUnsupported(E, "aggregate init-list expression");
    return;
  }

  const llvm::PointerType *APType = cast<llvm::PointerType>(DestPtr->getType());
  const llvm::ArrayType *AType =
    cast<llvm::ArrayType>(APType->getElementType());

  llvm::Constant *V = GetConstantInit(E, AType);
  if (V) {
    // Create global value to hold this array.
    V = new llvm::GlobalVariable(V->getType(), true,
                                 llvm::GlobalValue::InternalLinkage,
                                 V, ".array",
                                 &CGF.CGM.getModule());

    EmitAggregateCopy(DestPtr, V , E->getType());
    return;
  } else
    EmitNonConstInit(E, DestPtr, AType);
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
