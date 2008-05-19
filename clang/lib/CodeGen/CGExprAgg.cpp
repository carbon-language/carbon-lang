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
#include "llvm/Intrinsics.h"
using namespace clang;
using namespace CodeGen;

//===----------------------------------------------------------------------===//
//                        Aggregate Expression Emitter
//===----------------------------------------------------------------------===//

namespace  {
class VISIBILITY_HIDDEN AggExprEmitter : public StmtVisitor<AggExprEmitter> {
  CodeGenFunction &CGF;
  llvm::IRBuilder &Builder;
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

  void EmitAggregateClear(llvm::Value *DestPtr, QualType Ty);

  void EmitNonConstInit(InitListExpr *E);

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
  void VisitCXXDefaultArgExpr(CXXDefaultArgExpr *DAE) {
    Visit(DAE->getExpr());
  }

  void EmitInitializationToLValue(Expr *E, LValue Address);
  void EmitNullInitializationToLValue(LValue Address, QualType T);
  //  case Expr::ChooseExprClass:

};
}  // end anonymous namespace.

//===----------------------------------------------------------------------===//
//                                Utilities
//===----------------------------------------------------------------------===//

void AggExprEmitter::EmitAggregateClear(llvm::Value *DestPtr, QualType Ty) {
  assert(!Ty->isAnyComplexType() && "Shouldn't happen for complex");

  // Aggregate assignment turns into llvm.memset.
  const llvm::Type *BP = llvm::PointerType::getUnqual(llvm::Type::Int8Ty);
  if (DestPtr->getType() != BP)
    DestPtr = Builder.CreateBitCast(DestPtr, BP, "tmp");

  // Get size and alignment info for this aggregate.
  std::pair<uint64_t, unsigned> TypeInfo = CGF.getContext().getTypeInfo(Ty);

  // FIXME: Handle variable sized types.
  const llvm::Type *IntPtr = llvm::IntegerType::get(CGF.LLVMPointerWidth);

  llvm::Value *MemSetOps[4] = {
    DestPtr,
    llvm::ConstantInt::getNullValue(llvm::Type::Int8Ty),
    // TypeInfo.first describes size in bits.
    llvm::ConstantInt::get(IntPtr, TypeInfo.first/8),
    llvm::ConstantInt::get(llvm::Type::Int32Ty, TypeInfo.second/8)
  };
 
  Builder.CreateCall(CGF.CGM.getMemSetFn(), MemSetOps, MemSetOps+4);
}

void AggExprEmitter::EmitAggregateCopy(llvm::Value *DestPtr,
                                       llvm::Value *SrcPtr, QualType Ty) {
  assert(!Ty->isAnyComplexType() && "Shouldn't happen for complex");
  
  // Aggregate assignment turns into llvm.memcpy.
  const llvm::Type *BP = llvm::PointerType::getUnqual(llvm::Type::Int8Ty);
  if (DestPtr->getType() != BP)
    DestPtr = Builder.CreateBitCast(DestPtr, BP, "tmp");
  if (SrcPtr->getType() != BP)
    SrcPtr = Builder.CreateBitCast(SrcPtr, BP, "tmp");
  
  // Get size and alignment info for this aggregate.
  std::pair<uint64_t, unsigned> TypeInfo = CGF.getContext().getTypeInfo(Ty);
  
  // FIXME: Handle variable sized types.
  const llvm::Type *IntPtr = llvm::IntegerType::get(CGF.LLVMPointerWidth);
  
  llvm::Value *MemCpyOps[4] = {
    DestPtr, SrcPtr,
    // TypeInfo.first describes size in bits.
    llvm::ConstantInt::get(IntPtr, TypeInfo.first/8),
    llvm::ConstantInt::get(llvm::Type::Int32Ty, TypeInfo.second/8)
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
  llvm::BasicBlock *LHSBlock = llvm::BasicBlock::Create("cond.?");
  llvm::BasicBlock *RHSBlock = llvm::BasicBlock::Create("cond.:");
  llvm::BasicBlock *ContBlock = llvm::BasicBlock::Create("cond.cont");
  
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

void AggExprEmitter::EmitNonConstInit(InitListExpr *E) {

  const llvm::PointerType *APType =
    cast<llvm::PointerType>(DestPtr->getType());
  const llvm::Type *DestType = APType->getElementType();

  if (const llvm::ArrayType *AType = dyn_cast<llvm::ArrayType>(DestType)) {
    unsigned NumInitElements = E->getNumInits();

    unsigned i;
    for (i = 0; i != NumInitElements; ++i) {
      llvm::Value *NextVal = Builder.CreateStructGEP(DestPtr, i, ".array");
      Expr *Init = E->getInit(i);
      if (isa<InitListExpr>(Init))
        CGF.EmitAggExpr(Init, NextVal, VolatileDest);
      else
        Builder.CreateStore(CGF.EmitScalarExpr(Init), NextVal);
    }

    // Emit remaining default initializers
    unsigned NumArrayElements = AType->getNumElements();
    QualType QType = E->getInit(0)->getType();
    const llvm::Type *EType = AType->getElementType();
    for (/*Do not initialize i*/; i < NumArrayElements; ++i) {
      llvm::Value *NextVal = Builder.CreateStructGEP(DestPtr, i, ".array");
      if (EType->isFirstClassType())
        Builder.CreateStore(llvm::Constant::getNullValue(EType), NextVal);
      else
        EmitAggregateClear(NextVal, QType);
    }
  } else
    assert(false && "Invalid initializer");
}

void AggExprEmitter::EmitInitializationToLValue(Expr* E, LValue LV) {
  // FIXME: Are initializers affected by volatile?
  if (E->getType()->isComplexType()) {
    CGF.EmitComplexExprIntoAddr(E, LV.getAddress(), false);
  } else if (CGF.hasAggregateLLVMType(E->getType())) {
    CGF.EmitAnyExpr(E, LV.getAddress(), false);
  } else {
    CGF.EmitStoreThroughLValue(CGF.EmitAnyExpr(E), LV, E->getType());
  }
}

void AggExprEmitter::EmitNullInitializationToLValue(LValue LV, QualType T) {
  if (!CGF.hasAggregateLLVMType(T)) {
    // For non-aggregates, we can store zero
    const llvm::Type *T =
       cast<llvm::PointerType>(LV.getAddress()->getType())->getElementType();
    Builder.CreateStore(llvm::Constant::getNullValue(T), LV.getAddress());
  } else {
    // Otherwise, just memset the whole thing to zero.  This is legal
    // because in LLVM, all default initializers are guaranteed to have a
    // bit pattern of all zeros.
    // There's a potential optimization opportunity in combining
    // memsets; that would be easy for arrays, but relatively
    // difficult for structures with the current code.
    llvm::Value *MemSet = CGF.CGM.getIntrinsic(llvm::Intrinsic::memset_i64);
    uint64_t Size = CGF.getContext().getTypeSize(T);
    
    const llvm::Type *BP = llvm::PointerType::getUnqual(llvm::Type::Int8Ty);
    llvm::Value* DestPtr = Builder.CreateBitCast(LV.getAddress(), BP, "tmp");
    Builder.CreateCall4(MemSet, DestPtr, 
                        llvm::ConstantInt::get(llvm::Type::Int8Ty, 0),
                        llvm::ConstantInt::get(llvm::Type::Int64Ty, Size/8),
                        llvm::ConstantInt::get(llvm::Type::Int32Ty, 0));
  }
}

void AggExprEmitter::VisitInitListExpr(InitListExpr *E) {
  if (E->isConstantExpr(CGF.getContext(), 0)) {
    // FIXME: call into const expr emitter so that we can emit
    // a memcpy instead of storing the individual members.
    // This is purely for perf; both codepaths lead to equivalent
    // (although not necessarily identical) code.
    // It's worth noting that LLVM keeps on getting smarter, though,
    // so it might not be worth bothering.
  }
  
  // Handle initialization of an array.
  if (E->getType()->isArrayType()) {
    const llvm::PointerType *APType =
      cast<llvm::PointerType>(DestPtr->getType());
    const llvm::ArrayType *AType =
      cast<llvm::ArrayType>(APType->getElementType());
    
    uint64_t NumInitElements = E->getNumInits();

    if (E->getNumInits() > 0 &&
        E->getType().getCanonicalType().getUnqualifiedType() == 
          E->getInit(0)->getType().getCanonicalType().getUnqualifiedType()) {
      EmitAggLoadOfLValue(E->getInit(0));
      return;
    }

    uint64_t NumArrayElements = AType->getNumElements();
    QualType ElementType = E->getType()->getAsArrayType()->getElementType();
    
    for (uint64_t i = 0; i != NumArrayElements; ++i) {
      llvm::Value *NextVal = Builder.CreateStructGEP(DestPtr, i, ".array");
      if (i < NumInitElements)
        EmitInitializationToLValue(E->getInit(i), LValue::MakeAddr(NextVal));
      else
        EmitNullInitializationToLValue(LValue::MakeAddr(NextVal),
                                       ElementType);
    }
    return;
  }
  
  assert(E->getType()->isRecordType() && "Only support structs/unions here!");
  
  // Do struct initialization; this code just sets each individual member
  // to the approprate value.  This makes bitfield support automatic;
  // the disadvantage is that the generated code is more difficult for
  // the optimizer, especially with bitfields.
  unsigned NumInitElements = E->getNumInits();
  RecordDecl *SD = E->getType()->getAsRecordType()->getDecl();
  unsigned NumMembers = SD->getNumMembers() - SD->hasFlexibleArrayMember();
  unsigned CurInitVal = 0;
  bool isUnion = E->getType()->isUnionType();
  
  // Here we iterate over the fields; this makes it simpler to both
  // default-initialize fields and skip over unnamed fields.
  for (unsigned CurFieldNo = 0; CurFieldNo != NumMembers; ++CurFieldNo) {
    if (CurInitVal >= NumInitElements) {
      // No more initializers; we're done.
      break;
    }
    
    FieldDecl *CurField = SD->getMember(CurFieldNo);
    if (CurField->getIdentifier() == 0) {
      // Initializers can't initialize unnamed fields, e.g. "int : 20;"
      continue;
    }
    LValue FieldLoc = CGF.EmitLValueForField(DestPtr, CurField, isUnion);
    if (CurInitVal < NumInitElements) {
      // Store the initializer into the field
      // This will probably have to get a bit smarter when we support
      // designators in initializers
      EmitInitializationToLValue(E->getInit(CurInitVal++), FieldLoc);
    } else {
      // We're out of initalizers; default-initialize to null
      EmitNullInitializationToLValue(FieldLoc, CurField->getType());
    }

    // Unions only initialize one field.
    // (things can get weird with designators, but they aren't
    // supported yet.)
    if (E->getType()->isUnionType())
      break;
  }
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
