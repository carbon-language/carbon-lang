//===--- CGExpr.cpp - Emit LLVM Code from Expressions ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Expr nodes as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "CGCall.h"
#include "CGObjCRuntime.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "llvm/Target/TargetData.h"
using namespace clang;
using namespace CodeGen;

//===--------------------------------------------------------------------===//
//                        Miscellaneous Helper Methods
//===--------------------------------------------------------------------===//

/// CreateTempAlloca - This creates a alloca and inserts it into the entry
/// block.
llvm::AllocaInst *CodeGenFunction::CreateTempAlloca(const llvm::Type *Ty,
                                                    const char *Name) {
  if (!Builder.isNamePreserving())
    Name = "";
  return new llvm::AllocaInst(Ty, 0, Name, AllocaInsertPt);
}

/// EvaluateExprAsBool - Perform the usual unary conversions on the specified
/// expression and compare the result against zero, returning an Int1Ty value.
llvm::Value *CodeGenFunction::EvaluateExprAsBool(const Expr *E) {
  QualType BoolTy = getContext().BoolTy;
  if (!E->getType()->isAnyComplexType())
    return EmitScalarConversion(EmitScalarExpr(E), E->getType(), BoolTy);

  return EmitComplexToScalarConversion(EmitComplexExpr(E), E->getType(),BoolTy);
}

/// EmitAnyExpr - Emit code to compute the specified expression which can have
/// any type.  The result is returned as an RValue struct.  If this is an
/// aggregate expression, the aggloc/agglocvolatile arguments indicate where
/// the result should be returned.
RValue CodeGenFunction::EmitAnyExpr(const Expr *E, llvm::Value *AggLoc, 
                                    bool IsAggLocVolatile, bool IgnoreResult,
                                    bool IsInitializer) {
  if (!hasAggregateLLVMType(E->getType()))
    return RValue::get(EmitScalarExpr(E, IgnoreResult));
  else if (E->getType()->isAnyComplexType())
    return RValue::getComplex(EmitComplexExpr(E, false, false,
                                              IgnoreResult, IgnoreResult));
  
  EmitAggExpr(E, AggLoc, IsAggLocVolatile, IgnoreResult, IsInitializer);
  return RValue::getAggregate(AggLoc, IsAggLocVolatile);
}

/// EmitAnyExprToTemp - Similary to EmitAnyExpr(), however, the result
/// will always be accessible even if no aggregate location is
/// provided.
RValue CodeGenFunction::EmitAnyExprToTemp(const Expr *E, 
                                          bool IsAggLocVolatile,
                                          bool IsInitializer) {
  llvm::Value *AggLoc = 0;
  
  if (hasAggregateLLVMType(E->getType()) && 
      !E->getType()->isAnyComplexType())
    AggLoc = CreateTempAlloca(ConvertType(E->getType()), "agg.tmp");
  return EmitAnyExpr(E, AggLoc, IsAggLocVolatile, /*IgnoreResult=*/false, 
                     IsInitializer);
}

RValue CodeGenFunction::EmitReferenceBindingToExpr(const Expr* E,
                                                   QualType DestType,
                                                   bool IsInitializer) {
  RValue Val;
  if (E->isLvalue(getContext()) == Expr::LV_Valid) {
    // Emit the expr as an lvalue.
    LValue LV = EmitLValue(E);
    if (LV.isSimple())
      return RValue::get(LV.getAddress());
    Val = EmitLoadOfLValue(LV, E->getType());
  } else {
    // FIXME: Initializers don't work with casts yet. For example
    // const A& a = B();
    // if B inherits from A.
    Val = EmitAnyExprToTemp(E, /*IsAggLocVolatile=*/false,
                            IsInitializer);
    
    if (IsInitializer) {
      // We might have to destroy the temporary variable.
      if (const RecordType *RT = E->getType()->getAs<RecordType>()) {
        if (CXXRecordDecl *ClassDecl = dyn_cast<CXXRecordDecl>(RT->getDecl())) {
          if (!ClassDecl->hasTrivialDestructor()) {
            const CXXDestructorDecl *Dtor = 
              ClassDecl->getDestructor(getContext());
          
            CleanupScope scope(*this);
            EmitCXXDestructorCall(Dtor, Dtor_Complete, Val.getAggregateAddr());
          }
        }
      }
    }
  }

  if (Val.isAggregate()) {
    Val = RValue::get(Val.getAggregateAddr());
  } else {
    // Create a temporary variable that we can bind the reference to.
    llvm::Value *Temp = CreateTempAlloca(ConvertTypeForMem(E->getType()), 
                                         "reftmp");
    if (Val.isScalar())
      EmitStoreOfScalar(Val.getScalarVal(), Temp, false, E->getType());
    else
      StoreComplexToAddr(Val.getComplexVal(), Temp, false);
    Val = RValue::get(Temp);
  }

  return Val;
}


/// getAccessedFieldNo - Given an encoded value and a result number, return
/// the input field number being accessed.
unsigned CodeGenFunction::getAccessedFieldNo(unsigned Idx, 
                                             const llvm::Constant *Elts) {
  if (isa<llvm::ConstantAggregateZero>(Elts))
    return 0;
  
  return cast<llvm::ConstantInt>(Elts->getOperand(Idx))->getZExtValue();
}


//===----------------------------------------------------------------------===//
//                         LValue Expression Emission
//===----------------------------------------------------------------------===//

RValue CodeGenFunction::GetUndefRValue(QualType Ty) {
  if (Ty->isVoidType()) {
    return RValue::get(0);
  } else if (const ComplexType *CTy = Ty->getAsComplexType()) {
    const llvm::Type *EltTy = ConvertType(CTy->getElementType());
    llvm::Value *U = llvm::UndefValue::get(EltTy);
    return RValue::getComplex(std::make_pair(U, U));
  } else if (hasAggregateLLVMType(Ty)) {
    const llvm::Type *LTy = llvm::PointerType::getUnqual(ConvertType(Ty));
    return RValue::getAggregate(llvm::UndefValue::get(LTy));
  } else {
    return RValue::get(llvm::UndefValue::get(ConvertType(Ty)));
  }
}

RValue CodeGenFunction::EmitUnsupportedRValue(const Expr *E,
                                              const char *Name) {
  ErrorUnsupported(E, Name);
  return GetUndefRValue(E->getType());
}

LValue CodeGenFunction::EmitUnsupportedLValue(const Expr *E,
                                              const char *Name) {
  ErrorUnsupported(E, Name);
  llvm::Type *Ty = llvm::PointerType::getUnqual(ConvertType(E->getType()));
  return LValue::MakeAddr(llvm::UndefValue::get(Ty),
                          E->getType().getCVRQualifiers(),
                          getContext().getObjCGCAttrKind(E->getType()),
                          E->getType().getAddressSpace());
}

/// EmitLValue - Emit code to compute a designator that specifies the location
/// of the expression.
///
/// This can return one of two things: a simple address or a bitfield
/// reference.  In either case, the LLVM Value* in the LValue structure is
/// guaranteed to be an LLVM pointer type.
///
/// If this returns a bitfield reference, nothing about the pointee type of
/// the LLVM value is known: For example, it may not be a pointer to an
/// integer.
///
/// If this returns a normal address, and if the lvalue's C type is fixed
/// size, this method guarantees that the returned pointer type will point to
/// an LLVM type of the same size of the lvalue's type.  If the lvalue has a
/// variable length type, this is not possible.
///
LValue CodeGenFunction::EmitLValue(const Expr *E) {
  switch (E->getStmtClass()) {
  default: return EmitUnsupportedLValue(E, "l-value expression");

  case Expr::BinaryOperatorClass: 
    return EmitBinaryOperatorLValue(cast<BinaryOperator>(E));
  case Expr::CallExprClass: 
  case Expr::CXXOperatorCallExprClass:
    return EmitCallExprLValue(cast<CallExpr>(E));
  case Expr::VAArgExprClass:
    return EmitVAArgExprLValue(cast<VAArgExpr>(E));
  case Expr::DeclRefExprClass: 
  case Expr::QualifiedDeclRefExprClass:
    return EmitDeclRefLValue(cast<DeclRefExpr>(E));
  case Expr::ParenExprClass:return EmitLValue(cast<ParenExpr>(E)->getSubExpr());
  case Expr::PredefinedExprClass:
    return EmitPredefinedLValue(cast<PredefinedExpr>(E));
  case Expr::StringLiteralClass:
    return EmitStringLiteralLValue(cast<StringLiteral>(E));
  case Expr::ObjCEncodeExprClass:
    return EmitObjCEncodeExprLValue(cast<ObjCEncodeExpr>(E));

  case Expr::BlockDeclRefExprClass: 
    return EmitBlockDeclRefLValue(cast<BlockDeclRefExpr>(E));

  case Expr::CXXConditionDeclExprClass:
    return EmitCXXConditionDeclLValue(cast<CXXConditionDeclExpr>(E));
  case Expr::CXXTemporaryObjectExprClass:
  case Expr::CXXConstructExprClass:
    return EmitCXXConstructLValue(cast<CXXConstructExpr>(E));
  case Expr::CXXBindTemporaryExprClass:
    return EmitCXXBindTemporaryLValue(cast<CXXBindTemporaryExpr>(E));

  case Expr::ObjCMessageExprClass:
    return EmitObjCMessageExprLValue(cast<ObjCMessageExpr>(E));
  case Expr::ObjCIvarRefExprClass: 
    return EmitObjCIvarRefLValue(cast<ObjCIvarRefExpr>(E));
  case Expr::ObjCPropertyRefExprClass:
    return EmitObjCPropertyRefLValue(cast<ObjCPropertyRefExpr>(E));
  case Expr::ObjCImplctSetterGetterRefExprClass:
    return EmitObjCKVCRefLValue(cast<ObjCImplctSetterGetterRefExpr>(E));
  case Expr::ObjCSuperExprClass:
    return EmitObjCSuperExprLValue(cast<ObjCSuperExpr>(E));

  case Expr::StmtExprClass:
    return EmitStmtExprLValue(cast<StmtExpr>(E));
  case Expr::UnaryOperatorClass: 
    return EmitUnaryOpLValue(cast<UnaryOperator>(E));
  case Expr::ArraySubscriptExprClass:
    return EmitArraySubscriptExpr(cast<ArraySubscriptExpr>(E));
  case Expr::ExtVectorElementExprClass:
    return EmitExtVectorElementExpr(cast<ExtVectorElementExpr>(E));
  case Expr::MemberExprClass: return EmitMemberExpr(cast<MemberExpr>(E));
  case Expr::CompoundLiteralExprClass:
    return EmitCompoundLiteralLValue(cast<CompoundLiteralExpr>(E));
  case Expr::ConditionalOperatorClass:
    return EmitConditionalOperator(cast<ConditionalOperator>(E));
  case Expr::ChooseExprClass:
    return EmitLValue(cast<ChooseExpr>(E)->getChosenSubExpr(getContext()));
  case Expr::ImplicitCastExprClass:
  case Expr::CStyleCastExprClass:
  case Expr::CXXFunctionalCastExprClass:
  case Expr::CXXStaticCastExprClass:
  case Expr::CXXDynamicCastExprClass:
  case Expr::CXXReinterpretCastExprClass:
  case Expr::CXXConstCastExprClass:
    return EmitCastLValue(cast<CastExpr>(E));
  }
}

llvm::Value *CodeGenFunction::EmitLoadOfScalar(llvm::Value *Addr, bool Volatile,
                                               QualType Ty) {
  llvm::Value *V = Builder.CreateLoad(Addr, Volatile, "tmp");

  // Bool can have different representation in memory than in registers.
  if (Ty->isBooleanType())
    if (V->getType() != llvm::Type::getInt1Ty(VMContext))
      V = Builder.CreateTrunc(V, llvm::Type::getInt1Ty(VMContext), "tobool");
  
  return V;
}

void CodeGenFunction::EmitStoreOfScalar(llvm::Value *Value, llvm::Value *Addr,
                                        bool Volatile, QualType Ty) {
  
  if (Ty->isBooleanType()) {
    // Bool can have different representation in memory than in registers.
    const llvm::Type *SrcTy = Value->getType();
    const llvm::PointerType *DstPtr = cast<llvm::PointerType>(Addr->getType());
    if (DstPtr->getElementType() != SrcTy) {
      const llvm::Type *MemTy = 
        llvm::PointerType::get(SrcTy, DstPtr->getAddressSpace());
      Addr = Builder.CreateBitCast(Addr, MemTy, "storetmp");
    }
  }
  Builder.CreateStore(Value, Addr, Volatile);  
}

/// EmitLoadOfLValue - Given an expression that represents a value lvalue,
/// this method emits the address of the lvalue, then loads the result as an
/// rvalue, returning the rvalue.
RValue CodeGenFunction::EmitLoadOfLValue(LValue LV, QualType ExprType) {
  if (LV.isObjCWeak()) {
    // load of a __weak object. 
    llvm::Value *AddrWeakObj = LV.getAddress();
    llvm::Value *read_weak = CGM.getObjCRuntime().EmitObjCWeakRead(*this, 
                                                                   AddrWeakObj);
    return RValue::get(read_weak);
  }
      
  if (LV.isSimple()) {
    llvm::Value *Ptr = LV.getAddress();
    const llvm::Type *EltTy =
      cast<llvm::PointerType>(Ptr->getType())->getElementType();
    
    // Simple scalar l-value.
    if (EltTy->isSingleValueType())
      return RValue::get(EmitLoadOfScalar(Ptr, LV.isVolatileQualified(), 
                                          ExprType));
    
    assert(ExprType->isFunctionType() && "Unknown scalar value");
    return RValue::get(Ptr);
  }
  
  if (LV.isVectorElt()) {
    llvm::Value *Vec = Builder.CreateLoad(LV.getVectorAddr(),
                                          LV.isVolatileQualified(), "tmp");
    return RValue::get(Builder.CreateExtractElement(Vec, LV.getVectorIdx(),
                                                    "vecext"));
  }

  // If this is a reference to a subset of the elements of a vector, either
  // shuffle the input or extract/insert them as appropriate.
  if (LV.isExtVectorElt())
    return EmitLoadOfExtVectorElementLValue(LV, ExprType);

  if (LV.isBitfield())
    return EmitLoadOfBitfieldLValue(LV, ExprType);

  if (LV.isPropertyRef())
    return EmitLoadOfPropertyRefLValue(LV, ExprType);

  assert(LV.isKVCRef() && "Unknown LValue type!");
  return EmitLoadOfKVCRefLValue(LV, ExprType);
}

RValue CodeGenFunction::EmitLoadOfBitfieldLValue(LValue LV,
                                                 QualType ExprType) {
  unsigned StartBit = LV.getBitfieldStartBit();
  unsigned BitfieldSize = LV.getBitfieldSize();
  llvm::Value *Ptr = LV.getBitfieldAddr();

  const llvm::Type *EltTy = 
    cast<llvm::PointerType>(Ptr->getType())->getElementType();
  unsigned EltTySize = CGM.getTargetData().getTypeSizeInBits(EltTy);

  // In some cases the bitfield may straddle two memory locations.
  // Currently we load the entire bitfield, then do the magic to
  // sign-extend it if necessary. This results in somewhat more code
  // than necessary for the common case (one load), since two shifts
  // accomplish both the masking and sign extension.
  unsigned LowBits = std::min(BitfieldSize, EltTySize - StartBit);
  llvm::Value *Val = Builder.CreateLoad(Ptr, LV.isVolatileQualified(), "tmp");
  
  // Shift to proper location.
  if (StartBit)
    Val = Builder.CreateLShr(Val, llvm::ConstantInt::get(EltTy, StartBit), 
                             "bf.lo");
  
  // Mask off unused bits.
  llvm::Constant *LowMask = llvm::ConstantInt::get(VMContext, 
                                llvm::APInt::getLowBitsSet(EltTySize, LowBits));
  Val = Builder.CreateAnd(Val, LowMask, "bf.lo.cleared");
  
  // Fetch the high bits if necessary.
  if (LowBits < BitfieldSize) {
    unsigned HighBits = BitfieldSize - LowBits;
    llvm::Value *HighPtr = Builder.CreateGEP(Ptr, llvm::ConstantInt::get(
                            llvm::Type::getInt32Ty(VMContext), 1), "bf.ptr.hi");    
    llvm::Value *HighVal = Builder.CreateLoad(HighPtr, 
                                              LV.isVolatileQualified(),
                                              "tmp");
    
    // Mask off unused bits.
    llvm::Constant *HighMask = llvm::ConstantInt::get(VMContext,
                               llvm::APInt::getLowBitsSet(EltTySize, HighBits));
    HighVal = Builder.CreateAnd(HighVal, HighMask, "bf.lo.cleared");

    // Shift to proper location and or in to bitfield value.
    HighVal = Builder.CreateShl(HighVal, 
                                llvm::ConstantInt::get(EltTy, LowBits));
    Val = Builder.CreateOr(Val, HighVal, "bf.val");
  }

  // Sign extend if necessary.
  if (LV.isBitfieldSigned()) {
    llvm::Value *ExtraBits = llvm::ConstantInt::get(EltTy, 
                                                    EltTySize - BitfieldSize);
    Val = Builder.CreateAShr(Builder.CreateShl(Val, ExtraBits), 
                             ExtraBits, "bf.val.sext");
  }

  // The bitfield type and the normal type differ when the storage sizes
  // differ (currently just _Bool).
  Val = Builder.CreateIntCast(Val, ConvertType(ExprType), false, "tmp");

  return RValue::get(Val);
}

RValue CodeGenFunction::EmitLoadOfPropertyRefLValue(LValue LV,
                                                    QualType ExprType) {
  return EmitObjCPropertyGet(LV.getPropertyRefExpr());
}

RValue CodeGenFunction::EmitLoadOfKVCRefLValue(LValue LV,
                                               QualType ExprType) {
  return EmitObjCPropertyGet(LV.getKVCRefExpr());
}

// If this is a reference to a subset of the elements of a vector, create an
// appropriate shufflevector.
RValue CodeGenFunction::EmitLoadOfExtVectorElementLValue(LValue LV,
                                                         QualType ExprType) {
  llvm::Value *Vec = Builder.CreateLoad(LV.getExtVectorAddr(),
                                        LV.isVolatileQualified(), "tmp");
  
  const llvm::Constant *Elts = LV.getExtVectorElts();
  
  // If the result of the expression is a non-vector type, we must be
  // extracting a single element.  Just codegen as an extractelement.
  const VectorType *ExprVT = ExprType->getAsVectorType();
  if (!ExprVT) {
    unsigned InIdx = getAccessedFieldNo(0, Elts);
    llvm::Value *Elt = llvm::ConstantInt::get(
                                      llvm::Type::getInt32Ty(VMContext), InIdx);
    return RValue::get(Builder.CreateExtractElement(Vec, Elt, "tmp"));
  }

  // Always use shuffle vector to try to retain the original program structure
  unsigned NumResultElts = ExprVT->getNumElements();
  
  llvm::SmallVector<llvm::Constant*, 4> Mask;
  for (unsigned i = 0; i != NumResultElts; ++i) {
    unsigned InIdx = getAccessedFieldNo(i, Elts);
    Mask.push_back(llvm::ConstantInt::get(
                                     llvm::Type::getInt32Ty(VMContext), InIdx));
  }
  
  llvm::Value *MaskV = llvm::ConstantVector::get(&Mask[0], Mask.size());
  Vec = Builder.CreateShuffleVector(Vec,
                                    llvm::UndefValue::get(Vec->getType()),
                                    MaskV, "tmp");
  return RValue::get(Vec);
}



/// EmitStoreThroughLValue - Store the specified rvalue into the specified
/// lvalue, where both are guaranteed to the have the same type, and that type
/// is 'Ty'.
void CodeGenFunction::EmitStoreThroughLValue(RValue Src, LValue Dst, 
                                             QualType Ty) {
  if (!Dst.isSimple()) {
    if (Dst.isVectorElt()) {
      // Read/modify/write the vector, inserting the new element.
      llvm::Value *Vec = Builder.CreateLoad(Dst.getVectorAddr(),
                                            Dst.isVolatileQualified(), "tmp");
      Vec = Builder.CreateInsertElement(Vec, Src.getScalarVal(),
                                        Dst.getVectorIdx(), "vecins");
      Builder.CreateStore(Vec, Dst.getVectorAddr(),Dst.isVolatileQualified());
      return;
    }
  
    // If this is an update of extended vector elements, insert them as
    // appropriate.
    if (Dst.isExtVectorElt())
      return EmitStoreThroughExtVectorComponentLValue(Src, Dst, Ty);

    if (Dst.isBitfield())
      return EmitStoreThroughBitfieldLValue(Src, Dst, Ty);

    if (Dst.isPropertyRef())
      return EmitStoreThroughPropertyRefLValue(Src, Dst, Ty);

    if (Dst.isKVCRef())
      return EmitStoreThroughKVCRefLValue(Src, Dst, Ty);

    assert(0 && "Unknown LValue type");
  }
  
  if (Dst.isObjCWeak() && !Dst.isNonGC()) {
    // load of a __weak object. 
    llvm::Value *LvalueDst = Dst.getAddress();
    llvm::Value *src = Src.getScalarVal();
     CGM.getObjCRuntime().EmitObjCWeakAssign(*this, src, LvalueDst);
    return;
  }
  
  if (Dst.isObjCStrong() && !Dst.isNonGC()) {
    // load of a __strong object. 
    llvm::Value *LvalueDst = Dst.getAddress();
    llvm::Value *src = Src.getScalarVal();
#if 0
    // FIXME. We cannot positively determine if we have an 'ivar' assignment,
    // object assignment or an unknown assignment. For now, generate call to
    // objc_assign_strongCast assignment which is a safe, but consevative
    // assumption.
    if (Dst.isObjCIvar())
      CGM.getObjCRuntime().EmitObjCIvarAssign(*this, src, LvalueDst);
    else
      CGM.getObjCRuntime().EmitObjCGlobalAssign(*this, src, LvalueDst);
#endif
    if (Dst.isGlobalObjCRef())
      CGM.getObjCRuntime().EmitObjCGlobalAssign(*this, src, LvalueDst);
    else
      CGM.getObjCRuntime().EmitObjCStrongCastAssign(*this, src, LvalueDst);
    return;
  }
  
  assert(Src.isScalar() && "Can't emit an agg store with this method");
  EmitStoreOfScalar(Src.getScalarVal(), Dst.getAddress(),
                    Dst.isVolatileQualified(), Ty);
}

void CodeGenFunction::EmitStoreThroughBitfieldLValue(RValue Src, LValue Dst,
                                                     QualType Ty, 
                                                     llvm::Value **Result) {
  unsigned StartBit = Dst.getBitfieldStartBit();
  unsigned BitfieldSize = Dst.getBitfieldSize();
  llvm::Value *Ptr = Dst.getBitfieldAddr();

  const llvm::Type *EltTy = 
    cast<llvm::PointerType>(Ptr->getType())->getElementType();
  unsigned EltTySize = CGM.getTargetData().getTypeSizeInBits(EltTy);

  // Get the new value, cast to the appropriate type and masked to
  // exactly the size of the bit-field.
  llvm::Value *SrcVal = Src.getScalarVal();
  llvm::Value *NewVal = Builder.CreateIntCast(SrcVal, EltTy, false, "tmp");
  llvm::Constant *Mask = llvm::ConstantInt::get(VMContext,
                           llvm::APInt::getLowBitsSet(EltTySize, BitfieldSize));
  NewVal = Builder.CreateAnd(NewVal, Mask, "bf.value");

  // Return the new value of the bit-field, if requested.
  if (Result) {
    // Cast back to the proper type for result.
    const llvm::Type *SrcTy = SrcVal->getType();
    llvm::Value *SrcTrunc = Builder.CreateIntCast(NewVal, SrcTy, false,
                                                  "bf.reload.val");

    // Sign extend if necessary.
    if (Dst.isBitfieldSigned()) {
      unsigned SrcTySize = CGM.getTargetData().getTypeSizeInBits(SrcTy);
      llvm::Value *ExtraBits = llvm::ConstantInt::get(SrcTy,
                                                      SrcTySize - BitfieldSize);
      SrcTrunc = Builder.CreateAShr(Builder.CreateShl(SrcTrunc, ExtraBits), 
                                    ExtraBits, "bf.reload.sext");
    }

    *Result = SrcTrunc;
  }

  // In some cases the bitfield may straddle two memory locations.
  // Emit the low part first and check to see if the high needs to be
  // done.
  unsigned LowBits = std::min(BitfieldSize, EltTySize - StartBit);
  llvm::Value *LowVal = Builder.CreateLoad(Ptr, Dst.isVolatileQualified(),
                                           "bf.prev.low");

  // Compute the mask for zero-ing the low part of this bitfield.
  llvm::Constant *InvMask = 
    llvm::ConstantInt::get(VMContext,
             ~llvm::APInt::getBitsSet(EltTySize, StartBit, StartBit + LowBits));
  
  // Compute the new low part as
  //   LowVal = (LowVal & InvMask) | (NewVal << StartBit),
  // with the shift of NewVal implicitly stripping the high bits.
  llvm::Value *NewLowVal = 
    Builder.CreateShl(NewVal, llvm::ConstantInt::get(EltTy, StartBit), 
                      "bf.value.lo");  
  LowVal = Builder.CreateAnd(LowVal, InvMask, "bf.prev.lo.cleared");
  LowVal = Builder.CreateOr(LowVal, NewLowVal, "bf.new.lo");
    
  // Write back.
  Builder.CreateStore(LowVal, Ptr, Dst.isVolatileQualified());

  // If the low part doesn't cover the bitfield emit a high part.
  if (LowBits < BitfieldSize) {
    unsigned HighBits = BitfieldSize - LowBits;
    llvm::Value *HighPtr =  Builder.CreateGEP(Ptr, llvm::ConstantInt::get(
                            llvm::Type::getInt32Ty(VMContext), 1), "bf.ptr.hi");    
    llvm::Value *HighVal = Builder.CreateLoad(HighPtr, 
                                              Dst.isVolatileQualified(),
                                              "bf.prev.hi");
    
    // Compute the mask for zero-ing the high part of this bitfield.
    llvm::Constant *InvMask = 
      llvm::ConstantInt::get(VMContext, ~llvm::APInt::getLowBitsSet(EltTySize, 
                               HighBits));
  
    // Compute the new high part as
    //   HighVal = (HighVal & InvMask) | (NewVal lshr LowBits),
    // where the high bits of NewVal have already been cleared and the
    // shift stripping the low bits.
    llvm::Value *NewHighVal = 
      Builder.CreateLShr(NewVal, llvm::ConstantInt::get(EltTy, LowBits), 
                        "bf.value.high");  
    HighVal = Builder.CreateAnd(HighVal, InvMask, "bf.prev.hi.cleared");
    HighVal = Builder.CreateOr(HighVal, NewHighVal, "bf.new.hi");
    
    // Write back.
    Builder.CreateStore(HighVal, HighPtr, Dst.isVolatileQualified());
  }
}

void CodeGenFunction::EmitStoreThroughPropertyRefLValue(RValue Src,
                                                        LValue Dst,
                                                        QualType Ty) {
  EmitObjCPropertySet(Dst.getPropertyRefExpr(), Src);
}

void CodeGenFunction::EmitStoreThroughKVCRefLValue(RValue Src,
                                                   LValue Dst,
                                                   QualType Ty) {
  EmitObjCPropertySet(Dst.getKVCRefExpr(), Src);
}

void CodeGenFunction::EmitStoreThroughExtVectorComponentLValue(RValue Src,
                                                               LValue Dst,
                                                               QualType Ty) {
  // This access turns into a read/modify/write of the vector.  Load the input
  // value now.
  llvm::Value *Vec = Builder.CreateLoad(Dst.getExtVectorAddr(),
                                        Dst.isVolatileQualified(), "tmp");
  const llvm::Constant *Elts = Dst.getExtVectorElts();
  
  llvm::Value *SrcVal = Src.getScalarVal();
  
  if (const VectorType *VTy = Ty->getAsVectorType()) {
    unsigned NumSrcElts = VTy->getNumElements();
    unsigned NumDstElts =
       cast<llvm::VectorType>(Vec->getType())->getNumElements();
    if (NumDstElts == NumSrcElts) {
      // Use shuffle vector is the src and destination are the same number
      // of elements and restore the vector mask since it is on the side
      // it will be stored.
      llvm::SmallVector<llvm::Constant*, 4> Mask(NumDstElts);
      for (unsigned i = 0; i != NumSrcElts; ++i) {
        unsigned InIdx = getAccessedFieldNo(i, Elts);
        Mask[InIdx] = llvm::ConstantInt::get(
                                          llvm::Type::getInt32Ty(VMContext), i);
      }
    
      llvm::Value *MaskV = llvm::ConstantVector::get(&Mask[0], Mask.size());
      Vec = Builder.CreateShuffleVector(SrcVal,
                                        llvm::UndefValue::get(Vec->getType()),
                                        MaskV, "tmp");
    } else if (NumDstElts > NumSrcElts) {
      // Extended the source vector to the same length and then shuffle it
      // into the destination.
      // FIXME: since we're shuffling with undef, can we just use the indices
      //        into that?  This could be simpler.
      llvm::SmallVector<llvm::Constant*, 4> ExtMask;
      unsigned i;
      for (i = 0; i != NumSrcElts; ++i)
        ExtMask.push_back(llvm::ConstantInt::get(
                                         llvm::Type::getInt32Ty(VMContext), i));
      for (; i != NumDstElts; ++i)
        ExtMask.push_back(llvm::UndefValue::get(
                                            llvm::Type::getInt32Ty(VMContext)));
      llvm::Value *ExtMaskV = llvm::ConstantVector::get(&ExtMask[0],
                                                        ExtMask.size());
      llvm::Value *ExtSrcVal = 
        Builder.CreateShuffleVector(SrcVal,
                                    llvm::UndefValue::get(SrcVal->getType()),
                                    ExtMaskV, "tmp");
      // build identity
      llvm::SmallVector<llvm::Constant*, 4> Mask;
      for (unsigned i = 0; i != NumDstElts; ++i) {
        Mask.push_back(llvm::ConstantInt::get(
                                        llvm::Type::getInt32Ty(VMContext), i));
      }
      // modify when what gets shuffled in
      for (unsigned i = 0; i != NumSrcElts; ++i) {
        unsigned Idx = getAccessedFieldNo(i, Elts);
        Mask[Idx] = llvm::ConstantInt::get(
                               llvm::Type::getInt32Ty(VMContext), i+NumDstElts);
      }
      llvm::Value *MaskV = llvm::ConstantVector::get(&Mask[0], Mask.size());
      Vec = Builder.CreateShuffleVector(Vec, ExtSrcVal, MaskV, "tmp");
    } else {
      // We should never shorten the vector
      assert(0 && "unexpected shorten vector length");
    }
  } else {
    // If the Src is a scalar (not a vector) it must be updating one element.
    unsigned InIdx = getAccessedFieldNo(0, Elts);
    llvm::Value *Elt = llvm::ConstantInt::get(
                                      llvm::Type::getInt32Ty(VMContext), InIdx);
    Vec = Builder.CreateInsertElement(Vec, SrcVal, Elt, "tmp");
  }
  
  Builder.CreateStore(Vec, Dst.getExtVectorAddr(), Dst.isVolatileQualified());
}

LValue CodeGenFunction::EmitDeclRefLValue(const DeclRefExpr *E) {
  const VarDecl *VD = dyn_cast<VarDecl>(E->getDecl());
  
  if (VD && (VD->isBlockVarDecl() || isa<ParmVarDecl>(VD) ||
        isa<ImplicitParamDecl>(VD))) {
    LValue LV;
    bool NonGCable = VD->hasLocalStorage() && 
      !VD->hasAttr<BlocksAttr>();
    if (VD->hasExternalStorage()) {
      llvm::Value *V = CGM.GetAddrOfGlobalVar(VD);
      if (VD->getType()->isReferenceType())
        V = Builder.CreateLoad(V, "tmp");
      LV = LValue::MakeAddr(V, E->getType().getCVRQualifiers(),
                            getContext().getObjCGCAttrKind(E->getType()),
                            E->getType().getAddressSpace());
    } else {
      llvm::Value *V = LocalDeclMap[VD];
      assert(V && "DeclRefExpr not entered in LocalDeclMap?");
      // local variables do not get their gc attribute set.
      QualType::GCAttrTypes attr = QualType::GCNone;
      // local static?
      if (!NonGCable)
        attr = getContext().getObjCGCAttrKind(E->getType());
      if (VD->hasAttr<BlocksAttr>()) {
        bool needsCopyDispose = BlockRequiresCopying(VD->getType());
        const llvm::Type *PtrStructTy = V->getType();
        const llvm::Type *Ty = PtrStructTy;
        Ty = llvm::PointerType::get(Ty, 0);
        V = Builder.CreateStructGEP(V, 1, "forwarding");
        V = Builder.CreateBitCast(V, Ty);
        V = Builder.CreateLoad(V, false);
        V = Builder.CreateBitCast(V, PtrStructTy);
        V = Builder.CreateStructGEP(V, needsCopyDispose*2 + 4, "x");
      }
      if (VD->getType()->isReferenceType())
        V = Builder.CreateLoad(V, "tmp");
      LV = LValue::MakeAddr(V, E->getType().getCVRQualifiers(), attr,
                            E->getType().getAddressSpace());
    }
    LValue::SetObjCNonGC(LV, NonGCable);
    return LV;
  } else if (VD && VD->isFileVarDecl()) {
    llvm::Value *V = CGM.GetAddrOfGlobalVar(VD);
    if (VD->getType()->isReferenceType())
      V = Builder.CreateLoad(V, "tmp");
    LValue LV = LValue::MakeAddr(V, E->getType().getCVRQualifiers(),
                                 getContext().getObjCGCAttrKind(E->getType()),
                                 E->getType().getAddressSpace());
    if (LV.isObjCStrong())
      LV.SetGlobalObjCRef(LV, true);
    return LV;
  } else if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(E->getDecl())) {
    llvm::Value* V = CGM.GetAddrOfFunction(GlobalDecl(FD));
    if (!FD->hasPrototype()) {
      if (const FunctionProtoType *Proto =
              FD->getType()->getAsFunctionProtoType()) {
        // Ugly case: for a K&R-style definition, the type of the definition
        // isn't the same as the type of a use.  Correct for this with a
        // bitcast.
        QualType NoProtoType =
            getContext().getFunctionNoProtoType(Proto->getResultType());
        NoProtoType = getContext().getPointerType(NoProtoType);
        V = Builder.CreateBitCast(V, ConvertType(NoProtoType), "tmp");
      }
    }
    return LValue::MakeAddr(V, E->getType().getCVRQualifiers(),
                            getContext().getObjCGCAttrKind(E->getType()),
                            E->getType().getAddressSpace());
  } else if (const ImplicitParamDecl *IPD =
      dyn_cast<ImplicitParamDecl>(E->getDecl())) {
    llvm::Value *V = LocalDeclMap[IPD];
    assert(V && "BlockVarDecl not entered in LocalDeclMap?");
    return LValue::MakeAddr(V, E->getType().getCVRQualifiers(),
                            getContext().getObjCGCAttrKind(E->getType()),
                            E->getType().getAddressSpace());
  }
  assert(0 && "Unimp declref");
  //an invalid LValue, but the assert will
  //ensure that this point is never reached.
  return LValue();
}

LValue CodeGenFunction::EmitBlockDeclRefLValue(const BlockDeclRefExpr *E) {
  return LValue::MakeAddr(GetAddrOfBlockDecl(E), 
                          E->getType().getCVRQualifiers(),
                          getContext().getObjCGCAttrKind(E->getType()),
                          E->getType().getAddressSpace());
}

LValue CodeGenFunction::EmitUnaryOpLValue(const UnaryOperator *E) {
  // __extension__ doesn't affect lvalue-ness.
  if (E->getOpcode() == UnaryOperator::Extension)
    return EmitLValue(E->getSubExpr());
  
  QualType ExprTy = getContext().getCanonicalType(E->getSubExpr()->getType());
  switch (E->getOpcode()) {
  default: assert(0 && "Unknown unary operator lvalue!");
  case UnaryOperator::Deref:
    {
      QualType T = E->getSubExpr()->getType()->getPointeeType();
      assert(!T.isNull() && "CodeGenFunction::EmitUnaryOpLValue: Illegal type");
        
      LValue LV = LValue::MakeAddr(EmitScalarExpr(E->getSubExpr()),
                                   T.getCVRQualifiers(), 
                                   getContext().getObjCGCAttrKind(T),
                                   ExprTy.getAddressSpace());
     // We should not generate __weak write barrier on indirect reference
     // of a pointer to object; as in void foo (__weak id *param); *param = 0;
     // But, we continue to generate __strong write barrier on indirect write
     // into a pointer to object.
     if (getContext().getLangOptions().ObjC1 &&
         getContext().getLangOptions().getGCMode() != LangOptions::NonGC &&
         LV.isObjCWeak())
       LValue::SetObjCNonGC(LV, !E->isOBJCGCCandidate(getContext()));
     return LV;
    }
  case UnaryOperator::Real:
  case UnaryOperator::Imag:
    LValue LV = EmitLValue(E->getSubExpr());
    unsigned Idx = E->getOpcode() == UnaryOperator::Imag;
    return LValue::MakeAddr(Builder.CreateStructGEP(LV.getAddress(),
                                                    Idx, "idx"),
                            ExprTy.getCVRQualifiers(),
                            QualType::GCNone,
                            ExprTy.getAddressSpace());
  }
}

LValue CodeGenFunction::EmitStringLiteralLValue(const StringLiteral *E) {
  return LValue::MakeAddr(CGM.GetAddrOfConstantStringFromLiteral(E), 0);
}

LValue CodeGenFunction::EmitObjCEncodeExprLValue(const ObjCEncodeExpr *E) {
  return LValue::MakeAddr(CGM.GetAddrOfConstantStringFromObjCEncode(E), 0);
}


LValue CodeGenFunction::EmitPredefinedFunctionName(unsigned Type) {
  std::string GlobalVarName;

  switch (Type) {
  default:
    assert(0 && "Invalid type");
  case PredefinedExpr::Func:
    GlobalVarName = "__func__.";
    break;
  case PredefinedExpr::Function:
    GlobalVarName = "__FUNCTION__.";
    break;
  case PredefinedExpr::PrettyFunction:
    // FIXME:: Demangle C++ method names
    GlobalVarName = "__PRETTY_FUNCTION__.";
    break;
  }

  // FIXME: This isn't right at all.  The logic for computing this should go
  // into a method on PredefinedExpr.  This would allow sema and codegen to be
  // consistent for things like sizeof(__func__) etc.
  std::string FunctionName;
  if (const FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(CurCodeDecl)) {
    FunctionName = CGM.getMangledName(FD);
  } else {
    // Just get the mangled name; skipping the asm prefix if it
    // exists.
    FunctionName = CurFn->getName();
    if (FunctionName[0] == '\01')
      FunctionName = FunctionName.substr(1, std::string::npos);
  }

  GlobalVarName += FunctionName;
  llvm::Constant *C = 
    CGM.GetAddrOfConstantCString(FunctionName, GlobalVarName.c_str());
  return LValue::MakeAddr(C, 0);
}

LValue CodeGenFunction::EmitPredefinedLValue(const PredefinedExpr *E) {  
  switch (E->getIdentType()) {
  default:
    return EmitUnsupportedLValue(E, "predefined expression");
  case PredefinedExpr::Func:
  case PredefinedExpr::Function:
  case PredefinedExpr::PrettyFunction:
    return EmitPredefinedFunctionName(E->getIdentType());
  }
}

LValue CodeGenFunction::EmitArraySubscriptExpr(const ArraySubscriptExpr *E) {
  // The index must always be an integer, which is not an aggregate.  Emit it.
  llvm::Value *Idx = EmitScalarExpr(E->getIdx());
  QualType IdxTy  = E->getIdx()->getType();
  bool IdxSigned = IdxTy->isSignedIntegerType();

  // If the base is a vector type, then we are forming a vector element lvalue
  // with this subscript.
  if (E->getBase()->getType()->isVectorType()) {
    // Emit the vector as an lvalue to get its address.
    LValue LHS = EmitLValue(E->getBase());
    assert(LHS.isSimple() && "Can only subscript lvalue vectors here!");
    Idx = Builder.CreateIntCast(Idx, 
                          llvm::Type::getInt32Ty(VMContext), IdxSigned, "vidx");
    return LValue::MakeVectorElt(LHS.getAddress(), Idx,
      E->getBase()->getType().getCVRQualifiers());
  }
  
  // The base must be a pointer, which is not an aggregate.  Emit it.
  llvm::Value *Base = EmitScalarExpr(E->getBase());
  
  // Extend or truncate the index type to 32 or 64-bits.
  unsigned IdxBitwidth = cast<llvm::IntegerType>(Idx->getType())->getBitWidth();
  if (IdxBitwidth != LLVMPointerWidth)
    Idx = Builder.CreateIntCast(Idx,
                            llvm::IntegerType::get(VMContext, LLVMPointerWidth),
                                IdxSigned, "idxprom");

  // We know that the pointer points to a type of the correct size,
  // unless the size is a VLA or Objective-C interface.
  llvm::Value *Address = 0;
  if (const VariableArrayType *VAT = 
        getContext().getAsVariableArrayType(E->getType())) {
    llvm::Value *VLASize = GetVLASize(VAT);
    
    Idx = Builder.CreateMul(Idx, VLASize);
    
    QualType BaseType = getContext().getBaseElementType(VAT);
  
    uint64_t BaseTypeSize = getContext().getTypeSize(BaseType) / 8;
    Idx = Builder.CreateUDiv(Idx,
                             llvm::ConstantInt::get(Idx->getType(), 
                                                    BaseTypeSize));
    Address = Builder.CreateInBoundsGEP(Base, Idx, "arrayidx");
  } else if (const ObjCInterfaceType *OIT = 
             dyn_cast<ObjCInterfaceType>(E->getType())) {
    llvm::Value *InterfaceSize = 
      llvm::ConstantInt::get(Idx->getType(),
                             getContext().getTypeSize(OIT) / 8);
    
    Idx = Builder.CreateMul(Idx, InterfaceSize);

    llvm::Type *i8PTy =
            llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(VMContext));
    Address = Builder.CreateGEP(Builder.CreateBitCast(Base, i8PTy),
                                Idx, "arrayidx");
    Address = Builder.CreateBitCast(Address, Base->getType());
  } else {
    Address = Builder.CreateInBoundsGEP(Base, Idx, "arrayidx");
  }
  
  QualType T = E->getBase()->getType()->getPointeeType();
  assert(!T.isNull() && 
         "CodeGenFunction::EmitArraySubscriptExpr(): Illegal base type");
    
  LValue LV = LValue::MakeAddr(Address,
                               T.getCVRQualifiers(),
                               getContext().getObjCGCAttrKind(T),
                               E->getBase()->getType().getAddressSpace());
  if (getContext().getLangOptions().ObjC1 &&
      getContext().getLangOptions().getGCMode() != LangOptions::NonGC)
    LValue::SetObjCNonGC(LV, !E->isOBJCGCCandidate(getContext()));
  return LV;
}

static 
llvm::Constant *GenerateConstantVector(llvm::LLVMContext &VMContext,
                                       llvm::SmallVector<unsigned, 4> &Elts) {
  llvm::SmallVector<llvm::Constant *, 4> CElts;
  
  for (unsigned i = 0, e = Elts.size(); i != e; ++i)
    CElts.push_back(llvm::ConstantInt::get(
                                   llvm::Type::getInt32Ty(VMContext), Elts[i]));

  return llvm::ConstantVector::get(&CElts[0], CElts.size());
}

LValue CodeGenFunction::
EmitExtVectorElementExpr(const ExtVectorElementExpr *E) {
  // Emit the base vector as an l-value.
  LValue Base;

  // ExtVectorElementExpr's base can either be a vector or pointer to vector.
  if (!E->isArrow()) {
    assert(E->getBase()->getType()->isVectorType());
    Base = EmitLValue(E->getBase());
  } else {
    const PointerType *PT = E->getBase()->getType()->getAs<PointerType>();
    llvm::Value *Ptr = EmitScalarExpr(E->getBase());
    Base = LValue::MakeAddr(Ptr, PT->getPointeeType().getCVRQualifiers(),
                            QualType::GCNone,
                            PT->getPointeeType().getAddressSpace());
  }

  // Encode the element access list into a vector of unsigned indices.
  llvm::SmallVector<unsigned, 4> Indices;
  E->getEncodedElementAccess(Indices);

  if (Base.isSimple()) {
    llvm::Constant *CV = GenerateConstantVector(VMContext, Indices);
    return LValue::MakeExtVectorElt(Base.getAddress(), CV,
                                    Base.getQualifiers());
  }
  assert(Base.isExtVectorElt() && "Can only subscript lvalue vec elts here!");

  llvm::Constant *BaseElts = Base.getExtVectorElts();
  llvm::SmallVector<llvm::Constant *, 4> CElts;

  for (unsigned i = 0, e = Indices.size(); i != e; ++i) {
    if (isa<llvm::ConstantAggregateZero>(BaseElts))
      CElts.push_back(llvm::ConstantInt::get(
                                         llvm::Type::getInt32Ty(VMContext), 0));
    else
      CElts.push_back(BaseElts->getOperand(Indices[i]));
  }
  llvm::Constant *CV = llvm::ConstantVector::get(&CElts[0], CElts.size());
  return LValue::MakeExtVectorElt(Base.getExtVectorAddr(), CV,
                                  Base.getQualifiers());
}

LValue CodeGenFunction::EmitMemberExpr(const MemberExpr *E) {
  bool isUnion = false;
  bool isIvar = false;
  bool isNonGC = false;
  Expr *BaseExpr = E->getBase();
  llvm::Value *BaseValue = NULL;
  unsigned CVRQualifiers=0;

  // If this is s.x, emit s as an lvalue.  If it is s->x, emit s as a scalar.
  if (E->isArrow()) {
    BaseValue = EmitScalarExpr(BaseExpr);
    const PointerType *PTy = 
      BaseExpr->getType()->getAs<PointerType>();
    if (PTy->getPointeeType()->isUnionType())
      isUnion = true;
    CVRQualifiers = PTy->getPointeeType().getCVRQualifiers();
  } else if (isa<ObjCPropertyRefExpr>(BaseExpr) ||
             isa<ObjCImplctSetterGetterRefExpr>(BaseExpr)) {
    RValue RV = EmitObjCPropertyGet(BaseExpr);
    BaseValue = RV.getAggregateAddr();
    if (BaseExpr->getType()->isUnionType())
      isUnion = true;
    CVRQualifiers = BaseExpr->getType().getCVRQualifiers();
  } else {
    LValue BaseLV = EmitLValue(BaseExpr);
    if (BaseLV.isObjCIvar())
      isIvar = true;
    if (BaseLV.isNonGC())
      isNonGC = true;
    // FIXME: this isn't right for bitfields.
    BaseValue = BaseLV.getAddress();
    QualType BaseTy = BaseExpr->getType();
    if (BaseTy->isUnionType())
      isUnion = true;
    CVRQualifiers = BaseTy.getCVRQualifiers();
  }

  FieldDecl *Field = dyn_cast<FieldDecl>(E->getMemberDecl());
  // FIXME: Handle non-field member expressions
  assert(Field && "No code generation for non-field member references");
  LValue MemExpLV = EmitLValueForField(BaseValue, Field, isUnion,
                                       CVRQualifiers);
  LValue::SetObjCIvar(MemExpLV, isIvar);
  LValue::SetObjCNonGC(MemExpLV, isNonGC);
  return MemExpLV;
}

LValue CodeGenFunction::EmitLValueForBitfield(llvm::Value* BaseValue,
                                              FieldDecl* Field,
                                              unsigned CVRQualifiers) {
  CodeGenTypes::BitFieldInfo Info = CGM.getTypes().getBitFieldInfo(Field);

  // FIXME: CodeGenTypes should expose a method to get the appropriate type for
  // FieldTy (the appropriate type is ABI-dependent).
  const llvm::Type *FieldTy = 
    CGM.getTypes().ConvertTypeForMem(Field->getType());
  const llvm::PointerType *BaseTy =
  cast<llvm::PointerType>(BaseValue->getType());
  unsigned AS = BaseTy->getAddressSpace();
  BaseValue = Builder.CreateBitCast(BaseValue,
                                    llvm::PointerType::get(FieldTy, AS),
                                    "tmp");
  
  llvm::Value *Idx = 
    llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), Info.FieldNo);
  llvm::Value *V = Builder.CreateGEP(BaseValue, Idx, "tmp");
  
  return LValue::MakeBitfield(V, Info.Start, Info.Size,
                              Field->getType()->isSignedIntegerType(),
                            Field->getType().getCVRQualifiers()|CVRQualifiers);
}

LValue CodeGenFunction::EmitLValueForField(llvm::Value* BaseValue,
                                           FieldDecl* Field,
                                           bool isUnion,
                                           unsigned CVRQualifiers)
{
  if (Field->isBitField())
    return EmitLValueForBitfield(BaseValue, Field, CVRQualifiers);
  
  unsigned idx = CGM.getTypes().getLLVMFieldNo(Field);
  llvm::Value *V = Builder.CreateStructGEP(BaseValue, idx, "tmp");

  // Match union field type.
  if (isUnion) {
    const llvm::Type *FieldTy = 
      CGM.getTypes().ConvertTypeForMem(Field->getType());
    const llvm::PointerType * BaseTy = 
      cast<llvm::PointerType>(BaseValue->getType());
    unsigned AS = BaseTy->getAddressSpace();
    V = Builder.CreateBitCast(V, 
                              llvm::PointerType::get(FieldTy, AS), 
                              "tmp");
  }
  if (Field->getType()->isReferenceType())
    V = Builder.CreateLoad(V, "tmp");

  QualType::GCAttrTypes attr = QualType::GCNone;
  if (CGM.getLangOptions().ObjC1 &&
      CGM.getLangOptions().getGCMode() != LangOptions::NonGC) {
    QualType Ty = Field->getType();
    attr = Ty.getObjCGCAttr();
    if (attr != QualType::GCNone) {
      // __weak attribute on a field is ignored.
      if (attr == QualType::Weak)
        attr = QualType::GCNone;
    } else if (Ty->isObjCObjectPointerType())
      attr = QualType::Strong;
  }
  LValue LV =  
    LValue::MakeAddr(V, 
                     Field->getType().getCVRQualifiers()|CVRQualifiers,
                     attr,
                     Field->getType().getAddressSpace());
  return LV;
}

LValue CodeGenFunction::EmitCompoundLiteralLValue(const CompoundLiteralExpr* E){
  const llvm::Type *LTy = ConvertType(E->getType());
  llvm::Value *DeclPtr = CreateTempAlloca(LTy, ".compoundliteral");

  const Expr* InitExpr = E->getInitializer();
  LValue Result = LValue::MakeAddr(DeclPtr, E->getType().getCVRQualifiers(),
                                   QualType::GCNone,
                                   E->getType().getAddressSpace());

  if (E->getType()->isComplexType()) {
    EmitComplexExprIntoAddr(InitExpr, DeclPtr, false);
  } else if (hasAggregateLLVMType(E->getType())) {
    EmitAnyExpr(InitExpr, DeclPtr, false);
  } else {
    EmitStoreThroughLValue(EmitAnyExpr(InitExpr), Result, E->getType());
  }

  return Result;
}

LValue CodeGenFunction::EmitConditionalOperator(const ConditionalOperator* E) {
  if (E->isLvalue(getContext()) == Expr::LV_Valid)
    return EmitUnsupportedLValue(E, "conditional operator");

  // ?: here should be an aggregate.
  assert((hasAggregateLLVMType(E->getType()) && 
          !E->getType()->isAnyComplexType()) &&
         "Unexpected conditional operator!");

  llvm::Value *Temp = CreateTempAlloca(ConvertType(E->getType()));
  EmitAggExpr(E, Temp, false);

  return LValue::MakeAddr(Temp, E->getType().getCVRQualifiers(),
                          getContext().getObjCGCAttrKind(E->getType()),
                          E->getType().getAddressSpace());
 
}

/// EmitCastLValue - Casts are never lvalues.  If a cast is needed by the code
/// generator in an lvalue context, then it must mean that we need the address
/// of an aggregate in order to access one of its fields.  This can happen for
/// all the reasons that casts are permitted with aggregate result, including
/// noop aggregate casts, and cast from scalar to union.
LValue CodeGenFunction::EmitCastLValue(const CastExpr *E) {
  // If this is an aggregate-to-aggregate cast, just use the input's address as
  // the lvalue.
  if (getContext().hasSameUnqualifiedType(E->getType(),
                                          E->getSubExpr()->getType()))
    return EmitLValue(E->getSubExpr());

  // Otherwise, we must have a cast from scalar to union.
  assert(E->getType()->isUnionType() && "Expected scalar-to-union cast");
  
  // Casts are only lvalues when the source and destination types are the same.
  llvm::Value *Temp = CreateTempAlloca(ConvertType(E->getType()));
  EmitAnyExpr(E->getSubExpr(), Temp, false);
  
  return LValue::MakeAddr(Temp, E->getType().getCVRQualifiers(),
                          getContext().getObjCGCAttrKind(E->getType()),
                          E->getType().getAddressSpace());
}

//===--------------------------------------------------------------------===//
//                             Expression Emission
//===--------------------------------------------------------------------===//


RValue CodeGenFunction::EmitCallExpr(const CallExpr *E) {
  // Builtins never have block type.
  if (E->getCallee()->getType()->isBlockPointerType())
    return EmitBlockCallExpr(E);

  if (const CXXMemberCallExpr *CE = dyn_cast<CXXMemberCallExpr>(E))
    return EmitCXXMemberCallExpr(CE);
  
  const Decl *TargetDecl = 0;
  if (const ImplicitCastExpr *CE = dyn_cast<ImplicitCastExpr>(E->getCallee())) {
    if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(CE->getSubExpr())) {
      TargetDecl = DRE->getDecl();
      if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(TargetDecl))
        if (unsigned builtinID = FD->getBuiltinID(getContext()))
          return EmitBuiltinExpr(FD, builtinID, E);
    }
  }

  if (const CXXOperatorCallExpr *CE = dyn_cast<CXXOperatorCallExpr>(E))
    if (const CXXMethodDecl *MD = dyn_cast_or_null<CXXMethodDecl>(TargetDecl))
      return EmitCXXOperatorMemberCallExpr(CE, MD);
      
  llvm::Value *Callee = EmitScalarExpr(E->getCallee());
  return EmitCall(Callee, E->getCallee()->getType(),
                  E->arg_begin(), E->arg_end(), TargetDecl);
}

LValue CodeGenFunction::EmitBinaryOperatorLValue(const BinaryOperator *E) {
  // Comma expressions just emit their LHS then their RHS as an l-value.
  if (E->getOpcode() == BinaryOperator::Comma) {
    EmitAnyExpr(E->getLHS());
    return EmitLValue(E->getRHS());
  }
  
  // Can only get l-value for binary operator expressions which are a
  // simple assignment of aggregate type.
  if (E->getOpcode() != BinaryOperator::Assign)
    return EmitUnsupportedLValue(E, "binary l-value expression");

  llvm::Value *Temp = CreateTempAlloca(ConvertType(E->getType()));
  EmitAggExpr(E, Temp, false);
  // FIXME: Are these qualifiers correct?
  return LValue::MakeAddr(Temp, E->getType().getCVRQualifiers(),
                          getContext().getObjCGCAttrKind(E->getType()),
                          E->getType().getAddressSpace());
}

LValue CodeGenFunction::EmitCallExprLValue(const CallExpr *E) {
  RValue RV = EmitCallExpr(E);

  if (RV.isScalar()) {
    assert(E->getCallReturnType()->isReferenceType() &&
           "Can't have a scalar return unless the return type is a "
           "reference type!");
    
    return LValue::MakeAddr(RV.getScalarVal(), E->getType().getCVRQualifiers(), 
                            getContext().getObjCGCAttrKind(E->getType()),
                            E->getType().getAddressSpace());
  }
  
  return LValue::MakeAddr(RV.getAggregateAddr(),
                          E->getType().getCVRQualifiers(),
                          getContext().getObjCGCAttrKind(E->getType()),
                          E->getType().getAddressSpace());
}

LValue CodeGenFunction::EmitVAArgExprLValue(const VAArgExpr *E) {
  // FIXME: This shouldn't require another copy.
  llvm::Value *Temp = CreateTempAlloca(ConvertType(E->getType()));
  EmitAggExpr(E, Temp, false);
  return LValue::MakeAddr(Temp, E->getType().getCVRQualifiers(),
                          QualType::GCNone, E->getType().getAddressSpace());
}

LValue
CodeGenFunction::EmitCXXConditionDeclLValue(const CXXConditionDeclExpr *E) {
  EmitLocalBlockVarDecl(*E->getVarDecl());
  return EmitDeclRefLValue(E);
}

LValue CodeGenFunction::EmitCXXConstructLValue(const CXXConstructExpr *E) {
  llvm::Value *Temp = CreateTempAlloca(ConvertTypeForMem(E->getType()), "tmp");
  EmitCXXConstructExpr(Temp, E);
  return LValue::MakeAddr(Temp, E->getType().getCVRQualifiers(),
                          QualType::GCNone, E->getType().getAddressSpace());
}

LValue
CodeGenFunction::EmitCXXBindTemporaryLValue(const CXXBindTemporaryExpr *E) {
  LValue LV = EmitLValue(E->getSubExpr());
  
  PushCXXTemporary(E->getTemporary(), LV.getAddress());
  
  return LV;
}

LValue CodeGenFunction::EmitObjCMessageExprLValue(const ObjCMessageExpr *E) {
  // Can only get l-value for message expression returning aggregate type
  RValue RV = EmitObjCMessageExpr(E);
  // FIXME: can this be volatile?
  return LValue::MakeAddr(RV.getAggregateAddr(),
                          E->getType().getCVRQualifiers(),
                          getContext().getObjCGCAttrKind(E->getType()),
                          E->getType().getAddressSpace());
}

llvm::Value *CodeGenFunction::EmitIvarOffset(const ObjCInterfaceDecl *Interface,
                                             const ObjCIvarDecl *Ivar) {
  return CGM.getObjCRuntime().EmitIvarOffset(*this, Interface, Ivar);
}

LValue CodeGenFunction::EmitLValueForIvar(QualType ObjectTy,
                                          llvm::Value *BaseValue,
                                          const ObjCIvarDecl *Ivar,
                                          unsigned CVRQualifiers) {
  return CGM.getObjCRuntime().EmitObjCValueForIvar(*this, ObjectTy, BaseValue,
                                                   Ivar, CVRQualifiers);
}

LValue CodeGenFunction::EmitObjCIvarRefLValue(const ObjCIvarRefExpr *E) {
  // FIXME: A lot of the code below could be shared with EmitMemberExpr.
  llvm::Value *BaseValue = 0;
  const Expr *BaseExpr = E->getBase();
  unsigned CVRQualifiers = 0;
  QualType ObjectTy;
  if (E->isArrow()) {
    BaseValue = EmitScalarExpr(BaseExpr);
    ObjectTy = BaseExpr->getType()->getPointeeType();
    CVRQualifiers = ObjectTy.getCVRQualifiers();
  } else {
    LValue BaseLV = EmitLValue(BaseExpr);
    // FIXME: this isn't right for bitfields.
    BaseValue = BaseLV.getAddress();
    ObjectTy = BaseExpr->getType();
    CVRQualifiers = ObjectTy.getCVRQualifiers();
  }

  return EmitLValueForIvar(ObjectTy, BaseValue, E->getDecl(), CVRQualifiers);
}

LValue 
CodeGenFunction::EmitObjCPropertyRefLValue(const ObjCPropertyRefExpr *E) {
  // This is a special l-value that just issues sends when we load or
  // store through it.
  return LValue::MakePropertyRef(E, E->getType().getCVRQualifiers());
}

LValue 
CodeGenFunction::EmitObjCKVCRefLValue(
                                const ObjCImplctSetterGetterRefExpr *E) {
  // This is a special l-value that just issues sends when we load or
  // store through it.
  return LValue::MakeKVCRef(E, E->getType().getCVRQualifiers());
}

LValue
CodeGenFunction::EmitObjCSuperExprLValue(const ObjCSuperExpr *E) {
  return EmitUnsupportedLValue(E, "use of super");
}

LValue CodeGenFunction::EmitStmtExprLValue(const StmtExpr *E) {
  
  // Can only get l-value for message expression returning aggregate type
  RValue RV = EmitAnyExprToTemp(E);
  // FIXME: can this be volatile?
  return LValue::MakeAddr(RV.getAggregateAddr(),
                          E->getType().getCVRQualifiers(),
                          getContext().getObjCGCAttrKind(E->getType()),
                          E->getType().getAddressSpace());
}


RValue CodeGenFunction::EmitCall(llvm::Value *Callee, QualType CalleeType, 
                                 CallExpr::const_arg_iterator ArgBeg,
                                 CallExpr::const_arg_iterator ArgEnd,
                                 const Decl *TargetDecl) {
  // Get the actual function type. The callee type will always be a
  // pointer to function type or a block pointer type.
  assert(CalleeType->isFunctionPointerType() && 
         "Call must have function pointer type!");

  QualType FnType = CalleeType->getAs<PointerType>()->getPointeeType();
  QualType ResultType = FnType->getAsFunctionType()->getResultType();

  CallArgList Args;
  EmitCallArgs(Args, FnType->getAsFunctionProtoType(), ArgBeg, ArgEnd);

  return EmitCall(CGM.getTypes().getFunctionInfo(ResultType, Args), 
                  Callee, Args, TargetDecl);
}
