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
                                                    const llvm::Twine &Name) {
  if (!Builder.isNamePreserving())
    return new llvm::AllocaInst(Ty, 0, "", AllocaInsertPt);
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
/// aggregate expression, the aggloc/agglocvolatile arguments indicate where the
/// result should be returned.
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

/// EmitAnyExprToTemp - Similary to EmitAnyExpr(), however, the result will
/// always be accessible even if no aggregate location is provided.
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
  bool ShouldDestroyTemporaries = false;
  unsigned OldNumLiveTemporaries = 0;
  
  if (const CXXExprWithTemporaries *TE = dyn_cast<CXXExprWithTemporaries>(E)) {
    ShouldDestroyTemporaries = TE->shouldDestroyTemporaries();

    // Keep track of the current cleanup stack depth.
    if (ShouldDestroyTemporaries)
      OldNumLiveTemporaries = LiveTemporaries.size();
    
    E = TE->getSubExpr();
  }
  
  RValue Val;
  if (E->isLvalue(getContext()) == Expr::LV_Valid) {
    // Emit the expr as an lvalue.
    LValue LV = EmitLValue(E);
    if (LV.isSimple())
      return RValue::get(LV.getAddress());
    Val = EmitLoadOfLValue(LV, E->getType());
    
    if (ShouldDestroyTemporaries) {
      // Pop temporaries.
      while (LiveTemporaries.size() > OldNumLiveTemporaries)
        PopCXXTemporary();
    }      
  } else {
    const CXXRecordDecl *BaseClassDecl = 0;
    const CXXRecordDecl *DerivedClassDecl = 0;
    
    if (const CastExpr *CE = 
          dyn_cast<CastExpr>(E->IgnoreParenNoopCasts(getContext()))) {
      if (CE->getCastKind() == CastExpr::CK_DerivedToBase) {
        E = CE->getSubExpr();
        
        BaseClassDecl = 
          cast<CXXRecordDecl>(CE->getType()->getAs<RecordType>()->getDecl());
        DerivedClassDecl = 
          cast<CXXRecordDecl>(E->getType()->getAs<RecordType>()->getDecl());
      }
    }
      
    Val = EmitAnyExprToTemp(E, /*IsAggLocVolatile=*/false,
                            IsInitializer);

    if (ShouldDestroyTemporaries) {
      // Pop temporaries.
      while (LiveTemporaries.size() > OldNumLiveTemporaries)
        PopCXXTemporary();
    }      
    
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
    
    // Check if need to perform the derived-to-base cast.
    if (BaseClassDecl) {
      llvm::Value *Derived = Val.getAggregateAddr();
      llvm::Value *Base = 
        GetAddressCXXOfBaseClass(Derived, DerivedClassDecl, BaseClassDecl, 
                                 /*NullCheckValue=*/false);
      return RValue::get(Base);
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


/// getAccessedFieldNo - Given an encoded value and a result number, return the
/// input field number being accessed.
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
  if (Ty->isVoidType())
    return RValue::get(0);
  
  if (const ComplexType *CTy = Ty->getAs<ComplexType>()) {
    const llvm::Type *EltTy = ConvertType(CTy->getElementType());
    llvm::Value *U = llvm::UndefValue::get(EltTy);
    return RValue::getComplex(std::make_pair(U, U));
  }
  
  if (hasAggregateLLVMType(Ty)) {
    const llvm::Type *LTy = llvm::PointerType::getUnqual(ConvertType(Ty));
    return RValue::getAggregate(llvm::UndefValue::get(LTy));
  }
  
  return RValue::get(llvm::UndefValue::get(ConvertType(Ty)));
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
                          MakeQualifiers(E->getType()));
}

/// EmitLValue - Emit code to compute a designator that specifies the location
/// of the expression.
///
/// This can return one of two things: a simple address or a bitfield reference.
/// In either case, the LLVM Value* in the LValue structure is guaranteed to be
/// an LLVM pointer type.
///
/// If this returns a bitfield reference, nothing about the pointee type of the
/// LLVM value is known: For example, it may not be a pointer to an integer.
///
/// If this returns a normal address, and if the lvalue's C type is fixed size,
/// this method guarantees that the returned pointer type will point to an LLVM
/// type of the same size of the lvalue's type.  If the lvalue has a variable
/// length type, this is not possible.
///
LValue CodeGenFunction::EmitLValue(const Expr *E) {
  switch (E->getStmtClass()) {
  default: return EmitUnsupportedLValue(E, "l-value expression");

  case Expr::BinaryOperatorClass:
    return EmitBinaryOperatorLValue(cast<BinaryOperator>(E));
  case Expr::CallExprClass:
  case Expr::CXXMemberCallExprClass:
  case Expr::CXXOperatorCallExprClass:
    return EmitCallExprLValue(cast<CallExpr>(E));
  case Expr::VAArgExprClass:
    return EmitVAArgExprLValue(cast<VAArgExpr>(E));
  case Expr::DeclRefExprClass:
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
  case Expr::CXXExprWithTemporariesClass:
    return EmitCXXExprWithTemporariesLValue(cast<CXXExprWithTemporaries>(E));
  case Expr::CXXZeroInitValueExprClass:
    return EmitNullInitializationLValue(cast<CXXZeroInitValueExpr>(E));
  case Expr::CXXDefaultArgExprClass:
    return EmitLValue(cast<CXXDefaultArgExpr>(E)->getExpr());
  case Expr::CXXTypeidExprClass:
    return EmitCXXTypeidLValue(cast<CXXTypeidExpr>(E));

  case Expr::ObjCMessageExprClass:
    return EmitObjCMessageExprLValue(cast<ObjCMessageExpr>(E));
  case Expr::ObjCIvarRefExprClass:
    return EmitObjCIvarRefLValue(cast<ObjCIvarRefExpr>(E));
  case Expr::ObjCPropertyRefExprClass:
    return EmitObjCPropertyRefLValue(cast<ObjCPropertyRefExpr>(E));
  case Expr::ObjCImplicitSetterGetterRefExprClass:
    return EmitObjCKVCRefLValue(cast<ObjCImplicitSetterGetterRefExpr>(E));
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
  case Expr::MemberExprClass:
    return EmitMemberExpr(cast<MemberExpr>(E));
  case Expr::CompoundLiteralExprClass:
    return EmitCompoundLiteralLValue(cast<CompoundLiteralExpr>(E));
  case Expr::ConditionalOperatorClass:
    return EmitConditionalOperatorLValue(cast<ConditionalOperator>(E));
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

/// EmitLoadOfLValue - Given an expression that represents a value lvalue, this
/// method emits the address of the lvalue, then loads the result as an rvalue,
/// returning the rvalue.
RValue CodeGenFunction::EmitLoadOfLValue(LValue LV, QualType ExprType) {
  if (LV.isObjCWeak()) {
    // load of a __weak object.
    llvm::Value *AddrWeakObj = LV.getAddress();
    return RValue::get(CGM.getObjCRuntime().EmitObjCWeakRead(*this,
                                                             AddrWeakObj));
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

  // In some cases the bitfield may straddle two memory locations.  Currently we
  // load the entire bitfield, then do the magic to sign-extend it if
  // necessary. This results in somewhat more code than necessary for the common
  // case (one load), since two shifts accomplish both the masking and sign
  // extension.
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

  // The bitfield type and the normal type differ when the storage sizes differ
  // (currently just _Bool).
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

  // If the result of the expression is a non-vector type, we must be extracting
  // a single element.  Just codegen as an extractelement.
  const VectorType *ExprVT = ExprType->getAs<VectorType>();
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

    assert(Dst.isKVCRef() && "Unknown LValue type");
    return EmitStoreThroughKVCRefLValue(Src, Dst, Ty);
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
    if (Dst.isObjCIvar()) {
      assert(Dst.getBaseIvarExp() && "BaseIvarExp is NULL");
      const llvm::Type *ResultType = ConvertType(getContext().LongTy);
      llvm::Value *RHS = EmitScalarExpr(Dst.getBaseIvarExp());
      llvm::Value *dst = RHS;
      RHS = Builder.CreatePtrToInt(RHS, ResultType, "sub.ptr.rhs.cast");
      llvm::Value *LHS = 
        Builder.CreatePtrToInt(LvalueDst, ResultType, "sub.ptr.lhs.cast");
      llvm::Value *BytesBetween = Builder.CreateSub(LHS, RHS, "ivar.offset");
      CGM.getObjCRuntime().EmitObjCIvarAssign(*this, src, dst,
                                              BytesBetween);
    } else if (Dst.isGlobalObjCRef())
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

  // Get the new value, cast to the appropriate type and masked to exactly the
  // size of the bit-field.
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

  // In some cases the bitfield may straddle two memory locations.  Emit the low
  // part first and check to see if the high needs to be done.
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

  if (const VectorType *VTy = Ty->getAs<VectorType>()) {
    unsigned NumSrcElts = VTy->getNumElements();
    unsigned NumDstElts =
       cast<llvm::VectorType>(Vec->getType())->getNumElements();
    if (NumDstElts == NumSrcElts) {
      // Use shuffle vector is the src and destination are the same number of
      // elements and restore the vector mask since it is on the side it will be
      // stored.
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
      const llvm::Type *Int32Ty = llvm::Type::getInt32Ty(VMContext);
      unsigned i;
      for (i = 0; i != NumSrcElts; ++i)
        ExtMask.push_back(llvm::ConstantInt::get(Int32Ty, i));
      for (; i != NumDstElts; ++i)
        ExtMask.push_back(llvm::UndefValue::get(Int32Ty));
      llvm::Value *ExtMaskV = llvm::ConstantVector::get(&ExtMask[0],
                                                        ExtMask.size());
      llvm::Value *ExtSrcVal =
        Builder.CreateShuffleVector(SrcVal,
                                    llvm::UndefValue::get(SrcVal->getType()),
                                    ExtMaskV, "tmp");
      // build identity
      llvm::SmallVector<llvm::Constant*, 4> Mask;
      for (unsigned i = 0; i != NumDstElts; ++i)
        Mask.push_back(llvm::ConstantInt::get(Int32Ty, i));

      // modify when what gets shuffled in
      for (unsigned i = 0; i != NumSrcElts; ++i) {
        unsigned Idx = getAccessedFieldNo(i, Elts);
        Mask[Idx] = llvm::ConstantInt::get(Int32Ty, i+NumDstElts);
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
    const llvm::Type *Int32Ty = llvm::Type::getInt32Ty(VMContext);
    llvm::Value *Elt = llvm::ConstantInt::get(Int32Ty, InIdx);
    Vec = Builder.CreateInsertElement(Vec, SrcVal, Elt, "tmp");
  }

  Builder.CreateStore(Vec, Dst.getExtVectorAddr(), Dst.isVolatileQualified());
}

// setObjCGCLValueClass - sets class of he lvalue for the purpose of
// generating write-barries API. It is currently a global, ivar,
// or neither.
static void setObjCGCLValueClass(const ASTContext &Ctx, const Expr *E,
                                 LValue &LV) {
  if (Ctx.getLangOptions().getGCMode() == LangOptions::NonGC)
    return;
  
  if (isa<ObjCIvarRefExpr>(E)) {
    LV.SetObjCIvar(LV, true);
    ObjCIvarRefExpr *Exp = cast<ObjCIvarRefExpr>(const_cast<Expr*>(E));
    LV.setBaseIvarExp(Exp->getBase());
    LV.SetObjCArray(LV, E->getType()->isArrayType());
    return;
  }
  
  if (const DeclRefExpr *Exp = dyn_cast<DeclRefExpr>(E)) {
    if (const VarDecl *VD = dyn_cast<VarDecl>(Exp->getDecl())) {
      if ((VD->isBlockVarDecl() && !VD->hasLocalStorage()) ||
          VD->isFileVarDecl())
        LV.SetGlobalObjCRef(LV, true);
    }
    LV.SetObjCArray(LV, E->getType()->isArrayType());
    return;
  }
  
  if (const UnaryOperator *Exp = dyn_cast<UnaryOperator>(E)) {
    setObjCGCLValueClass(Ctx, Exp->getSubExpr(), LV);
    return;
  }
  
  if (const ParenExpr *Exp = dyn_cast<ParenExpr>(E)) {
    setObjCGCLValueClass(Ctx, Exp->getSubExpr(), LV);
    if (LV.isObjCIvar()) {
      // If cast is to a structure pointer, follow gcc's behavior and make it
      // a non-ivar write-barrier.
      QualType ExpTy = E->getType();
      if (ExpTy->isPointerType())
        ExpTy = ExpTy->getAs<PointerType>()->getPointeeType();
      if (ExpTy->isRecordType())
        LV.SetObjCIvar(LV, false); 
    }
    return;
  }
  if (const ImplicitCastExpr *Exp = dyn_cast<ImplicitCastExpr>(E)) {
    setObjCGCLValueClass(Ctx, Exp->getSubExpr(), LV);
    return;
  }
  
  if (const CStyleCastExpr *Exp = dyn_cast<CStyleCastExpr>(E)) {
    setObjCGCLValueClass(Ctx, Exp->getSubExpr(), LV);
    return;
  }
  
  if (const ArraySubscriptExpr *Exp = dyn_cast<ArraySubscriptExpr>(E)) {
    setObjCGCLValueClass(Ctx, Exp->getBase(), LV);
    if (LV.isObjCIvar() && !LV.isObjCArray()) 
      // Using array syntax to assigning to what an ivar points to is not 
      // same as assigning to the ivar itself. {id *Names;} Names[i] = 0;
      LV.SetObjCIvar(LV, false); 
    else if (LV.isGlobalObjCRef() && !LV.isObjCArray())
      // Using array syntax to assigning to what global points to is not 
      // same as assigning to the global itself. {id *G;} G[i] = 0;
      LV.SetGlobalObjCRef(LV, false);
    return;
  }
  
  if (const MemberExpr *Exp = dyn_cast<MemberExpr>(E)) {
    setObjCGCLValueClass(Ctx, Exp->getBase(), LV);
    // We don't know if member is an 'ivar', but this flag is looked at
    // only in the context of LV.isObjCIvar().
    LV.SetObjCArray(LV, E->getType()->isArrayType());
    return;
  }
}

static LValue EmitGlobalVarDeclLValue(CodeGenFunction &CGF,
                                      const Expr *E, const VarDecl *VD) {
  assert((VD->hasExternalStorage() || VD->isFileVarDecl()) &&
         "Var decl must have external storage or be a file var decl!");

  llvm::Value *V = CGF.CGM.GetAddrOfGlobalVar(VD);
  if (VD->getType()->isReferenceType())
    V = CGF.Builder.CreateLoad(V, "tmp");
  LValue LV = LValue::MakeAddr(V, CGF.MakeQualifiers(E->getType()));
  setObjCGCLValueClass(CGF.getContext(), E, LV);
  return LV;
}

LValue CodeGenFunction::EmitDeclRefLValue(const DeclRefExpr *E) {
  const NamedDecl *ND = E->getDecl();

  if (const VarDecl *VD = dyn_cast<VarDecl>(ND)) {
    
    // Check if this is a global variable.
    if (VD->hasExternalStorage() || VD->isFileVarDecl()) 
      return EmitGlobalVarDeclLValue(*this, E, VD);

    bool NonGCable = VD->hasLocalStorage() && !VD->hasAttr<BlocksAttr>();

    llvm::Value *V = LocalDeclMap[VD];
    assert(V && "DeclRefExpr not entered in LocalDeclMap?");

    Qualifiers Quals = MakeQualifiers(E->getType());
    // local variables do not get their gc attribute set.
    // local static?
    if (NonGCable) Quals.removeObjCGCAttr();

    if (VD->hasAttr<BlocksAttr>()) {
      V = Builder.CreateStructGEP(V, 1, "forwarding");
      V = Builder.CreateLoad(V, false);
      V = Builder.CreateStructGEP(V, getByRefValueLLVMField(VD),
                                  VD->getNameAsString());
    }
    if (VD->getType()->isReferenceType())
      V = Builder.CreateLoad(V, "tmp");
    LValue LV = LValue::MakeAddr(V, Quals);
    LValue::SetObjCNonGC(LV, NonGCable);
    setObjCGCLValueClass(getContext(), E, LV);
    return LV;
  }
  
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(ND)) {
    llvm::Value* V = CGM.GetAddrOfFunction(FD);
    if (!FD->hasPrototype()) {
      if (const FunctionProtoType *Proto =
              FD->getType()->getAs<FunctionProtoType>()) {
        // Ugly case: for a K&R-style definition, the type of the definition
        // isn't the same as the type of a use.  Correct for this with a
        // bitcast.
        QualType NoProtoType =
            getContext().getFunctionNoProtoType(Proto->getResultType());
        NoProtoType = getContext().getPointerType(NoProtoType);
        V = Builder.CreateBitCast(V, ConvertType(NoProtoType), "tmp");
      }
    }
    return LValue::MakeAddr(V, MakeQualifiers(E->getType()));
  }
  
  if (E->getQualifier()) {
    // FIXME: the qualifier check does not seem sufficient here
    return EmitPointerToDataMemberLValue(cast<FieldDecl>(ND));
  }
  
  assert(false && "Unhandled DeclRefExpr");
  
  // an invalid LValue, but the assert will
  // ensure that this point is never reached.
  return LValue();
}

LValue CodeGenFunction::EmitBlockDeclRefLValue(const BlockDeclRefExpr *E) {
  return LValue::MakeAddr(GetAddrOfBlockDecl(E), MakeQualifiers(E->getType()));
}

LValue CodeGenFunction::EmitUnaryOpLValue(const UnaryOperator *E) {
  // __extension__ doesn't affect lvalue-ness.
  if (E->getOpcode() == UnaryOperator::Extension)
    return EmitLValue(E->getSubExpr());

  QualType ExprTy = getContext().getCanonicalType(E->getSubExpr()->getType());
  switch (E->getOpcode()) {
  default: assert(0 && "Unknown unary operator lvalue!");
  case UnaryOperator::Deref: {
    QualType T = E->getSubExpr()->getType()->getPointeeType();
    assert(!T.isNull() && "CodeGenFunction::EmitUnaryOpLValue: Illegal type");

    Qualifiers Quals = MakeQualifiers(T);
    Quals.setAddressSpace(ExprTy.getAddressSpace());

    LValue LV = LValue::MakeAddr(EmitScalarExpr(E->getSubExpr()), Quals);
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
  case UnaryOperator::Imag: {
    LValue LV = EmitLValue(E->getSubExpr());
    unsigned Idx = E->getOpcode() == UnaryOperator::Imag;
    return LValue::MakeAddr(Builder.CreateStructGEP(LV.getAddress(),
                                                    Idx, "idx"),
                            MakeQualifiers(ExprTy));
  }
  case UnaryOperator::PreInc:
  case UnaryOperator::PreDec:
    return EmitUnsupportedLValue(E, "pre-inc/dec expression");
  }
}

LValue CodeGenFunction::EmitStringLiteralLValue(const StringLiteral *E) {
  return LValue::MakeAddr(CGM.GetAddrOfConstantStringFromLiteral(E),
                          Qualifiers());
}

LValue CodeGenFunction::EmitObjCEncodeExprLValue(const ObjCEncodeExpr *E) {
  return LValue::MakeAddr(CGM.GetAddrOfConstantStringFromObjCEncode(E),
                          Qualifiers());
}


LValue CodeGenFunction::EmitPredefinedFunctionName(unsigned Type) {
  std::string GlobalVarName;

  switch (Type) {
  default: assert(0 && "Invalid type");
  case PredefinedExpr::Func:
    GlobalVarName = "__func__.";
    break;
  case PredefinedExpr::Function:
    GlobalVarName = "__FUNCTION__.";
    break;
  case PredefinedExpr::PrettyFunction:
    GlobalVarName = "__PRETTY_FUNCTION__.";
    break;
  }

  llvm::StringRef FnName = CurFn->getName();
  if (FnName.startswith("\01"))
    FnName = FnName.substr(1);
  GlobalVarName += FnName;

  std::string FunctionName =
    PredefinedExpr::ComputeName(getContext(), (PredefinedExpr::IdentType)Type,
                                CurCodeDecl);

  llvm::Constant *C =
    CGM.GetAddrOfConstantCString(FunctionName, GlobalVarName.c_str());
  return LValue::MakeAddr(C, Qualifiers());
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

  // We know that the pointer points to a type of the correct size, unless the
  // size is a VLA or Objective-C interface.
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

    const llvm::Type *i8PTy = llvm::Type::getInt8PtrTy(VMContext);
    Address = Builder.CreateGEP(Builder.CreateBitCast(Base, i8PTy),
                                Idx, "arrayidx");
    Address = Builder.CreateBitCast(Address, Base->getType());
  } else {
    Address = Builder.CreateInBoundsGEP(Base, Idx, "arrayidx");
  }

  QualType T = E->getBase()->getType()->getPointeeType();
  assert(!T.isNull() &&
         "CodeGenFunction::EmitArraySubscriptExpr(): Illegal base type");

  Qualifiers Quals = MakeQualifiers(T);
  Quals.setAddressSpace(E->getBase()->getType().getAddressSpace());

  LValue LV = LValue::MakeAddr(Address, Quals);
  if (getContext().getLangOptions().ObjC1 &&
      getContext().getLangOptions().getGCMode() != LangOptions::NonGC) {
    LValue::SetObjCNonGC(LV, !E->isOBJCGCCandidate(getContext()));
    setObjCGCLValueClass(getContext(), E, LV);
  }
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
    Qualifiers Quals = MakeQualifiers(PT->getPointeeType());
    Quals.removeObjCGCAttr();
    Base = LValue::MakeAddr(Ptr, Quals);
  }

  // Encode the element access list into a vector of unsigned indices.
  llvm::SmallVector<unsigned, 4> Indices;
  E->getEncodedElementAccess(Indices);

  if (Base.isSimple()) {
    llvm::Constant *CV = GenerateConstantVector(VMContext, Indices);
    return LValue::MakeExtVectorElt(Base.getAddress(), CV,
                                    Base.getVRQualifiers());
  }
  assert(Base.isExtVectorElt() && "Can only subscript lvalue vec elts here!");

  llvm::Constant *BaseElts = Base.getExtVectorElts();
  llvm::SmallVector<llvm::Constant *, 4> CElts;

  const llvm::Type *Int32Ty = llvm::Type::getInt32Ty(VMContext);
  for (unsigned i = 0, e = Indices.size(); i != e; ++i) {
    if (isa<llvm::ConstantAggregateZero>(BaseElts))
      CElts.push_back(llvm::ConstantInt::get(Int32Ty, 0));
    else
      CElts.push_back(cast<llvm::Constant>(BaseElts->getOperand(Indices[i])));
  }
  llvm::Constant *CV = llvm::ConstantVector::get(&CElts[0], CElts.size());
  return LValue::MakeExtVectorElt(Base.getExtVectorAddr(), CV,
                                  Base.getVRQualifiers());
}

LValue CodeGenFunction::EmitMemberExpr(const MemberExpr *E) {
  bool isUnion = false;
  bool isNonGC = false;
  Expr *BaseExpr = E->getBase();
  llvm::Value *BaseValue = NULL;
  Qualifiers BaseQuals;

  // If this is s.x, emit s as an lvalue.  If it is s->x, emit s as a scalar.
  if (E->isArrow()) {
    BaseValue = EmitScalarExpr(BaseExpr);
    const PointerType *PTy =
      BaseExpr->getType()->getAs<PointerType>();
    if (PTy->getPointeeType()->isUnionType())
      isUnion = true;
    BaseQuals = PTy->getPointeeType().getQualifiers();
  } else if (isa<ObjCPropertyRefExpr>(BaseExpr->IgnoreParens()) ||
             isa<ObjCImplicitSetterGetterRefExpr>(
               BaseExpr->IgnoreParens())) {
    RValue RV = EmitObjCPropertyGet(BaseExpr);
    BaseValue = RV.getAggregateAddr();
    if (BaseExpr->getType()->isUnionType())
      isUnion = true;
    BaseQuals = BaseExpr->getType().getQualifiers();
  } else {
    LValue BaseLV = EmitLValue(BaseExpr);
    if (BaseLV.isNonGC())
      isNonGC = true;
    // FIXME: this isn't right for bitfields.
    BaseValue = BaseLV.getAddress();
    QualType BaseTy = BaseExpr->getType();
    if (BaseTy->isUnionType())
      isUnion = true;
    BaseQuals = BaseTy.getQualifiers();
  }

  NamedDecl *ND = E->getMemberDecl();
  if (FieldDecl *Field = dyn_cast<FieldDecl>(ND)) {
    LValue LV = EmitLValueForField(BaseValue, Field, isUnion,
                                   BaseQuals.getCVRQualifiers());
    LValue::SetObjCNonGC(LV, isNonGC);
    setObjCGCLValueClass(getContext(), E, LV);
    return LV;
  }
  
  if (VarDecl *VD = dyn_cast<VarDecl>(ND))
    return EmitGlobalVarDeclLValue(*this, E, VD);
  
  assert(false && "Unhandled member declaration!");
  return LValue();
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
                                           unsigned CVRQualifiers) {
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

  Qualifiers Quals = MakeQualifiers(Field->getType());
  Quals.addCVRQualifiers(CVRQualifiers);
  // __weak attribute on a field is ignored.
  if (Quals.getObjCGCAttr() == Qualifiers::Weak)
    Quals.removeObjCGCAttr();
  
  return LValue::MakeAddr(V, Quals);
}

LValue CodeGenFunction::EmitCompoundLiteralLValue(const CompoundLiteralExpr* E){
  const llvm::Type *LTy = ConvertType(E->getType());
  llvm::Value *DeclPtr = CreateTempAlloca(LTy, ".compoundliteral");

  const Expr* InitExpr = E->getInitializer();
  LValue Result = LValue::MakeAddr(DeclPtr, MakeQualifiers(E->getType()));

  if (E->getType()->isComplexType())
    EmitComplexExprIntoAddr(InitExpr, DeclPtr, false);
  else if (hasAggregateLLVMType(E->getType()))
    EmitAnyExpr(InitExpr, DeclPtr, false);
  else
    EmitStoreThroughLValue(EmitAnyExpr(InitExpr), Result, E->getType());

  return Result;
}

LValue 
CodeGenFunction::EmitConditionalOperatorLValue(const ConditionalOperator* E) {
  if (E->isLvalue(getContext()) == Expr::LV_Valid) {
    llvm::BasicBlock *LHSBlock = createBasicBlock("cond.true");
    llvm::BasicBlock *RHSBlock = createBasicBlock("cond.false");
    llvm::BasicBlock *ContBlock = createBasicBlock("cond.end");
    
    llvm::Value *Cond = EvaluateExprAsBool(E->getCond());
    Builder.CreateCondBr(Cond, LHSBlock, RHSBlock);
    
    EmitBlock(LHSBlock);

    LValue LHS = EmitLValue(E->getLHS());
    if (!LHS.isSimple())
      return EmitUnsupportedLValue(E, "conditional operator");

    llvm::Value *Temp = CreateTempAlloca(LHS.getAddress()->getType(),"condtmp");
    Builder.CreateStore(LHS.getAddress(), Temp);
    EmitBranch(ContBlock);
    
    EmitBlock(RHSBlock);
    LValue RHS = EmitLValue(E->getRHS());
    if (!RHS.isSimple())
      return EmitUnsupportedLValue(E, "conditional operator");

    Builder.CreateStore(RHS.getAddress(), Temp);
    EmitBranch(ContBlock);

    EmitBlock(ContBlock);
    
    Temp = Builder.CreateLoad(Temp, "lv");
    return LValue::MakeAddr(Temp, MakeQualifiers(E->getType()));
  }
  
  // ?: here should be an aggregate.
  assert((hasAggregateLLVMType(E->getType()) &&
          !E->getType()->isAnyComplexType()) &&
         "Unexpected conditional operator!");

  llvm::Value *Temp = CreateTempAlloca(ConvertType(E->getType()));
  EmitAggExpr(E, Temp, false);

  return LValue::MakeAddr(Temp, MakeQualifiers(E->getType()));
}

/// EmitCastLValue - Casts are never lvalues.  If a cast is needed by the code
/// generator in an lvalue context, then it must mean that we need the address
/// of an aggregate in order to access one of its fields.  This can happen for
/// all the reasons that casts are permitted with aggregate result, including
/// noop aggregate casts, and cast from scalar to union.
LValue CodeGenFunction::EmitCastLValue(const CastExpr *E) {
  switch (E->getCastKind()) {
  default:
    // If this is an lvalue cast, treat it as a no-op.
    // FIXME: We shouldn't need to check for this explicitly!
    if (const ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E))
      if (ICE->isLvalueCast())
        return EmitLValue(E->getSubExpr());
    
    assert(false && "Unhandled cast!");
      
  case CastExpr::CK_NoOp:
  case CastExpr::CK_ConstructorConversion:
  case CastExpr::CK_UserDefinedConversion:
    return EmitLValue(E->getSubExpr());
  
  case CastExpr::CK_DerivedToBase: {
    const RecordType *DerivedClassTy = 
      E->getSubExpr()->getType()->getAs<RecordType>();
    CXXRecordDecl *DerivedClassDecl = 
      cast<CXXRecordDecl>(DerivedClassTy->getDecl());

    const RecordType *BaseClassTy = E->getType()->getAs<RecordType>();
    CXXRecordDecl *BaseClassDecl = cast<CXXRecordDecl>(BaseClassTy->getDecl());
    
    LValue LV = EmitLValue(E->getSubExpr());
    
    // Perform the derived-to-base conversion
    llvm::Value *Base = 
      GetAddressCXXOfBaseClass(LV.getAddress(), DerivedClassDecl, 
                               BaseClassDecl, /*NullCheckValue=*/false);
    
    return LValue::MakeAddr(Base, MakeQualifiers(E->getType()));
  }
  case CastExpr::CK_ToUnion: {
    llvm::Value *Temp = CreateTempAlloca(ConvertType(E->getType()));
    EmitAnyExpr(E->getSubExpr(), Temp, false);

    return LValue::MakeAddr(Temp, MakeQualifiers(E->getType()));
  }
  case CastExpr::CK_BitCast: {
    // This must be a reinterpret_cast.
    const CXXReinterpretCastExpr *CE = cast<CXXReinterpretCastExpr>(E);
    
    LValue LV = EmitLValue(E->getSubExpr());
    llvm::Value *V = Builder.CreateBitCast(LV.getAddress(),
                                           ConvertType(CE->getTypeAsWritten()));
    return LValue::MakeAddr(V, MakeQualifiers(E->getType()));
  }

  }
}

LValue CodeGenFunction::EmitNullInitializationLValue(
                                              const CXXZeroInitValueExpr *E) {
  QualType Ty = E->getType();
  const llvm::Type *LTy = ConvertTypeForMem(Ty);
  llvm::AllocaInst *Alloc = CreateTempAlloca(LTy);
  unsigned Align = getContext().getTypeAlign(Ty)/8;
  Alloc->setAlignment(Align);
  LValue lvalue = LValue::MakeAddr(Alloc, Qualifiers());
  EmitMemSetToZero(lvalue.getAddress(), Ty);
  return lvalue;
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
        if (unsigned builtinID = FD->getBuiltinID())
          return EmitBuiltinExpr(FD, builtinID, E);
    }
  }

  if (const CXXOperatorCallExpr *CE = dyn_cast<CXXOperatorCallExpr>(E))
    if (const CXXMethodDecl *MD = dyn_cast_or_null<CXXMethodDecl>(TargetDecl))
      return EmitCXXOperatorMemberCallExpr(CE, MD);

  if (isa<CXXPseudoDestructorExpr>(E->getCallee())) {
    // C++ [expr.pseudo]p1:
    //   The result shall only be used as the operand for the function call
    //   operator (), and the result of such a call has type void. The only
    //   effect is the evaluation of the postfix-expression before the dot or
    //   arrow.
    EmitScalarExpr(E->getCallee());
    return RValue::get(0);
  }

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

  if (E->getOpcode() == BinaryOperator::PtrMemD ||
      E->getOpcode() == BinaryOperator::PtrMemI)
    return EmitPointerToDataMemberBinaryExpr(E);
  
  // Can only get l-value for binary operator expressions which are a
  // simple assignment of aggregate type.
  if (E->getOpcode() != BinaryOperator::Assign)
    return EmitUnsupportedLValue(E, "binary l-value expression");

  if (!hasAggregateLLVMType(E->getType())) {
    // Emit the LHS as an l-value.
    LValue LV = EmitLValue(E->getLHS());
    
    llvm::Value *RHS = EmitScalarExpr(E->getRHS());
    EmitStoreOfScalar(RHS, LV.getAddress(), LV.isVolatileQualified(), 
                      E->getType());
    return LV;
  }
  
  llvm::Value *Temp = CreateTempAlloca(ConvertType(E->getType()));
  EmitAggExpr(E, Temp, false);
  // FIXME: Are these qualifiers correct?
  return LValue::MakeAddr(Temp, MakeQualifiers(E->getType()));
}

LValue CodeGenFunction::EmitCallExprLValue(const CallExpr *E) {
  RValue RV = EmitCallExpr(E);

  if (!RV.isScalar())
    return LValue::MakeAddr(RV.getAggregateAddr(),MakeQualifiers(E->getType()));
    
  assert(E->getCallReturnType()->isReferenceType() &&
         "Can't have a scalar return unless the return type is a "
         "reference type!");

  return LValue::MakeAddr(RV.getScalarVal(), MakeQualifiers(E->getType()));
}

LValue CodeGenFunction::EmitVAArgExprLValue(const VAArgExpr *E) {
  // FIXME: This shouldn't require another copy.
  llvm::Value *Temp = CreateTempAlloca(ConvertType(E->getType()));
  EmitAggExpr(E, Temp, false);
  return LValue::MakeAddr(Temp, MakeQualifiers(E->getType()));
}

LValue
CodeGenFunction::EmitCXXConditionDeclLValue(const CXXConditionDeclExpr *E) {
  EmitLocalBlockVarDecl(*E->getVarDecl());
  return EmitDeclRefLValue(E);
}

LValue CodeGenFunction::EmitCXXConstructLValue(const CXXConstructExpr *E) {
  llvm::Value *Temp = CreateTempAlloca(ConvertTypeForMem(E->getType()), "tmp");
  EmitCXXConstructExpr(Temp, E);
  return LValue::MakeAddr(Temp, MakeQualifiers(E->getType()));
}

LValue
CodeGenFunction::EmitCXXTypeidLValue(const CXXTypeidExpr *E) {
  llvm::Value *Temp = EmitCXXTypeidExpr(E);
  return LValue::MakeAddr(Temp, MakeQualifiers(E->getType()));
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
  return LValue::MakeAddr(RV.getAggregateAddr(), MakeQualifiers(E->getType()));
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
  Qualifiers BaseQuals;
  QualType ObjectTy;
  if (E->isArrow()) {
    BaseValue = EmitScalarExpr(BaseExpr);
    ObjectTy = BaseExpr->getType()->getPointeeType();
    BaseQuals = ObjectTy.getQualifiers();
  } else {
    LValue BaseLV = EmitLValue(BaseExpr);
    // FIXME: this isn't right for bitfields.
    BaseValue = BaseLV.getAddress();
    ObjectTy = BaseExpr->getType();
    BaseQuals = ObjectTy.getQualifiers();
  }

  LValue LV = 
    EmitLValueForIvar(ObjectTy, BaseValue, E->getDecl(),
                      BaseQuals.getCVRQualifiers());
  setObjCGCLValueClass(getContext(), E, LV);
  return LV;
}

LValue
CodeGenFunction::EmitObjCPropertyRefLValue(const ObjCPropertyRefExpr *E) {
  // This is a special l-value that just issues sends when we load or store
  // through it.
  return LValue::MakePropertyRef(E, E->getType().getCVRQualifiers());
}

LValue CodeGenFunction::EmitObjCKVCRefLValue(
                                const ObjCImplicitSetterGetterRefExpr *E) {
  // This is a special l-value that just issues sends when we load or store
  // through it.
  return LValue::MakeKVCRef(E, E->getType().getCVRQualifiers());
}

LValue CodeGenFunction::EmitObjCSuperExprLValue(const ObjCSuperExpr *E) {
  return EmitUnsupportedLValue(E, "use of super");
}

LValue CodeGenFunction::EmitStmtExprLValue(const StmtExpr *E) {
  // Can only get l-value for message expression returning aggregate type
  RValue RV = EmitAnyExprToTemp(E);
  // FIXME: can this be volatile?
  return LValue::MakeAddr(RV.getAggregateAddr(), MakeQualifiers(E->getType()));
}


LValue CodeGenFunction::EmitPointerToDataMemberLValue(const FieldDecl *Field) {
  const CXXRecordDecl *ClassDecl = cast<CXXRecordDecl>(Field->getDeclContext());
  QualType NNSpecTy = 
    getContext().getCanonicalType(
      getContext().getTypeDeclType(const_cast<CXXRecordDecl*>(ClassDecl)));
  NNSpecTy = getContext().getPointerType(NNSpecTy);
  llvm::Value *V = llvm::Constant::getNullValue(ConvertType(NNSpecTy));
  LValue MemExpLV = EmitLValueForField(V, const_cast<FieldDecl*>(Field), 
                                       /*isUnion*/false, /*Qualifiers*/0);
  const llvm::Type *ResultType = ConvertType(getContext().getPointerDiffType());
  V = Builder.CreatePtrToInt(MemExpLV.getAddress(), ResultType, "datamember");
  return LValue::MakeAddr(V, MakeQualifiers(Field->getType()));
}

RValue CodeGenFunction::EmitCall(llvm::Value *Callee, QualType CalleeType,
                                 CallExpr::const_arg_iterator ArgBeg,
                                 CallExpr::const_arg_iterator ArgEnd,
                                 const Decl *TargetDecl) {
  // Get the actual function type. The callee type will always be a pointer to
  // function type or a block pointer type.
  assert(CalleeType->isFunctionPointerType() &&
         "Call must have function pointer type!");

  CalleeType = getContext().getCanonicalType(CalleeType);

  QualType FnType = cast<PointerType>(CalleeType)->getPointeeType();
  QualType ResultType = cast<FunctionType>(FnType)->getResultType();

  CallArgList Args;
  EmitCallArgs(Args, dyn_cast<FunctionProtoType>(FnType), ArgBeg, ArgEnd);

  // FIXME: We should not need to do this, it should be part of the function
  // type.
  unsigned CallingConvention = 0;
  if (const llvm::Function *F =
      dyn_cast<llvm::Function>(Callee->stripPointerCasts()))
    CallingConvention = F->getCallingConv();
  return EmitCall(CGM.getTypes().getFunctionInfo(ResultType, Args,
                                                 CallingConvention),
                  Callee, Args, TargetDecl);
}

LValue CodeGenFunction::
EmitPointerToDataMemberBinaryExpr(const BinaryOperator *E) {
  llvm::Value *BaseV = EmitLValue(E->getLHS()).getAddress();
  if (E->getOpcode() == BinaryOperator::PtrMemI)
    BaseV = Builder.CreateLoad(BaseV, "indir.ptr");
  const llvm::Type *i8Ty = llvm::Type::getInt8PtrTy(getLLVMContext());
  BaseV = Builder.CreateBitCast(BaseV, i8Ty);
  LValue RHSLV = EmitLValue(E->getRHS());
  llvm::Value *OffsetV = 
    EmitLoadOfLValue(RHSLV, E->getRHS()->getType()).getScalarVal();
  const llvm::Type* ResultType = ConvertType(getContext().getPointerDiffType());
  OffsetV = Builder.CreateBitCast(OffsetV, ResultType);
  llvm::Value *AddV = Builder.CreateInBoundsGEP(BaseV, OffsetV, "add.ptr");

  QualType Ty = E->getRHS()->getType();
  Ty = Ty->getAs<MemberPointerType>()->getPointeeType();
  
  const llvm::Type *PType = ConvertType(getContext().getPointerType(Ty));
  AddV = Builder.CreateBitCast(AddV, PType);
  return LValue::MakeAddr(AddV, MakeQualifiers(Ty));
}

