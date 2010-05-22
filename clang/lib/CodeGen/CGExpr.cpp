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
#include "CGRecordLayout.h"
#include "CGObjCRuntime.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "llvm/Intrinsics.h"
#include "clang/CodeGen/CodeGenOptions.h"
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

void CodeGenFunction::InitTempAlloca(llvm::AllocaInst *Var,
                                     llvm::Value *Init) {
  llvm::StoreInst *Store = new llvm::StoreInst(Init, Var);
  llvm::BasicBlock *Block = AllocaInsertPt->getParent();
  Block->getInstList().insertAfter(&*AllocaInsertPt, Store);
}

llvm::Value *CodeGenFunction::CreateIRTemp(QualType Ty,
                                           const llvm::Twine &Name) {
  llvm::AllocaInst *Alloc = CreateTempAlloca(ConvertType(Ty), Name);
  // FIXME: Should we prefer the preferred type alignment here?
  CharUnits Align = getContext().getTypeAlignInChars(Ty);
  Alloc->setAlignment(Align.getQuantity());
  return Alloc;
}

llvm::Value *CodeGenFunction::CreateMemTemp(QualType Ty,
                                            const llvm::Twine &Name) {
  llvm::AllocaInst *Alloc = CreateTempAlloca(ConvertTypeForMem(Ty), Name);
  // FIXME: Should we prefer the preferred type alignment here?
  CharUnits Align = getContext().getTypeAlignInChars(Ty);
  Alloc->setAlignment(Align.getQuantity());
  return Alloc;
}

/// EvaluateExprAsBool - Perform the usual unary conversions on the specified
/// expression and compare the result against zero, returning an Int1Ty value.
llvm::Value *CodeGenFunction::EvaluateExprAsBool(const Expr *E) {
  QualType BoolTy = getContext().BoolTy;
  if (E->getType()->isMemberFunctionPointerType()) {
    LValue LV = EmitAggExprToLValue(E);

    // Get the pointer.
    llvm::Value *FuncPtr = Builder.CreateStructGEP(LV.getAddress(), 0,
                                                   "src.ptr");
    FuncPtr = Builder.CreateLoad(FuncPtr);

    llvm::Value *IsNotNull = 
      Builder.CreateICmpNE(FuncPtr,
                            llvm::Constant::getNullValue(FuncPtr->getType()),
                            "tobool");

    return IsNotNull;
  }
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
    AggLoc = CreateMemTemp(E->getType(), "agg.tmp");
  return EmitAnyExpr(E, AggLoc, IsAggLocVolatile, /*IgnoreResult=*/false,
                     IsInitializer);
}

/// EmitAnyExprToMem - Evaluate an expression into a given memory
/// location.
void CodeGenFunction::EmitAnyExprToMem(const Expr *E,
                                       llvm::Value *Location,
                                       bool IsLocationVolatile,
                                       bool IsInit) {
  if (E->getType()->isComplexType())
    EmitComplexExprIntoAddr(E, Location, IsLocationVolatile);
  else if (hasAggregateLLVMType(E->getType()))
    EmitAggExpr(E, Location, IsLocationVolatile, /*Ignore*/ false, IsInit);
  else {
    RValue RV = RValue::get(EmitScalarExpr(E, /*Ignore*/ false));
    LValue LV = LValue::MakeAddr(Location, MakeQualifiers(E->getType()));
    EmitStoreThroughLValue(RV, LV, E->getType());
  }
}

/// \brief An adjustment to be made to the temporary created when emitting a
/// reference binding, which accesses a particular subobject of that temporary.
struct SubobjectAdjustment {
  enum { DerivedToBaseAdjustment, FieldAdjustment } Kind;
  
  union {
    struct {
      const CXXBaseSpecifierArray *BasePath;
      const CXXRecordDecl *DerivedClass;
    } DerivedToBase;
    
    struct {
      FieldDecl *Field;
      unsigned CVRQualifiers;
    } Field;
  };
  
  SubobjectAdjustment(const CXXBaseSpecifierArray *BasePath, 
                      const CXXRecordDecl *DerivedClass)
    : Kind(DerivedToBaseAdjustment) 
  {
    DerivedToBase.BasePath = BasePath;
    DerivedToBase.DerivedClass = DerivedClass;
  }
  
  SubobjectAdjustment(FieldDecl *Field, unsigned CVRQualifiers)
    : Kind(FieldAdjustment) 
  { 
    this->Field.Field = Field;
    this->Field.CVRQualifiers = CVRQualifiers;
  }
};

RValue CodeGenFunction::EmitReferenceBindingToExpr(const Expr* E,
                                                   bool IsInitializer) {
  bool ShouldDestroyTemporaries = false;
  unsigned OldNumLiveTemporaries = 0;

  if (const CXXDefaultArgExpr *DAE = dyn_cast<CXXDefaultArgExpr>(E))
    E = DAE->getExpr();

  if (const CXXExprWithTemporaries *TE = dyn_cast<CXXExprWithTemporaries>(E)) {
    ShouldDestroyTemporaries = true;
    
    // Keep track of the current cleanup stack depth.
    OldNumLiveTemporaries = LiveTemporaries.size();
    
    E = TE->getSubExpr();
  }
  
  RValue Val;
  if (E->isLvalue(getContext()) == Expr::LV_Valid) {
    // Emit the expr as an lvalue.
    LValue LV = EmitLValue(E);
    if (LV.isSimple()) {
      if (ShouldDestroyTemporaries) {
        // Pop temporaries.
        while (LiveTemporaries.size() > OldNumLiveTemporaries)
          PopCXXTemporary();
      }
      
      return RValue::get(LV.getAddress());
    }
    
    Val = EmitLoadOfLValue(LV, E->getType());
    
    if (ShouldDestroyTemporaries) {
      // Pop temporaries.
      while (LiveTemporaries.size() > OldNumLiveTemporaries)
        PopCXXTemporary();
    }      
  } else {
    QualType ResultTy = E->getType();
    
    llvm::SmallVector<SubobjectAdjustment, 2> Adjustments;
    do {
      if (const ParenExpr *PE = dyn_cast<ParenExpr>(E)) {
        E = PE->getSubExpr();
        continue;
      } 

      if (const CastExpr *CE = dyn_cast<CastExpr>(E)) {
        if ((CE->getCastKind() == CastExpr::CK_DerivedToBase ||
             CE->getCastKind() == CastExpr::CK_UncheckedDerivedToBase) &&
            E->getType()->isRecordType()) {
          E = CE->getSubExpr();
          CXXRecordDecl *Derived 
            = cast<CXXRecordDecl>(E->getType()->getAs<RecordType>()->getDecl());
          Adjustments.push_back(SubobjectAdjustment(&CE->getBasePath(), 
                                                    Derived));
          continue;
        }

        if (CE->getCastKind() == CastExpr::CK_NoOp) {
          E = CE->getSubExpr();
          continue;
        }
      } else if (const MemberExpr *ME = dyn_cast<MemberExpr>(E)) {
        if (ME->getBase()->isLvalue(getContext()) != Expr::LV_Valid &&
            ME->getBase()->getType()->isRecordType()) {
          if (FieldDecl *Field = dyn_cast<FieldDecl>(ME->getMemberDecl())) {
            E = ME->getBase();
            Adjustments.push_back(SubobjectAdjustment(Field,
                                              E->getType().getCVRQualifiers()));
            continue;
          }
        }
      }

      // Nothing changed.
      break;
    } while (true);
      
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

            {
              DelayedCleanupBlock Scope(*this);
              EmitCXXDestructorCall(Dtor, Dtor_Complete,
                                    /*ForVirtualBase=*/false,
                                    Val.getAggregateAddr());
              
              // Make sure to jump to the exit block.
              EmitBranch(Scope.getCleanupExitBlock());
            }
            if (Exceptions) {
              EHCleanupBlock Cleanup(*this);
              EmitCXXDestructorCall(Dtor, Dtor_Complete,
                                    /*ForVirtualBase=*/false,
                                    Val.getAggregateAddr());
            }
          }
        }
      }
    }
    
    // Check if need to perform derived-to-base casts and/or field accesses, to
    // get from the temporary object we created (and, potentially, for which we
    // extended the lifetime) to the subobject we're binding the reference to.
    if (!Adjustments.empty()) {
      llvm::Value *Object = Val.getAggregateAddr();
      for (unsigned I = Adjustments.size(); I != 0; --I) {
        SubobjectAdjustment &Adjustment = Adjustments[I-1];
        switch (Adjustment.Kind) {
        case SubobjectAdjustment::DerivedToBaseAdjustment:
          Object = GetAddressOfBaseClass(Object, 
                                         Adjustment.DerivedToBase.DerivedClass, 
                                         *Adjustment.DerivedToBase.BasePath, 
                                         /*NullCheckValue=*/false);
          break;
            
        case SubobjectAdjustment::FieldAdjustment: {
          unsigned CVR = Adjustment.Field.CVRQualifiers;
          LValue LV = EmitLValueForField(Object, Adjustment.Field.Field, CVR);
          if (LV.isSimple()) {
            Object = LV.getAddress();
            break;
          }
          
          // For non-simple lvalues, we actually have to create a copy of
          // the object we're binding to.
          QualType T = Adjustment.Field.Field->getType().getNonReferenceType()
                                                        .getUnqualifiedType();
          Object = CreateTempAlloca(ConvertType(T), "lv");
          EmitStoreThroughLValue(EmitLoadOfLValue(LV, T), 
                                 LValue::MakeAddr(Object, 
                                                  Qualifiers::fromCVRMask(CVR)),
                                 T);
          break;
        }
        }
      }
      
      const llvm::Type *ResultPtrTy
        = llvm::PointerType::get(ConvertType(ResultTy), 0);
      Object = Builder.CreateBitCast(Object, ResultPtrTy, "temp");
      return RValue::get(Object);
    }
  }

  if (Val.isAggregate()) {
    Val = RValue::get(Val.getAggregateAddr());
  } else {
    // Create a temporary variable that we can bind the reference to.
    llvm::Value *Temp = CreateMemTemp(E->getType(), "reftmp");
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

void CodeGenFunction::EmitCheck(llvm::Value *Address, unsigned Size) {
  if (!CatchUndefined)
    return;

  const llvm::Type *Size_tTy
    = llvm::IntegerType::get(VMContext, LLVMPointerWidth);
  Address = Builder.CreateBitCast(Address, PtrToInt8Ty);

  llvm::Value *F = CGM.getIntrinsic(llvm::Intrinsic::objectsize, &Size_tTy, 1);
  const llvm::IntegerType *Int1Ty = llvm::IntegerType::get(VMContext, 1);

  // In time, people may want to control this and use a 1 here.
  llvm::Value *Arg = llvm::ConstantInt::get(Int1Ty, 0);
  llvm::Value *C = Builder.CreateCall2(F, Address, Arg);
  llvm::BasicBlock *Cont = createBasicBlock();
  llvm::BasicBlock *Check = createBasicBlock();
  llvm::Value *NegativeOne = llvm::ConstantInt::get(Size_tTy, -1ULL);
  Builder.CreateCondBr(Builder.CreateICmpEQ(C, NegativeOne), Cont, Check);
    
  EmitBlock(Check);
  Builder.CreateCondBr(Builder.CreateICmpUGE(C,
                                        llvm::ConstantInt::get(Size_tTy, Size)),
                       Cont, getTrapBB());
  EmitBlock(Cont);
}


llvm::Value *CodeGenFunction::
EmitScalarPrePostIncDec(const UnaryOperator *E, LValue LV,
                        bool isInc, bool isPre) {
  QualType ValTy = E->getSubExpr()->getType();
  llvm::Value *InVal = EmitLoadOfLValue(LV, ValTy).getScalarVal();
  
  int AmountVal = isInc ? 1 : -1;
  
  if (ValTy->isPointerType() &&
      ValTy->getAs<PointerType>()->isVariableArrayType()) {
    // The amount of the addition/subtraction needs to account for the VLA size
    ErrorUnsupported(E, "VLA pointer inc/dec");
  }
  
  llvm::Value *NextVal;
  if (const llvm::PointerType *PT =
      dyn_cast<llvm::PointerType>(InVal->getType())) {
    llvm::Constant *Inc =
    llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), AmountVal);
    if (!isa<llvm::FunctionType>(PT->getElementType())) {
      QualType PTEE = ValTy->getPointeeType();
      if (const ObjCObjectType *OIT = PTEE->getAs<ObjCObjectType>()) {
        // Handle interface types, which are not represented with a concrete
        // type.
        int size = getContext().getTypeSize(OIT) / 8;
        if (!isInc)
          size = -size;
        Inc = llvm::ConstantInt::get(Inc->getType(), size);
        const llvm::Type *i8Ty = llvm::Type::getInt8PtrTy(VMContext);
        InVal = Builder.CreateBitCast(InVal, i8Ty);
        NextVal = Builder.CreateGEP(InVal, Inc, "add.ptr");
        llvm::Value *lhs = LV.getAddress();
        lhs = Builder.CreateBitCast(lhs, llvm::PointerType::getUnqual(i8Ty));
        LV = LValue::MakeAddr(lhs, MakeQualifiers(ValTy));
      } else
        NextVal = Builder.CreateInBoundsGEP(InVal, Inc, "ptrincdec");
    } else {
      const llvm::Type *i8Ty = llvm::Type::getInt8PtrTy(VMContext);
      NextVal = Builder.CreateBitCast(InVal, i8Ty, "tmp");
      NextVal = Builder.CreateGEP(NextVal, Inc, "ptrincdec");
      NextVal = Builder.CreateBitCast(NextVal, InVal->getType());
    }
  } else if (InVal->getType() == llvm::Type::getInt1Ty(VMContext) && isInc) {
    // Bool++ is an interesting case, due to promotion rules, we get:
    // Bool++ -> Bool = Bool+1 -> Bool = (int)Bool+1 ->
    // Bool = ((int)Bool+1) != 0
    // An interesting aspect of this is that increment is always true.
    // Decrement does not have this property.
    NextVal = llvm::ConstantInt::getTrue(VMContext);
  } else if (isa<llvm::IntegerType>(InVal->getType())) {
    NextVal = llvm::ConstantInt::get(InVal->getType(), AmountVal);
    
    // Signed integer overflow is undefined behavior.
    if (ValTy->isSignedIntegerType())
      NextVal = Builder.CreateNSWAdd(InVal, NextVal, isInc ? "inc" : "dec");
    else
      NextVal = Builder.CreateAdd(InVal, NextVal, isInc ? "inc" : "dec");
  } else {
    // Add the inc/dec to the real part.
    if (InVal->getType()->isFloatTy())
      NextVal =
      llvm::ConstantFP::get(VMContext,
                            llvm::APFloat(static_cast<float>(AmountVal)));
    else if (InVal->getType()->isDoubleTy())
      NextVal =
      llvm::ConstantFP::get(VMContext,
                            llvm::APFloat(static_cast<double>(AmountVal)));
    else {
      llvm::APFloat F(static_cast<float>(AmountVal));
      bool ignored;
      F.convert(Target.getLongDoubleFormat(), llvm::APFloat::rmTowardZero,
                &ignored);
      NextVal = llvm::ConstantFP::get(VMContext, F);
    }
    NextVal = Builder.CreateFAdd(InVal, NextVal, isInc ? "inc" : "dec");
  }
  
  // Store the updated result through the lvalue.
  if (LV.isBitField())
    EmitStoreThroughBitfieldLValue(RValue::get(NextVal), LV, ValTy, &NextVal);
  else
    EmitStoreThroughLValue(RValue::get(NextVal), LV, ValTy);
  
  // If this is a postinc, return the value read from memory, otherwise use the
  // updated value.
  return isPre ? NextVal : InVal;
}


CodeGenFunction::ComplexPairTy CodeGenFunction::
EmitComplexPrePostIncDec(const UnaryOperator *E, LValue LV,
                         bool isInc, bool isPre) {
  ComplexPairTy InVal = LoadComplexFromAddr(LV.getAddress(),
                                            LV.isVolatileQualified());
  
  llvm::Value *NextVal;
  if (isa<llvm::IntegerType>(InVal.first->getType())) {
    uint64_t AmountVal = isInc ? 1 : -1;
    NextVal = llvm::ConstantInt::get(InVal.first->getType(), AmountVal, true);
    
    // Add the inc/dec to the real part.
    NextVal = Builder.CreateAdd(InVal.first, NextVal, isInc ? "inc" : "dec");
  } else {
    QualType ElemTy = E->getType()->getAs<ComplexType>()->getElementType();
    llvm::APFloat FVal(getContext().getFloatTypeSemantics(ElemTy), 1);
    if (!isInc)
      FVal.changeSign();
    NextVal = llvm::ConstantFP::get(getLLVMContext(), FVal);
    
    // Add the inc/dec to the real part.
    NextVal = Builder.CreateFAdd(InVal.first, NextVal, isInc ? "inc" : "dec");
  }
  
  ComplexPairTy IncVal(NextVal, InVal.second);
  
  // Store the updated result through the lvalue.
  StoreComplexToAddr(IncVal, LV.getAddress(), LV.isVolatileQualified());
  
  // If this is a postinc, return the value read from memory, otherwise use the
  // updated value.
  return isPre ? IncVal : InVal;
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

LValue CodeGenFunction::EmitCheckedLValue(const Expr *E) {
  LValue LV = EmitLValue(E);
  if (!isa<DeclRefExpr>(E) && !LV.isBitField() && LV.isSimple())
    EmitCheck(LV.getAddress(), getContext().getTypeSize(E->getType()) / 8);
  return LV;
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

  case Expr::ObjCIsaExprClass:
    return EmitObjCIsaExpr(cast<ObjCIsaExpr>(E));
  case Expr::BinaryOperatorClass:
    return EmitBinaryOperatorLValue(cast<BinaryOperator>(E));
  case Expr::CompoundAssignOperatorClass:
    return EmitCompoundAssignOperatorLValue(cast<CompoundAssignOperator>(E));
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
  llvm::LoadInst *Load = Builder.CreateLoad(Addr, "tmp");
  if (Volatile)
    Load->setVolatile(true);

  // Bool can have different representation in memory than in registers.
  llvm::Value *V = Load;
  if (Ty->isBooleanType())
    if (V->getType() != llvm::Type::getInt1Ty(VMContext))
      V = Builder.CreateTrunc(V, llvm::Type::getInt1Ty(VMContext), "tobool");

  return V;
}

void CodeGenFunction::EmitStoreOfScalar(llvm::Value *Value, llvm::Value *Addr,
                                        bool Volatile, QualType Ty) {

  if (Ty->isBooleanType()) {
    // Bool can have different representation in memory than in registers.
    const llvm::PointerType *DstPtr = cast<llvm::PointerType>(Addr->getType());
    Value = Builder.CreateIntCast(Value, DstPtr->getElementType(), false);
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
    //
    // FIXME: We shouldn't have to use isSingleValueType here.
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

  if (LV.isBitField())
    return EmitLoadOfBitfieldLValue(LV, ExprType);

  if (LV.isPropertyRef())
    return EmitLoadOfPropertyRefLValue(LV, ExprType);

  assert(LV.isKVCRef() && "Unknown LValue type!");
  return EmitLoadOfKVCRefLValue(LV, ExprType);
}

RValue CodeGenFunction::EmitLoadOfBitfieldLValue(LValue LV,
                                                 QualType ExprType) {
  const CGBitFieldInfo &Info = LV.getBitFieldInfo();

  // Get the output type.
  const llvm::Type *ResLTy = ConvertType(ExprType);
  unsigned ResSizeInBits = CGM.getTargetData().getTypeSizeInBits(ResLTy);

  // Compute the result as an OR of all of the individual component accesses.
  llvm::Value *Res = 0;
  for (unsigned i = 0, e = Info.getNumComponents(); i != e; ++i) {
    const CGBitFieldInfo::AccessInfo &AI = Info.getComponent(i);

    // Get the field pointer.
    llvm::Value *Ptr = LV.getBitFieldBaseAddr();

    // Only offset by the field index if used, so that incoming values are not
    // required to be structures.
    if (AI.FieldIndex)
      Ptr = Builder.CreateStructGEP(Ptr, AI.FieldIndex, "bf.field");

    // Offset by the byte offset, if used.
    if (AI.FieldByteOffset) {
      const llvm::Type *i8PTy = llvm::Type::getInt8PtrTy(VMContext);
      Ptr = Builder.CreateBitCast(Ptr, i8PTy);
      Ptr = Builder.CreateConstGEP1_32(Ptr, AI.FieldByteOffset,"bf.field.offs");
    }

    // Cast to the access type.
    const llvm::Type *PTy = llvm::Type::getIntNPtrTy(VMContext, AI.AccessWidth,
                                                    ExprType.getAddressSpace());
    Ptr = Builder.CreateBitCast(Ptr, PTy);

    // Perform the load.
    llvm::LoadInst *Load = Builder.CreateLoad(Ptr, LV.isVolatileQualified());
    if (AI.AccessAlignment)
      Load->setAlignment(AI.AccessAlignment);

    // Shift out unused low bits and mask out unused high bits.
    llvm::Value *Val = Load;
    if (AI.FieldBitStart)
      Val = Builder.CreateLShr(Load, AI.FieldBitStart);
    Val = Builder.CreateAnd(Val, llvm::APInt::getLowBitsSet(AI.AccessWidth,
                                                            AI.TargetBitWidth),
                            "bf.clear");

    // Extend or truncate to the target size.
    if (AI.AccessWidth < ResSizeInBits)
      Val = Builder.CreateZExt(Val, ResLTy);
    else if (AI.AccessWidth > ResSizeInBits)
      Val = Builder.CreateTrunc(Val, ResLTy);

    // Shift into place, and OR into the result.
    if (AI.TargetBitOffset)
      Val = Builder.CreateShl(Val, AI.TargetBitOffset);
    Res = Res ? Builder.CreateOr(Res, Val) : Val;
  }

  // If the bit-field is signed, perform the sign-extension.
  //
  // FIXME: This can easily be folded into the load of the high bits, which
  // could also eliminate the mask of high bits in some situations.
  if (Info.isSigned()) {
    unsigned ExtraBits = ResSizeInBits - Info.getSize();
    if (ExtraBits)
      Res = Builder.CreateAShr(Builder.CreateShl(Res, ExtraBits),
                               ExtraBits, "bf.val.sext");
  }

  return RValue::get(Res);
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

    if (Dst.isBitField())
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
  const CGBitFieldInfo &Info = Dst.getBitFieldInfo();

  // Get the output type.
  const llvm::Type *ResLTy = ConvertTypeForMem(Ty);
  unsigned ResSizeInBits = CGM.getTargetData().getTypeSizeInBits(ResLTy);

  // Get the source value, truncated to the width of the bit-field.
  llvm::Value *SrcVal = Src.getScalarVal();

  if (Ty->isBooleanType())
    SrcVal = Builder.CreateIntCast(SrcVal, ResLTy, /*IsSigned=*/false);

  SrcVal = Builder.CreateAnd(SrcVal, llvm::APInt::getLowBitsSet(ResSizeInBits,
                                                                Info.getSize()),
                             "bf.value");

  // Return the new value of the bit-field, if requested.
  if (Result) {
    // Cast back to the proper type for result.
    const llvm::Type *SrcTy = Src.getScalarVal()->getType();
    llvm::Value *ReloadVal = Builder.CreateIntCast(SrcVal, SrcTy, false,
                                                   "bf.reload.val");

    // Sign extend if necessary.
    if (Info.isSigned()) {
      unsigned ExtraBits = ResSizeInBits - Info.getSize();
      if (ExtraBits)
        ReloadVal = Builder.CreateAShr(Builder.CreateShl(ReloadVal, ExtraBits),
                                       ExtraBits, "bf.reload.sext");
    }

    *Result = ReloadVal;
  }

  // Iterate over the components, writing each piece to memory.
  for (unsigned i = 0, e = Info.getNumComponents(); i != e; ++i) {
    const CGBitFieldInfo::AccessInfo &AI = Info.getComponent(i);

    // Get the field pointer.
    llvm::Value *Ptr = Dst.getBitFieldBaseAddr();

    // Only offset by the field index if used, so that incoming values are not
    // required to be structures.
    if (AI.FieldIndex)
      Ptr = Builder.CreateStructGEP(Ptr, AI.FieldIndex, "bf.field");

    // Offset by the byte offset, if used.
    if (AI.FieldByteOffset) {
      const llvm::Type *i8PTy = llvm::Type::getInt8PtrTy(VMContext);
      Ptr = Builder.CreateBitCast(Ptr, i8PTy);
      Ptr = Builder.CreateConstGEP1_32(Ptr, AI.FieldByteOffset,"bf.field.offs");
    }

    // Cast to the access type.
    const llvm::Type *PTy = llvm::Type::getIntNPtrTy(VMContext, AI.AccessWidth,
                                                     Ty.getAddressSpace());
    Ptr = Builder.CreateBitCast(Ptr, PTy);

    // Extract the piece of the bit-field value to write in this access, limited
    // to the values that are part of this access.
    llvm::Value *Val = SrcVal;
    if (AI.TargetBitOffset)
      Val = Builder.CreateLShr(Val, AI.TargetBitOffset);
    Val = Builder.CreateAnd(Val, llvm::APInt::getLowBitsSet(ResSizeInBits,
                                                            AI.TargetBitWidth));

    // Extend or truncate to the access size.
    const llvm::Type *AccessLTy =
      llvm::Type::getIntNTy(VMContext, AI.AccessWidth);
    if (ResSizeInBits < AI.AccessWidth)
      Val = Builder.CreateZExt(Val, AccessLTy);
    else if (ResSizeInBits > AI.AccessWidth)
      Val = Builder.CreateTrunc(Val, AccessLTy);

    // Shift into the position in memory.
    if (AI.FieldBitStart)
      Val = Builder.CreateShl(Val, AI.FieldBitStart);

    // If necessary, load and OR in bits that are outside of the bit-field.
    if (AI.TargetBitWidth != AI.AccessWidth) {
      llvm::LoadInst *Load = Builder.CreateLoad(Ptr, Dst.isVolatileQualified());
      if (AI.AccessAlignment)
        Load->setAlignment(AI.AccessAlignment);

      // Compute the mask for zeroing the bits that are part of the bit-field.
      llvm::APInt InvMask =
        ~llvm::APInt::getBitsSet(AI.AccessWidth, AI.FieldBitStart,
                                 AI.FieldBitStart + AI.TargetBitWidth);

      // Apply the mask and OR in to the value to write.
      Val = Builder.CreateOr(Builder.CreateAnd(Load, InvMask), Val);
    }

    // Write the value.
    llvm::StoreInst *Store = Builder.CreateStore(Val, Ptr,
                                                 Dst.isVolatileQualified());
    if (AI.AccessAlignment)
      Store->setAlignment(AI.AccessAlignment);
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

static LValue EmitFunctionDeclLValue(CodeGenFunction &CGF,
                                      const Expr *E, const FunctionDecl *FD) {
  llvm::Value* V = CGF.CGM.GetAddrOfFunction(FD);
  if (!FD->hasPrototype()) {
    if (const FunctionProtoType *Proto =
            FD->getType()->getAs<FunctionProtoType>()) {
      // Ugly case: for a K&R-style definition, the type of the definition
      // isn't the same as the type of a use.  Correct for this with a
      // bitcast.
      QualType NoProtoType =
          CGF.getContext().getFunctionNoProtoType(Proto->getResultType());
      NoProtoType = CGF.getContext().getPointerType(NoProtoType);
      V = CGF.Builder.CreateBitCast(V, CGF.ConvertType(NoProtoType), "tmp");
    }
  }
  return LValue::MakeAddr(V, CGF.MakeQualifiers(E->getType()));
}

LValue CodeGenFunction::EmitDeclRefLValue(const DeclRefExpr *E) {
  const NamedDecl *ND = E->getDecl();

  if (ND->hasAttr<WeakRefAttr>()) {
    const ValueDecl* VD = cast<ValueDecl>(ND);
    llvm::Constant *Aliasee = CGM.GetWeakRefReference(VD);

    Qualifiers Quals = MakeQualifiers(E->getType());
    LValue LV = LValue::MakeAddr(Aliasee, Quals);

    return LV;
  }

  if (const VarDecl *VD = dyn_cast<VarDecl>(ND)) {
    
    // Check if this is a global variable.
    if (VD->hasExternalStorage() || VD->isFileVarDecl()) 
      return EmitGlobalVarDeclLValue(*this, E, VD);

    bool NonGCable = VD->hasLocalStorage() && !VD->hasAttr<BlocksAttr>();

    llvm::Value *V = LocalDeclMap[VD];
    if (!V && getContext().getLangOptions().CPlusPlus &&
        VD->isStaticLocal()) 
      V = CGM.getStaticLocalDeclAddress(VD);
    assert(V && "DeclRefExpr not entered in LocalDeclMap?");

    Qualifiers Quals = MakeQualifiers(E->getType());
    // local variables do not get their gc attribute set.
    // local static?
    if (NonGCable) Quals.removeObjCGCAttr();

    if (VD->hasAttr<BlocksAttr>()) {
      V = Builder.CreateStructGEP(V, 1, "forwarding");
      V = Builder.CreateLoad(V);
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
  
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(ND))
    return EmitFunctionDeclLValue(*this, E, FD);
  
  // FIXME: the qualifier check does not seem sufficient here
  if (E->getQualifier()) {
    const FieldDecl *FD = cast<FieldDecl>(ND);
    llvm::Value *V = CGM.EmitPointerToDataMember(FD);

    return LValue::MakeAddr(V, MakeQualifiers(FD->getType()));
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
  case UnaryOperator::PreDec: {
    LValue LV = EmitLValue(E->getSubExpr());
    bool isInc = E->getOpcode() == UnaryOperator::PreInc;
    
    if (E->getType()->isAnyComplexType())
      EmitComplexPrePostIncDec(E, LV, isInc, true/*isPre*/);
    else
      EmitScalarPrePostIncDec(E, LV, isInc, true/*isPre*/);
    return LV;
  }
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
    PredefinedExpr::ComputeName((PredefinedExpr::IdentType)Type, CurCodeDecl);

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

llvm::BasicBlock *CodeGenFunction::getTrapBB() {
  const CodeGenOptions &GCO = CGM.getCodeGenOpts();

  // If we are not optimzing, don't collapse all calls to trap in the function
  // to the same call, that way, in the debugger they can see which operation
  // did in fact fail.  If we are optimizing, we collpase all call to trap down
  // to just one per function to save on codesize.
  if (GCO.OptimizationLevel
      && TrapBB)
    return TrapBB;

  llvm::BasicBlock *Cont = 0;
  if (HaveInsertPoint()) {
    Cont = createBasicBlock("cont");
    EmitBranch(Cont);
  }
  TrapBB = createBasicBlock("trap");
  EmitBlock(TrapBB);

  llvm::Value *F = CGM.getIntrinsic(llvm::Intrinsic::trap, 0, 0);
  llvm::CallInst *TrapCall = Builder.CreateCall(F);
  TrapCall->setDoesNotReturn();
  TrapCall->setDoesNotThrow();
  Builder.CreateUnreachable();

  if (Cont)
    EmitBlock(Cont);
  return TrapBB;
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

  // FIXME: As llvm implements the object size checking, this can come out.
  if (CatchUndefined) {
    if (const ImplicitCastExpr *ICE=dyn_cast<ImplicitCastExpr>(E->getBase())) {
      if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(ICE->getSubExpr())) {
        if (ICE->getCastKind() == CastExpr::CK_ArrayToPointerDecay) {
          if (const ConstantArrayType *CAT
              = getContext().getAsConstantArrayType(DRE->getType())) {
            llvm::APInt Size = CAT->getSize();
            llvm::BasicBlock *Cont = createBasicBlock("cont");
            Builder.CreateCondBr(Builder.CreateICmpULE(Idx,
                                  llvm::ConstantInt::get(Idx->getType(), Size)),
                                 Cont, getTrapBB());
            EmitBlock(Cont);
          }
        }
      }
    }
  }

  // We know that the pointer points to a type of the correct size, unless the
  // size is a VLA or Objective-C interface.
  llvm::Value *Address = 0;
  if (const VariableArrayType *VAT =
        getContext().getAsVariableArrayType(E->getType())) {
    llvm::Value *VLASize = GetVLASize(VAT);

    Idx = Builder.CreateMul(Idx, VLASize);

    QualType BaseType = getContext().getBaseElementType(VAT);

    CharUnits BaseTypeSize = getContext().getTypeSizeInChars(BaseType);
    Idx = Builder.CreateUDiv(Idx,
                             llvm::ConstantInt::get(Idx->getType(),
                                 BaseTypeSize.getQuantity()));
    Address = Builder.CreateInBoundsGEP(Base, Idx, "arrayidx");
  } else if (const ObjCObjectType *OIT =
               E->getType()->getAs<ObjCObjectType>()) {
    llvm::Value *InterfaceSize =
      llvm::ConstantInt::get(Idx->getType(),
          getContext().getTypeSizeInChars(OIT).getQuantity());

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
  llvm::SmallVector<llvm::Constant*, 4> CElts;

  for (unsigned i = 0, e = Elts.size(); i != e; ++i)
    CElts.push_back(llvm::ConstantInt::get(
                                   llvm::Type::getInt32Ty(VMContext), Elts[i]));

  return llvm::ConstantVector::get(&CElts[0], CElts.size());
}

LValue CodeGenFunction::
EmitExtVectorElementExpr(const ExtVectorElementExpr *E) {
  const llvm::Type *Int32Ty = llvm::Type::getInt32Ty(VMContext);

  // Emit the base vector as an l-value.
  LValue Base;

  // ExtVectorElementExpr's base can either be a vector or pointer to vector.
  if (E->isArrow()) {
    // If it is a pointer to a vector, emit the address and form an lvalue with
    // it.
    llvm::Value *Ptr = EmitScalarExpr(E->getBase());
    const PointerType *PT = E->getBase()->getType()->getAs<PointerType>();
    Qualifiers Quals = MakeQualifiers(PT->getPointeeType());
    Quals.removeObjCGCAttr();
    Base = LValue::MakeAddr(Ptr, Quals);
  } else if (E->getBase()->isLvalue(getContext()) == Expr::LV_Valid) {
    // Otherwise, if the base is an lvalue ( as in the case of foo.x.x),
    // emit the base as an lvalue.
    assert(E->getBase()->getType()->isVectorType());
    Base = EmitLValue(E->getBase());
  } else {
    // Otherwise, the base is a normal rvalue (as in (V+V).x), emit it as such.
    assert(E->getBase()->getType()->getAs<VectorType>() &&
           "Result must be a vector");
    llvm::Value *Vec = EmitScalarExpr(E->getBase());
    
    // Store the vector to memory (because LValue wants an address).
    llvm::Value *VecMem = CreateMemTemp(E->getBase()->getType());
    Builder.CreateStore(Vec, VecMem);
    Base = LValue::MakeAddr(VecMem, Qualifiers());
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
  bool isNonGC = false;
  Expr *BaseExpr = E->getBase();
  llvm::Value *BaseValue = NULL;
  Qualifiers BaseQuals;

  // If this is s.x, emit s as an lvalue.  If it is s->x, emit s as a scalar.
  if (E->isArrow()) {
    BaseValue = EmitScalarExpr(BaseExpr);
    const PointerType *PTy =
      BaseExpr->getType()->getAs<PointerType>();
    BaseQuals = PTy->getPointeeType().getQualifiers();
  } else if (isa<ObjCPropertyRefExpr>(BaseExpr->IgnoreParens()) ||
             isa<ObjCImplicitSetterGetterRefExpr>(
               BaseExpr->IgnoreParens())) {
    RValue RV = EmitObjCPropertyGet(BaseExpr);
    BaseValue = RV.getAggregateAddr();
    BaseQuals = BaseExpr->getType().getQualifiers();
  } else {
    LValue BaseLV = EmitLValue(BaseExpr);
    if (BaseLV.isNonGC())
      isNonGC = true;
    // FIXME: this isn't right for bitfields.
    BaseValue = BaseLV.getAddress();
    QualType BaseTy = BaseExpr->getType();
    BaseQuals = BaseTy.getQualifiers();
  }

  NamedDecl *ND = E->getMemberDecl();
  if (FieldDecl *Field = dyn_cast<FieldDecl>(ND)) {
    LValue LV = EmitLValueForField(BaseValue, Field, 
                                   BaseQuals.getCVRQualifiers());
    LValue::SetObjCNonGC(LV, isNonGC);
    setObjCGCLValueClass(getContext(), E, LV);
    return LV;
  }
  
  if (VarDecl *VD = dyn_cast<VarDecl>(ND))
    return EmitGlobalVarDeclLValue(*this, E, VD);

  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(ND))
    return EmitFunctionDeclLValue(*this, E, FD);

  assert(false && "Unhandled member declaration!");
  return LValue();
}

LValue CodeGenFunction::EmitLValueForBitfield(llvm::Value* BaseValue,
                                              const FieldDecl* Field,
                                              unsigned CVRQualifiers) {
  const CGRecordLayout &RL =
    CGM.getTypes().getCGRecordLayout(Field->getParent());
  const CGBitFieldInfo &Info = RL.getBitFieldInfo(Field);
  return LValue::MakeBitfield(BaseValue, Info,
                             Field->getType().getCVRQualifiers()|CVRQualifiers);
}

/// EmitLValueForAnonRecordField - Given that the field is a member of
/// an anonymous struct or union buried inside a record, and given
/// that the base value is a pointer to the enclosing record, derive
/// an lvalue for the ultimate field.
LValue CodeGenFunction::EmitLValueForAnonRecordField(llvm::Value *BaseValue,
                                                     const FieldDecl *Field,
                                                     unsigned CVRQualifiers) {
  llvm::SmallVector<const FieldDecl *, 8> Path;
  Path.push_back(Field);

  while (Field->getParent()->isAnonymousStructOrUnion()) {
    const ValueDecl *VD = Field->getParent()->getAnonymousStructOrUnionObject();
    if (!isa<FieldDecl>(VD)) break;
    Field = cast<FieldDecl>(VD);
    Path.push_back(Field);
  }

  llvm::SmallVectorImpl<const FieldDecl*>::reverse_iterator
    I = Path.rbegin(), E = Path.rend();
  while (true) {
    LValue LV = EmitLValueForField(BaseValue, *I, CVRQualifiers);
    if (++I == E) return LV;

    assert(LV.isSimple());
    BaseValue = LV.getAddress();
    CVRQualifiers |= LV.getVRQualifiers();
  }
}

LValue CodeGenFunction::EmitLValueForField(llvm::Value* BaseValue,
                                           const FieldDecl* Field,
                                           unsigned CVRQualifiers) {
  if (Field->isBitField())
    return EmitLValueForBitfield(BaseValue, Field, CVRQualifiers);

  const CGRecordLayout &RL =
    CGM.getTypes().getCGRecordLayout(Field->getParent());
  unsigned idx = RL.getLLVMFieldNo(Field);
  llvm::Value *V = Builder.CreateStructGEP(BaseValue, idx, "tmp");

  // Match union field type.
  if (Field->getParent()->isUnion()) {
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

LValue 
CodeGenFunction::EmitLValueForFieldInitialization(llvm::Value* BaseValue, 
                                                  const FieldDecl* Field,
                                                  unsigned CVRQualifiers) {
  QualType FieldType = Field->getType();
  
  if (!FieldType->isReferenceType())
    return EmitLValueForField(BaseValue, Field, CVRQualifiers);

  const CGRecordLayout &RL =
    CGM.getTypes().getCGRecordLayout(Field->getParent());
  unsigned idx = RL.getLLVMFieldNo(Field);
  llvm::Value *V = Builder.CreateStructGEP(BaseValue, idx, "tmp");

  assert(!FieldType.getObjCGCAttr() && "fields cannot have GC attrs");

  return LValue::MakeAddr(V, MakeQualifiers(FieldType));
}

LValue CodeGenFunction::EmitCompoundLiteralLValue(const CompoundLiteralExpr* E){
  llvm::Value *DeclPtr = CreateMemTemp(E->getType(), ".compoundliteral");
  const Expr* InitExpr = E->getInitializer();
  LValue Result = LValue::MakeAddr(DeclPtr, MakeQualifiers(E->getType()));

  EmitAnyExprToMem(InitExpr, DeclPtr, /*Volatile*/ false);

  return Result;
}

LValue 
CodeGenFunction::EmitConditionalOperatorLValue(const ConditionalOperator* E) {
  if (E->isLvalue(getContext()) == Expr::LV_Valid) {
    if (int Cond = ConstantFoldsToSimpleInteger(E->getCond())) {
      Expr *Live = Cond == 1 ? E->getLHS() : E->getRHS();
      if (Live)
        return EmitLValue(Live);
    }

    if (!E->getLHS())
      return EmitUnsupportedLValue(E, "conditional operator with missing LHS");

    llvm::BasicBlock *LHSBlock = createBasicBlock("cond.true");
    llvm::BasicBlock *RHSBlock = createBasicBlock("cond.false");
    llvm::BasicBlock *ContBlock = createBasicBlock("cond.end");
    
    EmitBranchOnBoolExpr(E->getCond(), LHSBlock, RHSBlock);
    
    // Any temporaries created here are conditional.
    BeginConditionalBranch();
    EmitBlock(LHSBlock);
    LValue LHS = EmitLValue(E->getLHS());
    EndConditionalBranch();
    
    if (!LHS.isSimple())
      return EmitUnsupportedLValue(E, "conditional operator");

    // FIXME: We shouldn't need an alloca for this.
    llvm::Value *Temp = CreateTempAlloca(LHS.getAddress()->getType(),"condtmp");
    Builder.CreateStore(LHS.getAddress(), Temp);
    EmitBranch(ContBlock);
    
    // Any temporaries created here are conditional.
    BeginConditionalBranch();
    EmitBlock(RHSBlock);
    LValue RHS = EmitLValue(E->getRHS());
    EndConditionalBranch();
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

  return EmitAggExprToLValue(E);
}

/// EmitCastLValue - Casts are never lvalues unless that cast is a dynamic_cast.
/// If the cast is a dynamic_cast, we can have the usual lvalue result,
/// otherwise if a cast is needed by the code generator in an lvalue context,
/// then it must mean that we need the address of an aggregate in order to
/// access one of its fields.  This can happen for all the reasons that casts
/// are permitted with aggregate result, including noop aggregate casts, and
/// cast from scalar to union.
LValue CodeGenFunction::EmitCastLValue(const CastExpr *E) {
  switch (E->getCastKind()) {
  default:
    return EmitUnsupportedLValue(E, "unexpected cast lvalue");

  case CastExpr::CK_Dynamic: {
    LValue LV = EmitLValue(E->getSubExpr());
    llvm::Value *V = LV.getAddress();
    const CXXDynamicCastExpr *DCE = cast<CXXDynamicCastExpr>(E);
    return LValue::MakeAddr(EmitDynamicCast(V, DCE),
                            MakeQualifiers(E->getType()));
  }

  case CastExpr::CK_NoOp: {
    LValue LV = EmitLValue(E->getSubExpr());
    if (LV.isPropertyRef()) {
      QualType QT = E->getSubExpr()->getType();
      RValue RV = EmitLoadOfPropertyRefLValue(LV, QT);
      assert(!RV.isScalar() && "EmitCastLValue - scalar cast of property ref");
      llvm::Value *V = RV.getAggregateAddr();
      return LValue::MakeAddr(V, MakeQualifiers(QT));
    }
    return LV;
  }
  case CastExpr::CK_ConstructorConversion:
  case CastExpr::CK_UserDefinedConversion:
  case CastExpr::CK_AnyPointerToObjCPointerCast:
    return EmitLValue(E->getSubExpr());
  
  case CastExpr::CK_UncheckedDerivedToBase:
  case CastExpr::CK_DerivedToBase: {
    const RecordType *DerivedClassTy = 
      E->getSubExpr()->getType()->getAs<RecordType>();
    CXXRecordDecl *DerivedClassDecl = 
      cast<CXXRecordDecl>(DerivedClassTy->getDecl());
    
    LValue LV = EmitLValue(E->getSubExpr());
    
    // Perform the derived-to-base conversion
    llvm::Value *Base = 
      GetAddressOfBaseClass(LV.getAddress(), DerivedClassDecl, 
                            E->getBasePath(), /*NullCheckValue=*/false);
    
    return LValue::MakeAddr(Base, MakeQualifiers(E->getType()));
  }
  case CastExpr::CK_ToUnion:
    return EmitAggExprToLValue(E);
  case CastExpr::CK_BaseToDerived: {
    const RecordType *DerivedClassTy = E->getType()->getAs<RecordType>();
    CXXRecordDecl *DerivedClassDecl = 
      cast<CXXRecordDecl>(DerivedClassTy->getDecl());
    
    LValue LV = EmitLValue(E->getSubExpr());
    
    // Perform the base-to-derived conversion
    llvm::Value *Derived = 
      GetAddressOfDerivedClass(LV.getAddress(), DerivedClassDecl, 
                               E->getBasePath(),/*NullCheckValue=*/false);
    
    return LValue::MakeAddr(Derived, MakeQualifiers(E->getType()));
  }
  case CastExpr::CK_BitCast: {
    // This must be a reinterpret_cast (or c-style equivalent).
    const ExplicitCastExpr *CE = cast<ExplicitCastExpr>(E);
    
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
  LValue LV = LValue::MakeAddr(CreateMemTemp(Ty), MakeQualifiers(Ty));
  EmitNullInitialization(LV.getAddress(), Ty);
  return LV;
}

//===--------------------------------------------------------------------===//
//                             Expression Emission
//===--------------------------------------------------------------------===//


RValue CodeGenFunction::EmitCallExpr(const CallExpr *E, 
                                     ReturnValueSlot ReturnValue) {
  // Builtins never have block type.
  if (E->getCallee()->getType()->isBlockPointerType())
    return EmitBlockCallExpr(E, ReturnValue);

  if (const CXXMemberCallExpr *CE = dyn_cast<CXXMemberCallExpr>(E))
    return EmitCXXMemberCallExpr(CE, ReturnValue);

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
      return EmitCXXOperatorMemberCallExpr(CE, MD, ReturnValue);

  if (isa<CXXPseudoDestructorExpr>(E->getCallee()->IgnoreParens())) {
    // C++ [expr.pseudo]p1:
    //   The result shall only be used as the operand for the function call
    //   operator (), and the result of such a call has type void. The only
    //   effect is the evaluation of the postfix-expression before the dot or
    //   arrow.
    EmitScalarExpr(E->getCallee());
    return RValue::get(0);
  }

  llvm::Value *Callee = EmitScalarExpr(E->getCallee());
  return EmitCall(E->getCallee()->getType(), Callee, ReturnValue,
                  E->arg_begin(), E->arg_end(), TargetDecl);
}

LValue CodeGenFunction::EmitBinaryOperatorLValue(const BinaryOperator *E) {
  // Comma expressions just emit their LHS then their RHS as an l-value.
  if (E->getOpcode() == BinaryOperator::Comma) {
    EmitAnyExpr(E->getLHS());
    EnsureInsertPoint();
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
  
  return EmitAggExprToLValue(E);
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
  return EmitAggExprToLValue(E);
}

LValue CodeGenFunction::EmitCXXConstructLValue(const CXXConstructExpr *E) {
  llvm::Value *Temp = CreateMemTemp(E->getType(), "tmp");
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
  return LValue::MakeAddr(RV.getAggregateAddr(), MakeQualifiers(E->getType()));
}

RValue CodeGenFunction::EmitCall(QualType CalleeType, llvm::Value *Callee,
                                 ReturnValueSlot ReturnValue,
                                 CallExpr::const_arg_iterator ArgBeg,
                                 CallExpr::const_arg_iterator ArgEnd,
                                 const Decl *TargetDecl) {
  // Get the actual function type. The callee type will always be a pointer to
  // function type or a block pointer type.
  assert(CalleeType->isFunctionPointerType() &&
         "Call must have function pointer type!");

  CalleeType = getContext().getCanonicalType(CalleeType);

  const FunctionType *FnType
    = cast<FunctionType>(cast<PointerType>(CalleeType)->getPointeeType());
  QualType ResultType = FnType->getResultType();

  CallArgList Args;
  EmitCallArgs(Args, dyn_cast<FunctionProtoType>(FnType), ArgBeg, ArgEnd);

  return EmitCall(CGM.getTypes().getFunctionInfo(Args, FnType),
                  Callee, ReturnValue, Args, TargetDecl);
}

LValue CodeGenFunction::
EmitPointerToDataMemberBinaryExpr(const BinaryOperator *E) {
  llvm::Value *BaseV;
  if (E->getOpcode() == BinaryOperator::PtrMemI)
    BaseV = EmitScalarExpr(E->getLHS());
  else
    BaseV = EmitLValue(E->getLHS()).getAddress();
  const llvm::Type *i8Ty = llvm::Type::getInt8PtrTy(getLLVMContext());
  BaseV = Builder.CreateBitCast(BaseV, i8Ty);
  llvm::Value *OffsetV = EmitScalarExpr(E->getRHS());
  llvm::Value *AddV = Builder.CreateInBoundsGEP(BaseV, OffsetV, "add.ptr");

  QualType Ty = E->getRHS()->getType();
  Ty = Ty->getAs<MemberPointerType>()->getPointeeType();
  
  const llvm::Type *PType = ConvertType(getContext().getPointerType(Ty));
  AddV = Builder.CreateBitCast(AddV, PType);
  return LValue::MakeAddr(AddV, MakeQualifiers(Ty));
}

