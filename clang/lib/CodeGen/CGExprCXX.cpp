//===--- CGExprCXX.cpp - Emit LLVM Code for C++ expressions ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with code generation of C++ expressions
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
using namespace clang;
using namespace CodeGen;

static uint64_t CalculateCookiePadding(ASTContext &Ctx, QualType ElementType) {
  const RecordType *RT = ElementType->getAs<RecordType>();
  if (!RT)
    return 0;
  
  const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(RT->getDecl());
  if (!RD)
    return 0;
  
  // Check if the class has a trivial destructor.
  if (RD->hasTrivialDestructor()) {
    // Check if the usual deallocation function takes two arguments.
    const CXXMethodDecl *UsualDeallocationFunction = 0;
    
    DeclarationName OpName =
      Ctx.DeclarationNames.getCXXOperatorName(OO_Array_Delete);
    DeclContext::lookup_const_iterator Op, OpEnd;
    for (llvm::tie(Op, OpEnd) = RD->lookup(OpName);
         Op != OpEnd; ++Op) {
      const CXXMethodDecl *Delete = cast<CXXMethodDecl>(*Op);

      if (Delete->isUsualDeallocationFunction()) {
        UsualDeallocationFunction = Delete;
        break;
      }
    }
    
    // No usual deallocation function, we don't need a cookie.
    if (!UsualDeallocationFunction)
      return 0;
    
    // The usual deallocation function doesn't take a size_t argument, so we
    // don't need a cookie.
    if (UsualDeallocationFunction->getNumParams() == 1)
      return 0;
        
    assert(UsualDeallocationFunction->getNumParams() == 2 && 
           "Unexpected deallocation function type!");
  }  
  
  // Padding is the maximum of sizeof(size_t) and alignof(ElementType)
  return std::max(Ctx.getTypeSize(Ctx.getSizeType()),
                  static_cast<uint64_t>(Ctx.getTypeAlign(ElementType))) / 8;
}

static uint64_t CalculateCookiePadding(ASTContext &Ctx, const CXXNewExpr *E) {
  if (!E->isArray())
    return 0;

  // No cookie is required if the new operator being used is 
  // ::operator new[](size_t, void*).
  const FunctionDecl *OperatorNew = E->getOperatorNew();
  if (OperatorNew->getDeclContext()->getLookupContext()->isFileContext()) {
    if (OperatorNew->getNumParams() == 2) {
      CanQualType ParamType = 
        Ctx.getCanonicalType(OperatorNew->getParamDecl(1)->getType());
      
      if (ParamType == Ctx.VoidPtrTy)
        return 0;
    }
  }
      
  return CalculateCookiePadding(Ctx, E->getAllocatedType());
  QualType T = E->getAllocatedType();
}

static llvm::Value *EmitCXXNewAllocSize(CodeGenFunction &CGF, 
                                        const CXXNewExpr *E,
                                        llvm::Value *& NumElements) {
  QualType Type = E->getAllocatedType();
  uint64_t TypeSizeInBytes = CGF.getContext().getTypeSize(Type) / 8;
  const llvm::Type *SizeTy = CGF.ConvertType(CGF.getContext().getSizeType());
  
  if (!E->isArray())
    return llvm::ConstantInt::get(SizeTy, TypeSizeInBytes);

  uint64_t CookiePadding = CalculateCookiePadding(CGF.getContext(), E);
  
  Expr::EvalResult Result;
  if (E->getArraySize()->Evaluate(Result, CGF.getContext()) &&
      !Result.HasSideEffects && Result.Val.isInt()) {

    uint64_t AllocSize = 
      Result.Val.getInt().getZExtValue() * TypeSizeInBytes + CookiePadding;
    
    NumElements = 
      llvm::ConstantInt::get(SizeTy, Result.Val.getInt().getZExtValue());
    
    return llvm::ConstantInt::get(SizeTy, AllocSize);
  }
  
  // Emit the array size expression.
  NumElements = CGF.EmitScalarExpr(E->getArraySize());
  
  // Multiply with the type size.
  llvm::Value *V = 
    CGF.Builder.CreateMul(NumElements, 
                          llvm::ConstantInt::get(SizeTy, TypeSizeInBytes));

  // And add the cookie padding if necessary.
  if (CookiePadding)
    V = CGF.Builder.CreateAdd(V, llvm::ConstantInt::get(SizeTy, CookiePadding));
  
  return V;
}

static void EmitNewInitializer(CodeGenFunction &CGF, const CXXNewExpr *E,
                               llvm::Value *NewPtr,
                               llvm::Value *NumElements) {
  if (E->isArray()) {
    if (CXXConstructorDecl *Ctor = E->getConstructor())
      CGF.EmitCXXAggrConstructorCall(Ctor, NumElements, NewPtr, 
                                     E->constructor_arg_begin(), 
                                     E->constructor_arg_end());
    return;
  }
  
  QualType AllocType = E->getAllocatedType();

  if (CXXConstructorDecl *Ctor = E->getConstructor()) {
    CGF.EmitCXXConstructorCall(Ctor, Ctor_Complete, NewPtr,
                               E->constructor_arg_begin(),
                               E->constructor_arg_end());

    return;
  }
    
  // We have a POD type.
  if (E->getNumConstructorArgs() == 0)
    return;

  assert(E->getNumConstructorArgs() == 1 &&
         "Can only have one argument to initializer of POD type.");
      
  const Expr *Init = E->getConstructorArg(0);
    
  if (!CGF.hasAggregateLLVMType(AllocType)) 
    CGF.EmitStoreOfScalar(CGF.EmitScalarExpr(Init), NewPtr,
                          AllocType.isVolatileQualified(), AllocType);
  else if (AllocType->isAnyComplexType())
    CGF.EmitComplexExprIntoAddr(Init, NewPtr, 
                                AllocType.isVolatileQualified());
  else
    CGF.EmitAggExpr(Init, NewPtr, AllocType.isVolatileQualified());
}

llvm::Value *CodeGenFunction::EmitCXXNewExpr(const CXXNewExpr *E) {
  QualType AllocType = E->getAllocatedType();
  FunctionDecl *NewFD = E->getOperatorNew();
  const FunctionProtoType *NewFTy = NewFD->getType()->getAs<FunctionProtoType>();

  CallArgList NewArgs;

  // The allocation size is the first argument.
  QualType SizeTy = getContext().getSizeType();

  llvm::Value *NumElements = 0;
  llvm::Value *AllocSize = EmitCXXNewAllocSize(*this, E, NumElements);
  
  NewArgs.push_back(std::make_pair(RValue::get(AllocSize), SizeTy));

  // Emit the rest of the arguments.
  // FIXME: Ideally, this should just use EmitCallArgs.
  CXXNewExpr::const_arg_iterator NewArg = E->placement_arg_begin();

  // First, use the types from the function type.
  // We start at 1 here because the first argument (the allocation size)
  // has already been emitted.
  for (unsigned i = 1, e = NewFTy->getNumArgs(); i != e; ++i, ++NewArg) {
    QualType ArgType = NewFTy->getArgType(i);

    assert(getContext().getCanonicalType(ArgType.getNonReferenceType()).
           getTypePtr() ==
           getContext().getCanonicalType(NewArg->getType()).getTypePtr() &&
           "type mismatch in call argument!");

    NewArgs.push_back(std::make_pair(EmitCallArg(*NewArg, ArgType),
                                     ArgType));

  }

  // Either we've emitted all the call args, or we have a call to a
  // variadic function.
  assert((NewArg == E->placement_arg_end() || NewFTy->isVariadic()) &&
         "Extra arguments in non-variadic function!");

  // If we still have any arguments, emit them using the type of the argument.
  for (CXXNewExpr::const_arg_iterator NewArgEnd = E->placement_arg_end();
       NewArg != NewArgEnd; ++NewArg) {
    QualType ArgType = NewArg->getType();
    NewArgs.push_back(std::make_pair(EmitCallArg(*NewArg, ArgType),
                                     ArgType));
  }

  // Emit the call to new.
  RValue RV =
    EmitCall(CGM.getTypes().getFunctionInfo(NewFTy->getResultType(), NewArgs),
             CGM.GetAddrOfFunction(NewFD), ReturnValueSlot(), NewArgs, NewFD);

  // If an allocation function is declared with an empty exception specification
  // it returns null to indicate failure to allocate storage. [expr.new]p13.
  // (We don't need to check for null when there's no new initializer and
  // we're allocating a POD type).
  bool NullCheckResult = NewFTy->hasEmptyExceptionSpec() &&
    !(AllocType->isPODType() && !E->hasInitializer());

  llvm::BasicBlock *NewNull = 0;
  llvm::BasicBlock *NewNotNull = 0;
  llvm::BasicBlock *NewEnd = 0;

  llvm::Value *NewPtr = RV.getScalarVal();

  if (NullCheckResult) {
    NewNull = createBasicBlock("new.null");
    NewNotNull = createBasicBlock("new.notnull");
    NewEnd = createBasicBlock("new.end");

    llvm::Value *IsNull =
      Builder.CreateICmpEQ(NewPtr,
                           llvm::Constant::getNullValue(NewPtr->getType()),
                           "isnull");

    Builder.CreateCondBr(IsNull, NewNull, NewNotNull);
    EmitBlock(NewNotNull);
  }

  if (uint64_t CookiePadding = CalculateCookiePadding(getContext(), E)) {
    uint64_t CookieOffset = 
      CookiePadding - getContext().getTypeSize(SizeTy) / 8;
    
    llvm::Value *NumElementsPtr = 
      Builder.CreateConstInBoundsGEP1_64(NewPtr, CookieOffset);
    
    NumElementsPtr = Builder.CreateBitCast(NumElementsPtr, 
                                           ConvertType(SizeTy)->getPointerTo());
    Builder.CreateStore(NumElements, NumElementsPtr);

    // Now add the padding to the new ptr.
    NewPtr = Builder.CreateConstInBoundsGEP1_64(NewPtr, CookiePadding);
  }
  
  NewPtr = Builder.CreateBitCast(NewPtr, ConvertType(E->getType()));

  EmitNewInitializer(*this, E, NewPtr, NumElements);

  if (NullCheckResult) {
    Builder.CreateBr(NewEnd);
    NewNotNull = Builder.GetInsertBlock();
    EmitBlock(NewNull);
    Builder.CreateBr(NewEnd);
    EmitBlock(NewEnd);

    llvm::PHINode *PHI = Builder.CreatePHI(NewPtr->getType());
    PHI->reserveOperandSpace(2);
    PHI->addIncoming(NewPtr, NewNotNull);
    PHI->addIncoming(llvm::Constant::getNullValue(NewPtr->getType()), NewNull);

    NewPtr = PHI;
  }

  return NewPtr;
}

static std::pair<llvm::Value *, llvm::Value *>
GetAllocatedObjectPtrAndNumElements(CodeGenFunction &CGF,
                                    llvm::Value *Ptr, QualType DeleteTy) {
  QualType SizeTy = CGF.getContext().getSizeType();
  const llvm::Type *SizeLTy = CGF.ConvertType(SizeTy);
  
  uint64_t DeleteTypeAlign = CGF.getContext().getTypeAlign(DeleteTy);
  uint64_t CookiePadding = std::max(CGF.getContext().getTypeSize(SizeTy),
                                    DeleteTypeAlign) / 8;
  assert(CookiePadding && "CookiePadding should not be 0.");

  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(CGF.getLLVMContext());
  uint64_t CookieOffset = 
    CookiePadding - CGF.getContext().getTypeSize(SizeTy) / 8;

  llvm::Value *AllocatedObjectPtr = CGF.Builder.CreateBitCast(Ptr, Int8PtrTy);
  AllocatedObjectPtr = 
    CGF.Builder.CreateConstInBoundsGEP1_64(AllocatedObjectPtr,
                                           -CookiePadding);

  llvm::Value *NumElementsPtr =
    CGF.Builder.CreateConstInBoundsGEP1_64(AllocatedObjectPtr, 
                                           CookieOffset);
  NumElementsPtr = 
    CGF.Builder.CreateBitCast(NumElementsPtr, SizeLTy->getPointerTo());
  
  llvm::Value *NumElements = CGF.Builder.CreateLoad(NumElementsPtr);
  NumElements = 
    CGF.Builder.CreateIntCast(NumElements, SizeLTy, /*isSigned=*/false);
  
  return std::make_pair(AllocatedObjectPtr, NumElements);
}

void CodeGenFunction::EmitDeleteCall(const FunctionDecl *DeleteFD,
                                     llvm::Value *Ptr,
                                     QualType DeleteTy) {
  const FunctionProtoType *DeleteFTy =
    DeleteFD->getType()->getAs<FunctionProtoType>();

  CallArgList DeleteArgs;

  // Check if we need to pass the size to the delete operator.
  llvm::Value *Size = 0;
  QualType SizeTy;
  if (DeleteFTy->getNumArgs() == 2) {
    SizeTy = DeleteFTy->getArgType(1);
    uint64_t DeleteTypeSize = getContext().getTypeSize(DeleteTy) / 8;
    Size = llvm::ConstantInt::get(ConvertType(SizeTy), DeleteTypeSize);
  }
  
  if (DeleteFD->getOverloadedOperator() == OO_Array_Delete &&
      
      CalculateCookiePadding(getContext(), DeleteTy)) {
    // We need to get the number of elements in the array from the cookie.
    llvm::Value *AllocatedObjectPtr;
    llvm::Value *NumElements;
    llvm::tie(AllocatedObjectPtr, NumElements) =
      GetAllocatedObjectPtrAndNumElements(*this, Ptr, DeleteTy);
    
    // Multiply the size with the number of elements.
    if (Size)
      Size = Builder.CreateMul(NumElements, Size);
    
    Ptr = AllocatedObjectPtr;
  }
  
  QualType ArgTy = DeleteFTy->getArgType(0);
  llvm::Value *DeletePtr = Builder.CreateBitCast(Ptr, ConvertType(ArgTy));
  DeleteArgs.push_back(std::make_pair(RValue::get(DeletePtr), ArgTy));

  if (Size)
    DeleteArgs.push_back(std::make_pair(RValue::get(Size), SizeTy));

  // Emit the call to delete.
  EmitCall(CGM.getTypes().getFunctionInfo(DeleteFTy->getResultType(),
                                          DeleteArgs),
           CGM.GetAddrOfFunction(DeleteFD), ReturnValueSlot(), 
           DeleteArgs, DeleteFD);
}

void CodeGenFunction::EmitCXXDeleteExpr(const CXXDeleteExpr *E) {
  
  // Get at the argument before we performed the implicit conversion
  // to void*.
  const Expr *Arg = E->getArgument();
  while (const ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(Arg)) {
    if (ICE->getCastKind() != CastExpr::CK_UserDefinedConversion &&
        ICE->getType()->isVoidPointerType())
      Arg = ICE->getSubExpr();
    else
      break;
  }
  
  QualType DeleteTy = Arg->getType()->getAs<PointerType>()->getPointeeType();

  llvm::Value *Ptr = EmitScalarExpr(Arg);

  // Null check the pointer.
  llvm::BasicBlock *DeleteNotNull = createBasicBlock("delete.notnull");
  llvm::BasicBlock *DeleteEnd = createBasicBlock("delete.end");

  llvm::Value *IsNull =
    Builder.CreateICmpEQ(Ptr, llvm::Constant::getNullValue(Ptr->getType()),
                         "isnull");

  Builder.CreateCondBr(IsNull, DeleteEnd, DeleteNotNull);
  EmitBlock(DeleteNotNull);
  
  bool ShouldCallDelete = true;
  
  // Call the destructor if necessary.
  if (const RecordType *RT = DeleteTy->getAs<RecordType>()) {
    if (CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(RT->getDecl())) {
      if (!RD->hasTrivialDestructor()) {
        const CXXDestructorDecl *Dtor = RD->getDestructor(getContext());
        if (E->isArrayForm()) {
          llvm::Value *AllocatedObjectPtr;
          llvm::Value *NumElements;
          llvm::tie(AllocatedObjectPtr, NumElements) =
            GetAllocatedObjectPtrAndNumElements(*this, Ptr, DeleteTy);
          
          EmitCXXAggrDestructorCall(Dtor, NumElements, Ptr);
        } else if (Dtor->isVirtual()) {
          const llvm::Type *Ty =
            CGM.getTypes().GetFunctionType(CGM.getTypes().getFunctionInfo(Dtor),
                                           /*isVariadic=*/false);
          
          llvm::Value *Callee = BuildVirtualCall(Dtor, Dtor_Deleting, Ptr, Ty);
          EmitCXXMemberCall(Dtor, Callee, Ptr, 0, 0);

          // The dtor took care of deleting the object.
          ShouldCallDelete = false;
        } else 
          EmitCXXDestructorCall(Dtor, Dtor_Complete, Ptr);
      }
    }
  }

  if (ShouldCallDelete)
    EmitDeleteCall(E->getOperatorDelete(), Ptr, DeleteTy);

  EmitBlock(DeleteEnd);
}

llvm::Value * CodeGenFunction::EmitCXXTypeidExpr(const CXXTypeidExpr *E) {
  QualType Ty = E->getType();
  const llvm::Type *LTy = ConvertType(Ty)->getPointerTo();
  
  if (E->isTypeOperand()) {
    llvm::Constant *TypeInfo = 
      CGM.GetAddrOfRTTIDescriptor(E->getTypeOperand());
    return Builder.CreateBitCast(TypeInfo, LTy);
  }
  
  Expr *subE = E->getExprOperand();
  Ty = subE->getType();
  CanQualType CanTy = CGM.getContext().getCanonicalType(Ty);
  Ty = CanTy.getUnqualifiedType().getNonReferenceType();
  if (const RecordType *RT = Ty->getAs<RecordType>()) {
    const CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
    if (RD->isPolymorphic()) {
      // FIXME: if subE is an lvalue do
      LValue Obj = EmitLValue(subE);
      llvm::Value *This = Obj.getAddress();
      LTy = LTy->getPointerTo()->getPointerTo();
      llvm::Value *V = Builder.CreateBitCast(This, LTy);
      // We need to do a zero check for *p, unless it has NonNullAttr.
      // FIXME: PointerType->hasAttr<NonNullAttr>()
      bool CanBeZero = false;
      if (UnaryOperator *UO = dyn_cast<UnaryOperator>(subE->IgnoreParens()))
        if (UO->getOpcode() == UnaryOperator::Deref)
          CanBeZero = true;
      if (CanBeZero) {
        llvm::BasicBlock *NonZeroBlock = createBasicBlock();
        llvm::BasicBlock *ZeroBlock = createBasicBlock();
        
        llvm::Value *Zero = llvm::Constant::getNullValue(LTy);
        Builder.CreateCondBr(Builder.CreateICmpNE(V, Zero),
                             NonZeroBlock, ZeroBlock);
        EmitBlock(ZeroBlock);
        /// Call __cxa_bad_typeid
        const llvm::Type *ResultType = llvm::Type::getVoidTy(VMContext);
        const llvm::FunctionType *FTy;
        FTy = llvm::FunctionType::get(ResultType, false);
        llvm::Value *F = CGM.CreateRuntimeFunction(FTy, "__cxa_bad_typeid");
        Builder.CreateCall(F)->setDoesNotReturn();
        Builder.CreateUnreachable();
        EmitBlock(NonZeroBlock);
      }
      V = Builder.CreateLoad(V, "vtable");
      V = Builder.CreateConstInBoundsGEP1_64(V, -1ULL);
      V = Builder.CreateLoad(V);
      return V;
    }
  }
  return Builder.CreateBitCast(CGM.GetAddrOfRTTIDescriptor(Ty), LTy);
}

llvm::Value *CodeGenFunction::EmitDynamicCast(llvm::Value *V,
                                              const CXXDynamicCastExpr *DCE) {
  QualType SrcTy = DCE->getSubExpr()->getType();
  QualType DestTy = DCE->getTypeAsWritten();
  QualType InnerType = DestTy->getPointeeType();
  
  const llvm::Type *LTy = ConvertType(DCE->getType());

  bool CanBeZero = false;
  bool ToVoid = false;
  bool ThrowOnBad = false;
  if (DestTy->isPointerType()) {
    // FIXME: if PointerType->hasAttr<NonNullAttr>(), we don't set this
    CanBeZero = true;
    if (InnerType->isVoidType())
      ToVoid = true;
  } else {
    LTy = LTy->getPointerTo();
    ThrowOnBad = true;
  }

  if (SrcTy->isPointerType() || SrcTy->isReferenceType())
    SrcTy = SrcTy->getPointeeType();
  SrcTy = SrcTy.getUnqualifiedType();

  if (DestTy->isPointerType() || DestTy->isReferenceType())
    DestTy = DestTy->getPointeeType();
  DestTy = DestTy.getUnqualifiedType();

  llvm::BasicBlock *ContBlock = createBasicBlock();
  llvm::BasicBlock *NullBlock = 0;
  llvm::BasicBlock *NonZeroBlock = 0;
  if (CanBeZero) {
    NonZeroBlock = createBasicBlock();
    NullBlock = createBasicBlock();
    Builder.CreateCondBr(Builder.CreateIsNotNull(V), NonZeroBlock, NullBlock);
    EmitBlock(NonZeroBlock);
  }

  llvm::BasicBlock *BadCastBlock = 0;

  const llvm::Type *PtrDiffTy = ConvertType(getContext().getPointerDiffType());

  // See if this is a dynamic_cast(void*)
  if (ToVoid) {
    llvm::Value *This = V;
    V = Builder.CreateBitCast(This, PtrDiffTy->getPointerTo()->getPointerTo());
    V = Builder.CreateLoad(V, "vtable");
    V = Builder.CreateConstInBoundsGEP1_64(V, -2ULL);
    V = Builder.CreateLoad(V, "offset to top");
    This = Builder.CreateBitCast(This, llvm::Type::getInt8PtrTy(VMContext));
    V = Builder.CreateInBoundsGEP(This, V);
    V = Builder.CreateBitCast(V, LTy);
  } else {
    /// Call __dynamic_cast
    const llvm::Type *ResultType = llvm::Type::getInt8PtrTy(VMContext);
    const llvm::FunctionType *FTy;
    std::vector<const llvm::Type*> ArgTys;
    const llvm::Type *PtrToInt8Ty
      = llvm::Type::getInt8Ty(VMContext)->getPointerTo();
    ArgTys.push_back(PtrToInt8Ty);
    ArgTys.push_back(PtrToInt8Ty);
    ArgTys.push_back(PtrToInt8Ty);
    ArgTys.push_back(PtrDiffTy);
    FTy = llvm::FunctionType::get(ResultType, ArgTys, false);

    // FIXME: Calculate better hint.
    llvm::Value *hint = llvm::ConstantInt::get(PtrDiffTy, -1ULL);
    
    assert(SrcTy->isRecordType() && "Src type must be record type!");
    assert(DestTy->isRecordType() && "Dest type must be record type!");
    
    llvm::Value *SrcArg
      = CGM.GetAddrOfRTTIDescriptor(SrcTy.getUnqualifiedType());
    llvm::Value *DestArg
      = CGM.GetAddrOfRTTIDescriptor(DestTy.getUnqualifiedType());
    
    V = Builder.CreateBitCast(V, PtrToInt8Ty);
    V = Builder.CreateCall4(CGM.CreateRuntimeFunction(FTy, "__dynamic_cast"),
                            V, SrcArg, DestArg, hint);
    V = Builder.CreateBitCast(V, LTy);

    if (ThrowOnBad) {
      BadCastBlock = createBasicBlock();

      Builder.CreateCondBr(Builder.CreateIsNotNull(V), ContBlock, BadCastBlock);
      EmitBlock(BadCastBlock);
      /// Call __cxa_bad_cast
      ResultType = llvm::Type::getVoidTy(VMContext);
      const llvm::FunctionType *FBadTy;
      FBadTy = llvm::FunctionType::get(ResultType, false);
      llvm::Value *F = CGM.CreateRuntimeFunction(FBadTy, "__cxa_bad_cast");
      Builder.CreateCall(F)->setDoesNotReturn();
      Builder.CreateUnreachable();
    }
  }
  
  if (CanBeZero) {
    Builder.CreateBr(ContBlock);
    EmitBlock(NullBlock);
    Builder.CreateBr(ContBlock);
  }
  EmitBlock(ContBlock);
  if (CanBeZero) {
    llvm::PHINode *PHI = Builder.CreatePHI(LTy);
    PHI->reserveOperandSpace(2);
    PHI->addIncoming(V, NonZeroBlock);
    PHI->addIncoming(llvm::Constant::getNullValue(LTy), NullBlock);
    V = PHI;
  }

  return V;
}
