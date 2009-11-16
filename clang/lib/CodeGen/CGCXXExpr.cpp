//===--- CGCXXExpr.cpp - Emit LLVM Code for C++ expressions ---------------===//
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

static uint64_t CalculateCookiePadding(ASTContext &Ctx, const CXXNewExpr *E) {
  if (!E->isArray())
    return 0;
  
  QualType T = E->getAllocatedType();
  
  const RecordType *RT = T->getAs<RecordType>();
  if (!RT)
    return 0;
  
  const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(RT->getDecl());
  if (!RD)
    return 0;
  
  // Check if the class has a trivial destructor.
  if (RD->hasTrivialDestructor()) {
    // FIXME: Check for a two-argument delete.
    return 0;
  }
  
  // Padding is the maximum of sizeof(size_t) and alignof(T)
  return std::max(Ctx.getTypeSize(Ctx.getSizeType()),
                  static_cast<uint64_t>(Ctx.getTypeAlign(T))) / 8;
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
  QualType AllocType = E->getAllocatedType();

  if (!E->isArray()) {
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
      CGF.Builder.CreateStore(CGF.EmitScalarExpr(Init), NewPtr);
    else if (AllocType->isAnyComplexType())
      CGF.EmitComplexExprIntoAddr(Init, NewPtr, 
                                  AllocType.isVolatileQualified());
    else
      CGF.EmitAggExpr(Init, NewPtr, AllocType.isVolatileQualified());
    return;
  }
  
  if (CXXConstructorDecl *Ctor = E->getConstructor())
    CGF.EmitCXXAggrConstructorCall(Ctor, NumElements, NewPtr);
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
             CGM.GetAddrOfFunction(NewFD), NewArgs, NewFD);

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
          QualType SizeTy = getContext().getSizeType();
          uint64_t CookiePadding = std::max(getContext().getTypeSize(SizeTy),
                static_cast<uint64_t>(getContext().getTypeAlign(DeleteTy))) / 8;
          if (CookiePadding) {
            llvm::Type *Ptr8Ty = 
              llvm::PointerType::get(llvm::Type::getInt8Ty(VMContext), 0);
            uint64_t CookieOffset =
              CookiePadding - getContext().getTypeSize(SizeTy) / 8;
            llvm::Value *AllocatedObjectPtr = 
              Builder.CreateConstInBoundsGEP1_64(
                            Builder.CreateBitCast(Ptr, Ptr8Ty), -CookiePadding);
            llvm::Value *NumElementsPtr =
              Builder.CreateConstInBoundsGEP1_64(AllocatedObjectPtr, 
                                                 CookieOffset);
            NumElementsPtr = Builder.CreateBitCast(NumElementsPtr,
                                          ConvertType(SizeTy)->getPointerTo());
            
            llvm::Value *NumElements = 
              Builder.CreateLoad(NumElementsPtr);
            NumElements = 
              Builder.CreateIntCast(NumElements, 
                                    llvm::Type::getInt64Ty(VMContext), false, 
                                    "count.tmp");
            EmitCXXAggrDestructorCall(Dtor, NumElements, Ptr);
            Ptr = AllocatedObjectPtr;
          }
        }
        else if (Dtor->isVirtual()) {
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

  if (ShouldCallDelete) {
    // Call delete.
    FunctionDecl *DeleteFD = E->getOperatorDelete();
    const FunctionProtoType *DeleteFTy =
      DeleteFD->getType()->getAs<FunctionProtoType>();

    CallArgList DeleteArgs;

    QualType ArgTy = DeleteFTy->getArgType(0);
    llvm::Value *DeletePtr = Builder.CreateBitCast(Ptr, ConvertType(ArgTy));
    DeleteArgs.push_back(std::make_pair(RValue::get(DeletePtr), ArgTy));

    if (DeleteFTy->getNumArgs() == 2) {
      QualType SizeTy = DeleteFTy->getArgType(1);
      uint64_t SizeVal = getContext().getTypeSize(DeleteTy) / 8;
      llvm::Constant *Size = llvm::ConstantInt::get(ConvertType(SizeTy),
                                                    SizeVal);
      DeleteArgs.push_back(std::make_pair(RValue::get(Size), SizeTy));
    }

    // Emit the call to delete.
    EmitCall(CGM.getTypes().getFunctionInfo(DeleteFTy->getResultType(),
                                            DeleteArgs),
             CGM.GetAddrOfFunction(DeleteFD),
             DeleteArgs, DeleteFD);
  }
  
  EmitBlock(DeleteEnd);
}

llvm::Value * CodeGenFunction::EmitCXXTypeidExpr(const CXXTypeidExpr *E) {
  QualType Ty = E->getType();
  const llvm::Type *LTy = ConvertType(Ty)->getPointerTo();
  if (E->isTypeOperand()) {
    QualType Ty = E->getTypeOperand();
    CanQualType CanTy = CGM.getContext().getCanonicalType(Ty);
    Ty = CanTy.getUnqualifiedType().getNonReferenceType();
    if (const RecordType *RT = Ty->getAs<RecordType>()) {
      const CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
      if (RD->isPolymorphic())
        return Builder.CreateBitCast(CGM.GenerateRttiRef(RD), LTy);
      return Builder.CreateBitCast(CGM.GenerateRtti(RD), LTy);
    }
    // FIXME: return the rtti for the non-class static type.
    ErrorUnsupported(E, "typeid expression");
    return 0;
  }
  Expr *subE = E->getExprOperand();
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
      if (UnaryOperator *UO = dyn_cast<UnaryOperator>(subE))
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
        Builder.CreateCall(F);
        Builder.CreateUnreachable();
        EmitBlock(NonZeroBlock);
      }
      V = Builder.CreateLoad(V, "vtable");
      V = Builder.CreateConstInBoundsGEP1_64(V, -1ULL);
      V = Builder.CreateLoad(V);
      return V;
    }      
    return CGM.GenerateRtti(RD);
  }
  // FIXME: return rtti for the non-class static type.
  ErrorUnsupported(E, "typeid expression");
  return 0;
}
