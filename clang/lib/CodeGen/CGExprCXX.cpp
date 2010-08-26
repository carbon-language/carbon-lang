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
#include "CGObjCRuntime.h"
#include "llvm/Intrinsics.h"
using namespace clang;
using namespace CodeGen;

RValue CodeGenFunction::EmitCXXMemberCall(const CXXMethodDecl *MD,
                                          llvm::Value *Callee,
                                          ReturnValueSlot ReturnValue,
                                          llvm::Value *This,
                                          llvm::Value *VTT,
                                          CallExpr::const_arg_iterator ArgBeg,
                                          CallExpr::const_arg_iterator ArgEnd) {
  assert(MD->isInstance() &&
         "Trying to emit a member call expr on a static method!");

  const FunctionProtoType *FPT = MD->getType()->getAs<FunctionProtoType>();

  CallArgList Args;

  // Push the this ptr.
  Args.push_back(std::make_pair(RValue::get(This),
                                MD->getThisType(getContext())));

  // If there is a VTT parameter, emit it.
  if (VTT) {
    QualType T = getContext().getPointerType(getContext().VoidPtrTy);
    Args.push_back(std::make_pair(RValue::get(VTT), T));
  }
  
  // And the rest of the call args
  EmitCallArgs(Args, FPT, ArgBeg, ArgEnd);

  QualType ResultType = FPT->getResultType();
  return EmitCall(CGM.getTypes().getFunctionInfo(ResultType, Args,
                                                 FPT->getExtInfo()),
                  Callee, ReturnValue, Args, MD);
}

/// canDevirtualizeMemberFunctionCalls - Checks whether virtual calls on given
/// expr can be devirtualized.
static bool canDevirtualizeMemberFunctionCalls(const Expr *Base) {
  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(Base)) {
    if (const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
      // This is a record decl. We know the type and can devirtualize it.
      return VD->getType()->isRecordType();
    }
    
    return false;
  }
  
  // We can always devirtualize calls on temporary object expressions.
  if (isa<CXXConstructExpr>(Base))
    return true;
  
  // And calls on bound temporaries.
  if (isa<CXXBindTemporaryExpr>(Base))
    return true;
  
  // Check if this is a call expr that returns a record type.
  if (const CallExpr *CE = dyn_cast<CallExpr>(Base))
    return CE->getCallReturnType()->isRecordType();
  
  // We can't devirtualize the call.
  return false;
}

RValue CodeGenFunction::EmitCXXMemberCallExpr(const CXXMemberCallExpr *CE,
                                              ReturnValueSlot ReturnValue) {
  if (isa<BinaryOperator>(CE->getCallee()->IgnoreParens())) 
    return EmitCXXMemberPointerCallExpr(CE, ReturnValue);
      
  const MemberExpr *ME = cast<MemberExpr>(CE->getCallee()->IgnoreParens());
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(ME->getMemberDecl());

  if (MD->isStatic()) {
    // The method is static, emit it as we would a regular call.
    llvm::Value *Callee = CGM.GetAddrOfFunction(MD);
    return EmitCall(getContext().getPointerType(MD->getType()), Callee,
                    ReturnValue, CE->arg_begin(), CE->arg_end());
  }
  
  const FunctionProtoType *FPT = MD->getType()->getAs<FunctionProtoType>();

  const llvm::Type *Ty =
    CGM.getTypes().GetFunctionType(CGM.getTypes().getFunctionInfo(MD),
                                   FPT->isVariadic());
  llvm::Value *This;

  if (ME->isArrow())
    This = EmitScalarExpr(ME->getBase());
  else {
    LValue BaseLV = EmitLValue(ME->getBase());
    This = BaseLV.getAddress();
  }

  if (MD->isCopyAssignment() && MD->isTrivial()) {
    // We don't like to generate the trivial copy assignment operator when
    // it isn't necessary; just produce the proper effect here.
    llvm::Value *RHS = EmitLValue(*CE->arg_begin()).getAddress();
    EmitAggregateCopy(This, RHS, CE->getType());
    return RValue::get(This);
  }

  // C++ [class.virtual]p12:
  //   Explicit qualification with the scope operator (5.1) suppresses the
  //   virtual call mechanism.
  //
  // We also don't emit a virtual call if the base expression has a record type
  // because then we know what the type is.
  llvm::Value *Callee;
  if (const CXXDestructorDecl *Destructor
             = dyn_cast<CXXDestructorDecl>(MD)) {
    if (Destructor->isTrivial())
      return RValue::get(0);
    if (MD->isVirtual() && !ME->hasQualifier() && 
        !canDevirtualizeMemberFunctionCalls(ME->getBase())) {
      Callee = BuildVirtualCall(Destructor, Dtor_Complete, This, Ty); 
    } else {
      Callee = CGM.GetAddrOfFunction(GlobalDecl(Destructor, Dtor_Complete), Ty);
    }
  } else if (MD->isVirtual() && !ME->hasQualifier() && 
             !canDevirtualizeMemberFunctionCalls(ME->getBase())) {
    Callee = BuildVirtualCall(MD, This, Ty); 
  } else {
    Callee = CGM.GetAddrOfFunction(MD, Ty);
  }

  return EmitCXXMemberCall(MD, Callee, ReturnValue, This, /*VTT=*/0,
                           CE->arg_begin(), CE->arg_end());
}

RValue
CodeGenFunction::EmitCXXMemberPointerCallExpr(const CXXMemberCallExpr *E,
                                              ReturnValueSlot ReturnValue) {
  const BinaryOperator *BO =
      cast<BinaryOperator>(E->getCallee()->IgnoreParens());
  const Expr *BaseExpr = BO->getLHS();
  const Expr *MemFnExpr = BO->getRHS();
  
  const MemberPointerType *MPT = 
    MemFnExpr->getType()->getAs<MemberPointerType>();

  const FunctionProtoType *FPT = 
    MPT->getPointeeType()->getAs<FunctionProtoType>();
  const CXXRecordDecl *RD = 
    cast<CXXRecordDecl>(MPT->getClass()->getAs<RecordType>()->getDecl());

  // Get the member function pointer.
  llvm::Value *MemFnPtr = EmitScalarExpr(MemFnExpr);

  // Emit the 'this' pointer.
  llvm::Value *This;
  
  if (BO->getOpcode() == BO_PtrMemI)
    This = EmitScalarExpr(BaseExpr);
  else 
    This = EmitLValue(BaseExpr).getAddress();

  // Ask the ABI to load the callee.  Note that This is modified.
  llvm::Value *Callee =
    CGM.getCXXABI().EmitLoadOfMemberFunctionPointer(CGF, This, MemFnPtr, MPT);
  
  CallArgList Args;

  QualType ThisType = 
    getContext().getPointerType(getContext().getTagDeclType(RD));

  // Push the this ptr.
  Args.push_back(std::make_pair(RValue::get(This), ThisType));
  
  // And the rest of the call args
  EmitCallArgs(Args, FPT, E->arg_begin(), E->arg_end());
  const FunctionType *BO_FPT = BO->getType()->getAs<FunctionProtoType>();
  return EmitCall(CGM.getTypes().getFunctionInfo(Args, BO_FPT), Callee, 
                  ReturnValue, Args);
}

RValue
CodeGenFunction::EmitCXXOperatorMemberCallExpr(const CXXOperatorCallExpr *E,
                                               const CXXMethodDecl *MD,
                                               ReturnValueSlot ReturnValue) {
  assert(MD->isInstance() &&
         "Trying to emit a member call expr on a static method!");
  if (MD->isCopyAssignment()) {
    const CXXRecordDecl *ClassDecl = cast<CXXRecordDecl>(MD->getDeclContext());
    if (ClassDecl->hasTrivialCopyAssignment()) {
      assert(!ClassDecl->hasUserDeclaredCopyAssignment() &&
             "EmitCXXOperatorMemberCallExpr - user declared copy assignment");
      LValue LV = EmitLValue(E->getArg(0));
      llvm::Value *This;
      if (LV.isPropertyRef()) {
        llvm::Value *AggLoc  = CreateMemTemp(E->getArg(1)->getType());
        EmitAggExpr(E->getArg(1), AggLoc, false /*VolatileDest*/);
        EmitObjCPropertySet(LV.getPropertyRefExpr(),
                            RValue::getAggregate(AggLoc, false /*VolatileDest*/));
        return RValue::getAggregate(0, false);
      }
      else
        This = LV.getAddress();
      
      llvm::Value *Src = EmitLValue(E->getArg(1)).getAddress();
      QualType Ty = E->getType();
      EmitAggregateCopy(This, Src, Ty);
      return RValue::get(This);
    }
  }

  const FunctionProtoType *FPT = MD->getType()->getAs<FunctionProtoType>();
  const llvm::Type *Ty =
    CGM.getTypes().GetFunctionType(CGM.getTypes().getFunctionInfo(MD),
                                   FPT->isVariadic());
  LValue LV = EmitLValue(E->getArg(0));
  llvm::Value *This;
  if (LV.isPropertyRef()) {
    RValue RV = EmitLoadOfPropertyRefLValue(LV, E->getArg(0)->getType());
    assert (!RV.isScalar() && "EmitCXXOperatorMemberCallExpr");
    This = RV.getAggregateAddr();
  }
  else
    This = LV.getAddress();

  llvm::Value *Callee;
  if (MD->isVirtual() && !canDevirtualizeMemberFunctionCalls(E->getArg(0)))
    Callee = BuildVirtualCall(MD, This, Ty);
  else
    Callee = CGM.GetAddrOfFunction(MD, Ty);

  return EmitCXXMemberCall(MD, Callee, ReturnValue, This, /*VTT=*/0,
                           E->arg_begin() + 1, E->arg_end());
}

void
CodeGenFunction::EmitCXXConstructExpr(llvm::Value *Dest,
                                      const CXXConstructExpr *E) {
  assert(Dest && "Must have a destination!");
  const CXXConstructorDecl *CD = E->getConstructor();
  
  // If we require zero initialization before (or instead of) calling the
  // constructor, as can be the case with a non-user-provided default
  // constructor, emit the zero initialization now.
  if (E->requiresZeroInitialization())
    EmitNullInitialization(Dest, E->getType());

  
  // If this is a call to a trivial default constructor, do nothing.
  if (CD->isTrivial() && CD->isDefaultConstructor())
    return;
  
  // Code gen optimization to eliminate copy constructor and return
  // its first argument instead, if in fact that argument is a temporary 
  // object.
  if (getContext().getLangOptions().ElideConstructors && E->isElidable()) {
    if (const Expr *Arg = E->getArg(0)->getTemporaryObject()) {
      EmitAggExpr(Arg, Dest, false);
      return;
    }
  }
  
  const ConstantArrayType *Array 
    = getContext().getAsConstantArrayType(E->getType());
  if (Array) {
    QualType BaseElementTy = getContext().getBaseElementType(Array);
    const llvm::Type *BasePtr = ConvertType(BaseElementTy);
    BasePtr = llvm::PointerType::getUnqual(BasePtr);
    llvm::Value *BaseAddrPtr =
      Builder.CreateBitCast(Dest, BasePtr);
    
    EmitCXXAggrConstructorCall(CD, Array, BaseAddrPtr, 
                               E->arg_begin(), E->arg_end());
  }
  else {
    CXXCtorType Type = 
      (E->getConstructionKind() == CXXConstructExpr::CK_Complete) 
      ? Ctor_Complete : Ctor_Base;
    bool ForVirtualBase = 
      E->getConstructionKind() == CXXConstructExpr::CK_VirtualBase;
    
    // Call the constructor.
    EmitCXXConstructorCall(CD, Type, ForVirtualBase, Dest,
                           E->arg_begin(), E->arg_end());
  }
}

static CharUnits CalculateCookiePadding(ASTContext &Ctx, QualType ElementType) {
  ElementType = Ctx.getBaseElementType(ElementType);
  const RecordType *RT = ElementType->getAs<RecordType>();
  if (!RT)
    return CharUnits::Zero();
  
  const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(RT->getDecl());
  if (!RD)
    return CharUnits::Zero();
  
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
      return CharUnits::Zero();
    
    // The usual deallocation function doesn't take a size_t argument, so we
    // don't need a cookie.
    if (UsualDeallocationFunction->getNumParams() == 1)
      return CharUnits::Zero();
        
    assert(UsualDeallocationFunction->getNumParams() == 2 && 
           "Unexpected deallocation function type!");
  }  
  
  // Padding is the maximum of sizeof(size_t) and alignof(ElementType)
  return std::max(Ctx.getTypeSizeInChars(Ctx.getSizeType()),
                  Ctx.getTypeAlignInChars(ElementType));
}

/// Check whether the given operator new[] is the global placement
/// operator new[].
static bool IsPlacementOperatorNewArray(ASTContext &Ctx,
                                        const FunctionDecl *Fn) {
  // Must be in global scope.  Note that allocation functions can't be
  // declared in namespaces.
  if (!Fn->getDeclContext()->getLookupContext()->isFileContext())
    return false;

  // Signature must be void *operator new[](size_t, void*).
  // The size_t is common to all operator new[]s.
  if (Fn->getNumParams() != 2)
    return false;

  CanQualType ParamType = Ctx.getCanonicalType(Fn->getParamDecl(1)->getType());
  return (ParamType == Ctx.VoidPtrTy);
}

static CharUnits CalculateCookiePadding(ASTContext &Ctx, const CXXNewExpr *E) {
  if (!E->isArray())
    return CharUnits::Zero();

  // No cookie is required if the new operator being used is 
  // ::operator new[](size_t, void*).
  const FunctionDecl *OperatorNew = E->getOperatorNew();
  if (IsPlacementOperatorNewArray(Ctx, OperatorNew))
    return CharUnits::Zero();

  return CalculateCookiePadding(Ctx, E->getAllocatedType());
}

static llvm::Value *EmitCXXNewAllocSize(ASTContext &Context,
                                        CodeGenFunction &CGF,
                                        const CXXNewExpr *E,
                                        llvm::Value *&NumElements,
                                        llvm::Value *&SizeWithoutCookie) {
  QualType ElemType = E->getAllocatedType();
  
  if (!E->isArray()) {
    CharUnits TypeSize = CGF.getContext().getTypeSizeInChars(ElemType);
    const llvm::Type *SizeTy = CGF.ConvertType(CGF.getContext().getSizeType());
    SizeWithoutCookie = llvm::ConstantInt::get(SizeTy, TypeSize.getQuantity());
    return SizeWithoutCookie;
  }

  // Emit the array size expression.
  // We multiply the size of all dimensions for NumElements.
  // e.g for 'int[2][3]', ElemType is 'int' and NumElements is 6.
  NumElements = CGF.EmitScalarExpr(E->getArraySize());
  while (const ConstantArrayType *CAT
             = CGF.getContext().getAsConstantArrayType(ElemType)) {
    ElemType = CAT->getElementType();
    llvm::Value *ArraySize
        = llvm::ConstantInt::get(CGF.CGM.getLLVMContext(), CAT->getSize());
    NumElements = CGF.Builder.CreateMul(NumElements, ArraySize);
  }

  CharUnits TypeSize = CGF.getContext().getTypeSizeInChars(ElemType);
  const llvm::Type *SizeTy = CGF.ConvertType(CGF.getContext().getSizeType());
  llvm::Value *Size = llvm::ConstantInt::get(SizeTy, TypeSize.getQuantity());
  
  // If someone is doing 'new int[42]' there is no need to do a dynamic check.
  // Don't bloat the -O0 code.
  if (llvm::ConstantInt *NumElementsC =
        dyn_cast<llvm::ConstantInt>(NumElements)) {
    // Determine if there is an overflow here by doing an extended multiply.
    llvm::APInt NEC = NumElementsC->getValue();
    NEC.zext(NEC.getBitWidth()*2);
    
    llvm::APInt SC = cast<llvm::ConstantInt>(Size)->getValue();
    SC.zext(SC.getBitWidth()*2);
    SC *= NEC;
    
    if (SC.countLeadingZeros() >= NumElementsC->getValue().getBitWidth()) {
      SC.trunc(NumElementsC->getValue().getBitWidth());
      Size = llvm::ConstantInt::get(Size->getContext(), SC);
    } else {
      // On overflow, produce a -1 so operator new throws.
      Size = llvm::Constant::getAllOnesValue(Size->getType());
    }
    
  } else {
    // Multiply with the type size.  This multiply can overflow, e.g. in:
    //   new double[n]
    // where n is 2^30 on a 32-bit machine or 2^62 on a 64-bit machine.  Because
    // of this, we need to detect the overflow and ensure that an exception is
    // called by forcing the size to -1 on overflow.
    llvm::Value *UMulF =
      CGF.CGM.getIntrinsic(llvm::Intrinsic::umul_with_overflow, &SizeTy, 1);
    llvm::Value *MulRes = CGF.Builder.CreateCall2(UMulF, NumElements, Size);
    // Branch on the overflow bit to the overflow block, which is lazily
    // created.
    llvm::Value *DidOverflow = CGF.Builder.CreateExtractValue(MulRes, 1);
    // Get the normal result of the multiplication.
    llvm::Value *V = CGF.Builder.CreateExtractValue(MulRes, 0);
    
    llvm::BasicBlock *NormalBB = CGF.createBasicBlock("no_overflow");
    llvm::BasicBlock *OverflowBB = CGF.createBasicBlock("overflow");
    
    CGF.Builder.CreateCondBr(DidOverflow, OverflowBB, NormalBB);

    llvm::BasicBlock *PrevBB = CGF.Builder.GetInsertBlock();
    
    // We just need the overflow block to build a PHI node.
    CGF.EmitBlock(OverflowBB);
    CGF.EmitBlock(NormalBB);
    
    llvm::PHINode *PN = CGF.Builder.CreatePHI(V->getType());
    
    PN->addIncoming(V, PrevBB);
    PN->addIncoming(llvm::Constant::getAllOnesValue(V->getType()), OverflowBB);
    Size = PN;
  }
  SizeWithoutCookie = Size;
  
  // Add the cookie padding if necessary.
  CharUnits CookiePadding = CalculateCookiePadding(CGF.getContext(), E);
  if (!CookiePadding.isZero())
    Size = CGF.Builder.CreateAdd(Size, 
        llvm::ConstantInt::get(SizeTy, CookiePadding.getQuantity()));
  
  return Size;
}

static void StoreAnyExprIntoOneUnit(CodeGenFunction &CGF, const CXXNewExpr *E,
                                    llvm::Value *NewPtr) {
  
  assert(E->getNumConstructorArgs() == 1 &&
         "Can only have one argument to initializer of POD type.");
  
  const Expr *Init = E->getConstructorArg(0);
  QualType AllocType = E->getAllocatedType();

  unsigned Alignment =
    CGF.getContext().getTypeAlignInChars(AllocType).getQuantity();
  if (!CGF.hasAggregateLLVMType(AllocType)) 
    CGF.EmitStoreOfScalar(CGF.EmitScalarExpr(Init), NewPtr,
                          AllocType.isVolatileQualified(), Alignment,
                          AllocType);
  else if (AllocType->isAnyComplexType())
    CGF.EmitComplexExprIntoAddr(Init, NewPtr, 
                                AllocType.isVolatileQualified());
  else
    CGF.EmitAggExpr(Init, NewPtr, AllocType.isVolatileQualified());
}

void
CodeGenFunction::EmitNewArrayInitializer(const CXXNewExpr *E, 
                                         llvm::Value *NewPtr,
                                         llvm::Value *NumElements) {
  // We have a POD type.
  if (E->getNumConstructorArgs() == 0)
    return;
  
  const llvm::Type *SizeTy = ConvertType(getContext().getSizeType());
  
  // Create a temporary for the loop index and initialize it with 0.
  llvm::Value *IndexPtr = CreateTempAlloca(SizeTy, "loop.index");
  llvm::Value *Zero = llvm::Constant::getNullValue(SizeTy);
  Builder.CreateStore(Zero, IndexPtr);
  
  // Start the loop with a block that tests the condition.
  llvm::BasicBlock *CondBlock = createBasicBlock("for.cond");
  llvm::BasicBlock *AfterFor = createBasicBlock("for.end");
  
  EmitBlock(CondBlock);
  
  llvm::BasicBlock *ForBody = createBasicBlock("for.body");
  
  // Generate: if (loop-index < number-of-elements fall to the loop body,
  // otherwise, go to the block after the for-loop.
  llvm::Value *Counter = Builder.CreateLoad(IndexPtr);
  llvm::Value *IsLess = Builder.CreateICmpULT(Counter, NumElements, "isless");
  // If the condition is true, execute the body.
  Builder.CreateCondBr(IsLess, ForBody, AfterFor);
  
  EmitBlock(ForBody);
  
  llvm::BasicBlock *ContinueBlock = createBasicBlock("for.inc");
  // Inside the loop body, emit the constructor call on the array element.
  Counter = Builder.CreateLoad(IndexPtr);
  llvm::Value *Address = Builder.CreateInBoundsGEP(NewPtr, Counter, 
                                                   "arrayidx");
  StoreAnyExprIntoOneUnit(*this, E, Address);
  
  EmitBlock(ContinueBlock);
  
  // Emit the increment of the loop counter.
  llvm::Value *NextVal = llvm::ConstantInt::get(SizeTy, 1);
  Counter = Builder.CreateLoad(IndexPtr);
  NextVal = Builder.CreateAdd(Counter, NextVal, "inc");
  Builder.CreateStore(NextVal, IndexPtr);
  
  // Finally, branch back up to the condition for the next iteration.
  EmitBranch(CondBlock);
  
  // Emit the fall-through block.
  EmitBlock(AfterFor, true);
}

static void EmitZeroMemSet(CodeGenFunction &CGF, QualType T,
                           llvm::Value *NewPtr, llvm::Value *Size) {
  llvm::LLVMContext &VMContext = CGF.CGM.getLLVMContext();
  const llvm::Type *BP = llvm::Type::getInt8PtrTy(VMContext);
  if (NewPtr->getType() != BP)
    NewPtr = CGF.Builder.CreateBitCast(NewPtr, BP, "tmp");
  
  CGF.Builder.CreateCall5(CGF.CGM.getMemSetFn(BP, CGF.IntPtrTy), NewPtr,
                llvm::Constant::getNullValue(llvm::Type::getInt8Ty(VMContext)),
                          Size,
                    llvm::ConstantInt::get(CGF.Int32Ty, 
                                           CGF.getContext().getTypeAlign(T)/8),
                          llvm::ConstantInt::get(llvm::Type::getInt1Ty(VMContext),
                                                 0));
}
                       
static void EmitNewInitializer(CodeGenFunction &CGF, const CXXNewExpr *E,
                               llvm::Value *NewPtr,
                               llvm::Value *NumElements,
                               llvm::Value *AllocSizeWithoutCookie) {
  if (E->isArray()) {
    if (CXXConstructorDecl *Ctor = E->getConstructor()) {
      bool RequiresZeroInitialization = false;
      if (Ctor->getParent()->hasTrivialConstructor()) {
        // If new expression did not specify value-initialization, then there
        // is no initialization.
        if (!E->hasInitializer() || Ctor->getParent()->isEmpty())
          return;
      
        if (CGF.CGM.getTypes().isZeroInitializable(E->getAllocatedType())) {
          // Optimization: since zero initialization will just set the memory
          // to all zeroes, generate a single memset to do it in one shot.
          EmitZeroMemSet(CGF, E->getAllocatedType(), NewPtr, 
                         AllocSizeWithoutCookie);
          return;
        }

        RequiresZeroInitialization = true;
      }
      
      CGF.EmitCXXAggrConstructorCall(Ctor, NumElements, NewPtr, 
                                     E->constructor_arg_begin(), 
                                     E->constructor_arg_end(),
                                     RequiresZeroInitialization);
      return;
    } else if (E->getNumConstructorArgs() == 1 &&
               isa<ImplicitValueInitExpr>(E->getConstructorArg(0))) {
      // Optimization: since zero initialization will just set the memory
      // to all zeroes, generate a single memset to do it in one shot.
      EmitZeroMemSet(CGF, E->getAllocatedType(), NewPtr, 
                     AllocSizeWithoutCookie);
      return;      
    } else {
      CGF.EmitNewArrayInitializer(E, NewPtr, NumElements);
      return;
    }
  }

  if (CXXConstructorDecl *Ctor = E->getConstructor()) {
    // Per C++ [expr.new]p15, if we have an initializer, then we're performing
    // direct initialization. C++ [dcl.init]p5 requires that we 
    // zero-initialize storage if there are no user-declared constructors.
    if (E->hasInitializer() && 
        !Ctor->getParent()->hasUserDeclaredConstructor() &&
        !Ctor->getParent()->isEmpty())
      CGF.EmitNullInitialization(NewPtr, E->getAllocatedType());
      
    CGF.EmitCXXConstructorCall(Ctor, Ctor_Complete, /*ForVirtualBase=*/false, 
                               NewPtr, E->constructor_arg_begin(),
                               E->constructor_arg_end());

    return;
  }
  // We have a POD type.
  if (E->getNumConstructorArgs() == 0)
    return;
  
  StoreAnyExprIntoOneUnit(CGF, E, NewPtr);
}

llvm::Value *CodeGenFunction::EmitCXXNewExpr(const CXXNewExpr *E) {
  QualType AllocType = E->getAllocatedType();
  FunctionDecl *NewFD = E->getOperatorNew();
  const FunctionProtoType *NewFTy = NewFD->getType()->getAs<FunctionProtoType>();

  CallArgList NewArgs;

  // The allocation size is the first argument.
  QualType SizeTy = getContext().getSizeType();

  llvm::Value *NumElements = 0;
  llvm::Value *AllocSizeWithoutCookie = 0;
  llvm::Value *AllocSize = EmitCXXNewAllocSize(getContext(),
                                               *this, E, NumElements,
                                               AllocSizeWithoutCookie);
  
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
    EmitCall(CGM.getTypes().getFunctionInfo(NewArgs, NewFTy),
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
  
  CharUnits CookiePadding = CalculateCookiePadding(getContext(), E);
  if (!CookiePadding.isZero()) {
    CharUnits CookieOffset = 
      CookiePadding - getContext().getTypeSizeInChars(SizeTy);
    
    llvm::Value *NumElementsPtr = 
      Builder.CreateConstInBoundsGEP1_64(NewPtr, CookieOffset.getQuantity());
    
    NumElementsPtr = Builder.CreateBitCast(NumElementsPtr, 
                                           ConvertType(SizeTy)->getPointerTo());
    Builder.CreateStore(NumElements, NumElementsPtr);

    // Now add the padding to the new ptr.
    NewPtr = Builder.CreateConstInBoundsGEP1_64(NewPtr, 
                                                CookiePadding.getQuantity());
  }
  
  if (AllocType->isArrayType()) {
    while (const ArrayType *AType = getContext().getAsArrayType(AllocType))
      AllocType = AType->getElementType();
    NewPtr = 
      Builder.CreateBitCast(NewPtr, 
                          ConvertType(getContext().getPointerType(AllocType)));
    EmitNewInitializer(*this, E, NewPtr, NumElements, AllocSizeWithoutCookie);
    NewPtr = Builder.CreateBitCast(NewPtr, ConvertType(E->getType()));
  }
  else {
    NewPtr = Builder.CreateBitCast(NewPtr, ConvertType(E->getType()));
    EmitNewInitializer(*this, E, NewPtr, NumElements, AllocSizeWithoutCookie);
  }
  
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
  
  CharUnits DeleteTypeAlign = CGF.getContext().getTypeAlignInChars(DeleteTy);
  CharUnits CookiePadding = 
    std::max(CGF.getContext().getTypeSizeInChars(SizeTy),
             DeleteTypeAlign);
  assert(!CookiePadding.isZero() && "CookiePadding should not be 0.");

  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(CGF.getLLVMContext());
  CharUnits CookieOffset = 
    CookiePadding - CGF.getContext().getTypeSizeInChars(SizeTy);

  llvm::Value *AllocatedObjectPtr = CGF.Builder.CreateBitCast(Ptr, Int8PtrTy);
  AllocatedObjectPtr = 
    CGF.Builder.CreateConstInBoundsGEP1_64(AllocatedObjectPtr,
                                           -CookiePadding.getQuantity());

  llvm::Value *NumElementsPtr =
    CGF.Builder.CreateConstInBoundsGEP1_64(AllocatedObjectPtr, 
                                           CookieOffset.getQuantity());
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
    CharUnits DeleteTypeSize = getContext().getTypeSizeInChars(DeleteTy);
    Size = llvm::ConstantInt::get(ConvertType(SizeTy), 
                                  DeleteTypeSize.getQuantity());
  }
  
  if (DeleteFD->getOverloadedOperator() == OO_Array_Delete &&
      !CalculateCookiePadding(getContext(), DeleteTy).isZero()) {
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
  EmitCall(CGM.getTypes().getFunctionInfo(DeleteArgs, DeleteFTy),
           CGM.GetAddrOfFunction(DeleteFD), ReturnValueSlot(), 
           DeleteArgs, DeleteFD);
}

void CodeGenFunction::EmitCXXDeleteExpr(const CXXDeleteExpr *E) {
  
  // Get at the argument before we performed the implicit conversion
  // to void*.
  const Expr *Arg = E->getArgument();
  while (const ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(Arg)) {
    if (ICE->getCastKind() != CK_UserDefinedConversion &&
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
        const CXXDestructorDecl *Dtor = RD->getDestructor();
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
          EmitCXXMemberCall(Dtor, Callee, ReturnValueSlot(), Ptr, /*VTT=*/0,
                            0, 0);

          // The dtor took care of deleting the object.
          ShouldCallDelete = false;
        } else 
          EmitCXXDestructorCall(Dtor, Dtor_Complete, /*ForVirtualBase=*/false,
                                Ptr);
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
        if (UO->getOpcode() == UO_Deref)
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
    
    // FIXME: What if exceptions are disabled?
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
      /// Invoke __cxa_bad_cast
      ResultType = llvm::Type::getVoidTy(VMContext);
      const llvm::FunctionType *FBadTy;
      FBadTy = llvm::FunctionType::get(ResultType, false);
      llvm::Value *F = CGM.CreateRuntimeFunction(FBadTy, "__cxa_bad_cast");
      if (llvm::BasicBlock *InvokeDest = getInvokeDest()) {
        llvm::BasicBlock *Cont = createBasicBlock("invoke.cont");
        Builder.CreateInvoke(F, Cont, InvokeDest)->setDoesNotReturn();
        EmitBlock(Cont);
      } else {
        // FIXME: Does this ever make sense?
        Builder.CreateCall(F)->setDoesNotReturn();
      }
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
