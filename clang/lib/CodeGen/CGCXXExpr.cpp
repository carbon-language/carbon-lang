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

llvm::Value *CodeGenFunction::EmitCXXNewExpr(const CXXNewExpr *E) {
  if (E->isArray()) {
    ErrorUnsupported(E, "new[] expression");
    return llvm::UndefValue::get(ConvertType(E->getType()));
  }

  QualType AllocType = E->getAllocatedType();
  FunctionDecl *NewFD = E->getOperatorNew();
  const FunctionProtoType *NewFTy = NewFD->getType()->getAs<FunctionProtoType>();

  CallArgList NewArgs;

  // The allocation size is the first argument.
  QualType SizeTy = getContext().getSizeType();
  llvm::Value *AllocSize =
    llvm::ConstantInt::get(ConvertType(SizeTy),
                           getContext().getTypeSize(AllocType) / 8);

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

  NewPtr = Builder.CreateBitCast(NewPtr, ConvertType(E->getType()));

  if (AllocType->isPODType()) {
    if (E->getNumConstructorArgs() > 0) {
      assert(E->getNumConstructorArgs() == 1 &&
             "Can only have one argument to initializer of POD type.");

      const Expr *Init = E->getConstructorArg(0);

      if (!hasAggregateLLVMType(AllocType))
        Builder.CreateStore(EmitScalarExpr(Init), NewPtr);
      else if (AllocType->isAnyComplexType())
        EmitComplexExprIntoAddr(Init, NewPtr, AllocType.isVolatileQualified());
      else
        EmitAggExpr(Init, NewPtr, AllocType.isVolatileQualified());
    }
  } else {
    // Call the constructor.
    CXXConstructorDecl *Ctor = E->getConstructor();

    EmitCXXConstructorCall(Ctor, Ctor_Complete, NewPtr,
                           E->constructor_arg_begin(),
                           E->constructor_arg_end());
  }

  if (NullCheckResult) {
    Builder.CreateBr(NewEnd);
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
  if (E->isArrayForm()) {
    ErrorUnsupported(E, "delete[] expression");
    return;
  };

  QualType DeleteTy =
    E->getArgument()->getType()->getAs<PointerType>()->getPointeeType();

  llvm::Value *Ptr = EmitScalarExpr(E->getArgument());

  // Null check the pointer.
  llvm::BasicBlock *DeleteNotNull = createBasicBlock("delete.notnull");
  llvm::BasicBlock *DeleteEnd = createBasicBlock("delete.end");

  llvm::Value *IsNull =
    Builder.CreateICmpEQ(Ptr, llvm::Constant::getNullValue(Ptr->getType()),
                         "isnull");

  Builder.CreateCondBr(IsNull, DeleteEnd, DeleteNotNull);
  EmitBlock(DeleteNotNull);

  // Call the destructor if necessary.
  if (const RecordType *RT = DeleteTy->getAs<RecordType>()) {
    if (CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(RT->getDecl())) {
      if (!RD->hasTrivialDestructor()) {
        const CXXDestructorDecl *Dtor = RD->getDestructor(getContext());
        if (Dtor->isVirtual()) {
          const llvm::Type *Ty =
            CGM.getTypes().GetFunctionType(CGM.getTypes().getFunctionInfo(Dtor),
                                           /*isVariadic=*/false);
          
          llvm::Value *Callee = BuildVirtualCall(Dtor, Ptr, Ty);
          EmitCXXMemberCall(Dtor, Callee, Ptr, 0, 0);
        } else 
          EmitCXXDestructorCall(Dtor, Dtor_Complete, Ptr);
      }
    }
  }

  // Call delete.
  FunctionDecl *DeleteFD = E->getOperatorDelete();
  const FunctionProtoType *DeleteFTy =
    DeleteFD->getType()->getAs<FunctionProtoType>();

  CallArgList DeleteArgs;

  QualType ArgTy = DeleteFTy->getArgType(0);
  llvm::Value *DeletePtr = Builder.CreateBitCast(Ptr, ConvertType(ArgTy));
  DeleteArgs.push_back(std::make_pair(RValue::get(DeletePtr), ArgTy));

  // Emit the call to delete.
  EmitCall(CGM.getTypes().getFunctionInfo(DeleteFTy->getResultType(),
                                          DeleteArgs),
           CGM.GetAddrOfFunction(DeleteFD),
           DeleteArgs, DeleteFD);

  EmitBlock(DeleteEnd);
}
