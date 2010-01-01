//===--- CGClass.cpp - Emit LLVM Code for C++ classes ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation of classes
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/RecordLayout.h"

using namespace clang;
using namespace CodeGen;

static uint64_t 
ComputeNonVirtualBaseClassOffset(ASTContext &Context, CXXBasePaths &Paths,
                                 unsigned Start) {
  uint64_t Offset = 0;

  const CXXBasePath &Path = Paths.front();
  for (unsigned i = Start, e = Path.size(); i != e; ++i) {
    const CXXBasePathElement& Element = Path[i];

    // Get the layout.
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(Element.Class);
    
    const CXXBaseSpecifier *BS = Element.Base;
    assert(!BS->isVirtual() && "Should not see virtual bases here!");
    
    const CXXRecordDecl *Base = 
      cast<CXXRecordDecl>(BS->getType()->getAs<RecordType>()->getDecl());
    
    // Add the offset.
    Offset += Layout.getBaseClassOffset(Base) / 8;
  }

  return Offset;
}

llvm::Constant *
CodeGenModule::GetCXXBaseClassOffset(const CXXRecordDecl *ClassDecl,
                                     const CXXRecordDecl *BaseClassDecl) {
  if (ClassDecl == BaseClassDecl)
    return 0;

  CXXBasePaths Paths(/*FindAmbiguities=*/false,
                     /*RecordPaths=*/true, /*DetectVirtual=*/false);
  if (!const_cast<CXXRecordDecl *>(ClassDecl)->
        isDerivedFrom(const_cast<CXXRecordDecl *>(BaseClassDecl), Paths)) {
    assert(false && "Class must be derived from the passed in base class!");
    return 0;
  }

  uint64_t Offset = ComputeNonVirtualBaseClassOffset(getContext(), Paths, 0);
  if (!Offset)
    return 0;

  const llvm::Type *PtrDiffTy = 
    Types.ConvertType(getContext().getPointerDiffType());

  return llvm::ConstantInt::get(PtrDiffTy, Offset);
}

static llvm::Value *GetCXXBaseClassOffset(CodeGenFunction &CGF,
                                          llvm::Value *BaseValue,
                                          const CXXRecordDecl *ClassDecl,
                                          const CXXRecordDecl *BaseClassDecl) {
  CXXBasePaths Paths(/*FindAmbiguities=*/false,
                     /*RecordPaths=*/true, /*DetectVirtual=*/false);
  if (!const_cast<CXXRecordDecl *>(ClassDecl)->
        isDerivedFrom(const_cast<CXXRecordDecl *>(BaseClassDecl), Paths)) {
    assert(false && "Class must be derived from the passed in base class!");
    return 0;
  }

  unsigned Start = 0;
  llvm::Value *VirtualOffset = 0;

  const CXXBasePath &Path = Paths.front();
  const CXXRecordDecl *VBase = 0;
  for (unsigned i = 0, e = Path.size(); i != e; ++i) {
    const CXXBasePathElement& Element = Path[i];
    if (Element.Base->isVirtual()) {
      Start = i+1;
      QualType VBaseType = Element.Base->getType();
      VBase = cast<CXXRecordDecl>(VBaseType->getAs<RecordType>()->getDecl());
    }
  }
  if (VBase)
    VirtualOffset = 
      CGF.GetVirtualCXXBaseClassOffset(BaseValue, ClassDecl, VBase);
  
  uint64_t Offset = 
    ComputeNonVirtualBaseClassOffset(CGF.getContext(), Paths, Start);
  
  if (!Offset)
    return VirtualOffset;
  
  const llvm::Type *PtrDiffTy = 
    CGF.ConvertType(CGF.getContext().getPointerDiffType());
  llvm::Value *NonVirtualOffset = llvm::ConstantInt::get(PtrDiffTy, Offset);
  
  if (VirtualOffset)
    return CGF.Builder.CreateAdd(VirtualOffset, NonVirtualOffset);
                    
  return NonVirtualOffset;
}

// FIXME: This probably belongs in CGVtable, but it relies on 
// the static function ComputeNonVirtualBaseClassOffset, so we should make that
// a CodeGenModule member function as well.
ThunkAdjustment
CodeGenModule::ComputeThunkAdjustment(const CXXRecordDecl *ClassDecl,
                                      const CXXRecordDecl *BaseClassDecl) {
  CXXBasePaths Paths(/*FindAmbiguities=*/false,
                     /*RecordPaths=*/true, /*DetectVirtual=*/false);
  if (!const_cast<CXXRecordDecl *>(ClassDecl)->
        isDerivedFrom(const_cast<CXXRecordDecl *>(BaseClassDecl), Paths)) {
    assert(false && "Class must be derived from the passed in base class!");
    return ThunkAdjustment();
  }

  unsigned Start = 0;
  uint64_t VirtualOffset = 0;

  const CXXBasePath &Path = Paths.front();
  const CXXRecordDecl *VBase = 0;
  for (unsigned i = 0, e = Path.size(); i != e; ++i) {
    const CXXBasePathElement& Element = Path[i];
    if (Element.Base->isVirtual()) {
      Start = i+1;
      QualType VBaseType = Element.Base->getType();
      VBase = cast<CXXRecordDecl>(VBaseType->getAs<RecordType>()->getDecl());
    }
  }
  if (VBase)
    VirtualOffset = 
      getVtableInfo().getVirtualBaseOffsetIndex(ClassDecl, BaseClassDecl);
  
  uint64_t Offset = 
    ComputeNonVirtualBaseClassOffset(getContext(), Paths, Start);
  return ThunkAdjustment(Offset, VirtualOffset);
}

llvm::Value *
CodeGenFunction::GetAddressOfBaseClass(llvm::Value *Value,
                                       const CXXRecordDecl *ClassDecl,
                                       const CXXRecordDecl *BaseClassDecl,
                                       bool NullCheckValue) {
  QualType BTy =
    getContext().getCanonicalType(
      getContext().getTypeDeclType(const_cast<CXXRecordDecl*>(BaseClassDecl)));
  const llvm::Type *BasePtrTy = llvm::PointerType::getUnqual(ConvertType(BTy));

  if (ClassDecl == BaseClassDecl) {
    // Just cast back.
    return Builder.CreateBitCast(Value, BasePtrTy);
  }
  
  llvm::BasicBlock *CastNull = 0;
  llvm::BasicBlock *CastNotNull = 0;
  llvm::BasicBlock *CastEnd = 0;
  
  if (NullCheckValue) {
    CastNull = createBasicBlock("cast.null");
    CastNotNull = createBasicBlock("cast.notnull");
    CastEnd = createBasicBlock("cast.end");
    
    llvm::Value *IsNull = 
      Builder.CreateICmpEQ(Value,
                           llvm::Constant::getNullValue(Value->getType()));
    Builder.CreateCondBr(IsNull, CastNull, CastNotNull);
    EmitBlock(CastNotNull);
  }
  
  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(VMContext);

  llvm::Value *Offset = 
    GetCXXBaseClassOffset(*this, Value, ClassDecl, BaseClassDecl);
  
  if (Offset) {
    // Apply the offset.
    Value = Builder.CreateBitCast(Value, Int8PtrTy);
    Value = Builder.CreateGEP(Value, Offset, "add.ptr");
  }
  
  // Cast back.
  Value = Builder.CreateBitCast(Value, BasePtrTy);
 
  if (NullCheckValue) {
    Builder.CreateBr(CastEnd);
    EmitBlock(CastNull);
    Builder.CreateBr(CastEnd);
    EmitBlock(CastEnd);
    
    llvm::PHINode *PHI = Builder.CreatePHI(Value->getType());
    PHI->reserveOperandSpace(2);
    PHI->addIncoming(Value, CastNotNull);
    PHI->addIncoming(llvm::Constant::getNullValue(Value->getType()), 
                     CastNull);
    Value = PHI;
  }
  
  return Value;
}

llvm::Value *
CodeGenFunction::GetAddressOfDerivedClass(llvm::Value *Value,
                                          const CXXRecordDecl *ClassDecl,
                                          const CXXRecordDecl *DerivedClassDecl,
                                          bool NullCheckValue) {
  QualType DerivedTy =
    getContext().getCanonicalType(
    getContext().getTypeDeclType(const_cast<CXXRecordDecl*>(DerivedClassDecl)));
  const llvm::Type *DerivedPtrTy = ConvertType(DerivedTy)->getPointerTo();
  
  if (ClassDecl == DerivedClassDecl) {
    // Just cast back.
    return Builder.CreateBitCast(Value, DerivedPtrTy);
  }

  llvm::BasicBlock *CastNull = 0;
  llvm::BasicBlock *CastNotNull = 0;
  llvm::BasicBlock *CastEnd = 0;
  
  if (NullCheckValue) {
    CastNull = createBasicBlock("cast.null");
    CastNotNull = createBasicBlock("cast.notnull");
    CastEnd = createBasicBlock("cast.end");
    
    llvm::Value *IsNull = 
    Builder.CreateICmpEQ(Value,
                         llvm::Constant::getNullValue(Value->getType()));
    Builder.CreateCondBr(IsNull, CastNull, CastNotNull);
    EmitBlock(CastNotNull);
  }
  
  llvm::Value *Offset = GetCXXBaseClassOffset(*this, Value, DerivedClassDecl,
                                              ClassDecl);
  if (Offset) {
    // Apply the offset.
    Value = Builder.CreatePtrToInt(Value, Offset->getType());
    Value = Builder.CreateSub(Value, Offset);
    Value = Builder.CreateIntToPtr(Value, DerivedPtrTy);
  } else {
    // Just cast.
    Value = Builder.CreateBitCast(Value, DerivedPtrTy);
  }

  if (NullCheckValue) {
    Builder.CreateBr(CastEnd);
    EmitBlock(CastNull);
    Builder.CreateBr(CastEnd);
    EmitBlock(CastEnd);
    
    llvm::PHINode *PHI = Builder.CreatePHI(Value->getType());
    PHI->reserveOperandSpace(2);
    PHI->addIncoming(Value, CastNotNull);
    PHI->addIncoming(llvm::Constant::getNullValue(Value->getType()), 
                     CastNull);
    Value = PHI;
  }
  
  return Value;
}

/// EmitClassAggrMemberwiseCopy - This routine generates code to copy a class
/// array of objects from SrcValue to DestValue. Copying can be either a bitwise
/// copy or via a copy constructor call.
//  FIXME. Consolidate this with EmitCXXAggrConstructorCall.
void CodeGenFunction::EmitClassAggrMemberwiseCopy(llvm::Value *Dest,
                                            llvm::Value *Src,
                                            const ArrayType *Array,
                                            const CXXRecordDecl *BaseClassDecl,
                                            QualType Ty) {
  const ConstantArrayType *CA = dyn_cast<ConstantArrayType>(Array);
  assert(CA && "VLA cannot be copied over");
  bool BitwiseCopy = BaseClassDecl->hasTrivialCopyConstructor();

  // Create a temporary for the loop index and initialize it with 0.
  llvm::Value *IndexPtr = CreateTempAlloca(llvm::Type::getInt64Ty(VMContext),
                                           "loop.index");
  llvm::Value* zeroConstant =
    llvm::Constant::getNullValue(llvm::Type::getInt64Ty(VMContext));
  Builder.CreateStore(zeroConstant, IndexPtr);
  // Start the loop with a block that tests the condition.
  llvm::BasicBlock *CondBlock = createBasicBlock("for.cond");
  llvm::BasicBlock *AfterFor = createBasicBlock("for.end");

  EmitBlock(CondBlock);

  llvm::BasicBlock *ForBody = createBasicBlock("for.body");
  // Generate: if (loop-index < number-of-elements fall to the loop body,
  // otherwise, go to the block after the for-loop.
  uint64_t NumElements = getContext().getConstantArrayElementCount(CA);
  llvm::Value * NumElementsPtr =
    llvm::ConstantInt::get(llvm::Type::getInt64Ty(VMContext), NumElements);
  llvm::Value *Counter = Builder.CreateLoad(IndexPtr);
  llvm::Value *IsLess = Builder.CreateICmpULT(Counter, NumElementsPtr,
                                              "isless");
  // If the condition is true, execute the body.
  Builder.CreateCondBr(IsLess, ForBody, AfterFor);

  EmitBlock(ForBody);
  llvm::BasicBlock *ContinueBlock = createBasicBlock("for.inc");
  // Inside the loop body, emit the constructor call on the array element.
  Counter = Builder.CreateLoad(IndexPtr);
  Src = Builder.CreateInBoundsGEP(Src, Counter, "srcaddress");
  Dest = Builder.CreateInBoundsGEP(Dest, Counter, "destaddress");
  if (BitwiseCopy)
    EmitAggregateCopy(Dest, Src, Ty);
  else if (CXXConstructorDecl *BaseCopyCtor =
           BaseClassDecl->getCopyConstructor(getContext(), 0)) {
    llvm::Value *Callee = CGM.GetAddrOfCXXConstructor(BaseCopyCtor,
                                                      Ctor_Complete);
    CallArgList CallArgs;
    // Push the this (Dest) ptr.
    CallArgs.push_back(std::make_pair(RValue::get(Dest),
                                      BaseCopyCtor->getThisType(getContext())));

    // Push the Src ptr.
    CallArgs.push_back(std::make_pair(RValue::get(Src),
                                     BaseCopyCtor->getParamDecl(0)->getType()));
    QualType ResultType =
      BaseCopyCtor->getType()->getAs<FunctionType>()->getResultType();
    EmitCall(CGM.getTypes().getFunctionInfo(ResultType, CallArgs),
             Callee, ReturnValueSlot(), CallArgs, BaseCopyCtor);
  }
  EmitBlock(ContinueBlock);

  // Emit the increment of the loop counter.
  llvm::Value *NextVal = llvm::ConstantInt::get(Counter->getType(), 1);
  Counter = Builder.CreateLoad(IndexPtr);
  NextVal = Builder.CreateAdd(Counter, NextVal, "inc");
  Builder.CreateStore(NextVal, IndexPtr);

  // Finally, branch back up to the condition for the next iteration.
  EmitBranch(CondBlock);

  // Emit the fall-through block.
  EmitBlock(AfterFor, true);
}

/// EmitClassAggrCopyAssignment - This routine generates code to assign a class
/// array of objects from SrcValue to DestValue. Assignment can be either a
/// bitwise assignment or via a copy assignment operator function call.
/// FIXME. This can be consolidated with EmitClassAggrMemberwiseCopy
void CodeGenFunction::EmitClassAggrCopyAssignment(llvm::Value *Dest,
                                            llvm::Value *Src,
                                            const ArrayType *Array,
                                            const CXXRecordDecl *BaseClassDecl,
                                            QualType Ty) {
  const ConstantArrayType *CA = dyn_cast<ConstantArrayType>(Array);
  assert(CA && "VLA cannot be asssigned");
  bool BitwiseAssign = BaseClassDecl->hasTrivialCopyAssignment();

  // Create a temporary for the loop index and initialize it with 0.
  llvm::Value *IndexPtr = CreateTempAlloca(llvm::Type::getInt64Ty(VMContext),
                                           "loop.index");
  llvm::Value* zeroConstant =
  llvm::Constant::getNullValue(llvm::Type::getInt64Ty(VMContext));
  Builder.CreateStore(zeroConstant, IndexPtr);
  // Start the loop with a block that tests the condition.
  llvm::BasicBlock *CondBlock = createBasicBlock("for.cond");
  llvm::BasicBlock *AfterFor = createBasicBlock("for.end");

  EmitBlock(CondBlock);

  llvm::BasicBlock *ForBody = createBasicBlock("for.body");
  // Generate: if (loop-index < number-of-elements fall to the loop body,
  // otherwise, go to the block after the for-loop.
  uint64_t NumElements = getContext().getConstantArrayElementCount(CA);
  llvm::Value * NumElementsPtr =
  llvm::ConstantInt::get(llvm::Type::getInt64Ty(VMContext), NumElements);
  llvm::Value *Counter = Builder.CreateLoad(IndexPtr);
  llvm::Value *IsLess = Builder.CreateICmpULT(Counter, NumElementsPtr,
                                              "isless");
  // If the condition is true, execute the body.
  Builder.CreateCondBr(IsLess, ForBody, AfterFor);

  EmitBlock(ForBody);
  llvm::BasicBlock *ContinueBlock = createBasicBlock("for.inc");
  // Inside the loop body, emit the assignment operator call on array element.
  Counter = Builder.CreateLoad(IndexPtr);
  Src = Builder.CreateInBoundsGEP(Src, Counter, "srcaddress");
  Dest = Builder.CreateInBoundsGEP(Dest, Counter, "destaddress");
  const CXXMethodDecl *MD = 0;
  if (BitwiseAssign)
    EmitAggregateCopy(Dest, Src, Ty);
  else {
    bool hasCopyAssign = BaseClassDecl->hasConstCopyAssignment(getContext(),
                                                               MD);
    assert(hasCopyAssign && "EmitClassAggrCopyAssignment - No user assign");
    (void)hasCopyAssign;
    const FunctionProtoType *FPT = MD->getType()->getAs<FunctionProtoType>();
    const llvm::Type *LTy =
    CGM.getTypes().GetFunctionType(CGM.getTypes().getFunctionInfo(MD),
                                   FPT->isVariadic());
    llvm::Constant *Callee = CGM.GetAddrOfFunction(MD, LTy);

    CallArgList CallArgs;
    // Push the this (Dest) ptr.
    CallArgs.push_back(std::make_pair(RValue::get(Dest),
                                      MD->getThisType(getContext())));

    // Push the Src ptr.
    CallArgs.push_back(std::make_pair(RValue::get(Src),
                                      MD->getParamDecl(0)->getType()));
    QualType ResultType = MD->getType()->getAs<FunctionType>()->getResultType();
    EmitCall(CGM.getTypes().getFunctionInfo(ResultType, CallArgs),
             Callee, ReturnValueSlot(), CallArgs, MD);
  }
  EmitBlock(ContinueBlock);

  // Emit the increment of the loop counter.
  llvm::Value *NextVal = llvm::ConstantInt::get(Counter->getType(), 1);
  Counter = Builder.CreateLoad(IndexPtr);
  NextVal = Builder.CreateAdd(Counter, NextVal, "inc");
  Builder.CreateStore(NextVal, IndexPtr);

  // Finally, branch back up to the condition for the next iteration.
  EmitBranch(CondBlock);

  // Emit the fall-through block.
  EmitBlock(AfterFor, true);
}

/// EmitClassMemberwiseCopy - This routine generates code to copy a class
/// object from SrcValue to DestValue. Copying can be either a bitwise copy
/// or via a copy constructor call.
void CodeGenFunction::EmitClassMemberwiseCopy(
                        llvm::Value *Dest, llvm::Value *Src,
                        const CXXRecordDecl *ClassDecl,
                        const CXXRecordDecl *BaseClassDecl, QualType Ty) {
  if (ClassDecl) {
    Dest = GetAddressOfBaseClass(Dest, ClassDecl, BaseClassDecl,
                                 /*NullCheckValue=*/false);
    Src = GetAddressOfBaseClass(Src, ClassDecl, BaseClassDecl,
                                /*NullCheckValue=*/false);
  }
  if (BaseClassDecl->hasTrivialCopyConstructor()) {
    EmitAggregateCopy(Dest, Src, Ty);
    return;
  }

  if (CXXConstructorDecl *BaseCopyCtor =
      BaseClassDecl->getCopyConstructor(getContext(), 0)) {
    llvm::Value *Callee = CGM.GetAddrOfCXXConstructor(BaseCopyCtor,
                                                      Ctor_Complete);
    CallArgList CallArgs;
    // Push the this (Dest) ptr.
    CallArgs.push_back(std::make_pair(RValue::get(Dest),
                                      BaseCopyCtor->getThisType(getContext())));

    // Push the Src ptr.
    CallArgs.push_back(std::make_pair(RValue::get(Src),
                       BaseCopyCtor->getParamDecl(0)->getType()));
    QualType ResultType =
    BaseCopyCtor->getType()->getAs<FunctionType>()->getResultType();
    EmitCall(CGM.getTypes().getFunctionInfo(ResultType, CallArgs),
             Callee, ReturnValueSlot(), CallArgs, BaseCopyCtor);
  }
}

/// EmitClassCopyAssignment - This routine generates code to copy assign a class
/// object from SrcValue to DestValue. Assignment can be either a bitwise
/// assignment of via an assignment operator call.
// FIXME. Consolidate this with EmitClassMemberwiseCopy as they share a lot.
void CodeGenFunction::EmitClassCopyAssignment(
                                        llvm::Value *Dest, llvm::Value *Src,
                                        const CXXRecordDecl *ClassDecl,
                                        const CXXRecordDecl *BaseClassDecl,
                                        QualType Ty) {
  if (ClassDecl) {
    Dest = GetAddressOfBaseClass(Dest, ClassDecl, BaseClassDecl,
                                 /*NullCheckValue=*/false);
    Src = GetAddressOfBaseClass(Src, ClassDecl, BaseClassDecl,
                                /*NullCheckValue=*/false);
  }
  if (BaseClassDecl->hasTrivialCopyAssignment()) {
    EmitAggregateCopy(Dest, Src, Ty);
    return;
  }

  const CXXMethodDecl *MD = 0;
  bool ConstCopyAssignOp = BaseClassDecl->hasConstCopyAssignment(getContext(),
                                                                 MD);
  assert(ConstCopyAssignOp && "EmitClassCopyAssignment - missing copy assign");
  (void)ConstCopyAssignOp;

  const FunctionProtoType *FPT = MD->getType()->getAs<FunctionProtoType>();
  const llvm::Type *LTy =
    CGM.getTypes().GetFunctionType(CGM.getTypes().getFunctionInfo(MD),
                                   FPT->isVariadic());
  llvm::Constant *Callee = CGM.GetAddrOfFunction(MD, LTy);

  CallArgList CallArgs;
  // Push the this (Dest) ptr.
  CallArgs.push_back(std::make_pair(RValue::get(Dest),
                                    MD->getThisType(getContext())));

  // Push the Src ptr.
  CallArgs.push_back(std::make_pair(RValue::get(Src),
                                    MD->getParamDecl(0)->getType()));
  QualType ResultType =
    MD->getType()->getAs<FunctionType>()->getResultType();
  EmitCall(CGM.getTypes().getFunctionInfo(ResultType, CallArgs),
           Callee, ReturnValueSlot(), CallArgs, MD);
}

/// SynthesizeDefaultConstructor - synthesize a default constructor
void
CodeGenFunction::SynthesizeDefaultConstructor(const CXXConstructorDecl *Ctor,
                                              CXXCtorType Type,
                                              llvm::Function *Fn,
                                              const FunctionArgList &Args) {
  assert(!Ctor->isTrivial() && "shouldn't need to generate trivial ctor");
  StartFunction(GlobalDecl(Ctor, Type), Ctor->getResultType(), Fn, Args, 
                SourceLocation());
  EmitCtorPrologue(Ctor, Type);
  FinishFunction();
}

/// SynthesizeCXXCopyConstructor - This routine implicitly defines body of a
/// copy constructor, in accordance with section 12.8 (p7 and p8) of C++03
/// The implicitly-defined copy constructor for class X performs a memberwise
/// copy of its subobjects. The order of copying is the same as the order of
/// initialization of bases and members in a user-defined constructor
/// Each subobject is copied in the manner appropriate to its type:
///  if the subobject is of class type, the copy constructor for the class is
///  used;
///  if the subobject is an array, each element is copied, in the manner
///  appropriate to the element type;
///  if the subobject is of scalar type, the built-in assignment operator is
///  used.
/// Virtual base class subobjects shall be copied only once by the
/// implicitly-defined copy constructor

void 
CodeGenFunction::SynthesizeCXXCopyConstructor(const CXXConstructorDecl *Ctor,
                                              CXXCtorType Type,
                                              llvm::Function *Fn,
                                              const FunctionArgList &Args) {
  const CXXRecordDecl *ClassDecl = Ctor->getParent();
  assert(!ClassDecl->hasUserDeclaredCopyConstructor() &&
      "SynthesizeCXXCopyConstructor - copy constructor has definition already");
  assert(!Ctor->isTrivial() && "shouldn't need to generate trivial ctor");
  StartFunction(GlobalDecl(Ctor, Type), Ctor->getResultType(), Fn, Args, 
                SourceLocation());

  FunctionArgList::const_iterator i = Args.begin();
  const VarDecl *ThisArg = i->first;
  llvm::Value *ThisObj = GetAddrOfLocalVar(ThisArg);
  llvm::Value *LoadOfThis = Builder.CreateLoad(ThisObj, "this");
  const VarDecl *SrcArg = (i+1)->first;
  llvm::Value *SrcObj = GetAddrOfLocalVar(SrcArg);
  llvm::Value *LoadOfSrc = Builder.CreateLoad(SrcObj);

  for (CXXRecordDecl::base_class_const_iterator Base = ClassDecl->bases_begin();
       Base != ClassDecl->bases_end(); ++Base) {
    // FIXME. copy constrution of virtual base NYI
    if (Base->isVirtual())
      continue;

    CXXRecordDecl *BaseClassDecl
      = cast<CXXRecordDecl>(Base->getType()->getAs<RecordType>()->getDecl());
    EmitClassMemberwiseCopy(LoadOfThis, LoadOfSrc, ClassDecl, BaseClassDecl,
                            Base->getType());
  }

  for (CXXRecordDecl::field_iterator I = ClassDecl->field_begin(),
       E = ClassDecl->field_end(); I != E; ++I) {
    const FieldDecl *Field = *I;
    
    QualType FieldType = getContext().getCanonicalType(Field->getType());
    const ConstantArrayType *Array =
      getContext().getAsConstantArrayType(FieldType);
    if (Array)
      FieldType = getContext().getBaseElementType(FieldType);

    if (const RecordType *FieldClassType = FieldType->getAs<RecordType>()) {
      CXXRecordDecl *FieldClassDecl
        = cast<CXXRecordDecl>(FieldClassType->getDecl());
      LValue LHS = EmitLValueForField(LoadOfThis, Field, false, 0);
      LValue RHS = EmitLValueForField(LoadOfSrc, Field, false, 0);
      if (Array) {
        const llvm::Type *BasePtr = ConvertType(FieldType);
        BasePtr = llvm::PointerType::getUnqual(BasePtr);
        llvm::Value *DestBaseAddrPtr =
          Builder.CreateBitCast(LHS.getAddress(), BasePtr);
        llvm::Value *SrcBaseAddrPtr =
          Builder.CreateBitCast(RHS.getAddress(), BasePtr);
        EmitClassAggrMemberwiseCopy(DestBaseAddrPtr, SrcBaseAddrPtr, Array,
                                    FieldClassDecl, FieldType);
      }
      else
        EmitClassMemberwiseCopy(LHS.getAddress(), RHS.getAddress(),
                                0 /*ClassDecl*/, FieldClassDecl, FieldType);
      continue;
    }
    
    if (Field->getType()->isReferenceType()) {
      unsigned FieldIndex = CGM.getTypes().getLLVMFieldNo(Field);
 
      llvm::Value *LHS = Builder.CreateStructGEP(LoadOfThis, FieldIndex,
                                                 "lhs.ref");
      
      llvm::Value *RHS = Builder.CreateStructGEP(LoadOfThis, FieldIndex,
                                                 "rhs.ref");

      // Load the value in RHS.
      RHS = Builder.CreateLoad(RHS);
      
      // And store it in the LHS
      Builder.CreateStore(RHS, LHS);

      continue;
    }
    // Do a built-in assignment of scalar data members.
    LValue LHS = EmitLValueForField(LoadOfThis, Field, false, 0);
    LValue RHS = EmitLValueForField(LoadOfSrc, Field, false, 0);

    if (!hasAggregateLLVMType(Field->getType())) {
      RValue RVRHS = EmitLoadOfLValue(RHS, Field->getType());
      EmitStoreThroughLValue(RVRHS, LHS, Field->getType());
    } else if (Field->getType()->isAnyComplexType()) {
      ComplexPairTy Pair = LoadComplexFromAddr(RHS.getAddress(),
                                               RHS.isVolatileQualified());
      StoreComplexToAddr(Pair, LHS.getAddress(), LHS.isVolatileQualified());
    } else {
      EmitAggregateCopy(LHS.getAddress(), RHS.getAddress(), Field->getType());
    }
  }

  InitializeVtablePtrs(ClassDecl);
  FinishFunction();
}

/// SynthesizeCXXCopyAssignment - Implicitly define copy assignment operator.
/// Before the implicitly-declared copy assignment operator for a class is
/// implicitly defined, all implicitly- declared copy assignment operators for
/// its direct base classes and its nonstatic data members shall have been
/// implicitly defined. [12.8-p12]
/// The implicitly-defined copy assignment operator for class X performs
/// memberwise assignment of its subob- jects. The direct base classes of X are
/// assigned first, in the order of their declaration in
/// the base-specifier-list, and then the immediate nonstatic data members of X
/// are assigned, in the order in which they were declared in the class
/// definition.Each subobject is assigned in the manner appropriate to its type:
///   if the subobject is of class type, the copy assignment operator for the
///   class is used (as if by explicit qualification; that is, ignoring any
///   possible virtual overriding functions in more derived classes);
///
///   if the subobject is an array, each element is assigned, in the manner
///   appropriate to the element type;
///
///   if the subobject is of scalar type, the built-in assignment operator is
///   used.
void CodeGenFunction::SynthesizeCXXCopyAssignment(const CXXMethodDecl *CD,
                                                  llvm::Function *Fn,
                                                  const FunctionArgList &Args) {

  const CXXRecordDecl *ClassDecl = cast<CXXRecordDecl>(CD->getDeclContext());
  assert(!ClassDecl->hasUserDeclaredCopyAssignment() &&
         "SynthesizeCXXCopyAssignment - copy assignment has user declaration");
  StartFunction(CD, CD->getResultType(), Fn, Args, SourceLocation());

  FunctionArgList::const_iterator i = Args.begin();
  const VarDecl *ThisArg = i->first;
  llvm::Value *ThisObj = GetAddrOfLocalVar(ThisArg);
  llvm::Value *LoadOfThis = Builder.CreateLoad(ThisObj, "this");
  const VarDecl *SrcArg = (i+1)->first;
  llvm::Value *SrcObj = GetAddrOfLocalVar(SrcArg);
  llvm::Value *LoadOfSrc = Builder.CreateLoad(SrcObj);

  for (CXXRecordDecl::base_class_const_iterator Base = ClassDecl->bases_begin();
       Base != ClassDecl->bases_end(); ++Base) {
    // FIXME. copy assignment of virtual base NYI
    if (Base->isVirtual())
      continue;

    CXXRecordDecl *BaseClassDecl
      = cast<CXXRecordDecl>(Base->getType()->getAs<RecordType>()->getDecl());
    EmitClassCopyAssignment(LoadOfThis, LoadOfSrc, ClassDecl, BaseClassDecl,
                            Base->getType());
  }

  for (CXXRecordDecl::field_iterator Field = ClassDecl->field_begin(),
       FieldEnd = ClassDecl->field_end();
       Field != FieldEnd; ++Field) {
    QualType FieldType = getContext().getCanonicalType((*Field)->getType());
    const ConstantArrayType *Array =
      getContext().getAsConstantArrayType(FieldType);
    if (Array)
      FieldType = getContext().getBaseElementType(FieldType);

    if (const RecordType *FieldClassType = FieldType->getAs<RecordType>()) {
      CXXRecordDecl *FieldClassDecl
      = cast<CXXRecordDecl>(FieldClassType->getDecl());
      LValue LHS = EmitLValueForField(LoadOfThis, *Field, false, 0);
      LValue RHS = EmitLValueForField(LoadOfSrc, *Field, false, 0);
      if (Array) {
        const llvm::Type *BasePtr = ConvertType(FieldType);
        BasePtr = llvm::PointerType::getUnqual(BasePtr);
        llvm::Value *DestBaseAddrPtr =
          Builder.CreateBitCast(LHS.getAddress(), BasePtr);
        llvm::Value *SrcBaseAddrPtr =
          Builder.CreateBitCast(RHS.getAddress(), BasePtr);
        EmitClassAggrCopyAssignment(DestBaseAddrPtr, SrcBaseAddrPtr, Array,
                                    FieldClassDecl, FieldType);
      }
      else
        EmitClassCopyAssignment(LHS.getAddress(), RHS.getAddress(),
                               0 /*ClassDecl*/, FieldClassDecl, FieldType);
      continue;
    }
    // Do a built-in assignment of scalar data members.
    LValue LHS = EmitLValueForField(LoadOfThis, *Field, false, 0);
    LValue RHS = EmitLValueForField(LoadOfSrc, *Field, false, 0);
    if (!hasAggregateLLVMType(Field->getType())) {
      RValue RVRHS = EmitLoadOfLValue(RHS, Field->getType());
      EmitStoreThroughLValue(RVRHS, LHS, Field->getType());
    } else if (Field->getType()->isAnyComplexType()) {
      ComplexPairTy Pair = LoadComplexFromAddr(RHS.getAddress(),
                                               RHS.isVolatileQualified());
      StoreComplexToAddr(Pair, LHS.getAddress(), LHS.isVolatileQualified());
    } else {
      EmitAggregateCopy(LHS.getAddress(), RHS.getAddress(), Field->getType());
    }
  }

  // return *this;
  Builder.CreateStore(LoadOfThis, ReturnValue);

  FinishFunction();
}

static void EmitBaseInitializer(CodeGenFunction &CGF, 
                                const CXXRecordDecl *ClassDecl,
                                CXXBaseOrMemberInitializer *BaseInit,
                                CXXCtorType CtorType) {
  assert(BaseInit->isBaseInitializer() &&
         "Must have base initializer!");

  llvm::Value *ThisPtr = CGF.LoadCXXThis();
  
  const Type *BaseType = BaseInit->getBaseClass();
  CXXRecordDecl *BaseClassDecl =
    cast<CXXRecordDecl>(BaseType->getAs<RecordType>()->getDecl());

  // FIXME: This method of determining whether a base is virtual is ridiculous;
  // it should be part of BaseInit.
  bool isBaseVirtual = false;
  for (CXXRecordDecl::base_class_const_iterator I = ClassDecl->vbases_begin(),
       E = ClassDecl->vbases_end(); I != E; ++I)
    if (I->getType()->getAs<RecordType>()->getDecl() == BaseClassDecl) {
      isBaseVirtual = true;
      break;
    }

  // The base constructor doesn't construct virtual bases.
  if (CtorType == Ctor_Base && isBaseVirtual)
    return;

  // Compute the offset to the base; we do this directly instead of using
  // GetAddressOfBaseClass because the class doesn't have a vtable pointer
  // at this point.
  // FIXME: This could be refactored back into GetAddressOfBaseClass if it took
  // an extra parameter for whether the derived class is the complete object
  // class.
  const ASTRecordLayout &Layout =
      CGF.getContext().getASTRecordLayout(ClassDecl);
  uint64_t Offset;
  if (isBaseVirtual)
    Offset = Layout.getVBaseClassOffset(BaseClassDecl);
  else
    Offset = Layout.getBaseClassOffset(BaseClassDecl);
  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(CGF.getLLVMContext());
  const llvm::Type *BaseClassType = CGF.ConvertType(QualType(BaseType, 0));
  llvm::Value *V = CGF.Builder.CreateBitCast(ThisPtr, Int8PtrTy);
  V = CGF.Builder.CreateConstInBoundsGEP1_64(V, Offset/8);
  V = CGF.Builder.CreateBitCast(V, BaseClassType->getPointerTo());

  // FIXME: This should always use Ctor_Base as the ctor type!  (But that
  // causes crashes in tests.)
  CGF.EmitCXXConstructorCall(BaseInit->getConstructor(),
                             CtorType, V,
                             BaseInit->const_arg_begin(),
                             BaseInit->const_arg_end());
}

static void EmitMemberInitializer(CodeGenFunction &CGF,
                                  const CXXRecordDecl *ClassDecl,
                                  CXXBaseOrMemberInitializer *MemberInit) {
  assert(MemberInit->isMemberInitializer() &&
         "Must have member initializer!");
  
  // non-static data member initializers.
  FieldDecl *Field = MemberInit->getMember();
  QualType FieldType = CGF.getContext().getCanonicalType(Field->getType());

  llvm::Value *ThisPtr = CGF.LoadCXXThis();
  LValue LHS;
  if (FieldType->isReferenceType()) {
    // FIXME: This is really ugly; should be refactored somehow
    unsigned idx = CGF.CGM.getTypes().getLLVMFieldNo(Field);
    llvm::Value *V = CGF.Builder.CreateStructGEP(ThisPtr, idx, "tmp");
    assert(!FieldType.getObjCGCAttr() && "fields cannot have GC attrs");
    LHS = LValue::MakeAddr(V, CGF.MakeQualifiers(FieldType));
  } else {
    LHS = CGF.EmitLValueForField(ThisPtr, Field, ClassDecl->isUnion(), 0);
  }

  // If we are initializing an anonymous union field, drill down to the field.
  if (MemberInit->getAnonUnionMember()) {
    Field = MemberInit->getAnonUnionMember();
    LHS = CGF.EmitLValueForField(LHS.getAddress(), Field,
                                 /*IsUnion=*/true, 0);
    FieldType = Field->getType();
  }

  // If the field is an array, branch based on the element type.
  const ConstantArrayType *Array =
    CGF.getContext().getAsConstantArrayType(FieldType);
  if (Array)
    FieldType = CGF.getContext().getBaseElementType(FieldType);

  // We lose the constructor for anonymous union members, so handle them
  // explicitly.
  // FIXME: This is somwhat ugly.
  if (MemberInit->getAnonUnionMember() && FieldType->getAs<RecordType>()) {
    if (MemberInit->getNumArgs())
      CGF.EmitAggExpr(*MemberInit->arg_begin(), LHS.getAddress(),
                      LHS.isVolatileQualified());
    else
      CGF.EmitAggregateClear(LHS.getAddress(), Field->getType());
    return;
  }

  if (FieldType->getAs<RecordType>()) {
    assert(MemberInit->getConstructor() &&
           "EmitCtorPrologue - no constructor to initialize member");
    if (Array) {
      const llvm::Type *BasePtr = CGF.ConvertType(FieldType);
      BasePtr = llvm::PointerType::getUnqual(BasePtr);
      llvm::Value *BaseAddrPtr =
        CGF.Builder.CreateBitCast(LHS.getAddress(), BasePtr);
      CGF.EmitCXXAggrConstructorCall(MemberInit->getConstructor(),
                                     Array, BaseAddrPtr,
                                     MemberInit->const_arg_begin(),
                                     MemberInit->const_arg_end());
    }
    else
      CGF.EmitCXXConstructorCall(MemberInit->getConstructor(),
                                 Ctor_Complete, LHS.getAddress(),
                                 MemberInit->const_arg_begin(),
                                 MemberInit->const_arg_end());
    return;
  }

  assert(MemberInit->getNumArgs() == 1 && "Initializer count must be 1 only");
  Expr *RhsExpr = *MemberInit->arg_begin();
  RValue RHS;
  if (FieldType->isReferenceType()) {
    RHS = CGF.EmitReferenceBindingToExpr(RhsExpr, FieldType,
                                    /*IsInitializer=*/true);
    CGF.EmitStoreThroughLValue(RHS, LHS, FieldType);
  } else if (Array) {
    CGF.EmitMemSetToZero(LHS.getAddress(), Field->getType());
  } else if (!CGF.hasAggregateLLVMType(RhsExpr->getType())) {
    RHS = RValue::get(CGF.EmitScalarExpr(RhsExpr, true));
    CGF.EmitStoreThroughLValue(RHS, LHS, FieldType);
  } else if (RhsExpr->getType()->isAnyComplexType()) {
    CGF.EmitComplexExprIntoAddr(RhsExpr, LHS.getAddress(),
                                LHS.isVolatileQualified());
  } else {
    // Handle member function pointers; other aggregates shouldn't get this far.
    CGF.EmitAggExpr(RhsExpr, LHS.getAddress(), LHS.isVolatileQualified());
  }
}

/// EmitCtorPrologue - This routine generates necessary code to initialize
/// base classes and non-static data members belonging to this constructor.
/// FIXME: This needs to take a CXXCtorType.
void CodeGenFunction::EmitCtorPrologue(const CXXConstructorDecl *CD,
                                       CXXCtorType CtorType) {
  const CXXRecordDecl *ClassDecl = CD->getParent();
  
  // FIXME: Add vbase initialization
  
  for (CXXConstructorDecl::init_const_iterator B = CD->init_begin(),
       E = CD->init_end();
       B != E; ++B) {
    CXXBaseOrMemberInitializer *Member = (*B);
    
    assert(LiveTemporaries.empty() &&
           "Should not have any live temporaries at initializer start!");

    if (Member->isBaseInitializer())
      EmitBaseInitializer(*this, ClassDecl, Member, CtorType);
    else
      EmitMemberInitializer(*this, ClassDecl, Member);

    // Pop any live temporaries that the initializers might have pushed.
    while (!LiveTemporaries.empty())
      PopCXXTemporary();
  }

  InitializeVtablePtrs(ClassDecl);
}

/// EmitDtorEpilogue - Emit all code that comes at the end of class's
/// destructor. This is to call destructors on members and base classes
/// in reverse order of their construction.
/// FIXME: This needs to take a CXXDtorType.
void CodeGenFunction::EmitDtorEpilogue(const CXXDestructorDecl *DD,
                                       CXXDtorType DtorType) {
  assert(!DD->isTrivial() &&
         "Should not emit dtor epilogue for trivial dtor!");

  const CXXRecordDecl *ClassDecl = DD->getParent();

  // Collect the fields.
  llvm::SmallVector<const FieldDecl *, 16> FieldDecls;
  for (CXXRecordDecl::field_iterator I = ClassDecl->field_begin(),
       E = ClassDecl->field_end(); I != E; ++I) {
    const FieldDecl *Field = *I;
    
    QualType FieldType = getContext().getCanonicalType(Field->getType());
    FieldType = getContext().getBaseElementType(FieldType);
    
    const RecordType *RT = FieldType->getAs<RecordType>();
    if (!RT)
      continue;
    
    CXXRecordDecl *FieldClassDecl = cast<CXXRecordDecl>(RT->getDecl());
    if (FieldClassDecl->hasTrivialDestructor())
        continue;
    
    FieldDecls.push_back(Field);
  }
  
  // Now destroy the fields.
  for (size_t i = FieldDecls.size(); i > 0; --i) {
    const FieldDecl *Field = FieldDecls[i - 1];
    
    QualType FieldType = Field->getType();
    const ConstantArrayType *Array = 
      getContext().getAsConstantArrayType(FieldType);
    if (Array)
      FieldType = getContext().getBaseElementType(FieldType);
    
    const RecordType *RT = FieldType->getAs<RecordType>();
    CXXRecordDecl *FieldClassDecl = cast<CXXRecordDecl>(RT->getDecl());

    llvm::Value *ThisPtr = LoadCXXThis();

    LValue LHS = EmitLValueForField(ThisPtr, Field, 
                                    /*isUnion=*/false,
                                    // FIXME: Qualifiers?
                                    /*CVRQualifiers=*/0);
    if (Array) {
      const llvm::Type *BasePtr = ConvertType(FieldType);
      BasePtr = llvm::PointerType::getUnqual(BasePtr);
      llvm::Value *BaseAddrPtr =
        Builder.CreateBitCast(LHS.getAddress(), BasePtr);
      EmitCXXAggrDestructorCall(FieldClassDecl->getDestructor(getContext()),
                                Array, BaseAddrPtr);
    } else
      EmitCXXDestructorCall(FieldClassDecl->getDestructor(getContext()),
                            Dtor_Complete, LHS.getAddress());
  }

  // Destroy non-virtual bases.
  for (CXXRecordDecl::reverse_base_class_const_iterator I = 
        ClassDecl->bases_rbegin(), E = ClassDecl->bases_rend(); I != E; ++I) {
    const CXXBaseSpecifier &Base = *I;
    
    // Ignore virtual bases.
    if (Base.isVirtual())
      continue;
    
    CXXRecordDecl *BaseClassDecl
      = cast<CXXRecordDecl>(Base.getType()->getAs<RecordType>()->getDecl());
    
    // Ignore trivial destructors.
    if (BaseClassDecl->hasTrivialDestructor())
      continue;
    const CXXDestructorDecl *D = BaseClassDecl->getDestructor(getContext());
    
    llvm::Value *V = GetAddressOfBaseClass(LoadCXXThis(),
                                           ClassDecl, BaseClassDecl, 
                                           /*NullCheckValue=*/false);
    EmitCXXDestructorCall(D, Dtor_Base, V);
  }

  // If we're emitting a base destructor, we don't want to emit calls to the
  // virtual bases.
  if (DtorType == Dtor_Base)
    return;
  
  // Handle virtual bases.
  for (CXXRecordDecl::reverse_base_class_const_iterator I = 
       ClassDecl->vbases_rbegin(), E = ClassDecl->vbases_rend(); I != E; ++I) {
    const CXXBaseSpecifier &Base = *I;
    CXXRecordDecl *BaseClassDecl
    = cast<CXXRecordDecl>(Base.getType()->getAs<RecordType>()->getDecl());
    
    // Ignore trivial destructors.
    if (BaseClassDecl->hasTrivialDestructor())
      continue;
    const CXXDestructorDecl *D = BaseClassDecl->getDestructor(getContext());
    llvm::Value *V = GetAddressOfBaseClass(LoadCXXThis(),
                                           ClassDecl, BaseClassDecl, 
                                           /*NullCheckValue=*/false);
    EmitCXXDestructorCall(D, Dtor_Base, V);
  }
    
  // If we have a deleting destructor, emit a call to the delete operator.
  if (DtorType == Dtor_Deleting) {
    assert(DD->getOperatorDelete() && 
           "operator delete missing - EmitDtorEpilogue");
    EmitDeleteCall(DD->getOperatorDelete(), LoadCXXThis(),
                   getContext().getTagDeclType(ClassDecl));
  }
}

void CodeGenFunction::SynthesizeDefaultDestructor(const CXXDestructorDecl *Dtor,
                                                  CXXDtorType DtorType,
                                                  llvm::Function *Fn,
                                                  const FunctionArgList &Args) {
  assert(!Dtor->getParent()->hasUserDeclaredDestructor() &&
         "SynthesizeDefaultDestructor - destructor has user declaration");

  StartFunction(GlobalDecl(Dtor, DtorType), Dtor->getResultType(), Fn, Args, 
                SourceLocation());

  EmitDtorEpilogue(Dtor, DtorType);
  FinishFunction();
}

/// EmitCXXAggrConstructorCall - This routine essentially creates a (nested)
/// for-loop to call the default constructor on individual members of the
/// array. 
/// 'D' is the default constructor for elements of the array, 'ArrayTy' is the
/// array type and 'ArrayPtr' points to the beginning fo the array.
/// It is assumed that all relevant checks have been made by the caller.
void
CodeGenFunction::EmitCXXAggrConstructorCall(const CXXConstructorDecl *D,
                                          const ConstantArrayType *ArrayTy,
                                          llvm::Value *ArrayPtr,
                                          CallExpr::const_arg_iterator ArgBeg,
                                          CallExpr::const_arg_iterator ArgEnd) {

  const llvm::Type *SizeTy = ConvertType(getContext().getSizeType());
  llvm::Value * NumElements =
    llvm::ConstantInt::get(SizeTy, 
                           getContext().getConstantArrayElementCount(ArrayTy));

  EmitCXXAggrConstructorCall(D, NumElements, ArrayPtr, ArgBeg, ArgEnd);
}

void
CodeGenFunction::EmitCXXAggrConstructorCall(const CXXConstructorDecl *D,
                                          llvm::Value *NumElements,
                                          llvm::Value *ArrayPtr,
                                          CallExpr::const_arg_iterator ArgBeg,
                                          CallExpr::const_arg_iterator ArgEnd) {
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
  llvm::Value *Address = Builder.CreateInBoundsGEP(ArrayPtr, Counter, 
                                                   "arrayidx");

  // C++ [class.temporary]p4: 
  // There are two contexts in which temporaries are destroyed at a different
  // point than the end of the full-expression. The first context is when a
  // default constructor is called to initialize an element of an array. 
  // If the constructor has one or more default arguments, the destruction of 
  // every temporary created in a default argument expression is sequenced 
  // before the construction of the next array element, if any.
  
  // Keep track of the current number of live temporaries.
  unsigned OldNumLiveTemporaries = LiveTemporaries.size();

  EmitCXXConstructorCall(D, Ctor_Complete, Address, ArgBeg, ArgEnd);

  // Pop temporaries.
  while (LiveTemporaries.size() > OldNumLiveTemporaries)
    PopCXXTemporary();
  
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

/// EmitCXXAggrDestructorCall - calls the default destructor on array
/// elements in reverse order of construction.
void
CodeGenFunction::EmitCXXAggrDestructorCall(const CXXDestructorDecl *D,
                                           const ArrayType *Array,
                                           llvm::Value *This) {
  const ConstantArrayType *CA = dyn_cast<ConstantArrayType>(Array);
  assert(CA && "Do we support VLA for destruction ?");
  uint64_t ElementCount = getContext().getConstantArrayElementCount(CA);
  
  const llvm::Type *SizeLTy = ConvertType(getContext().getSizeType());
  llvm::Value* ElementCountPtr = llvm::ConstantInt::get(SizeLTy, ElementCount);
  EmitCXXAggrDestructorCall(D, ElementCountPtr, This);
}

/// EmitCXXAggrDestructorCall - calls the default destructor on array
/// elements in reverse order of construction.
void
CodeGenFunction::EmitCXXAggrDestructorCall(const CXXDestructorDecl *D,
                                           llvm::Value *UpperCount,
                                           llvm::Value *This) {
  const llvm::Type *SizeLTy = ConvertType(getContext().getSizeType());
  llvm::Value *One = llvm::ConstantInt::get(SizeLTy, 1);
  
  // Create a temporary for the loop index and initialize it with count of
  // array elements.
  llvm::Value *IndexPtr = CreateTempAlloca(SizeLTy, "loop.index");

  // Store the number of elements in the index pointer.
  Builder.CreateStore(UpperCount, IndexPtr);

  // Start the loop with a block that tests the condition.
  llvm::BasicBlock *CondBlock = createBasicBlock("for.cond");
  llvm::BasicBlock *AfterFor = createBasicBlock("for.end");

  EmitBlock(CondBlock);

  llvm::BasicBlock *ForBody = createBasicBlock("for.body");

  // Generate: if (loop-index != 0 fall to the loop body,
  // otherwise, go to the block after the for-loop.
  llvm::Value* zeroConstant =
    llvm::Constant::getNullValue(SizeLTy);
  llvm::Value *Counter = Builder.CreateLoad(IndexPtr);
  llvm::Value *IsNE = Builder.CreateICmpNE(Counter, zeroConstant,
                                            "isne");
  // If the condition is true, execute the body.
  Builder.CreateCondBr(IsNE, ForBody, AfterFor);

  EmitBlock(ForBody);

  llvm::BasicBlock *ContinueBlock = createBasicBlock("for.inc");
  // Inside the loop body, emit the constructor call on the array element.
  Counter = Builder.CreateLoad(IndexPtr);
  Counter = Builder.CreateSub(Counter, One);
  llvm::Value *Address = Builder.CreateInBoundsGEP(This, Counter, "arrayidx");
  EmitCXXDestructorCall(D, Dtor_Complete, Address);

  EmitBlock(ContinueBlock);

  // Emit the decrement of the loop counter.
  Counter = Builder.CreateLoad(IndexPtr);
  Counter = Builder.CreateSub(Counter, One, "dec");
  Builder.CreateStore(Counter, IndexPtr);

  // Finally, branch back up to the condition for the next iteration.
  EmitBranch(CondBlock);

  // Emit the fall-through block.
  EmitBlock(AfterFor, true);
}

/// GenerateCXXAggrDestructorHelper - Generates a helper function which when
/// invoked, calls the default destructor on array elements in reverse order of
/// construction.
llvm::Constant * 
CodeGenFunction::GenerateCXXAggrDestructorHelper(const CXXDestructorDecl *D,
                                                 const ArrayType *Array,
                                                 llvm::Value *This) {
  FunctionArgList Args;
  ImplicitParamDecl *Dst =
    ImplicitParamDecl::Create(getContext(), 0,
                              SourceLocation(), 0,
                              getContext().getPointerType(getContext().VoidTy));
  Args.push_back(std::make_pair(Dst, Dst->getType()));
  
  llvm::SmallString<16> Name;
  llvm::raw_svector_ostream(Name) << "__tcf_" << (++UniqueAggrDestructorCount);
  QualType R = getContext().VoidTy;
  const CGFunctionInfo &FI = CGM.getTypes().getFunctionInfo(R, Args);
  const llvm::FunctionType *FTy = CGM.getTypes().GetFunctionType(FI, false);
  llvm::Function *Fn =
    llvm::Function::Create(FTy, llvm::GlobalValue::InternalLinkage,
                           Name.str(),
                           &CGM.getModule());
  IdentifierInfo *II = &CGM.getContext().Idents.get(Name.str());
  FunctionDecl *FD = FunctionDecl::Create(getContext(),
                                          getContext().getTranslationUnitDecl(),
                                          SourceLocation(), II, R, 0,
                                          FunctionDecl::Static,
                                          false, true);
  StartFunction(FD, R, Fn, Args, SourceLocation());
  QualType BaseElementTy = getContext().getBaseElementType(Array);
  const llvm::Type *BasePtr = ConvertType(BaseElementTy);
  BasePtr = llvm::PointerType::getUnqual(BasePtr);
  llvm::Value *BaseAddrPtr = Builder.CreateBitCast(This, BasePtr);
  EmitCXXAggrDestructorCall(D, Array, BaseAddrPtr);
  FinishFunction();
  llvm::Type *Ptr8Ty = llvm::PointerType::get(llvm::Type::getInt8Ty(VMContext),
                                              0);
  llvm::Constant *m = llvm::ConstantExpr::getBitCast(Fn, Ptr8Ty);
  return m;
}

void
CodeGenFunction::EmitCXXConstructorCall(const CXXConstructorDecl *D,
                                        CXXCtorType Type,
                                        llvm::Value *This,
                                        CallExpr::const_arg_iterator ArgBeg,
                                        CallExpr::const_arg_iterator ArgEnd) {
  if (D->isCopyConstructor()) {
    const CXXRecordDecl *ClassDecl = cast<CXXRecordDecl>(D->getDeclContext());
    if (ClassDecl->hasTrivialCopyConstructor()) {
      assert(!ClassDecl->hasUserDeclaredCopyConstructor() &&
             "EmitCXXConstructorCall - user declared copy constructor");
      const Expr *E = (*ArgBeg);
      QualType Ty = E->getType();
      llvm::Value *Src = EmitLValue(E).getAddress();
      EmitAggregateCopy(This, Src, Ty);
      return;
    }
  } else if (D->isTrivial()) {
    // FIXME: Track down why we're trying to generate calls to the trivial
    // default constructor!
    return;
  }

  llvm::Value *Callee = CGM.GetAddrOfCXXConstructor(D, Type);

  EmitCXXMemberCall(D, Callee, ReturnValueSlot(), This, ArgBeg, ArgEnd);
}

void CodeGenFunction::EmitCXXDestructorCall(const CXXDestructorDecl *DD,
                                            CXXDtorType Type,
                                            llvm::Value *This) {
  llvm::Value *Callee = CGM.GetAddrOfCXXDestructor(DD, Type);
  
  CallArgList Args;

  // Push the this ptr.
  Args.push_back(std::make_pair(RValue::get(This),
                                DD->getThisType(getContext())));
  
  // Add a VTT parameter if necessary.
  // FIXME: This should not be a dummy null parameter!
  if (Type == Dtor_Base && DD->getParent()->getNumVBases() != 0) {
    QualType T = getContext().getPointerType(getContext().VoidPtrTy);
    
    Args.push_back(std::make_pair(RValue::get(CGM.EmitNullConstant(T)), T));
  }

  // FIXME: We should try to share this code with EmitCXXMemberCall.
  
  QualType ResultType = DD->getType()->getAs<FunctionType>()->getResultType();
  EmitCall(CGM.getTypes().getFunctionInfo(ResultType, Args), Callee, 
           ReturnValueSlot(), Args, DD);
}

llvm::Value *
CodeGenFunction::GetVirtualCXXBaseClassOffset(llvm::Value *This,
                                              const CXXRecordDecl *ClassDecl,
                                           const CXXRecordDecl *BaseClassDecl) {
  const llvm::Type *Int8PtrTy = 
    llvm::Type::getInt8Ty(VMContext)->getPointerTo();

  llvm::Value *VTablePtr = Builder.CreateBitCast(This, 
                                                 Int8PtrTy->getPointerTo());
  VTablePtr = Builder.CreateLoad(VTablePtr, "vtable");

  int64_t VBaseOffsetIndex = 
    CGM.getVtableInfo().getVirtualBaseOffsetIndex(ClassDecl, BaseClassDecl);
  
  llvm::Value *VBaseOffsetPtr = 
    Builder.CreateConstGEP1_64(VTablePtr, VBaseOffsetIndex, "vbase.offset.ptr");
  const llvm::Type *PtrDiffTy = 
    ConvertType(getContext().getPointerDiffType());
  
  VBaseOffsetPtr = Builder.CreateBitCast(VBaseOffsetPtr, 
                                         PtrDiffTy->getPointerTo());
                                         
  llvm::Value *VBaseOffset = Builder.CreateLoad(VBaseOffsetPtr, "vbase.offset");
  
  return VBaseOffset;
}

void CodeGenFunction::InitializeVtablePtrs(const CXXRecordDecl *ClassDecl) {
  if (!ClassDecl->isDynamicClass())
    return;

  llvm::Constant *Vtable = CGM.getVtableInfo().getVtable(ClassDecl);
  CodeGenModule::AddrSubMap_t& AddressPoints =
      *(*CGM.AddressPoints[ClassDecl])[ClassDecl];
  llvm::Value *ThisPtr = LoadCXXThis();
  const ASTRecordLayout &Layout = getContext().getASTRecordLayout(ClassDecl);

  // Store address points for virtual bases
  for (CXXRecordDecl::base_class_const_iterator I = 
       ClassDecl->vbases_begin(), E = ClassDecl->vbases_end(); I != E; ++I) {
    const CXXBaseSpecifier &Base = *I;
    CXXRecordDecl *BaseClassDecl
      = cast<CXXRecordDecl>(Base.getType()->getAs<RecordType>()->getDecl());
    uint64_t Offset = Layout.getVBaseClassOffset(BaseClassDecl);
    InitializeVtablePtrsRecursive(BaseClassDecl, Vtable, AddressPoints,
                                  ThisPtr, Offset);
  }

  // Store address points for non-virtual bases and current class
  InitializeVtablePtrsRecursive(ClassDecl, Vtable, AddressPoints, ThisPtr, 0);
}

void CodeGenFunction::InitializeVtablePtrsRecursive(
        const CXXRecordDecl *ClassDecl,
        llvm::Constant *Vtable,
        CodeGenModule::AddrSubMap_t& AddressPoints,
        llvm::Value *ThisPtr,
        uint64_t Offset) {
  if (!ClassDecl->isDynamicClass())
    return;

  // Store address points for non-virtual bases
  const ASTRecordLayout &Layout = getContext().getASTRecordLayout(ClassDecl);
  for (CXXRecordDecl::base_class_const_iterator I = 
       ClassDecl->bases_begin(), E = ClassDecl->bases_end(); I != E; ++I) {
    const CXXBaseSpecifier &Base = *I;
    if (Base.isVirtual())
      continue;
    CXXRecordDecl *BaseClassDecl
      = cast<CXXRecordDecl>(Base.getType()->getAs<RecordType>()->getDecl());
    uint64_t NewOffset = Offset + Layout.getBaseClassOffset(BaseClassDecl);
    InitializeVtablePtrsRecursive(BaseClassDecl, Vtable, AddressPoints,
                                  ThisPtr, NewOffset);
  }

  // Compute the address point
  assert(AddressPoints.count(std::make_pair(ClassDecl, Offset)) &&
         "Missing address point for class");
  uint64_t AddressPoint = AddressPoints[std::make_pair(ClassDecl, Offset)];
  llvm::Value *VtableAddressPoint =
      Builder.CreateConstInBoundsGEP2_64(Vtable, 0, AddressPoint);

  // Compute the address to store the address point
  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(CGM.getLLVMContext());
  llvm::Value *VtableField = Builder.CreateBitCast(ThisPtr, Int8PtrTy);
  VtableField = Builder.CreateConstInBoundsGEP1_64(VtableField, Offset/8);
  const llvm::Type *AddressPointPtrTy =
      VtableAddressPoint->getType()->getPointerTo();
  VtableField = Builder.CreateBitCast(VtableField, AddressPointPtrTy);

  // Store address point
  Builder.CreateStore(VtableAddressPoint, VtableField);
}
