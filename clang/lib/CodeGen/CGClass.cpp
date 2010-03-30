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
#include "clang/AST/StmtCXX.h"

using namespace clang;
using namespace CodeGen;

static uint64_t 
ComputeNonVirtualBaseClassOffset(ASTContext &Context,
                                 const CXXBasePath &Path,
                                 unsigned Start) {
  uint64_t Offset = 0;

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
CodeGenModule::GetNonVirtualBaseClassOffset(const CXXRecordDecl *Class,
                                            const CXXRecordDecl *BaseClass) {
  if (Class == BaseClass)
    return 0;

  CXXBasePaths Paths(/*FindAmbiguities=*/false,
                     /*RecordPaths=*/true, /*DetectVirtual=*/false);
  if (!const_cast<CXXRecordDecl *>(Class)->
        isDerivedFrom(const_cast<CXXRecordDecl *>(BaseClass), Paths)) {
    assert(false && "Class must be derived from the passed in base class!");
    return 0;
  }

  uint64_t Offset = ComputeNonVirtualBaseClassOffset(getContext(),
                                                     Paths.front(), 0);
  if (!Offset)
    return 0;

  const llvm::Type *PtrDiffTy = 
    Types.ConvertType(getContext().getPointerDiffType());

  return llvm::ConstantInt::get(PtrDiffTy, Offset);
}

/// Gets the address of a virtual base class within a complete object.
/// This should only be used for (1) non-virtual bases or (2) virtual bases
/// when the type is known to be complete (e.g. in complete destructors).
///
/// The object pointed to by 'This' is assumed to be non-null.
llvm::Value *
CodeGenFunction::GetAddressOfBaseOfCompleteClass(llvm::Value *This,
                                                 bool isBaseVirtual,
                                                 const CXXRecordDecl *Derived,
                                                 const CXXRecordDecl *Base) {
  // 'this' must be a pointer (in some address space) to Derived.
  assert(This->getType()->isPointerTy() &&
         cast<llvm::PointerType>(This->getType())->getElementType()
           == ConvertType(Derived));

  // Compute the offset of the virtual base.
  uint64_t Offset;
  const ASTRecordLayout &Layout = getContext().getASTRecordLayout(Derived);
  if (isBaseVirtual)
    Offset = Layout.getVBaseClassOffset(Base);
  else
    Offset = Layout.getBaseClassOffset(Base);

  // Shift and cast down to the base type.
  // TODO: for complete types, this should be possible with a GEP.
  llvm::Value *V = This;
  if (Offset) {
    const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(getLLVMContext());
    V = Builder.CreateBitCast(V, Int8PtrTy);
    V = Builder.CreateConstInBoundsGEP1_64(V, Offset / 8);
  }
  V = Builder.CreateBitCast(V, ConvertType(Base)->getPointerTo());

  return V;
}

llvm::Value *
CodeGenFunction::GetAddressOfBaseClass(llvm::Value *Value,
                                       const CXXRecordDecl *Class,
                                       const CXXRecordDecl *BaseClass,
                                       bool NullCheckValue) {
  QualType BTy =
    getContext().getCanonicalType(
      getContext().getTypeDeclType(BaseClass));
  const llvm::Type *BasePtrTy = llvm::PointerType::getUnqual(ConvertType(BTy));

  if (Class == BaseClass) {
    // Just cast back.
    return Builder.CreateBitCast(Value, BasePtrTy);
  }

  CXXBasePaths Paths(/*FindAmbiguities=*/false,
                     /*RecordPaths=*/true, /*DetectVirtual=*/false);
  if (!const_cast<CXXRecordDecl *>(Class)->
        isDerivedFrom(const_cast<CXXRecordDecl *>(BaseClass), Paths)) {
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

  uint64_t Offset = 
    ComputeNonVirtualBaseClassOffset(getContext(), Paths.front(), Start);
  
  if (!Offset && !VBase) {
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
  
  if (VBase)
    VirtualOffset = GetVirtualBaseClassOffset(Value, Class, VBase);

  const llvm::Type *PtrDiffTy = ConvertType(getContext().getPointerDiffType());
  llvm::Value *NonVirtualOffset = 0;
  if (Offset)
    NonVirtualOffset = llvm::ConstantInt::get(PtrDiffTy, Offset);
  
  llvm::Value *BaseOffset;
  if (VBase) {
    if (NonVirtualOffset)
      BaseOffset = Builder.CreateAdd(VirtualOffset, NonVirtualOffset);
    else
      BaseOffset = VirtualOffset;
  } else
    BaseOffset = NonVirtualOffset;
  
  // Apply the base offset.
  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(getLLVMContext());
  Value = Builder.CreateBitCast(Value, Int8PtrTy);
  Value = Builder.CreateGEP(Value, BaseOffset, "add.ptr");
  
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
                                          const CXXRecordDecl *Class,
                                          const CXXRecordDecl *DerivedClass,
                                          bool NullCheckValue) {
  QualType DerivedTy =
    getContext().getCanonicalType(
    getContext().getTypeDeclType(const_cast<CXXRecordDecl*>(DerivedClass)));
  const llvm::Type *DerivedPtrTy = ConvertType(DerivedTy)->getPointerTo();
  
  if (Class == DerivedClass) {
    // Just cast back.
    return Builder.CreateBitCast(Value, DerivedPtrTy);
  }

  llvm::Value *NonVirtualOffset =
    CGM.GetNonVirtualBaseClassOffset(DerivedClass, Class);
  
  if (!NonVirtualOffset) {
    // No offset, we can just cast back.
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
  
  // Apply the offset.
  Value = Builder.CreatePtrToInt(Value, NonVirtualOffset->getType());
  Value = Builder.CreateSub(Value, NonVirtualOffset);
  Value = Builder.CreateIntToPtr(Value, DerivedPtrTy);

  // Just cast.
  Value = Builder.CreateBitCast(Value, DerivedPtrTy);

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

/// EmitCopyCtorCall - Emit a call to a copy constructor.
static void
EmitCopyCtorCall(CodeGenFunction &CGF,
                 const CXXConstructorDecl *CopyCtor, CXXCtorType CopyCtorType,
                 llvm::Value *ThisPtr, llvm::Value *VTT, llvm::Value *Src) {
  llvm::Value *Callee = CGF.CGM.GetAddrOfCXXConstructor(CopyCtor, CopyCtorType);

  CallArgList CallArgs;

  // Push the this ptr.
  CallArgs.push_back(std::make_pair(RValue::get(ThisPtr),
                                    CopyCtor->getThisType(CGF.getContext())));
  
  // Push the VTT parameter if necessary.
  if (VTT) {
    QualType T = CGF.getContext().getPointerType(CGF.getContext().VoidPtrTy);
    CallArgs.push_back(std::make_pair(RValue::get(VTT), T));
  }
 
  // Push the Src ptr.
  CallArgs.push_back(std::make_pair(RValue::get(Src),
                                    CopyCtor->getParamDecl(0)->getType()));


  {
    CodeGenFunction::CXXTemporariesCleanupScope Scope(CGF);

    // If the copy constructor has default arguments, emit them.
    for (unsigned I = 1, E = CopyCtor->getNumParams(); I < E; ++I) {
      const ParmVarDecl *Param = CopyCtor->getParamDecl(I);
      const Expr *DefaultArgExpr = Param->getDefaultArg();

      assert(DefaultArgExpr && "Ctor parameter must have default arg!");

      QualType ArgType = Param->getType();
      CallArgs.push_back(std::make_pair(CGF.EmitCallArg(DefaultArgExpr, 
                                                        ArgType),
                                        ArgType));
    }

    const FunctionProtoType *FPT =
      CopyCtor->getType()->getAs<FunctionProtoType>();
    CGF.EmitCall(CGF.CGM.getTypes().getFunctionInfo(CallArgs, FPT),
                 Callee, ReturnValueSlot(), CallArgs, CopyCtor);
  }
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
           BaseClassDecl->getCopyConstructor(getContext(), 0))
    EmitCopyCtorCall(*this, BaseCopyCtor, Ctor_Complete, Dest, 0, Src);

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
    BaseClassDecl->hasConstCopyAssignment(getContext(), MD);
    assert(MD && "EmitClassAggrCopyAssignment - No user assign");
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
    QualType SrcTy = MD->getParamDecl(0)->getType();
    RValue SrcValue = SrcTy->isReferenceType() ? RValue::get(Src) :
                                                 RValue::getAggregate(Src);
    CallArgs.push_back(std::make_pair(SrcValue, SrcTy));
    EmitCall(CGM.getTypes().getFunctionInfo(CallArgs, FPT),
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

/// GetVTTParameter - Return the VTT parameter that should be passed to a
/// base constructor/destructor with virtual bases.
static llvm::Value *GetVTTParameter(CodeGenFunction &CGF, GlobalDecl GD) {
  if (!CodeGenVTables::needsVTTParameter(GD)) {
    // This constructor/destructor does not need a VTT parameter.
    return 0;
  }
  
  const CXXRecordDecl *RD = cast<CXXMethodDecl>(CGF.CurFuncDecl)->getParent();
  const CXXRecordDecl *Base = cast<CXXMethodDecl>(GD.getDecl())->getParent();

  llvm::Value *VTT;

  uint64_t SubVTTIndex;

  // If the record matches the base, this is the complete ctor/dtor
  // variant calling the base variant in a class with virtual bases.
  if (RD == Base) {
    assert(!CodeGenVTables::needsVTTParameter(CGF.CurGD) &&
           "doing no-op VTT offset in base dtor/ctor?");
    SubVTTIndex = 0;
  } else {
    SubVTTIndex = CGF.CGM.getVTables().getSubVTTIndex(RD, Base);
    assert(SubVTTIndex != 0 && "Sub-VTT index must be greater than zero!");
  }
  
  if (CodeGenVTables::needsVTTParameter(CGF.CurGD)) {
    // A VTT parameter was passed to the constructor, use it.
    VTT = CGF.LoadCXXVTT();
    VTT = CGF.Builder.CreateConstInBoundsGEP1_64(VTT, SubVTTIndex);
  } else {
    // We're the complete constructor, so get the VTT by name.
    VTT = CGF.CGM.getVTables().getVTT(RD);
    VTT = CGF.Builder.CreateConstInBoundsGEP2_64(VTT, 0, SubVTTIndex);
  }

  return VTT;
}

                                    
/// EmitClassMemberwiseCopy - This routine generates code to copy a class
/// object from SrcValue to DestValue. Copying can be either a bitwise copy
/// or via a copy constructor call.
void CodeGenFunction::EmitClassMemberwiseCopy(
                        llvm::Value *Dest, llvm::Value *Src,
                        const CXXRecordDecl *ClassDecl,
                        const CXXRecordDecl *BaseClassDecl, QualType Ty) {
  CXXCtorType CtorType = Ctor_Complete;
  
  if (ClassDecl) {
    Dest = GetAddressOfBaseClass(Dest, ClassDecl, BaseClassDecl,
                                 /*NullCheckValue=*/false);
    Src = GetAddressOfBaseClass(Src, ClassDecl, BaseClassDecl,
                                /*NullCheckValue=*/false);

    // We want to call the base constructor.
    CtorType = Ctor_Base;
  }
  if (BaseClassDecl->hasTrivialCopyConstructor()) {
    EmitAggregateCopy(Dest, Src, Ty);
    return;
  }

  CXXConstructorDecl *BaseCopyCtor =
    BaseClassDecl->getCopyConstructor(getContext(), 0);
  if (!BaseCopyCtor)
    return;

  llvm::Value *VTT = GetVTTParameter(*this, GlobalDecl(BaseCopyCtor, CtorType));
  EmitCopyCtorCall(*this, BaseCopyCtor, CtorType, Dest, VTT, Src);
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
  BaseClassDecl->hasConstCopyAssignment(getContext(), MD);
  assert(MD && "EmitClassCopyAssignment - missing copy assign");

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
  QualType SrcTy = MD->getParamDecl(0)->getType();
  RValue SrcValue = SrcTy->isReferenceType() ? RValue::get(Src) :
                                               RValue::getAggregate(Src);
  CallArgs.push_back(std::make_pair(SrcValue, SrcTy));
  EmitCall(CGM.getTypes().getFunctionInfo(CallArgs, FPT),
           Callee, ReturnValueSlot(), CallArgs, MD);
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
CodeGenFunction::SynthesizeCXXCopyConstructor(const FunctionArgList &Args) {
  const CXXConstructorDecl *Ctor = cast<CXXConstructorDecl>(CurGD.getDecl());
  const CXXRecordDecl *ClassDecl = Ctor->getParent();
  assert(!ClassDecl->hasUserDeclaredCopyConstructor() &&
      "SynthesizeCXXCopyConstructor - copy constructor has definition already");
  assert(!Ctor->isTrivial() && "shouldn't need to generate trivial ctor");

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
      LValue LHS = EmitLValueForField(LoadOfThis, Field, 0);
      LValue RHS = EmitLValueForField(LoadOfSrc, Field, 0);
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
    
    // Do a built-in assignment of scalar data members.
    LValue LHS = EmitLValueForFieldInitialization(LoadOfThis, Field, 0);
    LValue RHS = EmitLValueForFieldInitialization(LoadOfSrc, Field, 0);

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

  InitializeVTablePointers(ClassDecl);
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
void CodeGenFunction::SynthesizeCXXCopyAssignment(const FunctionArgList &Args) {
  const CXXMethodDecl *CD = cast<CXXMethodDecl>(CurGD.getDecl());
  const CXXRecordDecl *ClassDecl = cast<CXXRecordDecl>(CD->getDeclContext());
  assert(!ClassDecl->hasUserDeclaredCopyAssignment() &&
         "SynthesizeCXXCopyAssignment - copy assignment has user declaration");

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
      LValue LHS = EmitLValueForField(LoadOfThis, *Field, 0);
      LValue RHS = EmitLValueForField(LoadOfSrc, *Field, 0);
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
    LValue LHS = EmitLValueForField(LoadOfThis, *Field, 0);
    LValue RHS = EmitLValueForField(LoadOfSrc, *Field, 0);
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

  // We can pretend to be a complete class because it only matters for
  // virtual bases, and we only do virtual bases for complete ctors.
  llvm::Value *V = ThisPtr;
  V = CGF.GetAddressOfBaseOfCompleteClass(V, isBaseVirtual,
                                          ClassDecl, BaseClassDecl);

  CGF.EmitAggExpr(BaseInit->getInit(), V, false, false, true);
  
  if (CGF.Exceptions && !BaseClassDecl->hasTrivialDestructor()) {
    // FIXME: Is this OK for C++0x delegating constructors?
    CodeGenFunction::EHCleanupBlock Cleanup(CGF);

    CXXDestructorDecl *DD = BaseClassDecl->getDestructor(CGF.getContext());
    CGF.EmitCXXDestructorCall(DD, Dtor_Base, V);
  }
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
  LValue LHS = CGF.EmitLValueForFieldInitialization(ThisPtr, Field, 0);
  
  // If we are initializing an anonymous union field, drill down to the field.
  if (MemberInit->getAnonUnionMember()) {
    Field = MemberInit->getAnonUnionMember();
    LHS = CGF.EmitLValueForField(LHS.getAddress(), Field, 0);
    FieldType = Field->getType();
  }

  // FIXME: If there's no initializer and the CXXBaseOrMemberInitializer
  // was implicitly generated, we shouldn't be zeroing memory.
  RValue RHS;
  if (FieldType->isReferenceType()) {
    RHS = CGF.EmitReferenceBindingToExpr(MemberInit->getInit(),
                                         /*IsInitializer=*/true);
    CGF.EmitStoreThroughLValue(RHS, LHS, FieldType);
  } else if (FieldType->isArrayType() && !MemberInit->getInit()) {
    CGF.EmitMemSetToZero(LHS.getAddress(), Field->getType());
  } else if (!CGF.hasAggregateLLVMType(Field->getType())) {
    RHS = RValue::get(CGF.EmitScalarExpr(MemberInit->getInit(), true));
    CGF.EmitStoreThroughLValue(RHS, LHS, FieldType);
  } else if (MemberInit->getInit()->getType()->isAnyComplexType()) {
    CGF.EmitComplexExprIntoAddr(MemberInit->getInit(), LHS.getAddress(),
                                LHS.isVolatileQualified());
  } else {
    CGF.EmitAggExpr(MemberInit->getInit(), LHS.getAddress(), 
                    LHS.isVolatileQualified(), false, true);
    
    if (!CGF.Exceptions)
      return;

    const RecordType *RT = FieldType->getAs<RecordType>();
    if (!RT)
      return;
    
    CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
    if (!RD->hasTrivialDestructor()) {
      // FIXME: Is this OK for C++0x delegating constructors?
      CodeGenFunction::EHCleanupBlock Cleanup(CGF);
      
      llvm::Value *ThisPtr = CGF.LoadCXXThis();
      LValue LHS = CGF.EmitLValueForField(ThisPtr, Field, 0);

      CXXDestructorDecl *DD = RD->getDestructor(CGF.getContext());
      CGF.EmitCXXDestructorCall(DD, Dtor_Complete, LHS.getAddress());
    }
  }
}

/// Checks whether the given constructor is a valid subject for the
/// complete-to-base constructor delegation optimization, i.e.
/// emitting the complete constructor as a simple call to the base
/// constructor.
static bool IsConstructorDelegationValid(const CXXConstructorDecl *Ctor) {

  // Currently we disable the optimization for classes with virtual
  // bases because (1) the addresses of parameter variables need to be
  // consistent across all initializers but (2) the delegate function
  // call necessarily creates a second copy of the parameter variable.
  //
  // The limiting example (purely theoretical AFAIK):
  //   struct A { A(int &c) { c++; } };
  //   struct B : virtual A {
  //     B(int count) : A(count) { printf("%d\n", count); }
  //   };
  // ...although even this example could in principle be emitted as a
  // delegation since the address of the parameter doesn't escape.
  if (Ctor->getParent()->getNumVBases()) {
    // TODO: white-list trivial vbase initializers.  This case wouldn't
    // be subject to the restrictions below.

    // TODO: white-list cases where:
    //  - there are no non-reference parameters to the constructor
    //  - the initializers don't access any non-reference parameters
    //  - the initializers don't take the address of non-reference
    //    parameters
    //  - etc.
    // If we ever add any of the above cases, remember that:
    //  - function-try-blocks will always blacklist this optimization
    //  - we need to perform the constructor prologue and cleanup in
    //    EmitConstructorBody.

    return false;
  }

  // We also disable the optimization for variadic functions because
  // it's impossible to "re-pass" varargs.
  if (Ctor->getType()->getAs<FunctionProtoType>()->isVariadic())
    return false;

  return true;
}

/// EmitConstructorBody - Emits the body of the current constructor.
void CodeGenFunction::EmitConstructorBody(FunctionArgList &Args) {
  const CXXConstructorDecl *Ctor = cast<CXXConstructorDecl>(CurGD.getDecl());
  CXXCtorType CtorType = CurGD.getCtorType();

  // Before we go any further, try the complete->base constructor
  // delegation optimization.
  if (CtorType == Ctor_Complete && IsConstructorDelegationValid(Ctor)) {
    EmitDelegateCXXConstructorCall(Ctor, Ctor_Base, Args);
    return;
  }

  Stmt *Body = Ctor->getBody();

  // Enter the function-try-block before the constructor prologue if
  // applicable.
  CXXTryStmtInfo TryInfo;
  bool IsTryBody = (Body && isa<CXXTryStmt>(Body));

  if (IsTryBody)
    TryInfo = EnterCXXTryStmt(*cast<CXXTryStmt>(Body));

  unsigned CleanupStackSize = CleanupEntries.size();

  // Emit the constructor prologue, i.e. the base and member
  // initializers.
  EmitCtorPrologue(Ctor, CtorType);

  // Emit the body of the statement.
  if (IsTryBody)
    EmitStmt(cast<CXXTryStmt>(Body)->getTryBlock());
  else if (Body)
    EmitStmt(Body);
  else {
    assert(Ctor->isImplicit() && "bodyless ctor not implicit");
    if (!Ctor->isDefaultConstructor()) {
      assert(Ctor->isCopyConstructor());
      SynthesizeCXXCopyConstructor(Args);
    }
  }

  // Emit any cleanup blocks associated with the member or base
  // initializers, which includes (along the exceptional path) the
  // destructors for those members and bases that were fully
  // constructed.
  EmitCleanupBlocks(CleanupStackSize);

  if (IsTryBody)
    ExitCXXTryStmt(*cast<CXXTryStmt>(Body), TryInfo);
}

/// EmitCtorPrologue - This routine generates necessary code to initialize
/// base classes and non-static data members belonging to this constructor.
void CodeGenFunction::EmitCtorPrologue(const CXXConstructorDecl *CD,
                                       CXXCtorType CtorType) {
  const CXXRecordDecl *ClassDecl = CD->getParent();

  llvm::SmallVector<CXXBaseOrMemberInitializer *, 8> MemberInitializers;
  
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
      MemberInitializers.push_back(Member);
  }

  InitializeVTablePointers(ClassDecl);

  for (unsigned I = 0, E = MemberInitializers.size(); I != E; ++I) {
    assert(LiveTemporaries.empty() &&
           "Should not have any live temporaries at initializer start!");
    
    EmitMemberInitializer(*this, ClassDecl, MemberInitializers[I]);
  }
}

/// EmitDestructorBody - Emits the body of the current destructor.
void CodeGenFunction::EmitDestructorBody(FunctionArgList &Args) {
  const CXXDestructorDecl *Dtor = cast<CXXDestructorDecl>(CurGD.getDecl());
  CXXDtorType DtorType = CurGD.getDtorType();

  Stmt *Body = Dtor->getBody();

  // If the body is a function-try-block, enter the try before
  // anything else --- unless we're in a deleting destructor, in which
  // case we're just going to call the complete destructor and then
  // call operator delete() on the way out.
  CXXTryStmtInfo TryInfo;
  bool isTryBody = (DtorType != Dtor_Deleting &&
                    Body && isa<CXXTryStmt>(Body));
  if (isTryBody)
    TryInfo = EnterCXXTryStmt(*cast<CXXTryStmt>(Body));

  llvm::BasicBlock *DtorEpilogue = createBasicBlock("dtor.epilogue");
  PushCleanupBlock(DtorEpilogue);

  bool SkipBody = false; // should get jump-threaded

  // If this is the deleting variant, just invoke the complete
  // variant, then call the appropriate operator delete() on the way
  // out.
  if (DtorType == Dtor_Deleting) {
    EmitCXXDestructorCall(Dtor, Dtor_Complete, LoadCXXThis());
    SkipBody = true;

  // If this is the complete variant, just invoke the base variant;
  // the epilogue will destruct the virtual bases.  But we can't do
  // this optimization if the body is a function-try-block, because
  // we'd introduce *two* handler blocks.
  } else if (!isTryBody && DtorType == Dtor_Complete) {
    EmitCXXDestructorCall(Dtor, Dtor_Base, LoadCXXThis());
    SkipBody = true;
      
  // Otherwise, we're in the base variant, so we need to ensure the
  // vtable ptrs are right before emitting the body.
  } else {
    InitializeVTablePointers(Dtor->getParent());
  }

  // Emit the body of the statement.
  if (SkipBody)
    (void) 0;
  else if (isTryBody)
    EmitStmt(cast<CXXTryStmt>(Body)->getTryBlock());
  else if (Body)
    EmitStmt(Body);
  else {
    assert(Dtor->isImplicit() && "bodyless dtor not implicit");
    // nothing to do besides what's in the epilogue
  }

  // Jump to the cleanup block.
  CleanupBlockInfo Info = PopCleanupBlock();
  assert(Info.CleanupBlock == DtorEpilogue && "Block mismatch!");
  EmitBlock(DtorEpilogue);

  // Emit the destructor epilogue now.  If this is a complete
  // destructor with a function-try-block, perform the base epilogue
  // as well.
  if (isTryBody && DtorType == Dtor_Complete)
    EmitDtorEpilogue(Dtor, Dtor_Base);
  EmitDtorEpilogue(Dtor, DtorType);

  // Link up the cleanup information.
  if (Info.SwitchBlock)
    EmitBlock(Info.SwitchBlock);
  if (Info.EndBlock)
    EmitBlock(Info.EndBlock);

  // Exit the try if applicable.
  if (isTryBody)
    ExitCXXTryStmt(*cast<CXXTryStmt>(Body), TryInfo);
}

/// EmitDtorEpilogue - Emit all code that comes at the end of class's
/// destructor. This is to call destructors on members and base classes
/// in reverse order of their construction.
void CodeGenFunction::EmitDtorEpilogue(const CXXDestructorDecl *DD,
                                       CXXDtorType DtorType) {
  assert(!DD->isTrivial() &&
         "Should not emit dtor epilogue for trivial dtor!");

  const CXXRecordDecl *ClassDecl = DD->getParent();

  // In a deleting destructor, we've already called the complete
  // destructor as a subroutine, so we just have to delete the
  // appropriate value.
  if (DtorType == Dtor_Deleting) {
    assert(DD->getOperatorDelete() && 
           "operator delete missing - EmitDtorEpilogue");
    EmitDeleteCall(DD->getOperatorDelete(), LoadCXXThis(),
                   getContext().getTagDeclType(ClassDecl));
    return;
  }

  // For complete destructors, we've already called the base
  // destructor (in GenerateBody), so we just need to destruct all the
  // virtual bases.
  if (DtorType == Dtor_Complete) {
    // Handle virtual bases.
    for (CXXRecordDecl::reverse_base_class_const_iterator I = 
           ClassDecl->vbases_rbegin(), E = ClassDecl->vbases_rend();
              I != E; ++I) {
      const CXXBaseSpecifier &Base = *I;
      CXXRecordDecl *BaseClassDecl
        = cast<CXXRecordDecl>(Base.getType()->getAs<RecordType>()->getDecl());
    
      // Ignore trivial destructors.
      if (BaseClassDecl->hasTrivialDestructor())
        continue;
      const CXXDestructorDecl *D = BaseClassDecl->getDestructor(getContext());
      llvm::Value *V = GetAddressOfBaseOfCompleteClass(LoadCXXThis(),
                                                       true,
                                                       ClassDecl,
                                                       BaseClassDecl);
      EmitCXXDestructorCall(D, Dtor_Base, V);
    }
    return;
  }

  assert(DtorType == Dtor_Base);

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
  {
    CXXTemporariesCleanupScope Scope(*this);

    EmitCXXConstructorCall(D, Ctor_Complete, Address, ArgBeg, ArgEnd);
  }

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
  const CGFunctionInfo &FI
      = CGM.getTypes().getFunctionInfo(R, Args, FunctionType::ExtInfo());
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
  if (D->isTrivial()) {
    if (ArgBeg == ArgEnd) {
      // Trivial default constructor, no codegen required.
      assert(D->isDefaultConstructor() &&
             "trivial 0-arg ctor not a default ctor");
      return;
    }

    assert(ArgBeg + 1 == ArgEnd && "unexpected argcount for trivial ctor");
    assert(D->isCopyConstructor() && "trivial 1-arg ctor not a copy ctor");

    const Expr *E = (*ArgBeg);
    QualType Ty = E->getType();
    llvm::Value *Src = EmitLValue(E).getAddress();
    EmitAggregateCopy(This, Src, Ty);
    return;
  }

  llvm::Value *VTT = GetVTTParameter(*this, GlobalDecl(D, Type));
  llvm::Value *Callee = CGM.GetAddrOfCXXConstructor(D, Type);

  EmitCXXMemberCall(D, Callee, ReturnValueSlot(), This, VTT, ArgBeg, ArgEnd);
}

void
CodeGenFunction::EmitDelegateCXXConstructorCall(const CXXConstructorDecl *Ctor,
                                                CXXCtorType CtorType,
                                                const FunctionArgList &Args) {
  CallArgList DelegateArgs;

  FunctionArgList::const_iterator I = Args.begin(), E = Args.end();
  assert(I != E && "no parameters to constructor");

  // this
  DelegateArgs.push_back(std::make_pair(RValue::get(LoadCXXThis()),
                                        I->second));
  ++I;

  // vtt
  if (llvm::Value *VTT = GetVTTParameter(*this, GlobalDecl(Ctor, CtorType))) {
    QualType VoidPP = getContext().getPointerType(getContext().VoidPtrTy);
    DelegateArgs.push_back(std::make_pair(RValue::get(VTT), VoidPP));

    if (CodeGenVTables::needsVTTParameter(CurGD)) {
      assert(I != E && "cannot skip vtt parameter, already done with args");
      assert(I->second == VoidPP && "skipping parameter not of vtt type");
      ++I;
    }
  }

  // Explicit arguments.
  for (; I != E; ++I) {
    
    const VarDecl *Param = I->first;
    QualType ArgType = Param->getType(); // because we're passing it to itself

    // StartFunction converted the ABI-lowered parameter(s) into a
    // local alloca.  We need to turn that into an r-value suitable
    // for EmitCall.
    llvm::Value *Local = GetAddrOfLocalVar(Param);
    RValue Arg;
 
    // For the most part, we just need to load the alloca, except:
    // 1) aggregate r-values are actually pointers to temporaries, and
    // 2) references to aggregates are pointers directly to the aggregate.
    // I don't know why references to non-aggregates are different here.
    if (ArgType->isReferenceType()) {
      const ReferenceType *RefType = ArgType->getAs<ReferenceType>();
      if (hasAggregateLLVMType(RefType->getPointeeType()))
        Arg = RValue::getAggregate(Local);
      else
        // Locals which are references to scalars are represented
        // with allocas holding the pointer.
        Arg = RValue::get(Builder.CreateLoad(Local));
    } else {
      if (hasAggregateLLVMType(ArgType))
        Arg = RValue::getAggregate(Local);
      else
        Arg = RValue::get(EmitLoadOfScalar(Local, false, ArgType));
    }

    DelegateArgs.push_back(std::make_pair(Arg, ArgType));
  }

  EmitCall(CGM.getTypes().getFunctionInfo(Ctor, CtorType),
           CGM.GetAddrOfCXXConstructor(Ctor, CtorType), 
           ReturnValueSlot(), DelegateArgs, Ctor);
}

void CodeGenFunction::EmitCXXDestructorCall(const CXXDestructorDecl *DD,
                                            CXXDtorType Type,
                                            llvm::Value *This) {
  llvm::Value *VTT = GetVTTParameter(*this, GlobalDecl(DD, Type));
  llvm::Value *Callee = CGM.GetAddrOfCXXDestructor(DD, Type);
  
  EmitCXXMemberCall(DD, Callee, ReturnValueSlot(), This, VTT, 0, 0);
}

llvm::Value *
CodeGenFunction::GetVirtualBaseClassOffset(llvm::Value *This,
                                           const CXXRecordDecl *ClassDecl,
                                           const CXXRecordDecl *BaseClassDecl) {
  const llvm::Type *Int8PtrTy = 
    llvm::Type::getInt8Ty(VMContext)->getPointerTo();

  llvm::Value *VTablePtr = Builder.CreateBitCast(This, 
                                                 Int8PtrTy->getPointerTo());
  VTablePtr = Builder.CreateLoad(VTablePtr, "vtable");

  int64_t VBaseOffsetOffset = 
    CGM.getVTables().getVirtualBaseOffsetOffset(ClassDecl, BaseClassDecl);
  
  llvm::Value *VBaseOffsetPtr = 
    Builder.CreateConstGEP1_64(VTablePtr, VBaseOffsetOffset, "vbase.offset.ptr");
  const llvm::Type *PtrDiffTy = 
    ConvertType(getContext().getPointerDiffType());
  
  VBaseOffsetPtr = Builder.CreateBitCast(VBaseOffsetPtr, 
                                         PtrDiffTy->getPointerTo());
                                         
  llvm::Value *VBaseOffset = Builder.CreateLoad(VBaseOffsetPtr, "vbase.offset");
  
  return VBaseOffset;
}

void
CodeGenFunction::InitializeVTablePointer(BaseSubobject Base, 
                                         bool BaseIsMorallyVirtual,
                                         llvm::Constant *VTable,
                                         const CXXRecordDecl *VTableClass) {
  const CXXRecordDecl *RD = Base.getBase();

  // Compute the address point.
  llvm::Value *VTableAddressPoint;

  // Check if we need to use a vtable from the VTT.
  if (CodeGenVTables::needsVTTParameter(CurGD) &&
      (RD->getNumVBases() || BaseIsMorallyVirtual)) {
    // Get the secondary vpointer index.
    uint64_t VirtualPointerIndex = 
     CGM.getVTables().getSecondaryVirtualPointerIndex(VTableClass, Base);
    
    /// Load the VTT.
    llvm::Value *VTT = LoadCXXVTT();
    if (VirtualPointerIndex)
      VTT = Builder.CreateConstInBoundsGEP1_64(VTT, VirtualPointerIndex);

    // And load the address point from the VTT.
    VTableAddressPoint = Builder.CreateLoad(VTT);
  } else {
    uint64_t AddressPoint = CGM.getVTables().getAddressPoint(Base, VTableClass);
    VTableAddressPoint =
      Builder.CreateConstInBoundsGEP2_64(VTable, 0, AddressPoint);
  }

  // Compute where to store the address point.
  llvm::Value *VTableField;
  
  if (CodeGenVTables::needsVTTParameter(CurGD) && BaseIsMorallyVirtual) {
    // We need to use the virtual base offset offset because the virtual base
    // might have a different offset in the most derived class.
    VTableField = GetAddressOfBaseClass(LoadCXXThis(), VTableClass, RD, 
                                        /*NullCheckValue=*/false);
  } else {
    const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(CGM.getLLVMContext());

    VTableField = Builder.CreateBitCast(LoadCXXThis(), Int8PtrTy);
    VTableField = 
      Builder.CreateConstInBoundsGEP1_64(VTableField, Base.getBaseOffset() / 8);  
  }

  // Finally, store the address point.
  const llvm::Type *AddressPointPtrTy =
    VTableAddressPoint->getType()->getPointerTo();
  VTableField = Builder.CreateBitCast(VTableField, AddressPointPtrTy);
  Builder.CreateStore(VTableAddressPoint, VTableField);
}

void
CodeGenFunction::InitializeVTablePointers(BaseSubobject Base, 
                                          bool BaseIsMorallyVirtual,
                                          bool BaseIsNonVirtualPrimaryBase,
                                          llvm::Constant *VTable,
                                          const CXXRecordDecl *VTableClass,
                                          VisitedVirtualBasesSetTy& VBases) {
  // If this base is a non-virtual primary base the address point has already
  // been set.
  if (!BaseIsNonVirtualPrimaryBase) {
    // Initialize the vtable pointer for this base.
    InitializeVTablePointer(Base, BaseIsMorallyVirtual, VTable, VTableClass);
  }
  
  const CXXRecordDecl *RD = Base.getBase();

  // Traverse bases.
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(), 
       E = RD->bases_end(); I != E; ++I) {
    CXXRecordDecl *BaseDecl
      = cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    // Ignore classes without a vtable.
    if (!BaseDecl->isDynamicClass())
      continue;

    uint64_t BaseOffset;
    bool BaseDeclIsMorallyVirtual = BaseIsMorallyVirtual;
    bool BaseDeclIsNonVirtualPrimaryBase;

    if (I->isVirtual()) {
      // Check if we've visited this virtual base before.
      if (!VBases.insert(BaseDecl))
        continue;

      const ASTRecordLayout &Layout = 
        getContext().getASTRecordLayout(VTableClass);

      BaseOffset = Layout.getVBaseClassOffset(BaseDecl);
      BaseDeclIsMorallyVirtual = true;
      BaseDeclIsNonVirtualPrimaryBase = false;
    } else {
      const ASTRecordLayout &Layout = getContext().getASTRecordLayout(RD);

      BaseOffset = Base.getBaseOffset() + Layout.getBaseClassOffset(BaseDecl);
      BaseDeclIsNonVirtualPrimaryBase = Layout.getPrimaryBase() == BaseDecl;
    }
    
    InitializeVTablePointers(BaseSubobject(BaseDecl, BaseOffset), 
                             BaseDeclIsMorallyVirtual, 
                             BaseDeclIsNonVirtualPrimaryBase, 
                             VTable, VTableClass, VBases);
  }
}

void CodeGenFunction::InitializeVTablePointers(const CXXRecordDecl *RD) {
  // Ignore classes without a vtable.
  if (!RD->isDynamicClass())
    return;

  // Get the VTable.
  llvm::Constant *VTable = CGM.getVTables().GetAddrOfVTable(RD);

  // Initialize the vtable pointers for this class and all of its bases.
  VisitedVirtualBasesSetTy VBases;
  InitializeVTablePointers(BaseSubobject(RD, 0),
                           /*BaseIsMorallyVirtual=*/false, 
                           /*BaseIsNonVirtualPrimaryBase=*/false, 
                           VTable, RD, VBases);
}
