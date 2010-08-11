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

#include "CGDebugInfo.h"
#include "CodeGenFunction.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/StmtCXX.h"

using namespace clang;
using namespace CodeGen;

static uint64_t 
ComputeNonVirtualBaseClassOffset(ASTContext &Context, 
                                 const CXXRecordDecl *DerivedClass,
                                 CastExpr::path_const_iterator Start,
                                 CastExpr::path_const_iterator End) {
  uint64_t Offset = 0;
  
  const CXXRecordDecl *RD = DerivedClass;
  
  for (CastExpr::path_const_iterator I = Start; I != End; ++I) {
    const CXXBaseSpecifier *Base = *I;
    assert(!Base->isVirtual() && "Should not see virtual bases here!");

    // Get the layout.
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
    
    const CXXRecordDecl *BaseDecl = 
      cast<CXXRecordDecl>(Base->getType()->getAs<RecordType>()->getDecl());
    
    // Add the offset.
    Offset += Layout.getBaseClassOffset(BaseDecl);
    
    RD = BaseDecl;
  }
  
  // FIXME: We should not use / 8 here.
  return Offset / 8;
}

llvm::Constant *
CodeGenModule::GetNonVirtualBaseClassOffset(const CXXRecordDecl *ClassDecl,
                                   CastExpr::path_const_iterator PathBegin,
                                   CastExpr::path_const_iterator PathEnd) {
  assert(PathBegin != PathEnd && "Base path should not be empty!");

  uint64_t Offset = 
    ComputeNonVirtualBaseClassOffset(getContext(), ClassDecl,
                                     PathBegin, PathEnd);
  if (!Offset)
    return 0;
  
  const llvm::Type *PtrDiffTy = 
  Types.ConvertType(getContext().getPointerDiffType());
  
  return llvm::ConstantInt::get(PtrDiffTy, Offset);
}

/// Gets the address of a direct base class within a complete object.
/// This should only be used for (1) non-virtual bases or (2) virtual bases
/// when the type is known to be complete (e.g. in complete destructors).
///
/// The object pointed to by 'This' is assumed to be non-null.
llvm::Value *
CodeGenFunction::GetAddressOfDirectBaseInCompleteClass(llvm::Value *This,
                                                   const CXXRecordDecl *Derived,
                                                   const CXXRecordDecl *Base,
                                                   bool BaseIsVirtual) {
  // 'this' must be a pointer (in some address space) to Derived.
  assert(This->getType()->isPointerTy() &&
         cast<llvm::PointerType>(This->getType())->getElementType()
           == ConvertType(Derived));

  // Compute the offset of the virtual base.
  uint64_t Offset;
  const ASTRecordLayout &Layout = getContext().getASTRecordLayout(Derived);
  if (BaseIsVirtual)
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

static llvm::Value *
ApplyNonVirtualAndVirtualOffset(CodeGenFunction &CGF, llvm::Value *ThisPtr,
                                uint64_t NonVirtual, llvm::Value *Virtual) {
  const llvm::Type *PtrDiffTy = 
    CGF.ConvertType(CGF.getContext().getPointerDiffType());
  
  llvm::Value *NonVirtualOffset = 0;
  if (NonVirtual)
    NonVirtualOffset = llvm::ConstantInt::get(PtrDiffTy, NonVirtual);
  
  llvm::Value *BaseOffset;
  if (Virtual) {
    if (NonVirtualOffset)
      BaseOffset = CGF.Builder.CreateAdd(Virtual, NonVirtualOffset);
    else
      BaseOffset = Virtual;
  } else
    BaseOffset = NonVirtualOffset;
  
  // Apply the base offset.
  const llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(CGF.getLLVMContext());
  ThisPtr = CGF.Builder.CreateBitCast(ThisPtr, Int8PtrTy);
  ThisPtr = CGF.Builder.CreateGEP(ThisPtr, BaseOffset, "add.ptr");

  return ThisPtr;
}

llvm::Value *
CodeGenFunction::GetAddressOfBaseClass(llvm::Value *Value, 
                                       const CXXRecordDecl *Derived,
                                       CastExpr::path_const_iterator PathBegin,
                                       CastExpr::path_const_iterator PathEnd,
                                       bool NullCheckValue) {
  assert(PathBegin != PathEnd && "Base path should not be empty!");

  CastExpr::path_const_iterator Start = PathBegin;
  const CXXRecordDecl *VBase = 0;
  
  // Get the virtual base.
  if ((*Start)->isVirtual()) {
    VBase = 
      cast<CXXRecordDecl>((*Start)->getType()->getAs<RecordType>()->getDecl());
    ++Start;
  }
  
  uint64_t NonVirtualOffset = 
    ComputeNonVirtualBaseClassOffset(getContext(), VBase ? VBase : Derived,
                                     Start, PathEnd);

  // Get the base pointer type.
  const llvm::Type *BasePtrTy = 
    ConvertType((PathEnd[-1])->getType())->getPointerTo();
  
  if (!NonVirtualOffset && !VBase) {
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

  llvm::Value *VirtualOffset = 0;

  if (VBase)
    VirtualOffset = GetVirtualBaseClassOffset(Value, Derived, VBase);

  // Apply the offsets.
  Value = ApplyNonVirtualAndVirtualOffset(*this, Value, NonVirtualOffset, 
                                          VirtualOffset);
  
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
                                          const CXXRecordDecl *Derived,
                                        CastExpr::path_const_iterator PathBegin,
                                          CastExpr::path_const_iterator PathEnd,
                                          bool NullCheckValue) {
  assert(PathBegin != PathEnd && "Base path should not be empty!");

  QualType DerivedTy =
    getContext().getCanonicalType(getContext().getTagDeclType(Derived));
  const llvm::Type *DerivedPtrTy = ConvertType(DerivedTy)->getPointerTo();
  
  llvm::Value *NonVirtualOffset =
    CGM.GetNonVirtualBaseClassOffset(Derived, PathBegin, PathEnd);
  
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
                             
/// GetVTTParameter - Return the VTT parameter that should be passed to a
/// base constructor/destructor with virtual bases.
static llvm::Value *GetVTTParameter(CodeGenFunction &CGF, GlobalDecl GD,
                                    bool ForVirtualBase) {
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
    assert(!ForVirtualBase && "Can't have same class as virtual base!");
    SubVTTIndex = 0;
  } else {
    const ASTRecordLayout &Layout = 
      CGF.getContext().getASTRecordLayout(RD);
    uint64_t BaseOffset = ForVirtualBase ? 
      Layout.getVBaseClassOffset(Base) : Layout.getBaseClassOffset(Base);

    SubVTTIndex = 
      CGF.CGM.getVTables().getSubVTTIndex(RD, BaseSubobject(Base, BaseOffset));
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

namespace {
  /// Call the destructor for a direct base class.
  struct CallBaseDtor : EHScopeStack::Cleanup {
    const CXXRecordDecl *BaseClass;
    bool BaseIsVirtual;
    CallBaseDtor(const CXXRecordDecl *Base, bool BaseIsVirtual)
      : BaseClass(Base), BaseIsVirtual(BaseIsVirtual) {}

    void Emit(CodeGenFunction &CGF, bool IsForEH) {
      const CXXRecordDecl *DerivedClass =
        cast<CXXMethodDecl>(CGF.CurCodeDecl)->getParent();

      const CXXDestructorDecl *D = BaseClass->getDestructor();
      llvm::Value *Addr = 
        CGF.GetAddressOfDirectBaseInCompleteClass(CGF.LoadCXXThis(),
                                                  DerivedClass, BaseClass,
                                                  BaseIsVirtual);
      CGF.EmitCXXDestructorCall(D, Dtor_Base, BaseIsVirtual, Addr);
    }
  };
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

  bool isBaseVirtual = BaseInit->isBaseVirtual();

  // The base constructor doesn't construct virtual bases.
  if (CtorType == Ctor_Base && isBaseVirtual)
    return;

  // We can pretend to be a complete class because it only matters for
  // virtual bases, and we only do virtual bases for complete ctors.
  llvm::Value *V = 
    CGF.GetAddressOfDirectBaseInCompleteClass(ThisPtr, ClassDecl,
                                              BaseClassDecl,
                                              isBaseVirtual);

  CGF.EmitAggExpr(BaseInit->getInit(), V, false, false, true);
  
  if (CGF.Exceptions && !BaseClassDecl->hasTrivialDestructor())
    CGF.EHStack.pushCleanup<CallBaseDtor>(EHCleanup, BaseClassDecl,
                                          isBaseVirtual);
}

static void EmitAggMemberInitializer(CodeGenFunction &CGF,
                                     LValue LHS,
                                     llvm::Value *ArrayIndexVar,
                                     CXXBaseOrMemberInitializer *MemberInit,
                                     QualType T,
                                     unsigned Index) {
  if (Index == MemberInit->getNumArrayIndices()) {
    CodeGenFunction::RunCleanupsScope Cleanups(CGF);
    
    llvm::Value *Dest = LHS.getAddress();
    if (ArrayIndexVar) {
      // If we have an array index variable, load it and use it as an offset.
      // Then, increment the value.
      llvm::Value *ArrayIndex = CGF.Builder.CreateLoad(ArrayIndexVar);
      Dest = CGF.Builder.CreateInBoundsGEP(Dest, ArrayIndex, "destaddress");
      llvm::Value *Next = llvm::ConstantInt::get(ArrayIndex->getType(), 1);
      Next = CGF.Builder.CreateAdd(ArrayIndex, Next, "inc");
      CGF.Builder.CreateStore(Next, ArrayIndexVar);      
    }
    
    CGF.EmitAggExpr(MemberInit->getInit(), Dest, 
                    LHS.isVolatileQualified(),
                    /*IgnoreResult*/ false,
                    /*IsInitializer*/ true);
    
    return;
  }
  
  const ConstantArrayType *Array = CGF.getContext().getAsConstantArrayType(T);
  assert(Array && "Array initialization without the array type?");
  llvm::Value *IndexVar
    = CGF.GetAddrOfLocalVar(MemberInit->getArrayIndex(Index));
  assert(IndexVar && "Array index variable not loaded");
  
  // Initialize this index variable to zero.
  llvm::Value* Zero
    = llvm::Constant::getNullValue(
                              CGF.ConvertType(CGF.getContext().getSizeType()));
  CGF.Builder.CreateStore(Zero, IndexVar);
                                   
  // Start the loop with a block that tests the condition.
  llvm::BasicBlock *CondBlock = CGF.createBasicBlock("for.cond");
  llvm::BasicBlock *AfterFor = CGF.createBasicBlock("for.end");
  
  CGF.EmitBlock(CondBlock);

  llvm::BasicBlock *ForBody = CGF.createBasicBlock("for.body");
  // Generate: if (loop-index < number-of-elements) fall to the loop body,
  // otherwise, go to the block after the for-loop.
  uint64_t NumElements = Array->getSize().getZExtValue();
  llvm::Value *Counter = CGF.Builder.CreateLoad(IndexVar);
  llvm::Value *NumElementsPtr =
    llvm::ConstantInt::get(Counter->getType(), NumElements);
  llvm::Value *IsLess = CGF.Builder.CreateICmpULT(Counter, NumElementsPtr,
                                                  "isless");
                                   
  // If the condition is true, execute the body.
  CGF.Builder.CreateCondBr(IsLess, ForBody, AfterFor);

  CGF.EmitBlock(ForBody);
  llvm::BasicBlock *ContinueBlock = CGF.createBasicBlock("for.inc");
  
  {
    CodeGenFunction::RunCleanupsScope Cleanups(CGF);
    
    // Inside the loop body recurse to emit the inner loop or, eventually, the
    // constructor call.
    EmitAggMemberInitializer(CGF, LHS, ArrayIndexVar, MemberInit, 
                             Array->getElementType(), Index + 1);
  }
  
  CGF.EmitBlock(ContinueBlock);

  // Emit the increment of the loop counter.
  llvm::Value *NextVal = llvm::ConstantInt::get(Counter->getType(), 1);
  Counter = CGF.Builder.CreateLoad(IndexVar);
  NextVal = CGF.Builder.CreateAdd(Counter, NextVal, "inc");
  CGF.Builder.CreateStore(NextVal, IndexVar);

  // Finally, branch back up to the condition for the next iteration.
  CGF.EmitBranch(CondBlock);

  // Emit the fall-through block.
  CGF.EmitBlock(AfterFor, true);
}

namespace {
  struct CallMemberDtor : EHScopeStack::Cleanup {
    FieldDecl *Field;
    CXXDestructorDecl *Dtor;

    CallMemberDtor(FieldDecl *Field, CXXDestructorDecl *Dtor)
      : Field(Field), Dtor(Dtor) {}

    void Emit(CodeGenFunction &CGF, bool IsForEH) {
      // FIXME: Is this OK for C++0x delegating constructors?
      llvm::Value *ThisPtr = CGF.LoadCXXThis();
      LValue LHS = CGF.EmitLValueForField(ThisPtr, Field, 0);

      CGF.EmitCXXDestructorCall(Dtor, Dtor_Complete, /*ForVirtualBase=*/false,
                                LHS.getAddress());
    }
  };
}
  
static void EmitMemberInitializer(CodeGenFunction &CGF,
                                  const CXXRecordDecl *ClassDecl,
                                  CXXBaseOrMemberInitializer *MemberInit,
                                  const CXXConstructorDecl *Constructor,
                                  FunctionArgList &Args) {
  assert(MemberInit->isMemberInitializer() &&
         "Must have member initializer!");
  
  // non-static data member initializers.
  FieldDecl *Field = MemberInit->getMember();
  QualType FieldType = CGF.getContext().getCanonicalType(Field->getType());

  llvm::Value *ThisPtr = CGF.LoadCXXThis();
  LValue LHS;
  
  // If we are initializing an anonymous union field, drill down to the field.
  if (MemberInit->getAnonUnionMember()) {
    Field = MemberInit->getAnonUnionMember();
    LHS = CGF.EmitLValueForAnonRecordField(ThisPtr, Field, 0);
    FieldType = Field->getType();
  } else {
    LHS = CGF.EmitLValueForFieldInitialization(ThisPtr, Field, 0);
  }

  // FIXME: If there's no initializer and the CXXBaseOrMemberInitializer
  // was implicitly generated, we shouldn't be zeroing memory.
  RValue RHS;
  if (FieldType->isReferenceType()) {
    RHS = CGF.EmitReferenceBindingToExpr(MemberInit->getInit(), Field);
    CGF.EmitStoreThroughLValue(RHS, LHS, FieldType);
  } else if (FieldType->isArrayType() && !MemberInit->getInit()) {
    CGF.EmitNullInitialization(LHS.getAddress(), Field->getType());
  } else if (!CGF.hasAggregateLLVMType(Field->getType())) {
    RHS = RValue::get(CGF.EmitScalarExpr(MemberInit->getInit()));
    CGF.EmitStoreThroughLValue(RHS, LHS, FieldType);
  } else if (MemberInit->getInit()->getType()->isAnyComplexType()) {
    CGF.EmitComplexExprIntoAddr(MemberInit->getInit(), LHS.getAddress(),
                                LHS.isVolatileQualified());
  } else {
    llvm::Value *ArrayIndexVar = 0;
    const ConstantArrayType *Array
      = CGF.getContext().getAsConstantArrayType(FieldType);
    if (Array && Constructor->isImplicit() && 
        Constructor->isCopyConstructor()) {
      const llvm::Type *SizeTy
        = CGF.ConvertType(CGF.getContext().getSizeType());
      
      // The LHS is a pointer to the first object we'll be constructing, as
      // a flat array.
      QualType BaseElementTy = CGF.getContext().getBaseElementType(Array);
      const llvm::Type *BasePtr = CGF.ConvertType(BaseElementTy);
      BasePtr = llvm::PointerType::getUnqual(BasePtr);
      llvm::Value *BaseAddrPtr = CGF.Builder.CreateBitCast(LHS.getAddress(), 
                                                           BasePtr);
      LHS = LValue::MakeAddr(BaseAddrPtr, CGF.MakeQualifiers(BaseElementTy));
      
      // Create an array index that will be used to walk over all of the
      // objects we're constructing.
      ArrayIndexVar = CGF.CreateTempAlloca(SizeTy, "object.index");
      llvm::Value *Zero = llvm::Constant::getNullValue(SizeTy);
      CGF.Builder.CreateStore(Zero, ArrayIndexVar);
      
      // If we are copying an array of scalars or classes with trivial copy 
      // constructors, perform a single aggregate copy.
      const RecordType *Record = BaseElementTy->getAs<RecordType>();
      if (!Record || 
          cast<CXXRecordDecl>(Record->getDecl())->hasTrivialCopyConstructor()) {
        // Find the source pointer. We knows it's the last argument because
        // we know we're in a copy constructor.
        unsigned SrcArgIndex = Args.size() - 1;
        llvm::Value *SrcPtr
          = CGF.Builder.CreateLoad(
                               CGF.GetAddrOfLocalVar(Args[SrcArgIndex].first));
        LValue Src = CGF.EmitLValueForFieldInitialization(SrcPtr, Field, 0);
        
        // Copy the aggregate.
        CGF.EmitAggregateCopy(LHS.getAddress(), Src.getAddress(), FieldType,
                              LHS.isVolatileQualified());
        return;
      }
      
      // Emit the block variables for the array indices, if any.
      for (unsigned I = 0, N = MemberInit->getNumArrayIndices(); I != N; ++I)
        CGF.EmitLocalBlockVarDecl(*MemberInit->getArrayIndex(I));
    }
    
    EmitAggMemberInitializer(CGF, LHS, ArrayIndexVar, MemberInit, FieldType, 0);
    
    if (!CGF.Exceptions)
      return;

    // FIXME: If we have an array of classes w/ non-trivial destructors, 
    // we need to destroy in reverse order of construction along the exception
    // path.
    const RecordType *RT = FieldType->getAs<RecordType>();
    if (!RT)
      return;
    
    CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
    if (!RD->hasTrivialDestructor())
      CGF.EHStack.pushCleanup<CallMemberDtor>(EHCleanup, Field,
                                              RD->getDestructor());
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
    if (CGDebugInfo *DI = getDebugInfo()) 
      DI->EmitStopPoint(Builder);
    EmitDelegateCXXConstructorCall(Ctor, Ctor_Base, Args);
    return;
  }

  Stmt *Body = Ctor->getBody();

  // Enter the function-try-block before the constructor prologue if
  // applicable.
  bool IsTryBody = (Body && isa<CXXTryStmt>(Body));
  if (IsTryBody)
    EnterCXXTryStmt(*cast<CXXTryStmt>(Body), true);

  EHScopeStack::stable_iterator CleanupDepth = EHStack.stable_begin();

  // Emit the constructor prologue, i.e. the base and member
  // initializers.
  EmitCtorPrologue(Ctor, CtorType, Args);

  // Emit the body of the statement.
  if (IsTryBody)
    EmitStmt(cast<CXXTryStmt>(Body)->getTryBlock());
  else if (Body)
    EmitStmt(Body);

  // Emit any cleanup blocks associated with the member or base
  // initializers, which includes (along the exceptional path) the
  // destructors for those members and bases that were fully
  // constructed.
  PopCleanupBlocks(CleanupDepth);

  if (IsTryBody)
    ExitCXXTryStmt(*cast<CXXTryStmt>(Body), true);
}

/// EmitCtorPrologue - This routine generates necessary code to initialize
/// base classes and non-static data members belonging to this constructor.
void CodeGenFunction::EmitCtorPrologue(const CXXConstructorDecl *CD,
                                       CXXCtorType CtorType,
                                       FunctionArgList &Args) {
  const CXXRecordDecl *ClassDecl = CD->getParent();

  llvm::SmallVector<CXXBaseOrMemberInitializer *, 8> MemberInitializers;
  
  for (CXXConstructorDecl::init_const_iterator B = CD->init_begin(),
       E = CD->init_end();
       B != E; ++B) {
    CXXBaseOrMemberInitializer *Member = (*B);
    
    if (Member->isBaseInitializer())
      EmitBaseInitializer(*this, ClassDecl, Member, CtorType);
    else
      MemberInitializers.push_back(Member);
  }

  InitializeVTablePointers(ClassDecl);

  for (unsigned I = 0, E = MemberInitializers.size(); I != E; ++I)
    EmitMemberInitializer(*this, ClassDecl, MemberInitializers[I], CD, Args);
}

/// EmitDestructorBody - Emits the body of the current destructor.
void CodeGenFunction::EmitDestructorBody(FunctionArgList &Args) {
  const CXXDestructorDecl *Dtor = cast<CXXDestructorDecl>(CurGD.getDecl());
  CXXDtorType DtorType = CurGD.getDtorType();

  // The call to operator delete in a deleting destructor happens
  // outside of the function-try-block, which means it's always
  // possible to delegate the destructor body to the complete
  // destructor.  Do so.
  if (DtorType == Dtor_Deleting) {
    EnterDtorCleanups(Dtor, Dtor_Deleting);
    EmitCXXDestructorCall(Dtor, Dtor_Complete, /*ForVirtualBase=*/false,
                          LoadCXXThis());
    PopCleanupBlock();
    return;
  }

  Stmt *Body = Dtor->getBody();

  // If the body is a function-try-block, enter the try before
  // anything else.
  bool isTryBody = (Body && isa<CXXTryStmt>(Body));
  if (isTryBody)
    EnterCXXTryStmt(*cast<CXXTryStmt>(Body), true);

  // Enter the epilogue cleanups.
  RunCleanupsScope DtorEpilogue(*this);
  
  // If this is the complete variant, just invoke the base variant;
  // the epilogue will destruct the virtual bases.  But we can't do
  // this optimization if the body is a function-try-block, because
  // we'd introduce *two* handler blocks.
  switch (DtorType) {
  case Dtor_Deleting: llvm_unreachable("already handled deleting case");

  case Dtor_Complete:
    // Enter the cleanup scopes for virtual bases.
    EnterDtorCleanups(Dtor, Dtor_Complete);

    if (!isTryBody) {
      EmitCXXDestructorCall(Dtor, Dtor_Base, /*ForVirtualBase=*/false,
                            LoadCXXThis());
      break;
    }
    // Fallthrough: act like we're in the base variant.
      
  case Dtor_Base:
    // Enter the cleanup scopes for fields and non-virtual bases.
    EnterDtorCleanups(Dtor, Dtor_Base);

    // Initialize the vtable pointers before entering the body.
    InitializeVTablePointers(Dtor->getParent());

    if (isTryBody)
      EmitStmt(cast<CXXTryStmt>(Body)->getTryBlock());
    else if (Body)
      EmitStmt(Body);
    else {
      assert(Dtor->isImplicit() && "bodyless dtor not implicit");
      // nothing to do besides what's in the epilogue
    }
    break;
  }

  // Jump out through the epilogue cleanups.
  DtorEpilogue.ForceCleanup();

  // Exit the try if applicable.
  if (isTryBody)
    ExitCXXTryStmt(*cast<CXXTryStmt>(Body), true);
}

namespace {
  /// Call the operator delete associated with the current destructor.
  struct CallDtorDelete : EHScopeStack::Cleanup {
    CallDtorDelete() {}

    void Emit(CodeGenFunction &CGF, bool IsForEH) {
      const CXXDestructorDecl *Dtor = cast<CXXDestructorDecl>(CGF.CurCodeDecl);
      const CXXRecordDecl *ClassDecl = Dtor->getParent();
      CGF.EmitDeleteCall(Dtor->getOperatorDelete(), CGF.LoadCXXThis(),
                         CGF.getContext().getTagDeclType(ClassDecl));
    }
  };

  struct CallArrayFieldDtor : EHScopeStack::Cleanup {
    const FieldDecl *Field;
    CallArrayFieldDtor(const FieldDecl *Field) : Field(Field) {}

    void Emit(CodeGenFunction &CGF, bool IsForEH) {
      QualType FieldType = Field->getType();
      const ConstantArrayType *Array =
        CGF.getContext().getAsConstantArrayType(FieldType);
      
      QualType BaseType =
        CGF.getContext().getBaseElementType(Array->getElementType());
      const CXXRecordDecl *FieldClassDecl = BaseType->getAsCXXRecordDecl();

      llvm::Value *ThisPtr = CGF.LoadCXXThis();
      LValue LHS = CGF.EmitLValueForField(ThisPtr, Field, 
                                          // FIXME: Qualifiers?
                                          /*CVRQualifiers=*/0);

      const llvm::Type *BasePtr = CGF.ConvertType(BaseType)->getPointerTo();
      llvm::Value *BaseAddrPtr =
        CGF.Builder.CreateBitCast(LHS.getAddress(), BasePtr);
      CGF.EmitCXXAggrDestructorCall(FieldClassDecl->getDestructor(),
                                    Array, BaseAddrPtr);
    }
  };

  struct CallFieldDtor : EHScopeStack::Cleanup {
    const FieldDecl *Field;
    CallFieldDtor(const FieldDecl *Field) : Field(Field) {}

    void Emit(CodeGenFunction &CGF, bool IsForEH) {
      const CXXRecordDecl *FieldClassDecl =
        Field->getType()->getAsCXXRecordDecl();

      llvm::Value *ThisPtr = CGF.LoadCXXThis();
      LValue LHS = CGF.EmitLValueForField(ThisPtr, Field, 
                                          // FIXME: Qualifiers?
                                          /*CVRQualifiers=*/0);

      CGF.EmitCXXDestructorCall(FieldClassDecl->getDestructor(),
                                Dtor_Complete, /*ForVirtualBase=*/false,
                                LHS.getAddress());
    }
  };
}

/// EmitDtorEpilogue - Emit all code that comes at the end of class's
/// destructor. This is to call destructors on members and base classes
/// in reverse order of their construction.
void CodeGenFunction::EnterDtorCleanups(const CXXDestructorDecl *DD,
                                        CXXDtorType DtorType) {
  assert(!DD->isTrivial() &&
         "Should not emit dtor epilogue for trivial dtor!");

  // The deleting-destructor phase just needs to call the appropriate
  // operator delete that Sema picked up.
  if (DtorType == Dtor_Deleting) {
    assert(DD->getOperatorDelete() && 
           "operator delete missing - EmitDtorEpilogue");
    EHStack.pushCleanup<CallDtorDelete>(NormalAndEHCleanup);
    return;
  }

  const CXXRecordDecl *ClassDecl = DD->getParent();

  // The complete-destructor phase just destructs all the virtual bases.
  if (DtorType == Dtor_Complete) {

    // We push them in the forward order so that they'll be popped in
    // the reverse order.
    for (CXXRecordDecl::base_class_const_iterator I = 
           ClassDecl->vbases_begin(), E = ClassDecl->vbases_end();
              I != E; ++I) {
      const CXXBaseSpecifier &Base = *I;
      CXXRecordDecl *BaseClassDecl
        = cast<CXXRecordDecl>(Base.getType()->getAs<RecordType>()->getDecl());
    
      // Ignore trivial destructors.
      if (BaseClassDecl->hasTrivialDestructor())
        continue;

      EHStack.pushCleanup<CallBaseDtor>(NormalAndEHCleanup,
                                        BaseClassDecl,
                                        /*BaseIsVirtual*/ true);
    }

    return;
  }

  assert(DtorType == Dtor_Base);
  
  // Destroy non-virtual bases.
  for (CXXRecordDecl::base_class_const_iterator I = 
        ClassDecl->bases_begin(), E = ClassDecl->bases_end(); I != E; ++I) {
    const CXXBaseSpecifier &Base = *I;
    
    // Ignore virtual bases.
    if (Base.isVirtual())
      continue;
    
    CXXRecordDecl *BaseClassDecl = Base.getType()->getAsCXXRecordDecl();
    
    // Ignore trivial destructors.
    if (BaseClassDecl->hasTrivialDestructor())
      continue;

    EHStack.pushCleanup<CallBaseDtor>(NormalAndEHCleanup,
                                      BaseClassDecl,
                                      /*BaseIsVirtual*/ false);
  }

  // Destroy direct fields.
  llvm::SmallVector<const FieldDecl *, 16> FieldDecls;
  for (CXXRecordDecl::field_iterator I = ClassDecl->field_begin(),
       E = ClassDecl->field_end(); I != E; ++I) {
    const FieldDecl *Field = *I;
    
    QualType FieldType = getContext().getCanonicalType(Field->getType());
    const ConstantArrayType *Array = 
      getContext().getAsConstantArrayType(FieldType);
    if (Array)
      FieldType = getContext().getBaseElementType(Array->getElementType());
    
    const RecordType *RT = FieldType->getAs<RecordType>();
    if (!RT)
      continue;
    
    CXXRecordDecl *FieldClassDecl = cast<CXXRecordDecl>(RT->getDecl());
    if (FieldClassDecl->hasTrivialDestructor())
        continue;

    if (Array)
      EHStack.pushCleanup<CallArrayFieldDtor>(NormalAndEHCleanup, Field);
    else
      EHStack.pushCleanup<CallFieldDtor>(NormalAndEHCleanup, Field);
  }
}

/// EmitCXXAggrConstructorCall - This routine essentially creates a (nested)
/// for-loop to call the default constructor on individual members of the
/// array. 
/// 'D' is the default constructor for elements of the array, 'ArrayTy' is the
/// array type and 'ArrayPtr' points to the beginning fo the array.
/// It is assumed that all relevant checks have been made by the caller.
///
/// \param ZeroInitialization True if each element should be zero-initialized
/// before it is constructed.
void
CodeGenFunction::EmitCXXAggrConstructorCall(const CXXConstructorDecl *D,
                                            const ConstantArrayType *ArrayTy,
                                            llvm::Value *ArrayPtr,
                                            CallExpr::const_arg_iterator ArgBeg,
                                            CallExpr::const_arg_iterator ArgEnd,
                                            bool ZeroInitialization) {

  const llvm::Type *SizeTy = ConvertType(getContext().getSizeType());
  llvm::Value * NumElements =
    llvm::ConstantInt::get(SizeTy, 
                           getContext().getConstantArrayElementCount(ArrayTy));

  EmitCXXAggrConstructorCall(D, NumElements, ArrayPtr, ArgBeg, ArgEnd,
                             ZeroInitialization);
}

void
CodeGenFunction::EmitCXXAggrConstructorCall(const CXXConstructorDecl *D,
                                          llvm::Value *NumElements,
                                          llvm::Value *ArrayPtr,
                                          CallExpr::const_arg_iterator ArgBeg,
                                          CallExpr::const_arg_iterator ArgEnd,
                                            bool ZeroInitialization) {
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

  // Zero initialize the storage, if requested.
  if (ZeroInitialization)
    EmitNullInitialization(Address, 
                           getContext().getTypeDeclType(D->getParent()));
  
  // C++ [class.temporary]p4: 
  // There are two contexts in which temporaries are destroyed at a different
  // point than the end of the full-expression. The first context is when a
  // default constructor is called to initialize an element of an array. 
  // If the constructor has one or more default arguments, the destruction of 
  // every temporary created in a default argument expression is sequenced 
  // before the construction of the next array element, if any.
  
  // Keep track of the current number of live temporaries.
  {
    RunCleanupsScope Scope(*this);

    EmitCXXConstructorCall(D, Ctor_Complete, /*ForVirtualBase=*/false, Address,
                           ArgBeg, ArgEnd);
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
  EmitCXXDestructorCall(D, Dtor_Complete, /*ForVirtualBase=*/false, Address);

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

void
CodeGenFunction::EmitCXXConstructorCall(const CXXConstructorDecl *D,
                                        CXXCtorType Type, bool ForVirtualBase,
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

  llvm::Value *VTT = GetVTTParameter(*this, GlobalDecl(D, Type), ForVirtualBase);
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
  if (llvm::Value *VTT = GetVTTParameter(*this, GlobalDecl(Ctor, CtorType),
                                         /*ForVirtualBase=*/false)) {
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
    RValue Arg = EmitDelegateCallArg(Param);

    DelegateArgs.push_back(std::make_pair(Arg, ArgType));
  }

  EmitCall(CGM.getTypes().getFunctionInfo(Ctor, CtorType),
           CGM.GetAddrOfCXXConstructor(Ctor, CtorType), 
           ReturnValueSlot(), DelegateArgs, Ctor);
}

void CodeGenFunction::EmitCXXDestructorCall(const CXXDestructorDecl *DD,
                                            CXXDtorType Type,
                                            bool ForVirtualBase,
                                            llvm::Value *This) {
  llvm::Value *VTT = GetVTTParameter(*this, GlobalDecl(DD, Type), 
                                     ForVirtualBase);
  llvm::Value *Callee = CGM.GetAddrOfCXXDestructor(DD, Type);
  
  EmitCXXMemberCall(DD, Callee, ReturnValueSlot(), This, VTT, 0, 0);
}

namespace {
  struct CallLocalDtor : EHScopeStack::Cleanup {
    const CXXDestructorDecl *Dtor;
    llvm::Value *Addr;

    CallLocalDtor(const CXXDestructorDecl *D, llvm::Value *Addr)
      : Dtor(D), Addr(Addr) {}

    void Emit(CodeGenFunction &CGF, bool IsForEH) {
      CGF.EmitCXXDestructorCall(Dtor, Dtor_Complete,
                                /*ForVirtualBase=*/false, Addr);
    }
  };
}

void CodeGenFunction::PushDestructorCleanup(const CXXDestructorDecl *D,
                                            llvm::Value *Addr) {
  EHStack.pushCleanup<CallLocalDtor>(NormalAndEHCleanup, D, Addr);
}

void CodeGenFunction::PushDestructorCleanup(QualType T, llvm::Value *Addr) {
  CXXRecordDecl *ClassDecl = T->getAsCXXRecordDecl();
  if (!ClassDecl) return;
  if (ClassDecl->hasTrivialDestructor()) return;

  const CXXDestructorDecl *D = ClassDecl->getDestructor();
  PushDestructorCleanup(D, Addr);
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
                                         const CXXRecordDecl *NearestVBase,
                                         uint64_t OffsetFromNearestVBase,
                                         llvm::Constant *VTable,
                                         const CXXRecordDecl *VTableClass) {
  const CXXRecordDecl *RD = Base.getBase();

  // Compute the address point.
  llvm::Value *VTableAddressPoint;

  // Check if we need to use a vtable from the VTT.
  if (CodeGenVTables::needsVTTParameter(CurGD) &&
      (RD->getNumVBases() || NearestVBase)) {
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
  llvm::Value *VirtualOffset = 0;
  uint64_t NonVirtualOffset = 0;
  
  if (CodeGenVTables::needsVTTParameter(CurGD) && NearestVBase) {
    // We need to use the virtual base offset offset because the virtual base
    // might have a different offset in the most derived class.
    VirtualOffset = GetVirtualBaseClassOffset(LoadCXXThis(), VTableClass, 
                                              NearestVBase);
    NonVirtualOffset = OffsetFromNearestVBase / 8;
  } else {
    // We can just use the base offset in the complete class.
    NonVirtualOffset = Base.getBaseOffset() / 8;
  }
  
  // Apply the offsets.
  llvm::Value *VTableField = LoadCXXThis();
  
  if (NonVirtualOffset || VirtualOffset)
    VTableField = ApplyNonVirtualAndVirtualOffset(*this, VTableField, 
                                                  NonVirtualOffset,
                                                  VirtualOffset);

  // Finally, store the address point.
  const llvm::Type *AddressPointPtrTy =
    VTableAddressPoint->getType()->getPointerTo();
  VTableField = Builder.CreateBitCast(VTableField, AddressPointPtrTy);
  Builder.CreateStore(VTableAddressPoint, VTableField);
}

void
CodeGenFunction::InitializeVTablePointers(BaseSubobject Base, 
                                          const CXXRecordDecl *NearestVBase,
                                          uint64_t OffsetFromNearestVBase,
                                          bool BaseIsNonVirtualPrimaryBase,
                                          llvm::Constant *VTable,
                                          const CXXRecordDecl *VTableClass,
                                          VisitedVirtualBasesSetTy& VBases) {
  // If this base is a non-virtual primary base the address point has already
  // been set.
  if (!BaseIsNonVirtualPrimaryBase) {
    // Initialize the vtable pointer for this base.
    InitializeVTablePointer(Base, NearestVBase, OffsetFromNearestVBase,
                            VTable, VTableClass);
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
    uint64_t BaseOffsetFromNearestVBase;
    bool BaseDeclIsNonVirtualPrimaryBase;

    if (I->isVirtual()) {
      // Check if we've visited this virtual base before.
      if (!VBases.insert(BaseDecl))
        continue;

      const ASTRecordLayout &Layout = 
        getContext().getASTRecordLayout(VTableClass);

      BaseOffset = Layout.getVBaseClassOffset(BaseDecl);
      BaseOffsetFromNearestVBase = 0;
      BaseDeclIsNonVirtualPrimaryBase = false;
    } else {
      const ASTRecordLayout &Layout = getContext().getASTRecordLayout(RD);

      BaseOffset = Base.getBaseOffset() + Layout.getBaseClassOffset(BaseDecl);
      BaseOffsetFromNearestVBase = 
        OffsetFromNearestVBase + Layout.getBaseClassOffset(BaseDecl);
      BaseDeclIsNonVirtualPrimaryBase = Layout.getPrimaryBase() == BaseDecl;
    }
    
    InitializeVTablePointers(BaseSubobject(BaseDecl, BaseOffset), 
                             I->isVirtual() ? BaseDecl : NearestVBase,
                             BaseOffsetFromNearestVBase,
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
  InitializeVTablePointers(BaseSubobject(RD, 0), /*NearestVBase=*/0, 
                           /*OffsetFromNearestVBase=*/0,
                           /*BaseIsNonVirtualPrimaryBase=*/false, 
                           VTable, RD, VBases);
}
