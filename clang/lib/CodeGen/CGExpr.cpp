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
#include "CGCXXABI.h"
#include "CGDebugInfo.h"
#include "CGRecordLayout.h"
#include "CGObjCRuntime.h"
#include "TargetInfo.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/Intrinsics.h"
#include "llvm/LLVMContext.h"
#include "llvm/Target/TargetData.h"
using namespace clang;
using namespace CodeGen;

//===--------------------------------------------------------------------===//
//                        Miscellaneous Helper Methods
//===--------------------------------------------------------------------===//

llvm::Value *CodeGenFunction::EmitCastToVoidPtr(llvm::Value *value) {
  unsigned addressSpace =
    cast<llvm::PointerType>(value->getType())->getAddressSpace();

  llvm::PointerType *destType = Int8PtrTy;
  if (addressSpace)
    destType = llvm::Type::getInt8PtrTy(getLLVMContext(), addressSpace);

  if (value->getType() == destType) return value;
  return Builder.CreateBitCast(value, destType);
}

/// CreateTempAlloca - This creates a alloca and inserts it into the entry
/// block.
llvm::AllocaInst *CodeGenFunction::CreateTempAlloca(llvm::Type *Ty,
                                                    const Twine &Name) {
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

llvm::AllocaInst *CodeGenFunction::CreateIRTemp(QualType Ty,
                                                const Twine &Name) {
  llvm::AllocaInst *Alloc = CreateTempAlloca(ConvertType(Ty), Name);
  // FIXME: Should we prefer the preferred type alignment here?
  CharUnits Align = getContext().getTypeAlignInChars(Ty);
  Alloc->setAlignment(Align.getQuantity());
  return Alloc;
}

llvm::AllocaInst *CodeGenFunction::CreateMemTemp(QualType Ty,
                                                 const Twine &Name) {
  llvm::AllocaInst *Alloc = CreateTempAlloca(ConvertTypeForMem(Ty), Name);
  // FIXME: Should we prefer the preferred type alignment here?
  CharUnits Align = getContext().getTypeAlignInChars(Ty);
  Alloc->setAlignment(Align.getQuantity());
  return Alloc;
}

/// EvaluateExprAsBool - Perform the usual unary conversions on the specified
/// expression and compare the result against zero, returning an Int1Ty value.
llvm::Value *CodeGenFunction::EvaluateExprAsBool(const Expr *E) {
  if (const MemberPointerType *MPT = E->getType()->getAs<MemberPointerType>()) {
    llvm::Value *MemPtr = EmitScalarExpr(E);
    return CGM.getCXXABI().EmitMemberPointerIsNotNull(*this, MemPtr, MPT);
  }

  QualType BoolTy = getContext().BoolTy;
  if (!E->getType()->isAnyComplexType())
    return EmitScalarConversion(EmitScalarExpr(E), E->getType(), BoolTy);

  return EmitComplexToScalarConversion(EmitComplexExpr(E), E->getType(),BoolTy);
}

/// EmitIgnoredExpr - Emit code to compute the specified expression,
/// ignoring the result.
void CodeGenFunction::EmitIgnoredExpr(const Expr *E) {
  if (E->isRValue())
    return (void) EmitAnyExpr(E, AggValueSlot::ignored(), true);

  // Just emit it as an l-value and drop the result.
  EmitLValue(E);
}

/// EmitAnyExpr - Emit code to compute the specified expression which
/// can have any type.  The result is returned as an RValue struct.
/// If this is an aggregate expression, AggSlot indicates where the
/// result should be returned.
RValue CodeGenFunction::EmitAnyExpr(const Expr *E, AggValueSlot AggSlot,
                                    bool IgnoreResult) {
  if (!hasAggregateLLVMType(E->getType()))
    return RValue::get(EmitScalarExpr(E, IgnoreResult));
  else if (E->getType()->isAnyComplexType())
    return RValue::getComplex(EmitComplexExpr(E, IgnoreResult, IgnoreResult));

  EmitAggExpr(E, AggSlot, IgnoreResult);
  return AggSlot.asRValue();
}

/// EmitAnyExprToTemp - Similary to EmitAnyExpr(), however, the result will
/// always be accessible even if no aggregate location is provided.
RValue CodeGenFunction::EmitAnyExprToTemp(const Expr *E) {
  AggValueSlot AggSlot = AggValueSlot::ignored();

  if (hasAggregateLLVMType(E->getType()) &&
      !E->getType()->isAnyComplexType())
    AggSlot = CreateAggTemp(E->getType(), "agg.tmp");
  return EmitAnyExpr(E, AggSlot);
}

/// EmitAnyExprToMem - Evaluate an expression into a given memory
/// location.
void CodeGenFunction::EmitAnyExprToMem(const Expr *E,
                                       llvm::Value *Location,
                                       Qualifiers Quals,
                                       bool IsInit) {
  // FIXME: This function should take an LValue as an argument.
  if (E->getType()->isAnyComplexType()) {
    EmitComplexExprIntoAddr(E, Location, Quals.hasVolatile());
  } else if (hasAggregateLLVMType(E->getType())) {
    CharUnits Alignment = getContext().getTypeAlignInChars(E->getType());
    EmitAggExpr(E, AggValueSlot::forAddr(Location, Alignment, Quals,
                                         AggValueSlot::IsDestructed_t(IsInit),
                                         AggValueSlot::DoesNotNeedGCBarriers,
                                         AggValueSlot::IsAliased_t(!IsInit)));
  } else {
    RValue RV = RValue::get(EmitScalarExpr(E, /*Ignore*/ false));
    LValue LV = MakeAddrLValue(Location, E->getType());
    EmitStoreThroughLValue(RV, LV);
  }
}

namespace {
/// \brief An adjustment to be made to the temporary created when emitting a
/// reference binding, which accesses a particular subobject of that temporary.
  struct SubobjectAdjustment {
    enum { DerivedToBaseAdjustment, FieldAdjustment } Kind;

    union {
      struct {
        const CastExpr *BasePath;
        const CXXRecordDecl *DerivedClass;
      } DerivedToBase;

      FieldDecl *Field;
    };

    SubobjectAdjustment(const CastExpr *BasePath,
                        const CXXRecordDecl *DerivedClass)
      : Kind(DerivedToBaseAdjustment) {
      DerivedToBase.BasePath = BasePath;
      DerivedToBase.DerivedClass = DerivedClass;
    }

    SubobjectAdjustment(FieldDecl *Field)
      : Kind(FieldAdjustment) {
      this->Field = Field;
    }
  };
}

static llvm::Value *
CreateReferenceTemporary(CodeGenFunction &CGF, QualType Type,
                         const NamedDecl *InitializedDecl) {
  if (const VarDecl *VD = dyn_cast_or_null<VarDecl>(InitializedDecl)) {
    if (VD->hasGlobalStorage()) {
      SmallString<256> Name;
      llvm::raw_svector_ostream Out(Name);
      CGF.CGM.getCXXABI().getMangleContext().mangleReferenceTemporary(VD, Out);
      Out.flush();

      llvm::Type *RefTempTy = CGF.ConvertTypeForMem(Type);
  
      // Create the reference temporary.
      llvm::GlobalValue *RefTemp =
        new llvm::GlobalVariable(CGF.CGM.getModule(), 
                                 RefTempTy, /*isConstant=*/false,
                                 llvm::GlobalValue::InternalLinkage,
                                 llvm::Constant::getNullValue(RefTempTy),
                                 Name.str());
      return RefTemp;
    }
  }

  return CGF.CreateMemTemp(Type, "ref.tmp");
}

static llvm::Value *
EmitExprForReferenceBinding(CodeGenFunction &CGF, const Expr *E,
                            llvm::Value *&ReferenceTemporary,
                            const CXXDestructorDecl *&ReferenceTemporaryDtor,
                            QualType &ObjCARCReferenceLifetimeType,
                            const NamedDecl *InitializedDecl) {
  // Look through single-element init lists that claim to be lvalues. They're
  // just syntactic wrappers in this case.
  if (const InitListExpr *ILE = dyn_cast<InitListExpr>(E)) {
    if (ILE->getNumInits() == 1 && ILE->isGLValue())
      E = ILE->getInit(0);
  }

  // Look through expressions for materialized temporaries (for now).
  if (const MaterializeTemporaryExpr *M 
                                      = dyn_cast<MaterializeTemporaryExpr>(E)) {
    // Objective-C++ ARC:
    //   If we are binding a reference to a temporary that has ownership, we 
    //   need to perform retain/release operations on the temporary.
    if (CGF.getContext().getLangOpts().ObjCAutoRefCount &&        
        E->getType()->isObjCLifetimeType() &&
        (E->getType().getObjCLifetime() == Qualifiers::OCL_Strong ||
         E->getType().getObjCLifetime() == Qualifiers::OCL_Weak ||
         E->getType().getObjCLifetime() == Qualifiers::OCL_Autoreleasing))
      ObjCARCReferenceLifetimeType = E->getType();
    
    E = M->GetTemporaryExpr();
  }

  if (const CXXDefaultArgExpr *DAE = dyn_cast<CXXDefaultArgExpr>(E))
    E = DAE->getExpr();
  
  if (const ExprWithCleanups *EWC = dyn_cast<ExprWithCleanups>(E)) {
    CGF.enterFullExpression(EWC);
    CodeGenFunction::RunCleanupsScope Scope(CGF);

    return EmitExprForReferenceBinding(CGF, EWC->getSubExpr(), 
                                       ReferenceTemporary, 
                                       ReferenceTemporaryDtor,
                                       ObjCARCReferenceLifetimeType,
                                       InitializedDecl);
  }

  RValue RV;
  if (E->isGLValue()) {
    // Emit the expression as an lvalue.
    LValue LV = CGF.EmitLValue(E);
    
    if (LV.isSimple())
      return LV.getAddress();
    
    // We have to load the lvalue.
    RV = CGF.EmitLoadOfLValue(LV);
  } else {
    if (!ObjCARCReferenceLifetimeType.isNull()) {
      ReferenceTemporary = CreateReferenceTemporary(CGF, 
                                                  ObjCARCReferenceLifetimeType, 
                                                    InitializedDecl);
      
      
      LValue RefTempDst = CGF.MakeAddrLValue(ReferenceTemporary, 
                                             ObjCARCReferenceLifetimeType);

      CGF.EmitScalarInit(E, dyn_cast_or_null<ValueDecl>(InitializedDecl),
                         RefTempDst, false);
      
      bool ExtendsLifeOfTemporary = false;
      if (const VarDecl *Var = dyn_cast_or_null<VarDecl>(InitializedDecl)) {
        if (Var->extendsLifetimeOfTemporary())
          ExtendsLifeOfTemporary = true;
      } else if (InitializedDecl && isa<FieldDecl>(InitializedDecl)) {
        ExtendsLifeOfTemporary = true;
      }
      
      if (!ExtendsLifeOfTemporary) {
        // Since the lifetime of this temporary isn't going to be extended,
        // we need to clean it up ourselves at the end of the full expression.
        switch (ObjCARCReferenceLifetimeType.getObjCLifetime()) {
        case Qualifiers::OCL_None:
        case Qualifiers::OCL_ExplicitNone:
        case Qualifiers::OCL_Autoreleasing:
          break;
            
        case Qualifiers::OCL_Strong: {
          assert(!ObjCARCReferenceLifetimeType->isArrayType());
          CleanupKind cleanupKind = CGF.getARCCleanupKind();
          CGF.pushDestroy(cleanupKind, 
                          ReferenceTemporary,
                          ObjCARCReferenceLifetimeType,
                          CodeGenFunction::destroyARCStrongImprecise,
                          cleanupKind & EHCleanup);
          break;
        }
          
        case Qualifiers::OCL_Weak:
          assert(!ObjCARCReferenceLifetimeType->isArrayType());
          CGF.pushDestroy(NormalAndEHCleanup, 
                          ReferenceTemporary,
                          ObjCARCReferenceLifetimeType,
                          CodeGenFunction::destroyARCWeak,
                          /*useEHCleanupForArray*/ true);
          break;
        }
        
        ObjCARCReferenceLifetimeType = QualType();
      }
      
      return ReferenceTemporary;
    }
    
    SmallVector<SubobjectAdjustment, 2> Adjustments;
    while (true) {
      E = E->IgnoreParens();

      if (const CastExpr *CE = dyn_cast<CastExpr>(E)) {
        if ((CE->getCastKind() == CK_DerivedToBase ||
             CE->getCastKind() == CK_UncheckedDerivedToBase) &&
            E->getType()->isRecordType()) {
          E = CE->getSubExpr();
          CXXRecordDecl *Derived 
            = cast<CXXRecordDecl>(E->getType()->getAs<RecordType>()->getDecl());
          Adjustments.push_back(SubobjectAdjustment(CE, Derived));
          continue;
        }

        if (CE->getCastKind() == CK_NoOp) {
          E = CE->getSubExpr();
          continue;
        }
      } else if (const MemberExpr *ME = dyn_cast<MemberExpr>(E)) {
        if (!ME->isArrow() && ME->getBase()->isRValue()) {
          assert(ME->getBase()->getType()->isRecordType());
          if (FieldDecl *Field = dyn_cast<FieldDecl>(ME->getMemberDecl())) {
            E = ME->getBase();
            Adjustments.push_back(SubobjectAdjustment(Field));
            continue;
          }
        }
      }

      if (const OpaqueValueExpr *opaque = dyn_cast<OpaqueValueExpr>(E))
        if (opaque->getType()->isRecordType())
          return CGF.EmitOpaqueValueLValue(opaque).getAddress();

      // Nothing changed.
      break;
    }
    
    // Create a reference temporary if necessary.
    AggValueSlot AggSlot = AggValueSlot::ignored();
    if (CGF.hasAggregateLLVMType(E->getType()) &&
        !E->getType()->isAnyComplexType()) {
      ReferenceTemporary = CreateReferenceTemporary(CGF, E->getType(), 
                                                    InitializedDecl);
      CharUnits Alignment = CGF.getContext().getTypeAlignInChars(E->getType());
      AggValueSlot::IsDestructed_t isDestructed
        = AggValueSlot::IsDestructed_t(InitializedDecl != 0);
      AggSlot = AggValueSlot::forAddr(ReferenceTemporary, Alignment,
                                      Qualifiers(), isDestructed,
                                      AggValueSlot::DoesNotNeedGCBarriers,
                                      AggValueSlot::IsNotAliased);
    }
    
    if (InitializedDecl) {
      // Get the destructor for the reference temporary.
      if (const RecordType *RT = E->getType()->getAs<RecordType>()) {
        CXXRecordDecl *ClassDecl = cast<CXXRecordDecl>(RT->getDecl());
        if (!ClassDecl->hasTrivialDestructor())
          ReferenceTemporaryDtor = ClassDecl->getDestructor();
      }
    }

    RV = CGF.EmitAnyExpr(E, AggSlot);

    // Check if need to perform derived-to-base casts and/or field accesses, to
    // get from the temporary object we created (and, potentially, for which we
    // extended the lifetime) to the subobject we're binding the reference to.
    if (!Adjustments.empty()) {
      llvm::Value *Object = RV.getAggregateAddr();
      for (unsigned I = Adjustments.size(); I != 0; --I) {
        SubobjectAdjustment &Adjustment = Adjustments[I-1];
        switch (Adjustment.Kind) {
        case SubobjectAdjustment::DerivedToBaseAdjustment:
          Object = 
              CGF.GetAddressOfBaseClass(Object, 
                                        Adjustment.DerivedToBase.DerivedClass, 
                              Adjustment.DerivedToBase.BasePath->path_begin(),
                              Adjustment.DerivedToBase.BasePath->path_end(),
                                        /*NullCheckValue=*/false);
          break;
            
        case SubobjectAdjustment::FieldAdjustment: {
          LValue LV = 
            CGF.EmitLValueForField(Object, Adjustment.Field, 0);
          if (LV.isSimple()) {
            Object = LV.getAddress();
            break;
          }
          
          // For non-simple lvalues, we actually have to create a copy of
          // the object we're binding to.
          QualType T = Adjustment.Field->getType().getNonReferenceType()
                                                  .getUnqualifiedType();
          Object = CreateReferenceTemporary(CGF, T, InitializedDecl);
          LValue TempLV = CGF.MakeAddrLValue(Object,
                                             Adjustment.Field->getType());
          CGF.EmitStoreThroughLValue(CGF.EmitLoadOfLValue(LV), TempLV);
          break;
        }

        }
      }

      return Object;
    }
  }

  if (RV.isAggregate())
    return RV.getAggregateAddr();

  // Create a temporary variable that we can bind the reference to.
  ReferenceTemporary = CreateReferenceTemporary(CGF, E->getType(), 
                                                InitializedDecl);


  unsigned Alignment =
    CGF.getContext().getTypeAlignInChars(E->getType()).getQuantity();
  if (RV.isScalar())
    CGF.EmitStoreOfScalar(RV.getScalarVal(), ReferenceTemporary,
                          /*Volatile=*/false, Alignment, E->getType());
  else
    CGF.StoreComplexToAddr(RV.getComplexVal(), ReferenceTemporary,
                           /*Volatile=*/false);
  return ReferenceTemporary;
}

RValue
CodeGenFunction::EmitReferenceBindingToExpr(const Expr *E,
                                            const NamedDecl *InitializedDecl) {
  llvm::Value *ReferenceTemporary = 0;
  const CXXDestructorDecl *ReferenceTemporaryDtor = 0;
  QualType ObjCARCReferenceLifetimeType;
  llvm::Value *Value = EmitExprForReferenceBinding(*this, E, ReferenceTemporary,
                                                   ReferenceTemporaryDtor,
                                                   ObjCARCReferenceLifetimeType,
                                                   InitializedDecl);
  if (!ReferenceTemporaryDtor && ObjCARCReferenceLifetimeType.isNull())
    return RValue::get(Value);
  
  // Make sure to call the destructor for the reference temporary.
  const VarDecl *VD = dyn_cast_or_null<VarDecl>(InitializedDecl);
  if (VD && VD->hasGlobalStorage()) {
    if (ReferenceTemporaryDtor) {
      llvm::Constant *DtorFn = 
        CGM.GetAddrOfCXXDestructor(ReferenceTemporaryDtor, Dtor_Complete);
      EmitCXXGlobalDtorRegistration(DtorFn, 
                                    cast<llvm::Constant>(ReferenceTemporary));
    } else {
      assert(!ObjCARCReferenceLifetimeType.isNull());
      // Note: We intentionally do not register a global "destructor" to
      // release the object.
    }
    
    return RValue::get(Value);
  }

  if (ReferenceTemporaryDtor)
    PushDestructorCleanup(ReferenceTemporaryDtor, ReferenceTemporary);
  else {
    switch (ObjCARCReferenceLifetimeType.getObjCLifetime()) {
    case Qualifiers::OCL_None:
      llvm_unreachable(
                      "Not a reference temporary that needs to be deallocated");
    case Qualifiers::OCL_ExplicitNone:
    case Qualifiers::OCL_Autoreleasing:
      // Nothing to do.
      break;        
        
    case Qualifiers::OCL_Strong: {
      bool precise = VD && VD->hasAttr<ObjCPreciseLifetimeAttr>();
      CleanupKind cleanupKind = getARCCleanupKind();
      pushDestroy(cleanupKind, ReferenceTemporary, ObjCARCReferenceLifetimeType,
                  precise ? destroyARCStrongPrecise : destroyARCStrongImprecise,
                  cleanupKind & EHCleanup);
      break;
    }
        
    case Qualifiers::OCL_Weak: {
      // __weak objects always get EH cleanups; otherwise, exceptions
      // could cause really nasty crashes instead of mere leaks.
      pushDestroy(NormalAndEHCleanup, ReferenceTemporary,
                  ObjCARCReferenceLifetimeType, destroyARCWeak, true);
      break;        
    }
    }
  }
  
  return RValue::get(Value);
}


/// getAccessedFieldNo - Given an encoded value and a result number, return the
/// input field number being accessed.
unsigned CodeGenFunction::getAccessedFieldNo(unsigned Idx,
                                             const llvm::Constant *Elts) {
  return cast<llvm::ConstantInt>(Elts->getAggregateElement(Idx))
      ->getZExtValue();
}

void CodeGenFunction::EmitCheck(llvm::Value *Address, unsigned Size) {
  if (!CatchUndefined)
    return;

  // This needs to be to the standard address space.
  Address = Builder.CreateBitCast(Address, Int8PtrTy);

  llvm::Value *F = CGM.getIntrinsic(llvm::Intrinsic::objectsize, IntPtrTy);

  // In time, people may want to control this and use a 1 here.
  llvm::Value *Arg = Builder.getFalse();
  llvm::Value *C = Builder.CreateCall2(F, Address, Arg);
  llvm::BasicBlock *Cont = createBasicBlock();
  llvm::BasicBlock *Check = createBasicBlock();
  llvm::Value *NegativeOne = llvm::ConstantInt::get(IntPtrTy, -1ULL);
  Builder.CreateCondBr(Builder.CreateICmpEQ(C, NegativeOne), Cont, Check);
    
  EmitBlock(Check);
  Builder.CreateCondBr(Builder.CreateICmpUGE(C,
                                        llvm::ConstantInt::get(IntPtrTy, Size)),
                       Cont, getTrapBB());
  EmitBlock(Cont);
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
    llvm::Type *EltTy = ConvertType(CTy->getElementType());
    llvm::Value *U = llvm::UndefValue::get(EltTy);
    return RValue::getComplex(std::make_pair(U, U));
  }
  
  // If this is a use of an undefined aggregate type, the aggregate must have an
  // identifiable address.  Just because the contents of the value are undefined
  // doesn't mean that the address can't be taken and compared.
  if (hasAggregateLLVMType(Ty)) {
    llvm::Value *DestPtr = CreateMemTemp(Ty, "undef.agg.tmp");
    return RValue::getAggregate(DestPtr);
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
  return MakeAddrLValue(llvm::UndefValue::get(Ty), E->getType());
}

LValue CodeGenFunction::EmitCheckedLValue(const Expr *E) {
  LValue LV = EmitLValue(E);
  if (!isa<DeclRefExpr>(E) && !LV.isBitField() && LV.isSimple())
    EmitCheck(LV.getAddress(), 
              getContext().getTypeSizeInChars(E->getType()).getQuantity());
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

  case Expr::ObjCPropertyRefExprClass:
    llvm_unreachable("cannot emit a property reference directly");

  case Expr::ObjCSelectorExprClass:
  return EmitObjCSelectorLValue(cast<ObjCSelectorExpr>(E));
  case Expr::ObjCIsaExprClass:
    return EmitObjCIsaExpr(cast<ObjCIsaExpr>(E));
  case Expr::BinaryOperatorClass:
    return EmitBinaryOperatorLValue(cast<BinaryOperator>(E));
  case Expr::CompoundAssignOperatorClass:
    if (!E->getType()->isAnyComplexType())
      return EmitCompoundAssignmentLValue(cast<CompoundAssignOperator>(E));
    return EmitComplexCompoundAssignmentLValue(cast<CompoundAssignOperator>(E));
  case Expr::CallExprClass:
  case Expr::CXXMemberCallExprClass:
  case Expr::CXXOperatorCallExprClass:
  case Expr::UserDefinedLiteralClass:
    return EmitCallExprLValue(cast<CallExpr>(E));
  case Expr::VAArgExprClass:
    return EmitVAArgExprLValue(cast<VAArgExpr>(E));
  case Expr::DeclRefExprClass:
    return EmitDeclRefLValue(cast<DeclRefExpr>(E));
  case Expr::ParenExprClass:
    return EmitLValue(cast<ParenExpr>(E)->getSubExpr());
  case Expr::GenericSelectionExprClass:
    return EmitLValue(cast<GenericSelectionExpr>(E)->getResultExpr());
  case Expr::PredefinedExprClass:
    return EmitPredefinedLValue(cast<PredefinedExpr>(E));
  case Expr::StringLiteralClass:
    return EmitStringLiteralLValue(cast<StringLiteral>(E));
  case Expr::ObjCEncodeExprClass:
    return EmitObjCEncodeExprLValue(cast<ObjCEncodeExpr>(E));
  case Expr::PseudoObjectExprClass:
    return EmitPseudoObjectLValue(cast<PseudoObjectExpr>(E));
  case Expr::InitListExprClass:
    assert(cast<InitListExpr>(E)->getNumInits() == 1 &&
           "Only single-element init list can be lvalue.");
    return EmitLValue(cast<InitListExpr>(E)->getInit(0));

  case Expr::CXXTemporaryObjectExprClass:
  case Expr::CXXConstructExprClass:
    return EmitCXXConstructLValue(cast<CXXConstructExpr>(E));
  case Expr::CXXBindTemporaryExprClass:
    return EmitCXXBindTemporaryLValue(cast<CXXBindTemporaryExpr>(E));
  case Expr::LambdaExprClass:
    return EmitLambdaLValue(cast<LambdaExpr>(E));

  case Expr::ExprWithCleanupsClass: {
    const ExprWithCleanups *cleanups = cast<ExprWithCleanups>(E);
    enterFullExpression(cleanups);
    RunCleanupsScope Scope(*this);
    return EmitLValue(cleanups->getSubExpr());
  }

  case Expr::CXXScalarValueInitExprClass:
    return EmitNullInitializationLValue(cast<CXXScalarValueInitExpr>(E));
  case Expr::CXXDefaultArgExprClass:
    return EmitLValue(cast<CXXDefaultArgExpr>(E)->getExpr());
  case Expr::CXXTypeidExprClass:
    return EmitCXXTypeidLValue(cast<CXXTypeidExpr>(E));

  case Expr::ObjCMessageExprClass:
    return EmitObjCMessageExprLValue(cast<ObjCMessageExpr>(E));
  case Expr::ObjCIvarRefExprClass:
    return EmitObjCIvarRefLValue(cast<ObjCIvarRefExpr>(E));
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
  case Expr::BinaryConditionalOperatorClass:
    return EmitConditionalOperatorLValue(cast<BinaryConditionalOperator>(E));
  case Expr::ChooseExprClass:
    return EmitLValue(cast<ChooseExpr>(E)->getChosenSubExpr(getContext()));
  case Expr::OpaqueValueExprClass:
    return EmitOpaqueValueLValue(cast<OpaqueValueExpr>(E));
  case Expr::SubstNonTypeTemplateParmExprClass:
    return EmitLValue(cast<SubstNonTypeTemplateParmExpr>(E)->getReplacement());
  case Expr::ImplicitCastExprClass:
  case Expr::CStyleCastExprClass:
  case Expr::CXXFunctionalCastExprClass:
  case Expr::CXXStaticCastExprClass:
  case Expr::CXXDynamicCastExprClass:
  case Expr::CXXReinterpretCastExprClass:
  case Expr::CXXConstCastExprClass:
  case Expr::ObjCBridgedCastExprClass:
    return EmitCastLValue(cast<CastExpr>(E));

  case Expr::MaterializeTemporaryExprClass:
    return EmitMaterializeTemporaryExpr(cast<MaterializeTemporaryExpr>(E));
  }
}

/// Given an object of the given canonical type, can we safely copy a
/// value out of it based on its initializer?
static bool isConstantEmittableObjectType(QualType type) {
  assert(type.isCanonical());
  assert(!type->isReferenceType());

  // Must be const-qualified but non-volatile.
  Qualifiers qs = type.getLocalQualifiers();
  if (!qs.hasConst() || qs.hasVolatile()) return false;

  // Otherwise, all object types satisfy this except C++ classes with
  // mutable subobjects or non-trivial copy/destroy behavior.
  if (const RecordType *RT = dyn_cast<RecordType>(type))
    if (const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(RT->getDecl()))
      if (RD->hasMutableFields() || !RD->isTrivial())
        return false;

  return true;
}

/// Can we constant-emit a load of a reference to a variable of the
/// given type?  This is different from predicates like
/// Decl::isUsableInConstantExpressions because we do want it to apply
/// in situations that don't necessarily satisfy the language's rules
/// for this (e.g. C++'s ODR-use rules).  For example, we want to able
/// to do this with const float variables even if those variables
/// aren't marked 'constexpr'.
enum ConstantEmissionKind {
  CEK_None,
  CEK_AsReferenceOnly,
  CEK_AsValueOrReference,
  CEK_AsValueOnly
};
static ConstantEmissionKind checkVarTypeForConstantEmission(QualType type) {
  type = type.getCanonicalType();
  if (const ReferenceType *ref = dyn_cast<ReferenceType>(type)) {
    if (isConstantEmittableObjectType(ref->getPointeeType()))
      return CEK_AsValueOrReference;
    return CEK_AsReferenceOnly;
  }
  if (isConstantEmittableObjectType(type))
    return CEK_AsValueOnly;
  return CEK_None;
}

/// Try to emit a reference to the given value without producing it as
/// an l-value.  This is actually more than an optimization: we can't
/// produce an l-value for variables that we never actually captured
/// in a block or lambda, which means const int variables or constexpr
/// literals or similar.
CodeGenFunction::ConstantEmission
CodeGenFunction::tryEmitAsConstant(DeclRefExpr *refExpr) {
  ValueDecl *value = refExpr->getDecl();

  // The value needs to be an enum constant or a constant variable.
  ConstantEmissionKind CEK;
  if (isa<ParmVarDecl>(value)) {
    CEK = CEK_None;
  } else if (VarDecl *var = dyn_cast<VarDecl>(value)) {
    CEK = checkVarTypeForConstantEmission(var->getType());
  } else if (isa<EnumConstantDecl>(value)) {
    CEK = CEK_AsValueOnly;
  } else {
    CEK = CEK_None;
  }
  if (CEK == CEK_None) return ConstantEmission();

  Expr::EvalResult result;
  bool resultIsReference;
  QualType resultType;

  // It's best to evaluate all the way as an r-value if that's permitted.
  if (CEK != CEK_AsReferenceOnly &&
      refExpr->EvaluateAsRValue(result, getContext())) {
    resultIsReference = false;
    resultType = refExpr->getType();

  // Otherwise, try to evaluate as an l-value.
  } else if (CEK != CEK_AsValueOnly &&
             refExpr->EvaluateAsLValue(result, getContext())) {
    resultIsReference = true;
    resultType = value->getType();

  // Failure.
  } else {
    return ConstantEmission();
  }

  // In any case, if the initializer has side-effects, abandon ship.
  if (result.HasSideEffects)
    return ConstantEmission();

  // Emit as a constant.
  llvm::Constant *C = CGM.EmitConstantValue(result.Val, resultType, this);

  // Make sure we emit a debug reference to the global variable.
  // This should probably fire even for 
  if (isa<VarDecl>(value)) {
    if (!getContext().DeclMustBeEmitted(cast<VarDecl>(value)))
      EmitDeclRefExprDbgValue(refExpr, C);
  } else {
    assert(isa<EnumConstantDecl>(value));
    EmitDeclRefExprDbgValue(refExpr, C);
  }

  // If we emitted a reference constant, we need to dereference that.
  if (resultIsReference)
    return ConstantEmission::forReference(C);

  return ConstantEmission::forValue(C);
}

llvm::Value *CodeGenFunction::EmitLoadOfScalar(LValue lvalue) {
  return EmitLoadOfScalar(lvalue.getAddress(), lvalue.isVolatile(),
                          lvalue.getAlignment().getQuantity(),
                          lvalue.getType(), lvalue.getTBAAInfo());
}

llvm::Value *CodeGenFunction::EmitLoadOfScalar(llvm::Value *Addr, bool Volatile,
                                              unsigned Alignment, QualType Ty,
                                              llvm::MDNode *TBAAInfo) {
  llvm::LoadInst *Load = Builder.CreateLoad(Addr);
  if (Volatile)
    Load->setVolatile(true);
  if (Alignment)
    Load->setAlignment(Alignment);
  if (TBAAInfo)
    CGM.DecorateInstruction(Load, TBAAInfo);
  // If this is an atomic type, all normal reads must be atomic
  if (Ty->isAtomicType())
    Load->setAtomic(llvm::SequentiallyConsistent);

  return EmitFromMemory(Load, Ty);
}

static bool isBooleanUnderlyingType(QualType Ty) {
  if (const EnumType *ET = dyn_cast<EnumType>(Ty))
    return ET->getDecl()->getIntegerType()->isBooleanType();
  return false;
}

llvm::Value *CodeGenFunction::EmitToMemory(llvm::Value *Value, QualType Ty) {
  // Bool has a different representation in memory than in registers.
  if (Ty->isBooleanType() || isBooleanUnderlyingType(Ty)) {
    // This should really always be an i1, but sometimes it's already
    // an i8, and it's awkward to track those cases down.
    if (Value->getType()->isIntegerTy(1))
      return Builder.CreateZExt(Value, Builder.getInt8Ty(), "frombool");
    assert(Value->getType()->isIntegerTy(8) && "value rep of bool not i1/i8");
  }

  return Value;
}

llvm::Value *CodeGenFunction::EmitFromMemory(llvm::Value *Value, QualType Ty) {
  // Bool has a different representation in memory than in registers.
  if (Ty->isBooleanType() || isBooleanUnderlyingType(Ty)) {
    assert(Value->getType()->isIntegerTy(8) && "memory rep of bool not i8");
    return Builder.CreateTrunc(Value, Builder.getInt1Ty(), "tobool");
  }

  return Value;
}

void CodeGenFunction::EmitStoreOfScalar(llvm::Value *Value, llvm::Value *Addr,
                                        bool Volatile, unsigned Alignment,
                                        QualType Ty,
                                        llvm::MDNode *TBAAInfo,
                                        bool isInit) {
  Value = EmitToMemory(Value, Ty);
  
  llvm::StoreInst *Store = Builder.CreateStore(Value, Addr, Volatile);
  if (Alignment)
    Store->setAlignment(Alignment);
  if (TBAAInfo)
    CGM.DecorateInstruction(Store, TBAAInfo);
  if (!isInit && Ty->isAtomicType())
    Store->setAtomic(llvm::SequentiallyConsistent);
}

void CodeGenFunction::EmitStoreOfScalar(llvm::Value *value, LValue lvalue,
    bool isInit) {
  EmitStoreOfScalar(value, lvalue.getAddress(), lvalue.isVolatile(),
                    lvalue.getAlignment().getQuantity(), lvalue.getType(),
                    lvalue.getTBAAInfo(), isInit);
}

/// EmitLoadOfLValue - Given an expression that represents a value lvalue, this
/// method emits the address of the lvalue, then loads the result as an rvalue,
/// returning the rvalue.
RValue CodeGenFunction::EmitLoadOfLValue(LValue LV) {
  if (LV.isObjCWeak()) {
    // load of a __weak object.
    llvm::Value *AddrWeakObj = LV.getAddress();
    return RValue::get(CGM.getObjCRuntime().EmitObjCWeakRead(*this,
                                                             AddrWeakObj));
  }
  if (LV.getQuals().getObjCLifetime() == Qualifiers::OCL_Weak)
    return RValue::get(EmitARCLoadWeak(LV.getAddress()));

  if (LV.isSimple()) {
    assert(!LV.getType()->isFunctionType());

    // Everything needs a load.
    return RValue::get(EmitLoadOfScalar(LV));
  }

  if (LV.isVectorElt()) {
    llvm::LoadInst *Load = Builder.CreateLoad(LV.getVectorAddr(),
                                              LV.isVolatileQualified());
    Load->setAlignment(LV.getAlignment().getQuantity());
    return RValue::get(Builder.CreateExtractElement(Load, LV.getVectorIdx(),
                                                    "vecext"));
  }

  // If this is a reference to a subset of the elements of a vector, either
  // shuffle the input or extract/insert them as appropriate.
  if (LV.isExtVectorElt())
    return EmitLoadOfExtVectorElementLValue(LV);

  assert(LV.isBitField() && "Unknown LValue type!");
  return EmitLoadOfBitfieldLValue(LV);
}

RValue CodeGenFunction::EmitLoadOfBitfieldLValue(LValue LV) {
  const CGBitFieldInfo &Info = LV.getBitFieldInfo();

  // Get the output type.
  llvm::Type *ResLTy = ConvertType(LV.getType());
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
    if (!AI.FieldByteOffset.isZero()) {
      Ptr = EmitCastToVoidPtr(Ptr);
      Ptr = Builder.CreateConstGEP1_32(Ptr, AI.FieldByteOffset.getQuantity(),
                                       "bf.field.offs");
    }

    // Cast to the access type.
    llvm::Type *PTy = llvm::Type::getIntNPtrTy(getLLVMContext(), AI.AccessWidth,
                       CGM.getContext().getTargetAddressSpace(LV.getType()));
    Ptr = Builder.CreateBitCast(Ptr, PTy);

    // Perform the load.
    llvm::LoadInst *Load = Builder.CreateLoad(Ptr, LV.isVolatileQualified());
    if (!AI.AccessAlignment.isZero())
      Load->setAlignment(AI.AccessAlignment.getQuantity());

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

// If this is a reference to a subset of the elements of a vector, create an
// appropriate shufflevector.
RValue CodeGenFunction::EmitLoadOfExtVectorElementLValue(LValue LV) {
  llvm::LoadInst *Load = Builder.CreateLoad(LV.getExtVectorAddr(),
                                            LV.isVolatileQualified());
  Load->setAlignment(LV.getAlignment().getQuantity());
  llvm::Value *Vec = Load;

  const llvm::Constant *Elts = LV.getExtVectorElts();

  // If the result of the expression is a non-vector type, we must be extracting
  // a single element.  Just codegen as an extractelement.
  const VectorType *ExprVT = LV.getType()->getAs<VectorType>();
  if (!ExprVT) {
    unsigned InIdx = getAccessedFieldNo(0, Elts);
    llvm::Value *Elt = llvm::ConstantInt::get(Int32Ty, InIdx);
    return RValue::get(Builder.CreateExtractElement(Vec, Elt));
  }

  // Always use shuffle vector to try to retain the original program structure
  unsigned NumResultElts = ExprVT->getNumElements();

  SmallVector<llvm::Constant*, 4> Mask;
  for (unsigned i = 0; i != NumResultElts; ++i)
    Mask.push_back(Builder.getInt32(getAccessedFieldNo(i, Elts)));

  llvm::Value *MaskV = llvm::ConstantVector::get(Mask);
  Vec = Builder.CreateShuffleVector(Vec, llvm::UndefValue::get(Vec->getType()),
                                    MaskV);
  return RValue::get(Vec);
}



/// EmitStoreThroughLValue - Store the specified rvalue into the specified
/// lvalue, where both are guaranteed to the have the same type, and that type
/// is 'Ty'.
void CodeGenFunction::EmitStoreThroughLValue(RValue Src, LValue Dst, bool isInit) {
  if (!Dst.isSimple()) {
    if (Dst.isVectorElt()) {
      // Read/modify/write the vector, inserting the new element.
      llvm::LoadInst *Load = Builder.CreateLoad(Dst.getVectorAddr(),
                                                Dst.isVolatileQualified());
      Load->setAlignment(Dst.getAlignment().getQuantity());
      llvm::Value *Vec = Load;
      Vec = Builder.CreateInsertElement(Vec, Src.getScalarVal(),
                                        Dst.getVectorIdx(), "vecins");
      llvm::StoreInst *Store = Builder.CreateStore(Vec, Dst.getVectorAddr(),
                                                   Dst.isVolatileQualified());
      Store->setAlignment(Dst.getAlignment().getQuantity());
      return;
    }

    // If this is an update of extended vector elements, insert them as
    // appropriate.
    if (Dst.isExtVectorElt())
      return EmitStoreThroughExtVectorComponentLValue(Src, Dst);

    assert(Dst.isBitField() && "Unknown LValue type");
    return EmitStoreThroughBitfieldLValue(Src, Dst);
  }

  // There's special magic for assigning into an ARC-qualified l-value.
  if (Qualifiers::ObjCLifetime Lifetime = Dst.getQuals().getObjCLifetime()) {
    switch (Lifetime) {
    case Qualifiers::OCL_None:
      llvm_unreachable("present but none");

    case Qualifiers::OCL_ExplicitNone:
      // nothing special
      break;

    case Qualifiers::OCL_Strong:
      EmitARCStoreStrong(Dst, Src.getScalarVal(), /*ignore*/ true);
      return;

    case Qualifiers::OCL_Weak:
      EmitARCStoreWeak(Dst.getAddress(), Src.getScalarVal(), /*ignore*/ true);
      return;

    case Qualifiers::OCL_Autoreleasing:
      Src = RValue::get(EmitObjCExtendObjectLifetime(Dst.getType(),
                                                     Src.getScalarVal()));
      // fall into the normal path
      break;
    }
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
      llvm::Type *ResultType = ConvertType(getContext().LongTy);
      llvm::Value *RHS = EmitScalarExpr(Dst.getBaseIvarExp());
      llvm::Value *dst = RHS;
      RHS = Builder.CreatePtrToInt(RHS, ResultType, "sub.ptr.rhs.cast");
      llvm::Value *LHS = 
        Builder.CreatePtrToInt(LvalueDst, ResultType, "sub.ptr.lhs.cast");
      llvm::Value *BytesBetween = Builder.CreateSub(LHS, RHS, "ivar.offset");
      CGM.getObjCRuntime().EmitObjCIvarAssign(*this, src, dst,
                                              BytesBetween);
    } else if (Dst.isGlobalObjCRef()) {
      CGM.getObjCRuntime().EmitObjCGlobalAssign(*this, src, LvalueDst,
                                                Dst.isThreadLocalRef());
    }
    else
      CGM.getObjCRuntime().EmitObjCStrongCastAssign(*this, src, LvalueDst);
    return;
  }

  assert(Src.isScalar() && "Can't emit an agg store with this method");
  EmitStoreOfScalar(Src.getScalarVal(), Dst, isInit);
}

void CodeGenFunction::EmitStoreThroughBitfieldLValue(RValue Src, LValue Dst,
                                                     llvm::Value **Result) {
  const CGBitFieldInfo &Info = Dst.getBitFieldInfo();

  // Get the output type.
  llvm::Type *ResLTy = ConvertTypeForMem(Dst.getType());
  unsigned ResSizeInBits = CGM.getTargetData().getTypeSizeInBits(ResLTy);

  // Get the source value, truncated to the width of the bit-field.
  llvm::Value *SrcVal = Src.getScalarVal();

  if (Dst.getType()->isBooleanType())
    SrcVal = Builder.CreateIntCast(SrcVal, ResLTy, /*IsSigned=*/false);

  SrcVal = Builder.CreateAnd(SrcVal, llvm::APInt::getLowBitsSet(ResSizeInBits,
                                                                Info.getSize()),
                             "bf.value");

  // Return the new value of the bit-field, if requested.
  if (Result) {
    // Cast back to the proper type for result.
    llvm::Type *SrcTy = Src.getScalarVal()->getType();
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
    unsigned addressSpace =
      cast<llvm::PointerType>(Ptr->getType())->getAddressSpace();

    // Only offset by the field index if used, so that incoming values are not
    // required to be structures.
    if (AI.FieldIndex)
      Ptr = Builder.CreateStructGEP(Ptr, AI.FieldIndex, "bf.field");

    // Offset by the byte offset, if used.
    if (!AI.FieldByteOffset.isZero()) {
      Ptr = EmitCastToVoidPtr(Ptr);
      Ptr = Builder.CreateConstGEP1_32(Ptr, AI.FieldByteOffset.getQuantity(),
                                       "bf.field.offs");
    }

    // Cast to the access type.
    llvm::Type *AccessLTy =
      llvm::Type::getIntNTy(getLLVMContext(), AI.AccessWidth);

    llvm::Type *PTy = AccessLTy->getPointerTo(addressSpace);
    Ptr = Builder.CreateBitCast(Ptr, PTy);

    // Extract the piece of the bit-field value to write in this access, limited
    // to the values that are part of this access.
    llvm::Value *Val = SrcVal;
    if (AI.TargetBitOffset)
      Val = Builder.CreateLShr(Val, AI.TargetBitOffset);
    Val = Builder.CreateAnd(Val, llvm::APInt::getLowBitsSet(ResSizeInBits,
                                                            AI.TargetBitWidth));

    // Extend or truncate to the access size.
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
      if (!AI.AccessAlignment.isZero())
        Load->setAlignment(AI.AccessAlignment.getQuantity());

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
    if (!AI.AccessAlignment.isZero())
      Store->setAlignment(AI.AccessAlignment.getQuantity());
  }
}

void CodeGenFunction::EmitStoreThroughExtVectorComponentLValue(RValue Src,
                                                               LValue Dst) {
  // This access turns into a read/modify/write of the vector.  Load the input
  // value now.
  llvm::LoadInst *Load = Builder.CreateLoad(Dst.getExtVectorAddr(),
                                            Dst.isVolatileQualified());
  Load->setAlignment(Dst.getAlignment().getQuantity());
  llvm::Value *Vec = Load;
  const llvm::Constant *Elts = Dst.getExtVectorElts();

  llvm::Value *SrcVal = Src.getScalarVal();

  if (const VectorType *VTy = Dst.getType()->getAs<VectorType>()) {
    unsigned NumSrcElts = VTy->getNumElements();
    unsigned NumDstElts =
       cast<llvm::VectorType>(Vec->getType())->getNumElements();
    if (NumDstElts == NumSrcElts) {
      // Use shuffle vector is the src and destination are the same number of
      // elements and restore the vector mask since it is on the side it will be
      // stored.
      SmallVector<llvm::Constant*, 4> Mask(NumDstElts);
      for (unsigned i = 0; i != NumSrcElts; ++i)
        Mask[getAccessedFieldNo(i, Elts)] = Builder.getInt32(i);

      llvm::Value *MaskV = llvm::ConstantVector::get(Mask);
      Vec = Builder.CreateShuffleVector(SrcVal,
                                        llvm::UndefValue::get(Vec->getType()),
                                        MaskV);
    } else if (NumDstElts > NumSrcElts) {
      // Extended the source vector to the same length and then shuffle it
      // into the destination.
      // FIXME: since we're shuffling with undef, can we just use the indices
      //        into that?  This could be simpler.
      SmallVector<llvm::Constant*, 4> ExtMask;
      for (unsigned i = 0; i != NumSrcElts; ++i)
        ExtMask.push_back(Builder.getInt32(i));
      ExtMask.resize(NumDstElts, llvm::UndefValue::get(Int32Ty));
      llvm::Value *ExtMaskV = llvm::ConstantVector::get(ExtMask);
      llvm::Value *ExtSrcVal =
        Builder.CreateShuffleVector(SrcVal,
                                    llvm::UndefValue::get(SrcVal->getType()),
                                    ExtMaskV);
      // build identity
      SmallVector<llvm::Constant*, 4> Mask;
      for (unsigned i = 0; i != NumDstElts; ++i)
        Mask.push_back(Builder.getInt32(i));

      // modify when what gets shuffled in
      for (unsigned i = 0; i != NumSrcElts; ++i)
        Mask[getAccessedFieldNo(i, Elts)] = Builder.getInt32(i+NumDstElts);
      llvm::Value *MaskV = llvm::ConstantVector::get(Mask);
      Vec = Builder.CreateShuffleVector(Vec, ExtSrcVal, MaskV);
    } else {
      // We should never shorten the vector
      llvm_unreachable("unexpected shorten vector length");
    }
  } else {
    // If the Src is a scalar (not a vector) it must be updating one element.
    unsigned InIdx = getAccessedFieldNo(0, Elts);
    llvm::Value *Elt = llvm::ConstantInt::get(Int32Ty, InIdx);
    Vec = Builder.CreateInsertElement(Vec, SrcVal, Elt);
  }

  llvm::StoreInst *Store = Builder.CreateStore(Vec, Dst.getExtVectorAddr(),
                                               Dst.isVolatileQualified());
  Store->setAlignment(Dst.getAlignment().getQuantity());
}

// setObjCGCLValueClass - sets class of he lvalue for the purpose of
// generating write-barries API. It is currently a global, ivar,
// or neither.
static void setObjCGCLValueClass(const ASTContext &Ctx, const Expr *E,
                                 LValue &LV,
                                 bool IsMemberAccess=false) {
  if (Ctx.getLangOpts().getGC() == LangOptions::NonGC)
    return;
  
  if (isa<ObjCIvarRefExpr>(E)) {
    QualType ExpTy = E->getType();
    if (IsMemberAccess && ExpTy->isPointerType()) {
      // If ivar is a structure pointer, assigning to field of
      // this struct follows gcc's behavior and makes it a non-ivar 
      // writer-barrier conservatively.
      ExpTy = ExpTy->getAs<PointerType>()->getPointeeType();
      if (ExpTy->isRecordType()) {
        LV.setObjCIvar(false);
        return;
      }
    }
    LV.setObjCIvar(true);
    ObjCIvarRefExpr *Exp = cast<ObjCIvarRefExpr>(const_cast<Expr*>(E));
    LV.setBaseIvarExp(Exp->getBase());
    LV.setObjCArray(E->getType()->isArrayType());
    return;
  }
  
  if (const DeclRefExpr *Exp = dyn_cast<DeclRefExpr>(E)) {
    if (const VarDecl *VD = dyn_cast<VarDecl>(Exp->getDecl())) {
      if (VD->hasGlobalStorage()) {
        LV.setGlobalObjCRef(true);
        LV.setThreadLocalRef(VD->isThreadSpecified());
      }
    }
    LV.setObjCArray(E->getType()->isArrayType());
    return;
  }
  
  if (const UnaryOperator *Exp = dyn_cast<UnaryOperator>(E)) {
    setObjCGCLValueClass(Ctx, Exp->getSubExpr(), LV, IsMemberAccess);
    return;
  }
  
  if (const ParenExpr *Exp = dyn_cast<ParenExpr>(E)) {
    setObjCGCLValueClass(Ctx, Exp->getSubExpr(), LV, IsMemberAccess);
    if (LV.isObjCIvar()) {
      // If cast is to a structure pointer, follow gcc's behavior and make it
      // a non-ivar write-barrier.
      QualType ExpTy = E->getType();
      if (ExpTy->isPointerType())
        ExpTy = ExpTy->getAs<PointerType>()->getPointeeType();
      if (ExpTy->isRecordType())
        LV.setObjCIvar(false); 
    }
    return;
  }

  if (const GenericSelectionExpr *Exp = dyn_cast<GenericSelectionExpr>(E)) {
    setObjCGCLValueClass(Ctx, Exp->getResultExpr(), LV);
    return;
  }

  if (const ImplicitCastExpr *Exp = dyn_cast<ImplicitCastExpr>(E)) {
    setObjCGCLValueClass(Ctx, Exp->getSubExpr(), LV, IsMemberAccess);
    return;
  }
  
  if (const CStyleCastExpr *Exp = dyn_cast<CStyleCastExpr>(E)) {
    setObjCGCLValueClass(Ctx, Exp->getSubExpr(), LV, IsMemberAccess);
    return;
  }

  if (const ObjCBridgedCastExpr *Exp = dyn_cast<ObjCBridgedCastExpr>(E)) {
    setObjCGCLValueClass(Ctx, Exp->getSubExpr(), LV, IsMemberAccess);
    return;
  }

  if (const ArraySubscriptExpr *Exp = dyn_cast<ArraySubscriptExpr>(E)) {
    setObjCGCLValueClass(Ctx, Exp->getBase(), LV);
    if (LV.isObjCIvar() && !LV.isObjCArray()) 
      // Using array syntax to assigning to what an ivar points to is not 
      // same as assigning to the ivar itself. {id *Names;} Names[i] = 0;
      LV.setObjCIvar(false); 
    else if (LV.isGlobalObjCRef() && !LV.isObjCArray())
      // Using array syntax to assigning to what global points to is not 
      // same as assigning to the global itself. {id *G;} G[i] = 0;
      LV.setGlobalObjCRef(false);
    return;
  }

  if (const MemberExpr *Exp = dyn_cast<MemberExpr>(E)) {
    setObjCGCLValueClass(Ctx, Exp->getBase(), LV, true);
    // We don't know if member is an 'ivar', but this flag is looked at
    // only in the context of LV.isObjCIvar().
    LV.setObjCArray(E->getType()->isArrayType());
    return;
  }
}

static llvm::Value *
EmitBitCastOfLValueToProperType(CodeGenFunction &CGF,
                                llvm::Value *V, llvm::Type *IRType,
                                StringRef Name = StringRef()) {
  unsigned AS = cast<llvm::PointerType>(V->getType())->getAddressSpace();
  return CGF.Builder.CreateBitCast(V, IRType->getPointerTo(AS), Name);
}

static LValue EmitGlobalVarDeclLValue(CodeGenFunction &CGF,
                                      const Expr *E, const VarDecl *VD) {
  assert((VD->hasExternalStorage() || VD->isFileVarDecl()) &&
         "Var decl must have external storage or be a file var decl!");

  llvm::Value *V = CGF.CGM.GetAddrOfGlobalVar(VD);
  llvm::Type *RealVarTy = CGF.getTypes().ConvertTypeForMem(VD->getType());
  V = EmitBitCastOfLValueToProperType(CGF, V, RealVarTy);
  CharUnits Alignment = CGF.getContext().getDeclAlign(VD);
  QualType T = E->getType();
  LValue LV;
  if (VD->getType()->isReferenceType()) {
    llvm::LoadInst *LI = CGF.Builder.CreateLoad(V);
    LI->setAlignment(Alignment.getQuantity());
    V = LI;
    LV = CGF.MakeNaturalAlignAddrLValue(V, T);
  } else {
    LV = CGF.MakeAddrLValue(V, E->getType(), Alignment);
  }
  setObjCGCLValueClass(CGF.getContext(), E, LV);
  return LV;
}

static LValue EmitFunctionDeclLValue(CodeGenFunction &CGF,
                                     const Expr *E, const FunctionDecl *FD) {
  llvm::Value *V = CGF.CGM.GetAddrOfFunction(FD);
  if (!FD->hasPrototype()) {
    if (const FunctionProtoType *Proto =
            FD->getType()->getAs<FunctionProtoType>()) {
      // Ugly case: for a K&R-style definition, the type of the definition
      // isn't the same as the type of a use.  Correct for this with a
      // bitcast.
      QualType NoProtoType =
          CGF.getContext().getFunctionNoProtoType(Proto->getResultType());
      NoProtoType = CGF.getContext().getPointerType(NoProtoType);
      V = CGF.Builder.CreateBitCast(V, CGF.ConvertType(NoProtoType));
    }
  }
  CharUnits Alignment = CGF.getContext().getDeclAlign(FD);
  return CGF.MakeAddrLValue(V, E->getType(), Alignment);
}

LValue CodeGenFunction::EmitDeclRefLValue(const DeclRefExpr *E) {
  const NamedDecl *ND = E->getDecl();
  CharUnits Alignment = getContext().getDeclAlign(ND);
  QualType T = E->getType();

  // FIXME: We should be able to assert this for FunctionDecls as well!
  // FIXME: We should be able to assert this for all DeclRefExprs, not just
  // those with a valid source location.
  assert((ND->isUsed(false) || !isa<VarDecl>(ND) ||
          !E->getLocation().isValid()) &&
         "Should not use decl without marking it used!");

  if (ND->hasAttr<WeakRefAttr>()) {
    const ValueDecl *VD = cast<ValueDecl>(ND);
    llvm::Constant *Aliasee = CGM.GetWeakRefReference(VD);
    return MakeAddrLValue(Aliasee, E->getType(), Alignment);
  }

  if (const VarDecl *VD = dyn_cast<VarDecl>(ND)) {
    // Check if this is a global variable.
    if (VD->hasExternalStorage() || VD->isFileVarDecl()) 
      return EmitGlobalVarDeclLValue(*this, E, VD);

    bool isBlockVariable = VD->hasAttr<BlocksAttr>();

    bool NonGCable = VD->hasLocalStorage() &&
                     !VD->getType()->isReferenceType() &&
                     !isBlockVariable;

    llvm::Value *V = LocalDeclMap[VD];
    if (!V && VD->isStaticLocal()) 
      V = CGM.getStaticLocalDeclAddress(VD);

    // Use special handling for lambdas.
    if (!V) {
      if (FieldDecl *FD = LambdaCaptureFields.lookup(VD))
        return EmitLValueForField(CXXABIThisValue, FD, 0);

      assert(isa<BlockDecl>(CurCodeDecl) && E->refersToEnclosingLocal());
      CharUnits alignment = getContext().getDeclAlign(VD);
      return MakeAddrLValue(GetAddrOfBlockDecl(VD, isBlockVariable),
                            E->getType(), alignment);
    }

    assert(V && "DeclRefExpr not entered in LocalDeclMap?");

    if (isBlockVariable)
      V = BuildBlockByrefAddress(V, VD);

    LValue LV;
    if (VD->getType()->isReferenceType()) {
      llvm::LoadInst *LI = Builder.CreateLoad(V);
      LI->setAlignment(Alignment.getQuantity());
      V = LI;
      LV = MakeNaturalAlignAddrLValue(V, T);
    } else {
      LV = MakeAddrLValue(V, T, Alignment);
    }

    if (NonGCable) {
      LV.getQuals().removeObjCGCAttr();
      LV.setNonGC(true);
    }
    setObjCGCLValueClass(getContext(), E, LV);
    return LV;
  }

  if (const FunctionDecl *fn = dyn_cast<FunctionDecl>(ND))
    return EmitFunctionDeclLValue(*this, E, fn);

  llvm_unreachable("Unhandled DeclRefExpr");
}

LValue CodeGenFunction::EmitUnaryOpLValue(const UnaryOperator *E) {
  // __extension__ doesn't affect lvalue-ness.
  if (E->getOpcode() == UO_Extension)
    return EmitLValue(E->getSubExpr());

  QualType ExprTy = getContext().getCanonicalType(E->getSubExpr()->getType());
  switch (E->getOpcode()) {
  default: llvm_unreachable("Unknown unary operator lvalue!");
  case UO_Deref: {
    QualType T = E->getSubExpr()->getType()->getPointeeType();
    assert(!T.isNull() && "CodeGenFunction::EmitUnaryOpLValue: Illegal type");

    LValue LV = MakeNaturalAlignAddrLValue(EmitScalarExpr(E->getSubExpr()), T);
    LV.getQuals().setAddressSpace(ExprTy.getAddressSpace());

    // We should not generate __weak write barrier on indirect reference
    // of a pointer to object; as in void foo (__weak id *param); *param = 0;
    // But, we continue to generate __strong write barrier on indirect write
    // into a pointer to object.
    if (getContext().getLangOpts().ObjC1 &&
        getContext().getLangOpts().getGC() != LangOptions::NonGC &&
        LV.isObjCWeak())
      LV.setNonGC(!E->isOBJCGCCandidate(getContext()));
    return LV;
  }
  case UO_Real:
  case UO_Imag: {
    LValue LV = EmitLValue(E->getSubExpr());
    assert(LV.isSimple() && "real/imag on non-ordinary l-value");
    llvm::Value *Addr = LV.getAddress();

    // __real is valid on scalars.  This is a faster way of testing that.
    // __imag can only produce an rvalue on scalars.
    if (E->getOpcode() == UO_Real &&
        !cast<llvm::PointerType>(Addr->getType())
           ->getElementType()->isStructTy()) {
      assert(E->getSubExpr()->getType()->isArithmeticType());
      return LV;
    }

    assert(E->getSubExpr()->getType()->isAnyComplexType());

    unsigned Idx = E->getOpcode() == UO_Imag;
    return MakeAddrLValue(Builder.CreateStructGEP(LV.getAddress(),
                                                  Idx, "idx"),
                          ExprTy);
  }
  case UO_PreInc:
  case UO_PreDec: {
    LValue LV = EmitLValue(E->getSubExpr());
    bool isInc = E->getOpcode() == UO_PreInc;
    
    if (E->getType()->isAnyComplexType())
      EmitComplexPrePostIncDec(E, LV, isInc, true/*isPre*/);
    else
      EmitScalarPrePostIncDec(E, LV, isInc, true/*isPre*/);
    return LV;
  }
  }
}

LValue CodeGenFunction::EmitStringLiteralLValue(const StringLiteral *E) {
  return MakeAddrLValue(CGM.GetAddrOfConstantStringFromLiteral(E),
                        E->getType());
}

LValue CodeGenFunction::EmitObjCEncodeExprLValue(const ObjCEncodeExpr *E) {
  return MakeAddrLValue(CGM.GetAddrOfConstantStringFromObjCEncode(E),
                        E->getType());
}


LValue CodeGenFunction::EmitPredefinedLValue(const PredefinedExpr *E) {
  switch (E->getIdentType()) {
  default:
    return EmitUnsupportedLValue(E, "predefined expression");

  case PredefinedExpr::Func:
  case PredefinedExpr::Function:
  case PredefinedExpr::PrettyFunction: {
    unsigned Type = E->getIdentType();
    std::string GlobalVarName;

    switch (Type) {
    default: llvm_unreachable("Invalid type");
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

    StringRef FnName = CurFn->getName();
    if (FnName.startswith("\01"))
      FnName = FnName.substr(1);
    GlobalVarName += FnName;

    const Decl *CurDecl = CurCodeDecl;
    if (CurDecl == 0)
      CurDecl = getContext().getTranslationUnitDecl();

    std::string FunctionName =
        (isa<BlockDecl>(CurDecl)
         ? FnName.str()
         : PredefinedExpr::ComputeName((PredefinedExpr::IdentType)Type, CurDecl));

    llvm::Constant *C =
      CGM.GetAddrOfConstantCString(FunctionName, GlobalVarName.c_str());
    return MakeAddrLValue(C, E->getType());
  }
  }
}

llvm::BasicBlock *CodeGenFunction::getTrapBB() {
  const CodeGenOptions &GCO = CGM.getCodeGenOpts();

  // If we are not optimzing, don't collapse all calls to trap in the function
  // to the same call, that way, in the debugger they can see which operation
  // did in fact fail.  If we are optimizing, we collapse all calls to trap down
  // to just one per function to save on codesize.
  if (GCO.OptimizationLevel && TrapBB)
    return TrapBB;

  llvm::BasicBlock *Cont = 0;
  if (HaveInsertPoint()) {
    Cont = createBasicBlock("cont");
    EmitBranch(Cont);
  }
  TrapBB = createBasicBlock("trap");
  EmitBlock(TrapBB);

  llvm::Value *F = CGM.getIntrinsic(llvm::Intrinsic::trap);
  llvm::CallInst *TrapCall = Builder.CreateCall(F);
  TrapCall->setDoesNotReturn();
  TrapCall->setDoesNotThrow();
  Builder.CreateUnreachable();

  if (Cont)
    EmitBlock(Cont);
  return TrapBB;
}

/// isSimpleArrayDecayOperand - If the specified expr is a simple decay from an
/// array to pointer, return the array subexpression.
static const Expr *isSimpleArrayDecayOperand(const Expr *E) {
  // If this isn't just an array->pointer decay, bail out.
  const CastExpr *CE = dyn_cast<CastExpr>(E);
  if (CE == 0 || CE->getCastKind() != CK_ArrayToPointerDecay)
    return 0;
  
  // If this is a decay from variable width array, bail out.
  const Expr *SubExpr = CE->getSubExpr();
  if (SubExpr->getType()->isVariableArrayType())
    return 0;
  
  return SubExpr;
}

LValue CodeGenFunction::EmitArraySubscriptExpr(const ArraySubscriptExpr *E) {
  // The index must always be an integer, which is not an aggregate.  Emit it.
  llvm::Value *Idx = EmitScalarExpr(E->getIdx());
  QualType IdxTy  = E->getIdx()->getType();
  bool IdxSigned = IdxTy->isSignedIntegerOrEnumerationType();

  // If the base is a vector type, then we are forming a vector element lvalue
  // with this subscript.
  if (E->getBase()->getType()->isVectorType()) {
    // Emit the vector as an lvalue to get its address.
    LValue LHS = EmitLValue(E->getBase());
    assert(LHS.isSimple() && "Can only subscript lvalue vectors here!");
    Idx = Builder.CreateIntCast(Idx, Int32Ty, IdxSigned, "vidx");
    return LValue::MakeVectorElt(LHS.getAddress(), Idx,
                                 E->getBase()->getType(), LHS.getAlignment());
  }

  // Extend or truncate the index type to 32 or 64-bits.
  if (Idx->getType() != IntPtrTy)
    Idx = Builder.CreateIntCast(Idx, IntPtrTy, IdxSigned, "idxprom");
  
  // FIXME: As llvm implements the object size checking, this can come out.
  if (CatchUndefined) {
    if (const ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E->getBase())){
      if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(ICE->getSubExpr())) {
        if (ICE->getCastKind() == CK_ArrayToPointerDecay) {
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
  CharUnits ArrayAlignment;
  if (const VariableArrayType *vla =
        getContext().getAsVariableArrayType(E->getType())) {
    // The base must be a pointer, which is not an aggregate.  Emit
    // it.  It needs to be emitted first in case it's what captures
    // the VLA bounds.
    Address = EmitScalarExpr(E->getBase());

    // The element count here is the total number of non-VLA elements.
    llvm::Value *numElements = getVLASize(vla).first;

    // Effectively, the multiply by the VLA size is part of the GEP.
    // GEP indexes are signed, and scaling an index isn't permitted to
    // signed-overflow, so we use the same semantics for our explicit
    // multiply.  We suppress this if overflow is not undefined behavior.
    if (getLangOpts().isSignedOverflowDefined()) {
      Idx = Builder.CreateMul(Idx, numElements);
      Address = Builder.CreateGEP(Address, Idx, "arrayidx");
    } else {
      Idx = Builder.CreateNSWMul(Idx, numElements);
      Address = Builder.CreateInBoundsGEP(Address, Idx, "arrayidx");
    }
  } else if (const ObjCObjectType *OIT = E->getType()->getAs<ObjCObjectType>()){
    // Indexing over an interface, as in "NSString *P; P[4];"
    llvm::Value *InterfaceSize =
      llvm::ConstantInt::get(Idx->getType(),
          getContext().getTypeSizeInChars(OIT).getQuantity());

    Idx = Builder.CreateMul(Idx, InterfaceSize);

    // The base must be a pointer, which is not an aggregate.  Emit it.
    llvm::Value *Base = EmitScalarExpr(E->getBase());
    Address = EmitCastToVoidPtr(Base);
    Address = Builder.CreateGEP(Address, Idx, "arrayidx");
    Address = Builder.CreateBitCast(Address, Base->getType());
  } else if (const Expr *Array = isSimpleArrayDecayOperand(E->getBase())) {
    // If this is A[i] where A is an array, the frontend will have decayed the
    // base to be a ArrayToPointerDecay implicit cast.  While correct, it is
    // inefficient at -O0 to emit a "gep A, 0, 0" when codegen'ing it, then a
    // "gep x, i" here.  Emit one "gep A, 0, i".
    assert(Array->getType()->isArrayType() &&
           "Array to pointer decay must have array source type!");
    LValue ArrayLV = EmitLValue(Array);
    llvm::Value *ArrayPtr = ArrayLV.getAddress();
    llvm::Value *Zero = llvm::ConstantInt::get(Int32Ty, 0);
    llvm::Value *Args[] = { Zero, Idx };
    
    // Propagate the alignment from the array itself to the result.
    ArrayAlignment = ArrayLV.getAlignment();

    if (getContext().getLangOpts().isSignedOverflowDefined())
      Address = Builder.CreateGEP(ArrayPtr, Args, "arrayidx");
    else
      Address = Builder.CreateInBoundsGEP(ArrayPtr, Args, "arrayidx");
  } else {
    // The base must be a pointer, which is not an aggregate.  Emit it.
    llvm::Value *Base = EmitScalarExpr(E->getBase());
    if (getContext().getLangOpts().isSignedOverflowDefined())
      Address = Builder.CreateGEP(Base, Idx, "arrayidx");
    else
      Address = Builder.CreateInBoundsGEP(Base, Idx, "arrayidx");
  }

  QualType T = E->getBase()->getType()->getPointeeType();
  assert(!T.isNull() &&
         "CodeGenFunction::EmitArraySubscriptExpr(): Illegal base type");

  
  // Limit the alignment to that of the result type.
  LValue LV;
  if (!ArrayAlignment.isZero()) {
    CharUnits Align = getContext().getTypeAlignInChars(T);
    ArrayAlignment = std::min(Align, ArrayAlignment);
    LV = MakeAddrLValue(Address, T, ArrayAlignment);
  } else {
    LV = MakeNaturalAlignAddrLValue(Address, T);
  }

  LV.getQuals().setAddressSpace(E->getBase()->getType().getAddressSpace());

  if (getContext().getLangOpts().ObjC1 &&
      getContext().getLangOpts().getGC() != LangOptions::NonGC) {
    LV.setNonGC(!E->isOBJCGCCandidate(getContext()));
    setObjCGCLValueClass(getContext(), E, LV);
  }
  return LV;
}

static
llvm::Constant *GenerateConstantVector(CGBuilderTy &Builder,
                                       SmallVector<unsigned, 4> &Elts) {
  SmallVector<llvm::Constant*, 4> CElts;
  for (unsigned i = 0, e = Elts.size(); i != e; ++i)
    CElts.push_back(Builder.getInt32(Elts[i]));

  return llvm::ConstantVector::get(CElts);
}

LValue CodeGenFunction::
EmitExtVectorElementExpr(const ExtVectorElementExpr *E) {
  // Emit the base vector as an l-value.
  LValue Base;

  // ExtVectorElementExpr's base can either be a vector or pointer to vector.
  if (E->isArrow()) {
    // If it is a pointer to a vector, emit the address and form an lvalue with
    // it.
    llvm::Value *Ptr = EmitScalarExpr(E->getBase());
    const PointerType *PT = E->getBase()->getType()->getAs<PointerType>();
    Base = MakeAddrLValue(Ptr, PT->getPointeeType());
    Base.getQuals().removeObjCGCAttr();
  } else if (E->getBase()->isGLValue()) {
    // Otherwise, if the base is an lvalue ( as in the case of foo.x.x),
    // emit the base as an lvalue.
    assert(E->getBase()->getType()->isVectorType());
    Base = EmitLValue(E->getBase());
  } else {
    // Otherwise, the base is a normal rvalue (as in (V+V).x), emit it as such.
    assert(E->getBase()->getType()->isVectorType() &&
           "Result must be a vector");
    llvm::Value *Vec = EmitScalarExpr(E->getBase());
    
    // Store the vector to memory (because LValue wants an address).
    llvm::Value *VecMem = CreateMemTemp(E->getBase()->getType());
    Builder.CreateStore(Vec, VecMem);
    Base = MakeAddrLValue(VecMem, E->getBase()->getType());
  }

  QualType type =
    E->getType().withCVRQualifiers(Base.getQuals().getCVRQualifiers());
  
  // Encode the element access list into a vector of unsigned indices.
  SmallVector<unsigned, 4> Indices;
  E->getEncodedElementAccess(Indices);

  if (Base.isSimple()) {
    llvm::Constant *CV = GenerateConstantVector(Builder, Indices);
    return LValue::MakeExtVectorElt(Base.getAddress(), CV, type,
                                    Base.getAlignment());
  }
  assert(Base.isExtVectorElt() && "Can only subscript lvalue vec elts here!");

  llvm::Constant *BaseElts = Base.getExtVectorElts();
  SmallVector<llvm::Constant *, 4> CElts;

  for (unsigned i = 0, e = Indices.size(); i != e; ++i)
    CElts.push_back(BaseElts->getAggregateElement(Indices[i]));
  llvm::Constant *CV = llvm::ConstantVector::get(CElts);
  return LValue::MakeExtVectorElt(Base.getExtVectorAddr(), CV, type,
                                  Base.getAlignment());
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
    LV.setNonGC(isNonGC);
    setObjCGCLValueClass(getContext(), E, LV);
    return LV;
  }
  
  if (VarDecl *VD = dyn_cast<VarDecl>(ND))
    return EmitGlobalVarDeclLValue(*this, E, VD);

  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(ND))
    return EmitFunctionDeclLValue(*this, E, FD);

  llvm_unreachable("Unhandled member declaration!");
}

LValue CodeGenFunction::EmitLValueForBitfield(llvm::Value *BaseValue,
                                              const FieldDecl *Field,
                                              unsigned CVRQualifiers) {
  const CGRecordLayout &RL =
    CGM.getTypes().getCGRecordLayout(Field->getParent());
  const CGBitFieldInfo &Info = RL.getBitFieldInfo(Field);
  return LValue::MakeBitfield(BaseValue, Info,
                          Field->getType().withCVRQualifiers(CVRQualifiers));
}

/// EmitLValueForAnonRecordField - Given that the field is a member of
/// an anonymous struct or union buried inside a record, and given
/// that the base value is a pointer to the enclosing record, derive
/// an lvalue for the ultimate field.
LValue CodeGenFunction::EmitLValueForAnonRecordField(llvm::Value *BaseValue,
                                             const IndirectFieldDecl *Field,
                                                     unsigned CVRQualifiers) {
  IndirectFieldDecl::chain_iterator I = Field->chain_begin(),
    IEnd = Field->chain_end();
  while (true) {
    LValue LV = EmitLValueForField(BaseValue, cast<FieldDecl>(*I),
                                   CVRQualifiers);
    if (++I == IEnd) return LV;

    assert(LV.isSimple());
    BaseValue = LV.getAddress();
    CVRQualifiers |= LV.getVRQualifiers();
  }
}

LValue CodeGenFunction::EmitLValueForField(llvm::Value *baseAddr,
                                           const FieldDecl *field,
                                           unsigned cvr) {
  if (field->isBitField())
    return EmitLValueForBitfield(baseAddr, field, cvr);

  const RecordDecl *rec = field->getParent();
  QualType type = field->getType();
  CharUnits alignment = getContext().getDeclAlign(field);

  bool mayAlias = rec->hasAttr<MayAliasAttr>();

  llvm::Value *addr = baseAddr;
  if (rec->isUnion()) {
    // For unions, there is no pointer adjustment.
    assert(!type->isReferenceType() && "union has reference member");
  } else {
    // For structs, we GEP to the field that the record layout suggests.
    unsigned idx = CGM.getTypes().getCGRecordLayout(rec).getLLVMFieldNo(field);
    addr = Builder.CreateStructGEP(addr, idx, field->getName());

    // If this is a reference field, load the reference right now.
    if (const ReferenceType *refType = type->getAs<ReferenceType>()) {
      llvm::LoadInst *load = Builder.CreateLoad(addr, "ref");
      if (cvr & Qualifiers::Volatile) load->setVolatile(true);
      load->setAlignment(alignment.getQuantity());

      if (CGM.shouldUseTBAA()) {
        llvm::MDNode *tbaa;
        if (mayAlias)
          tbaa = CGM.getTBAAInfo(getContext().CharTy);
        else
          tbaa = CGM.getTBAAInfo(type);
        CGM.DecorateInstruction(load, tbaa);
      }

      addr = load;
      mayAlias = false;
      type = refType->getPointeeType();
      if (type->isIncompleteType())
        alignment = CharUnits();
      else
        alignment = getContext().getTypeAlignInChars(type);
      cvr = 0; // qualifiers don't recursively apply to referencee
    }
  }
  
  // Make sure that the address is pointing to the right type.  This is critical
  // for both unions and structs.  A union needs a bitcast, a struct element
  // will need a bitcast if the LLVM type laid out doesn't match the desired
  // type.
  addr = EmitBitCastOfLValueToProperType(*this, addr,
                                         CGM.getTypes().ConvertTypeForMem(type),
                                         field->getName());

  if (field->hasAttr<AnnotateAttr>())
    addr = EmitFieldAnnotations(field, addr);

  LValue LV = MakeAddrLValue(addr, type, alignment);
  LV.getQuals().addCVRQualifiers(cvr);

  // __weak attribute on a field is ignored.
  if (LV.getQuals().getObjCGCAttr() == Qualifiers::Weak)
    LV.getQuals().removeObjCGCAttr();

  // Fields of may_alias structs act like 'char' for TBAA purposes.
  // FIXME: this should get propagated down through anonymous structs
  // and unions.
  if (mayAlias && LV.getTBAAInfo())
    LV.setTBAAInfo(CGM.getTBAAInfo(getContext().CharTy));

  return LV;
}

LValue 
CodeGenFunction::EmitLValueForFieldInitialization(llvm::Value *BaseValue, 
                                                  const FieldDecl *Field,
                                                  unsigned CVRQualifiers) {
  QualType FieldType = Field->getType();
  
  if (!FieldType->isReferenceType())
    return EmitLValueForField(BaseValue, Field, CVRQualifiers);

  const CGRecordLayout &RL =
    CGM.getTypes().getCGRecordLayout(Field->getParent());
  unsigned idx = RL.getLLVMFieldNo(Field);
  llvm::Value *V = Builder.CreateStructGEP(BaseValue, idx);
  assert(!FieldType.getObjCGCAttr() && "fields cannot have GC attrs");

  
  // Make sure that the address is pointing to the right type.  This is critical
  // for both unions and structs.  A union needs a bitcast, a struct element
  // will need a bitcast if the LLVM type laid out doesn't match the desired
  // type.
  llvm::Type *llvmType = ConvertTypeForMem(FieldType);
  unsigned AS = cast<llvm::PointerType>(V->getType())->getAddressSpace();
  V = Builder.CreateBitCast(V, llvmType->getPointerTo(AS));
  
  CharUnits Alignment = getContext().getDeclAlign(Field);
  return MakeAddrLValue(V, FieldType, Alignment);
}

LValue CodeGenFunction::EmitCompoundLiteralLValue(const CompoundLiteralExpr *E){
  if (E->isFileScope()) {
    llvm::Value *GlobalPtr = CGM.GetAddrOfConstantCompoundLiteral(E);
    return MakeAddrLValue(GlobalPtr, E->getType());
  }

  llvm::Value *DeclPtr = CreateMemTemp(E->getType(), ".compoundliteral");
  const Expr *InitExpr = E->getInitializer();
  LValue Result = MakeAddrLValue(DeclPtr, E->getType());

  EmitAnyExprToMem(InitExpr, DeclPtr, E->getType().getQualifiers(),
                   /*Init*/ true);

  return Result;
}

LValue CodeGenFunction::
EmitConditionalOperatorLValue(const AbstractConditionalOperator *expr) {
  if (!expr->isGLValue()) {
    // ?: here should be an aggregate.
    assert((hasAggregateLLVMType(expr->getType()) &&
            !expr->getType()->isAnyComplexType()) &&
           "Unexpected conditional operator!");
    return EmitAggExprToLValue(expr);
  }

  OpaqueValueMapping binding(*this, expr);

  const Expr *condExpr = expr->getCond();
  bool CondExprBool;
  if (ConstantFoldsToSimpleInteger(condExpr, CondExprBool)) {
    const Expr *live = expr->getTrueExpr(), *dead = expr->getFalseExpr();
    if (!CondExprBool) std::swap(live, dead);

    if (!ContainsLabel(dead))
      return EmitLValue(live);
  }

  llvm::BasicBlock *lhsBlock = createBasicBlock("cond.true");
  llvm::BasicBlock *rhsBlock = createBasicBlock("cond.false");
  llvm::BasicBlock *contBlock = createBasicBlock("cond.end");

  ConditionalEvaluation eval(*this);
  EmitBranchOnBoolExpr(condExpr, lhsBlock, rhsBlock);
    
  // Any temporaries created here are conditional.
  EmitBlock(lhsBlock);
  eval.begin(*this);
  LValue lhs = EmitLValue(expr->getTrueExpr());
  eval.end(*this);
    
  if (!lhs.isSimple())
    return EmitUnsupportedLValue(expr, "conditional operator");

  lhsBlock = Builder.GetInsertBlock();
  Builder.CreateBr(contBlock);
    
  // Any temporaries created here are conditional.
  EmitBlock(rhsBlock);
  eval.begin(*this);
  LValue rhs = EmitLValue(expr->getFalseExpr());
  eval.end(*this);
  if (!rhs.isSimple())
    return EmitUnsupportedLValue(expr, "conditional operator");
  rhsBlock = Builder.GetInsertBlock();

  EmitBlock(contBlock);

  llvm::PHINode *phi = Builder.CreatePHI(lhs.getAddress()->getType(), 2,
                                         "cond-lvalue");
  phi->addIncoming(lhs.getAddress(), lhsBlock);
  phi->addIncoming(rhs.getAddress(), rhsBlock);
  return MakeAddrLValue(phi, expr->getType());
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
  case CK_ToVoid:
    return EmitUnsupportedLValue(E, "unexpected cast lvalue");

  case CK_Dependent:
    llvm_unreachable("dependent cast kind in IR gen!");
 
  // These two casts are currently treated as no-ops, although they could
  // potentially be real operations depending on the target's ABI.
  case CK_NonAtomicToAtomic:
  case CK_AtomicToNonAtomic:

  case CK_NoOp:
  case CK_LValueToRValue:
    if (!E->getSubExpr()->Classify(getContext()).isPRValue() 
        || E->getType()->isRecordType())
      return EmitLValue(E->getSubExpr());
    // Fall through to synthesize a temporary.

  case CK_BitCast:
  case CK_ArrayToPointerDecay:
  case CK_FunctionToPointerDecay:
  case CK_NullToMemberPointer:
  case CK_NullToPointer:
  case CK_IntegralToPointer:
  case CK_PointerToIntegral:
  case CK_PointerToBoolean:
  case CK_VectorSplat:
  case CK_IntegralCast:
  case CK_IntegralToBoolean:
  case CK_IntegralToFloating:
  case CK_FloatingToIntegral:
  case CK_FloatingToBoolean:
  case CK_FloatingCast:
  case CK_FloatingRealToComplex:
  case CK_FloatingComplexToReal:
  case CK_FloatingComplexToBoolean:
  case CK_FloatingComplexCast:
  case CK_FloatingComplexToIntegralComplex:
  case CK_IntegralRealToComplex:
  case CK_IntegralComplexToReal:
  case CK_IntegralComplexToBoolean:
  case CK_IntegralComplexCast:
  case CK_IntegralComplexToFloatingComplex:
  case CK_DerivedToBaseMemberPointer:
  case CK_BaseToDerivedMemberPointer:
  case CK_MemberPointerToBoolean:
  case CK_ReinterpretMemberPointer:
  case CK_AnyPointerToBlockPointerCast:
  case CK_ARCProduceObject:
  case CK_ARCConsumeObject:
  case CK_ARCReclaimReturnedObject:
  case CK_ARCExtendBlockObject: 
  case CK_CopyAndAutoreleaseBlockObject: {
    // These casts only produce lvalues when we're binding a reference to a 
    // temporary realized from a (converted) pure rvalue. Emit the expression
    // as a value, copy it into a temporary, and return an lvalue referring to
    // that temporary.
    llvm::Value *V = CreateMemTemp(E->getType(), "ref.temp");
    EmitAnyExprToMem(E, V, E->getType().getQualifiers(), false);
    return MakeAddrLValue(V, E->getType());
  }

  case CK_Dynamic: {
    LValue LV = EmitLValue(E->getSubExpr());
    llvm::Value *V = LV.getAddress();
    const CXXDynamicCastExpr *DCE = cast<CXXDynamicCastExpr>(E);
    return MakeAddrLValue(EmitDynamicCast(V, DCE), E->getType());
  }

  case CK_ConstructorConversion:
  case CK_UserDefinedConversion:
  case CK_CPointerToObjCPointerCast:
  case CK_BlockPointerToObjCPointerCast:
    return EmitLValue(E->getSubExpr());
  
  case CK_UncheckedDerivedToBase:
  case CK_DerivedToBase: {
    const RecordType *DerivedClassTy = 
      E->getSubExpr()->getType()->getAs<RecordType>();
    CXXRecordDecl *DerivedClassDecl = 
      cast<CXXRecordDecl>(DerivedClassTy->getDecl());
    
    LValue LV = EmitLValue(E->getSubExpr());
    llvm::Value *This = LV.getAddress();
    
    // Perform the derived-to-base conversion
    llvm::Value *Base = 
      GetAddressOfBaseClass(This, DerivedClassDecl, 
                            E->path_begin(), E->path_end(),
                            /*NullCheckValue=*/false);
    
    return MakeAddrLValue(Base, E->getType());
  }
  case CK_ToUnion:
    return EmitAggExprToLValue(E);
  case CK_BaseToDerived: {
    const RecordType *DerivedClassTy = E->getType()->getAs<RecordType>();
    CXXRecordDecl *DerivedClassDecl = 
      cast<CXXRecordDecl>(DerivedClassTy->getDecl());
    
    LValue LV = EmitLValue(E->getSubExpr());
    
    // Perform the base-to-derived conversion
    llvm::Value *Derived = 
      GetAddressOfDerivedClass(LV.getAddress(), DerivedClassDecl, 
                               E->path_begin(), E->path_end(),
                               /*NullCheckValue=*/false);
    
    return MakeAddrLValue(Derived, E->getType());
  }
  case CK_LValueBitCast: {
    // This must be a reinterpret_cast (or c-style equivalent).
    const ExplicitCastExpr *CE = cast<ExplicitCastExpr>(E);
    
    LValue LV = EmitLValue(E->getSubExpr());
    llvm::Value *V = Builder.CreateBitCast(LV.getAddress(),
                                           ConvertType(CE->getTypeAsWritten()));
    return MakeAddrLValue(V, E->getType());
  }
  case CK_ObjCObjectLValueCast: {
    LValue LV = EmitLValue(E->getSubExpr());
    QualType ToType = getContext().getLValueReferenceType(E->getType());
    llvm::Value *V = Builder.CreateBitCast(LV.getAddress(), 
                                           ConvertType(ToType));
    return MakeAddrLValue(V, E->getType());
  }
  }
  
  llvm_unreachable("Unhandled lvalue cast kind?");
}

LValue CodeGenFunction::EmitNullInitializationLValue(
                                              const CXXScalarValueInitExpr *E) {
  QualType Ty = E->getType();
  LValue LV = MakeAddrLValue(CreateMemTemp(Ty), Ty);
  EmitNullInitialization(LV.getAddress(), Ty);
  return LV;
}

LValue CodeGenFunction::EmitOpaqueValueLValue(const OpaqueValueExpr *e) {
  assert(OpaqueValueMappingData::shouldBindAsLValue(e));
  return getOpaqueLValueMapping(e);
}

LValue CodeGenFunction::EmitMaterializeTemporaryExpr(
                                           const MaterializeTemporaryExpr *E) {
  RValue RV = EmitReferenceBindingToExpr(E, /*InitializedDecl=*/0);
  return MakeAddrLValue(RV.getScalarVal(), E->getType());
}


//===--------------------------------------------------------------------===//
//                             Expression Emission
//===--------------------------------------------------------------------===//

RValue CodeGenFunction::EmitCallExpr(const CallExpr *E, 
                                     ReturnValueSlot ReturnValue) {
  if (CGDebugInfo *DI = getDebugInfo())
    DI->EmitLocation(Builder, E->getLocStart());

  // Builtins never have block type.
  if (E->getCallee()->getType()->isBlockPointerType())
    return EmitBlockCallExpr(E, ReturnValue);

  if (const CXXMemberCallExpr *CE = dyn_cast<CXXMemberCallExpr>(E))
    return EmitCXXMemberCallExpr(CE, ReturnValue);

  if (const CUDAKernelCallExpr *CE = dyn_cast<CUDAKernelCallExpr>(E))
    return EmitCUDAKernelCallExpr(CE, ReturnValue);

  const Decl *TargetDecl = E->getCalleeDecl();
  if (const FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(TargetDecl)) {
    if (unsigned builtinID = FD->getBuiltinID())
      return EmitBuiltinExpr(FD, builtinID, E);
  }

  if (const CXXOperatorCallExpr *CE = dyn_cast<CXXOperatorCallExpr>(E))
    if (const CXXMethodDecl *MD = dyn_cast_or_null<CXXMethodDecl>(TargetDecl))
      return EmitCXXOperatorMemberCallExpr(CE, MD, ReturnValue);

  if (const CXXPseudoDestructorExpr *PseudoDtor 
          = dyn_cast<CXXPseudoDestructorExpr>(E->getCallee()->IgnoreParens())) {
    QualType DestroyedType = PseudoDtor->getDestroyedType();
    if (getContext().getLangOpts().ObjCAutoRefCount &&
        DestroyedType->isObjCLifetimeType() &&
        (DestroyedType.getObjCLifetime() == Qualifiers::OCL_Strong ||
         DestroyedType.getObjCLifetime() == Qualifiers::OCL_Weak)) {
      // Automatic Reference Counting:
      //   If the pseudo-expression names a retainable object with weak or
      //   strong lifetime, the object shall be released.
      Expr *BaseExpr = PseudoDtor->getBase();
      llvm::Value *BaseValue = NULL;
      Qualifiers BaseQuals;
      
      // If this is s.x, emit s as an lvalue. If it is s->x, emit s as a scalar.
      if (PseudoDtor->isArrow()) {
        BaseValue = EmitScalarExpr(BaseExpr);
        const PointerType *PTy = BaseExpr->getType()->getAs<PointerType>();
        BaseQuals = PTy->getPointeeType().getQualifiers();
      } else {
        LValue BaseLV = EmitLValue(BaseExpr);
        BaseValue = BaseLV.getAddress();
        QualType BaseTy = BaseExpr->getType();
        BaseQuals = BaseTy.getQualifiers();
      }
          
      switch (PseudoDtor->getDestroyedType().getObjCLifetime()) {
      case Qualifiers::OCL_None:
      case Qualifiers::OCL_ExplicitNone:
      case Qualifiers::OCL_Autoreleasing:
        break;
        
      case Qualifiers::OCL_Strong:
        EmitARCRelease(Builder.CreateLoad(BaseValue, 
                          PseudoDtor->getDestroyedType().isVolatileQualified()),
                       /*precise*/ true);
        break;

      case Qualifiers::OCL_Weak:
        EmitARCDestroyWeak(BaseValue);
        break;
      }
    } else {
      // C++ [expr.pseudo]p1:
      //   The result shall only be used as the operand for the function call
      //   operator (), and the result of such a call has type void. The only
      //   effect is the evaluation of the postfix-expression before the dot or
      //   arrow.      
      EmitScalarExpr(E->getCallee());
    }
    
    return RValue::get(0);
  }

  llvm::Value *Callee = EmitScalarExpr(E->getCallee());
  return EmitCall(E->getCallee()->getType(), Callee, ReturnValue,
                  E->arg_begin(), E->arg_end(), TargetDecl);
}

LValue CodeGenFunction::EmitBinaryOperatorLValue(const BinaryOperator *E) {
  // Comma expressions just emit their LHS then their RHS as an l-value.
  if (E->getOpcode() == BO_Comma) {
    EmitIgnoredExpr(E->getLHS());
    EnsureInsertPoint();
    return EmitLValue(E->getRHS());
  }

  if (E->getOpcode() == BO_PtrMemD ||
      E->getOpcode() == BO_PtrMemI)
    return EmitPointerToDataMemberBinaryExpr(E);

  assert(E->getOpcode() == BO_Assign && "unexpected binary l-value");

  // Note that in all of these cases, __block variables need the RHS
  // evaluated first just in case the variable gets moved by the RHS.
  
  if (!hasAggregateLLVMType(E->getType())) {
    switch (E->getLHS()->getType().getObjCLifetime()) {
    case Qualifiers::OCL_Strong:
      return EmitARCStoreStrong(E, /*ignored*/ false).first;

    case Qualifiers::OCL_Autoreleasing:
      return EmitARCStoreAutoreleasing(E).first;

    // No reason to do any of these differently.
    case Qualifiers::OCL_None:
    case Qualifiers::OCL_ExplicitNone:
    case Qualifiers::OCL_Weak:
      break;
    }

    RValue RV = EmitAnyExpr(E->getRHS());
    LValue LV = EmitLValue(E->getLHS());
    EmitStoreThroughLValue(RV, LV);
    return LV;
  }

  if (E->getType()->isAnyComplexType())
    return EmitComplexAssignmentLValue(E);

  return EmitAggExprToLValue(E);
}

LValue CodeGenFunction::EmitCallExprLValue(const CallExpr *E) {
  RValue RV = EmitCallExpr(E);

  if (!RV.isScalar())
    return MakeAddrLValue(RV.getAggregateAddr(), E->getType());
    
  assert(E->getCallReturnType()->isReferenceType() &&
         "Can't have a scalar return unless the return type is a "
         "reference type!");

  return MakeAddrLValue(RV.getScalarVal(), E->getType());
}

LValue CodeGenFunction::EmitVAArgExprLValue(const VAArgExpr *E) {
  // FIXME: This shouldn't require another copy.
  return EmitAggExprToLValue(E);
}

LValue CodeGenFunction::EmitCXXConstructLValue(const CXXConstructExpr *E) {
  assert(E->getType()->getAsCXXRecordDecl()->hasTrivialDestructor()
         && "binding l-value to type which needs a temporary");
  AggValueSlot Slot = CreateAggTemp(E->getType());
  EmitCXXConstructExpr(E, Slot);
  return MakeAddrLValue(Slot.getAddr(), E->getType());
}

LValue
CodeGenFunction::EmitCXXTypeidLValue(const CXXTypeidExpr *E) {
  return MakeAddrLValue(EmitCXXTypeidExpr(E), E->getType());
}

LValue
CodeGenFunction::EmitCXXBindTemporaryLValue(const CXXBindTemporaryExpr *E) {
  AggValueSlot Slot = CreateAggTemp(E->getType(), "temp.lvalue");
  Slot.setExternallyDestructed();
  EmitAggExpr(E->getSubExpr(), Slot);
  EmitCXXTemporary(E->getTemporary(), E->getType(), Slot.getAddr());
  return MakeAddrLValue(Slot.getAddr(), E->getType());
}

LValue
CodeGenFunction::EmitLambdaLValue(const LambdaExpr *E) {
  AggValueSlot Slot = CreateAggTemp(E->getType(), "temp.lvalue");
  EmitLambdaExpr(E, Slot);
  return MakeAddrLValue(Slot.getAddr(), E->getType());
}

LValue CodeGenFunction::EmitObjCMessageExprLValue(const ObjCMessageExpr *E) {
  RValue RV = EmitObjCMessageExpr(E);
  
  if (!RV.isScalar())
    return MakeAddrLValue(RV.getAggregateAddr(), E->getType());
  
  assert(E->getMethodDecl()->getResultType()->isReferenceType() &&
         "Can't have a scalar return unless the return type is a "
         "reference type!");
  
  return MakeAddrLValue(RV.getScalarVal(), E->getType());
}

LValue CodeGenFunction::EmitObjCSelectorLValue(const ObjCSelectorExpr *E) {
  llvm::Value *V = 
    CGM.getObjCRuntime().GetSelector(Builder, E->getSelector(), true);
  return MakeAddrLValue(V, E->getType());
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

LValue CodeGenFunction::EmitStmtExprLValue(const StmtExpr *E) {
  // Can only get l-value for message expression returning aggregate type
  RValue RV = EmitAnyExprToTemp(E);
  return MakeAddrLValue(RV.getAggregateAddr(), E->getType());
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

  CallArgList Args;
  EmitCallArgs(Args, dyn_cast<FunctionProtoType>(FnType), ArgBeg, ArgEnd);

  const CGFunctionInfo &FnInfo =
    CGM.getTypes().arrangeFunctionCall(Args, FnType);

  // C99 6.5.2.2p6:
  //   If the expression that denotes the called function has a type
  //   that does not include a prototype, [the default argument
  //   promotions are performed]. If the number of arguments does not
  //   equal the number of parameters, the behavior is undefined. If
  //   the function is defined with a type that includes a prototype,
  //   and either the prototype ends with an ellipsis (, ...) or the
  //   types of the arguments after promotion are not compatible with
  //   the types of the parameters, the behavior is undefined. If the
  //   function is defined with a type that does not include a
  //   prototype, and the types of the arguments after promotion are
  //   not compatible with those of the parameters after promotion,
  //   the behavior is undefined [except in some trivial cases].
  // That is, in the general case, we should assume that a call
  // through an unprototyped function type works like a *non-variadic*
  // call.  The way we make this work is to cast to the exact type
  // of the promoted arguments.
  if (isa<FunctionNoProtoType>(FnType) && !FnInfo.isVariadic()) {
    llvm::Type *CalleeTy = getTypes().GetFunctionType(FnInfo);
    CalleeTy = CalleeTy->getPointerTo();
    Callee = Builder.CreateBitCast(Callee, CalleeTy, "callee.knr.cast");
  }

  return EmitCall(FnInfo, Callee, ReturnValue, Args, TargetDecl);
}

LValue CodeGenFunction::
EmitPointerToDataMemberBinaryExpr(const BinaryOperator *E) {
  llvm::Value *BaseV;
  if (E->getOpcode() == BO_PtrMemI)
    BaseV = EmitScalarExpr(E->getLHS());
  else
    BaseV = EmitLValue(E->getLHS()).getAddress();

  llvm::Value *OffsetV = EmitScalarExpr(E->getRHS());

  const MemberPointerType *MPT
    = E->getRHS()->getType()->getAs<MemberPointerType>();

  llvm::Value *AddV =
    CGM.getCXXABI().EmitMemberDataPointerAddress(*this, BaseV, OffsetV, MPT);

  return MakeAddrLValue(AddV, MPT->getPointeeType());
}

static void
EmitAtomicOp(CodeGenFunction &CGF, AtomicExpr *E, llvm::Value *Dest,
             llvm::Value *Ptr, llvm::Value *Val1, llvm::Value *Val2,
             uint64_t Size, unsigned Align, llvm::AtomicOrdering Order) {
  if (E->isCmpXChg()) {
    // Note that cmpxchg only supports specifying one ordering and
    // doesn't support weak cmpxchg, at least at the moment.
    llvm::LoadInst *LoadVal1 = CGF.Builder.CreateLoad(Val1);
    LoadVal1->setAlignment(Align);
    llvm::LoadInst *LoadVal2 = CGF.Builder.CreateLoad(Val2);
    LoadVal2->setAlignment(Align);
    llvm::AtomicCmpXchgInst *CXI =
        CGF.Builder.CreateAtomicCmpXchg(Ptr, LoadVal1, LoadVal2, Order);
    CXI->setVolatile(E->isVolatile());
    llvm::StoreInst *StoreVal1 = CGF.Builder.CreateStore(CXI, Val1);
    StoreVal1->setAlignment(Align);
    llvm::Value *Cmp = CGF.Builder.CreateICmpEQ(CXI, LoadVal1);
    CGF.EmitStoreOfScalar(Cmp, CGF.MakeAddrLValue(Dest, E->getType()));
    return;
  }

  if (E->getOp() == AtomicExpr::Load) {
    llvm::LoadInst *Load = CGF.Builder.CreateLoad(Ptr);
    Load->setAtomic(Order);
    Load->setAlignment(Size);
    Load->setVolatile(E->isVolatile());
    llvm::StoreInst *StoreDest = CGF.Builder.CreateStore(Load, Dest);
    StoreDest->setAlignment(Align);
    return;
  }

  if (E->getOp() == AtomicExpr::Store) {
    assert(!Dest && "Store does not return a value");
    llvm::LoadInst *LoadVal1 = CGF.Builder.CreateLoad(Val1);
    LoadVal1->setAlignment(Align);
    llvm::StoreInst *Store = CGF.Builder.CreateStore(LoadVal1, Ptr);
    Store->setAtomic(Order);
    Store->setAlignment(Size);
    Store->setVolatile(E->isVolatile());
    return;
  }

  llvm::AtomicRMWInst::BinOp Op = llvm::AtomicRMWInst::Add;
  switch (E->getOp()) {
    case AtomicExpr::CmpXchgWeak:
    case AtomicExpr::CmpXchgStrong:
    case AtomicExpr::Store:
    case AtomicExpr::Init:
    case AtomicExpr::Load:  assert(0 && "Already handled!");
    case AtomicExpr::Add:   Op = llvm::AtomicRMWInst::Add;  break;
    case AtomicExpr::Sub:   Op = llvm::AtomicRMWInst::Sub;  break;
    case AtomicExpr::And:   Op = llvm::AtomicRMWInst::And;  break;
    case AtomicExpr::Or:    Op = llvm::AtomicRMWInst::Or;   break;
    case AtomicExpr::Xor:   Op = llvm::AtomicRMWInst::Xor;  break;
    case AtomicExpr::Xchg:  Op = llvm::AtomicRMWInst::Xchg; break;
  }
  llvm::LoadInst *LoadVal1 = CGF.Builder.CreateLoad(Val1);
  LoadVal1->setAlignment(Align);
  llvm::AtomicRMWInst *RMWI =
      CGF.Builder.CreateAtomicRMW(Op, Ptr, LoadVal1, Order);
  RMWI->setVolatile(E->isVolatile());
  llvm::StoreInst *StoreDest = CGF.Builder.CreateStore(RMWI, Dest);
  StoreDest->setAlignment(Align);
}

// This function emits any expression (scalar, complex, or aggregate)
// into a temporary alloca.
static llvm::Value *
EmitValToTemp(CodeGenFunction &CGF, Expr *E) {
  llvm::Value *DeclPtr = CGF.CreateMemTemp(E->getType(), ".atomictmp");
  CGF.EmitAnyExprToMem(E, DeclPtr, E->getType().getQualifiers(),
                       /*Init*/ true);
  return DeclPtr;
}

static RValue ConvertTempToRValue(CodeGenFunction &CGF, QualType Ty,
                                  llvm::Value *Dest) {
  if (Ty->isAnyComplexType())
    return RValue::getComplex(CGF.LoadComplexFromAddr(Dest, false));
  if (CGF.hasAggregateLLVMType(Ty))
    return RValue::getAggregate(Dest);
  return RValue::get(CGF.EmitLoadOfScalar(CGF.MakeAddrLValue(Dest, Ty)));
}

RValue CodeGenFunction::EmitAtomicExpr(AtomicExpr *E, llvm::Value *Dest) {
  QualType AtomicTy = E->getPtr()->getType()->getPointeeType();
  QualType MemTy = AtomicTy->getAs<AtomicType>()->getValueType();
  CharUnits sizeChars = getContext().getTypeSizeInChars(AtomicTy);
  uint64_t Size = sizeChars.getQuantity();
  CharUnits alignChars = getContext().getTypeAlignInChars(AtomicTy);
  unsigned Align = alignChars.getQuantity();
  unsigned MaxInlineWidth =
      getContext().getTargetInfo().getMaxAtomicInlineWidth();
  bool UseLibcall = (Size != Align || Size > MaxInlineWidth);



  llvm::Value *Ptr, *Order, *OrderFail = 0, *Val1 = 0, *Val2 = 0;
  Ptr = EmitScalarExpr(E->getPtr());

  if (E->getOp() == AtomicExpr::Init) {
    assert(!Dest && "Init does not return a value");
    Val1 = EmitScalarExpr(E->getVal1());
    llvm::StoreInst *Store = Builder.CreateStore(Val1, Ptr);
    Store->setAlignment(Size);
    Store->setVolatile(E->isVolatile());
    return RValue::get(0);
  }

  Order = EmitScalarExpr(E->getOrder());
  if (E->isCmpXChg()) {
    Val1 = EmitScalarExpr(E->getVal1());
    Val2 = EmitValToTemp(*this, E->getVal2());
    OrderFail = EmitScalarExpr(E->getOrderFail());
    (void)OrderFail; // OrderFail is unused at the moment
  } else if ((E->getOp() == AtomicExpr::Add || E->getOp() == AtomicExpr::Sub) &&
             MemTy->isPointerType()) {
    // For pointers, we're required to do a bit of math: adding 1 to an int*
    // is not the same as adding 1 to a uintptr_t.
    QualType Val1Ty = E->getVal1()->getType();
    llvm::Value *Val1Scalar = EmitScalarExpr(E->getVal1());
    CharUnits PointeeIncAmt =
        getContext().getTypeSizeInChars(MemTy->getPointeeType());
    Val1Scalar = Builder.CreateMul(Val1Scalar, CGM.getSize(PointeeIncAmt));
    Val1 = CreateMemTemp(Val1Ty, ".atomictmp");
    EmitStoreOfScalar(Val1Scalar, MakeAddrLValue(Val1, Val1Ty));
  } else if (E->getOp() != AtomicExpr::Load) {
    Val1 = EmitValToTemp(*this, E->getVal1());
  }

  if (E->getOp() != AtomicExpr::Store && !Dest)
    Dest = CreateMemTemp(E->getType(), ".atomicdst");

  if (UseLibcall) {
    // FIXME: Finalize what the libcalls are actually supposed to look like.
    // See also http://gcc.gnu.org/wiki/Atomic/GCCMM/LIbrary .
    return EmitUnsupportedRValue(E, "atomic library call");
  }
#if 0
  if (UseLibcall) {
    const char* LibCallName;
    switch (E->getOp()) {
    case AtomicExpr::CmpXchgWeak:
      LibCallName = "__atomic_compare_exchange_generic"; break;
    case AtomicExpr::CmpXchgStrong:
      LibCallName = "__atomic_compare_exchange_generic"; break;
    case AtomicExpr::Add:   LibCallName = "__atomic_fetch_add_generic"; break;
    case AtomicExpr::Sub:   LibCallName = "__atomic_fetch_sub_generic"; break;
    case AtomicExpr::And:   LibCallName = "__atomic_fetch_and_generic"; break;
    case AtomicExpr::Or:    LibCallName = "__atomic_fetch_or_generic"; break;
    case AtomicExpr::Xor:   LibCallName = "__atomic_fetch_xor_generic"; break;
    case AtomicExpr::Xchg:  LibCallName = "__atomic_exchange_generic"; break;
    case AtomicExpr::Store: LibCallName = "__atomic_store_generic"; break;
    case AtomicExpr::Load:  LibCallName = "__atomic_load_generic"; break;
    }
    llvm::SmallVector<QualType, 4> Params;
    CallArgList Args;
    QualType RetTy = getContext().VoidTy;
    if (E->getOp() != AtomicExpr::Store && !E->isCmpXChg())
      Args.add(RValue::get(EmitCastToVoidPtr(Dest)),
               getContext().VoidPtrTy);
    Args.add(RValue::get(EmitCastToVoidPtr(Ptr)),
             getContext().VoidPtrTy);
    if (E->getOp() != AtomicExpr::Load)
      Args.add(RValue::get(EmitCastToVoidPtr(Val1)),
               getContext().VoidPtrTy);
    if (E->isCmpXChg()) {
      Args.add(RValue::get(EmitCastToVoidPtr(Val2)),
               getContext().VoidPtrTy);
      RetTy = getContext().IntTy;
    }
    Args.add(RValue::get(llvm::ConstantInt::get(SizeTy, Size)),
             getContext().getSizeType());
    const CGFunctionInfo &FuncInfo =
        CGM.getTypes().arrangeFunctionCall(RetTy, Args, FunctionType::ExtInfo(),
                                           /*variadic*/ false);
    llvm::FunctionType *FTy = CGM.getTypes().GetFunctionType(FuncInfo, false);
    llvm::Constant *Func = CGM.CreateRuntimeFunction(FTy, LibCallName);
    RValue Res = EmitCall(FuncInfo, Func, ReturnValueSlot(), Args);
    if (E->isCmpXChg())
      return Res;
    if (E->getOp() == AtomicExpr::Store)
      return RValue::get(0);
    return ConvertTempToRValue(*this, E->getType(), Dest);
  }
#endif
  llvm::Type *IPtrTy =
      llvm::IntegerType::get(getLLVMContext(), Size * 8)->getPointerTo();
  llvm::Value *OrigDest = Dest;
  Ptr = Builder.CreateBitCast(Ptr, IPtrTy);
  if (Val1) Val1 = Builder.CreateBitCast(Val1, IPtrTy);
  if (Val2) Val2 = Builder.CreateBitCast(Val2, IPtrTy);
  if (Dest && !E->isCmpXChg()) Dest = Builder.CreateBitCast(Dest, IPtrTy);

  if (isa<llvm::ConstantInt>(Order)) {
    int ord = cast<llvm::ConstantInt>(Order)->getZExtValue();
    switch (ord) {
    case 0:  // memory_order_relaxed
      EmitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, Size, Align,
                   llvm::Monotonic);
      break;
    case 1:  // memory_order_consume
    case 2:  // memory_order_acquire
      EmitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, Size, Align,
                   llvm::Acquire);
      break;
    case 3:  // memory_order_release
      EmitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, Size, Align,
                   llvm::Release);
      break;
    case 4:  // memory_order_acq_rel
      EmitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, Size, Align,
                   llvm::AcquireRelease);
      break;
    case 5:  // memory_order_seq_cst
      EmitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, Size, Align,
                   llvm::SequentiallyConsistent);
      break;
    default: // invalid order
      // We should not ever get here normally, but it's hard to
      // enforce that in general.
      break; 
    }
    if (E->getOp() == AtomicExpr::Store || E->getOp() == AtomicExpr::Init)
      return RValue::get(0);
    return ConvertTempToRValue(*this, E->getType(), OrigDest);
  }

  // Long case, when Order isn't obviously constant.

  // Create all the relevant BB's
  llvm::BasicBlock *MonotonicBB = 0, *AcquireBB = 0, *ReleaseBB = 0,
                   *AcqRelBB = 0, *SeqCstBB = 0;
  MonotonicBB = createBasicBlock("monotonic", CurFn);
  if (E->getOp() != AtomicExpr::Store)
    AcquireBB = createBasicBlock("acquire", CurFn);
  if (E->getOp() != AtomicExpr::Load)
    ReleaseBB = createBasicBlock("release", CurFn);
  if (E->getOp() != AtomicExpr::Load && E->getOp() != AtomicExpr::Store)
    AcqRelBB = createBasicBlock("acqrel", CurFn);
  SeqCstBB = createBasicBlock("seqcst", CurFn);
  llvm::BasicBlock *ContBB = createBasicBlock("atomic.continue", CurFn);

  // Create the switch for the split
  // MonotonicBB is arbitrarily chosen as the default case; in practice, this
  // doesn't matter unless someone is crazy enough to use something that
  // doesn't fold to a constant for the ordering.
  Order = Builder.CreateIntCast(Order, Builder.getInt32Ty(), false);
  llvm::SwitchInst *SI = Builder.CreateSwitch(Order, MonotonicBB);

  // Emit all the different atomics
  Builder.SetInsertPoint(MonotonicBB);
  EmitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, Size, Align,
               llvm::Monotonic);
  Builder.CreateBr(ContBB);
  if (E->getOp() != AtomicExpr::Store) {
    Builder.SetInsertPoint(AcquireBB);
    EmitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, Size, Align,
                 llvm::Acquire);
    Builder.CreateBr(ContBB);
    SI->addCase(Builder.getInt32(1), AcquireBB);
    SI->addCase(Builder.getInt32(2), AcquireBB);
  }
  if (E->getOp() != AtomicExpr::Load) {
    Builder.SetInsertPoint(ReleaseBB);
    EmitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, Size, Align,
                 llvm::Release);
    Builder.CreateBr(ContBB);
    SI->addCase(Builder.getInt32(3), ReleaseBB);
  }
  if (E->getOp() != AtomicExpr::Load && E->getOp() != AtomicExpr::Store) {
    Builder.SetInsertPoint(AcqRelBB);
    EmitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, Size, Align,
                 llvm::AcquireRelease);
    Builder.CreateBr(ContBB);
    SI->addCase(Builder.getInt32(4), AcqRelBB);
  }
  Builder.SetInsertPoint(SeqCstBB);
  EmitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, Size, Align,
               llvm::SequentiallyConsistent);
  Builder.CreateBr(ContBB);
  SI->addCase(Builder.getInt32(5), SeqCstBB);

  // Cleanup and return
  Builder.SetInsertPoint(ContBB);
  if (E->getOp() == AtomicExpr::Store)
    return RValue::get(0);
  return ConvertTempToRValue(*this, E->getType(), OrigDest);
}

void CodeGenFunction::SetFPAccuracy(llvm::Value *Val, unsigned AccuracyN,
                                    unsigned AccuracyD) {
  assert(Val->getType()->isFPOrFPVectorTy());
  if (!AccuracyN || !isa<llvm::Instruction>(Val))
    return;

  llvm::Value *Vals[2];
  Vals[0] = llvm::ConstantInt::get(Int32Ty, AccuracyN);
  Vals[1] = llvm::ConstantInt::get(Int32Ty, AccuracyD);
  llvm::MDNode *Node = llvm::MDNode::get(getLLVMContext(), Vals);

  cast<llvm::Instruction>(Val)->setMetadata(llvm::LLVMContext::MD_fpaccuracy,
                                            Node);
}

namespace {
  struct LValueOrRValue {
    LValue LV;
    RValue RV;
  };
}

static LValueOrRValue emitPseudoObjectExpr(CodeGenFunction &CGF,
                                           const PseudoObjectExpr *E,
                                           bool forLValue,
                                           AggValueSlot slot) {
  llvm::SmallVector<CodeGenFunction::OpaqueValueMappingData, 4> opaques;

  // Find the result expression, if any.
  const Expr *resultExpr = E->getResultExpr();
  LValueOrRValue result;

  for (PseudoObjectExpr::const_semantics_iterator
         i = E->semantics_begin(), e = E->semantics_end(); i != e; ++i) {
    const Expr *semantic = *i;

    // If this semantic expression is an opaque value, bind it
    // to the result of its source expression.
    if (const OpaqueValueExpr *ov = dyn_cast<OpaqueValueExpr>(semantic)) {

      // If this is the result expression, we may need to evaluate
      // directly into the slot.
      typedef CodeGenFunction::OpaqueValueMappingData OVMA;
      OVMA opaqueData;
      if (ov == resultExpr && ov->isRValue() && !forLValue &&
          CodeGenFunction::hasAggregateLLVMType(ov->getType()) &&
          !ov->getType()->isAnyComplexType()) {
        CGF.EmitAggExpr(ov->getSourceExpr(), slot);

        LValue LV = CGF.MakeAddrLValue(slot.getAddr(), ov->getType());
        opaqueData = OVMA::bind(CGF, ov, LV);
        result.RV = slot.asRValue();

      // Otherwise, emit as normal.
      } else {
        opaqueData = OVMA::bind(CGF, ov, ov->getSourceExpr());

        // If this is the result, also evaluate the result now.
        if (ov == resultExpr) {
          if (forLValue)
            result.LV = CGF.EmitLValue(ov);
          else
            result.RV = CGF.EmitAnyExpr(ov, slot);
        }
      }

      opaques.push_back(opaqueData);

    // Otherwise, if the expression is the result, evaluate it
    // and remember the result.
    } else if (semantic == resultExpr) {
      if (forLValue)
        result.LV = CGF.EmitLValue(semantic);
      else
        result.RV = CGF.EmitAnyExpr(semantic, slot);

    // Otherwise, evaluate the expression in an ignored context.
    } else {
      CGF.EmitIgnoredExpr(semantic);
    }
  }

  // Unbind all the opaques now.
  for (unsigned i = 0, e = opaques.size(); i != e; ++i)
    opaques[i].unbind(CGF);

  return result;
}

RValue CodeGenFunction::EmitPseudoObjectRValue(const PseudoObjectExpr *E,
                                               AggValueSlot slot) {
  return emitPseudoObjectExpr(*this, E, false, slot).RV;
}

LValue CodeGenFunction::EmitPseudoObjectLValue(const PseudoObjectExpr *E) {
  return emitPseudoObjectExpr(*this, E, true, AggValueSlot::ignored()).LV;
}
