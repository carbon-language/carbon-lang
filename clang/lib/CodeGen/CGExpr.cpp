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
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/Intrinsics.h"
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

llvm::AllocaInst *CodeGenFunction::CreateIRTemp(QualType Ty,
                                                const llvm::Twine &Name) {
  llvm::AllocaInst *Alloc = CreateTempAlloca(ConvertType(Ty), Name);
  // FIXME: Should we prefer the preferred type alignment here?
  CharUnits Align = getContext().getTypeAlignInChars(Ty);
  Alloc->setAlignment(Align.getQuantity());
  return Alloc;
}

llvm::AllocaInst *CodeGenFunction::CreateMemTemp(QualType Ty,
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
  if (E->getType()->isAnyComplexType())
    EmitComplexExprIntoAddr(E, Location, Quals.hasVolatile());
  else if (hasAggregateLLVMType(E->getType()))
    EmitAggExpr(E, AggValueSlot::forAddr(Location, Quals, IsInit));
  else {
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
      llvm::SmallString<256> Name;
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
  // Look through expressions for materialized temporaries (for now).
  if (const MaterializeTemporaryExpr *M 
                                      = dyn_cast<MaterializeTemporaryExpr>(E)) {
    // Objective-C++ ARC:
    //   If we are binding a reference to a temporary that has ownership, we 
    //   need to perform retain/release operations on the temporary.
    if (CGF.getContext().getLangOptions().ObjCAutoRefCount &&        
        E->getType()->isObjCLifetimeType() &&
        (E->getType().getObjCLifetime() == Qualifiers::OCL_Strong ||
         E->getType().getObjCLifetime() == Qualifiers::OCL_Weak ||
         E->getType().getObjCLifetime() == Qualifiers::OCL_Autoreleasing))
      ObjCARCReferenceLifetimeType = E->getType();
    
    E = M->GetTemporaryExpr();
  }

  if (const CXXDefaultArgExpr *DAE = dyn_cast<CXXDefaultArgExpr>(E))
    E = DAE->getExpr();
  
  if (const ExprWithCleanups *TE = dyn_cast<ExprWithCleanups>(E)) {
    CodeGenFunction::RunCleanupsScope Scope(CGF);

    return EmitExprForReferenceBinding(CGF, TE->getSubExpr(), 
                                       ReferenceTemporary, 
                                       ReferenceTemporaryDtor,
                                       ObjCARCReferenceLifetimeType,
                                       InitializedDecl);
  }

  if (const ObjCPropertyRefExpr *PRE = 
      dyn_cast<ObjCPropertyRefExpr>(E->IgnoreParenImpCasts()))
    if (PRE->getGetterResultType()->isReferenceType())
      E = PRE;
    
  RValue RV;
  if (E->isGLValue()) {
    // Emit the expression as an lvalue.
    LValue LV = CGF.EmitLValue(E);
    if (LV.isPropertyRef()) {
      RV = CGF.EmitLoadOfPropertyRefLValue(LV);
      return RV.getScalarVal();
    }
    
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
    
    llvm::SmallVector<SubobjectAdjustment, 2> Adjustments;
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
      AggSlot = AggValueSlot::forAddr(ReferenceTemporary, Qualifiers(),
                                      InitializedDecl != 0);
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
      assert(0 && "Not a reference temporary that needs to be deallocated");
    case Qualifiers::OCL_ExplicitNone:
    case Qualifiers::OCL_Autoreleasing:
      // Nothing to do.
      break;        
        
    case Qualifiers::OCL_Strong: {
      bool precise = VD && VD->hasAttr<ObjCPreciseLifetimeAttr>();
      CleanupKind cleanupKind = getARCCleanupKind();
      // This local is a GCC and MSVC compiler workaround.
      Destroyer *destroyer = precise ? &destroyARCStrongPrecise :
                                       &destroyARCStrongImprecise;
      pushDestroy(cleanupKind, ReferenceTemporary, ObjCARCReferenceLifetimeType,
                  *destroyer, cleanupKind & EHCleanup);
      break;
    }
        
    case Qualifiers::OCL_Weak: {
      // This local is a GCC and MSVC compiler workaround.
      Destroyer *destroyer = &destroyARCWeak;
      // __weak objects always get EH cleanups; otherwise, exceptions
      // could cause really nasty crashes instead of mere leaks.
      pushDestroy(NormalAndEHCleanup, ReferenceTemporary,
                  ObjCARCReferenceLifetimeType, *destroyer, true);
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
  if (isa<llvm::ConstantAggregateZero>(Elts))
    return 0;

  return cast<llvm::ConstantInt>(Elts->getOperand(Idx))->getZExtValue();
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
    return EmitCallExprLValue(cast<CallExpr>(E));
  case Expr::VAArgExprClass:
    return EmitVAArgExprLValue(cast<VAArgExpr>(E));
  case Expr::DeclRefExprClass:
    return EmitDeclRefLValue(cast<DeclRefExpr>(E));
  case Expr::ParenExprClass:return EmitLValue(cast<ParenExpr>(E)->getSubExpr());
  case Expr::GenericSelectionExprClass:
    return EmitLValue(cast<GenericSelectionExpr>(E)->getResultExpr());
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
  case Expr::ExprWithCleanupsClass:
    return EmitExprWithCleanupsLValue(cast<ExprWithCleanups>(E));
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
  case Expr::ObjCPropertyRefExprClass:
    return EmitObjCPropertyRefLValue(cast<ObjCPropertyRefExpr>(E));
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

llvm::Value *CodeGenFunction::EmitLoadOfScalar(LValue lvalue) {
  return EmitLoadOfScalar(lvalue.getAddress(), lvalue.isVolatile(),
                          lvalue.getAlignment(), lvalue.getType(),
                          lvalue.getTBAAInfo());
}

llvm::Value *CodeGenFunction::EmitLoadOfScalar(llvm::Value *Addr, bool Volatile,
                                              unsigned Alignment, QualType Ty,
                                              llvm::MDNode *TBAAInfo) {
  llvm::LoadInst *Load = Builder.CreateLoad(Addr, "tmp");
  if (Volatile)
    Load->setVolatile(true);
  if (Alignment)
    Load->setAlignment(Alignment);
  if (TBAAInfo)
    CGM.DecorateInstruction(Load, TBAAInfo);

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
                                        llvm::MDNode *TBAAInfo) {
  Value = EmitToMemory(Value, Ty);
  
  llvm::StoreInst *Store = Builder.CreateStore(Value, Addr, Volatile);
  if (Alignment)
    Store->setAlignment(Alignment);
  if (TBAAInfo)
    CGM.DecorateInstruction(Store, TBAAInfo);
}

void CodeGenFunction::EmitStoreOfScalar(llvm::Value *value, LValue lvalue) {
  EmitStoreOfScalar(value, lvalue.getAddress(), lvalue.isVolatile(),
                    lvalue.getAlignment(), lvalue.getType(),
                    lvalue.getTBAAInfo());
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
    llvm::Value *Vec = Builder.CreateLoad(LV.getVectorAddr(),
                                          LV.isVolatileQualified(), "tmp");
    return RValue::get(Builder.CreateExtractElement(Vec, LV.getVectorIdx(),
                                                    "vecext"));
  }

  // If this is a reference to a subset of the elements of a vector, either
  // shuffle the input or extract/insert them as appropriate.
  if (LV.isExtVectorElt())
    return EmitLoadOfExtVectorElementLValue(LV);

  if (LV.isBitField())
    return EmitLoadOfBitfieldLValue(LV);

  assert(LV.isPropertyRef() && "Unknown LValue type!");
  return EmitLoadOfPropertyRefLValue(LV);
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
    llvm::Type *PTy = llvm::Type::getIntNPtrTy(getLLVMContext(),
                                                     AI.AccessWidth,
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
  llvm::Value *Vec = Builder.CreateLoad(LV.getExtVectorAddr(),
                                        LV.isVolatileQualified(), "tmp");

  const llvm::Constant *Elts = LV.getExtVectorElts();

  // If the result of the expression is a non-vector type, we must be extracting
  // a single element.  Just codegen as an extractelement.
  const VectorType *ExprVT = LV.getType()->getAs<VectorType>();
  if (!ExprVT) {
    unsigned InIdx = getAccessedFieldNo(0, Elts);
    llvm::Value *Elt = llvm::ConstantInt::get(Int32Ty, InIdx);
    return RValue::get(Builder.CreateExtractElement(Vec, Elt, "tmp"));
  }

  // Always use shuffle vector to try to retain the original program structure
  unsigned NumResultElts = ExprVT->getNumElements();

  llvm::SmallVector<llvm::Constant*, 4> Mask;
  for (unsigned i = 0; i != NumResultElts; ++i) {
    unsigned InIdx = getAccessedFieldNo(i, Elts);
    Mask.push_back(llvm::ConstantInt::get(Int32Ty, InIdx));
  }

  llvm::Value *MaskV = llvm::ConstantVector::get(Mask);
  Vec = Builder.CreateShuffleVector(Vec, llvm::UndefValue::get(Vec->getType()),
                                    MaskV, "tmp");
  return RValue::get(Vec);
}



/// EmitStoreThroughLValue - Store the specified rvalue into the specified
/// lvalue, where both are guaranteed to the have the same type, and that type
/// is 'Ty'.
void CodeGenFunction::EmitStoreThroughLValue(RValue Src, LValue Dst) {
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
      return EmitStoreThroughExtVectorComponentLValue(Src, Dst);

    if (Dst.isBitField())
      return EmitStoreThroughBitfieldLValue(Src, Dst);

    assert(Dst.isPropertyRef() && "Unknown LValue type");
    return EmitStoreThroughPropertyRefLValue(Src, Dst);
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
  EmitStoreOfScalar(Src.getScalarVal(), Dst);
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
  llvm::Value *Vec = Builder.CreateLoad(Dst.getExtVectorAddr(),
                                        Dst.isVolatileQualified(), "tmp");
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
      llvm::SmallVector<llvm::Constant*, 4> Mask(NumDstElts);
      for (unsigned i = 0; i != NumSrcElts; ++i) {
        unsigned InIdx = getAccessedFieldNo(i, Elts);
        Mask[InIdx] = llvm::ConstantInt::get(Int32Ty, i);
      }

      llvm::Value *MaskV = llvm::ConstantVector::get(Mask);
      Vec = Builder.CreateShuffleVector(SrcVal,
                                        llvm::UndefValue::get(Vec->getType()),
                                        MaskV, "tmp");
    } else if (NumDstElts > NumSrcElts) {
      // Extended the source vector to the same length and then shuffle it
      // into the destination.
      // FIXME: since we're shuffling with undef, can we just use the indices
      //        into that?  This could be simpler.
      llvm::SmallVector<llvm::Constant*, 4> ExtMask;
      unsigned i;
      for (i = 0; i != NumSrcElts; ++i)
        ExtMask.push_back(llvm::ConstantInt::get(Int32Ty, i));
      for (; i != NumDstElts; ++i)
        ExtMask.push_back(llvm::UndefValue::get(Int32Ty));
      llvm::Value *ExtMaskV = llvm::ConstantVector::get(ExtMask);
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
      llvm::Value *MaskV = llvm::ConstantVector::get(Mask);
      Vec = Builder.CreateShuffleVector(Vec, ExtSrcVal, MaskV, "tmp");
    } else {
      // We should never shorten the vector
      assert(0 && "unexpected shorten vector length");
    }
  } else {
    // If the Src is a scalar (not a vector) it must be updating one element.
    unsigned InIdx = getAccessedFieldNo(0, Elts);
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
        LV.setObjCIvar(false); 
    }
    return;
  }

  if (const GenericSelectionExpr *Exp = dyn_cast<GenericSelectionExpr>(E)) {
    setObjCGCLValueClass(Ctx, Exp->getResultExpr(), LV);
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

  if (const ObjCBridgedCastExpr *Exp = dyn_cast<ObjCBridgedCastExpr>(E)) {
    setObjCGCLValueClass(Ctx, Exp->getSubExpr(), LV);
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
    setObjCGCLValueClass(Ctx, Exp->getBase(), LV);
    // We don't know if member is an 'ivar', but this flag is looked at
    // only in the context of LV.isObjCIvar().
    LV.setObjCArray(E->getType()->isArrayType());
    return;
  }
}

static llvm::Value *
EmitBitCastOfLValueToProperType(CodeGenFunction &CGF,
                                llvm::Value *V, llvm::Type *IRType,
                                llvm::StringRef Name = llvm::StringRef()) {
  unsigned AS = cast<llvm::PointerType>(V->getType())->getAddressSpace();
  return CGF.Builder.CreateBitCast(V, IRType->getPointerTo(AS), Name);
}

static LValue EmitGlobalVarDeclLValue(CodeGenFunction &CGF,
                                      const Expr *E, const VarDecl *VD) {
  assert((VD->hasExternalStorage() || VD->isFileVarDecl()) &&
         "Var decl must have external storage or be a file var decl!");

  llvm::Value *V = CGF.CGM.GetAddrOfGlobalVar(VD);
  if (VD->getType()->isReferenceType())
    V = CGF.Builder.CreateLoad(V, "tmp");
  
  V = EmitBitCastOfLValueToProperType(CGF, V,
                                CGF.getTypes().ConvertTypeForMem(E->getType()));

  unsigned Alignment = CGF.getContext().getDeclAlign(VD).getQuantity();
  LValue LV = CGF.MakeAddrLValue(V, E->getType(), Alignment);
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
      V = CGF.Builder.CreateBitCast(V, CGF.ConvertType(NoProtoType), "tmp");
    }
  }
  unsigned Alignment = CGF.getContext().getDeclAlign(FD).getQuantity();
  return CGF.MakeAddrLValue(V, E->getType(), Alignment);
}

LValue CodeGenFunction::EmitDeclRefLValue(const DeclRefExpr *E) {
  const NamedDecl *ND = E->getDecl();
  unsigned Alignment = getContext().getDeclAlign(ND).getQuantity();

  if (ND->hasAttr<WeakRefAttr>()) {
    const ValueDecl *VD = cast<ValueDecl>(ND);
    llvm::Constant *Aliasee = CGM.GetWeakRefReference(VD);
    return MakeAddrLValue(Aliasee, E->getType(), Alignment);
  }

  if (const VarDecl *VD = dyn_cast<VarDecl>(ND)) {
    
    // Check if this is a global variable.
    if (VD->hasExternalStorage() || VD->isFileVarDecl()) 
      return EmitGlobalVarDeclLValue(*this, E, VD);

    bool NonGCable = VD->hasLocalStorage() &&
                     !VD->getType()->isReferenceType() &&
                     !VD->hasAttr<BlocksAttr>();

    llvm::Value *V = LocalDeclMap[VD];
    if (!V && VD->isStaticLocal()) 
      V = CGM.getStaticLocalDeclAddress(VD);
    assert(V && "DeclRefExpr not entered in LocalDeclMap?");

    if (VD->hasAttr<BlocksAttr>())
      V = BuildBlockByrefAddress(V, VD);
    
    if (VD->getType()->isReferenceType())
      V = Builder.CreateLoad(V, "tmp");

    V = EmitBitCastOfLValueToProperType(*this, V,
                                    getTypes().ConvertTypeForMem(E->getType()));

    LValue LV = MakeAddrLValue(V, E->getType(), Alignment);
    if (NonGCable) {
      LV.getQuals().removeObjCGCAttr();
      LV.setNonGC(true);
    }
    setObjCGCLValueClass(getContext(), E, LV);
    return LV;
  }

  if (const FunctionDecl *fn = dyn_cast<FunctionDecl>(ND))
    return EmitFunctionDeclLValue(*this, E, fn);

  assert(false && "Unhandled DeclRefExpr");
  
  // an invalid LValue, but the assert will
  // ensure that this point is never reached.
  return LValue();
}

LValue CodeGenFunction::EmitBlockDeclRefLValue(const BlockDeclRefExpr *E) {
  unsigned Alignment =
    getContext().getDeclAlign(E->getDecl()).getQuantity();
  return MakeAddrLValue(GetAddrOfBlockDecl(E), E->getType(), Alignment);
}

LValue CodeGenFunction::EmitUnaryOpLValue(const UnaryOperator *E) {
  // __extension__ doesn't affect lvalue-ness.
  if (E->getOpcode() == UO_Extension)
    return EmitLValue(E->getSubExpr());

  QualType ExprTy = getContext().getCanonicalType(E->getSubExpr()->getType());
  switch (E->getOpcode()) {
  default: assert(0 && "Unknown unary operator lvalue!");
  case UO_Deref: {
    QualType T = E->getSubExpr()->getType()->getPointeeType();
    assert(!T.isNull() && "CodeGenFunction::EmitUnaryOpLValue: Illegal type");

    LValue LV = MakeAddrLValue(EmitScalarExpr(E->getSubExpr()), T);
    LV.getQuals().setAddressSpace(ExprTy.getAddressSpace());

    // We should not generate __weak write barrier on indirect reference
    // of a pointer to object; as in void foo (__weak id *param); *param = 0;
    // But, we continue to generate __strong write barrier on indirect write
    // into a pointer to object.
    if (getContext().getLangOptions().ObjC1 &&
        getContext().getLangOptions().getGCMode() != LangOptions::NonGC &&
        LV.isObjCWeak())
      LV.setNonGC(!E->isOBJCGCCandidate(getContext()));
    return LV;
  }
  case UO_Real:
  case UO_Imag: {
    LValue LV = EmitLValue(E->getSubExpr());
    assert(LV.isSimple() && "real/imag on non-ordinary l-value");
    llvm::Value *Addr = LV.getAddress();

    // real and imag are valid on scalars.  This is a faster way of
    // testing that.
    if (!cast<llvm::PointerType>(Addr->getType())
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
                                 E->getBase()->getType());
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
  unsigned ArrayAlignment = 0;
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
    if (getLangOptions().isSignedOverflowDefined()) {
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

    if (getContext().getLangOptions().isSignedOverflowDefined())
      Address = Builder.CreateGEP(ArrayPtr, Args, Args+2, "arrayidx");
    else
      Address = Builder.CreateInBoundsGEP(ArrayPtr, Args, Args+2, "arrayidx");
  } else {
    // The base must be a pointer, which is not an aggregate.  Emit it.
    llvm::Value *Base = EmitScalarExpr(E->getBase());
    if (getContext().getLangOptions().isSignedOverflowDefined())
      Address = Builder.CreateGEP(Base, Idx, "arrayidx");
    else
      Address = Builder.CreateInBoundsGEP(Base, Idx, "arrayidx");
  }

  QualType T = E->getBase()->getType()->getPointeeType();
  assert(!T.isNull() &&
         "CodeGenFunction::EmitArraySubscriptExpr(): Illegal base type");

  // Limit the alignment to that of the result type.
  if (ArrayAlignment) {
    unsigned Align = getContext().getTypeAlignInChars(T).getQuantity();
    ArrayAlignment = std::min(Align, ArrayAlignment);
  }

  LValue LV = MakeAddrLValue(Address, T, ArrayAlignment);
  LV.getQuals().setAddressSpace(E->getBase()->getType().getAddressSpace());

  if (getContext().getLangOptions().ObjC1 &&
      getContext().getLangOptions().getGCMode() != LangOptions::NonGC) {
    LV.setNonGC(!E->isOBJCGCCandidate(getContext()));
    setObjCGCLValueClass(getContext(), E, LV);
  }
  return LV;
}

static
llvm::Constant *GenerateConstantVector(llvm::LLVMContext &VMContext,
                                       llvm::SmallVector<unsigned, 4> &Elts) {
  llvm::SmallVector<llvm::Constant*, 4> CElts;

  llvm::Type *Int32Ty = llvm::Type::getInt32Ty(VMContext);
  for (unsigned i = 0, e = Elts.size(); i != e; ++i)
    CElts.push_back(llvm::ConstantInt::get(Int32Ty, Elts[i]));

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
  llvm::SmallVector<unsigned, 4> Indices;
  E->getEncodedElementAccess(Indices);

  if (Base.isSimple()) {
    llvm::Constant *CV = GenerateConstantVector(getLLVMContext(), Indices);
    return LValue::MakeExtVectorElt(Base.getAddress(), CV, type);
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
  llvm::Constant *CV = llvm::ConstantVector::get(CElts);
  return LValue::MakeExtVectorElt(Base.getExtVectorAddr(), CV, type);
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

  assert(false && "Unhandled member declaration!");
  return LValue();
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

  unsigned alignment = getContext().getDeclAlign(field).getQuantity();
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
  llvm::Value *V = Builder.CreateStructGEP(BaseValue, idx, "tmp");
  assert(!FieldType.getObjCGCAttr() && "fields cannot have GC attrs");

  
  // Make sure that the address is pointing to the right type.  This is critical
  // for both unions and structs.  A union needs a bitcast, a struct element
  // will need a bitcast if the LLVM type laid out doesn't match the desired
  // type.
  llvm::Type *llvmType = ConvertTypeForMem(FieldType);
  unsigned AS = cast<llvm::PointerType>(V->getType())->getAddressSpace();
  V = Builder.CreateBitCast(V, llvmType->getPointerTo(AS));
  
  unsigned Alignment = getContext().getDeclAlign(Field).getQuantity();
  return MakeAddrLValue(V, FieldType, Alignment);
}

LValue CodeGenFunction::EmitCompoundLiteralLValue(const CompoundLiteralExpr *E){
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

  const Expr *condExpr = expr->getCond();
  bool CondExprBool;
  if (ConstantFoldsToSimpleInteger(condExpr, CondExprBool)) {
    const Expr *live = expr->getTrueExpr(), *dead = expr->getFalseExpr();
    if (!CondExprBool) std::swap(live, dead);

    if (!ContainsLabel(dead))
      return EmitLValue(live);
  }

  OpaqueValueMapping binding(*this, expr);

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

  case CK_GetObjCProperty: {
    LValue LV = EmitLValue(E->getSubExpr());
    assert(LV.isPropertyRef());
    RValue RV = EmitLoadOfPropertyRefLValue(LV);

    // Property is an aggregate r-value.
    if (RV.isAggregate()) {
      return MakeAddrLValue(RV.getAggregateAddr(), E->getType());
    }

    // Implicit property returns an l-value.
    assert(RV.isScalar());
    return MakeAddrLValue(RV.getScalarVal(), E->getSubExpr()->getType());
  }

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
  case CK_AnyPointerToBlockPointerCast:
  case CK_ObjCProduceObject:
  case CK_ObjCConsumeObject:
  case CK_ObjCReclaimReturnedObject: {
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
  case CK_AnyPointerToObjCPointerCast:
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
  assert(e->isGLValue() || e->getType()->isRecordType());
  return getOpaqueLValueMapping(e);
}

LValue CodeGenFunction::EmitMaterializeTemporaryExpr(
                                           const MaterializeTemporaryExpr *E) {
  RValue RV = EmitReferenceBindingToExpr(E->GetTemporaryExpr(),
                                         /*InitializedDecl=*/0);
  return MakeAddrLValue(RV.getScalarVal(), E->getType());
}


//===--------------------------------------------------------------------===//
//                             Expression Emission
//===--------------------------------------------------------------------===//

RValue CodeGenFunction::EmitCallExpr(const CallExpr *E, 
                                     ReturnValueSlot ReturnValue) {
  if (CGDebugInfo *DI = getDebugInfo()) {
    DI->setLocation(E->getLocStart());
    DI->UpdateLineDirectiveRegion(Builder);
    DI->EmitStopPoint(Builder);
  }

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

  if (const CXXPseudoDestructorExpr *PseudoDtor 
          = dyn_cast<CXXPseudoDestructorExpr>(E->getCallee()->IgnoreParens())) {
    QualType DestroyedType = PseudoDtor->getDestroyedType();
    if (getContext().getLangOptions().ObjCAutoRefCount &&
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
  AggValueSlot Slot = CreateAggTemp(E->getType(), "tmp");
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
  Slot.setLifetimeExternallyManaged();
  EmitAggExpr(E->getSubExpr(), Slot);
  EmitCXXTemporary(E->getTemporary(), Slot.getAddr());
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

  return EmitCall(CGM.getTypes().getFunctionInfo(Args, FnType),
                  Callee, ReturnValue, Args, TargetDecl);
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
