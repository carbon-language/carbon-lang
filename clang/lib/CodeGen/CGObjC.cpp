//===---- CGBuiltin.cpp - Emit LLVM Code for builtins ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Objective-C code as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CGObjCRuntime.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/StmtObjC.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Target/TargetData.h"
using namespace clang;
using namespace CodeGen;

/// Emits an instance of NSConstantString representing the object.
llvm::Value *CodeGenFunction::EmitObjCStringLiteral(const ObjCStringLiteral *E)
{
  llvm::Constant *C = 
      CGM.getObjCRuntime().GenerateConstantString(E->getString());
  // FIXME: This bitcast should just be made an invariant on the Runtime.
  return llvm::ConstantExpr::getBitCast(C, ConvertType(E->getType()));
}

/// Emit a selector.
llvm::Value *CodeGenFunction::EmitObjCSelectorExpr(const ObjCSelectorExpr *E) {
  // Untyped selector.
  // Note that this implementation allows for non-constant strings to be passed
  // as arguments to @selector().  Currently, the only thing preventing this
  // behaviour is the type checking in the front end.
  return CGM.getObjCRuntime().GetSelector(Builder, E->getSelector());
}

llvm::Value *CodeGenFunction::EmitObjCProtocolExpr(const ObjCProtocolExpr *E) {
  // FIXME: This should pass the Decl not the name.
  return CGM.getObjCRuntime().GenerateProtocolRef(Builder, E->getProtocol());
}


RValue CodeGenFunction::EmitObjCMessageExpr(const ObjCMessageExpr *E,
                                            ReturnValueSlot Return) {
  // Only the lookup mechanism and first two arguments of the method
  // implementation vary between runtimes.  We can get the receiver and
  // arguments in generic code.

  CGObjCRuntime &Runtime = CGM.getObjCRuntime();
  bool isSuperMessage = false;
  bool isClassMessage = false;
  ObjCInterfaceDecl *OID = 0;
  // Find the receiver
  llvm::Value *Receiver = 0;
  switch (E->getReceiverKind()) {
  case ObjCMessageExpr::Instance:
    Receiver = EmitScalarExpr(E->getInstanceReceiver());
    break;

  case ObjCMessageExpr::Class: {
    const ObjCObjectType *ObjTy
      = E->getClassReceiver()->getAs<ObjCObjectType>();
    assert(ObjTy && "Invalid Objective-C class message send");
    OID = ObjTy->getInterface();
    assert(OID && "Invalid Objective-C class message send");
    Receiver = Runtime.GetClass(Builder, OID);
    isClassMessage = true;
    break;
  }

  case ObjCMessageExpr::SuperInstance:
    Receiver = LoadObjCSelf();
    isSuperMessage = true;
    break;

  case ObjCMessageExpr::SuperClass:
    Receiver = LoadObjCSelf();
    isSuperMessage = true;
    isClassMessage = true;
    break;
  }

  CallArgList Args;
  EmitCallArgs(Args, E->getMethodDecl(), E->arg_begin(), E->arg_end());

  QualType ResultType =
    E->getMethodDecl() ? E->getMethodDecl()->getResultType() : E->getType();

  if (isSuperMessage) {
    // super is only valid in an Objective-C method
    const ObjCMethodDecl *OMD = cast<ObjCMethodDecl>(CurFuncDecl);
    bool isCategoryImpl = isa<ObjCCategoryImplDecl>(OMD->getDeclContext());
    return Runtime.GenerateMessageSendSuper(*this, Return, ResultType,
                                            E->getSelector(),
                                            OMD->getClassInterface(),
                                            isCategoryImpl,
                                            Receiver,
                                            isClassMessage,
                                            Args,
                                            E->getMethodDecl());
  }

  return Runtime.GenerateMessageSend(*this, Return, ResultType,
                                     E->getSelector(),
                                     Receiver, Args, OID,
                                     E->getMethodDecl());
}

/// StartObjCMethod - Begin emission of an ObjCMethod. This generates
/// the LLVM function and sets the other context used by
/// CodeGenFunction.
void CodeGenFunction::StartObjCMethod(const ObjCMethodDecl *OMD,
                                      const ObjCContainerDecl *CD) {
  FunctionArgList Args;
  // Check if we should generate debug info for this method.
  if (CGM.getDebugInfo() && !OMD->hasAttr<NoDebugAttr>())
    DebugInfo = CGM.getDebugInfo();

  llvm::Function *Fn = CGM.getObjCRuntime().GenerateMethod(OMD, CD);

  const CGFunctionInfo &FI = CGM.getTypes().getFunctionInfo(OMD);
  CGM.SetInternalFunctionAttributes(OMD, Fn, FI);

  Args.push_back(std::make_pair(OMD->getSelfDecl(),
                                OMD->getSelfDecl()->getType()));
  Args.push_back(std::make_pair(OMD->getCmdDecl(),
                                OMD->getCmdDecl()->getType()));

  for (ObjCMethodDecl::param_iterator PI = OMD->param_begin(),
       E = OMD->param_end(); PI != E; ++PI)
    Args.push_back(std::make_pair(*PI, (*PI)->getType()));

  StartFunction(OMD, OMD->getResultType(), Fn, Args, OMD->getLocStart());
}

/// Generate an Objective-C method.  An Objective-C method is a C function with
/// its pointer, name, and types registered in the class struture.
void CodeGenFunction::GenerateObjCMethod(const ObjCMethodDecl *OMD) {
  StartObjCMethod(OMD, OMD->getClassInterface());
  EmitStmt(OMD->getBody());
  FinishFunction(OMD->getBodyRBrace());
}

// FIXME: I wasn't sure about the synthesis approach. If we end up generating an
// AST for the whole body we can just fall back to having a GenerateFunction
// which takes the body Stmt.

/// GenerateObjCGetter - Generate an Objective-C property getter
/// function. The given Decl must be an ObjCImplementationDecl. @synthesize
/// is illegal within a category.
void CodeGenFunction::GenerateObjCGetter(ObjCImplementationDecl *IMP,
                                         const ObjCPropertyImplDecl *PID) {
  ObjCIvarDecl *Ivar = PID->getPropertyIvarDecl();
  const ObjCPropertyDecl *PD = PID->getPropertyDecl();
  bool IsAtomic =
    !(PD->getPropertyAttributes() & ObjCPropertyDecl::OBJC_PR_nonatomic);
  ObjCMethodDecl *OMD = PD->getGetterMethodDecl();
  assert(OMD && "Invalid call to generate getter (empty method)");
  StartObjCMethod(OMD, IMP->getClassInterface());
  
  // Determine if we should use an objc_getProperty call for
  // this. Non-atomic properties are directly evaluated.
  // atomic 'copy' and 'retain' properties are also directly
  // evaluated in gc-only mode.
  if (CGM.getLangOptions().getGCMode() != LangOptions::GCOnly &&
      IsAtomic &&
      (PD->getSetterKind() == ObjCPropertyDecl::Copy ||
       PD->getSetterKind() == ObjCPropertyDecl::Retain)) {
    llvm::Value *GetPropertyFn =
      CGM.getObjCRuntime().GetPropertyGetFunction();

    if (!GetPropertyFn) {
      CGM.ErrorUnsupported(PID, "Obj-C getter requiring atomic copy");
      FinishFunction();
      return;
    }

    // Return (ivar-type) objc_getProperty((id) self, _cmd, offset, true).
    // FIXME: Can't this be simpler? This might even be worse than the
    // corresponding gcc code.
    CodeGenTypes &Types = CGM.getTypes();
    ValueDecl *Cmd = OMD->getCmdDecl();
    llvm::Value *CmdVal = Builder.CreateLoad(LocalDeclMap[Cmd], "cmd");
    QualType IdTy = getContext().getObjCIdType();
    llvm::Value *SelfAsId =
      Builder.CreateBitCast(LoadObjCSelf(), Types.ConvertType(IdTy));
    llvm::Value *Offset = EmitIvarOffset(IMP->getClassInterface(), Ivar);
    llvm::Value *True =
      llvm::ConstantInt::get(Types.ConvertType(getContext().BoolTy), 1);
    CallArgList Args;
    Args.push_back(std::make_pair(RValue::get(SelfAsId), IdTy));
    Args.push_back(std::make_pair(RValue::get(CmdVal), Cmd->getType()));
    Args.push_back(std::make_pair(RValue::get(Offset), getContext().LongTy));
    Args.push_back(std::make_pair(RValue::get(True), getContext().BoolTy));
    // FIXME: We shouldn't need to get the function info here, the
    // runtime already should have computed it to build the function.
    RValue RV = EmitCall(Types.getFunctionInfo(PD->getType(), Args,
                                               FunctionType::ExtInfo()),
                         GetPropertyFn, ReturnValueSlot(), Args);
    // We need to fix the type here. Ivars with copy & retain are
    // always objects so we don't need to worry about complex or
    // aggregates.
    RV = RValue::get(Builder.CreateBitCast(RV.getScalarVal(),
                                           Types.ConvertType(PD->getType())));
    EmitReturnOfRValue(RV, PD->getType());
  } else {
    if (Ivar->getType()->isAnyComplexType()) {
      LValue LV = EmitLValueForIvar(TypeOfSelfObject(), LoadObjCSelf(), 
                                    Ivar, 0);
      ComplexPairTy Pair = LoadComplexFromAddr(LV.getAddress(),
                                               LV.isVolatileQualified());
      StoreComplexToAddr(Pair, ReturnValue, LV.isVolatileQualified());
    }
    else if (hasAggregateLLVMType(Ivar->getType())) {
      bool IsStrong = false;
      if ((IsAtomic || (IsStrong = IvarTypeWithAggrGCObjects(Ivar->getType())))
          && CurFnInfo->getReturnInfo().getKind() == ABIArgInfo::Indirect
          && CGM.getObjCRuntime().GetCopyStructFunction()) {
        LValue LV = EmitLValueForIvar(TypeOfSelfObject(), LoadObjCSelf(), 
                                      Ivar, 0);
        llvm::Value *GetCopyStructFn =
          CGM.getObjCRuntime().GetCopyStructFunction();
        CodeGenTypes &Types = CGM.getTypes();
        // objc_copyStruct (ReturnValue, &structIvar, 
        //                  sizeof (Type of Ivar), isAtomic, false);
        CallArgList Args;
        RValue RV = RValue::get(Builder.CreateBitCast(ReturnValue,
                                    Types.ConvertType(getContext().VoidPtrTy)));
        Args.push_back(std::make_pair(RV, getContext().VoidPtrTy));
        RV = RValue::get(Builder.CreateBitCast(LV.getAddress(),
                                    Types.ConvertType(getContext().VoidPtrTy)));
        Args.push_back(std::make_pair(RV, getContext().VoidPtrTy));
        // sizeof (Type of Ivar)
        uint64_t Size =  getContext().getTypeSize(Ivar->getType()) / 8;
        llvm::Value *SizeVal =
          llvm::ConstantInt::get(Types.ConvertType(getContext().LongTy), Size);
        Args.push_back(std::make_pair(RValue::get(SizeVal),
                                      getContext().LongTy));
        llvm::Value *isAtomic =
          llvm::ConstantInt::get(Types.ConvertType(getContext().BoolTy), 
                                 IsAtomic ? 1 : 0);
        Args.push_back(std::make_pair(RValue::get(isAtomic), 
                                      getContext().BoolTy));
        llvm::Value *hasStrong =
          llvm::ConstantInt::get(Types.ConvertType(getContext().BoolTy), 
                                 IsStrong ? 1 : 0);
        Args.push_back(std::make_pair(RValue::get(hasStrong), 
                                      getContext().BoolTy));
        EmitCall(Types.getFunctionInfo(getContext().VoidTy, Args,
                                       FunctionType::ExtInfo()),
                 GetCopyStructFn, ReturnValueSlot(), Args);
      }
      else {
        if (PID->getGetterCXXConstructor()) {
          ReturnStmt *Stmt = 
            new (getContext()) ReturnStmt(SourceLocation(), 
                                          PID->getGetterCXXConstructor(),
                                          0);
          EmitReturnStmt(*Stmt);
        }
        else {
          LValue LV = EmitLValueForIvar(TypeOfSelfObject(), LoadObjCSelf(), 
                                        Ivar, 0);
          EmitAggregateCopy(ReturnValue, LV.getAddress(), Ivar->getType());
        }
      }
    } else {
      LValue LV = EmitLValueForIvar(TypeOfSelfObject(), LoadObjCSelf(), 
                                    Ivar, 0);
      CodeGenTypes &Types = CGM.getTypes();
      RValue RV = EmitLoadOfLValue(LV, Ivar->getType());
      RV = RValue::get(Builder.CreateBitCast(RV.getScalarVal(),
                       Types.ConvertType(PD->getType())));
      EmitReturnOfRValue(RV, PD->getType());
    }
  }

  FinishFunction();
}

/// GenerateObjCSetter - Generate an Objective-C property setter
/// function. The given Decl must be an ObjCImplementationDecl. @synthesize
/// is illegal within a category.
void CodeGenFunction::GenerateObjCSetter(ObjCImplementationDecl *IMP,
                                         const ObjCPropertyImplDecl *PID) {
  ObjCIvarDecl *Ivar = PID->getPropertyIvarDecl();
  const ObjCPropertyDecl *PD = PID->getPropertyDecl();
  ObjCMethodDecl *OMD = PD->getSetterMethodDecl();
  assert(OMD && "Invalid call to generate setter (empty method)");
  StartObjCMethod(OMD, IMP->getClassInterface());

  bool IsCopy = PD->getSetterKind() == ObjCPropertyDecl::Copy;
  bool IsAtomic =
    !(PD->getPropertyAttributes() & ObjCPropertyDecl::OBJC_PR_nonatomic);

  // Determine if we should use an objc_setProperty call for
  // this. Properties with 'copy' semantics always use it, as do
  // non-atomic properties with 'release' semantics as long as we are
  // not in gc-only mode.
  if (IsCopy ||
      (CGM.getLangOptions().getGCMode() != LangOptions::GCOnly &&
       PD->getSetterKind() == ObjCPropertyDecl::Retain)) {
    llvm::Value *SetPropertyFn =
      CGM.getObjCRuntime().GetPropertySetFunction();

    if (!SetPropertyFn) {
      CGM.ErrorUnsupported(PID, "Obj-C getter requiring atomic copy");
      FinishFunction();
      return;
    }

    // Emit objc_setProperty((id) self, _cmd, offset, arg,
    //                       <is-atomic>, <is-copy>).
    // FIXME: Can't this be simpler? This might even be worse than the
    // corresponding gcc code.
    CodeGenTypes &Types = CGM.getTypes();
    ValueDecl *Cmd = OMD->getCmdDecl();
    llvm::Value *CmdVal = Builder.CreateLoad(LocalDeclMap[Cmd], "cmd");
    QualType IdTy = getContext().getObjCIdType();
    llvm::Value *SelfAsId =
      Builder.CreateBitCast(LoadObjCSelf(), Types.ConvertType(IdTy));
    llvm::Value *Offset = EmitIvarOffset(IMP->getClassInterface(), Ivar);
    llvm::Value *Arg = LocalDeclMap[*OMD->param_begin()];
    llvm::Value *ArgAsId =
      Builder.CreateBitCast(Builder.CreateLoad(Arg, "arg"),
                            Types.ConvertType(IdTy));
    llvm::Value *True =
      llvm::ConstantInt::get(Types.ConvertType(getContext().BoolTy), 1);
    llvm::Value *False =
      llvm::ConstantInt::get(Types.ConvertType(getContext().BoolTy), 0);
    CallArgList Args;
    Args.push_back(std::make_pair(RValue::get(SelfAsId), IdTy));
    Args.push_back(std::make_pair(RValue::get(CmdVal), Cmd->getType()));
    Args.push_back(std::make_pair(RValue::get(Offset), getContext().LongTy));
    Args.push_back(std::make_pair(RValue::get(ArgAsId), IdTy));
    Args.push_back(std::make_pair(RValue::get(IsAtomic ? True : False),
                                  getContext().BoolTy));
    Args.push_back(std::make_pair(RValue::get(IsCopy ? True : False),
                                  getContext().BoolTy));
    // FIXME: We shouldn't need to get the function info here, the runtime
    // already should have computed it to build the function.
    EmitCall(Types.getFunctionInfo(getContext().VoidTy, Args,
                                   FunctionType::ExtInfo()),
             SetPropertyFn,
             ReturnValueSlot(), Args);
  } else if (IsAtomic && hasAggregateLLVMType(Ivar->getType()) &&
             !Ivar->getType()->isAnyComplexType() &&
             IndirectObjCSetterArg(*CurFnInfo)
             && CGM.getObjCRuntime().GetCopyStructFunction()) {
    // objc_copyStruct (&structIvar, &Arg, 
    //                  sizeof (struct something), true, false);
    llvm::Value *GetCopyStructFn =
      CGM.getObjCRuntime().GetCopyStructFunction();
    CodeGenTypes &Types = CGM.getTypes();
    CallArgList Args;
    LValue LV = EmitLValueForIvar(TypeOfSelfObject(), LoadObjCSelf(), Ivar, 0);
    RValue RV = RValue::get(Builder.CreateBitCast(LV.getAddress(),
                                    Types.ConvertType(getContext().VoidPtrTy)));
    Args.push_back(std::make_pair(RV, getContext().VoidPtrTy));
    llvm::Value *Arg = LocalDeclMap[*OMD->param_begin()];
    llvm::Value *ArgAsPtrTy =
      Builder.CreateBitCast(Arg,
                            Types.ConvertType(getContext().VoidPtrTy));
    RV = RValue::get(ArgAsPtrTy);
    Args.push_back(std::make_pair(RV, getContext().VoidPtrTy));
    // sizeof (Type of Ivar)
    uint64_t Size =  getContext().getTypeSize(Ivar->getType()) / 8;
    llvm::Value *SizeVal =
      llvm::ConstantInt::get(Types.ConvertType(getContext().LongTy), Size);
    Args.push_back(std::make_pair(RValue::get(SizeVal),
                                  getContext().LongTy));
    llvm::Value *True =
      llvm::ConstantInt::get(Types.ConvertType(getContext().BoolTy), 1);
    Args.push_back(std::make_pair(RValue::get(True), getContext().BoolTy));
    llvm::Value *False =
      llvm::ConstantInt::get(Types.ConvertType(getContext().BoolTy), 0);
    Args.push_back(std::make_pair(RValue::get(False), getContext().BoolTy));
    EmitCall(Types.getFunctionInfo(getContext().VoidTy, Args,
                                   FunctionType::ExtInfo()),
             GetCopyStructFn, ReturnValueSlot(), Args);
  } else if (PID->getSetterCXXAssignment()) {
    EmitAnyExpr(PID->getSetterCXXAssignment(), (llvm::Value *)0, false, true,
                false);
                
  } else {
    // FIXME: Find a clean way to avoid AST node creation.
    SourceLocation Loc = PD->getLocation();
    ValueDecl *Self = OMD->getSelfDecl();
    ObjCIvarDecl *Ivar = PID->getPropertyIvarDecl();
    DeclRefExpr Base(Self, Self->getType(), Loc);
    ParmVarDecl *ArgDecl = *OMD->param_begin();
    DeclRefExpr Arg(ArgDecl, ArgDecl->getType(), Loc);
    ObjCIvarRefExpr IvarRef(Ivar, Ivar->getType(), Loc, &Base, true, true);
    
    // The property type can differ from the ivar type in some situations with
    // Objective-C pointer types, we can always bit cast the RHS in these cases.
    if (getContext().getCanonicalType(Ivar->getType()) !=
        getContext().getCanonicalType(ArgDecl->getType())) {
      ImplicitCastExpr ArgCasted(ImplicitCastExpr::OnStack,
                                 Ivar->getType(), CastExpr::CK_BitCast, &Arg,
                                 VK_RValue);
      BinaryOperator Assign(&IvarRef, &ArgCasted, BinaryOperator::Assign,
                            Ivar->getType(), Loc);
      EmitStmt(&Assign);
    } else {
      BinaryOperator Assign(&IvarRef, &Arg, BinaryOperator::Assign,
                            Ivar->getType(), Loc);
      EmitStmt(&Assign);
    }
  }

  FinishFunction();
}

void CodeGenFunction::GenerateObjCCtorDtorMethod(ObjCImplementationDecl *IMP,
                                                 ObjCMethodDecl *MD,
                                                 bool ctor) {
  llvm::SmallVector<CXXBaseOrMemberInitializer *, 8> IvarInitializers;
  MD->createImplicitParams(CGM.getContext(), IMP->getClassInterface());
  StartObjCMethod(MD, IMP->getClassInterface());
  for (ObjCImplementationDecl::init_const_iterator B = IMP->init_begin(),
       E = IMP->init_end(); B != E; ++B) {
    CXXBaseOrMemberInitializer *Member = (*B);
    IvarInitializers.push_back(Member);
  }
  if (ctor) {
    for (unsigned I = 0, E = IvarInitializers.size(); I != E; ++I) {
      CXXBaseOrMemberInitializer *IvarInit = IvarInitializers[I];
      FieldDecl *Field = IvarInit->getMember();
      QualType FieldType = Field->getType();
      ObjCIvarDecl  *Ivar = cast<ObjCIvarDecl>(Field);
      LValue LV = EmitLValueForIvar(TypeOfSelfObject(), 
                                    LoadObjCSelf(), Ivar, 0);
      EmitAggExpr(IvarInit->getInit(), LV.getAddress(),
                  LV.isVolatileQualified(), false, true);
    }
    // constructor returns 'self'.
    CodeGenTypes &Types = CGM.getTypes();
    QualType IdTy(CGM.getContext().getObjCIdType());
    llvm::Value *SelfAsId =
      Builder.CreateBitCast(LoadObjCSelf(), Types.ConvertType(IdTy));
    EmitReturnOfRValue(RValue::get(SelfAsId), IdTy);
  } else {
    // dtor
    for (size_t i = IvarInitializers.size(); i > 0; --i) {
      FieldDecl *Field = IvarInitializers[i - 1]->getMember();
      QualType FieldType = Field->getType();
      const ConstantArrayType *Array = 
        getContext().getAsConstantArrayType(FieldType);
      if (Array)
        FieldType = getContext().getBaseElementType(FieldType);
      
      ObjCIvarDecl  *Ivar = cast<ObjCIvarDecl>(Field);
      LValue LV = EmitLValueForIvar(TypeOfSelfObject(), 
                                    LoadObjCSelf(), Ivar, 0);
      const RecordType *RT = FieldType->getAs<RecordType>();
      CXXRecordDecl *FieldClassDecl = cast<CXXRecordDecl>(RT->getDecl());
      CXXDestructorDecl *Dtor = FieldClassDecl->getDestructor();
      if (!Dtor->isTrivial()) {
        if (Array) {
          const llvm::Type *BasePtr = ConvertType(FieldType);
          BasePtr = llvm::PointerType::getUnqual(BasePtr);
          llvm::Value *BaseAddrPtr =
            Builder.CreateBitCast(LV.getAddress(), BasePtr);
          EmitCXXAggrDestructorCall(Dtor,
                                    Array, BaseAddrPtr);
        } else {
          EmitCXXDestructorCall(Dtor,
                                Dtor_Complete, /*ForVirtualBase=*/false,
                                LV.getAddress());
        }
      }
    }
  }
  FinishFunction();
}

bool CodeGenFunction::IndirectObjCSetterArg(const CGFunctionInfo &FI) {
  CGFunctionInfo::const_arg_iterator it = FI.arg_begin();
  it++; it++;
  const ABIArgInfo &AI = it->info;
  // FIXME. Is this sufficient check?
  return (AI.getKind() == ABIArgInfo::Indirect);
}

bool CodeGenFunction::IvarTypeWithAggrGCObjects(QualType Ty) {
  if (CGM.getLangOptions().getGCMode() == LangOptions::NonGC)
    return false;
  if (const RecordType *FDTTy = Ty.getTypePtr()->getAs<RecordType>())
    return FDTTy->getDecl()->hasObjectMember();
  return false;
}

llvm::Value *CodeGenFunction::LoadObjCSelf() {
  const ObjCMethodDecl *OMD = cast<ObjCMethodDecl>(CurFuncDecl);
  return Builder.CreateLoad(LocalDeclMap[OMD->getSelfDecl()], "self");
}

QualType CodeGenFunction::TypeOfSelfObject() {
  const ObjCMethodDecl *OMD = cast<ObjCMethodDecl>(CurFuncDecl);
  ImplicitParamDecl *selfDecl = OMD->getSelfDecl();
  const ObjCObjectPointerType *PTy = cast<ObjCObjectPointerType>(
    getContext().getCanonicalType(selfDecl->getType()));
  return PTy->getPointeeType();
}

RValue CodeGenFunction::EmitObjCSuperPropertyGet(const Expr *Exp,
                                                 const Selector &S,
                                                 ReturnValueSlot Return) {
  llvm::Value *Receiver = LoadObjCSelf();
  const ObjCMethodDecl *OMD = cast<ObjCMethodDecl>(CurFuncDecl);
  bool isClassMessage = OMD->isClassMethod();
  bool isCategoryImpl = isa<ObjCCategoryImplDecl>(OMD->getDeclContext());
  return CGM.getObjCRuntime().GenerateMessageSendSuper(*this,
                                                       Return,
                                                       Exp->getType(),
                                                       S,
                                                       OMD->getClassInterface(),
                                                       isCategoryImpl,
                                                       Receiver,
                                                       isClassMessage,
                                                       CallArgList());

}

RValue CodeGenFunction::EmitObjCPropertyGet(const Expr *Exp,
                                            ReturnValueSlot Return) {
  Exp = Exp->IgnoreParens();
  // FIXME: Split it into two separate routines.
  if (const ObjCPropertyRefExpr *E = dyn_cast<ObjCPropertyRefExpr>(Exp)) {
    Selector S = E->getProperty()->getGetterName();
    if (isa<ObjCSuperExpr>(E->getBase()))
      return EmitObjCSuperPropertyGet(E, S, Return);
    return CGM.getObjCRuntime().
             GenerateMessageSend(*this, Return, Exp->getType(), S,
                                 EmitScalarExpr(E->getBase()),
                                 CallArgList());
  } else {
    const ObjCImplicitSetterGetterRefExpr *KE =
      cast<ObjCImplicitSetterGetterRefExpr>(Exp);
    Selector S = KE->getGetterMethod()->getSelector();
    llvm::Value *Receiver;
    if (KE->getInterfaceDecl()) {
      const ObjCInterfaceDecl *OID = KE->getInterfaceDecl();
      Receiver = CGM.getObjCRuntime().GetClass(Builder, OID);
    } else if (isa<ObjCSuperExpr>(KE->getBase()))
      return EmitObjCSuperPropertyGet(KE, S, Return);
    else
      Receiver = EmitScalarExpr(KE->getBase());
    return CGM.getObjCRuntime().
             GenerateMessageSend(*this, Return, Exp->getType(), S,
                                 Receiver,
                                 CallArgList(), KE->getInterfaceDecl());
  }
}

void CodeGenFunction::EmitObjCSuperPropertySet(const Expr *Exp,
                                               const Selector &S,
                                               RValue Src) {
  CallArgList Args;
  llvm::Value *Receiver = LoadObjCSelf();
  const ObjCMethodDecl *OMD = cast<ObjCMethodDecl>(CurFuncDecl);
  bool isClassMessage = OMD->isClassMethod();
  bool isCategoryImpl = isa<ObjCCategoryImplDecl>(OMD->getDeclContext());
  Args.push_back(std::make_pair(Src, Exp->getType()));
  CGM.getObjCRuntime().GenerateMessageSendSuper(*this,
                                                ReturnValueSlot(),
                                                getContext().VoidTy,
                                                S,
                                                OMD->getClassInterface(),
                                                isCategoryImpl,
                                                Receiver,
                                                isClassMessage,
                                                Args);
  return;
}

void CodeGenFunction::EmitObjCPropertySet(const Expr *Exp,
                                          RValue Src) {
  // FIXME: Split it into two separate routines.
  if (const ObjCPropertyRefExpr *E = dyn_cast<ObjCPropertyRefExpr>(Exp)) {
    Selector S = E->getProperty()->getSetterName();
    if (isa<ObjCSuperExpr>(E->getBase())) {
      EmitObjCSuperPropertySet(E, S, Src);
      return;
    }
    CallArgList Args;
    Args.push_back(std::make_pair(Src, E->getType()));
    CGM.getObjCRuntime().GenerateMessageSend(*this, ReturnValueSlot(),
                                             getContext().VoidTy, S,
                                             EmitScalarExpr(E->getBase()),
                                             Args);
  } else if (const ObjCImplicitSetterGetterRefExpr *E =
               dyn_cast<ObjCImplicitSetterGetterRefExpr>(Exp)) {
    const ObjCMethodDecl *SetterMD = E->getSetterMethod();
    Selector S = SetterMD->getSelector();
    CallArgList Args;
    llvm::Value *Receiver;
    if (E->getInterfaceDecl()) {
      const ObjCInterfaceDecl *OID = E->getInterfaceDecl();
      Receiver = CGM.getObjCRuntime().GetClass(Builder, OID);
    } else if (isa<ObjCSuperExpr>(E->getBase())) {
      EmitObjCSuperPropertySet(E, S, Src);
      return;
    } else
      Receiver = EmitScalarExpr(E->getBase());
    ObjCMethodDecl::param_iterator P = SetterMD->param_begin(); 
    Args.push_back(std::make_pair(Src, (*P)->getType()));
    CGM.getObjCRuntime().GenerateMessageSend(*this, ReturnValueSlot(),
                                             getContext().VoidTy, S,
                                             Receiver,
                                             Args, E->getInterfaceDecl());
  } else
    assert (0 && "bad expression node in EmitObjCPropertySet");
}

void CodeGenFunction::EmitObjCForCollectionStmt(const ObjCForCollectionStmt &S){
  llvm::Constant *EnumerationMutationFn =
    CGM.getObjCRuntime().EnumerationMutationFunction();
  llvm::Value *DeclAddress;
  QualType ElementTy;

  if (!EnumerationMutationFn) {
    CGM.ErrorUnsupported(&S, "Obj-C fast enumeration for this runtime");
    return;
  }

  if (const DeclStmt *SD = dyn_cast<DeclStmt>(S.getElement())) {
    EmitStmt(SD);
    assert(HaveInsertPoint() && "DeclStmt destroyed insert point!");
    const Decl* D = SD->getSingleDecl();
    ElementTy = cast<ValueDecl>(D)->getType();
    DeclAddress = LocalDeclMap[D];
  } else {
    ElementTy = cast<Expr>(S.getElement())->getType();
    DeclAddress = 0;
  }

  // Fast enumeration state.
  QualType StateTy = getContext().getObjCFastEnumerationStateType();
  llvm::Value *StatePtr = CreateMemTemp(StateTy, "state.ptr");
  EmitNullInitialization(StatePtr, StateTy);

  // Number of elements in the items array.
  static const unsigned NumItems = 16;

  // Get selector
  IdentifierInfo *II[] = {
    &CGM.getContext().Idents.get("countByEnumeratingWithState"),
    &CGM.getContext().Idents.get("objects"),
    &CGM.getContext().Idents.get("count")
  };
  Selector FastEnumSel =
    CGM.getContext().Selectors.getSelector(llvm::array_lengthof(II), &II[0]);

  QualType ItemsTy =
    getContext().getConstantArrayType(getContext().getObjCIdType(),
                                      llvm::APInt(32, NumItems),
                                      ArrayType::Normal, 0);
  llvm::Value *ItemsPtr = CreateMemTemp(ItemsTy, "items.ptr");

  llvm::Value *Collection = EmitScalarExpr(S.getCollection());

  CallArgList Args;
  Args.push_back(std::make_pair(RValue::get(StatePtr),
                                getContext().getPointerType(StateTy)));

  Args.push_back(std::make_pair(RValue::get(ItemsPtr),
                                getContext().getPointerType(ItemsTy)));

  const llvm::Type *UnsignedLongLTy = ConvertType(getContext().UnsignedLongTy);
  llvm::Constant *Count = llvm::ConstantInt::get(UnsignedLongLTy, NumItems);
  Args.push_back(std::make_pair(RValue::get(Count),
                                getContext().UnsignedLongTy));

  RValue CountRV =
    CGM.getObjCRuntime().GenerateMessageSend(*this, ReturnValueSlot(),
                                             getContext().UnsignedLongTy,
                                             FastEnumSel,
                                             Collection, Args);

  llvm::Value *LimitPtr = CreateMemTemp(getContext().UnsignedLongTy,
                                        "limit.ptr");
  Builder.CreateStore(CountRV.getScalarVal(), LimitPtr);

  llvm::BasicBlock *NoElements = createBasicBlock("noelements");
  llvm::BasicBlock *SetStartMutations = createBasicBlock("setstartmutations");

  llvm::Value *Limit = Builder.CreateLoad(LimitPtr);
  llvm::Value *Zero = llvm::Constant::getNullValue(UnsignedLongLTy);

  llvm::Value *IsZero = Builder.CreateICmpEQ(Limit, Zero, "iszero");
  Builder.CreateCondBr(IsZero, NoElements, SetStartMutations);

  EmitBlock(SetStartMutations);

  llvm::Value *StartMutationsPtr = CreateMemTemp(getContext().UnsignedLongTy);

  llvm::Value *StateMutationsPtrPtr =
    Builder.CreateStructGEP(StatePtr, 2, "mutationsptr.ptr");
  llvm::Value *StateMutationsPtr = Builder.CreateLoad(StateMutationsPtrPtr,
                                                      "mutationsptr");

  llvm::Value *StateMutations = Builder.CreateLoad(StateMutationsPtr,
                                                   "mutations");

  Builder.CreateStore(StateMutations, StartMutationsPtr);

  llvm::BasicBlock *LoopStart = createBasicBlock("loopstart");
  EmitBlock(LoopStart);

  llvm::Value *CounterPtr = CreateMemTemp(getContext().UnsignedLongTy,
                                       "counter.ptr");
  Builder.CreateStore(Zero, CounterPtr);

  llvm::BasicBlock *LoopBody = createBasicBlock("loopbody");
  EmitBlock(LoopBody);

  StateMutationsPtr = Builder.CreateLoad(StateMutationsPtrPtr, "mutationsptr");
  StateMutations = Builder.CreateLoad(StateMutationsPtr, "statemutations");

  llvm::Value *StartMutations = Builder.CreateLoad(StartMutationsPtr,
                                                   "mutations");
  llvm::Value *MutationsEqual = Builder.CreateICmpEQ(StateMutations,
                                                     StartMutations,
                                                     "tobool");


  llvm::BasicBlock *WasMutated = createBasicBlock("wasmutated");
  llvm::BasicBlock *WasNotMutated = createBasicBlock("wasnotmutated");

  Builder.CreateCondBr(MutationsEqual, WasNotMutated, WasMutated);

  EmitBlock(WasMutated);
  llvm::Value *V =
    Builder.CreateBitCast(Collection,
                          ConvertType(getContext().getObjCIdType()),
                          "tmp");
  CallArgList Args2;
  Args2.push_back(std::make_pair(RValue::get(V),
                                getContext().getObjCIdType()));
  // FIXME: We shouldn't need to get the function info here, the runtime already
  // should have computed it to build the function.
  EmitCall(CGM.getTypes().getFunctionInfo(getContext().VoidTy, Args2,
                                          FunctionType::ExtInfo()),
           EnumerationMutationFn, ReturnValueSlot(), Args2);

  EmitBlock(WasNotMutated);

  llvm::Value *StateItemsPtr =
    Builder.CreateStructGEP(StatePtr, 1, "stateitems.ptr");

  llvm::Value *Counter = Builder.CreateLoad(CounterPtr, "counter");

  llvm::Value *EnumStateItems = Builder.CreateLoad(StateItemsPtr,
                                                   "stateitems");

  llvm::Value *CurrentItemPtr =
    Builder.CreateGEP(EnumStateItems, Counter, "currentitem.ptr");

  llvm::Value *CurrentItem = Builder.CreateLoad(CurrentItemPtr, "currentitem");

  // Cast the item to the right type.
  CurrentItem = Builder.CreateBitCast(CurrentItem,
                                      ConvertType(ElementTy), "tmp");

  if (!DeclAddress) {
    LValue LV = EmitLValue(cast<Expr>(S.getElement()));

    // Set the value to null.
    Builder.CreateStore(CurrentItem, LV.getAddress());
  } else
    Builder.CreateStore(CurrentItem, DeclAddress);

  // Increment the counter.
  Counter = Builder.CreateAdd(Counter,
                              llvm::ConstantInt::get(UnsignedLongLTy, 1));
  Builder.CreateStore(Counter, CounterPtr);

  JumpDest LoopEnd = getJumpDestInCurrentScope("loopend");
  JumpDest AfterBody = getJumpDestInCurrentScope("afterbody");

  BreakContinueStack.push_back(BreakContinue(LoopEnd, AfterBody));

  EmitStmt(S.getBody());

  BreakContinueStack.pop_back();

  EmitBlock(AfterBody.getBlock());

  llvm::BasicBlock *FetchMore = createBasicBlock("fetchmore");

  Counter = Builder.CreateLoad(CounterPtr);
  Limit = Builder.CreateLoad(LimitPtr);
  llvm::Value *IsLess = Builder.CreateICmpULT(Counter, Limit, "isless");
  Builder.CreateCondBr(IsLess, LoopBody, FetchMore);

  // Fetch more elements.
  EmitBlock(FetchMore);

  CountRV =
    CGM.getObjCRuntime().GenerateMessageSend(*this, ReturnValueSlot(),
                                             getContext().UnsignedLongTy,
                                             FastEnumSel,
                                             Collection, Args);
  Builder.CreateStore(CountRV.getScalarVal(), LimitPtr);
  Limit = Builder.CreateLoad(LimitPtr);

  IsZero = Builder.CreateICmpEQ(Limit, Zero, "iszero");
  Builder.CreateCondBr(IsZero, NoElements, LoopStart);

  // No more elements.
  EmitBlock(NoElements);

  if (!DeclAddress) {
    // If the element was not a declaration, set it to be null.

    LValue LV = EmitLValue(cast<Expr>(S.getElement()));

    // Set the value to null.
    Builder.CreateStore(llvm::Constant::getNullValue(ConvertType(ElementTy)),
                        LV.getAddress());
  }

  EmitBlock(LoopEnd.getBlock());
}

void CodeGenFunction::EmitObjCAtTryStmt(const ObjCAtTryStmt &S) {
  CGM.getObjCRuntime().EmitTryStmt(*this, S);
}

void CodeGenFunction::EmitObjCAtThrowStmt(const ObjCAtThrowStmt &S) {
  CGM.getObjCRuntime().EmitThrowStmt(*this, S);
}

void CodeGenFunction::EmitObjCAtSynchronizedStmt(
                                              const ObjCAtSynchronizedStmt &S) {
  CGM.getObjCRuntime().EmitSynchronizedStmt(*this, S);
}

CGObjCRuntime::~CGObjCRuntime() {}


