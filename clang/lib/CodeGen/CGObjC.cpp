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

#include "CGDebugInfo.h"
#include "CGObjCRuntime.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "TargetInfo.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/StmtObjC.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Target/TargetData.h"
#include "llvm/InlineAsm.h"
using namespace clang;
using namespace CodeGen;

typedef llvm::PointerIntPair<llvm::Value*,1,bool> TryEmitResult;
static TryEmitResult
tryEmitARCRetainScalarExpr(CodeGenFunction &CGF, const Expr *e);

/// Given the address of a variable of pointer type, find the correct
/// null to store into it.
static llvm::Constant *getNullForVariable(llvm::Value *addr) {
  llvm::Type *type =
    cast<llvm::PointerType>(addr->getType())->getElementType();
  return llvm::ConstantPointerNull::get(cast<llvm::PointerType>(type));
}

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

/// \brief Adjust the type of the result of an Objective-C message send 
/// expression when the method has a related result type.
static RValue AdjustRelatedResultType(CodeGenFunction &CGF,
                                      const Expr *E,
                                      const ObjCMethodDecl *Method,
                                      RValue Result) {
  if (!Method)
    return Result;

  if (!Method->hasRelatedResultType() ||
      CGF.getContext().hasSameType(E->getType(), Method->getResultType()) ||
      !Result.isScalar())
    return Result;
  
  // We have applied a related result type. Cast the rvalue appropriately.
  return RValue::get(CGF.Builder.CreateBitCast(Result.getScalarVal(),
                                               CGF.ConvertType(E->getType())));
}

/// Decide whether to extend the lifetime of the receiver of a
/// returns-inner-pointer message.
static bool
shouldExtendReceiverForInnerPointerMessage(const ObjCMessageExpr *message) {
  switch (message->getReceiverKind()) {

  // For a normal instance message, we should extend unless the
  // receiver is loaded from a variable with precise lifetime.
  case ObjCMessageExpr::Instance: {
    const Expr *receiver = message->getInstanceReceiver();
    const ImplicitCastExpr *ice = dyn_cast<ImplicitCastExpr>(receiver);
    if (!ice || ice->getCastKind() != CK_LValueToRValue) return true;
    receiver = ice->getSubExpr()->IgnoreParens();

    // Only __strong variables.
    if (receiver->getType().getObjCLifetime() != Qualifiers::OCL_Strong)
      return true;

    // All ivars and fields have precise lifetime.
    if (isa<MemberExpr>(receiver) || isa<ObjCIvarRefExpr>(receiver))
      return false;

    // Otherwise, check for variables.
    const DeclRefExpr *declRef = dyn_cast<DeclRefExpr>(ice->getSubExpr());
    if (!declRef) return true;
    const VarDecl *var = dyn_cast<VarDecl>(declRef->getDecl());
    if (!var) return true;

    // All variables have precise lifetime except local variables with
    // automatic storage duration that aren't specially marked.
    return (var->hasLocalStorage() &&
            !var->hasAttr<ObjCPreciseLifetimeAttr>());
  }

  case ObjCMessageExpr::Class:
  case ObjCMessageExpr::SuperClass:
    // It's never necessary for class objects.
    return false;

  case ObjCMessageExpr::SuperInstance:
    // We generally assume that 'self' lives throughout a method call.
    return false;
  }

  llvm_unreachable("invalid receiver kind");
}

RValue CodeGenFunction::EmitObjCMessageExpr(const ObjCMessageExpr *E,
                                            ReturnValueSlot Return) {
  // Only the lookup mechanism and first two arguments of the method
  // implementation vary between runtimes.  We can get the receiver and
  // arguments in generic code.

  bool isDelegateInit = E->isDelegateInitCall();

  const ObjCMethodDecl *method = E->getMethodDecl();

  // We don't retain the receiver in delegate init calls, and this is
  // safe because the receiver value is always loaded from 'self',
  // which we zero out.  We don't want to Block_copy block receivers,
  // though.
  bool retainSelf =
    (!isDelegateInit &&
     CGM.getLangOptions().ObjCAutoRefCount &&
     method &&
     method->hasAttr<NSConsumesSelfAttr>());

  CGObjCRuntime &Runtime = CGM.getObjCRuntime();
  bool isSuperMessage = false;
  bool isClassMessage = false;
  ObjCInterfaceDecl *OID = 0;
  // Find the receiver
  QualType ReceiverType;
  llvm::Value *Receiver = 0;
  switch (E->getReceiverKind()) {
  case ObjCMessageExpr::Instance:
    ReceiverType = E->getInstanceReceiver()->getType();
    if (retainSelf) {
      TryEmitResult ter = tryEmitARCRetainScalarExpr(*this,
                                                   E->getInstanceReceiver());
      Receiver = ter.getPointer();
      if (ter.getInt()) retainSelf = false;
    } else
      Receiver = EmitScalarExpr(E->getInstanceReceiver());
    break;

  case ObjCMessageExpr::Class: {
    ReceiverType = E->getClassReceiver();
    const ObjCObjectType *ObjTy = ReceiverType->getAs<ObjCObjectType>();
    assert(ObjTy && "Invalid Objective-C class message send");
    OID = ObjTy->getInterface();
    assert(OID && "Invalid Objective-C class message send");
    Receiver = Runtime.GetClass(Builder, OID);
    isClassMessage = true;
    break;
  }

  case ObjCMessageExpr::SuperInstance:
    ReceiverType = E->getSuperType();
    Receiver = LoadObjCSelf();
    isSuperMessage = true;
    break;

  case ObjCMessageExpr::SuperClass:
    ReceiverType = E->getSuperType();
    Receiver = LoadObjCSelf();
    isSuperMessage = true;
    isClassMessage = true;
    break;
  }

  if (retainSelf)
    Receiver = EmitARCRetainNonBlock(Receiver);

  // In ARC, we sometimes want to "extend the lifetime"
  // (i.e. retain+autorelease) of receivers of returns-inner-pointer
  // messages.
  if (getLangOptions().ObjCAutoRefCount && method &&
      method->hasAttr<ObjCReturnsInnerPointerAttr>() &&
      shouldExtendReceiverForInnerPointerMessage(E))
    Receiver = EmitARCRetainAutorelease(ReceiverType, Receiver);

  QualType ResultType =
    method ? method->getResultType() : E->getType();

  CallArgList Args;
  EmitCallArgs(Args, method, E->arg_begin(), E->arg_end());

  // For delegate init calls in ARC, do an unsafe store of null into
  // self.  This represents the call taking direct ownership of that
  // value.  We have to do this after emitting the other call
  // arguments because they might also reference self, but we don't
  // have to worry about any of them modifying self because that would
  // be an undefined read and write of an object in unordered
  // expressions.
  if (isDelegateInit) {
    assert(getLangOptions().ObjCAutoRefCount &&
           "delegate init calls should only be marked in ARC");

    // Do an unsafe store of null into self.
    llvm::Value *selfAddr =
      LocalDeclMap[cast<ObjCMethodDecl>(CurCodeDecl)->getSelfDecl()];
    assert(selfAddr && "no self entry for a delegate init call?");

    Builder.CreateStore(getNullForVariable(selfAddr), selfAddr);
  }

  RValue result;
  if (isSuperMessage) {
    // super is only valid in an Objective-C method
    const ObjCMethodDecl *OMD = cast<ObjCMethodDecl>(CurFuncDecl);
    bool isCategoryImpl = isa<ObjCCategoryImplDecl>(OMD->getDeclContext());
    result = Runtime.GenerateMessageSendSuper(*this, Return, ResultType,
                                              E->getSelector(),
                                              OMD->getClassInterface(),
                                              isCategoryImpl,
                                              Receiver,
                                              isClassMessage,
                                              Args,
                                              method);
  } else {
    result = Runtime.GenerateMessageSend(*this, Return, ResultType,
                                         E->getSelector(),
                                         Receiver, Args, OID,
                                         method);
  }

  // For delegate init calls in ARC, implicitly store the result of
  // the call back into self.  This takes ownership of the value.
  if (isDelegateInit) {
    llvm::Value *selfAddr =
      LocalDeclMap[cast<ObjCMethodDecl>(CurCodeDecl)->getSelfDecl()];
    llvm::Value *newSelf = result.getScalarVal();

    // The delegate return type isn't necessarily a matching type; in
    // fact, it's quite likely to be 'id'.
    llvm::Type *selfTy =
      cast<llvm::PointerType>(selfAddr->getType())->getElementType();
    newSelf = Builder.CreateBitCast(newSelf, selfTy);

    Builder.CreateStore(newSelf, selfAddr);
  }

  return AdjustRelatedResultType(*this, E, method, result);
}

namespace {
struct FinishARCDealloc : EHScopeStack::Cleanup {
  void Emit(CodeGenFunction &CGF, Flags flags) {
    const ObjCMethodDecl *method = cast<ObjCMethodDecl>(CGF.CurCodeDecl);

    const ObjCImplDecl *impl = cast<ObjCImplDecl>(method->getDeclContext());
    const ObjCInterfaceDecl *iface = impl->getClassInterface();
    if (!iface->getSuperClass()) return;

    bool isCategory = isa<ObjCCategoryImplDecl>(impl);

    // Call [super dealloc] if we have a superclass.
    llvm::Value *self = CGF.LoadObjCSelf();

    CallArgList args;
    CGF.CGM.getObjCRuntime().GenerateMessageSendSuper(CGF, ReturnValueSlot(),
                                                      CGF.getContext().VoidTy,
                                                      method->getSelector(),
                                                      iface,
                                                      isCategory,
                                                      self,
                                                      /*is class msg*/ false,
                                                      args,
                                                      method);
  }
};
}

/// StartObjCMethod - Begin emission of an ObjCMethod. This generates
/// the LLVM function and sets the other context used by
/// CodeGenFunction.
void CodeGenFunction::StartObjCMethod(const ObjCMethodDecl *OMD,
                                      const ObjCContainerDecl *CD,
                                      SourceLocation StartLoc) {
  FunctionArgList args;
  // Check if we should generate debug info for this method.
  if (CGM.getModuleDebugInfo() && !OMD->hasAttr<NoDebugAttr>())
    DebugInfo = CGM.getModuleDebugInfo();

  llvm::Function *Fn = CGM.getObjCRuntime().GenerateMethod(OMD, CD);

  const CGFunctionInfo &FI = CGM.getTypes().getFunctionInfo(OMD);
  CGM.SetInternalFunctionAttributes(OMD, Fn, FI);

  args.push_back(OMD->getSelfDecl());
  args.push_back(OMD->getCmdDecl());

  for (ObjCMethodDecl::param_iterator PI = OMD->param_begin(),
       E = OMD->param_end(); PI != E; ++PI)
    args.push_back(*PI);

  CurGD = OMD;

  StartFunction(OMD, OMD->getResultType(), Fn, FI, args, StartLoc);

  // In ARC, certain methods get an extra cleanup.
  if (CGM.getLangOptions().ObjCAutoRefCount &&
      OMD->isInstanceMethod() &&
      OMD->getSelector().isUnarySelector()) {
    const IdentifierInfo *ident = 
      OMD->getSelector().getIdentifierInfoForSlot(0);
    if (ident->isStr("dealloc"))
      EHStack.pushCleanup<FinishARCDealloc>(getARCCleanupKind());
  }
}

static llvm::Value *emitARCRetainLoadOfScalar(CodeGenFunction &CGF,
                                              LValue lvalue, QualType type);

void CodeGenFunction::GenerateObjCGetterBody(ObjCIvarDecl *Ivar, 
                                             bool IsAtomic, bool IsStrong) {
  LValue LV = EmitLValueForIvar(TypeOfSelfObject(), LoadObjCSelf(), 
                                Ivar, 0);
  llvm::Value *GetCopyStructFn =
  CGM.getObjCRuntime().GetGetStructFunction();
  CodeGenTypes &Types = CGM.getTypes();
  // objc_copyStruct (ReturnValue, &structIvar, 
  //                  sizeof (Type of Ivar), isAtomic, false);
  CallArgList Args;
  RValue RV = RValue::get(Builder.CreateBitCast(ReturnValue, VoidPtrTy));
  Args.add(RV, getContext().VoidPtrTy);
  RV = RValue::get(Builder.CreateBitCast(LV.getAddress(), VoidPtrTy));
  Args.add(RV, getContext().VoidPtrTy);
  // sizeof (Type of Ivar)
  CharUnits Size =  getContext().getTypeSizeInChars(Ivar->getType());
  llvm::Value *SizeVal =
  llvm::ConstantInt::get(Types.ConvertType(getContext().LongTy),
                         Size.getQuantity());
  Args.add(RValue::get(SizeVal), getContext().LongTy);
  llvm::Value *isAtomic =
  llvm::ConstantInt::get(Types.ConvertType(getContext().BoolTy), 
                         IsAtomic ? 1 : 0);
  Args.add(RValue::get(isAtomic), getContext().BoolTy);
  llvm::Value *hasStrong =
  llvm::ConstantInt::get(Types.ConvertType(getContext().BoolTy), 
                         IsStrong ? 1 : 0);
  Args.add(RValue::get(hasStrong), getContext().BoolTy);
  EmitCall(Types.getFunctionInfo(getContext().VoidTy, Args,
                                 FunctionType::ExtInfo()),
           GetCopyStructFn, ReturnValueSlot(), Args);
}

/// Generate an Objective-C method.  An Objective-C method is a C function with
/// its pointer, name, and types registered in the class struture.
void CodeGenFunction::GenerateObjCMethod(const ObjCMethodDecl *OMD) {
  StartObjCMethod(OMD, OMD->getClassInterface(), OMD->getLocStart());
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
  StartObjCMethod(OMD, IMP->getClassInterface(), PID->getLocStart());
  
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
    Args.add(RValue::get(SelfAsId), IdTy);
    Args.add(RValue::get(CmdVal), Cmd->getType());
    Args.add(RValue::get(Offset), getContext().getPointerDiffType());
    Args.add(RValue::get(True), getContext().BoolTy);
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

    // objc_getProperty does an autorelease, so we should suppress ours.
    AutoreleaseResult = false;
  } else {
    const llvm::Triple &Triple = getContext().Target.getTriple();
    QualType IVART = Ivar->getType();
    if (IsAtomic &&
        IVART->isScalarType() &&
        (Triple.getArch() == llvm::Triple::arm ||
         Triple.getArch() == llvm::Triple::thumb) &&
        (getContext().getTypeSizeInChars(IVART) 
         > CharUnits::fromQuantity(4)) &&
        CGM.getObjCRuntime().GetGetStructFunction()) {
      GenerateObjCGetterBody(Ivar, true, false);
    }
    else if (IsAtomic &&
             (IVART->isScalarType() && !IVART->isRealFloatingType()) &&
             Triple.getArch() == llvm::Triple::x86 &&
             (getContext().getTypeSizeInChars(IVART) 
              > CharUnits::fromQuantity(4)) &&
             CGM.getObjCRuntime().GetGetStructFunction()) {
      GenerateObjCGetterBody(Ivar, true, false);
    }
    else if (IsAtomic &&
             (IVART->isScalarType() && !IVART->isRealFloatingType()) &&
             Triple.getArch() == llvm::Triple::x86_64 &&
             (getContext().getTypeSizeInChars(IVART) 
              > CharUnits::fromQuantity(8)) &&
             CGM.getObjCRuntime().GetGetStructFunction()) {
      GenerateObjCGetterBody(Ivar, true, false);
    }
    else if (IVART->isAnyComplexType()) {
      LValue LV = EmitLValueForIvar(TypeOfSelfObject(), LoadObjCSelf(), 
                                    Ivar, 0);
      ComplexPairTy Pair = LoadComplexFromAddr(LV.getAddress(),
                                               LV.isVolatileQualified());
      StoreComplexToAddr(Pair, ReturnValue, LV.isVolatileQualified());
    }
    else if (hasAggregateLLVMType(IVART)) {
      bool IsStrong = false;
      if ((IsStrong = IvarTypeWithAggrGCObjects(IVART))
          && CurFnInfo->getReturnInfo().getKind() == ABIArgInfo::Indirect
          && CGM.getObjCRuntime().GetGetStructFunction()) {
        GenerateObjCGetterBody(Ivar, IsAtomic, IsStrong);
      }
      else {
        const CXXRecordDecl *classDecl = IVART->getAsCXXRecordDecl();
        
        if (PID->getGetterCXXConstructor() &&
            classDecl && !classDecl->hasTrivialDefaultConstructor()) {
          ReturnStmt *Stmt = 
            new (getContext()) ReturnStmt(SourceLocation(), 
                                          PID->getGetterCXXConstructor(),
                                          0);
          EmitReturnStmt(*Stmt);
        } else if (IsAtomic &&
                   !IVART->isAnyComplexType() &&
                   Triple.getArch() == llvm::Triple::x86 &&
                   (getContext().getTypeSizeInChars(IVART) 
                    > CharUnits::fromQuantity(4)) &&
                   CGM.getObjCRuntime().GetGetStructFunction()) {
          GenerateObjCGetterBody(Ivar, true, false);
        }
        else if (IsAtomic &&
                 !IVART->isAnyComplexType() &&
                 Triple.getArch() == llvm::Triple::x86_64 &&
                 (getContext().getTypeSizeInChars(IVART) 
                  > CharUnits::fromQuantity(8)) &&
                 CGM.getObjCRuntime().GetGetStructFunction()) {
          GenerateObjCGetterBody(Ivar, true, false);
        }
        else {
          LValue LV = EmitLValueForIvar(TypeOfSelfObject(), LoadObjCSelf(), 
                                        Ivar, 0);
          EmitAggregateCopy(ReturnValue, LV.getAddress(), IVART);
        }
      }
    } else {
      LValue LV = EmitLValueForIvar(TypeOfSelfObject(), LoadObjCSelf(), 
                                    Ivar, 0);
      QualType propType = PD->getType();

      llvm::Value *value;
      if (propType->isReferenceType()) {
        value = LV.getAddress();
      } else {
        // We want to load and autoreleaseReturnValue ARC __weak ivars.
        if (LV.getQuals().getObjCLifetime() == Qualifiers::OCL_Weak) {
          value = emitARCRetainLoadOfScalar(*this, LV, IVART);

        // Otherwise we want to do a simple load, suppressing the
        // final autorelease.
        } else {
          value = EmitLoadOfLValue(LV).getScalarVal();
          AutoreleaseResult = false;
        }

        value = Builder.CreateBitCast(value, ConvertType(propType));
      }
      
      EmitReturnOfRValue(RValue::get(value), propType);
    }
  }

  FinishFunction();
}

void CodeGenFunction::GenerateObjCAtomicSetterBody(ObjCMethodDecl *OMD,
                                                   ObjCIvarDecl *Ivar) {
  // objc_copyStruct (&structIvar, &Arg, 
  //                  sizeof (struct something), true, false);
  llvm::Value *GetCopyStructFn =
  CGM.getObjCRuntime().GetSetStructFunction();
  CodeGenTypes &Types = CGM.getTypes();
  CallArgList Args;
  LValue LV = EmitLValueForIvar(TypeOfSelfObject(), LoadObjCSelf(), Ivar, 0);
  RValue RV =
    RValue::get(Builder.CreateBitCast(LV.getAddress(),
                Types.ConvertType(getContext().VoidPtrTy)));
  Args.add(RV, getContext().VoidPtrTy);
  llvm::Value *Arg = LocalDeclMap[*OMD->param_begin()];
  llvm::Value *ArgAsPtrTy =
  Builder.CreateBitCast(Arg,
                      Types.ConvertType(getContext().VoidPtrTy));
  RV = RValue::get(ArgAsPtrTy);
  Args.add(RV, getContext().VoidPtrTy);
  // sizeof (Type of Ivar)
  CharUnits Size =  getContext().getTypeSizeInChars(Ivar->getType());
  llvm::Value *SizeVal =
  llvm::ConstantInt::get(Types.ConvertType(getContext().LongTy), 
                         Size.getQuantity());
  Args.add(RValue::get(SizeVal), getContext().LongTy);
  llvm::Value *True =
  llvm::ConstantInt::get(Types.ConvertType(getContext().BoolTy), 1);
  Args.add(RValue::get(True), getContext().BoolTy);
  llvm::Value *False =
  llvm::ConstantInt::get(Types.ConvertType(getContext().BoolTy), 0);
  Args.add(RValue::get(False), getContext().BoolTy);
  EmitCall(Types.getFunctionInfo(getContext().VoidTy, Args,
                                 FunctionType::ExtInfo()),
           GetCopyStructFn, ReturnValueSlot(), Args);
}

static bool
IvarAssignHasTrvialAssignment(const ObjCPropertyImplDecl *PID,
                              QualType IvarT) {
  bool HasTrvialAssignment = true;
  if (PID->getSetterCXXAssignment()) {
    const CXXRecordDecl *classDecl = IvarT->getAsCXXRecordDecl();
    HasTrvialAssignment = 
      (!classDecl || classDecl->hasTrivialCopyAssignment());
  }
  return HasTrvialAssignment;
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
  StartObjCMethod(OMD, IMP->getClassInterface(), PID->getLocStart());
  const llvm::Triple &Triple = getContext().Target.getTriple();
  QualType IVART = Ivar->getType();
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
    Args.add(RValue::get(SelfAsId), IdTy);
    Args.add(RValue::get(CmdVal), Cmd->getType());
    Args.add(RValue::get(Offset), getContext().getPointerDiffType());
    Args.add(RValue::get(ArgAsId), IdTy);
    Args.add(RValue::get(IsAtomic ? True : False),  getContext().BoolTy);
    Args.add(RValue::get(IsCopy ? True : False), getContext().BoolTy);
    // FIXME: We shouldn't need to get the function info here, the runtime
    // already should have computed it to build the function.
    EmitCall(Types.getFunctionInfo(getContext().VoidTy, Args,
                                   FunctionType::ExtInfo()),
             SetPropertyFn,
             ReturnValueSlot(), Args);
  } else if (IsAtomic && hasAggregateLLVMType(IVART) &&
             !IVART->isAnyComplexType() &&
             IvarAssignHasTrvialAssignment(PID, IVART) &&
             ((Triple.getArch() == llvm::Triple::x86 &&
              (getContext().getTypeSizeInChars(IVART)
               > CharUnits::fromQuantity(4))) ||
              (Triple.getArch() == llvm::Triple::x86_64 &&
              (getContext().getTypeSizeInChars(IVART)
               > CharUnits::fromQuantity(8))))
             && CGM.getObjCRuntime().GetSetStructFunction()) {
          // objc_copyStruct (&structIvar, &Arg, 
          //                  sizeof (struct something), true, false);
    GenerateObjCAtomicSetterBody(OMD, Ivar);
  } else if (PID->getSetterCXXAssignment()) {
    EmitIgnoredExpr(PID->getSetterCXXAssignment());
  } else {
    if (IsAtomic &&
        IVART->isScalarType() &&
        (Triple.getArch() == llvm::Triple::arm ||
         Triple.getArch() == llvm::Triple::thumb) &&
        (getContext().getTypeSizeInChars(IVART)
          > CharUnits::fromQuantity(4)) &&
        CGM.getObjCRuntime().GetGetStructFunction()) {
      GenerateObjCAtomicSetterBody(OMD, Ivar);
    }
    else if (IsAtomic &&
             (IVART->isScalarType() && !IVART->isRealFloatingType()) &&
             Triple.getArch() == llvm::Triple::x86 &&
             (getContext().getTypeSizeInChars(IVART)
              > CharUnits::fromQuantity(4)) &&
             CGM.getObjCRuntime().GetGetStructFunction()) {
      GenerateObjCAtomicSetterBody(OMD, Ivar);
    }
    else if (IsAtomic &&
             (IVART->isScalarType() && !IVART->isRealFloatingType()) &&
             Triple.getArch() == llvm::Triple::x86_64 &&
             (getContext().getTypeSizeInChars(IVART)
              > CharUnits::fromQuantity(8)) &&
             CGM.getObjCRuntime().GetGetStructFunction()) {
      GenerateObjCAtomicSetterBody(OMD, Ivar);
    }
    else {
      // FIXME: Find a clean way to avoid AST node creation.
      SourceLocation Loc = PID->getLocStart();
      ValueDecl *Self = OMD->getSelfDecl();
      ObjCIvarDecl *Ivar = PID->getPropertyIvarDecl();
      DeclRefExpr Base(Self, Self->getType(), VK_RValue, Loc);
      ParmVarDecl *ArgDecl = *OMD->param_begin();
      QualType T = ArgDecl->getType();
      if (T->isReferenceType())
        T = cast<ReferenceType>(T)->getPointeeType();
      DeclRefExpr Arg(ArgDecl, T, VK_LValue, Loc);
      ObjCIvarRefExpr IvarRef(Ivar, Ivar->getType(), Loc, &Base, true, true);
    
      // The property type can differ from the ivar type in some situations with
      // Objective-C pointer types, we can always bit cast the RHS in these cases.
      if (getContext().getCanonicalType(Ivar->getType()) !=
          getContext().getCanonicalType(ArgDecl->getType())) {
        ImplicitCastExpr ArgCasted(ImplicitCastExpr::OnStack,
                                   Ivar->getType(), CK_BitCast, &Arg,
                                   VK_RValue);
        BinaryOperator Assign(&IvarRef, &ArgCasted, BO_Assign,
                              Ivar->getType(), VK_RValue, OK_Ordinary, Loc);
        EmitStmt(&Assign);
      } else {
        BinaryOperator Assign(&IvarRef, &Arg, BO_Assign,
                              Ivar->getType(), VK_RValue, OK_Ordinary, Loc);
        EmitStmt(&Assign);
      }
    }
  }

  FinishFunction();
}

namespace {
  struct DestroyIvar : EHScopeStack::Cleanup {
  private:
    llvm::Value *addr;
    const ObjCIvarDecl *ivar;
    CodeGenFunction::Destroyer &destroyer;
    bool useEHCleanupForArray;
  public:
    DestroyIvar(llvm::Value *addr, const ObjCIvarDecl *ivar,
                CodeGenFunction::Destroyer *destroyer,
                bool useEHCleanupForArray)
      : addr(addr), ivar(ivar), destroyer(*destroyer),
        useEHCleanupForArray(useEHCleanupForArray) {}

    void Emit(CodeGenFunction &CGF, Flags flags) {
      LValue lvalue
        = CGF.EmitLValueForIvar(CGF.TypeOfSelfObject(), addr, ivar, /*CVR*/ 0);
      CGF.emitDestroy(lvalue.getAddress(), ivar->getType(), destroyer,
                      flags.isForNormalCleanup() && useEHCleanupForArray);
    }
  };
}

/// Like CodeGenFunction::destroyARCStrong, but do it with a call.
static void destroyARCStrongWithStore(CodeGenFunction &CGF,
                                      llvm::Value *addr,
                                      QualType type) {
  llvm::Value *null = getNullForVariable(addr);
  CGF.EmitARCStoreStrongCall(addr, null, /*ignored*/ true);
}

static void emitCXXDestructMethod(CodeGenFunction &CGF,
                                  ObjCImplementationDecl *impl) {
  CodeGenFunction::RunCleanupsScope scope(CGF);

  llvm::Value *self = CGF.LoadObjCSelf();

  const ObjCInterfaceDecl *iface = impl->getClassInterface();
  for (const ObjCIvarDecl *ivar = iface->all_declared_ivar_begin();
       ivar; ivar = ivar->getNextIvar()) {
    QualType type = ivar->getType();

    // Check whether the ivar is a destructible type.
    QualType::DestructionKind dtorKind = type.isDestructedType();
    if (!dtorKind) continue;

    CodeGenFunction::Destroyer *destroyer = 0;

    // Use a call to objc_storeStrong to destroy strong ivars, for the
    // general benefit of the tools.
    if (dtorKind == QualType::DK_objc_strong_lifetime) {
      destroyer = &destroyARCStrongWithStore;

    // Otherwise use the default for the destruction kind.
    } else {
      destroyer = &CGF.getDestroyer(dtorKind);
    }

    CleanupKind cleanupKind = CGF.getCleanupKind(dtorKind);

    CGF.EHStack.pushCleanup<DestroyIvar>(cleanupKind, self, ivar, destroyer,
                                         cleanupKind & EHCleanup);
  }

  assert(scope.requiresCleanups() && "nothing to do in .cxx_destruct?");
}

void CodeGenFunction::GenerateObjCCtorDtorMethod(ObjCImplementationDecl *IMP,
                                                 ObjCMethodDecl *MD,
                                                 bool ctor) {
  MD->createImplicitParams(CGM.getContext(), IMP->getClassInterface());
  StartObjCMethod(MD, IMP->getClassInterface(), MD->getLocStart());

  // Emit .cxx_construct.
  if (ctor) {
    // Suppress the final autorelease in ARC.
    AutoreleaseResult = false;

    SmallVector<CXXCtorInitializer *, 8> IvarInitializers;
    for (ObjCImplementationDecl::init_const_iterator B = IMP->init_begin(),
           E = IMP->init_end(); B != E; ++B) {
      CXXCtorInitializer *IvarInit = (*B);
      FieldDecl *Field = IvarInit->getAnyMember();
      ObjCIvarDecl  *Ivar = cast<ObjCIvarDecl>(Field);
      LValue LV = EmitLValueForIvar(TypeOfSelfObject(), 
                                    LoadObjCSelf(), Ivar, 0);
      EmitAggExpr(IvarInit->getInit(), AggValueSlot::forLValue(LV, true));
    }
    // constructor returns 'self'.
    CodeGenTypes &Types = CGM.getTypes();
    QualType IdTy(CGM.getContext().getObjCIdType());
    llvm::Value *SelfAsId =
      Builder.CreateBitCast(LoadObjCSelf(), Types.ConvertType(IdTy));
    EmitReturnOfRValue(RValue::get(SelfAsId), IdTy);

  // Emit .cxx_destruct.
  } else {
    emitCXXDestructMethod(*this, IMP);
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

LValue
CodeGenFunction::EmitObjCPropertyRefLValue(const ObjCPropertyRefExpr *E) {
  // This is a special l-value that just issues sends when we load or
  // store through it.

  // For certain base kinds, we need to emit the base immediately.
  llvm::Value *Base;
  if (E->isSuperReceiver())
    Base = LoadObjCSelf();
  else if (E->isClassReceiver())
    Base = CGM.getObjCRuntime().GetClass(Builder, E->getClassReceiver());
  else
    Base = EmitScalarExpr(E->getBase());
  return LValue::MakePropertyRef(E, Base);
}

static RValue GenerateMessageSendSuper(CodeGenFunction &CGF,
                                       ReturnValueSlot Return,
                                       QualType ResultType,
                                       Selector S,
                                       llvm::Value *Receiver,
                                       const CallArgList &CallArgs) {
  const ObjCMethodDecl *OMD = cast<ObjCMethodDecl>(CGF.CurFuncDecl);
  bool isClassMessage = OMD->isClassMethod();
  bool isCategoryImpl = isa<ObjCCategoryImplDecl>(OMD->getDeclContext());
  return CGF.CGM.getObjCRuntime()
                .GenerateMessageSendSuper(CGF, Return, ResultType,
                                          S, OMD->getClassInterface(),
                                          isCategoryImpl, Receiver,
                                          isClassMessage, CallArgs);
}

RValue CodeGenFunction::EmitLoadOfPropertyRefLValue(LValue LV,
                                                    ReturnValueSlot Return) {
  const ObjCPropertyRefExpr *E = LV.getPropertyRefExpr();
  QualType ResultType = E->getGetterResultType();
  Selector S;
  const ObjCMethodDecl *method;
  if (E->isExplicitProperty()) {
    const ObjCPropertyDecl *Property = E->getExplicitProperty();
    S = Property->getGetterName();
    method = Property->getGetterMethodDecl();
  } else {
    method = E->getImplicitPropertyGetter();
    S = method->getSelector();
  }

  llvm::Value *Receiver = LV.getPropertyRefBaseAddr();

  if (CGM.getLangOptions().ObjCAutoRefCount) {
    QualType receiverType;
    if (E->isSuperReceiver())
      receiverType = E->getSuperReceiverType();
    else if (E->isClassReceiver())
      receiverType = getContext().getObjCClassType();
    else
      receiverType = E->getBase()->getType();
  }

  // Accesses to 'super' follow a different code path.
  if (E->isSuperReceiver())
    return AdjustRelatedResultType(*this, E, method,
                                   GenerateMessageSendSuper(*this, Return, 
                                                            ResultType,
                                                            S, Receiver, 
                                                            CallArgList()));
  const ObjCInterfaceDecl *ReceiverClass
    = (E->isClassReceiver() ? E->getClassReceiver() : 0);
  return AdjustRelatedResultType(*this, E, method,
          CGM.getObjCRuntime().
             GenerateMessageSend(*this, Return, ResultType, S,
                                 Receiver, CallArgList(), ReceiverClass));
}

void CodeGenFunction::EmitStoreThroughPropertyRefLValue(RValue Src,
                                                        LValue Dst) {
  const ObjCPropertyRefExpr *E = Dst.getPropertyRefExpr();
  Selector S = E->getSetterSelector();
  QualType ArgType = E->getSetterArgType();
  
  // FIXME. Other than scalars, AST is not adequate for setter and
  // getter type mismatches which require conversion.
  if (Src.isScalar()) {
    llvm::Value *SrcVal = Src.getScalarVal();
    QualType DstType = getContext().getCanonicalType(ArgType);
    llvm::Type *DstTy = ConvertType(DstType);
    if (SrcVal->getType() != DstTy)
      Src = 
        RValue::get(EmitScalarConversion(SrcVal, E->getType(), DstType));
  }
  
  CallArgList Args;
  Args.add(Src, ArgType);

  llvm::Value *Receiver = Dst.getPropertyRefBaseAddr();
  QualType ResultType = getContext().VoidTy;

  if (E->isSuperReceiver()) {
    GenerateMessageSendSuper(*this, ReturnValueSlot(),
                             ResultType, S, Receiver, Args);
    return;
  }

  const ObjCInterfaceDecl *ReceiverClass
    = (E->isClassReceiver() ? E->getClassReceiver() : 0);

  CGM.getObjCRuntime().GenerateMessageSend(*this, ReturnValueSlot(),
                                           ResultType, S, Receiver, Args,
                                           ReceiverClass);
}

void CodeGenFunction::EmitObjCForCollectionStmt(const ObjCForCollectionStmt &S){
  llvm::Constant *EnumerationMutationFn =
    CGM.getObjCRuntime().EnumerationMutationFunction();

  if (!EnumerationMutationFn) {
    CGM.ErrorUnsupported(&S, "Obj-C fast enumeration for this runtime");
    return;
  }

  CGDebugInfo *DI = getDebugInfo();
  if (DI) {
    DI->setLocation(S.getSourceRange().getBegin());
    DI->EmitRegionStart(Builder);
  }

  // The local variable comes into scope immediately.
  AutoVarEmission variable = AutoVarEmission::invalid();
  if (const DeclStmt *SD = dyn_cast<DeclStmt>(S.getElement()))
    variable = EmitAutoVarAlloca(*cast<VarDecl>(SD->getSingleDecl()));

  JumpDest LoopEnd = getJumpDestInCurrentScope("forcoll.end");
  JumpDest AfterBody = getJumpDestInCurrentScope("forcoll.next");

  // Fast enumeration state.
  QualType StateTy = getContext().getObjCFastEnumerationStateType();
  llvm::Value *StatePtr = CreateMemTemp(StateTy, "state.ptr");
  EmitNullInitialization(StatePtr, StateTy);

  // Number of elements in the items array.
  static const unsigned NumItems = 16;

  // Fetch the countByEnumeratingWithState:objects:count: selector.
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

  // Emit the collection pointer.  In ARC, we do a retain.
  llvm::Value *Collection;
  if (getLangOptions().ObjCAutoRefCount) {
    Collection = EmitARCRetainScalarExpr(S.getCollection());

    // Enter a cleanup to do the release.
    EmitObjCConsumeObject(S.getCollection()->getType(), Collection);
  } else {
    Collection = EmitScalarExpr(S.getCollection());
  }

  // Send it our message:
  CallArgList Args;

  // The first argument is a temporary of the enumeration-state type.
  Args.add(RValue::get(StatePtr), getContext().getPointerType(StateTy));

  // The second argument is a temporary array with space for NumItems
  // pointers.  We'll actually be loading elements from the array
  // pointer written into the control state; this buffer is so that
  // collections that *aren't* backed by arrays can still queue up
  // batches of elements.
  Args.add(RValue::get(ItemsPtr), getContext().getPointerType(ItemsTy));

  // The third argument is the capacity of that temporary array.
  llvm::Type *UnsignedLongLTy = ConvertType(getContext().UnsignedLongTy);
  llvm::Constant *Count = llvm::ConstantInt::get(UnsignedLongLTy, NumItems);
  Args.add(RValue::get(Count), getContext().UnsignedLongTy);

  // Start the enumeration.
  RValue CountRV =
    CGM.getObjCRuntime().GenerateMessageSend(*this, ReturnValueSlot(),
                                             getContext().UnsignedLongTy,
                                             FastEnumSel,
                                             Collection, Args);

  // The initial number of objects that were returned in the buffer.
  llvm::Value *initialBufferLimit = CountRV.getScalarVal();

  llvm::BasicBlock *EmptyBB = createBasicBlock("forcoll.empty");
  llvm::BasicBlock *LoopInitBB = createBasicBlock("forcoll.loopinit");

  llvm::Value *zero = llvm::Constant::getNullValue(UnsignedLongLTy);

  // If the limit pointer was zero to begin with, the collection is
  // empty; skip all this.
  Builder.CreateCondBr(Builder.CreateICmpEQ(initialBufferLimit, zero, "iszero"),
                       EmptyBB, LoopInitBB);

  // Otherwise, initialize the loop.
  EmitBlock(LoopInitBB);

  // Save the initial mutations value.  This is the value at an
  // address that was written into the state object by
  // countByEnumeratingWithState:objects:count:.
  llvm::Value *StateMutationsPtrPtr =
    Builder.CreateStructGEP(StatePtr, 2, "mutationsptr.ptr");
  llvm::Value *StateMutationsPtr = Builder.CreateLoad(StateMutationsPtrPtr,
                                                      "mutationsptr");

  llvm::Value *initialMutations =
    Builder.CreateLoad(StateMutationsPtr, "forcoll.initial-mutations");

  // Start looping.  This is the point we return to whenever we have a
  // fresh, non-empty batch of objects.
  llvm::BasicBlock *LoopBodyBB = createBasicBlock("forcoll.loopbody");
  EmitBlock(LoopBodyBB);

  // The current index into the buffer.
  llvm::PHINode *index = Builder.CreatePHI(UnsignedLongLTy, 3, "forcoll.index");
  index->addIncoming(zero, LoopInitBB);

  // The current buffer size.
  llvm::PHINode *count = Builder.CreatePHI(UnsignedLongLTy, 3, "forcoll.count");
  count->addIncoming(initialBufferLimit, LoopInitBB);

  // Check whether the mutations value has changed from where it was
  // at start.  StateMutationsPtr should actually be invariant between
  // refreshes.
  StateMutationsPtr = Builder.CreateLoad(StateMutationsPtrPtr, "mutationsptr");
  llvm::Value *currentMutations
    = Builder.CreateLoad(StateMutationsPtr, "statemutations");

  llvm::BasicBlock *WasMutatedBB = createBasicBlock("forcoll.mutated");
  llvm::BasicBlock *WasNotMutatedBB = createBasicBlock("forcoll.notmutated");

  Builder.CreateCondBr(Builder.CreateICmpEQ(currentMutations, initialMutations),
                       WasNotMutatedBB, WasMutatedBB);

  // If so, call the enumeration-mutation function.
  EmitBlock(WasMutatedBB);
  llvm::Value *V =
    Builder.CreateBitCast(Collection,
                          ConvertType(getContext().getObjCIdType()),
                          "tmp");
  CallArgList Args2;
  Args2.add(RValue::get(V), getContext().getObjCIdType());
  // FIXME: We shouldn't need to get the function info here, the runtime already
  // should have computed it to build the function.
  EmitCall(CGM.getTypes().getFunctionInfo(getContext().VoidTy, Args2,
                                          FunctionType::ExtInfo()),
           EnumerationMutationFn, ReturnValueSlot(), Args2);

  // Otherwise, or if the mutation function returns, just continue.
  EmitBlock(WasNotMutatedBB);

  // Initialize the element variable.
  RunCleanupsScope elementVariableScope(*this);
  bool elementIsVariable;
  LValue elementLValue;
  QualType elementType;
  if (const DeclStmt *SD = dyn_cast<DeclStmt>(S.getElement())) {
    // Initialize the variable, in case it's a __block variable or something.
    EmitAutoVarInit(variable);

    const VarDecl* D = cast<VarDecl>(SD->getSingleDecl());
    DeclRefExpr tempDRE(const_cast<VarDecl*>(D), D->getType(),
                        VK_LValue, SourceLocation());
    elementLValue = EmitLValue(&tempDRE);
    elementType = D->getType();
    elementIsVariable = true;

    if (D->isARCPseudoStrong())
      elementLValue.getQuals().setObjCLifetime(Qualifiers::OCL_ExplicitNone);
  } else {
    elementLValue = LValue(); // suppress warning
    elementType = cast<Expr>(S.getElement())->getType();
    elementIsVariable = false;
  }
  llvm::Type *convertedElementType = ConvertType(elementType);

  // Fetch the buffer out of the enumeration state.
  // TODO: this pointer should actually be invariant between
  // refreshes, which would help us do certain loop optimizations.
  llvm::Value *StateItemsPtr =
    Builder.CreateStructGEP(StatePtr, 1, "stateitems.ptr");
  llvm::Value *EnumStateItems =
    Builder.CreateLoad(StateItemsPtr, "stateitems");

  // Fetch the value at the current index from the buffer.
  llvm::Value *CurrentItemPtr =
    Builder.CreateGEP(EnumStateItems, index, "currentitem.ptr");
  llvm::Value *CurrentItem = Builder.CreateLoad(CurrentItemPtr);

  // Cast that value to the right type.
  CurrentItem = Builder.CreateBitCast(CurrentItem, convertedElementType,
                                      "currentitem");

  // Make sure we have an l-value.  Yes, this gets evaluated every
  // time through the loop.
  if (!elementIsVariable) {
    elementLValue = EmitLValue(cast<Expr>(S.getElement()));
    EmitStoreThroughLValue(RValue::get(CurrentItem), elementLValue);
  } else {
    EmitScalarInit(CurrentItem, elementLValue);
  }

  // If we do have an element variable, this assignment is the end of
  // its initialization.
  if (elementIsVariable)
    EmitAutoVarCleanups(variable);

  // Perform the loop body, setting up break and continue labels.
  BreakContinueStack.push_back(BreakContinue(LoopEnd, AfterBody));
  {
    RunCleanupsScope Scope(*this);
    EmitStmt(S.getBody());
  }
  BreakContinueStack.pop_back();

  // Destroy the element variable now.
  elementVariableScope.ForceCleanup();

  // Check whether there are more elements.
  EmitBlock(AfterBody.getBlock());

  llvm::BasicBlock *FetchMoreBB = createBasicBlock("forcoll.refetch");

  // First we check in the local buffer.
  llvm::Value *indexPlusOne
    = Builder.CreateAdd(index, llvm::ConstantInt::get(UnsignedLongLTy, 1));

  // If we haven't overrun the buffer yet, we can continue.
  Builder.CreateCondBr(Builder.CreateICmpULT(indexPlusOne, count),
                       LoopBodyBB, FetchMoreBB);

  index->addIncoming(indexPlusOne, AfterBody.getBlock());
  count->addIncoming(count, AfterBody.getBlock());

  // Otherwise, we have to fetch more elements.
  EmitBlock(FetchMoreBB);

  CountRV =
    CGM.getObjCRuntime().GenerateMessageSend(*this, ReturnValueSlot(),
                                             getContext().UnsignedLongTy,
                                             FastEnumSel,
                                             Collection, Args);

  // If we got a zero count, we're done.
  llvm::Value *refetchCount = CountRV.getScalarVal();

  // (note that the message send might split FetchMoreBB)
  index->addIncoming(zero, Builder.GetInsertBlock());
  count->addIncoming(refetchCount, Builder.GetInsertBlock());

  Builder.CreateCondBr(Builder.CreateICmpEQ(refetchCount, zero),
                       EmptyBB, LoopBodyBB);

  // No more elements.
  EmitBlock(EmptyBB);

  if (!elementIsVariable) {
    // If the element was not a declaration, set it to be null.

    llvm::Value *null = llvm::Constant::getNullValue(convertedElementType);
    elementLValue = EmitLValue(cast<Expr>(S.getElement()));
    EmitStoreThroughLValue(RValue::get(null), elementLValue);
  }

  if (DI) {
    DI->setLocation(S.getSourceRange().getEnd());
    DI->EmitRegionEnd(Builder);
  }

  // Leave the cleanup we entered in ARC.
  if (getLangOptions().ObjCAutoRefCount)
    PopCleanupBlock();

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

/// Produce the code for a CK_ObjCProduceObject.  Just does a
/// primitive retain.
llvm::Value *CodeGenFunction::EmitObjCProduceObject(QualType type,
                                                    llvm::Value *value) {
  return EmitARCRetain(type, value);
}

namespace {
  struct CallObjCRelease : EHScopeStack::Cleanup {
    CallObjCRelease(QualType type, llvm::Value *ptr, llvm::Value *condition)
      : type(type), ptr(ptr), condition(condition) {}
    QualType type;
    llvm::Value *ptr;
    llvm::Value *condition;

    void Emit(CodeGenFunction &CGF, Flags flags) {
      llvm::Value *object;

      // If we're in a conditional branch, we had to stash away in an
      // alloca the pointer to be released.
      llvm::BasicBlock *cont = 0;
      if (condition) {
        llvm::BasicBlock *release = CGF.createBasicBlock("release.yes");
        cont = CGF.createBasicBlock("release.cont");

        llvm::Value *cond = CGF.Builder.CreateLoad(condition);
        CGF.Builder.CreateCondBr(cond, release, cont);
        CGF.EmitBlock(release);
        object = CGF.Builder.CreateLoad(ptr);
      } else {
        object = ptr;
      }

      CGF.EmitARCRelease(object, /*precise*/ true);

      if (cont) CGF.EmitBlock(cont);
    }
  };
}

/// Produce the code for a CK_ObjCConsumeObject.  Does a primitive
/// release at the end of the full-expression.
llvm::Value *CodeGenFunction::EmitObjCConsumeObject(QualType type,
                                                    llvm::Value *object) {
  // If we're in a conditional branch, we need to make the cleanup
  // conditional.  FIXME: this really needs to be supported by the
  // environment.
  llvm::AllocaInst *cond;
  llvm::Value *ptr;
  if (isInConditionalBranch()) {
    cond = CreateTempAlloca(Builder.getInt1Ty(), "release.cond");
    ptr = CreateTempAlloca(object->getType(), "release.value");

    // The alloca is false until we get here.
    // FIXME: er. doesn't this need to be set at the start of the condition?
    InitTempAlloca(cond, Builder.getFalse());

    // Then it turns true.
    Builder.CreateStore(Builder.getTrue(), cond);
    Builder.CreateStore(object, ptr);
  } else {
    cond = 0;
    ptr = object;
  }

  EHStack.pushCleanup<CallObjCRelease>(getARCCleanupKind(), type, ptr, cond);
  return object;
}

llvm::Value *CodeGenFunction::EmitObjCExtendObjectLifetime(QualType type,
                                                           llvm::Value *value) {
  return EmitARCRetainAutorelease(type, value);
}


static llvm::Constant *createARCRuntimeFunction(CodeGenModule &CGM,
                                                llvm::FunctionType *type,
                                                StringRef fnName) {
  llvm::Constant *fn = CGM.CreateRuntimeFunction(type, fnName);

  // In -fobjc-no-arc-runtime, emit weak references to the runtime
  // support library.
  if (!CGM.getCodeGenOpts().ObjCRuntimeHasARC)
    if (llvm::Function *f = dyn_cast<llvm::Function>(fn))
      f->setLinkage(llvm::Function::ExternalWeakLinkage);

  return fn;
}

/// Perform an operation having the signature
///   i8* (i8*)
/// where a null input causes a no-op and returns null.
static llvm::Value *emitARCValueOperation(CodeGenFunction &CGF,
                                          llvm::Value *value,
                                          llvm::Constant *&fn,
                                          StringRef fnName) {
  if (isa<llvm::ConstantPointerNull>(value)) return value;

  if (!fn) {
    std::vector<llvm::Type*> args(1, CGF.Int8PtrTy);
    llvm::FunctionType *fnType =
      llvm::FunctionType::get(CGF.Int8PtrTy, args, false);
    fn = createARCRuntimeFunction(CGF.CGM, fnType, fnName);
  }

  // Cast the argument to 'id'.
  llvm::Type *origType = value->getType();
  value = CGF.Builder.CreateBitCast(value, CGF.Int8PtrTy);

  // Call the function.
  llvm::CallInst *call = CGF.Builder.CreateCall(fn, value);
  call->setDoesNotThrow();

  // Cast the result back to the original type.
  return CGF.Builder.CreateBitCast(call, origType);
}

/// Perform an operation having the following signature:
///   i8* (i8**)
static llvm::Value *emitARCLoadOperation(CodeGenFunction &CGF,
                                         llvm::Value *addr,
                                         llvm::Constant *&fn,
                                         StringRef fnName) {
  if (!fn) {
    std::vector<llvm::Type*> args(1, CGF.Int8PtrPtrTy);
    llvm::FunctionType *fnType =
      llvm::FunctionType::get(CGF.Int8PtrTy, args, false);
    fn = createARCRuntimeFunction(CGF.CGM, fnType, fnName);
  }

  // Cast the argument to 'id*'.
  llvm::Type *origType = addr->getType();
  addr = CGF.Builder.CreateBitCast(addr, CGF.Int8PtrPtrTy);

  // Call the function.
  llvm::CallInst *call = CGF.Builder.CreateCall(fn, addr);
  call->setDoesNotThrow();

  // Cast the result back to a dereference of the original type.
  llvm::Value *result = call;
  if (origType != CGF.Int8PtrPtrTy)
    result = CGF.Builder.CreateBitCast(result,
                        cast<llvm::PointerType>(origType)->getElementType());

  return result;
}

/// Perform an operation having the following signature:
///   i8* (i8**, i8*)
static llvm::Value *emitARCStoreOperation(CodeGenFunction &CGF,
                                          llvm::Value *addr,
                                          llvm::Value *value,
                                          llvm::Constant *&fn,
                                          StringRef fnName,
                                          bool ignored) {
  assert(cast<llvm::PointerType>(addr->getType())->getElementType()
           == value->getType());

  if (!fn) {
    std::vector<llvm::Type*> argTypes(2);
    argTypes[0] = CGF.Int8PtrPtrTy;
    argTypes[1] = CGF.Int8PtrTy;

    llvm::FunctionType *fnType
      = llvm::FunctionType::get(CGF.Int8PtrTy, argTypes, false);
    fn = createARCRuntimeFunction(CGF.CGM, fnType, fnName);
  }

  llvm::Type *origType = value->getType();

  addr = CGF.Builder.CreateBitCast(addr, CGF.Int8PtrPtrTy);
  value = CGF.Builder.CreateBitCast(value, CGF.Int8PtrTy);
    
  llvm::CallInst *result = CGF.Builder.CreateCall2(fn, addr, value);
  result->setDoesNotThrow();

  if (ignored) return 0;

  return CGF.Builder.CreateBitCast(result, origType);
}

/// Perform an operation having the following signature:
///   void (i8**, i8**)
static void emitARCCopyOperation(CodeGenFunction &CGF,
                                 llvm::Value *dst,
                                 llvm::Value *src,
                                 llvm::Constant *&fn,
                                 StringRef fnName) {
  assert(dst->getType() == src->getType());

  if (!fn) {
    std::vector<llvm::Type*> argTypes(2, CGF.Int8PtrPtrTy);
    llvm::FunctionType *fnType
      = llvm::FunctionType::get(CGF.Builder.getVoidTy(), argTypes, false);
    fn = createARCRuntimeFunction(CGF.CGM, fnType, fnName);
  }

  dst = CGF.Builder.CreateBitCast(dst, CGF.Int8PtrPtrTy);
  src = CGF.Builder.CreateBitCast(src, CGF.Int8PtrPtrTy);
    
  llvm::CallInst *result = CGF.Builder.CreateCall2(fn, dst, src);
  result->setDoesNotThrow();
}

/// Produce the code to do a retain.  Based on the type, calls one of:
///   call i8* @objc_retain(i8* %value)
///   call i8* @objc_retainBlock(i8* %value)
llvm::Value *CodeGenFunction::EmitARCRetain(QualType type, llvm::Value *value) {
  if (type->isBlockPointerType())
    return EmitARCRetainBlock(value);
  else
    return EmitARCRetainNonBlock(value);
}

/// Retain the given object, with normal retain semantics.
///   call i8* @objc_retain(i8* %value)
llvm::Value *CodeGenFunction::EmitARCRetainNonBlock(llvm::Value *value) {
  return emitARCValueOperation(*this, value,
                               CGM.getARCEntrypoints().objc_retain,
                               "objc_retain");
}

/// Retain the given block, with _Block_copy semantics.
///   call i8* @objc_retainBlock(i8* %value)
llvm::Value *CodeGenFunction::EmitARCRetainBlock(llvm::Value *value) {
  return emitARCValueOperation(*this, value,
                               CGM.getARCEntrypoints().objc_retainBlock,
                               "objc_retainBlock");
}

/// Retain the given object which is the result of a function call.
///   call i8* @objc_retainAutoreleasedReturnValue(i8* %value)
///
/// Yes, this function name is one character away from a different
/// call with completely different semantics.
llvm::Value *
CodeGenFunction::EmitARCRetainAutoreleasedReturnValue(llvm::Value *value) {
  // Fetch the void(void) inline asm which marks that we're going to
  // retain the autoreleased return value.
  llvm::InlineAsm *&marker
    = CGM.getARCEntrypoints().retainAutoreleasedReturnValueMarker;
  if (!marker) {
    StringRef assembly
      = CGM.getTargetCodeGenInfo()
           .getARCRetainAutoreleasedReturnValueMarker();

    // If we have an empty assembly string, there's nothing to do.
    if (assembly.empty()) {

    // Otherwise, at -O0, build an inline asm that we're going to call
    // in a moment.
    } else if (CGM.getCodeGenOpts().OptimizationLevel == 0) {
      llvm::FunctionType *type =
        llvm::FunctionType::get(llvm::Type::getVoidTy(getLLVMContext()),
                                /*variadic*/ false);
      
      marker = llvm::InlineAsm::get(type, assembly, "", /*sideeffects*/ true);

    // If we're at -O1 and above, we don't want to litter the code
    // with this marker yet, so leave a breadcrumb for the ARC
    // optimizer to pick up.
    } else {
      llvm::NamedMDNode *metadata =
        CGM.getModule().getOrInsertNamedMetadata(
                            "clang.arc.retainAutoreleasedReturnValueMarker");
      assert(metadata->getNumOperands() <= 1);
      if (metadata->getNumOperands() == 0) {
        llvm::Value *string = llvm::MDString::get(getLLVMContext(), assembly);
        metadata->addOperand(llvm::MDNode::get(getLLVMContext(), string));
      }
    }
  }

  // Call the marker asm if we made one, which we do only at -O0.
  if (marker) Builder.CreateCall(marker);

  return emitARCValueOperation(*this, value,
                     CGM.getARCEntrypoints().objc_retainAutoreleasedReturnValue,
                               "objc_retainAutoreleasedReturnValue");
}

/// Release the given object.
///   call void @objc_release(i8* %value)
void CodeGenFunction::EmitARCRelease(llvm::Value *value, bool precise) {
  if (isa<llvm::ConstantPointerNull>(value)) return;

  llvm::Constant *&fn = CGM.getARCEntrypoints().objc_release;
  if (!fn) {
    std::vector<llvm::Type*> args(1, Int8PtrTy);
    llvm::FunctionType *fnType =
      llvm::FunctionType::get(Builder.getVoidTy(), args, false);
    fn = createARCRuntimeFunction(CGM, fnType, "objc_release");
  }

  // Cast the argument to 'id'.
  value = Builder.CreateBitCast(value, Int8PtrTy);

  // Call objc_release.
  llvm::CallInst *call = Builder.CreateCall(fn, value);
  call->setDoesNotThrow();

  if (!precise) {
    SmallVector<llvm::Value*,1> args;
    call->setMetadata("clang.imprecise_release",
                      llvm::MDNode::get(Builder.getContext(), args));
  }
}

/// Store into a strong object.  Always calls this:
///   call void @objc_storeStrong(i8** %addr, i8* %value)
llvm::Value *CodeGenFunction::EmitARCStoreStrongCall(llvm::Value *addr,
                                                     llvm::Value *value,
                                                     bool ignored) {
  assert(cast<llvm::PointerType>(addr->getType())->getElementType()
           == value->getType());

  llvm::Constant *&fn = CGM.getARCEntrypoints().objc_storeStrong;
  if (!fn) {
    llvm::Type *argTypes[] = { Int8PtrPtrTy, Int8PtrTy };
    llvm::FunctionType *fnType
      = llvm::FunctionType::get(Builder.getVoidTy(), argTypes, false);
    fn = createARCRuntimeFunction(CGM, fnType, "objc_storeStrong");
  }

  addr = Builder.CreateBitCast(addr, Int8PtrPtrTy);
  llvm::Value *castValue = Builder.CreateBitCast(value, Int8PtrTy);
  
  Builder.CreateCall2(fn, addr, castValue)->setDoesNotThrow();

  if (ignored) return 0;
  return value;
}

/// Store into a strong object.  Sometimes calls this:
///   call void @objc_storeStrong(i8** %addr, i8* %value)
/// Other times, breaks it down into components.
llvm::Value *CodeGenFunction::EmitARCStoreStrong(LValue dst,
                                                 llvm::Value *newValue,
                                                 bool ignored) {
  QualType type = dst.getType();
  bool isBlock = type->isBlockPointerType();

  // Use a store barrier at -O0 unless this is a block type or the
  // lvalue is inadequately aligned.
  if (shouldUseFusedARCCalls() &&
      !isBlock &&
      !(dst.getAlignment() && dst.getAlignment() < PointerAlignInBytes)) {
    return EmitARCStoreStrongCall(dst.getAddress(), newValue, ignored);
  }

  // Otherwise, split it out.

  // Retain the new value.
  newValue = EmitARCRetain(type, newValue);

  // Read the old value.
  llvm::Value *oldValue = EmitLoadOfScalar(dst);

  // Store.  We do this before the release so that any deallocs won't
  // see the old value.
  EmitStoreOfScalar(newValue, dst);

  // Finally, release the old value.
  EmitARCRelease(oldValue, /*precise*/ false);

  return newValue;
}

/// Autorelease the given object.
///   call i8* @objc_autorelease(i8* %value)
llvm::Value *CodeGenFunction::EmitARCAutorelease(llvm::Value *value) {
  return emitARCValueOperation(*this, value,
                               CGM.getARCEntrypoints().objc_autorelease,
                               "objc_autorelease");
}

/// Autorelease the given object.
///   call i8* @objc_autoreleaseReturnValue(i8* %value)
llvm::Value *
CodeGenFunction::EmitARCAutoreleaseReturnValue(llvm::Value *value) {
  return emitARCValueOperation(*this, value,
                            CGM.getARCEntrypoints().objc_autoreleaseReturnValue,
                               "objc_autoreleaseReturnValue");
}

/// Do a fused retain/autorelease of the given object.
///   call i8* @objc_retainAutoreleaseReturnValue(i8* %value)
llvm::Value *
CodeGenFunction::EmitARCRetainAutoreleaseReturnValue(llvm::Value *value) {
  return emitARCValueOperation(*this, value,
                     CGM.getARCEntrypoints().objc_retainAutoreleaseReturnValue,
                               "objc_retainAutoreleaseReturnValue");
}

/// Do a fused retain/autorelease of the given object.
///   call i8* @objc_retainAutorelease(i8* %value)
/// or
///   %retain = call i8* @objc_retainBlock(i8* %value)
///   call i8* @objc_autorelease(i8* %retain)
llvm::Value *CodeGenFunction::EmitARCRetainAutorelease(QualType type,
                                                       llvm::Value *value) {
  if (!type->isBlockPointerType())
    return EmitARCRetainAutoreleaseNonBlock(value);

  if (isa<llvm::ConstantPointerNull>(value)) return value;

  llvm::Type *origType = value->getType();
  value = Builder.CreateBitCast(value, Int8PtrTy);
  value = EmitARCRetainBlock(value);
  value = EmitARCAutorelease(value);
  return Builder.CreateBitCast(value, origType);
}

/// Do a fused retain/autorelease of the given object.
///   call i8* @objc_retainAutorelease(i8* %value)
llvm::Value *
CodeGenFunction::EmitARCRetainAutoreleaseNonBlock(llvm::Value *value) {
  return emitARCValueOperation(*this, value,
                               CGM.getARCEntrypoints().objc_retainAutorelease,
                               "objc_retainAutorelease");
}

/// i8* @objc_loadWeak(i8** %addr)
/// Essentially objc_autorelease(objc_loadWeakRetained(addr)).
llvm::Value *CodeGenFunction::EmitARCLoadWeak(llvm::Value *addr) {
  return emitARCLoadOperation(*this, addr,
                              CGM.getARCEntrypoints().objc_loadWeak,
                              "objc_loadWeak");
}

/// i8* @objc_loadWeakRetained(i8** %addr)
llvm::Value *CodeGenFunction::EmitARCLoadWeakRetained(llvm::Value *addr) {
  return emitARCLoadOperation(*this, addr,
                              CGM.getARCEntrypoints().objc_loadWeakRetained,
                              "objc_loadWeakRetained");
}

/// i8* @objc_storeWeak(i8** %addr, i8* %value)
/// Returns %value.
llvm::Value *CodeGenFunction::EmitARCStoreWeak(llvm::Value *addr,
                                               llvm::Value *value,
                                               bool ignored) {
  return emitARCStoreOperation(*this, addr, value,
                               CGM.getARCEntrypoints().objc_storeWeak,
                               "objc_storeWeak", ignored);
}

/// i8* @objc_initWeak(i8** %addr, i8* %value)
/// Returns %value.  %addr is known to not have a current weak entry.
/// Essentially equivalent to:
///   *addr = nil; objc_storeWeak(addr, value);
void CodeGenFunction::EmitARCInitWeak(llvm::Value *addr, llvm::Value *value) {
  // If we're initializing to null, just write null to memory; no need
  // to get the runtime involved.  But don't do this if optimization
  // is enabled, because accounting for this would make the optimizer
  // much more complicated.
  if (isa<llvm::ConstantPointerNull>(value) &&
      CGM.getCodeGenOpts().OptimizationLevel == 0) {
    Builder.CreateStore(value, addr);
    return;
  }

  emitARCStoreOperation(*this, addr, value,
                        CGM.getARCEntrypoints().objc_initWeak,
                        "objc_initWeak", /*ignored*/ true);
}

/// void @objc_destroyWeak(i8** %addr)
/// Essentially objc_storeWeak(addr, nil).
void CodeGenFunction::EmitARCDestroyWeak(llvm::Value *addr) {
  llvm::Constant *&fn = CGM.getARCEntrypoints().objc_destroyWeak;
  if (!fn) {
    std::vector<llvm::Type*> args(1, Int8PtrPtrTy);
    llvm::FunctionType *fnType =
      llvm::FunctionType::get(Builder.getVoidTy(), args, false);
    fn = createARCRuntimeFunction(CGM, fnType, "objc_destroyWeak");
  }

  // Cast the argument to 'id*'.
  addr = Builder.CreateBitCast(addr, Int8PtrPtrTy);

  llvm::CallInst *call = Builder.CreateCall(fn, addr);
  call->setDoesNotThrow();
}

/// void @objc_moveWeak(i8** %dest, i8** %src)
/// Disregards the current value in %dest.  Leaves %src pointing to nothing.
/// Essentially (objc_copyWeak(dest, src), objc_destroyWeak(src)).
void CodeGenFunction::EmitARCMoveWeak(llvm::Value *dst, llvm::Value *src) {
  emitARCCopyOperation(*this, dst, src,
                       CGM.getARCEntrypoints().objc_moveWeak,
                       "objc_moveWeak");
}

/// void @objc_copyWeak(i8** %dest, i8** %src)
/// Disregards the current value in %dest.  Essentially
///   objc_release(objc_initWeak(dest, objc_readWeakRetained(src)))
void CodeGenFunction::EmitARCCopyWeak(llvm::Value *dst, llvm::Value *src) {
  emitARCCopyOperation(*this, dst, src,
                       CGM.getARCEntrypoints().objc_copyWeak,
                       "objc_copyWeak");
}

/// Produce the code to do a objc_autoreleasepool_push.
///   call i8* @objc_autoreleasePoolPush(void)
llvm::Value *CodeGenFunction::EmitObjCAutoreleasePoolPush() {
  llvm::Constant *&fn = CGM.getRREntrypoints().objc_autoreleasePoolPush;
  if (!fn) {
    llvm::FunctionType *fnType =
      llvm::FunctionType::get(Int8PtrTy, false);
    fn = createARCRuntimeFunction(CGM, fnType, "objc_autoreleasePoolPush");
  }

  llvm::CallInst *call = Builder.CreateCall(fn);
  call->setDoesNotThrow();

  return call;
}

/// Produce the code to do a primitive release.
///   call void @objc_autoreleasePoolPop(i8* %ptr)
void CodeGenFunction::EmitObjCAutoreleasePoolPop(llvm::Value *value) {
  assert(value->getType() == Int8PtrTy);

  llvm::Constant *&fn = CGM.getRREntrypoints().objc_autoreleasePoolPop;
  if (!fn) {
    std::vector<llvm::Type*> args(1, Int8PtrTy);
    llvm::FunctionType *fnType =
      llvm::FunctionType::get(Builder.getVoidTy(), args, false);

    // We don't want to use a weak import here; instead we should not
    // fall into this path.
    fn = createARCRuntimeFunction(CGM, fnType, "objc_autoreleasePoolPop");
  }

  llvm::CallInst *call = Builder.CreateCall(fn, value);
  call->setDoesNotThrow();
}

/// Produce the code to do an MRR version objc_autoreleasepool_push.
/// Which is: [[NSAutoreleasePool alloc] init];
/// Where alloc is declared as: + (id) alloc; in NSAutoreleasePool class.
/// init is declared as: - (id) init; in its NSObject super class.
///
llvm::Value *CodeGenFunction::EmitObjCMRRAutoreleasePoolPush() {
  CGObjCRuntime &Runtime = CGM.getObjCRuntime();
  llvm::Value *Receiver = Runtime.EmitNSAutoreleasePoolClassRef(Builder);
  // [NSAutoreleasePool alloc]
  IdentifierInfo *II = &CGM.getContext().Idents.get("alloc");
  Selector AllocSel = getContext().Selectors.getSelector(0, &II);
  CallArgList Args;
  RValue AllocRV =  
    Runtime.GenerateMessageSend(*this, ReturnValueSlot(), 
                                getContext().getObjCIdType(),
                                AllocSel, Receiver, Args); 

  // [Receiver init]
  Receiver = AllocRV.getScalarVal();
  II = &CGM.getContext().Idents.get("init");
  Selector InitSel = getContext().Selectors.getSelector(0, &II);
  RValue InitRV =
    Runtime.GenerateMessageSend(*this, ReturnValueSlot(),
                                getContext().getObjCIdType(),
                                InitSel, Receiver, Args); 
  return InitRV.getScalarVal();
}

/// Produce the code to do a primitive release.
/// [tmp drain];
void CodeGenFunction::EmitObjCMRRAutoreleasePoolPop(llvm::Value *Arg) {
  IdentifierInfo *II = &CGM.getContext().Idents.get("drain");
  Selector DrainSel = getContext().Selectors.getSelector(0, &II);
  CallArgList Args;
  CGM.getObjCRuntime().GenerateMessageSend(*this, ReturnValueSlot(),
                              getContext().VoidTy, DrainSel, Arg, Args); 
}

void CodeGenFunction::destroyARCStrongPrecise(CodeGenFunction &CGF,
                                              llvm::Value *addr,
                                              QualType type) {
  llvm::Value *ptr = CGF.Builder.CreateLoad(addr, "strongdestroy");
  CGF.EmitARCRelease(ptr, /*precise*/ true);
}

void CodeGenFunction::destroyARCStrongImprecise(CodeGenFunction &CGF,
                                                llvm::Value *addr,
                                                QualType type) {
  llvm::Value *ptr = CGF.Builder.CreateLoad(addr, "strongdestroy");
  CGF.EmitARCRelease(ptr, /*precise*/ false);  
}

void CodeGenFunction::destroyARCWeak(CodeGenFunction &CGF,
                                     llvm::Value *addr,
                                     QualType type) {
  CGF.EmitARCDestroyWeak(addr);
}

namespace {
  struct CallObjCAutoreleasePoolObject : EHScopeStack::Cleanup {
    llvm::Value *Token;

    CallObjCAutoreleasePoolObject(llvm::Value *token) : Token(token) {}

    void Emit(CodeGenFunction &CGF, Flags flags) {
      CGF.EmitObjCAutoreleasePoolPop(Token);
    }
  };
  struct CallObjCMRRAutoreleasePoolObject : EHScopeStack::Cleanup {
    llvm::Value *Token;

    CallObjCMRRAutoreleasePoolObject(llvm::Value *token) : Token(token) {}

    void Emit(CodeGenFunction &CGF, Flags flags) {
      CGF.EmitObjCMRRAutoreleasePoolPop(Token);
    }
  };
}

void CodeGenFunction::EmitObjCAutoreleasePoolCleanup(llvm::Value *Ptr) {
  if (CGM.getLangOptions().ObjCAutoRefCount)
    EHStack.pushCleanup<CallObjCAutoreleasePoolObject>(NormalCleanup, Ptr);
  else
    EHStack.pushCleanup<CallObjCMRRAutoreleasePoolObject>(NormalCleanup, Ptr);
}

static TryEmitResult tryEmitARCRetainLoadOfScalar(CodeGenFunction &CGF,
                                                  LValue lvalue,
                                                  QualType type) {
  switch (type.getObjCLifetime()) {
  case Qualifiers::OCL_None:
  case Qualifiers::OCL_ExplicitNone:
  case Qualifiers::OCL_Strong:
  case Qualifiers::OCL_Autoreleasing:
    return TryEmitResult(CGF.EmitLoadOfLValue(lvalue).getScalarVal(),
                         false);

  case Qualifiers::OCL_Weak:
    return TryEmitResult(CGF.EmitARCLoadWeakRetained(lvalue.getAddress()),
                         true);
  }

  llvm_unreachable("impossible lifetime!");
  return TryEmitResult();
}

static TryEmitResult tryEmitARCRetainLoadOfScalar(CodeGenFunction &CGF,
                                                  const Expr *e) {
  e = e->IgnoreParens();
  QualType type = e->getType();

  // As a very special optimization, in ARC++, if the l-value is the
  // result of a non-volatile assignment, do a simple retain of the
  // result of the call to objc_storeWeak instead of reloading.
  if (CGF.getLangOptions().CPlusPlus &&
      !type.isVolatileQualified() &&
      type.getObjCLifetime() == Qualifiers::OCL_Weak &&
      isa<BinaryOperator>(e) &&
      cast<BinaryOperator>(e)->getOpcode() == BO_Assign)
    return TryEmitResult(CGF.EmitScalarExpr(e), false);

  return tryEmitARCRetainLoadOfScalar(CGF, CGF.EmitLValue(e), type);
}

static llvm::Value *emitARCRetainAfterCall(CodeGenFunction &CGF,
                                           llvm::Value *value);

/// Given that the given expression is some sort of call (which does
/// not return retained), emit a retain following it.
static llvm::Value *emitARCRetainCall(CodeGenFunction &CGF, const Expr *e) {
  llvm::Value *value = CGF.EmitScalarExpr(e);
  return emitARCRetainAfterCall(CGF, value);
}

static llvm::Value *emitARCRetainAfterCall(CodeGenFunction &CGF,
                                           llvm::Value *value) {
  if (llvm::CallInst *call = dyn_cast<llvm::CallInst>(value)) {
    CGBuilderTy::InsertPoint ip = CGF.Builder.saveIP();

    // Place the retain immediately following the call.
    CGF.Builder.SetInsertPoint(call->getParent(),
                               ++llvm::BasicBlock::iterator(call));
    value = CGF.EmitARCRetainAutoreleasedReturnValue(value);

    CGF.Builder.restoreIP(ip);
    return value;
  } else if (llvm::InvokeInst *invoke = dyn_cast<llvm::InvokeInst>(value)) {
    CGBuilderTy::InsertPoint ip = CGF.Builder.saveIP();

    // Place the retain at the beginning of the normal destination block.
    llvm::BasicBlock *BB = invoke->getNormalDest();
    CGF.Builder.SetInsertPoint(BB, BB->begin());
    value = CGF.EmitARCRetainAutoreleasedReturnValue(value);

    CGF.Builder.restoreIP(ip);
    return value;

  // Bitcasts can arise because of related-result returns.  Rewrite
  // the operand.
  } else if (llvm::BitCastInst *bitcast = dyn_cast<llvm::BitCastInst>(value)) {
    llvm::Value *operand = bitcast->getOperand(0);
    operand = emitARCRetainAfterCall(CGF, operand);
    bitcast->setOperand(0, operand);
    return bitcast;

  // Generic fall-back case.
  } else {
    // Retain using the non-block variant: we never need to do a copy
    // of a block that's been returned to us.
    return CGF.EmitARCRetainNonBlock(value);
  }
}

static TryEmitResult
tryEmitARCRetainScalarExpr(CodeGenFunction &CGF, const Expr *e) {
  // Look through cleanups.
  if (const ExprWithCleanups *cleanups = dyn_cast<ExprWithCleanups>(e)) {
    CodeGenFunction::RunCleanupsScope scope(CGF);
    return tryEmitARCRetainScalarExpr(CGF, cleanups->getSubExpr());
  }

  // The desired result type, if it differs from the type of the
  // ultimate opaque expression.
  llvm::Type *resultType = 0;

  // If we're loading retained from a __strong xvalue, we can avoid 
  // an extra retain/release pair by zeroing out the source of this
  // "move" operation.
  if (e->isXValue() && !e->getType().isConstQualified() &&
      e->getType().getObjCLifetime() == Qualifiers::OCL_Strong) {
    // Emit the lvalue
    LValue lv = CGF.EmitLValue(e);
    
    // Load the object pointer and cast it to the appropriate type.
    QualType exprType = e->getType();
    llvm::Value *result = CGF.EmitLoadOfLValue(lv).getScalarVal();
    
    if (resultType)
      result = CGF.Builder.CreateBitCast(result, resultType);
    
    // Set the source pointer to NULL.
    llvm::Value *null 
      = llvm::ConstantPointerNull::get(
                            cast<llvm::PointerType>(CGF.ConvertType(exprType)));
    CGF.EmitStoreOfScalar(null, lv);
    
    return TryEmitResult(result, true);
  }

  while (true) {
    e = e->IgnoreParens();

    // There's a break at the end of this if-chain;  anything
    // that wants to keep looping has to explicitly continue.
    if (const CastExpr *ce = dyn_cast<CastExpr>(e)) {
      switch (ce->getCastKind()) {
      // No-op casts don't change the type, so we just ignore them.
      case CK_NoOp:
        e = ce->getSubExpr();
        continue;

      case CK_LValueToRValue: {
        TryEmitResult loadResult
          = tryEmitARCRetainLoadOfScalar(CGF, ce->getSubExpr());
        if (resultType) {
          llvm::Value *value = loadResult.getPointer();
          value = CGF.Builder.CreateBitCast(value, resultType);
          loadResult.setPointer(value);
        }
        return loadResult;
      }

      // These casts can change the type, so remember that and
      // soldier on.  We only need to remember the outermost such
      // cast, though.
      case CK_AnyPointerToObjCPointerCast:
      case CK_AnyPointerToBlockPointerCast:
      case CK_BitCast:
        if (!resultType)
          resultType = CGF.ConvertType(ce->getType());
        e = ce->getSubExpr();
        assert(e->getType()->hasPointerRepresentation());
        continue;

      // For consumptions, just emit the subexpression and thus elide
      // the retain/release pair.
      case CK_ObjCConsumeObject: {
        llvm::Value *result = CGF.EmitScalarExpr(ce->getSubExpr());
        if (resultType) result = CGF.Builder.CreateBitCast(result, resultType);
        return TryEmitResult(result, true);
      }

      // For reclaims, emit the subexpression as a retained call and
      // skip the consumption.
      case CK_ObjCReclaimReturnedObject: {
        llvm::Value *result = emitARCRetainCall(CGF, ce->getSubExpr());
        if (resultType) result = CGF.Builder.CreateBitCast(result, resultType);
        return TryEmitResult(result, true);
      }

      case CK_GetObjCProperty: {
        llvm::Value *result = emitARCRetainCall(CGF, ce);
        if (resultType) result = CGF.Builder.CreateBitCast(result, resultType);
        return TryEmitResult(result, true);
      }

      default:
        break;
      }

    // Skip __extension__.
    } else if (const UnaryOperator *op = dyn_cast<UnaryOperator>(e)) {
      if (op->getOpcode() == UO_Extension) {
        e = op->getSubExpr();
        continue;
      }

    // For calls and message sends, use the retained-call logic.
    // Delegate inits are a special case in that they're the only
    // returns-retained expression that *isn't* surrounded by
    // a consume.
    } else if (isa<CallExpr>(e) ||
               (isa<ObjCMessageExpr>(e) &&
                !cast<ObjCMessageExpr>(e)->isDelegateInitCall())) {
      llvm::Value *result = emitARCRetainCall(CGF, e);
      if (resultType) result = CGF.Builder.CreateBitCast(result, resultType);
      return TryEmitResult(result, true);
    }

    // Conservatively halt the search at any other expression kind.
    break;
  }

  // We didn't find an obvious production, so emit what we've got and
  // tell the caller that we didn't manage to retain.
  llvm::Value *result = CGF.EmitScalarExpr(e);
  if (resultType) result = CGF.Builder.CreateBitCast(result, resultType);
  return TryEmitResult(result, false);
}

static llvm::Value *emitARCRetainLoadOfScalar(CodeGenFunction &CGF,
                                                LValue lvalue,
                                                QualType type) {
  TryEmitResult result = tryEmitARCRetainLoadOfScalar(CGF, lvalue, type);
  llvm::Value *value = result.getPointer();
  if (!result.getInt())
    value = CGF.EmitARCRetain(type, value);
  return value;
}

/// EmitARCRetainScalarExpr - Semantically equivalent to
/// EmitARCRetainObject(e->getType(), EmitScalarExpr(e)), but making a
/// best-effort attempt to peephole expressions that naturally produce
/// retained objects.
llvm::Value *CodeGenFunction::EmitARCRetainScalarExpr(const Expr *e) {
  TryEmitResult result = tryEmitARCRetainScalarExpr(*this, e);
  llvm::Value *value = result.getPointer();
  if (!result.getInt())
    value = EmitARCRetain(e->getType(), value);
  return value;
}

llvm::Value *
CodeGenFunction::EmitARCRetainAutoreleaseScalarExpr(const Expr *e) {
  TryEmitResult result = tryEmitARCRetainScalarExpr(*this, e);
  llvm::Value *value = result.getPointer();
  if (result.getInt())
    value = EmitARCAutorelease(value);
  else
    value = EmitARCRetainAutorelease(e->getType(), value);
  return value;
}

std::pair<LValue,llvm::Value*>
CodeGenFunction::EmitARCStoreStrong(const BinaryOperator *e,
                                    bool ignored) {
  // Evaluate the RHS first.
  TryEmitResult result = tryEmitARCRetainScalarExpr(*this, e->getRHS());
  llvm::Value *value = result.getPointer();

  bool hasImmediateRetain = result.getInt();

  // If we didn't emit a retained object, and the l-value is of block
  // type, then we need to emit the block-retain immediately in case
  // it invalidates the l-value.
  if (!hasImmediateRetain && e->getType()->isBlockPointerType()) {
    value = EmitARCRetainBlock(value);
    hasImmediateRetain = true;
  }

  LValue lvalue = EmitLValue(e->getLHS());

  // If the RHS was emitted retained, expand this.
  if (hasImmediateRetain) {
    llvm::Value *oldValue =
      EmitLoadOfScalar(lvalue.getAddress(), lvalue.isVolatileQualified(),
                       lvalue.getAlignment(), e->getType(),
                       lvalue.getTBAAInfo());
    EmitStoreOfScalar(value, lvalue.getAddress(),
                      lvalue.isVolatileQualified(), lvalue.getAlignment(),
                      e->getType(), lvalue.getTBAAInfo());
    EmitARCRelease(oldValue, /*precise*/ false);
  } else {
    value = EmitARCStoreStrong(lvalue, value, ignored);
  }

  return std::pair<LValue,llvm::Value*>(lvalue, value);
}

std::pair<LValue,llvm::Value*>
CodeGenFunction::EmitARCStoreAutoreleasing(const BinaryOperator *e) {
  llvm::Value *value = EmitARCRetainAutoreleaseScalarExpr(e->getRHS());
  LValue lvalue = EmitLValue(e->getLHS());

  EmitStoreOfScalar(value, lvalue.getAddress(),
                    lvalue.isVolatileQualified(), lvalue.getAlignment(),
                    e->getType(), lvalue.getTBAAInfo());

  return std::pair<LValue,llvm::Value*>(lvalue, value);
}

void CodeGenFunction::EmitObjCAutoreleasePoolStmt(
                                             const ObjCAutoreleasePoolStmt &ARPS) {
  const Stmt *subStmt = ARPS.getSubStmt();
  const CompoundStmt &S = cast<CompoundStmt>(*subStmt);

  CGDebugInfo *DI = getDebugInfo();
  if (DI) {
    DI->setLocation(S.getLBracLoc());
    DI->EmitRegionStart(Builder);
  }

  // Keep track of the current cleanup stack depth.
  RunCleanupsScope Scope(*this);
  if (CGM.getCodeGenOpts().ObjCRuntimeHasARC) {
    llvm::Value *token = EmitObjCAutoreleasePoolPush();
    EHStack.pushCleanup<CallObjCAutoreleasePoolObject>(NormalCleanup, token);
  } else {
    llvm::Value *token = EmitObjCMRRAutoreleasePoolPush();
    EHStack.pushCleanup<CallObjCMRRAutoreleasePoolObject>(NormalCleanup, token);
  }

  for (CompoundStmt::const_body_iterator I = S.body_begin(),
       E = S.body_end(); I != E; ++I)
    EmitStmt(*I);

  if (DI) {
    DI->setLocation(S.getRBracLoc());
    DI->EmitRegionEnd(Builder);
  }
}

/// EmitExtendGCLifetime - Given a pointer to an Objective-C object,
/// make sure it survives garbage collection until this point.
void CodeGenFunction::EmitExtendGCLifetime(llvm::Value *object) {
  // We just use an inline assembly.
  llvm::FunctionType *extenderType
    = llvm::FunctionType::get(VoidTy, VoidPtrTy, /*variadic*/ false);
  llvm::Value *extender
    = llvm::InlineAsm::get(extenderType,
                           /* assembly */ "",
                           /* constraints */ "r",
                           /* side effects */ true);

  object = Builder.CreateBitCast(object, VoidPtrTy);
  Builder.CreateCall(extender, object)->setDoesNotThrow();
}

CGObjCRuntime::~CGObjCRuntime() {}
