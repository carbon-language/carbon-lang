//===--- SemaDeclAttr.cpp - Declaration Attribute Handling ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements decl-related attribute processing.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaInternal.h"
#include "TargetAttributesSema.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/DelayedDiagnostic.h"
#include "llvm/ADT/StringExtras.h"
using namespace clang;
using namespace sema;

/// These constants match the enumerated choices of
/// warn_attribute_wrong_decl_type and err_attribute_wrong_decl_type.
enum AttributeDeclKind {
  ExpectedFunction,
  ExpectedUnion,
  ExpectedVariableOrFunction,
  ExpectedFunctionOrMethod,
  ExpectedParameter,
  ExpectedParameterOrMethod,
  ExpectedFunctionMethodOrBlock,
  ExpectedClassOrVirtualMethod,
  ExpectedFunctionMethodOrParameter,
  ExpectedClass,
  ExpectedVirtualMethod,
  ExpectedClassMember,
  ExpectedVariable,
  ExpectedMethod,
  ExpectedVariableFunctionOrLabel,
  ExpectedFieldOrGlobalVar
};

//===----------------------------------------------------------------------===//
//  Helper functions
//===----------------------------------------------------------------------===//

static const FunctionType *getFunctionType(const Decl *D,
                                           bool blocksToo = true) {
  QualType Ty;
  if (const ValueDecl *decl = dyn_cast<ValueDecl>(D))
    Ty = decl->getType();
  else if (const FieldDecl *decl = dyn_cast<FieldDecl>(D))
    Ty = decl->getType();
  else if (const TypedefNameDecl* decl = dyn_cast<TypedefNameDecl>(D))
    Ty = decl->getUnderlyingType();
  else
    return 0;

  if (Ty->isFunctionPointerType())
    Ty = Ty->getAs<PointerType>()->getPointeeType();
  else if (blocksToo && Ty->isBlockPointerType())
    Ty = Ty->getAs<BlockPointerType>()->getPointeeType();

  return Ty->getAs<FunctionType>();
}

// FIXME: We should provide an abstraction around a method or function
// to provide the following bits of information.

/// isFunction - Return true if the given decl has function
/// type (function or function-typed variable).
static bool isFunction(const Decl *D) {
  return getFunctionType(D, false) != NULL;
}

/// isFunctionOrMethod - Return true if the given decl has function
/// type (function or function-typed variable) or an Objective-C
/// method.
static bool isFunctionOrMethod(const Decl *D) {
  return isFunction(D)|| isa<ObjCMethodDecl>(D);
}

/// isFunctionOrMethodOrBlock - Return true if the given decl has function
/// type (function or function-typed variable) or an Objective-C
/// method or a block.
static bool isFunctionOrMethodOrBlock(const Decl *D) {
  if (isFunctionOrMethod(D))
    return true;
  // check for block is more involved.
  if (const VarDecl *V = dyn_cast<VarDecl>(D)) {
    QualType Ty = V->getType();
    return Ty->isBlockPointerType();
  }
  return isa<BlockDecl>(D);
}

/// Return true if the given decl has a declarator that should have
/// been processed by Sema::GetTypeForDeclarator.
static bool hasDeclarator(const Decl *D) {
  // In some sense, TypedefDecl really *ought* to be a DeclaratorDecl.
  return isa<DeclaratorDecl>(D) || isa<BlockDecl>(D) || isa<TypedefNameDecl>(D) ||
         isa<ObjCPropertyDecl>(D);
}

/// hasFunctionProto - Return true if the given decl has a argument
/// information. This decl should have already passed
/// isFunctionOrMethod or isFunctionOrMethodOrBlock.
static bool hasFunctionProto(const Decl *D) {
  if (const FunctionType *FnTy = getFunctionType(D))
    return isa<FunctionProtoType>(FnTy);
  else {
    assert(isa<ObjCMethodDecl>(D) || isa<BlockDecl>(D));
    return true;
  }
}

/// getFunctionOrMethodNumArgs - Return number of function or method
/// arguments. It is an error to call this on a K&R function (use
/// hasFunctionProto first).
static unsigned getFunctionOrMethodNumArgs(const Decl *D) {
  if (const FunctionType *FnTy = getFunctionType(D))
    return cast<FunctionProtoType>(FnTy)->getNumArgs();
  if (const BlockDecl *BD = dyn_cast<BlockDecl>(D))
    return BD->getNumParams();
  return cast<ObjCMethodDecl>(D)->param_size();
}

static QualType getFunctionOrMethodArgType(const Decl *D, unsigned Idx) {
  if (const FunctionType *FnTy = getFunctionType(D))
    return cast<FunctionProtoType>(FnTy)->getArgType(Idx);
  if (const BlockDecl *BD = dyn_cast<BlockDecl>(D))
    return BD->getParamDecl(Idx)->getType();

  return cast<ObjCMethodDecl>(D)->param_begin()[Idx]->getType();
}

static QualType getFunctionOrMethodResultType(const Decl *D) {
  if (const FunctionType *FnTy = getFunctionType(D))
    return cast<FunctionProtoType>(FnTy)->getResultType();
  return cast<ObjCMethodDecl>(D)->getResultType();
}

static bool isFunctionOrMethodVariadic(const Decl *D) {
  if (const FunctionType *FnTy = getFunctionType(D)) {
    const FunctionProtoType *proto = cast<FunctionProtoType>(FnTy);
    return proto->isVariadic();
  } else if (const BlockDecl *BD = dyn_cast<BlockDecl>(D))
    return BD->isVariadic();
  else {
    return cast<ObjCMethodDecl>(D)->isVariadic();
  }
}

static bool isInstanceMethod(const Decl *D) {
  if (const CXXMethodDecl *MethodDecl = dyn_cast<CXXMethodDecl>(D))
    return MethodDecl->isInstance();
  return false;
}

static inline bool isNSStringType(QualType T, ASTContext &Ctx) {
  const ObjCObjectPointerType *PT = T->getAs<ObjCObjectPointerType>();
  if (!PT)
    return false;

  ObjCInterfaceDecl *Cls = PT->getObjectType()->getInterface();
  if (!Cls)
    return false;

  IdentifierInfo* ClsName = Cls->getIdentifier();

  // FIXME: Should we walk the chain of classes?
  return ClsName == &Ctx.Idents.get("NSString") ||
         ClsName == &Ctx.Idents.get("NSMutableString");
}

static inline bool isCFStringType(QualType T, ASTContext &Ctx) {
  const PointerType *PT = T->getAs<PointerType>();
  if (!PT)
    return false;

  const RecordType *RT = PT->getPointeeType()->getAs<RecordType>();
  if (!RT)
    return false;

  const RecordDecl *RD = RT->getDecl();
  if (RD->getTagKind() != TTK_Struct)
    return false;

  return RD->getIdentifier() == &Ctx.Idents.get("__CFString");
}

/// \brief Check if the attribute has exactly as many args as Num. May
/// output an error.
static bool checkAttributeNumArgs(Sema &S, const AttributeList &Attr,
                                  unsigned int Num) {
  if (Attr.getNumArgs() != Num) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << Num;
    return false;
  }

  return true;
}


/// \brief Check if the attribute has at least as many args as Num. May
/// output an error.
static bool checkAttributeAtLeastNumArgs(Sema &S, const AttributeList &Attr,
                                  unsigned int Num) {
  if (Attr.getNumArgs() < Num) {
    S.Diag(Attr.getLoc(), diag::err_attribute_too_few_arguments) << Num;
    return false;
  }

  return true;
}

///
/// \brief Check if passed in Decl is a field or potentially shared global var
/// \return true if the Decl is a field or potentially shared global variable
///
static bool mayBeSharedVariable(const Decl *D) {
  if (isa<FieldDecl>(D))
    return true;
  if (const VarDecl *vd = dyn_cast<VarDecl>(D))
    return (vd->hasGlobalStorage() && !(vd->isThreadSpecified()));

  return false;
}

/// \brief Check if the passed-in expression is of type int or bool.
static bool isIntOrBool(Expr *Exp) {
  QualType QT = Exp->getType();
  return QT->isBooleanType() || QT->isIntegerType();
}

///
/// \brief Check if passed in Decl is a pointer type.
/// Note that this function may produce an error message.
/// \return true if the Decl is a pointer type; false otherwise
///
static bool checkIsPointer(Sema &S, const Decl *D, const AttributeList &Attr) {
  if (const ValueDecl *vd = dyn_cast<ValueDecl>(D)) {
    QualType QT = vd->getType();
    if (QT->isAnyPointerType())
      return true;
    S.Diag(Attr.getLoc(), diag::warn_pointer_attribute_wrong_type)
      << Attr.getName()->getName() << QT;
  } else {
    S.Diag(Attr.getLoc(), diag::err_attribute_can_be_applied_only_to_value_decl)
      << Attr.getName();
  }
  return false;
}

/// \brief Checks that the passed in QualType either is of RecordType or points
/// to RecordType. Returns the relevant RecordType, null if it does not exit.
const RecordType *getRecordType(QualType QT) {
    const RecordType *RT = QT->getAs<RecordType>();
    // now check if we point to record type
    if(!RT && QT->isPointerType()){
        QualType PT = QT->getAs<PointerType>()->getPointeeType();
        RT = PT->getAs<RecordType>();
    }
    return RT;
}

/// \brief Thread Safety Analysis: Checks that all attribute arguments, starting
/// from Sidx, resolve to a lockable object. May flag an error.
static bool checkAttrArgsAreLockableObjs(Sema & S, Decl *D,
                                         const AttributeList & Attr,
                                         int Sidx = 0,
                                         bool ParamIdxOk = false) {
  for(unsigned int Idx = Sidx; Idx < Attr.getNumArgs(); ++Idx) {
    Expr *ArgExp = Attr.getArg(Idx);
    if (ArgExp->isTypeDependent())
      continue;

    QualType Arg_QT = ArgExp->getType();

    // Get record type.
    // first see if we can just cast to record type, or point to record type
    const RecordType *RT = getRecordType(Arg_QT);

    // now check if we idx into a record type function param
    if (!RT && ParamIdxOk) {
      FunctionDecl *FD = dyn_cast <FunctionDecl>(D);
      IntegerLiteral *IL = dyn_cast<IntegerLiteral>(ArgExp);
      if(FD && IL) {
        unsigned int NumParams = FD->getNumParams();
        llvm::APInt ArgValue = IL->getValue();
        uint64_t ParamIdx_from1 = ArgValue.getZExtValue();
        uint64_t ParamIdx_from0 = ParamIdx_from1 - 1;
        if(!ArgValue.isStrictlyPositive() || ParamIdx_from1 > NumParams) {
          S.Diag(Attr.getLoc(), diag::err_attribute_argument_out_of_range)
            << Attr.getName() << Idx + 1 << NumParams;
          return false;
        }
        Arg_QT = FD->getParamDecl(ParamIdx_from0)->getType();
        RT = getRecordType(Arg_QT);
      }
    }

    //  Flag error if could not get record type for this argument
    if (!RT) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_not_class)
        << Attr.getName();
      return false;
    }

    // Flag error if the type is not lockable
    if (!RT->getDecl()->getAttr<LockableAttr>()) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_not_lockable)
        << Attr.getName();
      return false;
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Attribute Implementations
//===----------------------------------------------------------------------===//

// FIXME: All this manual attribute parsing code is gross. At the
// least add some helper functions to check most argument patterns (#
// and types of args).

static void handleGuardedVarAttr(Sema &S, Decl *D, const AttributeList &Attr,
                                 bool pointer = false) {
  assert(!Attr.isInvalid());

  if (!checkAttributeNumArgs(S, Attr, 0))
    return;

  // D must be either a member field or global (potentially shared) variable.
  if (!mayBeSharedVariable(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFieldOrGlobalVar;
    return;
  }

  if (pointer && !checkIsPointer(S, D, Attr))
    return;

  if (pointer)
    D->addAttr(::new (S.Context) PtGuardedVarAttr(Attr.getLoc(), S.Context));
  else
    D->addAttr(::new (S.Context) GuardedVarAttr(Attr.getLoc(), S.Context));
}

static void handleGuardedByAttr(Sema &S, Decl *D, const AttributeList &Attr,
                                bool pointer = false) {
  assert(!Attr.isInvalid());

  if (!checkAttributeNumArgs(S, Attr, 1))
    return;

  // D must be either a member field or global (potentially shared) variable.
  if (!mayBeSharedVariable(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFieldOrGlobalVar;
    return;
  }

  if (pointer && !checkIsPointer(S, D, Attr))
    return;

  // check that all arguments are lockable objects
  if (!checkAttrArgsAreLockableObjs(S, D, Attr))
    return;

  if (pointer)
    D->addAttr(::new (S.Context) PtGuardedByAttr(Attr.getLoc(), S.Context));
  else
    D->addAttr(::new (S.Context) GuardedByAttr(Attr.getLoc(), S.Context));
}


static void handleLockableAttr(Sema &S, Decl *D, const AttributeList &Attr,
                               bool scoped = false) {
  assert(!Attr.isInvalid());

  if (!checkAttributeNumArgs(S, Attr, 0))
    return;

  if (!isa<CXXRecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedClass;
    return;
  }

  if (scoped)
    D->addAttr(::new (S.Context) ScopedLockableAttr(Attr.getLoc(), S.Context));
  else
    D->addAttr(::new (S.Context) LockableAttr(Attr.getLoc(), S.Context));
}

static void handleNoThreadSafetyAttr(Sema &S, Decl *D,
                                     const AttributeList &Attr) {
  assert(!Attr.isInvalid());

  if (!checkAttributeNumArgs(S, Attr, 0))
    return;

  if (!isa<FunctionDecl>(D) && !isa<FunctionTemplateDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionOrMethod;
    return;
  }

  D->addAttr(::new (S.Context) NoThreadSafetyAnalysisAttr(Attr.getLoc(),
                                                          S.Context));
}

static void handleAcquireOrderAttr(Sema &S, Decl *D, const AttributeList &Attr,
                                   bool before) {
  assert(!Attr.isInvalid());

  if (!checkAttributeAtLeastNumArgs(S, Attr, 1))
    return;

  // D must be either a member field or global (potentially shared) variable.
  ValueDecl *VD = dyn_cast<ValueDecl>(D);
  if (!VD || !mayBeSharedVariable(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFieldOrGlobalVar;
    return;
  }

  // Check that this attribute only applies to lockable types
  QualType QT = VD->getType();
  if (!QT->isDependentType()) {
    const RecordType *RT = getRecordType(QT);
    if (!RT || !RT->getDecl()->getAttr<LockableAttr>()) {
      S.Diag(Attr.getLoc(), diag::err_attribute_decl_not_lockable)
              << Attr.getName();
      return;
    }
  }

  // check that all arguments are lockable objects
  if (!checkAttrArgsAreLockableObjs(S, D, Attr))
    return;

  if (before)
    D->addAttr(::new (S.Context) AcquiredBeforeAttr(Attr.getLoc(), S.Context));
  else
    D->addAttr(::new (S.Context) AcquiredAfterAttr(Attr.getLoc(), S.Context));
}

static void handleLockFunAttr(Sema &S, Decl *D, const AttributeList &Attr,
                              bool exclusive = false) {
  assert(!Attr.isInvalid());

  // zero or more arguments ok

  // check that the attribute is applied to a function
  if (!isa<FunctionDecl>(D) && !isa<FunctionTemplateDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionOrMethod;
    return;
  }

  // check that all arguments are lockable objects
  if (!checkAttrArgsAreLockableObjs(S, D, Attr, 0, /*ParamIdxOk=*/true))
    return;

  if (exclusive)
    D->addAttr(::new (S.Context) ExclusiveLockFunctionAttr(Attr.getLoc(),
                                                           S.Context));
  else
    D->addAttr(::new (S.Context) SharedLockFunctionAttr(Attr.getLoc(),
                                                        S.Context));
}

static void handleTrylockFunAttr(Sema &S, Decl *D, const AttributeList &Attr,
                                 bool exclusive = false) {
  assert(!Attr.isInvalid());

  if (!checkAttributeAtLeastNumArgs(S, Attr, 1))
    return;


  if (!isa<FunctionDecl>(D) && !isa<FunctionTemplateDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionOrMethod;
    return;
  }

  if (!isIntOrBool(Attr.getArg(0))) {
    S.Diag(Attr.getLoc(), diag::err_attribute_first_argument_not_int_or_bool)
        << Attr.getName();
    return;
  }

  // check that all arguments are lockable objects
  if (!checkAttrArgsAreLockableObjs(S, D, Attr, 1))
    return;

  if (exclusive)
    D->addAttr(::new (S.Context) ExclusiveTrylockFunctionAttr(Attr.getLoc(),
                                                              S.Context));
  else
    D->addAttr(::new (S.Context) SharedTrylockFunctionAttr(Attr.getLoc(),
                                                           S.Context));
}

static void handleLocksRequiredAttr(Sema &S, Decl *D, const AttributeList &Attr,
                                    bool exclusive = false) {
  assert(!Attr.isInvalid());

  if (!checkAttributeAtLeastNumArgs(S, Attr, 1))
    return;

  if (!isa<FunctionDecl>(D) && !isa<FunctionTemplateDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionOrMethod;
    return;
  }

  // check that all arguments are lockable objects
  if (!checkAttrArgsAreLockableObjs(S, D, Attr))
    return;

  if (exclusive)
    D->addAttr(::new (S.Context) ExclusiveLocksRequiredAttr(Attr.getLoc(),
                                                            S.Context));
  else
    D->addAttr(::new (S.Context) SharedLocksRequiredAttr(Attr.getLoc(),
                                                         S.Context));
}

static void handleUnlockFunAttr(Sema &S, Decl *D,
                                const AttributeList &Attr) {
  assert(!Attr.isInvalid());

  // zero or more arguments ok

  if (!isa<FunctionDecl>(D) && !isa<FunctionTemplateDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionOrMethod;
    return;
  }

  // check that all arguments are lockable objects
  if (!checkAttrArgsAreLockableObjs(S, D, Attr, 0, /*ParamIdxOk=*/true))
    return;

  D->addAttr(::new (S.Context) UnlockFunctionAttr(Attr.getLoc(), S.Context));
}

static void handleLockReturnedAttr(Sema &S, Decl *D,
                                   const AttributeList &Attr) {
  assert(!Attr.isInvalid());

  if (!checkAttributeNumArgs(S, Attr, 1))
    return;

  if (!isa<FunctionDecl>(D) && !isa<FunctionTemplateDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionOrMethod;
    return;
  }

  // check that all arguments are lockable objects
  if (!checkAttrArgsAreLockableObjs(S, D, Attr))
    return;

  D->addAttr(::new (S.Context) LockReturnedAttr(Attr.getLoc(), S.Context));
}

static void handleLocksExcludedAttr(Sema &S, Decl *D,
                                    const AttributeList &Attr) {
  assert(!Attr.isInvalid());

  if (!checkAttributeAtLeastNumArgs(S, Attr, 1))
    return;

  if (!isa<FunctionDecl>(D) && !isa<FunctionTemplateDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionOrMethod;
    return;
  }

  // check that all arguments are lockable objects
  if (!checkAttrArgsAreLockableObjs(S, D, Attr))
    return;

  D->addAttr(::new (S.Context) LocksExcludedAttr(Attr.getLoc(), S.Context));
}


static void handleExtVectorTypeAttr(Sema &S, Scope *scope, Decl *D,
                                    const AttributeList &Attr) {
  TypedefNameDecl *tDecl = dyn_cast<TypedefNameDecl>(D);
  if (tDecl == 0) {
    S.Diag(Attr.getLoc(), diag::err_typecheck_ext_vector_not_typedef);
    return;
  }

  QualType curType = tDecl->getUnderlyingType();

  Expr *sizeExpr;

  // Special case where the argument is a template id.
  if (Attr.getParameterName()) {
    CXXScopeSpec SS;
    UnqualifiedId id;
    id.setIdentifier(Attr.getParameterName(), Attr.getLoc());
    
    ExprResult Size = S.ActOnIdExpression(scope, SS, id, false, false);
    if (Size.isInvalid())
      return;
    
    sizeExpr = Size.get();
  } else {
    // check the attribute arguments.
    if (!checkAttributeNumArgs(S, Attr, 1))
      return;

    sizeExpr = Attr.getArg(0);
  }

  // Instantiate/Install the vector type, and let Sema build the type for us.
  // This will run the reguired checks.
  QualType T = S.BuildExtVectorType(curType, sizeExpr, Attr.getLoc());
  if (!T.isNull()) {
    // FIXME: preserve the old source info.
    tDecl->setTypeSourceInfo(S.Context.getTrivialTypeSourceInfo(T));

    // Remember this typedef decl, we will need it later for diagnostics.
    S.ExtVectorDecls.push_back(tDecl);
  }
}

static void handlePackedAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;

  if (TagDecl *TD = dyn_cast<TagDecl>(D))
    TD->addAttr(::new (S.Context) PackedAttr(Attr.getLoc(), S.Context));
  else if (FieldDecl *FD = dyn_cast<FieldDecl>(D)) {
    // If the alignment is less than or equal to 8 bits, the packed attribute
    // has no effect.
    if (!FD->getType()->isIncompleteType() &&
        S.Context.getTypeAlign(FD->getType()) <= 8)
      S.Diag(Attr.getLoc(), diag::warn_attribute_ignored_for_field_of_type)
        << Attr.getName() << FD->getType();
    else
      FD->addAttr(::new (S.Context) PackedAttr(Attr.getLoc(), S.Context));
  } else
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << Attr.getName();
}

static void handleMsStructAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (TagDecl *TD = dyn_cast<TagDecl>(D))
    TD->addAttr(::new (S.Context) MsStructAttr(Attr.getLoc(), S.Context));
  else
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << Attr.getName();
}

static void handleIBAction(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;

  // The IBAction attributes only apply to instance methods.
  if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D))
    if (MD->isInstanceMethod()) {
      D->addAttr(::new (S.Context) IBActionAttr(Attr.getLoc(), S.Context));
      return;
    }

  S.Diag(Attr.getLoc(), diag::warn_attribute_ibaction) << Attr.getName();
}

static void handleIBOutlet(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;

  // The IBOutlet attributes only apply to instance variables of
  // Objective-C classes.
  if (isa<ObjCIvarDecl>(D) || isa<ObjCPropertyDecl>(D)) {
    D->addAttr(::new (S.Context) IBOutletAttr(Attr.getLoc(), S.Context));
    return;
  }

  S.Diag(Attr.getLoc(), diag::warn_attribute_iboutlet) << Attr.getName();
}

static void handleIBOutletCollection(Sema &S, Decl *D,
                                     const AttributeList &Attr) {

  // The iboutletcollection attribute can have zero or one arguments.
  if (Attr.getParameterName() && Attr.getNumArgs() > 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }

  // The IBOutletCollection attributes only apply to instance variables of
  // Objective-C classes.
  if (!(isa<ObjCIvarDecl>(D) || isa<ObjCPropertyDecl>(D))) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_iboutlet) << Attr.getName();
    return;
  }
  if (const ValueDecl *VD = dyn_cast<ValueDecl>(D))
    if (!VD->getType()->getAs<ObjCObjectPointerType>()) {
      S.Diag(Attr.getLoc(), diag::err_iboutletcollection_object_type) 
        << VD->getType() << 0;
      return;
    }
  if (const ObjCPropertyDecl *PD = dyn_cast<ObjCPropertyDecl>(D))
    if (!PD->getType()->getAs<ObjCObjectPointerType>()) {
      S.Diag(Attr.getLoc(), diag::err_iboutletcollection_object_type) 
        << PD->getType() << 1;
      return;
    }
  
  IdentifierInfo *II = Attr.getParameterName();
  if (!II)
    II = &S.Context.Idents.get("id");
  
  ParsedType TypeRep = S.getTypeName(*II, Attr.getLoc(), 
                        S.getScopeForContext(D->getDeclContext()->getParent()));
  if (!TypeRep) {
    S.Diag(Attr.getLoc(), diag::err_iboutletcollection_type) << II;
    return;
  }
  QualType QT = TypeRep.get();
  // Diagnose use of non-object type in iboutletcollection attribute.
  // FIXME. Gnu attribute extension ignores use of builtin types in
  // attributes. So, __attribute__((iboutletcollection(char))) will be
  // treated as __attribute__((iboutletcollection())).
  if (!QT->isObjCIdType() && !QT->isObjCClassType() &&
      !QT->isObjCObjectType()) {
    S.Diag(Attr.getLoc(), diag::err_iboutletcollection_type) << II;
    return;
  }
  D->addAttr(::new (S.Context) IBOutletCollectionAttr(Attr.getLoc(), S.Context,
                                                      QT));
}

static void possibleTransparentUnionPointerType(QualType &T) {
  if (const RecordType *UT = T->getAsUnionType())
    if (UT && UT->getDecl()->hasAttr<TransparentUnionAttr>()) {
      RecordDecl *UD = UT->getDecl();
      for (RecordDecl::field_iterator it = UD->field_begin(),
           itend = UD->field_end(); it != itend; ++it) {
        QualType QT = it->getType();
        if (QT->isAnyPointerType() || QT->isBlockPointerType()) {
          T = QT;
          return;
        }
      }
    }
}

static void handleNonNullAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // GCC ignores the nonnull attribute on K&R style function prototypes, so we
  // ignore it as well
  if (!isFunctionOrMethod(D) || !hasFunctionProto(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return;
  }

  // In C++ the implicit 'this' function parameter also counts, and they are
  // counted from one.
  bool HasImplicitThisParam = isInstanceMethod(D);
  unsigned NumArgs  = getFunctionOrMethodNumArgs(D) + HasImplicitThisParam;

  // The nonnull attribute only applies to pointers.
  SmallVector<unsigned, 10> NonNullArgs;

  for (AttributeList::arg_iterator I=Attr.arg_begin(),
                                   E=Attr.arg_end(); I!=E; ++I) {


    // The argument must be an integer constant expression.
    Expr *Ex = *I;
    llvm::APSInt ArgNum(32);
    if (Ex->isTypeDependent() || Ex->isValueDependent() ||
        !Ex->isIntegerConstantExpr(ArgNum, S.Context)) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_not_int)
        << "nonnull" << Ex->getSourceRange();
      return;
    }

    unsigned x = (unsigned) ArgNum.getZExtValue();

    if (x < 1 || x > NumArgs) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_out_of_bounds)
       << "nonnull" << I.getArgNum() << Ex->getSourceRange();
      return;
    }

    --x;
    if (HasImplicitThisParam) {
      if (x == 0) {
        S.Diag(Attr.getLoc(),
               diag::err_attribute_invalid_implicit_this_argument)
          << "nonnull" << Ex->getSourceRange();
        return;
      }
      --x;
    }

    // Is the function argument a pointer type?
    QualType T = getFunctionOrMethodArgType(D, x).getNonReferenceType();
    possibleTransparentUnionPointerType(T);
    
    if (!T->isAnyPointerType() && !T->isBlockPointerType()) {
      // FIXME: Should also highlight argument in decl.
      S.Diag(Attr.getLoc(), diag::warn_nonnull_pointers_only)
        << "nonnull" << Ex->getSourceRange();
      continue;
    }

    NonNullArgs.push_back(x);
  }

  // If no arguments were specified to __attribute__((nonnull)) then all pointer
  // arguments have a nonnull attribute.
  if (NonNullArgs.empty()) {
    for (unsigned I = 0, E = getFunctionOrMethodNumArgs(D); I != E; ++I) {
      QualType T = getFunctionOrMethodArgType(D, I).getNonReferenceType();
      possibleTransparentUnionPointerType(T);
      if (T->isAnyPointerType() || T->isBlockPointerType())
        NonNullArgs.push_back(I);
    }

    // No pointer arguments?
    if (NonNullArgs.empty()) {
      // Warn the trivial case only if attribute is not coming from a
      // macro instantiation.
      if (Attr.getLoc().isFileID())
        S.Diag(Attr.getLoc(), diag::warn_attribute_nonnull_no_pointers);
      return;
    }
  }

  unsigned* start = &NonNullArgs[0];
  unsigned size = NonNullArgs.size();
  llvm::array_pod_sort(start, start + size);
  D->addAttr(::new (S.Context) NonNullAttr(Attr.getLoc(), S.Context, start,
                                           size));
}

static void handleOwnershipAttr(Sema &S, Decl *D, const AttributeList &AL) {
  // This attribute must be applied to a function declaration.
  // The first argument to the attribute must be a string,
  // the name of the resource, for example "malloc".
  // The following arguments must be argument indexes, the arguments must be
  // of integer type for Returns, otherwise of pointer type.
  // The difference between Holds and Takes is that a pointer may still be used
  // after being held.  free() should be __attribute((ownership_takes)), whereas
  // a list append function may well be __attribute((ownership_holds)).

  if (!AL.getParameterName()) {
    S.Diag(AL.getLoc(), diag::err_attribute_argument_n_not_string)
        << AL.getName()->getName() << 1;
    return;
  }
  // Figure out our Kind, and check arguments while we're at it.
  OwnershipAttr::OwnershipKind K;
  switch (AL.getKind()) {
  case AttributeList::AT_ownership_takes:
    K = OwnershipAttr::Takes;
    if (AL.getNumArgs() < 1) {
      S.Diag(AL.getLoc(), diag::err_attribute_wrong_number_arguments) << 2;
      return;
    }
    break;
  case AttributeList::AT_ownership_holds:
    K = OwnershipAttr::Holds;
    if (AL.getNumArgs() < 1) {
      S.Diag(AL.getLoc(), diag::err_attribute_wrong_number_arguments) << 2;
      return;
    }
    break;
  case AttributeList::AT_ownership_returns:
    K = OwnershipAttr::Returns;
    if (AL.getNumArgs() > 1) {
      S.Diag(AL.getLoc(), diag::err_attribute_wrong_number_arguments)
          << AL.getNumArgs() + 1;
      return;
    }
    break;
  default:
    // This should never happen given how we are called.
    llvm_unreachable("Unknown ownership attribute");
  }

  if (!isFunction(D) || !hasFunctionProto(D)) {
    S.Diag(AL.getLoc(), diag::warn_attribute_wrong_decl_type)
      << AL.getName() << ExpectedFunction;
    return;
  }

  // In C++ the implicit 'this' function parameter also counts, and they are
  // counted from one.
  bool HasImplicitThisParam = isInstanceMethod(D);
  unsigned NumArgs  = getFunctionOrMethodNumArgs(D) + HasImplicitThisParam;

  StringRef Module = AL.getParameterName()->getName();

  // Normalize the argument, __foo__ becomes foo.
  if (Module.startswith("__") && Module.endswith("__"))
    Module = Module.substr(2, Module.size() - 4);

  SmallVector<unsigned, 10> OwnershipArgs;

  for (AttributeList::arg_iterator I = AL.arg_begin(), E = AL.arg_end(); I != E;
       ++I) {

    Expr *IdxExpr = *I;
    llvm::APSInt ArgNum(32);
    if (IdxExpr->isTypeDependent() || IdxExpr->isValueDependent()
        || !IdxExpr->isIntegerConstantExpr(ArgNum, S.Context)) {
      S.Diag(AL.getLoc(), diag::err_attribute_argument_not_int)
          << AL.getName()->getName() << IdxExpr->getSourceRange();
      continue;
    }

    unsigned x = (unsigned) ArgNum.getZExtValue();

    if (x > NumArgs || x < 1) {
      S.Diag(AL.getLoc(), diag::err_attribute_argument_out_of_bounds)
          << AL.getName()->getName() << x << IdxExpr->getSourceRange();
      continue;
    }
    --x;
    if (HasImplicitThisParam) {
      if (x == 0) {
        S.Diag(AL.getLoc(), diag::err_attribute_invalid_implicit_this_argument)
          << "ownership" << IdxExpr->getSourceRange();
        return;
      }
      --x;
    }

    switch (K) {
    case OwnershipAttr::Takes:
    case OwnershipAttr::Holds: {
      // Is the function argument a pointer type?
      QualType T = getFunctionOrMethodArgType(D, x);
      if (!T->isAnyPointerType() && !T->isBlockPointerType()) {
        // FIXME: Should also highlight argument in decl.
        S.Diag(AL.getLoc(), diag::err_ownership_type)
            << ((K==OwnershipAttr::Takes)?"ownership_takes":"ownership_holds")
            << "pointer"
            << IdxExpr->getSourceRange();
        continue;
      }
      break;
    }
    case OwnershipAttr::Returns: {
      if (AL.getNumArgs() > 1) {
          // Is the function argument an integer type?
          Expr *IdxExpr = AL.getArg(0);
          llvm::APSInt ArgNum(32);
          if (IdxExpr->isTypeDependent() || IdxExpr->isValueDependent()
              || !IdxExpr->isIntegerConstantExpr(ArgNum, S.Context)) {
            S.Diag(AL.getLoc(), diag::err_ownership_type)
                << "ownership_returns" << "integer"
                << IdxExpr->getSourceRange();
            return;
          }
      }
      break;
    }
    default:
      llvm_unreachable("Unknown ownership attribute");
    } // switch

    // Check we don't have a conflict with another ownership attribute.
    for (specific_attr_iterator<OwnershipAttr>
          i = D->specific_attr_begin<OwnershipAttr>(),
          e = D->specific_attr_end<OwnershipAttr>();
        i != e; ++i) {
      if ((*i)->getOwnKind() != K) {
        for (const unsigned *I = (*i)->args_begin(), *E = (*i)->args_end();
             I!=E; ++I) {
          if (x == *I) {
            S.Diag(AL.getLoc(), diag::err_attributes_are_not_compatible)
                << AL.getName()->getName() << "ownership_*";
          }
        }
      }
    }
    OwnershipArgs.push_back(x);
  }

  unsigned* start = OwnershipArgs.data();
  unsigned size = OwnershipArgs.size();
  llvm::array_pod_sort(start, start + size);

  if (K != OwnershipAttr::Returns && OwnershipArgs.empty()) {
    S.Diag(AL.getLoc(), diag::err_attribute_wrong_number_arguments) << 2;
    return;
  }

  D->addAttr(::new (S.Context) OwnershipAttr(AL.getLoc(), S.Context, K, Module,
                                             start, size));
}

/// Whether this declaration has internal linkage for the purposes of
/// things that want to complain about things not have internal linkage.
static bool hasEffectivelyInternalLinkage(NamedDecl *D) {
  switch (D->getLinkage()) {
  case NoLinkage:
  case InternalLinkage:
    return true;

  // Template instantiations that go from external to unique-external
  // shouldn't get diagnosed.
  case UniqueExternalLinkage:
    return true;

  case ExternalLinkage:
    return false;
  }
  llvm_unreachable("unknown linkage kind!");
  return false;
}

static void handleWeakRefAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // Check the attribute arguments.
  if (Attr.getNumArgs() > 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }

  if (!isa<VarDecl>(D) && !isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedVariableOrFunction;
    return;
  }

  NamedDecl *nd = cast<NamedDecl>(D);

  // gcc rejects
  // class c {
  //   static int a __attribute__((weakref ("v2")));
  //   static int b() __attribute__((weakref ("f3")));
  // };
  // and ignores the attributes of
  // void f(void) {
  //   static int a __attribute__((weakref ("v2")));
  // }
  // we reject them
  const DeclContext *Ctx = D->getDeclContext()->getRedeclContext();
  if (!Ctx->isFileContext()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_weakref_not_global_context) <<
        nd->getNameAsString();
    return;
  }

  // The GCC manual says
  //
  // At present, a declaration to which `weakref' is attached can only
  // be `static'.
  //
  // It also says
  //
  // Without a TARGET,
  // given as an argument to `weakref' or to `alias', `weakref' is
  // equivalent to `weak'.
  //
  // gcc 4.4.1 will accept
  // int a7 __attribute__((weakref));
  // as
  // int a7 __attribute__((weak));
  // This looks like a bug in gcc. We reject that for now. We should revisit
  // it if this behaviour is actually used.

  if (!hasEffectivelyInternalLinkage(nd)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_weakref_not_static);
    return;
  }

  // GCC rejects
  // static ((alias ("y"), weakref)).
  // Should we? How to check that weakref is before or after alias?

  if (Attr.getNumArgs() == 1) {
    Expr *Arg = Attr.getArg(0);
    Arg = Arg->IgnoreParenCasts();
    StringLiteral *Str = dyn_cast<StringLiteral>(Arg);

    if (!Str || !Str->isAscii()) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string)
          << "weakref" << 1;
      return;
    }
    // GCC will accept anything as the argument of weakref. Should we
    // check for an existing decl?
    D->addAttr(::new (S.Context) AliasAttr(Attr.getLoc(), S.Context,
                                           Str->getString()));
  }

  D->addAttr(::new (S.Context) WeakRefAttr(Attr.getLoc(), S.Context));
}

static void handleAliasAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }

  Expr *Arg = Attr.getArg(0);
  Arg = Arg->IgnoreParenCasts();
  StringLiteral *Str = dyn_cast<StringLiteral>(Arg);

  if (!Str || !Str->isAscii()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string)
      << "alias" << 1;
    return;
  }

  if (S.Context.Target.getTriple().isOSDarwin()) {
    S.Diag(Attr.getLoc(), diag::err_alias_not_supported_on_darwin);
    return;
  }

  // FIXME: check if target symbol exists in current file

  D->addAttr(::new (S.Context) AliasAttr(Attr.getLoc(), S.Context,
                                         Str->getString()));
}

static void handleNakedAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // Check the attribute arguments.
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;

  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return;
  }

  D->addAttr(::new (S.Context) NakedAttr(Attr.getLoc(), S.Context));
}

static void handleAlwaysInlineAttr(Sema &S, Decl *D,
                                   const AttributeList &Attr) {
  // Check the attribute arguments.
  if (Attr.hasParameterOrArguments()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return;
  }

  D->addAttr(::new (S.Context) AlwaysInlineAttr(Attr.getLoc(), S.Context));
}

static void handleMallocAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // Check the attribute arguments.
  if (Attr.hasParameterOrArguments()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    QualType RetTy = FD->getResultType();
    if (RetTy->isAnyPointerType() || RetTy->isBlockPointerType()) {
      D->addAttr(::new (S.Context) MallocAttr(Attr.getLoc(), S.Context));
      return;
    }
  }

  S.Diag(Attr.getLoc(), diag::warn_attribute_malloc_pointer_only);
}

static void handleMayAliasAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;

  D->addAttr(::new (S.Context) MayAliasAttr(Attr.getLoc(), S.Context));
}

static void handleNoCommonAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  assert(!Attr.isInvalid());
  if (isa<VarDecl>(D))
    D->addAttr(::new (S.Context) NoCommonAttr(Attr.getLoc(), S.Context));
  else
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedVariable;
}

static void handleCommonAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  assert(!Attr.isInvalid());
  if (isa<VarDecl>(D))
    D->addAttr(::new (S.Context) CommonAttr(Attr.getLoc(), S.Context));
  else
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedVariable;
}

static void handleNoReturnAttr(Sema &S, Decl *D, const AttributeList &attr) {
  if (hasDeclarator(D)) return;

  if (S.CheckNoReturnAttr(attr)) return;

  if (!isa<ObjCMethodDecl>(D)) {
    S.Diag(attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << attr.getName() << ExpectedFunctionOrMethod;
    return;
  }

  D->addAttr(::new (S.Context) NoReturnAttr(attr.getLoc(), S.Context));
}

bool Sema::CheckNoReturnAttr(const AttributeList &attr) {
  if (attr.hasParameterOrArguments()) {
    Diag(attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    attr.setInvalid();
    return true;
  }

  return false;
}

static void handleAnalyzerNoReturnAttr(Sema &S, Decl *D,
                                       const AttributeList &Attr) {
  
  // The checking path for 'noreturn' and 'analyzer_noreturn' are different
  // because 'analyzer_noreturn' does not impact the type.
  
  if(!checkAttributeNumArgs(S, Attr, 0))
      return;
  
  if (!isFunctionOrMethod(D) && !isa<BlockDecl>(D)) {
    ValueDecl *VD = dyn_cast<ValueDecl>(D);
    if (VD == 0 || (!VD->getType()->isBlockPointerType()
                    && !VD->getType()->isFunctionPointerType())) {
      S.Diag(Attr.getLoc(),
             Attr.isCXX0XAttribute() ? diag::err_attribute_wrong_decl_type
             : diag::warn_attribute_wrong_decl_type)
        << Attr.getName() << ExpectedFunctionMethodOrBlock;
      return;
    }
  }
  
  D->addAttr(::new (S.Context) AnalyzerNoReturnAttr(Attr.getLoc(), S.Context));
}

// PS3 PPU-specific.
static void handleVecReturnAttr(Sema &S, Decl *D, const AttributeList &Attr) {
/*
  Returning a Vector Class in Registers
  
  According to the PPU ABI specifications, a class with a single member of 
  vector type is returned in memory when used as the return value of a function.
  This results in inefficient code when implementing vector classes. To return
  the value in a single vector register, add the vecreturn attribute to the
  class definition. This attribute is also applicable to struct types.
  
  Example:
  
  struct Vector
  {
    __vector float xyzw;
  } __attribute__((vecreturn));
  
  Vector Add(Vector lhs, Vector rhs)
  {
    Vector result;
    result.xyzw = vec_add(lhs.xyzw, rhs.xyzw);
    return result; // This will be returned in a register
  }
*/
  if (!isa<RecordDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedClass;
    return;
  }

  if (D->getAttr<VecReturnAttr>()) {
    S.Diag(Attr.getLoc(), diag::err_repeat_attribute) << "vecreturn";
    return;
  }

  RecordDecl *record = cast<RecordDecl>(D);
  int count = 0;

  if (!isa<CXXRecordDecl>(record)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_vecreturn_only_vector_member);
    return;
  }

  if (!cast<CXXRecordDecl>(record)->isPOD()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_vecreturn_only_pod_record);
    return;
  }

  for (RecordDecl::field_iterator iter = record->field_begin();
       iter != record->field_end(); iter++) {
    if ((count == 1) || !iter->getType()->isVectorType()) {
      S.Diag(Attr.getLoc(), diag::err_attribute_vecreturn_only_vector_member);
      return;
    }
    count++;
  }

  D->addAttr(::new (S.Context) VecReturnAttr(Attr.getLoc(), S.Context));
}

static void handleDependencyAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (!isFunctionOrMethod(D) && !isa<ParmVarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionMethodOrParameter;
    return;
  }
  // FIXME: Actually store the attribute on the declaration
}

static void handleUnusedAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.hasParameterOrArguments()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  if (!isa<VarDecl>(D) && !isa<ObjCIvarDecl>(D) && !isFunctionOrMethod(D) &&
      !isa<TypeDecl>(D) && !isa<LabelDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedVariableFunctionOrLabel;
    return;
  }

  D->addAttr(::new (S.Context) UnusedAttr(Attr.getLoc(), S.Context));
}

static void handleUsedAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.hasParameterOrArguments()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  if (const VarDecl *VD = dyn_cast<VarDecl>(D)) {
    if (VD->hasLocalStorage() || VD->hasExternalStorage()) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << "used";
      return;
    }
  } else if (!isFunctionOrMethod(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedVariableOrFunction;
    return;
  }

  D->addAttr(::new (S.Context) UsedAttr(Attr.getLoc(), S.Context));
}

static void handleConstructorAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.getNumArgs() > 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_too_many_arguments) << 1;
    return;
  }

  int priority = 65535; // FIXME: Do not hardcode such constants.
  if (Attr.getNumArgs() > 0) {
    Expr *E = Attr.getArg(0);
    llvm::APSInt Idx(32);
    if (E->isTypeDependent() || E->isValueDependent() ||
        !E->isIntegerConstantExpr(Idx, S.Context)) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_int)
        << "constructor" << 1 << E->getSourceRange();
      return;
    }
    priority = Idx.getZExtValue();
  }

  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return;
  }

  D->addAttr(::new (S.Context) ConstructorAttr(Attr.getLoc(), S.Context,
                                               priority));
}

static void handleDestructorAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.getNumArgs() > 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_too_many_arguments) << 1;
    return;
  }

  int priority = 65535; // FIXME: Do not hardcode such constants.
  if (Attr.getNumArgs() > 0) {
    Expr *E = Attr.getArg(0);
    llvm::APSInt Idx(32);
    if (E->isTypeDependent() || E->isValueDependent() ||
        !E->isIntegerConstantExpr(Idx, S.Context)) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_int)
        << "destructor" << 1 << E->getSourceRange();
      return;
    }
    priority = Idx.getZExtValue();
  }

  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return;
  }

  D->addAttr(::new (S.Context) DestructorAttr(Attr.getLoc(), S.Context,
                                              priority));
}

static void handleDeprecatedAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  unsigned NumArgs = Attr.getNumArgs();
  if (NumArgs > 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_too_many_arguments) << 1;
    return;
  }
  
  // Handle the case where deprecated attribute has a text message.
  StringRef Str;
  if (NumArgs == 1) {
    StringLiteral *SE = dyn_cast<StringLiteral>(Attr.getArg(0));
    if (!SE) {
      S.Diag(Attr.getArg(0)->getLocStart(), diag::err_attribute_not_string)
        << "deprecated";
      return;
    }
    Str = SE->getString();
  }

  D->addAttr(::new (S.Context) DeprecatedAttr(Attr.getLoc(), S.Context, Str));
}

static void handleUnavailableAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  unsigned NumArgs = Attr.getNumArgs();
  if (NumArgs > 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_too_many_arguments) << 1;
    return;
  }
  
  // Handle the case where unavailable attribute has a text message.
  StringRef Str;
  if (NumArgs == 1) {
    StringLiteral *SE = dyn_cast<StringLiteral>(Attr.getArg(0));
    if (!SE) {
      S.Diag(Attr.getArg(0)->getLocStart(), 
             diag::err_attribute_not_string) << "unavailable";
      return;
    }
    Str = SE->getString();
  }
  D->addAttr(::new (S.Context) UnavailableAttr(Attr.getLoc(), S.Context, Str));
}

static void handleArcWeakrefUnavailableAttr(Sema &S, Decl *D, 
                                            const AttributeList &Attr) {
  unsigned NumArgs = Attr.getNumArgs();
  if (NumArgs > 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_too_many_arguments) << 0;
    return;
  }
  
  D->addAttr(::new (S.Context) ArcWeakrefUnavailableAttr(
                                          Attr.getLoc(), S.Context));
}

static void handleAvailabilityAttr(Sema &S, Decl *D,
                                   const AttributeList &Attr) {
  IdentifierInfo *Platform = Attr.getParameterName();
  SourceLocation PlatformLoc = Attr.getParameterLoc();

  StringRef PlatformName
    = AvailabilityAttr::getPrettyPlatformName(Platform->getName());
  if (PlatformName.empty()) {
    S.Diag(PlatformLoc, diag::warn_availability_unknown_platform)
      << Platform;

    PlatformName = Platform->getName();
  }

  AvailabilityChange Introduced = Attr.getAvailabilityIntroduced();
  AvailabilityChange Deprecated = Attr.getAvailabilityDeprecated();
  AvailabilityChange Obsoleted = Attr.getAvailabilityObsoleted();
  bool IsUnavailable = Attr.getUnavailableLoc().isValid();

  // Ensure that Introduced < Deprecated < Obsoleted (although not all
  // of these steps are needed).
  if (Introduced.isValid() && Deprecated.isValid() &&
      !(Introduced.Version < Deprecated.Version)) {
    S.Diag(Introduced.KeywordLoc, diag::warn_availability_version_ordering)
      << 1 << PlatformName << Deprecated.Version.getAsString()
      << 0 << Introduced.Version.getAsString();
    return;
  }

  if (Introduced.isValid() && Obsoleted.isValid() &&
      !(Introduced.Version < Obsoleted.Version)) {
    S.Diag(Introduced.KeywordLoc, diag::warn_availability_version_ordering)
      << 2 << PlatformName << Obsoleted.Version.getAsString()
      << 0 << Introduced.Version.getAsString();
    return;
  }

  if (Deprecated.isValid() && Obsoleted.isValid() &&
      !(Deprecated.Version < Obsoleted.Version)) {
    S.Diag(Deprecated.KeywordLoc, diag::warn_availability_version_ordering)
      << 2 << PlatformName << Obsoleted.Version.getAsString()
      << 1 << Deprecated.Version.getAsString();
    return;
  }

  D->addAttr(::new (S.Context) AvailabilityAttr(Attr.getLoc(), S.Context,
                                                Platform,
                                                Introduced.Version,
                                                Deprecated.Version,
                                                Obsoleted.Version,
                                                IsUnavailable));
}

static void handleVisibilityAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if(!checkAttributeNumArgs(S, Attr, 1))
    return;

  Expr *Arg = Attr.getArg(0);
  Arg = Arg->IgnoreParenCasts();
  StringLiteral *Str = dyn_cast<StringLiteral>(Arg);

  if (!Str || !Str->isAscii()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string)
      << "visibility" << 1;
    return;
  }

  StringRef TypeStr = Str->getString();
  VisibilityAttr::VisibilityType type;

  if (TypeStr == "default")
    type = VisibilityAttr::Default;
  else if (TypeStr == "hidden")
    type = VisibilityAttr::Hidden;
  else if (TypeStr == "internal")
    type = VisibilityAttr::Hidden; // FIXME
  else if (TypeStr == "protected")
    type = VisibilityAttr::Protected;
  else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_unknown_visibility) << TypeStr;
    return;
  }

  D->addAttr(::new (S.Context) VisibilityAttr(Attr.getLoc(), S.Context, type));
}

static void handleObjCMethodFamilyAttr(Sema &S, Decl *decl,
                                       const AttributeList &Attr) {
  ObjCMethodDecl *method = dyn_cast<ObjCMethodDecl>(decl);
  if (!method) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << ExpectedMethod;
    return;
  }

  if (Attr.getNumArgs() != 0 || !Attr.getParameterName()) {
    if (!Attr.getParameterName() && Attr.getNumArgs() == 1) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string)
        << "objc_method_family" << 1;
    } else {
      S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    }
    Attr.setInvalid();
    return;
  }

  StringRef param = Attr.getParameterName()->getName();
  ObjCMethodFamilyAttr::FamilyKind family;
  if (param == "none")
    family = ObjCMethodFamilyAttr::OMF_None;
  else if (param == "alloc")
    family = ObjCMethodFamilyAttr::OMF_alloc;
  else if (param == "copy")
    family = ObjCMethodFamilyAttr::OMF_copy;
  else if (param == "init")
    family = ObjCMethodFamilyAttr::OMF_init;
  else if (param == "mutableCopy")
    family = ObjCMethodFamilyAttr::OMF_mutableCopy;
  else if (param == "new")
    family = ObjCMethodFamilyAttr::OMF_new;
  else {
    // Just warn and ignore it.  This is future-proof against new
    // families being used in system headers.
    S.Diag(Attr.getParameterLoc(), diag::warn_unknown_method_family);
    return;
  }

  if (family == ObjCMethodFamilyAttr::OMF_init && 
      !method->getResultType()->isObjCObjectPointerType()) {
    S.Diag(method->getLocation(), diag::err_init_method_bad_return_type)
      << method->getResultType();
    // Ignore the attribute.
    return;
  }

  method->addAttr(new (S.Context) ObjCMethodFamilyAttr(Attr.getLoc(),
                                                       S.Context, family));
}

static void handleObjCExceptionAttr(Sema &S, Decl *D,
                                    const AttributeList &Attr) {
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;

  ObjCInterfaceDecl *OCI = dyn_cast<ObjCInterfaceDecl>(D);
  if (OCI == 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_requires_objc_interface);
    return;
  }

  D->addAttr(::new (S.Context) ObjCExceptionAttr(Attr.getLoc(), S.Context));
}

static void handleObjCNSObject(Sema &S, Decl *D, const AttributeList &Attr) {
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }
  if (TypedefNameDecl *TD = dyn_cast<TypedefNameDecl>(D)) {
    QualType T = TD->getUnderlyingType();
    if (!T->isPointerType() ||
        !T->getAs<PointerType>()->getPointeeType()->isRecordType()) {
      S.Diag(TD->getLocation(), diag::err_nsobject_attribute);
      return;
    }
  }
  D->addAttr(::new (S.Context) ObjCNSObjectAttr(Attr.getLoc(), S.Context));
}

static void
handleOverloadableAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }

  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_overloadable_not_function);
    return;
  }

  D->addAttr(::new (S.Context) OverloadableAttr(Attr.getLoc(), S.Context));
}

static void handleBlocksAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (!Attr.getParameterName()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string)
      << "blocks" << 1;
    return;
  }

  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }

  BlocksAttr::BlockType type;
  if (Attr.getParameterName()->isStr("byref"))
    type = BlocksAttr::ByRef;
  else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_type_not_supported)
      << "blocks" << Attr.getParameterName();
    return;
  }

  D->addAttr(::new (S.Context) BlocksAttr(Attr.getLoc(), S.Context, type));
}

static void handleSentinelAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.getNumArgs() > 2) {
    S.Diag(Attr.getLoc(), diag::err_attribute_too_many_arguments) << 2;
    return;
  }

  int sentinel = 0;
  if (Attr.getNumArgs() > 0) {
    Expr *E = Attr.getArg(0);
    llvm::APSInt Idx(32);
    if (E->isTypeDependent() || E->isValueDependent() ||
        !E->isIntegerConstantExpr(Idx, S.Context)) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_int)
       << "sentinel" << 1 << E->getSourceRange();
      return;
    }
    sentinel = Idx.getZExtValue();

    if (sentinel < 0) {
      S.Diag(Attr.getLoc(), diag::err_attribute_sentinel_less_than_zero)
        << E->getSourceRange();
      return;
    }
  }

  int nullPos = 0;
  if (Attr.getNumArgs() > 1) {
    Expr *E = Attr.getArg(1);
    llvm::APSInt Idx(32);
    if (E->isTypeDependent() || E->isValueDependent() ||
        !E->isIntegerConstantExpr(Idx, S.Context)) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_int)
        << "sentinel" << 2 << E->getSourceRange();
      return;
    }
    nullPos = Idx.getZExtValue();

    if (nullPos > 1 || nullPos < 0) {
      // FIXME: This error message could be improved, it would be nice
      // to say what the bounds actually are.
      S.Diag(Attr.getLoc(), diag::err_attribute_sentinel_not_zero_or_one)
        << E->getSourceRange();
      return;
    }
  }

  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    const FunctionType *FT = FD->getType()->getAs<FunctionType>();
    assert(FT && "FunctionDecl has non-function type?");

    if (isa<FunctionNoProtoType>(FT)) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_sentinel_named_arguments);
      return;
    }

    if (!cast<FunctionProtoType>(FT)->isVariadic()) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_sentinel_not_variadic) << 0;
      return;
    }
  } else if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D)) {
    if (!MD->isVariadic()) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_sentinel_not_variadic) << 0;
      return;
    }
  } else if (isa<BlockDecl>(D)) {
    // Note! BlockDecl is typeless. Variadic diagnostics will be issued by the
    // caller.
    ;
  } else if (const VarDecl *V = dyn_cast<VarDecl>(D)) {
    QualType Ty = V->getType();
    if (Ty->isBlockPointerType() || Ty->isFunctionPointerType()) {
      const FunctionType *FT = Ty->isFunctionPointerType() ? getFunctionType(D)
       : Ty->getAs<BlockPointerType>()->getPointeeType()->getAs<FunctionType>();
      if (!cast<FunctionProtoType>(FT)->isVariadic()) {
        int m = Ty->isFunctionPointerType() ? 0 : 1;
        S.Diag(Attr.getLoc(), diag::warn_attribute_sentinel_not_variadic) << m;
        return;
      }
    } else {
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
        << Attr.getName() << ExpectedFunctionMethodOrBlock;
      return;
    }
  } else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionMethodOrBlock;
    return;
  }
  D->addAttr(::new (S.Context) SentinelAttr(Attr.getLoc(), S.Context, sentinel,
                                            nullPos));
}

static void handleWarnUnusedResult(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;

  if (!isFunction(D) && !isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionOrMethod;
    return;
  }

  if (isFunction(D) && getFunctionType(D)->getResultType()->isVoidType()) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_void_function_method)
      << Attr.getName() << 0;
    return;
  }
  if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D))
    if (MD->getResultType()->isVoidType()) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_void_function_method)
      << Attr.getName() << 1;
      return;
    }
  
  D->addAttr(::new (S.Context) WarnUnusedResultAttr(Attr.getLoc(), S.Context));
}

static void handleWeakAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.hasParameterOrArguments()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  if (!isa<VarDecl>(D) && !isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedVariableOrFunction;
    return;
  }

  NamedDecl *nd = cast<NamedDecl>(D);

  // 'weak' only applies to declarations with external linkage.
  if (hasEffectivelyInternalLinkage(nd)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_weak_static);
    return;
  }

  nd->addAttr(::new (S.Context) WeakAttr(Attr.getLoc(), S.Context));
}

static void handleWeakImportAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;


  // weak_import only applies to variable & function declarations.
  bool isDef = false;
  if (!D->canBeWeakImported(isDef)) {
    if (isDef)
      S.Diag(Attr.getLoc(),
             diag::warn_attribute_weak_import_invalid_on_definition)
        << "weak_import" << 2 /*variable and function*/;
    else if (isa<ObjCPropertyDecl>(D) || isa<ObjCMethodDecl>(D) ||
             (S.Context.Target.getTriple().isOSDarwin() &&
              isa<ObjCInterfaceDecl>(D))) {
      // Nothing to warn about here.
    } else
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
        << Attr.getName() << ExpectedVariableOrFunction;

    return;
  }

  D->addAttr(::new (S.Context) WeakImportAttr(Attr.getLoc(), S.Context));
}

static void handleReqdWorkGroupSize(Sema &S, Decl *D,
                                    const AttributeList &Attr) {
  // Attribute has 3 arguments.
  if (!checkAttributeNumArgs(S, Attr, 3))
    return;

  unsigned WGSize[3];
  for (unsigned i = 0; i < 3; ++i) {
    Expr *E = Attr.getArg(i);
    llvm::APSInt ArgNum(32);
    if (E->isTypeDependent() || E->isValueDependent() ||
        !E->isIntegerConstantExpr(ArgNum, S.Context)) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_not_int)
        << "reqd_work_group_size" << E->getSourceRange();
      return;
    }
    WGSize[i] = (unsigned) ArgNum.getZExtValue();
  }
  D->addAttr(::new (S.Context) ReqdWorkGroupSizeAttr(Attr.getLoc(), S.Context,
                                                     WGSize[0], WGSize[1],
                                                     WGSize[2]));
}

static void handleSectionAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // Attribute has no arguments.
  if (!checkAttributeNumArgs(S, Attr, 1))
    return;

  // Make sure that there is a string literal as the sections's single
  // argument.
  Expr *ArgExpr = Attr.getArg(0);
  StringLiteral *SE = dyn_cast<StringLiteral>(ArgExpr);
  if (!SE) {
    S.Diag(ArgExpr->getLocStart(), diag::err_attribute_not_string) << "section";
    return;
  }

  // If the target wants to validate the section specifier, make it happen.
  std::string Error = S.Context.Target.isValidSectionSpecifier(SE->getString());
  if (!Error.empty()) {
    S.Diag(SE->getLocStart(), diag::err_attribute_section_invalid_for_target)
    << Error;
    return;
  }

  // This attribute cannot be applied to local variables.
  if (isa<VarDecl>(D) && cast<VarDecl>(D)->hasLocalStorage()) {
    S.Diag(SE->getLocStart(), diag::err_attribute_section_local_variable);
    return;
  }
  
  D->addAttr(::new (S.Context) SectionAttr(Attr.getLoc(), S.Context,
                                           SE->getString()));
}


static void handleNothrowAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.hasParameterOrArguments()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }
  
  if (NoThrowAttr *Existing = D->getAttr<NoThrowAttr>()) {
    if (Existing->getLocation().isInvalid())
      Existing->setLocation(Attr.getLoc());
  } else {
    D->addAttr(::new (S.Context) NoThrowAttr(Attr.getLoc(), S.Context));
  }
}

static void handleConstAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.hasParameterOrArguments()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  if (ConstAttr *Existing = D->getAttr<ConstAttr>()) {
   if (Existing->getLocation().isInvalid())
     Existing->setLocation(Attr.getLoc());
  } else {
    D->addAttr(::new (S.Context) ConstAttr(Attr.getLoc(), S.Context));
  }
}

static void handlePureAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;

  D->addAttr(::new (S.Context) PureAttr(Attr.getLoc(), S.Context));
}

static void handleCleanupAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (!Attr.getParameterName()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }

  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }

  VarDecl *VD = dyn_cast<VarDecl>(D);

  if (!VD || !VD->hasLocalStorage()) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << "cleanup";
    return;
  }

  // Look up the function
  // FIXME: Lookup probably isn't looking in the right place
  NamedDecl *CleanupDecl
    = S.LookupSingleName(S.TUScope, Attr.getParameterName(),
                         Attr.getParameterLoc(), Sema::LookupOrdinaryName);
  if (!CleanupDecl) {
    S.Diag(Attr.getParameterLoc(), diag::err_attribute_cleanup_arg_not_found) <<
      Attr.getParameterName();
    return;
  }

  FunctionDecl *FD = dyn_cast<FunctionDecl>(CleanupDecl);
  if (!FD) {
    S.Diag(Attr.getParameterLoc(),
           diag::err_attribute_cleanup_arg_not_function)
      << Attr.getParameterName();
    return;
  }

  if (FD->getNumParams() != 1) {
    S.Diag(Attr.getParameterLoc(),
           diag::err_attribute_cleanup_func_must_take_one_arg)
      << Attr.getParameterName();
    return;
  }

  // We're currently more strict than GCC about what function types we accept.
  // If this ever proves to be a problem it should be easy to fix.
  QualType Ty = S.Context.getPointerType(VD->getType());
  QualType ParamTy = FD->getParamDecl(0)->getType();
  if (S.CheckAssignmentConstraints(FD->getParamDecl(0)->getLocation(),
                                   ParamTy, Ty) != Sema::Compatible) {
    S.Diag(Attr.getParameterLoc(),
           diag::err_attribute_cleanup_func_arg_incompatible_type) <<
      Attr.getParameterName() << ParamTy << Ty;
    return;
  }

  D->addAttr(::new (S.Context) CleanupAttr(Attr.getLoc(), S.Context, FD));
  S.MarkDeclarationReferenced(Attr.getParameterLoc(), FD);
}

/// Handle __attribute__((format_arg((idx)))) attribute based on
/// http://gcc.gnu.org/onlinedocs/gcc/Function-Attributes.html
static void handleFormatArgAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (!checkAttributeNumArgs(S, Attr, 1))
    return;

  if (!isFunctionOrMethod(D) || !hasFunctionProto(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return;
  }

  // In C++ the implicit 'this' function parameter also counts, and they are
  // counted from one.
  bool HasImplicitThisParam = isInstanceMethod(D);
  unsigned NumArgs  = getFunctionOrMethodNumArgs(D) + HasImplicitThisParam;
  unsigned FirstIdx = 1;

  // checks for the 2nd argument
  Expr *IdxExpr = Attr.getArg(0);
  llvm::APSInt Idx(32);
  if (IdxExpr->isTypeDependent() || IdxExpr->isValueDependent() ||
      !IdxExpr->isIntegerConstantExpr(Idx, S.Context)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_int)
    << "format" << 2 << IdxExpr->getSourceRange();
    return;
  }

  if (Idx.getZExtValue() < FirstIdx || Idx.getZExtValue() > NumArgs) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_out_of_bounds)
    << "format" << 2 << IdxExpr->getSourceRange();
    return;
  }

  unsigned ArgIdx = Idx.getZExtValue() - 1;

  if (HasImplicitThisParam) {
    if (ArgIdx == 0) {
      S.Diag(Attr.getLoc(), diag::err_attribute_invalid_implicit_this_argument)
        << "format_arg" << IdxExpr->getSourceRange();
      return;
    }
    ArgIdx--;
  }

  // make sure the format string is really a string
  QualType Ty = getFunctionOrMethodArgType(D, ArgIdx);

  bool not_nsstring_type = !isNSStringType(Ty, S.Context);
  if (not_nsstring_type &&
      !isCFStringType(Ty, S.Context) &&
      (!Ty->isPointerType() ||
       !Ty->getAs<PointerType>()->getPointeeType()->isCharType())) {
    // FIXME: Should highlight the actual expression that has the wrong type.
    S.Diag(Attr.getLoc(), diag::err_format_attribute_not)
    << (not_nsstring_type ? "a string type" : "an NSString")
       << IdxExpr->getSourceRange();
    return;
  }
  Ty = getFunctionOrMethodResultType(D);
  if (!isNSStringType(Ty, S.Context) &&
      !isCFStringType(Ty, S.Context) &&
      (!Ty->isPointerType() ||
       !Ty->getAs<PointerType>()->getPointeeType()->isCharType())) {
    // FIXME: Should highlight the actual expression that has the wrong type.
    S.Diag(Attr.getLoc(), diag::err_format_attribute_result_not)
    << (not_nsstring_type ? "string type" : "NSString")
       << IdxExpr->getSourceRange();
    return;
  }

  D->addAttr(::new (S.Context) FormatArgAttr(Attr.getLoc(), S.Context,
                                             Idx.getZExtValue()));
}

enum FormatAttrKind {
  CFStringFormat,
  NSStringFormat,
  StrftimeFormat,
  SupportedFormat,
  IgnoredFormat,
  InvalidFormat
};

/// getFormatAttrKind - Map from format attribute names to supported format
/// types.
static FormatAttrKind getFormatAttrKind(StringRef Format) {
  // Check for formats that get handled specially.
  if (Format == "NSString")
    return NSStringFormat;
  if (Format == "CFString")
    return CFStringFormat;
  if (Format == "strftime")
    return StrftimeFormat;

  // Otherwise, check for supported formats.
  if (Format == "scanf" || Format == "printf" || Format == "printf0" ||
      Format == "strfmon" || Format == "cmn_err" || Format == "strftime" ||
      Format == "NSString" || Format == "CFString" || Format == "vcmn_err" ||
      Format == "zcmn_err" ||
      Format == "kprintf")  // OpenBSD.
    return SupportedFormat;

  if (Format == "gcc_diag" || Format == "gcc_cdiag" ||
      Format == "gcc_cxxdiag" || Format == "gcc_tdiag")
    return IgnoredFormat;
  
  return InvalidFormat;
}

/// Handle __attribute__((init_priority(priority))) attributes based on
/// http://gcc.gnu.org/onlinedocs/gcc/C_002b_002b-Attributes.html
static void handleInitPriorityAttr(Sema &S, Decl *D,
                                   const AttributeList &Attr) {
  if (!S.getLangOptions().CPlusPlus) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << Attr.getName();
    return;
  }
  
  if (!isa<VarDecl>(D) || S.getCurFunctionOrMethodDecl()) {
    S.Diag(Attr.getLoc(), diag::err_init_priority_object_attr);
    Attr.setInvalid();
    return;
  }
  QualType T = dyn_cast<VarDecl>(D)->getType();
  if (S.Context.getAsArrayType(T))
    T = S.Context.getBaseElementType(T);
  if (!T->getAs<RecordType>()) {
    S.Diag(Attr.getLoc(), diag::err_init_priority_object_attr);
    Attr.setInvalid();
    return;
  }
  
  if (Attr.getNumArgs() != 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    Attr.setInvalid();
    return;
  }
  Expr *priorityExpr = Attr.getArg(0);
  
  llvm::APSInt priority(32);
  if (priorityExpr->isTypeDependent() || priorityExpr->isValueDependent() ||
      !priorityExpr->isIntegerConstantExpr(priority, S.Context)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_not_int)
    << "init_priority" << priorityExpr->getSourceRange();
    Attr.setInvalid();
    return;
  }
  unsigned prioritynum = priority.getZExtValue();
  if (prioritynum < 101 || prioritynum > 65535) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_outof_range)
    <<  priorityExpr->getSourceRange();
    Attr.setInvalid();
    return;
  }
  D->addAttr(::new (S.Context) InitPriorityAttr(Attr.getLoc(), S.Context,
                                                prioritynum));
}

/// Handle __attribute__((format(type,idx,firstarg))) attributes based on
/// http://gcc.gnu.org/onlinedocs/gcc/Function-Attributes.html
static void handleFormatAttr(Sema &S, Decl *D, const AttributeList &Attr) {

  if (!Attr.getParameterName()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string)
      << "format" << 1;
    return;
  }

  if (Attr.getNumArgs() != 2) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 3;
    return;
  }

  if (!isFunctionOrMethodOrBlock(D) || !hasFunctionProto(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return;
  }

  // In C++ the implicit 'this' function parameter also counts, and they are
  // counted from one.
  bool HasImplicitThisParam = isInstanceMethod(D);
  unsigned NumArgs  = getFunctionOrMethodNumArgs(D) + HasImplicitThisParam;
  unsigned FirstIdx = 1;

  StringRef Format = Attr.getParameterName()->getName();

  // Normalize the argument, __foo__ becomes foo.
  if (Format.startswith("__") && Format.endswith("__"))
    Format = Format.substr(2, Format.size() - 4);

  // Check for supported formats.
  FormatAttrKind Kind = getFormatAttrKind(Format);
  
  if (Kind == IgnoredFormat)
    return;
  
  if (Kind == InvalidFormat) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_type_not_supported)
      << "format" << Attr.getParameterName()->getName();
    return;
  }

  // checks for the 2nd argument
  Expr *IdxExpr = Attr.getArg(0);
  llvm::APSInt Idx(32);
  if (IdxExpr->isTypeDependent() || IdxExpr->isValueDependent() ||
      !IdxExpr->isIntegerConstantExpr(Idx, S.Context)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_int)
      << "format" << 2 << IdxExpr->getSourceRange();
    return;
  }

  if (Idx.getZExtValue() < FirstIdx || Idx.getZExtValue() > NumArgs) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_out_of_bounds)
      << "format" << 2 << IdxExpr->getSourceRange();
    return;
  }

  // FIXME: Do we need to bounds check?
  unsigned ArgIdx = Idx.getZExtValue() - 1;

  if (HasImplicitThisParam) {
    if (ArgIdx == 0) {
      S.Diag(Attr.getLoc(),
             diag::err_format_attribute_implicit_this_format_string)
        << IdxExpr->getSourceRange();
      return;
    }
    ArgIdx--;
  }

  // make sure the format string is really a string
  QualType Ty = getFunctionOrMethodArgType(D, ArgIdx);

  if (Kind == CFStringFormat) {
    if (!isCFStringType(Ty, S.Context)) {
      S.Diag(Attr.getLoc(), diag::err_format_attribute_not)
        << "a CFString" << IdxExpr->getSourceRange();
      return;
    }
  } else if (Kind == NSStringFormat) {
    // FIXME: do we need to check if the type is NSString*?  What are the
    // semantics?
    if (!isNSStringType(Ty, S.Context)) {
      // FIXME: Should highlight the actual expression that has the wrong type.
      S.Diag(Attr.getLoc(), diag::err_format_attribute_not)
        << "an NSString" << IdxExpr->getSourceRange();
      return;
    }
  } else if (!Ty->isPointerType() ||
             !Ty->getAs<PointerType>()->getPointeeType()->isCharType()) {
    // FIXME: Should highlight the actual expression that has the wrong type.
    S.Diag(Attr.getLoc(), diag::err_format_attribute_not)
      << "a string type" << IdxExpr->getSourceRange();
    return;
  }

  // check the 3rd argument
  Expr *FirstArgExpr = Attr.getArg(1);
  llvm::APSInt FirstArg(32);
  if (FirstArgExpr->isTypeDependent() || FirstArgExpr->isValueDependent() ||
      !FirstArgExpr->isIntegerConstantExpr(FirstArg, S.Context)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_int)
      << "format" << 3 << FirstArgExpr->getSourceRange();
    return;
  }

  // check if the function is variadic if the 3rd argument non-zero
  if (FirstArg != 0) {
    if (isFunctionOrMethodVariadic(D)) {
      ++NumArgs; // +1 for ...
    } else {
      S.Diag(D->getLocation(), diag::err_format_attribute_requires_variadic);
      return;
    }
  }

  // strftime requires FirstArg to be 0 because it doesn't read from any
  // variable the input is just the current time + the format string.
  if (Kind == StrftimeFormat) {
    if (FirstArg != 0) {
      S.Diag(Attr.getLoc(), diag::err_format_strftime_third_parameter)
        << FirstArgExpr->getSourceRange();
      return;
    }
  // if 0 it disables parameter checking (to use with e.g. va_list)
  } else if (FirstArg != 0 && FirstArg != NumArgs) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_out_of_bounds)
      << "format" << 3 << FirstArgExpr->getSourceRange();
    return;
  }

  // Check whether we already have an equivalent format attribute.
  for (specific_attr_iterator<FormatAttr>
         i = D->specific_attr_begin<FormatAttr>(),
         e = D->specific_attr_end<FormatAttr>();
       i != e ; ++i) {
    FormatAttr *f = *i;
    if (f->getType() == Format &&
        f->getFormatIdx() == (int)Idx.getZExtValue() &&
        f->getFirstArg() == (int)FirstArg.getZExtValue()) {
      // If we don't have a valid location for this attribute, adopt the
      // location.
      if (f->getLocation().isInvalid())
        f->setLocation(Attr.getLoc());
      return;
    }
  }
  
  D->addAttr(::new (S.Context) FormatAttr(Attr.getLoc(), S.Context, Format,
                                          Idx.getZExtValue(),
                                          FirstArg.getZExtValue()));
}

static void handleTransparentUnionAttr(Sema &S, Decl *D,
                                       const AttributeList &Attr) {
  // check the attribute arguments.
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;


  // Try to find the underlying union declaration.
  RecordDecl *RD = 0;
  TypedefNameDecl *TD = dyn_cast<TypedefNameDecl>(D);
  if (TD && TD->getUnderlyingType()->isUnionType())
    RD = TD->getUnderlyingType()->getAsUnionType()->getDecl();
  else
    RD = dyn_cast<RecordDecl>(D);

  if (!RD || !RD->isUnion()) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedUnion;
    return;
  }

  if (!RD->isDefinition()) {
    S.Diag(Attr.getLoc(),
        diag::warn_transparent_union_attribute_not_definition);
    return;
  }

  RecordDecl::field_iterator Field = RD->field_begin(),
                          FieldEnd = RD->field_end();
  if (Field == FieldEnd) {
    S.Diag(Attr.getLoc(), diag::warn_transparent_union_attribute_zero_fields);
    return;
  }

  FieldDecl *FirstField = *Field;
  QualType FirstType = FirstField->getType();
  if (FirstType->hasFloatingRepresentation() || FirstType->isVectorType()) {
    S.Diag(FirstField->getLocation(),
           diag::warn_transparent_union_attribute_floating)
      << FirstType->isVectorType() << FirstType;
    return;
  }

  uint64_t FirstSize = S.Context.getTypeSize(FirstType);
  uint64_t FirstAlign = S.Context.getTypeAlign(FirstType);
  for (; Field != FieldEnd; ++Field) {
    QualType FieldType = Field->getType();
    if (S.Context.getTypeSize(FieldType) != FirstSize ||
        S.Context.getTypeAlign(FieldType) != FirstAlign) {
      // Warn if we drop the attribute.
      bool isSize = S.Context.getTypeSize(FieldType) != FirstSize;
      unsigned FieldBits = isSize? S.Context.getTypeSize(FieldType)
                                 : S.Context.getTypeAlign(FieldType);
      S.Diag(Field->getLocation(),
          diag::warn_transparent_union_attribute_field_size_align)
        << isSize << Field->getDeclName() << FieldBits;
      unsigned FirstBits = isSize? FirstSize : FirstAlign;
      S.Diag(FirstField->getLocation(),
             diag::note_transparent_union_first_field_size_align)
        << isSize << FirstBits;
      return;
    }
  }

  RD->addAttr(::new (S.Context) TransparentUnionAttr(Attr.getLoc(), S.Context));
}

static void handleAnnotateAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (!checkAttributeNumArgs(S, Attr, 1))
    return;

  Expr *ArgExpr = Attr.getArg(0);
  StringLiteral *SE = dyn_cast<StringLiteral>(ArgExpr);

  // Make sure that there is a string literal as the annotation's single
  // argument.
  if (!SE) {
    S.Diag(ArgExpr->getLocStart(), diag::err_attribute_not_string) <<"annotate";
    return;
  }
  D->addAttr(::new (S.Context) AnnotateAttr(Attr.getLoc(), S.Context,
                                            SE->getString()));
}

static void handleAlignedAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.getNumArgs() > 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }
  
  //FIXME: The C++0x version of this attribute has more limited applicabilty
  //       than GNU's, and should error out when it is used to specify a
  //       weaker alignment, rather than being silently ignored.

  if (Attr.getNumArgs() == 0) {
    D->addAttr(::new (S.Context) AlignedAttr(Attr.getLoc(), S.Context, true, 0));
    return;
  }

  S.AddAlignedAttr(Attr.getLoc(), D, Attr.getArg(0));
}

void Sema::AddAlignedAttr(SourceLocation AttrLoc, Decl *D, Expr *E) {
  if (E->isTypeDependent() || E->isValueDependent()) {
    // Save dependent expressions in the AST to be instantiated.
    D->addAttr(::new (Context) AlignedAttr(AttrLoc, Context, true, E));
    return;
  }

  // FIXME: Cache the number on the Attr object?
  llvm::APSInt Alignment(32);
  if (!E->isIntegerConstantExpr(Alignment, Context)) {
    Diag(AttrLoc, diag::err_attribute_argument_not_int)
      << "aligned" << E->getSourceRange();
    return;
  }
  if (!llvm::isPowerOf2_64(Alignment.getZExtValue())) {
    Diag(AttrLoc, diag::err_attribute_aligned_not_power_of_two)
      << E->getSourceRange();
    return;
  }

  D->addAttr(::new (Context) AlignedAttr(AttrLoc, Context, true, E));
}

void Sema::AddAlignedAttr(SourceLocation AttrLoc, Decl *D, TypeSourceInfo *TS) {
  // FIXME: Cache the number on the Attr object if non-dependent?
  // FIXME: Perform checking of type validity
  D->addAttr(::new (Context) AlignedAttr(AttrLoc, Context, false, TS));
  return;
}

/// handleModeAttr - This attribute modifies the width of a decl with primitive
/// type.
///
/// Despite what would be logical, the mode attribute is a decl attribute, not a
/// type attribute: 'int ** __attribute((mode(HI))) *G;' tries to make 'G' be
/// HImode, not an intermediate pointer.
static void handleModeAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // This attribute isn't documented, but glibc uses it.  It changes
  // the width of an int or unsigned int to the specified size.

  // Check that there aren't any arguments
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;


  IdentifierInfo *Name = Attr.getParameterName();
  if (!Name) {
    S.Diag(Attr.getLoc(), diag::err_attribute_missing_parameter_name);
    return;
  }

  StringRef Str = Attr.getParameterName()->getName();

  // Normalize the attribute name, __foo__ becomes foo.
  if (Str.startswith("__") && Str.endswith("__"))
    Str = Str.substr(2, Str.size() - 4);

  unsigned DestWidth = 0;
  bool IntegerMode = true;
  bool ComplexMode = false;
  switch (Str.size()) {
  case 2:
    switch (Str[0]) {
    case 'Q': DestWidth = 8; break;
    case 'H': DestWidth = 16; break;
    case 'S': DestWidth = 32; break;
    case 'D': DestWidth = 64; break;
    case 'X': DestWidth = 96; break;
    case 'T': DestWidth = 128; break;
    }
    if (Str[1] == 'F') {
      IntegerMode = false;
    } else if (Str[1] == 'C') {
      IntegerMode = false;
      ComplexMode = true;
    } else if (Str[1] != 'I') {
      DestWidth = 0;
    }
    break;
  case 4:
    // FIXME: glibc uses 'word' to define register_t; this is narrower than a
    // pointer on PIC16 and other embedded platforms.
    if (Str == "word")
      DestWidth = S.Context.Target.getPointerWidth(0);
    else if (Str == "byte")
      DestWidth = S.Context.Target.getCharWidth();
    break;
  case 7:
    if (Str == "pointer")
      DestWidth = S.Context.Target.getPointerWidth(0);
    break;
  }

  QualType OldTy;
  if (TypedefNameDecl *TD = dyn_cast<TypedefNameDecl>(D))
    OldTy = TD->getUnderlyingType();
  else if (ValueDecl *VD = dyn_cast<ValueDecl>(D))
    OldTy = VD->getType();
  else {
    S.Diag(D->getLocation(), diag::err_attr_wrong_decl)
      << "mode" << SourceRange(Attr.getLoc(), Attr.getLoc());
    return;
  }

  if (!OldTy->getAs<BuiltinType>() && !OldTy->isComplexType())
    S.Diag(Attr.getLoc(), diag::err_mode_not_primitive);
  else if (IntegerMode) {
    if (!OldTy->isIntegralOrEnumerationType())
      S.Diag(Attr.getLoc(), diag::err_mode_wrong_type);
  } else if (ComplexMode) {
    if (!OldTy->isComplexType())
      S.Diag(Attr.getLoc(), diag::err_mode_wrong_type);
  } else {
    if (!OldTy->isFloatingType())
      S.Diag(Attr.getLoc(), diag::err_mode_wrong_type);
  }

  // FIXME: Sync this with InitializePredefinedMacros; we need to match int8_t
  // and friends, at least with glibc.
  // FIXME: Make sure 32/64-bit integers don't get defined to types of the wrong
  // width on unusual platforms.
  // FIXME: Make sure floating-point mappings are accurate
  // FIXME: Support XF and TF types
  QualType NewTy;
  switch (DestWidth) {
  case 0:
    S.Diag(Attr.getLoc(), diag::err_unknown_machine_mode) << Name;
    return;
  default:
    S.Diag(Attr.getLoc(), diag::err_unsupported_machine_mode) << Name;
    return;
  case 8:
    if (!IntegerMode) {
      S.Diag(Attr.getLoc(), diag::err_unsupported_machine_mode) << Name;
      return;
    }
    if (OldTy->isSignedIntegerType())
      NewTy = S.Context.SignedCharTy;
    else
      NewTy = S.Context.UnsignedCharTy;
    break;
  case 16:
    if (!IntegerMode) {
      S.Diag(Attr.getLoc(), diag::err_unsupported_machine_mode) << Name;
      return;
    }
    if (OldTy->isSignedIntegerType())
      NewTy = S.Context.ShortTy;
    else
      NewTy = S.Context.UnsignedShortTy;
    break;
  case 32:
    if (!IntegerMode)
      NewTy = S.Context.FloatTy;
    else if (OldTy->isSignedIntegerType())
      NewTy = S.Context.IntTy;
    else
      NewTy = S.Context.UnsignedIntTy;
    break;
  case 64:
    if (!IntegerMode)
      NewTy = S.Context.DoubleTy;
    else if (OldTy->isSignedIntegerType())
      if (S.Context.Target.getLongWidth() == 64)
        NewTy = S.Context.LongTy;
      else
        NewTy = S.Context.LongLongTy;
    else
      if (S.Context.Target.getLongWidth() == 64)
        NewTy = S.Context.UnsignedLongTy;
      else
        NewTy = S.Context.UnsignedLongLongTy;
    break;
  case 96:
    NewTy = S.Context.LongDoubleTy;
    break;
  case 128:
    if (!IntegerMode) {
      S.Diag(Attr.getLoc(), diag::err_unsupported_machine_mode) << Name;
      return;
    }
    if (OldTy->isSignedIntegerType())
      NewTy = S.Context.Int128Ty;
    else
      NewTy = S.Context.UnsignedInt128Ty;
    break;
  }

  if (ComplexMode) {
    NewTy = S.Context.getComplexType(NewTy);
  }

  // Install the new type.
  if (TypedefNameDecl *TD = dyn_cast<TypedefNameDecl>(D)) {
    // FIXME: preserve existing source info.
    TD->setTypeSourceInfo(S.Context.getTrivialTypeSourceInfo(NewTy));
  } else
    cast<ValueDecl>(D)->setType(NewTy);
}

static void handleNoDebugAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;

  if (!isFunctionOrMethod(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return;
  }

  D->addAttr(::new (S.Context) NoDebugAttr(Attr.getLoc(), S.Context));
}

static void handleNoInlineAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;


  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return;
  }

  D->addAttr(::new (S.Context) NoInlineAttr(Attr.getLoc(), S.Context));
}

static void handleNoInstrumentFunctionAttr(Sema &S, Decl *D,
                                           const AttributeList &Attr) {
  // check the attribute arguments.
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;


  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return;
  }

  D->addAttr(::new (S.Context) NoInstrumentFunctionAttr(Attr.getLoc(),
                                                        S.Context));
}

static void handleConstantAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (S.LangOpts.CUDA) {
    // check the attribute arguments.
    if (Attr.hasParameterOrArguments()) {
      S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
      return;
    }

    if (!isa<VarDecl>(D)) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
        << Attr.getName() << ExpectedVariable;
      return;
    }

    D->addAttr(::new (S.Context) CUDAConstantAttr(Attr.getLoc(), S.Context));
  } else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << "constant";
  }
}

static void handleDeviceAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (S.LangOpts.CUDA) {
    // check the attribute arguments.
    if (Attr.getNumArgs() != 0) {
      S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
      return;
    }

    if (!isa<FunctionDecl>(D) && !isa<VarDecl>(D)) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
        << Attr.getName() << ExpectedVariableOrFunction;
      return;
    }

    D->addAttr(::new (S.Context) CUDADeviceAttr(Attr.getLoc(), S.Context));
  } else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << "device";
  }
}

static void handleGlobalAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (S.LangOpts.CUDA) {
    // check the attribute arguments.
    if (!checkAttributeNumArgs(S, Attr, 0))
      return;

    if (!isa<FunctionDecl>(D)) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
        << Attr.getName() << ExpectedFunction;
      return;
    }

    FunctionDecl *FD = cast<FunctionDecl>(D);
    if (!FD->getResultType()->isVoidType()) {
      TypeLoc TL = FD->getTypeSourceInfo()->getTypeLoc().IgnoreParens();
      if (FunctionTypeLoc* FTL = dyn_cast<FunctionTypeLoc>(&TL)) {
        S.Diag(FD->getTypeSpecStartLoc(), diag::err_kern_type_not_void_return)
          << FD->getType()
          << FixItHint::CreateReplacement(FTL->getResultLoc().getSourceRange(),
                                          "void");
      } else {
        S.Diag(FD->getTypeSpecStartLoc(), diag::err_kern_type_not_void_return)
          << FD->getType();
      }
      return;
    }

    D->addAttr(::new (S.Context) CUDAGlobalAttr(Attr.getLoc(), S.Context));
  } else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << "global";
  }
}

static void handleHostAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (S.LangOpts.CUDA) {
    // check the attribute arguments.
    if (!checkAttributeNumArgs(S, Attr, 0))
      return;


    if (!isa<FunctionDecl>(D)) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
        << Attr.getName() << ExpectedFunction;
      return;
    }

    D->addAttr(::new (S.Context) CUDAHostAttr(Attr.getLoc(), S.Context));
  } else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << "host";
  }
}

static void handleSharedAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (S.LangOpts.CUDA) {
    // check the attribute arguments.
    if (!checkAttributeNumArgs(S, Attr, 0))
      return;


    if (!isa<VarDecl>(D)) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
        << Attr.getName() << ExpectedVariable;
      return;
    }

    D->addAttr(::new (S.Context) CUDASharedAttr(Attr.getLoc(), S.Context));
  } else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << "shared";
  }
}

static void handleGNUInlineAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  // check the attribute arguments.
  if (!checkAttributeNumArgs(S, Attr, 0))
    return;

  FunctionDecl *Fn = dyn_cast<FunctionDecl>(D);
  if (Fn == 0) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunction;
    return;
  }

  if (!Fn->isInlineSpecified()) {
    S.Diag(Attr.getLoc(), diag::warn_gnu_inline_attribute_requires_inline);
    return;
  }

  D->addAttr(::new (S.Context) GNUInlineAttr(Attr.getLoc(), S.Context));
}

static void handleCallConvAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (hasDeclarator(D)) return;

  // Diagnostic is emitted elsewhere: here we store the (valid) Attr
  // in the Decl node for syntactic reasoning, e.g., pretty-printing.
  CallingConv CC;
  if (S.CheckCallingConvAttr(Attr, CC))
    return;

  if (!isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionOrMethod;
    return;
  }

  switch (Attr.getKind()) {
  case AttributeList::AT_fastcall:
    D->addAttr(::new (S.Context) FastCallAttr(Attr.getLoc(), S.Context));
    return;
  case AttributeList::AT_stdcall:
    D->addAttr(::new (S.Context) StdCallAttr(Attr.getLoc(), S.Context));
    return;
  case AttributeList::AT_thiscall:
    D->addAttr(::new (S.Context) ThisCallAttr(Attr.getLoc(), S.Context));
    return;
  case AttributeList::AT_cdecl:
    D->addAttr(::new (S.Context) CDeclAttr(Attr.getLoc(), S.Context));
    return;
  case AttributeList::AT_pascal:
    D->addAttr(::new (S.Context) PascalAttr(Attr.getLoc(), S.Context));
    return;
  case AttributeList::AT_pcs: {
    Expr *Arg = Attr.getArg(0);
    StringLiteral *Str = dyn_cast<StringLiteral>(Arg);
    if (!Str || !Str->isAscii()) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string)
        << "pcs" << 1;
      Attr.setInvalid();
      return;
    }

    StringRef StrRef = Str->getString();
    PcsAttr::PCSType PCS;
    if (StrRef == "aapcs")
      PCS = PcsAttr::AAPCS;
    else if (StrRef == "aapcs-vfp")
      PCS = PcsAttr::AAPCS_VFP;
    else {
      S.Diag(Attr.getLoc(), diag::err_invalid_pcs);
      Attr.setInvalid();
      return;
    }

    D->addAttr(::new (S.Context) PcsAttr(Attr.getLoc(), S.Context, PCS));
  }
  default:
    llvm_unreachable("unexpected attribute kind");
    return;
  }
}

static void handleOpenCLKernelAttr(Sema &S, Decl *D, const AttributeList &Attr){
  assert(!Attr.isInvalid());
  D->addAttr(::new (S.Context) OpenCLKernelAttr(Attr.getLoc(), S.Context));
}

bool Sema::CheckCallingConvAttr(const AttributeList &attr, CallingConv &CC) {
  if (attr.isInvalid())
    return true;

  if ((attr.getNumArgs() != 0 &&
      !(attr.getKind() == AttributeList::AT_pcs && attr.getNumArgs() == 1)) ||
      attr.getParameterName()) {
    Diag(attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    attr.setInvalid();
    return true;
  }

  // TODO: diagnose uses of these conventions on the wrong target. Or, better
  // move to TargetAttributesSema one day.
  switch (attr.getKind()) {
  case AttributeList::AT_cdecl: CC = CC_C; break;
  case AttributeList::AT_fastcall: CC = CC_X86FastCall; break;
  case AttributeList::AT_stdcall: CC = CC_X86StdCall; break;
  case AttributeList::AT_thiscall: CC = CC_X86ThisCall; break;
  case AttributeList::AT_pascal: CC = CC_X86Pascal; break;
  case AttributeList::AT_pcs: {
    Expr *Arg = attr.getArg(0);
    StringLiteral *Str = dyn_cast<StringLiteral>(Arg);
    if (!Str || !Str->isAscii()) {
      Diag(attr.getLoc(), diag::err_attribute_argument_n_not_string)
        << "pcs" << 1;
      attr.setInvalid();
      return true;
    }

    StringRef StrRef = Str->getString();
    if (StrRef == "aapcs") {
      CC = CC_AAPCS;
      break;
    } else if (StrRef == "aapcs-vfp") {
      CC = CC_AAPCS_VFP;
      break;
    }
    // FALLS THROUGH
  }
  default: llvm_unreachable("unexpected attribute kind"); return true;
  }

  return false;
}

static void handleRegparmAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (hasDeclarator(D)) return;

  unsigned numParams;
  if (S.CheckRegparmAttr(Attr, numParams))
    return;

  if (!isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << ExpectedFunctionOrMethod;
    return;
  }

  D->addAttr(::new (S.Context) RegparmAttr(Attr.getLoc(), S.Context, numParams));
}

/// Checks a regparm attribute, returning true if it is ill-formed and
/// otherwise setting numParams to the appropriate value.
bool Sema::CheckRegparmAttr(const AttributeList &Attr, unsigned &numParams) {
  if (Attr.isInvalid())
    return true;

  if (Attr.getNumArgs() != 1) {
    Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    Attr.setInvalid();
    return true;
  }

  Expr *NumParamsExpr = Attr.getArg(0);
  llvm::APSInt NumParams(32);
  if (NumParamsExpr->isTypeDependent() || NumParamsExpr->isValueDependent() ||
      !NumParamsExpr->isIntegerConstantExpr(NumParams, Context)) {
    Diag(Attr.getLoc(), diag::err_attribute_argument_not_int)
      << "regparm" << NumParamsExpr->getSourceRange();
    Attr.setInvalid();
    return true;
  }

  if (Context.Target.getRegParmMax() == 0) {
    Diag(Attr.getLoc(), diag::err_attribute_regparm_wrong_platform)
      << NumParamsExpr->getSourceRange();
    Attr.setInvalid();
    return true;
  }

  numParams = NumParams.getZExtValue();
  if (numParams > Context.Target.getRegParmMax()) {
    Diag(Attr.getLoc(), diag::err_attribute_regparm_invalid_number)
      << Context.Target.getRegParmMax() << NumParamsExpr->getSourceRange();
    Attr.setInvalid();
    return true;
  }

  return false;
}

static void handleLaunchBoundsAttr(Sema &S, Decl *D, const AttributeList &Attr){
  if (S.LangOpts.CUDA) {
    // check the attribute arguments.
    if (Attr.getNumArgs() != 1 && Attr.getNumArgs() != 2) {
      // FIXME: 0 is not okay.
      S.Diag(Attr.getLoc(), diag::err_attribute_too_many_arguments) << 2;
      return;
    }

    if (!isFunctionOrMethod(D)) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
        << Attr.getName() << ExpectedFunctionOrMethod;
      return;
    }

    Expr *MaxThreadsExpr = Attr.getArg(0);
    llvm::APSInt MaxThreads(32);
    if (MaxThreadsExpr->isTypeDependent() ||
        MaxThreadsExpr->isValueDependent() ||
        !MaxThreadsExpr->isIntegerConstantExpr(MaxThreads, S.Context)) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_int)
        << "launch_bounds" << 1 << MaxThreadsExpr->getSourceRange();
      return;
    }

    llvm::APSInt MinBlocks(32);
    if (Attr.getNumArgs() > 1) {
      Expr *MinBlocksExpr = Attr.getArg(1);
      if (MinBlocksExpr->isTypeDependent() ||
          MinBlocksExpr->isValueDependent() ||
          !MinBlocksExpr->isIntegerConstantExpr(MinBlocks, S.Context)) {
        S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_int)
          << "launch_bounds" << 2 << MinBlocksExpr->getSourceRange();
        return;
      }
    }

    D->addAttr(::new (S.Context) CUDALaunchBoundsAttr(Attr.getLoc(), S.Context,
                                                      MaxThreads.getZExtValue(),
                                                     MinBlocks.getZExtValue()));
  } else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << "launch_bounds";
  }
}

//===----------------------------------------------------------------------===//
// Checker-specific attribute handlers.
//===----------------------------------------------------------------------===//

static bool isValidSubjectOfNSAttribute(Sema &S, QualType type) {
  return type->isObjCObjectPointerType() || S.Context.isObjCNSObjectType(type);
}
static bool isValidSubjectOfCFAttribute(Sema &S, QualType type) {
  return type->isPointerType() || isValidSubjectOfNSAttribute(S, type);
}

static void handleNSConsumedAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  ParmVarDecl *param = dyn_cast<ParmVarDecl>(D);
  if (!param) {
    S.Diag(D->getLocStart(), diag::warn_attribute_wrong_decl_type)
      << SourceRange(Attr.getLoc()) << Attr.getName() << ExpectedParameter;
    return;
  }

  bool typeOK, cf;
  if (Attr.getKind() == AttributeList::AT_ns_consumed) {
    typeOK = isValidSubjectOfNSAttribute(S, param->getType());
    cf = false;
  } else {
    typeOK = isValidSubjectOfCFAttribute(S, param->getType());
    cf = true;
  }

  if (!typeOK) {
    S.Diag(D->getLocStart(), diag::warn_ns_attribute_wrong_parameter_type)
      << SourceRange(Attr.getLoc()) << Attr.getName() << cf;
    return;
  }

  if (cf)
    param->addAttr(::new (S.Context) CFConsumedAttr(Attr.getLoc(), S.Context));
  else
    param->addAttr(::new (S.Context) NSConsumedAttr(Attr.getLoc(), S.Context));
}

static void handleNSConsumesSelfAttr(Sema &S, Decl *D,
                                     const AttributeList &Attr) {
  if (!isa<ObjCMethodDecl>(D)) {
    S.Diag(D->getLocStart(), diag::warn_attribute_wrong_decl_type)
      << SourceRange(Attr.getLoc()) << Attr.getName() << ExpectedMethod;
    return;
  }

  D->addAttr(::new (S.Context) NSConsumesSelfAttr(Attr.getLoc(), S.Context));
}

static void handleNSReturnsRetainedAttr(Sema &S, Decl *D,
                                        const AttributeList &Attr) {

  QualType returnType;

  if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D))
    returnType = MD->getResultType();
  else if (ObjCPropertyDecl *PD = dyn_cast<ObjCPropertyDecl>(D))
    returnType = PD->getType();
  else if (S.getLangOptions().ObjCAutoRefCount && hasDeclarator(D) &&
           (Attr.getKind() == AttributeList::AT_ns_returns_retained))
    return; // ignore: was handled as a type attribute
  else if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
    returnType = FD->getResultType();
  else {
    S.Diag(D->getLocStart(), diag::warn_attribute_wrong_decl_type)
        << SourceRange(Attr.getLoc()) << Attr.getName()
        << ExpectedFunctionOrMethod;
    return;
  }

  bool typeOK;
  bool cf;
  switch (Attr.getKind()) {
  default: llvm_unreachable("invalid ownership attribute"); return;
  case AttributeList::AT_ns_returns_autoreleased:
  case AttributeList::AT_ns_returns_retained:
  case AttributeList::AT_ns_returns_not_retained:
    typeOK = isValidSubjectOfNSAttribute(S, returnType);
    cf = false;
    break;

  case AttributeList::AT_cf_returns_retained:
  case AttributeList::AT_cf_returns_not_retained:
    typeOK = isValidSubjectOfCFAttribute(S, returnType);
    cf = true;
    break;
  }

  if (!typeOK) {
    S.Diag(D->getLocStart(), diag::warn_ns_attribute_wrong_return_type)
      << SourceRange(Attr.getLoc())
      << Attr.getName() << isa<ObjCMethodDecl>(D) << cf;
    return;
  }

  switch (Attr.getKind()) {
    default:
      assert(0 && "invalid ownership attribute");
      return;
    case AttributeList::AT_ns_returns_autoreleased:
      D->addAttr(::new (S.Context) NSReturnsAutoreleasedAttr(Attr.getLoc(),
                                                             S.Context));
      return;
    case AttributeList::AT_cf_returns_not_retained:
      D->addAttr(::new (S.Context) CFReturnsNotRetainedAttr(Attr.getLoc(),
                                                            S.Context));
      return;
    case AttributeList::AT_ns_returns_not_retained:
      D->addAttr(::new (S.Context) NSReturnsNotRetainedAttr(Attr.getLoc(),
                                                            S.Context));
      return;
    case AttributeList::AT_cf_returns_retained:
      D->addAttr(::new (S.Context) CFReturnsRetainedAttr(Attr.getLoc(),
                                                         S.Context));
      return;
    case AttributeList::AT_ns_returns_retained:
      D->addAttr(::new (S.Context) NSReturnsRetainedAttr(Attr.getLoc(),
                                                         S.Context));
      return;
  };
}

static void handleObjCReturnsInnerPointerAttr(Sema &S, Decl *D,
                                              const AttributeList &attr) {
  SourceLocation loc = attr.getLoc();

  ObjCMethodDecl *method = dyn_cast<ObjCMethodDecl>(D);

  if (!isa<ObjCMethodDecl>(method)) {
    S.Diag(method->getLocStart(), diag::err_attribute_wrong_decl_type)
      << SourceRange(loc, loc) << attr.getName() << 13 /* methods */;
    return;
  }

  // Check that the method returns a normal pointer.
  QualType resultType = method->getResultType();
  if (!resultType->isPointerType() || resultType->isObjCRetainableType()) {
    S.Diag(method->getLocStart(), diag::warn_ns_attribute_wrong_return_type)
      << SourceRange(loc)
      << attr.getName() << /*method*/ 1 << /*non-retainable pointer*/ 2;

    // Drop the attribute.
    return;
  }

  method->addAttr(
    ::new (S.Context) ObjCReturnsInnerPointerAttr(loc, S.Context));
}

static void handleObjCOwnershipAttr(Sema &S, Decl *D,
                                    const AttributeList &Attr) {
  if (hasDeclarator(D)) return;

  SourceLocation L = Attr.getLoc();
  S.Diag(D->getLocStart(), diag::err_attribute_wrong_decl_type)
    << SourceRange(L, L) << Attr.getName() << 12 /* variable */;
}

static void handleObjCPreciseLifetimeAttr(Sema &S, Decl *D,
                                          const AttributeList &Attr) {
  if (!isa<VarDecl>(D) && !isa<FieldDecl>(D)) {
    SourceLocation L = Attr.getLoc();
    S.Diag(D->getLocStart(), diag::err_attribute_wrong_decl_type)
      << SourceRange(L, L) << Attr.getName() << 12 /* variable */;
    return;
  }

  ValueDecl *vd = cast<ValueDecl>(D);
  QualType type = vd->getType();

  if (!type->isDependentType() &&
      !type->isObjCLifetimeType()) {
    S.Diag(Attr.getLoc(), diag::err_objc_precise_lifetime_bad_type)
      << type;
    return;
  }

  Qualifiers::ObjCLifetime lifetime = type.getObjCLifetime();

  // If we have no lifetime yet, check the lifetime we're presumably
  // going to infer.
  if (lifetime == Qualifiers::OCL_None && !type->isDependentType())
    lifetime = type->getObjCARCImplicitLifetime();

  switch (lifetime) {
  case Qualifiers::OCL_None:
    assert(type->isDependentType() &&
           "didn't infer lifetime for non-dependent type?");
    break;

  case Qualifiers::OCL_Weak:   // meaningful
  case Qualifiers::OCL_Strong: // meaningful
    break;

  case Qualifiers::OCL_ExplicitNone:
  case Qualifiers::OCL_Autoreleasing:
    S.Diag(Attr.getLoc(), diag::warn_objc_precise_lifetime_meaningless)
      << (lifetime == Qualifiers::OCL_Autoreleasing);
    break;
  }

  D->addAttr(::new (S.Context)
                 ObjCPreciseLifetimeAttr(Attr.getLoc(), S.Context));
}

static bool isKnownDeclSpecAttr(const AttributeList &Attr) {
  return Attr.getKind() == AttributeList::AT_dllimport ||
         Attr.getKind() == AttributeList::AT_dllexport ||
         Attr.getKind() == AttributeList::AT_uuid;
}

//===----------------------------------------------------------------------===//
// Microsoft specific attribute handlers.
//===----------------------------------------------------------------------===//

static void handleUuidAttr(Sema &S, Decl *D, const AttributeList &Attr) {
  if (S.LangOpts.Microsoft || S.LangOpts.Borland) {
    // check the attribute arguments.
    if (!checkAttributeNumArgs(S, Attr, 1))
      return;

    Expr *Arg = Attr.getArg(0);
    StringLiteral *Str = dyn_cast<StringLiteral>(Arg);
    if (!Str || !Str->isAscii()) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string)
        << "uuid" << 1;
      return;
    }

    StringRef StrRef = Str->getString();

    bool IsCurly = StrRef.size() > 1 && StrRef.front() == '{' &&
                   StrRef.back() == '}';
    
    // Validate GUID length.
    if (IsCurly && StrRef.size() != 38) {
      S.Diag(Attr.getLoc(), diag::err_attribute_uuid_malformed_guid);
      return;
    }
    if (!IsCurly && StrRef.size() != 36) {
      S.Diag(Attr.getLoc(), diag::err_attribute_uuid_malformed_guid);
      return;
    }

    // GUID format is "XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX" or 
    // "{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}"
    StringRef::iterator I = StrRef.begin();
    if (IsCurly) // Skip the optional '{'
       ++I;

    for (int i = 0; i < 36; ++i) {
      if (i == 8 || i == 13 || i == 18 || i == 23) {
        if (*I != '-') {
          S.Diag(Attr.getLoc(), diag::err_attribute_uuid_malformed_guid);
          return;
        }
      } else if (!isxdigit(*I)) {
        S.Diag(Attr.getLoc(), diag::err_attribute_uuid_malformed_guid);
        return;
      }
      I++;
    }

    D->addAttr(::new (S.Context) UuidAttr(Attr.getLoc(), S.Context,
                                          Str->getString()));
  } else
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << "uuid";
}

//===----------------------------------------------------------------------===//
// Top Level Sema Entry Points
//===----------------------------------------------------------------------===//

static void ProcessNonInheritableDeclAttr(Sema &S, Scope *scope, Decl *D,
                                          const AttributeList &Attr) {
  switch (Attr.getKind()) {
  case AttributeList::AT_device:      handleDeviceAttr      (S, D, Attr); break;
  case AttributeList::AT_host:        handleHostAttr        (S, D, Attr); break;
  case AttributeList::AT_overloadable:handleOverloadableAttr(S, D, Attr); break;
  default:
    break;
  }
}

static void ProcessInheritableDeclAttr(Sema &S, Scope *scope, Decl *D,
                                       const AttributeList &Attr) {
  switch (Attr.getKind()) {
  case AttributeList::AT_IBAction:            handleIBAction(S, D, Attr); break;
    case AttributeList::AT_IBOutlet:          handleIBOutlet(S, D, Attr); break;
  case AttributeList::AT_IBOutletCollection:
      handleIBOutletCollection(S, D, Attr); break;
  case AttributeList::AT_address_space:
  case AttributeList::AT_opencl_image_access:
  case AttributeList::AT_objc_gc:
  case AttributeList::AT_vector_size:
  case AttributeList::AT_neon_vector_type:
  case AttributeList::AT_neon_polyvector_type:
    // Ignore these, these are type attributes, handled by
    // ProcessTypeAttributes.
    break;
  case AttributeList::AT_device:
  case AttributeList::AT_host:
  case AttributeList::AT_overloadable:
    // Ignore, this is a non-inheritable attribute, handled
    // by ProcessNonInheritableDeclAttr.
    break;
  case AttributeList::AT_alias:       handleAliasAttr       (S, D, Attr); break;
  case AttributeList::AT_aligned:     handleAlignedAttr     (S, D, Attr); break;
  case AttributeList::AT_always_inline:
    handleAlwaysInlineAttr  (S, D, Attr); break;
  case AttributeList::AT_analyzer_noreturn:
    handleAnalyzerNoReturnAttr  (S, D, Attr); break;
  case AttributeList::AT_annotate:    handleAnnotateAttr    (S, D, Attr); break;
  case AttributeList::AT_availability:handleAvailabilityAttr(S, D, Attr); break;
  case AttributeList::AT_carries_dependency:
                                      handleDependencyAttr  (S, D, Attr); break;
  case AttributeList::AT_common:      handleCommonAttr      (S, D, Attr); break;
  case AttributeList::AT_constant:    handleConstantAttr    (S, D, Attr); break;
  case AttributeList::AT_constructor: handleConstructorAttr (S, D, Attr); break;
  case AttributeList::AT_deprecated:  handleDeprecatedAttr  (S, D, Attr); break;
  case AttributeList::AT_destructor:  handleDestructorAttr  (S, D, Attr); break;
  case AttributeList::AT_ext_vector_type:
    handleExtVectorTypeAttr(S, scope, D, Attr);
    break;
  case AttributeList::AT_format:      handleFormatAttr      (S, D, Attr); break;
  case AttributeList::AT_format_arg:  handleFormatArgAttr   (S, D, Attr); break;
  case AttributeList::AT_global:      handleGlobalAttr      (S, D, Attr); break;
  case AttributeList::AT_gnu_inline:  handleGNUInlineAttr   (S, D, Attr); break;
  case AttributeList::AT_launch_bounds:
    handleLaunchBoundsAttr(S, D, Attr);
    break;
  case AttributeList::AT_mode:        handleModeAttr        (S, D, Attr); break;
  case AttributeList::AT_malloc:      handleMallocAttr      (S, D, Attr); break;
  case AttributeList::AT_may_alias:   handleMayAliasAttr    (S, D, Attr); break;
  case AttributeList::AT_nocommon:    handleNoCommonAttr    (S, D, Attr); break;
  case AttributeList::AT_nonnull:     handleNonNullAttr     (S, D, Attr); break;
  case AttributeList::AT_ownership_returns:
  case AttributeList::AT_ownership_takes:
  case AttributeList::AT_ownership_holds:
      handleOwnershipAttr     (S, D, Attr); break;
  case AttributeList::AT_naked:       handleNakedAttr       (S, D, Attr); break;
  case AttributeList::AT_noreturn:    handleNoReturnAttr    (S, D, Attr); break;
  case AttributeList::AT_nothrow:     handleNothrowAttr     (S, D, Attr); break;
  case AttributeList::AT_shared:      handleSharedAttr      (S, D, Attr); break;
  case AttributeList::AT_vecreturn:   handleVecReturnAttr   (S, D, Attr); break;

  case AttributeList::AT_objc_ownership:
    handleObjCOwnershipAttr(S, D, Attr); break;
  case AttributeList::AT_objc_precise_lifetime:
    handleObjCPreciseLifetimeAttr(S, D, Attr); break;

  case AttributeList::AT_objc_returns_inner_pointer:
    handleObjCReturnsInnerPointerAttr(S, D, Attr); break;

  // Checker-specific.
  case AttributeList::AT_cf_consumed:
  case AttributeList::AT_ns_consumed: handleNSConsumedAttr  (S, D, Attr); break;
  case AttributeList::AT_ns_consumes_self:
    handleNSConsumesSelfAttr(S, D, Attr); break;

  case AttributeList::AT_ns_returns_autoreleased:
  case AttributeList::AT_ns_returns_not_retained:
  case AttributeList::AT_cf_returns_not_retained:
  case AttributeList::AT_ns_returns_retained:
  case AttributeList::AT_cf_returns_retained:
    handleNSReturnsRetainedAttr(S, D, Attr); break;

  case AttributeList::AT_reqd_wg_size:
    handleReqdWorkGroupSize(S, D, Attr); break;

  case AttributeList::AT_init_priority: 
      handleInitPriorityAttr(S, D, Attr); break;
      
  case AttributeList::AT_packed:      handlePackedAttr      (S, D, Attr); break;
  case AttributeList::AT_MsStruct:    handleMsStructAttr    (S, D, Attr); break;
  case AttributeList::AT_section:     handleSectionAttr     (S, D, Attr); break;
  case AttributeList::AT_unavailable: handleUnavailableAttr (S, D, Attr); break;
  case AttributeList::AT_arc_weakref_unavailable: 
    handleArcWeakrefUnavailableAttr (S, D, Attr); 
    break;
  case AttributeList::AT_unused:      handleUnusedAttr      (S, D, Attr); break;
  case AttributeList::AT_used:        handleUsedAttr        (S, D, Attr); break;
  case AttributeList::AT_visibility:  handleVisibilityAttr  (S, D, Attr); break;
  case AttributeList::AT_warn_unused_result: handleWarnUnusedResult(S, D, Attr);
    break;
  case AttributeList::AT_weak:        handleWeakAttr        (S, D, Attr); break;
  case AttributeList::AT_weakref:     handleWeakRefAttr     (S, D, Attr); break;
  case AttributeList::AT_weak_import: handleWeakImportAttr  (S, D, Attr); break;
  case AttributeList::AT_transparent_union:
    handleTransparentUnionAttr(S, D, Attr);
    break;
  case AttributeList::AT_objc_exception:
    handleObjCExceptionAttr(S, D, Attr);
    break;
  case AttributeList::AT_objc_method_family:
    handleObjCMethodFamilyAttr(S, D, Attr);
    break;
  case AttributeList::AT_nsobject:    handleObjCNSObject    (S, D, Attr); break;
  case AttributeList::AT_blocks:      handleBlocksAttr      (S, D, Attr); break;
  case AttributeList::AT_sentinel:    handleSentinelAttr    (S, D, Attr); break;
  case AttributeList::AT_const:       handleConstAttr       (S, D, Attr); break;
  case AttributeList::AT_pure:        handlePureAttr        (S, D, Attr); break;
  case AttributeList::AT_cleanup:     handleCleanupAttr     (S, D, Attr); break;
  case AttributeList::AT_nodebug:     handleNoDebugAttr     (S, D, Attr); break;
  case AttributeList::AT_noinline:    handleNoInlineAttr    (S, D, Attr); break;
  case AttributeList::AT_regparm:     handleRegparmAttr     (S, D, Attr); break;
  case AttributeList::IgnoredAttribute:
    // Just ignore
    break;
  case AttributeList::AT_no_instrument_function:  // Interacts with -pg.
    handleNoInstrumentFunctionAttr(S, D, Attr);
    break;
  case AttributeList::AT_stdcall:
  case AttributeList::AT_cdecl:
  case AttributeList::AT_fastcall:
  case AttributeList::AT_thiscall:
  case AttributeList::AT_pascal:
  case AttributeList::AT_pcs:
    handleCallConvAttr(S, D, Attr);
    break;
  case AttributeList::AT_opencl_kernel_function:
    handleOpenCLKernelAttr(S, D, Attr);
    break;
  case AttributeList::AT_uuid:
    handleUuidAttr(S, D, Attr);
    break;

  // Thread safety attributes:
  case AttributeList::AT_guarded_var:
    handleGuardedVarAttr(S, D, Attr);
    break;
  case AttributeList::AT_pt_guarded_var:
    handleGuardedVarAttr(S, D, Attr, /*pointer = */true);
    break;
  case AttributeList::AT_scoped_lockable:
    handleLockableAttr(S, D, Attr, /*scoped = */true);
    break;
  case AttributeList::AT_no_thread_safety_analysis:
    handleNoThreadSafetyAttr(S, D, Attr);
    break;
  case AttributeList::AT_lockable:
    handleLockableAttr(S, D, Attr);
    break;
  case AttributeList::AT_guarded_by:
    handleGuardedByAttr(S, D, Attr);
    break;
  case AttributeList::AT_pt_guarded_by:
    handleGuardedByAttr(S, D, Attr, /*pointer = */true);
    break;
  case AttributeList::AT_exclusive_lock_function:
    handleLockFunAttr(S, D, Attr, /*exclusive = */true);
    break;
  case AttributeList::AT_exclusive_locks_required:
    handleLocksRequiredAttr(S, D, Attr, /*exclusive = */true);
    break;
  case AttributeList::AT_exclusive_trylock_function:
    handleTrylockFunAttr(S, D, Attr, /*exclusive = */true);
    break;
  case AttributeList::AT_lock_returned:
    handleLockReturnedAttr(S, D, Attr);
    break;
  case AttributeList::AT_locks_excluded:
    handleLocksExcludedAttr(S, D, Attr);
    break;
  case AttributeList::AT_shared_lock_function:
    handleLockFunAttr(S, D, Attr);
    break;
  case AttributeList::AT_shared_locks_required:
    handleLocksRequiredAttr(S, D, Attr);
    break;
  case AttributeList::AT_shared_trylock_function:
    handleTrylockFunAttr(S, D, Attr);
    break;
  case AttributeList::AT_unlock_function:
    handleUnlockFunAttr(S, D, Attr);
    break;
  case AttributeList::AT_acquired_before:
    handleAcquireOrderAttr(S, D, Attr, /*before = */true);
    break;
  case AttributeList::AT_acquired_after:
    handleAcquireOrderAttr(S, D, Attr, /*before = */false);
    break;

  default:
    // Ask target about the attribute.
    const TargetAttributesSema &TargetAttrs = S.getTargetAttributesSema();
    if (!TargetAttrs.ProcessDeclAttribute(scope, D, Attr, S))
      S.Diag(Attr.getLoc(), diag::warn_unknown_attribute_ignored)
        << Attr.getName();
    break;
  }
}

/// ProcessDeclAttribute - Apply the specific attribute to the specified decl if
/// the attribute applies to decls.  If the attribute is a type attribute, just
/// silently ignore it if a GNU attribute. FIXME: Applying a C++0x attribute to
/// the wrong thing is illegal (C++0x [dcl.attr.grammar]/4).
static void ProcessDeclAttribute(Sema &S, Scope *scope, Decl *D,
                                 const AttributeList &Attr,
                                 bool NonInheritable, bool Inheritable) {
  if (Attr.isInvalid())
    return;

  if (Attr.isDeclspecAttribute() && !isKnownDeclSpecAttr(Attr))
    // FIXME: Try to deal with other __declspec attributes!
    return;

  if (NonInheritable)
    ProcessNonInheritableDeclAttr(S, scope, D, Attr);

  if (Inheritable)
    ProcessInheritableDeclAttr(S, scope, D, Attr);
}

/// ProcessDeclAttributeList - Apply all the decl attributes in the specified
/// attribute list to the specified decl, ignoring any type attributes.
void Sema::ProcessDeclAttributeList(Scope *S, Decl *D,
                                    const AttributeList *AttrList,
                                    bool NonInheritable, bool Inheritable) {
  for (const AttributeList* l = AttrList; l; l = l->getNext()) {
    ProcessDeclAttribute(*this, S, D, *l, NonInheritable, Inheritable);
  }

  // GCC accepts
  // static int a9 __attribute__((weakref));
  // but that looks really pointless. We reject it.
  if (Inheritable && D->hasAttr<WeakRefAttr>() && !D->hasAttr<AliasAttr>()) {
    Diag(AttrList->getLoc(), diag::err_attribute_weakref_without_alias) <<
    dyn_cast<NamedDecl>(D)->getNameAsString();
    return;
  }
}

/// DeclClonePragmaWeak - clone existing decl (maybe definition),
/// #pragma weak needs a non-definition decl and source may not have one
NamedDecl * Sema::DeclClonePragmaWeak(NamedDecl *ND, IdentifierInfo *II) {
  assert(isa<FunctionDecl>(ND) || isa<VarDecl>(ND));
  NamedDecl *NewD = 0;
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(ND)) {
    NewD = FunctionDecl::Create(FD->getASTContext(), FD->getDeclContext(),
                                FD->getInnerLocStart(),
                                FD->getLocation(), DeclarationName(II),
                                FD->getType(), FD->getTypeSourceInfo());
    if (FD->getQualifier()) {
      FunctionDecl *NewFD = cast<FunctionDecl>(NewD);
      NewFD->setQualifierInfo(FD->getQualifierLoc());
    }
  } else if (VarDecl *VD = dyn_cast<VarDecl>(ND)) {
    NewD = VarDecl::Create(VD->getASTContext(), VD->getDeclContext(),
                           VD->getInnerLocStart(), VD->getLocation(), II,
                           VD->getType(), VD->getTypeSourceInfo(),
                           VD->getStorageClass(),
                           VD->getStorageClassAsWritten());
    if (VD->getQualifier()) {
      VarDecl *NewVD = cast<VarDecl>(NewD);
      NewVD->setQualifierInfo(VD->getQualifierLoc());
    }
  }
  return NewD;
}

/// DeclApplyPragmaWeak - A declaration (maybe definition) needs #pragma weak
/// applied to it, possibly with an alias.
void Sema::DeclApplyPragmaWeak(Scope *S, NamedDecl *ND, WeakInfo &W) {
  if (W.getUsed()) return; // only do this once
  W.setUsed(true);
  if (W.getAlias()) { // clone decl, impersonate __attribute(weak,alias(...))
    IdentifierInfo *NDId = ND->getIdentifier();
    NamedDecl *NewD = DeclClonePragmaWeak(ND, W.getAlias());
    NewD->addAttr(::new (Context) AliasAttr(W.getLocation(), Context,
                                            NDId->getName()));
    NewD->addAttr(::new (Context) WeakAttr(W.getLocation(), Context));
    WeakTopLevelDecl.push_back(NewD);
    // FIXME: "hideous" code from Sema::LazilyCreateBuiltin
    // to insert Decl at TU scope, sorry.
    DeclContext *SavedContext = CurContext;
    CurContext = Context.getTranslationUnitDecl();
    PushOnScopeChains(NewD, S);
    CurContext = SavedContext;
  } else { // just add weak to existing
    ND->addAttr(::new (Context) WeakAttr(W.getLocation(), Context));
  }
}

/// ProcessDeclAttributes - Given a declarator (PD) with attributes indicated in
/// it, apply them to D.  This is a bit tricky because PD can have attributes
/// specified in many different places, and we need to find and apply them all.
void Sema::ProcessDeclAttributes(Scope *S, Decl *D, const Declarator &PD,
                                 bool NonInheritable, bool Inheritable) {
  // It's valid to "forward-declare" #pragma weak, in which case we
  // have to do this.
  if (Inheritable) {
    LoadExternalWeakUndeclaredIdentifiers();
    if (!WeakUndeclaredIdentifiers.empty()) {
      if (NamedDecl *ND = dyn_cast<NamedDecl>(D)) {
        if (IdentifierInfo *Id = ND->getIdentifier()) {
          llvm::DenseMap<IdentifierInfo*,WeakInfo>::iterator I
            = WeakUndeclaredIdentifiers.find(Id);
          if (I != WeakUndeclaredIdentifiers.end() && ND->hasLinkage()) {
            WeakInfo W = I->second;
            DeclApplyPragmaWeak(S, ND, W);
            WeakUndeclaredIdentifiers[Id] = W;
          }
        }
      }
    }
  }

  // Apply decl attributes from the DeclSpec if present.
  if (const AttributeList *Attrs = PD.getDeclSpec().getAttributes().getList())
    ProcessDeclAttributeList(S, D, Attrs, NonInheritable, Inheritable);

  // Walk the declarator structure, applying decl attributes that were in a type
  // position to the decl itself.  This handles cases like:
  //   int *__attr__(x)** D;
  // when X is a decl attribute.
  for (unsigned i = 0, e = PD.getNumTypeObjects(); i != e; ++i)
    if (const AttributeList *Attrs = PD.getTypeObject(i).getAttrs())
      ProcessDeclAttributeList(S, D, Attrs, NonInheritable, Inheritable);

  // Finally, apply any attributes on the decl itself.
  if (const AttributeList *Attrs = PD.getAttributes())
    ProcessDeclAttributeList(S, D, Attrs, NonInheritable, Inheritable);
}

/// Is the given declaration allowed to use a forbidden type?
static bool isForbiddenTypeAllowed(Sema &S, Decl *decl) {
  // Private ivars are always okay.  Unfortunately, people don't
  // always properly make their ivars private, even in system headers.
  // Plus we need to make fields okay, too.
  if (!isa<FieldDecl>(decl) && !isa<ObjCPropertyDecl>(decl))
    return false;

  // Require it to be declared in a system header.
  return S.Context.getSourceManager().isInSystemHeader(decl->getLocation());
}

/// Handle a delayed forbidden-type diagnostic.
static void handleDelayedForbiddenType(Sema &S, DelayedDiagnostic &diag,
                                       Decl *decl) {
  if (decl && isForbiddenTypeAllowed(S, decl)) {
    decl->addAttr(new (S.Context) UnavailableAttr(diag.Loc, S.Context,
                        "this system declaration uses an unsupported type"));
    return;
  }

  S.Diag(diag.Loc, diag.getForbiddenTypeDiagnostic())
    << diag.getForbiddenTypeOperand() << diag.getForbiddenTypeArgument();
  diag.Triggered = true;
}

// This duplicates a vector push_back but hides the need to know the
// size of the type.
void Sema::DelayedDiagnostics::add(const DelayedDiagnostic &diag) {
  assert(StackSize <= StackCapacity);

  // Grow the stack if necessary.
  if (StackSize == StackCapacity) {
    unsigned newCapacity = 2 * StackCapacity + 2;
    char *newBuffer = new char[newCapacity * sizeof(DelayedDiagnostic)];
    const char *oldBuffer = (const char*) Stack;

    if (StackCapacity)
      memcpy(newBuffer, oldBuffer, StackCapacity * sizeof(DelayedDiagnostic));
    
    delete[] oldBuffer;
    Stack = reinterpret_cast<sema::DelayedDiagnostic*>(newBuffer);
    StackCapacity = newCapacity;
  }

  assert(StackSize < StackCapacity);
  new (&Stack[StackSize++]) DelayedDiagnostic(diag);
}

void Sema::DelayedDiagnostics::popParsingDecl(Sema &S, ParsingDeclState state,
                                              Decl *decl) {
  DelayedDiagnostics &DD = S.DelayedDiagnostics;

  // Check the invariants.
  assert(DD.StackSize >= state.SavedStackSize);
  assert(state.SavedStackSize >= DD.ActiveStackBase);
  assert(DD.ParsingDepth > 0);

  // Drop the parsing depth.
  DD.ParsingDepth--;

  // If there are no active diagnostics, we're done.
  if (DD.StackSize == DD.ActiveStackBase)
    return;

  // We only want to actually emit delayed diagnostics when we
  // successfully parsed a decl.
  if (decl && !decl->isInvalidDecl()) {
    // We emit all the active diagnostics, not just those starting
    // from the saved state.  The idea is this:  we get one push for a
    // decl spec and another for each declarator;  in a decl group like:
    //   deprecated_typedef foo, *bar, baz();
    // only the declarator pops will be passed decls.  This is correct;
    // we really do need to consider delayed diagnostics from the decl spec
    // for each of the different declarations.
    for (unsigned i = DD.ActiveStackBase, e = DD.StackSize; i != e; ++i) {
      DelayedDiagnostic &diag = DD.Stack[i];
      if (diag.Triggered)
        continue;

      switch (diag.Kind) {
      case DelayedDiagnostic::Deprecation:
        S.HandleDelayedDeprecationCheck(diag, decl);
        break;

      case DelayedDiagnostic::Access:
        S.HandleDelayedAccessCheck(diag, decl);
        break;

      case DelayedDiagnostic::ForbiddenType:
        handleDelayedForbiddenType(S, diag, decl);
        break;
      }
    }
  }

  // Destroy all the delayed diagnostics we're about to pop off.
  for (unsigned i = state.SavedStackSize, e = DD.StackSize; i != e; ++i)
    DD.Stack[i].Destroy();

  DD.StackSize = state.SavedStackSize;
}

static bool isDeclDeprecated(Decl *D) {
  do {
    if (D->isDeprecated())
      return true;
  } while ((D = cast_or_null<Decl>(D->getDeclContext())));
  return false;
}

void Sema::HandleDelayedDeprecationCheck(DelayedDiagnostic &DD,
                                         Decl *Ctx) {
  if (isDeclDeprecated(Ctx))
    return;

  DD.Triggered = true;
  if (!DD.getDeprecationMessage().empty())
    Diag(DD.Loc, diag::warn_deprecated_message)
      << DD.getDeprecationDecl()->getDeclName()
      << DD.getDeprecationMessage();
  else
    Diag(DD.Loc, diag::warn_deprecated)
      << DD.getDeprecationDecl()->getDeclName();
}

void Sema::EmitDeprecationWarning(NamedDecl *D, StringRef Message,
                                  SourceLocation Loc,
                                  const ObjCInterfaceDecl *UnknownObjCClass) {
  // Delay if we're currently parsing a declaration.
  if (DelayedDiagnostics.shouldDelayDiagnostics()) {
    DelayedDiagnostics.add(DelayedDiagnostic::makeDeprecation(Loc, D, Message));
    return;
  }

  // Otherwise, don't warn if our current context is deprecated.
  if (isDeclDeprecated(cast<Decl>(CurContext)))
    return;
  if (!Message.empty())
    Diag(Loc, diag::warn_deprecated_message) << D->getDeclName() 
                                             << Message;
  else {
    if (!UnknownObjCClass)
      Diag(Loc, diag::warn_deprecated) << D->getDeclName();
    else {
      Diag(Loc, diag::warn_deprecated_fwdclass_message) << D->getDeclName();
      Diag(UnknownObjCClass->getLocation(), diag::note_forward_class);
    }
  }
}
