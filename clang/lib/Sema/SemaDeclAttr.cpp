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

#include "Sema.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Parse/DeclSpec.h"
#include "llvm/ADT/StringExtras.h"
using namespace clang;

//===----------------------------------------------------------------------===//
//  Helper functions
//===----------------------------------------------------------------------===//

static const FunctionType *getFunctionType(const Decl *d,
                                           bool blocksToo = true) {
  QualType Ty;
  if (const ValueDecl *decl = dyn_cast<ValueDecl>(d))
    Ty = decl->getType();
  else if (const FieldDecl *decl = dyn_cast<FieldDecl>(d))
    Ty = decl->getType();
  else if (const TypedefDecl* decl = dyn_cast<TypedefDecl>(d))
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

/// isFunctionOrMethod - Return true if the given decl has function
/// type (function or function-typed variable).
static bool isFunction(const Decl *d) {
  return getFunctionType(d, false) != NULL;
}

/// isFunctionOrMethod - Return true if the given decl has function
/// type (function or function-typed variable) or an Objective-C
/// method.
static bool isFunctionOrMethod(const Decl *d) {
  return isFunction(d)|| isa<ObjCMethodDecl>(d);
}

/// isFunctionOrMethodOrBlock - Return true if the given decl has function
/// type (function or function-typed variable) or an Objective-C
/// method or a block.
static bool isFunctionOrMethodOrBlock(const Decl *d) {
  if (isFunctionOrMethod(d))
    return true;
  // check for block is more involved.
  if (const VarDecl *V = dyn_cast<VarDecl>(d)) {
    QualType Ty = V->getType();
    return Ty->isBlockPointerType();
  }
  return isa<BlockDecl>(d);
}

/// hasFunctionProto - Return true if the given decl has a argument
/// information. This decl should have already passed
/// isFunctionOrMethod or isFunctionOrMethodOrBlock.
static bool hasFunctionProto(const Decl *d) {
  if (const FunctionType *FnTy = getFunctionType(d))
    return isa<FunctionProtoType>(FnTy);
  else {
    assert(isa<ObjCMethodDecl>(d) || isa<BlockDecl>(d));
    return true;
  }
}

/// getFunctionOrMethodNumArgs - Return number of function or method
/// arguments. It is an error to call this on a K&R function (use
/// hasFunctionProto first).
static unsigned getFunctionOrMethodNumArgs(const Decl *d) {
  if (const FunctionType *FnTy = getFunctionType(d))
    return cast<FunctionProtoType>(FnTy)->getNumArgs();
  if (const BlockDecl *BD = dyn_cast<BlockDecl>(d))
    return BD->getNumParams();
  return cast<ObjCMethodDecl>(d)->param_size();
}

static QualType getFunctionOrMethodArgType(const Decl *d, unsigned Idx) {
  if (const FunctionType *FnTy = getFunctionType(d))
    return cast<FunctionProtoType>(FnTy)->getArgType(Idx);
  if (const BlockDecl *BD = dyn_cast<BlockDecl>(d))
    return BD->getParamDecl(Idx)->getType();

  return cast<ObjCMethodDecl>(d)->param_begin()[Idx]->getType();
}

static QualType getFunctionOrMethodResultType(const Decl *d) {
  if (const FunctionType *FnTy = getFunctionType(d))
    return cast<FunctionProtoType>(FnTy)->getResultType();
  return cast<ObjCMethodDecl>(d)->getResultType();
}

static bool isFunctionOrMethodVariadic(const Decl *d) {
  if (const FunctionType *FnTy = getFunctionType(d)) {
    const FunctionProtoType *proto = cast<FunctionProtoType>(FnTy);
    return proto->isVariadic();
  } else if (const BlockDecl *BD = dyn_cast<BlockDecl>(d))
    return BD->IsVariadic();
  else {
    return cast<ObjCMethodDecl>(d)->isVariadic();
  }
}

static inline bool isNSStringType(QualType T, ASTContext &Ctx) {
  const ObjCObjectPointerType *PT = T->getAs<ObjCObjectPointerType>();
  if (!PT)
    return false;

  const ObjCInterfaceType *ClsT =PT->getPointeeType()->getAs<ObjCInterfaceType>();
  if (!ClsT)
    return false;

  IdentifierInfo* ClsName = ClsT->getDecl()->getIdentifier();

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
  if (RD->getTagKind() != TagDecl::TK_struct)
    return false;

  return RD->getIdentifier() == &Ctx.Idents.get("__CFString");
}

//===----------------------------------------------------------------------===//
// Attribute Implementations
//===----------------------------------------------------------------------===//

// FIXME: All this manual attribute parsing code is gross. At the
// least add some helper functions to check most argument patterns (#
// and types of args).

static void HandleExtVectorTypeAttr(Scope *scope, Decl *d,
                                    const AttributeList &Attr, Sema &S) {
  TypedefDecl *tDecl = dyn_cast<TypedefDecl>(d);
  if (tDecl == 0) {
    S.Diag(Attr.getLoc(), diag::err_typecheck_ext_vector_not_typedef);
    return;
  }

  QualType curType = tDecl->getUnderlyingType();

  Expr *sizeExpr;

  // Special case where the argument is a template id.
  if (Attr.getParameterName()) {
    sizeExpr = S.ActOnDeclarationNameExpr(scope, Attr.getLoc(),
                               Attr.getParameterName(),
                               false, 0, false).takeAs<Expr>();
  } else {
    // check the attribute arguments.
    if (Attr.getNumArgs() != 1) {
      S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
      return;
    }
    sizeExpr = static_cast<Expr *>(Attr.getArg(0));
  }

  // Instantiate/Install the vector type, and let Sema build the type for us.
  // This will run the reguired checks.
  QualType T = S.BuildExtVectorType(curType, S.Owned(sizeExpr), Attr.getLoc());
  if (!T.isNull()) {
    tDecl->setUnderlyingType(T);

    // Remember this typedef decl, we will need it later for diagnostics.
    S.ExtVectorDecls.push_back(tDecl);
  }
}


/// HandleVectorSizeAttribute - this attribute is only applicable to integral
/// and float scalars, although arrays, pointers, and function return values are
/// allowed in conjunction with this construct. Aggregates with this attribute
/// are invalid, even if they are of the same size as a corresponding scalar.
/// The raw attribute should contain precisely 1 argument, the vector size for
/// the variable, measured in bytes. If curType and rawAttr are well formed,
/// this routine will return a new vector type.
static void HandleVectorSizeAttr(Decl *D, const AttributeList &Attr, Sema &S) {
  QualType CurType;
  if (ValueDecl *VD = dyn_cast<ValueDecl>(D))
    CurType = VD->getType();
  else if (TypedefDecl *TD = dyn_cast<TypedefDecl>(D))
    CurType = TD->getUnderlyingType();
  else {
    S.Diag(D->getLocation(), diag::err_attr_wrong_decl)
      << "vector_size" << SourceRange(Attr.getLoc(), Attr.getLoc());
    return;
  }

  // Check the attribute arugments.
  if (Attr.getNumArgs() != 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }
  Expr *sizeExpr = static_cast<Expr *>(Attr.getArg(0));
  llvm::APSInt vecSize(32);
  if (!sizeExpr->isIntegerConstantExpr(vecSize, S.Context)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_not_int)
      << "vector_size" << sizeExpr->getSourceRange();
    return;
  }
  // navigate to the base type - we need to provide for vector pointers, vector
  // arrays, and functions returning vectors.
  if (CurType->isPointerType() || CurType->isArrayType() ||
      CurType->isFunctionType()) {
    S.Diag(Attr.getLoc(), diag::err_unsupported_vector_size) << CurType;
    return;
    /* FIXME: rebuild the type from the inside out, vectorizing the inner type.
     do {
     if (PointerType *PT = dyn_cast<PointerType>(canonType))
     canonType = PT->getPointeeType().getTypePtr();
     else if (ArrayType *AT = dyn_cast<ArrayType>(canonType))
     canonType = AT->getElementType().getTypePtr();
     else if (FunctionType *FT = dyn_cast<FunctionType>(canonType))
     canonType = FT->getResultType().getTypePtr();
     } while (canonType->isPointerType() || canonType->isArrayType() ||
     canonType->isFunctionType());
     */
  }
  // the base type must be integer or float, and can't already be a vector.
  if (CurType->isVectorType() ||
      (!CurType->isIntegerType() && !CurType->isRealFloatingType())) {
    S.Diag(Attr.getLoc(), diag::err_attribute_invalid_vector_type) << CurType;
    return;
  }
  unsigned typeSize = static_cast<unsigned>(S.Context.getTypeSize(CurType));
  // vecSize is specified in bytes - convert to bits.
  unsigned vectorSize = static_cast<unsigned>(vecSize.getZExtValue() * 8);

  // the vector size needs to be an integral multiple of the type size.
  if (vectorSize % typeSize) {
    S.Diag(Attr.getLoc(), diag::err_attribute_invalid_size)
      << sizeExpr->getSourceRange();
    return;
  }
  if (vectorSize == 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_zero_size)
      << sizeExpr->getSourceRange();
    return;
  }

  // Success! Instantiate the vector type, the number of elements is > 0, and
  // not required to be a power of 2, unlike GCC.
  CurType = S.Context.getVectorType(CurType, vectorSize/typeSize);

  if (ValueDecl *VD = dyn_cast<ValueDecl>(D))
    VD->setType(CurType);
  else
    cast<TypedefDecl>(D)->setUnderlyingType(CurType);
}

static void HandlePackedAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() > 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  if (TagDecl *TD = dyn_cast<TagDecl>(d))
    TD->addAttr(::new (S.Context) PackedAttr);
  else if (FieldDecl *FD = dyn_cast<FieldDecl>(d)) {
    // If the alignment is less than or equal to 8 bits, the packed attribute
    // has no effect.
    if (!FD->getType()->isIncompleteType() &&
        S.Context.getTypeAlign(FD->getType()) <= 8)
      S.Diag(Attr.getLoc(), diag::warn_attribute_ignored_for_field_of_type)
        << Attr.getName() << FD->getType();
    else
      FD->addAttr(::new (S.Context) PackedAttr);
  } else
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << Attr.getName();
}

static void HandleIBOutletAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() > 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  // The IBOutlet attribute only applies to instance variables of Objective-C
  // classes.
  if (isa<ObjCIvarDecl>(d) || isa<ObjCPropertyDecl>(d))
    d->addAttr(::new (S.Context) IBOutletAttr());
  else
    S.Diag(Attr.getLoc(), diag::err_attribute_iboutlet);
}

static void HandleNonNullAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // GCC ignores the nonnull attribute on K&R style function prototypes, so we
  // ignore it as well
  if (!isFunctionOrMethod(d) || !hasFunctionProto(d)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << 0 /*function*/;
    return;
  }

  unsigned NumArgs = getFunctionOrMethodNumArgs(d);

  // The nonnull attribute only applies to pointers.
  llvm::SmallVector<unsigned, 10> NonNullArgs;

  for (AttributeList::arg_iterator I=Attr.arg_begin(),
                                   E=Attr.arg_end(); I!=E; ++I) {


    // The argument must be an integer constant expression.
    Expr *Ex = static_cast<Expr *>(*I);
    llvm::APSInt ArgNum(32);
    if (!Ex->isIntegerConstantExpr(ArgNum, S.Context)) {
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

    // Is the function argument a pointer type?
    QualType T = getFunctionOrMethodArgType(d, x);
    if (!T->isAnyPointerType() && !T->isBlockPointerType()) {
      // FIXME: Should also highlight argument in decl.
      S.Diag(Attr.getLoc(), diag::err_nonnull_pointers_only)
        << "nonnull" << Ex->getSourceRange();
      continue;
    }

    NonNullArgs.push_back(x);
  }

  // If no arguments were specified to __attribute__((nonnull)) then all pointer
  // arguments have a nonnull attribute.
  if (NonNullArgs.empty()) {
    for (unsigned I = 0, E = getFunctionOrMethodNumArgs(d); I != E; ++I) {
      QualType T = getFunctionOrMethodArgType(d, I);
      if (T->isAnyPointerType() || T->isBlockPointerType())
        NonNullArgs.push_back(I);
    }

    if (NonNullArgs.empty()) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_nonnull_no_pointers);
      return;
    }
  }

  unsigned* start = &NonNullArgs[0];
  unsigned size = NonNullArgs.size();
  std::sort(start, start + size);
  d->addAttr(::new (S.Context) NonNullAttr(start, size));
}

static void HandleAliasAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }

  Expr *Arg = static_cast<Expr*>(Attr.getArg(0));
  Arg = Arg->IgnoreParenCasts();
  StringLiteral *Str = dyn_cast<StringLiteral>(Arg);

  if (Str == 0 || Str->isWide()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string)
      << "alias" << 1;
    return;
  }

  const char *Alias = Str->getStrData();
  unsigned AliasLen = Str->getByteLength();

  // FIXME: check if target symbol exists in current file

  d->addAttr(::new (S.Context) AliasAttr(std::string(Alias, AliasLen)));
}

static void HandleAlwaysInlineAttr(Decl *d, const AttributeList &Attr,
                                   Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  if (!isa<FunctionDecl>(d)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
    << Attr.getName() << 0 /*function*/;
    return;
  }

  d->addAttr(::new (S.Context) AlwaysInlineAttr());
}

static void HandleMallocAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(d)) {
    QualType RetTy = FD->getResultType();
    if (RetTy->isAnyPointerType() || RetTy->isBlockPointerType()) {
      d->addAttr(::new (S.Context) MallocAttr());
      return;
    }
  }

  S.Diag(Attr.getLoc(), diag::warn_attribute_malloc_pointer_only);
}

static bool HandleCommonNoReturnAttr(Decl *d, const AttributeList &Attr,
                                     Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return false;
  }

  if (!isFunctionOrMethod(d) && !isa<BlockDecl>(d)) {
    ValueDecl *VD = dyn_cast<ValueDecl>(d);
    if (VD == 0 || !VD->getType()->isBlockPointerType()) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
        << Attr.getName() << 0 /*function*/;
      return false;
    }
  }

  return true;
}

static void HandleNoReturnAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  if (HandleCommonNoReturnAttr(d, Attr, S))
    d->addAttr(::new (S.Context) NoReturnAttr());
}

static void HandleAnalyzerNoReturnAttr(Decl *d, const AttributeList &Attr,
                                       Sema &S) {
  if (HandleCommonNoReturnAttr(d, Attr, S))
    d->addAttr(::new (S.Context) AnalyzerNoReturnAttr());
}

static void HandleUnusedAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  if (!isa<VarDecl>(d) && !isFunctionOrMethod(d)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << 2 /*variable and function*/;
    return;
  }

  d->addAttr(::new (S.Context) UnusedAttr());
}

static void HandleUsedAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  if (const VarDecl *VD = dyn_cast<VarDecl>(d)) {
    if (VD->hasLocalStorage() || VD->hasExternalStorage()) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << "used";
      return;
    }
  } else if (!isFunctionOrMethod(d)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << 2 /*variable and function*/;
    return;
  }

  d->addAttr(::new (S.Context) UsedAttr());
}

static void HandleConstructorAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0 && Attr.getNumArgs() != 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments)
      << "0 or 1";
    return;
  }

  int priority = 65535; // FIXME: Do not hardcode such constants.
  if (Attr.getNumArgs() > 0) {
    Expr *E = static_cast<Expr *>(Attr.getArg(0));
    llvm::APSInt Idx(32);
    if (!E->isIntegerConstantExpr(Idx, S.Context)) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_int)
        << "constructor" << 1 << E->getSourceRange();
      return;
    }
    priority = Idx.getZExtValue();
  }

  if (!isa<FunctionDecl>(d)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << 0 /*function*/;
    return;
  }

  d->addAttr(::new (S.Context) ConstructorAttr(priority));
}

static void HandleDestructorAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0 && Attr.getNumArgs() != 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments)
       << "0 or 1";
    return;
  }

  int priority = 65535; // FIXME: Do not hardcode such constants.
  if (Attr.getNumArgs() > 0) {
    Expr *E = static_cast<Expr *>(Attr.getArg(0));
    llvm::APSInt Idx(32);
    if (!E->isIntegerConstantExpr(Idx, S.Context)) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_int)
        << "destructor" << 1 << E->getSourceRange();
      return;
    }
    priority = Idx.getZExtValue();
  }

  if (!isa<FunctionDecl>(d)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << 0 /*function*/;
    return;
  }

  d->addAttr(::new (S.Context) DestructorAttr(priority));
}

static void HandleDeprecatedAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  d->addAttr(::new (S.Context) DeprecatedAttr());
}

static void HandleUnavailableAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  d->addAttr(::new (S.Context) UnavailableAttr());
}

static void HandleVisibilityAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }

  Expr *Arg = static_cast<Expr*>(Attr.getArg(0));
  Arg = Arg->IgnoreParenCasts();
  StringLiteral *Str = dyn_cast<StringLiteral>(Arg);

  if (Str == 0 || Str->isWide()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string)
      << "visibility" << 1;
    return;
  }

  const char *TypeStr = Str->getStrData();
  unsigned TypeLen = Str->getByteLength();
  VisibilityAttr::VisibilityTypes type;

  if (TypeLen == 7 && !memcmp(TypeStr, "default", 7))
    type = VisibilityAttr::DefaultVisibility;
  else if (TypeLen == 6 && !memcmp(TypeStr, "hidden", 6))
    type = VisibilityAttr::HiddenVisibility;
  else if (TypeLen == 8 && !memcmp(TypeStr, "internal", 8))
    type = VisibilityAttr::HiddenVisibility; // FIXME
  else if (TypeLen == 9 && !memcmp(TypeStr, "protected", 9))
    type = VisibilityAttr::ProtectedVisibility;
  else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_unknown_visibility) << TypeStr;
    return;
  }

  d->addAttr(::new (S.Context) VisibilityAttr(type));
}

static void HandleObjCExceptionAttr(Decl *D, const AttributeList &Attr,
                                    Sema &S) {
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  ObjCInterfaceDecl *OCI = dyn_cast<ObjCInterfaceDecl>(D);
  if (OCI == 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_requires_objc_interface);
    return;
  }

  D->addAttr(::new (S.Context) ObjCExceptionAttr());
}

static void HandleObjCNSObject(Decl *D, const AttributeList &Attr, Sema &S) {
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }
  if (TypedefDecl *TD = dyn_cast<TypedefDecl>(D)) {
    QualType T = TD->getUnderlyingType();
    if (!T->isPointerType() ||
        !T->getAs<PointerType>()->getPointeeType()->isRecordType()) {
      S.Diag(TD->getLocation(), diag::err_nsobject_attribute);
      return;
    }
  }
  D->addAttr(::new (S.Context) ObjCNSObjectAttr());
}

static void
HandleOverloadableAttr(Decl *D, const AttributeList &Attr, Sema &S) {
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }

  if (!isa<FunctionDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_overloadable_not_function);
    return;
  }

  D->addAttr(::new (S.Context) OverloadableAttr());
}

static void HandleBlocksAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  if (!Attr.getParameterName()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string)
      << "blocks" << 1;
    return;
  }

  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }

  BlocksAttr::BlocksAttrTypes type;
  if (Attr.getParameterName()->isStr("byref"))
    type = BlocksAttr::ByRef;
  else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_type_not_supported)
      << "blocks" << Attr.getParameterName();
    return;
  }

  d->addAttr(::new (S.Context) BlocksAttr(type));
}

static void HandleSentinelAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() > 2) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments)
      << "0, 1 or 2";
    return;
  }

  int sentinel = 0;
  if (Attr.getNumArgs() > 0) {
    Expr *E = static_cast<Expr *>(Attr.getArg(0));
    llvm::APSInt Idx(32);
    if (!E->isIntegerConstantExpr(Idx, S.Context)) {
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
    Expr *E = static_cast<Expr *>(Attr.getArg(1));
    llvm::APSInt Idx(32);
    if (!E->isIntegerConstantExpr(Idx, S.Context)) {
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

  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(d)) {
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
  } else if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(d)) {
    if (!MD->isVariadic()) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_sentinel_not_variadic) << 0;
      return;
    }
  } else if (isa<BlockDecl>(d)) {
    // Note! BlockDecl is typeless. Variadic diagnostics will be issued by the
    // caller.
    ;
  } else if (const VarDecl *V = dyn_cast<VarDecl>(d)) {
    QualType Ty = V->getType();
    if (Ty->isBlockPointerType() || Ty->isFunctionPointerType()) {
      const FunctionType *FT = Ty->isFunctionPointerType() ? getFunctionType(d)
        : Ty->getAs<BlockPointerType>()->getPointeeType()->getAs<FunctionType>();
      if (!cast<FunctionProtoType>(FT)->isVariadic()) {
        int m = Ty->isFunctionPointerType() ? 0 : 1;
        S.Diag(Attr.getLoc(), diag::warn_attribute_sentinel_not_variadic) << m;
        return;
      }
    } else {
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << 6 /*function, method or block */;
      return;
    }
  } else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << 6 /*function, method or block */;
    return;
  }
  d->addAttr(::new (S.Context) SentinelAttr(sentinel, nullPos));
}

static void HandleWarnUnusedResult(Decl *D, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  // TODO: could also be applied to methods?
  FunctionDecl *Fn = dyn_cast<FunctionDecl>(D);
  if (!Fn) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << 0 /*function*/;
    return;
  }

  Fn->addAttr(::new (S.Context) WarnUnusedResultAttr());
}

static void HandleWeakAttr(Decl *D, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  /* weak only applies to non-static declarations */
  bool isStatic = false;
  if (VarDecl *VD = dyn_cast<VarDecl>(D)) {
    isStatic = VD->getStorageClass() == VarDecl::Static;
  } else if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    isStatic = FD->getStorageClass() == FunctionDecl::Static;
  }
  if (isStatic) {
    S.Diag(Attr.getLoc(), diag::err_attribute_weak_static) <<
      dyn_cast<NamedDecl>(D)->getNameAsString();
    return;
  }

  // TODO: could also be applied to methods?
  if (!isa<FunctionDecl>(D) && !isa<VarDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << 2 /*variable and function*/;
    return;
  }

  D->addAttr(::new (S.Context) WeakAttr());
}

static void HandleWeakImportAttr(Decl *D, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  // weak_import only applies to variable & function declarations.
  bool isDef = false;
  if (VarDecl *VD = dyn_cast<VarDecl>(D)) {
    isDef = (!VD->hasExternalStorage() || VD->getInit());
  } else if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    isDef = FD->getBody();
  } else if (isa<ObjCPropertyDecl>(D) || isa<ObjCMethodDecl>(D)) {
    // We ignore weak import on properties and methods
    return;
  } else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
    << Attr.getName() << 2 /*variable and function*/;
    return;
  }

  // Merge should handle any subsequent violations.
  if (isDef) {
    S.Diag(Attr.getLoc(),
           diag::warn_attribute_weak_import_invalid_on_definition)
      << "weak_import" << 2 /*variable and function*/;
    return;
  }

  D->addAttr(::new (S.Context) WeakImportAttr());
}

static void HandleDLLImportAttr(Decl *D, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  // Attribute can be applied only to functions or variables.
  if (isa<VarDecl>(D)) {
    D->addAttr(::new (S.Context) DLLImportAttr());
    return;
  }

  FunctionDecl *FD = dyn_cast<FunctionDecl>(D);
  if (!FD) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << 2 /*variable and function*/;
    return;
  }

  // Currently, the dllimport attribute is ignored for inlined functions.
  // Warning is emitted.
  if (FD->isInline()) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << "dllimport";
    return;
  }

  // The attribute is also overridden by a subsequent declaration as dllexport.
  // Warning is emitted.
  for (AttributeList *nextAttr = Attr.getNext(); nextAttr;
       nextAttr = nextAttr->getNext()) {
    if (nextAttr->getKind() == AttributeList::AT_dllexport) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << "dllimport";
      return;
    }
  }

  if (D->getAttr<DLLExportAttr>()) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << "dllimport";
    return;
  }

  D->addAttr(::new (S.Context) DLLImportAttr());
}

static void HandleDLLExportAttr(Decl *D, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  // Attribute can be applied only to functions or variables.
  if (isa<VarDecl>(D)) {
    D->addAttr(::new (S.Context) DLLExportAttr());
    return;
  }

  FunctionDecl *FD = dyn_cast<FunctionDecl>(D);
  if (!FD) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << 2 /*variable and function*/;
    return;
  }

  // Currently, the dllexport attribute is ignored for inlined functions, unless
  // the -fkeep-inline-functions flag has been used. Warning is emitted;
  if (FD->isInline()) {
    // FIXME: ... unless the -fkeep-inline-functions flag has been used.
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << "dllexport";
    return;
  }

  D->addAttr(::new (S.Context) DLLExportAttr());
}

static void HandleReqdWorkGroupSize(Decl *D, const AttributeList &Attr,
                                    Sema &S) {
  // Attribute has 3 arguments.
  if (Attr.getNumArgs() != 3) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }

  unsigned WGSize[3];
  for (unsigned i = 0; i < 3; ++i) {
    Expr *E = static_cast<Expr *>(Attr.getArg(i));
    llvm::APSInt ArgNum(32);
    if (!E->isIntegerConstantExpr(ArgNum, S.Context)) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_not_int)
        << "reqd_work_group_size" << E->getSourceRange();
      return;
    }
    WGSize[i] = (unsigned) ArgNum.getZExtValue();
  }
  D->addAttr(::new (S.Context) ReqdWorkGroupSizeAttr(WGSize[0], WGSize[1],
                                                     WGSize[2]));
}

static void HandleSectionAttr(Decl *D, const AttributeList &Attr, Sema &S) {
  // Attribute has no arguments.
  if (Attr.getNumArgs() != 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }

  // Make sure that there is a string literal as the sections's single
  // argument.
  Expr *ArgExpr = static_cast<Expr *>(Attr.getArg(0));
  StringLiteral *SE = dyn_cast<StringLiteral>(ArgExpr);
  if (!SE) {
    S.Diag(ArgExpr->getLocStart(), diag::err_attribute_not_string) << "section";
    return;
  }

  std::string SectionStr(SE->getStrData(), SE->getByteLength());

  // If the target wants to validate the section specifier, make it happen.
  std::string Error = S.Context.Target.isValidSectionSpecifier(SectionStr);
  if (Error.empty()) {
    D->addAttr(::new (S.Context) SectionAttr(SectionStr));
    return;
  }

  S.Diag(SE->getLocStart(), diag::err_attribute_section_invalid_for_target)
    << Error;

}

static void HandleStdCallAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // Attribute has no arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  // Attribute can be applied only to functions.
  if (!isa<FunctionDecl>(d)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << 0 /*function*/;
    return;
  }

  // stdcall and fastcall attributes are mutually incompatible.
  if (d->getAttr<FastCallAttr>()) {
    S.Diag(Attr.getLoc(), diag::err_attributes_are_not_compatible)
      << "stdcall" << "fastcall";
    return;
  }

  d->addAttr(::new (S.Context) StdCallAttr());
}

static void HandleFastCallAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // Attribute has no arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  if (!isa<FunctionDecl>(d)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << 0 /*function*/;
    return;
  }

  // stdcall and fastcall attributes are mutually incompatible.
  if (d->getAttr<StdCallAttr>()) {
    S.Diag(Attr.getLoc(), diag::err_attributes_are_not_compatible)
      << "fastcall" << "stdcall";
    return;
  }

  d->addAttr(::new (S.Context) FastCallAttr());
}

static void HandleNothrowAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  d->addAttr(::new (S.Context) NoThrowAttr());
}

static void HandleConstAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  d->addAttr(::new (S.Context) ConstAttr());
}

static void HandlePureAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  d->addAttr(::new (S.Context) PureAttr());
}

static void HandleCleanupAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // Match gcc which ignores cleanup attrs when compiling C++.
  if (S.getLangOptions().CPlusPlus)
    return;

  if (!Attr.getParameterName()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }

  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }

  VarDecl *VD = dyn_cast<VarDecl>(d);

  if (!VD || !VD->hasLocalStorage()) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << "cleanup";
    return;
  }

  // Look up the function
  NamedDecl *CleanupDecl = S.LookupName(S.TUScope, Attr.getParameterName(),
                                        Sema::LookupOrdinaryName);
  if (!CleanupDecl) {
    S.Diag(Attr.getLoc(), diag::err_attribute_cleanup_arg_not_found) <<
      Attr.getParameterName();
    return;
  }

  FunctionDecl *FD = dyn_cast<FunctionDecl>(CleanupDecl);
  if (!FD) {
    S.Diag(Attr.getLoc(), diag::err_attribute_cleanup_arg_not_function) <<
      Attr.getParameterName();
    return;
  }

  if (FD->getNumParams() != 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_cleanup_func_must_take_one_arg) <<
      Attr.getParameterName();
    return;
  }

  // We're currently more strict than GCC about what function types we accept.
  // If this ever proves to be a problem it should be easy to fix.
  QualType Ty = S.Context.getPointerType(VD->getType());
  QualType ParamTy = FD->getParamDecl(0)->getType();
  if (S.CheckAssignmentConstraints(ParamTy, Ty) != Sema::Compatible) {
    S.Diag(Attr.getLoc(),
           diag::err_attribute_cleanup_func_arg_incompatible_type) <<
      Attr.getParameterName() << ParamTy << Ty;
    return;
  }

  d->addAttr(::new (S.Context) CleanupAttr(FD));
}

/// Handle __attribute__((format_arg((idx)))) attribute based on
/// http://gcc.gnu.org/onlinedocs/gcc/Function-Attributes.html
static void HandleFormatArgAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  if (Attr.getNumArgs() != 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }
  if (!isFunctionOrMethod(d) || !hasFunctionProto(d)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
    << Attr.getName() << 0 /*function*/;
    return;
  }
  // FIXME: in C++ the implicit 'this' function parameter also counts.  this is
  // needed in order to be compatible with GCC the index must start with 1.
  unsigned NumArgs  = getFunctionOrMethodNumArgs(d);
  unsigned FirstIdx = 1;
  // checks for the 2nd argument
  Expr *IdxExpr = static_cast<Expr *>(Attr.getArg(0));
  llvm::APSInt Idx(32);
  if (!IdxExpr->isIntegerConstantExpr(Idx, S.Context)) {
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

  // make sure the format string is really a string
  QualType Ty = getFunctionOrMethodArgType(d, ArgIdx);

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
  Ty = getFunctionOrMethodResultType(d);
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

  d->addAttr(::new (S.Context) FormatArgAttr(Idx.getZExtValue()));
}

/// Handle __attribute__((format(type,idx,firstarg))) attributes based on
/// http://gcc.gnu.org/onlinedocs/gcc/Function-Attributes.html
static void HandleFormatAttr(Decl *d, const AttributeList &Attr, Sema &S) {

  if (!Attr.getParameterName()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string)
      << "format" << 1;
    return;
  }

  if (Attr.getNumArgs() != 2) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 3;
    return;
  }

  if (!isFunctionOrMethodOrBlock(d) || !hasFunctionProto(d)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << 0 /*function*/;
    return;
  }

  unsigned NumArgs  = getFunctionOrMethodNumArgs(d);
  unsigned FirstIdx = 1;

  const char *Format = Attr.getParameterName()->getName();
  unsigned FormatLen = Attr.getParameterName()->getLength();

  // Normalize the argument, __foo__ becomes foo.
  if (FormatLen > 4 && Format[0] == '_' && Format[1] == '_' &&
      Format[FormatLen - 2] == '_' && Format[FormatLen - 1] == '_') {
    Format += 2;
    FormatLen -= 4;
  }

  bool Supported = false;
  bool is_NSString = false;
  bool is_strftime = false;
  bool is_CFString = false;

  switch (FormatLen) {
  default: break;
  case 5: Supported = !memcmp(Format, "scanf", 5); break;
  case 6: Supported = !memcmp(Format, "printf", 6); break;
  case 7: Supported = !memcmp(Format, "printf0", 7) ||
                      !memcmp(Format, "strfmon", 7); break;
  case 8:
    Supported = (is_strftime = !memcmp(Format, "strftime", 8)) ||
                (is_NSString = !memcmp(Format, "NSString", 8)) ||
                (is_CFString = !memcmp(Format, "CFString", 8));
    break;
  }

  if (!Supported) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_type_not_supported)
      << "format" << Attr.getParameterName()->getName();
    return;
  }

  // checks for the 2nd argument
  Expr *IdxExpr = static_cast<Expr *>(Attr.getArg(0));
  llvm::APSInt Idx(32);
  if (!IdxExpr->isIntegerConstantExpr(Idx, S.Context)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_int)
      << "format" << 2 << IdxExpr->getSourceRange();
    return;
  }

  // FIXME: We should handle the implicit 'this' parameter in a more generic
  // way that can be used for other arguments.
  bool HasImplicitThisParam = false;
  if (CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(d)) {
    if (MD->isInstance()) {
      HasImplicitThisParam = true;
      NumArgs++;
    }
  }

  if (Idx.getZExtValue() < FirstIdx || Idx.getZExtValue() > NumArgs) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_out_of_bounds)
      << "format" << 2 << IdxExpr->getSourceRange();
    return;
  }

  // FIXME: Do we need to bounds check?
  unsigned ArgIdx = Idx.getZExtValue() - 1;

  if (HasImplicitThisParam) ArgIdx--;

  // make sure the format string is really a string
  QualType Ty = getFunctionOrMethodArgType(d, ArgIdx);

  if (is_CFString) {
    if (!isCFStringType(Ty, S.Context)) {
      S.Diag(Attr.getLoc(), diag::err_format_attribute_not)
        << "a CFString" << IdxExpr->getSourceRange();
      return;
    }
  } else if (is_NSString) {
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
  Expr *FirstArgExpr = static_cast<Expr *>(Attr.getArg(1));
  llvm::APSInt FirstArg(32);
  if (!FirstArgExpr->isIntegerConstantExpr(FirstArg, S.Context)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_int)
      << "format" << 3 << FirstArgExpr->getSourceRange();
    return;
  }

  // check if the function is variadic if the 3rd argument non-zero
  if (FirstArg != 0) {
    if (isFunctionOrMethodVariadic(d)) {
      ++NumArgs; // +1 for ...
    } else {
      S.Diag(d->getLocation(), diag::err_format_attribute_requires_variadic);
      return;
    }
  }

  // strftime requires FirstArg to be 0 because it doesn't read from any
  // variable the input is just the current time + the format string.
  if (is_strftime) {
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

  d->addAttr(::new (S.Context) FormatAttr(std::string(Format, FormatLen),
                            Idx.getZExtValue(), FirstArg.getZExtValue()));
}

static void HandleTransparentUnionAttr(Decl *d, const AttributeList &Attr,
                                       Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  // Try to find the underlying union declaration.
  RecordDecl *RD = 0;
  TypedefDecl *TD = dyn_cast<TypedefDecl>(d);
  if (TD && TD->getUnderlyingType()->isUnionType())
    RD = TD->getUnderlyingType()->getAsUnionType()->getDecl();
  else
    RD = dyn_cast<RecordDecl>(d);

  if (!RD || !RD->isUnion()) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << 1 /*union*/;
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
  if (FirstType->isFloatingType() || FirstType->isVectorType()) {
    S.Diag(FirstField->getLocation(),
           diag::warn_transparent_union_attribute_floating);
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

  RD->addAttr(::new (S.Context) TransparentUnionAttr());
}

static void HandleAnnotateAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }
  Expr *ArgExpr = static_cast<Expr *>(Attr.getArg(0));
  StringLiteral *SE = dyn_cast<StringLiteral>(ArgExpr);

  // Make sure that there is a string literal as the annotation's single
  // argument.
  if (!SE) {
    S.Diag(ArgExpr->getLocStart(), diag::err_attribute_not_string) <<"annotate";
    return;
  }
  d->addAttr(::new (S.Context) AnnotateAttr(std::string(SE->getStrData(),
                                                        SE->getByteLength())));
}

static void HandleAlignedAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() > 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }

  unsigned Align = 0;
  if (Attr.getNumArgs() == 0) {
    // FIXME: This should be the target specific maximum alignment.
    // (For now we just use 128 bits which is the maximum on X86).
    Align = 128;
    d->addAttr(::new (S.Context) AlignedAttr(Align));
    return;
  }

  Expr *alignmentExpr = static_cast<Expr *>(Attr.getArg(0));
  llvm::APSInt Alignment(32);
  if (!alignmentExpr->isIntegerConstantExpr(Alignment, S.Context)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_not_int)
      << "aligned" << alignmentExpr->getSourceRange();
    return;
  }
  if (!llvm::isPowerOf2_64(Alignment.getZExtValue())) {
    S.Diag(Attr.getLoc(), diag::err_attribute_aligned_not_power_of_two)
      << alignmentExpr->getSourceRange();
    return;
  }

  d->addAttr(::new (S.Context) AlignedAttr(Alignment.getZExtValue() * 8));
}

/// HandleModeAttr - This attribute modifies the width of a decl with primitive
/// type.
///
/// Despite what would be logical, the mode attribute is a decl attribute, not a
/// type attribute: 'int ** __attribute((mode(HI))) *G;' tries to make 'G' be
/// HImode, not an intermediate pointer.
static void HandleModeAttr(Decl *D, const AttributeList &Attr, Sema &S) {
  // This attribute isn't documented, but glibc uses it.  It changes
  // the width of an int or unsigned int to the specified size.

  // Check that there aren't any arguments
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  IdentifierInfo *Name = Attr.getParameterName();
  if (!Name) {
    S.Diag(Attr.getLoc(), diag::err_attribute_missing_parameter_name);
    return;
  }
  const char *Str = Name->getName();
  unsigned Len = Name->getLength();

  // Normalize the attribute name, __foo__ becomes foo.
  if (Len > 4 && Str[0] == '_' && Str[1] == '_' &&
      Str[Len - 2] == '_' && Str[Len - 1] == '_') {
    Str += 2;
    Len -= 4;
  }

  unsigned DestWidth = 0;
  bool IntegerMode = true;
  bool ComplexMode = false;
  switch (Len) {
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
    if (!memcmp(Str, "word", 4))
      DestWidth = S.Context.Target.getPointerWidth(0);
    if (!memcmp(Str, "byte", 4))
      DestWidth = S.Context.Target.getCharWidth();
    break;
  case 7:
    if (!memcmp(Str, "pointer", 7))
      DestWidth = S.Context.Target.getPointerWidth(0);
    break;
  }

  QualType OldTy;
  if (TypedefDecl *TD = dyn_cast<TypedefDecl>(D))
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
    if (!OldTy->isIntegralType())
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
      NewTy = S.Context.LongLongTy;
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
    NewTy = S.Context.getFixedWidthIntType(128, OldTy->isSignedIntegerType());
    break;
  }

  if (ComplexMode) {
    NewTy = S.Context.getComplexType(NewTy);
  }

  // Install the new type.
  if (TypedefDecl *TD = dyn_cast<TypedefDecl>(D))
    TD->setUnderlyingType(NewTy);
  else
    cast<ValueDecl>(D)->setType(NewTy);
}

static void HandleNoDebugAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() > 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  if (!isFunctionOrMethod(d)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << 0 /*function*/;
    return;
  }

  d->addAttr(::new (S.Context) NoDebugAttr());
}

static void HandleNoInlineAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  if (!isa<FunctionDecl>(d)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
    << Attr.getName() << 0 /*function*/;
    return;
  }

  d->addAttr(::new (S.Context) NoInlineAttr());
}

static void HandleGNUInlineAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  FunctionDecl *Fn = dyn_cast<FunctionDecl>(d);
  if (Fn == 0) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << 0 /*function*/;
    return;
  }

  if (!Fn->isInline()) {
    S.Diag(Attr.getLoc(), diag::warn_gnu_inline_attribute_requires_inline);
    return;
  }

  d->addAttr(::new (S.Context) GNUInlineAttr());
}

static void HandleRegparmAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }

  if (!isFunctionOrMethod(d)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
    << Attr.getName() << 0 /*function*/;
    return;
  }

  Expr *NumParamsExpr = static_cast<Expr *>(Attr.getArg(0));
  llvm::APSInt NumParams(32);
  if (!NumParamsExpr->isIntegerConstantExpr(NumParams, S.Context)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_not_int)
      << "regparm" << NumParamsExpr->getSourceRange();
    return;
  }

  if (S.Context.Target.getRegParmMax() == 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_regparm_wrong_platform)
      << NumParamsExpr->getSourceRange();
    return;
  }

  if (NumParams.getLimitedValue(255) > S.Context.Target.getRegParmMax()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_regparm_invalid_number)
      << S.Context.Target.getRegParmMax() << NumParamsExpr->getSourceRange();
    return;
  }

  d->addAttr(::new (S.Context) RegparmAttr(NumParams.getZExtValue()));
}

//===----------------------------------------------------------------------===//
// Checker-specific attribute handlers.
//===----------------------------------------------------------------------===//

static void HandleNSReturnsRetainedAttr(Decl *d, const AttributeList &Attr,
                                        Sema &S) {

  QualType RetTy;

  if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(d))
    RetTy = MD->getResultType();
  else if (FunctionDecl *FD = dyn_cast<FunctionDecl>(d))
    RetTy = FD->getResultType();
  else {
    SourceLocation L = Attr.getLoc();
    S.Diag(d->getLocStart(), diag::warn_attribute_wrong_decl_type)
        << SourceRange(L, L) << Attr.getName() << 3 /* function or method */;
    return;
  }

  if (!(S.Context.isObjCNSObjectType(RetTy) || RetTy->getAs<PointerType>()
        || RetTy->getAs<ObjCObjectPointerType>())) {
    SourceLocation L = Attr.getLoc();
    S.Diag(d->getLocStart(), diag::warn_ns_attribute_wrong_return_type)
      << SourceRange(L, L) << Attr.getName();
    return;
  }

  switch (Attr.getKind()) {
    default:
      assert(0 && "invalid ownership attribute");
      return;
    case AttributeList::AT_cf_returns_retained:
      d->addAttr(::new (S.Context) CFReturnsRetainedAttr());
      return;
    case AttributeList::AT_ns_returns_retained:
      d->addAttr(::new (S.Context) NSReturnsRetainedAttr());
      return;
  };
}

//===----------------------------------------------------------------------===//
// Top Level Sema Entry Points
//===----------------------------------------------------------------------===//

/// ProcessDeclAttribute - Apply the specific attribute to the specified decl if
/// the attribute applies to decls.  If the attribute is a type attribute, just
/// silently ignore it.
static void ProcessDeclAttribute(Scope *scope, Decl *D,
                                 const AttributeList &Attr, Sema &S) {
  if (Attr.isDeclspecAttribute())
    // FIXME: Try to deal with __declspec attributes!
    return;
  switch (Attr.getKind()) {
  case AttributeList::AT_IBOutlet:    HandleIBOutletAttr  (D, Attr, S); break;
  case AttributeList::AT_address_space:
  case AttributeList::AT_objc_gc:
    // Ignore these, these are type attributes, handled by
    // ProcessTypeAttributes.
    break;
  case AttributeList::AT_alias:       HandleAliasAttr     (D, Attr, S); break;
  case AttributeList::AT_aligned:     HandleAlignedAttr   (D, Attr, S); break;
  case AttributeList::AT_always_inline:
    HandleAlwaysInlineAttr  (D, Attr, S); break;
  case AttributeList::AT_analyzer_noreturn:
    HandleAnalyzerNoReturnAttr  (D, Attr, S); break;
  case AttributeList::AT_annotate:    HandleAnnotateAttr  (D, Attr, S); break;
  case AttributeList::AT_constructor: HandleConstructorAttr(D, Attr, S); break;
  case AttributeList::AT_deprecated:  HandleDeprecatedAttr(D, Attr, S); break;
  case AttributeList::AT_destructor:  HandleDestructorAttr(D, Attr, S); break;
  case AttributeList::AT_dllexport:   HandleDLLExportAttr (D, Attr, S); break;
  case AttributeList::AT_dllimport:   HandleDLLImportAttr (D, Attr, S); break;
  case AttributeList::AT_ext_vector_type:
    HandleExtVectorTypeAttr(scope, D, Attr, S);
    break;
  case AttributeList::AT_fastcall:    HandleFastCallAttr  (D, Attr, S); break;
  case AttributeList::AT_format:      HandleFormatAttr    (D, Attr, S); break;
  case AttributeList::AT_format_arg:  HandleFormatArgAttr (D, Attr, S); break;
  case AttributeList::AT_gnu_inline:  HandleGNUInlineAttr(D, Attr, S); break;
  case AttributeList::AT_mode:        HandleModeAttr      (D, Attr, S); break;
  case AttributeList::AT_malloc:      HandleMallocAttr    (D, Attr, S); break;
  case AttributeList::AT_nonnull:     HandleNonNullAttr   (D, Attr, S); break;
  case AttributeList::AT_noreturn:    HandleNoReturnAttr  (D, Attr, S); break;
  case AttributeList::AT_nothrow:     HandleNothrowAttr   (D, Attr, S); break;

  // Checker-specific.
  case AttributeList::AT_ns_returns_retained:
  case AttributeList::AT_cf_returns_retained:
    HandleNSReturnsRetainedAttr(D, Attr, S); break;

  case AttributeList::AT_reqd_wg_size:
    HandleReqdWorkGroupSize(D, Attr, S); break;

  case AttributeList::AT_packed:      HandlePackedAttr    (D, Attr, S); break;
  case AttributeList::AT_section:     HandleSectionAttr   (D, Attr, S); break;
  case AttributeList::AT_stdcall:     HandleStdCallAttr   (D, Attr, S); break;
  case AttributeList::AT_unavailable: HandleUnavailableAttr(D, Attr, S); break;
  case AttributeList::AT_unused:      HandleUnusedAttr    (D, Attr, S); break;
  case AttributeList::AT_used:        HandleUsedAttr      (D, Attr, S); break;
  case AttributeList::AT_vector_size: HandleVectorSizeAttr(D, Attr, S); break;
  case AttributeList::AT_visibility:  HandleVisibilityAttr(D, Attr, S); break;
  case AttributeList::AT_warn_unused_result: HandleWarnUnusedResult(D,Attr,S);
    break;
  case AttributeList::AT_weak:        HandleWeakAttr      (D, Attr, S); break;
  case AttributeList::AT_weak_import: HandleWeakImportAttr(D, Attr, S); break;
  case AttributeList::AT_transparent_union:
    HandleTransparentUnionAttr(D, Attr, S);
    break;
  case AttributeList::AT_objc_exception:
    HandleObjCExceptionAttr(D, Attr, S);
    break;
  case AttributeList::AT_overloadable:HandleOverloadableAttr(D, Attr, S); break;
  case AttributeList::AT_nsobject:    HandleObjCNSObject  (D, Attr, S); break;
  case AttributeList::AT_blocks:      HandleBlocksAttr    (D, Attr, S); break;
  case AttributeList::AT_sentinel:    HandleSentinelAttr  (D, Attr, S); break;
  case AttributeList::AT_const:       HandleConstAttr     (D, Attr, S); break;
  case AttributeList::AT_pure:        HandlePureAttr      (D, Attr, S); break;
  case AttributeList::AT_cleanup:     HandleCleanupAttr   (D, Attr, S); break;
  case AttributeList::AT_nodebug:     HandleNoDebugAttr   (D, Attr, S); break;
  case AttributeList::AT_noinline:    HandleNoInlineAttr  (D, Attr, S); break;
  case AttributeList::AT_regparm:     HandleRegparmAttr   (D, Attr, S); break;
  case AttributeList::IgnoredAttribute:
  case AttributeList::AT_no_instrument_function:  // Interacts with -pg.
    // Just ignore
    break;
  default:
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << Attr.getName();
    break;
  }
}

/// ProcessDeclAttributeList - Apply all the decl attributes in the specified
/// attribute list to the specified decl, ignoring any type attributes.
void Sema::ProcessDeclAttributeList(Scope *S, Decl *D, const AttributeList *AttrList) {
  while (AttrList) {
    ProcessDeclAttribute(S, D, *AttrList, *this);
    AttrList = AttrList->getNext();
  }
}

/// DeclClonePragmaWeak - clone existing decl (maybe definition),
/// #pragma weak needs a non-definition decl and source may not have one
NamedDecl * Sema::DeclClonePragmaWeak(NamedDecl *ND, IdentifierInfo *II) {
  assert(isa<FunctionDecl>(ND) || isa<VarDecl>(ND));
  NamedDecl *NewD = 0;
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(ND)) {
    NewD = FunctionDecl::Create(FD->getASTContext(), FD->getDeclContext(),
                                FD->getLocation(), DeclarationName(II),
                                FD->getType(), FD->getDeclaratorInfo());
  } else if (VarDecl *VD = dyn_cast<VarDecl>(ND)) {
    NewD = VarDecl::Create(VD->getASTContext(), VD->getDeclContext(),
                           VD->getLocation(), II,
                           VD->getType(), VD->getDeclaratorInfo(),
                           VD->getStorageClass());
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
    NewD->addAttr(::new (Context) AliasAttr(NDId->getName()));
    NewD->addAttr(::new (Context) WeakAttr());
    WeakTopLevelDecl.push_back(NewD);
    // FIXME: "hideous" code from Sema::LazilyCreateBuiltin
    // to insert Decl at TU scope, sorry.
    DeclContext *SavedContext = CurContext;
    CurContext = Context.getTranslationUnitDecl();
    PushOnScopeChains(NewD, S);
    CurContext = SavedContext;
  } else { // just add weak to existing
    ND->addAttr(::new (Context) WeakAttr());
  }
}

/// ProcessDeclAttributes - Given a declarator (PD) with attributes indicated in
/// it, apply them to D.  This is a bit tricky because PD can have attributes
/// specified in many different places, and we need to find and apply them all.
void Sema::ProcessDeclAttributes(Scope *S, Decl *D, const Declarator &PD) {
  // Handle #pragma weak
  if (NamedDecl *ND = dyn_cast<NamedDecl>(D)) {
    if (ND->hasLinkage()) {
      WeakInfo W = WeakUndeclaredIdentifiers.lookup(ND->getIdentifier());
      if (W != WeakInfo()) {
        // Identifier referenced by #pragma weak before it was declared
        DeclApplyPragmaWeak(S, ND, W);
        WeakUndeclaredIdentifiers[ND->getIdentifier()] = W;
      }
    }
  }

  // Apply decl attributes from the DeclSpec if present.
  if (const AttributeList *Attrs = PD.getDeclSpec().getAttributes())
    ProcessDeclAttributeList(S, D, Attrs);

  // Walk the declarator structure, applying decl attributes that were in a type
  // position to the decl itself.  This handles cases like:
  //   int *__attr__(x)** D;
  // when X is a decl attribute.
  for (unsigned i = 0, e = PD.getNumTypeObjects(); i != e; ++i)
    if (const AttributeList *Attrs = PD.getTypeObject(i).getAttrs())
      ProcessDeclAttributeList(S, D, Attrs);

  // Finally, apply any attributes on the decl itself.
  if (const AttributeList *Attrs = PD.getAttributes())
    ProcessDeclAttributeList(S, D, Attrs);
}
