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
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/DelayedDiagnostic.h"
#include "llvm/ADT/StringExtras.h"
using namespace clang;
using namespace sema;

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

/// isFunction - Return true if the given decl has function
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

/// Return true if the given decl has a declarator that should have
/// been processed by Sema::GetTypeForDeclarator.
static bool hasDeclarator(const Decl *d) {
  // In some sense, TypedefDecl really *ought* to be a DeclaratorDecl.
  return isa<DeclaratorDecl>(d) || isa<BlockDecl>(d) || isa<TypedefDecl>(d);
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
    return BD->isVariadic();
  else {
    return cast<ObjCMethodDecl>(d)->isVariadic();
  }
}

static bool isInstanceMethod(const Decl *d) {
  if (const CXXMethodDecl *MethodDecl = dyn_cast<CXXMethodDecl>(d))
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
    CXXScopeSpec SS;
    UnqualifiedId id;
    id.setIdentifier(Attr.getParameterName(), Attr.getLoc());
    sizeExpr = S.ActOnIdExpression(scope, SS, id, false, false).takeAs<Expr>();
  } else {
    // check the attribute arguments.
    if (Attr.getNumArgs() != 1) {
      S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
      return;
    }
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

static void HandlePackedAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() > 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  if (TagDecl *TD = dyn_cast<TagDecl>(d))
    TD->addAttr(::new (S.Context) PackedAttr(Attr.getLoc(), S.Context));
  else if (FieldDecl *FD = dyn_cast<FieldDecl>(d)) {
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

static void HandleIBAction(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() > 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  // The IBAction attributes only apply to instance methods.
  if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(d))
    if (MD->isInstanceMethod()) {
      d->addAttr(::new (S.Context) IBActionAttr(Attr.getLoc(), S.Context));
      return;
    }

  S.Diag(Attr.getLoc(), diag::err_attribute_ibaction) << Attr.getName();
}

static void HandleIBOutlet(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() > 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  // The IBOutlet attributes only apply to instance variables of
  // Objective-C classes.
  if (isa<ObjCIvarDecl>(d) || isa<ObjCPropertyDecl>(d)) {
    d->addAttr(::new (S.Context) IBOutletAttr(Attr.getLoc(), S.Context));
    return;
  }

  S.Diag(Attr.getLoc(), diag::err_attribute_iboutlet) << Attr.getName();
}

static void HandleIBOutletCollection(Decl *d, const AttributeList &Attr,
                                     Sema &S) {

  // The iboutletcollection attribute can have zero or one arguments.
  if (Attr.getParameterName() && Attr.getNumArgs() > 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }

  // The IBOutletCollection attributes only apply to instance variables of
  // Objective-C classes.
  if (!(isa<ObjCIvarDecl>(d) || isa<ObjCPropertyDecl>(d))) {
    S.Diag(Attr.getLoc(), diag::err_attribute_iboutlet) << Attr.getName();
    return;
  }
  if (const ValueDecl *VD = dyn_cast<ValueDecl>(d))
    if (!VD->getType()->getAs<ObjCObjectPointerType>()) {
      S.Diag(Attr.getLoc(), diag::err_iboutletcollection_object_type) 
        << VD->getType() << 0;
      return;
    }
  if (const ObjCPropertyDecl *PD = dyn_cast<ObjCPropertyDecl>(d))
    if (!PD->getType()->getAs<ObjCObjectPointerType>()) {
      S.Diag(Attr.getLoc(), diag::err_iboutletcollection_object_type) 
        << PD->getType() << 1;
      return;
    }
  
  IdentifierInfo *II = Attr.getParameterName();
  if (!II)
    II = &S.Context.Idents.get("id");
  
  ParsedType TypeRep = S.getTypeName(*II, Attr.getLoc(), 
                        S.getScopeForContext(d->getDeclContext()->getParent()));
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
  d->addAttr(::new (S.Context) IBOutletCollectionAttr(Attr.getLoc(), S.Context,
                                                      QT));
}

static void HandleNonNullAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // GCC ignores the nonnull attribute on K&R style function prototypes, so we
  // ignore it as well
  if (!isFunctionOrMethod(d) || !hasFunctionProto(d)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << 0 /*function*/;
    return;
  }

  // In C++ the implicit 'this' function parameter also counts, and they are
  // counted from one.
  bool HasImplicitThisParam = isInstanceMethod(d);
  unsigned NumArgs  = getFunctionOrMethodNumArgs(d) + HasImplicitThisParam;

  // The nonnull attribute only applies to pointers.
  llvm::SmallVector<unsigned, 10> NonNullArgs;

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
    QualType T = getFunctionOrMethodArgType(d, x).getNonReferenceType();
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
    for (unsigned I = 0, E = getFunctionOrMethodNumArgs(d); I != E; ++I) {
      QualType T = getFunctionOrMethodArgType(d, I).getNonReferenceType();
      if (T->isAnyPointerType() || T->isBlockPointerType())
        NonNullArgs.push_back(I);
      else if (const RecordType *UT = T->getAsUnionType()) {
        if (UT && UT->getDecl()->hasAttr<TransparentUnionAttr>()) {
          RecordDecl *UD = UT->getDecl();
          for (RecordDecl::field_iterator it = UD->field_begin(),
               itend = UD->field_end(); it != itend; ++it) {
            T = it->getType();
            if (T->isAnyPointerType() || T->isBlockPointerType()) {
              NonNullArgs.push_back(I);
              break;
            }
          }
        }
      }
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
  d->addAttr(::new (S.Context) NonNullAttr(Attr.getLoc(), S.Context, start,
                                           size));
}

static void HandleOwnershipAttr(Decl *d, const AttributeList &AL, Sema &S) {
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

  if (!isFunction(d) || !hasFunctionProto(d)) {
    S.Diag(AL.getLoc(), diag::warn_attribute_wrong_decl_type) << AL.getName()
        << 0 /*function*/;
    return;
  }

  // In C++ the implicit 'this' function parameter also counts, and they are
  // counted from one.
  bool HasImplicitThisParam = isInstanceMethod(d);
  unsigned NumArgs  = getFunctionOrMethodNumArgs(d) + HasImplicitThisParam;

  llvm::StringRef Module = AL.getParameterName()->getName();

  // Normalize the argument, __foo__ becomes foo.
  if (Module.startswith("__") && Module.endswith("__"))
    Module = Module.substr(2, Module.size() - 4);

  llvm::SmallVector<unsigned, 10> OwnershipArgs;

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
      QualType T = getFunctionOrMethodArgType(d, x);
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
          i = d->specific_attr_begin<OwnershipAttr>(),
          e = d->specific_attr_end<OwnershipAttr>();
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

  d->addAttr(::new (S.Context) OwnershipAttr(AL.getLoc(), S.Context, K, Module,
                                             start, size));
}

static bool isStaticVarOrStaticFunction(Decl *D) {
  if (VarDecl *VD = dyn_cast<VarDecl>(D))
    return VD->getStorageClass() == SC_Static;
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
    return FD->getStorageClass() == SC_Static;
  return false;
}

static void HandleWeakRefAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // Check the attribute arguments.
  if (Attr.getNumArgs() > 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }

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
  const DeclContext *Ctx = d->getDeclContext()->getRedeclContext();
  if (!Ctx->isFileContext()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_weakref_not_global_context) <<
        dyn_cast<NamedDecl>(d)->getNameAsString();
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

  if (!isStaticVarOrStaticFunction(d)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_weakref_not_static) <<
      dyn_cast<NamedDecl>(d)->getNameAsString();
    return;
  }

  // GCC rejects
  // static ((alias ("y"), weakref)).
  // Should we? How to check that weakref is before or after alias?

  if (Attr.getNumArgs() == 1) {
    Expr *Arg = Attr.getArg(0);
    Arg = Arg->IgnoreParenCasts();
    StringLiteral *Str = dyn_cast<StringLiteral>(Arg);

    if (Str == 0 || Str->isWide()) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string)
          << "weakref" << 1;
      return;
    }
    // GCC will accept anything as the argument of weakref. Should we
    // check for an existing decl?
    d->addAttr(::new (S.Context) AliasAttr(Attr.getLoc(), S.Context,
                                           Str->getString()));
  }

  d->addAttr(::new (S.Context) WeakRefAttr(Attr.getLoc(), S.Context));
}

static void HandleAliasAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }

  Expr *Arg = Attr.getArg(0);
  Arg = Arg->IgnoreParenCasts();
  StringLiteral *Str = dyn_cast<StringLiteral>(Arg);

  if (Str == 0 || Str->isWide()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string)
      << "alias" << 1;
    return;
  }

  if (S.Context.Target.getTriple().getOS() == llvm::Triple::Darwin) {
    S.Diag(Attr.getLoc(), diag::err_alias_not_supported_on_darwin);
    return;
  }

  // FIXME: check if target symbol exists in current file

  d->addAttr(::new (S.Context) AliasAttr(Attr.getLoc(), S.Context,
                                         Str->getString()));
}

static void HandleNakedAttr(Decl *d, const AttributeList &Attr,
                                   Sema &S) {
  // Check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  if (!isa<FunctionDecl>(d)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << 0 /*function*/;
    return;
  }

  d->addAttr(::new (S.Context) NakedAttr(Attr.getLoc(), S.Context));
}

static void HandleAlwaysInlineAttr(Decl *d, const AttributeList &Attr,
                                   Sema &S) {
  // Check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  if (!isa<FunctionDecl>(d)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << 0 /*function*/;
    return;
  }

  d->addAttr(::new (S.Context) AlwaysInlineAttr(Attr.getLoc(), S.Context));
}

static void HandleMallocAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // Check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(d)) {
    QualType RetTy = FD->getResultType();
    if (RetTy->isAnyPointerType() || RetTy->isBlockPointerType()) {
      d->addAttr(::new (S.Context) MallocAttr(Attr.getLoc(), S.Context));
      return;
    }
  }

  S.Diag(Attr.getLoc(), diag::warn_attribute_malloc_pointer_only);
}

static void HandleMayAliasAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  d->addAttr(::new (S.Context) MayAliasAttr(Attr.getLoc(), S.Context));
}

static void HandleNoCommonAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  assert(Attr.isInvalid() == false);
  if (isa<VarDecl>(d))
    d->addAttr(::new (S.Context) NoCommonAttr(Attr.getLoc(), S.Context));
  else
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << 12 /* variable */;
}

static void HandleCommonAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  assert(Attr.isInvalid() == false);
  if (isa<VarDecl>(d))
    d->addAttr(::new (S.Context) CommonAttr(Attr.getLoc(), S.Context));
  else
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << 12 /* variable */;
}

static void HandleNoReturnAttr(Decl *d, const AttributeList &attr, Sema &S) {
  if (hasDeclarator(d)) return;

  if (S.CheckNoReturnAttr(attr)) return;

  if (!isa<ObjCMethodDecl>(d)) {
    S.Diag(attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << attr.getName() << 0 /*function*/;
    return;
  }

  d->addAttr(::new (S.Context) NoReturnAttr(attr.getLoc(), S.Context));
}

bool Sema::CheckNoReturnAttr(const AttributeList &attr) {
  if (attr.getNumArgs() != 0) {
    Diag(attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    attr.setInvalid();
    return true;
  }

  return false;
}

static void HandleAnalyzerNoReturnAttr(Decl *d, const AttributeList &Attr,
                                       Sema &S) {
  
  // The checking path for 'noreturn' and 'analyzer_noreturn' are different
  // because 'analyzer_noreturn' does not impact the type.
  
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }
  
  if (!isFunctionOrMethod(d) && !isa<BlockDecl>(d)) {
    ValueDecl *VD = dyn_cast<ValueDecl>(d);
    if (VD == 0 || (!VD->getType()->isBlockPointerType()
                    && !VD->getType()->isFunctionPointerType())) {
      S.Diag(Attr.getLoc(),
             Attr.isCXX0XAttribute() ? diag::err_attribute_wrong_decl_type
             : diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << 0 /*function*/;
      return;
    }
  }
  
  d->addAttr(::new (S.Context) AnalyzerNoReturnAttr(Attr.getLoc(), S.Context));
}

// PS3 PPU-specific.
static void HandleVecReturnAttr(Decl *d, const AttributeList &Attr,
                                       Sema &S) {
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
  if (!isa<RecordDecl>(d)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << 9 /*class*/;
    return;
  }

  if (d->getAttr<VecReturnAttr>()) {
    S.Diag(Attr.getLoc(), diag::err_repeat_attribute) << "vecreturn";
    return;
  }

  RecordDecl *record = cast<RecordDecl>(d);
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

  d->addAttr(::new (S.Context) VecReturnAttr(Attr.getLoc(), S.Context));
}

static void HandleForbidTemporariesAttr(Decl *d, const AttributeList &Attr,
                                        Sema &S) {
  assert(Attr.isInvalid() == false);

  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  if (!isa<TypeDecl>(d)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << 9 /*class*/;
    return;
  }

  d->addAttr(::new (S.Context) ForbidTemporariesAttr(Attr.getLoc(), S.Context));
}

static void HandleDependencyAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  if (!isFunctionOrMethod(d) && !isa<ParmVarDecl>(d)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_decl_type)
      << Attr.getName() << 8 /*function, method, or parameter*/;
    return;
  }
  // FIXME: Actually store the attribute on the declaration
}

static void HandleUnusedAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  if (!isa<VarDecl>(d) && !isa<ObjCIvarDecl>(d) && !isFunctionOrMethod(d) &&
      !isa<TypeDecl>(d)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << 2 /*variable and function*/;
    return;
  }

  d->addAttr(::new (S.Context) UnusedAttr(Attr.getLoc(), S.Context));
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

  d->addAttr(::new (S.Context) UsedAttr(Attr.getLoc(), S.Context));
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

  if (!isa<FunctionDecl>(d)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << 0 /*function*/;
    return;
  }

  d->addAttr(::new (S.Context) ConstructorAttr(Attr.getLoc(), S.Context,
                                               priority));
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

  if (!isa<FunctionDecl>(d)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << 0 /*function*/;
    return;
  }

  d->addAttr(::new (S.Context) DestructorAttr(Attr.getLoc(), S.Context,
                                              priority));
}

static void HandleDeprecatedAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  int noArgs = Attr.getNumArgs();
  if (noArgs > 1) {
    S.Diag(Attr.getLoc(), 
           diag::err_attribute_wrong_number_arguments) << "0 or 1";
    return;
  }
  // Handle the case where deprecated attribute has a text message.
  StringLiteral *SE;
  if (noArgs == 1) {
    Expr *ArgExpr = Attr.getArg(0);
    SE = dyn_cast<StringLiteral>(ArgExpr);
    if (!SE) {
      S.Diag(ArgExpr->getLocStart(), 
             diag::err_attribute_not_string) << "deprecated";
      return;
    }
  }
  else
    SE = StringLiteral::CreateEmpty(S.Context, 1);

  d->addAttr(::new (S.Context) DeprecatedAttr(Attr.getLoc(), S.Context,
                                              SE->getString()));
}

static void HandleUnavailableAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  int noArgs = Attr.getNumArgs();
  if (noArgs > 1) {
    S.Diag(Attr.getLoc(),
           diag::err_attribute_wrong_number_arguments) << "0 or 1";
    return;
  }
  // Handle the case where unavailable attribute has a text message.
  StringLiteral *SE;
  if (noArgs == 1) {
    Expr *ArgExpr = Attr.getArg(0);
    SE = dyn_cast<StringLiteral>(ArgExpr);
    if (!SE) {
      S.Diag(ArgExpr->getLocStart(), 
             diag::err_attribute_not_string) << "unavailable";
      return;
    }
  }
  else
    SE = StringLiteral::CreateEmpty(S.Context, 1);
  d->addAttr(::new (S.Context) UnavailableAttr(Attr.getLoc(), S.Context,
                                               SE->getString()));
}

static void HandleVisibilityAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }

  Expr *Arg = Attr.getArg(0);
  Arg = Arg->IgnoreParenCasts();
  StringLiteral *Str = dyn_cast<StringLiteral>(Arg);

  if (Str == 0 || Str->isWide()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string)
      << "visibility" << 1;
    return;
  }

  llvm::StringRef TypeStr = Str->getString();
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

  d->addAttr(::new (S.Context) VisibilityAttr(Attr.getLoc(), S.Context, type));
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

  D->addAttr(::new (S.Context) ObjCExceptionAttr(Attr.getLoc(), S.Context));
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
  D->addAttr(::new (S.Context) ObjCNSObjectAttr(Attr.getLoc(), S.Context));
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

  D->addAttr(::new (S.Context) OverloadableAttr(Attr.getLoc(), S.Context));
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

  BlocksAttr::BlockType type;
  if (Attr.getParameterName()->isStr("byref"))
    type = BlocksAttr::ByRef;
  else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_type_not_supported)
      << "blocks" << Attr.getParameterName();
    return;
  }

  d->addAttr(::new (S.Context) BlocksAttr(Attr.getLoc(), S.Context, type));
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
  d->addAttr(::new (S.Context) SentinelAttr(Attr.getLoc(), S.Context, sentinel,
                                            nullPos));
}

static void HandleWarnUnusedResult(Decl *D, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  if (!isFunction(D) && !isa<ObjCMethodDecl>(D)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << 0 /*function*/;
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

static void HandleWeakAttr(Decl *D, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  /* weak only applies to non-static declarations */
  if (isStaticVarOrStaticFunction(D)) {
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

  D->addAttr(::new (S.Context) WeakAttr(Attr.getLoc(), S.Context));
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
    isDef = FD->hasBody();
  } else if (isa<ObjCPropertyDecl>(D) || isa<ObjCMethodDecl>(D)) {
    // We ignore weak import on properties and methods
    return;
  } else if (!(S.LangOpts.ObjCNonFragileABI && isa<ObjCInterfaceDecl>(D))) {
    // Don't issue the warning for darwin as target; yet, ignore the attribute.
    if (S.Context.Target.getTriple().getOS() != llvm::Triple::Darwin ||
        !isa<ObjCInterfaceDecl>(D)) 
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

  D->addAttr(::new (S.Context) WeakImportAttr(Attr.getLoc(), S.Context));
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

static void HandleSectionAttr(Decl *D, const AttributeList &Attr, Sema &S) {
  // Attribute has no arguments.
  if (Attr.getNumArgs() != 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }

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


static void HandleNothrowAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  d->addAttr(::new (S.Context) NoThrowAttr(Attr.getLoc(), S.Context));
}

static void HandleConstAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  d->addAttr(::new (S.Context) ConstAttr(Attr.getLoc(), S.Context));
}

static void HandlePureAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    return;
  }

  d->addAttr(::new (S.Context) PureAttr(Attr.getLoc(), S.Context));
}

static void HandleCleanupAttr(Decl *d, const AttributeList &Attr, Sema &S) {
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
  if (S.CheckAssignmentConstraints(ParamTy, Ty) != Sema::Compatible) {
    S.Diag(Attr.getParameterLoc(),
           diag::err_attribute_cleanup_func_arg_incompatible_type) <<
      Attr.getParameterName() << ParamTy << Ty;
    return;
  }

  d->addAttr(::new (S.Context) CleanupAttr(Attr.getLoc(), S.Context, FD));
  S.MarkDeclarationReferenced(Attr.getParameterLoc(), FD);
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

  // In C++ the implicit 'this' function parameter also counts, and they are
  // counted from one.
  bool HasImplicitThisParam = isInstanceMethod(d);
  unsigned NumArgs  = getFunctionOrMethodNumArgs(d) + HasImplicitThisParam;
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

  d->addAttr(::new (S.Context) FormatArgAttr(Attr.getLoc(), S.Context,
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
static FormatAttrKind getFormatAttrKind(llvm::StringRef Format) {
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
      Format == "zcmn_err")
    return SupportedFormat;

  if (Format == "gcc_diag" || Format == "gcc_cdiag" ||
      Format == "gcc_cxxdiag" || Format == "gcc_tdiag")
    return IgnoredFormat;
  
  return InvalidFormat;
}

/// Handle __attribute__((init_priority(priority))) attributes based on
/// http://gcc.gnu.org/onlinedocs/gcc/C_002b_002b-Attributes.html
static void HandleInitPriorityAttr(Decl *d, const AttributeList &Attr, 
                                   Sema &S) {
  if (!S.getLangOptions().CPlusPlus) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << Attr.getName();
    return;
  }
  
  if (!isa<VarDecl>(d) || S.getCurFunctionOrMethodDecl()) {
    S.Diag(Attr.getLoc(), diag::err_init_priority_object_attr);
    Attr.setInvalid();
    return;
  }
  QualType T = dyn_cast<VarDecl>(d)->getType();
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
  d->addAttr(::new (S.Context) InitPriorityAttr(Attr.getLoc(), S.Context,
                                                prioritynum));
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

  // In C++ the implicit 'this' function parameter also counts, and they are
  // counted from one.
  bool HasImplicitThisParam = isInstanceMethod(d);
  unsigned NumArgs  = getFunctionOrMethodNumArgs(d) + HasImplicitThisParam;
  unsigned FirstIdx = 1;

  llvm::StringRef Format = Attr.getParameterName()->getName();

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
  QualType Ty = getFunctionOrMethodArgType(d, ArgIdx);

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
    if (isFunctionOrMethodVariadic(d)) {
      ++NumArgs; // +1 for ...
    } else {
      S.Diag(d->getLocation(), diag::err_format_attribute_requires_variadic);
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

  d->addAttr(::new (S.Context) FormatAttr(Attr.getLoc(), S.Context, Format,
                                          Idx.getZExtValue(),
                                          FirstArg.getZExtValue()));
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

static void HandleAnnotateAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }
  Expr *ArgExpr = Attr.getArg(0);
  StringLiteral *SE = dyn_cast<StringLiteral>(ArgExpr);

  // Make sure that there is a string literal as the annotation's single
  // argument.
  if (!SE) {
    S.Diag(ArgExpr->getLocStart(), diag::err_attribute_not_string) <<"annotate";
    return;
  }
  d->addAttr(::new (S.Context) AnnotateAttr(Attr.getLoc(), S.Context,
                                            SE->getString()));
}

static void HandleAlignedAttr(Decl *D, const AttributeList &Attr, Sema &S) {
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

  llvm::StringRef Str = Attr.getParameterName()->getName();

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
  if (TypedefDecl *TD = dyn_cast<TypedefDecl>(D)) {
    // FIXME: preserve existing source info.
    TD->setTypeSourceInfo(S.Context.getTrivialTypeSourceInfo(NewTy));
  } else
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

  d->addAttr(::new (S.Context) NoDebugAttr(Attr.getLoc(), S.Context));
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

  d->addAttr(::new (S.Context) NoInlineAttr(Attr.getLoc(), S.Context));
}

static void HandleNoInstrumentFunctionAttr(Decl *d, const AttributeList &Attr,
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

  d->addAttr(::new (S.Context) NoInstrumentFunctionAttr(Attr.getLoc(),
                                                        S.Context));
}

static void HandleConstantAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  if (S.LangOpts.CUDA) {
    // check the attribute arguments.
    if (Attr.getNumArgs() != 0) {
      S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
      return;
    }

    if (!isa<VarDecl>(d)) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
        << Attr.getName() << 12 /*variable*/;
      return;
    }

    d->addAttr(::new (S.Context) CUDAConstantAttr(Attr.getLoc(), S.Context));
  } else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << "constant";
  }
}

static void HandleDeviceAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  if (S.LangOpts.CUDA) {
    // check the attribute arguments.
    if (Attr.getNumArgs() != 0) {
      S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
      return;
    }

    if (!isa<FunctionDecl>(d) && !isa<VarDecl>(d)) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
        << Attr.getName() << 2 /*variable and function*/;
      return;
    }

    d->addAttr(::new (S.Context) CUDADeviceAttr(Attr.getLoc(), S.Context));
  } else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << "device";
  }
}

static void HandleGlobalAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  if (S.LangOpts.CUDA) {
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

    FunctionDecl *FD = cast<FunctionDecl>(d);
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

    d->addAttr(::new (S.Context) CUDAGlobalAttr(Attr.getLoc(), S.Context));
  } else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << "global";
  }
}

static void HandleHostAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  if (S.LangOpts.CUDA) {
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

    d->addAttr(::new (S.Context) CUDAHostAttr(Attr.getLoc(), S.Context));
  } else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << "host";
  }
}

static void HandleSharedAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  if (S.LangOpts.CUDA) {
    // check the attribute arguments.
    if (Attr.getNumArgs() != 0) {
      S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
      return;
    }

    if (!isa<VarDecl>(d)) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
        << Attr.getName() << 12 /*variable*/;
      return;
    }

    d->addAttr(::new (S.Context) CUDASharedAttr(Attr.getLoc(), S.Context));
  } else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << "shared";
  }
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

  if (!Fn->isInlineSpecified()) {
    S.Diag(Attr.getLoc(), diag::warn_gnu_inline_attribute_requires_inline);
    return;
  }

  d->addAttr(::new (S.Context) GNUInlineAttr(Attr.getLoc(), S.Context));
}

static void HandleCallConvAttr(Decl *d, const AttributeList &attr, Sema &S) {
  if (hasDeclarator(d)) return;

  // Diagnostic is emitted elsewhere: here we store the (valid) attr
  // in the Decl node for syntactic reasoning, e.g., pretty-printing.
  CallingConv CC;
  if (S.CheckCallingConvAttr(attr, CC))
    return;

  if (!isa<ObjCMethodDecl>(d)) {
    S.Diag(attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << attr.getName() << 0 /*function*/;
    return;
  }

  switch (attr.getKind()) {
  case AttributeList::AT_fastcall:
    d->addAttr(::new (S.Context) FastCallAttr(attr.getLoc(), S.Context));
    return;
  case AttributeList::AT_stdcall:
    d->addAttr(::new (S.Context) StdCallAttr(attr.getLoc(), S.Context));
    return;
  case AttributeList::AT_thiscall:
    d->addAttr(::new (S.Context) ThisCallAttr(attr.getLoc(), S.Context));
    return;
  case AttributeList::AT_cdecl:
    d->addAttr(::new (S.Context) CDeclAttr(attr.getLoc(), S.Context));
    return;
  case AttributeList::AT_pascal:
    d->addAttr(::new (S.Context) PascalAttr(attr.getLoc(), S.Context));
    return;
  default:
    llvm_unreachable("unexpected attribute kind");
    return;
  }
}

bool Sema::CheckCallingConvAttr(const AttributeList &attr, CallingConv &CC) {
  if (attr.isInvalid())
    return true;

  if (attr.getNumArgs() != 0) {
    Diag(attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    attr.setInvalid();
    return true;
  }

  // TODO: diagnose uses of these conventions on the wrong target.
  switch (attr.getKind()) {
  case AttributeList::AT_cdecl: CC = CC_C; break;
  case AttributeList::AT_fastcall: CC = CC_X86FastCall; break;
  case AttributeList::AT_stdcall: CC = CC_X86StdCall; break;
  case AttributeList::AT_thiscall: CC = CC_X86ThisCall; break;
  case AttributeList::AT_pascal: CC = CC_X86Pascal; break;
  default: llvm_unreachable("unexpected attribute kind"); return true;
  }

  return false;
}

static void HandleRegparmAttr(Decl *d, const AttributeList &attr, Sema &S) {
  if (hasDeclarator(d)) return;

  unsigned numParams;
  if (S.CheckRegparmAttr(attr, numParams))
    return;

  if (!isa<ObjCMethodDecl>(d)) {
    S.Diag(attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << attr.getName() << 0 /*function*/;
    return;
  }

  d->addAttr(::new (S.Context) RegparmAttr(attr.getLoc(), S.Context, numParams));
}

/// Checks a regparm attribute, returning true if it is ill-formed and
/// otherwise setting numParams to the appropriate value.
bool Sema::CheckRegparmAttr(const AttributeList &attr, unsigned &numParams) {
  if (attr.isInvalid())
    return true;

  if (attr.getNumArgs() != 1) {
    Diag(attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    attr.setInvalid();
    return true;
  }

  Expr *NumParamsExpr = attr.getArg(0);
  llvm::APSInt NumParams(32);
  if (NumParamsExpr->isTypeDependent() || NumParamsExpr->isValueDependent() ||
      !NumParamsExpr->isIntegerConstantExpr(NumParams, Context)) {
    Diag(attr.getLoc(), diag::err_attribute_argument_not_int)
      << "regparm" << NumParamsExpr->getSourceRange();
    attr.setInvalid();
    return true;
  }

  if (Context.Target.getRegParmMax() == 0) {
    Diag(attr.getLoc(), diag::err_attribute_regparm_wrong_platform)
      << NumParamsExpr->getSourceRange();
    attr.setInvalid();
    return true;
  }

  numParams = NumParams.getZExtValue();
  if (numParams > Context.Target.getRegParmMax()) {
    Diag(attr.getLoc(), diag::err_attribute_regparm_invalid_number)
      << Context.Target.getRegParmMax() << NumParamsExpr->getSourceRange();
    attr.setInvalid();
    return true;
  }

  return false;
}

static void HandleLaunchBoundsAttr(Decl *d, const AttributeList &Attr, Sema &S){
  if (S.LangOpts.CUDA) {
    // check the attribute arguments.
    if (Attr.getNumArgs() != 1 && Attr.getNumArgs() != 2) {
      S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments)
        << "1 or 2";
      return;
    }

    if (!isFunctionOrMethod(d)) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type)
      << Attr.getName() << 0 /*function*/;
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

    d->addAttr(::new (S.Context) CUDALaunchBoundsAttr(Attr.getLoc(), S.Context,
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

static void HandleNSConsumedAttr(Decl *d, const AttributeList &attr, Sema &S) {
  ParmVarDecl *param = dyn_cast<ParmVarDecl>(d);
  if (!param) {
    S.Diag(d->getLocStart(), diag::warn_attribute_wrong_decl_type)
      << SourceRange(attr.getLoc()) << attr.getName() << 4 /*parameter*/;
    return;
  }

  bool typeOK, cf;
  if (attr.getKind() == AttributeList::AT_ns_consumed) {
    typeOK = isValidSubjectOfNSAttribute(S, param->getType());
    cf = false;
  } else {
    typeOK = isValidSubjectOfCFAttribute(S, param->getType());
    cf = true;
  }

  if (!typeOK) {
    S.Diag(d->getLocStart(), diag::warn_ns_attribute_wrong_parameter_type)
      << SourceRange(attr.getLoc()) << attr.getName() << cf;
    return;
  }

  if (cf)
    param->addAttr(::new (S.Context) CFConsumedAttr(attr.getLoc(), S.Context));
  else
    param->addAttr(::new (S.Context) NSConsumedAttr(attr.getLoc(), S.Context));  
}

static void HandleNSConsumesSelfAttr(Decl *d, const AttributeList &attr,
                                     Sema &S) {
  if (!isa<ObjCMethodDecl>(d)) {
    S.Diag(d->getLocStart(), diag::warn_attribute_wrong_decl_type)
      << SourceRange(attr.getLoc()) << attr.getName() << 13 /*method*/;
    return;
  }

  d->addAttr(::new (S.Context) NSConsumesSelfAttr(attr.getLoc(), S.Context));
}

static void HandleNSReturnsRetainedAttr(Decl *d, const AttributeList &attr,
                                        Sema &S) {

  QualType returnType;

  if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(d))
    returnType = MD->getResultType();
  else if (FunctionDecl *FD = dyn_cast<FunctionDecl>(d))
    returnType = FD->getResultType();
  else {
    S.Diag(d->getLocStart(), diag::warn_attribute_wrong_decl_type)
        << SourceRange(attr.getLoc()) << attr.getName()
        << 3 /* function or method */;
    return;
  }

  bool typeOK;
  bool cf;
  switch (attr.getKind()) {
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
    S.Diag(d->getLocStart(), diag::warn_ns_attribute_wrong_return_type)
      << SourceRange(attr.getLoc())
      << attr.getName() << isa<ObjCMethodDecl>(d) << cf;
    return;
  }

  switch (attr.getKind()) {
    default:
      assert(0 && "invalid ownership attribute");
      return;
    case AttributeList::AT_ns_returns_autoreleased:
      d->addAttr(::new (S.Context) NSReturnsAutoreleasedAttr(attr.getLoc(),
                                                             S.Context));
      return;
    case AttributeList::AT_cf_returns_not_retained:
      d->addAttr(::new (S.Context) CFReturnsNotRetainedAttr(attr.getLoc(),
                                                            S.Context));
      return;
    case AttributeList::AT_ns_returns_not_retained:
      d->addAttr(::new (S.Context) NSReturnsNotRetainedAttr(attr.getLoc(),
                                                            S.Context));
      return;
    case AttributeList::AT_cf_returns_retained:
      d->addAttr(::new (S.Context) CFReturnsRetainedAttr(attr.getLoc(),
                                                         S.Context));
      return;
    case AttributeList::AT_ns_returns_retained:
      d->addAttr(::new (S.Context) NSReturnsRetainedAttr(attr.getLoc(),
                                                         S.Context));
      return;
  };
}

static bool isKnownDeclSpecAttr(const AttributeList &Attr) {
  return Attr.getKind() == AttributeList::AT_dllimport ||
         Attr.getKind() == AttributeList::AT_dllexport ||
         Attr.getKind() == AttributeList::AT_uuid;
}

//===----------------------------------------------------------------------===//
// Microsoft specific attribute handlers.
//===----------------------------------------------------------------------===//

static void HandleUuidAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  if (S.LangOpts.Microsoft || S.LangOpts.Borland) {
    // check the attribute arguments.
    if (Attr.getNumArgs() != 1) {
      S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
      return;
    }
    Expr *Arg = Attr.getArg(0);
    StringLiteral *Str = dyn_cast<StringLiteral>(Arg);
    if (Str == 0 || Str->isWide()) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string)
        << "uuid" << 1;
      return;
    }

    llvm::StringRef StrRef = Str->getString();

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
    llvm::StringRef::iterator I = StrRef.begin();
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

    d->addAttr(::new (S.Context) UuidAttr(Attr.getLoc(), S.Context, 
                                          Str->getString()));
  } else
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) << "uuid";
}

//===----------------------------------------------------------------------===//
// Top Level Sema Entry Points
//===----------------------------------------------------------------------===//

static void ProcessNonInheritableDeclAttr(Scope *scope, Decl *D,
                                          const AttributeList &Attr, Sema &S) {
  switch (Attr.getKind()) {
  case AttributeList::AT_device:      HandleDeviceAttr      (D, Attr, S); break;
  case AttributeList::AT_host:        HandleHostAttr        (D, Attr, S); break;
  case AttributeList::AT_overloadable:HandleOverloadableAttr(D, Attr, S); break;
  default:
    break;
  }
}

static void ProcessInheritableDeclAttr(Scope *scope, Decl *D,
                                       const AttributeList &Attr, Sema &S) {
  switch (Attr.getKind()) {
  case AttributeList::AT_IBAction:            HandleIBAction(D, Attr, S); break;
    case AttributeList::AT_IBOutlet:          HandleIBOutlet(D, Attr, S); break;
  case AttributeList::AT_IBOutletCollection:
      HandleIBOutletCollection(D, Attr, S); break;
  case AttributeList::AT_address_space:
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
  case AttributeList::AT_alias:       HandleAliasAttr       (D, Attr, S); break;
  case AttributeList::AT_aligned:     HandleAlignedAttr     (D, Attr, S); break;
  case AttributeList::AT_always_inline:
    HandleAlwaysInlineAttr  (D, Attr, S); break;
  case AttributeList::AT_analyzer_noreturn:
    HandleAnalyzerNoReturnAttr  (D, Attr, S); break;
  case AttributeList::AT_annotate:    HandleAnnotateAttr    (D, Attr, S); break;
  case AttributeList::AT_carries_dependency:
                                      HandleDependencyAttr  (D, Attr, S); break;
  case AttributeList::AT_common:      HandleCommonAttr      (D, Attr, S); break;
  case AttributeList::AT_constant:    HandleConstantAttr    (D, Attr, S); break;
  case AttributeList::AT_constructor: HandleConstructorAttr (D, Attr, S); break;
  case AttributeList::AT_deprecated:  HandleDeprecatedAttr  (D, Attr, S); break;
  case AttributeList::AT_destructor:  HandleDestructorAttr  (D, Attr, S); break;
  case AttributeList::AT_ext_vector_type:
    HandleExtVectorTypeAttr(scope, D, Attr, S);
    break;
  case AttributeList::AT_forbid_temporaries:
    HandleForbidTemporariesAttr(D, Attr, S); break;
  case AttributeList::AT_format:      HandleFormatAttr      (D, Attr, S); break;
  case AttributeList::AT_format_arg:  HandleFormatArgAttr   (D, Attr, S); break;
  case AttributeList::AT_global:      HandleGlobalAttr      (D, Attr, S); break;
  case AttributeList::AT_gnu_inline:  HandleGNUInlineAttr   (D, Attr, S); break;
  case AttributeList::AT_launch_bounds:
    HandleLaunchBoundsAttr(D, Attr, S);
    break;
  case AttributeList::AT_mode:        HandleModeAttr        (D, Attr, S); break;
  case AttributeList::AT_malloc:      HandleMallocAttr      (D, Attr, S); break;
  case AttributeList::AT_may_alias:   HandleMayAliasAttr    (D, Attr, S); break;
  case AttributeList::AT_nocommon:    HandleNoCommonAttr    (D, Attr, S); break;
  case AttributeList::AT_nonnull:     HandleNonNullAttr     (D, Attr, S); break;
  case AttributeList::AT_ownership_returns:
  case AttributeList::AT_ownership_takes:
  case AttributeList::AT_ownership_holds:
      HandleOwnershipAttr     (D, Attr, S); break;
  case AttributeList::AT_naked:       HandleNakedAttr       (D, Attr, S); break;
  case AttributeList::AT_noreturn:    HandleNoReturnAttr    (D, Attr, S); break;
  case AttributeList::AT_nothrow:     HandleNothrowAttr     (D, Attr, S); break;
  case AttributeList::AT_shared:      HandleSharedAttr      (D, Attr, S); break;
  case AttributeList::AT_vecreturn:   HandleVecReturnAttr   (D, Attr, S); break;

  // Checker-specific.
  case AttributeList::AT_cf_consumed:
  case AttributeList::AT_ns_consumed: HandleNSConsumedAttr  (D, Attr, S); break;
  case AttributeList::AT_ns_consumes_self:
    HandleNSConsumesSelfAttr(D, Attr, S); break;

  case AttributeList::AT_ns_returns_autoreleased:
  case AttributeList::AT_ns_returns_not_retained:
  case AttributeList::AT_cf_returns_not_retained:
  case AttributeList::AT_ns_returns_retained:
  case AttributeList::AT_cf_returns_retained:
    HandleNSReturnsRetainedAttr(D, Attr, S); break;

  case AttributeList::AT_reqd_wg_size:
    HandleReqdWorkGroupSize(D, Attr, S); break;

  case AttributeList::AT_init_priority: 
      HandleInitPriorityAttr(D, Attr, S); break;
      
  case AttributeList::AT_packed:      HandlePackedAttr      (D, Attr, S); break;
  case AttributeList::AT_section:     HandleSectionAttr     (D, Attr, S); break;
  case AttributeList::AT_unavailable: HandleUnavailableAttr (D, Attr, S); break;
  case AttributeList::AT_unused:      HandleUnusedAttr      (D, Attr, S); break;
  case AttributeList::AT_used:        HandleUsedAttr        (D, Attr, S); break;
  case AttributeList::AT_visibility:  HandleVisibilityAttr  (D, Attr, S); break;
  case AttributeList::AT_warn_unused_result: HandleWarnUnusedResult(D,Attr,S);
    break;
  case AttributeList::AT_weak:        HandleWeakAttr        (D, Attr, S); break;
  case AttributeList::AT_weakref:     HandleWeakRefAttr     (D, Attr, S); break;
  case AttributeList::AT_weak_import: HandleWeakImportAttr  (D, Attr, S); break;
  case AttributeList::AT_transparent_union:
    HandleTransparentUnionAttr(D, Attr, S);
    break;
  case AttributeList::AT_objc_exception:
    HandleObjCExceptionAttr(D, Attr, S);
    break;
  case AttributeList::AT_nsobject:    HandleObjCNSObject    (D, Attr, S); break;
  case AttributeList::AT_blocks:      HandleBlocksAttr      (D, Attr, S); break;
  case AttributeList::AT_sentinel:    HandleSentinelAttr    (D, Attr, S); break;
  case AttributeList::AT_const:       HandleConstAttr       (D, Attr, S); break;
  case AttributeList::AT_pure:        HandlePureAttr        (D, Attr, S); break;
  case AttributeList::AT_cleanup:     HandleCleanupAttr     (D, Attr, S); break;
  case AttributeList::AT_nodebug:     HandleNoDebugAttr     (D, Attr, S); break;
  case AttributeList::AT_noinline:    HandleNoInlineAttr    (D, Attr, S); break;
  case AttributeList::AT_regparm:     HandleRegparmAttr     (D, Attr, S); break;
  case AttributeList::IgnoredAttribute:
    // Just ignore
    break;
  case AttributeList::AT_no_instrument_function:  // Interacts with -pg.
    HandleNoInstrumentFunctionAttr(D, Attr, S);
    break;
  case AttributeList::AT_stdcall:
  case AttributeList::AT_cdecl:
  case AttributeList::AT_fastcall:
  case AttributeList::AT_thiscall:
  case AttributeList::AT_pascal:
    HandleCallConvAttr(D, Attr, S);
    break;
  case AttributeList::AT_uuid:
    HandleUuidAttr(D, Attr, S);
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
static void ProcessDeclAttribute(Scope *scope, Decl *D,
                                 const AttributeList &Attr, Sema &S,
                                 bool NonInheritable, bool Inheritable) {
  if (Attr.isInvalid())
    return;

  if (Attr.isDeclspecAttribute() && !isKnownDeclSpecAttr(Attr))
    // FIXME: Try to deal with other __declspec attributes!
    return;

  if (NonInheritable)
    ProcessNonInheritableDeclAttr(scope, D, Attr, S);

  if (Inheritable)
    ProcessInheritableDeclAttr(scope, D, Attr, S);
}

/// ProcessDeclAttributeList - Apply all the decl attributes in the specified
/// attribute list to the specified decl, ignoring any type attributes.
void Sema::ProcessDeclAttributeList(Scope *S, Decl *D,
                                    const AttributeList *AttrList,
                                    bool NonInheritable, bool Inheritable) {
  for (const AttributeList* l = AttrList; l; l = l->getNext()) {
    ProcessDeclAttribute(S, D, *l, *this, NonInheritable, Inheritable);
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
                                FD->getLocation(), DeclarationName(II),
                                FD->getType(), FD->getTypeSourceInfo());
    if (FD->getQualifier()) {
      FunctionDecl *NewFD = cast<FunctionDecl>(NewD);
      NewFD->setQualifierInfo(FD->getQualifier(), FD->getQualifierRange());
    }
  } else if (VarDecl *VD = dyn_cast<VarDecl>(ND)) {
    NewD = VarDecl::Create(VD->getASTContext(), VD->getDeclContext(),
                           VD->getLocation(), II,
                           VD->getType(), VD->getTypeSourceInfo(),
                           VD->getStorageClass(),
                           VD->getStorageClassAsWritten());
    if (VD->getQualifier()) {
      VarDecl *NewVD = cast<VarDecl>(NewD);
      NewVD->setQualifierInfo(VD->getQualifier(), VD->getQualifierRange());
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
  if (Inheritable && !WeakUndeclaredIdentifiers.empty()) {
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

/// PushParsingDeclaration - Enter a new "scope" of deprecation
/// warnings.
///
/// The state token we use is the start index of this scope
/// on the warning stack.
Sema::ParsingDeclStackState Sema::PushParsingDeclaration() {
  ParsingDeclDepth++;
  return (ParsingDeclStackState) DelayedDiagnostics.size();
}

void Sema::PopParsingDeclaration(ParsingDeclStackState S, Decl *D) {
  assert(ParsingDeclDepth > 0 && "empty ParsingDeclaration stack");
  ParsingDeclDepth--;

  if (DelayedDiagnostics.empty())
    return;

  unsigned SavedIndex = (unsigned) S;
  assert(SavedIndex <= DelayedDiagnostics.size() &&
         "saved index is out of bounds");

  unsigned E = DelayedDiagnostics.size();

  // We only want to actually emit delayed diagnostics when we
  // successfully parsed a decl.
  if (D) {
    // We really do want to start with 0 here.  We get one push for a
    // decl spec and another for each declarator;  in a decl group like:
    //   deprecated_typedef foo, *bar, baz();
    // only the declarator pops will be passed decls.  This is correct;
    // we really do need to consider delayed diagnostics from the decl spec
    // for each of the different declarations.
    for (unsigned I = 0; I != E; ++I) {
      if (DelayedDiagnostics[I].Triggered)
        continue;

      switch (DelayedDiagnostics[I].Kind) {
      case DelayedDiagnostic::Deprecation:
        HandleDelayedDeprecationCheck(DelayedDiagnostics[I], D);
        break;

      case DelayedDiagnostic::Access:
        HandleDelayedAccessCheck(DelayedDiagnostics[I], D);
        break;
      }
    }
  }

  // Destroy all the delayed diagnostics we're about to pop off.
  for (unsigned I = SavedIndex; I != E; ++I)
    DelayedDiagnostics[I].destroy();

  DelayedDiagnostics.set_size(SavedIndex);
}

static bool isDeclDeprecated(Decl *D) {
  do {
    if (D->hasAttr<DeprecatedAttr>())
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

void Sema::EmitDeprecationWarning(NamedDecl *D, llvm::StringRef Message,
                                  SourceLocation Loc,
                                  bool UnknownObjCClass) {
  // Delay if we're currently parsing a declaration.
  if (ParsingDeclDepth) {
    DelayedDiagnostics.push_back(DelayedDiagnostic::makeDeprecation(Loc, D, 
                                                                    Message));
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
    else
      Diag(Loc, diag::warn_deprecated_fwdclass_message) << D->getDeclName();
  }
}
