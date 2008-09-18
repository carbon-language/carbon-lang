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
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Parse/DeclSpec.h"
#include <llvm/ADT/StringExtras.h>
using namespace clang;

//===----------------------------------------------------------------------===//
//  Helper functions
//===----------------------------------------------------------------------===//

static const FunctionTypeProto *getFunctionProto(Decl *d) {
  QualType Ty;
  if (ValueDecl *decl = dyn_cast<ValueDecl>(d))
    Ty = decl->getType();
  else if (FieldDecl *decl = dyn_cast<FieldDecl>(d))
    Ty = decl->getType();
  else if (TypedefDecl* decl = dyn_cast<TypedefDecl>(d))
    Ty = decl->getUnderlyingType();
  else
    return 0;
  
  if (Ty->isFunctionPointerType())
    Ty = Ty->getAsPointerType()->getPointeeType();
  
  if (const FunctionType *FnTy = Ty->getAsFunctionType())
    return dyn_cast<FunctionTypeProto>(FnTy->getAsFunctionType());
  
  return 0;
}

static inline bool isNSStringType(QualType T, ASTContext &Ctx) {
  const PointerType *PT = T->getAsPointerType();
  if (!PT)
    return false;
  
  const ObjCInterfaceType *ClsT =PT->getPointeeType()->getAsObjCInterfaceType();
  if (!ClsT)
    return false;
  
  IdentifierInfo* ClsName = ClsT->getDecl()->getIdentifier();
  
  // FIXME: Should we walk the chain of classes?
  return ClsName == &Ctx.Idents.get("NSString") ||
         ClsName == &Ctx.Idents.get("NSMutableString");
}

//===----------------------------------------------------------------------===//
// Attribute Implementations
//===----------------------------------------------------------------------===//

// FIXME: All this manual attribute parsing code is gross. At the
// least add some helper functions to check most argument patterns (#
// and types of args).

static void HandleExtVectorTypeAttr(Decl *d, const AttributeList &Attr,
                                    Sema &S) {
  TypedefDecl *tDecl = dyn_cast<TypedefDecl>(d);
  if (tDecl == 0) {
    S.Diag(Attr.getLoc(), diag::err_typecheck_ext_vector_not_typedef);
    return;
  }
  
  QualType curType = tDecl->getUnderlyingType();
  // check the attribute arguments.
  if (Attr.getNumArgs() != 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
           std::string("1"));
    return;
  }
  Expr *sizeExpr = static_cast<Expr *>(Attr.getArg(0));
  llvm::APSInt vecSize(32);
  if (!sizeExpr->isIntegerConstantExpr(vecSize, S.Context)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_not_int,
           "ext_vector_type", sizeExpr->getSourceRange());
    return;
  }
  // unlike gcc's vector_size attribute, we do not allow vectors to be defined
  // in conjunction with complex types (pointers, arrays, functions, etc.).
  if (!curType->isIntegerType() && !curType->isRealFloatingType()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_invalid_vector_type,
           curType.getAsString());
    return;
  }
  // unlike gcc's vector_size attribute, the size is specified as the 
  // number of elements, not the number of bytes.
  unsigned vectorSize = static_cast<unsigned>(vecSize.getZExtValue()); 
  
  if (vectorSize == 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_zero_size,
           sizeExpr->getSourceRange());
    return;
  }
  // Instantiate/Install the vector type, the number of elements is > 0.
  tDecl->setUnderlyingType(S.Context.getExtVectorType(curType, vectorSize));
  // Remember this typedef decl, we will need it later for diagnostics.
  S.ExtVectorDecls.push_back(tDecl);
}


/// HandleVectorSizeAttribute - this attribute is only applicable to 
/// integral and float scalars, although arrays, pointers, and function
/// return values are allowed in conjunction with this construct. Aggregates
/// with this attribute are invalid, even if they are of the same size as a
/// corresponding scalar.
/// The raw attribute should contain precisely 1 argument, the vector size 
/// for the variable, measured in bytes. If curType and rawAttr are well
/// formed, this routine will return a new vector type.
static void HandleVectorSizeAttr(Decl *D, const AttributeList &Attr, Sema &S) {
  QualType CurType;
  if (ValueDecl *VD = dyn_cast<ValueDecl>(D))
    CurType = VD->getType();
  else if (TypedefDecl *TD = dyn_cast<TypedefDecl>(D))
    CurType = TD->getUnderlyingType();
  else {
    S.Diag(D->getLocation(), diag::err_attr_wrong_decl,
           std::string("vector_size"),
           SourceRange(Attr.getLoc(), Attr.getLoc()));
    return;
  }
    
  // Check the attribute arugments.
  if (Attr.getNumArgs() != 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
           std::string("1"));
    return;
  }
  Expr *sizeExpr = static_cast<Expr *>(Attr.getArg(0));
  llvm::APSInt vecSize(32);
  if (!sizeExpr->isIntegerConstantExpr(vecSize, S.Context)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_not_int,
           "vector_size", sizeExpr->getSourceRange());
    return;
  }
  // navigate to the base type - we need to provide for vector pointers, 
  // vector arrays, and functions returning vectors.
  if (CurType->isPointerType() || CurType->isArrayType() ||
      CurType->isFunctionType()) {
    assert(0 && "HandleVector(): Complex type construction unimplemented");
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
  // the base type must be integer or float.
  if (!CurType->isIntegerType() && !CurType->isRealFloatingType()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_invalid_vector_type,
           CurType.getAsString());
    return;
  }
  unsigned typeSize = static_cast<unsigned>(S.Context.getTypeSize(CurType));
  // vecSize is specified in bytes - convert to bits.
  unsigned vectorSize = static_cast<unsigned>(vecSize.getZExtValue() * 8); 
  
  // the vector size needs to be an integral multiple of the type size.
  if (vectorSize % typeSize) {
    S.Diag(Attr.getLoc(), diag::err_attribute_invalid_size,
           sizeExpr->getSourceRange());
    return;
  }
  if (vectorSize == 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_zero_size,
           sizeExpr->getSourceRange());
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
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
           std::string("0"));
    return;
  }
  
  if (TagDecl *TD = dyn_cast<TagDecl>(d))
    TD->addAttr(new PackedAttr());
  else if (FieldDecl *FD = dyn_cast<FieldDecl>(d)) {
    // If the alignment is less than or equal to 8 bits, the packed attribute
    // has no effect.
    if (!FD->getType()->isIncompleteType() &&
        S.Context.getTypeAlign(FD->getType()) <= 8)
      S.Diag(Attr.getLoc(), 
             diag::warn_attribute_ignored_for_field_of_type,
             Attr.getName()->getName(), FD->getType().getAsString());
    else
      FD->addAttr(new PackedAttr());
  } else
    S.Diag(Attr.getLoc(), diag::warn_attribute_ignored,
           Attr.getName()->getName());
}

static void HandleIBOutletAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() > 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
           std::string("0"));
    return;
  }
  
  // The IBOutlet attribute only applies to instance variables of Objective-C
  // classes.
  if (ObjCIvarDecl *ID = dyn_cast<ObjCIvarDecl>(d))
    ID->addAttr(new IBOutletAttr());
  else
    S.Diag(Attr.getLoc(), diag::err_attribute_iboutlet_non_ivar);
}

static void HandleNonNullAttr(Decl *d, const AttributeList &Attr, Sema &S) {

  // GCC ignores the nonnull attribute on K&R style function
  // prototypes, so we ignore it as well
  const FunctionTypeProto *proto = getFunctionProto(d);
  if (!proto) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type,
           "nonnull", "function");
    return;
  }
  
  unsigned NumArgs = proto->getNumArgs();

  // The nonnull attribute only applies to pointers.
  llvm::SmallVector<unsigned, 10> NonNullArgs;
  
  for (AttributeList::arg_iterator I=Attr.arg_begin(),
                                   E=Attr.arg_end(); I!=E; ++I) {
    
    
    // The argument must be an integer constant expression.
    Expr *Ex = static_cast<Expr *>(Attr.getArg(0));
    llvm::APSInt ArgNum(32);
    if (!Ex->isIntegerConstantExpr(ArgNum, S.Context)) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_not_int,
             "nonnull", Ex->getSourceRange());
      return;
    }
    
    unsigned x = (unsigned) ArgNum.getZExtValue();
        
    if (x < 1 || x > NumArgs) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_out_of_bounds,
             "nonnull", llvm::utostr_32(I.getArgNum()), Ex->getSourceRange());
      return;
    }
    
    --x;

    // Is the function argument a pointer type?
    if (!proto->getArgType(x)->isPointerType()) {
      // FIXME: Should also highlight argument in decl.
      S.Diag(Attr.getLoc(), diag::err_nonnull_pointers_only,
             "nonnull", Ex->getSourceRange());
      continue;
    }
    
    NonNullArgs.push_back(x);
  }
  
  // If no arguments were specified to __attribute__((nonnull)) then all
  // pointer arguments have a nonnull attribute.
  if (NonNullArgs.empty()) {
    unsigned idx = 0;
    
    for (FunctionTypeProto::arg_type_iterator
         I=proto->arg_type_begin(), E=proto->arg_type_end(); I!=E; ++I, ++idx)
      if ((*I)->isPointerType())
        NonNullArgs.push_back(idx);
    
    if (NonNullArgs.empty()) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_nonnull_no_pointers);
      return;
    }
  }

  unsigned* start = &NonNullArgs[0];
  unsigned size = NonNullArgs.size();
  std::sort(start, start + size);
  d->addAttr(new NonNullAttr(start, size));
}

static void HandleAliasAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
           std::string("1"));
    return;
  }
  
  Expr *Arg = static_cast<Expr*>(Attr.getArg(0));
  Arg = Arg->IgnoreParenCasts();
  StringLiteral *Str = dyn_cast<StringLiteral>(Arg);
  
  if (Str == 0 || Str->isWide()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string,
           "alias", std::string("1"));
    return;
  }
  
  const char *Alias = Str->getStrData();
  unsigned AliasLen = Str->getByteLength();
  
  // FIXME: check if target symbol exists in current file
  
  d->addAttr(new AliasAttr(std::string(Alias, AliasLen)));
}

static void HandleNoReturnAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
           std::string("0"));
    return;
  }
  
  FunctionDecl *Fn = dyn_cast<FunctionDecl>(d);
  if (!Fn) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type,
           "noreturn", "function");
    return;
  }
  
  d->addAttr(new NoReturnAttr());
}

static void HandleUnusedAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
           std::string("0"));
    return;
  }
  
  if (!isa<VarDecl>(d) && !getFunctionProto(d)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type,
           "unused", "variable and function");
    return;
  }
  
  d->addAttr(new UnusedAttr());
}

static void HandleConstructorAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0 && Attr.getNumArgs() != 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments, "0 or 1");
    return;
  } 

  int priority = 65535; // FIXME: Do not hardcode such constants.
  if (Attr.getNumArgs() > 0) {
    Expr *E = static_cast<Expr *>(Attr.getArg(0));
    llvm::APSInt Idx(32);
    if (!E->isIntegerConstantExpr(Idx, S.Context)) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_int,
             "constructor", "1", E->getSourceRange());
      return;
    }
    priority = Idx.getZExtValue();
  }
  
  FunctionDecl *Fn = dyn_cast<FunctionDecl>(d);
  if (!Fn) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type,
           "constructor", "function");
    return;
  }

  d->addAttr(new ConstructorAttr(priority));
}

static void HandleDestructorAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0 && Attr.getNumArgs() != 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments, "0 or 1");
    return;
  } 

  int priority = 65535; // FIXME: Do not hardcode such constants.
  if (Attr.getNumArgs() > 0) {
    Expr *E = static_cast<Expr *>(Attr.getArg(0));
    llvm::APSInt Idx(32);
    if (!E->isIntegerConstantExpr(Idx, S.Context)) {
      S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_int,
             "destructor", "1", E->getSourceRange());
      return;
    }
    priority = Idx.getZExtValue();
  }
  
  if (!isa<FunctionDecl>(d)) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type,
           "destructor", "function");
    return;
  }

  d->addAttr(new DestructorAttr(priority));
}

static void HandleDeprecatedAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
           std::string("0"));
    return;
  }
  
  d->addAttr(new DeprecatedAttr());
}

static void HandleVisibilityAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
           std::string("1"));
    return;
  }
  
  Expr *Arg = static_cast<Expr*>(Attr.getArg(0));
  Arg = Arg->IgnoreParenCasts();
  StringLiteral *Str = dyn_cast<StringLiteral>(Arg);
  
  if (Str == 0 || Str->isWide()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string,
           "visibility", std::string("1"));
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
    S.Diag(Attr.getLoc(), diag::warn_attribute_type_not_supported,
           "visibility", TypeStr);
    return;
  }
  
  d->addAttr(new VisibilityAttr(type));
}

static void HandleObjCGCAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  if (!Attr.getParameterName()) {    
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string,
           "objc_gc", std::string("1"));
    return;
  }
  
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
           std::string("1"));
    return;
  }
  
  const char *TypeStr = Attr.getParameterName()->getName();
  unsigned TypeLen = Attr.getParameterName()->getLength();
  
  ObjCGCAttr::GCAttrTypes type;
  
  if (TypeLen == 4 && !memcmp(TypeStr, "weak", 4))
    type = ObjCGCAttr::Weak;
  else if (TypeLen == 6 && !memcmp(TypeStr, "strong", 6))
    type = ObjCGCAttr::Strong;
  else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_type_not_supported,
           "objc_gc", TypeStr);
    return;
  }
  
  d->addAttr(new ObjCGCAttr(type));
}

static void HandleBlocksAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  if (!Attr.getParameterName()) {    
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string,
           "blocks", std::string("1"));
    return;
  }
  
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
           std::string("1"));
    return;
  }
  const char *TypeStr = Attr.getParameterName()->getName();
  unsigned TypeLen = Attr.getParameterName()->getLength();
  
  BlocksAttr::BlocksAttrTypes type;
  
  if (TypeLen == 5 && !memcmp(TypeStr, "byref", 5))
    type = BlocksAttr::ByRef;
  else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_type_not_supported,
           "blocks", TypeStr);
    return;
  }
  
  d->addAttr(new BlocksAttr(type));
}

static void HandleWeakAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
           std::string("0"));
    return;
  }
  
  d->addAttr(new WeakAttr());
}

static void HandleDLLImportAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
           std::string("0"));
    return;
  }
  
  d->addAttr(new DLLImportAttr());
}

static void HandleDLLExportAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
           std::string("0"));
    return;
  }
  
  d->addAttr(new DLLExportAttr());
}

static void HandleStdCallAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
           std::string("0"));
    return;
  }
  
  d->addAttr(new StdCallAttr());
}

static void HandleFastCallAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
           std::string("0"));
    return;
  }
  
  d->addAttr(new FastCallAttr());
}

static void HandleNothrowAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
           std::string("0"));
    return;
  }
  
  d->addAttr(new NoThrowAttr());
}

/// Handle __attribute__((format(type,idx,firstarg))) attributes
/// based on http://gcc.gnu.org/onlinedocs/gcc/Function-Attributes.html
static void HandleFormatAttr(Decl *d, const AttributeList &Attr, Sema &S) {

  if (!Attr.getParameterName()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string,
           "format", std::string("1"));
    return;
  }

  if (Attr.getNumArgs() != 2) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
           std::string("3"));
    return;
  }

  // GCC ignores the format attribute on K&R style function
  // prototypes, so we ignore it as well
  const FunctionTypeProto *proto = getFunctionProto(d);

  if (!proto) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type,
           "format", "function");
    return;
  }

  // FIXME: in C++ the implicit 'this' function parameter also counts.
  // this is needed in order to be compatible with GCC
  // the index must start in 1 and the limit is numargs+1
  unsigned NumArgs  = proto->getNumArgs();
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
  
  switch (FormatLen) {
  default: break;
  case 5: Supported = !memcmp(Format, "scanf", 5); break;
  case 6: Supported = !memcmp(Format, "printf", 6); break;
  case 7: Supported = !memcmp(Format, "strfmon", 7); break;
  case 8:
    Supported = (is_strftime = !memcmp(Format, "strftime", 8)) || 
                (is_NSString = !memcmp(Format, "NSString", 8));
    break;
  }
      
  if (!Supported) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_type_not_supported,
           "format", Attr.getParameterName()->getName());
    return;
  }

  // checks for the 2nd argument
  Expr *IdxExpr = static_cast<Expr *>(Attr.getArg(0));
  llvm::APSInt Idx(32);
  if (!IdxExpr->isIntegerConstantExpr(Idx, S.Context)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_int,
           "format", std::string("2"), IdxExpr->getSourceRange());
    return;
  }

  if (Idx.getZExtValue() < FirstIdx || Idx.getZExtValue() > NumArgs) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_out_of_bounds,
           "format", std::string("2"), IdxExpr->getSourceRange());
    return;
  }

  // FIXME: Do we need to bounds check?
  unsigned ArgIdx = Idx.getZExtValue() - 1;
  
  // make sure the format string is really a string
  QualType Ty = proto->getArgType(ArgIdx);

  if (is_NSString) {
    // FIXME: do we need to check if the type is NSString*?  What are
    //  the semantics?
    if (!isNSStringType(Ty, S.Context)) {
      // FIXME: Should highlight the actual expression that has the
      // wrong type.
      S.Diag(Attr.getLoc(), diag::err_format_attribute_not_NSString,
             IdxExpr->getSourceRange());
      return;
    }    
  } else if (!Ty->isPointerType() ||
             !Ty->getAsPointerType()->getPointeeType()->isCharType()) {
    // FIXME: Should highlight the actual expression that has the
    // wrong type.
    S.Diag(Attr.getLoc(), diag::err_format_attribute_not_string,
           IdxExpr->getSourceRange());
    return;
  }

  // check the 3rd argument
  Expr *FirstArgExpr = static_cast<Expr *>(Attr.getArg(1));
  llvm::APSInt FirstArg(32);
  if (!FirstArgExpr->isIntegerConstantExpr(FirstArg, S.Context)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_int,
           "format", std::string("3"), FirstArgExpr->getSourceRange());
    return;
  }

  // check if the function is variadic if the 3rd argument non-zero
  if (FirstArg != 0) {
    if (proto->isVariadic()) {
      ++NumArgs; // +1 for ...
    } else {
      S.Diag(d->getLocation(), diag::err_format_attribute_requires_variadic);
      return;
    }
  }

  // strftime requires FirstArg to be 0 because it doesn't read from any variable
  // the input is just the current time + the format string
  if (is_strftime) {
    if (FirstArg != 0) {
      S.Diag(Attr.getLoc(), diag::err_format_strftime_third_parameter,
             FirstArgExpr->getSourceRange());
      return;
    }
  // if 0 it disables parameter checking (to use with e.g. va_list)
  } else if (FirstArg != 0 && FirstArg != NumArgs) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_out_of_bounds,
           "format", std::string("3"), FirstArgExpr->getSourceRange());
    return;
  }

  d->addAttr(new FormatAttr(std::string(Format, FormatLen),
                            Idx.getZExtValue(), FirstArg.getZExtValue()));
}

static void HandleTransparentUnionAttr(Decl *d, const AttributeList &Attr,
                                       Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("0"));
    return;
  }

  // FIXME: This shouldn't be restricted to typedefs
  TypedefDecl *TD = dyn_cast<TypedefDecl>(d);
  if (!TD || !TD->getUnderlyingType()->isUnionType()) {
    S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type,
         "transparent_union", "union");
    return;
  }

  RecordDecl* RD = TD->getUnderlyingType()->getAsUnionType()->getDecl();

  // FIXME: Should we do a check for RD->isDefinition()?

  // FIXME: This isn't supposed to be restricted to pointers, but otherwise
  // we might silently generate incorrect code; see following code
  for (int i = 0; i < RD->getNumMembers(); i++) {
    if (!RD->getMember(i)->getType()->isPointerType()) {
      S.Diag(Attr.getLoc(), diag::warn_transparent_union_nonpointer);
      return;
    }
  }

  // FIXME: This is a complete hack; we should be properly propagating
  // transparent_union through Sema.  That said, this is close enough to
  // correctly compile all the common cases of transparent_union without
  // errors or warnings
  QualType NewTy = S.Context.VoidPtrTy;
  NewTy.addConst();
  TD->setUnderlyingType(NewTy);
}

static void HandleAnnotateAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
           std::string("1"));
    return;
  }
  Expr *argExpr = static_cast<Expr *>(Attr.getArg(0));
  StringLiteral *SE = dyn_cast<StringLiteral>(argExpr);
  
  // Make sure that there is a string literal as the annotation's single
  // argument.
  if (!SE) {
    S.Diag(Attr.getLoc(), diag::err_attribute_annotate_no_string);
    return;
  }
  d->addAttr(new AnnotateAttr(std::string(SE->getStrData(),
                                          SE->getByteLength())));
}

static void HandleAlignedAttr(Decl *d, const AttributeList &Attr, Sema &S) {
  // check the attribute arguments.
  if (Attr.getNumArgs() > 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
           std::string("1"));
    return;
  }

  unsigned Align = 0;
  if (Attr.getNumArgs() == 0) {
    // FIXME: This should be the target specific maximum alignment.
    // (For now we just use 128 bits which is the maximum on X86.
    Align = 128;
    return;
  }
  
  Expr *alignmentExpr = static_cast<Expr *>(Attr.getArg(0));
  llvm::APSInt Alignment(32);
  if (!alignmentExpr->isIntegerConstantExpr(Alignment, S.Context)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_not_int,
           "aligned", alignmentExpr->getSourceRange());
    return;
  }
  d->addAttr(new AlignedAttr(Alignment.getZExtValue() * 8));
}

/// HandleModeAttr - This attribute modifies the width of a decl with
/// primitive type.
///
/// Despite what would be logical, the mode attribute is a decl attribute,
/// not a type attribute: 'int ** __attribute((mode(HI))) *G;' tries to make
/// 'G' be HImode, not an intermediate pointer.
///
static void HandleModeAttr(Decl *D, const AttributeList &Attr, Sema &S) {
  // This attribute isn't documented, but glibc uses it.  It changes
  // the width of an int or unsigned int to the specified size.

  // Check that there aren't any arguments
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
           std::string("0"));
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
  switch (Len) {
  case 2:
    if (!memcmp(Str, "QI", 2)) { DestWidth =  8; break; }
    if (!memcmp(Str, "HI", 2)) { DestWidth = 16; break; }
    if (!memcmp(Str, "SI", 2)) { DestWidth = 32; break; }
    if (!memcmp(Str, "DI", 2)) { DestWidth = 64; break; }
    if (!memcmp(Str, "TI", 2)) { DestWidth = 128; break; }
    if (!memcmp(Str, "SF", 2)) { DestWidth = 32; IntegerMode = false; break; }
    if (!memcmp(Str, "DF", 2)) { DestWidth = 64; IntegerMode = false; break; }
    if (!memcmp(Str, "XF", 2)) { DestWidth = 96; IntegerMode = false; break; }
    if (!memcmp(Str, "TF", 2)) { DestWidth = 128; IntegerMode = false; break; }
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
    S.Diag(D->getLocation(), diag::err_attr_wrong_decl, "mode",
           SourceRange(Attr.getLoc(), Attr.getLoc()));
    return;
  }
  
  // FIXME: Need proper fixed-width types
  QualType NewTy;
  switch (DestWidth) {
  case 0:
    S.Diag(Attr.getLoc(), diag::err_unknown_machine_mode, Name->getName());
    return;
  default:
    S.Diag(Attr.getLoc(), diag::err_unsupported_machine_mode, Name->getName());
    return;
  case 8:
    assert(IntegerMode);
    if (OldTy->isSignedIntegerType())
      NewTy = S.Context.SignedCharTy;
    else
      NewTy = S.Context.UnsignedCharTy;
    break;
  case 16:
    assert(IntegerMode);
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
  }

  if (!OldTy->getAsBuiltinType())
    S.Diag(Attr.getLoc(), diag::err_mode_not_primitive);
  else if (!(IntegerMode && OldTy->isIntegerType()) &&
           !(!IntegerMode && OldTy->isFloatingType())) {
    S.Diag(Attr.getLoc(), diag::err_mode_wrong_type);
  }

  // Install the new type.
  if (TypedefDecl *TD = dyn_cast<TypedefDecl>(D))
    TD->setUnderlyingType(NewTy);
  else
    cast<ValueDecl>(D)->setType(NewTy);
}

//===----------------------------------------------------------------------===//
// Top Level Sema Entry Points
//===----------------------------------------------------------------------===//

/// HandleDeclAttribute - Apply the specific attribute to the specified decl if
/// the attribute applies to decls.  If the attribute is a type attribute, just
/// silently ignore it.
static void ProcessDeclAttribute(Decl *D, const AttributeList &Attr, Sema &S) {
  switch (Attr.getKind()) {
  case AttributeList::AT_IBOutlet:    HandleIBOutletAttr  (D, Attr, S); break;
  case AttributeList::AT_address_space:
    // Ignore this, this is a type attribute, handled by ProcessTypeAttributes.
    break;
  case AttributeList::AT_alias:       HandleAliasAttr     (D, Attr, S); break;
  case AttributeList::AT_aligned:     HandleAlignedAttr   (D, Attr, S); break;
  case AttributeList::AT_annotate:    HandleAnnotateAttr  (D, Attr, S); break;
  case AttributeList::AT_constructor: HandleConstructorAttr(D, Attr, S); break;
  case AttributeList::AT_deprecated:  HandleDeprecatedAttr(D, Attr, S); break;
  case AttributeList::AT_destructor:  HandleDestructorAttr(D, Attr, S); break;
  case AttributeList::AT_dllexport:   HandleDLLExportAttr (D, Attr, S); break;
  case AttributeList::AT_dllimport:   HandleDLLImportAttr (D, Attr, S); break;
  case AttributeList::AT_ext_vector_type:
    HandleExtVectorTypeAttr(D, Attr, S);
    break;
  case AttributeList::AT_fastcall:    HandleFastCallAttr  (D, Attr, S); break;
  case AttributeList::AT_format:      HandleFormatAttr    (D, Attr, S); break;
  case AttributeList::AT_mode:        HandleModeAttr      (D, Attr, S); break;
  case AttributeList::AT_nonnull:     HandleNonNullAttr   (D, Attr, S); break;
  case AttributeList::AT_noreturn:    HandleNoReturnAttr  (D, Attr, S); break;
  case AttributeList::AT_nothrow:     HandleNothrowAttr   (D, Attr, S); break;
  case AttributeList::AT_packed:      HandlePackedAttr    (D, Attr, S); break;
  case AttributeList::AT_stdcall:     HandleStdCallAttr   (D, Attr, S); break;
  case AttributeList::AT_unused:      HandleUnusedAttr    (D, Attr, S); break;
  case AttributeList::AT_vector_size: HandleVectorSizeAttr(D, Attr, S); break;
  case AttributeList::AT_visibility:  HandleVisibilityAttr(D, Attr, S); break;
  case AttributeList::AT_weak:        HandleWeakAttr      (D, Attr, S); break;
  case AttributeList::AT_transparent_union:
    HandleTransparentUnionAttr(D, Attr, S);
    break;
  case AttributeList::AT_objc_gc:     HandleObjCGCAttr    (D, Attr, S); break;
  case AttributeList::AT_blocks:      HandleBlocksAttr    (D, Attr, S); break;
  default:
#if 0
    // TODO: when we have the full set of attributes, warn about unknown ones.
    S.Diag(Attr->getLoc(), diag::warn_attribute_ignored,
           Attr->getName()->getName());
#endif
    break;
  }
}

/// ProcessDeclAttributeList - Apply all the decl attributes in the specified
/// attribute list to the specified decl, ignoring any type attributes.
void Sema::ProcessDeclAttributeList(Decl *D, const AttributeList *AttrList) {
  while (AttrList) {
    ProcessDeclAttribute(D, *AttrList, *this);
    AttrList = AttrList->getNext();
  }
}


/// ProcessDeclAttributes - Given a declarator (PD) with attributes indicated in
/// it, apply them to D.  This is a bit tricky because PD can have attributes
/// specified in many different places, and we need to find and apply them all.
void Sema::ProcessDeclAttributes(Decl *D, const Declarator &PD) {
  // Apply decl attributes from the DeclSpec if present.
  if (const AttributeList *Attrs = PD.getDeclSpec().getAttributes())
    ProcessDeclAttributeList(D, Attrs);
  
  // Walk the declarator structure, applying decl attributes that were in a type
  // position to the decl itself.  This handles cases like:
  //   int *__attr__(x)** D;
  // when X is a decl attribute.
  for (unsigned i = 0, e = PD.getNumTypeObjects(); i != e; ++i)
    if (const AttributeList *Attrs = PD.getTypeObject(i).getAttrs())
      ProcessDeclAttributeList(D, Attrs);
  
  // Finally, apply any attributes on the decl itself.
  if (const AttributeList *Attrs = PD.getAttributes())
    ProcessDeclAttributeList(D, Attrs);
}

