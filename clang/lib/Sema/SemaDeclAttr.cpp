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
#include "clang/Basic/TargetInfo.h"
using namespace clang;

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
  if (!T->isPointerType())
    return false;
  
  T = T->getAsPointerType()->getPointeeType().getCanonicalType();
  ObjCInterfaceType* ClsT = dyn_cast<ObjCInterfaceType>(T.getTypePtr());
  
  if (!ClsT)
    return false;
  
  IdentifierInfo* ClsName = ClsT->getDecl()->getIdentifier();
  
  // FIXME: Should we walk the chain of classes?
  return ClsName == &Ctx.Idents.get("NSString") ||
         ClsName == &Ctx.Idents.get("NSMutableString");
}

void Sema::ProcessDeclAttributes(Decl *D, Declarator &PD) {
  const AttributeList *DeclSpecAttrs = PD.getDeclSpec().getAttributes();
  const AttributeList *DeclaratorAttrs = PD.getAttributes();
  
  if (DeclSpecAttrs == 0 && DeclaratorAttrs == 0) return;

  ProcessDeclAttributeList(D, DeclSpecAttrs);
  
  // If there are any type attributes that were in the declarator, apply them to
  // its top level type.
  if (ValueDecl *VD = dyn_cast<ValueDecl>(D)) {
    QualType DT = VD->getType();
    ProcessTypeAttributes(DT, DeclaratorAttrs);
    VD->setType(DT);
  } else if (TypedefDecl *TD = dyn_cast<TypedefDecl>(D)) {
    QualType DT = TD->getUnderlyingType();
    ProcessTypeAttributes(DT, DeclaratorAttrs);
    TD->setUnderlyingType(DT);
  }
  
  ProcessDeclAttributeList(D, DeclaratorAttrs);
}

/// ProcessDeclAttributeList - Apply all the decl attributes in the specified
/// attribute list to the specified decl, ignoring any type attributes.
void Sema::ProcessDeclAttributeList(Decl *D, const AttributeList *AttrList) {
  while (AttrList) {
    ProcessDeclAttribute(D, *AttrList);
    AttrList = AttrList->getNext();
  }
}

/// HandleDeclAttribute - Apply the specific attribute to the specified decl if
/// the attribute applies to decls.  If the attribute is a type attribute, just
/// silently ignore it.
void Sema::ProcessDeclAttribute(Decl *D, const AttributeList &Attr) {
  switch (Attr.getKind()) {
  case AttributeList::AT_address_space:
    // Ignore this, this is a type attribute, handled by ProcessTypeAttributes.
    break;
  case AttributeList::AT_vector_size: HandleVectorSizeAttribute(D, Attr); break;
  case AttributeList::AT_ext_vector_type: 
    HandleExtVectorTypeAttribute(D, Attr);
    break;
  case AttributeList::AT_mode:       HandleModeAttribute(D, Attr);  break;
  case AttributeList::AT_alias:      HandleAliasAttribute(D, Attr); break;
  case AttributeList::AT_deprecated: HandleDeprecatedAttribute(D, Attr);break;
  case AttributeList::AT_visibility: HandleVisibilityAttribute(D, Attr);break;
  case AttributeList::AT_weak:       HandleWeakAttribute(D, Attr); break;
  case AttributeList::AT_dllimport:  HandleDLLImportAttribute(D, Attr); break;
  case AttributeList::AT_dllexport:  HandleDLLExportAttribute(D, Attr); break;
  case AttributeList::AT_nothrow:    HandleNothrowAttribute(D, Attr); break;
  case AttributeList::AT_stdcall:    HandleStdCallAttribute(D, Attr); break;
  case AttributeList::AT_fastcall:   HandleFastCallAttribute(D, Attr); break;
  case AttributeList::AT_aligned:    HandleAlignedAttribute(D, Attr); break;
  case AttributeList::AT_packed:     HandlePackedAttribute(D, Attr); break;
  case AttributeList::AT_annotate:   HandleAnnotateAttribute(D, Attr); break;
  case AttributeList::AT_noreturn:   HandleNoReturnAttribute(D, Attr); break;
  case AttributeList::AT_format:     HandleFormatAttribute(D, Attr); break;
  case AttributeList::AT_transparent_union:
    HandleTransparentUnionAttribute(D, Attr);
    break;
  default:
#if 0
    // TODO: when we have the full set of attributes, warn about unknown ones.
    Diag(Attr->getLoc(), diag::warn_attribute_ignored,
         Attr->getName()->getName());
#endif
    break;
  }
}

void Sema::HandleExtVectorTypeAttribute(Decl *d, const AttributeList &Attr) {
  TypedefDecl *tDecl = dyn_cast<TypedefDecl>(d);
  if (tDecl == 0) {
    Diag(Attr.getLoc(), diag::err_typecheck_ext_vector_not_typedef);
    return;
  }
  
  QualType curType = tDecl->getUnderlyingType();
  // check the attribute arguments.
  if (Attr.getNumArgs() != 1) {
    Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("1"));
    return;
  }
  Expr *sizeExpr = static_cast<Expr *>(Attr.getArg(0));
  llvm::APSInt vecSize(32);
  if (!sizeExpr->isIntegerConstantExpr(vecSize, Context)) {
    Diag(Attr.getLoc(), diag::err_attribute_argument_not_int,
         "ext_vector_type", sizeExpr->getSourceRange());
    return;
  }
  // unlike gcc's vector_size attribute, we do not allow vectors to be defined
  // in conjunction with complex types (pointers, arrays, functions, etc.).
  Type *canonType = curType.getCanonicalType().getTypePtr();
  if (!(canonType->isIntegerType() || canonType->isRealFloatingType())) {
    Diag(Attr.getLoc(), diag::err_attribute_invalid_vector_type,
         curType.getCanonicalType().getAsString());
    return;
  }
  // unlike gcc's vector_size attribute, the size is specified as the 
  // number of elements, not the number of bytes.
  unsigned vectorSize = static_cast<unsigned>(vecSize.getZExtValue()); 
  
  if (vectorSize == 0) {
    Diag(Attr.getLoc(), diag::err_attribute_zero_size,
         sizeExpr->getSourceRange());
    return;
  }
  // Instantiate/Install the vector type, the number of elements is > 0.
  tDecl->setUnderlyingType(Context.getExtVectorType(curType, vectorSize));
  // Remember this typedef decl, we will need it later for diagnostics.
  ExtVectorDecls.push_back(tDecl);
}


/// HandleVectorSizeAttribute - this attribute is only applicable to 
/// integral and float scalars, although arrays, pointers, and function
/// return values are allowed in conjunction with this construct. Aggregates
/// with this attribute are invalid, even if they are of the same size as a
/// corresponding scalar.
/// The raw attribute should contain precisely 1 argument, the vector size 
/// for the variable, measured in bytes. If curType and rawAttr are well
/// formed, this routine will return a new vector type.
void Sema::HandleVectorSizeAttribute(Decl *D, const AttributeList &Attr) {
  QualType CurType;
  if (ValueDecl *VD = dyn_cast<ValueDecl>(D))
    CurType = VD->getType();
  else if (TypedefDecl *TD = dyn_cast<TypedefDecl>(D))
    CurType = TD->getUnderlyingType();
  else {
    Diag(D->getLocation(), diag::err_attr_wrong_decl,std::string("vector_size"),
         SourceRange(Attr.getLoc(), Attr.getLoc()));
    return;
  }
    
  // Check the attribute arugments.
  if (Attr.getNumArgs() != 1) {
    Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("1"));
    return;
  }
  Expr *sizeExpr = static_cast<Expr *>(Attr.getArg(0));
  llvm::APSInt vecSize(32);
  if (!sizeExpr->isIntegerConstantExpr(vecSize, Context)) {
    Diag(Attr.getLoc(), diag::err_attribute_argument_not_int,
         "vector_size", sizeExpr->getSourceRange());
    return;
  }
  // navigate to the base type - we need to provide for vector pointers, 
  // vector arrays, and functions returning vectors.
  Type *canonType = CurType.getCanonicalType().getTypePtr();
  
  if (canonType->isPointerType() || canonType->isArrayType() ||
      canonType->isFunctionType()) {
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
  if (!(canonType->isIntegerType() || canonType->isRealFloatingType())) {
    Diag(Attr.getLoc(), diag::err_attribute_invalid_vector_type,
         CurType.getCanonicalType().getAsString());
    return;
  }
  unsigned typeSize = static_cast<unsigned>(Context.getTypeSize(CurType));
  // vecSize is specified in bytes - convert to bits.
  unsigned vectorSize = static_cast<unsigned>(vecSize.getZExtValue() * 8); 
  
  // the vector size needs to be an integral multiple of the type size.
  if (vectorSize % typeSize) {
    Diag(Attr.getLoc(), diag::err_attribute_invalid_size,
         sizeExpr->getSourceRange());
    return;
  }
  if (vectorSize == 0) {
    Diag(Attr.getLoc(), diag::err_attribute_zero_size,
         sizeExpr->getSourceRange());
    return;
  }
  
  // Success! Instantiate the vector type, the number of elements is > 0, and
  // not required to be a power of 2, unlike GCC.
  CurType = Context.getVectorType(CurType, vectorSize/typeSize);
  
  if (ValueDecl *VD = dyn_cast<ValueDecl>(D))
    VD->setType(CurType);
  else 
    cast<TypedefDecl>(D)->setUnderlyingType(CurType);
}

void Sema::HandlePackedAttribute(Decl *d, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.getNumArgs() > 0) {
    Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("0"));
    return;
  }
  
  if (TagDecl *TD = dyn_cast<TagDecl>(d))
    TD->addAttr(new PackedAttr());
  else if (FieldDecl *FD = dyn_cast<FieldDecl>(d)) {
    // If the alignment is less than or equal to 8 bits, the packed attribute
    // has no effect.
    if (!FD->getType()->isIncompleteType() &&
        Context.getTypeAlign(FD->getType()) <= 8)
      Diag(Attr.getLoc(), 
           diag::warn_attribute_ignored_for_field_of_type,
           Attr.getName()->getName(), FD->getType().getAsString());
    else
      FD->addAttr(new PackedAttr());
  } else
    Diag(Attr.getLoc(), diag::warn_attribute_ignored,
         Attr.getName()->getName());
}

void Sema::HandleAliasAttribute(Decl *d, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 1) {
    Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("1"));
    return;
  }
  
  Expr *Arg = static_cast<Expr*>(Attr.getArg(0));
  Arg = Arg->IgnoreParenCasts();
  StringLiteral *Str = dyn_cast<StringLiteral>(Arg);
  
  if (Str == 0 || Str->isWide()) {
    Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string,
         "alias", std::string("1"));
    return;
  }
  
  const char *Alias = Str->getStrData();
  unsigned AliasLen = Str->getByteLength();
  
  // FIXME: check if target symbol exists in current file
  
  d->addAttr(new AliasAttr(std::string(Alias, AliasLen)));
}

void Sema::HandleNoReturnAttribute(Decl *d, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("0"));
    return;
  }
  
  FunctionDecl *Fn = dyn_cast<FunctionDecl>(d);
  if (!Fn) {
    Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type,
         "noreturn", "function");
    return;
  }
  
  d->addAttr(new NoReturnAttr());
}

void Sema::HandleDeprecatedAttribute(Decl *d, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("0"));
    return;
  }
  
  d->addAttr(new DeprecatedAttr());
}

void Sema::HandleVisibilityAttribute(Decl *d, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 1) {
    Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("1"));
    return;
  }
  
  Expr *Arg = static_cast<Expr*>(Attr.getArg(0));
  Arg = Arg->IgnoreParenCasts();
  StringLiteral *Str = dyn_cast<StringLiteral>(Arg);
  
  if (Str == 0 || Str->isWide()) {
    Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string,
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
    Diag(Attr.getLoc(), diag::warn_attribute_type_not_supported,
         "visibility", TypeStr);
    return;
  }
  
  d->addAttr(new VisibilityAttr(type));
}

void Sema::HandleWeakAttribute(Decl *d, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("0"));
    return;
  }
  
  d->addAttr(new WeakAttr());
}

void Sema::HandleDLLImportAttribute(Decl *d, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("0"));
    return;
  }
  
  d->addAttr(new DLLImportAttr());
}

void Sema::HandleDLLExportAttribute(Decl *d, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("0"));
    return;
  }
  
  d->addAttr(new DLLExportAttr());
}

void Sema::HandleStdCallAttribute(Decl *d, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("0"));
    return;
  }
  
  d->addAttr(new StdCallAttr());
}

void Sema::HandleFastCallAttribute(Decl *d, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("0"));
    return;
  }
  
  d->addAttr(new FastCallAttr());
}

void Sema::HandleNothrowAttribute(Decl *d, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("0"));
    return;
  }
  
  d->addAttr(new NoThrowAttr());
}

/// Handle __attribute__((format(type,idx,firstarg))) attributes
/// based on http://gcc.gnu.org/onlinedocs/gcc/Function-Attributes.html
void Sema::HandleFormatAttribute(Decl *d, const AttributeList &Attr) {

  if (!Attr.getParameterName()) {
    Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string,
           "format", std::string("1"));
    return;
  }

  if (Attr.getNumArgs() != 2) {
    Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("3"));
    return;
  }

  // GCC ignores the format attribute on K&R style function
  // prototypes, so we ignore it as well
  const FunctionTypeProto *proto = getFunctionProto(d);

  if (!proto) {
    Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type,
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
  case 5:
    Supported = !memcmp(Format, "scanf", 5);
    break;
  case 6:
    Supported = !memcmp(Format, "printf", 6);
    break;
  case 7:
    Supported = !memcmp(Format, "strfmon", 7);
    break;
  case 8:
    Supported = (is_strftime = !memcmp(Format, "strftime", 8)) || 
                (is_NSString = !memcmp(Format, "NSString", 8));
    break;
  }
      
  if (!Supported) {
    Diag(Attr.getLoc(), diag::warn_attribute_type_not_supported,
           "format", Attr.getParameterName()->getName());
    return;
  }

  // checks for the 2nd argument
  Expr *IdxExpr = static_cast<Expr *>(Attr.getArg(0));
  llvm::APSInt Idx(Context.getTypeSize(IdxExpr->getType()));
  if (!IdxExpr->isIntegerConstantExpr(Idx, Context)) {
    Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_int,
           "format", std::string("2"), IdxExpr->getSourceRange());
    return;
  }

  if (Idx.getZExtValue() < FirstIdx || Idx.getZExtValue() > NumArgs) {
    Diag(Attr.getLoc(), diag::err_attribute_argument_out_of_bounds,
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
    if (!isNSStringType(Ty, Context)) {
      // FIXME: Should highlight the actual expression that has the
      // wrong type.
      Diag(Attr.getLoc(), diag::err_format_attribute_not_NSString,
           IdxExpr->getSourceRange());
      return;
    }    
  } else if (!Ty->isPointerType() ||
             !Ty->getAsPointerType()->getPointeeType()->isCharType()) {
    // FIXME: Should highlight the actual expression that has the
    // wrong type.
    Diag(Attr.getLoc(), diag::err_format_attribute_not_string,
         IdxExpr->getSourceRange());
    return;
  }

  // check the 3rd argument
  Expr *FirstArgExpr = static_cast<Expr *>(Attr.getArg(1));
  llvm::APSInt FirstArg(Context.getTypeSize(FirstArgExpr->getType()));
  if (!FirstArgExpr->isIntegerConstantExpr(FirstArg, Context)) {
    Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_int,
           "format", std::string("3"), FirstArgExpr->getSourceRange());
    return;
  }

  // check if the function is variadic if the 3rd argument non-zero
  if (FirstArg != 0) {
    if (proto->isVariadic()) {
      ++NumArgs; // +1 for ...
    } else {
      Diag(d->getLocation(), diag::err_format_attribute_requires_variadic);
      return;
    }
  }

  // strftime requires FirstArg to be 0 because it doesn't read from any variable
  // the input is just the current time + the format string
  if (is_strftime) {
    if (FirstArg != 0) {
      Diag(Attr.getLoc(), diag::err_format_strftime_third_parameter,
             FirstArgExpr->getSourceRange());
      return;
    }
  // if 0 it disables parameter checking (to use with e.g. va_list)
  } else if (FirstArg != 0 && FirstArg != NumArgs) {
    Diag(Attr.getLoc(), diag::err_attribute_argument_out_of_bounds,
           "format", std::string("3"), FirstArgExpr->getSourceRange());
    return;
  }

  d->addAttr(new FormatAttr(std::string(Format, FormatLen),
                            Idx.getZExtValue(), FirstArg.getZExtValue()));
}

void Sema::HandleTransparentUnionAttribute(Decl *d,
                                           const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 0) {
    Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("0"));
    return;
  }

  TypeDecl *decl = dyn_cast<TypeDecl>(d);

  if (!decl || !Context.getTypeDeclType(decl)->isUnionType()) {
    Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type,
         "transparent_union", "union");
    return;
  }

  //QualType QTy = Context.getTypeDeclType(decl);
  //const RecordType *Ty = QTy->getAsUnionType();

// FIXME
// Ty->addAttr(new TransparentUnionAttr());
}

void Sema::HandleAnnotateAttribute(Decl *d, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.getNumArgs() != 1) {
    Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("1"));
    return;
  }
  Expr *argExpr = static_cast<Expr *>(Attr.getArg(0));
  StringLiteral *SE = dyn_cast<StringLiteral>(argExpr);
  
  // Make sure that there is a string literal as the annotation's single
  // argument.
  if (!SE) {
    Diag(Attr.getLoc(), diag::err_attribute_annotate_no_string);
    return;
  }
  d->addAttr(new AnnotateAttr(std::string(SE->getStrData(),
                                          SE->getByteLength())));
}

void Sema::HandleAlignedAttribute(Decl *d, const AttributeList &Attr) {
  // check the attribute arguments.
  if (Attr.getNumArgs() > 1) {
    Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
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
  if (!alignmentExpr->isIntegerConstantExpr(Alignment, Context)) {
    Diag(Attr.getLoc(), diag::err_attribute_argument_not_int,
         "aligned", alignmentExpr->getSourceRange());
    return;
  }
  d->addAttr(new AlignedAttr(Alignment.getZExtValue() * 8));
}

/// HandleModeAttribute - This attribute modifies the width of a decl with
/// primitive type.
///
/// Despite what would be logical, the mode attribute is a decl attribute,
/// not a type attribute: 'int ** __attribute((mode(HI))) *G;' tries to make
/// 'G' be HImode, not an intermediate pointer.
///
void Sema::HandleModeAttribute(Decl *D, const AttributeList &Attr) {
  // This attribute isn't documented, but glibc uses it.  It changes
  // the width of an int or unsigned int to the specified size.

  // Check that there aren't any arguments
  if (Attr.getNumArgs() != 0) {
    Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("0"));
    return;
  }

  IdentifierInfo *Name = Attr.getParameterName();
  if (!Name) {
    Diag(Attr.getLoc(), diag::err_attribute_missing_parameter_name);
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
      DestWidth = Context.Target.getPointerWidth(0);
    if (!memcmp(Str, "byte", 4))
      DestWidth = Context.Target.getCharWidth();
    break;
  case 7:
    if (!memcmp(Str, "pointer", 7))
      DestWidth = Context.Target.getPointerWidth(0);
    break;
  }

  QualType OldTy;
  if (TypedefDecl *TD = dyn_cast<TypedefDecl>(D))
    OldTy = TD->getUnderlyingType();
  else if (ValueDecl *VD = dyn_cast<ValueDecl>(D))
    OldTy = VD->getType();
  else {
    Diag(D->getLocation(), diag::err_attr_wrong_decl, "mode",
         SourceRange(Attr.getLoc(), Attr.getLoc()));
    return;
  }
  
  // FIXME: Need proper fixed-width types
  QualType NewTy;
  switch (DestWidth) {
  case 0:
    Diag(Attr.getLoc(), diag::err_unknown_machine_mode, Name->getName());
    return;
  default:
    Diag(Attr.getLoc(), diag::err_unsupported_machine_mode, Name->getName());
    return;
  case 8:
    assert(IntegerMode);
    if (OldTy->isSignedIntegerType())
      NewTy = Context.SignedCharTy;
    else
      NewTy = Context.UnsignedCharTy;
    break;
  case 16:
    assert(IntegerMode);
    if (OldTy->isSignedIntegerType())
      NewTy = Context.ShortTy;
    else
      NewTy = Context.UnsignedShortTy;
    break;
  case 32:
    if (!IntegerMode)
      NewTy = Context.FloatTy;
    else if (OldTy->isSignedIntegerType())
      NewTy = Context.IntTy;
    else
      NewTy = Context.UnsignedIntTy;
    break;
  case 64:
    if (!IntegerMode)
      NewTy = Context.DoubleTy;
    else if (OldTy->isSignedIntegerType())
      NewTy = Context.LongLongTy;
    else
      NewTy = Context.UnsignedLongLongTy;
    break;
  }

  if (!OldTy->getAsBuiltinType())
    Diag(Attr.getLoc(), diag::err_mode_not_primitive);
  else if (!(IntegerMode && OldTy->isIntegerType()) &&
           !(!IntegerMode && OldTy->isFloatingType())) {
    Diag(Attr.getLoc(), diag::err_mode_wrong_type);
  }

  // Install the new type.
  if (TypedefDecl *TD = dyn_cast<TypedefDecl>(D))
    TD->setUnderlyingType(NewTy);
  else
    cast<ValueDecl>(D)->setType(NewTy);
}
