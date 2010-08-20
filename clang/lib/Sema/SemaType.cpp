//===--- SemaType.cpp - Semantic Analysis for Types -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements type-related semantic analysis.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/Sema.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/TypeLoc.h"
#include "clang/AST/TypeLocVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Sema/DeclSpec.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/ErrorHandling.h"
using namespace clang;

/// \brief Perform adjustment on the parameter type of a function.
///
/// This routine adjusts the given parameter type @p T to the actual
/// parameter type used by semantic analysis (C99 6.7.5.3p[7,8],
/// C++ [dcl.fct]p3). The adjusted parameter type is returned.
QualType Sema::adjustParameterType(QualType T) {
  // C99 6.7.5.3p7:
  //   A declaration of a parameter as "array of type" shall be
  //   adjusted to "qualified pointer to type", where the type
  //   qualifiers (if any) are those specified within the [ and ] of
  //   the array type derivation.
  if (T->isArrayType())
    return Context.getArrayDecayedType(T);
  
  // C99 6.7.5.3p8:
  //   A declaration of a parameter as "function returning type"
  //   shall be adjusted to "pointer to function returning type", as
  //   in 6.3.2.1.
  if (T->isFunctionType())
    return Context.getPointerType(T);

  return T;
}



/// isOmittedBlockReturnType - Return true if this declarator is missing a
/// return type because this is a omitted return type on a block literal. 
static bool isOmittedBlockReturnType(const Declarator &D) {
  if (D.getContext() != Declarator::BlockLiteralContext ||
      D.getDeclSpec().hasTypeSpecifier())
    return false;
  
  if (D.getNumTypeObjects() == 0)
    return true;   // ^{ ... }
  
  if (D.getNumTypeObjects() == 1 &&
      D.getTypeObject(0).Kind == DeclaratorChunk::Function)
    return true;   // ^(int X, float Y) { ... }
  
  return false;
}

typedef std::pair<const AttributeList*,QualType> DelayedAttribute;
typedef llvm::SmallVectorImpl<DelayedAttribute> DelayedAttributeSet;

static void ProcessTypeAttributeList(Sema &S, QualType &Type,
                                     bool IsDeclSpec,
                                     const AttributeList *Attrs,
                                     DelayedAttributeSet &DelayedFnAttrs);
static bool ProcessFnAttr(Sema &S, QualType &Type, const AttributeList &Attr);

static void ProcessDelayedFnAttrs(Sema &S, QualType &Type,
                                  DelayedAttributeSet &Attrs) {
  for (DelayedAttributeSet::iterator I = Attrs.begin(),
         E = Attrs.end(); I != E; ++I)
    if (ProcessFnAttr(S, Type, *I->first)) {
      S.Diag(I->first->getLoc(), diag::warn_function_attribute_wrong_type)
        << I->first->getName() << I->second;
      // Avoid any further processing of this attribute.
      I->first->setInvalid();
    }
  Attrs.clear();
}

static void DiagnoseDelayedFnAttrs(Sema &S, DelayedAttributeSet &Attrs) {
  for (DelayedAttributeSet::iterator I = Attrs.begin(),
         E = Attrs.end(); I != E; ++I) {
    S.Diag(I->first->getLoc(), diag::warn_function_attribute_wrong_type)
      << I->first->getName() << I->second;
    // Avoid any further processing of this attribute.
    I->first->setInvalid();
  }
  Attrs.clear();
}

/// \brief Convert the specified declspec to the appropriate type
/// object.
/// \param D  the declarator containing the declaration specifier.
/// \returns The type described by the declaration specifiers.  This function
/// never returns null.
static QualType ConvertDeclSpecToType(Sema &TheSema,
                                      Declarator &TheDeclarator,
                                      DelayedAttributeSet &Delayed) {
  // FIXME: Should move the logic from DeclSpec::Finish to here for validity
  // checking.
  const DeclSpec &DS = TheDeclarator.getDeclSpec();
  SourceLocation DeclLoc = TheDeclarator.getIdentifierLoc();
  if (DeclLoc.isInvalid())
    DeclLoc = DS.getSourceRange().getBegin();
  
  ASTContext &Context = TheSema.Context;

  QualType Result;
  switch (DS.getTypeSpecType()) {
  case DeclSpec::TST_void:
    Result = Context.VoidTy;
    break;
  case DeclSpec::TST_char:
    if (DS.getTypeSpecSign() == DeclSpec::TSS_unspecified)
      Result = Context.CharTy;
    else if (DS.getTypeSpecSign() == DeclSpec::TSS_signed)
      Result = Context.SignedCharTy;
    else {
      assert(DS.getTypeSpecSign() == DeclSpec::TSS_unsigned &&
             "Unknown TSS value");
      Result = Context.UnsignedCharTy;
    }
    break;
  case DeclSpec::TST_wchar:
    if (DS.getTypeSpecSign() == DeclSpec::TSS_unspecified)
      Result = Context.WCharTy;
    else if (DS.getTypeSpecSign() == DeclSpec::TSS_signed) {
      TheSema.Diag(DS.getTypeSpecSignLoc(), diag::ext_invalid_sign_spec)
        << DS.getSpecifierName(DS.getTypeSpecType());
      Result = Context.getSignedWCharType();
    } else {
      assert(DS.getTypeSpecSign() == DeclSpec::TSS_unsigned &&
        "Unknown TSS value");
      TheSema.Diag(DS.getTypeSpecSignLoc(), diag::ext_invalid_sign_spec)
        << DS.getSpecifierName(DS.getTypeSpecType());
      Result = Context.getUnsignedWCharType();
    }
    break;
  case DeclSpec::TST_char16:
      assert(DS.getTypeSpecSign() == DeclSpec::TSS_unspecified &&
        "Unknown TSS value");
      Result = Context.Char16Ty;
    break;
  case DeclSpec::TST_char32:
      assert(DS.getTypeSpecSign() == DeclSpec::TSS_unspecified &&
        "Unknown TSS value");
      Result = Context.Char32Ty;
    break;
  case DeclSpec::TST_unspecified:
    // "<proto1,proto2>" is an objc qualified ID with a missing id.
    if (DeclSpec::ProtocolQualifierListTy PQ = DS.getProtocolQualifiers()) {
      Result = Context.getObjCObjectType(Context.ObjCBuiltinIdTy,
                                         (ObjCProtocolDecl**)PQ,
                                         DS.getNumProtocolQualifiers());
      Result = Context.getObjCObjectPointerType(Result);
      break;
    }
    
    // If this is a missing declspec in a block literal return context, then it
    // is inferred from the return statements inside the block.
    if (isOmittedBlockReturnType(TheDeclarator)) {
      Result = Context.DependentTy;
      break;
    }

    // Unspecified typespec defaults to int in C90.  However, the C90 grammar
    // [C90 6.5] only allows a decl-spec if there was *some* type-specifier,
    // type-qualifier, or storage-class-specifier.  If not, emit an extwarn.
    // Note that the one exception to this is function definitions, which are
    // allowed to be completely missing a declspec.  This is handled in the
    // parser already though by it pretending to have seen an 'int' in this
    // case.
    if (TheSema.getLangOptions().ImplicitInt) {
      // In C89 mode, we only warn if there is a completely missing declspec
      // when one is not allowed.
      if (DS.isEmpty()) {
        TheSema.Diag(DeclLoc, diag::ext_missing_declspec)
          << DS.getSourceRange()
        << FixItHint::CreateInsertion(DS.getSourceRange().getBegin(), "int");
      }
    } else if (!DS.hasTypeSpecifier()) {
      // C99 and C++ require a type specifier.  For example, C99 6.7.2p2 says:
      // "At least one type specifier shall be given in the declaration
      // specifiers in each declaration, and in the specifier-qualifier list in
      // each struct declaration and type name."
      // FIXME: Does Microsoft really have the implicit int extension in C++?
      if (TheSema.getLangOptions().CPlusPlus &&
          !TheSema.getLangOptions().Microsoft) {
        TheSema.Diag(DeclLoc, diag::err_missing_type_specifier)
          << DS.getSourceRange();

        // When this occurs in C++ code, often something is very broken with the
        // value being declared, poison it as invalid so we don't get chains of
        // errors.
        TheDeclarator.setInvalidType(true);
      } else {
        TheSema.Diag(DeclLoc, diag::ext_missing_type_specifier)
          << DS.getSourceRange();
      }
    }

    // FALL THROUGH.
  case DeclSpec::TST_int: {
    if (DS.getTypeSpecSign() != DeclSpec::TSS_unsigned) {
      switch (DS.getTypeSpecWidth()) {
      case DeclSpec::TSW_unspecified: Result = Context.IntTy; break;
      case DeclSpec::TSW_short:       Result = Context.ShortTy; break;
      case DeclSpec::TSW_long:        Result = Context.LongTy; break;
      case DeclSpec::TSW_longlong:
        Result = Context.LongLongTy;
          
        // long long is a C99 feature.
        if (!TheSema.getLangOptions().C99 &&
            !TheSema.getLangOptions().CPlusPlus0x)
          TheSema.Diag(DS.getTypeSpecWidthLoc(), diag::ext_longlong);
        break;
      }
    } else {
      switch (DS.getTypeSpecWidth()) {
      case DeclSpec::TSW_unspecified: Result = Context.UnsignedIntTy; break;
      case DeclSpec::TSW_short:       Result = Context.UnsignedShortTy; break;
      case DeclSpec::TSW_long:        Result = Context.UnsignedLongTy; break;
      case DeclSpec::TSW_longlong:
        Result = Context.UnsignedLongLongTy;
          
        // long long is a C99 feature.
        if (!TheSema.getLangOptions().C99 &&
            !TheSema.getLangOptions().CPlusPlus0x)
          TheSema.Diag(DS.getTypeSpecWidthLoc(), diag::ext_longlong);
        break;
      }
    }
    break;
  }
  case DeclSpec::TST_float: Result = Context.FloatTy; break;
  case DeclSpec::TST_double:
    if (DS.getTypeSpecWidth() == DeclSpec::TSW_long)
      Result = Context.LongDoubleTy;
    else
      Result = Context.DoubleTy;
    break;
  case DeclSpec::TST_bool: Result = Context.BoolTy; break; // _Bool or bool
  case DeclSpec::TST_decimal32:    // _Decimal32
  case DeclSpec::TST_decimal64:    // _Decimal64
  case DeclSpec::TST_decimal128:   // _Decimal128
    TheSema.Diag(DS.getTypeSpecTypeLoc(), diag::err_decimal_unsupported);
    Result = Context.IntTy;
    TheDeclarator.setInvalidType(true);
    break;
  case DeclSpec::TST_class:
  case DeclSpec::TST_enum:
  case DeclSpec::TST_union:
  case DeclSpec::TST_struct: {
    TypeDecl *D 
      = dyn_cast_or_null<TypeDecl>(static_cast<Decl *>(DS.getTypeRep()));
    if (!D) {
      // This can happen in C++ with ambiguous lookups.
      Result = Context.IntTy;
      TheDeclarator.setInvalidType(true);
      break;
    }

    // If the type is deprecated or unavailable, diagnose it.
    TheSema.DiagnoseUseOfDecl(D, DS.getTypeSpecTypeLoc());
    
    assert(DS.getTypeSpecWidth() == 0 && DS.getTypeSpecComplex() == 0 &&
           DS.getTypeSpecSign() == 0 && "No qualifiers on tag names!");
    
    // TypeQuals handled by caller.
    Result = Context.getTypeDeclType(D);

    // In C++, make an ElaboratedType.
    if (TheSema.getLangOptions().CPlusPlus) {
      ElaboratedTypeKeyword Keyword
        = ElaboratedType::getKeywordForTypeSpec(DS.getTypeSpecType());
      Result = TheSema.getElaboratedType(Keyword, DS.getTypeSpecScope(),
                                         Result);
    }
    if (D->isInvalidDecl())
      TheDeclarator.setInvalidType(true);
    break;
  }
  case DeclSpec::TST_typename: {
    assert(DS.getTypeSpecWidth() == 0 && DS.getTypeSpecComplex() == 0 &&
           DS.getTypeSpecSign() == 0 &&
           "Can't handle qualifiers on typedef names yet!");
    Result = TheSema.GetTypeFromParser(DS.getTypeRep());
    if (Result.isNull())
      TheDeclarator.setInvalidType(true);
    else if (DeclSpec::ProtocolQualifierListTy PQ
               = DS.getProtocolQualifiers()) {
      if (const ObjCObjectType *ObjT = Result->getAs<ObjCObjectType>()) {
        // Silently drop any existing protocol qualifiers.
        // TODO: determine whether that's the right thing to do.
        if (ObjT->getNumProtocols())
          Result = ObjT->getBaseType();

        if (DS.getNumProtocolQualifiers())
          Result = Context.getObjCObjectType(Result,
                                             (ObjCProtocolDecl**) PQ,
                                             DS.getNumProtocolQualifiers());
      } else if (Result->isObjCIdType()) {
        // id<protocol-list>
        Result = Context.getObjCObjectType(Context.ObjCBuiltinIdTy,
                                           (ObjCProtocolDecl**) PQ,
                                           DS.getNumProtocolQualifiers());
        Result = Context.getObjCObjectPointerType(Result);
      } else if (Result->isObjCClassType()) {
        // Class<protocol-list>
        Result = Context.getObjCObjectType(Context.ObjCBuiltinClassTy,
                                           (ObjCProtocolDecl**) PQ,
                                           DS.getNumProtocolQualifiers());
        Result = Context.getObjCObjectPointerType(Result);
      } else {
        TheSema.Diag(DeclLoc, diag::err_invalid_protocol_qualifiers)
          << DS.getSourceRange();
        TheDeclarator.setInvalidType(true);
      }
    }

    // TypeQuals handled by caller.
    break;
  }
  case DeclSpec::TST_typeofType:
    // FIXME: Preserve type source info.
    Result = TheSema.GetTypeFromParser(DS.getTypeRep());
    assert(!Result.isNull() && "Didn't get a type for typeof?");
    // TypeQuals handled by caller.
    Result = Context.getTypeOfType(Result);
    break;
  case DeclSpec::TST_typeofExpr: {
    Expr *E = static_cast<Expr *>(DS.getTypeRep());
    assert(E && "Didn't get an expression for typeof?");
    // TypeQuals handled by caller.
    Result = TheSema.BuildTypeofExprType(E);
    if (Result.isNull()) {
      Result = Context.IntTy;
      TheDeclarator.setInvalidType(true);
    }
    break;
  }
  case DeclSpec::TST_decltype: {
    Expr *E = static_cast<Expr *>(DS.getTypeRep());
    assert(E && "Didn't get an expression for decltype?");
    // TypeQuals handled by caller.
    Result = TheSema.BuildDecltypeType(E);
    if (Result.isNull()) {
      Result = Context.IntTy;
      TheDeclarator.setInvalidType(true);
    }
    break;
  }
  case DeclSpec::TST_auto: {
    // TypeQuals handled by caller.
    Result = Context.UndeducedAutoTy;
    break;
  }

  case DeclSpec::TST_error:
    Result = Context.IntTy;
    TheDeclarator.setInvalidType(true);
    break;
  }

  // Handle complex types.
  if (DS.getTypeSpecComplex() == DeclSpec::TSC_complex) {
    if (TheSema.getLangOptions().Freestanding)
      TheSema.Diag(DS.getTypeSpecComplexLoc(), diag::ext_freestanding_complex);
    Result = Context.getComplexType(Result);
  } else if (DS.isTypeAltiVecVector()) {
    unsigned typeSize = static_cast<unsigned>(Context.getTypeSize(Result));
    assert(typeSize > 0 && "type size for vector must be greater than 0 bits");
    VectorType::AltiVecSpecific AltiVecSpec = VectorType::AltiVec;
    if (DS.isTypeAltiVecPixel())
      AltiVecSpec = VectorType::Pixel;
    else if (DS.isTypeAltiVecBool())
      AltiVecSpec = VectorType::Bool;
    Result = Context.getVectorType(Result, 128/typeSize, AltiVecSpec);
  }

  assert(DS.getTypeSpecComplex() != DeclSpec::TSC_imaginary &&
         "FIXME: imaginary types not supported yet!");

  // See if there are any attributes on the declspec that apply to the type (as
  // opposed to the decl).
  if (const AttributeList *AL = DS.getAttributes())
    ProcessTypeAttributeList(TheSema, Result, true, AL, Delayed);

  // Apply const/volatile/restrict qualifiers to T.
  if (unsigned TypeQuals = DS.getTypeQualifiers()) {

    // Enforce C99 6.7.3p2: "Types other than pointer types derived from object
    // or incomplete types shall not be restrict-qualified."  C++ also allows
    // restrict-qualified references.
    if (TypeQuals & DeclSpec::TQ_restrict) {
      if (Result->isAnyPointerType() || Result->isReferenceType()) {
        QualType EltTy;
        if (Result->isObjCObjectPointerType())
          EltTy = Result;
        else
          EltTy = Result->isPointerType() ?
                    Result->getAs<PointerType>()->getPointeeType() :
                    Result->getAs<ReferenceType>()->getPointeeType();

        // If we have a pointer or reference, the pointee must have an object
        // incomplete type.
        if (!EltTy->isIncompleteOrObjectType()) {
          TheSema.Diag(DS.getRestrictSpecLoc(),
               diag::err_typecheck_invalid_restrict_invalid_pointee)
            << EltTy << DS.getSourceRange();
          TypeQuals &= ~DeclSpec::TQ_restrict; // Remove the restrict qualifier.
        }
      } else {
        TheSema.Diag(DS.getRestrictSpecLoc(),
             diag::err_typecheck_invalid_restrict_not_pointer)
          << Result << DS.getSourceRange();
        TypeQuals &= ~DeclSpec::TQ_restrict; // Remove the restrict qualifier.
      }
    }

    // Warn about CV qualifiers on functions: C99 6.7.3p8: "If the specification
    // of a function type includes any type qualifiers, the behavior is
    // undefined."
    if (Result->isFunctionType() && TypeQuals) {
      // Get some location to point at, either the C or V location.
      SourceLocation Loc;
      if (TypeQuals & DeclSpec::TQ_const)
        Loc = DS.getConstSpecLoc();
      else if (TypeQuals & DeclSpec::TQ_volatile)
        Loc = DS.getVolatileSpecLoc();
      else {
        assert((TypeQuals & DeclSpec::TQ_restrict) &&
               "Has CVR quals but not C, V, or R?");
        Loc = DS.getRestrictSpecLoc();
      }
      TheSema.Diag(Loc, diag::warn_typecheck_function_qualifiers)
        << Result << DS.getSourceRange();
    }

    // C++ [dcl.ref]p1:
    //   Cv-qualified references are ill-formed except when the
    //   cv-qualifiers are introduced through the use of a typedef
    //   (7.1.3) or of a template type argument (14.3), in which
    //   case the cv-qualifiers are ignored.
    // FIXME: Shouldn't we be checking SCS_typedef here?
    if (DS.getTypeSpecType() == DeclSpec::TST_typename &&
        TypeQuals && Result->isReferenceType()) {
      TypeQuals &= ~DeclSpec::TQ_const;
      TypeQuals &= ~DeclSpec::TQ_volatile;
    }

    Qualifiers Quals = Qualifiers::fromCVRMask(TypeQuals);
    Result = Context.getQualifiedType(Result, Quals);
  }

  return Result;
}

static std::string getPrintableNameForEntity(DeclarationName Entity) {
  if (Entity)
    return Entity.getAsString();

  return "type name";
}

QualType Sema::BuildQualifiedType(QualType T, SourceLocation Loc,
                                  Qualifiers Qs) {
  // Enforce C99 6.7.3p2: "Types other than pointer types derived from
  // object or incomplete types shall not be restrict-qualified."
  if (Qs.hasRestrict()) {
    unsigned DiagID = 0;
    QualType ProblemTy;

    const Type *Ty = T->getCanonicalTypeInternal().getTypePtr();
    if (const ReferenceType *RTy = dyn_cast<ReferenceType>(Ty)) {
      if (!RTy->getPointeeType()->isIncompleteOrObjectType()) {
        DiagID = diag::err_typecheck_invalid_restrict_invalid_pointee;
        ProblemTy = T->getAs<ReferenceType>()->getPointeeType();
      }
    } else if (const PointerType *PTy = dyn_cast<PointerType>(Ty)) {
      if (!PTy->getPointeeType()->isIncompleteOrObjectType()) {
        DiagID = diag::err_typecheck_invalid_restrict_invalid_pointee;
        ProblemTy = T->getAs<PointerType>()->getPointeeType();
      }
    } else if (const MemberPointerType *PTy = dyn_cast<MemberPointerType>(Ty)) {
      if (!PTy->getPointeeType()->isIncompleteOrObjectType()) {
        DiagID = diag::err_typecheck_invalid_restrict_invalid_pointee;
        ProblemTy = T->getAs<PointerType>()->getPointeeType();
      }      
    } else if (!Ty->isDependentType()) {
      // FIXME: this deserves a proper diagnostic
      DiagID = diag::err_typecheck_invalid_restrict_invalid_pointee;
      ProblemTy = T;
    }

    if (DiagID) {
      Diag(Loc, DiagID) << ProblemTy;
      Qs.removeRestrict();
    }
  }

  return Context.getQualifiedType(T, Qs);
}

/// \brief Build a pointer type.
///
/// \param T The type to which we'll be building a pointer.
///
/// \param Loc The location of the entity whose type involves this
/// pointer type or, if there is no such entity, the location of the
/// type that will have pointer type.
///
/// \param Entity The name of the entity that involves the pointer
/// type, if known.
///
/// \returns A suitable pointer type, if there are no
/// errors. Otherwise, returns a NULL type.
QualType Sema::BuildPointerType(QualType T,
                                SourceLocation Loc, DeclarationName Entity) {
  if (T->isReferenceType()) {
    // C++ 8.3.2p4: There shall be no ... pointers to references ...
    Diag(Loc, diag::err_illegal_decl_pointer_to_reference)
      << getPrintableNameForEntity(Entity) << T;
    return QualType();
  }

  assert(!T->isObjCObjectType() && "Should build ObjCObjectPointerType");

  // Build the pointer type.
  return Context.getPointerType(T);
}

/// \brief Build a reference type.
///
/// \param T The type to which we'll be building a reference.
///
/// \param Loc The location of the entity whose type involves this
/// reference type or, if there is no such entity, the location of the
/// type that will have reference type.
///
/// \param Entity The name of the entity that involves the reference
/// type, if known.
///
/// \returns A suitable reference type, if there are no
/// errors. Otherwise, returns a NULL type.
QualType Sema::BuildReferenceType(QualType T, bool SpelledAsLValue,
                                  SourceLocation Loc,
                                  DeclarationName Entity) {
  bool LValueRef = SpelledAsLValue || T->getAs<LValueReferenceType>();

  // C++0x [dcl.typedef]p9: If a typedef TD names a type that is a
  //   reference to a type T, and attempt to create the type "lvalue
  //   reference to cv TD" creates the type "lvalue reference to T".
  // We use the qualifiers (restrict or none) of the original reference,
  // not the new ones. This is consistent with GCC.

  // C++ [dcl.ref]p4: There shall be no references to references.
  //
  // According to C++ DR 106, references to references are only
  // diagnosed when they are written directly (e.g., "int & &"),
  // but not when they happen via a typedef:
  //
  //   typedef int& intref;
  //   typedef intref& intref2;
  //
  // Parser::ParseDeclaratorInternal diagnoses the case where
  // references are written directly; here, we handle the
  // collapsing of references-to-references as described in C++
  // DR 106 and amended by C++ DR 540.

  // C++ [dcl.ref]p1:
  //   A declarator that specifies the type "reference to cv void"
  //   is ill-formed.
  if (T->isVoidType()) {
    Diag(Loc, diag::err_reference_to_void);
    return QualType();
  }

  // Handle restrict on references.
  if (LValueRef)
    return Context.getLValueReferenceType(T, SpelledAsLValue);
  return Context.getRValueReferenceType(T);
}

/// \brief Build an array type.
///
/// \param T The type of each element in the array.
///
/// \param ASM C99 array size modifier (e.g., '*', 'static').
///
/// \param ArraySize Expression describing the size of the array.
///
/// \param Loc The location of the entity whose type involves this
/// array type or, if there is no such entity, the location of the
/// type that will have array type.
///
/// \param Entity The name of the entity that involves the array
/// type, if known.
///
/// \returns A suitable array type, if there are no errors. Otherwise,
/// returns a NULL type.
QualType Sema::BuildArrayType(QualType T, ArrayType::ArraySizeModifier ASM,
                              Expr *ArraySize, unsigned Quals,
                              SourceRange Brackets, DeclarationName Entity) {

  SourceLocation Loc = Brackets.getBegin();
  if (getLangOptions().CPlusPlus) {
    // C++ [dcl.array]p1:
    //   T is called the array element type; this type shall not be a reference
    //   type, the (possibly cv-qualified) type void, a function type or an 
    //   abstract class type.
    //
    // Note: function types are handled in the common path with C.
    if (T->isReferenceType()) {
      Diag(Loc, diag::err_illegal_decl_array_of_references)
      << getPrintableNameForEntity(Entity) << T;
      return QualType();
    }
    
    if (T->isVoidType()) {
      Diag(Loc, diag::err_illegal_decl_array_incomplete_type) << T;
      return QualType();
    }
    
    if (RequireNonAbstractType(Brackets.getBegin(), T, 
                               diag::err_array_of_abstract_type))
      return QualType();
    
  } else {
    // C99 6.7.5.2p1: If the element type is an incomplete or function type,
    // reject it (e.g. void ary[7], struct foo ary[7], void ary[7]())
    if (RequireCompleteType(Loc, T,
                            diag::err_illegal_decl_array_incomplete_type))
      return QualType();
  }

  if (T->isFunctionType()) {
    Diag(Loc, diag::err_illegal_decl_array_of_functions)
      << getPrintableNameForEntity(Entity) << T;
    return QualType();
  }

  if (Context.getCanonicalType(T) == Context.UndeducedAutoTy) {
    Diag(Loc,  diag::err_illegal_decl_array_of_auto)
      << getPrintableNameForEntity(Entity);
    return QualType();
  }

  if (const RecordType *EltTy = T->getAs<RecordType>()) {
    // If the element type is a struct or union that contains a variadic
    // array, accept it as a GNU extension: C99 6.7.2.1p2.
    if (EltTy->getDecl()->hasFlexibleArrayMember())
      Diag(Loc, diag::ext_flexible_array_in_array) << T;
  } else if (T->isObjCObjectType()) {
    Diag(Loc, diag::err_objc_array_of_interfaces) << T;
    return QualType();
  }

  // C99 6.7.5.2p1: The size expression shall have integer type.
  if (ArraySize && !ArraySize->isTypeDependent() &&
      !ArraySize->getType()->isIntegerType()) {
    Diag(ArraySize->getLocStart(), diag::err_array_size_non_int)
      << ArraySize->getType() << ArraySize->getSourceRange();
    return QualType();
  }
  llvm::APSInt ConstVal(Context.getTypeSize(Context.getSizeType()));
  if (!ArraySize) {
    if (ASM == ArrayType::Star)
      T = Context.getVariableArrayType(T, 0, ASM, Quals, Brackets);
    else
      T = Context.getIncompleteArrayType(T, ASM, Quals);
  } else if (ArraySize->isTypeDependent() || ArraySize->isValueDependent()) {
    T = Context.getDependentSizedArrayType(T, ArraySize, ASM, Quals, Brackets);
  } else if (!ArraySize->isIntegerConstantExpr(ConstVal, Context) ||
             (!T->isDependentType() && !T->isIncompleteType() &&
              !T->isConstantSizeType())) {
    // Per C99, a variable array is an array with either a non-constant
    // size or an element type that has a non-constant-size
    T = Context.getVariableArrayType(T, ArraySize, ASM, Quals, Brackets);
  } else {
    // C99 6.7.5.2p1: If the expression is a constant expression, it shall
    // have a value greater than zero.
    if (ConstVal.isSigned() && ConstVal.isNegative()) {
      Diag(ArraySize->getLocStart(),
           diag::err_typecheck_negative_array_size)
        << ArraySize->getSourceRange();
      return QualType();
    }
    if (ConstVal == 0) {
      // GCC accepts zero sized static arrays. We allow them when
      // we're not in a SFINAE context.
      Diag(ArraySize->getLocStart(), 
           isSFINAEContext()? diag::err_typecheck_zero_array_size
                            : diag::ext_typecheck_zero_array_size)
        << ArraySize->getSourceRange();
    } else if (!T->isDependentType() && !T->isVariablyModifiedType() && 
               !T->isIncompleteType()) {
      // Is the array too large?      
      unsigned ActiveSizeBits
        = ConstantArrayType::getNumAddressingBits(Context, T, ConstVal);
      if (ActiveSizeBits > ConstantArrayType::getMaxSizeBits(Context))
        Diag(ArraySize->getLocStart(), diag::err_array_too_large)
          << ConstVal.toString(10)
          << ArraySize->getSourceRange();
    }
    
    T = Context.getConstantArrayType(T, ConstVal, ASM, Quals);
  }
  // If this is not C99, extwarn about VLA's and C99 array size modifiers.
  if (!getLangOptions().C99) {
    if (T->isVariableArrayType()) {
      // Prohibit the use of non-POD types in VLAs.
      if (!T->isDependentType() && 
          !Context.getBaseElementType(T)->isPODType()) {
        Diag(Loc, diag::err_vla_non_pod)
          << Context.getBaseElementType(T);
        return QualType();
      } 
      // Prohibit the use of VLAs during template argument deduction.
      else if (isSFINAEContext()) {
        Diag(Loc, diag::err_vla_in_sfinae);
        return QualType();
      }
      // Just extwarn about VLAs.
      else
        Diag(Loc, diag::ext_vla);
    } else if (ASM != ArrayType::Normal || Quals != 0)
      Diag(Loc, 
           getLangOptions().CPlusPlus? diag::err_c99_array_usage_cxx
                                     : diag::ext_c99_array_usage);
  }

  return T;
}

/// \brief Build an ext-vector type.
///
/// Run the required checks for the extended vector type.
QualType Sema::BuildExtVectorType(QualType T, ExprArg ArraySize,
                                  SourceLocation AttrLoc) {

  Expr *Arg = (Expr *)ArraySize.get();

  // unlike gcc's vector_size attribute, we do not allow vectors to be defined
  // in conjunction with complex types (pointers, arrays, functions, etc.).
  if (!T->isDependentType() &&
      !T->isIntegerType() && !T->isRealFloatingType()) {
    Diag(AttrLoc, diag::err_attribute_invalid_vector_type) << T;
    return QualType();
  }

  if (!Arg->isTypeDependent() && !Arg->isValueDependent()) {
    llvm::APSInt vecSize(32);
    if (!Arg->isIntegerConstantExpr(vecSize, Context)) {
      Diag(AttrLoc, diag::err_attribute_argument_not_int)
      << "ext_vector_type" << Arg->getSourceRange();
      return QualType();
    }

    // unlike gcc's vector_size attribute, the size is specified as the
    // number of elements, not the number of bytes.
    unsigned vectorSize = static_cast<unsigned>(vecSize.getZExtValue());

    if (vectorSize == 0) {
      Diag(AttrLoc, diag::err_attribute_zero_size)
      << Arg->getSourceRange();
      return QualType();
    }

    if (!T->isDependentType())
      return Context.getExtVectorType(T, vectorSize);
  }

  return Context.getDependentSizedExtVectorType(T, ArraySize.takeAs<Expr>(),
                                                AttrLoc);
}

/// \brief Build a function type.
///
/// This routine checks the function type according to C++ rules and
/// under the assumption that the result type and parameter types have
/// just been instantiated from a template. It therefore duplicates
/// some of the behavior of GetTypeForDeclarator, but in a much
/// simpler form that is only suitable for this narrow use case.
///
/// \param T The return type of the function.
///
/// \param ParamTypes The parameter types of the function. This array
/// will be modified to account for adjustments to the types of the
/// function parameters.
///
/// \param NumParamTypes The number of parameter types in ParamTypes.
///
/// \param Variadic Whether this is a variadic function type.
///
/// \param Quals The cvr-qualifiers to be applied to the function type.
///
/// \param Loc The location of the entity whose type involves this
/// function type or, if there is no such entity, the location of the
/// type that will have function type.
///
/// \param Entity The name of the entity that involves the function
/// type, if known.
///
/// \returns A suitable function type, if there are no
/// errors. Otherwise, returns a NULL type.
QualType Sema::BuildFunctionType(QualType T,
                                 QualType *ParamTypes,
                                 unsigned NumParamTypes,
                                 bool Variadic, unsigned Quals,
                                 SourceLocation Loc, DeclarationName Entity,
                                 const FunctionType::ExtInfo &Info) {
  if (T->isArrayType() || T->isFunctionType()) {
    Diag(Loc, diag::err_func_returning_array_function) 
      << T->isFunctionType() << T;
    return QualType();
  }
       
  bool Invalid = false;
  for (unsigned Idx = 0; Idx < NumParamTypes; ++Idx) {
    QualType ParamType = adjustParameterType(ParamTypes[Idx]);
    if (ParamType->isVoidType()) {
      Diag(Loc, diag::err_param_with_void_type);
      Invalid = true;
    }

    ParamTypes[Idx] = ParamType;
  }

  if (Invalid)
    return QualType();

  return Context.getFunctionType(T, ParamTypes, NumParamTypes, Variadic,
                                 Quals, false, false, 0, 0, Info);
}

/// \brief Build a member pointer type \c T Class::*.
///
/// \param T the type to which the member pointer refers.
/// \param Class the class type into which the member pointer points.
/// \param CVR Qualifiers applied to the member pointer type
/// \param Loc the location where this type begins
/// \param Entity the name of the entity that will have this member pointer type
///
/// \returns a member pointer type, if successful, or a NULL type if there was
/// an error.
QualType Sema::BuildMemberPointerType(QualType T, QualType Class,
                                      SourceLocation Loc,
                                      DeclarationName Entity) {
  // Verify that we're not building a pointer to pointer to function with
  // exception specification.
  if (CheckDistantExceptionSpec(T)) {
    Diag(Loc, diag::err_distant_exception_spec);

    // FIXME: If we're doing this as part of template instantiation,
    // we should return immediately.

    // Build the type anyway, but use the canonical type so that the
    // exception specifiers are stripped off.
    T = Context.getCanonicalType(T);
  }

  // C++ 8.3.3p3: A pointer to member shall not point to ... a member
  //   with reference type, or "cv void."
  if (T->isReferenceType()) {
    Diag(Loc, diag::err_illegal_decl_mempointer_to_reference)
      << (Entity? Entity.getAsString() : "type name") << T;
    return QualType();
  }

  if (T->isVoidType()) {
    Diag(Loc, diag::err_illegal_decl_mempointer_to_void)
      << (Entity? Entity.getAsString() : "type name");
    return QualType();
  }

  if (!Class->isDependentType() && !Class->isRecordType()) {
    Diag(Loc, diag::err_mempointer_in_nonclass_type) << Class;
    return QualType();
  }

  // In the Microsoft ABI, the class is allowed to be an incomplete
  // type. In such cases, the compiler makes a worst-case assumption.
  // We make no such assumption right now, so emit an error if the
  // class isn't a complete type.
  if (Context.Target.getCXXABI() == CXXABI_Microsoft &&
      RequireCompleteType(Loc, Class, diag::err_incomplete_type))
    return QualType();

  return Context.getMemberPointerType(T, Class.getTypePtr());
}

/// \brief Build a block pointer type.
///
/// \param T The type to which we'll be building a block pointer.
///
/// \param CVR The cvr-qualifiers to be applied to the block pointer type.
///
/// \param Loc The location of the entity whose type involves this
/// block pointer type or, if there is no such entity, the location of the
/// type that will have block pointer type.
///
/// \param Entity The name of the entity that involves the block pointer
/// type, if known.
///
/// \returns A suitable block pointer type, if there are no
/// errors. Otherwise, returns a NULL type.
QualType Sema::BuildBlockPointerType(QualType T, 
                                     SourceLocation Loc,
                                     DeclarationName Entity) {
  if (!T->isFunctionType()) {
    Diag(Loc, diag::err_nonfunction_block_type);
    return QualType();
  }

  return Context.getBlockPointerType(T);
}

QualType Sema::GetTypeFromParser(TypeTy *Ty, TypeSourceInfo **TInfo) {
  QualType QT = QualType::getFromOpaquePtr(Ty);
  if (QT.isNull()) {
    if (TInfo) *TInfo = 0;
    return QualType();
  }

  TypeSourceInfo *DI = 0;
  if (LocInfoType *LIT = dyn_cast<LocInfoType>(QT)) {
    QT = LIT->getType();
    DI = LIT->getTypeSourceInfo();
  }

  if (TInfo) *TInfo = DI;
  return QT;
}

/// GetTypeForDeclarator - Convert the type for the specified
/// declarator to Type instances.
///
/// If OwnedDecl is non-NULL, and this declarator's decl-specifier-seq
/// owns the declaration of a type (e.g., the definition of a struct
/// type), then *OwnedDecl will receive the owned declaration.
///
/// The result of this call will never be null, but the associated
/// type may be a null type if there's an unrecoverable error.
TypeSourceInfo *Sema::GetTypeForDeclarator(Declarator &D, Scope *S,
                                           TagDecl **OwnedDecl) {
  // Determine the type of the declarator. Not all forms of declarator
  // have a type.
  QualType T;
  TypeSourceInfo *ReturnTypeInfo = 0;
  
  llvm::SmallVector<DelayedAttribute,4> FnAttrsFromDeclSpec;

  switch (D.getName().getKind()) {
  case UnqualifiedId::IK_Identifier:
  case UnqualifiedId::IK_OperatorFunctionId:
  case UnqualifiedId::IK_LiteralOperatorId:
  case UnqualifiedId::IK_TemplateId:
    T = ConvertDeclSpecToType(*this, D, FnAttrsFromDeclSpec);
    
    if (!D.isInvalidType() && D.getDeclSpec().isTypeSpecOwned()) {
      TagDecl* Owned = cast<TagDecl>((Decl *)D.getDeclSpec().getTypeRep());
      // Owned is embedded if it was defined here, or if it is the
      // very first (i.e., canonical) declaration of this tag type.
      Owned->setEmbeddedInDeclarator(Owned->isDefinition() ||
                                     Owned->isCanonicalDecl());
      if (OwnedDecl) *OwnedDecl = Owned;
    }
    break;

  case UnqualifiedId::IK_ConstructorName:
  case UnqualifiedId::IK_ConstructorTemplateId:
  case UnqualifiedId::IK_DestructorName:
    // Constructors and destructors don't have return types. Use
    // "void" instead. 
    T = Context.VoidTy;
    break;

  case UnqualifiedId::IK_ConversionFunctionId:
    // The result type of a conversion function is the type that it
    // converts to.
    T = GetTypeFromParser(D.getName().ConversionFunctionId, 
                          &ReturnTypeInfo);
    break;
  }
  
  if (T.isNull())
    return Context.getNullTypeSourceInfo();

  if (T == Context.UndeducedAutoTy) {
    int Error = -1;

    switch (D.getContext()) {
    case Declarator::KNRTypeListContext:
      assert(0 && "K&R type lists aren't allowed in C++");
      break;
    case Declarator::PrototypeContext:
      Error = 0; // Function prototype
      break;
    case Declarator::MemberContext:
      switch (cast<TagDecl>(CurContext)->getTagKind()) {
      case TTK_Enum: assert(0 && "unhandled tag kind"); break;
      case TTK_Struct: Error = 1; /* Struct member */ break;
      case TTK_Union:  Error = 2; /* Union member */ break;
      case TTK_Class:  Error = 3; /* Class member */ break;
      }
      break;
    case Declarator::CXXCatchContext:
      Error = 4; // Exception declaration
      break;
    case Declarator::TemplateParamContext:
      Error = 5; // Template parameter
      break;
    case Declarator::BlockLiteralContext:
      Error = 6;  // Block literal
      break;
    case Declarator::FileContext:
    case Declarator::BlockContext:
    case Declarator::ForContext:
    case Declarator::ConditionContext:
    case Declarator::TypeNameContext:
      break;
    }

    if (Error != -1) {
      Diag(D.getDeclSpec().getTypeSpecTypeLoc(), diag::err_auto_not_allowed)
        << Error;
      T = Context.IntTy;
      D.setInvalidType(true);
    }
  }

  // The name we're declaring, if any.
  DeclarationName Name;
  if (D.getIdentifier())
    Name = D.getIdentifier();

  llvm::SmallVector<DelayedAttribute,4> FnAttrsFromPreviousChunk;

  // Walk the DeclTypeInfo, building the recursive type as we go.
  // DeclTypeInfos are ordered from the identifier out, which is
  // opposite of what we want :).
  for (unsigned i = 0, e = D.getNumTypeObjects(); i != e; ++i) {
    DeclaratorChunk &DeclType = D.getTypeObject(e-i-1);
    switch (DeclType.Kind) {
    default: assert(0 && "Unknown decltype!");
    case DeclaratorChunk::BlockPointer:
      // If blocks are disabled, emit an error.
      if (!LangOpts.Blocks)
        Diag(DeclType.Loc, diag::err_blocks_disable);

      T = BuildBlockPointerType(T, D.getIdentifierLoc(), Name);
      if (DeclType.Cls.TypeQuals)
        T = BuildQualifiedType(T, DeclType.Loc, DeclType.Cls.TypeQuals);
      break;
    case DeclaratorChunk::Pointer:
      // Verify that we're not building a pointer to pointer to function with
      // exception specification.
      if (getLangOptions().CPlusPlus && CheckDistantExceptionSpec(T)) {
        Diag(D.getIdentifierLoc(), diag::err_distant_exception_spec);
        D.setInvalidType(true);
        // Build the type anyway.
      }
      if (getLangOptions().ObjC1 && T->getAs<ObjCObjectType>()) {
        T = Context.getObjCObjectPointerType(T);
        if (DeclType.Ptr.TypeQuals)
          T = BuildQualifiedType(T, DeclType.Loc, DeclType.Ptr.TypeQuals);
        break;
      }
      T = BuildPointerType(T, DeclType.Loc, Name);
      if (DeclType.Ptr.TypeQuals)
        T = BuildQualifiedType(T, DeclType.Loc, DeclType.Ptr.TypeQuals);
      break;
    case DeclaratorChunk::Reference: {
      // Verify that we're not building a reference to pointer to function with
      // exception specification.
      if (getLangOptions().CPlusPlus && CheckDistantExceptionSpec(T)) {
        Diag(D.getIdentifierLoc(), diag::err_distant_exception_spec);
        D.setInvalidType(true);
        // Build the type anyway.
      }
      T = BuildReferenceType(T, DeclType.Ref.LValueRef, DeclType.Loc, Name);

      Qualifiers Quals;
      if (DeclType.Ref.HasRestrict)
        T = BuildQualifiedType(T, DeclType.Loc, Qualifiers::Restrict);
      break;
    }
    case DeclaratorChunk::Array: {
      // Verify that we're not building an array of pointers to function with
      // exception specification.
      if (getLangOptions().CPlusPlus && CheckDistantExceptionSpec(T)) {
        Diag(D.getIdentifierLoc(), diag::err_distant_exception_spec);
        D.setInvalidType(true);
        // Build the type anyway.
      }
      DeclaratorChunk::ArrayTypeInfo &ATI = DeclType.Arr;
      Expr *ArraySize = static_cast<Expr*>(ATI.NumElts);
      ArrayType::ArraySizeModifier ASM;
      if (ATI.isStar)
        ASM = ArrayType::Star;
      else if (ATI.hasStatic)
        ASM = ArrayType::Static;
      else
        ASM = ArrayType::Normal;
      if (ASM == ArrayType::Star &&
          D.getContext() != Declarator::PrototypeContext) {
        // FIXME: This check isn't quite right: it allows star in prototypes
        // for function definitions, and disallows some edge cases detailed
        // in http://gcc.gnu.org/ml/gcc-patches/2009-02/msg00133.html
        Diag(DeclType.Loc, diag::err_array_star_outside_prototype);
        ASM = ArrayType::Normal;
        D.setInvalidType(true);
      }
      T = BuildArrayType(T, ASM, ArraySize,
                         Qualifiers::fromCVRMask(ATI.TypeQuals),
                         SourceRange(DeclType.Loc, DeclType.EndLoc), Name);
      break;
    }
    case DeclaratorChunk::Function: {
      // If the function declarator has a prototype (i.e. it is not () and
      // does not have a K&R-style identifier list), then the arguments are part
      // of the type, otherwise the argument list is ().
      const DeclaratorChunk::FunctionTypeInfo &FTI = DeclType.Fun;

      // C99 6.7.5.3p1: The return type may not be a function or array type.
      // For conversion functions, we'll diagnose this particular error later.
      if ((T->isArrayType() || T->isFunctionType()) &&
          (D.getName().getKind() != UnqualifiedId::IK_ConversionFunctionId)) {
        Diag(DeclType.Loc, diag::err_func_returning_array_function) 
          << T->isFunctionType() << T;
        T = Context.IntTy;
        D.setInvalidType(true);
      }

      // cv-qualifiers on return types are pointless except when the type is a
      // class type in C++.
      if (T.getCVRQualifiers() && D.getDeclSpec().getTypeQualifiers() &&
          (!getLangOptions().CPlusPlus ||
           (!T->isDependentType() && !T->isRecordType()))) {
        unsigned Quals = D.getDeclSpec().getTypeQualifiers();
        std::string QualStr;
        unsigned NumQuals = 0;
        SourceLocation Loc;
        if (Quals & Qualifiers::Const) {
          Loc = D.getDeclSpec().getConstSpecLoc();
          ++NumQuals;
          QualStr = "const";
        }
        if (Quals & Qualifiers::Volatile) {
          if (NumQuals == 0) {
            Loc = D.getDeclSpec().getVolatileSpecLoc();
            QualStr = "volatile";
          } else
            QualStr += " volatile";
          ++NumQuals;
        }
        if (Quals & Qualifiers::Restrict) {
          if (NumQuals == 0) {
            Loc = D.getDeclSpec().getRestrictSpecLoc();
            QualStr = "restrict";
          } else
            QualStr += " restrict";
          ++NumQuals;
        }
        assert(NumQuals > 0 && "No known qualifiers?");
            
        SemaDiagnosticBuilder DB = Diag(Loc, diag::warn_qual_return_type);
        DB << QualStr << NumQuals;
        if (Quals & Qualifiers::Const)
          DB << FixItHint::CreateRemoval(D.getDeclSpec().getConstSpecLoc());
        if (Quals & Qualifiers::Volatile)
          DB << FixItHint::CreateRemoval(D.getDeclSpec().getVolatileSpecLoc());
        if (Quals & Qualifiers::Restrict)
          DB << FixItHint::CreateRemoval(D.getDeclSpec().getRestrictSpecLoc());
      }
      
      if (getLangOptions().CPlusPlus && D.getDeclSpec().isTypeSpecOwned()) {
        // C++ [dcl.fct]p6:
        //   Types shall not be defined in return or parameter types.
        TagDecl *Tag = cast<TagDecl>((Decl *)D.getDeclSpec().getTypeRep());
        if (Tag->isDefinition())
          Diag(Tag->getLocation(), diag::err_type_defined_in_result_type)
            << Context.getTypeDeclType(Tag);
      }

      // Exception specs are not allowed in typedefs. Complain, but add it
      // anyway.
      if (FTI.hasExceptionSpec &&
          D.getDeclSpec().getStorageClassSpec() == DeclSpec::SCS_typedef)
        Diag(FTI.getThrowLoc(), diag::err_exception_spec_in_typedef);

      if (!FTI.NumArgs && !FTI.isVariadic && !getLangOptions().CPlusPlus) {
        // Simple void foo(), where the incoming T is the result type.
        T = Context.getFunctionNoProtoType(T);
      } else {
        // We allow a zero-parameter variadic function in C if the
        // function is marked with the "overloadable" attribute. Scan
        // for this attribute now.
        if (!FTI.NumArgs && FTI.isVariadic && !getLangOptions().CPlusPlus) {
          bool Overloadable = false;
          for (const AttributeList *Attrs = D.getAttributes();
               Attrs; Attrs = Attrs->getNext()) {
            if (Attrs->getKind() == AttributeList::AT_overloadable) {
              Overloadable = true;
              break;
            }
          }

          if (!Overloadable)
            Diag(FTI.getEllipsisLoc(), diag::err_ellipsis_first_arg);
        }

        if (FTI.NumArgs && FTI.ArgInfo[0].Param == 0) {
          // C99 6.7.5.3p3: Reject int(x,y,z) when it's not a function
          // definition.
          Diag(FTI.ArgInfo[0].IdentLoc, diag::err_ident_list_in_fn_declaration);
          D.setInvalidType(true);
          break;
        }

        // Otherwise, we have a function with an argument list that is
        // potentially variadic.
        llvm::SmallVector<QualType, 16> ArgTys;
        ArgTys.reserve(FTI.NumArgs);

        for (unsigned i = 0, e = FTI.NumArgs; i != e; ++i) {
          ParmVarDecl *Param =
            cast<ParmVarDecl>(FTI.ArgInfo[i].Param.getAs<Decl>());
          QualType ArgTy = Param->getType();
          assert(!ArgTy.isNull() && "Couldn't parse type?");

          // Adjust the parameter type.
          assert((ArgTy == adjustParameterType(ArgTy)) && "Unadjusted type?");

          // Look for 'void'.  void is allowed only as a single argument to a
          // function with no other parameters (C99 6.7.5.3p10).  We record
          // int(void) as a FunctionProtoType with an empty argument list.
          if (ArgTy->isVoidType()) {
            // If this is something like 'float(int, void)', reject it.  'void'
            // is an incomplete type (C99 6.2.5p19) and function decls cannot
            // have arguments of incomplete type.
            if (FTI.NumArgs != 1 || FTI.isVariadic) {
              Diag(DeclType.Loc, diag::err_void_only_param);
              ArgTy = Context.IntTy;
              Param->setType(ArgTy);
            } else if (FTI.ArgInfo[i].Ident) {
              // Reject, but continue to parse 'int(void abc)'.
              Diag(FTI.ArgInfo[i].IdentLoc,
                   diag::err_param_with_void_type);
              ArgTy = Context.IntTy;
              Param->setType(ArgTy);
            } else {
              // Reject, but continue to parse 'float(const void)'.
              if (ArgTy.hasQualifiers())
                Diag(DeclType.Loc, diag::err_void_param_qualified);

              // Do not add 'void' to the ArgTys list.
              break;
            }
          } else if (!FTI.hasPrototype) {
            if (ArgTy->isPromotableIntegerType()) {
              ArgTy = Context.getPromotedIntegerType(ArgTy);
            } else if (const BuiltinType* BTy = ArgTy->getAs<BuiltinType>()) {
              if (BTy->getKind() == BuiltinType::Float)
                ArgTy = Context.DoubleTy;
            }
          }

          ArgTys.push_back(ArgTy);
        }

        llvm::SmallVector<QualType, 4> Exceptions;
        Exceptions.reserve(FTI.NumExceptions);
        for (unsigned ei = 0, ee = FTI.NumExceptions; ei != ee; ++ei) {
          // FIXME: Preserve type source info.
          QualType ET = GetTypeFromParser(FTI.Exceptions[ei].Ty);
          // Check that the type is valid for an exception spec, and drop it if
          // not.
          if (!CheckSpecifiedExceptionType(ET, FTI.Exceptions[ei].Range))
            Exceptions.push_back(ET);
        }

        T = Context.getFunctionType(T, ArgTys.data(), ArgTys.size(),
                                    FTI.isVariadic, FTI.TypeQuals,
                                    FTI.hasExceptionSpec,
                                    FTI.hasAnyExceptionSpec,
                                    Exceptions.size(), Exceptions.data(),
                                    FunctionType::ExtInfo());
      }

      // For GCC compatibility, we allow attributes that apply only to
      // function types to be placed on a function's return type
      // instead (as long as that type doesn't happen to be function
      // or function-pointer itself).
      ProcessDelayedFnAttrs(*this, T, FnAttrsFromPreviousChunk);

      break;
    }
    case DeclaratorChunk::MemberPointer:
      // The scope spec must refer to a class, or be dependent.
      CXXScopeSpec &SS = DeclType.Mem.Scope();
      QualType ClsType;
      if (SS.isInvalid()) {
        // Avoid emitting extra errors if we already errored on the scope.
        D.setInvalidType(true);
      } else if (isDependentScopeSpecifier(SS) ||
                 dyn_cast_or_null<CXXRecordDecl>(computeDeclContext(SS))) {
        NestedNameSpecifier *NNS
          = static_cast<NestedNameSpecifier*>(SS.getScopeRep());
        NestedNameSpecifier *NNSPrefix = NNS->getPrefix();
        switch (NNS->getKind()) {
        case NestedNameSpecifier::Identifier:
          ClsType = Context.getDependentNameType(ETK_None, NNSPrefix,
                                                 NNS->getAsIdentifier());
          break;

        case NestedNameSpecifier::Namespace:
        case NestedNameSpecifier::Global:
          llvm_unreachable("Nested-name-specifier must name a type");
          break;

        case NestedNameSpecifier::TypeSpec:
        case NestedNameSpecifier::TypeSpecWithTemplate:
          ClsType = QualType(NNS->getAsType(), 0);
          // Note: if NNS is dependent, then its prefix (if any) is already
          // included in ClsType; this does not hold if the NNS is
          // nondependent: in this case (if there is indeed a prefix)
          // ClsType needs to be wrapped into an elaborated type.
          if (NNSPrefix && !NNS->isDependent())
            ClsType = Context.getElaboratedType(ETK_None, NNSPrefix, ClsType);
          break;
        }
      } else {
        Diag(DeclType.Mem.Scope().getBeginLoc(),
             diag::err_illegal_decl_mempointer_in_nonclass)
          << (D.getIdentifier() ? D.getIdentifier()->getName() : "type name")
          << DeclType.Mem.Scope().getRange();
        D.setInvalidType(true);
      }

      if (!ClsType.isNull())
        T = BuildMemberPointerType(T, ClsType, DeclType.Loc, D.getIdentifier());
      if (T.isNull()) {
        T = Context.IntTy;
        D.setInvalidType(true);
      } else if (DeclType.Mem.TypeQuals) {
        T = BuildQualifiedType(T, DeclType.Loc, DeclType.Mem.TypeQuals);
      }
      break;
    }

    if (T.isNull()) {
      D.setInvalidType(true);
      T = Context.IntTy;
    }

    DiagnoseDelayedFnAttrs(*this, FnAttrsFromPreviousChunk);

    // See if there are any attributes on this declarator chunk.
    if (const AttributeList *AL = DeclType.getAttrs())
      ProcessTypeAttributeList(*this, T, false, AL, FnAttrsFromPreviousChunk);
  }

  if (getLangOptions().CPlusPlus && T->isFunctionType()) {
    const FunctionProtoType *FnTy = T->getAs<FunctionProtoType>();
    assert(FnTy && "Why oh why is there not a FunctionProtoType here?");

    // C++ 8.3.5p4: A cv-qualifier-seq shall only be part of the function type
    // for a nonstatic member function, the function type to which a pointer
    // to member refers, or the top-level function type of a function typedef
    // declaration.
    bool FreeFunction = (D.getContext() != Declarator::MemberContext &&
        (!D.getCXXScopeSpec().isSet() ||
         !computeDeclContext(D.getCXXScopeSpec(), /*FIXME:*/true)->isRecord()));
    if (FnTy->getTypeQuals() != 0 &&
        D.getDeclSpec().getStorageClassSpec() != DeclSpec::SCS_typedef &&
        (FreeFunction ||
         D.getDeclSpec().getStorageClassSpec() == DeclSpec::SCS_static)) {
      if (D.isFunctionDeclarator())
        Diag(D.getIdentifierLoc(), diag::err_invalid_qualified_function_type);
      else
        Diag(D.getIdentifierLoc(),
             diag::err_invalid_qualified_typedef_function_type_use)
          << FreeFunction;

      // Strip the cv-quals from the type.
      T = Context.getFunctionType(FnTy->getResultType(), FnTy->arg_type_begin(),
                                  FnTy->getNumArgs(), FnTy->isVariadic(), 0, 
                                  false, false, 0, 0, FunctionType::ExtInfo());
    }
  }

  // If there's a constexpr specifier, treat it as a top-level const.
  if (D.getDeclSpec().isConstexprSpecified()) {
    T.addConst();
  }

  // Process any function attributes we might have delayed from the
  // declaration-specifiers.
  ProcessDelayedFnAttrs(*this, T, FnAttrsFromDeclSpec);

  // If there were any type attributes applied to the decl itself, not
  // the type, apply them to the result type.  But don't do this for
  // block-literal expressions, which are parsed wierdly.
  if (D.getContext() != Declarator::BlockLiteralContext)
    if (const AttributeList *Attrs = D.getAttributes())
      ProcessTypeAttributeList(*this, T, false, Attrs,
                               FnAttrsFromPreviousChunk);

  DiagnoseDelayedFnAttrs(*this, FnAttrsFromPreviousChunk);

  if (T.isNull())
    return Context.getNullTypeSourceInfo();
  else if (D.isInvalidType())
    return Context.getTrivialTypeSourceInfo(T);
  return GetTypeSourceInfoForDeclarator(D, T, ReturnTypeInfo);
}

namespace {
  class TypeSpecLocFiller : public TypeLocVisitor<TypeSpecLocFiller> {
    const DeclSpec &DS;

  public:
    TypeSpecLocFiller(const DeclSpec &DS) : DS(DS) {}

    void VisitQualifiedTypeLoc(QualifiedTypeLoc TL) {
      Visit(TL.getUnqualifiedLoc());
    }
    void VisitTypedefTypeLoc(TypedefTypeLoc TL) {
      TL.setNameLoc(DS.getTypeSpecTypeLoc());
    }
    void VisitObjCInterfaceTypeLoc(ObjCInterfaceTypeLoc TL) {
      TL.setNameLoc(DS.getTypeSpecTypeLoc());
    }
    void VisitObjCObjectTypeLoc(ObjCObjectTypeLoc TL) {
      // Handle the base type, which might not have been written explicitly.
      if (DS.getTypeSpecType() == DeclSpec::TST_unspecified) {
        TL.setHasBaseTypeAsWritten(false);
        TL.getBaseLoc().initialize(SourceLocation());
      } else {
        TL.setHasBaseTypeAsWritten(true);
        Visit(TL.getBaseLoc());
      }

      // Protocol qualifiers.
      if (DS.getProtocolQualifiers()) {
        assert(TL.getNumProtocols() > 0);
        assert(TL.getNumProtocols() == DS.getNumProtocolQualifiers());
        TL.setLAngleLoc(DS.getProtocolLAngleLoc());
        TL.setRAngleLoc(DS.getSourceRange().getEnd());
        for (unsigned i = 0, e = DS.getNumProtocolQualifiers(); i != e; ++i)
          TL.setProtocolLoc(i, DS.getProtocolLocs()[i]);
      } else {
        assert(TL.getNumProtocols() == 0);
        TL.setLAngleLoc(SourceLocation());
        TL.setRAngleLoc(SourceLocation());
      }
    }
    void VisitObjCObjectPointerTypeLoc(ObjCObjectPointerTypeLoc TL) {
      TL.setStarLoc(SourceLocation());
      Visit(TL.getPointeeLoc());
    }
    void VisitTemplateSpecializationTypeLoc(TemplateSpecializationTypeLoc TL) {
      TypeSourceInfo *TInfo = 0;
      Sema::GetTypeFromParser(DS.getTypeRep(), &TInfo);

      // If we got no declarator info from previous Sema routines,
      // just fill with the typespec loc.
      if (!TInfo) {
        TL.initialize(DS.getTypeSpecTypeLoc());
        return;
      }

      TypeLoc OldTL = TInfo->getTypeLoc();
      if (TInfo->getType()->getAs<ElaboratedType>()) {
        ElaboratedTypeLoc ElabTL = cast<ElaboratedTypeLoc>(OldTL);
        TemplateSpecializationTypeLoc NamedTL =
          cast<TemplateSpecializationTypeLoc>(ElabTL.getNamedTypeLoc());
        TL.copy(NamedTL);
      }
      else
        TL.copy(cast<TemplateSpecializationTypeLoc>(OldTL));
    }
    void VisitTypeOfExprTypeLoc(TypeOfExprTypeLoc TL) {
      assert(DS.getTypeSpecType() == DeclSpec::TST_typeofExpr);
      TL.setTypeofLoc(DS.getTypeSpecTypeLoc());
      TL.setParensRange(DS.getTypeofParensRange());
    }
    void VisitTypeOfTypeLoc(TypeOfTypeLoc TL) {
      assert(DS.getTypeSpecType() == DeclSpec::TST_typeofType);
      TL.setTypeofLoc(DS.getTypeSpecTypeLoc());
      TL.setParensRange(DS.getTypeofParensRange());
      assert(DS.getTypeRep());
      TypeSourceInfo *TInfo = 0;
      Sema::GetTypeFromParser(DS.getTypeRep(), &TInfo);
      TL.setUnderlyingTInfo(TInfo);
    }
    void VisitBuiltinTypeLoc(BuiltinTypeLoc TL) {
      // By default, use the source location of the type specifier.
      TL.setBuiltinLoc(DS.getTypeSpecTypeLoc());
      if (TL.needsExtraLocalData()) {
        // Set info for the written builtin specifiers.
        TL.getWrittenBuiltinSpecs() = DS.getWrittenBuiltinSpecs();
        // Try to have a meaningful source location.
        if (TL.getWrittenSignSpec() != TSS_unspecified)
          // Sign spec loc overrides the others (e.g., 'unsigned long').
          TL.setBuiltinLoc(DS.getTypeSpecSignLoc());
        else if (TL.getWrittenWidthSpec() != TSW_unspecified)
          // Width spec loc overrides type spec loc (e.g., 'short int').
          TL.setBuiltinLoc(DS.getTypeSpecWidthLoc());
      }
    }
    void VisitElaboratedTypeLoc(ElaboratedTypeLoc TL) {
      ElaboratedTypeKeyword Keyword
        = TypeWithKeyword::getKeywordForTypeSpec(DS.getTypeSpecType());
      if (Keyword == ETK_Typename) {
        TypeSourceInfo *TInfo = 0;
        Sema::GetTypeFromParser(DS.getTypeRep(), &TInfo);
        if (TInfo) {
          TL.copy(cast<ElaboratedTypeLoc>(TInfo->getTypeLoc()));
          return;
        }
      }
      TL.setKeywordLoc(Keyword != ETK_None
                       ? DS.getTypeSpecTypeLoc()
                       : SourceLocation());
      const CXXScopeSpec& SS = DS.getTypeSpecScope();
      TL.setQualifierRange(SS.isEmpty() ? SourceRange(): SS.getRange());
      Visit(TL.getNextTypeLoc().getUnqualifiedLoc());
    }
    void VisitDependentNameTypeLoc(DependentNameTypeLoc TL) {
      ElaboratedTypeKeyword Keyword
        = TypeWithKeyword::getKeywordForTypeSpec(DS.getTypeSpecType());
      if (Keyword == ETK_Typename) {
        TypeSourceInfo *TInfo = 0;
        Sema::GetTypeFromParser(DS.getTypeRep(), &TInfo);
        if (TInfo) {
          TL.copy(cast<DependentNameTypeLoc>(TInfo->getTypeLoc()));
          return;
        }
      }
      TL.setKeywordLoc(Keyword != ETK_None
                       ? DS.getTypeSpecTypeLoc()
                       : SourceLocation());
      const CXXScopeSpec& SS = DS.getTypeSpecScope();
      TL.setQualifierRange(SS.isEmpty() ? SourceRange() : SS.getRange());
      // FIXME: load appropriate source location.
      TL.setNameLoc(DS.getTypeSpecTypeLoc());
    }
    void VisitDependentTemplateSpecializationTypeLoc(
                                 DependentTemplateSpecializationTypeLoc TL) {
      ElaboratedTypeKeyword Keyword
        = TypeWithKeyword::getKeywordForTypeSpec(DS.getTypeSpecType());
      if (Keyword == ETK_Typename) {
        TypeSourceInfo *TInfo = 0;
        Sema::GetTypeFromParser(DS.getTypeRep(), &TInfo);
        if (TInfo) {
          TL.copy(cast<DependentTemplateSpecializationTypeLoc>(
                    TInfo->getTypeLoc()));
          return;
        }
      }
      TL.initializeLocal(SourceLocation());
      TL.setKeywordLoc(Keyword != ETK_None
                       ? DS.getTypeSpecTypeLoc()
                       : SourceLocation());
      const CXXScopeSpec& SS = DS.getTypeSpecScope();
      TL.setQualifierRange(SS.isEmpty() ? SourceRange() : SS.getRange());
      // FIXME: load appropriate source location.
      TL.setNameLoc(DS.getTypeSpecTypeLoc());
    }

    void VisitTypeLoc(TypeLoc TL) {
      // FIXME: add other typespec types and change this to an assert.
      TL.initialize(DS.getTypeSpecTypeLoc());
    }
  };

  class DeclaratorLocFiller : public TypeLocVisitor<DeclaratorLocFiller> {
    const DeclaratorChunk &Chunk;

  public:
    DeclaratorLocFiller(const DeclaratorChunk &Chunk) : Chunk(Chunk) {}

    void VisitQualifiedTypeLoc(QualifiedTypeLoc TL) {
      llvm_unreachable("qualified type locs not expected here!");
    }

    void VisitBlockPointerTypeLoc(BlockPointerTypeLoc TL) {
      assert(Chunk.Kind == DeclaratorChunk::BlockPointer);
      TL.setCaretLoc(Chunk.Loc);
    }
    void VisitPointerTypeLoc(PointerTypeLoc TL) {
      assert(Chunk.Kind == DeclaratorChunk::Pointer);
      TL.setStarLoc(Chunk.Loc);
    }
    void VisitObjCObjectPointerTypeLoc(ObjCObjectPointerTypeLoc TL) {
      assert(Chunk.Kind == DeclaratorChunk::Pointer);
      TL.setStarLoc(Chunk.Loc);
    }
    void VisitMemberPointerTypeLoc(MemberPointerTypeLoc TL) {
      assert(Chunk.Kind == DeclaratorChunk::MemberPointer);
      TL.setStarLoc(Chunk.Loc);
      // FIXME: nested name specifier
    }
    void VisitLValueReferenceTypeLoc(LValueReferenceTypeLoc TL) {
      assert(Chunk.Kind == DeclaratorChunk::Reference);
      // 'Amp' is misleading: this might have been originally
      /// spelled with AmpAmp.
      TL.setAmpLoc(Chunk.Loc);
    }
    void VisitRValueReferenceTypeLoc(RValueReferenceTypeLoc TL) {
      assert(Chunk.Kind == DeclaratorChunk::Reference);
      assert(!Chunk.Ref.LValueRef);
      TL.setAmpAmpLoc(Chunk.Loc);
    }
    void VisitArrayTypeLoc(ArrayTypeLoc TL) {
      assert(Chunk.Kind == DeclaratorChunk::Array);
      TL.setLBracketLoc(Chunk.Loc);
      TL.setRBracketLoc(Chunk.EndLoc);
      TL.setSizeExpr(static_cast<Expr*>(Chunk.Arr.NumElts));
    }
    void VisitFunctionTypeLoc(FunctionTypeLoc TL) {
      assert(Chunk.Kind == DeclaratorChunk::Function);
      TL.setLParenLoc(Chunk.Loc);
      TL.setRParenLoc(Chunk.EndLoc);

      const DeclaratorChunk::FunctionTypeInfo &FTI = Chunk.Fun;
      for (unsigned i = 0, e = TL.getNumArgs(), tpi = 0; i != e; ++i) {
        ParmVarDecl *Param = FTI.ArgInfo[i].Param.getAs<ParmVarDecl>();
        TL.setArg(tpi++, Param);
      }
      // FIXME: exception specs
    }

    void VisitTypeLoc(TypeLoc TL) {
      llvm_unreachable("unsupported TypeLoc kind in declarator!");
    }
  };
}

/// \brief Create and instantiate a TypeSourceInfo with type source information.
///
/// \param T QualType referring to the type as written in source code.
///
/// \param ReturnTypeInfo For declarators whose return type does not show
/// up in the normal place in the declaration specifiers (such as a C++
/// conversion function), this pointer will refer to a type source information
/// for that return type.
TypeSourceInfo *
Sema::GetTypeSourceInfoForDeclarator(Declarator &D, QualType T,
                                     TypeSourceInfo *ReturnTypeInfo) {
  TypeSourceInfo *TInfo = Context.CreateTypeSourceInfo(T);
  UnqualTypeLoc CurrTL = TInfo->getTypeLoc().getUnqualifiedLoc();

  for (unsigned i = 0, e = D.getNumTypeObjects(); i != e; ++i) {
    DeclaratorLocFiller(D.getTypeObject(i)).Visit(CurrTL);
    CurrTL = CurrTL.getNextTypeLoc().getUnqualifiedLoc();
  }
  
  TypeSpecLocFiller(D.getDeclSpec()).Visit(CurrTL);
  
  // We have source information for the return type that was not in the
  // declaration specifiers; copy that information into the current type
  // location so that it will be retained. This occurs, for example, with 
  // a C++ conversion function, where the return type occurs within the
  // declarator-id rather than in the declaration specifiers.
  if (ReturnTypeInfo && D.getDeclSpec().getTypeSpecType() == TST_unspecified) {
    TypeLoc TL = ReturnTypeInfo->getTypeLoc();
    assert(TL.getFullDataSize() == CurrTL.getFullDataSize());
    memcpy(CurrTL.getOpaqueData(), TL.getOpaqueData(), TL.getFullDataSize());
  }
      
  return TInfo;
}

/// \brief Create a LocInfoType to hold the given QualType and TypeSourceInfo.
QualType Sema::CreateLocInfoType(QualType T, TypeSourceInfo *TInfo) {
  // FIXME: LocInfoTypes are "transient", only needed for passing to/from Parser
  // and Sema during declaration parsing. Try deallocating/caching them when
  // it's appropriate, instead of allocating them and keeping them around.
  LocInfoType *LocT = (LocInfoType*)BumpAlloc.Allocate(sizeof(LocInfoType), 8);
  new (LocT) LocInfoType(T, TInfo);
  assert(LocT->getTypeClass() != T->getTypeClass() &&
         "LocInfoType's TypeClass conflicts with an existing Type class");
  return QualType(LocT, 0);
}

void LocInfoType::getAsStringInternal(std::string &Str,
                                      const PrintingPolicy &Policy) const {
  assert(false && "LocInfoType leaked into the type system; an opaque TypeTy*"
         " was used directly instead of getting the QualType through"
         " GetTypeFromParser");
}

Sema::TypeResult Sema::ActOnTypeName(Scope *S, Declarator &D) {
  // C99 6.7.6: Type names have no identifier.  This is already validated by
  // the parser.
  assert(D.getIdentifier() == 0 && "Type name should have no identifier!");

  TagDecl *OwnedTag = 0;
  TypeSourceInfo *TInfo = GetTypeForDeclarator(D, S, &OwnedTag);
  QualType T = TInfo->getType();
  if (D.isInvalidType())
    return true;

  if (getLangOptions().CPlusPlus) {
    // Check that there are no default arguments (C++ only).
    CheckExtraCXXDefaultArguments(D);

    // C++0x [dcl.type]p3:
    //   A type-specifier-seq shall not define a class or enumeration
    //   unless it appears in the type-id of an alias-declaration
    //   (7.1.3).
    if (OwnedTag && OwnedTag->isDefinition())
      Diag(OwnedTag->getLocation(), diag::err_type_defined_in_type_specifier)
        << Context.getTypeDeclType(OwnedTag);
  }

  T = CreateLocInfoType(T, TInfo);
  return T.getAsOpaquePtr();
}



//===----------------------------------------------------------------------===//
// Type Attribute Processing
//===----------------------------------------------------------------------===//

/// HandleAddressSpaceTypeAttribute - Process an address_space attribute on the
/// specified type.  The attribute contains 1 argument, the id of the address
/// space for the type.
static void HandleAddressSpaceTypeAttribute(QualType &Type,
                                            const AttributeList &Attr, Sema &S){

  // If this type is already address space qualified, reject it.
  // Clause 6.7.3 - Type qualifiers: "No type shall be qualified by qualifiers
  // for two or more different address spaces."
  if (Type.getAddressSpace()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_address_multiple_qualifiers);
    Attr.setInvalid();
    return;
  }

  // Check the attribute arguments.
  if (Attr.getNumArgs() != 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    Attr.setInvalid();
    return;
  }
  Expr *ASArgExpr = static_cast<Expr *>(Attr.getArg(0));
  llvm::APSInt addrSpace(32);
  if (ASArgExpr->isTypeDependent() || ASArgExpr->isValueDependent() ||
      !ASArgExpr->isIntegerConstantExpr(addrSpace, S.Context)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_address_space_not_int)
      << ASArgExpr->getSourceRange();
    Attr.setInvalid();
    return;
  }

  // Bounds checking.
  if (addrSpace.isSigned()) {
    if (addrSpace.isNegative()) {
      S.Diag(Attr.getLoc(), diag::err_attribute_address_space_negative)
        << ASArgExpr->getSourceRange();
      Attr.setInvalid();
      return;
    }
    addrSpace.setIsSigned(false);
  }
  llvm::APSInt max(addrSpace.getBitWidth());
  max = Qualifiers::MaxAddressSpace;
  if (addrSpace > max) {
    S.Diag(Attr.getLoc(), diag::err_attribute_address_space_too_high)
      << Qualifiers::MaxAddressSpace << ASArgExpr->getSourceRange();
    Attr.setInvalid();
    return;
  }

  unsigned ASIdx = static_cast<unsigned>(addrSpace.getZExtValue());
  Type = S.Context.getAddrSpaceQualType(Type, ASIdx);
}

/// HandleObjCGCTypeAttribute - Process an objc's gc attribute on the
/// specified type.  The attribute contains 1 argument, weak or strong.
static void HandleObjCGCTypeAttribute(QualType &Type,
                                      const AttributeList &Attr, Sema &S) {
  if (Type.getObjCGCAttr() != Qualifiers::GCNone) {
    S.Diag(Attr.getLoc(), diag::err_attribute_multiple_objc_gc);
    Attr.setInvalid();
    return;
  }

  // Check the attribute arguments.
  if (!Attr.getParameterName()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string)
      << "objc_gc" << 1;
    Attr.setInvalid();
    return;
  }
  Qualifiers::GC GCAttr;
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    Attr.setInvalid();
    return;
  }
  if (Attr.getParameterName()->isStr("weak"))
    GCAttr = Qualifiers::Weak;
  else if (Attr.getParameterName()->isStr("strong"))
    GCAttr = Qualifiers::Strong;
  else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_type_not_supported)
      << "objc_gc" << Attr.getParameterName();
    Attr.setInvalid();
    return;
  }

  Type = S.Context.getObjCGCQualType(Type, GCAttr);
}

static QualType GetResultType(QualType T) {
  if (const PointerType *PT = T->getAs<PointerType>())
    T = PT->getPointeeType();
  else if (const BlockPointerType *BT = T->getAs<BlockPointerType>())
    T = BT->getPointeeType();
  return T->getAs<FunctionType>()->getResultType();
}

/// Process an individual function attribute.  Returns true if the
/// attribute does not make sense to apply to this type.
bool ProcessFnAttr(Sema &S, QualType &Type, const AttributeList &Attr) {
  if (Attr.getKind() == AttributeList::AT_noreturn) {
    // Complain immediately if the arg count is wrong.
    if (Attr.getNumArgs() != 0) {
      S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
      Attr.setInvalid();
      return false;
    }

    // Delay if this is not a function or pointer to block.
    if (!Type->isFunctionPointerType()
        && !Type->isBlockPointerType()
        && !Type->isFunctionType())
      return true;
    
    if (!GetResultType(Type)->isVoidType()) {
      S.Diag(Attr.getLoc(), diag::warn_noreturn_function_has_nonvoid_result)
        << (Type->isBlockPointerType() ? /* blocks */ 1 : /* functions */ 0);
    }
    
    // Otherwise we can process right away.
    Type = S.Context.getNoReturnType(Type);
    return false;
  }

  if (Attr.getKind() == AttributeList::AT_regparm) {
    // The warning is emitted elsewhere
    if (Attr.getNumArgs() != 1) {
      return false;
    }

    // Delay if this is not a function or pointer to block.
    if (!Type->isFunctionPointerType()
        && !Type->isBlockPointerType()
        && !Type->isFunctionType())
      return true;

    // Otherwise we can process right away.
    Expr *NumParamsExpr = static_cast<Expr *>(Attr.getArg(0));
    llvm::APSInt NumParams(32);

    // The warning is emitted elsewhere
    if (NumParamsExpr->isTypeDependent() || NumParamsExpr->isValueDependent() ||
        !NumParamsExpr->isIntegerConstantExpr(NumParams, S.Context))
      return false;

    Type = S.Context.getRegParmType(Type, NumParams.getZExtValue());
    return false;
  }

  // Otherwise, a calling convention.
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 0;
    Attr.setInvalid();
    return false;
  }

  QualType T = Type;
  if (const PointerType *PT = Type->getAs<PointerType>())
    T = PT->getPointeeType();
  const FunctionType *Fn = T->getAs<FunctionType>();

  // Delay if the type didn't work out to a function.
  if (!Fn) return true;

  // TODO: diagnose uses of these conventions on the wrong target.
  CallingConv CC;
  switch (Attr.getKind()) {
  case AttributeList::AT_cdecl: CC = CC_C; break;
  case AttributeList::AT_fastcall: CC = CC_X86FastCall; break;
  case AttributeList::AT_stdcall: CC = CC_X86StdCall; break;
  case AttributeList::AT_thiscall: CC = CC_X86ThisCall; break;
  default: llvm_unreachable("unexpected attribute kind"); return false;
  }

  CallingConv CCOld = Fn->getCallConv();
  if (S.Context.getCanonicalCallConv(CC) ==
      S.Context.getCanonicalCallConv(CCOld)) {
    Attr.setInvalid();
    return false;
  }

  if (CCOld != CC_Default) {
    // Should we diagnose reapplications of the same convention?
    S.Diag(Attr.getLoc(), diag::err_attributes_are_not_compatible)
      << FunctionType::getNameForCallConv(CC)
      << FunctionType::getNameForCallConv(CCOld);
    Attr.setInvalid();
    return false;
  }

  // Diagnose the use of X86 fastcall on varargs or unprototyped functions.
  if (CC == CC_X86FastCall) {
    if (isa<FunctionNoProtoType>(Fn)) {
      S.Diag(Attr.getLoc(), diag::err_cconv_knr)
        << FunctionType::getNameForCallConv(CC);
      Attr.setInvalid();
      return false;
    }

    const FunctionProtoType *FnP = cast<FunctionProtoType>(Fn);
    if (FnP->isVariadic()) {
      S.Diag(Attr.getLoc(), diag::err_cconv_varargs)
        << FunctionType::getNameForCallConv(CC);
      Attr.setInvalid();
      return false;
    }
  }

  Type = S.Context.getCallConvType(Type, CC);
  return false;
}

/// HandleVectorSizeAttribute - this attribute is only applicable to integral
/// and float scalars, although arrays, pointers, and function return values are
/// allowed in conjunction with this construct. Aggregates with this attribute
/// are invalid, even if they are of the same size as a corresponding scalar.
/// The raw attribute should contain precisely 1 argument, the vector size for
/// the variable, measured in bytes. If curType and rawAttr are well formed,
/// this routine will return a new vector type.
static void HandleVectorSizeAttr(QualType& CurType, const AttributeList &Attr,
                                 Sema &S) {
  // Check the attribute arugments.
  if (Attr.getNumArgs() != 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    Attr.setInvalid();
    return;
  }
  Expr *sizeExpr = static_cast<Expr *>(Attr.getArg(0));
  llvm::APSInt vecSize(32);
  if (sizeExpr->isTypeDependent() || sizeExpr->isValueDependent() ||
      !sizeExpr->isIntegerConstantExpr(vecSize, S.Context)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_not_int)
      << "vector_size" << sizeExpr->getSourceRange();
    Attr.setInvalid();
    return;
  }
  // the base type must be integer or float, and can't already be a vector.
  if (!CurType->isIntegerType() && !CurType->isRealFloatingType()) {
    S.Diag(Attr.getLoc(), diag::err_attribute_invalid_vector_type) << CurType;
    Attr.setInvalid();
    return;
  }
  unsigned typeSize = static_cast<unsigned>(S.Context.getTypeSize(CurType));
  // vecSize is specified in bytes - convert to bits.
  unsigned vectorSize = static_cast<unsigned>(vecSize.getZExtValue() * 8);

  // the vector size needs to be an integral multiple of the type size.
  if (vectorSize % typeSize) {
    S.Diag(Attr.getLoc(), diag::err_attribute_invalid_size)
      << sizeExpr->getSourceRange();
    Attr.setInvalid();
    return;
  }
  if (vectorSize == 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_zero_size)
      << sizeExpr->getSourceRange();
    Attr.setInvalid();
    return;
  }

  // Success! Instantiate the vector type, the number of elements is > 0, and
  // not required to be a power of 2, unlike GCC.
  CurType = S.Context.getVectorType(CurType, vectorSize/typeSize,
                                    VectorType::NotAltiVec);
}

void ProcessTypeAttributeList(Sema &S, QualType &Result,
                              bool IsDeclSpec, const AttributeList *AL,
                              DelayedAttributeSet &FnAttrs) {
  // Scan through and apply attributes to this type where it makes sense.  Some
  // attributes (such as __address_space__, __vector_size__, etc) apply to the
  // type, but others can be present in the type specifiers even though they
  // apply to the decl.  Here we apply type attributes and ignore the rest.
  for (; AL; AL = AL->getNext()) {
    // Skip attributes that were marked to be invalid.
    if (AL->isInvalid())
      continue;

    // If this is an attribute we can handle, do so now,
    // otherwise, add it to the FnAttrs list for rechaining.
    switch (AL->getKind()) {
    default: break;

    case AttributeList::AT_address_space:
      HandleAddressSpaceTypeAttribute(Result, *AL, S);
      break;
    case AttributeList::AT_objc_gc:
      HandleObjCGCTypeAttribute(Result, *AL, S);
      break;
    case AttributeList::AT_vector_size:
      HandleVectorSizeAttr(Result, *AL, S);
      break;

    case AttributeList::AT_noreturn:
    case AttributeList::AT_cdecl:
    case AttributeList::AT_fastcall:
    case AttributeList::AT_stdcall:
    case AttributeList::AT_thiscall:
    case AttributeList::AT_regparm:
      // Don't process these on the DeclSpec.
      if (IsDeclSpec ||
          ProcessFnAttr(S, Result, *AL))
        FnAttrs.push_back(DelayedAttribute(AL, Result));
      break;
    }
  }
}

/// @brief Ensure that the type T is a complete type.
///
/// This routine checks whether the type @p T is complete in any
/// context where a complete type is required. If @p T is a complete
/// type, returns false. If @p T is a class template specialization,
/// this routine then attempts to perform class template
/// instantiation. If instantiation fails, or if @p T is incomplete
/// and cannot be completed, issues the diagnostic @p diag (giving it
/// the type @p T) and returns true.
///
/// @param Loc  The location in the source that the incomplete type
/// diagnostic should refer to.
///
/// @param T  The type that this routine is examining for completeness.
///
/// @param PD The partial diagnostic that will be printed out if T is not a
/// complete type.
///
/// @returns @c true if @p T is incomplete and a diagnostic was emitted,
/// @c false otherwise.
bool Sema::RequireCompleteType(SourceLocation Loc, QualType T,
                               const PartialDiagnostic &PD,
                               std::pair<SourceLocation, 
                                         PartialDiagnostic> Note) {
  unsigned diag = PD.getDiagID();

  // FIXME: Add this assertion to make sure we always get instantiation points.
  //  assert(!Loc.isInvalid() && "Invalid location in RequireCompleteType");
  // FIXME: Add this assertion to help us flush out problems with
  // checking for dependent types and type-dependent expressions.
  //
  //  assert(!T->isDependentType() &&
  //         "Can't ask whether a dependent type is complete");

  // If we have a complete type, we're done.
  if (!T->isIncompleteType())
    return false;

  // If we have a class template specialization or a class member of a
  // class template specialization, or an array with known size of such,
  // try to instantiate it.
  QualType MaybeTemplate = T;
  if (const ConstantArrayType *Array = Context.getAsConstantArrayType(T))
    MaybeTemplate = Array->getElementType();
  if (const RecordType *Record = MaybeTemplate->getAs<RecordType>()) {
    if (ClassTemplateSpecializationDecl *ClassTemplateSpec
          = dyn_cast<ClassTemplateSpecializationDecl>(Record->getDecl())) {
      if (ClassTemplateSpec->getSpecializationKind() == TSK_Undeclared)
        return InstantiateClassTemplateSpecialization(Loc, ClassTemplateSpec,
                                                      TSK_ImplicitInstantiation,
                                                      /*Complain=*/diag != 0);
    } else if (CXXRecordDecl *Rec
                 = dyn_cast<CXXRecordDecl>(Record->getDecl())) {
      if (CXXRecordDecl *Pattern = Rec->getInstantiatedFromMemberClass()) {
        MemberSpecializationInfo *MSInfo = Rec->getMemberSpecializationInfo();
        assert(MSInfo && "Missing member specialization information?");
        // This record was instantiated from a class within a template.
        if (MSInfo->getTemplateSpecializationKind() 
                                               != TSK_ExplicitSpecialization)
          return InstantiateClass(Loc, Rec, Pattern,
                                  getTemplateInstantiationArgs(Rec),
                                  TSK_ImplicitInstantiation,
                                  /*Complain=*/diag != 0);
      }
    }
  }

  if (diag == 0)
    return true;

  const TagType *Tag = 0;
  if (const RecordType *Record = T->getAs<RecordType>())
    Tag = Record;
  else if (const EnumType *Enum = T->getAs<EnumType>())
    Tag = Enum;

  // Avoid diagnosing invalid decls as incomplete.
  if (Tag && Tag->getDecl()->isInvalidDecl())
    return true;

  // We have an incomplete type. Produce a diagnostic.
  Diag(Loc, PD) << T;

  // If we have a note, produce it.
  if (!Note.first.isInvalid())
    Diag(Note.first, Note.second);
    
  // If the type was a forward declaration of a class/struct/union
  // type, produce a note.
  if (Tag && !Tag->getDecl()->isInvalidDecl())
    Diag(Tag->getDecl()->getLocation(),
         Tag->isBeingDefined() ? diag::note_type_being_defined
                               : diag::note_forward_declaration)
        << QualType(Tag, 0);

  return true;
}

bool Sema::RequireCompleteType(SourceLocation Loc, QualType T,
                               const PartialDiagnostic &PD) {
  return RequireCompleteType(Loc, T, PD, 
                             std::make_pair(SourceLocation(), PDiag(0)));
}
  
bool Sema::RequireCompleteType(SourceLocation Loc, QualType T,
                               unsigned DiagID) {
  return RequireCompleteType(Loc, T, PDiag(DiagID),
                             std::make_pair(SourceLocation(), PDiag(0)));
}

/// \brief Retrieve a version of the type 'T' that is elaborated by Keyword
/// and qualified by the nested-name-specifier contained in SS.
QualType Sema::getElaboratedType(ElaboratedTypeKeyword Keyword,
                                 const CXXScopeSpec &SS, QualType T) {
  if (T.isNull())
    return T;
  NestedNameSpecifier *NNS;
  if (SS.isValid())
    NNS = static_cast<NestedNameSpecifier *>(SS.getScopeRep());
  else {
    if (Keyword == ETK_None)
      return T;
    NNS = 0;
  }
  return Context.getElaboratedType(Keyword, NNS, T);
}

QualType Sema::BuildTypeofExprType(Expr *E) {
  if (E->getType() == Context.OverloadTy) {
    // C++ [temp.arg.explicit]p3 allows us to resolve a template-id to a 
    // function template specialization wherever deduction cannot occur.
    if (FunctionDecl *Specialization
        = ResolveSingleFunctionTemplateSpecialization(E)) {
      // The access doesn't really matter in this case.
      DeclAccessPair Found = DeclAccessPair::make(Specialization,
                                                  Specialization->getAccess());
      E = FixOverloadedFunctionReference(E, Found, Specialization);
      if (!E)
        return QualType();      
    } else {
      Diag(E->getLocStart(),
           diag::err_cannot_determine_declared_type_of_overloaded_function)
        << false << E->getSourceRange();
      return QualType();
    }
  }
  
  return Context.getTypeOfExprType(E);
}

QualType Sema::BuildDecltypeType(Expr *E) {
  if (E->getType() == Context.OverloadTy) {
    // C++ [temp.arg.explicit]p3 allows us to resolve a template-id to a 
    // function template specialization wherever deduction cannot occur.
    if (FunctionDecl *Specialization
          = ResolveSingleFunctionTemplateSpecialization(E)) {
      // The access doesn't really matter in this case.
      DeclAccessPair Found = DeclAccessPair::make(Specialization,
                                                  Specialization->getAccess());
      E = FixOverloadedFunctionReference(E, Found, Specialization);
      if (!E)
        return QualType();      
    } else {
      Diag(E->getLocStart(),
           diag::err_cannot_determine_declared_type_of_overloaded_function)
        << true << E->getSourceRange();
      return QualType();
    }
  }
  
  return Context.getDecltypeType(E);
}
