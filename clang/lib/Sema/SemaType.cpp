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

#include "Sema.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/Parse/DeclSpec.h"
using namespace clang;

/// \brief Perform adjustment on the parameter type of a function.
///
/// This routine adjusts the given parameter type @p T to the actual
/// parameter type used by semantic analysis (C99 6.7.5.3p[7,8], 
/// C++ [dcl.fct]p3). The adjusted parameter type is returned. 
QualType Sema::adjustParameterType(QualType T) {
  // C99 6.7.5.3p7:
  if (T->isArrayType()) {
    // C99 6.7.5.3p7:
    //   A declaration of a parameter as "array of type" shall be
    //   adjusted to "qualified pointer to type", where the type
    //   qualifiers (if any) are those specified within the [ and ] of
    //   the array type derivation.
    return Context.getArrayDecayedType(T);
  } else if (T->isFunctionType())
    // C99 6.7.5.3p8:
    //   A declaration of a parameter as "function returning type"
    //   shall be adjusted to "pointer to function returning type", as
    //   in 6.3.2.1.
    return Context.getPointerType(T);

  return T;
}

/// \brief Convert the specified declspec to the appropriate type
/// object.
/// \param DS  the declaration specifiers
/// \param DeclLoc The location of the declarator identifier or invalid if none.
/// \returns The type described by the declaration specifiers, or NULL
/// if there was an error.
QualType Sema::ConvertDeclSpecToType(const DeclSpec &DS,
                                     SourceLocation DeclLoc) {
  // FIXME: Should move the logic from DeclSpec::Finish to here for validity
  // checking.
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
      Diag(DS.getTypeSpecSignLoc(), diag::ext_invalid_sign_spec)
        << DS.getSpecifierName(DS.getTypeSpecType());
      Result = Context.getSignedWCharType();
    } else {
      assert(DS.getTypeSpecSign() == DeclSpec::TSS_unsigned &&
        "Unknown TSS value");
      Diag(DS.getTypeSpecSignLoc(), diag::ext_invalid_sign_spec)
        << DS.getSpecifierName(DS.getTypeSpecType());
      Result = Context.getUnsignedWCharType();
    }
    break;
  case DeclSpec::TST_unspecified:
    // "<proto1,proto2>" is an objc qualified ID with a missing id.
    if (DeclSpec::ProtocolQualifierListTy PQ = DS.getProtocolQualifiers()) {
      Result = Context.getObjCQualifiedIdType((ObjCProtocolDecl**)PQ,
                                              DS.getNumProtocolQualifiers());
      break;
    }
      
    // Unspecified typespec defaults to int in C90.  However, the C90 grammar
    // [C90 6.5] only allows a decl-spec if there was *some* type-specifier,
    // type-qualifier, or storage-class-specifier.  If not, emit an extwarn.
    // Note that the one exception to this is function definitions, which are
    // allowed to be completely missing a declspec.  This is handled in the
    // parser already though by it pretending to have seen an 'int' in this
    // case.
    if (getLangOptions().ImplicitInt) {
      // In C89 mode, we only warn if there is a completely missing declspec
      // when one is not allowed.
      if (DS.isEmpty()) {
        if (DeclLoc.isInvalid())
          DeclLoc = DS.getSourceRange().getBegin();
        Diag(DeclLoc, diag::warn_missing_declspec)
          << DS.getSourceRange()
        << CodeModificationHint::CreateInsertion(DS.getSourceRange().getBegin(),
                                                 "int");
      }
    } else if (!DS.hasTypeSpecifier()) {
      // C99 and C++ require a type specifier.  For example, C99 6.7.2p2 says:
      // "At least one type specifier shall be given in the declaration
      // specifiers in each declaration, and in the specifier-qualifier list in
      // each struct declaration and type name."
      // FIXME: Does Microsoft really have the implicit int extension in C++?
      if (DeclLoc.isInvalid())
        DeclLoc = DS.getSourceRange().getBegin();

      if (getLangOptions().CPlusPlus && !getLangOptions().Microsoft)
        Diag(DeclLoc, diag::err_missing_type_specifier)
          << DS.getSourceRange();
      else
        Diag(DeclLoc, diag::warn_missing_type_specifier)
          << DS.getSourceRange();

      // FIXME: If we could guarantee that the result would be
      // well-formed, it would be useful to have a code insertion hint
      // here. However, after emitting this warning/error, we often
      // emit other errors.
    }
      
    // FALL THROUGH.  
  case DeclSpec::TST_int: {
    if (DS.getTypeSpecSign() != DeclSpec::TSS_unsigned) {
      switch (DS.getTypeSpecWidth()) {
      case DeclSpec::TSW_unspecified: Result = Context.IntTy; break;
      case DeclSpec::TSW_short:       Result = Context.ShortTy; break;
      case DeclSpec::TSW_long:        Result = Context.LongTy; break;
      case DeclSpec::TSW_longlong:    Result = Context.LongLongTy; break;
      }
    } else {
      switch (DS.getTypeSpecWidth()) {
      case DeclSpec::TSW_unspecified: Result = Context.UnsignedIntTy; break;
      case DeclSpec::TSW_short:       Result = Context.UnsignedShortTy; break;
      case DeclSpec::TSW_long:        Result = Context.UnsignedLongTy; break;
      case DeclSpec::TSW_longlong:    Result =Context.UnsignedLongLongTy; break;
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
    assert(0 && "FIXME: GNU decimal extensions not supported yet!"); 
  case DeclSpec::TST_class:
  case DeclSpec::TST_enum:
  case DeclSpec::TST_union:
  case DeclSpec::TST_struct: {
    Decl *D = static_cast<Decl *>(DS.getTypeRep());
    assert(D && "Didn't get a decl for a class/enum/union/struct?");
    assert(DS.getTypeSpecWidth() == 0 && DS.getTypeSpecComplex() == 0 &&
           DS.getTypeSpecSign() == 0 &&
           "Can't handle qualifiers on typedef names yet!");
    // TypeQuals handled by caller.
    Result = Context.getTypeDeclType(cast<TypeDecl>(D));
    break;
  }    
  case DeclSpec::TST_typename: {
    assert(DS.getTypeSpecWidth() == 0 && DS.getTypeSpecComplex() == 0 &&
           DS.getTypeSpecSign() == 0 &&
           "Can't handle qualifiers on typedef names yet!");
    Result = QualType::getFromOpaquePtr(DS.getTypeRep());

    if (DeclSpec::ProtocolQualifierListTy PQ = DS.getProtocolQualifiers()) {
      // FIXME: Adding a TST_objcInterface clause doesn't seem ideal, so
      // we have this "hack" for now... 
      if (const ObjCInterfaceType *Interface = Result->getAsObjCInterfaceType())
        Result = Context.getObjCQualifiedInterfaceType(Interface->getDecl(),
                                                       (ObjCProtocolDecl**)PQ,
                                               DS.getNumProtocolQualifiers());
      else if (Result == Context.getObjCIdType())
        // id<protocol-list>
        Result = Context.getObjCQualifiedIdType((ObjCProtocolDecl**)PQ,
                                                DS.getNumProtocolQualifiers());
      else if (Result == Context.getObjCClassType()) {
        if (DeclLoc.isInvalid())
          DeclLoc = DS.getSourceRange().getBegin();
        // Class<protocol-list>
        Diag(DeclLoc, diag::err_qualified_class_unsupported)
          << DS.getSourceRange();
      } else {
        if (DeclLoc.isInvalid())
          DeclLoc = DS.getSourceRange().getBegin();
        Diag(DeclLoc, diag::err_invalid_protocol_qualifiers)
          << DS.getSourceRange();
      }
    }
    // TypeQuals handled by caller.
    break;
  }
  case DeclSpec::TST_typeofType:
    Result = QualType::getFromOpaquePtr(DS.getTypeRep());
    assert(!Result.isNull() && "Didn't get a type for typeof?");
    // TypeQuals handled by caller.
    Result = Context.getTypeOfType(Result);
    break;
  case DeclSpec::TST_typeofExpr: {
    Expr *E = static_cast<Expr *>(DS.getTypeRep());
    assert(E && "Didn't get an expression for typeof?");
    // TypeQuals handled by caller.
    Result = Context.getTypeOfExprType(E);
    break;
  }
  case DeclSpec::TST_error:
    return QualType();
  }
  
  // Handle complex types.
  if (DS.getTypeSpecComplex() == DeclSpec::TSC_complex) {
    if (getLangOptions().Freestanding)
      Diag(DS.getTypeSpecComplexLoc(), diag::ext_freestanding_complex);
    Result = Context.getComplexType(Result);
  }
  
  assert(DS.getTypeSpecComplex() != DeclSpec::TSC_imaginary &&
         "FIXME: imaginary types not supported yet!");
  
  // See if there are any attributes on the declspec that apply to the type (as
  // opposed to the decl).
  if (const AttributeList *AL = DS.getAttributes())
    ProcessTypeAttributeList(Result, AL);
    
  // Apply const/volatile/restrict qualifiers to T.
  if (unsigned TypeQuals = DS.getTypeQualifiers()) {

    // Enforce C99 6.7.3p2: "Types other than pointer types derived from object
    // or incomplete types shall not be restrict-qualified."  C++ also allows
    // restrict-qualified references.
    if (TypeQuals & QualType::Restrict) {
      if (Result->isPointerType() || Result->isReferenceType()) {
        QualType EltTy = Result->isPointerType() ? 
          Result->getAsPointerType()->getPointeeType() :
          Result->getAsReferenceType()->getPointeeType();
      
        // If we have a pointer or reference, the pointee must have an object
        // incomplete type.
        if (!EltTy->isIncompleteOrObjectType()) {
          Diag(DS.getRestrictSpecLoc(),
               diag::err_typecheck_invalid_restrict_invalid_pointee)
            << EltTy << DS.getSourceRange();
          TypeQuals &= ~QualType::Restrict; // Remove the restrict qualifier.
        }
      } else {
        Diag(DS.getRestrictSpecLoc(),
             diag::err_typecheck_invalid_restrict_not_pointer)
          << Result << DS.getSourceRange();
        TypeQuals &= ~QualType::Restrict; // Remove the restrict qualifier.
      }
    }
    
    // Warn about CV qualifiers on functions: C99 6.7.3p8: "If the specification
    // of a function type includes any type qualifiers, the behavior is
    // undefined."
    if (Result->isFunctionType() && TypeQuals) {
      // Get some location to point at, either the C or V location.
      SourceLocation Loc;
      if (TypeQuals & QualType::Const)
        Loc = DS.getConstSpecLoc();
      else {
        assert((TypeQuals & QualType::Volatile) &&
               "Has CV quals but not C or V?");
        Loc = DS.getVolatileSpecLoc();
      }
      Diag(Loc, diag::warn_typecheck_function_qualifiers)
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
      TypeQuals &= ~QualType::Const;
      TypeQuals &= ~QualType::Volatile;
    }      
    
    Result = Result.getQualifiedType(TypeQuals);
  }
  return Result;
}

static std::string getPrintableNameForEntity(DeclarationName Entity) {
  if (Entity)
    return Entity.getAsString();
  
  return "type name";
}

/// \brief Build a pointer type.
///
/// \param T The type to which we'll be building a pointer.
///
/// \param Quals The cvr-qualifiers to be applied to the pointer type.
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
QualType Sema::BuildPointerType(QualType T, unsigned Quals, 
                                SourceLocation Loc, DeclarationName Entity) {
  if (T->isReferenceType()) {
    // C++ 8.3.2p4: There shall be no ... pointers to references ...
    Diag(Loc, diag::err_illegal_decl_pointer_to_reference)
      << getPrintableNameForEntity(Entity);
    return QualType();
  }

  // Enforce C99 6.7.3p2: "Types other than pointer types derived from
  // object or incomplete types shall not be restrict-qualified."
  if ((Quals & QualType::Restrict) && !T->isIncompleteOrObjectType()) {
    Diag(Loc, diag::err_typecheck_invalid_restrict_invalid_pointee)
      << T;
    Quals &= ~QualType::Restrict;
  }

  // Build the pointer type.
  return Context.getPointerType(T).getQualifiedType(Quals);
}

/// \brief Build a reference type.
///
/// \param T The type to which we'll be building a reference.
///
/// \param Quals The cvr-qualifiers to be applied to the reference type.
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
QualType Sema::BuildReferenceType(QualType T, bool LValueRef, unsigned Quals,
                                  SourceLocation Loc, DeclarationName Entity) {
  if (LValueRef) {
    if (const RValueReferenceType *R = T->getAsRValueReferenceType()) {
      // C++0x [dcl.typedef]p9: If a typedef TD names a type that is a
      //   reference to a type T, and attempt to create the type "lvalue
      //   reference to cv TD" creates the type "lvalue reference to T".
      // We use the qualifiers (restrict or none) of the original reference,
      // not the new ones. This is consistent with GCC.
      return Context.getLValueReferenceType(R->getPointeeType()).
               getQualifiedType(T.getCVRQualifiers());
    }
  }
  if (T->isReferenceType()) {
    // C++ [dcl.ref]p4: There shall be no references to references.
    // 
    // According to C++ DR 106, references to references are only
    // diagnosed when they are written directly (e.g., "int & &"),
    // but not when they happen via a typedef:
    //
    //   typedef int& intref;
    //   typedef intref& intref2;
    //
    // Parser::ParserDeclaratorInternal diagnoses the case where
    // references are written directly; here, we handle the
    // collapsing of references-to-references as described in C++
    // DR 106 and amended by C++ DR 540.
    return T;
  }

  // C++ [dcl.ref]p1:
  //   A declarator that specifies the type “reference to cv void”
  //   is ill-formed.
  if (T->isVoidType()) {
    Diag(Loc, diag::err_reference_to_void);
    return QualType();
  }

  // Enforce C99 6.7.3p2: "Types other than pointer types derived from
  // object or incomplete types shall not be restrict-qualified."
  if ((Quals & QualType::Restrict) && !T->isIncompleteOrObjectType()) {
    Diag(Loc, diag::err_typecheck_invalid_restrict_invalid_pointee)
      << T;
    Quals &= ~QualType::Restrict;
  }

  // C++ [dcl.ref]p1:
  //   [...] Cv-qualified references are ill-formed except when the
  //   cv-qualifiers are introduced through the use of a typedef
  //   (7.1.3) or of a template type argument (14.3), in which case
  //   the cv-qualifiers are ignored.
  //
  // We diagnose extraneous cv-qualifiers for the non-typedef,
  // non-template type argument case within the parser. Here, we just
  // ignore any extraneous cv-qualifiers.
  Quals &= ~QualType::Const;
  Quals &= ~QualType::Volatile;

  // Handle restrict on references.
  if (LValueRef)
    return Context.getLValueReferenceType(T).getQualifiedType(Quals);
  return Context.getRValueReferenceType(T).getQualifiedType(Quals);
}

/// \brief Build an array type.
///
/// \param T The type of each element in the array.
///
/// \param ASM C99 array size modifier (e.g., '*', 'static').
///  
/// \param ArraySize Expression describing the size of the array. 
///
/// \param Quals The cvr-qualifiers to be applied to the array's
/// element type.
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
                              SourceLocation Loc, DeclarationName Entity) {
  // C99 6.7.5.2p1: If the element type is an incomplete or function type, 
  // reject it (e.g. void ary[7], struct foo ary[7], void ary[7]())
  if (RequireCompleteType(Loc, T, 
                             diag::err_illegal_decl_array_incomplete_type))
    return QualType();

  if (T->isFunctionType()) {
    Diag(Loc, diag::err_illegal_decl_array_of_functions)
      << getPrintableNameForEntity(Entity);
    return QualType();
  }
    
  // C++ 8.3.2p4: There shall be no ... arrays of references ...
  if (T->isReferenceType()) {
    Diag(Loc, diag::err_illegal_decl_array_of_references)
      << getPrintableNameForEntity(Entity);
    return QualType();
  } 

  if (const RecordType *EltTy = T->getAsRecordType()) {
    // If the element type is a struct or union that contains a variadic
    // array, accept it as a GNU extension: C99 6.7.2.1p2.
    if (EltTy->getDecl()->hasFlexibleArrayMember())
      Diag(Loc, diag::ext_flexible_array_in_array) << T;
  } else if (T->isObjCInterfaceType()) {
    Diag(Loc, diag::warn_objc_array_of_interfaces) << T;
  }
      
  // C99 6.7.5.2p1: The size expression shall have integer type.
  if (ArraySize && !ArraySize->isTypeDependent() &&
      !ArraySize->getType()->isIntegerType()) {
    Diag(ArraySize->getLocStart(), diag::err_array_size_non_int)
      << ArraySize->getType() << ArraySize->getSourceRange();
    ArraySize->Destroy(Context);
    return QualType();
  }
  llvm::APSInt ConstVal(32);
  if (!ArraySize) {
    T = Context.getIncompleteArrayType(T, ASM, Quals);
  } else if (ArraySize->isValueDependent()) {
    T = Context.getDependentSizedArrayType(T, ArraySize, ASM, Quals);
  } else if (!ArraySize->isIntegerConstantExpr(ConstVal, Context) ||
             (!T->isDependentType() && !T->isConstantSizeType())) {
    // Per C99, a variable array is an array with either a non-constant
    // size or an element type that has a non-constant-size
    T = Context.getVariableArrayType(T, ArraySize, ASM, Quals);
  } else {
    // C99 6.7.5.2p1: If the expression is a constant expression, it shall
    // have a value greater than zero.
    if (ConstVal.isSigned()) {
      if (ConstVal.isNegative()) {
        Diag(ArraySize->getLocStart(),
             diag::err_typecheck_negative_array_size)
          << ArraySize->getSourceRange();
        return QualType();
      } else if (ConstVal == 0) {
        // GCC accepts zero sized static arrays.
        Diag(ArraySize->getLocStart(), diag::ext_typecheck_zero_array_size)
          << ArraySize->getSourceRange();
      }
    } 
    T = Context.getConstantArrayType(T, ConstVal, ASM, Quals);
  }
  // If this is not C99, extwarn about VLA's and C99 array size modifiers.
  if (!getLangOptions().C99) {
    if (ArraySize && !ArraySize->isTypeDependent() && 
        !ArraySize->isValueDependent() && 
        !ArraySize->isIntegerConstantExpr(Context))
      Diag(Loc, diag::ext_vla);
    else if (ASM != ArrayType::Normal || Quals != 0)
      Diag(Loc, diag::ext_c99_array_usage);
  }

  return T;
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
                                 SourceLocation Loc, DeclarationName Entity) {
  if (T->isArrayType() || T->isFunctionType()) {
    Diag(Loc, diag::err_func_returning_array_function) << T;
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
                                 Quals);
}
                                 
/// GetTypeForDeclarator - Convert the type for the specified
/// declarator to Type instances. Skip the outermost Skip type
/// objects.
QualType Sema::GetTypeForDeclarator(Declarator &D, Scope *S, unsigned Skip) {
  bool OmittedReturnType = false;

  if (D.getContext() == Declarator::BlockLiteralContext
      && Skip == 0
      && !D.getDeclSpec().hasTypeSpecifier()
      && (D.getNumTypeObjects() == 0
          || (D.getNumTypeObjects() == 1
              && D.getTypeObject(0).Kind == DeclaratorChunk::Function)))
    OmittedReturnType = true;

  // long long is a C99 feature.
  if (!getLangOptions().C99 && !getLangOptions().CPlusPlus0x &&
      D.getDeclSpec().getTypeSpecWidth() == DeclSpec::TSW_longlong)
    Diag(D.getDeclSpec().getTypeSpecWidthLoc(), diag::ext_longlong);

  // Determine the type of the declarator. Not all forms of declarator
  // have a type.
  QualType T;
  switch (D.getKind()) {
  case Declarator::DK_Abstract:
  case Declarator::DK_Normal:
  case Declarator::DK_Operator: {
    const DeclSpec &DS = D.getDeclSpec();
    if (OmittedReturnType) {
      // We default to a dependent type initially.  Can be modified by
      // the first return statement.
      T = Context.DependentTy;
    } else {
      T = ConvertDeclSpecToType(DS, D.getIdentifierLoc());
      if (T.isNull())
        return T;
    }
    break;
  }

  case Declarator::DK_Constructor:
  case Declarator::DK_Destructor:
  case Declarator::DK_Conversion:
    // Constructors and destructors don't have return types. Use
    // "void" instead. Conversion operators will check their return
    // types separately.
    T = Context.VoidTy;
    break;
  }

  // The name we're declaring, if any.
  DeclarationName Name;
  if (D.getIdentifier())
    Name = D.getIdentifier();

  // Walk the DeclTypeInfo, building the recursive type as we go.
  // DeclTypeInfos are ordered from the identifier out, which is
  // opposite of what we want :).
  for (unsigned i = Skip, e = D.getNumTypeObjects(); i != e; ++i) {
    DeclaratorChunk &DeclType = D.getTypeObject(e-i-1+Skip);
    switch (DeclType.Kind) {
    default: assert(0 && "Unknown decltype!");
    case DeclaratorChunk::BlockPointer:
      // If blocks are disabled, emit an error.
      if (!LangOpts.Blocks)
        Diag(DeclType.Loc, diag::err_blocks_disable);
        
      if (DeclType.Cls.TypeQuals)
        Diag(D.getIdentifierLoc(), diag::err_qualified_block_pointer_type);
      if (!T.getTypePtr()->isFunctionType())
        Diag(D.getIdentifierLoc(), diag::err_nonfunction_block_type);
      else
        T = Context.getBlockPointerType(T);
      break;
    case DeclaratorChunk::Pointer:
      T = BuildPointerType(T, DeclType.Ptr.TypeQuals, DeclType.Loc, Name);
      break;
    case DeclaratorChunk::Reference:
      T = BuildReferenceType(T, DeclType.Ref.LValueRef,
                             DeclType.Ref.HasRestrict ? QualType::Restrict : 0,
                             DeclType.Loc, Name);
      break;
    case DeclaratorChunk::Array: {
      DeclaratorChunk::ArrayTypeInfo &ATI = DeclType.Arr;
      Expr *ArraySize = static_cast<Expr*>(ATI.NumElts);
      ArrayType::ArraySizeModifier ASM;
      if (ATI.isStar)
        ASM = ArrayType::Star;
      else if (ATI.hasStatic)
        ASM = ArrayType::Static;
      else
        ASM = ArrayType::Normal;
      T = BuildArrayType(T, ASM, ArraySize, ATI.TypeQuals, DeclType.Loc, Name);
      break;
    }
    case DeclaratorChunk::Function: {
      // If the function declarator has a prototype (i.e. it is not () and
      // does not have a K&R-style identifier list), then the arguments are part
      // of the type, otherwise the argument list is ().
      const DeclaratorChunk::FunctionTypeInfo &FTI = DeclType.Fun;
      
      // C99 6.7.5.3p1: The return type may not be a function or array type.
      if (T->isArrayType() || T->isFunctionType()) {
        Diag(DeclType.Loc, diag::err_func_returning_array_function) << T;
        T = Context.IntTy;
        D.setInvalidType(true);
      }
        
      if (FTI.NumArgs == 0) {
        if (getLangOptions().CPlusPlus) {
          // C++ 8.3.5p2: If the parameter-declaration-clause is empty, the
          // function takes no arguments.
          T = Context.getFunctionType(T, NULL, 0, FTI.isVariadic,FTI.TypeQuals);
        } else if (FTI.isVariadic) {
          // We allow a zero-parameter variadic function in C if the
          // function is marked with the "overloadable"
          // attribute. Scan for this attribute now.
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
          T = Context.getFunctionType(T, NULL, 0, FTI.isVariadic, 0);
        } else {
          // Simple void foo(), where the incoming T is the result type.
          T = Context.getFunctionNoProtoType(T);
        }
      } else if (FTI.ArgInfo[0].Param == 0) {
        // C99 6.7.5.3p3: Reject int(x,y,z) when it's not a function definition.
        Diag(FTI.ArgInfo[0].IdentLoc, diag::err_ident_list_in_fn_declaration);        
      } else {
        // Otherwise, we have a function with an argument list that is
        // potentially variadic.
        llvm::SmallVector<QualType, 16> ArgTys;
        
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
              if (ArgTy.getCVRQualifiers())
                Diag(DeclType.Loc, diag::err_void_param_qualified);
              
              // Do not add 'void' to the ArgTys list.
              break;
            }
          } else if (!FTI.hasPrototype) {
            if (ArgTy->isPromotableIntegerType()) {
              ArgTy = Context.IntTy;
            } else if (const BuiltinType* BTy = ArgTy->getAsBuiltinType()) {
              if (BTy->getKind() == BuiltinType::Float)
                ArgTy = Context.DoubleTy;
            }
          }
          
          ArgTys.push_back(ArgTy);
        }
        T = Context.getFunctionType(T, &ArgTys[0], ArgTys.size(),
                                    FTI.isVariadic, FTI.TypeQuals);
      }
      break;
    }
    case DeclaratorChunk::MemberPointer:
      // The scope spec must refer to a class, or be dependent.
      DeclContext *DC = computeDeclContext(DeclType.Mem.Scope());
      QualType ClsType;
      // FIXME: Extend for dependent types when it's actually supported.
      // See ActOnCXXNestedNameSpecifier.
      if (CXXRecordDecl *RD = dyn_cast_or_null<CXXRecordDecl>(DC)) {
        ClsType = Context.getTagDeclType(RD);
      } else {
        if (DC) {
          Diag(DeclType.Mem.Scope().getBeginLoc(),
               diag::err_illegal_decl_mempointer_in_nonclass)
            << (D.getIdentifier() ? D.getIdentifier()->getName() : "type name")
            << DeclType.Mem.Scope().getRange();
        }
        D.setInvalidType(true);
        ClsType = Context.IntTy;
      }

      // C++ 8.3.3p3: A pointer to member shall not pointer to ... a member
      //   with reference type, or "cv void."
      if (T->isReferenceType()) {
        Diag(DeclType.Loc, diag::err_illegal_decl_pointer_to_reference)
          << (D.getIdentifier() ? D.getIdentifier()->getName() : "type name");
        D.setInvalidType(true);
        T = Context.IntTy;
      }
      if (T->isVoidType()) {
        Diag(DeclType.Loc, diag::err_illegal_decl_mempointer_to_void)
          << (D.getIdentifier() ? D.getIdentifier()->getName() : "type name");
        T = Context.IntTy;
      }

      // Enforce C99 6.7.3p2: "Types other than pointer types derived from
      // object or incomplete types shall not be restrict-qualified."
      if ((DeclType.Mem.TypeQuals & QualType::Restrict) &&
          !T->isIncompleteOrObjectType()) {
        Diag(DeclType.Loc, diag::err_typecheck_invalid_restrict_invalid_pointee)
          << T;
        DeclType.Mem.TypeQuals &= ~QualType::Restrict;
      }

      T = Context.getMemberPointerType(T, ClsType.getTypePtr()).
                    getQualifiedType(DeclType.Mem.TypeQuals);

      break;
    }

    if (T.isNull()) {
      D.setInvalidType(true);
      T = Context.IntTy;
    }

    // See if there are any attributes on this declarator chunk.
    if (const AttributeList *AL = DeclType.getAttrs())
      ProcessTypeAttributeList(T, AL);
  }

  if (getLangOptions().CPlusPlus && T->isFunctionType()) {
    const FunctionProtoType *FnTy = T->getAsFunctionProtoType();
    assert(FnTy && "Why oh why is there not a FunctionProtoType here ?");

    // C++ 8.3.5p4: A cv-qualifier-seq shall only be part of the function type
    // for a nonstatic member function, the function type to which a pointer
    // to member refers, or the top-level function type of a function typedef
    // declaration.
    if (FnTy->getTypeQuals() != 0 &&
        D.getDeclSpec().getStorageClassSpec() != DeclSpec::SCS_typedef &&
        ((D.getContext() != Declarator::MemberContext &&
          (!D.getCXXScopeSpec().isSet() ||
           !computeDeclContext(D.getCXXScopeSpec())->isRecord())) ||
         D.getDeclSpec().getStorageClassSpec() == DeclSpec::SCS_static)) {
      if (D.isFunctionDeclarator())
        Diag(D.getIdentifierLoc(), diag::err_invalid_qualified_function_type);
      else
        Diag(D.getIdentifierLoc(),
             diag::err_invalid_qualified_typedef_function_type_use);

      // Strip the cv-quals from the type.
      T = Context.getFunctionType(FnTy->getResultType(), FnTy->arg_type_begin(),
                                  FnTy->getNumArgs(), FnTy->isVariadic(), 0);
    }
  }
  
  // If there were any type attributes applied to the decl itself (not the
  // type, apply the type attribute to the type!)
  if (const AttributeList *Attrs = D.getAttributes())
    ProcessTypeAttributeList(T, Attrs);
  
  return T;
}

/// ObjCGetTypeForMethodDefinition - Builds the type for a method definition
/// declarator
QualType Sema::ObjCGetTypeForMethodDefinition(DeclPtrTy D) {
  ObjCMethodDecl *MDecl = cast<ObjCMethodDecl>(D.getAs<Decl>());
  QualType T = MDecl->getResultType();
  llvm::SmallVector<QualType, 16> ArgTys;
  
  // Add the first two invisible argument types for self and _cmd.
  if (MDecl->isInstanceMethod()) {
    QualType selfTy = Context.getObjCInterfaceType(MDecl->getClassInterface());
    selfTy = Context.getPointerType(selfTy);
    ArgTys.push_back(selfTy);
  } else
    ArgTys.push_back(Context.getObjCIdType());
  ArgTys.push_back(Context.getObjCSelType());
      
  for (ObjCMethodDecl::param_iterator PI = MDecl->param_begin(),
       E = MDecl->param_end(); PI != E; ++PI) {
    QualType ArgTy = (*PI)->getType();
    assert(!ArgTy.isNull() && "Couldn't parse type?");
    ArgTy = adjustParameterType(ArgTy);
    ArgTys.push_back(ArgTy);
  }
  T = Context.getFunctionType(T, &ArgTys[0], ArgTys.size(),
                              MDecl->isVariadic(), 0);
  return T;
}

/// UnwrapSimilarPointerTypes - If T1 and T2 are pointer types  that
/// may be similar (C++ 4.4), replaces T1 and T2 with the type that
/// they point to and return true. If T1 and T2 aren't pointer types
/// or pointer-to-member types, or if they are not similar at this
/// level, returns false and leaves T1 and T2 unchanged. Top-level
/// qualifiers on T1 and T2 are ignored. This function will typically
/// be called in a loop that successively "unwraps" pointer and
/// pointer-to-member types to compare them at each level.
bool Sema::UnwrapSimilarPointerTypes(QualType& T1, QualType& T2) {
  const PointerType *T1PtrType = T1->getAsPointerType(),
                    *T2PtrType = T2->getAsPointerType();
  if (T1PtrType && T2PtrType) {
    T1 = T1PtrType->getPointeeType();
    T2 = T2PtrType->getPointeeType();
    return true;
  }

  const MemberPointerType *T1MPType = T1->getAsMemberPointerType(),
                          *T2MPType = T2->getAsMemberPointerType();
  if (T1MPType && T2MPType &&
      Context.getCanonicalType(T1MPType->getClass()) ==
      Context.getCanonicalType(T2MPType->getClass())) {
    T1 = T1MPType->getPointeeType();
    T2 = T2MPType->getPointeeType();
    return true;
  }
  return false;
}

Sema::TypeResult Sema::ActOnTypeName(Scope *S, Declarator &D) {
  // C99 6.7.6: Type names have no identifier.  This is already validated by
  // the parser.
  assert(D.getIdentifier() == 0 && "Type name should have no identifier!");
  
  QualType T = GetTypeForDeclarator(D, S);
  if (T.isNull())
    return true;

  // Check that there are no default arguments (C++ only).
  if (getLangOptions().CPlusPlus)
    CheckExtraCXXDefaultArguments(D);

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
    return;
  }
  
  // Check the attribute arguments.
  if (Attr.getNumArgs() != 1) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }
  Expr *ASArgExpr = static_cast<Expr *>(Attr.getArg(0));
  llvm::APSInt addrSpace(32);
  if (!ASArgExpr->isIntegerConstantExpr(addrSpace, S.Context)) {
    S.Diag(Attr.getLoc(), diag::err_attribute_address_space_not_int)
      << ASArgExpr->getSourceRange();
    return;
  }

  unsigned ASIdx = static_cast<unsigned>(addrSpace.getZExtValue()); 
  Type = S.Context.getAddrSpaceQualType(Type, ASIdx);
}

/// HandleObjCGCTypeAttribute - Process an objc's gc attribute on the
/// specified type.  The attribute contains 1 argument, weak or strong.
static void HandleObjCGCTypeAttribute(QualType &Type, 
                                      const AttributeList &Attr, Sema &S) {
  if (Type.getObjCGCAttr() != QualType::GCNone) {
    S.Diag(Attr.getLoc(), diag::err_attribute_multiple_objc_gc);
    return;
  }
  
  // Check the attribute arguments.
  if (!Attr.getParameterName()) {    
    S.Diag(Attr.getLoc(), diag::err_attribute_argument_n_not_string)
      << "objc_gc" << 1;
    return;
  }
  QualType::GCAttrTypes GCAttr;
  if (Attr.getNumArgs() != 0) {
    S.Diag(Attr.getLoc(), diag::err_attribute_wrong_number_arguments) << 1;
    return;
  }
  if (Attr.getParameterName()->isStr("weak")) 
    GCAttr = QualType::Weak;
  else if (Attr.getParameterName()->isStr("strong"))
    GCAttr = QualType::Strong;
  else {
    S.Diag(Attr.getLoc(), diag::warn_attribute_type_not_supported)
      << "objc_gc" << Attr.getParameterName();
    return;
  }
  
  Type = S.Context.getObjCGCQualType(Type, GCAttr);
}

void Sema::ProcessTypeAttributeList(QualType &Result, const AttributeList *AL) {
  // Scan through and apply attributes to this type where it makes sense.  Some
  // attributes (such as __address_space__, __vector_size__, etc) apply to the
  // type, but others can be present in the type specifiers even though they
  // apply to the decl.  Here we apply type attributes and ignore the rest.
  for (; AL; AL = AL->getNext()) {
    // If this is an attribute we can handle, do so now, otherwise, add it to
    // the LeftOverAttrs list for rechaining.
    switch (AL->getKind()) {
    default: break;
    case AttributeList::AT_address_space:
      HandleAddressSpaceTypeAttribute(Result, *AL, *this);
      break;
    case AttributeList::AT_objc_gc:
      HandleObjCGCTypeAttribute(Result, *AL, *this);
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
/// @param diag The diagnostic value (e.g., 
/// @c diag::err_typecheck_decl_incomplete_type) that will be used
/// for the error message if @p T is incomplete.
///
/// @param Range1  An optional range in the source code that will be a
/// part of the "incomplete type" error message.
///
/// @param Range2  An optional range in the source code that will be a
/// part of the "incomplete type" error message.
///
/// @param PrintType If non-NULL, the type that should be printed
/// instead of @p T. This parameter should be used when the type that
/// we're checking for incompleteness isn't the type that should be
/// displayed to the user, e.g., when T is a type and PrintType is a
/// pointer to T.
///
/// @returns @c true if @p T is incomplete and a diagnostic was emitted,
/// @c false otherwise.
bool Sema::RequireCompleteType(SourceLocation Loc, QualType T, unsigned diag,
                                  SourceRange Range1, SourceRange Range2,
                                  QualType PrintType) {
  // If we have a complete type, we're done.
  if (!T->isIncompleteType())
    return false;

  // If we have a class template specialization or a class member of a
  // class template specialization, try to instantiate it.
  if (const RecordType *Record = T->getAsRecordType()) {
    if (ClassTemplateSpecializationDecl *ClassTemplateSpec
          = dyn_cast<ClassTemplateSpecializationDecl>(Record->getDecl())) {
      if (ClassTemplateSpec->getSpecializationKind() == TSK_Undeclared) {
        // Update the class template specialization's location to
        // refer to the point of instantiation.
        if (Loc.isValid())
          ClassTemplateSpec->setLocation(Loc);
        return InstantiateClassTemplateSpecialization(ClassTemplateSpec,
                                             /*ExplicitInstantiation=*/false);
      }
    } else if (CXXRecordDecl *Rec 
                 = dyn_cast<CXXRecordDecl>(Record->getDecl())) {
      if (CXXRecordDecl *Pattern = Rec->getInstantiatedFromMemberClass()) {
        // Find the class template specialization that surrounds this
        // member class.
        ClassTemplateSpecializationDecl *Spec = 0;
        for (DeclContext *Parent = Rec->getDeclContext(); 
             Parent && !Spec; Parent = Parent->getParent())
          Spec = dyn_cast<ClassTemplateSpecializationDecl>(Parent);
        assert(Spec && "Not a member of a class template specialization?");
        return InstantiateClass(Loc, Rec, Pattern,
                                Spec->getTemplateArgs(), 
                                Spec->getNumTemplateArgs());
      }
    }
  }

  if (PrintType.isNull())
    PrintType = T;

  // We have an incomplete type. Produce a diagnostic.
  Diag(Loc, diag) << PrintType << Range1 << Range2;

  // If the type was a forward declaration of a class/struct/union
  // type, produce 
  const TagType *Tag = 0;
  if (const RecordType *Record = T->getAsRecordType())
    Tag = Record;
  else if (const EnumType *Enum = T->getAsEnumType())
    Tag = Enum;

  if (Tag && !Tag->getDecl()->isInvalidDecl())
    Diag(Tag->getDecl()->getLocation(), 
         Tag->isBeingDefined() ? diag::note_type_being_defined
                               : diag::note_forward_declaration)
        << QualType(Tag, 0);

  return true;
}

/// \brief Retrieve a version of the type 'T' that is qualified by the
/// nested-name-specifier contained in SS.
QualType Sema::getQualifiedNameType(const CXXScopeSpec &SS, QualType T) {
  if (!SS.isSet() || SS.isInvalid() || T.isNull())
    return T;
  
  NestedNameSpecifier *NNS
    = static_cast<NestedNameSpecifier *>(SS.getScopeRep());
  return Context.getQualifiedNameType(NNS, T);
}
