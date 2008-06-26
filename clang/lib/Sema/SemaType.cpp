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
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Basic/LangOptions.h"
using namespace clang;

/// ConvertDeclSpecToType - Convert the specified declspec to the appropriate
/// type object.  This returns null on error.
QualType Sema::ConvertDeclSpecToType(const DeclSpec &DS) {
  // FIXME: Should move the logic from DeclSpec::Finish to here for validity
  // checking.
  QualType Result;
  
  switch (DS.getTypeSpecType()) {
  default: assert(0 && "Unknown TypeSpecType!");
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
  case DeclSpec::TST_unspecified:
    // Unspecified typespec defaults to int in C90.  However, the C90 grammar
    // [C90 6.5] only allows a decl-spec if there was *some* type-specifier,
    // type-qualifier, or storage-class-specifier.  If not, emit an extwarn.
    // Note that the one exception to this is function definitions, which are
    // allowed to be completely missing a declspec.  This is handled in the
    // parser already though by it pretending to have seen an 'int' in this
    // case.
    if (getLangOptions().ImplicitInt) {
      if ((DS.getParsedSpecifiers() & (DeclSpec::PQ_StorageClassSpecifier |
                                       DeclSpec::PQ_TypeSpecifier |
                                       DeclSpec::PQ_TypeQualifier)) == 0)
        Diag(DS.getSourceRange().getBegin(), diag::ext_missing_declspec);
    } else {
      // C99 and C++ require a type specifier.  For example, C99 6.7.2p2 says:
      // "At least one type specifier shall be given in the declaration
      // specifiers in each declaration, and in the specifier-qualifier list in
      // each struct declaration and type name."
      if (!DS.hasTypeSpecifier())
        Diag(DS.getSourceRange().getBegin(), diag::ext_missing_type_specifier);
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
  case DeclSpec::TST_typedef: {
    Decl *D = static_cast<Decl *>(DS.getTypeRep());
    assert(D && "Didn't get a decl for a typedef?");
    assert(DS.getTypeSpecWidth() == 0 && DS.getTypeSpecComplex() == 0 &&
           DS.getTypeSpecSign() == 0 &&
           "Can't handle qualifiers on typedef names yet!");

    // FIXME: Adding a TST_objcInterface clause doesn't seem ideal, so
    // we have this "hack" for now... 
    if (ObjCInterfaceDecl *ObjCIntDecl = dyn_cast<ObjCInterfaceDecl>(D)) {
      if (DS.getProtocolQualifiers() == 0) {
        Result = Context.getObjCInterfaceType(ObjCIntDecl);
        break;
      }
      
      Action::DeclTy **PPDecl = &(*DS.getProtocolQualifiers())[0];
      Result = Context.getObjCQualifiedInterfaceType(ObjCIntDecl,
                                   reinterpret_cast<ObjCProtocolDecl**>(PPDecl),
                                                 DS.getNumProtocolQualifiers());
      break;
    }
    else if (TypedefDecl *typeDecl = dyn_cast<TypedefDecl>(D)) {
      if (Context.getObjCIdType() == Context.getTypedefType(typeDecl)
          && DS.getProtocolQualifiers()) {
          // id<protocol-list>
        Action::DeclTy **PPDecl = &(*DS.getProtocolQualifiers())[0];
        Result = Context.getObjCQualifiedIdType(typeDecl->getUnderlyingType(),
                                 reinterpret_cast<ObjCProtocolDecl**>(PPDecl),
                                            DS.getNumProtocolQualifiers());
        break;
      }
    }
    // TypeQuals handled by caller.
    Result = Context.getTypeDeclType(dyn_cast<TypeDecl>(D));
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
    Result = Context.getTypeOfExpr(E);
    break;
  }
  }
  
  // Handle complex types.
  if (DS.getTypeSpecComplex() == DeclSpec::TSC_complex)
    Result = Context.getComplexType(Result);
  
  assert(DS.getTypeSpecComplex() != DeclSpec::TSC_imaginary &&
         "FIXME: imaginary types not supported yet!");
  
  // See if there are any attributes on the declspec that apply to the type (as
  // opposed to the decl).
  if (const AttributeList *AL = DS.getAttributes())
    ProcessTypeAttributes(Result, AL);
    
  // Apply const/volatile/restrict qualifiers to T.
  if (unsigned TypeQuals = DS.getTypeQualifiers()) {

    // Enforce C99 6.7.3p2: "Types other than pointer types derived from object
    // or incomplete types shall not be restrict-qualified."  C++ also allows
    // restrict-qualified references.
    if (TypeQuals & QualType::Restrict) {
      if (const PointerLikeType *PT = Result->getAsPointerLikeType()) {
        QualType EltTy = PT->getPointeeType();
      
        // If we have a pointer or reference, the pointee must have an object or
        // incomplete type.
        if (!EltTy->isIncompleteOrObjectType()) {
          Diag(DS.getRestrictSpecLoc(),
               diag::err_typecheck_invalid_restrict_invalid_pointee,
               EltTy.getAsString(), DS.getSourceRange());
          TypeQuals &= ~QualType::Restrict; // Remove the restrict qualifier.
        }
      } else {
        Diag(DS.getRestrictSpecLoc(),
             diag::err_typecheck_invalid_restrict_not_pointer,
             Result.getAsString(), DS.getSourceRange());
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
      Diag(Loc, diag::warn_typecheck_function_qualifiers,
           Result.getAsString(), DS.getSourceRange());
    }
    
    Result = Result.getQualifiedType(TypeQuals);
  }
  return Result;
}

/// GetTypeForDeclarator - Convert the type for the specified declarator to Type
/// instances.
QualType Sema::GetTypeForDeclarator(Declarator &D, Scope *S) {
  // long long is a C99 feature.
  if (!getLangOptions().C99 && !getLangOptions().CPlusPlus0x &&
      D.getDeclSpec().getTypeSpecWidth() == DeclSpec::TSW_longlong)
    Diag(D.getDeclSpec().getTypeSpecWidthLoc(), diag::ext_longlong);
  
  QualType T = ConvertDeclSpecToType(D.getDeclSpec());
  
  // Walk the DeclTypeInfo, building the recursive type as we go.  DeclTypeInfos
  // are ordered from the identifier out, which is opposite of what we want :).
  for (unsigned i = 0, e = D.getNumTypeObjects(); i != e; ++i) {
    DeclaratorChunk &DeclType = D.getTypeObject(e-i-1);
    switch (DeclType.Kind) {
    default: assert(0 && "Unknown decltype!");
    case DeclaratorChunk::Pointer:
      if (T->isReferenceType()) {
        // C++ 8.3.2p4: There shall be no ... pointers to references ...
        Diag(DeclType.Loc, diag::err_illegal_decl_pointer_to_reference,
             D.getIdentifier() ? D.getIdentifier()->getName() : "type name");
        D.setInvalidType(true);
        T = Context.IntTy;
      }

      // Enforce C99 6.7.3p2: "Types other than pointer types derived from
      // object or incomplete types shall not be restrict-qualified."
      if ((DeclType.Ptr.TypeQuals & QualType::Restrict) &&
          !T->isIncompleteOrObjectType()) {
        Diag(DeclType.Loc,
             diag::err_typecheck_invalid_restrict_invalid_pointee,
             T.getAsString());
        DeclType.Ptr.TypeQuals &= QualType::Restrict;
      }        
        
      // Apply the pointer typequals to the pointer object.
      T = Context.getPointerType(T).getQualifiedType(DeclType.Ptr.TypeQuals);
        
      // See if there are any attributes on the pointer that apply to it.
      if (const AttributeList *AL = DeclType.Ptr.AttrList)
        ProcessTypeAttributes(T, AL);
        
      break;
    case DeclaratorChunk::Reference:
      if (const ReferenceType *RT = T->getAsReferenceType()) {
        // C++ 8.3.2p4: There shall be no references to references.
        Diag(DeclType.Loc, diag::err_illegal_decl_reference_to_reference,
             D.getIdentifier() ? D.getIdentifier()->getName() : "type name");
        D.setInvalidType(true);
        T = RT->getPointeeType();
      }

      // Enforce C99 6.7.3p2: "Types other than pointer types derived from
      // object or incomplete types shall not be restrict-qualified."
      if (DeclType.Ref.HasRestrict &&
          !T->isIncompleteOrObjectType()) {
        Diag(DeclType.Loc,
             diag::err_typecheck_invalid_restrict_invalid_pointee,
             T.getAsString());
        DeclType.Ref.HasRestrict = false;
      }        

      T = Context.getReferenceType(T);

      // Handle restrict on references.
      if (DeclType.Ref.HasRestrict)
        T.addRestrict();
        
      // See if there are any attributes on the pointer that apply to it.
      if (const AttributeList *AL = DeclType.Ref.AttrList)
        ProcessTypeAttributes(T, AL);
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

      // C99 6.7.5.2p1: If the element type is an incomplete or function type, 
      // reject it (e.g. void ary[7], struct foo ary[7], void ary[7]())
      if (T->isIncompleteType()) { 
        Diag(D.getIdentifierLoc(), diag::err_illegal_decl_array_incomplete_type,
             T.getAsString());
        T = Context.IntTy;
        D.setInvalidType(true);
      } else if (T->isFunctionType()) {
        Diag(D.getIdentifierLoc(), diag::err_illegal_decl_array_of_functions,
             D.getIdentifier() ? D.getIdentifier()->getName() : "type name");
        T = Context.getPointerType(T);
        D.setInvalidType(true);
      } else if (const ReferenceType *RT = T->getAsReferenceType()) {
        // C++ 8.3.2p4: There shall be no ... arrays of references ...
        Diag(D.getIdentifierLoc(), diag::err_illegal_decl_array_of_references,
             D.getIdentifier() ? D.getIdentifier()->getName() : "type name");
        T = RT->getPointeeType();
        D.setInvalidType(true);
      } else if (const RecordType *EltTy = T->getAsRecordType()) {
        // If the element type is a struct or union that contains a variadic
        // array, reject it: C99 6.7.2.1p2.
        if (EltTy->getDecl()->hasFlexibleArrayMember()) {
          Diag(DeclType.Loc, diag::err_flexible_array_in_array,
               T.getAsString());
          T = Context.IntTy;
          D.setInvalidType(true);
        }
      }
      // C99 6.7.5.2p1: The size expression shall have integer type.
      if (ArraySize && !ArraySize->getType()->isIntegerType()) {
        Diag(ArraySize->getLocStart(), diag::err_array_size_non_int, 
             ArraySize->getType().getAsString(), ArraySize->getSourceRange());
        D.setInvalidType(true);
        delete ArraySize;
        ATI.NumElts = ArraySize = 0;
      }
      llvm::APSInt ConstVal(32);
      if (!ArraySize) {
        T = Context.getIncompleteArrayType(T, ASM, ATI.TypeQuals);
      } else if (!ArraySize->isIntegerConstantExpr(ConstVal, Context) ||
                 !T->isConstantSizeType()) {
        // Per C99, a variable array is an array with either a non-constant
        // size or an element type that has a non-constant-size
        T = Context.getVariableArrayType(T, ArraySize, ASM, ATI.TypeQuals);
      } else {
        // C99 6.7.5.2p1: If the expression is a constant expression, it shall
        // have a value greater than zero.
        if (ConstVal.isSigned()) {
          if (ConstVal.isNegative()) {
            Diag(ArraySize->getLocStart(), 
                 diag::err_typecheck_negative_array_size,
                 ArraySize->getSourceRange());
            D.setInvalidType(true);
          } else if (ConstVal == 0) {
            // GCC accepts zero sized static arrays.
            Diag(ArraySize->getLocStart(), diag::ext_typecheck_zero_array_size,
                 ArraySize->getSourceRange());
          }
        } 
        T = Context.getConstantArrayType(T, ConstVal, ASM, ATI.TypeQuals);
      }
      // If this is not C99, extwarn about VLA's and C99 array size modifiers.
      if (!getLangOptions().C99 && 
          (ASM != ArrayType::Normal ||
           (ArraySize && !ArraySize->isIntegerConstantExpr(Context))))
        Diag(D.getIdentifierLoc(), diag::ext_vla);
      break;
    }
    case DeclaratorChunk::Function:
      // If the function declarator has a prototype (i.e. it is not () and
      // does not have a K&R-style identifier list), then the arguments are part
      // of the type, otherwise the argument list is ().
      const DeclaratorChunk::FunctionTypeInfo &FTI = DeclType.Fun;
      
      // C99 6.7.5.3p1: The return type may not be a function or array type.
      if (T->isArrayType() || T->isFunctionType()) {
        Diag(DeclType.Loc, diag::err_func_returning_array_function,
             T.getAsString());
        T = Context.IntTy;
        D.setInvalidType(true);
      }
        
      if (!FTI.hasPrototype) {
        // Simple void foo(), where the incoming T is the result type.
        T = Context.getFunctionTypeNoProto(T);

        // C99 6.7.5.3p3: Reject int(x,y,z) when it's not a function definition.
        if (FTI.NumArgs != 0)
          Diag(FTI.ArgInfo[0].IdentLoc, diag::err_ident_list_in_fn_declaration);
        
      } else {
        // Otherwise, we have a function with an argument list that is
        // potentially variadic.
        llvm::SmallVector<QualType, 16> ArgTys;
        
        for (unsigned i = 0, e = FTI.NumArgs; i != e; ++i) {
          ParmVarDecl *Param = (ParmVarDecl *)FTI.ArgInfo[i].Param;
          QualType ArgTy = Param->getType();
          assert(!ArgTy.isNull() && "Couldn't parse type?");
          //
          // Perform the default function/array conversion (C99 6.7.5.3p[7,8]).
          // This matches the conversion that is done in 
          // Sema::ActOnParamDeclarator(). Without this conversion, the
          // argument type in the function prototype *will not* match the
          // type in ParmVarDecl (which makes the code generator unhappy).
          //
          // FIXME: We still apparently need the conversion in 
          // Sema::ActOnParamDeclarator(). This doesn't make any sense, since
          // it should be driving off the type being created here.
          // 
          // FIXME: If a source translation tool needs to see the original type,
          // then we need to consider storing both types somewhere...
          // 
          if (ArgTy->isArrayType()) {
            ArgTy = Context.getArrayDecayedType(ArgTy);
          } else if (ArgTy->isFunctionType())
            ArgTy = Context.getPointerType(ArgTy);
          
          // Look for 'void'.  void is allowed only as a single argument to a
          // function with no other parameters (C99 6.7.5.3p10).  We record
          // int(void) as a FunctionTypeProto with an empty argument list.
          else if (ArgTy->isVoidType()) {
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
          }
          
          ArgTys.push_back(ArgTy);
        }
        T = Context.getFunctionType(T, &ArgTys[0], ArgTys.size(),
                                    FTI.isVariadic);
      }
      break;
    }
  }
  
  return T;
}

/// ObjCGetTypeForMethodDefinition - Builds the type for a method definition
/// declarator
QualType Sema::ObjCGetTypeForMethodDefinition(DeclTy *D) {
  ObjCMethodDecl *MDecl = dyn_cast<ObjCMethodDecl>(static_cast<Decl *>(D));
  QualType T = MDecl->getResultType();
  llvm::SmallVector<QualType, 16> ArgTys;
  
  // Add the first two invisible argument types for self and _cmd.
  if (MDecl->isInstance()) {
    QualType selfTy = Context.getObjCInterfaceType(MDecl->getClassInterface());
    selfTy = Context.getPointerType(selfTy);
    ArgTys.push_back(selfTy);
  }
  else
    ArgTys.push_back(Context.getObjCIdType());
  ArgTys.push_back(Context.getObjCSelType());
      
  for (int i = 0, e = MDecl->getNumParams(); i != e; ++i) {
    ParmVarDecl *PDecl = MDecl->getParamDecl(i);
    QualType ArgTy = PDecl->getType();
    assert(!ArgTy.isNull() && "Couldn't parse type?");
    // Perform the default function/array conversion (C99 6.7.5.3p[7,8]).
    // This matches the conversion that is done in 
    // Sema::ActOnParamDeclarator(). 
    if (ArgTy->isArrayType())
      ArgTy = Context.getArrayDecayedType(ArgTy);
    else if (ArgTy->isFunctionType())
      ArgTy = Context.getPointerType(ArgTy);
    ArgTys.push_back(ArgTy);
  }
  T = Context.getFunctionType(T, &ArgTys[0], ArgTys.size(),
                              MDecl->isVariadic());
  return T;
}

Sema::TypeResult Sema::ActOnTypeName(Scope *S, Declarator &D) {
  // C99 6.7.6: Type names have no identifier.  This is already validated by
  // the parser.
  assert(D.getIdentifier() == 0 && "Type name should have no identifier!");
  
  QualType T = GetTypeForDeclarator(D, S);

  assert(!T.isNull() && "GetTypeForDeclarator() returned null type");
  
  // Check that there are no default arguments (C++ only).
  if (getLangOptions().CPlusPlus)
    CheckExtraCXXDefaultArguments(D);

  // In this context, we *do not* check D.getInvalidType(). If the declarator
  // type was invalid, GetTypeForDeclarator() still returns a "valid" type,
  // though it will not reflect the user specified type.
  return T.getAsOpaquePtr();
}

void Sema::ProcessTypeAttributes(QualType &Result, const AttributeList *AL) {
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
      Result = HandleAddressSpaceTypeAttribute(Result, AL);
      continue;
    }
  }
}

/// HandleAddressSpaceTypeAttribute - Process an address_space attribute on the
/// specified type.
QualType Sema::HandleAddressSpaceTypeAttribute(QualType Type, 
                                               const AttributeList *Attr) {
  // If this type is already address space qualified, reject it.
  // Clause 6.7.3 - Type qualifiers: "No type shall be qualified by qualifiers
  // for two or more different address spaces."
  if (Type.getAddressSpace()) {
    Diag(Attr->getLoc(), diag::err_attribute_address_multiple_qualifiers);
    return Type;
  }
  
  // Check the attribute arguments.
  if (Attr->getNumArgs() != 1) {
    Diag(Attr->getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("1"));
    return Type;
  }
  Expr *ASArgExpr = static_cast<Expr *>(Attr->getArg(0));
  llvm::APSInt addrSpace(32);
  if (!ASArgExpr->isIntegerConstantExpr(addrSpace, Context)) {
    Diag(Attr->getLoc(), diag::err_attribute_address_space_not_int,
         ASArgExpr->getSourceRange());
    return Type;
  }

  unsigned ASIdx = static_cast<unsigned>(addrSpace.getZExtValue()); 
  return Context.getASQualType(Type, ASIdx);
}

/// HandleModeTypeAttribute - Process a mode attribute on the
/// specified type.
QualType Sema::HandleModeTypeAttribute(QualType Type, 
                                       const AttributeList *Attr) {
  // This attribute isn't documented, but glibc uses it.  It changes
  // the width of an int or unsigned int to the specified size.

  // Check that there aren't any arguments
  if (Attr->getNumArgs() != 0) {
    Diag(Attr->getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("0"));
    return Type;
  }

  IdentifierInfo * Name = Attr->getParameterName();
  if (!Name) {
    Diag(Attr->getLoc(), diag::err_attribute_missing_parameter_name);
    return Type;
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
    if (!memcmp(Str, "word", 4)) {
      // FIXME: glibc uses this to define register_t; this is
      // narrover than a pointer on PIC16 and other embedded
      // platforms
      DestWidth = Context.getTypeSize(Context.VoidPtrTy);
      break;
    }
    if (!memcmp(Str, "byte", 4)) {
      DestWidth = Context.getTypeSize(Context.CharTy);
      break;
    }
    break;
  case 7:
    if (!memcmp(Str, "pointer", 7)) {
      DestWidth = Context.getTypeSize(Context.VoidPtrTy);
      IntegerMode = true;
      break;
    }
    break;
  }

  // FIXME: Need proper fixed-width types
  QualType RetTy;
  switch (DestWidth) {
  case 0:
    Diag(Attr->getLoc(), diag::err_unknown_machine_mode, 
         std::string(Str, Len));
    return Type;
  case 8:
    assert(IntegerMode);
    if (Type->isSignedIntegerType())
      RetTy = Context.SignedCharTy;
    else
      RetTy = Context.UnsignedCharTy;
    break;
  case 16:
    assert(IntegerMode);
    if (Type->isSignedIntegerType())
      RetTy = Context.ShortTy;
    else
      RetTy = Context.UnsignedShortTy;
    break;
  case 32:
    if (!IntegerMode)
      RetTy = Context.FloatTy;
    else if (Type->isSignedIntegerType())
      RetTy = Context.IntTy;
    else
      RetTy = Context.UnsignedIntTy;
    break;
  case 64:
    if (!IntegerMode)
      RetTy = Context.DoubleTy;
    else if (Type->isSignedIntegerType())
      RetTy = Context.LongLongTy;
    else
      RetTy = Context.UnsignedLongLongTy;
    break;
  default:
    Diag(Attr->getLoc(), diag::err_unsupported_machine_mode,
         std::string(Str, Len));
    return Type;
  }

  if (!Type->getAsBuiltinType())
    Diag(Attr->getLoc(), diag::err_mode_not_primitive);
  else if (!(IntegerMode && Type->isIntegerType()) &&
           !(!IntegerMode && Type->isFloatingType())) {
    Diag(Attr->getLoc(), diag::err_mode_wrong_type);
  }

  return RetTy;
}


