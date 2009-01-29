//===--- SemaInit.cpp - Semantic Analysis for Initializers ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for initializers.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "clang/Parse/Designator.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Basic/DiagnosticSema.h"
using namespace clang;

/// Recursively replaces NULL values within the given initializer list
/// with expressions that perform value-initialization of the
/// appropriate type.
static void fillInValueInitializations(ASTContext &Context, InitListExpr *ILE) {
  assert((ILE->getType() != Context.VoidTy) && "Should not have void type");
  if (const RecordType *RType = ILE->getType()->getAsRecordType()) {
    unsigned Init = 0, NumInits = ILE->getNumInits();
    for (RecordDecl::field_iterator Field = RType->getDecl()->field_begin(),
                                 FieldEnd = RType->getDecl()->field_end();
         Field != FieldEnd; ++Field) {
      if (Field->isUnnamedBitfield())
        continue;

      if (Init >= NumInits)
        break;

      // FIXME: Check for fields with reference type in C++?
      if (!ILE->getInit(Init))
        ILE->setInit(Init, 
                     new (Context) CXXZeroInitValueExpr(Field->getType(), 
                                                        SourceLocation(),
                                                        SourceLocation()));
      else if (InitListExpr *InnerILE = dyn_cast<InitListExpr>(ILE->getInit(Init)))
        fillInValueInitializations(Context, InnerILE);
      ++Init;
    }

    return;
  } 

  QualType ElementType;
  
  if (const ArrayType *AType = Context.getAsArrayType(ILE->getType()))
    ElementType = AType->getElementType();
  else if (const VectorType *VType = ILE->getType()->getAsVectorType())
    ElementType = VType->getElementType();
  else 
    ElementType = ILE->getType();
  
  for (unsigned Init = 0, NumInits = ILE->getNumInits(); Init != NumInits; 
       ++Init) {
    if (!ILE->getInit(Init))
      ILE->setInit(Init, new (Context) CXXZeroInitValueExpr(ElementType, 
                                                            SourceLocation(),
                                                            SourceLocation()));
    else if (InitListExpr *InnerILE = dyn_cast<InitListExpr>(ILE->getInit(Init)))
      fillInValueInitializations(Context, InnerILE);
  }
}

InitListChecker::InitListChecker(Sema *S, InitListExpr *IL, QualType &T) {
  hadError = false;
  SemaRef = S;

  unsigned newIndex = 0;
  unsigned newStructuredIndex = 0;
  FullyStructuredList 
    = getStructuredSubobjectInit(IL, newIndex, T, 0, 0, SourceRange());
  CheckExplicitInitList(IL, T, newIndex, FullyStructuredList, newStructuredIndex);

  if (!hadError) {
    fillInValueInitializations(SemaRef->Context, FullyStructuredList);
  }
}

int InitListChecker::numArrayElements(QualType DeclType) {
  // FIXME: use a proper constant
  int maxElements = 0x7FFFFFFF;
  if (const ConstantArrayType *CAT =
        SemaRef->Context.getAsConstantArrayType(DeclType)) {
    maxElements = static_cast<int>(CAT->getSize().getZExtValue());
  }
  return maxElements;
}

int InitListChecker::numStructUnionElements(QualType DeclType) {
  RecordDecl *structDecl = DeclType->getAsRecordType()->getDecl();
  int InitializableMembers = 0;
  for (RecordDecl::field_iterator Field = structDecl->field_begin(),
                               FieldEnd = structDecl->field_end();
       Field != FieldEnd; ++Field) {
    if ((*Field)->getIdentifier() || !(*Field)->isBitField())
      ++InitializableMembers;
  }
  if (structDecl->isUnion())
    return std::min(InitializableMembers, 1);
  return InitializableMembers - structDecl->hasFlexibleArrayMember();
}

void InitListChecker::CheckImplicitInitList(InitListExpr *ParentIList, 
                                            QualType T, unsigned &Index,
                                            InitListExpr *StructuredList,
                                            unsigned &StructuredIndex) {
  int maxElements = 0;
  
  if (T->isArrayType())
    maxElements = numArrayElements(T);
  else if (T->isStructureType() || T->isUnionType())
    maxElements = numStructUnionElements(T);
  else if (T->isVectorType())
    maxElements = T->getAsVectorType()->getNumElements();
  else
    assert(0 && "CheckImplicitInitList(): Illegal type");

  // FIXME: Perhaps we should move this warning elsewhere?
  if (maxElements == 0) {
    SemaRef->Diag(ParentIList->getInit(Index)->getLocStart(),
                  diag::err_implicit_empty_initializer);
    ++Index;
    hadError = true;
    return;
  }

  // Build a structured initializer list corresponding to this subobject.
  InitListExpr *StructuredSubobjectInitList
    = getStructuredSubobjectInit(ParentIList, Index, T, StructuredList, 
                                 StructuredIndex, 
                                 ParentIList->getInit(Index)->getSourceRange());
  unsigned StructuredSubobjectInitIndex = 0;

  // Check the element types and build the structural subobject.
  CheckListElementTypes(ParentIList, T, false, Index,
                        StructuredSubobjectInitList, 
                        StructuredSubobjectInitIndex);
}

void InitListChecker::CheckExplicitInitList(InitListExpr *IList, QualType &T,
                                            unsigned &Index,
                                            InitListExpr *StructuredList,
                                            unsigned &StructuredIndex) {
  assert(IList->isExplicit() && "Illegal Implicit InitListExpr");
  SyntacticToSemantic[IList] = StructuredList;
  StructuredList->setSyntacticForm(IList);
  CheckListElementTypes(IList, T, true, Index, StructuredList, 
                        StructuredIndex);
  IList->setType(T);
  StructuredList->setType(T);
  if (hadError)
    return;

  if (Index < IList->getNumInits()) {
    // We have leftover initializers
    if (IList->getNumInits() > 0 &&
        SemaRef->IsStringLiteralInit(IList->getInit(Index), T)) {
      // Special-case
      SemaRef->Diag(IList->getInit(Index)->getLocStart(),
                    diag::err_excess_initializers_in_char_array_initializer)
        << IList->getInit(Index)->getSourceRange();
      hadError = true; 
    } else if (!T->isIncompleteType()) {
      // Don't warn for incomplete types, since we'll get an error elsewhere
      SemaRef->Diag(IList->getInit(Index)->getLocStart(), 
                    diag::warn_excess_initializers)
        << IList->getInit(Index)->getSourceRange();
    }
  }

  if (T->isScalarType())
    SemaRef->Diag(IList->getLocStart(), diag::warn_braces_around_scalar_init)
      << IList->getSourceRange();
}

void InitListChecker::CheckListElementTypes(InitListExpr *IList,
                                            QualType &DeclType, 
                                            bool SubobjectIsDesignatorContext,
                                            unsigned &Index,
                                            InitListExpr *StructuredList,
                                            unsigned &StructuredIndex) {
  if (DeclType->isScalarType()) {
    CheckScalarType(IList, DeclType, Index, StructuredList, StructuredIndex);
  } else if (DeclType->isVectorType()) {
    CheckVectorType(IList, DeclType, Index, StructuredList, StructuredIndex);
  } else if (DeclType->isAggregateType() || DeclType->isUnionType()) {
    if (DeclType->isStructureType() || DeclType->isUnionType()) {
      RecordDecl *RD = DeclType->getAsRecordType()->getDecl();
      CheckStructUnionTypes(IList, DeclType, RD->field_begin(), 
                            SubobjectIsDesignatorContext, Index,
                            StructuredList, StructuredIndex);
    } else if (DeclType->isArrayType()) {
      llvm::APSInt Zero(
                      SemaRef->Context.getTypeSize(SemaRef->Context.getSizeType()),
                      false);
      CheckArrayType(IList, DeclType, Zero, SubobjectIsDesignatorContext, Index,
                     StructuredList, StructuredIndex);
    }
    else
      assert(0 && "Aggregate that isn't a structure or array?!");
  } else if (DeclType->isVoidType() || DeclType->isFunctionType()) {
    // This type is invalid, issue a diagnostic.
    ++Index;
    SemaRef->Diag(IList->getLocStart(), diag::err_illegal_initializer_type)
      << DeclType;
    hadError = true;
  } else {
    // In C, all types are either scalars or aggregates, but
    // additional handling is needed here for C++ (and possibly others?). 
    assert(0 && "Unsupported initializer type");
  }
}

void InitListChecker::CheckSubElementType(InitListExpr *IList,
                                          QualType ElemType, 
                                          unsigned &Index,
                                          InitListExpr *StructuredList,
                                          unsigned &StructuredIndex) {
  Expr *expr = IList->getInit(Index);
  if (InitListExpr *SubInitList = dyn_cast<InitListExpr>(expr)) {
    unsigned newIndex = 0;
    unsigned newStructuredIndex = 0;
    InitListExpr *newStructuredList 
      = getStructuredSubobjectInit(IList, Index, ElemType,
                                   StructuredList, StructuredIndex,
                                   SubInitList->getSourceRange());
    CheckExplicitInitList(SubInitList, ElemType, newIndex, 
                          newStructuredList, newStructuredIndex);
    ++StructuredIndex;
    ++Index;
  } else if (StringLiteral *lit =
             SemaRef->IsStringLiteralInit(expr, ElemType)) {
    SemaRef->CheckStringLiteralInit(lit, ElemType);
    UpdateStructuredListElement(StructuredList, StructuredIndex, lit);
    ++Index;
  } else if (ElemType->isScalarType()) {
    CheckScalarType(IList, ElemType, Index, StructuredList, StructuredIndex);
  } else if (expr->getType()->getAsRecordType() &&
             SemaRef->Context.typesAreCompatible(
               expr->getType().getUnqualifiedType(),
               ElemType.getUnqualifiedType())) {
    UpdateStructuredListElement(StructuredList, StructuredIndex, expr);
    ++Index;
    // FIXME: Add checking
  } else {
    CheckImplicitInitList(IList, ElemType, Index, StructuredList, 
                          StructuredIndex);
    ++StructuredIndex;
 }
}

void InitListChecker::CheckScalarType(InitListExpr *IList, QualType &DeclType,
                                      unsigned &Index,
                                      InitListExpr *StructuredList,
                                      unsigned &StructuredIndex) {
  if (Index < IList->getNumInits()) {
    Expr *expr = IList->getInit(Index);
    if (isa<InitListExpr>(expr)) {
      SemaRef->Diag(IList->getLocStart(),
                    diag::err_many_braces_around_scalar_init)
        << IList->getSourceRange();
      hadError = true;
      ++Index;
      ++StructuredIndex;
      return;
    } else if (isa<DesignatedInitExpr>(expr)) {
      SemaRef->Diag(expr->getSourceRange().getBegin(), 
                    diag::err_designator_for_scalar_init)
        << DeclType << expr->getSourceRange();
      hadError = true;
      ++Index;
      ++StructuredIndex;
      return;
    }

    Expr *savExpr = expr; // Might be promoted by CheckSingleInitializer.
    if (SemaRef->CheckSingleInitializer(expr, DeclType, false))
      hadError = true; // types weren't compatible.
    else if (savExpr != expr) {
      // The type was promoted, update initializer list.
      IList->setInit(Index, expr);
    }
    if (hadError)
      ++StructuredIndex;
    else
      UpdateStructuredListElement(StructuredList, StructuredIndex, expr);
    ++Index;
  } else {
    SemaRef->Diag(IList->getLocStart(), diag::err_empty_scalar_initializer)
      << IList->getSourceRange();
    hadError = true;
    ++Index;
    ++StructuredIndex;
    return;
  }
}

void InitListChecker::CheckVectorType(InitListExpr *IList, QualType DeclType, 
                                      unsigned &Index,
                                      InitListExpr *StructuredList,
                                      unsigned &StructuredIndex) {
  if (Index < IList->getNumInits()) {
    const VectorType *VT = DeclType->getAsVectorType();
    int maxElements = VT->getNumElements();
    QualType elementType = VT->getElementType();
    
    for (int i = 0; i < maxElements; ++i) {
      // Don't attempt to go past the end of the init list
      if (Index >= IList->getNumInits())
        break;
      CheckSubElementType(IList, elementType, Index,
                          StructuredList, StructuredIndex);
    }
  }
}

void InitListChecker::CheckArrayType(InitListExpr *IList, QualType &DeclType, 
                                     llvm::APSInt elementIndex,
                                     bool SubobjectIsDesignatorContext, 
                                     unsigned &Index,
                                     InitListExpr *StructuredList,
                                     unsigned &StructuredIndex) {
  // Check for the special-case of initializing an array with a string.
  if (Index < IList->getNumInits()) {
    if (StringLiteral *lit = 
        SemaRef->IsStringLiteralInit(IList->getInit(Index), DeclType)) {
      SemaRef->CheckStringLiteralInit(lit, DeclType);
      // We place the string literal directly into the resulting
      // initializer list. This is the only place where the structure
      // of the structured initializer list doesn't match exactly,
      // because doing so would involve allocating one character
      // constant for each string.
      UpdateStructuredListElement(StructuredList, StructuredIndex, lit);
      StructuredList->resizeInits(SemaRef->Context, StructuredIndex);
      ++Index;
      return;
    }
  }
  if (const VariableArrayType *VAT =
        SemaRef->Context.getAsVariableArrayType(DeclType)) {
    // Check for VLAs; in standard C it would be possible to check this
    // earlier, but I don't know where clang accepts VLAs (gcc accepts
    // them in all sorts of strange places).
    SemaRef->Diag(VAT->getSizeExpr()->getLocStart(),
                  diag::err_variable_object_no_init)
      << VAT->getSizeExpr()->getSourceRange();
    hadError = true;
    ++Index;
    ++StructuredIndex;
    return;
  }

  // We might know the maximum number of elements in advance.
  llvm::APSInt maxElements(elementIndex.getBitWidth(),
                           elementIndex.isUnsigned());
  bool maxElementsKnown = false;
  if (const ConstantArrayType *CAT =
        SemaRef->Context.getAsConstantArrayType(DeclType)) {
    maxElements = CAT->getSize();
    elementIndex.extOrTrunc(maxElements.getBitWidth());
    elementIndex.setIsUnsigned(maxElements.isUnsigned());
    maxElementsKnown = true;
  }

  QualType elementType = SemaRef->Context.getAsArrayType(DeclType)
                             ->getElementType();
  while (Index < IList->getNumInits()) {
    Expr *Init = IList->getInit(Index);
    if (DesignatedInitExpr *DIE = dyn_cast<DesignatedInitExpr>(Init)) {
      // If we're not the subobject that matches up with the '{' for
      // the designator, we shouldn't be handling the
      // designator. Return immediately.
      if (!SubobjectIsDesignatorContext)
        return;

      // Handle this designated initializer. elementIndex will be
      // updated to be the next array element we'll initialize.
      if (CheckDesignatedInitializer(IList, DIE, DIE->designators_begin(), 
                                     DeclType, 0, &elementIndex, Index,
                                     StructuredList, StructuredIndex)) {
        hadError = true;
        continue;
      }

      if (elementIndex.getBitWidth() > maxElements.getBitWidth())
        maxElements.extend(elementIndex.getBitWidth());
      else if (elementIndex.getBitWidth() < maxElements.getBitWidth())
        elementIndex.extend(maxElements.getBitWidth());
      elementIndex.setIsUnsigned(maxElements.isUnsigned());

      // If the array is of incomplete type, keep track of the number of
      // elements in the initializer.
      if (!maxElementsKnown && elementIndex > maxElements)
        maxElements = elementIndex;

      continue;
    }

    // If we know the maximum number of elements, and we've already
    // hit it, stop consuming elements in the initializer list.
    if (maxElementsKnown && elementIndex == maxElements)
      break;

    // Check this element.
    CheckSubElementType(IList, elementType, Index,
                        StructuredList, StructuredIndex);
    ++elementIndex;

    // If the array is of incomplete type, keep track of the number of
    // elements in the initializer.
    if (!maxElementsKnown && elementIndex > maxElements)
      maxElements = elementIndex;
  }
  if (DeclType->isIncompleteArrayType()) {
    // If this is an incomplete array type, the actual type needs to
    // be calculated here.
    llvm::APSInt Zero(maxElements.getBitWidth(), maxElements.isUnsigned());
    if (maxElements == Zero) {
      // Sizing an array implicitly to zero is not allowed by ISO C,
      // but is supported by GNU.
      SemaRef->Diag(IList->getLocStart(),
                    diag::ext_typecheck_zero_array_size);
    }

    DeclType = SemaRef->Context.getConstantArrayType(elementType, maxElements, 
                                                     ArrayType::Normal, 0);
  }
}

void InitListChecker::CheckStructUnionTypes(InitListExpr *IList, 
                                            QualType DeclType, 
                                            RecordDecl::field_iterator Field,
                                            bool SubobjectIsDesignatorContext, 
                                            unsigned &Index,
                                            InitListExpr *StructuredList,
                                            unsigned &StructuredIndex) {
  RecordDecl* structDecl = DeclType->getAsRecordType()->getDecl();
    
  // If the record is invalid, some of it's members are invalid. To avoid
  // confusion, we forgo checking the intializer for the entire record.
  if (structDecl->isInvalidDecl()) {
    hadError = true;
    return;
  }    
  // If structDecl is a forward declaration, this loop won't do
  // anything except look at designated initializers; That's okay,
  // because an error should get printed out elsewhere. It might be
  // worthwhile to skip over the rest of the initializer, though.
  RecordDecl *RD = DeclType->getAsRecordType()->getDecl();
  RecordDecl::field_iterator FieldEnd = RD->field_end();
  while (Index < IList->getNumInits()) {
    Expr *Init = IList->getInit(Index);

    if (DesignatedInitExpr *DIE = dyn_cast<DesignatedInitExpr>(Init)) {
      // If we're not the subobject that matches up with the '{' for
      // the designator, we shouldn't be handling the
      // designator. Return immediately.
      if (!SubobjectIsDesignatorContext)
        return;

      // Handle this designated initializer. Field will be updated to
      // the next field that we'll be initializing.
      if (CheckDesignatedInitializer(IList, DIE, DIE->designators_begin(), 
                                     DeclType, &Field, 0, Index,
                                     StructuredList, StructuredIndex))
        hadError = true;

      continue;
    }

    if (Field == FieldEnd) {
      // We've run out of fields. We're done.
      break;
    }

    // If we've hit the flexible array member at the end, we're done.
    if (Field->getType()->isIncompleteArrayType())
      break;

    if (!Field->getIdentifier() && Field->isBitField()) {
      // Don't initialize unnamed bitfields, e.g. "int : 20;"
      ++Field;
      continue;
    }

    CheckSubElementType(IList, Field->getType(), Index,
                        StructuredList, StructuredIndex);
    if (DeclType->isUnionType())
      break;

    ++Field;
  }

  // FIXME: Implement flexible array initialization GCC extension (it's a 
  // really messy extension to implement, unfortunately...the necessary
  // information isn't actually even here!)
}

/// @brief Check the well-formedness of a C99 designated initializer.
///
/// Determines whether the designated initializer @p DIE, which
/// resides at the given @p Index within the initializer list @p
/// IList, is well-formed for a current object of type @p DeclType
/// (C99 6.7.8). The actual subobject that this designator refers to
/// within the current subobject is returned in either 
/// @p NextField or @p NextElementIndex (whichever is appropriate).
///
/// @param IList  The initializer list in which this designated
/// initializer occurs.
///
/// @param DIE  The designated initializer and its initialization
/// expression.
///
/// @param DeclType  The type of the "current object" (C99 6.7.8p17),
/// into which the designation in @p DIE should refer.
///
/// @param NextField  If non-NULL and the first designator in @p DIE is
/// a field, this will be set to the field declaration corresponding
/// to the field named by the designator.
///
/// @param NextElementIndex  If non-NULL and the first designator in @p
/// DIE is an array designator or GNU array-range designator, this
/// will be set to the last index initialized by this designator.
///
/// @param Index  Index into @p IList where the designated initializer
/// @p DIE occurs.
///
/// @param StructuredList  The initializer list expression that
/// describes all of the subobject initializers in the order they'll
/// actually be initialized.
///
/// @returns true if there was an error, false otherwise.
bool 
InitListChecker::CheckDesignatedInitializer(InitListExpr *IList,
                                      DesignatedInitExpr *DIE, 
                                      DesignatedInitExpr::designators_iterator D,
                                      QualType &CurrentObjectType,
                                      RecordDecl::field_iterator *NextField,
                                      llvm::APSInt *NextElementIndex,
                                      unsigned &Index,
                                      InitListExpr *StructuredList,
                                      unsigned &StructuredIndex,
                                      bool FinishSubobjectInit) {
  if (D == DIE->designators_end()) {
    // Check the actual initialization for the designated object type.
    bool prevHadError = hadError;

    // Temporarily remove the designator expression from the
    // initializer list that the child calls see, so that we don't try
    // to re-process the designator.
    unsigned OldIndex = Index;
    IList->setInit(OldIndex, DIE->getInit());

    CheckSubElementType(IList, CurrentObjectType, Index,
                        StructuredList, StructuredIndex);

    // Restore the designated initializer expression in the syntactic
    // form of the initializer list.
    if (IList->getInit(OldIndex) != DIE->getInit())
      DIE->setInit(IList->getInit(OldIndex));
    IList->setInit(OldIndex, DIE);

    return hadError && !prevHadError;
  }

  bool IsFirstDesignator = (D == DIE->designators_begin());
  assert((IsFirstDesignator || StructuredList) && 
         "Need a non-designated initializer list to start from");

  // Determine the structural initializer list that corresponds to the
  // current subobject.
  StructuredList = IsFirstDesignator? SyntacticToSemantic[IList]
    : getStructuredSubobjectInit(IList, Index, CurrentObjectType, StructuredList,
                                 StructuredIndex,
                                 SourceRange(D->getStartLocation(),
                                             DIE->getSourceRange().getEnd()));
  assert(StructuredList && "Expected a structured initializer list");

  if (D->isFieldDesignator()) {
    // C99 6.7.8p7:
    //
    //   If a designator has the form
    //
    //      . identifier
    //
    //   then the current object (defined below) shall have
    //   structure or union type and the identifier shall be the
    //   name of a member of that type. 
    const RecordType *RT = CurrentObjectType->getAsRecordType();
    if (!RT) {
      SourceLocation Loc = D->getDotLoc();
      if (Loc.isInvalid())
        Loc = D->getFieldLoc();
      SemaRef->Diag(Loc, diag::err_field_designator_non_aggr)
        << SemaRef->getLangOptions().CPlusPlus << CurrentObjectType;
      ++Index;
      return true;
    }

    // Note: we perform a linear search of the fields here, despite
    // the fact that we have a faster lookup method, because we always
    // need to compute the field's index.
    IdentifierInfo *FieldName = D->getFieldName();
    unsigned FieldIndex = 0;
    RecordDecl::field_iterator Field = RT->getDecl()->field_begin(),
                            FieldEnd = RT->getDecl()->field_end();
    for (; Field != FieldEnd; ++Field) {
      if (Field->isUnnamedBitfield())
        continue;

      if (Field->getIdentifier() == FieldName)
        break;

      ++FieldIndex;
    }

    if (Field == FieldEnd) {
      // We did not find the field we're looking for. Produce a
      // suitable diagnostic and return a failure.
      DeclContext::lookup_result Lookup = RT->getDecl()->lookup(FieldName);
      if (Lookup.first == Lookup.second) {
        // Name lookup didn't find anything.
        SemaRef->Diag(D->getFieldLoc(), diag::err_field_designator_unknown)
          << FieldName << CurrentObjectType;
      } else {
        // Name lookup found something, but it wasn't a field.
        SemaRef->Diag(D->getFieldLoc(), diag::err_field_designator_nonfield)
          << FieldName;
        SemaRef->Diag((*Lookup.first)->getLocation(), 
                      diag::note_field_designator_found);
      }

      ++Index;
      return true;
    } else if (cast<RecordDecl>((*Field)->getDeclContext())
                 ->isAnonymousStructOrUnion()) {
      SemaRef->Diag(D->getFieldLoc(), diag::err_field_designator_anon_class)
        << FieldName
        << (cast<RecordDecl>((*Field)->getDeclContext())->isUnion()? 2 : 
            (int)SemaRef->getLangOptions().CPlusPlus);
      SemaRef->Diag((*Field)->getLocation(), diag::note_field_designator_found);
      ++Index;
      return true;
    }

    // All of the fields of a union are located at the same place in
    // the initializer list.
    if (RT->getDecl()->isUnion())
      FieldIndex = 0;

    // Update the designator with the field declaration.
    D->setField(*Field);
      
    // Make sure that our non-designated initializer list has space
    // for a subobject corresponding to this field.
    if (FieldIndex >= StructuredList->getNumInits())
      StructuredList->resizeInits(SemaRef->Context, FieldIndex + 1);

    // Recurse to check later designated subobjects.
    QualType FieldType = (*Field)->getType();
    unsigned newStructuredIndex = FieldIndex;
    if (CheckDesignatedInitializer(IList, DIE, ++D, FieldType, 0, 0, Index,
                                   StructuredList, newStructuredIndex))
      return true;

    // Find the position of the next field to be initialized in this
    // subobject.
    ++Field;
    ++FieldIndex;

    // If this the first designator, our caller will continue checking
    // the rest of this struct/class/union subobject.
    if (IsFirstDesignator) {
      if (NextField)
        *NextField = Field;
      StructuredIndex = FieldIndex;
      return false;
    }

    if (!FinishSubobjectInit)
      return false;

    // Check the remaining fields within this class/struct/union subobject.
    bool prevHadError = hadError;
    CheckStructUnionTypes(IList, CurrentObjectType, Field, false, Index,
                          StructuredList, FieldIndex);
    return hadError && !prevHadError;
  }

  // C99 6.7.8p6:
  //
  //   If a designator has the form
  //
  //      [ constant-expression ]
  //
  //   then the current object (defined below) shall have array
  //   type and the expression shall be an integer constant
  //   expression. If the array is of unknown size, any
  //   nonnegative value is valid.
  //
  // Additionally, cope with the GNU extension that permits
  // designators of the form
  //
  //      [ constant-expression ... constant-expression ]
  const ArrayType *AT = SemaRef->Context.getAsArrayType(CurrentObjectType);
  if (!AT) {
    SemaRef->Diag(D->getLBracketLoc(), diag::err_array_designator_non_array)
      << CurrentObjectType;
    ++Index;
    return true;
  }

  Expr *IndexExpr = 0;
  llvm::APSInt DesignatedStartIndex, DesignatedEndIndex;
  if (D->isArrayDesignator()) {
    IndexExpr = DIE->getArrayIndex(*D);
    
    bool ConstExpr 
      = IndexExpr->isIntegerConstantExpr(DesignatedStartIndex, SemaRef->Context);
    assert(ConstExpr && "Expression must be constant"); (void)ConstExpr;
    
    DesignatedEndIndex = DesignatedStartIndex;
  } else {
    assert(D->isArrayRangeDesignator() && "Need array-range designator");
    
    bool StartConstExpr
      = DIE->getArrayRangeStart(*D)->isIntegerConstantExpr(DesignatedStartIndex,
                                                           SemaRef->Context);
    assert(StartConstExpr && "Expression must be constant"); (void)StartConstExpr;

    bool EndConstExpr
      = DIE->getArrayRangeEnd(*D)->isIntegerConstantExpr(DesignatedEndIndex,
                                                         SemaRef->Context);
    assert(EndConstExpr && "Expression must be constant"); (void)EndConstExpr;
    
    IndexExpr = DIE->getArrayRangeEnd(*D);

    if (DesignatedStartIndex.getZExtValue() != DesignatedEndIndex.getZExtValue())
      SemaRef->Diag(D->getEllipsisLoc(), 
                    diag::warn_gnu_array_range_designator_side_effects)
        << SourceRange(D->getLBracketLoc(), D->getRBracketLoc());
  }

  if (isa<ConstantArrayType>(AT)) {
    llvm::APSInt MaxElements(cast<ConstantArrayType>(AT)->getSize(), false);
    DesignatedStartIndex.extOrTrunc(MaxElements.getBitWidth());
    DesignatedStartIndex.setIsUnsigned(MaxElements.isUnsigned());
    DesignatedEndIndex.extOrTrunc(MaxElements.getBitWidth());
    DesignatedEndIndex.setIsUnsigned(MaxElements.isUnsigned());
    if (DesignatedEndIndex >= MaxElements) {
      SemaRef->Diag(IndexExpr->getSourceRange().getBegin(),
                    diag::err_array_designator_too_large)
        << DesignatedEndIndex.toString(10) << MaxElements.toString(10)
        << IndexExpr->getSourceRange();
      ++Index;
      return true;
    }
  } else {
    // Make sure the bit-widths and signedness match.
    if (DesignatedStartIndex.getBitWidth() > DesignatedEndIndex.getBitWidth())
      DesignatedEndIndex.extend(DesignatedStartIndex.getBitWidth());
    else if (DesignatedStartIndex.getBitWidth() < DesignatedEndIndex.getBitWidth())
      DesignatedStartIndex.extend(DesignatedEndIndex.getBitWidth());
    DesignatedStartIndex.setIsUnsigned(true);
    DesignatedEndIndex.setIsUnsigned(true);
  }
  
  // Make sure that our non-designated initializer list has space
  // for a subobject corresponding to this array element.
  if (DesignatedEndIndex.getZExtValue() >= StructuredList->getNumInits())
    StructuredList->resizeInits(SemaRef->Context, 
                                DesignatedEndIndex.getZExtValue() + 1);

  // Repeatedly perform subobject initializations in the range
  // [DesignatedStartIndex, DesignatedEndIndex].

  // Move to the next designator
  unsigned ElementIndex = DesignatedStartIndex.getZExtValue();
  unsigned OldIndex = Index;
  ++D;
  while (DesignatedStartIndex <= DesignatedEndIndex) {
    // Recurse to check later designated subobjects.
    QualType ElementType = AT->getElementType();
    Index = OldIndex;
    if (CheckDesignatedInitializer(IList, DIE, D, ElementType, 0, 0, Index,
                                   StructuredList, ElementIndex,
                                   (DesignatedStartIndex == DesignatedEndIndex)))
      return true;

    // Move to the next index in the array that we'll be initializing.
    ++DesignatedStartIndex;
    ElementIndex = DesignatedStartIndex.getZExtValue();
  }

  // If this the first designator, our caller will continue checking
  // the rest of this array subobject.
  if (IsFirstDesignator) {
    if (NextElementIndex)
      *NextElementIndex = DesignatedStartIndex;
    StructuredIndex = ElementIndex;
    return false;
  }
  
  if (!FinishSubobjectInit)
    return false;

  // Check the remaining elements within this array subobject.
  bool prevHadError = hadError;
  CheckArrayType(IList, CurrentObjectType, DesignatedStartIndex, true, Index,
                 StructuredList, ElementIndex);
  return hadError && !prevHadError;  
}

// Get the structured initializer list for a subobject of type
// @p CurrentObjectType.
InitListExpr *
InitListChecker::getStructuredSubobjectInit(InitListExpr *IList, unsigned Index,
                                            QualType CurrentObjectType,
                                            InitListExpr *StructuredList,
                                            unsigned StructuredIndex,
                                            SourceRange InitRange) {
  Expr *ExistingInit = 0;
  if (!StructuredList)
    ExistingInit = SyntacticToSemantic[IList];
  else if (StructuredIndex < StructuredList->getNumInits())
    ExistingInit = StructuredList->getInit(StructuredIndex);
  
  if (InitListExpr *Result = dyn_cast_or_null<InitListExpr>(ExistingInit))
    return Result;

  if (ExistingInit) {
    // We are creating an initializer list that initializes the
    // subobjects of the current object, but there was already an
    // initialization that completely initialized the current
    // subobject, e.g., by a compound literal:
    // 
    // struct X { int a, b; };
    // struct X xs[] = { [0] = (struct X) { 1, 2 }, [0].b = 3 };
    // 
    // Here, xs[0].a == 0 and xs[0].b == 3, since the second,
    // designated initializer re-initializes the whole
    // subobject [0], overwriting previous initializers.
    SemaRef->Diag(InitRange.getBegin(), diag::warn_subobject_initializer_overrides)
      << InitRange;
    SemaRef->Diag(ExistingInit->getSourceRange().getBegin(), 
                  diag::note_previous_initializer)
      << /*FIXME:has side effects=*/0
      << ExistingInit->getSourceRange();
  }

  InitListExpr *Result 
    = new (SemaRef->Context) InitListExpr(SourceLocation(), 0, 0, 
                                          SourceLocation());
  Result->setType(CurrentObjectType);

  // Link this new initializer list into the structured initializer
  // lists.
  if (StructuredList)
    StructuredList->updateInit(StructuredIndex, Result);
  else {
    Result->setSyntacticForm(IList);
    SyntacticToSemantic[IList] = Result;
  }

  return Result;
}

/// Update the initializer at index @p StructuredIndex within the
/// structured initializer list to the value @p expr.
void InitListChecker::UpdateStructuredListElement(InitListExpr *StructuredList,
                                                  unsigned &StructuredIndex,
                                                  Expr *expr) {
  // No structured initializer list to update
  if (!StructuredList)
    return;

  if (Expr *PrevInit = StructuredList->updateInit(StructuredIndex, expr)) {
    // This initializer overwrites a previous initializer. Warn.
    SemaRef->Diag(expr->getSourceRange().getBegin(), 
                  diag::warn_initializer_overrides)
      << expr->getSourceRange();
    SemaRef->Diag(PrevInit->getSourceRange().getBegin(), 
                  diag::note_previous_initializer)
      << /*FIXME:has side effects=*/0
      << PrevInit->getSourceRange();
  }
  
  ++StructuredIndex;
}

/// Check that the given Index expression is a valid array designator
/// value. This is essentailly just a wrapper around
/// Expr::isIntegerConstantExpr that also checks for negative values
/// and produces a reasonable diagnostic if there is a
/// failure. Returns true if there was an error, false otherwise.  If
/// everything went okay, Value will receive the value of the constant
/// expression.
static bool 
CheckArrayDesignatorExpr(Sema &Self, Expr *Index, llvm::APSInt &Value) {
  SourceLocation Loc = Index->getSourceRange().getBegin();

  // Make sure this is an integer constant expression.
  if (!Index->isIntegerConstantExpr(Value, Self.Context, &Loc))
    return Self.Diag(Loc, diag::err_array_designator_nonconstant)
      << Index->getSourceRange();

  // Make sure this constant expression is non-negative.
  llvm::APSInt Zero(llvm::APSInt::getNullValue(Value.getBitWidth()), 
                    Value.isUnsigned());
  if (Value < Zero)
    return Self.Diag(Loc, diag::err_array_designator_negative)
      << Value.toString(10) << Index->getSourceRange();

  Value.setIsUnsigned(true);
  return false;
}

Sema::OwningExprResult Sema::ActOnDesignatedInitializer(Designation &Desig,
                                                        SourceLocation Loc,
                                                        bool UsedColonSyntax,
                                                        OwningExprResult Init) {
  typedef DesignatedInitExpr::Designator ASTDesignator;

  bool Invalid = false;
  llvm::SmallVector<ASTDesignator, 32> Designators;
  llvm::SmallVector<Expr *, 32> InitExpressions;

  // Build designators and check array designator expressions.
  for (unsigned Idx = 0; Idx < Desig.getNumDesignators(); ++Idx) {
    const Designator &D = Desig.getDesignator(Idx);
    switch (D.getKind()) {
    case Designator::FieldDesignator:
      Designators.push_back(ASTDesignator(D.getField(), D.getDotLoc(), 
                                          D.getFieldLoc()));
      break;

    case Designator::ArrayDesignator: {
      Expr *Index = static_cast<Expr *>(D.getArrayIndex());
      llvm::APSInt IndexValue;
      if (CheckArrayDesignatorExpr(*this, Index, IndexValue))
        Invalid = true;
      else {
        Designators.push_back(ASTDesignator(InitExpressions.size(),
                                            D.getLBracketLoc(), 
                                            D.getRBracketLoc()));
        InitExpressions.push_back(Index);
      }
      break;
    }

    case Designator::ArrayRangeDesignator: {
      Expr *StartIndex = static_cast<Expr *>(D.getArrayRangeStart());
      Expr *EndIndex = static_cast<Expr *>(D.getArrayRangeEnd());
      llvm::APSInt StartValue;
      llvm::APSInt EndValue;
      if (CheckArrayDesignatorExpr(*this, StartIndex, StartValue) ||
          CheckArrayDesignatorExpr(*this, EndIndex, EndValue))
        Invalid = true;
      else {
        // Make sure we're comparing values with the same bit width.
        if (StartValue.getBitWidth() > EndValue.getBitWidth())
          EndValue.extend(StartValue.getBitWidth());
        else if (StartValue.getBitWidth() < EndValue.getBitWidth())
          StartValue.extend(EndValue.getBitWidth());

        if (EndValue < StartValue) {
          Diag(D.getEllipsisLoc(), diag::err_array_designator_empty_range)
            << StartValue.toString(10) << EndValue.toString(10) 
            << StartIndex->getSourceRange() << EndIndex->getSourceRange();
          Invalid = true;
        } else {
          Designators.push_back(ASTDesignator(InitExpressions.size(),
                                              D.getLBracketLoc(), 
                                              D.getEllipsisLoc(),
                                              D.getRBracketLoc()));
          InitExpressions.push_back(StartIndex);
          InitExpressions.push_back(EndIndex);
        }
      }
      break;
    }
    }
  }

  if (Invalid || Init.isInvalid())
    return ExprError();

  // Clear out the expressions within the designation.
  Desig.ClearExprs(*this);

  DesignatedInitExpr *DIE
    = DesignatedInitExpr::Create(Context, &Designators[0], Designators.size(),
                                 &InitExpressions[0], InitExpressions.size(),
                                 Loc, UsedColonSyntax, 
                                 static_cast<Expr *>(Init.release()));
  return Owned(DIE);
}
