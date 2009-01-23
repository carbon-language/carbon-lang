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
#include "clang/Basic/Diagnostic.h"
#include <algorithm> // for std::count_if
#include <functional> // for std::mem_fun

using namespace clang;

InitListChecker::InitListChecker(Sema *S, InitListExpr *IL, QualType &T) {
  hadError = false;
  SemaRef = S;

  unsigned newIndex = 0;

  CheckExplicitInitList(IL, T, newIndex);
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
  const int InitializableMembers 
    = std::count_if(structDecl->field_begin(), structDecl->field_end(),
                    std::mem_fun(&FieldDecl::getDeclName));
  if (structDecl->isUnion())
    return std::min(InitializableMembers, 1);
  return InitializableMembers - structDecl->hasFlexibleArrayMember();
}

void InitListChecker::CheckImplicitInitList(InitListExpr *ParentIList, 
                                            QualType T, unsigned &Index) {
  llvm::SmallVector<Expr*, 4> InitExprs;
  int maxElements = 0;
  
  if (T->isArrayType())
    maxElements = numArrayElements(T);
  else if (T->isStructureType() || T->isUnionType())
    maxElements = numStructUnionElements(T);
  else if (T->isVectorType())
    maxElements = T->getAsVectorType()->getNumElements();
  else
    assert(0 && "CheckImplicitInitList(): Illegal type");

  if (maxElements == 0) {
    SemaRef->Diag(ParentIList->getInit(Index)->getLocStart(),
                  diag::err_implicit_empty_initializer);
    hadError = true;
    return;
  }

  // Check the element types *before* we create the implicit init list;
  // otherwise, we might end up taking the wrong number of elements
  unsigned NewIndex = Index;
  CheckListElementTypes(ParentIList, T, false, NewIndex);

  for (int i = 0; i < maxElements; ++i) {
    // Don't attempt to go past the end of the init list
    if (Index >= ParentIList->getNumInits())
      break;
    Expr* expr = ParentIList->getInit(Index);
    
    // Add the expr to the new implicit init list and remove if from the old.
    InitExprs.push_back(expr);
    ParentIList->removeInit(Index);
  }
  // Synthesize an "implicit" InitListExpr (marked by the invalid source locs).
  InitListExpr *ILE = new InitListExpr(SourceLocation(), 
                                       &InitExprs[0], InitExprs.size(), 
                                       SourceLocation(),
                                       ParentIList->hadDesignators());
  ILE->setType(T);

  // Modify the parent InitListExpr to point to the implicit InitListExpr.
  ParentIList->addInit(Index, ILE);
}

void InitListChecker::CheckExplicitInitList(InitListExpr *IList, QualType &T,
                                            unsigned &Index) {
  assert(IList->isExplicit() && "Illegal Implicit InitListExpr");

  CheckListElementTypes(IList, T, true, Index);
  IList->setType(T);
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
                                            unsigned &Index) {
  if (DeclType->isScalarType()) {
    CheckScalarType(IList, DeclType, 0, Index);
  } else if (DeclType->isVectorType()) {
    CheckVectorType(IList, DeclType, Index);
  } else if (DeclType->isAggregateType() || DeclType->isUnionType()) {
    if (DeclType->isStructureType() || DeclType->isUnionType()) {
      RecordDecl *RD = DeclType->getAsRecordType()->getDecl();
      CheckStructUnionTypes(IList, DeclType, RD->field_begin(), 
                            SubobjectIsDesignatorContext, Index);
    } else if (DeclType->isArrayType()) {
      llvm::APSInt Zero(
                      SemaRef->Context.getTypeSize(SemaRef->Context.getSizeType()),
                      false);
      CheckArrayType(IList, DeclType, Zero, SubobjectIsDesignatorContext, Index);
    }
    else
      assert(0 && "Aggregate that isn't a function or array?!");
  } else if (DeclType->isVoidType() || DeclType->isFunctionType()) {
    // This type is invalid, issue a diagnostic.
    Index++;
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
                                          Expr *expr,
                                          unsigned &Index) {
  if (InitListExpr *SubInitList = dyn_cast<InitListExpr>(expr)) {
    unsigned newIndex = 0;
    CheckExplicitInitList(SubInitList, ElemType, newIndex);
    Index++;
  } else if (StringLiteral *lit =
             SemaRef->IsStringLiteralInit(expr, ElemType)) {
    SemaRef->CheckStringLiteralInit(lit, ElemType);
    Index++;
  } else if (ElemType->isScalarType()) {
    CheckScalarType(IList, ElemType, expr, Index);
  } else if (expr->getType()->getAsRecordType() &&
             SemaRef->Context.typesAreCompatible(
               expr->getType().getUnqualifiedType(),
               ElemType.getUnqualifiedType())) {
    Index++;
    // FIXME: Add checking
  } else {
    CheckImplicitInitList(IList, ElemType, Index);
    Index++;
  }
}

void InitListChecker::CheckScalarType(InitListExpr *IList, QualType &DeclType,
                                      Expr *expr, unsigned &Index) {
  if (Index < IList->getNumInits()) {
    if (!expr)
      expr = IList->getInit(Index);
    if (isa<InitListExpr>(expr)) {
      SemaRef->Diag(IList->getLocStart(),
                    diag::err_many_braces_around_scalar_init)
        << IList->getSourceRange();
      hadError = true;
      ++Index;
      return;
    } else if (isa<DesignatedInitExpr>(expr)) {
      SemaRef->Diag(expr->getSourceRange().getBegin(), 
                    diag::err_designator_for_scalar_init)
        << DeclType << expr->getSourceRange();
      hadError = true;
      ++Index;
      return;
    }

    Expr *savExpr = expr; // Might be promoted by CheckSingleInitializer.
    if (SemaRef->CheckSingleInitializer(expr, DeclType, false))
      hadError = true; // types weren't compatible.
    else if (savExpr != expr) {
      // The type was promoted, update initializer list.
      if (DesignatedInitExpr *DIE 
            = dyn_cast<DesignatedInitExpr>(IList->getInit(Index)))
        DIE->setInit(expr);
      else
        IList->setInit(Index, expr);
    }
    ++Index;
  } else {
    SemaRef->Diag(IList->getLocStart(), diag::err_empty_scalar_initializer)
      << IList->getSourceRange();
    hadError = true;
    return;
  }
}

void InitListChecker::CheckVectorType(InitListExpr *IList, QualType DeclType, 
                                      unsigned &Index) {
  if (Index < IList->getNumInits()) {
    const VectorType *VT = DeclType->getAsVectorType();
    int maxElements = VT->getNumElements();
    QualType elementType = VT->getElementType();
    
    for (int i = 0; i < maxElements; ++i) {
      // Don't attempt to go past the end of the init list
      if (Index >= IList->getNumInits())
        break;
      CheckSubElementType(IList, elementType, IList->getInit(Index), Index);
    }
  }
}

void InitListChecker::CheckArrayType(InitListExpr *IList, QualType &DeclType, 
                                     llvm::APSInt elementIndex,
                                     bool SubobjectIsDesignatorContext, 
                                     unsigned &Index) {
  // Check for the special-case of initializing an array with a string.
  if (Index < IList->getNumInits()) {
    if (StringLiteral *lit = 
        SemaRef->IsStringLiteralInit(IList->getInit(Index), DeclType)) {
      SemaRef->CheckStringLiteralInit(lit, DeclType);
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
    return;
  }

  // We might know the maximum number of elements in advance.
  llvm::APSInt maxElements(elementIndex.getBitWidth(), elementIndex.isUnsigned());
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
                                     DeclType, 0, &elementIndex, Index)) {
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
    CheckSubElementType(IList, elementType, IList->getInit(Index), Index);
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
                                            unsigned &Index) {
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
                                     DeclType, &Field, 0, Index))
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

    if (!Field->getIdentifier()) {
      // Don't initialize unnamed fields, e.g. "int : 20;"
      ++Field;
      continue;
    }

    CheckSubElementType(IList, Field->getType(), IList->getInit(Index), Index);
    if (DeclType->isUnionType()) // FIXME: designated initializers?
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
/// @p DesignatedField or @p DesignatedIndex (whichever is
/// appropriate).
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
/// @returns true if there was an error, false otherwise.
bool 
InitListChecker::CheckDesignatedInitializer(InitListExpr *IList,
                                      DesignatedInitExpr *DIE, 
                                      DesignatedInitExpr::designators_iterator D,
                                      QualType &CurrentObjectType,
                                      RecordDecl::field_iterator *NextField,
                                      llvm::APSInt *NextElementIndex,
                                      unsigned &Index) {
  bool IsFirstDesignator = (D == DIE->designators_begin());

  if (D == DIE->designators_end()) {
    // Check the actual initialization for the designated object type.
    bool prevHadError = hadError;
    CheckSubElementType(IList, CurrentObjectType, DIE->getInit(), Index);
    return hadError && !prevHadError;
  }

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

    IdentifierInfo *FieldName = D->getFieldName();
    DeclContext::lookup_result Lookup = RT->getDecl()->lookup(FieldName);
    FieldDecl *DesignatedField = 0;
    if (Lookup.first == Lookup.second) {
      // Lookup did not find anything with this name.
      SemaRef->Diag(D->getFieldLoc(), diag::err_field_designator_unknown)
        << FieldName << CurrentObjectType;
    } else if (isa<FieldDecl>(*Lookup.first)) {
      // Name lookup found a field.
      DesignatedField = cast<FieldDecl>(*Lookup.first);
      // FIXME: Make sure this isn't a field in an anonymous
      // struct/union.
    } else {
      // Name lookup found something, but it wasn't a field.
      SemaRef->Diag(D->getFieldLoc(), diag::err_field_designator_nonfield)
        << FieldName;
      SemaRef->Diag((*Lookup.first)->getLocation(), 
                    diag::note_field_designator_found);
    }

    if (!DesignatedField) {
      ++Index;
      return true;
    }
        
    // Update the designator with the field declaration.
    D->setField(DesignatedField);
      
    // Recurse to check later designated subobjects.
    QualType FieldType = DesignatedField->getType();
    if (CheckDesignatedInitializer(IList, DIE, ++D, FieldType, 0, 0, Index))
      return true;

    // Find the position of the next field to be initialized in this
    // subobject.
    RecordDecl::field_iterator Field(DeclContext::decl_iterator(DesignatedField),
                                     RT->getDecl()->decls_end());
    ++Field;

    // If this the first designator, our caller will continue checking
    // the rest of this struct/class/union subobject.
    if (IsFirstDesignator) {
      if (NextField)
        *NextField = Field;
      return false;
    }

    // Check the remaining fields within this class/struct/union subobject.
    bool prevHadError = hadError;
    CheckStructUnionTypes(IList, CurrentObjectType, Field, false, Index);
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
  llvm::APSInt DesignatedIndex;
  if (D->isArrayDesignator())
    IndexExpr = DIE->getArrayIndex(*D);
  else {
    assert(D->isArrayRangeDesignator() && "Need array-range designator");
    IndexExpr = DIE->getArrayRangeEnd(*D);
  }

  bool ConstExpr 
    = IndexExpr->isIntegerConstantExpr(DesignatedIndex, SemaRef->Context);
  assert(ConstExpr && "Expression must be constant"); (void)ConstExpr;
  
  if (isa<ConstantArrayType>(AT)) {
    llvm::APSInt MaxElements(cast<ConstantArrayType>(AT)->getSize(), false);
    DesignatedIndex.extOrTrunc(MaxElements.getBitWidth());
    DesignatedIndex.setIsUnsigned(MaxElements.isUnsigned());
    if (DesignatedIndex >= MaxElements) {
      SemaRef->Diag(IndexExpr->getSourceRange().getBegin(),
                    diag::err_array_designator_too_large)
        << DesignatedIndex.toString(10) << MaxElements.toString(10)
        << IndexExpr->getSourceRange();
      ++Index;
      return true;
    }
  }
  
  // Recurse to check later designated subobjects.
  QualType ElementType = AT->getElementType();
  if (CheckDesignatedInitializer(IList, DIE, ++D, ElementType, 0, 0, Index))
    return true;

  // Move to the next index in the array that we'll be initializing.
  ++DesignatedIndex;

  // If this the first designator, our caller will continue checking
  // the rest of this array subobject.
  if (IsFirstDesignator) {
    if (NextElementIndex)
      *NextElementIndex = DesignatedIndex;
    return false;
  }
    
  // Check the remaining elements within this array subobject.
  bool prevHadError = hadError;
  CheckArrayType(IList, CurrentObjectType, DesignatedIndex, true, Index);
  return hadError && !prevHadError;  
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
      else if (EndValue < StartValue) {
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
