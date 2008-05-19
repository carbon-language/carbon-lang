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
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"

namespace clang {

InitListChecker::InitListChecker(Sema *S, InitListExpr *IL, QualType &T) {
  hadError = false;
  SemaRef = S;
  
  unsigned newIndex = 0;
    
  CheckExplicitInitList(IL, T, newIndex);
      
  if (!hadError && (newIndex < IL->getNumInits())) {
    // We have leftover initializers; warn
    SemaRef->Diag(IL->getInit(newIndex)->getLocStart(), 
                  diag::warn_excess_initializers, 
                  IL->getInit(newIndex)->getSourceRange());
  }
}

int InitListChecker::numArrayElements(QualType DeclType) {
  int maxElements;
  if (DeclType->isIncompleteArrayType()) {
    // FIXME: use a proper constant
    maxElements = 0x7FFFFFFF;
  } else if (const VariableArrayType *VAT =
             DeclType->getAsVariableArrayType()) {
    // Check for VLAs; in standard C it would be possible to check this
    // earlier, but I don't know where clang accepts VLAs (gcc accepts
    // them in all sorts of strange places).
    SemaRef->Diag(VAT->getSizeExpr()->getLocStart(),
                  diag::err_variable_object_no_init,
                  VAT->getSizeExpr()->getSourceRange());
    hadError = true;
    maxElements = 0x7FFFFFFF;
  } else {
    const ConstantArrayType *CAT = DeclType->getAsConstantArrayType();
    maxElements = static_cast<int>(CAT->getSize().getZExtValue());
  }
  return maxElements;
}

int InitListChecker::numStructUnionElements(QualType DeclType) {
  RecordDecl *structDecl = DeclType->getAsRecordType()->getDecl();
  if (structDecl->getKind() == Decl::Union)
    return std::min(structDecl->getNumMembers(), 1);
  return structDecl->getNumMembers() - structDecl->hasFlexibleArrayMember();
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

  // Check the element types *before* we create the implicit init list;
  // otherwise, we might end up taking the wrong number of elements
  unsigned NewIndex = Index;
  CheckListElementTypes(ParentIList, T, NewIndex);

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
                                       SourceLocation());
  ILE->setType(T);
  
  // Modify the parent InitListExpr to point to the implicit InitListExpr.
  ParentIList->addInit(Index, ILE);
}

void InitListChecker::CheckExplicitInitList(InitListExpr *IList, QualType &T,
                                            unsigned &Index) {
  //assert(IList->isExplicit() && "Illegal Implicit InitListExpr");
  if (IList->isExplicit() && T->isScalarType())
      SemaRef->Diag(IList->getLocStart(), diag::warn_braces_around_scalar_init, 
                    IList->getSourceRange());
  CheckListElementTypes(IList, T, Index);
  IList->setType(T);
}

void InitListChecker::CheckListElementTypes(InitListExpr *IList,
                                            QualType &DeclType, 
                                            unsigned &Index) {
  if (DeclType->isScalarType())
    CheckScalarType(IList, DeclType, Index);
  else if (DeclType->isVectorType())
    CheckVectorType(IList, DeclType, Index);
  else if (DeclType->isAggregateType() || DeclType->isUnionType()) {
    if (DeclType->isStructureType() || DeclType->isUnionType())
      CheckStructUnionTypes(IList, DeclType, Index);
    else if (DeclType->isArrayType()) 
      CheckArrayType(IList, DeclType, Index);
    else
      assert(0 && "Aggregate that isn't a function or array?!");
  } else {
    // In C, all types are either scalars or aggregates, but
    // additional handling is needed here for C++ (and possibly others?). 
    assert(0 && "Unsupported initializer type");
  }
}

void InitListChecker::CheckSubElementType(InitListExpr *IList,
                                          QualType ElemType, 
                                          unsigned &Index) {
  Expr* expr = IList->getInit(Index);
  if (ElemType->isScalarType()) {
    CheckScalarType(IList, ElemType, Index);
  } else if (StringLiteral *lit =
             SemaRef->IsStringLiteralInit(expr, ElemType)) {
    SemaRef->CheckStringLiteralInit(lit, ElemType);
    Index++;
  } else if (InitListExpr *SubInitList = dyn_cast<InitListExpr>(expr)) {
    unsigned newIndex = 0;
    CheckExplicitInitList(SubInitList, ElemType, newIndex);
    Index++;
  } else if (expr->getType()->getAsRecordType() &&
             SemaRef->Context.typesAreCompatible(
                  IList->getInit(Index)->getType(), ElemType)) {
    Index++;
    // FIXME: Add checking
  } else {
    CheckImplicitInitList(IList, ElemType, Index);
    Index++;
  }
}

void InitListChecker::CheckScalarType(InitListExpr *IList, QualType &DeclType, 
                                      unsigned &Index) {
  if (Index < IList->getNumInits()) {
    Expr* expr = IList->getInit(Index);
    if (InitListExpr *SubInitList = dyn_cast<InitListExpr>(expr)) {
      unsigned newIndex = 0;
      CheckExplicitInitList(SubInitList, DeclType, newIndex);
    } else {
      Expr *savExpr = expr; // Might be promoted by CheckSingleInitializer.
      if (SemaRef->CheckSingleInitializer(expr, DeclType))
        hadError |= true; // types weren't compatible.
      else if (savExpr != expr)
        // The type was promoted, update initializer list.
        IList->setInit(Index, expr);
    }
    ++Index;
  }
  // FIXME: Should an error be reported for empty initializer list + scalar?
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
      CheckSubElementType(IList, elementType, Index);
    }
  }
}

void InitListChecker::CheckArrayType(InitListExpr *IList, QualType &DeclType, 
                                     unsigned &Index) {
  // Check for the special-case of initializing an array with a string.
  if (Index < IList->getNumInits()) {
    if (StringLiteral *lit = 
        SemaRef->IsStringLiteralInit(IList->getInit(Index), DeclType)) {
      SemaRef->CheckStringLiteralInit(lit, DeclType);
      ++Index;
#if 0
      if (IList->isExplicit() && Index < IList->getNumInits()) {
        // We have leftover initializers; warn
        SemaRef->Diag(IList->getInit(Index)->getLocStart(), 
                      diag::err_excess_initializers_in_char_array_initializer, 
                      IList->getInit(Index)->getSourceRange());
      }
#endif
      return;
    }
  }
  int maxElements = numArrayElements(DeclType);
  QualType elementType = DeclType->getAsArrayType()->getElementType();
  int numElements = 0;
  for (int i = 0; i < maxElements; ++i, ++numElements) {
    // Don't attempt to go past the end of the init list
    if (Index >= IList->getNumInits())
      break;
    CheckSubElementType(IList, elementType, Index);
  }
  if (DeclType->isIncompleteArrayType()) {
    // If this is an incomplete array type, the actual type needs to
    // be calculated here
    if (numElements == 0) {
      // Sizing an array implicitly to zero is not allowed
      // (It could in theory be allowed, but it doesn't really matter.)
      SemaRef->Diag(IList->getLocStart(),
                    diag::err_at_least_one_initializer_needed_to_size_array);
      hadError = true;
    } else {
      llvm::APSInt ConstVal(32);
      ConstVal = numElements;
      DeclType = SemaRef->Context.getConstantArrayType(elementType, ConstVal, 
                                                       ArrayType::Normal, 0);
    }
  }
}

void InitListChecker::CheckStructUnionTypes(InitListExpr *IList, 
                                            QualType DeclType, 
                                            unsigned &Index) {
  RecordDecl* structDecl = DeclType->getAsRecordType()->getDecl();
    
  // If the record is invalid, some of it's members are invalid. To avoid
  // confusion, we forgo checking the intializer for the entire record.
  if (structDecl->isInvalidDecl()) {
    hadError = true;
    return;
  }    
  // If structDecl is a forward declaration, this loop won't do anything;
  // That's okay, because an error should get printed out elsewhere. It
  // might be worthwhile to skip over the rest of the initializer, though.
  int numMembers = numStructUnionElements(DeclType);
  for (int i = 0; i < numMembers; i++) {
    // Don't attempt to go past the end of the init list
    if (Index >= IList->getNumInits())
      break;
    FieldDecl * curField = structDecl->getMember(i);
    if (!curField->getIdentifier()) {
      // Don't initialize unnamed fields, e.g. "int : 20;"
      continue;
    }
    CheckSubElementType(IList, curField->getType(), Index);
    if (DeclType->isUnionType())
      break;
  }
  // FIXME: Implement flexible array initialization GCC extension (it's a 
  // really messy extension to implement, unfortunately...the necessary
  // information isn't actually even here!)
}
} // end namespace clang

