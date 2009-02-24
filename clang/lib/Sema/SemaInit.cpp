//===--- SemaInit.cpp - Semantic Analysis for Initializers ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements semantic analysis for initializers. The main entry
// point is Sema::CheckInitList(), but all of the work is performed
// within the InitListChecker class.
//
// This file also includes some miscellaneous other initialization checking
// code that is part of Sema.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "clang/Parse/Designator.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include <map>
using namespace clang;

//===----------------------------------------------------------------------===//
// Sema Initialization Checking
//===----------------------------------------------------------------------===//

StringLiteral *Sema::IsStringLiteralInit(Expr *Init, QualType DeclType) {
  if (const ArrayType *AT = Context.getAsArrayType(DeclType))
    if (AT->getElementType()->isCharType())
      return dyn_cast<StringLiteral>(Init->IgnoreParens());
  return 0;
}

bool Sema::CheckSingleInitializer(Expr *&Init, QualType DeclType, 
                                  bool DirectInit) {  
  // Get the type before calling CheckSingleAssignmentConstraints(), since
  // it can promote the expression.
  QualType InitType = Init->getType(); 
  
  if (getLangOptions().CPlusPlus) {
    // FIXME: I dislike this error message. A lot.
    if (PerformImplicitConversion(Init, DeclType, "initializing", DirectInit))
      return Diag(Init->getSourceRange().getBegin(),
                  diag::err_typecheck_convert_incompatible)
      << DeclType << Init->getType() << "initializing" 
      << Init->getSourceRange();
    
    return false;
  }
  
  AssignConvertType ConvTy = CheckSingleAssignmentConstraints(DeclType, Init);
  return DiagnoseAssignmentResult(ConvTy, Init->getLocStart(), DeclType,
                                  InitType, Init, "initializing");
}

bool Sema::CheckStringLiteralInit(StringLiteral *strLiteral, QualType &DeclT) {
  const ArrayType *AT = Context.getAsArrayType(DeclT);
  
  if (const IncompleteArrayType *IAT = dyn_cast<IncompleteArrayType>(AT)) {
    // C99 6.7.8p14. We have an array of character type with unknown size 
    // being initialized to a string literal.
    llvm::APSInt ConstVal(32);
    ConstVal = strLiteral->getByteLength() + 1;
    // Return a new array type (C99 6.7.8p22).
    DeclT = Context.getConstantArrayType(IAT->getElementType(), ConstVal, 
                                         ArrayType::Normal, 0);
  } else {
    const ConstantArrayType *CAT = cast<ConstantArrayType>(AT);
    // C99 6.7.8p14. We have an array of character type with known size.
    // FIXME: Avoid truncation for 64-bit length strings.
    if (strLiteral->getByteLength() > (unsigned)CAT->getSize().getZExtValue())
      Diag(strLiteral->getSourceRange().getBegin(),
           diag::warn_initializer_string_for_char_array_too_long)
      << strLiteral->getSourceRange();
  }
  // Set type from "char *" to "constant array of char".
  strLiteral->setType(DeclT);
  // For now, we always return false (meaning success).
  return false;
}

bool Sema::CheckInitializerTypes(Expr *&Init, QualType &DeclType,
                                 SourceLocation InitLoc,
                                 DeclarationName InitEntity,
                                 bool DirectInit) {
  if (DeclType->isDependentType() || Init->isTypeDependent())
    return false;
  
  // C++ [dcl.init.ref]p1:
  //   A variable declared to be a T&, that is "reference to type T"
  //   (8.3.2), shall be initialized by an object, or function, of
  //   type T or by an object that can be converted into a T.
  if (DeclType->isReferenceType())
    return CheckReferenceInit(Init, DeclType, 0, false, DirectInit);
  
  // C99 6.7.8p3: The type of the entity to be initialized shall be an array
  // of unknown size ("[]") or an object type that is not a variable array type.
  if (const VariableArrayType *VAT = Context.getAsVariableArrayType(DeclType))
    return Diag(InitLoc,  diag::err_variable_object_no_init)
    << VAT->getSizeExpr()->getSourceRange();
  
  InitListExpr *InitList = dyn_cast<InitListExpr>(Init);
  if (!InitList) {
    // FIXME: Handle wide strings
    if (StringLiteral *StrLiteral = IsStringLiteralInit(Init, DeclType))
      return CheckStringLiteralInit(StrLiteral, DeclType);
    
    // C++ [dcl.init]p14:
    //   -- If the destination type is a (possibly cv-qualified) class
    //      type:
    if (getLangOptions().CPlusPlus && DeclType->isRecordType()) {
      QualType DeclTypeC = Context.getCanonicalType(DeclType);
      QualType InitTypeC = Context.getCanonicalType(Init->getType());
      
      //   -- If the initialization is direct-initialization, or if it is
      //      copy-initialization where the cv-unqualified version of the
      //      source type is the same class as, or a derived class of, the
      //      class of the destination, constructors are considered.
      if ((DeclTypeC.getUnqualifiedType() == InitTypeC.getUnqualifiedType()) ||
          IsDerivedFrom(InitTypeC, DeclTypeC)) {
        CXXConstructorDecl *Constructor 
        = PerformInitializationByConstructor(DeclType, &Init, 1,
                                             InitLoc, Init->getSourceRange(),
                                             InitEntity, 
                                             DirectInit? IK_Direct : IK_Copy);
        return Constructor == 0;
      }
      
      //   -- Otherwise (i.e., for the remaining copy-initialization
      //      cases), user-defined conversion sequences that can
      //      convert from the source type to the destination type or
      //      (when a conversion function is used) to a derived class
      //      thereof are enumerated as described in 13.3.1.4, and the
      //      best one is chosen through overload resolution
      //      (13.3). If the conversion cannot be done or is
      //      ambiguous, the initialization is ill-formed. The
      //      function selected is called with the initializer
      //      expression as its argument; if the function is a
      //      constructor, the call initializes a temporary of the
      //      destination type.
      // FIXME: We're pretending to do copy elision here; return to
      // this when we have ASTs for such things.
      if (!PerformImplicitConversion(Init, DeclType, "initializing"))
        return false;
      
      if (InitEntity)
        return Diag(InitLoc, diag::err_cannot_initialize_decl)
        << InitEntity << (int)(Init->isLvalue(Context) == Expr::LV_Valid)
        << Init->getType() << Init->getSourceRange();
      else
        return Diag(InitLoc, diag::err_cannot_initialize_decl_noname)
        << DeclType << (int)(Init->isLvalue(Context) == Expr::LV_Valid)
        << Init->getType() << Init->getSourceRange();
    }
    
    // C99 6.7.8p16.
    if (DeclType->isArrayType())
      return Diag(Init->getLocStart(), diag::err_array_init_list_required)
      << Init->getSourceRange();
    
    return CheckSingleInitializer(Init, DeclType, DirectInit);
  } 
  
  bool hadError = CheckInitList(InitList, DeclType);
  Init = InitList;
  return hadError;
}

//===----------------------------------------------------------------------===//
// Semantic checking for initializer lists.
//===----------------------------------------------------------------------===//

/// @brief Semantic checking for initializer lists.
///
/// The InitListChecker class contains a set of routines that each
/// handle the initialization of a certain kind of entity, e.g.,
/// arrays, vectors, struct/union types, scalars, etc. The
/// InitListChecker itself performs a recursive walk of the subobject
/// structure of the type to be initialized, while stepping through
/// the initializer list one element at a time. The IList and Index
/// parameters to each of the Check* routines contain the active
/// (syntactic) initializer list and the index into that initializer
/// list that represents the current initializer. Each routine is
/// responsible for moving that Index forward as it consumes elements.
///
/// Each Check* routine also has a StructuredList/StructuredIndex
/// arguments, which contains the current the "structured" (semantic)
/// initializer list and the index into that initializer list where we
/// are copying initializers as we map them over to the semantic
/// list. Once we have completed our recursive walk of the subobject
/// structure, we will have constructed a full semantic initializer
/// list.
///
/// C99 designators cause changes in the initializer list traversal,
/// because they make the initialization "jump" into a specific
/// subobject and then continue the initialization from that
/// point. CheckDesignatedInitializer() recursively steps into the
/// designated subobject and manages backing out the recursion to
/// initialize the subobjects after the one designated.
namespace clang {
class InitListChecker {
  Sema *SemaRef;
  bool hadError;
  std::map<InitListExpr *, InitListExpr *> SyntacticToSemantic;
  InitListExpr *FullyStructuredList;
  
  void CheckImplicitInitList(InitListExpr *ParentIList, QualType T, 
                             unsigned &Index, InitListExpr *StructuredList,
                             unsigned &StructuredIndex,
                             bool TopLevelObject = false);
  void CheckExplicitInitList(InitListExpr *IList, QualType &T,
                             unsigned &Index, InitListExpr *StructuredList,
                             unsigned &StructuredIndex,
                             bool TopLevelObject = false);
  void CheckListElementTypes(InitListExpr *IList, QualType &DeclType, 
                             bool SubobjectIsDesignatorContext, 
                             unsigned &Index,
                             InitListExpr *StructuredList,
                             unsigned &StructuredIndex,
                             bool TopLevelObject = false);
  void CheckSubElementType(InitListExpr *IList, QualType ElemType, 
                           unsigned &Index,
                           InitListExpr *StructuredList,
                           unsigned &StructuredIndex);
  void CheckScalarType(InitListExpr *IList, QualType DeclType, 
                       unsigned &Index,
                       InitListExpr *StructuredList,
                       unsigned &StructuredIndex);
  void CheckReferenceType(InitListExpr *IList, QualType DeclType, 
                          unsigned &Index,
                          InitListExpr *StructuredList,
                          unsigned &StructuredIndex);
  void CheckVectorType(InitListExpr *IList, QualType DeclType, unsigned &Index,
                       InitListExpr *StructuredList,
                       unsigned &StructuredIndex);
  void CheckStructUnionTypes(InitListExpr *IList, QualType DeclType, 
                             RecordDecl::field_iterator Field, 
                             bool SubobjectIsDesignatorContext, unsigned &Index,
                             InitListExpr *StructuredList,
                             unsigned &StructuredIndex,
                             bool TopLevelObject = false);
  void CheckArrayType(InitListExpr *IList, QualType &DeclType, 
                      llvm::APSInt elementIndex, 
                      bool SubobjectIsDesignatorContext, unsigned &Index,
                      InitListExpr *StructuredList,
                      unsigned &StructuredIndex);
  bool CheckDesignatedInitializer(InitListExpr *IList, DesignatedInitExpr *DIE, 
                                  DesignatedInitExpr::designators_iterator D,
                                  QualType &CurrentObjectType, 
                                  RecordDecl::field_iterator *NextField,
                                  llvm::APSInt *NextElementIndex,
                                  unsigned &Index,
                                  InitListExpr *StructuredList,
                                  unsigned &StructuredIndex,
                                  bool FinishSubobjectInit,
                                  bool TopLevelObject);
  InitListExpr *getStructuredSubobjectInit(InitListExpr *IList, unsigned Index,
                                           QualType CurrentObjectType,
                                           InitListExpr *StructuredList,
                                           unsigned StructuredIndex,
                                           SourceRange InitRange);
  void UpdateStructuredListElement(InitListExpr *StructuredList,
                                   unsigned &StructuredIndex,
                                   Expr *expr);
  int numArrayElements(QualType DeclType);
  int numStructUnionElements(QualType DeclType);

  void FillInValueInitializations(InitListExpr *ILE);
public:
  InitListChecker(Sema *S, InitListExpr *IL, QualType &T);
  bool HadError() { return hadError; }

  // @brief Retrieves the fully-structured initializer list used for
  // semantic analysis and code generation.
  InitListExpr *getFullyStructuredList() const { return FullyStructuredList; }
};
}

/// Recursively replaces NULL values within the given initializer list
/// with expressions that perform value-initialization of the
/// appropriate type.
void InitListChecker::FillInValueInitializations(InitListExpr *ILE) {
  assert((ILE->getType() != SemaRef->Context.VoidTy) && 
         "Should not have void type");
  SourceLocation Loc = ILE->getSourceRange().getBegin();
  if (ILE->getSyntacticForm())
    Loc = ILE->getSyntacticForm()->getSourceRange().getBegin();
  
  if (const RecordType *RType = ILE->getType()->getAsRecordType()) {
    unsigned Init = 0, NumInits = ILE->getNumInits();
    for (RecordDecl::field_iterator Field = RType->getDecl()->field_begin(),
                                 FieldEnd = RType->getDecl()->field_end();
         Field != FieldEnd; ++Field) {
      if (Field->isUnnamedBitfield())
        continue;

      if (Init >= NumInits || !ILE->getInit(Init)) {
        if (Field->getType()->isReferenceType()) {
          // C++ [dcl.init.aggr]p9:
          //   If an incomplete or empty initializer-list leaves a
          //   member of reference type uninitialized, the program is
          //   ill-formed. 
          SemaRef->Diag(Loc, diag::err_init_reference_member_uninitialized)
            << Field->getType()
            << ILE->getSyntacticForm()->getSourceRange();
          SemaRef->Diag(Field->getLocation(), 
                        diag::note_uninit_reference_member);
          hadError = true;
          return;
        } else if (SemaRef->CheckValueInitialization(Field->getType(), Loc)) {
          hadError = true;
          return;
        }

        // FIXME: If value-initialization involves calling a
        // constructor, should we make that call explicit in the
        // representation (even when it means extending the
        // initializer list)?
        if (Init < NumInits && !hadError)
          ILE->setInit(Init, 
              new (SemaRef->Context) ImplicitValueInitExpr(Field->getType()));
      } else if (InitListExpr *InnerILE 
                 = dyn_cast<InitListExpr>(ILE->getInit(Init)))
        FillInValueInitializations(InnerILE);
      ++Init;

      // Only look at the first initialization of a union.
      if (RType->getDecl()->isUnion())
        break;
    }

    return;
  } 

  QualType ElementType;
  
  unsigned NumInits = ILE->getNumInits();
  unsigned NumElements = NumInits;
  if (const ArrayType *AType = SemaRef->Context.getAsArrayType(ILE->getType())) {
    ElementType = AType->getElementType();
    if (const ConstantArrayType *CAType = dyn_cast<ConstantArrayType>(AType))
      NumElements = CAType->getSize().getZExtValue();
  } else if (const VectorType *VType = ILE->getType()->getAsVectorType()) {
    ElementType = VType->getElementType();
    NumElements = VType->getNumElements();
  } else 
    ElementType = ILE->getType();
  
  for (unsigned Init = 0; Init != NumElements; ++Init) {
    if (Init >= NumInits || !ILE->getInit(Init)) {
      if (SemaRef->CheckValueInitialization(ElementType, Loc)) {
        hadError = true;
        return;
      }

      // FIXME: If value-initialization involves calling a
      // constructor, should we make that call explicit in the
      // representation (even when it means extending the
      // initializer list)?
      if (Init < NumInits && !hadError)
        ILE->setInit(Init, 
                     new (SemaRef->Context) ImplicitValueInitExpr(ElementType));
    }
    else if (InitListExpr *InnerILE =dyn_cast<InitListExpr>(ILE->getInit(Init)))
      FillInValueInitializations(InnerILE);
  }
}


InitListChecker::InitListChecker(Sema *S, InitListExpr *IL, QualType &T) {
  hadError = false;
  SemaRef = S;

  unsigned newIndex = 0;
  unsigned newStructuredIndex = 0;
  FullyStructuredList 
    = getStructuredSubobjectInit(IL, newIndex, T, 0, 0, SourceRange());
  CheckExplicitInitList(IL, T, newIndex, FullyStructuredList, newStructuredIndex,
                        /*TopLevelObject=*/true);

  if (!hadError)
    FillInValueInitializations(FullyStructuredList);
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
                                            unsigned &StructuredIndex,
                                            bool TopLevelObject) {
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
  unsigned StartIndex = Index;
  CheckListElementTypes(ParentIList, T, false, Index,
                        StructuredSubobjectInitList, 
                        StructuredSubobjectInitIndex,
                        TopLevelObject);
  unsigned EndIndex = (Index == StartIndex? StartIndex : Index - 1);
  
  // Update the structured sub-object initialize so that it's ending
  // range corresponds with the end of the last initializer it used.
  if (EndIndex < ParentIList->getNumInits()) {
    SourceLocation EndLoc 
      = ParentIList->getInit(EndIndex)->getSourceRange().getEnd();
    StructuredSubobjectInitList->setRBraceLoc(EndLoc);
  }
}

void InitListChecker::CheckExplicitInitList(InitListExpr *IList, QualType &T,
                                            unsigned &Index,
                                            InitListExpr *StructuredList,
                                            unsigned &StructuredIndex,
                                            bool TopLevelObject) {
  assert(IList->isExplicit() && "Illegal Implicit InitListExpr");
  SyntacticToSemantic[IList] = StructuredList;
  StructuredList->setSyntacticForm(IList);
  CheckListElementTypes(IList, T, true, Index, StructuredList, 
                        StructuredIndex, TopLevelObject);
  IList->setType(T);
  StructuredList->setType(T);
  if (hadError)
    return;

  if (Index < IList->getNumInits()) {
    // We have leftover initializers
    if (IList->getNumInits() > 0 &&
        SemaRef->IsStringLiteralInit(IList->getInit(Index), T)) {
      unsigned DK = diag::warn_excess_initializers_in_char_array_initializer;
      if (SemaRef->getLangOptions().CPlusPlus)
        DK = diag::err_excess_initializers_in_char_array_initializer;
      // Special-case
      SemaRef->Diag(IList->getInit(Index)->getLocStart(), DK)
        << IList->getInit(Index)->getSourceRange();
      hadError = true; 
    } else if (!T->isIncompleteType()) {
      // Don't complain for incomplete types, since we'll get an error
      // elsewhere
      QualType CurrentObjectType = StructuredList->getType();
      int initKind = 
        CurrentObjectType->isArrayType()? 0 :
        CurrentObjectType->isVectorType()? 1 :
        CurrentObjectType->isScalarType()? 2 :
        CurrentObjectType->isUnionType()? 3 :
        4;

      unsigned DK = diag::warn_excess_initializers;
      if (SemaRef->getLangOptions().CPlusPlus)
          DK = diag::err_excess_initializers;

      SemaRef->Diag(IList->getInit(Index)->getLocStart(), DK)
        << initKind << IList->getInit(Index)->getSourceRange();
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
                                            unsigned &StructuredIndex,
                                            bool TopLevelObject) {
  if (DeclType->isScalarType()) {
    CheckScalarType(IList, DeclType, Index, StructuredList, StructuredIndex);
  } else if (DeclType->isVectorType()) {
    CheckVectorType(IList, DeclType, Index, StructuredList, StructuredIndex);
  } else if (DeclType->isAggregateType()) {
    if (DeclType->isRecordType()) {
      RecordDecl *RD = DeclType->getAsRecordType()->getDecl();
      CheckStructUnionTypes(IList, DeclType, RD->field_begin(), 
                            SubobjectIsDesignatorContext, Index,
                            StructuredList, StructuredIndex,
                            TopLevelObject);
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
  } else if (DeclType->isRecordType()) {
    // C++ [dcl.init]p14:
    //   [...] If the class is an aggregate (8.5.1), and the initializer
    //   is a brace-enclosed list, see 8.5.1.
    //
    // Note: 8.5.1 is handled below; here, we diagnose the case where
    // we have an initializer list and a destination type that is not
    // an aggregate.
    // FIXME: In C++0x, this is yet another form of initialization.
    SemaRef->Diag(IList->getLocStart(), diag::err_init_non_aggr_init_list)
      << DeclType << IList->getSourceRange();
    hadError = true;
  } else if (DeclType->isReferenceType()) {
    CheckReferenceType(IList, DeclType, Index, StructuredList, StructuredIndex);
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
  } else if (ElemType->isReferenceType()) {
    CheckReferenceType(IList, ElemType, Index, StructuredList, StructuredIndex);
  } else {
    if (SemaRef->getLangOptions().CPlusPlus) {
      // C++ [dcl.init.aggr]p12:
      //   All implicit type conversions (clause 4) are considered when
      //   initializing the aggregate member with an ini- tializer from
      //   an initializer-list. If the initializer can initialize a
      //   member, the member is initialized. [...]
      ImplicitConversionSequence ICS 
        = SemaRef->TryCopyInitialization(expr, ElemType);
      if (ICS.ConversionKind != ImplicitConversionSequence::BadConversion) {
        if (SemaRef->PerformImplicitConversion(expr, ElemType, ICS, 
                                               "initializing"))
          hadError = true;
        UpdateStructuredListElement(StructuredList, StructuredIndex, expr);
        ++Index;
        return;
      }

      // Fall through for subaggregate initialization
    } else {
      // C99 6.7.8p13: 
      //
      //   The initializer for a structure or union object that has
      //   automatic storage duration shall be either an initializer
      //   list as described below, or a single expression that has
      //   compatible structure or union type. In the latter case, the
      //   initial value of the object, including unnamed members, is
      //   that of the expression.
      QualType ExprType = SemaRef->Context.getCanonicalType(expr->getType());
      QualType ElemTypeCanon = SemaRef->Context.getCanonicalType(ElemType);
      if (SemaRef->Context.typesAreCompatible(ExprType.getUnqualifiedType(),
                                          ElemTypeCanon.getUnqualifiedType())) {
        UpdateStructuredListElement(StructuredList, StructuredIndex, expr);
        ++Index;
        return;
      }

      // Fall through for subaggregate initialization
    }

    // C++ [dcl.init.aggr]p12:
    // 
    //   [...] Otherwise, if the member is itself a non-empty
    //   subaggregate, brace elision is assumed and the initializer is
    //   considered for the initialization of the first member of
    //   the subaggregate.
    if (ElemType->isAggregateType() || ElemType->isVectorType()) {
      CheckImplicitInitList(IList, ElemType, Index, StructuredList, 
                            StructuredIndex);
      ++StructuredIndex;
    } else {
      // We cannot initialize this element, so let
      // PerformCopyInitialization produce the appropriate diagnostic.
      SemaRef->PerformCopyInitialization(expr, ElemType, "initializing");
      hadError = true;
      ++Index;
      ++StructuredIndex;
    }
  }
}

void InitListChecker::CheckScalarType(InitListExpr *IList, QualType DeclType,
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

void InitListChecker::CheckReferenceType(InitListExpr *IList, QualType DeclType,
                                         unsigned &Index,
                                         InitListExpr *StructuredList,
                                         unsigned &StructuredIndex) {
  if (Index < IList->getNumInits()) {
    Expr *expr = IList->getInit(Index);
    if (isa<InitListExpr>(expr)) {
      SemaRef->Diag(IList->getLocStart(), diag::err_init_non_aggr_init_list)
        << DeclType << IList->getSourceRange();
      hadError = true;
      ++Index;
      ++StructuredIndex;
      return;
    } 

    Expr *savExpr = expr; // Might be promoted by CheckSingleInitializer.
    if (SemaRef->CheckReferenceInit(expr, DeclType))
      hadError = true;
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
    // FIXME: It would be wonderful if we could point at the actual
    // member. In general, it would be useful to pass location
    // information down the stack, so that we know the location (or
    // decl) of the "current object" being initialized.
    SemaRef->Diag(IList->getLocStart(), 
                  diag::err_init_reference_member_uninitialized)
      << DeclType
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
                                     StructuredList, StructuredIndex, true,
                                     false)) {
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
                                            unsigned &StructuredIndex,
                                            bool TopLevelObject) {
  RecordDecl* structDecl = DeclType->getAsRecordType()->getDecl();
    
  // If the record is invalid, some of it's members are invalid. To avoid
  // confusion, we forgo checking the intializer for the entire record.
  if (structDecl->isInvalidDecl()) {
    hadError = true;
    return;
  }    

  if (DeclType->isUnionType() && IList->getNumInits() == 0) {
    // Value-initialize the first named member of the union.
    RecordDecl *RD = DeclType->getAsRecordType()->getDecl();
    for (RecordDecl::field_iterator FieldEnd = RD->field_end();
         Field != FieldEnd; ++Field) {
      if (Field->getDeclName()) {
        StructuredList->setInitializedFieldInUnion(*Field);
        break;
      }
    }
    return;
  }

  // If structDecl is a forward declaration, this loop won't do
  // anything except look at designated initializers; That's okay,
  // because an error should get printed out elsewhere. It might be
  // worthwhile to skip over the rest of the initializer, though.
  RecordDecl *RD = DeclType->getAsRecordType()->getDecl();
  RecordDecl::field_iterator FieldEnd = RD->field_end();
  bool InitializedSomething = false;
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
                                     StructuredList, StructuredIndex,
                                     true, TopLevelObject))
        hadError = true;

      InitializedSomething = true;
      continue;
    }

    if (Field == FieldEnd) {
      // We've run out of fields. We're done.
      break;
    }

    // We've already initialized a member of a union. We're done.
    if (InitializedSomething && DeclType->isUnionType())
      break;

    // If we've hit the flexible array member at the end, we're done.
    if (Field->getType()->isIncompleteArrayType())
      break;

    if (Field->isUnnamedBitfield()) {
      // Don't initialize unnamed bitfields, e.g. "int : 20;"
      ++Field;
      continue;
    }

    CheckSubElementType(IList, Field->getType(), Index,
                        StructuredList, StructuredIndex);
    InitializedSomething = true;

    if (DeclType->isUnionType()) {
      // Initialize the first field within the union.
      StructuredList->setInitializedFieldInUnion(*Field);
    }

    ++Field;
  }

  if (Field == FieldEnd || !Field->getType()->isIncompleteArrayType() || 
      Index >= IList->getNumInits() || 
      !isa<InitListExpr>(IList->getInit(Index)))
    return;

  // Handle GNU flexible array initializers.
  if (!TopLevelObject && 
      cast<InitListExpr>(IList->getInit(Index))->getNumInits() > 0) {
    SemaRef->Diag(IList->getInit(Index)->getSourceRange().getBegin(), 
                  diag::err_flexible_array_init_nonempty)
      << IList->getInit(Index)->getSourceRange().getBegin();
    SemaRef->Diag(Field->getLocation(), diag::note_flexible_array_member)
      << *Field;
    hadError = true;
  }

  CheckSubElementType(IList, Field->getType(), Index, StructuredList,
                      StructuredIndex);
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
                                            bool FinishSubobjectInit,
                                            bool TopLevelObject) {
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
    if (RT->getDecl()->isUnion()) {
      FieldIndex = 0;
      StructuredList->setInitializedFieldInUnion(*Field);
    }

    // Update the designator with the field declaration.
    D->setField(*Field);
      
    // Make sure that our non-designated initializer list has space
    // for a subobject corresponding to this field.
    if (FieldIndex >= StructuredList->getNumInits())
      StructuredList->resizeInits(SemaRef->Context, FieldIndex + 1);

    // This designator names a flexible array member.
    if (Field->getType()->isIncompleteArrayType()) {
      bool Invalid = false;
      DesignatedInitExpr::designators_iterator NextD = D;
      ++NextD;
      if (NextD != DIE->designators_end()) {
        // We can't designate an object within the flexible array
        // member (because GCC doesn't allow it).
        SemaRef->Diag(NextD->getStartLocation(), 
                      diag::err_designator_into_flexible_array_member)
          << SourceRange(NextD->getStartLocation(), 
                         DIE->getSourceRange().getEnd());
        SemaRef->Diag(Field->getLocation(), diag::note_flexible_array_member)
          << *Field;
        Invalid = true;
      }

      if (!hadError && !isa<InitListExpr>(DIE->getInit())) {
        // The initializer is not an initializer list.
        SemaRef->Diag(DIE->getInit()->getSourceRange().getBegin(),
                      diag::err_flexible_array_init_needs_braces)
          << DIE->getInit()->getSourceRange();
        SemaRef->Diag(Field->getLocation(), diag::note_flexible_array_member)
          << *Field;
        Invalid = true;
      }

      // Handle GNU flexible array initializers.
      if (!Invalid && !TopLevelObject && 
          cast<InitListExpr>(DIE->getInit())->getNumInits() > 0) {
        SemaRef->Diag(DIE->getSourceRange().getBegin(), 
                      diag::err_flexible_array_init_nonempty)
          << DIE->getSourceRange().getBegin();
        SemaRef->Diag(Field->getLocation(), diag::note_flexible_array_member)
          << *Field;
        Invalid = true;
      }

      if (Invalid) {
        ++Index;
        return true;
      }

      // Initialize the array.
      bool prevHadError = hadError;
      unsigned newStructuredIndex = FieldIndex;
      unsigned OldIndex = Index;
      IList->setInit(Index, DIE->getInit());
      CheckSubElementType(IList, Field->getType(), Index, 
                          StructuredList, newStructuredIndex);
      IList->setInit(OldIndex, DIE);
      if (hadError && !prevHadError) {
        ++Field;
        ++FieldIndex;
        if (NextField)
          *NextField = Field;
        StructuredIndex = FieldIndex;
        return true;
      }
    } else {
      // Recurse to check later designated subobjects.
      QualType FieldType = (*Field)->getType();
      unsigned newStructuredIndex = FieldIndex;
      if (CheckDesignatedInitializer(IList, DIE, ++D, FieldType, 0, 0, Index,
                                     StructuredList, newStructuredIndex,
                                     true, false))
        return true;
    }

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
      FullyStructuredList->sawArrayRangeDesignator();
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
                                   (DesignatedStartIndex == DesignatedEndIndex),
                                   false))
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
  CheckArrayType(IList, CurrentObjectType, DesignatedStartIndex, false, Index,
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

  SourceLocation StartLoc;
  if (Index < IList->getNumInits())
    StartLoc = IList->getInit(Index)->getSourceRange().getBegin();
  InitListExpr *Result 
    = new (SemaRef->Context) InitListExpr(StartLoc, 0, 0, 
                                          IList->getSourceRange().getEnd());
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

bool Sema::CheckInitList(InitListExpr *&InitList, QualType &DeclType) {
  InitListChecker CheckInitList(this, InitList, DeclType);
  if (!CheckInitList.HadError())
    InitList = CheckInitList.getFullyStructuredList();

  return CheckInitList.HadError();
}

/// \brief Diagnose any semantic errors with value-initialization of
/// the given type.
///
/// Value-initialization effectively zero-initializes any types
/// without user-declared constructors, and calls the default
/// constructor for a for any type that has a user-declared
/// constructor (C++ [dcl.init]p5). Value-initialization can fail when
/// a type with a user-declared constructor does not have an
/// accessible, non-deleted default constructor. In C, everything can
/// be value-initialized, which corresponds to C's notion of
/// initializing objects with static storage duration when no
/// initializer is provided for that object. 
///
/// \returns true if there was an error, false otherwise.
bool Sema::CheckValueInitialization(QualType Type, SourceLocation Loc) {
  // C++ [dcl.init]p5:
  //
  //   To value-initialize an object of type T means:

  //     -- if T is an array type, then each element is value-initialized;
  if (const ArrayType *AT = Context.getAsArrayType(Type))
    return CheckValueInitialization(AT->getElementType(), Loc);

  if (const RecordType *RT = Type->getAsRecordType()) {
    if (const CXXRecordType *CXXRec = dyn_cast<CXXRecordType>(RT)) {
      // -- if T is a class type (clause 9) with a user-declared
      //    constructor (12.1), then the default constructor for T is
      //    called (and the initialization is ill-formed if T has no
      //    accessible default constructor);
      if (CXXRec->getDecl()->hasUserDeclaredConstructor())
        // FIXME: Eventually, we'll need to put the constructor decl
        // into the AST.
        return PerformInitializationByConstructor(Type, 0, 0, Loc,
                                                  SourceRange(Loc), 
                                                  DeclarationName(),
                                                  IK_Direct);
    }
  }

  if (Type->isReferenceType()) {
    // C++ [dcl.init]p5:
    //   [...] A program that calls for default-initialization or
    //   value-initialization of an entity of reference type is
    //   ill-formed. [...]
    // FIXME: Once we have code that goes through this path, add an
    // actual diagnostic :)
  }

  return false;
}
