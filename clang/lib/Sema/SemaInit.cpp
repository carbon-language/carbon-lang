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
// This file also implements Sema::CheckInitializerTypes.
//
//===----------------------------------------------------------------------===//

#include "SemaInit.h"
#include "Lookup.h"
#include "Sema.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/Designator.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/TypeLoc.h"
#include "llvm/Support/ErrorHandling.h"
#include <map>
using namespace clang;

//===----------------------------------------------------------------------===//
// Sema Initialization Checking
//===----------------------------------------------------------------------===//

static Expr *IsStringInit(Expr *Init, QualType DeclType, ASTContext &Context) {
  const ArrayType *AT = Context.getAsArrayType(DeclType);
  if (!AT) return 0;

  if (!isa<ConstantArrayType>(AT) && !isa<IncompleteArrayType>(AT))
    return 0;

  // See if this is a string literal or @encode.
  Init = Init->IgnoreParens();

  // Handle @encode, which is a narrow string.
  if (isa<ObjCEncodeExpr>(Init) && AT->getElementType()->isCharType())
    return Init;

  // Otherwise we can only handle string literals.
  StringLiteral *SL = dyn_cast<StringLiteral>(Init);
  if (SL == 0) return 0;

  QualType ElemTy = Context.getCanonicalType(AT->getElementType());
  // char array can be initialized with a narrow string.
  // Only allow char x[] = "foo";  not char x[] = L"foo";
  if (!SL->isWide())
    return ElemTy->isCharType() ? Init : 0;

  // wchar_t array can be initialized with a wide string: C99 6.7.8p15 (with
  // correction from DR343): "An array with element type compatible with a
  // qualified or unqualified version of wchar_t may be initialized by a wide
  // string literal, optionally enclosed in braces."
  if (Context.typesAreCompatible(Context.getWCharType(),
                                 ElemTy.getUnqualifiedType()))
    return Init;

  return 0;
}

static void CheckStringInit(Expr *Str, QualType &DeclT, Sema &S) {
  // Get the length of the string as parsed.
  uint64_t StrLength =
    cast<ConstantArrayType>(Str->getType())->getSize().getZExtValue();


  const ArrayType *AT = S.Context.getAsArrayType(DeclT);
  if (const IncompleteArrayType *IAT = dyn_cast<IncompleteArrayType>(AT)) {
    // C99 6.7.8p14. We have an array of character type with unknown size
    // being initialized to a string literal.
    llvm::APSInt ConstVal(32);
    ConstVal = StrLength;
    // Return a new array type (C99 6.7.8p22).
    DeclT = S.Context.getConstantArrayType(IAT->getElementType(),
                                           ConstVal,
                                           ArrayType::Normal, 0);
    return;
  }

  const ConstantArrayType *CAT = cast<ConstantArrayType>(AT);

  // C99 6.7.8p14. We have an array of character type with known size.  However,
  // the size may be smaller or larger than the string we are initializing.
  // FIXME: Avoid truncation for 64-bit length strings.
  if (StrLength-1 > CAT->getSize().getZExtValue())
    S.Diag(Str->getSourceRange().getBegin(),
           diag::warn_initializer_string_for_char_array_too_long)
      << Str->getSourceRange();

  // Set the type to the actual size that we are initializing.  If we have
  // something like:
  //   char x[1] = "foo";
  // then this will set the string literal's type to char[1].
  Str->setType(DeclT);
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
namespace {
class InitListChecker {
  Sema &SemaRef;
  bool hadError;
  std::map<InitListExpr *, InitListExpr *> SyntacticToSemantic;
  InitListExpr *FullyStructuredList;

  void CheckImplicitInitList(const InitializedEntity &Entity,
                             InitListExpr *ParentIList, QualType T,
                             unsigned &Index, InitListExpr *StructuredList,
                             unsigned &StructuredIndex,
                             bool TopLevelObject = false);
  void CheckExplicitInitList(const InitializedEntity &Entity,
                             InitListExpr *IList, QualType &T,
                             unsigned &Index, InitListExpr *StructuredList,
                             unsigned &StructuredIndex,
                             bool TopLevelObject = false);
  void CheckListElementTypes(const InitializedEntity &Entity,
                             InitListExpr *IList, QualType &DeclType,
                             bool SubobjectIsDesignatorContext,
                             unsigned &Index,
                             InitListExpr *StructuredList,
                             unsigned &StructuredIndex,
                             bool TopLevelObject = false);
  void CheckSubElementType(const InitializedEntity &Entity,
                           InitListExpr *IList, QualType ElemType,
                           unsigned &Index,
                           InitListExpr *StructuredList,
                           unsigned &StructuredIndex);
  void CheckScalarType(const InitializedEntity &Entity,
                       InitListExpr *IList, QualType DeclType,
                       unsigned &Index,
                       InitListExpr *StructuredList,
                       unsigned &StructuredIndex);
  void CheckReferenceType(const InitializedEntity &Entity,
                          InitListExpr *IList, QualType DeclType,
                          unsigned &Index,
                          InitListExpr *StructuredList,
                          unsigned &StructuredIndex);
  void CheckVectorType(const InitializedEntity &Entity,
                       InitListExpr *IList, QualType DeclType, unsigned &Index,
                       InitListExpr *StructuredList,
                       unsigned &StructuredIndex);
  void CheckStructUnionTypes(const InitializedEntity &Entity,
                             InitListExpr *IList, QualType DeclType,
                             RecordDecl::field_iterator Field,
                             bool SubobjectIsDesignatorContext, unsigned &Index,
                             InitListExpr *StructuredList,
                             unsigned &StructuredIndex,
                             bool TopLevelObject = false);
  void CheckArrayType(const InitializedEntity &Entity,
                      InitListExpr *IList, QualType &DeclType,
                      llvm::APSInt elementIndex,
                      bool SubobjectIsDesignatorContext, unsigned &Index,
                      InitListExpr *StructuredList,
                      unsigned &StructuredIndex);
  bool CheckDesignatedInitializer(const InitializedEntity &Entity,
                                  InitListExpr *IList, DesignatedInitExpr *DIE,
                                  unsigned DesigIdx,
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

  void FillInValueInitForField(unsigned Init, FieldDecl *Field,
                               const InitializedEntity &ParentEntity,
                               InitListExpr *ILE, bool &RequiresSecondPass);
  void FillInValueInitializations(const InitializedEntity &Entity,
                                  InitListExpr *ILE, bool &RequiresSecondPass);
public:
  InitListChecker(Sema &S, const InitializedEntity &Entity,
                  InitListExpr *IL, QualType &T);
  bool HadError() { return hadError; }

  // @brief Retrieves the fully-structured initializer list used for
  // semantic analysis and code generation.
  InitListExpr *getFullyStructuredList() const { return FullyStructuredList; }
};
} // end anonymous namespace

void InitListChecker::FillInValueInitForField(unsigned Init, FieldDecl *Field,
                                        const InitializedEntity &ParentEntity,
                                              InitListExpr *ILE, 
                                              bool &RequiresSecondPass) {
  SourceLocation Loc = ILE->getSourceRange().getBegin();
  unsigned NumInits = ILE->getNumInits();
  InitializedEntity MemberEntity 
    = InitializedEntity::InitializeMember(Field, &ParentEntity);
  if (Init >= NumInits || !ILE->getInit(Init)) {
    // FIXME: We probably don't need to handle references
    // specially here, since value-initialization of references is
    // handled in InitializationSequence.
    if (Field->getType()->isReferenceType()) {
      // C++ [dcl.init.aggr]p9:
      //   If an incomplete or empty initializer-list leaves a
      //   member of reference type uninitialized, the program is
      //   ill-formed.
      SemaRef.Diag(Loc, diag::err_init_reference_member_uninitialized)
        << Field->getType()
        << ILE->getSyntacticForm()->getSourceRange();
      SemaRef.Diag(Field->getLocation(),
                   diag::note_uninit_reference_member);
      hadError = true;
      return;
    }
    
    InitializationKind Kind = InitializationKind::CreateValue(Loc, Loc, Loc,
                                                              true);
    InitializationSequence InitSeq(SemaRef, MemberEntity, Kind, 0, 0);
    if (!InitSeq) {
      InitSeq.Diagnose(SemaRef, MemberEntity, Kind, 0, 0);
      hadError = true;
      return;
    }
    
    Sema::OwningExprResult MemberInit
      = InitSeq.Perform(SemaRef, MemberEntity, Kind, 
                        Sema::MultiExprArg(SemaRef, 0, 0));
    if (MemberInit.isInvalid()) {
      hadError = true;
      return;
    }
    
    if (hadError) {
      // Do nothing
    } else if (Init < NumInits) {
      ILE->setInit(Init, MemberInit.takeAs<Expr>());
    } else if (InitSeq.getKind()
                 == InitializationSequence::ConstructorInitialization) {
      // Value-initialization requires a constructor call, so
      // extend the initializer list to include the constructor
      // call and make a note that we'll need to take another pass
      // through the initializer list.
      ILE->updateInit(SemaRef.Context, Init, MemberInit.takeAs<Expr>());
      RequiresSecondPass = true;
    }
  } else if (InitListExpr *InnerILE
               = dyn_cast<InitListExpr>(ILE->getInit(Init)))
    FillInValueInitializations(MemberEntity, InnerILE, 
                               RequiresSecondPass);  
}

/// Recursively replaces NULL values within the given initializer list
/// with expressions that perform value-initialization of the
/// appropriate type.
void 
InitListChecker::FillInValueInitializations(const InitializedEntity &Entity,
                                            InitListExpr *ILE,
                                            bool &RequiresSecondPass) {
  assert((ILE->getType() != SemaRef.Context.VoidTy) &&
         "Should not have void type");
  SourceLocation Loc = ILE->getSourceRange().getBegin();
  if (ILE->getSyntacticForm())
    Loc = ILE->getSyntacticForm()->getSourceRange().getBegin();

  if (const RecordType *RType = ILE->getType()->getAs<RecordType>()) {
    if (RType->getDecl()->isUnion() &&
        ILE->getInitializedFieldInUnion())
      FillInValueInitForField(0, ILE->getInitializedFieldInUnion(),
                              Entity, ILE, RequiresSecondPass);
    else {
      unsigned Init = 0;
      for (RecordDecl::field_iterator
             Field = RType->getDecl()->field_begin(),
             FieldEnd = RType->getDecl()->field_end();
           Field != FieldEnd; ++Field) {
        if (Field->isUnnamedBitfield())
          continue;

        if (hadError)
          return;

        FillInValueInitForField(Init, *Field, Entity, ILE, RequiresSecondPass);
        if (hadError)
          return;

        ++Init;

        // Only look at the first initialization of a union.
        if (RType->getDecl()->isUnion())
          break;
      }
    }

    return;
  }

  QualType ElementType;

  InitializedEntity ElementEntity = Entity;
  unsigned NumInits = ILE->getNumInits();
  unsigned NumElements = NumInits;
  if (const ArrayType *AType = SemaRef.Context.getAsArrayType(ILE->getType())) {
    ElementType = AType->getElementType();
    if (const ConstantArrayType *CAType = dyn_cast<ConstantArrayType>(AType))
      NumElements = CAType->getSize().getZExtValue();
    ElementEntity = InitializedEntity::InitializeElement(SemaRef.Context, 
                                                         0, Entity);
  } else if (const VectorType *VType = ILE->getType()->getAs<VectorType>()) {
    ElementType = VType->getElementType();
    NumElements = VType->getNumElements();
    ElementEntity = InitializedEntity::InitializeElement(SemaRef.Context, 
                                                         0, Entity);
  } else
    ElementType = ILE->getType();

  
  for (unsigned Init = 0; Init != NumElements; ++Init) {
    if (hadError)
      return;

    if (ElementEntity.getKind() == InitializedEntity::EK_ArrayElement ||
        ElementEntity.getKind() == InitializedEntity::EK_VectorElement)
      ElementEntity.setElementIndex(Init);

    if (Init >= NumInits || !ILE->getInit(Init)) {
      InitializationKind Kind = InitializationKind::CreateValue(Loc, Loc, Loc,
                                                                true);
      InitializationSequence InitSeq(SemaRef, ElementEntity, Kind, 0, 0);
      if (!InitSeq) {
        InitSeq.Diagnose(SemaRef, ElementEntity, Kind, 0, 0);
        hadError = true;
        return;
      }

      Sema::OwningExprResult ElementInit
        = InitSeq.Perform(SemaRef, ElementEntity, Kind, 
                          Sema::MultiExprArg(SemaRef, 0, 0));
      if (ElementInit.isInvalid()) {
        hadError = true;
        return;
      }

      if (hadError) {
        // Do nothing
      } else if (Init < NumInits) {
        ILE->setInit(Init, ElementInit.takeAs<Expr>());
      } else if (InitSeq.getKind()
                   == InitializationSequence::ConstructorInitialization) {
        // Value-initialization requires a constructor call, so
        // extend the initializer list to include the constructor
        // call and make a note that we'll need to take another pass
        // through the initializer list.
        ILE->updateInit(SemaRef.Context, Init, ElementInit.takeAs<Expr>());
        RequiresSecondPass = true;
      }
    } else if (InitListExpr *InnerILE
                 = dyn_cast<InitListExpr>(ILE->getInit(Init)))
      FillInValueInitializations(ElementEntity, InnerILE, RequiresSecondPass);
  }
}


InitListChecker::InitListChecker(Sema &S, const InitializedEntity &Entity,
                                 InitListExpr *IL, QualType &T)
  : SemaRef(S) {
  hadError = false;

  unsigned newIndex = 0;
  unsigned newStructuredIndex = 0;
  FullyStructuredList
    = getStructuredSubobjectInit(IL, newIndex, T, 0, 0, IL->getSourceRange());
  CheckExplicitInitList(Entity, IL, T, newIndex, 
                        FullyStructuredList, newStructuredIndex,
                        /*TopLevelObject=*/true);

  if (!hadError) {
    bool RequiresSecondPass = false;
    FillInValueInitializations(Entity, FullyStructuredList, RequiresSecondPass);
    if (RequiresSecondPass && !hadError)
      FillInValueInitializations(Entity, FullyStructuredList, 
                                 RequiresSecondPass);
  }
}

int InitListChecker::numArrayElements(QualType DeclType) {
  // FIXME: use a proper constant
  int maxElements = 0x7FFFFFFF;
  if (const ConstantArrayType *CAT =
        SemaRef.Context.getAsConstantArrayType(DeclType)) {
    maxElements = static_cast<int>(CAT->getSize().getZExtValue());
  }
  return maxElements;
}

int InitListChecker::numStructUnionElements(QualType DeclType) {
  RecordDecl *structDecl = DeclType->getAs<RecordType>()->getDecl();
  int InitializableMembers = 0;
  for (RecordDecl::field_iterator
         Field = structDecl->field_begin(),
         FieldEnd = structDecl->field_end();
       Field != FieldEnd; ++Field) {
    if ((*Field)->getIdentifier() || !(*Field)->isBitField())
      ++InitializableMembers;
  }
  if (structDecl->isUnion())
    return std::min(InitializableMembers, 1);
  return InitializableMembers - structDecl->hasFlexibleArrayMember();
}

void InitListChecker::CheckImplicitInitList(const InitializedEntity &Entity,
                                            InitListExpr *ParentIList,
                                            QualType T, unsigned &Index,
                                            InitListExpr *StructuredList,
                                            unsigned &StructuredIndex,
                                            bool TopLevelObject) {
  int maxElements = 0;

  if (T->isArrayType())
    maxElements = numArrayElements(T);
  else if (T->isRecordType())
    maxElements = numStructUnionElements(T);
  else if (T->isVectorType())
    maxElements = T->getAs<VectorType>()->getNumElements();
  else
    assert(0 && "CheckImplicitInitList(): Illegal type");

  if (maxElements == 0) {
    SemaRef.Diag(ParentIList->getInit(Index)->getLocStart(),
                  diag::err_implicit_empty_initializer);
    ++Index;
    hadError = true;
    return;
  }

  // Build a structured initializer list corresponding to this subobject.
  InitListExpr *StructuredSubobjectInitList
    = getStructuredSubobjectInit(ParentIList, Index, T, StructuredList,
                                 StructuredIndex,
          SourceRange(ParentIList->getInit(Index)->getSourceRange().getBegin(),
                      ParentIList->getSourceRange().getEnd()));
  unsigned StructuredSubobjectInitIndex = 0;

  // Check the element types and build the structural subobject.
  unsigned StartIndex = Index;
  CheckListElementTypes(Entity, ParentIList, T, 
                        /*SubobjectIsDesignatorContext=*/false, Index,
                        StructuredSubobjectInitList,
                        StructuredSubobjectInitIndex,
                        TopLevelObject);
  unsigned EndIndex = (Index == StartIndex? StartIndex : Index - 1);
  StructuredSubobjectInitList->setType(T);

  // Update the structured sub-object initializer so that it's ending
  // range corresponds with the end of the last initializer it used.
  if (EndIndex < ParentIList->getNumInits()) {
    SourceLocation EndLoc
      = ParentIList->getInit(EndIndex)->getSourceRange().getEnd();
    StructuredSubobjectInitList->setRBraceLoc(EndLoc);
  }
  
  // Warn about missing braces.
  if (T->isArrayType() || T->isRecordType()) {
    SemaRef.Diag(StructuredSubobjectInitList->getLocStart(),
                 diag::warn_missing_braces)
    << StructuredSubobjectInitList->getSourceRange()
    << FixItHint::CreateInsertion(StructuredSubobjectInitList->getLocStart(), 
                                  "{")
    << FixItHint::CreateInsertion(SemaRef.PP.getLocForEndOfToken(
                                      StructuredSubobjectInitList->getLocEnd()), 
                                  "}");
  }
}

void InitListChecker::CheckExplicitInitList(const InitializedEntity &Entity,
                                            InitListExpr *IList, QualType &T,
                                            unsigned &Index,
                                            InitListExpr *StructuredList,
                                            unsigned &StructuredIndex,
                                            bool TopLevelObject) {
  assert(IList->isExplicit() && "Illegal Implicit InitListExpr");
  SyntacticToSemantic[IList] = StructuredList;
  StructuredList->setSyntacticForm(IList);
  CheckListElementTypes(Entity, IList, T, /*SubobjectIsDesignatorContext=*/true, 
                        Index, StructuredList, StructuredIndex, TopLevelObject);
  QualType ExprTy = T.getNonLValueExprType(SemaRef.Context);
  IList->setType(ExprTy);
  StructuredList->setType(ExprTy);
  if (hadError)
    return;

  if (Index < IList->getNumInits()) {
    // We have leftover initializers
    if (StructuredIndex == 1 &&
        IsStringInit(StructuredList->getInit(0), T, SemaRef.Context)) {
      unsigned DK = diag::warn_excess_initializers_in_char_array_initializer;
      if (SemaRef.getLangOptions().CPlusPlus) {
        DK = diag::err_excess_initializers_in_char_array_initializer;
        hadError = true;
      }
      // Special-case
      SemaRef.Diag(IList->getInit(Index)->getLocStart(), DK)
        << IList->getInit(Index)->getSourceRange();
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
      if (SemaRef.getLangOptions().CPlusPlus) {
        DK = diag::err_excess_initializers;
        hadError = true;
      }
      if (SemaRef.getLangOptions().OpenCL && initKind == 1) {
        DK = diag::err_excess_initializers;
        hadError = true;
      }

      SemaRef.Diag(IList->getInit(Index)->getLocStart(), DK)
        << initKind << IList->getInit(Index)->getSourceRange();
    }
  }

  if (T->isScalarType() && !TopLevelObject)
    SemaRef.Diag(IList->getLocStart(), diag::warn_braces_around_scalar_init)
      << IList->getSourceRange()
      << FixItHint::CreateRemoval(IList->getLocStart())
      << FixItHint::CreateRemoval(IList->getLocEnd());
}

void InitListChecker::CheckListElementTypes(const InitializedEntity &Entity,
                                            InitListExpr *IList,
                                            QualType &DeclType,
                                            bool SubobjectIsDesignatorContext,
                                            unsigned &Index,
                                            InitListExpr *StructuredList,
                                            unsigned &StructuredIndex,
                                            bool TopLevelObject) {
  if (DeclType->isScalarType()) {
    CheckScalarType(Entity, IList, DeclType, Index,
                    StructuredList, StructuredIndex);
  } else if (DeclType->isVectorType()) {
    CheckVectorType(Entity, IList, DeclType, Index, 
                    StructuredList, StructuredIndex);
  } else if (DeclType->isAggregateType()) {
    if (DeclType->isRecordType()) {
      RecordDecl *RD = DeclType->getAs<RecordType>()->getDecl();
      CheckStructUnionTypes(Entity, IList, DeclType, RD->field_begin(),
                            SubobjectIsDesignatorContext, Index,
                            StructuredList, StructuredIndex,
                            TopLevelObject);
    } else if (DeclType->isArrayType()) {
      llvm::APSInt Zero(
                      SemaRef.Context.getTypeSize(SemaRef.Context.getSizeType()),
                      false);
      CheckArrayType(Entity, IList, DeclType, Zero, 
                     SubobjectIsDesignatorContext, Index,
                     StructuredList, StructuredIndex);
    } else
      assert(0 && "Aggregate that isn't a structure or array?!");
  } else if (DeclType->isVoidType() || DeclType->isFunctionType()) {
    // This type is invalid, issue a diagnostic.
    ++Index;
    SemaRef.Diag(IList->getLocStart(), diag::err_illegal_initializer_type)
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
    SemaRef.Diag(IList->getLocStart(), diag::err_init_non_aggr_init_list)
      << DeclType << IList->getSourceRange();
    hadError = true;
  } else if (DeclType->isReferenceType()) {
    CheckReferenceType(Entity, IList, DeclType, Index,
                       StructuredList, StructuredIndex);
  } else if (DeclType->isObjCObjectType()) {
    SemaRef.Diag(IList->getLocStart(), diag::err_init_objc_class)
      << DeclType;
    hadError = true;
  } else {
    SemaRef.Diag(IList->getLocStart(), diag::err_illegal_initializer_type)
      << DeclType;
    hadError = true;
  }
}

void InitListChecker::CheckSubElementType(const InitializedEntity &Entity,
                                          InitListExpr *IList,
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
    CheckExplicitInitList(Entity, SubInitList, ElemType, newIndex,
                          newStructuredList, newStructuredIndex);
    ++StructuredIndex;
    ++Index;
  } else if (Expr *Str = IsStringInit(expr, ElemType, SemaRef.Context)) {
    CheckStringInit(Str, ElemType, SemaRef);
    UpdateStructuredListElement(StructuredList, StructuredIndex, Str);
    ++Index;
  } else if (ElemType->isScalarType()) {
    CheckScalarType(Entity, IList, ElemType, Index, 
                    StructuredList, StructuredIndex);
  } else if (ElemType->isReferenceType()) {
    CheckReferenceType(Entity, IList, ElemType, Index,
                       StructuredList, StructuredIndex);
  } else {
    if (SemaRef.getLangOptions().CPlusPlus) {
      // C++ [dcl.init.aggr]p12:
      //   All implicit type conversions (clause 4) are considered when
      //   initializing the aggregate member with an ini- tializer from
      //   an initializer-list. If the initializer can initialize a
      //   member, the member is initialized. [...]

      // FIXME: Better EqualLoc?
      InitializationKind Kind = 
        InitializationKind::CreateCopy(expr->getLocStart(), SourceLocation());
      InitializationSequence Seq(SemaRef, Entity, Kind, &expr, 1);
      
      if (Seq) {
        Sema::OwningExprResult Result = 
          Seq.Perform(SemaRef, Entity, Kind,
                      Sema::MultiExprArg(SemaRef, (void **)&expr, 1));
        if (Result.isInvalid())
          hadError = true;
        
        UpdateStructuredListElement(StructuredList, StructuredIndex, 
                                    Result.takeAs<Expr>());
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
      if ((ElemType->isRecordType() || ElemType->isVectorType()) &&
          SemaRef.Context.hasSameUnqualifiedType(expr->getType(), ElemType)) {
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
      CheckImplicitInitList(Entity, IList, ElemType, Index, StructuredList,
                            StructuredIndex);
      ++StructuredIndex;
    } else {
      // We cannot initialize this element, so let
      // PerformCopyInitialization produce the appropriate diagnostic.
      SemaRef.PerformCopyInitialization(Entity, SourceLocation(), 
                                        SemaRef.Owned(expr));
      IList->setInit(Index, 0);
      hadError = true;
      ++Index;
      ++StructuredIndex;
    }
  }
}

void InitListChecker::CheckScalarType(const InitializedEntity &Entity,
                                      InitListExpr *IList, QualType DeclType,
                                      unsigned &Index,
                                      InitListExpr *StructuredList,
                                      unsigned &StructuredIndex) {
  if (Index < IList->getNumInits()) {
    Expr *expr = IList->getInit(Index);
    if (isa<InitListExpr>(expr)) {
      SemaRef.Diag(IList->getLocStart(),
                    diag::err_many_braces_around_scalar_init)
        << IList->getSourceRange();
      hadError = true;
      ++Index;
      ++StructuredIndex;
      return;
    } else if (isa<DesignatedInitExpr>(expr)) {
      SemaRef.Diag(expr->getSourceRange().getBegin(),
                    diag::err_designator_for_scalar_init)
        << DeclType << expr->getSourceRange();
      hadError = true;
      ++Index;
      ++StructuredIndex;
      return;
    }

    Sema::OwningExprResult Result =
      SemaRef.PerformCopyInitialization(Entity, expr->getLocStart(),
                                        SemaRef.Owned(expr));

    Expr *ResultExpr = 0;

    if (Result.isInvalid())
      hadError = true; // types weren't compatible.
    else {
      ResultExpr = Result.takeAs<Expr>();
      
      if (ResultExpr != expr) {
        // The type was promoted, update initializer list.
        IList->setInit(Index, ResultExpr);
      }
    }
    if (hadError)
      ++StructuredIndex;
    else
      UpdateStructuredListElement(StructuredList, StructuredIndex, ResultExpr);
    ++Index;
  } else {
    SemaRef.Diag(IList->getLocStart(), diag::err_empty_scalar_initializer)
      << IList->getSourceRange();
    hadError = true;
    ++Index;
    ++StructuredIndex;
    return;
  }
}

void InitListChecker::CheckReferenceType(const InitializedEntity &Entity,
                                         InitListExpr *IList, QualType DeclType,
                                         unsigned &Index,
                                         InitListExpr *StructuredList,
                                         unsigned &StructuredIndex) {
  if (Index < IList->getNumInits()) {
    Expr *expr = IList->getInit(Index);
    if (isa<InitListExpr>(expr)) {
      SemaRef.Diag(IList->getLocStart(), diag::err_init_non_aggr_init_list)
        << DeclType << IList->getSourceRange();
      hadError = true;
      ++Index;
      ++StructuredIndex;
      return;
    }

    Sema::OwningExprResult Result =
      SemaRef.PerformCopyInitialization(Entity, expr->getLocStart(),
                                        SemaRef.Owned(expr));

    if (Result.isInvalid())
      hadError = true;

    expr = Result.takeAs<Expr>();
    IList->setInit(Index, expr);

    if (hadError)
      ++StructuredIndex;
    else
      UpdateStructuredListElement(StructuredList, StructuredIndex, expr);
    ++Index;
  } else {
    // FIXME: It would be wonderful if we could point at the actual member. In
    // general, it would be useful to pass location information down the stack,
    // so that we know the location (or decl) of the "current object" being
    // initialized.
    SemaRef.Diag(IList->getLocStart(),
                  diag::err_init_reference_member_uninitialized)
      << DeclType
      << IList->getSourceRange();
    hadError = true;
    ++Index;
    ++StructuredIndex;
    return;
  }
}

void InitListChecker::CheckVectorType(const InitializedEntity &Entity,
                                      InitListExpr *IList, QualType DeclType,
                                      unsigned &Index,
                                      InitListExpr *StructuredList,
                                      unsigned &StructuredIndex) {
  if (Index < IList->getNumInits()) {
    const VectorType *VT = DeclType->getAs<VectorType>();
    unsigned maxElements = VT->getNumElements();
    unsigned numEltsInit = 0;
    QualType elementType = VT->getElementType();

    if (!SemaRef.getLangOptions().OpenCL) {
      InitializedEntity ElementEntity =
        InitializedEntity::InitializeElement(SemaRef.Context, 0, Entity);

      for (unsigned i = 0; i < maxElements; ++i, ++numEltsInit) {
        // Don't attempt to go past the end of the init list
        if (Index >= IList->getNumInits())
          break;
        
        ElementEntity.setElementIndex(Index);
        CheckSubElementType(ElementEntity, IList, elementType, Index,
                            StructuredList, StructuredIndex);
      }
    } else {
      InitializedEntity ElementEntity =
        InitializedEntity::InitializeElement(SemaRef.Context, 0, Entity);
      
      // OpenCL initializers allows vectors to be constructed from vectors.
      for (unsigned i = 0; i < maxElements; ++i) {
        // Don't attempt to go past the end of the init list
        if (Index >= IList->getNumInits())
          break;
        
        ElementEntity.setElementIndex(Index);

        QualType IType = IList->getInit(Index)->getType();
        if (!IType->isVectorType()) {
          CheckSubElementType(ElementEntity, IList, elementType, Index,
                              StructuredList, StructuredIndex);
          ++numEltsInit;
        } else {
          QualType VecType;
          const VectorType *IVT = IType->getAs<VectorType>();
          unsigned numIElts = IVT->getNumElements();
          
          if (IType->isExtVectorType())
            VecType = SemaRef.Context.getExtVectorType(elementType, numIElts);
          else
            VecType = SemaRef.Context.getVectorType(elementType, numIElts,
                                                    IVT->getAltiVecSpecific());
          CheckSubElementType(ElementEntity, IList, VecType, Index,
                              StructuredList, StructuredIndex);
          numEltsInit += numIElts;
        }
      }
    }

    // OpenCL requires all elements to be initialized.
    if (numEltsInit != maxElements)
      if (SemaRef.getLangOptions().OpenCL)
        SemaRef.Diag(IList->getSourceRange().getBegin(),
                     diag::err_vector_incorrect_num_initializers)
          << (numEltsInit < maxElements) << maxElements << numEltsInit;
  }
}

void InitListChecker::CheckArrayType(const InitializedEntity &Entity,
                                     InitListExpr *IList, QualType &DeclType,
                                     llvm::APSInt elementIndex,
                                     bool SubobjectIsDesignatorContext,
                                     unsigned &Index,
                                     InitListExpr *StructuredList,
                                     unsigned &StructuredIndex) {
  // Check for the special-case of initializing an array with a string.
  if (Index < IList->getNumInits()) {
    if (Expr *Str = IsStringInit(IList->getInit(Index), DeclType,
                                 SemaRef.Context)) {
      CheckStringInit(Str, DeclType, SemaRef);
      // We place the string literal directly into the resulting
      // initializer list. This is the only place where the structure
      // of the structured initializer list doesn't match exactly,
      // because doing so would involve allocating one character
      // constant for each string.
      UpdateStructuredListElement(StructuredList, StructuredIndex, Str);
      StructuredList->resizeInits(SemaRef.Context, StructuredIndex);
      ++Index;
      return;
    }
  }
  if (const VariableArrayType *VAT =
        SemaRef.Context.getAsVariableArrayType(DeclType)) {
    // Check for VLAs; in standard C it would be possible to check this
    // earlier, but I don't know where clang accepts VLAs (gcc accepts
    // them in all sorts of strange places).
    SemaRef.Diag(VAT->getSizeExpr()->getLocStart(),
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
        SemaRef.Context.getAsConstantArrayType(DeclType)) {
    maxElements = CAT->getSize();
    elementIndex.extOrTrunc(maxElements.getBitWidth());
    elementIndex.setIsUnsigned(maxElements.isUnsigned());
    maxElementsKnown = true;
  }

  QualType elementType = SemaRef.Context.getAsArrayType(DeclType)
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
      if (CheckDesignatedInitializer(Entity, IList, DIE, 0,
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

    InitializedEntity ElementEntity =
      InitializedEntity::InitializeElement(SemaRef.Context, StructuredIndex, 
                                           Entity);
    // Check this element.
    CheckSubElementType(ElementEntity, IList, elementType, Index,
                        StructuredList, StructuredIndex);
    ++elementIndex;

    // If the array is of incomplete type, keep track of the number of
    // elements in the initializer.
    if (!maxElementsKnown && elementIndex > maxElements)
      maxElements = elementIndex;
  }
  if (!hadError && DeclType->isIncompleteArrayType()) {
    // If this is an incomplete array type, the actual type needs to
    // be calculated here.
    llvm::APSInt Zero(maxElements.getBitWidth(), maxElements.isUnsigned());
    if (maxElements == Zero) {
      // Sizing an array implicitly to zero is not allowed by ISO C,
      // but is supported by GNU.
      SemaRef.Diag(IList->getLocStart(),
                    diag::ext_typecheck_zero_array_size);
    }

    DeclType = SemaRef.Context.getConstantArrayType(elementType, maxElements,
                                                     ArrayType::Normal, 0);
  }
}

void InitListChecker::CheckStructUnionTypes(const InitializedEntity &Entity,
                                            InitListExpr *IList,
                                            QualType DeclType,
                                            RecordDecl::field_iterator Field,
                                            bool SubobjectIsDesignatorContext,
                                            unsigned &Index,
                                            InitListExpr *StructuredList,
                                            unsigned &StructuredIndex,
                                            bool TopLevelObject) {
  RecordDecl* structDecl = DeclType->getAs<RecordType>()->getDecl();

  // If the record is invalid, some of it's members are invalid. To avoid
  // confusion, we forgo checking the intializer for the entire record.
  if (structDecl->isInvalidDecl()) {
    hadError = true;
    return;
  }

  if (DeclType->isUnionType() && IList->getNumInits() == 0) {
    // Value-initialize the first named member of the union.
    RecordDecl *RD = DeclType->getAs<RecordType>()->getDecl();
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
  RecordDecl *RD = DeclType->getAs<RecordType>()->getDecl();
  RecordDecl::field_iterator FieldEnd = RD->field_end();
  bool InitializedSomething = false;
  bool CheckForMissingFields = true;
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
      if (CheckDesignatedInitializer(Entity, IList, DIE, 0,
                                     DeclType, &Field, 0, Index,
                                     StructuredList, StructuredIndex,
                                     true, TopLevelObject))
        hadError = true;

      InitializedSomething = true;

      // Disable check for missing fields when designators are used.
      // This matches gcc behaviour.
      CheckForMissingFields = false;
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

    InitializedEntity MemberEntity =
      InitializedEntity::InitializeMember(*Field, &Entity);
    CheckSubElementType(MemberEntity, IList, Field->getType(), Index,
                        StructuredList, StructuredIndex);
    InitializedSomething = true;

    if (DeclType->isUnionType()) {
      // Initialize the first field within the union.
      StructuredList->setInitializedFieldInUnion(*Field);
    }

    ++Field;
  }

  // Emit warnings for missing struct field initializers.
  if (InitializedSomething && CheckForMissingFields && Field != FieldEnd && 
      !Field->getType()->isIncompleteArrayType() && !DeclType->isUnionType()) {
    // It is possible we have one or more unnamed bitfields remaining.
    // Find first (if any) named field and emit warning.
    for (RecordDecl::field_iterator it = Field, end = RD->field_end();
         it != end; ++it) {
      if (!it->isUnnamedBitfield()) {
        SemaRef.Diag(IList->getSourceRange().getEnd(),
                     diag::warn_missing_field_initializers) << it->getName();
        break;
      }
    }
  }

  if (Field == FieldEnd || !Field->getType()->isIncompleteArrayType() ||
      Index >= IList->getNumInits())
    return;

  // Handle GNU flexible array initializers.
  if (!TopLevelObject &&
      (!isa<InitListExpr>(IList->getInit(Index)) ||
       cast<InitListExpr>(IList->getInit(Index))->getNumInits() > 0)) {
    SemaRef.Diag(IList->getInit(Index)->getSourceRange().getBegin(),
                  diag::err_flexible_array_init_nonempty)
      << IList->getInit(Index)->getSourceRange().getBegin();
    SemaRef.Diag(Field->getLocation(), diag::note_flexible_array_member)
      << *Field;
    hadError = true;
    ++Index;
    return;
  } else {
    SemaRef.Diag(IList->getInit(Index)->getSourceRange().getBegin(),
                 diag::ext_flexible_array_init)
      << IList->getInit(Index)->getSourceRange().getBegin();
    SemaRef.Diag(Field->getLocation(), diag::note_flexible_array_member)
      << *Field;
  }

  InitializedEntity MemberEntity =
    InitializedEntity::InitializeMember(*Field, &Entity);
    
  if (isa<InitListExpr>(IList->getInit(Index)))
    CheckSubElementType(MemberEntity, IList, Field->getType(), Index, 
                        StructuredList, StructuredIndex);
  else
    CheckImplicitInitList(MemberEntity, IList, Field->getType(), Index, 
                          StructuredList, StructuredIndex);
}

/// \brief Expand a field designator that refers to a member of an
/// anonymous struct or union into a series of field designators that
/// refers to the field within the appropriate subobject.
///
/// Field/FieldIndex will be updated to point to the (new)
/// currently-designated field.
static void ExpandAnonymousFieldDesignator(Sema &SemaRef,
                                           DesignatedInitExpr *DIE,
                                           unsigned DesigIdx,
                                           FieldDecl *Field,
                                        RecordDecl::field_iterator &FieldIter,
                                           unsigned &FieldIndex) {
  typedef DesignatedInitExpr::Designator Designator;

  // Build the path from the current object to the member of the
  // anonymous struct/union (backwards).
  llvm::SmallVector<FieldDecl *, 4> Path;
  SemaRef.BuildAnonymousStructUnionMemberPath(Field, Path);

  // Build the replacement designators.
  llvm::SmallVector<Designator, 4> Replacements;
  for (llvm::SmallVector<FieldDecl *, 4>::reverse_iterator
         FI = Path.rbegin(), FIEnd = Path.rend();
       FI != FIEnd; ++FI) {
    if (FI + 1 == FIEnd)
      Replacements.push_back(Designator((IdentifierInfo *)0,
                                    DIE->getDesignator(DesigIdx)->getDotLoc(),
                                DIE->getDesignator(DesigIdx)->getFieldLoc()));
    else
      Replacements.push_back(Designator((IdentifierInfo *)0, SourceLocation(),
                                        SourceLocation()));
    Replacements.back().setField(*FI);
  }

  // Expand the current designator into the set of replacement
  // designators, so we have a full subobject path down to where the
  // member of the anonymous struct/union is actually stored.
  DIE->ExpandDesignator(SemaRef.Context, DesigIdx, &Replacements[0],
                        &Replacements[0] + Replacements.size());

  // Update FieldIter/FieldIndex;
  RecordDecl *Record = cast<RecordDecl>(Path.back()->getDeclContext());
  FieldIter = Record->field_begin();
  FieldIndex = 0;
  for (RecordDecl::field_iterator FEnd = Record->field_end();
       FieldIter != FEnd; ++FieldIter) {
    if (FieldIter->isUnnamedBitfield())
        continue;

    if (*FieldIter == Path.back())
      return;

    ++FieldIndex;
  }

  assert(false && "Unable to find anonymous struct/union field");
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
/// @param DIE The designated initializer expression.
///
/// @param DesigIdx  The index of the current designator.
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
InitListChecker::CheckDesignatedInitializer(const InitializedEntity &Entity,
                                            InitListExpr *IList,
                                      DesignatedInitExpr *DIE,
                                      unsigned DesigIdx,
                                      QualType &CurrentObjectType,
                                      RecordDecl::field_iterator *NextField,
                                      llvm::APSInt *NextElementIndex,
                                      unsigned &Index,
                                      InitListExpr *StructuredList,
                                      unsigned &StructuredIndex,
                                            bool FinishSubobjectInit,
                                            bool TopLevelObject) {
  if (DesigIdx == DIE->size()) {
    // Check the actual initialization for the designated object type.
    bool prevHadError = hadError;

    // Temporarily remove the designator expression from the
    // initializer list that the child calls see, so that we don't try
    // to re-process the designator.
    unsigned OldIndex = Index;
    IList->setInit(OldIndex, DIE->getInit());

    CheckSubElementType(Entity, IList, CurrentObjectType, Index,
                        StructuredList, StructuredIndex);

    // Restore the designated initializer expression in the syntactic
    // form of the initializer list.
    if (IList->getInit(OldIndex) != DIE->getInit())
      DIE->setInit(IList->getInit(OldIndex));
    IList->setInit(OldIndex, DIE);

    return hadError && !prevHadError;
  }

  bool IsFirstDesignator = (DesigIdx == 0);
  assert((IsFirstDesignator || StructuredList) &&
         "Need a non-designated initializer list to start from");

  DesignatedInitExpr::Designator *D = DIE->getDesignator(DesigIdx);
  // Determine the structural initializer list that corresponds to the
  // current subobject.
  StructuredList = IsFirstDesignator? SyntacticToSemantic[IList]
    : getStructuredSubobjectInit(IList, Index, CurrentObjectType,
                                 StructuredList, StructuredIndex,
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
    const RecordType *RT = CurrentObjectType->getAs<RecordType>();
    if (!RT) {
      SourceLocation Loc = D->getDotLoc();
      if (Loc.isInvalid())
        Loc = D->getFieldLoc();
      SemaRef.Diag(Loc, diag::err_field_designator_non_aggr)
        << SemaRef.getLangOptions().CPlusPlus << CurrentObjectType;
      ++Index;
      return true;
    }

    // Note: we perform a linear search of the fields here, despite
    // the fact that we have a faster lookup method, because we always
    // need to compute the field's index.
    FieldDecl *KnownField = D->getField();
    IdentifierInfo *FieldName = D->getFieldName();
    unsigned FieldIndex = 0;
    RecordDecl::field_iterator
      Field = RT->getDecl()->field_begin(),
      FieldEnd = RT->getDecl()->field_end();
    for (; Field != FieldEnd; ++Field) {
      if (Field->isUnnamedBitfield())
        continue;

      if (KnownField == *Field || Field->getIdentifier() == FieldName)
        break;

      ++FieldIndex;
    }

    if (Field == FieldEnd) {
      // There was no normal field in the struct with the designated
      // name. Perform another lookup for this name, which may find
      // something that we can't designate (e.g., a member function),
      // may find nothing, or may find a member of an anonymous
      // struct/union.
      DeclContext::lookup_result Lookup = RT->getDecl()->lookup(FieldName);
      FieldDecl *ReplacementField = 0;
      if (Lookup.first == Lookup.second) {
        // Name lookup didn't find anything. Determine whether this
        // was a typo for another field name.
        LookupResult R(SemaRef, FieldName, D->getFieldLoc(), 
                       Sema::LookupMemberName);
        if (SemaRef.CorrectTypo(R, /*Scope=*/0, /*SS=*/0, RT->getDecl(), false,
                                Sema::CTC_NoKeywords) && 
            (ReplacementField = R.getAsSingle<FieldDecl>()) &&
            ReplacementField->getDeclContext()->getLookupContext()
                                                      ->Equals(RT->getDecl())) {
          SemaRef.Diag(D->getFieldLoc(), 
                       diag::err_field_designator_unknown_suggest)
            << FieldName << CurrentObjectType << R.getLookupName()
            << FixItHint::CreateReplacement(D->getFieldLoc(),
                                            R.getLookupName().getAsString());
          SemaRef.Diag(ReplacementField->getLocation(), 
                       diag::note_previous_decl)
            << ReplacementField->getDeclName();
        } else {
          SemaRef.Diag(D->getFieldLoc(), diag::err_field_designator_unknown)
            << FieldName << CurrentObjectType;
          ++Index;
          return true;
        }
      } else if (!KnownField) {
        // Determine whether we found a field at all.
        ReplacementField = dyn_cast<FieldDecl>(*Lookup.first);
      }

      if (!ReplacementField) {
        // Name lookup found something, but it wasn't a field.
        SemaRef.Diag(D->getFieldLoc(), diag::err_field_designator_nonfield)
          << FieldName;
        SemaRef.Diag((*Lookup.first)->getLocation(),
                      diag::note_field_designator_found);
        ++Index;
        return true;
      }

      if (!KnownField && 
          cast<RecordDecl>((ReplacementField)->getDeclContext())
                                                 ->isAnonymousStructOrUnion()) {
        // Handle an field designator that refers to a member of an
        // anonymous struct or union.
        ExpandAnonymousFieldDesignator(SemaRef, DIE, DesigIdx,
                                       ReplacementField,
                                       Field, FieldIndex);
        D = DIE->getDesignator(DesigIdx);
      } else if (!KnownField) {
        // The replacement field comes from typo correction; find it
        // in the list of fields.
        FieldIndex = 0;
        Field = RT->getDecl()->field_begin();
        for (; Field != FieldEnd; ++Field) {
          if (Field->isUnnamedBitfield())
            continue;

          if (ReplacementField == *Field || 
              Field->getIdentifier() == ReplacementField->getIdentifier())
            break;

          ++FieldIndex;
        }
      }
    } else if (!KnownField &&
               cast<RecordDecl>((*Field)->getDeclContext())
                 ->isAnonymousStructOrUnion()) {
      ExpandAnonymousFieldDesignator(SemaRef, DIE, DesigIdx, *Field,
                                     Field, FieldIndex);
      D = DIE->getDesignator(DesigIdx);
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
      StructuredList->resizeInits(SemaRef.Context, FieldIndex + 1);

    // This designator names a flexible array member.
    if (Field->getType()->isIncompleteArrayType()) {
      bool Invalid = false;
      if ((DesigIdx + 1) != DIE->size()) {
        // We can't designate an object within the flexible array
        // member (because GCC doesn't allow it).
        DesignatedInitExpr::Designator *NextD
          = DIE->getDesignator(DesigIdx + 1);
        SemaRef.Diag(NextD->getStartLocation(),
                      diag::err_designator_into_flexible_array_member)
          << SourceRange(NextD->getStartLocation(),
                         DIE->getSourceRange().getEnd());
        SemaRef.Diag(Field->getLocation(), diag::note_flexible_array_member)
          << *Field;
        Invalid = true;
      }

      if (!hadError && !isa<InitListExpr>(DIE->getInit())) {
        // The initializer is not an initializer list.
        SemaRef.Diag(DIE->getInit()->getSourceRange().getBegin(),
                      diag::err_flexible_array_init_needs_braces)
          << DIE->getInit()->getSourceRange();
        SemaRef.Diag(Field->getLocation(), diag::note_flexible_array_member)
          << *Field;
        Invalid = true;
      }

      // Handle GNU flexible array initializers.
      if (!Invalid && !TopLevelObject &&
          cast<InitListExpr>(DIE->getInit())->getNumInits() > 0) {
        SemaRef.Diag(DIE->getSourceRange().getBegin(),
                      diag::err_flexible_array_init_nonempty)
          << DIE->getSourceRange().getBegin();
        SemaRef.Diag(Field->getLocation(), diag::note_flexible_array_member)
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

      InitializedEntity MemberEntity =
        InitializedEntity::InitializeMember(*Field, &Entity);
      CheckSubElementType(MemberEntity, IList, Field->getType(), Index,
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
      
      InitializedEntity MemberEntity =
        InitializedEntity::InitializeMember(*Field, &Entity);
      if (CheckDesignatedInitializer(MemberEntity, IList, DIE, DesigIdx + 1, 
                                     FieldType, 0, 0, Index, 
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

    // We've already initialized something in the union; we're done.
    if (RT->getDecl()->isUnion())
      return hadError;

    // Check the remaining fields within this class/struct/union subobject.
    bool prevHadError = hadError;
    
    CheckStructUnionTypes(Entity, IList, CurrentObjectType, Field, false, Index,
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
  const ArrayType *AT = SemaRef.Context.getAsArrayType(CurrentObjectType);
  if (!AT) {
    SemaRef.Diag(D->getLBracketLoc(), diag::err_array_designator_non_array)
      << CurrentObjectType;
    ++Index;
    return true;
  }

  Expr *IndexExpr = 0;
  llvm::APSInt DesignatedStartIndex, DesignatedEndIndex;
  if (D->isArrayDesignator()) {
    IndexExpr = DIE->getArrayIndex(*D);
    DesignatedStartIndex = IndexExpr->EvaluateAsInt(SemaRef.Context);
    DesignatedEndIndex = DesignatedStartIndex;
  } else {
    assert(D->isArrayRangeDesignator() && "Need array-range designator");


    DesignatedStartIndex =
      DIE->getArrayRangeStart(*D)->EvaluateAsInt(SemaRef.Context);
    DesignatedEndIndex =
      DIE->getArrayRangeEnd(*D)->EvaluateAsInt(SemaRef.Context);
    IndexExpr = DIE->getArrayRangeEnd(*D);

    if (DesignatedStartIndex.getZExtValue() !=DesignatedEndIndex.getZExtValue())
      FullyStructuredList->sawArrayRangeDesignator();
  }

  if (isa<ConstantArrayType>(AT)) {
    llvm::APSInt MaxElements(cast<ConstantArrayType>(AT)->getSize(), false);
    DesignatedStartIndex.extOrTrunc(MaxElements.getBitWidth());
    DesignatedStartIndex.setIsUnsigned(MaxElements.isUnsigned());
    DesignatedEndIndex.extOrTrunc(MaxElements.getBitWidth());
    DesignatedEndIndex.setIsUnsigned(MaxElements.isUnsigned());
    if (DesignatedEndIndex >= MaxElements) {
      SemaRef.Diag(IndexExpr->getSourceRange().getBegin(),
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
    else if (DesignatedStartIndex.getBitWidth() <
             DesignatedEndIndex.getBitWidth())
      DesignatedStartIndex.extend(DesignatedEndIndex.getBitWidth());
    DesignatedStartIndex.setIsUnsigned(true);
    DesignatedEndIndex.setIsUnsigned(true);
  }

  // Make sure that our non-designated initializer list has space
  // for a subobject corresponding to this array element.
  if (DesignatedEndIndex.getZExtValue() >= StructuredList->getNumInits())
    StructuredList->resizeInits(SemaRef.Context,
                                DesignatedEndIndex.getZExtValue() + 1);

  // Repeatedly perform subobject initializations in the range
  // [DesignatedStartIndex, DesignatedEndIndex].

  // Move to the next designator
  unsigned ElementIndex = DesignatedStartIndex.getZExtValue();
  unsigned OldIndex = Index;
  
  InitializedEntity ElementEntity =
    InitializedEntity::InitializeElement(SemaRef.Context, 0, Entity);

  while (DesignatedStartIndex <= DesignatedEndIndex) {
    // Recurse to check later designated subobjects.
    QualType ElementType = AT->getElementType();
    Index = OldIndex;
    
    ElementEntity.setElementIndex(ElementIndex);
    if (CheckDesignatedInitializer(ElementEntity, IList, DIE, DesigIdx + 1, 
                                   ElementType, 0, 0, Index, 
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
  CheckArrayType(Entity, IList, CurrentObjectType, DesignatedStartIndex, 
                 /*SubobjectIsDesignatorContext=*/false, Index,
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
    SemaRef.Diag(InitRange.getBegin(),
                 diag::warn_subobject_initializer_overrides)
      << InitRange;
    SemaRef.Diag(ExistingInit->getSourceRange().getBegin(),
                  diag::note_previous_initializer)
      << /*FIXME:has side effects=*/0
      << ExistingInit->getSourceRange();
  }

  InitListExpr *Result
    = new (SemaRef.Context) InitListExpr(SemaRef.Context,
                                         InitRange.getBegin(), 0, 0,
                                         InitRange.getEnd());

  Result->setType(CurrentObjectType.getNonLValueExprType(SemaRef.Context));

  // Pre-allocate storage for the structured initializer list.
  unsigned NumElements = 0;
  unsigned NumInits = 0;
  if (!StructuredList)
    NumInits = IList->getNumInits();
  else if (Index < IList->getNumInits()) {
    if (InitListExpr *SubList = dyn_cast<InitListExpr>(IList->getInit(Index)))
      NumInits = SubList->getNumInits();
  }

  if (const ArrayType *AType
      = SemaRef.Context.getAsArrayType(CurrentObjectType)) {
    if (const ConstantArrayType *CAType = dyn_cast<ConstantArrayType>(AType)) {
      NumElements = CAType->getSize().getZExtValue();
      // Simple heuristic so that we don't allocate a very large
      // initializer with many empty entries at the end.
      if (NumInits && NumElements > NumInits)
        NumElements = 0;
    }
  } else if (const VectorType *VType = CurrentObjectType->getAs<VectorType>())
    NumElements = VType->getNumElements();
  else if (const RecordType *RType = CurrentObjectType->getAs<RecordType>()) {
    RecordDecl *RDecl = RType->getDecl();
    if (RDecl->isUnion())
      NumElements = 1;
    else
      NumElements = std::distance(RDecl->field_begin(),
                                  RDecl->field_end());
  }

  if (NumElements < NumInits)
    NumElements = IList->getNumInits();

  Result->reserveInits(SemaRef.Context, NumElements);

  // Link this new initializer list into the structured initializer
  // lists.
  if (StructuredList)
    StructuredList->updateInit(SemaRef.Context, StructuredIndex, Result);
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

  if (Expr *PrevInit = StructuredList->updateInit(SemaRef.Context,
                                                  StructuredIndex, expr)) {
    // This initializer overwrites a previous initializer. Warn.
    SemaRef.Diag(expr->getSourceRange().getBegin(),
                  diag::warn_initializer_overrides)
      << expr->getSourceRange();
    SemaRef.Diag(PrevInit->getSourceRange().getBegin(),
                  diag::note_previous_initializer)
      << /*FIXME:has side effects=*/0
      << PrevInit->getSourceRange();
  }

  ++StructuredIndex;
}

/// Check that the given Index expression is a valid array designator
/// value. This is essentailly just a wrapper around
/// VerifyIntegerConstantExpression that also checks for negative values
/// and produces a reasonable diagnostic if there is a
/// failure. Returns true if there was an error, false otherwise.  If
/// everything went okay, Value will receive the value of the constant
/// expression.
static bool
CheckArrayDesignatorExpr(Sema &S, Expr *Index, llvm::APSInt &Value) {
  SourceLocation Loc = Index->getSourceRange().getBegin();

  // Make sure this is an integer constant expression.
  if (S.VerifyIntegerConstantExpression(Index, &Value))
    return true;

  if (Value.isSigned() && Value.isNegative())
    return S.Diag(Loc, diag::err_array_designator_negative)
      << Value.toString(10) << Index->getSourceRange();

  Value.setIsUnsigned(true);
  return false;
}

Sema::OwningExprResult Sema::ActOnDesignatedInitializer(Designation &Desig,
                                                        SourceLocation Loc,
                                                        bool GNUSyntax,
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
      if (!Index->isTypeDependent() &&
          !Index->isValueDependent() &&
          CheckArrayDesignatorExpr(*this, Index, IndexValue))
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
      bool StartDependent = StartIndex->isTypeDependent() ||
                            StartIndex->isValueDependent();
      bool EndDependent = EndIndex->isTypeDependent() ||
                          EndIndex->isValueDependent();
      if ((!StartDependent &&
           CheckArrayDesignatorExpr(*this, StartIndex, StartValue)) ||
          (!EndDependent &&
           CheckArrayDesignatorExpr(*this, EndIndex, EndValue)))
        Invalid = true;
      else {
        // Make sure we're comparing values with the same bit width.
        if (StartDependent || EndDependent) {
          // Nothing to compute.
        } else if (StartValue.getBitWidth() > EndValue.getBitWidth())
          EndValue.extend(StartValue.getBitWidth());
        else if (StartValue.getBitWidth() < EndValue.getBitWidth())
          StartValue.extend(EndValue.getBitWidth());

        if (!StartDependent && !EndDependent && EndValue < StartValue) {
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
    = DesignatedInitExpr::Create(Context,
                                 Designators.data(), Designators.size(),
                                 InitExpressions.data(), InitExpressions.size(),
                                 Loc, GNUSyntax, Init.takeAs<Expr>());
  return Owned(DIE);
}

bool Sema::CheckInitList(const InitializedEntity &Entity,
                         InitListExpr *&InitList, QualType &DeclType) {
  InitListChecker CheckInitList(*this, Entity, InitList, DeclType);
  if (!CheckInitList.HadError())
    InitList = CheckInitList.getFullyStructuredList();

  return CheckInitList.HadError();
}

//===----------------------------------------------------------------------===//
// Initialization entity
//===----------------------------------------------------------------------===//

InitializedEntity::InitializedEntity(ASTContext &Context, unsigned Index, 
                                     const InitializedEntity &Parent)
  : Parent(&Parent), Index(Index) 
{
  if (const ArrayType *AT = Context.getAsArrayType(Parent.getType())) {
    Kind = EK_ArrayElement;
    Type = AT->getElementType();
  } else {
    Kind = EK_VectorElement;
    Type = Parent.getType()->getAs<VectorType>()->getElementType();
  }
}

InitializedEntity InitializedEntity::InitializeBase(ASTContext &Context, 
                                                    CXXBaseSpecifier *Base,
                                                    bool IsInheritedVirtualBase)
{
  InitializedEntity Result;
  Result.Kind = EK_Base;
  Result.Base = reinterpret_cast<uintptr_t>(Base);
  if (IsInheritedVirtualBase)
    Result.Base |= 0x01;
  
  Result.Type = Base->getType();
  return Result;
}

DeclarationName InitializedEntity::getName() const {
  switch (getKind()) {
  case EK_Parameter:
    if (!VariableOrMember)
      return DeclarationName();
    // Fall through

  case EK_Variable:
  case EK_Member:
    return VariableOrMember->getDeclName();

  case EK_Result:
  case EK_Exception:
  case EK_New:
  case EK_Temporary:
  case EK_Base:
  case EK_ArrayElement:
  case EK_VectorElement:
  case EK_BlockElement:
    return DeclarationName();
  }
  
  // Silence GCC warning
  return DeclarationName();
}

DeclaratorDecl *InitializedEntity::getDecl() const {
  switch (getKind()) {
  case EK_Variable:
  case EK_Parameter:
  case EK_Member:
    return VariableOrMember;

  case EK_Result:
  case EK_Exception:
  case EK_New:
  case EK_Temporary:
  case EK_Base:
  case EK_ArrayElement:
  case EK_VectorElement:
  case EK_BlockElement:
    return 0;
  }
  
  // Silence GCC warning
  return 0;
}

bool InitializedEntity::allowsNRVO() const {
  switch (getKind()) {
  case EK_Result:
  case EK_Exception:
    return LocAndNRVO.NRVO;
    
  case EK_Variable:
  case EK_Parameter:
  case EK_Member:
  case EK_New:
  case EK_Temporary:
  case EK_Base:
  case EK_ArrayElement:
  case EK_VectorElement:
  case EK_BlockElement:
    break;
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Initialization sequence
//===----------------------------------------------------------------------===//

void InitializationSequence::Step::Destroy() {
  switch (Kind) {
  case SK_ResolveAddressOfOverloadedFunction:
  case SK_CastDerivedToBaseRValue:
  case SK_CastDerivedToBaseXValue:
  case SK_CastDerivedToBaseLValue:
  case SK_BindReference:
  case SK_BindReferenceToTemporary:
  case SK_ExtraneousCopyToTemporary:
  case SK_UserConversion:
  case SK_QualificationConversionRValue:
  case SK_QualificationConversionXValue:
  case SK_QualificationConversionLValue:
  case SK_ListInitialization:
  case SK_ConstructorInitialization:
  case SK_ZeroInitialization:
  case SK_CAssignment:
  case SK_StringInit:
  case SK_ObjCObjectConversion:
    break;
    
  case SK_ConversionSequence:
    delete ICS;
  }
}

bool InitializationSequence::isDirectReferenceBinding() const {
  return getKind() == ReferenceBinding && Steps.back().Kind == SK_BindReference;
}

bool InitializationSequence::isAmbiguous() const {
  if (getKind() != FailedSequence)
    return false;
  
  switch (getFailureKind()) {
  case FK_TooManyInitsForReference:
  case FK_ArrayNeedsInitList:
  case FK_ArrayNeedsInitListOrStringLiteral:
  case FK_AddressOfOverloadFailed: // FIXME: Could do better
  case FK_NonConstLValueReferenceBindingToTemporary:
  case FK_NonConstLValueReferenceBindingToUnrelated:
  case FK_RValueReferenceBindingToLValue:
  case FK_ReferenceInitDropsQualifiers:
  case FK_ReferenceInitFailed:
  case FK_ConversionFailed:
  case FK_TooManyInitsForScalar:
  case FK_ReferenceBindingToInitList:
  case FK_InitListBadDestinationType:
  case FK_DefaultInitOfConst:
  case FK_Incomplete:
    return false;
    
  case FK_ReferenceInitOverloadFailed:
  case FK_UserConversionOverloadFailed:
  case FK_ConstructorOverloadFailed:
    return FailedOverloadResult == OR_Ambiguous;
  }
  
  return false;
}

bool InitializationSequence::isConstructorInitialization() const {
  return !Steps.empty() && Steps.back().Kind == SK_ConstructorInitialization;
}

void InitializationSequence::AddAddressOverloadResolutionStep(
                                                      FunctionDecl *Function,
                                                      DeclAccessPair Found) {
  Step S;
  S.Kind = SK_ResolveAddressOfOverloadedFunction;
  S.Type = Function->getType();
  S.Function.Function = Function;
  S.Function.FoundDecl = Found;
  Steps.push_back(S);
}

void InitializationSequence::AddDerivedToBaseCastStep(QualType BaseType, 
                                    ImplicitCastExpr::ResultCategory Category) {
  Step S;
  switch (Category) {
  case ImplicitCastExpr::RValue: S.Kind = SK_CastDerivedToBaseRValue; break;
  case ImplicitCastExpr::XValue: S.Kind = SK_CastDerivedToBaseXValue; break;
  case ImplicitCastExpr::LValue: S.Kind = SK_CastDerivedToBaseLValue; break;
  default: llvm_unreachable("No such category");
  }
  S.Type = BaseType;
  Steps.push_back(S);
}

void InitializationSequence::AddReferenceBindingStep(QualType T, 
                                                     bool BindingTemporary) {
  Step S;
  S.Kind = BindingTemporary? SK_BindReferenceToTemporary : SK_BindReference;
  S.Type = T;
  Steps.push_back(S);
}

void InitializationSequence::AddExtraneousCopyToTemporary(QualType T) {
  Step S;
  S.Kind = SK_ExtraneousCopyToTemporary;
  S.Type = T;
  Steps.push_back(S);
}

void InitializationSequence::AddUserConversionStep(FunctionDecl *Function,
                                                   DeclAccessPair FoundDecl,
                                                   QualType T) {
  Step S;
  S.Kind = SK_UserConversion;
  S.Type = T;
  S.Function.Function = Function;
  S.Function.FoundDecl = FoundDecl;
  Steps.push_back(S);
}

void InitializationSequence::AddQualificationConversionStep(QualType Ty,
                                    ImplicitCastExpr::ResultCategory Category) {
  Step S;
  switch (Category) {
  case ImplicitCastExpr::RValue:
    S.Kind = SK_QualificationConversionRValue;
    break;
  case ImplicitCastExpr::XValue:
    S.Kind = SK_QualificationConversionXValue;
    break;
  case ImplicitCastExpr::LValue:
    S.Kind = SK_QualificationConversionLValue;
    break;
  default: llvm_unreachable("No such category");
  }
  S.Type = Ty;
  Steps.push_back(S);
}

void InitializationSequence::AddConversionSequenceStep(
                                       const ImplicitConversionSequence &ICS,
                                                       QualType T) {
  Step S;
  S.Kind = SK_ConversionSequence;
  S.Type = T;
  S.ICS = new ImplicitConversionSequence(ICS);
  Steps.push_back(S);
}

void InitializationSequence::AddListInitializationStep(QualType T) {
  Step S;
  S.Kind = SK_ListInitialization;
  S.Type = T;
  Steps.push_back(S);
}

void 
InitializationSequence::AddConstructorInitializationStep(
                                              CXXConstructorDecl *Constructor,
                                                       AccessSpecifier Access,
                                                         QualType T) {
  Step S;
  S.Kind = SK_ConstructorInitialization;
  S.Type = T;
  S.Function.Function = Constructor;
  S.Function.FoundDecl = DeclAccessPair::make(Constructor, Access);
  Steps.push_back(S);
}

void InitializationSequence::AddZeroInitializationStep(QualType T) {
  Step S;
  S.Kind = SK_ZeroInitialization;
  S.Type = T;
  Steps.push_back(S);
}

void InitializationSequence::AddCAssignmentStep(QualType T) {
  Step S;
  S.Kind = SK_CAssignment;
  S.Type = T;
  Steps.push_back(S);
}

void InitializationSequence::AddStringInitStep(QualType T) {
  Step S;
  S.Kind = SK_StringInit;
  S.Type = T;
  Steps.push_back(S);
}

void InitializationSequence::AddObjCObjectConversionStep(QualType T) {
  Step S;
  S.Kind = SK_ObjCObjectConversion;
  S.Type = T;
  Steps.push_back(S);
}

void InitializationSequence::SetOverloadFailure(FailureKind Failure, 
                                                OverloadingResult Result) {
  SequenceKind = FailedSequence;
  this->Failure = Failure;
  this->FailedOverloadResult = Result;
}

//===----------------------------------------------------------------------===//
// Attempt initialization
//===----------------------------------------------------------------------===//

/// \brief Attempt list initialization (C++0x [dcl.init.list]) 
static void TryListInitialization(Sema &S, 
                                  const InitializedEntity &Entity,
                                  const InitializationKind &Kind,
                                  InitListExpr *InitList,
                                  InitializationSequence &Sequence) {
  // FIXME: We only perform rudimentary checking of list
  // initializations at this point, then assume that any list
  // initialization of an array, aggregate, or scalar will be
  // well-formed. When we actually "perform" list initialization, we'll
  // do all of the necessary checking.  C++0x initializer lists will
  // force us to perform more checking here.
  Sequence.setSequenceKind(InitializationSequence::ListInitialization);

  QualType DestType = Entity.getType();

  // C++ [dcl.init]p13:
  //   If T is a scalar type, then a declaration of the form 
  //
  //     T x = { a };
  //
  //   is equivalent to
  //
  //     T x = a;
  if (DestType->isScalarType()) {
    if (InitList->getNumInits() > 1 && S.getLangOptions().CPlusPlus) {
      Sequence.SetFailed(InitializationSequence::FK_TooManyInitsForScalar);
      return;
    }

    // Assume scalar initialization from a single value works.
  } else if (DestType->isAggregateType()) {
    // Assume aggregate initialization works.
  } else if (DestType->isVectorType()) {
    // Assume vector initialization works.
  } else if (DestType->isReferenceType()) {
    // FIXME: C++0x defines behavior for this.
    Sequence.SetFailed(InitializationSequence::FK_ReferenceBindingToInitList);
    return;
  } else if (DestType->isRecordType()) {
    // FIXME: C++0x defines behavior for this
    Sequence.SetFailed(InitializationSequence::FK_InitListBadDestinationType);
  }

  // Add a general "list initialization" step.
  Sequence.AddListInitializationStep(DestType);
}

/// \brief Try a reference initialization that involves calling a conversion
/// function.
static OverloadingResult TryRefInitWithConversionFunction(Sema &S,
                                             const InitializedEntity &Entity,
                                             const InitializationKind &Kind,
                                                          Expr *Initializer,
                                                          bool AllowRValues,
                                             InitializationSequence &Sequence) {
  QualType DestType = Entity.getType();
  QualType cv1T1 = DestType->getAs<ReferenceType>()->getPointeeType();
  QualType T1 = cv1T1.getUnqualifiedType();
  QualType cv2T2 = Initializer->getType();
  QualType T2 = cv2T2.getUnqualifiedType();

  bool DerivedToBase;
  bool ObjCConversion;
  assert(!S.CompareReferenceRelationship(Initializer->getLocStart(), 
                                         T1, T2, DerivedToBase,
                                         ObjCConversion) &&
         "Must have incompatible references when binding via conversion");
  (void)DerivedToBase;
  (void)ObjCConversion;

  // Build the candidate set directly in the initialization sequence
  // structure, so that it will persist if we fail.
  OverloadCandidateSet &CandidateSet = Sequence.getFailedCandidateSet();
  CandidateSet.clear();

  // Determine whether we are allowed to call explicit constructors or
  // explicit conversion operators.
  bool AllowExplicit = Kind.getKind() == InitializationKind::IK_Direct;
  
  const RecordType *T1RecordType = 0;
  if (AllowRValues && (T1RecordType = T1->getAs<RecordType>()) &&
      !S.RequireCompleteType(Kind.getLocation(), T1, 0)) {
    // The type we're converting to is a class type. Enumerate its constructors
    // to see if there is a suitable conversion.
    CXXRecordDecl *T1RecordDecl = cast<CXXRecordDecl>(T1RecordType->getDecl());
    DeclContext::lookup_iterator Con, ConEnd;
    for (llvm::tie(Con, ConEnd) = S.LookupConstructors(T1RecordDecl);
         Con != ConEnd; ++Con) {
      NamedDecl *D = *Con;
      DeclAccessPair FoundDecl = DeclAccessPair::make(D, D->getAccess());

      // Find the constructor (which may be a template).
      CXXConstructorDecl *Constructor = 0;
      FunctionTemplateDecl *ConstructorTmpl = dyn_cast<FunctionTemplateDecl>(D);
      if (ConstructorTmpl)
        Constructor = cast<CXXConstructorDecl>(
                                         ConstructorTmpl->getTemplatedDecl());
      else
        Constructor = cast<CXXConstructorDecl>(D);
      
      if (!Constructor->isInvalidDecl() &&
          Constructor->isConvertingConstructor(AllowExplicit)) {
        if (ConstructorTmpl)
          S.AddTemplateOverloadCandidate(ConstructorTmpl, FoundDecl,
                                         /*ExplicitArgs*/ 0,
                                         &Initializer, 1, CandidateSet);
        else
          S.AddOverloadCandidate(Constructor, FoundDecl,
                                 &Initializer, 1, CandidateSet);
      }
    }    
  }
  
  const RecordType *T2RecordType = 0;
  if ((T2RecordType = T2->getAs<RecordType>()) &&
      !S.RequireCompleteType(Kind.getLocation(), T2, 0)) {
    // The type we're converting from is a class type, enumerate its conversion
    // functions.
    CXXRecordDecl *T2RecordDecl = cast<CXXRecordDecl>(T2RecordType->getDecl());

    // Determine the type we are converting to. If we are allowed to
    // convert to an rvalue, take the type that the destination type
    // refers to.
    QualType ToType = AllowRValues? cv1T1 : DestType;

    const UnresolvedSetImpl *Conversions
      = T2RecordDecl->getVisibleConversionFunctions();
    for (UnresolvedSetImpl::const_iterator I = Conversions->begin(),
           E = Conversions->end(); I != E; ++I) {
      NamedDecl *D = *I;
      CXXRecordDecl *ActingDC = cast<CXXRecordDecl>(D->getDeclContext());
      if (isa<UsingShadowDecl>(D))
        D = cast<UsingShadowDecl>(D)->getTargetDecl();
      
      FunctionTemplateDecl *ConvTemplate = dyn_cast<FunctionTemplateDecl>(D);
      CXXConversionDecl *Conv;
      if (ConvTemplate)
        Conv = cast<CXXConversionDecl>(ConvTemplate->getTemplatedDecl());
      else
        Conv = cast<CXXConversionDecl>(D);
      
      // If the conversion function doesn't return a reference type,
      // it can't be considered for this conversion unless we're allowed to
      // consider rvalues.
      // FIXME: Do we need to make sure that we only consider conversion 
      // candidates with reference-compatible results? That might be needed to 
      // break recursion.
      if ((AllowExplicit || !Conv->isExplicit()) &&
          (AllowRValues || Conv->getConversionType()->isLValueReferenceType())){
        if (ConvTemplate)
          S.AddTemplateConversionCandidate(ConvTemplate, I.getPair(),
                                           ActingDC, Initializer,
                                           ToType, CandidateSet);
        else
          S.AddConversionCandidate(Conv, I.getPair(), ActingDC,
                                   Initializer, ToType, CandidateSet);
      }
    }
  }
  
  SourceLocation DeclLoc = Initializer->getLocStart();

  // Perform overload resolution. If it fails, return the failed result.  
  OverloadCandidateSet::iterator Best;
  if (OverloadingResult Result 
        = S.BestViableFunction(CandidateSet, DeclLoc, Best))
    return Result;

  FunctionDecl *Function = Best->Function;

  // Compute the returned type of the conversion.
  if (isa<CXXConversionDecl>(Function))
    T2 = Function->getResultType();
  else
    T2 = cv1T1;

  // Add the user-defined conversion step.
  Sequence.AddUserConversionStep(Function, Best->FoundDecl,
                                 T2.getNonLValueExprType(S.Context));

  // Determine whether we need to perform derived-to-base or 
  // cv-qualification adjustments.
  ImplicitCastExpr::ResultCategory Category = ImplicitCastExpr::RValue;
  if (T2->isLValueReferenceType())
    Category = ImplicitCastExpr::LValue;
  else if (const RValueReferenceType *RRef = T2->getAs<RValueReferenceType>())
    Category = RRef->getPointeeType()->isFunctionType() ?
        ImplicitCastExpr::LValue : ImplicitCastExpr::XValue;

  bool NewDerivedToBase = false;
  bool NewObjCConversion = false;
  Sema::ReferenceCompareResult NewRefRelationship
    = S.CompareReferenceRelationship(DeclLoc, T1, 
                                     T2.getNonLValueExprType(S.Context),
                                     NewDerivedToBase, NewObjCConversion);
  if (NewRefRelationship == Sema::Ref_Incompatible) {
    // If the type we've converted to is not reference-related to the
    // type we're looking for, then there is another conversion step
    // we need to perform to produce a temporary of the right type
    // that we'll be binding to.
    ImplicitConversionSequence ICS;
    ICS.setStandard();
    ICS.Standard = Best->FinalConversion;
    T2 = ICS.Standard.getToType(2);
    Sequence.AddConversionSequenceStep(ICS, T2);
  } else if (NewDerivedToBase)
    Sequence.AddDerivedToBaseCastStep(
                                S.Context.getQualifiedType(T1,
                                  T2.getNonReferenceType().getQualifiers()), 
                                      Category);
  else if (NewObjCConversion)
    Sequence.AddObjCObjectConversionStep(
                                S.Context.getQualifiedType(T1,
                                  T2.getNonReferenceType().getQualifiers()));

  if (cv1T1.getQualifiers() != T2.getNonReferenceType().getQualifiers())
    Sequence.AddQualificationConversionStep(cv1T1, Category);
  
  Sequence.AddReferenceBindingStep(cv1T1, !T2->isReferenceType());
  return OR_Success;
}
  
/// \brief Attempt reference initialization (C++0x [dcl.init.ref]) 
static void TryReferenceInitialization(Sema &S, 
                                       const InitializedEntity &Entity,
                                       const InitializationKind &Kind,
                                       Expr *Initializer,
                                       InitializationSequence &Sequence) {
  Sequence.setSequenceKind(InitializationSequence::ReferenceBinding);

  QualType DestType = Entity.getType();
  QualType cv1T1 = DestType->getAs<ReferenceType>()->getPointeeType();
  Qualifiers T1Quals;
  QualType T1 = S.Context.getUnqualifiedArrayType(cv1T1, T1Quals);
  QualType cv2T2 = Initializer->getType();
  Qualifiers T2Quals;
  QualType T2 = S.Context.getUnqualifiedArrayType(cv2T2, T2Quals);
  SourceLocation DeclLoc = Initializer->getLocStart();

  // If the initializer is the address of an overloaded function, try
  // to resolve the overloaded function. If all goes well, T2 is the
  // type of the resulting function.
  if (S.Context.getCanonicalType(T2) == S.Context.OverloadTy) {
    DeclAccessPair Found;
    FunctionDecl *Fn = S.ResolveAddressOfOverloadedFunction(Initializer, 
                                                            T1,
                                                            false,
                                                            Found);
    if (!Fn) {
      Sequence.SetFailed(InitializationSequence::FK_AddressOfOverloadFailed);
      return;
    }

    Sequence.AddAddressOverloadResolutionStep(Fn, Found);
    cv2T2 = Fn->getType();
    T2 = cv2T2.getUnqualifiedType();
  }

  // Compute some basic properties of the types and the initializer.
  bool isLValueRef = DestType->isLValueReferenceType();
  bool isRValueRef = !isLValueRef;
  bool DerivedToBase = false;
  bool ObjCConversion = false;
  Expr::Classification InitCategory = Initializer->Classify(S.Context);
  Sema::ReferenceCompareResult RefRelationship
    = S.CompareReferenceRelationship(DeclLoc, cv1T1, cv2T2, DerivedToBase,
                                     ObjCConversion);

  // C++0x [dcl.init.ref]p5:
  //   A reference to type "cv1 T1" is initialized by an expression of type 
  //   "cv2 T2" as follows:
  //
  //     - If the reference is an lvalue reference and the initializer 
  //       expression
  // Note the analogous bullet points for rvlaue refs to functions. Because
  // there are no function rvalues in C++, rvalue refs to functions are treated
  // like lvalue refs.
  OverloadingResult ConvOvlResult = OR_Success;
  bool T1Function = T1->isFunctionType();
  if (isLValueRef || T1Function) {
    if (InitCategory.isLValue() && 
        RefRelationship >= Sema::Ref_Compatible_With_Added_Qualification) {
      //   - is an lvalue (but is not a bit-field), and "cv1 T1" is 
      //     reference-compatible with "cv2 T2," or
      //
      // Per C++ [over.best.ics]p2, we don't diagnose whether the lvalue is a 
      // bit-field when we're determining whether the reference initialization
      // can occur. However, we do pay attention to whether it is a bit-field
      // to decide whether we're actually binding to a temporary created from
      // the bit-field.
      if (DerivedToBase)
        Sequence.AddDerivedToBaseCastStep(
                         S.Context.getQualifiedType(T1, T2Quals), 
                         ImplicitCastExpr::LValue);
      else if (ObjCConversion)
        Sequence.AddObjCObjectConversionStep(
                                     S.Context.getQualifiedType(T1, T2Quals));

      if (T1Quals != T2Quals)
        Sequence.AddQualificationConversionStep(cv1T1,ImplicitCastExpr::LValue);
      bool BindingTemporary = T1Quals.hasConst() && !T1Quals.hasVolatile() &&
        (Initializer->getBitField() || Initializer->refersToVectorElement());
      Sequence.AddReferenceBindingStep(cv1T1, BindingTemporary);
      return;
    }
    
    //     - has a class type (i.e., T2 is a class type), where T1 is not 
    //       reference-related to T2, and can be implicitly converted to an 
    //       lvalue of type "cv3 T3," where "cv1 T1" is reference-compatible 
    //       with "cv3 T3" (this conversion is selected by enumerating the 
    //       applicable conversion functions (13.3.1.6) and choosing the best
    //       one through overload resolution (13.3)),
    // If we have an rvalue ref to function type here, the rhs must be
    // an rvalue.
    if (RefRelationship == Sema::Ref_Incompatible && T2->isRecordType() &&
        (isLValueRef || InitCategory.isRValue())) {
      ConvOvlResult = TryRefInitWithConversionFunction(S, Entity, Kind, 
                                                       Initializer,
                                                   /*AllowRValues=*/isRValueRef,
                                                       Sequence);
      if (ConvOvlResult == OR_Success)
        return;
      if (ConvOvlResult != OR_No_Viable_Function) {
        Sequence.SetOverloadFailure(
                      InitializationSequence::FK_ReferenceInitOverloadFailed,
                                    ConvOvlResult);
      }
    }
  }

  //     - Otherwise, the reference shall be an lvalue reference to a 
  //       non-volatile const type (i.e., cv1 shall be const), or the reference
  //       shall be an rvalue reference and the initializer expression shall 
  //       be an rvalue or have a function type.
  // We handled the function type stuff above.
  if (!((isLValueRef && T1Quals.hasConst() && !T1Quals.hasVolatile()) ||
        (isRValueRef && InitCategory.isRValue()))) {
    if (ConvOvlResult && !Sequence.getFailedCandidateSet().empty())
      Sequence.SetOverloadFailure(
                        InitializationSequence::FK_ReferenceInitOverloadFailed,
                                  ConvOvlResult);
    else if (isLValueRef)
      Sequence.SetFailed(InitCategory.isLValue()
        ? (RefRelationship == Sema::Ref_Related
             ? InitializationSequence::FK_ReferenceInitDropsQualifiers
             : InitializationSequence::FK_NonConstLValueReferenceBindingToUnrelated)
        : InitializationSequence::FK_NonConstLValueReferenceBindingToTemporary);
    else
      Sequence.SetFailed(
                    InitializationSequence::FK_RValueReferenceBindingToLValue);

    return;
  }

  //       - [If T1 is not a function type], if T2 is a class type and
  if (!T1Function && T2->isRecordType()) {
    bool isXValue = InitCategory.isXValue();
    //       - the initializer expression is an rvalue and "cv1 T1" is 
    //         reference-compatible with "cv2 T2", or
    if (InitCategory.isRValue() && 
        RefRelationship >= Sema::Ref_Compatible_With_Added_Qualification) {
      // The corresponding bullet in C++03 [dcl.init.ref]p5 gives the
      // compiler the freedom to perform a copy here or bind to the
      // object, while C++0x requires that we bind directly to the
      // object. Hence, we always bind to the object without making an
      // extra copy. However, in C++03 requires that we check for the
      // presence of a suitable copy constructor:
      //
      //   The constructor that would be used to make the copy shall
      //   be callable whether or not the copy is actually done.
      if (!S.getLangOptions().CPlusPlus0x)
        Sequence.AddExtraneousCopyToTemporary(cv2T2);

      if (DerivedToBase)
        Sequence.AddDerivedToBaseCastStep(
                         S.Context.getQualifiedType(T1, T2Quals), 
                         isXValue ? ImplicitCastExpr::XValue
                                  : ImplicitCastExpr::RValue);
      else if (ObjCConversion)
        Sequence.AddObjCObjectConversionStep(
                                     S.Context.getQualifiedType(T1, T2Quals));

      if (T1Quals != T2Quals)
        Sequence.AddQualificationConversionStep(cv1T1,
                     isXValue ? ImplicitCastExpr::XValue
                              : ImplicitCastExpr::RValue);
      Sequence.AddReferenceBindingStep(cv1T1, /*bindingTemporary=*/!isXValue);
      return;
    }

    //       - T1 is not reference-related to T2 and the initializer expression
    //         can be implicitly converted to an rvalue of type "cv3 T3" (this
    //         conversion is selected by enumerating the applicable conversion
    //         functions (13.3.1.6) and choosing the best one through overload 
    //         resolution (13.3)),
    if (RefRelationship == Sema::Ref_Incompatible) {
      ConvOvlResult = TryRefInitWithConversionFunction(S, Entity,
                                                       Kind, Initializer,
                                                       /*AllowRValues=*/true,
                                                       Sequence);
      if (ConvOvlResult)
        Sequence.SetOverloadFailure(
                      InitializationSequence::FK_ReferenceInitOverloadFailed,
                                    ConvOvlResult);
        
      return;
    }
    
    Sequence.SetFailed(InitializationSequence::FK_ReferenceInitDropsQualifiers);
    return;
  }
  
  //      - If the initializer expression is an rvalue, with T2 an array type,
  //        and "cv1 T1" is reference-compatible with "cv2 T2," the reference
  //        is bound to the object represented by the rvalue (see 3.10).
  // FIXME: How can an array type be reference-compatible with anything?
  // Don't we mean the element types of T1 and T2?
  
  //      - Otherwise, a temporary of type “cv1 T1” is created and initialized
  //        from the initializer expression using the rules for a non-reference
  //        copy initialization (8.5). The reference is then bound to the 
  //        temporary. [...]

  // Determine whether we are allowed to call explicit constructors or
  // explicit conversion operators.
  bool AllowExplicit = (Kind.getKind() == InitializationKind::IK_Direct);

  InitializedEntity TempEntity = InitializedEntity::InitializeTemporary(cv1T1);

  if (S.TryImplicitConversion(Sequence, TempEntity, Initializer,
                              /*SuppressUserConversions*/ false,
                              AllowExplicit,
                              /*FIXME:InOverloadResolution=*/false)) {
    // FIXME: Use the conversion function set stored in ICS to turn
    // this into an overloading ambiguity diagnostic. However, we need
    // to keep that set as an OverloadCandidateSet rather than as some
    // other kind of set.
    if (ConvOvlResult && !Sequence.getFailedCandidateSet().empty())
      Sequence.SetOverloadFailure(
                        InitializationSequence::FK_ReferenceInitOverloadFailed,
                                  ConvOvlResult);
    else
      Sequence.SetFailed(InitializationSequence::FK_ReferenceInitFailed);
    return;
  }

  //        [...] If T1 is reference-related to T2, cv1 must be the
  //        same cv-qualification as, or greater cv-qualification
  //        than, cv2; otherwise, the program is ill-formed.
  unsigned T1CVRQuals = T1Quals.getCVRQualifiers();
  unsigned T2CVRQuals = T2Quals.getCVRQualifiers();
  if (RefRelationship == Sema::Ref_Related && 
      (T1CVRQuals | T2CVRQuals) != T1CVRQuals) {
    Sequence.SetFailed(InitializationSequence::FK_ReferenceInitDropsQualifiers);
    return;
  }

  Sequence.AddReferenceBindingStep(cv1T1, /*bindingTemporary=*/true);
  return;
}

/// \brief Attempt character array initialization from a string literal
/// (C++ [dcl.init.string], C99 6.7.8). 
static void TryStringLiteralInitialization(Sema &S, 
                                           const InitializedEntity &Entity,
                                           const InitializationKind &Kind,
                                           Expr *Initializer,
                                       InitializationSequence &Sequence) {
  Sequence.setSequenceKind(InitializationSequence::StringInit);
  Sequence.AddStringInitStep(Entity.getType());
}

/// \brief Attempt initialization by constructor (C++ [dcl.init]), which
/// enumerates the constructors of the initialized entity and performs overload
/// resolution to select the best.
static void TryConstructorInitialization(Sema &S, 
                                         const InitializedEntity &Entity,
                                         const InitializationKind &Kind,
                                         Expr **Args, unsigned NumArgs,
                                         QualType DestType,
                                         InitializationSequence &Sequence) {
  Sequence.setSequenceKind(InitializationSequence::ConstructorInitialization);
  
  // Build the candidate set directly in the initialization sequence
  // structure, so that it will persist if we fail.
  OverloadCandidateSet &CandidateSet = Sequence.getFailedCandidateSet();
  CandidateSet.clear();
  
  // Determine whether we are allowed to call explicit constructors or
  // explicit conversion operators.
  bool AllowExplicit = (Kind.getKind() == InitializationKind::IK_Direct ||
                        Kind.getKind() == InitializationKind::IK_Value ||
                        Kind.getKind() == InitializationKind::IK_Default);

  // The type we're constructing needs to be complete.
  if (S.RequireCompleteType(Kind.getLocation(), DestType, 0)) {
    Sequence.SetFailed(InitializationSequence::FK_Incomplete);
    return;
  }
  
  // The type we're converting to is a class type. Enumerate its constructors
  // to see if one is suitable.
  const RecordType *DestRecordType = DestType->getAs<RecordType>();
  assert(DestRecordType && "Constructor initialization requires record type");  
  CXXRecordDecl *DestRecordDecl
    = cast<CXXRecordDecl>(DestRecordType->getDecl());
    
  DeclContext::lookup_iterator Con, ConEnd;
  for (llvm::tie(Con, ConEnd) = S.LookupConstructors(DestRecordDecl);
       Con != ConEnd; ++Con) {
    NamedDecl *D = *Con;
    DeclAccessPair FoundDecl = DeclAccessPair::make(D, D->getAccess());
    bool SuppressUserConversions = false;
    
    // Find the constructor (which may be a template).
    CXXConstructorDecl *Constructor = 0;
    FunctionTemplateDecl *ConstructorTmpl = dyn_cast<FunctionTemplateDecl>(D);
    if (ConstructorTmpl)
      Constructor = cast<CXXConstructorDecl>(
                                           ConstructorTmpl->getTemplatedDecl());
    else {
      Constructor = cast<CXXConstructorDecl>(D);

      // If we're performing copy initialization using a copy constructor, we 
      // suppress user-defined conversions on the arguments.
      // FIXME: Move constructors?
      if (Kind.getKind() == InitializationKind::IK_Copy &&
          Constructor->isCopyConstructor())
        SuppressUserConversions = true;
    }
    
    if (!Constructor->isInvalidDecl() &&
        (AllowExplicit || !Constructor->isExplicit())) {
      if (ConstructorTmpl)
        S.AddTemplateOverloadCandidate(ConstructorTmpl, FoundDecl,
                                       /*ExplicitArgs*/ 0,
                                       Args, NumArgs, CandidateSet,
                                       SuppressUserConversions);
      else
        S.AddOverloadCandidate(Constructor, FoundDecl,
                               Args, NumArgs, CandidateSet,
                               SuppressUserConversions);
    }
  }    
    
  SourceLocation DeclLoc = Kind.getLocation();
  
  // Perform overload resolution. If it fails, return the failed result.  
  OverloadCandidateSet::iterator Best;
  if (OverloadingResult Result 
        = S.BestViableFunction(CandidateSet, DeclLoc, Best)) {
    Sequence.SetOverloadFailure(
                          InitializationSequence::FK_ConstructorOverloadFailed, 
                                Result);
    return;
  }

  // C++0x [dcl.init]p6:
  //   If a program calls for the default initialization of an object
  //   of a const-qualified type T, T shall be a class type with a
  //   user-provided default constructor.
  if (Kind.getKind() == InitializationKind::IK_Default &&
      Entity.getType().isConstQualified() &&
      cast<CXXConstructorDecl>(Best->Function)->isImplicit()) {
    Sequence.SetFailed(InitializationSequence::FK_DefaultInitOfConst);
    return;
  }

  // Add the constructor initialization step. Any cv-qualification conversion is
  // subsumed by the initialization.
  Sequence.AddConstructorInitializationStep(
                                      cast<CXXConstructorDecl>(Best->Function), 
                                      Best->FoundDecl.getAccess(),
                                      DestType);
}

/// \brief Attempt value initialization (C++ [dcl.init]p7).
static void TryValueInitialization(Sema &S, 
                                   const InitializedEntity &Entity,
                                   const InitializationKind &Kind,
                                   InitializationSequence &Sequence) {
  // C++ [dcl.init]p5:
  //
  //   To value-initialize an object of type T means:
  QualType T = Entity.getType();
  
  //     -- if T is an array type, then each element is value-initialized;
  while (const ArrayType *AT = S.Context.getAsArrayType(T))
    T = AT->getElementType();
  
  if (const RecordType *RT = T->getAs<RecordType>()) {
    if (CXXRecordDecl *ClassDecl = dyn_cast<CXXRecordDecl>(RT->getDecl())) {
      // -- if T is a class type (clause 9) with a user-declared
      //    constructor (12.1), then the default constructor for T is
      //    called (and the initialization is ill-formed if T has no
      //    accessible default constructor);
      //
      // FIXME: we really want to refer to a single subobject of the array,
      // but Entity doesn't have a way to capture that (yet).
      if (ClassDecl->hasUserDeclaredConstructor())
        return TryConstructorInitialization(S, Entity, Kind, 0, 0, T, Sequence);
      
      // -- if T is a (possibly cv-qualified) non-union class type
      //    without a user-provided constructor, then the object is
      //    zero-initialized and, if T’s implicitly-declared default
      //    constructor is non-trivial, that constructor is called.
      if ((ClassDecl->getTagKind() == TTK_Class ||
           ClassDecl->getTagKind() == TTK_Struct)) {
        Sequence.AddZeroInitializationStep(Entity.getType());
        return TryConstructorInitialization(S, Entity, Kind, 0, 0, T, Sequence);        
      }
    }
  }

  Sequence.AddZeroInitializationStep(Entity.getType());
  Sequence.setSequenceKind(InitializationSequence::ZeroInitialization);
}

/// \brief Attempt default initialization (C++ [dcl.init]p6).
static void TryDefaultInitialization(Sema &S,
                                     const InitializedEntity &Entity,
                                     const InitializationKind &Kind,
                                     InitializationSequence &Sequence) {
  assert(Kind.getKind() == InitializationKind::IK_Default);
  
  // C++ [dcl.init]p6:
  //   To default-initialize an object of type T means:
  //     - if T is an array type, each element is default-initialized;
  QualType DestType = Entity.getType();
  while (const ArrayType *Array = S.Context.getAsArrayType(DestType))
    DestType = Array->getElementType();
         
  //     - if T is a (possibly cv-qualified) class type (Clause 9), the default
  //       constructor for T is called (and the initialization is ill-formed if
  //       T has no accessible default constructor);
  if (DestType->isRecordType() && S.getLangOptions().CPlusPlus) {
    return TryConstructorInitialization(S, Entity, Kind, 0, 0, DestType,
                                        Sequence);
  }
  
  //     - otherwise, no initialization is performed.
  Sequence.setSequenceKind(InitializationSequence::NoInitialization);
  
  //   If a program calls for the default initialization of an object of
  //   a const-qualified type T, T shall be a class type with a user-provided 
  //   default constructor.
  if (DestType.isConstQualified() && S.getLangOptions().CPlusPlus)
    Sequence.SetFailed(InitializationSequence::FK_DefaultInitOfConst);
}

/// \brief Attempt a user-defined conversion between two types (C++ [dcl.init]),
/// which enumerates all conversion functions and performs overload resolution
/// to select the best.
static void TryUserDefinedConversion(Sema &S, 
                                     const InitializedEntity &Entity,
                                     const InitializationKind &Kind,
                                     Expr *Initializer,
                                     InitializationSequence &Sequence) {
  Sequence.setSequenceKind(InitializationSequence::UserDefinedConversion);
  
  QualType DestType = Entity.getType();
  assert(!DestType->isReferenceType() && "References are handled elsewhere");
  QualType SourceType = Initializer->getType();
  assert((DestType->isRecordType() || SourceType->isRecordType()) &&
         "Must have a class type to perform a user-defined conversion");
  
  // Build the candidate set directly in the initialization sequence
  // structure, so that it will persist if we fail.
  OverloadCandidateSet &CandidateSet = Sequence.getFailedCandidateSet();
  CandidateSet.clear();
  
  // Determine whether we are allowed to call explicit constructors or
  // explicit conversion operators.
  bool AllowExplicit = Kind.getKind() == InitializationKind::IK_Direct;
  
  if (const RecordType *DestRecordType = DestType->getAs<RecordType>()) {
    // The type we're converting to is a class type. Enumerate its constructors
    // to see if there is a suitable conversion.
    CXXRecordDecl *DestRecordDecl
      = cast<CXXRecordDecl>(DestRecordType->getDecl());
    
    // Try to complete the type we're converting to.
    if (!S.RequireCompleteType(Kind.getLocation(), DestType, 0)) {    
      DeclContext::lookup_iterator Con, ConEnd;
      for (llvm::tie(Con, ConEnd) = S.LookupConstructors(DestRecordDecl);
           Con != ConEnd; ++Con) {
        NamedDecl *D = *Con;
        DeclAccessPair FoundDecl = DeclAccessPair::make(D, D->getAccess());
        
        // Find the constructor (which may be a template).
        CXXConstructorDecl *Constructor = 0;
        FunctionTemplateDecl *ConstructorTmpl
          = dyn_cast<FunctionTemplateDecl>(D);
        if (ConstructorTmpl)
          Constructor = cast<CXXConstructorDecl>(
                                           ConstructorTmpl->getTemplatedDecl());
        else
          Constructor = cast<CXXConstructorDecl>(D);
        
        if (!Constructor->isInvalidDecl() &&
            Constructor->isConvertingConstructor(AllowExplicit)) {
          if (ConstructorTmpl)
            S.AddTemplateOverloadCandidate(ConstructorTmpl, FoundDecl,
                                           /*ExplicitArgs*/ 0,
                                           &Initializer, 1, CandidateSet,
                                           /*SuppressUserConversions=*/true);
          else
            S.AddOverloadCandidate(Constructor, FoundDecl,
                                   &Initializer, 1, CandidateSet,
                                   /*SuppressUserConversions=*/true);
        }
      }    
    }
  }

  SourceLocation DeclLoc = Initializer->getLocStart();

  if (const RecordType *SourceRecordType = SourceType->getAs<RecordType>()) {
    // The type we're converting from is a class type, enumerate its conversion
    // functions.

    // We can only enumerate the conversion functions for a complete type; if
    // the type isn't complete, simply skip this step.
    if (!S.RequireCompleteType(DeclLoc, SourceType, 0)) {
      CXXRecordDecl *SourceRecordDecl
        = cast<CXXRecordDecl>(SourceRecordType->getDecl());
      
      const UnresolvedSetImpl *Conversions
        = SourceRecordDecl->getVisibleConversionFunctions();
      for (UnresolvedSetImpl::const_iterator I = Conversions->begin(),
           E = Conversions->end(); 
           I != E; ++I) {
        NamedDecl *D = *I;
        CXXRecordDecl *ActingDC = cast<CXXRecordDecl>(D->getDeclContext());
        if (isa<UsingShadowDecl>(D))
          D = cast<UsingShadowDecl>(D)->getTargetDecl();
        
        FunctionTemplateDecl *ConvTemplate = dyn_cast<FunctionTemplateDecl>(D);
        CXXConversionDecl *Conv;
        if (ConvTemplate)
          Conv = cast<CXXConversionDecl>(ConvTemplate->getTemplatedDecl());
        else
          Conv = cast<CXXConversionDecl>(D);
        
        if (AllowExplicit || !Conv->isExplicit()) {
          if (ConvTemplate)
            S.AddTemplateConversionCandidate(ConvTemplate, I.getPair(),
                                             ActingDC, Initializer, DestType,
                                             CandidateSet);
          else
            S.AddConversionCandidate(Conv, I.getPair(), ActingDC,
                                     Initializer, DestType, CandidateSet);
        }
      }
    }
  }
  
  // Perform overload resolution. If it fails, return the failed result.  
  OverloadCandidateSet::iterator Best;
  if (OverloadingResult Result
        = S.BestViableFunction(CandidateSet, DeclLoc, Best)) {
    Sequence.SetOverloadFailure(
                        InitializationSequence::FK_UserConversionOverloadFailed, 
                                Result);
    return;
  }

  FunctionDecl *Function = Best->Function;
  
  if (isa<CXXConstructorDecl>(Function)) {
    // Add the user-defined conversion step. Any cv-qualification conversion is
    // subsumed by the initialization.
    Sequence.AddUserConversionStep(Function, Best->FoundDecl, DestType);
    return;
  }

  // Add the user-defined conversion step that calls the conversion function.
  QualType ConvType = Function->getCallResultType();
  if (ConvType->getAs<RecordType>()) {
    // If we're converting to a class type, there may be an copy if
    // the resulting temporary object (possible to create an object of
    // a base class type). That copy is not a separate conversion, so
    // we just make a note of the actual destination type (possibly a
    // base class of the type returned by the conversion function) and
    // let the user-defined conversion step handle the conversion.
    Sequence.AddUserConversionStep(Function, Best->FoundDecl, DestType);
    return;
  }

  Sequence.AddUserConversionStep(Function, Best->FoundDecl, ConvType);
    
  // If the conversion following the call to the conversion function
  // is interesting, add it as a separate step.
  if (Best->FinalConversion.First || Best->FinalConversion.Second ||
      Best->FinalConversion.Third) {
    ImplicitConversionSequence ICS;
    ICS.setStandard();
    ICS.Standard = Best->FinalConversion;
    Sequence.AddConversionSequenceStep(ICS, DestType);
  }
}

bool Sema::TryImplicitConversion(InitializationSequence &Sequence,
                                 const InitializedEntity &Entity,
                                 Expr *Initializer,
                                 bool SuppressUserConversions,
                                 bool AllowExplicitConversions,
                                 bool InOverloadResolution) {
  ImplicitConversionSequence ICS
    = TryImplicitConversion(Initializer, Entity.getType(),
                            SuppressUserConversions,
                            AllowExplicitConversions, 
                            InOverloadResolution);
  if (ICS.isBad()) return true;

  // Perform the actual conversion.
  Sequence.AddConversionSequenceStep(ICS, Entity.getType());
  return false;
}

InitializationSequence::InitializationSequence(Sema &S,
                                               const InitializedEntity &Entity,
                                               const InitializationKind &Kind,
                                               Expr **Args,
                                               unsigned NumArgs)
    : FailedCandidateSet(Kind.getLocation()) {
  ASTContext &Context = S.Context;
  
  // C++0x [dcl.init]p16:
  //   The semantics of initializers are as follows. The destination type is 
  //   the type of the object or reference being initialized and the source 
  //   type is the type of the initializer expression. The source type is not
  //   defined when the initializer is a braced-init-list or when it is a 
  //   parenthesized list of expressions.
  QualType DestType = Entity.getType();

  if (DestType->isDependentType() ||
      Expr::hasAnyTypeDependentArguments(Args, NumArgs)) {
    SequenceKind = DependentSequence;
    return;
  }

  QualType SourceType;
  Expr *Initializer = 0;
  if (NumArgs == 1) {
    Initializer = Args[0];
    if (!isa<InitListExpr>(Initializer))
      SourceType = Initializer->getType();
  }
  
  //     - If the initializer is a braced-init-list, the object is 
  //       list-initialized (8.5.4).
  if (InitListExpr *InitList = dyn_cast_or_null<InitListExpr>(Initializer)) {
    TryListInitialization(S, Entity, Kind, InitList, *this);
    return;
  }
  
  //     - If the destination type is a reference type, see 8.5.3.
  if (DestType->isReferenceType()) {
    // C++0x [dcl.init.ref]p1:
    //   A variable declared to be a T& or T&&, that is, "reference to type T"
    //   (8.3.2), shall be initialized by an object, or function, of type T or
    //   by an object that can be converted into a T.
    // (Therefore, multiple arguments are not permitted.)
    if (NumArgs != 1)
      SetFailed(FK_TooManyInitsForReference);
    else
      TryReferenceInitialization(S, Entity, Kind, Args[0], *this);
    return;
  }
  
  //     - If the destination type is an array of characters, an array of 
  //       char16_t, an array of char32_t, or an array of wchar_t, and the 
  //       initializer is a string literal, see 8.5.2.
  if (Initializer && IsStringInit(Initializer, DestType, Context)) {
    TryStringLiteralInitialization(S, Entity, Kind, Initializer, *this);
    return;
  }
  
  //     - If the initializer is (), the object is value-initialized.
  if (Kind.getKind() == InitializationKind::IK_Value ||
      (Kind.getKind() == InitializationKind::IK_Direct && NumArgs == 0)) {
    TryValueInitialization(S, Entity, Kind, *this);
    return;
  }
  
  // Handle default initialization.
  if (Kind.getKind() == InitializationKind::IK_Default){
    TryDefaultInitialization(S, Entity, Kind, *this);
    return;
  }

  //     - Otherwise, if the destination type is an array, the program is 
  //       ill-formed.
  if (const ArrayType *AT = Context.getAsArrayType(DestType)) {
    if (AT->getElementType()->isAnyCharacterType())
      SetFailed(FK_ArrayNeedsInitListOrStringLiteral);
    else
      SetFailed(FK_ArrayNeedsInitList);
    
    return;
  }

  // Handle initialization in C
  if (!S.getLangOptions().CPlusPlus) {
    setSequenceKind(CAssignment);
    AddCAssignmentStep(DestType);
    return;
  }
  
  //     - If the destination type is a (possibly cv-qualified) class type:
  if (DestType->isRecordType()) {
    //     - If the initialization is direct-initialization, or if it is 
    //       copy-initialization where the cv-unqualified version of the 
    //       source type is the same class as, or a derived class of, the 
    //       class of the destination, constructors are considered. [...]
    if (Kind.getKind() == InitializationKind::IK_Direct ||
        (Kind.getKind() == InitializationKind::IK_Copy &&
         (Context.hasSameUnqualifiedType(SourceType, DestType) ||
          S.IsDerivedFrom(SourceType, DestType))))
      TryConstructorInitialization(S, Entity, Kind, Args, NumArgs, 
                                   Entity.getType(), *this);
    //     - Otherwise (i.e., for the remaining copy-initialization cases), 
    //       user-defined conversion sequences that can convert from the source
    //       type to the destination type or (when a conversion function is 
    //       used) to a derived class thereof are enumerated as described in
    //       13.3.1.4, and the best one is chosen through overload resolution
    //       (13.3).
    else
      TryUserDefinedConversion(S, Entity, Kind, Initializer, *this);
    return;
  }
  
  if (NumArgs > 1) {
    SetFailed(FK_TooManyInitsForScalar);
    return;
  }
  assert(NumArgs == 1 && "Zero-argument case handled above");
  
  //    - Otherwise, if the source type is a (possibly cv-qualified) class 
  //      type, conversion functions are considered.
  if (!SourceType.isNull() && SourceType->isRecordType()) {
    TryUserDefinedConversion(S, Entity, Kind, Initializer, *this);
    return;
  }
  
  //    - Otherwise, the initial value of the object being initialized is the
  //      (possibly converted) value of the initializer expression. Standard
  //      conversions (Clause 4) will be used, if necessary, to convert the
  //      initializer expression to the cv-unqualified version of the 
  //      destination type; no user-defined conversions are considered.
  if (S.TryImplicitConversion(*this, Entity, Initializer,
                              /*SuppressUserConversions*/ true,
                              /*AllowExplicitConversions*/ false,
                              /*InOverloadResolution*/ false))
    SetFailed(InitializationSequence::FK_ConversionFailed);
  else
    setSequenceKind(StandardConversion);
}

InitializationSequence::~InitializationSequence() {
  for (llvm::SmallVectorImpl<Step>::iterator Step = Steps.begin(),
                                          StepEnd = Steps.end();
       Step != StepEnd; ++Step)
    Step->Destroy();
}

//===----------------------------------------------------------------------===//
// Perform initialization
//===----------------------------------------------------------------------===//
static Sema::AssignmentAction 
getAssignmentAction(const InitializedEntity &Entity) {
  switch(Entity.getKind()) {
  case InitializedEntity::EK_Variable:
  case InitializedEntity::EK_New:
    return Sema::AA_Initializing;

  case InitializedEntity::EK_Parameter:
    if (Entity.getDecl() && 
        isa<ObjCMethodDecl>(Entity.getDecl()->getDeclContext()))
      return Sema::AA_Sending;

    return Sema::AA_Passing;

  case InitializedEntity::EK_Result:
    return Sema::AA_Returning;

  case InitializedEntity::EK_Exception:
  case InitializedEntity::EK_Base:
    llvm_unreachable("No assignment action for C++-specific initialization");
    break;

  case InitializedEntity::EK_Temporary:
    // FIXME: Can we tell apart casting vs. converting?
    return Sema::AA_Casting;
    
  case InitializedEntity::EK_Member:
  case InitializedEntity::EK_ArrayElement:
  case InitializedEntity::EK_VectorElement:
  case InitializedEntity::EK_BlockElement:
    return Sema::AA_Initializing;
  }

  return Sema::AA_Converting;
}

/// \brief Whether we should binding a created object as a temporary when
/// initializing the given entity.
static bool shouldBindAsTemporary(const InitializedEntity &Entity) {
  switch (Entity.getKind()) {
  case InitializedEntity::EK_ArrayElement:
  case InitializedEntity::EK_Member:
  case InitializedEntity::EK_Result:
  case InitializedEntity::EK_New:
  case InitializedEntity::EK_Variable:
  case InitializedEntity::EK_Base:
  case InitializedEntity::EK_VectorElement:
  case InitializedEntity::EK_Exception:
  case InitializedEntity::EK_BlockElement:
    return false;
    
  case InitializedEntity::EK_Parameter:
  case InitializedEntity::EK_Temporary:
    return true;
  }
  
  llvm_unreachable("missed an InitializedEntity kind?");
}

/// \brief Whether the given entity, when initialized with an object
/// created for that initialization, requires destruction.
static bool shouldDestroyTemporary(const InitializedEntity &Entity) {
  switch (Entity.getKind()) {
    case InitializedEntity::EK_Member:
    case InitializedEntity::EK_Result:
    case InitializedEntity::EK_New:
    case InitializedEntity::EK_Base:
    case InitializedEntity::EK_VectorElement:
    case InitializedEntity::EK_BlockElement:
      return false;
      
    case InitializedEntity::EK_Variable:
    case InitializedEntity::EK_Parameter:
    case InitializedEntity::EK_Temporary:
    case InitializedEntity::EK_ArrayElement:
    case InitializedEntity::EK_Exception:
      return true;
  }
  
  llvm_unreachable("missed an InitializedEntity kind?");  
}

/// \brief Make a (potentially elidable) temporary copy of the object
/// provided by the given initializer by calling the appropriate copy
/// constructor.
///
/// \param S The Sema object used for type-checking.
///
/// \param T The type of the temporary object, which must either by
/// the type of the initializer expression or a superclass thereof.
///
/// \param Enter The entity being initialized.
///
/// \param CurInit The initializer expression.
///
/// \param IsExtraneousCopy Whether this is an "extraneous" copy that
/// is permitted in C++03 (but not C++0x) when binding a reference to
/// an rvalue.
///
/// \returns An expression that copies the initializer expression into
/// a temporary object, or an error expression if a copy could not be
/// created.
static Sema::OwningExprResult CopyObject(Sema &S,
                                         QualType T,
                                         const InitializedEntity &Entity,
                                         Sema::OwningExprResult CurInit,
                                         bool IsExtraneousCopy) {
  // Determine which class type we're copying to.
  Expr *CurInitExpr = (Expr *)CurInit.get();
  CXXRecordDecl *Class = 0; 
  if (const RecordType *Record = T->getAs<RecordType>())
    Class = cast<CXXRecordDecl>(Record->getDecl());
  if (!Class)
    return move(CurInit);

  // C++0x [class.copy]p34:
  //   When certain criteria are met, an implementation is allowed to
  //   omit the copy/move construction of a class object, even if the
  //   copy/move constructor and/or destructor for the object have
  //   side effects. [...]
  //     - when a temporary class object that has not been bound to a
  //       reference (12.2) would be copied/moved to a class object
  //       with the same cv-unqualified type, the copy/move operation
  //       can be omitted by constructing the temporary object
  //       directly into the target of the omitted copy/move
  // 
  // Note that the other three bullets are handled elsewhere. Copy
  // elision for return statements and throw expressions are handled as part
  // of constructor initialization, while copy elision for exception handlers 
  // is handled by the run-time.
  bool Elidable = CurInitExpr->isTemporaryObject() &&
     S.Context.hasSameUnqualifiedType(T, CurInitExpr->getType());
  SourceLocation Loc;
  switch (Entity.getKind()) {
  case InitializedEntity::EK_Result:
    Loc = Entity.getReturnLoc();
    break;
      
  case InitializedEntity::EK_Exception:
    Loc = Entity.getThrowLoc();
    break;
    
  case InitializedEntity::EK_Variable:
    Loc = Entity.getDecl()->getLocation();
    break;

  case InitializedEntity::EK_ArrayElement:
  case InitializedEntity::EK_Member:
  case InitializedEntity::EK_Parameter:
  case InitializedEntity::EK_Temporary:
  case InitializedEntity::EK_New:
  case InitializedEntity::EK_Base:
  case InitializedEntity::EK_VectorElement:
  case InitializedEntity::EK_BlockElement:
    Loc = CurInitExpr->getLocStart();
    break;
  }

  // Make sure that the type we are copying is complete.   
  if (S.RequireCompleteType(Loc, T, S.PDiag(diag::err_temp_copy_incomplete)))
    return move(CurInit);

  // Perform overload resolution using the class's copy constructors.
  DeclContext::lookup_iterator Con, ConEnd;
  OverloadCandidateSet CandidateSet(Loc);
  for (llvm::tie(Con, ConEnd) = S.LookupConstructors(Class);
       Con != ConEnd; ++Con) {
    // Only consider copy constructors.
    CXXConstructorDecl *Constructor = dyn_cast<CXXConstructorDecl>(*Con);
    if (!Constructor || Constructor->isInvalidDecl() ||
        !Constructor->isCopyConstructor() ||
        !Constructor->isConvertingConstructor(/*AllowExplicit=*/false))
      continue;

    DeclAccessPair FoundDecl
      = DeclAccessPair::make(Constructor, Constructor->getAccess());
    S.AddOverloadCandidate(Constructor, FoundDecl,
                           &CurInitExpr, 1, CandidateSet);
  }
  
  OverloadCandidateSet::iterator Best;
  switch (S.BestViableFunction(CandidateSet, Loc, Best)) {
  case OR_Success:
    break;
      
  case OR_No_Viable_Function:
    S.Diag(Loc, IsExtraneousCopy && !S.isSFINAEContext()
           ? diag::ext_rvalue_to_reference_temp_copy_no_viable
           : diag::err_temp_copy_no_viable)
      << (int)Entity.getKind() << CurInitExpr->getType()
      << CurInitExpr->getSourceRange();
    S.PrintOverloadCandidates(CandidateSet, Sema::OCD_AllCandidates,
                              &CurInitExpr, 1);
    if (!IsExtraneousCopy || S.isSFINAEContext())
      return S.ExprError();
    return move(CurInit);
      
  case OR_Ambiguous:
    S.Diag(Loc, diag::err_temp_copy_ambiguous)
      << (int)Entity.getKind() << CurInitExpr->getType()
      << CurInitExpr->getSourceRange();
    S.PrintOverloadCandidates(CandidateSet, Sema::OCD_ViableCandidates,
                              &CurInitExpr, 1);
    return S.ExprError();
    
  case OR_Deleted:
    S.Diag(Loc, diag::err_temp_copy_deleted)
      << (int)Entity.getKind() << CurInitExpr->getType()
      << CurInitExpr->getSourceRange();
    S.Diag(Best->Function->getLocation(), diag::note_unavailable_here)
      << Best->Function->isDeleted();
    return S.ExprError();
  }

  CXXConstructorDecl *Constructor = cast<CXXConstructorDecl>(Best->Function);
  ASTOwningVector<&ActionBase::DeleteExpr> ConstructorArgs(S);
  CurInit.release(); // Ownership transferred into MultiExprArg, below.

  S.CheckConstructorAccess(Loc, Constructor, Entity,
                           Best->FoundDecl.getAccess(), IsExtraneousCopy);

  if (IsExtraneousCopy) {
    // If this is a totally extraneous copy for C++03 reference
    // binding purposes, just return the original initialization
    // expression. We don't generate an (elided) copy operation here
    // because doing so would require us to pass down a flag to avoid
    // infinite recursion, where each step adds another extraneous,
    // elidable copy.

    // Instantiate the default arguments of any extra parameters in
    // the selected copy constructor, as if we were going to create a
    // proper call to the copy constructor.
    for (unsigned I = 1, N = Constructor->getNumParams(); I != N; ++I) {
      ParmVarDecl *Parm = Constructor->getParamDecl(I);
      if (S.RequireCompleteType(Loc, Parm->getType(),
                                S.PDiag(diag::err_call_incomplete_argument)))
        break;

      // Build the default argument expression; we don't actually care
      // if this succeeds or not, because this routine will complain
      // if there was a problem.
      S.BuildCXXDefaultArgExpr(Loc, Constructor, Parm);
    }

    return S.Owned(CurInitExpr);
  }
  
  // Determine the arguments required to actually perform the
  // constructor call (we might have derived-to-base conversions, or
  // the copy constructor may have default arguments).
  if (S.CompleteConstructorCall(Constructor,
                                Sema::MultiExprArg(S, 
                                                   (void **)&CurInitExpr,
                                                   1),
                                Loc, ConstructorArgs))
    return S.ExprError();

  // Actually perform the constructor call.
  CurInit = S.BuildCXXConstructExpr(Loc, T, Constructor, Elidable,
                                    move_arg(ConstructorArgs));
  
  // If we're supposed to bind temporaries, do so.
  if (!CurInit.isInvalid() && shouldBindAsTemporary(Entity))
    CurInit = S.MaybeBindToTemporary(CurInit.takeAs<Expr>());
  return move(CurInit);
}

void InitializationSequence::PrintInitLocationNote(Sema &S,
                                              const InitializedEntity &Entity) {
  if (Entity.getKind() == InitializedEntity::EK_Parameter && Entity.getDecl()) {
    if (Entity.getDecl()->getLocation().isInvalid())
      return;

    if (Entity.getDecl()->getDeclName())
      S.Diag(Entity.getDecl()->getLocation(), diag::note_parameter_named_here)
        << Entity.getDecl()->getDeclName();
    else
      S.Diag(Entity.getDecl()->getLocation(), diag::note_parameter_here);
  }
}

Action::OwningExprResult 
InitializationSequence::Perform(Sema &S,
                                const InitializedEntity &Entity,
                                const InitializationKind &Kind,
                                Action::MultiExprArg Args,
                                QualType *ResultType) {
  if (SequenceKind == FailedSequence) {
    unsigned NumArgs = Args.size();
    Diagnose(S, Entity, Kind, (Expr **)Args.release(), NumArgs);
    return S.ExprError();
  }
  
  if (SequenceKind == DependentSequence) {
    // If the declaration is a non-dependent, incomplete array type
    // that has an initializer, then its type will be completed once
    // the initializer is instantiated.
    if (ResultType && !Entity.getType()->isDependentType() &&
        Args.size() == 1) {
      QualType DeclType = Entity.getType();
      if (const IncompleteArrayType *ArrayT
                           = S.Context.getAsIncompleteArrayType(DeclType)) {
        // FIXME: We don't currently have the ability to accurately
        // compute the length of an initializer list without
        // performing full type-checking of the initializer list
        // (since we have to determine where braces are implicitly
        // introduced and such).  So, we fall back to making the array
        // type a dependently-sized array type with no specified
        // bound.
        if (isa<InitListExpr>((Expr *)Args.get()[0])) {
          SourceRange Brackets;

          // Scavange the location of the brackets from the entity, if we can.
          if (DeclaratorDecl *DD = Entity.getDecl()) {
            if (TypeSourceInfo *TInfo = DD->getTypeSourceInfo()) {
              TypeLoc TL = TInfo->getTypeLoc();
              if (IncompleteArrayTypeLoc *ArrayLoc
                                      = dyn_cast<IncompleteArrayTypeLoc>(&TL))
              Brackets = ArrayLoc->getBracketsRange();
            }
          }

          *ResultType
            = S.Context.getDependentSizedArrayType(ArrayT->getElementType(),
                                                   /*NumElts=*/0,
                                                   ArrayT->getSizeModifier(),
                                       ArrayT->getIndexTypeCVRQualifiers(),
                                                   Brackets);
        }

      }
    }

    if (Kind.getKind() == InitializationKind::IK_Copy || Kind.isExplicitCast())
      return Sema::OwningExprResult(S, Args.release()[0]);

    if (Args.size() == 0)
      return S.Owned((Expr *)0);

    unsigned NumArgs = Args.size();
    return S.Owned(new (S.Context) ParenListExpr(S.Context,
                                                 SourceLocation(),
                                                 (Expr **)Args.release(), 
                                                 NumArgs,
                                                 SourceLocation()));
  }

  if (SequenceKind == NoInitialization)
    return S.Owned((Expr *)0);
  
  QualType DestType = Entity.getType().getNonReferenceType();
  // FIXME: Ugly hack around the fact that Entity.getType() is not
  // the same as Entity.getDecl()->getType() in cases involving type merging,
  //  and we want latter when it makes sense.
  if (ResultType)
    *ResultType = Entity.getDecl() ? Entity.getDecl()->getType() :
                                     Entity.getType();

  Sema::OwningExprResult CurInit = S.Owned((Expr *)0);
  
  assert(!Steps.empty() && "Cannot have an empty initialization sequence");
  
  // For initialization steps that start with a single initializer, 
  // grab the only argument out the Args and place it into the "current"
  // initializer.
  switch (Steps.front().Kind) {
  case SK_ResolveAddressOfOverloadedFunction:
  case SK_CastDerivedToBaseRValue:
  case SK_CastDerivedToBaseXValue:
  case SK_CastDerivedToBaseLValue:
  case SK_BindReference:
  case SK_BindReferenceToTemporary:
  case SK_ExtraneousCopyToTemporary:
  case SK_UserConversion:
  case SK_QualificationConversionLValue:
  case SK_QualificationConversionXValue:
  case SK_QualificationConversionRValue:
  case SK_ConversionSequence:
  case SK_ListInitialization:
  case SK_CAssignment:
  case SK_StringInit:
  case SK_ObjCObjectConversion:
    assert(Args.size() == 1);
    CurInit = Sema::OwningExprResult(S, ((Expr **)(Args.get()))[0]->Retain());
    if (CurInit.isInvalid())
      return S.ExprError();
    break;
    
  case SK_ConstructorInitialization:
  case SK_ZeroInitialization:
    break;
  }
    
  // Walk through the computed steps for the initialization sequence, 
  // performing the specified conversions along the way.
  bool ConstructorInitRequiresZeroInit = false;
  for (step_iterator Step = step_begin(), StepEnd = step_end();
       Step != StepEnd; ++Step) {
    if (CurInit.isInvalid())
      return S.ExprError();
    
    Expr *CurInitExpr = (Expr *)CurInit.get();
    QualType SourceType = CurInitExpr? CurInitExpr->getType() : QualType();
    
    switch (Step->Kind) {
    case SK_ResolveAddressOfOverloadedFunction:
      // Overload resolution determined which function invoke; update the 
      // initializer to reflect that choice.
      S.CheckAddressOfMemberAccess(CurInitExpr, Step->Function.FoundDecl);
      S.DiagnoseUseOfDecl(Step->Function.FoundDecl, Kind.getLocation());
      CurInit = S.FixOverloadedFunctionReference(move(CurInit),
                                                 Step->Function.FoundDecl,
                                                 Step->Function.Function);
      break;
        
    case SK_CastDerivedToBaseRValue:
    case SK_CastDerivedToBaseXValue:
    case SK_CastDerivedToBaseLValue: {
      // We have a derived-to-base cast that produces either an rvalue or an
      // lvalue. Perform that cast.
      
      CXXCastPath BasePath;

      // Casts to inaccessible base classes are allowed with C-style casts.
      bool IgnoreBaseAccess = Kind.isCStyleOrFunctionalCast();
      if (S.CheckDerivedToBaseConversion(SourceType, Step->Type,
                                         CurInitExpr->getLocStart(),
                                         CurInitExpr->getSourceRange(), 
                                         &BasePath, IgnoreBaseAccess))
        return S.ExprError();
        
      if (S.BasePathInvolvesVirtualBase(BasePath)) {
        QualType T = SourceType;
        if (const PointerType *Pointer = T->getAs<PointerType>())
          T = Pointer->getPointeeType();
        if (const RecordType *RecordTy = T->getAs<RecordType>())
          S.MarkVTableUsed(CurInitExpr->getLocStart(), 
                           cast<CXXRecordDecl>(RecordTy->getDecl()));
      }

      ImplicitCastExpr::ResultCategory Category =
          Step->Kind == SK_CastDerivedToBaseLValue ?
              ImplicitCastExpr::LValue :
              (Step->Kind == SK_CastDerivedToBaseXValue ?
                   ImplicitCastExpr::XValue :
                   ImplicitCastExpr::RValue);
      CurInit = S.Owned(ImplicitCastExpr::Create(S.Context,
                                                 Step->Type,
                                                 CastExpr::CK_DerivedToBase,
                                                 (Expr*)CurInit.release(),
                                                 &BasePath, Category));
      break;
    }
        
    case SK_BindReference:
      if (FieldDecl *BitField = CurInitExpr->getBitField()) {
        // References cannot bind to bit fields (C++ [dcl.init.ref]p5).
        S.Diag(Kind.getLocation(), diag::err_reference_bind_to_bitfield)
          << Entity.getType().isVolatileQualified()
          << BitField->getDeclName()
          << CurInitExpr->getSourceRange();
        S.Diag(BitField->getLocation(), diag::note_bitfield_decl);
        return S.ExprError();
      }

      if (CurInitExpr->refersToVectorElement()) {
        // References cannot bind to vector elements.
        S.Diag(Kind.getLocation(), diag::err_reference_bind_to_vector_element)
          << Entity.getType().isVolatileQualified()
          << CurInitExpr->getSourceRange();
        PrintInitLocationNote(S, Entity);
        return S.ExprError();
      }
        
      // Reference binding does not have any corresponding ASTs.

      // Check exception specifications
      if (S.CheckExceptionSpecCompatibility(CurInitExpr, DestType))
        return S.ExprError();

      break;

    case SK_BindReferenceToTemporary:
      // Reference binding does not have any corresponding ASTs.

      // Check exception specifications
      if (S.CheckExceptionSpecCompatibility(CurInitExpr, DestType))
        return S.ExprError();

      break;
        
    case SK_ExtraneousCopyToTemporary:
      CurInit = CopyObject(S, Step->Type, Entity, move(CurInit), 
                           /*IsExtraneousCopy=*/true);
      break;

    case SK_UserConversion: {
      // We have a user-defined conversion that invokes either a constructor
      // or a conversion function.
      CastExpr::CastKind CastKind = CastExpr::CK_Unknown;
      bool IsCopy = false;
      FunctionDecl *Fn = Step->Function.Function;
      DeclAccessPair FoundFn = Step->Function.FoundDecl;
      bool CreatedObject = false;
      bool IsLvalue = false;
      if (CXXConstructorDecl *Constructor = dyn_cast<CXXConstructorDecl>(Fn)) {
        // Build a call to the selected constructor.
        ASTOwningVector<&ActionBase::DeleteExpr> ConstructorArgs(S);
        SourceLocation Loc = CurInitExpr->getLocStart();
        CurInit.release(); // Ownership transferred into MultiExprArg, below.

        // Determine the arguments required to actually perform the constructor
        // call.
        if (S.CompleteConstructorCall(Constructor,
                                      Sema::MultiExprArg(S, 
                                                         (void **)&CurInitExpr,
                                                         1),
                                      Loc, ConstructorArgs))
          return S.ExprError();
        
        // Build the an expression that constructs a temporary.
        CurInit = S.BuildCXXConstructExpr(Loc, Step->Type, Constructor, 
                                          move_arg(ConstructorArgs));
        if (CurInit.isInvalid())
          return S.ExprError();

        S.CheckConstructorAccess(Kind.getLocation(), Constructor, Entity,
                                 FoundFn.getAccess());
        S.DiagnoseUseOfDecl(FoundFn, Kind.getLocation());
        
        CastKind = CastExpr::CK_ConstructorConversion;
        QualType Class = S.Context.getTypeDeclType(Constructor->getParent());
        if (S.Context.hasSameUnqualifiedType(SourceType, Class) ||
            S.IsDerivedFrom(SourceType, Class))
          IsCopy = true;
        
        CreatedObject = true;
      } else {
        // Build a call to the conversion function.
        CXXConversionDecl *Conversion = cast<CXXConversionDecl>(Fn);
        IsLvalue = Conversion->getResultType()->isLValueReferenceType();
        S.CheckMemberOperatorAccess(Kind.getLocation(), CurInitExpr, 0,
                                    FoundFn);
        S.DiagnoseUseOfDecl(FoundFn, Kind.getLocation());
        
        // FIXME: Should we move this initialization into a separate 
        // derived-to-base conversion? I believe the answer is "no", because
        // we don't want to turn off access control here for c-style casts.
        if (S.PerformObjectArgumentInitialization(CurInitExpr, /*Qualifier=*/0,
                                                  FoundFn, Conversion))
          return S.ExprError();

        // Do a little dance to make sure that CurInit has the proper
        // pointer.
        CurInit.release();
        
        // Build the actual call to the conversion function.
        CurInit = S.Owned(S.BuildCXXMemberCallExpr(CurInitExpr, FoundFn,
                                                   Conversion));
        if (CurInit.isInvalid() || !CurInit.get())
          return S.ExprError();
        
        CastKind = CastExpr::CK_UserDefinedConversion;
        
        CreatedObject = Conversion->getResultType()->isRecordType();
      }
      
      bool RequiresCopy = !IsCopy && 
        getKind() != InitializationSequence::ReferenceBinding;
      if (RequiresCopy || shouldBindAsTemporary(Entity))
        CurInit = S.MaybeBindToTemporary(CurInit.takeAs<Expr>());
      else if (CreatedObject && shouldDestroyTemporary(Entity)) {
        CurInitExpr = static_cast<Expr *>(CurInit.get());
        QualType T = CurInitExpr->getType();
        if (const RecordType *Record = T->getAs<RecordType>()) {
          CXXDestructorDecl *Destructor 
            = S.LookupDestructor(cast<CXXRecordDecl>(Record->getDecl()));
          S.CheckDestructorAccess(CurInitExpr->getLocStart(), Destructor, 
                                  S.PDiag(diag::err_access_dtor_temp) << T);
          S.MarkDeclarationReferenced(CurInitExpr->getLocStart(), Destructor);
        }
      }
      
      CurInitExpr = CurInit.takeAs<Expr>();
      // FIXME: xvalues
      CurInit = S.Owned(ImplicitCastExpr::Create(S.Context,
                                                 CurInitExpr->getType(),
                                                 CastKind, CurInitExpr, 0,
               IsLvalue ? ImplicitCastExpr::LValue : ImplicitCastExpr::RValue));
      
      if (RequiresCopy)
        CurInit = CopyObject(S, Entity.getType().getNonReferenceType(), Entity,
                             move(CurInit), /*IsExtraneousCopy=*/false);

      break;
    }

    case SK_QualificationConversionLValue:
    case SK_QualificationConversionXValue:
    case SK_QualificationConversionRValue: {
      // Perform a qualification conversion; these can never go wrong.
      ImplicitCastExpr::ResultCategory Category =
          Step->Kind == SK_QualificationConversionLValue ?
              ImplicitCastExpr::LValue :
              (Step->Kind == SK_QualificationConversionXValue ?
                   ImplicitCastExpr::XValue :
                   ImplicitCastExpr::RValue);
      S.ImpCastExprToType(CurInitExpr, Step->Type, CastExpr::CK_NoOp, Category);
      CurInit.release();
      CurInit = S.Owned(CurInitExpr);
      break;
    }

    case SK_ConversionSequence: {
      bool IgnoreBaseAccess = Kind.isCStyleOrFunctionalCast();

      if (S.PerformImplicitConversion(CurInitExpr, Step->Type, *Step->ICS,
                                      Sema::AA_Converting, IgnoreBaseAccess))
        return S.ExprError();
        
      CurInit.release();
      CurInit = S.Owned(CurInitExpr);
      break;
    }
        
    case SK_ListInitialization: {
      InitListExpr *InitList = cast<InitListExpr>(CurInitExpr);
      QualType Ty = Step->Type;
      if (S.CheckInitList(Entity, InitList, ResultType? *ResultType : Ty))
        return S.ExprError();

      CurInit.release();
      CurInit = S.Owned(InitList);
      break;
    }

    case SK_ConstructorInitialization: {
      unsigned NumArgs = Args.size();
      CXXConstructorDecl *Constructor
        = cast<CXXConstructorDecl>(Step->Function.Function);
      
      // Build a call to the selected constructor.
      ASTOwningVector<&ActionBase::DeleteExpr> ConstructorArgs(S);
      SourceLocation Loc = (Kind.isCopyInit() && Kind.getEqualLoc().isValid())
                             ? Kind.getEqualLoc()
                             : Kind.getLocation();
          
      // Determine the arguments required to actually perform the constructor
      // call.
      if (S.CompleteConstructorCall(Constructor, move(Args), 
                                    Loc, ConstructorArgs))
        return S.ExprError();
          
      // Build the expression that constructs a temporary.
      if (Entity.getKind() == InitializedEntity::EK_Temporary &&
          NumArgs != 1 && // FIXME: Hack to work around cast weirdness
          (Kind.getKind() == InitializationKind::IK_Direct ||
           Kind.getKind() == InitializationKind::IK_Value)) {
        // An explicitly-constructed temporary, e.g., X(1, 2).
        unsigned NumExprs = ConstructorArgs.size();
        Expr **Exprs = (Expr **)ConstructorArgs.take();
        S.MarkDeclarationReferenced(Loc, Constructor);
        CurInit = S.Owned(new (S.Context) CXXTemporaryObjectExpr(S.Context,
                                                                 Constructor,
                                                              Entity.getType(),
                                                                 Loc,
                                                                 Exprs, 
                                                                 NumExprs,
                                                Kind.getParenRange().getEnd(),
                                             ConstructorInitRequiresZeroInit));
      } else {
        CXXConstructExpr::ConstructionKind ConstructKind =
          CXXConstructExpr::CK_Complete;
        
        if (Entity.getKind() == InitializedEntity::EK_Base) {
          ConstructKind = Entity.getBaseSpecifier()->isVirtual() ?
            CXXConstructExpr::CK_VirtualBase : 
            CXXConstructExpr::CK_NonVirtualBase;
        }    
        
        // If the entity allows NRVO, mark the construction as elidable
        // unconditionally.
        if (Entity.allowsNRVO())
          CurInit = S.BuildCXXConstructExpr(Loc, Entity.getType(),
                                            Constructor, /*Elidable=*/true,
                                            move_arg(ConstructorArgs),
                                            ConstructorInitRequiresZeroInit,
                                            ConstructKind);
        else
          CurInit = S.BuildCXXConstructExpr(Loc, Entity.getType(),
                                            Constructor, 
                                            move_arg(ConstructorArgs),
                                            ConstructorInitRequiresZeroInit,
                                            ConstructKind);
      }
      if (CurInit.isInvalid())
        return S.ExprError();

      // Only check access if all of that succeeded.
      S.CheckConstructorAccess(Loc, Constructor, Entity,
                               Step->Function.FoundDecl.getAccess());
      S.DiagnoseUseOfDecl(Step->Function.FoundDecl, Loc);
      
      if (shouldBindAsTemporary(Entity))
        CurInit = S.MaybeBindToTemporary(CurInit.takeAs<Expr>());
      
      break;
    }
        
    case SK_ZeroInitialization: {
      step_iterator NextStep = Step;
      ++NextStep;
      if (NextStep != StepEnd && 
          NextStep->Kind == SK_ConstructorInitialization) {
        // The need for zero-initialization is recorded directly into
        // the call to the object's constructor within the next step.
        ConstructorInitRequiresZeroInit = true;
      } else if (Kind.getKind() == InitializationKind::IK_Value &&
                 S.getLangOptions().CPlusPlus &&
                 !Kind.isImplicitValueInit()) {
        CurInit = S.Owned(new (S.Context) CXXScalarValueInitExpr(Step->Type,
                                                   Kind.getRange().getBegin(),
                                                    Kind.getRange().getEnd()));
      } else {
        CurInit = S.Owned(new (S.Context) ImplicitValueInitExpr(Step->Type));
      }
      break;
    }

    case SK_CAssignment: {
      QualType SourceType = CurInitExpr->getType();
      Sema::AssignConvertType ConvTy =
        S.CheckSingleAssignmentConstraints(Step->Type, CurInitExpr);

      // If this is a call, allow conversion to a transparent union.
      if (ConvTy != Sema::Compatible &&
          Entity.getKind() == InitializedEntity::EK_Parameter &&
          S.CheckTransparentUnionArgumentConstraints(Step->Type, CurInitExpr)
            == Sema::Compatible)
        ConvTy = Sema::Compatible;

      bool Complained;
      if (S.DiagnoseAssignmentResult(ConvTy, Kind.getLocation(),
                                     Step->Type, SourceType,
                                     CurInitExpr, 
                                     getAssignmentAction(Entity),
                                     &Complained)) {
        PrintInitLocationNote(S, Entity);
        return S.ExprError();
      } else if (Complained)
        PrintInitLocationNote(S, Entity);

      CurInit.release();
      CurInit = S.Owned(CurInitExpr);
      break;
    }

    case SK_StringInit: {
      QualType Ty = Step->Type;
      CheckStringInit(CurInitExpr, ResultType ? *ResultType : Ty, S);
      break;
    }

    case SK_ObjCObjectConversion:
      S.ImpCastExprToType(CurInitExpr, Step->Type, 
                          CastExpr::CK_ObjCObjectLValueCast,
                          S.CastCategory(CurInitExpr));
      CurInit.release();
      CurInit = S.Owned(CurInitExpr);
      break;
    }
  }
  
  return move(CurInit);
}

//===----------------------------------------------------------------------===//
// Diagnose initialization failures
//===----------------------------------------------------------------------===//
bool InitializationSequence::Diagnose(Sema &S, 
                                      const InitializedEntity &Entity,
                                      const InitializationKind &Kind,
                                      Expr **Args, unsigned NumArgs) {
  if (SequenceKind != FailedSequence)
    return false;
  
  QualType DestType = Entity.getType();
  switch (Failure) {
  case FK_TooManyInitsForReference:
    // FIXME: Customize for the initialized entity?
    if (NumArgs == 0)
      S.Diag(Kind.getLocation(), diag::err_reference_without_init)
        << DestType.getNonReferenceType();
    else  // FIXME: diagnostic below could be better!
      S.Diag(Kind.getLocation(), diag::err_reference_has_multiple_inits)
        << SourceRange(Args[0]->getLocStart(), Args[NumArgs - 1]->getLocEnd());
    break;
    
  case FK_ArrayNeedsInitList:
  case FK_ArrayNeedsInitListOrStringLiteral:
    S.Diag(Kind.getLocation(), diag::err_array_init_not_init_list)
      << (Failure == FK_ArrayNeedsInitListOrStringLiteral);
    break;
      
  case FK_AddressOfOverloadFailed: {
    DeclAccessPair Found;
    S.ResolveAddressOfOverloadedFunction(Args[0], 
                                         DestType.getNonReferenceType(),
                                         true,
                                         Found);
    break;
  }
      
  case FK_ReferenceInitOverloadFailed:
  case FK_UserConversionOverloadFailed:
    switch (FailedOverloadResult) {
    case OR_Ambiguous:
      if (Failure == FK_UserConversionOverloadFailed)
        S.Diag(Kind.getLocation(), diag::err_typecheck_ambiguous_condition)
          << Args[0]->getType() << DestType
          << Args[0]->getSourceRange();
      else
        S.Diag(Kind.getLocation(), diag::err_ref_init_ambiguous)
          << DestType << Args[0]->getType()
          << Args[0]->getSourceRange();

      S.PrintOverloadCandidates(FailedCandidateSet, Sema::OCD_ViableCandidates,
                                Args, NumArgs);
      break;
        
    case OR_No_Viable_Function:
      S.Diag(Kind.getLocation(), diag::err_typecheck_nonviable_condition)
        << Args[0]->getType() << DestType.getNonReferenceType()
        << Args[0]->getSourceRange();
      S.PrintOverloadCandidates(FailedCandidateSet, Sema::OCD_AllCandidates,
                                Args, NumArgs);
      break;
        
    case OR_Deleted: {
      S.Diag(Kind.getLocation(), diag::err_typecheck_deleted_function)
        << Args[0]->getType() << DestType.getNonReferenceType()
        << Args[0]->getSourceRange();
      OverloadCandidateSet::iterator Best;
      OverloadingResult Ovl = S.BestViableFunction(FailedCandidateSet,
                                                   Kind.getLocation(),
                                                   Best);
      if (Ovl == OR_Deleted) {
        S.Diag(Best->Function->getLocation(), diag::note_unavailable_here)
          << Best->Function->isDeleted();
      } else {
        llvm_unreachable("Inconsistent overload resolution?");
      }
      break;
    }
        
    case OR_Success:
      llvm_unreachable("Conversion did not fail!");
      break;
    }
    break;
      
  case FK_NonConstLValueReferenceBindingToTemporary:
  case FK_NonConstLValueReferenceBindingToUnrelated:
    S.Diag(Kind.getLocation(), 
           Failure == FK_NonConstLValueReferenceBindingToTemporary
             ? diag::err_lvalue_reference_bind_to_temporary
             : diag::err_lvalue_reference_bind_to_unrelated)
      << DestType.getNonReferenceType().isVolatileQualified()
      << DestType.getNonReferenceType()
      << Args[0]->getType()
      << Args[0]->getSourceRange();
    break;
      
  case FK_RValueReferenceBindingToLValue:
    S.Diag(Kind.getLocation(), diag::err_lvalue_to_rvalue_ref)
      << Args[0]->getSourceRange();
    break;
      
  case FK_ReferenceInitDropsQualifiers:
    S.Diag(Kind.getLocation(), diag::err_reference_bind_drops_quals)
      << DestType.getNonReferenceType()
      << Args[0]->getType()
      << Args[0]->getSourceRange();
    break;
      
  case FK_ReferenceInitFailed:
    S.Diag(Kind.getLocation(), diag::err_reference_bind_failed)
      << DestType.getNonReferenceType()
      << (Args[0]->isLvalue(S.Context) == Expr::LV_Valid)
      << Args[0]->getType()
      << Args[0]->getSourceRange();
    break;
      
  case FK_ConversionFailed:
    S.Diag(Kind.getLocation(), diag::err_init_conversion_failed)
      << (int)Entity.getKind()
      << DestType
      << (Args[0]->isLvalue(S.Context) == Expr::LV_Valid)
      << Args[0]->getType()
      << Args[0]->getSourceRange();
    break;

  case FK_TooManyInitsForScalar: {
    SourceRange R;

    if (InitListExpr *InitList = dyn_cast<InitListExpr>(Args[0]))
      R = SourceRange(InitList->getInit(1)->getLocStart(),
                      InitList->getLocEnd());
    else
      R = SourceRange(Args[0]->getLocStart(), Args[NumArgs - 1]->getLocEnd());

    S.Diag(Kind.getLocation(), diag::err_excess_initializers)
      << /*scalar=*/2 << R;
    break;
  }

  case FK_ReferenceBindingToInitList:
    S.Diag(Kind.getLocation(), diag::err_reference_bind_init_list)
      << DestType.getNonReferenceType() << Args[0]->getSourceRange();
    break;

  case FK_InitListBadDestinationType:
    S.Diag(Kind.getLocation(), diag::err_init_list_bad_dest_type)
      << (DestType->isRecordType()) << DestType << Args[0]->getSourceRange();
    break;
      
  case FK_ConstructorOverloadFailed: {
    SourceRange ArgsRange;
    if (NumArgs)
      ArgsRange = SourceRange(Args[0]->getLocStart(), 
                              Args[NumArgs - 1]->getLocEnd());
    
    // FIXME: Using "DestType" for the entity we're printing is probably
    // bad.
    switch (FailedOverloadResult) {
      case OR_Ambiguous:
        S.Diag(Kind.getLocation(), diag::err_ovl_ambiguous_init)
          << DestType << ArgsRange;
        S.PrintOverloadCandidates(FailedCandidateSet,
                                  Sema::OCD_ViableCandidates, Args, NumArgs);
        break;
        
      case OR_No_Viable_Function:
        if (Kind.getKind() == InitializationKind::IK_Default &&
            (Entity.getKind() == InitializedEntity::EK_Base ||
             Entity.getKind() == InitializedEntity::EK_Member) &&
            isa<CXXConstructorDecl>(S.CurContext)) {
          // This is implicit default initialization of a member or
          // base within a constructor. If no viable function was
          // found, notify the user that she needs to explicitly
          // initialize this base/member.
          CXXConstructorDecl *Constructor
            = cast<CXXConstructorDecl>(S.CurContext);
          if (Entity.getKind() == InitializedEntity::EK_Base) {
            S.Diag(Kind.getLocation(), diag::err_missing_default_ctor)
              << Constructor->isImplicit()
              << S.Context.getTypeDeclType(Constructor->getParent())
              << /*base=*/0
              << Entity.getType();

            RecordDecl *BaseDecl
              = Entity.getBaseSpecifier()->getType()->getAs<RecordType>()
                                                                  ->getDecl();
            S.Diag(BaseDecl->getLocation(), diag::note_previous_decl)
              << S.Context.getTagDeclType(BaseDecl);
          } else {
            S.Diag(Kind.getLocation(), diag::err_missing_default_ctor)
              << Constructor->isImplicit()
              << S.Context.getTypeDeclType(Constructor->getParent())
              << /*member=*/1
              << Entity.getName();
            S.Diag(Entity.getDecl()->getLocation(), diag::note_field_decl);

            if (const RecordType *Record
                                 = Entity.getType()->getAs<RecordType>())
              S.Diag(Record->getDecl()->getLocation(), 
                     diag::note_previous_decl)
                << S.Context.getTagDeclType(Record->getDecl());
          }
          break;
        }

        S.Diag(Kind.getLocation(), diag::err_ovl_no_viable_function_in_init)
          << DestType << ArgsRange;
        S.PrintOverloadCandidates(FailedCandidateSet, Sema::OCD_AllCandidates,
                                  Args, NumArgs);
        break;
        
      case OR_Deleted: {
        S.Diag(Kind.getLocation(), diag::err_ovl_deleted_init)
          << true << DestType << ArgsRange;
        OverloadCandidateSet::iterator Best;
        OverloadingResult Ovl = S.BestViableFunction(FailedCandidateSet,
                                                     Kind.getLocation(),
                                                     Best);
        if (Ovl == OR_Deleted) {
          S.Diag(Best->Function->getLocation(), diag::note_unavailable_here)
            << Best->Function->isDeleted();
        } else {
          llvm_unreachable("Inconsistent overload resolution?");
        }
        break;
      }
        
      case OR_Success:
        llvm_unreachable("Conversion did not fail!");
        break;
    }
    break;
  }
      
  case FK_DefaultInitOfConst:
    if (Entity.getKind() == InitializedEntity::EK_Member &&
        isa<CXXConstructorDecl>(S.CurContext)) {
      // This is implicit default-initialization of a const member in
      // a constructor. Complain that it needs to be explicitly
      // initialized.
      CXXConstructorDecl *Constructor = cast<CXXConstructorDecl>(S.CurContext);
      S.Diag(Kind.getLocation(), diag::err_uninitialized_member_in_ctor)
        << Constructor->isImplicit()
        << S.Context.getTypeDeclType(Constructor->getParent())
        << /*const=*/1
        << Entity.getName();
      S.Diag(Entity.getDecl()->getLocation(), diag::note_previous_decl)
        << Entity.getName();
    } else {
      S.Diag(Kind.getLocation(), diag::err_default_init_const)
        << DestType << (bool)DestType->getAs<RecordType>();
    }
    break;
      
    case FK_Incomplete:
      S.RequireCompleteType(Kind.getLocation(), DestType, 
                            diag::err_init_incomplete_type);
      break;      
  }
  
  PrintInitLocationNote(S, Entity);
  return true;
}

void InitializationSequence::dump(llvm::raw_ostream &OS) const {
  switch (SequenceKind) {
  case FailedSequence: {
    OS << "Failed sequence: ";
    switch (Failure) {
    case FK_TooManyInitsForReference:
      OS << "too many initializers for reference";
      break;
      
    case FK_ArrayNeedsInitList:
      OS << "array requires initializer list";
      break;
      
    case FK_ArrayNeedsInitListOrStringLiteral:
      OS << "array requires initializer list or string literal";
      break;
      
    case FK_AddressOfOverloadFailed:
      OS << "address of overloaded function failed";
      break;
      
    case FK_ReferenceInitOverloadFailed:
      OS << "overload resolution for reference initialization failed";
      break;
      
    case FK_NonConstLValueReferenceBindingToTemporary:
      OS << "non-const lvalue reference bound to temporary";
      break;
      
    case FK_NonConstLValueReferenceBindingToUnrelated:
      OS << "non-const lvalue reference bound to unrelated type";
      break;
      
    case FK_RValueReferenceBindingToLValue:
      OS << "rvalue reference bound to an lvalue";
      break;
      
    case FK_ReferenceInitDropsQualifiers:
      OS << "reference initialization drops qualifiers";
      break;
      
    case FK_ReferenceInitFailed:
      OS << "reference initialization failed";
      break;
      
    case FK_ConversionFailed:
      OS << "conversion failed";
      break;
      
    case FK_TooManyInitsForScalar:
      OS << "too many initializers for scalar";
      break;
      
    case FK_ReferenceBindingToInitList:
      OS << "referencing binding to initializer list";
      break;
      
    case FK_InitListBadDestinationType:
      OS << "initializer list for non-aggregate, non-scalar type";
      break;
      
    case FK_UserConversionOverloadFailed:
      OS << "overloading failed for user-defined conversion";
      break;
      
    case FK_ConstructorOverloadFailed:
      OS << "constructor overloading failed";
      break;
      
    case FK_DefaultInitOfConst:
      OS << "default initialization of a const variable";
      break;
        
    case FK_Incomplete:
      OS << "initialization of incomplete type";
      break;
    }   
    OS << '\n';
    return;
  }
      
  case DependentSequence:
    OS << "Dependent sequence: ";
    return;
      
  case UserDefinedConversion:
    OS << "User-defined conversion sequence: ";
    break;
      
  case ConstructorInitialization:
    OS << "Constructor initialization sequence: ";
    break;
      
  case ReferenceBinding:
    OS << "Reference binding: ";
    break;
      
  case ListInitialization:
    OS << "List initialization: ";
    break;

  case ZeroInitialization:
    OS << "Zero initialization\n";
    return;
      
  case NoInitialization:
    OS << "No initialization\n";
    return;
      
  case StandardConversion:
    OS << "Standard conversion: ";
    break;
      
  case CAssignment:
    OS << "C assignment: ";
    break;
      
  case StringInit:
    OS << "String initialization: ";
    break;
  }
  
  for (step_iterator S = step_begin(), SEnd = step_end(); S != SEnd; ++S) {
    if (S != step_begin()) {
      OS << " -> ";
    }
    
    switch (S->Kind) {
    case SK_ResolveAddressOfOverloadedFunction:
      OS << "resolve address of overloaded function";
      break;
      
    case SK_CastDerivedToBaseRValue:
      OS << "derived-to-base case (rvalue" << S->Type.getAsString() << ")";
      break;
      
    case SK_CastDerivedToBaseXValue:
      OS << "derived-to-base case (xvalue" << S->Type.getAsString() << ")";
      break;
      
    case SK_CastDerivedToBaseLValue:
      OS << "derived-to-base case (lvalue" << S->Type.getAsString() << ")";
      break;
      
    case SK_BindReference:
      OS << "bind reference to lvalue";
      break;
      
    case SK_BindReferenceToTemporary:
      OS << "bind reference to a temporary";
      break;
      
    case SK_ExtraneousCopyToTemporary:
      OS << "extraneous C++03 copy to temporary";
      break;

    case SK_UserConversion:
      OS << "user-defined conversion via " << S->Function.Function;
      break;

    case SK_QualificationConversionRValue:
      OS << "qualification conversion (rvalue)";

    case SK_QualificationConversionXValue:
      OS << "qualification conversion (xvalue)";

    case SK_QualificationConversionLValue:
      OS << "qualification conversion (lvalue)";
      break;
      
    case SK_ConversionSequence:
      OS << "implicit conversion sequence (";
      S->ICS->DebugPrint(); // FIXME: use OS
      OS << ")";
      break;
      
    case SK_ListInitialization:
      OS << "list initialization";
      break;
      
    case SK_ConstructorInitialization:
      OS << "constructor initialization";
      break;
      
    case SK_ZeroInitialization:
      OS << "zero initialization";
      break;
      
    case SK_CAssignment:
      OS << "C assignment";
      break;
      
    case SK_StringInit:
      OS << "string initialization";
      break;

    case SK_ObjCObjectConversion:
      OS << "Objective-C object conversion";
      break;
    }
  }
}

void InitializationSequence::dump() const {
  dump(llvm::errs());
}

//===----------------------------------------------------------------------===//
// Initialization helper functions
//===----------------------------------------------------------------------===//
Sema::OwningExprResult 
Sema::PerformCopyInitialization(const InitializedEntity &Entity,
                                SourceLocation EqualLoc,
                                OwningExprResult Init) {
  if (Init.isInvalid())
    return ExprError();

  Expr *InitE = (Expr *)Init.get();
  assert(InitE && "No initialization expression?");

  if (EqualLoc.isInvalid())
    EqualLoc = InitE->getLocStart();

  InitializationKind Kind = InitializationKind::CreateCopy(InitE->getLocStart(),
                                                           EqualLoc);
  InitializationSequence Seq(*this, Entity, Kind, &InitE, 1);
  Init.release();
  return Seq.Perform(*this, Entity, Kind, 
                     MultiExprArg(*this, (void**)&InitE, 1));
}
