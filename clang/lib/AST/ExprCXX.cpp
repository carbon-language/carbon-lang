//===--- ExprCXX.cpp - (C++) Expression AST Node Implementation -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the subclesses of Expr class declared in ExprCXX.h
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/IdentifierTable.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/TypeLoc.h"
using namespace clang;


//===----------------------------------------------------------------------===//
//  Child Iterators for iterating over subexpressions/substatements
//===----------------------------------------------------------------------===//

QualType CXXTypeidExpr::getTypeOperand() const {
  assert(isTypeOperand() && "Cannot call getTypeOperand for typeid(expr)");
  return Operand.get<TypeSourceInfo *>()->getType().getNonReferenceType()
                                                        .getUnqualifiedType();
}

// CXXTypeidExpr - has child iterators if the operand is an expression
Stmt::child_iterator CXXTypeidExpr::child_begin() {
  return isTypeOperand() ? child_iterator() 
                         : reinterpret_cast<Stmt **>(&Operand);
}
Stmt::child_iterator CXXTypeidExpr::child_end() {
  return isTypeOperand() ? child_iterator() 
                         : reinterpret_cast<Stmt **>(&Operand) + 1;
}

// CXXBoolLiteralExpr
Stmt::child_iterator CXXBoolLiteralExpr::child_begin() {
  return child_iterator();
}
Stmt::child_iterator CXXBoolLiteralExpr::child_end() {
  return child_iterator();
}

// CXXNullPtrLiteralExpr
Stmt::child_iterator CXXNullPtrLiteralExpr::child_begin() {
  return child_iterator();
}
Stmt::child_iterator CXXNullPtrLiteralExpr::child_end() {
  return child_iterator();
}

// CXXThisExpr
Stmt::child_iterator CXXThisExpr::child_begin() { return child_iterator(); }
Stmt::child_iterator CXXThisExpr::child_end() { return child_iterator(); }

// CXXThrowExpr
Stmt::child_iterator CXXThrowExpr::child_begin() { return &Op; }
Stmt::child_iterator CXXThrowExpr::child_end() {
  // If Op is 0, we are processing throw; which has no children.
  return Op ? &Op+1 : &Op;
}

// CXXDefaultArgExpr
Stmt::child_iterator CXXDefaultArgExpr::child_begin() {
  return child_iterator();
}
Stmt::child_iterator CXXDefaultArgExpr::child_end() {
  return child_iterator();
}

// CXXScalarValueInitExpr
Stmt::child_iterator CXXScalarValueInitExpr::child_begin() {
  return child_iterator();
}
Stmt::child_iterator CXXScalarValueInitExpr::child_end() {
  return child_iterator();
}

// CXXNewExpr
CXXNewExpr::CXXNewExpr(ASTContext &C, bool globalNew, FunctionDecl *operatorNew,
                       Expr **placementArgs, unsigned numPlaceArgs,
                       SourceRange TypeIdParens, Expr *arraySize,
                       CXXConstructorDecl *constructor, bool initializer,
                       Expr **constructorArgs, unsigned numConsArgs,
                       FunctionDecl *operatorDelete, QualType ty,
                       SourceLocation startLoc, SourceLocation endLoc)
  : Expr(CXXNewExprClass, ty, ty->isDependentType(), ty->isDependentType()),
    GlobalNew(globalNew),
    Initializer(initializer), SubExprs(0), OperatorNew(operatorNew),
    OperatorDelete(operatorDelete), Constructor(constructor),
    TypeIdParens(TypeIdParens), StartLoc(startLoc), EndLoc(endLoc) {
      
  AllocateArgsArray(C, arraySize != 0, numPlaceArgs, numConsArgs);
  unsigned i = 0;
  if (Array)
    SubExprs[i++] = arraySize;
  for (unsigned j = 0; j < NumPlacementArgs; ++j)
    SubExprs[i++] = placementArgs[j];
  for (unsigned j = 0; j < NumConstructorArgs; ++j)
    SubExprs[i++] = constructorArgs[j];
}

void CXXNewExpr::AllocateArgsArray(ASTContext &C, bool isArray,
                                   unsigned numPlaceArgs, unsigned numConsArgs){
  assert(SubExprs == 0 && "SubExprs already allocated");
  Array = isArray;
  NumPlacementArgs = numPlaceArgs;
  NumConstructorArgs = numConsArgs; 
  
  unsigned TotalSize = Array + NumPlacementArgs + NumConstructorArgs;
  SubExprs = new (C) Stmt*[TotalSize];
}


Stmt::child_iterator CXXNewExpr::child_begin() { return &SubExprs[0]; }
Stmt::child_iterator CXXNewExpr::child_end() {
  return &SubExprs[0] + Array + getNumPlacementArgs() + getNumConstructorArgs();
}

// CXXDeleteExpr
Stmt::child_iterator CXXDeleteExpr::child_begin() { return &Argument; }
Stmt::child_iterator CXXDeleteExpr::child_end() { return &Argument+1; }

// CXXPseudoDestructorExpr
Stmt::child_iterator CXXPseudoDestructorExpr::child_begin() { return &Base; }
Stmt::child_iterator CXXPseudoDestructorExpr::child_end() {
  return &Base + 1;
}

PseudoDestructorTypeStorage::PseudoDestructorTypeStorage(TypeSourceInfo *Info)
 : Type(Info) 
{
  Location = Info->getTypeLoc().getLocalSourceRange().getBegin();
}

QualType CXXPseudoDestructorExpr::getDestroyedType() const {
  if (TypeSourceInfo *TInfo = DestroyedType.getTypeSourceInfo())
    return TInfo->getType();
  
  return QualType();
}

SourceRange CXXPseudoDestructorExpr::getSourceRange() const {
  SourceLocation End = DestroyedType.getLocation();
  if (TypeSourceInfo *TInfo = DestroyedType.getTypeSourceInfo())
    End = TInfo->getTypeLoc().getLocalSourceRange().getEnd();
  return SourceRange(Base->getLocStart(), End);
}


// UnresolvedLookupExpr
UnresolvedLookupExpr *
UnresolvedLookupExpr::Create(ASTContext &C, bool Dependent,
                             CXXRecordDecl *NamingClass,
                             NestedNameSpecifier *Qualifier,
                             SourceRange QualifierRange, DeclarationName Name,
                             SourceLocation NameLoc, bool ADL,
                             const TemplateArgumentListInfo &Args,
                             UnresolvedSetIterator Begin, 
                             UnresolvedSetIterator End) 
{
  void *Mem = C.Allocate(sizeof(UnresolvedLookupExpr) + 
                         ExplicitTemplateArgumentList::sizeFor(Args));
  UnresolvedLookupExpr *ULE
    = new (Mem) UnresolvedLookupExpr(C, 
                                     Dependent ? C.DependentTy : C.OverloadTy,
                                     Dependent, NamingClass,
                                     Qualifier, QualifierRange,
                                     Name, NameLoc, ADL,
                                     /*Overload*/ true,
                                     /*ExplicitTemplateArgs*/ true,
                                     Begin, End);

  reinterpret_cast<ExplicitTemplateArgumentList*>(ULE+1)->initializeFrom(Args);

  return ULE;
}

UnresolvedLookupExpr *
UnresolvedLookupExpr::CreateEmpty(ASTContext &C, unsigned NumTemplateArgs) {
  std::size_t size = sizeof(UnresolvedLookupExpr);
  if (NumTemplateArgs != 0)
    size += ExplicitTemplateArgumentList::sizeFor(NumTemplateArgs);

  void *Mem = C.Allocate(size, llvm::alignof<UnresolvedLookupExpr>());
  UnresolvedLookupExpr *E = new (Mem) UnresolvedLookupExpr(EmptyShell());
  E->HasExplicitTemplateArgs = NumTemplateArgs != 0;
  return E;
}

OverloadExpr::OverloadExpr(StmtClass K, ASTContext &C, QualType T, 
                           bool Dependent, NestedNameSpecifier *Qualifier, 
                           SourceRange QRange, DeclarationName Name, 
                           SourceLocation NameLoc, bool HasTemplateArgs,
                           UnresolvedSetIterator Begin, 
                           UnresolvedSetIterator End)
  : Expr(K, T, Dependent, Dependent),
  Results(0), NumResults(0), Name(Name), Qualifier(Qualifier), 
  QualifierRange(QRange), NameLoc(NameLoc), 
  HasExplicitTemplateArgs(HasTemplateArgs)
{
  initializeResults(C, Begin, End);
}

void OverloadExpr::initializeResults(ASTContext &C,
                                     UnresolvedSetIterator Begin,
                                     UnresolvedSetIterator End) {
  assert(Results == 0 && "Results already initialized!");
  NumResults = End - Begin;
  if (NumResults) {
    Results = static_cast<DeclAccessPair *>(
                                C.Allocate(sizeof(DeclAccessPair) * NumResults, 
                                           llvm::alignof<DeclAccessPair>()));
    memcpy(Results, &*Begin.getIterator(), 
           NumResults * sizeof(DeclAccessPair));
  }
}


bool OverloadExpr::ComputeDependence(UnresolvedSetIterator Begin,
                                     UnresolvedSetIterator End,
                                     const TemplateArgumentListInfo *Args) {
  for (UnresolvedSetImpl::const_iterator I = Begin; I != End; ++I)
    if ((*I)->getDeclContext()->isDependentContext())
      return true;

  if (Args && TemplateSpecializationType::anyDependentTemplateArguments(*Args))
    return true;

  return false;
}

CXXRecordDecl *OverloadExpr::getNamingClass() const {
  if (isa<UnresolvedLookupExpr>(this))
    return cast<UnresolvedLookupExpr>(this)->getNamingClass();
  else
    return cast<UnresolvedMemberExpr>(this)->getNamingClass();
}

Stmt::child_iterator UnresolvedLookupExpr::child_begin() {
  return child_iterator();
}
Stmt::child_iterator UnresolvedLookupExpr::child_end() {
  return child_iterator();
}
// UnaryTypeTraitExpr
Stmt::child_iterator UnaryTypeTraitExpr::child_begin() {
  return child_iterator();
}
Stmt::child_iterator UnaryTypeTraitExpr::child_end() {
  return child_iterator();
}

// DependentScopeDeclRefExpr
DependentScopeDeclRefExpr *
DependentScopeDeclRefExpr::Create(ASTContext &C,
                                  NestedNameSpecifier *Qualifier,
                                  SourceRange QualifierRange,
                                  DeclarationName Name,
                                  SourceLocation NameLoc,
                                  const TemplateArgumentListInfo *Args) {
  std::size_t size = sizeof(DependentScopeDeclRefExpr);
  if (Args) size += ExplicitTemplateArgumentList::sizeFor(*Args);
  void *Mem = C.Allocate(size);

  DependentScopeDeclRefExpr *DRE
    = new (Mem) DependentScopeDeclRefExpr(C.DependentTy,
                                          Qualifier, QualifierRange,
                                          Name, NameLoc,
                                          Args != 0);

  if (Args)
    reinterpret_cast<ExplicitTemplateArgumentList*>(DRE+1)
      ->initializeFrom(*Args);

  return DRE;
}

DependentScopeDeclRefExpr *
DependentScopeDeclRefExpr::CreateEmpty(ASTContext &C,
                                       unsigned NumTemplateArgs) {
  std::size_t size = sizeof(DependentScopeDeclRefExpr);
  if (NumTemplateArgs)
    size += ExplicitTemplateArgumentList::sizeFor(NumTemplateArgs);
  void *Mem = C.Allocate(size);

  return new (Mem) DependentScopeDeclRefExpr(QualType(), 0, SourceRange(),
                                             DeclarationName(),SourceLocation(),
                                             NumTemplateArgs != 0);
}

StmtIterator DependentScopeDeclRefExpr::child_begin() {
  return child_iterator();
}

StmtIterator DependentScopeDeclRefExpr::child_end() {
  return child_iterator();
}

bool UnaryTypeTraitExpr::EvaluateTrait(ASTContext& C) const {
  switch(UTT) {
  default: assert(false && "Unknown type trait or not implemented");
  case UTT_IsPOD: return QueriedType->isPODType();
  case UTT_IsLiteral: return QueriedType->isLiteralType();
  case UTT_IsClass: // Fallthrough
  case UTT_IsUnion:
    if (const RecordType *Record = QueriedType->getAs<RecordType>()) {
      bool Union = Record->getDecl()->isUnion();
      return UTT == UTT_IsUnion ? Union : !Union;
    }
    return false;
  case UTT_IsEnum: return QueriedType->isEnumeralType();
  case UTT_IsPolymorphic:
    if (const RecordType *Record = QueriedType->getAs<RecordType>()) {
      // Type traits are only parsed in C++, so we've got CXXRecords.
      return cast<CXXRecordDecl>(Record->getDecl())->isPolymorphic();
    }
    return false;
  case UTT_IsAbstract:
    if (const RecordType *RT = QueriedType->getAs<RecordType>())
      return cast<CXXRecordDecl>(RT->getDecl())->isAbstract();
    return false;
  case UTT_IsEmpty:
    if (const RecordType *Record = QueriedType->getAs<RecordType>()) {
      return !Record->getDecl()->isUnion()
          && cast<CXXRecordDecl>(Record->getDecl())->isEmpty();
    }
    return false;
  case UTT_HasTrivialConstructor:
    // http://gcc.gnu.org/onlinedocs/gcc/Type-Traits.html:
    //   If __is_pod (type) is true then the trait is true, else if type is
    //   a cv class or union type (or array thereof) with a trivial default
    //   constructor ([class.ctor]) then the trait is true, else it is false.
    if (QueriedType->isPODType())
      return true;
    if (const RecordType *RT =
          C.getBaseElementType(QueriedType)->getAs<RecordType>())
      return cast<CXXRecordDecl>(RT->getDecl())->hasTrivialConstructor();
    return false;
  case UTT_HasTrivialCopy:
    // http://gcc.gnu.org/onlinedocs/gcc/Type-Traits.html:
    //   If __is_pod (type) is true or type is a reference type then
    //   the trait is true, else if type is a cv class or union type
    //   with a trivial copy constructor ([class.copy]) then the trait
    //   is true, else it is false.
    if (QueriedType->isPODType() || QueriedType->isReferenceType())
      return true;
    if (const RecordType *RT = QueriedType->getAs<RecordType>())
      return cast<CXXRecordDecl>(RT->getDecl())->hasTrivialCopyConstructor();
    return false;
  case UTT_HasTrivialAssign:
    // http://gcc.gnu.org/onlinedocs/gcc/Type-Traits.html:
    //   If type is const qualified or is a reference type then the
    //   trait is false. Otherwise if __is_pod (type) is true then the
    //   trait is true, else if type is a cv class or union type with
    //   a trivial copy assignment ([class.copy]) then the trait is
    //   true, else it is false.
    // Note: the const and reference restrictions are interesting,
    // given that const and reference members don't prevent a class
    // from having a trivial copy assignment operator (but do cause
    // errors if the copy assignment operator is actually used, q.v.
    // [class.copy]p12).

    if (C.getBaseElementType(QueriedType).isConstQualified())
      return false;
    if (QueriedType->isPODType())
      return true;
    if (const RecordType *RT = QueriedType->getAs<RecordType>())
      return cast<CXXRecordDecl>(RT->getDecl())->hasTrivialCopyAssignment();
    return false;
  case UTT_HasTrivialDestructor:
    // http://gcc.gnu.org/onlinedocs/gcc/Type-Traits.html:
    //   If __is_pod (type) is true or type is a reference type
    //   then the trait is true, else if type is a cv class or union
    //   type (or array thereof) with a trivial destructor
    //   ([class.dtor]) then the trait is true, else it is
    //   false.
    if (QueriedType->isPODType() || QueriedType->isReferenceType())
      return true;
    if (const RecordType *RT =
          C.getBaseElementType(QueriedType)->getAs<RecordType>())
      return cast<CXXRecordDecl>(RT->getDecl())->hasTrivialDestructor();
    return false;
  }
}

SourceRange CXXConstructExpr::getSourceRange() const { 
  // FIXME: Should we know where the parentheses are, if there are any?
  for (std::reverse_iterator<Stmt**> I(&Args[NumArgs]), E(&Args[0]); I!=E;++I) {
    // Ignore CXXDefaultExprs when computing the range, as they don't
    // have a range.
    if (!isa<CXXDefaultArgExpr>(*I))
      return SourceRange(Loc, (*I)->getLocEnd());
  }
  
  return SourceRange(Loc);
}

SourceRange CXXOperatorCallExpr::getSourceRange() const {
  OverloadedOperatorKind Kind = getOperator();
  if (Kind == OO_PlusPlus || Kind == OO_MinusMinus) {
    if (getNumArgs() == 1)
      // Prefix operator
      return SourceRange(getOperatorLoc(),
                         getArg(0)->getSourceRange().getEnd());
    else
      // Postfix operator
      return SourceRange(getArg(0)->getSourceRange().getEnd(),
                         getOperatorLoc());
  } else if (Kind == OO_Call) {
    return SourceRange(getArg(0)->getSourceRange().getBegin(), getRParenLoc());
  } else if (Kind == OO_Subscript) {
    return SourceRange(getArg(0)->getSourceRange().getBegin(), getRParenLoc());
  } else if (getNumArgs() == 1) {
    return SourceRange(getOperatorLoc(), getArg(0)->getSourceRange().getEnd());
  } else if (getNumArgs() == 2) {
    return SourceRange(getArg(0)->getSourceRange().getBegin(),
                       getArg(1)->getSourceRange().getEnd());
  } else {
    return SourceRange();
  }
}

Expr *CXXMemberCallExpr::getImplicitObjectArgument() {
  if (MemberExpr *MemExpr = dyn_cast<MemberExpr>(getCallee()->IgnoreParens()))
    return MemExpr->getBase();

  // FIXME: Will eventually need to cope with member pointers.
  return 0;
}

SourceRange CXXMemberCallExpr::getSourceRange() const {
  SourceLocation LocStart = getCallee()->getLocStart();
  if (LocStart.isInvalid() && getNumArgs() > 0)
    LocStart = getArg(0)->getLocStart();
  return SourceRange(LocStart, getRParenLoc());
}


//===----------------------------------------------------------------------===//
//  Named casts
//===----------------------------------------------------------------------===//

/// getCastName - Get the name of the C++ cast being used, e.g.,
/// "static_cast", "dynamic_cast", "reinterpret_cast", or
/// "const_cast". The returned pointer must not be freed.
const char *CXXNamedCastExpr::getCastName() const {
  switch (getStmtClass()) {
  case CXXStaticCastExprClass:      return "static_cast";
  case CXXDynamicCastExprClass:     return "dynamic_cast";
  case CXXReinterpretCastExprClass: return "reinterpret_cast";
  case CXXConstCastExprClass:       return "const_cast";
  default:                          return "<invalid cast>";
  }
}

CXXStaticCastExpr *CXXStaticCastExpr::Create(ASTContext &C, QualType T,
                                             CastKind K, Expr *Op,
                                             const CXXCastPath *BasePath,
                                             TypeSourceInfo *WrittenTy,
                                             SourceLocation L) {
  unsigned PathSize = (BasePath ? BasePath->size() : 0);
  void *Buffer = C.Allocate(sizeof(CXXStaticCastExpr)
                            + PathSize * sizeof(CXXBaseSpecifier*));
  CXXStaticCastExpr *E =
    new (Buffer) CXXStaticCastExpr(T, K, Op, PathSize, WrittenTy, L);
  if (PathSize) E->setCastPath(*BasePath);
  return E;
}

CXXStaticCastExpr *CXXStaticCastExpr::CreateEmpty(ASTContext &C,
                                                  unsigned PathSize) {
  void *Buffer =
    C.Allocate(sizeof(CXXStaticCastExpr) + PathSize * sizeof(CXXBaseSpecifier*));
  return new (Buffer) CXXStaticCastExpr(EmptyShell(), PathSize);
}

CXXDynamicCastExpr *CXXDynamicCastExpr::Create(ASTContext &C, QualType T,
                                               CastKind K, Expr *Op,
                                               const CXXCastPath *BasePath,
                                               TypeSourceInfo *WrittenTy,
                                               SourceLocation L) {
  unsigned PathSize = (BasePath ? BasePath->size() : 0);
  void *Buffer = C.Allocate(sizeof(CXXDynamicCastExpr)
                            + PathSize * sizeof(CXXBaseSpecifier*));
  CXXDynamicCastExpr *E =
    new (Buffer) CXXDynamicCastExpr(T, K, Op, PathSize, WrittenTy, L);
  if (PathSize) E->setCastPath(*BasePath);
  return E;
}

CXXDynamicCastExpr *CXXDynamicCastExpr::CreateEmpty(ASTContext &C,
                                                    unsigned PathSize) {
  void *Buffer =
    C.Allocate(sizeof(CXXDynamicCastExpr) + PathSize * sizeof(CXXBaseSpecifier*));
  return new (Buffer) CXXDynamicCastExpr(EmptyShell(), PathSize);
}

CXXReinterpretCastExpr *
CXXReinterpretCastExpr::Create(ASTContext &C, QualType T, CastKind K, Expr *Op,
                               const CXXCastPath *BasePath,
                               TypeSourceInfo *WrittenTy, SourceLocation L) {
  unsigned PathSize = (BasePath ? BasePath->size() : 0);
  void *Buffer =
    C.Allocate(sizeof(CXXReinterpretCastExpr) + PathSize * sizeof(CXXBaseSpecifier*));
  CXXReinterpretCastExpr *E =
    new (Buffer) CXXReinterpretCastExpr(T, K, Op, PathSize, WrittenTy, L);
  if (PathSize) E->setCastPath(*BasePath);
  return E;
}

CXXReinterpretCastExpr *
CXXReinterpretCastExpr::CreateEmpty(ASTContext &C, unsigned PathSize) {
  void *Buffer = C.Allocate(sizeof(CXXReinterpretCastExpr)
                            + PathSize * sizeof(CXXBaseSpecifier*));
  return new (Buffer) CXXReinterpretCastExpr(EmptyShell(), PathSize);
}

CXXConstCastExpr *CXXConstCastExpr::Create(ASTContext &C, QualType T, Expr *Op,
                                           TypeSourceInfo *WrittenTy,
                                           SourceLocation L) {
  return new (C) CXXConstCastExpr(T, Op, WrittenTy, L);
}

CXXConstCastExpr *CXXConstCastExpr::CreateEmpty(ASTContext &C) {
  return new (C) CXXConstCastExpr(EmptyShell());
}

CXXFunctionalCastExpr *
CXXFunctionalCastExpr::Create(ASTContext &C, QualType T,
                              TypeSourceInfo *Written, SourceLocation L,
                              CastKind K, Expr *Op, const CXXCastPath *BasePath,
                               SourceLocation R) {
  unsigned PathSize = (BasePath ? BasePath->size() : 0);
  void *Buffer = C.Allocate(sizeof(CXXFunctionalCastExpr)
                            + PathSize * sizeof(CXXBaseSpecifier*));
  CXXFunctionalCastExpr *E =
    new (Buffer) CXXFunctionalCastExpr(T, Written, L, K, Op, PathSize, R);
  if (PathSize) E->setCastPath(*BasePath);
  return E;
}

CXXFunctionalCastExpr *
CXXFunctionalCastExpr::CreateEmpty(ASTContext &C, unsigned PathSize) {
  void *Buffer = C.Allocate(sizeof(CXXFunctionalCastExpr)
                            + PathSize * sizeof(CXXBaseSpecifier*));
  return new (Buffer) CXXFunctionalCastExpr(EmptyShell(), PathSize);
}


CXXDefaultArgExpr *
CXXDefaultArgExpr::Create(ASTContext &C, SourceLocation Loc, 
                          ParmVarDecl *Param, Expr *SubExpr) {
  void *Mem = C.Allocate(sizeof(CXXDefaultArgExpr) + sizeof(Stmt *));
  return new (Mem) CXXDefaultArgExpr(CXXDefaultArgExprClass, Loc, Param, 
                                     SubExpr);
}

CXXTemporary *CXXTemporary::Create(ASTContext &C,
                                   const CXXDestructorDecl *Destructor) {
  return new (C) CXXTemporary(Destructor);
}

CXXBindTemporaryExpr *CXXBindTemporaryExpr::Create(ASTContext &C,
                                                   CXXTemporary *Temp,
                                                   Expr* SubExpr) {
  assert(SubExpr->getType()->isRecordType() &&
         "Expression bound to a temporary must have record type!");

  return new (C) CXXBindTemporaryExpr(Temp, SubExpr);
}

CXXBindReferenceExpr *CXXBindReferenceExpr::Create(ASTContext &C, Expr *SubExpr,
                                                   bool ExtendsLifetime, 
                                                   bool RequiresTemporaryCopy) {
  return new (C) CXXBindReferenceExpr(SubExpr, 
                                      ExtendsLifetime,
                                      RequiresTemporaryCopy);
}

CXXTemporaryObjectExpr::CXXTemporaryObjectExpr(ASTContext &C,
                                               CXXConstructorDecl *Cons,
                                               QualType writtenTy,
                                               SourceLocation tyBeginLoc,
                                               Expr **Args,
                                               unsigned NumArgs,
                                               SourceLocation rParenLoc,
                                               bool ZeroInitialization)
  : CXXConstructExpr(C, CXXTemporaryObjectExprClass, writtenTy, tyBeginLoc,
                     Cons, false, Args, NumArgs, ZeroInitialization),
  TyBeginLoc(tyBeginLoc), RParenLoc(rParenLoc) {
}

CXXConstructExpr *CXXConstructExpr::Create(ASTContext &C, QualType T,
                                           SourceLocation Loc,
                                           CXXConstructorDecl *D, bool Elidable,
                                           Expr **Args, unsigned NumArgs,
                                           bool ZeroInitialization,
                                           ConstructionKind ConstructKind) {
  return new (C) CXXConstructExpr(C, CXXConstructExprClass, T, Loc, D, 
                                  Elidable, Args, NumArgs, ZeroInitialization,
                                  ConstructKind);
}

CXXConstructExpr::CXXConstructExpr(ASTContext &C, StmtClass SC, QualType T,
                                   SourceLocation Loc,
                                   CXXConstructorDecl *D, bool elidable,
                                   Expr **args, unsigned numargs,
                                   bool ZeroInitialization, 
                                   ConstructionKind ConstructKind)
: Expr(SC, T,
       T->isDependentType(),
       (T->isDependentType() ||
        CallExpr::hasAnyValueDependentArguments(args, numargs))),
  Constructor(D), Loc(Loc), Elidable(elidable), 
  ZeroInitialization(ZeroInitialization), ConstructKind(ConstructKind),
  Args(0), NumArgs(numargs) 
{
  if (NumArgs) {
    Args = new (C) Stmt*[NumArgs];
    
    for (unsigned i = 0; i != NumArgs; ++i) {
      assert(args[i] && "NULL argument in CXXConstructExpr");
      Args[i] = args[i];
    }
  }
}

CXXExprWithTemporaries::CXXExprWithTemporaries(ASTContext &C,
                                               Expr *subexpr,
                                               CXXTemporary **temps,
                                               unsigned numtemps)
  : Expr(CXXExprWithTemporariesClass, subexpr->getType(),
       subexpr->isTypeDependent(), subexpr->isValueDependent()),
    SubExpr(subexpr), Temps(0), NumTemps(0) {
  if (numtemps) {
    setNumTemporaries(C, numtemps);
    for (unsigned i = 0; i != numtemps; ++i)
      Temps[i] = temps[i];
  }
}

void CXXExprWithTemporaries::setNumTemporaries(ASTContext &C, unsigned N) {
  assert(Temps == 0 && "Cannot resize with this");
  NumTemps = N;
  Temps = new (C) CXXTemporary*[NumTemps];
}


CXXExprWithTemporaries *CXXExprWithTemporaries::Create(ASTContext &C,
                                                       Expr *SubExpr,
                                                       CXXTemporary **Temps,
                                                       unsigned NumTemps) {
  return new (C) CXXExprWithTemporaries(C, SubExpr, Temps, NumTemps);
}

// CXXBindTemporaryExpr
Stmt::child_iterator CXXBindTemporaryExpr::child_begin() {
  return &SubExpr;
}

Stmt::child_iterator CXXBindTemporaryExpr::child_end() {
  return &SubExpr + 1;
}

// CXXBindReferenceExpr
Stmt::child_iterator CXXBindReferenceExpr::child_begin() {
  return &SubExpr;
}

Stmt::child_iterator CXXBindReferenceExpr::child_end() {
  return &SubExpr + 1;
}

// CXXConstructExpr
Stmt::child_iterator CXXConstructExpr::child_begin() {
  return &Args[0];
}
Stmt::child_iterator CXXConstructExpr::child_end() {
  return &Args[0]+NumArgs;
}

// CXXExprWithTemporaries
Stmt::child_iterator CXXExprWithTemporaries::child_begin() {
  return &SubExpr;
}

Stmt::child_iterator CXXExprWithTemporaries::child_end() {
  return &SubExpr + 1;
}

CXXUnresolvedConstructExpr::CXXUnresolvedConstructExpr(
                                                 SourceLocation TyBeginLoc,
                                                 QualType T,
                                                 SourceLocation LParenLoc,
                                                 Expr **Args,
                                                 unsigned NumArgs,
                                                 SourceLocation RParenLoc)
  : Expr(CXXUnresolvedConstructExprClass, T.getNonReferenceType(),
         T->isDependentType(), true),
    TyBeginLoc(TyBeginLoc),
    Type(T),
    LParenLoc(LParenLoc),
    RParenLoc(RParenLoc),
    NumArgs(NumArgs) {
  Stmt **StoredArgs = reinterpret_cast<Stmt **>(this + 1);
  memcpy(StoredArgs, Args, sizeof(Expr *) * NumArgs);
}

CXXUnresolvedConstructExpr *
CXXUnresolvedConstructExpr::Create(ASTContext &C,
                                   SourceLocation TyBegin,
                                   QualType T,
                                   SourceLocation LParenLoc,
                                   Expr **Args,
                                   unsigned NumArgs,
                                   SourceLocation RParenLoc) {
  void *Mem = C.Allocate(sizeof(CXXUnresolvedConstructExpr) +
                         sizeof(Expr *) * NumArgs);
  return new (Mem) CXXUnresolvedConstructExpr(TyBegin, T, LParenLoc,
                                              Args, NumArgs, RParenLoc);
}

CXXUnresolvedConstructExpr *
CXXUnresolvedConstructExpr::CreateEmpty(ASTContext &C, unsigned NumArgs) {
  Stmt::EmptyShell Empty;
  void *Mem = C.Allocate(sizeof(CXXUnresolvedConstructExpr) +
                         sizeof(Expr *) * NumArgs);
  return new (Mem) CXXUnresolvedConstructExpr(Empty, NumArgs);
}

Stmt::child_iterator CXXUnresolvedConstructExpr::child_begin() {
  return child_iterator(reinterpret_cast<Stmt **>(this + 1));
}

Stmt::child_iterator CXXUnresolvedConstructExpr::child_end() {
  return child_iterator(reinterpret_cast<Stmt **>(this + 1) + NumArgs);
}

CXXDependentScopeMemberExpr::CXXDependentScopeMemberExpr(ASTContext &C,
                                                 Expr *Base, QualType BaseType,
                                                 bool IsArrow,
                                                 SourceLocation OperatorLoc,
                                                 NestedNameSpecifier *Qualifier,
                                                 SourceRange QualifierRange,
                                          NamedDecl *FirstQualifierFoundInScope,
                                                 DeclarationName Member,
                                                 SourceLocation MemberLoc,
                                   const TemplateArgumentListInfo *TemplateArgs)
  : Expr(CXXDependentScopeMemberExprClass, C.DependentTy, true, true),
    Base(Base), BaseType(BaseType), IsArrow(IsArrow),
    HasExplicitTemplateArgs(TemplateArgs != 0),
    OperatorLoc(OperatorLoc),
    Qualifier(Qualifier), QualifierRange(QualifierRange),
    FirstQualifierFoundInScope(FirstQualifierFoundInScope),
    Member(Member), MemberLoc(MemberLoc) {
  if (TemplateArgs)
    getExplicitTemplateArgumentList()->initializeFrom(*TemplateArgs);
}

CXXDependentScopeMemberExpr *
CXXDependentScopeMemberExpr::Create(ASTContext &C,
                                Expr *Base, QualType BaseType, bool IsArrow,
                                SourceLocation OperatorLoc,
                                NestedNameSpecifier *Qualifier,
                                SourceRange QualifierRange,
                                NamedDecl *FirstQualifierFoundInScope,
                                DeclarationName Member,
                                SourceLocation MemberLoc,
                                const TemplateArgumentListInfo *TemplateArgs) {
  if (!TemplateArgs)
    return new (C) CXXDependentScopeMemberExpr(C, Base, BaseType,
                                               IsArrow, OperatorLoc,
                                               Qualifier, QualifierRange,
                                               FirstQualifierFoundInScope,
                                               Member, MemberLoc);

  std::size_t size = sizeof(CXXDependentScopeMemberExpr);
  if (TemplateArgs)
    size += ExplicitTemplateArgumentList::sizeFor(*TemplateArgs);

  void *Mem = C.Allocate(size, llvm::alignof<CXXDependentScopeMemberExpr>());
  return new (Mem) CXXDependentScopeMemberExpr(C, Base, BaseType,
                                               IsArrow, OperatorLoc,
                                               Qualifier, QualifierRange,
                                               FirstQualifierFoundInScope,
                                               Member, MemberLoc, TemplateArgs);
}

CXXDependentScopeMemberExpr *
CXXDependentScopeMemberExpr::CreateEmpty(ASTContext &C,
                                         unsigned NumTemplateArgs) {
  if (NumTemplateArgs == 0)
    return new (C) CXXDependentScopeMemberExpr(C, 0, QualType(),
                                               0, SourceLocation(), 0,
                                               SourceRange(), 0,
                                               DeclarationName(),
                                               SourceLocation());

  std::size_t size = sizeof(CXXDependentScopeMemberExpr) +
                     ExplicitTemplateArgumentList::sizeFor(NumTemplateArgs);
  void *Mem = C.Allocate(size, llvm::alignof<CXXDependentScopeMemberExpr>());
  CXXDependentScopeMemberExpr *E
    =  new (Mem) CXXDependentScopeMemberExpr(C, 0, QualType(),
                                             0, SourceLocation(), 0,
                                             SourceRange(), 0,
                                             DeclarationName(),
                                             SourceLocation(), 0);
  E->HasExplicitTemplateArgs = true;
  return E;
}

Stmt::child_iterator CXXDependentScopeMemberExpr::child_begin() {
  return child_iterator(&Base);
}

Stmt::child_iterator CXXDependentScopeMemberExpr::child_end() {
  if (isImplicitAccess())
    return child_iterator(&Base);
  return child_iterator(&Base + 1);
}

UnresolvedMemberExpr::UnresolvedMemberExpr(ASTContext &C, QualType T, 
                                           bool Dependent, 
                                           bool HasUnresolvedUsing,
                                           Expr *Base, QualType BaseType,
                                           bool IsArrow,
                                           SourceLocation OperatorLoc,
                                           NestedNameSpecifier *Qualifier,
                                           SourceRange QualifierRange,
                                           DeclarationName MemberName,
                                           SourceLocation MemberLoc,
                                   const TemplateArgumentListInfo *TemplateArgs,
                                           UnresolvedSetIterator Begin, 
                                           UnresolvedSetIterator End)
  : OverloadExpr(UnresolvedMemberExprClass, C, T, Dependent,
                 Qualifier, QualifierRange, MemberName, MemberLoc,
                 TemplateArgs != 0, Begin, End),
    IsArrow(IsArrow), HasUnresolvedUsing(HasUnresolvedUsing),
    Base(Base), BaseType(BaseType), OperatorLoc(OperatorLoc) {
  if (TemplateArgs)
    getExplicitTemplateArgs().initializeFrom(*TemplateArgs);
}

UnresolvedMemberExpr *
UnresolvedMemberExpr::Create(ASTContext &C, bool Dependent,
                             bool HasUnresolvedUsing,
                             Expr *Base, QualType BaseType, bool IsArrow,
                             SourceLocation OperatorLoc,
                             NestedNameSpecifier *Qualifier,
                             SourceRange QualifierRange,
                             DeclarationName Member,
                             SourceLocation MemberLoc,
                             const TemplateArgumentListInfo *TemplateArgs,
                             UnresolvedSetIterator Begin, 
                             UnresolvedSetIterator End) {
  std::size_t size = sizeof(UnresolvedMemberExpr);
  if (TemplateArgs)
    size += ExplicitTemplateArgumentList::sizeFor(*TemplateArgs);

  void *Mem = C.Allocate(size, llvm::alignof<UnresolvedMemberExpr>());
  return new (Mem) UnresolvedMemberExpr(C, 
                             Dependent ? C.DependentTy : C.OverloadTy,
                             Dependent, HasUnresolvedUsing, Base, BaseType,
                             IsArrow, OperatorLoc, Qualifier, QualifierRange,
                             Member, MemberLoc, TemplateArgs, Begin, End);
}

UnresolvedMemberExpr *
UnresolvedMemberExpr::CreateEmpty(ASTContext &C, unsigned NumTemplateArgs) {
  std::size_t size = sizeof(UnresolvedMemberExpr);
  if (NumTemplateArgs != 0)
    size += ExplicitTemplateArgumentList::sizeFor(NumTemplateArgs);

  void *Mem = C.Allocate(size, llvm::alignof<UnresolvedMemberExpr>());
  UnresolvedMemberExpr *E = new (Mem) UnresolvedMemberExpr(EmptyShell());
  E->HasExplicitTemplateArgs = NumTemplateArgs != 0;
  return E;
}

CXXRecordDecl *UnresolvedMemberExpr::getNamingClass() const {
  // Unlike for UnresolvedLookupExpr, it is very easy to re-derive this.

  // If there was a nested name specifier, it names the naming class.
  // It can't be dependent: after all, we were actually able to do the
  // lookup.
  CXXRecordDecl *Record = 0;
  if (getQualifier()) {
    Type *T = getQualifier()->getAsType();
    assert(T && "qualifier in member expression does not name type");
    Record = T->getAsCXXRecordDecl();
    assert(Record && "qualifier in member expression does not name record");
  }
  // Otherwise the naming class must have been the base class.
  else {
    QualType BaseType = getBaseType().getNonReferenceType();
    if (isArrow()) {
      const PointerType *PT = BaseType->getAs<PointerType>();
      assert(PT && "base of arrow member access is not pointer");
      BaseType = PT->getPointeeType();
    }
    
    Record = BaseType->getAsCXXRecordDecl();
    assert(Record && "base of member expression does not name record");
  }
  
  return Record;
}

Stmt::child_iterator UnresolvedMemberExpr::child_begin() {
  return child_iterator(&Base);
}

Stmt::child_iterator UnresolvedMemberExpr::child_end() {
  if (isImplicitAccess())
    return child_iterator(&Base);
  return child_iterator(&Base + 1);
}
