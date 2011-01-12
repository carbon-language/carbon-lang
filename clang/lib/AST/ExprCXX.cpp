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

QualType CXXUuidofExpr::getTypeOperand() const {
  assert(isTypeOperand() && "Cannot call getTypeOperand for __uuidof(expr)");
  return Operand.get<TypeSourceInfo *>()->getType().getNonReferenceType()
                                                        .getUnqualifiedType();
}

// CXXUuidofExpr - has child iterators if the operand is an expression
Stmt::child_iterator CXXUuidofExpr::child_begin() {
  return isTypeOperand() ? child_iterator() 
                         : reinterpret_cast<Stmt **>(&Operand);
}
Stmt::child_iterator CXXUuidofExpr::child_end() {
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
SourceRange CXXScalarValueInitExpr::getSourceRange() const {
  SourceLocation Start = RParenLoc;
  if (TypeInfo)
    Start = TypeInfo->getTypeLoc().getBeginLoc();
  return SourceRange(Start, RParenLoc);
}

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
                       TypeSourceInfo *AllocatedTypeInfo,
                       SourceLocation startLoc, SourceLocation endLoc,
                       SourceLocation constructorLParen,
                       SourceLocation constructorRParen)
  : Expr(CXXNewExprClass, ty, VK_RValue, OK_Ordinary,
         ty->isDependentType(), ty->isDependentType(),
         ty->containsUnexpandedParameterPack()),
    GlobalNew(globalNew),
    Initializer(initializer), SubExprs(0), OperatorNew(operatorNew),
    OperatorDelete(operatorDelete), Constructor(constructor),
    AllocatedTypeInfo(AllocatedTypeInfo), TypeIdParens(TypeIdParens),
    StartLoc(startLoc), EndLoc(endLoc), ConstructorLParen(constructorLParen),
    ConstructorRParen(constructorRParen) {
  AllocateArgsArray(C, arraySize != 0, numPlaceArgs, numConsArgs);
  unsigned i = 0;
  if (Array) {
    if (arraySize->containsUnexpandedParameterPack())
      ExprBits.ContainsUnexpandedParameterPack = true;

    SubExprs[i++] = arraySize;
  }

  for (unsigned j = 0; j < NumPlacementArgs; ++j) {
    if (placementArgs[j]->containsUnexpandedParameterPack())
      ExprBits.ContainsUnexpandedParameterPack = true;

    SubExprs[i++] = placementArgs[j];
  }

  for (unsigned j = 0; j < NumConstructorArgs; ++j) {
    if (constructorArgs[j]->containsUnexpandedParameterPack())
      ExprBits.ContainsUnexpandedParameterPack = true;

    SubExprs[i++] = constructorArgs[j];
  }
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
QualType CXXDeleteExpr::getDestroyedType() const {
  const Expr *Arg = getArgument();
  while (const ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(Arg)) {
    if (ICE->getCastKind() != CK_UserDefinedConversion &&
        ICE->getType()->isVoidPointerType())
      Arg = ICE->getSubExpr();
    else
      break;
  }
  // The type-to-delete may not be a pointer if it's a dependent type.
  const QualType ArgType = Arg->getType();

  if (ArgType->isDependentType() && !ArgType->isPointerType())
    return QualType();

  return ArgType->getAs<PointerType>()->getPointeeType();
}

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

CXXPseudoDestructorExpr::CXXPseudoDestructorExpr(ASTContext &Context,
    Expr *Base, bool isArrow, SourceLocation OperatorLoc,
    NestedNameSpecifier *Qualifier, SourceRange QualifierRange,
    TypeSourceInfo *ScopeType, SourceLocation ColonColonLoc,
    SourceLocation TildeLoc, PseudoDestructorTypeStorage DestroyedType)
  : Expr(CXXPseudoDestructorExprClass,
         Context.getPointerType(Context.getFunctionType(Context.VoidTy, 0, 0,
                                         FunctionProtoType::ExtProtoInfo())),
         VK_RValue, OK_Ordinary,
         /*isTypeDependent=*/(Base->isTypeDependent() ||
           (DestroyedType.getTypeSourceInfo() &&
            DestroyedType.getTypeSourceInfo()->getType()->isDependentType())),
         /*isValueDependent=*/Base->isValueDependent(),
         // ContainsUnexpandedParameterPack
         (Base->containsUnexpandedParameterPack() ||
          (Qualifier && Qualifier->containsUnexpandedParameterPack()) ||
          (ScopeType && 
           ScopeType->getType()->containsUnexpandedParameterPack()) ||
          (DestroyedType.getTypeSourceInfo() &&
           DestroyedType.getTypeSourceInfo()->getType()
                                   ->containsUnexpandedParameterPack()))),
    Base(static_cast<Stmt *>(Base)), IsArrow(isArrow),
    OperatorLoc(OperatorLoc), Qualifier(Qualifier),
    QualifierRange(QualifierRange), 
    ScopeType(ScopeType), ColonColonLoc(ColonColonLoc), TildeLoc(TildeLoc),
    DestroyedType(DestroyedType) { }

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
UnresolvedLookupExpr::Create(ASTContext &C, 
                             CXXRecordDecl *NamingClass,
                             NestedNameSpecifier *Qualifier,
                             SourceRange QualifierRange,
                             const DeclarationNameInfo &NameInfo,
                             bool ADL,
                             const TemplateArgumentListInfo &Args,
                             UnresolvedSetIterator Begin, 
                             UnresolvedSetIterator End) 
{
  void *Mem = C.Allocate(sizeof(UnresolvedLookupExpr) + 
                         ExplicitTemplateArgumentList::sizeFor(Args));
  return new (Mem) UnresolvedLookupExpr(C, NamingClass,
                                        Qualifier, QualifierRange, NameInfo,
                                        ADL, /*Overload*/ true, &Args,
                                        Begin, End);
}

UnresolvedLookupExpr *
UnresolvedLookupExpr::CreateEmpty(ASTContext &C, unsigned NumTemplateArgs) {
  std::size_t size = sizeof(UnresolvedLookupExpr);
  if (NumTemplateArgs != 0)
    size += ExplicitTemplateArgumentList::sizeFor(NumTemplateArgs);

  void *Mem = C.Allocate(size, llvm::alignOf<UnresolvedLookupExpr>());
  UnresolvedLookupExpr *E = new (Mem) UnresolvedLookupExpr(EmptyShell());
  E->HasExplicitTemplateArgs = NumTemplateArgs != 0;
  return E;
}

OverloadExpr::OverloadExpr(StmtClass K, ASTContext &C, 
                           NestedNameSpecifier *Qualifier, SourceRange QRange,
                           const DeclarationNameInfo &NameInfo,
                           const TemplateArgumentListInfo *TemplateArgs,
                           UnresolvedSetIterator Begin, 
                           UnresolvedSetIterator End,
                           bool KnownDependent,
                           bool KnownContainsUnexpandedParameterPack)
  : Expr(K, C.OverloadTy, VK_LValue, OK_Ordinary, KnownDependent, 
         KnownDependent,
         (KnownContainsUnexpandedParameterPack ||
          NameInfo.containsUnexpandedParameterPack() ||
          (Qualifier && Qualifier->containsUnexpandedParameterPack()))),
    Results(0), NumResults(End - Begin), NameInfo(NameInfo), 
    Qualifier(Qualifier), QualifierRange(QRange), 
    HasExplicitTemplateArgs(TemplateArgs != 0)
{
  NumResults = End - Begin;
  if (NumResults) {
    // Determine whether this expression is type-dependent.
    for (UnresolvedSetImpl::const_iterator I = Begin; I != End; ++I) {
      if ((*I)->getDeclContext()->isDependentContext() ||
          isa<UnresolvedUsingValueDecl>(*I)) {
        ExprBits.TypeDependent = true;
        ExprBits.ValueDependent = true;
      }
    }

    Results = static_cast<DeclAccessPair *>(
                                C.Allocate(sizeof(DeclAccessPair) * NumResults, 
                                           llvm::alignOf<DeclAccessPair>()));
    memcpy(Results, &*Begin.getIterator(), 
           NumResults * sizeof(DeclAccessPair));
  }

  // If we have explicit template arguments, check for dependent
  // template arguments and whether they contain any unexpanded pack
  // expansions.
  if (TemplateArgs) {
    bool Dependent = false;
    bool ContainsUnexpandedParameterPack = false;
    getExplicitTemplateArgs().initializeFrom(*TemplateArgs, Dependent,
                                             ContainsUnexpandedParameterPack);

    if (Dependent) {
        ExprBits.TypeDependent = true;
        ExprBits.ValueDependent = true;
    } 
    if (ContainsUnexpandedParameterPack)
      ExprBits.ContainsUnexpandedParameterPack = true;
  }

  if (isTypeDependent())
    setType(C.DependentTy);
}

void OverloadExpr::initializeResults(ASTContext &C,
                                     UnresolvedSetIterator Begin,
                                     UnresolvedSetIterator End) {
  assert(Results == 0 && "Results already initialized!");
  NumResults = End - Begin;
  if (NumResults) {
     Results = static_cast<DeclAccessPair *>(
                               C.Allocate(sizeof(DeclAccessPair) * NumResults,
 
                                          llvm::alignOf<DeclAccessPair>()));
     memcpy(Results, &*Begin.getIterator(), 
            NumResults * sizeof(DeclAccessPair));
  }
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

//BinaryTypeTraitExpr
Stmt::child_iterator BinaryTypeTraitExpr::child_begin() {
  return child_iterator();
}
Stmt::child_iterator BinaryTypeTraitExpr::child_end() {
  return child_iterator();
}

// DependentScopeDeclRefExpr
DependentScopeDeclRefExpr::DependentScopeDeclRefExpr(QualType T,
                            NestedNameSpecifier *Qualifier,
                            SourceRange QualifierRange,
                            const DeclarationNameInfo &NameInfo,
                            const TemplateArgumentListInfo *Args)
  : Expr(DependentScopeDeclRefExprClass, T, VK_LValue, OK_Ordinary,
         true, true,
         (NameInfo.containsUnexpandedParameterPack() ||
          (Qualifier && Qualifier->containsUnexpandedParameterPack()))),
    NameInfo(NameInfo), QualifierRange(QualifierRange), Qualifier(Qualifier),
    HasExplicitTemplateArgs(Args != 0)
{
  if (Args) {
    bool Dependent = true;
    bool ContainsUnexpandedParameterPack
      = ExprBits.ContainsUnexpandedParameterPack;

    reinterpret_cast<ExplicitTemplateArgumentList*>(this+1)
      ->initializeFrom(*Args, Dependent, ContainsUnexpandedParameterPack);
    ExprBits.ContainsUnexpandedParameterPack = ContainsUnexpandedParameterPack;
  }
}

DependentScopeDeclRefExpr *
DependentScopeDeclRefExpr::Create(ASTContext &C,
                                  NestedNameSpecifier *Qualifier,
                                  SourceRange QualifierRange,
                                  const DeclarationNameInfo &NameInfo,
                                  const TemplateArgumentListInfo *Args) {
  std::size_t size = sizeof(DependentScopeDeclRefExpr);
  if (Args)
    size += ExplicitTemplateArgumentList::sizeFor(*Args);
  void *Mem = C.Allocate(size);
  return new (Mem) DependentScopeDeclRefExpr(C.DependentTy,
                                             Qualifier, QualifierRange,
                                             NameInfo, Args);
}

DependentScopeDeclRefExpr *
DependentScopeDeclRefExpr::CreateEmpty(ASTContext &C,
                                       unsigned NumTemplateArgs) {
  std::size_t size = sizeof(DependentScopeDeclRefExpr);
  if (NumTemplateArgs)
    size += ExplicitTemplateArgumentList::sizeFor(NumTemplateArgs);
  void *Mem = C.Allocate(size);
  return new (Mem) DependentScopeDeclRefExpr(QualType(), 0, SourceRange(),
                                             DeclarationNameInfo(), 0);
}

StmtIterator DependentScopeDeclRefExpr::child_begin() {
  return child_iterator();
}

StmtIterator DependentScopeDeclRefExpr::child_end() {
  return child_iterator();
}

SourceRange CXXConstructExpr::getSourceRange() const {
  if (ParenRange.isValid())
    return SourceRange(Loc, ParenRange.getEnd());

  SourceLocation End = Loc;
  for (unsigned I = getNumArgs(); I > 0; --I) {
    const Expr *Arg = getArg(I-1);
    if (!Arg->isDefaultArgument()) {
      SourceLocation NewEnd = Arg->getLocEnd();
      if (NewEnd.isValid()) {
        End = NewEnd;
        break;
      }
    }
  }

  return SourceRange(Loc, End);
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

CXXRecordDecl *CXXMemberCallExpr::getRecordDecl() {
  Expr* ThisArg = getImplicitObjectArgument();
  if (!ThisArg)
    return 0;

  if (ThisArg->getType()->isAnyPointerType())
    return ThisArg->getType()->getPointeeType()->getAsCXXRecordDecl();

  return ThisArg->getType()->getAsCXXRecordDecl();
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
                                             ExprValueKind VK,
                                             CastKind K, Expr *Op,
                                             const CXXCastPath *BasePath,
                                             TypeSourceInfo *WrittenTy,
                                             SourceLocation L, 
                                             SourceLocation RParenLoc) {
  unsigned PathSize = (BasePath ? BasePath->size() : 0);
  void *Buffer = C.Allocate(sizeof(CXXStaticCastExpr)
                            + PathSize * sizeof(CXXBaseSpecifier*));
  CXXStaticCastExpr *E =
    new (Buffer) CXXStaticCastExpr(T, VK, K, Op, PathSize, WrittenTy, L,
                                   RParenLoc);
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
                                               ExprValueKind VK,
                                               CastKind K, Expr *Op,
                                               const CXXCastPath *BasePath,
                                               TypeSourceInfo *WrittenTy,
                                               SourceLocation L, 
                                               SourceLocation RParenLoc) {
  unsigned PathSize = (BasePath ? BasePath->size() : 0);
  void *Buffer = C.Allocate(sizeof(CXXDynamicCastExpr)
                            + PathSize * sizeof(CXXBaseSpecifier*));
  CXXDynamicCastExpr *E =
    new (Buffer) CXXDynamicCastExpr(T, VK, K, Op, PathSize, WrittenTy, L,
                                    RParenLoc);
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
CXXReinterpretCastExpr::Create(ASTContext &C, QualType T, ExprValueKind VK,
                               CastKind K, Expr *Op,
                               const CXXCastPath *BasePath,
                               TypeSourceInfo *WrittenTy, SourceLocation L, 
                               SourceLocation RParenLoc) {
  unsigned PathSize = (BasePath ? BasePath->size() : 0);
  void *Buffer =
    C.Allocate(sizeof(CXXReinterpretCastExpr) + PathSize * sizeof(CXXBaseSpecifier*));
  CXXReinterpretCastExpr *E =
    new (Buffer) CXXReinterpretCastExpr(T, VK, K, Op, PathSize, WrittenTy, L,
                                        RParenLoc);
  if (PathSize) E->setCastPath(*BasePath);
  return E;
}

CXXReinterpretCastExpr *
CXXReinterpretCastExpr::CreateEmpty(ASTContext &C, unsigned PathSize) {
  void *Buffer = C.Allocate(sizeof(CXXReinterpretCastExpr)
                            + PathSize * sizeof(CXXBaseSpecifier*));
  return new (Buffer) CXXReinterpretCastExpr(EmptyShell(), PathSize);
}

CXXConstCastExpr *CXXConstCastExpr::Create(ASTContext &C, QualType T,
                                           ExprValueKind VK, Expr *Op,
                                           TypeSourceInfo *WrittenTy,
                                           SourceLocation L, 
                                           SourceLocation RParenLoc) {
  return new (C) CXXConstCastExpr(T, VK, Op, WrittenTy, L, RParenLoc);
}

CXXConstCastExpr *CXXConstCastExpr::CreateEmpty(ASTContext &C) {
  return new (C) CXXConstCastExpr(EmptyShell());
}

CXXFunctionalCastExpr *
CXXFunctionalCastExpr::Create(ASTContext &C, QualType T, ExprValueKind VK,
                              TypeSourceInfo *Written, SourceLocation L,
                              CastKind K, Expr *Op, const CXXCastPath *BasePath,
                               SourceLocation R) {
  unsigned PathSize = (BasePath ? BasePath->size() : 0);
  void *Buffer = C.Allocate(sizeof(CXXFunctionalCastExpr)
                            + PathSize * sizeof(CXXBaseSpecifier*));
  CXXFunctionalCastExpr *E =
    new (Buffer) CXXFunctionalCastExpr(T, VK, Written, L, K, Op, PathSize, R);
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

CXXTemporaryObjectExpr::CXXTemporaryObjectExpr(ASTContext &C,
                                               CXXConstructorDecl *Cons,
                                               TypeSourceInfo *Type,
                                               Expr **Args,
                                               unsigned NumArgs,
                                               SourceRange parenRange,
                                               bool ZeroInitialization)
  : CXXConstructExpr(C, CXXTemporaryObjectExprClass, 
                     Type->getType().getNonReferenceType(), 
                     Type->getTypeLoc().getBeginLoc(),
                     Cons, false, Args, NumArgs, ZeroInitialization,
                     CXXConstructExpr::CK_Complete, parenRange),
    Type(Type) {
}

SourceRange CXXTemporaryObjectExpr::getSourceRange() const {
  return SourceRange(Type->getTypeLoc().getBeginLoc(),
                     getParenRange().getEnd());
}

CXXConstructExpr *CXXConstructExpr::Create(ASTContext &C, QualType T,
                                           SourceLocation Loc,
                                           CXXConstructorDecl *D, bool Elidable,
                                           Expr **Args, unsigned NumArgs,
                                           bool ZeroInitialization,
                                           ConstructionKind ConstructKind,
                                           SourceRange ParenRange) {
  return new (C) CXXConstructExpr(C, CXXConstructExprClass, T, Loc, D, 
                                  Elidable, Args, NumArgs, ZeroInitialization,
                                  ConstructKind, ParenRange);
}

CXXConstructExpr::CXXConstructExpr(ASTContext &C, StmtClass SC, QualType T,
                                   SourceLocation Loc,
                                   CXXConstructorDecl *D, bool elidable,
                                   Expr **args, unsigned numargs,
                                   bool ZeroInitialization, 
                                   ConstructionKind ConstructKind,
                                   SourceRange ParenRange)
  : Expr(SC, T, VK_RValue, OK_Ordinary,
         T->isDependentType(), T->isDependentType(),
         T->containsUnexpandedParameterPack()),
    Constructor(D), Loc(Loc), ParenRange(ParenRange), Elidable(elidable),
    ZeroInitialization(ZeroInitialization), ConstructKind(ConstructKind),
    Args(0), NumArgs(numargs) 
{
  if (NumArgs) {
    Args = new (C) Stmt*[NumArgs];
    
    for (unsigned i = 0; i != NumArgs; ++i) {
      assert(args[i] && "NULL argument in CXXConstructExpr");

      if (args[i]->isValueDependent())
        ExprBits.ValueDependent = true;
      if (args[i]->containsUnexpandedParameterPack())
        ExprBits.ContainsUnexpandedParameterPack = true;
  
      Args[i] = args[i];
    }
  }
}

ExprWithCleanups::ExprWithCleanups(ASTContext &C,
                                   Expr *subexpr,
                                   CXXTemporary **temps,
                                   unsigned numtemps)
  : Expr(ExprWithCleanupsClass, subexpr->getType(),
         subexpr->getValueKind(), subexpr->getObjectKind(),
         subexpr->isTypeDependent(), subexpr->isValueDependent(),
         subexpr->containsUnexpandedParameterPack()),
    SubExpr(subexpr), Temps(0), NumTemps(0) {
  if (numtemps) {
    setNumTemporaries(C, numtemps);
    for (unsigned i = 0; i != numtemps; ++i)
      Temps[i] = temps[i];
  }
}

void ExprWithCleanups::setNumTemporaries(ASTContext &C, unsigned N) {
  assert(Temps == 0 && "Cannot resize with this");
  NumTemps = N;
  Temps = new (C) CXXTemporary*[NumTemps];
}


ExprWithCleanups *ExprWithCleanups::Create(ASTContext &C,
                                           Expr *SubExpr,
                                           CXXTemporary **Temps,
                                           unsigned NumTemps) {
  return new (C) ExprWithCleanups(C, SubExpr, Temps, NumTemps);
}

// CXXBindTemporaryExpr
Stmt::child_iterator CXXBindTemporaryExpr::child_begin() {
  return &SubExpr;
}

Stmt::child_iterator CXXBindTemporaryExpr::child_end() {
  return &SubExpr + 1;
}

// CXXConstructExpr
Stmt::child_iterator CXXConstructExpr::child_begin() {
  return &Args[0];
}
Stmt::child_iterator CXXConstructExpr::child_end() {
  return &Args[0]+NumArgs;
}

// ExprWithCleanups
Stmt::child_iterator ExprWithCleanups::child_begin() {
  return &SubExpr;
}

Stmt::child_iterator ExprWithCleanups::child_end() {
  return &SubExpr + 1;
}

CXXUnresolvedConstructExpr::CXXUnresolvedConstructExpr(TypeSourceInfo *Type,
                                                 SourceLocation LParenLoc,
                                                 Expr **Args,
                                                 unsigned NumArgs,
                                                 SourceLocation RParenLoc)
  : Expr(CXXUnresolvedConstructExprClass, 
         Type->getType().getNonReferenceType(),
         VK_LValue, OK_Ordinary,
         Type->getType()->isDependentType(), true,
         Type->getType()->containsUnexpandedParameterPack()),
    Type(Type),
    LParenLoc(LParenLoc),
    RParenLoc(RParenLoc),
    NumArgs(NumArgs) {
  Stmt **StoredArgs = reinterpret_cast<Stmt **>(this + 1);
  for (unsigned I = 0; I != NumArgs; ++I) {
    if (Args[I]->containsUnexpandedParameterPack())
      ExprBits.ContainsUnexpandedParameterPack = true;

    StoredArgs[I] = Args[I];
  }
}

CXXUnresolvedConstructExpr *
CXXUnresolvedConstructExpr::Create(ASTContext &C,
                                   TypeSourceInfo *Type,
                                   SourceLocation LParenLoc,
                                   Expr **Args,
                                   unsigned NumArgs,
                                   SourceLocation RParenLoc) {
  void *Mem = C.Allocate(sizeof(CXXUnresolvedConstructExpr) +
                         sizeof(Expr *) * NumArgs);
  return new (Mem) CXXUnresolvedConstructExpr(Type, LParenLoc,
                                              Args, NumArgs, RParenLoc);
}

CXXUnresolvedConstructExpr *
CXXUnresolvedConstructExpr::CreateEmpty(ASTContext &C, unsigned NumArgs) {
  Stmt::EmptyShell Empty;
  void *Mem = C.Allocate(sizeof(CXXUnresolvedConstructExpr) +
                         sizeof(Expr *) * NumArgs);
  return new (Mem) CXXUnresolvedConstructExpr(Empty, NumArgs);
}

SourceRange CXXUnresolvedConstructExpr::getSourceRange() const {
  return SourceRange(Type->getTypeLoc().getBeginLoc(), RParenLoc);
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
                                          DeclarationNameInfo MemberNameInfo,
                                   const TemplateArgumentListInfo *TemplateArgs)
  : Expr(CXXDependentScopeMemberExprClass, C.DependentTy,
         VK_LValue, OK_Ordinary, true, true,
         ((Base && Base->containsUnexpandedParameterPack()) ||
          (Qualifier && Qualifier->containsUnexpandedParameterPack()) ||
          MemberNameInfo.containsUnexpandedParameterPack())),
    Base(Base), BaseType(BaseType), IsArrow(IsArrow),
    HasExplicitTemplateArgs(TemplateArgs != 0),
    OperatorLoc(OperatorLoc),
    Qualifier(Qualifier), QualifierRange(QualifierRange),
    FirstQualifierFoundInScope(FirstQualifierFoundInScope),
    MemberNameInfo(MemberNameInfo) {
  if (TemplateArgs) {
    bool Dependent = true;
    bool ContainsUnexpandedParameterPack = false;
    getExplicitTemplateArgs().initializeFrom(*TemplateArgs, Dependent,
                                             ContainsUnexpandedParameterPack);
    if (ContainsUnexpandedParameterPack)
      ExprBits.ContainsUnexpandedParameterPack = true;
  }
}

CXXDependentScopeMemberExpr::CXXDependentScopeMemberExpr(ASTContext &C,
                          Expr *Base, QualType BaseType,
                          bool IsArrow,
                          SourceLocation OperatorLoc,
                          NestedNameSpecifier *Qualifier,
                          SourceRange QualifierRange,
                          NamedDecl *FirstQualifierFoundInScope,
                          DeclarationNameInfo MemberNameInfo)
  : Expr(CXXDependentScopeMemberExprClass, C.DependentTy,
         VK_LValue, OK_Ordinary, true, true,
         ((Base && Base->containsUnexpandedParameterPack()) ||
          (Qualifier && Qualifier->containsUnexpandedParameterPack()) ||
          MemberNameInfo.containsUnexpandedParameterPack())),
    Base(Base), BaseType(BaseType), IsArrow(IsArrow),
    HasExplicitTemplateArgs(false), OperatorLoc(OperatorLoc),
    Qualifier(Qualifier), QualifierRange(QualifierRange),
    FirstQualifierFoundInScope(FirstQualifierFoundInScope),
    MemberNameInfo(MemberNameInfo) { }

CXXDependentScopeMemberExpr *
CXXDependentScopeMemberExpr::Create(ASTContext &C,
                                Expr *Base, QualType BaseType, bool IsArrow,
                                SourceLocation OperatorLoc,
                                NestedNameSpecifier *Qualifier,
                                SourceRange QualifierRange,
                                NamedDecl *FirstQualifierFoundInScope,
                                DeclarationNameInfo MemberNameInfo,
                                const TemplateArgumentListInfo *TemplateArgs) {
  if (!TemplateArgs)
    return new (C) CXXDependentScopeMemberExpr(C, Base, BaseType,
                                               IsArrow, OperatorLoc,
                                               Qualifier, QualifierRange,
                                               FirstQualifierFoundInScope,
                                               MemberNameInfo);

  std::size_t size = sizeof(CXXDependentScopeMemberExpr);
  if (TemplateArgs)
    size += ExplicitTemplateArgumentList::sizeFor(*TemplateArgs);

  void *Mem = C.Allocate(size, llvm::alignOf<CXXDependentScopeMemberExpr>());
  return new (Mem) CXXDependentScopeMemberExpr(C, Base, BaseType,
                                               IsArrow, OperatorLoc,
                                               Qualifier, QualifierRange,
                                               FirstQualifierFoundInScope,
                                               MemberNameInfo, TemplateArgs);
}

CXXDependentScopeMemberExpr *
CXXDependentScopeMemberExpr::CreateEmpty(ASTContext &C,
                                         unsigned NumTemplateArgs) {
  if (NumTemplateArgs == 0)
    return new (C) CXXDependentScopeMemberExpr(C, 0, QualType(),
                                               0, SourceLocation(), 0,
                                               SourceRange(), 0,
                                               DeclarationNameInfo());

  std::size_t size = sizeof(CXXDependentScopeMemberExpr) +
                     ExplicitTemplateArgumentList::sizeFor(NumTemplateArgs);
  void *Mem = C.Allocate(size, llvm::alignOf<CXXDependentScopeMemberExpr>());
  CXXDependentScopeMemberExpr *E
    =  new (Mem) CXXDependentScopeMemberExpr(C, 0, QualType(),
                                             0, SourceLocation(), 0,
                                             SourceRange(), 0,
                                             DeclarationNameInfo(), 0);
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

UnresolvedMemberExpr::UnresolvedMemberExpr(ASTContext &C, 
                                           bool HasUnresolvedUsing,
                                           Expr *Base, QualType BaseType,
                                           bool IsArrow,
                                           SourceLocation OperatorLoc,
                                           NestedNameSpecifier *Qualifier,
                                           SourceRange QualifierRange,
                                   const DeclarationNameInfo &MemberNameInfo,
                                   const TemplateArgumentListInfo *TemplateArgs,
                                           UnresolvedSetIterator Begin, 
                                           UnresolvedSetIterator End)
  : OverloadExpr(UnresolvedMemberExprClass, C, 
                 Qualifier, QualifierRange, MemberNameInfo,
                 TemplateArgs, Begin, End,
                 // Dependent
                 ((Base && Base->isTypeDependent()) ||
                  BaseType->isDependentType()),
                 // Contains unexpanded parameter pack
                 ((Base && Base->containsUnexpandedParameterPack()) ||
                  BaseType->containsUnexpandedParameterPack())),
    IsArrow(IsArrow), HasUnresolvedUsing(HasUnresolvedUsing),
    Base(Base), BaseType(BaseType), OperatorLoc(OperatorLoc) {
}

UnresolvedMemberExpr *
UnresolvedMemberExpr::Create(ASTContext &C, 
                             bool HasUnresolvedUsing,
                             Expr *Base, QualType BaseType, bool IsArrow,
                             SourceLocation OperatorLoc,
                             NestedNameSpecifier *Qualifier,
                             SourceRange QualifierRange,
                             const DeclarationNameInfo &MemberNameInfo,
                             const TemplateArgumentListInfo *TemplateArgs,
                             UnresolvedSetIterator Begin, 
                             UnresolvedSetIterator End) {
  std::size_t size = sizeof(UnresolvedMemberExpr);
  if (TemplateArgs)
    size += ExplicitTemplateArgumentList::sizeFor(*TemplateArgs);

  void *Mem = C.Allocate(size, llvm::alignOf<UnresolvedMemberExpr>());
  return new (Mem) UnresolvedMemberExpr(C, 
                             HasUnresolvedUsing, Base, BaseType,
                             IsArrow, OperatorLoc, Qualifier, QualifierRange,
                             MemberNameInfo, TemplateArgs, Begin, End);
}

UnresolvedMemberExpr *
UnresolvedMemberExpr::CreateEmpty(ASTContext &C, unsigned NumTemplateArgs) {
  std::size_t size = sizeof(UnresolvedMemberExpr);
  if (NumTemplateArgs != 0)
    size += ExplicitTemplateArgumentList::sizeFor(NumTemplateArgs);

  void *Mem = C.Allocate(size, llvm::alignOf<UnresolvedMemberExpr>());
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

Stmt::child_iterator CXXNoexceptExpr::child_begin() {
  return child_iterator(&Operand);
}
Stmt::child_iterator CXXNoexceptExpr::child_end() {
  return child_iterator(&Operand + 1);
}

SourceRange PackExpansionExpr::getSourceRange() const {
  return SourceRange(Pattern->getLocStart(), EllipsisLoc);
}

Stmt::child_iterator PackExpansionExpr::child_begin() {
  return child_iterator(&Pattern);
}

Stmt::child_iterator PackExpansionExpr::child_end() {
  return child_iterator(&Pattern + 1);
}

SourceRange SizeOfPackExpr::getSourceRange() const {
  return SourceRange(OperatorLoc, RParenLoc);
}

Stmt::child_iterator SizeOfPackExpr::child_begin() {
  return child_iterator();
}

Stmt::child_iterator SizeOfPackExpr::child_end() {
  return child_iterator();
}
