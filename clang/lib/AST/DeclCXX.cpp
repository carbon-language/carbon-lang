//===--- DeclCXX.cpp - C++ Declaration AST Node Implementation ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the C++ related Decl classes.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/DeclCXX.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/IdentifierTable.h"
#include "llvm/ADT/STLExtras.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// Decl Allocation/Deallocation Method Implementations
//===----------------------------------------------------------------------===//

TemplateTypeParmDecl *
TemplateTypeParmDecl::Create(ASTContext &C, DeclContext *DC,
                             SourceLocation L, IdentifierInfo *Id,
                             bool Typename) {
  void *Mem = C.getAllocator().Allocate<TemplateTypeParmDecl>();
  return new (Mem) TemplateTypeParmDecl(DC, L, Id, Typename);
}

NonTypeTemplateParmDecl *
NonTypeTemplateParmDecl::Create(ASTContext &C, DeclContext *DC, 
                                SourceLocation L, IdentifierInfo *Id,
                                QualType T, SourceLocation TypeSpecStartLoc) {
  void *Mem = C.getAllocator().Allocate<NonTypeTemplateParmDecl>();
  return new (Mem) NonTypeTemplateParmDecl(DC, L, Id, T, TypeSpecStartLoc);
}

CXXRecordDecl::CXXRecordDecl(TagKind TK, DeclContext *DC,
                             SourceLocation L, IdentifierInfo *Id) 
  : RecordDecl(CXXRecord, TK, DC, L, Id),
    UserDeclaredConstructor(false), UserDeclaredCopyConstructor(false),
    UserDeclaredDestructor(false), Aggregate(true), Polymorphic(false), 
    Bases(0), NumBases(0),
    Conversions(DC, DeclarationName()) { }

CXXRecordDecl *CXXRecordDecl::Create(ASTContext &C, TagKind TK, DeclContext *DC,
                                     SourceLocation L, IdentifierInfo *Id,
                                     CXXRecordDecl* PrevDecl) {
  void *Mem = C.getAllocator().Allocate<CXXRecordDecl>();
  CXXRecordDecl* R = new (Mem) CXXRecordDecl(TK, DC, L, Id);
  C.getTypeDeclType(R, PrevDecl);  
  return R;
}

CXXRecordDecl::~CXXRecordDecl() {
  delete [] Bases;
}

void 
CXXRecordDecl::setBases(CXXBaseSpecifier const * const *Bases, 
                        unsigned NumBases) {
  // C++ [dcl.init.aggr]p1: 
  //   An aggregate is an array or a class (clause 9) with [...]
  //   no base classes [...].
  Aggregate = false;

  if (this->Bases)
    delete [] this->Bases;

  this->Bases = new CXXBaseSpecifier[NumBases];
  this->NumBases = NumBases;
  for (unsigned i = 0; i < NumBases; ++i)
    this->Bases[i] = *Bases[i];
}

bool CXXRecordDecl::hasConstCopyConstructor(ASTContext &Context) const {
  QualType ClassType = Context.getTypeDeclType(const_cast<CXXRecordDecl*>(this));
  DeclarationName ConstructorName 
    = Context.DeclarationNames.getCXXConstructorName(
                                           Context.getCanonicalType(ClassType));
  unsigned TypeQuals;
  DeclContext::lookup_const_iterator Con, ConEnd;
  for (llvm::tie(Con, ConEnd) = this->lookup(Context, ConstructorName);
       Con != ConEnd; ++Con) {
    if (cast<CXXConstructorDecl>(*Con)->isCopyConstructor(Context, TypeQuals) &&
        (TypeQuals & QualType::Const) != 0)
      return true;
  }

  return false;
}

void 
CXXRecordDecl::addedConstructor(ASTContext &Context, 
                                CXXConstructorDecl *ConDecl) {
  if (!ConDecl->isImplicitlyDeclared()) {
    // Note that we have a user-declared constructor.
    UserDeclaredConstructor = true;

    // C++ [dcl.init.aggr]p1: 
    //   An aggregate is an array or a class (clause 9) with no
    //   user-declared constructors (12.1) [...].
    Aggregate = false;

    // Note when we have a user-declared copy constructor, which will
    // suppress the implicit declaration of a copy constructor.
    if (ConDecl->isCopyConstructor(Context))
      UserDeclaredCopyConstructor = true;
  }
}

void CXXRecordDecl::addConversionFunction(ASTContext &Context, 
                                          CXXConversionDecl *ConvDecl) {
  Conversions.addOverload(ConvDecl);
}

CXXMethodDecl *
CXXMethodDecl::Create(ASTContext &C, CXXRecordDecl *RD,
                      SourceLocation L, DeclarationName N,
                      QualType T, bool isStatic, bool isInline,
                      ScopedDecl *PrevDecl) {
  void *Mem = C.getAllocator().Allocate<CXXMethodDecl>();
  return new (Mem) CXXMethodDecl(CXXMethod, RD, L, N, T, isStatic, isInline, 
                                 PrevDecl);
}

QualType CXXMethodDecl::getThisType(ASTContext &C) const {
  // C++ 9.3.2p1: The type of this in a member function of a class X is X*.
  // If the member function is declared const, the type of this is const X*,
  // if the member function is declared volatile, the type of this is
  // volatile X*, and if the member function is declared const volatile,
  // the type of this is const volatile X*.

  assert(isInstance() && "No 'this' for static methods!");
  QualType ClassTy = C.getTagDeclType(const_cast<CXXRecordDecl*>(getParent()));
  ClassTy = ClassTy.getWithAdditionalQualifiers(getTypeQualifiers());
  return C.getPointerType(ClassTy).withConst();
}

CXXBaseOrMemberInitializer::
CXXBaseOrMemberInitializer(QualType BaseType, Expr **Args, unsigned NumArgs) 
  : Args(0), NumArgs(0) {
  BaseOrMember = reinterpret_cast<uintptr_t>(BaseType.getTypePtr());
  assert((BaseOrMember & 0x01) == 0 && "Invalid base class type pointer");
  BaseOrMember |= 0x01;
  
  if (NumArgs > 0) {
    this->NumArgs = NumArgs;
    this->Args = new Expr*[NumArgs];
    for (unsigned Idx = 0; Idx < NumArgs; ++Idx)
      this->Args[Idx] = Args[Idx];
  }
}

CXXBaseOrMemberInitializer::
CXXBaseOrMemberInitializer(FieldDecl *Member, Expr **Args, unsigned NumArgs)
  : Args(0), NumArgs(0) {
  BaseOrMember = reinterpret_cast<uintptr_t>(Member);
  assert((BaseOrMember & 0x01) == 0 && "Invalid member pointer");  

  if (NumArgs > 0) {
    this->NumArgs = NumArgs;
    this->Args = new Expr*[NumArgs];
    for (unsigned Idx = 0; Idx < NumArgs; ++Idx)
      this->Args[Idx] = Args[Idx];
  }
}

CXXBaseOrMemberInitializer::~CXXBaseOrMemberInitializer() {
  delete [] Args;
}

CXXConstructorDecl *
CXXConstructorDecl::Create(ASTContext &C, CXXRecordDecl *RD,
                           SourceLocation L, DeclarationName N,
                           QualType T, bool isExplicit,
                           bool isInline, bool isImplicitlyDeclared) {
  assert(N.getNameKind() == DeclarationName::CXXConstructorName &&
         "Name must refer to a constructor");
  void *Mem = C.getAllocator().Allocate<CXXConstructorDecl>();
  return new (Mem) CXXConstructorDecl(RD, L, N, T, isExplicit, isInline,
                                      isImplicitlyDeclared);
}

bool CXXConstructorDecl::isDefaultConstructor() const {
  // C++ [class.ctor]p5:
  //   A default constructor for a class X is a constructor of class
  //   X that can be called without an argument.
  return (getNumParams() == 0) ||
         (getNumParams() > 0 && getParamDecl(0)->getDefaultArg() != 0);
}

bool 
CXXConstructorDecl::isCopyConstructor(ASTContext &Context, 
                                      unsigned &TypeQuals) const {
  // C++ [class.copy]p2:
  //   A non-template constructor for class X is a copy constructor
  //   if its first parameter is of type X&, const X&, volatile X& or
  //   const volatile X&, and either there are no other parameters
  //   or else all other parameters have default arguments (8.3.6).
  if ((getNumParams() < 1) ||
      (getNumParams() > 1 && getParamDecl(1)->getDefaultArg() == 0))
    return false;

  const ParmVarDecl *Param = getParamDecl(0);

  // Do we have a reference type?
  const ReferenceType *ParamRefType = Param->getType()->getAsReferenceType();
  if (!ParamRefType)
    return false;

  // Is it a reference to our class type?
  QualType PointeeType 
    = Context.getCanonicalType(ParamRefType->getPointeeType());
  QualType ClassTy 
    = Context.getTagDeclType(const_cast<CXXRecordDecl*>(getParent()));
  if (PointeeType.getUnqualifiedType() != ClassTy)
    return false;

  // We have a copy constructor.
  TypeQuals = PointeeType.getCVRQualifiers();
  return true;
}

bool CXXConstructorDecl::isConvertingConstructor() const {
  // C++ [class.conv.ctor]p1:
  //   A constructor declared without the function-specifier explicit
  //   that can be called with a single parameter specifies a
  //   conversion from the type of its first parameter to the type of
  //   its class. Such a constructor is called a converting
  //   constructor.
  if (isExplicit())
    return false;

  return (getNumParams() == 0 && 
          getType()->getAsFunctionTypeProto()->isVariadic()) ||
         (getNumParams() == 1) ||
         (getNumParams() > 1 && getParamDecl(1)->getDefaultArg() != 0);
}

CXXDestructorDecl *
CXXDestructorDecl::Create(ASTContext &C, CXXRecordDecl *RD,
                          SourceLocation L, DeclarationName N,
                          QualType T, bool isInline, 
                          bool isImplicitlyDeclared) {
  assert(N.getNameKind() == DeclarationName::CXXDestructorName &&
         "Name must refer to a destructor");
  void *Mem = C.getAllocator().Allocate<CXXDestructorDecl>();
  return new (Mem) CXXDestructorDecl(RD, L, N, T, isInline, 
                                     isImplicitlyDeclared);
}

CXXConversionDecl *
CXXConversionDecl::Create(ASTContext &C, CXXRecordDecl *RD,
                          SourceLocation L, DeclarationName N,
                          QualType T, bool isInline, bool isExplicit) {
  assert(N.getNameKind() == DeclarationName::CXXConversionFunctionName &&
         "Name must refer to a conversion function");
  void *Mem = C.getAllocator().Allocate<CXXConversionDecl>();
  return new (Mem) CXXConversionDecl(RD, L, N, T, isInline, isExplicit);
}

CXXClassVarDecl *CXXClassVarDecl::Create(ASTContext &C, CXXRecordDecl *RD,
                                   SourceLocation L, IdentifierInfo *Id,
                                   QualType T, ScopedDecl *PrevDecl) {
  void *Mem = C.getAllocator().Allocate<CXXClassVarDecl>();
  return new (Mem) CXXClassVarDecl(RD, L, Id, T, PrevDecl);
}

OverloadedFunctionDecl *
OverloadedFunctionDecl::Create(ASTContext &C, DeclContext *DC,
                               DeclarationName N) {
  void *Mem = C.getAllocator().Allocate<OverloadedFunctionDecl>();
  return new (Mem) OverloadedFunctionDecl(DC, N);
}

LinkageSpecDecl::LinkageSpecDecl(SourceLocation L, LanguageIDs lang, 
                                 Decl **InDecls, unsigned InNumDecls)
  : Decl(LinkageSpec, L), Language(lang), HadBraces(true),
    Decls(0), NumDecls(InNumDecls) {
  Decl **NewDecls = new Decl*[NumDecls];
  for (unsigned I = 0; I < NumDecls; ++I)
    NewDecls[I] = InDecls[I];
  Decls = NewDecls;
}

LinkageSpecDecl::~LinkageSpecDecl() {
  if (HadBraces)
    delete [] (Decl**)Decls;
}

LinkageSpecDecl *LinkageSpecDecl::Create(ASTContext &C,
                                         SourceLocation L,
                                         LanguageIDs Lang, Decl *D) {
  void *Mem = C.getAllocator().Allocate<LinkageSpecDecl>();
  return new (Mem) LinkageSpecDecl(L, Lang, D);
}

LinkageSpecDecl *LinkageSpecDecl::Create(ASTContext &C,
                                         SourceLocation L,
                                         LanguageIDs Lang, 
                                         Decl **Decls, unsigned NumDecls) {
  void *Mem = C.getAllocator().Allocate<LinkageSpecDecl>();
  return new (Mem) LinkageSpecDecl(L, Lang, Decls, NumDecls);
}

LinkageSpecDecl::decl_const_iterator LinkageSpecDecl::decls_begin() const {
  if (hasBraces()) return (Decl**)Decls;
  else return (Decl**)&Decls;
}

LinkageSpecDecl::decl_iterator LinkageSpecDecl::decls_end() const {
  if (hasBraces()) return (Decl**)Decls + NumDecls;
  else return (Decl**)&Decls + 1;
}
