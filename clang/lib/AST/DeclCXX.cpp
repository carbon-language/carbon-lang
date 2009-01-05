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

TemplateParameterList::TemplateParameterList(Decl **Params, unsigned NumParams)
  : NumParams(NumParams) {
  for (unsigned Idx = 0; Idx < NumParams; ++Idx)
    begin()[Idx] = Params[Idx];
}

TemplateParameterList *
TemplateParameterList::Create(ASTContext &C, Decl **Params, 
                              unsigned NumParams) {
  unsigned Size = sizeof(TemplateParameterList) + sizeof(Decl *) * NumParams;
  unsigned Align = llvm::AlignOf<TemplateParameterList>::Alignment;
  void *Mem = C.getAllocator().Allocate(Size, Align);
  return new (Mem) TemplateParameterList(Params, NumParams);
}

CXXRecordDecl::CXXRecordDecl(TagKind TK, DeclContext *DC,
                             SourceLocation L, IdentifierInfo *Id) 
  : RecordDecl(CXXRecord, TK, DC, L, Id),
    UserDeclaredConstructor(false), UserDeclaredCopyConstructor(false),
    UserDeclaredCopyAssignment(false), UserDeclaredDestructor(false),
    Aggregate(true), PlainOldData(true), Polymorphic(false), Bases(0),
    NumBases(0), Conversions(DC, DeclarationName()) { }

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
  QualType ClassType
    = Context.getTypeDeclType(const_cast<CXXRecordDecl*>(this));
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

bool CXXRecordDecl::hasConstCopyAssignment(ASTContext &Context) const {
  QualType ClassType = Context.getCanonicalType(Context.getTypeDeclType(
    const_cast<CXXRecordDecl*>(this)));
  DeclarationName OpName =Context.DeclarationNames.getCXXOperatorName(OO_Equal);

  DeclContext::lookup_const_iterator Op, OpEnd;
  for (llvm::tie(Op, OpEnd) = this->lookup(Context, OpName);
       Op != OpEnd; ++Op) {
    // C++ [class.copy]p9:
    //   A user-declared copy assignment operator is a non-static non-template
    //   member function of class X with exactly one parameter of type X, X&,
    //   const X&, volatile X& or const volatile X&.
    const CXXMethodDecl* Method = cast<CXXMethodDecl>(*Op);
    if (Method->isStatic())
      continue;
    // TODO: Skip templates? Or is this implicitly done due to parameter types?
    const FunctionTypeProto *FnType =
      Method->getType()->getAsFunctionTypeProto();
    assert(FnType && "Overloaded operator has no prototype.");
    // Don't assert on this; an invalid decl might have been left in the AST.
    if (FnType->getNumArgs() != 1 || FnType->isVariadic())
      continue;
    bool AcceptsConst = true;
    QualType ArgType = FnType->getArgType(0);
    if (const ReferenceType *Ref = ArgType->getAsReferenceType()) {
      ArgType = Ref->getPointeeType();
      // Is it a non-const reference?
      if (!ArgType.isConstQualified())
        AcceptsConst = false;
    }
    if (Context.getCanonicalType(ArgType).getUnqualifiedType() != ClassType)
      continue;

    // We have a single argument of type cv X or cv X&, i.e. we've found the
    // copy assignment operator. Return whether it accepts const arguments.
    return AcceptsConst;
  }
  assert(isInvalidDecl() &&
         "No copy assignment operator declared in valid code.");
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

    // C++ [class]p4:
    //   A POD-struct is an aggregate class [...]
    PlainOldData = false;

    // Note when we have a user-declared copy constructor, which will
    // suppress the implicit declaration of a copy constructor.
    if (ConDecl->isCopyConstructor(Context))
      UserDeclaredCopyConstructor = true;
  }
}

void CXXRecordDecl::addedAssignmentOperator(ASTContext &Context,
                                            CXXMethodDecl *OpDecl) {
  // We're interested specifically in copy assignment operators.
  // Unlike addedConstructor, this method is not called for implicit
  // declarations.
  const FunctionTypeProto *FnType = OpDecl->getType()->getAsFunctionTypeProto();
  assert(FnType && "Overloaded operator has no proto function type.");
  assert(FnType->getNumArgs() == 1 && !FnType->isVariadic());
  QualType ArgType = FnType->getArgType(0);
  if (const ReferenceType *Ref = ArgType->getAsReferenceType())
    ArgType = Ref->getPointeeType();

  ArgType = ArgType.getUnqualifiedType();
  QualType ClassType = Context.getCanonicalType(Context.getTypeDeclType(
    const_cast<CXXRecordDecl*>(this)));

  if (ClassType != Context.getCanonicalType(ArgType))
    return;

  // This is a copy assignment operator.
  // Suppress the implicit declaration of a copy constructor.
  UserDeclaredCopyAssignment = true;

  // C++ [class]p4:
  //   A POD-struct is an aggregate class that [...] has no user-defined copy
  //   assignment operator [...].
  PlainOldData = false;
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

LinkageSpecDecl *LinkageSpecDecl::Create(ASTContext &C,
                                         DeclContext *DC, 
                                         SourceLocation L,
                                         LanguageIDs Lang, bool Braces) {
  void *Mem = C.getAllocator().Allocate<LinkageSpecDecl>();
  return new (Mem) LinkageSpecDecl(DC, L, Lang, Braces);
}
