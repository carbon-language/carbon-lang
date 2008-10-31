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
using namespace clang;

//===----------------------------------------------------------------------===//
// Decl Allocation/Deallocation Method Implementations
//===----------------------------------------------------------------------===//
 
CXXFieldDecl *CXXFieldDecl::Create(ASTContext &C, CXXRecordDecl *RD,
                                   SourceLocation L, IdentifierInfo *Id,
                                   QualType T, Expr *BW) {
  void *Mem = C.getAllocator().Allocate<CXXFieldDecl>();
  return new (Mem) CXXFieldDecl(RD, L, Id, T, BW);
}

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

void CXXRecordDecl::Destroy(ASTContext &C) {
  for (OverloadedFunctionDecl::function_iterator func 
         = Constructors.function_begin();
       func != Constructors.function_end(); ++func)
    (*func)->Destroy(C);
  RecordDecl::Destroy(C);
}

void 
CXXRecordDecl::setBases(CXXBaseSpecifier const * const *Bases, 
                        unsigned NumBases) {
  if (this->Bases)
    delete [] this->Bases;

  this->Bases = new CXXBaseSpecifier[NumBases];
  this->NumBases = NumBases;
  for (unsigned i = 0; i < NumBases; ++i)
    this->Bases[i] = *Bases[i];
}

CXXMethodDecl *
CXXMethodDecl::Create(ASTContext &C, CXXRecordDecl *RD,
                      SourceLocation L, IdentifierInfo *Id,
                      QualType T, bool isStatic, bool isInline,
                      ScopedDecl *PrevDecl) {
  void *Mem = C.getAllocator().Allocate<CXXMethodDecl>();
  return new (Mem) CXXMethodDecl(RD, L, Id, T, isStatic, isInline, PrevDecl);
}

QualType CXXMethodDecl::getThisType(ASTContext &C) const {
  // C++ 9.3.2p1: The type of this in a member function of a class X is X*.
  // If the member function is declared const, the type of this is const X*,
  // if the member function is declared volatile, the type of this is
  // volatile X*, and if the member function is declared const volatile,
  // the type of this is const volatile X*.

  assert(isInstance() && "No 'this' for static methods!");
  QualType ClassTy = C.getTagDeclType(const_cast<CXXRecordDecl*>(
                                            cast<CXXRecordDecl>(getParent())));
  ClassTy = ClassTy.getWithAdditionalQualifiers(getTypeQualifiers());
  return C.getPointerType(ClassTy).withConst();
}

CXXConstructorDecl *
CXXConstructorDecl::Create(ASTContext &C, CXXRecordDecl *RD,
                           SourceLocation L, IdentifierInfo *Id,
                           QualType T, bool isExplicit,
                           bool isInline, bool isImplicitlyDeclared) {
  void *Mem = C.getAllocator().Allocate<CXXConstructorDecl>();
  return new (Mem) CXXConstructorDecl(RD, L, Id, T, isExplicit, isInline,
                                      isImplicitlyDeclared);
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

CXXClassVarDecl *CXXClassVarDecl::Create(ASTContext &C, CXXRecordDecl *RD,
                                   SourceLocation L, IdentifierInfo *Id,
                                   QualType T, ScopedDecl *PrevDecl) {
  void *Mem = C.getAllocator().Allocate<CXXClassVarDecl>();
  return new (Mem) CXXClassVarDecl(RD, L, Id, T, PrevDecl);
}

OverloadedFunctionDecl *
OverloadedFunctionDecl::Create(ASTContext &C, DeclContext *DC,
                               IdentifierInfo *Id) {
  void *Mem = C.getAllocator().Allocate<OverloadedFunctionDecl>();
  return new (Mem) OverloadedFunctionDecl(DC, Id);
}
