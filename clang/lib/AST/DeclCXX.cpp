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
  assert(isInstance() && "No 'this' for static methods!");
  QualType ClassTy = C.getTagDeclType(const_cast<CXXRecordDecl*>(
                                            cast<CXXRecordDecl>(getParent())));
  QualType ThisTy = C.getPointerType(ClassTy);
  ThisTy.addConst();
  return ThisTy;
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
