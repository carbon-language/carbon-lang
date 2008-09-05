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
                                     SourceLocation L, IdentifierInfo *Id) {
  Kind DK;
  switch (TK) {
    default: assert(0 && "Invalid TagKind!");
    case TK_enum:   assert(0 && "Enum TagKind passed for Record!");
    case TK_struct: DK = CXXStruct; break;
    case TK_union:  DK = CXXUnion;  break;
    case TK_class:  DK = CXXClass;  break;
  }
  void *Mem = C.getAllocator().Allocate<CXXRecordDecl>();
  return new (Mem) CXXRecordDecl(DK, DC, L, Id);
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
  QualType ClassTy = C.getTagDeclType(cast<CXXRecordDecl>(getParent()));
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
