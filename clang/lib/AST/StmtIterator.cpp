//===--- StmtIterator.cpp - Iterators for Statements ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines internal methods for StmtIterator.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/StmtIterator.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Decl.h"

using namespace clang;

static inline VariableArrayType* FindVA(Type* t) {
  while (ArrayType* vt = dyn_cast<ArrayType>(t)) {
    if (VariableArrayType* vat = dyn_cast<VariableArrayType>(vt))
      if (vat->getSizeExpr())
        return vat;
    
    t = vt->getElementType().getTypePtr();
  }
  
  return NULL;
}

void StmtIteratorBase::NextVA() {
  assert (getVAPtr());

  VariableArrayType* p = getVAPtr();
  p = FindVA(p->getElementType().getTypePtr());
  setVAPtr(p);

  if (!p && inDecl()) {
    if (VarDecl* VD = dyn_cast<VarDecl>(decl)) 
      if (VD->Init)
        return;
      
    NextDecl();
  } else if (inSizeOfTypeVA()) {
    assert(!decl);
    RawVAPtr = 0;
  }
}

void StmtIteratorBase::NextDecl(bool ImmediateAdvance) {
  assert (inDecl());
  assert (getVAPtr() == NULL);
  assert (decl);
  
  if (ImmediateAdvance) {
    decl = decl->getNextDeclarator();

    if (!decl) {
      RawVAPtr = 0;
      return;
    }
  }    
  
  for ( ; decl ; decl = decl->getNextDeclarator()) {
    if (VarDecl* VD = dyn_cast<VarDecl>(decl)) {        
      if (VariableArrayType* VAPtr = FindVA(VD->getType().getTypePtr())) {
        setVAPtr(VAPtr);
        return;
      }
      
      if (VD->getInit())
        return;    
    }
    else if (TypedefDecl* TD = dyn_cast<TypedefDecl>(decl)) {
      if (VariableArrayType* VAPtr = 
           FindVA(TD->getUnderlyingType().getTypePtr())) {
        setVAPtr(VAPtr);
        return;
      }
    }
    else if (EnumConstantDecl* ECD = dyn_cast<EnumConstantDecl>(decl))
      if (ECD->getInitExpr())
        return;  
  }
  
  if (!decl) {
    RawVAPtr = 0;
    return;
  }
}

StmtIteratorBase::StmtIteratorBase(ScopedDecl* d)
  : decl(d), RawVAPtr(DeclMode) {
  assert (decl);
  NextDecl(false);
}

StmtIteratorBase::StmtIteratorBase(VariableArrayType* t)
: decl(NULL), RawVAPtr(SizeOfTypeVAMode) {
  RawVAPtr |= reinterpret_cast<uintptr_t>(t);
}

Stmt*& StmtIteratorBase::GetDeclExpr() const {
  if (VariableArrayType* VAPtr = getVAPtr()) {
    assert (VAPtr->SizeExpr);
    return VAPtr->SizeExpr;
  }

  if (VarDecl* VD = dyn_cast<VarDecl>(decl)) {
    assert (VD->Init);
    return VD->Init;
  }

  EnumConstantDecl* ECD = cast<EnumConstantDecl>(decl);
  return ECD->Init;
}
