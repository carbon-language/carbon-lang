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

  if (!p) {
    if (inDecl()) {
      if (VarDecl* VD = dyn_cast<VarDecl>(decl)) 
        if (VD->Init)
          return;
      
      NextDecl();
    }
    else if (inDeclGroup()) {
      if (VarDecl* VD = dyn_cast<VarDecl>(*DGI)) 
        if (VD->Init)
          return;
      
      NextDecl();  
    }
  } else if (inSizeOfTypeVA()) {
    assert(!decl);
    RawVAPtr = 0;
  }
}

void StmtIteratorBase::NextDecl(bool ImmediateAdvance) {
  assert (getVAPtr() == NULL);
  
  if (inDecl()) {
    assert (decl);
    
    if (ImmediateAdvance)
      decl = decl->getNextDeclarator();
    
    for ( ; decl ; decl = decl->getNextDeclarator())
      if (HandleDecl(decl))
        return;
  }
  else {
    assert (inDeclGroup());
    
    if (ImmediateAdvance)
      ++DGI;
    
    for ( ; DGI != DGE; ++DGI)
      if (HandleDecl(*DGI))
        return;
  }

  RawVAPtr = 0;
}

bool StmtIteratorBase::HandleDecl(Decl* D) {
    
  if (VarDecl* VD = dyn_cast<VarDecl>(D)) {        
    if (VariableArrayType* VAPtr = FindVA(VD->getType().getTypePtr())) {
      setVAPtr(VAPtr);
      return true;
    }
        
    if (VD->getInit())
      return true;
  }
  else if (TypedefDecl* TD = dyn_cast<TypedefDecl>(D)) {
    if (VariableArrayType* VAPtr = 
        FindVA(TD->getUnderlyingType().getTypePtr())) {
      setVAPtr(VAPtr);
      return true;
    }
  }
  else if (EnumConstantDecl* ECD = dyn_cast<EnumConstantDecl>(D)) {
    if (ECD->getInitExpr())
      return true;
  }

  return false;  
}

StmtIteratorBase::StmtIteratorBase(ScopedDecl* d)
  : decl(d), RawVAPtr(DeclMode) {
  assert (decl);
  NextDecl(false);
}

StmtIteratorBase::StmtIteratorBase(Decl** dgi, Decl** dge)
  : DGI(dgi), RawVAPtr(DeclGroupMode), DGE(dge) {
  NextDecl(false);
}

StmtIteratorBase::StmtIteratorBase(VariableArrayType* t)
: decl(0), RawVAPtr(SizeOfTypeVAMode) {
  RawVAPtr |= reinterpret_cast<uintptr_t>(t);
}

Stmt*& StmtIteratorBase::GetDeclExpr() const {
  
  if (inDeclGroup()) {
    VarDecl* VD = cast<VarDecl>(*DGI);
    return VD->Init;
  }
  
  assert (inDecl() || inSizeOfTypeVA());
  
  if (VariableArrayType* VAPtr = getVAPtr()) {
    assert (VAPtr->SizeExpr);
    return VAPtr->SizeExpr;
  }
  
  assert (inDecl());

  if (VarDecl* VD = dyn_cast<VarDecl>(decl)) {
    assert (VD->Init);
    return VD->Init;
  }

  EnumConstantDecl* ECD = cast<EnumConstantDecl>(decl);
  return ECD->Init;
}
