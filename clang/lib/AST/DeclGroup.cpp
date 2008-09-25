//===--- DeclGroup.cpp - Classes for representing groups of Decls -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the DeclGroup, DeclGroupRef, and OwningDeclGroup classes.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/DeclGroup.h"
#include "clang/AST/Decl.h"
#include "clang/AST/ASTContext.h"
#include "llvm/Support/Allocator.h"

using namespace clang;

DeclGroup* DeclGroup::Create(ASTContext& C, unsigned numdecls, Decl** decls) {
  unsigned size = sizeof(DeclGroup) + sizeof(Decl*) * numdecls;
  unsigned alignment = llvm::AlignOf<DeclGroup>::Alignment;  
  void* mem = C.getAllocator().Allocate(size, alignment);                                                   
  new (mem) DeclGroup(numdecls, decls);
  return static_cast<DeclGroup*>(mem);
}

DeclGroup::DeclGroup(unsigned numdecls, Decl** decls) {
  assert (numdecls > 0);
  assert (decls);
  memcpy(this+1, decls, numdecls * sizeof(*decls));
}

void DeclGroup::Destroy(ASTContext& C) {
  Decl** Decls = (Decl**) this + 1;
  
  for (unsigned i = 0; i < NumDecls; ++i)
    Decls[i]->Destroy(C);
  
  this->~DeclGroup();
  C.getAllocator().Deallocate((void*) this);
}

DeclGroupOwningRef::~DeclGroupOwningRef() {
  assert (ThePtr == 0 && "Destroy method not called.");
}

void DeclGroupOwningRef::Destroy(ASTContext& C) {
  if (!ThePtr)
    return;
  
  if (getKind() == DeclKind)
    reinterpret_cast<Decl*>(ThePtr)->Destroy(C);
  else
    reinterpret_cast<DeclGroup*>(ThePtr & ~Mask)->Destroy(C);
  
  ThePtr = 0;
}
