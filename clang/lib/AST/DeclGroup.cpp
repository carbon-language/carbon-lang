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
#include "llvm/Bitcode/Serialize.h"
#include "llvm/Bitcode/Deserialize.h"

using namespace clang;

DeclGroup* DeclGroup::Create(ASTContext& C, unsigned numdecls, Decl** decls) {
  assert (numdecls > 0);
  unsigned size = sizeof(DeclGroup) + sizeof(Decl*) * numdecls;
  unsigned alignment = llvm::AlignOf<DeclGroup>::Alignment;  
  void* mem = C.getAllocator().Allocate(size, alignment);
  new (mem) DeclGroup(numdecls, decls);
  return static_cast<DeclGroup*>(mem);
}

/// Emit - Serialize a DeclGroup to Bitcode.
void DeclGroup::Emit(llvm::Serializer& S) const {
  S.EmitInt(NumDecls);
  S.BatchEmitOwnedPtrs(NumDecls, &(*this)[0]);
}

/// Read - Deserialize a DeclGroup from Bitcode.
DeclGroup* DeclGroup::Create(llvm::Deserializer& D, ASTContext& C) {
  unsigned NumDecls = (unsigned) D.ReadInt();
  unsigned size = sizeof(DeclGroup) + sizeof(Decl*) * NumDecls;
  unsigned alignment = llvm::AlignOf<DeclGroup>::Alignment;  
  DeclGroup* DG = (DeclGroup*) C.getAllocator().Allocate(size, alignment);
  new (DG) DeclGroup();
  DG->NumDecls = NumDecls;
  D.BatchReadOwnedPtrs(NumDecls, &(*DG)[0], C);
  return DG;
}
  
DeclGroup::DeclGroup(unsigned numdecls, Decl** decls) : NumDecls(numdecls) {
  assert (numdecls > 0);
  assert (decls);
  memcpy(this+1, decls, numdecls * sizeof(*decls));
}

void DeclGroup::Destroy(ASTContext& C) {
  Decl** Decls = (Decl**) (this + 1);
  
  for (unsigned i = 0; i < NumDecls; ++i)
    Decls[i]->Destroy(C);
  
  this->~DeclGroup();
  C.getAllocator().Deallocate((void*) this);
}

DeclGroupOwningRef::~DeclGroupOwningRef() {
  assert (D == 0 && "Destroy method not called.");
}

void DeclGroupOwningRef::Destroy(ASTContext& C) {
  if (!D)
    return;
  
  if (getKind() == DeclKind)
    D->Destroy(C);
  else    
    reinterpret_cast<DeclGroup*>(reinterpret_cast<uintptr_t>(D) &
                                 ~Mask)->Destroy(C);
  
  D = 0;
}

void DeclGroupRef::Emit(llvm::Serializer& S) const {
  if (getKind() == DeclKind) {
    S.EmitBool(false);
    S.EmitPtr(D);
  }
  else {
    S.EmitBool(true);
    S.EmitPtr(reinterpret_cast<DeclGroup*>(reinterpret_cast<uintptr_t>(D) 
                                           & ~Mask));        
  }
}

DeclGroupRef DeclGroupRef::ReadVal(llvm::Deserializer& D) {
  if (D.ReadBool())
    return DeclGroupRef(D.ReadPtr<Decl>());
  
  return DeclGroupRef(D.ReadPtr<DeclGroup>());
}

void DeclGroupOwningRef::Emit(llvm::Serializer& S) const {
  if (getKind() == DeclKind) {
    S.EmitBool(false);
    S.EmitOwnedPtr(D);
  }
  else {
    S.EmitBool(true);
    S.EmitOwnedPtr(reinterpret_cast<DeclGroup*>(reinterpret_cast<uintptr_t>(D)
                                                & ~Mask));        
  }
}

DeclGroupOwningRef& DeclGroupOwningRef::Read(llvm::Deserializer& Dezr, 
                                             ASTContext& C) {
  
  if (!Dezr.ReadBool())
    D = Dezr.ReadOwnedPtr<Decl>(C);
  else {
    uintptr_t x = reinterpret_cast<uintptr_t>(Dezr.ReadOwnedPtr<DeclGroup>(C));
    D = reinterpret_cast<Decl*>(x | DeclGroupKind);
  }
  
  return *this;
}
