//===--- DeclGroup.cpp - Classes for representing groups of Decls -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the DeclGroup and DeclGroupRef classes.
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
  void* mem = C.Allocate(size, alignment);
  new (mem) DeclGroup(numdecls, decls);
  return static_cast<DeclGroup*>(mem);
}

/// Emit - Serialize a DeclGroup to Bitcode.
void DeclGroup::Emit(llvm::Serializer& S) const {
  S.EmitInt(NumDecls);
  S.BatchEmitOwnedPtrs(NumDecls, &(*this)[0]);
}

/// Read - Deserialize a DeclGroup from Bitcode.
DeclGroup* DeclGroup::Read(llvm::Deserializer& D, ASTContext& C) {
  unsigned NumDecls = (unsigned) D.ReadInt();
  unsigned size = sizeof(DeclGroup) + sizeof(Decl*) * NumDecls;
  unsigned alignment = llvm::AlignOf<DeclGroup>::Alignment;  
  DeclGroup* DG = (DeclGroup*) C.Allocate(size, alignment);
  new (DG) DeclGroup();
  DG->NumDecls = NumDecls;
  D.BatchReadOwnedPtrs(NumDecls, &(*DG)[0], C);
  return DG;
}
  
DeclGroup::DeclGroup(unsigned numdecls, Decl** decls) : NumDecls(numdecls) {
  assert(numdecls > 0);
  assert(decls);
  memcpy(this+1, decls, numdecls * sizeof(*decls));
}

void DeclGroup::Destroy(ASTContext& C) {
  this->~DeclGroup();
  C.Deallocate((void*) this);
}

void DeclGroupRef::Emit(llvm::Serializer& S) const {
  if (isSingleDecl()) {
    S.EmitBool(false);
    S.EmitPtr(D);
  } else {
    S.EmitBool(true);
    S.EmitPtr(&getDeclGroup());        
  }
}

DeclGroupRef DeclGroupRef::ReadVal(llvm::Deserializer& D) {
  if (D.ReadBool())
    return DeclGroupRef(D.ReadPtr<Decl>());
  
  return DeclGroupRef(D.ReadPtr<DeclGroup>());
}
