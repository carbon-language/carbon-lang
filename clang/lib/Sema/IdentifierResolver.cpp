//===- IdentifierResolver.cpp - Lexical Scope Name lookup -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the IdentifierResolver class, which is used for lexical
// scoped lookup, based on identifier.
//
//===----------------------------------------------------------------------===//

#include "IdentifierResolver.h"
#include <list>
#include <vector>

using namespace clang;


/// IdDeclInfoMap - Associates IdDeclInfos with Identifiers.
/// Allocates 'pools' (vectors of IdDeclInfos) to avoid allocating each
/// individual IdDeclInfo to heap.
class IdentifierResolver::IdDeclInfoMap {
  static const unsigned int VECTOR_SIZE = 512;
  // Holds vectors of IdDeclInfos that serve as 'pools'.
  // New vectors are added when the current one is full.
  std::list< std::vector<IdDeclInfo> > IDIVecs;
  unsigned int CurIndex;
  
public:
  IdDeclInfoMap() : CurIndex(VECTOR_SIZE) {}

  /// Returns the IdDeclInfo associated to the IdentifierInfo.
  /// It creates a new IdDeclInfo if one was not created before for this id.
  IdDeclInfo &operator[](IdentifierInfo *II);
};


IdentifierResolver::IdentifierResolver() : IdDeclInfos(new IdDeclInfoMap) {}
IdentifierResolver::~IdentifierResolver() {
  delete IdDeclInfos;
}

/// AddDecl - Link the decl to its shadowed decl chain.
void IdentifierResolver::AddDecl(NamedDecl *D) {
  IdentifierInfo *II = D->getIdentifier();
  void *Ptr = II->getFETokenInfo<void>();

  if (!Ptr) {
    II->setFETokenInfo(D);
    return;
  }

  IdDeclInfo *IDI;

  if (isDeclPtr(Ptr)) {
    II->setFETokenInfo(NULL);
    IDI = &(*IdDeclInfos)[II];
    NamedDecl *PrevD = static_cast<NamedDecl*>(Ptr);
    IDI->AddDecl(PrevD);
  } else
    IDI = toIdDeclInfo(Ptr);

  IDI->AddDecl(D);
}

/// AddShadowedDecl - Link the decl to its shadowed decl chain putting it
/// after the decl that the iterator points to, thus the 'Shadow' decl will be
/// encountered before the 'D' decl.
void IdentifierResolver::AddShadowedDecl(NamedDecl *D, NamedDecl *Shadow) {
  assert(D->getIdentifier() == Shadow->getIdentifier() && "Different ids!");
  assert(LookupContext(D) == LookupContext(Shadow) && "Different context!");

  IdentifierInfo *II = D->getIdentifier();
  void *Ptr = II->getFETokenInfo<void>();
  assert(Ptr && "No decl from Ptr ?");

  IdDeclInfo *IDI;

  if (isDeclPtr(Ptr)) {
    II->setFETokenInfo(NULL);
    IDI = &(*IdDeclInfos)[II];
    NamedDecl *PrevD = static_cast<NamedDecl*>(Ptr);
    assert(PrevD == Shadow && "Invalid shadow decl ?");
    IDI->AddDecl(D);
    IDI->AddDecl(PrevD);
    return;
  }

  IDI = toIdDeclInfo(Ptr);
  IDI->AddShadowed(D, Shadow);
}

/// RemoveDecl - Unlink the decl from its shadowed decl chain.
/// The decl must already be part of the decl chain.
void IdentifierResolver::RemoveDecl(NamedDecl *D) {
  assert(D && "null param passed");
  IdentifierInfo *II = D->getIdentifier();
  void *Ptr = II->getFETokenInfo<void>();

  assert(Ptr && "Didn't find this decl on its identifier's chain!");

  if (isDeclPtr(Ptr)) {
    assert(D == Ptr && "Didn't find this decl on its identifier's chain!");
    II->setFETokenInfo(NULL);
    return;
  }
  
  return toIdDeclInfo(Ptr)->RemoveDecl(D);
}

/// begin - Returns an iterator for all decls, starting at the given
/// declaration context.
IdentifierResolver::iterator
IdentifierResolver::begin(const IdentifierInfo *II, DeclContext *Ctx) {
  assert(Ctx && "null param passed");

  void *Ptr = II->getFETokenInfo<void>();
  if (!Ptr) return end(II);

  LookupContext LC(Ctx);

  if (isDeclPtr(Ptr)) {
    NamedDecl *D = static_cast<NamedDecl*>(Ptr);

    if (LC.isEqOrContainedBy(LookupContext(D)))
      return iterator(D);
    else
      return end(II);

  }
  
  IdDeclInfo *IDI = toIdDeclInfo(Ptr);
  return iterator(IDI->FindContext(LC));
}

/// ctx_begin - Returns an iterator for only decls that belong to the given
/// declaration context.
IdentifierResolver::ctx_iterator
IdentifierResolver::ctx_begin(const IdentifierInfo *II, DeclContext *Ctx) {
  assert(Ctx && "null param passed");

  void *Ptr = II->getFETokenInfo<void>();
  if (!Ptr) return ctx_end(II);

  LookupContext LC(Ctx);

  if (isDeclPtr(Ptr)) {
    NamedDecl *D = static_cast<NamedDecl*>(Ptr);

    if (LC == LookupContext(D))
      return ctx_iterator(D);
    else
      return ctx_end(II);

  }
  
  IdDeclInfo *IDI = toIdDeclInfo(Ptr);
  IdDeclInfo::DeclsTy::iterator I = IDI->FindContext(LookupContext(Ctx));
  if (I != IDI->decls_begin() && LC != LookupContext(*(I-1)))
    I = IDI->decls_begin();

  return ctx_iterator(I);
}


/// Returns the IdDeclInfo associated to the IdentifierInfo.
/// It creates a new IdDeclInfo if one was not created before for this id.
IdentifierResolver::IdDeclInfo &
IdentifierResolver::IdDeclInfoMap::operator[](IdentifierInfo *II) {
  assert (II && "null IdentifierInfo passed");
  void *Ptr = II->getFETokenInfo<void>();

  if (Ptr) return *toIdDeclInfo(Ptr);

  if (CurIndex == VECTOR_SIZE) {
    // Add a IdDeclInfo vector 'pool'
    IDIVecs.push_back(std::vector<IdDeclInfo>());
    // Fill the vector
    IDIVecs.back().resize(VECTOR_SIZE);
    CurIndex = 0;
  }
  IdDeclInfo *IDI = &IDIVecs.back()[CurIndex];
  II->setFETokenInfo(reinterpret_cast<void*>(
                              reinterpret_cast<uintptr_t>(IDI) | 0x1)
                                                                     );
  ++CurIndex;
  return *IDI;
}
