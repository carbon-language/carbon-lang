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

/// begin - Returns an iterator for decls of identifier 'II', starting at
/// declaration context 'Ctx'. If 'LookInParentCtx' is true, it will walk the
/// decls of parent declaration contexts too.
IdentifierResolver::iterator
IdentifierResolver::begin(const IdentifierInfo *II, DeclContext *Ctx,
                          bool LookInParentCtx) {
  assert(Ctx && "null param passed");

  void *Ptr = II->getFETokenInfo<void>();
  if (!Ptr) return end();

  LookupContext LC(Ctx);

  if (isDeclPtr(Ptr)) {
    NamedDecl *D = static_cast<NamedDecl*>(Ptr);
    LookupContext DC(D);

    if (( LookInParentCtx && LC.isEqOrContainedBy(DC)) ||
        (!LookInParentCtx && LC == DC))
      return iterator(D);
    else
      return end();
  }

  IdDeclInfo *IDI = toIdDeclInfo(Ptr);

  IdDeclInfo::DeclsTy::iterator I;
  if (LookInParentCtx)
    I = IDI->FindContext(LC);
  else {
    for (I = IDI->decls_end(); I != IDI->decls_begin(); --I)
      if (LookupContext(*(I-1)) == LC)
        break;
  }

  if (I != IDI->decls_begin())
    return iterator(I-1, LookInParentCtx);
  else // No decls found.
    return end();
}

/// PreIncIter - Do a preincrement when 'Ptr' is a BaseIter.
void IdentifierResolver::iterator::PreIncIter() {
  NamedDecl *D = **this;
  LookupContext Ctx(D);
  void *InfoPtr = D->getIdentifier()->getFETokenInfo<void>();
  assert(!isDeclPtr(InfoPtr) && "Decl with wrong id ?");
  IdDeclInfo *Info = toIdDeclInfo(InfoPtr);

  BaseIter I = getIterator();
  if (LookInParentCtx())
    I = Info->FindContext(Ctx, I);
  else {
    if (I != Info->decls_begin() && LookupContext(*(I-1)) != Ctx) {
      // The next decl is in different declaration context.
      // Skip remaining decls and set the iterator to the end.
      I = Info->decls_begin();
    }
  }

  if (I != Info->decls_begin())
    *this = iterator(I-1, LookInParentCtx());
  else // No more decls.
    *this = end();
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
