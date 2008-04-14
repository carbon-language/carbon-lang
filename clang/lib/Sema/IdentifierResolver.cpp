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
#include "clang/Basic/IdentifierTable.h"
#include "clang/AST/Decl.h"
#include "clang/Parse/Scope.h"
#include <list>
#include <vector>

using namespace clang;

namespace {

class IdDeclInfo;

/// Identifier's FETokenInfo contains a Decl pointer if lower bit == 0.
static inline bool isDeclPtr(void *Ptr) {
  return (reinterpret_cast<uintptr_t>(Ptr) & 0x1) == 0;
}

/// Identifier's FETokenInfo contains a IdDeclInfo pointer if lower bit == 1.
static inline IdDeclInfo *toIdDeclInfo(void *Ptr) {
  return reinterpret_cast<IdDeclInfo*>(
                  reinterpret_cast<uintptr_t>(Ptr) & ~0x1
                                                          );
}


/// IdDeclInfo - Keeps track of information about decls associated to a
/// particular identifier. IdDeclInfos are lazily constructed and assigned
/// to an identifier the first time a decl with that identifier is shadowed
/// in some scope.
class IdDeclInfo {
  typedef llvm::SmallVector<NamedDecl *, 2> ShadowedTy;
  ShadowedTy ShadowedDecls;

public:
  typedef ShadowedTy::iterator ShadowedIter;

  inline ShadowedIter shadowed_begin() { return ShadowedDecls.begin(); }
  inline ShadowedIter shadowed_end() { return ShadowedDecls.end(); }

  /// Add a decl in the scope chain.
  void PushShadowed(NamedDecl *D) {
    assert(D && "Decl null");
    ShadowedDecls.push_back(D);
  }

  /// Add the decl at the top of scope chain.
  void PushGlobalShadowed(NamedDecl *D) {
    assert(D && "Decl null");
    ShadowedDecls.insert(ShadowedDecls.begin(), D);
  }

  /// RemoveShadowed - Remove the decl from the scope chain.
  /// The decl must already be part of the decl chain.
  void RemoveShadowed(NamedDecl *D);
};


/// IdDeclInfoMap - Associates IdDeclInfos with Identifiers.
/// Allocates 'pools' (vectors of IdDeclInfos) to avoid allocating each
/// individual IdDeclInfo to heap.
class IdDeclInfoMap {
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

} // end anonymous namespace


IdentifierResolver::IdentifierResolver() : IdDeclInfos(new IdDeclInfoMap) {}
IdentifierResolver::~IdentifierResolver() {
  delete static_cast<IdDeclInfoMap*>(IdDeclInfos);
}

/// AddDecl - Link the decl to its shadowed decl chain.
void IdentifierResolver::AddDecl(NamedDecl *D, Scope *S) {
  assert(D && S && "null param passed");
  IdentifierInfo *II = D->getIdentifier();
  void *Ptr = II->getFETokenInfo<void>();

  if (!Ptr) {
    II->setFETokenInfo(D);
    return;
  }

  IdDeclInfo *IDI;

  if (isDeclPtr(Ptr)) {
    II->setFETokenInfo(NULL);
    IdDeclInfoMap &Map = *static_cast<IdDeclInfoMap*>(IdDeclInfos);
    IDI = &Map[II];
    IDI->PushShadowed(static_cast<NamedDecl*>(Ptr));
  } else
    IDI = toIdDeclInfo(Ptr);

  // C++ [basic.scope]p4:
  //   -- exactly one declaration shall declare a class name or
  //   enumeration name that is not a typedef name and the other
  //   declarations shall all refer to the same object or
  //   enumerator, or all refer to functions and function templates;
  //   in this case the class name or enumeration name is hidden.
  if (isa<TagDecl>(D)) {
    // We are pushing the name of a tag (enum or class).
    IdDeclInfo::ShadowedIter TopIter = IDI->shadowed_end() - 1;
    if (S->isDeclScope(*TopIter)) {
      // There is already a declaration with the same name in the same
      // scope. It must be found before we find the new declaration,
      // so swap the order on the shadowed declaration stack.
      NamedDecl *Temp = *TopIter;
      *TopIter = D;
      D = Temp;
    }
  }

  IDI->PushShadowed(D);
}

/// AddGlobalDecl - Link the decl at the top of the shadowed decl chain.
void IdentifierResolver::AddGlobalDecl(NamedDecl *D) {
  assert(D && "null param passed");
  IdentifierInfo *II = D->getIdentifier();
  void *Ptr = II->getFETokenInfo<void>();

  if (!Ptr) {
    II->setFETokenInfo(D);
    return;
  }

  IdDeclInfo *IDI;

  if (isDeclPtr(Ptr)) {
    II->setFETokenInfo(NULL);
    IdDeclInfoMap &Map = *static_cast<IdDeclInfoMap*>(IdDeclInfos);
    IDI = &Map[II];
    IDI->PushShadowed(static_cast<NamedDecl*>(Ptr));
  } else
    IDI = toIdDeclInfo(Ptr);

  IDI->PushGlobalShadowed(D);
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
  
  return toIdDeclInfo(Ptr)->RemoveShadowed(D);
}

/// Lookup - Find the non-shadowed decl that belongs to a particular
/// Decl::IdentifierNamespace.
NamedDecl *IdentifierResolver::Lookup(const IdentifierInfo *II, unsigned NS) {
  assert(II && "null param passed");
  void *Ptr = II->getFETokenInfo<void>();

  if (!Ptr) return NULL;

  if (isDeclPtr(Ptr)) {
    NamedDecl *D = static_cast<NamedDecl*>(Ptr);
    return (D->getIdentifierNamespace() & NS) ? D : NULL;
  }

  IdDeclInfo *IDI = toIdDeclInfo(Ptr);

  // ShadowedDecls are ordered from most shadowed to less shadowed.
  // So we do a reverse iteration from end to begin.
  for (IdDeclInfo::ShadowedIter SI = IDI->shadowed_end();
       SI != IDI->shadowed_begin(); --SI) {
    NamedDecl *D = *(SI-1);
    if (D->getIdentifierNamespace() & NS)
      return D;
  }

  // we didn't find the decl.
  return NULL;
}

/// RemoveShadowed - Remove the decl from the scope chain.
/// The decl must already be part of the decl chain.
void IdDeclInfo::RemoveShadowed(NamedDecl *D) {
  assert(D && "null decl passed");
  assert(!ShadowedDecls.empty() &&
         "Didn't find this decl on its identifier's chain!");

  // common case
  if (D == ShadowedDecls.back()) {
    ShadowedDecls.pop_back();
    return;
  }

  for (ShadowedIter SI = ShadowedDecls.end()-1;
       SI != ShadowedDecls.begin(); --SI) {
    if (*(SI-1) == D) {
      ShadowedDecls.erase(SI-1);
      return;
    }
  }

  assert(false && "Didn't find this decl on its identifier's chain!");
}

/// Returns the IdDeclInfo associated to the IdentifierInfo.
/// It creates a new IdDeclInfo if one was not created before for this id.
IdDeclInfo &IdDeclInfoMap::operator[](IdentifierInfo *II) {
  assert (II && "null IdentifierInfo passed");
  void *Ptr = II->getFETokenInfo<void>();

  if (Ptr) {
    assert(!isDeclPtr(Ptr) && "didn't clear decl for FEToken");
    return *toIdDeclInfo(Ptr);
  }

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
