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
// scoped lookup, based on declaration names.
//
//===----------------------------------------------------------------------===//

#include "IdentifierResolver.h"
#include "clang/Basic/LangOptions.h"
#include <list>
#include <vector>

using namespace clang;

//===----------------------------------------------------------------------===//
// IdDeclInfoMap class
//===----------------------------------------------------------------------===//

/// IdDeclInfoMap - Associates IdDeclInfos with declaration names.
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

  /// Returns the IdDeclInfo associated to the DeclarationName.
  /// It creates a new IdDeclInfo if one was not created before for this id.
  IdDeclInfo &operator[](DeclarationName Name);
};


//===----------------------------------------------------------------------===//
// IdDeclInfo Implementation
//===----------------------------------------------------------------------===//

/// RemoveDecl - Remove the decl from the scope chain.
/// The decl must already be part of the decl chain.
void IdentifierResolver::IdDeclInfo::RemoveDecl(NamedDecl *D) {
  for (DeclsTy::iterator I = Decls.end(); I != Decls.begin(); --I) {
    if (D == *(I-1)) {
      Decls.erase(I-1);
      return;
    }
  }

  assert(0 && "Didn't find this decl on its identifier's chain!");
}

bool
IdentifierResolver::IdDeclInfo::ReplaceDecl(NamedDecl *Old, NamedDecl *New) {
  for (DeclsTy::iterator I = Decls.end(); I != Decls.begin(); --I) {
    if (Old == *(I-1)) {
      *(I - 1) = New;
      return true;
    }
  }

  return false;
}


//===----------------------------------------------------------------------===//
// IdentifierResolver Implementation
//===----------------------------------------------------------------------===//

IdentifierResolver::IdentifierResolver(const LangOptions &langOpt)
    : LangOpt(langOpt), IdDeclInfos(new IdDeclInfoMap) {
}
IdentifierResolver::~IdentifierResolver() {
  delete IdDeclInfos;
}

/// isDeclInScope - If 'Ctx' is a function/method, isDeclInScope returns true
/// if 'D' is in Scope 'S', otherwise 'S' is ignored and isDeclInScope returns
/// true if 'D' belongs to the given declaration context.
bool IdentifierResolver::isDeclInScope(Decl *D, DeclContext *Ctx,
                                       ASTContext &Context, Scope *S) const {
  Ctx = Ctx->getLookupContext();

  if (Ctx->isFunctionOrMethod()) {
    // Ignore the scopes associated within transparent declaration contexts.
    while (S->getEntity() &&
           ((DeclContext *)S->getEntity())->isTransparentContext())
      S = S->getParent();

    if (S->isDeclScope(Action::DeclPtrTy::make(D)))
      return true;
    if (LangOpt.CPlusPlus) {
      // C++ 3.3.2p3:
      // The name declared in a catch exception-declaration is local to the
      // handler and shall not be redeclared in the outermost block of the
      // handler.
      // C++ 3.3.2p4:
      // Names declared in the for-init-statement, and in the condition of if,
      // while, for, and switch statements are local to the if, while, for, or
      // switch statement (including the controlled statement), and shall not be
      // redeclared in a subsequent condition of that statement nor in the
      // outermost block (or, for the if statement, any of the outermost blocks)
      // of the controlled statement.
      //
      assert(S->getParent() && "No TUScope?");
      if (S->getParent()->getFlags() & Scope::ControlScope)
        return S->getParent()->isDeclScope(Action::DeclPtrTy::make(D));
    }
    return false;
  }

  return D->getDeclContext()->getLookupContext()->Equals(Ctx);
}

/// AddDecl - Link the decl to its shadowed decl chain.
void IdentifierResolver::AddDecl(NamedDecl *D) {
  DeclarationName Name = D->getDeclName();
  void *Ptr = Name.getFETokenInfo<void>();

  if (!Ptr) {
    Name.setFETokenInfo(D);
    return;
  }

  IdDeclInfo *IDI;

  if (isDeclPtr(Ptr)) {
    Name.setFETokenInfo(NULL);
    IDI = &(*IdDeclInfos)[Name];
    NamedDecl *PrevD = static_cast<NamedDecl*>(Ptr);
    IDI->AddDecl(PrevD);
  } else
    IDI = toIdDeclInfo(Ptr);

  IDI->AddDecl(D);
}

/// RemoveDecl - Unlink the decl from its shadowed decl chain.
/// The decl must already be part of the decl chain.
void IdentifierResolver::RemoveDecl(NamedDecl *D) {
  assert(D && "null param passed");
  DeclarationName Name = D->getDeclName();
  void *Ptr = Name.getFETokenInfo<void>();

  assert(Ptr && "Didn't find this decl on its identifier's chain!");

  if (isDeclPtr(Ptr)) {
    assert(D == Ptr && "Didn't find this decl on its identifier's chain!");
    Name.setFETokenInfo(NULL);
    return;
  }

  return toIdDeclInfo(Ptr)->RemoveDecl(D);
}

bool IdentifierResolver::ReplaceDecl(NamedDecl *Old, NamedDecl *New) {
  assert(Old->getDeclName() == New->getDeclName() &&
         "Cannot replace a decl with another decl of a different name");

  DeclarationName Name = Old->getDeclName();
  void *Ptr = Name.getFETokenInfo<void>();

  if (!Ptr)
    return false;

  if (isDeclPtr(Ptr)) {
    if (Ptr == Old) {
      Name.setFETokenInfo(New);
      return true;
    }
    return false;
  }

  return toIdDeclInfo(Ptr)->ReplaceDecl(Old, New);
}

/// begin - Returns an iterator for decls with name 'Name'.
IdentifierResolver::iterator
IdentifierResolver::begin(DeclarationName Name) {
  void *Ptr = Name.getFETokenInfo<void>();
  if (!Ptr) return end();

  if (isDeclPtr(Ptr))
    return iterator(static_cast<NamedDecl*>(Ptr));

  IdDeclInfo *IDI = toIdDeclInfo(Ptr);

  IdDeclInfo::DeclsTy::iterator I = IDI->decls_end();
  if (I != IDI->decls_begin())
    return iterator(I-1);
  // No decls found.
  return end();
}

void IdentifierResolver::AddDeclToIdentifierChain(IdentifierInfo *II,
                                                  NamedDecl *D) {
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

//===----------------------------------------------------------------------===//
// IdDeclInfoMap Implementation
//===----------------------------------------------------------------------===//

/// Returns the IdDeclInfo associated to the DeclarationName.
/// It creates a new IdDeclInfo if one was not created before for this id.
IdentifierResolver::IdDeclInfo &
IdentifierResolver::IdDeclInfoMap::operator[](DeclarationName Name) {
  void *Ptr = Name.getFETokenInfo<void>();

  if (Ptr) return *toIdDeclInfo(Ptr);

  if (CurIndex == VECTOR_SIZE) {
    // Add a IdDeclInfo vector 'pool'
    IDIVecs.push_back(std::vector<IdDeclInfo>());
    // Fill the vector
    IDIVecs.back().resize(VECTOR_SIZE);
    CurIndex = 0;
  }
  IdDeclInfo *IDI = &IDIVecs.back()[CurIndex];
  Name.setFETokenInfo(reinterpret_cast<void*>(
                              reinterpret_cast<uintptr_t>(IDI) | 0x1)
                                                                     );
  ++CurIndex;
  return *IDI;
}
