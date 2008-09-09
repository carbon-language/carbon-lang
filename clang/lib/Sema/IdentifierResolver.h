//===- IdentifierResolver.h - Lexical Scope Name lookup ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the IdentifierResolver class, which is used for lexical
// scoped lookup, based on identifier.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_SEMA_IDENTIFIERRESOLVER_H
#define LLVM_CLANG_AST_SEMA_IDENTIFIERRESOLVER_H

#include "clang/Basic/IdentifierTable.h"
#include "clang/Parse/Scope.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"

namespace clang {

/// IdentifierResolver - Keeps track of shadowed decls on enclosing scopes.
/// It manages the shadowing chains of identifiers and implements efficent decl
/// lookup based on an identifier.
class IdentifierResolver {

  /// LookupContext - A wrapper for DeclContext. DeclContext is only part of
  /// ScopedDecls, LookupContext can be used with all decls (assumes
  /// translation unit context for non ScopedDecls).
  class LookupContext {
    DeclContext *Ctx;

    /// TUCtx - Provides a common value for translation unit context for all
    /// decls.
    /// FIXME: When (if ?) all decls can point to their translation unit context
    /// remove this hack.
    static inline DeclContext *TUCtx() {
      return reinterpret_cast<DeclContext*>(-1);
    }

    /// getContext - Returns translation unit context for non ScopedDecls and
    /// for EnumConstantDecls returns the parent context of their EnumDecl.
    static DeclContext *getContext(Decl *D);

  public:
    LookupContext(Decl *D) {
      Ctx = getContext(D);
    }
    LookupContext(DeclContext *DC) {
      if (!DC || isa<TranslationUnitDecl>(DC))
        Ctx = TUCtx();
      else
        Ctx = DC;
    }

    bool isTU() const {
      return (Ctx == TUCtx());
    }

    /// getParent - Returns the parent context. This should not be called for
    /// a translation unit context.
    LookupContext getParent() const {
      assert(!isTU() && "TU has no parent!");
      return LookupContext(Ctx->getParent());
    }

    /// isEqOrContainedBy - Returns true of the given context is the same or a
    /// parent of this one.
    bool isEqOrContainedBy(const LookupContext &PC) const;

    bool operator==(const LookupContext &RHS) const {
      return Ctx == RHS.Ctx;
    }
    bool operator!=(const LookupContext &RHS) const {
      return Ctx != RHS.Ctx;
    }
  };

  /// IdDeclInfo - Keeps track of information about decls associated to a
  /// particular identifier. IdDeclInfos are lazily constructed and assigned
  /// to an identifier the first time a decl with that identifier is shadowed
  /// in some scope.
  class IdDeclInfo {
  public:
    typedef llvm::SmallVector<NamedDecl*, 2> DeclsTy;

    inline DeclsTy::iterator decls_begin() { return Decls.begin(); }
    inline DeclsTy::iterator decls_end() { return Decls.end(); }

    /// FindContext - Returns an iterator pointing just after the decl that is
    /// in the given context or in a parent of it. The search is in reverse
    /// order, from end to begin.
    DeclsTy::iterator FindContext(const LookupContext &Ctx) {
      return FindContext(Ctx, Decls.end());
    }

    /// FindContext - Returns an iterator pointing just after the decl that is
    /// in the given context or in a parent of it. The search is in reverse
    /// order, from end to begin.
    DeclsTy::iterator FindContext(const LookupContext &Ctx,
                                  const DeclsTy::iterator &Start);

    void AddDecl(NamedDecl *D) {
      Decls.insert(FindContext(LookupContext(D)), D);
    }

    /// AddShadowed - Add a decl by putting it directly above the 'Shadow' decl.
    /// Later lookups will find the 'Shadow' decl first. The 'Shadow' decl must
    /// be already added to the scope chain and must be in the same context as
    /// the decl that we want to add.
    void AddShadowed(NamedDecl *D, NamedDecl *Shadow);

    /// RemoveDecl - Remove the decl from the scope chain.
    /// The decl must already be part of the decl chain.
    void RemoveDecl(NamedDecl *D);

  private:
    DeclsTy Decls;
  };

public:

  /// iterator - Iterate over the decls of a specified identifier.
  /// It will walk or not the parent declaration contexts depending on how
  /// it was instantiated.
  class iterator {
    /// Ptr - There are 3 forms that 'Ptr' represents:
    /// 1) A single NamedDecl. (Ptr & 0x1 == 0)
    /// 2) A IdDeclInfo::DeclsTy::iterator that traverses only the decls of the
    ///    same declaration context. (Ptr & 0x3 == 0x1)
    /// 3) A IdDeclInfo::DeclsTy::iterator that traverses the decls of parent
    ///    declaration contexts too. (Ptr & 0x3 == 0x3)
    uintptr_t Ptr;
    typedef IdDeclInfo::DeclsTy::iterator BaseIter;

    /// A single NamedDecl. (Ptr & 0x1 == 0)
    iterator(NamedDecl *D) {
      Ptr = reinterpret_cast<uintptr_t>(D);
      assert((Ptr & 0x1) == 0 && "Invalid Ptr!");
    }
    /// A IdDeclInfo::DeclsTy::iterator that walks or not the parent declaration
    /// contexts depending on 'LookInParentCtx'.
    iterator(BaseIter I, bool LookInParentCtx) {
      Ptr = reinterpret_cast<uintptr_t>(I) | 0x1;
      assert((Ptr & 0x2) == 0 && "Invalid Ptr!");
      if (LookInParentCtx) Ptr |= 0x2;
    }

    bool isIterator() const { return (Ptr & 0x1); }

    bool LookInParentCtx() const {
      assert(isIterator() && "Ptr not an iterator!");
      return (Ptr & 0x2) != 0;
    }

    BaseIter getIterator() const {
      assert(isIterator() && "Ptr not an iterator!");
      return reinterpret_cast<BaseIter>(Ptr & ~0x3);
    }
    
    friend class IdentifierResolver;
  public:
    iterator() : Ptr(0) {}

    NamedDecl *operator*() const {
      if (isIterator())
        return *getIterator();
      else
        return reinterpret_cast<NamedDecl*>(Ptr);
    }
    
    bool operator==(const iterator &RHS) const {
      return Ptr == RHS.Ptr;
    }
    bool operator!=(const iterator &RHS) const {
      return Ptr != RHS.Ptr;
    }
    
    // Preincrement.
    iterator& operator++() {
      if (!isIterator()) // common case.
        Ptr = 0;
      else
        PreIncIter();
      return *this;
    }

  private:
    void PreIncIter();
  };

  /// begin - Returns an iterator for decls of identifier 'II', starting at
  /// declaration context 'Ctx'. If 'LookInParentCtx' is true, it will walk the
  /// decls of parent declaration contexts too.
  /// Default for 'LookInParentCtx is true.
  static iterator begin(const IdentifierInfo *II, DeclContext *Ctx,
                        bool LookInParentCtx = true);

  /// end - Returns an iterator that has 'finished'.
  static iterator end() {
    return iterator();
  }

  /// isDeclInScope - If 'Ctx' is a function/method, isDeclInScope returns true
  /// if 'D' is in Scope 'S', otherwise 'S' is ignored and isDeclInScope returns
  /// true if 'D' belongs to the given declaration context.
  bool isDeclInScope(Decl *D, DeclContext *Ctx, Scope *S = 0) const;

  /// AddDecl - Link the decl to its shadowed decl chain.
  void AddDecl(NamedDecl *D);

  /// AddShadowedDecl - Link the decl to its shadowed decl chain putting it
  /// after the decl that the iterator points to, thus the 'Shadow' decl will be
  /// encountered before the 'D' decl.
  void AddShadowedDecl(NamedDecl *D, NamedDecl *Shadow);

  /// RemoveDecl - Unlink the decl from its shadowed decl chain.
  /// The decl must already be part of the decl chain.
  void RemoveDecl(NamedDecl *D);

  explicit IdentifierResolver(const LangOptions &LangOpt);
  ~IdentifierResolver();

private:
  const LangOptions &LangOpt;

  class IdDeclInfoMap;
  IdDeclInfoMap *IdDeclInfos;

  /// Identifier's FETokenInfo contains a Decl pointer if lower bit == 0.
  static inline bool isDeclPtr(void *Ptr) {
    return (reinterpret_cast<uintptr_t>(Ptr) & 0x1) == 0;
  }

  /// Identifier's FETokenInfo contains a IdDeclInfo pointer if lower bit == 1.
  static inline IdDeclInfo *toIdDeclInfo(void *Ptr) {
    assert((reinterpret_cast<uintptr_t>(Ptr) & 0x1) == 1
          && "Ptr not a IdDeclInfo* !");
    return reinterpret_cast<IdDeclInfo*>(
                    reinterpret_cast<uintptr_t>(Ptr) & ~0x1
                                                            );
  }
};

} // end namespace clang

#endif
