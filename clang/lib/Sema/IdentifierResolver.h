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
    static DeclContext *getContext(Decl *D) {
      DeclContext *Ctx;

      if (CXXFieldDecl *FD = dyn_cast<CXXFieldDecl>(D))
        return FD->getParent();

      if (EnumConstantDecl *EnumD = dyn_cast<EnumConstantDecl>(D)) {
        Ctx = EnumD->getDeclContext()->getParent();
      } else if (ScopedDecl *SD = dyn_cast<ScopedDecl>(D))
        Ctx = SD->getDeclContext();
      else
        return TUCtx();

      if (isa<TranslationUnitDecl>(Ctx))
        return TUCtx();

      return Ctx;
    }

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
    bool isEqOrContainedBy(const LookupContext &PC) const {
      if (PC.isTU()) return true;

      for (LookupContext Next = *this; !Next.isTU();  Next = Next.getParent())
        if (Next.Ctx == PC.Ctx) return true;

      return false;
    }

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
                                  const DeclsTy::iterator &Start) {
      for (DeclsTy::iterator I = Start; I != Decls.begin(); --I) {
        if (Ctx.isEqOrContainedBy(LookupContext(*(I-1))))
          return I;
      }

      return Decls.begin();
    }

    /// iterator - Iterate over the decls by walking their parent contexts too.
    class iterator {
    public:
      typedef DeclsTy::iterator BaseIter;

      iterator(const BaseIter &DeclIt) : DI(DeclIt) {}
      const BaseIter &getBase() { return DI; }

      NamedDecl *&operator*() const {
        return *(DI-1);
      }
      
      bool operator==(const iterator &RHS) const {
        return DI == RHS.DI;
      }
      bool operator!=(const iterator &RHS) const {
        return DI != RHS.DI;
      }
      
      // Preincrement.
      iterator& operator++() {
        NamedDecl *D = **this;
        void *Ptr = D->getIdentifier()->getFETokenInfo<void>();
        assert(!isDeclPtr(Ptr) && "Decl with wrong id ?");
        DI = toIdDeclInfo(Ptr)->FindContext(LookupContext(D), DI-1);
        return *this;
      }

    private:
      BaseIter DI;
    };

    /// ctx_iterator - Iterator over the decls of a specific context only.
    class ctx_iterator {
    public:
      typedef DeclsTy::iterator BaseIter;

      ctx_iterator(const BaseIter &DeclIt) : DI(DeclIt) {}
      const BaseIter &getBase() { return DI; }

      NamedDecl *&operator*() const {
        return *(DI-1);
      }
      
      bool operator==(const ctx_iterator &RHS) const {
        return DI == RHS.DI;
      }
      bool operator!=(const ctx_iterator &RHS) const {
        return DI != RHS.DI;
      }
      
      // Preincrement.
      ctx_iterator& operator++() {
        NamedDecl *D = **this;
        void *Ptr = D->getIdentifier()->getFETokenInfo<void>();
        assert(!isDeclPtr(Ptr) && "Decl with wrong id ?");
        IdDeclInfo *Info = toIdDeclInfo(Ptr);
        
        --DI;
        if (DI != Info->Decls.begin() &&
            LookupContext(D) != LookupContext(**this))
          DI = Info->Decls.begin();
        return *this;
      }

    private:
      BaseIter DI;
    };

    void AddDecl(NamedDecl *D) {
      Decls.insert(FindContext(LookupContext(D)), D);
    }

    /// AddShadowed - Add a decl by putting it directly above the 'Shadow' decl.
    /// Later lookups will find the 'Shadow' decl first. The 'Shadow' decl must
    /// be already added to the scope chain and must be in the same context as
    /// the decl that we want to add.
    void AddShadowed(NamedDecl *D, NamedDecl *Shadow) {
      assert(LookupContext(D) == LookupContext(Shadow) &&
             "Decl and Shadow not in same context!");

      for (DeclsTy::iterator I = Decls.end(); I != Decls.begin(); --I) {
        if (Shadow == *(I-1)) {
          Decls.insert(I-1, D);
          return;
        }
      }

      assert(0 && "Shadow wasn't in scope chain!");
    }

    /// RemoveDecl - Remove the decl from the scope chain.
    /// The decl must already be part of the decl chain.
    void RemoveDecl(NamedDecl *D) {
      for (DeclsTy::iterator I = Decls.end(); I != Decls.begin(); --I) {
        if (D == *(I-1)) {
          Decls.erase(I-1);
          return;
        }
      }

      assert(0 && "Didn't find this decl on its identifier's chain!");
    }

  private:
    DeclsTy Decls;
  };

  /// SwizzledIterator - Can be instantiated either with a single NamedDecl*
  /// (the common case where only one decl is associated with an identifier) or
  /// with an 'Iter' iterator, when there are more than one decls to lookup.
  template<typename Iter>
  class SwizzledIterator {
    uintptr_t Ptr;

    SwizzledIterator() : Ptr(0) {}
    SwizzledIterator(NamedDecl *D) {
      Ptr = reinterpret_cast<uintptr_t>(D);
    }
    SwizzledIterator(Iter I) {
      Ptr = reinterpret_cast<uintptr_t>(I.getBase()) | 0x1;
    }

    bool isIterator() const { return (Ptr & 0x1); }

    Iter getIterator() const {
      assert(isIterator() && "Ptr not an iterator.");
      return reinterpret_cast<typename Iter::BaseIter>(Ptr & ~0x1);
    }

    friend class IdentifierResolver;
  public:
    NamedDecl *operator*() const {
      if (isIterator())
        return *getIterator();
      else
        return reinterpret_cast<NamedDecl*>(Ptr);
    }
    
    bool operator==(const SwizzledIterator &RHS) const {
      return Ptr == RHS.Ptr;
    }
    bool operator!=(const SwizzledIterator &RHS) const {
      return Ptr != RHS.Ptr;
    }

    // Preincrement.
    SwizzledIterator& operator++() {
      if (isIterator()) {
        Iter I = getIterator();
        ++I;
        Ptr = reinterpret_cast<uintptr_t>(I.getBase()) | 0x1;
      }
      else  // This is a single NamedDecl*.
        Ptr = 0;

      return *this;
    }
  };

public:

  typedef SwizzledIterator<IdDeclInfo::iterator> iterator;
  typedef SwizzledIterator<IdDeclInfo::ctx_iterator> ctx_iterator;

  /// begin - Returns an iterator for all decls, starting at the given
  /// declaration context.
  static iterator begin(const IdentifierInfo *II, DeclContext *Ctx);

  static iterator end(const IdentifierInfo *II) {
    void *Ptr = II->getFETokenInfo<void>();
    if (!Ptr || isDeclPtr(Ptr))
      return iterator();

    IdDeclInfo *IDI = toIdDeclInfo(Ptr);
    return iterator(IDI->decls_begin());
  }

  /// ctx_begin - Returns an iterator for only decls that belong to the given
  /// declaration context.
  static ctx_iterator ctx_begin(const IdentifierInfo *II, DeclContext *Ctx);

  static ctx_iterator ctx_end(const IdentifierInfo *II) {
    void *Ptr = II->getFETokenInfo<void>();
    if (!Ptr || isDeclPtr(Ptr))
      return ctx_iterator();

    IdDeclInfo *IDI = toIdDeclInfo(Ptr);
    return ctx_iterator(IDI->decls_begin());
  }

  /// isDeclInScope - If 'Ctx' is a function/method, isDeclInScope returns true
  /// if 'D' is in Scope 'S', otherwise 'S' is ignored and isDeclInScope returns
  /// true if 'D' belongs to the given declaration context.
  static bool isDeclInScope(Decl *D, DeclContext *Ctx, Scope *S = 0) {
    if (Ctx->isFunctionOrMethod())
      return S->isDeclScope(D);

    return LookupContext(D) == LookupContext(Ctx);
  }

  /// AddDecl - Link the decl to its shadowed decl chain.
  void AddDecl(NamedDecl *D);

  /// AddShadowedDecl - Link the decl to its shadowed decl chain putting it
  /// after the decl that the iterator points to, thus the 'Shadow' decl will be
  /// encountered before the 'D' decl.
  void AddShadowedDecl(NamedDecl *D, NamedDecl *Shadow);

  /// RemoveDecl - Unlink the decl from its shadowed decl chain.
  /// The decl must already be part of the decl chain.
  void RemoveDecl(NamedDecl *D);

  IdentifierResolver();
  ~IdentifierResolver();

private:
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
