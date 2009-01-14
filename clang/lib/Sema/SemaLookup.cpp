//===--------------------- SemaLookup.cpp - Name Lookup  ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements name lookup for C, C++, Objective-C, and
//  Objective-C++.
//
//===----------------------------------------------------------------------===//
#include "Sema.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/STLExtras.h"

using namespace clang;

/// MaybeConstructOverloadSet - Name lookup has determined that the
/// elements in [I, IEnd) have the name that we are looking for, and
/// *I is a match for the namespace. This routine returns an
/// appropriate Decl for name lookup, which may either be *I or an
/// OverloadeFunctionDecl that represents the overloaded functions in
/// [I, IEnd). 
///
/// The existance of this routine is temporary; LookupDecl should
/// probably be able to return multiple results, to deal with cases of
/// ambiguity and overloaded functions without needing to create a
/// Decl node.
template<typename DeclIterator>
static Decl *
MaybeConstructOverloadSet(ASTContext &Context, 
                          DeclIterator I, DeclIterator IEnd) {
  assert(I != IEnd && "Iterator range cannot be empty");
  assert(!isa<OverloadedFunctionDecl>(*I) && 
         "Cannot have an overloaded function");

  if (isa<FunctionDecl>(*I)) {
    // If we found a function, there might be more functions. If
    // so, collect them into an overload set.
    DeclIterator Last = I;
    OverloadedFunctionDecl *Ovl = 0;
    for (++Last; Last != IEnd && isa<FunctionDecl>(*Last); ++Last) {
      if (!Ovl) {
        // FIXME: We leak this overload set. Eventually, we want to
        // stop building the declarations for these overload sets, so
        // there will be nothing to leak.
        Ovl = OverloadedFunctionDecl::Create(Context, 
                                         cast<ScopedDecl>(*I)->getDeclContext(),
                                             (*I)->getDeclName());
        Ovl->addOverload(cast<FunctionDecl>(*I));
      }
      Ovl->addOverload(cast<FunctionDecl>(*Last));
    }
    
    // If we had more than one function, we built an overload
    // set. Return it.
    if (Ovl)
      return Ovl;
  }
  
  return *I;
}

/// @brief Constructs name lookup criteria.
///
/// @param K The kind of name that we're searching for.
///
/// @param RedeclarationOnly If true, then name lookup will only look
/// into the current scope for names, not in parent scopes. This
/// option should be set when we're looking to introduce a new
/// declaration into scope.
///
/// @param CPlusPlus Whether we are performing C++ name lookup or not.
Sema::LookupCriteria::LookupCriteria(NameKind K, bool RedeclarationOnly,
                                     bool CPlusPlus)  
  : Kind(K), AllowLazyBuiltinCreation(K == Ordinary), 
    RedeclarationOnly(RedeclarationOnly) { 
  switch (Kind) {
  case Ordinary:
    IDNS = Decl::IDNS_Ordinary;
    if (CPlusPlus)
      IDNS |= Decl::IDNS_Tag | Decl::IDNS_Member;
    break;

  case Tag:
    IDNS = Decl::IDNS_Tag;
    break;

  case Member:
    IDNS = Decl::IDNS_Member;
    if (CPlusPlus)
      IDNS |= Decl::IDNS_Tag | Decl::IDNS_Ordinary;    
    break;

  case NestedNameSpecifier:
  case Namespace:
    IDNS = Decl::IDNS_Ordinary | Decl::IDNS_Tag | Decl::IDNS_Member;
    break;
  }
}

/// isLookupResult - Determines whether D is a suitable lookup result
/// according to the lookup criteria.
bool Sema::LookupCriteria::isLookupResult(Decl *D) const {
  switch (Kind) {
  case Ordinary:
  case Tag:
  case Member:
    return D->isInIdentifierNamespace(IDNS);

  case NestedNameSpecifier:
    return isa<TypedefDecl>(D) || D->isInIdentifierNamespace(Decl::IDNS_Tag);

  case Namespace:
    return isa<NamespaceDecl>(D);
  }

  assert(false && "isLookupResult always returns before this point");
  return false;
}

/// @brief Determine the result of name lookup.
Sema::LookupResult::LookupKind Sema::LookupResult::getKind() const {
  switch (StoredKind) {
  case SingleDecl:
    return (reinterpret_cast<Decl *>(First) != 0)? Found : NotFound;

  case OverloadedDeclFromIdResolver:
  case OverloadedDeclFromDeclContext:
    return FoundOverloaded;

  case AmbiguousLookup:
    return Ambiguous;
  }

  // We can't get here, but GCC complains nonetheless.
  return Ambiguous;
}

/// @brief Converts the result of name lookup into a single (possible
/// NULL) pointer to a declaration.
///
/// The resulting declaration will either be the declaration we found
/// (if only a single declaration was found), an
/// OverloadedFunctionDecl (if an overloaded function was found), or
/// NULL (if no declaration was found). This conversion must not be
/// used anywhere where name lookup could result in an ambiguity. 
///
/// The OverloadedFunctionDecl conversion is meant as a stop-gap
/// solution, since it causes the OverloadedFunctionDecl to be
/// leaked. FIXME: Eventually, there will be a better way to iterate
/// over the set of overloaded functions returned by name lookup.
Decl *Sema::LookupResult::getAsDecl() const {
  switch (StoredKind) {
  case SingleDecl:
    return reinterpret_cast<Decl *>(First);

  case OverloadedDeclFromIdResolver:
    return MaybeConstructOverloadSet(*Context,
                         IdentifierResolver::iterator::getFromOpaqueValue(First),
                         IdentifierResolver::iterator::getFromOpaqueValue(Last));

  case OverloadedDeclFromDeclContext:
    return MaybeConstructOverloadSet(*Context, 
                           reinterpret_cast<DeclContext::lookup_iterator>(First),
                           reinterpret_cast<DeclContext::lookup_iterator>(Last));

  case AmbiguousLookup:
    assert(false && 
           "Name lookup returned an ambiguity that could not be handled");
    break;
  }

  return 0;
}

/// @brief Perform unqualified name lookup starting from a given
/// scope.
///
/// Unqualified name lookup (C++ [basic.lookup.unqual], C99 6.2.1) is
/// used to find names within the current scope. For example, 'x' in
/// @code
/// int x;
/// int f() {
///   return x; // unqualified name look finds 'x' in the global scope
/// }
/// @endcode
///
/// Different lookup criteria can find different names. For example, a
/// particular scope can have both a struct and a function of the same
/// name, and each can be found by certain lookup criteria. For more
/// information about lookup criteria, see the documentation for the
/// class LookupCriteria.
///
/// @param S        The scope from which unqualified name lookup will
/// begin. If the lookup criteria permits, name lookup may also search
/// in the parent scopes.
///
/// @param Name     The name of the entity that we are searching for.
///
/// @param Criteria The criteria that this routine will use to
/// determine which names are visible and which names will be
/// found. Note that name lookup will find a name that is visible by
/// the given criteria, but the entity itself may not be semantically
/// correct or even the kind of entity expected based on the
/// lookup. For example, searching for a nested-name-specifier name
/// might result in an EnumDecl, which is visible but is not permitted
/// as a nested-name-specifier in C++03.
///
/// @returns The result of name lookup, which includes zero or more
/// declarations and possibly additional information used to diagnose
/// ambiguities.
Sema::LookupResult 
Sema::LookupName(Scope *S, DeclarationName Name, LookupCriteria Criteria) {
  if (!Name) return LookupResult(Context, 0);

  if (!getLangOptions().CPlusPlus) {
    // Unqualified name lookup in C/Objective-C is purely lexical, so
    // search in the declarations attached to the name.

    // For the purposes of unqualified name lookup, structs and unions
    // don't have scopes at all. For example:
    //
    //   struct X {
    //     struct T { int i; } x;
    //   };
    //
    //   void f() {
    //     struct T t; // okay: T is defined lexically within X, but
    //                 // semantically at global scope
    //   };
    //
    // FIXME: Is there a better way to deal with this?
    DeclContext *SearchCtx = CurContext;
    while (isa<RecordDecl>(SearchCtx) || isa<EnumDecl>(SearchCtx))
      SearchCtx = SearchCtx->getParent();
    IdentifierResolver::iterator I
      = IdResolver.begin(Name, SearchCtx, !Criteria.RedeclarationOnly);
    
    // Scan up the scope chain looking for a decl that matches this
    // identifier that is in the appropriate namespace.  This search
    // should not take long, as shadowing of names is uncommon, and
    // deep shadowing is extremely uncommon.
    for (; I != IdResolver.end(); ++I)
      if (Criteria.isLookupResult(*I))
        return LookupResult(Context, *I);
  } else {
    // Unqualified name lookup in C++ requires looking into scopes
    // that aren't strictly lexical, and therefore we walk through the
    // context as well as walking through the scopes.

    // FIXME: does "true" for LookInParentCtx actually make sense?
    IdentifierResolver::iterator 
      I = IdResolver.begin(Name, CurContext, true/*LookInParentCtx*/),
      IEnd = IdResolver.end();
    for (; S; S = S->getParent()) {
      // Check whether the IdResolver has anything in this scope.
      for (; I != IEnd && S->isDeclScope(*I); ++I) {
        if (Criteria.isLookupResult(*I)) {
          // We found something.  Look for anything else in our scope
          // with this same name and in an acceptable identifier
          // namespace, so that we can construct an overload set if we
          // need to.
          IdentifierResolver::iterator LastI = I;
          for (++LastI; LastI != IEnd; ++LastI) {
            if (!S->isDeclScope(*LastI))
              break;
          }
          return LookupResult(Context, I, LastI);
        }
      }
      
      // If there is an entity associated with this scope, it's a
      // DeclContext. We might need to perform qualified lookup into
      // it.
      // FIXME: We're performing redundant lookups here, where the
      // scope stack mirrors the semantic nested of classes and
      // namespaces. We can save some work by checking the lexical
      // scope against the semantic scope and avoiding any lookups
      // when they are the same.
      // FIXME: In some cases, we know that every name that could be
      // found by this qualified name lookup will also be on the
      // identifier chain. For example, inside a class without any
      // base classes, we never need to perform qualified lookup
      // because all of the members are on top of the identifier
      // chain. However, we cannot perform this optimization when the
      // lexical and semantic scopes don't line up, e.g., in an
      // out-of-line member definition.
      DeclContext *Ctx = static_cast<DeclContext *>(S->getEntity());
      while (Ctx && Ctx->isFunctionOrMethod())
        Ctx = Ctx->getParent();
      while (Ctx && (Ctx->isNamespace() || Ctx->isRecord())) {
        // Look for declarations of this name in this scope.
        if (LookupResult Result = LookupQualifiedName(Ctx, Name, Criteria))
          return Result;
        
        if (Criteria.RedeclarationOnly && !Ctx->isTransparentContext())
          return LookupResult(Context, 0);

        Ctx = Ctx->getParent();
      }
    }
  }

  // If we didn't find a use of this identifier, and if the identifier
  // corresponds to a compiler builtin, create the decl object for the builtin
  // now, injecting it into translation unit scope, and return it.
  if (Criteria.Kind == LookupCriteria::Ordinary) {
    IdentifierInfo *II = Name.getAsIdentifierInfo();
    if (Criteria.AllowLazyBuiltinCreation && II) {
      // If this is a builtin on this (or all) targets, create the decl.
      if (unsigned BuiltinID = II->getBuiltinID())
        return LookupResult(Context,
                            LazilyCreateBuiltin((IdentifierInfo *)II, BuiltinID,
                                                S));
    }
    if (getLangOptions().ObjC1 && II) {
      // @interface and @compatibility_alias introduce typedef-like names.
      // Unlike typedef's, they can only be introduced at file-scope (and are 
      // therefore not scoped decls). They can, however, be shadowed by
      // other names in IDNS_Ordinary.
      ObjCInterfaceDeclsTy::iterator IDI = ObjCInterfaceDecls.find(II);
      if (IDI != ObjCInterfaceDecls.end())
        return LookupResult(Context, IDI->second);
      ObjCAliasTy::iterator I = ObjCAliasDecls.find(II);
      if (I != ObjCAliasDecls.end())
        return LookupResult(Context, I->second->getClassInterface());
    }
  }
  return LookupResult(Context, 0);
}

/// @brief Perform qualified name lookup into a given context.
///
/// Qualified name lookup (C++ [basic.lookup.qual]) is used to find
/// names when the context of those names is explicit specified, e.g.,
/// "std::vector" or "x->member".
///
/// Different lookup criteria can find different names. For example, a
/// particular scope can have both a struct and a function of the same
/// name, and each can be found by certain lookup criteria. For more
/// information about lookup criteria, see the documentation for the
/// class LookupCriteria.
///
/// @param LookupCtx The context in which qualified name lookup will
/// search. If the lookup criteria permits, name lookup may also search
/// in the parent contexts or (for C++ classes) base classes.
///
/// @param Name     The name of the entity that we are searching for.
///
/// @param Criteria The criteria that this routine will use to
/// determine which names are visible and which names will be
/// found. Note that name lookup will find a name that is visible by
/// the given criteria, but the entity itself may not be semantically
/// correct or even the kind of entity expected based on the
/// lookup. For example, searching for a nested-name-specifier name
/// might result in an EnumDecl, which is visible but is not permitted
/// as a nested-name-specifier in C++03.
///
/// @returns The result of name lookup, which includes zero or more
/// declarations and possibly additional information used to diagnose
/// ambiguities.
Sema::LookupResult
Sema::LookupQualifiedName(DeclContext *LookupCtx, DeclarationName Name,
                          LookupCriteria Criteria) {
  assert(LookupCtx && "Sema::LookupQualifiedName requires a lookup context");
  
  if (!Name) return LookupResult(Context, 0);

  // If we're performing qualified name lookup (e.g., lookup into a
  // struct), find fields as part of ordinary name lookup.
  if (Criteria.Kind == LookupCriteria::Ordinary)
    Criteria.IDNS |= Decl::IDNS_Member;

  // Perform qualified name lookup into the LookupCtx.
  // FIXME: Will need to look into base classes and such.
  DeclContext::lookup_iterator I, E;
  for (llvm::tie(I, E) = LookupCtx->lookup(Name); I != E; ++I)
    if (Criteria.isLookupResult(*I))
      return LookupResult(Context, I, E);

  return LookupResult(Context, 0);
}

/// @brief Performs name lookup for a name that was parsed in the
/// source code, and may contain a C++ scope specifier.
///
/// This routine is a convenience routine meant to be called from
/// contexts that receive a name and an optional C++ scope specifier
/// (e.g., "N::M::x"). It will then perform either qualified or
/// unqualified name lookup (with LookupQualifiedName or LookupName,
/// respectively) on the given name and return those results.
///
/// @param S        The scope from which unqualified name lookup will
/// begin.
/// 
/// @param SS       An optional C++ scope-specified, e.g., "::N::M".
///
/// @param Name     The name of the entity that name lookup will
/// search for.
///
/// @param Criteria The criteria that will determine which entities
/// are visible to name lookup.
///
/// @returns The result of qualified or unqualified name lookup.
Sema::LookupResult
Sema::LookupParsedName(Scope *S, const CXXScopeSpec &SS, 
                       DeclarationName Name, LookupCriteria Criteria) {
  if (SS.isSet())
    return LookupQualifiedName(static_cast<DeclContext *>(SS.getScopeRep()),
                               Name, Criteria);

  return LookupName(S, Name, Criteria);
}


