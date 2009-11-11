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
#include "clang/AST/ASTContext.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/ErrorHandling.h"
#include <set>
#include <vector>
#include <iterator>
#include <utility>
#include <algorithm>

using namespace clang;

namespace {
  class UnqualUsingEntry {
    const DeclContext *Nominated;
    const DeclContext *CommonAncestor;

  public:
    UnqualUsingEntry(const DeclContext *Nominated,
                     const DeclContext *CommonAncestor)
      : Nominated(Nominated), CommonAncestor(CommonAncestor) {
    }

    const DeclContext *getCommonAncestor() const {
      return CommonAncestor;
    }

    const DeclContext *getNominatedNamespace() const {
      return Nominated;
    }

    // Sort by the pointer value of the common ancestor.
    struct Comparator {
      bool operator()(const UnqualUsingEntry &L, const UnqualUsingEntry &R) {
        return L.getCommonAncestor() < R.getCommonAncestor();
      }

      bool operator()(const UnqualUsingEntry &E, const DeclContext *DC) {
        return E.getCommonAncestor() < DC;
      }

      bool operator()(const DeclContext *DC, const UnqualUsingEntry &E) {
        return DC < E.getCommonAncestor();
      }
    };
  };

  /// A collection of using directives, as used by C++ unqualified
  /// lookup.
  class UnqualUsingDirectiveSet {
    typedef llvm::SmallVector<UnqualUsingEntry, 8> ListTy;

    ListTy list;
    llvm::SmallPtrSet<DeclContext*, 8> visited;

  public:
    UnqualUsingDirectiveSet() {}

    void visitScopeChain(Scope *S, Scope *InnermostFileScope) {
      // C++ [namespace.udir]p1: 
      //   During unqualified name lookup, the names appear as if they
      //   were declared in the nearest enclosing namespace which contains
      //   both the using-directive and the nominated namespace.
      DeclContext *InnermostFileDC
        = static_cast<DeclContext*>(InnermostFileScope->getEntity());
      assert(InnermostFileDC && InnermostFileDC->isFileContext());

      for (; S; S = S->getParent()) {
        if (DeclContext *Ctx = static_cast<DeclContext*>(S->getEntity())) {
          DeclContext *EffectiveDC = (Ctx->isFileContext() ? Ctx : InnermostFileDC);
          visit(Ctx, EffectiveDC);
        } else {
          Scope::udir_iterator I = S->using_directives_begin(),
                             End = S->using_directives_end();
          
          for (; I != End; ++I)
            visit(I->getAs<UsingDirectiveDecl>(), InnermostFileDC);
        }
      }
    }

    // Visits a context and collect all of its using directives
    // recursively.  Treats all using directives as if they were
    // declared in the context.
    //
    // A given context is only every visited once, so it is important
    // that contexts be visited from the inside out in order to get
    // the effective DCs right.
    void visit(DeclContext *DC, DeclContext *EffectiveDC) {
      if (!visited.insert(DC))
        return;

      addUsingDirectives(DC, EffectiveDC);
    }

    // Visits a using directive and collects all of its using
    // directives recursively.  Treats all using directives as if they
    // were declared in the effective DC.
    void visit(UsingDirectiveDecl *UD, DeclContext *EffectiveDC) {
      DeclContext *NS = UD->getNominatedNamespace();
      if (!visited.insert(NS))
        return;

      addUsingDirective(UD, EffectiveDC);
      addUsingDirectives(NS, EffectiveDC);
    }

    // Adds all the using directives in a context (and those nominated
    // by its using directives, transitively) as if they appeared in
    // the given effective context.
    void addUsingDirectives(DeclContext *DC, DeclContext *EffectiveDC) {
      llvm::SmallVector<DeclContext*,4> queue;
      while (true) {
        DeclContext::udir_iterator I, End;
        for (llvm::tie(I, End) = DC->getUsingDirectives(); I != End; ++I) {
          UsingDirectiveDecl *UD = *I;
          DeclContext *NS = UD->getNominatedNamespace();
          if (visited.insert(NS)) {
            addUsingDirective(UD, EffectiveDC);
            queue.push_back(NS);
          }
        }

        if (queue.empty())
          return;

        DC = queue.back();
        queue.pop_back();
      }
    }

    // Add a using directive as if it had been declared in the given
    // context.  This helps implement C++ [namespace.udir]p3:
    //   The using-directive is transitive: if a scope contains a
    //   using-directive that nominates a second namespace that itself
    //   contains using-directives, the effect is as if the
    //   using-directives from the second namespace also appeared in
    //   the first.
    void addUsingDirective(UsingDirectiveDecl *UD, DeclContext *EffectiveDC) {
      // Find the common ancestor between the effective context and
      // the nominated namespace.
      DeclContext *Common = UD->getNominatedNamespace();
      while (!Common->Encloses(EffectiveDC))
        Common = Common->getParent();
      Common = Common->getPrimaryContext();
      
      list.push_back(UnqualUsingEntry(UD->getNominatedNamespace(), Common));
    }

    void done() {
      std::sort(list.begin(), list.end(), UnqualUsingEntry::Comparator());
    }

    typedef ListTy::iterator iterator;
    typedef ListTy::const_iterator const_iterator;
    
    iterator begin() { return list.begin(); }
    iterator end() { return list.end(); }
    const_iterator begin() const { return list.begin(); }
    const_iterator end() const { return list.end(); }

    std::pair<const_iterator,const_iterator>
    getNamespacesFor(DeclContext *DC) const {
      return std::equal_range(begin(), end(), DC->getPrimaryContext(),
                              UnqualUsingEntry::Comparator());
    }
  };
}

// Retrieve the set of identifier namespaces that correspond to a
// specific kind of name lookup.
inline unsigned
getIdentifierNamespacesFromLookupNameKind(Sema::LookupNameKind NameKind,
                                          bool CPlusPlus) {
  unsigned IDNS = 0;
  switch (NameKind) {
  case Sema::LookupOrdinaryName:
  case Sema::LookupOperatorName:
  case Sema::LookupRedeclarationWithLinkage:
    IDNS = Decl::IDNS_Ordinary;
    if (CPlusPlus)
      IDNS |= Decl::IDNS_Tag | Decl::IDNS_Member;
    break;

  case Sema::LookupTagName:
    IDNS = Decl::IDNS_Tag;
    break;

  case Sema::LookupMemberName:
    IDNS = Decl::IDNS_Member;
    if (CPlusPlus)
      IDNS |= Decl::IDNS_Tag | Decl::IDNS_Ordinary;
    break;

  case Sema::LookupNestedNameSpecifierName:
  case Sema::LookupNamespaceName:
    IDNS = Decl::IDNS_Ordinary | Decl::IDNS_Tag | Decl::IDNS_Member;
    break;

  case Sema::LookupObjCProtocolName:
    IDNS = Decl::IDNS_ObjCProtocol;
    break;

  case Sema::LookupObjCImplementationName:
    IDNS = Decl::IDNS_ObjCImplementation;
    break;

  case Sema::LookupObjCCategoryImplName:
    IDNS = Decl::IDNS_ObjCCategoryImpl;
    break;
  }
  return IDNS;
}

// Necessary because CXXBasePaths is not complete in Sema.h
void Sema::LookupResult::deletePaths(CXXBasePaths *Paths) {
  delete Paths;
}

void Sema::LookupResult::resolveKind() {
  unsigned N = Decls.size();

  // Fast case: no possible ambiguity.
  if (N <= 1) return;

  // Don't do any extra resolution if we've already resolved as ambiguous.
  if (Kind == Ambiguous) return;

  llvm::SmallPtrSet<NamedDecl*, 16> Unique;

  bool Ambiguous = false;
  bool HasTag = false, HasFunction = false, HasNonFunction = false;

  unsigned UniqueTagIndex = 0;
  
  unsigned I = 0;
  while (I < N) {
    NamedDecl *D = Decls[I];
    assert(D == D->getUnderlyingDecl());

    NamedDecl *CanonD = cast<NamedDecl>(D->getCanonicalDecl());
    if (!Unique.insert(CanonD)) {
      // If it's not unique, pull something off the back (and
      // continue at this index).
      Decls[I] = Decls[--N];
    } else if (isa<UnresolvedUsingDecl>(D)) {
      // FIXME: proper support for UnresolvedUsingDecls.
      Decls[I] = Decls[--N];
    } else {
      // Otherwise, do some decl type analysis and then continue.
      if (isa<TagDecl>(D)) {
        if (HasTag)
          Ambiguous = true;
        UniqueTagIndex = I;
        HasTag = true;
      } else if (D->isFunctionOrFunctionTemplate()) {
        HasFunction = true;
      } else {
        if (HasNonFunction)
          Ambiguous = true;
        HasNonFunction = true;
      }
      I++;
    }
  }

  // C++ [basic.scope.hiding]p2:
  //   A class name or enumeration name can be hidden by the name of
  //   an object, function, or enumerator declared in the same
  //   scope. If a class or enumeration name and an object, function,
  //   or enumerator are declared in the same scope (in any order)
  //   with the same name, the class or enumeration name is hidden
  //   wherever the object, function, or enumerator name is visible.
  // But it's still an error if there are distinct tag types found,
  // even if they're not visible. (ref?)
  if (HasTag && !Ambiguous && (HasFunction || HasNonFunction))
    Decls[UniqueTagIndex] = Decls[--N];

  Decls.set_size(N);

  if (HasFunction && HasNonFunction)
    Ambiguous = true;

  if (Ambiguous)
    setAmbiguous(LookupResult::AmbiguousReference);
  else if (N > 1)
    Kind = LookupResult::FoundOverloaded;
  else
    Kind = LookupResult::Found;
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
NamedDecl *Sema::LookupResult::getAsSingleDecl(ASTContext &C) const {
  size_t size = Decls.size();
  if (size == 0) return 0;
  if (size == 1) return *begin();

  if (isAmbiguous()) return 0;

  iterator I = begin(), E = end();

  OverloadedFunctionDecl *Ovl
    = OverloadedFunctionDecl::Create(C, (*I)->getDeclContext(),
                                        (*I)->getDeclName());
  for (; I != E; ++I) {
    NamedDecl *ND = *I;
    assert(ND->getUnderlyingDecl() == ND
           && "decls in lookup result should have redirections stripped");
    assert(ND->isFunctionOrFunctionTemplate());
    if (isa<FunctionDecl>(ND))
      Ovl->addOverload(cast<FunctionDecl>(ND));
    else
      Ovl->addOverload(cast<FunctionTemplateDecl>(ND));
    // FIXME: UnresolvedUsingDecls.
  }
  
  return Ovl;
}

void Sema::LookupResult::addDeclsFromBasePaths(const CXXBasePaths &P) {
  CXXBasePaths::paths_iterator I, E;
  DeclContext::lookup_iterator DI, DE;
  for (I = P.begin(), E = P.end(); I != E; ++I)
    for (llvm::tie(DI,DE) = I->Decls; DI != DE; ++DI)
      addDecl(*DI);
}

void Sema::LookupResult::setAmbiguousBaseSubobjects(CXXBasePaths &P) {
  Paths = new CXXBasePaths;
  Paths->swap(P);
  addDeclsFromBasePaths(*Paths);
  resolveKind();
  setAmbiguous(AmbiguousBaseSubobjects);
}

void Sema::LookupResult::setAmbiguousBaseSubobjectTypes(CXXBasePaths &P) {
  Paths = new CXXBasePaths;
  Paths->swap(P);
  addDeclsFromBasePaths(*Paths);
  resolveKind();
  setAmbiguous(AmbiguousBaseSubobjectTypes);
}

void Sema::LookupResult::print(llvm::raw_ostream &Out) {
  Out << Decls.size() << " result(s)";
  if (isAmbiguous()) Out << ", ambiguous";
  if (Paths) Out << ", base paths present";
  
  for (iterator I = begin(), E = end(); I != E; ++I) {
    Out << "\n";
    (*I)->print(Out, 2);
  }
}

// Adds all qualifying matches for a name within a decl context to the
// given lookup result.  Returns true if any matches were found.
static bool LookupDirect(Sema::LookupResult &R,
                         const DeclContext *DC,
                         DeclarationName Name,
                         Sema::LookupNameKind NameKind,
                         unsigned IDNS) {
  bool Found = false;

  DeclContext::lookup_const_iterator I, E;
  for (llvm::tie(I, E) = DC->lookup(Name); I != E; ++I)
    if (Sema::isAcceptableLookupResult(*I, NameKind, IDNS))
      R.addDecl(*I), Found = true;

  return Found;
}

// Performs C++ unqualified lookup into the given file context.
static bool
CppNamespaceLookup(Sema::LookupResult &R, ASTContext &Context, DeclContext *NS,
                   DeclarationName Name, Sema::LookupNameKind NameKind,
                   unsigned IDNS, UnqualUsingDirectiveSet &UDirs) {

  assert(NS && NS->isFileContext() && "CppNamespaceLookup() requires namespace!");

  // Perform direct name lookup into the LookupCtx.
  bool Found = LookupDirect(R, NS, Name, NameKind, IDNS);

  // Perform direct name lookup into the namespaces nominated by the
  // using directives whose common ancestor is this namespace.
  UnqualUsingDirectiveSet::const_iterator UI, UEnd;
  llvm::tie(UI, UEnd) = UDirs.getNamespacesFor(NS);

  for (; UI != UEnd; ++UI)
    if (LookupDirect(R, UI->getNominatedNamespace(), Name, NameKind, IDNS))
      Found = true;

  R.resolveKind();

  return Found;
}

static bool isNamespaceOrTranslationUnitScope(Scope *S) {
  if (DeclContext *Ctx = static_cast<DeclContext*>(S->getEntity()))
    return Ctx->isFileContext();
  return false;
}

// Find the next outer declaration context corresponding to this scope.
static DeclContext *findOuterContext(Scope *S) {
  for (S = S->getParent(); S; S = S->getParent())
    if (S->getEntity())
      return static_cast<DeclContext *>(S->getEntity())->getPrimaryContext();
  
  return 0;
}

bool
Sema::CppLookupName(LookupResult &R, Scope *S, DeclarationName Name,
                    LookupNameKind NameKind, bool RedeclarationOnly) {
  assert(getLangOptions().CPlusPlus &&
         "Can perform only C++ lookup");
  unsigned IDNS
    = getIdentifierNamespacesFromLookupNameKind(NameKind, /*CPlusPlus*/ true);

  // If we're testing for redeclarations, also look in the friend namespaces.
  if (RedeclarationOnly) {
    if (IDNS & Decl::IDNS_Tag) IDNS |= Decl::IDNS_TagFriend;
    if (IDNS & Decl::IDNS_Ordinary) IDNS |= Decl::IDNS_OrdinaryFriend;
  }

  Scope *Initial = S;
  IdentifierResolver::iterator
    I = IdResolver.begin(Name),
    IEnd = IdResolver.end();

  // First we lookup local scope.
  // We don't consider using-directives, as per 7.3.4.p1 [namespace.udir]
  // ...During unqualified name lookup (3.4.1), the names appear as if
  // they were declared in the nearest enclosing namespace which contains
  // both the using-directive and the nominated namespace.
  // [Note: in this context, "contains" means "contains directly or
  // indirectly".
  //
  // For example:
  // namespace A { int i; }
  // void foo() {
  //   int i;
  //   {
  //     using namespace A;
  //     ++i; // finds local 'i', A::i appears at global scope
  //   }
  // }
  //
  for (; S && !isNamespaceOrTranslationUnitScope(S); S = S->getParent()) {
    // Check whether the IdResolver has anything in this scope.
    bool Found = false;
    for (; I != IEnd && S->isDeclScope(DeclPtrTy::make(*I)); ++I) {
      if (isAcceptableLookupResult(*I, NameKind, IDNS)) {
        Found = true;
        R.addDecl(*I);
      }
    }
    if (Found) {
      R.resolveKind();
      return true;
    }

    if (DeclContext *Ctx = static_cast<DeclContext*>(S->getEntity())) {
      DeclContext *OuterCtx = findOuterContext(S);
      for (; Ctx && Ctx->getPrimaryContext() != OuterCtx; 
           Ctx = Ctx->getLookupParent()) {
        if (Ctx->isFunctionOrMethod())
          continue;
        
        // Perform qualified name lookup into this context.
        // FIXME: In some cases, we know that every name that could be found by
        // this qualified name lookup will also be on the identifier chain. For
        // example, inside a class without any base classes, we never need to
        // perform qualified lookup because all of the members are on top of the
        // identifier chain.
        if (LookupQualifiedName(R, Ctx, Name, NameKind, RedeclarationOnly))
          return true;
      }
    }
  }

  // Stop if we ran out of scopes.
  // FIXME:  This really, really shouldn't be happening.
  if (!S) return false;

  // Collect UsingDirectiveDecls in all scopes, and recursively all
  // nominated namespaces by those using-directives.
  //
  // FIXME: Cache this sorted list in Scope structure, and DeclContext, so we
  // don't build it for each lookup!

  UnqualUsingDirectiveSet UDirs;
  UDirs.visitScopeChain(Initial, S);
  UDirs.done();

  // Lookup namespace scope, and global scope.
  // Unqualified name lookup in C++ requires looking into scopes
  // that aren't strictly lexical, and therefore we walk through the
  // context as well as walking through the scopes.

  for (; S; S = S->getParent()) {
    DeclContext *Ctx = static_cast<DeclContext *>(S->getEntity());
    if (Ctx->isTransparentContext())
      continue;

    assert(Ctx && Ctx->isFileContext() &&
           "We should have been looking only at file context here already.");

    // Check whether the IdResolver has anything in this scope.
    bool Found = false;
    for (; I != IEnd && S->isDeclScope(DeclPtrTy::make(*I)); ++I) {
      if (isAcceptableLookupResult(*I, NameKind, IDNS)) {
        // We found something.  Look for anything else in our scope
        // with this same name and in an acceptable identifier
        // namespace, so that we can construct an overload set if we
        // need to.
        Found = true;
        R.addDecl(*I);
      }
    }

    // Look into context considering using-directives.
    if (CppNamespaceLookup(R, Context, Ctx, Name, NameKind, IDNS, UDirs))
      Found = true;

    if (Found) {
      R.resolveKind();
      return true;
    }

    if (RedeclarationOnly && !Ctx->isTransparentContext())
      return false;
  }

  return !R.empty();
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
/// @param Loc      If provided, the source location where we're performing
/// name lookup. At present, this is only used to produce diagnostics when
/// C library functions (like "malloc") are implicitly declared.
///
/// @returns The result of name lookup, which includes zero or more
/// declarations and possibly additional information used to diagnose
/// ambiguities.
bool Sema::LookupName(LookupResult &R, Scope *S, DeclarationName Name,
                      LookupNameKind NameKind, bool RedeclarationOnly,
                      bool AllowBuiltinCreation, SourceLocation Loc) {
  if (!Name) return false;

  if (!getLangOptions().CPlusPlus) {
    // Unqualified name lookup in C/Objective-C is purely lexical, so
    // search in the declarations attached to the name.
    unsigned IDNS = 0;
    switch (NameKind) {
    case Sema::LookupOrdinaryName:
      IDNS = Decl::IDNS_Ordinary;
      break;

    case Sema::LookupTagName:
      IDNS = Decl::IDNS_Tag;
      break;

    case Sema::LookupMemberName:
      IDNS = Decl::IDNS_Member;
      break;

    case Sema::LookupOperatorName:
    case Sema::LookupNestedNameSpecifierName:
    case Sema::LookupNamespaceName:
      assert(false && "C does not perform these kinds of name lookup");
      break;

    case Sema::LookupRedeclarationWithLinkage:
      // Find the nearest non-transparent declaration scope.
      while (!(S->getFlags() & Scope::DeclScope) ||
             (S->getEntity() &&
              static_cast<DeclContext *>(S->getEntity())
                ->isTransparentContext()))
        S = S->getParent();
      IDNS = Decl::IDNS_Ordinary;
      break;

    case Sema::LookupObjCProtocolName:
      IDNS = Decl::IDNS_ObjCProtocol;
      break;

    case Sema::LookupObjCImplementationName:
      IDNS = Decl::IDNS_ObjCImplementation;
      break;

    case Sema::LookupObjCCategoryImplName:
      IDNS = Decl::IDNS_ObjCCategoryImpl;
      break;
    }

    // Scan up the scope chain looking for a decl that matches this
    // identifier that is in the appropriate namespace.  This search
    // should not take long, as shadowing of names is uncommon, and
    // deep shadowing is extremely uncommon.
    bool LeftStartingScope = false;

    for (IdentifierResolver::iterator I = IdResolver.begin(Name),
                                   IEnd = IdResolver.end();
         I != IEnd; ++I)
      if ((*I)->isInIdentifierNamespace(IDNS)) {
        if (NameKind == LookupRedeclarationWithLinkage) {
          // Determine whether this (or a previous) declaration is
          // out-of-scope.
          if (!LeftStartingScope && !S->isDeclScope(DeclPtrTy::make(*I)))
            LeftStartingScope = true;

          // If we found something outside of our starting scope that
          // does not have linkage, skip it.
          if (LeftStartingScope && !((*I)->hasLinkage()))
            continue;
        }

        R.addDecl(*I);

        if ((*I)->getAttr<OverloadableAttr>()) {
          // If this declaration has the "overloadable" attribute, we
          // might have a set of overloaded functions.

          // Figure out what scope the identifier is in.
          while (!(S->getFlags() & Scope::DeclScope) ||
                 !S->isDeclScope(DeclPtrTy::make(*I)))
            S = S->getParent();

          // Find the last declaration in this scope (with the same
          // name, naturally).
          IdentifierResolver::iterator LastI = I;
          for (++LastI; LastI != IEnd; ++LastI) {
            if (!S->isDeclScope(DeclPtrTy::make(*LastI)))
              break;
            R.addDecl(*LastI);
          }
        }

        R.resolveKind();

        return true;
      }
  } else {
    // Perform C++ unqualified name lookup.
    if (CppLookupName(R, S, Name, NameKind, RedeclarationOnly))
      return true;
  }

  // If we didn't find a use of this identifier, and if the identifier
  // corresponds to a compiler builtin, create the decl object for the builtin
  // now, injecting it into translation unit scope, and return it.
  if (NameKind == LookupOrdinaryName ||
      NameKind == LookupRedeclarationWithLinkage) {
    IdentifierInfo *II = Name.getAsIdentifierInfo();
    if (II && AllowBuiltinCreation) {
      // If this is a builtin on this (or all) targets, create the decl.
      if (unsigned BuiltinID = II->getBuiltinID()) {
        // In C++, we don't have any predefined library functions like
        // 'malloc'. Instead, we'll just error.
        if (getLangOptions().CPlusPlus &&
            Context.BuiltinInfo.isPredefinedLibFunction(BuiltinID))
          return false;

        NamedDecl *D = LazilyCreateBuiltin((IdentifierInfo *)II, BuiltinID,
                                           S, RedeclarationOnly, Loc);
        if (D) R.addDecl(D);
        return (D != NULL);
      }
    }
  }
  return false;
}

/// @brief Perform qualified name lookup in the namespaces nominated by
/// using directives by the given context.
///
/// C++98 [namespace.qual]p2:
///   Given X::m (where X is a user-declared namespace), or given ::m
///   (where X is the global namespace), let S be the set of all
///   declarations of m in X and in the transitive closure of all
///   namespaces nominated by using-directives in X and its used
///   namespaces, except that using-directives are ignored in any
///   namespace, including X, directly containing one or more
///   declarations of m. No namespace is searched more than once in
///   the lookup of a name. If S is the empty set, the program is
///   ill-formed. Otherwise, if S has exactly one member, or if the
///   context of the reference is a using-declaration
///   (namespace.udecl), S is the required set of declarations of
///   m. Otherwise if the use of m is not one that allows a unique
///   declaration to be chosen from S, the program is ill-formed.
/// C++98 [namespace.qual]p5:
///   During the lookup of a qualified namespace member name, if the
///   lookup finds more than one declaration of the member, and if one
///   declaration introduces a class name or enumeration name and the
///   other declarations either introduce the same object, the same
///   enumerator or a set of functions, the non-type name hides the
///   class or enumeration name if and only if the declarations are
///   from the same namespace; otherwise (the declarations are from
///   different namespaces), the program is ill-formed.
static bool LookupQualifiedNameInUsingDirectives(Sema::LookupResult &R,
                                                 DeclContext *StartDC,
                                                 DeclarationName Name,
                                                 Sema::LookupNameKind NameKind,
                                                 unsigned IDNS) {
  assert(StartDC->isFileContext() && "start context is not a file context");

  DeclContext::udir_iterator I = StartDC->using_directives_begin();
  DeclContext::udir_iterator E = StartDC->using_directives_end();

  if (I == E) return false;

  // We have at least added all these contexts to the queue.
  llvm::DenseSet<DeclContext*> Visited;
  Visited.insert(StartDC);

  // We have not yet looked into these namespaces, much less added
  // their "using-children" to the queue.
  llvm::SmallVector<NamespaceDecl*, 8> Queue;

  // We have already looked into the initial namespace; seed the queue
  // with its using-children.
  for (; I != E; ++I) {
    NamespaceDecl *ND = (*I)->getNominatedNamespace()->getOriginalNamespace();
    if (Visited.insert(ND).second)
      Queue.push_back(ND);
  }

  // The easiest way to implement the restriction in [namespace.qual]p5
  // is to check whether any of the individual results found a tag
  // and, if so, to declare an ambiguity if the final result is not
  // a tag.
  bool FoundTag = false;
  bool FoundNonTag = false;

  Sema::LookupResult LocalR;

  bool Found = false;
  while (!Queue.empty()) {
    NamespaceDecl *ND = Queue.back();
    Queue.pop_back();

    // We go through some convolutions here to avoid copying results
    // between LookupResults.
    bool UseLocal = !R.empty();
    Sema::LookupResult &DirectR = UseLocal ? LocalR : R;
    bool FoundDirect = LookupDirect(DirectR, ND, Name, NameKind, IDNS);

    if (FoundDirect) {
      // First do any local hiding.
      DirectR.resolveKind();

      // If the local result is a tag, remember that.
      if (DirectR.isSingleTagDecl())
        FoundTag = true;
      else
        FoundNonTag = true;

      // Append the local results to the total results if necessary.
      if (UseLocal) {
        R.addAllDecls(LocalR);
        LocalR.clear();
      }
    }

    // If we find names in this namespace, ignore its using directives.
    if (FoundDirect) {
      Found = true;
      continue;
    }

    for (llvm::tie(I,E) = ND->getUsingDirectives(); I != E; ++I) {
      NamespaceDecl *Nom = (*I)->getNominatedNamespace();
      if (Visited.insert(Nom).second)
        Queue.push_back(Nom);
    }
  }

  if (Found) {
    if (FoundTag && FoundNonTag)
      R.setAmbiguousQualifiedTagHiding();
    else
      R.resolveKind();
  }

  return Found;
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
bool Sema::LookupQualifiedName(LookupResult &R, DeclContext *LookupCtx,
                               DeclarationName Name, LookupNameKind NameKind,
                               bool RedeclarationOnly) {
  assert(LookupCtx && "Sema::LookupQualifiedName requires a lookup context");

  if (!Name)
    return false;

  // If we're performing qualified name lookup (e.g., lookup into a
  // struct), find fields as part of ordinary name lookup.
  unsigned IDNS
    = getIdentifierNamespacesFromLookupNameKind(NameKind,
                                                getLangOptions().CPlusPlus);
  if (NameKind == LookupOrdinaryName)
    IDNS |= Decl::IDNS_Member;

  // Make sure that the declaration context is complete.
  assert((!isa<TagDecl>(LookupCtx) ||
          LookupCtx->isDependentContext() ||
          cast<TagDecl>(LookupCtx)->isDefinition() ||
          Context.getTypeDeclType(cast<TagDecl>(LookupCtx))->getAs<TagType>()
            ->isBeingDefined()) &&
         "Declaration context must already be complete!");

  // Perform qualified name lookup into the LookupCtx.
  if (LookupDirect(R, LookupCtx, Name, NameKind, IDNS)) {
    R.resolveKind();
    return true;
  }

  // Don't descend into implied contexts for redeclarations.
  // C++98 [namespace.qual]p6:
  //   In a declaration for a namespace member in which the
  //   declarator-id is a qualified-id, given that the qualified-id
  //   for the namespace member has the form
  //     nested-name-specifier unqualified-id
  //   the unqualified-id shall name a member of the namespace
  //   designated by the nested-name-specifier.
  // See also [class.mfct]p5 and [class.static.data]p2.
  if (RedeclarationOnly)
    return false;

  // If this is a namespace, look it up in 
  if (LookupCtx->isFileContext())
    return LookupQualifiedNameInUsingDirectives(R, LookupCtx, Name, NameKind,
                                                IDNS);

  // If this isn't a C++ class, we aren't allowed to look into base
  // classes, we're done.
  if (!isa<CXXRecordDecl>(LookupCtx))
    return false;

  // Perform lookup into our base classes.
  CXXRecordDecl *LookupRec = cast<CXXRecordDecl>(LookupCtx);
  CXXBasePaths Paths;
  Paths.setOrigin(LookupRec);

  // Look for this member in our base classes
  CXXRecordDecl::BaseMatchesCallback *BaseCallback = 0;
  switch (NameKind) {
    case LookupOrdinaryName:
    case LookupMemberName:
    case LookupRedeclarationWithLinkage:
      BaseCallback = &CXXRecordDecl::FindOrdinaryMember;
      break;
      
    case LookupTagName:
      BaseCallback = &CXXRecordDecl::FindTagMember;
      break;
      
    case LookupOperatorName:
    case LookupNamespaceName:
    case LookupObjCProtocolName:
    case LookupObjCImplementationName:
    case LookupObjCCategoryImplName:
      // These lookups will never find a member in a C++ class (or base class).
      return false;
      
    case LookupNestedNameSpecifierName:
      BaseCallback = &CXXRecordDecl::FindNestedNameSpecifierMember;
      break;
  }
  
  if (!LookupRec->lookupInBases(BaseCallback, Name.getAsOpaquePtr(), Paths))
    return false;

  // C++ [class.member.lookup]p2:
  //   [...] If the resulting set of declarations are not all from
  //   sub-objects of the same type, or the set has a nonstatic member
  //   and includes members from distinct sub-objects, there is an
  //   ambiguity and the program is ill-formed. Otherwise that set is
  //   the result of the lookup.
  // FIXME: support using declarations!
  QualType SubobjectType;
  int SubobjectNumber = 0;
  for (CXXBasePaths::paths_iterator Path = Paths.begin(), PathEnd = Paths.end();
       Path != PathEnd; ++Path) {
    const CXXBasePathElement &PathElement = Path->back();

    // Determine whether we're looking at a distinct sub-object or not.
    if (SubobjectType.isNull()) {
      // This is the first subobject we've looked at. Record its type.
      SubobjectType = Context.getCanonicalType(PathElement.Base->getType());
      SubobjectNumber = PathElement.SubobjectNumber;
    } else if (SubobjectType
                 != Context.getCanonicalType(PathElement.Base->getType())) {
      // We found members of the given name in two subobjects of
      // different types. This lookup is ambiguous.
      R.setAmbiguousBaseSubobjectTypes(Paths);
      return true;
    } else if (SubobjectNumber != PathElement.SubobjectNumber) {
      // We have a different subobject of the same type.

      // C++ [class.member.lookup]p5:
      //   A static member, a nested type or an enumerator defined in
      //   a base class T can unambiguously be found even if an object
      //   has more than one base class subobject of type T.
      Decl *FirstDecl = *Path->Decls.first;
      if (isa<VarDecl>(FirstDecl) ||
          isa<TypeDecl>(FirstDecl) ||
          isa<EnumConstantDecl>(FirstDecl))
        continue;

      if (isa<CXXMethodDecl>(FirstDecl)) {
        // Determine whether all of the methods are static.
        bool AllMethodsAreStatic = true;
        for (DeclContext::lookup_iterator Func = Path->Decls.first;
             Func != Path->Decls.second; ++Func) {
          if (!isa<CXXMethodDecl>(*Func)) {
            assert(isa<TagDecl>(*Func) && "Non-function must be a tag decl");
            break;
          }

          if (!cast<CXXMethodDecl>(*Func)->isStatic()) {
            AllMethodsAreStatic = false;
            break;
          }
        }

        if (AllMethodsAreStatic)
          continue;
      }

      // We have found a nonstatic member name in multiple, distinct
      // subobjects. Name lookup is ambiguous.
      R.setAmbiguousBaseSubobjects(Paths);
      return true;
    }
  }

  // Lookup in a base class succeeded; return these results.

  DeclContext::lookup_iterator I, E;
  for (llvm::tie(I,E) = Paths.front().Decls; I != E; ++I)
    R.addDecl(*I);
  R.resolveKind();
  return true;
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
/// @param SS       An optional C++ scope-specifier, e.g., "::N::M".
///
/// @param Name     The name of the entity that name lookup will
/// search for.
///
/// @param Loc      If provided, the source location where we're performing
/// name lookup. At present, this is only used to produce diagnostics when
/// C library functions (like "malloc") are implicitly declared.
///
/// @param EnteringContext Indicates whether we are going to enter the
/// context of the scope-specifier SS (if present).
///
/// @returns True if any decls were found (but possibly ambiguous)
bool Sema::LookupParsedName(LookupResult &R, Scope *S, const CXXScopeSpec *SS,
                            DeclarationName Name, LookupNameKind NameKind,
                            bool RedeclarationOnly, bool AllowBuiltinCreation,
                            SourceLocation Loc,
                            bool EnteringContext) {
  if (SS && SS->isInvalid()) {
    // When the scope specifier is invalid, don't even look for
    // anything.
    return false;
  }

  if (SS && SS->isSet()) {
    if (DeclContext *DC = computeDeclContext(*SS, EnteringContext)) {
      // We have resolved the scope specifier to a particular declaration
      // contex, and will perform name lookup in that context.
      if (!DC->isDependentContext() && RequireCompleteDeclContext(*SS))
        return false;

      return LookupQualifiedName(R, DC, Name, NameKind, RedeclarationOnly);
    }

    // We could not resolve the scope specified to a specific declaration
    // context, which means that SS refers to an unknown specialization.
    // Name lookup can't find anything in this case.
    return false;
  }

  // Perform unqualified name lookup starting in the given scope.
  return LookupName(R, S, Name, NameKind, RedeclarationOnly,
                    AllowBuiltinCreation, Loc);
}


/// @brief Produce a diagnostic describing the ambiguity that resulted
/// from name lookup.
///
/// @param Result       The ambiguous name lookup result.
///
/// @param Name         The name of the entity that name lookup was
/// searching for.
///
/// @param NameLoc      The location of the name within the source code.
///
/// @param LookupRange  A source range that provides more
/// source-location information concerning the lookup itself. For
/// example, this range might highlight a nested-name-specifier that
/// precedes the name.
///
/// @returns true
bool Sema::DiagnoseAmbiguousLookup(LookupResult &Result, DeclarationName Name,
                                   SourceLocation NameLoc,
                                   SourceRange LookupRange) {
  assert(Result.isAmbiguous() && "Lookup result must be ambiguous");

  switch (Result.getAmbiguityKind()) {
  case LookupResult::AmbiguousBaseSubobjects: {
    CXXBasePaths *Paths = Result.getBasePaths();
    QualType SubobjectType = Paths->front().back().Base->getType();
    Diag(NameLoc, diag::err_ambiguous_member_multiple_subobjects)
      << Name << SubobjectType << getAmbiguousPathsDisplayString(*Paths)
      << LookupRange;
    
    DeclContext::lookup_iterator Found = Paths->front().Decls.first;
    while (isa<CXXMethodDecl>(*Found) &&
           cast<CXXMethodDecl>(*Found)->isStatic())
      ++Found;
    
    Diag((*Found)->getLocation(), diag::note_ambiguous_member_found);
    
    return true;
  }

  case LookupResult::AmbiguousBaseSubobjectTypes: {
    Diag(NameLoc, diag::err_ambiguous_member_multiple_subobject_types)
      << Name << LookupRange;
    
    CXXBasePaths *Paths = Result.getBasePaths();
    std::set<Decl *> DeclsPrinted;
    for (CXXBasePaths::paths_iterator Path = Paths->begin(),
                                      PathEnd = Paths->end();
         Path != PathEnd; ++Path) {
      Decl *D = *Path->Decls.first;
      if (DeclsPrinted.insert(D).second)
        Diag(D->getLocation(), diag::note_ambiguous_member_found);
    }

    return true;
  }

  case LookupResult::AmbiguousTagHiding: {
    Diag(NameLoc, diag::err_ambiguous_tag_hiding) << Name << LookupRange;

    llvm::SmallPtrSet<NamedDecl*,8> TagDecls;

    LookupResult::iterator DI, DE = Result.end();
    for (DI = Result.begin(); DI != DE; ++DI)
      if (TagDecl *TD = dyn_cast<TagDecl>(*DI)) {
        TagDecls.insert(TD);
        Diag(TD->getLocation(), diag::note_hidden_tag);
      }

    for (DI = Result.begin(); DI != DE; ++DI)
      if (!isa<TagDecl>(*DI))
        Diag((*DI)->getLocation(), diag::note_hiding_object);

    // For recovery purposes, go ahead and implement the hiding.
    Result.hideDecls(TagDecls);

    return true;
  }

  case LookupResult::AmbiguousReference: {
    Diag(NameLoc, diag::err_ambiguous_reference) << Name << LookupRange;
  
    LookupResult::iterator DI = Result.begin(), DE = Result.end();
    for (; DI != DE; ++DI)
      Diag((*DI)->getLocation(), diag::note_ambiguous_candidate) << *DI;

    return true;
  }
  }

  llvm::llvm_unreachable("unknown ambiguity kind");
  return true;
}

static void
addAssociatedClassesAndNamespaces(QualType T,
                                  ASTContext &Context,
                          Sema::AssociatedNamespaceSet &AssociatedNamespaces,
                                  Sema::AssociatedClassSet &AssociatedClasses);

static void CollectNamespace(Sema::AssociatedNamespaceSet &Namespaces,
                             DeclContext *Ctx) {
  if (Ctx->isFileContext())
    Namespaces.insert(Ctx);
}

// \brief Add the associated classes and namespaces for argument-dependent
// lookup that involves a template argument (C++ [basic.lookup.koenig]p2).
static void
addAssociatedClassesAndNamespaces(const TemplateArgument &Arg,
                                  ASTContext &Context,
                           Sema::AssociatedNamespaceSet &AssociatedNamespaces,
                                  Sema::AssociatedClassSet &AssociatedClasses) {
  // C++ [basic.lookup.koenig]p2, last bullet:
  //   -- [...] ;
  switch (Arg.getKind()) {
    case TemplateArgument::Null:
      break;

    case TemplateArgument::Type:
      // [...] the namespaces and classes associated with the types of the
      // template arguments provided for template type parameters (excluding
      // template template parameters)
      addAssociatedClassesAndNamespaces(Arg.getAsType(), Context,
                                        AssociatedNamespaces,
                                        AssociatedClasses);
      break;

    case TemplateArgument::Template: {
      // [...] the namespaces in which any template template arguments are
      // defined; and the classes in which any member templates used as
      // template template arguments are defined.
      TemplateName Template = Arg.getAsTemplate();
      if (ClassTemplateDecl *ClassTemplate
                 = dyn_cast<ClassTemplateDecl>(Template.getAsTemplateDecl())) {
        DeclContext *Ctx = ClassTemplate->getDeclContext();
        if (CXXRecordDecl *EnclosingClass = dyn_cast<CXXRecordDecl>(Ctx))
          AssociatedClasses.insert(EnclosingClass);
        // Add the associated namespace for this class.
        while (Ctx->isRecord())
          Ctx = Ctx->getParent();
        CollectNamespace(AssociatedNamespaces, Ctx);
      }
      break;
    }
      
    case TemplateArgument::Declaration:
    case TemplateArgument::Integral:
    case TemplateArgument::Expression:
      // [Note: non-type template arguments do not contribute to the set of
      //  associated namespaces. ]
      break;

    case TemplateArgument::Pack:
      for (TemplateArgument::pack_iterator P = Arg.pack_begin(),
                                        PEnd = Arg.pack_end();
           P != PEnd; ++P)
        addAssociatedClassesAndNamespaces(*P, Context,
                                          AssociatedNamespaces,
                                          AssociatedClasses);
      break;
  }
}

// \brief Add the associated classes and namespaces for
// argument-dependent lookup with an argument of class type
// (C++ [basic.lookup.koenig]p2).
static void
addAssociatedClassesAndNamespaces(CXXRecordDecl *Class,
                                  ASTContext &Context,
                            Sema::AssociatedNamespaceSet &AssociatedNamespaces,
                            Sema::AssociatedClassSet &AssociatedClasses) {
  // C++ [basic.lookup.koenig]p2:
  //   [...]
  //     -- If T is a class type (including unions), its associated
  //        classes are: the class itself; the class of which it is a
  //        member, if any; and its direct and indirect base
  //        classes. Its associated namespaces are the namespaces in
  //        which its associated classes are defined.

  // Add the class of which it is a member, if any.
  DeclContext *Ctx = Class->getDeclContext();
  if (CXXRecordDecl *EnclosingClass = dyn_cast<CXXRecordDecl>(Ctx))
    AssociatedClasses.insert(EnclosingClass);
  // Add the associated namespace for this class.
  while (Ctx->isRecord())
    Ctx = Ctx->getParent();
  CollectNamespace(AssociatedNamespaces, Ctx);

  // Add the class itself. If we've already seen this class, we don't
  // need to visit base classes.
  if (!AssociatedClasses.insert(Class))
    return;

  // -- If T is a template-id, its associated namespaces and classes are
  //    the namespace in which the template is defined; for member
  //    templates, the member template’s class; the namespaces and classes
  //    associated with the types of the template arguments provided for
  //    template type parameters (excluding template template parameters); the
  //    namespaces in which any template template arguments are defined; and
  //    the classes in which any member templates used as template template
  //    arguments are defined. [Note: non-type template arguments do not
  //    contribute to the set of associated namespaces. ]
  if (ClassTemplateSpecializationDecl *Spec
        = dyn_cast<ClassTemplateSpecializationDecl>(Class)) {
    DeclContext *Ctx = Spec->getSpecializedTemplate()->getDeclContext();
    if (CXXRecordDecl *EnclosingClass = dyn_cast<CXXRecordDecl>(Ctx))
      AssociatedClasses.insert(EnclosingClass);
    // Add the associated namespace for this class.
    while (Ctx->isRecord())
      Ctx = Ctx->getParent();
    CollectNamespace(AssociatedNamespaces, Ctx);

    const TemplateArgumentList &TemplateArgs = Spec->getTemplateArgs();
    for (unsigned I = 0, N = TemplateArgs.size(); I != N; ++I)
      addAssociatedClassesAndNamespaces(TemplateArgs[I], Context,
                                        AssociatedNamespaces,
                                        AssociatedClasses);
  }

  // Add direct and indirect base classes along with their associated
  // namespaces.
  llvm::SmallVector<CXXRecordDecl *, 32> Bases;
  Bases.push_back(Class);
  while (!Bases.empty()) {
    // Pop this class off the stack.
    Class = Bases.back();
    Bases.pop_back();

    // Visit the base classes.
    for (CXXRecordDecl::base_class_iterator Base = Class->bases_begin(),
                                         BaseEnd = Class->bases_end();
         Base != BaseEnd; ++Base) {
      const RecordType *BaseType = Base->getType()->getAs<RecordType>();
      // In dependent contexts, we do ADL twice, and the first time around,
      // the base type might be a dependent TemplateSpecializationType, or a
      // TemplateTypeParmType. If that happens, simply ignore it.
      // FIXME: If we want to support export, we probably need to add the
      // namespace of the template in a TemplateSpecializationType, or even
      // the classes and namespaces of known non-dependent arguments.
      if (!BaseType)
        continue;
      CXXRecordDecl *BaseDecl = cast<CXXRecordDecl>(BaseType->getDecl());
      if (AssociatedClasses.insert(BaseDecl)) {
        // Find the associated namespace for this base class.
        DeclContext *BaseCtx = BaseDecl->getDeclContext();
        while (BaseCtx->isRecord())
          BaseCtx = BaseCtx->getParent();
        CollectNamespace(AssociatedNamespaces, BaseCtx);

        // Make sure we visit the bases of this base class.
        if (BaseDecl->bases_begin() != BaseDecl->bases_end())
          Bases.push_back(BaseDecl);
      }
    }
  }
}

// \brief Add the associated classes and namespaces for
// argument-dependent lookup with an argument of type T
// (C++ [basic.lookup.koenig]p2).
static void
addAssociatedClassesAndNamespaces(QualType T,
                                  ASTContext &Context,
                            Sema::AssociatedNamespaceSet &AssociatedNamespaces,
                                  Sema::AssociatedClassSet &AssociatedClasses) {
  // C++ [basic.lookup.koenig]p2:
  //
  //   For each argument type T in the function call, there is a set
  //   of zero or more associated namespaces and a set of zero or more
  //   associated classes to be considered. The sets of namespaces and
  //   classes is determined entirely by the types of the function
  //   arguments (and the namespace of any template template
  //   argument). Typedef names and using-declarations used to specify
  //   the types do not contribute to this set. The sets of namespaces
  //   and classes are determined in the following way:
  T = Context.getCanonicalType(T).getUnqualifiedType();

  //    -- If T is a pointer to U or an array of U, its associated
  //       namespaces and classes are those associated with U.
  //
  // We handle this by unwrapping pointer and array types immediately,
  // to avoid unnecessary recursion.
  while (true) {
    if (const PointerType *Ptr = T->getAs<PointerType>())
      T = Ptr->getPointeeType();
    else if (const ArrayType *Ptr = Context.getAsArrayType(T))
      T = Ptr->getElementType();
    else
      break;
  }

  //     -- If T is a fundamental type, its associated sets of
  //        namespaces and classes are both empty.
  if (T->getAs<BuiltinType>())
    return;

  //     -- If T is a class type (including unions), its associated
  //        classes are: the class itself; the class of which it is a
  //        member, if any; and its direct and indirect base
  //        classes. Its associated namespaces are the namespaces in
  //        which its associated classes are defined.
  if (const RecordType *ClassType = T->getAs<RecordType>())
    if (CXXRecordDecl *ClassDecl
        = dyn_cast<CXXRecordDecl>(ClassType->getDecl())) {
      addAssociatedClassesAndNamespaces(ClassDecl, Context,
                                        AssociatedNamespaces,
                                        AssociatedClasses);
      return;
    }

  //     -- If T is an enumeration type, its associated namespace is
  //        the namespace in which it is defined. If it is class
  //        member, its associated class is the member’s class; else
  //        it has no associated class.
  if (const EnumType *EnumT = T->getAs<EnumType>()) {
    EnumDecl *Enum = EnumT->getDecl();

    DeclContext *Ctx = Enum->getDeclContext();
    if (CXXRecordDecl *EnclosingClass = dyn_cast<CXXRecordDecl>(Ctx))
      AssociatedClasses.insert(EnclosingClass);

    // Add the associated namespace for this class.
    while (Ctx->isRecord())
      Ctx = Ctx->getParent();
    CollectNamespace(AssociatedNamespaces, Ctx);

    return;
  }

  //     -- If T is a function type, its associated namespaces and
  //        classes are those associated with the function parameter
  //        types and those associated with the return type.
  if (const FunctionType *FnType = T->getAs<FunctionType>()) {
    // Return type
    addAssociatedClassesAndNamespaces(FnType->getResultType(),
                                      Context,
                                      AssociatedNamespaces, AssociatedClasses);

    const FunctionProtoType *Proto = dyn_cast<FunctionProtoType>(FnType);
    if (!Proto)
      return;

    // Argument types
    for (FunctionProtoType::arg_type_iterator Arg = Proto->arg_type_begin(),
                                           ArgEnd = Proto->arg_type_end();
         Arg != ArgEnd; ++Arg)
      addAssociatedClassesAndNamespaces(*Arg, Context,
                                        AssociatedNamespaces, AssociatedClasses);

    return;
  }

  //     -- If T is a pointer to a member function of a class X, its
  //        associated namespaces and classes are those associated
  //        with the function parameter types and return type,
  //        together with those associated with X.
  //
  //     -- If T is a pointer to a data member of class X, its
  //        associated namespaces and classes are those associated
  //        with the member type together with those associated with
  //        X.
  if (const MemberPointerType *MemberPtr = T->getAs<MemberPointerType>()) {
    // Handle the type that the pointer to member points to.
    addAssociatedClassesAndNamespaces(MemberPtr->getPointeeType(),
                                      Context,
                                      AssociatedNamespaces,
                                      AssociatedClasses);

    // Handle the class type into which this points.
    if (const RecordType *Class = MemberPtr->getClass()->getAs<RecordType>())
      addAssociatedClassesAndNamespaces(cast<CXXRecordDecl>(Class->getDecl()),
                                        Context,
                                        AssociatedNamespaces,
                                        AssociatedClasses);

    return;
  }

  // FIXME: What about block pointers?
  // FIXME: What about Objective-C message sends?
}

/// \brief Find the associated classes and namespaces for
/// argument-dependent lookup for a call with the given set of
/// arguments.
///
/// This routine computes the sets of associated classes and associated
/// namespaces searched by argument-dependent lookup
/// (C++ [basic.lookup.argdep]) for a given set of arguments.
void
Sema::FindAssociatedClassesAndNamespaces(Expr **Args, unsigned NumArgs,
                                 AssociatedNamespaceSet &AssociatedNamespaces,
                                 AssociatedClassSet &AssociatedClasses) {
  AssociatedNamespaces.clear();
  AssociatedClasses.clear();

  // C++ [basic.lookup.koenig]p2:
  //   For each argument type T in the function call, there is a set
  //   of zero or more associated namespaces and a set of zero or more
  //   associated classes to be considered. The sets of namespaces and
  //   classes is determined entirely by the types of the function
  //   arguments (and the namespace of any template template
  //   argument).
  for (unsigned ArgIdx = 0; ArgIdx != NumArgs; ++ArgIdx) {
    Expr *Arg = Args[ArgIdx];

    if (Arg->getType() != Context.OverloadTy) {
      addAssociatedClassesAndNamespaces(Arg->getType(), Context,
                                        AssociatedNamespaces,
                                        AssociatedClasses);
      continue;
    }

    // [...] In addition, if the argument is the name or address of a
    // set of overloaded functions and/or function templates, its
    // associated classes and namespaces are the union of those
    // associated with each of the members of the set: the namespace
    // in which the function or function template is defined and the
    // classes and namespaces associated with its (non-dependent)
    // parameter types and return type.
    DeclRefExpr *DRE = 0;
    TemplateIdRefExpr *TIRE = 0;
    Arg = Arg->IgnoreParens();
    if (UnaryOperator *unaryOp = dyn_cast<UnaryOperator>(Arg)) {
      if (unaryOp->getOpcode() == UnaryOperator::AddrOf) {
        DRE = dyn_cast<DeclRefExpr>(unaryOp->getSubExpr());
        TIRE = dyn_cast<TemplateIdRefExpr>(unaryOp->getSubExpr());
      }
    } else {
      DRE = dyn_cast<DeclRefExpr>(Arg);
      TIRE = dyn_cast<TemplateIdRefExpr>(Arg);
    }

    OverloadedFunctionDecl *Ovl = 0;
    if (DRE)
      Ovl = dyn_cast<OverloadedFunctionDecl>(DRE->getDecl());
    else if (TIRE)
      Ovl = TIRE->getTemplateName().getAsOverloadedFunctionDecl();
    if (!Ovl)
      continue;

    for (OverloadedFunctionDecl::function_iterator Func = Ovl->function_begin(),
                                                FuncEnd = Ovl->function_end();
         Func != FuncEnd; ++Func) {
      FunctionDecl *FDecl = dyn_cast<FunctionDecl>(*Func);
      if (!FDecl)
        FDecl = cast<FunctionTemplateDecl>(*Func)->getTemplatedDecl();

      // Add the namespace in which this function was defined. Note
      // that, if this is a member function, we do *not* consider the
      // enclosing namespace of its class.
      DeclContext *Ctx = FDecl->getDeclContext();
      CollectNamespace(AssociatedNamespaces, Ctx);

      // Add the classes and namespaces associated with the parameter
      // types and return type of this function.
      addAssociatedClassesAndNamespaces(FDecl->getType(), Context,
                                        AssociatedNamespaces,
                                        AssociatedClasses);
    }
  }
}

/// IsAcceptableNonMemberOperatorCandidate - Determine whether Fn is
/// an acceptable non-member overloaded operator for a call whose
/// arguments have types T1 (and, if non-empty, T2). This routine
/// implements the check in C++ [over.match.oper]p3b2 concerning
/// enumeration types.
static bool
IsAcceptableNonMemberOperatorCandidate(FunctionDecl *Fn,
                                       QualType T1, QualType T2,
                                       ASTContext &Context) {
  if (T1->isDependentType() || (!T2.isNull() && T2->isDependentType()))
    return true;

  if (T1->isRecordType() || (!T2.isNull() && T2->isRecordType()))
    return true;

  const FunctionProtoType *Proto = Fn->getType()->getAs<FunctionProtoType>();
  if (Proto->getNumArgs() < 1)
    return false;

  if (T1->isEnumeralType()) {
    QualType ArgType = Proto->getArgType(0).getNonReferenceType();
    if (Context.getCanonicalType(T1).getUnqualifiedType()
          == Context.getCanonicalType(ArgType).getUnqualifiedType())
      return true;
  }

  if (Proto->getNumArgs() < 2)
    return false;

  if (!T2.isNull() && T2->isEnumeralType()) {
    QualType ArgType = Proto->getArgType(1).getNonReferenceType();
    if (Context.getCanonicalType(T2).getUnqualifiedType()
          == Context.getCanonicalType(ArgType).getUnqualifiedType())
      return true;
  }

  return false;
}

/// \brief Find the protocol with the given name, if any.
ObjCProtocolDecl *Sema::LookupProtocol(IdentifierInfo *II) {
  Decl *D = LookupSingleName(TUScope, II, LookupObjCProtocolName);
  return cast_or_null<ObjCProtocolDecl>(D);
}

/// \brief Find the Objective-C category implementation with the given
/// name, if any.
ObjCCategoryImplDecl *Sema::LookupObjCCategoryImpl(IdentifierInfo *II) {
  Decl *D = LookupSingleName(TUScope, II, LookupObjCCategoryImplName);
  return cast_or_null<ObjCCategoryImplDecl>(D);
}

void Sema::LookupOverloadedOperatorName(OverloadedOperatorKind Op, Scope *S,
                                        QualType T1, QualType T2,
                                        FunctionSet &Functions) {
  // C++ [over.match.oper]p3:
  //     -- The set of non-member candidates is the result of the
  //        unqualified lookup of operator@ in the context of the
  //        expression according to the usual rules for name lookup in
  //        unqualified function calls (3.4.2) except that all member
  //        functions are ignored. However, if no operand has a class
  //        type, only those non-member functions in the lookup set
  //        that have a first parameter of type T1 or "reference to
  //        (possibly cv-qualified) T1", when T1 is an enumeration
  //        type, or (if there is a right operand) a second parameter
  //        of type T2 or "reference to (possibly cv-qualified) T2",
  //        when T2 is an enumeration type, are candidate functions.
  DeclarationName OpName = Context.DeclarationNames.getCXXOperatorName(Op);
  LookupResult Operators;
  LookupName(Operators, S, OpName, LookupOperatorName);

  assert(!Operators.isAmbiguous() && "Operator lookup cannot be ambiguous");

  if (Operators.empty())
    return;

  for (LookupResult::iterator Op = Operators.begin(), OpEnd = Operators.end();
       Op != OpEnd; ++Op) {
    if (FunctionDecl *FD = dyn_cast<FunctionDecl>(*Op)) {
      if (IsAcceptableNonMemberOperatorCandidate(FD, T1, T2, Context))
        Functions.insert(FD); // FIXME: canonical FD
    } else if (FunctionTemplateDecl *FunTmpl
                 = dyn_cast<FunctionTemplateDecl>(*Op)) {
      // FIXME: friend operators?
      // FIXME: do we need to check IsAcceptableNonMemberOperatorCandidate,
      // later?
      if (!FunTmpl->getDeclContext()->isRecord())
        Functions.insert(FunTmpl);
    }
  }
}

static void CollectFunctionDecl(Sema::FunctionSet &Functions,
                                Decl *D) {
  if (FunctionDecl *Func = dyn_cast<FunctionDecl>(D))
    Functions.insert(Func);
  else if (FunctionTemplateDecl *FunTmpl = dyn_cast<FunctionTemplateDecl>(D))
    Functions.insert(FunTmpl);
}

void Sema::ArgumentDependentLookup(DeclarationName Name, bool Operator,
                                   Expr **Args, unsigned NumArgs,
                                   FunctionSet &Functions) {
  // Find all of the associated namespaces and classes based on the
  // arguments we have.
  AssociatedNamespaceSet AssociatedNamespaces;
  AssociatedClassSet AssociatedClasses;
  FindAssociatedClassesAndNamespaces(Args, NumArgs,
                                     AssociatedNamespaces,
                                     AssociatedClasses);

  QualType T1, T2;
  if (Operator) {
    T1 = Args[0]->getType();
    if (NumArgs >= 2)
      T2 = Args[1]->getType();
  }

  // C++ [basic.lookup.argdep]p3:
  //   Let X be the lookup set produced by unqualified lookup (3.4.1)
  //   and let Y be the lookup set produced by argument dependent
  //   lookup (defined as follows). If X contains [...] then Y is
  //   empty. Otherwise Y is the set of declarations found in the
  //   namespaces associated with the argument types as described
  //   below. The set of declarations found by the lookup of the name
  //   is the union of X and Y.
  //
  // Here, we compute Y and add its members to the overloaded
  // candidate set.
  for (AssociatedNamespaceSet::iterator NS = AssociatedNamespaces.begin(),
                                     NSEnd = AssociatedNamespaces.end();
       NS != NSEnd; ++NS) {
    //   When considering an associated namespace, the lookup is the
    //   same as the lookup performed when the associated namespace is
    //   used as a qualifier (3.4.3.2) except that:
    //
    //     -- Any using-directives in the associated namespace are
    //        ignored.
    //
    //     -- Any namespace-scope friend functions declared in
    //        associated classes are visible within their respective
    //        namespaces even if they are not visible during an ordinary
    //        lookup (11.4).
    DeclContext::lookup_iterator I, E;
    for (llvm::tie(I, E) = (*NS)->lookup(Name); I != E; ++I) {
      Decl *D = *I;
      // If the only declaration here is an ordinary friend, consider
      // it only if it was declared in an associated classes.
      if (D->getIdentifierNamespace() == Decl::IDNS_OrdinaryFriend) {
        DeclContext *LexDC = D->getLexicalDeclContext();
        if (!AssociatedClasses.count(cast<CXXRecordDecl>(LexDC)))
          continue;
      }

      FunctionDecl *Fn;
      if (!Operator || !(Fn = dyn_cast<FunctionDecl>(D)) ||
          IsAcceptableNonMemberOperatorCandidate(Fn, T1, T2, Context))
        CollectFunctionDecl(Functions, D);
    }
  }
}
