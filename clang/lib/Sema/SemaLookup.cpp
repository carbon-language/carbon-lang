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
#include "SemaInherit.h"
#include "clang/AST/ASTContext.h"
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
#include <set>
#include <vector>
#include <iterator>
#include <utility>
#include <algorithm>

using namespace clang;

typedef llvm::SmallVector<UsingDirectiveDecl*, 4> UsingDirectivesTy;
typedef llvm::DenseSet<NamespaceDecl*> NamespaceSet;
typedef llvm::SmallVector<Sema::LookupResult, 3> LookupResultsTy;

/// UsingDirAncestorCompare - Implements strict weak ordering of
/// UsingDirectives. It orders them by address of its common ancestor.
struct UsingDirAncestorCompare {

  /// @brief Compares UsingDirectiveDecl common ancestor with DeclContext.
  bool operator () (UsingDirectiveDecl *U, const DeclContext *Ctx) const {
    return U->getCommonAncestor() < Ctx;
  }

  /// @brief Compares UsingDirectiveDecl common ancestor with DeclContext.
  bool operator () (const DeclContext *Ctx, UsingDirectiveDecl *U) const {
    return Ctx < U->getCommonAncestor();
  }

  /// @brief Compares UsingDirectiveDecl common ancestors.
  bool operator () (UsingDirectiveDecl *U1, UsingDirectiveDecl *U2) const {
    return U1->getCommonAncestor() < U2->getCommonAncestor();
  }
};

/// AddNamespaceUsingDirectives - Adds all UsingDirectiveDecl's to heap UDirs
/// (ordered by common ancestors), found in namespace NS,
/// including all found (recursively) in their nominated namespaces.
void AddNamespaceUsingDirectives(ASTContext &Context, 
                                 DeclContext *NS,
                                 UsingDirectivesTy &UDirs,
                                 NamespaceSet &Visited) {
  DeclContext::udir_iterator I, End;

  for (llvm::tie(I, End) = NS->getUsingDirectives(); I !=End; ++I) {
    UDirs.push_back(*I);
    std::push_heap(UDirs.begin(), UDirs.end(), UsingDirAncestorCompare());
    NamespaceDecl *Nominated = (*I)->getNominatedNamespace();
    if (Visited.insert(Nominated).second)
      AddNamespaceUsingDirectives(Context, Nominated, UDirs, /*ref*/ Visited);
  }
}

/// AddScopeUsingDirectives - Adds all UsingDirectiveDecl's found in Scope S,
/// including all found in the namespaces they nominate.
static void AddScopeUsingDirectives(ASTContext &Context, Scope *S, 
                                    UsingDirectivesTy &UDirs) {
  NamespaceSet VisitedNS;

  if (DeclContext *Ctx = static_cast<DeclContext*>(S->getEntity())) {

    if (NamespaceDecl *NS = dyn_cast<NamespaceDecl>(Ctx))
      VisitedNS.insert(NS);

    AddNamespaceUsingDirectives(Context, Ctx, UDirs, /*ref*/ VisitedNS);

  } else {
    Scope::udir_iterator I = S->using_directives_begin(),
                         End = S->using_directives_end();

    for (; I != End; ++I) {
      UsingDirectiveDecl *UD = I->getAs<UsingDirectiveDecl>();
      UDirs.push_back(UD);
      std::push_heap(UDirs.begin(), UDirs.end(), UsingDirAncestorCompare());

      NamespaceDecl *Nominated = UD->getNominatedNamespace();
      if (!VisitedNS.count(Nominated)) {
        VisitedNS.insert(Nominated);
        AddNamespaceUsingDirectives(Context, Nominated, UDirs, 
                                    /*ref*/ VisitedNS);
      }
    }
  }
}

/// MaybeConstructOverloadSet - Name lookup has determined that the
/// elements in [I, IEnd) have the name that we are looking for, and
/// *I is a match for the namespace. This routine returns an
/// appropriate Decl for name lookup, which may either be *I or an
/// OverloadedFunctionDecl that represents the overloaded functions in
/// [I, IEnd). 
///
/// The existance of this routine is temporary; users of LookupResult
/// should be able to handle multiple results, to deal with cases of
/// ambiguity and overloaded functions without needing to create a
/// Decl node.
template<typename DeclIterator>
static NamedDecl *
MaybeConstructOverloadSet(ASTContext &Context, 
                          DeclIterator I, DeclIterator IEnd) {
  assert(I != IEnd && "Iterator range cannot be empty");
  assert(!isa<OverloadedFunctionDecl>(*I) && 
         "Cannot have an overloaded function");

  if ((*I)->isFunctionOrFunctionTemplate()) {
    // If we found a function, there might be more functions. If
    // so, collect them into an overload set.
    DeclIterator Last = I;
    OverloadedFunctionDecl *Ovl = 0;
    for (++Last; 
         Last != IEnd && (*Last)->isFunctionOrFunctionTemplate(); 
         ++Last) {
      if (!Ovl) {
        // FIXME: We leak this overload set. Eventually, we want to stop
        // building the declarations for these overload sets, so there will be
        // nothing to leak.
        Ovl = OverloadedFunctionDecl::Create(Context, (*I)->getDeclContext(),
                                             (*I)->getDeclName());
        NamedDecl *ND = (*I)->getUnderlyingDecl();
        if (isa<FunctionDecl>(ND))
          Ovl->addOverload(cast<FunctionDecl>(ND));
        else
          Ovl->addOverload(cast<FunctionTemplateDecl>(ND));
      }

      NamedDecl *ND = (*Last)->getUnderlyingDecl();
      if (isa<FunctionDecl>(ND))
        Ovl->addOverload(cast<FunctionDecl>(ND));
      else
        Ovl->addOverload(cast<FunctionTemplateDecl>(ND));
    }
    
    // If we had more than one function, we built an overload
    // set. Return it.
    if (Ovl)
      return Ovl;
  }
  
  return *I;
}

/// Merges together multiple LookupResults dealing with duplicated Decl's.
static Sema::LookupResult
MergeLookupResults(ASTContext &Context, LookupResultsTy &Results) {
  typedef Sema::LookupResult LResult;
  typedef llvm::SmallPtrSet<NamedDecl*, 4> DeclsSetTy;

  // Remove duplicated Decl pointing at same Decl, by storing them in
  // associative collection. This might be case for code like:
  //
  //    namespace A { int i; }
  //    namespace B { using namespace A; }
  //    namespace C { using namespace A; }
  //
  //    void foo() {
  //      using namespace B;
  //      using namespace C;
  //      ++i; // finds A::i, from both namespace B and C at global scope
  //    }
  //
  //  C++ [namespace.qual].p3:
  //    The same declaration found more than once is not an ambiguity
  //    (because it is still a unique declaration).
  DeclsSetTy FoundDecls;

  // Counter of tag names, and functions for resolving ambiguity
  // and name hiding.
  std::size_t TagNames = 0, Functions = 0, OrdinaryNonFunc = 0;

  LookupResultsTy::iterator I = Results.begin(), End = Results.end();

  // No name lookup results, return early.
  if (I == End) return LResult::CreateLookupResult(Context, 0);

  // Keep track of the tag declaration we found. We only use this if
  // we find a single tag declaration.
  TagDecl *TagFound = 0;

  for (; I != End; ++I) {
    switch (I->getKind()) {
    case LResult::NotFound:
      assert(false &&
             "Should be always successful name lookup result here.");
      break;

    case LResult::AmbiguousReference:
    case LResult::AmbiguousBaseSubobjectTypes:
    case LResult::AmbiguousBaseSubobjects:
      assert(false && "Shouldn't get ambiguous lookup here.");
      break;

    case LResult::Found: {
      NamedDecl *ND = I->getAsDecl()->getUnderlyingDecl();
        
      if (TagDecl *TD = dyn_cast<TagDecl>(ND)) {
        TagFound = TD->getCanonicalDecl();
        TagNames += FoundDecls.insert(TagFound)?  1 : 0;
      } else if (ND->isFunctionOrFunctionTemplate())
        Functions += FoundDecls.insert(ND)? 1 : 0;
      else
        FoundDecls.insert(ND);
      break;
    }

    case LResult::FoundOverloaded:
      for (LResult::iterator FI = I->begin(), FEnd = I->end(); FI != FEnd; ++FI)
        Functions += FoundDecls.insert(*FI)? 1 : 0;
      break;
    }
  }
  OrdinaryNonFunc = FoundDecls.size() - TagNames - Functions;
  bool Ambiguous = false, NameHidesTags = false;
  
  if (FoundDecls.size() == 1) {
    // 1) Exactly one result.
  } else if (TagNames > 1) {
    // 2) Multiple tag names (even though they may be hidden by an
    // object name).
    Ambiguous = true;
  } else if (FoundDecls.size() - TagNames == 1) {
    // 3) Ordinary name hides (optional) tag.
    NameHidesTags = TagFound;
  } else if (Functions) {
    // C++ [basic.lookup].p1:
    // ... Name lookup may associate more than one declaration with
    // a name if it finds the name to be a function name; the declarations
    // are said to form a set of overloaded functions (13.1).
    // Overload resolution (13.3) takes place after name lookup has succeeded.
    //
    if (!OrdinaryNonFunc) {
      // 4) Functions hide tag names.
      NameHidesTags = TagFound;
    } else {
      // 5) Functions + ordinary names.
      Ambiguous = true;
    }
  } else {
    // 6) Multiple non-tag names
    Ambiguous = true;
  }

  if (Ambiguous)
    return LResult::CreateLookupResult(Context, 
                                       FoundDecls.begin(), FoundDecls.size());
  if (NameHidesTags) {
    // There's only one tag, TagFound. Remove it.
    assert(TagFound && FoundDecls.count(TagFound) && "No tag name found?");
    FoundDecls.erase(TagFound);
  }

  // Return successful name lookup result.
  return LResult::CreateLookupResult(Context,
                                MaybeConstructOverloadSet(Context,
                                                          FoundDecls.begin(),
                                                          FoundDecls.end()));
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

Sema::LookupResult
Sema::LookupResult::CreateLookupResult(ASTContext &Context, NamedDecl *D) {
  if (D)
    D = D->getUnderlyingDecl();
  
  LookupResult Result;
  Result.StoredKind = (D && isa<OverloadedFunctionDecl>(D))?
    OverloadedDeclSingleDecl : SingleDecl;
  Result.First = reinterpret_cast<uintptr_t>(D);
  Result.Last = 0;
  Result.Context = &Context;
  return Result;
}

/// @brief Moves the name-lookup results from Other to this LookupResult.
Sema::LookupResult
Sema::LookupResult::CreateLookupResult(ASTContext &Context, 
                                       IdentifierResolver::iterator F, 
                                       IdentifierResolver::iterator L) {
  LookupResult Result;
  Result.Context = &Context;

  if (F != L && (*F)->isFunctionOrFunctionTemplate()) {
    IdentifierResolver::iterator Next = F;
    ++Next;
    if (Next != L && (*Next)->isFunctionOrFunctionTemplate()) {
      Result.StoredKind = OverloadedDeclFromIdResolver;
      Result.First = F.getAsOpaqueValue();
      Result.Last = L.getAsOpaqueValue();
      return Result;
    }
  } 

  NamedDecl *D = *F;
  if (D)
    D = D->getUnderlyingDecl();

  Result.StoredKind = SingleDecl;
  Result.First = reinterpret_cast<uintptr_t>(D);
  Result.Last = 0;
  return Result;
}

Sema::LookupResult
Sema::LookupResult::CreateLookupResult(ASTContext &Context, 
                                       DeclContext::lookup_iterator F, 
                                       DeclContext::lookup_iterator L) {
  LookupResult Result;
  Result.Context = &Context;

  if (F != L && (*F)->isFunctionOrFunctionTemplate()) {
    DeclContext::lookup_iterator Next = F;
    ++Next;
    if (Next != L && (*Next)->isFunctionOrFunctionTemplate()) {
      Result.StoredKind = OverloadedDeclFromDeclContext;
      Result.First = reinterpret_cast<uintptr_t>(F);
      Result.Last = reinterpret_cast<uintptr_t>(L);
      return Result;
    }
  }

  NamedDecl *D = *F;
  if (D)
    D = D->getUnderlyingDecl();
  
  Result.StoredKind = SingleDecl;
  Result.First = reinterpret_cast<uintptr_t>(D);
  Result.Last = 0;
  return Result;
}

/// @brief Determine the result of name lookup.
Sema::LookupResult::LookupKind Sema::LookupResult::getKind() const {
  switch (StoredKind) {
  case SingleDecl:
    return (reinterpret_cast<Decl *>(First) != 0)? Found : NotFound;

  case OverloadedDeclSingleDecl:
  case OverloadedDeclFromIdResolver:
  case OverloadedDeclFromDeclContext:
    return FoundOverloaded;

  case AmbiguousLookupStoresBasePaths:
    return Last? AmbiguousBaseSubobjectTypes : AmbiguousBaseSubobjects;

  case AmbiguousLookupStoresDecls:
    return AmbiguousReference;
  }

  // We can't ever get here.
  return NotFound;
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
NamedDecl *Sema::LookupResult::getAsDecl() const {
  switch (StoredKind) {
  case SingleDecl:
    return reinterpret_cast<NamedDecl *>(First);

  case OverloadedDeclFromIdResolver:
    return MaybeConstructOverloadSet(*Context,
                         IdentifierResolver::iterator::getFromOpaqueValue(First),
                         IdentifierResolver::iterator::getFromOpaqueValue(Last));

  case OverloadedDeclFromDeclContext:
    return MaybeConstructOverloadSet(*Context, 
                           reinterpret_cast<DeclContext::lookup_iterator>(First),
                           reinterpret_cast<DeclContext::lookup_iterator>(Last));

  case OverloadedDeclSingleDecl:
    return reinterpret_cast<OverloadedFunctionDecl*>(First);

  case AmbiguousLookupStoresDecls:
  case AmbiguousLookupStoresBasePaths:
    assert(false && 
           "Name lookup returned an ambiguity that could not be handled");
    break;
  }

  return 0;
}

/// @brief Retrieves the BasePaths structure describing an ambiguous
/// name lookup, or null.
BasePaths *Sema::LookupResult::getBasePaths() const {
  if (StoredKind == AmbiguousLookupStoresBasePaths)
      return reinterpret_cast<BasePaths *>(First);
  return 0;
}

Sema::LookupResult::iterator::reference 
Sema::LookupResult::iterator::operator*() const {
  switch (Result->StoredKind) {
  case SingleDecl:
    return reinterpret_cast<NamedDecl*>(Current);

  case OverloadedDeclSingleDecl:
    return *reinterpret_cast<NamedDecl**>(Current);

  case OverloadedDeclFromIdResolver:
    return *IdentifierResolver::iterator::getFromOpaqueValue(Current);

  case AmbiguousLookupStoresBasePaths:
    if (Result->Last)
      return *reinterpret_cast<NamedDecl**>(Current);

    // Fall through to handle the DeclContext::lookup_iterator we're
    // storing.

  case OverloadedDeclFromDeclContext:
  case AmbiguousLookupStoresDecls:
    return *reinterpret_cast<DeclContext::lookup_iterator>(Current);
  }

  return 0;
}

Sema::LookupResult::iterator& Sema::LookupResult::iterator::operator++() {
  switch (Result->StoredKind) {
  case SingleDecl:
    Current = reinterpret_cast<uintptr_t>((NamedDecl*)0);
    break;

  case OverloadedDeclSingleDecl: {
    NamedDecl ** I = reinterpret_cast<NamedDecl**>(Current);
    ++I;
    Current = reinterpret_cast<uintptr_t>(I);
    break;
  }

  case OverloadedDeclFromIdResolver: {
    IdentifierResolver::iterator I 
      = IdentifierResolver::iterator::getFromOpaqueValue(Current);
    ++I;
    Current = I.getAsOpaqueValue();
    break;
  }

  case AmbiguousLookupStoresBasePaths: 
    if (Result->Last) {
      NamedDecl ** I = reinterpret_cast<NamedDecl**>(Current);
      ++I;
      Current = reinterpret_cast<uintptr_t>(I);
      break;
    }
    // Fall through to handle the DeclContext::lookup_iterator we're
    // storing.

  case OverloadedDeclFromDeclContext:
  case AmbiguousLookupStoresDecls: {
    DeclContext::lookup_iterator I 
      = reinterpret_cast<DeclContext::lookup_iterator>(Current);
    ++I;
    Current = reinterpret_cast<uintptr_t>(I);
    break;
  }
  }

  return *this;
}

Sema::LookupResult::iterator Sema::LookupResult::begin() {
  switch (StoredKind) {
  case SingleDecl:
  case OverloadedDeclFromIdResolver:
  case OverloadedDeclFromDeclContext:
  case AmbiguousLookupStoresDecls:
    return iterator(this, First);

  case OverloadedDeclSingleDecl: {
    OverloadedFunctionDecl * Ovl =
      reinterpret_cast<OverloadedFunctionDecl*>(First);
    return iterator(this, 
                    reinterpret_cast<uintptr_t>(&(*Ovl->function_begin())));
  }

  case AmbiguousLookupStoresBasePaths:
    if (Last)
      return iterator(this, 
              reinterpret_cast<uintptr_t>(getBasePaths()->found_decls_begin()));
    else
      return iterator(this,
              reinterpret_cast<uintptr_t>(getBasePaths()->front().Decls.first));
  }

  // Required to suppress GCC warning.
  return iterator();
}

Sema::LookupResult::iterator Sema::LookupResult::end() {
  switch (StoredKind) {
  case SingleDecl:
  case OverloadedDeclFromIdResolver:
  case OverloadedDeclFromDeclContext:
  case AmbiguousLookupStoresDecls:
    return iterator(this, Last);

  case OverloadedDeclSingleDecl: {
    OverloadedFunctionDecl * Ovl =
      reinterpret_cast<OverloadedFunctionDecl*>(First);
    return iterator(this, 
                    reinterpret_cast<uintptr_t>(&(*Ovl->function_end())));
  }

  case AmbiguousLookupStoresBasePaths:
    if (Last)
      return iterator(this, 
               reinterpret_cast<uintptr_t>(getBasePaths()->found_decls_end()));
    else
      return iterator(this, reinterpret_cast<uintptr_t>(
                                     getBasePaths()->front().Decls.second));
  }

  // Required to suppress GCC warning.
  return iterator();
}

void Sema::LookupResult::Destroy() {
  if (BasePaths *Paths = getBasePaths())
    delete Paths;
  else if (getKind() == AmbiguousReference)
    delete[] reinterpret_cast<NamedDecl **>(First);
}

static void
CppNamespaceLookup(ASTContext &Context, DeclContext *NS,
                   DeclarationName Name, Sema::LookupNameKind NameKind,
                   unsigned IDNS, LookupResultsTy &Results,
                   UsingDirectivesTy *UDirs = 0) {

  assert(NS && NS->isFileContext() && "CppNamespaceLookup() requires namespace!");

  // Perform qualified name lookup into the LookupCtx.
  DeclContext::lookup_iterator I, E;
  for (llvm::tie(I, E) = NS->lookup(Name); I != E; ++I)
    if (Sema::isAcceptableLookupResult(*I, NameKind, IDNS)) {
      Results.push_back(Sema::LookupResult::CreateLookupResult(Context, I, E));
      break;
    }

  if (UDirs) {
    // For each UsingDirectiveDecl, which common ancestor is equal
    // to NS, we preform qualified name lookup into namespace nominated by it.
    UsingDirectivesTy::const_iterator UI, UEnd;
    llvm::tie(UI, UEnd) =
      std::equal_range(UDirs->begin(), UDirs->end(), NS,
                       UsingDirAncestorCompare());
 
    for (; UI != UEnd; ++UI)
      CppNamespaceLookup(Context, (*UI)->getNominatedNamespace(),
                         Name, NameKind, IDNS, Results);
  }
}

static bool isNamespaceOrTranslationUnitScope(Scope *S) {
  if (DeclContext *Ctx = static_cast<DeclContext*>(S->getEntity()))
    return Ctx->isFileContext();
  return false;
}

std::pair<bool, Sema::LookupResult>
Sema::CppLookupName(Scope *S, DeclarationName Name,
                    LookupNameKind NameKind, bool RedeclarationOnly) {
  assert(getLangOptions().CPlusPlus &&
         "Can perform only C++ lookup");
  unsigned IDNS 
    = getIdentifierNamespacesFromLookupNameKind(NameKind, /*CPlusPlus*/ true);
  Scope *Initial = S;
  DeclContext *OutOfLineCtx = 0;
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
    for (; I != IEnd && S->isDeclScope(DeclPtrTy::make(*I)); ++I) {
      if (isAcceptableLookupResult(*I, NameKind, IDNS)) {
        // We found something.  Look for anything else in our scope
        // with this same name and in an acceptable identifier
        // namespace, so that we can construct an overload set if we
        // need to.
        IdentifierResolver::iterator LastI = I;
        for (++LastI; LastI != IEnd; ++LastI) {
          if (!S->isDeclScope(DeclPtrTy::make(*LastI)))
            break;
        }
        LookupResult Result =
          LookupResult::CreateLookupResult(Context, I, LastI);
        return std::make_pair(true, Result);
      }
    }
    if (DeclContext *Ctx = static_cast<DeclContext*>(S->getEntity())) {
      LookupResult R;
      // Perform member lookup into struct.
      // FIXME: In some cases, we know that every name that could be found by
      // this qualified name lookup will also be on the identifier chain. For
      // example, inside a class without any base classes, we never need to
      // perform qualified lookup because all of the members are on top of the
      // identifier chain.
      if (isa<RecordDecl>(Ctx)) {
        R = LookupQualifiedName(Ctx, Name, NameKind, RedeclarationOnly);
        if (R)
          return std::make_pair(true, R);
      }
      if (Ctx->getParent() != Ctx->getLexicalParent() 
          || isa<CXXMethodDecl>(Ctx)) {
        // It is out of line defined C++ method or struct, we continue
        // doing name lookup in parent context. Once we will find namespace
        // or translation-unit we save it for possible checking
        // using-directives later.
        for (OutOfLineCtx = Ctx; OutOfLineCtx && !OutOfLineCtx->isFileContext();
             OutOfLineCtx = OutOfLineCtx->getParent()) {
          R = LookupQualifiedName(OutOfLineCtx, Name, NameKind, RedeclarationOnly);
          if (R)
            return std::make_pair(true, R);
        }
      }
    }
  }

  // Collect UsingDirectiveDecls in all scopes, and recursively all
  // nominated namespaces by those using-directives.
  // UsingDirectives are pushed to heap, in common ancestor pointer value order.
  // FIXME: Cache this sorted list in Scope structure, and DeclContext, so we
  // don't build it for each lookup!
  UsingDirectivesTy UDirs;
  for (Scope *SC = Initial; SC; SC = SC->getParent())
    if (SC->getFlags() & Scope::DeclScope)
      AddScopeUsingDirectives(Context, SC, UDirs);

  // Sort heapified UsingDirectiveDecls.
  std::sort_heap(UDirs.begin(), UDirs.end(), UsingDirAncestorCompare());

  // Lookup namespace scope, and global scope.
  // Unqualified name lookup in C++ requires looking into scopes
  // that aren't strictly lexical, and therefore we walk through the
  // context as well as walking through the scopes.

  LookupResultsTy LookupResults;
  assert((!OutOfLineCtx || OutOfLineCtx->isFileContext()) &&
         "We should have been looking only at file context here already.");
  bool LookedInCtx = false;
  LookupResult Result;
  while (OutOfLineCtx &&
         OutOfLineCtx != S->getEntity() &&
         OutOfLineCtx->isNamespace()) {
    LookedInCtx = true;

    // Look into context considering using-directives.
    CppNamespaceLookup(Context, OutOfLineCtx, Name, NameKind, IDNS,
                       LookupResults, &UDirs);

    if ((Result = MergeLookupResults(Context, LookupResults)) ||
        (RedeclarationOnly && !OutOfLineCtx->isTransparentContext()))
      return std::make_pair(true, Result);

    OutOfLineCtx = OutOfLineCtx->getParent();
  }

  for (; S; S = S->getParent()) {
    DeclContext *Ctx = static_cast<DeclContext *>(S->getEntity());
    if (Ctx->isTransparentContext())
      continue;

    assert(Ctx && Ctx->isFileContext() &&
           "We should have been looking only at file context here already.");

    // Check whether the IdResolver has anything in this scope.
    for (; I != IEnd && S->isDeclScope(DeclPtrTy::make(*I)); ++I) {
      if (isAcceptableLookupResult(*I, NameKind, IDNS)) {
        // We found something.  Look for anything else in our scope
        // with this same name and in an acceptable identifier
        // namespace, so that we can construct an overload set if we
        // need to.
        IdentifierResolver::iterator LastI = I;
        for (++LastI; LastI != IEnd; ++LastI) {
          if (!S->isDeclScope(DeclPtrTy::make(*LastI)))
            break;
        }
        
        // We store name lookup result, and continue trying to look into
        // associated context, and maybe namespaces nominated by
        // using-directives.
        LookupResults.push_back(
          LookupResult::CreateLookupResult(Context, I, LastI));
        break;
      }
    }

    LookedInCtx = true;
    // Look into context considering using-directives.
    CppNamespaceLookup(Context, Ctx, Name, NameKind, IDNS,
                       LookupResults, &UDirs);

    if ((Result = MergeLookupResults(Context, LookupResults)) ||
        (RedeclarationOnly && !Ctx->isTransparentContext()))
      return std::make_pair(true, Result);
  }

  if (!(LookedInCtx || LookupResults.empty())) {
    // We didn't Performed lookup in Scope entity, so we return
    // result form IdentifierResolver.
    assert((LookupResults.size() == 1) && "Wrong size!");
    return std::make_pair(true, LookupResults.front());
  }
  return std::make_pair(false, LookupResult());
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
Sema::LookupResult 
Sema::LookupName(Scope *S, DeclarationName Name, LookupNameKind NameKind,
                 bool RedeclarationOnly, bool AllowBuiltinCreation,
                 SourceLocation Loc) {
  if (!Name) return LookupResult::CreateLookupResult(Context, 0);

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
          }

          return LookupResult::CreateLookupResult(Context, I, LastI);
        }

        // We have a single lookup result.
        return LookupResult::CreateLookupResult(Context, *I);
      }
  } else {
    // Perform C++ unqualified name lookup.
    std::pair<bool, LookupResult> MaybeResult =
      CppLookupName(S, Name, NameKind, RedeclarationOnly);
    if (MaybeResult.first)
      return MaybeResult.second;
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
          return LookupResult::CreateLookupResult(Context, 0);

        return LookupResult::CreateLookupResult(Context,
                            LazilyCreateBuiltin((IdentifierInfo *)II, BuiltinID,
                                                S, RedeclarationOnly, Loc));
      }
    }
  }
  return LookupResult::CreateLookupResult(Context, 0);
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
                          LookupNameKind NameKind, bool RedeclarationOnly) {
  assert(LookupCtx && "Sema::LookupQualifiedName requires a lookup context");
  
  if (!Name) return LookupResult::CreateLookupResult(Context, 0);

  // If we're performing qualified name lookup (e.g., lookup into a
  // struct), find fields as part of ordinary name lookup.
  unsigned IDNS
    = getIdentifierNamespacesFromLookupNameKind(NameKind, 
                                                getLangOptions().CPlusPlus);
  if (NameKind == LookupOrdinaryName)
    IDNS |= Decl::IDNS_Member;

  // Perform qualified name lookup into the LookupCtx.
  DeclContext::lookup_iterator I, E;
  for (llvm::tie(I, E) = LookupCtx->lookup(Name); I != E; ++I)
    if (isAcceptableLookupResult(*I, NameKind, IDNS))
      return LookupResult::CreateLookupResult(Context, I, E);

  // If this isn't a C++ class or we aren't allowed to look into base
  // classes, we're done.
  if (RedeclarationOnly || !isa<CXXRecordDecl>(LookupCtx))
    return LookupResult::CreateLookupResult(Context, 0);

  // Perform lookup into our base classes.
  BasePaths Paths;
  Paths.setOrigin(Context.getTypeDeclType(cast<RecordDecl>(LookupCtx)));

  // Look for this member in our base classes
  if (!LookupInBases(cast<CXXRecordDecl>(LookupCtx), 
                     MemberLookupCriteria(Name, NameKind, IDNS), Paths))
    return LookupResult::CreateLookupResult(Context, 0);

  // C++ [class.member.lookup]p2:
  //   [...] If the resulting set of declarations are not all from
  //   sub-objects of the same type, or the set has a nonstatic member
  //   and includes members from distinct sub-objects, there is an
  //   ambiguity and the program is ill-formed. Otherwise that set is
  //   the result of the lookup.
  // FIXME: support using declarations!
  QualType SubobjectType;
  int SubobjectNumber = 0;
  for (BasePaths::paths_iterator Path = Paths.begin(), PathEnd = Paths.end();
       Path != PathEnd; ++Path) {
    const BasePathElement &PathElement = Path->back();

    // Determine whether we're looking at a distinct sub-object or not.
    if (SubobjectType.isNull()) {
      // This is the first subobject we've looked at. Record it's type.
      SubobjectType = Context.getCanonicalType(PathElement.Base->getType());
      SubobjectNumber = PathElement.SubobjectNumber;
    } else if (SubobjectType 
                 != Context.getCanonicalType(PathElement.Base->getType())) {
      // We found members of the given name in two subobjects of
      // different types. This lookup is ambiguous.
      BasePaths *PathsOnHeap = new BasePaths;
      PathsOnHeap->swap(Paths);
      return LookupResult::CreateLookupResult(Context, PathsOnHeap, true);
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
      BasePaths *PathsOnHeap = new BasePaths;
      PathsOnHeap->swap(Paths);
      return LookupResult::CreateLookupResult(Context, PathsOnHeap, false);
    }
  }

  // Lookup in a base class succeeded; return these results.

  // If we found a function declaration, return an overload set.
  if ((*Paths.front().Decls.first)->isFunctionOrFunctionTemplate())
    return LookupResult::CreateLookupResult(Context, 
                        Paths.front().Decls.first, Paths.front().Decls.second);

  // We found a non-function declaration; return a single declaration.
  return LookupResult::CreateLookupResult(Context, *Paths.front().Decls.first);
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
/// @param Loc      If provided, the source location where we're performing
/// name lookup. At present, this is only used to produce diagnostics when 
/// C library functions (like "malloc") are implicitly declared.
///
/// @returns The result of qualified or unqualified name lookup.
Sema::LookupResult
Sema::LookupParsedName(Scope *S, const CXXScopeSpec *SS, 
                       DeclarationName Name, LookupNameKind NameKind,
                       bool RedeclarationOnly, bool AllowBuiltinCreation,
                       SourceLocation Loc) {
  if (SS && (SS->isSet() || SS->isInvalid())) {
    // If the scope specifier is invalid, don't even look for
    // anything.
    if (SS->isInvalid())
      return LookupResult::CreateLookupResult(Context, 0);

    assert(!isUnknownSpecialization(*SS) && "Can't lookup dependent types");

    if (isDependentScopeSpecifier(*SS)) {
      // Determine whether we are looking into the current
      // instantiation. 
      NestedNameSpecifier *NNS 
        = static_cast<NestedNameSpecifier *>(SS->getScopeRep());
      CXXRecordDecl *Current = getCurrentInstantiationOf(NNS);
      assert(Current && "Bad dependent scope specifier");
      
      // We nested name specifier refers to the current instantiation,
      // so now we will look for a member of the current instantiation
      // (C++0x [temp.dep.type]).
      unsigned IDNS = getIdentifierNamespacesFromLookupNameKind(NameKind, true);
      DeclContext::lookup_iterator I, E;
      for (llvm::tie(I, E) = Current->lookup(Name); I != E; ++I)
        if (isAcceptableLookupResult(*I, NameKind, IDNS))
          return LookupResult::CreateLookupResult(Context, I, E);
    }

    if (RequireCompleteDeclContext(*SS))
      return LookupResult::CreateLookupResult(Context, 0);

    return LookupQualifiedName(computeDeclContext(*SS),
                               Name, NameKind, RedeclarationOnly);
  }

  LookupResult result(LookupName(S, Name, NameKind, RedeclarationOnly, 
                    AllowBuiltinCreation, Loc));
  
  return(result);
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

  if (BasePaths *Paths = Result.getBasePaths()) {
    if (Result.getKind() == LookupResult::AmbiguousBaseSubobjects) {
      QualType SubobjectType = Paths->front().back().Base->getType();
      Diag(NameLoc, diag::err_ambiguous_member_multiple_subobjects)
        << Name << SubobjectType << getAmbiguousPathsDisplayString(*Paths)
        << LookupRange;

      DeclContext::lookup_iterator Found = Paths->front().Decls.first;
      while (isa<CXXMethodDecl>(*Found) && 
             cast<CXXMethodDecl>(*Found)->isStatic())
        ++Found;

      Diag((*Found)->getLocation(), diag::note_ambiguous_member_found);

      Result.Destroy();
      return true;
    } 

    assert(Result.getKind() == LookupResult::AmbiguousBaseSubobjectTypes &&
           "Unhandled form of name lookup ambiguity");

    Diag(NameLoc, diag::err_ambiguous_member_multiple_subobject_types)
      << Name << LookupRange;

    std::set<Decl *> DeclsPrinted;
    for (BasePaths::paths_iterator Path = Paths->begin(), PathEnd = Paths->end();
         Path != PathEnd; ++Path) {
      Decl *D = *Path->Decls.first;
      if (DeclsPrinted.insert(D).second)
        Diag(D->getLocation(), diag::note_ambiguous_member_found);
    }

    Result.Destroy();
    return true;
  } else if (Result.getKind() == LookupResult::AmbiguousReference) {
    Diag(NameLoc, diag::err_ambiguous_reference) << Name << LookupRange;

    NamedDecl **DI = reinterpret_cast<NamedDecl **>(Result.First),
            **DEnd = reinterpret_cast<NamedDecl **>(Result.Last);

    for (; DI != DEnd; ++DI)
      Diag((*DI)->getLocation(), diag::note_ambiguous_candidate) << *DI;

    Result.Destroy();
    return true;
  }

  assert(false && "Unhandled form of name lookup ambiguity");

  // We can't reach here.
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
      
    case TemplateArgument::Declaration:
      // [...] the namespaces in which any template template arguments are 
      // defined; and the classes in which any member templates used as 
      // template template arguments are defined.
      if (ClassTemplateDecl *ClassTemplate 
            = dyn_cast<ClassTemplateDecl>(Arg.getAsDecl())) {
        DeclContext *Ctx = ClassTemplate->getDeclContext();
        if (CXXRecordDecl *EnclosingClass = dyn_cast<CXXRecordDecl>(Ctx))
          AssociatedClasses.insert(EnclosingClass);
        // Add the associated namespace for this class.
        while (Ctx->isRecord())
          Ctx = Ctx->getParent();
        CollectNamespace(AssociatedNamespaces, Ctx);
      }
      break;
      
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
  if (T->getAsBuiltinType())
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
  if (const EnumType *EnumT = T->getAsEnumType()) {
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
  if (const FunctionType *FunctionType = T->getAsFunctionType()) {
    // Return type
    addAssociatedClassesAndNamespaces(FunctionType->getResultType(), 
                                      Context,
                                      AssociatedNamespaces, AssociatedClasses);

    const FunctionProtoType *Proto = dyn_cast<FunctionProtoType>(FunctionType);
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

  const FunctionProtoType *Proto = Fn->getType()->getAsFunctionProtoType();
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
  Decl *D = LookupName(TUScope, II, LookupObjCProtocolName).getAsDecl();
  return cast_or_null<ObjCProtocolDecl>(D);
}

/// \brief Find the Objective-C category implementation with the given
/// name, if any.
ObjCCategoryImplDecl *Sema::LookupObjCCategoryImpl(IdentifierInfo *II) {
  Decl *D = LookupName(TUScope, II, LookupObjCCategoryImplName).getAsDecl();
  return cast_or_null<ObjCCategoryImplDecl>(D);
}

// Attempts to find a declaration in the given declaration context
// with exactly the given type.  Returns null if no such declaration
// was found.
Decl *Sema::LookupQualifiedNameWithType(DeclContext *DC,
                                        DeclarationName Name,
                                        QualType T) {
  LookupResult result =
    LookupQualifiedName(DC, Name, LookupOrdinaryName, true);

  CanQualType CQT = Context.getCanonicalType(T);

  for (LookupResult::iterator ir = result.begin(), ie = result.end();
       ir != ie; ++ir)
    if (FunctionDecl *CurFD = dyn_cast<FunctionDecl>(*ir))
      if (Context.getCanonicalType(CurFD->getType()) == CQT)
        return CurFD;

  return NULL;
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
  LookupResult Operators = LookupName(S, OpName, LookupOperatorName);
  
  assert(!Operators.isAmbiguous() && "Operator lookup cannot be ambiguous");

  if (!Operators)
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

void Sema::ArgumentDependentLookup(DeclarationName Name,
                                   Expr **Args, unsigned NumArgs,
                                   FunctionSet &Functions) {
  // Find all of the associated namespaces and classes based on the
  // arguments we have.
  AssociatedNamespaceSet AssociatedNamespaces;
  AssociatedClassSet AssociatedClasses;
  FindAssociatedClassesAndNamespaces(Args, NumArgs, 
                                     AssociatedNamespaces,
                                     AssociatedClasses);

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
      // Only count friend declarations which were declared in
      // associated classes.
      if (D->isInIdentifierNamespace(Decl::IDNS_Friend)) {
        DeclContext *LexDC = D->getLexicalDeclContext();
        if (!AssociatedClasses.count(cast<CXXRecordDecl>(LexDC)))
          continue;
      }
      
      CollectFunctionDecl(Functions, D);
    }
  }
}
